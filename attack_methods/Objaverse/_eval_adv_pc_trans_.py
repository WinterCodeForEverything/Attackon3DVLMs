import argparse
import json
import os
import random
random.seed(0)

import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
#from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm



from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.utils import disable_torch_init
from pointllm.data import ObjectPointCloudDataset
from pointllm.conversation import conv_templates, SeparatorStyle

from pointllm.eval.evaluator import start_evaluation

from whitebox.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator
# from pytorch3d.loss import chamfer_distance


device = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]


def init_model(model_name):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv



def load_dataset(data_path, anno_path, use_color, use_normal):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        # pointnum=pointnum,
        # conversation_types=conversation_types,
        use_color=use_color,
        use_normal=use_normal,
        # tokenizer=None # * load point cloud only
    )
    print("Done!")
    return dataset



def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def generate_outputs( model, tokenizer, input_ids, point_clouds, stopping_criteria, temperature=1.0, top_k=50, max_length=2048, top_p=0.95, num_beams=1 ):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample= True if temperature > 0.0 else False,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            num_beams=num_beams,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs


def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file, args):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        if point_clouds.shape[-1] > 6:
            point_clouds = point_clouds[..., :6].contiguous()
        labels = batch["labels"]
        obj_ids = batch["object_ids"]

        batchsize = point_clouds.shape[0]

        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, 
                                   temperature=args.temperature, top_k=args.top_k,  max_length=args.max_length,
                                   top_p=args.top_p, num_beams=args.num_beams
                                   )

        for obj_id, output, label in zip(obj_ids, outputs, labels):
            responses.append({
                "object_id": obj_id,
                "ground_truth": label,
                "model_output": output,
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def start_pc_evaluation(args, model, evaluated_output_file = None):
    
    pc_evaluator = PointCloudEvaluator(k_nn=32, device=device)

    

    ori_dataset = load_dataset(data_path= args.ori_data_path, anno_path=args.ori_anno_path, 
                                use_color= args.use_color, use_normal=args.use_normal )
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    adv_dataset = load_dataset(data_path= args.adv_data_path, anno_path=args.adv_anno_path, 
                                use_color= args.use_color, use_normal=args.use_normal )
    adv_dataloader = get_dataloader(adv_dataset, args.batch_size, args.shuffle, args.num_workers)

    # pc_encoder = model.get_pc_encoder()
    # pc_encoder.eval()
    
    
    results = []
    overall_results = {
            'knn_dist': [],
            'uniform_dist': [],
            'curv_std_dist': [],
            'chamfer_dist': [],
            'hausdorff_dist': []
    }
    
    if evaluated_output_file is not None:
        evaluated_output_file = os.path.join(args.output_dir, evaluated_output_file)
        evaluated_results = json.load(open(evaluated_output_file, 'r'))
        evaluated_results = evaluated_results['results']
    
    statistic_object_ids = []
    statistic_ori_xyz, statistic_ori_normal, statistic_adv_xyz = [], [], []
    for i, (ori_data, adv_data ) in enumerate(zip(ori_dataloader, adv_dataloader)):
        ori_pc = ori_data['point_clouds'].to(device)
        adv_pc = adv_data['point_clouds'].to(device)
        
        obj_ids = adv_data['object_ids']
        
        ori_xyz = ori_pc[..., :3].contiguous()
        #ori_rgb = ori_pc[:, :, 3:6]
        ori_normal = ori_pc[..., 6:9].contiguous()
        adv_xyz = adv_pc[..., :3].contiguous()
        #adv_rgb = adv_pc[:, :, 3:6]
        
        for j in range(adv_pc.shape[0]):
            obj_id = obj_ids[j]
            current_id = i* args.batch_size + j
            if evaluated_output_file is not None:
                #print(evaluated_results[current_id]['object_id'], obj_id)
                assert obj_id == evaluated_results[current_id]['object_id']
                if evaluated_results[current_id]['cls_result'] == 'T':
                        statistic_object_ids.append(obj_id)
                        statistic_ori_xyz.append(ori_xyz[j])
                        statistic_adv_xyz.append(adv_xyz[j])
                        statistic_ori_normal.append(ori_normal[j])
            else:
                statistic_object_ids.append(obj_id)
                statistic_ori_xyz.append(ori_xyz[j])
                statistic_adv_xyz.append(adv_xyz[j])
                statistic_ori_normal.append(ori_normal[j])
    
    # statistic_ori_xyz = torch.stack(statistic_ori_xyz, dim=0)
    # statistic_adv_xyz = torch.stack(statistic_adv_xyz, dim=0)
    # statistic_ori_normal = torch.stack(statistic_ori_normal, dim=0)
    
    for j in range(len(statistic_ori_xyz)):
        obj_id = statistic_object_ids[j]
        ori_xyz = statistic_ori_xyz[j].unsqueeze(0)
        adv_xyz = statistic_adv_xyz[j].unsqueeze(0)
        ori_normal = statistic_ori_normal[j].unsqueeze(0)
        pc_result = pc_evaluator.evaluate(ori_xyz, adv_xyz, ori_normal)
        
        pc_result =  {metric: float(value.cpu().numpy()) for metric, value in pc_result.items()}
        for metrics, values in pc_result.items():
            overall_results[metrics].append( values )

        pc_result['ori_id'] = obj_id
        # pc_result['ori_id'] = ori_data['indice'].cpu().numpy().tolist()
        # pc_result['adv_id'] = adv_data['indice'].cpu().numpy().tolist()
        results.append(pc_result)
        
    average_results = {}
    for metrics, values in overall_results.items():
        average_results[metrics] = float(np.mean((values)))
        
    print(f"Average knn_dist: {average_results['knn_dist']}")
    print(f"Average uniform_dist: {average_results['uniform_dist']}")
    print(f"Average curv_std_dist: {average_results['curv_std_dist']}")
    print(f"Average chamfer_dist: {average_results['chamfer_dist']}")
    print(f"Average hausdorff_dist: {average_results['hausdorff_dist']}")
    
    # * save the results
    save_results = {
        'overall_results': average_results,
        'results': results
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'pc_evaluation_results_recall.json'), 'w') as fp:
        json.dump(save_results, fp, indent=4)



def main(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # * output file 
    args.output_file = f"ModelNet_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    model, tokenizer, conv = init_model(args.model_name)
    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):

        dataset = load_dataset(data_path= args.adv_data_path, anno_path=args.adv_anno_path, 
                                use_color= args.use_color, use_normal=args.use_normal )
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)

        #annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file, args )

        # * release model and tokenizer, and release cuda memory
        # del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    if args.start_caption_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type= args.caption_eval_type, model_type=args.gpt_type, parallel=True, num_workers=20)
        
    if args.start_pc_eval:
        start_pc_evaluation( args, model, evaluated_output_file )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--do_sample",  action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    # * dataset type
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--ori_data_path", type=str, default='./ori_data', required=False)
    parser.add_argument("--adv_data_path", type=str, default='./adv_data', required=False)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_ori_200_GT.json", required=False)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_adv_200_GT.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=False)
    parser.add_argument("--use_normal",  action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="", required=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--caption_eval_type", type=str, default="modelnet-close-set-classification", choices=["modelnet-close-set-classification",
                            "open-free-form-classification", "object-captioning", "unknown-target-classification"], help="Type of the evaluation.")
    parser.add_argument("--start_caption_eval", action="store_true", default=False)
    parser.add_argument("--start_pc_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-4.1-mini", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    
    args = parser.parse_args()


    main(args)  