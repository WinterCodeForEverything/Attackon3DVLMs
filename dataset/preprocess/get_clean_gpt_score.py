import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset.data.modelnet import ModelNet
from dataset.data.objaverse import ObjaverseDataset

from victim_models.gpt4point.models import load_model_and_preprocess


from victim_models.pointllm.conversation import conv_templates, SeparatorStyle
from victim_models.pointllm.utils import disable_torch_init
from victim_models.pointllm.model import *

from attack_methods.evaluation.model_generator import PointLLMGenerator, GPT4PointGenerator
from attack_methods.evaluation.gpt_evaluator import GPTBatchEvaluation, GPTObjectCaptioningEvaluator


def init_PointLLM(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    dtype = getattr(torch, args.dtype)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=dtype).to(args.device)
    model.get_model().fix_pointnet = False
    model.get_model().fix_llm = False
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()
    
    #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = "What is this?"

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
    
    #qs = qs + "This is an object of "
    conv.append_message(conv.roles[0], qs)

    return model, tokenizer, conv



def load_dataset(dataset_type, data_path, anno_path=None, normalized_pc=True,
                 has_color=True, has_normal=True, redirect_normals=False,
                 use_color=True, use_normal=True, pts_processor=None, text_processor=None):
    # print(f"Loading {split} split of ModelNet datasets.")
    if dataset_type == 'ModelNet':
        dataset = ModelNet(data_path=data_path, normalized_pc=normalized_pc,
                           has_color=has_color, has_normal=has_normal,
                           redirect_normals=redirect_normals, use_color=use_color, use_normal=use_normal,
                           pts_processor=pts_processor, text_processor=text_processor )
    elif dataset_type == 'Objaverse':
        dataset = ObjaverseDataset(data_path=data_path, anno_path=anno_path, 
                                   normalized_pc=normalized_pc, has_color=has_color, has_normal=has_normal,
                                   redirect_normals=redirect_normals, use_color=use_color, use_normal=use_normal,
                                   pts_processor=pts_processor, text_processor=text_processor)
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def main(args):
    
    valid_state = 1
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    dataset_type = args.dataset_type
    is_targeted = args.is_targeted
    model_type = args.model_type
    
    average_gpt_score = 0.0
    total_evaluate_results = []
    if os.path.exists(args.output_eval_file):
        saved_results = json.load(open(args.output_eval_file, "r"))
        average_gpt_score = saved_results.get('average_gpt_score', 0.0)
        total_evaluate_results = saved_results.get('results', [])
    
    if 'pointllm' in model_type:
        model, tokenizer, conv = init_PointLLM(args)
        generator = PointLLMGenerator(model=model, tokenizer=tokenizer, conv=conv, llm_args=args)
    
    elif 'gpt4point' in model_type:
        model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=args.device)
        generator = GPT4PointGenerator(model=model, llm_args=args)

    eval_func = GPTObjectCaptioningEvaluator(generator, args.gpt_type, 0)

    # Load the dataset
    ori_dataset = load_dataset( dataset_type = dataset_type, data_path= args.ori_data_path, anno_path= args.ori_anno_path,
                                normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                                    pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
                                    text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    
    tgt_dataset = load_dataset( dataset_type = dataset_type, data_path= args.tgt_data_path, anno_path= args.tgt_anno_path,
                                normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                                    pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
                                    text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)

    
    # Run evaluation
    total_gpt_score = average_gpt_score * len(total_evaluate_results)
    try:
    # if 1:
        for i, (ori_data_dict, tgt_data_dict) in tqdm(enumerate(zip(ori_dataloader, tgt_dataloader)), total=len(ori_dataloader)):
            if i * args.batch_size < len(total_evaluate_results):
                print(f"Skipping batch {i}, already evaluated {i * args.batch_size} samples.")
                continue

            ori_id = ori_data_dict['object_id']
            tgt_id = tgt_data_dict['object_id']   
            ori_pc = ori_data_dict['point_cloud'].to(device=device, dtype=dtype)

            if is_targeted:
                tgt_labels = tgt_data_dict['label']
            else:
                tgt_labels = ori_data_dict['label']
                
            # Evaluate the outputs using the evaluator
            _, evaluation_results  = eval_func(ori_pc, tgt_labels)

            for j, result in enumerate(evaluation_results):
                if is_targeted:
                    result['object_id'] = f"{ori_id[j]}_{tgt_id[j]}"  
                else:
                    result['object_id'] = f"{ori_id[j]}"
                  
                total_gpt_score += result.get('gpt_score', 0.0)
                total_evaluate_results.append(result)
    
    except Exception as e:
        valid_state = 0
        print(f"Error during evaluation: {e}")

    average_gpt_score = total_gpt_score / len(total_evaluate_results) if total_evaluate_results else 0.0
    print(f"Average GPT Score: {average_gpt_score:.4f}")
    saved_results = {
        'average_gpt_score': average_gpt_score,
        'results': total_evaluate_results
    }
    
    # * Save evaluation results
    with open(args.output_eval_file, "w") as f:
        json.dump(saved_results, f, indent=4)
        print(f"Saved evaluation results to {args.output_eval_file}")
        
    return valid_state

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Path or name of the pretrained model')
    parser.add_argument('--model_type', type=str, default='vicuna', help='Type of the model (for GPT4Point)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for the model (e.g., float16, float32)')
    
    parser.add_argument("--dataset_type", type=str, default='Objaverse', choices=['Objaverse'])
    parser.add_argument('--ori_data_path', type=str, default='/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_ori_npy_captioning')
    parser.add_argument('--tgt_data_path', type=str, default='/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_tgt_npy_captioning')
    parser.add_argument('--ori_anno_path', type=str, default='/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_ori_npy_captioning_brief_description_200.json')
    parser.add_argument('--tgt_anno_path', type=str, default='/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_tgt_npy_captioning_brief_description_200.json')
    parser.add_argument('--output_eval_file', type=str, default='/mnt/ssd/liuchao/Attackon3DVLMs/pointllm_objaverse_ori_8192pts_eval.json')

    parser.add_argument('--normalized_pc', action='store_true')
    parser.add_argument('--has_color', action='store_true')
    parser.add_argument('--has_normal', action='store_true')
    parser.add_argument('--redirect_normal', action='store_true')
    parser.add_argument('--use_color', action='store_true')
    parser.add_argument('--use_normal', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--is_targeted', action='store_true', help='Whether the attack is targeted or not')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-4.1-mini", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    args = parser.parse_args()
    
    valid_state = 0
    while valid_state == 0:
        valid_state = main(args)
