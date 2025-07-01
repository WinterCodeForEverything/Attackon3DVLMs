#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.data import ModelNet
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import wandb
from attack_methods.ShapeAttack.HiT_ADV import HiT_ADV
import pickle

from attack_methods.ModelNet40.evaluate import Evaluation
from attack_methods.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator

import json


device = "cuda" if torch.cuda.is_available() else "cpu"
cat_file = '/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt'
label_names = [line.rstrip() for line in open(cat_file)]

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).to(device)
    model.get_model().fix_pointnet = False
    model.get_model().fix_llm = False
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv



def load_dataset(pc_file, config_path, split, subset_nums, use_color):
    print(f"Loading {split} split of ModelNet datasets.")
    dataset = ModelNet(pc_file=pc_file, config_path=config_path, split=split, subset_nums=subset_nums, use_color=use_color)
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class FeatureSimilarityLoss(torch.nn.Module):
    def __init__(self, model):
        super(FeatureSimilarityLoss, self).__init__()
        self.model = model
        
    def forward(self, adv_pc, tgt_feature):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        adv_pc = adv_pc.to(device).to(torch.bfloat16)
        adv_feature = self.model(adv_pc)
        #tgt_feature = self.model(tgt_pc)
        
        # Normalize the features
        adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
        #print(f"adv_feature: {adv_feature.shape}")
        
        # Compute the embedding similarity
        embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2), dim=1)
        
        loss = 1 - embedding_sim 
        
        return loss

class ModelGeneration(torch.nn.Module):
    def __init__(self, model, tokenizer, conv,  args):
        super(ModelGeneration, self).__init__()
        self.model = model
        self.conv = conv
        self.tokenizer = tokenizer
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_length = args.max_length
        # stopping_criteria
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        qs = PROMPT_LISTS[args.prompt_index]

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
        self.input_ids_ = input_ids_
        
        self.stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)
        

        
    def generate( self, point_clouds ):
        
        B = point_clouds.shape[0]
        input_ids = self.input_ids_.repeat(B, 1) # * tensor of B, L
        
        self.model.eval() 
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_clouds.to(torch.bfloat16),
                do_sample= True if self.temperature > 0.0 else False,
                temperature= self.temperature if self.temperature > 0.0 else None,
                top_k=self.top_k,
                max_length=self.max_length,
                top_p=self.top_p,
                num_beams=self.num_beams,
                stopping_criteria=[self.stopping_criteria]) # * B, L'

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]

        return outputs
    



def start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, output_eval_dir, evaluated_results = None):
    
    pc_evaluator = PointCloudEvaluator(k_nn=32, device=device)

    

    # ori_dataset = load_dataset(data_path= args.ori_data_path, anno_path=args.ori_anno_path, 
    #                             use_color= args.use_color, use_normal=args.use_normal )
    # ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    # adv_dataset = load_dataset(data_path= args.adv_data_path, anno_path=args.adv_anno_path, 
    #                             use_color= args.use_color, use_normal=args.use_normal )
    # adv_dataloader = get_dataloader(adv_dataset, args.batch_size, args.shuffle, args.num_workers)

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
    
    statistic_object_ids = []
    statistic_ori_xyz, statistic_ori_normal, statistic_adv_xyz = [], [], []
    for i, (obj_id, ori_pc, adv_pc ) in enumerate(zip(obj_id_list, ori_pc_list, adv_pc_list)):
        # ori_pc = ori_data['point_clouds'].to(device)
        # adv_pc = adv_data['point_clouds'].to(device)
        
        # obj_ids = adv_data['object_ids']
        
        ori_xyz = ori_pc[..., :3]
        ori_normal = ori_pc[..., 6:9]
        adv_xyz = adv_pc[..., :3]
        
        if evaluated_results is not None:
            if evaluated_results[i]['accuracy'] == 1:
                    statistic_object_ids.append(obj_id)
                    statistic_ori_xyz.append(torch.from_numpy(ori_xyz).to(device, dtype=torch.float32))
                    statistic_adv_xyz.append(torch.from_numpy(adv_xyz).to(device, dtype=torch.float32))
                    statistic_ori_normal.append(torch.from_numpy(ori_normal).to(device, dtype=torch.float32))
        else:
            statistic_object_ids.append(obj_id)
            statistic_ori_xyz.append(torch.from_numpy(ori_xyz).to(device, dtype=torch.float32))
            statistic_adv_xyz.append(torch.from_numpy(adv_xyz).to(device))
            statistic_ori_normal.append(torch.from_numpy(ori_normal).to(device, dtype=torch.float32))
    print( "len(statistic_object_ids):", len(statistic_object_ids))
    for j in range(len(statistic_ori_xyz)):
        obj_id = statistic_object_ids[j]
        ori_xyz = statistic_ori_xyz[j].unsqueeze(0)
        adv_xyz = statistic_adv_xyz[j].unsqueeze(0)
        ori_normal = statistic_ori_normal[j].unsqueeze(0)
        pc_result = pc_evaluator.evaluate(ori_xyz, adv_xyz, ori_normal)
        
        pc_result =  {metric: float(value.cpu().numpy()) for metric, value in pc_result.items()}
        for metrics, values in pc_result.items():
            overall_results[metrics].append( values )

        pc_result['obj_id'] = obj_id
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
    os.makedirs(output_eval_dir, exist_ok=True)
    with open(os.path.join(output_eval_dir, 'pc_evaluation_results_recall_in_training.json'), 'w') as fp:
        json.dump(save_results, fp, indent=4)


def main(args):
    
    output_pc_file = args.output_pc_file
    
    saved_results = {}
    success_rate = 0.0
    eval_results = []
    ori_pc_list, obj_id_list = [], []
    adv_pc_list, adv_label_list = [], []
    output_file = f"ModelNet_classification_prompt_{args.prompt_index}_evaluated_{args.gpt_type}_in_training.json"
    output_eval_file = os.path.join(args.output_eval_path, output_file )
    
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
    elif os.path.exists(output_eval_file) and os.path.exists(output_pc_file):
        eval_data = json.load(open(output_eval_file, "r"))
        eval_results = eval_data.get('eval_results', [])
        success_rate = eval_data.get('ASR', 0.0)
        adv_pc_list, adv_label_list = pickle.load(open(output_pc_file, "rb"))
        total_ori_pc_list, _ = pickle.load(open(args.ori_data_path, "rb")) 
        ori_pc_list = total_ori_pc_list[:len(adv_pc_list)] # * ensure the length of ori_pc_list is the same as adv_pc_list
        obj_id_list = list(range(len(adv_pc_list))) # * use index as object id
        assert len(adv_pc_list) == len(eval_results), "The length of adversarial point clouds and evaluation results should be the same."
        print(f"Loaded {len(eval_results)} evaluation results")
        
        
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)

    ori_dataset = load_dataset(pc_file= args.ori_data_path, config_path=None, split=args.ori_split, subset_nums=args.subset_nums, use_color=args.use_color) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataset = load_dataset(pc_file= args.tgt_data_path, config_path=None, split=args.tgt_split, subset_nums=args.subset_nums, use_color=args.use_color) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)

    #pc_encoder = init_pc_encoder(args)
    
    model, tokenizer, conv = init_model(args)
    
    candidate_lists_names = label_names
    genetrate = ModelGeneration( model, tokenizer, conv, args )
    eval_func = Evaluation( genetrate, args.gpt_type, candidate_lists_names )
    
    pc_encoder = model.get_pc_encoder()
    
    adv_func = FeatureSimilarityLoss(pc_encoder)
    HiT_attacker = HiT_ADV( adv_func, eval_func, attack_lr=args.attack_lr,
                               central_num=args.central_num, total_central_num=args.total_central_num,
                               init_weight=args.init_weight, max_weight=args.max_weight, binary_step=args.binary_step,
                               num_iter=args.num_iter, clip_func=None,
                               cd_weight=args.cd_weight, ker_weight=args.ker_weight,
                               hide_weight=args.hide_weight, curv_loss_knn=args.curv_loss_knn,
                               max_sigm=args.max_sigm, min_sigm=args.min_sigm,
                               budget=args.budget, use_wandb = args.wandb )

    eval_attack_num = len(eval_results)
    success_nums = 0
    
    try:
        for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
            if i * args.batch_size < eval_attack_num:
                print(f"Skip {i}th batch, since it has been evaluated.")
                continue
            obj_id = ori_data_dict['indice']
            ori_pc = ori_data_dict['point_clouds'].to(device)
            tgt_pc = tgt_data_dict['point_clouds'].to(device)
            tgt_labels = tgt_data_dict['labels']
            
            with torch.no_grad():
                tgt_feature = pc_encoder(tgt_pc[..., :6].to(model.dtype))
                if torch.isnan(tgt_feature).any():
                    print(f"NaN detected in tgt_feature. Skip this sample.")
                    continue
                tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
                
            adv_pc, success_num, eval_result = HiT_attacker.attack(ori_pc, tgt_labels, tgt_feature)
            success_nums += success_num
            for j in range(obj_id.shape[0]):
                if i * args.batch_size + j < eval_attack_num:
                    continue
                eval_result[j].update({'object_id': obj_id[j].item()})
            #adv_pc = torch.from_numpy(adv_pc).to(device)
            eval_results.extend(eval_result)
            #adv_pc = torch.from_numpy(adv_pc).to(device)
            
            # save adversarial point cloud
            # ori_pc_ids = ori_data_dict['indice'].cpu().numpy()
            # ori_pc = ori_pc.cpu().numpy()
            # tgt_pc = tgt_pc.cpu().numpy()
            # saved_ori_pc = np.concatenate((ori_pc[..., :3], ori_pc[..., 6:]), axis=-1)
            # saved_adv_pc = np.concatenate((adv_pc[..., :3], adv_pc[..., 6:]), axis=-1) 
            # saved_tgt_pc = np.concatenate((tgt_pc[..., :3], tgt_pc[..., 6:]), axis=-1)
            saved_lable = tgt_data_dict['labels'].cpu().numpy()

            ori_pc = ori_pc.float().cpu().numpy()
            for b, (id, op, ap, adv_label) in enumerate( zip(obj_id, ori_pc, adv_pc, saved_lable) ):
                obj_id_list.append(id.item())
                ori_pc_list.append(op)
                adv_pc_list.append(ap)
                adv_label_list.append(adv_label)
                
    except Exception as e:
        print(f"Error occurred during the attack: {e}. Skipping the rest of the batches.")
         
    success_rate += success_nums / len(ori_dataset)       
    print("attack success rate: ", success_rate)
    saved_results['ASR'] = success_rate
    saved_results['eval_results'] = eval_results
    
    # * Save evaluation results
    with open(output_eval_file, "w") as f:
        json.dump(saved_results, f, indent=4)
        print(f"Saved evaluation results to {output_eval_file}")
            
            
    # * Save the adversarial point clouds and labels
    with open(output_pc_file, "wb") as f:
        pickle.dump((adv_pc_list, adv_label_list), f)
        print(f"Saved all adversarial point clouds to {output_pc_file}")
        
    if args.start_pc_eval:
        print("Starting point cloud evaluation...")
        start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, args.output_eval_path, eval_results)
        print("Point cloud evaluation finished.")
    
 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument('--cd_weight', type=float, default=0.0001, help='cd_weight')
    parser.add_argument('--ker_weight', type=float, default=1., help='ker_weight')
    parser.add_argument('--hide_weight', type=float, default=1., help='hide_weight')
    parser.add_argument('--max_sigm', type=float, default=1.2, help='max_sigm')
    parser.add_argument('--min_sigm', type=float, default=0.1, help='min_sigm')
    parser.add_argument('--init_weight', type=float, default=1.0, help='init_weight')
    parser.add_argument('--max_weight', type=float, default=8.0, help='max_weight')
    parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1], help="0: What is this? 1: This is an object of ")

    
    parser.add_argument('--central_num', type=int, default=26)
    parser.add_argument('--total_central_num', type=int, default=2048)
    parser.add_argument('--curv_loss_knn', type=int, default=16, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--budget', type=float, default=0.55,
                        help='0.5 for l2 attack, 0.05 for linf attack')
    
    #CWPerturb_args
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    
    # parser.add_argument("--epsilon", type=float, default=1.0)
    # parser.add_argument("--gamma1", type=float, default=1.0)
    # parser.add_argument("--steps", type=int, default=300)

    #evaluate
    parser.add_argument("--start_pc_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-4.1-mini", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    parser.add_argument("--output_eval_path", type=str, default='./eval_results', help="Path to save the evaluation results.")
    

    # data 
    #parser.add_argument("--data_path", type=str, default="data/modelnet40", required=False)
    parser.add_argument("--ori_data_path", type=str, default='./data')
    parser.add_argument("--tgt_data_path", type=str, default='./data')
    parser.add_argument("--subset_nums", type=int, default=-1)
    #parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--ori_split", type=str, default='test')
    parser.add_argument("--tgt_split", type=str, default='target')
    parser.add_argument("--use_color",  action="store_true", default=False)
    #parser.add_argument("--use_normal",  action="store_true", default=False)
    parser.add_argument("--output_pc_file", type=str, default="data/modelnet40_adv.dat ")

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')

    args = parser.parse_args()

    main(args)
    
    