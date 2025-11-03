#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import datetime
import torch


from tqdm import tqdm
import os
import wandb
# from attack_methods.ShapeAttack.HiT_ADV import HiT_ADV
import pickle
import json

from victim_models.gpt4point.models import load_model_and_preprocess
from victim_models.pointllm.initialization import init_PointLLM


# from attack_methods.evaluation.loss_module import FeatureSimilarityLoss, GPT4PointCaptionLoss, PointLLMCaptionLoss

from attack_methods.evaluation.model_generator import PointLLMGenerator, GPT4PointGenerator
from attack_methods.evaluation.gpt_evaluator import GPTEvaluation, GPTBatchEvaluation, GPTObjectCaptioningEvaluator

# from attack_methods.ModelNet40.evaluate import Evaluation
from attack_methods.evaluation.pc_evaluator.pc_evaluator import start_pc_evaluation
from attack_methods.helper import load_attack_results, setup_seed, feature_normalize, get_defensor
from dataset.data.dataloader import load_dataset, get_dataloader

def main(args):
    
    setup_seed(42)
    valid_state = 1
    #lr = args.lr
    #budget = args.budget
    model_name = args.model_name
    model_type = args.model_type
    is_targeted = args.is_targeted 
    dataset_type = args.dataset_type
    task_type = args.task_type
    assert dataset_type in ['ModelNet', 'Objaverse'], f"Unsupported dataset type: {dataset_type}"
    output_pc_path = args.output_pc_path
    device = args.device
    dtype = getattr(torch, args.dtype)

    success_rate = 0.0
    saved_results = {} 
    eval_results = []
    # attack_id_list, attack_pc_list = [], []
    # adv_pc_file_name_list = []
    # adv_pc_list, adv_label_list = [], []

    defense = f"{args.defend_method}_defense" if args.has_defense else 'no_defense'
    output_file = f"{dataset_type}_{task_type}_{args.gpt_type}_in_evaluation_{defense}.json"
    output_eval_file = os.path.join(args.output_eval_path, output_file )
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
    elif os.path.exists(output_eval_file) and os.path.exists(output_pc_path):
        pass
        # ( success_rate, 
        #  average_gpt_score, 
        #  adv_pc_list, 
        #  adv_label_list, 
        #  attack_pc_list, 
        #  eval_results ) = load_attack_results( args.ori_data_path, args.ori_anno_path, args.tgt_anno_path, args.adv_anno_path,
        #                                       args.output_eval_path, output_pc_path, dataset_type, is_targeted ) 
        
        # assert len(adv_pc_list) == len(eval_results), "The length of adversarial point clouds and evaluation results should be the same."
        # print(f"Loaded {len(eval_results)} evaluation results")
    
    # if args.start_pc_eval and not args.start_gpt_eval:
    #     print("Starting point cloud evaluation...")
    #     if task_type == 'captioning':
    #         start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, device=device)
    #     else:
    #         start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, eval_results, device=device)
    #     print("Point cloud evaluation finished.")

    #     return valid_state
        
    # if attack_method == 'vision_attack':
    if 'gpt4point' in model_type:
        model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
        # pc_encoder = model.get_pc_encoder_qformer()
        # for param in pc_encoder.parameters():
        #     # param.data = param.data.to(dtype)
        #     param.requires_grad = True
        # adv_func = FeatureSimilarityLoss( pc_encoder, dtype= dtype, is_targeted=is_targeted )
        generator = GPT4PointGenerator(model=model, llm_args=args)
    elif 'pointllm' in model_type:
        model, tokenizer, conv = init_PointLLM( model_name, dtype, device )
        # pc_encoder = model.get_pc_encoder()
        # adv_func = FeatureSimilarityLoss(pc_encoder, dtype= dtype, is_targeted=is_targeted )
        generator = PointLLMGenerator( model, tokenizer, conv, args )
    # elif attack_method == 'caption_attack': 
    #     if 'gpt4point' in model_type:
    #         model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
    #         for param in model.parameters():
    #             # param.data = param.data.to(dtype)
    #             param.requires_grad = True
    #         adv_func = GPT4PointCaptionLoss(model, text_processor=txt_processors['eval'], is_targeted=is_targeted)
    #         generator = GPT4PointGenerator(model=model, llm_args=args)
    #     elif 'pointllm' in model_type:
    #         model, tokenizer, conv = init_PointLLM( model_name, dtype, device )
    #         adv_func = PointLLMCaptionLoss(model, tokenizer, conv, is_targeted=is_targeted ) 
    #         # generator = PointLLMGenerator(model=model, tokenizer=tokenizer, conv=conv, llm_args=args)
    #         generator = PointLLMGenerator( model, tokenizer, conv, args )
    
    defensor = get_defensor( args.defend_method, npoint=args.pointnum ) if args.has_defense else None
        
    if task_type == 'classification':
        # evaluator = GPTEvaluation( generator, args.gpt_type, is_targeted=is_targeted )
        if 'gpt4point' in model_type:
            evaluator = GPTBatchEvaluation( generator, args.gpt_type, is_targeted=is_targeted )
        elif 'pointllm' in model_type:
             evaluator = GPTEvaluation( generator, args.gpt_type, is_targeted=is_targeted, has_defense=args.has_defense, defensor=defensor )
    elif task_type == 'captioning':
        evaluator = GPTObjectCaptioningEvaluator( generator, args.gpt_type, is_targeted=is_targeted,
                                                  success_score_threshold=args.success_score_threshold )
        
    # ori_dataset = load_dataset( dataset_type = dataset_type, data_path= args.ori_data_path, category_file= args.modelnet_cat_file,
    #                            anno_path= args.ori_anno_path, normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
    #                                 redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
    #                                 pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
    #                                 text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    # ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    
  
    adv_dataset = load_dataset( dataset_type = dataset_type, data_path= args.adv_data_path, category_file= args.modelnet_cat_file,
                               anno_path= args.adv_anno_path, normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal, 
                                    do_preprocess=args.do_preprocess, redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                                    pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
                                    text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    adv_dataloader = get_dataloader(adv_dataset, args.batch_size, args.shuffle, args.num_workers)
        
    # success_nums = success_rate * len(attack_pc_list)
    # total_gpt_score = average_gpt_score * len(attack_pc_list)
    success_nums = 0
    total_nums = 0

    # pretrained_eval_results_dict = {}
    # if negitive_samples_filter_eval_path is not None:
    #     pretrained_eval_results = json.load(open(negitive_samples_filter_eval_path, "r")).get('results', [])
    #     for result in pretrained_eval_results:
    #         pretrained_eval_results_dict[str(result['object_id'])] = result

    # try:
    if 1:
        with torch.no_grad():
            for adv_data_dict in tqdm(adv_dataloader):
                # if 'pointllm' in model_type and task_type == 'classification' and i % 4 != 0:
                #     continue

                # adv_id = adv_data_dict['object_id']
                adv_pc = adv_data_dict['point_cloud'].to(device=device, dtype=dtype)
                adv_labels = adv_data_dict['label']
                
                
                success_attack, eval_result = evaluator( adv_pc, adv_labels )  # warm up

                # for j, er in enumerate(eval_result):
                #     er['object_id'] = adv_id[j].item()
                
                success_nums += success_attack.sum().item()
                total_nums += len(success_attack)
                eval_results.extend(eval_result)       
    # except Exception as e:
    #     valid_state = 0
    #     print(f"Error occurred during the attack: {e}. Skipping the rest of the batches.")

    saved_results['attack_pc_number'] = total_nums
    success_rate = success_nums / total_nums if total_nums > 0 else 0.0
    saved_results['ASR'] = success_rate
    saved_results['eval_results'] = eval_results
    
    # * Save evaluation results
    with open(output_eval_file, "w") as f:
        json.dump(saved_results, f, indent=4)
        print(f"Saved evaluation results to {output_eval_file}")
            
    # if args.start_pc_eval:
    #     print("Starting point cloud evaluation...")
    #     if task_type == 'captioning':
    #         start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, device=device)
    #     else:
    #         start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, eval_results, device=device)
    #     print("Point cloud evaluation finished.")

    return valid_state
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # victim model config
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--model_type", type=str, default="gpt4point_opt2.7b")
    parser.add_argument("--attack_method", type=str, default="vision_attack")
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help='Data type for the model')
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # attack/eval config (kept only params used in this script)
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "captioning"], help="Task type for the attack.")
    parser.add_argument("--is_targeted",  action="store_true", default=False)
    parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1], help="0: What is this? 1: This is an object of ")
    parser.add_argument('--success_score_threshold', type=int, default=50, help='Success score threshold for the attack on captioning task.')

    # defense config
    parser.add_argument("--has_defense",  action="store_true", default=False)
    parser.add_argument("--defend_method", type=str, default="sor", choices=["sor", "srs", "dupnet"], help="Defense method to use.")
    parser.add_argument("--pointnum", type=int, default=8192)
    
    # evaluate
    parser.add_argument("--start_pc_eval",  action="store_true", default=False)
    parser.add_argument("--negitive_samples_filter_eval_path", type=str, default=None, help="Path to save the evaluation results of negative samples filter.")
    parser.add_argument("--start_gpt_eval",  action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-5-nano", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    parser.add_argument("--output_eval_path", type=str, default='./eval_results', help="Path to save the evaluation results.")
    
    # data 
    parser.add_argument("--dataset_type", type=str, default='ModelNet', choices=['ModelNet', 'Objaverse'])
    parser.add_argument("--adv_data_path", type=str, default='./data')
    parser.add_argument("--adv_anno_path", type=str, default=None)
    parser.add_argument("--do_preprocess", type=bool,  default=False)
    parser.add_argument("--normalized_pc", type=bool,  default=False)
    parser.add_argument("--has_color",  type=bool,  default=True)
    parser.add_argument("--has_normal",  type=bool,   default=True)
    parser.add_argument("--use_color",  type=bool,  default=True)
    parser.add_argument("--use_normal",  type=bool,   default=True)
    parser.add_argument("--redirect_normal",  type=bool,  default=False)
    parser.add_argument("--output_pc_path", type=str, default="data/adv")
    parser.add_argument("--modelnet_cat_file", type=str, default="./data/modelnet40_category.txt", help="Path to the ModelNet category file.")
    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    args = parser.parse_args()

    valid_state = 0
    while valid_state == 0:
        valid_state = main(args)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # victim model config
#     parser.add_argument("--model_name", type=str, \
#         default="RunsenXu/PointLLM_7B_v1.2") 
#     parser.add_argument("--model_type", type=str, \
#         default="gpt4point_opt2.7b") 
#     parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
#     parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help='Data type for the model')
    
#     parser.add_argument("--temperature", type=float, default=1.0)
#     parser.add_argument("--max_length", type=int, default=2048)
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--top_k", type=int, default=50)
#     parser.add_argument("--top_p", type=float, default=0.95)
    
#     # attack config
#     parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "captioning"], help="Task type for the attack.")
#     parser.add_argument("--attack_method",  type=str, default="vision_attack", choices=["vision_attack", "caption_attack"])
#     parser.add_argument("--is_targeted",  action="store_true", default=False)
#     parser.add_argument("--attack_lr", type=float, default=0.01)
#     parser.add_argument('--cd_weight', type=float, default=0.0001, help='cd_weight')
#     parser.add_argument('--ker_weight', type=float, default=1., help='ker_weight')
#     parser.add_argument('--hide_weight', type=float, default=1., help='hide_weight')
#     parser.add_argument('--max_sigm', type=float, default=1.2, help='max_sigm')
#     parser.add_argument('--min_sigm', type=float, default=0.1, help='min_sigm')
#     parser.add_argument('--init_weight', type=float, default=1.0, help='init_weight')
#     parser.add_argument('--max_weight', type=float, default=8.0, help='max_weight')
#     parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1], help="0: What is this? 1: This is an object of ")
#     parser.add_argument('--success_score_threshold', type=int, default=50, help='Success score threshold for the attack on captioning task.')

#     parser.add_argument('--central_num', type=int, default=26)
#     parser.add_argument('--total_central_num', type=int, default=2048)
#     parser.add_argument('--curv_loss_knn', type=int, default=16, metavar='N', help='Num of nearest neighbors to use')
#     parser.add_argument('--budget', type=float, default=0.55,
#                         help='0.5 for l2 attack, 0.05 for linf attack')

#     parser.add_argument('--binary_step', type=int, default=10, metavar='N',
#                         help='Binary search step')
#     parser.add_argument('--num_iter', type=int, default=100, metavar='N',
#                         help='Number of iterations in each search step')
    
#     # defense config
#     parser.add_argument("--has_defense",  action="store_true", default=False)
#     parser.add_argument("--defend_method", type=str, default="sor", choices=["sor", "srs", "dupnet"], help="Defense method to use.")
    
#     # parser.add_argument("--epsilon", type=float, default=1.0)
#     # parser.add_argument("--gamma1", type=float, default=1.0)
#     # parser.add_argument("--steps", type=int, default=300)
    
#     #evaluate
#     parser.add_argument("--start_pc_eval",  action="store_true", default=False)
#     parser.add_argument("--negitive_samples_filter_eval_path", type=str, default=None, help="Path to save the evaluation results of negative samples filter.")
#     parser.add_argument("--start_gpt_eval",  action="store_true", default=False)
#     parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-5-nano", "gpt-4o-mini"], help="Type of the model used to evaluate.")
#     parser.add_argument("--output_eval_path", type=str, default='./eval_results', help="Path to save the evaluation results.")
    

#     # data 
#     parser.add_argument("--dataset_type", type=str, default='ModelNet', choices=['ModelNet', 'Objaverse'])
#     parser.add_argument("--adv_data_path", type=str, default='./data')
#     parser.add_argument("--pointnum", type=int, default=8192)
#     parser.add_argument("--adv_anno_path", type=str, default=None)
#     parser.add_argument("--normalized_pc", type=bool,  default=False)
#     parser.add_argument("--has_color",  type=bool,  default=True)
#     parser.add_argument("--has_normal",  type=bool,   default=True)
#     parser.add_argument("--use_color",  type=bool,  default=True)
#     parser.add_argument("--use_normal",  type=bool,   default=True)
#     parser.add_argument("--redirect_normal",  type=bool,  default=False)
#     parser.add_argument("--output_pc_path", type=str, default="data/adv")
#     parser.add_argument("--modelnet_cat_file", type=str, default="./data/modelnet40_category.txt", help="Path to the ModelNet category file.")
#     # * data loader, batch_size, shuffle, num_workers
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--shuffle", type=bool, default=False)
#     parser.add_argument("--num_workers", type=int, default=10)
    
    
#     # logging
#     parser.add_argument("--wandb", action="store_true", default=False)
#     parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
#     parser.add_argument("--wandb_run_name", type=str, default='temp_run')
#     # parser.add_argument("--wandb_run_id", type=str, default=None, help="wandb run id for resuming the run")

#     args = parser.parse_args()

#     valid_state = 0
#     while valid_state == 0:
#         valid_state = main(args)
    
    