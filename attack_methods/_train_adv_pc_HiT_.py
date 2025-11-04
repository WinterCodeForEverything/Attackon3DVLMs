#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import datetime
import torch


# from tqdm import tqdm
import os
import wandb
from attack_methods.ShapeAttack.HiT_ADV import HiT_ADV
import pickle
import json

from victim_models.gpt4point.models import load_model_and_preprocess
from victim_models.pointllm.initialization import init_PointLLM


from attack_methods.evaluation.loss_module import FeatureSimilarityLoss, GPT4PointCaptionLoss, PointLLMCaptionLoss

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
    attack_method = args.attack_method
    is_targeted = args.is_targeted 
    dataset_type = args.dataset_type
    task_type = args.task_type
    assert dataset_type in ['ModelNet', 'Objaverse'], f"Unsupported dataset type: {dataset_type}"
    output_pc_path = args.output_pc_path
    device = args.device
    dtype = getattr(torch, args.dtype)
    negitive_samples_filter_eval_path = args.negitive_samples_filter_eval_path
    
    success_rate = 0.0
    average_gpt_score = 0.0
    saved_results = {} 
    eval_results = []
    attack_id_list, attack_pc_list = [], []
    adv_pc_file_name_list = []
    adv_pc_list, adv_label_list = [], []
    
    output_file = f"{dataset_type}_{task_type}_{args.gpt_type}_in_training.json"
    output_eval_file = os.path.join(args.output_eval_path, output_file )
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
    elif os.path.exists(output_eval_file) and os.path.exists(output_pc_path):
        ( success_rate, 
         average_gpt_score, 
         adv_pc_list, 
         adv_label_list, 
         attack_id_list,
         attack_pc_list, 
         eval_results ) = load_attack_results( args.ori_data_path, args.ori_anno_path, args.tgt_anno_path, args.adv_anno_path,
                                             output_eval_file, output_pc_path, dataset_type, is_targeted ) 
        
        assert len(adv_pc_list) == len(eval_results), "The length of adversarial point clouds and evaluation results should be the same."
        print(f"Loaded {len(eval_results)} evaluation results")
        
    print('attack_id_list:', attack_id_list)
    
    if args.start_pc_eval and not args.start_gpt_eval:
        print("Starting point cloud evaluation...")
        if task_type == 'captioning':
            start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, device=device)
        else:
            start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, eval_results, device=device)
        print("Point cloud evaluation finished.")

        return valid_state
        
    # if args.wandb:
    #     run = wandb.init(project=args.wandb_project_name, 
    #                     name=args.wandb_run_name, 
    #                     id=args.wandb_run_name,
    #                     resume="allow",          # "allow" will resume if run exists, otherwise create new
    #                     reinit="finish_previous"
    #                     )
        
    if args.wandb:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.wandb_run_name}_{current_time}"
        run = wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            id=run_name,  # Use run_name as the run id for resuming
            resume="allow",          # "allow" will resume if run exists, otherwise create new
            reinit='finish_previous',
        )
    
    if attack_method == 'vision_attack':
        if 'gpt4point' in model_type:
            model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
            pc_encoder = model.get_pc_encoder_qformer()
            for param in pc_encoder.parameters():
                # param.data = param.data.to(dtype)
                param.requires_grad = True
            adv_func = FeatureSimilarityLoss( pc_encoder, dtype= dtype, is_targeted=is_targeted )
            generator = GPT4PointGenerator(model=model, llm_args=args)
        elif 'pointllm' in model_type:
            model, tokenizer, conv = init_PointLLM( model_name, dtype, device )
            pc_encoder = model.get_pc_encoder()
            adv_func = FeatureSimilarityLoss(pc_encoder, dtype= dtype, is_targeted=is_targeted )
            generator = PointLLMGenerator( model, tokenizer, conv, args )
    elif attack_method == 'caption_attack': 
        if 'gpt4point' in model_type:
            model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
            for param in model.parameters():
                # param.data = param.data.to(dtype)
                param.requires_grad = True
            adv_func = GPT4PointCaptionLoss(model, text_processor=txt_processors['eval'], is_targeted=is_targeted)
            generator = GPT4PointGenerator(model=model, llm_args=args)
        elif 'pointllm' in model_type:
            model, tokenizer, conv = init_PointLLM( model_name, dtype, device )
            adv_func = PointLLMCaptionLoss(model, tokenizer, conv, is_targeted=is_targeted ) 
            # generator = PointLLMGenerator(model=model, tokenizer=tokenizer, conv=conv, llm_args=args)
            generator = PointLLMGenerator( model, tokenizer, conv, args )
    
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
        
    ori_dataset = load_dataset( dataset_type = dataset_type, data_path= args.ori_data_path, category_file= args.modelnet_cat_file,
                               anno_path= args.ori_anno_path, normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                                    pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
                                    text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    
    tgt_dataset = load_dataset( dataset_type = dataset_type, data_path= args.tgt_data_path, category_file= args.modelnet_cat_file, 
                               anno_path= args.tgt_anno_path, normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                                    pts_processor= vis_processors['eval'] if 'gpt4point' in model_type else None,
                                    text_processor=txt_processors['eval'] if 'gpt4point' in model_type else None )
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)


    HiT_attacker = HiT_ADV( adv_func, evaluator, attack_lr=args.attack_lr,
                            central_num=args.central_num, total_central_num=args.total_central_num,
                            init_weight=args.init_weight, max_weight=args.max_weight, binary_step=args.binary_step,
                            num_iter=args.num_iter, clip_func=None,
                            cd_weight=args.cd_weight, ker_weight=args.ker_weight,
                            hide_weight=args.hide_weight, curv_loss_knn=args.curv_loss_knn,
                            max_sigm=args.max_sigm, min_sigm=args.min_sigm,
                            budget=args.budget, use_wandb = args.wandb )
    
        
    success_nums = success_rate * len(attack_pc_list)
    total_gpt_score = average_gpt_score * len(attack_pc_list)

    pretrained_eval_results_dict = {}
    if negitive_samples_filter_eval_path is not None:
        pretrained_eval_results = json.load(open(negitive_samples_filter_eval_path, "r")).get('results', [])
        for result in pretrained_eval_results:
            pretrained_eval_results_dict[str(result['object_id'])] = result

    try:
    # if 1:
        for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
            if 'pointllm' in model_type and task_type == 'classification' and i % 4 != 0:
                continue
            
            ori_id = ori_data_dict['object_id']
            tgt_id = tgt_data_dict['object_id']   
            ori_pc = ori_data_dict['point_cloud'].to(device=device, dtype=dtype)
            ori_labels = ori_data_dict['label']

            if is_targeted:
                tgt_labels = tgt_data_dict['label']
                if attack_method == 'vision_attack':
                    with torch.no_grad():
                        tgt_pc = tgt_data_dict['point_cloud'].to(device, dtype=dtype) # * convert to float32
                        tgt_feature = feature_normalize(pc_encoder(tgt_pc[..., :6]))
                else:
                    tgt_feature = None
            else:
                tgt_labels = ori_labels
                if attack_method == 'vision_attack':
                    with torch.no_grad():
                        tgt_feature = feature_normalize(pc_encoder(ori_pc[..., :6]))
                else:
                    tgt_feature = None
                
            attack_ids, attack_pcs = [], []
            attack_ori_labels, attack_tgt_labels = [], []   
            attack_tgt_features = [] if tgt_feature is not None else None  
            for j in range(len(ori_id)):
                attack_id = f"{ori_id[j]}_{tgt_id[j]}" if is_targeted  else f"{ori_id[j]}" 

                if attack_id not in attack_id_list: 
                    attack_ids.append(attack_id)
                    attack_pcs.append(ori_pc[j])
                    attack_ori_labels.append(ori_labels[j])
                    attack_tgt_labels.append(tgt_labels[j])
                    if tgt_feature is not None:
                        attack_tgt_features.append(tgt_feature[j])
            if len(attack_ids) == 0:
                print(f"Skip {i}th batch, since all the objects have been evaluated.")
                continue    
                
            attack_pcs = torch.stack(attack_pcs, dim=0)
            attack_tgt_features = torch.stack(attack_tgt_features, dim=0) if attack_tgt_features is not None else None
            if negitive_samples_filter_eval_path is not None:
                if is_targeted:
                    evaluation_results = [pretrained_eval_results_dict[str(attack_id).split('_')[0]] for attack_id in attack_ids]
                else:
                    evaluation_results = [pretrained_eval_results_dict[str(attack_id)] for attack_id in attack_ids]
                predict_accuracies = [result['accuracy'] for result in evaluation_results]
                predict_accuracies = torch.tensor(predict_accuracies, dtype=torch.int, device=device) 
            else:
                predict_accuracies, evaluation_results = evaluator(attack_pcs, attack_ori_labels)
                predict_accuracies = 1 - predict_accuracies if not is_targeted else predict_accuracies
            
            if predict_accuracies.sum() == 0:
                # print(evaluation_results)
                print(f"Skip {i}th batch, since all data are negative samples.")
                continue
        
            # Only keep attack_ids where predict_accuracies == 1
            attack_ids = [attack_ids[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            # print(attack_ids)
            attack_pcs =  [attack_pcs[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            attack_tgt_features = [attack_tgt_features[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]  if attack_tgt_features is not None else None
            
            attack_ori_labels = [attack_ori_labels[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            attack_tgt_labels = [attack_tgt_labels[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]

            attack_pcs = torch.stack(attack_pcs, dim=0)
            attack_tgt_features = torch.stack(attack_tgt_features, dim=0)  if attack_tgt_features is not None else None
            adv_pcs, success_num, eval_result = HiT_attacker.attack( attack_pcs, attack_tgt_labels, attack_tgt_features )
            success_nums += success_num

            for j, (attack_id, attack_ori_label) in enumerate(zip(attack_ids, attack_ori_labels)):
                eval_result[j].update({'object_id': attack_id} )
                if task_type == 'captioning':
                    total_gpt_score += eval_result[j].get('gpt_score', 0.0)
                if is_targeted:
                    eval_result[j].update({'original_label': attack_ori_label} )
            eval_results.extend(eval_result)

            adv_pcs = adv_pcs.float().cpu().numpy()
            attack_pcs = attack_pcs.float().cpu().numpy()
            
            for attack_id, attack_pc, adv_pc, attack_tgt_label in  zip(attack_ids, attack_pcs, adv_pcs, attack_tgt_labels):
                attack_id_list.append(attack_id)
                if dataset_type == 'Objaverse':
                    adv_pc_file_name_list.append(f"{attack_id}_{args.pointnum}.npy")
                attack_pc_list.append(attack_pc)
                adv_pc_list.append(adv_pc)
                adv_label_list.append(attack_tgt_label)
            # if i >= 10:
            break
            
    except Exception as e:
        valid_state = 0
        print(f"Error occurred during the attack: {e}. Skipping the rest of the batches.")

    saved_results['attack_pc_number'] = len(attack_pc_list)
    success_rate = success_nums / len(attack_pc_list) if len(attack_pc_list) > 0 else 0.0
    print("attack success rate: ", success_rate) 
    saved_results['ASR'] = success_rate
    if task_type == 'captioning':
        average_gpt_score = total_gpt_score / len(attack_pc_list) if len(attack_pc_list) > 0 else 0.0
        print("average gpt score: ", average_gpt_score)
        saved_results['average_gpt_score'] = average_gpt_score
    saved_results['eval_results'] = eval_results
    
    # * Save evaluation results
    with open(output_eval_file, "w") as f:
        json.dump(saved_results, f, indent=4)
        print(f"Saved evaluation results to {output_eval_file}")
            
    # * Save the adversarial point clouds and labels
    if dataset_type == 'ModelNet':
        output_base_pc_path = os.path.dirname(output_pc_path)
        print(f"Creating directory: {output_base_pc_path}")
        if not os.path.exists(output_base_pc_path):
            os.makedirs(output_base_pc_path)
        with open(output_pc_path, "wb") as f:
            pickle.dump((adv_pc_list, adv_label_list), f)
    else:
        if not os.path.exists(output_pc_path):
            os.makedirs(output_pc_path)
        for i, (adv_name, adv_pc) in enumerate(zip(adv_pc_file_name_list, adv_pc_list)):
            output_pc_file = os.path.join(output_pc_path, adv_name)
            np.save(output_pc_file, adv_pc)
    print(f"Saved {len(adv_pc_list)} adversarial point clouds to {output_pc_path}")
            
    if args.start_pc_eval:
        print("Starting point cloud evaluation...")
        if task_type == 'captioning':
            start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, device=device)
        else:
            start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, eval_results, device=device)
        print("Point cloud evaluation finished.")

    return valid_state
   
 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # victim model config
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--model_type", type=str, \
        default="gpt4point_opt2.7b") 
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help='Data type for the model')
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    # attack config
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "captioning"], help="Task type for the attack.")
    parser.add_argument("--attack_method",  type=str, default="vision_attack", choices=["vision_attack", "caption_attack"])
    parser.add_argument("--is_targeted",  action="store_true", default=False)
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument('--cd_weight', type=float, default=0.0001, help='cd_weight')
    parser.add_argument('--ker_weight', type=float, default=1., help='ker_weight')
    parser.add_argument('--hide_weight', type=float, default=1., help='hide_weight')
    parser.add_argument('--max_sigm', type=float, default=1.2, help='max_sigm')
    parser.add_argument('--min_sigm', type=float, default=0.1, help='min_sigm')
    parser.add_argument('--init_weight', type=float, default=1.0, help='init_weight')
    parser.add_argument('--max_weight', type=float, default=8.0, help='max_weight')
    parser.add_argument('--success_score_threshold', type=int, default=50, help='Success score threshold for the attack on captioning task.')

    parser.add_argument('--central_num', type=int, default=26)
    parser.add_argument('--total_central_num', type=int, default=2048)
    parser.add_argument('--curv_loss_knn', type=int, default=16, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--budget', type=float, default=0.55,
                        help='0.5 for l2 attack, 0.05 for linf attack')

    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    
    # defense config
    parser.add_argument("--has_defense",  action="store_true", default=False)
    parser.add_argument("--defend_method", type=str, default="sor", choices=["sor", "srs", "dupnet"], help="Defense method to use.")
    
    # parser.add_argument("--epsilon", type=float, default=1.0)
    # parser.add_argument("--gamma1", type=float, default=1.0)
    # parser.add_argument("--steps", type=int, default=300)
    
    #evaluate
    parser.add_argument("--start_pc_eval",  action="store_true", default=False)
    parser.add_argument("--negitive_samples_filter_eval_path", type=str, default=None, help="Path to save the evaluation results of negative samples filter.")
    parser.add_argument("--start_gpt_eval",  action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-5-nano", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    parser.add_argument("--output_eval_path", type=str, default='./eval_results', help="Path to save the evaluation results.")
    

    # data 
    parser.add_argument("--dataset_type", type=str, default='ModelNet', choices=['ModelNet', 'Objaverse'])
    parser.add_argument("--ori_data_path", type=str, default='./data')
    parser.add_argument("--tgt_data_path", type=str, default='./data')
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--ori_anno_path", type=str, default=None)
    parser.add_argument("--adv_anno_path", type=str, default=None)
    parser.add_argument("--tgt_anno_path", type=str, default=None)
    parser.add_argument("--do_preprocess", type=bool,  default=True)
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
    
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    # parser.add_argument("--wandb_run_id", type=str, default=None, help="wandb run id for resuming the run")

    args = parser.parse_args()

    valid_state = 0
    while valid_state == 0:
        valid_state = main(args)
    
    