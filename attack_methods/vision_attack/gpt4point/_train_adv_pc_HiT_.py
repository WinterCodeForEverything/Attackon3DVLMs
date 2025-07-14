#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


# from tqdm import tqdm
from transformers import AutoTokenizer
import os
import wandb
from attack_methods.ShapeAttack.HiT_ADV import HiT_ADV
import pickle
import json

from victim_models.gpt4point.models import load_model_and_preprocess
from dataset.data.modelnet import ModelNet
from dataset.data.objaverse import ObjaverseDataset
from dataset.data.utils import redirect_normals

from attack_methods.evaluation.gpt_evaluator import GPTEvaluation

# from attack_methods.ModelNet40.evaluate import Evaluation
from attack_methods.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator


device = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100


    
PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]


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

class FeatureSimilarityLoss(torch.nn.Module):
    def __init__(self, model, is_targeted=True):
        super(FeatureSimilarityLoss, self).__init__()
        self.model = model
        self.is_targeted = is_targeted
        
    def forward(self, adv_pc, tgt_feature):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        # adv_pc = adv_pc.to(device)
        adv_feature = self.model(adv_pc)
        #tgt_feature = self.model(tgt_pc)
        
        # Normalize the features
        adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
        #print(f"adv_feature: {adv_feature.shape}")
        
        # Compute the embedding similarity
        embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2), dim=1)
        
        if self.is_targeted:
            loss = 1 - embedding_sim 
        else:
            loss = embedding_sim
        
        return loss


class ModelGeneration(torch.nn.Module):
    def __init__(self, model, args):
        super(ModelGeneration, self).__init__()
        self.model = model
        self.num_beams = args.num_beams
        self.max_length = args.max_length
        self.top_p = args.top_p
        self.temperature = args.temperature
        
    def generate(self, points):
        """
        Generate the output of the model.
        """
        
        samples = {
            "point": points
        }
        outputs = self.model.generate(
            samples,
            num_beams=self.num_beams,
            use_nucleus_sampling= True if self.temperature > 0.0 else False,
            max_length=self.max_length,
            top_p=self.top_p,
            temperature=self.temperature if self.temperature > 0.0 else None,
        )
        
        return outputs

def start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, output_eval_dir, evaluated_results = None):
    
    pc_evaluator = PointCloudEvaluator(k_nn=32, device=device)
    
    
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
    # print( "len(statistic_object_ids):", len(statistic_object_ids))
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
    
    #lr = args.lr
    #budget = args.budget
    is_targeted = args.is_targeted
    dataset_type = args.dataset_type
    assert dataset_type in ['ModelNet', 'Objaverse'], f"Unsupported dataset type: {dataset_type}"
    output_pc_path = args.output_pc_path
    #pointnum = args.pointnum
    #prompt_index = args.prompt_index
    #batch_size = args.batch_size
    
    if dataset_type == 'Objaverse' and  not os.path.exists(args.adv_anno_path):
        ori_annos = json.load(open(args.ori_anno_path, "r"))
        tgt_annos = json.load(open(args.tgt_anno_path, "r"))
        adv_annos = []
        for i in range(len(ori_annos)):
            adv_anno = tgt_annos[i].copy()
            adv_anno['object_id'] = f"{ori_annos[i]['object_id']}_{tgt_annos[i]['object_id']}"
            adv_annos.append(adv_anno)
        with open(args.adv_anno_path, "w") as f:
            json.dump(adv_annos, f, indent=4)
    elif dataset_type == 'Objaverse' and os.path.exists(args.adv_anno_path):
        adv_annos = json.load(open(args.adv_anno_path, "r"))
    
    success_rate = 0.0
    saved_results = {"prompt": PROMPT_LISTS[args.prompt_index] }
    eval_results = []
    ori_pc_list, obj_id_list = [], []
    adv_pc_file_name_list = []
    adv_pc_list, adv_label_list = [], []
    
    output_file = f"{dataset_type}_classification_prompt_{args.prompt_index}_evaluated_{args.gpt_type}_in_training.json"
    output_eval_file = os.path.join(args.output_eval_path, output_file )
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
    elif os.path.exists(output_eval_file) and os.path.exists(output_pc_path):
        eval_data = json.load(open(output_eval_file, "r"))
        eval_results = eval_data.get('eval_results', [])
        success_rate = eval_data.get('ASR', 0.0)
        if dataset_type == 'ModelNet':
            adv_pc_list, adv_label_list = pickle.load(open(output_pc_path, "rb"))
            obj_id_list = list(range(len(adv_pc_list))) # * use index as object id
            total_ori_pc_list, _ = pickle.load(open(args.ori_data_path, "rb")) 
            ori_pc_list = total_ori_pc_list[:len(adv_pc_list)] # * ensure the length of ori_pc_list is the same as adv_pc_list
        elif dataset_type == 'Objaverse':
            for i, file in enumerate(os.listdir(output_pc_path)):
                if file.endswith('.npy'):
                    obj_id_list.append(file.split(f'_{args.pointnum}.npy')[0])
                    adv_pc_file_name_list.append(file)
                    adv_pc = np.load(os.path.join(output_pc_path, file))
                    # print( adv_pc.dtype, adv_pc.shape )
                    adv_pc_list.append(adv_pc)
                    assert adv_annos[i]['conversations'][1]['from'] == 'gpt'
                    adv_label = adv_annos[i]['conversations'][1]['value']
                    adv_label_list.append(adv_label)
                    
                    ori_pc_name = file.split('_')[0] + f"_{args.pointnum}.npy"
                    ori_pc = np.load(os.path.join(args.ori_data_path, ori_pc_name))
                    ori_pc_list.append(ori_pc)
            
        
        assert len(adv_pc_list) == len(eval_results), "The length of adversarial point clouds and evaluation results should be the same."
        print(f"Loaded {len(eval_results)} evaluation results")
        
        if args.redirect_normal and len(adv_pc_list) != 0 and len(ori_pc_list) != 0:
            ori_pc_list = [redirect_normals(pc) for pc in ori_pc_list]
            adv_pc_list = [redirect_normals(pc) for pc in adv_pc_list]
    
    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)
        
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    # model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = True
        # param = param.to(torch.bfloat16) # * convert model to bfloat16

    ori_dataset = load_dataset( dataset_type = dataset_type, data_path= args.ori_data_path, anno_path= args.ori_anno_path,
                                 normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                               pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataset = load_dataset( dataset_type = dataset_type, data_path= args.tgt_data_path, anno_path= args.tgt_anno_path,
                                 normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal,
                               pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)



    pc_encoder = model.get_pc_encoder_qformer()

    adv_func = FeatureSimilarityLoss(pc_encoder)
    
    genetrate = ModelGeneration( model, args )
    eval_func = GPTEvaluation( genetrate, args.gpt_type, is_targeted=is_targeted)

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
    if dataset_type == 'ModelNet':
        modelnet_cat_list = [line.rstrip() for line in open(args.modelnet_cat_file)]

    try:
        for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
            if i * args.batch_size < eval_attack_num:
                print(f"Skip {i}th batch, since it has been evaluated.")
                continue
            ori_id = ori_data_dict['object_id']
            tgt_id = tgt_data_dict['object_id']
            ori_pc = ori_data_dict['point_cloud'].to(device, dtype= torch.float32) # * convert to float32
            tgt_pc = tgt_data_dict['point_cloud'].to(device, dtype= torch.float32) # * convert to float32
            
            if is_targeted:
                if dataset_type == 'ModelNet':
                    tgt_labels =  [modelnet_cat_list[label.item()] for label in tgt_data_dict['label']]
                else:
                    tgt_labels = tgt_data_dict['label']
                with torch.no_grad():
                    tgt_feature = pc_encoder(tgt_pc[..., :6])
                    if torch.isnan(tgt_feature).any():
                        print(f"NaN detected in tgt_feature. Skip this sample.")
                        continue
                    tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            else:
                if dataset_type == 'ModelNet':
                    tgt_labels =  [modelnet_cat_list[label.item()] for label in ori_data_dict['label']]
                else:
                    tgt_labels = ori_data_dict['label']
                with torch.no_grad():
                    tgt_feature = pc_encoder(ori_pc[..., :6])
                    if torch.isnan(tgt_feature).any():
                        print(f"NaN detected in tgt_feature. Skip this sample.")
                        continue
                    tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)


                
            adv_pc, success_num, eval_result = HiT_attacker.attack( ori_pc, tgt_labels, tgt_feature )
            success_nums += success_num
            for j in range(ori_pc.shape[0]):
                if i * args.batch_size + j < eval_attack_num:
                    continue
                if is_targeted:
                    eval_result[j].update({'object_id': f"{ori_id[j]}_{tgt_id[j]}"})
                else:
                    eval_result[j].update({'object_id': f"{ori_id[j]}"})
            
            eval_results.extend(eval_result)

            ori_pc = ori_pc.float().cpu().numpy() 
            if is_targeted:
                saved_label = tgt_data_dict['label']
                for b, (oid, op, tid, ap, adv_label) in enumerate( zip(ori_id, ori_pc, tgt_id, adv_pc, saved_label) ):
                    if dataset_type == 'ModelNet':
                        obj_id_list.append(oid.item())
                    else:
                        obj_id_list.append(f"{oid}_{tid}")
                        adv_pc_file_name_list.append(f"{oid}_{tid}_{args.pointnum}.npy")
                    ori_pc_list.append(op)
                    adv_pc_list.append(ap)
                    adv_label_list.append(adv_label)
            else:
                saved_label = ori_data_dict['label']
                for b, (oid, op, ap, adv_label) in enumerate( zip(ori_id, ori_pc, adv_pc, saved_label) ):
                    if dataset_type == 'ModelNet':
                        obj_id_list.append(oid.item())
                    else:
                        obj_id_list.append(f"{oid}")
                        adv_pc_file_name_list.append(f"{oid}_{args.pointnum}.npy")
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
    if dataset_type == 'ModelNet':
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
        start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, args.output_eval_path, eval_results)
        print("Point cloud evaluation finished.")
    
 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="gpt4point_opt") 
    parser.add_argument("--model_type", type=str, \
        default="gpt4point_opt2.7b") 
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    parser.add_argument("--is_targeted",  action="store_true", default=False)
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
    parser.add_argument("--start_pc_eval",  action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-4.1-mini", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    parser.add_argument("--output_eval_path", type=str, default='./eval_results', help="Path to save the evaluation results.")
    

    # data 
    parser.add_argument("--dataset_type", type=str, default='ModelNet', choices=['ModelNet', 'Objaverse'])
    parser.add_argument("--ori_data_path", type=str, default='./data')
    parser.add_argument("--tgt_data_path", type=str, default='./data')
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--ori_anno_path", type=str, default=None)
    parser.add_argument("--adv_anno_path", type=str, default=None)
    parser.add_argument("--tgt_anno_path", type=str, default=None)
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

    args = parser.parse_args()

    main(args)
    
    