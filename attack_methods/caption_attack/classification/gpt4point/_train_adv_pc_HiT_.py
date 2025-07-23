#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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
from attack_methods.evaluation.pc_evaluator.pc_evaluator import start_pc_evaluation

# from attack_methods.ModelNet40.evaluate import Evaluation
# from attack_methods.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator


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

class LLMLoss(torch.nn.Module):
    def __init__(self, model, allowed_class=None, text_processor=None, is_targeted=True):
        super(LLMLoss, self).__init__()
        self.model = model
        self.allowed_class = allowed_class
        # self.vis_processors = vis_processors
        self.text_processor = text_processor
        # print("vis_processors:", vis_processors)
        # print("txt_processors:", txt_processors)
        self.is_targeted = is_targeted
    
        
    def forward(self, adv_pc, tgt_labels ):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        
        captions = []
        for label in tgt_labels:
            caption = label
            if self.text_processor is not None:
                caption = self.text_processor(caption)
            captions.append(caption)

        sample = {
            "point": adv_pc,
            "text_input": captions,
        }
        
        outputs, targets = self.model(
            sample
        )
        # print(outputs["logits"].shape)
        
        loss = self.compute_loss(outputs["logits"], targets)
        
        if not self.is_targeted:
            loss = -1 * loss  # * for untargeted attack, we want to maximize the loss
        
        return loss
        
     
    def compute_loss(self, logits, labels):
        """
        Compute the loss for the adversarial point clouds.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()  # * B, L, V
        vocab_size = shift_logits.shape[-1]
        shift_labels = labels[..., 1:].contiguous()  # * B, L
        B = shift_labels.shape[0]
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')  # Compute loss per token, ignoring -100
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(B, -1)
        loss = loss.sum(dim=1) / (loss != 0).sum(dim=1).clamp(min=1)  # Compute mean loss per batch, ignoring padding
        
        return loss   


    def multi_target_ce(
        self,
        logits: torch.Tensor,
        targets: torch.LongTensor,
        # allowed: torch.Tensor,
        ignore_index: int = -100,
        kappa: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute per-sample multi-target margin loss over time (or spatial) dimension.

        Args:
            logits:     (B, L, C) float tensor of model outputs.
            targets:    (B, L)   long tensor of true labels, or ignore_index.
            allowed:    bool tensor of shape (B, L, C) or (C,),
                        where True indicates a valid target class.
            ignore_index: int label in `targets` to skip.
            kappa:      margin constant; loss = ReLU(true_logit - lse_allowed + kappa).

        Returns:
            loss: Tensor of shape (B,) giving the mean loss over valid positions for each sample.
        """
        B, L, C = logits.shape
        device = logits.device

        allowed = self.allowed_class
        # Broadcast allowed mask if needed
        if allowed.dim() == 1:
            allowed = allowed.view(1, 1, C).expand(B, L, C)
        allowed = allowed.to(device)

        # Compute per-position loss
        # Gather true-logit: shape (B, L)
        idx = torch.arange(B, device=device)[:, None]
        pos = torch.arange(L, device=device)[None, :]
        true_logit = logits[idx, pos, targets.clamp(min=0)]  # we’ll mask out ignore_index next

        # Mask out invalid classes to -inf, then logsumexp over C
        inf_logits = logits.masked_fill(~allowed, float('-inf'))
        lse_allowed = torch.logsumexp(inf_logits, dim=2)    # shape (B, L)

        # Raw margin loss per position
        loss_raw = F.relu(true_logit - lse_allowed + kappa)  # (B, L)

        # Mask out positions where targets == ignore_index
        valid = (targets != ignore_index)                     # (B, L)
        loss_raw = loss_raw * valid.to(loss_raw.dtype)

        # Sum and count per sample
        sum_loss   = loss_raw.sum(dim=1)                      # (B,)
        count_loss = valid.sum(dim=1).clamp(min=1).to(sum_loss.dtype)  # (B,)

        # Per-sample mean (zero if no valid positions)
        loss_per_sample = sum_loss / count_loss               # (B,)

        return loss_per_sample


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
            top_p=self.top_p if self.temperature > 0.0 else None,
            temperature=self.temperature if self.temperature > 0.0 else None,
        )
        
        return outputs


def main(args):
    
    valid_state = 1
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
    attack_id_list, attack_pc_list = [], []
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
        for eval_result in eval_results:
            attack_id_list.append(eval_result['object_id']) # * use index as object id
        if dataset_type == 'ModelNet':
            adv_pc_list, adv_label_list = pickle.load(open(output_pc_path, "rb"))
            total_ori_pc_list, _ = pickle.load(open(args.ori_data_path, "rb")) 
            attack_pc_list = [total_ori_pc_list[int(attack_id.split('_')[0])] for attack_id in attack_id_list]
        elif dataset_type == 'Objaverse':
            for i, file in enumerate(os.listdir(output_pc_path)):
                if file.endswith('.npy'):
                    adv_pc_file_name_list.append(file)
                    adv_pc = np.load(os.path.join(output_pc_path, file))
                    adv_pc_list.append(adv_pc)
                    assert adv_annos[i]['conversations'][1]['from'] == 'gpt'
                    adv_label = adv_annos[i]['conversations'][1]['value']
                    adv_label_list.append(adv_label)
                    
                    attack_pc_name = file.split('_')[0] + f"_{args.pointnum}.npy"
                    attack_pc = np.load(os.path.join(args.ori_data_path, attack_pc_name))
                    attack_pc_list.append(attack_pc)
            
        
        assert len(adv_pc_list) == len(eval_results), "The length of adversarial point clouds and evaluation results should be the same."
        print(f"Loaded {len(eval_results)} evaluation results")
        
        # if args.redirect_normal and len(adv_pc_list) != 0 and len(attack_pc_list) != 0:
        #     attack_pc_list = [redirect_normals(pc) for pc in attack_pc_list]
        #     adv_pc_list = [redirect_normals(pc) for pc in adv_pc_list]
    
    
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



    adv_func = LLMLoss( model, is_targeted= is_targeted )
    
    generator = ModelGeneration( model, args )
    # filter_function = GPTEvaluation( generator, args.gpt_type )
    eval_func = GPTEvaluation( generator, args.gpt_type, is_targeted= is_targeted )

    HiT_attacker = HiT_ADV( adv_func, eval_func, attack_lr=args.attack_lr,
                               central_num=args.central_num, total_central_num=args.total_central_num,
                               init_weight=args.init_weight, max_weight=args.max_weight, binary_step=args.binary_step,
                               num_iter=args.num_iter, clip_func=None,
                               cd_weight=args.cd_weight, ker_weight=args.ker_weight,
                               hide_weight=args.hide_weight, curv_loss_knn=args.curv_loss_knn,
                               max_sigm=args.max_sigm, min_sigm=args.min_sigm,
                               budget=args.budget, use_wandb = args.wandb )
    
    
    
    # eval_attack_num = len(eval_results)
    success_nums = success_rate * len(attack_pc_list)
    if dataset_type == 'ModelNet':
        modelnet_cat_list = [line.rstrip() for line in open(args.modelnet_cat_file)]

    try:
    # if 1:
        for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
            ori_id = ori_data_dict['object_id']
            tgt_id = tgt_data_dict['object_id']   
            ori_pc = ori_data_dict['point_cloud'].to(device, dtype= torch.float32) # * convert to float32

            if dataset_type == 'ModelNet':
                ori_labels = [modelnet_cat_list[label.item()] for label in ori_data_dict['label']]
            else:
                ori_labels = ori_data_dict['label']
            # tgt_pc = tgt_data_dict['point_cloud'].to(device, dtype= torch.float32) # * convert to float32 
            if is_targeted:
                if dataset_type == 'ModelNet':
                    tgt_labels =  [modelnet_cat_list[label.item()] for label in tgt_data_dict['label']]
                else:
                    tgt_labels = tgt_data_dict['label']
            else:
                tgt_labels = ori_labels
            
            attack_ids, attack_pcs = [], []
            attack_ori_labels, attack_tgt_labels = [], []     
            for j in range(len(ori_id)):
                if is_targeted:
                    attack_id = f"{ori_id[j]}_{tgt_id[j]}"  
                else:
                    attack_id = f"{ori_id[j]}"  

                if attack_id not in attack_id_list: 
                    attack_ids.append(attack_id)
                    attack_pcs.append(ori_pc[j])
                    attack_ori_labels.append(ori_labels[j])
                    attack_tgt_labels.append(tgt_labels[j])
            if len(attack_ids) == 0:
                print(f"Skip {i}th batch, since all the objects have been evaluated.")
                continue    
            
                
            attack_pcs = torch.stack(attack_pcs, dim=0)
            predict_accuracies, evaluation_results = eval_func(attack_pcs, attack_ori_labels)
            
            if not is_targeted:
                predict_accuracies = 1 - predict_accuracies
            
            if predict_accuracies.sum() == 0:
                print(evaluation_results)
                print(f"Skip {i}th batch, since all data are negative samples.")
                continue
           
            # print(attack_ids)
            # print(predict_accuracies)
            # Only keep attack_ids where predict_accuracies == 1
            attack_ids = [attack_ids[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            # print(attack_ids)
            attack_pcs =  [attack_pcs[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            
            attack_ori_labels = [attack_ori_labels[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]
            attack_tgt_labels = [attack_tgt_labels[idx] for idx, acc in enumerate(predict_accuracies) if acc == 1]

            attack_pcs = torch.stack(attack_pcs, dim=0)
            adv_pcs, success_num, eval_result = HiT_attacker.attack( attack_pcs, attack_tgt_labels )
            success_nums += success_num

            for j, (attack_id, attack_ori_label) in enumerate(zip(attack_ids, attack_ori_labels)):
                eval_result[j].update({'object_id': attack_id} )
                if is_targeted:
                    eval_result[j].update({'original_label': attack_ori_label} )
            eval_results.extend(eval_result)
            
            attack_pcs = attack_pcs.float().cpu().numpy()
            
            for attack_id, attack_pc, adv_pc, attack_tgt_label in  zip(attack_ids, attack_pcs, adv_pcs, attack_tgt_labels):
                attack_id_list.append(attack_id)
                if dataset_type == 'Objaverse':
                    adv_pc_file_name_list.append(f"{attack_id}_{args.pointnum}.npy")
                attack_pc_list.append(attack_pc)
                adv_pc_list.append(adv_pc)
                adv_label_list.append(attack_tgt_label)
            
            # break
            
    except Exception as e:
        valid_state = 0
        print(f"Error occurred during the attack: {e}. Skipping the rest of the batches.")
    
    saved_results['attack_pc_number'] = len(attack_pc_list)
    success_rate = success_nums / len(attack_pc_list)
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
        start_pc_evaluation(attack_pc_list, adv_pc_list, attack_id_list, args.output_eval_path, eval_results, device)
        print("Point cloud evaluation finished.")
    
    
    return valid_state

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

    valid_state = 0
    while valid_state == 0:
        valid_state = main(args)
    
    