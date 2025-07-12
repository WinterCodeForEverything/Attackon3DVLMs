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

from dataset.data.modelnet import ModelNet
from dataset.data.objaverse import ObjaverseDataset
from dataset.data.utils import compute_normals, redirect_normals

from victim_models.pointllm.conversation import conv_templates, SeparatorStyle
from victim_models.pointllm.utils import disable_torch_init
from victim_models.pointllm.model import *
from victim_models.pointllm.model.utils import KeywordsStoppingCriteria

from attack_methods.evaluation.gpt_evaluator import GPTEvaluation

# from attack_methods.ModelNet40.evaluate import Evaluation
from attack_methods.evaluation.pc_evaluator.pc_evaluator import start_pc_evaluation


device = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100


    
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
    
    #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
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

# def generate_tgt_caption(tgt_labels, prompt_index=0):
#     tgt_captions = []
#     for label_name in range(tgt_labels):
#         if prompt_index == 0:
#             tgt_caption = f"This is an object of {label_name}."
#         elif prompt_index == 1:
#             tgt_caption = label_name
#         tgt_captions.append(tgt_caption)

#     return tgt_captions


class LLMLoss(torch.nn.Module):
    def __init__(self, model, tokenizer, conv, prompt_index, is_targeted=True):
        super(LLMLoss, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_index = prompt_index
        self.conv = conv
        self.is_targeted = is_targeted
    
    def generate_and_mask_target(
        self,
        conversations
    ):

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length= self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets
        sep = self.conv.sep + self.conv.roles[1] + ": " 
        # print(f"sep: {sep}")
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2: # * can handle padded tokens
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                    
            # # Detokenize targets to text
            # detokenized_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
            # print(f"Detokenized targets: {detokenized_targets}")

        return input_ids, targets


    def forward(self, adv_pc, tgt_labels ):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        
        # tgt_captions = generate_tgt_caption( tgt_labels, 1 )
        tgt_captions = tgt_labels
        
        conversations = []
        for tgt_caption in tgt_captions:
            tgt_conv = self.conv.copy()
            tgt_conv.append_message( self.conv.roles[1], tgt_caption )
            conversations.append( tgt_conv.get_prompt() )
        
        input_caption_ids, tgt_caption_ids = self.generate_and_mask_target(
            conversations
        )
        
        outputs = self.model(
                input_ids=input_caption_ids.to(device), 
                point_clouds= adv_pc.to(self.model.dtype),
                labels= tgt_caption_ids.to(device),
                return_dict=True,
                    )
        if self.is_targeted:
            loss = outputs.loss
        else:
            loss = -1 * outputs.loss
        # print(f"loss: {loss}")

        return loss

class ModelGeneration(torch.nn.Module):
    def __init__(self, model, tokenizer, conv,  args):
        super(ModelGeneration, self).__init__()
        self.model = model
        # self.conv = conv
        self.tokenizer = tokenizer
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_length = args.max_length
        # stopping_criteria
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

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
    
def main(args):
    
    #lr = args.lr
    #budget = args.budget
    dataset_type = args.dataset_type
    is_targeted = args.is_targeted
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
        
    ori_dataset = load_dataset( dataset_type = dataset_type, data_path= args.ori_data_path, anno_path= args.ori_anno_path,
                                 normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal ) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    
    tgt_dataset = load_dataset( dataset_type = dataset_type, data_path= args.tgt_data_path, anno_path= args.tgt_anno_path,
                                normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal ) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)



    model, tokenizer, conv = init_model(args)
    adv_func = LLMLoss(model, tokenizer, conv, args.prompt_index, is_targeted=args.is_targeted ) 
    
    genetration = ModelGeneration( model, tokenizer, conv, args )
    eval_func = GPTEvaluation( genetration, args.gpt_type, is_targeted=args.is_targeted )
    
    


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
            ori_pc = ori_data_dict['point_cloud'].to(device, dtype= model.dtype) 
            tgt_pc = tgt_data_dict['point_cloud'].to(device, dtype=  model.dtype) 
            if is_targeted:
                if dataset_type == 'ModelNet':
                    tgt_labels =  [modelnet_cat_list[label.item()] for label in tgt_data_dict['label']]
                else:
                    tgt_labels = tgt_data_dict['label']
            else:
                if dataset_type == 'ModelNet':
                    tgt_labels = [modelnet_cat_list[label.item()] for label in ori_data_dict['label']]
                else:
                    tgt_labels = ori_data_dict['label']

                
            adv_pc, success_num, eval_result = HiT_attacker.attack( ori_pc, tgt_labels )
            success_nums += success_num
            for j in range(ori_pc.shape[0]):
                if i * args.batch_size + j < eval_attack_num:
                    continue
                eval_result[j].update({'object_id': f"{ori_id[j]}"})
            #adv_pc = torch.from_numpy(adv_pc).to(device)
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
        start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, args.output_eval_path, device, eval_results)
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
    
    