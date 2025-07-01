#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


# from tqdm import tqdm
from transformers import AutoTokenizer
import os
import wandb
from attack_methods.ShapeAttack.HiT_ADV import HiT_ADV
import pickle
import json


# from pointllm.data import ModelNet
# from victim_models.gpt4point.models.gpt4point_models.gpt4point_opt import GPT4Point_OPT
from victim_models.gpt4point.models import load_model_and_preprocess



from attack_methods.ModelNet40.evaluate import Evaluation
from attack_methods.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator

#from ._train_adv_pc_trans_ import generate_tgt_caption, generate_and_mask_target


device = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100

cat_file = '/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt'
label_names = [line.rstrip() for line in open(cat_file)]


class ModelNet(Dataset):
    def __init__(self, pc_file, pts_processor, text_processor):
        super(ModelNet, self).__init__()
        
        self.pts_processor = pts_processor
        self.text_processor = text_processor

        self.catfile = "/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"
        #os.path.join(os.path.dirname(__file__), "modelnet_config", 'modelnet40_shape_names_modified.txt')

        # "tv_stand" -> "tv stand"
        self.categories = [line.rstrip() for line in open(self.catfile)] # * list of category names

        # print('Load processed data from %s...' % pc_file)
        with open(pc_file, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)
        
        # * print len
        print(f"Load {len(self.list_of_points)} data from {pc_file}.")

    def __len__(self):
        return len(self.list_of_labels)
    

    def __getitem__(self, index):
        point, label = self.list_of_points[index], self.list_of_labels[index]

        # print(f"point shape: {point.shape}, label: {label}")
        # point color
        if point.shape[-1] == 3:
            white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # white rgb
            white_colors = np.tile(white_color, (point.shape[0], 1))
            point = np.concatenate((point, white_colors), axis=1)
            
        elif point.shape[-1] == 9:
            raw_point = point[:, :6]  # * remove the normal channel if it exists

        # point transform
        raw_point = self.pts_processor(raw_point)
        point[:, :6] = raw_point

        # caption = "A 3D point cloud of " + self.categories[int(label)] + "."
        # print(f"caption before process: {caption}")
        # caption = self.text_processor( caption )
        # print(f"caption after process: {caption}")

        return {
            "indice": index,
            "point": point,
            "label": label
        }
    
PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

def generate_tgt_caption(tgt_labels, prompt_index=0):
    tgt_captions = []
    for b in range(tgt_labels.shape[0]):
        label_name= label_names[int(tgt_labels[b].item())]
        if prompt_index == 0:
            tgt_caption = f"This is an object of {label_name}."
        elif prompt_index == 1:
            tgt_caption = label_name
        tgt_captions.append(tgt_caption)

    return tgt_captions

def generate_and_mask_target(
    tokenizer,
    conv,
    conversations
):

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " # This is an object of
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
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


def load_dataset(pc_file, pts_processor, text_processor):
    # print(f"Loading {split} split of ModelNet datasets.")
    dataset = ModelNet(pc_file=pc_file, pts_processor=pts_processor, text_processor=text_processor)
    print("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class LLMLoss(torch.nn.Module):
    def __init__(self, model, text_processor=None):
        super(LLMLoss, self).__init__()
        self.model = model
        # self.vis_processors = vis_processors
        self.text_processor = text_processor
        # print("vis_processors:", vis_processors)
        # print("txt_processors:", txt_processors)
    
        
    def forward(self, adv_pc, tgt_labels ):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        
        # # Ensure adv_pc is float32 for model compatibility
        # adv_pc = adv_pc.to(torch.float32)
        
        # adv_pc = self.vis_processors['eval'](adv_pc) # * process the point cloud
        
        captions = []
        for b in range(tgt_labels.shape[0]):
            caption = label_names[int(tgt_labels[b].item())]
            if self.text_processor is not None:
                caption = self.text_processor(caption)
            # caption = label_name      #"A 3D point cloud of " + label_name + "."
            # caption = self.text_processor(caption) # * process the text
            # print(f"Caption for label {label_name}: {caption}")
            captions.append(caption)
        # caption = "A 3D point cloud of " + label_names[int(tgt_labels[0].item())] + "."
        
        
        sample = {
            "point": adv_pc,
            "text_input": captions,
        }
        
        outputs, targets = self.model(
            sample
        )
        
        loss = self.compute_loss(outputs["logits"], targets)
        # print(loss)
        
        # loss = outputs["loss"]
        # print(f"loss: {loss}")
        
        return loss
        
        # tgt_captions = generate_tgt_caption( tgt_labels, self.prompt_index )
     
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
            temperature=self.temperature,
        )
        
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
    output_pc_file = args.output_pc_file
    #pointnum = args.pointnum
    #prompt_index = args.prompt_index
    #batch_size = args.batch_size
    
    
    success_rate = 0.0
    saved_results = {"prompt": PROMPT_LISTS[args.prompt_index] }
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
        
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    model = model.to(torch.float32)
    for param in model.parameters():
        param.requires_grad = True
        # param = param.to(torch.bfloat16) # * convert model to bfloat16

    ori_dataset = load_dataset(pc_file= args.ori_data_path, pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataset = load_dataset(pc_file= args.tgt_data_path, pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)



    # model, tokenizer, conv = init_model(args)
    adv_func = LLMLoss( model )
    
    # candidate_lists_names = []
    candidate_lists_names = label_names
    genetrate = ModelGeneration( model, args )
    eval_func = Evaluation( genetrate, args.gpt_type, candidate_lists_names )

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
            ori_pc = ori_data_dict['point'].to(device, dtype=torch.float32) # * convert to float32
            # Transform ori_pc to model's datatype
            # ori_pc = ori_pc.to()
            tgt_labels = tgt_data_dict['label'].to(device)
                
            adv_pc, success_num, eval_result = HiT_attacker.attack( ori_pc, tgt_labels )
            success_nums += success_num
            for j in range(obj_id.shape[0]):
                if i * args.batch_size + j < eval_attack_num:
                    continue
                eval_result[j].update({'object_id': obj_id[j].item()})
            #adv_pc = torch.from_numpy(adv_pc).to(device)
            eval_results.extend(eval_result)
            
            saved_lable = tgt_data_dict['label'].cpu().numpy()

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
        print(f"Saved {len(adv_pc_list)} adversarial point clouds to {output_pc_file}")
        
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')

    args = parser.parse_args()

    main(args)
    
    