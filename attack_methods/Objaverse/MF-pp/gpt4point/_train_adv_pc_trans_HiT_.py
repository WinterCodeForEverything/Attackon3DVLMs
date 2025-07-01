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


class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 pts_processor=None,
                 text_processor=None,
                #  tokenizer=None,
                #  pointnum=8192,
                #  split='train',
                #  conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 use_color=True,
                 use_normal=True,
                #  data_args=None
                ):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(ObjectPointCloudDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.pts_processor = pts_processor
        self.text_processor = text_processor
        # self.tokenizer = tokenizer
        # self.split = split 
        # if conversation_types is None:
        #     self.conversation_types = ("simple_description",)
        # else:
        #     self.conversation_types = conversation_types

        # self.data_args = data_args
        # self.normalize_pc = True
        self.use_color = use_color
        self.use_normal = use_normal

        # self.pointnum = pointnum
        # self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        # self.point_indicator = '<point>'

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        # # * print the conversations_type
        # print(f"Using conversation_type: {self.conversation_types}") 
        # # * print before filtering
        # print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # # * iterate the list and filter
        # # * these two ids have corrupted colored point files, so filter them when use_color is True
        # filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # # Iterate the list, filter those "conversation_type" not in self.conversation_types
        # self.list_data_dict = [
        #     data for data in self.list_data_dict 
        #     if data.get('conversation_type', 'simple_description') in self.conversation_types 
        #     and data.get('object_id') not in filter_ids
        # ]

        # # * print after filtering
        # print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # # * print the size of different conversation_type
        # for conversation_type in self.conversation_types:
        #     print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

        # if self.data_args is not None and self.data_args.data_debug_num > 0:
        #     self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
        #     # * print all the scan_id in debug mode, not using for loop
        #     print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))
        # elif self.data_args is not None and self.data_args.split_train_val:
        #     # * split train and val with 9:1 ratios
        #     if self.split == 'train':
        #         self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
        #         print(f"Train set size: {len(self.list_data_dict)}")
        #     else:
        #         self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
        #         print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    
    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))
            
        if not self.use_normal:
            point_cloud = point_cloud[:, :6]

            if not self.use_color:
                point_cloud = point_cloud[:, :3]
        else:
            if not self.use_color:
                point_cloud = np.concatenate( (point_cloud[:, :3], point_cloud[:, 6:9] ), axis=1 )

        return point_cloud

    # def pc_norm(self, pc):
    #     """ pc: NxC, return NxC """
    #     xyz = pc[:, :3]
    #     other_feature = pc[:, 3:]

    #     centroid = np.mean(xyz, axis=0)
    #     xyz = xyz - centroid
    #     m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    #     xyz = xyz / m

    #     pc = np.concatenate((xyz, other_feature), axis=1)
    #     return pc
    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        # if isinstance(index, int):
        #     sources = [sources]
        # assert len(sources) == 1, "sources should be a list"
        
        object_id = self.list_data_dict[index]['object_id']

        # Point cloud representation
        point = self._load_point_cloud(object_id) # * N, C
        
        raw_point = None
        if point.shape[-1] == 3:
            white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # white rgb
            white_colors = np.tile(white_color, (point.shape[0], 1))
            raw_point = np.concatenate((point, white_colors), axis=1)
        elif point.shape[-1] == 9:
            raw_point = point[:, :6]  # * remove the normal channel if it exists

        # point transform
        if self.pts_processor is not None:
            raw_point = self.pts_processor(raw_point)
        if raw_point is not None:
            point[:, :6] = raw_point
        
        label = sources['conversations'][0]['value']
        
        if self.text_processor is not None: 
            label = self.text_processor(label)

        return {
            "object_id": object_id,
            "point": point,
            "label": label
        }
        # if self.point_indicator in sources[0]['conversations'][0]['value']:

        #     object_id = self.list_data_dict[index]['object_id']

        #     # Point cloud representation
        #     point_cloud = self._load_point_cloud(object_id) # * N, C
            
            
            

            # if self.normalize_pc:
            #     point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm

            #print( point_cloud.shape )
            #test = torch.from_numpy(point_cloud.astype(np.float32)) 
        #     if self.tokenizer is None:
        #         data_dict = dict(
        #             point_clouds= torch.from_numpy(point_cloud.astype(np.float32)),
        #             object_ids=object_id
        #         )
        #         return data_dict

        #     sources = preprocess_multimodal_point_cloud(
        #         copy.deepcopy([e["conversations"] for e in sources]), self.point_backbone_config, point_indicator=self.point_indicator)
        # else:
        #     sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess_v1(
        #     sources,
        #     self.tokenizer)

        # if isinstance(index, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0])

        # # point exist in the data
        # if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
        #     data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))

        # return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)
 
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



def load_dataset(data_path, anno_path, pts_processor, txt_processor, use_color, use_normal):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pts_processor=pts_processor, 
        text_processor=txt_processor,
        use_color=use_color,
        use_normal=use_normal,
    )
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
        # adv_pc = adv_pc.to(device)
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
    output_pc_path = args.output_pc_path
    #pointnum = args.pointnum
    #prompt_index = args.prompt_index
    #batch_size = args.batch_size
    
    
    success_rate = 0.0
    saved_results = {"prompt": PROMPT_LISTS[args.prompt_index] }
    eval_results = []
    ori_pc_list = []
    adv_pc_file_name_list = []
    adv_pc_list, adv_label_list = [], []
    
    if not os.path.exists(args.adv_anno_path):
        ori_annos = json.load(open(args.ori_anno_path, "r"))
        tgt_annos = json.load(open(args.tgt_anno_path, "r"))
        adv_annos = []
        for i in range(len(ori_annos)):
            adv_anno = tgt_annos[i].copy()
            adv_anno['object_id'] = f"{ori_annos[i]['object_id']}_{tgt_annos[i]['object_id']}"
            adv_annos.append(adv_anno)
        with open(args.adv_anno_path, "w") as f:
            json.dump(adv_annos, f, indent=4)
    
    output_file = f"Objaverse_classification_prompt_{args.prompt_index}_evaluated_{args.gpt_type}_in_training.json"
    output_eval_file = os.path.join(args.output_eval_path, output_file )
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
    elif os.path.exists(output_eval_file) and os.path.exists(output_pc_path) and \
        os.path.exists(args.adv_anno_path):
        eval_data = json.load(open(output_eval_file, "r"))
        eval_results = eval_data.get('eval_results', [])
        success_rate = eval_data.get('ASR', 0.0)
        adv_annos = json.load(open(args.adv_anno_path, "r"))
        
        for i, file in enumerate(os.listdir(output_pc_path)):
            if file.endswith('.npy'):
                adv_pc_file_name_list.append(file)
                adv_pc = np.load(os.path.join(output_pc_path, file))
                # print( adv_pc.dtype, adv_pc.shape )
                adv_pc_list.append(adv_pc)
                assert adv_annos[i]['conversations'][1]['from'] == 'gpt'
                adv_label = adv_annos[i]['conversations'][1]['value']
                adv_label_list.append(adv_label)
                
                ori_pc_name = file.split('_')[0] + f"_{args.pointnum}.npy"
                ori_pc = np.load(os.path.join(args.data_path, ori_pc_name))
                ori_pc_list.append(ori_pc)
                # print( ori_pc.dtype, ori_pc.shape )
        
        
        assert len(adv_pc_list) == len(eval_results) == len(adv_label_list), "The length of adversarial point clouds and evaluation results should be the same."
        print(f"Loaded {len(eval_results)} evaluation results")
    
    
    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)
        
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    # model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = True
        # param = param.to(torch.bfloat16) # * convert model to bfloat16

    ori_dataset = load_dataset( data_path= args.data_path, anno_path=args.ori_anno_path, 
                               pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataset = load_dataset(data_path= args.data_path, anno_path=args.tgt_anno_path,
                               pts_processor=vis_processors['eval'], text_processor = txt_processors['eval']) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)



    pc_encoder = model.get_pc_encoder()

    
    adv_func = FeatureSimilarityLoss(pc_encoder)
    
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
            ori_id = ori_data_dict['object_id']
            ori_pc = ori_data_dict['point'].to(device) # * convert to float32
            tgt_pc = tgt_data_dict['point'].to(device) # * convert to float32
            tgt_labels = tgt_data_dict['label'].to(device)
            tgt_id = tgt_data_dict['object_id']

            
            with torch.no_grad():
                tgt_feature = pc_encoder(tgt_pc[..., :6])
                if torch.isnan(tgt_feature).any():
                    print(f"NaN detected in tgt_feature. Skip this sample.")
                    continue
                tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)

                
            adv_pc, success_num, eval_result = HiT_attacker.attack( ori_pc, tgt_labels, tgt_feature )
            success_nums += success_num
            for j in range(ori_id.shape[0]):
                if i * args.batch_size + j < eval_attack_num:
                    continue
                eval_result[j].update({'object_id': f"{ori_id[j]}_{tgt_id[j]}"})
            #adv_pc = torch.from_numpy(adv_pc).to(device)
            eval_results.extend(eval_result)
            
            saved_label = tgt_data_dict['label']

            ori_pc = ori_pc.float().cpu().numpy()
            for b, (oid, op, tid, ap, adv_label) in enumerate( zip(ori_id, ori_pc, tgt_id, adv_pc, saved_label) ):
                ori_pc_list.append(op)
                adv_pc_file_name_list.append(f"{oid}_{tid}_{args.pointnum}.npy")
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
    if not os.path.exists(output_pc_path):
        os.makedirs(output_pc_path)
        
    for i, (adv_name, adv_pc) in enumerate(zip(adv_pc_file_name_list, adv_pc_list)):
        output_pc_file = os.path.join(output_pc_path, adv_name)
        np.save(output_pc_file, adv_pc)
    print(f"Saved {len(adv_pc_list)} adversarial point clouds to {output_pc_path}")
    
    if args.start_pc_eval:
        print("Starting point cloud evaluation...")
        start_pc_evaluation(ori_pc_list, adv_pc_list, adv_pc_file_name_list, args.output_eval_path, eval_results)
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
    
    