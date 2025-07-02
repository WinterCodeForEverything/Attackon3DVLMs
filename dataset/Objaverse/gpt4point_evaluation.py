import argparse
import json
import os
import random
random.seed(0)

import nltk
nltk.download('wordnet')
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge import Rouge
# #from sentence_transformers import SentenceTransformer, util
# from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from victim_models.gpt4point.models import load_model_and_preprocess



# from pointllm.model import *
# from pointllm.model.utils import KeywordsStoppingCriteria
# from pointllm.utils import disable_torch_init
# from pointllm.data import ObjectPointCloudDataset
# from pointllm.conversation import conv_templates, SeparatorStyle

# from pointllm.eval.evaluator import start_evaluation

from attack_methods.Objaverse.evaluate import Evaluation

# from pytorch3d.loss import chamfer_distance


device = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]

class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 pts_processor=None,
                 text_processor=None,
                #  tokenizer=None,
                pointnum=8192,
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

        self.pointnum = pointnum
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
        else:
            raw_point = point

        # point transform
        if self.pts_processor is not None:
            raw_point = self.pts_processor(raw_point)
            point[:, :6] = raw_point
        
        assert sources['conversations'][1]['from'] == 'gpt'
        label = sources['conversations'][1]['value']
        
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
 

# def init_model(model_name):
#     # Model
#     disable_torch_init()
#     model_name = os.path.expanduser(model_name)

#     # * print the model_name (get the basename)
#     print(f'[INFO] Model name: {os.path.basename(model_name)}')

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).to(device)
#     model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

#     conv_mode = "vicuna_v1_1"

#     conv = conv_templates[conv_mode].copy()

#     return model, tokenizer, conv




def load_dataset(data_path, anno_path, pts_processor, txt_processor, use_color=True, use_normal=True):
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



def main(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # * output file 
    output_file = f"GPT4Point_Objaverse_classification_prompt{args.prompt_index}.json"
    output_file_path = os.path.join( args.output_dir, output_file)

    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)

    generater = ModelGeneration(model, args)
    
    evaluator = Evaluation(generater, args.gpt_type)

    dataset = load_dataset( data_path=args.data_path, anno_path=args.anno_path, 
                               pts_processor=vis_processors['eval'], txt_processor = txt_processors['eval']) # * defalut config
    dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
    
    
    saved_results = []
    average_accuracy = 0.0
    
    if os.path.exists(output_file_path):
        print(f"[INFO] {output_file_path} already exists, loading the results...")
        with open(output_file_path, 'r') as fp:
            saved = json.load(fp)
            saved_results = saved['results']
            average_accuracy = saved['accuracy'] * len(saved_results)# * total accuracy
            print(f"[INFO] Loaded {len(saved_results)} results with correct preditiction: {average_accuracy:.4f}")

    
    try:
        for i, data_dict in tqdm(enumerate(dataloader)):
            if i* args.batch_size < len(saved_results):
                # * if the result already exists, skip it
                continue
            obj_id = data_dict['object_id']
            obj_pc = data_dict['point'].to(device) # * convert to float32
            obj_label = data_dict['label']

            
            accuracies, results = evaluator( obj_pc, obj_label )
            
            for j in range(obj_pc.shape[0]):
                if i* args.batch_size + j < len(saved_results):
                    # * if the result already exists, skip it
                    continue
                result = results[j].copy()
                result['object_id'] = obj_id[j]
                saved_results.append(result)
                average_accuracy += accuracies[j]
    except Exception as e:
        print(f"[ERROR] An error occurred during evaluation: {e}")   
    
    average_accuracy /= len(saved_results)
    
    saved = {
        'accuracy': average_accuracy,
        'results': saved_results
    }
    
    # * save the results
    with open(output_file_path, 'w') as fp:
        json.dump(saved, fp, indent=4)
    
            

    
    
    
    # # * First inferencing, then evaluate
    # if not os.path.exists(args.output_file_path):

    #     dataset = load_dataset(data_path= args.adv_data_path, anno_path=args.adv_anno_path, 
    #                             use_color= args.use_color, use_normal=args.use_normal )
    #     dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)

    #     #annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

    #     print(f'[INFO] Start generating results for {args.output_file}.')
    #     results = start_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file, args )

    #     # * release model and tokenizer, and release cuda memory
    #     # del model
    #     del tokenizer
    #     torch.cuda.empty_cache()
    # else:
    #     # * directly load the results
    #     print(f'[INFO] {args.output_file_path} already exists, directly loading...')
    #     with open(args.output_file_path, 'r') as fp:
    #         results = json.load(fp)

    # evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    # if args.start_caption_eval:
    #     start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type= args.caption_eval_type, model_type=args.gpt_type, parallel=True, num_workers=20)
        
    # if args.start_pc_eval:
    #     start_pc_evaluation( args, model, evaluated_output_file )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name", type=str, \
        default="gpt4point_opt") 
    parser.add_argument("--model_type", type=str, \
        default="gpt4point_opt2.7b") 
    parser.add_argument("--do_sample",  action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    # * dataset type
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default='', required=True)
    parser.add_argument("--anno_path", type=str, default="", required=True)
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