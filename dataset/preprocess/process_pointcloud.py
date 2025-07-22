import os
import json
import numpy as np
import random
import pickle
import shutil
import argparse
from tqdm import tqdm

from dataset.data.modelnet import ModelNet
from dataset.data.objaverse import ObjaverseDataset

# pc_catfile = "/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"

# categories = [line.rstrip() for line in open(pc_catfile)] 

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

def main(args):
    
    input_dataset = load_dataset( dataset_type = args.dataset_type, data_path= args.input_path, anno_path= args.anno_path,
                                 normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                    redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal )
    print(f"Loaded {len(input_dataset)} point clouds from {args.input_path}.")
    
    ori_pc_path = args.ori_pc_path
    tgt_pc_path = args.tgt_pc_path
    
    if args.dataset_type == 'Objaverse':
        anno_dict = {}
        input_anno = json.load(open(args.anno_path, "r"))
        for item in input_anno:
            object_id = item['object_id']
            anno_dict[object_id] = item
    
    ori_id_list, ori_pc_list, ori_label_list = [], [], []
    for i, data in tqdm(enumerate(input_dataset)):
        id = data['object_id']
        pc = data['point_cloud']
        label = data['label']
        ori_id_list.append(id)
        ori_pc_list.append(pc)
        ori_label_list.append(label)
        
    if args.dataset_type == 'ModelNet':
        with open(ori_pc_path, "wb") as f:
            pickle.dump((ori_pc_list, ori_label_list), f)      
    elif args.dataset_type == 'Objaverse':
        if not os.path.exists(ori_pc_path):
            os.makedirs(ori_pc_path)
        ori_anno_list = []
        for i in range(len(ori_label_list)):
            file_name = f"{ori_id_list[i]}_{args.pointnum}.npy"
            output_file = os.path.join(ori_pc_path, file_name)
            np.save(output_file, ori_pc_list[i])
            ori_anno_list.append(anno_dict[ori_id_list[i]])
           
        with open(os.path.join(args.ori_anno_path), "w") as f:
            json.dump(ori_anno_list, f, indent=4)
        
        print(f"Saved {len(ori_pc_list)} original point clouds to {args.ori_pc_path}.") 
        
            
    
    tgt_id_list, tgt_pc_list, tgt_label_list = [], [], []      
    for i, ori_pc in tqdm(enumerate(ori_pc_list)):
        # Randomly choose a point cloud with a different label
        if args.dataset_type == 'ModelNet':
            while True:
                rand_idx = random.randint(0, len(ori_pc_list) - 1)
                if ori_label_list[rand_idx].item() != ori_label_list[i].item():
                    tgt_id_list.append(ori_id_list[rand_idx])
                    tgt_pc_list.append(ori_pc_list[rand_idx])
                    tgt_label_list.append(ori_label_list[rand_idx])
                    break
        elif args.dataset_type == 'Objaverse':
            ori_id = ori_id_list[i]
            if args.train_anno_path is not None:
                with open(args.train_anno_path, "r") as json_file:
                    target_choose_data_list = json.load(json_file)
            else:
                with open(args.anno_path, "r") as json_file:
                    target_choose_data_list = json.load(json_file)
            while True:
                rand_idx = random.randint(0, len(target_choose_data_list) - 1)
                target_obj_id = target_choose_data_list[rand_idx]['object_id']
                if target_obj_id != ori_id:  # Ensure different point clouds
                    # shutil.copy(os.path.join(input_path, f"{ori_id}_{args.pointnum}.npy"),
                    #             os.path.join(tgt_pc_path, f"{target_obj_id}_{args.pointnum}.npy"))
                    tgt_label_list.append(target_choose_data_list[rand_idx])
                    break


    if args.dataset_type == 'ModelNet':
        with open(tgt_pc_path, "wb") as f:
            pickle.dump((tgt_pc_list, tgt_label_list), f)
    elif args.dataset_type == 'Objaverse':
        if not os.path.exists(tgt_pc_path):
            os.makedirs(tgt_pc_path)
        with open(args.tgt_anno_path, "w") as json_file:
            json.dump( tgt_label_list, json_file )
        processed_tgt_dataset = load_dataset( dataset_type = args.dataset_type, data_path= args.input_path, anno_path= args.tgt_anno_path,
                                normalized_pc=args.normalized_pc, has_color=args.has_color, has_normal=args.has_normal,
                                redirect_normals=args.redirect_normal, use_color=args.use_color, use_normal=args.use_normal )
        
        for i, data in tqdm(enumerate(processed_tgt_dataset)):
            pc = data['point_cloud']
            tgt_pc_list.append(pc)
            np.save(os.path.join(args.tgt_pc_path, f"{data['object_id']}_{args.pointnum}.npy"), pc)
        print(f"Saved {len(tgt_pc_list)} target point clouds to {args.tgt_pc_path}.")

    assert len(ori_pc_list) == len(tgt_pc_list), "The number of original and target point clouds must be the same."
       




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, required=True, choices=['ModelNet', 'Objaverse'],
                        help='Type of dataset to process.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input dataset.')
    parser.add_argument('--anno_path', type=str, default=None,
                        help='Path to the annotation file (for Objaverse).')
    parser.add_argument('--train_anno_path', type=str, default=None,
                        help='Path to the training annotation file (for Objaverse).')
    parser.add_argument('--ori_pc_path', type=str, required=True,
                        help='Path to save original point clouds.')
    parser.add_argument('--ori_anno_path', type=str, default=None,
                        help='Path to the original annotation file (for Objaverse).')
    parser.add_argument('--tgt_pc_path', type=str, required=True,
                        help='Path to save target point clouds.')
    parser.add_argument('--tgt_anno_path', type=str, default=None,
                        help='Path to the target annotation file (for Objaverse).')
    parser.add_argument('--normalized_pc', action='store_true',  default=False,
                        help='Whether to normalize point clouds.')
    parser.add_argument('--has_color', action='store_true',  default=False,
                        help='Whether point clouds have color information.')
    parser.add_argument('--has_normal', action='store_true',  default=False,  
                        help='Whether point clouds have normal information.')
    parser.add_argument('--redirect_normal', action='store_true',  default=False,
                        help='Whether to redirect normals.')
    parser.add_argument('--use_color', action='store_true', default=False,
                        help='Whether to use color information.')
    parser.add_argument('--use_normal', action='store_true', default=False,
                        help='Whether to use normal information.')
    parser.add_argument('--pointnum', type=int, default=8192,
                        help='Number of points per point cloud.')
    args = parser.parse_args()
    main(args)
    
    
    
    