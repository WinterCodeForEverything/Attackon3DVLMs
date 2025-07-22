# import os
# import json
# import numpy as np
# import random
import pickle
# import shutil
import argparse



def main(args):
    
    ori_pc_file = args.ori_pc_file
    tgt_pc_file = args.tgt_pc_file
    sampled_ori_pc_file = args.sampled_ori_pc_file
    sampled_tgt_pc_file = args.sampled_tgt_pc_file
    max_samples = args.sample_num
    sample_num_per_category = args.sample_num_per_category
    
    categories = [line.rstrip() for line in open(args.pc_catfile)] 

    ori_pc_list, ori_label_list = pickle.load(open(ori_pc_file, "rb"))

    grouped_data = {}
    for i, (ori_pc, ori_label) in enumerate(zip(ori_pc_list, ori_label_list)):
        label_name = categories[ori_label.item()]
        if label_name not in grouped_data:
            grouped_data[label_name] = []
        grouped_data[label_name].append((i, ori_pc, ori_label))

    # Sort groups by the number of point clouds in decreasing order
    sorted_grouped_data = sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True)

    sampled_data = []
    total_sampled = 0
    max_samples = args.sample_num

    for label_name, items in sorted_grouped_data:
        if total_sampled >= max_samples:
            break
        if len(items) > sample_num_per_category:
            sampled_items = items[:sample_num_per_category]
        else:
            sampled_items = items
        sampled_data.extend(sampled_items)
        total_sampled += len(sampled_items)
        if total_sampled >= max_samples:
            sampled_data = sampled_data[:max_samples]
            break

    print(f"Total sampled point clouds: {len(sampled_data)}")

    sample_id_list, sampled_point_cloud_list, sampled_label_list = [], [], []
    for (id, point_cloud, label) in sampled_data:
        sample_id_list.append(id)
        sampled_point_cloud_list.append(point_cloud)
        sampled_label_list.append(label)

    # Save the sorted grouped data to a new file
    with open(sampled_ori_pc_file, "wb") as f:
        pickle.dump((sampled_point_cloud_list, sampled_label_list), f)
        print(f"Saved grouped and sorted data to {sampled_ori_pc_file}")

    tgt_pc_list, tgt_label_list = pickle.load(open(tgt_pc_file, "rb"))
    # Select target point clouds and labels according to sample_id_list
    target_point_cloud_list = [tgt_pc_list[i] for i in sample_id_list]
    target_label_list = [tgt_label_list[i] for i in sample_id_list]

    # Save the target point clouds and labels
    with open(sampled_tgt_pc_file, "wb") as f:
        pickle.dump((target_point_cloud_list, target_label_list), f)
        print(f"Saved target point clouds and labels to {sampled_tgt_pc_file}")


if __name__ == '__main__':
    
    # gpt_eval_file = '/home/liuchao/PointLLM2/objaverse_result/objaverse_eval/PointLLM_brief_description_val_3000_GT_Objaverse_classification_prompt0_evaluated_gpt-4o-mini.json'
    # input_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_gpt4point_ori_8192pts.dat"
    # ori_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_gpt4point_ori_8192pts.dat"
    # tgt_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_gpt4point_tgt_8192pts.dat"
    # sampled_ori_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_gpt4point_ori_8192pts_sampled.dat"
    # sampled_tgt_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_gpt4point_tgt_8192pts_sampled.dat"
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_catfile', type=str, default='/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt',
                        help='Path to the point cloud category file.')
    parser.add_argument('--ori_pc_file', type=str, required=True, help='Path to original point cloud file')
    parser.add_argument('--tgt_pc_file', type=str, required=True, help='Path to target point cloud file')
    parser.add_argument('--sampled_ori_pc_file', type=str, required=True, help='Path to save sampled original point cloud file')
    parser.add_argument('--sampled_tgt_pc_file', type=str, required=True, help='Path to save sampled target point cloud file')
    parser.add_argument('--sample_num', type=int, required=True, help='Maximum number of samples')
    parser.add_argument('--sample_num_per_category', type=int, required=True, help='Maximum samples per category')

    args = parser.parse_args()
    main(args)