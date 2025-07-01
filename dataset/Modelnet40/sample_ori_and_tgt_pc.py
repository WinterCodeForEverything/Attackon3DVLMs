import os
import json
import numpy as np
import random
import pickle
import shutil

pc_catfile = "/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"

categories = [line.rstrip() for line in open(pc_catfile)] 


if __name__ == '__main__':
    
    # gpt_eval_file = '/home/liuchao/PointLLM2/objaverse_result/objaverse_eval/PointLLM_brief_description_val_3000_GT_Objaverse_classification_prompt0_evaluated_gpt-4o-mini.json'
    input_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_test_8192pts_gpt4point_cp.dat"
    ori_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_ori_8192pts_200.dat"
    tgt_pc_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_tgt_8192pts_200.dat"
    point_cloud_list, label_list = pickle.load(open(input_file, "rb"))
    
    grouped_data = {}
    for point_cloud, label in zip(point_cloud_list, label_list):
        label_name = categories[label.item()]
        if label_name not in grouped_data:
            grouped_data[label_name] = []
        grouped_data[label_name].append((point_cloud, label))

    # Sort groups by the number of point clouds in decreasing order
    sorted_grouped_data = sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True)

    sampled_data = []
    total_sampled = 0
    max_samples = 200

    for label_name, items in sorted_grouped_data:
        if total_sampled >= max_samples:
            break
        if len(items) > 10:
            sampled_items = items[:10]
        else:
            sampled_items = items
        sampled_data.extend(sampled_items)
        total_sampled += len(sampled_items)
        if total_sampled >= max_samples:
            sampled_data = sampled_data[:max_samples]
            break
        
    print(f"Total sampled point clouds: {len(sampled_data)}")

    sampled_point_cloud_list, sampled_label_list = [], []
    for (point_cloud, label) in sampled_data:
        sampled_point_cloud_list.append(point_cloud)
        sampled_label_list.append(label)
        
    # Save the sorted grouped data to a new file
    with open(ori_pc_file, "wb") as f:
        pickle.dump((sampled_point_cloud_list, sampled_label_list ), f)
        print(f"Saved grouped and sorted data to {ori_pc_file}")
        
    target_point_cloud_list, target_label_list = [], []
    while len(target_point_cloud_list) < len(sampled_point_cloud_list):
        idx = random.randint(0, len(point_cloud_list) - 1)
        target_point_cloud = point_cloud_list[idx]
        target_label = label_list[idx]

        # Ensure the sampled point cloud and target point cloud have different labels
        sampled_label = sampled_label_list[len(target_point_cloud_list)]
        if target_label != sampled_label:
            target_point_cloud_list.append(target_point_cloud)
            target_label_list.append(target_label)

    print(f"Total target point clouds sampled: {len(target_point_cloud_list)}")

    # Save the target point clouds and labels
    with open(tgt_pc_file, "wb") as f:
        pickle.dump((target_point_cloud_list, target_label_list), f)
        print(f"Saved target point clouds and labels to {tgt_pc_file}")


    
    
