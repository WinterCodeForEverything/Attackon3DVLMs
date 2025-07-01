import os
import json
import numpy as np
import random
import pickle


if __name__ == '__main__':
    gpt_eval_file = '/home/liuchao/GPT4Point/output/GPT4Point/modelnet40_opt2.7b_eval/eval_result.json'
    input_file = "/mnt/ssd/liuchao/PointLLM/modelnet40_test_8192pts_fps.dat"
    output_file = "/mnt/ssd/liuchao/Attackon3DVLMs/modelnet40_test_8192pts_gpt4point_cp.dat"
    output_eval_path = "/home/liuchao/Attackon3DVLM/modelnet40_results/modelnet40_test_8192pts_gpt4point_cp"
    
    if not os.path.exists(output_eval_path):
        os.makedirs(output_eval_path)
        
    sample_num = 20000
    
    gpt_eval_results = json.load(open(gpt_eval_file, "r"))
    gpt_eval_results = gpt_eval_results['eval_results']
    
    chosen_index = []
    chosen_gpt_eval_results = []
    for item in gpt_eval_results:
        if item['accuracy'] == 1:
            chosen_index.append(item['pcd_id'])
            chosen_gpt_eval_results.append(item)
            
    print(f"Chosen index: {len(chosen_index)}")


    if len(chosen_index) > sample_num:
        inter_ = len(chosen_index) / sample_num
        sampled_chosen_index = [chosen_index[int(i * inter_)] for i in range(sample_num)]
    else:
        sampled_chosen_index = chosen_index
    ori_pc_list, ori_label_list = pickle.load(open(input_file, "rb"))

    sampled_pc_list, sample_label_list = [], []
    for index in sampled_chosen_index:
        sampled_pc_list.append(ori_pc_list[index])
        sample_label_list.append(ori_label_list[index])
        
    for i, sampled_pc in enumerate(sampled_pc_list):
        if sampled_pc.shape[-1] == 6:
            sampled_pc_list[i] = np.concatenate((sampled_pc[:, :3], np.zeros_like(sampled_pc[:, :3]), sampled_pc[:, 3:]), axis=-1)
        
    # * Save the sampled point clouds and labels
    with open(output_file, "wb") as f:
        pickle.dump((sampled_pc_list, sample_label_list), f)
        
    # Save chosen gpt eval results
    with open(os.path.join(output_eval_path, "chosen_gpt_eval_results.json"), "w") as f:
        json.dump(chosen_gpt_eval_results, f, indent=2)
            
    
            
    
