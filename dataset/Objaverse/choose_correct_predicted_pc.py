import os
import json
# import numpy as np
# import random
# import pickle
import shutil


if __name__ == '__main__':
    gpt_eval_file = "/mnt/ssd/liuchao/Attackon3DVLMs/GPT4Point_Objaverse_classification_prompt0.json"
    input_path = "/mnt/ssd/liuchao/PointLLM/Objaverse_npy"
    output_path = "/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_npy_gpt4point_cp"
    output_eval_file = "/mnt/ssd/liuchao/Attackon3DVLMs/GPT4Point_Objaverse_classification_prompt0_correct_prediction.json"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    sample_num = 20000
    
    gpt_eval_results = json.load(open(gpt_eval_file, "r"))
    gpt_eval_results = gpt_eval_results['results']
    
    chosen_index = []
    chosen_gpt_eval_results = []
    for item in gpt_eval_results:
        if item['cls_result'] == 'T':
            chosen_index.append(item['object_id'])
            chosen_gpt_eval_results.append(item)
            
    print(f"Chosen index: {len(chosen_index)}")
    
    for index in chosen_index:
        input_file = os.path.join(input_path, f"{index}_8192.npy")
        output_file = os.path.join(output_path, f"{index}_8192.npy")
        
        if not os.path.exists(output_file):
            if not os.path.exists(input_file):
                print(f"File {input_file} does not exist, skipping...")
                continue
            
            # Copy the file from input_file to output_file without opening files manually
            shutil.copy2(input_file, output_file)
            
            # # Load the point cloud
            # ori_pc = np.load(input_file)
            
            # # If the point cloud has 6 channels, convert it to 3 channels with normals
            # if ori_pc.shape[-1] == 6:
            #     ori_pc = np.concatenate((ori_pc[:, :3], np.zeros_like(ori_pc[:, :3]), ori_pc[:, 3:]), axis=-1)
                
            # # Save the point cloud
            # np.save(output_file, ori_pc)


    # if len(chosen_index) > sample_num:
    #     inter_ = len(chosen_index) / sample_num
    #     sampled_chosen_index = [chosen_index[int(i * inter_)] for i in range(sample_num)]
    # else:
    #     sampled_chosen_index = chosen_index
        
    
        
     
        
    # ori_pc_list, ori_label_list = pickle.load(open(input_file, "rb"))

    # sampled_pc_list, sample_label_list = [], []
    # for index in sampled_chosen_index:
    #     sampled_pc_list.append(ori_pc_list[index])
    #     sample_label_list.append(ori_label_list[index])
        
    # for i, sampled_pc in enumerate(sampled_pc_list):
    #     if sampled_pc.shape[-1] == 6:
    #         sampled_pc_list[i] = np.concatenate((sampled_pc[:, :3], np.zeros_like(sampled_pc[:, :3]), sampled_pc[:, 3:]), axis=-1)
        
    # # * Save the sampled point clouds and labels
    # with open(output_file, "wb") as f:
    #     pickle.dump((sampled_pc_list, sample_label_list), f)
        
    # Save chosen gpt eval results
    with open(output_eval_file, "w") as f:
        json.dump(chosen_gpt_eval_results, f, indent=2)
            
    
            
    
