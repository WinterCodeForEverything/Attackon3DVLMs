import os
import json
import numpy as np
import random
import pickle
import shutil


if __name__ == '__main__':
    
    gpt_eval_file = "/mnt/ssd/liuchao/Attackon3DVLMs/GPT4Point_Objaverse_classification_prompt0_correct_prediction.json"
    input_path = "/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_npy_gpt4point_cp_normals"
    ori_pc_path = "/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_npy_gpt4point_cp_normals_ori"
    tgt_pc_path = "/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_npy_gpt4point_cp_normals_tgt"
    
    desciption_file = '/mnt/ssd/liuchao/Attackon3DVLMs/PointLLM_brief_description_val_3000_GT.json'
    
    gpt_eval_results = json.load(open(gpt_eval_file, "r"))
    # gpt_eval_results = gpt_eval_results['results']
    
    desciption_objaverse = json.load(open(desciption_file, "r"))
    desciption_objaverse_dict = {item['object_id']: item for item in desciption_objaverse}
    
    
    chosen_index = []
    chosen_gpt_eval_results = []
    for item in gpt_eval_results:
        if item['cls_result'] == 'T':
            chosen_index.append(item['object_id'])
            chosen_gpt_eval_results.append(item)
    
    if not os.path.exists(ori_pc_path):
        os.makedirs(ori_pc_path)
    if not os.path.exists(tgt_pc_path):
        os.makedirs(tgt_pc_path)
        
    sample_num = len(chosen_index) // 2
    ori_index = random.sample(chosen_index, sample_num)
    remaining_index = list(set(chosen_index) - set(ori_index))
    tgt_index = random.sample(remaining_index, sample_num)
    
    ori_desciption = []
    for index in ori_index:
        ori_desciption.append(desciption_objaverse_dict[index])
        
        input_file = os.path.join(input_path, f"{index}_8192.npy")
        output_file = os.path.join(ori_pc_path, f"{index}_8192.npy")
        
        if not os.path.exists(output_file):
            if not os.path.exists(input_file):
                print(f"File {input_file} does not exist, skipping...")
                continue
            
            # Copy the file from input_file to output_file without opening files manually
            shutil.copy2(input_file, output_file)
    
    ori_desciption_file = f'/mnt/ssd/liuchao/Attackon3DVLMs/GPT4Point_brief_description_ori_{sample_num}_GT.json'
    json.dump(ori_desciption, open(ori_desciption_file, "w"), indent=2)
    
    tgt_desciption = []
    for index in tgt_index:
        tgt_desciption.append(desciption_objaverse_dict[index])
        
        input_file = os.path.join(input_path, f"{index}_8192.npy")
        output_file = os.path.join(tgt_pc_path, f"{index}_8192.npy")
        
        if not os.path.exists(output_file):
            if not os.path.exists(input_file):
                print(f"File {input_file} does not exist, skipping...")
                continue
            
            # Copy the file from input_file to output_file without opening files manually
            shutil.copy2(input_file, output_file)
    
    tgt_desciption_file = f'/mnt/ssd/liuchao/Attackon3DVLMs/GPT4Point_brief_description_tgt_{sample_num}_GT.json'
    json.dump(tgt_desciption, open(tgt_desciption_file, "w"), indent=2)