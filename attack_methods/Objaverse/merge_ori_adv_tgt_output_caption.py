import os
import json

import numpy as np
#import shutil
#import random

ori_data_path = '/mnt/ssd/liuchao/PointLLM/Objaverse_npy'
adv_data_path = '/mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_epsilon_0.05'
ori_output_cap = './result/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json'
adv_output_cap = './result/noise_size_5_result/PointLLM_brief_description_adv_200_Objaverse_classification_prompt0.json'
tgt_output_cap = './result/PointLLM_brief_description_tgt_200_Objaverse_classification_prompt0.json'
output_pc = './result/pointcloud'
output_merge_cap = './result/noise_size_5_result/PointLLM_brief_description_mrg_200_Objaverse_classification_prompt0.json'

with open(ori_output_cap, "r") as json_file:
    list_ori_cap_dict = json.load(json_file)

with open(adv_output_cap, "r") as json_file:
    list_adv_cap_dict = json.load(json_file)
    
with open(tgt_output_cap, "r") as json_file:
    list_tgt_cap_dict = json.load(json_file)
    
assert len(list_ori_cap_dict) == len(list_adv_cap_dict) == len(list_tgt_cap_dict)

output_dict = {
    'prompt': list_ori_cap_dict['prompt']
}
results = []
for ori_cap, adv_cap, tgt_cap in zip(list_ori_cap_dict['results'], list_adv_cap_dict['results'], list_tgt_cap_dict['results'] ):
    assert adv_cap['object_id'] == tgt_cap['object_id']
    result = {
        'object_id': ori_cap['object_id'],
        'ori_gt': ori_cap['ground_truth'],
        'tgt_gt': adv_cap['ground_truth'],
        'ori_caption': ori_cap['model_output'],
        'adv_caption': adv_cap['model_output'],
        'tgt_caption': tgt_cap['model_output']
    }
    results.append(result)
    output_pc_path = os.path.join(output_pc, f"{ori_cap['object_id']}_8192")
    
    # if os.path.exists(output_pc_path):
    #     continue
    #os.makedirs(output_pc_path)
    
    # Load the .npy file and save it as a .txt file
    # ori_pc = np.load(os.path.join(ori_data_path, f"{ori_cap['object_id']}_8192.npy"))
    # np.savetxt(os.path.join(output_pc_path, "ori.txt"), ori_pc)

    adv_pc = np.load(os.path.join(adv_data_path, f"{adv_cap['object_id']}_8192.npy"))
    np.savetxt(os.path.join(output_pc_path, "adv_0.05.txt"), adv_pc)
    
    # tgt_pc = np.load(os.path.join(ori_data_path, f"{tgt_cap['object_id']}_8192.npy"))
    # np.savetxt(os.path.join(output_pc_path, "tgt.txt"), tgt_pc)    
    
    # shutil.copy(
    #     os.path.join(ori_data_path, f"{ori_cap['object_id']}_8192.npy"),
    #     os.path.join(output_pc_path, "ori.npy")
    # )
    # shutil.copy(
    #     os.path.join(adv_data_path, f"{adv_cap['object_id']}_8192.npy"),
    #     os.path.join(output_pc_path, "adv.npy")
    # )
    # shutil.copy(
    #     os.path.join(ori_data_path, f"{tgt_cap['object_id']}_8192.npy"),
    #     os.path.join(output_pc_path, "tgt.npy")
    # )

    output_dict['results'] = results        

            
with open(output_merge_cap, "w") as json_file:
    json.dump(output_dict, json_file, indent=4)


