import os
import json

import numpy as np
import random

ori_data_path = '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json'
adv_data_path = '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_3000_GT.json'
output_adv_data_path = '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json'

with open(ori_data_path, "r") as json_file:
    list_ori_data_dict = json.load(json_file)
    
with open(adv_data_path, "r") as json_file:
    list_adv_data_dict = json.load(json_file)
    
# Ensure the lengths are the same by randomly sampling from list_adv_data_dict
assert len(list_adv_data_dict) >= len(list_ori_data_dict)
sample_list_adv_data_dict = random.sample(list_adv_data_dict, len(list_ori_data_dict))
    
for ori_data, adv_data in zip(list_ori_data_dict, sample_list_adv_data_dict):
    if ori_data['object_id'] == adv_data['object_id']:
        while True:
            new_sample = random.choice(list_adv_data_dict)
            if new_sample['object_id'] != ori_data['object_id']:
                sample_list_adv_data_dict[sample_list_adv_data_dict.index(adv_data)] = new_sample
                break
            

            
with open(output_adv_data_path, "w") as json_file:
    json.dump(sample_list_adv_data_dict, json_file, indent=4)


