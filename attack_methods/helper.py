import os
import json
import pickle
import numpy as np
import torch
import random

from defend_methods.drop_points import SRSDefense, SORDefense
from defend_methods.DUP_Net import DUPNet

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def feature_normalize(feature):
    # feature: the feature dimension should be the last dimension
    if torch.isnan(feature).any():
        print(f"NaN detected in feature. Skip this sample.")
        return None
    feature = feature / feature.norm(dim=-1, keepdim=True)
    return feature


def load_attack_results(
            ori_data_path,
            ori_anno_path,
            tgt_anno_path,
            adv_anno_path,
            saved_eval_file,    
            saved_pc_path, 
            dataset_type, 
            is_targeted= True,
            pointnum=8192,
            ):
    
    if dataset_type == 'Objaverse' and not os.path.exists(adv_anno_path):
        ori_annos = json.load(open(ori_anno_path, "r"))
        tgt_annos = json.load(open(tgt_anno_path, "r"))
        adv_annos = []
        for i in range(len(ori_annos)):
            adv_anno = tgt_annos[i].copy()
            adv_anno['object_id'] = f"{ori_annos[i]['object_id']}_{tgt_annos[i]['object_id']}"
            adv_annos.append(adv_anno)
        with open(adv_anno_path, "w") as f:
            json.dump(adv_annos, f, indent=4)
    elif dataset_type == 'Objaverse' and os.path.exists(adv_anno_path):
        adv_annos = json.load(open(adv_anno_path, "r"))

    eval_results = []
    attack_id_list, attack_pc_list = [], []
    adv_pc_file_name_list = []
    adv_pc_list, adv_label_list = [], []
    
    eval_data = json.load(open(saved_eval_file, "r"))
    eval_results = eval_data.get('eval_results', [])
    success_rate = eval_data.get('ASR', 0.0)
    average_gpt_score = eval_data.get('average_gpt_score', 0.0)
    for eval_result in eval_results:
        attack_id_list.append(eval_result['object_id']) # * use index as object id
    if dataset_type == 'ModelNet':
        adv_pc_list, adv_label_list = pickle.load(open(saved_pc_path, "rb"))
        total_ori_pc_list, _ = pickle.load(open(ori_data_path, "rb")) 
        if is_targeted:
            attack_pc_list = [total_ori_pc_list[int(attack_id.split('_')[0])] for attack_id in attack_id_list]
        else:
            attack_pc_list = [total_ori_pc_list[int(attack_id)] for attack_id in attack_id_list] #.split('_')[0]
    elif dataset_type == 'Objaverse':
        for i, file in enumerate(os.listdir(saved_pc_path)):
            if file.endswith('.npy'):
                adv_pc_file_name_list.append(file)
                adv_pc = np.load(os.path.join(saved_pc_path, file))
                adv_pc_list.append(adv_pc)
                assert adv_annos[i]['conversations'][1]['from'] == 'gpt'
                adv_label = adv_annos[i]['conversations'][1]['value']
                adv_label_list.append(adv_label)
                
                attack_pc_name = file.split('_')[0] + f"_{pointnum}.npy"
                attack_pc = np.load(os.path.join(ori_data_path, attack_pc_name))
                attack_pc_list.append(attack_pc)
    
    return success_rate, average_gpt_score, adv_pc_list, adv_label_list, attack_id_list, attack_pc_list, eval_results


def get_defensor( method, npoint=8192):
    """Set the pre-processing based defense module.

    Args:
        method (str): defense method name.
    """
    if method == 'sor':
        defensor = SORDefense(k=2, alpha=1.1, npoint=npoint)
    elif method == 'srs':
        defensor = SRSDefense(drop_num=500)
    elif method == 'dupnet':
        defensor = DUPNet(sor_k=2, sor_alpha=1.1, npoint=npoint, up_ratio=4)
    else:
        raise NotImplementedError
    return defensor