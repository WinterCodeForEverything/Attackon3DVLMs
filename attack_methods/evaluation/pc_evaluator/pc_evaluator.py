import os
import logging
import torch
import math
from typing import List
from attack_methods.util import dist_utils
from tqdm import tqdm
import numpy as np
from datetime import datetime
import open3d as o3d
from attack_methods.FGM.GeoA3_args import uniform_loss
from pytorch3d.loss import chamfer_distance
import json

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PointCloudEvaluator:
    def __init__(self, k_nn=4, device='cuda'):
        
        #metrics in HiT_Adv
        self.k_nn = k_nn
        self.knn_dist_metric = dist_utils.KNNDist(k=k_nn).to(device)
        #self.uniform_dist_metric = uniform_loss
        self.curv_std_metric = dist_utils.CurvStdDist(k=k_nn).to(device)
        
    def evaluate(self, ori_pc, adv_pc, ori_normal ):
        
        
        knn_dist = self.knn_dist_metric.forward(pc=adv_pc, weights=None, batch_avg=True)
        uniform_dist = uniform_loss(adv_pc=adv_pc, k=self.k_nn)
        curv_std_dist = self.curv_std_metric.forward(ori_data=ori_pc, adv_data=adv_pc, ori_normal=ori_normal)

        
        adv_data_normalize = adv_pc.cpu().numpy().copy()
        for j in range(adv_pc.shape[0]):
            adv_data_normalize[j] = pc_normalize(adv_data_normalize[j])
        adv_data_normalize = torch.Tensor(adv_data_normalize).cuda()
        chamfer_dist, _ = chamfer_distance(ori_pc, adv_pc)
        hausdorff_dist, _ = chamfer_distance(ori_pc, adv_pc,  point_reduction='max')
        # hausdorff_dist = torch.max(
        #     torch.max(torch.min(torch.cdist(ori_pc, adv_pc), dim=1).values),
        #     torch.max(torch.min(torch.cdist(adv_pc, ori_pc), dim=1).values)
        # )
        
        return {
            'knn_dist': knn_dist,
            'uniform_dist': uniform_dist,
            'curv_std_dist': curv_std_dist,
            'chamfer_dist': chamfer_dist,
            'hausdorff_dist': hausdorff_dist
        }
    
 
def start_pc_evaluation(ori_pc_list, adv_pc_list, obj_id_list, output_eval_dir, device, evaluated_results = None):
    
    pc_evaluator = PointCloudEvaluator(k_nn=32, device=device)
    
    
    results = []
    overall_results = {
            'knn_dist': [],
            'uniform_dist': [],
            'curv_std_dist': [],
            'chamfer_dist': [],
            'hausdorff_dist': []
    }
    
    statistic_object_ids = []
    statistic_ori_xyz, statistic_ori_normal, statistic_adv_xyz = [], [], []
    for i, (obj_id, ori_pc, adv_pc ) in enumerate(zip(obj_id_list, ori_pc_list, adv_pc_list)):
        # ori_pc = ori_data['point_clouds'].to(device)
        # adv_pc = adv_data['point_clouds'].to(device)
        
        # obj_ids = adv_data['object_ids']
        
        ori_xyz = ori_pc[..., :3]
        ori_normal = ori_pc[..., 6:9]
        adv_xyz = adv_pc[..., :3]
        
        if evaluated_results is not None:
            if evaluated_results[i]['accuracy'] == 1:
                    statistic_object_ids.append(obj_id)
                    statistic_ori_xyz.append(torch.from_numpy(ori_xyz).to(device, dtype=torch.float32))
                    statistic_adv_xyz.append(torch.from_numpy(adv_xyz).to(device, dtype=torch.float32))
                    statistic_ori_normal.append(torch.from_numpy(ori_normal).to(device, dtype=torch.float32))
        else:
            statistic_object_ids.append(obj_id)
            statistic_ori_xyz.append(torch.from_numpy(ori_xyz).to(device, dtype=torch.float32))
            statistic_adv_xyz.append(torch.from_numpy(adv_xyz).to(device))
            statistic_ori_normal.append(torch.from_numpy(ori_normal).to(device, dtype=torch.float32))
    # print( "len(statistic_object_ids):", len(statistic_object_ids))
    for j in range(len(statistic_ori_xyz)):
        obj_id = statistic_object_ids[j]
        ori_xyz = statistic_ori_xyz[j].unsqueeze(0)
        adv_xyz = statistic_adv_xyz[j].unsqueeze(0)
        ori_normal = statistic_ori_normal[j].unsqueeze(0)
        pc_result = pc_evaluator.evaluate(ori_xyz, adv_xyz, ori_normal)
        
        pc_result =  {metric: float(value.cpu().numpy()) for metric, value in pc_result.items()}
        for metrics, values in pc_result.items():
            overall_results[metrics].append( values )

        pc_result['obj_id'] = obj_id
        results.append(pc_result)
        
    print("re-evaluated ASR:", len(statistic_adv_xyz) / len(adv_pc_list) )
        
    average_results = {}
    for metrics, values in overall_results.items():
        average_results[metrics] = float(np.mean((values)))
        
    print(f"Average knn_dist: {average_results['knn_dist']}")
    print(f"Average uniform_dist: {average_results['uniform_dist']}")
    print(f"Average curv_std_dist: {average_results['curv_std_dist']}")
    print(f"Average chamfer_dist: {average_results['chamfer_dist']}")
    print(f"Average hausdorff_dist: {average_results['hausdorff_dist']}")
    
    # * save the results
    save_results = {
        'overall_results': average_results,
        'results': results
    }
    os.makedirs(output_eval_dir, exist_ok=True)
    with open(os.path.join(output_eval_dir, 'pc_evaluation_results_recall_in_training.json'), 'w') as fp:
        json.dump(save_results, fp, indent=4)

   

