# import time
import torch
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import wandb


class Native_Attacker:
    def __init__(self,  adv_func, 
                        evaluator, 
                        perturbate_object = 'pc',
                        pc_perturbation_budget = 0.55, 
                        attack_lr = 0.01, 
                        attack_iters=300, 
                        use_wandb = False):
        # self.model = model
        self.adv_func = adv_func
        self.evaluator = evaluator
        self.perturbate_object = perturbate_object
        self.pc_perturbation_budget = pc_perturbation_budget
        self.attack_lr = attack_lr
        self.attack_iters = attack_iters
        # self.perturbate_pc_color = perturbate_pc_color
        self.use_wandb = use_wandb

    def attack( self, ori_pc,  target, ori_feature = None, tgt_feature = None ):

        # self.model.eval()
        ori_adv_pc, ori_adv_feature = None, None
        if self.perturbate_object == 'pc':
            ori_adv_pc = ori_pc.clone().detach()
            ori_adv_pc.requires_grad_(True)

            perturbation_noise = torch.zeros_like(ori_adv_pc)
            perturbation_noise.requires_grad_(True)
            optimizer = optim.Adam([{'params': perturbation_noise, 'lr': self.attack_lr}])
        elif self.perturbate_object == 'feature':
            ori_adv_feature = ori_feature.clone().detach()
            ori_adv_feature.requires_grad_(True)

            perturbation_noise = torch.zeros_like(ori_adv_feature)
            perturbation_noise.requires_grad_(True)
            optimizer = optim.Adam([{'params': perturbation_noise, 'lr': self.attack_lr}])

        eval_adv_pc, eval_adv_feature = None, None
        success_attack, eval_results = None, None
        for iter in tqdm(range(self.attack_iters)):

            adv_pc = ori_adv_pc + perturbation_noise if ori_adv_pc is not None else None
            adv_feature = ori_adv_feature + perturbation_noise if ori_adv_feature is not None else None
            # with torch.no_grad():
            #     # xyz from -1.0 to 1.0, rgb from 0.0 to 1.0
            #     adv_pc.data[..., :3] = torch.clamp(adv_pc.data[..., :3], -1.0, 1.0)
            #     adv_pc.data[..., 3:] = torch.clamp(adv_pc.data[..., 3:], 0.0, 1.0)
            # # print(adv_pc.min().item(), adv_pc.max().item(), adv_pc.mean().item())
        
            
            if tgt_feature is not None:
                loss = self.adv_func(adv_pc, adv_feature, tgt_feature) 
            else:
                loss = self.adv_func(adv_pc, adv_feature, target)
            
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # set the pertubation noise to be within the budget
            with torch.no_grad():
                if self.perturbate_object == 'pc':
                    perturbation_noise.data = torch.clamp(perturbation_noise.data, -self.pc_perturbation_budget, self.pc_perturbation_budget)
            # if  (iter+1) % 100 == 0:        #iter != 0 and iter % ( self.attack_iters -1 ) == 0:
            #     eval_adv_pc = adv_pc.detach() if adv_pc is not None else None
            #     eval_adv_feature = adv_feature.detach() if adv_feature is not None else None
            #     with torch.no_grad():
            #         success_attack, eval_results = self.evaluator(eval_adv_pc, eval_adv_feature, target)

            #     success_num = np.sum(success_attack) if success_attack is not None else 0
            if self.use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "max_perturbation": perturbation_noise.abs().max().item(),
                    "mean_perturbation": perturbation_noise.abs().mean().item(),
                })

            # # Check if all values in eval_accuracy are 1 (for np.array of int)
            # all_success = success_attack is not None and np.all(success_attack == 1)
            # if (iter+1) % 100 == 0 and all_success:
            #     break
        eval_adv_pc = adv_pc.detach() if adv_pc is not None else None
        eval_adv_feature = adv_feature.detach() if adv_feature is not None else None
        with torch.no_grad():
            success_attack, eval_results = self.evaluator(eval_adv_pc, eval_adv_feature, target)

        success_num = np.sum(success_attack) if success_attack is not None else 0
        return eval_adv_pc, eval_adv_feature, success_num, eval_results
