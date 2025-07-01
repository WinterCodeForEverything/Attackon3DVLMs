#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import os
# import json

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from dataset import ImageNet
from torch.utils.data import DataLoader

# from transformers import AutoTokenizer
import tokenizers
import wandb
import pickle
from tqdm import tqdm
from PIL import Image
#from pointllm.utils import process_v1  

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')






device = "cuda" if torch.cuda.is_available() else "cpu"
# IGNORE_INDEX = -100

cat_file = '/home/liuchao/PointLLM2/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt'
label_names = [line.rstrip() for line in open(cat_file)]

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]



def init_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model = model.to(torch.bfloat16).to(device)
    # for p in model.get_model().parameters():
    #     p.requires_grad = True
    # for name, param in model.get_model().named_parameters():
        
    #     print(f"Parameter {name} requires_grad: {param.requires_grad}")
    #     # param.requires_grad = True
    
    
    # model.config.mm_use_im_start_end = True
    # model.config.tune_mm_mlp_adapter = True
    # model.config.unfreeze_mm_vision_tower = True
    # print(model.config)
    
    conv = conversation_lib.conv_templates[args.conv_mode].copy()


    # model.get_model().fix_pointnet = False
    # model.get_model().fix_llm = False
    # model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    

    return model, tokenizer, conv, image_processor

def load_dataset( img_file, image_preocessor= None, config = True ):
    print(f"Loading {img_file}.")
    dataset = ImageNet(img_file=img_file, image_preocessor=image_preocessor, config = config)
    print("Done!")  
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def denormalize_clip(img_tensor, image_processor=None):
    """
    img_tensor: torch.Tensor with shape [B, 3, H, W] normalized by CLIP
    returns: List of PIL.Image in [0, 255] range
    """
    
    # CLIP normalization stats
    if image_processor is not None:
        clip_mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        clip_std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
    else:
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    # Clone to avoid modifying in-place
    img = img_tensor.clone().cpu()

    # Denormalize
    img = img * clip_std + clip_mean

    # Clamp to [0,1]
    img = img.clamp(0, 1)

    # Convert to numpy [B, H, W, C]
    img_np = img.permute(0, 2, 3, 1).numpy()

    # Convert to [0,255] uint8
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Convert to list of PIL images
    return [Image.fromarray(img_uint8[b]) for b in range(img_uint8.shape[0])]


def main(args):
    lr = args.lr
    budget = args.budget
    output_file = args.output_file
    batch_size = args.batch_size
    gamma = args.gamma
    
    model, tokenizer, conv, image_processor = init_model(args)
    model.train()
        
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)

    ori_dataset = load_dataset( img_file=args.ori_data_path, image_preocessor=image_processor, config= model.config ) # * defalut config
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataset = load_dataset( img_file=args.tgt_data_path, image_preocessor=image_processor, config= model.config ) # * defalut config
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)

    adv_img_list = []
    adv_label_list = []
    current_step = 0

    if os.path.exists(output_file):
        adv_img_list, _ = pickle.load(open(output_file, "rb"))
        current_step = len(adv_img_list)
        print(f"Loaded {current_step} adversarial point clouds from {output_file}")
        
    
    for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):

        if i * batch_size < current_step:
            continue

        ori_imgs =ori_data_dict['images'].to(device, dtype=torch.bfloat16)
        tgt_imgs = tgt_data_dict['images'].to(device, dtype=torch.bfloat16)
            
        with torch.no_grad():
            tgt_feature = model.encode_images(tgt_imgs)
            # print(tgt_feature.shape)
            if torch.isnan(tgt_feature).any():
                print(f"NaN detected in tgt_feature. Skip this sample.")
                continue
            tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            
        delta = torch.zeros_like(ori_imgs, requires_grad=True)
        
        opt = optim.Adam([
            {'params': delta, 'lr': lr},
        ], weight_decay=0.)
        scheduler = ExponentialLR(opt, gamma=gamma)
        # mask = torch.ones_like(ori_pc, dtype=torch.bool)
        # mask[..., 3:] = 0
        for j in range(args.steps):
            #adv_pc = pc_norm(ori_pc + delta)
            adv_imgs = ori_imgs + delta
            adv_feature = model.encode_images(adv_imgs)
            if torch.isnan(adv_feature).any():
                print(f"NaN detected in adv_feature. Skip this sample.")
                continue
            # adv_norms = adv_feature.norm(dim=2)
            # print(f"adv_feature norm: {adv_norms[0, :100]}")
            adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
            
            #l2_loss = torch.norm(adv_feature - tgt_feature, dim=2).mean()
            embedding_sim = torch.sum(adv_feature * tgt_feature, dim=2)  # computed from normalized features (therefore it is cos sim.)
            cosine_loss = torch.mean( 1.0 - embedding_sim )
            loss = cosine_loss
            #loss = l2_loss + cosine_loss
            
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            if j != 0 and j % 100 == 0:
                scheduler.step()
            
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, min=-budget, max=budget).detach()
            
            if j != 0 and j % 100 == 0:
                torch.cuda.empty_cache()

            # Log metrics to wandb
            if args.wandb:
                wandb.log({
                    f"loss": loss.item(),
                    f"embedding_similarity": embedding_sim.mean().item(),
                    f"max_delta": torch.max(torch.abs(delta)).item(),
                    f"mean_delta": torch.mean(torch.abs(delta)).item(),
                })
            
            # if j != 0 and j % 100 == 0:
            #     print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, \
            #         loss={loss.item():.5f}, \
            #         max coord delta={torch.max(torch.abs(delta)).item():.3f},   \
            #         mean coord delta={torch.mean(torch.abs(delta)).item():.3f} "
            #         )
        
        # save adversarial images
        adv_imgs = ori_imgs + delta
        adv_labels = tgt_data_dict['labels'].cpu().numpy()
        for b, label in enumerate(adv_labels):
            adv_img_list.append(adv_imgs[b].cpu().detach().float().numpy())
            adv_label_list.append(label)
        

    current_step = len(adv_img_list)
    with open(output_file, "wb") as f:
        pickle.dump((adv_img_list, adv_label_list), f)
        print(f"Saved {current_step} adversarial images to {output_file}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_path", type=str, \
        default="liuhaotian/llava-v1.5-7b") 
    parser.add_argument("--model_base", type=str, \
        default=None) 
    parser.add_argument("--conv_mode", type=str, \
        default="vicuna_v1")
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    

    # data 
    #parser.add_argument("--data_path", type=str, default="data/modelnet40", required=True)
    parser.add_argument("--ori_data_path", type=str, default="data/adv_pc")
    parser.add_argument("--tgt_data_path", type=str, default="data/adv_pc")
    parser.add_argument("--output_file", type=str, default="data/adv_pc.dat")
    

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    

    args = parser.parse_args()

    main(args)
    
    
    
    
    