#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
import torch.optim as optim

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


def generate_tgt_caption(tgt_labels):
    tgt_captions = []
    for b in range(tgt_labels.shape[0]):
        label_name= label_names[int(tgt_labels[b].item())]
        tgt_caption = f"This is an object of {label_name}."
        tgt_captions.append(tgt_caption)

    return tgt_captions




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

# def denormalize_clip(img_tensor, image_processor=None):
#     """
#     img_tensor: torch.Tensor with shape [3, H, W] normalized by CLIP
#     returns: PIL.Image in [0, 255] range
#     """
    
#     # CLIP normalization stats
#     if image_processor is not None:
#         clip_mean = torch.tensor(image_processor.image_mean)
#         clip_std = torch.tensor(image_processor.image_std)
#     else:
#         clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
#         clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
#     # Clone to avoid modifying in-place
#     img = img_tensor.clone().cpu()

#     # Denormalize
#     img = img * clip_std[:, None, None] + clip_mean[:, None, None]

#     # Clamp to [0,1]
#     img = img.clamp(0, 1)

#     # Convert to numpy [H, W, C]
#     img_np = img.permute(1, 2, 0).numpy()

#     # Convert to [0,255] uint8
#     img_uint8 = (img_np * 255).astype(np.uint8)

#     # Convert to PIL image
#     return Image.fromarray(img_uint8)



def generate_and_mask_target(
    tokenizer,
    conv,
    conversations
):

    image_tokens = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # Pad image_tokens to the same length and stack to get input_ids
    max_len = max(token.shape[0] for token in image_tokens)
    input_ids = torch.stack([
        torch.cat([token, torch.full((max_len - token.shape[0],), tokenizer.pad_token_id, dtype=token.dtype)])
        if token.shape[0] < max_len else token
        for token in image_tokens
    ])


    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            # if has_image:
            #     round_len = len(tokenizer_image_token(rou, tokenizer))
            #     instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            # else:
            #     round_len = len(tokenizer(rou).input_ids)
            #     instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return input_ids, targets


def main(args):
    lr = args.lr
    budget = args.budget
    output_file = args.output_file
    # output_visual_pc_path = args.output_visual_pc_path
    #pointnum = args.pointnum
    prompt_index = args.prompt_index
    batch_size = args.batch_size
    
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

    
    

    
    #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    

    qs = PROMPT_LISTS[prompt_index]
    
    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    
    #tgt_labels = [line.rstrip() for line in open("./objaverse_tgt_label.txt")]

    adv_img_list = []
    adv_label_list = []
    current_step = 0

    if os.path.exists(output_file):
        adv_img_list, _ = pickle.load(open(output_file, "rb"))
        current_step = len(adv_img_list)
        print(f"Loaded {current_step} adversarial point clouds from {output_file}")
        
    # try:
    if 1:
        for i, (ori_data_dict, tgt_data_dict) in enumerate(zip( ori_dataloader, tgt_dataloader ) ):
            if i * batch_size < current_step:
                continue

            ori_img = ori_data_dict['images']
            # adv_img = ori_img.clone()
            
            # ori_img_temp = image_processor(ori_img, return_tensors='pt')
            
            # print(ori_img.shape)
            # print(f"Value range of ori_img: min={ori_img.min().item()}, max={ori_img.max().item()}")
            
            # print(ori_img[0, :, :10, 0])
            
            adv_img = ori_img.clone().detach().requires_grad_(True)
            adv_img = adv_img.to(model.device, dtype=torch.bfloat16)
            
            
            tgt_labels = tgt_data_dict['labels']
            tgt_captions = generate_tgt_caption( tgt_labels )
            
            conversations = []
            for tgt_caption in tgt_captions:
                tgt_conv = conv.copy()
                tgt_conv.append_message(conv.roles[1], tgt_caption)
                conversations.append(tgt_conv.get_prompt())
                # print(f"Conversations: {conversations[-1]}")
            
            input_caption_ids, tgt_caption_ids = generate_and_mask_target(
                tokenizer,
                conv,
                conversations
            )
            
            # # Decode target captions for debugging
            # decoded_targets = [
            #     tokenizer.decode(tgt_caption_ids[b][tgt_caption_ids[b] != IGNORE_INDEX], skip_special_tokens=True)
            #     for b in range(tgt_caption_ids.shape[0])
            # ]
            # for idx, decoded_target in enumerate(decoded_targets):
            #     print(f"Target {idx}: {decoded_target}")
            
            perturb_mat = torch.zeros_like(adv_img)
            perturb_mat.requires_grad_(True)
            
            opt = optim.Adam([
                    {'params': perturb_mat, 'lr': lr},
                ], weight_decay=0.)
            
            
            for j in tqdm(range(args.steps)):
                
                with torch.no_grad():
                    perturb_mat.data = torch.clamp(perturb_mat.data, min=-budget, max=budget).detach()
                temp_adv_img = adv_img + perturb_mat
                with torch.no_grad():
                    temp_adv_img.data = torch.clamp(temp_adv_img.data, min=-2.0, max=2.0).detach()

                outputs = model(
                        input_ids=input_caption_ids.to(device), 
                        images= temp_adv_img,
                        # image_sizes= temp_adv_img.shape[-2:],
                        labels= tgt_caption_ids.to(device),
                        return_dict=True,
                            )
                

                # print(outputs.keys())
                
                # for k, v in outputs.items():
                #     if isinstance(v, torch.Tensor):
                #         v.retain_grad()
                #         print(f"Output {k} shape: {v.shape}, grad: {v.grad}")
                loss = outputs.loss

                opt.zero_grad()
                loss.backward()
                
                opt.step()
                
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")
                #     else:
                #         print(f"{name}: No gradient")
                # break
                
                # temp_adv_img = adv_img + perturb_mat
                # outputs = model(
                #         input_ids=input_caption_ids.to(device), 
                #         images= temp_adv_img,
                #         # image_sizes= temp_adv_img.shape[-2:],
                #         labels= tgt_caption_ids.to(device),
                #         return_dict=True,
                #             )
                # loss = outputs.loss.mean()
                
                
                # opt.zero_grad()
                # loss.backward()
                # opt.step()
                
                #temp_adv_pc = pc_norm(temp_adv_pc)
                
                if j % 100 == 0:
                    torch.cuda.empty_cache()
            
                # Log metrics to wandb
                if args.wandb:
                    wandb.log({
                        f"loss": loss.item(),
                        f"max_perturb": torch.max(torch.abs(perturb_mat)).item(),
                        f"mean_perturb": torch.mean(torch.abs(perturb_mat)).item(),
                    })
                # print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, \
                #       loss={loss.item():.5f} "
                #       )
  
            adv_img = (adv_img + perturb_mat).cpu().detach().float()

            # print(adv_img.shape)
            # print(f"Value range of ori_img: min={adv_img.min().item()}, max={adv_img.max().item()}")
            
            adv_img = denormalize_clip(adv_img, image_processor=image_processor)
            #save adversarial point cloud
            for b in range( len(adv_img) ):
                adv_img_list.append(adv_img[b])
                adv_label_list.append(np.asarray([tgt_labels[b].item()]))
                
                
    # except Exception as e:
    #     print(f"Exception: {e} at {i+1}/{args.num_samples//args.batch_size} step:{j+1}/{args.steps}")
    # finally:
    #     current_step = len(adv_pc_list)

    current_step = len(adv_img_list)
    with open(output_file, "wb") as f:
        pickle.dump((adv_img_list, adv_label_list), f)
        print(f"Saved {current_step} adversarial point clouds to {output_file}")


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
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1], help="0: What is this? 1: This is an object of ")

    

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
    
    
    
    
    