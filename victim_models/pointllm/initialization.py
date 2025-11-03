import os
from transformers import AutoTokenizer
from victim_models.pointllm.conversation import conv_templates
from victim_models.pointllm.utils import disable_torch_init
from victim_models.pointllm.model import PointLLMLlamaForCausalLM

def init_PointLLM(model_name, dtype, device, is_eval=True):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=dtype).to(device)
    if is_eval:
        model.eval()
    model.get_model().fix_pointnet = False
    model.get_model().fix_llm = False
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()
    
    #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = "What is this?"

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    
    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    #qs = qs + "This is an object of "
    conv.append_message(conv.roles[0], qs)

    return model, tokenizer, conv
