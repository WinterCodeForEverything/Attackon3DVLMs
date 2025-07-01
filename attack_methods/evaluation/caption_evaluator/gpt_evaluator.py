#import argparse
import torch
#from torch.utils.data import DataLoader
# import os
# from pointllm.conversation import conv_templates, SeparatorStyle
# from pointllm.utils import disable_torch_init
# from pointllm.model import *
# from pointllm.model.utils import KeywordsStoppingCriteria
# from pointllm.data import ObjectPointCloudDataset
# from tqdm import tqdm
# from transformers import AutoTokenizer
#from pointllm.eval.evaluator import start_evaluation

# import os
# import json
import random
random.seed(0)
import re

PROMPT_LISTS = [
    "What is this object?", #Please describe it in few words, including its color, category and possibly the key characteristics.
    "This is an object of ",
    "Caption this 3D model in detail."
]
device = "cuda" if torch.cuda.is_available() else "cpu"

gpt4_open_free_from_cls_prompt = """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the type of object, not attributes such as color, size, or shape. Respond with 'T' if they refer to the same thing and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Example:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
Output: T#Both refer to a staircase.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """ # * about 230 input tokens

chatgpt_close_set_cls_prompt = """Given the following free-form description of a 3D object, please determine the most probable class index from the following 40 available categories, even if the description doesn't clearly refer to any one of them. Make your best-educated guess based on the information provided. If the description already contains a valid index, then the index should be selected. If it contains more than one valid index, then randomly select one index (specify your reason). If there is no valid index and it cannot be inferred from the information, return '-1#NA#Cannot infer'.
Categories:
{candidate_lists}
Reply with the format of 'index#class#short reason (no more than 10 words)'.

Examples:
Input: This is a 3D object model of a cartoon white truck.
Output: 7#car#Closest match to 'car' in categories.

Input: A green leaf in a flower pot.
Output: 26#plant#The primary subject 'leaf' directly indicates a plant.

Input: It's difficult to determine the exact type of this object due to insufficient details. But it seems to be like a piece of furniture.
Output: 33#table#Randomly select one kind of furniture from the list.

Input:  I cannot determine the specific type of the object without additional information or context.
Output: -1#NA#Cannot infer.

Now analyze the following:
Input: """

gpt4_object_captioning_prompt = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Example:
Human: A white brown skeleton
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
Output: 50#mention white; skeleton and robot have similar appearence.

Now score the following:
Human: {ground_truth}
Model: {model_output}
Output: """

chatgpt_object_captioning_prompt = gpt4_object_captioning_prompt
chatgpt_open_free_from_cls_prompt = gpt4_open_free_from_cls_prompt
gpt4_close_set_cls_prompt = chatgpt_close_set_cls_prompt

# GPT_PRICES = {
#     # * check https://openai.com/pricing for updated price
#     "gpt-3.5-turbo-0613": {
#         "price_1k_prompt_tokens": 0.0015,
#         "price_1k_completion_tokens": 0.002
#     },
#     "gpt-3.5-turbo-1106": {
#         "price_1k_prompt_tokens": 0.0010,
#         "price_1k_completion_tokens": 0.002
#     },
#     "gpt-4-0613":{
#         "price_1k_prompt_tokens": 0.03,
#         "price_1k_completion_tokens": 0.06  
#     },
#     "gpt-4-1106-preview":{
#         "price_1k_prompt_tokens": 0.01,
#         "price_1k_completion_tokens": 0.03
#     }
# }


# def init_model(model_name):
#     # Model
#     disable_torch_init()
#     model_name = os.path.expanduser(model_name)

#     # * print the model_name (get the basename)
#     print(f'[INFO] Model name: {os.path.basename(model_name)}')

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
#     model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

#     conv_mode = "vicuna_v1_1"

#     conv = conv_templates[conv_mode].copy()

#     return model, tokenizer, conv

class GPTEvaluator:
    def __init__(self, gpt_type): #model, tokenizer, conv, prompt_index, gpt_type
        self.gpt_prompt = chatgpt_object_captioning_prompt if "gpt-3.5" in gpt_type else gpt4_object_captioning_prompt
        
    def evaluate(self, adv_caption, tgt_caption):

        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=tgt_caption, model_output=adv_caption)}]
        
        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 

        prompt_tokens = gpt_response['usage']['prompt_tokens']
        completion_tokens = gpt_response['usage']['completion_tokens']

        gpt_response = gpt_response['choices'][0]["message"]['content']

        gpt_score, reason = self.parse_gpt_response_evaluate(gpt_response) # return 0, "INVALID", gpt_response if not valid
        
        gpt_result = {
            'gpt_score': gpt_score,
            'reason': reason,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }

        return gpt_result
    
    def parse_gpt_response_evaluate(self, gpt_response):
        """
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """

        # * use regular expression to extract
        pattern = r'(\d*#.*)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        gpt_score = gpt_response_list[0]
        reason = gpt_response_list[1] if len(gpt_response_list) > 1 else ""

        try:
            # * convert to int
            gpt_score = int(gpt_score)
            if gpt_score not in range(101): # * in 0-100
                # * not valid range
                gpt_score = -1
        except ValueError:
            print(f"Error: unale to parse {gpt_response}.")
            gpt_score = -1

        if gpt_score == -1:
            reason = gpt_response
        
        return gpt_score, reason



    # def generate_outputs(self, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    #     self.model.eval() 
    #     with torch.inference_mode():
    #         output_ids = self.model.generate(
    #             input_ids,
    #             point_clouds=point_clouds,
    #             do_sample=do_sample,
    #             temperature=temperature,
    #             top_k=top_k,
    #             max_length=max_length,
    #             top_p=top_p,
    #             stopping_criteria=[stopping_criteria]) # * B, L'

    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    #     outputs = [output.strip() for output in outputs]

    #     return outputs

    # def evaluate(self, adv_pc = None, tgt_pc = None, adv_caption = None, tgt_caption = None):
    #     assert (adv_pc is None) != (adv_caption is None), "Only one of adv_pc and adv_caption can be None, not both or neither."
    #     assert (tgt_pc is None) != (tgt_caption is None), "Only one of tgt_pc and tgt_caption can be None, not both or neither."
    #     assert (adv_pc is None) == (tgt_pc is None), "adv_pc and tgt_pc must be both None or both not None."

    #     #prompt_index = self.prompt_index
    #     input_ids_ =self.prefix_input_ids # * tensor of 1, L
    #     if adv_pc is not None:
    #         B = adv_pc.shape[0]
    #         input_ids = input_ids_.repeat(B, 1)
    #         adv_caption = self.generate_outputs( input_ids, adv_pc, self.stopping_criteria)
    #         tgt_caption = self.generate_outputs( input_ids, tgt_pc, self.stopping_criteria)

    #     gpt_score, reason, prompt_tokens, completion_tokens = self.gpt_evaluate(adv_caption, tgt_caption)
        
    #     return gpt_score, reason
            
        

            

            
    #     # # * First inferencing, then evaluate
    #     # if not os.path.exists(args.output_file_path):
    #     #     # * need inferencing
    #     #     # * load annotation files
    #     #     with open(args.anno_path, 'r') as fp:
    #     #         annos = json.load(fp)

    #     #     dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
    #     #     dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
            
    #     #     model, tokenizer, conv = init_model(args)

    #     #     # * convert annos file from [{"object_id": }] to {"object_id": }
    #     #     annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

    #     #     print(f'[INFO] Start generating results for {args.output_file}.')
    #     #     results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

    #     #     # * release model and tokenizer, and release cuda memory
    #     #     del model
    #     #     del tokenizer
    #     #     torch.cuda.empty_cache()
    #     # else:
    #     #     # * directly load the results
    #     #     print(f'[INFO] {args.output_file_path} already exists, directly loading...')
    #     #     with open(args.output_file_path, 'r') as fp:
    #     #         results = json.load(fp)

    #     # if args.start_eval:
    #     #     evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    #     #     eval_type_mapping = {
    #     #         "captioning": "object-captioning",
    #     #         "classification": "open-free-form-classification"
    #     #     }
    #     #     start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True, num_workers=20)
