import numpy as np
import torch
import os
from pointllm.eval.utils import OpenAIGPT
import random
random.seed(0)
import re

import json

# from pointllm.model.utils import KeywordsStoppingCriteria
# from pointllm.conversation import conv_templates, SeparatorStyle

device = "cuda" if torch.cuda.is_available() else "cpu"

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

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]



class Evaluation(torch.nn.Module):
    def __init__(self, Generation, gpt_type, candidate_lists_names ):
        super(Evaluation, self).__init__()
        self.Generation = Generation
        self.gpt_prompt = chatgpt_close_set_cls_prompt 
        
        # * make the prompt
        self.candidate_lists_names = candidate_lists_names
        candidate_lists = [f'{i}: {cat}' for i, cat in enumerate(self.candidate_lists_names)]
        self.num_categories = len(candidate_lists)
        self.candidate_lists = '\n'.join(candidate_lists)
        self.gpt_prompt = self.gpt_prompt.format(num_categories=self.num_categories, candidate_lists=self.candidate_lists) + "{model_output}\nOutput: "
        
        #print(f"OpenAIGPT config: ")
        #print(self.default_chat_parameters)
        
        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 2048
        }

        
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
    
        
    def forward(self, adv_pc, tgt_labels):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
            
        model_outputs = self.Generation.generate(adv_pc) # * B, L'
            
        # outputs = self.generate_outputs( adv_pc, temperature=self.temperature, top_k=self.top_k, 
        #                 max_length=self.max_length, top_p=self.top_p, num_beams=self.num_beams )
        
        # B = adv_pc.shape[0]
        B = len(model_outputs)
        accuracies = np.zeros(B, dtype=np.int32)
        results = []
        for i in range(B):
            result = {
                'ground_truth': tgt_labels[i].item(),
                'model_output': model_outputs[i],
                
            }
            accuracy, cls_result, cls_label, reason = self.evaluate_result( result )
            accuracies[i] = accuracy
            result.update({
                'accuracy': accuracy,
                'cls_result': cls_result,
                'cls_label': cls_label,
                'reason': reason
            })
            results.append(result)

        return accuracies, results
    
    
    def evaluate_result(self, result):
        ground_truth = result['ground_truth']
        model_output = result['model_output']

        messages = [{"role": "user", "content": self.gpt_prompt.format(model_output=model_output)}]
        
        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 


        gpt_response = gpt_response['choices'][0]["message"]['content']

        accuracy, cls_result, cls_label, reason = self.parse_gpt_response_evaluate(gpt_response, ground_truth) # return 0, "INVALID", gpt_response if not valid

        return  accuracy, cls_result, cls_label, reason
    
    def parse_gpt_response_evaluate(self, gpt_response, ground_truth):
        """
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """

        # * use regular expression to extract
        pattern = r'(\d+#[^#]*#.*$)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        cls_result = gpt_response_list[0]
        cls_label = gpt_response_list[1] if len(gpt_response_list) > 1 else ""
        reason = gpt_response_list[2] if len(gpt_response_list) > 2 else ""

        try:
            # * convert to int
            cls_result = int(cls_result)
            if cls_result not in range(self.num_categories) or cls_label == "NA":
                # * not valid range
                cls_result = -1
        except ValueError:
            print(f"Error: unale to parse {gpt_response}.")
            cls_result = -1

        # if cls_result == -1:
        #     # * random choose one index from 0 to self.num_categories
        #     cls_result = random.choice(range(self.num_categories))
        #     cls_label = "INVALID"
        #     reason = gpt_response
        
        accuracy = 1 if cls_result == ground_truth else 0 

        return accuracy, cls_result, cls_label, reason


