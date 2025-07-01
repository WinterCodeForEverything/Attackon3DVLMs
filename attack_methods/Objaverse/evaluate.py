import numpy as np
import torch
import os
from pointllm.eval.utils import OpenAIGPT
import random
random.seed(0)
import re

import openai

gpt4_open_free_from_cls_prompt = """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the type of object, not attributes such as color, size, or shape. Respond with 'T' if they refer to the same thing and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Example:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
Output: T#Both refer to a staircase.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """ # * about 230 input tokens


class Evaluation(torch.nn.Module):
    def __init__(self, Generation, gpt_type):   
        super(Evaluation, self).__init__()
        self.Generation = Generation
        self.gpt_prompt = gpt4_open_free_from_cls_prompt 
        

        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 2048
        }

        
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
    
        
    def forward(self, adv_pc, tgt_labels ):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
            
        model_outputs = self.Generation.generate(adv_pc) # * B, L'
        
        B = adv_pc.shape[0]
        accuracies = np.zeros(B, dtype=np.int32)
        results = []
        for i in range(B):
            result = {
                'ground_truth': tgt_labels[i],
                'model_output': model_outputs[i],
                
            }
            accuracy, cls_result, reason = self.evaluate_result( result )
            accuracies[i] = accuracy
            result.update({
                'accuracy': accuracy,
                'cls_result': cls_result,
                'reason': reason
            })
            results.append(result)

        return accuracies, results
    
    def evaluate_result(self, result):
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]

        try:
            gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False)   
        except openai.RateLimitError:
        # handle rate limit error
            print(f"GPT response: {gpt_response}")
        

        # prompt_tokens = gpt_response['usage']['prompt_tokens']
        # completion_tokens = gpt_response['usage']['completion_tokens']

        gpt_response = gpt_response['choices'][0]["message"]['content']
        
        #print(f"GPT response: {gpt_response}")


        accuracy, cls_result, reason = self.parse_gpt_response_evaluate(gpt_response) # return 0, "INVALID", gpt_response if not valid

        return accuracy, cls_result, reason

    def parse_gpt_response_evaluate(self, gpt_response):
        gpt_response = gpt_response.strip()

        cls_result = gpt_response[0].upper()
        reason = gpt_response[2:] if len(gpt_response) > 2 else ""

        if cls_result not in ['T', 'F']:
            self.invalid_responses += 1
            return 0, "INVALID", gpt_response

        accuracy = 1 if cls_result == 'T' else 0

        return accuracy, cls_result, reason


