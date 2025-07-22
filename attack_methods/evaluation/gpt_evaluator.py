import numpy as np
import torch
import os
import openai
import random
random.seed(0)
# import re



# gpt4_open_free_from_cls_prompt = """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the category of object, not attributes such as color, size, or shape, neight cartoon or realistic. Respond with 'T' if they refer to the same thing and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
# Example 1:
# Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
# Output: T#Both refer to a staircase.

# Example 2:
# Input: 1. Animation of a penguin with a crown. 2. a penguin wearing a hat
# Output: T#Both refer to a penguin.

# Example 3:
# Input: 1. a toy house with cubical body and triangular roof and open door". 2. a small house with a roof, featuring a bird, a plane, and a birdhouse

# Now, analyze the following:
# Input: 1. {ground_truth} 2. {model_output}
# Output: """ # * about 230 input tokens

gpt4_open_free_from_cls_prompt = """You are given two sentences. Your task is to determine whether they refer to the same general **category** of object or concept — regardless of color, material, size, realism, or artistic style (e.g. cartoon vs realistic). Focus purely on the object's type or category.

Respond with:
- 'T' if both refer to the same object category.
- 'F' if they refer to different categories.

Also include a **brief reason (no more than 20 words)** after a '#' symbol.

### Examples:

Example 1:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown  
Output: T#Both are staircases.

Example 2:
Input: 1. Animation of a penguin with a crown. 2. a penguin wearing a hat  
Output: T#Both describe penguins.

Example 3:
Input: 1. a red car. 2. a white boat.  
Output: F#Car and boat are different vehicle types.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}  
Output:"""

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




class OpenAIGPT():
    def __init__(self, model="gpt-40-mini", temperature=1, top_p=1, max_tokens=2048, **kwargs) -> None:
        #setup_openai(model)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.default_chat_parameters = {
            "model": model, 
            "temperature": temperature, 
            "top_p": top_p, 
            "max_tokens": max_tokens,
            **kwargs
        }

    # @retry_with_exponential_backoff 
    def safe_chat_complete(self, messages, content_only=True, **kwargs):
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)

        response = self.client.chat.completions.create(
            messages=messages,
            **chat_parameters
        )
        response_dict = response.model_dump()
        

        if content_only:
            response_dict = response_dict['choices'][0]["message"]['content']

        return response_dict



class GPTEvaluation(torch.nn.Module):
    def __init__(self, Generation, gpt_type, is_targeted=True):   
        super(GPTEvaluation, self).__init__()
        self.Generation = Generation
        self.gpt_prompt = gpt4_open_free_from_cls_prompt 
        self.invalid_responses = 0
        self.is_targeted = is_targeted
        

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
            if not self.is_targeted:
                accuracy = 1 - accuracy # * for untargeted attack, we want to maximize the difference
            # accuracy =
            accuracies[i] = accuracy
            result.update({
                'accuracy': accuracy,
                'cls_result': cls_result,
                'reason': reason
            })
            # print(result)
            results.append(result)
        # print(results)

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

        # Attempt to split using the '#' delimiter
        if '#' in gpt_response:
            parts = gpt_response.split('#', 1)
            cls_result = parts[0].strip().upper()
            reason = parts[1].strip()
        else:
            # Fall back to first character as label and rest as reason
            cls_result = gpt_response[0].strip().upper() if gpt_response else ""
            reason = gpt_response[1:].strip() if len(gpt_response) > 1 else ""

        # Validate the classification result
        if cls_result not in ['T', 'F']:
            self.invalid_responses += 1
            return 0, "INVALID", gpt_response

        accuracy = 1 if cls_result == 'T' else 0
        return accuracy, cls_result, reason


