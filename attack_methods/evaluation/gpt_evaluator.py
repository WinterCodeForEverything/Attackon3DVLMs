import numpy as np
import torch
import os
import openai
import random
random.seed(0)
import re



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

# gpt4_object_captioning_prompt = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

# Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

# Example:
# Human: A white brown skeleton
# Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
# Output: 50#mention white; skeleton and robot have similar appearence.

# Now score the following:
# Human: {ground_truth}
# Model: {model_output}
# Output: """

gpt4_object_captioning_prompt = """You are evaluating the alignment between a model-generated caption and a human-written ground truth caption for a 3D object.

1. Identify distinct descriptive aspects (e.g., color, material, shape, function) mentioned in the human caption.
2. Compare them to the model's caption. If a model captures an aspect exactly, give full credit. If it partially matches (e.g., similar concept or synonym), give partial credit.
3. Compute a final score from 0 to 100 based on the proportion of correctly or partially matched aspects.
4. Return your answer **in exactly one line** using the format: <score>#<short reason>

The reason should be concise (≤15 words), summarizing why that score was assigned.

Only output one line with no extra explanation.

Example:
Human: A white brown skeleton  
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.  
Output: 50#mention white; skeleton and robot have similar appearance.

Now score the following:
Human: {ground_truth}  
Model: {model_output}  
Output:"""



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
    def __init__(self, Generation, 
                        gpt_type, 
                        is_targeted=True,
                        has_defense=False,
                        defensor=None):   
        super(GPTEvaluation, self).__init__()
        self.Generation = Generation
        self.gpt_prompt = gpt4_open_free_from_cls_prompt 
        self.invalid_responses = 0
        self.is_targeted = is_targeted
        self.has_defense = has_defense
        self.defensor = defensor
        assert (not has_defense) or (defensor is not None), "Defensor must be provided if has_defense is True."
        
        

        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 1024
        }

        
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
    
        
    def forward(self, adv_pc, tgt_labels ):
        
        if self.has_defense:
            adv_pc = self.defensor(adv_pc)
        
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
            print("openai.RateLimitError:", openai.RateLimitError)
            # print(f"GPT response: {gpt_response}")
        

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


gpt4_open_free_from_cls_batched_prompt = """You are given several pairs of sentences. For each pair, determine whether they refer to the same general **category** of object or concept — ignoring differences in style, size, color, material, or realism (e.g., cartoon vs realistic). Focus only on the object's **type** or **semantic category**.

For each pair:
- Respond with 'T' if both refer to the same category.
- Respond with 'F' if they refer to different categories.
- Follow the response with a '#' and a brief justification (≤20 words).

### Format:
Input i: <sentence 1> || <sentence 2>  
Output i: T/F#<brief reason>

### Example:

Input 1: Spiral staircase that goes from a ground floor || This is a 3D model of wooden stairs in light brown  
Output 1: T#Both refer to staircases.

Input 2: Animation of a penguin with a crown || A penguin wearing a hat  
Output 2: T#Both describe penguins.

Input 3: A red car || A white boat  
Output 3: F#Car and boat are different types of vehicles.

Now, analyze the following:

{batched_inputs}
"""


class GPTBatchEvaluation(torch.nn.Module):
    def __init__(self, Generation, 
                        gpt_type, 
                        is_targeted=True,
                        has_defense=False,
                        defensor=None):    
        super(GPTBatchEvaluation, self).__init__()
        self.Generation = Generation
        self.gpt_prompt = gpt4_open_free_from_cls_batched_prompt
        self.invalid_responses = 0
        self.is_targeted = is_targeted
        self.has_defense = has_defense
        self.defensor = defensor
        assert (not has_defense) or (defensor is not None), "Defensor must be provided if has_defense is True."
        

        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 2048
        }

        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
        # Ensure results length matches B by padding or truncating

    def forward(self, adv_pc, tgt_labels):
        if self.has_defense:
            adv_pc = self.defensor(adv_pc)
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]

        model_outputs = self.Generation.generate(adv_pc)  # * B, L'
        B = adv_pc.shape[0]
        accuracies = np.zeros(B, dtype=np.int32)
        results = []

        # Build batched input string for the prompt
        batched_inputs = ""
        for i in range(B):
            batched_inputs += f"Input {i+1}: {tgt_labels[i]} || {model_outputs[i]}\n"

        prompt = self.gpt_prompt.format(batched_inputs=batched_inputs)
        messages = [{"role": "user", "content": prompt}]

        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False)
        gpt_response_content = gpt_response['choices'][0]["message"]['content']

        # Parse each output line
        output_lines = re.findall(r'Output \d+:\s*(.*)', gpt_response_content)
        # If not found, fallback to splitting by lines and filtering
        if not output_lines:
            output_lines = [line for line in gpt_response_content.split('\n') if line.strip().startswith(('T', 'F'))]

        # If fewer outputs than inputs, pad with "F#Missed output"
        while len(output_lines) < B:
            output_lines.append("F#Missed output")
        if len(output_lines) > B:
            output_lines = output_lines[:B]

        for i in range(B):
            output_line = output_lines[i].strip()
            # print(f"Output {i+1}: {output_line}")
            accuracy, cls_result, reason = self.parse_gpt_response_evaluate(output_line)
            if not self.is_targeted:
                accuracy = 1 - accuracy
            accuracies[i] = accuracy
            result = {
                'ground_truth': tgt_labels[i],
                'model_output': model_outputs[i],
                'accuracy': accuracy,
                'cls_result': cls_result,
                'reason': reason
            }
            results.append(result)

        return accuracies, results

    def evaluate_result(self, result):
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        batched_inputs = f"Input 1: {ground_truth} || {model_output}\n"
        prompt = self.gpt_prompt.format(batched_inputs=batched_inputs)
        messages = [{"role": "user", "content": prompt}]

        try:
            gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False)
        except openai.RateLimitError:
            print("openai.RateLimitError:", openai.RateLimitError)

        gpt_response_content = gpt_response['choices'][0]["message"]['content']
        # Parse output line
        match = re.search(r'Output 1:\s*(.*)', gpt_response_content)
        output_line = match.group(1).strip() if match else gpt_response_content.strip()
        accuracy, cls_result, reason = self.parse_gpt_response_evaluate(output_line)
        return accuracy, cls_result, reason

    def parse_gpt_response_evaluate(self, gpt_response):
        gpt_response = gpt_response.strip()
        # Attempt to split using the '#' delimiter
        if '#' in gpt_response:
            parts = gpt_response.split('#', 1)
            cls_result = parts[0].strip().upper()
            reason = parts[1].strip()
        else:
            cls_result = gpt_response[0].strip().upper() if gpt_response else ""
            reason = gpt_response[1:].strip() if len(gpt_response) > 1 else ""

        if cls_result not in ['T', 'F']:
            self.invalid_responses += 1
            return 0, "INVALID", gpt_response

        accuracy = 1 if cls_result == 'T' else 0
        return accuracy, cls_result, reason


class GPTObjectCaptioningEvaluator(torch.nn.Module):
    def __init__(self, Generation, gpt_type, success_score_threshold, is_targeted=True):
        super(GPTObjectCaptioningEvaluator, self).__init__()
        self.Generation = Generation
        self.gpt_type = gpt_type
        self.success_score_threshold = success_score_threshold
        self.is_targeted = is_targeted
        self.gpt_prompt = gpt4_object_captioning_prompt
        
        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 4096
        }

        
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
        
    def forward( self, adv_pc, tgt_labels ):
        
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
            gpt_score, reason = self.evaluate_result( result )
            accuracy = 0
            if gpt_score >= self.success_score_threshold:
                accuracy = 1
            if not self.is_targeted:
                accuracy = 1 - accuracy # * for untargeted attack, we want to maximize the difference
            accuracies[i] = accuracy
            result.update({
                'accuracy': accuracy,
                'gpt_score': gpt_score,
                'reason': reason
            })
            # print(result)
            results.append(result)
        # print(results)

        return accuracies, results


    def parse_gpt_response_evaluate(self, gpt_response ):
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

    def evaluate_result(self, result):
        
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]

        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 

        # prompt_tokens = gpt_response['usage']['prompt_tokens']
        # completion_tokens = gpt_response['usage']['completion_tokens']

        gpt_response = gpt_response['choices'][0]["message"]['content']

        gpt_score, reason = self.parse_gpt_response_evaluate(gpt_response) # return 0, "INVALID", gpt_response if not valid

        return gpt_score, reason

gpt4_object_captioning_batch_prompt = (
    "You are given several pairs of captions. For each pair, evaluate how well the model-generated caption matches the human-written ground truth caption for a 3D object.\n"
    "For each pair:\n"
    "1. Identify distinct descriptive aspects in the human caption (such as color, material, shape, function, etc).\n"
    "2. Compare each aspect to the model's caption. Give full credit if the aspect is exactly matched, partial credit for similar or related concepts.\n"
    "3. Calculate a score from 0 to 100 based on the proportion of aspects correctly or partially matched.\n"
    "4. Respond in exactly one line per pair, using the format: Output i: <score>#<short reason>\n"
    "The reason should be concise (≤15 words), summarizing the main factors for the score.\n"
    "Only output one line per pair, with no extra explanation or commentary.\n"
    "Example:\n"
    "Input 1: Human: A white brown skeleton || Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.\n"
    "Output 1: 50#mentions white; skeleton and robot have similar appearance.\n"
    "Now score the following:\n"
    "{batched_inputs}"
)

class GPTObjectCaptioningBatchEvaluator(torch.nn.Module):
    def __init__(self, Generation, gpt_type, success_score_threshold, is_targeted=True):
        super(GPTObjectCaptioningBatchEvaluator, self).__init__()
        self.Generation = Generation
        self.gpt_type = gpt_type
        self.success_score_threshold = success_score_threshold
        self.is_targeted = is_targeted
        self.gpt_prompt = gpt4_object_captioning_batch_prompt

        self.default_chat_parameters = {
            "model": gpt_type,
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 2096
        }
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)

    def forward(self, adv_pc, tgt_labels):
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]

        model_outputs = self.Generation.generate(adv_pc)
        B = adv_pc.shape[0]
        accuracies = np.zeros(B, dtype=np.int32)
        results = []

        # Build batched input string for the prompt
        batched_inputs = ""
        for i in range(B):
            batched_inputs += f"Input {i+1}: Human: {tgt_labels[i]} || Model: {model_outputs[i]}\n"

        prompt = self.gpt_prompt.format(batched_inputs=batched_inputs)
        messages = [{"role": "user", "content": prompt}]

        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False)
        gpt_response_content = gpt_response['choices'][0]["message"]['content']

        # Parse each output line
        output_lines = re.findall(r'Output \d+:\s*(.*)', gpt_response_content)
        # If not found, fallback to splitting by lines and filtering
        if not output_lines:
            output_lines = [line for line in gpt_response_content.split('\n') if line.strip() and (line.strip()[0].isdigit() or '#' in line)]

        # If fewer outputs than inputs, pad with "0#Missed output"
        while len(output_lines) < B:
            output_lines.append("0#Missed output")
        if len(output_lines) > B:
            output_lines = output_lines[:B]

        for i in range(B):
            output_line = output_lines[i].strip()
            gpt_score, reason = self.parse_gpt_response_evaluate(output_line)
            accuracy = 1 if gpt_score >= self.success_score_threshold else 0
            if not self.is_targeted:
                accuracy = 1 - accuracy
            result = {
                'ground_truth': tgt_labels[i],
                'model_output': model_outputs[i],
                'accuracy': accuracy,
                'gpt_score': gpt_score,
                'reason': reason
            }
            accuracies[i] = accuracy
            results.append(result)

        return accuracies, results

    def parse_gpt_response_evaluate(self, gpt_response):
        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')
        gpt_score = gpt_response_list[0]
        reason = gpt_response_list[1] if len(gpt_response_list) > 1 else ""

        try:
            gpt_score = int(gpt_score)
            if gpt_score not in range(101):
                gpt_score = -1
        except ValueError:
            gpt_score = -1

        if gpt_score == -1:
            reason = gpt_response

        return gpt_score, reason

    def evaluate_result(self, result):
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        batched_inputs = f"Input 1: Human: {ground_truth} || Model: {model_output}\n"
        prompt = self.gpt_prompt.format(batched_inputs=batched_inputs)
        messages = [{"role": "user", "content": prompt}]

        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False)
        gpt_response_content = gpt_response['choices'][0]["message"]['content']
        match = re.search(r'Output 1:\s*(.*)', gpt_response_content)
        output_line = match.group(1).strip() if match else gpt_response_content.strip()
        gpt_score, reason = self.parse_gpt_response_evaluate(output_line)
        return gpt_score, reason
