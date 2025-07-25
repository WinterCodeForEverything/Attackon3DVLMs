import torch
from victim_models.pointllm.conversation import  SeparatorStyle
from victim_models.pointllm.model.utils import KeywordsStoppingCriteria




class PointLLMGenerator(torch.nn.Module):
    def __init__(self, model, tokenizer, conv,  llm_args):
        super(PointLLMGenerator, self).__init__()
        self.model = model
        # self.conv = conv
        self.tokenizer = tokenizer
        self.temperature = llm_args.temperature
        self.top_k = llm_args.top_k
        self.top_p = llm_args.top_p
        self.num_beams = llm_args.num_beams
        self.max_length = llm_args.max_length
        self.dtype = getattr(torch, llm_args.dtype)
        # stopping_criteria
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        
        input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L
        self.input_ids_ = input_ids_
        
        self.stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)
        

        
    def generate( self, point_clouds ):
        
        B = point_clouds.shape[0]
        input_ids = self.input_ids_.repeat(B, 1) # * tensor of B, L
        
        self.model.eval() 
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_clouds.to(dtype=self.dtype), 
                do_sample= True if self.temperature > 0.0 else False,
                temperature= self.temperature if self.temperature > 0.0 else None,
                top_k=self.top_k,
                max_length=self.max_length,
                top_p=self.top_p,
                num_beams=self.num_beams,
                stopping_criteria=[self.stopping_criteria]) # * B, L'

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]

        return outputs
   
class GPT4PointGenerator(torch.nn.Module):
    def __init__(self, model, llm_args):
        super(GPT4PointGenerator, self).__init__()
        self.model = model
        # self.dtype = getattr(torch, llm_args.dtype)
        self.num_beams = llm_args.num_beams
        self.max_length = llm_args.max_length
        self.top_p = llm_args.top_p
        self.temperature = llm_args.temperature
        
    def generate(self, points):
        """
        Generate the output of the model.
        """
        
        samples = {
            "point": points,
        }
        outputs = self.model.generate(
            samples,
            num_beams=self.num_beams,
            use_nucleus_sampling= True if self.temperature > 0.0 else False,
            max_length=self.max_length,
            top_p=self.top_p if self.temperature > 0.0 else None,
            temperature=self.temperature if self.temperature > 0.0 else None,
        )
        
        return outputs

