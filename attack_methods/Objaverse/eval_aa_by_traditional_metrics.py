import argparse
import json
import os
import random
random.seed(0)

import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
#from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch


import numpy as np
from tqdm import tqdm

#from pointllm.eval.traditional_evaluator import TraditionalMetricEvaluator

class TraditionalMetricEvaluator():
    def __init__(self, inputs, output_dir, output_file):
        self.results = inputs['results']
        self.inference_prompt = inputs['prompt']
        self.output_dir = output_dir
        self.output_file = output_file
        self.rouge = Rouge()
        self.response_data = []

        self.ground_truths = []
        self.generated_captions = []

        #self.sbert_model = SentenceTransformer('all-mpnet-base-v2')

        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

        self.scores = {
            'bleu-1': [],
            'bleu-2': [],
            'bleu-3': [],
            'bleu-4': [],
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'meteor': [],
            #'sbert_similarity': [],
            'simcse_similarity': [],
            'success_aa': []
        }

    def evaluate_result(self, result):
        object_id = result['object_id']
        ori_caption = result['ori_caption']
        adv_caption = result['adv_caption']
        tgt_caption = result['tgt_caption']

        # metrics to evaluate the similarity between adversarial caption and target caption
        # create a SmoothingFunction object
        smoothing_function = SmoothingFunction().method1 # * used to deal with non-overlap n-gram

        # calculate BLEU-1 score with smoothing function
        bleu_1_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)

        # calculate BLEU-2, BLEU-3, and BLEU-4 scores
        bleu_2_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu_3_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu_4_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        # calculate ROUGE-L score
        rouge_scores_l = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-l']
        rouge_scores_1 = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-1']
        rouge_scores_2 = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-2']

        # calculate METEOR score
        meteor_scores = meteor_score([tgt_caption.split()], adv_caption.split())

        # # Calculate SBERT similarity
        # embeddings = self.sbert_model.encode([tgt_caption, adv_caption])
        # sbert_similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()

        # calculate SimCSE similarity
        # Tokenize input texts
        inputs = self.simcse_tokenizer([tgt_caption, adv_caption], padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarity
        simcse_similarity = 1 - cosine(embeddings[0], embeddings[1]) # * consine actually calculates consine distance, which is 1 - consine similarity
        simcse_similarity = float(simcse_similarity)
        # calculate SimCSE similarity between adversarial caption and original caption
        # Tokenize input texts
        inputs_ori = self.simcse_tokenizer([ori_caption, adv_caption], padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings_ori = self.simcse_model(**inputs_ori, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarity
        simcse_similarity_ori = 1 - cosine(embeddings_ori[0], embeddings_ori[1])
        simcse_similarity_ori = float(simcse_similarity_ori)
        
        scores = {
            'bleu-1': bleu_1_score * 100,
            'bleu-2': bleu_2_score * 100,
            'bleu-3': bleu_3_score * 100,
            'bleu-4': bleu_4_score * 100,
            'rouge-l': rouge_scores_l['f'] * 100,
            'rouge-1': rouge_scores_1['f'] * 100,
            'rouge-2': rouge_scores_2['f'] * 100,
            'meteor': meteor_scores * 100,
            #'sbert_similarity': sbert_similarity * 100,
            'simcse_similarity': simcse_similarity * 100,
            'success_aa': float(simcse_similarity > simcse_similarity_ori) * 100
        }

        return object_id, ori_caption, adv_caption, tgt_caption, scores

    def evaluate(self):
        print("Starting evaluation...")

        success_num = 0
        for result in tqdm(self.results, desc="Evaluating"):  
            object_id, ori_caption, adv_caption, tgt_caption, scores = self.evaluate_result(result)

            # save the object_id, model_output, ground_truth, and scores for each result
            self.response_data.append({
                'object_id': object_id,
                'ori_caption': ori_caption,
                'adv_caption': adv_caption,
                'tgt_caption': tgt_caption,
                'scores': scores,
            })

            # save the scores for overall results
            for metric, score in scores.items():
                self.scores[metric].append(score)
        
        print("Evaluation finished.")
        self.save_results()
        self.print_results()

    def save_results(self):
        output_path = os.path.join(self.output_dir, self.output_file)

        with open(output_path, 'w') as f:
            results_to_save = {
                #'success_rate': f"{np.mean(self.scores['success_aa']):.4f}",
                'inference_prompt': self.inference_prompt,
                'overall_scores': {metric: f"{np.mean(scores):.4f}" for metric, scores in self.scores.items()},
                'results': self.response_data,
            }
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {output_path}")

    def print_results(self):
        print('-' * 80)
        print("Results:")
        for metric, scores in self.scores.items():
            print(f"Average {metric.upper()} Score: {np.mean(scores):.4f}")


def start_evaluation(results, output_dir, output_file,
                        parallel=True, num_workers=20):
    """
    Args:
        results: dict or file path to the json file containing the dict
        output_file: the path the final evaluation results to be saved.
    """
    if isinstance(results, str):
        with open(results, 'r') as fp:
            results = json.load(fp)
            
    

    evaluator = TraditionalMetricEvaluator( results, output_dir, output_file ) 
    evaluator.evaluate()
    
if __name__ == '__main__':
    results = './result/noise_size_5_result/PointLLM_brief_description_mrg_200_Objaverse_classification_prompt0.json'
    output_dir = './result/noise_size_5_result/MF_pp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = 'traditional_metrics.json'
    start_evaluation(results, output_dir, output_file)