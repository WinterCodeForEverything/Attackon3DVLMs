# import argparse
# import json
# import os
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


# import numpy as np
# from tqdm import tqdm

#from pointllm.eval.traditional_evaluator import TraditionalMetricEvaluator

class TraditionalMetricEvaluator:
    def __init__(self):
        self.rouge = Rouge()

        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")


    def evaluate(self, adv_caption, tgt_caption):


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
        
        traditional_result = {
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
        }

        return traditional_result

