"""
Language model scorer for perplexity computation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LMScorer:
    def __init__(self, model_name='gpt2-xl', model=None, tokenizer=None):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            print('[LMScorer]: model reused.')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.cuda()
            print('[LMScorer]: new model loaded.')

    @torch.no_grad()
    def sentence_score(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        outputs = self.model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss).item()
        return ppl
