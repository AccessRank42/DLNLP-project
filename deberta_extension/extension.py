import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

class DeBERTaExtension:
    def __init__(self, model_id="deepset/deberta-v3-large-squad2", batch_size=16):
        self.dataset = load_dataset(
            'json', 
            data_files={'validation': 'dev-v1.1.json'}, 
            field='data'
        )['validation']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=model_id, 
            tokenizer=self.tokenizer, 
            device=0,          #use cuda device
            batch_size=batch_size # we enable batch processing
        )