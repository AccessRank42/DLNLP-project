## Project for DLNLP

We attempt to recreate the results of Dynamic Coattention Networks For Question Answering
and compare them to SOTA models.

Use the included environment.yml file to set up the used conda environment, then try out Jupyter notebook interface.ipynb.

Trained models should be available at:
https://drive.google.com/drive/folders/10YLpWcEBIQhPuEPOJjG5vb2hNW7GUzKs?usp=sharing


## Evaluation Phase

We evaluated four systems:

1. **DCN baseline (trained from scratch)**
2. **BERT warm-start** – bert-large-uncased-whole-word-masking-finetuned-squad  
3. **DeBERTa extension** – deepset/deberta-v3-large-squad2  
4. **DistilBERT** – distilbert-base-uncased-distilled-squad  

### Metrics

- Exact Match (EM)
- F1 score

### Results

| Model       | EM   | F1   |
|--------------|------|------|
| DCN          | 1.4  | 7.7  |
| BERT         | 64.4 | 79.1 |
| DistilBERT   | 66.9 | 76.8 |
| DeBERTa      | 83.5 | 89.6 |

---

## Example Predictions

### BERT
Q: Which NFL team represented the AFC at Super Bowl 50?  
G: Denver Broncos  
P: denver broncos  

### DeBERTa
Q: Where did Super Bowl 50 take place?  
G: Santa Clara, California  
P: Levi's Stadium  

### DistilBERT
Q: What color was used to emphasize the 50th anniversary?  
G: gold  
P: gold  

---

## Evaluation Artifacts

All outputs are available on Google Drive:

👉 **https://drive.google.com/drive/folders/1geYGuUf9SNxvZRsBk_JO7Gipez7cuM1Z**
