import os
import json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset, load_from_disk

# ==========================================================
# CONFIG — SAFE + SEPARATE CACHE FOR DEBERTA
# ==========================================================

MODEL_NAME = "deepset/deberta-v3-large-squad2"

MAX_LEN = 384
DOC_STRIDE = 128

BATCH_SIZE = 16          # سریع‌تر
DEV_LIMIT = 1000         # فقط 1000 نمونه برای سرعت

# 👇 فقط این خط تغییر اساسی است
ARTIFACTS = Path("artifacts_deberta")
ARTIFACTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
def log(msg):
    print(msg)


# ==========================================================
# 1) LOAD DATASET  (CACHED)
# ==========================================================

def load_squad_dev():

    cache = ARTIFACTS / "dev_dataset"

    if cache.exists():
        log("[1/6] Loading cached dataset ...")
        ds = load_from_disk(str(cache))
    else:
        log("[1/6] Downloading SQuAD dev ...")
        ds = load_dataset("squad", split="validation")
        ds.save_to_disk(str(cache))

    # ---- فقط 1000 نمونه ----
    ds = ds.select(range(min(DEV_LIMIT, len(ds))))
    log(f"[1/6] Using {len(ds)} examples")
    return ds


# ==========================================================
# 2) TOKENIZE (WITH RESUME)
# ==========================================================

def tokenize_dataset(ds, tokenizer):

    cache = ARTIFACTS / "tokenized_dataset"

    if cache.exists():
        log("[2/6] Loading cached tokenization ...")
        return load_from_disk(str(cache))

    log("[2/6] Tokenizing ...")

    def fn(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=MAX_LEN,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            idx = sample_map[i]
            tokenized["example_id"].append(examples["id"][idx])

        return tokenized

    features = ds.map(fn, batched=True, remove_columns=ds.column_names)
    features.save_to_disk(str(cache))

    return features


# ==========================================================
# 3) SAFE INFERENCE WITH RESUME
# ==========================================================

def run_inference(features, model):

    cache_path = ARTIFACTS / "raw_predictions.json"

    if cache_path.exists():
        log("[3/6] Resuming from previous logits ...")
        all_results = json.load(open(cache_path, encoding="utf-8"))
        done = {r["example_id"] for r in all_results}
    else:
        all_results = []
        done = set()

    model.eval()

    SAVE_EVERY = 30

    for i in tqdm(range(0, len(features), BATCH_SIZE), desc="Inference"):

        batch = features[i:i+BATCH_SIZE]

        if all(ex in done for ex in batch["example_id"]):
            continue

        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(DEVICE),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(DEVICE),
        }

        with torch.no_grad():
            out = model(**inputs)

        start = out.start_logits.cpu().tolist()
        end = out.end_logits.cpu().tolist()

        for j in range(len(start)):
            ex_id = batch["example_id"][j]

            if ex_id in done:
                continue

            all_results.append({
                "example_id": ex_id,
                "start_logits": start[j],
                "end_logits": end[j],
                "input_ids": batch["input_ids"][j],
            })

            done.add(ex_id)

        if len(all_results) % SAVE_EVERY == 0:
            json.dump(all_results, open(cache_path, "w", encoding="utf-8"))

    json.dump(all_results, open(cache_path, "w", encoding="utf-8"))
    return all_results


# ==========================================================
# 4) POSTPROCESS
# ==========================================================

def logits_to_text(all_results, tokenizer):

    predictions = {}

    for r in all_results:
        s = int(torch.tensor(r["start_logits"]).argmax())
        e = int(torch.tensor(r["end_logits"]).argmax())

        if e < s:
            e = s

        tokens = tokenizer.convert_ids_to_tokens(
            r["input_ids"][s:e+1]
        )

        text = tokenizer.convert_tokens_to_string(tokens)
        predictions[r["example_id"]] = text

    with open(ARTIFACTS / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    return predictions


# ==========================================================
# 5) METRICS
# ==========================================================

import re, string
from collections import Counter

def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def f1_score(pred, gold):

    p = normalize(pred).split()
    g = normalize(gold).split()

    common = Counter(p) & Counter(g)
    num = sum(common.values())

    if num == 0:
        return 0

    precision = num / len(p)
    recall = num / len(g)

    return 2 * precision * recall / (precision + recall)


def em_score(pred, gold):
    return int(normalize(pred) == normalize(gold))


def compute_metrics(predictions, dataset):

    em, f1 = [], []

    for ex in dataset:
        qid = ex["id"]
        gold = ex["answers"]["text"][0]

        pred = predictions.get(qid, "")

        em.append(em_score(pred, gold))
        f1.append(f1_score(pred, gold))

    res = {
        "EM": round(100 * sum(em) / len(em), 2),
        "F1": round(100 * sum(f1) / len(f1), 2)
    }

    json.dump(res, open(ARTIFACTS/"metrics.json","w"), indent=2)

    return res


# ==========================================================
# MAIN
# ==========================================================

def main():

    log(f"Using device: {DEVICE}")

    ds = load_squad_dev()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    features = tokenize_dataset(ds, tokenizer)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    all_results = run_inference(features, model)

    preds = logits_to_text(all_results, tokenizer)

    metrics = compute_metrics(preds, ds)

    print("\n===== RESULTS =====")
    print(metrics)

    print("\n===== 10 SAMPLES =====")
    for ex in ds.select(range(10)):
        qid = ex["id"]
        print("Q:", ex["question"])
        print("G:", ex["answers"]["text"][0])
        print("P:", preds.get(qid,""))
        print("-"*50)


if __name__ == "__main__":
    main()
