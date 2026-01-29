# ===== dcn_final_eval.py =====
# DCN official-style evaluation on SQuAD v1.1 dev
# SAFE PIPELINE: builds uuid-aligned dev files if missing, saves progress & partial outputs.

import os
import re
import json
import time
import argparse
from collections import Counter

import numpy as np
import chainer as ch
from tqdm import tqdm

from model import DynamicCoattentionNW
from preprocessing.vocab import get_glove, PAD_ID, UNK_ID


# --------------------------
# Utils: official SQuAD v1.1 metrics
# --------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / max(1, len(pred_tokens))
    recall = 1.0 * num_same / max(1, len(gold_tokens))
    return (2 * precision * recall) / max(1e-12, (precision + recall))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def squad_v11_evaluate(dev_json_path: str, predictions: dict) -> dict:
    with open(dev_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    dataset = dataset_json["data"]
    total = 0
    exact_match = 0.0
    f1 = 0.0

    missing = 0

    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                qid = qa["id"]
                if qid not in predictions:
                    missing += 1
                    continue
                pred = predictions[qid]
                gold_texts = [a["text"] for a in qa["answers"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, pred, gold_texts)
                f1 += metric_max_over_ground_truths(f1_score, pred, gold_texts)

    exact_match = 100.0 * exact_match / max(1, total)
    f1 = 100.0 * f1 / max(1, total)

    return {
        "exact_match": exact_match,
        "f1": f1,
        "total": total,
        "missing_predictions": missing,
    }


# --------------------------
# Tokenization / preprocess (aligned with squad_preprocess.py)
# --------------------------
def ensure_nltk_punkt():
    # Do not fail if already available.
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # your project used punkt_tab; standard punkt also works
        nltk.download("punkt")
    return nltk


def tokenize_like_project(nltk, sequence: str):
    tokens = [t.replace("``", '"').replace("''", '"').lower() for t in nltk.word_tokenize(sequence)]
    return tokens


def get_char_word_loc_mapping(context: str, context_tokens):
    acc = ""
    current_token_idx = 0
    mapping = {}

    for char_idx, char in enumerate(context):
        if char != " " and char != "\n":
            acc += char
            if current_token_idx >= len(context_tokens):
                return None
            context_token = context_tokens[current_token_idx]
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)
                acc = ""
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    return mapping


def build_dev_eval_files(dev_json_path: str, out_dir: str):
    """
    Build uuid-aligned files:
      dev_eval.context, dev_eval.question, dev_eval.span, dev_eval.uuid
    Uses same discard logic as preprocessing/squad_preprocess.py (no shuffle).
    """
    os.makedirs(out_dir, exist_ok=True)

    ctx_path = os.path.join(out_dir, "dev_eval.context")
    q_path   = os.path.join(out_dir, "dev_eval.question")
    span_path= os.path.join(out_dir, "dev_eval.span")
    uuid_path= os.path.join(out_dir, "dev_eval.uuid")

    # If already built, skip
    if all(os.path.exists(p) for p in [ctx_path, q_path, span_path, uuid_path]):
        return {
            "context": ctx_path,
            "question": q_path,
            "span": span_path,
            "uuid": uuid_path,
            "rebuilt": False,
        }

    nltk = ensure_nltk_punkt()

    print("\n[BUILD] Creating uuid-aligned dev_eval.* files (one-time)...")

    with open(dev_json_path, "r", encoding="utf-8") as f:
        dev = json.load(f)
    data = dev["data"]

    # write atomically
    tmp_ctx  = ctx_path + ".tmp"
    tmp_q    = q_path + ".tmp"
    tmp_span = span_path + ".tmp"
    tmp_uuid = uuid_path + ".tmp"

    num_written = 0
    num_mappingprob = 0
    num_tokenprob = 0
    num_spanalignprob = 0

    with open(tmp_ctx, "w", encoding="utf-8") as fctx, \
         open(tmp_q, "w", encoding="utf-8") as fq, \
         open(tmp_span, "w", encoding="utf-8") as fspan, \
         open(tmp_uuid, "w", encoding="utf-8") as fuuid:

        for article in tqdm(data, desc="[BUILD] dev_eval"):
            for para in article["paragraphs"]:
                context = para["context"]
                context = context.replace("''", '" ').replace("``", '" ')
                context_tokens = tokenize_like_project(nltk, context)
                context_lc = context.lower()

                charloc2wordloc = get_char_word_loc_mapping(context_lc, context_tokens)
                if charloc2wordloc is None:
                    # discard all qas in this paragraph
                    num_mappingprob += len(para["qas"])
                    continue

                for qa in para["qas"]:
                    question = qa["question"]
                    question_tokens = tokenize_like_project(nltk, question)

                    ans_text = qa["answers"][0]["text"].lower()
                    ans_start_char = qa["answers"][0]["answer_start"]
                    ans_end_char = ans_start_char + len(ans_text)

                    # span alignment check
                    if context_lc[ans_start_char:ans_end_char] != ans_text:
                        num_spanalignprob += 1
                        continue

                    ans_start_word = charloc2wordloc[ans_start_char][1]
                    ans_end_word = charloc2wordloc[ans_end_char - 1][1]
                    if ans_end_word < ans_start_word:
                        num_spanalignprob += 1
                        continue

                    ans_tokens = context_tokens[ans_start_word:ans_end_word + 1]
                    # tokenization alignment check
                    if "".join(ans_tokens) != "".join(ans_text.split()):
                        num_tokenprob += 1
                        continue

                    # write one example
                    fctx.write(" ".join(context_tokens) + "\n")
                    fq.write(" ".join(question_tokens) + "\n")
                    fspan.write(f"{ans_start_word} {ans_end_word}\n")
                    fuuid.write(qa["id"] + "\n")
                    num_written += 1

    # finalize
    os.replace(tmp_ctx, ctx_path)
    os.replace(tmp_q, q_path)
    os.replace(tmp_span, span_path)
    os.replace(tmp_uuid, uuid_path)

    print("[BUILD] Done.")
    print("  written:", num_written)
    print("  discarded (mapping):", num_mappingprob)
    print("  discarded (tokenization span unaligned):", num_tokenprob)
    print("  discarded (char span misaligned):", num_spanalignprob)

    return {
        "context": ctx_path,
        "question": q_path,
        "span": span_path,
        "uuid": uuid_path,
        "rebuilt": True,
    }


# --------------------------
# Batching utilities (lightweight, uuid-aware)
# --------------------------
def sentence_to_token_ids(sentence: str, word2id: dict):
    toks = sentence.strip().split()
    ids = [word2id.get(w, UNK_ID) for w in toks]
    return toks, ids


def pad_to_len(seq, L, pad_id=PAD_ID):
    if len(seq) >= L:
        return seq[:L]
    return seq + [pad_id] * (L - len(seq))


def safe_span_to_text(context_tokens, s: int, e: int):
    s = int(max(0, s))
    e = int(min(len(context_tokens) - 1, e))
    if e < s:
        e = s
    toks = context_tokens[s:e+1]
    out = []
    for t in toks:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="ignore")
        out.append(str(t))
    return " ".join(out)


def iter_eval_batches(paths: dict, word2id: dict, batch_size: int, max_seq: int):
    """
    Yield batches with:
      context_ids: np.int32 [B, max_seq]
      qn_ids:      np.int32 [B, max_seq]
      context_tokens: list[list[str]]
      uuid: list[str]
      gold_span: np.int32 [B,2]
    """
    with open(paths["context"], "r", encoding="utf-8") as fctx, \
         open(paths["question"], "r", encoding="utf-8") as fq, \
         open(paths["span"], "r", encoding="utf-8") as fsp, \
         open(paths["uuid"], "r", encoding="utf-8") as fid:

        buf_ctx_ids, buf_q_ids = [], []
        buf_ctx_toks = []
        buf_uuid = []
        buf_span = []

        for ctx_line, q_line, sp_line, id_line in zip(fctx, fq, fsp, fid):
            ctx_tokens, ctx_ids = sentence_to_token_ids(ctx_line, word2id)
            q_tokens, q_ids = sentence_to_token_ids(q_line, word2id)
            gs, ge = [int(x) for x in sp_line.strip().split()]
            qid = id_line.strip()

            # truncate/pad to max_seq
            ctx_ids = pad_to_len(ctx_ids, max_seq)
            q_ids = pad_to_len(q_ids, max_seq)

            buf_ctx_ids.append(ctx_ids)
            buf_q_ids.append(q_ids)
            buf_ctx_toks.append(ctx_tokens)
            buf_uuid.append(qid)
            buf_span.append([gs, ge])

            if len(buf_uuid) == batch_size:
                yield {
                    "context_ids": np.array(buf_ctx_ids, dtype=np.int32),
                    "qn_ids": np.array(buf_q_ids, dtype=np.int32),
                    "context_tokens": buf_ctx_toks,
                    "uuid": buf_uuid,
                    "gold_span": np.array(buf_span, dtype=np.int32),
                }
                buf_ctx_ids, buf_q_ids, buf_ctx_toks, buf_uuid, buf_span = [], [], [], [], []

        # last partial
        if buf_uuid:
            yield {
                "context_ids": np.array(buf_ctx_ids, dtype=np.int32),
                "qn_ids": np.array(buf_q_ids, dtype=np.int32),
                "context_tokens": buf_ctx_toks,
                "uuid": buf_uuid,
                "gold_span": np.array(buf_span, dtype=np.int32),
            }


# --------------------------
# Main eval runner (safe saving + resume)
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="DCANW_E10.model")
    ap.add_argument("--dev_json", default="data/dev-v1.1.json")
    ap.add_argument("--glove_path", default="data/glove.840B.300d.txt")
    ap.add_argument("--max_seq", type=int, default=600)
    ap.add_argument("--hid", type=int, default=200)
    ap.add_argument("--maxout", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--decoder_it", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=500)  # save partial preds every N examples
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out_dir", default="artifacts_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    progress_path = os.path.join(args.out_dir, "progress.json")
    partial_path = os.path.join(args.out_dir, "preds_partial.json")
    final_pred_path = os.path.join(args.out_dir, "dcn_predictions.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    samples_path = os.path.join(args.out_dir, "samples.txt")

    # 1) build dev_eval.* with uuids
    dev_eval_paths = build_dev_eval_files(args.dev_json, out_dir="data")

    # 2) load GloVe
    print("\n[1/5] Loading GloVe ...")
    emb_mat, word2id, id2word = get_glove(args.glove_path, 300)
    print("[1/5] Emb shape:", emb_mat.shape)

    # 3) build model + load checkpoint
    print("\n[2/5] Building model ...")
    model = DynamicCoattentionNW(args.max_seq, args.hid, args.decoder_it, args.maxout, args.dropout, emb_mat)

    print("[2/5] Loading checkpoint:", args.checkpoint)
    ch.serializers.load_npz(args.checkpoint, model)
    print("[2/5] OK")

    # 4) resume state (optional)
    start_index = 0
    predictions = {}
    samples = []

    if os.path.exists(progress_path) and os.path.exists(partial_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            start_index = int(st.get("done", 0))
            with open(partial_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"\n[RESUME] Found progress: {start_index} examples already done. Resuming...")
        except Exception:
            start_index = 0
            predictions = {}

    # 5) inference loop (save partial)
    print("\n[3/5] Running inference over dev_eval ...")
    done = 0
    t0 = time.time()

    for batch in tqdm(iter_eval_batches(dev_eval_paths, word2id, args.batch, args.max_seq),
                      desc="DCN dev inference"):
        # Skip already processed examples (resume)
        bsz = len(batch["uuid"])
        if done + bsz <= start_index:
            done += bsz
            continue

        model.reset_state()

        c = ch.Variable(batch["context_ids"])
        q = ch.Variable(batch["qn_ids"])

        # IMPORTANT: model expects t to compute loss; give dummy span.
        dummy = np.zeros((c.shape[0], 2), dtype=np.int32)
        s_idx, e_idx, _ = model(c, q, dummy)  # s_idx/e_idx are arrays of positions already

        # ensure arrays
        s_idx = np.array(s_idx).reshape(-1)
        e_idx = np.array(e_idx).reshape(-1)

        for i in range(bsz):
            qid = batch["uuid"][i]
            pred_text = safe_span_to_text(batch["context_tokens"][i], int(s_idx[i]), int(e_idx[i]))
            predictions[qid] = pred_text

            if len(samples) < args.samples:
                gs, ge = map(int, batch["gold_span"][i])
                gold_text = safe_span_to_text(batch["context_tokens"][i], gs, ge)
                samples.append((qid, pred_text, gold_text))

        done += bsz

        # periodic save (crash-safe)
        if done % args.save_every < bsz:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"done": done, "checkpoint": args.checkpoint, "time_sec": int(time.time() - t0)}, f, indent=2)
            # also save samples quickly
            with open(samples_path, "w", encoding="utf-8") as f:
                for qid, p, g in samples:
                    f.write(f"QID: {qid}\nPRED: {p}\nGOLD: {g}\n{'-'*60}\n")

    # final save
    with open(final_pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(samples_path, "w", encoding="utf-8") as f:
        for qid, p, g in samples:
            f.write(f"QID: {qid}\nPRED: {p}\nGOLD: {g}\n{'-'*60}\n")

    print(f"\n[3/5] Saved final predictions → {final_pred_path}")

    # 6) official metrics
    print("\n[4/5] Computing official SQuAD v1.1 metrics ...")
    metrics = squad_v11_evaluate(args.dev_json, predictions)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== OFFICIAL RESULTS (SQuAD v1.1 dev) =====")
    print("Exact Match (EM):", round(metrics["exact_match"], 2))
    print("F1:", round(metrics["f1"], 2))
    print("Missing predictions:", metrics["missing_predictions"], "/", metrics["total"])
    print("Saved metrics →", metrics_path)
    print("Saved samples →", samples_path)
    print("\nDONE ✅")


if __name__ == "__main__":
    main()
