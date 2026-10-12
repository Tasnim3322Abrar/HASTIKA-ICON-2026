"""
=====================================================================
HASTIKA DATASET — FULL TRAINING PIPELINE (Task A)
Kannada-English Code-Mixed Text Classification
=====================================================================

Pipeline stages (mirrors your flowchart):
  1. Dataset Inspection
  2. Label / Text Analysis
  3. Quality Control          -> duplicates, missing, label conflicts
  4. Text Normalization       -> URLs, mentions, hashtags, repeated chars,
                                  whitespace (Kannada/English/emoji/punct preserved)
  5. Train/Validation Split
  6. TF-IDF baseline
  7. BiLSTM
  8. Transformers (XLM-R, MuRIL, IndicBERT)
  9. Model Comparison
 10. Emoji Ablation (keep vs demojize-to-text)
 11. Error Analysis
 12. Final Fine-tuning (best config, more data/epochs)
 13. Inference + Submission

BEFORE RUNNING ON KAGGLE:
  - Kaggle notebook settings -> Internet: ON (needed to download pretrained
    weights the first time). If your competition requires Internet: OFF at
    submission time, pre-download the winning model into a Kaggle Dataset
    and point `--local_model_dir` at it instead of the hub name.
  - Enable GPU (Settings -> Accelerator -> GPU T4/P100).
  - Update CONFIG paths/column names below to match your actual CSV.

Install (run once in a notebook cell, not needed if Kaggle image has them):
    !pip -q install transformers datasets emoji sentencepiece accelerate
=====================================================================
"""

import os
import re
import json
import random
import warnings
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# 0. CONFIG
# =====================================================================
CONFIG = {
    "train_path": "/kaggle/input/hastika-dataset/train.csv",
    "test_path": "/kaggle/input/hastika-dataset/test.csv",
    "text_col": "text",
    "label_col": "label",
    "id_col": "id",
    "output_dir": "/kaggle/working",

    "seed": 42,
    "val_size": 0.15,

    # "keep"  -> leave emoji characters as-is
    # "text"  -> convert emoji to :text: description (emoji.demojize)
    "emoji_mode": "keep",
    "lowercase_english": True,
    "repeat_char_limit": 2,          # "soooo" -> "soo"

    "max_len": 128,                  # set from p95 of token lengths (see inspect step)
    "batch_size": 16,
    "eval_batch_size": 32,
    "epochs": 5,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "label_smoothing": 0.05,
    "use_class_weights": True,
    "grad_accum_steps": 1,

    "models_to_try": {
        "XLM-R": "xlm-roberta-base",
        "MuRIL": "google/muril-base-cased",
        "IndicBERT": "ai4bharat/indic-bert",
    },

    # after model comparison, set this manually (or it's auto-picked by script)
    "final_model_key": None,   # e.g. "MuRIL"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


set_seed(CONFIG["seed"])


# =====================================================================
# 1. DATASET INSPECTION
# =====================================================================
def inspect_dataset(df: pd.DataFrame, name: str = "train"):
    print(f"\n{'=' * 60}\nINSPECTING: {name}  shape={df.shape}\n{'=' * 60}")
    print(df.head(3))
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())

    if CONFIG["label_col"] in df.columns:
        print("\nLabel distribution:")
        print(df[CONFIG["label_col"]].value_counts())
        print(df[CONFIG["label_col"]].value_counts(normalize=True).round(3))

    text_col = CONFIG["text_col"]
    if text_col in df.columns:
        lens = df[text_col].astype(str).apply(lambda x: len(x.split()))
        print("\nText length (words) stats:")
        print(lens.describe())
        print(f"p90={lens.quantile(0.90):.0f}  p95={lens.quantile(0.95):.0f}  p99={lens.quantile(0.99):.0f}")
        print("-> use p95/p99 to set CONFIG['max_len'] for tokenization")

    # quick script mix check (Kannada vs Latin) — useful sanity check for code-mixed data
    def script_ratio(text):
        text = str(text)
        if not text:
            return 0.0
        kannada = sum(1 for c in text if "\u0C80" <= c <= "\u0CFF")
        return kannada / max(len(text), 1)

    if text_col in df.columns:
        ratios = df[text_col].apply(script_ratio)
        print(f"\nMean Kannada-script character ratio: {ratios.mean():.3f}")


# =====================================================================
# 2. QUALITY CONTROL
# =====================================================================
def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    text_col, label_col = CONFIG["text_col"], CONFIG["label_col"]
    n0 = len(df)

    # --- missing values ---
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    print(f"After dropping missing/empty: {n0} -> {len(df)}")

    # --- exact duplicate rows (same text + same label) ---
    n1 = len(df)
    df = df.drop_duplicates(subset=[text_col, label_col], keep="first")
    print(f"After exact duplicate removal: {n1} -> {len(df)}")

    # --- label conflicts: identical text, different labels ---
    n2 = len(df)
    grp = df.groupby(text_col)[label_col].nunique()
    conflicting_texts = grp[grp > 1].index
    if len(conflicting_texts) > 0:
        print(f"Found {len(conflicting_texts)} texts with conflicting labels -> resolving by majority vote")

        def resolve(group):
            counts = group[label_col].value_counts()
            top = counts.iloc[0]
            # tie -> drop all rows for this text (ambiguous label)
            if (counts == top).sum() > 1:
                return group.iloc[0:0]
            majority_label = counts.idxmax()
            row = group.iloc[[0]].copy()
            row[label_col] = majority_label
            return row

        conflict_df = df[df[text_col].isin(conflicting_texts)]
        resolved = conflict_df.groupby(text_col, group_keys=False).apply(resolve)
        df = pd.concat([df[~df[text_col].isin(conflicting_texts)], resolved], ignore_index=True)
    print(f"After conflict resolution: {n2} -> {len(df)}")

    df = df.reset_index(drop=True)
    return df


# =====================================================================
# 3. TEXT NORMALIZATION
# =====================================================================
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # 3+ repeats of any char
MULTI_WS_RE = re.compile(r"\s+")

try:
    import emoji as emoji_lib
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False


def normalize_text(text: str, emoji_mode: str = "keep") -> str:
    """
    - URLs      -> <URL>
    - Mentions  -> <USER>
    - Hashtags  -> '#word' -> 'word' (kept as normal token, not dropped)
    - Kannada, English, punctuation are left untouched
    - Emoji     -> kept as-is, OR converted to text description
    - Repeated characters collapsed to CONFIG['repeat_char_limit']
    - Whitespace collapsed
    """
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)

    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    text = HASHTAG_RE.sub(r"\1", text)

    if emoji_mode == "text" and HAS_EMOJI_LIB:
        text = emoji_lib.demojize(text, delimiters=(" [", "] "))
        text = text.replace("_", " ")

    if CONFIG["lowercase_english"]:
        # .lower() only affects ASCII/Latin letters; Kannada is caseless & unaffected
        text = text.lower()

    limit = CONFIG["repeat_char_limit"]
    text = REPEAT_RE.sub(lambda m: m.group(1) * limit, text)

    text = MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalize_dataframe(df: pd.DataFrame, emoji_mode: str) -> pd.DataFrame:
    df = df.copy()
    df["norm_text"] = df[CONFIG["text_col"]].apply(lambda t: normalize_text(t, emoji_mode))
    df = df[df["norm_text"].str.len() > 0].reset_index(drop=True)
    return df


# =====================================================================
# 4. TRAIN / VALIDATION SPLIT
# =====================================================================
def make_split(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=CONFIG["val_size"],
        random_state=CONFIG["seed"],
        stratify=df[CONFIG["label_col"]],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# =====================================================================
# 5. TF-IDF BASELINE
# =====================================================================
def run_tfidf_baseline(train_df, val_df, label_encoder):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    y_train = label_encoder.transform(train_df[CONFIG["label_col"]])
    y_val = label_encoder.transform(val_df[CONFIG["label_col"]])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",       # char n-grams handle code-mixed/noisy text well
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["norm_text"])
    X_val = vectorizer.transform(val_df["norm_text"])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if CONFIG["use_class_weights"] else None,
        C=1.0,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1_macro = f1_score(y_val, preds, average="macro")
    print(f"[TF-IDF+LogReg] acc={acc:.4f}  macro-F1={f1_macro:.4f}")
    return {"model": "TF-IDF+LogReg", "accuracy": acc, "macro_f1": f1_macro,
            "vectorizer": vectorizer, "clf": clf}


# =====================================================================
# 6. BiLSTM
# =====================================================================
def build_vocab(texts, min_freq=2):
    counter = Counter()
    tokenizer = lambda t: re.findall(r"\w+|[^\s\w]", t, re.UNICODE)
    for t in texts:
        counter.update(tokenizer(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counter.items():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab, tokenizer


def run_bilstm(train_df, val_df, label_encoder):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, tokenizer = build_vocab(train_df["norm_text"].tolist())
    vocab_size = len(vocab)

    def encode(text, max_len=CONFIG["max_len"]):
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokenizer(text)][:max_len]
        return ids if ids else [vocab["<UNK>"]]

    class TextDataset(Dataset):
        def __init__(self, df, label_encoder):
            self.texts = df["norm_text"].tolist()
            self.labels = label_encoder.transform(df[CONFIG["label_col"]])

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            ids = torch.tensor(encode(self.texts[idx]), dtype=torch.long)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return ids, label

    def collate_fn(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        padded = pad_sequence(seqs, batch_first=True, padding_value=vocab["<PAD>"])
        return padded, lengths, torch.stack(labels)

    train_ds = TextDataset(train_df, label_encoder)
    val_ds = TextDataset(val_df, label_encoder)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["eval_batch_size"], shuffle=False, collate_fn=collate_fn)

    num_classes = len(label_encoder.classes_)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=200, hidden_dim=128, num_classes=2, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                                 bidirectional=True, dropout=0.3)
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x, lengths):
            emb = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # masked mean pooling
            mask = (x != 0).unsqueeze(-1).float()
            summed = (out * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-6)
            pooled = summed / counts
            return self.fc(self.dropout(pooled))

    model = BiLSTMClassifier(vocab_size, num_classes=num_classes).to(device)

    if CONFIG["use_class_weights"]:
        counts = np.bincount(train_ds.labels, minlength=num_classes)
        weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CONFIG["label_smoothing"])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_f1 = -1
    best_state = None
    for epoch in range(min(CONFIG["epochs"], 10)):
        model.train()
        total_loss = 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths, y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x = x.to(device)
                logits = model(x, lengths)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        print(f"[BiLSTM] epoch {epoch+1} loss={total_loss/len(train_loader):.4f} "
              f"val_acc={acc:.4f} val_macroF1={f1_macro:.4f}")
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return {"model": "BiLSTM", "accuracy": acc, "macro_f1": best_f1,
            "state_dict": best_state, "vocab": vocab}


# =====================================================================
# 7. TRANSFORMERS (XLM-R / MuRIL / IndicBERT)
# =====================================================================
def run_transformer(model_name, train_df, val_df, label_encoder, num_epochs=None, extra_train_df=None):
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from datasets import Dataset as HFDataset
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    num_epochs = num_epochs or CONFIG["epochs"]
    num_classes = len(label_encoder.classes_)

    if extra_train_df is not None:
        train_df = pd.concat([train_df, extra_train_df], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_hf(df, has_labels=True):
        data = {"text": df["norm_text"].tolist()}
        if has_labels:
            data["label"] = label_encoder.transform(df[CONFIG["label_col"]]).tolist()
        return HFDataset.from_dict(data)

    train_hf = to_hf(train_df)
    val_hf = to_hf(val_df)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_len"])

    train_hf = train_hf.map(tokenize_fn, batched=True)
    val_hf = val_hf.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if CONFIG["use_class_weights"]:
        y_train = np.array(train_hf["label"])
        counts = np.bincount(y_train, minlength=num_classes)
        class_weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = class_weights.to(logits.device) if class_weights is not None else None
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=CONFIG["label_smoothing"])
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

    safe_name = model_name.replace("/", "_")
    run_dir = os.path.join(CONFIG["output_dir"], f"run_{safe_name}")

    args = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=CONFIG["lr"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        num_train_epochs=num_epochs,
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[{model_name}] {metrics}")

    return {
        "model": model_name,
        "accuracy": metrics["eval_accuracy"],
        "macro_f1": metrics["eval_macro_f1"],
        "weighted_f1": metrics["eval_weighted_f1"],
        "trainer": trainer,
        "tokenizer": tokenizer,
    }


# =====================================================================
# 8. MODEL COMPARISON
# =====================================================================
def compare_models(results: list):
    comp_df = pd.DataFrame([
        {"model": r["model"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
        for r in results
    ]).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print("\n=== MODEL COMPARISON ===")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(CONFIG["output_dir"], "model_comparison.csv"), index=False)
    return comp_df


# =====================================================================
# 9. EMOJI ABLATION
# =====================================================================
def run_emoji_ablation(best_model_name, raw_train_df, raw_val_df, label_encoder):
    results = []
    for mode in ["keep", "text"]:
        print(f"\n--- Emoji ablation: mode={mode} ---")
        tr = normalize_dataframe(raw_train_df, mode)
        va = normalize_dataframe(raw_val_df, mode)
        res = run_transformer(best_model_name, tr, va, label_encoder, num_epochs=max(2, CONFIG["epochs"] - 2))
        res["emoji_mode"] = mode
        results.append(res)

    abl_df = pd.DataFrame([{"emoji_mode": r["emoji_mode"], "macro_f1": r["macro_f1"],
                             "accuracy": r["accuracy"]} for r in results])
    print("\n=== EMOJI ABLATION RESULTS ===")
    print(abl_df.to_string(index=False))
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"Best emoji mode: {best['emoji_mode']}")
    return best["emoji_mode"], abl_df


# =====================================================================
# 10. ERROR ANALYSIS
# =====================================================================
def error_analysis(trainer, val_df, tokenizer, label_encoder):
    from sklearn.metrics import confusion_matrix, classification_report
    from datasets import Dataset as HFDataset

    val_hf = HFDataset.from_dict({"text": val_df["norm_text"].tolist()})
    val_hf = val_hf.map(lambda b: tokenizer(b["text"], truncation=True, max_length=CONFIG["max_len"]), batched=True)

    preds_output = trainer.predict(val_hf)
    preds = np.argmax(preds_output.predictions, axis=-1)
    true = label_encoder.transform(val_df[CONFIG["label_col"]])

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(true, preds))
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(true, preds, target_names=[str(c) for c in label_encoder.classes_]))

    val_df = val_df.copy()
    val_df["pred_label"] = label_encoder.inverse_transform(preds)
    misclassified = val_df[val_df[CONFIG["label_col"]] != val_df["pred_label"]]
    print(f"\nMisclassified: {len(misclassified)} / {len(val_df)}")
    print(misclassified[[CONFIG["text_col"], CONFIG["label_col"], "pred_label"]].head(20))
    misclassified.to_csv(os.path.join(CONFIG["output_dir"], "misclassified.csv"), index=False)
    return misclassified


# =====================================================================
# 11 & 12. FINAL FINE-TUNING + SUBMISSION
# =====================================================================
def final_fine_tune_and_predict(best_model_name, best_emoji_mode, full_train_df, val_df, test_df, label_encoder):
    tr = normalize_dataframe(full_train_df, best_emoji_mode)
    va = normalize_dataframe(val_df, best_emoji_mode)
    te = normalize_dataframe(test_df, best_emoji_mode)

    result = run_transformer(
        best_model_name, tr, va, label_encoder,
        num_epochs=CONFIG["epochs"] + 1,          # slightly longer final run
    )
    trainer, tokenizer = result["trainer"], result["tokenizer"]

    from datasets import Dataset as HFDataset
    test_hf = HFDataset.from_dict({"text": te["norm_text"].tolist()})
    test_hf = test_hf.map(lambda b: tokenizer(b["text"], truncation=True, max_length=CONFIG["max_len"]), batched=True)

    preds_output = trainer.predict(test_hf)
    preds = np.argmax(preds_output.predictions, axis=-1)
    pred_labels = label_encoder.inverse_transform(preds)

    sub = pd.DataFrame({
        CONFIG["id_col"]: test_df[CONFIG["id_col"]].values,
        CONFIG["label_col"]: pred_labels,
    })
    sub_path = os.path.join(CONFIG["output_dir"], "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    model_dir = os.path.join(CONFIG["output_dir"], "final_model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved final model to {model_dir}")

    return sub, result


# =====================================================================
# MAIN
# =====================================================================
def main():
    from sklearn.preprocessing import LabelEncoder

    # ---- load ----
    raw_train = pd.read_csv(CONFIG["train_path"])
    raw_test = pd.read_csv(CONFIG["test_path"])

    # ---- 1. inspection ----
    inspect_dataset(raw_train, "train")
    inspect_dataset(raw_test, "test")

    # ---- 2. quality control (train only; never drop test rows) ----
    clean_train = quality_control(raw_train)

    # ---- label encoder fit on cleaned train labels ----
    label_encoder = LabelEncoder()
    label_encoder.fit(clean_train[CONFIG["label_col"]])
    print("Classes:", list(label_encoder.classes_))

    # ---- 3. normalization (default emoji mode) ----
    norm_train = normalize_dataframe(clean_train, CONFIG["emoji_mode"])
    norm_test = normalize_dataframe(raw_test, CONFIG["emoji_mode"])

    # ---- 4. split ----
    train_df, val_df = make_split(norm_train)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    all_results = []

    # ---- 5. TF-IDF baseline ----
    all_results.append(run_tfidf_baseline(train_df, val_df, label_encoder))

    # ---- 6. BiLSTM ----
    all_results.append(run_bilstm(train_df, val_df, label_encoder))

    # ---- 7. Transformers ----
    transformer_results = {}
    for key, model_name in CONFIG["models_to_try"].items():
        res = run_transformer(model_name, train_df, val_df, label_encoder)
        transformer_results[key] = res
        all_results.append(res)

    # ---- 8. comparison ----
    comp_df = compare_models(all_results)

    # pick best transformer (not TF-IDF/BiLSTM) to carry forward
    best_key = max(transformer_results, key=lambda k: transformer_results[k]["macro_f1"])
    best_model_name = CONFIG["models_to_try"][best_key]
    print(f"\nBest transformer: {best_key} ({best_model_name})")

    # ---- 9. emoji ablation on best model ----
    best_emoji_mode, abl_df = run_emoji_ablation(best_model_name, clean_train, val_df, label_encoder)

    # ---- 10. error analysis on best transformer/emoji combo ----
    best_result = transformer_results[best_key]
    error_analysis(best_result["trainer"], val_df, best_result["tokenizer"], label_encoder)

    # ---- 11 & 12. final fine-tune + submission ----
    submission, final_result = final_fine_tune_and_predict(
        best_model_name, best_emoji_mode, clean_train, val_df, raw_test, label_encoder
    )

    print("\nDONE. Final macro-F1 on val:", final_result["macro_f1"])
    return submission


if __name__ == "__main__":
    main()
