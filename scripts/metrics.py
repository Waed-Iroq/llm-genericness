#!/usr/bin/env python3
import argparse, glob, json, os, sys
from pathlib import Path
from collections import defaultdict, Counter
import datetime, hashlib, shutil, subprocess


import numpy as np
import pandas as pd
import regex as re

# Optional deps (SBERT)
_SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    _SBERT_AVAILABLE = False

# NLTK (BLEU/self-BLEU)
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
SMOOTH = SmoothingFunction().method1


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Compute diversity + genericness metrics over the latest responses file.")
    p.add_argument("--responses-glob", default="data/responses/*.jsonl",
                   help="Glob for responses jsonl files (default: data/responses/*.jsonl)")
    p.add_argument("--out-dir", default="data/analysis",
                   help="Directory to write CSVs (default: data/analysis)")
    p.add_argument("--min-per-prompt", type=int, default=2,
                   help="Minimum responses per prompt to compute set metrics (default: 2)")
    p.add_argument("--skip-sbert", action="store_true",
                   help="Skip SBERT semantic similarity even if installed")
    p.add_argument("--topk", type=int, default=5,
                   help="Show top-k template patterns per task in stdout (default: 5)")
    return p.parse_args()


# =========================
# I/O utils
# =========================
def latest_file(glob_pat: str) -> Path:
    files = [Path(p) for p in glob.glob(glob_pat)]
    if not files:
        print(f"[metrics] No files matched: {glob_pat}", file=sys.stderr)
        sys.exit(1)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _git_rev_short():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_load_generation_config(resp_path: Path) -> dict:
    """
    Look for a generation config next to the responses file:
    e.g., if responses are under data/responses/run_temp02/*.jsonl,
    we try data/responses/run_temp02/config.json.
    """
    gen_cfg = {}
    try:
        cfg_path = resp_path.parent / "config.json"
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                gen_cfg = json.load(f)
    except Exception:
        pass
    return gen_cfg

def _save_analysis_config(out_dir: Path, resp_path: Path, args, df: pd.DataFrame, sbert_used: bool, gen_cfg: dict):
    """
    Write a single config.json into the analysis output directory that contains:
      - generation config (if found)
      - analysis metadata (timestamp, git commit, args, input checksum, row counts)
    """
    meta = {
        "analysis": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "date": datetime.date.today().isoformat(),
            "git_commit": _git_rev_short(),
            "script": "metrics.py",
            "responses_glob": args.responses_glob,
            "resolved_responses_path": str(resp_path),
            "responses_sha256": _sha256(resp_path),
            "min_per_prompt": args.min_per_prompt,
            "skip_sbert": bool(args.skip_sbert),
            "sbert_used": bool(sbert_used),
            "topk": args.topk,
            "n_rows": int(len(df)),
            "n_tasks": int(df["task"].nunique() if "task" in df.columns else 0),
        }
    }
    # Merge generation config (if any) under "generation"
    cfg = {"generation": gen_cfg, **meta}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)



def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL at {path}: {e}")
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return pd.DataFrame(rows)


# =========================
# Text normalization
# =========================
def normalize(txt: str) -> str:
    if not txt:
        return ""
    t = txt.lower()
    t = (t.replace("’", "'")
           .replace("‘", "'")
           .replace("“", '"')
           .replace("”", '"'))
    t = " ".join(t.split())
    return t


# =========================
# Diversity metrics
# =========================
def distinct_n(texts, n=2) -> float:
    all_ngrams = []
    for t in texts:
        toks = normalize(t).split()
        all_ngrams += list(ngrams(toks, n))
    unique = len(set(all_ngrams))
    total = len(all_ngrams) if all_ngrams else 1
    return unique / total


def self_bleu(texts) -> float:
    scores = []
    tokenized = [normalize(t).split() for t in texts]
    for i, hyp in enumerate(tokenized):
        refs = [r for j, r in enumerate(tokenized) if j != i]
        if not refs:
            continue
        scores.append(sentence_bleu(refs, hyp, smoothing_function=SMOOTH))
    return float(np.mean(scores)) if scores else float("nan")


def mean_pairwise_cosine(texts, model=None) -> float:
    if len(texts) < 2 or model is None:
        return float("nan")
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    cs = cosine_similarity(embs)
    tri = cs[np.triu_indices_from(cs, k=1)]
    return float(tri.mean()) if len(tri) else float("nan")


# =========================
# Expanded regex taxonomy
# =========================
DISCLAIMER_PATTERNS = [
    r"\bas an ai\b",
    r"\bas an ai language model\b",
    r"\bi (?:do|don't|cannot|can’t|can't|am not able to) (?:have|provide|offer) (?:personal|subjective) (?:opinions|feelings|beliefs)\b",
    r"\bi (?:lack|do not have|don't have) (?:personal )?experiences\b",
    r"\bi am (?:just|simply) an? (?:ai|assistant)\b",
    r"\bi am not a substitute for\b",
    r"\bi(?:'| a)m (?:here|designed) to provide (?:information|guidance)\b",
]

CAPABILITY_PATTERNS = [
    r"\bi (?:don'?t|do not|cannot|can’t) access (?:the|real\-?time) internet\b",
    r"\bi (?:don'?t|do not) have (?:real\-?time|live) (?:data|information|updates)\b",
    r"\bi (?:can(?:not|’t)|cannot) (?:browse|search) the web\b",
    r"\bi (?:can(?:not|’t)|cannot) view (?:images|files|attachments)\b",
    r"\bi (?:don'?t|do not) have access to your (?:personal|private) data\b",
]

SAFETY_PATTERNS = [
    r"\bi (?:can(?:not|’t)|cannot|am not allowed to) provide (?:medical|legal|financial|professional) advice\b",
    r"\bthis (?:is|would be) (?:medical|legal|financial) advice\b",
    r"\bfor (?:medical|legal|financial) questions, consult (?:a|an) (?:professional|specialist)\b",
    r"\bseek (?:professional|medical|legal) (?:help|advice|guidance)\b",
]

HEDGE_PATTERNS = [
    r"\bit depends(?: on|,|\.)?",
    r"\bthere('?s| is| are) no (?:single|one) right answer\b",
    r"\bmany factors (?:could|can|may) (?:influence|affect)\b",
    r"\bcontext (?:is|can be|often is) important\b",
    r"\bin some cases\b",
    r"\bon the other hand\b",
    r"\bvaries (?:depending|based) on\b",
    r"\bul?ltimately(,|\b)",
    r"\bin the end(,|\b)",
    r"\bit (?:might|may|can) be (?:helpful|useful)\b",
]

FILLER_PATTERNS = [
    r"^(?:that'?s|that is) a great question\b",
    r"^i understand your (?:concern|situation|question)\b",
    r"^in general(,|\b)",
    r"\bit'?s important to (?:remember|note)\b",
    r"\bit'?s worth (?:noting|considering)\b",
    r"\bkeep in mind(,|\b)",
    r"\ba common (?:approach|strategy|recommendation) is\b",
    r"\bpeople often (?:find|say|recommend)\b",
    r"^here(?:'| )?s (?:a|one) (?:simple|quick) (?:guide|overview)\b",
]

DEFERRAL_PATTERNS = [
    r"\bonly you can (?:decide|determine)\b",
    r"\byou (?:should|must|have to) decide what(?:'s| is) best\b",
    r"\bit'?s (?:a|an) personal decision\b",
    r"\byou'?ll need to weigh (?:the pros and cons|your options)\b",
    r"\bconsider (?:what|which) (?:matters|works) best for you\b",
]

UNIVERSAL_ADVICE_PATTERNS = [
    r"\bstart small\b",
    r"\bset (?:clear|specific) goals\b",
    r"\bbuild (?:a|an) routine\b",
    r"\bstay consistent\b",
    r"\bfind what works (?:best )?for you\b",
    r"\btrack your progress\b",
    r"\bcelebrate small wins\b",
    r"\bseek support (?:from|through)\b",
    r"\bremember to take breaks\b",
    r"\bprioritize self\-?care\b",
]

STRUCTURE_PATTERNS = [
    r"\bhere (?:are|'?s) (?:a few|some) (?:steps|tips|ideas|suggestions):?",
    r"\bconsider the following\b",
    r"\btry the following\b",
    r"\byou can follow these steps\b",
    r"\blet'?s break it down\b",
    r"\bfor example:(?:\s*[-*1\.])?",
]


def _compile_all(patts): 
    return [re.compile(p, re.I) for p in patts]


DISCLAIMER = _compile_all(DISCLAIMER_PATTERNS)
CAPABILITY = _compile_all(CAPABILITY_PATTERNS)
SAFETY     = _compile_all(SAFETY_PATTERNS)
HEDGE      = _compile_all(HEDGE_PATTERNS)
FILLER     = _compile_all(FILLER_PATTERNS)
DEFERRAL   = _compile_all(DEFERRAL_PATTERNS)
UNIV       = _compile_all(UNIVERSAL_ADVICE_PATTERNS)
STRUCT     = _compile_all(STRUCTURE_PATTERNS)


def any_match(text, patterns):
    s = normalize(text)
    return any(p.search(s) for p in patterns)


def tag_generic_flags(df: pd.DataFrame) -> pd.DataFrame:
    def _tag(row):
        t = row.get("response", "") or ""
        row["disc"]     = any_match(t, DISCLAIMER)
        row["cap"]      = any_match(t, CAPABILITY)
        row["safety"]   = any_match(t, SAFETY)
        row["hedge"]    = any_match(t, HEDGE)
        row["filler"]   = any_match(t, FILLER)
        row["deferral"] = any_match(t, DEFERRAL)
        row["univ"]     = any_match(t, UNIV)
        row["struct"]   = any_match(t, STRUCT)

        # existing flag
        row["generic_any"] = any(
            row[k] for k in ["disc","cap","safety","hedge","filler","deferral","univ","struct"]
        )

        # === NEW: not_generic + dominant category ===
        row["not_generic"] = not row["generic_any"]

        # pick one dominant category if generic, else "not_generic"
        priority = ["disc","cap","safety","hedge","filler","deferral","univ","struct"]
        dom = "not_generic"
        for c in priority:
            if row[c]:
                dom = c
                break
        row["dominant_category"] = dom

        return row

    return df.apply(_tag, axis=1)



# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Ensure NLTK punkt exists
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    resp_path = latest_file(args.responses_glob)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metrics] Using responses file: {resp_path}")
    df = read_jsonl(resp_path)
    # Try to load generation-time config (if responses folder has config.json)
    gen_cfg = _maybe_load_generation_config(resp_path)
    if gen_cfg:
        print("[metrics] Found generation config next to responses; will merge into analysis config.")
    else:
        print("[metrics] No generation config found; analysis config will include only analysis metadata.")


    # Required fields
    required = {"task", "prompt_id", "response"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in responses: {missing}")

    if "group" not in df.columns:
        df["group"] = "default"

    # Tag genericness
    df = tag_generic_flags(df)

    # Optional SBERT
    sbert_model = None
    if _SBERT_AVAILABLE and not args.skip_sbert:
        try:
            sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            print("[metrics] SBERT not available; skipping semantic similarity.", file=sys.stderr)

    # ---- Aggregate by (task, group, prompt_id)
    rows = []
    for (task, group, prompt_id), g in df.groupby(["task", "group", "prompt_id"], dropna=False):
        texts = [str(x) for x in g["response"].tolist()]
        rec = {
            "task": task,
            "group": group,
            "prompt_id": prompt_id,
            "n": len(texts),
            "pct_generic": 100 * g["generic_any"].mean(),
            "pct_not_generic": 100 * g["not_generic"].mean(),   # <-- NEW
            "pct_disc":     100 * g["disc"].mean(),
            "pct_cap":      100 * g["cap"].mean(),
            "pct_safety":   100 * g["safety"].mean(),
            "pct_hedge":    100 * g["hedge"].mean(),
            "pct_filler":   100 * g["filler"].mean(),
            "pct_deferral": 100 * g["deferral"].mean(),
            "pct_univ":     100 * g["univ"].mean(),
            "pct_struct":   100 * g["struct"].mean(),
            "distinct1":    distinct_n(texts, n=1),
            "distinct2":    distinct_n(texts, n=2),
            "self_bleu":    self_bleu(texts) if len(texts) >= args.min_per_prompt else float("nan"),
            "mean_sem_sim": mean_pairwise_cosine(texts, model=sbert_model) if len(texts) >= args.min_per_prompt else float("nan"),
        }
        rows.append(rec)

    by_prompt = pd.DataFrame(rows).sort_values(["task","group","prompt_id"])
    out1 = out_dir / "metrics_by_prompt.csv"
    by_prompt.to_csv(out1, index=False)
    print(f"[metrics] Wrote {out1}")

    # ---- Aggregate by task
    agg_map = {
        "pct_generic": "mean",
        "pct_not_generic": "mean",   # <-- NEW
        "pct_disc": "mean",
        "pct_cap": "mean",
        "pct_safety": "mean",
        "pct_hedge": "mean",
        "pct_filler": "mean",
        "pct_deferral": "mean",
        "pct_univ": "mean",
        "pct_struct": "mean",
        "distinct1": "mean",
        "distinct2": "mean",
        "self_bleu": "mean",
        "mean_sem_sim": "mean",
    }
    by_task = by_prompt.groupby("task").agg(agg_map).reset_index()
    out2 = out_dir / "metrics_by_task.csv"
    by_task.to_csv(out2, index=False)
    print(f"[metrics] Wrote {out2}")

    # ---- Quick preview
    with pd.option_context('display.max_columns', None):
        print("\n[metrics] Summary by task:\n", by_task)

    # ---- Top templates per task (preview)
    def top_templates_for_group(texts, topk):
        counters = defaultdict(Counter)
        for t in texts:
            s = normalize(t)
            for lab, pats in [
                ("disc", DISCLAIMER), ("cap", CAPABILITY), ("safety", SAFETY),
                ("hedge", HEDGE), ("filler", FILLER), ("deferral", DEFERRAL),
                ("univ", UNIV), ("struct", STRUCT),
            ]:
                for p in pats:
                    if p.search(s):
                        counters[lab][p.pattern] += 1
        # return top-k per label (pattern + count)
        return {lab: counters[lab].most_common(topk) for lab in counters}

    print("\n[metrics] Top templates per task (pattern, count):")
    for task, g in df.groupby("task"):
        texts = g["response"].astype(str).tolist()
        tops = top_templates_for_group(texts, args.topk)
        print(f"\n  Task = {task}")
        for lab in ["disc","cap","safety","hedge","filler","deferral","univ","struct"]:
            if lab in tops and len(tops[lab]) > 0:
                print(f"    {lab:8s}: {tops[lab]}")
            else:
                print(f"    {lab:8s}: []")

                
    sbert_used = sbert_model is not None
    # ---- Save merged config into the analysis folder
    _save_analysis_config(out_dir=out_dir, resp_path=resp_path, args=args,
                      df=df, sbert_used=sbert_used, gen_cfg=gen_cfg)
    print(f"[metrics] Wrote {out_dir / 'config.json'}")



if __name__ == "__main__":
    # Ensure punkt is present
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    main()
