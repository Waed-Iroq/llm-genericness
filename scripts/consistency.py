#!/usr/bin/env python3
import argparse, re, numpy as np, pandas as pd

def canon_id(pid: str) -> str:
    s = str(pid)
    s = re.sub(r'([._-]p\d+)$', '', s, flags=re.I)      # foo_p3 -> foo
    s = re.sub(r'([._-]?para\d+)$', '', s, flags=re.I)  # foo_para2 -> foo
    return s

def icc1k(M: np.ndarray) -> float:
    # Shrout & Fleiss ICC(1,k) on rows=targets (base prompts), cols=raters (paraphrases)
    if M.size == 0 or M.shape[1] < 2: return np.nan
    n, k = M.shape
    row_means = M.mean(axis=1)
    grand = row_means.mean()
    MSB = k * ((row_means - grand)**2).sum() / (n - 1) if n > 1 else np.nan
    MSW = ((M - row_means[:, None])**2).sum() / (n*(k-1)) if k > 1 else np.nan
    denom = MSB + (k - 1) * MSW
    return (MSB - MSW) / denom if denom and denom != 0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-by-prompt", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_by_prompt)
    if "pct_generic" not in df.columns:
        raise SystemExit("pct_generic not found in metrics_by_prompt.csv")

    # derive base prompt id if you have paraphrase ids like foo_p1, foo_p2; else fallback to group
    df["canon_id"] = df["prompt_id"].astype(str).map(canon_id)
    if df["canon_id"].nunique() == df["prompt_id"].nunique() and "group" in df.columns:
        # no paraphrase pattern detected; treat `group` as paraphrase and `prompt_id` as base
        df["canon_id"] = df["prompt_id"].astype(str)

    # per-base stats
    rows = []
    for (task, cid), g in df.groupby(["task","canon_id"]):
        # need ≥2 paraphrases for consistency
        if g["prompt_id"].nunique() < 2: 
            continue
        vals = (g.groupby("prompt_id")["pct_generic"].mean()/100.0).values  # per paraphrase
        rows.append({
            "task": task,
            "canon_id": cid,
            "k_paraphrases": len(vals),
            "mean_pct_generic": vals.mean()*100,
            "sd_paraphrases_pp": vals.std(ddof=1)*100
        })
    per_base = pd.DataFrame(rows)

    # task-level consistency summary (+ ICC)
    task_stats = []
    for task, tdf in df.groupby("task"):
        piv = tdf.pivot_table(index="canon_id", columns="prompt_id", values="pct_generic", aggfunc="mean")
        piv = piv.dropna(thresh=2)  # keep base prompts with ≥2 paraphrases
        icc = icc1k((piv.values/100.0))
        sd_pp = per_base.loc[per_base["task"]==task, "sd_paraphrases_pp"].median() if not per_base.empty else np.nan
        task_stats.append({"task": task, "median_within_base_sd_pp": sd_pp, "icc_paraphrases": icc, "n_bases": int(piv.shape[0])})
    by_task = pd.DataFrame(task_stats).sort_values("task")

    # overall summary
    overall_icc = icc1k(pd.concat([df.pivot_table(index="canon_id", columns="prompt_id", values="pct_generic", aggfunc="mean") for _, df in df.groupby("task")], axis=0).dropna(thresh=2).values/100.0)
    overall = {
        "overall_median_within_base_sd_pp": per_base["sd_paraphrases_pp"].median() if not per_base.empty else np.nan,
        "overall_icc_paraphrases": overall_icc,
        "n_bases_total": int(per_base["canon_id"].nunique() if not per_base.empty else 0)
    }

    if args.out_csv:
        with pd.ExcelWriter(args.out_csv.replace(".csv",".xlsx")) as xl:
            per_base.to_excel(xl, "per_base", index=False)
            by_task.to_excel(xl, "by_task", index=False)

    print("\n=== Consistency across paraphrases ===")
    print(pd.DataFrame([overall]))
    print("\nBy task:")
    print(by_task)

if __name__ == "__main__":
    main()
