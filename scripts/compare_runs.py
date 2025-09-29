#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYS = ["pct_generic","pct_not_generic","distinct2","self_bleu","mean_sem_sim"]


def delta_tasks(tA, tB):
    A = tA.set_index("task")
    B = tB.set_index("task")
    both = A.join(B, lsuffix="_A", rsuffix="_B", how="inner")

    out = both.reset_index()[["task"]].copy()
    for k in KEYS:
        a, b = f"{k}_A", f"{k}_B"
        if a in both.columns and b in both.columns:
            out[f"Δ_{k}"] = both[b] - both[a]
    return out




def parse_args():
    p = argparse.ArgumentParser(description="Compare two analysis runs and plot deltas with CIs.")
    p.add_argument("--run-a", required=True, help="Dir with metrics_by_task.csv & metrics_by_prompt.csv (baseline)")
    p.add_argument("--run-b", required=True, help="Dir with metrics_by_task.csv & metrics_by_prompt.csv (experiment)")
    p.add_argument("--label-a", default="temp0.2")
    p.add_argument("--label-b", default="temp0.7")
    p.add_argument("--out-dir", default="data/analysis/compare")
    p.add_argument("--boots", type=int, default=2000, help="Bootstrap reps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load(run_dir):
    run_dir = Path(run_dir)
    t = pd.read_csv(run_dir/"metrics_by_task.csv")
    p = pd.read_csv(run_dir/"metrics_by_prompt.csv")
    return t, p



def bootstrap_prompt_delta(pA, pB, metric, rng, per_task=True):
    # Align on (task, prompt_id)
    need = {"task","prompt_id",metric}
    if not need.issubset(pA.columns) or not need.issubset(pB.columns):
        return None
    A = pA[list(need)].dropna().rename(columns={metric: metric+"_A"})
    B = pB[list(need)].dropna().rename(columns={metric: metric+"_B"})
    M = A.merge(B, on=["task","prompt_id"], how="inner").dropna()
    if M.empty: return None

    def _boot_group(df):
        # resample prompt_ids with replacement within the group
        ids = df["prompt_id"].unique()
        choice = rng.choice(ids, size=len(ids), replace=True)
        S = df[df["prompt_id"].isin(choice)]
        return (S[metric+"_B"].mean() - S[metric+"_A"].mean())

    if per_task:
        rows = []
        for task, g in M.groupby("task"):
            if g.empty: continue
            vals = [_boot_group(g) for _ in range(boots)]
            rows.append((task, np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)))
        return pd.DataFrame(rows, columns=["task", f"Δ_{metric}_mean", f"Δ_{metric}_ci_lo", f"Δ_{metric}_ci_hi"])
    else:
        # pooled across tasks
        vals = [_boot_group(M) for _ in range(boots)]
        return pd.Series({
            f"Δ_{metric}_mean": np.mean(vals),
            f"Δ_{metric}_ci_lo": np.percentile(vals, 2.5),
            f"Δ_{metric}_ci_hi": np.percentile(vals, 97.5),
        })

def bar_with_ci(df, metric_key, title, out_path):
    # df columns: task, Δ_xxx_mean, Δ_xxx_ci_lo, Δ_xxx_ci_hi
    m = f"Δ_{metric_key}_mean"
    lo = f"Δ_{metric_key}_ci_lo"
    hi = f"Δ_{metric_key}_ci_hi"
    if not {m,lo,hi,"task"}.issubset(df.columns): return
    df = df.sort_values(m)
    x = np.arange(len(df))
    y = df[m].values
    yerr = np.vstack([y - df[lo].values, df[hi].values - y])
    plt.figure(figsize=(max(8, 0.5*len(df)), 5))
    plt.bar(x, y, color="#4e79a7")
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=1)
    plt.xticks(x, df["task"], rotation=30, ha="right")
    plt.axhline(0, color="grey", lw=1)
    plt.ylabel(f"Delta ({metric_key})  [RunB − RunA]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tA, pA = load(args.run_a)
    tB, pB = load(args.run_b)

    # 1) Quick deltas by task (point estimates)
    deltas = delta_tasks(tA, tB)
    deltas.to_csv(out_dir/"deltas_by_task.csv", index=False)

    # 2) Bootstrap 95% CIs per task (prompt-level resampling)
    rng = np.random.default_rng(args.seed)
    global boots; boots = args.boots  # used in helper

    per_task_tables = []
    for metric in ["pct_generic","distinct2","self_bleu","mean_sem_sim"]:
        tbl = bootstrap_prompt_delta(pA, pB, metric, rng, per_task=True)
        if tbl is not None:
            per_task_tables.append(tbl)
    if per_task_tables:
        ci_merged = per_task_tables[0]
        for tbl in per_task_tables[1:]:
            ci_merged = ci_merged.merge(tbl, on="task", how="outer")
        ci_merged.to_csv(out_dir/"bootstrap_ci_by_task.csv", index=False)

        # 3) CI bar plots for the key metrics
        for metric, title in [
            ("pct_generic", "Δ %Generic (temp0.7 − temp0.2)"),
            ("distinct2",   "Δ Distinct-2 (temp0.7 − temp0.2)"),
            ("self_bleu",   "Δ Self-BLEU (temp0.7 − temp0.2)"),
            ("mean_sem_sim","Δ SBERT cosine (temp0.7 − temp0.2)"),
        ]:
            if f"Δ_{metric}_mean" in ci_merged.columns:
                bar_with_ci(ci_merged[["task", f"Δ_{metric}_mean", f"Δ_{metric}_ci_lo", f"Δ_{metric}_ci_hi"]],
                            metric, title, out_dir/f"delta_{metric}.png")

    print(f"[compare] Wrote: {out_dir}/deltas_by_task.csv")
    if per_task_tables:
        print(f"[compare] Wrote: {out_dir}/bootstrap_ci_by_task.csv and delta_*.png")
