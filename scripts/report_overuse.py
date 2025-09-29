#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd

METRICS = ["pct_generic","pct_disc","pct_cap","pct_safety","pct_hedge","pct_filler","pct_deferral","pct_univ","pct_struct"]

def boot_mean_ci(x, boots=5000, seed=42, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(boots)])
    lo, hi = np.percentile(bs, [100*alpha/2, 100*(1-alpha/2)])
    return x.mean(), lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-by-prompt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_by_prompt)
    cols = [c for c in METRICS if c in df.columns]
    if not cols: raise SystemExit("No genericness columns found.")

    # Per task
    rows = []
    for task, g in df.groupby("task"):
        for m in cols:
            mean, lo, hi = boot_mean_ci(g[m].values, boots=args.boots)
            rows.append({"task": task, "metric": m, "mean": mean, "ci_low": lo, "ci_high": hi, "n_prompts": int(g.shape[0])})
    by_task = pd.DataFrame(rows).sort_values(["metric","task"])

    # Overall (across prompts, unweighted)
    rows = []
    for m in cols:
        mean, lo, hi = boot_mean_ci(df[m].values, boots=args.boots)
        rows.append({"metric": m, "overall_mean": mean, "ci_low": lo, "ci_high": hi, "n_prompts": int(df.shape[0])})
    overall = pd.DataFrame(rows).sort_values("metric")

    # Save
    out_task = f"{args.out_dir}/overuse_by_task.csv"
    out_over = f"{args.out_dir}/overuse_overall.csv"
    by_task.to_csv(out_task, index=False)
    overall.to_csv(out_over, index=False)
    print(f"Wrote {out_task}\nWrote {out_over}")

if __name__ == "__main__":
    main()
