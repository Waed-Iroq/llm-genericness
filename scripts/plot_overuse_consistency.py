#!/usr/bin/env python3
import argparse, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def boot_mean_ci(x, boots=5000, seed=42, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(boots)])
    lo, hi = np.percentile(bs, [100*alpha/2, 100*(1-alpha/2)])
    return float(x.mean()), float(lo), float(hi)

def canon_id(pid: str) -> str:
    s = str(pid)
    s = re.sub(r'([._-]p\d+)$', '', s, flags=re.I)      # foo_p3 -> foo
    s = re.sub(r'([._-]?para\d+)$', '', s, flags=re.I)  # foo_para2 -> foo
    return s

def icc1k(M: np.ndarray) -> float:
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
    ap.add_argument("--metrics-by-prompt", required=True,
                    help="e.g., data/analysis/run_temp07/metrics_by_prompt.csv")
    ap.add_argument("--out-dir", required=True,
                    help="e.g., data/analysis/run_temp07")
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_by_prompt)

    # ---------- A) OVERUSE @ T=0.7 ----------
    cols = [c for c in [
        "pct_generic","pct_disc","pct_cap","pct_safety","pct_hedge",
        "pct_filler","pct_deferral","pct_univ","pct_struct"
    ] if c in df.columns]

    # Overall bar with 95% CI
    stats = []
    for m in cols:
        mean, lo, hi = boot_mean_ci(df[m].values, boots=args.boots)
        stats.append((m, mean, lo, hi))
    stats.sort(key=lambda x: x[1], reverse=True)

    labels = [s[0] for s in stats]
    means  = [s[1] for s in stats]
    los    = [s[2] for s in stats]
    his    = [s[3] for s in stats]
    errs   = [ (m - l, h - m) for (m,l,h) in zip(means,los,his) ]
    neg    = [e[0] for e in errs]
    pos    = [e[1] for e in errs]

    plt.figure(figsize=(9,5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=[neg,pos], capsize=4, alpha=0.8)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Percent of responses")
    plt.title("Overuse of generic patterns @ T=0.7 (mean ± 95% CI)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/overuse_overall_bar.png", dpi=200)
    plt.close()

    # Per-task %generic with CI
    per_task = []
    for task, g in df.groupby("task"):
        mean, lo, hi = boot_mean_ci(g["pct_generic"].values, boots=args.boots)
        per_task.append((task, mean, lo, hi, int(g.shape[0])))
    per_task.sort(key=lambda t: t[1], reverse=True)

    tlabels = [t[0] for t in per_task]
    tmeans  = [t[1] for t in per_task]
    tneg    = [t[1]-t[2] for t in per_task]
    tpos    = [t[3]-t[1] for t in per_task]
    tn      = [t[4] for t in per_task]

    plt.figure(figsize=(10,5))
    x = np.arange(len(tlabels))
    plt.bar(x, tmeans, yerr=[tneg,tpos], capsize=4, alpha=0.85)
    for i,(m,n) in enumerate(zip(tmeans, tn)):
        plt.text(i, m + 1.0, f"n={n}", ha="center", va="bottom", fontsize=8)
    plt.axhline(0, lw=1, ls="--", color="black")
    plt.xticks(x, tlabels, rotation=30, ha="right")
    plt.ylabel("% generic")
    plt.title("% generic by task @ T=0.7 (95% CI, bootstrap over prompts)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/overuse_by_task_bar.png", dpi=200)
    plt.close()

    # Heatmap: category breakdown by task (means only)
    cat_cols = [c for c in cols if c != "pct_generic"]
    if cat_cols:
        heat = df.groupby("task")[cat_cols].mean().loc[tlabels]  # order by tlabels
        data = heat.values
        plt.figure(figsize=(max(8, len(cat_cols)*1.2), max(4, len(tlabels)*0.5 + 2)))
        im = plt.imshow(data, aspect="auto")
        plt.colorbar(im, label="percent")
        plt.yticks(np.arange(len(tlabels)), tlabels)
        plt.xticks(np.arange(len(cat_cols)), cat_cols, rotation=30, ha="right")
        plt.title("Generic category breakdown by task @ T=0.7 (mean %)")
        # optional: annotate
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center", fontsize=7, color="white")
        plt.tight_layout()
        plt.savefig(f"{args.out_dir}/generic_heatmap_by_task.png", dpi=200)
        plt.close()

    # ---------- B) CONSISTENCY ACROSS PARAPHRASES ----------
    # Build base prompt ids
    df["canon_id"] = df["prompt_id"].astype(str).map(canon_id)
    if df["canon_id"].nunique() == df["prompt_id"].nunique() and "group" in df.columns:
        df["canon_id"] = df["prompt_id"].astype(str)  # fallback

    # Per-base SD across paraphrases (percentage points)
    rows = []
    for (task, cid), g in df.groupby(["task","canon_id"]):
        if g["prompt_id"].nunique() < 2: 
            continue
        vals = (g.groupby("prompt_id")["pct_generic"].mean()).values
        rows.append((task, cid, float(np.std(vals, ddof=1))))
    per_base = pd.DataFrame(rows, columns=["task","canon_id","sd_pp"])
    # Box/violin of SD by task
    if not per_base.empty:
        groups = [per_base.loc[per_base["task"]==t, "sd_pp"].values for t in tlabels if t in per_base["task"].unique()]
        plt.figure(figsize=(10,5))
        parts = plt.violinplot(groups, positions=np.arange(1, len(groups)+1), showmeans=True, showextrema=True)
        plt.xticks(np.arange(1, len(groups)+1), [t for t in tlabels if t in per_base["task"].unique()], rotation=30, ha="right")
        plt.ylabel("Within-base SD of %generic (pp)")
        plt.title("Consistency across paraphrases @ T=0.7 (lower = more consistent)")
        plt.tight_layout()
        plt.savefig(f"{args.out_dir}/consistency_sd_violin.png", dpi=200)
        plt.close()

        # ICC by task
        bars = []
        for t, g in df.groupby("task"):
            piv = g.pivot_table(index="canon_id", columns="prompt_id", values="pct_generic", aggfunc="mean")
            piv = piv.dropna(thresh=2)
            if piv.shape[0] >= 2 and piv.shape[1] >= 2:
                r = icc1k((piv.values/100.0))
                bars.append((t, r, piv.shape[0]))
        if bars:
            bars.sort(key=lambda x: (x[1] if x[1]==x[1] else -1), reverse=True)
            blabels = [b[0] for b in bars]
            brs     = [b[1] for b in bars]
            bn      = [b[2] for b in bars]
            x = np.arange(len(bars))
            plt.figure(figsize=(10,5))
            plt.bar(x, brs, alpha=0.85)
            for i,(r,n) in enumerate(zip(brs,bn)):
                plt.text(i, (r if r==r else 0) + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)
            plt.xticks(x, blabels, rotation=30, ha="right")
            plt.ylabel("ICC(1,k) across paraphrases")
            plt.title("Reliability of genericness across paraphrases by task")
            plt.ylim(-0.2,1.0)
            plt.axhline(0, lw=1, ls="--", color="black")
            plt.tight_layout()
            plt.savefig(f"{args.out_dir}/consistency_icc_bar.png", dpi=200)
            plt.close()

if __name__ == "__main__":
    main()
