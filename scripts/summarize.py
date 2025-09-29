#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap


def parse_args():
    p = argparse.ArgumentParser(description="Make figures from expanded metrics CSVs.")
    p.add_argument("--in-dir", default="data/analysis", help="Directory with metrics CSVs")
    p.add_argument("--out-dir", default="data/analysis", help="Directory to write PNGs")
    p.add_argument("--task-csv", default="metrics_by_task.csv", help="Task-level CSV filename")
    p.add_argument("--prompt-csv", default="metrics_by_prompt.csv", help="Prompt-level CSV filename")
    return p.parse_args()


def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


# ---- helpers to discover categories dynamically
SPECIAL_PCTS = {"pct_generic", "pct_not_generic"}

def get_all_pct_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("pct_")]

def get_generic_breakdown_cols(df: pd.DataFrame):
    """
    Return all pct_* columns EXCEPT the top-level ones (generic / not_generic).
    This lets the script adapt if you add more categories in metrics.py later.
    """
    return [c for c in get_all_pct_columns(df) if c not in SPECIAL_PCTS]


# ---- label helpers
def _pretty_labels(names, wrap=12, split_underscores=True):
    pretty = []
    for s in names:
        s = str(s)
        if split_underscores:
            s = s.replace("_", " ")
        pretty.append("\n".join(textwrap.wrap(s, width=wrap)) or s)
    return pretty

def _apply_xtick_format(ax, rotation=30, ha="right", fontsize=9):
    for lab in ax.get_xticklabels():
        lab.set_rotation(rotation)
        lab.set_ha(ha)
    ax.tick_params(axis="x", labelsize=fontsize)


# ---- plots
def bar_generic_by_task(df_task: pd.DataFrame, out_path: Path):
    df = df_task.sort_values("task").copy()
    tasks = df["task"].tolist()
    pretty = _pretty_labels(tasks, wrap=12, split_underscores=True)
    x = np.arange(len(df))

    has_generic = "pct_generic" in df.columns
    has_not_generic = "pct_not_generic" in df.columns

    plt.figure(figsize=(max(8.5, 0.6 * len(tasks)), 4.8))

    if has_generic and has_not_generic:
        width = 0.38
        g = df["pct_generic"].fillna(0).to_numpy()
        ng = df["pct_not_generic"].fillna(0).to_numpy()
        plt.bar(x - width / 2, g, width, label="% generic")
        plt.bar(x + width / 2, ng, width, label="% not_generic")
        plt.xticks(x, pretty)
        plt.ylabel("Percent (%)")
        plt.title("Generic vs Not-Generic by Task (avg across prompts)")
        plt.legend()
    elif has_generic:
        plt.bar(x, df["pct_generic"].fillna(0).to_numpy())
        plt.xticks(x, pretty)
        plt.ylabel("%")
        plt.title("% Generic by Task (avg across prompts)")
    elif has_not_generic:
        plt.bar(x, df["pct_not_generic"].fillna(0).to_numpy())
        plt.xticks(x, pretty)
        plt.ylabel("%")
        plt.title("% Not-Generic by Task (avg across prompts)")
    else:
        plt.close()
        return

    ax = plt.gca()
    _apply_xtick_format(ax, rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def stacked_generic_breakdown(df_task: pd.DataFrame, out_path: Path):
    """
    Stacked bars for ALL category pct_* columns EXCEPT pct_generic / pct_not_generic.
    Works with any number of categories.
    """
    cats = get_generic_breakdown_cols(df_task)
    if not cats:
        return

    df = df_task.sort_values("task").copy()
    tasks = df["task"].tolist()
    pretty = _pretty_labels(tasks, wrap=12, split_underscores=True)
    x = np.arange(len(tasks))
    bottoms = np.zeros(len(df))

    plt.figure(figsize=(max(9, 0.7 * len(tasks)), 5))
    for c in cats:
        vals = df[c].fillna(0).to_numpy()
        plt.bar(x, vals, bottom=bottoms, label=c.replace("pct_", "").replace("_", " ").title())
        bottoms += vals

    plt.xticks(x, pretty)
    ax = plt.gca()
    _apply_xtick_format(ax, rotation=30, ha="right", fontsize=9)
    plt.ylabel("% generic (by type)")
    plt.title("Genericness Breakdown by Task")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def boxplot_metric_by_task(df_prompt: pd.DataFrame, metric: str, title: str, out_path: Path):
    if metric not in df_prompt.columns:
        return
    df = df_prompt[["task", metric]].dropna()
    if df.empty:
        return

    # Stable alphabetical task order
    grouped = list(df.groupby("task"))
    labels = [name for name, _ in grouped]
    groups = [g[metric].values for _, g in grouped]
    pretty = _pretty_labels(labels, wrap=12, split_underscores=True)

    plt.figure(figsize=(max(7.5, 0.6 * len(labels)), 4.5))
    plt.boxplot(groups, labels=pretty, showfliers=False)
    ax = plt.gca()
    _apply_xtick_format(ax, rotation=30, ha="right", fontsize=9)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def heatmap_generic_breakdown(df_task: pd.DataFrame, out_path: Path):
    cats = get_generic_breakdown_cols(df_task)
    if not cats:
        return

    labels = [c.replace("pct_", "").replace("_", " ").title() for c in cats]
    df = df_task.set_index("task")[cats].fillna(0)

    plt.figure(figsize=(max(8, 1.1 * len(cats)), max(4.8, 0.5 * len(df.index))))
    im = plt.imshow(df.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="%")
    # pretty y-tick labels (wrap underscores)
    y_pretty = _pretty_labels(df.index.tolist(), wrap=16, split_underscores=True)
    plt.xticks(ticks=np.arange(len(cats)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(df.index)), labels=y_pretty)
    plt.title("Genericness Category Heatmap (avg % by task)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def scatter_bleu_vs_sbert(df_prompt: pd.DataFrame, out_path: Path):
    needed = {"self_bleu", "mean_sem_sim", "task"}
    if not needed.issubset(df_prompt.columns):
        return
    df = df_prompt[["task", "self_bleu", "mean_sem_sim"]].dropna()
    if df.empty:
        return

    plt.figure(figsize=(7.5, 5.5))
    for t in sorted(df["task"].unique()):
        d = df[df["task"] == t]
        plt.scatter(d["self_bleu"], d["mean_sem_sim"], s=20, alpha=0.6, label=t)
    plt.xlabel("Self-BLEU (↑ = more lexical sameness)")
    plt.ylabel("SBERT cosine (↑ = more semantic sameness)")
    plt.title("Self-BLEU vs SBERT (per prompt)")
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def scatter_bleu_vs_distinct2(df_prompt: pd.DataFrame, out_path: Path):
    needed = {"self_bleu", "distinct2", "task"}
    if not needed.issubset(df_prompt.columns):
        return
    df = df_prompt[["task", "self_bleu", "distinct2"]].dropna()
    if df.empty:
        return

    plt.figure(figsize=(7.5, 5.5))
    for t in sorted(df["task"].unique()):
        d = df[df["task"] == t]
        plt.scatter(d["self_bleu"], d["distinct2"], s=20, alpha=0.6, label=t)
    plt.xlabel("Self-BLEU (↑ = more lexical sameness)")
    plt.ylabel("Distinct-2 (↑ = more lexical diversity)")
    plt.title("Self-BLEU vs Distinct-2 (per prompt)")
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()




def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ensure_out(out_dir)

    task_csv = in_dir / args.task_csv
    prompt_csv = in_dir / args.prompt_csv

    if not task_csv.exists():
        raise FileNotFoundError(f"Missing {task_csv}. Run scripts/metrics.py first.")
    if not prompt_csv.exists():
        raise FileNotFoundError(f"Missing {prompt_csv}. Run scripts/metrics.py first.")

    df_task = pd.read_csv(task_csv)
    df_prompt = pd.read_csv(prompt_csv)

    # 1) Generic vs Not-Generic by task
    bar_generic_by_task(df_task, out_dir / "generic_by_task.png")

    # 2) Breakdown by type (stacked bars) — dynamic categories
    stacked_generic_breakdown(df_task, out_dir / "generic_breakdown_by_task.png")

    # 3) Self-BLEU by task (boxplot across prompts)
    boxplot_metric_by_task(
        df_prompt,
        metric="self_bleu",
        title="Self-BLEU by Task (per-prompt distribution)",
        out_path=out_dir / "self_bleu_by_task.png",
    )

    # 4) Distinct-2 by task (boxplot across prompts)
    boxplot_metric_by_task(
        df_prompt,
        metric="distinct2",
        title="Distinct-2 by Task (per-prompt distribution)",
        out_path=out_dir / "distinct2_by_task.png",
    )

    # 5) Heatmap of categories by task — dynamic categories
    heatmap_generic_breakdown(df_task, out_dir / "generic_heatmap_by_task.png")

    # 6) SBERT by task (boxplot) — only if present
    boxplot_metric_by_task(
        df_prompt,
        metric="mean_sem_sim",
        title="SBERT cosine similarity by Task (per-prompt distribution)",
        out_path=out_dir / "sbert_by_task.png",
    )

    # 7) Self-BLEU vs SBERT scatter — only if both present
    scatter_bleu_vs_sbert(df_prompt, out_dir / "bleu_vs_sbert_scatter.png")

    # 8) Self-BLEU vs Distinct-2 scatter — only if both present
    scatter_bleu_vs_distinct2(df_prompt, out_dir / "bleu_vs_distinct2_scatter.png")



if __name__ == "__main__":
    main()
