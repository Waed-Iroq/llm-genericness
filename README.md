# Do LLMs Overuse Generic Responses?

📌 **Tel Aviv University – NLP Final Project (2025)**  
**Students:**

- Waed Iraki
- Juan Shehady

---

## 📖 Project Overview

This project investigates whether modern large language models (LLMs), such as **GPT-3.5**, tend to **overuse generic and templated responses** (e.g., _"As an AI language model…"_, _"It depends…"_) in open-ended tasks like advice or explanations.

We study **genericness**, **diversity**, and **consistency** across multiple tasks and paraphrased prompts.

---

## 🎯 Research Question

> _Do large language models overuse generic responses when answering open-ended prompts, and how consistent is this behavior across paraphrased prompts and task types?_

---

## 🛠️ Methodology

- **Task Types:** writing help, moral reasoning, everyday advice, creative tasks, etc.
- **Prompt Design:** multiple paraphrased prompts per task.
- **Models Queried:** GPT-3.5 via API.
- **Analysis Metrics:**
  - _Lexical diversity:_ distinct-1, distinct-2
  - _Repetition:_ Self-BLEU
  - _Semantic similarity:_ SBERT cosine similarity
  - _Genericness:_ phrase overlap analysis

---

## 📂 Repository Structure

│
├── data/
│ ├── prompts/ # Original & paraphrased prompts
│ ├── responses/ # Collected model responses
│ └── analysis/ # Processed results, CSVs, plots
│
├── scripts/
│ ├── collect.py # Query models with prompts
│ ├── metrics.py # Compute distinct-n, BLEU, etc.
│ ├── summarize.py # Summarize results
│ ├── compare_runs.py # Compare across runs
│ ├── consistency.py # Analyze consistency
│ ├── plot_overuse_consistency.py
│ └── report_overuse.py
│
├── requirements.txt # Python dependencies
├── .env # Template for environment variables
├── .gitignore

## Collect Reponses

python scripts/collect.py --prompts data/prompts/randomPrompts/everyday_advice.jsonl \
 --output data/responses/temp07/gpt35_t07_seed42.jsonl \
 --model gpt-3.5-turbo \
 --temperature 0.7

## Compute Metrics

python scripts/metrics.py \
 --input data/responses/temp07/gpt35_t07_seed42.jsonl \
 --output data/analysis/run_temp07/

## Compare Runs

python scripts/compare_runs.py \
 --run-a data/analysis/run_temp02 \
 --run-b data/analysis/run_temp07 \
 --label-a temp0.2 \
 --label-b temp0.7 \
 --out-dir data/analysis/compare_temp07_vs_02
