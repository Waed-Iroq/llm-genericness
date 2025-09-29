#!/usr/bin/env python3
import os, json, time, uuid, random, argparse, glob, sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect multiple generations per prompt from JSONL files under data/prompts/"
    )
    # NOTE: recursive ** so it finds files in subfolders like randomPrompts/, rephrasedPrompts/, etc.
    p.add_argument(
        "--prompts-glob",
        default="data/prompts/**/*.jsonl",
        help='Glob to prompt files (supports **). Example: "data/prompts/{randomPrompts,rephrasedPrompts}/*.jsonl"',
    )
    p.add_argument(
        "--out",
        default="data/responses/run1.jsonl",
        help="Output JSONL file (default: data/responses/run1.jsonl)",
    )
    p.add_argument(
        "--n-gen",
        type=int,
        default=2,
        help="Generations per prompt (default: 2)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--provider",
        choices=["none", "openai"],
        default="none",
        help="LLM provider (default: none -> stub that raises)",
    )
    p.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        help="Model name to use (default: env LLM_MODEL or 'gpt-3.5-turbo')",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (rate limiting safety)",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on total prompts processed (for dry runs)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompt order before generation",
    )
    return p.parse_args()

# ---------- Prompt loader ----------

def load_prompts(glob_pat: str) -> Iterator[Dict[str, Any]]:
    # recursive=True enables ** patterns and subfolders
    files = sorted(glob.glob(glob_pat, recursive=True))
    if not files:
        print(f"[collect] No prompt files matched: {glob_pat}", file=sys.stderr)
        return
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bad JSONL in {fp} (line {lineno}): {e}") from e
                # sanity: required fields
                for req in ("task", "prompt_id", "prompt"):
                    if req not in obj:
                        raise RuntimeError(f"Missing '{req}' in {fp} line {lineno}: {line}")
                # optional group
                obj.setdefault("group", "default")
                # keep source for tracing/debug
                obj["_source_file"] = fp
                yield obj

# ---------- Provider(s) ----------

def call_model_stub(prompt: str, temperature: float, seed: Optional[int], model: str) -> str:
    raise NotImplementedError(
        "No provider configured. Run with --provider openai (set OPENAI_API_KEY), "
        "or implement call_model_stub() to use your own provider."
    )

def call_model_openai(prompt: str, temperature: float, seed: Optional[int], model: str) -> str:
    # Chat Completions API (SDK-compatible access)
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)

    # NOTE: Most OpenAI chat endpoints ignore user-provided seeds; we pass it through for reproducibility hooks.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Try common access paths
        try:
            return resp.choices[0].message.content
        except Exception:
            pass
        try:
            return resp.choices[0]["message"]["content"]
        except Exception:
            pass
        try:
            return str(resp.choices[0].message)
        except Exception:
            pass
        return str(resp)
    except Exception as e:
        # We still emit a record so downstream analysis can see the error
        return f"[ERROR] {e!r}"

def call_model(prompt: str, temperature: float, seed: Optional[int], model: str, provider: str) -> str:
    if provider == "openai":
        return call_model_openai(prompt, temperature, seed, model)
    return call_model_stub(prompt, temperature, seed, model)

# ---------- Main ----------

def main():
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_iter = load_prompts(args.prompts_glob)
    prompts = list(prompts_iter)

    if not prompts:
        print(f"[collect] No prompts loaded. Check your --prompts-glob: {args.prompts_glob}", file=sys.stderr)
        sys.exit(1)

    if args.shuffle:
        random.shuffle(prompts)

    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    total_expected = len(prompts) * args.n_gen
    print(f"[collect] Files: {args.prompts_glob}")
    print(f"[collect] Prompts loaded: {len(prompts)}  |  Generations per prompt: {args.n_gen}")
    print(f"[collect] Output -> {out_path}")
    print(f"[collect] Provider: {args.provider}  |  Model: {args.model}  |  Temp: {args.temperature}")

    with open(out_path, "w", encoding="utf-8") as out:
        pbar = tqdm(total=total_expected, desc="Collecting")
        for item in prompts:
            for k in range(args.n_gen):
                seed = random.randint(0, 2_147_483_647)
                text = call_model(
                    prompt=item["prompt"],
                    temperature=args.temperature,
                    seed=seed,
                    model=args.model,
                    provider=args.provider,
                )
                rec = {
                    # original prompt fields (minus internals)
                    "task": item["task"],
                    "group": item.get("group", "default"),
                    "prompt_id": item["prompt_id"],
                    "prompt": item["prompt"],
                    # generation metadata
                    "response_id": str(uuid.uuid4())[:8],
                    "k": k,
                    "seed": seed,
                    "temperature": args.temperature,
                    "model": args.model,
                    "response": text,
                    "ts": int(time.time()),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.update(1)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        pbar.close()

    print(f"[collect] Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
