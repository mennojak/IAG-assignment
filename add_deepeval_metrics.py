"""Adds DeepEval summarization scores (alignment, coverage, min of both) to an EXISTING results folder, using the saved
summaries (summaries.json) and a LOCAL Ollama judge. Nothing is re-generated.

    ollama pull qwen2.5:7b-instruct          (the judge; any local model works)
    py -m pip install deepeval
    py add_deepeval_metrics.py --results results --judge qwen2.5:7b-instruct
    py add_deepeval_metrics.py --results results --limit 2        # quick smoke test on 2 summaries

Only baseline + agent are scored by default (add --systems baseline agent ablation_no_plan for all three).
Resumable: every score is cached in results/deepeval_cache.json, so you can stop (Ctrl+C) and re-run.
Writes results.csv (old copy: results_before_deepeval.csv), aggregate.csv and median_per_system.csv."""
from __future__ import annotations

import argparse, json, shutil, time
from pathlib import Path

import pandas as pd

from data_loader import load_tweets, subsample
from run_experiments import TOPICS


SOURCE_HEADER = ("The following are {n} tweets posted by football fans about the {team}. Each numbered line is one tweet. "
                 "A summary of them describes what the tweets express, so statements such as 'many tweets express support' "
                 "are claims about these tweets.\n")


def build_source(tweets):
    team = tweets[0].get("team", "team")
    return SOURCE_HEADER.format(n=len(tweets), team=team) + "\n".join(f"{i}. {t['text']}" for i, t in enumerate(tweets, 1))


def make_metric(judge: str, n: int):
    from deepeval.metrics import SummarizationMetric
    try:
        from deepeval.models import OllamaModel
        model = OllamaModel(model=judge)
    except Exception:  # older deepeval: set once with `deepeval set-ollama <model>` and pass nothing
        model = None
    kw = dict(threshold=0.5, n=n, verbose_mode=False, truncation_limit=None)
    if model is not None:
        kw["model"] = model
    try:
        return SummarizationMetric(**kw)
    except TypeError:
        kw.pop("truncation_limit", None)
        return SummarizationMetric(**kw)


def score(metric, tweets, summary):
    from deepeval.test_case import LLMTestCase
    metric.measure(LLMTestCase(input=build_source(tweets), actual_output=summary))
    bd = getattr(metric, "score_breakdown", None) or {}
    al = getattr(metric, "alignment_score", None)
    co = getattr(metric, "coverage_score", None)
    if al is None:
        al = next((v for k, v in bd.items() if k.lower().startswith("align")), None)
    if co is None:
        co = next((v for k, v in bd.items() if k.lower().startswith("cover")), None)
    return {"deepeval_alignment": al, "deepeval_coverage": co, "deepeval_summarization": float(metric.score)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--judge", default="qwen2.5:7b-instruct", help="Ollama model name (without the 'ollama:' prefix)")
    ap.add_argument("--systems", nargs="+", default=["baseline", "agent"])
    ap.add_argument("--n", type=int, default=5, help="number of assessment questions DeepEval generates for coverage")
    ap.add_argument("--per-cell", type=int, default=0, help="must match the value used in run_experiments (0 = all tweets)")
    ap.add_argument("--limit", type=int, default=0, help="only score the first N summaries (smoke test)")
    a = ap.parse_args()
    d = Path(a.results)
    df = pd.read_csv(d / "results.csv")
    texts = {(t["topic"], t["prompt"], int(t["run"]), t["system"]): t["summary"]
             for t in json.loads((d / "summaries.json").read_text())}
    cache_path = d / "deepeval_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    tweets_by_topic = {}
    for topic in df["topic"].unique():
        tw = load_tweets([TOPICS[topic]])
        tweets_by_topic[topic] = subsample(tw, a.per_cell) if a.per_cell else tw
    metric = make_metric(a.judge, a.n)

    todo = [(i, r) for i, r in df.iterrows() if r["system"] in a.systems and not r.get("empty_summary", 0)]
    if a.limit:
        todo = todo[:a.limit]
    t0 = time.time()
    for n, (i, r) in enumerate(todo, 1):
        key = f'{r["topic"]}|{r["prompt"]}|{int(r["run"])}|{r["system"]}'
        if key in cache:
            continue
        text = texts.get((r["topic"], r["prompt"], int(r["run"]), r["system"]), "")
        if not text.strip():
            continue
        try:
            cache[key] = score(metric, tweets_by_topic[r["topic"]], text)
        except Exception as e:
            print(f"  [{n}/{len(todo)}] {key}: FAILED {type(e).__name__}: {str(e)[:100]}")
            continue
        cache_path.write_text(json.dumps(cache, indent=1))
        print(f"  [{n}/{len(todo)}] {key}: {cache[key]}  ({time.time() - t0:.0f}s elapsed)")

    shutil.copy(d / "results.csv", d / "results_before_deepeval.csv")
    cols = ["deepeval_alignment", "deepeval_coverage", "deepeval_summarization"]
    for c in cols:
        df[c] = [cache.get(f'{r["topic"]}|{r["prompt"]}|{int(r["run"])}|{r["system"]}', {}).get(c) for _, r in df.iterrows()]
    df.to_csv(d / "results.csv", index=False)
    skip = ("model", "topic", "prompt", "run", "system")
    num = [c for c in df.columns if c not in skip and (c == "sim_parity" or not c.startswith("sim_"))]
    df.groupby(["topic", "prompt", "system"])[num].agg(["mean", "std"]).round(3).to_csv(d / "aggregate.csv")
    df.groupby("system")[num].median().round(3).to_csv(d / "median_per_system.csv")
    print("\n=== DeepEval mean per system ===")
    print(df.groupby("system")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()