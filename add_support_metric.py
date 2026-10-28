"""Sentence-level SUPPORT rate: for every sentence of a summary, a local Ollama judge answers one simple question
("do the tweets support this sentence?"). Score per summary = mean over sentences (yes=1, partly=0.5, no=0).
Used instead of DeepEval's alignment, which a 7B judge answers inconsistently (verdict contradicts its own reason).
Same numbered-tweet source and the same judge for every system.

    py add_support_metric.py --results results --judge qwen-judge --limit 2
    py add_support_metric.py --results results --judge qwen-judge
Cached in results/support_cache.json (resumable). Adds column `support` to results.csv (old copy: results_before_support.csv)."""
from __future__ import annotations

import argparse, json, re, shutil, time
from pathlib import Path

import pandas as pd

from add_deepeval_metrics import build_source
from data_loader import load_tweets
from run_experiments import TOPICS

SYSTEM = ("You check whether a sentence from a summary is supported by a list of tweets. The sentence describes what the tweets "
          "express (e.g. 'many tweets express support' is supported if several tweets clearly do). Answer with JSON only: "
          '{"supported": "yes" | "partly" | "no", "why": "<one short sentence>"}. '
          '"yes" = clearly supported by the tweets, "partly" = only some of it is supported, "no" = not supported or contradicted.')


def sentences(summary: str) -> list[str]:
    s = re.sub(r"\(?\s*(?:tweets?:?\s*)?\[[\d,\s]+\]\)?", "", summary)          # drop "(tweets [1, 2])" citations
    s = re.sub(r"\[[\d,\s]+\]", "", s)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", s.strip())
    return [p.strip() for p in parts if len(p.split()) >= 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--judge", default="qwen-judge")
    ap.add_argument("--systems", nargs="+", default=["baseline", "agent"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-sentences", type=int, default=8, help="score at most N sentences per summary (ablation can be long)")
    a = ap.parse_args()
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=a.judge, temperature=0, format="json", num_ctx=16384)
    d = Path(a.results)
    df = pd.read_csv(d / "results.csv")
    texts = {(t["topic"], t["prompt"], int(t["run"]), t["system"]): t["summary"]
             for t in json.loads((d / "summaries.json").read_text())}
    src = {t: build_source(load_tweets([TOPICS[t]])) for t in df["topic"].unique()}
    cp = d / "support_cache.json"
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    todo = [r for _, r in df.iterrows() if r["system"] in a.systems and not r.get("empty_summary", 0)]
    if a.limit:
        todo = todo[:a.limit]
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        key = f'{r["topic"]}|{r["prompt"]}|{int(r["run"])}|{r["system"]}'
        if key in cache:
            continue
        sents = sentences(texts.get((r["topic"], r["prompt"], int(r["run"]), r["system"]), ""))[:a.max_sentences]
        vals, why = [], []
        for s in sents:
            try:
                out = json.loads(llm.invoke([("system", SYSTEM), ("user", f"TWEETS:\n{src[r['topic']]}\n\nSENTENCE: {s}")]).content)
                v = {"yes": 1.0, "partly": 0.5, "no": 0.0}.get(str(out.get("supported", "")).lower())
            except Exception as e:
                v, out = None, {"why": f"error {type(e).__name__}"}
            if v is not None:
                vals.append(v)
            why.append({"sentence": s, "supported": out.get("supported"), "why": out.get("why")})
        cache[key] = {"support": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals), "detail": why}
        cp.write_text(json.dumps(cache, indent=1))
        print(f"  [{n}/{len(todo)}] {key}: support={cache[key]['support']} over {len(vals)} sentences ({time.time() - t0:.0f}s)")
    shutil.copy(d / "results.csv", d / "results_before_support.csv")
    df["support"] = [cache.get(f'{r["topic"]}|{r["prompt"]}|{int(r["run"])}|{r["system"]}', {}).get("support") for _, r in df.iterrows()]
    df.to_csv(d / "results.csv", index=False)
    print("\n=== support (mean per system) ===")
    print(df.groupby("system")["support"].mean().round(3).to_string())


if __name__ == "__main__":
    main()