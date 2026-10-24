"""Full experiment of the project: 3 topics x 2 user prompts x 3 runs x {baseline LLM, ontology agent}.

    python run_experiments.py --model ollama:qwen2.5:7b-instruct --runs 3 --temperature 0.7 --refs refs
    python run_experiments.py --offline --runs 1 --per-cell 10        # pipeline smoke test (no LLM)

Outputs (in --out, default results/): results.csv (one row per summary), summaries.json (all texts + the agent's
citations), aggregate.csv (mean +- std per topic/prompt/system) and the aggregate table on stdout.
--ablation adds a third system: the agent without its ontology plan (free-form draft + the same checks).
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import pandas as pd

from agent import SummaryAgent
from data_loader import load_tweets, subsample
from metrics import fairness_metrics, load_refs, polarity_dist, polarity_skew, rouge_scores
from ontology_store import TEAM_TOPIC, OntologyStore
from prompts import PROMPTS

TOPICS = {"49ers": "data/49ers.txt", "Patriots": "data/Patriots.txt", "Seahawks": "data/Seahawks.txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", default="infoiag_ontology.owl")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "anthropic:claude-sonnet-5-5"))
    ap.add_argument("--temperature", type=float, default=0.7, help="> 0 so the 3 runs of a cell can differ")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-cell", type=int, default=0, help="subsample N tweets per (group, topic); 0 = all 30")
    ap.add_argument("--topics", nargs="+", default=list(TOPICS), choices=list(TOPICS))
    ap.add_argument("--refs", default=None, help="directory with <topic>.txt reference summaries (blank-line separated)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=0.3)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.offline:
        from llm import HeuristicBackend; backend = HeuristicBackend()
    else:
        from llm import LangChainBackend; backend = LangChainBackend(a.model, temperature=a.temperature)

    out = Path(a.out); out.mkdir(exist_ok=True)
    rows, texts = [], []
    for topic in a.topics:
        tweets = load_tweets([TOPICS[topic]])
        if a.per_cell:
            tweets = subsample(tweets, a.per_cell)
        team = tweets[0]["team"]
        src = polarity_dist(backend.extract_opinions([t["text"] for t in tweets], [t["team"] for t in tweets]))
        refs = load_refs(a.refs, topic)
        print(f"== {topic}: {len(tweets)} tweets, {len(refs)} reference summaries, source polarity "
              f"{dict(zip(['Pos', 'Neg', 'Neu'], map(float, src.round(2))))}")
        for pname, prompt in PROMPTS.items():
            for run in range(1, a.runs + 1):
                systems = ["baseline", "agent"] + (["ablation_no_plan"] if a.ablation else [])
                for system in systems:
                    t0 = time.perf_counter()
                    cites = None
                    if system == "baseline":
                        text = backend.plain_summary(tweets, prompt)
                    else:
                        ag = SummaryAgent(OntologyStore(a.owl), backend, [dict(t) for t in tweets], prompt,
                                          a.max_iters, use_ontology_context=(system == "agent"), tolerance=a.tolerance,
                                          verbose=False, k=a.k)
                        fin = ag.run()
                        text = fin["summary"]
                        cites = [{"claim": c["sentence"], "tweet_ids": c["tweet_ids"]} for c in fin["best"]["claims"]]
                    secs = time.perf_counter() - t0
                    m = fairness_metrics(text, tweets)
                    m["polarity_skew"] = polarity_skew(text, src, backend, team)
                    m.update(rouge_scores(text, refs))
                    row = {"topic": topic, "prompt": pname, "run": run, "system": system, "seconds": round(secs, 1), **m}
                    rows.append(row)
                    texts.append({"topic": topic, "prompt": pname, "run": run, "system": system, "summary": text,
                                  "citations": cites})
                    print(f"  {topic:9s} {pname:15s} run{run} {system:17s} parity={m['sim_parity']} "
                          f"ec={m['equal_coverage_nn']} skew={m['polarity_skew']} {secs:.0f}s")
                    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                    (out / "summaries.json").write_text(json.dumps(texts, indent=2, ensure_ascii=False))
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("topic", "prompt", "run", "system") and not c.startswith("sim_")]
    agg = df.groupby(["topic", "prompt", "system"])[num].agg(["mean", "std"]).round(3)
    agg.to_csv(out / "aggregate.csv")
    print("\n=== mean over runs, per system ===")
    print(df.groupby("system")[num].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
