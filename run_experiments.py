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
from metrics import (bertscore_f1, deepeval_summarization, fairness_metrics, load_refs, polarity_dist,
                     intensity_metrics, intensity_selection, label_summary, rouge_scores, skew_from_labels)
from ontology_store import TEAM_TOPIC, OntologyStore
from prompts import EXTRA_PROMPTS, PROMPTS

TOPICS = {"49ers": "data/49ers.txt", "Patriots": "data/Patriots.txt", "Seahawks": "data/Seahawks.txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", default="infoiag_ontology.owl")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "ollama:qwen2.5:7b-instruct"))
    ap.add_argument("--temperature", type=float, default=0.7, help="> 0 so the 3 runs of a cell can differ")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-cell", type=int, default=0, help="subsample N tweets per (group, topic); 0 = all 30")
    ap.add_argument("--topics", nargs="+", default=list(TOPICS), choices=list(TOPICS))
    ap.add_argument("--refs", default=None, help="directory with <topic>.txt reference summaries (blank-line separated)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=0.3)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS) + list(EXTRA_PROMPTS),
                    help="which user prompts to run (default: the two report prompts)")
    ap.add_argument("--bertscore", action="store_true", help="BERTScore F1 vs --refs (pip install bert-score)")
    ap.add_argument("--deepeval-judge", default=None, help="e.g. ollama:qwen2.5:7b-instruct -> DeepEval summarization metric")
    ap.add_argument("--no-length-match", action="store_true",
                    help="do NOT give the baseline the agent's sentence budget (--k); default: same length for both")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep the finished rows in --out and only compute the missing ones")
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
    if a.resume and (out / "results.csv").exists() and (out / "summaries.json").exists():
        rows = pd.read_csv(out / "results.csv").to_dict("records")
        for r_ in rows:                       # rows written by an older version have no empty_summary column
            if pd.isna(r_.get("empty_summary", float("nan"))):
                r_["empty_summary"] = int(r_.get("n_sentences", 1) == 0)
        texts = json.loads((out / "summaries.json").read_text())
        print(f"resuming: {len(rows)} finished rows found in {out}/")
    done = {(r_["topic"], r_["prompt"], int(r_["run"]), r_["system"]) for r_ in rows}
    for topic in a.topics:
        tweets = load_tweets([TOPICS[topic]])
        if a.per_cell:
            tweets = subsample(tweets, a.per_cell)
        team = tweets[0]["team"]
        src_recs = backend.extract_opinions([t["text"] for t in tweets], [t["team"] for t in tweets])
        src = polarity_dist(src_recs)
        refs = load_refs(a.refs, topic)
        print(f"== {topic}: {len(tweets)} tweets, {len(refs)} reference summaries, source polarity "
              f"{dict(zip(['Pos', 'Neg', 'Neu'], map(float, src.round(2))))}")
        for pname in a.prompts:
            prompt = {**PROMPTS, **EXTRA_PROMPTS}[pname]
            for run in range(1, a.runs + 1):
                systems = ["baseline", "agent"] + (["ablation_no_plan"] if a.ablation else [])
                for system in systems:
                    if (topic, pname, run, system) in done:
                        continue
                    t0 = time.perf_counter()
                    cites = None
                    if system == "baseline":
                        text = backend.plain_summary(tweets, prompt, None if a.no_length_match else a.k)
                    else:
                        ag = SummaryAgent(OntologyStore(a.owl), backend, [dict(t) for t in tweets], prompt,
                                          a.max_iters, use_ontology_context=(system == "agent"), tolerance=a.tolerance,
                                          verbose=False, k=a.k)
                        fin = ag.run()
                        text = fin["summary"]
                        cites = [{"claim": c["sentence"], "tweet_ids": c["tweet_ids"]} for c in fin["best"]["claims"]]
                    secs = time.perf_counter() - t0
                    m = fairness_metrics(text, tweets)
                    labels = label_summary(text, backend, team)        # one labelling pass, identical for every system
                    m["polarity_skew"] = skew_from_labels(labels, src) if m["n_sentences"] else None
                    m.update(intensity_metrics(labels, src_recs))
                    m.update(intensity_selection(text, tweets, src_recs))
                    m.update(rouge_scores(text, refs))
                    if a.bertscore:
                        m.update(bertscore_f1(text, refs))
                    m.update(deepeval_summarization(text, tweets, a.deepeval_judge))
                    row = {"model": a.model if not a.offline else "offline", "topic": topic, "prompt": pname, "run": run, "system": system,
                           "seconds": round(secs, 1), "empty_summary": int(m["n_sentences"] == 0), **m}
                    if m["n_sentences"] == 0:
                        print(f"  !! {topic} {pname} run{run} {system}: EMPTY summary ({text[:80]!r}) - metrics left missing")
                    rows.append(row)
                    texts.append({"topic": topic, "prompt": pname, "run": run, "system": system, "summary": text,
                                  "citations": cites})
                    print(f"  {topic:9s} {pname:15s} run{run} {system:17s} parity={m.get('sim_parity')} "
                          f"ec={m.get('equal_coverage')} ecnn={m.get('equal_coverage_nn')} skew={m.get('polarity_skew')} "
                          f"high={m.get('high_share')} (src {m.get('source_high_share')}) selgap={m.get('sel_intensity_gap')} "
                          f"{m['n_sentences']}sent {secs:.0f}s")
                    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                    (out / "summaries.json").write_text(json.dumps(texts, indent=2, ensure_ascii=False))
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("model", "topic", "prompt", "run", "system") and (c == "sim_parity" or not c.startswith("sim_"))]
    agg = df.groupby(["topic", "prompt", "system"])[num].agg(["mean", "std"]).round(3)
    agg.to_csv(out / "aggregate.csv")
    print("\n=== mean over runs, per system ===")
    print(df.groupby("system")[num].mean().round(3).T.to_string())
    print("\n=== median seconds per system (robust against one stalled run) ===")
    print(df.groupby("system")["seconds"].median().round(1).to_string())
    df.groupby("system")[num].median().round(3).to_csv(out / "median_per_system.csv")


if __name__ == "__main__":
    main()
