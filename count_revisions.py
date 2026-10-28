"""Counts how many revision loops the agent needs (for the runtime table) WITHOUT touching results/.
Re-runs ONLY the agent (18 runs, ~6-8 min with the local model) and writes revisions.csv + revisions_summary.csv.

    py count_revisions.py                      # same settings as run_experiments.py (temperature 0.7, k=5, max-iters 4)
    py count_revisions.py --runs 1 --topics 49ers      # quick test

revisions   = number of rewrite rounds (drafts evaluated - 1)
unresolved  = 1 if the returned draft still has violations (cap reached or no further improvement)
Note: the agent's wording is sampled at temperature 0.7, so these are fresh runs, not the exact runs behind results.csv."""
from __future__ import annotations

import argparse, os, time
from pathlib import Path

import pandas as pd

from agent import SummaryAgent
from data_loader import load_tweets
from ontology_store import OntologyStore
from prompts import PROMPTS
from run_experiments import TOPICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", default="infoiag_ontology.owl")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "ollama:qwen2.5:7b-instruct"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=0.3)
    ap.add_argument("--topics", nargs="+", default=list(TOPICS), choices=list(TOPICS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from llm import LangChainBackend
    backend = LangChainBackend(a.model, temperature=a.temperature)
    rows = []
    for topic in a.topics:
        tweets = load_tweets([TOPICS[topic]])
        for pname, prompt in PROMPTS.items():
            for run in range(1, a.runs + 1):
                t0 = time.perf_counter()
                ag = SummaryAgent(OntologyStore(a.owl), backend, [dict(t) for t in tweets], prompt, a.max_iters,
                                  use_ontology_context=True, tolerance=a.tolerance, verbose=False, k=a.k)
                fin = ag.run()
                hist = fin["history"]
                best = fin["best"]
                kinds = sorted({v["kind"] for v in best["violations"]})
                rows.append({"topic": topic, "prompt": pname, "run": run, "drafts": len(hist), "revisions": len(hist) - 1,
                             "best_iteration": best["iteration"], "unresolved": int(bool(best["violations"])),
                             "final_violations": ";".join(kinds), "seconds": round(time.perf_counter() - t0, 1)})
                print(f"  {topic}|{pname}|{run}: revisions={rows[-1]['revisions']} unresolved={rows[-1]['unresolved']} {kinds}")
    d = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(exist_ok=True)
    d.to_csv(out / "revisions.csv", index=False)
    g = d.groupby(["topic", "prompt"]).agg(rev_mean=("revisions", "mean"), rev_std=("revisions", "std"),
                                           unresolved=("unresolved", "sum"), runs=("run", "count")).round(2)
    g.to_csv(out / "revisions_summary.csv")
    print("\n", g.to_string(), f"\n\noverall mean revisions: {d.revisions.mean():.2f}; runs ending with violations: {d.unresolved.sum()}/{len(d)}")


if __name__ == "__main__":
    main()