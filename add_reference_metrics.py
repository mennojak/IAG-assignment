"""Adds ROUGE (and optionally BERTScore) against the DivSumm reference summaries to an EXISTING results folder,
using the saved summaries (summaries.json). Use it when the references were not available during the run:

    python add_reference_metrics.py --results results --refs refs --bertscore

refs/<topic>.txt (49ers.txt, Patriots.txt, Seahawks.txt): the reference summaries of that topic, separated by a blank line.
Writes results.csv (the old one is kept as results_before_refs.csv), aggregate.csv and median_per_system.csv.
Both metrics only look at the final text, so nothing has to be re-run."""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

import pandas as pd

from metrics import bertscore_f1, load_refs, rouge_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--bertscore", action="store_true")
    a = ap.parse_args()
    d = Path(a.results)
    df = pd.read_csv(d / "results.csv")
    texts = {(t["topic"], t["prompt"], int(t["run"]), t["system"]): t["summary"]
             for t in json.loads((d / "summaries.json").read_text())}
    shutil.copy(d / "results.csv", d / "results_before_refs.csv")
    refs = {topic: load_refs(a.refs, topic) for topic in df["topic"].unique()}
    for topic, rs in refs.items():
        print(f"{topic}: {len(rs)} reference summaries" + ("" if rs else "  <-- none found, skipped"))
    new = []
    for _, row in df.iterrows():
        text = texts.get((row["topic"], row["prompt"], int(row["run"]), row["system"]), "")
        rs = refs[row["topic"]]
        m = {}
        if text.strip() and rs:
            m.update(rouge_scores(text, rs))
            if a.bertscore:
                m.update(bertscore_f1(text, rs))
        new.append(m)
    add = pd.DataFrame(new, index=df.index)
    df = pd.concat([df.drop(columns=[c for c in add.columns if c in df.columns]), add], axis=1)
    df.to_csv(d / "results.csv", index=False)
    skip = ("model", "topic", "prompt", "run", "system")
    num = [c for c in df.columns if c not in skip and (c == "sim_parity" or not c.startswith("sim_"))]
    df.groupby(["topic", "prompt", "system"])[num].agg(["mean", "std"]).round(3).to_csv(d / "aggregate.csv")
    med = df.groupby("system")[num].median().round(3)
    med.to_csv(d / "median_per_system.csv")
    print("\n=== mean per system ===")
    print(df.groupby("system")[num].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
