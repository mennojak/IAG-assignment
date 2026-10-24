"""python main.py --per-cell 15 --prompt "Summarize the opinions about the NFL teams" """
import argparse, json, os
from pathlib import Path

from data_loader import load_tweets, subsample
from agent import SummaryAgent
from ontology_store import OntologyStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", default="infoiag_ontology.owl")
    ap.add_argument("--data", nargs="+", default=["data/49ers.txt", "data/Patriots.txt", "data/Seahawks.txt"],
                    help="DivSumm .txt files (group<TAB>text, team taken from the file name) or CSV with group,text[,team]")
    ap.add_argument("--per-cell", type=int, default=0, help="subsample at most N tweets per (group, team); 0 = all")
    ap.add_argument("--prompt", default="Summarize the opinions expressed in these tweets about the NFL teams.")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "anthropic:claude-sonnet-5-5"))
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--k", type=int, default=5, help="summary sentences per team (a number in the prompt overrides it)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-ontology-context", action="store_true", help="ablation: generator does not see the ontology view")
    ap.add_argument("--offline", action="store_true", help="use the keyword/heuristic backend (no API key needed)")
    ap.add_argument("--tolerance", type=float, default=0.5, help="equal-coverage tolerance; lower = stricter (0.2 -> each group >= 27%% of claims)")
    ap.add_argument("--require-reasoner", action="store_true", help="abort if Pellet/Java cannot run")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    tweets = load_tweets(a.data)
    if a.per_cell:
        tweets = subsample(tweets, a.per_cell)
    print(f"loaded {len(tweets)} tweets")
    if a.offline:
        from llm import HeuristicBackend; backend = HeuristicBackend()
    else:
        from llm import LangChainBackend; backend = LangChainBackend(a.model, temperature=a.temperature)

    agent = SummaryAgent(OntologyStore(a.owl), backend, tweets, a.prompt, a.max_iters,
                         use_ontology_context=not a.no_ontology_context, tolerance=a.tolerance, require_reasoner=a.require_reasoner, k=a.k)
    final = agent.run()
    best = final["best"]
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "summary.txt").write_text(final["summary"] + "\n")
    cites = [{"claim": c["sentence"], "team": c.get("team"), "tweet_ids": c["tweet_ids"]} for c in best["claims"]]
    Path(a.out, "citations.json").write_text(json.dumps(cites, indent=2))
    Path(a.out, "report.json").write_text(json.dumps(final["history"], indent=2, default=str))
    print("\n=== FINAL SUMMARY (iteration %d, score %d) ===" % (best["iteration"], best["score"]))
    print(final["summary"])
    print("metrics:", best["metrics"])


if __name__ == "__main__":
    main()
