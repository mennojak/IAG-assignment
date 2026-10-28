"""Diagnose DeepEval: shows which summary claims the judge marked unsupported and why.
    py debug_deepeval.py --judge qwen-judge
"""
import argparse, json
from pathlib import Path
from deepeval.metrics import SummarizationMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from data_loader import load_tweets
from run_experiments import TOPICS
from add_deepeval_metrics import build_source

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="qwen-judge")
ap.add_argument("--system", default="agent")
ap.add_argument("--results", default="results")
a = ap.parse_args()

tweets = load_tweets([TOPICS["49ers"]])
s = next(t for t in json.loads((Path(a.results) / "summaries.json").read_text())
         if t["topic"] == "49ers" and t["prompt"] == "strongest_opinions" and t["run"] == 1 and t["system"] == a.system)
text = build_source(tweets)
print("INPUT chars:", len(text), "| approx tokens:", len(text) // 4)
print("SUMMARY:", s["summary"], "\n")
m = SummarizationMetric(threshold=0.5, n=5, verbose_mode=True, model=OllamaModel(model=a.judge))
m.measure(LLMTestCase(input=text, actual_output=s["summary"]))
print("\nscore:", m.score, "| reason:", m.reason)
for attr in ("claims", "truths", "assessment_questions", "coverage_verdicts", "alignment_verdicts", "score_breakdown"):
    print(f"\n--- {attr} ---\n", getattr(m, attr, "n/a"))