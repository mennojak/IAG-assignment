"""System-independent metrics: they work on the FINAL TEXT only, so the ontology agent and the plain-LLM
baseline are scored by exactly the same code.

fairness
  group_sim / sim_parity   mean cosine similarity (TF-IDF) between each summary sentence and the tweets of each
                           social group (mean of the 3 most similar tweets per group). sim_parity = min/max over
                           groups: 1.0 = the summary is equally close to all groups.
  equal_coverage_nn        each summary sentence is assigned to the group whose tweets it is closest to;
                           1 - (max share - min share). 1.0 = every group 'owns' the same number of sentences.
  polarity_skew            total-variation distance between the sentiment distribution of the source tweets and of
                           the summary sentences (0 = faithful; large = the summary over/under-states a side).
quality
  rouge1 / rouge2 / rougeL F1 against the human reference summaries (optional --refs), averaged over references.
cost
  seconds (measured by the caller).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

POLS = ["Positive", "Negative", "Neutral"]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 3]


def _group_sims(sents: list[str], tweets: list[dict], top: int = 3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    groups = sorted({t["group"] for t in tweets})
    vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True).fit([t["text"] for t in tweets] + sents)
    S = vec.transform(sents)
    out = np.zeros((len(sents), len(groups)))
    for j, g in enumerate(groups):
        T = vec.transform([t["text"] for t in tweets if t["group"] == g])
        sim = (S @ T.T).toarray()
        out[:, j] = np.sort(sim, axis=1)[:, -top:].mean(axis=1)
    return groups, out


def fairness_metrics(summary: str, tweets: list[dict]) -> dict:
    sents = sentences(summary)
    if not sents:
        return {"n_sentences": 0, "sim_parity": 0.0, "equal_coverage_nn": 0.0}
    groups, M = _group_sims(sents, tweets)
    mean = M.mean(axis=0)
    owner = Counter(M.argmax(axis=1))
    shares = np.array([owner.get(j, 0) for j in range(len(groups))]) / len(sents)
    res = {"n_sentences": len(sents), "n_words": len(summary.split()),
           "sim_parity": round(float(mean.min() / mean.max()) if mean.max() > 0 else 0.0, 3),
           "equal_coverage_nn": round(float(1 - (shares.max() - shares.min())), 3)}
    res.update({f"sim_{g}": round(float(m), 3) for g, m in zip(groups, mean)})
    return res


def polarity_dist(recs) -> np.ndarray:
    c = Counter(r.polarity for r in recs if r.team)
    tot = sum(c.values())
    return np.array([c[p] / tot if tot else 0.0 for p in POLS])


def polarity_skew(summary: str, source_dist: np.ndarray, backend, team: str | None) -> float:
    sents = sentences(summary)
    if not sents:
        return 1.0
    recs = backend.extract_opinions(sents, [team] * len(sents) if team else None)
    d = polarity_dist(recs)
    if d.sum() == 0:
        return 1.0
    return round(float(0.5 * np.abs(d - source_dist).sum()), 3)


def load_refs(refs_dir: str | None, topic: str) -> list[str]:
    """refs/<topic>.txt: one reference summary per paragraph (blank-line separated)."""
    if not refs_dir:
        return []
    f = next((p for p in Path(refs_dir).glob("*.txt") if p.stem.lower() == topic.lower()), None)
    return [x.strip() for x in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="replace")) if x.strip()] if f else []


def rouge_scores(summary: str, refs: list[str]) -> dict:
    if not refs:
        return {}
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = [sc.score(r, summary) for r in refs]
    return {k: round(float(np.mean([r[k].fmeasure for r in rows])), 3) for k in ("rouge1", "rouge2", "rougeL")}
