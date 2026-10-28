"""System-independent metrics: they work on the FINAL TEXT only, so the ontology agent and the plain-LLM
baseline are scored by exactly the same code.

fairness
  group_sim / sim_parity   mean cosine similarity (TF-IDF) between each summary sentence and the tweets of each
                           social group (mean of the 3 most similar tweets per group). sim_parity = min/max over
                           groups: 1.0 = the summary is equally close to all groups.
  equal_coverage           Equal Coverage as in the report: mean |p(d,s) - p(d,s|group)|, LOWER = fairer (see equal_coverage()).
  equal_coverage_nn        each summary sentence is assigned to the group whose tweets it is closest to;
                           1 - (max share - min share). 1.0 = every group 'owns' the same number of sentences.
  sel_intensity_gap        mean intensity of the source tweets each summary sentence is most similar to, minus the mean over all
                           tweets: > 0 = the summary is built from louder tweets than a fair sample (intensity bias in selection).
  high_share / high_gap    share of High-intensity sentences in the summary, and (summary - source): > 0 = louder than the tweets.
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
        # no usable sentence (empty / one-word output): leave the metrics missing (NaN) instead of a fake 0.0, so the
        # means ignore it; run_experiments records it in the `empty_summary` column
        return {"n_sentences": 0, "n_words": len(summary.split()), "sim_parity": None, "equal_coverage_nn": None,
                "equal_coverage": None}
    groups, M = _group_sims(sents, tweets)
    mean = M.mean(axis=0)
    owner = Counter(M.argmax(axis=1))
    shares = np.array([owner.get(j, 0) for j in range(len(groups))]) / len(sents)
    res = {"n_sentences": len(sents), "n_words": len(summary.split()),
           "sim_parity": round(float(mean.min() / mean.max()) if mean.max() > 0 else 0.0, 3),
           "equal_coverage_nn": round(float(1 - (shares.max() - shares.min())), 3),
           "equal_coverage": equal_coverage(summary, tweets)}
    res.update({f"sim_{g}": round(float(m), 3) for g, m in zip(groups, mean)})
    return res


def polarity_dist(recs) -> np.ndarray:
    c = Counter(r.polarity for r in recs if r.team)
    tot = sum(c.values())
    return np.array([c[p] / tot if tot else 0.0 for p in POLS])


_RANK = {"Low": 1, "Medium": 2, "High": 3}


def intensity_selection(summary: str, tweets: list[dict], src_recs, top: int = 3) -> dict:
    """Intensity bias in WHICH tweets a summary draws on (not in how emphatic its own wording is).
    Each summary sentence is matched to its `top` most similar labelled source tweets (TF-IDF cosine); we average the
    labelled intensity (Low=1, Medium=2, High=3) of those tweets and compare it with the mean over ALL labelled tweets.
      sel_intensity       mean intensity of the tweets the summary is built from
      src_intensity       mean intensity of all labelled tweets
      sel_intensity_gap   sel - src; > 0: the summary leans on louder tweets than a fair sample would
      sel_high_share      share of High tweets among the matched ones (compare with source_high_share)
    Works on the final text only, so every system is scored identically (labels come from the source, not the summary)."""
    sents = sentences(summary)
    idx = [i for i, r in enumerate(src_recs) if r.team]
    if not sents or not idx:
        return {}
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [tweets[i]["text"] for i in idx]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True).fit(texts + sents)
    sim = (vec.transform(sents) @ vec.transform(texts).T).toarray()
    rank = np.array([_RANK[src_recs[i].intensity] for i in idx], dtype=float)
    k = min(top, len(idx))
    pick = np.argsort(-sim, axis=1)[:, :k]                       # k most similar tweets per sentence
    sel = rank[pick]                                             # (sentences, k)
    return {"sel_intensity": round(float(sel.mean()), 3), "src_intensity": round(float(rank.mean()), 3),
            "sel_intensity_gap": round(float(sel.mean() - rank.mean()), 3),
            "sel_high_share": round(float((sel == 3).mean()), 3)}


def label_summary(summary: str, backend, team: str | None) -> list:
    """Opinion labels (polarity, intensity, ...) of every summary sentence, the same way for every system."""
    sents = sentences(summary)
    return backend.extract_opinions(sents, [team] * len(sents) if team else None) if sents else []


def skew_from_labels(recs, source_dist: np.ndarray) -> float:
    d = polarity_dist(recs)
    if d.sum() == 0:
        return 1.0
    return round(float(0.5 * np.abs(d - source_dist).sum()), 3)


def polarity_skew(summary: str, source_dist: np.ndarray, backend, team: str | None) -> float:
    return skew_from_labels(label_summary(summary, backend, team), source_dist)


def high_share(recs) -> float | None:
    """Share of High-intensity opinions among the labelled items (intensity-bias diagnostic)."""
    labelled = [r for r in recs if r.team]
    return round(sum(r.intensity == "High" for r in labelled) / len(labelled), 3) if labelled else None


def intensity_metrics(summary_recs, source_recs) -> dict:
    """high_share: how many summary sentences are emphatic; high_gap = summary - source (> 0: the summary is louder
    than the tweets, the effect expected from a 'focus on the strongest opinions' request)."""
    s, o = high_share(summary_recs), high_share(source_recs)
    return {"high_share": s, "source_high_share": o, "high_gap": None if s is None or o is None else round(s - o, 3)}


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


def equal_coverage(summary: str, tweets: list[dict]) -> float | None:
    """Equal Coverage (Olabisi et al. 2022 form): EC = 1/K * sum_k |p(d,s) - p(d,s | a=k)|, LOWER = fairer.
    Coverage of one tweet d by the summary s is ESTIMATED as its highest TF-IDF cosine similarity to any summary
    sentence; p(d,s) is the mean over all tweets and p(d,s | a=k) the mean over the tweets of social group k.
    (An estimate, not the authors' exact coverage model; the same estimator is used for every system.)"""
    sents = sentences(summary)
    if not sents:
        return None
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [t["text"] for t in tweets]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True).fit(texts + sents)
    cov = (vec.transform(texts) @ vec.transform(sents).T).toarray().max(axis=1)
    overall = float(cov.mean())
    groups = sorted({t["group"] for t in tweets})
    per = [float(cov[[i for i, t in enumerate(tweets) if t["group"] == g]].mean()) for g in groups]
    return round(float(np.mean([abs(overall - p) for p in per])), 4)


_BERT = {"scorer": None, "failed": False}


def bertscore_f1(summary: str, refs: list[str]) -> dict:
    """BERTScore F1 against the human references (mean over references). Needs `pip install bert-score`
    (downloads roberta-large on first use). The model is loaded ONCE per process and reused; if it cannot be
    loaded (missing package, no network) the problem is reported once and BERTScore is skipped from then on."""
    if not refs or _BERT["failed"]:
        return {}
    try:
        if _BERT["scorer"] is None:
            from bert_score import BERTScorer
            print("  loading BERTScore model (roberta-large, first time downloads ~1.4 GB) ...")
            _BERT["scorer"] = BERTScorer(lang="en")
        _, _, f1 = _BERT["scorer"].score([summary] * len(refs), refs)
        return {"bertscore_f1": round(float(f1.mean()), 3)}
    except Exception as e:                       # optional dependency / no network
        _BERT["failed"] = True
        print(f"  (BERTScore skipped for the rest of this run: {type(e).__name__}: {str(e)[:100]}\n"
              "   install it with:  py -m pip install bert-score)")
        return {}


def deepeval_summarization(summary: str, tweets: list[dict], judge: str | None) -> dict:
    """DeepEval SummarizationMetric (alignment, coverage, min of both) with a local judge, e.g. judge='ollama:qwen2.5:7b-instruct'.
    Needs `pip install deepeval`. NOTE: not exercised in the offline test; check it once on your machine."""
    if not judge:
        return {}
    try:
        from deepeval.metrics import SummarizationMetric
        from deepeval.test_case import LLMTestCase
        model = judge
        if judge.startswith("ollama:"):
            from deepeval.models import OllamaModel
            model = OllamaModel(model=judge.split(":", 1)[1])
        m = SummarizationMetric(threshold=0.5, model=model, verbose_mode=False)
        m.measure(LLMTestCase(input="\n".join(t["text"] for t in tweets), actual_output=summary))
        bd = getattr(m, "score_breakdown", None) or {}
        out = {"deepeval_summarization": round(float(m.score), 3)}
        for k in ("alignment", "coverage"):
            v = next((v for kk, v in bd.items() if kk.lower().startswith(k)), None)
            if v is not None:
                out[f"deepeval_{k}"] = round(float(v), 3)
        return out
    except Exception as e:
        print(f"  (DeepEval skipped: {type(e).__name__}: {str(e)[:80]})")
        return {}