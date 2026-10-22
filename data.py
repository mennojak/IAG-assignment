"""Data reading + one-off annotation.   REPORT: "Data reading" (parse tweets + group labels).
Called by agent.step1_preprocessing. Annotation uses llm.call_llm + SYSTEM_ANNOTATE."""
from llm import call_llm, SYSTEM_ANNOTATE

def load_tweets(topic):
    """-> [{"id": "t1", "group": "White|Hispanic|AfricanAmerican", "text": str}]  (90 per topic, 30 per group)
    Each tweet is one document, no chunking (as stated in the report)."""
    ...

def load_references(topic):
    """-> list[str], the DivSumm reference summaries (used for ROUGE / BERTScore)."""
    ...

def annotate_all():
    """Run ONCE, save to a JSON cache. DivSumm has no sentiment labels, so the LLM labels each tweet:
    {"tweet_id", "team", "polarity": Positive|Negative|Neutral, "intensity": Low|Medium|High,
     "kind": Prediction|Support|Criticism|FactualClaim}. One opinion per tweet.
    These become the Opinion individuals in the ontology (Tweet/Opinion extra concept).
    TODO: hand-check ~30 labels and report agreement; the ontology check is only as good as these labels."""
    ...

def load_opinions(topic):
    """Read the cache."""
    ...
