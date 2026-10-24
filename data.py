"""Data reading + one-off annotation.   REPORT: "Data reading" (parse tweets + group labels).
Called by agent.step1_preprocessing. Annotation uses llm.call_llm + SYSTEM_ANNOTATE."""

from llm import call_llm, SYSTEM_ANNOTATE
import json
import re
import config
from collections import Counter



VALID_POLARITY = {"Positive", "Negative", "Neutral"}
VALID_INTENSITY = {"Low", "Medium", "High"}
VALID_KIND = {"Prediction", "Support", "Criticism", "FactualClaim"}

GROUP_ALIASES = {
    "Hisp": "Hispanic",
    "AA": "AfricanAmerican",
    "White": "White",
    # Accept full labels too.
    "Hispanic": "Hispanic",
    "AfricanAmerican": "AfricanAmerican",
}

def _read_source_text(path):
    """Read UTF-8 if possible; the uploaded original 49ers file is CP1252."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _dataset(topic: str) -> dict:
    txt_path = config.DATA_DIR / f"{topic}.txt"

    if txt_path.is_file():
        tweets = []
        for lineno, line in enumerate(_read_source_text(txt_path).splitlines(), start=1):
            if not line.strip():
                continue  # Skip blank lines, but preserve original line numbers in errors.
            if "\t" not in line:
                raise ValueError(
                    f"{txt_path}, line {lineno}: expected GROUP<TAB>TWEET text"
                )
            raw_group, text = line.split("\t", 1)
            raw_group = raw_group.strip()
            group = GROUP_ALIASES.get(raw_group)
            if group is None:
                raise ValueError(
                    f"{txt_path}, line {lineno}: unknown group {raw_group!r}"
                )
            text = text.strip()
            if not text:
                raise ValueError(f"{txt_path}, line {lineno}: empty tweet")
            tweets.append({"id": f"t{len(tweets) + 1}", "group": group, "text": text})
        return {"tweets": tweets, "references": []}

    raise FileNotFoundError(
        f"Could not find {txt_path}). "
        "Put 49ers.txt, patriots.txt, and seahawks.txt in the project's data folder."
    )

def load_tweets(topic: str) -> list[dict]:
    """-> [{"id": "t1", "group": "White|Hispanic|AfricanAmerican", "text": str}]  (90 per topic, 30 per group)
    Each tweet is one document, no chunking (as stated in the report)."""
    raw = _dataset(topic).get("tweets", [])
    if not raw:
        raise ValueError(f"No tweets in dataset for {topic}")
    result = []
    seen = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError("Tweets must be objects with text, group and optional id")
        tid = str(item.get("id", f"t{i}"))
        group = item.get("group")
        text = str(item.get("text", "")).strip()
        if not re.fullmatch(r"t[1-9]\d*", tid) or tid in seen:
            raise ValueError(f"Tweet IDs must be unique and look like 't1': {tid}")
        if group not in config.GROUPS:
            raise ValueError(f"Tweet {tid}: group must be one of {config.GROUPS}; got {group!r}")
        if not text:
            raise ValueError(f"Tweet {tid} has empty text")
        result.append({"id": tid, "group": group, "text": text})
        seen.add(tid)
    return result

    """-> list[str], the DivSumm reference summaries (used for ROUGE / BERTScore)."""
def load_references(topic: str) -> list[str]:
    values = _dataset(topic).get("references", [])
    if not isinstance(values, list) or not all(isinstance(s, str) for s in values):
        raise ValueError("references must be a list of summary strings")
    return [s for s in values if s.strip()]

def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"No annotation JSON found: {text[:180]!r}")
        obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected an annotation JSON object")
    return obj

# def annotate_all():
#     """Run ONCE, save to a JSON cache. DivSumm has no sentiment labels, so the LLM labels each tweet:
#     {"tweet_id", "team", "polarity": Positive|Negative|Neutral, "intensity": Low|Medium|High,
#      "kind": Prediction|Support|Criticism|FactualClaim}. One opinion per tweet.
#     These become the Opinion individuals in the ontology (Tweet/Opinion extra concept).
#     TODO: hand-check ~30 labels and report agreement; the ontology check is only as good as these labels."""
#     ...

def annotate_all(topics: list[str] | None = None, *, force: bool = False) -> None:
    """One opinion per tweet; overwrite only if force=True. Review annotations manually."""
    for topic in topics or config.TOPICS:
        path = config.DATA_DIR / f"opinions_{topic}.json"
        tweets = load_tweets(topic)
        if path.exists() and not force:
            print(f"Annotations already exist: {path} (use --force-annotations to regenerate)")
            continue
        opinions = []
        for i, tweet in enumerate(tweets, 1):
            prompt = f"Team: {config.TEAM_NAMES[topic]}\nTweet [{tweet['id']}]: {tweet['text']}"
            obj = _parse_json_object(call_llm(SYSTEM_ANNOTATE, prompt, temperature=0))
            polarity = obj.get("polarity")
            intensity = obj.get("intensity")
            kind = obj.get("kind")
            if polarity not in VALID_POLARITY or intensity not in VALID_INTENSITY or kind not in VALID_KIND:
                raise ValueError(f"Invalid label for {tweet['id']}: {obj}. Check/correct this annotation")
            opinions.append({"tweet_id": tweet["id"], "team": config.TEAM_NAMES[topic],
                             "polarity": polarity, "intensity": intensity, "kind": kind})
            print(f"[{topic}] annotation {i}/{len(tweets)}", flush=True)
        path.write_text(json.dumps(opinions, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {path}")

def load_opinions(topic: str) -> list[dict]:
    path = config.DATA_DIR / f"opinions_{topic}.json"
    if not path.exists():
        raise FileNotFoundError(f"No opinion cache at {path}. Run: python main.py --annotate --topics {topic}")
    opinions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(opinions, list):
        raise ValueError("Opinion cache must be a JSON list")
    return opinions

if __name__ == "__main__":
    from pprint import pprint

    TOPIC = "49ers"
    RUN_ANNOTATION = False  # True calls Ollama and writes the opinion cache.

    def heading(label):
        print(f"\n{'=' * 15} {label} {'=' * 15}")

    heading("_dataset")
    try:
        dataset = _dataset(TOPIC)
        print("Dataset keys:", list(dataset.keys()))
        print("Dataset rows:", len(dataset["tweets"]))
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR:", exc)

    heading("load_tweets")
    try:
        tweets = load_tweets(TOPIC)
        print("Number of tweets:", len(tweets))
        print("Group counts:", dict(Counter(t["group"] for t in tweets)))
        print("First 3 tweets:")
        for tweet in tweets[:3]:
            pprint(tweet)
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR:", exc)

    heading("load_references")
    try:
        references = load_references(TOPIC)
        print("Number of references:", len(references))
        if not references:
            print("No reference summaries supplied; reference-based metrics will be skipped.")
        if references:
            print("First reference:", references[0])
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR:", exc)

    heading("_parse_json_object")
    example = '```json\n{"polarity": "Positive", "intensity": "High", "kind": "Support"}\n```'
    pprint(_parse_json_object(example))

    heading("annotate_all")
    if RUN_ANNOTATION:
        try:
            annotate_all([TOPIC])
        except Exception as exc:
            print("ERROR:", exc)
    else:
        print("Skipped (set RUN_ANNOTATION = True to call Ollama).")

    heading("load_opinions")
    try:
        opinions = load_opinions(TOPIC)
        print("Number of opinions:", len(opinions))
        if opinions:
            print("First opinion:")
            pprint(opinions[0])
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR:", exc)

    heading("DATA TESTS FINISHED")
