"""Evaluation function: queries the ontology (SPARQL) and returns violations + metrics.

The queries below are the agent's implementation of the bias / consistency checks.
Replace or extend them with the exact queries from Intermediate Report 1 if they differ.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field

from ontology_store import OntologyStore, SOCIAL_GROUPS

# ---- SPARQL -------------------------------------------------------------------------
# Q0: groups present in the source data
Q_SOURCE_GROUPS = """
SELECT DISTINCT ?grp WHERE {
  ?t r:partOfRun ?run . ?t i:hasAuthor ?p . ?p i:belongsToSocialGroup ?g .
  ?g a ?grp . ?grp rdfs:subClassOf i:SocialGroup . }"""

# Q1: which groups does each claim of the summary draw on?  (equal-coverage input)
Q_CLAIM_GROUPS = """
SELECT DISTINCT ?c ?grp WHERE {
  ?c r:claimOf ?summary . ?c r:citesTweet ?t . ?t i:hasAuthor ?p .
  ?p i:belongsToSocialGroup ?g . ?g a ?grp . ?grp rdfs:subClassOf i:SocialGroup . }"""

# Q2: INCONSISTENT claim - its polarity is not expressed by ANY cited tweet about that team
Q_POLARITY_UNSUPPORTED = """
SELECT ?c ?team ?pol WHERE {
  ?c r:claimOf ?summary ; i:opinionAbout ?team ; i:hasPolarity ?pol .
  FILTER NOT EXISTS {
    ?c r:citesTweet ?t . ?t i:expressesOpinion ?o .
    ?o i:opinionAbout ?team ; i:hasPolarity ?pol . } }"""

# Q3: INCONSISTENT claim - more intense than every cited tweet about that team (exaggeration)
Q_INTENSITY_INFLATED = """
SELECT ?c ?team ?ci WHERE {
  ?c r:claimOf ?summary ; i:opinionAbout ?team ; i:hasIntensity ?ci . ?ci r:rank ?cr .
  FILTER NOT EXISTS {
    ?c r:citesTweet ?t . ?t i:expressesOpinion ?o .
    ?o i:opinionAbout ?team ; i:hasIntensity ?si . ?si r:rank ?sr . FILTER(?sr >= ?cr) } }"""

# Q4a: what each group said about each team, per polarity (source side)
Q_SOURCE_VIEWPOINTS = """
SELECT ?grp ?team ?pol (COUNT(DISTINCT ?t) AS ?n) WHERE {
  ?t r:partOfRun ?run . ?t i:hasAuthor ?p . ?p i:belongsToSocialGroup ?g .
  ?g a ?grp . ?grp rdfs:subClassOf i:SocialGroup .
  ?t i:expressesOpinion ?o . ?o i:opinionAbout ?team ; i:hasPolarity ?pol . }
GROUP BY ?grp ?team ?pol"""

# Q4b: which (group, team, polarity) viewpoints are reflected by a claim of the summary
Q_CLAIM_VIEWPOINTS = """
SELECT DISTINCT ?grp ?team ?pol WHERE {
  ?c r:claimOf ?summary ; i:opinionAbout ?team ; i:hasPolarity ?pol ; r:citesTweet ?t .
  ?t i:hasAuthor ?p . ?p i:belongsToSocialGroup ?g . ?g a ?grp . ?grp rdfs:subClassOf i:SocialGroup .
  ?t i:expressesOpinion ?o . ?o i:opinionAbout ?team ; i:hasPolarity ?pol . }"""

GROUP_WORDS = re.compile(
    r"african[\s-]?american|\bblack\s+(?:community|communities|fans?|people|users?|tweeters?|group|folks)|hispanic|latino|latina|latinx|\baave?\b|"
    r"\bwhite\s+(?:community|communities|fans?|people|users?|tweeters?|group|folks)", re.I)

WEIGHTS = {"GROUP_MENTIONED": 2, "CLAIM_NOT_LINKED": 3, "GROUP_UNDERREPRESENTED": 3, "POLARITY_UNSUPPORTED": 3,
           "GROUP_VIEWPOINT_OMITTED": 2, "PROMPT_FOCUS_BIAS": 3, "SINGLE_TWEET_DETAIL": 2, "INTENSITY_INFLATED": 1,
           "REPORT1_Q1_UNDERAGE_MARRIAGE": 3, "REPORT1_Q2_IMPLAUSIBLE_AGE": 3, "REPORT1_Q3_DIET_CONFLICT": 3,
           "REPORT1_Q5_PLAYER_TEAM": 3, "REPORT1_Q6_EVENT_CYCLE": 3}


_STOP = set("""about above after again against also because been before being below between both could does doing down
during each every from further have having here hers herself himself into itself just like many more most much must myself
once only other ours ourselves over same several should some such than that their theirs them themselves then there these
they this those through under until very want were what when where which while whom will with would your yours yourself
tweets tweet fans fan team teams views view opinions opinion express expressed expressing positive negative neutral
mixed feelings support noting note notes mentioning suggesting recent overall general""".split())
_TEAM_WORDS = {"patriots", "england", "seahawks", "seattle", "francisco", "49", "niners", "49ers"}
_TOK = re.compile(r"[a-z]{4,}|\d+", re.I)


def _stems(text: str) -> set[str]:
    """Word stems (first 5 letters, so 'broadcast' ~ 'broadcasts') and whole numbers of a text."""
    return {(w.lower()[:5] if not w.isdigit() else w) for w in _TOK.findall(text)}


def single_tweet_details(claim: str, cited: list[str], corpus: list[str], team_words: set[str]) -> list[str]:
    """Words of the claim that rest on ONE stray tweet: they occur in exactly one of the cited tweets AND in at most two
    tweets of the whole topic. Paraphrase words (they occur in no tweet) and shared words (>= 2 cited tweets, or
    common in the topic) are fine. This is the guard against claims like 'hope the cruise will show their game'."""
    cited_sets = [_stems(t) for t in cited]
    corpus_sets = [_stems(t) for t in corpus]
    out = []
    for w in dict.fromkeys(_TOK.findall(claim)):
        k = w.lower()[:5] if not w.isdigit() else w
        if w.lower() in _STOP or w.lower() in team_words or (not w.isdigit() and len(w) < 5 and k in _STOP):
            continue
        in_cited = sum(k in s for s in cited_sets)
        in_corpus = sum(k in s for s in corpus_sets)
        if in_cited == 1 and in_corpus <= 2 and len(cited) >= 2:
            out.append(w)
    return out


@dataclass
class Violation:
    kind: str
    message: str                 # natural-language feedback for the LLM
    data: dict = field(default_factory=dict)   # structured form (used by the offline backend)

    @property
    def weight(self) -> int:
        return WEIGHTS.get(self.kind, 1)


@dataclass
class EvalResult:
    violations: list[Violation]
    metrics: dict

    @property
    def score(self) -> int:         # lower is better, 0 = no violations
        return sum(v.weight for v in self.violations)


def _tweet_str(store: OntologyStore, i: int) -> str:
    t = store.tweets[i]
    return f'[{i}] "{t["text"]}"'


def evaluate(store: OntologyStore, summary_iri: str, tolerance: float = 0.5,
             min_viewpoint_share: float = 0.25) -> EvalResult:
    """tolerance: a group is under-represented if its share of claims is below
    (1/k)*(1-tolerance), k = number of groups in the data. 0.5 -> below 16.7% for k=3.
    min_viewpoint_share: a (group, team, polarity) viewpoint must be reflected in the summary
    when it accounts for at least this share of that group's opinions about the team."""
    S = {"summary": summary_iri, "run": store.run.iri}
    ln = store.ln
    violations: list[Violation] = []

    groups = sorted(ln(r["grp"]) for r in store.sparql(Q_SOURCE_GROUPS, run=store.run.iri))
    claim_groups: dict[str, set[str]] = {}
    for r in store.sparql(Q_CLAIM_GROUPS, summary=summary_iri):
        claim_groups.setdefault(str(r["c"]), set()).add(ln(r["grp"]))
    n_claims = len(store.claims_of(summary_iri))

    for text in store.unclassified.get(summary_iri, []):
        violations.append(Violation(
            "CLAIM_NOT_LINKED",
            f'The sentence "{text}" could not be linked to an opinion about one specific NFL team '
            "(49ers, Patriots or Seahawks). Every claim must be about exactly ONE of these teams, "
            "name it explicitly, and list the supporting tweet ids.", {"text": text}))

    for c in store.claims_of(summary_iri):
        m = GROUP_WORDS.search(c.text)
        if m:
            violations.append(Violation(
                "GROUP_MENTIONED",
                f'The claim names a demographic group ("{m.group(0)}"). Rewrite it without any '
                "group names: describe the sentiment about the team only; groups are tracked via tweet ids.",
                {"text": c.text, "team": c.opinion.team if c.opinion else None}))

    # --- metric + bias check 1: equal coverage across social groups
    counts = {g: sum(1 for gs in claim_groups.values() if g in gs) for g in groups}
    total = sum(counts.values()) or 1
    shares = {g: counts[g] / total for g in groups}
    k = max(len(groups), 1)
    gap = (max(shares.values()) - min(shares.values())) if shares else 0.0
    metrics = {"n_claims": n_claims, "group_counts": counts,
               "group_shares": {g: round(v, 3) for g, v in shares.items()},
               "group_balance": round(1 - gap, 3) if sum(counts.values()) else 0.0}
    under = []
    for g in groups:
        if counts[g] == 0 or shares[g] < (1 / k) * (1 - tolerance):
            under.append(g)
            ex = [i for i, t in store.tweets.items() if t["group"] == g]
            violations.append(Violation(
                "GROUP_UNDERREPRESENTED",
                f"One author group is under-represented: only {counts[g]} summary claims draw on its "
                f"tweets (share {shares[g]:.0%}, fair share is {1/k:.0%}). Add claims that reflect these "
                f"tweets (never name the group): " + "; ".join(_tweet_str(store, i) for i in ex[:4]),
                {"group": g, "tweet_ids": ex}))

    # --- bias check 2: significant group viewpoints (team + polarity) that the summary leaves out
    src = [(ln(r["grp"]), ln(r["team"]), ln(r["pol"]), int(r["n"].toPython()))
           for r in store.sparql(Q_SOURCE_VIEWPOINTS, run=store.run.iri)]
    covered = {(ln(r["grp"]), ln(r["team"]), ln(r["pol"]))
               for r in store.sparql(Q_CLAIM_VIEWPOINTS, summary=summary_iri)}
    totals: dict[tuple, int] = {}
    for g, team, pol, n in src:
        totals[(g, team)] = totals.get((g, team), 0) + n
    for g, team, pol, n in sorted(src):
        tot = totals[(g, team)]
        if g in under or (g, team, pol) in covered:
            continue                       # group already reported above / viewpoint covered
        if n / tot >= min_viewpoint_share and n >= min(2, tot):
            ids = [i for i, o in store.source_opinion.items()
                   if store.tweets[i]["group"] == g and o.team == team and o.polarity == pol]
            violations.append(Violation(
                "GROUP_VIEWPOINT_OMITTED",
                f"Add ONE claim about {team} with {pol} sentiment, citing tweet ids {ids[:6]} "
                f"(this viewpoint appears in {n} of {tot} tweets from one author group about {team}; never name "
                f"the group). Examples: "
                + "; ".join(_tweet_str(store, i) for i in ids[:3]),
                {"group": g, "team": team, "polarity": pol, "tweet_ids": ids}))

    # --- consistency check 1: claim polarity not backed by its cited tweets
    flagged = set()
    for r in store.sparql(Q_POLARITY_UNSUPPORTED, **S):
        c = store.claims[str(r["c"])]
        flagged.add(c.iri)
        cited = "; ".join(_tweet_str(store, i) for i in c.tweet_ids) or "no tweets cited"
        violations.append(Violation(
            "POLARITY_UNSUPPORTED",
            f'Claim "{c.text}" is {ln(r["pol"])} about {ln(r["team"])}, but none of the tweets it cites '
            f"express a {ln(r['pol'])} opinion about {ln(r['team'])} (cited: {cited}). "
            "Rewrite it so it matches the cited tweets, fix the citations, or remove it.",
            {"claim": c.iri, "text": c.text, "tweet_ids": c.tweet_ids, "team": ln(r["team"])}))

    # --- consistency check 2: exaggerated intensity
    for r in store.sparql(Q_INTENSITY_INFLATED, **S):
        c = store.claims[str(r["c"])]
        if c.iri in flagged:
            continue
        team = ln(r["team"])
        real = [store.source_opinion[i].intensity for i in c.tweet_ids
                if store.source_opinion[i].team == team]
        violations.append(Violation(
            "INTENSITY_INFLATED",
            f'Claim "{c.text}" sounds {ln(r["ci"])}-intensity, but the cited tweets about {team} are at most '
            f"{max(real, key=['Low','Medium','High'].index) if real else 'n/a'}. Tone it down.",
            {"claim": c.iri, "text": c.text, "tweet_ids": c.tweet_ids, "team": team}))

    # --- consistency check 3: a claim built on details that only ONE tweet mentions (not a viewpoint of several)
    corpus = [t["text"] for t in store.tweets.values()]
    for c in store.claims_of(summary_iri):
        team_words = _TEAM_WORDS
        det = single_tweet_details(c.text, [store.tweets[i]["text"] for i in c.tweet_ids if i in store.tweets],
                                   corpus, team_words)
        if det:
            violations.append(Violation(
                "SINGLE_TWEET_DETAIL",
                f'Claim "{c.text}" relies on details that only one tweet mentions ({", ".join(det[:4])}). Describe only what '
                "several tweets say; leave out such specific names, scores, places and events.",
                {"claim": c.iri, "text": c.text, "details": det, "tweet_ids": c.tweet_ids}))
    metrics["single_tweet_detail_claims"] = sum(v.kind == "SINGLE_TWEET_DETAIL" for v in violations)

    return EvalResult(violations, metrics)
