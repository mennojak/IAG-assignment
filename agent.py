"""LangGraph agent:  ingest -> draft -> evaluate -> (revise -> evaluate)* -> finish"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from evaluation import EvalResult, evaluate
from llm import TEAM_LABEL, Backend, clean_sentence, template_sentence
from report1_checks import Report1Checker
from ontology_store import INTENSITIES, POLARITIES, TEAM_TOPIC, OntologyStore, OpinionRecord
from prompts import PromptSpec, parse_prompt

MAX_LLM_TRIES = 2      # per claim (draft + one rewrite); afterwards the deterministic template sentence is used
MAX_PER_GROUP = 2      # tweets cited per author group in one claim
MIN_SUPPORT = 3        # an aspect gets its own claim only with >= 3 tweets AND >= 8% of the team's tweets ...
MIN_SUPPORT_FRAC = 0.08   # ... otherwise it is folded into the general claim (no claims about one stray tweet)
DEFAULT_K = 5          # sentences per team/topic (the DivSumm references have 5 sentences)

Q_VIEW = """
SELECT ?grp ?team ?pol ?inten ?kind ?id WHERE {
  ?t r:partOfRun ?run ; r:tweetId ?id ; i:hasAuthor ?p ; i:expressesOpinion ?o .
  ?p i:belongsToSocialGroup ?g . ?g a ?grp . ?grp rdfs:subClassOf i:SocialGroup .
  ?o i:opinionAbout ?team ; i:hasPolarity ?pol ; i:hasIntensity ?inten ; i:hasKind ?kind . }
ORDER BY ?grp ?team ?pol ?id"""


def ontology_view(store: OntologyStore) -> str:
    """Ontology-derived context for the generator: which group said what about which team."""
    rows = store.sparql(Q_VIEW, run=store.run.iri)
    agg: dict[tuple, list] = {}
    for r in rows:
        key = tuple(store.ln(r[k]) for k in ("grp", "team", "pol"))
        agg.setdefault(key, []).append((int(r["id"].toPython()), store.ln(r["inten"]), store.ln(r["kind"])))
    lines = []
    for (g, team, pol), items in agg.items():
        ids = ", ".join(str(i) for i, _, _ in items[:6]) + (", ..." if len(items) > 6 else "")
        top = max((x[1] for x in items), key=["Low", "Medium", "High"].index)
        lines.append(f"- group={g} | team={team} | sentiment={pol} | n={len(items)} | max intensity={top} | e.g. tweets {ids}")
    groups = sorted({k[0] for k in agg})
    return (f"{len(groups)} social groups ({', '.join(groups)}); each deserves ~{100//max(len(groups),1)}% of the claims.\n"
            + "\n".join(lines))


def _typical_intensity(levels: list[str]) -> str:
    """Most common intensity among the cited tweets (ties -> the milder one), so one emphatic
    tweet does not license emphatic wording for the whole claim."""
    c = Counter(levels)
    return max(c, key=lambda k: (c[k], -INTENSITIES.index(k)))


def _quantifier(n: int, total: int) -> str:
    share = n / max(total, 1)
    return "Most" if share >= 0.5 else "Many" if share >= 0.3 else "Several" if share >= 0.15 else "Some"


def plan_from_ontology(store: OntologyStore, spec: PromptSpec | None = None) -> list[dict]:
    """Candidate claims, one per (team, sentiment, aspect), built from the SPARQL view of the ontology.
    Every author group that holds that viewpoint contributes tweets, so group balance and viewpoint
    coverage are guaranteed by construction instead of left to the LLM. Aspects (offense, coaching, ...)
    make the claims specific; aspects supported by too few tweets are folded into a general claim."""
    spec = spec or PromptSpec()
    rows = store.sparql(Q_VIEW, run=store.run.iri)
    cell: dict[tuple, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    team_total: dict[str, int] = defaultdict(int)
    for r in rows:
        team, pol, grp = store.ln(r["team"]), store.ln(r["pol"]), store.ln(r["grp"])
        tid = int(r["id"].toPython())
        asp = store.source_opinion[tid].aspect or "other"
        cell[(team, pol)][asp][grp].append((tid, store.ln(r["inten"])))
        team_total[team] += 1
    cands = []
    for team in TEAM_TOPIC:
        pols = [p for p in POLARITIES if (team, p) in cell]
        min_sup = max(MIN_SUPPORT, math.ceil(MIN_SUPPORT_FRAC * team_total[team]))
        for pol in pols:
            merged: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for asp, groups in cell[(team, pol)].items():
                n = sum(len(v) for v in groups.values())
                key = asp if (asp != "other" and n >= min_sup) else "general"
                for g, items in groups.items():
                    merged[key][g] += items
            for asp, groups in merged.items():
                n = sum(len(v) for v in groups.values())
                if n < 2 and (len(pols) > 1 or len(merged) > 1):
                    continue                               # a single stray tweet is not a viewpoint
                picked = []
                for g in sorted(groups):
                    picked += sorted(groups[g])[:MAX_PER_GROUP]
                cands.append({"team": team, "polarity": pol, "aspect": "" if asp == "general" else asp,
                              "ids": [i for i, _ in picked], "support": n,
                              "intensity": _typical_intensity([x for _, x in picked]),
                              "quantifier": _quantifier(n, team_total[team]), "sentence": "",
                              "attempts": 0, "issues": []})
    return select_k(cands, spec.k or DEFAULT_K, spec.focus)


def _order_key(p: dict, focus: str | None):
    teams = list(TEAM_TOPIC)
    return (teams.index(p["team"]) if p["team"] in teams else 99,
            0 if p["polarity"] == focus else 1, POLARITIES.index(p["polarity"]), -p.get("support", 0))


def select_k(cands: list[dict], k: int, focus: str | None) -> list[dict]:
    """Pick at most k claims per team. First one claim for every non-neutral viewpoint (the largest one), so
    a one-sided request can never silence the other side; then fill the remaining slots by support,
    preferring the requested focus. Neutral claims only fill leftover slots."""
    chosen: list[dict] = []
    for team in TEAM_TOPIC:
        mine = [c for c in cands if c["team"] == team]
        if not mine:
            continue
        pick: list[dict] = []
        for pol in POLARITIES[:2]:                         # Positive, Negative
            best = max((c for c in mine if c["polarity"] == pol), key=lambda c: c["support"], default=None)
            if best:
                pick.append(best)
        rest = sorted((c for c in mine if c not in pick),
                      key=lambda c: (c["polarity"] == "Neutral", c["polarity"] != focus, -c["support"]))
        pick += rest[:max(k - len(pick), 0)]
        chosen += pick[:max(k, 1)]
    return sorted(chosen, key=lambda p: _order_key(p, focus))


def _enforce_k(plan: list[dict], k: int, focus: str | None) -> list[dict]:
    """After revision added tweets/claims: drop the weakest claims above k, never the last claim of a viewpoint."""
    for team in TEAM_TOPIC:
        while True:
            mine = [p for p in plan if p["team"] == team]
            if len(mine) <= k:
                break
            cnt = Counter(p["polarity"] for p in mine)
            droppable = [p for p in mine if cnt[p["polarity"]] > 1 or p["polarity"] == "Neutral"]
            if not droppable:
                break
            plan.remove(min(droppable, key=lambda p: p["support"]))
    return sorted(plan, key=lambda p: _order_key(p, focus))


def plan_claims(plan: list[dict]) -> list[dict]:
    return [{"sentence": p["sentence"], "tweet_ids": list(p["ids"]), "team": p["team"],
             "template": p.get("template", False), "polarity": p["polarity"], "intensity": p["intensity"],
             "aspect": p.get("aspect", "")}
            for p in plan]


def render_summary(claims: list[dict]) -> str:
    """Readable final text: one paragraph per team, no tweet ids, no group names."""
    paras = []
    for team in TEAM_TOPIC:
        sents = [c["sentence"].strip() for c in claims if c.get("team") == team]
        if sents:
            paras.append(" ".join(sents))
    other = [c["sentence"].strip() for c in claims if c.get("team") not in TEAM_TOPIC]
    return "\n\n".join(paras + ([" ".join(other)] if other else []))


class State(TypedDict, total=False):
    claims: list[dict]
    iteration: int
    result: EvalResult
    history: list[dict]
    best: dict
    summary: str
    plan: list[dict]
    stale: int


class SummaryAgent:
    def __init__(self, store: OntologyStore, backend: Backend, tweets: list[dict], request: str,
                 max_iters: int = 4, use_ontology_context: bool = True, tolerance: float = 0.5,
                 verbose: bool = True, require_reasoner: bool = False, k: int = DEFAULT_K):
        self.store, self.backend, self.tweets, self.request = store, backend, tweets, request
        self.max_iters, self.use_ctx, self.tol, self.verbose = max_iters, use_ontology_context, tolerance, verbose
        self.require_reasoner = require_reasoner
        self.spec = parse_prompt(request, k)           # sentence budget + one-sided focus read from the user prompt
        self.r1 = Report1Checker(store)
        self.view: str | None = None
        self.graph = self._build()

    def log(self, *a):
        if self.verbose: print(*a)

    # ---------------------------------------------------------------- nodes
    def ingest(self, st: State) -> State:
        opinions = self.backend.extract_opinions([t["text"] for t in self.tweets],
                                                 [t.get("team") for t in self.tweets])
        self.store.add_source(self.tweets, opinions)
        ok, msg = self.store.check_consistency(self.require_reasoner)
        self.log(f"[ingest] {len(self.tweets)} tweets, {sum(o.team is not None for o in opinions)} opinions "
                 f"extracted into the ontology; reasoner: {msg}")
        self.q4_src = self.r1.q4_source()
        self.log(f"[ingest] Report-1 Q4 (topic x group tweet counts): {self.q4_src}")
        self.view = ontology_view(self.store) if self.use_ctx else None
        self.log(f"[prompt] k={self.spec.k} sentences per team, focus={self.spec.focus or 'none (balanced)'}")
        return {"iteration": 0, "history": [], "stale": 0}

    def _write(self, p: dict, feedback: list[str] | None = None) -> None:
        """(Re)write the sentence of one planned claim. Retries when the model puts tweet ids into the
        sentence; after MAX_LLM_TRIES attempts the deterministic template sentence is used."""
        texts = [self.store.tweets[i]["text"] for i in p["ids"]]
        sent = None
        while sent is None and p["attempts"] < MAX_LLM_TRIES:
            p["attempts"] += 1
            sent = clean_sentence(self.backend.write_claim(p["team"], p["polarity"], p["intensity"],
                                                           p["quantifier"], texts, feedback,
                                                           p.get("aspect", ""), self.request))
        p["template"] = sent is None
        p["sentence"] = sent or template_sentence(p["team"], p["polarity"], p["intensity"], p["quantifier"],
                                                    p.get("aspect", ""))

    def draft(self, st: State) -> State:
        if not self.use_ctx:      # ablation: free-form draft without the ontology view
            claims = self.backend.draft_summary(self.tweets, self.request, None)
            self.log(f"[draft] {len(claims)} claims (free-form, no ontology plan)")
            return {"claims": claims}
        plan = plan_from_ontology(self.store, self.spec)
        for p in plan:
            self._write(p)
        self.log(f"[draft] {len(plan)} planned claims (team x sentiment x aspect), written one by one")
        return {"plan": plan, "claims": plan_claims(plan)}

    def evaluate(self, st: State) -> State:
        it = st["iteration"] + 1
        claims = st["claims"]
        opinions = self.backend.extract_opinions([c["sentence"] for c in claims])   # NLP/LLM extraction
        for k, c in enumerate(claims):     # template sentences: labels are known by construction, not re-guessed
            if c.get("template"):
                opinions[k] = OpinionRecord(c["team"], c["polarity"], c["intensity"], opinions[k].kind)
        iri = self.store.add_summary(it, claims, opinions)                           # facts -> ontology
        res = evaluate(self.store, iri, self.tol)                                    # SPARQL queries
        facts = self.backend.extract_facts([c["sentence"] for c in claims])          # facts for Report-1 Q1,2,3,5,6
        r1_vs, r1_status = self.r1.check(claims, facts)                              # DL / SPARQL / ASK on scratch ontology
        seen = {v.data.get("group") for v in res.violations if v.kind == "GROUP_UNDERREPRESENTED"}
        r1_vs += [v for v in self.r1.q4_violations(iri, self.tol) if v.data["group"] not in seen]   # Q4 per topic
        res.violations.extend(r1_vs)
        res.violations.extend(self._focus_violations(iri))
        res.metrics["prompt"] = {"k": self.spec.k, "focus": self.spec.focus}
        res.metrics["report1"] = r1_status
        res.metrics["q4_summary"] = self.r1.q4_summary(iri)
        entry = {"iteration": it, "score": res.score, "metrics": res.metrics, "claims": list(claims),
                 "violations": [{"kind": v.kind, "message": v.message} for v in res.violations]}
        history = st["history"] + [entry]
        best, stale = st.get("best"), st.get("stale", 0)
        if best is None or res.score < best["score"]:
            best, stale = entry, 0          # strictly better draft
        else:
            stale += 1
        self.log(f"[evaluate #{it}] score={res.score}  equal_coverage={res.metrics['equal_coverage']}  "
                 f"shares={res.metrics['group_shares']}")
        for v in res.violations:
            self.log(f"    - {v.kind}: {v.message[:140]}")
        return {"iteration": it, "result": res, "history": history, "best": best, "stale": stale}

    _CLAIM_FIXES = {
        "GROUP_MENTIONED": "it names a demographic group; describe only the sentiment about the team",
        "POLARITY_UNSUPPORTED": "its sentiment does not match the tweets; it must be clearly {pol}",
        "INTENSITY_INFLATED": "it sounds stronger than the tweets; use milder, more restrained wording",
        "CLAIM_NOT_LINKED": "it must name exactly one team explicitly and take a clear stance",
        "REPORT1_Q1_UNDERAGE_MARRIAGE": "it states a marriage involving a minor, which the ontology forbids; drop such facts",
        "REPORT1_Q2_IMPLAUSIBLE_AGE": "it states an implausible age; drop such facts",
        "REPORT1_Q3_DIET_CONFLICT": "it has someone eating a food the ontology says they cannot eat; drop such facts",
        "REPORT1_Q5_PLAYER_TEAM": "it gives a player a team that conflicts with another stated team; drop such facts",
        "REPORT1_Q6_EVENT_CYCLE": "it orders events cyclically; drop such facts",
    }

    def _focus_violations(self, iri: str, min_share: float = 0.2) -> list:
        """PROMPT_FOCUS_BIAS: the request asks for one side only (e.g. 'the complaints'), but another sentiment is
        held by >= 20% of the tweets about a team and the summary says nothing about it."""
        from evaluation import Violation
        if not self.spec.one_sided:
            return []
        said = {(c.opinion.team, c.opinion.polarity) for c in self.store.claims_of(iri) if c.opinion}
        by_team: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, o in self.store.source_opinion.items():
            if o.team and o.polarity != "Neutral":
                by_team[o.team][o.polarity].append(i)
        out = []
        for team, pols in by_team.items():
            tot = sum(len(v) for v in pols.values())
            for pol, ids in pols.items():
                if len(ids) >= 2 and len(ids) / tot >= min_share and (team, pol) not in said:
                    mix = self._balanced(ids)
                    out.append(Violation(
                        "PROMPT_FOCUS_BIAS",
                        f"The request only asks for {self.spec.focus.lower()} views, but {len(ids)} of {tot} opinionated "
                        f"tweets about {team} are {pol}. Add ONE neutral claim about this {pol} viewpoint.",
                        {"team": team, "polarity": pol, "tweet_ids": mix}))
        return out

    def _balanced(self, ids: list[int], per_group: int = MAX_PER_GROUP) -> list[int]:
        """Round-robin over author groups so an added claim cites every group that holds the viewpoint."""
        by_g: dict[str, list[int]] = defaultdict(list)
        for i in ids:
            by_g[self.store.tweets[i]["group"]].append(i)
        return [i for g in sorted(by_g) for i in by_g[g][:per_group]]

    def _add_tweets(self, plan: list[dict], ids: list[int], dirty: set) -> None:
        """Route source tweets into the planned claim with the same team + sentiment (+ aspect when it exists);
        open a new claim only when that team/sentiment has none yet."""
        for i in ids:
            rec = self.store.source_opinion.get(i)
            if rec is None or rec.team is None:
                continue
            same = [q for q in plan if q["team"] == rec.team and q["polarity"] == rec.polarity]
            p = next((q for q in same if q.get("aspect", "") == (rec.aspect if rec.aspect != "other" else "")), None) \
                or max(same, key=lambda q: q["support"], default=None)
            if p is None:
                p = {"team": rec.team, "polarity": rec.polarity, "aspect": "", "ids": [], "support": 0,
                     "intensity": rec.intensity, "quantifier": "Some", "sentence": "", "attempts": 0, "issues": []}
                plan.append(p)
            if i not in p["ids"]:
                p["ids"].append(i)
                p["support"] += 1
                p["intensity"] = _typical_intensity([self.store.source_opinion[j].intensity for j in p["ids"]])
            dirty.add(id(p))

    def revise(self, st: State) -> State:
        if "plan" not in st:      # ablation path
            claims = self.backend.revise_summary(self.tweets, self.request, st["claims"],
                                                 st["result"].violations, None)
            self.log(f"[revise] -> {len(claims)} claims")
            return {"claims": claims}
        plan = [dict(p, ids=list(p["ids"]), issues=[]) for p in st["plan"]]
        dirty: set[int] = set()
        for v in st["result"].violations:
            d = v.data
            if v.kind in self._CLAIM_FIXES:
                for p in plan:
                    if p["sentence"] == d.get("text"):
                        # temperature 0 would repeat the same sentence, so show it and forbid repeating it
                        p["issues"].append(f'the sentence "{p["sentence"]}" was rejected because '
                                           + self._CLAIM_FIXES[v.kind].format(pol=p["polarity"].upper())
                                           + "; write a clearly different, shorter, more restrained sentence")
                        dirty.add(id(p))
            elif v.kind == "GROUP_VIEWPOINT_OMITTED":
                self._add_tweets(plan, d["tweet_ids"][:MAX_PER_GROUP], dirty)
            elif v.kind == "GROUP_UNDERREPRESENTED":
                self._add_tweets(plan, d["tweet_ids"][:MAX_PER_GROUP + 1], dirty)
            elif v.kind == "PROMPT_FOCUS_BIAS":
                self._add_tweets(plan, d["tweet_ids"], dirty)
        for p in plan:
            if id(p) in dirty:
                self._write(p, p["issues"] or None)      # only flagged claims are rewritten
        self.log(f"[revise] rewrote {len(dirty)} of {len(plan)} claims")
        plan = _enforce_k(plan, self.spec.k or DEFAULT_K, self.spec.focus)
        return {"plan": plan, "claims": plan_claims(plan)}

    def finish(self, st: State) -> State:
        best = st["best"]
        text = render_summary(best["claims"])
        return {"summary": text}

    def route(self, st: State) -> str:
        if not st["result"].violations or st["iteration"] >= self.max_iters or st.get("stale", 0) >= 3:
            return "finish"
        return "revise"

    def _build(self):
        g = StateGraph(State)
        for name, fn in [("ingest", self.ingest), ("draft", self.draft), ("evaluate", self.evaluate),
                         ("revise", self.revise), ("finish", self.finish)]:
            g.add_node(name, fn)
        g.add_edge(START, "ingest"); g.add_edge("ingest", "draft"); g.add_edge("draft", "evaluate")
        g.add_conditional_edges("evaluate", self.route, {"revise": "revise", "finish": "finish"})
        g.add_edge("revise", "evaluate"); g.add_edge("finish", END)
        return g.compile()

    def run(self) -> dict[str, Any]:
        return self.graph.invoke({}, {"recursion_limit": 8 + 3 * self.max_iters})
