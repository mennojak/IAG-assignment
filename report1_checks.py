"""The six ontology queries of Intermediate Report 1, applied to the facts of a summary draft.

Q1 underage marriage   (DL)     Person and (isMarriedTo some Minor)
Q2 implausible age     (DL)     Person and (hasAge some xsd:integer[> 110])
Q3 diet / allergy      (SPARQL) SELECT ?person ?food WHERE { ?person :cannotEat ?food }
Q4 topic x group count (SPARQL) per-topic social-group coverage (source data AND summary citations)
Q5 player-team         (DL)     Player and (playsFor value :Team)   (playsFor is functional, teams differ)
Q6 event ordering      (ASK)    ?a :followedBy ?b . ?b :followedBy ?a

Facts are extracted from the summary sentences (LLM / NLP), added to a SCRATCH copy of the ontology and
checked there with Pellet (Q1, Q2, Q3, Q5) or SPARQL (Q3, Q4, Q6); the scratch world is thrown away, so a
contradicting fact never corrupts the main world. A summary about NFL tweets rarely contains such facts, in
which case these checks pass vacuously; test_report1_queries.py shows each query firing on a bad claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from owlready2 import World, OwlReadyInconsistentOntologyError, sync_reasoner_pellet

from evaluation import Violation
from ontology_store import TEAM_TOPIC

Q3 = "SELECT ?person ?food WHERE { ?person :cannotEat ?food . }"
Q4 = """SELECT ?topic ?group (COUNT(?tweet) AS ?count) WHERE {
    VALUES ?topic { :49ersTopic :PatriotsTopic :SeahawksTopic }
    ?tweet :aboutTopic ?topic .
    ?author :authorsTweet ?tweet .
    ?author :belongsToSocialGroup ?g .
    ?g a ?group .
} GROUP BY ?topic ?group"""
Q6 = """ASK WHERE {
    ?eventA :followedBy ?eventB .
    ?eventB :followedBy ?eventA .
    FILTER(?eventA != ?eventB)
}"""
Q4_SUMMARY = """SELECT ?topic ?group (COUNT(DISTINCT ?tweet) AS ?count) WHERE {
    ?c r:claimOf ?summary ; r:citesTweet ?tweet .
    ?tweet i:aboutTopic ?topic .
    ?author i:authorsTweet ?tweet .
    ?author i:belongsToSocialGroup ?g .
    ?g a ?group . ?group rdfs:subClassOf i:SocialGroup .
} GROUP BY ?topic ?group"""


@dataclass
class Fact:
    """kind: marriage(a,b) | age(a,value) | plays_for(a=player,b=team) | eats(a=person,b=food) | before(a,b)"""
    kind: str
    a: str
    b: str | None = None
    value: int | None = None


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip()) or "x"


class Report1Checker:
    def __init__(self, store, source_counts_only: bool = False):
        self.store = store
        self.ln = store.ln
        self.owl = str(Path(store.owl_path).resolve())

    # ---------------------------------------------------------------- scratch world
    def _scratch(self):
        w = World()
        onto = w.get_ontology(f"file://{self.owl}").load()
        index: dict[str, object] = {}
        for ind in onto.individuals():
            keys = {_canon(ind.name)} | {_canon(n) for n in getattr(ind, "hasName", []) or []}
            for k in list(keys):
                if " " in k:
                    keys.add(k.split()[-1])             # last name: "brady" -> TomBrady
            for k in keys:
                index.setdefault(k, ind)
        return w, onto, index

    @staticmethod
    def _person(onto, index, name, cls="Person"):
        k = _canon(name)
        ind = index.get(k) or index.get(k.split()[-1] if k else k)
        if ind is None:
            ind = getattr(onto, cls)(f"x_{_slug(name)}")
            index[k] = ind
        return ind

    @staticmethod
    def _reason(w, onto, infer=False) -> bool:
        """True = consistent, False = Pellet reports the ontology inconsistent."""
        try:
            with onto:
                sync_reasoner_pellet(w, infer_property_values=infer, infer_data_property_values=infer, debug=0)
            return True
        except OwlReadyInconsistentOntologyError:
            return False

    def _q(self, w, onto, body):
        return w.as_rdflib_graph().query(f"PREFIX : <{onto.base_iri}>\n" + body)

    # ---------------------------------------------------------------- checks
    def _q1_q2(self, facts: list[Fact]):
        """Marriage/age facts of one claim -> Q1 / Q2. Returns 'Q1', 'Q2' or None."""
        ma = [f for f in facts if f.kind in ("marriage", "age")]
        if not ma:
            return None

        def build(only_ages: bool):
            w, onto, idx = self._scratch()
            for f in ma:
                if f.kind == "age" and f.value is not None:
                    self._person(onto, idx, f.a).hasAge.append(int(f.value))
                elif f.kind == "marriage" and not only_ages and f.b:
                    self._person(onto, idx, f.a).isMarriedTo.append(self._person(onto, idx, f.b))
            return w, onto
        w, onto = build(False)
        if self._reason(w, onto):
            return None
        w, onto = build(True)                      # does the age alone already clash?  -> Q2, else the marriage -> Q1
        return "Q2" if not self._reason(w, onto) else "Q1"

    def _q3(self, facts: list[Fact]):
        eats = [f for f in facts if f.kind == "eats" and f.b]
        if not eats:
            return []
        w, onto, idx = self._scratch()
        self._reason(w, onto, infer=True)
        banned = {(self.ln(r["person"]), self.ln(r["food"])) for r in map(lambda x: x.asdict(), self._q(w, onto, Q3))}
        hits = []
        for f in eats:
            p, food = idx.get(_canon(f.a)) or idx.get(_canon(f.a).split()[-1]), idx.get(_canon(f.b))
            if p is not None and food is not None and (p.name, food.name) in banned:
                hits.append(f)
        return hits

    def _q5(self, facts_by_claim: list[list[Fact]]):
        """Player->team facts over the WHOLE summary (+ the ontology's own ABox). Returns {player: [claim idx]}."""
        from llm import normalize_team
        by_player: dict[str, set] = {}
        where: dict[str, list[int]] = {}
        for k, fs in enumerate(facts_by_claim):
            for f in fs:
                team = normalize_team(f.b, f.b or "") if f.kind == "plays_for" else None
                if team:
                    by_player.setdefault(_canon(f.a), set()).add(team)
                    where.setdefault(_canon(f.a), []).append(k)
        flagged = {}
        for name, teams in by_player.items():
            w, onto, idx = self._scratch()
            p = self._person(onto, idx, name, "Player")
            if p.__class__.__name__ != "Player" and onto.Player not in p.is_a:
                p.is_a.append(onto.Player)
            for t in teams:
                onto._add_obj_triple_spo(p.storid, onto.playsFor.storid, getattr(onto, t).storid)
            if not self._reason(w, onto):
                flagged[name] = where[name]
        return flagged

    def _q6(self, facts_by_claim: list[list[Fact]]):
        """Event-order facts over the whole summary -> ASK cycle. Returns claim idxs involved, or []."""
        order = [(k, f) for k, fs in enumerate(facts_by_claim) for f in fs if f.kind == "before" and f.b]
        if not order:
            return []
        w, onto, idx = self._scratch()
        ev = {}
        for _, f in order:
            for n in (f.a, f.b):
                ev.setdefault(_canon(n), onto.Event(f"x_event_{_slug(n)}"))
        for _, f in order:
            ev[_canon(f.a)].followedBy.append(ev[_canon(f.b)])
        return sorted({k for k, _ in order}) if self._q(w, onto, Q6).askAnswer else []

    # ---------------------------------------------------------------- Q4 (source + summary)
    def q4_source(self) -> dict:
        """Report-1 Q4 verbatim on the loaded tweets -> {topic: {group: count}}."""
        out: dict[str, dict] = {}
        for r in self.store.graph.query(f"PREFIX : <{self.store.base}>\n" + Q4):
            r = r.asdict(); g = self.ln(r["group"])
            if g in ("AfricanAmerican", "Hispanic", "White"):
                out.setdefault(self.ln(r["topic"]), {})[g] = int(r["count"].toPython())
        return out

    def q4_summary(self, summary_iri: str) -> dict:
        out: dict[str, dict] = {}
        for r in self.store.sparql(Q4_SUMMARY, summary=summary_iri):
            out.setdefault(self.ln(r["topic"]), {})[self.ln(r["group"])] = int(r["count"].toPython())
        return out

    def q4_violations(self, summary_iri: str, tolerance: float) -> list[Violation]:
        """Per topic: a group whose share of the summary's cited tweets is below (1/k)(1-tol)."""
        vs, src = [], self.q4_source()
        for topic, cnt in self.q4_summary(summary_iri).items():
            groups = sorted(src.get(topic, cnt))
            tot = sum(cnt.get(g, 0) for g in groups) or 1
            for g in groups:
                if cnt.get(g, 0) / tot < (1 / len(groups)) * (1 - tolerance):
                    team = next((t for t, tp in TEAM_TOPIC.items() if tp == topic), None)
                    ids = [i for i, t in self.store.tweets.items()
                           if t["group"] == g and self.store.source_opinion[i].team == team]
                    vs.append(Violation("GROUP_UNDERREPRESENTED",
                              f"For the topic {topic} one author group has only {cnt.get(g, 0)} of {tot} cited tweets "
                              "(Report-1 query 4). Add claims about this topic that reflect these tweets (never name the group): "
                              + "; ".join(f'[{i}] "{self.store.tweets[i]["text"]}"' for i in ids[:4]),
                              {"group": g, "tweet_ids": ids, "topic": topic}))
        return vs

    # ---------------------------------------------------------------- driver
    def check(self, claims: list[dict], facts_by_claim: list[list[Fact]]) -> tuple[list[Violation], dict]:
        vs, status = [], {}
        msg = {"Q1": "states a marriage involving a minor (Report-1 query 1)",
               "Q2": "states an implausible age (Report-1 query 2)"}

        def flag(kind, k, why):
            vs.append(Violation(kind, f'The claim contradicts the ontology: it {why}. Remove or correct this factual '
                                      "statement; keep only what the cited tweets say.",
                                {"text": claims[k]["sentence"], "team": claims[k].get("team")}))
        q12 = q3 = 0
        for k, fs in enumerate(facts_by_claim):
            hit = self._q1_q2(fs)
            if hit:
                flag("REPORT1_Q1_UNDERAGE_MARRIAGE" if hit == "Q1" else "REPORT1_Q2_IMPLAUSIBLE_AGE", k, msg[hit])
            q12 += any(f.kind in ("marriage", "age") for f in fs)
            hits = self._q3(fs)
            for f in hits:
                flag("REPORT1_Q3_DIET_CONFLICT", k, f"has {f.a} eating {f.b}, which the ontology infers {f.a} cannot eat")
            q3 += any(f.kind == "eats" for f in fs)
        for player, ks in self._q5(facts_by_claim).items():
            for k in sorted(set(ks)):
                flag("REPORT1_Q5_PLAYER_TEAM", k, f"gives {player} a team that conflicts with another team stated for the same player")
        for k in self._q6(facts_by_claim):
            flag("REPORT1_Q6_EVENT_CYCLE", k, "puts events in a cyclic before/after order")
        flat = [f for fs in facts_by_claim for f in fs]
        status = {"Q1_Q2_facts_checked": q12, "Q3_facts_checked": q3,
                  "Q5_facts_checked": sum(f.kind == "plays_for" for f in flat),
                  "Q6_facts_checked": sum(f.kind == "before" for f in flat),
                  "flagged": sorted({v.kind for v in vs})}
        return vs, status
