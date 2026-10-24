"""Ontology access layer (Owlready2 + rdflib SPARQL + optional Pellet).

The Protege ontology is loaded read-only. Everything the agent adds (the tweets of the
current run, their extracted opinions, and the claims of each summary draft) is created
as individuals of YOUR classes (Tweet, Person, Opinion, Team, ...). The few extra
bookkeeping terms the agent needs (Run, Summary, claimOf, citesTweet, ...) live in a
small runtime ontology (`infoiag-runtime`), so infoiag_ontology.owl is never modified.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from owlready2 import (DataProperty, FunctionalProperty, ObjectProperty, Thing, World,
                       OwlReadyInconsistentOntologyError, sync_reasoner_pellet)

RT_IRI = "http://www.semanticweb.org/group11/ontologies/2026/infoiag-runtime#"

# Team individual -> Topic individual (the ontology has no direct Team-Topic property)
TEAM_TOPIC = {
    "SanFrancisco49ers": "49ersTopic",
    "NewEnglandPatriots": "PatriotsTopic",
    "SeattleSeahawks": "SeahawksTopic",
}
SOCIAL_GROUPS = ["AfricanAmerican", "Hispanic", "White"]
POLARITIES = ["Positive", "Negative", "Neutral"]
INTENSITIES = ["Low", "Medium", "High"]
KINDS = ["Support", "Criticism", "Prediction", "FactualClaim"]
# sub-themes of a team opinion; used to write specific claims instead of one generic claim per sentiment
ASPECTS = ["offense", "defense", "quarterback", "coaching", "roster", "game_result", "fans", "schedule_media",
           "prediction", "other"]


@dataclass
class OpinionRecord:
    """Result of NLP/LLM extraction for one tweet or one summary sentence."""
    team: str | None            # ontology Team individual name, or None if no team opinion
    polarity: str = "Neutral"
    intensity: str = "Medium"
    kind: str = "FactualClaim"
    aspect: str = ""            # what the opinion is about (see ASPECTS); only filled for source tweets


@dataclass
class ClaimInfo:
    iri: str
    text: str
    tweet_ids: list[int]
    opinion: OpinionRecord | None


class OntologyStore:
    def __init__(self, owl_path: str):
        self.world = World()
        path = Path(owl_path).resolve()
        self.onto = self.world.get_ontology(f"file://{path}").load()
        self.base = self.onto.base_iri
        self.owl_path = str(path)
        from owlready2 import destroy_entity     # Report-1 example tweets must not pollute the counts (file untouched)
        for n in ("Tweet1", "Tweet2", "Tweet3", "Author1", "Author2", "Author3", "Opinion1", "Opinion2", "Opinion3"):
            ex = self.world[self.base + n]
            if ex is not None:
                destroy_entity(ex)
        self.rt = self.world.get_ontology(RT_IRI)
        O = self.onto
        with self.rt:
            class Run(Thing): pass
            class Summary(Thing): pass
            class partOfRun(ObjectProperty): domain = [O.Tweet]; range = [Run]
            class forRun(ObjectProperty, FunctionalProperty): domain = [Summary]; range = [Run]
            class claimOf(ObjectProperty, FunctionalProperty): domain = [O.Opinion]; range = [Summary]
            class citesTweet(ObjectProperty): domain = [O.Opinion]; range = [O.Tweet]
            class tweetId(DataProperty, FunctionalProperty): range = [int]
            class claimText(DataProperty, FunctionalProperty): range = [str]
            class rank(DataProperty, FunctionalProperty): range = [int]
        self.Run, self.Summary = Run, Summary
        for i, name in enumerate(INTENSITIES, start=1):      # Low=1 < Medium=2 < High=3
            self._ind(name).rank = i
        self.run = Run("run_1")
        self.tweet_ind: dict[int, object] = {}
        self.tweets: dict[int, dict] = {}
        self.source_opinion: dict[int, OpinionRecord] = {}
        self.claims: dict[str, ClaimInfo] = {}
        self._summary_claims: dict[str, list[str]] = {}
        self.unclassified: dict[str, list[str]] = {}
        self._groups: dict[str, object] = {}
        self.graph = self.world.as_rdflib_graph()

    # ------------------------------------------------------------------ helpers
    def _ind(self, name: str):
        ind = self.world[self.base + name]
        if ind is None:
            raise KeyError(f"'{name}' not found in the ontology ({self.base})")
        return ind

    def team_names(self) -> list[str]:
        return list(TEAM_TOPIC)

    def _group(self, name: str):
        if name not in self._groups:
            if name not in SOCIAL_GROUPS:
                raise ValueError(f"unknown social group '{name}', expected one of {SOCIAL_GROUPS}")
            self._groups[name] = getattr(self.onto, name)(f"group_{name}")
        return self._groups[name]

    def _fill_opinion(self, op, rec: OpinionRecord):
        op.opinionAbout = self._ind(rec.team)
        op.hasPolarity = self._ind(rec.polarity)
        op.hasIntensity = self._ind(rec.intensity)
        op.hasKind = self._ind(rec.kind)

    # --------------------------------------------------------- populate the ABox
    def add_source(self, tweets: list[dict], opinions: list[OpinionRecord]):
        """tweets: [{id, group, text}] ; opinions: aligned extraction results."""
        O = self.onto
        for tw, rec in zip(tweets, opinions):
            i = int(tw["id"])
            author = O.Person(f"author_{i}")
            author.belongsToSocialGroup = self._group(tw["group"])
            t = O.Tweet(f"src_tweet_{i}")
            t.hasAuthor = author
            t.hasText = [tw["text"]]
            t.tweetId = i
            t.partOfRun = [self.run]
            topic = tw.get("team") or rec.team
            if topic:
                t.aboutTopic = self._ind(TEAM_TOPIC[topic])
            if rec.team:
                op = O.Opinion(f"src_opinion_{i}")
                self._fill_opinion(op, rec)
                t.expressesOpinion = [op]
            self.tweet_ind[i], self.tweets[i], self.source_opinion[i] = t, tw, rec

    def add_summary(self, iteration: int, claims: list[dict], opinions: list[OpinionRecord]) -> str:
        """claims: [{sentence, tweet_ids}] ; opinions aligned (extracted FROM the sentence)."""
        s = self.Summary(f"summary_{iteration}")
        s.forRun = self.run
        self._summary_claims[s.iri] = []
        self.unclassified[s.iri] = []
        for k, (cl, rec) in enumerate(zip(claims, opinions)):
            ids = [int(x) for x in cl.get("tweet_ids", []) if int(x) in self.tweet_ind]
            info = ClaimInfo(iri="", text=cl["sentence"], tweet_ids=ids, opinion=rec if rec.team else None)
            if not rec.team:
                self.unclassified[s.iri].append(cl["sentence"])
            if rec.team:
                op = self.onto.Opinion(f"claim_{iteration}_{k}")
                self._fill_opinion(op, rec)
                op.claimOf = s
                op.claimText = cl["sentence"]
                op.citesTweet = [self.tweet_ind[x] for x in ids]
                info.iri = op.iri
                self.claims[op.iri] = info
                self._summary_claims[s.iri].append(op.iri)
        return s.iri

    def claims_of(self, summary_iri: str) -> list[ClaimInfo]:
        return [self.claims[i] for i in self._summary_claims.get(summary_iri, [])]

    # ------------------------------------------------------------- SPARQL access
    _PREFIX = "PREFIX i: <{b}>\nPREFIX r: <{r}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"

    def sparql(self, body: str, **bindings):
        from rdflib import URIRef
        q = self._PREFIX.format(b=self.base, r=self.rt.base_iri) + body
        init = {k: URIRef(v) for k, v in bindings.items()}
        return [res.asdict() for res in self.graph.query(q, initBindings=init)]

    @staticmethod
    def ln(term) -> str:
        s = str(term)
        return s.rsplit("#", 1)[-1] if "#" in s else s

    # --------------------------------------------------------------- reasoning
    _Q_MULTI_GROUP = """SELECT ?p (COUNT(DISTINCT ?g) AS ?n) WHERE { ?p i:belongsToSocialGroup ?g }
        GROUP BY ?p HAVING (COUNT(DISTINCT ?g) > 1)"""
    _Q_MULTI_POLARITY = """SELECT ?o (COUNT(DISTINCT ?x) AS ?n) WHERE { ?o i:hasPolarity ?x }
        GROUP BY ?o HAVING (COUNT(DISTINCT ?x) > 1)"""

    def sparql_consistency(self) -> list[str]:
        """Java-free sanity checks (run always): one social group per author, one polarity per opinion."""
        issues = []
        for q, what in ((self._Q_MULTI_GROUP, "author in several social groups"),
                        (self._Q_MULTI_POLARITY, "opinion with several polarities")):
            issues += [f"{what}: {self.ln(r[next(iter(r))])}" for r in self.sparql(q)]
        return issues

    def check_consistency(self, require_reasoner: bool = False) -> tuple[bool, str]:
        """SPARQL sanity checks always run. Pellet (needs Java) runs on top when available.
        With require_reasoner=True a missing/failed reasoner raises instead of being skipped."""
        issues = self.sparql_consistency()
        if issues:
            return False, "inconsistent facts: " + "; ".join(issues[:3])
        if shutil.which("java") is None:
            msg = ("Pellet NOT run: 'java' not found on PATH (install a JDK, reopen the terminal, check "
                   "`java -version`); SPARQL consistency checks passed")
            if require_reasoner:
                raise RuntimeError(msg)
            return True, msg
        try:
            with self.onto:
                sync_reasoner_pellet(self.world, infer_property_values=False,
                                     infer_data_property_values=False, debug=0)
            return True, "consistent (Pellet + SPARQL checks)"
        except OwlReadyInconsistentOntologyError:
            return False, "ontology is INCONSISTENT after adding the extracted facts"
        except Exception as e:
            lines = [l.strip() for l in str(e).splitlines() if l.strip()]
            first = next((l for l in lines if "Exception" in l or "Error" in l and "message is" not in l),
                         lines[-1] if lines else "")[:300]
            msg = f"Pellet failed ({type(e).__name__}: {first}); SPARQL consistency checks passed"
            if require_reasoner:
                raise RuntimeError(msg) from e
            return True, msg
