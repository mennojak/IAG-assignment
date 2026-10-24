"""LLM backends. `LangChainBackend` is the real agent (any LangChain chat model);
`HeuristicBackend` is a deterministic offline stand-in used to test the pipeline without an API key."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from ontology_store import ASPECTS, INTENSITIES, KINDS, POLARITIES, TEAM_TOPIC, OpinionRecord

TEAM_LABEL = {"SanFrancisco49ers": "San Francisco 49ers",
              "NewEnglandPatriots": "New England Patriots",
              "SeattleSeahawks": "Seattle Seahawks"}


def normalize_team(value: str | None, text: str = "") -> str | None:
    """Map whatever the model wrote ('Patriots', 'New England Patriots', ...) to an ontology Team id.
    Falls back to scanning the sentence itself."""
    if value in TEAM_TOPIC:
        return value
    for src in (value or "", text or ""):
        low = src.lower()
        for team, pat in _ALIASES.items():
            if re.search(pat, low):
                return team
    return None


_ADV = {"Low": "mildly ", "Medium": "", "High": "strongly "}
_ID_REF = re.compile(r"\btweets?\s*[\[\(#]?\s*\d|[\[\(]\s*\d+\s*(?:,\s*\d+\s*)*[\]\)]", re.I)


ASPECT_PHRASE = {"offense": "the offense", "defense": "the defense", "quarterback": "the quarterback play",
                 "coaching": "the coaching", "roster": "the roster moves", "game_result": "recent game results",
                 "fans": "the fan base", "schedule_media": "game scheduling and broadcasts",
                 "prediction": "the season outlook"}


def template_sentence(team: str, polarity: str, intensity: str, quantifier: str, aspect: str = "") -> str:
    """Deterministic, neutral fallback used when the LLM keeps failing the ontology checks."""
    label = TEAM_LABEL[team]
    ph = ASPECT_PHRASE.get(aspect)
    if polarity == "Neutral":
        return f"{quantifier} tweets mention the {label}{' and ' + ph if ph else ''} without taking a clear side."
    about = f"about {ph} of the {label}" if ph else f"about the {label}"
    return f"{quantifier} tweets express {_ADV[intensity]}{polarity.lower()} views {about}."


def clean_sentence(s: str | None) -> str | None:
    """One clean sentence, or None when the model wrote tweet ids / multiple lines into it."""
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip().strip('"').strip()
    if not s or _ID_REF.search(s):
        return None
    return s


class Backend(Protocol):
    def extract_opinions(self, texts: list[str], topic_teams: list[str | None] | None = None) -> list[OpinionRecord]: ...
    def draft_summary(self, tweets: list[dict], request: str, ontology_view: str | None) -> list[dict]: ...
    def revise_summary(self, tweets: list[dict], request: str, claims: list[dict],
                       violations: list, ontology_view: str | None) -> list[dict]: ...
    def write_claim(self, team: str, polarity: str, intensity: str, quantifier: str,
                    texts: list[str], feedback: list[str] | None = None, aspect: str = "",
                    request: str = "") -> str: ...
    def plain_summary(self, tweets: list[dict], request: str) -> str: ...
    def extract_facts(self, texts: list[str]) -> list[list]: ...


# ===================================================================== LangChain (real)
class _Extracted(BaseModel):
    n: int = Field(0, description="the number of the input item this annotation belongs to")
    team: Optional[str] = Field(None, description="Team the opinion is about, one of the allowed team ids, or null")
    polarity: str = Field("Neutral", description="Positive | Negative | Neutral")
    intensity: str = Field("Medium", description="Low | Medium | High")
    kind: str = Field("FactualClaim", description="Support | Criticism | Prediction | FactualClaim")
    aspect: str = Field("other", description="what the opinion is about: " + " | ".join(ASPECTS))


class _ExtractedBatch(BaseModel):
    items: list[_Extracted] = Field(description="exactly one item per input, same order")


class _Claim(BaseModel):
    sentence: str = Field(description="ONE sentence about ONE team; no demographic group names")
    tweet_ids: list[int] = Field(description="ids of the tweets that support this sentence")


class _Summary(BaseModel):
    claims: list[_Claim]


class _FactItem(BaseModel):
    n: int = Field(description="number of the input sentence this fact comes from")
    kind: str = Field(description="marriage | age | plays_for | eats | before")
    a: str = Field(description="marriage: first person; age: person; plays_for: player; eats: person; before: earlier event")
    b: Optional[str] = Field(None, description="marriage: second person; plays_for: team name; eats: food; before: later event")
    value: Optional[int] = Field(None, description="age in years (kind=age only)")


class _FactBatch(BaseModel):
    facts: list[_FactItem] = Field(default_factory=list, description="only facts literally stated; empty list if none")


class _OneSentence(BaseModel):
    sentence: str = Field(description="exactly one neutral sentence, no tweet ids, no group names, no quotes")


def _tweets_block(tweets: list[dict]) -> str:
    return "\n".join(f'[{t["id"]}] (group: {t["group"]}) {t["text"]}' for t in tweets)


class LangChainBackend:
    def __init__(self, model: str = "anthropic:claude-sonnet-5-5", temperature: float | None = None):
        from langchain.chat_models import init_chat_model
        kwargs = {} if temperature is None else {"temperature": temperature}
        self.batch = 40
        if model.startswith("ollama:"):
            # Ollama's default context (~4k tokens) would silently truncate the prompts
            kwargs.setdefault("temperature", 0)
            kwargs.setdefault("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", "16384")))
            self.batch = 15                       # small local models do better on small batches
        self.llm = init_chat_model(model, **kwargs)
        self._teams = ", ".join(TEAM_TOPIC)

    def _structured(self, schema, system: str, user: str):
        msgs = [("system", system), ("human", user)]
        try:
            return self.llm.with_structured_output(schema, method="json_schema").invoke(msgs)
        except Exception:      # fallback for models/providers without native JSON-schema output
            return self.llm.with_structured_output(schema).invoke(msgs)

    def extract_opinions(self, texts, topic_teams=None, batch: int | None = None):
        batch = batch or self.batch
        system = (
            "You annotate short texts (tweets or summary sentences) for an ontology of opinions.\n"
            f"Allowed team ids: {self._teams}. Use null when the text expresses no opinion about the relevant team.\n"
            "If an item has a [topic: TEAM] tag, annotate the opinion toward THAT team (the tweet was collected "
            "about it); use null if the tweet merely mentions the team, asks a question, or is about something else.\n"
            "Without a tag, annotate the team the text is mainly about.\n"
            f"polarity: {POLARITIES}; intensity: {INTENSITIES} (hedged/mild = Low, plain = Medium, "
            f"emphatic/swearing/'greatest ever'/many exclamation marks = High); kind: {KINDS} "
            "(Support = praise/backing, Criticism = complaint/hate, Prediction = claim about the future, "
            "FactualClaim = neutral statement); aspect: " + " | ".join(ASPECTS) + " (the specific thing the opinion is about: "
            "offense, defense, the quarterback, coaching, roster moves/trades, a game result, the fans, "
            "schedule/TV broadcast, a season prediction, or other). Tweets are informal and use slang/AAE/Spanglish: interpret them "
            "charitably and do not read dialect or profanity as negativity by itself.\n"
            "Return one item per input with n = the input number.")
        recs: list[OpinionRecord] = []
        for s0 in range(0, len(texts), batch):
            chunk = texts[s0:s0 + batch]
            hints = (topic_teams or [None] * len(texts))[s0:s0 + batch]
            user = "\n".join(f"{i+1}. " + (f"[topic: {h}] " if h else "") + t
                             for i, (t, h) in enumerate(zip(chunk, hints)))
            by_n = {e.n: e for e in self._structured(_ExtractedBatch, system, user).items}
            for i, h in enumerate(hints):
                e = by_n.get(i + 1)
                if e is None:
                    recs.append(OpinionRecord(None)); continue
                if h:      # tweet collected about a team: opinion is toward that team, or none
                    team = h if (e.team and h in TEAM_TOPIC) else None
                else:      # summary sentence: be tolerant about how the model names the team
                    team = normalize_team(e.team, chunk[i])
                recs.append(OpinionRecord(team,
                                          e.polarity if e.polarity in POLARITIES else "Neutral",
                                          e.intensity if e.intensity in INTENSITIES else "Medium",
                                          e.kind if e.kind in KINDS else "FactualClaim",
                                          e.aspect if e.aspect in ASPECTS else "other"))
        return recs

    _RULES = (
        "Rules (they are checked automatically against an ontology afterwards):\n"
        "1. Every sentence is about exactly ONE team and lists the ids of the tweets it is based on.\n"
        "2. The sentence's sentiment (positive/negative/neutral) must be expressed by at least one cited tweet "
        "about that team. Never generalise beyond the cited tweets and never exaggerate their intensity.\n"
        "3. Equal coverage: every social group in the data must contribute a comparable number of claims, "
        "and every group's viewpoint on every team it discussed must be reflected. Do not let one group dominate.\n"
        "4. NEVER name demographic groups (African American, Hispanic, White, 'community', ...) in a sentence, "
        "and never write tweet ids inside a sentence: groups are tracked automatically from the tweet_ids field.\n"
        "5. ONE sentiment per sentence. If tweets are split, write separate sentences (e.g. one positive, one negative); "
        "never write 'mixed feelings' or 'some ... but others ...' in one sentence.\n"
        "6. Write neutral reporting style, e.g. 'Several tweets praise the Patriots' defence and Brady.'")

    def _draft_global(self, tweets, request, ontology_view):
        ctx = f"\nStructured view of the data retrieved from the ontology:\n{ontology_view}\n" if ontology_view else ""
        system = "You write neutral, bias-free summaries of tweet collections.\n" + self._RULES
        user = f"Request: {request}\n{ctx}\nTweets:\n{_tweets_block(tweets)}"
        return self._post(self._structured(_Summary, system, user).claims)

    def _revise_global(self, tweets, request, claims, violations, ontology_view):
        ctx = f"\nStructured view of the data retrieved from the ontology:\n{ontology_view}\n" if ontology_view else ""
        system = ("You revise a tweet summary so that it passes automatic bias/consistency checks.\n"
                  + self._RULES + "\nKeep the claims that were not flagged; fix, replace or add only what the feedback requires.")
        prev = "\n".join(f'- "{c["sentence"]}" (tweets {c["tweet_ids"]})' for c in claims)
        fb = "\n".join(f"- [{v.kind}] {v.message}" for v in violations)
        user = (f"Request: {request}\n{ctx}\nTweets:\n{_tweets_block(tweets)}\n\n"
                f"Current summary claims:\n{prev}\n\nProblems found by the ontology queries:\n{fb}")
        return self._post(self._structured(_Summary, system, user).claims)

    def extract_facts(self, texts):
        """Facts needed by Report-1 queries 1,2,3,5,6 (marriage, age, eats, player->team, event order).
        Only facts literally stated in the sentence; opinions/sentiment are NOT facts."""
        from report1_checks import Fact
        system = ("Extract ONLY explicit factual statements from the numbered sentences, as structured facts: "
                  "marriage(a,b), age(a,value), plays_for(a=player,b=team), eats(a=person,b=food), "
                  "before(a=earlier event,b=later event). Do not extract opinions, predictions or sentiment. "
                  "Most sentences contain no such fact: return an empty list then.")
        out: list[list] = [[] for _ in texts]
        for s0 in range(0, len(texts), self.batch):
            chunk = texts[s0:s0 + self.batch]
            user = "\n".join(f"{i+1}. {t}" for i, t in enumerate(chunk))
            for f in self._structured(_FactBatch, system, user).facts:
                if 1 <= f.n <= len(chunk) and f.kind in ("marriage", "age", "plays_for", "eats", "before") and f.a:
                    out[s0 + f.n - 1].append(Fact(f.kind, f.a, f.b, f.value))
        return out

    def write_claim(self, team, polarity, intensity, quantifier, texts, feedback=None, aspect="", request=""):
        """Verbalise ONE planned claim. The plan (team, polarity, aspect, tweets) comes from the ontology;
        the LLM only chooses the wording. It never sees author groups or tweet ids."""
        label = TEAM_LABEL[team]
        level = {"Low": "mild and hedged", "Medium": "plain and restrained (avoid words like strong, unwavering, "
                 "overwhelming, passionate, fierce)", "High": "emphatic"}[intensity]
        topic = (f"- The sentence is specifically about {ASPECT_PHRASE[aspect]}. Name concrete details that appear "
                 "in the tweets (players, games, events) instead of generic phrases.\n") if aspect in ASPECT_PHRASE else ""
        req = (f"- The user's request was: \"{request}\". Follow its audience and style, but the sentence must stay "
               "neutral and faithful to the tweets: do not add praise or criticism that the tweets do not "
               "contain and do not take sides.\n") if request else ""
        system = (
            "You write ONE neutral sentence that summarises what a few tweets say about an NFL team.\n"
            f"- The sentence is about the {label} only, and names the team.\n" + topic + req +
            f"- Overall sentiment must be clearly {polarity.upper()} (Positive = praise/support, Negative = "
            "criticism/disappointment, Neutral = no clear stance).\n"
            f"- Wording strength: {level}. Do not use intensifiers (strongly, overwhelmingly, furious, "
            "love, hate) unless the strength is emphatic.\n"
            f"- Begin with '{quantifier} tweets' or '{quantifier} fans'.\n"
            "- Do not quote tweets, do not mention tweet numbers or ids, and never mention demographic groups, "
            "communities or fans' ethnicity.\n"
            "- Do not add facts that are not in the tweets. Keep it under 30 words.")
        user = "Tweets:\n" + "\n".join(f"- {t}" for t in texts)
        if feedback:
            user += "\n\nYour previous sentence was rejected by an automatic check: " + "; ".join(feedback)
        return self._structured(_OneSentence, system, user).sentence

    def plain_summary(self, tweets: list[dict], request: str) -> str:
        """BASELINE: plain LLM, no ontology, no plan, no checks; sees only the tweets and the request."""
        system = "You summarise collections of tweets. Follow the user's request."
        user = f"Request: {request}\n\nTweets:\n" + "\n".join(f"- {t['text']}" for t in tweets)
        return str(self.llm.invoke([("system", system), ("human", user)]).content).strip()

    # ------------------------------------------------------------------ per-team generation
    _REF = re.compile(r"\s*[\(\[]\s*(?:tweets?|ids?)[:\s#]*([\d,\s#&and]+?)\s*[\)\]]", re.I)

    def _post(self, claims) -> list[dict]:
        """Strip tweet references the model wrote into the sentence (and rescue their ids)."""
        out = []
        for c in claims:
            d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
            ids = [int(x) for x in d.get("tweet_ids", [])]
            for m in self._REF.finditer(d["sentence"]):
                ids += [int(x) for x in re.findall(r"\d+", m.group(1))]
            d["sentence"] = re.sub(r"\s+", " ", self._REF.sub("", d["sentence"])).strip()
            d["tweet_ids"] = list(dict.fromkeys(ids))
            if d["sentence"]:
                out.append(d)
        return out

    @staticmethod
    def _team_view(view: str | None, team: str) -> str | None:
        if not view:
            return None
        return "\n".join(l for l in view.splitlines() if l.startswith("-") is False or f"team={team}" in l)

    def _can_split(self, tweets) -> bool:
        return all(t.get("team") in TEAM_TOPIC for t in tweets)

    def _team_call(self, team, tweets, request, view, prev=None, feedback=None):
        label = TEAM_LABEL[team]
        system = ("You write neutral, bias-free summaries of tweet collections.\n" + self._RULES +
                  f"\nYou are summarising ONLY the tweets about the {label}. Write 3 to 8 claims, each about the {label}, "
                  "covering every social group's viewpoint, in proportion and without exaggeration."
                  + ("\nKeep claims that were not flagged; fix, replace or add only what the feedback requires." if prev else ""))
        ctx = f"\nOntology view:\n{view}\n" if view else ""
        user = f"Request: {request}\n{ctx}\nTweets about the {label}:\n{_tweets_block(tweets)}"
        if prev is not None:
            user += ("\n\nCurrent claims:\n" + "\n".join(f'- "{c["sentence"]}" (tweets {c["tweet_ids"]})' for c in prev)
                     + "\n\nFix these problems found by the ontology queries:\n"
                     + "\n".join(f"- {v.message}" for v in feedback))
        res = self._post(self._structured(_Summary, system, user).claims)
        for c in res:
            c["team"] = team
        return res

    def draft_summary(self, tweets, request, ontology_view):
        if not self._can_split(tweets):
            return self._draft_global(tweets, request, ontology_view)
        out = []
        for team in TEAM_TOPIC:
            tt = [t for t in tweets if t["team"] == team]
            if tt:
                out += self._team_call(team, tt, request, self._team_view(ontology_view, team))
        return out

    def revise_summary(self, tweets, request, claims, violations, ontology_view):
        if not self._can_split(tweets):
            return self._revise_global(tweets, request, claims, violations, ontology_view)
        out = []
        for team in TEAM_TOPIC:
            tt = [t for t in tweets if t["team"] == team]
            mine = [c for c in claims if (c.get("team") or normalize_team(None, c["sentence"])) == team]
            sents = {c["sentence"] for c in mine}
            vs = [v for v in violations if v.data.get("team") == team or v.data.get("text") in sents]
            if tt and vs:
                out += self._team_call(team, tt, request, self._team_view(ontology_view, team), mine, vs)
            else:
                out += mine
        # claims whose team could not be determined are dropped (they were flagged as CLAIM_NOT_LINKED)
        return out


# ===================================================================== offline stand-in
_ALIASES = {"SanFrancisco49ers": r"49ers|niners|san francisco",
            "NewEnglandPatriots": r"patriots|pats\b|gopats|new england",
            "SeattleSeahawks": r"seahawks|seattle|hawks"}
_POS = r"love|great|greatest|best|excited|win|beat|well|good|amazing|awesome|positive"
_NEG = r"hate|pissed|sick of|terrible|awful|worst|overrated|furious|bad|negative"
_HIGH = r"!!|furious|greatest|worst|hate|pissed|awful|terrible|strongly|so excited"
_LOW = r"mildly|slightly|ok\b|nothing special|kind of"


_ASPECT_KW = {"quarterback": r"\bqb\b|quarterback|brady|kaepernick|wilson|garoppolo|\bkap\b",
              "defense": r"defen[cs]e|d-line|secondary|legion of boom|sack",
              "offense": r"offen[cs]e|touchdown|running back|receiver|rushing",
              "coaching": r"coach|belichick|harbaugh|carroll",
              "roster": r"trade|sign|draft|released|roster|cut\b|leaving|joining|going to the",
              "schedule_media": r"channel|broadcast|\btv\b|stream|schedule|kickoff|cruise|watch",
              "game_result": r"\bbeat|\blost|\bloss|score|\bwon\b|\bwin\b|defeat",
              "prediction": r"gonna|going to win|super bowl|playoff|season|predict",
              "fans": r"fans?\b|stadium|12th man|crowd"}


def _aspect(text):
    return next((a for a, pat in _ASPECT_KW.items() if re.search(pat, text.lower())), "other")


def _kind(text, pol):
    if re.search(r"gonna|going to|will |predict", text): return "Prediction"
    return {"Positive": "Support", "Negative": "Criticism"}.get(pol, "FactualClaim")


class HeuristicBackend:
    """Keyword NLP + a deliberately naive first draft. For testing the loop offline only."""

    def extract_opinions(self, texts, topic_teams=None):
        out = []
        for idx, t in enumerate(texts):
            hint = topic_teams[idx] if topic_teams else None
            low = t.lower()
            hits = [(m.start(), team) for team, pat in _ALIASES.items() if (m := re.search(pat, low))]
            if not hits and not hint:
                out.append(OpinionRecord(None)); continue
            team = hint or min(hits)[1]
            neg = len(re.findall(_NEG, low)) - (1 if "not scared" in low else 0)
            pos = len(re.findall(_POS, low)) + (1 if "not scared" in low else 0)
            if re.search(r"\bpositive\b", low): pos += 2
            if re.search(r"\bnegative\b", low): neg += 2
            pol = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
            if hint and pol == "Neutral":
                out.append(OpinionRecord(None)); continue
            inten = "High" if re.search(_HIGH, low) else "Low" if re.search(_LOW, low) else "Medium"
            out.append(OpinionRecord(team, pol, inten, _kind(low, pol), _aspect(low)))
        return out

    # claim sentences are generated from a template so the keyword extractor can read them back
    @staticmethod
    def _sentence(team, pol, inten):
        adv = {"Low": "mildly ", "Medium": "", "High": "strongly "}[inten]
        return f"Some tweeters were {adv}{pol.lower()} about the {TEAM_LABEL[team]}."

    def _clusters(self, tweets):
        recs = self.extract_opinions([t["text"] for t in tweets], [t.get("team") for t in tweets])
        cl = defaultdict(list)
        for t, r in zip(tweets, recs):
            if r.team:
                cl[(t["group"], r.team, r.polarity)].append((int(t["id"]), r))
        return cl

    def _claim(self, items):
        ids = [i for i, _ in items]
        top = max((r.intensity for _, r in items), key=INTENSITIES.index)
        r0 = items[0][1]
        return {"sentence": self._sentence(r0.team, r0.polarity, top), "tweet_ids": ids}

    def extract_facts(self, texts):
        from report1_checks import Fact
        out = []
        for t in texts:
            fs = []
            for m in re.finditer(r"([A-Z][\w']+(?: [A-Z][\w']+)?) (?:is|was) (\d{1,3}) years? old", t):
                fs.append(Fact("age", m.group(1), value=int(m.group(2))))
            for m in re.finditer(r"([A-Z][\w']+(?: [A-Z][\w']+)?) (?:married|is married to|got married to) ([A-Z][\w']+(?: [A-Z][\w']+)?)", t):
                fs.append(Fact("marriage", m.group(1), m.group(2)))
            for m in re.finditer(r"([A-Z][\w']+(?: [A-Z][\w']+)?) plays for the ([A-Za-z0-9 ]+?)(?:[.,]| and |$)", t):
                fs.append(Fact("plays_for", m.group(1), m.group(2)))
            for m in re.finditer(r"([A-Z][\w']+(?: [A-Z][\w']+)?) (?:ate|eats|eating) (?:some )?([A-Za-z]+)", t):
                fs.append(Fact("eats", m.group(1), m.group(2)))
            for m in re.finditer(r"([A-Z][\w ]+?) happened before ([A-Z][\w ]+?)[.,]", t):
                fs.append(Fact("before", m.group(1), m.group(2)))
            out.append(fs)
        return out

    def write_claim(self, team, polarity, intensity, quantifier, texts, feedback=None, aspect="", request=""):
        return template_sentence(team, polarity, intensity, quantifier, aspect)

    def plain_summary(self, tweets, request):
        return " ".join(t["text"] for t in tweets[:5])      # offline stand-in baseline: first five tweets

    def draft_summary(self, tweets, request, ontology_view):
        # naive: only looks at the first tweets (position bias) + one unsupported generalisation
        first = [t for t in tweets if int(t["id"]) <= 4]
        claims = [self._claim(v) for v in self._clusters(first).values()]
        claims.append({"sentence": "Fans were strongly positive about the Seattle Seahawks.", "tweet_ids": [4]})
        return claims

    def revise_summary(self, tweets, request, claims, violations, ontology_view):
        cl = self._clusters(tweets)
        by_text = {c["sentence"]: c for c in claims}
        for v in violations:
            d = v.data
            if v.kind in ("POLARITY_UNSUPPORTED", "INTENSITY_INFLATED"):
                by_text.pop(d["text"], None)           # drop; re-added below from the source clusters
        for v in violations:
            d = v.data
            if v.kind == "GROUP_UNDERREPRESENTED":
                for (g, team, pol), items in cl.items():
                    if g == d["group"]:
                        c = self._claim(items); by_text.setdefault(c["sentence"], {"sentence": c["sentence"], "tweet_ids": []})
                        by_text[c["sentence"]]["tweet_ids"] = sorted(set(by_text[c["sentence"]]["tweet_ids"] + c["tweet_ids"]))
            elif v.kind == "GROUP_VIEWPOINT_OMITTED":
                for (g, team, pol), items in cl.items():
                    if g == d["group"] and team == d["team"] and pol == d["polarity"]:
                        c = self._claim(items); by_text.setdefault(c["sentence"], {"sentence": c["sentence"], "tweet_ids": []})
                        by_text[c["sentence"]]["tweet_ids"] = sorted(set(by_text[c["sentence"]]["tweet_ids"] + c["tweet_ids"]))
        return list(by_text.values())
