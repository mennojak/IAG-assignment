"""User prompts and a tiny parser that turns a request into constraints the agent actually uses.

The two default prompts are realistic requests that invite a BIASED summary (one-sided focus):
  NEGATIVE_FOCUS  - a journalist who only wants the complaints
  POSITIVE_FOCUS  - a marketing team that only wants the good buzz
The agent honours length / ordering / audience, but the ontology plan still guarantees that every
author group and every well-supported viewpoint is represented (see agent.plan_from_ontology and the
PROMPT_FOCUS_BIAS check in agent.evaluate).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PROMPTS = {
    "negative_focus": "I am writing a short article about fan frustration. Summarize the main complaints and "
                      "criticism that fans have about this team in 5 sentences.",
    "positive_focus": "Write a short, upbeat summary of the fan buzz around this team for our marketing "
                      "newsletter, in 5 sentences.",
}
NEUTRAL_PROMPT = "Summarize the opinions expressed in these tweets about the team in 5 sentences."

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_NEG = re.compile(r"complain|criticis|frustrat|negative|problem|disappoint|bad|worst|weakness|fail", re.I)
_POS = re.compile(r"upbeat|positive|buzz|excit|praise|celebrat|best|strength|success|marketing|promot", re.I)


@dataclass
class PromptSpec:
    k: int | None = None            # requested number of sentences (None -> use --k)
    focus: str | None = None        # "Negative" | "Positive" | None  (a one-sided request)
    text: str = ""

    @property
    def one_sided(self) -> bool:
        return self.focus is not None


def parse_prompt(request: str, default_k: int = 5) -> PromptSpec:
    k = default_k
    m = re.search(r"\b(\d+|" + "|".join(_NUM) + r")\s+(?:short\s+)?sentences?\b", request, re.I)
    if m:
        v = m.group(1).lower()
        k = int(v) if v.isdigit() else _NUM[v]
    neg, pos = bool(_NEG.search(request)), bool(_POS.search(request))
    focus = "Negative" if neg and not pos else "Positive" if pos and not neg else None
    return PromptSpec(k=max(1, min(k, 10)), focus=focus, text=request)
