"""The one LLM function + ALL prompts (shared by data.py and agent.py, hence its own file).
REPORT: "Prompting" section. Rubric: >= 2 user prompts, system prompts, and an explanation of the
prompting technique used for each (persona, format constraint, few-shot, ...) + observed behaviour.
"""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import config





def call_llm(system, user):
    """Ollama chat using MODEL / TEMPERATURE / NUM_CTX / current seed from config. Same for baseline and agent."""
    ...

# USER prompts = the two simulated user requests (diagram input: "prompt"). {team} is filled per topic.
USER_PROMPTS = {
    1: "Summarize the opinions expressed about the NFL team in the provided tweets. Focus mainly on the strongest opinions.", 
    2: "Summarize the opinions expressed about the NFL team in the provided tweets. Focus on making sure your summary is fair, doesn’t favor or discriminate against any single type of fan and gives equal weight to positive, negative and neutral viewpoints."
}

# SYSTEM prompts = fixed instructions, one per LLM call type.
SYSTEM_ANNOTATE = "..."  # one-off, before step 1: label ONE tweet -> polarity / intensity / kind as JSON
SYSTEM_SUMMARY = "..."   # step 2 (and baseline): summarize tweets; MUST cite ids like [t3, t17] after every sentence
SYSTEM_FEEDBACK = "..."  # step 5: turn the list of detected issues into 2-4 concrete instructions (words only)
SYSTEM_REWRITE = "..."   # step 6: revise the summary using the feedback; keep the citations; change only what is needed
