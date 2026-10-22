"""The agent (LangGraph) and the baseline.  Functions are numbered like the architecture figure.
REPORT: "Agent Architecture": figure, nodes table, state, flow.

    input (dataset + prompt)
      step 1 preprocessing -> step 2 summary maker -> step 3 information extractor
      -> step 4 ontology reasoning tool -> step 5 bias consistency evaluator -> [problem found?]
          yes -> step 6 rewrite -> back to step 3        no -> return the summary
    (add to the figure: the loop also stops when MAX_REVISIONS is reached)
"""
from typing import TypedDict
import config, data, ontology
from llm import call_llm, SYSTEM_SUMMARY, SYSTEM_FEEDBACK, SYSTEM_REWRITE

# ---------------- State   REPORT: "State" table (paste as the typed schema) ----------------
class AgentState(TypedDict):
    topic: str; prompt_id: int; user_prompt: str
    tweets: list; opinions: list          # set in step 1
    summary: str
    claims: list                          # [{"id", "sentence", "team", "cited_tweet_ids"}]
    ontology_results: dict                # format defined in ontology.check()
    issues: list                          # [{"type", "evidence", "claim_ids"}]
    issue_history: list                   # issues per iteration (for error analysis)
    feedback: str
    revision_count: int
    consistent: bool

# ---------------- Steps ----------------
def step1_preprocessing(s):      # TOOL
    # s["tweets"] = data.load_tweets(topic); s["opinions"] = data.load_opinions(topic)
    # revision_count = 0; issue_history = []
    ...

def step2_summary_maker(s):      # LLM
    # summary = call_llm(SYSTEM_SUMMARY, user prompt + tweets formatted as "[t3] text")
    ...

def step3_information_extractor(s):   # TOOL (not an LLM: an 8B model is unreliable at structured output)
    # split the summary into sentences; read the [tN] citations with a regex; keep only ids that exist
    # -> claims. (Report change: "claims with cited tweets" replace subject-relation-object triplets.)
    ...

def step4_ontology_reasoning_tool(s):   # TOOL
    # s["ontology_results"] = ontology.check(tweets, opinions, claims, topic)
    ...

def step5_bias_consistency_evaluator(s):   # DECISION (fixed rules) + LLM only for the wording
    # inconsistent ontology                                              -> INCONSISTENT
    # min(coverage_by_group) < MIN_RATIO * max(coverage_by_group)        -> GROUP_SKEW
    # a polarity's represented share < MIN_RATIO * its available share   -> POLARITY_SKEW
    # unsupported claims -> UNSUPPORTED ; mismatched claims -> MISATTRIBUTED
    # store issues, append to issue_history, consistent = (no issues)
    # if issues: feedback = call_llm(SYSTEM_FEEDBACK, issues)
    ...

def problem_found(s):            # the question block = CONDITIONAL EDGE
    # "rewrite" if issues exist and revision_count < MAX_REVISIONS, else "end" (return the summary)
    ...

def step6_rewrite(s):            # LLM
    # summary = call_llm(SYSTEM_REWRITE, user prompt + summary + feedback + tweets); revision_count += 1
    # keep sentences without issues unchanged (narrative coherence). Then the graph goes back to step 3.
    ...

# ---------------- Graph ----------------
# step1 -> step2 -> step3 -> step4 -> step5 -> problem_found --"rewrite"--> step6 -> step3
#                                                           \--"end"-----> return summary
def run(topic, pid, prompt):
    """Build + run the graph. -> {"summary", "revisions", "issue_history"}"""
    ...

def run_baseline(topic, pid, prompt):
    """Baseline = step 1 + step 2 only (same load, same SYSTEM_SUMMARY, same call_llm). No ontology, no loop.
    -> same dict, revisions = 0, issue_history = []"""
    ...
