# INFOIAG agent: ontology-driven bias-mitigated summarisation

    pip install -r requirements.txt
    ollama pull qwen2.5:7b-instruct          # or: export ANTHROPIC_API_KEY=...  / --model openai:gpt-4o
    python main.py --model ollama:qwen2.5:7b-instruct --per-cell 8 --prompt "Summarize the opinions about the NFL teams"
    python main.py --offline                 # keyword backend, pipeline test only
    python main.py --no-ontology-context     # ablation: free-form draft, no ontology plan
    python main.py --require-reasoner        # abort if Pellet/Java cannot run

Experiments (3 topics x 2 prompts x 3 runs, baseline vs agent, all metrics -> results/):

    python run_experiments.py --model ollama:qwen2.5:7b-instruct --runs 3 --temperature 0.7 --refs refs --ablation
    # refs/<topic>.txt (49ers.txt, Patriots.txt, Seahawks.txt): DivSumm reference summaries, blank-line separated;
    # without --refs ROUGE is skipped. Metrics (metrics.py) work on the final text only, so agent and baseline are
    # scored identically: sim_parity, equal_coverage_nn, polarity_skew, rouge1/2/L, seconds.

User prompts (prompts.py): the request is parsed (number of sentences, one-sided focus such as "complaints" or
"upbeat/marketing") and used in three places: the claim budget k, the claim ORDER (focus first), and the writer's
instructions. A one-sided request can never silence the other side: select_k keeps the largest positive AND negative
claim, and the PROMPT_FOCUS_BIAS check flags a summary that omits a viewpoint held by >= 20% of the tweets.

Pipeline (LangGraph, agent.py):
1. ingest: LLM extracts one opinion per tweet -> Tweet/Person/Opinion individuals; consistency check
   (SPARQL checks always, Pellet when Java is on PATH).
2. plan + draft: a SPARQL query over the ontology yields candidate claims per (team, sentiment, aspect); aspects with too few tweets
   are folded into a general claim, and select_k keeps at most k per team (every viewpoint first),
   citing tweets from every author group that holds that viewpoint (balance by construction).
   The LLM only writes one neutral sentence per planned claim; it never sees author groups or tweet ids.
3. evaluate: opinions are extracted from each sentence, added to the ontology, and the SPARQL checks run
   (group named, polarity unsupported, intensity inflated, coverage, omitted viewpoints).
4. revise: ONLY flagged claims are rewritten (2 LLM tries, then a deterministic template sentence).
   Stops when clean, at --max-iters, or after 3 rounds without improvement. Lower score = better;
   the best draft is returned.

Report-1 queries (report1_checks.py): the six queries of Intermediate Report 1 run in evaluate.
Q4 (topic x group counts) runs verbatim on the source tweets and, in a summary variant, on the tweets each
summary cites. Facts for Q1/Q2/Q3/Q5/Q6 (marriage, age, eats, player->team, event order) are extracted from the
summary sentences and checked on a scratch copy of the ontology (Pellet / SPARQL / ASK), never on the main world.
Without such facts they pass vacuously; `python test_report1_queries.py` shows each query firing on a bad claim.

Output: out/summary.txt (one paragraph per team, no ids), out/citations.json (claim -> tweet ids),
out/report.json (per-iteration violations/metrics).
