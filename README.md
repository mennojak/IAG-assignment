# INFOIAG: ontology-driven, bias-mitigated tweet summarisation

An LLM agent that summarises DivSumm tweets (White, Hispanic and African-American dialect groups) about three NFL
teams (49ers, Patriots, Seahawks) while keeping the summary fair. It builds an OWL ontology from the tweets, plans the
summary with SPARQL queries, writes it sentence by sentence, then checks and repairs its own output with the reasoner.

## What you can do with it

| I want to... | Command |
|---|---|
| Summarise with my own prompt | `python main.py --model ollama:qwen2.5:7b-instruct --data data/Patriots.txt --prompt "Summarise the complaints about this team in 3 sentences."` |
| Summarise all three teams | `python main.py --model ollama:qwen2.5:7b-instruct` |
| Try it quickly on fewer tweets | add `--per-cell 8` |
| Test the pipeline with no LLM | `python main.py --offline` (keyword backend, output is crude) |
| Compare against a plain LLM and an ablation | `python run_experiments.py --model ollama:qwen2.5:7b-instruct --runs 3 --temperature 0.7 --ablation` |
| See each ontology query fire on a bad claim | `python test_report1_queries.py` |
| Look at the ontology | open `infoiag_ontology.owl` in Protege |

## Setup

    pip install -r requirements.txt
    ollama pull qwen2.5:7b-instruct  
    # Java on PATH is needed for the Pellet reasoner; without it only the SPARQL checks run
    # (use --require-reasoner to make main.py abort instead).

## Writing prompts

`prompts.py` reads your prompt for:
- **length**: "in 3 sentences" sets k, otherwise `--k` (default 5 sentences per team);
- **one-sided focus**: words like *complaints, criticism, frustration* (negative) or *upbeat, buzz, marketing* (positive).

The focus puts that sentiment first and sets the audience/tone, but the agent still includes every well-supported
viewpoint (the largest positive and negative claim are always kept) and flags a summary that omits a view held by
20% or more of the tweets (`PROMPT_FOCUS_BIAS`). Two ready-made biased prompts are in `prompts.PROMPTS`.

## Command-line options (main.py)

`--data FILE...` input (`.txt` DivSumm files `group<TAB>text`, or CSV `id,group,text[,team]`; see `data/sample_tweets.csv`),
`--prompt`, `--model`, `--k`, `--per-cell N`, `--max-iters`, `--tolerance` (lower = stricter group balance),
`--temperature`, `--no-ontology-context` (ablation), `--offline`, `--require-reasoner`, `--out DIR`.

## Output

- `out/summary.txt`: the summary, one paragraph per team, no tweet ids.
- `out/citations.json`: each sentence with the tweet ids it is based on.
- `out/report.json`: per-iteration violations, scores and metrics.

## How the agent works (LangGraph, `agent.py`)

1. **ingest**: the LLM labels each tweet (team, sentiment, strength, aspect such as offense or coaching); the tweets
   become individuals in the ontology; a consistency check runs (SPARQL always, Pellet if Java is available).
2. **plan + draft**: a SPARQL query yields candidate claims per (team, sentiment, aspect); thin aspects are folded
   into a general claim; at most k claims per team are kept, every viewpoint first. Each claim cites tweets from every
   author group holding that view. The LLM only words one neutral sentence per claim and never sees group labels.
3. **evaluate**: opinions are extracted from each sentence and the SPARQL checks run: group named, sentiment
   unsupported, intensity inflated, group under-represented, viewpoint omitted, prompt-focus bias, plus the six
   Report-1 queries (`report1_checks.py`).
4. **revise**: only flagged sentences are rewritten (two LLM tries, then a template sentence). It stops when clean, at
   `--max-iters`, or after 3 rounds without improvement, and returns the best draft.

Module map: `agent.py` graph and planning, `ontology_store.py` OWL/SPARQL, `llm.py` model backends, `prompts.py`
prompt parsing, `report1_checks.py` Report-1 queries, `evaluation.py` checks and scoring, `metrics.py` text-only
metrics, `run_experiments.py` experiment grid, `data_loader.py` data loading.

## Experiments and metrics

`run_experiments.py` runs 3 topics x 2 prompts x N runs for the baseline (plain LLM, no ontology), the agent, and with
`--ablation` the agent without its plan. Results go to `results/` (`results.csv`, `summaries.json`, `aggregate.csv`).
Metrics work on the final text only, so every system is scored identically: `polarity_skew` (sentiment mix vs the
tweets, lower is better), `sim_parity` and `equal_coverage_nn` (similarity to each social group, higher is better),
ROUGE (needs `--refs refs`, with `refs/<topic>.txt` holding the DivSumm references separated by blank lines), seconds.
Use the median for seconds, since a stalled run can distort the mean.

## Known limitations

- Only the three NFL teams and three dialect groups are modelled; other topics need new ontology classes.
- The agent gives positive and negative views balanced slots, not slots proportional to their share of the tweets.
- Wording is not fact-checked against the tweets; the LLM can still add details the tweets do not contain.
- The Person/Company/Country queries find nothing to check on NFL tweets (they pass vacuously);
  `test_report1_queries.py` shows they fire on a bad claim.
- Each agent run takes about 2 minutes with a local 7B model.
