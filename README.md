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
- **strongest opinions** ("strongest", "most extreme", ...): fills spare slots with the most intense claims, `INTENSITY_INFLATED` still
  stops exaggeration, and `PROMPT_FOCUS_BIAS` flags a summary that leaves out a view held by >= 20% of the tweets;
- **equal weight** ("fair/equal weight ... neutral"): neutral views get a guaranteed slot like positive and negative ones;
- **one-sided focus**: words like *complaints, criticism, frustration* (negative) or *upbeat, buzz, marketing* (positive).

The focus puts that sentiment first and sets the audience/tone, but the agent still includes every well-supported
viewpoint (the largest positive and negative claim are always kept) and flags a summary that omits a view held by
20% or more of the tweets (`PROMPT_FOCUS_BIAS`). Two ready-made prompts (the ones in the report) are in `prompts.PROMPTS`: `strongest_opinions` (asks for the
"strongest opinions", an intensity bias) and `fair_equal_weight` (asks for equal weight of positive, negative AND neutral
views). `prompts.EXTRA_PROMPTS` keeps the older one-sided prompts (`negative_focus`, `positive_focus`).

## Command-line options (main.py)

`--data FILE...` input (`.txt` DivSumm files `group<TAB>text`, or CSV `id,group,text[,team]`; see `data/sample_tweets.csv`),
`--prompt`, `--model`, `--k`, `--per-cell N`, `--max-iters`, `--tolerance` (lower = stricter group balance),
`--temperature`, `--no-ontology-context` (ablation), `--offline`, `--require-reasoner`, `--out DIR`.

## Output

- `out/summary.txt`: the summary, one paragraph per team, no tweet ids.
- `out/citations.json`: each sentence with the tweet ids it is based on.
- `out/report.json`: per-iteration violations, scores and metrics.

## How the agent works (LangGraph, `agent.py`)

Graph: `ingest -> draft -> extract -> ontology_check -> evaluate -> (revise -> extract ...)* -> finish`
(`python draw_graph.py` writes `graph.mmd` / `graph.png` for the report).

1. **ingest**: the LLM labels each tweet (team, sentiment, strength, aspect such as offense or coaching); the tweets
   become individuals in the ontology; a consistency check runs (SPARQL always, Pellet if Java works). The result
   (`reasoner:` message) is stored in `out/report.json`, so you can see whether Pellet really ran.
2. **draft**: a SPARQL query yields candidate claims per (team, sentiment, aspect); thin aspects are folded into a general
   claim; at most k claims per team are kept, every viewpoint first. Each claim cites tweets from every author group
   holding that view. The LLM only words one neutral sentence per claim and never sees group labels.
3. **extract** (Information Extractor): opinions and Report-1 facts are extracted from each sentence and added to the ontology.
4. **ontology_check** (Ontology Reasoning Tool): SPARQL checks (group named, sentiment unsupported, intensity inflated, group
   under-represented, viewpoint omitted, `SINGLE_TWEET_DETAIL`) plus the six Report-1 queries (`report1_checks.py`, DL/Pellet on a scratch copy).
5. **evaluate** (Bias Consistency Evaluator): prompt-focus check (`PROMPT_FOCUS_BIAS`), weighted score, best draft so far.
   Conditional edge: finish when clean, at `--max-iters`, or after 3 rounds without improvement; otherwise revise.
6. **revise** (Rewrite): only flagged sentences are rewritten (two LLM tries, then a template sentence), then back to extract.

State: `claims, plan, iteration, summary_iri, facts, result, history, best, stale, summary`.
The best-scoring draft is returned, which is not always violation-free.

Module map: `agent.py` graph and planning, `ontology_store.py` OWL/SPARQL, `llm.py` model backends, `prompts.py`
prompt parsing, `report1_checks.py` Report-1 queries, `evaluation.py` SPARQL checks and scoring, `metrics.py` text-only
metrics, `run_experiments.py` experiment grid, `data_loader.py` data loading, `add_reference_metrics.py` ROUGE/BERTScore afterwards, `draw_graph.py` graph figure.

## Experiments and metrics

`run_experiments.py` runs 3 topics x 2 prompts x N runs for the baseline (plain LLM, no ontology), the agent, and with
`--ablation` the agent without its plan. Results go to `results/` (`results.csv`, `summaries.json`, `aggregate.csv`).
Metrics work on the final text only, so every system is scored identically: `sel_intensity_gap` (mean intensity of the source tweets each summary sentence is built from, minus the mean over all tweets; > 0 = the summary leans on louder tweets, i.e. intensity bias in *selection*; also `sel_high_share`), `high_share` / `high_gap` (share of emphatic sentences in the summary and summary-minus-source; > 0 = louder than the tweets, tests the "strongest opinions" hypothesis for every system alike), `polarity_skew` (sentiment mix vs the
tweets, lower is better), `sim_parity` and `equal_coverage_nn` (similarity to each social group, higher is better; simple proxies),
`equal_coverage` (report formula, LOWER = fairer, TF-IDF max-similarity estimate of coverage), ROUGE and `--bertscore` (need
`--refs refs`, see `refs/README.md`), optional DeepEval (`--deepeval-judge ollama:<model>`, needs `pip install deepeval`), seconds.
The baseline gets the same sentence budget as the agent (`--k`, default 5) through its system message, so length does not favour either side; `--no-length-match` switches this off. The runner also prints/saves the median per system (robust against a stalled run). The `model` is stored in every row of `results.csv`.

## Reproducibility

All labelling (tweets, summary sentences, facts) runs at temperature 0 on a separate model handle and is cached per
(text, topic) inside a process; only the *writing* uses `--temperature`. So the source polarity/intensity labels, and
therefore `polarity_skew`, `high_gap` and `sel_intensity_gap`, no longer drift between runs, and the agent no longer
re-labels the same tweets on every run (faster experiments). The three runs per cell differ only through the writing step.

Full experiment (3 topics x 2 prompts x 3 runs x baseline/agent/ablation; roughly 2 min per agent run):

    python run_experiments.py --runs 3 --temperature 0.7 --ablation --refs refs --bertscore

## Interrupted or incomplete runs

- `--resume` continues an interrupted run: finished rows in `--out` (default `results/`) are kept and only the missing
  (topic, prompt, run, system) combinations are computed. Use the same options as the original run.
- A summary with no usable sentence (the model returned nothing) no longer crashes the run: it is recorded with
  `empty_summary = 1` and its metrics left missing (NaN), so the means ignore it and you can report the failure rate.
- No reference summaries during the run? Add them to `refs/` later and run
  `python add_reference_metrics.py --results results --refs refs --bertscore`: it adds ROUGE/BERTScore from the saved
  `summaries.json` and rewrites `results.csv`, `aggregate.csv` and `median_per_system.csv` (the old csv is kept).

## Known limitations

- Only the three NFL teams and three dialect groups are modelled; other topics need new ontology classes.
- The agent gives positive and negative views balanced slots, not slots proportional to their share of the tweets.
- Wording is only partly fact-checked: `SINGLE_TWEET_DETAIL` flags claims that lean on words/numbers found in just one cited tweet (and in at most two tweets of the topic), but invented details that appear in no tweet are not caught.
- The Person/Company/Country queries find nothing to check on NFL tweets (they pass vacuously);
  `test_report1_queries.py` shows they fire on a bad claim.
- Pellet needs a recent Java; if `reasoner:` in `report.json` says "Pellet failed", only the SPARQL checks ran and Q1/Q2/Q5 never fire.
- The `ablation_no_plan` arm still runs the same ontology checks, so it measures the value of the plan, not of the ontology.
- Each agent run takes about 2 minutes with a local 7B model.
