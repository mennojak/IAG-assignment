# Changes in v3.1 (bug fix)
- FIX: `run_experiments.py` crashed with `KeyError: 'equal_coverage'` when a system returned an empty summary. Empty summaries are now
  recorded (`empty_summary` column, metrics NaN, no fake 0.0/1.0 values) instead of crashing.
- NEW: `--resume` continues an interrupted run (also from rows written by v3).
- NEW: `add_reference_metrics.py` adds ROUGE/BERTScore to an existing results folder, so the experiment does not have to wait for `refs/`.
- Tested offline: forced empty baseline summary, resume incl. old-style rows without `empty_summary`, ROUGE backfill.

# Changes in v3
- Deterministic labelling: tweets, summary sentences and facts are labelled by a separate temperature-0 model handle
  (`LangChainBackend.label_llm`); writing still uses `--temperature`. Labels are cached per (text, topic) in a process.
- New metric `sel_intensity_gap` (+ `sel_intensity`, `src_intensity`, `sel_high_share`) in `metrics.intensity_selection`: intensity
  bias in WHICH tweets a summary is built from, identical for every system. Unit-tested on loud/calm/mixed summaries.
- Not run with a real model here (no Ollama in my sandbox); stub-tested that labelling uses temperature 0, writing keeps `--temperature`,
  and repeat labelling makes no LLM call.

# Changes in v2 (on top of the first fixed version)
- Baseline is length-matched: it gets the agent's sentence budget (`--k`) in its system message; `--no-length-match` disables it.
- Aggregate table now shows `sim_parity` (only the per-group `sim_<group>` columns stay hidden).
- New metrics `high_share`, `source_high_share`, `high_gap` for ALL systems (intensity-bias diagnostic); one labelling pass per summary
  also feeds `polarity_skew`.
- The agent's internal `equal_coverage` is now `group_balance` (higher = more balanced). `equal_coverage` in the results is the report formula (lower = fairer).
- New check `SINGLE_TWEET_DETAIL` (evaluation.py): flags a claim built on words/numbers found in exactly one cited tweet and in at most two
  tweets of the topic. The writer prompt no longer says "name concrete details"; it now allows details only if >= 2 tweets mention them.
- Verified offline + on the real Patriots tweets (flags "42-14", passes generic/shared-detail claims). NOT yet run with a real model.

# Changes in this version
- Prompts now match the report: `strongest_opinions` and `fair_equal_weight` (`prompts.py`); old ones kept in `EXTRA_PROMPTS`.
- Agent understands "strongest opinions" (intensity focus) and "equal weight incl. neutral" (neutral gets a guaranteed slot).
- Graph split into extract / ontology_check / evaluate / revise nodes (= Information Extractor, Ontology Reasoning Tool,
  Bias Consistency Evaluator, Rewrite); `draw_graph.py` produces the figure.
- `report.json` records whether Pellet really ran (`reasoner`) and the share of high-intensity opinions (source vs summary).
- Metrics: Equal Coverage in the report's formula, BERTScore (optional), DeepEval (optional, untested), median runtime.
- Removed dead `config.py`, `data.py`, duplicate `data/infoiag_ontology.owl`. Default model now matches the README (ollama qwen2.5).
- `results_old_prompts/` = the earlier experiment (negative/positive-focus prompts); rerun for the report prompts.
- Not done: DivSumm reference summaries are not in the zip (`refs/README.md`); DeepEval not run; Pellet could not run in my test environment (Java version).
