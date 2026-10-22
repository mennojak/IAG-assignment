"""ORCHESTRATOR.   Run:  python main.py
REPORT: "Dataset and Experimental Setup" + "Baseline and Experimental Conditions".
2 arms (baseline, agent) x 3 topics x 2 prompts x 3 runs = 36 runs, then prints the result tables.
Both arms get the same LLM, settings, seed, prompt: the ONLY difference is the ontology loop.
"""
import config
import data, agent, metrics
from llm import USER_PROMPTS

def main():
    # data.annotate_all() if the annotation cache does not exist yet (one-off)
    # for topic in TOPICS:
    #     tweets = data.load_tweets(topic); refs = data.load_references(topic)     # only needed for metrics
    #     for pid in PROMPT_IDS:
    #         prompt = USER_PROMPTS[pid].format(team=TEAM_NAMES[topic])
    #         for run in range(3):
    #             for arm in ARMS:
    #                 set the LLM seed to SEEDS[run]
    #                 t0 = now
    #                 out = agent.run_baseline(topic, pid, prompt) or agent.run(topic, pid, prompt)
    #                 time_s = now - t0            # time ONLY the arm, not the metrics
    #                 scores = metrics.compute_all(out["summary"], tweets, refs)
    #                 append {arm, topic, pid, run, time_s, scores, **out} as one JSON line to RESULTS_FILE
    # metrics.analyze(RESULTS_FILE)
    ...

if __name__ == "__main__":
    main()
