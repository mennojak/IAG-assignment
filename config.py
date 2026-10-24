"""All settings in one place.   REPORT: "Implementation" + "Dataset and Experimental Setup"."""

from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MODEL = "llama3.1:8b"      # run through Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.7          # not 0, otherwise the 3 runs per condition are identical
NUM_CTX = 8192             # Ollama's default context is too small for 90 tweets
SEEDS = [11, 22, 33]       # run i uses SEEDS[i] for BOTH arms, so the comparison is paired
CURRENT_SEED = SEEDS[0]           # for debugging, run with a single seed

TOPICS = ["49ers", "patriots", "seahawks"]
TEAM_NAMES = {"49ers": "San Francisco 49ers", "patriots": "New England Patriots", "seahawks": "Seattle Seahawks"}
PROMPT_IDS = [1, 2]
ARMS = ["baseline", "agent"]
GROUPS = ["White", "Hispanic", "AfricanAmerican"]

MAX_REVISIONS = 3          # re-check loop stops after this many rewrites
MIN_RATIO = 0.5            # fixed rule: under-represented if below 0.5 x the best-covered group/polarity
RESULTS_FILE = "results/runs.jsonl"

# Quick smoke test without any extra code: temporarily set TOPICS = ["49ers"] and SEEDS = [11].
