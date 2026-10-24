"""Load tweets from the DivSumm-style .txt files (group<TAB>text, one file per team) or a CSV."""
from __future__ import annotations

import html
import random
import re
from pathlib import Path

import pandas as pd

GROUP_MAP = {"aa": "AfricanAmerican", "africanamerican": "AfricanAmerican",
             "hisp": "Hispanic", "hispanic": "Hispanic", "white": "White"}
TEAM_FILE = {"49ers": "SanFrancisco49ers", "patriots": "NewEnglandPatriots", "seahawks": "SeattleSeahawks"}


def _clean(t: str) -> str:
    t = html.unescape(t.strip())
    t = t.replace("\\n", " ").replace("\ufffd", "")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1].strip()
    return t


def _team_from_name(name: str) -> str | None:
    low = name.lower()
    return next((v for k, v in TEAM_FILE.items() if k in low), None)


def load_tweets(paths: list[str]) -> list[dict]:
    rows, seen = [], set()
    for p in map(Path, paths):
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            recs = [(str(r["group"]), str(r["text"]), r.get("team") if "team" in df else None)
                    for _, r in df.iterrows()]
        else:
            team = _team_from_name(p.stem)
            recs = []
            for line in p.read_text(encoding="cp1252", errors="replace").splitlines():
                if "\t" in line:
                    g, txt = line.split("\t", 1)
                    recs.append((g, txt, team))
        for g, txt, team in recs:
            group = GROUP_MAP.get(g.strip().lower())
            txt = _clean(txt)
            if group is None or not txt:
                continue
            team = team if isinstance(team, str) and team else None
            key = (group, team, txt)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"group": group, "text": txt, "team": team})
    for i, r in enumerate(rows, start=1):
        r["id"] = i
    return rows


def subsample(tweets: list[dict], per_cell: int, seed: int = 0) -> list[dict]:
    """Keep at most `per_cell` tweets per (group, team); ids are re-numbered."""
    rnd = random.Random(seed)
    cells: dict[tuple, list] = {}
    for t in tweets:
        cells.setdefault((t["group"], t["team"]), []).append(t)
    keep = []
    for v in cells.values():
        rnd.shuffle(v)
        keep += v[:per_cell]
    keep.sort(key=lambda t: t["id"])
    for i, t in enumerate(keep, start=1):
        t["id"] = i
    return keep
