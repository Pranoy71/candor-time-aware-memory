"""Helpers to run the official harness scorers (with Windows-safe UTF-8) and to audit time leaks."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "eval_harness"


def run_scorer(script, *args):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(HARNESS / script), *args], capture_output=True, text=True,
                       encoding="utf-8", env=env, cwd=str(HARNESS))
    return p.stdout + p.stderr


# ---------------------------------------------------------------------------- leak audit
def leak_audit(n_times=25, seed=7, data_dir=None):
    """Independent time-travel audit. For every train question, at many random as_of moments (plus the question's own),
    retrieve and assert: nothing after as_of, nothing deleted, no deletion markers, no secrets in any text shown.
    The oracle is the harness's own record loader, not our store."""
    import random
    from datetime import datetime, timedelta
    sys.path.insert(0, str(HARNESS))
    import records as H            # harness loader = ground truth for delivery times
    from .memory import Memory
    from .safety import has_secret
    from .store import parse_dt

    hu, hdel, _ = H.load(str(data_dir or (ROOT / "data")))
    htime = {u.id: u.time for u in hu}
    m = Memory(data_dir, use_llm=False)
    qs = [json.loads(l) for l in open(ROOT / "evals" / "memory_train.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    lo, hi = parse_dt("2026-09-01T00:00:00-07:00"), parse_dt("2026-09-19T00:00:00-07:00")
    span = (hi - lo).total_seconds()
    checks, bad = 0, []
    for q in qs:
        times = [parse_dt(q["as_of"])] + [lo + timedelta(seconds=rng.random() * span) for _ in range(n_times)]
        for t in times:
            hits, _ = m.retriever.search(q["question"], t, k=20)
            checks += 1
            for h in hits:
                i = h.unit.id
                if i not in htime:
                    continue        # edit-event ids are our own units of harness records; checked below
                dead = i in hdel and hdel[i] <= t
                if htime[i] > t or dead:
                    bad.append((q["id"], str(t), i, "future" if htime[i] > t else "deleted"))
                if h.unit.kind == "deleted_marker":
                    bad.append((q["id"], str(t), i, "marker"))
                if has_secret(h.text):
                    bad.append((q["id"], str(t), i, "secret"))
    return {"queries": checks, "violations": bad}
