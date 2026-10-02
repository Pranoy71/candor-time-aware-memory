"""Runs the official score_memory.py with a polite judge: spaced calls, retry on 429/5xx, and a disk cache so a
crash or quota stop resumes where it left off. The harness files are not modified; we wrap judge._post at runtime."""
import hashlib
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "eval_harness"
sys.path.insert(0, str(HARNESS))
import judge  # noqa: E402

CACHE = ROOT / ".cache" / "judge_cache.jsonl"
INTERVAL = float(os.environ.get("CANDOR_JUDGE_INTERVAL", "13"))
_orig, _last, _mem = judge._post, [0.0], {}
if CACHE.exists():
    for line in open(CACHE, encoding="utf-8"):
        try:
            d = json.loads(line)
            _mem[d["k"]] = d["v"]
        except Exception:
            pass


def polite_post(url, headers, body, retries=3):
    k = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if k in _mem:
        return _mem[k]
    for attempt in range(10):
        wait = INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            out = _orig(url, headers, body, retries=1)
            _mem[k] = out
            CACHE.parent.mkdir(exist_ok=True)
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": out}) + "\n")
            return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                pause = min(120, 20 * (attempt + 1))
                print(f"  [judge] HTTP {e.code}, waiting {pause}s (attempt {attempt + 1}/10)", file=sys.stderr, flush=True)
                time.sleep(pause)
                continue
            raise
    raise RuntimeError("judge: gave up after repeated rate limits; re-run to resume from cache")


judge._post = polite_post

if __name__ == "__main__":
    import score_memory
    score_memory.main()
