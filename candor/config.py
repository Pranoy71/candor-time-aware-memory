"""Paths and environment. Standard library only (no python-dotenv)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CANDOR_DATA", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CANDOR_CACHE", ROOT / ".cache"))
TZ_NAME = "America/Los_Angeles"


def load_env(path=None):
    """Load KEY=VALUE lines from .env into os.environ (without overriding real env vars)."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

# Default for the model-written query plan (qplan.py). OFF: in an offline replay of the real cached plans it lowered retrieval on
# train (100% -> 88%) and did not help on the dev set (see README). Override per run with CANDOR_PLAN=1.
PLAN_DEFAULT = False
