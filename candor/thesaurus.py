"""A small, generic work-language thesaurus for query expansion (stemmed on load).
Not tied to this dataset: these are ordinary business/office synonyms. Expansion terms are down-weighted."""
from . import textproc as tp

GROUPS = [
    "slip delay push postpone reschedule move later extend shift pushback",
    "launch release ship rollout deploy golive live",
    "date deadline due schedule when timeline target",
    "decide decision agree lock confirm approve settle final call",
    "owner own responsible assign lead drive owe",
    "done complete finish land deliver ready posted shipped sent send",
    "promise commit pledge owe follow",
    "hire hiring role headcount recruit candidate posting backfill",
    "price pricing cost fee rate quote proposal offer",
    "sign close contract agreement deal signature renew",
    "flight fly depart airline ticket travel trip airport",
    "meeting call sync standup session huddle",
    "calendar agenda event schedule",
    "prefer like favorite rather usually habit style",
    "cut drop remove descope skip scrap",
    "problem issue bug regression failure broken fix",
    "why reason because cause due",
    "budget spend cost funding raise round",
    "customer client account buyer",
    "database db postgres sqlite storage",
    "say said tell told mention mentioned",
    "p95 95th percentile p99 99th tail",
    "latency response speed slow fast performance perf",
    "median p50 average typical",
]
_map = {}
for g in GROUPS:
    words = [tp.stem(w) for w in g.split()]
    for w in words:
        _map.setdefault(w, set()).update(x for x in words if x != w)


def expand(stems):
    """Return expansion stems (excluding those already present)."""
    have = set(stems)
    out = []
    for s in stems:
        for x in sorted(_map.get(s, ())):
            if x not in have and x not in out:
                out.append(x)
    return out
