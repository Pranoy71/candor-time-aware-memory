"""Failure taxonomy. For every question with gold evidence, say *where* it went wrong:
   SEARCH_MISS     a needed record is not even in the top 20 (rank shown from a wider pool)
   RANK_MISS       found in the top 20 but not the top 10 (the scored window)
   ANSWER_MISS     the evidence was retrieved but the answer lacks the key facts
   WRONG_ABSTAIN   the memory answered "I don't have that" although evidence was retrieved or exists
   SHOULD_ABSTAIN  an unanswerable question got a confident answer
   OK              everything needed is in the top 10 and the answer has the key facts
This answers 'did retrieval miss it, or did the answer step?' for each failure, without a judge."""
import json
import re
from collections import Counter

from .store import parse_dt


def _has_terms(answer, groups):
    a = (answer or "").lower()
    return all(any(t.lower() in a for t in grp) for grp in groups)


def diagnose(mem, gold_rows, answers=None, wide=100):
    """answers: {id: row} from a previous `memory` run (so the LLM path is diagnosed without new calls)."""
    out = []
    for g in gold_rows:
        as_of = parse_dt(g["as_of"])
        hits, _ = mem.retriever.search(g["question"], as_of, k=wide)
        ranked = [h.unit.id for h in hits]
        a = (answers or {}).get(g["id"])
        if a:
            ranked20 = a["retrieved"][:20]
            ranks = {i: n + 1 for n, i in enumerate(ranked20)}
        else:
            ranks = {i: n + 1 for n, i in enumerate(ranked)}
        row = {"id": g["id"], "category": g.get("category", ""), "question": g["question"], "groups": []}
        needed = g.get("needed") or []
        worst = 0
        for grp in needed:
            best = min([ranks[i] for i in grp if i in ranks] or [10 ** 6])
            wide_rank = min([n + 1 for n, i in enumerate(ranked) if i in set(grp)] or [10 ** 6])
            row["groups"].append({"ids": grp, "rank": best if best < 10 ** 6 else None,
                                  "wide_rank": wide_rank if wide_rank < 10 ** 6 else None})
            worst = max(worst, best)
        retrieval_ok = worst <= 10 if needed else True
        if not g.get("answerable", True):
            cls = "OK" if (a is None or a["abstained"]) else "SHOULD_ABSTAIN"
        elif not retrieval_ok:
            cls = "SEARCH_MISS" if worst > 20 else "RANK_MISS"
        elif a:
            if a["abstained"]:
                cls = "WRONG_ABSTAIN"
            elif g.get("key_terms") and not _has_terms(a["answer"], g["key_terms"]):
                cls = "ANSWER_MISS"
            else:
                cls = "OK"
        else:
            cls = "OK"
        row["class"] = cls
        out.append(row)
    return out


def render(rows):
    c = Counter(r["class"] for r in rows)
    lines = ["%-8s %-16s %-14s %s" % ("id", "category", "class", "needed groups: best rank in top-20 (wide rank)")]
    for r in rows:
        gs = " ".join(f"{g['rank'] or '-'}({g['wide_rank'] or '-'})" for g in r["groups"]) or "-"
        lines.append("%-8s %-16s %-14s %s" % (r["id"], r["category"][:16], r["class"], gs))
    lines.append("")
    lines.append("  ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    return "\n".join(lines)
