"""Commitments ledger and topic timeline — both derived from the same time-gated evidence, so they inherit the
guarantees for free: `--as-of` shows exactly what you could have known then.

commitments: finds promises ("I'll send you the proposal by Friday"), follows each through later evidence
             (moved / done / cancelled) and reports open, done, moved, cancelled or overdue.
timeline:    how a fact changed ("launch date": Sep 30 -> Oct 14 -> Oct 21), with who said it and where.
Deterministic by default; one optional LLM call tidies the wording but can never add a record that retrieval did not release.
"""
import re
from collections import defaultdict
from datetime import timedelta

from . import textproc as tp
from .llm import LLMError
from .store import LOCAL, parse_dt

PROMISE = re.compile(r"\b(i(?:'ll| will|'m going to| am going to| can have| can get| can send| promised| owe)|let me|"
                     r"i'll get|we'll (?:send|have|get|post|share)|(?:will|can) (?:send|get|have|post|share|set up))\b", re.I)
DELIVERABLE = re.compile(r"\b(proposal|deck|doc|plan|mockups?|fix|report|environment|contract|numbers|follow up|follow-up|update|"
                         r"review|agenda|ticket|summary|notes|draft|recap|dashboard|schedule|invite|access|demo|estimate|quote|"
                         r"pre-?read|test plan|rollback|posting|offer)\b", re.I)
DUE = re.compile(r"\bby\s+(?:end of day|eod|tomorrow|next week|monday|tuesday|wednesday|thursday|friday|(?:the\s+)?\d{1,2}(?:st|nd|rd|th)|"
                 r"(?:sep|oct)\w*\s+\d{1,2})|\b(?:until|before)\s+(?:monday|tuesday|wednesday|thursday|friday)|\btomorrow\b|"
                 r"\bthis (?:afternoon|week)\b|\bnext (?:week|monday|tuesday|wednesday|thursday|friday)\b", re.I)
DONE = re.compile(r"\b(sent|went out|posted|delivered|shipped|landed|done|finished|completed|attached|as promised|here is|here's|"
                  r"just emailed|uploaded|published|merged|set up|is ready|are ready|in staging)\b", re.I)
CANCELLED = re.compile(r"\b(no longer|not needed|don't need|cancel(?:led|ed)?|scratch that|off the table|pushed (?:the )?\w+ to|"
                       r"postponed|dropped|never mind|not doing)\b", re.I)
MOVED = re.compile(r"\b(push(?:ed)? (?:it |that )?(?:to|back)|move(?:d)? (?:it |that )?to|until|extension|can i have until|"
                   r"could i have until|delay|slip|later than planned)\b", re.I)
SKIP_TOPIC = set(tp.STOP) | {"send", "get", "have", "let", "make", "take", "look", "give", "put", "sure", "okay", "yeah", "thing", "thanks", "thank"}


def _topic(text):
    return {t for t in tp.tokens(text) if t not in SKIP_TOPIC and len(t) > 3 and not t.isdigit()}


def _due(text, ref):
    """Forward-looking due date: 'by Friday' means the coming Friday, 'tomorrow' the next day."""
    ref = ref.astimezone(LOCAL)
    today, t = ref.date(), text.lower()
    ds = sorted(d for d in tp.dates_in_text(text, ref, explicit_only=True) if d >= today)
    if ds:
        return ds[0]
    if re.search(r"\b(today|eod|end of (?:the )?day|this afternoon|tonight)\b", t):
        return today
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    m = re.search(r"\b(?:by|until|before|on|next|this)?\s*(monday|tuesday|wednesday|thursday|friday)\b", t)
    if m:
        wd = tp.WEEKDAYS.index(m.group(1))
        return today + timedelta(days=(wd - today.weekday()) % 7 or 7)
    if re.search(r"\bnext week\b", t):
        return today + timedelta(days=7 - today.weekday())
    return None


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if len(x.split()) >= 4]


def _who(u):
    return re.sub(r"\s*<.*", "", u.meta["from"]) if u.source == "email" else u.speaker


def _text(st, u, as_of):
    if u.source == "dictation":
        return u.meta.get("cleaned") or u.text
    return st.current_text(u, as_of)[0]


def commitments(mem, as_of, llm=None):
    as_of = parse_dt(as_of) if isinstance(as_of, str) else as_of
    st = mem.store
    bulk = ("promotions", "updates", "notifications", "social")
    vis = [u for u in st.visible(as_of) if u.source in ("meeting", "slack", "email", "dictation") and u.kind != "edit"
           and not (u.source == "email" and any(l in bulk for l in u.meta.get("labels", [])))
           and not (u.source == "dictation" and u.meta.get("delivery_state") == "discarded")
           and not (u.source == "meeting" and not u.speaker_known)]
    cands = []
    for u in vis:
        for sent in _sentences(_text(st, u, as_of)):
            if sent.endswith("?") or sent.startswith(">"):
                continue
            if PROMISE.search(sent) and DELIVERABLE.search(sent):
                cands.append({"owner": _who(u), "text": sent[:220], "unit": u, "topic": _topic(sent), "made": u.time, "due": _due(sent, u.time)})
    merged = []
    for c in sorted(cands, key=lambda c: c["made"]):
        for m in merged:
            inter = m["topic"] & c["topic"]
            if m["owner"] == c["owner"] and len(inter) >= 2 and len(inter) / max(1, len(m["topic"] | c["topic"])) >= 0.3 \
                    and (c["made"] - m["made"]) < timedelta(days=3):
                m["topic"] |= c["topic"]
                m["due"] = m["due"] or c["due"]
                m["evidence"].append(c["unit"].id)
                break
        else:
            c["evidence"] = [c["unit"].id]
            merged.append(c)
    today = as_of.astimezone(LOCAL).date()
    for m in merged:
        m["status"], m["note"] = "open", ""
        for u in vis:                                     # chronological: later evidence overrides earlier
            if u.time <= m["made"] or u.id in m["evidence"]:
                continue
            sents = _sentences(_text(st, u, as_of))
            for k, sent in enumerate(sents):
                ov = _topic((sents[k - 1] if k else "") + " " + sent) & m["topic"]
                if len(ov) < 2 or len(ov) / max(1, len(m["topic"])) < 0.34:
                    continue
                stamp = f"{u.time.astimezone(LOCAL):%b %d}"
                if CANCELLED.search(sent):
                    m["status"], m["note"] = "cancelled", f"{stamp}: {sent[:110]}"
                elif MOVED.search(sent) and _due(sent, u.time) and _due(sent, u.time) != m["due"]:
                    m["due"], m["status"], m["note"] = _due(sent, u.time), "moved", f"moved to {_due(sent, u.time):%b %d} ({stamp})"
                elif MOVED.search(sent) or re.search(r"\bi'd\b|\bi told\b|\bi had\b", sent, re.I):
                    continue                       # talk about a delay or a past promise is not completion
                elif DONE.search(sent) and (_who(u) == m["owner"] or u.source in ("slack", "email")) and not PROMISE.search(sent):
                    m["status"], m["note"] = "done", f"{stamp}: {sent[:110]}"
                else:
                    continue
                m["evidence"].append(u.id)
                break
        if m["status"] in ("open", "moved") and m["due"] and m["due"] < today:
            m["status"] = "overdue"
    merged = [m for m in merged if m["due"] or m["status"] != "open"]
    return [{"owner": m["owner"], "what": m["text"], "made": m["made"].astimezone(LOCAL).strftime("%Y-%m-%d"),
             "due": str(m["due"]) if m["due"] else "", "status": m["status"], "note": m["note"],
             "evidence": list(dict.fromkeys(m["evidence"]))[:5]} for m in merged]


def render_commitments(items):
    if not items:
        return "No commitments found."
    order = {"overdue": 0, "open": 1, "moved": 2, "done": 3, "cancelled": 4}
    lines = []
    for it in sorted(items, key=lambda i: (order.get(i["status"], 9), i["due"] or "9")):
        lines.append(f"[{it['status'].upper():9}] {it['owner']}: {it['what'][:120]}\n             made {it['made']}"
                     + (f", due {it['due']}" if it["due"] else "") + (f" — {it['note']}" if it["note"] else "")
                     + f"   ({', '.join(it['evidence'][:3])})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- timeline
VALUE = re.compile(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}|"
                   r"\$\d[\d,.]*[kKmM]?|\b\d+(?:\.\d+)?\s?(?:%|ms|seconds?|days?|weeks?|vehicles?)|\bq[1-4]\b|\bv\d+(?:\.\d+)?\b)", re.I)


def _canon(v):
    v = re.sub(r"\s+", " ", v.lower()).strip(". ")
    m = re.match(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2})", v)
    return f"{m.group(1)} {int(m.group(2))}" if m else v


def timeline(mem, topic, as_of, llm=None):
    as_of = parse_dt(as_of) if isinstance(as_of, str) else as_of
    hits, _ = mem.retriever.search(topic, as_of, k=20)
    tt = set(tp.tokens(topic))
    rows = []
    for h in sorted(hits, key=lambda h: h.unit.time):
        best = None
        for sent in re.split(r"(?<=[.!?])\s+|\n+", h.text):
            ov = len(tt & set(tp.tokens(sent)))
            if ov and VALUE.search(sent) and (best is None or ov > best[0]):
                best = (ov, sent.strip())
        if best:
            vals = [_canon(v) for v in VALUE.findall(best[1])]
            rows.append({"when": h.unit.time.astimezone(LOCAL).strftime("%Y-%m-%d %H:%M"), "id": h.unit.id, "who": h.unit.speaker or h.unit.source,
                         "where": h.unit.where, "text": best[1][:160], "values": vals})
    # value trail: the order in which each distinct value first appears
    trail, seen = [], set()
    for r in rows:
        for v in r["values"]:
            if v not in seen:
                seen.add(v)
                trail.append((v, r["when"][:10], r["id"]))
    return {"topic": topic, "as_of": str(as_of), "rows": rows, "trail": trail}


def render_timeline(t):
    lines = [f"Timeline: {t['topic']}   (as of {t['as_of'][:16]})", ""]
    for r in t["rows"]:
        lines.append(f"{r['when']}  {r['who'][:22]:22} {r['text'][:100]}   [{r['id']}]")
    if t["trail"]:
        lines += ["", "Values seen, in order: " + " -> ".join(f"{v} ({d})" for v, d, _ in t["trail"][:12])]
    if not t["rows"]:
        lines.append("Nothing in memory about that yet.")
    return "\n".join(lines)
