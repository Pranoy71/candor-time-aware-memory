"""Model before search: turn a question into an evidence plan.

The model never sees the records. It sees the question, the current time, and the *vocabulary of the data* (people, channels,
titles of meetings and events that already exist at that time), and returns:
  needs    the separate pieces of evidence a good answer requires (original value, change, cause, the person's own words ...),
           each with 1-3 short search phrasings written the way the records are likely to phrase it
  anchors  time anchors that must be looked up first ("the flight to Denver"), so their dates can be applied to the search
  entities people / organisations the question is about
One cached call per question. Any failure returns None and search simply runs without a plan.
"""
import re

from .llm import LLMError
from .store import LOCAL

SYSTEM = """You plan the search for a question about one person's work life (meetings, Slack, email, dictation, calendar, notes).
You do not see the records. You see the question, the current time, and some names and titles that exist in the data.
Return JSON:
{
 "type": "fact|current_value|history|why|who|when|status|compare|list|unanswerable_maybe",
 "needs": [ {"label": "<the piece of evidence>", "queries": ["<2-8 word search phrasing in the words a record would use>", "..."]} ],
 "anchors": [ {"query": "<what to look up to get a date or date range, e.g. 'flight to Denver'>", "span": false} ],
 "entities": ["<people or organisations the question is about, full names if known>"]
}
Guidelines
- Split a question into the separate pieces of evidence it needs. "Why did X move?" needs the original value, the new value, and the cause.
  "Did X happen?" needs the promise and the completion (or its absence). "What did A say vs B?" needs one piece per speaker.
- Write queries the way people actually talk in meetings, chat and email, using plain synonyms ("push back" and "delay", "lean no" and "weak").
  Do not copy the question's abstract wording. Use names from the vocabulary when they fit.
- \"span\" is true when the question means the whole duration of something (\"while I'm in Denver\"), false for a single day (\"the day I fly\").
- At most 4 needs, at most 3 queries each, at most 2 anchors. If the question names no time anchor, return an empty anchors list.
- Never answer the question. Only plan the search."""


def vocab_hint(store, as_of, limit=60):
    people = sorted({u["real_name"] for u in store.user_rows if not u.get("is_bot")})
    chans = [c["name"] for c in store.channel_rows if not c.get("is_dm")]
    titles = []
    for m in store.meetings.values():
        if parse(m["start"]) <= as_of:
            titles.append(m["title"])
    for u in store.units:
        if u.source == "calendar" and u.time <= as_of and u.title not in titles:
            titles.append(u.title)
    return f"PEOPLE: {', '.join(people)}\nCHANNELS: {', '.join(chans)}\nMEETING/EVENT TITLES: {'; '.join(titles[:limit])}"


def parse(s):
    from .store import parse_dt
    return parse_dt(s)


def make_plan(llm, store, question, as_of):
    if llm is None or not getattr(llm, "available", False):
        return None
    user = (f"NOW: {as_of.astimezone(LOCAL):%A %Y-%m-%d %H:%M}\nQUESTION: {question}\n\n{vocab_hint(store, as_of)}")
    try:
        d = llm.complete_json(SYSTEM, user, max_tokens=700)
    except LLMError:
        return None
    return clean(d)


def clean(d):
    if not isinstance(d, dict):
        return None
    needs = []
    for n in (d.get("needs") or [])[:4]:
        if not isinstance(n, dict):
            continue
        qs = [re.sub(r"\s+", " ", str(q)).strip()[:80] for q in (n.get("queries") or [])[:3] if isinstance(q, (str, int))]
        qs = [q for q in qs if q]
        if qs:
            needs.append({"label": str(n.get("label", ""))[:80], "queries": qs})
    anchors = [{"query": str(a.get("query", ""))[:80], "span": bool(a.get("span"))} for a in (d.get("anchors") or [])[:2] if isinstance(a, dict) and a.get("query")]
    ents = [str(e)[:60] for e in (d.get("entities") or [])[:6] if isinstance(e, str)]
    if not needs and not anchors:
        return None
    return {"type": str(d.get("type", ""))[:30], "needs": needs, "anchors": anchors, "entities": ents}
