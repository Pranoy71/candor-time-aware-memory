"""Tokenising, light stemming, and date-phrase parsing. Standard library only."""
import re
from datetime import date, datetime, timedelta

STOP = set("""a an the and or but if then than that this these those of to in on at by for with from as is are was were be been being
am do does did doing have has had having i me my mine we our ours you your yours he him his she her hers it its they them their
what which who whom whose when where why how there here about into over under again further once so too very can could should would
will shall may might must just also not no nor only own same such up down out off any each few more most other some s t don now
tell said say says did get got go going gone let us please many much long""".split())
# words that matter for meaning even though they are short / common
KEEP = {"not", "no", "still", "now"}

_tok = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def stem(w):
    w = _stem(w)
    if len(w) > 4 and w.endswith("e"):  # price/pricing, share/sharing, decide/deciding all meet at one stem
        w = w[:-1]
    return w


def _stem(w):
    if len(w) <= 3 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("ied", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""),
                     ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: len(w) - len(suf)] + rep
            if suf in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            if suf in ("es",) and not base.endswith(("ch", "sh", "x", "z", "ss")):
                base = w[:-1]  # "shares" -> "share"
            return base
    return w


def tokens(text, stop=True, do_stem=True):
    out = []
    for m in _tok.finditer((text or "").lower().replace("’", "'")):
        w = m.group(0)
        if "'" in w:
            w = w.split("'")[0]
        if not w:
            continue
        if stop and w in STOP and w not in KEEP:
            continue
        out.append(stem(w) if do_stem else w)
    return out


# ---------------------------------------------------------------- dates in questions
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MON_RE = "|".join([m for m in MONTHS] + [m[:3] for m in MONTHS if m != "may"] + ["sept"])
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _month(name):
    name = name.lower()
    for i, m in enumerate(MONTHS):
        if m == name or m[:3] == name[:3]:
            return i + 1
    return None


def dates_in_text(text, ref, explicit_only=False):
    """Calendar dates the text mentions, resolved against `ref` (a datetime). Returns a set of date objects."""
    t = (text or "").lower()
    out = set()
    year = ref.year
    for m in re.finditer(rf"\b({MON_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t):
        mo = _month(m.group(1))
        try:
            out.add(date(year, mo, int(m.group(2))))
        except (ValueError, TypeError):
            pass
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MON_RE})\b", t):
        mo = _month(m.group(2))
        try:
            out.add(date(year, mo, int(m.group(1))))
        except (ValueError, TypeError):
            pass
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        try:
            out.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", t):
        try:
            out.add(date(year, int(m.group(1)), int(m.group(2))))
        except ValueError:
            pass
    if explicit_only:
        return out
    today = ref.date()
    if re.search(r"\byesterday\b", t):
        out.add(today - timedelta(days=1))
    if re.search(r"\b(today|this morning|this afternoon|tonight)\b", t):
        out.add(today)
    if re.search(r"\btomorrow\b", t):
        out.add(today + timedelta(days=1))
    for m in re.finditer(r"\b(last|this|next|on)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t):
        wd = WEEKDAYS.index(m.group(2))
        mod = m.group(1)
        delta = (today.weekday() - wd) % 7
        if mod == "last":
            d = today - timedelta(days=delta or 7)
        elif mod == "next":
            d = today + timedelta(days=(wd - today.weekday()) % 7 or 7)
        else:  # bare weekday: the most recent one (past) — questions about the past dominate
            d = today - timedelta(days=delta)
        out.add(d)
    return out


CURRENT_CUES = re.compile(r"\b(now|current|currently|still|latest|today|these days|at the moment|right now|"
                          r"is it|are we|do i still|does .* still|anymore|up to date)\b", re.I)
HISTORY_CUES = re.compile(r"\b(originally|first|initially|at first|history|changed|change|moved|slip|slipped|why|"
                          r"how many days|how long|before|earlier|previously|was|were|used to)\b", re.I)


def wants_current(question):
    return bool(CURRENT_CUES.search(question or ""))


STATUS_START = re.compile(r"^\s*(has|have|had|is|are|was|were|did|does|do)\b", re.I)
STATUS_CUES = re.compile(r"\b(yet|already|still|anymore|signed|sent|done|finished|completed|landed|status|so far)\b", re.I)

# words in a question that point at a particular kind of source
SOURCE_CUES = {
    "dictation": re.compile(r"\b(dictat\w*|note to self|voice note|i (?:said|noted) to myself)\b", re.I),
    "email": re.compile(r"\b(e-?mail\w*|inbox|wrote to|replied|reply|subject)\b", re.I),
    "slack": re.compile(r"\b(slack|dm|dms|channel|posted|thread)\b", re.I),
    "meeting": re.compile(r"\b(meeting|call|standup|1:1|one-on-one|in the room|on the call|go/no-go|recording|discussed)\b", re.I),
    "calendar": re.compile(r"\b(calendar|invite|invitation|event|schedule[sd]?|agenda|booked)\b", re.I),
    "codex": re.compile(r"\b(codex|repo|prototype|code|coding|commit|branch)\b", re.I),
    "chatgpt": re.compile(r"\b(chatgpt|gpt|chat gpt|asked ai|the assistant)\b", re.I),
}


def source_cues(question):
    return {s for s, rx in SOURCE_CUES.items() if rx.search(question or "")}


def wants_status(question):
    return bool(STATUS_START.search(question or "") or STATUS_CUES.search(question or "") or CURRENT_CUES.search(question or ""))


def date_phrase_tokens(text):
    """Stems of tokens that belong to a date phrase ('September 30', 'Sep 10'), so they can be down-weighted."""
    t = (text or "").lower()
    out = set()
    for m in re.finditer(rf"\b(({MON_RE})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?({MON_RE}))\b", t):
        out.update(tokens(m.group(0)))
    return out
