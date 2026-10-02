"""Natural-language time resolution anchored at a 'now' (America/Los_Angeles). Standard library only."""
import re
from datetime import datetime, timedelta

from .store import LOCAL
from . import textproc as tp

WD = tp.WEEKDAYS
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty-five": 45, "half an": 0.5, "half a": 0.5}


def _num(s):
    s = s.lower().strip()
    if s in NUM_WORDS:
        return NUM_WORDS[s]
    try:
        return float(s)
    except ValueError:
        return 1


def _hm(h, m, mer):
    h, m = int(h), int(m or 0)
    if mer:
        mer = mer.lower().replace(".", "")
        if mer == "pm" and h < 12:
            h += 12
        if mer == "am" and h == 12:
            h = 0
    elif 1 <= h <= 6:      # bare "at 2" in a working day means 2pm
        h += 12
    return h, m


def parse_duration(text):
    """Return (minutes, (start, end)) or (None, None)."""
    m = re.search(r"\b(?:for\s+)?(half an? hour|an? hour and a half|(\d+(?:\.\d+)?|an?|one|two|three)\s*[- ]?(hours?|hrs?|minutes?|mins?))\b", text, re.I)
    if not m:
        return None, None
    if m.group(1).lower().startswith("half"):
        return 30, m.span()
    if "and a half" in m.group(1).lower():
        return 90, m.span()
    n = _num(m.group(2))
    unit = m.group(3).lower()
    return int(n * (60 if unit.startswith("h") else 1)), m.span()


def find_when(text, now):
    """Extract a date/time reference. Returns dict(date, time, delta, rel, spans) — parts that were not found are None."""
    t = text
    out = {"date": None, "time": None, "delta": None, "rel": None, "spans": []}
    today = now.astimezone(LOCAL).date()

    def take(m):
        out["spans"].append(m.span())

    m = re.search(r"\b(\d+(?:\.\d+)?|an?|one|two|three|half an?)\s*(hours?|hrs?|minutes?|mins?)\s+(before|after|ahead of|prior to)\s+(?:the\s+|my\s+)?(.+?)(?=\s+(?:to|and|so|because|then)\b|[.,;?!]|$)", t, re.I)
    if m:
        n = _num(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)
        sign = 1 if m.group(3).lower() == "after" else -1
        out["rel"] = (timedelta(minutes=sign * n), m.group(4).strip())
        take(m)
        return out
    m = re.search(r"\bin\s+(\d+(?:\.\d+)?|an?|one|two|three|half an?)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?)\b", t, re.I)
    if m:
        n, u = _num(m.group(1)), m.group(2).lower()
        out["delta"] = timedelta(minutes=n) if u.startswith("min") else timedelta(hours=n) if u.startswith(("hour", "hr")) \
            else timedelta(days=n) if u.startswith("day") else timedelta(weeks=n)
        take(m)
    for pat, off in ((r"\bday after tomorrow\b", 2), (r"\btomorrow\b", 1), (r"\btoday\b", 0), (r"\btonight\b", 0)):
        m = re.search(pat, t, re.I)
        if m and out["date"] is None:
            out["date"] = today + timedelta(days=off)
            if "tonight" in pat and out["time"] is None:
                out["time"] = (19, 0)
            take(m)
    m = re.search(r"\b(?:(next|this|on|by)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t, re.I)
    if m and out["date"] is None:
        wd = WD.index(m.group(2).lower())
        if (m.group(1) or "").lower() == "next":      # that weekday in the next calendar week (Mon-Sun)
            out["date"] = today + timedelta(days=7 - today.weekday() + wd)
        else:
            out["date"] = today + timedelta(days=(wd - today.weekday()) % 7)
        take(m)
    if out["date"] is None:
        m = re.search(rf"\b(?:on\s+)?({tp.MON_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t, re.I) or \
            re.search(rf"\b(?:on\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({tp.MON_RE})\b", t, re.I)
        if m:
            a, b = m.group(1), m.group(2)
            mon, day = (tp._month(a), int(b)) if not a.isdigit() else (tp._month(b), int(a))
            d = datetime(today.year, mon, day).date()
            if d < today:
                d = d.replace(year=today.year + 1)
            out["date"] = d
            take(m)
    if out["date"] is None:
        m = re.search(r"\b(?:on\s+|by\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", t, re.I)
        if m:
            day = int(m.group(1))
            y, mo = today.year, today.month
            d = None
            for k in range(0, 3):
                mm = mo + k
                yy = y + (mm - 1) // 12
                mm = (mm - 1) % 12 + 1
                try:
                    cand = datetime(yy, mm, day).date()
                except ValueError:
                    continue
                if cand >= today:
                    d = cand
                    break
            out["date"] = d
            take(m)
    if out["date"] is None:
        m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)
        if m:
            try:
                out["date"] = datetime(today.year, int(m.group(1)), int(m.group(2))).date()
                take(m)
            except ValueError:
                pass
    # time of day
    m = re.search(r"\b(?:at\s+|@\s*)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", t, re.I)
    if m:
        out["time"] = _hm(m.group(1), m.group(2), m.group(3))
        take(m)
    else:
        m = re.search(r"\b(?:at|@)\s*(\d{1,2})(?::(\d{2}))?\b(?!\s*(?:th|st|nd|rd|%|\w*minutes?|hours?))", t, re.I)
        if m:
            out["time"] = _hm(m.group(1), m.group(2), None)
            take(m)
        else:
            m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
            if m:
                out["time"] = (int(m.group(1)), int(m.group(2)))
                take(m)
    for pat, hm in ((r"\bnoon\b", (12, 0)), (r"\bmidnight\b", (0, 0)), (r"\b(?:eod|end of (?:the )?day)\b", (17, 0)),
                    (r"\bthis morning\b|\bin the morning\b", (9, 0)), (r"\bthis afternoon\b|\bin the afternoon\b", (15, 0)),
                    (r"\bthis evening\b|\bin the evening\b", (18, 0))):
        m = re.search(pat, t, re.I)
        if m and out["time"] is None:
            out["time"] = hm
            take(m)
    return out


def resolve(when, now, default_time=(9, 0), base_date=None):
    """Turn find_when() output into an aware datetime (None if nothing usable)."""
    now = now.astimezone(LOCAL)
    if when.get("delta") is not None:
        return now + when["delta"]
    if when["date"] is None and when["time"] is None:
        return None
    d = when["date"] or base_date or now.date()
    h, m = when["time"] or default_time
    dt = datetime(d.year, d.month, d.day, h, m, tzinfo=LOCAL)
    if when["date"] is None and base_date is None and dt <= now:
        dt += timedelta(days=1)      # "at 9" said after 9 means tomorrow
    return dt


def strip_spans(text, spans):
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + " " + text[b:]
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def iso(dt):
    return dt.astimezone(LOCAL).isoformat(timespec="seconds")
