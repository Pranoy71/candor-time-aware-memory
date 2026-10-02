"""Unit store with a bi-temporal visibility gate.

Every citable thing (meeting segment, Slack message/edit, email, dictation, calendar event, Codex session,
ChatGPT message) is a Unit with an `available_at` time. Edits and deletions are events. `Store.visible(as_of)`
is the ONLY way the rest of the system sees data: it drops everything that did not exist yet, drops deleted
records, and swaps in edited text. Secrets are redacted and planted instructions neutralised at load time,
so they can never be indexed, retrieved, cited or repeated.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from datetime import tzinfo

try:
    from zoneinfo import ZoneInfo
except ImportError:          # Python < 3.9
    ZoneInfo = None

from . import safety
from .config import DATA_DIR, TZ_NAME



class _USPacific(tzinfo):
    """Dependency-free America/Los_Angeles (US rules since 2007) for machines without a tz database (plain Windows)."""

    def _dst(self, dt):
        y = dt.year
        d = datetime(y, 3, 8)
        start = d + timedelta(days=(6 - d.weekday()) % 7, hours=2)        # 2nd Sunday of March, 02:00 standard time
        e = datetime(y, 11, 1)
        end = e + timedelta(days=(6 - e.weekday()) % 7, hours=1)          # 1st Sunday of November, 01:00 standard time
        naive = dt.replace(tzinfo=None)
        return start <= naive < end

    def utcoffset(self, dt):
        return timedelta(hours=-7) if self._dst(dt) else timedelta(hours=-8)

    def dst(self, dt):
        return timedelta(hours=1) if self._dst(dt) else timedelta(0)

    def tzname(self, dt):
        return "PDT" if self._dst(dt) else "PST"

    def fromutc(self, dt):
        naive = dt.replace(tzinfo=None)
        for off in (-7, -8):
            cand = (naive + timedelta(hours=off)).replace(tzinfo=self)
            if cand.utcoffset() == timedelta(hours=off):
                return cand
        return (naive + timedelta(hours=-8)).replace(tzinfo=self)


def _make_local():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(TZ_NAME)
        except Exception:
            pass
    return _USPacific()


LOCAL = _make_local()
_WD = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]


def expand_recurrence(first_day, rules, horizon_days=200):
    """Dates on which a (simple) recurring event occurs. Supports FREQ=DAILY/WEEKLY, BYDAY, INTERVAL, COUNT,
    UNTIL and EXDATE. Returns a sorted tuple of dates (always includes first_day unless excluded)."""
    rrule = next((r for r in rules if r.startswith("RRULE:")), None)
    if not rrule:
        return (first_day,)
    kv = dict(p.split("=", 1) for p in rrule[6:].split(";") if "=" in p)
    freq, interval = kv.get("FREQ", "WEEKLY"), int(kv.get("INTERVAL", 1))
    byday = [d[-2:] for d in kv.get("BYDAY", "").split(",") if d] or [_WD[first_day.weekday()]]
    count = int(kv["COUNT"]) if "COUNT" in kv else None
    until = None
    if "UNTIL" in kv:
        u = kv["UNTIL"][:8]
        until = first_day.replace(year=int(u[:4]), month=int(u[4:6]), day=int(u[6:8]))
    ex = set()
    for r in rules:
        if r.startswith("EXDATE"):
            for tok in r.split(":")[-1].split(","):
                tok = tok.strip()[:8]
                if len(tok) == 8 and tok.isdigit():
                    ex.add(first_day.replace(year=int(tok[:4]), month=int(tok[4:6]), day=int(tok[6:8])))
    out, n = [], 0
    week0 = first_day - timedelta(days=first_day.weekday())
    for i in range(horizon_days):
        d = first_day + timedelta(days=i)
        if until and d > until:
            break
        if freq == "DAILY":
            ok = i % interval == 0
        elif freq == "WEEKLY":
            ok = _WD[d.weekday()] in byday and (((d - timedelta(days=d.weekday())) - week0).days // 7) % interval == 0
        else:
            ok = i == 0
        if ok:
            n += 1
            if count and n > count:
                break
            if d not in ex:
                out.append(d)
    return tuple(out)


def parse_dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=LOCAL)


@dataclass
class Unit:
    id: str
    record: str                     # containing record (meeting id, chatgpt conversation id, else same as id)
    source: str                     # meeting | slack | email | dictation | calendar | codex | chatgpt
    time: datetime                  # available_at
    text: str                       # searchable body (redacted, neutralised)
    speaker: str = ""               # who said/wrote it ("" if unknown)
    speaker_known: bool = True      # False for unidentified meeting speakers
    speaker_conf: float = 1.0
    where: str = ""                 # channel / meeting title / app: short location label
    title: str = ""
    idx: int = -1                   # position inside its record (meeting segment order)
    dates: tuple = ()               # extra calendar dates this unit is "about" (event day, ...)
    kind: str = "message"           # message | edit | deleted_marker | calendar | session | ...
    target: str = ""                # for edits: the message being edited
    thread: str = ""                # thread parent id (slack)
    channel_id: str = ""
    injected: bool = False          # contained a planted instruction (neutralised)
    meta: dict = field(default_factory=dict)

    @property
    def date(self):
        return self.time.astimezone(LOCAL).date()


class Store:
    def __init__(self, data_dir=None):
        self.dir = Path(data_dir or DATA_DIR)
        self.units = []
        self.by_id = {}
        self.deleted = {}          # target id -> deletion time
        self.edits = {}            # target id -> [(time, new text, edit event id)]
        self.people = {}           # slack id -> real name
        self.user_rows = []
        self.channels = {}
        self.channel_rows = []
        self.records = {}          # record id -> ordered list of unit ids (meeting segments)
        self.meetings = {}         # meeting id -> metadata
        self._load()

    # ------------------------------------------------------------------ loading
    def _add(self, u):
        u.text = safety.redact(u.text)
        u.text, inj = safety.neutralise(u.text)
        u.injected = u.injected or inj
        self.units.append(u)
        self.by_id[u.id] = u
        return u

    def _load(self):
        d = self.dir
        self.user_rows = json.load(open(d / "connectors/slack/users.json", encoding="utf-8"))
        self.people = {u["id"]: u.get("real_name") or u.get("name") for u in self.user_rows}
        self.channel_rows = json.load(open(d / "connectors/slack/channels.json", encoding="utf-8"))
        self.channels = {c["id"]: c for c in self.channel_rows}

        # meetings: one unit per diarized segment
        for f in sorted((d / "native/meetings").glob("*.json")):
            m = json.load(open(f, encoding="utf-8"))
            start = parse_dt(m["start"])
            self.meetings[m["id"]] = {k: v for k, v in m.items() if k != "segments"}
            ids = []
            for i, s in enumerate(m["segments"]):
                name = s.get("speaker_name")
                u = Unit(id=s["seg_id"], record=m["id"], source="meeting",
                         time=start + timedelta(seconds=s["end_s"]), text=s["text"],
                         speaker=name or s.get("speaker_label") or "Unknown speaker",
                         speaker_known=bool(name), speaker_conf=float(s.get("speaker_confidence") or 0.0),
                         where=m["title"], title=m["title"], idx=i, channel_id=s.get("channel", ""),
                         meta={"start_s": s["start_s"], "label": s.get("speaker_label"),
                               "meeting_date": m["start"][:10]})
                self._add(u)
                ids.append(u.id)
            self.records[m["id"]] = ids

        for line in open(d / "native/dictation/dictations.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            body = x.get("cleaned_text") or x.get("raw_transcript") or ""
            raw = x.get("raw_transcript") or ""
            self._add(Unit(id=x["id"], record=x["id"], source="dictation", time=parse_dt(x["timestamp"]),
                           text=body + ("\n" + raw if raw and raw != body else ""), speaker="Alex Rivera",
                           where=f"{x.get('target_app', '')} {x.get('target_context', '')}".strip(),
                           title=x.get("mode", ""), kind=x.get("mode", "dictation"),
                           meta={"delivery_state": x.get("delivery_state"), "mode": x.get("mode"),
                                 "target_app": x.get("target_app"), "target_context": x.get("target_context"),
                                 "cleaned": body}))

        for line in open(d / "connectors/slack/messages.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            t = parse_dt(x["ts"])
            ch = self.channels.get(x.get("channel_id"), {})
            chname = ch.get("name", x.get("channel_id", ""))
            if ch.get("is_dm"):
                others = [self.people.get(m, m) for m in ch.get("members", []) if m != "U01ALEX"]
                chname = "DM with " + (", ".join(others) or chname)
            st = x.get("subtype")
            if st == "message_deleted":
                self.deleted[x["target_id"]] = t
                self._add(Unit(id=x["id"], record=x["id"], source="slack", time=t, text="", kind="deleted_marker",
                               target=x["target_id"], where=chname, channel_id=x.get("channel_id", "")))
                continue
            if st == "message_changed":
                tgt = x["target_id"]
                self.edits.setdefault(tgt, []).append((t, x["text"], x["id"]))
                orig = None
                u = Unit(id=x["id"], record=x["id"], source="slack", time=t, text=x["text"], where=chname,
                         kind="edit", target=tgt, channel_id=x.get("channel_id", ""))
                self._add(u)
                continue
            who = self.people.get(x.get("user"), x.get("bot_name") or x.get("user") or "")
            self._add(Unit(id=x["id"], record=x["id"], source="slack", time=t, text=x.get("text", ""),
                           speaker=who, where=chname, thread=x.get("thread_parent_id") or "",
                           channel_id=x.get("channel_id", ""),
                           meta={"user": x.get("user"), "reactions": x.get("reactions") or [],
                                 "bot": x.get("subtype") == "bot_message"}))
        # edit events inherit the speaker of the message they edit
        for tgt, evs in self.edits.items():
            for (_, _, eid) in evs:
                if tgt in self.by_id:
                    self.by_id[eid].speaker = self.by_id[tgt].speaker
                    self.by_id[eid].thread = self.by_id[tgt].thread

        for line in open(d / "connectors/gmail/messages.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            body = f"{x['subject']}\n{x['body']}"
            atts = ", ".join(a.get("filename", "") for a in x.get("attachments") or [])
            if atts:
                body += f"\n(attachments: {atts})"
            self._add(Unit(id=x["id"], record=x["id"], source="email", time=parse_dt(x["date"]), text=body,
                           speaker=x["from"], where=x["subject"], title=x["subject"], thread=x.get("thread_id", ""),
                           meta={"from": x["from"], "to": x["to"], "cc": x.get("cc") or [],
                                 "labels": x.get("labels") or [], "subject": x["subject"], "body": x["body"]}))

        for line in open(d / "connectors/google_calendar/events.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            st, en = x["start"], x["end"]
            s0 = st.get("dateTime") or st.get("date")
            e0 = en.get("dateTime") or en.get("date")
            att = ", ".join(f"{a['email']} ({a.get('responseStatus', '')})" for a in x.get("attendees", []))
            body = (f"{x['summary']} | {s0} to {e0} | {x.get('location') or ''} | attendees: {att} | "
                    f"{x.get('description') or ''} | status: {x['status']}"
                    f"{' | repeats ' + str(x['recurrence']) if x.get('recurrence') else ''}")
            edate = parse_dt(s0).date() if "T" in s0 else datetime.fromisoformat(s0).date()
            occ = expand_recurrence(edate, x.get("recurrence") or []) if x.get("recurrence") else (edate,)
            self._add(Unit(id=x["id"], record=x["id"], source="calendar", time=parse_dt(x["updated"]), text=body,
                           speaker=x.get("organizer", ""), where=x.get("location") or "", title=x["summary"],
                           kind="calendar", dates=occ,
                           meta={"start": s0, "end": e0, "summary": x["summary"], "status": x["status"],
                                 "attendees": [a["email"] for a in x.get("attendees", [])],
                                 "recurrence": x.get("recurrence"), "location": x.get("location"),
                                 "all_day": "T" not in s0}))

        for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
            ev = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            meta, body = ev[0], ev[1:]
            chunks = [f"{e.get('role', e.get('tool', e['type']))}: {e.get('content') or e.get('input', '')}"
                      + (f"\n-> {e['output']}" if e.get("output") else "") for e in body]
            last = body[-1]["timestamp"] if body else meta["started_at"]
            self._add(Unit(id=meta["id"], record=meta["id"], source="codex", time=parse_dt(last),
                           text="\n".join(chunks), speaker="Alex Rivera", where=f"Codex {meta.get('repo', '')}",
                           title=meta.get("repo", ""), kind="session", meta={"chunks": chunks, "repo": meta.get("repo")}))

        for c in json.load(open(d / "connectors/chatgpt/conversations.json", encoding="utf-8")):
            for m in c["messages"]:
                self._add(Unit(id=m["id"], record=c["id"], source="chatgpt", time=parse_dt(m["create_time"]),
                               text=m["content"], speaker="Alex Rivera" if m["role"] == "user" else "ChatGPT",
                               where=f"ChatGPT: {c['title']}", title=c["title"],
                               meta={"role": m["role"], "conv": c["id"]}))
        self.units.sort(key=lambda u: (u.time, u.id))

    # ------------------------------------------------------------------ the gate
    def is_deleted(self, uid, as_of):
        t = self.deleted.get(uid)
        return t is not None and t <= as_of

    def current_text(self, u, as_of):
        """Text of a unit as of `as_of` (latest edit that already happened wins)."""
        ev = [(t, txt) for (t, txt, _) in self.edits.get(u.id, []) if t <= as_of]
        if not ev:
            return u.text, False
        txt = sorted(ev)[-1][1]
        return safety.neutralise(safety.redact(txt))[0], True

    def visible(self, as_of):
        """Units a system may know at `as_of`. Deleted-marker units are never returned."""
        out = []
        for u in self.units:
            if u.time > as_of or u.kind == "deleted_marker" or self.is_deleted(u.id, as_of):
                continue
            out.append(u)
        return out

    def neighbors(self, u, n=2):
        """Adjacent meeting segments (same meeting)."""
        ids = self.records.get(u.record)
        if not ids or u.idx < 0:
            return []
        lo, hi = max(0, u.idx - n), min(len(ids), u.idx + n + 1)
        return [self.by_id[i] for i in ids[lo:hi] if i != u.id]
