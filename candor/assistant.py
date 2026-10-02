"""TextOS assistant: talk to your memory and turn commands into real, reviewable effects — safely.

  * Questions go to memory (time-travel with /asof), commands go to the action planner.
  * Nothing executes without you typing y. Even then only local-safe effects run:
      calendar.create/update_event -> a .ics file you can double-click (updates reuse the event UID)
      reminder.create               -> out/reminders.jsonl + a .ics VTODO with an alarm
      app.open                      -> launches the app (allow-listed names only)
      gmail.send                    -> a mailto: link + out/outbox.jsonl entry (never sent by us)
      slack.send_message            -> out/outbox.jsonl entry (never sent by us)
      confirm                       -> destructive actions are never automated; we tell you what to review
  * clarify questions are asked back and the command is re-planned with your answer.
"""
import json
import os
import platform
import re
import subprocess
import sys
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .actions import APPS, Planner
from .memory import Memory
from .store import LOCAL, parse_dt

OUT = Path(__file__).resolve().parent.parent / "out"


def _ics_dt(iso):
    return parse_dt(iso).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _esc(s):
    return str(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def write_ics(kind, args, uid=None):
    """kind: event | todo. Returns path."""
    OUT.mkdir(exist_ok=True)
    (OUT / "exports").mkdir(exist_ok=True)
    uid = uid or f"{uuid.uuid4()}@candor"
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Candor//TextOS//EN", "METHOD:PUBLISH"]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if kind == "event":
        lines += ["BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{now}", f"SUMMARY:{_esc(args.get('title') or args.get('summary') or 'Event')}",
                  f"DTSTART:{_ics_dt(args['start'])}"]
        if args.get("end"):
            lines.append(f"DTEND:{_ics_dt(args['end'])}")
        for a in args.get("attendees", []) or []:
            lines.append(f"ATTENDEE;CN={a}:mailto:{a}")
        lines.append("END:VEVENT")
    else:
        lines += ["BEGIN:VTODO", f"UID:{uid}", f"DTSTAMP:{now}", f"SUMMARY:{_esc(args['text'])}", f"DUE:{_ics_dt(args['due'])}",
                  "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_esc(args['text'])}", "TRIGGER;RELATED=END:PT0S", "END:VALARM", "END:VTODO"]
    lines.append("END:VCALENDAR")
    p = OUT / "exports" / f"{kind}-{re.sub(r'[^a-z0-9]+', '-', uid.lower())[:40]}.ics"
    p.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return p


def open_app(name):
    key = name.lower()
    if key not in {v.lower() for v in APPS.values()} | set(APPS):
        return f"'{name}' is not on the allow-list, not opening it."
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            subprocess.Popen(["open", "-a", name])
        elif sysname == "Windows":
            os.startfile(name)      # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", name])
        return f"opened {name}"
    except Exception as e:
        return f"could not open {name}: {e}"


def outbox(entry):
    OUT.mkdir(exist_ok=True)
    with open(OUT / "outbox.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def execute_local(action, as_of, do=True):
    """Execute one local-safe action. Returns a human-readable result line."""
    t, a = action["type"], action["args"]
    if t == "calendar.create_event":
        return f"wrote calendar file: {write_ics('event', a)}"
    if t == "calendar.update_event":
        return f"wrote updated event (same UID {a['event_id']}): {write_ics('event', a, uid=a['event_id'])}"
    if t == "reminder.create":
        OUT.mkdir(exist_ok=True)
        with open(OUT / "reminders.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
        return f"saved reminder + alarm file: {write_ics('todo', a)}"
    if t == "app.open":
        return open_app(a["app"])
    if t == "gmail.send":
        outbox({"type": t, "args": a, "status": "queued-not-sent"})
        q = urllib.parse.urlencode({"cc": ",".join(a.get("cc", [])), "subject": a.get("subject", ""), "body": a.get("body", "")}, quote_via=urllib.parse.quote)
        return f"NOT sent. Review and send yourself: mailto:{','.join(a['to'])}?{q}"
    if t == "slack.send_message":
        outbox({"type": t, "args": a, "status": "queued-not-sent"})
        return "NOT sent: queued in out/outbox.jsonl for you to review and post."
    if t == "confirm":
        return "Not executed. Destructive actions are never automated; do it yourself once you have checked: " + a["summary"]
    return "nothing to execute"


def describe(a):
    t, g = a["type"], a["args"]
    if t == "slack.send_message":
        return f"Slack -> {g['to']}: {g['text']}"
    if t == "gmail.send":
        return f"Email -> {', '.join(g['to'])}" + (f" (cc {', '.join(g['cc'])})" if g.get("cc") else "") + f"\n      Subject: {g['subject']}\n      " + g["body"].replace("\n", "\n      ")
    if t == "calendar.create_event":
        return f"Create event '{g['title']}' {g['start']} -> {g['end']} with {', '.join(g.get('attendees', [])) or 'nobody else'}"
    if t == "calendar.update_event":
        return f"Update {g['event_id']}: {g.get('start')} -> {g.get('end')}"
    if t == "reminder.create":
        return f"Remind at {g['due']}: {g['text']}"
    if t == "app.open":
        return f"Open app {g['app']}"
    return f"{t}: {json.dumps(g, ensure_ascii=False)}"


def main(execute=False, as_of="2026-09-18T18:00:00-07:00", no_llm=False):
    from .llm import LLM
    llm = None if no_llm else LLM()
    llm = llm if (llm and llm.available) else None
    mem = Memory(llm=llm, use_llm=llm is not None)
    planner = Planner(mem.store, memory=mem, llm=llm)
    run_exec = execute
    now = parse_dt(as_of)
    print(f"TextOS  (as of {now:%a %b %d %H:%M})  — ask a question or give a command.  /asof <ISO>  /why  /quit"
          f"  |  execution: {'ON (asks first)' if run_exec else 'OFF (dry run; start with --execute)'}")
    last = None
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/q", "exit", "quit"):
            return
        if line.startswith("/asof"):
            try:
                now = parse_dt(line.split(None, 1)[1])
                print(f"time travel: it is now {now:%a %b %d %H:%M}. Nothing later exists.")
            except Exception:
                print("usage: /asof 2026-09-16T10:00:00-07:00")
            continue
        if line == "/why":
            if last:
                for i in last["sources"] or last["retrieved"][:5]:
                    u = mem.store.by_id[i]
                    print(f"  [{i}] {u.time.astimezone(LOCAL):%b %d %H:%M} {u.where[:40]}: {u.text[:110]!r}")
            continue
        acts = planner.plan(line, now)
        while acts and acts[0]["type"] == "clarify":
            ans = input(f"  ? {acts[0]['args']['question']}\n  > ").strip()
            if not ans:
                acts = []
                break
            first = acts[0]["args"]["question"]
            m = re.search(r"Which (\w+)", first)
            # replace the ambiguous first name in the original command by the full name the user gave
            line = re.sub(r"\b" + re.escape(m.group(1)) + r"\b", ans, line, count=1, flags=re.I) if m and len(ans.split()) >= 2 else f"{line} {ans}"
            acts = planner.plan(line, now)
            if acts and acts[0]["type"] == "clarify":
                print("  Still ambiguous; try again with the full name.")
                acts = []
        for a in acts:
            if a["type"] == "memory.ask":
                r = mem.ask(a["args"]["question"], now)
                last = r
                print(r["answer"])
                print("  sources:", ", ".join(r["sources"]) or "-")
                continue
            print("  planned:", describe(a))
            if not run_exec:
                continue
            if a["type"] == "confirm":
                print("  " + execute_action(a, now))
                continue
            if input("  run it? [y/N] ").strip().lower() == "y":
                print("  ->", execute_action(a, now))


def execute_action(a, now):
    return execute_local(a, now)
