#!/usr/bin/env python3
"""Candor — one entry point.  Author: Bishal Chandra Debnath

  python run.py memory  --questions Q.jsonl  --out A.jsonl        answer memory questions
  python run.py actions --commands  C.jsonl  --out P.jsonl        plan actions (dry run)
  python run.py eval                                              run every scorer on the train sets + leak audit
  python run.py enrich [--limit N]                                one-time model pass that annotates every record (resumable)
  python run.py report                                            write the latest scores into docs/RESULTS.md and the README
  python run.py audit                                             time-travel / deletion / secret leak audit
  python run.py diagnose [--gold G] [--answers A]                 failure taxonomy: search miss vs answer miss
  python run.py ask "question" [--as-of ISO]                      ask one question
  python run.py act "command" [--as-of ISO]                       plan one command
  python run.py commitments [--as-of ISO]                         ledger of promises and their status
  python run.py timeline "topic" [--as-of ISO]                    how a fact changed over time
  python run.py assistant [--execute]                             interactive assistant with safe local execution

Standard library only. Needs no key to run (degraded answers); set GEMINI_API_KEY in .env for full answers.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_AS_OF = "2026-09-18T18:00:00-07:00"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_llm(args):
    from candor.llm import LLM
    if getattr(args, "no_llm", False):
        return None
    llm = LLM()
    return llm if llm.available else None


def _ablations(args):
    if getattr(args, "no_enrich", False):
        os.environ["CANDOR_ENRICH"] = "0"
    if getattr(args, "no_plan", False):
        os.environ["CANDOR_PLAN"] = "0"
    if getattr(args, "plan", False):
        os.environ["CANDOR_PLAN"] = "1"


def cmd_enrich(args):
    from candor.enrich import build
    from candor.llm import LLM
    from candor.store import Store
    llm = LLM()
    if not llm.available:
        sys.exit("enrich needs a model key (GEMINI_API_KEY in .env)")
    r = build(Store(), llm, limit=args.limit)
    print(r, llm.stats())


def cmd_memory(args):
    _ablations(args)
    from candor.memory import Memory
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=False if llm is None else True, llm_rerank=not args.no_rerank)
    if llm is None:
        print("[candor] no LLM key: using deterministic retrieval + extractive answers (set GEMINI_API_KEY in .env for full answers)", file=sys.stderr)
    rows, t0 = [], time.time()
    qs = read_jsonl(args.questions)
    for i, q in enumerate(qs, 1):
        try:
            r = mem.ask(q["question"], q["as_of"])
        except Exception as e:      # never lose a line: a crash becomes an honest abstention
            r = {"answer": "I don't have that in memory.", "sources": [], "retrieved": [], "abstained": True, "_meta": {"error": repr(e)}}
        rows.append({"id": q["id"], "answer": r["answer"], "sources": r["sources"], "retrieved": r["retrieved"], "abstained": r["abstained"]})
        print(f"[{i}/{len(qs)}] {q['id']} {'(abstained) ' if r['abstained'] else ''}{r['answer'][:90]}", file=sys.stderr)
    write_jsonl(args.out, rows)
    if llm is not None:
        print("[candor] llm:", llm.stats(), file=sys.stderr)
    print(f"[candor] wrote {len(rows)} answers to {args.out} in {time.time() - t0:.1f}s", file=sys.stderr)


def cmd_actions(args):
    from candor.memory import Memory
    from candor.actions import Planner
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=False if llm is None else True)
    planner = Planner(mem.store, memory=mem, llm=llm)
    rows = []
    for c in read_jsonl(args.commands):
        try:
            acts = planner.plan(c["command"], c["as_of"])
        except Exception as e:
            acts = [{"type": "clarify", "args": {"question": "Sorry, I couldn't process that command."}}]
        rows.append({"id": c["id"], "actions": acts})
    write_jsonl(args.out, rows)
    print(f"[candor] wrote {len(rows)} action plans to {args.out}", file=sys.stderr)


def cmd_eval(args):
    _ablations(args)
    from candor.evalrun import run_scorer, leak_audit
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    ns = argparse.Namespace(questions=str(ROOT / "evals/memory_train.jsonl"), out=str(out / "memory_answers.jsonl"),
                            no_llm=args.no_llm, no_rerank=args.no_rerank, no_enrich=False, no_plan=False, plan=False)
    cmd_memory(ns)
    print("\n=== RETRIEVAL ===")
    print(run_scorer("score_retrieval.py", "--gold", "../evals/memory_train.jsonl", "--answers", str(out / "memory_answers.jsonl"),
                     "--out", str(out / "retrieval_report.json")))
    if args.dev:     # the questions I did NOT tune on: retrieval score plus the failure taxonomy
        dev_out = out / "memory_dev_answers.jsonl"
        cmd_memory(argparse.Namespace(questions=str(ROOT / "evals/memory_dev.jsonl"), out=str(dev_out), no_llm=args.no_llm,
                                      no_rerank=args.no_rerank, no_enrich=False, no_plan=False, plan=False))
        print("\n=== DEV SET: RETRIEVAL ===")
        print(run_scorer("score_retrieval.py", "--gold", "../evals/memory_dev.jsonl", "--answers", str(dev_out), "--out", str(out / "retrieval_dev_report.json")))
        from candor.diagnose import diagnose, render
        from candor.memory import Memory
        gold = read_jsonl(ROOT / "evals/memory_dev.jsonl")
        print("=== DEV SET: FAILURE TAXONOMY ===")
        print(render(diagnose(Memory(use_llm=False), gold, {r["id"]: r for r in read_jsonl(dev_out)})))
    judge = args.judge
    jargs = ["--judge", judge]
    if judge == "gemini":      # the official judge speaks OpenAI-compatible; Gemini's free tier exposes that endpoint
        from candor.llm import LLM
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("--judge gemini needs GEMINI_API_KEY in .env")
        os.environ["OPENAI_API_KEY"] = key
        os.environ["OPENAI_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai"
        model = args.judge_model or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        jargs = ["--judge", "openai", "--model", model]
    elif args.judge_model:
        jargs += ["--model", args.judge_model]
    print("=== ANSWERS (judge: %s) ===" % judge)
    print(run_scorer("../candor/judge_runner.py" if judge != "none" else "score_memory.py", "--gold", "../evals/memory_train.jsonl", "--answers", str(out / "memory_answers.jsonl"),
                     "--out", str(out / "memory_report.json"), *jargs))
    ns = argparse.Namespace(commands=str(out / "action_commands.jsonl"), out=str(out / "action_predictions.jsonl"), no_llm=args.no_llm)
    write_jsonl(ns.commands, [{"id": d["id"], "command": d["command"], "as_of": d["as_of"]} for d in read_jsonl(ROOT / "evals/actions_train.jsonl")])
    cmd_actions(ns)
    print("\n=== ACTIONS ===")
    print(run_scorer("score_actions.py", "--gold", "../evals/actions_train.jsonl", "--predictions", str(out / "action_predictions.jsonl"),
                     "--out", str(out / "actions_report.json")))
    if not args.skip_audit:
        cmd_audit(argparse.Namespace(times=10))
    from candor.evalrun import write_report       # the README results block always matches the last eval
    write_report()
    print("[candor] README results block and docs/RESULTS.md updated from this run")


def cmd_audit(args):
    from candor.evalrun import leak_audit
    r = leak_audit(n_times=args.times)
    print(f"=== LEAK AUDIT === {r['queries']} retrievals across random as_of times")
    if r["violations"]:
        print("VIOLATIONS:", len(r["violations"]))
        for v in r["violations"][:20]:
            print("  ", v)
        sys.exit(1)
    print("0 violations: nothing from the future, nothing deleted, no secrets, no deletion markers.")


def cmd_report(args):
    from candor.evalrun import write_report
    print(write_report())


def cmd_diagnose(args):
    from candor.memory import Memory
    from candor.diagnose import diagnose, render
    gold = read_jsonl(args.gold)
    answers = {r["id"]: r for r in read_jsonl(args.answers)} if args.answers else None
    mem = Memory(use_llm=False)
    print(render(diagnose(mem, gold, answers)))


def cmd_ask(args):
    from candor.memory import Memory
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=llm is not None)
    r = mem.ask(args.question, args.as_of)
    print(r["answer"])
    print("sources:", ", ".join(r["sources"]) or "-", "| abstained:", r["abstained"])
    print("retrieved:", ", ".join(r["retrieved"][:10]))


def cmd_act(args):
    from candor.memory import Memory
    from candor.actions import Planner
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=llm is not None)
    print(json.dumps(Planner(mem.store, memory=mem, llm=llm).plan(args.command, args.as_of), indent=2, ensure_ascii=False))


def cmd_commitments(args):
    from candor.memory import Memory
    from candor.ledger import commitments, render_commitments
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=llm is not None)
    print(render_commitments(commitments(mem, args.as_of, llm)))


def cmd_timeline(args):
    from candor.memory import Memory
    from candor.ledger import timeline, render_timeline
    llm = make_llm(args)
    mem = Memory(llm=llm, use_llm=llm is not None)
    print(render_timeline(timeline(mem, args.topic, args.as_of, llm)))


def cmd_assistant(args):
    from candor.assistant import main
    main(execute=args.execute, as_of=args.as_of, no_llm=args.no_llm)


def main():
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        p.add_argument("--no-llm", action="store_true", help="never call an LLM (deterministic mode)")
        return p

    p = add("memory", cmd_memory, "answer a JSONL of questions")
    p.add_argument("--questions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-rerank", action="store_true", help="keep deterministic retrieval order (ignore LLM evidence picks)")
    p.add_argument("--no-enrich", action="store_true", help="ablation: ignore enrich/annotations.jsonl")
    p.add_argument("--no-plan", action="store_true", help="skip the model's query plan (the default)")
    p.add_argument("--plan", action="store_true", help="experimental: let the model plan the search (off by default, see README)")
    p = add("enrich", cmd_enrich, "one-time model pass that annotates every record (resumable, about 130 calls)")
    p.add_argument("--limit", type=int, default=None, help="stop after N batches (use to spread over several days of quota)")
    p = add("actions", cmd_actions, "plan actions for a JSONL of commands (dry run)")
    p.add_argument("--commands", required=True)
    p.add_argument("--out", required=True)
    p = add("eval", cmd_eval, "run all scorers on the train sets")
    p.add_argument("--judge", default="none", help="none | gemini | anthropic | openai | claude-cli (see eval_harness/judge.py)")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-enrich", action="store_true")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--dev", action="store_true", help="also run the 28-question dev set I did not tune on (about 60 more model calls)")
    p = add("audit", cmd_audit, "leak audit")
    p.add_argument("--times", type=int, default=25)
    p = add("report", cmd_report, "write the latest scorer results into docs/RESULTS.md and the README results block")
    p = add("diagnose", cmd_diagnose, "failure taxonomy: search miss vs answer miss")
    p.add_argument("--gold", default=str(ROOT / "evals/memory_dev.jsonl"))
    p.add_argument("--answers", default=None, help="a memory answers JSONL, to diagnose the LLM path too")
    for name, fn, h in (("ask", cmd_ask, "ask one question"), ("act", cmd_act, "plan one command"),
                        ("commitments", cmd_commitments, "commitments ledger"), ("timeline", cmd_timeline, "topic timeline"),
                        ("assistant", cmd_assistant, "interactive assistant")):
        p = add(name, fn, h)
        p.add_argument("--as-of", default=DEFAULT_AS_OF)
        if name == "ask":
            p.add_argument("question")
        if name == "act":
            p.add_argument("command")
        if name == "timeline":
            p.add_argument("topic")
        if name == "assistant":
            p.add_argument("--execute", action="store_true", help="really execute local-safe actions (.ics / reminders file / open app)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
