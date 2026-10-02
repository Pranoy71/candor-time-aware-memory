"""Run: python -m unittest discover -s tests -v   (standard library only; no key needed)"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONUTF8"] = "1"

from candor import safety, timeparse as T
from candor.actions import Planner
from candor.evalrun import run_scorer, leak_audit
from candor.llm import MockLLM, parse_json
from candor.memory import Memory
from candor.store import Store, parse_dt

_MEM = None


def mem():
    global _MEM
    if _MEM is None:
        _MEM = Memory(use_llm=False)
    return _MEM


class TestGate(unittest.TestCase):
    def test_future_hidden(self):
        st = mem().store
        t = parse_dt("2026-09-10T09:00:00-07:00")
        vis = {u.id for u in st.visible(t)}
        self.assertNotIn("EM-F-036", vis)            # Oct 21 Linear email (Sep 16)
        self.assertTrue(all(u.time <= t for u in st.visible(t)))

    def test_deleted_gone_from_deletion_time(self):
        st = mem().store
        dele = st.deleted["SL-DM-AB-0915-2"]
        from datetime import timedelta
        posted = st.by_id["SL-DM-AB-0915-2"].time
        self.assertLess(posted, dele)
        before = {u.id for u in st.visible(dele - timedelta(seconds=1))}
        after = {u.id for u in st.visible(dele)}
        self.assertTrue("SL-DM-AB-0915-2" in before, "message should exist until the moment it is deleted")
        self.assertTrue("SL-DM-AB-0915-2" not in after, "message must be gone from the deletion time on")

    def test_edit_replaces_text_from_edit_time(self):
        st = mem()
        u = st.store.by_id["SL-RP-0916-1"]
        edit_t = st.store.edits["SL-RP-0916-1"][0][0]
        old, e1 = st.store.current_text(u, edit_t.replace(hour=edit_t.hour - 1))
        new, e2 = st.store.current_text(u, edit_t)
        self.assertFalse(e1)
        self.assertTrue(e2)
        self.assertIn("60", old)
        self.assertIn("61", new)

    def test_recurrence_expansion(self):
        from datetime import date
        st = mem().store
        on = {u.id for u in st.units if u.source == "calendar" and date(2026, 9, 23) in u.dates}
        self.assertTrue({"CAL-BOARD", "CAL-F-04", "CAL-STANDUP", "CAL-F-25"} <= on)
        self.assertFalse(any(date(2026, 9, 25) in u.dates for u in st.units if u.id == "CAL-F-07"))   # EXDATE


class TestSafety(unittest.TestCase):
    def test_secret_never_indexed(self):
        for u in mem().store.units:
            self.assertFalse(safety.has_secret(u.text), u.id)

    def test_injection_neutralised(self):
        u = mem().store.by_id["EM-F-050"]
        self.assertTrue(u.injected)
        self.assertNotIn("<!--", u.text)

    def test_output_sanitised(self):
        self.assertNotIn("sk-abc12345678", safety.sanitize_output("the key is sk-abc12345678 ok"))

    def test_leak_audit(self):
        r = leak_audit(n_times=4)
        self.assertEqual(r["violations"], [], r["violations"][:5])


class TestRetrieval(unittest.TestCase):
    def test_train_retrieval_regression(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.jsonl"
            qs = [json.loads(l) for l in open(ROOT / "evals/memory_train.jsonl", encoding="utf-8")]
            with open(out, "w", encoding="utf-8") as f:
                for q in qs:
                    r = mem().ask(q["question"], q["as_of"])
                    f.write(json.dumps({"id": q["id"], "answer": r["answer"], "sources": r["sources"],
                                        "retrieved": r["retrieved"], "abstained": r["abstained"]}) + "\n")
            rep = run_scorer("score_retrieval.py", "--gold", "../evals/memory_train.jsonl", "--answers", str(out))
            m = re.search(r"retrieval score\s+([\d.]+)%", rep)
            self.assertGreaterEqual(float(m.group(1)), 88.0, rep[-400:])
            self.assertIn("forbidden records retrieved (top 20): 0", rep)

    def test_two_sarahs(self):
        r = mem().ask("What did Sarah Kim say about SSO?", "2026-09-18T18:00:00-07:00")
        self.assertTrue(any(i.startswith("SL-DM-AS") for i in r["retrieved"][:5]))

    def test_unanswerable_abstains_offline(self):
        r = mem().ask("What is Dana's salary?", "2026-09-18T18:00:00-07:00")
        self.assertTrue(r["abstained"])


class TestLLMPath(unittest.TestCase):
    """The answer writer with a scripted LLM: proves validation, not model quality."""

    def test_invalid_and_future_ids_dropped_and_secret_redacted(self):
        def fake(system, user):
            return json.dumps({"answer": "Launch is Oct 21. key sk-brightline-abcdef123456",
                               "sources": ["EM-F-036", "NOPE-1", "SL-RP-0910-1"], "relevant": ["NOPE-1", "EM-F-036"], "abstain": False})
        m = Memory(llm=MockLLM(fake))
        r = m.ask("When is Route Planner v2 launching?", "2026-09-12T12:00:00-07:00")    # EM-F-036 is Sep 16 -> future
        self.assertNotIn("EM-F-036", r["sources"] + r["retrieved"])
        self.assertNotIn("NOPE-1", r["sources"] + r["retrieved"])
        self.assertNotIn("sk-brightline", r["answer"])

    def test_abstain_clears_sources(self):
        m = Memory(llm=MockLLM(lambda s, u: json.dumps({"answer": "I don't have that in memory.", "sources": ["EM-F-036"], "relevant": [], "abstain": True})))
        r = m.ask("What is Dana's salary?", "2026-09-18T18:00:00-07:00")
        self.assertTrue(r["abstained"])
        self.assertEqual(r["sources"], [])

    def test_llm_failure_falls_back(self):
        from candor.llm import LLMError

        class Boom(MockLLM):
            def complete(self, *a, **k):
                raise LLMError("quota")
        r = Memory(llm=Boom(lambda s, u: "")).ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")
        self.assertTrue(r["retrieved"])
        self.assertIn("error", r["_meta"])

    def test_rerank_promotes_but_never_drops(self):
        base = Memory(use_llm=False).ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")["retrieved"]
        pick = base[7]
        m = Memory(llm=MockLLM(lambda s, u: json.dumps({"answer": "Oct 21.", "sources": [pick], "relevant": [pick], "abstain": False})))
        r = m.ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")["retrieved"]
        self.assertEqual(r[0], pick)
        self.assertTrue(set(base) <= set(r) | set(base[-1:]))

    def test_planted_instruction_not_in_prompt(self):
        seen = []
        m = Memory(llm=MockLLM(lambda s, u: (seen.append(u), json.dumps({"answer": "No.", "sources": [], "relevant": [], "abstain": False}))[1]))
        m.ask("Has Acme signed the contract?", "2026-09-18T18:00:00-07:00")
        self.assertNotIn("<!--", seen[0])
        self.assertNotIn("sk-brightline", seen[0])

    def test_parse_json_tolerant(self):
        self.assertEqual(parse_json("```json\n{\"a\": 1}\n```"), {"a": 1})
        self.assertEqual(parse_json("Sure! {\"a\": [1]} hope that helps"), {"a": [1]})


class TestTimeAndActions(unittest.TestCase):
    now = parse_dt("2026-09-16T12:00:00-07:00")

    def test_parse(self):
        w = T.find_when("tomorrow at 2", self.now)
        self.assertEqual(T.iso(T.resolve(w, self.now)), "2026-09-17T14:00:00-07:00")
        w = T.find_when("on the 25th at 9am", self.now)
        self.assertEqual(T.iso(T.resolve(w, self.now)), "2026-09-25T09:00:00-07:00")
        w = T.find_when("next Tuesday at noon", parse_dt("2026-09-17T12:00:00-07:00"))
        self.assertEqual(T.iso(T.resolve(w, self.now)), "2026-09-22T12:00:00-07:00")

    def test_train_actions(self):
        for f in ("actions_train.jsonl", "actions_extra.jsonl"):
            p = Planner(mem().store, memory=mem())
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "p.jsonl"
                with open(out, "w", encoding="utf-8") as fh:
                    for l in open(ROOT / "evals" / f, encoding="utf-8"):
                        c = json.loads(l)
                        fh.write(json.dumps({"id": c["id"], "actions": p.plan(c["command"], c["as_of"])}) + "\n")
                rep = run_scorer("score_actions.py", "--gold", f"../evals/{f}", "--predictions", str(out))
                self.assertIn("pass rate 100.0%", rep, rep[-600:])

    def test_safety_rules(self):
        p = Planner(mem().store, memory=mem())
        a = p.plan("Delete all my emails from Marcus", "2026-09-18T09:00:00-07:00")
        self.assertEqual([x["type"] for x in a], ["confirm"])
        a = p.plan("Message Sarah", "2026-09-18T09:00:00-07:00")
        self.assertEqual(a[0]["type"], "clarify")
        a = p.plan("blorp the frobnicator", "2026-09-18T09:00:00-07:00")
        self.assertEqual(a[0]["type"], "clarify")          # unknown command never guesses

    def test_llm_actions_validated(self):
        bad = {"actions": [{"type": "slack.send_message", "args": {"to": "U-FAKE", "text": "hi"}},
                           {"type": "calendar.update_event", "args": {"event_id": "CAL-NOPE", "start": "x"}},
                           {"type": "app.open", "args": {"app": "Figma"}}]}
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: json.dumps(bad)))
        a = p.plan("gimme figma", "2026-09-18T09:00:00-07:00")
        self.assertEqual([x["type"] for x in a], ["app.open"])       # hallucinated ids dropped


class TestCLI(unittest.TestCase):
    def test_memory_and_actions_commands(self):
        with tempfile.TemporaryDirectory() as d:
            q, o = Path(d) / "q.jsonl", Path(d) / "a.jsonl"
            q.write_text(open(ROOT / "examples/memory_questions.example.jsonl", encoding="utf-8").read(), encoding="utf-8")
            r = subprocess.run([sys.executable, str(ROOT / "run.py"), "memory", "--no-llm", "--questions", str(q), "--out", str(o)],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            rows = [json.loads(l) for l in open(o, encoding="utf-8")]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(set(row), {"id", "answer", "sources", "retrieved", "abstained"})
                self.assertLessEqual(len(row["retrieved"]), 20)
            c, p = Path(d) / "c.jsonl", Path(d) / "p.jsonl"
            c.write_text(open(ROOT / "examples/action_commands.example.jsonl", encoding="utf-8").read(), encoding="utf-8")
            r = subprocess.run([sys.executable, str(ROOT / "run.py"), "actions", "--no-llm", "--commands", str(c), "--out", str(p)],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            self.assertEqual(len([l for l in open(p, encoding="utf-8")]), 2)


class TestLedger(unittest.TestCase):
    def test_pricing_promise_followed_through(self):
        from candor.ledger import commitments, timeline
        items = commitments(mem(), "2026-09-18T18:00:00-07:00")
        pr = [i for i in items if "pricing proposal" in i["what"].lower() and i["owner"] == "Alex Rivera"]
        self.assertTrue(pr and all(i["status"] in ("done", "moved") for i in pr), pr)
        early = commitments(mem(), "2026-09-09T18:00:00-07:00")
        self.assertFalse(any("Sep 15" in i["note"] for i in early))        # the future is not visible
        t = timeline(mem(), "launch date", "2026-09-12T12:00:00-07:00")
        self.assertNotIn("oct 21", [v for v, _, _ in t["trail"]])


if __name__ == "__main__":
    unittest.main()
