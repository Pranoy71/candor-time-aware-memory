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
        m = Memory(llm=MockLLM(fake), use_plan=True)
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

    def test_llm_actions_grounded(self):
        """The model proposes by name; code resolves. A person who does not exist is never acted on."""
        fake = {"actions": [{"type": "slack.send_message", "to": ["Nobody Real"], "text": "hi"}]}
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: json.dumps(fake)))
        a = p.plan("ping nobody real about stuff", "2026-09-18T09:00:00-07:00")
        self.assertEqual([x["type"] for x in a], ["clarify"])
        ok = {"actions": [{"type": "app.open", "app": "Figma", "open_kind": "app"}]}
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: json.dumps(ok)))
        self.assertEqual(p.plan("gimme figma", "2026-09-18T09:00:00-07:00")[0]["args"]["app"], "Figma")

    def test_model_dates_are_cross_checked(self):
        """The model says 3pm tomorrow; the command says 'tomorrow at 2'. Code trusts the explicit phrase."""
        bad = {"actions": [{"type": "calendar.create_event", "title": "Sync with Ben", "start": "2026-09-17T15:00:00-07:00",
                            "end": "2026-09-17T15:30:00-07:00", "attendees": ["Ben Carter"]}]}
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: json.dumps(bad)))
        a = p.plan("Book 30 minutes with Ben tomorrow at 2 about the NRR fix", "2026-09-16T12:00:00-07:00")
        self.assertEqual(a[0]["args"]["start"], "2026-09-17T14:00:00-07:00")
        self.assertIn("ben@brightline.example.com", a[0]["args"]["attendees"])

    def test_ambiguous_name_and_destructive_never_reach_the_model(self):
        calls = []
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: (calls.append(1), "{}")[1]))
        self.assertEqual(p.plan("Delete all my emails from Marcus", "2026-09-18T09:00:00-07:00")[0]["type"], "confirm")
        self.assertEqual(p.plan("What's our launch date again?", "2026-09-18T09:00:00-07:00")[0]["type"], "memory.ask")
        self.assertEqual(calls, [])
        amb = {"actions": [{"type": "slack.send_message", "to": ["Sarah"], "text": "hi"}]}
        p = Planner(mem().store, memory=mem(), llm=MockLLM(lambda s, u: json.dumps(amb)))
        self.assertEqual(p.plan("Message Sarah about the pricing proposal", "2026-09-18T09:00:00-07:00")[0]["type"], "clarify")

    def test_person_known_only_from_memory(self):
        d = Planner(mem().store, memory=mem()).dir
        who = {p["name"]: p for p in d.people}
        self.assertIn("Jordan Ellis", who)
        self.assertTrue(who["Jordan Ellis"]["email"])
        self.assertNotIn("All", who)


class TestEnrichment(unittest.TestCase):
    """Index-time enrichment must never let the future into the past."""

    @staticmethod
    def marker(uid):
        import hashlib
        return "zq" + hashlib.md5(uid.encode()).hexdigest()[:10]

    @classmethod
    def setUpClass(cls):
        from candor import enrich
        cls.enrich = enrich
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "ann.jsonl"
        cls.prompts = []

        def fake(system, user):
            cls.prompts.append(user)
            ids = re.findall(r"^\[([^\]]+)\]", user, re.M)
            return json.dumps({"items": [{"id": i, "ctx": f"context zebrafoxtrot {i}", "kw": ["zebrafoxtrot", cls.marker(i)], "dates": ["2026-09-23"]}
                                         for i in ids]})
        cls.llm = MockLLM(fake)
        cls.report = enrich.build(mem().store, cls.llm, limit=40, path=cls.path, progress=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_annotation_visible_only_after_its_whole_batch_exists(self):
        st = mem().store
        rows = [json.loads(l) for l in open(self.path, encoding="utf-8")]
        self.assertTrue(rows)
        for r in rows:
            self.assertGreaterEqual(parse_dt(r["avail"]), st.by_id[r["id"]].time)

    def test_model_never_sees_a_record_newer_than_the_annotation_time(self):
        st = mem().store
        avail = {json.loads(l)["id"]: parse_dt(json.loads(l)["avail"]) for l in open(self.path, encoding="utf-8")}
        for prompt in self.prompts:
            ids = re.findall(r"^\[([^\]]+)\]", prompt, re.M)
            stamp = max(avail[i] for i in ids if i in avail)
            self.assertTrue(all(st.by_id[i].time <= stamp for i in ids))

    def test_unknown_ids_dropped_and_secrets_redacted(self):
        batch = self.enrich.make_batches(mem().store)[0]
        out = self.enrich.parse_items({"items": [{"id": "NOT-SHOWN", "ctx": "x", "kw": [], "dates": []},
                                                 {"id": batch[0].id, "ctx": "key sk-brightline-abcdef123456", "kw": ["a"], "dates": ["bad", "2026-09-10"]}]}, batch)
        self.assertEqual(list(out), [batch[0].id])
        self.assertNotIn("sk-brightline", out[batch[0].id]["ctx"])
        self.assertEqual(out[batch[0].id]["dates"], ["2026-09-10"])

    def test_retrieval_uses_annotation_only_after_it_is_released(self):
        from datetime import timedelta
        os.environ["CANDOR_ANN"] = str(self.path)
        try:
            m = Memory(use_llm=False, enrich=True)
            rows = [json.loads(l) for l in open(self.path, encoding="utf-8")]
            late = max(rows, key=lambda r: r["avail"])
            u = m.store.by_id[late["id"]]
            after = parse_dt(late["avail"]) + timedelta(seconds=1)
            before = parse_dt(late["avail"]) - timedelta(seconds=1)
            hits_after, _ = m.retriever.search(self.marker(late["id"]), after, k=20)
            hits_before, _ = m.retriever.search(self.marker(late["id"]), before, k=20)
            self.assertTrue(any(h.unit.id == late["id"] for h in hits_after))
            self.assertFalse(any(h.unit.id == late["id"] for h in hits_before), "annotation used before its release time")
            ids_before = {h.unit.id for h in hits_before}
            self.assertTrue(all(m.store.by_id[i].time <= before for i in ids_before))
        finally:
            os.environ.pop("CANDOR_ANN", None)

    def test_enrichment_off_by_switch(self):
        os.environ["CANDOR_ANN"] = str(self.path)
        try:
            m = Memory(use_llm=False, enrich=False)
            hits, _ = m.retriever.search("zebrafoxtrot", parse_dt("2026-09-18T18:00:00-07:00"), k=5)
            self.assertEqual(hits, [])
        finally:
            os.environ.pop("CANDOR_ANN", None)


class TestPlanFlow(unittest.TestCase):
    """The model-shaped path with a scripted model: planning, fusion, second pass, validation."""

    def _answer(self, abstain, missing=None, sources=("EM-F-036",)):
        return json.dumps({"facts": [], "answer": "I don't have that in memory." if abstain else "Oct 21.", "sources": [] if abstain else list(sources),
                           "relevant": [] if abstain else list(sources), "abstain": abstain, "missing_queries": missing or []})

    def test_plan_then_answer_then_second_pass(self):
        calls = []

        def fake(system, user):
            calls.append(system[:20])
            if "plan the search" in system:
                return json.dumps({"type": "fact", "needs": [{"label": "launch", "queries": ["launch date moved", "go live date"]}],
                                   "anchors": [], "entities": []})
            n = sum(1 for c in calls if c.startswith("You are the memory"))
            return self._answer(abstain=(n == 1), missing=["launch date october"])
        m = Memory(llm=MockLLM(fake), use_plan=True)
        r = m.ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")
        self.assertEqual(len(calls), 3)               # plan, answer (abstains), second pass
        self.assertFalse(r["abstained"])
        self.assertTrue(r["_meta"]["plan"] and r["_meta"]["second_pass"])

    def test_true_abstention_survives_second_pass(self):
        def fake(system, user):
            if "plan the search" in system:
                return json.dumps({"needs": [{"label": "x", "queries": ["salary dana"]}], "anchors": []})
            return self._answer(abstain=True, missing=["dana salary"])
        r = Memory(llm=MockLLM(fake), use_plan=True).ask("What is Dana's salary?", "2026-09-18T18:00:00-07:00")
        self.assertTrue(r["abstained"])
        self.assertEqual(r["sources"], [])

    def test_bad_plan_cannot_lose_what_plain_search_found(self):
        plain = Memory(use_llm=False).ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")["retrieved"][:3]

        def fake(system, user):
            if "plan the search" in system:
                return json.dumps({"needs": [{"label": "junk", "queries": ["banana smoothie recipe"]}], "anchors": []})
            return self._answer(abstain=False)
        got = Memory(llm=MockLLM(fake), use_plan=True).ask("When is Route Planner v2 launching?", "2026-09-18T18:00:00-07:00")["retrieved"]
        self.assertTrue(set(plain) <= set(got[:10]))

    def test_adversarial_plan_cannot_evict_plain_top3(self):
        for q, when in (("What's on my calendar the day I fly to Denver?", "2026-09-18T18:00:00-07:00"),
                        ("How many regression cases were passing on Sep 16?", "2026-09-16T14:00:00-07:00")):
            m = Memory(use_llm=False, enrich=False)
            as_of = parse_dt(when)
            plain = [h.unit.id for h in m.retriever.search(q, as_of, k=20)[0][:3]]
            wide = {"needs": [{"label": str(n), "queries": [f"calendar events meetings agenda {n}", f"schedule day {n} plan", f"lunch dinner {n}"]} for n in range(4)],
                    "anchors": [{"query": "flight to Denver", "span": True}, {"query": "regression run", "span": True}]}
            got = [h.unit.id for h in m.retriever.search_plan(q, as_of, wide, k=20)[0][:10]]
            self.assertTrue(set(plain) <= set(got), (q, plain, got))

    def test_garbage_plan_is_ignored(self):
        from candor import qplan
        self.assertIsNone(qplan.clean({"needs": "nope"}))
        self.assertIsNone(qplan.clean(["a"]))
        self.assertEqual(len(qplan.clean({"needs": [{"queries": ["a", "b", "c", "d"]}] * 9})["needs"]), 4)


class TestLedgerModelAndDiagnose(unittest.TestCase):
    def test_model_commitments_must_cite_shown_records(self):
        from candor.ledger import commitments

        def fake(system, user):
            ids = re.findall(r"^\[([^\]]+)\]", user, re.M)
            return json.dumps({"items": [{"owner": "Alex Rivera", "to": "Ben Carter", "what": "try to review the churn fix", "due": "2026-09-20",
                                          "firm": "tentative", "ids": [ids[0], "FAKE-1"]},
                                         {"owner": "Ghost", "what": "do a thing", "ids": ["FAKE-2"]}]})
        items = commitments(mem(), "2026-09-18T18:00:00-07:00", MockLLM(fake))
        self.assertFalse(any(i["owner"] == "Ghost" for i in items))
        self.assertTrue(any(i["firm"] == "tentative" for i in items))
        self.assertTrue(all("FAKE-1" not in i["evidence"] for i in items))

    def test_diagnose_separates_search_from_answer(self):
        from candor.diagnose import diagnose
        gold = [{"id": "G1", "question": "When is Route Planner v2 launching?", "as_of": "2026-09-18T18:00:00-07:00", "answerable": True,
                 "needed": [["SL-F-0159", "DCT-F-031"]], "key_terms": [["oct"]], "category": "x"},
                {"id": "G2", "question": "q", "as_of": "2026-09-18T18:00:00-07:00", "answerable": True, "needed": [["NOT-A-RECORD"]], "category": "x"},
                {"id": "G3", "question": "What is Dana's salary?", "as_of": "2026-09-18T18:00:00-07:00", "answerable": False, "needed": [], "category": "x"}]
        ans = {"G1": {"retrieved": ["SL-F-0159"], "answer": "Nobody knows", "abstained": False},
               "G2": {"retrieved": ["EM-F-001"], "answer": "x", "abstained": False},
               "G3": {"retrieved": [], "answer": "I made it up", "abstained": False}}
        got = {r["id"]: r["class"] for r in diagnose(mem(), gold, ans)}
        self.assertEqual(got, {"G1": "ANSWER_MISS", "G2": "SEARCH_MISS", "G3": "SHOULD_ABSTAIN"})

    def test_results_block_is_generated_from_scorer_files(self):
        from candor.evalrun import build_report
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "actions_report.json").write_text(json.dumps({"summary": {"n": 5, "pass_rate": 0.8, "arg_accuracy": 0.9}}), encoding="utf-8")
            block = build_report(d)
            self.assertIn("pass rate 80.0%", block)
            self.assertNotIn("Retrieval", block)


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
