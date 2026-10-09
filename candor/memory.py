"""Memory facade: question + as_of -> answer, sources, ranked retrieved ids, abstained.

v2 flow (every model step is optional and falls back to the deterministic result):
  1. plan     qplan.make_plan: evidence needs, rewrites and time anchors (1 call)
  2. search   Retriever.search_plan: plain search fused with one search per planned phrasing, seats reserved per need
  3. answer   one call over ~28 candidates: facts first, then answer, sources and the evidence it found most relevant
  4. recover  if it abstains but names what is missing, search once more with those phrasings and ask once more
  5. validate every id re-checked against the time gate; secrets and planted instructions stripped
"""
import os

from . import answer as A, qplan, safety
from .config import PLAN_DEFAULT
from .llm import LLM, LLMError
from .retrieve import Retriever
from .store import Store, parse_dt


class Memory:
    def __init__(self, data_dir=None, llm=None, use_llm=True, llm_rerank=True, cfg=None, use_plan=None, enrich=None):
        self.store = Store(data_dir)
        self.retriever = Retriever(self.store, cfg, enrich=enrich)
        self.llm = llm if llm is not None else (LLM() if use_llm else None)
        self.llm_rerank = llm_rerank
        self.use_plan = (os.environ.get("CANDOR_PLAN", "1" if PLAN_DEFAULT else "0") != "0") if use_plan is None else use_plan
        self.errors = 0

    def _search(self, question, as_of, plan, k=30, extra_needs=None):
        if plan or extra_needs:
            return self.retriever.search_plan(question, as_of, plan or {}, k=k, extra_needs=extra_needs)
        return self.retriever.search(question, as_of, k=k)

    def ask(self, question, as_of):
        as_of = parse_dt(as_of) if isinstance(as_of, str) else as_of
        llm_ok = self.llm is not None and getattr(self.llm, "available", False)
        plan = qplan.make_plan(self.llm, self.store, question, as_of) if (llm_ok and self.use_plan) else None
        hits, q = self._search(question, as_of, plan, k=30)
        rank_ids = [h.unit.id for h in hits]
        focus = sorted(q["dates"])
        meta = {"llm": False, "error": None, "plan": bool(plan), "second_pass": False}
        out = None
        if llm_ok:
            try:
                d, allowed = A.ask_llm(self.llm, self.store, question, as_of, hits, focus)
                ans, sources, relevant, abstain = A.validate(d, allowed, self.store, as_of)
                meta["llm"] = True
                if abstain and d.get("missing_queries"):
                    mq = [str(x)[:80] for x in d["missing_queries"][:3] if isinstance(x, str)]
                    if mq:
                        hits2, _ = self._search(question, as_of, plan, k=40, extra_needs=[{"label": "missing", "queries": mq}])
                        d2, allowed2 = A.ask_llm(self.llm, self.store, question, as_of, hits2, focus, second_pass=True)
                        a2, s2, r2, ab2 = A.validate(d2, allowed2, self.store, as_of)
                        meta["second_pass"] = True
                        if not ab2:
                            ans, sources, relevant, abstain, hits = a2, s2, r2, ab2, hits2
                            rank_ids = [h.unit.id for h in hits2]
                order = list(rank_ids)
                if self.llm_rerank and relevant:
                    order = list(dict.fromkeys(relevant[:6] + rank_ids))
                out = (ans, sources or ([] if abstain else order[:3]), order, abstain)
            except LLMError as e:
                meta["error"] = str(e)
                self.errors += 1
        if out is None:      # deterministic fallback
            if not hits or A.unanswerable(self.retriever, question, as_of, hits):
                out = ("I don't have that in memory.", [], rank_ids, True)
            else:
                out = (safety.sanitize_output(A.extractive(hits)), rank_ids[:3], rank_ids, False)
        ans, sources, order, abstain = out
        return {"answer": ans, "sources": sources, "retrieved": order[:20], "abstained": abstain, "_meta": meta}
