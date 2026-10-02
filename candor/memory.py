"""Memory facade: question + as_of -> answer, sources, ranked retrieved ids, abstained."""
from datetime import datetime

from . import answer as A, safety
from .llm import LLM, LLMError
from .retrieve import Retriever
from .store import Store, parse_dt


class Memory:
    def __init__(self, data_dir=None, llm=None, use_llm=True, llm_rerank=True, cfg=None):
        self.store = Store(data_dir)
        self.retriever = Retriever(self.store, cfg)
        self.llm = llm if llm is not None else (LLM() if use_llm else None)
        self.llm_rerank = llm_rerank
        self.errors = 0

    def ask(self, question, as_of):
        as_of = parse_dt(as_of) if isinstance(as_of, str) else as_of
        hits, q = self.retriever.search(question, as_of, k=20)
        focus = sorted(q["dates"])
        rank_ids = [h.unit.id for h in hits]
        used_llm, err = False, None
        out = None
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                d, allowed = A.ask_llm(self.llm, self.store, question, as_of, hits, focus + self._anchor_dates(question, as_of))
                ans, sources, relevant, abstain = A.validate(d, allowed, self.store, as_of)
                used_llm = True
                order = list(rank_ids)
                if self.llm_rerank and relevant:
                    order = list(dict.fromkeys(relevant + rank_ids))
                out = (ans, sources or ([] if abstain else order[:3]), order, abstain)
            except LLMError as e:
                err = str(e)
                self.errors += 1
        if out is None:      # deterministic fallback
            if not hits or A.unanswerable(self.retriever, question, as_of, hits):
                out = ("I don't have that in memory.", [], rank_ids, True)
            else:
                ans = safety.sanitize_output(A.extractive(hits))
                out = (ans, rank_ids[:3], rank_ids, False)
        ans, sources, order, abstain = out
        return {"answer": ans, "sources": sources, "retrieved": order[:20], "abstained": abstain,
                "_meta": {"llm": used_llm, "error": err}}

    def _anchor_dates(self, question, as_of):
        return []
