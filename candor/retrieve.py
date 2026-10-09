"""Retrieval: time-gated hybrid lexical search with entity/date/recency signals and diversity control.

Pipeline
  1. gate      Store.visible(as_of): nothing later than as_of, nothing deleted, edits applied.
  2. score     BM25F-lite over body + metadata (speaker, channel/title, recipients). IDF is computed over the
               VISIBLE set only, so the future cannot influence ranking.
  3. context   meeting segments borrow a fraction of their neighbours' scores; thread replies borrow from parents.
  4. signals   named-person match, dates named in the question, recency for "current" questions,
               query-term coverage (a unit matching most distinct terms beats one repeating a single term).
  5. expand    pseudo-relevance feedback: rare terms from the best hits find the rest of the story.
  6. diversify greedy selection with a per-record decay, so one long meeting cannot fill the top 10.
Steps 1-6 are deterministic. Model-shaped retrieval is layered on top of them, never instead of them:
  * enrichment (enrich.py): per-record context/keywords/dates added to the word index, visible only after their causal time;
  * query planning (qplan.py + Retriever.search_plan): the model rewrites the question into evidence needs and anchors, each
    searched by the steps above and fused by reciprocal rank with seats reserved per need.
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import textproc as tp
from . import thesaurus
from .store import Store
from .enrich import Annotations

K1, B = 1.2, 0.6
IDF_POW = 1.0
META_W = 0.6
ENR_W = 0.3        # weight of enrichment words relative to body words (0.7 cost train questions; 0.3 kept the dev gain)


@dataclass
class Hit:
    unit: object
    score: float
    parts: dict = field(default_factory=dict)
    text: str = ""            # text as of as_of (edits applied)
    edited: bool = False


class Index:
    def __init__(self, store: Store, enrich=True):
        self.store = store
        self.n = len(store.units)
        self.pos = {u.id: i for i, u in enumerate(store.units)}
        self.body_tf, self.meta_tf, self.post = [], [], defaultdict(list)
        self.blen = []
        self.chunks = {}          # codex unit id -> list of token Counters (chunk-level matching)
        for i, u in enumerate(store.units):
            b = Counter(tp.tokens(u.text))
            m = Counter(tp.tokens(self._meta_text(u)))
            self.body_tf.append(b)
            self.meta_tf.append(m)
            self.blen.append(sum(b.values()))
            for t in set(b) | set(m):
                self.post[t].append(i)
            if u.source == "codex":
                self.chunks[u.id] = [Counter(tp.tokens(c)) for c in u.meta["chunks"]]
        # enrichment: extra words per unit, usable only once its causal time has passed (see enrich.py)
        self.ann = Annotations() if enrich else Annotations("/nonexistent")
        self.enr_tf = [Counter() for _ in store.units]
        self.enr_dates = [() for _ in store.units]
        for i, u in enumerate(store.units):
            a = self.ann.by_id.get(u.id)
            if a:
                self.enr_tf[i] = Counter(tp.tokens(a["ctx"] + " " + " ".join(a["kw"])))
                self.enr_dates[i] = tuple(datetime.fromisoformat(d).date() for d in a["dates"])
                for t in self.enr_tf[i]:
                    self.post[t].append(i)
        for i, u in enumerate(store.units):          # an edit event is the same message with newer text: as findable as its target
            if u.kind == "edit" and u.target in self.pos and not self.enr_tf[i]:
                a = self.ann.by_id.get(u.target)
                if a:
                    j = self.pos[u.target]
                    self.enr_tf[i], self.enr_dates[i] = self.enr_tf[j], self.enr_dates[j]
                    self.ann.by_id[u.id] = {**a, "id": u.id, "avail_dt": max(a["avail_dt"], u.time)}
                    for t in self.enr_tf[i]:
                        self.post[t].append(i)
        self.people = self._people_index()

    # ---------------------------------------------------------------- metadata text
    def _meta_text(self, u):
        parts = [u.speaker if u.speaker_known else "", u.where, u.title]
        if u.source == "email":
            hdrs = [u.meta.get("from", "")] + u.meta.get("to", []) + u.meta.get("cc", [])
            parts += hdrs
            for h in hdrs:   # split a compound domain label (acmefreight -> acme freight) with the corpus vocabulary
                m = re.search(r"@([a-z0-9\-]+)\.", h.lower())
                if m:
                    parts.append(self._split_org(m.group(1)))
        if u.source == "calendar":
            parts += u.meta.get("attendees", [])
        if u.source == "meeting":
            parts += [self.store.meetings[u.record].get("location", "")]
        if u.source == "slack" and u.channel_id in self.store.channels:
            parts += [self.store.channels[u.channel_id]["name"]]
        return " ".join(p for p in parts if p)

    def _split_org(self, label):
        if not hasattr(self, "_vocab"):
            self._vocab = set()
            for u in self.store.units:
                self._vocab.update(re.findall(r"[a-z]{3,}", (u.text + " " + u.where).lower()))
        for i in range(3, len(label) - 2):
            a, b = label[:i], label[i:]
            if a in self._vocab and b in self._vocab:
                return f"{a} {b}"
        return label

    def _people_index(self):
        """name -> set of token stems, for spotting people in questions ('Sarah Patel' vs 'Sarah Kim')."""
        people = {}
        for row in self.store.user_rows:
            if row["id"].startswith("B"):
                continue
            nm = row.get("real_name") or row.get("name")
            people[nm] = {"first": tp.stem(nm.split()[0].lower()), "full": [tp.stem(w.lower()) for w in nm.split()],
                          "email": row.get("email", "")}
        # external people appear in email headers only
        ext = {}
        for u in self.store.units:
            if u.source != "email":
                continue
            for hdr in [u.meta["from"]] + u.meta["to"] + u.meta["cc"]:
                m = re.match(r"\s*([^<]+?)\s*<([^>]+)>", hdr)
                if m and m.group(1) not in people:
                    ext[m.group(1)] = m.group(2)
        for nm, em in ext.items():
            if len(nm.split()) >= 2 and "example.com" in em:
                people[nm] = {"first": tp.stem(nm.split()[0].lower()), "full": [tp.stem(w.lower()) for w in nm.split()], "email": em}
        return people

    # ---------------------------------------------------------------- scoring
    def _visible_mask(self, as_of):
        vis = set()
        for i, u in enumerate(self.store.units):
            if u.time > as_of or u.kind == "deleted_marker" or self.store.is_deleted(u.id, as_of):
                continue
            vis.add(i)
        return vis

    def enr_ok(self, i, as_of):
        """Is unit i's annotation usable at as_of? (released by its batch time, and the message has not been edited since)"""
        a = self.ann.by_id.get(self.store.units[i].id)
        if not a or a["avail_dt"] > as_of:
            return False
        u = self.store.units[i]
        if u.kind == "edit":
            return True                      # inherited from the target, released no earlier than the edit itself
        return not any(t <= as_of for (t, _, _) in self.store.edits.get(u.id, []))

    def bm25(self, qterms, vis, as_of, weights=None, meta_bonus=0.25):
        """BM25 over the BODY only (proper length normalisation) plus a small bonus when a query term also
        appears in the metadata (speaker, title, channel, recipients). Returns ({doc: score}, {doc: terms})."""
        N = len(vis)
        avg = sum(self.blen[i] for i in vis) / max(1, N)
        scores = defaultdict(float)
        matched = defaultdict(set)
        for t in qterms:
            docs = [i for i in self.post.get(t, ()) if i in vis]
            df = len(docs)
            if not df:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5)) ** IDF_POW
            w = (weights or {}).get(t, 1.0)
            for i in docs:
                u = self.store.units[i]
                tf = self.body_tf[i].get(t, 0)
                if ENR_W and self.enr_tf[i].get(t) and self.enr_ok(i, as_of):
                    tf += ENR_W * self.enr_tf[i][t]
                L = self.blen[i]
                if u.id in self.store.edits:      # edited message: score the current text
                    txt, edited = self.store.current_text(u, as_of)
                    if edited:
                        c = Counter(tp.tokens(txt))
                        tf, L = c.get(t, 0), sum(c.values())
                s = 0.0
                if tf:
                    s = idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * L / avg))
                    matched[i].add(t)
                elif not self.meta_tf[i].get(t):
                    continue
                if self.meta_tf[i].get(t):
                    s += meta_bonus * idf * (1.0 if tf else 0.5)
                    if not tf:
                        matched[i].add(t)
                scores[i] += s * w
        return scores, matched


ANCHOR = re.compile(r"\b(?:the\s+(?:day|morning|afternoon|evening|night|week)\s+(?:i|we|of|before|after|that|when)|"
                    r"(?:when|while|whenever)\s+i(?:'m|\s+am|\s+will\s+be)?\s+(?:in|at|on|flying|fly|leave|leaving|travel|traveling|travelling|land|landing)|"
                    r"during\s+(?:my|the|our)|on\s+the\s+day\s+of|same\s+day\s+as|that\s+day|same\s+day)\b", re.I)


class Retriever:
    def __init__(self, store: Store, cfg=None, enrich=None):
        import os
        self.store = store
        if enrich is None:
            enrich = os.environ.get("CANDOR_ENRICH", "1") != "0"
        self.index = Index(store, enrich=enrich)
        self.cfg = {"nbr": 0.25, "date": 1.5, "person": 1.25, "recency": 0.0, "prf": 0.5, "prf_docs": 6, "prf_terms": 10,
                    "thes": 0.5, "rec_decay_meeting": 0.6, "rec_decay_other": 0.85, "anchor": 3.0,
                    "source": 1.5, "datew": 0.6, "entity_latest": 0.0, "entity": 2.0, "recency": 0.6, "cov": 0.5, "why": 1.0}
        self.cfg.update(cfg or {})

    # ---------------------------------------------------------------- query understanding
    def analyse(self, question, as_of):
        toks = tp.tokens(question)
        terms = list(dict.fromkeys(toks))
        qset = set(toks)
        people = []
        for nm, info in self.index.people.items():
            if all(w in qset for w in info["full"]):
                people.append((nm, 1.0))
        first_hits = defaultdict(list)
        for nm, info in self.index.people.items():
            if info["first"] in qset and not any(nm == p for p, _ in people):
                first_hits[info["first"]].append(nm)
        matched_first = {self.index.people[p]["first"] for p, _ in people}
        for f, names in first_hits.items():
            if f in matched_first:   # "Sarah Patel" was given in full; do not also boost Sarah Kim
                continue
            for nm in names:
                people.append((nm, 0.6 if len(names) > 1 else 1.0))
        return {"terms": terms, "people": people, "dates": tp.dates_in_text(question, as_of),
                "current": tp.wants_status(question), "anchor": bool(ANCHOR.search(question)),
                "sources": tp.source_cues(question), "why": bool(re.match(r"\s*why\b", question, re.I)),
                "datetok": tp.date_phrase_tokens(question)}

    # ---------------------------------------------------------------- one scoring pass
    def _pass(self, terms, weights, vis, as_of):
        st, ix = self.store, self.index
        base, matched = ix.bm25(terms, vis, as_of, weights)
        score = dict(base)
        lam = self.cfg["nbr"]
        if lam:                      # context smoothing: meeting neighbours, thread parents
            for i, sc in base.items():
                u = st.units[i]
                if u.source == "meeting":
                    for nb in st.neighbors(u, 1):
                        j = ix.pos[nb.id]
                        if j in vis:
                            score[j] = score.get(j, 0.0) + lam * sc * 0.5
                elif u.thread and u.thread in ix.pos and ix.pos[u.thread] in vis:
                    j = ix.pos[u.thread]
                    score[j] = score.get(j, 0.0) + lam * sc * 0.5
        return score, matched

    def _prf_terms(self, score, vis, exclude):
        """RM3-lite: rare, characteristic terms of the current best hits."""
        st, ix = self.store, self.index
        top = sorted(score.items(), key=lambda kv: -kv[1])
        docs = [(i, sc) for i, sc in top if ix.blen[i] >= 6][: self.cfg["prf_docs"]]
        if not docs:
            return {}
        tot = sum(sc for _, sc in docs)
        N = len(vis)
        w = defaultdict(float)
        for i, sc in docs:
            L = max(1, ix.blen[i])
            for t, c in ix.body_tf[i].items():
                if t in exclude or len(t) < 3:
                    continue
                df = sum(1 for j in ix.post[t] if j in vis)
                if df > 0.08 * N or df < 2:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                w[t] += (sc / tot) * (c / L) * idf
        best = sorted(w.items(), key=lambda kv: -kv[1])[: self.cfg["prf_terms"]]
        return dict(best)

    # ---------------------------------------------------------------- search
    def search(self, question, as_of, k=20, extra_terms=None, anchor_hint=None):
        st, ix = self.store, self.index
        vis = ix._visible_mask(as_of)
        q = self.analyse(question, as_of)
        terms = list(q["terms"])
        weights = {t: self.cfg["datew"] for t in terms if t in q["datetok"]}
        if q["why"]:
            for t in ("becaus", "caus", "reason", "issue", "problem", "found", "regression", "blocker"):
                if t not in terms:
                    terms.append(t)
                    weights[t] = self.cfg["why"] * 0.6
        for t in thesaurus.expand(terms):
            terms.append(t)
            weights[t] = self.cfg["thes"]
        for t in (extra_terms or []):
            if t not in terms:
                terms.append(t)
                weights[t] = 0.5
        score, matched = self._pass(terms, weights, vis, as_of)
        if not score:
            return [], q

        if self.cfg["prf"]:
            fb = self._prf_terms(score, vis, set(terms))
            if fb:
                mx = max(fb.values())
                s2, _ = self._pass(list(fb), {t: v / mx for t, v in fb.items()}, vis, as_of)
                m1 = max(score.values())
                m2 = max(s2.values()) if s2 else 1.0
                for i, v in s2.items():
                    score[i] = score.get(i, 0.0) + self.cfg["prf"] * m1 * (v / m2) * 0.5

        qdates = set(q["dates"])
        anchor_dates = set()
        am = ANCHOR.search(question)
        if am:                       # two-hop: "the day I fly to Denver" -> look the anchor up, take its dates (and the span between)
            clause = question[am.end():].strip(" ?.")
            span = bool(re.match(r"(?:when|while|whenever)\s+i|during", am.group(0), re.I))
            if clause:
                anchor_dates |= self.resolve_anchors([{"query": clause, "span": span}], as_of)
        anchor_dates -= qdates
        if anchor_hint and not anchor_dates:     # the question's own words are more reliable than a model-named anchor
            anchor_dates |= set(anchor_hint)

        # idf-weighted coverage of the question's own (non-expansion) terms
        own = [t for t in q["terms"] if t in ix.post]
        N = len(vis)
        idfw = {}
        for t in own:
            df = sum(1 for j in ix.post[t] if j in vis)
            if df:
                idfw[t] = math.log(1 + (N - df + 0.5) / (df + 0.5)) * (self.cfg["datew"] if t in q["datetok"] else 1.0)
        tot = sum(idfw.values()) or 1.0
        cov = {i: sum(idfw[t] for t in matched.get(i, ()) if t in idfw) / tot for i in score}

        caps = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", question)[1:]
        calendar_words = set(tp.MONTHS) | {m[:3] for m in tp.MONTHS} | {"sept"} | set(tp.WEEKDAYS) | {w[:3] for w in tp.WEEKDAYS}
        ent_terms = [e for e in (tp.stem(w.lower()) for w in caps if w.lower() not in tp.STOP and w.lower() not in calendar_words)
                     if e in ix.post and 0 < sum(1 for j in ix.post[e] if j in vis and ix.body_tf[j].get(e)) <= 0.12 * N]
        ent_terms = [e for e in ent_terms if not any(e in info["full"] for nm, _ in q["people"] for info in [ix.people[nm]])]

        if anchor_dates:             # date-matched calendar items are candidates even with no shared words
            floor = 0.35 * max(score.values())
            for i in vis:
                u = st.units[i]
                if u.source == "calendar" and set(u.dates) & anchor_dates and u.meta.get("status") != "cancelled":
                    score[i] = max(score.get(i, 0.0), floor)

        tmax = max(st.units[i].time for i in vis)
        tmin = min(st.units[i].time for i in vis)
        span = max(1.0, (tmax - tmin).total_seconds())
        out = []
        for i, sc in score.items():
            u = st.units[i]
            parts = {"bm25": sc}
            m = 1.0
            for nm, w in q["people"]:
                info = ix.people[nm]
                hit = (u.speaker == nm) or (info["email"] and info["email"] in " ".join(
                    [u.meta.get("from", "")] + u.meta.get("to", []) + u.meta.get("cc", []) + u.meta.get("attendees", [])
                    + [u.where, u.text[:400]]))
                if hit:
                    m *= 1 + (self.cfg["person"] - 1) * w
                    parts["person"] = 1
            udates = {u.date, *u.dates}
            if ix.enr_dates[i] and ix.enr_ok(i, as_of):
                udates |= set(ix.enr_dates[i])
            if qdates and udates & qdates:
                m *= self.cfg["date"]
                parts["date"] = 1
            if anchor_dates and (udates | {d for d in tp.dates_in_text(u.text[:300], u.time, True)}) & anchor_dates \
                    and u.source in ("calendar", "email", "slack", "meeting"):
                m *= self.cfg["anchor"]
                parts["anchor"] = 1
            if ent_terms:
                if any(ix.body_tf[i].get(e) for e in ent_terms):
                    m *= self.cfg["entity"]
                    parts["entity"] = 1
                elif any(ix.meta_tf[i].get(e) for e in ent_terms):
                    m *= 1 + (self.cfg["entity"] - 1) * 0.4
            if q["sources"] and u.source in q["sources"]:
                m *= self.cfg["source"]
                parts["source"] = 1
            if self.cfg["cov"]:
                m *= 1 + self.cfg["cov"] * cov.get(i, 0.0)
            if self.cfg["recency"] and q["current"]:
                m *= 1 + self.cfg["recency"] * (u.time - tmin).total_seconds() / span
            txt, edited = st.current_text(u, as_of)
            out.append(Hit(u, sc * m, parts, txt, edited))
        if q["current"] and self.cfg["entity_latest"]:
            out = self._inject_entity_latest(question, out, vis, as_of)
        out.sort(key=lambda h: (-h.score, h.unit.time))
        return self._diversify(out, k), q

    def _inject_entity_latest(self, question, out, vis, as_of):
        """Status questions ("has Acme signed?") are answered by the latest news about the entity, which often
        shares nothing but the name with the question. Guarantee the most recent entity mentions a seat."""
        st, ix = self.store, self.index
        caps = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", question)[1:]]
        ents = [tp.stem(w.lower()) for w in caps if w.lower() not in tp.STOP]
        N = len(vis)
        ents = [e for e in ents if e in ix.post and 0 < sum(1 for j in ix.post[e] if j in vis) <= 0.05 * N]
        if not ents or not out:
            return out
        have = {h.unit.id: h for h in out}
        cand = []
        for i in vis:
            u = st.units[i]
            if u.source in ("codex",) or ix.blen[i] < 4:
                continue
            if any(ix.body_tf[i].get(e) for e in ents):
                cand.append(u)
        cand.sort(key=lambda u: u.time, reverse=True)
        floor = self.cfg["entity_latest"] * max(h.score for h in out)
        for u in cand[:4]:
            txt, edited = st.current_text(u, as_of)
            if u.id in have:
                have[u.id].score = max(have[u.id].score, floor)
            else:
                out.append(Hit(u, floor, {"entity_latest": 1}, txt, edited))
        return out

    def _key(self, h):
        u = h.unit
        return u.record if u.source in ("meeting", "chatgpt") else (u.source, u.channel_id or u.record)

    def _diversify(self, hits, k):
        pool = hits[:100]
        chosen, seen = [], Counter()
        while pool and len(chosen) < k:
            best, bi = None, -1
            for j, h in enumerate(pool):
                dec = self.cfg["rec_decay_meeting"] if h.unit.source == "meeting" else self.cfg["rec_decay_other"]
                adj = h.score * (dec ** seen[self._key(h)])
                if best is None or adj > best:
                    best, bi = adj, j
            h = pool.pop(bi)
            seen[self._key(h)] += 1
            chosen.append(h)
        return chosen


    # ---------------------------------------------------------------- model-shaped search
    def resolve_anchors(self, anchors, as_of, store_text=None):
        """Look each anchor up ("flight to Denver"), collect the explicit dates of the best hits, and fill the span between them."""
        out = set()
        for a in anchors:
            hits, _ = self.search(a["query"], as_of, k=3)
            dates = set()
            for h in hits[:2]:
                dates |= {d for d in tp.dates_in_text(h.text, h.unit.time, True) if d >= h.unit.date}
            if a.get("span") and len(dates) >= 2:   # "while I'm in Denver": a trip covers the days between its first and last date
                lo, hi = min(dates), max(dates)
                if (hi - lo).days <= 14:
                    dates |= {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
            out |= dates
        return out

    def search_plan(self, question, as_of, plan, k=30, extra_needs=None):
        """Fuse the plain search with one search per planned phrasing, then reserve a top-10 seat for every evidence need.

        The plain ranking keeps the largest weight, so a poor plan can reorder the tail but cannot lose what v1 found."""
        duration_cue = bool(re.search(r"\b(while|during|throughout|whole|entire|trip|stay|week|days|there)\b", question, re.I))
        anchors = self.resolve_anchors([{**a, "span": bool(a.get("span")) and duration_cue} for a in plan.get("anchors", [])], as_of) if plan else set()
        base, q = self.search(question, as_of, k=60)          # the plain search: never touched by the plan
        GENERIC = {"calendar", "schedule", "event", "meeting", "agenda", "today", "day", "date", "time", "thing", "stuff", "info", "update"}
        needs = []
        for n in list((plan or {}).get("needs", [])) + list(extra_needs or []):
            qs = [qq for qq in n["queries"] if len([t for t in tp.tokens(qq) if t not in GENERIC]) >= 2]
            if qs:
                needs.append({**n, "queries": qs})
        if not needs and not anchors:
            return base[:k], q
        RRF = 60
        n_lists = sum(len(n["queries"]) for n in needs) + (1 if anchors else 0)
        base_w = max(1.5, 0.5 * n_lists)         # the plain search always carries at least half of the fused mass
        fused, hit_of = defaultdict(float), {}
        for r, h in enumerate(base):
            fused[h.unit.id] += base_w / (RRF + r + 1)
            hit_of[h.unit.id] = h
        if anchors:                  # model-named time anchors only add one more ranked list; they cannot rewrite the plain one
            anch, _ = self.search(question, as_of, k=60, anchor_hint=anchors)
            for r, h in enumerate(anch):
                fused[h.unit.id] += 1.0 / (RRF + r + 1)
                hit_of.setdefault(h.unit.id, h)
        need_best = []
        for n in needs:
            nf = defaultdict(float)
            for qq in n["queries"]:
                hs, _ = self.search(qq, as_of, k=40, anchor_hint=anchors)
                for r, h in enumerate(hs):
                    nf[h.unit.id] += 1.0 / (RRF + r + 1)
                    hit_of.setdefault(h.unit.id, h)
            for uid, v in nf.items():
                fused[uid] += v
            ranked = sorted(nf, key=lambda i: -nf[i])
            need_best.append(ranked[:2])
        order = sorted(fused, key=lambda i: (-fused[i], hit_of[i].unit.time))
        top = order[:10]
        base_top = [h.unit.id for h in base[:3]]
        # a plan may reorder the tail but never evict what the plain search ranked best: its top 3 always keep a seat
        protected = set(base_top) | set(order[:2])
        for b in base_top:
            if b not in top:
                victims = [i for i in reversed(top) if i not in protected]
                if victims:
                    top[top.index(victims[0])] = b
        # seat reservation: every need's best record must sit inside the scored window
        for best in need_best:
            if best and not any(b in top for b in best):
                want = best[0]
                victims = [i for i in reversed(top) if i not in protected and not any(i in b[:1] for b in need_best)]
                if victims:
                    top[top.index(victims[0])] = want
                    protected.add(want)
        final = top + [i for i in order if i not in top]
        return [hit_of[i] for i in final[:k]], q
