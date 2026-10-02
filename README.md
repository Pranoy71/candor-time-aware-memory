# Candor: a memory that knows *when* it is

**Author: Bishal Chandra Debnath**

Candor answers questions about one person's work life (meetings, Slack, email, dictation, calendar, Codex, ChatGPT) **as of any
moment**, and turns plain-English commands into planned actions (dry run by default). It uses only the Python standard library
(3.9+), so there is nothing to install.

```
python run.py memory  --questions Q.jsonl  --out A.jsonl     # memory interface
python run.py actions --commands  C.jsonl  --out P.jsonl     # actions interface (dry run)
python run.py eval                                           # every scorer on the train sets + leak audit
```

## Quickstart

```bash
git clone <this repo> && cd candor
cp .env.example .env        # Windows: copy .env.example .env, then paste a Gemini key (free: aistudio.google.com/apikey)
python run.py eval --judge gemini     # about 3 min for answers + about 6 min for the judge
python -m unittest discover -s tests  # 23 tests, no key needed
```

* **No key?** Everything still runs: deterministic retrieval, extractive answers, rule-based actions. Only answer quality drops.
* **Model:** I use one model, `gemini-3.5-flash-lite` (override with `GEMINI_MODEL`). Every reply is cached in `.cache/llm_cache.jsonl`,
  and the judge's verdicts in `.cache/judge_cache.jsonl`, so an interrupted run resumes instead of starting over.
* **Windows:** only the official `score_actions.py` needs `pip install tzdata` (it uses `zoneinfo`). My own code does not; it falls back
  to a built-in US Pacific rule set that I spot-checked against `zoneinfo` in summer, winter and just after both DST changes.
* **Try it:** `python run.py ask "Did John agree to cut dark mode?"`, `python run.py assistant`, `python run.py commitments`,
  `python run.py timeline "launch date" --as-of 2026-09-12T12:00:00-07:00`

## What makes this problem hard, and what I did about each part

| Trap in the data | My answer |
|---|---|
| **Time.** The same question has different right answers on Sep 9, Sep 12 and Sep 18 (launch: Sep 30, then Oct 14, then Oct 21). | One choke point, `Store.visible(as_of)`. Retrieval, the answer writer, the ledger and the assistant only see what it releases. I never summarise at ingest time, so the future cannot leak into the past. |
| **Edits and deletes** arrive as separate events (60/64 became 61/64; two messages were deleted). | Edits are units of their own and also replace the target's text from the edit time on. Deleted records vanish from the deletion time. Deletion markers are never returned. |
| **A pasted key** that was later deleted. | Redacted by pattern when the data loads, so it can never be indexed, retrieved, cited or quoted. Checked again on every output. |
| **Planted instructions** (one email hides text aimed at AI summarisers). | Neutralised at load and again before any prompt or output, and the prompt says evidence is data. A test asserts the planted text never reaches the LLM prompt. |
| **Two Sarahs**, unidentified or low-confidence speakers. | A people index with full names. Actions ask "which one?" when a first name is ambiguous, and the answer prompt says to attribute only with confidence. |
| **Who said it.** "Dana said John said…" is not John's own word; corrections; promised versus done. | Every unit keeps its speaker, certainty and delivery state (a dictation that was *sent* versus *discarded*). The prompt has explicit rules for second-hand reports, disagreement, corrections and "promised is not done". |
| **Recurring events.** "What's on my calendar the day I fly to Denver?" includes a weekly 1:1 and a daily standup. | I expand RRULEs (DAILY/WEEKLY, BYDAY, INTERVAL, UNTIL, COUNT, EXDATE) so every event has its real dates. |

## Architecture

```
 data/ ──► Store ──visible(as_of)──► Retriever ──top 20──► Answer writer ──► validate ──► {answer, sources, retrieved, abstained}
            │ units: meeting segments, slack messages/edits, emails,     │ 1 LLM call per question (cached)   │ ids re-checked against the gate
            │ dictations, calendar events, codex sessions, chatgpt msgs  │ fallback: extractive text + abstain │ secrets and instructions stripped
            │ + edit/delete events, recurrence expansion, redaction      └ returns its best evidence → promoted in `retrieved`
            └─► Planner (rules → LLM fallback → shared resolver) ─► actions       Ledger, timeline and assistant reuse the same gate
```

**Unit granularity.** The citable unit is the most specific id the brief allows: a meeting *segment*, a Slack message or edit, a ChatGPT
*message*, an email, a dictation, a calendar event, or a Codex session (matched chunk by chunk but cited as the session).

**Retrieval** (`candor/retrieve.py`, deterministic, no key needed). BM25 over the body, with IDF computed over the *visible* set only. On top:
a small bonus for metadata matches (speaker, title, recipients; an email domain like `acmefreight` is split into "acme freight");
neighbour smoothing for meeting segments and thread replies; person matching (a full name beats a first name); dates named in the
question; source cues ("dictate" points at dictations, "calendar" at events); a named-entity prior (a rare capitalised word in the question
outranks units that lack it); idf-weighted coverage; recency for status questions ("has X…?"); a small work-language thesaurus;
pseudo-relevance feedback (rare terms from the best hits pull in the rest of the story); a two-hop temporal anchor ("the day I fly to
Denver" finds the flight, takes its dates, and pulls calendar items on them); extra weight on causal words for "why" questions; and a
per-record decay in the final selection so one long meeting cannot fill the top 10.

**Answer writer** (`candor/answer.py`). One call per question: the top 16 records as numbered, time-stamped, speaker-attributed evidence,
plus rules for current versus history, attribution, disagreement, commitments, abstention and evidence-as-data. It returns the answer, the
ids it relied on, and the evidence it found most relevant. I promote that evidence to the front of `retrieved` and never drop anything.
Every cited id is re-validated (released, not deleted, not invented). If the LLM is missing or fails, a deterministic path answers
extractively and abstains when a distinctive question term appears nowhere in visible memory.

**Actions** (`candor/actions.py`). An LLM is good at understanding a sentence but cannot know that Sarah Kim is `U03SARAHK`, that "board
deck prep" is `CAL-BOARDPREP`, or what "tomorrow at 2" means today. So plain code owns ids, emails, event lookup, time arithmetic and the
safety rules. Rules handle the common commands with no LLM call at all; the LLM is only asked when the rules do not recognise a command,
and its output goes through the same resolver (hallucinated ids are dropped, and a test covers that). Two candidates means `clarify`;
delete or cancel means `confirm` and never acts; a question means `memory.ask`; an unknown command means `clarify` rather than a guess.

## Results (train sets, from `python run.py eval --judge gemini`)

| Metric | Result | How I read it |
|---|---|---|
| Retrieval (main score) | **27/27 (100%)**, top-5 exact 96%, MRR 0.96, 0 forbidden records | I tuned this on the train questions, so I expect the hidden set to score lower. |
| Answers (judge `gemini-3.5-flash-lite`) | **98.2% strict**, 26 correct, 1 partial, 0 hard failures; cited sources recall 0.97, precision 0.98 | The judge is the same model family as my answer writer, so it may be lenient. The one partial (MEM-TR-10) dropped Marcus's specific Q4 / $120k figures from a "disagreement" answer. |
| Answers without any LLM | 59% strict | Extractive text only; this is the fallback, not the real path. |
| Effect of LLM evidence promotion | top-5 retrieval went from 68% (deterministic order) to 96%, MRR from about 0.77 to 0.96 | Measured on the same train questions; it only reorders, never removes. |
| Actions, train | **12/12**, arguments 100% | Fully rule-based; no LLM call was needed. |
| Actions, 17 extra cases I wrote (`evals/actions_extra.jsonl`) | 17/17 | I wrote them after the rules, so they are not a hold-out; they cover other verbs and phrasings. |
| Retrieval, 20 reworded questions (`evals/memory_paraphrase.jsonl`) | 20/20 | Not a true hold-out either: I reworded the questions but they reuse train's gold evidence. It did expose one vocabulary gap ("95th percentile" vs "p95"), which I fixed. |
| Leak audit (`python run.py audit`) | 702 retrievals at random `as_of` moments (297 inside `eval`), **0 violations** | The oracle is the harness's own record loader, not my store: no future records, no deleted ones, no secrets, no deletion markers. |
| Tests | 23 pass | Gate, edits, deletes, recurrence, redaction, injection, time parsing, actions, CLI, ledger, and LLM validation with a scripted LLM. |

How retrieval improved while I built it: 64% (plain BM25), 76% (body-only scoring plus a metadata bonus), 84% (thesaurus, feedback, org
aliasing), 88%, 92% (anchor hop, entity prior), 96% (fixed months being mistaken for entities), 100% (causal weighting for "why").

**Cost and speed.** At most one LLM call per question (27 calls, about 170 s with the spacing I use for free-tier limits) and one per
command the rules cannot parse. Google changes free-tier quotas often, so the client spaces calls, honours the server's retry hint,
and stops cleanly on a daily-quota error.

## Two extra features that reuse the same time gate

* **Commitments ledger and timeline** (`candor/ledger.py`). `python run.py commitments --as-of …` finds promises, follows each through later
  evidence (moved, done, cancelled, overdue) and shows the trail. `timeline "launch date"` shows how a fact changed (Sep 30, Oct 14, Oct 21)
  and who said it. It is built on `visible(as_of)`, so asking "as of Sep 9" shows a world without Sep 15. It is a conservative heuristic view over
  evidence, not an oracle.
* **TextOS assistant** (`candor/assistant.py`). `python run.py assistant --execute`: ask or command, `/asof` to time-travel, `/why` for sources.
  Nothing runs without `y`, and only local-safe effects happen: `.ics` files for events (updates reuse the event UID), a reminders file with an
  alarm, allow-listed app launches, a `mailto:` link for email and an outbox entry for Slack. **It never sends anything itself.** Destructive
  requests are refused and handed back to the user.

## Limits and what I would do next

* No embeddings. Lexical plus structure was cheap and testable; fusing dense retrieval by reciprocal rank is the obvious next step.
* The thesaurus and the causal/status cues are hand-written and English-only, and the entity prior assumes proper nouns are capitalised.
* A calendar event is available from its `updated` time, which is one reading of a file that stores only each event's latest state
  (see `docs/questions_for_organizers.md`).
* RRULE support covers the forms in this data, not MONTHLY or YEARLY rules.
* The ledger's completion detection works sentence by sentence and can miss or over-read.
* Temperature is 0 and replies are cached, so repeated prompts give identical answers; a first call to the model can still vary.

## Layout

```
run.py              CLI (memory | actions | eval | audit | ask | act | commitments | timeline | assistant)
candor/store.py     units, time gate, edits/deletes, recurrence, redaction     candor/retrieve.py   ranking pipeline
candor/answer.py    evidence pack, prompt, validation, extractive fallback     candor/memory.py     question to answer
candor/actions.py   planner, resolver, LLM fallback      candor/timeparse.py   "tomorrow at 2"      candor/llm.py   client + cache
candor/ledger.py    commitments + timeline               candor/assistant.py   TextOS               candor/safety.py redaction + injection
candor/judge_runner.py  throttled, cached wrapper around the official judge
tests/              unittest suite     evals/*_extra|paraphrase.jsonl   my own cases     docs/   demo script, questions for the organisers
```
