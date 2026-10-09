# Candor: a memory that knows *when* it is

**Author: Bishal Chandra Debnath**

Candor answers questions about one person's work life (meetings, Slack, email, dictation, calendar, Codex, ChatGPT) **as of any
moment**, and turns plain-English commands into planned actions (dry run by default). Python standard library only (3.9+), nothing to install.

```
python run.py memory  --questions Q.jsonl  --out A.jsonl     # memory interface
python run.py actions --commands  C.jsonl  --out P.jsonl     # actions interface (dry run)
python run.py eval --judge gemini [--dev]                     # every scorer on the train sets (+ my dev set) + leak audit
```

## Quickstart

```bash
git clone https://github.com/Pranoy71/candor-time-aware-memory && cd candor-time-aware-memory
cp .env.example .env          # Windows: copy .env.example .env, then paste a Gemini key (free: aistudio.google.com/apikey)
python run.py eval --judge gemini --dev   # model calls are spaced for free-tier limits; results are cached, so re-runs are quick
python run.py report                      # writes the scores into docs/RESULTS.md and the results block below
python -m unittest discover -s tests      # 39 tests, no key needed
```

* **No key?** Everything still runs (deterministic retrieval, extractive answers, rule-based actions), and the committed enrichment
  (`enrich/annotations.jsonl`, see below) still helps retrieval. Only the model-written answers and model-first actions need a key.
* **Model:** one model, `gemini-3.5-flash-lite` (override with `GEMINI_MODEL`). Replies are cached in `.cache/llm_cache.jsonl`, judge verdicts in
  `.cache/judge_cache.jsonl`, so an interrupted run resumes instead of restarting.
* **Windows:** only the official `score_actions.py` needs `pip install tzdata`. My code falls back to a built-in US Pacific rule set.
* **Ablations:** `--plan` (switch on the experimental model query plan, off by default), `--no-enrich` (ignore the annotations), `--no-rerank` (ignore the model's evidence picks),
  `CANDOR_ACTIONS=rules` (rule-based actions only). They are how I check that each model step earns its place.

## v2: what the organisers told me, and what I changed

I received hidden-set results for v1. They were honest and they were fair:

| v1 on the hidden set | With model | Without model |
|---|---|---|
| Retrieval | 79% | 75% |
| Answers | 72% | 40% |
| Actions (bonus) | 8/13 | 8/13 |

My train numbers (100% retrieval, 98% answers) had been tuned on 27 questions, and the hidden set showed the real gap. The feedback was that the
model only wrote the final answer, so it never shaped what was searched or ranked, and that my action rules never handed anything to it.
Here is each point and what I did about it.

| Feedback | What I did |
|---|---|
| The model never shapes search or ranking. | **Model before search** (`candor/qplan.py`): one cached call turns the question into separate *evidence needs* with phrasings in the words the records would use, searched and fused with the plain ranking. **I built it, measured it, and it is OFF by default** (`PLAN_DEFAULT` in `candor/config.py`, `--plan` to enable): replaying the plans Gemini actually wrote, it lowered retrieval (details and the bugs I found in the results section). The model does shape retrieval in two measured ways that stay on: index-time enrichment, and the evidence it picks at answer time. |
| ...or sees the candidates before the final list. | **Model after search**: one call over about 28 candidates writes the facts it finds, then the answer, the sources and the evidence it found most relevant, which I promote to the front of `retrieved`. |
| "I don't know" although the answer is in the data. | The prompt now says to abstain only when the *topic is absent*, and to answer with what the evidence shows (and say what it does not) when it is partial. If it still abstains, it names what is missing, I search once more with those phrasings, and it gets one second look before abstaining. `python run.py diagnose` labels every miss as `SEARCH_MISS`, `RANK_MISS`, `ANSWER_MISS`, `WRONG_ABSTAIN` or `SHOULD_ABSTAIN`, so I can see which step failed instead of guessing. |
| Records cannot be found by their own words. | **Index-time enrichment** (`candor/enrich.py`): one pass of about 130 calls writes, for each record, a context line (who is speaking about what, with pronouns and "the candidate" resolved), search keywords and synonyms, and absolute dates. It is committed in `enrich/annotations.jsonl`, so it helps with no key too. **Causality is enforced, not assumed:** the model sees only the records of one batch, an annotation becomes visible at the timestamp of the *latest* record in its batch, it is ignored for a message edited by the query time, and tests assert all of this with a scripted model. |
| Date lookups only handle some phrasings of travel. | Anchors are looked up in memory (the flight's dates, and the days between them for "while I'm in Denver"), whatever the phrasing, and applied to every source. The model can also name anchors in its plan. |
| Commitments are pattern-based. | `python run.py commitments` now has the model read *windows* of real speech (restarts, fillers) and extract promises with owner, beneficiary, due date and firmness (`firm`, `tentative`, `conditional`). An extracted commitment must cite records it was shown, or it is dropped. Status (open, moved, done, cancelled, overdue) is still computed only from records visible at `as_of`. The pattern matcher stays as a fallback and as a second opinion. |
| Actions: people from memory; "open X" is not always an app. | Actions are **model-first**: the model proposes using names, and my code resolves every name, id, email, event and time. The directory now includes people who exist only in memory (calendar attendees, addresses quoted in messages), for example Jordan Ellis, who never appears in an email header. "Open X" is classified as app, document or event; a document or event is opened in the app that holds it, found through memory. Questions and destructive commands never reach the model: code sends them to `memory.ask` and `confirm`. If the model gives a time that disagrees with an explicit phrase in the command, the phrase wins. |
| A comment mentions an embedding layer that is not in the repo. | Removed. I add no embeddings: the task fixes one model, and it is a text model. The query-expansion half of that comment is now real (`qplan.py`). |

## What makes this problem hard, and what I did about each part

| Trap in the data | My answer |
|---|---|
| **Time.** The same question has different right answers on Sep 9, Sep 12 and Sep 18 (launch: Sep 30, then Oct 14, then Oct 21). | One choke point, `Store.visible(as_of)`. Retrieval, enrichment, the answer writer, the ledger and the assistant only see what it releases. |
| **Edits and deletes** arrive as separate events. | Edits are units of their own and also replace the target's text from the edit time on. Deleted records vanish from the deletion time. Deletion markers are never returned. |
| **A pasted key** that was later deleted. | Redacted by pattern when the data loads, so it cannot be indexed, retrieved, cited, annotated or quoted. Checked again on every output. |
| **Planted instructions** (an email hides text aimed at AI summarisers). | Neutralised at load and again before any prompt or output; every prompt says evidence is data. A test asserts the planted text never reaches a prompt. |
| **Two Sarahs**, unidentified speakers. | A people index with full names; ambiguous first names produce `clarify`; the answer prompt attributes only with confidence. |
| **Who said it.** "Dana said John said…", corrections, promised versus done. | Every unit keeps speaker, certainty and delivery state; the prompt has explicit rules for second-hand reports, disagreement, corrections and "promised is not done". |
| **Recurring events.** | RRULEs are expanded (DAILY/WEEKLY, BYDAY, INTERVAL, UNTIL, COUNT, EXDATE) so every event has its real dates. |

## Architecture

```
 data/ ─► Store ─visible(as_of)─► Retriever ◄── enrich/annotations.jsonl (visible only after its batch time)
           │                        ▲  │
           │          qplan.make_plan  └─ search_plan: plain + per-need searches, RRF fusion, seats per need
           │                        │
           ▼                        ▼
   Planner (model-first, code-grounded)      Answer writer: ~28 candidates → facts → answer → validate → 2nd look if it abstains
   Ledger / timeline / assistant reuse the same gate
```

**Retrieval core** (`candor/retrieve.py`, deterministic): BM25 over the body (IDF over the *visible* set only), a metadata bonus, neighbour smoothing for
meeting segments and thread replies, person matching, dates named in the question, source cues ("dictate" points at dictations), a named-entity
prior, idf-weighted coverage, recency for status questions, a small work-language thesaurus, pseudo-relevance feedback, a two-hop temporal
anchor, causal weighting for "why", and a per-record decay so one long meeting cannot fill the top 10.

**Answer writer** (`candor/answer.py`) re-validates every cited id against the time gate and strips secrets and planted instructions.
If the model is missing or fails, a deterministic path answers extractively and abstains when a distinctive term appears nowhere in memory.

## Results

I report three kinds of numbers and I keep them apart.

**1. Train sets, tuned.** I tuned v1 on these, so they overstate. Latest scorer output (generated by `python run.py report` from `out/`):

<!-- RESULTS:START -->
| Metric | Result |
|---|---|
| Retrieval, train (main score) | 100.0% (95% CI 100%-100%) over 25 scored questions; MRR 0.96; complete in top 5 100.0%, top 10 100.0%; harm-only clean 100.0% |
| Answers, train (judge: openai:gemini-3.5-flash-lite) | strict 100.0% (95% CI 100%-100%), lenient 100.0% (95% CI 100%-100%), hard failures 0; source recall 1.0, precision 0.9487 |
| Actions, train | pass rate 100.0%, argument accuracy 100.0% (n=12) |
<!-- RESULTS:END -->

**2. My dev set, not tuned on** (`evals/memory_dev.jsonl`, 28 questions over storylines the train set never asked about: the hiring loop, Pinecrest,
RoadSignal, the board pre-read, Denver, ChatGPT drafts, time-travel traps and abstentions; I hand-checked every gold id). In deterministic mode,
before any model stage, retrieval scores **84% (21/25) against 100% on train**, which is the same kind of gap the hidden set showed. All four
misses are long-meeting questions: two are `SEARCH_MISS` (the debrief segments never name the candidate, and "plan after the debrief" shares no
words with "second system design round"), two are `RANK_MISS` (found at rank 17 and 19). That is precisely what enrichment and the query plan target.
Two things I tried on it and rejected, because they traded one train question for another: raising the weight of rare terms, and giving
meeting segments more of their neighbours' scores.

**3. What the model stages did.** Two kinds of evidence, kept apart. Both are small samples, so treat one question as noise.

*With the live model (train, n=27; strict = the official judge `gemini-3.5-flash-lite`):*

| Run | Retrieval | Answers |
|---|---|---|
| v1, the version the organisers scored | 100% | 98.2% |
| v2 first run: plan + enrichment (weight 0.7) | 92.0% | 92.6% |
| v2: enrichment only (`--no-plan`) | 96.0% | 100% |
| v2: plan only (`--no-enrich`) | 92.0% | 96.3% |
| v2b: plan + enrichment, after I fixed the plan bugs below | 92.0% | 92.6% |
| v2c: shipped (enrichment 0.3, plan off, weekday rule) | 100% (top-5 100%, MRR 0.96) | 100% |

On the dev set the shipped configuration scores 88.5% retrieval (n=28): 25 questions OK and 3 search misses on long meeting transcripts (DEV-07, DEV-08, DEV-28). That, not the train number, is my honest estimate.

On the dev set the failure taxonomy never showed an `ANSWER_MISS` or `WRONG_ABSTAIN`: when the right records were retrieved, the answer step used them.
The remaining dev failures are all `SEARCH_MISS` on long meeting transcripts (DEV-07, DEV-08, DEV-28). The model's evidence picks at answer time are the strongest
effect I measured: v1's deterministic order had top-5 retrieval of 68% on train (MRR 0.78); v2 with enrichment and the model's picks had 96% (MRR 0.92).

*Offline, deterministic, no model (this is what a no-key run does), and, for the plan, replaying the exact plans Gemini wrote from the cache:*

| Configuration | Train (tuned) | Dev (untuned) | Reworded |
|---|---|---|---|
| A: no enrichment, no plan (v1-style search) | 100% | 84.6% | 100% |
| **B: enrichment at 0.3, no plan (shipped)** | 92% | 88.5% | 95% |
| enrichment at 0.7 (what the first v2 run used) | 88% | 88.5% | 95% |
| C: plan, no enrichment | 88% | 84.6% | not replayable |
| D: plan + enrichment | 88% | 88.5% | not replayable |

How I read it: the plan **hurts** train and adds nothing on dev, so it is off. Enrichment is roughly a wash on the questions I tuned against (train, reworded: it moved
one acceptable record from rank 8 to rank 11 in TR-03) and gains one question on the untuned dev set (DEV-01). One dev question is not proof; I ship it
because the dev set is the only unbiased sample I have and because the annotations are good (for example "Leah Brooks recaps the interview loop for Jordan Ellis").
Honestly, I could not show that enrichment helps.

*Plan bugs I found by replaying the real plans, all fixed (with a regression test) and none of them enough to make the plan help:* the model's time anchors were
applied to the plain search itself, so an irrelevant anchor rewrote v1's own ranking (a synthetic plan dropped the flight email from rank 1 to 14); seat protection guarded
the fused top 3 instead of v1's top 3; a model-named anchor ("Product offsite") was added to the question's own correct anchor ("fly to Denver") and polluted the dates;
bare-word queries such as "calendar" and "events" flooded the fused list; and the plan could outweigh the plain search. Also: an edit event had no annotation while the
message it edits did, which pushed it below near-duplicates, so edits now inherit their target's annotation. The remaining correction: DEV-26 (the Denver dinner) was
**my gold error**; Alex did choose Tavernetta, and the model was right.

An experiment I do not report: I tried fusing enrichment as a separate ranked list, but the implementation never took effect (I caught that on re-inspection), so it
tells me nothing. I also rejected, earlier, more weight on rare terms and more weight on neighbouring segments, which each traded one train question for another.

Other checks that do not depend on a model: a leak audit over hundreds of retrievals at random `as_of` moments using the harness's own record loader as
the oracle (`python run.py audit`: no future record, no deleted record, no secret, no deletion marker), 39 unit tests (gate, edits, deletes,
recurrence, redaction, injection, time parsing, enrichment causality, plan fusion with a scripted model, action grounding, ledger), 17 action cases I wrote
(`evals/actions_extra.jsonl`) and 20 reworded retrieval questions (`evals/memory_paraphrase.jsonl`, which reuse train's gold evidence, so they are not a hold-out).

## What didn't work, and what still doesn't

* **Plain BM25 with metadata in the same field: 64%.** Meeting titles matched every segment of that meeting. Fix: score the body alone, metadata as a small bonus.
* **"Sep" in "Sep 10" was treated as an entity** and broke the dictation question; months and weekdays are now excluded.
* **My first two-hop anchor used wrong dates** ("Wednesday" resolved backwards); anchors now use explicit dates only.
* **More weight on rare terms, or on neighbouring segments**, improved one dev question and broke train ones, so I dropped both.
* **Rules alone cannot generalise on actions** (8/13 hidden, identical with and without a model), which is why v2 puts the model first.
* **The official judge crashed on Gemini's 429 rate limit.** `candor/judge_runner.py` wraps it with spacing, retries and a cache, leaving the harness untouched.
* **Still open:** the model's plan can be wrong (fusion limits the damage, it does not remove it); the thesaurus and cues are English-only; the entity prior assumes
  capitalised proper nouns; RRULE support does not cover MONTHLY/YEARLY; a calendar event is available from its `updated` time, one reading of a file that stores
  only the latest state; "open a document" opens the app that holds it, which is my reading of an ambiguous spec; commitment status is still sentence-level.

## Tools, models and cost

* **Runtime model:** `gemini-3.5-flash-lite` through the free Gemini API, for planning, answers, enrichment, actions, commitments and as the judge. No other model.
* **Code:** Python standard library only.
* **Built with:** an AI coding assistant (Claude) in a chat window, under my direction; I ran every evaluation myself and made the final decisions. The scorers in `eval_harness/` are the organisers' and are unchanged.
* **Budget on a 500-requests-a-day key:** enrichment is one-time (about 130 calls, resumable with `python run.py enrich --limit N`, committed afterwards);
  a question costs 1 call (answer), or 2 when it abstains and re-searches (3 with the experimental plan); an action costs 1; commitments cost at most 40 and only if you ask for them.
* **Money:** ₹0 for model API usage (Gemini free tier).

## Two extra features that reuse the same time gate

* **Commitments ledger and timeline** (`candor/ledger.py`): `python run.py commitments --as-of …`, `python run.py timeline "launch date"`. Asking "as of Sep 9" shows a world without Sep 15.
* **TextOS assistant** (`candor/assistant.py`): `python run.py assistant --execute`. Nothing runs without `y`; only local-safe effects happen (`.ics` files, a reminders
  file with an alarm, allow-listed app launches, a `mailto:` link). **It never sends anything itself.** Destructive requests are refused and handed back.

## Layout

```
run.py                 CLI: memory | actions | eval | report | diagnose | enrich | audit | ask | act | commitments | timeline | assistant
candor/store.py        units, time gate, edits/deletes, recurrence, redaction     candor/retrieve.py   ranking, plan fusion, anchors
candor/qplan.py        model before search                                         candor/enrich.py     index-time annotations (causal)
candor/answer.py       evidence pack, prompt, validation, extractive fallback      candor/memory.py     the question flow
candor/actions.py      model-first planner, grounding, directory, rules fallback   candor/timeparse.py  "tomorrow at 2"
candor/ledger.py       commitments + timeline          candor/assistant.py   TextOS     candor/diagnose.py   failure taxonomy
candor/llm.py          client + cache                  candor/judge_runner.py  throttled judge     candor/safety.py  redaction + injection
tests/                 unittest suite                  evals/memory_dev.jsonl  my dev set          docs/  demo script, questions for the organisers
```
