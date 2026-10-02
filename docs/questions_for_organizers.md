# Questions worth emailing (diptopal@joinsettle.info)

1. **Calendar `updated` semantics.** `events.jsonl` holds only each event's latest state, stamped with `updated`. I treat an event as
   available from `updated`, so for an `as_of` before that it is hidden even if the event existed earlier in an older form.
   Is that the intended reading, or should earlier states be inferred from the invitation emails
   (for example EM-F-003 vs EM-0915-CAL-UPD for board deck prep)?
2. **Abstention questions.** When a question is unanswerable (for example Dana's salary) and the system abstains, is `retrieved`
   scored only for forbidden records? I return what retrieval fetched rather than an empty list, because the field is defined as
   "what you would hand to the answer writer". Is an empty list preferred?
3. **`as_of` range.** Will hidden questions use `as_of` values outside the train range (before Sep 8, or after Sep 18)?
   I return an honest abstention when nothing relevant exists yet; I want to be sure that is the expected behaviour.
