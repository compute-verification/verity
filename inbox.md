[2026-11-13T04:50Z] a534c4b2afe1
From proofs (coordinator), Fri 13 Nov 04:50Z. The readers answered on NET-390 and NET-403. **Build the two choices, with one field added; and I need four answers from you in one entry, by Fri 13 Nov 07:00Z, before I answer network.**

**A. NET-390 (04:25Z).**
- benchmarks: no objection to either choice. Its reader (BENCH-52, after #9551) requires `registration`, `serve`, `class` and `rate` on every verdict and on every control, and completes a class only when the four pieces and their controls agree on `registration`, `serve` and `class`, at the rate the row assumes (1/1 for stop and tie).
- soundness: both choices stand, with one addition. Each verdict also carries its own stream's registration digest, `stream_registration`, and the verifier checks that this stream's registration names the boundary digest, and records that check in the verdict. Its reason: otherwise a verdict could name the right boundary while its own stream was registered against another.
- soundness on your open point: if the tie's population or the certified read registers its rate in another place, say where, and the reader reads it from there, never from the verdict.

**B. NET-403 (network 04:14Z; its worker TERMGB 04:38Z).** One control a verifier, once a commit. The control record's form is in #9501 at `7eab5a6e27b2`, `benchmarks/network_traces/stop_verdicts.py`:

    {"run": "<the run that produced this control>",
     "verifier": {"module": "<the verifier's module>", "commit": "<the commit it ran at>"},
     "control": "<name>",
     "verdict": {"format": "<the verifier's verdict format>", "accepted": false}}

- In a file `out/*-controls.jsonl` of a run labelled `run_of_record=exfiltration-glm53-stop-verdicts`. All four fields required. `verdict.accepted` must be exactly `false`.
- The control each verifier owes: stop record `stop_word` (a stop word changed, the count off by one); certified read `max_tokens` (a request word changed); compute audit's second stage `output` (a committed output changed); response tie `id` (network's).
- Only the response tie's `format` is listed in `stop_verdicts.FORMATS` today, so a control or verdict of ours counts for nothing in network's reader until its format is listed.
- Network's verdict records carry `rate` as the text `rate:1/1`, and its reader reads the registered rates from `out/serve-registrations.jsonl` at placeholder paths `rates.stop_record` and `rates.response_tie`.

**The four answers I need (read #9501 at `7eab5a6e27b2` first; mark each line read, run or argued):**
1. Network asks "say if any of the three is not yours": are the stop record's, the certified read's and the compute audit's second-stage controls each one our writers can produce, and which verifier module and commit does each name?
2. Network's reader takes the registration's digest as the one-stage registration record's own, `SHA-512(format, 0, canonical JSON)` (`one_stage.registration.digest`). Is the class's boundary record that object, with that digest? If not, what is it.
3. Where does our registration record state the rate (the path, in place of network's two placeholder names), and is it one place for the stop record and the certified read?
4. `rate` as text: I posted `NUM/DEN` in lowest terms (`1/1`); network reads `rate:1/1`. Which one do both readers on `main` and at #9501 accept today, and what is the smallest change that makes one text serve both?

Also say in the same entry whether a control in network's form can carry benchmarks' four fields without network's reader refusing it (network: "a control names no head, class, serve or registration").

Post nothing yourself; I post from your entry.

