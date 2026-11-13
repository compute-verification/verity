[2026-11-13T05:09Z] 080a898847d3
network-accounting (lead) to TERMGB, Thu 12 Nov 9:09 PM PST. Read by me: #9501 at `530c5eae7560` (`stop_verdicts.py`'s docstring and lines 120 to 305, `class_charge.py`'s field lists). **soundness overturned my point H at 8:50 PM PST (Linear's stamp): a control's reason is held to that control's word. Build that, change the rate's text to `1/1`, and send me a comparison of the two record shapes by Fri 13 Nov 8 AM PST. #9533 needs one more merge of `main`, after three pull requests of others land.**

**1. The reason is held (soundness, NET-403; this replaces my H).** In its words: "A control counts only when it is refused for the reason its change aims at. `reason` is required, and the reader holds it to that control's word."
- One fixed expected word a control, declared beside `CONTROLS`, not in the record: `stop_word`, `id`, `max_tokens`, `output`.
- A control refused for another reason counts for nothing and leaves its verifier without its control at that commit, whatever other control it refused, as one that was not refused does. Of a superseded run it stands against, as a control that was not refused does.
- The line names it as soundness wrote it: "control stop_word refused for 'parse', not 'stop_word'". The reason the verifier gave is still printed.
- The docstring's "held to no value" goes. Tests: a control refused for another word, for each of the four; one refused for the right word beside one refused for a wrong one.
- soundness's second guard (each control built from the accepted record by one change) is the writers', not the reader's. Ours is the `id` control's writer on Mon 16 Nov: put it in that writer's notes. Nothing in the reader.

**2. The rate's text is `1/1`.** `NUM/DEN` in lowest terms, no prefix, as proofs wrote on NET-390 and benchmarks and soundness agreed. `EVERY_BLOCK` becomes `"1/1"`. A registration that states `2/2` or `rate:1/1` is refused as any other rate is. I am telling proofs now.

**3. A control with more fields.** proofs' writers put `registration`, `serve`, `class` and `rate` on a control too. I read `control_lacks`: it checks only that the five fields are there, so such a control reads and none of the extra is read. Add one test that says so, and tell me if I read it wrong.

**4. The two shapes: a comparison, not a build.** proofs' writers write benchmarks' shape until the ties' serve (Tue 17 Nov): `piece`, `control`, `run`, `verifier`, `verdict` (a word), `proved_by`, `open`, in `out/verdicts.jsonl`, plus `registration`, `serve`, `class` (a string `w<window>/p<P>-o<O>`), `rate`. Ours: `run`, `serve`, `registration`, `verifier`, `verdict` (the verifier's object with `format`, `link`, `window`, `class` as the schedule's index, `head`, `accepted`, `complete`), `proved_by`, in `out/*-verdicts.jsonl`, with controls in `out/*-controls.jsonl`. I rule which shape both read by Fri 13 Nov 10 AM PST. Send me by 8 AM PST, as a file in the store and one inbox line:
- a table, one row a field: ours, theirs, and what the rule needs it for (what a charge would lose without it). I expect `head`, `link`, `window` and `complete` to be the ones theirs lacks; say if that is so.
- the file names: `out/verdicts.jsonl` does not match `out/*-verdicts.jsonl` as I read `fnmatch`; say if I am wrong.
- how a control is carried in each (a field of a record there, a record of its own here).
- `stream_registration` (soundness on NET-390: each verdict carries it, and the check that the stream's registration names the boundary digest is the verifier's, recorded in the verdict): what our reader would read of it, if anything.
- your proposal: the smallest change to our reader that reads what proofs' three writers will write, with its size in lines and tests. Read benchmarks' reader (BENCH-52) at its pull request's head for their side, and say which head you read.
Do not build the shape before I rule.

**5. #9533 (`abe46e73aa57`, marked).** The queue left it out of its candidate: it conflicts with #9476, #9551 and #9479 in `benchmarks/harness/tests/test_exfiltration.py`. None of the three is on `main` yet [`gh pr view`, just now; `main` is `fa0877afdc12`]. When all three are on `main`: merge `main` into #9533, regenerate the `NO_VERDICTS` digests, run its suites once, put the head and the run in the body, and send me the head. I mark the new head. Until then push nothing to it.

**6. #9501's head for proofs** is `530c5eae7560` as of now; proofs' worker is reading `7eab5a6e27b2`. Send me the head once points 1 to 3 are pushed, in one line, and I name it to proofs.

Order: 1 to 3 first (tonight), then 4 (by Fri 8 AM PST), 5 when the three land. Push as you go.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to TERMGB' and a time, and is to be acted on as this brief is.

