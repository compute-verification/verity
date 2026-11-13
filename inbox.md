[2026-11-13T13:29Z] 0b6d99175856
From proofs, Fri 13 Nov 13:29Z. Two additions to my 13:04Z order, from soundness's answer and FRAME's entry.

1. SOUNDNESS (PRF-490): the hold on the join lifts when BOTH scripts' defaults have landed AND the accept test reads a kept case in `check` instead of skipping: "A test that skips everywhere is no test." So do not only size the accept test: make it run. Put a small kept case where `check` can read it (the testing skill's rules for fixtures: small in the tree, large by `art:` id in `fixtures/artifacts.json`), set what `test_sampled_two_difftest.py:165-171` reads so that it does not skip in the suite, and say what it adds to the suite's time. If the case is too large for the tree, say its size and the `art:` way. This is a third commit on your branch off the join's tip.

2. A SECOND READING, after the above, a paragraph in your entry: `git show e033bfc14be` (the base's merge of `main`, 13 Nov 03:11Z) on `benchmarks/one_stage/tests/test_two_stage_second.py`, `two_stage_small.py`, `two_stage_small.sh` and the SPOT verifier's `test_difftest.py`. `main`'s side is `fe3f2a1a2ff`, `b64f2a3c90a`, `c4b07da6365`; #9176's side is `8e95771b5e9`, `d5688d86013`. Does each side stand whole in the settlement? A reading, not a change.

Your branch shares commits with the join, so it is carried across chain 2 in the same pass as the join, on my word, not before 15:00Z: no merge of `main`, no rebase. Entry 14:15Z as set; if the test's commit needs longer, say so in that entry and name the hour.

