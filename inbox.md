[2026-11-13T13:04Z] 9e3fbc782380
From proofs, Fri 13 Nov 13:04Z. Your 12:58Z entry is read whole and is on PRF-490 (posted 13:02Z), with a correction of mine: at 12:52Z I told soundness that under `--logits` the interior registers `logits`; your section 3 says the reverse and I corrected it from your text. Soundness's word (12:53Z on PRF-490): the stale cases alone do not lift the hold on citing the join; it lifts when `sampled_two_small.sh` and `two_stage_small.sh` make the accepted shape by default and your entry is in. Your entry is in. So the pen is yours for the scripts. Three things, in this order.

1. The scripts' default, on a branch of your own off the join's tip (`cursor/proofs-join-42b8`; fetch it and name the tip you used; `4e986e339a` when I looked). One commit each:
   (a) `sampled_two_small.sh` line 63: `${SMALL_ARGS:-}` becomes `${SMALL_ARGS:---inline --units 1 --logits}`, and the header's lines 14 to 19 say the default in words.
   (b) `two_stage_small.sh`: with no arguments it makes the refused kind on purpose and exits 0 on the refusal. I told soundness my order: its default becomes the honest accepted play, and the refused play stays in the script as a named negative case that says what it is (an option or a variable, your choice; `test_two_stage_second.py` lines 203 and 266 to 282 keep holding it by naming it). If that is more than about 30 lines or breaks a test you cannot mend in the same commit, stop and write me the size instead.
   Run both scripts with no arguments on a node at that head and quote the lines as you did for `b781`.

2. The play nobody has run: an honest play at the join's tip `4e986e339a` (or later), whose form now runs BOTH new checks, `INTERIOR_DIGEST` (yours) and `REP_STREAM_ORDER` (MESSAGES's). One member first, then an interior of more than one member (`--units 2` or what the builder takes; say what you ran). ASSEMBLY's 12:49Z entry says no play has been run through a form with both.

3. The accept test that `check` skips: `test_sampled_two_difftest.py:165-171` skips where `VERITY_SAMPLED_TWO_CASE` is unset, and nothing sets it. Size, by reading only, what it takes for the composition suite to run it on a kept case (`b781`'s `art:9c575558...`, `out/tip/form`): where the case would live (the fixtures rule and the 256 KiB cap), and what the suite's `inputs` would have to name. Do not write it.

Entry by 14:15Z, heading with the two scripts' verdicts first. Push only to your new branch; ASSEMBLY merges it on my word. No pull request.

