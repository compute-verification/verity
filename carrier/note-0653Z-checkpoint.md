# CARRIER, @NOW@ Sat 14 Nov, checkpoint: #9636's carry is past the terms move's base with one settlement of mine; the second job crosses the 26 moves left

Clock read with `date -u` at @NOW@.

- [run] **Scratch branch** `cursor/proofs-carrier-9636-carry-9257` at `b443ddb01f1faa26a55d6adc2e28a2a04b5520d0`, pushed. It is #9636's head, the first four moves crossed by `restack.py` (run `r20261114-055226-0840`, vy-nebius-cpu-5, done, rc 0), then one merge.
- [run] **The one settlement so far**, `b443ddb01f1f`: `restack.py` stopped at the terms move's base `92b5727aca40` in four files. `test_stop_audit.py` is #9458's and I took `a275f10724a9`, the carried line's own settlement, as it is. The other three are #9636's `certified` beside the family change that base holds: `run`'s signature in `one_stage/verifier.py`, its two calls in `service.py`, one docstring in `Spot/Pieces.lean`. Four hunks, each both sides. None asks what the check should do.
- [derived] My 06:00Z note put those three files' hunks at #9555. Half of each is the family change, which is older and in the terms move's base. #9555's half (`key`, `main`) comes with the merge of #9458's head.
- [run] **Running**: `r20261114-061855-bc20` on vy-nebius-cpu-5, from `b443ddb01f1f`: 25 moves up to Dither's (`--main e4a8682800d6`), then the clear audit's (`--main ac8b3b51e490`), the order `main` took them in. It waited 16 minutes for a slot and started at 06:35Z. Done about 07:15Z [guess].
- [run] #9458's own branch is now at `1b4b9acc84f3c4bdb47cc35e00905a3f0ddc50e9`, the hold line. I merge that head.
- [run] `main` is at `512644d14fbc`: six landings since 06:00Z, no move. The carried #9458 merges clean with it. `AGENTS.md` changed in one sentence (this repository is public whole); reread.
- [run] #9513, #9810, #9717 and #9549 are not on `main` at 06:37Z. No `ChargeOnce.lean` pull request yet.
- [guess] The ready note about 10:30Z still stands; the slot wait cost 16 minutes.

**Not established**

- That the 26 moves cross without another stop.
- Everything after the moves: the merge of #9458's head and of `main`, the locks, the check runs.
