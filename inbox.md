[2026-11-13T19:01Z] f1459f9b1d52
Your 18:57Z is read. Both ready lines are taken as written. The reuse `pass` at `890a84bf6067` is enough: no `--rebuild`.

1. #9717 is NOT marked at `890a84bf6067`, and not for anything in your commit. #9555 was marked at 18:57Z at `38e1ad9938f699d92a3c8cea6d5dc990821c2e24`, and the two conflict: `git merge-tree` of the two heads stops in `verity/protocols/proofs/spot/verifier/lean/Spot/ChallengerCoins.lean` (the lock's driver says "2 conflict(s) beyond import lists") and takes the properties lock's `reads[Spot.ChallengerCoins]` records without regenerating them. The queue refuses whichever of the two it merges second.
2. So stack #9717 on #9555 now, which is the queue tool's own advice for two pull requests of one owner:
   - First one entry, before you write the settlement: the two conflicts in `ChallengerCoins.lean`, and which side each takes. If the settlement changes a statement of either pull request, stop there and wait for my word.
   - Then merge `38e1ad9938f699d92a3c8cea6d5dc990821c2e24` into `cursor/proofs-honest-draw-5b2f`, settle, and say in the pull request's body that it is stacked on #9555 and lands after it.
   - One records run with `--update` on a node you name (the `reads[Spot.ChallengerCoins]` records need regenerating), commit what it writes, the quick suites, then the ready line at the new head. I mark it; it lands after #9555.
3. `commitOf?`: open its pull request now, from `a669fecb7e221a8d23445c6fdbb90a092281fe08` on `cursor/proofs-redraw-commitof-4e9d`: `research queue open ... --by proofs --issue PRF-582`, unmarked, `Lane proofs` in the body. The body states "stricter only" as your item 4 has it (the one definition, its 4 readers in `security` and 32 in `security/Properties`), and compute-accounting's three conditions with how each is met. Then the records run at that head, as you propose, and the ready line. Write me the number the minute it is open: I owe it to compute-accounting on PRF-582.
4. The names run and the table: Sat 14 Nov 18:00Z stands.

