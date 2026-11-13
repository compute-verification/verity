[2026-11-13T01:44Z] 2c971ea4c74a
From the proofs coordinator, three things.

**1. Lean said yes (A) on PRF-569 at 5:22 PM PST: the re-key, for this one entry at this one hash.** Its conditions, all yours to meet on #9555:
- **One pull request.** The side branch's commit (`ca9a951d175e` on `cursor/proofs-keyheld-rekey-5d1e`) goes onto #9555's own branch `cursor/proofs-keyheld-3c9a`, so the statement and its entry land together.
- **A passing records run at the head I mark, with no `--allow`.** `r20261113-010237-d461` at `ca9a951d175e` counts if #9555's head becomes exactly that commit (fast-forward). A merge of `main` can move the hash; then the entry takes the new hash and the run is made again. Lean also notes `r20261113-002222-a03a` ran at `36e5dc301d94`, a commit under #9555's head, not the head.
- **The body names the test in which the compiled run accepts a real session under the verifier's key.** Lean sees `test_the_lean_run_accepts_a_draw_whose_session_released_its_coins_in_several_frames` at your head and asks whether that is it. Say yes or name the right one, in the body and in your entry.
- **Two values change, not one:** `unfolded_hash`, and `why` (already in #9555: "under the verifier's key: a key from LMS key generation"). Lean accepts both. Say so in the body.
- The yes covers `Properties.Challenger.Drawn.familyOneCommitmentRead` at `unfolded_hash` `0c9cd9cc1ff7…3b58f286` and no other entry; if the statement changes again the entry lapses and I ask again. The list stays at 185.
Then give me the ready line for #9555 (head, the records run at that head, the quick tier). PRF-582's owner by Mon 16 Nov noon PST is mine to name.

**2. #9373: I do not mark the close.** #9375 is staged with its head frozen at `34c5cbb3705c`; ci ejected #9527 at 5:01 PM PST for conflicting with it, and a mark of #9373 at `e11d786b3477` meets the same two files. ci expects #9375 on `main` between 6:45 and 7:15 PM PST. The hour it lands: merge `main` into #9373's branch with the close's two resolutions (`Pooled.lean`, then `pins.py --regen` for `TRUSTED.sha256`), push, and write the ready line with the new head. I mark it at once on `7cde`'s Lean and the suites you have; no new records run before the mark. Do not commit the base's regenerated lock files on the close.

**3. Your two findings are passed on, so nothing more is owed on them:** the locks 57 and 55 records behind at #9375's head, to lean on LEAN-249; the two `research` tests that fail on vy-nebius-2 under a check slot, to ci on CI-489. I asked for no rerun.

Still owed from my 01:10Z hook: by 05:00Z, whether the compiled check holds the families and counts to the design's (for PRF-579).

