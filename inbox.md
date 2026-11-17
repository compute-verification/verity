[2026-11-17T01:06Z] cd6febe518c8
MODEL: Claude Opus 5.5 — DRAWMAP return (6:00 PM PST order)

Plainly: the draw map cannot be proven green by 6:00 PM PST, so it waits for Tuesday 9:00 AM with the Lean maps. The mended head is 1515f0fbb62 on cursor/pouw-draw-map-1de3, pushed. Its runs were launched at 01:00Z and will not finish before 02:00Z: the fresh adapter suite alone took 6036 s of test time on 3 workers in 24e0.

(1) vllm's 16 failures, each decided. One is renamed on both sides; the rest are kept on both sides. A line for each is in handins/draw-map.md.
- Both sides renamed: the fake gate's `made["session"].draw` (rt4 ×3), which is `_Session`'s attribute. SPOT's `].draw` hold now excludes that site.
- Kept on both sides:
  - recorded texts (commit_rules claims, served_check scope): pinned;
  - `VERITY_DRAWS_POLL_S`: a setting's environment name;
  - `draws=2`: the stored pouw setting `draws`;
  - `T.draw_first`: proofs' name;
  - "the window's draw session's signed coins": the message continues with a Lean-stated piece;
  - "draw: the window's draw coin is malformed": the label "draw" is Lean-read;
  - "its open commits draw_k": the signed ledger field;
  - "no fresh draw session": pouw_h1_gpu's reserve name.
- The three keep-list breaks the suite doesn't reach, also kept: `"$HELP" draw`, `OUT/draws`, and the "draws = derives" test name.

(2) My own runs on b1c61686617, observed:
- r20261116-232149-cd30, the PoUW group: failed. 11 of 13 suites passed. pearlplus_lean has 16 failures, from the Lean-read work key `over_cap_draws`. served_zk has 2: the challenger's select replies ERROR instead of SAMPLED.
- r20261116-232308-2f31, the difftests: failed. 6 in the window difftest, from the Lean-read `unit_draw`.
- r20261116-232228-24e0, verity-vllm: failed, with exactly vllm's 16 plus main's two credit-gate reds. repository passed.

Both Lean-key causes came from one over-broad exception I added at 23:00Z (dict keys after "is None else {"). It is narrowed to its one site in 1515f0fbb62. served_zk's 2 failures are still open: they appear on every after-head since 49bd47e4e93 and the test skips on the VM, so I have no cause yet.

Why the hand-in went out before 24e0 was read: it was due at 00:00Z, and I published it at 23:27Z with the runs marked as running. That was the wrong call. The note now says so.

(3) Runs on 1515f0fbb62 (launched, not finished):
- r20261117-005921-0f29: vllm's own script verbatim, from their run c2fe (root tests, vllm lint, `suites.py --fresh verity-vllm`);
- r20261117-010001-d5b9: the PoUW group;
- r20261117-010043-e248: the difftests.

(4) handins/draw-map.toml and draw-map.md are updated and synced. They carry the head, counts, run ids, a line for each of the 16 and for my own failures, and the not-proven status.

Counts on 1515f0fbb62, draw-family occurrences before → after, and files changed:
- compute-accounting: 3798 → 2641, 67 files;
- top: 7810 → 6261, 73 files;
- vllm: 385 → 161, 26 files;
- proofs: 15263 → 15224, 7 files (callers only);
- circuits: 94 → 82, 5 files;
- kernels: 25 → 18, 1 file.

I'll label the three runs with their results when they end, for Tuesday's window.

