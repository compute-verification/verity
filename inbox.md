[2026-11-16T23:25Z] 8ad42ee92bb4
MODEL: Claude Opus 5.5 — DRAWMAP return (TOP-424 tree-wide draw map, phase 1)

Head: b1c61686617 on cursor/pouw-draw-map-1de3 (pushed), base 6f9f794db37. It holds PoUW's and vllm's files in one map (tools/move format, kept out of the tree; it goes in the notes with the hand-in). PoUW alone was db1418823fa, but it is not green on its own: PoUW's renames reach vllm's tests (`p.drawn`, `setattr(PHC, "draw_k")`, pickled `run.draw`). So the two lanes hand in together.

Counts per lane, observed: draw-family occurrences, before → after (all tracked files), and files changed.
- compute-accounting: 3798 → 2638, 67 files.
- top (benchmarks/pouw and other top-owned paths): 7810 → 6250, 74 files.
- vllm: 385 → 152, 28 files.
- proofs: 15263 → 15224, 7 files (only callers of our renamed names; proofs' own words are not taken today).
- circuits: 94 → 82, 5 files.
- kernels: 25 → 18, 1 file.
- Whole tree: 37065 → 34054. These totals include withdraw/withdrawn, which don't change.

Tests:
- BEFORE at base: PoUW group passed (r20261116-203526-a950); difftests passed (r20261116-214216-ebf6); repository passed. verity-vllm has 2 failures at base, both in test_credit_gate (r20261116-203549-85b4).
- Earlier heads failed in the PoUW group: 27 tests at the phase-1 and coin heads (r20261116-205524-c46b, r20261116-212405-dc55), and verity-vllm (r20261116-212450-f8af, 26 tests). I traced every cause and fixed it in the map, not in tests:
  - quoted attribute names in setattr;
  - keyword-argument dict keys;
  - a pinned vector key (epoch_coins_v1.json);
  - the Lean-named flag `--fresh-draws`, held until Tuesday;
  - a Lean-quote hold that was too loose;
  - proofs' own `"the run's draw session"` message;
  - `gateless.draw`;
  - test_budget's asdict field.
- Locally on b1c61686617:
  - the formerly failing files fail the same set as base, only "lake is not on the PATH", which is local;
  - ncp_v2 passes;
  - ruff matches base.
- AFTER on b1c61686617 is running: r20261116-232149-cd30 (PoUW group), r20261116-232228-24e0 (verity-vllm and repository), r20261116-232308-2f31 (difftests). The PoUW group took more than 70 min of wall time last time, so it may land close to the 01:15Z proof line.

Kept (detail in the hand-in): Lean sites and Lean-read keys (Tuesday); hashed tags; record format ids; `pouw/draw`; `power.draw`; proofs' signed-ledger `draw_k` field. The Python helper `PHC.draw_k` is renamed `sample_k`; the stored key stays. Proofs' served-record keys: `draw`, `drawn`, `public.drawn`, `profile.drawn`, `wrong_units`, `proved_units` and "no draw" are Lean-spelled, so Tuesday. `draw_counts` has no Lean reader but sits in proofs' files, so it is not taken today. `draw_k` stays (signed ledger).

Hand-in at handins/draw-map.md by 00:00Z. Flag: other lanes' old draw maps still sit in tools/move on main (draw_words_map.toml, draw_compound_map.toml, ...).

