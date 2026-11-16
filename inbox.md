[2026-11-16T18:31Z] 02289d01e8b7
MODEL: Claude Opus 5.5 — DRAWMAP first return (TOP-424 draw map, dry run)

Recount (observed, the lint's `sites()` on base 9aabe6f8dc8, our three trees): 3,450 sites in 157 files, 589 of them Lean in 28 files.
Map: `tools/move/pouw_draw_map.toml` on `cursor/pouw-draw-map-1de3`, head 7e21db9f181 (map applied, pushed). Re-applying from base rewrites 0 files (idempotent).

By kind (renamed / kept): py:use 794/47, py:def 265/3, py:docstring 305/71, py:message 213/257, py:key 50/572, py:test-name 43/0, py:flag 8/19, py:comment 8/0, md 8/28, sh:code 2/107, sh:comment 0/45, sh:flag 1/5, toml 0/10, lean 0/589. Total 1,697 renamed, 1,753 kept, in 91 files with a renamed site; plus 28 other lanes' files (users of our names: vllm, one_stage, dither, pearl_c4, flock test).

Kept, by reason: Lean, waiting for lean's pass (589); quoted token: record key, label, port or reply (586); message a Lean verifier states word for word (214); shell env names, output names and log text, where only flags are mapped (157); domain tag, coin family, Lean subcommand or power.draw (46); asdict `draws` read back by key (39); flags another lane, a README or Lean calls (34); README Lean names and difftest ports (23); SPOT's Draw objects (15); held file names drawn.py, drawn.sh, drawn_cost.sh (13); record keys given as keywords (11); tests quoting a Lean message (10); excluded paths (9); the period open's `draw_k` field (6); toml (1).

Tests (observed): BEFORE at base: group of 13 PoUW and benchmark suites all pass (r20261116-160711-9163); difftests pass (r20261116-160836-3b0e); repository passes; verity-vllm fails 2 at base (test_credit_gate: test_a_period_open_without_draw_k_is_refused, test_a_window_presented_outside_its_periods_close_or_out_of_order_is_refused; r20261116-160757-ed03). The AFTER runs on earlier heads found these causes, each fixed in the map, not in tests: a hash-pinned fixture rewritten (now excluded); members read through instances in other lanes' files (pearl_c4's `p.drawn`, dither's `RoundPipe.draw`); aliases re-exported or chained (`CO.Wd.DRAW`, a re-exported `PHC.draw_k`); SPOT's "a retry never redraws" phrase; and messages a Lean verifier states. AFTER v4 at 7e21db9f181 is running: r20261116-182716-8360 (13 suites), r20261116-182756-672d (vllm and repository), r20261116-182819-74f7 (difftests). One difftest cause is still open (a port named with its verdict, "drawn accept"), and its fix is in the next map.

Next: merge origin/main (the tests move, d4a407c0dbd), regenerate from scratch on the new base, BEFORE and AFTER there, and the final return by 00:30Z.

No blocking questions. To flag for your post (inferred, not decided): vllm wants `flip` for coin identifiers (tp_draw→tp_flip); the map uses `sample*` for identifiers and "flip" only in coin phrases. vllm would rename the settings `Settings.draws`/`draw_session` on both sides; the map keeps them as stored config keys. vllm's TSV attributes `ChallengePhase.draw` to us; it is vllm's own (commit_coins.py), and the map leaves it alone.

