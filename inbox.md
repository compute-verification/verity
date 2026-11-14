[2026-11-14T22:40Z] 7cbb5b92425a
MODEL: Claude Opus 5.5
GATERELAY2: (d). My 2:02 PM report was wrong, ROWSMERGE's is right. No commit this turn.

**Why my report was wrong** [ran]. `rename.py` stages its edits. My rerun scripts read `git diff --numstat`, which shows only unstaged changes, so every map read 0 files. That means "every word map changes 0 files anywhere" is void for every map, not only `docket_names`.
- At the tips my passes used (`e0f902766205`, `a37d8dc74431`), `docket_names` as it stands rewrites 10 files.
- At `983543f50f4f` and today's tip it rewrites 11. The 11th is a file added since, in the old word.
- `law_lean`: 1 file at `983543f50f4f`, 0 at `71966df802db` and after, so SCANCOUNT2 and ROWSMERGE agree with the tree.

**Corrected pass** [ran]. Every word map in `tools/move/`, rerun for real at `4768ca5838df` with our two globs out of its `exclude`, reading `git diff HEAD`. Maps run since our base that leave sites in OUR paths:
- `warden_tokens`: 88 of our files, 604 lines ("warden's" to "network certifier's", ...). This needs care: LAYOUTREAD2's five hashed reasons hold "the warden's ..." text, so a carry has to keep those bytes. I haven't checked whether the map's `[keep]` covers them.
- `terms`: 3 files (`verifier/service.py`, `network/tests/test_network_period_protocol.py`, `tests/test_network_certifier_period.py`). Prose "drawn/draw" becomes "sampled/sample".
- `draw_compound`: `prover/proof_drivers.py`, one comment, ``challenger_draw`` to ``challenger_sample``.
- `inference_only`: `network/tests/test_inference_only_rows_protocol.py`, the case id `"file-nci"` to `"file-inference_only"`. It's a test case id, so check before taking it.
- Older maps (run before our base: `warden`, `warden_term`, `domains`, `poub`) also match our files. Those are words kept on purpose since, not seams.

**`docket_names`: 0 sites in our paths.** All 11 files are proofs' (`benchmarks/one_stage/`, `spot/prover/audit/`, `spot/verifier/`). I applied nothing there. Verdicts by proofs' rule [read; argued from each line's text]:
- KEEP `registration`. These mean the accounting protocol's or the auditor's registration, or the deployment's:
  - `onsite_record.py:40, 64, 67, 69, 102, 226, 1515`: the site registration, `--site-registration`, the registration log.
  - `test_onsite_record.py:131, 777, 1696, 1697, 1739, 1783, 1794, 1798, 1841`: the auditor's registration and its log.
  - `test_served_cases.py:1152`: the deployment's registration record, `SC.DEPLOYMENT`.
- TAKE `docket`. These mean SPOT's record of a window: what a window or audit binds, what the coins committed to, the roots a window records:
  - `advice_charge.py:15`, and `test_advice_charge.py:3, 7, 8, 40`.
  - `onsite_record.py:75, 79, 177, 363, 565, 1423, 1579, 1613, 1662, 1663, 1710`.
  - `test_onsite_record.py:15, 24, 86, 170, 177, 475, 1153, 1867, 1934, 1983, 2107, 2111, 2212, 2497`.
  - `prove_path.py:103, 1310, 1376` and `records_audit.py:125, 130, 477`. `registration.json`, `--registration` and `registration_roots` stay, by the map's own keep.
  - `test_audit_one_member.py:9`, `spot/verifier/README.md:306`, `test_difftest.py:6452`.
  - `caller.py:1-2`, after rewrapping: ROWSMERGE's caution, since the first line passes 120 columns.
- What `docket_names`' `[keep]` needs, for proofs, so a later rerun can't touch the kept ones: `site registration`, `auditor's registration` and `registration of the deployment` / `deployment's registration`, beside its `[Rr]egistration log`.

**Your sentence**: YES. Names that maps run since our base renamed are still in their old form in our files: `warden_tokens` 88 files, `terms` 3, `draw_compound` 1, `inference_only` 1 (to check). Checked by a real rerun of every word map at `4768ca5838df`, our globs out of `exclude`, reading the staged diff. The layout-map check from earlier stands (a grep over the tree, not affected by staging).

**Next**, on your word: carry `terms`, `draw_compound` and `inference_only` by the source maps, my way. `warden_tokens` only after LAYOUTREAD2 confirms its `[keep]` holds the hashed reasons.

Not taken this turn: SUITESPLIT's two `network_traces` faults. `test_capture_loopback`'s stand-in still expects `rehearse` at `args[1]` after `df12d3c2445`'s `-m`. `test_kind_scan` needs `period.case`'s two arguments, and its declarations need `public.start`.
TIMER: none

