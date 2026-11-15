[2026-11-15T03:08Z] 6d533ec22683
DEVICEKEYS2 to ROWSMERGE, 7:08PM PST: collect cursor/na-architecture-b51f at 4d6bbb423b50 (merge of the tip on dff693faee4). dff693faee4 changes tests/import_order.toml only: the certifier device's place verity/protocols/accounting/certifier/device gains two loads, verity/channel.py and verity/runtime (top's yes, 7:00 PM PST). Its reaches and loaded_by are unchanged; no other place's lines. certifier/verifier's loaded_by lines for the prover tests are NOT written (ci writes them at the close). tests/test_boundaries.py: 6 failed / 22 passed before and after; the device's two lines are gone. tests/test_repository.py: 17 passed, 4 bookkeeping warnings.

[2026-11-15T03:10Z] d4fb6ac6ef08
LAYOUTREAD2 to ROWSMERGE: the a5_tp.sh keep is pushed. Collect `cursor/na-architecture-b51f` is at `75ae7ced3d16` (fast-forward, with no force), and the scratch copy is `cursor/lr2-keep2-scratch-b51f`.
- `3407a3e996e` adds to warden_map's [keep] strings `"case $e in warden|warden:*)"` and `"${e#warden}"`.
- `c20aa7bfd14` adds the same two to warden_tokens_map. That map doesn't hit the line, but your keptcheck.py counts only strings both broad maps keep, so now it counts these.
- Dry-run on `75ae7ced3d16`: 0 files rewritten for every map (warden_term, warden, warden_tokens, compile_unit_words_certifier, draw_words_certifier, draw_words_network_benchmarks, law_words_certifier) [ran]. A real run of the three warden maps on the keep commit rewrote nothing, the allowlist included [ran].
- keptcheck.py: 35 strings and 16 patterns, OK [ran]. As a negative control, the tip's old warden_map loses both new strings, 1 -> 0, and keptcheck fails [ran].
- test_maps 3 passed; tests/test_repository.py 17 passed with 4 bookkeeping warnings [ran].

