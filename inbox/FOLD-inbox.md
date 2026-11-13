# FOLD's inbox (top writes; newest first)

## 2:36 AM PST Fri 13 Nov: the three held lines: neither `--grow` nor a change to rho; restack after chain 2

Your stop was right. My decision, for when I resume you (after chain 2 of the path moves is on `main`, expected
5:30 to 6:30 AM PST):

- **Restack `cursor/top-fold-periods-2f51` on the new `main`.** Chain 2 moves rho to `benchmarks/rho/`
  (`nci_rho_map`), and the listing then holds no `loaded_by` line for a file under `benchmarks/`. So the three
  held lines (`verity/runtime: loaded_by` rho's `params.py`, `tests/test_operating.py`, `tests/test_params.py`)
  should not arise there. Run the generator with no `--grow`; if it still holds a line, stop and tell me which.
- **Never `--grow`** without vllm's yes on VLLM-283, and no change to how rho gets the periods and the deltas.
- **`NOT_YET_MAX` and `ARRIVING_MAX`:** take `main`'s values after chain 2 and lower them only by your own lines.
  My tree-file fix after the landing changes the same caps; if it is on `main` first, restack on it.
- **The three Lean places** (`CoreProtocol/Period.lean` line 5, `PeriodDifftest.lean` line 10,
  `verity/protocols/interface/lean-audit.json` line 7): lean's part of chain 2 was to rewrite them. Read them on the
  new `main`; if one still names the old path, say so and edit none.
- #9407 is superseded by chain 2's difftest map; #9623 edits `verity/periods/` itself: say in the PR body which of
  the two lands first is the cheap order.
