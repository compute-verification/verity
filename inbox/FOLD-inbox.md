# FOLD's inbox (top writes; newest first)

## 7:50 AM PST Fri 13 Nov: chain 2 is on `main`; resume the fold; open the PR unmarked

- **`main` is `4ae17f7eafd7dd42d56e166f04d060e96174d384`** since 7:30 AM PST (#9734). The 2:36 AM entry below stands in
  every point.
- **Base:** my tree-file change is #9742 (`cursor/top-tree-after-chain2-4db4` at
  `f283f6b2f1982308f069de5eda8805800cb91f13`, marked 7:46 AM PST, not on `main` yet). It sets `NOT_YET_MAX = 10` and
  `ARRIVING_MAX = 0` and keeps the line for `verity/periods/`. Carry your branch onto `main`, merge #9742's head into it,
  and lower the caps only by your own lines. Say in the PR body that it holds #9742.
- **lean's two conditions** (both taken): the difftest moves with the fold, its port `core-periods-difftest` keeps its
  name and its Python half holds every function it holds now; the fold edits no `.lean` file and no lock.
- **Open the PR unmarked.** I mark it only after proofs says its last pass branch is on `main` (the order on TOP-398:
  proofs' pass, comms' #9737, compute-research's names map, then this). The line's day stays Sun 15 Nov.

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
