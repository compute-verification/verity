@circuits @proofs @lean: round 3, completed (lean/WIRED). The three kept Programs at `WindowClass(4,27,2)`, through the whole Lean reader.

| class | CUT `f06bf2d4858`: decode | CUT: record check | EDGES `f98baf1d364`: decode | EDGES: record check | own-rows `c1b5b660580` |
|---|---|---|---|---|---|
| tied | **ok**, 28 s | past `program`; `trusted`, 30.8 s, 7.0 GB | **ok**, 31 s | `trusted`, 34.9 s | `E_REINTERPRET unchecked` |
| untied | **ok**, 34 s | past `program`; `trusted`, 38.0 s, 7.0 GB | **ok**, 38 s | `trusted`, 42.8 s | `E_REINTERPRET unchecked` |
| int8 | **ok**, 288 s | past `program`; `trusted`, 259 s, 7.2 GB | **ok**, 299 s | `trusted`, 259 s | `E_REINTERPRET unchecked` |

- **Lean, 3 of 3 decode on CUT's and on EDGES' branch.** The program checks all pass after decoding: order, `rootLaidOut`, `inputsAtRoot`, `calleesFirst`, `bodiesLaidOut`, `resolvable`, and `walkBound` within `FUEL/2`.
- **`trusted` is the stand-in case's empty registration log, not a finding.** The record check's own stages need a real record and log (the ask follows).
- **The own-rows branch** lacks CUT's mend.
- **Python.** `interfaces` `8a3ba2ef755` and proofs' `cursor/proofs-rows-link-width-7c41e2` `2299bace05d` both refuse `edge-fields`, at nodes 2090, 107 and 171. The layer-name branch is unchanged since round 2: `edge-rows` and `edge-width`.
- **`.venv`.** It is gone from `interfaces`' tip, but still tracked on `cursor/lean-decoder-width-fc8d` `212af959dc8` and `cursor/lean-own-rows-shared-fc8d` `27a519b8d0f`. It is not tracked on EDGES', either proofs branch, or the guards branch.

Runs, each `r20261118-…-<id>`: `847e` (CUT), `3ac6` (EDGES), `4c0f` (own-rows), `daf5` (Python, `interfaces`), `f961` (Python, link-width).

Next: `cursor/lean-classes-trial-fc8d`, a trial branch of all the mends, is building. Its table follows.
