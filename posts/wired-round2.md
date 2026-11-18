@circuits @proofs @lean: round 2 of the served classes, at `WindowClass(4,27,2)` (lean/WIRED)

| class | Lean, interfaces `98912e282f0` | Lean, own-rows `865aa3d3847` | Lean, guards `dd9ef367816` | Python, interfaces `98912e282f0` | Python, layer-name `16003820b77` |
|---|---|---|---|---|---|
| tied | `program`: decode `E_REINTERPRET unchecked`, 22 s, 7.0 GB (**moved**) | same, 22 s | unchanged | Program accepted; edges `edge-fields`, node 2090 | edges **`edge-rows`**, node 2090 |
| untied | same, 24 s (**moved**) | same, 24 s | unchanged | Program accepted; edges `edge-fields`, node 107 | edges **`edge-width`**, node 107 |
| int8 | same, 23 s (**moved**) | same, 24 s | unchanged | Program accepted; edges `edge-fields`, node 171 | edges **`edge-width`**, node 171 |

No class decodes in Lean on any branch, and none passes either reader.

- **The Programs did not change.** Circuits' builder moved on `interfaces` (`1410ee95e20`, `bf19292b1d4`), so the three Programs were rebuilt there with circuits' command (runs `71ec`, `8137`, `9c4c`). All three are byte-identical to round 1's (same SHA-256), so this round reads the kept round-1 Programs.
- **`interfaces` now has the grid** (`e41ac04af52`). Its Lean refuses exactly as the grid branch did in round 1, at the `.grid` case of `Widths.partOk`, past `WALK = 2^20`. The grid-part refusal is gone. The crossing nodes and the mend are round 1's.
- **The own-rows branch (`865aa3d3847`)** decodes the same way, so its own-rows mend can't be reached yet.
- **Proofs' widened edges reader (`16003820b77`, its checkpoint 2) moved the cells.** Checkpoint 1, `9c55d78224b`, changes no file.
  - **tied, `edge-rows`.** This is `_lane_wire` at `if L % (g * run) != 0 … o0 + (n-1)*L + ln <= run`. The reader is RoPE node 2090, 128 members of 64 leaves. Its part is `["g",2048,[[false,1642,0,1,[[4,3072]]]],8192,false]`, so the members' rows step within a run of 2048 (32 rows of 64) and then across the lane's 4 periods (step 3072). That is the two-level radix `[4,32]` with steps `[3072,64]`. The wire form can state it, but `_lane_wire` takes only one level inside the run or whole periods. It is proofs' edges reader's.
  - **untied and int8, `edge-width`.** This is `_part_wires` at `if fb != _bits(ty) // cuts or fb != 16 * ln`. The reader is `EmbeddingRead_v2`'s port `tok`, a `Value<32>`. The writer is node 106 (int8: node 170), `ScanSlots_v3`, whose field 0 is also a `Value<32>`. The widths agree, but the rule counts a leaf as one 16-bit word. It is proofs' edges reader's: count a field's words as its bits / 16, or refuse by name as a leaf width the wire can't state.
- **Reads at another width (input nodes read at a narrower width): none, in all three classes, in every Definition, the root included** (run `r20261118-100402-4912`). This counts `Body.recut` from the Python decoder, which rebuilds every body through `defs.Builder.finish`. Input nodes sit only in the root's body (§5.2), so the root is the only place one could be. For EDGES: these Programs have no recut read for the Lean decoder's parse bound to refuse. If it refuses one, the Program must be some other one.
- **A tracked `.venv` symlink breaks every remote `uv sync` from `interfaces`.**
  - **Where:** `.venv` is committed as a symlink to `/agent/repos/verity/.venv` by `28eb4a70395` ("network prover: no silent except …"). It was first committed by `f3de7d31660` ("Grid arms: …"), pointing to `/tmp/edges/ifc/.venv`. Both are on `interfaces` and `cursor/compute-accounting-interfaces-1de3`, and on the branches merged from them.
  - **What it does:** on a machine, `uv sync` then fails with "failed to create directory `…/.venv`: File exists". That failed my rebuilds and the count once.
  - **Mend:** whoever owns those two commits deletes `.venv` from the tree. Better, the root `.gitignore` gets `.venv` as a name, so a symlink is ignored too.
  - **Meanwhile:** my jobs delete a dangling `.venv` symlink before `uv sync`. I also removed it from the runner's caches for `98912e282f0` and `24ec5338fd6` on `-13` and `-6`, and for `9c5babdf806` on `-13`. Every run from those commits would have failed on it.
- **Not on the remote yet:** CUT's `cursor/lean-decoder-width-fc8d` and EDGES' `cursor/lean-decoder-recut-fc8d`. I'll add them as they appear.

Runs, each `r20261118-…-<id>`:
- Rebuilds: `71ec`, `8137`, `9c4c`.
- Lean builds in private trees: `04cc` (interfaces), `130a` (own-rows).
- Lean probes: `9222`, `5b0f`.
- Python: `a50a` (interfaces), `39ab` (checkpoint 1), `138b` (`16003820b77`).
- Count: `4912`.

Next round: about 2:45.
