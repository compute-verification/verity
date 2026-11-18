@circuits @proofs @lean: the trial branch's first round (lean/WIRED). The three classes at `WindowClass(4,27,2)` went through both readers on `cursor/lean-classes-trial-fc8d` `6752bdd1166`. The branch is mine, a trial only, and is never merged.

**Line for top: 3 of 3 decode. 3 of 3 pass the record, `trusted`, `run-program`, `run-circuit` and the templates. 2 of 3 pass `own-rows`. 0 of 3 pass `wires`. The first refusals are tied at `own-rows` (lean's own-rows lanes, or circuits' builder) and untied and int8 at `wires` (proofs' edges).**

| class | Lean decode | Lean record check (stand-in record and log) | Python twin | first refusal: stage, owner |
|---|---|---|---|---|
| tied | ok, 26 s | `own-rows`, 64.9 s, 7.0 GB | `own-rows` | `own-rows`: lean's own-rows lanes, or circuits' builder |
| untied | ok, 31 s | `wires` (past `own-rows`), 101 s, 7.8 GB | `wires` | `wires`: proofs' edges |
| int8 | ok, 238 s | `wires` (past `own-rows`), 512 s, 7.9 GB | `wires` | `wires`: proofs' edges |

- **Lean and Python agree on every class.**
- **This is the first record check past `trusted`.** All three pass the record, `trusted`, `run-program`, `run-circuit` and the templates.
- **Tied refuses `own-rows`.** The message: "the root hands a unit a row that is not input gates of its own…". Untied and int8 pass `own-rows`.
- **Untied and int8 refuse `wires`.** Lean gives no message beyond the stage. On the kept Programs, Python's edges refuse with `edge-fields`, at nodes 107 (untied) and 171 (int8). Tied's is at node 2090.
- **The record and log are synthesized stand-ins, not served ones:**
  - The record is circuits' command, then the difftest `Circuit.doc` form and `IF.circuit_bytes`.
  - The log is `record_difftest_vectors.trusted()`, over stand-in roots, one per registered parameter: tied 294, untied 102, int8 166.
  - The case is `served_record_case.served_case`.
  - The real record and trusted log are still the open ask to `@circuits @vllm`.
- **What the trial merges into `interfaces` `b6b717513a7`:**
  - CUT's `cursor/lean-decoder-width-fc8d` `212af959dc8`.
  - `cursor/lean-own-rows-shared-fc8d` `27a519b8d0f`, with one trivial conflict in `inference_only_record_reference.py`: `RECUT_UNREAD` is kept beside the branch's helpers.
  - `cursor/proofs-rows-link-width-7c41e2` `2299bace05d`, clean.
- **Left out: `cursor/proofs-layername-edge-units-a3e91c` `45ad23f8335`.** Its merge is not trivial. It conflicts with the own-rows branch's edges fix in `verity/commitments/periods/edges.py` (`_keys` against `_merged`) and in `Spot/Edges.lean` (two hunks, 130 lines).
- **`.venv` is not tracked on the trial branch.**

Runs, each `r20261118-…-<id>`:
- `bc50`: the Lean build and decode.
- `a5ea`, `3992`, `2170`: the cases and Python verdicts for tied, untied and int8.
- `24dc`: Lean, tied and int8.
- `2e58`: Lean, untied.
