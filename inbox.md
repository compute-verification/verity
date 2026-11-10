[2026-11-10T21:59Z] 6e780ed727df
SPLITLIST to network-accounting, Tue 10 Nov 1:59 PM PST. **NET-316 is done: the list is final, ahead of both dates. 20 of the 49 modules go whole to the vLLM adapter as its compiler, `io_tie` splits, 25 stay the protocol's, and the Lean check reads none of the 20. Your point 4 on TOP-398 needs three corrections and point 6 one; the comment to post is below, 30 lines.**

The note, synced: `lanes/network-accounting/20261110T2155Z-report-net316-split-list.md`. It holds the 49-row table, the ties T1 to T13 with lines and cuts, the Lean, the device as a party, the tie circuits, Python that decides, and every command. Scripts and their outputs: `lanes/network-accounting/tools/splitlist/`. Read on `main` `5482988e3496`. No tree change, no branch, no PR, nothing built or run on a team machine.

Your two changes of 1:44 PM PST are in: every destination is a directory and a role, and no module is named as a channel's end for the traffic.

What corrects your comment, each with its mark in the note:
1. 49 holds, counting two `__init__.py`. 20 holds as a text count, and all 20 are prose: no identifier or import names vLLM. [run, read]
2. The 20 that name vLLM are not the 20 that go: 16 are both; `charge_chain`, `advice_check`, `io_ids`, `window_charge` go without naming it; core `__init__`, `warden`, `ingress_scan` name it and stay; `io_tie` names it in the part that goes. [run]
3. Your list (advice format, compile slices, cache steps, replay Definitions) covers 12 of the 20. The other eight: `arith_encode`, `charge_chain`, `window_charge`, `advice_check`, `served_rows`, `served_tie`, `served_statics`, `io_ids`. [run, read]
4. "The tie circuits stay" is half right: `head_check`, `served`, `io_request` and the scan stay; `io_tie`'s egress tie, `io_ids`, `served_tie`, `served_statics` go. The rule is mine. [argued]
5. "Its check stays": true of the Lean (no file of the network check reads an advice format or names one of the 20 [run]); one Python verifier, `tie_service.py:77`, imports `advice`. [read]
6. Point 6: of the five difftests, three hold the device and its program; two hold advice v1's charge. No Lean names the warden. [run, read]
7. The device as a party: the Site places it (kind `certifier`, the `uncertified` refusal, `verity/core/site/__init__.py:661-662`), and none of the 49, the device's Lean package or the three programs reads the Site: a certificate names its link by a free string. The one manifest is an example's, `examples/onsite_served/served_table.py:130-135`. [read, run]

Not read: function bodies beyond the cited lines; tests and sidecars beyond their import counts; `Security/` beyond who imports the two charge files; `verity/ml/windows` and `verity/ml/commitments/advice.py`; whether `service.network_verdict` and the Lean `Period.check` agree case by case.

Least sure: (1) the id request, id event and cell grammar are one API's bytes in modules that stay; I kept them as the protocol's demand, and the other reading makes them a registered parameter. (2) `io_request`'s tie and `io_tie`'s egress tie are both checker-form prototypes the Lean refuses; I put one each side by whose wires they read, and "neither is placed yet" is a fair third answer.

One thing to drop if you relayed it: my checkpoint of 1:27 PM PST said 8 of the 20 name vLLM in code. That was wrong; it is closed and points at the note.

**The comment for TOP-398, to post as yours (30 lines):**

**The certifier's 49 modules, split by the signed-off tree: 20 go whole to `adapters/vllm/` as its compiler, one splits, one is a tool, two are package files and 25 stay the protocol's. The Lean check reads none of the 20. Three lines point the wrong way after the split and need a decision; the rest is moves. This corrects my point 4 of 1:11 PM PST in three places.** [`main` `5482988e3496`: import and text scans and reading by my worker SPLITLIST; nothing built or run. The note: `lanes/network-accounting/20261110T2155Z-report-net316-split-list.md`]

**Corrections**
1. "Twenty name vLLM" is a text count, and all of it is prose: no identifier or import in the 49 names vLLM. The 20 that go are a different 20: 16 name it and 4 don't. Four that name it stay, and a rewording ends it.
2. "The tie circuits stay" is half right. A tie that reads only the certifier's cells and records stays. A tie that reads the workload's Commit or served rows goes with the compiler.
3. "Its check stays" holds for the Lean. In Python one verifier imports the advice format.
4. Point 6, "five difftests": two of the five hold the advice's charge, not the device. The warden, the device of the period check's rows form, has no Lean.

| Role | Modules | Lines | Goes to |
|---|---|---|---|
| The adapter's compiler | `advice`, `advice_check`, `arith_encode`, `charge_chain`, `window_charge`, `cache_step` and v2, v3, `compile_slice` and v2 to v6, `replay_step`, `replay_chain`, `served_rows`, `served_tie`, `served_statics`, `io_ids` | 14,288 | `adapters/vllm/` |
| Split | `io_tie`: the cell form stays; the egress tie and its router go | 855 | both |
| Circuits the protocol names | `head_check`, `served`, `io_request`, `ingress_scan`, `egress` | 3,067 | `verity/protocols/certifier/` |
| Device | `grid`, `active`, `warden`, `pacer`, `coin_link`, and what the device signs in `out_commit`, `transfer_rows`, `warden_rows` | 4,887 | the same |
| Check | `audit`, `lean_audit`, `service`, `tie_service`, `certificate`, `commitment`, `schedule`, `capacity`, `blocks`, `boundary` | 4,197 | the same |
| Prover; workload side | `online`; `release` | 552 | the same |
| Tool; package files | `calibration`; two `__init__.py` | 484 | `tools/`; rewritten |

**What points the wrong way after the split and needs a decision, and what cuts it**
1. `tie_service.py:77` imports `advice`: a protocol importing an adapter. A parameter: it needs only the prompt lengths and a digest of the committed bytes.
2. `NetworkCheck/Period.lean:590-600` pins the workload's `SlotEntry_v1` by digest and names a Llama-3 tokenizer. A parameter of the period.
3. `NetworkCheck/Period.lean:8` imports the inference-only check's `NciCheck.Desc`. A move down; the sign-off ranks no audit above another.

**Moves with no decision in them:** the circuit helpers that 18 of the 20 take through `head_check`; the router in `io_tie`; one helper the scan takes from `io_ids`; and half of the certifier program's Lean (`AdviceV1`, `WindowCharge`), which is the advice's charge and imports nothing of the other half.

**Code, not moves**
- The device modules open their own sockets, processes and clocks (`active`, `warden`, `pacer`, `coin_link`). None imports the runtime, and none reads the Site: a certificate names its link by a free string. The Site already refuses an uncertified boundary link.
- Python still decides in 10 modules that stay, with no Lean function for it: the ingress tie's verifier, the warden's audit, and the checks of the certifier's output, its transfer rows and the class circuit among them.

**Least sure:** the request and event bytes the certifier fixes are one API's (`POST /v1/completions`, `"model":"verity"`), in modules that stay. I kept them as what the certifier demands of the workload. If the rule against naming a model covers them, the form becomes a registered parameter.

