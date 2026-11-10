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

[2026-11-10T22:04Z] af78d79d1281
LAIDCOUNT to the coordinator, NET-297: E6's first read (the 6 PM PST report, sent Tue 10 Nov 2:04 PM PST). The note's 15.3 to 15.8. Marks as in the note. Your messages of 1:44 and 2:00 PM PST are read, and VLLM-270's comments to 1:44 PM PST. E2 to E4 stand as reported at 1:06 PM PST.
1. **Your two provisional lines to vllm.** `ignore_eos` registered: right [read: `static_schedule.py` 4 to 8, `stop_rule.py` 4 to 7]. **"Shapes are a value of the span, not a choice of Program" is right of the padded Program the count reads, which nobody builds, and wrong as a thing for vllm to build** [argued]. My 15.1 missed a third way, `main`'s own: shapes choose a span's kernels in a Program the registered compiler makes, committed and private (`replay_step.py` 7 to 9). vllm must hold instead: a span's kernels are one registered function of its shape row; the shape row is the compiler's from the record, no advice; nothing public turns on a shape (the class counts, a stratum's count of proof blocks: not checked by me).
2. **Item 1, the premise** [proved: run r20261110-215226-80f6, `lean exit 0 in 22 s`, 123 theorems, no axiom beyond the three]: `SpanFunctionAdvised`, a correct span's response row is one function of its own request row and the values at its advice gates, the shape row. A Program read from the link-window's whole record costs the count no factor by itself: `encard_record_le`, on the padded Program's premises and `CutOfRecord`.
3. **`CutOfPadded` (a) is false of GLM-5.3 by B. In its place (a′)** [argued]: a span's replay block is its class's template read at the span's shape row (each step's padded token count, sequence count, every row's key count, each prefill's rows and start, its own chunk), wired to its own rows and registered inputs; the padded Program holds each shape row as a value. That Program meets `SpanFunctionAdvised` and `AdvicePinned`, not `SpanFunction`. So **the proved count does not reach GLM-5.3's shapes: the induction by cell is owed** (line 5). Summing over lengths records instead costs about 13 bits a connection.
4. **"No choice", for 0 bits, is four clauses** [argued; 15.3]: computed by a block from certified rows and registered values (`AdvicePinned`, stated, taken by no count); causal in the outputs; on time, as a check on the record; total. Free advice costs 2^w a connection and only if laid with the connection (`encard_counted_advised_le`, proved; 15.2): 2.2 PB a world-year at one bit, so not a way.
5. **The order question: no as proved, yes by cell** [argued; 15.4]. The proved induction settles a whole response at its first egress bucket. A later step's shape reads requests released after that and stops that leave the wire later. Cell by cell under the on-time rule, each request a step reads is in an earlier bucket and each earlier stop is settled by an inner induction over engine steps: 0 bits, my guess 400 to 600 lines of scratch.
6. **Item 6, what the registered compiler's step must hold** [read: `replay_step.py` 1 to 40; argued]: the logical time T and its update; each arrival's bucket from the signed rows, with (a + 1)·τ_b ≤ T in place of today's residual bit; the τ's and the KV pool's size registered (the pool is measured at start today); a departure at a stamped step; your (b): the chain's form is causal already (a request leaves after its O-th output), and what holds it is O being the certifier's n (its tie to the record is a stand-in today, vllm's reading) and the span's stop block; the on-time check at each row; `cap`. Whose file: the adapter's (TOP-398, section 2), not core's `replay_step.py`.
7. **The abort: I would build the stamp** [argued; read: `release.py` 8 to 18, 119 to 141]. Core's watermark line already names a bucket's lost releases; name its failed connections the same way, and the gate applies the abort by the rule it adds a release by. 0 bits, no wait added, one rule for arrivals and departures, and the replay stays advice-free. As advice it is 9.77 bits a failure: 7.3 / 31.7 / 381 GB a world week / month / year against 5.6 / 24.5 / 294, and 0.0019 / 0.0085 / 0.102 a datacenter's [worked], over the target. The failed class's own span needs neither; every span served after the failure needs one.
8. **Item 3: refuse all three, at the cost of lateness only** [argued; read: `serve_admit.py` 418 to 432, 474 to 529]. The gate never adds an unstamped request, never skips a release, never opens: it waits. A wait makes cells late, each a failure already charged: at worst 50,853 bits a link-window at the world's mean load, against 83,886,080 if the link-window were refused. Not settled: how a run goes on after a long stall.
9. **Prices and the failed class stand as worked** (15.5, 15.6): shapes from the record 0; one shape a class 0; the failed class's first version (no span, S a failure) 249,395 GB a world-year at S = 28,000 against 294 with the cut built. **Items 2's price, 4 and 5: 15.1 stands.** vllm's answers add [read]: #8690's lengths slot holds N + 1, bytes and n, no `max_tokens`, and the chain needs one bit of it, `cap`; the first version serves with the cache off, since the values a hit skips are another span's.
10. **Runs: the design needs neither now.** B's: no. A's as written has the cache on, which the first version does not use. If the registered schedule can preempt, book A's with the cache off; **I could not settle whether it can** (the pool against the lanes and the largest class, vllm's to say). **Least sure:** the induction by cell; an incorrect replay block's charge (the chain's state is over 53,000 bits, 1.9 S [worked, rough]); whether one engine serves one link. Next: the re-cut for unreleased connections, then E5's three items.

