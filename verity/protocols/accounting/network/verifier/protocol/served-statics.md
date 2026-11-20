Part of network attestation's [PROTOCOL.md](../PROTOCOL.md).

## The served rows held to their Programs (`verity_vllm.compiler.served_statics`)

This file describes Definitions and a design: no proof of them runs, and no check in it is run by an auditor. The one
driver that runs any is the developer's own, natively, in plain, on the prover's rows (the open item).

The compile slice reads an execution phase's served rows as a committed hidden row, ``srv``. Its b‖c against the
served-rows registration has a native check, `verity_vllm.compiler.served_rows.leaf_ok`, which only its tests call: no
driver and no auditor-side code runs it. A tie is the Definition built to bind those rows to the batch the server ran; no
proof of it runs, so today nothing binds them. The routed tie
(`verity_vllm.compiler.served_tie.ServedRowsTie_v2`) states that they agree with the Commit's committed positions. Each of
its proof blocks hashes the execution phase's whole ``srv`` row and each step's positions in-circuit, about 400 KB at the
served class, which no Flock proof block of 2^27 holds. `ServedStaticsTie_v1{R, G, IW, L, T}` states that they agree with
the Programs' statics instead, and reads its block's rows through the Lean verifier's registered reads, so its proof block hashes only those
rows.

**The rows registered.** `verity_vllm.compiler.served_rows.register(..., G=G)` commits each block of G steps of execution phase w
(j = 0 … P / G − 1) as leaf w·(P / G) + j of one hm96-sha512/row/v2 registration in a frame-v3-sha512 tree, under the
run root's binding. G is public and in the record (`Served.record`'s ``G``). At G = P an execution phase is one row, the slice's
``srv``. At G < P the slice is `verity_vllm.compiler.compile_slice.CompileSliceRowGroups_v1{…, G}`, whose ``srv0`` …
``srv{P/G − 1}`` are the execution phase's blocks: proof block w's row ``srv{j}`` is to be held by its b‖c to leaf w·(P / G) + j, and
step u is checked against row u // G. Its ANDs are the G = P proof block's: at 8eb1's engine class, G = 2 gives 27 rows
and 1,021,302 ANDs, as at G = 54 = P.

**The proof block** reads two rows:
- ``pub``, opened: the block's G steps' statics. For each step, R slots of (valid, start, n), in the Programs' batch
  order, which is the order of the step's rows in advice v0. The verifier is to hold its own copy, compiled from the
  advice; today it compiles none and holds none.
- ``srb``: the block's served rows, G steps of R slots of (valid, request, kind, start, n). It is to be a registered port:
  the verifier's own ``--registered`` port at the registration's root,
  proof block u reading leaf w·(P / G) + j. Lean refuses a statement whose ``srb`` table row doesn't climb its path to
  that root at that leaf.

The verdict has bit 0 ``count`` (every slot's valid and n are the statics') and bit 1 ``pos`` (every start is). ``ACCEPT``
is 0x3. The request ids and kinds are not the Program's to show. In the design, the ingress-ids tie (`io_ids`) is to hold
each prefill slot's ids to its attested request's prompt ids at those positions, and the egress tie (`io_tie`) each
decode slot's input to the request's last sample; no proof of either runs. A zero-knowledge proof of `IoTieRequest`
would hold each request's prompt length and arrival ordinal to the network attester's ingress cells,
not the ids' values.

**Positions.** The tie reads the Programs' positions, which are public, and binds no position leaf of the Commit. On vLLM's
eager path a Program's RoPE positions are static structure (`RotaryEmbeddingRule` refuses computed ones), so no eager
Program reads a committed position. The compiled path's RoPE reads ``c00.input/positions``, a committed value
(`verity_vllm.program.predict.compiled`). For a compiled-path execution phase the committed positions are what the compute read, and the
routed tie states that the served rows agree with them.

**Idle blocks are proof blocks.** Under the tie, every block of every execution phase would be a proof block, under
`Selection` All, the idle blocks too. A step past the execution phase's executed count is idle by public structure, and its
Programs' statics are zero. But the compile slice reads every step's served rows, and only the tie would show an idle
block's rows zero. No block is tied today, so every block's rows are the prover's word, the idle ones' too.

**Cost** (`size`, pinned in `adapters/vllm/verity_vllm/compiler/served_statics.circuit-check.json`). A step is
R·(1 + LW + TW) − 2 ANDs, and the steps' flags cost 2·(G − 1) more. Every committed row a proof block reads is hashed in-circuit,
so its two rows, G·R·(3 + IW + 2·(LW + TW)) bits, set its Flock block
(`benchmarks/prove_path/served_statics_zk.py cost`):

| class | G | proof block ANDs | block (k_log) |
|---|---|---|---|
| the sealed 9c10 run (R 8, P 256, IW 16, L 512, T 2048) | 16 | 2,942 | 21 |
| the served class (R 256, P 256, IW 16, L 131072, T 8192) | 16 | 135,166 | 25 |
| the served class | 64 | 540,670 | 27 |
| the served class | 128 | 1,081,342 | 28 |

So a served-class proof block fits 2^27 up to G = 64, four proof blocks an execution phase.

**Open.**
- vLLM's serve call (`commit_served`) registers at G = P. It is to pass the tie's G.
- No zero-knowledge proof of either tie runs, and nothing the verifier accepts rests on one. Both are Definitions with
  pinned costs. `benchmarks/prove_path/served_tie_window.py` checks the routed tie natively, in plain, on a sealed
  execution phase's real rows, and checks the statics against each step's committed positions (`statics_held`), taking
  them from the honest rows. The verifier is to compile them from the advice.
