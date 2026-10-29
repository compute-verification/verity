Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The served rows held to their Programs (`served_statics`)

The compile slice reads a window's served rows as a committed hidden row, ``srv``, and the verifier holds its b‖c to the
served-rows registration natively. What binds those rows to the batch the server ran is a tie. The routed tie
(`served_tie.ServedRowsTie_v2`) holds them to the Commit's committed positions. Each of its units hashes the window's
whole ``srv`` row and each step's positions in-circuit, about 400 KB at the served class, which no Flock unit of
2^27 holds. `ServedStaticsTie_v1{R, G, IW, L, T}` holds them to the Programs' statics instead, and reads its block's rows
through the Lean verifier's registered reads, so its unit hashes only those rows.

**The rows registered.** `served_rows.register(..., G=G)` commits each block of G steps of window w (j = 0 … P / G − 1)
as leaf w·(P / G) + j of one hm96-sha512/row/v2 registration in a frame-v3-sha512 tree, under the run root's binding. G
is public and in the record (`Served.record`'s ``G``). At G = P a window is one row, the slice's ``srv``. At G < P the
slice is `compile_slice.CompileSliceBlocks_v1{…, G}`, whose ``srv0`` … ``srv{P/G − 1}`` are the window's blocks: unit w's
row ``srv{j}`` is held by its b‖c to leaf w·(P / G) + j, and step u is checked against row u // G. Its ANDs are the
G = P unit's: at 8eb1's engine class, G = 2 gives 27 rows and 1,021,302 ANDs, as at G = 54 = P.

**The unit** reads two rows:
- ``pub``, opened: the block's G steps' statics. For each step, R slots of (valid, start, n), in the Programs' batch
  order, which is the order of the step's rows in advice v0. The verifier holds its own copy, from the Programs it
  compiled.
- ``srb``: the block's served rows, G steps of R slots of (valid, request, kind, start, n). It is a registered port:
  `served_rows.reads_port` is the verifier's own ``--registered`` port at the registration's root, unit u reading leaf
  w·(P / G) + j. Lean refuses a statement whose ``srb`` table row doesn't climb its path to that root at that leaf.

The verdict has bit 0 ``count`` (every slot's valid and n are the statics') and bit 1 ``pos`` (every start is). ``ACCEPT``
is 0x3. The request ids and kinds are not the Program's to show. `IoTieRequest` holds each prefill slot's ids to its
certified request's prompt ids at those positions, and the egress tie holds each decode slot's input to the request's last
sample.

**Positions.** The tie reads the Programs' positions, which are public, and binds no position leaf of the Commit. On vLLM's
eager path a Program's RoPE positions are static structure (`RotaryEmbeddingRule` refuses computed ones), so no eager
Program reads a committed position. The compiled path's RoPE reads ``c00.input/positions``, a committed value
(`verity_vllm.program.predict.compiled`). For a compiled-path window the committed positions are what the compute read, and the
routed tie holds the served rows to them.

**Idle blocks are proved.** Every block of every window is a unit, under `Selection` All, the idle blocks too. A step
past the window's executed count is idle by public structure, and its Programs' statics are zero. But the compile slice
reads every step's served rows, and only the tie shows an idle block's rows zero. A block left untied would make its rows
the prover's word.

**Cost** (`size`, pinned in `served_statics.circuit-check.json`). A step is R·(1 + LW + TW) − 2 ANDs, and the steps'
flags cost 2·(G − 1) more. Every committed row a unit reads is hashed in-circuit, so its two rows, G·R·(3 + IW +
2·(LW + TW)) bits, set its Flock block (`benchmarks/prove_path/served_statics_zk.py cost`):

| class | G | unit ANDs | block (k_log) |
|---|---|---|---|
| the sealed 9c10 run (R 8, P 256, IW 16, L 512, T 2048) | 16 | 2,942 | 21 |
| the served class (R 256, P 256, IW 16, L 131072, T 8192) | 16 | 135,166 | 25 |
| the served class | 64 | 540,670 | 27 |
| the served class | 128 | 1,081,342 | 28 |

So a served-class unit fits 2^27 up to G = 64, four units a window.

**Open.**
- vLLM's serve call (`commit_served`) registers at G = P. It is to pass the tie's G.
- `benchmarks/prove_path/served_statics_zk.py` takes the Programs' statics from the honest rows, as compile(advice) lays
  out a step whose advice's rows they are, and checks them natively against each step's committed positions
  (`served_tie_window.statics_held`). The verifier is to compile them from the advice.
