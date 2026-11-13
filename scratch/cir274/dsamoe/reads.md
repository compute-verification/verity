# CIR-274 4b (DSAMOE): reads no form takes

Each line: the node (rank 0, GLM-5.3 registered rank form), the reader's port, the words read, the writer's row, and why
neither wire form (one writer's row whole; a port cut k ways along its outer elements, or a gather with cut parts) nor a
registered read (rows of the port's width at public positions of one weight) takes it. Tags: [dump] read from
`dump.py`'s node listing at LP = 32, 1 layer, this session; [test] a test of `test_node_window_sparse.py`; [inferred].

1. **`BatchedGemm_v1{B=8,K=192,N=512}` (q_nope absorption, nodes 7 and 27 at LP = 32, 1 layer) and
   `BatchedGemm_v1{B=8,K=512,N=256}` (the value up-projection, nodes 10 and 30), port `w`.** A `SLICES` unit (b, n) reads
   `w[b, n, :]`, K words. The Definition's `w` is `Array<B, Array<N, Array<K>>>`, and the Program reads it from the weights
   record one leaf at a time at a stride of N (leaves 118947840, 118948352, ... for K=192, N=512; 119734272, 119734528, ...
   for K=512, N=256) [dump]: the record holds the field as (B, K, N), so a unit's K weights are a strided column, not a run.
   A registered read (`_Builder.read`, `IF.Read`) reads rows of the port's width from one contiguous run of one weight, so
   `NodeWindow.weight` refuses it. Not a wire at all: the port is left a free input under `weights` (its label named),
   B·N units of K words per node. Cure (not taken): register the field in (B, N, K) order as a parameter of its own, or a
   strided read form; vllm's or the registration's call.

## Frame: what FUSED would need (not a read; nothing lowered needs it)

`Piece` places member m's rows at writer unit `row.unit + m·member + i·step`, affine in m. A pair row's members are
m = t·TOPK + k, and a slot's key reads its own router output (`o<TOPK + k>`), so a template whose units are keyed reads
has its slot-k units contiguous only in an order with k slower than t, such as `(b TOPK + k) T + t`. That order isn't
affine in m once T > 1. The apart blocks avoid it: the reads are table-major, and the block GEMMs, which a reader of the
node reads, are member-major. A FUSED block (`MoeExpertFp8ReadGemm(W)_v1`) is both keyed and read by the node's reader,
so it would need either a `Piece` with two member digits, (TOPK, step_k) and (T, step_t), or a unit-by-unit pattern.
The same two-digit `Piece` would also cut w2's routed-weight wires from one per member to one per slot. [inferred from
`node_window.Piece` and `_slots`; not run]

## The experts' key wires, and what a `Piece` with two member digits would take

Not built: the lead decides. Lines are at the families head `b5c74a0683`. Measured figures are from `p3_keycost.py`;
the 78-layer ones are in `census-78.md`.

**What the patterns are.** Every lowered template that reads a slot's key does it through an `IF.Wire` from the
router-slots unit's output `o<TOPK + k>`. A wire has one writer output (`interface.py`, `Wire`), and a table read's
root, domain and opening each name one table (`IF.TableRead`, and `IF.Opening`, which is `IF.TableOpening` on `main`). So a contiguous run of read units shares
its slot k (through the wire) and its block b (through the table). The runs are (b, k), and that gives B·TOPK key wires
per read template per node in either order: table-major `(b TOPK + k) T + t` (taken), or slot-major `(k B + b) T + t`,
which would make the key wires one per slot and move the same B·TOPK into openings and roots (three entries a run instead
of one wire). w2's routed weight `g` is a second, smaller set. Members are token-major (m = t TOPK + k), and the block
GEMMs are member-major (`m B + b`) so that a member's words are one `Piece`. A slot's `g` row is again one output
(`o<k>`), so a run of GEMM units that share a slot is one member's B units, and that gives one wire per member, M per W
node (`node_window_sparse.py:214-220`).

Measured at 4 layers (one MoE layer, a prefill of T = 43 and a decode of T = 1, run `r20261113-025444-67c6`, source
`b5c74a0683`):
- 13,238 wire patterns after `IF.coalesce`, of which 12,096 are key wires (reader `EmbeddingRead_v1`, port `tok`) and
  352 are `g` wires (reader `MoeExpertFp8BlockGemmW_v3`, port `g`).
- The key wires are 2,721,147 of the description's 4,417,654 bytes.
- `check_wires` takes 0.09 s with them and 0.02 s without; `ordered` takes 0.05 s with and 0.02 s without.

**A two-digit `Piece`: what it changes.** `Piece(lo, hi, row, step, member)` places member m's rows at unit
`row.unit + m·member + i·step` (`node_window.py:82-101`). Two digits would let member m = t TOPK + k sit at
`row.unit + t·s_t + k·s_k + i·step`, so the w2 block GEMMs could run slot-major, `(k T + t) B + b`, with `g` one wire
per slot. The frame lines it touches:
1. `Piece` and `Piece.rows` (`node_window.py:82-101`): `member` becomes digits `((T, s_t), (TOPK, s_k))`, and `rows`
   adds Σ digit·step. `Piece.member` has no other reader: `NodeWindow.rows` (`node_window.py:253-273`) calls `p.rows`
   only, and `NodeWindow.out`'s tiling check (`node_window.py:360-372`) reads `lo`, `hi` and `row.words`.
2. `_whole` and `_gather` (`node_window.py:399-442`) with `window_builder._step` (`window_builder.py:119-127`). A
   reader that takes the output whole gets one gather at one unit step across members, else one gather a member
   (`node_window.py:407-408`). The w2 output's readers are such gathers: `MoeSum_v1{TOPK=8,H=192}` reads it through
   `AllGather_v1{WORLD=6,N=32}`. So `_step` has to find two digits (as `_lead` does for a feed,
   `node_window.py:523-540`), and `_gather` has to feed `parts` with them (`node_window.py:421, 432`). Otherwise each
   member's gather is fed alone, adding M·G patterns where it saves M.
3. `feed` and `_lead` (`node_window.py:308-324, 523-540`). These already take a reader shape of several digits, but only
   the digits the reader passes. A reader that passes `(M, …)` (the dense half's `MoeSum` and w2's quantiser, and my own
   `xq`, `sx` feeds) finds no one-digit step over two-digit rows and falls back to a feed a unit. So either each reader
   of an expert output passes `(T, TOPK, …)`, which touches `node_window_dense.py`, or `_lead` refines a digit that
   fails into the writer's digits, which is a frame change. `_lead` cannot find T and TOPK alone: they come from the
   writer.
4. My lowering: `expert_gemm` (`node_window_sparse.py:209-226`) changes its GEMM unit order and its `Piece`, and
   `test_moe_expert_units` changes with it.

**Checks that read it.** A `Piece` never reaches the Circuit. The checks see only the wires it produces, whose
`radix`/`step` are already tuples of any length (`IF.Wire`). `IF.check_wires`, `IF.coalesce`, `IF.ordered`,
`check_inputs`, `check_proof_block` and the frame's `_check_rows` need no change. The tests of the frame
(`test_node_window.py`'s `Piece` and gather cases) and mine do.

**The pattern count it would give.** It removes `g` wires only, from M per W node to TOPK per W node: at 4 layers
352 → 16 (8 at the prefill, 8 at the decode), so 13,238 → 12,902 patterns, 2.5 % fewer. The key wires stay at
B·TOPK per read template per node, 12,096 at 4 layers: a `Piece` places writer rows for readers, and the key wires'
count comes from the wire's one output and the opening's one table. Cutting them takes a format change, neither the
frame's nor this lane's: a wire whose writer output steps by a digit (`o<TOPK + k>` over k), or a table read and opening
over a family of tables indexed by a digit. Either changes `interface.Wire` or `Table`/`Opening`, `check_wires`,
`coalesce`, `ordered` and the Lean verifier's reading of them. At 78 layers (run `r20261113-025656-e215`, `census-78.md`), the key wires are 907,200 of 945,322 patterns
(207.9 MB of the 308.7 MB description; `check_wires` 6.76 s with them, 0.67 s without), and `g` is 26,400. A two-digit
`Piece` would give 1,200 `g` wires, so 920,122 patterns, 2.7 % fewer. [the counts are measured; that the two-digit `Piece` yields TOPK per W node is inferred from
`_slots`, not built]
