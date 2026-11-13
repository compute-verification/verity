# CIR-274: the node window's frame, and how a family's lowering is added

Worker REQBLOCK, Thu 12 Nov 5:05 PM PST [Fri 01:05Z]. The frame is
`adapters/vllm/verity_vllm/program/predict/node_window.py` with its test
`adapters/vllm/tests/program/predict/test_node_window.py`. Both are on `cursor/circuits-cir274-node-window-6f04`, over
`main` at `4f0154e42026a0d1b4921830b9c6dc47aa5214de` (the head is in the return and the checkpoint). Both are new files.
No existing file changed, and `scratch/cir274/needs-vllm.md` does not exist because the frame needed no hunk.

Tags: [run ID] is a run's output read in this session, [read FILE:LINE] is a line read, [test] is the test file run
locally (6 passed, 12.2 s, 574 MiB peak), and [inferred] is inferred.

## 1. Piece 1: the class Program, rank by rank [r20261113-003137-40d9]

Rank r's request Program is `steps.request_program(reg, u)`. Here `reg` is `compile_cost.registered(r, layers, "eager")`
and `u` is `plan_steps(reg, *session([(0, 43, 2)])).requests`' unit, so one declared request of 43 prompt ids and
2 outputs.

| layers | rank digests (SHA-512 prefixes, r0..r7) | nodes per rank | node Program | `check_ranks` |
|---|---|---|---|---|
| 4 | 7f36f17ae005, 17957abb42a6, dc49bdf8b1e2, f9245cecd17e, ee38f5c9f38e, c0c4a9928502, de608165876d, 1f1f1d78cd96 | 319 | 6ea9ac277fe6…, 52,224 nodes | passes, 8 ranks x 6,528 nodes, 2,444,104,450 gates |
| 78 | 3c99109308321b3b, 97eb491793a75c35, ed44f49ebfa33508, bb971727dd07a47c, c4b151e4fc4e43ce, 983714f7291741ca, 0cbe005f8117017d, 8ff437a283eded3e | 6,979 | 5b8e2ecd4bfb810b…, 1,634,592 nodes | passes, 8 ranks x 204,324 nodes, 142,303,712,050 gates; 1.83 GiB peak, 184.5 s |

The full 78-layer digests are in that run's `p1_l78.json`. At 78 layers each rank has 2,214 params: the prompt, the
weights record, and 2,212 `peers_*` partials.

## 2. The frame

`node_window(fn, families=FAMILIES, grain=Grain(), name, coord=1, provable=True) -> (Circuit, NodeWindow)`.

- **The walk.** The frame walks rank r's Program `fn` node by node. A node whose family (`fn.name`, or the id's head
  before `{`) is in `families` is lowered at its step. A step ends at its token select (`SELECTS`, the
  `serving_templates.STAGES` rows), so class G has 2 steps. Any other node emits nothing, and every row a lowered reader
  would read of it is a free input.
- **Free inputs.** Every `peers_*` input (another rank's partial) and the prompt are free inputs too.
- **The census.** `NodeWindow.census(circuit)` reports the window's digest, plus units, templates and wires by the family
  that emitted them. It lists free inputs by the family whose rows they are, the unlowered nodes by family, and every
  `peers_*` input with the nodes that read it.
- **The digest** is SHA-512 of `interface.circuit_bytes(circuit.description())`. The window is one slot (the request) at
  the Program's steps (`Blocks(1, steps, inference(spans, 1))`), and `interface.Circuit.__post_init__` runs
  `check_wires` and `check_inputs` [read verity/commitments/windows/interface.py, `Circuit`].
- **COORDS GEMMs at the coordinate grain.** At `coord=1`, the serving coordinate (`serving_templates.coordinate`), each
  output word of a GEMM is a unit. Its coordinates are coordinate-major: unit (j, t), coordinate j of member t, is
  `at + j*M + t`.

### 2.1 Adding a family: the contract

A lowering is a function `lower(w: NodeWindow, j: int, node) -> None`, registered as `FAMILIES[family] = lower` (or
passed in `families=`). The frame calls it once per node of that family, with `w.b.label` already set to `(0, step)`.
`node` is a `call` or `batch` node of the Program (`node.fn`, `node.args`, `node.n`, `node.axes`, `node.form`).

What it is handed, all on `w`:

- `w.members(node)` gives M, the number of batch members (1 for a call).
- `w.arg(node, k, t)` gives member t's argument k as `Refs`. An argument batched on axis 0 gives its t-th slice, axis None
  gives the whole argument, and any other axis is refused.
- `w.take(node, k, lo=None, hi=None, *, whole=False, most=None)` gives each member's writer rows for argument k's leaves
  [lo, hi), a list of M tuples of `Row`. Or it returns a `_Free` when any of those rows is not written in the window
  (an unlowered node, a peer, or the prompt).
  - With `whole=True`, each member gets one row. Where the words are several writer rows, the frame gathers them (§2.3).
  - With `most=k`, it gives the first-level gather chunks that stand for the rows, where at most k of them cover the
    range; otherwise the whole gathered row. Use this for a column block of a row that was gathered before.
  - Call `take` before allocating your units, so that every gather it makes comes before its readers in unit order.
- `w.weight(refs, words)` gives the first word of the one weights-record run `refs` reads, which must be exactly `words`
  long. `w.label(refs, words)` gives that weight's label, for `serving_templates.in_layer`.
- `_part(spec, family, **statics)` gives the bound template of `family` among `serving_templates.unit_templates(spec)`
  at those statics. Use only templates the serving registration lists.

What it must emit:

1. **Units:** `w.unit(family, fn, count) -> (tid, at)`. This allocates `count` units of `fn` and records them under
   `family` for the census. Don't call `w.b.unit` directly.
2. **Feeds:** `w.feed(tid, port, at, grid, shape)`. `grid` is a `take` result in unit order, one entry per unit, and
   `shape` is the mixed radix of the units, the first digit slowest. The frame emits one `_Builder.feed` with `lead`
   digits where every digit shifts the first unit's rows by a constant step, else one feed per unit. A `_Free` grid
   leaves the port free for all `prod(shape)` units and records each source as a `Free(tid, port, at, count, kind,
   family, name)`. This is the only way a lowering declares free rows: it never invents rows.
   A lowering that knows its own constant-step pattern may call `w.b.feed(tid, port, at, count, rows, lead)` directly,
   as the GEMM's xq/sx and the norms' scale do.
3. **Weights:** `w.b.read(tid, port, at, radix, base, step)`. `base` comes from `w.weight(...)`, and `step` is words per
   digit. It registers the weight as rows of the port's width, so every read must start on a row.
4. **Gathers:** none by hand. `take(..., whole=True)` and `take(..., most=...)` make them.
5. **The output:** `w.out(j, node, pieces)`, exactly once. The `Piece(lo, hi, row, step=1, member=1)`s tile one
   member's output words. Words [lo, hi) are rows of `row.words` words: member m's first is unit
   `row.unit + m*member`, and each next one is `step` units on. A read that cuts a piece's row is refused by name.

Refused by name (`Unexpressible`):

- a lowering that states no output;
- a node with a return that isn't whole u16 words;
- a read of the weights record where a wire is due, or one that spans two weights;
- a gather whose rows are not one width, or whose two levels don't fit 128 rows;
- at `settle`: a port read at two row counts, rows that aren't the port's outer elements cut that many ways, a unit
  past 256 rows, and (with `provable`) any unit `check_proof_block` refuses.

### 2.2 The `_Builder` calls the frame makes (`window_builder.py` as on `main`)

This is the one place to edit if vllm's refactor (`cursor/vllm-refactor-c1f1`) changes `_Builder` [read
adapters/vllm/verity_vllm/program/predict/window_builder.py:130-300].

| call | where | arguments |
|---|---|---|
| `_Builder(grain, 1, provable)` | `NodeWindow.__init__` | one program, i.e. one slot |
| `b.weights = ((label, first word, words), ...)` | `NodeWindow.__init__` | an attribute set from the weights record's fields (`_spans`) |
| `b.label = (0, step)` | `NodeWindow.lower`, before each lowering | slot 0 |
| `b.unit(fn, count)` | `NodeWindow.unit` only | returns `(tid, at)` |
| `b.feed(tid, port, at, count, rows, lead)` | `NodeWindow.feed`, `_gather`, `block_fp8_linear` (xq, sx), `scale_norm` (scale/r) | `lead` digits `((radix, unit step), ...)`, slowest first |
| `b.read(tid, port, at, radix, base, step)` | `block_fp8_linear` (w, ws), `scale_norm` (w) | weights-record words; `read` calls `b.weight` itself |
| `b.settle()` | `NodeWindow.circuit` | |
| `b.merged()` | `NodeWindow.circuit` | gives table reads and openings (none yet) |
| `b.fns, b.units, b.cuts, b.params, b.tables, b.reads, b.wires, b.spans, b.grain` | `NodeWindow.circuit`, `census`, `scale_norm` | attributes read |

Also imported from `window_builder`: `WEIGHTS`, `Grain`, `Row`, `_spans`, `_step`, `_word`, `inference`.

### 2.3 Gathers: the two-level rule

A whole read of n writer rows of c words (R = n·c) works as follows:

- **One level** where n + 1 ≤ 128: `AllGather_v1{WORLD=n, N=c}`, one per member.
- **Two levels** otherwise. The first is `AllGather_v1{WORLD=K/c, N=c}`, R/K of them per member, group-major
  (unit `a1 + g*M + t`). K is the largest multiple of c dividing R with K/c + 1 ≤ 128, which is `grain.chunk` at c = 1.
  The second is `AllGather_v1{WORLD=R/K, N=K}`, one per member.

The first level's K-word chunks are registered, so a later column block (`take(..., most=...)`) reads the chunks rather
than the coordinates. A gather is made once for the rows it holds and reused.

Rows ending at 64-word chunks and gathered at 2048 words:
`AllGather_v1{WORLD=64,N=1}` x 32 per member, then `AllGather_v1{WORLD=32,N=64}` [test: `test_two_level_gather`].

### 2.4 What is lowered now (delivery B)

- **`block_fp8_linear`**, for `DeepGemmFp8Linear` and `FlashInferFp8Linear`, as their `PARTS`. Per member, the
  quantiser (`Fp8GroupQuantUe8m0_v1{K,128}` or `FlashInferFp8Quant1x128_v1{K}`) reads x whole. Then come the N
  coordinates `ScaledMmFp8BlockSharedScale_v1{K,N=1,G=128}`, each reading the quantiser's q and s, its weight row, and its
  128-block's scale row. DeepGEMM's row returns (q, s, y); FlashInfer's returns y, with q and s as mids that are wired.
- **`scale_norm`**, for `DsaRMSNorm`, `RMSNormCudaVector8` and `RMSNormFusedCudaVector8` (v1/v3), cut at the scale.
  Per member, the SumSq reads x (and res) whole. Then N/G apply units, block-major, read their G columns as the rows that
  hold them (the chunks where there are any), their G weights, and the scale. G is the widest halving of N within
  `grain.norm` = 256: 256 for N = 2048, 192 for N = 6144. A row outside every decoder layer, the final `model.norm`,
  keeps its `FINAL` cut: one unit of the row per member.
- **Why these first.** This is the hardest first family the brief asked for. `DsaRMSNorm`'s SumSq reads q_a's
  y[0:2048], which is 2,048 one-word GEMM coordinates, so it needs the two-level gather. Its apply units read the
  64-word chunks, cut 4 ways (`DsaRMSNormApply_v1{G=256}` x cut 4). q_b's quantiser then reads the norm's 8 blocks of
  256 through `AllGather_v1{WORLD=8,N=256}`.

### 2.5 Delivery B's census: rank 0 at (43, 2), 4 layers [r20261113-005531-62da]

The Program's digest is 7f36f17ae005…, the same as piece 1's rank 0 (319 nodes). The window settles with every
template provable, and the Circuit passes `check_wires` and `check_inputs`. It took 9.0 s at a 0.528 GiB peak.

- Window digest: `d5301f52fe1a2e2a20afabcab027a811e20350bd4d7818257583e7f00a2d4c44c3cb45093edb572f8a933b956a5511fc21c47b6680fe951c9792cd99e4463f05`.
- 3,433,430 units, 23 templates, 117 wire patterns after `coalesce`. 2 steps.

| family | units | templates | wire patterns (before coalesce, by reader) |
|---|---|---|---|
| DeepGemmFp8Linear | 3,336,284 | 8 | 52 |
| FlashInferFp8Linear | 77,588 | 8 | 45 |
| RMSNormFusedCudaVector8 | 9,934 | 3 | 20 |
| DsaRMSNorm | 1,584 | 2 | 24 |
| RMSNormCudaVector8 | 1,452 | 2 | 2 |
| gather | 6,588 | 4 | 40 |

The template counts add up to more than 23 because the two linears share their `ScaledMm` and quantiser templates.

Free inputs by the family whose rows they are (ports / unit-ports / words):

| family | nodes | read by | ports | unit-ports | words |
|---|---|---|---|---|---|
| AllReduceMultimem | 1, 54, 59, 112, 117, 170, 175, 228, 242 | the prefill norms' SumSq and apply x (and res), and the final norm | 19 | 12,772 | 4,761,600 |
| BatchedGemm | 52, 110, 168, 226, 257, 271, 285, 299 | o_proj's quantiser x | 8 | 176 | 360,448 |
| SiluMul | 57, 115, 173, 232, 261, 275, 289, 304 | down_proj's quantiser x | 8 | 176 | 214,016 |
| AllReduceRmsNormMnnvl | 259, 263, 273, 277, 287, 291, 301 | the decode linears' quantiser x | 7 | 7 | 43,008 |
| AllReduceMnnvl | 248 | the decode's first norm | 2 | 33 | 12,288 |

No lowered unit reads a peer directly. The 140 `peers_*` inputs are read only by unlowered collectives (`AllReduceMultimem`,
`AllGather`, and the decode's `AllReduceMnnvl` and `AllReduceRmsNormMnnvl`), and the census lists each one with its
nodes. Unlowered nodes by family: AllGather 2, AllReduceMnnvl 1, AllReduceMultimem 9, AllReduceRmsNormMnnvl 8,
BatchedGemm 16, CublasSplitKGemmF32Out 1, DsaLatentRow 8, EmbeddingShard 2, Fp8GroupQuantUe8m0 4, Gemm 2,
LlBf16DotprodGemm 1, MoeExpertFp8Gemm 2, MoeExpertFp8GemmW 2, MoeRouterSigmoidBias 2, MoeSum 2, ResidualAdd 2,
RoPEHeadGptJ 8, SiluMul 10, SparseMlaSplit 176, StopCheck 1, TokenSelect 2.

## 3. GLM's families in two halves

Every family rank 0's Program has at (43, 2) and 4 layers [read r20261113-003336-60e6's `dump_l4_r0.json`]. Each half adds
its lowerings to `node_window.py` (`FAMILIES`) and its tests to `test_node_window.py`. Each half writes only its own
functions and appends its own tests, so the halves don't conflict.

- **Test fixtures.** The tests build φ inline (nothing in `benchmarks/` is imported). `glm_request(LP, layers, rank=0,
  moe=False)` builds rank r's registered-form request Program at LP prompt ids and 2 outputs. With `moe=True`, every
  layer is an MoE layer, so a 1-layer MoE test builds in about 3 s [test: `test_moe_layer`]. Keep LP = 32 (DeepGEMM's
  branch, M ≥ 32) and 1 layer: the module's tests run in about 12 s at 574 MiB [test].
- **A half's check** per family: the unit count by template against the arithmetic of its split; one wire from its
  writer to its reader (`(writer, reader, port)` in `circuit.wires`); its family gone from `census["unlowered"]`; and the
  families it fed gone from `census["free"]`. `settle` checks `check_proof_block` per template; `test_window_units`
  also calls it directly.

Splits are cited from `pipeline/serving_templates.py`: `COORDS` :22, `WIDE_COORDS` :28, `COLS` :35, `SLICES` :46,
`PARTS` :68, `FINAL` :87, `KCHUNKS` :147, `CHAINS` :158, `STAGES` :171, `EXPERTS` :182, `FUSED` :186, `SLOTS` :189,
`HISTORY` :195, with widths in `unit_templates` :388-480.

### 4a: dense

| family (nodes at 4 layers) | split | existing code to call | test the half adds |
|---|---|---|---|
| DeepGemmFp8Linear, FlashInferFp8Linear (20 + 20) | `PARTS` (quantiser, then `COORDS` GEMM) | done: `block_fp8_linear` | (done) `test_window_units` |
| RMSNormCudaVector8, RMSNormFusedCudaVector8, DsaRMSNorm (2 + 8 + 8) | `PARTS` scale cut, `FINAL` outside layers | done: `scale_norm` | (done) `test_two_level_gather` |
| AllReduceRmsNormMnnvl (8, decode) | `PARTS`: `AllReduceResidualMnnvl_v1` (`COLS` N) → `RmsNormMnnvlSumSq_v1` → `RmsNormMnnvlApply_v1` (`COLS` G); `FINAL`: residual, then `RmsNormMnnvl_v1` | extend `scale_norm`'s shape. The residual's peer port is free; its own partial is wired | the decode's norm reads the residual's column blocks; the free list holds its peers |
| AllReduceMnnvl (1 decode, plus the prefill's at other lengths) | `COLS` N | `ST.unit_templates` halvings; peers free | its blocks wired to the next norm's SumSq through the gather |
| ResidualAdd (2) | `COLS` N | as above | blocks at G that the fused norm's apply reads without a gather |
| SiluMul (10) | the row itself (not in `COLS`) | `take(whole=True)` of gate_up's 3072 coordinates (two levels: K = 96, 32 chunks) [inferred from §2.3] | down_proj's quantiser x wired; `SiluMul` gone from `free` |
| MoeSum (2) | `COLS` H | per `TOPK` segment, `take(..., most=)` | its blocks read the expert coordinates |
| EmbeddingShard (2) | `SLICES` (H) | `pipeline/slice_rows.py`; `window_circuit._Embedding` shows the table read (`b.table`, `b.roots`, `b.openings`) | the embedding's 6144 columns gathered into the first norm's x. **Check first:** a column slice reads a strided weight column, which `_Builder.read`'s contiguous rows don't take [inferred] |
| Gemm_v2 LM head (2) | `SLICES` (N) | `slice_rows`; unit n reads x whole (the final norm's `FINAL` row) and its weight row | 19,360 one-word slices gathered into `AllGather_v1{8,19360}`'s own part |
| AllGather_v1{8,19360} (2) | `SLICES` (WORLD) | its own part from the LM head (two-level gather); the 7 peers' parts free | `TokenSelect` reads its 8 slices |
| RoPEHeadGptJ (8, batch n = 8) | the row itself | each head's rope columns [192, 256) of q_b, 64 coordinates (one level) | `BatchedGemm`'s / SparseMla's read of it wired (with 4b) |
| TokenSelect_v4 (2), StopCheck (1) | `STAGES` holds `TokenSelect_v1` only, so v4 is the row itself | one unit per step; the served select `GreedySelect_v1` is piece 5, not 4a | the step's end; nothing free past it |

### 4b: DSA and MoE

| family (nodes at 4 layers) | split | existing code to call | test the half adds |
|---|---|---|---|
| DsaLatentRow (8) | the row itself; `HISTORY`'s writer | `take(whole=True)` of q_a's y[2048:2624], 576 coordinates (two levels) [inferred from §2.3] | SparseMla's key blocks read its rows |
| BatchedGemm (16) | `SLICES` (B, N) | `slice_rows`; unit (b, n) reads head b's 192 nope coordinates (≤ 256 rows, or gathered) | o_proj's quantiser x wired; `BatchedGemm` gone from `free` |
| SparseMlaSplit (176: 43 prefill tokens + 1 decode, per layer) | `CHAINS` (block, end, scores, values, cols, combine, split) | `pipeline/chain_rows.py` (`plan`, `places`, `units`, `cuts`), `sparse_mla_split.CUTS`, `ST.chain_ways`; `window_circuit._Attention`/`_cache`/`_block` show a chain over key blocks | one token's chain wired from the latent rows and q, at LP = 32 |
| MoeRouterSigmoidBias (2) | `SLOTS` → `MoeRouterSigmoidBiasSlots_v1` | `ST.unit_templates` | the expert blocks' table keys read its slots |
| CublasSplitKGemmF32Out (1), LlBf16DotprodGemm (1): the router gate | `SLICES` (N) | `slice_rows` | the router reads its 256 slices |
| MoeExpertFp8Gemm, MoeExpertFp8GemmW (2 + 2) | `COORDS`, `KCHUNKS` (`MoeExpertFp8CoordinateChunk_v1`), `EXPERTS` (block GEMMs and `EmbeddingRead_v1` table reads), `FUSED` | `pipeline/kchunk_rows.py`, `pipeline/expert_rows.py`, `pipeline/expert_tables.py`, `ST.block_rows`, `ST.fused_block`, `ST.chunk_widths`; tables via `b.table` / `b.roots` / `b.openings` as in `window_circuit._Embedding` | a 1-layer `moe=True` window with the experts' units; `MoeSum` reads them |
| Fp8GroupQuantUe8m0 as a node (4: the MoE input's K = 6144 and K = 256) | the row itself | `take(whole=True)` | the expert GEMMs read its q and s |
| the DSA indexer (`DsaMqaLogits`, `long_row.units`) | `SLICES`, long rows | `pipeline/long_row.py` | **not met at (43, 2)**: no indexer node at 43 ids [read r20261113-003336-60e6]. Its test needs a longer prompt, which is past this class |

## 4. Reads that neither wire form takes

A wire reads one writer's row whole. A port with several writers is either cut k ways along its outer elements or read
through `AllGather_v1` with cut parts [read verity/commitments/windows/interface.py:38-45, as cited in CIR-274].

- **Among the lowered families: none.** At (43, 2) and 4 layers, every read of a lowered unit is whole writer rows: the
  window settled [r20261113-005531-62da], and `Piece.rows` refuses any read that cuts a row by name. The same holds at
  LP = 32 with 1 layer, dense and MoE [test].
- **Met: `AllReduceMultimem{WORLD=8,SIZE=264192}`'s output, read by token.** Nodes 1, 54, 59, 112, 117, 170, 175, 228
  and 242 are one call each over all 43 tokens (SIZE = 43·6144). Each reader member t reads words [t·6144, (t+1)·6144)
  of it:
  - node 2 `RMSNormCudaVector8` member t, port x;
  - nodes 55, 60, 113, 118, 171, 176 and 229 `RMSNormFusedCudaVector8` member t, port x (node 55 also res, from node 1);
  - node 243 (the final norm), words [258048, 264192) [read r20261113-003336-60e6's dump].

  As one writer row (the unsplit call), each of these is part of a row. As `COLS` blocks, `unit_templates` gives widths
  264192/k for k = 2, 4, …, all 129·2^m [read serving_templates.py:444-450]. None of them puts a block edge at t·6144
  for t = 1..42 (43 is prime, and 6144 = 3·2¹¹), so every member's read still cuts a block. The frame has these rows
  free for now, so `settle` does not refuse them yet. They come back when piece 2's combine outputs replace them.
  - The cure is not a fused Definition. It is block widths that divide 6144, i.e. `COLS` widths along the token
    stride (a change to `serving_templates.unit_templates`, vllm's file), or the combine's outputs cut per token in
    `sites_circuit`.
  - Separately, `serving_templates.cols_refused` raises on this id ("TREE '(': a node is a rank or a pair of nodes, not
    '('"), and `unit_templates` returns nothing for it: rank 0's `AllReduceMultimem` has no serving template at this
    class [read: local call of both on the dump's id, this session].
- **Not yet known [inferred]:** whether SparseMla's key blocks read `DsaLatentRow`'s 576-word row whole, or its 512 and
  64 parts apart (the case CIR-271 named). `settle` will name it once 4b lowers the chain.

## 5. What is not established

- 78 layers: the window was built at 4 layers only. At 78 layers the linears' units grow about 20x, roughly 70 M units
  [inferred]. Expected peak: well under 64 GB from 0.53 GiB at 4 layers [inferred]. Per the memory rule, it is not run
  before the lead's yes.
- Ranks 1 to 7: built for piece 1 but not lowered.
- That `coord=1` is the grain serving commits: `WIDE_COORDS` lets the block-scaled GEMM be c outputs wide. The frame
  takes `coord=c` for c dividing 128, untested.
