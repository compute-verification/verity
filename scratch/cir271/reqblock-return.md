**REQBLOCK (CIR-271), sizing the serving-cut builder. The smallest class that keeps all 29 Definitions is 43 prompt ids and 2 outputs (a prefill and one decode step), at all 78 layers.**
- Its node Program is 1,627,200 calls in 142 bindings, built in 50.9 s at a 2.65 GiB peak (run r20261112-234945-6a7d). It is about 1.70 billion proof blocks, inferred from the card's blocks per call.
- The collectives already lower to units that pass both checks at one and two prompt tokens: `pipeline/tp/wires.sites_circuit`.
- `window_circuit`'s `_Builder` is the frame to extend, but it knows only Llama's 10 families.
- The day for the whole builder is Thu 19 Nov 6 PM PST [Fri 20 Nov 02:00Z], and Mon 16 Nov 6 PM PST [Tue 17 Nov 02:00Z] for a first cut: rank 0 with every Definition but the collectives.

All runs were on vy-nebius-cpu-13, from `scratch/cir271/` on `cursor/circuits-cir271-scratch-6f04`, over `main`'s code at `5f0c008aa04`. Each runs `build_node_program` eager with one request declared in φ's workload (part 1 below says why). I changed no builder.

1. **The smallest class that keeps every Definition: (43, 2), at 78 layers and eight ranks, with a prefill and one decode step.**
   - The ladder (`smallest.py`, runs r20261112-233124-a438, r20261112-234639-9638 and r20261112-234945-6a7d, at 3 or 4 layers, then all layers for 42 and 43). Each line counts the 29 Definitions of the (256, 32) Program that are missing at that shape:
     - Fewer than 4 layers: 8 to 11 are missing at 3 layers, including the router, the MoE GEMMs, MoeSum, ResidualAdd and the quantizer. Layer 3 is the first MoE layer.
     - A prefill alone (T 0): 4 are missing at 64 prompt ids, 4 layers: `AllReduceMnnvl`, `AllReduceRmsNormMnnvl`, `FlashInferFp8Linear` and `LlBf16DotprodGemm`. So one decode step is needed.
     - 42 prompt ids or fewer: `AllReduceMultimem` and `RMSNormFusedCudaVector8` are missing (r20261112-234945-6a7d, at 4 layers and at all layers). At 43 none is missing.
     - The cause is the route: a prefill of M rows takes the fused MNNVL all-reduce while M ≤ `fused_max_tokens(6144, 8)` = 524,288 // 12,288 = 42 (`verity/definitions/registry/tp_allreduce.py:131`, `:258-263`; `frontend/rules/vllm_bindings/collectives.py:152`). Past that it takes multimem plus the fused CUDA norm.
     - Under 32 prompt ids a 30th Definition, `LlBf16SplitKGemm`, appears (at 8 and 16, r20261112-233124-a438). It is not in the (256, 32) Program.
     - At all 78 layers (r20261112-234536-99bb), with T 1:
       - 1 prompt id: 73,968 calls, 4 Definitions missing;
       - 16: 628,248 calls, 4 missing;
       - 64: 2,403,192 calls in 184 bindings, none missing;
       - 256: 9,497,976 calls in 569 bindings, none missing.
   - Eight ranks: one rank holds all 29 Definitions. The node's 629 bindings at (256, 32) are rank 0's 622 plus seven more `EmbeddingShard` bindings, one per rank (run r20261112-202908-e9ba). The eight ranks are needed for the collectives' wires: with fewer, every other rank's partial is a free input (`pipeline/tp/wires.py:36-37`).
   - Select: greedy has one select path. The node calls `TokenSelect_v1{V=154880}`. The served registration (VLLM-412) checks `GreedySelect_v1{V=154880,N=7744}`, which is not on `main`. No second path is needed.
   - Kinds of serving cut at (43, 2), from its families (run r20261112-234945-6a7d): CUT, COORDS, COLS, SLICES, PARTS, KCHUNKS, CHAINS, STAGES and EXPERTS, plus rows that are their own unit. WIDE_COORDS and FINAL come in through PARTS: DeepGemm's GEMM part is `ScaledMmFp8BlockSharedScale`, and the final norm is outside the layers (`serving_templates.py:16-89`; inferred from the dictionaries, not from a unit list).
   - The count: the Program has 1,627,200 calls, 1,634,592 body nodes and 142 bindings (r20261112-234945-6a7d). At 42 prompt ids it has 1,589,000 calls and 140 bindings.
   - Proof blocks: about 1,700,949,600, inferred and not shown by any run.
     - The estimate (`blocks_estimate.py`) is the calls by family times the card's blocks per call at (256, 32). It holds on a check: at 4 layers it gives `MoeExpertFp8GemmW` exactly the card's count over 75 MoE layers.
     - The largest terms are `MoeExpertFp8GemmW` 1.30 billion (6,144 a call), `BatchedGemm` 169 million, `MoeExpertFp8Gemm` 108 million and `DeepGemmFp8Linear` 70 million.
     - It overstates `AllReduceMultimem` (20.6 million), because its blocks scale with the prompt.

2. **What exists to build on, by split** (`adapters/vllm/verity_vllm/pipeline/serving_templates.py`). Two pieces of code already emit units as calls whose operands are each one writer's whole row, and pass `check_proof_block` and `check_wires`:
   - `program/predict/window_circuit.lower` (`:712-740`) over `window_builder._Builder`: `unit` `:158`, `feed` `:175`, `weight` `:196`, and `settle` `:244-280`.
     - `settle` cuts each port once per writer row and runs `check_proof_block` on every template (`:274`). `Circuit` runs `check_wires` (`verity/commitments/windows/interface.py:1317`).
     - It lowers Llama's `FAMILIES` only (`window_circuit.py:672-673`), refuses peers (`:197-200`), and under class G stops at the launch (`:680`).
   - `pipeline/tp/wires.sites_circuit` (`:269`), over `pipeline/tp/grain.lower` and `program/tp_binding.derive` (`:188`).
     - It emits every collective site of the eight rank Programs: per-rank coordinate gathers into 96-word rows, one combine unit a chunk with its parts cut 8 ways, the chunks' gather, the fused norm, and the token select's groups plus combine (`wires.py:10-20`).
     - `tests/program/predict/test_tp_steps.py:83-108` asserts, for GLM-5.3 TP8 with one- and two-token prompts, no free read of a combine output and no unit failing `check_proof_block`. That test is opt-in (`VERITY_VLLM_CLASS_G_WINDOW=1`); I did not run it.

   | split | emits calls today | names or lays out only | what a builder lacks |
   |---|---|---|---|
   | COORDS (`:22`): MoE expert GEMMs, FlashInfer, `ScaledMmFp8BlockSharedScale` | `tp/grain.py:5-12`: only coordinates that produce a collective's partial | `coordinate` `:486`, `coord_widths` `:343`, `part_rows.wire` `part_rows.py:334` (within a row) | q/kv projections and MoE coordinates as calls, with their operands: the quantized row whole, and weight rows (`weight_rows.cut` `:133`) |
   | WIDE_COORDS (`:28`) | none | `coord_widths` `:343` | the same at c outputs |
   | COLS (`:35`): MoeSum, ResidualAdd, all-reduces, the norms' apply | `tp/grain.py:7-9` (all-reduce chunks) | `cols_refused` `:303`, `col_span` `:141` | MoeSum, ResidualAdd and the apply blocks as calls |
   | SLICES (`:46`): LM-head GEMMs, BatchedGemm, DsaMqaLogits, EmbeddingShard, AllGather | `window_circuit` Gemm_v2 slices; `tp/wires` the logits' gather | `unit_templates` `:388` | the rest as calls |
   | PARTS (`:68`): fused all-reduce norm, DeepGemm and FlashInfer, the three RMSNorms | `tp/grain.py:9-12` (fused norm); `window_circuit` RMSNormFusedCuda (Llama) | `part` `:353`, `part_rows.wire` `:334` | the quantization/GEMM and sum-of-squares/apply parts as calls |
   | KCHUNKS (`:147`) | none | `kchunk_rows.units`/`wire` `:54`/`:130` (within a row) | the chunk chain as calls |
   | CHAINS (`:158`): SparseMla | `window_circuit` attention by key block (Llama only) | `chain_ways` `:367`, `sparse_mla_split.CUTS` `:549`, `chain_rows.units`/`wire` `:224`/`:364` (within a row) | SparseMla's head chain as calls |
   | STAGES (`:171`) | `tp/wires.py:16-20` (`TokenSelectGroups_v1`, `TokenSelectCombine_v2`) | `greedy_stages.cut` `:150`, `stage_rows` `:63`/`:185` | `GreedySelect_v1{V=154880,N=7744}` (on `cursor/proofs-gather-served-select-b88b` only) |
   | EXPERTS (`:182`) | none | `expert_rows.cut`/`wire` `:258`/`:280` (wires a read's key to its router slot) | the expert block GEMMs as calls |
   | DSA top-k reads | `long_row.units` `long_row.py:142` | | nothing |

   A builder would call `_Builder` through a node-by-node lowering like `window_circuit._Program.lower`, covering GLM's 29 families, and `sites_circuit` for the collectives. Today's `_Program` lacks GLM's families, peers and a class launch.

3. **Where the wires will not pass.** A wire reads one writer's row whole, and a port with several writers is cut k ways along its outer elements or read through `AllGather_v1` with cut parts (`interface.py:38-45`). A unit has at most 256 rows, one of them past 111 B, within 2^27 bits (`interface.py:307-311`, `:322-344`).
   - The collectives use both forms and pass at one and two tokens (read, not run). Per rank, the 6,144 coordinate words go through `AllGather_v1{WORLD=96,N=1}`. A combine reads 8 parts of 96 words, and `AllGather_v1{WORLD=64,N=96}` builds the row (`grain.py:5-12`). Inferred: at (43, 2) the multimem all-reduce is 264,192 words a rank (43 × 6,144) in more chunks, and fits.
   - A whole-row reader after a COORDS or COLS split (RoPE and the norms after the q/kv coordinates; the residual add after MoeSum) would see 6,144 one-word writers, past 256 for the cut form. It needs the same two-level gather. That fits, but nothing emits it outside a collective site.
   - Attention and the indexer read one writer row a key. At (43, 2) that is at most 44 rows, which fits; at (256, 32) it is 287, past 256.
   - Where neither form fits (inferred): in the node form, a key's 576-word latent row comes from two writers, `DsaRMSNorm` (512) and `RoPEHeadGptJ` (64). The cut form splits only between outer elements, and `AllGather_v1`'s parts are equal.
     - The registered rank form has one writer there, `DsaLatentRow{KVL=512,R=64}` (`verity/definitions/registry/dsa_norm.circuit-check.py:23`; 22,386 calls at (256, 32), run r20261112-202908-e9ba). So lowering the registered form removes this case.
     - Any read of part of a writer's row is the same case, which `wires.py:33-34` leaves as a free input. How many there are is not known until a `settle` names them.

4. **The pieces, in order:**
   1. The class Program at (43, 2), as the registered rank form. With one declared request it builds today: run r20261112-202908-e9ba built that form at (256, 31). vllm owns a class launch without a declared request (`steps.py:474-486`, `launch_context.py:133`). Done-check: the eight rank digests and `check_ranks`.
   2. The collectives: `compile_tp` plus `sites_circuit` at (43, 2). Done-check: `test_tp_steps.py:83` at that prompt, with unwired and proof blocks both empty. It needs only piece 1.
   3. The frame: `adapters/vllm/verity_vllm/program/predict/node_window.py`, `_Program` for GLM with peers taken from `sites_circuit`'s combine outputs. Done-check: `check_wires`, `check_proof_block` and a digest at one MoE layer (4 layers).
   4. The family lowerings, by two workers side by side once piece 3 fixes its interface:
      - (a) the dense families: DeepGemm and FlashInfer parts with `Fp8GroupQuantUe8m0`, the LM-head slices, `EmbeddingShard`, the norms' parts and columns, RoPE, SiluMul, ResidualAdd, MoeSum, and the part-3 gathers;
      - (b) DSA and MoE: the indexer with `long_row.units`, the SparseMla chain, BatchedGemm, the router, `expert_rows`'s block GEMMs and the KCHUNKS chains.
      Done-check: `check_proof_block` per template, and `adapters/vllm/tests/program/predict/test_node_window.py` at 4 layers.
   5. The served select, `GreedySelect_v1{V=154880,N=7744}`, once proofs' branch lands. Done-check: its circuit-check report.
   6. Assembly at (43, 2) with 78 layers: `check_wires`, `check_proof_block` on every template, the window digest, and a unit census against part 1's estimate. It runs remotely. Two workers can split pieces 2 and 3, and then 4a and 4b.

5. **The day.** The whole is Thu 19 Nov 6 PM PST [Fri 20 Nov 02:00Z]; a first cut, rank 0 alone at (43, 2) with all 29 Definitions and the collectives' peer rows left free as in today's per-rank windows, is Mon 16 Nov 6 PM PST [Tue 17 Nov 02:00Z].
   - It rests on four things:
     - two workers from Fri 13 Nov;
     - vllm taking a new module under `program/predict/`;
     - proofs' `GreedySelect_v1` landing by Tue 17 Nov;
     - cpu-13 for runs: the (43, 2) node builds in under a minute at 2.65 GiB, but the window will be larger.
   - The least-trusted part: the count of reads neither wire form takes (part 3). It is unknown until a `settle` names them, and each one needs either a writer re-cut or a fused Definition like `DsaLatentRow`.

**The rest of the return, one line each:**
- Part 1 (the node Program at (256, 32)):
  - It builds eager and compiled to `3b8107b44d08a34dfa0c…` (r20261112-202908-e9ba and r20261112-203414-18cf): 5.7 min, an 8.4 GiB peak, 10,655,712 body nodes and 629 bindings of 29 Definitions.
  - Each of its eight rank spans passes `check_ranks`.
  - At 64 outputs it builds to `9945cab7165ae8f6a0cc…`, eager (r20261112-210018-493e and r20261112-210018-837d) and compiled (r20261112-212333-94fd).
  - It refuses without a declared request at `launch_context.py:133` (`KeyError 'sweep'`).
- Part 2 (partition): no `TemplatePartition` over the node's templates can be formed, because its root is one call of the node body (`QueryInapplicable`, `partition_object.py:317-345`).
- Part 3, between this Program and one registered class Definition:
  - stop rule: count and finish are `StopCheck_v1` statics (`stop_rule.py:93-102`), and the node has none (`node.py:254`);
  - form: the node has no dead_free, batched or table_reads (`build.py:750-771`);
  - lengths: `class_program` refuses under class G (`window_circuit.py:680`, r20261112-203331-aaaf);
  - launch: the key includes the batch's launch (`steps.py:223-235`);
  - ports: no `b.request`, `b.prompt` or `b.egress` (`boundary_rows.py:100-118`);
  - Layout: one check template whose calls sit at the root (`boundary_rows.py:325-345`);
  - the three conditions: condition 3 is met only by the node, which joins its ranks without peers but is registered nowhere (`node.py:266`).
- Builders: none holds positions past a count by gates on class G. `SlotEntry_v1` does (`io_shell.py:538-569`), but only Llama's window binds it (`window_circuit.py:728`).
- Peer partials: the node Program has no parameter that is another rank's partial (r20261112-202908-e9ba). The registered rank form has 35,392.
- Select: the builds call `TokenSelect_v1` (node) and `TokenSelect_v4` (rank form), not `GreedySelect_v1{V=154880,N=7744}`, which is not on `main`.
- The padded Definition: vllm writes it, as the class Definition at the served registration's (P, 64) per VLLM-412's rule (a), with its count committed and ids past n zeroed in gates. The builder's first test is at (43, 2). This is my recommendation, not a ruling.
- VLLM-412, read: the served registration differs from compile_cost's in:
  - compiled with FlashInfer and Model Runner V2, where compile_cost names triton MoE and b1-eager-v3;
  - `GreedySelect_v1` against `TokenSelect`;
  - stop ids [154820, 154827, 154829] at T 64 against ignore_eos at 32 or 64;
  - 8 requests against one, so the launches differ;
  - seed 0.
  I did not rebuild at it: this sizing came first.
- Size: the census's 11,275,271,992 blocks at (256, 32) reproduce from its per-family figures. The (43, 2) class is about 15% of it (inferred).

**Not established:**
- That the collectives pass at (43, 2): `test_tp_steps` asserts it at one and two tokens only, and I did not run it.
- The proof-block count at (43, 2): it is an estimate.
- How many partial-row reads a GLM lowering meets.
- That WIDE_COORDS and FINAL appear: read from the dictionaries.
- Why d1f1's root count (76,783) differs from 114,451.
- Whether a fixed launch gives two spans one Program.

The statement I trust least is the day. It rests on piece 3's interface holding for 21 families that no lowering covers yet, and on the count of reads in part 3, which no run has measured.

@@STAMP@@
