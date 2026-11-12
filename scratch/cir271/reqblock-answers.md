**REQBLOCK, answers to proofs' three questions (CIR-271).**
- **(1)** One unit of the node's request Program at (256, 32) is a Program row: one call of a registered template, such as a whole MoE expert GEMM over a token row, not one proof block. The Program has 10,607,496 such calls over eight ranks, against the card's 11,275,271,992 proof blocks.
- **(2)** No row gathered from many writers is committed as one writer's output row today. In the Program as built, a gathered operand is a `Concat` of other nodes' outputs, with no node of its own. Several serving cuts leave a whole-row reader one row per writer. Round 2's wires refuse such a row except through a wire cut per writer or a gather unit.
- **(3)** Yes: in the node Program every cross-rank partial is a value that one rank's node returns and another rank's collective reads, inside one body Definition. None of its 18,201 parameters is a partial. The per-rank registered form still takes 35,392 partials as `peers_*` Inputs.

The figures come from my runs on vy-nebius-cpu-13, over `main`'s code at `5f0c008aa04`: r20261112-202908-e9ba (eager) and r20261112-203414-18cf (compiled), both at 256 prompt ids and 32 outputs, which built one digest, `3b8107b44d08a34d…`. [run] means one of those runs shows it; [read] means I read it in code; [inferred] means arithmetic or a reading that no run checked.

1. **What one unit is.**
   - The node body holds 10,655,712 nodes: 10,607,496 calls of 629 template bindings (29 Definitions) and 48,216 primitives, 1,331,964 nodes a rank [run]. The root is the parameters plus one call of the body (`adapters/vllm/verity_vllm/program/predict/node.py:18-19`, `262-273`) [read].
   - Rank 0's registered form (`steps.request_program`) rebuilt in the same runs has your d1f1 digest `615bc15c…`, with 330 templates and 1,267,135 instances [run]. Your account of it to proofs stands.
   - The census cuts each row into proof blocks: its own Definition where it fits, else one unit template per slot of its split (`benchmarks/one_stage/block_census.py:1-16`, `116-139`) [read].
   - One family shows the ratio exactly. The node's 1,377,600 `MoeExpertFp8GemmW` calls [run] × 6,144 output coordinates = 8,463,974,400, which is exactly the card's two `MoeExpertFp8GemmW_v1{…N=1…}` rows (7,549,747,200 + 914,227,200, `benchmarks/sampling_plan/glm53_blocks.json`) [read; the product is inferred]. Averaged over all rows, that is about 1,063 proof blocks a call [inferred].
   - For the Definition's units to be proof blocks, each row would have to be emitted as its serving cut (`pipeline/serving_templates.py:16-182`: COORDS coordinates, COLS column blocks, SLICES, PARTS, KCHUNKS, CHAINS, STAGES and EXPERTS). Every unit would be a call whose operands are each one whole parameter row or one whole output of one unit, as `pipeline/long_row.py:1-3` and `12-15` emits `DsaTopKReadRows_v1` as its cut for `Layout` to take `inline` [read]. No builder emits that for the request Program today [read].

2. **Who commits a row that many blocks wrote.**
   - **The Program as built:** a call's operand is a `Coll` whose refs may concatenate slices of many nodes' outputs. A collective's operand, for one, is every rank's rows concatenated (`program/predict/common.py:197-204`) [read]. A gather is wiring, with no node and no gate, and the census counts a gather row as no proof block (`block_census.py:15-16`) [read]. The Program commits nothing itself: the serving rows commit each member's ports, read as the replay reads them (`pipeline/serving_rows.py:1-3`) [read].
   - **Splits that leave a whole-row reader one row per writer** (`serving_templates.py`) [read]:
     - COORDS: one output word a unit (`17-23`).
     - WIDE_COORDS: c words a unit (`24-28`).
     - COLS: column blocks, which include the all-reduces `AllReduceMnnvl` and `AllReduceMultimem` and `MoeSum` (`29-38`).
     - SLICES: one slice a unit, which includes `AllGather` by rank and the LM head (`39-50`).
     - The norms' PARTS: apply blocks, "so that no unit returns a whole row" (`62-66`).
     - EXPERTS: N/n blocks (`177-182`).
   - **Splits that end in a unit writing its output whole** [read]:
     - The row itself, uncut.
     - A STAGES row's last stage: the greedy select's `GreedyCombine` writes the token (`164-172`).
     - A norm's FINAL cut outside the layers, the row whole (`82-89`).
     - The quantising part of `DeepGemmFp8Linear`, whose q and s come out whole while its GEMM part stays in coordinates (`71-72`).
     - KCHUNKS and CHAINS end per coordinate, head or column block, not per row (`144-160`).
   - **What the wires accept:** `boundary_rows.Layout` declares no wires itself; it is a population of one check template (`pipeline/boundary_rows.py:339-345`) [read]. An `inline` class gets its replay block's wires from round 2 (`interior_rows.py:340-343`) [read]. `check_wires` takes only one writer's whole output row: "a row gathered from several writers is cut or read through a gather unit" (`verity/commitments/windows/interface.py:808-813`, and `38-45`: a port read from several writers is cut k ways, one writer a row, or read through an `AllGather_v1` whose parts are cut) [read]. So proofs' reading holds, with those two exceptions.

3. **The cross-rank partials.**
   - `build_node_program` runs the eight forwards into one Builder. At each collective, each rank posts its own rows and takes every rank's rows in rank order (`node.py:117-133`, `147-156`), and its call reads them as one `Concat` (`common.py:197-204`) [read]. So each partial is a node's output read by another rank's collective call in the same body.
   - Parameters: the prompt and 18,200 θ weight fields (`node.py:191-198`), and none other [run: the parameters other than the prompt and weights came out as none]. Rank 0's registered form takes 35,392 `peers_*` Inputs, its world − 1 per collective (`build.py:732-736`) [run].
   - The collectives at (256, 32) over the eight ranks, with what each call reads [counts run; reads read in `program/predict/dsa.py`]:
     - `AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=1,CLUSTER=8,…}`, 38,688 calls: per token row, each rank's 6,144-value partial of that row, plus the calling rank's residual and norm weight (`dsa.py:396-408`).
     - `AllReduceMultimem_v1{WORLD=8,SIZE=1572864,…}`, 1,256 calls: each rank's whole partial over the launch's rows, rank-major; 1,572,864 = 256 × 6,144 (`dsa.py:377-394`).
     - `AllReduceMnnvl_v1{WORLD=8,N=6144,SHOT=1}`, 248 calls: each rank's whole partial of one row (`dsa.py:377-394`).
     - `AllGather_v1{WORLD=8,N=19360}`, 256 calls (8 ranks × 32 samples): each rank's 19,360-logit vocabulary shard of the sampled row (`dsa.py:369-374`).
   - The per-rank call counts fit two fused all-reduces a layer in each of 31 decode steps (38,688 / 8 = 4,836 = 78 × 2 × 31), and 2 a layer plus 1 in the prefill (1,256 / 8 = 157) [inferred].
   - So in the node Program the partials are values of one Definition. As round-2 wires, each collective's operand is still a row from 8 writers, so it needs the wire cut or `AllGather`'s gather form (`interface.py:38-45`) [read]. Under the serving cut each rank's partial row is itself written by many units, since an `o_proj` or MoE row is split into coordinates, so the cut would be finer still [inferred].

**Not established:**
- Which primitive the body's 48,216 primitive nodes are.
- Whether every card type is at k_log 27 or less: I did not check the card.
- That no builder on any branch emits the request Program as its cut: I read `main` only.
- That the per-writer cut of a collective's operand would pass `check_wires` at these sizes.

Parts 2 and 3 of my brief, and the 64-output builds (eager and compiled, both done), follow with my report by 5:30 PM PST.

circuits worker REQBLOCK, Thu 12 Nov 1:58 PM PST.
