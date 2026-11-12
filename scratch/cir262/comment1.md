circuits' worker FP8READ, Thu 12 Nov 7:27 AM PST (15:27Z), first comment. **Draft names, statics and ports for FlashMLA's FP8 sparse read at GLM-5.3's sizes are below. One position is `FlashMlaV32Decode_v1`, and a row of 1 + K positions calls it once a position. None of the five registered FlashMLA Definitions holds here with new statics alone, though the block's, the split end's and the combine's arithmetic carry over by source. circuit-check held 18 targets at these sizes with 0 failures. Nothing has run on a card yet: the kernel's own run is today from 10:05 AM PST, so every sentence on arithmetic below is read from source, not seen.**

**The Definitions (draft; the values are GLM-5.3's at 8 ranks on the H200)**

- **`FlashMlaV32Decode_v1{NH, SPLITS, DN, DR, QT, BN, DOT, SCALE}`**, one position's attention output. Bound at NH = 8, DN = 512, DR = 64, QT = 128, BN = 64, DOT = `HopperBF16WgmmaDot16_v1`, SCALE = 0.0625, and SPLITS by the step (see below). N = 2048 rows.
  - In: `q`: NH × (DN + DR) bf16. `nope`: N × DN e4m3. `scale`: N × DN/QT f32. `rope`: N × DR bf16. `idx`: N i32.
  - Out: NH × DN bf16.
- **`FlashMlaV32KvRow_v1{DN, DR, QT}`** at (512, 64, 128), one cache row as the kernel stages it.
  - In: `nope`: DN e4m3; `scale`: DN/QT f32; `rope`: DR bf16; `idx`: i32.
  - Out: a staged row: DN + DR bf16 and one valid bit.
  - The three byte ports in that order are the row's 656 bytes. They match the read-back's ports (`Fp8DsMlaUpconvertRow_v1`) and the write's outputs as the lead gave them on CIR-246 at 6:30 AM PST; I read neither pull request.
- **`FlashMlaV32Block_v1{DK, DV, BN, FIRST, DOT, SCALE}`** at (576, 512, 64, true or false, `HopperBF16WgmmaDot16_v1`, 0.0625), one block of 64 rows for one head.
  - In: `q`: DK bf16; `k`: BN staged rows; and unless FIRST the state: `nm_prev`: f32, `l_prev`: 4 f32, `o_prev`: DV f32.
  - Out: the state: `nm`: f32 (the running maximum, negated), `l`: 4 f32, `o`: DV f32.
- **`FlashMlaV32SplitEnd_v1{DV, PARTIAL}`** at (512, true or false), a split's end.
  - In: `nm`: f32; `l`: 4 f32; `o`: DV f32.
  - Out: DV bf16 where the position has one split; else (PARTIAL) `oacc`: DV f32 and `lse`: f32.
- **`FlashMlaV32CombineWeights_v1{NS}`**, NS the number of splits (2 to 32). In: `lse`: NS f32. Out: NS f32.
- **`FlashMlaV32CombineValues_v1{NS, G}`** at G = 256. In: `w`: NS f32; `oacc`: NS × G f32. Out: G bf16.
- **`FlashMlaV32DecodeHead_v1{SPLITS, DK, DV, BN, DOT, SCALE}`**, one head. In: `q`: DK bf16; `k`: N staged rows. Out: DV bf16.

The module is `verity/definitions/registry/flashmla_v32.py`. "V32" is FlashMLA's name for the kernel instance (`ModelType::V32`), not a model's.

**A row of 1 + K positions can call them.** The kernel takes every position as its own query row with its own 2048 indices, so a request of 1 + K positions is 1 + K calls of `FlashMlaV32Decode_v1`. One thing a row cannot know alone: SPLITS.

- **SPLITS is the step's, not the row's.** vLLM hands the kernel one request of T query rows, T the step's tokens on this route. The planner cuts each position's 32 blocks by T alone: 32 splits at T ≤ 3, 16 at T 4 to 6, down to one split from T = 67 [computed from the planner's source by my reference; the card confirms it today]. One request of 1 + 5 positions alone is T = 6, so 16 splits. Two such requests are T = 12, so 8.
- So a step's record has to carry T (or SPLITS), as it carries the route.

**Least sure to stay, most first**

1. **The cut's unit names and G.** `CombineWeights`, `CombineValues` and G = 256 exist only because 32 partials over 512 columns are past one Flock block. If your roles name the units a position is staged as, these are the ones that could move.
2. **`idx` as an i32 port of the row.** The row reads only whether it is -1. A one-bit `valid` port is the other form.
3. **The row as three ports, not one 656-byte port.** It follows the write and the read-back; if one moves, all three do.
4. **`QT`.** The write calls the same number `TILE`; the read-back has no static for it.
5. **The values of SPLITS**, until the card shows the planner's table.

The token's name, its statics' names and its five ports I expect to stay.

**Which requests reach this kernel [read from vLLM `d9105ea8001e`, not run]**

- **A "decode" on this backend is wider than a request generating one token.** The builder's threshold at 8 heads on sm_90 is 256 query tokens, or 1 + K if larger (`flashmla_sparse.py:279-291`, `v1/attention/backend.py:627-657`). Decodes must be uniform (`:265`), so a step's decodes are its leading requests whose query is as long as the first's, when that is at most 256 (`v1/attention/backends/utils.py:768-839`).
- **So a request whose prompt is at most 256 tokens, leading its step, is read by this kernel and never by the dense prefill.** T is then its whole query.
- Every other request of the step is a prefill. Its tokens take the dense prefill when the step's longest prefill is at most 2048 (`index_topk`), and join this kernel's call otherwise (`sparse_mla_attention.py:265-310`, `mla_attention.py:890-898`).
- This narrows the lead's line of 6:13 AM PST on CIR-246, "the step's longest prefill": that decides only among the requests the split above already calls prefills.

**The five registered Definitions at this instance [from source and circuit-check; the card's SASS is still to come]**

- **`FlashMlaRowValid_v1`: does not hold.** It is "index not -1 and position below the length". This instance has no length, so validity is the index alone, inside the new row.
- **`FlashMlaKvRow_v1`: does not hold.** Its scale is one UE8M0 byte. Here a row has four f32 scales, each rounded to bf16 before one bf16 multiply.
- **`FlashMlaBlock_v1`: its arithmetic holds, its shape does not.** It has one width for key and value. Here the key is 576 wide and the value its first 512, so at D = 576 it carries 64 columns the kernel never forms: 384 dead gates a block in a head, in my earlier run `r20261112-143014-4013` [read]. The new block calls the registered body with the value's width.
- **`FlashMlaSplit_v1`: its end holds on a constant sink of -inf.** A split of two blocks is already past one Flock block (2^28 bits, same run), so the end is its own unit.
- **`FlashMlaCombine_v1`: its arithmetic holds with the sink term absent.** The call passes no sink and the kernel skips the term. At 32 splits and 512 columns it needs 2^28 bits (same run), so it is cut into weights and values. At 16 splits it fits.
- **The 56 pad heads** have no Definition: their queries are zero and their outputs are sliced away. They enter only the planner's divisor (64 heads).
- **Not modelled at all:** the index conversion from a request's positions to cache slots.

**circuit-check, run `r20261112-150200-2bd9` [read]: 18 targets, 0 failures**

- The sidecar's 12 bindings: 0 failures, 0 unchecked, 0 dead gates, 0 mismatches against the reference on random vectors.
- One head at 32, 16 and 1 splits and two combines at served sizes: 5 targets, 0 failures.
- One whole position at 16 splits: 0 failures, 1 warning. It is 10,960,944,280 ANDs, cut into 2048 rows, 256 blocks, 128 ends, 8 weights and 16 values. Only 3 edge vectors were compared, with 0 mismatches, and the partition check did not run on the whole (5,899,984 gates, over its budget of 5,000,000).
- Every unit fits a block of 2^27 bits. The largest is the later block: 37,826,567 ANDs, 123,469,824 bits.
- The run waited 84 s for a check slot and held it 17 min 33 s.

**An edit inside a registered module, said before I push it**

It changes no registered Definition's words: the program digests of 13 registered FlashMLA bindings are equal before and after [computed on my VM], and the adapter's 35 quick FlashMLA program tests pass.

- `verity/definitions/registry/flashmla.py`, lines at `main` `5c0ae25d6c28`:
  - `block()` :288-320 gains a keyword `DV`, the value's width, default D. Lines 314 and 318 use it.
  - `split()` :344-362: its end (:353-362) moves into a new `split_end()`, which `split()` calls.
  - `combine()` :389-417: the part before the weighing loop moves into a new `combine_scale()`, with the sink's lines (:407-410) under `if sink is not None`.
- `verity/definitions/numerics/flashmla_twin.py`: `combine()` :264-290 splits the same way.
- It goes in a commit of its own. The other form is a second copy of about 50 lines in my module.

**The two reads of one cache row [by hand, not yet on the card]**

By the source they differ. This kernel rounds the scale to bf16 and multiplies in bf16; the dense prefill's read-back multiplies in f32. At scale 1 + 2^-8 + 2^-20 and value 1.75 this kernel gives 1.765625 and the read-back 1.7578125, one bf16 step apart. Today's run plants that row.

**Where the source differs from NEWPIN's note**

- "A request of at most 2,048 tokens stays on dense prefill": the choice is the step's, and a request of at most 256 query tokens can skip the dense prefill altogether (above).
- Its list of what is missing is short. Also missing were the two widths (576 and 512), no sink and no length, and units that fit a block.
- The sparse backend's own read-back (`flashmla_sparse.py:797`) does not run below 32 heads. The read-back that runs is the dense prefill's (`mla_attention.py:2834`, `:2955`).

**Sat 14 Nov 8 AM PST holds as it stands.** The Definitions, their tests, the bindings and the capture script are written. What can move it is today's card run finding the kernel differs from my reference.

Not done: no card run, so no SASS, no capture and no planner table from the kernel. The pins update pass has not run. No Boolean `_v2` forms. Under a CUDA graph T may be the graph's padded token count; I did not read that.
