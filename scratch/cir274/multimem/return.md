**REQBLOCK's return, CIR-274: piece 1 (the mypy fix on #9627) and piece 2 (AllReduceMultimem lowered in rank 0's node window).** Fri 13 Nov 12:30 AM PST [08:30Z]. Tags: [read RUN] means I read it from that run's record; [read git] or [read gh] means I read it from git or gh this session; [inferred] means I did not measure it.

**What happened to this machine.** The 07:58Z stop left it intact. It is the same machine (up 24 days), all trees were clean at their pushed heads, and nothing was lost [read uptime/git]. Coordinator messages reached me at my next shell command: 07:18Z (the moves), 07:42Z (INF-783 rules) and 08:10Z (push a checkpoint). No timer, Slack or Linear event reached me.

**#9364 merged into `main` at 08:15:20Z** [read gh]. Per the brief I did nothing to #9627.

### Piece 1: #9627 (`cursor/circuits-cir274-tp-batch-reader-6f04`)
- **Head:** `3d00e506a3c9f88125e58a7662b564a8c2966e5f`. One commit on `adapters/vllm/tests/program/predict/test_tp_binding.py`, with no merge of `main` and the import block untouched [read git].
- **mypy** as the lint runs it: "Success: no issues found in 1 source file" [read r20261113-050139-29d2].
- **That file's tests:** 9 passed [read, this session].
- **Ratchet gate:** `failures: []` [read, this session].
- **Suites** `--quick --fresh verity-vllm repository`, run r20261113-050213-ed4b at `3d00e506`, both suites passed [read r20261113-050213-ed4b]:
  - repository: 388 passed;
  - verity-vllm: 14,603 passed, 479 skipped.
- **PR body:** the Runs and Lint sections carry the new head and these run ids; the collect runs' lines say they ran at b90c374b's library text.

### Piece 2: branch `cursor/circuits-cir274-multimem-6f04`, head `5005ee9e7e27a930dad569aba4021e7d11f22cc5` (from DENSE's `30dac672`)
**Files:**
- `adapters/vllm/verity_vllm/program/predict/node_window_multimem.py` (new);
- `adapters/vllm/tests/program/predict/test_node_window_multimem.py` (new);
- one docstring line in `node_window_values.py` (the `frame:` commit below).

There are no hunks in vllm's files on `main`; `pipeline/tp/grain.py` and `tp_combine` are imported, not changed. No file sits under `verity/definitions/`, `verity/protocols/`, `verity/hardware/`, `verity/configuration/` or `security/` [read git]. No lint-list entry was added. The ratchet gate reads `failures: []` at each pushed head (28753ebd failed it, see the `frame:` commit; 63cc12a8, 4288774e, 26d5e1ae and 5005ee9e pass) [read: `ratchet_gate.py origin/main HEAD --json` at each].

**The one `frame:` commit.** `63cc12a8 frame: node_window_values.run's docstring says its sample's readers, the term GLOSSARY.md keeps`.
- It changes one line of a docstring and no code, so it has no test.
- Why it was needed: DENSE's head `30dac672` already fails the ratchet gate. The repository's retired-terms check finds "sampled units" at `node_window_values.py:634` [read: the gate's JSON at `30dac672`]. Fixing it is the only way my heads could read `failures: []`.

**The form, and why.** The node window lowers `AllReduceMultimem_v1{WORLD=8,SIZE}` by calling vllm's `grain.lower` itself, not a copy. It passes rank 0's partial as producer runs and the site's reads as the leaf ranges its readers take, the same arguments `wires.py` passes it.

- **Coordinate grain** (82 of the 157 sites at 78 layers: the DeepGEMM linears' coordinates and the embedding shard's columns). The units are exactly `grain.lower`'s:
  - one `AllGather_v1{WORLD=96,N=1}` per 96-word chunk of rank 0's partial;
  - one `AllReduceMultimem_v1{SIZE=96}` per chunk, with `parts` cut 8 ways: row 0 is the gather's row (wired), rows 1–7 are the other ranks' `peers_*` partials, left free;
  - one `AllGather_v1{WORLD=64,N=96}` per token row, which whole reads of the row reuse.
- **One addition at coordinate grain:** one `AllGather_v1{WORLD=2,N=96}` per 192-column block. The norms' apply templates (`RMSNormApplyCudaVector8_v1{G=192}` and the fused one) read 192-column blocks. They read the decode's COLS all-reduce as one 192-word row. The frame refuses a port read at two row counts, so the prefill must also hand them one 192-word row.
- **Row grain** (75 sites: the MoE layer's final all-reduce, whose own partial is DENSE's `ResidualAdd_v1{N=192}` blocks). Here `grain.lower` would give one `SIZE=6144` combine per token row. That row would have to gather 32 residual blocks on the way in, and the next norm's 192-block reads would cut it on the way out, which `Piece.rows` refuses. So I lower these as `SIZE/192` column blocks: `tp_combine.row_binding` at `rows=SIZE/192`, which is `AllReduceMultimem_v1{SIZE=192}` because every rank uses the same tree. Row 0 of each block is the residual add's block (wired); rows 1–7 are the peers (free).
- **Embedding site:** the embedding shard has no lowering. That site gets no input gathers, and row 0 of its combines is free under `EmbeddingShard`.

**Does the node window emit the same units as the TP window for rank 0's sites?**
- **`SIZE=96` (coordinate): yes.** 225,664 units at 78 layers, the TP record's figure [read r20261113-055026-4829], in the same position order (chunk c of site s), with the same input and output gather templates.
- **Two differences at coordinate sites:**
  - the block gathers, which are node-window-only (112,832 units at 78 layers);
  - the embedding site's input gathers, which are absent here: 222,912 = 81 × 2,752 instead of TP's 82 × 2,752 [read 4829; the 82 × 2,752 is inferred from TP's form].
- **`SIZE=6144` (row grain): no.** TP has 3,225 units of `SIZE=6144`; the node window has 103,200 of `SIZE=192` (75 sites × 1,376) [read 4829]. Each computes the same words, since the reduction is per element with one tree [inferred; the value check checks mine against the Program, not against TP's units].
- **What a reader of both must reconcile on Thu 19 Nov:**
  - coordinate sites: row r of combine unit c is wired from rank r's chunk-gather unit c, the same in both windows;
  - row-grain sites: either TP's row grain learns 192-column blocks reading each rank's residual-add block, or the node window lowers `ResidualAdd` whole. A reader must map TP's token unit t to the node window's units t·32 + b;
  - the embedding site's row 0 stays free until `EmbeddingShard` is lowered;
  - the block gathers have no TP counterpart.

**Figures: 43 ids, 2 outputs, rank 0, families frame + dense + sparse + multimem.** "Before" figures are as you gave them.

| | 4 layers, before | 4 layers, now [read r20261113-055056-24b4] | 78 layers, before | 78 layers, now [read r20261113-055026-4829] |
|---|---|---|---|---|
| window digest | `61b9f8d8516fe67f…` | `3ee9420daaf7545f…` | `f4c1c567288f…` | `08a3b9bbb6575290…` |
| units | 5,137,384 | 5,191,393 | 123,537,384 | 124,208,701 |
| templates | 105 | 108 | 105 | 108 |
| wire patterns | 13,238 | 13,266 | 945,322 | 946,126 |
| peak GiB | | 1.012 | 10.9 | 13.916 (watch 14,214,264 kB) |
| window build (s) | | 28.2 | 594 | 681.6 |
| still unlowered | | EmbeddingShard 2, StopCheck 1, TokenSelect 2 | the same + AllReduceMultimem 157 | EmbeddingShard 2, StopCheck 1, TokenSelect 2 |
| free | | EmbeddingShard 270,336 words, peers 17,302,208, weights | AllReduceMultimem 82,962,432 words, … | EmbeddingShard 270,336 words, peers 297,370,304, weights; **nothing under AllReduceMultimem** |

New units at 78 layers [read 4829]:
- `SIZE=96`: 225,664;
- `SIZE=192`: 103,200;
- `AllGather{96,1}`: 222,912;
- `AllGather{64,96}`: 3,526;
- `AllGather{2,96}`: 112,832.

At 4 layers: 22,016 / 1,376 / 19,264 / 344 / 11,008 [read 24b4].

The other ranks' partials are free inputs listed by name: 1,099 distinct `peers_*` names at 78 layers (= 157 sites × 7), each at its row (r, 1, 8) [read 4829].

The checks all passed at both sizes: `settle` with `check_proof_block` (provable), the `Circuit`'s `check_wires` and `check_inputs`, and `check_proof_block` again per template with its cuts [read 24b4, 4829].

**Value check:** DENSE's driver (r20261113-035817-256b's `script.py`) with the multimem family added, at 4 layers, 43 ids, seed 274.
- **Run:** r20261113-073556-0a01 at `5005ee9e`; peak 7.36 GiB (watch 7,611,312 kB); 1,971 s [read 0a01].
- **Window:** the plain and stand-in windows are both `3ee9420daaf7545f…`, so the stand-ins don't change the window [read 0a01].
- **Result:** 108 templates, 41,695 units checked, 0 mismatches, 0 units with undeclared inputs [read 0a01].
- **Every template was checked or evaluated:** the list of unchecked and unevaluated templates is empty [read 0a01].
- **Each new template was checked against the Program with 0 mismatches** [read 0a01]:

  | template | checked | of units |
  |---|---|---|
  | `SIZE=192` | 1,024 | 1,376 |
  | `SIZE=96` | 1,024 | 22,016 |
  | `AllGather{2,96}` | 1,024 | 14,048 |
  | `AllGather{64,96}` | 344 | 344 |
  | `AllGather{96,1}` | 1,024 | 30,624 |

  Each gather template's unit count includes other families' units of the same template.
- **Planted error** (`plant_wire_row(c, "AllReduceMultimem", "parts", 3, keep=True)`): the first row-0 wire into the all-reduce, copied to row 3. It landed on `AllReduceMultimem_v1{SIZE=192}`, wire 150, units [0, 1376).
  - It covers 1,024 of the reader's 1,024 sampled units.
  - The check caught it: 1,024 mismatches of 1,024 checked, 0 with undeclared inputs; the first is unit 0, o0, 190 of 192 words differing [read 0a01].
  - No plant was aimed at the `SIZE=96` template.

**Tests.** My tests plus the frame's, dense and sparse files in one run (`test_node_window_multimem.py`, `test_node_window.py`, `_dense.py`, `_sparse.py`, `-n 4`): r20261113-055126-6a3b at `5005ee9e`, rc 0. The line reads 38 dots and [100%], with no count line printed; mypy on my two files: "Success: no issues found in 2 source files" [read 6a3b].

**Limits:** every run had `--mem-gb 32` and a watch that kills the process past 48 GB. None came near: the largest peak was 14.2 GB, the 78-layer census [read the watch files].

**Run ids, each with its source commit:**
- probe: r20261113-051030-9b44 @30dac672;
- 4-layer census: r20261113-052539-a8df @4288774e (same digest), then r20261113-055056-24b4 @5005ee9e;
- 78-layer census: r20261113-055026-4829 @5005ee9e;
- tests: r20261113-052512-d309 @4288774e (2 failures from my peer-name assumption, fixed in 26d5e1ae), r20261113-053325-21bc @26d5e1ae (passed), r20261113-055126-6a3b @5005ee9e (passed);
- value check: r20261113-073556-0a01 @5005ee9e;
- stopped or timed out:
  - r20261113-053230-a9a9 (78 layers at 26d5e1ae): I cancelled it at 17 min. `reads()` scanned an unbatched argument once per member; 5005ee9e scans each argument once, and the 4-layer digest is unchanged;
  - r20261113-053257-b863 (my first value driver): hit its 120 min cap with no output;
  - r20261113-063504-7436: I cancelled it at 60 min.

  Both slow value drivers lacked `import verity.definitions.numerics.rows`, so they evaluated with the reference kernels.

**What was not run:**
- the value check at 78 layers;
- a plant on the `SIZE=96` template;
- any comparison of my `SIZE=192` units with TP's `SIZE=6144` units;
- wiring from `sites_circuit`;
- the full verity-vllm suite at the branch head (only the four named files);
- `circuit-check`, because no Definition, template or Boolean lowering changed: only existing templates are used (`AllReduceMultimem_v1`, `AllGather_v1`).

**The statement I trust least:** that the row-grain `SIZE=192` combines and TP's `SIZE=6144` combines compute the same words, so that wiring them together on 19 Nov is only a re-indexing. That rests on the reduction being per element with one tree, which I inferred and did not run. The value check shows my units equal the Program's node, not TP's units.
