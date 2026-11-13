# CIR-274 piece 5: the served token select in rank 0's node window

Branch `cursor/circuits-cir274-select-6f04`, head `c3f66d06e5cbc889487966c532674e2ce900151a` (one parent, `bc1841a446`). Its
first-parent line from the joint head `30dac672d2`: the merge of `main` `9182900fb94e` (`61b37295e9`), the lowering
`node_window_select.py` (`fe983814d2`), its test (`d705c29a31`), REQBLOCK's `63cc12a88e` cherry-picked as it is (`7a8901d06d`),
the test's retired term reworded (`bc1841a446`), the test's evaluations by the reference evaluator (`c3f66d06e5`). The piece's
diff past the merge: `node_window_select.py` (+141), `test_node_window_select.py` (+126), `node_window_values.py` (1 line, the
cherry-pick). No hunk in `node_window.py`, `node_window_dense.py`, `node_window_sparse.py` or `serving_templates.py`; no
`select-frame.patch` and no `needs-vllm.md` section were needed.

Every figure below is from the named run's own output (`~/.research/runs/RUN/out/*.json`, `stdout.log`). The window and value
runs are at `bc1841a446`; `c3f66d06e5` changes only the test file.

## a. The window at 43 prompt ids, 2 outputs, 4 layers, rank 0

Run `r20261113-083913-d5b0` (source `bc1841a446`), `select_run.py --ids 43 --layers 4`.

- `settle` (`NodeWindow.circuit`), `check_wires`: passed. `check_proof_block` on all 125 templates under their cuts: none refused.
- Program `7f36f17ae005fc05…` (319 nodes). Window digest
  `366fba94c7fcb59030cdc9bc018087de1ce9e6096dc0873f09fc2214b2bd904351b0c7a9c371811b212224dd276b8ed050418b8383fdfce0f4c64f5484b8ba57`.
- 125 templates, 5,137,424 units, 13,417 wire patterns.
- Both selects lowered: nodes 246 and 317, each `TokenSelect_v4{V=154880}` (a call, one member), as 20 templates of 2 units
  each (40 units): `GreedyChunk_v2{N=7744,OFF=7744c}` for c = 0..18 and `GreedyCombine_v2{V=154880,N=7744}`; every `logits` port
  cut 64 ways (rows of 121 lanes); the combine's `r0`..`r18` each wired to its chunk's `o0`.
- Unlowered: `AllReduceMultimem` 9, `EmbeddingShard` 2, `StopCheck` 1 (`TokenSelect` 2 before the select: run
  `r20261113-084004-3223`).
- Free, the select's: 2,240 `logits` rows of 121 lanes, all `peer`: 160 under each of `peers_63`..`peers_69` (node 246) and
  `peers_133`..`peers_139` (node 317). By template: `GreedyChunk_v2/logits` 2,112, `GreedyCombine_v2/logits` 128.
- Free, the whole window (census `free`): peers 4,270 unit ports, 929,152 words, 2,317 ports; `AllReduceMultimem` 12,772 /
  4,761,600 / 19; `EmbeddingShard` 32 / 6,144 / 1; weights 1,081,344 / 322,961,408 / 16.

Without the select's family (run `r20261113-084004-3223`, same source): digest `61b9f8d8516fe67f…`, 105 templates, 5,137,384
units, 13,238 wire patterns. With it: the 20 templates and 40 units above and nothing else: no new template outside them and no
template's units changed.

## b. The value check on the 20 templates

`select_run.py`: the 20 templates' units recomputed from their declared inputs (rank 0's rows from the wires, the peers' from
their stand-ins), the combine against the Program's `TokenSelect_v4` output, each chunk against `greedy_stages.twin` on the
Program's logits row; then a plain `Check` (no twin anchors) recomputes the combine from the chunks recomputed from their logits.

| run (source `bc1841a446`) | layers | tokens (node 246, node 317) | units checked, mismatches | plant 1 | plant 2 |
|---|---|---|---|---|---|
| `r20261113-083913-d5b0` | 4 | 2639, 569 | 40, 0 | 1 of 2 | 2 of 2 |
| `r20261113-083913-0328` (tie) | 4 | 38820, 38820 | 40, 0 | 1 of 2 | 2 of 2 |
| `r20261113-090649-32bf` | 1 | 447, 15224 | 40, 0 | 1 of 2 | 1 of 2 |
| `r20261113-090649-61b1` (tie) | 1 | 38820, 38820 | 40, 0 | 1 of 2 | 2 of 2 |

- In every run the twin, the Program's v4 value, the units from scratch and `TokenSelect_v1{V}` give one token per select.
- Plant 1: chunk 0's `logits` wire on the row holding its best lane (row 21, lane 2,639, at 4 layers) moved one writer row on
  (`base` + 1); chunk 0 then fails: `[16512, 2639, 0]` wanted, `[16479, 7438, 0]` got. Plant 2: the combine's `r<w>` (w the
  first select's token's chunk) wired to chunk w + 1's output; the combine fails on every select whose token is in chunk w
  (4 layers: w = 0, `[2639, 0]` wanted, `[17893, 0]` got; tie: w = 5, both selects, got 92,933, the chunk-12 tie lane; 1 layer:
  only node 63, as node 83's token 15224 is in chunk 1).
- Tie: the peers' logits set to `0x7F7F` (the largest finite bf16) at lanes 38,820 and 45,720 (chunk 5) and 92,933 (chunk 12),
  rank 0's own maximum at most 4.0 in both tie runs; the units' answer is 38,820, the first, and equals `TokenSelect_v1`'s.
- The Program's v4 value and the driver's v1 come from `numerics.rows`' argmax kernel (`evaluate_batch` "auto"); the stage
  units have no kernel and are their own circuits. Run `r20261113-091549-4a01` (source `c3f66d06e5`, `select_fullv.py`)
  evaluates v4, v1 and the 20 units by the reference evaluator at V = 154,880 on four rows (two N(0, 1/4), the tie row, a NaN
  first logit): all `[12070, 74587, 38820, 0]`, the twin's.

## c, d

Run `r20261113-091549-c3a3` (source `c3f66d06e5`): ruff clean on the two new files; `test_node_window_select.py` (5),
`test_node_window.py` (7), `test_node_window_dense.py` (15), `test_node_window_sparse.py` (12): 39 passed, 0.70 GB peak.
Run `r20261113-091550-f5bf` (source `c3f66d06e5`): `ratchet_gate.py 9182900fb94e… HEAD --json`, `failures: []`. At
`d705c29a31` the gate (`r20261113-075512-9584`) failed on three retired-term sites: `node_window_values.py:634` (fixed by
cherry-picking `63cc12a88e`) and two in the new test (reworded).

## Item 3: `TokenSelect_v4` against `TokenSelect_v1`

v1 is one `ArgmaxStep_v1` scan over logits[1:V] from (logits[0], 0); v4 cuts the row into chunks of 64 (`TokenSelectChunk_v2`)
and reduces their partials from logits[0] (`TokenSelectCombine_v2`), linear in V where v1 lowers quadratic. Both take lane i
iff it is strictly above the best, so the lowest index of the largest logit wins, a NaN is never taken and a NaN logits[0]
gives 0; the stages are that function over 20 chunks of 7,744 lanes. Equality is shown by evaluation, not proved: by the
reference evaluator at V = 200 and 130 on tie-heavy rows (signed zeros, infinities, NaNs; the test), at V = 154,880 on four rows
(`r20261113-091549-4a01`), and on the Program's own logits at both selects (the table above). The v4 node maps to the stages in
`node_window_select.token_select`; its output is the combine's `o0` (the I32's two words).

## Item 4: the logits read

The node's logits are `AllGather_v1{WORLD=8,N=19360}`'s output, which the lowering reads through (`through`) to its eight parts:
rank 0's are its LM head's words, the other seven a peer's input. Each chunk's and the combine's `logits` port is cut along its
outer elements into 64 rows of 121 lanes: 121 is the gcd of the chunk width 7,744, the part boundaries (multiples of 19,360;
alone they give 3,872) and the LM head's first-level gather chunk (`AllGather_v1{WORLD=121,N=1}`), so every row is one rank's and
each of rank 0's is one existing first-level chunk row, wired whole. Rank 0's 320 rows (chunks 0, 1 and half of 2, both selects)
come from that gather; the peers' 2,240 are free under their names. The LM head's side gains no unit and no template (run
`r20261113-084004-3223`). A cut at 3,872 lanes would need a gather of 32 first-level rows for each of rank 0's rows (inferred
from the frame's `_whole`, not run).

## Item 5: `StopCheck_v1`

`StopCheck_v1{T=2,STOP=[],MAX_TOKENS=2,PROMPT_LEN=43,MAX_MODEL_LEN=4096,FINISH=2}` (node 318) reads `ids` = two leaves, nodes
246 and 317's outputs, which now resolve to the two combine units' `o0` rows (2 words each): a wire takes that read. What still
refuses it: it has no lowering family, and `check_proof_block` refuses the block itself, `ids 8 B, o0 2 B: no row past 111 B, so
none has a compression before its last SHA-512 block and C-Flock's sha512x3 range, which holds the pin, has no slot` (run
`r20261113-083913-d5b0`). It needs a binding with a row past 111 B, or merging into a larger block, before a lowering can
place it.

## What I would change in the frame

- `node_window_select` reaches four frame privates (`_Free`, `_family`, `NodeWindow._chunked`, `NodeWindow._whole`); a public
  "the one writer row holding these lanes, or a `_Free`" (my `_row`) and the all-gather read-through (`through`) belong in the
  frame, where another reader of a gathered output would use them.
- The census lumps every peer input under `peers`; a count by peer would show a select's seven without a probe.

## Not shown

The equality of v4 and the stages is evaluated, not proved (no Lean lemma). The value check uses made-up inputs (the peers'
logits N(0, 1/4), made-up weights), not served captures, and the Program's own v4 value comes from the row kernel. The peers'
2,240 rows stay free inputs in rank 0's window; nothing here ties them to the peers' windows. Only rank 0, at 43 ids, 2 outputs,
1 and 4 layers. `StopCheck`, `AllReduceMultimem` and `EmbeddingShard` stay unlowered.
