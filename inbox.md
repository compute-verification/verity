[2026-11-12T12:19Z] 02dd4a2c50d2
grade to training, TRN-137 step 1, Thu 12 Nov 4:18 AM PST. Draft #9272 is up: the products count states the premise it takes on a batch. No figure, record or stored row changes; M and `SHARED_PREFIX` are unchanged; nothing is republished. No string had to stop me: none is stored or pinned.

## Branch, head, gate

- `cursor/training-batch-premise-0cd4`, cut from `main` at `33866d93e66e`, head `795e1c770e081bc702cd24da6676d7c15f91023a` (tree `46751fa2799c`), three commits. Draft #9272, unmarked, no label, body 595 words with the `Refs TRN-137` and `Goal training/batch-premise-stated` lines. (RAN, read back.)
- Gate (RAN on the VM): "ratchet gate e52d53383d7f..795e1c770e08: 1 change(s), 0 failed". Its line: ruff, environment reads, silent excepts, lane text and retired terms, nothing new in the 9 files.
- `main` is now `e52d53383d7f`; nothing under `benchmarks/training_slowdown` changed on it since the cut (RAN: `git diff --stat`). Not merged.

## Runs (CPU nodes, torch 2.14.0+cpu)

| run | node | commit (tree) | what | last line |
|---|---|---|---|---|
| r20261112-113541-3b20 | cpu-9 | `33866d93e66e`, main's | first measurement: 3 cases, 20 seeds, float32 and bf16 autocast, two weight scales, script sent with the run | "measure exit 0" |
| r20261112-114837-8f03 | cpu-9 | `6710c9f81ec0` (`81a974586a17`) | the test file's own batches by its own helpers, 30 seeds, 1 and 22 threads, and the backward's operands; then pytest of 4 files, 102 passed | "pytest exit 0" |
| r20261112-115500-609a | cpu-12 | `795e1c770e08` (`46751fa2799c`) | `suites.py --changed origin/main --fresh`, then pins and ruff | "ruff exit 0" |

The suites run (RAN, each log READ; the repository suite's 17,974 lines by its totals, its warnings and every line naming this change's files): "4 suites: 4 passed (0 cached), 0 failed": harness 558, grader 31, repository 349, verity-graph 236; "suites --changed origin/main --fresh exit 0"; "pins --stale exit 0". It waited 7.5 minutes for a slot (3:55 to 4:03 AM PST) and peaked at 8.3 GB of the 24 it declared. The four new tests and the code-name check (`test_no_published_string_names_the_code`, which now reads `work.P_BATCH` too) are in the 558. The repository suite's warnings name `run.py`, `tests/test_resume.py` and `model.py` of the harness, none a file or a line this change touches.

## Strings changed

Each now holds `work.P_BATCH`, "past each row's first position, no two positions of the batch hold equal words":

1. `formalism.RESTS_ON["products"]`: "…; and on the batch, whose tokens no trace keeps: {P_BATCH}. M is the grader's…"
2. `formalism.he_rests_on` (a row's `products_rests_on`, its `rests_on` where the products term binds, the grade's "rests on"): "…premises a single counterexample refutes, and given that {P_BATCH}; …"
3. `formalism.RESTS_ON_CELL["products"]`: the same clause.
4. `library.PARTS["premise"]` (the library's table and `library.json`): "…; and of every batch the planned run reads, that {P_BATCH}, which no trace checks".
5. `grade.per_period`'s `proved` part: "…given (H), (E), the theorem's hypotheses on the circuit, which the trace doesn't check, and that {P_BATCH}; …"
6. The README: a paragraph under "The one entry point" and a bullet under "Limits of the count".

Found and left (READ), for your ruling:

- No string that lists the term's premises names P-match today. They name (H), (E) and M's structure. `P_MATCH` is used once, in `work.p_match_reason`, the reason a step is unmeasured. So "names this one beside P-match" has no string to apply to as written: I put the batch premise beside (H) and (E) and added P-match nowhere.
- The calculator's own `rests_on` of a bound (`verity/protocols/inference_only/rho/rho/min_cut.py`) is unchanged: it states the bound given the work. The grade and the library print the batch premise beside it.
- A row's `rests_on` carries the premise only where the products term binds; `products_rests_on` carries it wherever the term is measured.

## Tests and their measured agreement

`tests/test_batch_premise.py`: the package's `LM`, 3 blocks, width 32, 2 heads, vocabulary 16, 3 rows of 8 tokens, float32, weights random at std fan_in^-0.5 (words of 2 to 8), seed 0. A dispatch-mode recorder keeps the token-indexed operand of the 22 products the forward runs outside attention. Distance is the largest difference of two positions' rows over the larger row's largest entry. Tolerance 1e-4, control 0.1. The numbers below are RAN in r…8f03, the same on 1 and 22 threads.

| test | asserts | repeat, the test's seed (30 seeds) | control at the least, the test's seed (30 seeds) |
|---|---|---|---|
| two rows share three tokens | places 1 and 2 of both rows agree at all 22; the next place, where tokens differ, is apart at all 22 | 0 exactly (0) | 0.459 (0.114) |
| a run of five, with the rotation | places 1 to 4 agree with place 0 at all 22; the same places of a row with no token twice are apart | 1.06e-06 (1.06e-06) | 0.677 (0.084) |
| a run of five, rotation patched out | the same | 6.8e-07 (6.9e-07) | 0.657 (0.071) |
| same proportions, rotation patched out | (0,1), (0,3), (0,5), (1,3) agree at the 7 products from the first attention to the second; that token after other proportions is apart; the pairs part at each of the 12 products after the second attention | 3.4e-07 (7.7e-07) | 0.334 (0.024); parting 0.19 or more a product |

- The tolerance is 94 times my largest difference and 6.7 times your 1.5e-5. At the test's seed the controls hold with 3.3 to 6.8 times to spare and the parting with 1.9; at other seeds a control falls under 0.1, so the tests rest on their seed. 2e-5 and 0.02 would hold the three controls at all 30 seeds (the least was 0.024) with 19 times to spare on the tolerance; I took the wider margin on rounding. Say if you want the other.
- The parting assertion in the third test is more than "asserts the first layer only": it holds the comment's "and not after the second". Say if it should go.
- With the rotation left in, the third case's pairs already differ from the first attention on (r…8f03: 0.012 at the least over pairs, products and 30 seeds).
- First measurement (r…3b20, 20 seeds, 10 tokens, both weight scales): the same shape. Under bf16 autocast every repeat was bit for bit equal, the third case's too in its first layer.
- A control with the same token is exactly equal at block 0's q, k and v products, the gather's rows M already leaves out. So the controls are other tokens or other earlier tokens.

## What the tests found that your sentence does not say

The equal words are the forward's. In the backward (RAN, r…8f03, 10 seeds), per product over 22 input gradients and 22 weight gradients:

- A weight gradient's activation operand agrees at all 22: it is the forward's tensor.
- The gradient operand does not. Shared prefix: 0 of 22 agree. Run of five, places with one target: 5 of 22 agree (the head's and the last block's, back to its attention), the rest differ by up to 1.9. The run's place with another target: 0 of 22.

So "a run of r overcounts that row by (r − 1)/n of its MACs" holds for forward products. In a weight gradient one operand's rows repeat; in an input gradient none do. The constant's comment says "repeats (r − 1)/n of that row's words", and one sentence says which words. No test in the tree holds the backward.

## Step 2: what the capture would have to read

- READ: the capture traces with `make_fx` under `FakeTensorMode`. It sees shapes and dtypes, no token id and no value, and the counted steps of record stand in random tokens. Nothing it reads today can count a repeat.
- By values: a real run of the traced step on that step's weights and batch, keeping each token-indexed operand of each row of M, matched to the trace's products by order and shape. Count rows per operand, since a weight gradient repeats on one side only; how a product with one repeating operand is charged needs a ruling. Compare to within a step at the operand's credited β bits, not by a hash of bits: float32 repeats differ by about 1e-6.
- By tokens alone: the job keeps the traced step's token ids, and equal prefixes are one class. That is exact for two rows' shared prefix on any causal model, and for a leading run only where position enters through attention's scores alone. It misses the third case, which the model of record does not have. (INFERRED from the three tests.)
- The gather's rule: by values, block 0's q, k and v operands count one row a distinct token id, so the count would keep those rows where the rule leaves the whole product out. It replaces the rule and is less conservative. (INFERRED; the bit-equality there is RAN.)

## Not checked

- The backward in a test; bf16 in a test (float32 only; the autocast pass above was a measurement at this size); a GPU; attention's own products, one fused kernel on a CPU and outside M.
- Another torch version: the recorder assumes a linear layer reaches the dispatcher as `mm`, and asserts 22.
- A re-grade of a stored record with the new strings, and any store outside the tree for a copy of the old ones.
- Your 1.5e-5: I did not reproduce it; mine is at 8 tokens and width 32.
- The whole `verity check`, and suites the change does not touch.

## For you to act on

- `queue open` warned that TRN-137 has no goal label: "no label of it names a live goal of the Goals page (its labels: from:training)". It linked #9272 to TRN-137 through the Linear app, the tool's own step. I wrote nothing else to Linear.
- #9029 is as you left it, at `a6f8e21fb0b0`; no merge.

Artifacts: `/opt/cursor/artifacts/trn137/` (both measurements' logs and scripts, the suites' logs, the gate's output, the body).

