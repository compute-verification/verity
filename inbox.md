[2026-11-12T05:47Z] f32d05d5edf7
PRICER (TRN-135), job 1: what a Program states of min_cut.bound's inputs. Branch cursor/training-slowdown-off-program-914d, head f7e4dbd878; the reading is benchmarks/circuit_properties/circuit_properties/work.py. Figures are RAN on the VM unless marked.

1. Each input of min_cut.bound, and whether a Program states it
- |C| (size): stated. root.gates, the Program's own gate count, already O(description) (batches and scans multiply). Unit caveat: on a word Program a gate is a word Definition's gate, not a Boolean gate, so c and k (Boolean units) are compared with a smaller |C| than the trace's step_gates. On the tiny lf step: 13,326,455 Program gates per iteration vs the trace's ~1.5e11 Boolean gates (RAN). A Boolean lowering of the same Program would close this; the reader doesn't lower.
- Which activations are matrix products, and their MACs: stated, read through a reader-side table of registered Definition ids (LinearRef_v*, DotRef_v*, each with which operand is x and which w). MACs = words of the larger operand; k_dim = words of the smaller. Nothing beyond the Program, but the table is the reader's, keyed on registry ids.
- Which products are charged: stated, by data flow. A product is charged when both operands are fresh (not θ, not reachable only from θ), not a lookup's result, and not the same array (self). In a scan, a weight carried in the state is charged from the iteration at which the carry stops being θ (iteration 2 on), which the loop test checks.
- O: stated. Leaves of what each charged product returns × the leaf's width (32 on f32).
- The sharing F needs: stated. Union-find over array intervals (overlapping slices merge; a callee's parameter binds to its caller's argument); the share graph's largest clique ω, F = share(max(2, ω)). Tiny lf: ω = 3 (q, k, v reading one activation), F = share(3) = 2/√3, equal to the trace's (RAN).
- The lookups left out of N: stated. A Gather whose table is θ-side and whose index is fresh is a lookup; "looked" propagates through reads unless a read is fresh and not looked. Tiny lf per iteration: lookup exclusions = 851,968 MACs, exactly the trace's "gathered" (RAN).
- β_w, β_x, b: β_x from min_cut.beta(k); β_w and b_y = the operand's word width. The docstring's "a weight with a shorter code costs its code bits" is not stated in a Program: the width is used (32 on f32 vs the trace's bf16 16, and its rates-based β of 0.9–7.7 bits).

Per-iteration agreement on tiny lf with AdamW (step_program inside a scan, iteration 3 minus iteration 2, vs capture_step's trace, RAN): N = 11,012,096 vs the trace's affected MACs 11,010,048 (difference 2,048 = DotRef dots of softmax backward, which the trace has no product row for); O = 6,950,912 vs 6,946,816 (difference 4,096 = those dots' outputs). Slowdowns at (c=1e12, k=65536): trace 1.287 (its he keeps 368,640 MACs of a subset), mine ≈ 4.98e5 (7.08e6 weight MACs at 32-bit words over 13.3e6 word gates). The ratio is the blocks ratio × the |C| ratio, which the test holds; the gap is the Boolean-vs-word |C| and widths, not N.

2. What the trace's classification uses that a Program doesn't state
- Token dims: not readable from one Program (shapes are fixed; nothing says which axis is tokens). Needed by the trace only to pick self/gathered rows and he subsets; the reading doesn't need them.
- Gathers: readable, as Gather prims with a θ-side table. Not caught: an embedding written as a one-hot matmul (then it is a charged product, which is conservative for the bound, not unsound).
- Weights vs activations: readable in a loop: carried state and shared args at depth with no fresh level are weights; xs are activations. Risk: a submitter can carry activations through the state to make them read as weights. That changes which kind a sub-work prices, not N.
- Adapters (rates, β_eff): not readable. Rates are a benchmark heuristic about the data, not a sound ceiling a Program states; the reading uses the width.
- The three kinds of product (weight×act, act×act, self): readable from data flow (weight = no fresh level; self = one array class); they enter the bound only through ω and the kind sub-works, both read.

3. What a submission carries beside the Program
Nothing, given the reader's product and lookup tables (keyed on registered Definition ids). Optional: each weight's code bits, which only lowers β_w below the width; absent, the width is charged, which is conservative.

4. Placement and imports
work.py sits in benchmarks/circuit_properties/circuit_properties/, imports verity.circuits (Program, refs, defs, program) and the calculator's min_cut by path (the rho dir on sys.path, as references.py does). No edge into verity/ from benchmarks; benchmarks loading benchmarks is allowed by tests/test_boundaries.py (BENCHMARKS_IMPORTERS lists outside importers only). The comparison test is in benchmarks/training_slowdown/tests/ (rho's periods imports `run`, which collides with circuit_properties' run.py inside its own suite); it imports neither rho nor periods, so #9003's test_one_calculator holds. training_slowdown's [tool.verity.tests] inputs gains benchmarks/circuit_properties/circuit_properties.

Not checked: the suites on a node (run r20261112-054616-5d26 queued, ~11 min wait); ruff in the suites' own lint; any Program but the tiny lf step and hand-built ones; a Boolean lowering's |C|.

[2026-11-12T05:48Z] c22cf1bf08c7
QAT (TRN-107), following my stop: two facts, nothing pushed since.

- r20261112-053026-0004, the harness's suite --fresh at b4d02a5a6f (#8910's head, without (b)): "549 passed in 344 s", "suites benchmarks/training_slowdown --fresh exit 0" [RAN, node computeinstance-e05m8fsfqd6p24xpft]. Record art:30e93b9245d770b7a8a5466c648b6b2c268eef0ebc27f11727b41f693d53843e, preserved=yes.
- Why cde7 said preserved=NO: my fetch ran at 05:19:00.593Z, while custody was still writing (it verified at 05:19:00Z). The job had kept its outputs correctly, and the store holds the attempt as run_record art:8ce14c21d199ff98836d493bff1bace4f94104f79cfed06c188a7d8147777cbe [RAN, research data show]. For the rerun, I fetch only after `research data show RUN` names a run_record.

Still waiting on your call: (i) `_fake_copy` or (ii) `_optimizers` taking `differentiable`, and whether that goes in as a fourth commit. 3f217ad834 stays on cursor/training-qat-resume-fb49 until then.

