[2026-11-16T22:00Z] 2d61ea7f2d3b
From PAGE (training), 2:00 PM PST [read the clock]. PR 6 is open as draft #10607 [ran]: https://github.com/compute-verification/verity/pull/10607
- head 1ed9b643a5a10d6e854288310121ffee1fdf4d9a, branch cursor/training-delete-covering-rest-4207;
- 13 files, +5/-932;
- not linked in Linear: `research queue link 10607 --issue TRN-177`.
It differs from the brief in one way [read]. `Definitions/Slowdown/Covering.lean` also keeps the `MulBlocks` structure, because Capability (NAMES2's) reads it: `Capability/Consumers.lean` `ofUses` and `Capability/Toy.lean` `B`. Cutting it would break NAMES2's files; it goes with Capability. Only `MulBlocks.Completes`, `weightsCompleted` and `Cuts` go from that file.
`PartitionCovers` also reads `Covers`, not just `InferenceBlock`, so `Covers`, `Subcircuit`/`inner` and `sources` stay too. `Proofs/Slowdown/Covering.lean` keeps only `FeedsWithin.mem_right`.
Import graph and token match [ran]: I ran `git grep -w` on all 47 removed declaration names. Outside the deleted code, the only hits are:
- the Cover docstrings, which #10564 deletes;
- the calculator's classes.py, profile.py, the premises.toml rows and test_classes, which #10597 deletes;
- unrelated declarations that share the short name `Cuts`.
So it lands after #10597 and #10564.
Lock records to remove are in the body:
- the two guarantees (guarantees, unwitnessed, shape.frozen/inline);
- open ParameterHard and ParameterHardAt;
- the four modules' reads;
- three definitions of `Definitions.Slowdown.Covering`'s reads.
PartitionCovers' record should not change [inferred: its declarations are byte-identical; no Lean ran].
Not run: Lean. There is no built security/.lake here, lake build is refused, and research run is down. `verity lean try` on `Proofs/Slowdown/Cut.lean` and `Proofs/InferenceOnly/PartitionCovers.lean` is owed on a node.
Ran: test_doc_paths and the retired-term lint, 43 passed; test_metrics_registry (handle registry), 11 passed.
The READING (grade.py) follows by inbox.

[2026-11-16T22:06Z] 097bb00261e2
PAGE, 2:05 PM PST. READING: grade.py on the page's count (main e80c1a37).

**The count is already there.** `formalism.bound`'s products term is `calculator.slowdown`, which calls `min_cut.slowdown_of` and then `min_cut.bound` [read]. `bound().slowdown` is max(1, cν/μ), and `macs_a_block` is μ [ran: equal at two policies, F = 1 and 2/√3]. So `measure`, `published` and `bound` become one call:

`calculator.slowdown(pol, products, {gates: |C|, tau_bits: τ, code_bits: 8|E|})`

Dropped: Fit, `partial`, the four sensitivities [read].

**A traced step.**
- ρ = nc/|C|, ν and o are the whole audited circuit's [read].
- The grade scales the one traced step's N and O by tokens to the prefix's changed steps (the stand-in), over the prefix's |C| counted from the op log. So N is scaled, |C| counted [read].
- A one-step bound doesn't follow from the page: a block cut down to one step can read over k bits from other steps [inferred].
- No trace gives n or T1/T2. β_w, β_x and b_y are premise rows plus the code cap [read].
- So the honest output is the count's inputs (c, k, N, O, |C|, ν, o, β's, b_y, F, μ) and cν/μ on the search prefix: a lower bound on ρ under T1, T2 and the stand-in, not a per-config slowdown [inferred].
- S = p·bound and per_period's C·bound are products the page doesn't state [read].

**Library and job.**
- `library.py` imports no formalism and already reads `calculator.slowdown` [ran rg; read]. Only its `per_period(…, {"binding_kind": …})` key changes, if renamed.
- `run.py` imports neither [ran rg]. Its trace's `circuit` block loses σ, τ, a, u, u_in, the tail, Δ_grad and the cuts, keeping `gates`, which `library` and `calculator` read [read]. Most of `capture/circuit.py` (462 lines) goes [inferred].

**Lost silently unless moved** [read]: `other_program` (is the trace the run's program; only `formalism.complete` runs it) and `refusal`'s read-back checks.

**Questions for the lead or Daniel** (I decide none):
- (a) Print S and per_period, or only the bound and its inputs?
- (b) Fit's c/(c−d): drop it, or print it beside as the attack side?
- (c) The calculator drops the first ⌊k/(τT)⌋ steps, which needs τ: keep τ alone, or drop the rule?
- (d) Keep β_w ≤ 8|E| / trained weights?
- (e) Keep the other_program and read-back refusals?
- (f) One traced step standing in for N, or a trace for each structure?
- (g) Does `capture.circuit` keep only `gates` and `affected_share` (the update-free check)?

| name / test file | goes, or what replaces it | tag |
|---|---|---|
| `published` | `calculator.slowdown`'s `slowdown`, `binds` and `term`; no Fit, no `partial` | read |
| `bound` | the same call's `bound`, already the bound of record | read |
| `measure` | goes; its refusals move to `traced` per (e) | read |
| `of_record` | \|C\| as a sum of `gates.of_log` over the prefix's steps and setup, with no min_cut call; products from `work.products_of_trace`, which `traced` already calls | read |
| `of_trace` | goes; τ only if (c) keeps it | read |
| `TRACED_STRUCTURE` | goes, with `with_circuit` | read |
| `CELL_SENSITIVITIES` | goes, with `blank`'s sensitivities and their printed line | read |
| `READ_BACKS` | its two keys move into `TRACE`, or go per (e) | read |
| `Refused` | moves into grade | inferred |
| `STAND_IN`, `traced_label` | move into `work.py`, beside `products_of_trace` | inferred |
| `products_of_log` | moves into `work.py` with `log_products`, `other_program`, `PLAIN_PRODUCTS` and `SDPA`: it reads the log, not v2 | inferred |
| test_formalism (41 tests) | ~24 v2 tests go; ~13 products-term tests are calculator's and are re-pointed or dropped; four move with their code (lines 329, 342, 400, 918); `he_row` moves | read names, inferred split |
| test_grade (37 tests) | 23 touch F, most only through `PLAIN_PRODUCTS` in the helper `traced`; the tests at lines 261, 317, 458, 658, 829, 916 and 949 are rewritten or re-imported; the line-346 test and the `blank` test lose their sensitivity parts | read |
| test_policy | one line, Fit at the reference policy, goes | read |
| test_one_calculator | one test loses its `Measured`/`published` half; the `formalism.py` allowance entries go | read |
| test_circuit | goes with `capture.circuit` per (g) | read |
| test_library | one `he_rests_on` assert moves with that printer | read |
| test_repeats | `he_row` and `COUNTED_PRODUCTS` are re-imported | read |
| test_run | line 491's key check shrinks to `gates`; lines 727 and 845 re-import `other_program` | read |

