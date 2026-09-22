# A5/6 checkpoint — 2026-09-10 19:05Z (after the integrator's A2/A7 integration note)

Worktree `contract/a56-coverage` off `346aa4ac` (not rebased); A2's files read at mainline `b47e0d85` with `git show`.

## Done since 1845Z

1. **`coverage_v1` rebuilt on A2's `veritor.interface.expected`** (no duplicate derivation of positions). `expected_statements(circuit, partition, Q)` now consumes A2 `ExpectedObligation`s (`ExpectedValue(tree, leaf, offset)` in/out, `subcircuit`, `pins`) returned by `partition.locate(vu_id)`; the digest covers `{family, subcircuit, incoming, outgoing, pins}`. `check_coverage(expected, proofs, *, trees=, rank_of=)` does the per-VU covered / uncovered / duplicate / unexpected accounting over all accepted proofs and, when `trees` + `rank_of` are given, runs A2's `check_expected_positions` per proof on the expectations that proof matched — any problem voids the proof's coverage (fail closed). `obligations_of_typed_object` reads the bound object only (`leaf.root`, `leaf.id`, `reference.offset`, `obligation.subcircuit`, the `Input` gate's `value` per outgoing slot).
2. **Tests** `tests/contract/a56_test_coverage_v1.py`: 10 tests, all pass, including the archived INT-B0 object (30 obligations located from their declared boundary via `gemm_coordinate_expectation` / `input_equation_expectation`, passing A2's position check; a provenance-relabelled duplicate exposed). Kind-table tests unchanged (3 pass).
3. **Conformance rows updated** (`conformance-rows.md`): R9 carries my reading of the legacy V5 (`rec_p1_verify.py:292-325`: provenance labels compared with located units, positions never read — EA, agrees with A2's M1); R10 records the `tc-ampere-bf16@2` non-finite exclusion as the declared family rule (`tc.rs:9-12`, NaN → `0x7FFF` `:257-282`) with an HC *risk* against the serving trace, and a TokenSelect specification paragraph (kernel-derived tie/NaN rule; three unmeasured NaN rules in play — registry `b1.py:461`, dormant `relations.py:277-278/:377`, harness `registry.py:99-100`; EA until measured); TokenSelect's coverage row notes A2's M2 (outgoing value must be the step-1 boundary leaf `output_token/t`, not the interior `sampler.select` copy).
4. `coverage.md` / `coverage.json` regenerated (counts unchanged; TokenSelect note extended).

## Proposed call boundary for A3 (`verifier_v3`) — please confirm before A3 depends on it

```python
from e2e.coverage_v1 import expected_statements, check_coverage
from e2e.coverage_v1.coverage import AcceptedProof, obligations_of_typed_object, CoverageError
from veritor.interface.expected import CommittedTree, rank_oracle          # A2
from verity_vllm.identity import sort_key                                  # tree rank order

# 1. A3 builds the two verifier-side inputs once per record, from J's commitments + description:
trees   = {tree_id: CommittedTree(root_hex, leaf_count) for tree_id, ... in committed_trees_of_J}   # + registered weights
rank_of = rank_oracle({tree_id: committed_id_list, ...}, sort_key)                                   # from ids committed before the challenge

# 2. Partition = A2 locator: locate(vu_id) -> ExpectedObligation via gemm_coordinate_expectation / input_equation_expectation
#    (ProgramLocator + Description; never provenance).  Circuit.kind_program(vu_id, e) -> (family_id, subcircuit_digest).
expected = expected_statements(circuit, partition, Q)          # raises CoverageError -> the record is REFUSED (not "uncovered")

# 3. For every proof that (a) verified under an approved (ELF, vk) and (b) passed check_binding against its typed object:
proofs = [AcceptedProof(proof_id, obligations_of_typed_object(obj), typed_object=obj) for proof_id, obj in accepted]

# 4.
report = check_coverage(expected, proofs, trees=trees, rank_of=rank_of)   # report.ok iff every V in Q covered exactly once,
                                                                          # no unexpected obligation, no position problem
record["coverage"] = report.to_json()                                     # per_vu, uncovered, duplicate, unexpected, position_problems
```

Contract of the boundary: `expected_statements` refuses (exception) — it never returns a partial map; `check_coverage` never raises on proof content (everything is a verdict in the report); an empty `Q` with zero proofs is `ok`. `Circuit`/`Partition` are `Protocol`s (duck-typed) so A3 can wrap `ProgramLocator` without importing anything from this package beyond the two functions. Digest tag `veritor/coverage_v1/statement` is part of the certificate schema (A7/A9 consume `report.to_json()`).

## CLAIMS / INTERFACE QUESTIONS

- **CLAIM (unchanged):** `vllm-poc/e2e/coverage_v1/**`, `tests/contract/a56_*`, `out/contract/a56-coverage/**`.
- **Q-A56-1 (A3):** who calls `veritor.interface.binding.check_binding` and the approved-guest gate — I assume A3 before constructing `AcceptedProof`; `check_coverage` does not re-verify proofs.
- **Q-A56-2 (A2/A3):** `Circuit.kind_program` for GemmCoordinate = `(family.id, programs.gemm_coordinate(K).digest)`; for InputEq the only builder of the body is the *prover-side* `verity_vllm.typed_obligation_export.input_equation_subcircuit(expected, width).digest` (`:423-430`). The verifier should not import the exporter to derive its expectation — I propose the `InputEq<W>[v]` subcircuit constructor move to (or be mirrored in) `veritor.interface.programs` (A2's file), and `input_equation_expectation` fill `subcircuit` from it. Until then my digest pins the word via `pins` and the `Circuit` supplies the digest.
- **Q-A56-4 (A2, interface):** the fallback elementwise builder (`e2e/coverage_v1/elem_builder.py`, SiluMulBf16 / Bf16AddF2fp) emits *raw* format-3 statements because `veritor.interface.typed_obligation.SUPPORTED_FAMILIES` / `family_gate_set` (A2's file) admit only the tc families. Request: add `veritor.elem-bf16@1` (Python mirror `zk/bench/elem_gates.make_elem_bf16_gate_set`, digest `06172374…`, width 32) so these obligations can travel as typed objects through `check_binding` and `check_expected_positions` like GemmCoordinate. Until then A3 would need the raw-statement path (`obligations_of_raw_statement`) beside the typed one.
- **Q-A56-3 (integrator):** TokenSelect's builder (fallback scope) needs A2's Q1 decided — outgoing value = boundary `output_token/t` (my recommendation) vs. agreement check before Q. I will not build it against the interior leaf.

## Also done (19:20Z addendum)

Fallback builder `e2e/coverage_v1/elem_builder.py` for `SiluMulBf16_v1` / `Bf16AddF2fp_v1` (raw format-3 statements over the record's packed rows; transparent-verified on REAL B0 rows; flipped word rejected; coverage accounted; 4 tests pass). Not pipeline-wired, no SP1 proof run; `RopeOut`/`RopeOutAdd` not built (cos/sin come from the registered derived table, opening-only). Effect in `coverage.json` as `n_R_with_a56_fallbacks`: 101,556,712,720 → 100,907,352,784 per RU at 6500/1599. Note: the 18:45Z commit's `coverage.md/json` had been regenerated without the two count inputs (configurations empty) — restored in this commit from `lane/record-6k` / `lane/partition`.

## Next

`report.md` (final).
