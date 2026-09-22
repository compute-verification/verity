# A5/6 checkpoint 18:45Z — coverage table delivered; coverage_v1 module drafted

Based on: mainline `346aa4ac`, worktree `contract/a56-coverage`. Deliverables in this commit:

- `out/contract/a56-coverage/coverage.md` + `coverage.json` (schema `veritor/a56-coverage/v1`, generator `vllm-poc/e2e/coverage_v1/kind_table.py`; reproduce with `PYTHONPATH=vllm-poc:src python -m e2e.coverage_v1.kind_table --expected-work <lane/record-6k:out/record-6k/expected_work.json> --census <lane/partition:out/census/census_small.json> --out out/contract/a56-coverage`).
- `out/contract/a56-coverage/conformance-rows.md` (R9 EA in verifier_v2 V5; R10 GemmCoordinate matches on the finite domain, **HC on any non-finite word** (tc.rs NaN/Inf ban vs total registered primitive); R18 numbers).
- `vllm-poc/e2e/coverage_v1/{__init__,coverage,kind_table}.py` — module drafted; tests next.

## Headline

Under every approval scenario (approved only / + range guest / + fa2 guest) the as-built unsupported set is identical: every kind except GemmCoordinate {K=576, K=1536} and Input — n_R = 101,556,712,720 per RU at 6500/1599 (37 of 40 kinds), N = 406,226,850,880 over 4 RUs; at s = 2⁻³⁰ the expected number of unsupported selected checks per RU is 94.58 and the probability of at least one is 1 − 10^(−41). The range guest adds no kind (identical gate sets — it adds framings/range openings), the fa2 guest adds none at per-VU granularity (`in` is 16-bit: whole-head only). Completion of a P1/P1v2 record depends on **builders + committed interiors**, not on approvals. The checker-missing floor (if every builder the approved guest can already serve existed) is the FTZ attention interior, 10 kinds, n_R = 45,039,253,230.

## coverage.json schema (v1)

```
{ schema: "veritor/a56-coverage/v1", based_on, label, status_vocabulary[], scenarios{approved, plus_range, plus_fa2},
  guests{<id>: {approval, source, families, framings, ...}},
  kinds: [ { kind, legacy, aliases[], circuit (file:line), 
             checker: {exists, guests[], gate_set, ops[], evidence[], granularity, domain_caveat},
             builder: {exists, evidence[], granularity, note}, framings[], note,
             status: {approved, plus_range, plus_fa2}  # one of status_vocabulary } ],
  configurations: { "<shape>": { source, N_R, N_V_per_RU, kinds, counts_per_RU{kind: n},
      by_scenario: { <scenario>: { unsupported_kinds[], n_R, N_4RU, checker_missing_kinds[], n_R_checker_missing_only,
                                   N_4RU_checker_missing_only, checker_missing_kinds_if_larger_units_allowed[], n_R_checker_missing_if_larger_units_allowed,
                                   exposure_per_RU{ "2^-14"|"2^-19"|"2^-30": {probability, complement_log10, expected_unsupported_selected (exact fraction), expected_unsupported_selected_decimal} },
                                   exposure_4RU{...}, exposure_per_RU_if_every_builder_existed{...}, exposure_4RU_if_every_builder_existed{...} } } } } }
```
A9's runner: read `configurations[<shape>].by_scenario[<scenario>].n_R / N_4RU / exposure_*`; A7's certificate: `kinds[*].status[<scenario>]` per selected kind, plus `label`.

## Proposed R9 signatures (for A3's verifier_v3 — please confirm before A3 depends on them)

```python
from e2e.coverage_v1 import expected_statements, check_coverage, statement_digest, obligations_of_typed_object
from e2e.coverage_v1.coverage import Located, ExpectedStatement, ObligationView, AcceptedProof, CoverageReport, CoverageError

# circuit.kind_program(located: Located) -> (family_id: str, program_digest_hex: str)   # Protocol
# partition.locate(vu_id: str) -> Located(vu_id, kind, ru, inputs: tuple[Position], outputs: tuple[Position], pins)
#   Position = (tree_id, leaf_canonical_id, *offset)   # the committed word; tree_id so root binding is checkable
expected_statements(circuit, partition, Q: Iterable[str|int]) -> dict[vu_id, ExpectedStatement]   # raises CoverageError (refusal)
check_coverage(expected, proofs: Iterable[AcceptedProof], *, strict_unexpected=True) -> CoverageReport
#   AcceptedProof(proof_id, obligations: tuple[ObligationView]); ObligationView(family, program_digest, inputs, outputs)
#   CoverageReport.covered{vu: proof_id} / .uncovered / .duplicate{vu: proof_ids} / .unexpected[(proof_id, digest)] / .ok / .per_vu() / .to_json()
obligations_of_typed_object(TypedObligationSet | its to_json()) -> tuple[ObligationView]   # from the BOUND object only
statement_digest(family, program_digest, inputs, outputs) -> hex   # tagged sha256 "veritor/coverage_v1/statement" over canonical JSON
```
Precondition (documented): SP1 verify under an approved (ELF, vk) + `check_binding` passed for each proof before its obligations are passed in. The prover's `selected_vus` / `provenance.units` are never read.

## CLAIMS / INTERFACE QUESTIONS

- **INTERFACE (A3/integrator):** the `Located` positions need the partition to name, per selected VU, the committed `(tree, leaf canonical id, offset)` of every boundary word — today only `mapping.units.jsonl` (prover-side) does that for GemmCoordinate. Who owns the verifier-side locator (`partition_sample.py`/population, A1/A3)? My module consumes it through the `Partition` Protocol; I will provide a GemmCoordinate/InputEq reference locator over `capture` layouts if A2 confirms the canonical-id grammar (`p=…/op=…/port=0`, `w=<param>/c=<row>`).
- **INTERFACE (A2):** the R10 well-typing reading (value-checked at the consuming gate, schema width prover-chosen; `dtype`/`shape_text` hashed but not compared to the schema) is at the commitment level — please confirm or take it as an A2 row.
- **CLAIM (none new).** No guest edits. PROPOSED guest patch to raise with the integrator: make `tc_dot` total per the registered extension (as `zk/sp1-attn/common/src/fa2.rs:254-298` already does) — removes the HC on non-finite words; under the user's security gate.
- **Fallback builders (objective 4):** the only kind whose whole boundary is committed today besides GEMM is `SiluMulBf16_v1|batch` (gate_up output row → down-proj input row), `Bf16AddF2fp_v1|batch` (fused residual add) and `TokenSelect` (logits row → token id). A record-wired builder for these needs the family to be admitted to `veritor.interface.typed_obligation.family_gate_set` (`src/veritor/interface/typed_obligation.py:1011-1036` knows only tc-ampere-bf16/tensor-core/toy families — `elem-bf16@1`, `reduce-mufu@1` are NOT typed-object families, so `check_binding` cannot bind them today). That file is outside my writable set → **CLAIM: `src/veritor/interface/typed_obligation.py` `family_gate_set`/`SUPPORTED_FAMILIES` extension for `veritor.elem-bf16@1` and `veritor.reduce-mufu@1`** (or the integrator assigns it). Until then a fallback builder can only emit raw format-2 statements (bench style), which `verifier_v2` cannot bind.

Next: tests for `coverage_v1` (duplicate-hides-omission, empty Q, batching, larger unit, refusal), then the SiluMulBf16 builder if the claim is granted, then report.md.
