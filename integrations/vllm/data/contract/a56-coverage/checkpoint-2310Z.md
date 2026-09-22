# A5/6 checkpoint 23:10Z — mainline merged; forced head run on the real TINY record; census of the unsupported selection

Branch `contract/a56-coverage` on mainline `b745af3b` (merged first, clean). Tests: `tests/contract -k "a56 or coverage_v1 or elem or builders"` 20 passed.

## 1. Patches — all three ALREADY APPLIED by A3 ckpt 7; none to apply (`patches/STATUS.md`)

`git apply --check` forward and reverse both fail on `b745af3b` because the content is present with A3's edits: `locator.py:44/318-323` (`ATTENTION_KINDS`, `locate_attention`), `expected.py` (`Deriver.attention_row/.config`, head → `head_expectation`, `kind_program` head, `coverage_inputs(config=)`), `rec_p1_obligations.py:151/247-260` (`BUILDERS` dispatch). No double application. `forced_head_run.py` now detects the mainline content and runs without its patched overlay.

## 2. Q-A56-9 — CLOSED, report corrected

Confirmed on mainline: `check_family_approved` is called as `V4.family.<chunk>` (`reverify.py:564-573`, fail closed); `approved_guests.json` lists 8 families on `9fd00e60` incl. A2's `elem-bf16@1` / `reduce-mufu@1`. Report §(m) row rewritten; the harness shows the check refusing the fa2 family under `9fd00e60` with the list quoted.

## 3. Q-A56-6 — forced head run on the TINY record: covered / neighbour refused / fa2 refused (report §(j), `records/forced_head_run-TINY-*.json`)

APFS clone of `A9-TINY-04137cde/V5-TINY-bex0-inc` → `/tmp/a56/rec-tiny`; `lane_run --gb 3 --class verify-sliced --lane a56`; MAINLINE `verifier_v3`; two FORCED (labelled) units, `t=3/layers.0/attn` head 0 (prefill T=4) and `t=5/layers.0/attn` head 0 (decode T=6), RU 0:

| step | side | result | s |
|---|---|---|---|
| program rebuilt + `ProgramLocator` + `head_interval` | verifier | 242 675 685 gates, digest = `maps.json`; unit placed | 0.6 |
| `coverage_inputs(config=C)` + `expected_statements` | verifier | Q_proof = {forced}; expectation from committed ids: 576 in / 64 out (T=4), subcircuit `2cc714e8…` | 3.8 |
| `dispatch(LocatedUnit, CaptureRecord)` | prover | fa2 statement from the record's committed rows: 2 obligations, 233 gates, 26 840 B + witness 17 218 B; fold == committed out row; transparent check 6 ms | 0.26 |
| `head_views_of_raw_statement` + `check_coverage` | verifier | views == builder's; **covered = {forced}, unexpected 0** | 0.017 |
| R9 negative (neighbouring head 1) | verifier | **uncovered {forced} + 1 unexpected → refused** | 0.036 |
| refusal of fa2 without a host | verifier | `approval_of(33cba058…)` → `GUEST_NOT_APPROVED`; `check_family_approved(fa2 family, 9fd00e60)` → not implemented (list quoted) | 0.001 |

No fa2 proving on the laptop (pod MEASUREMENT stands: T=9 core 624 s; approved-host verify-many refusal 2–6 ms). **Q-A56-10:** the proof artefact in the harness output needs a ≤ 1 h CPU pod slot (harness ready, `--fa2-host/--approved-host`); it adds no new number.

**F2 (defect):** `attn_gates.cached_block(D, 64, nvis, FIRST=True)` emits outgoing gates out of order for nvis ∈ {3, 5, 7} → `_lower_subcircuit` refuses ("kind outputs must be strictly increasing"); head units at t ∈ {2, 4, 6} have no expectation/statement today (forced t=2 refuses at `verifier.expected_statement`). Probed nvis 1…65 at D=64/128: exactly those three. Fix in `zk/bench/attn_gates.py` (construction order); 3/1599 positions at scale.
**F3 (gap):** the raw fa2 statement has no typed object → A2's `position_check` (via `check_coverage(trees=, rank_of=)`) refuses "no typed object" and covers nothing; positions are checked by `head_views_of_raw_statement` against committed ids instead. For the certificate path O1 needs a typed-object head statement (or a raw-statement adapter for `position_check`).

## 4. Census of the unsupported selection (report §(n); `records/unsupported-census-{TINY,32x15}.{json,md}`; by LOCATION via `ProgramLocator`)

| bucket | TINY 653 | 32×15 1 479 | owner |
|---|---|---|---|
| attention body (`…/attn`) | 112 (17.2 %) | 355 (24.0 %) | A5/6 — uncommitted interior; head VU PROPOSED |
| CK-ELEM's four (buildable) | 136 (20.8 %) = SiluMul 69 · Bf16Add 40 · RopeOut 15 · RopeOutAdd 12 (matches A3) | 346 (23.4 %) | CK-ELEM |
| rest: RMSNorm pieces in `pre`/`post` (`F32Mul\|batch` 111/203, `Bf16ToF32\|batch` 103/238, `F32Add\|prim` 48/99, `F32Mul\|prim` 48/91, non-attn `F2fpBf16\|batch` 50/94), `final` `ArgmaxStep` 45/51, `embed` gather 0/2 | 405 (62.0 %) | 778 (52.6 %) | assigned to CK-ELEM in the 20:40Z brief (sub-1.2 % kinds); **no builder on `b745af3b`** — integrator to confirm the queue or name a third owner |

`F2fpBf16|batch` splits by location (attn 19/51 → A5/6 interior; pre/post → CK-ELEM). Attention units occur on every step (prefill and decode); the head builder covered a decode head (t=5) on the record.

## 5. New/changed files

`vllm-poc/e2e/coverage_v1/forced_head_run.py` (mainline detection, string vu keys, hex roots, rank-ordered ids, verifier-side fa2 refusal, typed-position-check report), `vllm-poc/e2e/coverage_v1/unsupported_census.py` (new), `out/contract/a56-coverage/{report.md §(j)(k)(m)(n), patches/STATUS.md, records/*}`.

Stopping, per 23:12Z. F1 (fa2 560 s floor → 22–33 CPU-h/RU at s = 2⁻¹⁴) and O1/O2 go to the user with decision (12) as framed in `report.md` §(i)/(l).
