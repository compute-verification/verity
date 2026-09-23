"""R19 REFERENCE REGRESSION -- does a NEW vLLM integration (v2) numerically reproduce the v1 reference?

THE REFERENCE.  v1 = `lane/vllm-integration` at release tag `r19-int-21` (commit 89af272a8c1d37a44b3e8f3253378d7a152fc281, tagged
`v1-reference`) and the R19 rows it recorded.  Twelve rows are frozen in `fixtures.toml` (row -> files -> sha256/bytes) and
`expected/<row>.json` (the contract, lifted from each row's OWN documents by `lift_expected.py`, never typed by hand):

    GREEN   #11 llama32-1b B1 i4096/o512 (l40s)   #23 llama32-1b B64 (l40s)   #39 qwen25-15b B1 i4096/o512 (l40s)   #60 mistral-7b B8 (l40s)
            #67 olmoe-1b-7b B32 (l40s)            #68 olmoe-1b-7b B32 mixed-arrivals (l40s)   #73 qwen3-4b B8 (h100)   #74 qwen3-4b-fp8 B8 (h100)
    BY NAME #4 smollm2-135m (R17 row; only the CPU record audit exists)   #57 gemma2-2b B8 -- the NEGATIVE: Commit REFUSED at manifest build
            (unmodelled Bf16MulScalarTensor_v1 unnamed crossing) and that refusal must reproduce -- or the F-r19-int-20 decision below
            #70 olmoe TP2 B8 (no fold Match, Commit rc=1)   #75 qwen3-30b-a3b TP2 B2 (rank Programs only, no manifest)

WHAT "REPRODUCES" MEANS (one check module per oracle under `checks/`; each lifts `expected` from the record and recomputes `actual`
with the current code, and the two are diffed as canonical JSON on every field the expected document names):

    manifest_digest       the Program digest bit-for-bit and the manifest of record byte-identical on IDENTITY_KEY
                          (`required_manifest.IDENTITY_KEY` = program_digest, request_id, step, invocation, op_path, output_member,
                          element_range, rank): manifest_digest, identity count, identity_rows_sha256      STRONG   T0 (rebuild: CPU minutes)
    program_digest        the digest chain of the record is one digest (build_summary == artifact.json == workload_program ==
                          the Commit's key); the T2 rebuild (derive_step on the meta device, 1-4 h, 18-181 GB) is documented, not run   STRONG   T2
    global_match_checks   GM-01 verdict, G1..G8 status + problem counts, X-09 verdict / XREQ total, alternate criterion, lag account   weak*   T0
    executed_prefix       executed_prefix_of_record re-run over the record's served tokens / caps / declared EOS + LAG == tokens.json,
                          G6 lengths, eos_stop_requests                                                                               weak*   T0
    coverage              coverage_check(manifest of record, instrumented layouts) == the Commit's recorded coverage                    STRONG   T0(+manifest)
    commit_summary        commit_summary.summarize re-derivation == recorded (stable fields) + commit/verdict.json lifted as the verdict   STRONG   T0
                          (commit_summary.json is deprecated as a verdict, F-r19-int-21; its execution-configuration fields stay the contract)
    verdict               verdict.json (vllm-verdict/v1; read, or rebuilt in memory from the Commit evidence): outcome == the fixture
                          class, compat.commit_pass == v1, every check group == the v1 facts of runs.jsonl, #57 names fa2_layers_bad     STRONG   T0
    decomp_hashes         per-request canonical_sha256{a,b}, derived_digest, n_instances, LP / T / steps / prefill chunks, XREQ counts
                          and the attribution table of match_decomp.json recomputed over the 0.6-5.8 GB fold record by the v2 path
                          (observe.iter_observations -> resolve -> resolve_decomp; batch_decomp.Projection / compare_steps wrapped)     STRONG   T1
    replay_partition      the sampled-replay PARTITION of sampled_replay_p0.json (population by family / not-evaluable by name / strata
                          sizes / lifetime split / R17-1 not-executed split / the seeded draw's per-stratum counts and picks per
                          (request, step) / phase anchors) recomputed over the Programs + manifest of record; the evaluations
                          themselves (binding map, vault only) are carried as facts, not recomputed                                   STRONG   T1

    * weak = PASS-side only (no reference row fails a G-check) or a rule no reference row exercises (no EOS stop anywhere: lag.exercised
      is false on every row) -- the check catches a v2 that changes a verdict or a count, not one that changes the unexercised branch.

CPU-ONLY.  Every check here runs without a GPU.  T0 inputs are <= ~100 MB per row (the small records); `manifest_digest` needs the
row's request Programs (50-300 MB of instances.json.gz) and rebuilds for minutes; `coverage` needs the 39-222 MB manifest.json.  T1
needs the multi-GB fold records / binding maps; T2 the Build rebuild.  NONE of it is meant for the laptop tonight: run on a pod.

NOT PINNED BY ANY RECORD (gaps -- do not read a green run as covering these): committed bytes / the arena, opening proofs (counts only),
challenge derivation, C2 replay inputs, the weights root beyond `weights_pin`, Match per-instance attribution tables, fast-vs-record
implementation equivalence, the fail-closed paths (one negative: #57), EOS / lag rows, the TP2 fold Match.

LABELLED DECISIONS (design rule, Draft 3 §7: a difference from a reference manifest is sanctioned only by a recorded decision naming
the finding or the changed policy).  `[[rows.<row>.decisions]]` in fixtures.toml: `check`, `since` (release / commit), `finding`,
`reason`, `expected` (values the check's `actual` must state).  A check whose result differs from v1 consults them (`resolver.sanction`):
ALL of one decision's `expected` reproduced -> PASS with `sanctioned_by=<finding>` in the check's canonical JSON (junit property too);
neither v1 nor a decision reproduced -> FAIL naming both.  Malformed / `expected`-less decisions sanction nothing.  Today one decision:
#57 `manifest_digest` since r19-int-22 (F-r19-int-20: `Bf16MulScalarTensor_v1`, classified interior BY NAME by v1, is a module-body
output read by layer 0; promotion adds exactly its 432 identities -> 163484, complete) -- `test_negative_57` accepts the refusal OR it.
A decision's `status` is `accepted` (absent = accepted: Daniel confirmed it, sanctions under every engine) or `pending_review`
(labelled by a lane, awaiting Daniel: consulted ONLY when the actual is not the v1 baseline -- `VERITY_REGRESSION_ENGINE=v2` or a
candidate document -- so the v2 engine may reproduce v1 OR the decision, and which one is recorded as junit property `matched`; under the
v1 engine the decision is listed as "not consulted" and the baseline must still reproduce v1).  Today two pending: #67 / #68
`manifest_digest` (F-r19-int-24: `MoeSum_v1`, FusedMoE.forward's returned tensor, read by the next layer's fused add+norm -> boundary
under Q_module_body_v1; v1 classed it interior by name).  Decisions are hand-written and carried over when lift_expected.py
regenerates fixtures.toml.

HOW TO RUN.  Skipped unless `VERITY_REGRESSION=1`.  Inputs resolve per `resolver.py`: a directory of row dirs (`VERITY_REGRESSION_ROWS_ROOT`),
the recorded live paths (tp2), or the data store (`store_io.fetch` of the `fixture/v1` trees named in fixtures.toml; snapshot
`r19-reference-v1`).  Self-check (the current code against its own records):

    cd integrations/vllm && VERITY_REGRESSION=1 VERITY_REGRESSION_ROWS_ROOT=/workspace/regress/rows \\
        python -m pytest tests/regression -m regression -k "T0" -q

A v2 implementation checks itself by pointing `VERITY_REGRESSION_CANDIDATE` at a directory of its own row dirs (`<root>/<row>/manifest.json`,
`global_match.json`, `commit_summary.json`, `commit/verdict.json`, `build_summary.json`, ...): every check then reads the candidate's document
instead of recomputing, and diffs it against the frozen expected:

    cd integrations/vllm && VERITY_REGRESSION=1 VERITY_REGRESSION_CANDIDATE=/path/to/v2/rows python -m pytest tests/regression -m regression -q

Markers: `regression` (all of these), `pod` (T1/T2 and anything reading > 100 MB), `weak` (PASS-side / unexercised).  Green tonight =
T0 byte-equal on every row where the inputs resolve + the #57 refusal reproduces + tests/ir green.
"""
from __future__ import annotations

import os

import pytest

from . import checks, resolver, store_io

pytestmark = pytest.mark.regression

ENABLED = os.environ.get("VERITY_REGRESSION") == "1"
TIER_ENV = "VERITY_REGRESSION_TIERS"           # e.g. "T0" or "T0,T1"; default T0 only


def _tiers() -> set[str]:
    return set((os.environ.get(TIER_ENV) or "T0").split(","))


def _params():
    if not resolver.FIXTURES.is_file():
        return []
    out = []
    for row in resolver.rows():
        for chk in checks.all_checks():
            marks = [pytest.mark.regression]
            if chk.TIER != "T0" or chk.NAME in ("coverage", "manifest_digest"):
                marks.append(pytest.mark.pod)
            if chk.STRENGTH == "weak":
                marks.append(pytest.mark.weak)
            out.append(pytest.param(row.key, chk.NAME, marks=marks, id=f"{chk.TIER}-{chk.NAME}-r{row.number}"))
    return out


@pytest.fixture(scope="session")
def store_state() -> dict:
    return {"reachable": store_io.reachable() if ENABLED else False, "candidate": os.environ.get(resolver.CANDIDATE_ENV)}


@pytest.mark.parametrize("row_key,check_name", _params())
def test_reproduces(row_key: str, check_name: str, store_state: dict, record_property) -> None:
    if not ENABLED:
        pytest.skip("set VERITY_REGRESSION=1 to run the R19 reference regression")
    chk = checks.load(check_name)
    if chk.TIER not in _tiers():
        pytest.skip(f"{check_name} is tier {chk.TIER}; set {TIER_ENV}={chk.TIER} to run it")
    row = next(r for r in resolver.rows() if r.key == row_key)
    if not row._roots() and not (row.spec.get("artifacts") or {}) and not store_state["candidate"]:
        pytest.skip(f"{row.key}: no replica reachable (no rows root, no live dir, no store artifact ids)" + ("" if store_state["reachable"] else "; store not reachable"))
    why = chk.applies(row)
    if why:
        pytest.skip(f"{check_name} does not apply to #{row.number}: {why}")
    exp = row.expected().get(check_name)
    if exp is None:
        try:
            exp = chk.expected(row)
        except resolver.NotResolvable as e:
            pytest.skip(f"expected/{row.key}.json lacks {check_name} and it cannot be lifted here: {e}")
    try:
        act = chk.actual(row)
    except NotImplementedError as e:
        pytest.skip(f"stubbed: {e}")
    except resolver.NotResolvable as e:
        pytest.skip(str(e))
    problems = resolver.sanction(row, check_name, act, resolver.diff(exp, act))     # a labelled decision may sanction a difference from v1
    if act.get("sanctioned_by"):
        record_property("sanctioned_by", act["sanctioned_by"])
        record_property("decision_status", act.get("decision_status") or "accepted")
    record_property("matched", resolver.matched(act, problems))                     # which side reproduced: v1 | decision <finding> (<status>)
    record_property("engine", resolver.engine())
    strength = "WEAK (PASS-side / unexercised rule)" if chk.STRENGTH == "weak" else "strong"
    what = "the v1 reference" + (" nor any labelled decision on the fixture" if row.decisions(check_name) else "")
    assert not problems, (f"#{row.number} {row.key}: {check_name} [{chk.TIER}, {strength}] does not reproduce {what} "
                          f"(source={act.get('source')}):\n  " + "\n  ".join(problems[:40]) + f"\n  actual: {resolver.canonical(act)[:1500]}")


def test_negative_57_refusal_reproduces(record_property) -> None:
    """#57 gemma2-2b: the v1 Commit REFUSED to build the manifest (unmodelled `Bf16MulScalarTensor_v1` unnamed crossing) -- the negative
    case.  The current code, rebuilding from #57's Programs, must refuse the same way (no complete manifest; the unmodelled family named)
    OR state exactly what a labelled decision on the fixture sanctions (`[[rows.<row>.decisions]]`, check `manifest_digest`; today
    F-r19-int-20: the promoted, complete manifest) -- anything else fails naming both."""
    if not ENABLED:
        pytest.skip("set VERITY_REGRESSION=1 to run the R19 reference regression")
    row = next((r for r in resolver.rows() if r.number == 57), None)
    if row is None:
        pytest.skip("#57 not in fixtures.toml")
    if row.programs_root() is None and row.candidate_root() is None:
        pytest.skip("#57: request Programs not reachable here")
    from .checks import manifest_digest
    act = manifest_digest.actual(row)
    refused = act.get("rebuild_rc", 0) != 0 or act.get("complete") is False
    if refused:
        tail = (act.get("rebuild_tail") or "") + resolver.canonical(act)
        assert "Bf16MulScalarTensor" in tail or "unmodelled" in tail, f"#57 refused, but not by the recorded name: {tail[-600:]}"
        return
    v1 = [(f"expected the v1 REFUSAL (complete: false, unmodelled Bf16MulScalarTensor_v1 unnamed crossing); got complete={act.get('complete')!r}, "
           f"identities={act.get('identities')!r}, manifest_digest={act.get('manifest_digest')!r}")]
    problems = resolver.sanction(row, manifest_digest.NAME, act, v1)
    what = "nor any labelled decision on the fixture" if row.decisions(manifest_digest.NAME) else "(fixtures.toml records no decision for #57/manifest_digest)"
    assert not problems, f"#57: the current code does not reproduce the v1 refusal {what}:\n  " + "\n  ".join(problems) + f"\n  actual: {resolver.canonical(act)[:1500]}"
    record_property("sanctioned_by", act["sanctioned_by"])
    record_property("decision_status", act.get("decision_status") or "accepted")
    record_property("matched", resolver.matched(act, problems))
    record_property("engine", resolver.engine())
