"""REFERENCE REGRESSION: the negative of workload #57 (gemma2-2b) and the labelled decisions of fixtures.toml.  The checks themselves
are `checks/`'s, each held by its own test in this directory; `resolver.py` resolves a row's inputs (a directory of row dirs, the
recorded live paths, or the data store).

#57's Commit refused to build the manifest (unmodelled `Bf16MulScalarTensor_v1` unnamed crossing) and that refusal must reproduce, or be
sanctioned by a labelled decision.  Skipped unless `VERITY_REGRESSION=1`; it recomputes from the row's Programs (CPU minutes), so it is
marked `rebuild` and runs under `VERITY_REGRESSION_REBUILD=1` (`-m "regression and rebuild"`).

LABELLED DECISIONS (design rule, Draft 3 §7: a difference from a reference manifest is sanctioned only by a recorded decision naming
the finding or the changed policy).  `[[rows.<row>.decisions]]` in fixtures.toml: `check`, `since` (release / commit), `finding`,
`reason`, `expected` (values the check's `actual` must state).  A check whose result differs from v1 consults them (`resolver.sanction`):
ALL of one decision's `expected` reproduced -> PASS with `sanctioned_by=<finding>` in the check's canonical JSON (junit property too);
neither v1 nor a decision reproduced -> FAIL naming both.  Malformed / `expected`-less decisions sanction nothing.  #57's
`manifest_digest` decisions are what `test_negative_57` accepts in place of the refusal.
A decision's `status` is `accepted` (absent = accepted: confirmed) or `pending_review` (labelled, awaiting review);
both sanction, and which side reproduced is recorded as junit property `matched`; `python -m tests.regression.decisions --pending`
lists the pending ones.  A decision may also own an INPUT pin (`pins."manifest.json" = { bytes,
sha256 }`: #57's promoted r19-int-22 manifest) -- `resolver.Row._checked` accepts it where the frozen v1 pin fails, junit property
`pinned_by`.  `python -m tests.regression.decisions` (or `-k decisions -s`) lists them all with their status.  Decisions are
hand-written and carried over when lift_expected.py regenerates fixtures.toml.
"""
from __future__ import annotations

import os
from typing import NoReturn

import pytest

from . import resolver
from . import decisions as D
from .checks import manifest_digest

pytestmark = pytest.mark.regression

ENABLED = os.environ.get("VERITY_REGRESSION") == "1"


def _unresolved(why: str) -> NoReturn:
    """A reference input (or the expected contract) that no replica gives here: a skip, but a failure under VERITY_STORE_REQUIRED=1
    (a recorded check, the post-train job), where only a check retired in fixtures.toml may skip for want of its inputs."""
    if resolver.store_required():
        pytest.fail(f"{why}\n  ({resolver.STORE_REQUIRED_ENV}=1: an input that does not resolve fails; a check that cannot run on a row's "
                    "record is retired by `[rows.<row>.retired.<check>]` in fixtures.toml, with its reason)")
    raise pytest.skip.Exception(why)


def _skip_rebuild(what: str, candidate: str | None) -> None:
    if not resolver.rebuilds() and not candidate:
        pytest.skip(f"{what} recomputes from the row's Programs (CPU minutes to half an hour a row): it runs after each train on "
                    f"main (`tools/check/post_train.py`), not in `check`; {resolver.REBUILD_ENV}=1 runs it here")


def _skip_retired(row: resolver.Row, check_name: str, record_property) -> None:
    ret = row.retired(check_name)
    if ret:
        record_property("retired", ret.get("reason") or "?")
        pytest.skip(f"{check_name} is retired on #{row.number}: {ret.get('reason')} (since {ret.get('since')}; needs {ret.get('needs')})")


@pytest.mark.rebuild
def test_negative_57_refusal_reproduces(record_property) -> None:
    """#57 gemma2-2b: the v1 Commit REFUSED to build the manifest (unmodelled `Bf16MulScalarTensor_v1` unnamed crossing) -- the negative
    case.  The current code, rebuilding from #57's Programs, must refuse the same way (no complete manifest; the unmodelled family named)
    OR state exactly what a labelled decision on the fixture sanctions (`[[rows.<row>.decisions]]`, check `manifest_digest`:
    the promoted, complete manifest; the Q_word v1 manifest of record) -- anything else fails naming both."""
    if not ENABLED:
        pytest.skip("set VERITY_REGRESSION=1 to run the reference regression")
    _skip_rebuild("the #57 negative", os.environ.get(resolver.CANDIDATE_ENV))
    numbered_57 = [r for r in resolver.rows() if r.number == 57]
    if not numbered_57:
        pytest.skip("#57 not in fixtures.toml")
    row = numbered_57[0]
    _skip_retired(row, manifest_digest.NAME, record_property)
    try:
        reachable = row.programs_root() is not None or row.candidate_root() is not None
    except resolver.NotResolvable as e:
        _unresolved(f"#57: {e}")
    if not reachable:
        _unresolved("#57: request Programs not reachable here")
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
    record_property("engine", resolver.ENGINE)


def test_decisions_are_listed_and_well_formed(record_property) -> None:
    """Every labelled decision of fixtures.toml on one screen (`pytest tests/regression -k decisions -s`, or
    `python -m tests.regression.decisions`; no store, no VERITY_REGRESSION needed): row, check, finding, status, since, what it
    sanctions (`expected`) and pins (`pins`), the reason.  Fails on a malformed decision (missing DECISION_FIELDS, unknown status)
    -- `sanction` would ignore it silently otherwise -- so the pending ones (`status = "pending_review"`) are always reviewable."""
    rows_ = D.listing()
    text = D.render(rows_)
    print("\n" + text)
    retired = D.retirements()
    print(D.render_retired(retired))
    record_property("decisions_n", str(len(rows_)))
    record_property("pending_n", str(sum(1 for d in rows_ if d["status"] == "pending_review")))
    record_property("retired_n", str(len(retired)))
    bad = [f"#{d['row']} {d['check']} {d['finding']}: {d['problem']}" for d in rows_ if d["problem"]]
    bad += [f"#{r['row']} {r['check']} retired: {r['problem']}" for r in retired if r["problem"]]
    assert not bad, "malformed labelled decision(s) or retirement(s) in fixtures.toml:\n  " + "\n  ".join(bad)
