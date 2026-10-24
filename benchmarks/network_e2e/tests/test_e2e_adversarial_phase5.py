"""Red-team (phase 5): what the end-to-end report (`e2e.py`) marks OK on the advice/varied profile against what it
establishes. Each case is a pure call into the real `report`/`negative`, and each PASSES by exhibiting the gap, with the
one-line fix in its docstring; none changes the code. The report itself is candid (its `stubbed`, `not_covered` and
zero-knowledge-gaps say most of this in prose); these cases pin the parts the OK boolean and the Negatives section fold
in regardless.

1. A negative on the advice profile reads as "refused" from any refusing check verdict, not from the violation it names:
   a drop/late/extra run that the rehearsal's own check refused for an UNRELATED reason (a coverage/status fault, e.g. a
   stalled ssh forward) still counts, and the report prints "refused by the rehearsal's own check". `negatives.py`'s own
   `judge` ties refusal to the named code (`by_code`); `e2e.negative` on the advised profile does not.
   Fix: on the advised profile require the check's `io.violations`/status to name the negative's own violation, not just
   `verdict == "refused"` (one clause in `e2e.negative`).
2. The advised report is OK while almost nothing ran: proofs not run, N2 gives no verdict, the advice is not bound to the
   runs and is never rebuilt (`bits.rebuilt` is None, which passes `is not False`), and NEG's registry did not run. The
   OK rests on the honest self-check accepting and the three negatives "refused" (gap 1). A stage that silently didn't
   run does not lower the OK. Fix: the OK (and the headline) on the advice profile should name the floor it establishes
   (certified I/O + self-check only), e.g. gate it on `bits.bound`/`rebuilt` being True where an advice was given, and
   qualify the "# ... OK" title with the profile's floor so the glance matches the body.
3. `ran` reads a negative with no `server.json` as run (`shim_requests` is None, and `None != 0`), rather than not-run.
   The drop commit made `requests == 0` read as not-run, but a missing file is None, not 0.
   Fix: `ran = isinstance(shim_requests, int) and shim_requests != 0`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()
chain = e2e.chain
ADVISED = {"dir": "d", "gate": None}


def _run(verdict: str, violations: list[str], requests: int | None = 40) -> dict:
    """A run as `report` reads it on the advice profile: no N2 proofs, the rehearsal's own check its only verdict."""
    io = {"accepted": verdict == "accepted", "requests": requests, "tokens": 1, "finish": None,
          "violations": violations}
    return ADVISED | {"shim_requests": requests, "proofs": None,
                      "network": {"accepted": None, "windows": None, "refused": None, "digest": None},
                      "check": {"verdict": verdict, "io": io}}


def test_a_negative_is_refused_by_any_check_verdict_not_by_its_named_violation():
    """Gap 1: on the advice profile a drop run the check refused for an unrelated coverage fault reads as refused, and
    by "the rehearsal's own check", the same words the report prints for an honest drop refusal."""
    confounded = _run("refused", ["coverage-never-established"])  # NOT drop's finish/counts violation
    got = e2e.negative(confounded, [])
    assert got["refused"] is True and got["ran"] is True
    assert got["by"] == ["the rehearsal's own check"]  # no mention of which violation, and none required

    # a drop the self-check accepted is (correctly) not refused -- the verdict is the only lever the advised path reads
    accepted = _run("accepted", [])
    assert e2e.negative(accepted, [])["refused"] is False


def test_the_advised_report_is_ok_while_nothing_but_the_self_check_ran():
    """Gap 2: OK on the advice profile with proofs not run, no N2 verdict, the advice unbound and never rebuilt, and
    NEG's registry not run; the three negatives "refused" only by the self-check, for confounded reasons (gap 1)."""
    runs = {"honest": _run("accepted", []),
            "late": _run("refused", ["coverage-never-established"]),
            "extra": _run("refused", ["coverage-never-established"]),
            "drop": _run("refused", ["coverage-never-established"])}
    n2 = {s: {"windows": [], "network": None} for s in runs}  # N2 gives no verdict on the advice profile
    proofs = {"ran": False, "why": "N2-ADVICE hasn't landed"}
    meta = {"profile": chain.ADVICE, "from": ["x"], "commit": "c", "model": "m", "upstream": "u",
            "where": "off the prover's node", "windows": 8}
    stages = {
        "advice": {"file": "advice-0e35.bin", "form": "I1 v0 bytes", "file_sha512": "ab" * 32,
                   "v0_sha512": "ab" * 32, "bits": 169, "requests": 2, "steps": 3, "outputs": 6},
        "bits": {"T": 60, "v0_total": 169, "config": None, "bound": False, "public_fields": True, "rebuilt": None,
                 "why": None, "bound_why": "the advice's 2 requests aren't the 40 certified ones", "windows": []},
        "prove_path": {"ok": True, "advice_is_given": True, "arms": {}, "dir": "pp0e35"},
        "negatives": {"ran": False, "why": "its rows take N2's verdict, which the advice profile has none of"},
    }
    r = e2e.report(runs, n2, proofs, meta, stages)

    assert r["ok"] is True
    assert r["proofs"]["ran"] is False                       # nothing proved in ZK
    assert r["link_windows"] == []                           # N2 gave no verdict
    assert stages["bits"]["bound"] is False                  # advice not these runs'
    assert stages["bits"]["rebuilt"] is None                 # the rebuild from the certified I/O never ran
    assert stages["negatives"]["ran"] is False               # NEG's registry did not run
    assert all(n["refused"] and n["by"] == ["the rehearsal's own check"] for n in r["negatives"].values())

    md = e2e.markdown(r)
    assert md.startswith("# Network accounting end to end: OK")       # the glanceable headline
    assert "**drop**: refused by the rehearsal's own check" in md     # reads as proved-against-drop; it is not


def test_a_negative_with_no_server_json_reads_as_run():
    """Gap 3: a run whose server.json is missing (shim_requests None) reads as run, where requests == 0 reads as not."""
    assert e2e.negative(_run("refused", ["x"], requests=None), [])["ran"] is True
    assert e2e.negative(_run("refused", ["x"], requests=0), [])["ran"] is False
