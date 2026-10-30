"""Adversarial cases, turned into regressions of what the end-to-end report (`e2e.py`) marks OK on the advice
profile, each a pure call into the real `report`/`negative`:

1. A negative is refused by the rehearsal's own check only for its own violation (`e2e.OWN_CODE`): a drop the check
   refused for a coverage fault is not drop's refusal.
2. The OK needs a given advice to be these runs' and rebuilt from their certified I/O, and the title says what ran.
3. The prove path's committed advice is held to the ``--advice`` file, not to these runs, and an unbound prove path
   stays out of the OK; the serving config the job was given stays out of it too.
4. A run with no ``server.json`` didn't run, as one whose shim received no request.
"""

from __future__ import annotations

import importlib.util
import json
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


def _run(verdict: str, codes: list[str], requests: int | None = 40) -> dict:
    """A run as `report` reads it on the advice profile: no N2 proofs, the rehearsal's own check its only verdict, which
    found ``codes``."""
    io = {"accepted": verdict == "accepted", "requests": requests, "tokens": 1, "finish": None,
          "violations": [[0, c[3:]] for c in codes if c.startswith("io:")]}
    return ADVISED | {"shim_requests": requests, "proofs": None,
                      "network": {"accepted": None, "windows": None, "refused": None, "digest": None},
                      "check": {"verdict": verdict, "codes": codes, "io": io}}


def test_a_negative_is_refused_by_its_own_violation_not_by_any_refusing_verdict():
    confounded = _run("refused", ["audit.status:coverage-never-established"])
    got = e2e.negative(confounded, [], "drop")
    assert got["refused"] is False and got["ran"] is True and got["by"] == []
    assert got["own_code"] == "io:short of max_tokens without EOS"
    assert e2e.negative(confounded, [], "late")["by"] == [
        "the rehearsal's own check (audit.status:coverage-never-established)"]
    own = e2e.negative(_run("refused", ["io:short of max_tokens without EOS"]), [], "drop")
    assert own["refused"] is True and own["by"] == ["the rehearsal's own check (io:short of max_tokens without EOS)"]
    assert e2e.negative(_run("accepted", []), [], "drop")["refused"] is False
    assert e2e.negative(confounded, [])["refused"] is False


#: the I/O tie, required on the advice profile (`chain.tie_gate`), accepting the honest run
TIED = {"ran": True, "why": None, "from": "ids.jsonl", "stand_ins": {}, "runs": {"honest": {
    "verdict": "accepted", "ok": True, "codes": [], "failing": {}, "windows": [], "unplaced": 0, "changed": None,
    "requests": 40, "outputs": 40, "placed": 40, "held": {"egress": 8, "ingress": 8}, "missing": {}, "finish": True,
    "why": None}}}


def _advised(stages: dict, codes: str = "own") -> dict:
    found = {"late": "audit.status:coverage-never-established", "extra": "io:longer than max_tokens",
             "drop": "io:short of max_tokens without EOS"}
    runs = {"honest": _run("accepted", [])} | {
        s: _run("refused", [found[s] if codes == "own" else "audit.status:coverage-never-established"])
        for s in ("late", "extra", "drop")}
    n2 = {s: {"windows": [], "network": None} for s in runs}
    meta = {"profile": chain.ADVICE, "from": ["x"], "commit": "c", "model": "m", "upstream": "u",
            "where": "off the prover's node", "windows": 8}
    return e2e.report(runs, n2, {"ran": False, "why": "no N2 unit holds the advice profile's egress counts"}, meta,
                      {"tie": TIED} | stages)


UNBOUND = {
    "advice": {"file": "advice-0e35.bin", "form": "I1 v0 bytes", "file_sha512": "ab" * 32, "v0_sha512": "ab" * 32,
               "bits": 169, "requests": 2, "steps": 3, "outputs": 6},
    "bits": {"T": 60, "v0_total": 169, "config": None, "bound": False, "public_fields": True, "rebuilt": None,
             "why": None, "bound_why": "the advice's 2 requests aren't the 40 certified ones", "windows": []},
    "prove_path": {"ok": True, "committed_is_the_advice_file": True, "bound": False,
                   "bound_why": "the --advice file isn't these runs' advice", "arms": {}, "dir": "pp0e35"},
    "negatives": {"ran": False, "why": "its rows take N2's verdict, which the advice profile has none of"},
}


def test_the_advised_report_isnt_ok_on_an_unbound_advice_and_its_title_says_what_ran():
    r = _advised(UNBOUND, codes="confounded")
    assert r["ok"] is False
    assert r["why_not"] == ["the advice isn't these runs'", "the advice wasn't rebuilt from the certified I/O",
                            "extra not refused", "drop not refused"]
    assert r["negatives"]["late"]["refused"] is True
    md = e2e.markdown(r)
    assert md.splitlines()[0] == (
        "# Network accounting end to end: NOT OK on the rehearsal's own check and the I/O tie (`io_tie`'s reference in "
        "the clear, on stand-ins) (not run: N2's verdicts of record, the proofs (n2_zk, held by Lean), the advice, "
        "these runs' and rebuilt from their certified I/O, the prove path, of these runs, the negatives registry)")
    assert "**drop**: NOT refused by nothing (not proved); the rehearsal's own check refused it for " \
           "audit.status:coverage-never-established, not its own violation io:short of max_tokens without EOS" in md
    bound = UNBOUND | {"bits": UNBOUND["bits"] | {"bound": True, "rebuilt": True}}
    r = _advised(bound)
    assert r["ok"] is True and r["why_not"] == []
    assert r["ran"] == ["the rehearsal's own check", "the advice, these runs' and rebuilt from their certified I/O",
                        e2e.TIE_STAGE]
    assert _advised(bound | {"tie": None})["why_not"] == [
        "the I/O tie didn't run, and the advice profile requires it: no tie stage"]
    assert _advised(UNBOUND | {"bits": bound["bits"] | {"rebuilt": None}})["why_not"] == [
        "the advice wasn't rebuilt from the certified I/O"]


def test_an_unbound_prove_path_and_the_given_serving_config_stay_out_of_the_ok():
    bound = UNBOUND | {"bits": UNBOUND["bits"] | {"bound": True, "rebuilt": True}}
    failed = bound | {"prove_path": bound["prove_path"] | {"ok": False}}
    assert _advised(failed)["ok"] is True
    assert _advised(failed | {"prove_path": failed["prove_path"] | {"bound": True}})["why_not"] == [
        "the prove path failed"]
    off = bound | {"bits": bound["bits"] | {"config": {"block_size": 16}, "public_fields": False}}
    assert _advised(off)["ok"] is True
    md = chain.markdown(off | {"prove_path": None})
    assert any("the one the job was given (--serving-config), not one the endpoint registered" in x for x in md)
    pp = chain.markdown(UNBOUND)
    assert any("committed is the --advice file: True. Bound to these runs: False (the --advice file isn't these "
               "runs' advice), so its ok is not the report's" in x for x in pp)
    assert "advice_is_given" not in json.dumps(UNBOUND)


def test_a_negative_with_no_server_json_didnt_run():
    assert e2e.negative(_run("refused", ["io:short of max_tokens without EOS"], requests=None), [], "drop")["ran"] is False
    assert e2e.negative(_run("refused", ["io:short of max_tokens without EOS"], requests=0), [], "drop")["ran"] is False
    assert e2e.negative(_run("refused", ["io:short of max_tokens without EOS"], requests=3), [], "drop")["ran"] is True
