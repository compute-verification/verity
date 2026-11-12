"""kind_scan: what the network certifier's piece releases holds only fields of the four public kinds and the network
certifier's certificates. A rows-form period (`period_case`'s accept, its registrations' provenance the scan's and V3c's
committed window's, as `scan_zk` and `prove_path.py window-commit` write them) and its answer with the piece's placement
scan clean as DIR/case.json and as the hold's three period files. A field no declaration names is a finding wherever it
lies: the window's own number in V3c's provenance, a count of requests in an answer, an advice length beside the
parameters, and a file DIR doesn't declare, a log among them. What either development check would feed the record
is a finding naming the check: an advice binding in an audit's registration (what `records_audit.py audit --opening`
opens), at any length, the padded width included, and the chain's endpoint step (CR_ENDPOINT's rebuild), or any step
the network certifier path doesn't run. On a finding the piece refuses, releasing nothing, its findings (file and key path)
in its hold.

What a verifier on another machine receives of a served run's OUT (``--received``: the certifier's piece, V3c's audit,
the registered compiler, the advice and weights registrations and the window circuit's count) scans clean as the run
writes it, and anything that reads the advice or counts the window is a finding: a log left in either dir, an undeclared
file, the advice's own length in its registration, a workload in φ, the advice's length in a count, a compile in the
audit's summary and an advice binding in its registration."""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]

import certifier_piece as CP  # noqa: E402
import kind_scan as KS  # noqa: E402
from verity.circuits.codec import canonical_json  # noqa: E402

sys.path.append(str(ROOT / "benchmarks" / "one_stage"))
import served_verdicts as SV  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "protocols" / "certifier" / "verifier" / "network" / "tests" / "period_case.py"
#: a scan round's registration provenance (`scan_zk`'s), its state's reads and row 0's opening included
SCAN_WINDOW = {"kind": "stand-in", "link": "registered: each row port at the network certifier's signed root",
               "of": "round honest: link-windows [0] of the network certifier's run",
               "members": {"first": {"template": "IngressScanFirst_v3", "units": [0]},
                           "egress-0": {"template": "EgressBucket_v1", "units": [1, 2]}},
               "reads": {"scan/state": {"o1": {"leaf": 1, "units": [0, 1]}, "sin": {"leaf": 1, "units": [1, 2]}}},
               "empty": {"scan/state": {"words": 4, "salt": "07" * 192, "path": "08" * 64}}}
#: V3c's committed window's provenance: `prove_path.py window-commit`'s class and scan state, `rows.registration`'s kind
#: and run root, `Window.write`'s circuit record and registered roots, `rows.write`'s index leaf and order
V3C_WINDOW = {"kind": "served", "run": "ab" * 64, "window_circuit": {"class": [4, 32, 4, 4, 16]},
              "activations": {"program": "window-circuit", "values": {"scan/state": {
                  "value": "scan/state", "root": "cd" * 64, "leaves": 3, "words": 40, "domain": "ef" * 64}}},
              "circuit": "01" * 64, "registered_roots": {"weights/embed@4": "02" * 64}, "index_leaf": "03" * 64,
              "order": "verity-vllm/serving-rows/order/v0"}
PLACEMENT = {"kind": "not-placed", "parties": {"network_certifier": "gate", "prover": "node", "verifier": "node"},
             CP.PREMISE: "assumed, the auditor's devices not placed", "uncounted": "every link of node's but one",
             "links": [{"name": "certified", "from": "gate", "to": "node", "counted": True, "via": "the tunnel"}]}
#: the advice's registered width ℓ_a, and a realized length beside it
L_A, REALIZED = 8388608, 392


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


def _case(period, v3c: dict = V3C_WINDOW) -> tuple[dict, dict]:
    """The rows-form accept, its registrations' provenance V3c's (``v3c``) and the scan's, and its answer as the piece
    writes it."""
    case = period.case("accept", 1, "rows")
    answer = period.reference(case)
    assert answer["verdict"] == "accept"
    com = json.loads(bytes.fromhex(case["commitment"]))
    com = {"audits": [{**r, "window": v3c} for r in com["audits"]],
           "windows": [{**r, "window": SCAN_WINDOW} for r in com["windows"]]}
    return {**case, "commitment": canonical_json(com).hex()}, answer | {"profile": answer["profile"] | {
        "placement": PLACEMENT}}


def _dir(d: Path, case: dict, answer: dict, hold: bool = False) -> Path:
    """DIR as the piece releases it (``case.json``, `certifier_piece.write_case`'s), or its hold (``period/``)."""
    d.mkdir(parents=True)
    (d / "answer.json").write_text(json.dumps(answer))
    if hold:
        (d / "period").mkdir()
        (d / "period" / "commitment.bin").write_bytes(bytes.fromhex(case["commitment"]))
        (d / "period" / "evidence.json").write_bytes(bytes.fromhex(case["evidence"]))
        (d / "period" / "params.json").write_bytes(canonical_json(case["params"]))
    else:
        (d / "case.json").write_bytes(canonical_json(case) + b"\n")
    return d


def _paths(found: list[KS.Finding]) -> list[tuple[str, str]]:
    return [(f.file, f.path) for f in found]


def test_the_pieces_period_and_answer_are_of_the_public_kinds_in_either_layout(tmp_path, period):
    case, answer = _case(period)
    for hold in (False, True):
        found, kinds = KS.scan(_dir(tmp_path / str(hold), case, answer, hold))
        assert found == [] and all(kinds[k] for k in KS.KINDS), kinds
    assert KS.main([str(tmp_path / "False")]) == 0


def test_a_field_no_declaration_names_is_a_finding_wherever_it_lies(tmp_path, period):
    """The window's own number in each audit's V3c provenance (which counts the run's windows), a count of requests in
    the period's answer and in an execution phase's one-stage answer, and an advice length beside the parameters."""
    case, answer = _case(period, V3C_WINDOW | {"window_circuit": {"class": [4, 32, 4, 4, 16], "window": 3}})
    n = len(case["params"]["audits"])
    ev = json.loads(bytes.fromhex(case["evidence"]))
    ev["windows"][0]["answer"]["public"]["requests"] = 7
    case = {**case, "evidence": canonical_json(ev).hex(), "params": {**case["params"], "advice_bits": REALIZED}}
    answer = answer | {"public": answer["public"] | {"requests": 7}}
    want = [("answer.json", "public.requests"),
            *[("case.json", "commitment.audits[].window.window_circuit.window")] * n,
            ("case.json", "evidence.windows[].answer.public.requests"), ("case.json", "params.advice_bits")]
    assert sorted(_paths(KS.scan(_dir(tmp_path / "dir", case, answer))[0])) == want
    held = KS.scan(_dir(tmp_path / "hold", case, answer, hold=True))[0]
    assert sorted(_paths(held)) == [("answer.json", "public.requests"),
                                    *[("period/commitment.bin", "audits[].window.window_circuit.window")] * n,
                                    ("period/evidence.json", "windows[].answer.public.requests"),
                                    ("period/params.json", "advice_bits")]
    assert KS.main([str(tmp_path / "dir")]) == 1


def test_an_audit_that_opened_the_advice_is_a_finding_naming_the_check_at_any_length(tmp_path, period):
    """An advice binding (`interface.advice_binding`: the compiler and the advice's root and length) in an audit's
    registration is what `records_audit.py audit --opening` opens, so it is `OPENING`'s finding (each audit's) at the
    padded width ℓ_a as at the realized length, in either layout, the finding's text holding neither length."""
    for bits in (L_A, REALIZED):
        v3c = V3C_WINDOW | {"compiler": {"name": "i2", "digest": "04" * 64, "commit": "05" * 20},
                            "advice": {"root": "06" * 64, "bits": bits}}
        case, answer = _case(period, v3c)
        n = len(case["params"]["audits"])
        for hold, at in ((False, "case.json"), (True, "period/commitment.bin")):
            found = KS.scan(_dir(tmp_path / f"{bits}-{hold}", case, answer, hold))[0]
            pre = "commitment." if not hold else ""
            assert sorted(_paths(found)) == sorted(
                [(at, f"{pre}audits[].window.advice"), (at, f"{pre}audits[].window.compiler")] * n)
            assert all(f.why == KS.OPENING.why and str(bits) not in str(f) for f in found)
    assert KS.main([str(tmp_path / f"{L_A}-False")]) == 1


def test_the_chains_endpoint_rebuild_or_any_other_step_is_a_finding(tmp_path, period):
    """The report's steps are the network certifier path's (`STEPS`, the five `chain_record.report` writes there, which
    test_certifier_piece holds a published report to): with them it is clean; the endpoint step (CR_ENDPOINT's rebuild,
    and its tie negatives beside it) is `ENDPOINT`'s finding, and a step of no other name is one of no declared kind."""
    def found(steps: dict, name: str) -> list[tuple[str, str, str]]:
        d = _dir(tmp_path / name, *_case(period))
        (d / "chain" / "report").mkdir(parents=True)
        (d / "chain" / "report" / "report.json").write_text(json.dumps({
            "schema": "s", "verdict": "accepted", "path": "warden", "scope": None, "stubbed": ["x"],
            "negatives_refused": {"n2/tampered": True, "scan/forged": True}, "steps": steps}))
        return [(f.file, f.path, f.why) for f in KS.scan(d)[0]]
    step = {"verdict": "accepted", "in_the_clear": "the certificates", "eps_at_delta": None}
    assert found(dict.fromkeys(KS.STEPS, step), "warden") == []
    rel = "chain/report/report.json"
    assert found({**dict.fromkeys(KS.STEPS, step), "endpoint": step}, "endpoint") == [
        (rel, "steps.endpoint", KS.ENDPOINT.why)]
    assert found({**dict.fromkeys(KS.STEPS, step), "other": {"verdict": "accepted"}}, "other") == [
        (rel, "steps.other.verdict", "a str of no declared kind")]


def test_a_file_dir_doesnt_declare_is_a_finding_a_log_among_them(tmp_path, period):
    """A report in another form, the split's chained file in another encoding and each of the steps' logs (which count
    the window's requests) are findings; a refusal's text is declared whole."""
    d = _dir(tmp_path / "d", *_case(period))
    (d / "chain" / "report").mkdir(parents=True)
    (d / "chain" / "report" / "report.md").write_text("Requests: 7\n")
    (d / CP.CHAINED).write_text("not json")
    for log in CP.LOGS:
        (d / log).write_text(f"advice {REALIZED} bits, window 3\n")
    (d / CP.REFUSED).write_text("the chain refused\n")
    assert sorted(_paths(KS.scan(d)[0])) == sorted([(CP.CHAINED, ""), ("chain/report/report.md", ""),
                                                   *((log, "") for log in CP.LOGS)])


def test_dirs_files_are_the_pieces():
    """Every file the piece releases but its logs has its declaration, and the logs have none: they move to the serve
    dir's private/ before anything leaves the node (`served_verdicts.sh`'s CERTIFIER_PRIVATE)."""
    assert {*CP.PIECE_FILES, CP.CASE, CP.CHAINED, CP.REFUSED} - set(CP.LOGS) <= set(KS.DOCS) | set(KS.TEXT)
    assert not set(CP.LOGS) & (set(KS.DOCS) | set(KS.TEXT))
    assert f"profile.placement.{CP.PREMISE}" in KS.ANSWER


#: a served run's registered compiler as `prove_path.py compiler` prints it (`prove_path.compiler_doc`)
COMPILER = {"name": "verity-vllm/capped/v1", "digest": "0a" * 64, "cap_bits": L_A,
            "inner": {"name": "verity_vllm.program.predict.steps.compile_steps", "digest": "0b" * 64},
            "served": {"name": "verity_vllm.program.predict.steps.compile_steps", "digest": "0b" * 64},
            "transfer_gates": ["kv"], "default_seed": 0,
            "phi": {"schema": "phi/v1", "model": {"repo": "unsloth/Llama-3.2-1B", "revision": "95" * 20, "hf_config": None},
                    "target": {"gpu": "H200"}, "tp": 1, "sampling": {"kind": "greedy"},
                    "engine": {"execution": "bi-eager", "pin": {"world": 1}}, "workload": {}, "execution": "bi-eager",
                    "profile": "b1-eager-v3"}}
#: a window's advice registration as `verity-vllm advice` writes it, with its requests and the window's number, its advice
#: committed padded to ℓ_a (the rows ℓ_a fixes)
ADVICE = {"schema": "verity-vllm/advice-registration/v1", "scheme": "verity-vllm/advice/v1",
          "binding": {"compiler": {"name": COMPILER["name"], "digest": COMPILER["digest"]},
                      "advice": {"root": "06" * 64, "bits": L_A}},
          "compiler": {"name": COMPILER["name"], "digest": COMPILER["digest"], "cap_bits": L_A, "inner": COMPILER["inner"]},
          "rows": L_A // 4096, "row_bits": 4096, "T": 600, "window": 3,
          "commit": {"run_id": "RUN/w3", "run_root": "ab" * 64}, "registered_roots": {"weights/embed@4": "02" * 64},
          "requests": 2, "compiled": [{"request": "r0"}], "refused": [{"request": "r1", "why": "a chunked prefill"}]}
#: the committer's weights registration (`window_weights.weights_registration`)
WEIGHTS = {"schema": "verity-vllm/weights-registration/v1", "program": "window-circuit",
           "model": {"repo": "unsloth/Llama-3.2-1B", "revision": "95" * 20}, "grain": {"rows": 9, "gemm_words": 512},
           "values": {"weights/embed@4": {"value": "weights/embed@4", "root": "02" * 64, "leaves": 4, "words": 40,
                                          "domain": "ef" * 64}},
           "unrooted": {"lm_head": "tied to the embedding"}}


def _out(tmp_path: Path, period) -> Path:
    """A served run's OUT as the verifier receives it (`RECEIVED`): the certifier's piece (with the served.json a
    deferral's follow-on leaves, naming the served node it audited), V3c's audit as the run
    copies it (its summary and audit record as `served_integrity.public_copy` writes them, the registration it audited
    and its period audit), the registered compiler, the window's advice registration in its public form
    (`served_verdicts.public_registration`), the weights registration and a class's count."""
    out = tmp_path / "out"
    case, answer = _case(period)
    piece = _dir(out / "pieces" / "certifier", case, answer)
    (piece / "piece.json").write_text(json.dumps({"server": "node", "rc": 0, "limit": "CERTIFIER_TIMEOUT"}))
    (piece / "served.json").write_text(json.dumps({"server": "served-node"}))
    com, ev = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    reg, units = com["audits"][0], [0, 1]
    v3c = out / "v3c-audit"
    (v3c / "window").mkdir(parents=True)
    (v3c / "summary.json").write_text(json.dumps({"checks_before_draw": {"registration": True},
                                                  "fixed_before_draw": {"ok": True, "codes": []},
                                                  "audit": {"accepted": True, "why": {"lean-verify/m0": None}}}))
    (v3c / "window" / "audit.json").write_text(json.dumps({
        "accepted": True, "registration": reg, "proved_units": units,
        "checks": {"draw": {"ok": True, "codes": []},
                   "proved": {"ok": True, "codes": [], "detail": {"units": {"drawn": units, "statement": units}}}},
        "draw": {"law": "subset", "k": 2, "population": reg["population"], "units": units},
        "wrong_units": {"delta": 1e-6, "of": reg["population"], "bound": 9, "rule": "at most `bound` incorrect proof blocks"}}))
    (v3c / "registration.json").write_text(json.dumps(reg))
    (v3c / "period.json").write_text(json.dumps({"answer": ev["audits"][0], "registration": reg, "free": 3,
                                                 **{k: case["params"]["audits"][0][k] for k in ("steps", "program")}}))
    (out / "compiler.json").write_text(json.dumps(COMPILER))
    (out / "registration.json").write_text(json.dumps(SV.public_registration(ADVICE)))
    (out / "weights_registration.json").write_text(json.dumps(WEIGHTS))
    (out / "window-circuit-4-32-4.json").write_text(json.dumps({
        "count": {"free": 3, "inputs": 10, "registered": 7, "units": 1}, "seconds": {"circuit": 1.5, "record": 0.2}}))
    return out


def test_what_a_verifier_receives_of_a_served_run_is_of_the_public_kinds(tmp_path, period):
    """OUT as the run writes it scans clean, every file the scan names read (the leaves counted by kind), and every
    file of OUT is one the scan reads."""
    out = _out(tmp_path, period)
    found, kinds = KS.received(out)
    assert found == [] and all(kinds[k] for k in KS.KINDS), (found, kinds)
    assert KS.main(["--received", str(out)]) == 0
    received = {f"{d}/*" for d in KS.RECEIVED_DIRS} | set(KS.RECEIVED)
    published = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert all(any(fnmatch.fnmatchcase(p, g) for g in received) for p in published), published


def _edit(p: Path, f) -> None:
    p.write_text(json.dumps(f(json.loads(p.read_text()))))


@pytest.mark.parametrize("plant, want", [
    (lambda o: (o / "pieces" / "certifier" / "chain.log").write_text("chain on window 3\n"),
     [("pieces/certifier/chain.log", "")]),
    (lambda o: (o / "v3c-audit" / "audit.log").write_text('RECORDS {"population": 37}\n'), [("v3c-audit/audit.log", "")]),
    (lambda o: (o / "v3c-audit" / "main.txt").write_text("ab" * 20 + "\n"), [("v3c-audit/main.txt", "")]),
    (lambda o: _edit(o / "registration.json", lambda r: {**r, "binding": {**r["binding"], "advice": {
        **r["binding"]["advice"], "bits": REALIZED}}, "rows": 1, "bits_by_window": {"3": REALIZED}}),
     [("registration.json", p) for p in ("binding.advice.bits", "bits_by_window.3", "rows")]),
    (lambda o: (o / "registration.json").write_text(json.dumps(ADVICE)),
     [("registration.json", p) for p in ("compiled[].request", "refused[].request", "refused[].why", "requests",
                                         "window")]),
    (lambda o: _edit(o / "compiler.json", lambda c: {**c, "phi": {**c["phi"], "workload": {
        "requests": [{"request_id": "r0", "prompt": 11, "max_tokens": 8}]}}}),
     [("compiler.json", f"phi.workload.requests[].{k}") for k in ("max_tokens", "prompt", "request_id")]),
    (lambda o: _edit(o / "window-circuit-4-32-4.json", lambda c: {**c, "bits": REALIZED}),
     [("window-circuit-4-32-4.json", "bits")]),
    (lambda o: _edit(o / "v3c-audit" / "summary.json", lambda s: {**s, "compile": {"ok": True, "bits": REALIZED}}),
     [("v3c-audit/summary.json", "compile")]),
    (lambda o: _edit(o / "v3c-audit" / "registration.json", lambda r: {**r, "window": {
        **r["window"], "advice": {"root": "06" * 64, "bits": L_A}}}),
     [("v3c-audit/registration.json", "window.advice")]),
    (lambda o: _edit(o / "v3c-audit" / "window" / "audit.json", lambda r: {**r, "statement": {"requests": 2}}),
     [("v3c-audit/window/audit.json", "statement.requests")]),
])
def test_what_reads_the_advice_or_counts_the_window_is_a_finding_in_what_a_verifier_receives(
        tmp_path, period, plant, want):
    """A log or main's history left in the piece's dir or V3c's audit; the advice's own length in its registration,
    or the registration as `verity-vllm advice` writes it (its requests, compiles and window); a workload in the
    compiler's φ; the advice's length in a class's count; a compile in V3c's summary (the audit opened the advice); an
    advice binding in the registration V3c audited; and a field of the audit record `served_integrity.answer` doesn't
    read: each is a finding, and the scan exits 1."""
    out = _out(tmp_path, period)
    plant(out)
    found = KS.received(out)[0]
    assert sorted(_paths(found)) == sorted(want), found
    assert KS.main(["--received", str(out)]) == 1
    assert all(str(REALIZED) not in str(f) and "r0" not in str(f) for f in found)


def test_a_received_file_that_is_a_symlink_or_isnt_json_is_a_finding(tmp_path, period):
    out = _out(tmp_path, period)
    (out / "weights_registration.json").unlink()
    (out / "weights_registration.json").symlink_to(out / "compiler.json")
    (out / "window-circuit-4-32-4.json").write_text("registered 7 bits")
    assert sorted(_paths(KS.received(out)[0])) == [("weights_registration.json", ""), ("window-circuit-4-32-4.json", "")]
    with pytest.raises(SystemExit):
        KS.main([])
