"""kind_scan: what the network certifier's piece releases holds only fields of the four public kinds and the network
certifier's certificates. A rows-form period (`period_case`'s accept, its registrations' provenance the scan's and V3c's
committed window's, as `scan_zk` and `prove_path.py window-commit` write them) and its answer with the piece's placement
scan clean as DIR/case.json and as the hold's three period files. A field no declaration names is a finding wherever it
lies: the window's own number in V3c's provenance, a count of requests in an answer, an unpadded advice length beside
the parameters, and a file DIR doesn't declare; the logs aren't read. An advice binding's length is structure only at
the registered width ℓ_a: the realized length, or any with no ℓ_a given, is a finding whose text never holds the
length. On a finding the piece refuses, releasing nothing, its findings (file and key path) in its hold."""
from __future__ import annotations

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
from verity.ml.circuits.codec import canonical_json  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
#: a scan round's registration provenance (`scan_zk`'s)
SCAN_WINDOW = {"kind": "stand-in", "link": "registered: each row port at the network certifier's signed root",
               "of": "round honest: link-windows [0] of the network certifier's run",
               "members": {"first": {"template": "IngressScanFirst_v3", "units": [0]},
                           "egress-0": {"template": "EgressBucket_v1", "units": [1, 2]}}}
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
    (d / "piece.log").write_text(f"advice {REALIZED} bits, window 3\n")
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
    """The window's own number in V3c's provenance (which counts the run's windows), a count of requests in the
    period's answer and in a link-window's one-stage answer, and an advice length beside the parameters."""
    case, answer = _case(period, V3C_WINDOW | {"window_circuit": {"class": [4, 32, 4, 4, 16], "window": 3}})
    ev = json.loads(bytes.fromhex(case["evidence"]))
    ev["windows"][0]["answer"]["public"]["requests"] = 7
    case = {**case, "evidence": canonical_json(ev).hex(), "params": {**case["params"], "advice_bits": REALIZED}}
    answer = answer | {"public": answer["public"] | {"requests": 7}}
    want = [("answer.json", "public.requests"), ("case.json", "commitment.audits[].window.window_circuit.window"),
            ("case.json", "evidence.windows[].answer.public.requests"), ("case.json", "params.advice_bits")]
    assert sorted(_paths(KS.scan(_dir(tmp_path / "dir", case, answer))[0])) == want
    held = KS.scan(_dir(tmp_path / "hold", case, answer, hold=True))[0]
    assert sorted(_paths(held)) == [("answer.json", "public.requests"),
                                    ("period/commitment.bin", "audits[].window.window_circuit.window"),
                                    ("period/evidence.json", "windows[].answer.public.requests"),
                                    ("period/params.json", "advice_bits")]
    assert KS.main([str(tmp_path / "dir")]) == 1


def test_an_advice_length_is_structure_only_at_its_registered_width(tmp_path, period):
    """An advice binding (`interface.advice_binding`) in V3c's provenance: its length at ℓ_a is structure, given ℓ_a;
    the realized length, or any length with no ℓ_a given, is a finding naming neither."""
    def found(bits: int, widths: dict | None, name: str) -> list[KS.Finding]:
        v3c = V3C_WINDOW | {"compiler": {"name": "i2", "digest": "04" * 64, "commit": "05" * 20},
                            "advice": {"root": "06" * 64, "bits": bits}}
        return KS.scan(_dir(tmp_path / name, *_case(period, v3c)), widths)[0]
    assert found(L_A, {KS.WIDTH.name: L_A}, "padded") == []
    for bits, widths, name in ((REALIZED, {KS.WIDTH.name: L_A}, "realized"), (L_A, None, "unstated")):
        (f,) = found(bits, widths, name)
        assert (f.file, f.path) == ("case.json", "commitment.audits[].window.advice.bits")
        assert f.why.startswith(KS.WIDTH_WHY) and str(REALIZED) not in str(f) and str(L_A) not in str(f)
    assert KS.main([str(tmp_path / "realized"), "--advice-bits", str(L_A)]) == 1


def test_a_file_dir_doesnt_declare_is_a_finding_and_the_logs_arent_read(tmp_path, period):
    d = _dir(tmp_path / "d", *_case(period))
    (d / "chain" / "report").mkdir(parents=True)
    (d / "chain" / "report" / "report.md").write_text("Requests: 7\n")
    (d / CP.CHAINED).write_text("not json")
    assert sorted(_paths(KS.scan(d)[0])) == [(CP.CHAINED, ""), ("chain/report/report.md", "")]


def test_dirs_files_are_the_pieces():
    """Every file the piece releases has its declaration, and its logs are the ones the scan passes over."""
    assert {*CP.PIECE_FILES, CP.CASE, CP.CHAINED} - set(CP.LOGS) <= set(KS.DOCS) and KS.LOGS == CP.LOGS
    assert f"profile.placement.{CP.PREMISE}" in KS.ANSWER


def test_the_piece_refuses_on_a_finding_releasing_nothing(tmp_path):
    """`certifier_piece.settle` scans the hold before the leak guard: a field of no public kind in the answer refuses
    the piece with nothing in DIR, the refusal naming the file but not the field, the findings in the hold."""
    serve, d = tmp_path / "serve", tmp_path / "dir"
    (serve / CP.PRIVATE / "run").mkdir(parents=True)
    (serve / CP.PRIVATE / "run" / "fetch.tar.gz").write_bytes(b"")
    d.mkdir()
    planted = CP.answer("reject", "stub") | {"requests": 7}
    with pytest.raises(CP.NotRun, match=r"the kind scan found fields of no public kind \(1\) in \['answer.json'\]") as e:
        CP.settle(serve, d, "CERTIFIER_TIMEOUT", lambda hold: (planted, None))
    assert "requests" not in str(e.value) and list(d.iterdir()) == []
    (hold,) = (serve / CP.PRIVATE / "held").iterdir()
    assert (hold / CP.KIND_SCAN).read_text() == "answer.json requests: a int of no declared kind\n"
