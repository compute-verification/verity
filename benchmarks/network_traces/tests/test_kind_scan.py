"""kind_scan: what the network certifier's piece releases holds only fields of the four public kinds and the network
certifier's certificates. A rows-form period (`period_case`'s accept, its registrations' provenance the scan's and V3c's
committed window's, as `scan_zk` and `prove_path.py window-commit` write them) and its answer with the piece's placement
scan clean as DIR/case.json and as the hold's three period files. A field no declaration names is a finding wherever it
lies: the window's own number in V3c's provenance, a count of requests in an answer, an advice length beside the
parameters, a free count in an audit's, and a file DIR doesn't declare, a log among them. Each Program DIR names is the
verifier's: the parameters' program, the scan's descriptor and each scan registration's the scan Program pinned at the
schedule (`scan_service.SCAN_PINS`), and each audit's descriptor and registration the class circuit, so another (a
Program, the scan's annotations, which no digest covers, or a schedule no class pins) is a finding in either layout,
naming no value; DIR without the period's files (the split's chain step) holds the scan's answers to some class's
Program.
A key a table declares by no name comes from the source it names: an answer's bounds per template from the templates its
profile draws by, a report step's bounds from the δs the chain bounds at (`chain_record.eps_at_delta`'s), a scan round's
members from its link-windows and the schedule's lanes (`scan_zk.members_of`'s), the report's negatives from those the
chain counts (`chain_record.report`'s: the rehearsal's, N2's and the scan's). A key from anywhere else is a finding
naming the source, never the key, and another key of a fixed vocabulary (the scan state's port and runs, the profile's
level, the placement's parties) a field of no declared kind; a bare ``*`` is left only where `OPEN` lists it.
What either development check would feed the record is a finding naming the check: an advice binding in an audit's registration (what `records_audit.py audit --opening`
opens), at any length, the padded width included, and the chain's endpoint step (CR_ENDPOINT's rebuild), or any step
the active-warden path doesn't run. On a finding the piece refuses, releasing nothing, its findings (file and key path)
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
import certifier_serve as CS  # noqa: E402
import chain_record as CR  # noqa: E402
import kind_scan as KS  # noqa: E402
import n2_zk as N2  # noqa: E402
import scan_zk as SZ  # noqa: E402
from verity.core.protocols.certifier.ingress_scan import SCAN_STATE  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402

sys.path.append(str(ROOT / "benchmarks" / "one_stage"))
import served_verdicts as SV  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
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
              "activations": {"program": "window-circuit", "values": {SCAN_STATE: {
                  "value": SCAN_STATE, "root": "cd" * 64, "leaves": 3, "words": 40, "domain": "ef" * 64}}},
              "circuit": "01" * 64, "registered_roots": {"weights/embed@4": "02" * 64}, "index_leaf": "03" * 64,
              "order": "verity-vllm/serving-rows/order/v0"}
PLACEMENT = {"kind": "not-placed", "parties": {"network_certifier": "gate", "prover": "node", "verifier": "node"},
             CP.PREMISE: "assumed, the auditor's devices not placed", "uncounted": "every link of node's but one",
             "links": [{"name": "certified", "from": "gate", "to": "node", "counted": True, "via": "the tunnel"}]}
#: the advice's registered width ℓ_a, and a realized length beside it
L_A, REALIZED = 8388608, 392


@pytest.fixture(scope="module")
def period():
    """`period_case`, with the rows-form accept's stand-in scan pinned at its schedule (`SCAN_PINS`), as a class's
    scan is."""
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    p = mod.params(mod.case("accept", 1, "rows")["params"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(mod.SCAN_PINS, p.schedule.digest.hex(), p.program)
        yield mod


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
    the period's answer and in an execution phase's one-stage answer, an advice length beside the parameters, and a
    free count in an audit's (which the period counts off its Program)."""
    case, answer = _case(period, V3C_WINDOW | {"window_circuit": {"class": [4, 32, 4, 4, 16], "window": 3}})
    n = len(case["params"]["audits"])
    ev = json.loads(bytes.fromhex(case["evidence"]))
    ev["windows"][0]["answer"]["public"]["requests"] = 7
    params = json.loads(canonical_json(case["params"])) | {"advice_bits": REALIZED}
    params["audits"][0]["free"] = REALIZED
    case = {**case, "evidence": canonical_json(ev).hex(), "params": params}
    answer = answer | {"public": answer["public"] | {"requests": 7}}
    want = [("answer.json", "public.requests"),
            *[("case.json", "commitment.audits[].window.window_circuit.window")] * n,
            ("case.json", "evidence.windows[].answer.public.requests"), ("case.json", "params.advice_bits"),
            ("case.json", "params.audits[].free")]
    assert sorted(_paths(KS.scan(_dir(tmp_path / "dir", case, answer))[0])) == want
    held = KS.scan(_dir(tmp_path / "hold", case, answer, hold=True))[0]
    assert sorted(_paths(held)) == [("answer.json", "public.requests"),
                                    *[("period/commitment.bin", "audits[].window.window_circuit.window")] * n,
                                    ("period/evidence.json", "windows[].answer.public.requests"),
                                    ("period/params.json", "advice_bits"), ("period/params.json", "audits[].free")]
    assert KS.main([str(tmp_path / "dir")]) == 1


OTHER = "ee" * 64
#: a Program the verifier doesn't fix, planted in the rows-form accept's parameters and commitment: the findings it
#: makes, each the period's part and key path (by the period's count of audits, where the plant reaches each audit),
#: and why
UNPINNED = {
    "the parameters' program": (lambda p, c: p.update(program=OTHER), [("params", "program")], KS.NOT_PINNED),
    "another scan": (lambda p, c: p["scan"].update(root="other"), [("params", "scan")], KS.NOT_PINNED),
    "the scan's annotations": (lambda p, c: p["scan"].update(annotations={"note": "x"}),
                               [("params", "scan.annotations")], KS.ANNOTATED),
    "a window's registration": (lambda p, c: c["windows"][0].update(program=OTHER),
                                [("commitment", "windows[].program")], KS.NOT_PINNED),
    "another audit": (lambda p, c: p["audits"][0]["program"].update(root="other"),
                      [("params", "audits[].program")], KS.NOT_CLASS),
    "an audit's registration": (lambda p, c: c["audits"][0].update(program=OTHER),
                                [("commitment", "audits[].program")], KS.NOT_CLASS),
    "another class circuit": (lambda p, c: p.update(circuit=OTHER),
                              lambda n: [("params", "audits[].program"), ("commitment", "audits[].program")] * n,
                              KS.NOT_CLASS),
}


def _where(part: str, path: str, hold: bool) -> tuple[str, str]:
    """A period's ``part`` and key path as a finding names them in DIR/case.json, or in the hold's period files."""
    return ({"params": "period/params.json", "commitment": "period/commitment.bin"}[part], path) if hold else \
        ("case.json", f"{part}.{path}")


@pytest.mark.parametrize("name", list(UNPINNED))
def test_a_program_the_verifier_doesnt_fix_is_a_finding_in_either_layout(tmp_path, period, name):
    plant, want, why = UNPINNED[name]
    case, answer = _case(period)
    pin, circuit = case["params"]["program"], case["params"]["circuit"]
    params, com = json.loads(canonical_json(case["params"])), json.loads(bytes.fromhex(case["commitment"]))
    plant(params, com)
    want = want(len(params["audits"])) if callable(want) else want
    case = {**case, "params": params, "commitment": canonical_json(com).hex()}
    for hold in (False, True):
        found = KS.scan(_dir(tmp_path / str(hold), case, answer, hold))[0]
        assert sorted((f.file, f.path, f.why) for f in found) == sorted((*_where(*w, hold), why) for w in want)
        assert not any(v in str(f) for f in found for v in (OTHER, pin, circuit))
    assert KS.main([str(tmp_path / "False")]) == 1


def test_a_schedule_no_scan_class_pins_is_a_finding(tmp_path, period, monkeypatch):
    case, answer = _case(period)
    monkeypatch.delitem(KS.SCAN_PINS, period.params(case["params"]).schedule.digest.hex())
    for hold in (False, True):
        found = KS.scan(_dir(tmp_path / str(hold), case, answer, hold))[0]
        assert [(f.file, f.path, f.why) for f in found] == [(*_where("params", "schedule", hold), KS.NO_PIN)]


def test_the_scans_answers_name_the_scan_program_pinned_at_the_schedule_or_without_it_some_classs(tmp_path, period):
    """The scan's answers (``chain/scan/answers.json``) beside the period name the scan Program pinned at its schedule;
    in a DIR without the period's files (the split's chain step), any class's. Another is a finding."""
    case, answer = _case(period)
    pin = case["params"]["program"]
    other_class = next(v for v in KS.SCAN_PINS.values() if v != pin)
    for split, held in ((False, [pin]), (True, [pin, other_class])):
        for i, program in enumerate((pin, other_class, OTHER)):
            d = tmp_path / f"{split}-{i}"
            if split:
                d.mkdir()
                (d / "answer.json").write_text(json.dumps(answer))
            else:
                _dir(d, case, answer)
            (d / "chain" / "scan").mkdir(parents=True)
            (d / "chain" / "scan" / "answers.json").write_text(json.dumps(
                {"format": "answers", "delta": "1/100", "registration": {"program": program}, "windows": []}))
            want = [] if program in held else [("chain/scan/answers.json", "registration.program")]
            assert _paths(KS.scan(d)[0]) == want, (split, i)


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
    """The report's steps are the active-warden path's (`STEPS`, the five `chain_record.report` writes there, which
    test_certifier_piece holds a published report to): with them it is clean; the endpoint step (CR_ENDPOINT's rebuild,
    and its tie negatives beside it) is `ENDPOINT`'s finding, and a step of no other name is one of no declared kind."""
    def found(steps: dict, name: str) -> list[tuple[str, str, str]]:
        d = _dir(tmp_path / name, *_case(period))
        (d / "chain" / "report").mkdir(parents=True)
        (d / "chain" / "report" / "report.json").write_text(json.dumps({
            "schema": "s", "verdict": "accepted", "path": "warden", "scope": None, "stubbed": ["x"],
            "negatives_refused": {"n2/tampered": True, "scan/chain": True}, "steps": steps}))
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


#: the bare ``*`` keys no source names yet, by the files' declarations, each its key path's tail to the ``*``: in DIR a
#: committed window's registered roots and the parameters' pins; in V3c's audit and the received files, their owners'.
#: It only shrinks.
OPEN = {
    "DOCS": {"window.registered_roots.*", "audits[].pins.*"},
    "V3C_DOCS": {"checks_before_draw.*", "checks.*", "wrong_units.per_template.*", "window.registered_roots.*"},
    "RECEIVED": {"registered_roots.*", "values.*", "unrooted.*", "count.*", "seconds.*"},
}


def _keys(docs: str) -> list[list[str]]:
    return [p.split(".") for table in getattr(KS, docs).values() for p in table]


@pytest.mark.parametrize("docs", list(OPEN))
def test_a_bare_star_is_left_only_where_open_lists_it(docs):
    """A key no table names is a ``<name>`` that names its source, or a bare ``*`` `OPEN` lists, each entry of which
    is still one."""
    stars = {".".join(ps[:i + 1]) for ps in _keys(docs) for i, s in enumerate(ps) if s == "*"}
    tails = {s: {t for t in OPEN[docs] if s == t or s.endswith(f".{t}")} for s in stars}
    assert sorted(s for s, ts in tails.items() if not ts) == []
    assert set().union(*tails.values()) == OPEN[docs]


def test_each_source_is_one_a_table_names():
    """Each source is one a table names, and each fixed one is what its producer writes: the δs
    `chain_record.eps_at_delta` keys, and the negatives `chain_record.report` counts, a rehearsal's tampers but those
    its own check accepts (``serve --tamper`` takes `certifier_serve.TAMPERS`), N2's members but honest and the scan's
    rounds but those whose window is accepted."""
    assert {s[1:-1] for d in OPEN for ps in _keys(d) for s in ps if s[:1] == "<"} == set(KS.SOURCES)
    assert set(CR.eps_at_delta(4, 2, "subset:2")["incorrect_at_delta"]) == KS.DELTA_KEYS
    counted = {*(f"rehearsal/{s}" for s in {"honest", *CS.TAMPERS} - {*CR.REHEARSAL_ACCEPTS}),
               *(f"n2/{m}" for m in {*N2.ROUNDS, N2.OTHER} - {"honest"}),
               *(f"scan/{r}" for r in {*SZ.ROUNDS} - {*CR.SCAN_ACCEPTS})}
    assert KS.NEGATIVES == counted, "set kind_scan.NEGATIVES to the negatives chain_record.report counts"


SECRET = "a key that carries a witness"


def test_a_key_from_anywhere_but_its_source_is_a_finding_naming_the_source(tmp_path, period):
    """An execution phase's bounds per template, keyed by the templates its profile draws by, a report step's bounds at
    the δs the chain bounds at (`chain_record.eps_at_delta`'s) and the report's negatives, each one the chain counts,
    scan clean; another key of any is a finding at its ``<name>``, naming the source and never the key."""
    case, answer = _case(period)
    ev = json.loads(bytes.fromhex(case["evidence"]))
    a = ev["windows"][0]["answer"]
    a["profile"]["levels"][0]["draw"] = {"law": "stratified", "strata": [{"name": "t0", "size": 2, "k": 1}]}
    eps = CR.eps_at_delta(4, 2, "subset:2")

    def scanned(name: str, per_template: dict, at_delta: dict, negatives: dict) -> Path:
        a["public"]["per_template"] = per_template
        d = _dir(tmp_path / name, {**case, "evidence": canonical_json(ev).hex()}, answer)
        steps = {s: {"verdict": "accepted", "in_the_clear": "the rows", "eps_at_delta": eps} for s in KS.STEPS}
        steps["n2"]["eps_at_delta"] = eps | {"incorrect_at_delta": at_delta}
        (d / "chain" / "report").mkdir(parents=True)
        (d / "chain" / "report" / "report.json").write_text(json.dumps({
            "schema": "s", "verdict": "accepted", "path": "warden", "scope": None, "stubbed": [],
            "negatives_refused": negatives, "steps": steps}))
        return d

    every = dict.fromkeys(KS.NEGATIVES, True) | {"n2/other_subset": "escaped the draw", "rehearsal/late": False}
    assert KS.scan(scanned("honest", {"t0": 1}, eps["incorrect_at_delta"], every))[0] == []
    d = scanned("planted", {"t0": 1, SECRET: 0}, eps["incorrect_at_delta"] | {"0.25": 1},
                {"n2/tampered": True, SECRET: True})
    found = KS.scan(d)[0]
    assert sorted((f.file, f.path, f.why) for f in found) == [
        ("case.json", "evidence.windows[].answer.public.per_template.<template>", KS.SOURCES["template"][1]),
        ("chain/report/report.json", "negatives_refused.<negative>", KS.SOURCES["negative"][1]),
        ("chain/report/report.json", "steps.n2.eps_at_delta.incorrect_at_delta.<delta>", KS.SOURCES["delta"][1])]
    assert not any(SECRET in str(f) or "0.25" in str(f) for f in found)
    assert KS.main([str(d)]) == 1


def test_a_scan_rounds_members_are_of_its_link_windows_and_the_schedules_lanes(tmp_path, period):
    """A round's members (`scan_zk.members_of`) scan clean in either layout: the honest round's ``first``, ``last`` and
    each of its link-windows' ingress and egress (the rows form's one link-window, the scan's answers' group), and the
    block round's one a lane of the period's schedule (``block-c<j>``). A member of another link-window or lane, a
    lane's beside no period (the scan's answers), and any other key is a finding at ``<member>`` naming the source,
    never the key."""
    case, answer = _case(period)
    lanes, unit = case["params"]["schedule"]["lanes"], {"template": "t", "units": [0]}
    honest = dict.fromkeys(("first", "ingress-0", "last", "egress-0"), unit)
    block = {f"block-c{j}": unit for j in range(lanes)}
    planted = {"ingress-1": unit, SECRET: unit}, {f"block-c{lanes}": unit}

    def members(windows: dict, audits: dict) -> dict:
        com = json.loads(bytes.fromhex(case["commitment"]))
        com["windows"][0]["window"] = SCAN_WINDOW | {"members": windows}
        com["audits"][0]["window"] = SCAN_WINDOW | {"members": audits}
        return {**case, "commitment": canonical_json(com).hex()}

    for hold in (False, True):
        assert KS.scan(_dir(tmp_path / f"honest-{hold}", members(honest, block), answer, hold))[0] == []
        d = _dir(tmp_path / f"planted-{hold}", members(honest | planted[0], block | planted[1]), answer, hold)
        found = KS.scan(d)[0]
        assert sorted((f.file, f.path, f.why) for f in found) == sorted(
            (*_where("commitment", f"{part}[].window.members.<member>", hold), KS.SOURCES["member"][1])
            for part in ("windows", "windows", "audits"))
        assert not any(k in str(f) for f in found for k in (SECRET, "ingress-1", f"block-c{lanes}"))
    d = _dir(tmp_path / "answers", case, answer)
    (d / "chain" / "scan").mkdir(parents=True)
    group = dict.fromkeys(("first", "ingress-0", "ingress-1", "last", "egress-0", "egress-1"), unit)
    (d / "chain" / "scan" / "answers.json").write_text(json.dumps({
        "format": "answers", "delta": "1/100", "windows": [{"window": 0}, {"window": 1}],
        "registration": {"program": case["params"]["program"], "window": SCAN_WINDOW | {"members": group}}}))
    assert KS.scan(d)[0] == []
    (d / "chain" / "scan" / "answers.json").write_text(json.dumps({
        "format": "answers", "delta": "1/100", "windows": [{"window": 0}, {"window": 1}],
        "registration": {"program": case["params"]["program"], "window": SCAN_WINDOW | {
            "members": group | {"egress-2": unit, "block-c0": unit}}}}))
    assert [(f.file, f.path, f.why) for f in KS.scan(d)[0]] == [
        ("chain/scan/answers.json", "registration.window.members.<member>", KS.SOURCES["member"][1])] * 2
    assert KS.main([str(d)]) == 1


def test_another_key_of_a_fixed_vocabulary_is_a_field_of_no_declared_kind(tmp_path, period):
    """The scan's state read under a run but ``sin`` and ``o1`` or opened under a port but ``scan/state`` (the one
    `stateRead` and `stateEmpty` read), a per-window value a committed window commits but the scan's state
    (`ingress_scan.SCAN_STATE`, the one `prove_path.py window-commit` commits), a profile's count drawn at a level but
    ``unit``, and a party the placement doesn't name."""
    case, answer = _case(period)
    com, ev = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    w = com["windows"][0]["window"]
    w["reads"]["scan/state"]["o2"] = w["reads"]["scan/state"]["o1"]
    w["empty"] = {"scan/view": w["empty"]["scan/state"]}
    values = com["audits"][0]["window"]["activations"]["values"]
    values["weights/embed@4"] = values[SCAN_STATE] | {"value": "weights/embed@4"}
    ev["windows"][0]["answer"]["profile"]["drawn"]["token"] = 1
    case = {**case, "commitment": canonical_json(com).hex(), "evidence": canonical_json(ev).hex()}
    answer = answer | {"profile": answer["profile"] | {"placement": PLACEMENT | {
        "parties": PLACEMENT["parties"] | {"auditor": "gate"}}}}
    pre = "case.json", "commitment.windows[].window"
    assert set(_paths(KS.scan(_dir(tmp_path / "d", case, answer))[0])) == {
        ("answer.json", "profile.placement.parties.auditor"),
        *((pre[0], f"{pre[1]}.reads.scan/state.o2.{k}") for k in ("leaf", "units[]")),
        *((pre[0], f"{pre[1]}.empty.scan/view.{k}") for k in ("words", "salt", "path")),
        *(("case.json", f"commitment.audits[].window.activations.values.weights/embed@4.{k}")
          for k in values[SCAN_STATE]),
        ("case.json", "evidence.windows[].answer.profile.drawn.token")}


#: a served run's registered compiler as `prove_path.py compiler` prints it (`prove_path.compiler_doc`)
COMPILER = {"name": "verity-vllm/capped/v1", "digest": "0a" * 64, "cap_bits": L_A,
            "inner": {"name": "verity_vllm.program.predict.steps.compile_steps", "digest": "0b" * 64},
            "served": {"name": "verity_vllm.program.predict.steps.compile_steps", "digest": "0b" * 64},
            "transfer_gates": ["kv"], "default_seed": 0,
            "phi": {"schema": "phi/v1", "model": {"repo": "unsloth/Llama-3.2-1B", "revision": "95" * 20, "hf_config": None},
                    "target": {"gpu": "H200"}, "tp": 1, "sampling": {"kind": "greedy"},
                    "engine": {"execution": "bi-eager", "pin": {"world": 1}}, "workload": {}, "execution": "bi-eager",
                    "profile": "b1-eager-v3"}}
#: a window's advice registration as `verity-vllm advice` writes it, with its requests, the advice's own length and the
#: window's number
ADVICE = {"schema": "verity-vllm/advice-registration/v1", "scheme": "verity-vllm/advice/v1",
          "binding": {"compiler": {"name": COMPILER["name"], "digest": COMPILER["digest"]},
                      "advice": {"root": "06" * 64, "bits": REALIZED}},
          "compiler": {"name": COMPILER["name"], "digest": COMPILER["digest"], "cap_bits": L_A, "inner": COMPILER["inner"]},
          "rows": 1, "row_bits": 4096, "T": 600, "bits_by_window": {"3": REALIZED}, "window": 3,
          "commit": {"run_id": "RUN/w3", "run_root": "ab" * 64}, "registered_roots": {"weights/embed@4": "02" * 64},
          "requests": 2, "compiled": [{"request": "r0"}], "refused": [{"request": "r1", "why": "a chunked prefill"}]}
#: the committer's weights registration (`window_weights.weights_registration`)
WEIGHTS = {"schema": "verity-vllm/weights-registration/v1", "program": "window-circuit",
           "model": {"repo": "unsloth/Llama-3.2-1B", "revision": "95" * 20}, "grain": {"rows": 9, "gemm_words": 512},
           "values": {"weights/embed@4": {"value": "weights/embed@4", "root": "02" * 64, "leaves": 4, "words": 40,
                                          "domain": "ef" * 64}},
           "unrooted": {"lm_head": "tied to the embedding"}}


def _out(tmp_path: Path, period) -> Path:
    """A served run's OUT as the verifier receives it (`RECEIVED`): the certifier's piece, V3c's audit as the run
    copies it (its summary and audit record as `served_integrity.public_copy` writes them, the registration it audited
    and its period audit), the registered compiler, the window's advice registration in its public form
    (`served_verdicts.public_registration`), the weights registration and a class's count."""
    out = tmp_path / "out"
    case, answer = _case(period)
    piece = _dir(out / "pieces" / "certifier", case, answer)
    (piece / "piece.json").write_text(json.dumps({"server": "node", "rc": 0, "limit": "CERTIFIER_TIMEOUT"}))
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
    (v3c / "period.json").write_text(json.dumps({"answer": ev["audits"][0], "registration": reg,
                                                 "free": period.params(case["params"]).audits[0].free,
                                                 **{k: case["params"]["audits"][0][k] for k in ("steps", "program")}}))
    (out / "compiler.json").write_text(json.dumps(COMPILER))
    (out / "registration.json").write_text(json.dumps(SV.public_registration(ADVICE)))
    (out / "weights_registration.json").write_text(json.dumps(WEIGHTS))
    (out / "window-circuit-4-32-4.json").write_text(json.dumps({
        "count": {"free": 3, "inputs": 10, "registered": 7, "units": 1}, "seconds": {"circuit": 1.5, "record": 0.2}}))
    return out


def test_what_a_verifier_receives_of_a_served_run_is_of_the_public_kinds(tmp_path, period):
    """OUT as the run writes it scans clean, every file `served_verdicts.RECEIVED` names read (the leaves counted by
    kind), and the paths the scan reads are the ones publish leaves as they are."""
    out = _out(tmp_path, period)
    found, kinds = KS.received(out)
    assert found == [] and all(kinds[k] for k in KS.KINDS), (found, kinds)
    assert KS.main(["--received", str(out)]) == 0
    assert {f"{d}/*" for d in KS.RECEIVED_DIRS} | set(KS.RECEIVED) == set(SV.RECEIVED)
    published = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert all(any(fnmatch.fnmatchcase(p, g) for g in SV.RECEIVED) for p in published), published


def _edit(p: Path, f) -> None:
    p.write_text(json.dumps(f(json.loads(p.read_text()))))


@pytest.mark.parametrize("plant, want", [
    (lambda o: (o / "pieces" / "certifier" / "chain.log").write_text("chain on window 3\n"),
     [("pieces/certifier/chain.log", "")]),
    (lambda o: (o / "v3c-audit" / "audit.log").write_text('RECORDS {"population": 37}\n'), [("v3c-audit/audit.log", "")]),
    (lambda o: (o / "v3c-audit" / "main.txt").write_text("ab" * 20 + "\n"), [("v3c-audit/main.txt", "")]),
    (lambda o: _edit(o / "registration.json", lambda r: {**r, "binding": {**r["binding"], "advice": {
        **r["binding"]["advice"], "bits": REALIZED}}, "rows": 1}),
     [("registration.json", "binding.advice.bits"), ("registration.json", "rows")]),
    (lambda o: (o / "registration.json").write_text(json.dumps(ADVICE)),
     [("registration.json", p) for p in ("binding.advice.bits", "bits_by_window.3", "compiled[].request",
                                         "refused[].request", "refused[].why", "requests", "rows", "window")]),
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
    or the registration as `verity-vllm advice` writes it (its requests, compiles, length and window); a workload in the
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
