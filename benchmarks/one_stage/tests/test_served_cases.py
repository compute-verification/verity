"""served_cases.py: sampled proofs' v2 case and the inference-only case (`verity/inference-only/v0`) for a served
window, from its records audit's v1 case (a5's ``window/case.json``) and v2 case (``case-v2.json``), each in a 0700
directory beside its Lean answer, in the form `composite.py --cases` reads, and both answers naming the window's
registration record's digest.

On a toy window (two templates of four slots each, as a served window's population program holds its class's
templates) the inference-only case is accepted by the policy's Lean check (`inference-only check`) at Γ its gates and α
its input bits a gate (which ``--from-division`` reads off the check's profile at no budget), and refused one gate or
one bit under; v2's Lean (`sampled-proofs-one-stage-zk check`) names the
record's digest as its commitment (it rejects the toy's proofs, which are no proofs). A tampered field is refused: by the
Lean check (the record naming another Program, an input node in a block, θ on an unregistered input, an evidence key,
the record repeated in the log) or by the producer before any Lean runs (a record that isn't v1's Program's, a previous
record with no log, α and Γ not given, a v2 case that isn't the audit's), and an answer that isn't the window's record's
is written nowhere. The slow test makes the inference-only case on the 4,32,4 class's own Program (TINY on the H200)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import stat
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_cases as SC  # noqa: E402

_spec = importlib.util.spec_from_file_location("one_stage_replay", HERE.parent / "replay.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RP
_spec.loader.exec_module(RP)

LAKE = shutil.which("lake") is not None
lean = pytest.mark.skipif(not LAKE, reason="no Lean toolchain")

XOR8 = PrimitiveDefinition("ServedCasesTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("ServedCasesTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("ServedCasesTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
SLOTS = 4
INF = SC.INFERENCE_ONLY
UNNAMED = "the run's registration record is not the declared circuit's: its program is not the circuit's SHA-512"


def v1_case(program, query, previous: str | None = None) -> dict:
    """v1's case as a5 writes it (`replay.Case.case`'s params, a v2 record, its evidence) for ``program``."""
    part = P.partition(program, query)
    ports = {"x": {"leaves": 8, "domain": "02" * 64}}
    law = {"law": "stratified", "k": 1}
    rec = R.record(program, query, law, S.SCHEME, [{"port": "x", "root": "01" * 64, **ports["x"]}],
                   {"kind": "served", "run": "r-served-cases", "row": 0}, previous=previous,
                   registered=[{"value": "weights/w", "root": "ab" * 64, "leaves": 8, "words": 4, "domain": "cd" * 64}])
    params = RP.params(program, query, law, ports, None, "1/100", partition=part)
    evidence = {"draw": {"units": [0]}, "sessions": ["5e" * 64], "verdicts": {"m0": {"accepted": True}}}
    return {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json(evidence).hex(),
            "params": params}


def toy() -> tuple[object, dict]:
    templates = [PAIR, TRIPLE]
    return P.mixed_population_program([(t, SLOTS) for t in templates]), P.template_instances_query(templates)


def audit_dir(root: Path, v1: dict, *, v2: bool | str = True) -> Path:
    """A records audit's out directory holding v1's case and, with ``v2``, a5's v2 case over two blob files (a string:
    a5's ``case-v2-error.txt`` instead)."""
    w = root / "audit" / "window"
    w.mkdir(parents=True)
    (w / "case.json").write_text(json.dumps(v1, sort_keys=True))
    if isinstance(v2, str):
        (w / "case-v2-error.txt").write_text(v2 + "\n")
    elif v2:
        blobs = [root / "proof" / n for n in ("circuit.txt", "pub-4.bin")]
        blobs[0].parent.mkdir()
        for b in blobs:
            b.write_bytes(b.name.encode())
        evidence = {k: x for k, x in json.loads(bytes.fromhex(v1["evidence"])).items() if k != "verdicts"}
        evidence["proofs"] = {"public": "aa" * 64, "sessions": [{"name": "5e" * 64, "record": "bb" * 64, "reps": {}}]}
        (w / "case-v2.json").write_text(json.dumps({
            "blobs": [str(b) for b in blobs], "commitment": v1["commitment"],
            "evidence": codec.canonical_json(evidence).hex(),
            "params": {**v1["params"], "proofs": {"circuit": "cc" * 64, "registered": None}}}, sort_keys=True))
    return root / "audit"


def digest_of(v1: dict) -> str:
    return R.digest(json.loads(bytes.fromhex(v1["commitment"])))


def stand_in(tmp_path: Path, name: str, answer: dict | None) -> Path:
    """A Lean `check` stand-in: records what it was handed (its argv, its files' modes and the commitment's bytes) and
    prints ``answer``'s line (none for None)."""
    exe, seen = tmp_path / name, tmp_path / f"{name}.seen"
    line = "" if answer is None else json.dumps(answer, sort_keys=True)
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\na = sys.argv[1:]\n"
                   "files = a[1:4]\n"
                   "seen = {'argv': a, 'modes': [os.stat(f).st_mode & 0o777 for f in files],\n"
                   "        'dir': os.stat(os.path.dirname(files[0])).st_mode & 0o777,\n"
                   "        'commitment': open(files[1], 'rb').read().hex()}\n"
                   f"open({str(seen)!r}, 'w').write(json.dumps(seen))\n"
                   f"print({line!r})\n")
    exe.chmod(0o755)
    return exe


def answer(id: str, commitment: str | None, verdict: str = "accept") -> dict:
    return {"commitment": commitment, "profile": None, "protocol": id, "public": {} if verdict == "accept" else None,
            "reason": None if verdict == "accept" else "stand-in", "verdict": verdict}


def run(capsys, *argv) -> tuple[int, list[dict], list[dict], str]:
    """`served_cases.main`: its exit, its CASE and NOT_WRITTEN lines' objects, and all it printed."""
    rc = SC.main([str(a) for a in argv])
    out = capsys.readouterr().out
    lines = out.splitlines()
    cases = [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("CASE ")]
    refused = [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("NOT_WRITTEN ")]
    return rc, cases, refused, out


def budgets(v1: dict, exe: Path, scratch: Path) -> tuple[int, int]:
    """The one inference block's gates and fresh-input bits: the Lean check's profile on the case at no budget."""
    got, _ = SC.lean_answer(exe, SC.inference_only_case(v1, *SC.UNBOUNDED), scratch)
    assert got["verdict"] == "accept" and len(got["profile"]["blocks"]) == 1, got
    return tuple(got["profile"]["blocks"][0])


@pytest.fixture
def toy_v1() -> dict:
    return v1_case(*toy())


@pytest.fixture(scope="module")
def inference_only_exe() -> Path:
    return SC.build(*SC.INFERENCE_ONLY_LEAN)


@pytest.fixture(scope="module")
def v2_exe() -> Path:
    return PF.v2_executable()


def test_both_cases_go_private_beside_answers_naming_the_records_digest(tmp_path, capsys):
    """Each case is written in its own 0700 directory (files 0600) with the answer line its Lean printed, the Lean ran on
    files in a 0700 scratch directory that is gone after, v2's blobs went to it, and nothing printed carries a case's
    bytes."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    audit = audit_dir(tmp_path, v1)
    v2x, regx = stand_in(tmp_path, "v2", answer(SC.V2, d)), stand_in(tmp_path, "reg", answer(INF, d))
    private = tmp_path / "serve" / "private"
    rc, cases, refused, out = run(capsys, "--audit", audit, "--private", private, "--alpha", "3/2", "--gamma", 100,
                                  "--v2-exe", v2x, "--inference-only-exe", regx)
    assert (rc, refused) == (0, []), out
    assert cases == [{"dir": str(private / "sampled-proofs-v2"), "protocol": SC.V2, "verdict": "accept"},
                     {"dir": str(private / "inference-only"), "protocol": INF, "verdict": "accept"}]
    v2_case = json.loads((audit / "window" / "case-v2.json").read_text())
    for c in cases:
        cd = Path(c["dir"])
        assert stat.S_IMODE(cd.stat().st_mode) == 0o700
        assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in cd.iterdir()} == {"case.json": 0o600, "answer.json": 0o600}
        got = json.loads((cd / "case.json").read_text())
        assert got["commitment"] == v1["commitment"]
        assert json.loads((cd / "answer.json").read_text())["commitment"] == d
    assert json.loads((private / "sampled-proofs-v2" / "case.json").read_text()) == v2_case
    reg = json.loads((private / "inference-only" / "case.json").read_text())
    assert sorted(reg) == ["commitment", "evidence", "params"]
    assert sorted(reg["params"]) == ["alpha", "gamma", "log", "parameters", "program"]
    assert reg["params"]["alpha"] == [3, 2] and reg["params"]["gamma"] == 100 and reg["params"]["log"] == []
    assert json.loads(reg["params"]["program"]) == v1["params"]["program"]
    assert reg["params"]["parameters"] == [["0.x", Array(SLOTS, Array(2, Bits(8))).to_json()],
                                           ["1.x", Array(SLOTS, Array(3, Bits(8))).to_json()]]
    ev = json.loads(bytes.fromhex(reg["evidence"]))
    assert sorted(ev) == ["owner", "provenance", "theta"] and ev["theta"] == ev["provenance"] == []
    assert set(ev["owner"]) <= {"input", 0} and 0 in ev["owner"]
    v2seen, regseen = (json.loads((tmp_path / f"{n}.seen").read_text()) for n in ("v2", "reg"))
    assert v2seen["argv"][4:] == v2_case["blobs"] and regseen["argv"][4:] == []
    assert all(s["modes"] == [0o600] * 3 and s["dir"] == 0o700 for s in (v2seen, regseen))
    assert v2seen["commitment"] == regseen["commitment"] == v1["commitment"]
    assert list((private / "scratch").iterdir()) == []
    for secret in (v1["commitment"], reg["evidence"], v2_case["evidence"], reg["params"]["program"][:64]):
        assert secret not in out


def test_composite_reads_each_case_as_a_piece_of_the_windows_record(tmp_path, capsys):
    """`composite.py`'s own reader (`case_dirs`, `piece`) takes both directories: each piece is its protocol's, its
    digest the record's, and its blobs v2's, and both products compose both. Composed, its public output passes its own
    leak check and holds neither case's evidence nor the Program's text."""
    spec = importlib.util.spec_from_file_location("one_stage_composite", HERE.parent / "composite.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    audit = audit_dir(tmp_path, v1)
    private = tmp_path / "private"
    rc, cases, _, out = run(capsys, "--audit", audit, "--private", private, "--alpha", "1", "--gamma", 100,
                            "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, d)),
                            "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, d)))
    assert rc == 0, out
    pieces = {p["id"]: p for p in map(C.piece, C.case_dirs([private]))}
    assert set(pieces) == {SC.V2, INF}
    assert all(p["digest"] == bytes.fromhex(d) and p["commitment"] == bytes.fromhex(v1["commitment"])
               for p in pieces.values())
    assert pieces[SC.V2]["blobs"] == json.loads((audit / "window" / "case-v2.json").read_text())["blobs"]
    assert all({SC.V2, INF} <= set(C.COMPONENTS[p]) for p in ("onsite", "remote"))
    capsys.readouterr()
    pub = tmp_path / "public"
    assert C.main(["remote", "--cases", str(private), "--private-out", str(tmp_path / "composed"),
                   "--public-out", str(pub)]) == 0, capsys.readouterr()
    public = b"".join(f.read_bytes() for f in sorted(pub.iterdir())) + capsys.readouterr().out.encode()
    reg = json.loads((private / "inference-only" / "case.json").read_text())
    for secret in (reg["evidence"], reg["params"]["program"][:64], pieces[SC.V2]["evidence"].hex()):
        assert secret.encode() not in public
    assert {c["commitment"] for c in json.loads((pub / "composite.json").read_text())["cases"]} == {v1["commitment"]}


@lean
@pytest.mark.parametrize("change, verdict, reason", [
    ("at-budget", "accept", None),
    ("gamma-1", "reject", "the division does not comply: inference block 0 has more than Γ gates"),
    ("alpha-under", "reject", "the division does not comply: inference block 0 reads more than α bits of fresh input "
                              "per gate"),
])
def test_the_lean_check_accepts_the_registration_case_at_its_budgets_and_not_past_them(
        tmp_path, capsys, inference_only_exe, toy_v1, change, verdict, reason):
    """At Γ the one block's gates and α its input bits a gate (the Lean's profile, which is `nci_reference`'s),
    `inference-only check` accepts; one gate or one bit under, it rejects naming the budget. Its answer names the
    record's digest, so the case is written either way, its verdict the Lean's."""
    v1 = toy_v1
    gates, bits = budgets(v1, inference_only_exe, tmp_path / "s")
    case = SC.inference_only_case(v1, 10**9, 10**12)
    inst = {k: case["params"][k] for k in ("alpha", "gamma", "program")} | json.loads(bytes.fromhex(case["evidence"]))
    assert SC._reference().check_desc(inst)["ius"] == [[gates, bits]]
    gamma, alpha = {"at-budget": (gates, f"{bits}/{gates}"), "gamma-1": (gates - 1, f"{bits}/{gates}"),
                    "alpha-under": (gates, f"{bits - 1}/{gates}")}[change]
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--alpha", alpha, "--gamma", gamma, "--inference-only-exe", inference_only_exe,
                                  "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, digest_of(v1))))
    assert rc == 0 and refused == [], out
    assert [c["verdict"] for c in cases if c["protocol"] == INF] == [verdict]
    got = json.loads((tmp_path / "private" / "inference-only" / "answer.json").read_text())
    assert got["protocol"] == INF and got["commitment"] == digest_of(v1) and got["reason"] == reason
    if verdict == "accept":
        assert got["profile"] == {"blocks": [[gates, bits]], "gates": got["profile"]["gates"]}


@lean
@pytest.mark.parametrize("change, reason", [
    ("record-program", UNNAMED),
    ("input-in-block", "input-in-iu"),
    ("theta-unregistered", "theta-not-registered"),
    ("evidence-key", "evidence: the keys of the evidence are not owner, provenance, theta"),
    ("log-repeat", "trusted"),
])
def test_the_lean_check_refuses_a_tampered_registration_case(tmp_path, inference_only_exe, change, reason):
    program, query = toy()
    v1 = v1_case(program, query)
    case = SC.inference_only_case(v1, 10**9, 10**12)
    rec = json.loads(bytes.fromhex(case["commitment"]))
    ev = json.loads(bytes.fromhex(case["evidence"]))
    first_input = ev["owner"].index("input")
    if change == "record-program":
        case["commitment"] = codec.canonical_json({**rec, "program": "ee" * 64}).hex()
    elif change == "input-in-block":
        ev["owner"][first_input] = 0
    elif change == "theta-unregistered":
        ev["theta"], ev["provenance"] = [first_input], [{"kind": "registered", "registration": digest_of(v1),
                                                         "root": "ab" * 64, "value": "weights/w"}]
    elif change == "evidence-key":
        ev["x"] = 1
    elif change == "log-repeat":
        case["params"]["log"] = [rec]
    case["evidence"] = codec.canonical_json(ev).hex()
    got, _ = SC.lean_answer(inference_only_exe, case, tmp_path / "scratch")
    assert got["protocol"] == INF and got["verdict"] == "reject", got
    assert got["reason"] == reason or got["reason"].startswith(reason + ":"), got


@lean
def test_v2s_lean_names_the_records_digest_so_its_rejection_is_written(tmp_path, capsys, v2_exe):
    """`sampled-proofs-one-stage-zk check` reads the toy's v2 case through its commitment and names the record's digest,
    the one `registration.digest` computes; the toy's proofs are no proofs, so it rejects, and that rejection is the
    case's verdict (a failed proof is a visible failure, never a missing case)."""
    program, query = toy()
    v1 = v1_case(program, query)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--v2-exe", v2_exe, "--alpha", "1", "--gamma", 1,
                                  "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, digest_of(v1))))
    assert rc == 0 and refused == [], out
    assert [c["verdict"] for c in cases if c["protocol"] == SC.V2] == ["reject"]
    got = json.loads((tmp_path / "private" / "sampled-proofs-v2" / "answer.json").read_text())
    assert got["protocol"] == SC.V2 and got["commitment"] == digest_of(v1)


@pytest.mark.parametrize("change, reason", [
    ("other-program", "the record names another Program than v1's params"),
    ("previous-no-log", "pass the deployment's log (--log)"),
    ("log-unchained", "the log and the record don't chain"),
    ("no-alpha-gamma", "α and Γ are the policy's, not the run's: pass --alpha and --gamma"),
    ("gamma-bool", "Γ is a natural"),
])
def test_the_producer_refuses_a_registration_case_that_would_not_be_the_windows(tmp_path, capsys, change, reason):
    """Refused before any Lean runs, with v2's case still written: the record naming another Program than v1's params,
    a previous record with no log or a log that doesn't chain to it, α and Γ not given."""
    program, query = toy()
    v1 = v1_case(program, query, previous="77" * 64 if change in ("previous-no-log", "log-unchained") else None)
    if change == "other-program":
        rec = json.loads(bytes.fromhex(v1["commitment"]))
        v1["commitment"] = codec.canonical_json({**rec, "program": "ee" * 64}).hex()
    argv = ["--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
            "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, digest_of(v1))),
            "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, digest_of(v1)))]
    if change != "no-alpha-gamma":
        argv += ["--alpha", "1", "--gamma", 1]
    if change == "log-unchained":
        (tmp_path / "log.json").write_text(json.dumps([{"format": R.FORMAT_V2, "previous": "66" * 64}]))
        argv += ["--log", tmp_path / "log.json"]
    if change == "gamma-bool":
        with pytest.raises(SC.Refused, match=reason):
            SC.inference_only_case(v1, 1, True)
        return
    rc, cases, refused, out = run(capsys, *argv)
    assert rc == 1 and [c["protocol"] for c in cases] == [SC.V2], out
    assert [r["protocol"] for r in refused] == [INF] and reason in refused[0]["reason"], refused
    assert not (tmp_path / "private" / "inference-only").exists() and not (tmp_path / "reg.seen").exists()


@pytest.mark.parametrize("profile, verdict, reason", [
    ({"blocks": [[10, 3], [6, 4]], "gates": 16}, "accept", None),
    (None, "reject", "the case's own division doesn't pass the check at no budget: stand-in"),
    (None, "accept", "the check's profile isn't a nonempty list"),
    ({"blocks": [[0, 1]], "gates": 0}, "accept", "the check's profile isn't a nonempty list"),
])
def test_from_division_reads_alpha_and_gamma_off_the_checks_profile(tmp_path, capsys, profile, verdict, reason):
    """With ``--from-division``, Γ is the largest block's gates and α the most fresh-input bits a gate of any block, from
    the check's profile at no budget; the CASE line publishes only those numbers, and the case's private
    ``policy.json`` says where they come from. A rejection there, or a profile with no blocks or a block of no gates,
    is refused before the case is checked."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--from-division", "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, d)),
                                  "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, d, verdict)
                                                                 | {"profile": profile}))
    if reason is None:
        assert rc == 0 and refused == [], out
        assert [c.get("alpha_gamma") for c in cases] == [None, [[2, 3], 10]]
        assert SC.FROM_DIVISION not in out
        d = tmp_path / "private" / "inference-only"
        params = json.loads((d / "case.json").read_text())["params"]
        assert (params["alpha"], params["gamma"]) == ([2, 3], 10)
        assert json.loads((d / "policy.json").read_text()) == {"alpha": [2, 3], "gamma": 10, "read": SC.FROM_DIVISION}
        assert stat.S_IMODE((d / "policy.json").stat().st_mode) == 0o600
    else:
        assert rc == 1 and [r["protocol"] for r in refused] == [INF] and reason in refused[0]["reason"], refused
        assert not (tmp_path / "private" / "inference-only").exists()


@lean
def test_from_division_is_accepted_at_the_divisions_own_budgets(tmp_path, capsys, inference_only_exe, toy_v1):
    """The real check at no budget, then at what it read off: Γ the one block's gates and α its input bits a gate, at
    which the case is accepted."""
    gates, bits = budgets(toy_v1, inference_only_exe, tmp_path / "s")
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, toy_v1), "--private", tmp_path / "private",
                                  "--from-division", "--inference-only-exe", inference_only_exe,
                                  "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, digest_of(toy_v1))))
    assert rc == 0 and refused == [], out
    f = Fraction(bits, gates)
    assert [(c["verdict"], c.get("alpha_gamma")) for c in cases if c["protocol"] == INF] == [
        ("accept", [[f.numerator, f.denominator], gates])]
    params = json.loads((tmp_path / "private" / "inference-only" / "case.json").read_text())["params"]
    assert (params["alpha"], params["gamma"]) == ([f.numerator, f.denominator], gates)


def test_the_parameters_are_the_population_programs_and_no_other_programs():
    """The program function's parameters come from the population program rebuilt over the root's templates, one
    template's included; a Program that isn't one (here a template on its own) is refused, its parameters unknown."""
    one = v1_case(P.population_program(PAIR, SLOTS), P.template_instance_query(PAIR))
    assert SC.inference_only_case(one, 1, 1)["params"]["parameters"] == [["x", Array(SLOTS, Array(2, Bits(8))).to_json()]]
    lone = Program(PAIR)
    desc = json.loads(codec.canonical_json(codec.encode_program(lone)))
    rec = {"format": R.FORMAT_V2, "previous": None, "program": codec.program_digest(desc)}
    with pytest.raises(SC.Refused, match="isn't the population program over its root's templates"):
        SC.inference_only_case({"commitment": codec.canonical_json(rec).hex(), "params": {"program": desc}}, 1, 1)


def test_a_log_that_chains_to_the_record_is_the_cases_log(tmp_path):
    program, query = toy()
    first = {"format": R.FORMAT_V2, "previous": None, "program": "11" * 64}
    v1 = v1_case(program, query, previous=R.digest(first))
    assert SC.inference_only_case(v1, 1, 1, [first])["params"]["log"] == [first]


@pytest.mark.parametrize("change, reason", [
    ("error", "no v2 case: ValueError: v2 verifies --zk sessions; this audit's are plain"),
    ("none", "no v2 case: the audit's a5 writes no case-v2.json"),
    ("other-record", "v2's case commits to another record than the audit's"),
    ("blob-gone", "v2's case names 1 blob(s) that aren't files"),
    ("key", "v2's case holds exactly"),
])
def test_v2_is_refused_when_its_case_is_not_the_audits(tmp_path, capsys, change, reason):
    """a5's v2 case is taken only as a5 wrote it for this audit: none when a5 wrote its error instead (which says why) or
    nothing, and refused when it commits to another record, names a blob that isn't there or holds another key."""
    program, query = toy()
    v1 = v1_case(program, query)
    audit = audit_dir(tmp_path, v1, v2={"error": "ValueError: v2 verifies --zk sessions; this audit's are plain",
                                        "none": False}.get(change, True))
    f = audit / "window" / "case-v2.json"
    if f.is_file():
        c = json.loads(f.read_text())
        if change == "other-record":
            c["commitment"] = codec.canonical_json({**json.loads(bytes.fromhex(c["commitment"])), "row": 1}).hex()
        elif change == "blob-gone":
            Path(c["blobs"][0]).unlink()
        elif change == "key":
            c["public"] = {}
        f.write_text(json.dumps(c))
    rc, cases, refused, out = run(capsys, "--audit", audit, "--private", tmp_path / "private", "--alpha", "1",
                                  "--gamma", 1, "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, digest_of(v1))),
                                  "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, digest_of(v1))))
    assert rc == 1 and [c["protocol"] for c in cases] == [INF], out
    assert [r["protocol"] for r in refused] == [SC.V2] and reason in refused[0]["reason"], refused
    assert not (tmp_path / "private" / "sampled-proofs-v2").exists() and not (tmp_path / "v2.seen").exists()


@pytest.mark.parametrize("change, reason", [
    ("other-digest", "'s answer commits to"),
    ("null-digest", "'s answer commits to None"),
    ("other-protocol", "the answer is 'verity/pouw/v1''s"),
    ("no-line", "answered no line"),
])
def test_an_answer_that_is_not_the_windows_records_is_written_nowhere(tmp_path, capsys, change, reason):
    """Each Lean answer must be its protocol's and name the window's registration record's digest, which is what
    remote's and onsite's record tie compare; otherwise neither the case nor the answer is written."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    bad = {"other-digest": answer(INF, "ee" * 64), "null-digest": answer(INF, None, "reject"),
           "other-protocol": answer("verity/pouw/v1", d), "no-line": None}[change]
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--alpha", "1", "--gamma", 1,
                                  "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, d)),
                                  "--inference-only-exe", stand_in(tmp_path, "reg", bad))
    assert rc == 1 and [c["protocol"] for c in cases] == [SC.V2], out
    assert [r["protocol"] for r in refused] == [INF] and reason in refused[0]["reason"], refused
    assert not (tmp_path / "private" / "inference-only").exists()


def test_the_serve_dir_names_its_one_audited_window(tmp_path, capsys):
    """``--serve-dir`` takes the one ``v3c-w*/audit`` holding a case (`served_verdicts.sh`'s ``$VW/audit``), and refuses
    both cases when there are two or none."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    serve = tmp_path / "serve"
    audit_dir(serve / "v3c-w0-a", v1)
    exes = ["--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, d)),
            "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, d))]
    rc, cases, _, out = run(capsys, "--serve-dir", serve, "--private", serve / "private", "--alpha", "1", "--gamma", 1,
                            *exes)
    assert rc == 0 and len(cases) == 2, out
    audit_dir(serve / "v3c-w1-a", v1)
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", serve / "private2", "--alpha", "1",
                                  "--gamma", 1, *exes)
    assert rc == 1 and cases == [] and {r["protocol"] for r in refused} == {SC.V2, INF}, out
    assert "holds 2 audited windows" in refused[0]["reason"]


@pytest.mark.parametrize("argv", [
    ["--private", "p"],
    ["--audit", "a", "--private", "p", "--from-division", "--alpha", "1"],
    ["--audit", "a", "--private", "p", "--from-division", "--gamma", "1"],
])
def test_a_usage_error_exits_64(capsys, argv):
    with pytest.raises(SystemExit) as e:
        SC.main(argv)
    assert e.value.code == SC.USAGE


#: the H200's declared target (`pins/NVIDIA_H200.json`), at which V2 serves the 4,32,4 class
H200 = {"compute_capability": [9, 0], "num_sms": 132}
#: `integrations/vllm`'s tests' TINY Llama, at whose caps (64 positions) the 4,32,4 class's 36 fit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}


@lean
@pytest.mark.slow
def test_the_registration_case_on_the_4_32_4_class_is_accepted_at_its_budgets(tmp_path, inference_only_exe):
    """The 4,32,4 class's window circuit (TINY on the H200: about 85 s and 3 GB to build) as the served window's audit
    holds it, its population program over the class's templates: the inference-only case `inference_only_case` makes from
    it decodes in the policy's Lean check, which accepts it at the α and Γ ``--from-division`` reads off its division
    (`division_policy`: Γ its one block's gates, α its input bits a gate) and rejects it a gate under. The toy tests
    can't catch what only the served shapes have: the class's own Definitions
    (heads cut by key block, scans, registered reads) through `nci_reference.root_nodes` and the Lean decoder, the
    description's walk bound, and the size of a served Program's text through the CLI's files."""
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit

    reg = Registered.of(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target=H200), default_seed=7)
    circuit = class_circuit(reg, WindowClass(4, 32, 4), None, Grain(gemm_words=1024, rows=256, select=64),
                            provable=False)
    program, query = circuit.program()
    v1 = v1_case(program, query)
    alpha, gates = SC.division_policy(v1, [], inference_only_exe, tmp_path / "s")
    bits = alpha * gates
    assert bits.denominator == 1, (alpha, gates)
    bits = bits.numerator
    for gamma, verdict in ((gates, "accept"), (gates - 1, "reject")):
        got, _ = SC.lean_answer(inference_only_exe, SC.inference_only_case(v1, alpha, gamma), tmp_path / "s")
        assert (got["protocol"], got["commitment"], got["verdict"]) == (INF, digest_of(v1), verdict), got
    print(json.dumps({"class": "4,32,4", "templates": len(circuit.templates), "gates": gates, "input_bits": bits,
                      "program_bytes": len(codec.canonical_json(v1["params"]["program"]))}, sort_keys=True))
