"""served_cases.py: one-stage v1's case (no composite's), the sampled proofs case (`verity/sampled-proofs/v1`:
v2's case, ``case-v2.json``, in the form ``one-stage``) and the inference-only case (`verity/inference-only/v0`) for a
served window, from its records audit's v1 case (a5's ``window/case.json``), PoUW's period case once its windows name the
same record, Pearl+'s run case from the parts its piece wrote, its commitment naming that record, and with ``--v2`` its
bare v2 case, each in a 0700 directory beside its Lean answer, in the directory form a
composition reads as a case (remote's three subprotocols; no composite runs v2's bare case), and every answer naming the
window's registration record's digest. A window whose a5 wrote no v2 case has no sampled proofs case, refused saying why.

Under H1's tie the window's record R is the Commit record H1 named before its sample coin, a v2 record chained right
after H1's window registration W and reusing W's tables, as soundness's ``remotetie_demo.py`` builds them (`tied_serve`):
all three cases commit to R, the policy's log is the serve's H1 log through W, and a period whose windows name another
record (H1's own W, or none) is refused before any composite, so neither product is handed it. v1's own ``parameters``
name the window's tables as values ahead of its templates', and a Program that isn't that population is refused.

On a toy window (two templates of four slots each, as a served window's population program holds its class's
templates) v1's Lean (`sampled-proofs-one-stage check`) accepts an audit of it whose verdict of record accepted, rejects
one whose verdict refused, and names the record's digest either way; the inference-only case is accepted by the policy's
Lean check (`inference-only check`) at Γ its gates and α its input bits a gate (which ``--from-division`` reads off the
check's profile at no budget), and refused one gate or one bit under; v2's Lean (`sampled-proofs-one-stage-zk check`)
names the record's digest as its commitment (it rejects the toy's v2 case, which holds no coins and no proofs), and so
does sampled proofs' Lean (`vstar-recursive sampled check`) on that case in its form, which refuses v1's case in the same
form by name. A
tampered field is refused: by the Lean check (the record naming another Program, an input node in a block, θ on an
unregistered input, an evidence key, the record repeated in the log) or by the producer before any Lean runs (no v1 case
or one holding another key, a record that isn't v1's Program's, a previous record with no log, α and Γ not given, a v2
case that isn't the audit's, which refuses sampled proofs' too), and an answer that isn't the window's record's is
written nowhere, its refusal saying the
answer's reason. The slow test makes the inference-only case on the 4,32,4 class's own Program (TINY on the H200)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import stat
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.protocols.pouw.verifier import credit as Cr
from verity.protocols.pouw.verifier import weights as PW
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as DR
from verity.protocols.spot.verifier.one_stage import partition as P
from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity.protocols.spot.verifier.one_stage import registered as G
from verity.protocols.spot.verifier.one_stage import registration as R
from verity.circuits import codec, partition_object as PO
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array, Value as Bits
from verity.commitments.crypto import lms

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
V1 = SC.V1
SAMPLED = SC.SAMPLED
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


def audited_v1_case(program, query, accepted: bool = True) -> dict:
    """v1's case of a whole audit of ``program`` (the sampled proofs difftest's accepted case): its record over the
    window's ports and their window domains, a stratified sample's first proof blocks of each stratum proved, the
    verifier's receipt, and the verdict of record ``accepted``."""
    law, leaf = {"law": "stratified", "k": 2}, "ee" * 64
    part, population = P.partition(program, query), P.units(program, query)
    ports = {p: {"domain": R.window_domain(PO.program_sha512(program), P.digest(part), p, schema, leaves, "r-test"),
                 "leaves": leaves}
             for p, schema, leaves in (("out", "u16", 2 * population), ("x", "hm96-sha512/row/v1", population))}
    roots = {"out": "cd" * 64, "x": "ab" * 64}
    rec = R.record(program, query, law, S.SCHEME, [{"port": p, "root": roots[p], **ports[p]} for p in ports],
                   {"kind": "served", "row": 101, "run": "r-test"}, leaf_layer=leaf)
    drawn = DR.derive(program, query, law)
    units = sorted(u for s in drawn["strata"] for u in [u for lo, hi in s["units"] for u in range(lo, hi)][:s["k"]])
    draw = {**drawn, "population": population, "units": units}
    proved = DR.proved(draw)
    evidence = {"draw": draw, "instances": len(proved), "proved_units": proved,
                "receipt": R.receipt(rec, "v", received="2026-10-07T00:00:00Z"), "served": None, "session_roots": None,
                "sessions": [], "statement": {}, "verdicts": {"lean": {"accepted": accepted, "of_record": True}}}
    params = {"anchors": [], "delta": "1/20", "law": law, "leaf_layer": leaf, "log": [], "partition": None,
              "ports": ports, "program": json.loads(codec.canonical_json(codec.encode_program(program))),
              "query": query, "registered": None, "scheme": S.SCHEME, "work_table": None}
    return {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json(evidence).hex(),
            "params": params}


def toy() -> tuple[object, dict]:
    templates = [PAIR, TRIPLE]
    return P.mixed_population_program([(t, SLOTS) for t in templates]), P.template_instances_query(templates)


def audit_dir(root: Path, v1: dict, *, v2: bool | str = True) -> Path:
    """A records audit's out directory holding v1's case and, with ``v2``, a5's v2 case over two blob files (a string:
    a5's ``case-v2-error.txt`` instead). The v2 case's params name a challenger (a stand-in key of LMS's form at the
    deployed parameters, the only ones v2 reads a key under) and its window's root, without which v2's Lean refuses
    the params before it reads the record."""
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
        evidence["proofs"] = {"sessions": [{"name": "5e" * 64, "record": "bb" * 64, "reps": {}}]}
        key = {"H": lms.DEPLOYED.H, "W": lms.DEPLOYED.W, "public": "00" * 80}
        (w / "case-v2.json").write_text(json.dumps({
            "blobs": [str(b) for b in blobs], "commitment": v1["commitment"],
            "evidence": codec.canonical_json(evidence).hex(),
            "params": {**v1["params"], "challenger": {"key": key, "label": "draw"},
                       "proofs": {"circuit": "cc" * 64, "public": "aa" * 64, "registered": None},
                       "window_root": "0f" * 64}},
            sort_keys=True))
    return root / "audit"


def digest_of(v1: dict) -> str:
    return R.digest(json.loads(bytes.fromhex(v1["commitment"])))


def stand_in(tmp_path: Path, name: str, answer: dict | None) -> Path:
    """A Lean `check` stand-in: records what it was handed (its argv, its files' modes and the commitment's bytes) and
    prints ``answer``'s line (none for None)."""
    exe, seen = tmp_path / name, tmp_path / f"{name}.seen"
    line = "" if answer is None else json.dumps(answer, sort_keys=True)
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\na = sys.argv[1:]\n"
                   "files = a[a.index('check') + 1:][:3]\n"
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


def stand_ins(tmp_path: Path, case: dict, **answers: dict | None) -> list:
    """``--v1-exe``, ``--sampled-exe`` and ``--inference-only-exe`` stand-ins (``v1``, ``sampled``, ``reg``) answering for
    ``case``'s record, or ``answers``' instead."""
    d = digest_of(case)
    got = {"v1": answer(V1, d), "sampled": answer(SAMPLED, d), "reg": answer(INF, d)} | answers
    return ["--v1-exe", stand_in(tmp_path, "v1", got["v1"]), "--sampled-exe", stand_in(tmp_path, "sampled", got["sampled"]),
            "--inference-only-exe", stand_in(tmp_path, "reg", got["reg"])]


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


#: H1's toy window as soundness's ``remotetie_demo.py`` builds it: one round at K = 2048 with m = 4, its A table and
#: input ids registered under the H1 window Program's domain and the window's open root (stand-in digests and roots),
#: each as credit names it in the Commit record and as H1 names it in its own registration
H1 = hashlib.sha512(b"the H1 window Program's descriptor").hexdigest()
OPEN = hashlib.sha512(b"the H1 window's open root").hexdigest()
K, M = 2048, 4
TABLES = ((Cr.commit_rows(0, K), f"pouw/h1/a/{K}@0", PW.chunks(K) * M), (Cr.commit_ids(0), "pouw/h1/ids@0", 8))
#: the auditor's ledger digest a period case commits to
LEDGER = "1e" * 64


def h1_registration(previous: str | None = None, entry: int = 0) -> dict:
    """H1's window registration W at the window's close (`pouw_h1_commit.registration`): a v2 record of the toy's
    tables under the H1 window Program, chained after ``previous``."""
    entries = []
    for _, value, leaves in TABLES:
        e = {"value": value, "leaves": leaves, "words": 1, "root": "ab" * 64}
        entries.append({**e, "domain": Cr.domain_of(H1, OPEN, e)})
    return {"format": R.FORMAT_V2, "protocol": R.PROTOCOL, "program": H1,
            "window": {"kind": "served", "entry": entry, "open_root": OPEN, "refused": []},
            "registered": entries, "previous": previous}


def tied_v1_case(w: dict) -> dict:
    """v1's case over the window's record R as V3c's window-commit writes it under H1's tie: a v2 record of the toy's
    population, which takes the window's tables as values ahead of its templates' parameters, registering each at W's
    root, domain and leaves, chained right after W; its params the v2 params as a5 writes them under the tie
    (`replay.params` with ``log`` [W], the registered words, ``outputs`` and ``schemas``, `a5.tie`)."""
    templates = [PAIR, TRIPLE]
    program = P.mixed_population_program([(t, SLOTS) for t in templates],
                                         values=[(name, leaves, 16) for name, _, leaves in TABLES])
    query = P.template_instances_query(templates)
    law, ports = {"law": "stratified", "k": 1}, {"x": {"leaves": 8, "domain": "02" * 64}}
    reused = {e["value"]: e for e in w["registered"]}
    rec = R.record(program, query, law, S.SCHEME, [{"port": "x", "root": "01" * 64, **ports["x"]}],
                   {"kind": "served", "run": "r-served-cases", "row": 0}, previous=R.digest(w),
                   registered=[{**reused[value], "value": name} for name, value, _ in TABLES])
    params = RP.params(program, query, law, ports, None, "1/100", partition=P.partition(program, query), log=[w],
                       registered={name: 1 for name, _, _ in TABLES},
                       outputs={name: {"record": R.digest(w), "value": value} for name, value, _ in TABLES},
                       schemas={name: G.SCHEMA_V2 for name, _, _ in TABLES})
    evidence = {"draw": {"units": [0]}, "sessions": ["5e" * 64], "verdicts": {"m0": {"accepted": True}}}
    return {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json(evidence).hex(),
            "params": json.loads(json.dumps(params))}


def h1_log(h1: Path, records: list[tuple[int, str, dict]]) -> None:
    """The serve's H1 log in ``h1`` as `serve_tie` writes it: each ``(window, kind, record)`` in
    ``w<window>/<kind>.json`` and a line naming that file and the record's digest."""
    lines = []
    for k, kind, r in records:
        f = h1 / f"w{k}" / f"{kind}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, sort_keys=True))
        lines.append({"digest": R.digest(r), "file": f"w{k}/{kind}.json", "kind": kind, "window": k})
    (h1 / "log.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))


def tied_serve(root: Path, commits: list | None = None) -> tuple[Path, dict, dict]:
    """A serve dir under H1's tie (`served_verdicts.sh`'s H1_TIE) at its one audited window: V3c's audit of the window's
    record R (``v3c-w0-…/audit``), H1's log of W then R (``h1/``), and PoUW's period case as `served_pouw.sh` has
    `pouw_piece.py period` write it (``private/pouw/period``), its windows' ``commit`` each of ``commits`` (default: one
    window naming R, its canonical JSON in hex as credit reads it). The dir, v1's case and W."""
    w = h1_registration()
    v1 = tied_v1_case(w)
    audit_dir(root / "v3c-w0-4-32-4", v1)
    h1_log(root / "h1", [(0, "registration", w), (0, "record", json.loads(bytes.fromhex(v1["commitment"])))])
    piece = root / SC.PERIOD_PIECE
    piece.mkdir(parents=True)
    evidence = {"pins": [], "windows": [{"commit": c} for c in ([v1["commitment"]] if commits is None else commits)]}
    (piece / "case.json").write_text(json.dumps({"blobs": [], "commitment": LEDGER, "params": {"period": "p"},
                                                 "evidence": codec.canonical_json(evidence).hex()}, sort_keys=True))
    (piece / "answer.json").write_text(json.dumps(answer(SC.PERIOD, LEDGER), sort_keys=True) + "\n")
    return root, v1, w


@pytest.fixture
def toy_v1() -> dict:
    return v1_case(*toy())


@pytest.fixture(scope="module")
def inference_only_exe() -> Path:
    return SC.build(*SC.INFERENCE_ONLY_LEAN)


@pytest.fixture(scope="module")
def v1_exe() -> Path:
    return SC.build(*SC.V1_LEAN)


@pytest.fixture(scope="module")
def v2_exe() -> Path:
    return PF.v2_executable()


@pytest.fixture(scope="module")
def sampled_exe() -> Path:
    return SC.build(*SC.SAMPLED_LEAN)


def test_each_case_goes_private_beside_an_answer_naming_the_records_digest(tmp_path, capsys):
    """Each case is written in its own 0700 directory (files 0600) with the answer line its Lean printed, the Lean ran
    on files in a 0700 scratch directory that is gone after, v1's case is a5's as a5 wrote it, sampled proofs' is v2's
    with the form ``one-stage`` in its params and its evidence and went to `vstar-recursive sampled check` with v2's
    blobs, v2's blobs went to its Lean, and nothing printed carries a case's bytes."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    audit = audit_dir(tmp_path, v1)
    private = tmp_path / "serve" / "private"
    rc, cases, refused, out = run(capsys, "--audit", audit, "--private", private, "--alpha", "3/2", "--gamma", 100,
                                  *stand_ins(tmp_path, v1), "--v2", "--v2-exe",
                                  stand_in(tmp_path, "v2", answer(SC.V2, d)))
    assert (rc, refused) == (0, []), out
    assert cases == [{"dir": str(private / "sampled-proofs-v1"), "protocol": V1, "verdict": "accept"},
                     {"dir": str(private / "sampled-proofs"), "protocol": SAMPLED, "verdict": "accept"},
                     {"dir": str(private / "inference-only"), "protocol": INF, "verdict": "accept"},
                     {"dir": str(private / "sampled-proofs-v2"), "protocol": SC.V2, "verdict": "accept"}]
    v2_case = json.loads((audit / "window" / "case-v2.json").read_text())
    for c in cases:
        cd = Path(c["dir"])
        assert stat.S_IMODE(cd.stat().st_mode) == 0o700
        assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in cd.iterdir()} == {"case.json": 0o600, "answer.json": 0o600}
        got = json.loads((cd / "case.json").read_text())
        assert got["commitment"] == v1["commitment"]
        assert json.loads((cd / "answer.json").read_text())["commitment"] == d
    assert json.loads((private / "sampled-proofs-v1" / "case.json").read_text()) == v1
    assert json.loads((private / "sampled-proofs-v2" / "case.json").read_text()) == v2_case
    sampled = json.loads((private / "sampled-proofs" / "case.json").read_text())
    assert sampled == {**v2_case, "params": {**v2_case["params"], "form": "one-stage"}, "evidence": codec.canonical_json(
        {**json.loads(bytes.fromhex(v2_case["evidence"])), "form": "one-stage"}).hex()}
    assert "proofs" in sampled["params"] and "verdicts" not in json.loads(bytes.fromhex(sampled["evidence"]))
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
    v1seen, sseen, v2seen, regseen = (json.loads((tmp_path / f"{n}.seen").read_text())
                                      for n in ("v1", "sampled", "v2", "reg"))
    assert v2seen["argv"][4:] == v2_case["blobs"] and v1seen["argv"][4:] == regseen["argv"][4:] == []
    assert sseen["argv"][:2] == ["sampled", "check"] and sseen["argv"][5:] == v2_case["blobs"]
    assert all(s["modes"] == [0o600] * 3 and s["dir"] == 0o700 for s in (v1seen, sseen, v2seen, regseen))
    assert v1seen["commitment"] == sseen["commitment"] == v2seen["commitment"] == regseen["commitment"] == v1["commitment"]
    assert list((private / "scratch").iterdir()) == []
    for secret in (v1["commitment"], v1["evidence"], reg["evidence"], v2_case["evidence"], sampled["evidence"],
                   reg["params"]["program"][:64]):
        assert secret not in out


def test_without_v2_no_v2_case_is_made(tmp_path, capsys):
    """Without ``--v2`` no bare v2 case is made and v2's Lean never runs. A window whose a5 wrote no v2 case (a plain
    audit's) has no sampled proofs case either, which is refused saying why (remote's composite then names sampled
    proofs missing), and v1's and the inference-only case are still written."""
    program, query = toy()
    v1 = v1_case(program, query)
    private = tmp_path / "private"
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1, v2="ValueError: this audit's are plain"),
                                  "--private", private, "--alpha", "1", "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, INF], out
    assert refused == [{"protocol": SAMPLED, "reason": "no v2 case: ValueError: this audit's are plain"}], refused
    assert sorted(p.name for p in private.iterdir()) == ["inference-only", "sampled-proofs-v1", "scratch"]
    assert not (tmp_path / "sampled.seen").exists()


def test_on_a_tree_without_the_policy_the_inference_only_case_is_refused_by_name(tmp_path, capsys, monkeypatch):
    """Where the tree leaves the inference-only policy out (the public release: no module `SC.POLICY`, which names the
    policy's Lean package and the Python its check is held to), the inference-only case is refused saying so, never
    skipped, its Lean never runs, and v1's and sampled proofs' cases are still written; a caller that asks for the
    package is refused the same way."""
    program, query = toy()
    v1 = v1_case(program, query)
    private = tmp_path / "private"
    monkeypatch.setattr(SC, "POLICY", "a_module_this_tree_leaves_out")
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", private,
                                  "--alpha", "1", "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, SAMPLED], out
    assert [r["protocol"] for r in refused] == [INF] and "is not on this tree" in refused[0]["reason"], refused
    assert sorted(p.name for p in private.iterdir()) == ["sampled-proofs", "sampled-proofs-v1", "scratch"]
    assert not (tmp_path / "reg.seen").exists()
    with pytest.raises(SC.Refused, match="is not on this tree"):
        SC.build(*SC.INFERENCE_ONLY_LEAN)


@lean
@pytest.mark.parametrize("accepted, verdict", [(True, "accept"), (False, "reject")])
def test_v1s_lean_checks_the_audits_case_and_names_the_records_digest(tmp_path, capsys, v1_exe, accepted, verdict):
    """`sampled-proofs-one-stage check` on a5's case of a whole audit of the toy (its record, sample, receipt and
    verdict of record): it accepts when the verdict of record accepted and rejects when it refused, and names the
    record's digest either way, so its verdict is the case's (a failed proof is a visible failure, never a missing
    case)."""
    v1 = audited_v1_case(*toy(), accepted=accepted)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private",
                                  tmp_path / "private", "--alpha", "1", "--gamma", 1, "--v1-exe", v1_exe,
                                  "--sampled-exe", stand_in(tmp_path, "sampled", answer(SAMPLED, digest_of(v1))),
                                  "--inference-only-exe", stand_in(tmp_path, "reg", answer(INF, digest_of(v1))))
    assert rc == 0 and refused == [], out
    assert [c["verdict"] for c in cases if c["protocol"] == V1] == [verdict]
    got = json.loads((tmp_path / "private" / "sampled-proofs-v1" / "answer.json").read_text())
    assert got["protocol"] == V1 and got["commitment"] == digest_of(v1), got
    assert got["reason"] == (None if accepted else "verdict lean did not accept"), got


@lean
@pytest.mark.parametrize("change, verdict, reason", [
    ("at-budget", "accept", None),
    ("gamma-1", "reject", "the division does not comply: inference block 0 has more than Γ gates"),
    ("alpha-under", "reject", "the division does not comply: inference block 0 reads more than α bits of fresh input "
                              "per gate"),
])
def test_the_lean_check_accepts_the_inference_only_case_at_its_budgets_and_not_past_them(
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
                                  "--v1-exe", stand_in(tmp_path, "v1", answer(V1, digest_of(v1))),
                                  "--sampled-exe", stand_in(tmp_path, "sampled", answer(SAMPLED, digest_of(v1))))
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
def test_the_lean_check_refuses_a_tampered_inference_only_case(tmp_path, inference_only_exe, change, reason):
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
    the one `registration.digest` computes; the toy's v2 case holds no coins and no proofs, so it rejects, and that
    rejection is the case's verdict (a failed proof is a visible failure, never a missing case)."""
    program, query = toy()
    v1 = v1_case(program, query)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--v2", "--v2-exe", v2_exe, "--alpha", "1", "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 0 and refused == [], out
    assert [c["verdict"] for c in cases if c["protocol"] == SC.V2] == ["reject"]
    got = json.loads((tmp_path / "private" / "sampled-proofs-v2" / "answer.json").read_text())
    assert got["protocol"] == SC.V2 and got["commitment"] == digest_of(v1)


@lean
def test_sampled_proofs_lean_names_the_records_digest_and_refuses_v1_by_name(tmp_path, capsys, sampled_exe):
    """`vstar-recursive sampled check` reads the toy's v2 case in the form ``one-stage`` through its commitment and names
    the record's digest; it rejects it, as v2's Lean does, and that rejection is the case's verdict. v1's case in the
    same form is refused by name at its params, the reason remote's composite would give for a v1 case."""
    program, query = toy()
    v1 = v1_case(program, query)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--alpha", "1", "--gamma", 1, *stand_ins(tmp_path, v1), "--sampled-exe", sampled_exe)
    assert rc == 0 and refused == [], out
    assert [c["verdict"] for c in cases if c["protocol"] == SAMPLED] == ["reject"]
    got = json.loads((tmp_path / "private" / "sampled-proofs" / "answer.json").read_text())
    assert got["protocol"] == SAMPLED and got["commitment"] == digest_of(v1), got
    got, _ = SC.lean_answer(sampled_exe, SC.with_form(v1), tmp_path / "scratch", SC.SAMPLED_CMD)
    assert got["protocol"] == SAMPLED and got["verdict"] == "reject", got
    assert got["reason"].startswith("params: the parameters name no proofs: one-stage v1 "), got
    assert "is refused" in got["reason"], got


@pytest.mark.parametrize("change, reason", [
    ("other-program", "the record names another Program than v1's params"),
    ("previous-no-log", "pass the deployment's log (--log)"),
    ("log-unchained", "the log and the record don't chain"),
    ("no-alpha-gamma", "α and Γ are the policy's, not the run's: pass --alpha and --gamma"),
    ("gamma-bool", "Γ is a natural"),
])
def test_the_producer_refuses_an_inference_only_case_that_would_not_be_the_windows(tmp_path, capsys, change, reason):
    """Refused before any Lean runs, with v1's case still written: the record naming another Program than v1's params,
    a previous record with no log or a log that doesn't chain to it, α and Γ not given."""
    program, query = toy()
    v1 = v1_case(program, query, previous="77" * 64 if change in ("previous-no-log", "log-unchained") else None)
    if change == "other-program":
        rec = json.loads(bytes.fromhex(v1["commitment"]))
        v1["commitment"] = codec.canonical_json({**rec, "program": "ee" * 64}).hex()
    argv = ["--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private", *stand_ins(tmp_path, v1)]
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
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, SAMPLED], out
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
    the check's profile at no budget. The CASE line publishes neither (on a one-block window they'd be its block's gates
    and fresh-input bits): they stay in the case's private ``policy.json``, beside where they come from. A rejection
    there, or a profile with no blocks or a block of no gates, is refused before the case is checked."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--from-division", *stand_ins(tmp_path, v1, reg=answer(INF, d, verdict)
                                                                | {"profile": profile}))
    if reason is None:
        assert rc == 0 and refused == [], out
        assert [sorted(c) for c in cases] == [["dir", "protocol", "verdict"]] * 3
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
                                  "--v1-exe", stand_in(tmp_path, "v1", answer(V1, digest_of(toy_v1))),
                                  "--sampled-exe", stand_in(tmp_path, "sampled", answer(SAMPLED, digest_of(toy_v1))))
    assert rc == 0 and refused == [], out
    f = Fraction(bits, gates)
    assert [(sorted(c), c["verdict"]) for c in cases if c["protocol"] == INF] == [(["dir", "protocol", "verdict"],
                                                                                    "accept")]
    d = tmp_path / "private" / "inference-only"
    params = json.loads((d / "case.json").read_text())["params"]
    assert (params["alpha"], params["gamma"]) == ([f.numerator, f.denominator], gates)
    assert json.loads((d / "policy.json").read_text()) == {"alpha": [f.numerator, f.denominator], "gamma": gates,
                                                           "read": SC.FROM_DIVISION}


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
    """a5's v2 case is taken only as a5 wrote it for this audit, for sampled proofs' case and with ``--v2`` for the bare
    one: none when a5 wrote its error instead (which says why) or nothing, and refused when it commits to another record,
    names a blob that isn't there or holds another key."""
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
                                  "--gamma", 1, *stand_ins(tmp_path, v1),
                                  "--v2", "--v2-exe", stand_in(tmp_path, "v2", answer(SC.V2, digest_of(v1))))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, INF], out
    assert [r["protocol"] for r in refused] == [SAMPLED, SC.V2] and all(reason in r["reason"] for r in refused), refused
    for d, seen in (("sampled-proofs", "sampled"), ("sampled-proofs-v2", "v2")):
        assert not (tmp_path / "private" / d).exists() and not (tmp_path / f"{seen}.seen").exists()


@pytest.mark.parametrize("change, reason", [
    ("error", "no v1 case: ValueError: the audit's record has no receipt"),
    ("key", "window/case.json is not a case: it holds exactly ['commitment', 'evidence', 'params']"),
])
def test_no_case_is_written_when_a5s_v1_case_is_not_one(tmp_path, capsys, change, reason):
    """v1's case is the base of every other: when a5 wrote its ``case-error.txt`` instead (which says why) or a case
    holding another key, every case is refused and no Lean runs."""
    program, query = toy()
    v1 = v1_case(program, query)
    audit = audit_dir(tmp_path, v1, v2=False)
    if change == "error":
        (audit / "window" / "case.json").unlink()
        (audit / "window" / "case-error.txt").write_text("ValueError: the audit's record has no receipt\n")
    else:
        (audit / "window" / "case.json").write_text(json.dumps(v1 | {"public": {}}))
    rc, cases, refused, out = run(capsys, "--audit", audit, "--private", tmp_path / "private", "--alpha", "1",
                                  "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and cases == [], out
    assert [r["protocol"] for r in refused] == [V1, SAMPLED, INF] and all(reason in r["reason"] for r in refused), refused
    assert not (tmp_path / "private").exists() and not (tmp_path / "v1.seen").exists()


@pytest.mark.parametrize("who", [V1, SAMPLED, INF])
@pytest.mark.parametrize("change, reason", [
    ("other-digest", "'s answer commits to"),
    ("null-digest", "'s answer commits to None, not the window's registration record's digest: stand-in"),
    ("other-protocol", "the answer is 'verity/pouw/v1''s"),
    ("no-line", "answered no line"),
])
def test_an_answer_that_is_not_the_windows_records_is_written_nowhere(tmp_path, capsys, who, change, reason):
    """Each Lean answer must be its protocol's and name the window's registration record's digest, which is what
    remote's and onsite's record tie compare; otherwise neither the case nor the answer is written, and the other case
    still is."""
    program, query = toy()
    v1 = v1_case(program, query)
    d = digest_of(v1)
    bad = {"other-digest": answer(who, "ee" * 64), "null-digest": answer(who, None, "reject"),
           "other-protocol": answer("verity/pouw/v1", d), "no-line": None}[change]
    exes = stand_ins(tmp_path, v1, **{{V1: "v1", SAMPLED: "sampled", INF: "reg"}[who]: bad})
    rc, cases, refused, out = run(capsys, "--audit", audit_dir(tmp_path, v1), "--private", tmp_path / "private",
                                  "--alpha", "1", "--gamma", 1, *exes)
    others = [x for x in (V1, SAMPLED, INF) if x != who]
    assert rc == 1 and [c["protocol"] for c in cases] == others, out
    assert [r["protocol"] for r in refused] == [who] and reason in refused[0]["reason"], refused
    assert not (tmp_path / "private" / SC.DIRS[who]).exists()
    assert all((tmp_path / "private" / SC.DIRS[x]).is_dir() for x in others)


def test_the_serve_dir_names_its_one_audited_window(tmp_path, capsys):
    """``--serve-dir`` takes the one ``v3c-w*/audit`` holding a case (`served_verdicts.sh`'s ``$VW/audit``), and refuses
    every case when there are two or none."""
    program, query = toy()
    v1 = v1_case(program, query)
    serve = tmp_path / "serve"
    audit_dir(serve / "v3c-w0-a", v1)
    exes = stand_ins(tmp_path, v1)
    rc, cases, _, out = run(capsys, "--serve-dir", serve, "--private", serve / "private", "--alpha", "1", "--gamma", 1,
                            *exes)
    assert rc == 0 and len(cases) == 3, out
    audit_dir(serve / "v3c-w1-a", v1)
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", serve / "private2", "--alpha", "1",
                                  "--gamma", 1, *exes)
    assert rc == 1 and cases == [] and [r["protocol"] for r in refused] == [V1, SAMPLED, INF], out
    assert "holds 2 audited windows" in refused[0]["reason"]


def test_under_h1s_tie_every_case_commits_to_the_record_h1_named(tmp_path, capsys):
    """The serve dir under H1's tie: the one-stage case is V3c's audit's, the policy's commits to the same R with its log
    the serve's H1 log through W (which R is chained right after), and PoUW's period case, every window naming R, is
    written as `pouw_piece.py` wrote it beside its own answer. R's digest is the one credit reads off each period window's
    Commit record, so the three name one record, the tie remote checks."""
    serve, v1, w = tied_serve(tmp_path / "serve")
    rec, d = json.loads(bytes.fromhex(v1["commitment"])), digest_of(v1)
    assert rec["previous"] == R.digest(w) and Cr.commit_of(v1["commitment"])[0][0].hex() == d
    private = serve / "private"
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", private, "--alpha", "1", "--gamma", 1,
                                  *stand_ins(tmp_path, v1))
    assert (rc, refused) == (0, []), out
    assert cases == [{"dir": str(private / d_), "protocol": id, "verdict": "accept"} for id, d_ in (
        (V1, "sampled-proofs-v1"), (SAMPLED, "sampled-proofs"), (INF, "inference-only"), (SC.PERIOD, "pouw-period"))]
    piece, got = serve / SC.PERIOD_PIECE, private / "pouw-period"
    assert stat.S_IMODE(got.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in got.iterdir()} == {"case.json": 0o600, "answer.json": 0o600}
    assert json.loads((got / "case.json").read_text()) == json.loads((piece / "case.json").read_text())
    assert (got / "answer.json").read_text() == (piece / "answer.json").read_text()
    reg = json.loads((private / "inference-only" / "case.json").read_text())
    assert reg["commitment"] == v1["commitment"] and reg["params"]["log"] == [w]
    assert reg["params"]["parameters"] == v1["params"]["parameters"]
    assert [n for n, _ in reg["params"]["parameters"]] == [name for name, _, _ in TABLES] + ["0.x", "1.x"]
    assert json.loads((tmp_path / "reg.seen").read_text())["commitment"] == v1["commitment"]
    for secret in (v1["commitment"], reg["evidence"], json.loads((piece / "case.json").read_text())["evidence"]):
        assert secret not in out


@pytest.mark.parametrize("change, reason", [
    ("h1-registration", "window 0 of the period names the Commit record {w}…, not the window's registration record "
                        "{r}…: remote's record tie rejects it"),
    ("second-window", "window 1 of the period names the Commit record {w}…"),
    ("no-commit", "window 0 of the period names no Commit record, so not the window's registration record {r}…"),
    ("v1-record", "window 0 of the period names no Commit record"),
    ("no-windows", "the period's evidence names no windows"),
    ("other-answer", f"the period's answer is {V1!r}'s, not {SC.PERIOD}'s"),
    ("no-answer", "the period's case has no answer line"),
    ("case-key", "case.json is not a case"),
])
def test_a_period_on_another_record_is_refused_before_any_composite(tmp_path, capsys, change, reason):
    """A period case is taken only when every window names R as its Commit record: one naming H1's own registration W
    (the record H1 named before the tie), one window of two on W, none, or a v1 record, is refused here and so is handed to
    neither product; so is a period case with no windows, no answer, another protocol's answer or another key. v1's and
    the policy's cases are still written."""
    w = h1_registration()
    on_r, on_w = tied_v1_case(w)["commitment"], codec.canonical_json(w).hex()
    commits = {"h1-registration": [on_w], "second-window": [on_r, on_w], "no-commit": [None], "no-windows": [],
               "v1-record": [codec.canonical_json({**w, "format": R.FORMAT}).hex()]}.get(change)
    serve, v1, _ = tied_serve(tmp_path / "serve", commits)
    piece = serve / SC.PERIOD_PIECE
    if change == "other-answer":
        (piece / "answer.json").write_text(json.dumps(answer(V1, LEDGER)) + "\n")
    elif change == "no-answer":
        (piece / "answer.json").unlink()
    elif change == "case-key":
        (piece / "case.json").write_text(json.dumps({**json.loads((piece / "case.json").read_text()), "public": {}}))
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", serve / "private", "--alpha", "1",
                                  "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, SAMPLED, INF], out
    want = reason.format(w=R.digest(w)[:16], r=digest_of(v1)[:16])
    assert [r["protocol"] for r in refused] == [SC.PERIOD] and want in refused[0]["reason"], refused
    assert not (serve / "private" / "pouw-period").exists()


def pearlplus_parts(serve: Path) -> tuple[Path, dict]:
    """Pearl+'s case's parts as `served_passes.py case` writes them in the serve dir (`PEARL_PLUS_PIECE`): one pass's
    manifest (its salt key among its fields), the one file its opened tile reads by the blob holding it, and the Lean
    verdict of record's verdict, tiles drawn and work."""
    piece = serve / SC.PEARL_PLUS_PIECE
    (piece / "blobs").mkdir(parents=True)
    blob = b"the pass's rows"
    (piece / "blobs" / hashlib.sha512(blob).hexdigest()).write_bytes(blob)
    parts = {"schema": "verity-pouw/pearl-plus-case-parts/v1", "delta": "1/100",
             "params": {"coins": {"epoch": 0, "root": "c0" * 64, "scheme": "s"}, "draws": 3,
                        "record": {"commitment": "d0" * 64, "source": "e0" * 32}},
             "manifests": [{"hm96_salt_key": "5a" * 32, "run": "r"}],
             "evidence": {"files": [{"pearl/w0/rows.bin": hashlib.sha512(blob).hexdigest()}]},
             "lean": {"verdict": "accept", "reason": None, "drawn": 3, "work": {"credited": 7, "macs": 9}}}
    (piece / "parts.json").write_text(json.dumps(parts, sort_keys=True))
    return piece, parts


def test_pearl_plus_s_case_is_its_piece_s_parts_naming_the_records_digest(tmp_path, capsys):
    """Pearl+'s run case from the parts its piece wrote (`served_passes.py case`): its params as written, its commitment
    the retained manifests and R's digest (the tie remote's composite of record checks against the policy's record), as
    canonical JSON, its evidence each pass's files by blob digest and its blobs the parts' own; its answer the Lean verdict
    of record's under the commitment's digest, SHA-512(tag ‖ 0 ‖ commitment). Nothing of the commitment is printed. A
    part naming a blob that isn't a file is refused saying so, and the other cases are still written."""
    serve, v1, _ = tied_serve(tmp_path / "serve")
    piece, parts = pearlplus_parts(serve)
    private = serve / "private"
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", private, "--alpha", "1", "--gamma", 1,
                                  *stand_ins(tmp_path, v1))
    assert (rc, refused) == (0, []), out
    assert cases[-1] == {"dir": str(private / "pouw-pearl-plus"), "protocol": SC.PEARL_PLUS, "verdict": "accept"}
    assert [c["protocol"] for c in cases] == [V1, SAMPLED, INF, SC.PERIOD, SC.PEARL_PLUS]
    got = private / "pouw-pearl-plus"
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in got.iterdir()} == {"case.json": 0o600, "answer.json": 0o600}
    case = json.loads((got / "case.json").read_text())
    commitment = bytes.fromhex(case["commitment"])
    assert commitment == codec.canonical_json({"manifests": parts["manifests"], "record": digest_of(v1)})
    assert (case["params"], bytes.fromhex(case["evidence"])) == (parts["params"], codec.canonical_json(parts["evidence"]))
    h = parts["evidence"]["files"][0]["pearl/w0/rows.bin"]
    assert case["blobs"] == [str(piece / "blobs" / h)]
    assert json.loads((got / "answer.json").read_text()) == {
        "protocol": SC.PEARL_PLUS, "verdict": "accept", "reason": None,
        "commitment": hashlib.sha512(b"verity/pouw/pearl-plus/manifests/v1\0" + commitment).hexdigest(),
        "public": {"credited": 7, "macs": 9, "delta": "1/100"}, "profile": {"draws": 3, "drawn": 3, "passes": 1}}
    assert "5a" * 32 not in out and case["commitment"] not in out
    (piece / "blobs" / h).unlink()
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", tmp_path / "p2", "--alpha", "1",
                                  "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, SAMPLED, INF, SC.PERIOD], out
    assert [r["protocol"] for r in refused] == [SC.PEARL_PLUS], refused
    assert "the parts name 1 blob(s) that aren't files" in refused[0]["reason"] and h not in out


def test_period_names_the_period_case_to_check_against_the_audits_record(tmp_path, capsys):
    """``--period`` names the period case beside ``--audit``; with neither it nor a serve dir holding one, no period case
    is made, and a v1 case that isn't one refuses the period with every other."""
    serve, v1, w = tied_serve(tmp_path / "serve")
    audit, piece, exes = serve / "v3c-w0-4-32-4" / "audit", serve / SC.PERIOD_PIECE, stand_ins(tmp_path, v1)
    log = tmp_path / "log.json"
    log.write_text(json.dumps([w]))
    rc, cases, _, out = run(capsys, "--audit", audit, "--private", tmp_path / "p1", "--period", piece, "--log", log,
                            "--alpha", "1", "--gamma", 1, *exes)
    assert rc == 0 and [c["protocol"] for c in cases] == [V1, SAMPLED, INF, SC.PERIOD], out
    rc, cases, _, out = run(capsys, "--audit", audit, "--private", tmp_path / "p2", "--log", log, "--alpha", "1",
                            "--gamma", 1, *exes)
    assert rc == 0 and [c["protocol"] for c in cases] == [V1, SAMPLED, INF], out
    (audit / "window" / "case.json").unlink()
    (audit / "window" / "case-error.txt").write_text("ValueError: the audit's record has no receipt\n")
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", tmp_path / "p3", "--alpha", "1",
                                  "--gamma", 1, *exes)
    assert rc == 1 and cases == [] and [r["protocol"] for r in refused] == [V1, SAMPLED, INF, SC.PERIOD], out
    assert not (tmp_path / "p3").exists()


def test_the_policys_log_is_the_serves_h1_log_through_the_records_previous(tmp_path):
    """Over a serve of several windows, each window's W then R, the log before a window's R is every record through its
    own W, which chains from the deployment's first record (null previous) to R; later windows' records aren't in it."""
    w0 = h1_registration()
    r0 = {"format": R.FORMAT_V2, "previous": R.digest(w0), "program": "11" * 64}
    w1 = h1_registration(R.digest(r0), entry=1)
    r1 = {"format": R.FORMAT_V2, "previous": R.digest(w1), "program": "11" * 64}
    w2 = h1_registration(R.digest(r1), entry=2)
    h1_log(tmp_path, [(0, "registration", w0), (0, "record", r0), (1, "registration", w1), (1, "record", r1),
                      (2, "registration", w2)])
    assert SC.h1_log(tmp_path / "log.jsonl", r1) == [w0, r0, w1]
    assert SC.h1_log(tmp_path / "log.jsonl", r0) == [w0]
    assert SC.h1_log(tmp_path / "nowhere.jsonl", {**r0, "previous": None}) == []
    SC._chain([w0, r0, w1], r1)


@pytest.mark.parametrize("change, reason", [
    ("other-file", "the serve's H1 log names w0/registration.json as the record of digest"),
    ("no-file", "a line of the serve's H1 log names no record file and digest"),
    ("unchained", "the record's previous is no record the serve's H1 log names"),
    ("not-json", "the record names a previous record, and the serve's H1 log isn't readable: JSONDecodeError"),
    ("no-log", "the record names a previous record, so pass the deployment's log (--log)"),
])
def test_the_policy_case_is_refused_when_the_serves_h1_log_does_not_reach_its_record(tmp_path, capsys, change, reason):
    """The policy's log is read off the serve's H1 log only as it names its records: a line whose file isn't the record
    of its digest, a line naming none, a log that never reaches R's previous or isn't JSON, or no log for an R naming
    one, refuses the policy's case alone (one-stage's and the period's are still written)."""
    serve, v1, w = tied_serve(tmp_path / "serve")
    h1 = serve / "h1"
    if change == "other-file":
        (h1 / "w0" / "registration.json").write_text(json.dumps({**w, "window": {**w["window"], "entry": 9}}))
    elif change == "no-file":
        (h1 / "log.jsonl").write_text(json.dumps({"digest": R.digest(w), "kind": "registration", "window": 0}) + "\n")
    elif change == "unchained":
        h1_log(h1, [(0, "registration", h1_registration(entry=5))])
    elif change == "not-json":
        (h1 / "log.jsonl").write_text("{\n")
    else:
        (h1 / "log.jsonl").unlink()
    rc, cases, refused, out = run(capsys, "--serve-dir", serve, "--private", serve / "private", "--alpha", "1",
                                  "--gamma", 1, *stand_ins(tmp_path, v1))
    assert rc == 1 and [c["protocol"] for c in cases] == [V1, SAMPLED, SC.PERIOD], out
    assert [r["protocol"] for r in refused] == [INF] and reason in refused[0]["reason"], refused


@pytest.mark.parametrize("change, reason", [
    ("table-leaves", "isn't the population program over its root's templates and the values its params' parameters "
                     "name"),
    ("template-type", "v1's params' parameters aren't its Program's program function's"),
    ("value-type", "v1's params' parameter input-ids/0 is no template's and no registered value"),
    ("none", "isn't the population program over its root's templates, so its program function's parameters"),
])
def test_v1s_parameters_name_the_windows_tables_and_are_its_programs(change, reason):
    """The policy case's parameters are the ones the rebuilt population program over v1's templates and v1's tables has:
    tables of other leaves, another template parameter's type or a table that isn't an array of values is refused, and
    so is a window Program that takes tables when v1's params name none (a5's params before the v2 params)."""
    w = h1_registration()
    v1 = tied_v1_case(w)
    ps = v1["params"]["parameters"]
    at = {n: i for i, (n, _) in enumerate(ps)}
    if change == "table-leaves":
        ps[at[Cr.commit_rows(0, K)]][1] = Array(3, Bits(16)).to_json()
    elif change == "template-type":
        ps[at["0.x"]][1] = Array(SLOTS, Array(2, Bits(16))).to_json()
    elif change == "value-type":
        ps[at[Cr.commit_ids(0)]][1] = Bits(16).to_json()
    else:
        del v1["params"]["parameters"]
    with pytest.raises(SC.Refused, match=reason.replace("(", r"\(").replace(")", r"\)")):
        SC.inference_only_case(v1, 1, 1, [w])


@lean
def test_the_lean_check_accepts_the_policy_case_on_the_tied_record(tmp_path, inference_only_exe, toy_v1):
    """`inference-only check` on the policy case over R with its log [W]: it reads W as the deployment's first record and
    R as chained right after it, names R's digest, and accepts at no budget with the same profile as the toy's untied
    record, so the window's tables, parameters no node reads, add no gate and no fresh-input bit. Without W in the log
    R's chain breaks, and it rejects."""
    w = h1_registration()
    v1 = tied_v1_case(w)
    case = SC.inference_only_case(v1, *SC.UNBOUNDED, [w])
    got, _ = SC.lean_answer(inference_only_exe, case, tmp_path / "s")
    assert (got["protocol"], got["commitment"], got["verdict"]) == (INF, digest_of(v1), "accept"), got
    assert got["profile"]["blocks"] == [list(budgets(toy_v1, inference_only_exe, tmp_path / "s"))], got
    case["params"]["log"] = []
    got, _ = SC.lean_answer(inference_only_exe, case, tmp_path / "s")
    assert (got["protocol"], got["verdict"], got["reason"].split(":")[0]) == (INF, "reject", "trusted"), got


@pytest.mark.parametrize("argv", [
    ["--private", "p"],
    ["--audit", "a", "--private", "p", "--from-division", "--alpha", "1"],
    ["--audit", "a", "--private", "p", "--from-division", "--gamma", "1"],
    ["--audit", "a", "--private", "p", "--alpha", "1", "--gamma", "1", "--v2-exe", "x"],
])
def test_a_usage_error_exits_64(capsys, argv):
    with pytest.raises(SystemExit) as e:
        SC.main(argv)
    assert e.value.code == SC.USAGE


#: the H200's declared target (`pins/NVIDIA_H200.json`), at which V2 serves the 4,32,4 class
H200 = {"compute_capability": [9, 0], "num_sms": 132}
#: `adapters/vllm`'s tests' TINY Llama, at whose caps (64 positions) the 4,32,4 class's 36 fit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}


@lean
@pytest.mark.slow
def test_the_inference_only_case_on_the_4_32_4_class_is_accepted_at_its_budgets(tmp_path, inference_only_exe):
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
