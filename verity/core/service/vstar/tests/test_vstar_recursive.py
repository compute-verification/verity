"""The recursive audit's verdict (`VStar.Recursive.partVerdicts`, `vstar-recursive verdicts`: V*'s compiled check over a
whole recursive session, one part per V* statement at the verifier's inner shape) against `flock-verify verify --zk` at
each part's flags, as pod 85's lean step runs them (`85-rec-reprice.sh` STEP=lean: `--rec-inner`, `--rec-part`,
`--rec-rows`, `--rec-bindings`, the inner statement's files, `--rec-inner-session` and `--rec-inner-transcript`). The rows
are the one shared file every part is verified against (`rec_vstage.staged_shared_rows`, 85's ``rows.json``), which the
compiled check reads from the evidence (the staged rows), not from the verifier's inputs.

The verifier's own inputs (`own_json`) and the evidence's `recursive` (`proofs_json`) are built from the files those
flags name, by SHA-512, and every file is handed to `verdicts` as a blob. The format tests hold that build to what
`ownOfJson` and `proofsOf` parse. With ``VERITY_VSTAR_RECURSIVE_DATA=DIR``, a `recursion_e2e` run's kept data (its
``data/TAG``, `KEEP_DATA=1`, after STEP=lean), three recursive sessions are put to both: the honest one, every part
accepted by each; the inner session with rep 0's challenge coin flipped (85's ``inner-neg-challenge``); and the honest
one with one bit of part 0's first rep file flipped. Each part's verdict is the same in both, and a refusal of the whole
check (an input `verify` refuses before any session) is `flock-verify`'s exit 2 at every part.

The refusals the compiled check adds to `flock-verify`'s per-part path are each held to a refusal by both: the inner
statement under Flock's plain verifier (`innerOk`, here an inner draw that does not parse), and plain SHA-512 leaves under
`--zk` (`verifyCmd`'s message, which `partVerdict` gives word for word; its statement is the constant
`verity/flock-circuit`, so only `flock-verify`'s side is driven). `fits`' bound, `2^56` bytes, is past any file a test can
stage; `statementFits`, the same bound on the whole statement the inputs name, is driven at a limit of the honest
statement's own size (`vstar-recursive statement LIMIT`), which also checks that each parsed field's canonical bytes
parse back to what the check read, the `Names` pins at the honest session."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import settings
import verity_difftest as D

LEAN = Path(__file__).resolve().parents[1]
RECURSIVE = LEAN / "recursive"
FLOCK = LEAN.parent / "flock"
STATEMENT = "verity/flock-circuit"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def sha(p: Path) -> str:
    return hashlib.sha512(p.read_bytes()).hexdigest()


# --- what the verifier holds and what the prover sends, from a recursive session's files -----------------------------------

class Inner:
    """The inner statement's files (85's ``stage/``) and the inner session (``inner/s<k>``: ``session/`` and the coin
    server's transcript ``public.json``)."""

    def __init__(self, stage: Path, session: Path):
        (self.pub,) = sorted(stage.glob("pub-*.bin"))
        self.circuit, self.partition = stage / "circuit.txt", stage / "partition.json"
        self.program, self.draw = stage / "program.json", stage / "draw.json"
        self.session, self.transcript = session / "session", session / "public.json"

    def flags(self) -> list[str]:
        return ["--rec-inner-circuit", str(self.circuit), "--rec-inner-public", str(self.pub), "--rec-inner-partition",
                str(self.partition), "--rec-inner-statement", "verity/flock-circuit+plain-leaves",
                "--rec-inner-statement-coins", "os", "--rec-inner-draw", str(self.draw), "--rec-inner-session",
                str(self.session), "--rec-inner-transcript", str(self.transcript)]


class Part:
    """V*'s statement ``name`` as staged (``stmt``: its circuit, public file, partition, program and registered reads), the
    shared ``rows`` file and the verifier's ``bindings`` file for it, and its proved session (``session``)."""

    def __init__(self, name: str, stmt: Path, rows: Path, bindings: Path, session: Path):
        self.name, self.stmt, self.rows, self.bindings, self.session = name, stmt, rows, bindings, session

    def reps(self) -> list[Path]:
        return sorted(self.session.glob("*.bin"))


def _json(p: Path):
    return json.loads(p.read_text())


def own_json(inner: Inner, parts: list[Part]) -> dict:
    """`ownOfJson`'s input: the inner statement's program and partition, and `recursive` (the inner statement's digests
    and each part's verifier-held inputs, in V*'s order)."""
    def part(p: Part) -> dict:
        reg = p.stmt / "registered.json"
        return {"bindings": _json(p.bindings), "circuit": sha(p.stmt / "circuit.txt"), "partition": _json(p.stmt / "partition.json"),
                "program": _json(p.stmt / "program.json"), "registered": _json(reg) if reg.is_file() else None}
    return {"partition": _json(inner.partition), "program": _json(inner.program),
            "recursive": {"circuit": sha(inner.circuit), "draw": sha(inner.draw), "parts": [part(p) for p in parts],
                          "public": sha(inner.pub), "registration": None}}


def proofs_json(inner: Inner, parts: list[Part]) -> dict:
    """`proofsOf`'s input: the inner session's record and transcript, the staged shared rows, and each part's public file,
    record and rep files."""
    return {"parts": [{"public": sha(p.stmt / "pub.bin"), "record": sha(p.session / "session.json"),
                       "reps": {f.name: sha(f) for f in p.reps()}} for p in parts],
            "record": sha(inner.session / "session.json"), "rows": sha(parts[0].rows), "transcript": sha(inner.transcript)}


def blobs(inner: Inner, parts: list[Part]) -> list[Path]:
    fs = [inner.circuit, inner.pub, inner.draw, inner.session / "session.json", inner.transcript, parts[0].rows]
    for p in parts:
        fs += [p.stmt / "circuit.txt", p.stmt / "pub.bin", p.session / "session.json", *p.reps()]
    return list(dict.fromkeys(fs))


def lean_verdicts(exe: Path, inner: Inner, parts: list[Part], tmp: Path) -> dict:
    """`vstar-recursive verdicts`' line: ``{"parts": [...]}`` or ``{"refused": why}``."""
    own, proofs = tmp / "own.json", tmp / "proofs.json"
    own.write_text(D.canonical(own_json(inner, parts)))
    proofs.write_text(D.canonical(proofs_json(inner, parts)))
    p = subprocess.run([str(exe), "verdicts", str(own), str(proofs), *map(str, blobs(inner, parts))], capture_output=True,
                       text=True, check=False)
    lines = p.stdout.splitlines()
    assert p.returncode in (0, 1) and len(lines) == 1, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    out = json.loads(lines[0])
    assert p.returncode == (0 if "parts" in out and all(x["accepted"] for x in out["parts"]) else 1), out
    return out


def lean_statement(exe: Path, inner: Inner, parts: list[Part], tmp: Path, limit: int) -> dict:
    """`vstar-recursive statement LIMIT`' line: the statement's size (`statementSize`), whether each parsed field's
    canonical bytes parse back to what the check read (`roundTrips`), and `statementFits LIMIT`'s refusal ("" if none)."""
    own, proofs = tmp / "own.json", tmp / "proofs.json"
    own.write_text(D.canonical(own_json(inner, parts)))
    proofs.write_text(D.canonical(proofs_json(inner, parts)))
    p = subprocess.run([str(exe), "statement", str(limit), str(own), str(proofs), *map(str, blobs(inner, parts))],
                       capture_output=True, text=True, check=False)
    lines = p.stdout.splitlines()
    assert p.returncode == 0 and len(lines) == 1, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    return json.loads(lines[0])


def flock_verdict(exe: Path, inner: Inner, rec_inner: Path, j: int, p: Part) -> dict:
    """`flock-verify verify --zk` on part ``j``'s session at 85's flags: exit code, whether its last VERDICT accepted, why."""
    reg = p.stmt / "registered.json"
    cmd = [str(exe), "verify", "--zk", "--statement", STATEMENT, "--circuit", str(p.stmt / "circuit.txt"), "--public",
           str(p.stmt / "pub.bin"), "--partition", str(p.stmt / "partition.json"), "--program", str(p.stmt / "program.json"),
           *(["--registered", str(reg)] if reg.is_file() else []), "--rec-inner", str(rec_inner), "--rec-rows", str(p.rows),
           "--rec-part", str(j), "--rec-bindings", str(p.bindings), *inner.flags(), "--session", str(p.session)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = (r.stdout + r.stderr).splitlines()
    vs = [json.loads(ln[8:]) for ln in lines if ln.startswith("VERDICT ")]
    why = (vs[-1].get("why") or "") if vs else next((ln for ln in reversed(lines) if ln.strip()), "")[:400]
    return {"rc": r.returncode, "accepted": r.returncode == 0 and bool(vs and vs[-1].get("accepted")), "why": why}


def agree(lean: dict, flock: list[dict], parts: list[Part]) -> list[tuple[str, bool]]:
    """Each part's verdict is `flock-verify`'s; a whole-check refusal is its exit 2 (refused before any session) at every
    part. Returns each part's (name, accepted)."""
    print(json.dumps({"lean": lean, "flock-verify": flock}))
    if "refused" in lean:
        assert all(f["rc"] == 2 for f in flock), (lean, flock)
        return [(p.name, False) for p in parts]
    got = [(x["statement"], x["accepted"]) for x in lean["parts"]]
    assert [n for n, _ in got] == [p.name for p in parts]
    assert [a for _, a in got] == [f["accepted"] for f in flock], [(p.name, x, f) for p, x, f in zip(parts, lean["parts"], flock)]
    assert all(f["rc"] in (0, 1) for f in flock), flock
    return got


# --- the format: what the build writes is what `ownOfJson` and `proofsOf` parse -----------------------------------------------

D64 = "ab" * 64


def _own(**rec) -> dict:
    part = {"bindings": {}, "circuit": D64, "partition": {}, "program": {}, "registered": None}
    return {"partition": {"program": D64}, "program": {"p": 1},
            "recursive": {"circuit": D64, "draw": D64, "parts": [part], "public": D64, "registration": None} | rec}


def _proofs(**kw) -> dict:
    return {"parts": [{"public": D64, "record": D64, "reps": {"circuit.rep0.bin": D64}}], "record": D64, "rows": D64,
            "transcript": D64} | kw


def _verdicts(tmp: Path, own: dict, proofs: dict) -> dict:
    (tmp / "own.json").write_text(D.canonical(own))
    (tmp / "proofs.json").write_text(D.canonical(proofs))
    p = subprocess.run([str(D.build(RECURSIVE, "vstar-recursive")), "verdicts", str(tmp / "own.json"), str(tmp / "proofs.json")],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 1, (p.stdout, p.stderr)
    return json.loads(p.stdout)


def test_the_built_inputs_parse_and_the_check_asks_for_the_inner_circuit(tmp_path):
    """`own_json`'s and `proofs_json`'s shape parses (`ownOfJson`, `proofsOf`): with no blobs the check's first refusal is
    the inner circuit's file, which it reads first."""
    assert _verdicts(tmp_path, _own(), _proofs()) == {"refused": f"the inner circuit: no blob has the SHA-512 {D64}"}


@pytest.mark.parametrize(("own", "proofs", "why"), [
    (_own(extra=1), _proofs(), "the keys of recursive are not"),
    (_own(draw="AB" * 64), _proofs(), "recursive: draw is not a SHA-512 digest"),
    (_own(), _proofs(extra=1), "the keys of the evidence's recursive are not"),
])
def test_a_wrong_input_is_refused_before_any_blob(tmp_path, own, proofs, why):
    assert why in _verdicts(tmp_path, own, proofs)["refused"]


#: `verifyCmd`'s refusal of plain SHA-512 leaves under `--zk`, `partVerdict`'s reason for the same statement
PLAIN_LEAVES = ("verify --zk: refused: {name}, a statement with plain SHA-512 leaves (flock-leaf/sha512-unsalted): "
                "--zk's hiding needs hm96-sha512/v1 leaves")


def test_plain_leaves_under_zk_are_refused_with_the_compiled_checks_reason(tmp_path):
    """`flock-verify verify --zk` at a plain-leaves statement refuses before reading a file (exit 2), with the reason
    `partVerdict`'s `need (!tags0.plainLeaves)` gives."""
    name = "verity/flock-circuit+plain-leaves"
    for f in ("c", "p", "comp"):
        (tmp_path / f).write_bytes(b"")
    (tmp_path / "s").mkdir()
    r = subprocess.run([str(D.build(FLOCK, "flock-verify")), "verify", "--zk", "--statement", name, "--circuit",
                        str(tmp_path / "c"), "--comp", str(tmp_path / "comp"), "--public", str(tmp_path / "p"),
                        "--session", str(tmp_path / "s")], capture_output=True, text=True, check=False)
    assert r.returncode == 2 and PLAIN_LEAVES.format(name=name) in r.stderr, (r.returncode, r.stdout, r.stderr)


# --- a whole recursive session: a recursion_e2e run's kept data ---------------------------------------------------------------

DATA = settings.VSTAR_RECURSIVE_DATA
DEPLOYED = pytest.mark.skipif(not DATA, reason="a recursion_e2e run's kept data/TAG (KEEP_DATA=1, after STEP=lean): set "
                              "VERITY_VSTAR_RECURSIVE_DATA=DIR")


class Session:
    """85's data dir: the inner statement and its last inner session, V*'s statements at the verifier's inner shape
    (`rec-inner`, `rec-statements`), and for each the staged statement and proved session (``v-sess/<s>``, whose
    ``<s>.stmt`` names its staged directory), with the shared rows and each part's bindings as 85's lean step writes them."""

    def __init__(self, dd: Path, o: Path):
        from verity_flock import rec_bindings as RB
        from verity_flock import rec_vstage as VS

        self.dd, self.o = dd, o
        self.flock = D.build(FLOCK, "flock-verify")
        self.exe = D.build(RECURSIVE, "vstar-recursive")
        last = max((dd / "inner").glob("s[0-9]*"), key=lambda p: int(p.name[1:]))
        self.inner = Inner(dd / "stage", last)
        i = self.inner
        self.rec_inner = o / "rec-inner.json"
        self.rec_inner.write_text(subprocess.run([str(self.flock), "rec-inner", "--circuit", str(i.circuit), "--public",
                                                  str(i.pub), "--partition", str(i.partition), "--program", str(i.program)],
                                                 capture_output=True, text=True, check=True).stdout)
        txt = subprocess.run([str(self.flock), "rec-statements", "--rec-inner", str(self.rec_inner)], capture_output=True,
                             text=True, check=True).stdout
        (line,) = [json.loads(ln[9:]) for ln in txt.splitlines() if ln.startswith("RECVSTAR ")]
        names = [s[0] for s in line["statements"]]
        rows = o / "rows.json"
        rows.write_text(json.dumps(VS.staged_shared_rows(dd / "v")))
        by_unit = {}
        for s in sorted(p for p in (dd / "v-sess").iterdir() if p.is_dir() and not p.name.startswith("forged-")):
            c = Path((dd / "v-sess" / f"{s.name}.stmt").read_text().strip())
            with (c / "circuit.txt").open() as f:
                unit = f.readline().split()[1]
            b = o / f"bindings-{s.name}.json"
            b.write_text(json.dumps(RB.bindings(_json(c / "partition.json"), _json(c / "program.json"))))
            by_unit[unit] = Part(unit, c, rows, b, s)
        missing = [n for n in names if n not in by_unit]
        assert not missing, f"{dd}: no proved session of V*'s {missing}: the run's LEVELS left them out"
        self.parts = [by_unit[n] for n in names]

    def check(self, inner: Inner, parts: list[Part], tmp: Path) -> list[tuple[str, bool]]:
        lean = lean_verdicts(self.exe, inner, parts, tmp)
        with ThreadPoolExecutor(settings.LEAN_JOBS) as ex:
            flock = list(ex.map(lambda jp: flock_verdict(self.flock, inner, self.rec_inner, *jp), enumerate(parts)))
        return agree(lean, flock, parts)


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> Session:
    return Session(Path(DATA), tmp_path_factory.mktemp("own"))


@pytest.mark.slow
@DEPLOYED
def test_the_honest_recursive_session_is_accepted_part_for_part(session, tmp_path):
    """Every part of the honest session is accepted by V*'s compiled check, as by `flock-verify` at its flags."""
    got = session.check(session.inner, session.parts, tmp_path)
    assert all(a for _, a in got), got


@pytest.mark.slow
@DEPLOYED
def test_the_honest_statement_fits_and_is_refused_at_a_limit_of_its_own_size(session, tmp_path):
    """`statementFits`: the statement the honest session's inputs name (`statementSize`) is under `FIELD_MAX` (`2^56`)
    bytes, and every parsed field it carries as canonical bytes parses back to what the check read (`roundTrips`: the
    `Names` pins hold at the honest session). A statement of `2^56` bytes can't be staged, so the oversized case is the
    honest statement at a limit of exactly its size: refused, and accepted one byte above."""
    got = lean_statement(session.exe, session.inner, session.parts, tmp_path, 2 ** 56)
    assert got.get("round_trips") is True and 0 < got["size"] < 2 ** 56 and got["fits"] == "", got
    n = got["size"]
    assert lean_statement(session.exe, session.inner, session.parts, tmp_path, n + 1)["fits"] == ""
    over = lean_statement(session.exe, session.inner, session.parts, tmp_path, n)
    assert over["fits"] == f"the statement the inputs name is {n} bytes, {n} or more", over
    print(json.dumps({"size": n, "round_trips": got["round_trips"]}))


@pytest.mark.slow
@DEPLOYED
def test_a_flipped_inner_coin_is_refused_where_flock_verify_refuses_it(session, tmp_path):
    """85's inner session with rep 0's challenge coin flipped (``inner-neg-challenge``, `rec_inner.py forge`): each part's
    verdict is `flock-verify`'s, and some part refuses it."""
    neg = session.dd / "inner-neg-challenge"
    if not (neg / "session" / "session.json").is_file():
        pytest.skip(f"{neg}: the run's lean step made no flipped inner session (NEG_INNER without challenge)")
    got = session.check(Inner(session.dd / "stage", neg), session.parts, tmp_path)
    assert not all(a for _, a in got), got


@pytest.mark.slow
@DEPLOYED
def test_a_rep_file_one_bit_off_is_refused_at_its_part(session, tmp_path):
    """Part 0's session with one bit of its first rep file flipped, on a copy: refused at part 0 by both, every other part
    as the honest session."""
    p0 = session.parts[0]
    cp = tmp_path / "sess0"
    shutil.copytree(p0.session, cp)
    f = min(cp.glob("*.bin"))
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 1
    f.write_bytes(bytes(b))
    parts = [Part(p0.name, p0.stmt, p0.rows, p0.bindings, cp), *session.parts[1:]]
    got = session.check(session.inner, parts, tmp_path)
    assert got[0][1] is False and all(a for _, a in got[1:]), got


@pytest.mark.slow
@DEPLOYED
def test_an_inner_draw_that_does_not_parse_is_refused_by_both(session, tmp_path):
    """The inner statement's draw cut short: the compiled check refuses the whole check at Flock's plain verifier on the
    inner statement (`innerOk`), and `flock-verify` accepts no part."""
    inner = copy.copy(session.inner)
    inner.draw = tmp_path / "draw.json"
    inner.draw.write_bytes(session.inner.draw.read_bytes()[:-2])
    lean = lean_verdicts(session.exe, inner, session.parts, tmp_path)
    assert lean.get("refused", "").startswith("rec inner statement: Flock's plain verifier refuses it: draw: not JSON"), lean
    with ThreadPoolExecutor(settings.LEAN_JOBS) as ex:
        flock = list(ex.map(lambda jp: flock_verdict(session.flock, inner, session.rec_inner, *jp), enumerate(session.parts)))
    print(json.dumps({"lean": lean, "flock-verify": flock}))
    assert not any(f["accepted"] for f in flock), flock
