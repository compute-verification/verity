"""The recursive audit's event and the premises it takes of V*'s sessions, evaluated on recorded recursive sessions
(`flock-witness`, `Flock/Witness.lean`; `Proofs.Flock.Recursive.Witness` proves each check's `.ok` is the model's
predicate). What a recorded honest session meets is asserted; what none meets yet is asserted refused by name, so the
change that closes it has to say so here. Two recorded sets, both fetched fixtures:

* `rec-coins/16-16-8-n64` (`r20261009-111638-6720`'s inner statement and honest inner session, and level 3's `pub.bin`,
  `registered.json` and outer session, proved again at the current leaf scheme by `r20261010-040650-84a9`), with level
  3's statement restaged by `rec_vstage.stage` as `test_lean_rec_coins` restages it, and checked as
  `verity/flock-circuit`; its outer session was proved before #1677 (art:225370f8… is the level proved again with
  #1677's OS masks, by `r20261010-212517-4f68`, and accepted by today's verifier). The inner description is not
  registered (no staging runs `inner_registered register`). V*'s sessions hide their outputs, as `ZkOuterR.hTags`
  (`TagsOkZHJ`) takes them, and their shared-row public file passes `ScopeZR`, `Typable`'s statement check; the
  public-output one-table checks, `TagsOk` and `ScopeZ`, refuse them.
* `f1-gate`, the F1 gate (`PublicAudit`, the recursive audit at public units): art:df3af922…, 85 at K = 4096, N = 2,
  built, proved and verified at b795ecbde by `r20261010-210008-003d` through `r20261010-214046-3e44` (the custody), V*'s
  statements staged with the inner program; the fixture holds the files these tests read, the circuits and V*'s
  programs xz-compressed, the inner transcript is tracked beside it (`f1-gate-transcript`), and each is checked against
  the art's own listing once decompressed; RecCap's statement and session were staged and proved over the gate's inner
  session after it (`r20261020-002938-635a`, `85-rec-reprice.sh` STEP=vcap and STEP=oprove LEVELS=cap). Lean's verifier
  builds the inner statement from its own partition object and program (`AcceptsPublic`), V*'s seven staged statements
  are the ones `outerStatementsAt` lists, each session is one of them byte for byte (`RecVStar.checkStmt`) and draws every unit,
  and the rate is below 1 at `k = 2^23` (at `2^21` for these sessions). The sessions were proved with #1677's OS masks
  in `verity/flock-circuit`'s `--zk` identity, and the verifier of record accepts them as `verity/flock-circuit`: the
  gate is witnessed under today's identity. The event's last conjunct is shown in its executable form only: the honest
  outputs are the circuit's, and a public file whose one output row is committed one bit off still sets up. That is
  the executable fact `¬ holds` is about, not that Lean's `holds` flips there: no Lean lemma links execution to
  `holds`, and the cheater's row not being the circuit's output is up to a SHA-512 second preimage for that row.

Each inner setup reads the 567 MB inner circuit (about a minute and 3 GB), the gate's files decompress to 2.3 GB, and its
sessions take up to about 13 minutes each, run at once, so the module is slow; it catches what reading the theorems
can't: a premise or event conjunct that no honest session meets."""
from __future__ import annotations

import concurrent.futures as cf
import functools
import hashlib
import json
import lzma
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import keyed
import numpy as np
import pytest
from test_lean_rec_coins import built, forged  # noqa: F401
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate_call
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.program import Program
from verity_flock import circuit as C
from verity_flock import rec_bindings as RB
from verity_flock import rec_live as RL
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-witness"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
STATEMENT = "verity/flock-circuit"
#: the statement rec-coins' level-3 session was proved as: `verity/flock-circuit` before #1677 put the OS's masks in its
#: identity
PROVED = "verity/flock-circuit@cec3e347"
#: `CoinTree.checkHello`'s refusal of a session whose Hello is not the verifier's own
S2_R7 = "S2/R7: the session parameters are not the verifier's"
PRF208 = "PRF-208: fixtures are re-proved after F1 (#5181) lands; this skip goes with that re-proof"
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves (`test_lean_rec_coins`)
CIRCUIT_SHA512 = ("885c04a867b23838a460033b3cfcbd8cad28bc5f994dd10e907724d293479cf7a525e331ed7152ab5e451bb27f39905f695af6295408db"
                  "459a42e2dda3e07463")

#: art:df3af922…'s files the gate's tests read, a file `f` stored as `f.xz` when large, and `prover.sha256`: their lines of
#: the art's own listing (`out/data.sha256`)
F1 = HERE / "fixtures" / "f1-gate"
#: the gate's inner transcript, the coin server's record of its inner session (art:df3af922…'s `inner/s1/public.json`,
#: xz-compressed, and its line of the art's listing), tracked: a check's pod fetches a registered fixture only once `main`
#: registers it
F1_TRANSCRIPT = HERE / "fixtures" / "f1-gate-transcript"
#: V*'s statements at the gate, each staged under `v/` with its outer session under `v-sess/`, in `RecVStar.statements`' order
F1_PARTS = ("L0", "L1", "L2", "L3", "alg/p0", "alg/p1", "cap/L1", "cap/L2", "cap/L3", "cap")
#: `RecVStar.statements` at the inner shape, each with its population: a later level's RecCap holds both reps' tops, 64,
#: 64 and 32 nodes each at levels 1 to 3
F1_STATEMENTS = (("rec-open/l64-h8", 436), ("rec-open/l16-h7", 212), ("rec-open/l16-h5", 142), ("rec-open/l16-h4", 106),
                 ("rec-residuals/m27-k24-c8-p0", 2), ("rec-residuals/m27-k24-c8-p1", 2), ("rec-cap/n128-l1", 128),
                 ("rec-cap/n128-l2", 128), ("rec-cap/n64-l3", 64), ("rec-cap/n128", 128))
#: a standalone program of one call, to evaluate the inner unit's Definition with the IR oracle
_ROOT = CompositeDefinition("Standalone", 1, ("FN",), lambda D: D.FN.signature(), lambda B, D, *args: B.call(D.FN, *args),
                            doc="root wrapper: one call of FN over Input gates", register=False)

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@functools.cache
def _build() -> None:
    subprocess.run(["lake", "build", "flock-witness"], cwd=LEAN, check=True, capture_output=True)


def _witness(*args: str) -> dict[str, dict]:
    p = keyed.run([str(BIN), *args])
    checks = [json.loads(line[len("WITNESS "):]) for line in p.stdout.splitlines() if line.startswith("WITNESS ")]
    assert checks, p.stderr
    return {c["check"]: c for c in checks}


def _kept(once, name: str, run) -> dict[str, dict]:
    """`run`'s checks, run once in the whole run (`once`) and read back by every worker."""
    return json.loads((once(name, lambda d: (d / "w.json").write_text(json.dumps(run()))) / "w.json").read_text())


@pytest.fixture(scope="module")
def inner(once) -> dict[str, dict]:
    assert (SET / "u" / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins`")
    _build()
    u = SET / "u" / "stage"
    return _kept(once, "witness-inner", functools.partial(
        _witness, "inner", "--circuit", str(u / "circuit.txt"), "--public", str(u / "pub-64.bin"), "--partition",
        str(u / "partition.json"), "--record", str(SET / "inner-honest" / "session" / "session.json")))


@pytest.fixture(scope="module")
def level(inner, forged) -> list[str]:  # noqa: F811
    """`flock-witness session`'s files and session at the recorded level, its statement restaged (`test_lean_rec_coins`'
    `forged`, whose circuit, program and partition do not read the coefficients)."""
    level = forged
    assert hashlib.sha512((level / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return ["--circuit", str(level / "circuit.txt"), "--public", str(SET / f"L{LEVEL}" / "pub.bin"), "--partition",
            str(level / "partition.json"), "--program", str(level / "program.json"), "--registered",
            str(SET / f"L{LEVEL}" / "registered.json"), "--session", str(SET / f"outer-L{LEVEL}" / "session")]


@pytest.fixture(scope="module")
def outer(level, once) -> dict[str, dict]:
    return _kept(once, "witness-outer", functools.partial(_witness, "session", "--statement", STATEMENT, *level))


def test_lean_accepts_the_inner_statement(inner):
    """`AcceptsStmt`: `setupTables` at the inner tags sets the statement up as one table, and `Zk.schedule` takes its m."""
    assert inner["accepts_stmt"]["ok"], inner["accepts_stmt"].get("why")


def test_the_inner_session_is_the_inner_statements(inner):
    assert inner["record_sigma"]["ok"], inner["record_sigma"].get("why")


def test_no_recorded_inner_description_is_registered(inner):
    """`AcceptsDescription`, the event's second conjunct, at no recorded session: the inner statement's header names no
    registered rows. It holds once a staging runs `inner_registered register` on the inner statement."""
    c = inner["accepts_description"]
    assert not c["ok"]
    assert c["why"] == ("registered reads: the inner statement names no registered rows: its description committed afresh "
                        "opens no registered root")


@pytest.mark.parametrize("check", ["tags_ok_zhj", "scope_zr", "scope_ok", "full_draw"])
def test_the_recorded_level_meets(outer, check):
    """`TagsOkZHJ` (`ZkOuterR.hTags`), `ScopeZR` (the event's statement check and `Typable`'s, with the registered reads),
    `VStmt.hscope`, and the full draw `RecordCustody` reads."""
    assert outer[check]["ok"], outer[check].get("why")


@pytest.mark.skip(reason=PRF208)
def test_the_recorded_level_was_proved_before_the_os_masks(outer, level):
    """`AcceptsZKR` refuses the level's session as `verity/flock-circuit` at its Hello (its re-record's prover,
    505b9bfb1, lacked #1677's `osMasks`) and accepts it as `PROVED`. Holds as `verity/flock-circuit` once the fixture
    takes the level proved again with the OS-masks prover (art:225370f8…)."""
    assert (outer["accepts_zkr"]["ok"], outer["accepts_zkr"].get("why")) == (False, S2_R7)
    c = _witness("session", "--statement", PROVED, *level)["accepts_zkr"]
    assert c["ok"], c.get("why")


def test_the_recorded_level_fails_scope_z(outer):
    """`ScopeZ`, the statement check without the registered reads, which the fail-closed `--zk` headlines read, refuses
    V*'s shared-row public file. `ZkOuterR.Typable` takes `ScopeZR`, which the level meets."""
    c = outer["scope_z"]
    assert not c["ok"]
    assert c["why"] == ("refused: a shared-row public file (header shared_rows): a statement form the soundness proof does "
                        "not cover")


def test_the_recorded_level_hides_its_outputs(outer):
    """Every V* session is a `verity/flock-circuit` statement's, which hides its outputs: `TagsOk`, the public-output
    one-table tags, refuses it, and `ZkOuterR.hTags` (`TagsOkZHJ`) takes it (`test_the_recorded_level_meets`)."""
    c = outer["tags_ok"]
    assert not c["ok"]
    assert c["why"].startswith(f"{STATEMENT}: hidden outputs true,"), c["why"]


# --- the F1 gate ----------------------------------------------------------------------------------------------------------

def _f1_file(dst: Path, line: str, root: Path = F1) -> str | None:
    """One line of `root`'s `prover.sha256`: its file under `dst` (linked, or decompressed from `.xz`), or its path if the
    bytes are not the listed ones."""
    digest, rel = line.split("  ", 1)
    src, out, h = root / rel, dst / rel, hashlib.sha256()
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        out.symlink_to(src)
        h.update(src.read_bytes())
    else:
        with lzma.open(src.with_name(src.name + ".xz")) as f, out.open("wb") as o:
            while b := f.read(1 << 24):
                h.update(b)
                o.write(b)
    return None if h.hexdigest() == digest else rel


@pytest.fixture(scope="module")
def f1_gate(once) -> Path:
    """The gate's files as the prover wrote them."""
    assert (F1 / "prover.sha256").is_file(), (
        f"{F1} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/f1-gate`")

    def make(g):
        lines = [(ln, root) for root in (F1, F1_TRANSCRIPT) for ln in (root / "prover.sha256").read_text().splitlines()]
        with cf.ThreadPoolExecutor(8) as ex:
            bad = [r for r in ex.map(lambda x: _f1_file(g, *x), lines) if r]
        assert not bad, f"not the prover's bytes: {bad}"

    return once("witness-f1-gate", make)


def _f1_inner(g: Path, public: Path, *extra: str) -> dict[str, dict]:
    """`flock-witness inner` on the gate's inner statement with the verifier's own partition object and program."""
    st = g / "stage"
    return _witness("inner", "--circuit", str(st / "circuit.txt"), "--public", str(public), "--partition",
                    str(st / "partition.json"), "--program", str(st / "program.json"), *extra)


def _unit(g: Path, part: str) -> str:
    with (g / "v" / part / "circuit.txt").open() as f:
        return f.readline().split()[1]


def _f1_inst(g: Path) -> tuple[dict, list, dict, dict, bytes]:
    """`stage/inst-2.bin` in `circuit.write`'s shared-row layout: its header, META's ports, every table's rows and salts
    (port-major), and what follows them, which is the verifier's file's body (every row's `b || c`, then the refs)."""
    raw = (g / "stage" / "inst-2.bin").read_bytes()
    at = raw.index(b"\n") + 1
    head = json.loads(raw[:at])
    with (g / "stage" / "circuit.txt").open() as f:
        f.readline()
        ports = C._meta_ports(json.loads(f.readline()[len("META "):]))
    n = head["shared_rows"]
    rows, salts = {}, {}
    for p in ports:
        k = 2 * n[p.name] * p.words
        rows[p.name] = np.frombuffer(raw[at:at + k], dtype="<u2").reshape(n[p.name], p.words)
        at += k
    for p in ports:
        salts[p.name] = [raw[at + C.SALT_BYTES * i:at + C.SALT_BYTES * (i + 1)] for i in range(n[p.name])]
        at += C.SALT_BYTES * n[p.name]
    return head, ports, rows, salts, raw[at:]


def _committed(head: dict, ports: list, rows: dict, salts: dict) -> tuple[dict, bytes]:
    """The header's `frame_v3` and every row's `b || c` (port-major), recomputed from the rows and salts."""
    fv, bcs = C.statement(ports, SimpleNamespace(name=head["set"]), *head["range"], rows, salts)
    return fv, b"".join(bc for p in ports for bc in bcs[p.name])


def _inputs(head: dict, ports: list, rows: dict, public: bytes, i: int) -> tuple:
    """Instance `i`'s rows `p0`, `p1` and `out`, through its refs (u32 LE, instance-major, after every row's `b || c`)."""
    names = [p.name for p in ports]
    at = C.BC_BYTES * sum(head["shared_rows"].values()) + 4 * len(names) * i
    refs = np.frombuffer(public[at:at + 4 * len(names)], dtype="<u4")
    return tuple(rows[nm][refs[names.index(nm)]] for nm in ("p0", "p1", "out"))


@functools.cache
def _unit_call():
    prog = Program(bind(_ROOT, FN=bind(G.GemmCoordinate, K=4096, **G.GEMM_HOPPER)))
    (call,) = [c for c in prog.circuit.root_calls() if not c.is_primitive]
    return prog.circuit, call


def _gemm(x, w) -> int:
    """The inner unit's Definition on one coordinate, by the IR oracle."""
    (y,) = evaluate_call(*_unit_call(), [int(v) for v in (*x, *w)])
    return y


def _f1_cheat(g: Path, dst: Path) -> Path:
    """The verifier's file of the gate's inner statement with instance 0's output row committed one bit off: the row's
    `b || c` under its own salt, and the `out` tree's root over it."""
    head, ports, rows, salts, public = _f1_inst(g)
    rows = dict(rows, out=rows["out"].copy())
    rows["out"][0, 0] ^= 1
    fv, bc = _committed(head, ports, rows, salts)
    f = dst / "pub-2.bin"
    f.write_bytes(json.dumps(dict(head, rows=False, frame_v3=fv), sort_keys=True).encode() + b"\n" + bc + public[len(bc):])
    return f


@pytest.fixture(scope="module")
def f1_inner(f1_gate, once) -> dict[str, dict]:
    _build()
    g = f1_gate
    return _kept(once, "witness-f1-inner", functools.partial(
        _f1_inner, g, g / "stage" / "pub-2.bin", "--record", str(g / "inner" / "s1" / "session" / "session.json")))


class _Runs(dict):
    """`f1`'s runs as futures: reading one waits for that run only."""

    def __getitem__(self, k):
        v = super().__getitem__(k)
        return v.result() if isinstance(v, cf.Future) else v


@pytest.fixture(scope="module")
def f1(f1_gate, f1_inner, once):
    """Every other `flock-witness` run the gate's tests read, at once: the inner statement at the cheater's public file, and
    each of V*'s sessions as `verity/flock-circuit` with the verifier's own V* inputs (the inner shape `accepts_stmt` prints,
    which is `rec-inner`'s; `--rec-part` its unit's index in the printed statements; `rec_bindings` of its partition
    object and program; the family's shared rows, the algebra parts' registered row ports). Each run is made once in the
    whole run (`once`), and a test waits only for the runs it reads."""
    g = f1_gate

    def inputs(d):
        (d / "inner.json").write_text(json.dumps(f1_inner["accepts_stmt"]["shape"]))
        (d / "draw.json").write_text(json.dumps({"law": "subset", "population": 2, "k": 2, "units": [0, 1]}))
        tops = [g / "v" / p for p in F1_PARTS if not p.startswith("alg/")]
        (d / "rows.json").write_text(json.dumps(S.shared_rows([g / "v" / p for p in F1_PARTS if p.startswith("alg/")], (),
                                                              tops)))
        _f1_cheat(g, d)
        for part in F1_PARTS:
            v = g / "v" / part
            (d / f"bindings-{part.replace('/', '-')}.json").write_text(json.dumps(RB.bindings(
                json.loads((v / "partition.json").read_text()), json.loads((v / "program.json").read_text()))))

    d = once("witness-f1-inputs", inputs)
    names = f1_inner["accepts_stmt"]["statements"]
    jobs = {"cheat": functools.partial(_f1_inner, g, d / "pub-2.bin")}
    for part in F1_PARTS:
        v, label = g / "v" / part, part.replace("/", "-")
        files = ["--circuit", str(v / "circuit.txt"), "--public", str(v / "pub.bin"), "--partition", str(v / "partition.json"),
                 "--program", str(v / "program.json"), "--registered", str(v / "registered.json")]
        sess = ["--session", str(g / "v-sess" / label)]
        jobs[part, STATEMENT] = functools.partial(
            _witness, "session", "--statement", STATEMENT, *files, "--rec-inner", str(d / "inner.json"), "--rec-part",
            str(names.index(_unit(g, part))), "--rec-bindings", str(d / f"bindings-{label}.json"), "--rec-rows",
            str(d / "rows.json"), *sess)
    ex = cf.ThreadPoolExecutor(len(jobs))
    yield _Runs({k: ex.submit(_kept, once, "witness-f1-" + (k if k == "cheat" else k[0].replace("/", "-")), fn)
                 for k, fn in jobs.items()}, **{"cheat-file": d / "pub-2.bin", "inputs": d})
    ex.shutdown(wait=False)


def test_f1_lean_builds_the_inner_statement_from_its_own_partition_object_and_program(f1_inner):
    """`AcceptsPublic obj prog st` (`acceptsPublic_of_check`): with the verifier's own partition object and program the
    inner statement sets up as one table whose units the verifier derives (`Q_word` v1), and `Zk.schedule` takes its m;
    s1, the gate's inner session, is a session of it."""
    c = f1_inner["accepts_stmt"]
    assert c["ok"], c.get("why")
    assert c["m"] == 27
    assert (c["partition_check"]["units"], c["partition_check"]["query"]) == ("derived", "Q_word v1")
    assert f1_inner["record_sigma"]["ok"], f1_inner["record_sigma"].get("why")


def test_f1_vstars_statements_are_outer_statements_at(f1_gate, f1_inner):
    """`outerStatementsAt st`: the inner shape `innerAt` reads (m, k_log, 2 + points·regions claims, regions; `rec-inner`'s),
    `outerParts` over it (the exe's copy, `outerPartsOf`, is it by `rfl`) and `RecVStar.statements` there, which are the
    units of the gate's ten staged statements, the last four RecCaps over levels 1 to 3's tops and level 0's 128 nodes,
    which `outerParts` counts (levels + groups + levels)."""
    c = f1_inner["accepts_stmt"]
    assert c["shape"] == {"format": "verity/rec-inner/v1", "m": 27, "k_log": 24, "claims": 8, "regions": 3}
    assert c["outer_parts"] == len(c["statements"]) == len(F1_PARTS)
    assert c["statements"] == [nm for nm, _ in F1_STATEMENTS] == [_unit(f1_gate, p) for p in F1_PARTS]


@pytest.mark.parametrize("part", F1_PARTS)
def test_f1_the_session_meets(f1, part):
    """V*'s statement `j` at its session: `checkStmt` (`AcceptsZKV`'s second conjunct, so the session's circuit file is
    Lean's own `RecVStar.circuitFile` under the verifier's bindings, byte for byte, and its registered reads pass
    `checkOwn`, `checkRows`, `checkAcc` and `checkAccReads`), `TagsOkZHJ` (`ZkOuterR.hTags`), `ScopeZR`, `VStmt.hscope`,
    and the full draw over its population (`RecordCustody`)."""
    w, j = f1[part, STATEMENT], F1_PARTS.index(part)
    assert w["accepts_zkv"]["part"] == j
    assert "check_stmt" not in w["accepts_zkv"], w["accepts_zkv"]["check_stmt"]
    for check in ("tags_ok_zhj", "scope_zr", "scope_ok", "full_draw"):
        assert w[check]["ok"], (check, w[check].get("why"))
    assert w["full_draw"]["population"] == F1_STATEMENTS[j][1]


@pytest.mark.skip(reason=PRF208)
@pytest.mark.parametrize("part", F1_PARTS)
def test_f1_the_session_is_accepted_as_the_verifiers_statement(f1, part):
    """`AcceptsZKR`, and so `AcceptsZKV`, accepts every gate session as `verity/flock-circuit`, the verifier of record's
    `--zk` identity with #1677's OS masks, which the gate's prover had."""
    w = f1[part, STATEMENT]
    for check in ("accepts_zkr", "accepts_zkv"):
        assert w[check]["ok"], (check, w[check].get("why"))


@pytest.mark.skip(reason=PRF208)
@pytest.mark.parametrize("part", ["L3", "cap"])
def test_f1_verify_refuses_the_session_only_for_its_undrawn_inner_session(built, f1_gate, f1, once, part):  # noqa: F811
    """`flock-verify verify --zk`, the verifier of record, passes level 3 (`rec-open/l16-h4`, 106 units, the smallest
    level) and RecCap (`rec-cap/n128`, `RecCoins.checkCap`) with the verifier's own V* inputs, the inner statement and
    session its coefficients are checked against, and the coin server's inner transcript (`inner/s1/public.json`), whose
    link points, level-3 tops and `root_B` the unit is held to (`RecVStar.checkTopRows`, `RecCoins.checkLink`), as
    `85-rec-reprice.sh`'s STEP=lean runs it, through every check and every refusal the verdict runs last (the form's, a
    draw-less session's, and a V* session's short of every unit, `RecSetup.fullDraw`) but the one after them: the gate's
    inner session ran before the coin server drew it (#2468), so its record holds no draw, and a session over it is refused
    (`RecSetup.innerDrawn`), with that reason alone, and not as a private circuit's. The checks above are the model's
    predicates; this is the shipped verdict, so a refusal added to it that an honest V* session meets fails here and nowhere
    else. About two minutes, beside the gate's witness runs."""
    g, d = f1_gate, f1["inputs"]
    v, st = g / "v" / part, g / "stage"
    names = [nm for nm, _ in F1_STATEMENTS]

    def run():
        # the gate's record predates `rec_live.record_bytes`, the format the verifier reads (`RecSetup.checkReads`)
        transcript = d / f"transcript-{part}.json"
        transcript.write_text(RL.record_bytes(json.loads((g / "inner" / "s1" / "public.json").read_text())))
        p = keyed.run([str(LEAN / ".lake" / "build" / "bin" / "flock-verify"), "verify", "--zk", "--statement", STATEMENT,
                       "--circuit", str(v / "circuit.txt"), "--public", str(v / "pub.bin"), "--partition",
                       str(v / "partition.json"), "--program", str(v / "program.json"), "--registered",
                       str(v / "registered.json"), "--rec-inner", str(d / "inner.json"), "--rec-part",
                       str(names.index(_unit(g, part))), "--rec-bindings", str(d / f"bindings-{part}.json"), "--rec-rows",
                       str(d / "rows.json"), "--rec-inner-circuit", str(st / "circuit.txt"), "--rec-inner-public",
                       str(st / "pub-2.bin"), "--rec-inner-partition", str(st / "partition.json"),
                       "--rec-inner-statement", "verity/flock-circuit+plain-leaves", "--rec-inner-statement-coins", "os",
                       "--rec-inner-draw", str(d / "draw.json"), "--rec-inner-session",
                       str(g / "inner" / "s1" / "session"), "--rec-inner-transcript", str(transcript), "--session",
                       str(g / "v-sess" / part)])
        return {"rc": p.returncode, "verdicts": [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]}

    r = _kept(once, f"verify-f1-{part}", run)
    assert r["rc"] == 1 and [(x["accepted"], x["why"], x["private_circuit"]) for x in r["verdicts"]] == [
        (False, RL.NO_INNER_DRAW, False)], r


def test_f1_the_sessions_fail_the_public_output_checks(f1):
    """As at rec-coins' level: `TagsOk` refuses V*'s hidden outputs and `ScopeZ` their shared-row public files, while
    `TagsOkZHJ` and `ScopeZR`, which `ZkOuterR` takes, pass (`test_f1_the_session_meets`)."""
    for part in F1_PARTS:
        w = f1[part, STATEMENT]
        assert w["tags_ok"]["why"].startswith(f"{STATEMENT}: hidden outputs true,"), (part, w["tags_ok"])
        assert w["scope_z"]["why"] == ("refused: a shared-row public file (header shared_rows): a statement form the soundness "
                                       "proof does not cover"), part


def test_f1_the_rate_is_below_one_at_k_2_21(f1):
    """`_hr`: `rateZ j k = 2^(level 0's logLen) / (e·k)` (`ZkLink.rateZ`) over each session's own plan, its statement's
    `Zk.schedule`, whose level 0's `logLen = logCols + logInvRate` (`Accounting.Level.logLen`) is `cols + rate`. The largest
    is L0's (m 33): 21, so at `k = 2^21` every recorded session's rate is at most `ρ = 1/e < 1`. `_hr` is over every
    session the prover may supply, so the Lean takes `k = 2^23` (`Witness.hr_inv_exp zr` is `_hr`, through
    `ZkHidden.hr_planZHJ_23`), the cap `Zk.schedule` puts on level 0's `logLen` at every `m` it accepts
    (`ZkBind.zkLogLenCap_23`), which the recorded sessions are under."""
    zkr = {p: f1[p, STATEMENT]["accepts_zkr"] for p in F1_PARTS}
    for p, c in zkr.items():
        assert c["level0"]["d"] == c["level0"]["cols"] + c["level0"]["rate"], p
    assert (zkr["L0"]["m"], zkr["L0"]["level0"]["d"]) == (33, 21)
    k = 2 ** max(c["level0"]["d"] for c in zkr.values())
    assert k == 2 ** 21 <= 2 ** 23
    assert max(2 ** c["level0"]["d"] / (math.e * k) for c in zkr.values()) == 1 / math.e < 1


def test_f1_the_honest_outputs_are_the_circuits(f1_gate):
    """The event's `¬ holds`, executably, at the honest s1: every output row is the inner unit's Definition on its
    instance's input rows (the IR oracle), and every row's `b || c` under its salt and the trees' roots are the verifier's
    file's, `pub-2.bin` byte for byte."""
    st = f1_gate / "stage"
    head, ports, rows, salts, public = _f1_inst(f1_gate)
    assert bind(G.GemmCoordinate, K=4096, **G.GEMM_HOPPER).id == json.loads((st / "rec.json").read_text())["definition"]
    fv, bc = _committed(head, ports, rows, salts)
    assert fv == head["frame_v3"]
    assert public[:len(bc)] == bc
    assert (st / "pub-2.bin").read_bytes() == json.dumps(dict(head, rows=False), sort_keys=True).encode() + b"\n" + public
    for i in range(head["instances"]):
        x, w, out = _inputs(head, ports, rows, public, i)
        assert int(out[0]) == _gemm(x, w) and not out[1:].any(), i


def test_f1_a_wrong_output_still_sets_up(f1_gate, f1_inner, f1):
    """The cheater: instance 0's output row one bit off, committed under its salt. Lean's verifier still builds the inner
    statement from its own partition object and program (`AcceptsPublic` holds), at another Σ, while the committed output
    is not the circuit's. This is the executable fact `¬ holds` is about, not that Lean's `holds` flips."""
    head, ports, rows, _, public = _f1_inst(f1_gate)
    c = f1["cheat"]["accepts_stmt"]
    assert c["ok"], c.get("why")
    assert c["sigma"] != f1_inner["accepts_stmt"]["sigma"]
    assert (c["m"], c["partition_check"]["units"]) == (27, "derived")
    roots = json.loads(f1["cheat-file"].read_bytes().split(b"\n", 1)[0])["frame_v3"]["roots"]
    assert {nm for nm, r in roots.items() if r != head["frame_v3"]["roots"][nm]} == {"out"}
    x, w, out = _inputs(head, ports, rows, public, 0)
    assert int(out[0]) == _gemm(x, w) != int(out[0]) ^ 1
