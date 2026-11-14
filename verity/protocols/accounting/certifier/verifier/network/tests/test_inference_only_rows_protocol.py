"""The inference-only policy's check in rows as a core protocol (`RowsProtocol.lean`, `inference-only-rows`,
`verity/inference-only/rows/v1`) against `nci_record_reference` behind the protocol's decoders.

The reference decodes the protocol's parts (`RowsProtocol.lean`'s module docstring), rebuilds the record check's
instance and trusted input (the params' log with the run's own record, the commitment, appended, and that record's
digest the run), refuses with `check_detail`'s refusal, holds `circuits` to the record's templates (`front`), reads
each template's rows of one unit from the blob its digest names (`unit_rows`, from the unit circuit's header), and
decides the rule with `block_rows` and the count with `chosen`. `unit_rows` reads a header where the Lean parses the
whole file, so the files the Lean refuses that have one are a fixed list (`REFUSED`), the reference's only refusal of
a file it does not compute.

Two kinds of case:

* Through `inference-only-rows serve` (`PORT`): the record difftest's cases (`record_difftest_vectors.case`), split at
  the run's own record into params, commitment and evidence, each template's blob a few bytes that are no circuit
  file, then at most one change (`EXPECT`); each ends in a refusal, `circuit-file` at the latest. Every circuit file
  is tens of megabytes, so only one random case in `SERVE_FILE_ONE_IN` reads one (`SERVE_FILE_CASES`): one honest
  case of two templates on a file of Flock's fixtures, at its limits and one past each.
* Through `inference-only-rows check PARAMS COMMITMENT EVIDENCE BLOB...` (`CHECK_CASES`): that honest case on two
  files of two values of a unit's rows: the accept at `Γ` and `α`, `size` and `nci` one past each, a template's file
  swapped for one of more rows, a file missing, and each file the Lean refuses.

The harness compares everything but a reject's reason; each test here also holds the reason's name, its text before
the first colon, on both sides."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import Reject
from verity_difftest import protocol as P

PROTOCOLS = Path(__file__).resolve().parents[5]
LEAN = PROTOCOLS / "policy" / "inference_only" / "verifier" / "nci" / "lean"
FIXTURES = PROTOCOLS / "proofs" / "flock" / "prover" / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/inference-only/rows/v1"
EXE = "inference-only-rows"
PARAMS_KEYS = ["alpha", "circuits", "gamma", "log", "pins", "program"]
ALPHA_FORM = "the params' alpha is not [num, den], two natural numbers, den not 0"
PINS_FORM = "the params' pins are not a list of SHA-512s in lower-case hex"
CIRCUITS_FORM = "the params' circuits are not a list of [template id, SHA-512 in lower-case hex] pairs"
CIRCUITS = "circuits: the params' circuits do not name the record's templates, each once, in the record's order"

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
ref = vectors.ref
NR = vectors.reference

# --- the circuit files ------------------------------------------------------------------------------------------------

#: The flat circuit files the Lean takes, under `FIXTURES`, with the rows of one unit `unit_rows` reads from each.
FILES = {"plain-tree": 192_544, "zk-sessions": 192_544, "forged-proofs": 275_329}
#: The circuit files the Lean refuses, each with the text its refusal holds. The first four have a unit circuit's header,
#: and differ from the taken ones in META's leaf scheme alone.
REFUSED = {"hidden-outputs/flat/rope-head": "META: leaf_scheme is not the pinned leaf-commitment scheme",
           "hidden-outputs/flat/gemm-coordinate": "META: leaf_scheme is not the pinned leaf-commitment scheme",
           "public-inputs": "META: leaf_scheme is not the pinned leaf-commitment scheme",
           "public-inputs-blocks": "META: leaf_scheme is not the pinned leaf-commitment scheme",
           "hidden-outputs/typed/rope-head": "circuit: not a flock-circuit file",
           "hidden-outputs/typed/gemm-coordinate": "circuit: not a flock-circuit file"}


def circuit_file(name: str) -> Path:
    path = FIXTURES / name / "stage" / "circuit.txt"
    assert path.is_file(), (f"{path} is missing: `uv run research data fetch-fixtures {path.parents[1]}` fetches it, "
                            "as the suite runner does before this suite")
    return path


@lru_cache(maxsize=None)
def refused_digests() -> frozenset[str]:
    """The SHA-512s of the files of `REFUSED`."""
    return frozenset(hashlib.sha512(circuit_file(n).read_bytes()).hexdigest() for n in REFUSED)


def file_rows(b: bytes, refused: frozenset[str]) -> int:
    """`RowsProtocol.rowsOf`: the rows of one unit of the file's template, or why it is not counted."""
    if hashlib.sha512(b).hexdigest() in refused:
        raise Reject("the file is one the Lean's parse refuses (REFUSED)")
    try:
        return ref.unit_rows(b)
    except ValueError as e:
        raise Reject(str(e)) from None


# --- the reference ----------------------------------------------------------------------------------------------------


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def params(j) -> dict:
    """`RowsProtocol.paramsOf`, in its order."""
    if not isinstance(j, dict) or sorted(j) != PARAMS_KEYS:
        raise Reject(f"the keys of the params are not {', '.join(PARAMS_KEYS)}")
    if not isinstance(j["program"], str):
        raise Reject("the params' program is not a string (the descriptor's text)")
    if not isinstance(j["log"], list):
        raise Reject("the params' log is not a list of registration records")
    if not _nat(j["gamma"]):
        raise Reject("the params' gamma is not a natural number")
    a = j["alpha"]
    if not (isinstance(a, list) and len(a) == 2 and all(map(_nat, a)) and a[1] != 0):
        raise Reject(ALPHA_FORM)
    if not (isinstance(j["pins"], list) and all(map(ref.is_digest, j["pins"]))):
        raise Reject(PINS_FORM)
    cs = j["circuits"]
    if not (isinstance(cs, list) and all(isinstance(e, list) and len(e) == 2 and isinstance(e[0], str)
                                         and ref.is_digest(e[1]) for e in cs)):
        raise Reject(CIRCUITS_FORM)
    return {**j, "num": a[0], "den": a[1], "circuits": [tuple(e) for e in cs]}


def commitment(b: bytes) -> tuple[dict, str]:
    """`Protocol.commitmentOf`: the run's registration record and its digest."""
    record = P.canonical_json(b)
    if not isinstance(record, dict):
        raise Reject("the registration record is not an object")
    return record, NR.record_digest(record)


def evidence(b: bytes) -> dict:
    """`RowsProtocol.evidenceOf`."""
    j = P.canonical_json(b)
    if not isinstance(j, dict) or sorted(j) not in (["record"], ["order", "record"]):
        raise Reject("the keys of the evidence are not record, or order and record")
    if not isinstance(j["record"], str):
        raise Reject("the evidence's record is not a string (the circuit record's text)")
    return j


def inputs(p: dict, c: tuple[dict, str], e: dict) -> tuple[dict, dict]:
    """`RowsProtocol.instanceOf` and `trustedOf`: what `Record` reads."""
    inst = {"alpha": [p["num"], p["den"]], "gamma": p["gamma"], "pins": p["pins"], "program": p["program"],
            "record": e["record"], **({"order": e["order"]} if "order" in e else {})}
    return inst, {"log": [*p["log"], c[0]], "params": [], "run": c[1]}


def check(p: dict, c: tuple[dict, str], e: dict, blobs: list[bytes], refused: frozenset[str] = frozenset()) -> tuple:
    """`RowsProtocol.checkBlobs`, in its order."""
    inst, tr = inputs(p, c, e)
    detail = ref.check_detail(inst, tr)
    if "error" in detail:
        return "reject", f"{detail['error']}: {detail['message']}" if "message" in detail else detail["error"]
    if [tid for tid, _ in p["circuits"]] != [t.tid for t in ref.front(inst, tr)[-1]]:
        return "reject", CIRCUITS
    files = {hashlib.sha512(b).hexdigest(): b for b in blobs}
    table = {}
    for tid, digest in p["circuits"]:
        if digest not in files:
            return "reject", f"circuit-file: template {tid}: no blob has the SHA-512 {digest}"
        try:
            table[tid] = file_rows(files[digest], refused)
        except Reject as why:
            return "reject", f"circuit-file: template {tid}: {why}"
    out = ref.block_rows(inst, tr, {"table": table, "gamma": p["gamma"], "alpha": [p["num"], p["den"]]})
    if out["failure"] == "size":
        return "reject", "size: a block has more than Γ rows"
    if out["failure"] == "nci":
        return "reject", "nci: a block reads more than α fresh bits a row"
    return "accept", {}, {"blocks": detail["blocks"], "chosen": ref.chosen(inst, tr)["chosen"],
                          "fresh": out["fresh"], "rows": out["rows"]}


def answer(case, blobs: list[bytes], refused: frozenset[str] = frozenset()) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence,
                    check=lambda p, c, e: check(p, c, e, blobs, refused), digest=lambda c: bytes.fromhex(c[1]))


def reference(case) -> dict:
    """`Core.BlobProtocol.case`: the case's blobs split off (absent: none), then the answer on them."""
    if isinstance(case, dict) and "blobs" in case:
        bs = case["blobs"]
        if not (isinstance(bs, list) and all(isinstance(b, str) for b in bs)):
            return P.outcome(ID, "reject", "case: the case's blobs are not a list of hex strings")
        try:
            blobs = [P.unhex(b) for b in bs]
        except Reject as e:
            return P.outcome(ID, "reject", f"case: {e}")
        return answer({k: v for k, v in case.items() if k != "blobs"}, blobs)
    return answer(case, [])


def name(out: dict) -> str:
    """What refused, the reason's text before the first colon; `accept` for an accept."""
    return "accept" if out["verdict"] == "accept" else out["reason"].split(":")[0]


# --- the cases through `serve` ----------------------------------------------------------------------------------------

#: Each change and the verdict and name it aims at; `code` is a refusal of `Record`'s.
EXPECT = {"honest": ("reject", "circuit-file"), "no-blob": ("reject", "circuit-file"),
          "blob-changed": ("reject", "circuit-file"), "refused": ("reject", "code"),
          "circuits-drop": ("reject", "circuits"), "circuits-extra": ("reject", "circuits"),
          "circuits-order": ("reject", "circuits"), "circuits-id": ("reject", "circuits"),
          "record-program": ("reject", "run-program"), "record-circuit": ("reject", "run-circuit"),
          "log-repeat": ("reject", "trusted"),
          "params-key": ("reject", "params"), "program-type": ("reject", "params"), "log-type": ("reject", "params"),
          "gamma-type": ("reject", "params"), "alpha-den-0": ("reject", "params"), "pins-type": ("reject", "params"),
          "circuits-type": ("reject", "params"),
          "commitment-type": ("reject", "commitment"), "commitment-noncanonical": ("reject", "commitment"),
          "evidence-key": ("reject", "evidence"), "evidence-noncanonical": ("reject", "evidence"),
          "record-type": ("reject", "evidence"), "blobs-type": ("reject", "case")}
#: The first seeds of the record difftest's random cases, from which honest and refused cases are taken.
SEEDS = 120


def no_file(tid: str) -> bytes:
    """Template `tid`'s blob in a `serve` case: a few bytes that are no circuit file."""
    return f"no circuit file: {tid}".encode()


@lru_cache(maxsize=None)
def split(seed: int) -> dict | None:
    """Case `seed` as the protocol's parts: params (with `circuits` naming each template's `no_file`), the run's own
    record and the evidence, with the stage the record difftest broke it at; `None` where it broke a field the
    protocol's own decoders read (`instance`, `params`) or its trusted input names no run of its log (the protocol
    rebuilds the trusted input). Shared: a caller copies before it changes a part."""
    v = vectors.case(seed)
    if v["broken"] in ("instance", "params"):
        return None
    inst, tr = v["case"]["instance"], v["case"]["trusted"]
    log = tr.get("log") if isinstance(tr, dict) else None
    if not isinstance(log, list) or not all(isinstance(r, dict) for r in log):
        return None
    digests = [NR.record_digest(r) for r in log]
    if tr.get("run") not in digests:
        return None
    k = digests.index(tr["run"])
    tids = [ref.codec.descriptor_id(fn) for fn, _ in v["circuit"].templates]
    return {"params": {"alpha": inst["alpha"], "gamma": inst["gamma"], "log": log[:k], "pins": inst["pins"],
                       "program": inst["program"],
                       "circuits": [[t, hashlib.sha512(no_file(t)).hexdigest()] for t in tids]},
            "record": log[k], "evidence": {"record": inst["record"], **({"order": inst["order"]} if "order" in inst else {})},
            "tids": tids, "broken": v["broken"]}


@lru_cache(maxsize=None)
def seeds() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The seeds whose split `Record` takes and whose `circuits` name its templates, and those `Record` refuses."""
    honest, refused = [], []
    for seed in range(SEEDS):
        s = split(seed)
        if s is None:
            continue
        out = name(reference(case_of(seed, "honest")))
        if out == "circuit-file":
            honest.append(seed)
        elif out not in ("circuits", "accept", "size", "nci"):
            refused.append(seed)
    return tuple(honest), tuple(refused)


def case_of(seed: int, m: str, pick: int = 0) -> dict:
    """Case `seed` with change `m`; `pick` chooses among a change's variants."""
    s = split(seed)
    p, record, e = json.loads(json.dumps(s["params"])), s["record"], dict(s["evidence"])
    blobs = [no_file(t) for t in s["tids"]]
    c_bytes = e_bytes = None
    cs = p["circuits"]
    if m == "no-blob":
        blobs = blobs[:pick % len(blobs)]
    elif m == "blob-changed":
        blobs[pick % len(blobs)] += b"."
    elif m == "circuits-drop":
        del cs[pick % len(cs)]
    elif m == "circuits-extra":
        cs.insert(pick % (len(cs) + 1), list(cs[pick % len(cs)]))
    elif m == "circuits-order":
        if len({t for t, _ in cs}) > 1:
            cs.reverse()
        else:
            cs[0][0] += "x"
    elif m == "circuits-id":
        cs[pick % len(cs)][0] = f"Other{pick}_v1"
    elif m == "record-program":
        c_bytes = D.canonical({**record, "program": f"{pick:02x}" * 64}).encode()
    elif m == "record-circuit":
        c_bytes = D.canonical({**record, "window": {**record["window"], "circuit": f"{pick:02x}" * 64}}).encode()
    elif m == "log-repeat":
        p["log"] = [*p["log"], record]
    elif m == "params-key":
        if pick % 2:
            p["x"] = 1
        else:
            del p[PARAMS_KEYS[pick // 2 % len(PARAMS_KEYS)]]
    elif m == "program-type":
        p["program"] = [p["program"]]
    elif m == "log-type":
        p["log"] = {"records": p["log"]}
    elif m == "gamma-type":
        p["gamma"] = [str(p["gamma"]), -1, None][pick % 3]
    elif m == "alpha-den-0":
        p["alpha"] = [[1, 0], [1], [1, 2, 3], [-1, 1]][pick % 4]
    elif m == "pins-type":
        p["pins"] = [[*p["pins"], "AB" * 64], "pins", [*p["pins"], 1]][pick % 3]
    elif m == "circuits-type":
        cs[pick % len(cs)] = [[cs[0][0]], [cs[0][0], cs[0][1].upper()], [1, cs[0][1]], cs[0][0]][pick % 4]
    elif m == "commitment-type":
        c_bytes = D.canonical([record]).encode()
    elif m == "commitment-noncanonical":
        c_bytes = json.dumps(record, indent=1).encode()
    elif m == "evidence-key":
        if pick % 2:
            e["x"] = 1
        else:
            del e["record"]
    elif m == "evidence-noncanonical":
        e_bytes = json.dumps(e, indent=1).encode()
    elif m == "record-type":
        e["record"] = json.loads(e["record"])
    case = {"commitment": (c_bytes or D.canonical(record).encode()).hex(),
            "evidence": (e_bytes or D.canonical(e).encode()).hex(), "params": p, "blobs": [b.hex() for b in blobs]}
    if m == "blobs-type":
        case["blobs"] = [["zz"], ["abc"], [1], "00"][pick % 4]
    return case


@st.composite
def made(draw, m: str) -> dict:
    """A case with change `m` (`honest` for none), on an honest split (a refused one for `refused`)."""
    honest, refused = seeds()
    return case_of(draw(st.sampled_from(refused if m == "refused" else honest)), m, draw(st.integers(0, 9)))


#: The honest case on circuit files a random `serve` run takes now and then (`SERVE_FILE_ONE_IN` of its cases), each
#: with the verdict it aims at: both templates on one taken file of 56 MB, at its limits and one past each.
SERVE_FILES = ("plain-tree", "plain-tree")
SERVE_FILE_CASES = {"file-accept": "accept", "file-size": "size", "file-nci": "nci"}
SERVE_FILE_ONE_IN = 40
#: what `limits` names: `at`, `Γ` and `α` exactly what a block holds; `rows`, `Γ` one under it; `fresh`, `α` one
#: fresh bit under it
LIMITS = {"accept": "at", "size": "rows", "nci": "fresh"}

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


def tag_of(case) -> str:
    """A case's key in `TAGS`: a digest of its canonical text, which is over 100 MB for a case on a circuit file."""
    return hashlib.sha256(D.canonical(case).encode()).hexdigest()


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(EXPECT)) -> dict:
    """A case with one change of `only`, or one in `SERVE_FILE_ONE_IN` a case of `SERVE_FILE_CASES`, tagged with it
    for `counted`."""
    if draw(st.integers(0, SERVE_FILE_ONE_IN - 1)) == SERVE_FILE_ONE_IN // 2:
        m = draw(st.sampled_from(tuple(SERVE_FILE_CASES)))
        case = serve_file_case(m)
    else:
        m = draw(st.sampled_from(only))
        case = draw(made(m))
    TAGS[tag_of(case)] = m
    return case


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(tag_of(case), "?"), name(out))] += 1
    return out


PORT = D.ProtocolPort("inference-only-rows", counted, cases(), package=LEAN, of_record="lean")
server = D.server_fixture(PORT)


def test_each_change_takes_its_path_on_both_sides(server):
    """Each change, on the first honest splits (refused ones for `refused`), gets the verdict and the refusal it aims
    at, by name, from the reference and from the Lean, and the two agree on all but the reason's text after its name."""
    honest, refused = seeds()
    assert len(honest) >= 20 and len(refused) >= 20, (len(honest), len(refused))
    for m, (verdict, kind) in EXPECT.items():
        for seed in (refused if m == "refused" else honest)[:4]:
            for pick in range(3):
                case = case_of(seed, m, pick)
                py = reference(case)
                lean = json.loads(PORT.lean(server, case)[1])
                assert py["verdict"] == verdict, (m, seed, pick, py)
                assert name(lean) == name(py), (m, seed, pick, lean["reason"], py["reason"])
                assert {k: v for k, v in lean.items() if k != "reason"} == {k: v for k, v in py.items() if k != "reason"}
                if kind == "code":
                    assert name(py) == split(seed)["broken"], (m, seed, py)
                else:
                    assert name(py) == kind, (m, seed, pick, py)


@pytest.mark.slow
def test_the_rows_protocol_agrees_with_the_reference(server):
    """Lean and the reference agree on every case, and every change and every case of `SERVE_FILE_CASES` takes its
    path at least once. Slow: one case in `SERVE_FILE_ONE_IN` reads a real circuit file of tens of megabytes, its hex
    a line of over 100 MB, a few seconds a case on each side. What it catches that the name-by-name test above cannot:
    an accept and its profile, and `size` and `nci`, the rule in rows, decided on a real file's rows through `serve`
    beside the reference, on random record cases rather than the first honest splits."""
    SEEN.clear()
    stats = D.run(PORT, examples=300, server=server)
    aims = {**{m: kind for m, (_, kind) in EXPECT.items() if kind != "code"}, **SERVE_FILE_CASES}
    missed = [m for m, kind in aims.items() if not SEEN[(m, kind)]]
    assert not missed, (missed, SEEN)
    assert stats.counts.get("reject", 0) > 200, stats


# --- the cases on circuit files ---------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def file_seed() -> int:
    """The first honest split of two templates and more than one block, each template with units in a block, whose
    blocks read fresh bits."""
    for seed in seeds()[0]:
        s = split(seed)
        if len(set(s["tids"])) != 2:
            continue
        p, c, e = params(s["params"]), (s["record"], NR.record_digest(s["record"])), s["evidence"]
        inst, tr = inputs(p, c, e)
        detail = ref.check_detail(inst, tr)
        if detail["fresh"] > 0 and detail["blocks"] > 1 and all(t.units // detail["blocks"] > 0
                                                                for t in ref.front(inst, tr)[-1]):
            return seed
    raise AssertionError("no honest split of two templates and two blocks whose blocks read fresh bits")


@lru_cache(maxsize=None)
def file_bytes(n: str) -> bytes:
    return circuit_file(n).read_bytes()


@lru_cache(maxsize=None)
def file_hex(n: str) -> str:
    return file_bytes(n).hex()


def digest_of(n: str) -> str:
    return hashlib.sha512(file_bytes(n)).hexdigest()


@lru_cache(maxsize=None)
def block_on(files: tuple[str, str]) -> tuple[int, int]:
    """A block's rows and fresh bits on the honest case, its templates on `files`, as `FILES` counts them."""
    s = split(file_seed())
    p = params({**s["params"], "circuits": [[t, digest_of(n)] for t, n in zip(s["tids"], files, strict=True)]})
    inst, tr = inputs(p, (s["record"], NR.record_digest(s["record"])), s["evidence"])
    out = ref.block_rows(inst, tr, {"table": {t: FILES[n] for t, n in zip(s["tids"], files, strict=True)},
                                    "gamma": 0, "alpha": [0, 1]})
    return out["rows"], out["fresh"]


def file_case(names: tuple[str | None, ...], limits: str, at: tuple[str, str]) -> dict:
    """The honest case with each template's `circuits` entry naming the file `names` gives (`None`: `at`'s), and
    `limits` (`LIMITS`) of a block on the files `at`."""
    s = split(file_seed())
    rows, fresh = block_on(at)
    gamma, alpha = {"at": (rows, [fresh, rows]), "rows": (rows - 1, [fresh, rows]),
                    "fresh": (rows, [fresh - 1, rows])}[limits]
    named = [n or a for n, a in zip(names, at, strict=True)]
    p = {**s["params"], "gamma": gamma, "alpha": alpha,
         "circuits": [[t, digest_of(n)] for t, n in zip(s["tids"], named, strict=True)]}
    return {"commitment": D.canonical(s["record"]).encode().hex(), "evidence": D.canonical(s["evidence"]).encode().hex(),
            "params": p}


def serve_file_case(m: str) -> dict:
    """The `serve` case of `SERVE_FILE_CASES`' `m`, its one blob the file of `SERVE_FILES`."""
    return {**file_case(SERVE_FILES, LIMITS[SERVE_FILE_CASES[m]], SERVE_FILES), "blobs": [file_hex(SERVE_FILES[0])]}


#: Each case through the executable's `check`: the file each template's `circuits` entry names (`None`: the file the
#: honest case names, its blob left out), the limits (`LIMITS`) of a block on the honest case's files (`CHECK_AT`), and
#: the verdict and name the reference gives.
CHECK_AT = ("plain-tree", "forged-proofs")
CHECK_CASES = {"accept": (CHECK_AT, "at", "accept"),
               "zk-sessions": (("zk-sessions", "forged-proofs"), "at", "accept"),
               "size": (CHECK_AT, "rows", "size"),
               "nci": (CHECK_AT, "fresh", "nci"),
               "more-rows": (("forged-proofs", "forged-proofs"), "at", "size"),
               "no-blob": (("plain-tree", None), "at", "circuit-file"),
               **{f"refused:{n}": ((n, None), "at", "circuit-file") for n in REFUSED}}


def test_unit_rows_reads_each_files_unit():
    """`unit_rows` on each taken file is the rows of one unit `FILES` pins, two values of it; a typed file is refused."""
    assert {n: ref.unit_rows(file_bytes(n)) for n in FILES} == FILES
    for n in REFUSED:
        if "typed" in n:
            with pytest.raises(ValueError):
                ref.unit_rows(file_bytes(n))


@pytest.mark.slow
@pytest.mark.parametrize("which", list(CHECK_CASES))
def test_the_rows_protocol_on_circuit_files(which, tmp_path):
    """The executable's `check` on the honest case's parts and its circuit files, against the reference: the same
    outcome but for the reason's text after its name, the name `CHECK_CASES` gives, and for a file the Lean refuses,
    the refusal `REFUSED` names. Slow: each run parses the circuit files it reads whole, one to three seconds a file."""
    names, limits, want = CHECK_CASES[which]
    case = file_case(names, limits, CHECK_AT)
    given = [n for n in dict.fromkeys(names) if n is not None]
    exe = D.build(LEAN, EXE)
    parts = []
    for k, data in (("params", D.canonical(case["params"]).encode()), ("commitment", bytes.fromhex(case["commitment"])),
                    ("evidence", bytes.fromhex(case["evidence"]))):
        (tmp_path / k).write_bytes(data)
        parts.append(str(tmp_path / k))
    run = subprocess.run([str(exe), "check", *parts, *(str(circuit_file(n)) for n in given)], capture_output=True,
                         text=True)
    assert run.returncode in (0, 1), (run.returncode, run.stderr[-2000:])
    lean = json.loads(run.stdout)
    py = answer(case, [file_bytes(n) for n in given], refused_digests())
    assert (name(lean), name(py)) == (want, want), (which, lean["reason"], py["reason"])
    assert {k: v for k, v in lean.items() if k != "reason"} == {k: v for k, v in py.items() if k != "reason"}, which
    assert run.returncode == (0 if want == "accept" else 1), which
    if which.startswith("refused:"):
        assert REFUSED[which.split(":", 1)[1]] in lean["reason"], lean["reason"]
    if want == "accept":
        assert [lean["profile"][k] for k in ("rows", "fresh")] == list(block_on(CHECK_AT)), lean
