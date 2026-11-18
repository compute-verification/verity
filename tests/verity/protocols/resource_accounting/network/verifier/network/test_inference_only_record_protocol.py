"""The inference-only policy's check of a circuit record as a core protocol (`RecordProtocol.lean`,
`inference-only-record`, `verity/inference-only/record/v1`) against `inference_only_record_reference` behind the
protocol's decoders.

The reference decodes the protocol's parts (`RecordProtocol.lean`'s module docstring), rebuilds the record check's
instance and trusted input (the params' log with the run's own record, the commitment, appended, and that record's
digest the run), refuses with `check_detail`'s refusal, and decides the rule on its summary, the count with `chosen`.
The cases are the record difftest's (`record_difftest_vectors.case`), split at the run's own record into params,
commitment and evidence, with `Γ` and `α` at what a block holds, one past either, or one change (`EXPECT`), through
`inference-only-record serve`. The harness compares everything but a reject's reason; each test here also holds the
reason's name, its text before the first colon, on both sides."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import Reject
from verity_difftest import protocol as P

PROTOCOLS = (Path(__file__).resolve().parents[7] / "verity" / "protocols")
LEAN = PROTOCOLS / "workload_classification" / "training_detection" / "inference_only" / "verifier" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/inference-only/record/v1"
PARAMS_KEYS = ["alpha", "gamma", "log", "pins", "program"]
ALPHA_FORM = "the params' alpha is not [num, den], two natural numbers, den not 0"
PINS_FORM = "the params' pins are not a list of SHA-512s in lower-case hex"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
ref = vectors.ref
NR = vectors.reference

# --- the reference ----------------------------------------------------------------------------------------------------


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def params(j) -> dict:
    """`RecordProtocol.paramsOf`, in its order."""
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
    return {**j, "num": a[0], "den": a[1]}


def commitment(b: bytes) -> tuple[dict, str]:
    """`Protocol.commitmentOf`: the run's registration record and its digest."""
    record = P.canonical_json(b)
    if not isinstance(record, dict):
        raise Reject("the registration record is not an object")
    return record, NR.record_digest(record)


def evidence(b: bytes) -> dict:
    """`RecordProtocol.evidenceOf`."""
    j = P.canonical_json(b)
    if not isinstance(j, dict) or sorted(j) not in (["record"], ["order", "record"]):
        raise Reject("the keys of the evidence are not record, or order and record")
    if not isinstance(j["record"], str):
        raise Reject("the evidence's record is not a string (the circuit record's text)")
    return j


def inputs(p: dict, c: tuple[dict, str], e: dict) -> tuple[dict, dict]:
    """`RecordProtocol.instanceOf` and `trustedOf`: what `Record` reads."""
    inst = {"alpha": [p["num"], p["den"]], "gamma": p["gamma"], "pins": p["pins"], "program": p["program"],
            "record": e["record"], **({"order": e["order"]} if "order" in e else {})}
    return inst, {"log": [*p["log"], c[0]], "params": [], "run": c[1]}


def check(p: dict, c: tuple[dict, str], e: dict) -> tuple:
    """`RecordProtocol.check`, in its order."""
    inst, tr = inputs(p, c, e)
    detail = ref.check_detail(inst, tr)
    if "error" in detail:
        return "reject", f"{detail['error']}: {detail['message']}" if "message" in detail else detail["error"]
    if detail["failure"] == "size":
        return "reject", "size: a block has more than Γ gates"
    if detail["failure"] == "nci":
        return "reject", "inference_only: a block reads more than α fresh bits a gate"
    return "accept", {}, {"blocks": detail["blocks"], "chosen": ref.chosen(inst, tr)["chosen"],
                          "fresh": detail["fresh"], "gates": detail["gates"]}


def reference(case) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: bytes.fromhex(c[1]))


def name(out: dict) -> str:
    """What refused, the reason's text before the first colon; `accept` for an accept."""
    return "accept" if out["verdict"] == "accept" else out["reason"].split(":")[0]


# --- the cases --------------------------------------------------------------------------------------------------------

#: Each change and the verdict and name it aims at; `code` is a refusal of `Record`'s.
EXPECT = {"accept": ("accept", "accept"), "size": ("reject", "size"), "inference_only": ("reject", "inference_only"),
          "refused": ("reject", "code"),
          "record-program": ("reject", "run-program"), "record-circuit": ("reject", "run-circuit"),
          "log-repeat": ("reject", "trusted"),
          "params-key": ("reject", "params"), "program-type": ("reject", "params"), "log-type": ("reject", "params"),
          "gamma-type": ("reject", "params"), "alpha-den-0": ("reject", "params"), "pins-type": ("reject", "params"),
          "commitment-type": ("reject", "commitment"), "commitment-noncanonical": ("reject", "commitment"),
          "evidence-key": ("reject", "evidence"), "evidence-noncanonical": ("reject", "evidence"),
          "record-type": ("reject", "evidence")}
#: The first seeds of the record difftest's random cases, from which honest and refused cases are taken.
SEEDS = 120


@lru_cache(maxsize=None)
def split(seed: int) -> dict | None:
    """Case `seed` as the protocol's parts, with the stage the record difftest broke it at and, where `Record` reads
    it, a block's gates and fresh bits; `None` where it broke a field the protocol's own decoders read (`instance`,
    `params`) or its trusted input names no run of its log (the protocol rebuilds the trusted input). Shared: a caller
    copies before it changes a part."""
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
    detail = ref.check_detail(inst, {"log": log[:k + 1], "params": [], "run": tr["run"]})
    return {"params": {"alpha": inst["alpha"], "gamma": inst["gamma"], "log": log[:k], "pins": inst["pins"],
                       "program": inst["program"]},
            "record": log[k], "evidence": {"record": inst["record"], **({"order": inst["order"]} if "order" in inst else {})},
            "broken": v["broken"], "block": None if "error" in detail else (detail["gates"], detail["fresh"])}


@lru_cache(maxsize=None)
def seeds() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The seeds whose split `Record` reads with a block of gates and fresh bits, and those `Record` refuses."""
    honest, refused = [], []
    for seed in range(SEEDS):
        s = split(seed)
        if s is None:
            continue
        if s["block"] is None:
            refused.append(seed)
        elif min(s["block"]) > 0:
            honest.append(seed)
    return tuple(honest), tuple(refused)


def case_of(seed: int, m: str, pick: int = 0) -> dict:
    """Case `seed` with change `m`; `pick` chooses among a change's variants. On a read split, `Γ` and `α` are what a
    block holds, but one under it for `size` (`Γ`) and `inference_only` (`α`)."""
    s = split(seed)
    p, record, e = json.loads(json.dumps(s["params"])), s["record"], dict(s["evidence"])
    if s["block"] is not None:
        gates, fresh = s["block"]
        p["gamma"] = gates - (m == "size")
        p["alpha"] = [fresh - (m == "inference_only"), gates]
    c_bytes = e_bytes = None
    if m == "record-program":
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
    return {"commitment": (c_bytes or D.canonical(record).encode()).hex(),
            "evidence": (e_bytes or D.canonical(e).encode()).hex(), "params": p}


TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


@st.composite
def cases(draw) -> dict:
    """A case with one change of `EXPECT`, on an honest split (a refused one for `refused`), tagged with it for
    `counted`."""
    m = draw(st.sampled_from(tuple(EXPECT)))
    honest, refused = seeds()
    case = case_of(draw(st.sampled_from(refused if m == "refused" else honest)), m, draw(st.integers(0, 9)))
    TAGS[D.canonical(case)] = m
    return case


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), name(out))] += 1
    return out


PORT = D.ProtocolPort("inference-only-record", counted, cases(), package=LEAN, of_record="lean")
server = D.server_fixture(PORT)


def test_each_change_takes_its_path_on_both_sides(server):
    """Each change, on the first honest splits (refused ones for `refused`), gets the verdict and the refusal it aims
    at, by name, from the reference and from the Lean, and the two agree on all but the reason's text after its name:
    an accept and its profile at a block's own `Γ` and `α`, and `size` and `inference_only` one under either."""
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
def test_the_record_protocol_agrees_with_the_reference(server):
    """Lean and the reference agree on every case, and every change takes its path at least once. Slow: 300 random
    cases through `serve`. What it catches that the name-by-name test above cannot: the two sides on random record
    cases rather than the first honest splits."""
    SEEN.clear()
    stats = D.run(PORT, examples=300, server=server)
    missed = [m for m, (_, kind) in EXPECT.items() if kind != "code" and not SEEN[(m, kind)]]
    assert not missed, (missed, SEEN)
    assert stats.counts.get("accept", 0) > 0 and stats.counts.get("reject", 0) > 100, stats
