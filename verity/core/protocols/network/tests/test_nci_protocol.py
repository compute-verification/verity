"""The NCI check as a core protocol (`NciCheck.Protocol`, `nci serve`, `verity/nci/v1`) against `nci_reference`'s
`check_baseline` behind the protocol's decoders, through the difftest harness's `ProtocolPort`.

The reference decodes the protocol's formats (`NciCheck/Protocol.lean`'s module docstring), rebuilds the check of
record's instance and trusted input (the params' log with the run's own record, the commitment, appended, and that
record's digest the run) and runs `check_baseline`, which `NciCheck.BaselineDifftest` holds `NciCheck.Baseline.check` to.
The cases are θ's provenance difftest's declarations (`baseline_difftest_vectors.case`), split at the run's own record
into the protocol's params, commitment and evidence, then at most one change (`MUTATIONS`): the division made to fail
(`gamma` 0), the run's record repeated in the log, and malformed params, commitments and evidence. Onsite's difftest
(`verity/core/protocols/composition/tests/`) draws its registration's cases from `cases` and answers them with
`reference`."""

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

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/nci/v1"
PARAMS_KEYS = ["alpha", "gamma", "log", "parameters", "program"]
EVIDENCE_KEYS = ["owner", "provenance", "theta"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("baseline_difftest_vectors")
NR = vectors.reference

# --- the reference --------------------------------------------------------------------------------------------------------


def _exactly(j, keys: list[str], what: str) -> None:
    if not isinstance(j, dict) or sorted(j) != keys:
        raise Reject(f"the keys of {what} are not {', '.join(keys)}")


def params(j) -> dict:
    _exactly(j, PARAMS_KEYS, "the params")
    if not isinstance(j["program"], str):
        raise Reject("the params' program is not a string (the descriptor's text)")
    if not isinstance(j["log"], list):
        raise Reject("the params' log is not a list of registration records")
    return j


def commitment(b: bytes) -> tuple[dict, str]:
    record = P.canonical_json(b)
    if not isinstance(record, dict):
        raise Reject("the registration record is not an object")
    return record, NR.record_digest(record)


def evidence(b: bytes) -> dict:
    j = P.canonical_json(b)
    _exactly(j, EVIDENCE_KEYS, "the evidence")
    return j


def check(p: dict, c: tuple[dict, str], e: dict) -> tuple:
    record, run = c
    instance = {"alpha": p["alpha"], "gamma": p["gamma"], "owner": e["owner"], "program": p["program"],
                "provenance": e["provenance"], "theta": e["theta"]}
    trusted = {"log": [*p["log"], record], "params": p["parameters"], "run": run}
    out = NR.check_baseline(instance, trusted)
    if "error" in out:
        return "reject", out["error"]
    if out["complies"] is True:
        return "accept", {}, {"gates": out["gates"], "ius": out["ius"]}
    return "reject", "the division does not comply"


def reference(case) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: bytes.fromhex(c[1]))


# --- the cases ------------------------------------------------------------------------------------------------------------

#: Each change and the verdict and kind it aims at (`_kind`): the decoder that refuses it, the check of record's code, or
#: `comply` for a division that fails the policy.
EXPECT = {"honest": ("accept", "accept"), "refused": ("reject", "code"), "noncompliant": ("reject", "comply"),
          "log-repeat": ("reject", "trusted"), "alpha-den-0": ("reject", "params"), "params-key": ("reject", "params"),
          "program-type": ("reject", "params"), "log-type": ("reject", "params"),
          "commitment-type": ("reject", "commitment"), "commitment-noncanonical": ("reject", "commitment"),
          "evidence-key": ("reject", "evidence"), "evidence-noncanonical": ("reject", "evidence")}
MUTATIONS = list(EXPECT)
#: The first seeds of the provenance difftest's declarations, from which honest and refused cases are drawn.
SEEDS = 240


def split(seed: int) -> dict | None:
    """Declaration `seed` as the protocol's parts: params, the run's own record and the evidence, with the code the
    provenance difftest built it for (`None` when it isn't broken); `None` when its trusted input names no run of its
    log (a broken `trusted`, which the protocol, rebuilding the trusted input, can't carry)."""
    v = vectors.case(seed)
    inst, tr = v["case"]["instance"], v["case"]["trusted"]
    log = tr.get("log") if isinstance(tr, dict) else None
    if not isinstance(log, list) or not all(isinstance(r, dict) for r in log):
        return None
    digests = [NR.record_digest(r) for r in log]
    if tr.get("run") not in digests:
        return None
    k = digests.index(tr["run"])
    return {"params": {"alpha": inst["alpha"], "gamma": inst["gamma"], "log": log[:k], "parameters": tr["params"],
                       "program": inst["program"]},
            "record": log[k], "evidence": {key: inst.get(key) for key in EVIDENCE_KEYS}, "malformed": v["malformed"]}


@lru_cache(maxsize=None)
def seeds() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The seeds whose split complies, and those whose split the check of record refuses."""
    honest, refused = [], []
    for seed in range(SEEDS):
        s = split(seed)
        if s is None:
            continue
        out = check(s["params"], (s["record"], NR.record_digest(s["record"])), s["evidence"])
        if out[0] == "accept":
            honest.append(seed)
        elif out[1] != "the division does not comply":
            refused.append(seed)
    return tuple(honest), tuple(refused)


def case_of(seed: int, m: str, pick: int = 0) -> dict:
    """Declaration `seed`'s case with change `m`; `pick` chooses among a change's variants."""
    s = split(seed)
    p, record, e = dict(s["params"]), s["record"], dict(s["evidence"])
    c_bytes = D.canonical(record).encode()
    e_bytes = None
    if m == "noncompliant":
        p["gamma"] = 0
    elif m == "log-repeat":
        p["log"] = [*p["log"], record]
    elif m == "alpha-den-0":
        p["alpha"] = [1, 0]
    elif m == "params-key":
        if pick % 2:
            p["x"] = 1
        else:
            del p[PARAMS_KEYS[pick // 2 % len(PARAMS_KEYS)]]
    elif m == "program-type":
        p["program"] = [p["program"]]
    elif m == "log-type":
        p["log"] = {"records": p["log"]}
    elif m == "commitment-type":
        c_bytes = D.canonical([record]).encode()
    elif m == "commitment-noncanonical":
        c_bytes = json.dumps(record, indent=1).encode()
    elif m == "evidence-key":
        if pick % 2:
            e["x"] = 1
        else:
            del e[EVIDENCE_KEYS[pick // 2 % len(EVIDENCE_KEYS)]]
    elif m == "evidence-noncanonical":
        e_bytes = json.dumps(e, indent=1).encode()
    if e_bytes is None:
        e_bytes = D.canonical(e).encode()
    return {"commitment": c_bytes.hex(), "evidence": e_bytes.hex(), "params": p}


TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    """A declaration's case with one change of `only` (`honest` for none)."""
    m = draw(st.sampled_from(only))
    honest, refused = seeds()
    seed = draw(st.sampled_from(refused if m == "refused" else honest))
    case = case_of(seed, m, draw(st.integers(0, 9)))
    TAGS[D.canonical(case)] = m
    return case


def _kind(out: dict) -> str:
    """The path an outcome took: `accept`, the decoder that refused, the check's code, or `comply`."""
    if out["verdict"] == "accept":
        return "accept"
    reason = out["reason"]
    for prefix in ("params", "commitment", "evidence", "case"):
        if reason.startswith(prefix + ": "):
            return prefix
    if reason == "the division does not comply" or reason.startswith("the division does not comply:"):
        return "comply"
    return reason.split(":")[0]


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), out["verdict"], _kind(out))] += 1
    return out


PORT = D.ProtocolPort("nci", counted, cases(), package=LEAN)
server = D.server_fixture(PORT)


def test_each_change_takes_its_path():
    """Each change, on the first honest declarations (refused ones for `refused`), gets the verdict and the path it is
    aimed at in the reference, and an accepted case's profile is the check of record's report."""
    honest, refused = seeds()
    assert len(honest) >= 20 and len(refused) >= 20, (len(honest), len(refused))
    for m, (verdict, kind) in EXPECT.items():
        for seed in (refused if m == "refused" else honest)[:6]:
            for pick in range(4):
                out = reference(case_of(seed, m, pick))
                got = _kind(out)
                assert out["verdict"] == verdict, (m, seed, pick, out)
                if kind == "code":
                    assert got == split(seed)["malformed"] or got not in ("accept", "comply"), (m, seed, out)
                else:
                    assert got == kind, (m, seed, pick, out)
    s = split(honest[0])
    out = reference(case_of(honest[0], "honest"))
    inst = {**{k: s["params"][k] for k in ("alpha", "gamma", "program")}, **s["evidence"]}
    trusted = {"log": [*s["params"]["log"], s["record"]], "params": s["params"]["parameters"],
               "run": NR.record_digest(s["record"])}
    verdict = NR.check_baseline(inst, trusted)
    assert out["profile"] == {"gates": verdict["gates"], "ius": verdict["ius"]} and out["public"] == {}


def test_the_nci_protocol_agrees_with_the_reference(server):
    """Lean and the reference agree on every case, and every change takes its path at least once."""
    SEEN.clear()
    stats = D.run(PORT, examples=300, server=server)
    by_path = Counter()
    for (tag, _, kind), k in SEEN.items():
        by_path[(tag, kind)] += k
    missed = [m for m, (_, kind) in EXPECT.items() if kind != "code" and not by_path[(m, kind)]]
    assert not missed, (missed, by_path)
    assert stats.counts.get("accept", 0) > 10 and stats.counts.get("reject", 0) > 100, stats
