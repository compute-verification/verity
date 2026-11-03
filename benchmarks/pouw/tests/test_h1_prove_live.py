"""h1_prove's live units (`h1_prove/live.py`), without a prover or Lean: on a served window's dump (`pouw_h1_live.dump`'s
form) the verifier's files are the live Program's, every opened row opens its registered root, each round's staged
statement is the verifier's own public file byte for byte with ok 1 at the served rows, and a prover that claims one row
more stages ok 0; a tampered opening is refused. The live Program is credit's, `PouwLiveWindow_v2`, each unit reading the
public max_tokens as its caps."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.circuit import live as L
from verity.core.protocols.pouw.circuit import live_reads as LR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

H1 = Path(__file__).resolve().parents[1] / "h1_prove"
spec = importlib.util.spec_from_file_location("h1_prove_live", H1 / "live.py")
LV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LV)

T, S, PREFIX = 4, 2, "h1/"
PROGRAM, WINDOW, SESSION = "ab" * 64, "cd" * 64, "session-1"
#: request 0 with a 2-token prompt and request 1 with a 1-token one: round 0's prefill serves 3 rows, round 1's decode 2
HEADERS = [(0, 0, 0, 3, 2), (1, 0, 2, 3, 1)]
#: their public max_tokens, the caps their headers carry
MAX_TOKENS = [3, 3]
SLOTS = [[1, 1, 2, 0], [1, 2, 0, 0]]


def salts(n: int) -> list[bytes]:
    return [os.urandom(LV.U.SALT_BYTES) for _ in range(n)]


def dump(out: Path) -> dict:
    """A served window's live dump, as `pouw_h1_live.dump` writes it."""
    heads = HEADERS + [(0,) * LR.HEADER] * (T - len(HEADERS))
    req = G.Registered.commit(PROGRAM, LR.table(PREFIX, LR.REQUESTS), LR.header_words(heads), salts(T), G.WORD_BITS,
                              LR.REQUEST_SCHEMA, session=SESSION)
    slots = [G.Registered.commit(PROGRAM, LR.table(PREFIX, LR.SLOTS, s), LR.slot_words(SLOTS[s]), salts(T), G.WORD_BITS,
                                 G.SCHEMA_V2, window=WINDOW) for s in range(S)]
    served = [sum(1 for x in r if x) for r in SLOTS]
    pub = {"T": T, "S": S, "prefix": PREFIX, "program": PROGRAM, "window": WINDOW, "session": SESSION,
           "roots": {r.value: r.root.hex() for r in (req, *slots)}, "served": served, "max_tokens": MAX_TOKENS}
    cap = LR.caps_table(PROGRAM, PREFIX, WINDOW, T, MAX_TOKENS)[2]
    out.mkdir(parents=True)
    (out / "window.json").write_text(json.dumps(pub))
    for s, m in enumerate(served):
        reads = {f"slot_{i}": (slots[s], i) for i in range(T)} | {f"req_{i}": (req, i) for i in range(T)}
        reads |= {f"cap_{i}": (cap, i) for i in range(T)}
        reads |= {"m": (LR.m_table(PROGRAM, PREFIX, WINDOW, s, m)[2], 0), "s": (LR.round_table(PROGRAM, PREFIX, WINDOW, s)[2], 0),
                  "o0": (LR.ok_table(PROGRAM, PREFIX, WINDOW, S)[2], s)}
        z = {"units": np.array([s])}
        for p, (reg, i) in reads.items():
            z |= {f"w_{p}": np.asarray([reg.rows[i]], np.uint16), f"s_{p}": np.frombuffer(reg.salts[i], np.uint8).reshape(1, -1),
                  f"v_{p}": np.array([reg.value]), f"r_{p}": np.array([i]), f"p_{p}": np.array([reg.path(i).hex()])}
        (out / str(s)).mkdir()
        np.savez(out / str(s) / "rows.npz", **z)
        (out / str(s) / "opening.json").write_text(json.dumps({p: {"value": reg.value, "index": i, "bc": reg.bcs[i].hex(),
                                                                   "path": reg.path(i).hex()} for p, (reg, i) in reads.items()}))
    return pub


def test_the_dump_is_live_ok_at_the_served_rows_only():
    heads = HEADERS + [(0,) * LR.HEADER] * (T - len(HEADERS))
    assert [L.live_ok(SLOTS[s], heads, m, s) for s, m in enumerate((3, 2))] == [1, 1]
    assert [L.live_ok(SLOTS[s], heads, m, s) for s, m in enumerate((4, 3))] == [0, 0]
    assert L.live_ok(SLOTS[0], heads, 3, 1) == 0
    caps = LR.caps(MAX_TOKENS, T)
    assert [L.live_ok_v2(SLOTS[s], heads, m, s, caps) for s, m in enumerate((3, 2))] == [1, 1]
    assert L.live_ok_v2(SLOTS[1], heads, 2, 1, LR.caps([3, 2], T)) == 0
    assert list(CC.OK_ROW) == LR.ok_table(PROGRAM, PREFIX, WINDOW, S)[0][0]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("live")
    pub = dump(base / "dump")
    rc = LV.main(["prepare", "--live", str(base / "dump"), "--work", str(base / "work")])
    return base, pub, rc


def test_prepare_stages_every_round_as_the_verifiers_own_file(prepared):
    """Each round's statement, staged on the prover's rows and bound to the live Program, is the verifier's own public file
    (from the opened ``b || c`` and its own ``m@<s>``, ``round@<s>`` and ``live-ok`` rows) with ok 1 at m_s; ``m-claimed`` stages ok 0 at
    m_0 + 1; the plan holds every honest round, ``m-other``, ``m-claimed`` and ``other-round``."""
    base, pub, rc = prepared
    work = base / "work"
    record = json.loads((work / "record.json").read_text())
    assert rc == 0 and record["problems"] == []
    assert {t: (r["round"], r["m"], r["staged_ok"]) for t, r in record["rounds"].items()} == {
        "honest-0": (0, 3, 1), "honest-1": (1, 2, 1), "m-claimed": (0, 4, 0)}
    for s in range(S):
        assert (work / "st" / f"honest-{s}" / "pub.bin").read_bytes() == (work / "verifier" / f"honest-{s}.pub.bin").read_bytes()
    plan = [x.split("\t") for x in (work / "plan.tsv").read_text().splitlines()]
    assert [x[1] for x in plan if x[0] == "S"] == ["honest-0", "m-claimed", "honest-1"]
    assert {x[1]: x[-1] for x in plan if x[0] == "C"} == {"honest-0": "ACCEPT", "m-other": "REJECT", "m-claimed": "REJECT",
                                                          "honest-1": "ACCEPT", "other-round": "REJECT"}
    reg = json.loads((work / "verifier" / "m-other.registered.json").read_text())
    assert reg["ports"]["m"]["known"]["rows"] == [LR.slot_words([4])[0]]
    assert json.loads((work / "verifier" / "honest-1.registered.json").read_text())["ports"]["s"]["known"]["rows"] == [LR.slot_words([1])[0]]
    assert (work / "verifier" / "m-other.pub.bin").read_bytes() != (work / "verifier" / "honest-0.pub.bin").read_bytes()
    assert record["program_sha512"] == json.loads((work / "verifier" / "partition.json").read_text())["program"]
    assert json.loads((work / "verifier" / "program.json").read_text())["program_fn"].startswith("PouwLiveWindow_v2{")
    caps = json.loads((work / "verifier" / "honest-0.registered.json").read_text())["ports"]
    assert [caps[f"cap_{i}"]["known"]["rows"][i] for i in range(T)] == LR.slot_words(LR.caps(MAX_TOKENS, T))


def test_a_tampered_opening_is_refused(prepared):
    base, pub, _rc = prepared
    opening = json.loads((base / "dump" / "1" / "opening.json").read_text())
    reg = LV.files(pub, 1, pub["served"][1])
    assert LV.check_openings(pub, 1, opening, reg) == []
    bad = dict(opening, slot_2=dict(opening["slot_2"], bc=opening["slot_3"]["bc"]))
    assert LV.check_openings(pub, 1, bad, reg) == ["live@1: port slot_2's opened row is not row 2 of h1/slots@1's registered root"]
    other = json.loads((base / "dump" / "0" / "opening.json").read_text())
    assert len(LV.check_openings(pub, 1, other, reg)) == T


def test_evidence_is_credits_live_statements_on_the_staged_program(prepared, tmp_path):
    """``evidence``: the live descriptor whose SHA-512 is the program each round was staged under, which core's credit
    reads T, S and the prefix back from (`credit.live_of`), and one statement a round in credit's form
    (`credit.LIVE_STATEMENT_KEYS`): the circuit, the verifier's own public file and the honest statement's sessions. A round
    not yet proved is refused, naming it."""
    import hashlib
    import shutil

    from verity.core.protocols.pouw import credit as CR

    base, _pub, _rc = prepared
    work = tmp_path / "work"
    shutil.copytree(base / "work", work)
    with pytest.raises(ValueError, match="round 0's statement honest-0 has no session"):
        LV.evidence(work)
    for s in range(S):
        (work / "st" / f"honest-{s}" / "sessions" / f"s{s}").mkdir(parents=True)
    doc = LV.evidence(work)
    raw = bytes.fromhex(doc["descriptor"])
    record = json.loads((work / "record.json").read_text())
    assert hashlib.sha512(raw).hexdigest() == record["program_sha512"]
    got, why = CR.live_of(raw)
    assert why is None and (got["T"], got["S"], got["prefix"]) == (T, S, PREFIX)
    assert [sorted(st) for st in doc["statements"]] == [sorted(CR.LIVE_STATEMENT_KEYS)] * S
    assert [(st["round"], st["public"], st["sessions"]) for st in doc["statements"]] == [
        (s, str(work / "verifier" / f"honest-{s}.pub.bin"), [str(work / "st" / f"honest-{s}" / "sessions" / f"s{s}")])
        for s in range(S)]
    assert all(st["circuit"] == str(work / "circuit.txt") for st in doc["statements"])
