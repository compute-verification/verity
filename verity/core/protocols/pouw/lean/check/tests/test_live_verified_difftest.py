"""The Lean verifier of record of a served window's live@<s> statements (`PouwCheck.LiveVerified`,
`pouw-check-difftest`'s ``credit-live-verified``) held to its Python reference, `credit._live_asks` and
`credit._live_verified`, on `pouw_live_verified_cases`: with `runs` null, the verifier runs the Lean names (or the
verdict, when it names none); with the answers, the verdict, each round's outcome or the refusal, string for string.
`credit.window` and `credit.period_credit` take their live outcomes from the Lean (`credit._lean_live_verified`), so
here the Lean is of record."""
from __future__ import annotations

import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_live_verified_cases as cases
from pouw_verified_cases import replay
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import verifier as V

if shutil.which("lake") is None:
    pytest.skip("lake is not on the PATH", allow_module_level=True)

LEAN = Path(__file__).resolve().parents[1]
VERIFIED: Counter[str] = Counter()


def _template(why: str) -> str:
    return re.sub(r"[0-9a-f]{128}", "<sha>", re.sub(r"\d+", "#", why))


def _args(case: dict) -> tuple | tuple[str, str]:
    """`lived`'s arguments after the live evidence for `case`, or the refusal of its live Program or tables (which
    `credit._live` makes before it asks `lived`)."""
    raw = bytes.fromhex(case["descriptor"])
    got, why = Cr.live_of(raw)
    if why is not None:
        return why
    roots, why = Cr._live_roots(case["entries"], got["prefix"], got["T"], got["S"])
    if why is not None:
        return why
    return raw, got, roots, case["ms"], case["program"], case["window"], case["session"], case["max_tokens"]


def live_verified(case: dict) -> dict:
    args = _args(case)
    got: list = []
    refused = args if len(args) == 2 else None
    if refused is None:
        if case["runs"] is None:
            asks, refused = Cr._live_asks(case["files"], case["statements"], *args)
            if asks:
                VERIFIED["run"] += 1
                return {"run": asks}
        else:
            got, refused = Cr._live_verified(case["files"], replay(case["runs"]))(
                {"statements": case["statements"]}, *args)
    VERIFIED[f"{refused[0]}: {_template(refused[1])}" if refused else "accept"] += 1
    return {"refused": list(refused)} if refused is not None else {"outcomes": got}


LIVE_PORT = D.Port("pouw-check-difftest", "credit-live-verified", live_verified, package=LEAN, need_rejects=False,
                   of_record=V.RECORD_VERDICT,
                   inputs=st.builds(cases.live_verified_instance, st.randoms(use_true_random=False),
                                    st.sampled_from(["honest"] * (len(cases.NAMES) // 3) + cases.NAMES)))
server = D.server_fixture(LIVE_PORT)


def test_live_verified(server):
    """Generated windows' live statements, null and answered, then each mutation over a few seeds, so every refusal
    before the verifier runs and of its answers is reached whatever the sample."""
    VERIFIED.clear()
    stats = D.run(LIVE_PORT, examples=200, server=server)
    swept = Counter(D.replay(LIVE_PORT, cases.live_verified_instance(random.Random(f"{name}/{seed}"), name), server)[0]
                    for name in ["honest", *cases.NAMES] for seed in range(4))
    print(f"pouw-check-difftest credit-live-verified: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by "
          f"outcome: {dict(sorted(VERIFIED.items()))}")
    assert {"run", "accept"} <= set(VERIFIED)
    for part in ("circuit: the committed circuit files are not",
                 "circuit: the period's open commits no circuit file for the live Program's template",
                 "live: the window's live evidence is not {descriptor, statements}",
                 "a live statement and the window has", "two live statements and the window has",
                 "live: the window's live circuit is not a live Program", "live: the window registers ",
                 "rows of # words, not the T = #", "could not be verified, so the window stays refused",
                 "live@#: the statement's public file", "was rejected by the verifier of record", "not unit #"):
        assert any(part in k for k in VERIFIED), part


def committed_files(case: dict) -> dict:
    files, why = Cr.committed_files(case["ledger"], case["name"])
    VERIFIED["files: " + (why[0] if why else "accept")] += 1
    return {"refused": list(why)} if why is not None else {"files": files}


FILES_PORT = D.Port("pouw-check-difftest", "credit-committed-files", committed_files, package=LEAN,
                    need_rejects=False, of_record=V.RECORD_VERDICT,
                    inputs=st.builds(cases.files_instance, st.randoms(use_true_random=False),
                                     st.sampled_from(cases.FILES_NAMES)))


def test_committed_files(server):
    """A period's open's circuit files read by the Lean of record as `credit.committed_files` reads them: honest, and
    each form that commits none."""
    VERIFIED.clear()
    stats = D.run(FILES_PORT, examples=60, server=server)
    for name in cases.FILES_NAMES:
        for seed in range(3):
            D.replay(FILES_PORT, cases.files_instance(random.Random(f"files/{name}/{seed}"), name), server)
    print(f"pouw-check-difftest credit-committed-files: {stats}; by outcome: {dict(sorted(VERIFIED.items()))}")
    assert set(VERIFIED) == {"files: accept", "files: circuit"}, VERIFIED


def test_served_live_answers():
    """`credit._lean_live_verified`, the served path through `verifier.lean`, gives `credit._live_verified`'s verdict on
    every answered case, running the verifier on the Lean's asks only: the same statements, with the same inputs, in the
    same order as the reference."""
    for name in ["honest", *cases.NAMES]:
        for seed in range(3):
            case = cases.live_verified_instance(random.Random(f"served/{name}/{seed}"), name)
            args = _args(case)
            if case["runs"] is None or len(args) == 2:
                continue
            calls: dict[str, list] = {"lean": [], "python": []}

            def logged(side: str):
                run = replay(case["runs"])

                def go(proof, **kw):
                    calls[side].append((proof, kw))
                    return run(proof, **kw)
                return go
            live = {"descriptor": case["descriptor"], "statements": case["statements"]}
            got = Cr._lean_live_verified(case["files"], case["entries"], logged("lean"))(live, *args)
            want = Cr._live_verified(case["files"], logged("python"))(live, *args)
            assert got == want, (name, seed, got, want)
            assert calls["lean"][:len(calls["python"])] == calls["python"], (name, seed)
