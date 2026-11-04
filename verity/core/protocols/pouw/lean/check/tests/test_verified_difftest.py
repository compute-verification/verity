"""The Lean verifier of record of a served window's statements (`PouwCheck.Verified`, `pouw-check-difftest`'s
``credit-verified``) held to its Python reference, `credit._asks` and `credit._verified`, on `pouw_verified_cases`:
with `runs` null, the verifier runs the Lean names (or the verdict, when it names none); with the answers, the verdict,
each sampled proof block's outcome or the refusal, string for string. `credit.window` and `credit.period_credit` take their
outcomes from the Lean (`credit._lean_verified`), so here the Lean is of record."""
from __future__ import annotations

import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_verified_cases as cases
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import verifier as V

if shutil.which("lake") is None:
    pytest.skip("lake is not on the PATH", allow_module_level=True)

LEAN = Path(__file__).resolve().parents[1]
VERIFIED: Counter[str] = Counter()


def _template(why: str) -> str:
    return re.sub(r"[0-9a-f]{128}", "<sha>", re.sub(r"\d+", "#", why))


def verified(case: dict) -> dict:
    raw = bytes.fromhex(case["descriptor"])
    lay, refused = Cr.layout_of(raw)
    got: list = []
    if refused is None:
        args = (case["statements"], case["files"], raw, lay, case["entries"], case["window"], case["units"],
                case["steps"])
        if case["runs"] is None:
            asks, refused = Cr._asks(*args, case["salt"])
            if asks:
                VERIFIED["run"] += 1
                return {"run": asks}
        else:
            got, refused = Cr._verified(*args, cases.replay(case["runs"]), case["salt"])
    VERIFIED[f"{refused[0]}: {_template(refused[1])}" if refused else "accept"] += 1
    return {"refused": list(refused)} if refused is not None else {"outcomes": got}


VERIFIED_PORT = D.Port("pouw-check-difftest", "credit-verified", verified, package=LEAN, need_rejects=False,
                       of_record=V.RECORD_VERDICT,
                       inputs=st.builds(cases.verified_instance, st.randoms(use_true_random=False),
                                        st.sampled_from(["honest"] * (len(cases.NAMES) // 3) + cases.NAMES)))
server = D.server_fixture(VERIFIED_PORT)


def test_verified(server):
    """Generated windows' statements, null and answered, then each mutation over a few seeds (`replay`), so every
    refusal before the verifier runs and of its answers is reached whatever the sample."""
    VERIFIED.clear()
    stats = D.run(VERIFIED_PORT, examples=300, server=server)
    swept = Counter(D.replay(VERIFIED_PORT, cases.verified_instance(random.Random(f"{name}/{seed}"), name), server)[0]
                    for name in ["honest", *cases.NAMES] for seed in range(4))
    print(f"pouw-check-difftest credit-verified: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(VERIFIED.items()))}")
    assert {"run", "accept"} <= set(VERIFIED)
    for part in ("circuit: the committed circuit files are not", "proof: the window's statements are not a list",
                 "outcome: call # has two statements", "outcome: call # has a statement and no drawn unit",
                 "circuit: the period's open commits no circuit file for K = #",
                 "registration: the window registers ", "which the window registers not", "layout: ",
                 "proof: call #'s statement could not be verified", "'s statement: the statement's public file",
                 "failed: call #'s session ", "opened a public table at another port",
                 "opened the ok rows at another port", "which is no unit of call #"):
        assert any(part in k for k in VERIFIED), part


def test_served_answers():
    """`credit._lean_verified`, the served path through `verifier.lean`, gives `credit._verified`'s verdict on every
    answered case, running the verifier on the Lean's asks only: the same statements, with the same inputs, in the same
    order as the reference."""
    for name in ["honest", *cases.NAMES]:
        for seed in range(3):
            case = cases.verified_instance(random.Random(f"served/{name}/{seed}"), name)
            if case["runs"] is None:
                continue
            raw = bytes.fromhex(case["descriptor"])
            lay, refused = Cr.layout_of(raw)
            if refused is not None:
                continue
            calls: dict[str, list] = {"lean": [], "python": []}

            def logged(side: str):
                run = cases.replay(case["runs"])

                def go(proof, **kw):
                    calls[side].append((proof, kw))
                    return run(proof, **kw)
                return go
            args = (case["statements"], case["files"], raw)
            got = Cr._lean_verified(*args, case["entries"], case["window"], case["units"], case["steps"],
                                    logged("lean"), case["salt"])
            want = Cr._verified(*args, lay, case["entries"], case["window"], case["units"], case["steps"],
                                logged("python"), case["salt"])
            assert got == want, (name, seed, got, want)
            assert calls["lean"][:len(calls["python"])] == calls["python"], (name, seed)
