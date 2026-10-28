"""The statements difftest's cases: the arguments of `credit._verified` (the Lean `PouwCheck.Verified`), a served window's
statements, its period's committed circuit files and the verifier of record's answers.

A `verified` case is a served window (`pouw_credit_cases.layout`, its descriptor under a prefix or none, its calls over
one to three rounds), one to six sampled proof blocks, the entries a statement of each drawn call reads (its round's A table of
its K, its z table and the weights' ``b/<K>``, `credit.registered_file`'s, each registered once), one statement a drawn
call in a shuffled order (its circuit and public files' paths and one to three sessions' directories), each K's
committed circuit file, the window's open root, rounds and Program salt, and `runs`: null, the first of the Lean's two
calls (`credit._lean_verified`), or the answers the served path gives: one a statement `credit._asks` names, in its
order, until one the Lean refuses at, each session accepted with the units of its call split among them and its
registered finding the ``--registered`` file's ports at their values and roots.

Its mutations before any run (`PRE_NAMES`, with `runs` null or answered) break the circuit files' form, the statements'
form or name a call past the window's, give a call two statements or one to a call with no sampled proof block, drop a K's
circuit file, register the window's ok rows (``ok-registered``) or none of a value a statement reads (``unregistered``),
and give a descriptor that is no H1 window's. Its mutations of the answers (`RUN_NAMES`) make one statement's verifier
fail to run (``error``) or refuse it (``refused``), reject one session (``rejected``), open a public table or the ok
rows at another port, value or root, open no ok row (``ok-none``), and prove a unit of another call (``unit-other``).

`tests/test_verified_difftest.py` draws them for the harness, replaying `runs` as the verifier (`replay`).
"""
from __future__ import annotations

import bisect
import hashlib
import itertools
import random

import pouw_credit_cases as C
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

PRE_NAMES = ["files-form", "statements-form", "call-past", "twice", "undrawn", "no-file", "ok-registered",
             "unregistered", "layout"]
RUN_NAMES = ["error", "refused", "rejected", "pub-other", "ok-other", "ok-none", "unit-other"]
NAMES = PRE_NAMES + RUN_NAMES


def _starts(lay: dict) -> tuple[list[int], list[int]]:
    sizes = [size for size, _work in Cr.call_work(lay)]
    return sizes, list(itertools.accumulate(sizes, initial=0))[:-1]


def _reads(lay: dict, prefix: str, steps: int, c: int) -> list[str]:
    """The values call `c`'s statement reads that the window registers (`credit.registered_file`'s)."""
    k = {lin: k for lin, k, _n in lay["linears"]}[lay["calls"][c][0]]
    return [f"{prefix}a/{k}@{Cr.step_row(lay, steps, c)[0]}", f"{prefix}zy@{c}", f"{prefix}b/{k}"]


def _finding(rng: random.Random, session: str, ask: dict, units: list[int], program: str) -> dict:
    """An accepted session's finding as `credit._finding_json` writes it: its registered ports the ``--registered``
    file's, at their values and roots, some of the A and B ports among them."""
    ports = ask["registered"]["ports"]
    opened = [p for p in ports if p in (Cr.PUB_PORT, Cr.OK_PORT) or rng.random() < 0.5]
    return {"session": session, "accepted": True, "why": "accepted", "program": program,
            "registered": {p: {"value": ports[p]["value"], "root": ports[p]["root"]} for p in opened}, "units": units}


def _answers(rng: random.Random, case: dict, asks: list[dict], name: str, starts: list[int], sizes: list[int]) -> list:
    """The served path's answers to `asks` (`credit._answer`'s form), with `name`'s mutation at one of them, cut after
    the first the Lean refuses at, as `credit._lean_verified` stops."""
    program = hashlib.sha512(bytes.fromhex(case["descriptor"])).hexdigest()
    at = rng.randrange(len(asks)) if asks else None
    out = []
    for j, a in enumerate(asks):
        s = case["statements"][a["statement"]]
        c = s["call"]
        mine = [u for u in case["units"] if starts[c] <= u < starts[c] + sizes[c]]
        parts = [mine[i::len(s["sessions"])] for i in range(len(s["sessions"]))]
        fs = [_finding(rng, x, a, part, program) for x, part in zip(s["sessions"], parts)]
        ans = {"findings": fs}
        if j == at:
            f = rng.choice(fs)
            ports = a["registered"]["ports"]
            pub, ok = ports[Cr.PUB_PORT], ports[Cr.OK_PORT]
            match name:
                case "error":
                    ans = {"error": rng.choice(["flock-verify ended on a signal", "no such file: public.bin"])}
                case "refused":
                    ans = {"refused": [rng.choice(["proof", "circuit", "registration"]),
                                       "the statement's public file is not its circuit's"]}
                case "rejected":
                    f.update(accepted=False, why=rng.choice(["sumcheck round 3", "opening root"]), units=[])
                case "pub-other":
                    other = rng.choice([{"value": pub["value"], "root": C._hex(rng)},
                                        {"value": f"{pub['value'].rsplit('@', 1)[0]}@{c + 1}", "root": pub["root"]}])
                    if rng.random() < 0.5:
                        f["registered"][Cr.PUB_PORT] = other
                    else:
                        f["registered"]["o9"] = other
                case "ok-other":
                    if rng.random() < 0.5:
                        f["registered"][Cr.OK_PORT] = {"value": ok["value"], "root": C._hex(rng)}
                    else:
                        f["registered"]["o9"] = {"value": ok["value"], "root": ok["root"]}
                case "ok-none":
                    del f["registered"][Cr.OK_PORT]
                case "unit-other":
                    f["units"] = sorted({*f["units"], rng.choice(
                        [u for u in range(sum(sizes)) if not starts[c] <= u < starts[c] + sizes[c]] or [sum(sizes)])})
        out.append(ans)
        if "findings" not in ans or not all(f["accepted"] for f in ans["findings"]):
            break
    return out


def verified_instance(rng: random.Random, name: str = "honest") -> dict:
    lay = C.layout(rng)
    prefix = rng.choice(["", f"h1/w-{rng.randrange(100)}/"])
    steps = rng.choice([s for s in (1, 2, 3) if len(lay["calls"]) % s == 0])
    sizes, starts = _starts(lay)
    units = sorted(rng.sample(range(sum(sizes)), rng.randint(1, min(6, sum(sizes)))))
    drawn = sorted({bisect.bisect_right(starts, u) - 1 for u in units})
    reads = {v for c in drawn for v in _reads(lay, prefix, steps, c)}
    entries = [{"value": v, "root": C._hex(rng)} for v in sorted(reads)]
    rng.shuffle(entries)
    statements = [{"call": c, "circuit": f"/w/c{c}/circuit.bin", "public": f"/w/c{c}/public.bin",
                   "sessions": [f"/w/c{c}/s{j}" for j in range(rng.randint(1, 3))]} for c in drawn]
    rng.shuffle(statements)
    files = {f"k{k}": C._hex(rng) for k in sorted({k for _lin, k, _n in lay["linears"]})}
    case = {"statements": statements, "files": files, "descriptor": C.descriptor(rng, lay, prefix), "entries": entries,
            "window": C._hex(rng), "units": units, "steps": steps, "salt": C._hex(rng), "runs": None}
    s = rng.choice(statements)
    match name:
        case "honest" | "error" | "refused" | "rejected" | "pub-other" | "ok-other" | "ok-none" | "unit-other":
            pass
        case "files-form":
            case["files"] = rng.choice([[], None, {**files, f"k{lay['linears'][0][1]}": C._hex(rng).upper()},
                                        {**files, "k32": C._hex(rng)[:-2]}, {**files, "x": 7}])
        case "statements-form":
            bad = rng.choice([{**s, "sessions": []}, {**s, "call": rng.choice([-1, "1", True, None])},
                              {k: v for k, v in s.items() if k != "public"}, {**s, "extra": 1},
                              {**s, "sessions": ["/w/s", 3]}, {**s, "circuit": None}])
            case["statements"] = rng.choice([[bad], [*statements, bad], {"0": s}])
        case "call-past":
            statements.append({**s, "call": len(lay["calls"]) + rng.randrange(3)})
        case "twice":
            statements.insert(rng.randrange(len(statements) + 1), {**s, "sessions": ["/w/again"]})
        case "undrawn":
            idle = [c for c in range(len(lay["calls"])) if c not in drawn]
            if idle:
                statements.insert(rng.randrange(len(statements) + 1), {**s, "call": rng.choice(idle)})
            else:
                statements.insert(rng.randrange(len(statements) + 1), {**s})
        case "no-file":
            k = {lin: k for lin, k, _n in lay["linears"]}[lay["calls"][s["call"]][0]]
            del files[f"k{k}"]
        case "ok-registered":
            entries.insert(rng.randrange(len(entries) + 1), {"value": f"{prefix}{Cr.OK_VALUE}", "root": C._hex(rng)})
        case "unregistered":
            v = rng.choice(_reads(lay, prefix, steps, s["call"]))
            case["entries"] = [e for e in entries if e["value"] != v] if rng.random() < 0.5 else [
                *entries, {"value": v, "root": C._hex(rng)}]
        case "layout":
            case["descriptor"] = rng.choice([b"{}", b"[1]", C.desc_bytes({"schema": Cr.SCHEMA})]).hex()
        case _:
            raise ValueError(name)
    if name in RUN_NAMES or rng.random() < 0.6:
        raw = bytes.fromhex(case["descriptor"])
        layout, why = Cr.layout_of(raw)
        asks = [] if why is not None else Cr._asks(case["statements"], case["files"], raw, layout, case["entries"],
                                                   case["window"], case["units"], steps, case["salt"])[0]
        case["runs"] = _answers(rng, case, asks, name, starts, sizes)
    return case


def replay(runs: list) -> Cr.Run:
    """A verifier of record that answers `runs` in turn, as `credit._answer` reads its run."""
    answers = iter(runs)

    def run(_proof, *, circuit_sha512, program, partition, registered):
        a = next(answers)
        if "error" in a:
            raise ValueError(a["error"])
        if "refused" in a:
            return None, tuple(a["refused"])
        return [PF.Finding(f["session"], f["accepted"], f["why"], f["program"], f["registered"], tuple(f["units"]))
                for f in a["findings"]], None
    return run
