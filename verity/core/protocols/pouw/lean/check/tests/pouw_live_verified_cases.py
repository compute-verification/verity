"""The live statements difftest's cases: the arguments of `credit._live_verified`'s `lived` (the Lean
`PouwCheck.LiveVerified`), a served window's live@<s> statements, its period's committed circuit files, the tables its
live units read and the verifier of record's answers.

A `live_verified` case is a live Program (`pouw_credit_cases.live_program`, at T of one to four rows, one to three
rounds, under a prefix or none), its registered entries (each round's slot table and round 0's request table, T rows
each, `credit._live_roots`', with other values beside them), each step's m_s (at most T), the window Program's digest,
its open root, its session and its period's committed max_tokens (fewer or more requests than T), one statement a round in a shuffled order (its circuit and public files' paths and one or
two sessions' directories), the committed circuit files (the live template's among them), and `runs`: null, the first
of the Lean's two calls (`credit._lean_live_verified`), or the answers the served path gives: one a statement
`credit._live_asks` names, in its order, until one the Lean refuses at, each session accepted with its round's unit and
its registered finding the ``--registered`` file's ports at their values and roots.

Its mutations before any run (`PRE_NAMES`, with `runs` null or answered) break the circuit files' form, drop the live
circuit file, break the statements' form, name a round past the window's or one twice, give a descriptor that is no
v2 live Program (``not-live``, a v1 Program among them), and leave a live table unregistered, registered twice or at other than T rows
(``no-table``). Its mutations of the answers (`RUN_NAMES`) make one statement's verifier fail to run (``error``) or
refuse it (``refused``), reject one session (``rejected``), and prove another unit than its round's (``unit-other``).

`tests/test_live_verified_difftest.py` draws them for the harness, replaying `runs` as the verifier
(`pouw_verified_cases.replay`).
"""
from __future__ import annotations

import hashlib
import random

import pouw_credit_cases as C
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw.circuit import live_reads as LR

PRE_NAMES = ["files-form", "no-live-file", "statements-form", "round-past", "twice", "not-live", "no-table"]
RUN_NAMES = ["error", "refused", "rejected", "unit-other"]
NAMES = PRE_NAMES + RUN_NAMES


def _finding(rng: random.Random, session: str, ask: dict, s: int, program: str) -> dict:
    """An accepted session's finding as `credit._finding_json` writes it: its round's unit, and its registered ports the
    ``--registered`` file's, at their values and roots."""
    ports = ask["registered"]["ports"]
    return {"session": session, "accepted": True, "why": "accepted", "program": program,
            "registered": {p: {"value": x["value"], "root": x["root"]} for p, x in ports.items()}, "units": [s]}


def _answers(rng: random.Random, case: dict, asks: list[dict], name: str, steps: int) -> list:
    """The served path's answers to `asks` (`credit._answer`'s form), with `name`'s mutation at one of them, cut after
    the first the Lean refuses at, as `credit._lean_live_verified` stops."""
    program = hashlib.sha512(bytes.fromhex(case["descriptor"])).hexdigest()
    at = rng.randrange(len(asks)) if asks else None
    out = []
    for j, a in enumerate(asks):
        st = case["statements"][a["statement"]]
        s = st["round"]
        fs = [_finding(rng, x, a, s, program) for x in st["sessions"]]
        ans = {"findings": fs}
        if j == at:
            f = rng.choice(fs)
            match name:
                case "error":
                    ans = {"error": rng.choice(["flock-verify ended on a signal", "no such file: public.bin"])}
                case "refused":
                    ans = {"refused": [rng.choice(["proof", "circuit", "registration"]),
                                       "the statement's public file is not its circuit's"]}
                case "rejected":
                    f.update(accepted=False, why=rng.choice(["sumcheck round 3", "opening root"]), units=[])
                case "unit-other":
                    f["units"] = rng.choice([[s + 1], [], [s, s], [s + steps], [s, s + 1]])
        out.append(ans)
        if "findings" not in ans or not all(f["accepted"] for f in ans["findings"]):
            break
    return out


def _entries(rng: random.Random, prefix: str, t: int, steps: int) -> list[dict]:
    def entry(v: str, words: int) -> dict:
        return {"value": v, "root": C._hex(rng), "leaves": t, "words": words, "domain": C._hex(rng)}
    out = [entry(LR.table(prefix, LR.SLOTS, s), LR.SLOT_WORDS) for s in range(steps)]
    out.append(entry(LR.table(prefix, LR.REQUESTS), LR.HEADER_WORDS))
    out += [entry(f"{prefix}{v}", rng.randint(1, 4)) for v in rng.sample(["a/64@0", "zy@0", "b/64", "ids@0"], 2)]
    rng.shuffle(out)
    return out


def live_verified_instance(rng: random.Random, name: str = "honest") -> dict:
    t, steps = rng.randint(1, 4), rng.randint(1, 3)
    prefix = rng.choice(["", f"h1/w-{rng.randrange(100)}/"])
    p = C.live_program(t, steps, prefix)
    statements = [{"round": s, "circuit": f"/w/live-{s}/circuit.bin", "public": f"/w/live-{s}/public.bin",
                   "sessions": [f"/w/live-{s}/s{j}" for j in range(rng.randint(1, 2))]} for s in range(steps)]
    rng.shuffle(statements)
    files = {Cr.LIVE: C._hex(rng), **{f"k{k}": C._hex(rng) for k in rng.sample([64, 128, 4096], rng.randint(0, 2))}}
    entries = _entries(rng, prefix, t, steps)
    case = {"files": files, "statements": statements, "descriptor": C.live_hex(p), "entries": entries,
            "ms": [rng.randint(0, t) for _s in range(steps)], "program": C._hex(rng), "window": C._hex(rng),
            "session": f"session-{rng.randrange(1 << 16)}",
            "max_tokens": [rng.randint(1, 40) for _q in range(rng.randint(1, t + 1))], "runs": None}
    st = rng.choice(statements)
    match name:
        case "honest" | "error" | "refused" | "rejected" | "unit-other":
            pass
        case "files-form":
            case["files"] = rng.choice([[], None, {**files, Cr.LIVE: C._hex(rng).upper()},
                                        {**files, "k32": C._hex(rng)[:-2]}, {**files, "x": 7}])
        case "no-live-file":
            del files[Cr.LIVE]
        case "statements-form":
            bad = rng.choice([{**st, "sessions": []}, {**st, "round": rng.choice([-1, "1", True, None])},
                              {k: v for k, v in st.items() if k != "public"}, {**st, "extra": 1},
                              {**st, "sessions": ["/w/s", 3]}, {**st, "circuit": None}, {**st, "call": 0}])
            case["statements"] = rng.choice([[bad], [*statements, bad], {"0": st}])
        case "round-past":
            statements.insert(rng.randrange(len(statements) + 1), {**st, "round": steps + rng.randrange(3)})
        case "twice":
            statements.insert(rng.randrange(len(statements) + 1), {**st, "sessions": ["/w/again"]})
        case "not-live":
            zero = f"{Cr.LIVE_FN}{{T=0,S={steps},PREFIX=\"\"}}"
            case["descriptor"] = rng.choice([b"{}".hex(), C.live_hex({**p, "schema": "verity-ir/descriptor/v0"}),
                                             C.live_hex({**p, "program_fn": zero}),
                                             C.live_hex({**p, "program_fn": "PouwH1Window_v3{}"}),
                                             C.live_hex(C.live_program(t, steps, prefix, version=1))])
        case "no-table":
            v = rng.choice([LR.table(prefix, LR.SLOTS, rng.randrange(steps)), LR.table(prefix, LR.REQUESTS)])
            e = next(x for x in entries if x["value"] == v)
            match rng.randrange(3):
                case 0:
                    entries.remove(e)
                case 1:
                    entries.append({**e, "root": C._hex(rng)})
                case _:
                    e["leaves"] = t + 1
        case _:
            raise ValueError(name)
    if name in RUN_NAMES or rng.random() < 0.6:
        case["runs"] = _answers(rng, case, reference_asks(case), name, steps)
    return case


#: each way a period's open commits no circuit files of `credit.committed_files`' form: none, null, empty, a list, a key
#: that is no template (``k0``, ``k064``, ``K64``, ``x``), or a digest that is none (upper case, short, a number)
FILES_NAMES = ["honest", "files-none", "files-null", "files-empty", "files-list", "files-key", "files-digest"]


def files_instance(rng: random.Random, name: str = "honest") -> dict:
    """A `credit-committed-files` case: the ledger and name of `pouw_credit_cases.period_instance`, its open committing
    each K's and the live template's circuit files, then mutation `name` of them."""
    files = {**{f"k{k}": C._hex(rng) for k in rng.sample([1, 64, 128, 4096], rng.randint(1, 3))},
             **({Cr.LIVE: C._hex(rng)} if rng.random() < 0.8 else {})}
    key, v = rng.choice(sorted(files.items()))
    match name:
        case "honest":
            committed = {Cr.CIRCUIT_FILES: files}
        case "files-none":
            committed = {}
        case "files-null":
            committed = {Cr.CIRCUIT_FILES: None}
        case "files-empty":
            committed = {Cr.CIRCUIT_FILES: {}}
        case "files-list":
            committed = {Cr.CIRCUIT_FILES: rng.choice([[v], v, 7])}
        case "files-key":
            committed = {Cr.CIRCUIT_FILES: {**files, rng.choice(["k0", "k064", "K64", "x", "lives", "k", "k1x"]): v}}
        case "files-digest":
            committed = {Cr.CIRCUIT_FILES: {**files, key: rng.choice([v.upper(), v[:-2], 7, None, [v]])}}
        case _:
            raise ValueError(name)
    d = C.period_instance(rng, committed_too=committed)
    return {"ledger": d["ledger"], "name": d["name"]}


def reference_asks(case: dict) -> list[dict]:
    """The verifier runs `credit._live_asks` names for `case` (none when its live Program or tables are refused)."""
    raw = bytes.fromhex(case["descriptor"])
    got, why = Cr.live_of(raw)
    if why is not None:
        return []
    roots, why = Cr._live_roots(case["entries"], got["prefix"], got["T"], got["S"])
    if why is not None:
        return []
    return Cr._live_asks(case["files"], case["statements"], raw, got, roots, case["ms"], case["program"],
                         case["window"], case["session"], case["max_tokens"])[0]
