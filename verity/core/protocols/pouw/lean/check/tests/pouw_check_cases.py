"""The difftest's cases: generated windows, each a `serve` case ``{"commitment": hex, "evidence": hex, "params": JSON}``.

A case is an honest window (random grid bits, tile work, ε, δ, range and level; the registration one-stage's record of
it, the service's answer an acceptance of that registration with its profile and draw), then one mutation (`MUTATIONS`)
aimed at one refusal of `core_check`: of the parameters, the record, the answer, or a rule of the check.  Every string it
writes is ASCII and every number an integer, where Lean's canonical JSON and Python's agree.  `tests/test_difftest.py`
draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py` holds each mutation to
the refusal it aims at.
"""
from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd

EPS = ["1", "1/2", "1/3", "2/3", "1/10", "1/20", "3/4"]
DELTA = ["1/2", "1/10", "1/100", "1/1000", "1/65536", "1/1048576", "3/7"]
TILE_WORK = [1, 7, 64, 4096]


def _hex(rng: random.Random, n: int = 64) -> str:
    return rng.randbytes(n).hex()


def honest(rng: random.Random) -> dict:
    """An honest window: the verifier's parameters, the registration, and the service's accepting answer."""
    bits = rng.randrange(5)
    tw = rng.choice(TILE_WORK)
    params = {"check": {"entries": rng.randint(2, 32), "scheme": CC.SCHEME, "template": f"Pc8TileHidden{{{tw},1,1,sm120}}",
                        "tile_work": tw},
              "context": f"served-zk/run-{rng.randrange(1000)}/{_hex(rng, 8)}", "delta": rng.choice(DELTA),
              "eps": rng.choice(EPS), "grid_bits": bits, "query": _hex(rng)}
    top = rng.choice([4, 12, 24, 40])
    hi = rng.randint(2, 1 << top)
    ell_hi = Wd.level(Wd.floor_point(hi, bits), bits) - 1
    ell = rng.randint(0, ell_hi) if ell_hi >= 0 else 0
    lo = rng.randint(1, Wd.point(ell, bits))
    params["range"] = [lo, max(hi, Wd.point(ell + 1, bits))]
    w = Wd.Work.at(ell, bits)
    k = Wd.k_prime(Fraction(params["eps"]), Fraction(params["delta"]), w.credited * tw, w.n, tw)
    root_w = _hex(rng)
    items = {"check": dict(params["check"]), "root_W": root_w, "work": w.to_json(tw),
             "sampling": Wd.sampling(Fraction(params["eps"]), Fraction(params["delta"]), w, tw, bits)}
    program, partition = _hex(rng), _hex(rng)
    rec = {"format": CC.FORMATS[0] if rng.random() < 0.9 else CC.FORMATS[1], "protocol": CC.ONE_STAGE,
           "program": program, "query": params["query"], "partition": partition, "population": w.n,
           "law": {"law": "subset", "k": k}, "scheme": CC.SCHEME,
           "roots": [{"port": "entries", "root": _hex(rng), "leaves": params["check"]["entries"], "domain": _hex(rng)},
                     {"port": "W", "root": root_w, "leaves": rng.randint(1, 1 << 20), "domain": _hex(rng)}],
           "leaf_layer": None, "anchors": [], "window": {"kind": "served", "context": params["context"], "public": items},
           "previous": None}
    if rng.random() < 0.3:
        rec["roots"].reverse()
    units = sorted(rng.sample(range(w.n), k))
    answer = {"commitment": None, "protocol": CC.ONE_STAGE, "reason": None, "verdict": "accept",
              "profile": {"covered": [], "delta": params["delta"], "drawn": {CC.LEVEL: k}, "linked": rng.random() < 0.5,
                          "levels": [{"draw": {"subset": k}, "name": CC.LEVEL, "query": partition, "size": w.n}],
                          "prescribed": sorted(rng.sample(["prompt", "weights", "noise"], rng.randrange(3))),
                          "programs": [program]},
              "public": {"draw": {"k": k, "law": "subset", "population": w.n, "units": units}}}
    return {"params": params, "record": rec, "answer": answer}


def _canon(x) -> str:
    return CC._canon(x)


def _set(obj, path: str, value) -> None:
    *up, last = path.split(".")
    for key in up:
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    if isinstance(obj, list):
        obj[int(last)] = value
    elif value is _DROP:
        del obj[last]
    else:
        obj[last] = value


_DROP = object()

#: (name, input, path, value): one field of one input set to `value` (`_DROP` removes it), or a named edit below
MUTATIONS = [
    ("params-keys", "params", "extra", 1), ("params-check-keys", "params", "check.extra", 1),
    ("params-entries", "params", "check.entries", 1), ("params-scheme", "params", "check.scheme", "blake3"),
    ("params-template", "params", "check.template", ""), ("params-tile-work", "params", "check.tile_work", 0),
    ("params-context", "params", "context", ""), ("params-delta-type", "params", "delta", 1),
    ("params-delta-form", "params", "delta", "01/2"), ("params-delta-den", "params", "delta", "1/1"),
    ("params-delta-zero-den", "params", "delta", "3/0"), ("params-delta-lowest", "params", "delta", "2/4"),
    ("params-delta-ratio", "params", "delta", "1/2/3"), ("params-delta-range", "params", "delta", "1"),
    ("params-delta-zero", "params", "delta", "0"), ("params-eps-range", "params", "eps", "3/2"),
    ("params-eps-zero", "params", "eps", "0"), ("params-eps-empty", "params", "eps", ""),
    ("params-grid", "params", "grid_bits", "3"), ("params-query", "params", "query", "ab"),
    ("params-range", "params", "range", [3, 2]), ("params-range-zero", "params", "range", [0, 5]),
    ("params-eps-other", "params", "eps", "1/7"), ("params-delta-other", "params", "delta", "1/9"),
    ("params-grid-other", "params", "grid_bits", 7), ("params-tile-other", "params", "check.tile_work", 3),
    ("params-template-other", "params", "check.template", "Other{1}"), ("params-query-other", "params", "query", None),
    ("params-range-low", "params", "range", None), ("params-range-high", "params", "range", None),
    ("record-format", "record", "format", "verity/registration/v0"), ("record-program", "record", "program", 7),
    ("record-partition", "record", "partition", _DROP), ("record-population-type", "record", "population", "8"),
    ("record-law-missing", "record", "law", _DROP), ("record-roots", "record", "roots", [{"port": "W"}]),
    ("record-window-keys", "record", "window.extra", 1), ("record-window-public", "record", "window.public", []),
    ("record-kind", "record", "window.kind", "stand-in"), ("record-context", "record", "window.context", "elsewhere"),
    ("record-extra-item", "record", "window.public.verdict", True), ("record-missing-item", "record", "window.public.root_W", _DROP),
    ("record-level-missing", "record", "window.public.work.level", _DROP),
    ("record-level-type", "record", "window.public.work.level", "3"), ("record-level-high", "record", "window.public.work.level", None),
    ("record-level-huge", "record", "window.public.work.level", 10 ** 30), ("record-level-step", "record", "window.public.work.level", None),
    ("record-work", "record", "window.public.work.credited", None), ("record-check", "record", "window.public.check.entries", None),
    ("record-sampling", "record", "window.public.sampling.law.k", None), ("record-root-w", "record", "window.public.root_W", None),
    ("record-no-w-root", "record", "roots", None), ("record-population", "record", "population", None),
    ("record-law", "record", "law.k", None), ("record-law-kind", "record", "law", {"law": "bernoulli", "p": "1/2"}),
    ("record-query", "record", "query", None), ("record-scheme", "record", "scheme", "frame-v3/hm96"),
    ("answer-keys", "answer", "extra", 1), ("answer-protocol-type", "answer", "protocol", 3),
    ("answer-verdict", "answer", "verdict", "maybe"), ("answer-commitment-type", "answer", "commitment", 5),
    ("answer-shape", "answer", "reason", "why"), ("answer-profile", "answer", "profile.levels", {}),
    ("answer-profile-delta", "answer", "profile.delta", "2/4"), ("answer-profile-draw", "answer", "profile.levels.0.draw", {"all": 1}),
    ("answer-public", "answer", "public", {"units": []}), ("answer-public-law", "answer", "public.draw.law", "bernoulli"),
    ("answer-protocol", "answer", "protocol", "verity/pous/v1"), ("answer-other", "answer", "commitment", None),
    ("answer-null", "answer", "commitment", _DROP), ("answer-reject", "answer", None, "reject"),
    ("answer-undecided", "answer", None, "undecided"), ("answer-programs", "answer", "profile.programs", None),
    ("answer-programs-two", "answer", "profile.programs", None), ("answer-level-query", "answer", "profile.levels.0.query", None),
    ("answer-level-size", "answer", "profile.levels.0.size", None), ("answer-level-draw", "answer", "profile.levels.0.draw", None),
    ("answer-level-bernoulli", "answer", "profile.levels.0.draw", {"bernoulli": "1/2"}),
    ("answer-level-stratified", "answer", "profile.levels.0.draw",
     {"law": "stratified", "strata": [{"k": 1, "name": "t", "size": 2}]}),
    ("answer-level-name", "answer", "profile.levels.0.name", "tile"), ("answer-levels-two", "answer", "profile.levels", None),
    ("answer-delta", "answer", "profile.delta", "1/3"), ("answer-drawn", "answer", "profile.drawn", None),
    ("answer-draw-k", "answer", "public.draw.k", None), ("answer-draw-population", "answer", "public.draw.population", None),
    ("answer-units-order", "answer", "public.draw.units", None), ("answer-units-range", "answer", "public.draw.units", None),
    ("answer-units-count", "answer", "public.draw.units", None), ("answer-unsorted-lists", "answer", "profile.covered", ["z", "a"]),
    ("answer-commitment-odd", "answer", "commitment", None), ("answer-commitment-upper", "answer", "commitment", None),
    ("answer-commitment-empty", "answer", "commitment", ""), ("answer-verdict-type", "answer", "verdict", 1),
    ("answer-verdict-missing", "answer", "verdict", _DROP), ("answer-reason-type", "answer", "reason", 5),
    ("answer-public-missing", "answer", "public", _DROP), ("answer-public-null", "answer", "public", None),
    ("answer-reject-public", "answer", "verdict", "reject"), ("answer-profile-type", "answer", "profile", [1]),
    ("answer-public-bernoulli", "answer", "public.draw", None),
    ("answer-public-bernoulli-unreduced", "answer", "public.draw", None),
    ("answer-public-stratified", "answer", "public.draw", None), ("answer-public-work", "answer", "public.draw", None),
    ("answer-public-work-bare", "answer", "public.draw", None),
    ("answer-public-law-other", "answer", "public.draw.law", "uniform"),
    ("answer-public-law-type", "answer", "public.draw.law", 3),
    ("answer-public-subset-closure", "answer", "public.draw.closure", [0]),
    ("answer-public-stratum-keys", "answer", "public.draw", None),
    ("answer-public-stratum-range", "answer", "public.draw", None),
    ("answer-public-stratum-work", "answer", "public.draw", None),
    ("answer-public-closure-type", "answer", "public.draw", None),
    ("answer-public-p-type", "answer", "public.draw", None),
    ("answer-public-units-type", "answer", "public.draw.units", "all"),
    ("answer-public-unit-type", "answer", "public.draw.units", None),
    ("answer-public-population-type", "answer", "public.draw.population", "8"),
    ("answer-drawn-extra", "answer", "profile.drawn.tile", 1), ("answer-drawn-type", "answer", "profile.drawn.unit", "3"),
    ("answer-linked-type", "answer", "profile.linked", 1), ("answer-programs-type", "answer", "profile.programs", "p"),
    ("answer-program-type", "answer", "profile.programs", [1]),
    ("answer-level-keys", "answer", "profile.levels.0.extra", 1),
    ("answer-level-missing", "answer", "profile.levels.0.size", _DROP),
    ("answer-level-null-query", "answer", "profile.levels.0.query", None),
    ("answer-level-null-draw", "answer", "profile.levels.0.draw", None),
    ("answer-level-strata-keys", "answer", "profile.levels.0.draw",
     {"law": "stratified", "strata": [{"k": 1, "name": "t", "size": 2, "x": 0}]}),
    ("answer-level-draw-two", "answer", "profile.levels.0.draw", {"bernoulli": "1/2", "subset": 3}),
    ("answer-level-bernoulli-lowest", "answer", "profile.levels.0.draw", {"bernoulli": "2/4"}),
    ("answer-delta-type", "answer", "profile.delta", 1), ("answer-level-renamed", "answer", "profile", None),
    ("case-commitment-upper", None, None, None), ("case-evidence-odd", None, None, None),
    ("case-params-missing", None, None, None), ("case-evidence-missing", None, None, None),
    ("case-commitment-type", None, None, None), ("case-not-object", None, None, None),
    ("noncanonical-record", None, None, None),
    ("noncanonical-answer", None, None, None), ("float-answer", None, None, None), ("utf8-record", None, None, None),
]
NAMES = [m[0] for m in MUTATIONS]


def _derived(name: str, case: dict, rng: random.Random):
    """The value of a mutation whose value depends on the honest case."""
    p, rec, ans = case["params"], case["record"], case["answer"]
    items, bits, tw = rec["window"]["public"], p["grid_bits"], p["check"]["tile_work"]
    ell = items["work"]["level"]
    if name in ("params-query-other", "answer-other"):
        return _hex(rng)
    if name == "params-range-low":
        return [Wd.point(ell, bits) + 1, max(p["range"][1], Wd.point(ell, bits) + 1)]
    if name == "params-range-high":
        return [1, max(1, Wd.point(ell + 1, bits) - 1)]
    if name == "record-level-high":
        return Wd.level(Wd.floor_point(p["range"][1], bits), bits) + rng.randrange(3)
    if name == "record-level-step":
        return ell + 1 if Wd.point(ell + 2, bits) <= p["range"][1] else ell - 1 if ell >= 1 and Wd.point(ell - 1, bits) >= p["range"][0] else ell + 7
    if name == "record-work":
        return items["work"]["credited"] + 1
    if name == "record-check":
        return items["check"]["entries"] + 1
    if name == "record-sampling":
        return items["sampling"]["law"]["k"] + 1
    if name == "record-root-w":
        return _hex(rng)
    if name == "record-no-w-root":
        return [r for r in rec["roots"] if r["port"] != "W"]
    if name == "record-population":
        return rec["population"] + rng.choice([-1, 1])
    if name == "record-law":
        return rec["law"]["k"] + 1
    if name == "record-query":
        return _hex(rng)
    if name == "answer-programs":
        return [_hex(rng)]
    if name == "answer-programs-two":
        return [rec["program"], rec["program"]]
    if name == "answer-level-query":
        return _hex(rng)
    if name == "answer-level-size":
        return rec["population"] + 1
    if name == "answer-level-draw":
        return {"subset": rec["law"]["k"] + 1}
    if name == "answer-levels-two":
        return [{"draw": None, "name": "window", "query": None, "size": 1}, *ans["profile"]["levels"]]
    if name == "answer-drawn":
        return {CC.LEVEL: rec["law"]["k"] + 1}
    if name == "answer-draw-k":
        return rec["law"]["k"] + 1
    if name == "answer-draw-population":
        return rec["population"] * 2
    if name == "answer-commitment-odd":
        return ans["commitment"][:-1]
    if name == "answer-commitment-upper":
        return ans["commitment"].upper()
    if name in ("answer-public-null", "answer-level-null-query", "answer-level-null-draw"):
        return None
    if name == "answer-level-renamed":
        prof = ans["profile"]
        return {**prof, "drawn": {"tile": prof["drawn"][CC.LEVEL]}, "levels": [{**prof["levels"][0], "name": "tile"}]}
    units, n, k = ans["public"]["draw"]["units"], rec["population"], rec["law"]["k"]
    stratum = {"k": k, "template": "t0", "units": [[0, n]]}
    work = {**stratum, "floor": 1, "work": 3}
    if name == "answer-public-bernoulli":
        return {"law": "bernoulli", "p": "1/2", "population": n, "units": units}
    if name == "answer-public-bernoulli-unreduced":
        return {"law": "bernoulli", "p": "2/4", "population": n, "units": units}
    if name == "answer-public-p-type":
        return {"law": "bernoulli", "p": 1, "population": n, "units": units}
    if name == "answer-public-stratified":
        return {"k": k, "law": "stratified", "population": n, "strata": [stratum], "units": units}
    if name == "answer-public-work":
        return {"closure": units[:1], "k": k, "law": "work", "population": n, "strata": [work], "units": units}
    if name == "answer-public-work-bare":
        return {"k": k, "law": "work", "population": n, "strata": [work], "units": units}
    if name == "answer-public-stratum-keys":
        return {"k": k, "law": "stratified", "population": n, "strata": [{"template": "t0", "units": [[0, n]]}],
                "units": units}
    if name == "answer-public-stratum-range":
        return {"k": k, "law": "stratified", "population": n, "strata": [{**stratum, "units": [[0]]}], "units": units}
    if name == "answer-public-stratum-work":
        return {"k": k, "law": "work", "population": n, "strata": [{**work, "work": "3"}], "units": units}
    if name == "answer-public-closure-type":
        return {"closure": "none", "k": k, "law": "work", "population": n, "strata": [work], "units": units}
    if name == "answer-public-unit-type":
        return units[:-1] + [-1]
    if name == "answer-units-order":
        return list(reversed(units)) if len(units) > 1 else units + units
    if name == "answer-units-range":
        return units[:-1] + [rec["population"]]
    if name == "answer-units-count":
        return units[:-1]
    raise AssertionError(name)


def instance(rng: random.Random, name: str = "honest") -> dict:
    """An honest window from `rng`, then the mutation `name` (none for "honest")."""
    case = honest(rng)
    which, path, value = next(m[1:] for m in MUTATIONS if m[0] == name) if name != "honest" else (None, None, None)

    def mutate(part: str) -> None:
        nonlocal value
        if which != part:
            return
        if part == "answer" and path is None:
            case["answer"].update({"verdict": value, "reason": f"the audit's {value} ({rng.randrange(100)})",
                                   "profile": None, "public": None})
            return
        if value is None:
            value = _derived(name, case, rng)
        _set(case[part], path, value)

    mutate("params")
    mutate("record")
    params, rec, ans = case["params"], case["record"], case["answer"]
    rec_b = _canon(rec).encode()
    if isinstance(rec.get("format"), str):
        ans["commitment"] = hashlib.sha512(rec["format"].encode() + b"\0" + rec_b).hexdigest()
    mutate("answer")
    if name == "answer-null":
        ans["commitment"] = None
    ans_b = _canon(ans).encode()
    if name == "noncanonical-record":
        rec_b = json.dumps(rec, sort_keys=True).encode()
    elif name == "noncanonical-answer":
        ans_b = _canon(dict(reversed(list(ans.items())))).encode().replace(b'{"', b'{ "', 1)
    elif name == "float-answer":
        ans_b = ans_b.replace(b'"verdict":"accept"', b'"verdict":"accept","x":1.5')
    elif name == "utf8-record":
        rec_b = rec_b + b"\xff"
    raw = {"commitment": rec_b.hex(), "evidence": ans_b.hex(), "params": params}
    if name == "case-commitment-upper":
        raw["commitment"] = raw["commitment"].upper()
    elif name == "case-evidence-odd":
        raw["evidence"] += "0"
    elif name in ("case-params-missing", "case-evidence-missing"):
        del raw[name.split("-")[1]]
    elif name == "case-commitment-type":
        raw["commitment"] = len(rec_b)
    elif name == "case-not-object":
        return [raw]
    return raw
