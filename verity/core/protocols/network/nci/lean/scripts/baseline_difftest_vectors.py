"""The provenance check's difftest data: declarations with θ, θ's provenance and the service's trusted input, for
`NciCheck/BaselineDifftestMain.lean`.

    python scripts/baseline_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes three files under `DIR/baseline_difftest/`, one line per case: `cases.jsonl`, `{"instance", "trusted"}` as
`NciCheck.Baseline.check` reads them (`NciCheck.Baseline`'s docstring); `expanded.jsonl`, the same case expanded gate by
gate (`nci_reference.expand_baseline`) as `NciCheck.Baseline.checkUnrolled` reads it, or `null` where the policy check
refuses the instance or its provenance is not one entry per θ entry; and `python.jsonl`, `nci_reference.check_baseline`'s
results. The Lean audit runs `NciCheck/BaselineDifftestMain.lean` on the three (`runs` in `lean-audit.json`).

Case `seed`'s declaration, for the first `COUNT`, is one of `difftest_vectors`' shapes with each parameter bound at
registration with probability 0.6, divided into one IU, with `Γ` and `α` large enough that the policy check gives a
verdict. θ is a random set of its registered input nodes. Its trusted input is a deployment's log in R6's chain: up to
three records before the run's own, so the run is the deployment's first in a quarter of the cases, and up to one after.
Each parameter holding a θ node is registered in the run's record under the root its provenance cites: the log's first
record's registered value of the same root and shape (one entry, when the run's record is that first record and the
value is the parameter's name). The records carry decoy values and ports. Just under half the cases are broken in one
way, each refused with one of `CODES`; an entry of another kind (`output` among them) is `theta-provenance`, and one
citing a later record that registers the same value under the same root is `theta-not-at-deployment-start`. The data
ends with `pinned()`: on a declaration worked by hand, two cases the check accepts (the first record cited from the
deployment's first run and from its third), one negative per code, an entry of the `output` kind θ no longer takes
(`theta-provenance`), `theta-not-registered`, and two of a field of another JSON type than the string or null the check
compares it with (`TYPED`): the cited entry's root a number (`theta-root`) and a record's `previous` neither its
predecessor's digest nor null (`trusted`).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path

import verity.ml.circuits
from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value, type_from_json

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("difftest_vectors")
reference = vectors.reference

COUNT = 1200
CODES = ["theta-provenance", "trusted", "theta-record-unknown", "theta-not-at-deployment-start", "theta-root",
         "theta-unbound", "theta-width"]
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
# pinned breaks of a field's JSON type, and the code each is refused with
TYPED = [("root-type", "theta-root"), ("previous-type", "trusted")]


def _hex(rng) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(128))


def declaration(rng, seed: int) -> tuple[str, list]:
    """A declaration's Program as canonical descriptor text, and its program function's parameters as JSON."""
    body, ret = vectors.shape(rng, rng.choice(vectors.SHAPES), seed)
    fn = vectors.composite(f"NciBase{seed}", body, ret)
    names = [n for n, _ in fn.signature()[0]]
    registered = [n for n in names if rng.random() < 0.6] or names[:1]
    text = codec.canonical_json(codec.encode_program(Program(fn, registered=tuple(registered)))).decode()
    return text, [[n, t.to_json()] for n, t in fn.signature()[0]]


def hand() -> tuple[str, list]:
    """A declaration worked by hand: a: Value<8> (registered), b: Value<16> (bound per run), c: Array<4, Value<32>>
    (registered); d = f(a, b, c[0]): Value<4>; e = f(d, c[3]): Value<1>. Root nodes 0 a, 1 b, 2 c (4 gates), 3 d, 4 e."""
    f1, f2 = vectors.prim((8, 16, 32), 4), vectors.prim((4, 32), 1)

    def body(B, S, a, b, c):
        d = B.call(f1, a, b, c[0])
        return B.call(f2, d, c[3])
    params = (("a", Value(8)), ("b", Value(16)), ("c", Array(4, Value(32))))
    fn = bind(CompositeDefinition("HandBase", 1, (), lambda S: (params, Value(1)), body, register=False))
    text = codec.canonical_json(codec.encode_program(Program(fn, registered=("a", "c")))).decode()
    return text, [[n, t.to_json()] for n, t in params]


def _divisor(rng, n: int) -> int:
    return rng.choice([d for d in range(1, n + 1) if n % d == 0])


def _entry(value: str, root: str, rows: int, words: int, bits: int, domain: str) -> dict:
    return {"value": value, "root": root, "leaves": rows, "words": words, "domain": domain,
            **({"word_bits": bits} if bits != 16 else {})}


def build(rng, text: str, params: list, theta_pick, bad: str | None, run_at: int | None = None) -> dict:
    """A case on the declaration `text`: θ the registered input nodes `theta_pick` chooses, its provenance and the trusted
    input, the run's own record at position `run_at` of the log (0 to 3 at random when None), broken as `bad` names
    (None: not broken; "output": one entry a port of the log's first record under the cited root and the run's rows, in
    the `output` form the check once took). Returns the case with what it was made from."""
    desc = json.loads(text)
    program = codec.decode_program(desc, reference.declared(desc))
    typed = [(n, type_from_json(t)) for n, t in params]
    own = reference.owners(program, typed)
    assert own is not None, "the parameters lower onto the program's input nodes"
    inp, reg, spans = reference.root_nodes(text)
    nodes = program.root.body.nodes
    counts = [0] * len(params)
    for k, i in enumerate(own):
        if i is not None:
            counts[i] += spans[k][1] - spans[k][0]
    theta = theta_pick([k for k in range(len(nodes)) if reg[k] and spans[k][1] > spans[k][0]])
    owner = ["input" if inp[k] else 0 for k in range(len(nodes))]
    inst = {"program": text, "theta": theta, "owner": owner, "gamma": program.root.gates,
            "alpha": [64 * program.root.gates + 1, 1]}

    run_at = rng.randint(0, 3) if run_at is None else run_at
    after = rng.randint(0, 1)
    recs = [{"format": "verity/registration/v2", "protocol": PROTOCOL, "program": _hex(rng),
             "window": {"kind": "served", "n": i}, "registered": [], "roots": []} for i in range(run_at + 1 + after)]
    recs[run_at]["program"] = codec.program_digest(desc)
    for r in recs:
        for _ in range(rng.randint(0, 2)):
            r["registered"].append(_entry(f"x{rng.randrange(9)}", _hex(rng), 1, 1, 16, _hex(rng)))
        for _ in range(rng.randint(0, 2)):
            r["roots"].append({"port": f"q{rng.randrange(9)}", "root": _hex(rng), "leaves": rng.randint(1, 8),
                               "domain": _hex(rng)})
    cited: dict[int, dict] = {}
    for i in sorted({own[k] for k in theta}):
        name = params[i][0]
        k0 = next(k for k in theta if own[k] == i)
        bits = 16 * -(-nodes[k0].fn.ret.w // 16)
        root, words = _hex(rng), _divisor(rng, counts[i])
        value = name if rng.random() < 0.7 else f"w{i}"
        rows = counts[i] // words
        recs[0]["registered"].append(_entry(value, root, rows, words, bits, _hex(rng)))
        cited[i] = {"value": value, "root": root, "rows": rows, "words": words, "bits": bits}
        if run_at != 0 or value != name:
            recs[run_at]["registered"].append(_entry(name, root, rows, words, bits, _hex(rng)))
    for i, (name, _) in enumerate(params):
        if i not in cited and any(own[k] == i and reg[k] for k in range(len(nodes))):
            recs[run_at]["registered"].append(_entry(name, _hex(rng), 1, max(counts[i], 1), 16, _hex(rng)))
    for r in recs:
        rng.shuffle(r["registered"])
        rng.shuffle(r["roots"])

    entries = [cited[own[k]] for k in theta]
    target = rng.randrange(len(theta)) if theta else None
    i_t = own[theta[target]] if theta else None
    later = None
    # breaks of a record's contents come before the chain, so the chain still holds
    if bad == "theta-width" and theta:
        c = cited[i_t]
        e = next(e for e in recs[0]["registered"] if e["value"] == c["value"] and e["root"] == c["root"])
        if rng.random() < 0.5:
            e["leaves"] += 1
        elif "word_bits" in e:
            del e["word_bits"]
        else:
            e["word_bits"] = 32
    elif bad == "output" and theta:
        c = cited[i_t]
        recs[0]["roots"].append({"port": f"out{i_t}", "root": c["root"], "leaves": c["rows"], "domain": _hex(rng)})
    elif bad == "theta-not-at-deployment-start" and theta and len(recs) > 1:
        # a record after the first registers the cited value as the first does: only the record's position is wrong
        c = cited[i_t]
        later = rng.randrange(1, run_at) if run_at > 1 else rng.randrange(1, len(recs))
        if not any(e["value"] == c["value"] and e["root"] == c["root"] for e in recs[later]["registered"]):
            recs[later]["registered"].append(_entry(c["value"], c["root"], c["rows"], c["words"], c["bits"], _hex(rng)))
    elif bad == "theta-unbound" and theta:
        # when the run's record is the first and registers the cited value under the parameter's own name, the run's
        # entry is the cited one, and only the program can be wrong without breaking theta-root first
        shared = run_at == 0 and cited[i_t]["value"] == params[i_t][0]
        how = "program" if shared else rng.choice(["root", "program", "missing"])
        mine = [e for e in recs[run_at]["registered"] if e["value"] == params[i_t][0]]
        if how == "root":
            mine[-1]["root"] = _hex(rng)
        elif how == "program":
            recs[run_at]["program"] = _hex(rng)
        else:
            recs[run_at]["registered"] = [e for e in recs[run_at]["registered"] if e["value"] != params[i_t][0]]
    elif bad == "theta-root" and theta and rng.random() < 0.5:
        c = cited[i_t]
        recs[0]["registered"] = [e for e in recs[0]["registered"] if e["value"] != c["value"]]
    elif bad == "root-type" and theta:
        # the cited entry's root a JSON number: the check compares it with the cited string as a string (`strOf?`)
        c = cited[i_t]
        e = next(e for e in recs[0]["registered"] if e["value"] == c["value"] and e["root"] == c["root"])
        e["root"] = int(c["root"][:12], 16)
    prev = None
    digests = []
    for r in recs:
        r["previous"] = prev
        prev = reference.record_digest(r)
        digests.append(prev)

    inst["provenance"] = [{"kind": "registered", "registration": digests[0], "value": c["value"], "root": c["root"]}
                          for c in entries]
    trusted = {"log": recs, "run": digests[run_at], "params": copy.deepcopy(params)}

    p = inst["provenance"][target] if theta else None
    if bad == "theta-provenance":
        how = rng.choice(["missing", "short", "kind", "extra", "type"] if theta else ["missing", "long"])
        if how == "missing":
            del inst["provenance"]
        elif how == "short":
            inst["provenance"] = inst["provenance"][:-1]
        elif how == "long":
            inst["provenance"] = [{"kind": "registered", "registration": digests[0], "value": "x", "root": _hex(rng)}]
        elif how == "kind":
            p["kind"] = rng.choice(["output", "weights", 1])
        elif how == "extra":
            p["extra"] = 1
        else:
            p["root"] = 7
    elif bad == "output" and theta:
        c = cited[i_t]
        inst["provenance"][target] = {"kind": "output", "record": digests[0], "port": f"out{i_t}", "root": c["root"]}
    elif bad == "trusted":
        how = rng.choice(["chain", "run", "keys", "params", "type"])
        if how == "chain":
            trusted["log"][rng.randrange(len(recs))]["previous"] = _hex(rng)
        elif how == "run":
            trusted["run"] = _hex(rng)
        elif how == "keys":
            trusted["extra"] = []
        elif how == "params" and len(params) > 0:
            trusted["params"] = trusted["params"] + [trusted["params"][0]]
        else:
            trusted["params"] = [[n, ["Value", 0]] for n, _ in params] or [["z", ["Value", 0]]]
    elif bad == "previous-type":
        # the first record's `previous` the empty string rather than null, or a later one's its digest in a list
        n = rng.randrange(len(recs))
        trusted["log"][n]["previous"] = "" if n == 0 else [trusted["log"][n]["previous"]]
    elif bad == "theta-record-unknown" and theta:
        p["registration"] = _hex(rng)
    elif bad == "theta-not-at-deployment-start" and later is not None:
        p["registration"] = digests[later]
    elif bad == "theta-root" and theta:
        p["root"] = _hex(rng)
    return {"case": {"instance": inst, "trusted": trusted}, "malformed": bad, "theta": len(theta)}


def case(seed: int) -> dict:
    rng = random.Random(seed)
    text, params = declaration(rng, seed)
    bad = rng.choice(CODES) if rng.random() < 0.45 else None

    def pick(cands: list[int]) -> list[int]:
        chosen = [k for k in cands if rng.random() < 0.6] or cands[:1]
        return rng.sample(chosen, len(chosen))
    return build(rng, text, params, pick, bad)


def pinned() -> list[dict]:
    """On `hand()`, θ = [a, c]. Two cases accepted, each citing the log's first record: from the deployment's first run,
    whose own record that is, and from its third. Then one per code, each the first broken one of its kind the seeds give,
    `theta-not-at-deployment-start` from the third run citing the second record, which registers the same value under the
    same root before the run's own (a registration mid-deployment); one entry of the `output` kind from the third run
    (`theta-provenance`); θ holding b, bound per run (`theta-not-registered`); and `TYPED`'s, from the third run."""
    text, params = hand()
    out = []
    plan = [(None, None, 0), (None, None, 2),
            *((c, c, 2 if c == "theta-not-at-deployment-start" else None) for c in CODES),
            ("output", "theta-provenance", 2)]
    for bad, code, at in plan:
        for seed in range(1000):
            rng = random.Random(10_000 + seed)
            made = build(rng, text, params, lambda c: c, bad, at)
            if reference.check_baseline(made["case"]["instance"], made["case"]["trusted"]).get("error") == code:
                out.append(made | {"malformed": code, "seed": seed})
                break
        else:
            raise AssertionError(f"no seed gives a pinned case refused {code}")
    reg = copy.deepcopy(out[0])
    inst = reg["case"]["instance"]
    inst["theta"] = [0, 1]
    inst["provenance"] = [inst["provenance"][0], inst["provenance"][0]]
    out.append(reg | {"malformed": "theta-not-registered"})
    for bad, code in TYPED:
        for seed in range(1000):
            made = build(random.Random(20_000 + seed), text, params, lambda c: c, bad, 2)
            if reference.check_baseline(made["case"]["instance"], made["case"]["trusted"]).get("error") == code:
                out.append(made | {"malformed": code, "seed": seed})
                break
        else:
            raise AssertionError(f"no seed gives a pinned {bad} case refused {code}")
    return out


def total(count: int = COUNT) -> int:
    return count + len(pinned())


def expanded(c: dict):
    """The case gate by gate, or None where the policy check refuses it or its provenance is not one per θ entry."""
    inst = c["instance"]
    if "error" in reference.check_desc(inst):
        return None
    prov = inst.get("provenance")
    if not (isinstance(prov, list) and len(prov) == len(inst["theta"])):
        return None
    return {"instance": reference.expand_baseline(inst), "trusted": c["trusted"]}


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """The three files for cases 0 … count − 1, then `pinned()`'s, under ``out/baseline_difftest``."""
    d = out / "baseline_difftest"
    d.mkdir(parents=True, exist_ok=True)
    names = ("cases", "expanded", "python")
    paths = {f"baseline_difftest/{n}.jsonl": d / f"{n}.jsonl" for n in names}
    cases = [*(case(seed)["case"] for seed in range(count)), *(p["case"] for p in pinned())]
    with open(d / "cases.jsonl", "w") as fc, open(d / "expanded.jsonl", "w") as fe, open(d / "python.jsonl", "w") as fp:
        for c in cases:
            fc.write(line(c))
            fe.write(line(expanded(c)))
            fp.write(line(reference.check_baseline(c["instance"], c["trusted"])))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
