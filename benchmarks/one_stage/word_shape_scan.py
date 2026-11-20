"""The word-parameter scan of a circuit's descriptor: every Definition reachable from the root's templates whose parameter
or return type is not all bits (Lean's `Ty.bits`) and that reaches no word primitive, transitively (`Flock.Lift.closure`'s
`wordy` over `isWordPrim`), on the descriptor JSON as Lean reads it. Such a Definition `Flock.Lift.lift` leaves as it is.
Its second scan, ``unaligned``: every template (a Definition the root calls) with a parameter, or a return field
(`fieldsOf`), whose bits are not a multiple of 16 and which another parameter or field follows, which `lowerBlock` lays
in whole 16-bit words while `Lower.build` places the leaves one after another; a cut port's rows (`ioCuts`) are not
read. ``python word_shape_scan.py C.json ...`` prints one ``SCAN`` line per file, one ``HIT`` line per Definition of
the first scan and one ``UNALIGNED`` line per template of the second."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Any


def width_after(prefix: str, fid: str) -> int | None:
    m = re.fullmatch(re.escape(prefix) + r"([1-9][0-9]*)_v1", fid)
    return int(m.group(1)) if m else None


def is_input_id(fid: str) -> bool:
    return any(width_after(p, fid) is not None for p in ("Input", "RegisteredInput", "SuppliedInput"))


def gate_kind(fid: str) -> bool:
    if fid in ("Not_v1", "Const1[0x0]_v1", "Const1[0x1]_v1"):
        return True
    for p in ("And", "Xor"):
        n = width_after(p, fid)
        if n is not None:
            return n >= 2
    return False


def is_boolean_prim(fid: str) -> bool:
    return gate_kind(fid) or fid in ("Input1_v1", "RegisteredInput1_v1", "SuppliedInput1_v1")


def is_word_prim(prim: bool, fid: str) -> bool:
    return prim and not is_input_id(fid) and not is_boolean_prim(fid)


def bits(t: Any) -> bool:
    tag = t[0]
    if tag == "Value":
        return t[1] == 1
    if tag == "Array":
        return bits(t[2])
    if tag == "Tuple":
        return all(bits(x) for x in t[1])
    if tag == "Record":
        return all(bits(x[1]) for x in t[1])
    raise ValueError(f"type {t}")


def width(t: Any) -> int:
    tag = t[0]
    if tag == "Value":
        return t[1]
    if tag == "Array":
        return t[1] * width(t[2])
    if tag == "Tuple":
        return sum(width(x) for x in t[1])
    if tag == "Record":
        return sum(width(x[1]) for x in t[1])
    raise ValueError(f"type {t}")


def fields(t: Any) -> list:
    if t[0] == "Tuple":
        return list(t[1])
    if t[0] == "Record":
        return [x[1] for x in t[1]]
    return [t]


def unaligned_of(params: list, ret: Any) -> dict | None:
    """The parameters' and return fields' bits where one not a multiple of 16 is followed by another; else None."""
    pb, fb = [width(t) for t in params], [width(f) for f in fields(ret)]
    bad = lambda bs: any(b % 16 for b in bs[:-1])
    return {"param_bits": pb, "field_bits": fb} if bad(pb) or bad(fb) else None


def scan(desc: dict, name: str) -> dict:
    defs = desc["definitions"]
    callees = desc.get("callees")
    types = desc.get("types")

    def fn_of(f: Any) -> str:
        if isinstance(f, int):
            return callees[f]
        if isinstance(f, dict):
            return f["fn"]
        return f

    def ty(t: Any) -> Any:
        return types[t] if isinstance(t, int) and types is not None else t

    def sig(d: dict) -> tuple[list, Any]:
        if d["form"] == "primitive":
            return [ty(t) for _, t in d["params"]], ty(d["ret"])
        b = d["body"]
        return [ty(t) for _, t in b["params"]], ty(b["ret_type"])

    calls = {fid: ([] if d["form"] == "primitive" else sorted({fn_of(n["fn"]) for n in d["body"]["nodes"]}))
             for fid, d in defs.items()}
    wordy: dict[str, bool] = {}

    def w(fid: str) -> bool:
        stack = [(fid, False)]
        while stack:
            f, done = stack.pop()
            if f in wordy:
                continue
            d = defs[f]
            if d["form"] == "primitive":
                wordy[f] = is_word_prim(True, f)
                continue
            if done:
                wordy[f] = any(wordy[c] for c in calls[f])
                continue
            stack.append((f, True))
            stack.extend((c, False) for c in calls[f] if c not in wordy)
        return wordy[fid]

    root = desc["root"]
    templates = calls[root]
    reach, todo = set(), list(templates)
    while todo:
        f = todo.pop()
        if f in reach:
            continue
        reach.add(f)
        todo.extend(calls[f])
    hit = []
    for f in sorted(reach):
        d = defs[f]
        if d["form"] == "primitive":
            continue
        ps, r = sig(d)
        if (not all(bits(p) for p in ps) or not bits(r)) and not w(f):
            hit.append({"id": f, "template": f in templates, "params": ps, "ret": r})
    unaligned = [{"id": f, **u} for f in templates if defs[f]["form"] != "primitive"
                 for u in [unaligned_of(*sig(defs[f]))] if u is not None]
    wide_bool = sorted(f for f in reach if defs[f]["form"] == "primitive" and is_boolean_prim(f)
                       and not (all(bits(p) for p in sig(defs[f])[0]) and bits(sig(defs[f])[1])))
    return {"circuit": name, "definitions": len(defs), "reached": len(reach), "templates": templates,
            "composites_reached": sum(defs[f]["form"] != "primitive" for f in reach),
            "shape": len(hit), "shape_templates": sum(h["template"] for h in hit), "hits": hit,
            "unaligned": unaligned,
            "wide_boolean_prims": wide_bool}


COUNTS = ("templates: the distinct Definitions the root's nodes call, input primitives (Input*) among them; "
          "templates_inputs: how many of those are input primitives; definitions: every Definition the descriptor "
          "holds; reached: those reachable from the templates; shape: reached Definitions of the first scan's shape; "
          "unaligned: templates of the second scan's")


def report(desc: dict, name: str, raw: bytes | None = None) -> dict:
    """`scan`'s line, after the description's: its ``program_sha512``, the digest a registration and a partition
    object name it by (`codec.program_digest`), and ``raw``'s sha256 and size, the file's."""
    from verity.circuits.codec import program_digest
    out = scan(desc, name)
    print("DESCRIPTION " + json.dumps({"circuit": name, "program_sha512": program_digest(desc)}
                                      | ({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                                         if raw is not None else {})), flush=True)
    print("COUNTS " + COUNTS, flush=True)
    print("SCAN " + json.dumps({k: v for k, v in out.items() if k not in ("hits", "unaligned")}
                               | {"templates_count": len(out["templates"]),
                                  "templates_inputs": sum(t.startswith("Input") for t in out["templates"]),
                                  "hit_ids": [h["id"] for h in out["hits"]], "unaligned": len(out["unaligned"]),
                                  "unaligned_ids": [u["id"] for u in out["unaligned"]]}), flush=True)
    for h in out["hits"][:200]:
        print("HIT " + json.dumps({"circuit": name, **h}), flush=True)
    for u in out["unaligned"][:200]:
        print("UNALIGNED " + json.dumps({"circuit": name, **u}), flush=True)
    return out


def scan_program(program: Any, name: str) -> dict:
    """`report` of ``program``'s descriptor (`codec.descriptor_chunks`)."""
    from verity.circuits.codec import descriptor_chunks
    raw = b"".join(descriptor_chunks(program))
    return report(json.loads(raw), name, raw)


def selftest() -> None:
    """The smallest case of the shape, C(a : Value 32, b : Value 1) = Not b, found, and a word one that reaches Add32 not."""
    v = lambda w: ["Value", w]
    desc = {"root": "R", "definitions": {
        "R": {"form": "composite", "body": {"params": [], "ret_type": v(1), "nodes": [{"fn": "C"}, {"fn": "D"}]}},
        "C": {"form": "composite", "body": {"params": [["a", v(32)], ["b", v(1)]], "ret_type": v(1), "nodes": [{"fn": "Not_v1"}]}},
        "D": {"form": "composite", "body": {"params": [["a", v(32)]], "ret_type": v(32), "nodes": [{"fn": "Add32_v1"}]}},
        "Not_v1": {"form": "primitive", "params": [["x", v(1)]], "ret": v(1)},
        "Add32_v1": {"form": "primitive", "params": [["x", v(32)], ["y", v(32)]], "ret": v(32)}}}
    out = scan(desc, "selftest")
    assert [h["id"] for h in out["hits"]] == ["C"], out
    print("SELFTEST ok: C(a : Value 32, b : Value 1) = Not b is found, D (reaches Add32_v1) is not", flush=True)
    bit3 = ["Array", 3, v(1)]
    desc = {"root": "R", "definitions": {
        "R": {"form": "composite", "body": {"params": [], "ret_type": v(1), "nodes": [{"fn": "E"}, {"fn": "F"}, {"fn": "G"}]}},
        "E": {"form": "composite", "body": {"params": [["x", bit3], ["y", v(1)]], "ret_type": ["Tuple", [bit3, v(1)]],
                                            "nodes": [{"fn": "Not_v1"}]}},
        "F": {"form": "composite", "body": {"params": [["x", ["Array", 16, v(1)]], ["y", bit3]],
                                            "ret_type": ["Tuple", [["Array", 16, v(1)], bit3]], "nodes": [{"fn": "Not_v1"}]}},
        "G": {"form": "composite", "body": {"params": [["x", ["Array", 16, v(1)]], ["y", v(1)]], "ret_type": bit3,
                                            "nodes": [{"fn": "Not_v1"}]}},
        "Not_v1": {"form": "primitive", "params": [["x", v(1)]], "ret": v(1)}}}
    out = scan(desc, "selftest-unaligned")
    assert out["unaligned"] == [{"id": "E", "param_bits": [3, 1], "field_bits": [3, 1]}], out["unaligned"]
    print("SELFTEST ok: E(x : Array 3 (Value 1), y : Value 1) = (x, Not y) is unaligned, F and G (each short field "
          "last) are not", flush=True)


if __name__ == "__main__":
    selftest()
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            raw = f.read()
        report(json.loads(raw), path, raw)
