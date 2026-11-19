"""The program decoder's difftest data (PROTOCOL.md §4): descriptors and the Python reference's verdict on each, for the Lean
decoder (`Circuits/Descriptor.lean`'s `decode`).

    uv run python tests/verity/circuits/decode_difftest.py --out DIR

Writes `DIR/decode_difftest/`, one line per case: `cases.jsonl`, `{"name", "descriptor"}`, the descriptor's text as the
decoder reads it; `python.jsonl`, `{"name", "ok", "error", "message", "program_digest", "boolean", "inputs",
"definition_digests"}`, `codec.decode_program`'s verdict on that text: on a refusal its first `E_` code (none for a refusal
without one) and its message, on an accept the program's digest, its Boolean map, its `inputs` and each definition's digest
(§4.8); and `digests.json`, each file's SHA-256.

The cases: `format_vectors.json`'s descriptors (the format's, `wide`'s and `registered_program`'s), each of
`test_format_spec.REJECT`'s refusals of the first, a `boolean` key named twice (a JSON object keeps a key's last value,
in both decoders' parsers), `chain()`, a GEMM chain whose word primitives' bodies are the catalog's, and
`recut_vectors.reads()`, reads of an input node at another width, and `supplied_cases()`, a supplied input's mark (§5.4)
as a program and a block carry it and each of its refusals."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import recut_vectors
import test_supplied_block as SB
import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
import test_format_spec as spec

from verity.circuits import codec
from verity.circuits import restrict as R
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.ops import gemm as G
from verity.commitments.periods import interface as I

VECTORS = json.loads((Path(__file__).parent / "format_vectors.json").read_text())


def chain() -> str:
    gemm = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
    gid, n = codec.descriptor_id(gemm), 8
    wires = (I.Wire(gid, "x", gid, "o0", at=2, radix=(n - 2,), base=0, step=(1,)),)
    c = I.Circuit("chain", ((gemm, n),), external=(f"{gid}/x",), wires=wires, order=I.ordered({gid: n}, wires, {gid: range(0, n, 2)}))
    return b"".join(codec.descriptor_chunks(c.program()[0])).decode()


def supplied_cases() -> list[tuple[str, str]]:
    """`test_supplied_block`'s program (`k` supplied, `const.uint/v1`), the block reading part of `k` (a mark with `of`), and
    the program with its mark edited to each refusal, in `check_entry`'s order."""
    c = b"".join(codec.descriptor_chunks(SB.program()))
    block = R.restrict(c, SB.PARTITION, [(1, 2)], R.Bits(R.program(c))).descriptor
    base = json.loads(c)
    mark = base["inputs"][1][2]
    edits = [("mark: bits 0", lambda i: i[1].__setitem__(2, {**mark, "bits": 0})),
             ("mark: a key beyond the four", lambda i: i[1].__setitem__(2, {**mark, "seed": 1})),
             ("mark: of with lo = hi", lambda i: i[1].__setitem__(2, {**mark, "of": ["k", 1, 1]})),
             ("supplied nodes, no mark", lambda i: i[1].pop()),
             ("a mark on run nodes", lambda i: i[0].append(mark)),
             ("mark: no such function", lambda i: i[1].__setitem__(2, {**mark, "supplied": "const.uint/v2"})),
             ("mark: the wrong args", lambda i: i[1].__setitem__(2, {**mark, "args": {"round": 5}})),
             ("mark: no value at its args", lambda i: i[1].__setitem__(2, {**mark, "args": {"bits": 16, "value": 70000}})),
             ("mark: bits not its nodes'", lambda i: i[1].__setitem__(2, {**mark, "bits": 8})),
             ("mark: a function of another length", lambda i: i[1].__setitem__(2, {**mark, "supplied": "log.base/v1", "args": {}})),
             ("mark: digest of a 2-byte value", lambda i: i[1].__setitem__(2, {**mark, "layout": "digest"}))]
    edits += [(f"mark: {fn} at {args}", lambda i, fn=fn, args=args: i[1].__setitem__(2, {**mark, "supplied": fn, "args": args}))
              for fn, args in (("const.uint/v1", {"bits": 32, "value": 5}), ("const.uint/v1", {"bits": 32, "value": 1 << 32}),
                               ("const.uint/v1", {"bits": 12, "value": 5}))]
    out = [("supplied: program", c.decode()), ("supplied: block", block.decode() if isinstance(block, bytes) else block)]
    for label, edit in edits:
        d = json.loads(c)
        edit(d["inputs"])
        out.append((f"supplied: {label}", json.dumps(d, ensure_ascii=False, separators=(",", ":"))))
    return out


def cases() -> list[tuple[str, str, object]]:
    base = VECTORS["descriptor"]
    out = [("format", base, spec.REG), ("wide", spec.wide_descriptor(), spec.REG),
           ("registered_program", VECTORS["registered_program"]["descriptor"], spec.REG)]
    for label, edit, _why in spec.REJECT:
        d = json.loads(base)
        edit(d)
        out.append((f"reject: {label}", json.dumps(d, ensure_ascii=False, separators=(",", ":")), spec.REG))
    first = json.loads(base)["boolean"]["TSum2_v1"]
    out.append(("a boolean key named twice, the last kept", base.replace('"boolean":{', f'"boolean":{{"TAdd32_v1":"{first}",', 1), spec.REG))
    out.append(("chain", chain(), REGISTRY))
    out += [(name, text, spec.REG) for name, text in recut_vectors.reads()]
    out += [(name, text, REGISTRY) for name, text in supplied_cases()]
    return out


def verdict(text: str, registry) -> dict:
    try:
        p = codec.decode_program(json.loads(text), registry)
    except Exception as e:  # noqa: BLE001
        m = re.search(r"\bE_[A-Z_]+", str(e))
        return {"ok": False, "error": m[0] if m else None, "message": f"{type(e).__name__}: {e}"}
    return {"ok": True, "program_digest": codec.program_digest(p.desc), "boolean": p.desc["boolean"], "inputs": p.inputs,
            "definition_digests": dict(sorted(codec.definition_digests(p.desc).items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    d = ap.parse_args().out / "decode_difftest"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cases.jsonl", "w") as fc, open(d / "python.jsonl", "w") as fp:
        for name, text, registry in cases():
            fc.write(json.dumps({"name": name, "descriptor": text}) + "\n")
            fp.write(json.dumps({"name": name, **verdict(text, registry)}) + "\n")
    digests = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("cases.jsonl", "python.jsonl")}
    (d / "digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    print(json.dumps(digests))


if __name__ == "__main__":
    main()
