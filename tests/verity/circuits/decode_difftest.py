"""The program decoder's difftest data (PROTOCOL.md §4): descriptors and the Python reference's verdict on each, for the Lean
decoder (`Circuits/Program.lean`'s `decode`).

    uv run python tests/verity/circuits/decode_difftest.py --out DIR

Writes `DIR/decode_difftest/`, one line per case: `cases.jsonl`, `{"name", "descriptor"}`, the descriptor's text as the
decoder reads it; `python.jsonl`, `{"name", "ok", "error", "message", "program_digest", "boolean"}`, `codec.decode_program`'s
verdict on that text: on a refusal its first `E_` code (none for a refusal without one) and its message, on an accept the
program's digest and its Boolean map; and `digests.json`, each file's SHA-256.

The cases: `format_vectors.json`'s descriptors (the format's, `wide`'s and `registered_program`'s), each of
`test_format_spec.REJECT`'s refusals of the first, a `boolean` key named twice (a JSON object keeps a key's last value,
in both decoders' parsers), `chain()`, a GEMM chain whose word primitives' bodies are the catalog's, and
`recut_vectors.reads()`, reads of an input node at another width, `test_codec_refusals.DECODER`'s descriptors and
`FORWARD`'s controls, and `decode_refusals.CASES`, the refusals of the Lean decoder no other case reaches."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import decode_refusals
import recut_vectors
import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
import test_codec_refusals as refusals
import test_format_spec as spec

from verity.circuits import codec
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


def cases() -> list[tuple[str, str, object]]:
    base = VECTORS["descriptor"]
    out = [("format", base, spec.REG), ("wide", VECTORS["wide"]["descriptor"], spec.REG),
           ("registered_program", VECTORS["registered_program"]["descriptor"], spec.REG)]
    for label, edit, _why in spec.REJECT:
        d = json.loads(base)
        edit(d)
        out.append((f"reject: {label}", json.dumps(d, ensure_ascii=False, separators=(",", ":")), spec.REG))
    first = json.loads(base)["boolean"]["TSum2_v1"]
    out.append(("a boolean key named twice, the last kept", base.replace('"boolean":{', f'"boolean":{{"TAdd32_v1":"{first}",', 1), spec.REG))
    out.append(("chain", chain(), REGISTRY))
    out += [(f"a constant bit as a word's Boolean body{', inside' if inside else ''}", json.dumps(d, separators=(",", ":")), reg)
            for inside in (False, True) for d, reg in [refusals.bit_word("const", inside)]]
    out += [(name, text, spec.REG) for name, text in recut_vectors.reads()]
    out += refusal_cases()
    out += decode_refusals.cases()
    return out


def refusal_cases() -> list[tuple[str, str, object]]:
    """`test_codec_refusals.DECODER`'s descriptors, each the smallest that trips one refusal of `codec.py`'s decoder, and
    `test_codec_refusals.FORWARD`'s controls, each forward reference's read in honest order."""
    out = []
    for label, make, edit, _text in refusals.DECODER:
        d, registry = make()
        edit(d)
        out.append((f"refusal: {label}", json.dumps(d, ensure_ascii=False, separators=(",", ":")), registry))
    for label, make, honest, _forward, _outcome in refusals.FORWARD:
        d, registry = make()
        honest(d)
        out.append((f"control: {label}", json.dumps(d, ensure_ascii=False, separators=(",", ":")), registry))
    return out


def verdict(text: str, registry) -> dict:
    try:
        p = codec.decode_program(json.loads(text), registry)
    except Exception as e:  # noqa: BLE001
        m = re.search(r"\bE_[A-Z_]+", str(e))
        return {"ok": False, "error": m[0] if m else None, "message": f"{type(e).__name__}: {e}"}
    return {"ok": True, "program_digest": codec.program_digest(p.desc), "boolean": p.desc["boolean"]}


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
