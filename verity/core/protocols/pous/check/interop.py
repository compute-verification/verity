"""Byte-for-byte interop of the two implementations of `p2-audit-wire/v4`: the Python reference
`benchmarks/pous/p2_v1/wire.py` and the Lean `pous_check` (`PousCheck.Wire`, `PousCheck.Audit.check`).

    uv run python verity/core/protocols/pous/check/interop.py --all [--out FILE]

The transcripts are every vector the P2 vectors files (`benchmarks/pous/p2_v1/`) hold:

* every transcript of `audit-negative-vectors.json`, `audit-vectors.json`'s two-round end-to-end one, and its full-width
  response in a one-block setup, public and private (`scripts/difftest_vectors.py` builds them, and checks both files against
  `reference.py`'s SHA-512), the verifier holding its own root of the private setup. Each side gives its setup's outputs (the epoch id,
  the declared root and the identity it binds, how many blocks it carried, its failure code, where the audit begins) and
  its audit's (the `FINAL` frame, its verdict, code, record count and transcript hash), or the fault where it stops, its
  code and message, or the refusal of the verifier's inputs: `wire._parse_setup`, `wire.verify_audit` and
  `wire.parse_final` on one side, `pous-audit-difftest serve` on the other;
* `vectors.json`'s vectors and σ rejection cases: each entry's hash and whether `reference.ENTRY_SHA512` pins it, and its keys and decode
  (`reference.keys`, `decode`, `sigma` on one side, `pous-setup-difftest`'s `vector` and `sigma` on the other).

Both sides write canonical JSON, compared byte for byte. The script prints each mismatch with both outputs, then
`N transcripts, M mismatches`, and exits 0 iff M is 0 and N is every vector the files hold. `--out` writes one line per
transcript with both outputs.

Lean implements the verifier only: the setup and response frames both read are the vectors', which Python's prover wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from verity_difftest import Server, build, canonical

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def _generator():
    spec = importlib.util.spec_from_file_location("pous_difftest_vectors", HERE / "scripts" / "difftest_vectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fault(e) -> dict:
    return {"fault": e.code, "message": str(e)}


def transcript(gen, wire, inst: dict) -> dict:
    """`wire.py`'s outputs on a transcript, in the form of `PousCheck.Difftest.transcript`'s."""
    frames = [bytes.fromhex(f) for f in inst["frames"]]
    held = gen.verifier(inst)
    try:
        s, cursor = wire._parse_setup(frames, held["private_root"])
        setup = {"blocks": len(s.blocks), "commitment": s.commitment.hex(), "cursor": cursor, "epoch_id": s.epoch_id.hex(),
                 "failure": s.failure, "manifest": s.manifest.hex(), "root": s.root.hex(), "salt": s.salt.hex()}
    except wire.WireError as e:
        setup = _fault(e)
    except ValueError as e:
        setup = {"refused": str(e)}
    try:
        final = wire.verify_audit(frames, inst["elapsed_ns"], **held)
        verdict, code, records = wire.parse_final(final)
        o = 10 + wire.D + wire.S + 1 + 1 + 2 + 8 + 8
        audit = {"code": code, "final": final.hex(), "records": records, "transcript_hash": final[o:o + wire.D].hex(),
                 "verdict": verdict}
    except wire.WireError as e:
        audit = _fault(e)
    except ValueError as e:
        audit = {"refused": str(e)}
    return {"audit": audit, "setup": setup}


def vector(reference, pins: dict[int, str], v: dict) -> dict:
    """`reference.py`'s outputs on a `vectors.json` vector, in the form of `PousCheck.SetupDifftest.vector`'s."""
    q = reference.vector_params(v["w"], v["c0"])
    n = q.width_bytes
    entry = hashlib.sha512(canonical({k: x for k, x in v.items() if k != "sha512"}).encode()).hexdigest()
    k, t, attempt = reference.keys(bytes.fromhex(v["setup_salt"]), q, bytes.fromhex(v["model_commitment"]), v["segment"],
                                   v["block"])
    c = int(v["c"], 16)
    m = reference.decode(q, c, k, t)
    return {"K": k.to_bytes(n, "big").hex(), "canonical": c < q.p, "decoded_m": m.to_bytes(n, "big").hex(),
            "fits": m < 1 << 8 * q.payload_bytes, "pinned": pins.get(v["w"]) == entry, "sha512": entry,
            "t": t.to_bytes(n, "big").hex(), "t_rejection_index": attempt}


def sigma(reference, v: dict) -> dict:
    """`reference.py`'s outputs on a `vectors.json` σ rejection case, in the form of `PousCheck.SetupDifftest.sigma`'s."""
    q = reference.vector_params(v["w"], v["c0"])
    n, r, k = q.width_bytes, int(v["r"], 16), int(v["K"], 16)
    return {"p": q.p.to_bytes(n, "big").hex(), "r_xor_K": (r ^ k).to_bytes(n, "big").hex(), "rejects": r ^ k >= q.p,
            "sigma": reference.sigma(q, r, k).to_bytes(n, "big").hex()}


def _python_side(fn, *args) -> str:
    try:
        return canonical(fn(*args))
    except Exception as e:
        return canonical({"exception": f"{type(e).__name__}: {e}"})


def _lean_side(server: Server, fn: str, value) -> str:
    kind, text = server.call(fn, value)
    return text if kind == "output" else canonical({"reject": text})


def compare() -> tuple[list[dict], int]:
    """Every transcript with both outputs, and how many vectors the files hold."""
    gen = _generator()
    reference, wire, _ = py = gen._python(ROOT)
    doc = json.loads(reference.VECTORS.read_bytes())
    if hashlib.sha512(reference.VECTORS.read_bytes()).hexdigest() != reference.VECTORS_SHA512:
        raise SystemExit("vectors.json is not the file reference.VECTORS_SHA512 pins")
    negative = json.loads(reference.NEGATIVE_VECTORS.read_bytes())
    expected = negative["case_count"] + 1 + 2 + len(doc["vectors"]) + len(doc["sigma_rejection_vectors"])
    pins, out = reference.ENTRY_SHA512, []
    with Server([str(build(HERE, "pous-audit-difftest")), "serve"]) as lean:
        for inst in gen._vector_audits(py):
            out.append({"id": inst["name"], "kind": "transcript", "python": _python_side(transcript, gen, wire, inst),
                        "lean": _lean_side(lean, "transcript", inst)})
    with Server([str(build(HERE, "pous-setup-difftest"))]) as lean:
        for v in doc["vectors"]:
            out.append({"id": v["id"], "kind": "vector", "python": _python_side(vector, reference, pins, v),
                        "lean": _lean_side(lean, "vector", v)})
        for v in doc["sigma_rejection_vectors"]:
            out.append({"id": v["id"], "kind": "sigma", "python": _python_side(sigma, reference, v),
                        "lean": _lean_side(lean, "sigma", v)})
    return out, expected


def report(results: list[dict], expected: int, out: Path | None = None) -> int:
    """Print each mismatch and the count, write `out`; the exit code."""
    mismatches = [r for r in results if r["python"] != r["lean"]]
    for r in mismatches:
        print(f"MISMATCH {r['kind']} {r['id']}\n  python {r['python']}\n  lean   {r['lean']}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(canonical(r | {"match": r["python"] == r["lean"]}) + "\n" for r in results))
    print(f"{len(results)} transcripts, {len(mismatches)} mismatches")
    if len(results) != expected:
        print(f"expected {expected} transcripts, every vector of the vectors files")
        return 1
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", required=True, help="every vector of the vectors files")
    ap.add_argument("--out", type=Path, help="write one JSON line per transcript with both outputs")
    a = ap.parse_args(argv)
    return report(*compare(), a.out)


if __name__ == "__main__":
    sys.exit(main())
