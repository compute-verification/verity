"""One real linear layer as one circuit of one template: SmolLM2-135M's layer-0 ``q_proj`` (a BF16 weight [576, 576]) cut
into `Gemm_v2{K=576, N=8}` slices, ``TOKENS`` rows of its real input (the checkpoint's embeddings of fixed token ids
through layer 0's RMSNorm, computed here in float32 and rounded to BF16, not served) against each slice: 288 units, no
wired port between them, C built by circuits' `ops.product` (its blocks of ``CAP`` units, its outputs by C's own
evaluator). The weight is a registered read (its root pinned before the run) or a run input each unit commits
(``--weights committed``), each unit's row a run input.

    one_linear.py window --out DIR --rule RULE [--forge all|U] [--weights registered|committed] [--cache DIR]
    one_linear.py levels
    one_linear.py c --out DIR

``window``: the window committed and written for ``records_audit.py audit`` (as `served_door.py window` writes the staged
one), its sampling design the one ``RULE`` names over its proof blocks; ``--forge all`` changes one output word of every
block's first unit before the commitment (so every drawn block holds a forged unit), ``--forge U`` of unit U alone;
``--weights committed``: ``w`` a run input each unit commits, not the registered read.
``levels``: the door's refusal of this circuit's partition at two levels, before any coin (`one_circuit.block`).
``c``: C's descriptor bytes and its template query, as ``DIR/C.json`` and ``DIR/query.json`` (`door_walls.py census
--circuit DIR`, `word_shape_scan.py DIR/C.json`)."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL, REVISION = "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
WEIGHT, EMBED, NORM, EPS = ("model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight",
                            "model.layers.0.input_layernorm.weight", 1e-5)
TOKENS = (504, 3575, 282, 4649)
K, E, N = 576, 576, 8
SLICES = E // N
UNITS = len(TOKENS) * SLICES
CAP = 2
RUN = "ab" * 64
SCOPE = "one template, no wired port between units"


def product(w: np.ndarray | None = None, x: np.ndarray | None = None, registered: bool = False):
    """Circuits' `ops.product` of the weight [E, K] and the rows x [len(TOKENS), K] (BF16 words; zeros where None,
    which C does not read): ``UNITS`` units in (token, slice) order, blocks of ``CAP``."""
    from verity.circuits.ops import gemm as G, product as PR

    w = np.zeros((E, K), np.uint16) if w is None else w
    x = np.zeros((len(TOKENS), K), np.uint16) if x is None else x
    return PR.product(w, x, k=K, s=N, units=SLICES, block=CAP, k_log=24, g=1, dot=G.GEMM_AMPERE, registered=registered)


def circuit(registered: bool = True):
    """The window's circuit, refused unless its Program is `product`'s C: ``x`` external, ``w`` the registered weight's
    rows, unit (t, j) reading words [j·N·K, (j + 1)·N·K); with ``registered`` false, ``w`` external too."""
    from verity.circuits.codec import descriptor_id, program_digest

    p = product(registered=registered)
    g = p.template
    gid = descriptor_id(g)
    c = _circuit(g, gid, registered)
    if (got := program_digest(c.program()[0])) != p.digest:
        raise SystemExit(f"the window's circuit is not ops.product's C: program_sha512 {got[:16]}…, "
                         f"product's {p.digest[:16]}…")
    return c


def _circuit(g, gid: str, registered: bool):
    from verity.commitments.periods import interface as I

    if not registered:
        return I.Circuit("one_linear", ((g, UNITS),), external=(f"{gid}/x", f"{gid}/w"))
    reg = I.Registered({"w": E * K}, reads=(I.Read(gid, "w", "w", 0, (len(TOKENS), SLICES), 0, (0, N * K)),))
    return I.Circuit("one_linear", ((g, UNITS),), external=(f"{gid}/x",), registered=reg)


def _ranged(url: str, lo: int, hi: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        got = r.read()
    if len(got) != hi - lo:
        raise RuntimeError(f"{url}: bytes [{lo}, {hi}) came back as {len(got)}")
    return got


def checkpoint(cache: Path) -> dict[str, np.ndarray]:
    """The three tensors' BF16 words (u16), read by HTTP range from the checkpoint's safetensors file and kept under
    ``cache``; each file's SHA-512 is printed in the window's STEP line."""
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors"
    head = cache / "header.json"
    if not head.exists():
        n = struct.unpack("<Q", _ranged(url, 0, 8))[0]
        head.write_bytes(_ranged(url, 8, 8 + n))
    meta = json.loads(head.read_bytes())
    base = 8 + head.stat().st_size
    out = {}
    for name in (WEIGHT, NORM):
        t = meta[name]
        if t["dtype"] != "BF16":
            raise RuntimeError(f"{name}: {t['dtype']}, not BF16")
        f = cache / f"{name}.bin"
        if not f.exists():
            lo, hi = t["data_offsets"]
            f.write_bytes(_ranged(url, base + lo, base + hi))
        out[name] = np.frombuffer(f.read_bytes(), "<u2").reshape(t["shape"])
    t = meta[EMBED]
    width = t["shape"][1] * 2
    rows = []
    for tok in TOKENS:
        f = cache / f"{EMBED}.{tok}.bin"
        if not f.exists():
            lo = t["data_offsets"][0] + tok * width
            f.write_bytes(_ranged(url, base + lo, base + lo + width))
        rows.append(np.frombuffer(f.read_bytes(), "<u2"))
    out[EMBED] = np.stack(rows)
    return out


def _f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _bf16(f: np.ndarray) -> np.ndarray:
    u = np.asarray(f, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def values(ck: dict[str, np.ndarray]):
    """`product` at the checkpoint's weight and rows, and the units' ``x`` [UNITS, K], ``w`` [UNITS, N, K] and outputs
    [UNITS, N] (`product`'s, by C's evaluator)."""
    h = _f32(ck[EMBED])
    p = product(ck[WEIGHT], _bf16(h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + EPS) * _f32(ck[NORM])))
    xs = np.repeat(p.x, SLICES, axis=0)
    ws = np.tile(p.weight.reshape(SLICES, N, K), (len(TOKENS), 1, 1))
    return p, xs, ws, np.asarray(p.outputs, np.uint16).reshape(UNITS, N)


def program_id() -> str:
    """The weight's program: the checkpoint and tensor it is, which its pin's domain binds."""
    return hashlib.sha512(f"{MODEL}@{REVISION}/{WEIGHT}".encode()).hexdigest()


def window(a: argparse.Namespace) -> int:
    from verity.commitments import docket_digest as DD, registered_value as RV
    from verity.commitments.periods import interface as IF
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity.protocols.proofs.spot.prover import docket as DK
    from verity.protocols.proofs.spot.verifier import service as S

    t0, out = time.perf_counter(), Path(a.out)
    ck = checkpoint(Path(a.cache))
    p, xs, ws, ys = values(ck)
    if a.forge is not None:
        for u in (range(0, UNITS, CAP) if a.forge == "all" else (int(a.forge),)):
            ys[u, 0] ^= 1
    reg = a.weights == "registered"
    c = circuit(reg)
    weights = RV.Registered.commit(program_id(), "w", ck[WEIGHT].reshape(SLICES, N * K), schema=V2)
    program, query = c.program()
    law = S.params_sampling_design(a.rule, program, query, None)[0]
    win = IF.Window(c, RUN, law, CAP, registered={"w": weights} if reg else None)
    win.supply()
    win.record(p.template, [xs, None if reg else ws], [ys], {"one_linear": 0})
    out.mkdir(parents=True, exist_ok=True)
    win.write(win.close(), out, {"one_linear": "window"}, registrar=DK)
    (out / "pins.json").write_text(json.dumps([{"program": program_id(), "values": {"w": weights.entry()}}] if reg else []))
    print("STEP " + json.dumps({"step": "window", "scope": SCOPE, "wall_s": round(time.perf_counter() - t0, 1),
                                "units": UNITS, "blocks": UNITS // CAP, "law": law, "forge": a.forge, "weights": a.weights,
                                "checkpoint": {"model": MODEL, "revision": REVISION, "tensor": WEIGHT,
                                               "sha512": {n: hashlib.sha512(v.tobytes()).hexdigest() for n, v in ck.items()}},
                                "circuit_sha512": hashlib.sha512((out / "circuit.json").read_bytes()).hexdigest(),
                                "docket_digest": DD.digest(json.loads((out / "registration.json").read_text()))},
                               default=str), flush=True)
    return 0


def two_levels() -> tuple[bytes, dict]:
    """C's bytes and its partition object at two levels: the whole population, then the ``CAP``-unit proof blocks."""
    p = product()
    return p.description, p.partition_levels


def levels(_a: argparse.Namespace) -> int:
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC

    c, part = two_levels()
    try:
        OC.block(c, part, 0)
    except OC.Refused as e:
        print("REFUSED " + json.dumps({"levels": 2, "why": str(e)}), flush=True)
        return 0
    print("ACCEPTED " + json.dumps({"levels": 2}), flush=True)
    return 1


def c(a: argparse.Namespace) -> int:
    p, out = product(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "C.json").write_bytes(p.description)
    (out / "query.json").write_text(json.dumps(p.query))
    print("C " + json.dumps({"sha256": hashlib.sha256(p.description).hexdigest(), "bytes": len(p.description),
                             "program_sha512": p.digest}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("--out", required=True)
    w.add_argument("--rule", required=True)
    w.add_argument("--forge")
    w.add_argument("--weights", choices=("registered", "committed"), default="registered")
    w.add_argument("--cache", default=str(Path.home() / ".cache/verity-one-linear"))
    sub.add_parser("levels")
    sub.add_parser("c").add_argument("--out", required=True)
    a = ap.parse_args()
    return {"window": window, "levels": levels, "c": c}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
