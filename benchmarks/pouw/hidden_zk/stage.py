"""stage.py ROLE K N OUT: one Pearl-C4 unit class over N instances, staged for Flock by `class_statement.stage_unit` with
PoUW's rows (`leaves.statement_rows`) and, for a row unit, its anchors as public input ports (`anchors.public_ports`).

    row_a  Pc4RowA_v1{K}: x (bf16, row-seg) -> A'_i's codes, scales, P_A,i (nvfp4-p, v2) and ok; public e, fa, fb
    row_b  Pc4RowB_v1{K}: x (bf16, v2) -> B~_j's codes, scales, P_B,j and ok; public e, fb, fa, g
    tile   Pc4Tile_v1{K,TM=1,TN=1}: an A row and a B row (nvfp4-p) -> C~ and U (one v2 run); no public input
    tileTMxTN   the same over TM A rows and TN B rows
    hidden[TMxTN]   Pc4TileHidden_v1{K,TM,TN}: the same rows -> the tile digest (its tile row)

x is a seeded N(0, 1) row in BF16; a tile's rows are `pouw_rows_vectors`' seeded nvfp4-p rows. The anchors are the
verifier's own (`TileAnchors.of` the scheme's seeds). Writes circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under
OUT, and for a row unit the verifier's own file public-inputs.bin (`TileAnchors.public_inputs`) and the controls' files:
public-inputs-wrong.bin (one anchor bit flipped, the verifier's file wrong) and pub-chosen.bin (the same bit flipped in the
public file's tail, an anchor the prover chose).

`HZ_STAGE` (``fast``, the default, or ``library``): whether `fast_stage` patches the library first, for the same bytes in
less time and memory; stage.json records which.

DEVELOPMENT BINDING: core's BitAnd_v1, E4m3ToF32_v1 and F32ToE2m1Sat_v1 have no Flock piece on this branch, so the row
units lower them through `core3_scratch`'s pieces; every result of a row unit staged here is labelled 'core3 scratch'. The
PoUW word primitives Keccak-f is written in have none either, so the hiding tile lowers them through `hash_scratch`'s
pieces, checked against their primitives first; its results are labelled 'hash scratch'."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.circuit import pc4
from verity.core.protocols.pouw.circuit.anchors import TileAnchors, public_ports
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c4 as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROLE, K, N, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
FLIP = {"row_a": "fa", "row_b": "g"}      # the anchor whose word 0, instance 0, bit 0 the controls flip
STAGE = os.environ.get("HZ_STAGE", "fast")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part(name: str) -> bytes:
    return hashlib.sha256(f"verity/pouw/hidden-zk/v1\0{name}".encode()).digest()


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def flip_at(data: bytes, ports: list[dict], port: str, tail: bool) -> int:
    """The byte offset of word 0 of ``port`` in instance 0's regions: in a --public-inputs file (after its header line), or
    in a public file's tail (the last ``instances`` x regions words)."""
    words = sum(p["words"] for p in ports)
    before = sum(p["words"] for p in ports[:[p["port"] for p in ports].index(port)])
    if tail:
        return len(data) - 2 * N * words + 2 * before
    return len(data.split(b"\n", 1)[0]) + 1 + 2 * before


def main() -> None:
    t0 = time.time()
    if STAGE not in ("fast", "library"):
        raise SystemExit(f"stage.py: HZ_STAGE is fast or library, not {STAGE!r}")
    if STAGE == "fast":
        sys.path.insert(0, str(HERE))
        import fast_stage
        fast_stage.install(OUT)
    salt_key = hashlib.sha256(b"verity/pouw/hidden-zk/salts\0" + str(K).encode()).digest()
    rec: dict = {"role": ROLE, "k": K, "n": N, "staging": STAGE}
    if ROLE in ("row_a", "row_b"):
        sys.path.insert(0, str(HERE))
        import core3_scratch as K3
        IL.PIECES.update(K3.PIECES)
        rec["binding"] = "core3 scratch"
        shape = pc4.Shape(N, K, 1, tile=(1, 1)) if ROLE == "row_a" else pc4.Shape(1, K, N, tile=(1, 1))
        ctx = Context(part("salt"), Matmul(3, Shape(shape.m, shape.k, shape.n), 9), part("root_a"), part("root_b"))
        ta = TileAnchors.of(F.PearlC4("nvfp4"), ctx, shape)
        d = pc4.templates(K)[ROLE]
        ports = public_ports(shape, ROLE)
        sr = L.statement_rows(pc4.PORTS[ROLE], K)
        (xs, xb), outs_ = sr["row_schemas"][0], sr["row_schemas"][1:]
        in_rows = [("x", xs, xb, ["x"])]
        rng = np.random.default_rng(20261003 + K)
        rows = {"x": np.stack([bf16(rng.standard_normal(K)) for _ in range(N)])}
        regs = [ta.regions(ROLE, i) for i in range(N)]
        for p in ports:
            rows[p] = np.array([r[p] for r in regs], dtype=np.uint16)
    elif ROLE.startswith(("tile", "hidden")):
        V = _load("pouw_rows_vectors", ROOT / "verity" / "core" / "protocols" / "pouw" / "tests" / "pouw_rows_vectors.py")
        hidden = ROLE.startswith("hidden")
        if hidden:
            sys.path.insert(0, str(HERE))
            import hash_scratch as H
            bad = H.check()
            assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
            IL.PIECES.update(H.PIECES)
            rec["binding"] = "hash scratch"
        tm, tn = map(int, ROLE.removeprefix("hidden" if hidden else "tile").split("x")) if "x" in ROLE else (1, 1)
        rec["tile"] = [tm, tn]
        d = pc4.templates(K, (tm, tn), hidden=hidden)["tile"]
        ports = {}
        bits = 16 * L.ROW_WORDS["nvfp4-p"](K)
        in_rows = [(f"{s}{r}", L.ROW_SCHEMA["nvfp4-p", role], bits, [f"{p}[{r}]" for p in ps])
                   for s, role, ps, t in (("a", ROLE_X, ("ac", "as", "pa"), tm), ("b", ROLE_W, ("bc", "bs", "pb"), tn))
                   for r in range(t)]
        if hidden:
            sr = L.statement_rows(pc4.PORTS["tile"], K, {p: t for ps, t in ((("ac", "as", "pa"), tm), (("bc", "bs", "pb"), tn))
                                                          for p in ps})
            outs_ = sr["row_schemas"][tm + tn:]
        else:
            outs_ = [[SCHEMA_HM96_SHA512_ROW_V2, None]]
            sr = {"out_row_ports": [2]}
        rows = {name: np.array([V.words("nvfp4-p", V.inputs(f"hidden-zk-{K}-{name}-{i}", "nvfp4-p", K)) for i in range(N)],
                               dtype=np.uint16) for name, *_ in in_rows}
    else:
        raise SystemExit(f"stage.py: ROLE is row_a, row_b, tile[TMxTN] or hidden[TMxTN], not {ROLE!r}")
    runs = sr.get("out_row_ports") or [1] * len(outs_)
    out_rows, at = [], 0
    for (s, _b), r in zip(outs_, runs):
        out_rows.append((s, tuple(range(at, at + r))))
        at += r
    rec.update(definition=d.id, in_rows=[[r, s, b] for r, s, b, _ in in_rows], out_rows=[[s, list(ps)] for s, ps in out_rows],
               public_inputs=ports)
    got = CS.stage_unit(d, f"unit/pouw-pc4-{ROLE}", in_rows, out_rows, rows, OUT, salt_key, public_inputs=ports or None)
    t_stage = time.time() - t0
    text = fast_stage.composed() if STAGE == "fast" else (OUT / "circuit.txt").read_text()
    meta = got["meta"]
    pin = FC.digest(text)
    (OUT / "pin.txt").write_text(pin)
    if ports:
        own = ta.public_inputs(ROLE, range(N))
        ta.check_public_inputs(ROLE, range(N), own)
        (OUT / "public-inputs.bin").write_bytes(own)
        head = json.loads(own.split(b"\n", 1)[0])
        pub = (OUT / "pub.bin").read_bytes()
        assert pub.endswith(own.split(b"\n", 1)[1]), "the public file's tail is not the verifier's own words"
        wrong = bytearray(own)
        wrong[flip_at(own, head["ports"], FLIP[ROLE], tail=False)] ^= 1
        (OUT / "public-inputs-wrong.bin").write_bytes(bytes(wrong))
        chosen = bytearray(pub)
        chosen[flip_at(pub, head["ports"], FLIP[ROLE], tail=True)] ^= 1
        (OUT / "pub-chosen.bin").write_bytes(bytes(chosen))
        rec.update(own_bytes=len(own), own_sha256=hashlib.sha256(own).hexdigest(), flipped={"port": FLIP[ROLE], "word": 0,
                   "instance": 0, "bit": 0})
    rec.update(ands=got["ands"], k_log=meta["k_log"], vus_per_block=meta.get("vus_per_block"),
               row_schemas=meta["row_schemas"], out_row_ports=meta.get("out_row_ports"),
               circuit_bytes=fast_stage.size(text) if STAGE == "fast" else len(text),
               circuit_sha512=pin, class_sha512=FC.class_digest(text),
               inst_bytes=(OUT / "inst.bin").stat().st_size, pub_bytes=(OUT / "pub.bin").stat().st_size,
               seconds={"stage": round(t_stage, 1), "total": round(time.time() - t0, 1)}, maxrss_gb=gb())
    if STAGE == "fast":
        del text
        fast_stage.release()
    (OUT / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
