"""stage_tile.py COMMIT_DIR PARAMS UNIT OUT PASS_DIR: unit UNIT of a served window as one C-Flock --zk unit, on the committer's
rows and the pass's tile.

The unit is `Pc8TileLeafH3{K=4096,DEV=sm120}`: A row i (fp8-p, role X, row-seg) and B row j (fp8-p, role W, v2) of the
window's entry the unit falls in (`pouw.window.tile_at` of ``advice.json``: a call's kept rows, or the filler's zero rows),
under the salts `served_commit.py commit` committed them with, so the proof's committed rows have the b ‖ c the entry's
leaves hold; then the leaf's advice (a leaf-advice row: word (i, j)'s 1 KiB segment of its tile's -h3 messages and every
segment's digest; a leaf-at row: the word's place in the tile, `words.leaf_at`), and its output row, the tile's -h3 leaf
and the place.  A call's tile is the pass's 64 x 64 tile holding word (i, j) (`audit.tile_refs` of the pass's workload),
its checked words recomputed from the pass's rows as its verifier does (`PearlC.encode`, `checked`, on `fast_tile`), and
the leaf the unit outputs must be the one the pass committed (its ``leaves`` file); a filler's is
`served_commit.filler_tile`.  `class_statement.stage_unit` writes circuit.txt, inst.bin and pub.bin under OUT; this writes
pin.txt, stage.json and opening.json (the prover's: the entry, its salt and path to the window root, each row's frame-v3
path to the entry's side root, and the output with its row's salt and the tile's opening under the pass's tile
commitment, `served_commit.leaf_opening`).  ``SZ_LEAF_TAMPER=1`` (a control) flips a byte of a digest the leaf reads but
the unit doesn't compute, so the unit proves another tile's leaf, which the window's check must reject.  `HZ_STAGE`
(``fast``, the default, or ``library``) as in `hidden_zk/stage.py`."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hidden_zk"))
sys.path.insert(0, str(HERE.parent / "pearl_c_vllm"))

from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X, sha512_row_seg_segment_prefix  # noqa: E402
from verity.protocols.accounting.work.pouw import window as Wd  # noqa: E402
from verity.protocols.accounting.work.pouw.audit import HM96_TRANSCRIPT, Workload, hm96_salt, tile_refs  # noqa: E402
from verity.protocols.accounting.work.pouw.circuit import leaves as L  # noqa: E402
from verity.protocols.accounting.work.pouw.circuit import pc8  # noqa: E402
from verity.protocols.accounting.work.pouw.circuit import words as W  # noqa: E402
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape  # noqa: E402
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402

import served_commit as SC  # noqa: E402

DEV = "sm120"


def segment(checked: C.Checked, r: int, col: int) -> tuple[list[int], list[bytes], int]:
    """Word (r, col)'s segment of the tile's -h3 messages as 32-bit words, every segment's digest (zero past the tile's)
    and the word's place."""
    rows = len(checked.c)
    joined = b"".join(C.messages(checked))
    n_seg = rows // 2
    digs = [hashlib.sha512(sha512_row_seg_segment_prefix(q, n_seg, 8192) + joined[1024 * q:1024 * (q + 1)]).digest()
            for q in range(n_seg)] + [bytes(64)] * (W.LEAF_SEGMENTS - n_seg)
    s = r // 2
    return list(struct.unpack("<256I", joined[1024 * s:1024 * (s + 1)])), digs, W.leaf_at(r, col, rows)


def call_tile(pass_dir: Path, record: dict, i: int, j: int) -> dict:
    """Word (i, j) of call ``record``'s tile in the pass: its id, checked words, committed leaf, salt and path."""
    import fast_tile as FT
    import verify_run as VR
    PD = SC._pd()
    man = json.loads((pass_dir / "manifest.json").read_bytes())
    scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    refs = tile_refs(scheme, Workload(matmuls))
    pos = next(p for p, u in enumerate(man["matmuls"]) if u["index"] == record["matmul"])
    ref = SC.tile_of(SimpleNamespace(positions={record["matmul"]: pos}, refs=refs), record["matmul"], i, j)
    u, files = man["matmuls"][pos], VR.Files(pass_dir)
    w = man["weights"][str(u["weight"])]
    k = u["k"]
    a = [VR.f32_row(files(u["rows"]), k, r) for r in ref.tile.rows]
    b = [VR.f32_row(files(w["rows"]), k, c) for c in ref.tile.cols]
    ctx = Context(bytes.fromhex(man["salt"]), matmuls[pos], bytes.fromhex(u["root_a"]), bytes.fromhex(w["root"]))
    undo = FT.install(scheme)
    try:
        noise = scheme.noise(ctx)
        checked = scheme.checked(ctx, scheme.encode(ctx, noise, a, b, ref.tile))
    finally:
        undo()
    c = man["commitment"]
    committed = bytes(files(man["leaves"])[64 * ref.id:64 * ref.id + 64])
    assert C.leaf(checked, "h3") == committed, f"tile {ref.id}'s recomputed leaf is not the one the pass committed"
    salt = hm96_salt(bytes.fromhex(man["hm96_salt_key"]), HM96_TRANSCRIPT, ref.id)
    return {"id": ref.id, "rows": ref.tile.rows, "cols": ref.tile.cols, "checked": checked, "leaf": committed, "salt": salt,
            "path": list(VR.opening(files(man["tree_t"]), c["count"], ref.id, committed + salt, 64).path)}


def main() -> None:
    t0 = time.time()
    src, params, unit, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), int(sys.argv[3]), Path(sys.argv[4])
    pass_dir = Path(sys.argv[5])
    stage = os.environ.get("HZ_STAGE", "fast")
    tamper = os.environ.get("SZ_LEAF_TAMPER") == "1"
    out.mkdir(parents=True, exist_ok=True)
    if stage == "fast":
        import fast_stage
        fast_stage.install(out)
    advice = json.loads((src / "advice.json").read_text())
    priv = SC.load(src / "prover.npz")
    k = params["k"]
    e, i, j = Wd.tile_at(advice["entries"], unit)
    entry = advice["entries"][e]
    side = {"a": priv[e]["a"], "b": priv[e]["b"]}
    if entry["kind"] == "call":
        tile = call_tile(pass_dir, entry["record"], i, j)
        r, col, checked = i - tile["rows"][0], j - tile["cols"][0], tile["checked"]
    else:
        tile, r, col, checked = None, i % 64, j % 64, SC.filler_tile(k)
    seg, digs, at = segment(checked, r, col)
    if tamper:
        q = (r // 2 + 1) % (len(checked.c) // 2)
        digs[q] = bytes([digs[q][0] ^ 1]) + digs[q][1:]
    want = tile["leaf"] if tile else SC.filler_leaf(k)
    d = pc8.tile_leaf_h3(k, DEV)
    sr = L.statement_rows(pc8.PORTS["tile_leaf_h3"], k, {p: 1 for p in ("ac", "pa", "bc", "pb")})
    in_rows = [("a0", L.ROW_SCHEMA["fp8-p", ROLE_X], 16 * L.ROW_WORDS["fp8-p"](k), ["ac[0]", "pa[0]"]),
               ("b0", L.ROW_SCHEMA["fp8-p", ROLE_W], 16 * L.ROW_WORDS["fp8-p"](k), ["bc[0]", "pb[0]"]),
               ("x0", L.ROW_SCHEMA["leaf-advice", ROLE_X], 16 * L.ROW_WORDS["leaf-advice"](k), ["seg", "dig"]),
               ("t0", L.ROW_SCHEMA["leaf-at", ROLE_X], 16 * L.ROW_WORDS["leaf-at"](k), ["at"])]
    assert [list(x) for x in sr["row_schemas"][:len(in_rows)]] == [[s, b] for _r, s, b, _p in in_rows]
    out_rows, at_ = [], 0
    for (s, _b), n in zip(sr["row_schemas"][len(in_rows):], sr.get("out_row_ports") or [1]):
        out_rows.append((s, tuple(range(at_, at_ + n))))
        at_ += n
    rows = {"a0": side["a"]["words"][i][None, :], "b0": side["b"]["words"][j][None, :],
            "x0": np.array([L.words_leaf_advice(seg, digs)], np.uint16), "t0": np.array([L.words_leaf_at(at)], np.uint16)}
    salts = {"a0": [side["a"]["salts"][i]], "b0": [side["b"]["salts"][j]]}
    got = CS.stage_unit(d, "unit/pouw-pc8-leaf-h3-served", in_rows, out_rows, rows, out, os.urandom(32), salts=salts)
    text = fast_stage.composed() if stage == "fast" else (out / "circuit.txt").read_text()
    pin = FC.digest(text)
    (out / "pin.txt").write_text(pin)
    body = (out / "pub.bin").read_bytes().split(b"\n", 1)[1]
    assert body[:128] == side["a"]["bcs"][i] and body[128:256] == side["b"]["bcs"][j], \
        "the staged rows' b || c are not the committed rows'"
    leaf = np.asarray(got["outs"]["o0"][0], dtype="<u2").tobytes()
    assert int.from_bytes(np.asarray(got["outs"]["o1"][0], dtype="<u2").tobytes(), "little") == at
    assert (leaf != want) if tamper else (leaf == want), "the unit's leaf is not its tile's" if not tamper else "the tamper changed nothing"
    out_salt = got["salts"]["o0"][0]
    assert body[SC.OUT_ROW * 128:(SC.OUT_ROW + 1) * 128] == L.commit("leaf-h3", ROLE_X, L.words_leaf_h3(leaf, at), out_salt), \
        "the staged output row's b || c is not the leaf and place's"
    lo = SC.leaf_opening(leaf, at, out_salt, *((tile["id"], tile["salt"], tile["path"]) if tile else ()))
    (out / "opening.json").write_text(json.dumps(SC.opening(advice, priv, params, unit, lo)))
    rec = {"definition": d.id, "k": k, "unit": unit, "entry": e, "kind": entry["kind"], "staging": stage, "binding": "library",
           "ands": got["ands"], "k_log": got["meta"]["k_log"], "row_schemas": got["meta"]["row_schemas"],
           "circuit_bytes": fast_stage.size(text) if stage == "fast" else len(text), "circuit_sha512": pin,
           "class_sha512": FC.class_digest(text), "leaf": leaf.hex(), "at": at, "tile": tile["id"] if tile else None,
           "tampered": tamper, "seconds": round(time.time() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    if stage == "fast":
        del text
        fast_stage.release()
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
