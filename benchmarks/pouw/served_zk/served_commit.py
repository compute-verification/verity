"""One served call's commitment in the ZK statement's format, from a retained Pearl-C pass, and the check that a hidden tile
proof's committed rows are that commitment's.

  python served_commit.py commit PASS_DIR OUT [--call I]     the committer: the call's fp8-p rows, salted and committed
  python served_commit.py draw COMMITMENT OUT [--source HEX]  the verifier: a tile of the call, drawn after the commitment
  python served_commit.py check COMMITMENT DRAW PUB OPENING   the verifier: the proof's rows are the drawn rows' leaves

- **The call** is one of the pass's ``o_proj``-shaped calls (k = n = `K`), drawn from ``os.urandom`` unless ``--call`` names
  it.  Its A rows (the pass's retained activations, BF16 widened to FP32) and its weight's B rows are formed on the CPU as
  `pearl_c` forms them under the pass's epoch: seeds from the manifest's salt, the call's A root and the weight's root
  (`PearlC.noise`), forming v1 and the peel factors (`fast_tile`'s batch forms, the scalar reference's bytes).
- **The rows** are the statement's: each A row i < real_rows is A′_i ‖ P_A,i, an fp8-p row of role X (``hm96-sha512/row-seg/v1``),
  each B row j < n is B̃_j ‖ P_B,j, of role W (``hm96-sha512/row/v2``), `leaves.words_fp8_p`.  A row whose forming the scheme
  refuses is excluded: its leaf commits a zero row, and the draw skips it.
- **The commitment** is C-Flock's: each row's b ‖ c is hm96-sha512 over its row digest under a fresh 192-byte salt
  (`leaves.commit`), its leaf `row_leaf` of `leaves.tree_leaf(b ‖ c)`, and each side one frame-v3-sha512 tree under a domain
  bound (`BINDING_TAG`) to the pass's run, its manifest's SHA-256, the call, the side and its schema.  ``commitment.json`` is
  public: the roots, domains and counts, with the served -h2 commitment it sits beside.  ``prover.npz`` holds the rows and
  salts, the prover's alone.
- **The draw** is a tile (i, j) of the call's admitted rows, from ``os.urandom`` after the commitment is fixed, and names the
  commitment's SHA-256.
- **The check** reads the proof's public file (``pub.bin``, `verity_flock.circuit.write`): instance 0's b ‖ c of its A row
  and B row, the first two rows, must be the leaves at the drawn positions, each by its frame-v3 path to the commitment's
  root (``opening.json``, the prover's).  A proof whose A or B row is any other row, or whose rows another commitment holds,
  is refused here; a proof whose opened row is not its committed row is refused by the proof's own verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from verity.primitives.commitments.identity import identity_digest
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X, row_leaf
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes import pearl_kw as P

SCHEMA = "verity-pouw/served-zk-commitment/v0"
DRAW_SCHEMA = "verity-pouw/served-zk-draw/v0"
BINDING_TAG = "verity/pouw/served-zk/binding/v0"
#: the frame's owner, as C-Flock's statement domains have it (`verity_flock.ir_frame.OWNER`)
OWNER = -1
SALT_BYTES = 192
K = 4096
SIDES = {"a": ROLE_X, "b": ROLE_W}
#: the rows a forming batch takes at once
CHUNK = 256


def f32_rows(data, k: int, idx) -> list[bytes]:
    """Rows `idx` of a BF16 row file as the committed FP32 words (BF16 widened, exact)."""
    out = []
    for i in idx:
        out.append((np.frombuffer(data[2 * k * i:2 * k * (i + 1)], "<u2").astype("<u4") << 16).tobytes())
    return out


def _form(scheme: C.PearlC, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]) -> list[list[int]]:
    """The fp8-p words of rows `idx` (FP32 bytes `rows`) on one side, as `PearlC.encode` forms them; raises `P.Reject`."""
    seed_e = noise.seed_a if side == "a" else noise.seed_b
    f, e = scheme._side(rows, idx, 0 if side == "a" else 1, seed_e, noise.seed_b, k)
    peel = C.peel_factors_a if side == "a" else C.peel_factors_b
    p = peel(f.codes, C.scaled_noise(f, e), noise.seed_b, k, scheme.line_norm, scheme.device)
    ok = [P.f32_of_bits(r) * P.f32_of_bits(a) >= P.SIGMA_MIN and C.live_row(x) for r, a, x in zip(f.rho, f.alpha, f.x)]
    return [L.words_fp8_p(c, q) if good else None for c, q, good in zip(f.codes, p, ok)]


def form(scheme: C.PearlC, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]) -> list[list[int] | None]:
    """Each row's fp8-p words, or None for a row the scheme refuses (alone, so one refusal leaves its batch's others)."""
    out: list[list[int] | None] = []
    for at in range(0, len(idx), CHUNK):
        r, i = rows[at:at + CHUNK], idx[at:at + CHUNK]
        try:
            out += _form(scheme, noise, k, side, r, i)
        except P.Reject:
            for one, j in zip(r, i):
                try:
                    out += _form(scheme, noise, k, side, [one], [j])
                except P.Reject:
                    out.append(None)
    return out


def domain(binding: dict, side: str, count: int) -> tuple[CommitmentDomain, str]:
    """Side `side`'s frame-v3-sha512 domain over `count` rows, bound to `binding` (the pass, the call) and the side's schema."""
    schema = L.ROW_SCHEMA["fp8-p", SIDES[side]]
    b = bytes.fromhex(identity_digest(BINDING_TAG, {**binding, "side": side, "schema": schema, "rows": count}))
    return CommitmentDomain(b, OWNER, RangeIndexedDomain(0, count), hash="sha512"), schema


def leaves_of(dom: CommitmentDomain, schema: str, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]


def path(levels, index: int) -> list[str]:
    out = []
    for level in levels[:-1]:
        out.append(level[index ^ 1].hex())
        index >>= 1
    return out


def verify_path(dom: CommitmentDomain, root: bytes, index: int, leaf: bytes, sibs: list[bytes]) -> bool:
    if not 0 <= index < dom.count or len(sibs) != merkle_depth(dom.count):
        return False
    digest = leaf
    for level, s in enumerate(sibs):
        digest = dom.node(level, index >> 1, *((digest, s) if index % 2 == 0 else (s, digest)))
        index >>= 1
    return digest == root


def commit_words(binding: dict, words: dict[str, list[list[int] | None]], k: int, salts: dict | None = None) -> tuple[dict, dict]:
    """The public commitment's sides and the prover's package: per side its rows' words (an excluded row zero), salts
    (fresh from the OS unless given), b ‖ c and tree levels."""
    zero = [0] * L.ROW_WORDS["fp8-p"](k)
    pub, priv = {}, {}
    for side, ws in words.items():
        dom, schema = domain(binding, side, len(ws))
        s = salts[side] if salts else [os.urandom(SALT_BYTES) for _ in ws]
        bcs = [L.commit("fp8-p", SIDES[side], w if w is not None else zero, salt) for w, salt in zip(ws, s)]
        levels = tree_levels(dom, leaves_of(dom, schema, bcs))
        pub[side] = {"schema": schema, "rows": len(ws), "binding": dom.binding.hex(), "domain_id": dom.domain_id.hex(),
                     "root": levels[-1][0].hex(), "excluded": [i for i, w in enumerate(ws) if w is None]}
        priv[side] = {"words": np.array([w if w is not None else zero for w in ws], dtype=np.uint16), "salts": s, "bcs": bcs}
    return pub, priv


def opening(commitment: dict, side: str, bcs: list[bytes], index: int) -> dict:
    """The prover's frame-v3 path for row `index` of one side, from every row's b ‖ c."""
    dom = side_domain(commitment, side)
    return {"index": index, "path": path(tree_levels(dom, leaves_of(dom, commitment["sides"][side]["schema"], bcs)), index)}


def save(path: Path, priv: dict) -> None:
    np.savez(path, **{f"{s}_words": p["words"] for s, p in priv.items()},
             **{f"{s}_salts": np.frombuffer(b"".join(p["salts"]), np.uint8).reshape(-1, SALT_BYTES) for s, p in priv.items()},
             **{f"{s}_bcs": np.frombuffer(b"".join(p["bcs"]), np.uint8).reshape(-1, 128) for s, p in priv.items()})


def load(path: Path) -> dict:
    z = np.load(path)
    return {s: {"words": z[f"{s}_words"], "salts": [r.tobytes() for r in z[f"{s}_salts"]], "bcs": [r.tobytes() for r in z[f"{s}_bcs"]]}
            for s in SIDES}


def side_domain(c: dict, side: str) -> CommitmentDomain:
    s = c["sides"][side]
    dom, schema = domain(c["binding"], side, s["rows"])
    if dom.domain_id.hex() != s["domain_id"] or schema != s["schema"]:
        raise ValueError(f"side {side}: its domain is not the one its binding names")
    return dom


def check(commitment: dict, draw: dict, pub: bytes, opening: dict) -> str | None:
    """None if the proof's public file opens the drawn rows under the commitment, else why not."""
    raw = json.dumps(commitment, sort_keys=True).encode()
    if draw.get("commitment_sha256") != hashlib.sha256(raw).hexdigest():
        return "the draw names another commitment"
    head, body = pub.split(b"\n", 1)
    h = json.loads(head)
    if h.get("instances") != 1:
        return f"the proof has {h.get('instances')} instances, not one tile"
    ports = list(h["frame_v3"]["schemas"].items())
    if [s for _p, s in ports[:2]] != [commitment["sides"][s]["schema"] for s in ("a", "b")]:
        return f"the proof's first rows are {ports[:2]}, not an A row and a B row of the commitment's schemas"
    bc = {"a": body[0:128], "b": body[128:256]}
    for side in ("a", "b"):
        dom = side_domain(commitment, side)
        at = draw[side]
        if at in commitment["sides"][side]["excluded"]:
            return f"side {side}: row {at} is excluded"
        o = opening[side]
        if o["index"] != at:
            return f"side {side}: the opening is of row {o['index']}, not the drawn row {at}"
        leaf = row_leaf(dom, at, L.tree_leaf(bc[side]), commitment["sides"][side]["schema"])
        if not verify_path(dom, bytes.fromhex(commitment["sides"][side]["root"]), at, leaf, [bytes.fromhex(x) for x in o["path"]]):
            return f"side {side}: the proof's committed row is not the commitment's row {at}"
    return None


# ---- the CLI ------------------------------------------------------------------------------------------------------------

def cmd_commit(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import fast_tile as FT
    from verity_vllm.protocol_options import pouw_pearl_c_device as PD

    pass_dir = Path(a.pass_dir)
    raw = (pass_dir / "manifest.json").read_bytes()
    man = json.loads(raw)
    if man.get("schema") != PD.MANIFEST_SCHEMA:
        raise SystemExit(f"{pass_dir}: not a retained pass ({man.get('schema')!r})")
    scheme = PD.scheme_of(man["scheme"])
    calls = [u for u in man["matmuls"] if u["k"] == K and u["n"] == K]
    if not calls:
        raise SystemExit(f"{pass_dir}: no call of k = n = {K}")
    source = os.urandom(32)
    u = next(x for x in calls if x["index"] == a.call) if a.call is not None else \
        calls[int.from_bytes(hashlib.sha256(source).digest()[:8], "little") % len(calls)]
    w = man["weights"][str(u["weight"])]
    k, n, real = u["k"], u["n"], u["real_rows"]
    ctx = Context(bytes.fromhex(man["salt"]), Matmul(u["index"], Shape(u["m"], k, n), u["weight"]), bytes.fromhex(u["root_a"]),
                  bytes.fromhex(w["root"]))
    noise = scheme.noise(ctx)
    undo = FT.install(scheme)
    try:
        a_data = (pass_dir / u["rows"]["path"]).read_bytes()
        words = {"a": form(scheme, noise, k, "a", f32_rows(a_data, k, range(real)), list(range(real)))}
        b_data = np.memmap(pass_dir / w["rows"]["path"], dtype=np.uint8, mode="r")
        words["b"] = form(scheme, noise, k, "b", f32_rows(b_data, k, range(n)), list(range(n)))
    finally:
        undo()
    binding = {"run": man.get("run"), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "scheme": man["scheme"],
               "matmul": u["index"], "shape": [u["m"], k, n], "weight": u["weight"], "real_rows": real}
    sides, priv = commit_words(binding, words, k)
    c = {"schema": SCHEMA, "binding": binding, "sides": sides,
         "served": {"transcript_root": man.get("transcript_root"), "commitment": man.get("commitment"),
                    "records_root": man.get("records_root"), "root_a": u["root_a"], "weight_root": w["root"]},
         "call_draw": {"source": source.hex(), "among": [x["index"] for x in calls]} if a.call is None else {"given": a.call}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_text(json.dumps(c, sort_keys=True))
    save(out / "prover.npz", priv)
    print(json.dumps({"call": u["index"], "rows": {s: v["rows"] for s, v in sides.items()},
                      "excluded": {s: len(v["excluded"]) for s, v in sides.items()},
                      "roots": {s: v["root"][:16] for s, v in sides.items()}}), flush=True)
    return 0


def cmd_draw(a) -> int:
    raw = Path(a.commitment).read_bytes()
    c = json.loads(raw)
    source = bytes.fromhex(a.source) if a.source else os.urandom(32)
    pick = {}
    for side, tag in (("a", b"a"), ("b", b"b")):
        ok = [i for i in range(c["sides"][side]["rows"]) if i not in set(c["sides"][side]["excluded"])]
        pick[side] = ok[int.from_bytes(hashlib.sha256(source + tag).digest()[:8], "little") % len(ok)]
    d = {"schema": DRAW_SCHEMA, "commitment_sha256": hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(),
         "source": source.hex(), **pick}
    Path(a.out).write_text(json.dumps(d, sort_keys=True))
    print(json.dumps(d), flush=True)
    return 0


def cmd_check(a) -> int:
    c, d = json.loads(Path(a.commitment).read_text()), json.loads(Path(a.draw).read_text())
    why = check(c, d, Path(a.pub).read_bytes(), json.loads(Path(a.opening).read_text()))
    print(f"ACCEPT {a.pub} opens rows a={d['a']} b={d['b']} of {a.commitment}" if why is None else f"REJECT {a.pub}: {why}", flush=True)
    return 0 if (why is None) != a.expect_reject else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("commit")
    p.add_argument("pass_dir")
    p.add_argument("out")
    p.add_argument("--call", type=int)
    p = sub.add_parser("draw")
    p.add_argument("commitment")
    p.add_argument("out")
    p.add_argument("--source", help="the draw's source (hex), for a control's draw of the same rows against another commitment")
    p = sub.add_parser("check")
    for x in ("commitment", "draw", "pub", "opening"):
        p.add_argument(x)
    p.add_argument("--expect-reject", action="store_true")
    a = ap.parse_args(argv)
    return {"commit": cmd_commit, "draw": cmd_draw, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
