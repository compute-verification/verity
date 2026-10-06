"""A served audit window's commitment in the ZK statement's format, from a retained Pearl-C pass, registered, drawn and
judged through the proof service (`service.py`, P1's calls), and the check that a hidden tile proof's committed rows are
the drawn tile's.

  python served_commit.py params PASS_DIR OUT [--eps E --delta D --entries P]   the verifier's own parameters of the window
  python served_commit.py commit PASS_DIR PARAMS OUT [--calls N | --call I ...]   the committer: the window, salted and committed
  python served_commit.py register COMMIT_DIR PARAMS OUT [--extra ITEM=JSON]  the committer: `register`, the public items only
  python served_commit.py receive PARAMS REGISTRATION OUT                     the verifier: `receive`, and its receipt
  python served_commit.py edit REGISTRATION OUT KEY=JSON ...                  a control's registration, a field changed
  python served_commit.py select PARAMS REGISTRATION RECEIPT OUT --lean F --log DIR   the verifier: `select`, Lean's draw
  python served_commit.py check PARAMS REGISTRATION DRAW UNIT PUB OPENING     the verifier: the proof's rows are the unit's
  python served_commit.py outcome PARAMS REGISTRATION RECEIPT DRAW UNITS OUT --log DIR   the verifier: `outcome`
  python served_commit.py shape DIR DIR                                       two staged units' proofs have one shape

- **The window** is SZ_CALLS of the pass's ``o_proj``-shaped calls (k = n = `K`), picked from ``os.urandom`` unless
  ``--call`` names them.  Each call's A rows (the pass's retained activations, BF16 widened to FP32) and its weight's B rows
  are formed on the CPU as `pearl_c` forms them under the pass's epoch (`PearlC.noise`, forming v1, the peel factors), and
  become the statement's fp8-p rows (A′_i ‖ P_A,i of role X, ``hm96-sha512/row-seg/v1``; B̃_j ‖ P_B,j of role W,
  ``hm96-sha512/row/v2``; `leaves.words_fp8_p`).  A row whose forming the scheme refuses is no tile's.
- **The tiles** are each call's kept rows' products, then filler tiles up to the padded count: the window's N_real real
  tiles register the grid level of `pouw.window.bucket`, credited g_ℓ <= N_real tiles and padded to N = g_(ℓ+1).  A filler
  tile is a tile of the same template on zero rows (both sides), committed like any row, so its proof has the shape of
  every other tile's.  Unit t is `pouw.window.tile_at` of the advice: the population is
  `population_program(Pc8TileHidden{K,1,1,sm120}, N)` under `Q_template_instance`.
- **The commitment** is C-Flock's: each row's b ‖ c is hm96-sha512 over its row digest under a fresh 192-byte salt, each side
  of each entry (a call, or the filler) one frame-v3-sha512 tree under one-stage's served domain of the window's program,
  partition and context, its port ``<entry>/<side>``.  Each entry (`pouw.window`: its sides' roots, rows and kept rows, its
  tile range and the call's record) is one ``hm96-sha512/row/v2`` row of its SHA-512 in the **window root**, padded with
  empty entries to the public bound P (L2).  W, every weight the pass's k = n = K calls read, is committed by weight row
  (BF16, ``hm96-sha512/row/v2``) in **W's root**, beside it; the tiles read the formed B̃ rows of each call, not W's rows,
  until registered rows (P7) read B from W.  ``commitment.json`` is what the committer registers: the context, the level,
  the public items and the two roots.  ``advice.json`` (the entries and their salts) and ``prover.npz`` (rows, salts, b ‖ c)
  are the prover's.
- **The service** (`service.py`): `register` writes one-stage's v1 record whose window holds the public items, exactly
  `pouw.window.REGISTERED`, and refuses anything else; `receive` holds it to the verifier's own parameters (its context,
  ε, δ, the grid, P and |W|), recomputing every item from the registered level; `select` is Lean ``flock-verify draw
  --population N --k K′`` from the verifier's own randomness, after the receipt and once; `outcome` judges every drawn
  unit and publishes exactly `pouw.window.PUBLIC`.
- **The check** reads a proof's public file (``pub.bin``): instance 0's b ‖ c of its A row and B row must be the unit's rows,
  each by its frame-v3 path to its entry's side root, and the entry by its path to the registered window root.  The
  opening hands the verifier the drawn units' entries: a development shortcut, until position reads in gates (the joint
  note's step 8) take it over.  Nothing it reads is in the outcome's public items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from verity.primitives.commitments.identity import identity_digest
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.primitives.circuits.codec import canonical_json
from verity.protocols.accounting.work.pouw import window as Wd
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.circuit import pc8
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes import pearl_kw as P
from verity.protocols.verification.sampled_proofs.one_stage import partition as PT
from verity.protocols.verification.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import service as S  # noqa: E402

SCHEMA = "verity-pouw/served-zk-window/v1"
#: the registered scheme: one window root over hm96-sha512 entry rows, each entry's sides hm96-sha512 rows, all in
#: frame-v3-sha512 trees, and W's root by weight row
SCHEME = "verity/pouw/served-zk/window/hm96-sha512+frame-v3-sha512/v1"
VERIFIER = "served-zk/lean-flock-verify"
#: the frame's owner, as C-Flock's statement domains have it (`verity_flock.ir_frame.OWNER`)
OWNER = -1
SALT_BYTES = 192
K = 4096
TILE = (1, 1)
DEV = "sm120"
SIDES = {"a": ROLE_X, "b": ROLE_W}
#: the rows a forming batch takes at once
CHUNK = 256
ENTRY_SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
W_SCHEMA = L.ROW_SCHEMA["bf16", ROLE_W]
W_DOMAIN = "verity/pouw/served-zk/weights-domain/v0"
#: the public bound on a window's entries (its calls and the filler), and the tile counts a window may register
ENTRIES = 16
RANGE = (1, 1 << 40)
EPS, DELTA = "1/2", "1/10"


# ---- the window's parameters and its spec ---------------------------------------------------------------------------------

def template(k: int):
    return pc8.templates(k, TILE, DEV, hidden=True)["tile"]


def tile_work(params: dict) -> int:
    return params["k"] * TILE[0] * TILE[1]


@lru_cache(maxsize=8)
def population(k: int, n: int):
    """The window's tiles as one-stage's population: (program, query, partition object)."""
    d = template(k)
    prog = PT.population_program(d, n)
    q = PT.template_instance_query(PT.template_id(d))
    return prog, q, PT.partition(prog, q)


def work_at(params: dict, ell: int) -> Wd.Work:
    w = Wd.Work.at(ell, params["grid_bits"])
    lo, hi = params["range"]
    if not (lo <= w.credited and w.n <= hi):
        raise S.Refused(f"level {ell} ({w.credited} credited, {w.n} tiles) is outside the window's range {lo}..{hi}")
    return w


def check_item(params: dict) -> dict:
    return {"template": template(params["k"]).id, "tile_work": tile_work(params), "scheme": SCHEME, "entries": params["entries"]}


def items_of(params: dict, ell: int, root_w: str | None) -> dict:
    """The registered public items at level `ell` (`pouw.window.REGISTERED`)."""
    w = work_at(params, ell)
    return {"work": w.to_json(tile_work(params)), "check": check_item(params),
            "sampling": Wd.sampling(Fraction(params["eps"]), Fraction(params["delta"]), w, tile_work(params), params["grid_bits"]),
            "root_W": root_w}


def spec_of(params: dict, ell: int) -> S.Spec:
    w = work_at(params, ell)
    prog, q, _ = population(params["k"], w.n)
    law = items_of(params, ell, None)["sampling"]["law"]
    return S.Spec(values=(S.Value("window", ENTRY_SCHEMA, "serve"), S.Value("W", W_SCHEMA, "epoch", registered=True)),
                  selection=law, check=S.Check(prog, q, SCHEME, ("tile digest",)),
                  public=tuple(S.Public(i, k, r, s) for i, k, s, r in Wd.PUBLIC), context=params["context"])


def _frame(tag: str, fields: dict, count: int) -> CommitmentDomain:
    return CommitmentDomain(bytes.fromhex(identity_digest(tag, fields)), OWNER, RangeIndexedDomain(0, count), hash="sha512")


def _served(params: dict, ell: int, port: str, schema: str, count: int) -> CommitmentDomain:
    """One-stage's served domain (`registration.served_domain`) of the window's program and partition at level `ell`."""
    _, _, part = population(params["k"], work_at(params, ell).n)
    return _frame(R.SERVED_DOMAIN, {"program": part["program"], "partition": PT.digest(part), "port": port, "schema": schema,
                                    "leaves": int(count), "run": params["context"]}, count)


def window_domain(params: dict, ell: int) -> CommitmentDomain:
    return _served(params, ell, "window", ENTRY_SCHEMA, params["entries"])


def side_domain(params: dict, ell: int, e: int, side: str, rows: int) -> tuple[CommitmentDomain, str]:
    schema = L.ROW_SCHEMA["fp8-p", SIDES[side]]
    return _served(params, ell, f"{e}/{side}", schema, rows), schema


def weights_domain(params: dict) -> CommitmentDomain:
    return _frame(W_DOMAIN, {"template": template(params["k"]).id, "port": "W", "schema": W_SCHEMA,
                             "leaves": params["w_rows"], "deployment": params["deployment"]}, params["w_rows"])


def ports(params: dict, ell: int) -> dict:
    """The verifier's own ports: the window root over P entries, and W's root over |W| rows."""
    return {"window": {"leaves": params["entries"], "domain": window_domain(params, ell).domain_id.hex()},
            "W": {"leaves": params["w_rows"], "domain": weights_domain(params).domain_id.hex()}}


def level_of(rec: dict) -> int:
    try:
        ell = rec["window"]["public"]["work"]["level"]
    except (KeyError, TypeError) as e:
        raise S.Refused(f"the registration names no work level: {e}") from None
    if not isinstance(ell, int) or isinstance(ell, bool) or ell < 0:
        raise S.Refused(f"the registered level {ell!r} is not a level")
    return ell


# ---- rows, trees and openings ------------------------------------------------------------------------------------------------

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


def side_leaves(dom: CommitmentDomain, schema: str, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]


def entry_bc(entry: dict, salt: bytes) -> bytes:
    """An entry's b ‖ c: hm96-sha512 over the ``sha512/row/v2`` digest of its SHA-512 (32 words), under `salt`."""
    h = hashlib.sha512(canonical_json(entry)).digest()
    return L.INST.commit_string(L.KEY, sha512_row_v2_digest(h, 8 * len(h)), salt)


def entry_leaf(dom: CommitmentDomain, e: int, bc: bytes) -> bytes:
    return row_leaf(dom, e, L.tree_leaf(bc), ENTRY_SCHEMA)


def _salts(n: int) -> list[bytes]:
    return [os.urandom(SALT_BYTES) for _ in range(n)]


def commit_side(dom: CommitmentDomain, schema: str, side: str, words: list[list[int]], salts: list[bytes]) -> tuple[str, dict]:
    bcs = [L.commit("fp8-p", SIDES[side], w, s) for w, s in zip(words, salts)]
    return tree_levels(dom, side_leaves(dom, schema, bcs))[-1][0].hex(), {"words": np.array(words, dtype=np.uint16),
                                                                          "salts": salts, "bcs": bcs}


def commit_weights(params: dict, rows, salts: list[bytes] | None = None) -> tuple[str, list[bytes]]:
    """W's root: each BF16 weight row (raw bytes, k little-endian words) an ``hm96-sha512/row/v2`` row under a fresh salt."""
    dom = weights_domain(params)
    leaves, used = [], []
    for i, raw in enumerate(rows):
        s = salts[i] if salts else os.urandom(SALT_BYTES)
        used.append(s)
        bc = L.INST.commit_string(L.KEY, sha512_row_v2_digest(bytes(raw), 8 * len(raw)), s)
        leaves.append(row_leaf(dom, i, L.tree_leaf(bc), W_SCHEMA))
    if len(leaves) != params["w_rows"]:
        raise ValueError(f"W has {len(leaves)} rows, not the parameters' {params['w_rows']}")
    return tree_levels(dom, leaves)[-1][0].hex(), used


def filler_shape(fill: int) -> tuple[int, int]:
    """The filler's zero rows (A, B): about the square root of its tiles each, so its tiles cover `fill`."""
    b = max(1, math.isqrt(fill - 1) + 1)
    return -(-fill // b), b


def commit_window(params: dict, calls: list[dict], root_w: str, salts: dict | None = None) -> tuple[dict, dict, dict]:
    """The window's commitment from its calls (each `{"record", "words": {"a", "b"}}`, a side's words None where the scheme
    refused the row): (``commitment.json``, ``advice.json``, the prover's rows).  `salts` replays the salts of an earlier
    commit's prover file (`load`)."""
    k = params["k"]
    zero = [0] * L.ROW_WORDS["fp8-p"](k)
    keeps = [{s: [i for i, w in enumerate(c["words"][s]) if w is not None] for s in SIDES} for c in calls]
    real = sum(len(kp["a"]) * len(kp["b"]) for kp in keeps)
    work = Wd.bucket(real, params["grid_bits"])
    ell = work.level
    work_at(params, ell)
    if len(calls) + 1 > params["entries"]:
        raise ValueError(f"{len(calls)} calls and the filler are more entries than the bound {params['entries']}")
    fa, fb = filler_shape(work.n - real)
    sides = [{s: [w if w is not None else zero for w in c["words"][s]] for s in SIDES} for c in calls]
    sides.append({"a": [zero] * fa, "b": [zero] * fb})
    entries, priv, at = [], {}, 0
    for e, (ws, kind) in enumerate(zip(sides, ["call"] * len(calls) + ["filler"])):
        entry = {"kind": kind, "lo": at}
        keep = keeps[e] if kind == "call" else {"a": list(range(fa)), "b": list(range(fb))}
        size = len(keep["a"]) * len(keep["b"]) if kind == "call" else work.n - real
        entry["hi"] = at + size
        at += size
        priv[e] = {}
        for s in SIDES:
            dom, schema = side_domain(params, ell, e, s, len(ws[s]))
            root, priv[e][s] = commit_side(dom, schema, s, ws[s], salts[e][s] if salts else _salts(len(ws[s])))
            entry[s] = {"rows": len(ws[s]), "keep": keep[s], "root": root}
        if kind == "call":
            entry["record"] = calls[e]["record"]
        entries.append(entry)
    entries += [{"kind": "none", "index": e} for e in range(len(entries), params["entries"])]
    problem = Wd.window_problem(entries, work, params["entries"], params["grid_bits"])
    if problem:
        raise ValueError(f"the window statement fails on the committer's own advice: {problem}")
    dom = window_domain(params, ell)
    esalts = salts["entries"] if salts else _salts(len(entries))
    levels = tree_levels(dom, [entry_leaf(dom, e, entry_bc(x, s)) for e, (x, s) in enumerate(zip(entries, esalts))])
    roots = [{"port": "window", "root": levels[-1][0].hex(), "leaves": params["entries"], "domain": dom.domain_id.hex()},
             {"port": "W", "root": root_w, "leaves": params["w_rows"], "domain": weights_domain(params).domain_id.hex()}]
    public = {"schema": SCHEMA, "context": params["context"], "level": ell, "items": items_of(params, ell, root_w),
              "roots": roots}
    advice = {"schema": SCHEMA, "level": ell, "real": real, "entries": entries, "entry_salts": [s.hex() for s in esalts],
              "window_path": [path(levels, e) for e in range(len(entries))]}
    priv["entries"] = esalts
    return public, advice, priv


def save(p: Path, priv: dict) -> None:
    arrays = {}
    for e, sides in priv.items():
        if e == "entries":
            continue
        for s, x in sides.items():
            arrays[f"e{e}_{s}_words"] = x["words"]
            arrays[f"e{e}_{s}_salts"] = np.frombuffer(b"".join(x["salts"]), np.uint8).reshape(-1, SALT_BYTES)
            arrays[f"e{e}_{s}_bcs"] = np.frombuffer(b"".join(x["bcs"]), np.uint8).reshape(-1, 128)
    np.savez(p, **arrays)


def load(p: Path) -> dict:
    z = np.load(p)
    out: dict = {}
    for key in z.files:
        e, s, what = key.split("_")
        x = out.setdefault(int(e[1:]), {}).setdefault(s, {})
        x[what] = z[key] if what == "words" else [r.tobytes() for r in z[key]]
    return out


def opening(advice: dict, priv: dict, params: dict, unit: int) -> dict:
    """The prover's opening of unit `unit`: its entry with its salt and path to the window root, and each side's row with
    its path to the entry's side root."""
    e, i, j = Wd.tile_at(advice["entries"], unit)
    entry = advice["entries"][e]
    out = {"unit": unit, "entry": e, "entry_value": entry, "entry_salt": advice["entry_salts"][e],
           "window_path": advice["window_path"][e]}
    for side, idx in (("a", i), ("b", j)):
        dom, schema = side_domain(params, advice["level"], e, side, entry[side]["rows"])
        levels = tree_levels(dom, side_leaves(dom, schema, priv[e][side]["bcs"]))
        out[side] = {"index": idx, "path": path(levels, idx)}
    return out


def check(params: dict, rec: dict, draw: dict, unit: int, pub: bytes, op: dict, drawn: bool = True) -> str | None:
    """None if the proof's public file opens unit `unit`'s rows under the registered window root, else why not."""
    if drawn and unit not in draw.get("units", ()):
        return f"unit {unit} is not drawn"
    try:
        ell = level_of(rec)
        work_at(params, ell)
    except S.Refused as e:
        return str(e)
    roots = {r["port"]: r for r in rec["roots"]}
    dom = window_domain(params, ell)
    if roots.get("window", {}).get("domain") != dom.domain_id.hex():
        return "the registered window domain is not the window's served domain"
    e, entry = op.get("entry"), op.get("entry_value")
    if not isinstance(e, int) or not isinstance(entry, dict):
        return "the opening names no entry"
    leaf = entry_leaf(dom, e, entry_bc(entry, bytes.fromhex(op["entry_salt"])))
    if not verify_path(dom, bytes.fromhex(roots["window"]["root"]), e, leaf, [bytes.fromhex(x) for x in op["window_path"]]):
        return f"entry {e} is not the registered window's entry"
    try:
        at = dict(zip(("a", "b"), Wd.tile_in(entry, unit)))
    except (ValueError, KeyError, IndexError) as x:
        return f"entry {e}: {x}"
    head, body = pub.split(b"\n", 1)
    h = json.loads(head)
    if h.get("instances") != 1:
        return f"the proof has {h.get('instances')} instances, not one tile"
    want = [L.ROW_SCHEMA["fp8-p", SIDES[s]] for s in ("a", "b")]
    if [s for _p, s in list(h["frame_v3"]["schemas"].items())[:2]] != want:
        return "the proof's first rows are not an A row and a B row of the registered schemas"
    bc = {"a": body[0:128], "b": body[128:256]}
    for side in ("a", "b"):
        o = op[side]
        if o["index"] != at[side]:
            return f"side {side}: the opening is of row {o['index']}, not the unit's row {at[side]}"
        sdom, schema = side_domain(params, ell, e, side, entry[side]["rows"])
        leaf = row_leaf(sdom, at[side], L.tree_leaf(bc[side]), schema)
        if not verify_path(sdom, bytes.fromhex(entry[side]["root"]), at[side], leaf, [bytes.fromhex(x) for x in o["path"]]):
            return f"side {side}: the proof's committed row is not entry {e}'s row {at[side]}"
    return None


# ---- verdicts --------------------------------------------------------------------------------------------------------------

def lean_verdict(path_: Path) -> dict:
    """The Lean verifier's verdict from its output (`VERDICT {…}`), of record; accepted null when it printed none."""
    try:
        lines = [json.loads(x[8:]) for x in path_.read_text().splitlines() if x.startswith("VERDICT ")]
    except (OSError, ValueError):
        lines = []
    if not lines:
        return {"of_record": True, "accepted": None, "verifier": "lean flock-verify verify --zk"}
    v = lines[0]
    return {"of_record": True, "accepted": bool(v.get("accepted")), "why": v.get("why", ""), "verifier": "lean flock-verify verify --zk",
            "digest": hashlib.sha512(canonical_json(v)).hexdigest()}


def rust_verdict(path_: Path | None) -> dict | None:
    """The Rust prover-side verifier's verdict of the --zk session (`LIVE\\t{…}` lines), beside the verifier of record."""
    if path_ is None:
        return None
    try:
        lines = [json.loads(x[5:]) for x in path_.read_text().splitlines() if x.startswith("LIVE\t")]
    except (OSError, ValueError):
        lines = []
    v = lines[0].get("verdict", lines[0]) if lines else None
    return {"of_record": False, "accepted": None if v is None else bool(v.get("accepted")), "why": (v or {}).get("why", ""),
            "verifier": "rust flock-circuit --zk (live)"}


def statement(stage: dict, pub: bytes) -> dict:
    """A unit's statement by digest: its template, circuit and class, its public file, its size; nothing of its tile."""
    return {"definition": stage["definition"], "circuit_sha512": stage["circuit_sha512"], "class_sha512": stage["class_sha512"],
            "public_sha512": hashlib.sha512(pub).hexdigest(), "ands": stage["ands"], "k_log": stage["k_log"],
            "binding": stage["binding"]}


def outcome(params: dict, rec: dict, receipt: dict, draw: dict, units: list[dict], log: Path) -> tuple[S.Outcome, dict]:
    spec = spec_of(params, level_of(rec))
    o, audit = S.outcome(spec, rec, receipt, draw, units, float(Fraction(params["delta"])), log, {})
    return o, audit


# ---- the CLI ------------------------------------------------------------------------------------------------------------

def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def _pass(pass_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    from verity_vllm.protocol_options import pouw_pearl_c_device as PD
    raw = (pass_dir / "manifest.json").read_bytes()
    man = json.loads(raw)
    if man.get("schema") != PD.MANIFEST_SCHEMA:
        raise SystemExit(f"{pass_dir}: not a retained pass ({man.get('schema')!r})")
    calls = [u for u in man["matmuls"] if u["k"] == K and u["n"] == K]
    if not calls:
        raise SystemExit(f"{pass_dir}: no call of k = n = {K}")
    return PD, man, hashlib.sha256(raw).hexdigest(), calls


def _weights(man: dict, calls: list[dict]) -> list[int]:
    return sorted({int(u["weight"]) for u in calls})


def cmd_params(a) -> int:
    """The verifier's own parameters, from the served request it audits: its run and manifest name the window, and the
    model's k = n = K weights give |W|."""
    _, man, sha, calls = _pass(Path(a.pass_dir))
    rows = sum(next(u["n"] for u in calls if int(u["weight"]) == w) for w in _weights(man, calls))
    deployment = f"served-zk/{man.get('run')}"
    params = {"context": f"{deployment}/{sha[:16]}", "deployment": deployment, "k": K, "eps": a.eps, "delta": a.delta,
              "grid_bits": Wd.GRID_BITS, "range": list(RANGE), "entries": a.entries, "w_rows": rows}
    Path(a.out).write_text(json.dumps(params, sort_keys=True))
    print(json.dumps(params), flush=True)
    return 0


def cmd_commit(a) -> int:
    pass_dir = Path(a.pass_dir)
    PD, man, sha, calls = _pass(pass_dir)
    import fast_tile as FT
    params = _json(a.params)
    scheme = PD.scheme_of(man["scheme"])
    source = os.urandom(32)
    if a.call:
        chosen = [next(x for x in calls if x["index"] == i) for i in a.call]
    else:
        order = sorted(calls, key=lambda u: hashlib.sha256(source + u["index"].to_bytes(8, "little")).digest())
        chosen = sorted(order[:a.calls], key=lambda u: u["index"])
    undo = FT.install(scheme)
    window = []
    try:
        for u in chosen:
            w = man["weights"][str(u["weight"])]
            k, n, real = u["k"], u["n"], u["real_rows"]
            ctx = Context(bytes.fromhex(man["salt"]), Matmul(u["index"], Shape(u["m"], k, n), u["weight"]),
                          bytes.fromhex(u["root_a"]), bytes.fromhex(w["root"]))
            noise = scheme.noise(ctx)
            a_data = (pass_dir / u["rows"]["path"]).read_bytes()
            b_data = np.memmap(pass_dir / w["rows"]["path"], dtype=np.uint8, mode="r")
            words = {"a": form(scheme, noise, k, "a", f32_rows(a_data, k, range(real)), list(range(real))),
                     "b": form(scheme, noise, k, "b", f32_rows(b_data, k, range(n)), list(range(n)))}
            record = {"matmul": u["index"], "shape": [u["m"], k, n], "weight": u["weight"], "real_rows": real,
                      "served": {"root_a": u["root_a"], "weight_root": w["root"]}}
            window.append({"record": record, "words": words})
            print(json.dumps({"call": u["index"], "excluded": {s: sum(x is None for x in v) for s, v in words.items()}}), flush=True)
    finally:
        undo()

    def w_rows():
        for wid in _weights(man, calls):
            data = np.memmap(pass_dir / man["weights"][str(wid)]["rows"]["path"], dtype=np.uint8, mode="r")
            n = next(u["n"] for u in calls if int(u["weight"]) == wid)
            for j in range(n):
                yield data[2 * K * j:2 * K * (j + 1)]
    root_w, _ = commit_weights(params, w_rows())
    public, advice, priv = commit_window(params, window, root_w)
    advice["provenance"] = {"run": man.get("run"), "manifest_sha256": sha, "scheme": man["scheme"],
                            "transcript_root": man.get("transcript_root"), "commitment": man.get("commitment"),
                            "records_root": man.get("records_root"),
                            "call_draw": {"source": source.hex(), "among": [x["index"] for x in calls]} if not a.call else {"given": a.call}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_text(json.dumps(public, sort_keys=True))
    (out / "advice.json").write_text(json.dumps(advice, sort_keys=True))
    save(out / "prover.npz", priv)
    print(json.dumps({"calls": [c["record"]["matmul"] for c in window], "real": advice["real"], "work": public["items"]["work"],
                      "law": public["items"]["sampling"]["law"], "filler": advice["entries"][len(window)]["hi"] - advice["real"],
                      "roots": {r["port"]: r["root"][:16] for r in public["roots"]}}), flush=True)
    return 0


def cmd_register(a) -> int:
    c, params = _json(Path(a.commit_dir) / "commitment.json"), _json(a.params)
    items = dict(c["items"])
    for x in a.extra or ():
        key, _, value = x.partition("=")
        items[key] = json.loads(value)
    try:
        rec = S.register(spec_of(params, c["level"]), items, c["roots"])
    except S.Refused as e:
        print(f"REFUSED {e}", flush=True)
        return 0 if a.expect_refused else 1
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    print(f"REGISTERED {json.dumps({'registration': R.digest(rec)[:32], 'population': rec['population'], 'law': rec['law'], 'public': sorted(rec['window']['public'])})}",
          flush=True)
    return 1 if a.expect_refused else 0


def receive(params: dict, rec: dict) -> tuple[dict, dict | None]:
    """The verifier's checks against its own parameters: the registered level is in its range, and every item but the
    commitments is its own recomputation at that level."""
    try:
        ell = level_of(rec)
        spec = spec_of(params, ell)
        expect = items_of(params, ell, None)
    except S.Refused as e:
        return {"ok": False, "codes": ["level"], "detail": {"level": str(e)}}, None
    return S.receive(spec, rec, ports(params, ell), expect, verifier=VERIFIER)


def cmd_receive(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v, receipt = receive(_json(a.params), _json(a.registration))
    (out / "check.json").write_text(json.dumps(v, sort_keys=True))
    if receipt is not None:
        (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    print(f"{'RECEIVED' if receipt else 'REFUSED'} {json.dumps(v)}", flush=True)
    return 0 if (receipt is not None) != a.expect_refused else 1


def cmd_edit(a) -> int:
    """A control's registration: `registration` with each dotted KEY set to JSON (a public item added, a level raised)."""
    rec = _json(a.registration)
    for x in a.set:
        key, _, value = x.partition("=")
        *up, last = key.split(".")
        at = rec
        for k in up:
            at = at[k]
        at[last] = json.loads(value)
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    return 0


def lean_sampler(lean: str):
    def sample(population: int, k: int) -> dict:
        p = subprocess.run([lean, "draw", "--population", str(population), "--k", str(k)], capture_output=True, text=True)
        lines = [x[5:] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
        if p.returncode != 0 or len(lines) != 1:
            raise S.Refused(f"flock-verify draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
        return json.loads(lines[0])
    return sample


def cmd_select(a) -> int:
    params, rec, receipt = _json(a.params), _json(a.registration), _json(a.receipt)
    try:
        draw = S.select(spec_of(params, level_of(rec)), rec, receipt, lean_sampler(a.lean), Path(a.log))
    except S.Refused as e:
        print(f"REFUSED {e}", flush=True)
        return 0 if a.expect_refused else 1
    Path(a.out).write_text(json.dumps(draw, sort_keys=True))
    print(f"DRAWN {json.dumps(draw)}", flush=True)
    return 1 if a.expect_refused else 0


def cmd_check(a) -> int:
    params, reg, d = _json(a.params), _json(a.registration), _json(a.draw)
    why = check(params, reg, d, a.unit, Path(a.pub).read_bytes(), _json(a.opening), drawn=not a.undrawn)
    if a.out:
        Path(a.out).write_text(json.dumps({"of_record": False, "accepted": why is None, "why": why or "",
                                           "verifier": "served_zk opening check"}, sort_keys=True))
    print(f"ACCEPT {a.pub} opens unit {a.unit} of {a.registration}" if why is None else f"REJECT {a.pub}: {why}", flush=True)
    return 0 if (why is None) != a.expect_reject else 1


def cmd_outcome(a) -> int:
    params, reg, receipt, d = _json(a.params), _json(a.registration), _json(a.receipt), _json(a.draw)
    units = []
    for u in _json(a.units):
        pub = Path(u["pub"]).read_bytes()
        units.append({"unit": u["unit"], "statement": statement(_json(u["stage"]), pub),
                      "verdicts": {"lean": lean_verdict(Path(u["lean"])), "opening": _json(u["opening"]),
                                   **({"rust": rust_verdict(Path(u["rust"]))} if u.get("rust") else {})}})
    try:
        o, audit = outcome(params, reg, receipt, d, units, Path(a.log))
    except S.Refused as e:
        print(f"REFUSED {e}", flush=True)
        return 0 if a.expect_refused else 1
    if a.expect_refused:
        print("NOT REFUSED", flush=True)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "outcome.json").write_text(json.dumps({"accepted": o.accepted, "public": o.public, "profile": o.profile,
                                                  "terms": o.terms, "record": o.record, "verdict": o.verdict}, sort_keys=True))
    (out / "audit.json").write_text(json.dumps(audit, sort_keys=True))
    print(f"OUTCOME {'ACCEPT' if o.accepted else 'REJECT'} {json.dumps({k: v.get('accepted') for k, v in audit['verdicts'].items()})} "
          f"public {json.dumps(sorted(o.public))}", flush=True)
    return 0 if o.accepted != a.expect_reject else 1


def shape(d1: Path, d2: Path) -> dict:
    """What two staged units' proofs show the verifier beyond their bytes: circuit, class, the public file's and each
    session file's size."""
    def one(d: Path) -> dict:
        s = _json(d / "stage.json")
        sess = sorted((d / "sessions").glob("*/")) if (d / "sessions").is_dir() else []
        files = sorted((p.relative_to(sess[-1]).as_posix(), p.stat().st_size) for p in sess[-1].rglob("*") if p.is_file()) if sess else []
        return {"circuit_sha512": s["circuit_sha512"], "class_sha512": s["class_sha512"], "ands": s["ands"], "k_log": s["k_log"],
                "pub_bytes": (d / "pub.bin").stat().st_size, "session_files": files}
    x, y = one(d1), one(d2)
    return {"same": x == y and bool(x["session_files"]), "a": x, "b": y}


def cmd_shape(a) -> int:
    v = shape(Path(a.a), Path(a.b))
    print(f"{'SAME' if v['same'] else 'DIFFERENT'} {json.dumps({k: v['a'][k] for k in ('circuit_sha512', 'pub_bytes')})} "
          f"{len(v['a']['session_files'])} session files", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(v, sort_keys=True))
    return 0 if v["same"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("params")
    p.add_argument("pass_dir")
    p.add_argument("out")
    p.add_argument("--eps", default=EPS)
    p.add_argument("--delta", default=DELTA)
    p.add_argument("--entries", type=int, default=ENTRIES)
    p = sub.add_parser("commit")
    for x in ("pass_dir", "params", "out"):
        p.add_argument(x)
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--call", type=int, action="append")
    p = sub.add_parser("register")
    for x in ("commit_dir", "params", "out"):
        p.add_argument(x)
    p.add_argument("--extra", action="append", help="ITEM=JSON: one more public item, which register refuses")
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("receive")
    for x in ("params", "registration", "out"):
        p.add_argument(x)
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("edit")
    p.add_argument("registration")
    p.add_argument("out")
    p.add_argument("set", nargs="+", help="KEY=JSON, KEY dotted")
    p = sub.add_parser("select")
    for x in ("params", "registration", "receipt", "out"):
        p.add_argument(x)
    p.add_argument("--lean", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("check")
    for x in ("params", "registration", "draw"):
        p.add_argument(x)
    p.add_argument("unit", type=int)
    p.add_argument("pub")
    p.add_argument("opening")
    p.add_argument("--out", help="the opening check's verdict, for the outcome")
    p.add_argument("--undrawn", action="store_true", help="a control: open a unit the draw did not draw")
    p.add_argument("--expect-reject", action="store_true")
    p = sub.add_parser("outcome")
    for x in ("params", "registration", "receipt", "draw", "units", "out"):
        p.add_argument(x)
    p.add_argument("--log", required=True)
    p.add_argument("--expect-reject", action="store_true")
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("shape")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    a = ap.parse_args(argv)
    return {"params": cmd_params, "commit": cmd_commit, "register": cmd_register, "receive": cmd_receive,
            "edit": cmd_edit, "select": cmd_select, "check": cmd_check, "outcome": cmd_outcome, "shape": cmd_shape}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
