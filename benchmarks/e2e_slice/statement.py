"""The round's spot check as a C-Flock statement: GemmCoordinate_v3{K = 128, HopperBF16WgmmaDot16} (Boolean), one instance a
drawn coordinate (i, l) of the drawn call's tile at step j, staged by `class_statement.stage_unit` with every instance's own
rows (no shared rows, no public input ports), its output hidden (one hm96-sha512/row/v2 output row).

    x    = [A_i,S | U_i | A_i,S | U_i]      S = columns 32 j .. 32 j + 31, U_i U's row i (U_{i, k mod 32} over S)
    wrow = [B_l,S | B_l,S | F_l,S | F_l,S]
    out  = bf16(x . wrow) = bf16((A' B'^T)_il's step-j increment), exact in the f32 accumulator (|increment| < 2^19)

Its four input rows, each 64 bf16 words (1024 bits), hm96-sha512/row/v2:
    xa  x[0..31], x[64..95]    = A_i,S twice: operand A's committed leaf (i, j) (commit.py), the committer's salt
    xu  x[32..63], x[96..127]  = U_i twice: the coin's noise, salt public (public_salt)
    wb  wrow[0..63]            = B_l,S twice: operand B's committed leaf (l, j), the committer's salt
    wf  wrow[64..127]          = F_l,S twice: the coin's noise, salt public
so the challenger holds xa and wb to the trees committed before the coin, and recomputes xu and wf's b || c from the coin.
What it doesn't bind: the receipts (no Definition recomputes a duo receipt), and the output, a bf16 the verifier never sees.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import commit as CM
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS

K = 128
NAME = "unit/e2e-slice-oproj-step-k128"
PORTS = ("xa", "xu", "wb", "wf")
IN_ROWS = [("xa", V2, 1024, [f"x[{r}]" for r in (*range(0, 32), *range(64, 96))]),
           ("xu", V2, 1024, [f"x[{r}]" for r in (*range(32, 64), *range(96, 128))]),
           ("wb", V2, 1024, [f"wrow[{r}]" for r in range(64)]),
           ("wf", V2, 1024, [f"wrow[{r}]" for r in range(64, 128)])]
SALT_TAG = b"verity/e2e-slice/public-salt/v1\0"


def definition():
    return bind(BG.GemmCoordinate, K=K, **BG.GEMM_HOPPER)


def public_salt(sigma: bytes, call: int, port: str, inst: int) -> bytes:
    return hashlib.shake_256(SALT_TAG + sigma + call.to_bytes(4, "big") + port.encode() + b"\0" + inst.to_bytes(4, "big")).digest(CM.SALT)


def twice(v: np.ndarray) -> np.ndarray:
    w = CM.bf16_words(v)
    return np.concatenate([w, w], axis=-1)


def noise_rows(U: np.ndarray, F: dict, draw: dict) -> dict:
    """xu and wf per instance: U's rows (m x 32) and F's rows ({l: F_l (k,)}) at the drawn coordinates and step."""
    j = draw["step"]
    return {"xu": np.stack([twice(U[i]) for i, _l in draw["coords"]]),
            "wf": np.stack([twice(F[l][32 * j:32 * j + 32]) for _i, l in draw["coords"]])}


def bf16_rne(v: np.ndarray) -> np.ndarray:
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def stage(out: Path, A: np.ndarray, B: np.ndarray, U: np.ndarray, F: dict, draw: dict, sigma: bytes, salts: dict,
          salt_key: bytes) -> dict:
    """Stages the drawn spot check under ``out`` (circuit.txt, inst.bin, pub.bin) from the call's A, B, its U and F rows;
    ``salts`` xa's and wb's committed salts per instance. Each output is checked against the exact step increment in bf16."""
    j, coords = draw["step"], draw["coords"]
    rows = {"xa": np.stack([twice(A[i, 32 * j:32 * j + 32]) for i, _l in coords]),
            "wb": np.stack([twice(B[l, 32 * j:32 * j + 32]) for _i, l in coords]), **noise_rows(U, F, draw)}
    pub = {p: [public_salt(sigma, draw["call"], p, n) for n in range(len(coords))] for p in ("xu", "wf")}
    got = CS.stage_unit(definition(), NAME, IN_ROWS, [(V2, (0,))], rows, out, salt_key, salts={**salts, **pub})
    inc = np.array([int((A[i, 32 * j:32 * j + 32].astype(np.int64) + U[i]) @ (B[l, 32 * j:32 * j + 32].astype(np.int64)
                                                                              + F[l][32 * j:32 * j + 32])) for i, l in coords])
    o0 = np.asarray(got["outs"]["o0"], np.uint16)[:, 0]
    if not np.array_equal(o0, bf16_rne(inc.astype(np.float32))):
        raise AssertionError("the statement's outputs are not the step increments in bf16")
    text = (out / "circuit.txt").read_text()
    return {"pin": FC.digest(text), "ands": got["ands"], "unit_nnz": got["unit_nnz"], "k_log": got["meta"]["k_log"],
            "instances": len(coords), "circuit_bytes": len(text), "increments": [int(x) for x in inc]}


def ports(text: str) -> list[str]:
    """The circuit's row ports in META order (the public file's order)."""
    return [p.name for p in FC._meta_ports(json.loads(text.split("\n", 2)[1][5:]))]


def public_rows(pub: Path, names: list[str]) -> tuple[dict, dict]:
    """The public file's header and every row's b || c by port: {port: [bc, ...]} (instance-major, ``names`` in META order)."""
    raw = pub.read_bytes()
    head_b, body = raw.split(b"\n", 1)
    head = json.loads(head_b)
    n = head["instances"]
    if len(body) != 128 * n * len(names):
        raise ValueError(f"public file: {len(body)} bytes of b || c, not {n} instances x {len(names)} rows")
    return head, {p: [body[128 * (i * len(names) + q):128 * (i * len(names) + q + 1)] for i in range(n)] for q, p in enumerate(names)}
