"""The gate on a `chain gate` dump: every arm's X_1 .. X_S bit for bit against the CPU chain, the lt arm's row digests and
roots against `rowleaf.sha512_row_seg_digest` and `audit.commit_rows` (HM96, gen.py's salt key and domains), and the
control arm (step 0's GEMM skipped, its output left poisoned) rejected. One JSON line; exit 1 on any failure.

The CPU chain: C_s = X_s W_s in float64, exact because |C| <= k 128^2 <= 2^27 < 2^53, then C_s's s32 words little-endian as
X_{s+1}'s bytes.

    python ref.py DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw import audit
from verity.ml.commitments.rowleaf import sha512_row_seg_digest

from gen import SALT_KEY, binding


def chain(x0: np.ndarray, ws: list[np.ndarray]) -> list[np.ndarray]:
    """X_1 .. X_S from X_0 (m x k s8) and W_s^T (n x k s8)."""
    xs, x = [], x0
    for wt in ws:
        c = x.astype(np.float64) @ wt.astype(np.float64).T
        if np.abs(c).max() >= 2**31:
            raise ValueError("an s32 word saturates")
        x = c.astype(np.int64).astype("<i4").view(np.int8).reshape(x.shape)
        xs.append(x)
    return xs


def byte_stats(x: np.ndarray) -> list[dict]:
    """Per byte position of an s32 word (0 low .. 3 high): distinct values and the empirical entropy in bits."""
    out = []
    for p in range(4):
        v = x[:, p::4].ravel()
        _, counts = np.unique(v, return_counts=True)
        q = counts / counts.sum()
        out.append({"byte": p, "distinct": int(len(counts)), "entropy_bits": round(float(-(q * np.log2(q)).sum()), 3)})
    return out


def main(d: Path) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, k, steps = meta["m"], meta["k"], meta["S"]
    n = k // 4
    x0 = np.fromfile(d / "X0.bin", np.int8).reshape(m, k)
    ws = [np.fromfile(d / f"W{s}.bin", np.int8).reshape(n, k) for s in range(steps)]
    ref = chain(x0, ws)
    res: dict = {"dir": str(d), "m": m, "k": k, "S": steps, "elements": m * k * steps}
    ok = True
    for arm in meta["arms"]:
        bad = [int((np.fromfile(d / f"{arm}_X{s}.bin", np.int8).reshape(m, k) != ref[s - 1]).sum()) for s in range(1, steps + 1)]
        res[f"{arm}_mismatches"] = bad
        ok &= not any(bad)
    ctl = [int((np.fromfile(d / f"{meta['control']}_X{s}.bin", np.int8).reshape(m, k) != ref[s - 1]).sum()) for s in range(1, steps + 1)]
    res["control_mismatches"] = ctl
    res["control_rejected"] = bool(ctl[0])
    ok &= res["control_rejected"]
    dig_bad, roots_ok = 0, []
    for j in range(1, steps + 1):
        rows = [ref[j - 1][i].tobytes() for i in range(m)]
        dev = np.fromfile(d / f"rowdig{j}.bin", np.uint8).reshape(m, 64)
        dig_bad += sum(sha512_row_seg_digest(r, 8 * len(r)) != dev[i].tobytes() for i, r in enumerate(rows))
        tree = audit.commit_rows(binding(j), j, rows, audit.HM96, SALT_KEY, nonce=j)
        roots_ok.append(tree.commitment.root == (d / f"root{j}.bin").read_bytes())
    res["rowdig_mismatches"] = int(dig_bad)
    res["roots_match"] = roots_ok
    ok &= dig_bad == 0 and all(roots_ok)
    res["max_abs_c_log2"] = round(float(np.log2(np.abs(ref[-1].view("<i4").astype(np.int64)).max() + 1)), 2)
    res["x_last_bytes"] = byte_stats(ref[-1])
    res["pass"] = bool(ok)
    print(json.dumps(res))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
