"""Off the card: form.py's outputs against #295's own D-3s forming, and the bound unit's words against NCP's chain on them.

- The check runs (OUT/form/check/): every formed row, whole, against #295's `D3Family("d3s", "xq", "round11", "f16").side`
  with its own `ncp_perm` and its own stream (`D3Family.stream`: SHAKE256), or BLAKE3's XOF of the same message; and the
  first four rows' raw streams against the same streams (BLAKE3's also against the `blake3` package).
- The unit (OUT/form/arms.json): check.py's `check_arms`, with every sampled lane's operand rows formed by #295 from
  gen_any's X_q and W, in place of check.py's stand-ins; the negative control must fail every comparison.

#295's forming runs in a subprocess on PR #295's tree (PR295, default /tmp/pr295; its verity has `verity.ml.tc.fp16` and
not the sm_120 atom, the workspace's the reverse).

  uv run python benchmarks/pouw/ncp_sm120/check_form.py OUT [PR295]     exit 0 when every comparison matches
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def codes_any(seed: int, rows: np.ndarray, k: int) -> np.ndarray:
    """gen_any's bytes for the given rows of a (· x k) buffer: mix32(idx * 0x9E3779B9 + seed) & 0xFF, NaN codes xor 1."""
    idx = (rows.astype(np.uint64)[:, None] * np.uint64(k) + np.arange(k, dtype=np.uint64)[None, :]).astype(np.uint32)
    x = idx * np.uint32(0x9E3779B9) + np.uint32(seed)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    c = (x & 0xFF).astype(np.uint8)
    return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)


# ---- the reference, in the subprocess on #295's tree ---------------------------------------------------------------------

_REF = {}


def _ref_init(context: dict) -> None:
    from verity.commitments import blake3
    from verity.randomness import frame
    from verity_pouw.protocol import Context, Matmul, Shape
    from verity_pouw.schemes import fp8_is

    fam = fp8_is.D3s("xq", "round11", "f16")
    assert fam.tag == "d3s-v0-f16", fam.tag
    _REF.update(fam=fam, fp8_is=fp8_is, blake3=blake3, frame=frame, Context=Context, Matmul=Matmul, Shape=Shape,
                context={k: bytes.fromhex(v) if isinstance(v, str) else v for k, v in context.items()})


def _b3_xof(data: bytes, n: int) -> bytes:
    """BLAKE3's XOF of a one-chunk message (the spec's root output blocks, counter 0, 1, ...)."""
    b3 = _REF["blake3"]
    blocks = [data[i:i + 64] for i in range(0, len(data), 64)] or [b""]
    cv = b3.IV
    for i, blk in enumerate(blocks[:-1]):
        cv = tuple(b3.compress(cv, blk, 0, 64, b3.CHUNK_START if i == 0 else 0)[:8])
    last = blocks[-1]
    flags = b3.CHUNK_END | (b3.CHUNK_START if len(blocks) == 1 else 0) | b3.ROOT
    out = b"".join(w.to_bytes(4, "little") for i in range(-(-n // 64))
                   for w in b3.compress(cv, last.ljust(64, b"\0"), i, len(last), flags))
    return out[:n]


def _stream(xof: str, left: bool, k: int, line: int) -> bytes:
    fam, c = _REF["fam"], _REF["context"]
    name = "E1" if left else "F1"
    if xof == "shake":
        ctx = _REF["Context"](c["salt"], _REF["Matmul"](c["index"], _REF["Shape"](1, k, 1), c["weight"]), c["root_a"], b"")
        return fam.stream(name, ctx, line)
    frame = _REF["frame"]
    msg = f"verity/pouw/fp8-is/{fam.tag}/{name}".encode() + b"\0" + b"".join(
        frame(p) for p in (c["salt"], c["index"], c["root_a"], c["weight"], line))
    return _b3_xof(msg, 2 * k)


def _form(job: tuple) -> tuple:
    """(xof, left, k, line, codes) -> the formed row (3k codes) and its stream."""
    xof, left, k, line, codes = job
    fam, fp8_is = _REF["fam"], _REF["fp8_is"]
    pi, inv = fp8_is.ncp_perm(k, fam.P_SEED)
    h = fam.parse(bytes(codes), k, left)
    stream = _stream(xof, left, k, line)
    return bytes(fam.side(h, stream, pi, inv, left)), stream


def reference(req_path: str, out_path: str) -> None:
    req = pickle.load(open(req_path, "rb"))
    workers = max(1, (os.cpu_count() or 2) - 1)
    with cf.ProcessPoolExecutor(workers, initializer=_ref_init, initargs=(req["context"],)) as ex:
        got = list(ex.map(_form, req["jobs"], chunksize=1))
    pickle.dump(got, open(out_path, "wb"))


def run_reference(out: str, pr295: str, context: dict, jobs: list[tuple], name: str) -> list[tuple]:
    req, res = os.path.join(out, "form", f"{name}.req.pkl"), os.path.join(out, "form", f"{name}.ref.pkl")
    pickle.dump({"context": context, "jobs": jobs}, open(req, "wb"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(pr295, "packages", "verity", "src"),
                                                      os.path.join(pr295, "protocols", "pouw")]))
    py = os.path.join(HERE, "..", "..", "..", ".venv", "bin", "python")
    r = subprocess.run([py if os.path.exists(py) else sys.executable, os.path.abspath(__file__), "--reference", req, res],
                       env=env, cwd="/", capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"#295's reference failed:\n{r.stderr[-3000:]}")
    return pickle.load(open(res, "rb"))


# ---- the comparisons, in the workspace's verity --------------------------------------------------------------------------

def check_runs(out: str, pr295: str) -> list[str]:
    import blake3 as b3pkg

    where = os.path.join(out, "form", "check")
    meta = json.load(open(os.path.join(where, "meta.json")))
    rows = meta["rows"]
    jobs, index = [], []
    for run in meta["runs"]:
        k = run["k"]
        codes = np.frombuffer(open(os.path.join(where, run["tag"] + ".codes.bin"), "rb").read(), dtype=np.uint8).reshape(rows, k)
        for r in range(rows):
            jobs.append((run["xof"], run["left"], k, r, codes[r].tobytes()))
            index.append((run["tag"], r))
    ref = dict(zip(index, run_reference(out, pr295, meta["context"], jobs, "check")))
    fails = []
    for run in meta["runs"]:
        k, tag = run["k"], run["tag"]
        got = open(os.path.join(where, tag + ".formed.bin"), "rb").read()
        raw = open(os.path.join(where, tag + ".xof.bin"), "rb").read()
        bad_rows = [r for r in range(rows) if got[3 * k * r:3 * k * (r + 1)] != ref[(tag, r)][0]]
        bad_codes = sum(int((np.frombuffer(got[3 * k * r:3 * k * (r + 1)], np.uint8) != np.frombuffer(ref[(tag, r)][0], np.uint8)).sum())
                        for r in bad_rows)
        bad_xof = [r for r in range(4) if raw[2 * k * r:2 * k * (r + 1)] != ref[(tag, r)][1]]
        if run["xof"] == "b3":
            for r in range(4):
                msg = form_message(meta, run["left"], r)
                if b3pkg.blake3(msg).digest(length=2 * k) != ref[(tag, r)][1]:
                    fails.append(f"{tag}: the reference BLAKE3 XOF of row {r} differs from the blake3 package")
        if bad_rows:
            fails.append(f"{tag}: rows {bad_rows} differ ({bad_codes} of {len(bad_rows) * 3 * k} codes)")
        if bad_xof:
            fails.append(f"{tag}: raw streams of rows {bad_xof} differ")
        print(f"form {tag}: {rows} rows x {3 * k} codes {'ok' if not bad_rows else 'DIFFER'}, "
              f"4 raw streams {'ok' if not bad_xof else 'DIFFER'}", flush=True)
    return fails


def form_message(meta: dict, left: bool, line: int) -> bytes:
    from verity.randomness import frame

    c = {k: bytes.fromhex(v) if isinstance(v, str) else v for k, v in meta["context"].items()}
    return f"verity/pouw/fp8-is/{meta['tag']}/{'E1' if left else 'F1'}".encode() + b"\0" + b"".join(
        frame(p) for p in (c["salt"], c["index"], c["root_a"], c["weight"], line))


def check_unit(out: str, pr295: str) -> list[str]:
    spec = importlib.util.spec_from_file_location("ncp_check", os.path.join(HERE, "check.py"))
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    res = json.load(open(os.path.join(out, "form", "arms.json")))
    jobs, need = [], {}
    for shape in res["shapes"]:
        k, xof = shape["k"], shape["xof"]
        a_lines, b_lines = set(), set()
        for arm in shape["arms"]:
            for ln in arm["lanes"]:
                rows, cols, _ = check.lane_cells(shape, ln["tm"], ln["tn"], ln["tid"])
                a_lines |= {int(r) for r in rows if r < shape["rows"]}
                b_lines |= {int(c) for c in cols}
        for left, lines, seed in ((True, sorted(a_lines), shape["seeds"][0]), (False, sorted(b_lines), shape["seeds"][1])):
            codes = codes_any(seed, np.array(lines, dtype=np.int64), k) if lines else np.zeros((0, k), np.uint8)
            for line, row in zip(lines, codes):
                need[(shape["kind"], xof, left, line)] = len(jobs)
                jobs.append((xof, left, k, line, row.tobytes()))
    print(f"unit: #295 forms {len(jobs)} rows for the sampled lanes", flush=True)
    got = run_reference(out, pr295, res["shapes"][0]["context"], jobs, "unit")

    def operands(meta: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = meta["K"]
        pick = lambda left, line: np.frombuffer(got[need[(meta["kind"], meta["xof"], left, int(line))]][0], np.uint8)
        a = np.stack([pick(True, r) if r < meta["rows"] else np.zeros(K, np.uint8) for r in rows])
        return a, np.stack([pick(False, c) for c in cols])

    check.operands = operands
    return check.check_arms(os.path.join(out, "form"))


def main(argv: list[str]) -> int:
    if argv[1] == "--reference":
        reference(argv[2], argv[3])
        return 0
    out = argv[1]
    pr295 = argv[2] if len(argv) > 2 else os.environ.get("PR295", "/tmp/pr295")
    fails = []
    if os.path.exists(os.path.join(out, "form", "check", "meta.json")):
        fails += check_runs(out, pr295)
    if os.path.exists(os.path.join(out, "form", "arms.json")):
        fails += check_unit(out, pr295)
    for f in fails:
        print("FAIL", f)
    print("all match" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
