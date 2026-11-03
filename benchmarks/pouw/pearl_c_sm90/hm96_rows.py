"""One-stage's commitment of a Pearl+ call's A, timed beside the call on one H100 or H200: an added pass, timing only, on
no verdict's path.  A call's M x K BF16 A is M rows of 2K bytes (`rowleaf.bit_row`'s layout of BF16 elements), each a hiding
row under ``hm96-sha512/row/v2``, in one frame-v3-sha512 tree under one-stage's window domain for a port of M leaves
(`window.window_domain`, the port's fields `PORT`):

  x     SHA-512(P(16 K) || row)                                    sha512/row/v2
  y     SHA-512(Ks || u64be(i) || u64be(r)), r = 0, 1, 2           the served committer's derived salts (`step_rows.salt`)
  h     hm96.SHA512.tree_leaf(default key, commit_string(x, y))
  leaf  the frame-v3-sha512 row leaf of h at rank i, the tree padded to a power of two with the domain's pads

hashed by the served committer's kernels (integrations/vllm's native_rows.cu over verity/ml/kernels' row_commit.cuh and
sha512.cuh), which `build` compiles alone into a shared library with its own CUDA runtime, driven here through ctypes on
torch's current stream.  It is its own pass over A, not fused into Pearl+'s row digest.

  python hm96_rows.py build OUT [--cuda /usr/local/cuda-13.0]     OUT/hm96_rows.so, ptxas.txt and build.json
  python hm96_rows.py time --lib SO --out DIR --m M ... [--step-m M ... --layers L] [--model llama32_1b] [--reps 15]
  python hm96_rows.py report HM96_JSON --timing TIMING_JSON [--whole WHOLE_JSON] [--out REPORT_JSON]

`time` first holds the kernels to `reference` (rowleaf's row digest, the hm96 gates' leaf, the frame-v3 gates' leaf and node,
the domain's pads) on `VECTORS` and on every point's A under every form, and times nothing when any leaf or root differs.
Then, at each linear of `--model` and each M, it times the commitment's leaves and tree under each leaf form of `FORMS`
(native_rows.cu's ``VERITY_ROWS_FORM``), eagerly and replayed as a CUDA graph, interleaved rep by rep, each sample `INNER`
commitments between events on the current stream; and at each M of `--step-m`, a decode step's commitments over L layers
as one graph (`time_step`).  Writes hm96.json in DIR.  `report` gives each point's µs as a share of the same (n, k, M)'s
pearlc arms in glm53_timing's timing.json, and with whole.json each forward's added time, its calls' points summed
(``calls-``) and a step of its shapes (``step-``), as a share of each of its arms' forward.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.gates import frame_v3 as F  # noqa: E402
from verity.ml.commitments.gates import hm96 as H  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix  # noqa: E402
from verity.ml.commitments.window import WINDOW_DOMAIN, window_domain  # noqa: E402
from verity_vllm.commit import step_rows  # noqa: E402

SOURCE = ROOT / "integrations/vllm/verity_vllm/commit/committer/native_rows.cu"
KERNELS = ROOT / "verity/ml/kernels"
#: the served committer's CUDA flags (native_jit.CUDA_CFLAGS) for a shared library of sm_90a
NVCC = ("-O3", "--use_fast_math", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-gencode", "arch=compute_90a,code=sm_90a",
        "-Xptxas", "-v")
LAUNCHERS = ("verity_rows_params_bytes", "verity_rows_build", "verity_rows_leaves_sf_launch", "verity_rows_levels_launch",
             "verity_rows_leaves_form")
#: the most levels one rows_levels_kernel launch folds
FUSED_LEVELS = 10
FORMS = ("split", "fused")
INNER = 8
EPOCH = 20261103
#: the arm's port, its fields fixed: a domain's bytes are constants the kernels take, so its cost reads none of them
PORT = {"program": "pearl-plus/a-rows", "partition": "pearl-plus/a-rows", "port": "a", "run": "pearl-plus/a-rows"}
KS = hashlib.sha512(b"pearl-plus/a-rows/salt-key").digest()
#: (M, K): a decode call's tree with pads, down_proj's K at full width, one leaf, and a tree two level launches fold
VECTORS = ((5, 2048), (8, 8192), (1, 2048), (1100, 32))


def domain(m: int) -> CommitmentDomain:
    """One-stage's window domain of the arm's port of m leaves, held to `window.window_domain`."""
    binding = identity_digest_sha512(WINDOW_DOMAIN, {"program": PORT["program"], "partition": PORT["partition"],
                                                     "port": PORT["port"], "schema": SCHEMA, "leaves": m, "run": PORT["run"]})
    d = CommitmentDomain(binding, -1, RangeIndexedDomain(0, m), hash="sha512")
    if d.domain_id.hex() != window_domain(PORT["program"], PORT["partition"], PORT["port"], SCHEMA, m, PORT["run"]):
        raise AssertionError("the arm's domain is not window_domain's")
    return d


def rows_of(data: bytes, m: int, k: int) -> list[bytes]:
    if len(data) != 2 * m * k:
        raise ValueError(f"{m} rows of {k} BF16 elements are {2 * m * k} bytes, not {len(data)}")
    return [data[2 * k * i:2 * k * (i + 1)] for i in range(m)]


def vector(m: int, k: int) -> bytes:
    return random.Random(f"hm96-rows/{m}x{k}").randbytes(2 * m * k)


def reference(rows: list[bytes], k: int) -> tuple[tuple[bytes, ...], ...]:
    """The tree's levels, leaves first and the root last: each row's x by rowleaf, its h by the hm96 gates' reference, each
    leaf and node by the frame-v3 gates', the pads the domain's."""
    d, n_bits = domain(len(rows)), 16 * k
    level = [F.leaf_ref(d.domain_id, SCHEMA, H.reference(sha512_row_v2_digest(r, n_bits), step_rows.salt(KS, i))[2], i)
             for i, r in enumerate(rows)]
    level += [d.pad(i) for i in range(len(rows), 1 << merkle_depth(len(rows)))]
    out = [tuple(level)]
    while len(level) > 1:
        level = [F.node_ref(d.domain_id, len(out) - 1, j, level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
        out.append(tuple(level))
    return tuple(out)


def level_launches(width: int) -> list[tuple[int, int, int]]:
    """(nodes in, their level, levels folded) of each rows_levels_kernel launch up a tree of `width` leaves (a power of two)."""
    out, n, level = [], width, 0
    while n > 1:
        k = min(FUSED_LEVELS, n.bit_length() - 1)
        out.append((n, level, k))
        n, level = n >> k, level + k
    return out


class Rows:
    """`build`'s library: native_rows.cu's launchers, and the hm96 key's nibble table on the device."""

    def __init__(self, lib: str):
        import torch

        L = self.lib = C.CDLL(lib)
        vp, u64, i32 = C.c_void_p, C.c_uint64, C.c_int
        L.verity_rows_params_bytes.restype = i32
        L.verity_rows_build.argtypes = [C.c_char_p] * 6 + [i32, C.c_uint32, vp]
        L.verity_rows_build.restype = i32
        L.verity_rows_leaves_sf_launch.argtypes = [vp, vp, u64, u64, vp, vp, u64, u64, u64, vp, vp, vp, vp, u64, vp]
        L.verity_rows_leaves_sf_launch.restype = i32
        L.verity_rows_levels_launch.argtypes = [vp, vp, vp, u64, i32, u64, i32, vp]
        L.verity_rows_levels_launch.restype = i32
        L.verity_rows_leaves_form.restype = i32
        self.tn = torch.frombuffer(bytearray(step_rows.t_nibbles()), dtype=torch.uint8).cuda()

    def params(self, m: int, k: int) -> C.Array:
        out = C.create_string_buffer(self.lib.verity_rows_params_bytes())
        schema = SCHEMA.encode()
        rc = self.lib.verity_rows_build(domain(m).domain_id, KS, sha512_row_v2_prefix(16 * k), hm96.SHA512.salt_prefix,
                                        hm96.SHA512.leaf_prefix(step_rows.KEY), schema, len(schema), 2 * k, out)
        if rc:
            raise RuntimeError(f"verity_rows_build: {rc}")
        return out


class Commit:
    """One call's commitment of its M x K BF16 A (a contiguous CUDA tensor): the leaves in the form `VERITY_ROWS_FORM`
    names when it is called, then the tree, on the stream it is given."""

    def __init__(self, rows: Rows, x):
        import torch

        m, k = x.shape
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16 or k % 8:
            raise ValueError("A is a contiguous, 16-byte aligned BF16 tensor of a multiple of 8 columns")
        self.m, self.k, self.width = m, k, 1 << merkle_depth(m)
        self.rows, self.x, self.params = rows, x, rows.params(m, k)
        u8 = dict(dtype=torch.uint8, device=x.device)
        self.leaves = torch.empty(self.width, 64, **u8)
        self.scratch = torch.empty(192 * self.width, **u8)
        self.plan = level_launches(self.width)
        self.bufs = [torch.empty(n - (n >> f), 64, **u8) for n, _, f in self.plan]

    def __call__(self, stream: int) -> None:
        L, m = self.rows.lib, self.m
        rc = L.verity_rows_leaves_sf_launch(self.params, self.x.data_ptr(), 2 * self.k, m, None, self.rows.tn.data_ptr(), 0,
                                            self.width, m, self.leaves.data_ptr(), None, None, self.scratch.data_ptr(), 0, stream)
        cur = self.leaves.data_ptr()
        for (n, level, f), buf in zip(self.plan, self.bufs):
            rc = rc or L.verity_rows_levels_launch(self.params, cur, buf.data_ptr(), n, level, 0, f, stream)
            cur = buf.data_ptr() + 64 * (n - (n >> f) - (n >> f))
        if rc:
            raise RuntimeError(f"native_rows.cu: CUDA error {rc}")

    def root(self) -> bytes:
        return bytes((self.bufs[-1][-1] if self.plan else self.leaves[0]).cpu().numpy())

    def leaf_bytes(self) -> list[bytes]:
        b = self.leaves.cpu().numpy().tobytes()
        return [b[64 * i:64 * i + 64] for i in range(self.width)]


def checked(rows: Rows, x, data: bytes, name: str) -> dict:
    """Every form's leaves and root of A against `reference` over its bytes."""
    import torch

    lv = reference(rows_of(data, *x.shape), x.shape[1])
    c, got = Commit(rows, x), {}
    for form in FORMS:
        os.environ["VERITY_ROWS_FORM"] = form
        for t in (c.leaves, *c.bufs):
            t.fill_(0)
        c(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        got[form] = {"leaves": c.leaf_bytes() == list(lv[0]), "root": c.root() == lv[-1][0]}
    return {"point": name, "m": x.shape[0], "k": x.shape[1], "root": lv[-1][0].hex(), "forms": got,
            "ok": all(v["leaves"] and v["root"] for v in got.values())}


def gate(rows: Rows) -> list[dict]:
    import torch

    out = []
    for m, k in VECTORS:
        data = vector(m, k)
        x = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).cuda().view(m, k)
        out.append(checked(rows, x, data, f"vector {m}x{k}"))
    return out


def graphed(f, form: str):
    """f (called with a stream) under `form` captured as one CUDA graph: (replay, None), or (None, why not)."""
    import torch

    os.environ["VERITY_ROWS_FORM"] = form
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            f(side.cuda_stream)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return g.replay, None
    except Exception as e:  # noqa: BLE001 - recorded, the arm left out
        return None, f"{type(e).__name__}: {e}"[:400]


def sample(f, form: str) -> float:
    """`INNER` calls of f between events on the current stream: µs a call."""
    import torch

    os.environ["VERITY_ROWS_FORM"] = form
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(INNER):
        f()
    e.record()
    e.synchronize()
    return 1e3 * s.elapsed_time(e) / INNER


def time_point(rows: Rows, x, reps: int) -> tuple[dict, dict]:
    """Each form's commitment of A, eager and graphed: (µs samples by arm, errors by arm)."""
    import torch

    c, stream = Commit(rows, x), torch.cuda.current_stream().cuda_stream
    arms, errors = {}, {}
    for form in FORMS:
        arms[f"{form}-eager"] = (lambda: c(stream), form)
        sample(*arms[f"{form}-eager"])
        g, err = graphed(c, form)
        if g:
            arms[f"{form}-graph"] = (g, form)
            sample(g, form)
        else:
            errors[f"{form}-graph"] = err
    return interleaved(arms, reps), errors


def interleaved(arms: dict, reps: int) -> dict:
    """Each arm's samples, every arm once a rep in an order rotated each rep."""
    names, runs = list(arms), {nm: [] for nm in arms}
    for rep in range(reps):
        for nm in names[rep % len(names):] + names[:rep % len(names)]:
            runs[nm].append(sample(*arms[nm]))
    return runs


def time_step(rows: Rows, model, m: int, layers: int, reps: int, gen) -> dict:
    """A decode step's commitments, one a call of each linear of each of `layers` layers on its own A of m rows, captured
    as one CUDA graph under each form: serially on one stream, and concurrently, the calls dealt over torch's stream pool
    (``streams`` distinct), each forking from the capture's stream and joining it at the end.  Every call's root is held to
    `reference` after each graph's replay."""
    import torch

    lins = [lin for _ in range(layers) for lin in model.linears()]
    xs = [torch.randn(m, lin.k, device="cuda", generator=gen).to(torch.bfloat16) for lin in lins]
    cs, streams = [Commit(rows, x) for x in xs], [torch.cuda.Stream() for _ in xs]
    roots = [reference(rows_of(x.view(torch.uint8).cpu().numpy().tobytes(), m, x.shape[1]), x.shape[1])[-1][0] for x in xs]

    def serial(stream: int) -> None:
        for c in cs:
            c(stream)

    def concurrent(_stream: int) -> None:
        cur = torch.cuda.current_stream()
        for c, s in zip(cs, streams):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                c(s.cuda_stream)
        for s in streams:
            cur.wait_stream(s)

    arms, errors, ok = {}, {}, {}
    for form in FORMS:
        for nm, f in (("serial", serial), ("concurrent", concurrent)):
            g, err = graphed(f, form)
            if g is None:
                errors[f"{form}-{nm}"] = err
                continue
            for c in cs:
                for t in (c.leaves, *c.bufs):
                    t.fill_(0)
            g()
            torch.cuda.synchronize()
            ok[f"{form}-{nm}"] = [c.root() for c in cs] == roots
            arms[f"{form}-{nm}"] = (g, form)
    runs = interleaved(arms, reps) if all(ok.values()) else {}
    return {"m": m, "layers": layers, "calls": len(cs), "shapes": [[lin.n, lin.k] for lin in lins],
            "streams": len({s.cuda_stream for s in streams}), "roots_ok": ok,
            "us": {nm: statistics.median(v) for nm, v in runs.items()}, "runs": runs, "errors": errors}


def time_main(a) -> int:
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, model = Rows(a.lib), importlib.import_module(a.model)
    head = {"model": a.model, "device": torch.cuda.get_device_name(), "torch": torch.__version__, "reps": a.reps,
            "inner": INNER, "port": PORT, "schema": SCHEMA, "forms": FORMS, "device_form": rows.lib.verity_rows_leaves_form(),
            "phases": os.environ.get("VERITY_ROWS_PHASES", ""),
            "lib_sha256": hashlib.sha256(Path(a.lib).read_bytes()).hexdigest()}
    checks = gate(rows)
    doc = {**head, "checks": checks, "rows": []}
    (out / "hm96.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("gate: " + ", ".join(f"{c['point']} {'ok' if c['ok'] else 'DIFFERS ' + json.dumps(c['forms'])}" for c in checks),
          flush=True)
    if not all(c["ok"] for c in checks):
        return 1
    gen = torch.Generator(device="cuda").manual_seed(EPOCH)
    for lin in model.linears():
        for m in a.m:
            x = torch.randn(m, lin.k, device="cuda", generator=gen).to(torch.bfloat16)
            chk = checked(rows, x, x.view(torch.uint8).cpu().numpy().tobytes(), f"{lin.name} m {m}")
            doc["checks"].append(chk)
            if not chk["ok"]:
                (out / "hm96.json").write_text(json.dumps(doc, indent=1) + "\n")
                print(f"{chk['point']}: DIFFERS {json.dumps(chk['forms'])}", flush=True)
                return 1
            runs, errors = time_point(rows, x, a.reps)
            us = {nm: statistics.median(v) for nm, v in runs.items()}
            doc["rows"].append({"name": lin.name, "n": lin.n, "k": lin.k, "m": m, "us": us, "runs": runs, "errors": errors})
            print(f"{lin.name} n {lin.n} k {lin.k} m {m}: " + ", ".join(f"{nm} {v:.1f} us" for nm, v in us.items())
                  + "".join(f"; {nm}: {e}" for nm, e in errors.items()), flush=True)
            (out / "hm96.json").write_text(json.dumps(doc, indent=1) + "\n")
    doc["steps"] = []
    for m in a.step_m:
        st = time_step(rows, model, m, a.layers, a.reps, gen)
        doc["steps"].append(st)
        print(f"step m {m}, {st['calls']} calls ({a.layers} layers): "
              + ", ".join(f"{nm} {v:.1f} us" for nm, v in st["us"].items())
              + f"; roots {'equal' if all(st['roots_ok'].values()) else 'DIFFER ' + json.dumps(st['roots_ok'])}"
              + "".join(f"; {nm}: {e}" for nm, e in st["errors"].items()), flush=True)
        (out / "hm96.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if all(all(st["roots_ok"].values()) for st in doc["steps"]) else 1


def report(hm: dict, timing: dict, whole: dict | None) -> dict:
    """Each point's µs over the same (n, k, M)'s pearlc arms, and the decode forward's added time over each of its arms."""
    pearl = {(r["n"], r["k"], r["m"]): r["ms"] for r in timing["rows"]}
    points = []
    for r in hm["rows"]:
        ms = {nm: v for nm, v in pearl.get((r["n"], r["k"], r["m"]), {}).items() if nm.startswith("pearlc")}
        points.append({**{f: r[f] for f in ("name", "n", "k", "m", "us")}, "pearlc_ms": ms,
                       "share": {arm: {nm: us / 1e3 / v for nm, v in ms.items()} for arm, us in r["us"].items()}})
    forwards = []
    for w in (whole or {}).get("rows", []):
        us_at = {(r["n"], r["k"]): r["us"] for r in hm["rows"] if r["m"] == w["m"]}
        shapes = [tuple(s) for s in w["shapes"]]
        if not shapes or any(s not in us_at for s in shapes):
            continue
        added = {f"calls-{arm}": sum(us_at[s][arm] for s in shapes) / 1e3 for arm in us_at[shapes[0]]
                 if all(arm in us_at[s] for s in shapes)}
        for st in hm.get("steps", []):
            if st["m"] == w["m"] and sorted(map(tuple, st["shapes"])) == sorted(shapes):
                added.update({f"step-{arm}": v / 1e3 for arm, v in st["us"].items()})
        forwards.append({"m": w["m"], "calls": w["calls"], "forward_ms": w["ms"], "added_ms": added,
                         "share": {arm: {nm: v / f for nm, f in w["ms"].items()} for arm, v in added.items()}})
    return {"points": points, "forwards": forwards}


def report_main(a) -> int:
    hm, timing = json.loads(Path(a.hm96).read_text()), json.loads(Path(a.timing).read_text())
    whole = json.loads(Path(a.whole).read_text()) if a.whole else None
    r = report(hm, timing, whole)
    for p in r["points"]:
        best = min(p["us"], key=p["us"].get)
        print(f"{p['name']} k {p['k']} m {p['m']}: {best} {p['us'][best]:.1f} us, "
              + ", ".join(f"{v:.1%} of {nm}" for nm, v in p["share"][best].items()))
    for f in r["forwards"]:
        best = min(f["added_ms"], key=f["added_ms"].get)
        print(f"forward m {f['m']} ({f['calls']} calls): {best} adds {f['added_ms'][best]:.3f} ms, "
              + ", ".join(f"{v:.1%} of {nm}" for nm, v in f["share"][best].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    return 0


def build(out: Path, cuda: Path) -> int:
    """nvcc of native_rows.cu alone into out/hm96_rows.so; exits non-zero unless it exports every launcher of `LAUNCHERS`."""
    out.mkdir(parents=True, exist_ok=True)
    so, nvcc = out / "hm96_rows.so", cuda / "bin" / "nvcc"
    cmd = [str(nvcc), *NVCC, "-I", str(KERNELS), str(SOURCE), "-o", str(so)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "ptxas.txt").write_text(p.stdout + p.stderr)
    if p.returncode:
        print(p.stderr[-4000:], file=sys.stderr)
        return p.returncode
    syms = subprocess.run(["nm", "-D", "--defined-only", str(so)], capture_output=True, text=True, check=True).stdout.split()
    missing = [s for s in LAUNCHERS if s not in syms]
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in (SOURCE, KERNELS / "row_commit.cuh", KERNELS / "sha512.cuh")}
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout
    (out / "build.json").write_text(json.dumps({"nvcc": version, "command": ["nvcc", *NVCC, "-I", "verity/ml/kernels",
                                                                             str(SOURCE.relative_to(ROOT)), "-o", so.name],
                                                "sources": sources, "lib_sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
                                                "missing": missing}, indent=1) + "\n")
    print(f"{so}: {'missing ' + ', '.join(missing) if missing else 'every launcher exported'}")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--cuda", default=os.environ.get("CUDA", "/usr/local/cuda-13.0"))
    t = sub.add_parser("time")
    t.add_argument("--lib", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="llama32_1b", help="whose linears (this directory's module)")
    t.add_argument("--m", nargs="+", type=int, required=True)
    t.add_argument("--step-m", nargs="*", type=int, default=[], help="M of each decode step timed whole (`time_step`)")
    t.add_argument("--layers", type=int, default=1, help="--step-m: the step's layers, each its own calls of every linear")
    t.add_argument("--reps", type=int, default=15)
    r = sub.add_parser("report")
    r.add_argument("hm96")
    r.add_argument("--timing", required=True)
    r.add_argument("--whole")
    r.add_argument("--out")
    a = p.parse_args(argv)
    if a.mode == "build":
        return build(Path(a.out), Path(a.cuda))
    return time_main(a) if a.mode == "time" else report_main(a)


if __name__ == "__main__":
    sys.exit(main())
