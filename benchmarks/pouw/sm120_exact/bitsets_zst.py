"""Compress, preserve, then delete the v2-hot full-unit bitsets under node 2's `/workspace/pouw/gpu3-fp8/out` (compute
accounting, 11:50 AM PDT 1 Oct 2026, on Daniel's yes at 11:48 AM PDT): nothing is lost, and a unit's raw directory goes only
after its compressed copy is verified on the store's remote.

Each unit is a `hot.py --mode v2hot` directory: `{hot,zero}.{rows,cols}.u32` (T x m x n/32 little-endian uint32 each; bit
(t, i, j) of the rows layout is `(w[t, i, j >> 5] >> (j & 31)) & 1`, and the cols layout holds the same bit at
`w'[t, j, i >> 5]`, bit `i & 31`), plus `unit.json`, `H.npy`, `okA.npy`, `okB.npy`. Per unit, one at a time:

1. Per chain, the cols file is rebuilt from the rows file plane by plane (`cols_of_rows`) and compared with the raw cols file on
   the full arrays; the rows file is streamed once into `zstd -LEVEL`, and both raw files are hashed (sha256) on the same read.
   A unit whose cols file is not exactly the transpose keeps its cols as well (`*.cols.u32.zst` in its tree) and is not deleted.
2. Round trip: each `.zst` is decompressed and its sha256 compared with the raw file's.
3. The unit's tree (the small files, the `.zst` files, `README.txt`, `rebuild_cols.py`, `checks.json`) goes into the evidence
   store with `research data put --kind evidence/v1 --tree ... --preserve`, and `research data preserved` must pass.
4. Only then the unit (and, for the unlink, its parent) is made writable, the raw directory is removed under
   `ionice -c3 nice -n19`, the parent's mode is restored, and `df` is recorded before and after.
5. Every step appends to `--manifest` (JSON lines: unit, art id, sha256s, bytes, df), which is restartable: a unit with a
   `preserved` line whose directory still exists is re-verified and deleted; one with a `deleted` line is done.

Timed windows: the job holds from 30 min before every line of `--windows` (`<UTC start> <minutes>`) until the window ends, and
re-reads the file every 20 s. It starts no unit that would reach a hold, stops feeding work at a hold, and SIGSTOPs its child
processes (zstd, the store upload, rm) until the hold ends. An unreadable file or line holds too.

Run on node 2 through `research run --queue`, CPU only (see the launch in the lane's handoff); `--dry-run` does steps 1-2 alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CHAINS = ("hot", "zero")
SMALL = ("unit.json", "H.npy", "okA.npy", "okB.npy")
LEAD_S = 30 * 60
BATCH = 32
CHUNK = 64 << 20
LANE = "pouw-fp8-security"
REBUILD = '''"""Rebuild <chain>.cols.u32 from <chain>.rows.u32.zst in this directory and check it against checks.json's sha256.
python3 rebuild_cols.py hot|zero [--out DIR]   (needs numpy and the zstd CLI)"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np


def cols_of_rows(p):
    b = np.unpackbits(p.view(np.uint8), axis=1, bitorder="little")
    out = np.empty((b.shape[1], b.shape[0]), np.uint8)
    for i0 in range(0, b.shape[0], 1024):
        out[:, i0:i0 + 1024] = b[i0:i0 + 1024].T
    return np.packbits(out, axis=1, bitorder="little").view(np.uint32)


here = Path(__file__).resolve().parent
chain = sys.argv[1]
dest = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else here
u, ck = json.loads((here / "unit.json").read_text()), json.loads((here / "checks.json").read_text())
m, n = u["m"], u["n"]
plane = m * n // 8
z = subprocess.Popen(["zstd", "-dc", str(here / f"{chain}.rows.u32.zst")], stdout=subprocess.PIPE)
h = hashlib.sha256()
with open(dest / f"{chain}.cols.u32", "wb") as f:
    while True:
        buf = z.stdout.read(plane)
        if not buf:
            break
        c = cols_of_rows(np.frombuffer(buf, np.uint32).reshape(m, n // 32)).tobytes()
        h.update(c)
        f.write(c)
assert z.wait() == 0
want = ck["chains"][chain]["cols"]["sha256"]
print(chain, "cols sha256", h.hexdigest(), "OK" if h.hexdigest() == want else f"MISMATCH (want {want})")
sys.exit(h.hexdigest() != want)
'''


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{utc()} {msg}", flush=True)


def cols_of_rows(p: np.ndarray) -> np.ndarray:
    """One plane's cols layout (n x m/32 uint32) from its rows layout (m x n/32 uint32)."""
    b = np.unpackbits(p.view(np.uint8), axis=1, bitorder="little")
    out = np.empty((b.shape[1], b.shape[0]), np.uint8)
    for i0 in range(0, b.shape[0], 1024):
        out[:, i0:i0 + 1024] = b[i0:i0 + 1024].T
    return np.packbits(out, axis=1, bitorder="little").view(np.uint32)


def check_plane(task: tuple[str, str, int, int, int]) -> tuple[int, int]:
    """(plane, mismatched cols words) for plane t, read from the files with pread (the main loop's read keeps them cached)."""
    rows_path, cols_path, t, m, n = task
    size = m * n // 8
    with open(rows_path, "rb") as fr, open(cols_path, "rb") as fc:
        r = os.pread(fr.fileno(), size, t * size)
        c = os.pread(fc.fileno(), size, t * size)
    if len(r) != size or len(c) != size:
        return t, -1
    got = cols_of_rows(np.frombuffer(r, np.uint32).reshape(m, n // 32))
    return t, int((got != np.frombuffer(c, np.uint32).reshape(n, m // 32)).sum())


class Gate:
    """The timed-window hold: `held()` from 30 min before each window line to its end, failing closed on a bad file."""

    def __init__(self, path: Path):
        self.path, self.children, self.lock = path, set(), threading.Lock()
        self.stopped, self.state = False, (True, "not read yet")
        self.refresh()
        threading.Thread(target=self._watch, daemon=True).start()

    def windows(self) -> list[tuple[float, float, str]]:
        out = []
        for line in self.path.read_text().splitlines():
            body = line.split("#", 1)[0].strip()
            if not body:
                continue
            start, minutes = body.split()[:2]
            s = dt.datetime.strptime(start, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc).timestamp()
            out.append((s - LEAD_S, s + float(minutes) * 60, line.strip()))
        return out

    def refresh(self) -> tuple[bool, str]:
        try:
            now = time.time()
            ws = self.windows()
            hit = [w for w in ws if w[0] <= now < w[1]]
            self.state = (True, f"window {hit[0][2]!r}") if hit else (False, "")
            self.upcoming = sorted(w for w in ws if w[1] > now)
        except Exception as e:  # noqa: BLE001 - any unreadable file or line holds
            self.state, self.upcoming = (True, f"windows file unreadable: {e}"), []
        return self.state

    def _signal_children(self, sig: int) -> None:
        with self.lock:
            for p in list(self.children):
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        pass

    def _watch(self) -> None:
        while True:
            held, why = self.refresh()
            if held and not self.stopped:
                self.stopped = True
                log(f"HOLD ({why}): children stopped")
                self._signal_children(signal.SIGSTOP)
            elif not held and self.stopped:
                self.stopped = False
                log("RESUME: children continued")
                self._signal_children(signal.SIGCONT)
            elif held:
                self._signal_children(signal.SIGSTOP)
            time.sleep(20)

    def popen(self, argv: list[str], **kw) -> subprocess.Popen:
        self.wait()
        p = subprocess.Popen(argv, start_new_session=True, **kw)
        with self.lock:
            self.children.add(p)
        return p

    def done(self, p: subprocess.Popen) -> None:
        with self.lock:
            self.children.discard(p)

    def wait(self) -> None:
        said = False
        while self.refresh()[0]:
            if not said:
                log(f"holding: {self.state[1]}")
                said = True
            time.sleep(20)
        if said:
            log("hold over")

    def clear_for(self, seconds: float) -> None:
        """Wait until no hold begins within `seconds` from now."""
        while True:
            self.wait()
            now = time.time()
            nxt = [w for w in self.upcoming if w[0] < now + seconds]
            if not nxt:
                return
            log(f"not starting: hold from {utc(nxt[0][0])} ({nxt[0][2]!r}) is within {seconds / 60:.0f} min; waiting for it")
            while time.time() < nxt[0][1] and not self.refresh()[0]:
                time.sleep(20)


def run(gate: Gate, argv: list[str], **kw) -> subprocess.CompletedProcess:
    p = gate.popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
    try:
        out, err = p.communicate()
    finally:
        gate.done(p)
    return subprocess.CompletedProcess(argv, p.returncode, out, err)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def disk(path: str = "/workspace") -> dict:
    s = os.statvfs(path)
    total, free = s.f_blocks * s.f_frsize, s.f_bavail * s.f_frsize
    used = total - s.f_bfree * s.f_frsize
    return {"utc": utc(), "used_bytes": used, "avail_bytes": free, "total_bytes": total,
            "pct": round(100 * used / (used + free), 2)}


def chain_pass(gate: Gate, pool: Pool, ud: Path, tree: Path, chain: str, m: int, n: int, T: int, level: int, threads: int,
               with_cols: bool) -> dict:
    """One read of rows and cols: sha256 of both, rows (and with_cols, cols) streamed into zstd, every plane's transpose checked."""
    rp, cp = ud / f"{chain}.rows.u32", ud / f"{chain}.cols.u32"
    size = m * n // 8
    for p in (rp, cp):
        if p.stat().st_size != T * size:
            raise RuntimeError(f"{p}: {p.stat().st_size} bytes, want {T * size}")
    zs = {lay: tree / f"{chain}.{lay}.u32.zst" for lay in (("rows", "cols") if with_cols else ("rows",))}
    zp = {}
    for lay, z in zs.items():
        with open(z, "wb") as zf:
            zp[lay] = gate.popen(["zstd", f"-{level}", f"-T{threads}", "-q", "-c", f"--stream-size={T * size}"],
                                 stdin=subprocess.PIPE, stdout=zf)
    hr, hc = hashlib.sha256(), hashlib.sha256()
    bad, pending = {}, None
    try:
        with open(rp, "rb") as fr, open(cp, "rb") as fc:
            for p0 in range(0, T, BATCH):
                gate.wait()
                nt = min(BATCH, T - p0)
                r, c = fr.read(nt * size), fc.read(nt * size)
                hr.update(r)
                hc.update(c)
                zp["rows"].stdin.write(r)
                if with_cols:
                    zp["cols"].stdin.write(c)
                del r, c
                if pending is not None:
                    bad.update(dict(x for x in pending.get() if x[1]))
                pending = pool.map_async(check_plane, [(str(rp), str(cp), t, m, n) for t in range(p0, p0 + nt)])
            if pending is not None:
                bad.update(dict(x for x in pending.get() if x[1]))
    finally:
        for p in zp.values():
            p.stdin.close()
            p.wait()
            gate.done(p)
    if any(p.returncode for p in zp.values()):
        raise RuntimeError(f"{ud.name}/{chain}: zstd exit {[p.returncode for p in zp.values()]}")
    raw = {"rows": {"bytes": T * size, "sha256": hr.hexdigest()}, "cols": {"bytes": T * size, "sha256": hc.hexdigest()}}
    out = {"rows": raw["rows"], "cols": raw["cols"], "planes": T, "transpose_mismatched_planes": len(bad),
           "transpose_mismatched_words": int(sum(v for v in bad.values() if v > 0)), "short_reads": sum(v < 0 for v in bad.values()),
           "zst": {}}
    for lay, z in zs.items():
        h = hashlib.sha256()
        p = gate.popen(["zstd", "-dc", "-q", str(z)], stdout=subprocess.PIPE)
        try:
            while b := p.stdout.read(CHUNK):
                gate.wait()
                h.update(b)
        finally:
            p.wait()
            gate.done(p)
        out["zst"][lay] = {"file": z.name, "bytes": z.stat().st_size, "sha256": sha_file(z), "decompressed_sha256": h.hexdigest(),
                           "round_trip_ok": p.returncode == 0 and h.hexdigest() == raw[lay]["sha256"]}
    return out


def readme(name: str, u: dict, ck: dict, level: int, zver: str, with_cols: bool) -> str:
    lines = [f"{name}: v2-hot full-unit bitsets (benchmarks/pouw/sm120_exact/hot.py --mode {u.get('mode')}, salt {u.get('salt')!r}),",
             f"compressed on node 2 by benchmarks/pouw/sm120_exact/bitsets_zst.py with `zstd -{level}` ({zver}); the raw directory was",
             f"/workspace/pouw/gpu3-fp8/out/{ck['unit']}.",
             f"<chain>.rows.u32.zst is the raw <chain>.rows.u32: {u['planes']} planes x {u['m']} rows x {u['n'] // 32} little-endian uint32;",
             "bit (t, i, j) = (w[t, i, j >> 5] >> (j & 31)) & 1, i a row of A and j a row of B.",
             "<chain>.cols.u32 holds the same bits column-major: w'[t, j, i >> 5] bit (i & 31) = bit (t, i, j)."]
    if with_cols:
        lines.append("Its cols files did NOT match the transpose of its rows files, so <chain>.cols.u32.zst are stored too (checks.json).")
    else:
        lines.append("It is not stored: `python3 rebuild_cols.py hot|zero` (numpy, zstd CLI) rebuilds it here and checks its sha256.")
    lines.append("sha256 of the raw files (checks.json has bytes and the .zst hashes):")
    for c, v in ck["chains"].items():
        lines += [f"  {c}.rows.u32 {v['rows']['sha256']}", f"  {c}.cols.u32 {v['cols']['sha256']}"]
    return "\n".join(lines) + "\n"


class Manifest:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, dict] = {}
        if paths[0].exists():
            for line in paths[0].read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.state.setdefault(rec["unit"], {}).update(rec)

    def add(self, rec: dict) -> None:
        rec = {"utc": utc(), **rec}
        line = json.dumps(rec, sort_keys=True) + "\n"
        for p in self.paths:
            with open(p, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        self.state.setdefault(rec["unit"], {}).update(rec)


class Store:
    """`research data` with the job's own key (cred.json + store.toml in --cred-dir) and a private local store per unit."""

    def __init__(self, src: Path, cred_dir: Path):
        cred = json.loads((cred_dir / "cred.json").read_text())
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "R2_", "RESEARCH_STORE"))}
        self.env.update({k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")},
                        RESEARCH_STORE_CONFIG=str(cred_dir / "store.toml"), PYTHONPATH=str(src), RESEARCH_NOTES_SYNC="0")
        self.expires = cred.get("expires_utc")

    def __call__(self, gate: Gate, store: Path, *args: str) -> subprocess.CompletedProcess:
        return run(gate, [sys.executable, "-m", "research", "data", *args], env={**self.env, "RESEARCH_STORE": str(store)})

    def preserved(self, gate: Gate, store: Path, art: str) -> tuple[bool, str]:
        r = self(gate, store, "preserved", art)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]


def writable(p: Path) -> int:
    mode = stat.S_IMODE(p.lstat().st_mode)
    os.chmod(p, mode | stat.S_IWUSR)
    return mode


def delete_unit(gate: Gate, root: Path, ud: Path) -> dict:
    if ud.is_symlink() or not ud.is_dir() or not ud.resolve().is_relative_to(root.resolve()) or ud.resolve() == root.resolve():
        raise RuntimeError(f"refusing to delete {ud}: not a unit directory under {root}")
    before = disk()
    raw = sum(f.stat().st_size for f in ud.iterdir() if f.is_file())
    parent_mode = writable(ud.parent)
    try:
        writable(ud)
        r = run(gate, ["ionice", "-c3", "nice", "-n19", "rm", "-rf", "--one-file-system", "--", str(ud)])
    finally:
        os.chmod(ud.parent, parent_mode)
    if r.returncode or ud.exists():
        raise RuntimeError(f"rm {ud}: exit {r.returncode}: {r.stderr.strip()[-300:]}")
    return {"df_before": before, "df_after": disk(), "raw_bytes_deleted": raw}


def units(root: Path, sets: list[str], keep: set[str]) -> tuple[list[str], list[dict]]:
    todo, links = [], []
    for s in sets:
        for d in sorted((root / s / "units").iterdir()):
            rel = str(d.relative_to(root))
            if d.is_symlink():
                links.append({"unit": rel, "symlink_to": os.readlink(d)})
            elif rel not in keep and d.is_dir():
                todo.append(rel)
    return todo, links


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/workspace/pouw/gpu3-fp8/out")
    ap.add_argument("--sets", default="v2hot,fix2,v2hot-cancel")
    ap.add_argument("--keep", default=",".join(f"fix2/units/cancel-pair@flat@{u}" for u in range(1, 5)))
    ap.add_argument("--windows", default="/workspace/pouw/fill/windows")
    ap.add_argument("--manifest", default=None, help="default <root>/bitsets-zst/manifest.jsonl")
    ap.add_argument("--out", default=None, help="default $RESEARCH_RUN_DIR/out")
    ap.add_argument("--cred-dir", default=str(Path.home() / ".fp8-bitsets"))
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16, help="zstd -T")
    ap.add_argument("--workers", type=int, default=16, help="transpose-check processes")
    ap.add_argument("--budget-min", type=float, default=330, help="start no unit after this many minutes")
    ap.add_argument("--limit", type=int, default=0, help="at most this many units (0: all)")
    ap.add_argument("--only", default="", help="comma-separated units (relative to --root) to do, in this order")
    ap.add_argument("--dry-run", action="store_true", help="checks and compression only: no store, no delete")
    a = ap.parse_args()
    if not 3 <= a.level <= 9:
        ap.error("--level is 3..9 (compute accounting's range)")
    t0 = time.time()
    root = Path(a.root)
    out = Path(a.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out.mkdir(parents=True, exist_ok=True)
    keep = {k for k in a.keep.split(",") if k}
    todo, links = units(root, a.sets.split(","), keep)
    if a.only:
        todo = [u for u in a.only.split(",") if u in todo]
    mpath = Path(a.manifest or root / "bitsets-zst" / "manifest.jsonl")
    man = Manifest([mpath, out / "manifest.jsonl"] if not a.dry_run else [out / "manifest.dry-run.jsonl"])
    gate = Gate(Path(a.windows))
    src = Path(__file__).resolve().parents[3] / "tools" / "research" / "src"
    store = None if a.dry_run else Store(src, Path(a.cred_dir))
    zver = subprocess.run(["zstd", "--version"], capture_output=True, text=True).stdout.strip()
    log(f"start: {len(todo)} units to do, {len(keep)} kept raw, {len(links)} symlinks left as they are; level {a.level}; "
        f"{zver}; manifest {mpath}; key expires {store.expires if store else '-'}; df {disk()['pct']}%")
    summary = {"done": [], "failed": [], "skipped": [], "level": a.level, "zstd": zver, "kept_raw": sorted(keep), "symlinks": links}
    est = 15 * 60.0
    with Pool(a.workers) as pool:
        for rel in todo:
            if a.limit and len(summary["done"]) + len(summary["failed"]) >= a.limit:
                break
            if (time.time() - t0) / 60 > a.budget_min:
                log(f"budget spent ({a.budget_min} min): stopping before {rel}")
                break
            st = man.state.get(rel, {})
            if st.get("status") == "deleted":
                summary["skipped"].append(rel)
                continue
            ud = root / rel
            gate.clear_for(est)
            t1 = time.time()
            scratch = Path(tempfile.mkdtemp(prefix="bitsets-zst-"))
            try:
                if st.get("status") == "preserved" and not ud.exists() and not a.dry_run:
                    man.add({"unit": rel, "status": "deleted", "art": st["art"], "df_after": disk(),
                             "note": "the directory was already gone at restart (a delete interrupted after rm)"})
                    summary["done"].append(rel)
                    continue
                if st.get("status") == "preserved" and ud.exists() and not a.dry_run:
                    ok, why = store.preserved(gate, scratch / "store", st["art"])
                    if not ok:
                        raise RuntimeError(f"{rel}: its manifest art {st['art']} is not preserved now: {why}")
                    rec = delete_unit(gate, root, ud)
                    man.add({"unit": rel, "status": "deleted", "art": st["art"], **rec})
                    summary["done"].append(rel)
                    log(f"{rel}: resumed delete of preserved {st['art']}; df {rec['df_after']['pct']}%")
                    continue
                files = sorted(f.name for f in ud.iterdir())
                want = {*SMALL, *(f"{c}.{lay}.u32" for c in CHAINS for lay in ("rows", "cols"))}
                if set(files) != want:
                    raise RuntimeError(f"{rel}: files {files} are not the expected {sorted(want)}")
                u = json.loads((ud / "unit.json").read_text())
                m, n, T = int(u["m"]), int(u["n"]), int(u["planes"])
                tree = scratch / "tree"
                tree.mkdir()
                small = {}
                for f in SMALL:
                    shutil.copy2(ud / f, tree / f)
                    small[f] = {"bytes": (ud / f).stat().st_size, "sha256": sha_file(ud / f)}
                    if sha_file(tree / f) != small[f]["sha256"]:
                        raise RuntimeError(f"{rel}/{f}: copy differs")
                ck = {"unit": rel, "level": a.level, "zstd": zver, "small": small, "chains": {}}
                with_cols = False
                for c in CHAINS:
                    ck["chains"][c] = chain_pass(gate, pool, ud, tree, c, m, n, T, a.level, a.threads, False)
                    if ck["chains"][c]["transpose_mismatched_planes"]:
                        with_cols = True
                if with_cols:
                    log(f"{rel}: cols are not the transpose of rows; storing cols too, and keeping the raw directory")
                    for c in CHAINS:
                        ck["chains"][c] = chain_pass(gate, pool, ud, tree, c, m, n, T, a.level, a.threads, True)
                ck["transpose_ok"] = not with_cols
                ck["round_trip_ok"] = all(z["round_trip_ok"] for v in ck["chains"].values() for z in v["zst"].values())
                (tree / "checks.json").write_text(json.dumps(ck, indent=1, sort_keys=True) + "\n")
                (tree / "README.txt").write_text(readme(Path(rel).name, u, ck, a.level, zver, with_cols))
                (tree / "rebuild_cols.py").write_text(REBUILD)
                tree_bytes = sum(f.stat().st_size for f in tree.iterdir())
                zst_bytes = sum(z["bytes"] for v in ck["chains"].values() for z in v["zst"].values())
                raw_bytes = sum((ud / f).stat().st_size for f in files)
                rec = {"unit": rel, "tree_bytes": tree_bytes, "zst_bytes": zst_bytes, "raw_bytes": raw_bytes,
                       "transpose_ok": ck["transpose_ok"], "round_trip_ok": ck["round_trip_ok"], "small": small,
                       "chains": {c: {"rows": v["rows"], "cols": v["cols"], "zst": v["zst"],
                                      "transpose_mismatched_planes": v["transpose_mismatched_planes"]} for c, v in ck["chains"].items()}}
                if not ck["round_trip_ok"]:
                    raise RuntimeError(f"{rel}: round trip failed: {json.dumps(ck['chains'])[:400]}")
                if a.dry_run:
                    man.add({**rec, "status": "dry-run"})
                    (summary["done"] if ck["transpose_ok"] else summary["failed"]).append(rel)
                    log(f"{rel}: dry run ok: zst {zst_bytes / 2**20:.1f} MiB of {raw_bytes / 2**30:.2f} GiB; {time.time() - t1:.0f} s")
                    continue
                meta = {"lane": LANE, "what": f"v2-hot full-unit bitsets of {rel}: rows layout zstd -{a.level}, cols rebuildable "
                        "(README.txt, rebuild_cols.py); compress-preserve-delete of node 2's gpu3-fp8/out (compute accounting, 11:50 AM PDT)",
                        "unit": rel, "node": "vy-nebius-2", "source_dir": str(ud), "zstd_level": a.level, "transpose_ok": ck["transpose_ok"],
                        "raw_bytes": raw_bytes, "zst_bytes": zst_bytes, "run_id": os.environ.get("RESEARCH_RUN_ID")}
                r = store(gate, scratch / "store", "put", "--kind", "evidence/v1", "--meta", json.dumps(meta, sort_keys=True),
                          "--tree", str(tree), "--preserve")
                arts = re.findall(r"art:[0-9a-f]{64}", r.stdout)
                if r.returncode or not arts:
                    raise RuntimeError(f"{rel}: research data put exit {r.returncode}: {(r.stdout + r.stderr).strip()[-600:]}")
                art = arts[-1]
                ok, why = store.preserved(gate, scratch / "store", art)
                if not ok:
                    raise RuntimeError(f"{rel}: {art} not preserved: {why}")
                man.add({**rec, "status": "preserved", "art": art})
                log(f"{rel}: {art} preserved: zst {zst_bytes / 2**20:.1f} MiB of {raw_bytes / 2**30:.2f} GiB")
                if not ck["transpose_ok"]:
                    summary["failed"].append({"unit": rel, "why": "cols not the transpose of rows; raw kept", "art": art})
                    continue
                d = delete_unit(gate, root, ud)
                man.add({"unit": rel, "status": "deleted", "art": art, **d})
                summary["done"].append(rel)
                log(f"{rel}: deleted {d['raw_bytes_deleted'] / 2**30:.2f} GiB; df {d['df_before']['pct']}% -> {d['df_after']['pct']}%; "
                    f"{time.time() - t1:.0f} s")
            except Exception as e:  # noqa: BLE001 - a unit that fails any check keeps its raw directory
                summary["failed"].append({"unit": rel, "why": str(e)[:800]})
                man.add({"unit": rel, "status": "failed", "why": str(e)[:800]})
                log(f"{rel}: FAILED, raw kept: {e}")
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            est = max(est * 0.5, (time.time() - t1) * 1.5, 5 * 60.0)
    remaining = [u for u in todo if man.state.get(u, {}).get("status") != ("dry-run" if a.dry_run else "deleted")]
    summary.update(remaining=remaining, df=disk(), minutes=round((time.time() - t0) / 60, 1))
    if store is not None and mpath.exists():
        scratch = Path(tempfile.mkdtemp(prefix="bitsets-zst-"))
        try:
            meta = {"lane": LANE, "what": "manifest of node 2's gpu3-fp8/out bitsets compress-preserve-delete: unit -> art id -> "
                    "sha256s -> bytes -> df (JSON lines, cumulative over runs)", "run_id": os.environ.get("RESEARCH_RUN_ID"),
                    "units_deleted": sum(1 for v in man.state.values() if v.get("status") == "deleted")}
            r = store(gate, scratch / "store", "put", "--kind", "evidence/v1", "--meta", json.dumps(meta, sort_keys=True),
                      "--file", str(mpath), "--preserve")
            arts = re.findall(r"art:[0-9a-f]{64}", r.stdout)
            ok, why = store.preserved(gate, scratch / "store", arts[-1]) if arts and not r.returncode else (False, r.stderr[-400:])
            summary["manifest"] = {"art": arts[-1] if arts else None, "preserved": ok, "sha256": sha_file(mpath),
                                   "why": None if ok else why}
            log(f"manifest {summary['manifest']['art']} preserved={ok}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    log(f"end: {len(summary['done'])} done, {len(summary['failed'])} failed, {len(remaining)} remaining; df {summary['df']['pct']}%")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
