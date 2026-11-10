#!/usr/bin/env python3
"""`pack_profile`: where the bytes of recorded checks' verdict packs are.

    python tools/verity/check/pack_profile.py --scratch DIR --out DIR [--history-since rYYYYMMDD] RUN [RUN ...]

Each RUN's `verdicts.tar.gz` is fetched as `check.attempt_pack` fetches it, into a store of its own under --scratch, read as a
stream (no member is extracted to disk) and deleted.  Written under --out, per pack (`pack-RUN.json`) and together
(`summary.md`): entries by path family (the first two and three path components, and for an entry with a `key.request.target`
its kind), with counts, uncompressed bytes and the ten largest; each family's bytes by top-level JSON key and, over a sample, by
schema path (a dict whose keys aren't a few identifiers is one bucket, `{*}`); how much of the `reads` maps repeats between
entries; and what the pack would weigh gzipped without a family or without one top-level key, and whole as xz with a dictionary
spanning entries (`verdicts.XZ_FILTERS`).  Only paths, key names, counts and sizes are written, never an entry's content.

--history-since lists the recorded checks since then (`history.json`): each one's arguments, pack sizes and what its
circuit-check step said of its cache.

Inside a `research run` on a machine the store is read with the run's custody key, as `check.custody_dir` finds it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import lzma
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import traceback
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "research" / "src"))

NAME, REUSED, CHECK = "verdicts.tar.gz", "verdicts-reused.tar.xz", "tools/verity/check/check.py"
RECORDED_CHECKS = "(tool = 'check' OR (tool = 'python3' AND manifest_json LIKE '%tools/verity/check/check.py%'))"
XZ_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 1, "dict_size": 64 << 20}]
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{0,39}")
#: one entry in this many gets the schema-path walk (by its name's CRC), and every entry at least `DEEP_BIG` bytes
DEEP_EVERY, DEEP_BIG = 16, 4 << 20
#: a family's strip variants are decided once this many entries or bytes are buffered, for one at least `STRIP_MIN` bytes
STRIP_N, STRIP_BUF, STRIP_MIN = 100, 32 << 20, 4 << 20
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def dumps(v) -> str:
    return json.dumps(v, separators=(",", ":"), default=str)


@contextlib.contextmanager
def store_access():
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    d = Path(run_dir).resolve().parent.parent / "requests" / run_id / "custody" if run_dir and run_id else None
    if d is not None and (d / "store.toml").is_file() and (d / "cred.json").is_file():
        from research.store.custody import custody_env

        with custody_env(d):
            yield "the run's custody key"
    else:
        yield "this machine's store configuration"


def run_files(store, outputs: dict) -> tuple[str | None, dict[str, int]]:
    for art in dict.fromkeys(a for a in (outputs.get("run_files"), outputs.get("run_record")) if a):
        files = {f["path"]: f["bytes"] for f in store.listing(art)}
        if NAME in files:
            return art, files
    return None, {}


class Fam:
    __slots__ = ("n", "bytes", "top")

    def __init__(self):
        self.n, self.bytes, self.top = 0, 0, []

    def add(self, name: str, size: int) -> None:
        self.n += 1
        self.bytes += size
        if len(self.top) < 10:
            heapq.heappush(self.top, (size, name))
        elif size > self.top[0][0]:
            heapq.heapreplace(self.top, (size, name))

    def doc(self) -> dict:
        return {"n": self.n, "bytes": self.bytes, "top": [[s, n] for s, n in sorted(self.top, reverse=True)]}


def family(parts: tuple[str, ...], depth: int) -> str:
    return "/".join(parts[:min(depth, len(parts) - 1)])


def kind_of(doc) -> str:
    key = doc.get("key") if isinstance(doc, dict) else None
    req = key.get("request") if isinstance(key, dict) else None
    t = req.get("target") if isinstance(req, dict) else None
    if isinstance(t, list) and t and isinstance(t[0], str) and t[0] in ("task-list", "catalog-unit"):
        return t[0]
    return "target" if t is not None else "-"


def walk(v, path: str, depth: int, acc: Counter, cnt: Counter) -> None:
    if not isinstance(v, (dict, list)) or depth == 0 or not v:
        acc[path] += len(dumps(v))
        return
    if isinstance(v, dict):
        if len(v) <= 32 and all(IDENT.fullmatch(k) for k in v):
            for k, x in v.items():
                walk(x, f"{path}.{k}", depth - 1, acc, cnt)
            return
        p, vals = path + "{*}", list(v.values())
        acc[p + "<names>"] += sum(len(k) + 3 for k in v)
    else:
        p, vals = path + "[]", v
    cnt[p] += len(vals)
    if all(not isinstance(x, (dict, list)) for x in vals):
        acc[p + "<scalars>"] += len(dumps(vals))
    else:
        for x in vals:
            walk(x, p, depth - 1, acc, cnt)


def frame(name: str, payload: bytes) -> bytes:
    return name.encode()[:512].ljust(512, b"\0") + payload + b"\0" * (-len(payload) % 512)


class Gz:
    """A gzip stream at the level `verdicts._write` uses, counted and thrown away."""
    __slots__ = ("z", "bytes")

    def __init__(self):
        self.z, self.bytes = zlib.compressobj(9, zlib.DEFLATED, 31), 0

    def feed(self, data: bytes) -> None:
        self.bytes += len(self.z.compress(data))

    def close(self) -> int:
        self.bytes += len(self.z.flush())
        return self.bytes


class Strip:
    """One family's variants: its entries without one top-level key, gzipped, for each key holding a tenth of its bytes."""

    def __init__(self):
        self.buf: list[tuple[str, dict[str, str]]] | None = []
        self.buffered = 0
        self.seen: Counter = Counter()
        self.gz: dict[str, Gz] = {}

    def add(self, name: str, parts: dict[str, str]) -> None:
        if self.buf is None:
            self._feed(name, parts)
            return
        self.buf.append((name, parts))
        size = sum(map(len, parts.values()))
        self.buffered += size
        for k, s in parts.items():
            self.seen[k] += len(s)
        if len(self.buf) >= STRIP_N or self.buffered >= STRIP_BUF:
            self.decide()

    def decide(self) -> None:
        if self.buf is None:
            return
        buf, self.buf = self.buf, None
        if self.buffered >= STRIP_MIN and len(self.seen) <= 8:
            self.gz = {k: Gz() for k, s in self.seen.most_common(3) if s * 10 >= self.buffered}
        for name, parts in buf:
            self._feed(name, parts)

    def _feed(self, name: str, parts: dict[str, str]) -> None:
        for k, z in self.gz.items():
            body = "{" + ",".join(f"{json.dumps(j)}:{s}" for j, s in parts.items() if j != k) + "}"
            z.feed(frame(name, body.encode()))

    def close(self) -> dict[str, int]:
        self.decide()
        return {k: z.close() for k, z in self.gz.items()}


def profile(rid: str, scratch: str, out_dir: str) -> dict:
    from research.store.local import LocalStore

    t0 = time.monotonic()
    home = Path(scratch) / "store" / rid
    store = LocalStore(root=home)
    att = store.get_attempt(rid) or {}
    art, files = run_files(store, att.get("outputs") or {})
    argv = att.get("argv") or []
    ex = att.get("execution") or {}
    res: dict = {"run": rid, "machine": ex.get("machine"), "start_utc": ex.get("start_utc"), "wall_s": ex.get("wall_s"),
                 "commit": (att.get("source") or {}).get("commit") if isinstance(att.get("source"), dict) else None,
                 "args": args_of(argv), "pack_bytes": files.get(NAME), "reused_bytes": files.get(REUSED)}
    if art is None:
        res["error"] = "the attempt published no verdicts.tar.gz"
        return res
    pack = store.fetch(art, Path(scratch) / "packs" / rid, paths=[NAME]) / NAME
    log(f"{rid}: fetched {pack.stat().st_size / 2**20:.1f} MiB in {time.monotonic() - t0:.0f} s")
    fam = {2: defaultdict(Fam), 3: defaultdict(Fam)}
    kinds: dict[str, Fam] = defaultdict(Fam)
    keys: dict[str, Counter] = defaultdict(Counter)
    kind_keys: dict[str, Counter] = defaultdict(Counter)
    deep: dict[str, Counter] = defaultdict(Counter)
    deep_cnt: dict[str, Counter] = defaultdict(Counter)
    deep_of: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    whole, unparsed = Fam(), 0
    gz: dict[str, Gz] = defaultdict(Gz)
    strip: dict[str, Strip] = defaultdict(Strip)
    xz, xz_bytes = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=XZ_FILTERS), 0
    seen: set[int] = set()
    reads = {"entries": 0, "items": 0, "distinct_items": 0, "distinct_item_bytes": 0}
    maps: dict[str, dict] = defaultdict(lambda: {"n": 0, "bytes": 0, "distinct": set(), "distinct_bytes": 0})
    reports = {"n": 0, "bytes": 0, "distinct": set(), "distinct_bytes": 0}
    names: list[str] = []
    done = 0
    with tarfile.open(pack, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            name, size = m.name, len(data)
            parts = tuple(name.split("/"))
            f2 = family(parts, 2)
            names.append(name)
            whole.add(name, size)
            fam[2][f2].add(name, size)
            fam[3][family(parts, 3)].add(name, size)
            framed = frame(name, data)
            gz[f2].feed(framed)
            xz_bytes += len(xz.compress(framed))
            try:
                doc = json.loads(data)
            except ValueError:
                unparsed += 1
                continue
            if not isinstance(doc, dict):
                keys[f2][f"<{type(doc).__name__}>"] += size
                continue
            kind = f"{f2} [{kind_of(doc)}]"
            kinds[kind].add(name, size)
            part = {k: dumps(v) for k, v in doc.items()}
            for k, s in part.items():
                keys[f2][k] += len(s)
                kind_keys[kind][k] += len(s)
            strip[f2].add(name, part)
            if size >= DEEP_BIG or zlib.crc32(name.encode()) % DEEP_EVERY == 0:
                walk(doc, "", 6, deep[kind], deep_cnt[kind])
                deep_of[kind][0] += 1
                deep_of[kind][1] += size
            r = doc.get("reads")
            if isinstance(r, dict):
                reads["entries"] += 1
                for rk, mm in r.items():
                    if not isinstance(mm, dict):
                        continue
                    s = json.dumps(mm, sort_keys=True, separators=(",", ":"), default=str)
                    d = maps[rk]
                    d["n"] += 1
                    d["bytes"] += len(s)
                    h = hashlib.sha1(s.encode()).digest()
                    if h not in d["distinct"]:
                        d["distinct"].add(h)
                        d["distinct_bytes"] += len(s)
                    reads["items"] += len(mm)
                    for item, dig in mm.items():
                        dig = dig if dig is None or isinstance(dig, str) else str(dig)
                        hh = hash((rk, item, dig))
                        if hh not in seen:
                            seen.add(hh)
                            reads["distinct_item_bytes"] += len(item) + (len(dig) if dig is not None else 4) + 6
            if "report" in part:
                s = part["report"]
                reports["n"] += 1
                reports["bytes"] += len(s)
                h = hashlib.sha1(s.encode()).digest()
                if h not in reports["distinct"]:
                    reports["distinct"].add(h)
                    reports["distinct_bytes"] += len(s)
            done += 1
            if done % 2000 == 0:
                log(f"{rid}: {done} entries, {whole.bytes / 2**30:.2f} GiB read")
    xz_bytes += len(xz.flush())
    reads["distinct_items"] = len(seen)
    res.update({
        "entries": whole.n, "bytes": whole.bytes, "unparsed": unparsed, "top": whole.doc()["top"],
        "families": {str(d): {k: v.doc() for k, v in sorted(fs.items(), key=lambda kv: -kv[1].bytes)} for d, fs in fam.items()},
        "kinds": {k: v.doc() for k, v in sorted(kinds.items(), key=lambda kv: -kv[1].bytes)},
        "keys": {k: dict(c.most_common(12)) for k, c in keys.items()},
        "kind_keys": {k: dict(c.most_common(12)) for k, c in kind_keys.items()},
        "deep": {k: {"sampled_entries": deep_of[k][0], "sampled_bytes": deep_of[k][1], "paths": dict(c.most_common(25)),
                     "items": {p: deep_cnt[k][p] for p in deep_cnt[k]}} for k, c in deep.items()},
        "reads": reads,
        "read_maps": {k: {"n": d["n"], "bytes": d["bytes"], "distinct": len(d["distinct"]), "distinct_bytes": d["distinct_bytes"]}
                      for k, d in sorted(maps.items(), key=lambda kv: -kv[1]["bytes"])},
        "reports": {"n": reports["n"], "bytes": reports["bytes"], "distinct": len(reports["distinct"]),
                    "distinct_bytes": reports["distinct_bytes"]},
        "gz_by_family": {k: z.close() for k, z in gz.items()},
        "gz_family_without_key": {k: got for k, s in strip.items() if (got := s.close())},
        "xz_whole": xz_bytes,
        "seconds": round(time.monotonic() - t0, 1),
    })
    shutil.rmtree(home, ignore_errors=True)
    shutil.rmtree(pack.parent, ignore_errors=True)
    (Path(out_dir) / f"pack-{rid}.json").write_text(json.dumps(res, indent=1) + "\n")
    log(f"{rid}: {whole.n} entries, {whole.bytes / 2**30:.2f} GiB, done in {res['seconds']} s")
    return {**res, "names": names}


def args_of(argv: list[str]) -> list[str]:
    rest = argv[argv.index(CHECK) + 1:] if CHECK in argv else list(argv)
    return ["PACK" if i and rest[i - 1] == "--verdicts-in" else a for i, a in enumerate(rest)]


def history(since: str, scratch: str, out_dir: str) -> dict:
    from research.store.local import LocalStore

    t0 = time.monotonic()
    base = Path(scratch) / "store" / "history"
    store = LocalStore(root=base / "main")
    store.refresh(attempts_only=True, since=since)
    rows = store.index.sql("SELECT id, state, source_commit, host, start_utc, wall_s, manifest_json FROM attempts WHERE "
                           f"{RECORDED_CHECKS} ORDER BY id")
    log(f"history: {len(rows)} recorded checks since {since}, attempts refreshed in {time.monotonic() - t0:.0f} s")
    local = threading.local()

    def one(row) -> dict:
        rid, state, commit, host, start, wall, mj = row
        out: dict = {"run": rid, "state": state, "commit": (commit or "")[:10], "host": host, "start_utc": start,
                     "wall_s": round(wall) if wall else None}
        try:
            if not hasattr(local, "store"):
                local.store = LocalStore(root=base / f"t{threading.get_ident()}")
            doc = json.loads(mj)
            out["args"] = args_of(doc.get("argv") or [])
            art, files = run_files(local.store, doc.get("outputs") or {})
            out["pack_bytes"], out["reused_bytes"] = files.get(NAME), files.get(REUSED)
            if art and 0 < files.get("steps.jsonl", 0) < (4 << 20) and ("circuit-check" in " ".join(out["args"])
                                                                    or (files.get(NAME) or 0) >= (32 << 20)):
                d = base / "steps" / rid
                p = local.store.fetch(art, d, paths=["steps.jsonl"]) / "steps.jsonl"
                for line in p.read_text().splitlines():
                    st = json.loads(line)
                    if st.get("step") == "circuit-check":
                        c = st.get("cache")
                        if isinstance(c, dict) and isinstance(c.get("units"), dict):
                            c = {**c, "units": {k: len(v) if isinstance(v, list) else v for k, v in c["units"].items()}}
                        out["circuit_check"] = {"status": st.get("status"), "seconds": st.get("seconds"), "cache": c}
                shutil.rmtree(d, ignore_errors=True)
        except Exception as e:  # noqa: BLE001  (one run's row says why it has no sizes)
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out

    with ThreadPoolExecutor(12) as ex:
        got = list(ex.map(one, rows))
    doc = {"since": since, "runs": got, "seconds": round(time.monotonic() - t0, 1)}
    (Path(out_dir) / "history.json").write_text(json.dumps(doc, indent=1) + "\n")
    shutil.rmtree(base, ignore_errors=True)
    log(f"history: {len(got)} runs listed in {doc['seconds']} s")
    return doc


def _raw(pack: Path):
    """The pack's tar stream, gunzipped, in 4 MiB pieces."""
    d = zlib.decompressobj(31)
    with open(pack, "rb") as f:
        while chunk := f.read(4 << 20):
            yield d.decompress(chunk)
    yield d.flush()


def codec(pack: str, how: str, scratch: str) -> dict:
    """One pass over the pack's tar stream: its bytes, the bytes ``how`` writes it in, and this process's CPU seconds and peak
    memory.  ``none`` only gunzips, which every other pass's seconds include."""
    import resource

    t0, c0 = time.monotonic(), time.process_time()
    raw = out = 0
    res: dict = {"how": how}
    if how.startswith("gzip-"):
        z = zlib.compressobj(int(how[5:]), zlib.DEFLATED, 31)
    elif how == "xz":
        z = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=XZ_FILTERS)
    else:
        z = None
    kept = Path(scratch) / "codec.tar.xz"
    with open(kept, "wb") if how == "xz" else contextlib.nullcontext() as f:
        for piece in _raw(Path(pack)):
            raw += len(piece)
            if z is not None:
                got = z.compress(piece)
                out += len(got)
                if f is not None:
                    f.write(got)
        if z is not None:
            got = z.flush()
            out += len(got)
            if f is not None:
                f.write(got)
    res.update(raw=raw, out=out or None, wall_s=round(time.monotonic() - t0, 1), cpu_s=round(time.process_time() - c0, 1))
    if how == "xz":
        t1, c1, back = time.monotonic(), time.process_time(), 0
        with lzma.open(kept, "rb") as f:
            while piece := f.read(4 << 20):
                back += len(piece)
        res.update(read_back=back, read_wall_s=round(time.monotonic() - t1, 1), read_cpu_s=round(time.process_time() - c1, 1))
        kept.unlink()
    res["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return res


def codecs(rid: str, scratch: str, out_dir: str) -> dict:
    """What writing ``rid``'s pack costs gzipped at three levels and as xz with the dictionary, one pass each in its own process."""
    from research.store.local import LocalStore

    store = LocalStore(root=Path(scratch) / "store" / rid)
    art, files = run_files(store, (store.get_attempt(rid) or {}).get("outputs") or {})
    pack = store.fetch(art, Path(scratch) / "packs" / rid, paths=[NAME]) / NAME
    log(f"{rid}: fetched {pack.stat().st_size / 2**20:.1f} MiB")
    hows = ["none", "gzip-9", "gzip-6", "gzip-1", "xz"]
    with ProcessPoolExecutor(len(hows)) as ex:
        got = list(ex.map(codec, [str(pack)] * len(hows), hows, [scratch] * len(hows)))
    doc = {"run": rid, "pack_bytes": files.get(NAME), "passes": got}
    (Path(out_dir) / f"codecs-{rid}.json").write_text(json.dumps(doc, indent=1) + "\n")
    for g in got:
        log(f"{rid}: {g}")
    pack.unlink()
    return doc


def probe(out_dir: str) -> dict:
    """Whether this machine's check cache, or a directory at its top, is a link or on another filesystem (`verdicts.entries`
    walks without following links): names, link status and device numbers only."""
    import socket

    base = Path(os.environ.get("CHECK_CACHE") or Path.home() / ".cache" / "verity-check")
    doc: dict = {"host": socket.gethostname(), "set": bool(os.environ.get("CHECK_CACHE")), "exists": base.is_dir(),
                 "link": base.is_symlink(), "top": {}}
    if base.is_dir():
        dev = base.stat().st_dev
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                doc["top"][p.name] = {"link": p.is_symlink(), "same_filesystem": p.stat().st_dev == dev,
                                      "children": sum(1 for _ in p.iterdir())}
    (Path(out_dir) / "probe.json").write_text(json.dumps(doc, indent=1) + "\n")
    log(f"probe: {doc}")
    return doc


def mib(n: int | None) -> str:
    return "-" if n is None else f"{n / 2**20:.1f}"


def summary(packs: list[dict], hist: dict | None) -> str:
    out = ["# Verdict packs: where the bytes are", "", "Sizes in MiB (2^20 bytes); `raw` is uncompressed.", ""]
    for p in packs:
        out += [f"## {p['run']}", ""]
        if "error" in p:
            out += [p["error"], ""]
            continue
        out += [f"{p['machine']}, started {p['start_utc']}, {' '.join(p['args'])}", "",
                f"Pack {mib(p['pack_bytes'])} gzipped, {p['entries']} entries, {mib(p['bytes'])} raw; reused pack "
                f"{mib(p['reused_bytes'])}; whole as xz with a 64 MiB dictionary {mib(p['xz_whole'])}.", "",
                "| family | entries | raw | gz alone | largest entry |", "|---|---:|---:|---:|---:|"]
        for k, f in list(p["families"]["2"].items())[:10]:
            out.append(f"| `{k}` | {f['n']} | {mib(f['bytes'])} | {mib(p['gz_by_family'].get(k))} | {mib(f['top'][0][0])} |")
        out += ["", "| kind | entries | raw | top-level keys (raw) |", "|---|---:|---:|---|"]
        for k, f in list(p["kinds"].items())[:8]:
            ks = ", ".join(f"`{j}` {mib(s)}" for j, s in p["kind_keys"].get(k, {}).items())
            out.append(f"| `{k}` | {f['n']} | {mib(f['bytes'])} | {ks} |")
        out += ["", "Gzipped without one top-level key: " + ("; ".join(
            f"`{k}` without `{j}` {mib(s)}" for k, d in p["gz_family_without_key"].items() for j, s in d.items()) or "none"), ""]
        r = p["reads"]
        out += [f"`reads`: {r['entries']} entries hold {r['items']} items, {r['distinct_items']} distinct "
                f"({mib(r['distinct_item_bytes'])} once each).", "",
                "| reads map | maps | raw | distinct | raw once each |", "|---|---:|---:|---:|---:|"]
        for k, d in list(p["read_maps"].items())[:12]:
            out.append(f"| `{k}` | {d['n']} | {mib(d['bytes'])} | {d['distinct']} | {mib(d['distinct_bytes'])} |")
        d = p["reports"]
        out += ["", f"`report`: {d['n']}, {mib(d['bytes'])} raw; {d['distinct']} distinct, {mib(d['distinct_bytes'])}.", "",
                "Ten largest entries:", ""]
        out += [f"- {mib(s)} `{n}`" for s, n in p["top"]]
        for k, dd in list(p["deep"].items())[:3]:
            share = dd["sampled_bytes"] or 1
            out += ["", f"Schema paths of `{k}` over {dd['sampled_entries']} sampled entries ({mib(dd['sampled_bytes'])} raw):", ""]
            out += [f"- {100 * s / share:.1f}% `{path or '.'}`" + (f" ({dd['items'][path.split('<')[0]]} items)"
                                                                 if path.split('<')[0] in dd["items"] else "")
                    for path, s in list(dd["paths"].items())[:14]]
        out.append("")
    ok = [p for p in packs if "names" in p]
    if len(ok) > 1:
        out += ["## Entries two packs both hold (by path)", "", "| | " + " | ".join(p["run"][-4:] for p in ok) + " |",
                "|---|" + "---:|" * len(ok)]
        sets = [set(p["names"]) for p in ok]
        for p, a in zip(ok, sets):
            out.append(f"| {p['run']} | " + " | ".join(str(len(a & b)) for b in sets) + " |")
        out += ["", f"All together: {sum(map(len, sets))} entries, {len(set().union(*sets))} distinct paths.", ""]
    if hist and hist.get("runs"):
        out += ["## Recorded checks with a pack of 32 MiB or more, or whose circuit-check stored targets", "",
                "| run | host | commit | arguments | pack | circuit-check cache |", "|---|---|---|---|---:|---|"]
        for h in hist["runs"]:
            cc = h.get("circuit_check") or {}
            c = cc.get("cache") if isinstance(cc.get("cache"), dict) else {}
            if (h.get("pack_bytes") or 0) >= (32 << 20) or c.get("stored"):
                a = " ".join(x if len(x) < 40 else x[:12] + "..." + x[-24:] for x in h.get("args", []))
                out.append(f"| {h['run']} | {h['host']} | {h['commit']} | {a} | {mib(h.get('pack_bytes'))} | "
                           f"{cc.get('status', '')} hits {c.get('hits')} stored {c.get('stored')} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pack_profile", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--history-since")
    ap.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--codecs", metavar="RUN", help="what writing RUN's pack costs gzipped and as xz, instead of the profiles")
    ap.add_argument("--probe", action="store_true", help="whether this machine's check cache holds links, instead of the profiles")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.probe:
        probe(str(a.out))
    if a.codecs:
        with store_access() as how:
            log(f"store read with {how}")
            codecs(a.codecs, str(a.scratch), str(a.out))
    if a.probe or a.codecs:
        return 0
    with store_access() as how:
        log(f"store read with {how}; {a.jobs} processes")
        packs, hist = [], None
        with ProcessPoolExecutor(max(1, a.jobs)) as ex:
            jobs = [(rid, ex.submit(profile, rid, str(a.scratch), str(a.out))) for rid in a.runs]
            h = ex.submit(history, a.history_since, str(a.scratch), str(a.out)) if a.history_since else None
            for rid, j in jobs:
                try:
                    packs.append(j.result())
                except Exception as e:  # noqa: BLE001  (the other packs' profiles still count)
                    traceback.print_exc()
                    packs.append({"run": rid, "error": f"{type(e).__name__}: {e}"[:300]})
            if h is not None:
                try:
                    hist = h.result()
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    hist = {"error": f"{type(e).__name__}: {e}"[:300]}
                    (a.out / "history.json").write_text(json.dumps(hist) + "\n")
    (a.out / "summary.md").write_text(summary(packs, hist))
    log(f"wrote {a.out / 'summary.md'}")
    return 1 if any("error" in p for p in packs) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
