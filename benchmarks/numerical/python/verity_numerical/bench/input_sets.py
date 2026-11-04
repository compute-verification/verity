"""Input sets: one file layout for a subcircuit's captured and synthetic inputs, each input with its exact output.

A set is a directory ``<name>/`` (the exporter writes it under ``<out>/sets/``), exactly the layout of the vLLM VU exporter's
``vllm-vu-set/v1`` (``verity_vllm.pipeline.vu_export``), which this module reads and writes:

  ``<port>.<u8|u16|u32|u64>``  flat little-endian words, instance-major; a port whose width varies across instances also has
                               ``<port>.offsets.u64`` (n + 1 word offsets)
  ``manifest.json``            schema, set, relation, n, ports, the sha256 of every file, ``content_digest`` (:data:`CONTENT_RULE`:
                               provenance and index excluded) and ``provenance``
  ``index.json``               one row per instance, columns :data:`INDEX_COLUMNS` (a captured row: the VU it came from; a synthetic
                               row: ``source = "synthetic"`` and ``sub = {"draw": j, ...}``)

Captured sets carry ``schema = "vllm-vu-set/v1"``; synthetic ones (``verity_numerical.bench.generate``) ``"input-set/v1"``, with the
same fields.  :data:`PROVENANCE_FIELDS` are every set's provenance keys: a synthetic set writes all of them (the row, model and run
keys ``None``), and :func:`provenance` fills ``source`` (``captured`` for a ``vllm-vu-set/v1``) and ``recipe`` (``None``) for a set
that predates them.  A set's subcircuit is derived from its ``relation`` (``templates.identify``), never stored beside it.

    python -m verity_numerical.bench.input_sets show DIR
    python -m verity_numerical.bench.input_sets verify DIR [--lo L --hi H]      files against the manifest, then every instance
                                                                                re-evaluated by the IR evaluator
    python -m verity_numerical.bench.input_sets meta DIR                        the ``input-set/v1`` store meta (``research data put``)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import templates as TM

SCHEMA = "input-set/v1"
CAPTURED_SCHEMA = "vllm-vu-set/v1"
SCHEMAS = (SCHEMA, CAPTURED_SCHEMA)
KIND = "input-set/v1"
INDEX_COLUMNS = ["id", "vu", "request", "engine_step", "op_path", "row", "phase", "source", "sub"]
PROVENANCE_FIELDS = ("source", "recipe", "row", "row_key", "model", "revision", "research_run", "run_root", "program_digests",
                     "manifest_digest", "pair", "selection", "versions")
CONTENT_RULE = "sha256 of the canonical JSON of {schema, set, relation, n, ports, files without index.json}"
_WORD = {1: "u8", 2: "u16", 4: "u32", 8: "u64"}


def _unsigned(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a)).reshape(-1)
    if a.dtype.kind == "u":
        return a
    return a.view(TM.WORD_DTYPES[_WORD[a.dtype.itemsize]])


def content_digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write(root: str | Path, name: str, relation: Mapping[str, Any], ports: Mapping[str, Sequence[np.ndarray]], index_rows: Sequence[list],
          provenance: Mapping[str, Any], schema: str = SCHEMA) -> dict[str, Any]:
    """Write set ``name`` under ``root``: ``ports[port][i]`` the words of instance i, ``index_rows[i]`` its index row.  Returns the
    exporter's summary ``{set, n, bytes, content_digest, relation}``."""
    if schema not in SCHEMAS:
        raise ValueError(f"schema {schema!r} (one of {SCHEMAS})")
    n = len(index_rows)
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}

    def put(fname: str, arr: np.ndarray) -> None:
        b = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        (d / fname).write_bytes(b)
        files[fname] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

    for k in sorted(ports):
        arrs = [_unsigned(a) for a in ports[k]]
        if len(arrs) != n:
            raise ValueError(f"set {name} port {k}: {len(arrs)} instances, the index has {n}")
        dts = {a.dtype for a in arrs}
        if len(dts) != 1:
            raise ValueError(f"set {name} port {k}: word types {sorted(map(str, dts))}")
        wname = _WORD[arrs[0].dtype.itemsize]
        sizes = {int(a.size) for a in arrs}
        put(f"{k}.{wname}", np.concatenate(arrs))
        p: dict[str, Any] = {"dtype": wname, "file": f"{k}.{wname}", "words": sizes.pop() if len(sizes) == 1 else None}
        if p["words"] is None:
            put(f"{k}.offsets.u64", np.concatenate([[0], np.cumsum([a.size for a in arrs])]).astype(np.uint64))
            p["offsets"] = f"{k}.offsets.u64"
        meta[k] = p
    idx = {"schema": f"{schema}/index", "set": name, "columns": INDEX_COLUMNS, "rows": [list(r) for r in index_rows]}
    ib = (json.dumps(idx, separators=(",", ":")) + "\n").encode()
    (d / "index.json").write_bytes(ib)
    files["index.json"] = {"sha256": hashlib.sha256(ib).hexdigest(), "bytes": len(ib)}
    body = {"schema": schema, "set": name, "relation": dict(relation), "n": n, "ports": meta,
            "files": {k: v for k, v in files.items() if k != "index.json"}}
    digest = content_digest(body)
    man = dict(body, files=files, content_digest=digest, provenance=dict(provenance), content_digest_rule=CONTENT_RULE)
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str) + "\n")
    return {"set": name, "n": n, "bytes": sum(v["bytes"] for v in files.values()), "content_digest": digest,
            "relation": relation.get("subcircuit")}


def provenance(man: Mapping[str, Any]) -> dict[str, Any]:
    """A set's provenance with every :data:`PROVENANCE_FIELDS` key (``source`` from the schema when the set predates it)."""
    p = dict(man.get("provenance") or {})
    p.setdefault("source", "captured" if man.get("schema") == CAPTURED_SCHEMA else None)
    for k in PROVENANCE_FIELDS:
        p.setdefault(k, None)
    return p


@dataclass
class InputSet:
    """A set on disk.  ``port(name, lo, hi)`` reads instances ``[lo, hi)`` of a port: a 2-d array when its width is fixed, else a
    list of arrays."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> "InputSet":
        p = Path(path)
        if not (p / "manifest.json").is_file() and (p / "sets").is_dir():
            subs = [d for d in sorted((p / "sets").iterdir()) if (d / "manifest.json").is_file()]
            if len(subs) != 1:
                raise ValueError(f"{p}: {len(subs)} sets under sets/; name one")
            p = subs[0]
        man = json.loads((p / "manifest.json").read_text())
        if man.get("schema") not in SCHEMAS:
            raise ValueError(f"{p}: schema {man.get('schema')!r} is not an input set ({SCHEMAS})")
        return cls(p, man)

    @property
    def name(self) -> str:
        return self.manifest["set"]

    @property
    def n(self) -> int:
        return int(self.manifest["n"])

    @property
    def relation(self) -> dict[str, Any]:
        return self.manifest["relation"]

    @property
    def content_digest(self) -> str:
        return self.manifest["content_digest"]

    @property
    def provenance(self) -> dict[str, Any]:
        return provenance(self.manifest)

    @property
    def source(self) -> str | None:
        return self.provenance["source"]

    @property
    def subcircuit(self) -> TM.Subcircuit:
        return TM.identify(self.relation)

    def manifest_sha256(self) -> str:
        return hashlib.sha256((self.path / "manifest.json").read_bytes()).hexdigest()

    def index(self) -> list[list]:
        return json.loads((self.path / "index.json").read_text())["rows"]

    def port(self, name: str, lo: int = 0, hi: int | None = None) -> np.ndarray | list[np.ndarray]:
        p = self.manifest["ports"][name]
        hi = self.n if hi is None else hi
        dt = np.dtype(TM.WORD_DTYPES[p["dtype"]]).newbyteorder("<")
        if p.get("words") is not None:
            w = int(p["words"])
            arr = np.fromfile(self.path / p["file"], dtype=dt, count=(hi - lo) * w, offset=lo * w * dt.itemsize)
            return arr.reshape(hi - lo, w).astype(dt.newbyteorder("="))
        offs = np.fromfile(self.path / p["offsets"], dtype="<u8").astype(np.int64)
        arr = np.fromfile(self.path / p["file"], dtype=dt, count=int(offs[hi] - offs[lo]), offset=int(offs[lo]) * dt.itemsize)
        return [arr[offs[i] - offs[lo]:offs[i + 1] - offs[lo]].astype(dt.newbyteorder("=")) for i in range(lo, hi)]

    def instances(self, lo: int = 0, hi: int | None = None) -> dict[str, Any]:
        return {k: self.port(k, lo, hi) for k in self.manifest["ports"]}

    def check_files(self) -> list[str]:
        bad = []
        for fname, f in self.manifest["files"].items():
            fp = self.path / fname
            if not fp.is_file():
                bad.append(f"{fname}: missing")
            elif hashlib.sha256(fp.read_bytes()).hexdigest() != f["sha256"]:
                bad.append(f"{fname}: sha256 differs from the manifest")
        body = {k: self.manifest[k] for k in ("schema", "set", "relation", "n", "ports")}
        body["files"] = {k: v for k, v in self.manifest["files"].items() if k != "index.json"}
        if content_digest(body) != self.content_digest:
            bad.append("content_digest does not match the manifest's body")
        return bad

    def ref(self, lo: int = 0, hi: int | None = None, art: str | None = None) -> dict[str, Any]:
        """The result fingerprint's ``instances`` entry for ids ``[lo, hi)`` (``contract.instances_ref``'s fields, plus the set's
        subcircuit, content digest, source and store id)."""
        hi = self.n if hi is None else hi
        if not 0 <= lo < hi <= self.n:
            raise ValueError(f"range [{lo}, {hi}) outside the set's {self.n} instances")
        return {"dataset": f"{self.manifest['schema']}:{self.name}", "tier": self.name, "range": [lo, hi],
                "manifest_sha256": self.manifest_sha256(), "content_digest": self.content_digest, "subcircuit": self.subcircuit.id,
                "source": self.source, "art": art}

    def meta(self) -> dict[str, Any]:
        """The ``input-set/v1`` store meta."""
        sub = self.subcircuit
        pv = self.provenance
        return {"set": self.name, "subcircuit": sub.id, "template": sub.template.id, "parameters": sub.params, "source": pv["source"],
                "n": self.n, "content_digest": self.content_digest, "schema": self.manifest["schema"], "relation": self.relation,
                "row": pv["row"], "model": pv["model"], "research_run": pv["research_run"],
                "seed": (pv["recipe"] or {}).get("seed") if isinstance(pv["recipe"], Mapping) else None}


def verify(path: str | Path, lo: int = 0, hi: int | None = None, chunk: int = 256) -> dict[str, Any]:
    """Every file digest against the manifest, then every instance of ``[lo, hi)`` re-evaluated by the IR evaluator and compared
    word for word with its recorded outputs."""
    s = InputSet.open(path)
    hi = s.n if hi is None else hi
    files = s.check_files()
    if files:
        return {"set": s.name, "ok": False, "why": files}
    sub = s.subcircuit
    bad: list[int] = []
    for a in range(lo, hi, chunk):
        b = min(hi, a + chunk)
        batch = s.instances(a, b)
        got = sub.evaluate({p.name: batch[p.name] for p in sub.inputs})
        for p in sub.outputs:
            want = batch[p.name]
            for r in range(b - a):
                if not np.array_equal(np.asarray(got[p.name][r]).astype(np.uint64), np.asarray(want[r]).reshape(-1).astype(np.uint64)):
                    bad.append(a + r)
    bad = sorted(set(bad))
    return {"set": s.name, "subcircuit": sub.id, "source": s.source, "ok": not bad, "range": [lo, hi], "checked": hi - lo,
            "bad_n": len(bad), "bad": bad[:16], "content_digest": s.content_digest}


def subset(src: str | Path, dst_root: str | Path, lo: int, hi: int) -> Path:
    """Instances ``[lo, hi)`` of a set as a set of their own under ``dst_root`` (same name, relation and schema; provenance adds
    ``subset_of``: the source's content digest and the range)."""
    s = InputSet.open(src)
    rows = s.index()[lo:hi]
    rows = [[i, *r[1:]] for i, r in enumerate(rows)]
    ports = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in s.instances(lo, hi).items()}
    prov = dict(s.manifest.get("provenance") or {}, subset_of={"content_digest": s.content_digest, "range": [lo, hi]})
    write(dst_root, s.name, s.relation, ports, rows, prov, schema=s.manifest["schema"])
    return Path(dst_root) / s.name


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity_numerical.bench.input_sets")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "verify", "meta"):
        p = sp.add_parser(name)
        p.add_argument("path")
        if name == "verify":
            p.add_argument("--lo", type=int, default=0)
            p.add_argument("--hi", type=int, default=None)
    a = ap.parse_args(list(argv) if argv is not None else None)
    s = InputSet.open(a.path)
    if a.cmd == "show":
        sub = s.subcircuit
        print(json.dumps({"set": s.name, "n": s.n, "subcircuit": sub.id, "display": sub.display, "source": s.source,
                          "content_digest": s.content_digest, "ports": s.manifest["ports"], "provenance": s.provenance}, indent=1, default=str))
        return 0
    if a.cmd == "meta":
        print(json.dumps(s.meta(), indent=1, default=str))
        return 0
    r = verify(s.path, a.lo, a.hi)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
