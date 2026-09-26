"""A served row's Program as a graph (`vllm-program-graph/v1`): module bodies, their Calls, the dataflow between them.

Read from the Program and the IR, never from names: the Calls, their Definitions and their reads are the Build's request
Program (`query.program_view.from_instances`, the same columnar body the required-value manifest is built from); a Call's
module is the runtime correspondence's recorded owning module (`correspondence.reader_for_query.Correspondence.module_of`), the
unit `Q_module_body_v1` partitions by.

  modules    every module body (the Calls one module issued itself; "" = the Calls no module issued) plus the modules that
             only contain others, as a tree: parent = the nearest enclosing module (torch `named_modules()` nesting, where a
             child's path extends its parent's by `.`), order = the Program order of the module's first Call among its siblings
  groups     a body's Calls grouped by Definition: the spec, except that a static taking more than `VARYING_MAX` values inside
             one (module, family) -- attention's history length T, one per position -- is kept as a range; each group has its
             Call count, its subcircuit template (`vu_export.template_of`: the exporter's decomposition) and instance count,
             and, from a run's record, its verification units in the run and the exported VUs
  edges      producer group -> consumer group, counted in Values read (a Call reading one port of another Call is one read);
             reads of prescribed inputs (weights, prompt) per group by input port
  hashes     per module body a structural hash (its groups in order: family, constant statics, varying static names, Call
             count, reads inside the body by group index, reads from outside and of prescribed inputs by count) and a subtree
             hash (the body hash and the children's subtree hashes in order), so identical siblings fold (`layers.0` ..
             `layers.15`) without reading a name

A row with several request Programs (B > 1) is their union: the same module and group keys, counts summed, one graph.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from verity_vllm.config import option
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.pipeline.vu_export import template_of
from verity_vllm.query.program_view import ProgramView, _read_instance_header, from_instances, spec_statics

SCHEMA = "vllm-program-graph/v1"
VARYING_MAX = 8


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _parent(path: str, known: set[str]) -> str:
    """The nearest enclosing module of `path` ("" = the root)."""
    p = path
    while "." in p:
        p = p.rsplit(".", 1)[0]
        if p in known:
            return p
    return ""


class _Acc:
    """The graph over one or more request Programs of one row."""

    def __init__(self) -> None:
        self.modules: dict[str, dict[str, Any]] = {}
        self.groups: dict[tuple[str, str], dict[str, Any]] = {}
        self.edges: Counter = Counter()
        self.param_reads: Counter = Counter()
        self.programs: list[dict[str, Any]] = []
        self.call_keys: list[list[tuple[str, str]]] = []

    def add(self, P: ProgramView, digest: str | None, label: str) -> None:
        cols = P.columns
        if cols is None:
            raise ValueError("program_graph reads the columnar body (from_instances)")
        C = Correspondence(P)
        n = cols.n
        mods = [C.module_of(k) for k in range(n)]
        fam_specs: dict[tuple[str, str], set[str]] = {}
        for k in range(n):
            fam_specs.setdefault((mods[k], cols.family_of(k)), set()).add(cols.specs[cols.spec_idx[k]])
        varying: dict[tuple[str, str], set[str]] = {}
        for key, specs in fam_specs.items():
            vals: dict[str, set] = {}
            for s in specs:
                for a, v in spec_statics(s).items():
                    vals.setdefault(a, set()).add(json.dumps(v, sort_keys=True, default=str))
            varying[key] = {a for a, vs in vals.items() if len(vs) > VARYING_MAX}
        gkey_of_spec: dict[tuple[str, str], str] = {}
        call_group = np.empty(n, dtype=np.int64)
        gindex: dict[tuple[str, str], int] = {}
        gkeys: list[tuple[str, str]] = []
        for k in range(n):
            m, spec = mods[k], cols.specs[cols.spec_idx[k]]
            fam = cols.family_of(k)
            gk = gkey_of_spec.get((m, spec))
            if gk is None:
                var = varying[(m, fam)]
                st = spec_statics(spec)
                gk = gkey_of_spec[(m, spec)] = fam + json.dumps({a: v for a, v in st.items() if a not in var}, sort_keys=True, default=str) \
                    + ("|" + ",".join(sorted(var)) if var else "")
            key = (m, gk)
            gi = gindex.get(key)
            if gi is None:
                gi = gindex[key] = len(gkeys)
                gkeys.append(key)
            call_group[k] = gi
            g = self.groups.get(key)
            if g is None:
                st = spec_statics(spec)
                var = varying[(m, fam)]
                g = self.groups[key] = {"module": m, "family": fam, "statics": {a: v for a, v in st.items() if a not in var},
                                        "varying": {a: [None, None] for a in sorted(var)}, "calls": 0, "instances": 0,
                                        "specs": Counter(), "first": None}
            g["calls"] += 1
            g["specs"][spec] += 1
            _t, _p, mult = template_of(spec)
            g["instances"] += mult
            for a in g["varying"]:
                v = spec_statics(spec).get(a)
                lo, hi = g["varying"][a]
                g["varying"][a] = [v if lo is None or v < lo else lo, v if hi is None or v > hi else hi]
            mo = self.modules.setdefault(m, {"calls": 0, "first": None})
            mo["calls"] += 1
            if mo["first"] is None or (len(self.programs), k) < mo["first"]:
                mo["first"] = (len(self.programs), k)
            if g["first"] is None:
                g["first"] = (len(self.programs), k)
        off = np.frombuffer(cols.read_off, dtype=np.int64)
        prod_all = np.frombuffer(cols.read_call, dtype=np.int32)
        port_all = np.frombuffer(cols.read_port, dtype=np.uint16)
        G = max(len(gkeys), 1)
        edge_c: Counter = Counter()
        pin_c: Counter = Counter()
        c0 = 0
        while c0 < n:                                              # consumer Calls in blocks of ~2^24 reads: memory stays O(block)
            c1 = int(np.searchsorted(off, off[c0] + (1 << 24), side="right"))
            c1 = min(max(c1 - 1, c0 + 1), n)
            lo, hi = int(off[c0]), int(off[c1])
            cons = np.repeat(call_group[c0:c1], np.diff(off[c0:c1 + 1]))
            prod = prod_all[lo:hi]
            calls = prod >= 0
            if calls.any():
                u, c = np.unique(call_group[prod[calls]] * G + cons[calls], return_counts=True)
                edge_c.update(dict(zip(u.tolist(), c.tolist())))
            if (~calls).any():
                ids = (-prod[~calls].astype(np.int64)) << 16 | port_all[lo:hi][~calls].astype(np.int64)
                u, c = np.unique(cons[~calls] << 40 | ids, return_counts=True)
                pin_c.update(dict(zip(u.tolist(), c.tolist())))
            c0 = c1
        for code, cnt in edge_c.items():
            self.edges[(gkeys[code // G], gkeys[code % G])] += int(cnt)
        for code, cnt in pin_c.items():
            g, iid, pi = code >> 40, -((code >> 16) & ((1 << 24) - 1)), code & 0xFFFF
            inp = P.input(int(iid))
            self.param_reads[(gkeys[g], f"{inp.name}:{cols.port_names[pi]}" if cols.port_names[pi] != "out" else inp.name)] += int(cnt)
        self.call_keys.append([gkeys[g] for g in call_group.tolist()])
        self.programs.append({"label": label, "program_digest": digest, "calls": n, "modules": len(set(mods))})

    def graph(self, units: dict[tuple[str, str], int] | None = None, exported: dict[tuple[str, str], int] | None = None) -> dict[str, Any]:
        bodies = set(self.modules)
        known = set(bodies)
        for m in list(bodies):
            p = m
            while "." in p:
                p = p.rsplit(".", 1)[0]
                known.add(p)
        known.discard("")
        tree: dict[str, dict[str, Any]] = {"": {"path": "", "parent": None, "body": "" in bodies, "children": []}}
        for m in sorted(known):
            tree[m] = {"path": m, "parent": _parent(m, known), "body": m in bodies, "children": []}

        def first(m: str) -> tuple:
            own = self.modules.get(m, {}).get("first")
            kids = [first(c) for c in known if tree[c]["parent"] == m]
            return min([x for x in [own, *kids] if x is not None], default=(1 << 62, 0))

        firsts = {m: first(m) for m in tree}
        for m in sorted(known, key=lambda x: firsts[x]):
            tree[tree[m]["parent"]]["children"].append(m)
        gid = {key: f"g{i}" for i, key in enumerate(sorted(self.groups, key=lambda k: self.groups[k]["first"]))}
        groups_of: dict[str, list[tuple[str, str]]] = {}
        for key in sorted(self.groups, key=lambda k: self.groups[k]["first"]):
            groups_of.setdefault(key[0], []).append(key)
        out_groups = []
        for key in sorted(self.groups, key=lambda k: self.groups[k]["first"]):
            g = self.groups[key]
            spec0 = g["specs"].most_common(1)[0][0]
            t, p, _m = template_of(spec0)
            if g["varying"]:
                p = {a: v for a, v in p.items() if a not in g["varying"]}
            out_groups.append({"id": gid[key], "module": g["module"], "definition": g["family"], "statics": g["statics"],
                               "varying": g["varying"], "specs": len(g["specs"]), "spec": spec0 if len(g["specs"]) == 1 else None,
                               "calls": g["calls"], "template": t, "template_params": p, "instances": g["instances"],
                               "units_in_run": (units or {}).get(key), "exported_vus": (exported or {}).get(key),
                               "prescribed_reads": {pn: c for (gk, pn), c in sorted(self.param_reads.items()) if gk == key}})
        # structural hashes: a body's groups in order, reads by group index inside the body, outside reads by count
        body_hash: dict[str, str] = {}
        for m, keys in groups_of.items():
            idx = {k: i for i, k in enumerate(keys)}
            sig = []
            for k in keys:
                g = self.groups[k]
                inside = sorted((idx[a], c) for (a, b), c in self.edges.items() if b == k and a in idx)
                outside = sum(c for (a, b), c in self.edges.items() if b == k and a not in idx)
                prescribed = sum(c for (gk, _pn), c in self.param_reads.items() if gk == k)
                sig.append([g["family"], g["statics"], sorted(g["varying"]), g["calls"], inside, outside, prescribed])
            body_hash[m] = _h(sig)
        subtree: dict[str, str] = {}

        def sub(m: str) -> str:
            if m not in subtree:
                subtree[m] = _h([body_hash.get(m), [sub(c) for c in tree[m]["children"]]])
            return subtree[m]

        for m in tree:
            sub(m)
        out_modules = [{"path": m, "parent": tree[m]["parent"], "order": (tree[tree[m]["parent"]]["children"].index(m) if m else 0),
                        "body": tree[m]["body"], "children": tree[m]["children"], "calls": self.modules.get(m, {}).get("calls", 0),
                        "groups": [gid[k] for k in groups_of.get(m, [])], "body_hash": body_hash.get(m), "subtree_hash": subtree[m]}
                       for m in sorted(tree, key=lambda x: firsts[x])]
        mod_edges: Counter = Counter()
        for (a, b), c in self.edges.items():
            if a[0] != b[0]:
                mod_edges[(a[0], b[0])] += c
        return {"schema": SCHEMA, "programs": self.programs, "modules": out_modules, "groups": out_groups,
                "edges": [{"from": gid[a], "to": gid[b], "reads": c} for (a, b), c in sorted(self.edges.items(), key=lambda e: (gid[e[0][0]], gid[e[0][1]]))],
                "module_edges": [{"from": a, "to": b, "reads": c} for (a, b), c in sorted(mod_edges.items())],
                "rules": {"module_of": "Correspondence.module_of (the recorded owning module; Q_module_body_v1's unit)",
                          "parent": "the nearest enclosing module: a child's path extends its parent's by '.' (torch named_modules nesting)",
                          "order": "Program order of the first Call in the module's subtree", "group": f"Definition, a static with > {VARYING_MAX} "
                          "values in one (module, family) kept as a range", "edge": "Values read: one per (consumer Call, producer Call, port)",
                          "body_hash": "sha256 over the body's groups in order: family, constant statics, varying static names, Call count, "
                          "reads inside the body by group index, reads from outside (count), prescribed-input reads (count)",
                          "subtree_hash": "sha256 of [body_hash, children's subtree hashes in order]"}}


def wrapper_prefix(strata: dict[str, dict], acc: _Acc) -> str:
    """The strata name the collector's op_path: the Program module path minus the export wrapper's prefix, determined here as the
    prefix under which the most strata op_paths are Program module paths."""
    mods = {k[0] for k in acc.groups}
    ops = {k.split("|")[4] for k in strata if k.count("|") >= 4}
    prefixes = {m[: len(m) - len(o)] for m in mods for o in ops if o and m.endswith(o) and (len(m) == len(o) or m[len(m) - len(o) - 1] == ".")}
    return max(sorted(prefixes), key=lambda p: sum(1 for o in ops if p + o in mods), default="")


def units_from_strata(strata: dict[str, dict], acc: _Acc, prefix: str) -> dict[tuple[str, str], int]:
    """A run's verification units per group, from its replay strata (`spec|request|rank|phase|op_path|lifetime` -> population)."""
    out: Counter = Counter()
    unplaced: Counter = Counter()
    for k, v in strata.items():
        parts = k.split("|")
        if len(parts) < 5:
            continue
        spec, m = parts[0], prefix + parts[4]
        fam, st = spec.split("{", 1)[0], spec_statics(spec)
        for key, g in acc.groups.items():
            if key[0] == m and g["family"] == fam and all(st.get(a) == v for a, v in g["statics"].items() if a in st):
                out[key] += int(v.get("population", 0))
                break
        else:
            unplaced[spec] += int(v.get("population", 0))
    # a stratum whose op_path is no module (the sampler's rows, addressed at the runner's sampler, one module per Call in the
    # correspondence): its units go to the groups of that Definition no stratum placed, in proportion to their Calls
    for spec, n in unplaced.items():
        fam, st = spec.split("{", 1)[0], spec_statics(spec)
        cand = [k for k, g in acc.groups.items() if k not in out and g["family"] == fam and all(st.get(a) == v for a, v in g["statics"].items() if a in st)]
        tot = sum(acc.groups[k]["calls"] for k in cand)
        if not tot:
            continue
        share = {k: n * acc.groups[k]["calls"] // tot for k in cand}
        for k in sorted(cand, key=lambda k: -(n * acc.groups[k]["calls"] % tot))[: n - sum(share.values())]:
            share[k] += 1
        out.update({k: c for k, c in share.items() if c})
    return dict(out)


def exported_from_vus(vus_jsonl: str | Path, acc: _Acc, prefix: str, request_programs: dict[str, str] | None = None) -> dict[tuple[str, str], int]:
    """Exported VUs per group: by (op_path, spec); a VU whose op_path is no module (the sampler's) by its Call id in its request's
    Program -- the one Program, or the one `request_programs` (request -> Build dir, the VU store's `program.json`) names."""
    label_ix = {p["label"]: k for k, p in enumerate(acc.programs)}
    out: Counter = Counter()
    for line in Path(vus_jsonl).read_text().splitlines():
        e = json.loads(line)
        if e.get("result") is not True:
            continue
        m, spec = prefix + str(e["op_path"]), str(e["spec"])
        fam, st = spec.split("{", 1)[0], spec_statics(spec)
        for key, g in acc.groups.items():
            if key[0] == m and g["family"] == fam and all(st.get(a) == v for a, v in g["statics"].items()):
                out[key] += 1
                break
        else:
            d = (request_programs or {}).get(str(e.get("request")))
            k = label_ix.get(Path(d).name) if d else (0 if len(acc.call_keys) == 1 else None)
            if k is not None and 0 <= int(e["i"]) < len(acc.call_keys[k]):
                out[acc.call_keys[k][int(e["i"])]] += 1
    return dict(out)


def program_graph(instance_paths: Iterable[str | Path], *, row: dict[str, Any] | None = None, strata: dict[str, dict] | None = None,
                  vus_jsonl: str | Path | None = None, request_programs: dict[str, str] | None = None) -> dict[str, Any]:
    """The row's graph over its request Programs (`instances.json.gz` paths), with the run's units (`strata`, a replay record's)
    and exported VUs (`vus_jsonl`, a VU export's; `request_programs` from its store) when given."""
    acc = _Acc()
    for p in instance_paths:
        hdr = _read_instance_header(str(p))
        acc.add(from_instances(str(p)), hdr.get("digest"), Path(p).parent.name)
    prefix = wrapper_prefix(strata, acc) if strata else ""
    units = units_from_strata(strata, acc, prefix) if strata else None
    exported = exported_from_vus(vus_jsonl, acc, prefix, request_programs) if vus_jsonl else None
    return {"row": row or {}, "wrapper_prefix": prefix, **acc.graph(units, exported)}


@dataclass(frozen=True, kw_only=True)
class Options:
    """write a served row's program.json from its request Programs"""

    out: str = option("out", help="the program.json to write")
    instances: list[str] = option("instances", nargs="+", help="the row's request Programs (instances.json.gz)")
    strata: str | None = option("--strata", default=None, help="a replay record (sampled_replay_p0.json, or a regression record) for the run's units")
    vus: str | None = option("--vus", default=None, help="a VU export's vus.jsonl for the exported VUs")
    store: str | None = option("--store", default=None, help="the VU export's store (<export>/store): request -> Program, to place sampler VUs")
    row: str | None = option("--row", default=None, help="JSON object: the row block (row number, serving config, run)")


def main(a: Options) -> int:
    strata = None
    if a.strata:
        rec = json.loads(Path(a.strata).read_text())
        strata = (rec.get("sampled_replay") or (rec.get("replay_partition") or {})).get("strata")
    g = program_graph(a.instances, row=json.loads(a.row) if a.row else None, strata=strata, vus_jsonl=a.vus,
                      request_programs=request_programs_of(a.store) if a.store else None)
    Path(a.out).write_text(json.dumps(g, separators=(",", ":"), default=str) + "\n")
    return 0


def request_programs_of(store: str | Path) -> dict[str, str]:
    """request -> Build dir of its Program, from a VU store's `programs/*/program.json`."""
    out: dict[str, str] = {}
    for p in Path(store).glob("programs/*/program.json"):
        out.update({r: d for r, d in (json.loads(p.read_text()).get("build_dirs") or {}).items() if d})
    return out
