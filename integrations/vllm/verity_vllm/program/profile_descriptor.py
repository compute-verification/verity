"""Profile a serialised descriptor (`verity.ml.circuits.codec`) by field: where the bytes go.

Bytes are measured as `canonical_json` bytes (the digest encoding: sorted keys, compact separators) of each
sub-object, so a definition's parts sum to its total up to the JSON structure around them (keys, braces,
commas), which is reported as `structure`. The profile is a pure measurement: nothing here changes the codec.

What is attributed, per definition and then aggregated:

  * `bindings`          the static binding record (`{"C": {...}, "T": 512}`); `C` is the whole model config
  * `sig`               body params + ret_type (the specialised signature)
  * `node.fn`           the callee id string of every body node (`Gemm_v1{K=1536,N=2048}`, a spec id
                        repeated once per node; `Const_v1{...}` ids carry the constant)
  * `node.type`         the `args[].type` schema of every node argument (repeated per node)
  * `node.refs.<form>`  the reference sequences by form: `affine`, `strided`, `explicit` (one triple per
                        leaf), `concat` (a list of parts)
  * `node.out`          the node output types
  * `node.meta`         form / axes / n / n_xs
  * `ret`               the body's return references
  * `structure`         the JSON framing around the above

Families are `base_id` (`Attention_v1`, ...) with the number of specialisations and their bytes; the root
body (`Root[Serve_v1{...}]`) is its own family. The "kinds of repeated content" table sums each attribution
over the descriptor and counts distinct payloads, so `bytes - distinct_bytes` is the bound on what sharing
one copy could save *inside this codec* (a reference to an existing definition, a shared definition, or a
range where an explicit list is affine).

    PYTHONPATH=. verity-vllm profile-descriptor --config B0 --LP 32 --STEPS 7 --out prof-B0.json
    PYTHONPATH=. verity-vllm profile-descriptor --desc program.json --out prof.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest

from verity_vllm.config import UsageError, option

_FAMILY = re.compile(r"^([A-Za-z0-9]+_v\d+)")


def _sz(o: Any) -> int:
    return len(canonical_json(o))


def _digest(o: Any) -> str:
    return hashlib.sha1(canonical_json(o)).hexdigest()


def _refs_form(r: dict) -> str:
    (k,) = r.keys()
    return k


def _refs_leaves(r: dict) -> int:
    k = _refs_form(r)
    if k == "affine":
        return r["affine"][3]
    if k == "strided":
        return r["strided"][6]
    if k == "explicit":
        return len(r["explicit"])
    if k == "alias":  # v1: the sequence of an earlier node argument (its length is not local to the part)
        return 0
    return sum(_refs_leaves(p) for p in r["concat"])


def _explicit_is_affine(items: list) -> bool:
    """Would this explicit list be one `affine` run (same space/idx, consecutive leaves)?"""
    if len(items) < 2:
        return True
    s, i, b = items[0]
    return all(x[0] == s and x[1] == i and x[2] == b + j for j, x in enumerate(items))


def _explicit_runs(items: list) -> int:
    """Number of affine runs an explicit list decomposes into (the `concat` of `affine`s it could be)."""
    runs = 0
    prev = None
    for it in items:
        if prev is None or not (it[0] == prev[0] and it[1] == prev[1] and it[2] == prev[2] + 1):
            runs += 1
        prev = it
    return runs


def _flatten(parts):
    for p in parts:
        if _refs_form(p) == "concat":
            yield from _flatten(p["concat"])
        else:
            yield p


class Profile:
    def __init__(self) -> None:
        self.top: dict[str, int] = {}
        self.defs: dict[str, dict] = {}           # per-definition attribution
        self.kinds: dict[str, dict] = defaultdict(lambda: {"bytes": 0, "count": 0, "digests": Counter(), "distinct_bytes": 0})
        self.families: dict[str, dict] = defaultdict(lambda: {"specialisations": 0, "bytes": 0, "parts": Counter()})
        self.node_fn_bytes: Counter = Counter()   # callee id -> bytes of its string occurrences
        self.node_fn_count: Counter = Counter()
        self.explicit: dict[str, Any] = {"lists": 0, "bytes": 0, "leaves": 0, "affine_lists": 0, "affine_bytes": 0,
                                         "runs": 0, "by_family": Counter()}
        self.concat_parts = 0
        self.concat_stats: dict[str, Counter] = defaultdict(Counter)  # family -> concats / parts / single-leaf parts / same-target parts
        self.total = 0

    def _kind(self, kind: str, payload: Any, nbytes: int | None = None) -> int:
        n = _sz(payload) if nbytes is None else nbytes
        k = self.kinds[kind]
        k["bytes"] += n
        k["count"] += 1
        d = _digest(payload)
        if d not in k["digests"]:
            k["distinct_bytes"] += n
        k["digests"][d] += 1
        return n

    def add_definition(self, fid: str, d: dict, callees: list | None = None) -> None:
        fam = "Root" if fid.startswith("Root[") else (_FAMILY.match(fid).group(1) if _FAMILY.match(fid) else fid)
        parts: Counter = Counter()
        total = _sz(d)
        if d["form"] == "primitive":
            parts["sig"] += self._kind("sig", [d["params"], d["ret"]])
        else:
            parts["bindings"] += self._kind("bindings", d["bindings"])
            body = d["body"]
            parts["sig"] += self._kind("sig", [body["params"], body["ret_type"]])
            parts["ret"] += self._kind(f"ret.{_refs_form(body['ret'])}", body["ret"])
            for n in body["nodes"]:
                fn = n["fn"]
                nb = len(canonical_json(fn))
                parts["node.fn"] += self._kind("node.fn", fn, nb)
                label = callees[fn] if callees is not None and isinstance(fn, int) else fn  # v1: interned
                self.node_fn_bytes[label] += nb
                self.node_fn_count[label] += 1
                for a in n["args"]:
                    parts["node.type"] += self._kind("node.type", a["type"])
                    r = a["refs"]
                    form = _refs_form(r)
                    rb = self._kind(f"node.refs.{form}", r)
                    parts[f"node.refs.{form}"] += rb
                    self._explicit_stats(r, rb, fam)
                parts["node.out"] += self._kind("node.out", n["out"])
                meta = {k: v for k, v in n.items() if k in ("form", "axes", "n", "n_xs")}
                parts["node.meta"] += self._kind("node.meta", meta)
        parts["structure"] += total - sum(parts.values())
        self.defs[fid] = {"family": fam, "bytes": total, "parts": dict(parts),
                          "nodes": len(d["body"]["nodes"]) if d["form"] == "composite" else 0}
        f = self.families[fam]
        f["specialisations"] += 1
        f["bytes"] += total
        f["parts"].update(parts)

    def _explicit_stats(self, r: dict, rb: int, fam: str, top: bool = True) -> None:
        form = _refs_form(r)
        if form == "explicit":
            items = r["explicit"]
            self.explicit["lists"] += 1
            self.explicit["bytes"] += rb
            self.explicit["leaves"] += len(items)
            self.explicit["runs"] += _explicit_runs(items)
            self.explicit["by_family"][fam] += rb
            if _explicit_is_affine(items):
                self.explicit["affine_lists"] += 1
                self.explicit["affine_bytes"] += rb
        elif form == "concat":
            parts = r["concat"]
            self.concat_parts += len(parts)
            if top:  # statistics over the top-level concat only (nested parts are flattened into it)
                flat = list(_flatten(parts))
                cs = self.concat_stats[fam]
                cs["concats"] += 1
                cs["parts"] += len(flat)
                cs["bytes"] += rb
                cs["single_leaf_parts"] += sum(1 for p in flat if _refs_form(p) == "affine" and p["affine"][3] == 1)
                cs["alias_parts"] += sum(1 for p in flat if _refs_form(p) == "alias")
                targets = Counter((_refs_form(p), tuple(p[_refs_form(p)][:2])) for p in flat)
                cs["same_target_parts"] += max(targets.values()) if len(targets) == 1 else 0
                cs["all_parts_distinct_nodes"] += 1 if len(targets) == len(flat) else 0
            for p in parts:
                self._explicit_stats(p, _sz(p), fam, top=False)

    def finish(self, desc: dict) -> dict:
        for k in ("schema", "root", "program_fn", "callees", "types"):  # the last two: v1 interning tables
            if k in desc:
                self.top[k] = _sz(desc[k])
        self.top["annotations"] = _sz(desc.get("annotations", {}))
        self.top["definitions"] = sum(x["bytes"] for x in self.defs.values()) + _sz({k: 0 for k in self.defs}) - 2 * len(self.defs)
        self.total = _sz({k: v for k, v in desc.items() if k != "annotations"})
        kinds = {k: {"bytes": v["bytes"], "count": v["count"], "distinct": len(v["digests"]), "distinct_bytes": v["distinct_bytes"],
                     "shareable_bytes": v["bytes"] - v["distinct_bytes"]} for k, v in self.kinds.items()}
        fams = {k: {"specialisations": v["specialisations"], "bytes": v["bytes"], "parts": dict(v["parts"])}
                for k, v in sorted(self.families.items(), key=lambda kv: -kv[1]["bytes"])}
        offenders = sorted(((fam, part, b) for fam, v in self.families.items() for part, b in v["parts"].items()),
                           key=lambda x: -x[2])
        top_fn = [{"fn": fn, "occurrences": self.node_fn_count[fn], "bytes": b} for fn, b in self.node_fn_bytes.most_common(15)]
        ex = dict(self.explicit)
        ex["by_family"] = dict(ex["by_family"].most_common())
        return {"schema": "verity-ir/descriptor-profile/v0", "descriptor_bytes": self.total, "top_level": self.top,
                "definitions": len(self.defs), "families": fams, "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1]["bytes"])),
                "offenders": [{"family": f, "part": p, "bytes": b, "share": b / self.total} for f, p, b in offenders[:25]],
                "node_fn_top": top_fn, "explicit": ex, "concat_parts": self.concat_parts,
                "concat_by_family": {k: dict(v) for k, v in sorted(self.concat_stats.items(), key=lambda kv: -kv[1]["bytes"])},
                "largest_definitions": sorted(({"id": k, **v} for k, v in self.defs.items()), key=lambda x: -x["bytes"])[:15]}


def profile(desc: dict) -> dict:
    p = Profile()
    for fid, d in desc["definitions"].items():
        p.add_definition(fid, d, desc.get("callees"))
    out = p.finish(desc)
    out["program_digest"] = program_digest(desc)
    out["gates"] = desc.get("annotations", {}).get("gates")
    return out


def build(config: str, LP: int, STEPS: int) -> dict:
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1
    C = {"B1": b1.B1_CONFIG, "B0": b1.B0_CONFIG}[config]
    t0 = time.time()
    program = Program(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
    t1 = time.time()
    desc = encode_program(program)
    t2 = time.time()
    desc["annotations"] = {"profile_build_s": round(t1 - t0, 1), "profile_encode_s": round(t2 - t1, 1), "gates": program.gates}
    return desc


def render(prof: dict) -> str:
    out = [f"descriptor {prof['descriptor_bytes']:,} B, {prof['definitions']} definitions, digest {prof.get('program_digest', '?')[:16]}…, gates {prof.get('gates')}"]
    out.append("top-level: " + ", ".join(f"{k} {v:,}" for k, v in prof["top_level"].items()))
    out.append("families:")
    for k, v in prof["families"].items():
        out.append(f"  {k:36s} {v['specialisations']:>5} spec {v['bytes']:>14,} B  {100 * v['bytes'] / prof['descriptor_bytes']:5.1f}%")
    out.append("kinds:")
    for k, v in prof["kinds"].items():
        out.append(f"  {k:24s} {v['bytes']:>14,} B  n={v['count']:>9,} distinct={v['distinct']:>8,} shareable={v['shareable_bytes']:>14,} B")
    out.append("offenders:")
    for o in prof["offenders"][:12]:
        out.append(f"  {o['family']:28s} {o['part']:20s} {o['bytes']:>14,} B {100 * o['share']:5.1f}%")
    ex = prof["explicit"]
    for fam, cs in list(prof["concat_by_family"].items())[:4]:
        out.append(f"concat {fam:20s} {cs['concats']:>7,} concats {cs['parts']:>11,} parts {cs['bytes']:>13,} B; single-leaf parts {cs['single_leaf_parts']:,}; "
                   f"same-target parts {cs['same_target_parts']:,}; concats whose parts are all distinct nodes {cs['all_parts_distinct_nodes']:,}")
    out.append(f"explicit lists: {ex['lists']:,} lists / {ex['bytes']:,} B / {ex['leaves']:,} leaves; "
               f"{ex['affine_lists']:,} are single affine runs ({ex['affine_bytes']:,} B); {ex['runs']:,} affine runs total")
    out.append("node.fn top:")
    for t in prof["node_fn_top"][:8]:
        out.append(f"  {t['fn'][:90]:90s} x{t['occurrences']:>8,} {t['bytes']:>12,} B")
    return "\n".join(out)


@dataclass(frozen=True, kw_only=True)
class Options:
    desc: Path | None = option("--desc", default=None, type=Path, help="a serialised descriptor (JSON)")
    config: str | None = option("--config", default=None, choices=["B0", "B1"])
    LP: int | None = option("--LP", default=None, type=int)
    STEPS: int | None = option("--STEPS", default=None, type=int)
    out: Path | None = option("--out", default=None, type=Path)
    save_desc: Path | None = option("--save-desc", default=None, type=Path,
                                    help="also write the built descriptor (canonical JSON)")


def main(a: Options) -> int:
    if a.desc:
        desc = json.loads(a.desc.read_text())
    else:
        if not (a.config and a.LP is not None and a.STEPS is not None):
            raise UsageError("--desc or --config/--LP/--STEPS")
        desc = build(a.config, a.LP, a.STEPS)
        print(f"[profile] built {a.config} LP={a.LP} STEPS={a.STEPS}: {desc['annotations']}", file=sys.stderr, flush=True)
    if a.save_desc:
        a.save_desc.write_bytes(canonical_json(desc))
    prof = profile(desc)
    prof["source"] = {"desc": str(a.desc) if a.desc else None, "config": a.config, "LP": a.LP, "STEPS": a.STEPS,
                      "annotations": desc.get("annotations", {})}
    if a.out:
        a.out.write_text(json.dumps(prof, indent=1))
    print(render(prof))
    return 0
