"""The Boolean re-verification of a deferred Commit: the Commit's own C2 replay from its sealed bundle (`pipeline.cpu_replay`), with
every picked row evaluated by the Boolean version of its Definition (`verity.ml.definitions.boolean.boolean_version`, the switch's
one table) on the committed inputs, bit-sliced (`verity.ml.circuits.evaluation.bits.evaluate_bits`), and compared with the committed outputs.

    verity-vllm boolean-replay BUNDLE|KEEP --row-dir D --checkpoint-dir CK --out OUT [--checkpoints-manifest M --model REPO
                               --revision REV] [--word]

BUNDLE is the Commit's sealed replay bundle, or KEEP the slim keep its decided replay left (`commit/replay_slim_p<pair>`, the kept
leaves, on the node; `cpu_replay.open_keep`).  Like the replay it reruns, it's a prover-side check on clear values.  The Commit's records (`runs.jsonl`, `verdict.json`, `summary.json`, the binding map) are copied
into ``OUT/boolean`` beside a link to it, so the replay rewrites the copies and never the record; `cpu_replay.of_record` checks the
copies and the bundle against each other (manifest sha256, run root, binding map), the draw is the replay's own (seeded from the
run root), and the Program is the
Build's of record (`row_dir/build_request`, its digest checked against the Commit's).  A row whose Definition has no Boolean version,
or whose Boolean version does not keep the Call's boundary (the word view of its signature is not the Call's), is not evaluated, by
name.  ``--word`` replays the same draw with the word evaluators into ``OUT/word``, the baseline.  After the prewarm the forked pool
is fitted to the cgroup's memory from each pick's bound (`BooleanRows.peak_bytes`, `driver.fit_pool`); a pick that cannot fit is
refused (exit 2).

``OUT/boolean_replay.json`` cites the Commit (Program digest, run root, binding map sha256, bundle manifest sha256, the draw's seed),
holds σ over the Program's row Definitions (word id -> Boolean id, or why not), and per replay the picks by family: picked,
evaluated, mismatched.  ``OUT/boolean/rows.jsonl`` is one line per row the Boolean evaluator was asked for (interior rows included).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import descriptor_id
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.config import option
from verity_vllm.pipeline import cpu_replay as CR
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity_vllm.program.instances_file import read_instances
from verity_vllm.query.program_view import ports_of_type

SCHEMA = "verity-vllm/boolean-replay/v1"
RECORDS = ("runs.jsonl", "verdict.json", "summary.json")
_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


@dataclass(frozen=True, kw_only=True)
class Options(CR.Options):
    out: str = option("--out", help="the directory the copies of the Commit's records, the replays and boolean_replay.json go to")
    word: bool = option("--word", action="store_true", default=False, help="also replay the same draw with the word evaluators")
    seconds_per_vu: float = option("--seconds-per-vu", type=float, default=60.0,
                                   help="the forked pool's time per picked VU (the replay's default is 3 s, for the word kernels)")


_RUN_MAX = 1 << 20  # elements converted at once (an embedding table's 28M words would take 7 GB of temporaries whole)
#: a worker's measured peak over `BooleanRows.peak_bytes`' bound, up to 1.145 on an LM head (182.1 GiB against a 159 GiB bound): the
#: allocator keeps what the worker's previous pick freed
PEAK_MARGIN = 1.25


def _runs(widths: list[int]) -> list[tuple[int, int, int, int]]:
    """(width, first element, end element, first bit) of each run of equal element widths, at most `_RUN_MAX` elements long."""
    out, lo, bit = [], 0, 0
    for j in range(1, len(widths) + 1):
        if j == len(widths) or widths[j] != widths[lo] or j - lo == _RUN_MAX:
            out.append((widths[lo], lo, j, bit))
            bit += widths[lo] * (j - lo)
            lo = j
    return out


def to_bits(words: np.ndarray, widths: list[int]) -> np.ndarray:
    """The 0/1 leaves of ``words`` (one per element of widths ``widths``), least significant bit first; a word its element cannot
    hold (a negative one as its two's complement) is refused."""
    a = np.asarray(words).reshape(-1)
    if a.size != len(widths):
        raise Declined(f"{a.size} words for {len(widths)} elements")
    out = np.empty(sum(widths), dtype=np.uint8)
    for w, lo, hi, bit in _runs(widths):
        v = a[lo:hi]
        if v.dtype.kind == "i" and v.size and int(v.min()) < -(1 << (w - 1)):
            raise Declined(f"word {int(v.min())} below a {w}-bit element")
        u = v.astype(np.int64).view(np.uint64) if v.dtype.kind == "i" else v.astype(np.uint64)
        if w < 64 and v.size and (v.dtype.kind != "i" or int(v.min()) >= 0) and int(u.max()) >> w:
            raise Declined(f"word {int(u.max())} wider than its {w}-bit element")
        out[bit:bit + w * (hi - lo)] = ((u[:, None] >> np.arange(w, dtype=np.uint64)) & np.uint64(1)).reshape(-1)
    return out


def to_words(bits: np.ndarray, widths: list[int]) -> np.ndarray:
    """``to_bits``' inverse: one uint64 word per element."""
    b = np.asarray(bits).reshape(-1).astype(np.uint64)
    out = np.empty(len(widths), dtype=np.uint64)
    for w, lo, hi, bit in _runs(widths):
        out[lo:hi] = (b[bit:bit + w * (hi - lo)].reshape(hi - lo, w) << np.arange(w, dtype=np.uint64)).sum(axis=1)
    return out


class BooleanRows:
    """The row evaluator `evaluate.ROW_EVALUATORS` takes: row ``i``'s Call (root node ``K + i`` of the Program) evaluated by the
    Boolean version of its Definition on the row's operands, its output keyed by member where the row commits one, else by port."""

    def __init__(self, nodes: list[Any], K: int, log: Path | None = None, seconds_per_vu: float = 60.0) -> None:
        self.nodes, self.K, self.log, self.seconds_per_vu = nodes, K, log, seconds_per_vu
        self.sigma: dict[str, str | None] = {}
        self.why: dict[str, str] = {}
        self._fn: dict[str, Any] = {}

    def version(self, spec: str, nd: Any = None) -> Any:
        """The Boolean version of the Call ``nd``'s Definition, checked to keep its boundary; Declined by name otherwise (``nd``
        None: a spec already looked up)."""
        if spec not in self.sigma:
            from verity.ml.definitions.boolean import NoBooleanVersion, boolean_version
            if nd is None:
                raise Declined(f"{spec} is no row of the Program")
            try:
                b = boolean_version(nd.fn)
            except NoBooleanVersion as e:
                b, self.why[spec] = None, f"no Boolean version: {e}"[:400]
            if b is not None:
                params, ret = b.signature()
                words = [BL.word_type(t) for _, t in params]
                if nd.form not in ("call", "primitive"):
                    b, self.why[spec] = None, f"a {nd.form} node, not a Call"
                elif not BL.is_boolean(b) or words != [a.type for a in nd.args] or BL.word_type(ret) != nd.out:
                    self.why[spec] = (f"{b.id} does not keep the Call's boundary: on words it takes {words} and returns "
                                      f"{BL.word_type(ret)}, the Call {[a.type for a in nd.args]} and {nd.out}")[:400]
                    b = None
            self._fn[spec], self.sigma[spec] = b, (b.id if b is not None else None)
        if self.sigma[spec] is None:
            raise Declined(self.why[spec])
        return self._fn[spec]

    def __call__(self, pi: Any, r: dict, d: dict, ops: Any) -> dict[str, np.ndarray]:
        from verity.ml.circuits.evaluation.bits import evaluate_bits

        t0, i = time.perf_counter(), int(r["i"])
        nd, line = self.nodes[self.K + i], {"i": i, "spec": r["spec"]}
        try:
            b = self.version(r["spec"], nd)
            bits = [to_bits(ops[g], BL.elements(BL.bit_type(a.type)))[None, :] for g, a in enumerate(nd.args)]
            words = to_words(np.asarray(evaluate_bits(b, *bits)).reshape(-1), BL.elements(BL.bit_type(nd.out)))
        except Declined as e:
            self._write(line | {"why": str(e)[:400]})
            raise
        members, out = pi.member_of_port(i), {}
        for p in ports_of_type(nd.out):
            out[members.get(p.name, p.name)] = words[p.lo:p.hi].astype(_UINT.get(-(-(p.width or 64) // 8) * 8, np.uint64))
        self._write(line | {"boolean": b.id, "s": round(time.perf_counter() - t0, 3)})
        return out

    def peak_bytes(self, vu: dict) -> int:
        """A bound on the bytes one pick's evaluation (`__call__`) allocates, times `PEAK_MARGIN`: the bits of every argument, with
        each conversion's element-width list and one run's temporaries (`to_bits`), the bit-sliced evaluation (`bits.peak_bytes`), and
        the output's words (`to_words`).  Its operands are views of the weights or a few committed words, and are not counted; a row
        with no Boolean version is declined before it allocates (0)."""
        from verity.ml.circuits.evaluation.bits import peak_bytes

        nd = self.nodes[self.K + int(vu["i"])]
        try:
            b = self.version(str(vu["spec"]), nd)
        except Declined:
            return 0

        def port(t: Any) -> tuple[int, int, int, int]:   # elements, bits, a run's elements, the widest element
            return t.leaves, BL.bit_type(t).leaves, min(_RUN_MAX, t.leaves), max(t.width_set() or {1})

        args = [port(a.type) for a in nd.args]
        e, n, k, w = port(nd.out)
        convert = max((16 * e_ + 8 * k_ * (1 + 2 * w_) for e_, _, k_, w_ in args), default=0)
        words = n + 8 * n + 24 * e + 8 * k * (w + 1)
        return int(PEAK_MARGIN * (sum(a[1] for a in args) + max(peak_bytes(b, 1), convert, words)))

    def prewarm(self, picked: list[dict]) -> dict[str, str]:
        """Each picked spec's Boolean version compiled in the parent, before the replay forks (`driver._prewarm_evaluators`)."""
        from verity.ml.circuits.evaluation.bits import compile as compile_bits

        out = {}
        for spec in sorted({str(vu.get("spec")) for vu in picked}):
            t0 = time.perf_counter()
            try:
                c = compile_bits(self.version(spec))
                out[f"boolean:{spec}"] = f"{c.target}: {sum(c.counts.values())} gates, compiled in {time.perf_counter() - t0:.1f} s"
            except Declined as e:
                out[f"boolean:{spec}"] = f"not compiled: {e}"[:300]
        return out

    def _write(self, line: dict) -> None:
        if self.log is None:
            return
        fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(line | {"pid": os.getpid()}) + "\n").encode())
        finally:
            os.close(fd)


def _sha256(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def scratch(bundle: Path, out: Path) -> Path:
    """``out`` holding copies of the Commit's records beside a link to ``bundle``; the link the replay opens."""
    pair = int(json.loads((bundle / "context.json").read_text())["pair"])
    out.mkdir(parents=True, exist_ok=True)
    for name in (*RECORDS, f"binding_map_p{pair}.json"):
        if (bundle.parent / name).is_file():
            shutil.copyfile(bundle.parent / name, out / name)
    link = out / bundle.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(bundle.resolve(), target_is_directory=True)
    return link


def outcome(art: dict[str, Any]) -> dict[str, Any]:
    """The replay's picks by family (picked, evaluated, mismatched) and its grade, from its `sampled_replay` record.  The uniform draw's
    one stratum names no family: a family's evaluated count is then its picks less its not-evaluated ones, known only while the record
    lists every not-evaluated pick (it lists at most 12; None past that), and its mismatches are `mismatch_by`'s, which counts them all."""
    sr = art["sampled_replay"]
    s, t, fam = sr.get("sample") or {}, sr.get("timing") or {}, {}
    for f, n in (t.get("per_family_n") or {}).items():
        fam[f] = {"picked": int(n), "evaluated": 0, "mismatch": 0}
    if s.get("strata_by") == "uniform":
        ne, wrong = sr.get("not_evaluated") or [], (sr.get("mismatch_by") or {}).get("by_family") or {}
        gone = Counter(str(x.get("spec")).split("{")[0] for x in ne) if len(ne) == int(s.get("not_evaluated_n") or 0) else None
        for f, d in fam.items():
            d["evaluated"], d["mismatch"] = None if gone is None else d["picked"] - gone[f], int(wrong.get(f, 0))
    else:
        for key, st in (sr.get("strata") or {}).items():
            f = fam.setdefault(key.split("|")[0].split("{")[0], {"picked": 0, "evaluated": 0, "mismatch": 0})
            f["evaluated"] += int(st.get("evaluated", 0))
            f["mismatch"] += int(st.get("mismatch", 0))
    return {"result": sr["result"], "grade": sr["grade"], "why": sr["why"],
            **{k: s.get(k) for k in ("seed", "seed_source", "k", "picked", "evaluated", "equal", "mismatch_n", "not_evaluated_n")},
            "by_family": fam, "not_evaluated": sr.get("not_evaluated"), "mismatch": sr.get("mismatch"),
            "wall_s": t.get("wall_s"), "workers": t.get("workers")}


def _replay(o: Options, link: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    commit_pass = CR.replay(dataclasses.replace(o, bundle=str(link)))
    pair = int(json.loads((link / "context.json").read_text())["pair"])
    art = json.loads((link.parent / f"sampled_replay_p{pair}.json").read_text())
    return outcome(art) | {"commit_pass_of_copy": commit_pass, "replay_wall_s": round(time.perf_counter() - t0, 3)}


def run(o: Options) -> dict[str, Any]:
    from verity.ml.definitions.boolean import load_program

    bundle, out = Path(o.bundle), Path(o.out)
    ctx = json.loads((bundle / "context.json").read_text())
    pair = int(ctx["pair"])
    record = {n: _sha256(bundle.parent / n) for n in (*RECORDS, f"binding_map_p{pair}.json")}
    dec, digest = load_program(Path(o.row_dir) / "build_request" / "descriptor.json.gz")
    if digest != ctx["args"].get("program_digest"):
        raise CR.Refused(f"the row dir's request Program {digest[:16]} is not the Commit's {str(ctx['args'].get('program_digest'))[:16]}")
    rows = read_instances(str(Path(o.row_dir) / "build_request" / "instances.json.gz"), expand=False)["rows"]
    nodes = dec.root.body.nodes
    K = len(nodes) - len(rows)
    bad = [r["i"] for r in rows if descriptor_id(nodes[K + int(r["i"])].fn) != r["spec"]]
    if bad:
        raise CR.Refused(f"{len(bad)} instance rows are not their Program node's Call (first: row {bad[0]})")
    keep = json.loads((bundle / CR.SLIM_DOC).read_text()) if (bundle / CR.SLIM_DOC).is_file() else None
    doc: dict[str, Any] = {"schema": SCHEMA, "commit": {"dir": str(bundle.parent.resolve()), "pair": pair, "program_digest": digest,
                                                        "run_root": ctx["run_root"], "binding_map_sha256": record[f"binding_map_p{pair}.json"],
                                                        "bundle_manifest_sha256": keep["bundle_manifest_sha256"] if keep else _sha256(bundle / "MANIFEST.json")}}
    if keep:
        doc["commit"]["keep"] = {"path": str(bundle), "dump_sha256": keep["dump_sha256"], "bytes": keep["bytes"], "picked": keep["picked"]}
    ev = BooleanRows(nodes, K, out / "boolean" / "rows.jsonl", o.seconds_per_vu)
    for spec, k in sorted({r["spec"]: K + int(r["i"]) for r in rows}.items()):
        try:
            ev.version(spec, nodes[k])
        except Declined:
            pass
    doc["sigma"], doc["sigma_gaps"] = dict(ev.sigma), dict(ev.why)
    doc["sigma_by_family"] = dict(Counter(f"{s.split('{')[0]} -> {(b or 'none').split('{')[0]}" for s, b in ev.sigma.items()))
    link = scratch(bundle, out / "boolean")
    (out / "boolean" / "rows.jsonl").unlink(missing_ok=True)
    EV.ROW_EVALUATORS.append(ev)
    try:
        doc["boolean"] = _replay(o, link)
    finally:
        EV.ROW_EVALUATORS.remove(ev)
    if o.word:
        doc["word"] = _replay(o, scratch(bundle, out / "word"))
    doc["commit"]["records_unchanged"] = record == {n: _sha256(bundle.parent / n) for n in record}
    doc["commit"]["seed"] = doc["boolean"].get("seed")
    b = doc["boolean"]
    doc["result"] = {"evaluated": b["evaluated"], "equal": b["equal"], "mismatch": b["mismatch_n"], "not_evaluated": b["not_evaluated_n"],
                     "all_equal": b["mismatch_n"] == 0 and b["evaluated"] == b["picked"], "picked": b["picked"]}
    (out / "boolean_replay.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc


def main(o: Options) -> int:
    from verity_vllm.check.replay.driver import PoolDoesNotFit

    try:
        doc = run(o)
    except (CR.Refused, PoolDoesNotFit) as e:
        print(f"[boolean-replay] REFUSED: {e}", flush=True)
        return 2
    r = doc["result"]
    print(f"[boolean-replay] {r['evaluated']}/{r['picked']} picks evaluated on bits, {r['equal']} equal to the committed outputs, "
          f"{r['mismatch']} not; {r['not_evaluated']} not evaluated; records unchanged {doc['commit']['records_unchanged']}", flush=True)
    return 0 if r["all_equal"] else 3
