"""E1 — operator-boundary comparison: every boundary row the reference interpreter produces, hashed exactly as the store hashes it
(`verity_vllm.hashing.leaf_hash` = `H("verity-vllm/leaf/v1", canonical_id, dtype, shape, raw LE bytes)`), against the committed leaf
of the RU tree (`adapter/selective/ru/<req>/<step>/index.json#entries[*].leaf`).

Two ways to use it:

* live, from `producer.py`: :class:`BoundaryIndex` maps every committed leaf of the RU to the gates of the pinned program
  (`verity_vllm.correspondence.core.CircuitMap.locate`), grouped by the root call that completes them; :meth:`BoundaryIndex.compare_root`
  hashes the interpreter's words after each root call and records agree / mismatch (first differing word and its exact bits when the
  archive carries `values.bin`; otherwise the interpreter's bits and both leaf digests).
* offline, `python -m e2e.interior.compare_boundaries --record <rec dir> --request <id> --values <producer out dir>`: re-reads the
  producer's value-record JSONL and repeats the comparison independently of the producer process (E9's independent re-check).

Nothing here is authority: the committed leaf is the store's; the interpreter's words are the reference semantics; a mismatch is a
mismatch between the two, reported by canonical id, gate, word index and bits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (REPO, REPO / "vllm-poc", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, NotInProgram  # noqa: E402
from verity_vllm.hashing import leaf_hash  # noqa: E402

DTYPE_NP = {"bfloat16": np.uint16, "float32": np.uint32, "int32": np.uint32, "float8_e4m3fn": np.uint8, "uint8": np.uint8}
DTYPE_BITS = {"bfloat16": 16, "float32": 32, "int32": 32, "float8_e4m3fn": 8, "uint8": 8}


def ru_steps(rec_dir: Path, request: str) -> list[int]:
    d = rec_dir / "adapter" / "selective" / "ru" / request
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit())


def load_ru_entries(rec_dir: Path, request: str) -> list[tuple[int, dict]]:
    """Every committed leaf entry of the RU, as ``(step, entry)``."""
    out: list[tuple[int, dict]] = []
    for step in ru_steps(rec_dir, request):
        idx = json.loads((rec_dir / "adapter" / "selective" / "ru" / request / f"{step:06d}" / "index.json").read_text())
        for e in idx["entries"]:
            out.append((step, e))
    return out


class BoundaryIndex:
    """Committed operator-boundary leaves of one RU -> gates of the pinned program, grouped by the root call whose completion
    makes all of the leaf's words available (``layout.node_at(max(gates))``: a leaf's gates are either inside that root call or
    returns of earlier root calls, which the producer retains)."""

    def __init__(self, program, rec_dir: Path, request: str, *, names: dict[str, str] | None = None) -> None:
        self.program = program
        self.layout = program.layout
        self.rec_dir = Path(rec_dir)
        self.request = request
        self.cmap = CircuitMap(program, names)
        self.by_root: dict[int, list[dict]] = defaultdict(list)
        self.not_in_program: list[dict] = []
        self.locate_errors: list[dict] = []
        self.runs: list[tuple[int, int]] = []  # committed gate runs [lo, hi) over the whole RU (for the inventory's boundary test)
        self.n_entries = 0
        self.n_words = 0
        self.values_bin: dict[int, np.memmap | None] = {}
        t0 = time.perf_counter()
        for step, e in load_ru_entries(self.rec_dir, request):
            self.n_entries += 1
            cid = e["canonical_id"]
            try:
                ident = CaptureIdentity.parse(cid)
                call, gates = self.cmap.locate(ident)
            except NotInProgram as ex:
                self.not_in_program.append({"canonical_id": cid, "step": step, "reason": str(ex)})
                continue
            except Exception as ex:  # noqa: BLE001
                self.locate_errors.append({"canonical_id": cid, "step": step, "error": f"{type(ex).__name__}: {str(ex)[:160]}"})
                continue
            g = np.asarray(gates, dtype=np.int64)
            self.n_words += int(g.size)
            k = self.layout.node_at(int(g.max()))
            self.by_root[k].append({"entry": e, "step": step, "gates": g, "call": call.id})
            breaks = np.nonzero(np.diff(g) != 1)[0]
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks + 1, [g.size]])
            for s, t in zip(starts, ends):
                self.runs.append((int(g[s]), int(g[t - 1]) + 1))
        self.runs.sort()
        self.build_s = time.perf_counter() - t0
        # results
        self.rows_compared = 0
        self.rows_agree = 0
        self.mismatches: list[dict] = []
        self.not_evaluated: list[dict] = []
        self.per_step: dict[int, dict[str, int]] = defaultdict(lambda: {"compared": 0, "agree": 0, "mismatch": 0})

    # -- committed words (only when the archive is not slim) -----------------------------------------------------------------
    def committed_words(self, step: int, e: dict) -> np.ndarray | None:
        if step not in self.values_bin:
            p = self.rec_dir / "adapter" / "selective" / "ru" / self.request / f"{step:06d}" / "values.bin"
            self.values_bin[step] = np.memmap(p, dtype=np.uint8, mode="r") if p.exists() else None
        vb = self.values_bin[step]
        if vb is None:
            return None
        raw = vb[e["offset"]:e["offset"] + e["nbytes"]]
        return np.frombuffer(raw.tobytes(), dtype=DTYPE_NP[e["dtype"]])

    # -- one leaf --------------------------------------------------------------------------------------------------------------
    def compare_leaf(self, item: dict, words: np.ndarray) -> dict:
        e, step = item["entry"], item["step"]
        cid, dtype, shape = e["canonical_id"], e["dtype"], e["shape"]
        raw = np.ascontiguousarray(words.astype(DTYPE_NP[dtype])).tobytes()
        lh = leaf_hash(cid, dtype, shape, raw).hex()
        agree = lh == e["leaf"]
        rec = {"canonical_id": cid, "step": step, "call": item["call"], "words": int(words.size), "dtype": dtype, "shape": shape,
               "gate_lo": int(item["gates"].min()), "gate_hi": int(item["gates"].max()) + 1,
               "leaf_interpreter": lh, "leaf_committed": e["leaf"], "agrees": agree}
        if not agree:
            cw = self.committed_words(step, e)
            if cw is not None and cw.size == words.size:
                diff = np.nonzero(cw.astype(np.uint64) != words.astype(np.uint64))[0]
                if diff.size:
                    j = int(diff[0])
                    nb = DTYPE_BITS[dtype] // 8
                    rec["first_differing_word"] = {"word": j, "gate": int(item["gates"][j]),
                                                   "interpreter_bits": f"0x{int(words[j]):0{2 * nb}x}", "committed_bits": f"0x{int(cw[j]):0{2 * nb}x}",
                                                   "differing_words": int(diff.size)}
                else:
                    rec["note"] = "every word equal but the leaf digest differs: canonical id / dtype / shape framing differs"
            else:
                nb = DTYPE_BITS[dtype] // 8
                rec["committed_words"] = "UNAVAILABLE (slim archive: index.json only, no values.bin)"
                rec["interpreter_words_hex_le"] = [f"{int(w):0{2 * nb}x}" for w in words[:64].astype(np.uint64)]
                rec["interpreter_words_truncated_to"] = min(64, int(words.size))
        return rec

    def compare_root(self, k: int, transcript, *, on_mismatch=None) -> list[dict]:
        """Compare every committed leaf completed by root node ``k``; ``transcript[g]`` must give the interpreter's word of gate g."""
        out: list[dict] = []
        for item in self.by_root.get(k, ()):
            g = item["gates"]
            try:
                words = np.fromiter((transcript[int(x)] for x in g), dtype=np.uint64, count=g.size)
            except KeyError as ex:
                self.not_evaluated.append({"canonical_id": item["entry"]["canonical_id"], "step": item["step"], "missing_gate": int(ex.args[0])})
                continue
            rec = self.compare_leaf(item, words)
            self.rows_compared += 1
            ps = self.per_step[item["step"]]
            ps["compared"] += 1
            if rec["agrees"]:
                self.rows_agree += 1
                ps["agree"] += 1
            else:
                ps["mismatch"] += 1
                self.mismatches.append(rec)
                if on_mismatch is not None:
                    on_mismatch(rec)
            out.append(rec)
        return out

    # -- summary ---------------------------------------------------------------------------------------------------------------
    def pending(self) -> int:
        return sum(len(v) for v in self.by_root.values()) - self.rows_compared - len(self.not_evaluated)

    def summary(self, *, label: str = "MEASURED") -> dict[str, Any]:
        first = self.mismatches[0] if self.mismatches else None
        verdict = ("NOT DONE: no boundary row compared yet" if self.rows_compared == 0 else
                   (f"all {self.rows_compared} boundary rows compared agree" + ("" if self.pending() == 0 else f" ({self.pending()} committed leaves not yet reached)")
                    if not self.mismatches else
                    f"first mismatch at {first['canonical_id']} (gates [{first['gate_lo']},{first['gate_hi']}))"))
        return {"schema": "veritor/e1/boundary-compare/v1", "label": label, "request_id": self.request,
                "committed_leaves": self.n_entries, "committed_words": self.n_words, "located_leaves": sum(len(v) for v in self.by_root.values()),
                "not_in_program": self.not_in_program, "locate_errors": self.locate_errors,
                "rows_compared": self.rows_compared, "rows_agree": self.rows_agree, "rows_mismatch": len(self.mismatches),
                "rows_not_evaluated": len(self.not_evaluated), "rows_pending": self.pending(),
                "per_step": {str(k): v for k, v in sorted(self.per_step.items())},
                "verdict": verdict, "first_mismatch": first, "mismatches": self.mismatches[:64], "not_evaluated": self.not_evaluated[:64],
                "hash": "verity_vllm.hashing.leaf_hash: sha256(u32be(len(tag))||tag||(u64be(len(part))||part)*) over "
                        "(canonical_id, dtype, shape text, raw LE words), tag verity-vllm/leaf/v1 -- the store's leaf function, imported, not reimplemented",
                "index_build_s": round(self.build_s, 3)}


# ---------------------------------------------------------------------------------------------------------------------------------
# offline: re-read the producer's JSONL and compare
# ---------------------------------------------------------------------------------------------------------------------------------


class JsonlTranscript:
    """``transcript[g]`` over the producer's value-record files (loaded file by file, on demand, by gate interval)."""

    def __init__(self, values_dir: Path, inventory: dict) -> None:
        self.values_dir = Path(values_dir)
        self.files = sorted(inventory["files"], key=lambda f: f["lo"])
        self.los = [f["lo"] for f in self.files]
        self.cache: dict[int, dict[int, int]] = {}
        self.retained: dict[int, int] = {}

    def _load(self, i: int) -> dict[int, int]:
        d = self.cache.get(i)
        if d is None:
            d = {}
            p = self.values_dir / self.files[i]["path"]
            opener = __import__("gzip").open if p.suffix == ".gz" else open
            with opener(p, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    d[r["g"]] = int.from_bytes(bytes.fromhex(r["v"]), "little")
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = d
        return d

    def __getitem__(self, g: int) -> int:
        import bisect
        i = bisect.bisect_right(self.los, g) - 1
        if i < 0 or not (self.files[i]["lo"] <= g < self.files[i]["hi"]):
            raise KeyError(g)
        return self._load(i)[g]


def by_layer_kind(records: list[dict]) -> dict[str, dict[str, int]]:
    """[E1 night 2026-09-12] agree / mismatch counts per (layer, boundary kind) -- the canonical id's ``l=`` field and the operator kind =
    the last component of its ``op=`` (``rmsnorm``, ``gemm``, ``rope``, ``attn``, ``act_mul``, ``embedding``, ``argmax``…) with its ``port``;
    'the numerics are the finding' wants the breakdown, not one verdict.  Keys sort by layer (embed first, then 0..NL-1, then norm / lm_head / sampler)."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"compared": 0, "agree": 0, "mismatch": 0})
    for r in records:
        f = dict(p.split("=", 1) for p in r["canonical_id"].split("/") if "=" in p)
        op = f.get("op", "?")
        key = f"l={f.get('l', '?')}/{op.rsplit('.', 1)[-1]}/port={f.get('port', '?')}"
        d = out[key]
        d["compared"] += 1
        d["agree" if r["agrees"] else "mismatch"] += 1
    order = {"embed": -1, "norm": 10**6, "lm_head": 10**6 + 1, "sampler": 10**6 + 2}
    def lk(kv: tuple) -> tuple:
        l = kv[0].split("/")[0][2:]
        return (order.get(l, int(l) if l.isdigit() else 10**7), kv[0])
    return dict(sorted(out.items(), key=lk))


def by_kind(records: list[dict]) -> dict[str, dict[str, int]]:
    """The same counts summed over layers, per operator kind (the per-kind verdict the brief asks for)."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"compared": 0, "agree": 0, "mismatch": 0})
    for r in records:
        f = dict(p.split("=", 1) for p in r["canonical_id"].split("/") if "=" in p)
        d = out[f"{f.get('op', '?').rsplit('.', 1)[-1]}/port={f.get('port', '?')}"]
        d["compared"] += 1
        d["agree" if r["agrees"] else "mismatch"] += 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="offline boundary comparison over a producer output directory")
    ap.add_argument("--record", type=Path, required=True, help="record dir holding adapter/selective/ru/<request>/… (and protocol_instance.json unless --declared-case)")
    ap.add_argument("--request", required=True)
    ap.add_argument("--values", type=Path, required=True, help="producer --out dir (holds inventory.json or files_so_far.json, and interior/…)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--declared-case", default=None, help="[E1 night] program from the registry case (b1.<case>_CONFIG) + --lp/--steps, as the producer's declared-inputs mode built it "
                                                        "(a diagnostic capture has no protocol_instance.json)")
    ap.add_argument("--lp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--attention", default=None, help="[E7 D57] declared attention kernel geometry JSON: with --declared-case -> Serve_v4; with a record -> DIAGNOSTIC override of the pinned identity (as producer.py --attention)")
    a = ap.parse_args(argv)
    if a.declared_case:
        from verity.ir import bind
        from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
        from verity_vllm.registry import b1
        geom = json.loads(a.attention) if getattr(a, "attention", None) else None  # [E7 D57]
        fam = "Serve_v4" if geom is not None else b1.DEFAULT_SERVE_FAMILY
        program = LazyProgram(b1.serve_program(dict(getattr(b1, f"{a.declared_case}_CONFIG")), a.lp, a.steps, family=fam, attention=geom))
        inst = {"digest": lazy_program_digest(program), "identity": {"family": fam, "case": a.declared_case, "LP": a.lp, "STEPS": a.steps, **({"attention": geom} if geom else {})}}
        dp = a.values / "declared_program.json"
        if dp.exists() and json.loads(dp.read_text())["digest"] != inst["digest"]:
            print("PROBLEMS", [f"the rows were produced for program digest {json.loads(dp.read_text())['digest']}, this tree binds {inst['digest']}"])
            return 2
    else:
        from e2e.verifier_v3 import locator as vloc
        inst = json.loads((a.record / "protocol_instance.json").read_text())["program"]
        if getattr(a, "attention", None):  # [E7 D57] DIAGNOSTIC geometry override (see producer.py --attention)
            ident = {**inst["identity"], "family": "Serve_v4", "attention": json.loads(a.attention)}
            ident.pop("function", None)
            print("GEOMETRY OVERRIDE (DIAGNOSTIC: not the record's pinned program)", json.dumps(ident["attention"], sort_keys=True))
            inst = {"identity": ident, "digest": None, "gates": None}
        program, C, problems = vloc.build_pinned_program(inst["identity"], expected_digest=inst.get("digest"), expected_gates=inst.get("gates"), lazy=True)
        if problems:
            print("PROBLEMS", problems)
            return 2
    inv_p, fsf_p = a.values / "inventory.json", a.values / "files_so_far.json"
    inv = json.loads(inv_p.read_text()) if inv_p.exists() else {"files": json.loads(fsf_p.read_text()), "source": "files_so_far.json (run in progress or declared-inputs mode)"}
    bi = BoundaryIndex(program, a.record, a.request)
    tr = JsonlTranscript(a.values, inv)
    done_roots = {f["k"] for f in inv["files"]}
    recs: list[dict] = []
    for k in sorted(bi.by_root):
        if k in done_roots:
            recs.extend(bi.compare_root(k, tr))
    s = bi.summary()
    s["program"] = inst
    s["values_files"] = {"count": len(inv["files"]), "source": inv.get("source", "inventory.json")}
    s["by_layer_kind"] = by_layer_kind(recs)
    s["by_kind"] = by_kind(recs)
    print(json.dumps({k: v for k, v in s.items() if k not in ("mismatches", "not_in_program", "not_evaluated", "by_layer_kind", "by_kind")}, indent=1))
    print("by kind (all layers):")
    for key, d in s["by_kind"].items():
        print(f"  {key:40s} compared {d['compared']:5d} agree {d['agree']:5d} mismatch {d['mismatch']:5d}")
    print("by layer x kind:")
    for key, d in s["by_layer_kind"].items():
        print(f"  {key:40s} compared {d['compared']:5d} agree {d['agree']:5d} mismatch {d['mismatch']:5d}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(s, indent=1))
    return 0 if not s["rows_mismatch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
