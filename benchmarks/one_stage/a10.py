"""Run A10: A11's served-step kinds and A9's routed-expert launches (GLM-5.3 TP8 B1, rank 0) on the values the Commit
committed, in place of the Match capture's stand-ins, with every value where the two differ recorded.

    python benchmarks/one_stage/a10.py values --values DIR --bundle DIR --commit DIR --build DIR --manifest FILE \\
      --roles derive-report.json --kinds kinds.json --out OUT [--try KINDS] [a11.py's --steps --layers --per-kind ...]
    python benchmarks/one_stage/a10.py audit <a11.py's arguments, with --load-values OUT/saved-a11 or OUT/saved-a9>

* **The record.** The Commit r20261019-074856-7c3a ran with ``--replay-deferred`` and ``retain=host``. Each rank's bundle
  holds every byte that rank committed. Rank 0's bundle (``--bundle``) is opened as the deferred replay opens it
  (`commit_replay._of_record`): its manifest, the rank's run root in ``runs.jsonl`` (``--commit``, the Commit's records)
  and its combination into the TP run root, the store's step roots, the binding map. Its Program of record (``--build``,
  the Build's rank 0 dir) is loaded by `rank_replay.load`, which refuses a digest not of record. Its committed members
  are read through the binding map and the required manifest's aliases (``--manifest``, held to the sha256 the bundle
  names), and each read is verified against the run root. Its weights are the slice's checkpoint tensors (``ckpt/``)
  opened against the committed weights root (`cpu_replay.open_weights`): rank 0 of 8 composed from the whole tensors,
  and layer 3's routed experts from `a11_slice.ckpt_experts`' share at world 1. The Commit's verdict is PENDING until its
  deferred replay decides it, and these are rank 0's values only.
* **Calls to rows.** A11's fold of the Match capture (``--values``, as `a9.py` folds it) gives the Calls the kinds are
  picked from. Each step's Calls of a Definition family are paired in order with that step's rows of the family: the fold
  and the Program order a step's families differently (the fold emits a layer's MqaLogits before its TopKs, the Program
  interleaves them per token), but each family's Calls alike. A pair aligns only on the same layer, spec and
  request-local row (the token, for a row named by token), and a family whose counts differ at a step aligns nothing. A
  Call is read only through its aligned row.
* **Values.** A Call's operands are its row's, resolved as the replay resolves them (`evaluate._evaluate_row`): the
  committed member words, the opened weights, the prompt, and any interior (uncommitted) producer evaluated from its own
  committed inputs, which ``compare.json`` names by operand. Its returns are held to the members the row commits
  (`member_of_port`); a return no member holds is interior and named. `a11.operands`, `a11.call_twin` and
  `a9.capture_words` read these in place of the capture, so `a11.py`'s and `a9.py`'s own loops pick, evaluate and stage
  the same kinds and units into ``OUT/saved-a11`` and ``OUT/saved-a9``. Every operand and return is compared word by word
  with the capture's (``OUT/compare.json``).
* **Audit.** ``audit`` is `a11.py` with the registration window naming the committed values. Run it with ``--load-values``
  on each saved directory, as a10.sh does.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a9  # noqa: E402
import a11  # noqa: E402
import a11_kinds as K  # noqa: E402

COMMIT = "r20261019-074856-7c3a"
REPLAY = "r20261019-110314-3fd7"
PENDING = (f"the Commit's verdict is PENDING (the deferred CPU replay {REPLAY} decides it; its attempt 1 r20261019-091406-48bd was "
           "cancelled); rank 0's values only")
WINDOW_OF = (f"GLM-5.3 TP8 B1 rank 0's committed step values: row {a11.ROW}'s Commit {COMMIT} (--replay-deferred, retain=host), "
             "its bundle opened against the rank's run root of record and its weights against the committed weights root; " + PENDING)
LINK = ("provenance, not proof: every value was opened from the Commit's bundle and verified against rank 0's run root of record "
        "(each weight tensor against its committed tensor root) before staging, but the units' roots are a fresh commitment to "
        "those values, so their link to the served root is that opening, not a proof inside the audit; " + PENDING)
EXPERTS = re.compile(r"\.mlp\.experts\.")
F64 = re.compile(r'\{\s*"f64"\s*:\s*"(0x[^"]+)"\s*\}')
LAYER = re.compile(r"layers\.(\d+)")
TOKEN = re.compile(r"\[t=(\d+)")

A11_FOLD, A11_OPERANDS, A11_SAVE, A9_CAPTURE = a11.fold_values, a11.operands, a11.save_values, a9.capture_words


def unsigned(w) -> np.ndarray:
    """Words as `a11.operands` returns them: each leaf's bits, unsigned, in u64."""
    w = np.asarray(w).reshape(-1)
    if w.dtype.kind in "ifb":
        w = w.view(f"u{w.dtype.itemsize}")
    return w.astype(np.uint64)


def norm_spec(s) -> str:
    """A spec label and a Definition id alike (``DOT={"fn":"X"}`` vs ``DOT=X``, ``{"f64":"0x1.4p+1"}`` vs ``2.5``, ``'`` vs
    ``"``, ``[]`` vs ``()``, ``true`` vs ``True``)."""
    s = F64.sub(lambda m: repr(float.fromhex(m[1])), str(s))
    s = re.sub(r"""[{}"'\s]""", "", s).replace("fn:", "")
    return s.replace("[", "(").replace("]", ")").lower()


def layer_of(*names) -> int | None:
    for s in names:
        m = LAYER.search(str(s or ""))
        if m:
            return int(m[1])
    return None


def diff(a: np.ndarray | None, b: np.ndarray | None) -> dict:
    """``a`` (committed) against ``b`` (the capture's), word by word."""
    if a is None or b is None:
        return {"compared": False, "committed_n": None if a is None else int(a.size), "capture_n": None if b is None else int(b.size)}
    if a.size != b.size:
        return {"compared": False, "committed_n": int(a.size), "capture_n": int(b.size)}
    bad = np.flatnonzero(a != b)
    return {"compared": True, "n": int(a.size), "differ": int(bad.size),
            "first": [[int(i), f"{int(a[i]):x}", f"{int(b[i]):x}"] for i in bad[:4]]}


class Committed:
    """Rank 0's committed values (module docstring)."""

    def __init__(self, a) -> None:
        from verity_vllm.check import oracle_compare as OC
        from verity_vllm.check.checkpoint_index import CheckpointIndex
        from verity_vllm.check.replay.opening import CommittedStore
        from verity_vllm.pipeline import cpu_replay as CPU, replay_bundle as RB
        from verity_vllm.pipeline.tp import commit_replay as CR, rank_replay as RR
        t0 = time.time()
        commit = Path(a.commit)
        b = RB.open_bundle(a.bundle)
        ctx, bdir = b["context"], Path(b["path"])
        dd = json.loads((commit / "replay_deferred.json").read_text())
        _row, c, store, run, bmap = CR._of_record(commit, dd, b, 0, 0)
        cfg = dict(ctx["replay"], programs={"0": a.build})
        p = RR.load(cfg, 0, int(dd["world"]))
        sha = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
        if sha != ctx.get("required_manifest_sha256"):
            raise SystemExit(f"{a.manifest}: sha256 {sha[:16]} is not the Commit's required manifest {str(ctx.get('required_manifest_sha256'))[:16]}")
        self.reader = OC.OpenedReader(store, run)
        self.store = CommittedStore.of_record(bmap, self.reader, RR.manifest_of_rank(json.loads(Path(a.manifest).read_text()), 0))
        del bmap
        tree = json.loads((bdir / "weights" / "tree.json").read_text())
        root = (c.get("weight_registration") or {}).get("root")
        ck = CheckpointIndex(str(Path(a.values) / "ckpt"), verify_shards=False)
        dense, experts = copy.copy(ck), copy.copy(ck)
        dense.tensors = {n: t for n, t in ck.tensors.items() if not EXPERTS.search(n)}
        experts.tensors = {n: t for n, t in ck.tensors.items() if EXPERTS.search(n)}
        words, wd = CPU.open_weights(bdir, tree, root, dense, threads=a.threads, tp={"world": int(dd["world"]), "rank": 0})
        wd["ue8m0"] = self.rewritten(tree, dense, words, Path(a.roles).parent, p.index.prefix, {"world": int(dd["world"]), "rank": 0})
        ew, we = CPU.open_weights(bdir, tree, root, experts, threads=a.threads, tp={"world": 1, "rank": 0})
        words.update({n: w for n, w in ew.items() if EXPERTS.search(n)})
        self.consumed: set[str] = set()
        self.weights = CPU.provider(words, p.index.prefix, self.consumed)
        self.pi, self.rid = p.index, next(iter(p.progs))
        self.arrive = int(p.arrivals.get(self.rid, 0))
        summary = json.loads((commit / "summary.json").read_text()) if (commit / "summary.json").exists() else {}
        self.record = {
            "commit": COMMIT, "verdict": PENDING, "summary_pass": summary.get("pass"), "commit_pass": c.get("commit_pass"),
            "bundle": {"path": b["path"], "manifest_sha256": b["manifest_sha256"], "code": ctx.get("code")},
            "run_root": c.get("run_root"), "tp_run_root": ctx.get("tp_run_root"), "binding_map_digest": (c.get("binding") or {}).get("map_digest"),
            "program": {"dir": next(iter(p.dirs.values())), "digest": p.digs[self.rid], "request": self.rid, "arrive": self.arrive,
                        "prefix": p.index.prefix},
            "required_manifest_sha256": sha,
            "weights": {"root": root, "dense": {k: wd[k] for k in ("equal", "why", "opened", "servable", "mismatched_n", "mismatched", "unopened_n", "ue8m0")},
                        "experts": {k: we[k] for k in ("equal", "why", "opened", "mismatched_n", "mismatched")},
                        "opened": len(words)},
            "open_s": round(time.time() - t0, 1)}
        self.res = None
        self.calls: dict[str, dict] = {}
        self._ops: dict[int, list[np.ndarray]] = {}    # the last Call's only: one w13 operand is GBs
        self._outs: dict[int, tuple] = {}
        u = wd["ue8m0"]
        print(f"A10 committed: bundle {b['manifest_sha256'][:16]} of record, Program {p.digs[self.rid][:16]}, weights dense "
              f"{wd['opened']}/{wd['servable']} opened ({wd['mismatched_n']} mismatched), {u['opened']} more through vLLM's load-time "
              f"rewrite ({len(u['mismatched'])} composed that way did not open), experts {we['opened']} opened "
              f"({we['mismatched_n']} mismatched), {self.record['open_s']} s; {PENDING}", flush=True)

    @staticmethod
    def rewritten(tree: dict, ck, words: dict, program_dir: Path, prefix: str, tp: dict) -> dict:
        """The registered tensors the checkpoint's own words do not open that vLLM rewrites at load (the Build's `weight_roles`:
        DeepGEMM's ue8m0 requant, MLA's kv_b absorbed into W_UK_T/W_UV), composed through `fp8_ue8m0.Rewrites` as the deferred
        replay's attempt 2 composes them, each opened only against its committed tensor root; added to ``words``."""
        from verity_vllm.check import fp8_ue8m0
        from verity_vllm.check import weights_of_record as WR
        from verity_vllm.pipeline import cpu_replay as CPU
        roles = {(f[len(prefix):] if prefix and f.startswith(prefix) else f): r for f, r in fp8_ue8m0.build_roles(str(program_dir)).items()}
        names = tree["names"]
        dims = {n: (list(s), CPU.WORDS[dt][0]) for n, dt, s, _b in names if dt in CPU.WORDS}
        rw = fp8_ue8m0.Rewrites(list(dims), roles, lambda n: WR.compose(n, *dims[n], ck, **tp), ck.config)
        out = {"roles": len(roles), "covered": 0, "opened": 0, "mismatched": [], "unopened": 0}
        for i, (n, dt, _s, nb) in enumerate(names):
            if n in words or n not in dims or not rw.covers(n):
                continue
            out["covered"] += 1
            try:
                arr, _note = rw.compose(n, dims[n][0])
            except (KeyError, ValueError):
                out["unopened"] += 1
                continue
            raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
            if raw.size != int(nb) or CPU.tensor_root_of(tree, i, raw) != bytes.fromhex(tree["tensor_roots"][i]):
                out["mismatched"].append(n)
                continue
            words[n] = raw.view(CPU.WORDS[dt][1])
            out["opened"] += 1
        return out

    # ---- the fold's Calls aligned with the Program's rows
    def align(self, res) -> None:
        if self.res is res:
            return
        t0 = time.time()
        self.res, self.rows = res, {}
        rows, calls, nrows = defaultdict(list), defaultdict(list), Counter()
        for r in self.pi.rows:
            s = self.arrive + self.pi.step_of(int(r["i"]))
            rows[(s, K.family(r["spec"]))].append(r)
            nrows[s] += 1
        for m in res.instances:
            calls[(m["step"], K.family(m["spec"]))].append(m)
        stats: dict[str, dict] = {}
        for (s, f), ms in sorted(calls.items()):
            rs = rows.get((s, f), [])
            st = stats.setdefault(str(s), {"calls": 0, "rows": nrows[s], "aligned": 0, "refused": Counter(), "first": {}})
            st["calls"] += len(ms)
            if len(ms) != len(rs):
                st["refused"]["count"] += len(ms)
                st["first"].setdefault(f"count {f}", [len(ms), len(rs)])
                continue
            for m, r in zip(ms, rs):
                why = self.mismatch(m, r)
                if why:
                    st["refused"][why] += 1
                    st["first"].setdefault(f"{why} {f}", [m["name"], str(m["spec"])[:200], r.get("name"), str(r["spec"])[:200]])
                    continue
                self.rows[id(m)] = r
                st["aligned"] += 1
        self.record["alignment"] = {"steps": stats, "s": round(time.time() - t0, 1)}
        print("A10 alignment: " + json.dumps(self.record["alignment"], default=str)[:4000], flush=True)

    def mismatch(self, m: dict, r: dict) -> str | None:
        """Why the Call ``m`` is not row ``r``'s, if it is not: its layer, its spec, or its request-local row."""
        d = self.pi.describe(r) or {}
        if layer_of(m.get("hint"), m["name"]) != layer_of(d.get("module"), r.get("name")):
            return "layer"
        if norm_spec(m["spec"]) != norm_spec(r["spec"]):
            return "spec"
        t = TOKEN.search(str(r.get("name") or ""))
        if m.get("row") != d.get("row") and (t is None or int(t[1]) != m.get("position")):
            return "row"
        return None

    def row_of(self, m: dict) -> dict:
        r = self.rows.get(id(m))
        if r is None:
            raise LookupError(f"{m['name']}: no Program row of record aligned with this Call (compare.json's record.alignment)")
        return r

    def entry(self, m: dict) -> dict:
        key = f"{m['name']}@{m['step']}#{m['event']}"
        r = self.rows.get(id(m))
        return self.calls.setdefault(key, {"kind": K.family(m["spec"]), "name": m["name"], "step": m["step"], "event": m["event"],
                                           "node": m["node"], "row": None if r is None else int(r["i"])})

    # ---- operands
    def sources(self, r: dict) -> list[list[str]]:
        """Each operand's sources: committed members, weights, the prompt or another parameter, interior producers, constants."""
        from verity_vllm.check.replay import evaluate as EV
        from verity_vllm.check.replay.index import const_word, row_args
        out = []
        for stored in r.get("args") or []:
            if EV._weight_runs(self.pi, stored) is not None:
                out.append(["weights"])
                continue
            (group,) = row_args({"args": [stored]})
            kinds = set()
            for sl in group:
                kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
                if kind == "p":
                    kinds.add(self.pi.pnames[a] if a < len(self.pi.pnames) else f"p{a}")
                elif self.pi.member_at(a, lo, hi) is not None:
                    kinds.add("committed")
                else:
                    pr = self.pi.by_i.get(a)
                    kinds.add("const" if const_word(pr) is not None else f"interior:{K.family(str((pr or {}).get('spec')))}")
            out.append(sorted(kinds))
        return out

    def operands(self, res, store, vals, m: dict) -> list[np.ndarray]:
        """The Call's operands from its row of record (module docstring), each compared with the capture's."""
        from verity_vllm.check.replay import evaluate as EV
        from verity_vllm.commit.opened import OpeningNotVerified
        if id(m) in self._ops:
            return self._ops[id(m)]
        self.align(res)
        e = self.entry(m)
        r = self.row_of(m)
        got: list = []

        def hook(pi, row, d, ops):
            if row is r:
                got.append([ops[g] for g in range(len(ops))])
                return {}
            if d["family"] in EV.DELEGATES:
                raise EV._Unresolved(f"interior row {row.get('i')} ({d['family']}) is a delegated family, which A10 does not evaluate")
            return EV.evaluate(d["family"], d["statics"], ops.inputs, ops.weights)

        EV.ROW_EVALUATORS.append(hook)
        try:
            EV._evaluate_row(self.pi, self.store, self.weights, r, self.rid, self.arrive, [], depth=0, interior={})
        except EV._Unresolved as ex:
            e["operands_why"] = str(ex)[:400]
            raise LookupError(f"committed operands of row {r['i']}: {ex}") from None
        except OpeningNotVerified as ex:
            e["operands_why"] = f"OPENING NOT VERIFIED: {ex}"[:400]
            raise LookupError(e["operands_why"]) from None
        finally:
            EV.ROW_EVALUATORS.remove(hook)
        ops = [unsigned(w) for w in got[0]]
        nargs = len(res.body.nodes[m["node"]].args)
        if len(ops) != nargs:
            e["operands_why"] = f"row {r['i']} has {len(ops)} operands, the Call {nargs}"
            raise LookupError(e["operands_why"])
        e["sources"] = self.sources(r)
        try:
            cap = A11_OPERANDS(res, store, vals, m)
        except LookupError as ex:
            cap, e["capture_why"] = None, str(ex)[:300]
        e["operands"] = [diff(x, None if cap is None else cap[k]) for k, x in enumerate(ops)]
        if cap is not None and any(d.get("differ") for d in e["operands"]):
            try:
                alt = A11_OPERANDS(res, store, vals, m, first="provenance")
                e["operands_provenance"] = [diff(x, alt[k]) for k, x in enumerate(ops)]
            except LookupError as ex:
                e["operands_provenance"] = str(ex)[:300]
        self._ops = {id(m): ops}
        return ops

    # ---- returns
    def outputs(self, m: dict) -> tuple[dict[int, tuple[str, np.ndarray]], list[int], list[tuple[int, int]]]:
        """The Call's return components the row commits ({component: (member, words)}), those it does not, and the ranges."""
        from verity_vllm.check.fold_compare import _component_ranges
        from verity_vllm.check.replay import evaluate as EV
        if id(m) in self._outs:
            return self._outs[id(m)]
        r = self.row_of(m)
        d = self.pi.describe(r)
        if d is None:
            raise LookupError(f"row {r['i']} is unmodelled")
        comps = _component_ranges(m)
        ports = self.pi.member_of_port(int(r["i"])) or EV.plane_ports(d)
        members = {str(x[0]): int(x[1]) for x in d["members"]}
        outs, interior = {}, []
        for k, (clo, chi) in enumerate(comps):
            member = ports.get(str(k))
            # a one-output relation's return is its one member whatever its port is named, as the replay holds it
            if member is None and len(members) == 1 and (len(comps) == 1 or (not ports and k == len(comps) - 1)):
                member = next(iter(members))
            if member not in members:
                interior.append(k)
                continue
            width = members[member]
            if width != chi - clo:
                raise LookupError(f"row {r['i']} member {member}: {width} words a row, return component {k} has {chi - clo}")
            w, why = self.store.words(self.rid, self.arrive + d["step"], d["op_path"], member, d["row"] * width, (d["row"] + 1) * width)
            if w is None:
                raise LookupError(f"row {r['i']} member {member}: {why}")
            outs[k] = (member, unsigned(w))
        self._outs = {id(m): (outs, interior, comps)}
        return outs, interior, comps

    @staticmethod
    def capture_out(store, m: dict, k: int, comps: list, w: np.ndarray, writes) -> dict:
        """Return component ``k`` as committed against the capture's snapshot of its view at the Call, and when they differ, the
        first later snapshot before another writer that equals it (`a11.call_twin`'s ``late``)."""
        views = m["output_views"]
        first = len(comps) - len(views)
        view = views[k - first] if 0 <= k - first < len(views) else None
        if view is None or view.numel != w.size:
            return {"compared": False, "why": "no output view of this component's size"}
        s = store.find(view, at=m["event"])
        d = diff(w, unsigned(store.extract(s, view))) if s is not None else {"compared": False, "why": "no snapshot at the Call"}
        if d.get("differ") and writes is not None:
            s2 = next((x for x in a11.later_snapshots(store, view, m["event"], writes)
                       if not diff(w, unsigned(store.extract(x, view))).get("differ")), None)
            if s2 is not None:
                d["late"] = s2.event
        return d

    def call_twin(self, store, m: dict, got: list, writes=None) -> dict:
        """`a11.call_twin` against the committed members; the capture's output snapshots against those members too."""
        e = self.entry(m)
        try:
            outs, interior, comps = self.outputs(m)
        except LookupError as ex:
            e["outputs_why"] = str(ex)[:400]
            return {"compared": 0, "agree": False, "why": f"committed outputs: {ex}"[:400]}
        compared, bad = 0, []
        for k, (member, w) in outs.items():
            clo, chi = comps[k]
            idx = [i for i in range(clo, chi) if got[i] is not None]
            bad += [i for i in idx if int(got[i]) != int(w[i - clo])]
            compared += len(idx)
        e["outputs"] = {"members": {str(k): member for k, (member, _w) in outs.items()}, "interior_components": interior,
                        "evaluated_vs_committed": {"compared": compared, "differ": len(bad), "first": bad[:4]},
                        "capture_vs_committed": {str(k): self.capture_out(store, m, k, comps, w, writes) for k, (_mb, w) in outs.items()}}
        if not outs:
            return {"compared": 0, "agree": None, "why": "no return component is a committed member", "interior_components": interior}
        return {"compared": compared, "of": len(got), "components": sorted(outs), "of_components": len(comps),
                "agree": (not bad) if compared else None, "first_diff": bad[:4], "source": "committed",
                **({"interior_components": interior} if interior else {})}

    def capture_words(self, store, m: dict, late) -> np.ndarray:
        """`a9.capture_words`: the Call's one output member as committed, compared with the capture's snapshot."""
        outs, _interior, comps = self.outputs(m)
        k = len(comps) - 1
        if k not in outs:
            raise LookupError(f"{m['name']}: its output is not a committed member")
        w = outs[k][1]
        try:
            self.entry(m)["a9_capture"] = diff(w, np.asarray(A9_CAPTURE(store, m, late), dtype=np.uint64))
        except (LookupError, TypeError, AttributeError) as ex:
            self.entry(m)["a9_capture"] = {"compared": False, "why": f"{type(ex).__name__}: {ex}"[:300]}
        return w


def install(C: Committed) -> None:
    """A11's and A9's value sources read the committed values; one fold for both, with A9's expert-share weights."""
    fold: list = []

    def fold_values(vdir, profile_name, roles, last_step=None, weights_cls=None):
        if not fold:
            fold.append(A11_FOLD(vdir, profile_name, roles, last_step, weights_cls=a9.ShareWeights))
            C.align(fold[0][0])
        return fold[0]

    def save_values(d, staged, rows_of_kind, weights_notes, a_):
        for s in staged.values():
            s["values_of"] = WINDOW_OF
            for t in (s.get("twin") or {}).get("each") or []:
                t["operands"] = "committed"
        A11_SAVE(d, staged, rows_of_kind, {"committed": C.record["weights"], "stand_in_fold": weights_notes}, a_)

    a11.fold_values = fold_values
    a11.operands = lambda res, store, vals, m, first="snapshot": C.operands(res, store, vals, m)
    a11.call_twin = C.call_twin
    a11.save_values = save_values
    a9.capture_words = C.capture_words


def summarize(C: Committed) -> dict:
    """Per kind: the Calls read, how many operands and returns differ from the capture's, and why a Call was not read."""
    per: dict[str, dict] = {}
    for e in C.calls.values():
        k = per.setdefault(e["kind"], {"calls": 0, "operands_read": 0, "operand_words": 0, "operand_words_differ": 0, "operands_differ": 0,
                                       "outputs_read": 0, "output_words_differ_capture": 0, "evaluated_differ_committed": 0,
                                       "interior_operands": Counter(), "why": []})
        k["calls"] += 1
        for d, src in zip(e.get("operands") or [], e.get("sources") or []):
            k["operands_read"] += 1
            k["operand_words"] += d.get("n", 0)
            k["operand_words_differ"] += d.get("differ", 0)
            k["operands_differ"] += bool(d.get("differ"))
            for s in src:
                if s.startswith("interior:"):
                    k["interior_operands"][s] += 1
        o = e.get("outputs")
        if o:
            k["outputs_read"] += 1
            k["evaluated_differ_committed"] += o["evaluated_vs_committed"]["differ"]
            k["output_words_differ_capture"] += sum(c.get("differ", 0) for c in o["capture_vs_committed"].values())
        for x in ("operands_why", "outputs_why", "capture_why"):
            if e.get(x) and len(k["why"]) < 3:
                k["why"].append(f"{x}: {e[x]}")
    return per


def values(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="a10.py values")
    ap.add_argument("--values", required=True, help="a11_slice.py's value slice, with ckpt/ holding the layer and head subsets and A9's expert share")
    ap.add_argument("--bundle", required=True, help="rank 0's replay bundle (replay_bundle_p0_rank0)")
    ap.add_argument("--commit", required=True, help="the Commit's records (runs.jsonl, replay_deferred.json, summary.json)")
    ap.add_argument("--build", required=True, help="the Build's rank 0 dir (build_request_LP256_T31 under it)")
    ap.add_argument("--manifest", required=True, help="the Commit's required manifest")
    ap.add_argument("--roles", required=True)
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-a9", action="store_true")
    ap.add_argument("--no-a11", action="store_true")
    a, rest = ap.parse_known_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C = Committed(a)
    install(C)
    rc = {}
    common = ["--values", a.values, "--roles", a.roles, "--profile", a.profile]
    try:
        if not a.no_a11:
            sys.argv = ["a11.py", *common, "--kinds", a.kinds, "--bin", "-", "--lean", "-", "--out", str(out / "a11"),
                        "--save-values", str(out / "saved-a11"), *rest]
            rc["a11"] = a11.main()
        if not a.no_a9:
            sys.argv = ["a9.py", *common, "--out", str(out / "a9"), "--save-values", str(out / "saved-a9")]
            rc["a9"] = a9.main()
    except Exception as ex:  # noqa: BLE001
        rc["error"] = f"{type(ex).__name__}: {ex}"[:600]
        rc["trace"] = traceback.format_exc()[-3000:]
        print(rc["trace"], flush=True)
    finally:
        C.record.update(rc=rc, reads=C.reader.record(), weights_consumed=sorted(C.consumed), per_kind=summarize(C))
        (out / "compare.json").write_text(json.dumps({"record": C.record, "calls": C.calls}, indent=1, sort_keys=True, default=str))
        print("A10 " + json.dumps({"rc": {k: v for k, v in rc.items() if k != "trace"}, "per_kind": C.record["per_kind"]},
                                  sort_keys=True, default=str)[:6000], flush=True)
    return 0 if rc and all(v == 0 for k, v in rc.items() if k in ("a11", "a9")) and "error" not in rc else 1


def audit(argv: list[str]) -> int:
    a11.WINDOW_OF, a11.LINK = WINDOW_OF, LINK
    sys.argv = ["a11.py", *argv]
    return a11.main()


if __name__ == "__main__":
    mode, *args = sys.argv[1:] or ["-h"]
    if mode not in ("values", "audit"):
        raise SystemExit(__doc__)
    raise SystemExit(values(args) if mode == "values" else audit(args))
