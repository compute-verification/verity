"""[R17 D90, COORD R17-4] The CROSS-RANK leg of the per-rank Match: the two ranks' folds (and, when given, the collective recorder's rows
and the rank Programs' `tp_links.json`) must state the SAME collectives -- so that rank r's `peers_<request>` root field (step, k) IS the other
rank's own partial of collective (step, k), produced in the other rank's log by its own EmbeddingShard / Gemm instance, and never a value
either rank invented.  Everything here is structural (kind, N, rows per request, site, order, count, producer class, link binding); the
VALUE of the received collective is the value leg's (`tp2_match` provenance: received == Def(part_rank0, part_rank1) bit-exact from the
recorded partials; TP-12 at Commit from the committed partials).  The checks FAIL BY NAME:

    count          the two ranks (and the recorder) issued the same number of collectives in every step, over the same steps
    order          collective k of step s has the same kind on both ranks (a reordered lane fails here)
    op             ... and the same N (AllReduce2_v1{N} / AllGather2_v1{N})
    segmentation   ... and the same rows, split over the same requests with the same per-request row counts
    site           ... issued by the same module
    shard_linkage  rank r's peers field (s, k) for request q has the shape of rank 1-r's own partial rows of q at (s, k)
    producer       rank r's own partial at (s, k) is produced IN r's LOG by an instance of an admitted partial-producer class
    step           the collectives of step s carry step s on both ranks (an off-by-one step fails here)
    recorder       the recorder's (rank, step, k) rows agree in kind / module / shape / count with the fold's collectives
    value          (recorder digests, every collective) both ranks RECEIVED the same bytes at (s, k), with the Definition's received geometry;
                   publishes `peer_binding`: the digest rank r's peers field (s, k) must bind at Commit = rank 1-r's partial at (s, k)
    provenance     (tp2_match `check`) received == AllReduce2/AllGather2(part_rank0, part_rank1) bit-exact at EVERY (s, k), both ranks
    links          the rank Program's link k binds peer input k to collective k of rank 1-r with the fold's kind / shape / site (a wrong
                   peer -- peer_collective_k != k or peer_rank == rank -- fails here); the site's class admits the own partial's producer
                   class (an equal-result-but-wrong-producer fails here)
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

PARTIAL_PRODUCERS = ("EmbeddingShard_v1", "Gemm_v1", "Gemm_v2", "MoeSum", "MoeExpertSum", "batch:Gemm")   # own-partial producer classes
#: the producer class the ISSUING MODULE's class admits (the link table names the class: `... (RowParallelLinear)`): an all-reduce issued
#: by a RowParallelLinear reduces a K-sharded Gemm output, never an embedding row (equal-result-but-wrong-producer fails here by name)
PRODUCER_BY_SITE_CLASS = {"VocabParallelEmbedding": ("EmbeddingShard_v1",),
                          "RowParallelLinear": ("Gemm_v1", "Gemm_v2", "batch:Gemm"),
                          "LogitsProcessor": ("Gemm_v1", "Gemm_v2", "batch:Gemm"), "ParallelLMHead": ("Gemm_v1", "Gemm_v2", "batch:Gemm"),
                          "FusedMoE": ("MoeSum", "MoeExpertSum"), "SharedFusedMoE": ("MoeSum", "MoeExpertSum"), "MoERunner": ("MoeSum", "MoeExpertSum")}


def _site_class(module_path: str | None) -> str | None:
    s = module_path or ""
    if " (" in s and s.endswith(")"):
        return s[s.rindex(" (") + 2:-1]
    return None


def _norm_site(s: str | None) -> str:
    """A module path with its ` (<Class>)` suffix dropped, as the recorder / fold spell sites (`model.named_modules()` vocabulary)."""
    return (s or "").split(" (")[0].strip()


def _same_site(a: str | None, b: str | None) -> bool:
    """[lane vllm-tp-n] The derive's link module (`link.module`: module_path minus the derive's RECORDED model root -- data the derive wrote
    by object identity) vs the fold's / recorder's site: equal, or a dotted suffix of one another (a v1 tp_links without a recorded root
    spells the traced wrapper's path).  No vocabulary prefix is spelled here (`derive_step.same_module_path`)."""
    from verity_vllm.pipeline.build import same_module_path
    return same_module_path(a, b)


def collectives_from_summary(summary: dict) -> list[dict]:
    return list(summary.get("collectives") or [])


def _own_run(rs: list, c: dict) -> Any:
    """The run of a collective row's operand (`parts` in rank order) that holds this rank's own part: the first run for rank 0, the last for
    rank world-1, and for a rank between them the run holding leaf `rank * N` (its part's first leaf)."""
    rank, world = int(c["rank"]), int(c.get("world") or 2)
    if rank == 0 or rank == world - 1 or not rs:
        return rs[0] if rank == 0 else rs[-1]
    at, off = rank * int(c["N"]), 0
    for r in rs:
        if off <= at < off + int(r[3]) - int(r[2]):
            return r
        off += int(r[3]) - int(r[2])
    return rs[-1]


def own_producers(result: Any) -> dict[tuple[int, int], list[str]]:
    """From an in-memory FoldResult: for every collective (step, k) the spec ids of the instances that PRODUCED the own-partial rows
    (the node-space runs of operand 0 of each collective row instance).  A row whose own part is not a node output (a parameter or an
    unwritten read) yields the marker `<not-a-node>`."""
    from verity.ml.circuits.refs import runs
    out: dict[tuple[int, int], list[str]] = {}
    node_spec = {}
    for i, n in enumerate(result.body.nodes):
        fid = getattr(n.fn, "id", None) or str(n.fn)
        node_spec[i] = fid
    for m in result.instances:
        c = m.get("collective")
        if not c:
            continue
        key = (int(c["step"]), int(c["k"]))
        node = result.body.nodes[m["node"]]
        arg = node.args[0]
        own = _own_run(list(runs(arg.refs)), c)
        if own[0] == "n":
            out.setdefault(key, []).append(node_spec.get(own[1], "?"))
        else:
            out.setdefault(key, []).append("<not-a-node>")
    return out


def own_producers_from_record(instances_path: str | Path, program_nodes_spec: dict[int, str] | None = None) -> dict[tuple[int, int], list[str]]:
    """The same from a persisted record: `instances.jsonl` rows carrying `collective` + `operand_refs` (node-space runs kept in the file);
    the producer's spec is read from the row whose `node` the own run names."""
    spec_of: dict[int, str] = {}
    rows = []
    with open(instances_path) as f:
        for line in f:
            o = json.loads(line)
            if "node" not in o:
                continue
            spec_of[int(o["node"])] = o.get("spec")
            if o.get("collective"):
                rows.append(o)
    out: dict[tuple[int, int], list[str]] = {}
    for o in rows:
        c = o["collective"]
        key = (int(c["step"]), int(c["k"]))
        refs = (o.get("operand_refs") or [[]])[0]
        rank, world = int(c["rank"]), int(c.get("world") or 2)
        if 0 < rank < world - 1:
            from verity_vllm.program.compact import runs_from_file_items
            rs = runs_from_file_items(refs)
            own = _own_run(rs, c) if rs else None
        else:
            own = refs[0] if rank == 0 else refs[-1]
            if own and own[0] == "N":   # [R17 mfast F2] v2 file: a node-space progression ["N", idx, lo, hi, dI, dL, dH, count]; the own part
                from verity_vllm.program.compact import runs_from_file_items   # is its first (rank 0) / last (rank world-1) run
                r = runs_from_file_items([own])
                own = r[0] if rank == 0 else r[-1]
        if own and own[0] == "n":
            out.setdefault(key, []).append(spec_of.get(int(own[1]), "?"))
        else:
            out.setdefault(key, []).append("<not-a-node>")
    return out


def load_recorder(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class _Check:
    def __init__(self, res: dict, name: str, what: str) -> None:
        self.res, self.name, self.what, self.fails = res, name, what, []

    def fail(self, msg: str) -> None:
        if len(self.fails) < 32:
            self.fails.append(msg)
        self.n_fail = getattr(self, "n_fail", 0) + 1

    def close(self) -> bool:
        ok = not self.fails
        self.res["checks"].append({"name": self.name, "what": self.what, "pass": ok, "n_fail": getattr(self, "n_fail", 0), "fails": self.fails})
        return ok


def cross_rank_check(colls: dict[int, list[dict]], *, producers: dict[int, dict[tuple[int, int], list[str]]] | None = None,
                     fold_errors: dict[int, int] | None = None, recorder: dict[int, list[dict]] | None = None,
                     links: dict[int, list[dict] | dict[str, list[dict]]] | None = None, provenance: dict | None = None,
                     recorder_sha256: dict[int, str] | None = None, require_all: bool = False) -> dict:
    """`colls[r]` = rank r's fold `collectives` (fold_summary.json / FoldResult.collectives); `producers[r][(s,k)]` = own-partial producer
    specs (`own_producers*`); `fold_errors[r]` = the rank fold's error count (a fold with errors never passes: an Unresolved own part is a
    collective input nobody computed); `recorder[r]` = coll.rank<r>.jsonl rows; `links[r]` = tp_links.json `links` of rank r's Program
    (B1) or {request_id: links of that request's Program} (B>1); `provenance` = tp2_match.json's `provenance` (the stored-step value leg)."""
    res: dict[str, Any] = {"schema": "tp2/rank_match/v1", "ranks": sorted(colls), "checks": [], "verdict": None}
    ranks = sorted(colls)
    world = len(ranks)
    # [lane vllm-tp-n] any world >= 2: the ranks must be exactly 0..world-1; every pairwise check below runs rank r against rank 0 and the
    # peer checks over every OTHER rank (world-1 peers per collective); at world 2 this is the R17 two-rank rule verbatim
    if world < 2 or ranks != list(range(world)):
        res["checks"].append({"name": "ranks", "pass": False, "fails": [f"need ranks 0..world-1 (world >= 2), got {ranks}"]})
        res["verdict"] = "FAIL"
        return res
    res["world"] = world
    peers_of = {r: [p for p in ranks if p != r] for r in ranks}
    by_key = {r: {(int(c["step"]), int(c["k"])): c for c in colls[r]} for r in ranks}
    for r in ranks:
        if len(by_key[r]) != len(colls[r]):
            res["checks"].append({"name": "count", "pass": False, "fails": [f"rank {r}: duplicate (step, k) among its collectives"]})
            res["verdict"] = "FAIL"
            return res
    steps = {r: sorted({s for s, _k in by_key[r]}) for r in ranks}

    c = _Check(res, "count", "same steps and same number of collectives per step on every rank")
    for r in ranks[1:]:
        if steps[0] != steps[r]:
            c.fail(f"steps differ: rank0 {steps[0][:8]}.. rank{r} {steps[r][:8]}..")
    for s in sorted(set().union(*(set(steps[r]) for r in ranks))):
        ns = [sum(1 for (ss, _k) in by_key[r] if ss == s) for r in ranks]
        if len(set(ns)) != 1:
            c.fail(f"step {s}: rank0 issued {ns[0]} collectives, rank1 {ns[1]}" if world == 2 else f"step {s}: collectives issued per rank {ns}")
    for r in ranks:
        for s in steps[r]:
            ks = sorted(k for (ss, k) in by_key[r] if ss == s)
            if ks != list(range(len(ks))):
                c.fail(f"rank {r} step {s}: collective ordinals are {ks[:8]}.., not 0..{len(ks) - 1}")
    c.close()
    common = sorted(set.intersection(*(set(by_key[r]) for r in ranks)))

    c = _Check(res, "order", "collective k of step s has the same kind on every rank")
    for key in common:
        for r in ranks[1:]:
            if by_key[0][key]["kind"] != by_key[r][key]["kind"]:
                c.fail(f"(step {key[0]}, k {key[1]}): rank0 {by_key[0][key]['kind']} vs rank{r} {by_key[r][key]['kind']}")
    c.close()

    c = _Check(res, "op", "same N (AllReduce2_v1{N} / AllGather2_v1{N}; AllReduce_v2{WORLD,N} / AllGather_v1{WORLD,N} beyond world 2) on every rank")
    for key in common:
        for r in ranks[1:]:
            if int(by_key[0][key]["N"]) != int(by_key[r][key]["N"]):
                c.fail(f"(step {key[0]}, k {key[1]}): N {by_key[0][key]['N']} vs {by_key[r][key]['N']}" + ("" if world == 2 else f" (rank0 vs rank{r})"))
    c.close()

    c = _Check(res, "segmentation", "same rows, over the same requests with the same per-request row counts")
    for key in common:
        a = by_key[0][key]
        fa = {f["request"]: int(f["rows"]) for f in a.get("fields") or []}
        for r in ranks[1:]:
            b = by_key[r][key]
            tag = "" if world == 2 else f" (rank0 vs rank{r})"
            if int(a["rows"]) != int(b["rows"]):
                c.fail(f"(step {key[0]}, k {key[1]}): rows {a['rows']} vs {b['rows']}{tag}")
            fb = {f["request"]: int(f["rows"]) for f in b.get("fields") or []}
            if fa != fb:
                c.fail(f"(step {key[0]}, k {key[1]}): rows by request {fa} vs {fb}{tag}")
    c.close()

    c = _Check(res, "site", "issued by the same module on every rank")
    for key in common:
        for r in ranks[1:]:
            if _norm_site(by_key[0][key]["site"]) != _norm_site(by_key[r][key]["site"]):
                c.fail(f"(step {key[0]}, k {key[1]}): site {by_key[0][key]['site']!r} vs {by_key[r][key]['site']!r}" + ("" if world == 2 else f" (rank0 vs rank{r})"))
    c.close()

    c = _Check(res, "shard_linkage", "rank r's peers field (s, k, peer p) per request has the shape of rank p's own partial rows of that request, "
                                     "for every other rank p")
    for key in common:
        for r in ranks:
            mine = by_key[r][key]
            for f in mine.get("fields") or []:
                p = int(f.get("peer_rank", peers_of[r][0]))         # world-2 folds before the peer_rank field: the one other rank
                if p == r or p not in by_key:
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): peers field for request {f['request']!r} names peer rank {p} (not another rank)")
                    continue
                other = by_key[p][key]
                of = next((g for g in (other.get("fields") or []) if g["request"] == f["request"]), None)
                if of is None:
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): peers field for request {f['request']!r} has no own partial on rank {p}")
                elif int(of["rows"]) != int(f["rows"]) or int(other["N"]) != int(mine["N"]):
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}) request {f['request']!r}: peers field [{f['rows']}, {mine['N']}] vs rank {p} "
                           f"own partial [{of['rows']}, {other['N']}]")
            if world > 2:
                want = {(f["request"], int(f["peer_rank"])) for f in mine.get("fields") or [] if "peer_rank" in f}
                reqs = {f["request"] for f in mine.get("fields") or []}
                need = {(q, p) for q in reqs for p in peers_of[r]}
                if want != need:
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): peers fields cover (request, peer) {sorted(want)[:4]}.., every request x every other rank required")
    c.close()

    c = _Check(res, "producer", "rank r's own partial at (s, k) is produced in r's log by an admitted partial-producer instance")
    if fold_errors:
        for r in ranks:
            if fold_errors.get(r):
                c.fail(f"rank {r}: fold recorded {fold_errors[r]} errors (an own partial nobody computed is among them, or worse)")
    if producers is not None:
        for r in ranks:
            for key in sorted(by_key[r]):
                specs = producers.get(r, {}).get(key)
                if not specs:
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): no producer recorded for the own partial")
                    continue
                bad = [sp for sp in specs if not any(str(sp).startswith(p) for p in PARTIAL_PRODUCERS)]
                if bad:
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): own partial produced by {sorted(set(map(str, bad)))[:3]}, not an admitted "
                           f"partial producer {PARTIAL_PRODUCERS}")
                if len(specs) != int(by_key[r][key]["rows"]):
                    c.fail(f"rank {r} (step {key[0]}, k {key[1]}): {len(specs)} producer rows for {by_key[r][key]['rows']} collective rows")
    c.close()

    c = _Check(res, "step", "the collectives filed under step s carry step s on both ranks")
    for r in ranks:
        for key, col in by_key[r].items():
            if int(col.get("step", key[0])) != key[0]:
                c.fail(f"rank {r}: collective keyed {key} carries step {col.get('step')}")
    c.close()

    if recorder is not None:
        c = _Check(res, "recorder", "the collective recorder's (rank, step, k) rows agree with the fold in count / kind / module / shape")
        for r in ranks:
            rows = {(int(x["step"]), int(x["k"])): x for x in recorder.get(r, [])}
            if len(rows) != len(recorder.get(r, [])):
                c.fail(f"rank {r}: duplicate (step, k) in the recorder")
            if set(rows) != set(by_key[r]):
                only_rec, only_fold = sorted(set(rows) - set(by_key[r]))[:5], sorted(set(by_key[r]) - set(rows))[:5]
                c.fail(f"rank {r}: recorder has {len(rows)} collectives, fold {len(by_key[r])} (recorder-only {only_rec}, fold-only {only_fold})")
            for key in sorted(set(rows) & set(by_key[r])):
                x, col = rows[key], by_key[r][key]
                if x["kind"] != col["kind"]:
                    c.fail(f"rank {r} {key}: recorder kind {x['kind']} vs fold {col['kind']}")
                if not _same_site(x.get("module"), col["site"]):
                    c.fail(f"rank {r} {key}: recorder module {x.get('module')!r} vs fold site {col['site']!r}")
                if list(x.get("shape") or []) != [int(col["rows"]), int(col["N"])]:
                    c.fail(f"rank {r} {key}: recorder partial shape {x.get('shape')} vs fold [{col['rows']}, {col['N']}]")
        c.close()

    if recorder is not None:
        # VALUE leg from the recorder's digests (every collective, both ranks; the recorder ran in the observer's engine run): the two ranks
        # RECEIVED the same bytes at (s, k), the received geometry is the Definition's, and the bytes rank r's peers field (s, k) must bind at
        # Commit are rank 1-r's PARTIAL at (s, k) -- published per (rank, step, k) as `peer_binding` (part_sha256 of the other rank, received
        # sha256 of this rank) so the Commit binder is checked against a digest fixed at Match, not trusted by construction (rev M-0026 R1).
        c = _Check(res, "value", "every rank received the same bytes at (s, k) (recorder digests) with the Definition's received geometry")
        rows_by = {r: {(int(x["step"]), int(x["k"])): x for x in recorder.get(r, [])} for r in ranks}
        binding: dict[str, dict[str, Any]] = {}
        for key in common:
            xs = {r: rows_by[r].get(key) for r in ranks}
            missing = [r for r in ranks if xs[r] is None]
            if missing:
                c.fail(f"{key}: recorder row missing on rank {missing[0]}")
                continue
            x0 = xs[0]
            for r in ranks[1:]:
                if not x0.get("out_sha256") or x0.get("out_sha256") != xs[r].get("out_sha256"):
                    c.fail(f"{key}: received sha256 differs across ranks ({str(x0.get('out_sha256'))[:12]} vs {str(xs[r].get('out_sha256'))[:12]})" + ("" if world == 2 else f" (rank0 vs rank{r})"))
            col = by_key[0][key]
            want_out = [int(col["rows"]), int(col["N"]) * (world if col["kind"] == "all_gather" else 1)]
            for r in ranks:
                x = xs[r]
                if list(x.get("out_shape") or []) != want_out:
                    c.fail(f"rank {r} {key}: received shape {x.get('out_shape')} vs the Definition's {want_out}")
                if not x.get("part_sha256"):
                    c.fail(f"rank {r} {key}: recorder row without a partial digest")
            # one binding per (rank r, peer p, s, k): the digest rank r's peers leaf of collective k from rank p must bind at Commit.  World 2
            # keeps the R17 key `rank<r>:s<s>_k<k>` (one peer); beyond, `rank<r>:s<s>_k<k>_r<p>` (the fold's field name per peer)
            for r in ranks:
                mine = xs[r]
                for p in peers_of[r]:
                    name = f"rank{r}:s{key[0]}_k{key[1]}" if world == 2 else f"rank{r}:s{key[0]}_k{key[1]}_r{p}"
                    binding[name] = {"rank": r, "step": key[0], "k": key[1], "kind": col["kind"], "site": _norm_site(col["site"]),
                                     "peer_rank": p, "peer_partial_sha256": xs[p].get("part_sha256"),
                                     "own_partial_sha256": mine.get("part_sha256"), "received_sha256": mine.get("out_sha256")}
        c.close()
        res["peer_binding"] = binding

    if provenance is not None:
        # the value leg over the STORED steps (tp2_match `check` part 2: received == AllReduce2/AllGather2(part_rank0, part_rank1) bit-exact,
        # both ranks) is REQUIRED here: every common (step, k) must have been checked, none mismatching
        c = _Check(res, "provenance", "tp2_match provenance: received == Def(part_rank0 .. part_rank{world-1}) bit-exact at every (step, k), every rank, "
                                      "over stored tensors DIGEST-BOUND to the recorder rows this leg loaded (rev F-r17-03)")
        n_checked, n_mis = int(provenance.get("collectives_checked") or 0), int(provenance.get("n_mismatches") or len(provenance.get("mismatches") or []))
        if not provenance.get("pass"):
            c.fail(f"provenance FAIL: {n_mis} mismatches, first {json.dumps((provenance.get('mismatches') or [None])[0])[:200]}")
        if n_checked != len(common):
            c.fail(f"provenance checked {n_checked} collectives, the folds state {len(common)} (steps stored: {provenance.get('steps_checked')})")
        # [rev F-r17-03] the provenance record must be THIS run's: its recorder-file digests are the digests of the rows loaded here, every
        # stored tensor was digested against its row (part + out, both ranks = 4 per collective), none mismatching.  A record without
        # digests (pre-F-r17-03, or hand-written) or from another run is refused by name.
        psha = provenance.get("recorder_sha256")
        if not isinstance(psha, dict) or not psha:
            c.fail("provenance record carries no recorder_sha256: not bound to any recorder rows (another run, a hand-written file, or a pre-F-r17-03 tp2_match)")
        elif recorder_sha256 is None:
            c.fail("the recorder files' digests were not given to this leg (recorder_sha256=None): the provenance record cannot be bound to the rows loaded")
        else:
            for r in ranks:
                if psha.get(str(r)) != recorder_sha256.get(r):
                    c.fail(f"rank {r}: provenance recorder_sha256 {str(psha.get(str(r)))[:12]} != the coll.rank{r}.jsonl loaded here {str(recorder_sha256.get(r))[:12]} "
                           "(provenance from ANOTHER run / other recorder rows)")
        n_dig, n_dig_mis = int(provenance.get("tensors_digested") or 0), int(provenance.get("n_digest_mismatches") or len(provenance.get("digest_mismatches") or []))
        if n_dig_mis:
            c.fail(f"{n_dig_mis} stored tensors do not digest to their recorder rows, first {json.dumps((provenance.get('digest_mismatches') or [None])[0])[:200]}")
        if n_dig != 2 * world * len(common):
            c.fail(f"provenance digested {n_dig} stored tensors, {2 * world * len(common)} required (part + out, every rank, every common (step, k))")
        c.close()

    if links is not None:
        c = _Check(res, "links", "rank r's Program link k binds its world-1 peer inputs to collective k of every other rank with the fold's kind / shape / site")
        for r in ranks:
            # B1: one Program, its k = issue order over steps.  B>1: one Program per request (`links[r]` = {request: links}); request q's k
            # runs over the collectives that carry a peers field of q, in (step, k) order -- the same k the compare's PEER source uses.
            Lr = links.get(r)
            per_req: dict[str, list[dict]] = dict(Lr) if isinstance(Lr, dict) else {"*": list(Lr or [])}
            for q, L in per_req.items():
                flat = [by_key[r][key] for key in sorted(by_key[r]) if q == "*" or any(f["request"] == q for f in by_key[r][key].get("fields") or [])]
                tag = f"rank {r}" + (f" request {q!r}" if q != "*" else "")
                if len(L) != len(flat):
                    c.fail(f"{tag}: {len(L)} links for {len(flat)} fold collectives")
                for k, (ln, col) in enumerate(zip(L, flat)):
                    if int(ln.get("k", k)) != k or int(ln.get("peer_collective_k", -1)) != k:
                        c.fail(f"{tag} link {k}: binds peer collective {ln.get('peer_collective_k')} (wrong peer)")
                    # v2 links carry `peers = [{rank, input, operand, collective_k}]` (one per other rank); a v1 link the one `peer_rank`
                    lps = ln.get("peers") if isinstance(ln.get("peers"), list) else [{"rank": ln.get("peer_rank", -1), "collective_k": ln.get("peer_collective_k", k)}]
                    got_ranks = sorted(int(x.get("rank", -1)) for x in lps)
                    if got_ranks != peers_of[r]:
                        c.fail(f"{tag} link {k}: peer ranks {got_ranks} (must be every other rank {peers_of[r]})")
                    if any(int(x.get("collective_k", k)) != k for x in lps):
                        c.fail(f"{tag} link {k}: a peer binds collective {[x.get('collective_k') for x in lps]} (wrong peer collective)")
                    if ln.get("kind") != col["kind"]:
                        c.fail(f"{tag} link {k}: kind {ln.get('kind')} vs fold {col['kind']}")
                    rows_q = int(col["rows"]) if q == "*" else int(next(f["rows"] for f in col["fields"] if f["request"] == q))
                    if list(ln.get("shape") or []) != [rows_q, int(col["N"])]:
                        c.fail(f"{tag} link {k}: shape {ln.get('shape')} vs fold [{rows_q}, {col['N']}]")
                    if not _same_site(ln.get("module", ln.get("module_path")), col["site"]):
                        c.fail(f"{tag} link {k}: module {ln.get('module', ln.get('module_path'))!r} vs fold site {col['site']!r}")
                    cls = _site_class(ln.get("module_path"))
                    want = PRODUCER_BY_SITE_CLASS.get(cls or "")
                    if want and producers is not None:
                        key = (int(col["step"]), int(col["k"]))
                        specs = producers.get(r, {}).get(key) or []
                        bad = sorted({str(sp) for sp in specs if not any(str(sp).startswith(p) for p in want)})
                        if bad:
                            c.fail(f"{tag} link {k} ({cls}): own partial produced by {bad[:3]}, the site class admits {want} (wrong producer)")
        c.close()

    if require_all:
        # [rev M-0092, not blocking] the library itself refuses an incomplete leg set: any second consumer inherits the entry point's rule
        c = _Check(res, "complete", "every leg present: recorder rows, link tables and the provenance record of the same run (require_all)")
        for name, val in (("recorder", recorder), ("links", links), ("provenance", provenance)):
            if val is None:
                c.fail(f"{name} leg not run (input absent): with peers as a free input a rank-local Match alone is trivially satisfiable")
        c.close()
    res["verdict"] = "PASS" if all(ch["pass"] for ch in res["checks"]) else "FAIL"
    res["failed"] = [ch["name"] for ch in res["checks"] if not ch["pass"]]
    res["n_collectives"] = {r: len(colls[r]) for r in ranks}
    return res


def commit_binding(match_dir: str | Path, *, world: int = 2, match_oracle: dict[str, dict] | None = None, require: bool = False) -> dict:
    """[R17 tp D90, rev M-0026 R1 at Commit] The COMMIT-STAGE binder of the peer inputs: the fold Match published, per (rank r, step, k), the
    digest its `peers_r[s,k]` leaf is bound to (`peer_binding` of tp2_rank_match.json = the recorder's digest of rank 1-r's partial at (s,k));
    the Commit compared every committed rank-local partial against the SAME recorder's stored tensors bit-exact (`match_oracle`, per rank,
    exhaustive over (step, k)).  This closes the chain BY VALUE: fold peer leaf (r, s, k) == recorder part digest (1-r, s, k) == the bytes
    rank 1-r COMMITTED at (s, k).  Checks, each named: the fold Match record exists and PASSed; every binding's peer digest equals the
    recorder row of the OTHER rank at the same (s, k) and its own digest the recorder row of its own rank; the binding covers every recorder
    (s, k) of every rank (2 x n); the Commit's match oracle PASSed on every rank.  Absent fold Match: result None (the row's Match is
    collective-level only, ◐) unless `require`, then a named FAIL.  CPU only, digests only (no tensors re-read)."""
    md = Path(match_dir)
    fm, rm = md / "fold_match" / "tp2_fold_match.json", md / "fold_match" / "tp2_rank_match.json"
    res: dict[str, Any] = {"schema": "verity/tp2/commit-binding/v1", "fold_match": str(fm), "rank_match": str(rm), "failures": [], "n_bindings": 0, "n_recorder": {}}
    if not fm.exists() or not rm.exists():
        res["result"] = False if require else None
        res["why"] = ("fold Match record missing (tp2_fold_match.json / tp2_rank_match.json): the peer leaves have no Match-stage binding" + (" -- REQUIRED" if require else "; Match is collective-level only (◐)"))
        return res
    try:
        fmd, rmd = json.loads(fm.read_text()), json.loads(rm.read_text())
    except Exception as e:  # noqa: BLE001
        res["result"], res["why"] = False, f"fold Match record unreadable: {e!r}"
        return res
    fails = res["failures"]
    if fmd.get("pass") is not True:
        fails.append(f"tp2_fold_match verdict {fmd.get('pass')} (legs {fmd.get('legs')}): a Commit bound to a failed fold Match is not a PASS ingredient")
    if rmd.get("verdict") != "PASS":
        fails.append(f"tp2_rank_match verdict {rmd.get('verdict')} failed={rmd.get('failed')}")
    binding = rmd.get("peer_binding") or {}
    res["n_bindings"] = len(binding)
    rec: dict[int, dict[tuple[int, int], dict]] = {}
    for r in range(world):
        p = md / "capture" / f"coll.rank{r}.jsonl"
        if not p.exists():
            fails.append(f"rank {r}: recorder rows {p.name} missing from the Match capture (the Commit oracle's source)")
            rec[r] = {}
            continue
        rec[r] = {(int(x["step"]), int(x["k"])): x for x in load_recorder(p)}
        res["n_recorder"][str(r)] = len(rec[r])
    n_bad = 0
    covered: dict[int, set[tuple[int, int]]] = {r: set() for r in range(world)}
    covered_peer: dict[tuple[int, int], set[tuple[int, int]]] = {(r, p): set() for r in range(world) for p in range(world) if p != r}
    for name, b in binding.items():
        r, s, k = int(b["rank"]), int(b["step"]), int(b["k"])
        pr = int(b.get("peer_rank", -1))
        if not (0 <= r < world) or not (0 <= pr < world) or pr == r:
            n_bad += 1
            if len(fails) < 24:
                fails.append(f"{name}: peer_rank {pr} for rank {r} (must be another rank of world {world})")
            continue
        mine, other = rec.get(r, {}).get((s, k)), rec.get(pr, {}).get((s, k))
        if mine is None or other is None:
            n_bad += 1
            if len(fails) < 24:
                fails.append(f"{name}: no recorder row at (step {s}, k {k}) on rank {r if mine is None else pr}")
            continue
        if not b.get("peer_partial_sha256") or b["peer_partial_sha256"] != other.get("part_sha256"):
            n_bad += 1
            if len(fails) < 24:
                fails.append(f"{name}: fold peer leaf digest {str(b.get('peer_partial_sha256'))[:12]} != rank {pr}'s partial {str(other.get('part_sha256'))[:12]} at (step {s}, k {k}) (wrong peer value)")
        if not b.get("own_partial_sha256") or b["own_partial_sha256"] != mine.get("part_sha256"):
            n_bad += 1
            if len(fails) < 24:
                fails.append(f"{name}: fold own part digest {str(b.get('own_partial_sha256'))[:12]} != rank {r}'s recorded partial at (step {s}, k {k})")
        if not b.get("received_sha256") or b["received_sha256"] != mine.get("out_sha256"):
            n_bad += 1
            if len(fails) < 24:
                fails.append(f"{name}: received digest differs from the recorder at (step {s}, k {k})")
        covered[r].add((s, k))
        covered_peer[(r, pr)].add((s, k))
    res["n_bad_bindings"] = n_bad
    for r in range(world):
        miss = sorted(set(rec.get(r, {})) - covered[r])
        if miss:
            fails.append(f"rank {r}: {len(miss)} recorder collectives without a fold binding (first {miss[:3]}) -- the fold did not state every peer leaf")
        for p in range(world):
            if p != r and not miss:
                missp = sorted(set(rec.get(r, {})) - covered_peer[(r, p)])
                if missp:
                    fails.append(f"rank {r}: {len(missp)} recorder collectives without a fold binding to peer rank {p} (first {missp[:3]})")
    if match_oracle is not None:
        for r in range(world):
            mo = match_oracle.get(str(r)) or match_oracle.get(r) or {}      # per-rank blocks are keyed int in-process, str after JSON
            if mo.get("verdict") != "PASS":
                fails.append(f"rank {r}: Commit match oracle {mo.get('verdict') or 'not run'} (committed partials vs the recorder's stored tensors): the binding has no committed value to bind to")
    else:
        fails.append("Commit match oracle not run (--no-match-oracle): the binding digests have no committed value to bind to")
    res["result"] = not fails
    res["why"] = None if not fails else "; ".join(fails[:3])
    res["rule"] = ("fold peers_r[s,k,p] leaf digest == recorder partial digest of rank p at (s,k) == bytes rank p committed at (s,k) (match oracle bit-exact), "
                   "every (s,k) of every rank, every other rank p; fold Match + rank Match PASS")
    return res
