import pytest
"""[R17 mfast] cb_a.compact: run progressions are the record's maximal runs, run-length-encoded over (idx, lo, hi) -- built from the
descriptor's refs JSON without expanding a Strided or a gather over consecutive producers, equal iff the run lists are equal, expandable
back to the record's bytes."""
import json
import random

from verity_vllm.program import compact as K
from verity.ml.circuits.refs import refs_from_json, runs


def _rand_refs(rng, depth=0):
    kind = rng.choice(["affine", "strided", "strided", "explicit", "concat", "gather"] if depth < 3 else ["affine", "strided", "explicit"])
    sp, idx = rng.choice(["n", "p"]), rng.randint(0, 3)
    if kind == "affine":
        return {"affine": [sp, idx, rng.randint(0, 40), rng.randint(0, 6)]}
    if kind == "strided":
        inner, rows = rng.randint(1, 4), rng.randint(0, 12)
        outer = rng.choice([inner, inner, inner + rng.randint(0, 5), rng.randint(0, 3), 0, -2])
        istr = rng.choice([1, 0, 3]) if inner == 1 else rng.choice([1, 1, 1, 2, 0])
        tail = rng.randint(0, inner - 1) if rng.random() < 0.2 else 0
        return {"strided": [sp, idx, rng.randint(0, 40), inner, outer, istr, inner * rows + tail]}
    if kind == "explicit":
        return {"explicit": [[sp, idx, rng.randint(0, 40)] for _ in range(rng.randint(0, 8))]}
    if kind == "gather":   # the attention shape: the same window over consecutive producer nodes (idx delta d)
        lo, ln, d, n = rng.randint(0, 5), rng.randint(1, 4), rng.choice([1, 1, 2, 5]), rng.randint(0, 12)
        i0 = rng.randint(100, 120)
        return {"concat": [{"affine": ["n", i0 + k * d, lo, ln]} for k in range(n)]}
    return {"concat": [_rand_refs(rng, depth + 1) for _ in range(rng.randint(0, 4))]}


def test_compact_is_the_rle_of_the_records_maximal_runs_and_expands_back():
    rng = random.Random(1234)
    for _ in range(6000):
        j = _rand_refs(rng)
        refs = refs_from_json(j)
        R = runs(refs)
        a = K.from_json(j)
        assert a == K.from_runs(R), (j, R, a)      # built from the JSON == built from the expanded runs
        assert K.from_refs(refs) == a, (j, R, a)   # built from the fold's own Refs objects (the writer's path, inlined Affine stretches)
        assert K.file_items_from_refs(refs) == K.file_items_from_refs_json(j) == K.file_items_from_runs(R)
        assert K.expand(a) == R                     # and inverts: equal runs <-> equal progressions
        assert K.n_runs(a) == len(R)
        assert K.first_runs(a, 2) == R[:2]
        assert [r[1] for r in R] == [i for t in a for i in K.idx_values(t)]


def test_from_refs_inlined_affine_stretches_hand_over_and_resume_exactly_like_the_accumulator():
    """[R17 mfast, fold/persist; mprof M-1098 (i)] `from_refs` folds a Concat's stretch of Affine parts in one inlined loop and hands the
    accumulator over to the generic path at every non-Affine part.  Every transition must be the greedy rule's: contiguous Affines
    merging into ONE open run (the record's push), an open run closing into out[-1] as run 2 (fixing the delta) / run k+1 of a
    progression / a new singleton, a Strided or Explicit in the middle taking the open run with it, an empty Affine skipped, a stretch
    that is the whole Concat, a stretch that ends the Concat with its last run left open for finish()."""
    from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided
    A, S, E = Affine, Strided, Explicit
    cases = [
        Concat([A("n", 5, 0, 3), A("n", 5, 3, 4), A("n", 6, 0, 3)]),                                  # merge, then a new idx
        Concat([A("n", 10, 2, 5), A("n", 11, 2, 5), A("n", 12, 2, 5), A("n", 13, 2, 5)]),              # a gather: one progression, last open
        Concat([A("n", 10, 2, 5), A("n", 11, 2, 5), S("p", 0, 0, 4, 8, 1, 16), A("n", 12, 2, 5), A("n", 13, 2, 5)]),   # Strided in the middle
        Concat([A("n", 1, 0, 2), E((("n", 1, 2), ("n", 1, 3), ("n", 9, 0))), A("n", 9, 1, 3), A("n", 9, 4, 1)]),          # Explicit contiguous with the open run
        Concat([A("n", 3, 0, 0), A("n", 3, 0, 2), A("n", 3, 0, 0), A("n", 4, 0, 2), A("n", 5, 0, 2), A("n", 6, 0, 2), A("n", 7, 0, 2), A("n", 8, 0, 2), A("n", 9, 0, 2)]),  # empties
        Concat([Concat([A("n", 1, 0, 2), A("n", 2, 0, 2)]), Concat([A("n", 3, 0, 2), A("n", 4, 0, 2)]), A("n", 5, 0, 2)]),  # nested stretches join
        Concat([S("n", 7, 0, 1, 0, 0, 5), A("n", 7, 0, 1), A("n", 7, 1, 3)]),                          # broadcast then contiguous runs
        Concat([]), A("n", 1, 0, 0), E(()), S("p", 2, 0, 3, 3, 1, 9),
        Concat([A("p", 0, 0, 4), A("p", 0, 8, 4), A("p", 0, 16, 4), A("n", 0, 0, 4), A("n", 0, 8, 4), A("n", 0, 16, 4)]),  # space change mid-progression
    ]
    for r in cases:
        R = runs(r)
        ref = K.from_runs(R)
        assert K.from_refs(r) == ref == K.from_json(r.to_json()), (r, R)
        assert K.file_items_from_refs(r) == K.file_items_from_refs_json(r.to_json()) == K.file_items_from_runs(R)
        assert K.expand(K.from_refs(r)) == R
    # and 3000 random object trees built directly (not through JSON): Concats of mixed parts with long Affine stretches
    rng = random.Random(77)
    for _ in range(3000):
        parts = []
        for _ in range(rng.randint(0, 6)):
            k = rng.random()
            if k < 0.6:
                i0, d, lo, ln = rng.randint(0, 30), rng.choice([1, 1, 2, 0]), rng.randint(0, 3), rng.randint(0, 3)
                parts += [A(rng.choice(["n", "n", "p"]), i0 + q * d, lo + (q * ln if rng.random() < 0.3 else 0), ln) for q in range(rng.randint(0, 9))]
            elif k < 0.8:
                inner = rng.randint(1, 3)
                parts.append(S(rng.choice(["n", "p"]), rng.randint(0, 5), rng.randint(0, 10), inner, rng.choice([inner, inner + 2, 0]), rng.choice([1, 1, 0, 2]), inner * rng.randint(0, 5)))
            else:
                parts.append(E(tuple((rng.choice(["n", "p"]), rng.randint(0, 5), rng.randint(0, 10)) for _ in range(rng.randint(0, 4)))))
        r = Concat(parts) if len(parts) != 1 else parts[0]
        R = runs(r)
        assert K.from_refs(r) == K.from_runs(R) == K.from_json(r.to_json()), (r, R)


def test_alias_splices_the_aliased_operand_like_the_codec():
    rng = random.Random(7)
    for _ in range(1500):
        j0, j1, j2 = _rand_refs(rng), _rand_refs(rng), _rand_refs(rng)
        node0 = K.from_json(j0)
        a = K.from_json({"concat": [j1, {"alias": [0, 0]}, j2]}, alias=lambda k, a_: node0)
        R = runs(refs_from_json({"concat": [j1, j0, j2]}))
        assert a == K.from_runs(R) and K.expand(a) == R


def test_rle_of_mapped_progressions_equals_rle_of_the_mapped_runs():
    rng = random.Random(99)
    for _ in range(6000):
        items = []
        for _ in range(rng.randint(0, 6)):
            head = (rng.choice(["N", "W"]), rng.randint(0, 2))
            c = rng.randint(1, 5)
            base = (rng.randint(0, 3), rng.randint(0, 20), rng.randint(0, 20))
            delta = (rng.choice([0, 1]), rng.choice([0, 1, 3, -2]), rng.choice([0, 1, 3, -2])) if c > 1 else (0, 0, 0)
            items.append(head + (base, delta, c))
        m = K.rle_mapped(items)
        assert m == K.rle_runs(K.expand(items), 3), (items, m)
        assert K.expand(m) == K.expand(items)


def test_the_real_shapes_are_one_progression_each():
    # the attention K gather of a prefill at position T: window [0, 512) over T consecutive producer nodes -> ONE tuple
    T = 265
    a = K.from_json({"concat": [{"affine": ["n", 3934 + k, 0, 512]} for k in range(T)]})
    assert a == [("n", (3934, 0, 512), (1, 0, 0), T)]
    # in decode: one producer per engine step, idx delta = the step's instance count
    a = K.from_json({"concat": [{"affine": ["n", 158970 + 328 * k, 0, 512]} for k in range(40)]})
    assert a == [("n", (158970, 0, 512), (328, 0, 0), 40)]
    # one KV head's rows out of a [T][KVH][D] cache: T rows of D at stride KVH*D
    T, KVH, D = 4096, 8, 128
    a = K.from_json({"strided": ["n", 7, 3 * D, D, KVH * D, 1, T * D]})
    assert a == [("n", (7, 3 * D, 4 * D), (0, KVH * D, KVH * D), T)]
    assert K.n_runs(a) == T and K.expand(a)[1] == ("n", 7, 3 * D + KVH * D, 4 * D + KVH * D)
    # contiguous rows collapse to one run, as the record's push does; a broadcast keeps its multiplicity
    assert K.from_json({"strided": ["n", 7, 0, D, D, 1, 5 * D]}) == [("n", (7, 0, 5 * D), (0, 0, 0), 1)]
    assert K.from_json({"strided": ["n", 3, 11, 1, 0, 0, 9]}) == [("n", (3, 11, 12), (0, 0, 0), 9)]
    # [GLM-5.3] a transposed weight block (the MLA-absorb BatchedGemm read, ~1e6 leaves per operand at full scale): one progression of
    # unit runs per row -- O(rows) progressions and O(rows) work, never the per-leaf O(count)
    Kd, Nd = 512, 16
    a = K.from_json({"strided": ["p", 2, 40, Kd, 1, Nd, Kd * Nd]})
    assert a == [("p", (2, 40 + q, 41 + q), (0, Nd, Nd), Kd) for q in range(Nd)]
    assert K.n_runs(a) == Kd * Nd
    j = {"strided": ["p", 2, 40, Kd, 1, Nd, Kd * Nd - 3]}   # partial last row
    assert K.from_json(j) == K.from_runs(runs(refs_from_json(j)))
    assert K.file_items_from_refs_json(j) == [["P", 2, 40 + q, 41 + q, 0, Nd, Nd, Kd if q < Nd - 1 else Kd - 3] for q in range(Nd)]


def test_bind_splits_counts_readers_per_progression_exactly_as_the_record_counts_runs():
    """compare._bind_splits in compact mode consults the reader count of every candidate Const32 only: a progression over one node is
    `count` reads, a node-index progression (the gathers) is one read of every candidate it visits (bisection), so the binding decision
    (bound / shared / read by a gather) is the record's on the same instance sequence."""
    from verity_vllm.check.match import program_compare as C
    from tests.check.test_compare_splits_binding import _prog

    def raw(**kw) -> C.Prog:
        """The builder's instance sequence BEFORE binding (its _bind_splits replaced by a no-op for the call)."""
        saved = C.Prog._bind_splits
        try:
            C.Prog._bind_splits = lambda self: None
            return _prog("raw", **kw)
        finally:
            C.Prog._bind_splits = saved

    def bound(R: C.Prog, compact: bool) -> C.Prog:
        Q = C.Prog.__new__(C.Prog)
        Q.__dict__.update(R.__dict__)
        Q.compact = compact
        Q.inst = [dict(r, args=[K.from_runs(a) if compact else list(a) for a in r["args"]]) for r in R.inst]
        Q.k2i = {r["k"]: i for i, r in enumerate(Q.inst)}
        Q._bind_splits()
        Q._segment()
        return Q

    def same(R: C.Prog) -> tuple[C.Prog, C.Prog]:
        P, Q = bound(R, False), bound(R, True)
        assert (P.n_splits_consts_bound, P.splits_values, P.splits_nodes) == (Q.n_splits_consts_bound, Q.splits_values, Q.splits_nodes)
        assert [r["k"] for r in P.inst] == [r["k"] for r in Q.inst] and P.steps == Q.steps
        for i in range(len(P.inst)):
            assert [K.expand(a) for a in Q.inst[i]["args"]] == P.inst[i]["args"]
        return P, Q

    S = [32, 16, 8]
    P, _ = same(raw(splits_form="const", S=S, weights=2))
    assert P.n_splits_consts_bound == 3
    P, _ = same(raw(splits_form="const", S=[16, 16, 16], weights=2, shared_const=True))
    assert P.n_splits_consts_bound == 0                    # one constant read by three selects: shared, never bound
    same(raw(splits_form="input", S=S, splits_node=2, weights=3))
    same(raw(splits_form="static", S=S, weights=2))
    # a gather over consecutive nodes that VISITS the first select's splits constant makes it shared (readers 2) in both forms
    R = raw(splits_form="const", S=S, weights=2)
    k0 = next(r["k"] for r in R.inst if r["spec"].startswith("Const32[0x20]"))
    R.inst.append({"k": R.inst[-1]["k"] + 1, "spec": "Gemm_v1{M=1,N=16,K=8}", "form": "call",
                   "args": [[("n", k0 - 2 + d, 0, 1) for d in range(5)]], "out": "bf16[16]"})
    P, Q = same(R)
    assert Q.inst[-1]["args"][0] == [("n", (k0 - 2, 0, 1), (1, 0, 0), 5)]     # one progression: the bisection branch
    assert P.n_splits_consts_bound == 2 and k0 in {r["k"] for r in P.inst}


def test_tp_rank_compare_in_compact_form_is_the_records(tmp_path, monkeypatch):
    """[R17 D90 x mfast] tp's two-rank D90 fixture (tests/check/test_tp2_d90_fixture.py) through compare with COMPACT operands:
    the PEER canonical source of every collective instance (a progression of reads of the request's peers node, field-relative) expands
    to the record's per-run list on both sides, the positive passes both ranks and every compare negative fails, exactly as in run form."""
    import pytest
    from verity_vllm.check.match import program_compare as C
    try:
        from tests.check import test_tp2_d90_fixture as T
    except ImportError as e:   # pragma: no cover - the fixture needs the resolver/fold stack
        pytest.skip(f"tp2 D90 fixture unavailable: {e}")

    def run(compact: bool, sub: str, rank: int, fold_kw: dict, prog_kw: dict):
        d = tmp_path / f"{sub}_{'c' if compact else 'r'}"
        d.mkdir()
        res, A, B = T.compare_rank(d, rank, T.fold_rank(rank, **fold_kw), T.rank_program(rank, **prog_kw), compact=compact)
        return res, A, B

    def canon_runs(P: C.Prog, i: int) -> list:
        return [K.expand_mixed(a) for a in P.canon(i)]

    for rank in (0, 1):
        res_r, Ar, Br = run(False, f"pos{rank}", rank, {}, {})
        res_c, Ac, Bc = run(True, f"pos{rank}", rank, {}, {})
        assert Ac.compact and Bc.compact and not Ar.compact
        assert res_r["ok"] and res_c["ok"], (res_r, res_c)
        assert res_r["canonical_sha256"] == res_c["canonical_sha256"]
        assert len(Ar.inst) == len(Ac.inst) and len(Br.inst) == len(Bc.inst)
        for i in range(len(Ar.inst)):
            assert [list(a) for a in Ar.canon(i)] == canon_runs(Ac, i), (i, Ar.inst[i]["spec"])
        for i in range(len(Br.inst)):
            assert [list(a) for a in Br.canon(i)] == canon_runs(Bc, i), (i, Br.inst[i]["spec"])
        ar = next(i for i, r in enumerate(Ac.inst) if r["spec"].startswith("AllReduce2_v1"))
        assert any(t[0] == "PEER" for t in Ac.canon(ar)[0]), Ac.canon(ar)
    for name, fold_kw, prog_kw in (("swapped_rank", {}, {"swap_parts": True}), ("wrong_peer", {}, {"peer_of": {1: 0}}),
                                   ("off_by_one_step", {}, {"peer_of": {2: 5, 5: 2}}), ("wrong_op", {}, {"wrong_op_k": 0}),
                                   ("equal_result_wrong_producer", {"mut": {"wrong_producer": True}}, {}),
                                   ("wrong_count", {"mut": {"drop_collective": 0}}, {}), ("reordered_lanes", {"mut": {"reorder": True}}, {})):
        res_r, _, _ = run(False, name, 0, fold_kw, prog_kw)
        res_c, _, _ = run(True, name, 0, fold_kw, prog_kw)
        assert not res_r["ok"] and not res_c["ok"], name
        assert res_r["canonical_sha256"] == res_c["canonical_sha256"], name
        assert res_r["totals"] == res_c["totals"], name
    # [F2 STORE axis, rev M-0500 (1)] the two-rank fold persisted as runs and as progressions: tp's collective-row reader
    # (rank_match.own_producers_from_record, the one consumer of NODE-space refs from the file) reads the same producers from either store,
    # and the cross-rank verdict over them is the same PASS; every negative above is a compare negative, this leg is the record reader's.
    import os
    from verity_vllm.pipeline import match as RC
    from verity_vllm.check.match import rank_match as RM
    folds = {r: T.fold_rank(r) for r in (0, 1)}
    prod: dict[str, dict] = {}
    for form in ("runs", "progressions"):
        monkeypatch.setenv(RC.INSTANCES_FORM_ENV, form)
        prod[form] = {}
        for r in (0, 1):
            d = tmp_path / f"store_{form}_rank{r}"
            d.mkdir()
            T.persist_fold(folds[r], d, r)
            hdr = json.loads((d / "instances.jsonl").read_text().splitlines()[0])
            assert hdr["operand_refs_form"] == form and hdr["schema"] == (RC.INSTANCES_SCHEMA_V2 if form == "progressions" else RC.INSTANCES_SCHEMA)
            prod[form][r] = RM.own_producers_from_record(d / "instances.jsonl")
    monkeypatch.delenv(RC.INSTANCES_FORM_ENV, raising=False)
    assert prod["runs"] == prod["progressions"] and prod["runs"][0] and prod["runs"][0] == {k: v for k, v in RM.own_producers(folds[0]).items()}
    verdicts = [RM.cross_rank_check({0: folds[0].collectives, 1: folds[1].collectives}, producers=prod[form],
                                    fold_errors={0: len(folds[0].errors), 1: len(folds[1].errors)})["verdict"] for form in ("runs", "progressions")]
    assert verdicts == ["PASS", "PASS"], verdicts


def _prog_facts(P) -> dict:
    return {"digest": P.digest, "inputs": P.inputs, "inst": P.inst, "n_nodes": P.n_nodes, "param_names": P.param_names,
            "rows": P.oracle_rows, "wmap": P.oracle_wmap, "input_base": P.input_base, "n_input_nodes": P.n_input_nodes,
            "peer_fields": P.peer_fields, "peer_index": P.peer_index, "tp": getattr(P, "tp", None), "seed_index": P.seed_index}


@pytest.mark.slow
def test_nothing_beside_the_files_of_record_is_read_a_forged_compact_sidecar_changes_no_document(tmp_path, monkeypatch):
    """[rev F-r17-12, M-0497] unit 7's fold-dir sidecars are withdrawn: the fold writes none, and the loader (record form and compact form)
    reads program.json + instances.jsonl only.  Negative: the forgery rev demonstrated -- another fold's compact copy placed in this fold
    dir with this fold's true digest / byte sizes / sha256 pasted in -- and, harder, a sidecar carrying rows that would turn the verdict:
    both leave the fast document == the record document, and impl.config.fold_load names the record files and the stored form read."""
    import os
    from verity_vllm.pipeline import match as RC
    from verity_vllm.check.match import program_compare as C
    from verity_vllm.check.match import global_match as gm
    from verity_vllm.check.match import global_match_fast as GMF
    from verity_vllm.program import instances_form as IF
    from tests.check import test_global_match as T
    gp, oracle = T.assemble([T.R0, T.R1_LATE], tmp_path)
    assert not [f for f in os.listdir(oracle) if ".compact." in f], os.listdir(oracle)   # the fold emits no sidecar
    # a "sidecar" pair whose source binding is perfect (the true digest, bytes and sha256 of THIS fold's record files) and whose content
    # says every instance is at step 99 -- were anything to read it, the G3/G4/G5 population would move
    pj = json.load(open(os.path.join(oracle, "program.json")))
    forged_rows = []
    for line in open(os.path.join(oracle, "instances.jsonl")).read().splitlines()[1:]:
        o = json.loads(line); o["step"] = 99; forged_rows.append(json.dumps(o))
    src = {name: {"bytes": os.path.getsize(os.path.join(oracle, name)), "sha256": IF.sha256_file(os.path.join(oracle, name))[0]}
           for name in ("program.json", "instances.jsonl")}
    open(os.path.join(oracle, "program.compact.json"), "w").write(json.dumps({"schema": "verity/program-compact/v1", "form": "progressions/v1",
                                                                            "digest": pj["digest"], "source": src, "n_rows": len(forged_rows), "nodes": []}))
    open(os.path.join(oracle, "instances.compact.jsonl"), "w").write("\n".join([json.dumps({"schema": "verity/instances-compact/v1", "digest": pj["digest"]})] + forged_rows) + "\n")
    ref = gm.check(gp, oracle)
    assert ref["verdict"] == "PASS"
    for compact in (False, True):
        C.LOAD_NOTES.clear()
        P = C.load_oracle(oracle, "o", compact=compact)
        assert {k: C.LOAD_NOTES[oracle][k] for k in ("source", "instances_schema", "operand_refs_form")} == {"source": "record files", "instances_schema": RC.INSTANCES_SCHEMA, "operand_refs_form": "runs"}
        # [n2] the two record files are named by content too (sha256 as read)
        from verity_vllm.program.instances_form import sha256_file as _sf
        assert C.LOAD_NOTES[oracle]["instances_sha256"] == _sf(os.path.join(oracle, "instances.jsonl"))[0]
        assert C.LOAD_NOTES[oracle]["program_sha256"] == _sf(os.path.join(oracle, "program.json"))[0]
        assert {r["step"] for r in P.oracle_rows} != {99} and P.compact == compact
    monkeypatch.setenv(GMF.FLAG, "fast")
    fast = gm.check(gp, oracle)
    cfg = gm.impl_info(GMF)["config"]   # the CLI's impl block, read after check() as main() does

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "impl", "phases", "utc")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    assert strip(fast) == strip(ref)
    assert cfg["fold_load"][oracle]["source"] == "record files" and cfg["fold_load"][oracle]["operand_refs_form"] == "runs"
    assert "MATCH_COMPACT_LOAD" not in os.environ and not hasattr(C, "compact_provenance")


def test_record_document_states_the_stored_form_it_loaded_like_the_fast_one(tmp_path, monkeypatch):
    """[rev S-r17-30 n1 (M-0608), M-0500 (2)] the RECORD checker's document states fold_load (per fold dir: source = the record files,
    instances_schema, operand_refs_form) exactly as the fast one does, so a record document of a v1 store and one of a v2 store of the
    same capture are distinguishable BY CONTENT (before n1 both had impl.config absent -> indistinguishable).  Through main() (the CLI
    every row of record runs) on both stores, record and fast; the fast document keeps naming it under impl.config.fold_load too."""
    import os
    from verity_vllm.pipeline import match as RC
    from verity_vllm.pipeline import cli
    from verity_vllm.check.match import global_match_fast as GMF
    from tests.check import test_global_match as T
    monkeypatch.delenv(RC.INSTANCES_FORM_ENV, raising=False)
    gp1, o1 = T.assemble([T.R0, T.R1_LATE], tmp_path / "v1")
    monkeypatch.setenv(RC.INSTANCES_FORM_ENV, "progressions")
    gp2, o2 = T.assemble([T.R0, T.R1_LATE], tmp_path / "v2")
    monkeypatch.delenv(RC.INSTANCES_FORM_ENV, raising=False)
    docs = {}
    for impl in ("record", "fast"):
        monkeypatch.setenv(GMF.FLAG, impl)
        for tag, (gp, o) in (("v1", (gp1, o1)), ("v2", (gp2, o2))):
            gpp = tmp_path / f"gp_{tag}.json"; gpp.write_text(json.dumps(gp))
            out = tmp_path / f"gm_{impl}_{tag}.json"
            cli.main(["global-match", "--oracle", o, "--global-program", str(gpp), "--no-x09", "--no-alternate", "--out", str(out)])
            docs[(impl, tag)] = json.load(open(out))
    for (impl, tag), d in docs.items():
        assert d["verdict"] == "PASS" and d["impl"]["name"] == impl, (impl, tag, d["reasons"])
        fl = d["impl"]["fold_load"]
        o = o1 if tag == "v1" else o2
        assert {k: fl[o][k] for k in ("source", "instances_schema", "operand_refs_form")} == {
            "source": "record files", "instances_schema": RC.INSTANCES_SCHEMA if tag == "v1" else RC.INSTANCES_SCHEMA_V2,
            "operand_refs_form": "runs" if tag == "v1" else "progressions"}, (impl, tag, fl)
        # [n2] EVERY document (record and fast, both stores) ties itself to the two fold files by content: sha256 as read == the files'
        from verity_vllm.program.instances_form import sha256_file as _sf
        assert fl[o]["instances_sha256"] == _sf(os.path.join(o, "instances.jsonl"))[0] and fl[o]["program_sha256"] == _sf(os.path.join(o, "program.json"))[0], (impl, tag)
        assert all(v["source"] == "record files" for v in fl.values())     # the component folds too: nothing beside the record files (F-r17-12)
        if impl == "fast":
            assert d["impl"]["config"]["fold_load"] == fl                   # the fast documents of record keep naming it under config
        else:
            assert "config" not in d["impl"]                                # the record has no declared configuration block
    # distinguishable by content: the two RECORD documents of one capture differ in impl.fold_load and (paths aside) nowhere else
    r1, r2 = docs[("record", "v1")], docs[("record", "v2")]
    assert r1["impl"]["fold_load"] != r2["impl"]["fold_load"]
    assert {v["operand_refs_form"] for v in r1["impl"]["fold_load"].values()} == {"runs"}
    assert "progressions" in {v["operand_refs_form"] for v in r2["impl"]["fold_load"].values()}

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "impl", "phases", "utc", "oracle_dir", "log", "path", "dir")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    assert strip(r1) == strip(r2) == strip(docs[("fast", "v1")]) == strip(docs[("fast", "v2")])


def test_stored_form_progressions_is_lossless_over_the_records_run_lists():
    """[R17 mfast F2, COORD R17-34 (2) / M-0496; GLM-5.3 "P"] instances.jsonl v2: node- and parameter-space references stored as
    progressions ["N"|"P", idx, lo, hi, dI, dL, dH, count].  Over random Refs JSON (every fold shape incl. the attention gather,
    broadcasts and the scattered strideds): (1) runs_from_file_items(file_items_from_refs_json(j)) == refs.runs(j) -- the stored form
    expands to the record's runs exactly; (2) the same items from the run list (the converter's path) -- from_runs is the reference
    from_json equals; (3) no "n"/"p" run sequence survives un-compacted."""
    rng = random.Random(17)
    n_items = n_runs = 0
    for _ in range(3000):
        j = _rand_refs(rng)
        r = runs(refs_from_json(j))
        items = K.file_items_from_refs_json(j)
        assert [tuple(x) for x in K.runs_from_file_items(items)] == r, (j, items, r)
        assert items == K.file_items_from_runs(r), (j, items)
        assert all((it[0] in ("N", "P") and len(it) == 8 and it[7] >= 2) or len(it) == 4 for it in items), items
        assert not any(it[0] in ("N", "P") and it[4:7] == [0, 0, 0] and it[7] == 1 for it in items)
        assert [tuple(x) for x in K.runs_from_file_items([list(x) for x in r])] == r   # a v1 file passes through
        n_items += len(items); n_runs += len(r)
    assert n_items < n_runs


@pytest.mark.slow
def test_persist_program_progressions_form_gives_the_same_prog_and_document_and_converts_losslessly(tmp_path, monkeypatch):
    """[R17 mfast F2] VERITY_INSTANCES_FORM=progressions: persist_program writes instances.jsonl v2 straight from the fold's Refs (no runs()
    expansion); (1) it converts to the v1 file BYTE FOR BYTE (rows) and v1 -> v2 gives the v2 bytes; (2) load_oracle (record form and
    compact form) and the GM-01 document are identical over both files; (3) no sidecar, same fold digest; (4) the header AND the loader
    name the form."""
    import os
    import shutil
    from verity_vllm.pipeline import match as RC
    from verity_vllm.check.match import program_compare as C
    from verity_vllm.check.match import global_match as gm
    from verity_vllm.program import instances_form as IF
    from tests.check import test_global_match as T
    monkeypatch.delenv(RC.INSTANCES_FORM_ENV, raising=False)
    gp1, o1 = T.assemble([T.R0, T.R1_LATE, T.R2_LATE], tmp_path / "v1")
    monkeypatch.setenv(RC.INSTANCES_FORM_ENV, "progressions")
    gp2, o2 = T.assemble([T.R0, T.R1_LATE, T.R2_LATE], tmp_path / "v2")
    monkeypatch.delenv(RC.INSTANCES_FORM_ENV, raising=False)
    i1, i2 = os.path.join(o1, "instances.jsonl"), os.path.join(o2, "instances.jsonl")
    h1, h2 = json.loads(open(i1).readline()), json.loads(open(i2).readline())
    assert h1["schema"] == RC.INSTANCES_SCHEMA and h1["operand_refs_form"] == "runs"
    assert h2["schema"] == RC.INSTANCES_SCHEMA_V2 and h2["operand_refs_form"] == "progressions" and h1["digest"] == h2["digest"]
    assert IF.instances_form_of(i1) == "runs" and IF.instances_form_of(i2) == "progressions"
    rows1, rows2 = open(i1).read().split("\n")[1:], open(i2).read().split("\n")[1:]
    assert rows1 != rows2 and len(rows1) == len(rows2)
    assert any('"N",' in l for l in rows2) and any('["p",' in l for l in rows2)   # (the toy's gathers are short: the size win is the real fold's)
    # (1) lossless both ways, byte for byte on the rows
    c12, c21 = str(tmp_path / "v1_as_v2.jsonl"), str(tmp_path / "v2_as_v1.jsonl")
    r = IF.convert_instances(i1, c12, "progressions")
    assert r["items_out"] < r["items_in"] and open(c12).read().split("\n")[1:] == rows2
    IF.convert_instances(i2, c21, "runs")
    assert open(c21).read().split("\n")[1:] == rows1
    assert json.loads(open(c21).readline())["operand_refs_form"] == "runs" and json.loads(open(c12).readline())["schema"] == RC.INSTANCES_SCHEMA_V2
    # (2) the same Prog (record form and compact form) and the same GM-01 document
    for compact in (False, True):
        C.LOAD_NOTES.clear()
        assert _prog_facts(C.load_oracle(o1, "o", compact=compact)) == _prog_facts(C.load_oracle(o2, "o", compact=compact))
        # (4) the form READ is named per fold dir (-> impl.config.fold_load): the two documents are distinguishable by content
        _form_keys = ("source", "instances_schema", "operand_refs_form")
        assert {k: C.LOAD_NOTES[o1][k] for k in _form_keys} == {"source": "record files", "instances_schema": RC.INSTANCES_SCHEMA, "operand_refs_form": "runs"}
        assert {k: C.LOAD_NOTES[o2][k] for k in _form_keys} == {"source": "record files", "instances_schema": RC.INSTANCES_SCHEMA_V2, "operand_refs_form": "progressions"}
        # [n2] and by content: the instances.jsonl digests differ between the two stores (the program.json ones are equal -- same descriptor)
        assert C.LOAD_NOTES[o1]["instances_sha256"] == IF.sha256_file(i1)[0] != C.LOAD_NOTES[o2]["instances_sha256"] == IF.sha256_file(i2)[0]
        assert C.LOAD_NOTES[o1]["program_sha256"] == C.LOAD_NOTES[o2]["program_sha256"] == IF.sha256_file(os.path.join(o1, "program.json"))[0]

    def strip(o):   # the two folds live in two directories: every path key is volatile, nothing else is
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "impl", "phases", "utc", "oracle_dir", "log", "path", "dir")}
        return [strip(v) for v in o] if isinstance(o, list) else o
    d1, d2 = gm.check(gp1, o1), gm.check(gp2, o2)
    assert d1["verdict"] == "PASS" and strip(d1) == strip(d2)
    from verity_vllm.check.match import global_match_fast as GMF
    monkeypatch.setenv(GMF.FLAG, "fast")
    f1, f2 = gm.check(gp1, o1), gm.check(gp2, o2)
    assert strip(f1) == strip(f2) == strip(d1)
    # (3) nothing but the two record files was written for Match (F-r17-12: no sidecar), and the fold digest of record is the SAME under both
    # stores (it is the Program descriptor's digest, program.json == fold_summary.digest; only F-r16-24's instances_sha256 tie differs)
    assert not [f for f in os.listdir(o2) if ".compact." in f] and h1["digest"] == h2["digest"] == json.load(open(os.path.join(o2, "program.json")))["digest"]
    assert IF.sha256_file(i1)[0] != IF.sha256_file(i2)[0]


def test_persist_program_writer_is_byte_identical_to_the_generic_conversion(tmp_path, monkeypatch):
    """[R17 mfast, COORD R17-39 (a) fold/persist] persist_program converts instance rows and journal entries BY SHAPE (a ViewDesc's six
    fields, an operand_refs item's list) instead of walking every element through the generic converter.  The files of record must not
    change by a byte: every row of instances.jsonl (both stored forms) and every line of accesses.jsonl.gz equals what the generic path --
    dataclasses.asdict for the views, then _jsonable over the whole record, json.dumps -- writes for the same FoldResult."""
    import dataclasses
    import gzip
    import os
    from verity_vllm.pipeline import match as RC
    from verity_vllm.program.compact import file_items_from_refs_json
    from tests.check import test_global_match as T
    from verity_vllm.observe.fold.fold import fold
    from verity.ml.circuits.refs import runs
    evs = T.batched_log([T.R0, T.R1_LATE, T.R2_LATE], wl_sha=None)
    res = fold(evs, T.profile(), strict=True)
    assert res.report.unsupported == 0 and not res.errors and res.instances and res.accesses

    def generic_view(v):
        return dataclasses.asdict(v) if dataclasses.is_dataclass(v) else RC._jsonable(v)

    def generic_rows(form):        # the pre-R17-39 writer, verbatim
        out = []
        for m in res.instances:
            node = res.body.nodes[m["node"]] if m.get("node") is not None else None
            rec = {k: m.get(k) for k in ("node", "step", "event", "resolution", "pattern", "spec", "row", "rows", "name", "hint", "request", "position")}
            if m.get("operand_roots"):
                rec["operand_roots"] = m["operand_roots"]
            if m.get("collective"):
                rec["collective"] = m["collective"]
            rec["operand_views"] = [generic_view(v) for v in m.get("operand_views") or []]
            rec["output_views"] = [generic_view(v) for v in m.get("output_views") or []]
            if node is not None:
                rec["operand_refs"] = ([file_items_from_refs_json(arg.refs.to_json()) for arg in node.args] if form == "progressions"
                                       else [[list(r) for r in runs(arg.refs)] for arg in node.args])
            out.append(json.dumps(RC._jsonable(rec), separators=(",", ":")))
        return out

    def generic_journal():
        out = []
        for r in res.accesses:
            r2 = dict(r)
            r2["view"] = generic_view(r2["view"]) if r2.get("view") is not None else None
            out.append(json.dumps(RC._jsonable(r2), separators=(",", ":")))
        return out

    for form in ("runs", "progressions"):
        monkeypatch.setenv(RC.INSTANCES_FORM_ENV, form)
        d = tmp_path / form
        d.mkdir()
        w = RC.persist_program(res, d, profile_name="toy", log_path="toy")
        assert w["instances"]["operand_refs_form"] == form
        rows = open(os.path.join(d, "instances.jsonl")).read().split("\n")[1:-1]
        assert rows == generic_rows(form) and len(rows) == len(res.instances)
        lines = gzip.open(os.path.join(d, "accesses.jsonl.gz"), "rt").read().split("\n")[:-1]
        assert lines == generic_journal() and len(lines) == len(res.accesses)
        assert any('"operand_views":[{"alloc":' in r for r in rows) and any('"view":{"alloc":' in l for l in lines)   # the shape path ran
