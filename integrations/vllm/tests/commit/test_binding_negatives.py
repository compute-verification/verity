"""The six mandatory binding negatives against the binding map + coverage checker (`verity_vllm/commit/binding.py`),
through the committer's offline core (no engine, no CUDA): the verifier holds the ORIGINAL RunCommitment and the binding map fixed at
finalize; the expected population is independent of the collector (structure x workload).

| mutation                                  | expected                                             | test |
| drop output member `/1`, bind `/0` twice  | coverage FAIL (identity_check alone PASSES)          | test_drop_member_and_duplicate_other_is_rejected |
| omit a whole producer family              | coverage FAIL                                        | test_omitted_family_is_rejected |
| omit one request / step / TP rank         | coverage FAIL                                        | test_omitted_{request,step,rank}_is_rejected |
| read a buffer after overwrite             | openings verify, value-correspondence FAIL           | test_late_read_verifies_but_fails_correspondence |
| relabel data from a different occurrence  | coverage ok, openings verify, correspondence FAIL    | test_relabelled_occurrence_is_rejected |
| wrong root or layout                      | opening / context FAIL                               | test_wrong_root_or_layout_is_rejected |
Openings are drawn from the expected population (incl. the last step and every family), not from the leaves that exist.
"""
from __future__ import annotations

import hashlib
import random

import pytest

from verity_vllm.commit import binding as B, step_rows
from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = step_rows.ROW_BYTES
PD = hashlib.sha256(b"program").hexdigest()
LAYERS = 2
STEPS = 3


def structure() -> dict:
    mods = [{"op_path": "model.embed_tokens", "class": "VocabParallelEmbedding", "invocations": 1, "members": ["in", "out"]},
            {"op_path": "model.layers.0.input_layernorm", "class": "RMSNorm", "invocations": 1, "members": ["out"]}]
    for L in range(LAYERS):
        if L > 0:
            mods.append({"op_path": f"model.layers.{L}.input_layernorm", "class": "RMSNorm", "invocations": 1, "members": ["0", "1"]})
        mods.append({"op_path": f"model.layers.{L}.self_attn.qkv_proj", "class": "Linear", "invocations": 1, "members": ["0"]})
    mods.append({"op_path": "model.rotary_emb", "class": "RotaryEmbedding", "invocations": LAYERS, "members": ["0", "1"]})
    mods.append({"op_path": "runner.sampler", "class": "Sampler", "invocations": 1, "members": ["sampled_token_ids"]})
    return {"modules": mods, "attention_layers": list(range(LAYERS)), "hidden": None, "moe_layers": [], "moe_planes": []}


def workload(steps: int = STEPS, requests=("r0",), ranks=(0,)) -> dict:
    return {"program_digest": PD, "requests": list(requests), "steps": steps, "ranks": list(ranks)}


def step_names() -> list[str]:
    """Acquisition order of one honest step (the shared rotary module is invoked once per layer)."""
    names = ["model.embed_tokens/in", "model.embed_tokens", "model.layers.0.input_layernorm"]
    for L in range(LAYERS):
        if L > 0:
            names += [f"model.layers.{L}.input_layernorm/0", f"model.layers.{L}.input_layernorm/1"]
        names += [f"model.layers.{L}.self_attn.qkv_proj/0", "model.rotary_emb/0", "model.rotary_emb/1"]
    names.append("runner.sampler/sampled_token_ids")
    return names


def produce(seed: int = 7, steps: int = STEPS) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(seed)
    return [[(n, rng.randbytes(6000 + 17 * i)) for i, n in enumerate(step_names())] for _ in range(steps)]


def committer(program: bytes = b"program") -> NativeHostCommitter:
    return NativeHostCommitter(None, run_id="run", hash_threads=1, program_digest=hashlib.sha256(program).digest())


def commit(stream, program: bytes = b"program", **map_kw):
    c = committer(program)
    run = c.commit_offline(stream)
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, hashlib.sha256(program).hexdigest(), chunk=CHUNK, **map_kw)
    return c, run, bmap


def reference(stream) -> dict[str, bytes]:
    """The producer's bytes per (step, name, invocation) -- what a value-correspondence check compares against."""
    ref: dict[str, bytes] = {}
    for step, tensors in enumerate(stream):
        seen: dict[str, int] = {}
        for n, raw in tensors:
            k = seen.get(n, 0)
            seen[n] = k + 1
            ref[f"{step}|{n}|{k}"] = raw
    return ref


def open_population(c, run, bmap, pop, k: int = 16, salt: bytes = b"t"):
    picks = B.challenge_identities(pop, bmap, run.run_root, k, salt=salt)
    assert picks, "no identities drawn"
    bound = [(ident, e, leaf) for ident, e, leaf in picks if e is not None]
    unbound = [ident for ident, e, _ in picks if e is None]
    ops = c.open([(e["identity"]["step"], leaf) for _, e, leaf in bound])
    return picks, bound, unbound, ops


def lookup(ref):
    """Producer bytes for a map entry, keyed the way `reference()` keys them (step|name|invocation)."""
    return lambda e: ref.get(f"{e['identity']['step']}|{e['name']}|{e['identity']['invocation']}")


def correspondence(ops, bound, ref) -> dict:
    return B.value_correspondence(ops, [e for _, e, _ in bound], lookup(ref), chunk=CHUNK)


# ---------------------------------------------------------------------- positive

def test_honest_run_covers_population_and_openings_correspond():
    stream = produce()
    c, run, bmap = commit(stream)
    pop = list(B.expected_population(structure(), workload()))
    cov = B.coverage_check(pop, bmap)
    assert cov["ok"], cov
    assert cov["expected_n"] == len(pop) and cov["missing_n"] == 0 and cov["extra_n"] == 0
    assert B.map_context_check(bmap, run) == (True, "PASS")
    picks, bound, unbound, ops = open_population(c, run, bmap, pop)
    assert not unbound
    assert all(c.verify(run, o) for o in ops)
    assert max(e["identity"]["step"] for _, e, _ in bound) == STEPS - 1, "the forced late-step pick is in the sample"
    fams = {e["identity"]["op_path"].split(".")[-1] for _, e, _ in bound}
    assert {"embed_tokens", "input_layernorm", "qkv_proj", "rotary_emb", "sampler"} <= fams
    vc = correspondence(ops, bound, reference(stream))
    assert vc["ok"] and vc["checked"] == len(ops), vc
    # the map is fixed before the challenge: its digest is over the entries, and the challenge changes with it
    assert bmap["digest"] == hashlib.sha256(__import__("json").dumps(bmap["entries"], sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    other = dict(bmap, digest="00" * 32)
    assert B.challenge_identities(pop, other, run.run_root, 16, salt=b"t") != picks
    # registered inputs (weights): required by the population when registered, covered by the map's registered-input entry, never drawn as a leaf
    st_w = dict(structure(), weights_registered=True)
    pop_w = list(B.expected_population(st_w, workload()))
    assert not B.coverage_check(pop_w, bmap)["ok"]                      # this run registered no weights -> missing
    bmap_w = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, chunk=CHUNK, weight_registration={"root": "ab" * 32})
    assert B.coverage_check(pop_w, bmap_w)["ok"]
    assert all(e is not None and e["kind"] == "leaf" for _, e, _ in B.challenge_identities(pop_w, bmap_w, run.run_root, 40))


def test_map_leaves_are_the_committers_leaves():
    """A map built without `chunk`, as the TP worker builds it, indexes the committer's own leaves. rows-sha512 commits 4 KiB rows, and a
    map in 256 B units sent its population draws past each step's stream: `identity_of` raised IndexError on every
    rank."""
    stream = produce()
    c = committer()
    rows = c.chunk
    run = c.commit_offline(stream)
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD)
    leaves = [e for e in bmap["entries"] if e["kind"] == "leaf"]
    assert leaves and all(e["leaf_lo"] == e["stream_offset"] // rows for e in leaves)
    pop = list(B.expected_population(structure(), workload()))
    picks, bound, unbound, ops = open_population(c, run, bmap, pop)
    assert not unbound and all(c.verify(run, o) for o in ops)
    vc = B.value_correspondence(ops, [e for _, e, _ in bound], lookup(reference(stream)), chunk=rows)
    assert vc["ok"] and vc["checked"] == len(ops), vc
    with pytest.raises(ValueError, match=f"{rows}-byte leaves"):
        B.build_binding_map(c._layouts, run.steps, run.run_root, PD, chunk=256)


# ---------------------------------------------------------------------- 1. drop /1, bind /0 twice

def test_drop_member_and_duplicate_other_is_rejected():
    stream = produce()
    victim = f"model.layers.{LAYERS - 1}.input_layernorm"
    mutated = [[(victim + "/0" if n == victim + "/1" else n, raw) for n, raw in tensors] for tensors in stream]   # same count, /0 twice
    c, run, bmap = commit(mutated)
    # the count and identity checks pass: equal counts, and the per-step name multiset is the same in every step (step 0 is itself mutated)
    man = {"modules": sorted({n.split("/")[0] for n in step_names()}), "attention_layers": [], "fa2_hidden_per_layer": 0, "sampler_ids": 1}
    assert c.identity_check(man)["ok"] is True, "the identity check alone does not see the dropped member"
    assert sum(s.n_tensors for s in run.steps) == sum(len(t) for t in stream)
    cov = B.coverage_check(list(B.expected_population(structure(), workload())), bmap)
    assert not cov["ok"]
    assert cov["missing_n"] == STEPS and all(m["op_path"] == victim and m["output_member"] == "1" for m in cov["missing"]), cov
    # the second `/0` is a second acquisition of that member in the step -> invocation 1, which the population does not require:
    # it is reported as EXTRA (allowed), never as coverage of `/1`
    assert cov["extra_n"] == STEPS and all(k[2] == 1 and k[4] == "0" for k in map(tuple, cov["extra"])), cov


# ---------------------------------------------------------------------- 2. omit a producer family

def test_omitted_family_is_rejected():
    stream = [[(n, raw) for n, raw in tensors if "qkv_proj" not in n] for tensors in produce()]
    c, run, bmap = commit(stream)
    cov = B.coverage_check(list(B.expected_population(structure(), workload())), bmap)
    assert not cov["ok"] and cov["missing_n"] == STEPS * LAYERS
    assert {m["op_path"].rsplit(".", 1)[-1] for m in cov["missing"]} == {"qkv_proj"}


# ---------------------------------------------------------------------- 3. omit one request / step / rank

def test_omitted_step_is_rejected():
    c, run, bmap = commit(produce(steps=STEPS - 1))
    cov = B.coverage_check(list(B.expected_population(structure(), workload(steps=STEPS))), bmap)
    assert not cov["ok"] and {m["step"] for m in cov["missing"]} == {STEPS - 1}
    # and openings drawn from the expected population reach the missing late step: they cannot be opened
    picks = B.challenge_identities(list(B.expected_population(structure(), workload(steps=STEPS))), bmap, run.run_root, 8)
    assert any(e is None and ident[2] == STEPS - 1 for ident, e, _ in picks)


def test_omitted_request_is_rejected():
    # B=2: every per-token member's leading dimension is split between r0 and r1 by the engine's row order of that step (request_rows:
    # (request, token_lo, token_hi)); entries carry REQUEST-LOCAL element ranges + the global element offset (the manifest's convention)
    stream = produce()
    lens = {i: len(raw) for i, (n, raw) in enumerate(stream[0])}                                # by tensor ordinal (the shared rotary repeats its name)
    rows_ok = {s: [("r0", 0, 100), ("r1", 100, max(lens.values()))] for s in range(STEPS)}      # tensors differ in length: only members whose
    c, run, bmap2 = commit(stream, request_rows=rows_ok)                                        # rows == tokens split; others bind whole to both
    pop2 = list(B.expected_population(structure(), workload(requests=("r0", "r1"))))
    cov = B.coverage_check(pop2, bmap2)
    assert cov["ok"], cov
    e_r1 = [e for e in bmap2["entries"] if e["identity"]["request_id"] == "r1" and e["identity"]["element_range"][1] != lens[e["tensor_ordinal"]]]
    assert e_r1 and all(e["identity"]["element_range"][0] == 0 and e["element_offset"] == 100 for e in e_r1), "request-local ranges + global offset"
    # a manifest range for r1 that exceeds what r1's rows hold is NOT covered by r0's rows or by the whole-member entry of another request
    big = [B.identity(PD, "r1", 1, 0, "model.embed_tokens", "out", 0, 10 ** 9, 0)]
    assert not B.coverage_check(big, bmap2)["ok"]
    # the same run without r1 in the row map (B=1 map: everything is r0) -> every r1 identity is missing
    honest = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, chunk=CHUNK)
    cov = B.coverage_check(pop2, honest)
    assert not cov["ok"] and {m["request_id"] for m in cov["missing"]} == {"r1"}
    assert cov["bound_n"] == len(pop2) // 2
    # a request dropped from ONE step's row map is a missing identity in that step only
    rows_gap = {s: (rows_ok[s] if s != 1 else [("r0", 0, max(lens.values()))]) for s in range(STEPS)}
    _c, _run, bmap_gap = commit(stream, request_rows=rows_gap)
    cov = B.coverage_check(pop2, bmap_gap)
    assert not cov["ok"] and {(m["request_id"], m["step"]) for m in cov["missing"]} == {("r1", 1)}


def test_global_manifest_addresses_must_follow_engine_row_order(tmp_path):
    """B=2 global manifest (the required-value manifest's shape: workload ids in `requests_declared`, GLOBAL element ranges, batch-level rows with
    `request_id "*"` + `engine_step`): ranges at the engine's row order are covered; ranges assuming declaration order when the runner
    put r1's rows first are `address_mismatch` (bound elsewhere, NOT covered) -- the live Llama B2 step-0 disagreement."""
    stream = produce()
    lens = {i: len(raw) for i, (n, raw) in enumerate(stream[0])}
    n_tok = max(lens.values())
    rows = {s: [("r1", 0, 40), ("r0", 40, n_tok)] for s in range(STEPS)}                        # engine: the short request's rows FIRST
    _c, _run, bmap = commit(stream, request_rows=rows)
    WID = ["req-aaaa-0000", "req-bbbb-1111"]
    emb = [e for e in bmap["entries"] if e["name"] == "model.embed_tokens" and e["identity"]["step"] == 1]
    per_tok = {e["identity"]["request_id"]: e for e in emb}
    assert per_tok["r1"]["element_offset"] == 0 and per_tok["r0"]["element_offset"] == 40 * (lens[1] // n_tok)

    def manifest(r1_lo: int, r1_hi: int) -> str:
        from verity_vllm.query.manifest.format import manifest_digest as _mdig

        ids = [{"family": "instance_outputs", "program_digest": "other-program", "request_id": WID[1], "step": 1, "invocation": 0, "op_path": "model.embed_tokens",
                "output_member": "out", "element_range": [r1_lo, r1_hi], "rank": 0},
               {"family": "instance_outputs", "program_digest": "other-program", "request_id": "*", "requests": WID, "step": "*", "engine_step": 2, "invocation": 0,
                "op_path": "model.rotary_emb", "output_member": "0", "element_range": [0, None], "rank": 0}]
        doc = {"schema": "verity-sweep/required-manifest/v1", "global": True, "program_digest": "other-program", "manifest_digest": _mdig(ids),
               "requests_declared": [{"request_id": WID[0], "index": 0}, {"request_id": WID[1], "index": 1}], "identities": ids}
        p = tmp_path / f"m_{r1_lo}.json"
        p.write_text(__import__("json").dumps(doc))
        return str(p)

    r1_len = per_tok["r1"]["identity"]["element_range"][1]
    ok_pop, hdr = B.load_required_manifest(manifest(0, r1_len), PD, requests=["r0", "r1"], steps=STEPS, arrivals={"r0": 0, "r1": 0})
    assert hdr["global"] and hdr["requests_declared_n"] == 2 and len(ok_pop) == 3                # 1 per-request row + the batch row x 2 requests
    assert {p[1] for p in ok_pop} == {"r0", "r1"} and all(p[2] == 2 for p in ok_pop if p[4] == "model.rotary_emb")   # engine_step is absolute
    cov = B.coverage_check(ok_pop, bmap)
    assert cov["ok"] and cov["address_mismatch_n"] == 0, cov
    r0_len = per_tok["r0"]["identity"]["element_range"][1]
    bad_pop, _ = B.load_required_manifest(manifest(r0_len, r0_len + r1_len), PD, requests=["r0", "r1"], steps=STEPS, arrivals={"r0": 0, "r1": 0})
    cov = B.coverage_check(bad_pop, bmap)                                                       # declaration-order address for r1: bound, wrong place
    assert not cov["ok"] and cov["missing_n"] == 1 and cov["address_mismatch_n"] == 1
    assert cov["missing"][0]["request_id"] == "r1" and cov["missing"][0]["bound_at"] == [[[0, r1_len], 0]]


def test_omitted_rank_is_rejected():
    c, run, bmap = commit(produce(), rank=0)
    cov = B.coverage_check(list(B.expected_population(structure(), workload(ranks=(0, 1)))), bmap)
    assert not cov["ok"] and {m["rank"] for m in cov["missing"]} == {1} and cov["bound_n"] == cov["expected_n"] // 2


# ---------------------------------------------------------------------- 4. late read: openings verify, value wrong

def test_late_read_verifies_but_fails_correspondence():
    stream = produce()
    victim = "model.layers.0.self_attn.qkv_proj/0"
    overwritten = [[(n, bytes(b ^ 0x5A for b in raw) if n == victim else raw) for n, raw in tensors] for tensors in stream]   # RoPE-in-place twin
    c, run, bmap = commit(overwritten)
    pop = list(B.expected_population(structure(), workload()))
    assert B.coverage_check(pop, bmap)["ok"]                    # the identity IS bound (to the wrong bytes)
    ident = B.identity(PD, "r0", 1, 0, "model.layers.0.self_attn.qkv_proj", "0", 0, -1, 0)
    e = B.map_index(bmap)[("r0", 1, 0, "model.layers.0.self_attn.qkv_proj", "0", 0)][0]
    ops = c.open([(1, e["leaf_lo"] + 1)])
    assert all(c.verify(run, o) for o in ops), "openings verify against the committed root"
    vc = B.value_correspondence(ops, [e], lookup(reference(stream)), chunk=CHUNK)
    assert not vc["ok"] and vc["bad"][0]["why"] == "opened bytes != producer bytes"


# ---------------------------------------------------------------------- 5. relabel data from a different occurrence

def test_relabelled_occurrence_is_rejected():
    stream = produce()
    a, b = "model.layers.0.self_attn.qkv_proj/0", "model.layers.1.self_attn.qkv_proj/0"
    swapped = []
    for tensors in stream:
        d = dict(tensors)
        swapped.append([(n, d[b] if n == a else d[a] if n == b else raw) for n, raw in tensors])
    c, run, bmap = commit(swapped)
    pop = list(B.expected_population(structure(), workload()))
    assert B.coverage_check(pop, bmap)["ok"]                    # identity coverage cannot see it: both identities have a binding
    idx = B.map_index(bmap)
    ea, eb = idx[("r0", 2, 0, a.split("/")[0], "0", 0)][0], idx[("r0", 2, 0, b.split("/")[0], "0", 0)][0]
    ops = c.open([(2, ea["leaf_lo"]), (2, eb["leaf_lo"])])
    assert all(c.verify(run, o) for o in ops)
    ref = reference(stream)
    vc = B.value_correspondence(ops, [ea, eb], lookup(ref), chunk=CHUNK)
    assert not vc["ok"] and vc["bad_n"] == 2


# ---------------------------------------------------------------------- 6. wrong root or layout

def test_wrong_root_or_layout_is_rejected():
    stream = produce()
    c, run, bmap = commit(stream)
    c2, run2, bmap2 = commit(stream, program=b"other-program")          # same bytes, other Program -> other roots
    assert run.run_root != run2.run_root
    ok, why = B.map_context_check(bmap, run2)
    assert not ok and "run_root" in why
    ok, why = B.map_context_check(dict(bmap, run_root=run2.run_root.hex()), run2)
    assert not ok and "step_root" in why
    ok, why = B.map_context_check(dict(bmap, layout="chunk-leaf-v2"), run)
    assert not ok and "layout" in why
    pop = list(B.expected_population(structure(), workload()))
    picks, bound, unbound, ops = open_population(c, run, bmap, pop)
    assert all(c.verify(run, o) for o in ops)
    assert not any(c.verify(run2, o) for o in ops), "openings of run A never verify against run B's roots"
    assert not any(c2.verify(run, o) for o in ops), "a verifier holding the other context rejects them"


# ---------------------------------------------------------------------- B8 global binding map (request-local addressing + composition)

B8_LP = [5, 9, 7, 6, 2, 1, 3, 8]            # prompt lengths (the real row: [389, 1007, 757, 515, 13, 8, 195, 1024])
B8_CAP = [3, 4, 1, 5, 2, 2, 1, 5]           # max_tokens per request (the real row: [47, 77, 5, 128, 15, 21, 2, 123]); all arrive at step 0
B8_MEMBERS = ["model.embed_tokens/in", "model.embed_tokens", "model.layers.0.input_layernorm", "runner.sampler/sampled_token_ids"]


def b8_declared() -> dict[int, list[tuple[str, int]]]:
    """Declared composition per engine step, Program x workload: request k is active at steps [0, cap_k); rows = LP at its prefill, 1 after."""
    out: dict[int, list[tuple[str, int]]] = {}
    for e in range(max(B8_CAP)):
        out[e] = [(f"r{k}", B8_LP[k] if e == 0 else 1) for k in range(8) if e < B8_CAP[k]]
    return out


def b8_engine_rows(declared: dict[int, list[tuple[str, int]]]) -> dict[int, list[tuple[str, int, int]]]:
    """The engine's row order per step, vLLM v1 InputBatch style: slots filled in arrival order; a finished request's slot is refilled by
    the LAST slot's request (condense) -- so from step 2 on the row order is NOT the declaration order."""
    slots = [f"r{k}" for k in range(8)]
    rows: dict[int, list[tuple[str, int, int]]] = {}
    for e in sorted(declared):
        active = dict(declared[e])
        # condense: remove finished requests, moving the last slot into each freed one
        for i in range(len(slots)):
            while i < len(slots) and slots[i] not in active:
                last = slots.pop()
                if i < len(slots):
                    slots[i] = last
        lo, order = 0, []
        for rid in slots:
            n = active[rid]
            order.append((rid, lo, lo + n))
            lo += n
        rows[e] = order
    return rows


def b8_stream(rows: dict[int, list[tuple[str, int, int]]], seed: int = 3) -> list[list[tuple[str, bytes]]]:
    """One honest step per engine step: per-token members are `tokens` bytes long (rows == tokens -> split per request), the sampled-ids
    member `n_seq` bytes (rows == len(requests) -> one slot per request)."""
    rng = random.Random(seed)
    out = []
    for e in sorted(rows):
        tokens, n_seq = max(hi for _r, _lo, hi in rows[e]), len(rows[e])
        out.append([(n, rng.randbytes(n_seq if n.endswith("sampled_token_ids") else tokens)) for n in B8_MEMBERS])
    return out


def b8_manifest(tmp_path, declared: dict[int, list[tuple[str, int]]]) -> str:
    """The global manifest the way `required_manifest.build_global_manifest` writes it (request-LOCAL ranges, `engine_steps`
    composition, `composition_reference`, request ids r<k> with the workload UUID beside)."""
    import json

    ids = []
    for e, members in sorted(declared.items()):
        for rid, n in members:
            for name in B8_MEMBERS:
                op, mem = (name.split("/", 1) + ["out"])[:2] if "/" in name else (name, "out")
                fam = "sampled_token_ids" if mem == "sampled_token_ids" else ("input_token_ids" if mem == "in" else "instance_outputs")
                hi = 1 if fam == "sampled_token_ids" else n
                ids.append({"family": fam, "program_digest": PD, "request_id": rid, "step": e, "engine_step": e, "invocation": 0, "op_path": op,
                            "output_member": mem, "element_range": [0, hi], "rows": n, "width": 1, "rank": 0})
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    doc = {"schema": "verity-sweep/required-manifest/v1", "global": True, "addressing": "request-local", "program_digest": PD, "program_digest_kind": "workload",
           "manifest_digest": _mdig(ids), "complete": True, "unmodelled": {},
           "requests_declared": [{"request_id": f"r{k}", "workload_request_id": f"uuid-{k}", "index": k, "prompt_len": B8_LP[k], "max_tokens": B8_CAP[k], "arrive_step": 0} for k in range(8)],
           "engine_steps": [{"engine_step": e, "members": [{"request_id": rid, "step": e, "rows": n} for rid, n in members], "rows": sum(n for _r, n in members), "n_seq": len(members)}
                            for e, members in sorted(declared.items())],
           "composition_reference": {"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                                     "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids", "per_row": 1}},
           "identities": ids}
    p = tmp_path / "b8_manifest.json"
    p.write_text(json.dumps(doc))
    return str(p)


def b8_check(tmp_path, rows, declared=None):
    declared = declared or b8_declared()
    stream = b8_stream(rows)
    _c, _run, bmap = commit(stream, request_rows=rows)
    pop, hdr = B.load_required_manifest(b8_manifest(tmp_path, declared), PD, requests=[f"r{k}" for k in range(8)], steps=max(B8_CAP), arrivals={f"r{k}": 0 for k in range(8)})
    cov = B.coverage_check(pop, bmap)
    comp = B.composition_check(hdr, bmap)
    return bmap, pop, cov, comp


def test_b8_global_map_positive_engine_order_differs_from_declaration(tmp_path):
    declared = b8_declared()
    rows = b8_engine_rows(declared)
    # the engine's row order really differs from the declaration order once requests finish (r7 condensed into r2's slot at step 1, ...)
    assert [r for r, _lo, _hi in rows[0]] == [f"r{k}" for k in range(8)]
    assert [r for r, _lo, _hi in rows[2]] != [r for r, _n in declared[2]] and set(r for r, _lo, _hi in rows[2]) == set(r for r, _n in declared[2])
    bmap, pop, cov, comp = b8_check(tmp_path, rows)
    assert cov["ok"] and cov["address_mismatch_n"] == 0, cov["missing"][:3]
    assert comp["ok"] and comp["checked_steps"] == max(B8_CAP), comp
    assert {e["identity"]["request_id"] for e in bmap["entries"]} == {f"r{k}" for k in range(8)}
    # every per-token entry is request-local ([0, rows)) at the engine's offset
    emb0 = {e["identity"]["request_id"]: e for e in bmap["entries"] if e["name"] == "model.embed_tokens/in" and e["identity"]["step"] == 0}
    assert emb0["r1"]["identity"]["element_range"] == [0, 9] and emb0["r1"]["element_offset"] == 5


def test_b8_omit_r5_fails_named(tmp_path):
    rows = b8_engine_rows(b8_declared())
    rows = {e: [(r, lo, hi) for r, lo, hi in rr if r != "r5"] for e, rr in rows.items()}
    rows = {e: _repack(rr) for e, rr in rows.items()}                  # the server never ran r5: its rows are not in the members
    _bmap, _pop, cov, comp = b8_check(tmp_path, rows)
    assert not cov["ok"] and {m["request_id"] for m in cov["missing"]} == {"r5"}
    assert not comp["ok"] and comp["problems"][0]["engine_step"] == 0 and comp["problems"][0]["missing"] == ["r5 x1"], comp["problems"][0]
    assert "r5" in comp["error"]


def test_b8_relabel_r3_as_r2_fails_named(tmp_path):
    rows = b8_engine_rows(b8_declared())
    rows = {e: [("r2" if r == "r3" else r, lo, hi) for r, lo, hi in rr] for e, rr in rows.items()}
    _bmap, _pop, cov, comp = b8_check(tmp_path, rows)
    assert not cov["ok"] and {m["request_id"] for m in cov["missing"]} == {"r3"}                # r3's identities have no binding under its name
    p0 = comp["problems"][0]
    assert not comp["ok"] and p0["engine_step"] == 0 and p0["missing"] == ["r3 x6"] and p0["unexpected"] == ["r2 x6"] and p0["duplicated"] == ["r2"], p0
    # step 1 (all decode, every block 1 row; r2 itself finished after step 0): the relabel is invisible to widths -- the composition
    # still names it: r3 missing, an r2 the declaration does not have
    p1 = next(p for p in comp["problems"] if p["engine_step"] == 1)
    assert p1["missing"] == ["r3 x1"] and p1["unexpected"] == ["r2 x1"]


def test_b8_duplicate_r0_into_missing_r7_slot_fails_named(tmp_path):
    declared = b8_declared()
    rows = b8_engine_rows(declared)
    # at the decode steps where both run (1, 2) r7 is omitted and an r0 acquisition is bound a second time under r7's name (same rows,
    # same offset); step 0 and the steps after r0 finished stay honest
    bad = {}
    for e, rr in rows.items():
        if e not in (1, 2):
            bad[e] = rr
            continue
        rr = _repack([(r, lo, hi) for r, lo, hi in rr if r != "r7"])
        r0 = next(x for x in rr if x[0] == "r0")
        bad[e] = rr + [("r7", r0[1], r0[2])]
    _bmap, pop, cov, comp = b8_check(tmp_path, bad)
    # decode rows are 1 wide for every request: request-local coverage alone is BLIND to the duplicate (r7's ranges are "bound")
    assert cov["ok"], cov["missing"][:3]
    # the composition is not: r7's block overlaps r0's (same rows bound twice) and the sampled slot count / partition disagree
    assert not comp["ok"] and comp["problems"][0]["engine_step"] == 1
    p1 = comp["problems"][0]
    assert p1["overlaps"] and "r7" in p1["overlaps"][0] and "r0" not in p1["missing"], p1
    assert "overlaps" in comp["error"] and "r7" in comp["error"]


def _repack(rr: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out, lo = [], 0
    for r, a, b in rr:
        out.append((r, lo, lo + (b - a)))
        lo += b - a
    return out


# ---------------------------------------------------------------------- manifest input trust

def _b8_doc(tmp_path):
    import json

    p = b8_manifest(tmp_path, b8_declared())
    return p, json.loads(open(p).read())


def _write(tmp_path, name, doc):
    import json

    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_manifest_edited_after_digesting_is_refused(tmp_path):
    p, doc = _b8_doc(tmp_path)
    doc["identities"].pop(7)                                              # rows changed, manifest_digest kept
    with pytest.raises(ValueError, match="edited after digesting"):
        B.load_required_manifest(_write(tmp_path, "edited.json", doc), PD)
    doc["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="edited after digesting"):
        B.load_required_manifest(_write(tmp_path, "edited2.json", doc), PD)
    doc["schema"] = "verity-sweep/required-manifest/v0"
    with pytest.raises(ValueError, match="schema"):
        B.load_required_manifest(_write(tmp_path, "schema.json", doc), PD)


def test_manifest_zero_width_identity_is_refused_not_vacuously_covered(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["identities"][3]["element_range"] = [0, 0]                        # an EXISTING identity made zero-width
    doc["manifest_digest"] = _mdig(doc["identities"])                     # ... with an honestly recomputed digest
    with pytest.raises(ValueError, match="zero-width"):
        B.load_required_manifest(_write(tmp_path, "zw.json", doc), PD)
    p, doc = _b8_doc(tmp_path)
    doc["identities"].append(dict(doc["identities"][0], op_path="model.layers.0.mlp.phantom", element_range=[0, 0]))   # phantom op, zero width
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="zero-width"):
        B.load_required_manifest(_write(tmp_path, "zw2.json", doc), PD)


def test_manifest_request_id_not_r_k_is_refused(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["identities"][0]["request_id"] = "5b7fee10-6a81-52b7-b730-133d6cc768da"   # a bare workload UUID no declaration resolves
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="not r<k>"):
        B.load_required_manifest(_write(tmp_path, "uuid.json", doc), PD)
    # ... but a UUID that requests_declared maps to r<k> is fine
    p, doc = _b8_doc(tmp_path)
    doc["identities"][0]["request_id"] = "uuid-0"
    doc["requests_declared"][0]["request_id"] = "uuid-0"
    doc["manifest_digest"] = _mdig(doc["identities"])
    pop, hdr = B.load_required_manifest(_write(tmp_path, "uuid_ok.json", doc), PD)
    assert hdr["manifest_digest_verified"] and any(i[1] == "r0" for i in pop)


def test_manifest_program_digest_must_be_a_string(tmp_path):
    from verity_vllm.query.manifest.format import manifest_digest as _mdig

    p, doc = _b8_doc(tmp_path)
    doc["program_digest"] = [PD, "another"]                               # the rc3 merge_manifests shape: a list is not a Program of record
    doc["manifest_digest"] = _mdig(doc["identities"])
    with pytest.raises(ValueError, match="program_digest must be one string"):
        B.load_required_manifest(_write(tmp_path, "pdlist.json", doc), PD)


def test_composition_check_skips_a_one_request_manifest_whose_engine_steps_are_numbers():
    """A B=1 `build` manifest's engine_steps are the request's engine-step NUMBERS (ints, rc3
    schema); only build-global writes the dict composition.  The int list must be skipped (recorded), never indexed as a dict."""
    bmap = {"entries": []}
    r = B.composition_check({"engine_steps": [0, 1, 2, 3], "composition_reference": {}}, bmap)
    assert r["ok"] and r["checked_steps"] == 0 and "no engine-step composition" in r["skipped"]
    r = B.composition_check({"engine_steps": list(range(32))}, bmap)
    assert r["ok"] and r["checked_steps"] == 0
    r = B.composition_check({}, bmap)
    assert r["ok"] and r["checked_steps"] == 0
    # a global manifest with no reference member stays a hard error (composition unverifiable), not a skip
    r = B.composition_check({"engine_steps": [{"engine_step": 0, "members": [{"request_id": "r0", "step": 0, "rows": 4}], "rows": 4, "n_seq": 1}]}, bmap)
    assert not r["ok"] and "composition unverifiable" in r["error"]


def test_commit_delta_refuses_a_required_manifest_without_the_build_program_digest():
    """--required-manifest without --program-digest is a named refusal before any engine is built (commit_delta calls this right
    after parse_args): a manifest cannot vouch for its own Program."""
    with pytest.raises(SystemExit, match="REFUSED: --required-manifest requires --program-digest.*cannot vouch for its own Program"):
        B.require_build_program_digest("/x/manifest.json", None)
    with pytest.raises(SystemExit):
        B.require_build_program_digest("/x/manifest.json", "  ")
    B.require_build_program_digest("/x/manifest.json", PD)                    # given: proceeds (the manifest/Build equality is checked later)
    B.require_build_program_digest(None, None)                                # no manifest: nothing to refuse


def test_load_required_manifest_accepts_a_one_request_manifest_whose_engine_steps_are_numbers(tmp_path):
    """`required_manifest build` (B=1) writes `engine_steps` as ints (arrival + s), which a composition normalisation must not
    subscript ('int' object is not subscriptable -> every B=1 Commit 'binding map not built').  The loader must keep the numbers, declare NO composition, and composition_check must skip (named) on that header."""
    p, doc = _b8_doc(tmp_path)
    doc["global"] = False
    doc.pop("requests_declared"); doc.pop("composition_reference")
    doc["engine_steps"] = [0, 1, 2]                                        # what `build` writes for one request arriving at engine step 0
    p1 = _write(tmp_path, "b1_numbers.json", doc)
    pop, hdr = B.load_required_manifest(p1, PD, requests=[f"r{k}" for k in range(8)], steps=max(B8_CAP), arrivals={f"r{k}": 0 for k in range(8)})
    assert pop and hdr["engine_steps"] == [] and hdr["engine_step_numbers"] == [0, 1, 2]
    res = B.composition_check(hdr, {"entries": []})
    assert res["ok"] and res["checked_steps"] == 0 and "no engine-step composition" in res["skipped"]
    # ... and the global (dict) form still normalises to the composition the check holds the map to
    p2, doc2 = _b8_doc(tmp_path)
    _pop2, hdr2 = B.load_required_manifest(p2, PD, requests=[f"r{k}" for k in range(8)], steps=max(B8_CAP), arrivals={f"r{k}": 0 for k in range(8)})
    assert hdr2["engine_steps"] and hdr2["engine_step_numbers"] == [int(es["engine_step"]) for es in doc2["engine_steps"]]
