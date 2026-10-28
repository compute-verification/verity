"""The POD consumer (LIFTING-SPEC v0.7 B9.1 (5)): `NativeHostCommitter.finalize` materialises the inactive rows:
padding steps registered before the run-root fold (`add_padding_steps`) become steps of the ONE run root -- a step nobody executed is
appended as a step of its own; a MIXED step (B > 1) is REFOLDED from the committer's retained leaf digests + the padding leaves, no
active byte re-read.  Openings of executed and padding leaves verify against the committer's original roots with the committer's
`verify()`; a tampered padding word does not.  The pod's padded map (`padded_binding_map_of_committer`) binds the FULL cap population.
Torch-free: the host (unpacked) committer path -- `commit_offline` is the exact hashing / layout / root code of the worker thread."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity_vllm.commit import binding as B
from verity_vllm.commit import padding_steps as PS
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit import scheme

PDB = hashlib.sha256(b"pod-consumer-test-program").digest()
PD = PDB.hex()
CHUNK = 256
CAP = 5
OPS = [("model.embed_tokens", "in", "i32", 1, "input_token_ids"), ("model.layers.0.input_layernorm", "out", "bf16", 64, "instance_outputs"),
       ("model.layers.0.self_attn.qkv_proj", "0", "bf16", 96, "instance_outputs"), ("logits_processor", "out", "bf16", 128, "instance_outputs"),
       ("runner.sampler", "sampled_token_ids", "i64", 1, "sampled_token_ids")]
NB = {"i32": 4, "bf16": 2, "u32": 4, "i64": 8}


def _identities(requests: list[str]) -> list[dict]:
    ids = []
    for rid in requests:
        for s in range(CAP):
            for op, mem, dt, n, fam in OPS:
                d = {"family": fam, "program_digest": PD, "request_id": rid, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                     "element_range": [0, n], "rank": 0, "engine_step": s, "dtype": dt}
                if fam == "input_token_ids":
                    d["rows"] = 1
                ids.append(d)
    return ids


def _executed(present: dict[int, list[str]]) -> list[list[tuple[str, bytes]]]:
    """Per executed engine step, the committed tensors (name, bytes): one row per present request, deterministic non-⊥ payloads."""
    out = []
    for s in sorted(present):
        rows = len(present[s])
        out.append([(PS.tensor_name(op, mem), bytes((11 + i + s) % 251 for i in range(n * NB[dt] * rows))) for op, mem, dt, n, _f in OPS])
    return out


def _committer() -> NativeHostCommitter:
    return NativeHostCommitter(None, run_id="run", chunk=CHUNK, hash_threads=1, program_digest=PDB)


def _ids_at(ids: list[dict], rid: str, step: int) -> list[dict]:
    return [d for d in ids if d["request_id"] == rid and d["step"] == step]


def _cov(ids: list[dict], pm: dict, rid: str) -> tuple[int, int]:
    req = [B.identity(d["program_digest"], d["request_id"], d["step"], d["invocation"], d["op_path"], d["output_member"], *d["element_range"], d["rank"])
           for d in ids if d["request_id"] == rid]
    cov = B.coverage_check(req, pm)
    return cov["bound_n"], cov["expected_n"]


# ---------------------------------------------------------------------- #3: B1 early stop -- the cap suffix appended, ONE run root

def test_b1_early_stop_the_padding_steps_are_steps_of_the_one_run_root_and_their_openings_verify():
    ids = _identities(["r0"])
    executed = {0: ["r0"], 1: ["r0"]}                                  # executed prefix of record = 2 of the cap 5
    com = _committer()

    def padding(c: NativeHostCommitter) -> list[PS.PaddingStep]:
        return [PS.padding_step_commit(s, _ids_at(ids, "r0", s), program_digest=PDB, run_id="run", geo_digest=c._geo_digest, chunk=CHUNK,
                                       layout="host-pos-leaf") for s in range(len(executed), CAP)]
    run = com.commit_offline(_executed(executed), padding=padding)
    plain = _committer().commit_offline(_executed(executed))

    assert len(run.steps) == CAP and len(plain.steps) == 2
    assert [s.root for s in run.steps[:2]] == [s.root for s in plain.steps], "the executed steps' roots are untouched by the padding"
    assert run.run_root != plain.run_root
    assert run.run_root == scheme.run_root(PDB, com._geo_digest, [s.root for s in run.steps]), "finalize's fold over ALL S = cap steps"
    for s in range(2, CAP):
        ps = com._padding[s]
        assert run.steps[s].root == ps.root and run.steps[s].n_leaves == ps.n_leaves and run.steps[s].n_bytes == ps.n_bytes
        assert run.steps[s].layout_digest == PS.layout_digest(ps.metas) == com.layout_digest(com._layouts[s])
    # openings: an executed leaf, the first and last padding leaves -- through the committer's own open() / verify()
    sample = [(1, 0), (2, 0), (CAP - 1, run.steps[CAP - 1].n_leaves - 1)]
    ops = com.open(sample)
    assert all(com.verify(run, o) for o in ops)
    pad_ops = [o for o in ops if o.step >= 2]
    assert all(o.identity.get("padding") for o in pad_ops) and not ops[0].identity.get("padding")
    # every padding byte opened is a canonical ⊥ word of its tensor's container (or the block's zero fill past the tensor)
    for o in pad_ops:
        t = o.identity["tensors"][0]
        dt = next(d["dtype"] for d in _ids_at(ids, "r0", o.step) if PS.padding_tensor_name(d["op_path"], d["output_member"], "r0") == t["name"])
        word = PS.bot_word(dt)
        body = o.value[t["first_byte"]:t["last_byte"]]
        assert len(body) % len(word) == 0 and set(body[i:i + len(word)] for i in range(0, len(body), len(word))) == {word}
    # a tampered padding word (a non-⊥ payload smuggled into the inactive suffix) fails against the original root
    bad = pad_ops[0]
    bad.value = bytes([bad.value[0] ^ 0x01]) + bad.value[1:]
    assert not com.verify(run, bad)
    # the pod's padded map binds the FULL cap population; the executed prefix is 'active', the suffix 'bot-by-enablement'
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r0")
    assert pm["schema"] == PS.PADDED_MAP_SCHEMA and pm["run_root"] == run.run_root.hex()
    assert _cov(ids, pm, "r0") == (CAP * len(OPS), CAP * len(OPS))
    acts = {s: {e["activity"] for e in pm["entries"] if e["identity"]["step"] == s} for s in range(CAP)}
    assert acts == {0: {"active"}, 1: {"active"}, 2: {"bot-by-enablement"}, 3: {"bot-by-enablement"}, 4: {"bot-by-enablement"}}
    assert all(e["step_root"] == run.steps[e["identity"]["step"]].root.hex() for e in pm["entries"] if e["kind"] == "leaf")
    assert all(e["encoding"]["bot_word_hex"] == PS.bot_word(e["encoding"]["committed_dtype"]).hex() for e in pm["entries"] if e["activity"] == "bot-by-enablement")


# ---------------------------------------------------------------------- #4: B2 mixed stop -- the mixed step REFOLDED to one root

def test_b2_mixed_stop_the_mixed_steps_are_refolded_from_retained_leaf_digests_and_both_kinds_of_leaf_open():
    ids = _identities(["r0", "r1"])
    executed = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"], 3: ["r1"], 4: ["r1"]}   # r0's executed prefix = 2; r1 runs to the cap
    com = _committer()
    plain = _committer().commit_offline(_executed(executed))

    def padding(c: NativeHostCommitter) -> list[PS.PaddingStep]:
        out = []
        for s in range(2, CAP):                                        # mixed: r1 executed the step, r0's rows are padding appended to it
            sc = c._done[s]
            out.append(PS.padding_step_commit(s, _ids_at(ids, "r0", s), program_digest=PDB, run_id="run", geo_digest=c._geo_digest, chunk=CHUNK,
                                              leaf_base=sc.n_leaves, ordinal_base=len(c._layouts[s]), layout="host-pos-leaf"))
        return out
    run = com.commit_offline(_executed(executed), padding=padding)

    assert len(run.steps) == CAP == len(plain.steps)
    assert [s.root for s in run.steps[:2]] == [s.root for s in plain.steps[:2]]
    for s in range(2, CAP):
        ps, sc, pl = com._padding[s], run.steps[s], plain.steps[s]
        assert ps.appended and ps.leaf_base == pl.n_leaves
        assert sc.n_leaves == pl.n_leaves + ps.n_leaves and sc.n_bytes == pl.n_bytes + ps.n_bytes and sc.root != pl.root
        # the ONE root: the executed leaf digests (retained, never re-hashed) + the padding leaves, bound over executed + padding metas
        exec_leaves = plain.steps[s] and _committer_levels(plain, s)
        leaves = exec_leaves + list(ps.levels[0])
        lv = scheme.fold_levels(leaves)
        assert sc.merkle_root == lv[-1][0]
        assert sc.root == scheme.step_root(PDB, PS.ctx_digest("run", s), com._geo_digest, com.layout_digest(com._layouts[s]), len(leaves), lv[-1][0])
        assert sc.layout_digest == com.layout_digest(com._layouts[s]) == PS.layout_digest(com._layouts[s])
    assert run.run_root == scheme.run_root(PDB, com._geo_digest, [s.root for s in run.steps])
    # openings at a mixed step: an executed leaf (r1's live row -- its path is through the REFOLDED tree) and a padding leaf (r0's ⊥ row)
    s = 3
    sample = [(s, 0), (s, run.steps[s].n_leaves - 1), (0, 1)]
    ops = com.open(sample)
    assert all(com.verify(run, o) for o in ops)
    assert not ops[0].identity.get("padding") and ops[1].identity.get("padding") and not ops[2].identity.get("padding")
    # cross-request poisoning: the padding leaf's tensors are r0's synthesized ones, disjoint from r1's executed tensors
    pad_names = {t["name"] for t in ops[1].identity["tensors"]}
    exec_ordinals = {m.ordinal for m in com._layouts[s] if m.stream_off < com._padding[s].stream_base}
    assert all(t["tensor"] not in exec_ordinals for t in ops[1].identity["tensors"]) and pad_names
    # tamper either kind at the mixed step -> the one root refuses
    for o in ops[:2]:
        o.value = bytes([o.value[0] ^ 0x01]) + o.value[1:]
        assert not com.verify(run, o)
    # the padded map: r0's population complete over the cap (its suffix by enablement), r1's complete (all active), one step root per step
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r1")   # executed rows bind r1 (single-request-per-row test map)
    assert _cov(ids, pm, "r0")[0] == 3 * len(OPS), "r0's inactive suffix (3 steps) is bound by enablement"
    assert all(e["step_root"] == run.steps[e["identity"]["step"]].root.hex() for e in pm["entries"] if e["kind"] == "leaf"), "mixed steps carry the refold, never null"
    assert pm["padding"]["mixed_steps"] == [2, 3, 4]


def _committer_levels(plain, s: int) -> list[bytes]:
    """Recompute the executed leaf digests of `plain`'s step s the way the committer retained them (host pos_leaf per tensor chunk)."""
    from verity_vllm.commit.scheme import pos_leaf

    c = _committer()
    c.commit_offline(_executed({0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"], 3: ["r1"], 4: ["r1"]}))
    assert c._done[s].root == plain.steps[s].root
    out = []
    for m in c._layouts[s]:
        raw = bytes(m.host)
        out.extend(pos_leaf(raw[o:o + CHUNK]) for o in range(0, len(raw), CHUNK))
    return out


# ---------------------------------------------------------------------- refusals: the rule never guesses a placement

def test_the_pod_consumer_refuses_a_misplaced_padding_step():
    ids = _identities(["r0"])
    executed = {0: ["r0"], 1: ["r0"]}
    com = _committer()
    geo = hashlib.sha256(b"cmt-integ/offline-geo/run").digest()
    # (a) a gap: padding for step 3 when step 2 is neither executed nor padded
    with pytest.raises(RuntimeError, match="contiguous"):
        com.commit_offline(_executed(executed), padding=[PS.padding_step_commit(3, _ids_at(ids, "r0", 3), program_digest=PDB, run_id="run",
                                                                                   geo_digest=geo, chunk=CHUNK, layout="host-pos-leaf")])
    # (b) appended to an executed step at the wrong leaf base (built against another committer's layout)
    with pytest.raises(RuntimeError, match="laid out after"):
        _committer().commit_offline(_executed(executed), padding=[PS.padding_step_commit(1, _ids_at(ids, "r0", 1), program_digest=PDB, run_id="run",
                                                                                            geo_digest=geo, chunk=CHUNK, leaf_base=1, layout="host-pos-leaf")])
    # (c) a whole padding step that claims to be appended
    ps = PS.padding_step_commit(2, _ids_at(ids, "r0", 2), program_digest=PDB, run_id="run", geo_digest=geo, chunk=CHUNK, leaf_base=4, layout="host-pos-leaf")
    with pytest.raises(RuntimeError, match="appended"):
        _committer().commit_offline(_executed(executed), padding=[ps])
    # (d) the same step twice
    ps2 = PS.padding_step_commit(2, _ids_at(ids, "r0", 2), program_digest=PDB, run_id="run", geo_digest=geo, chunk=CHUNK, layout="host-pos-leaf")
    with pytest.raises(ValueError, match="twice"):
        _committer().commit_offline(_executed(executed), padding=[ps2, ps2])


# ---------------------------------------------------------------------- the hook, end to end (commit_delta --padded-finalize's path)

def test_padding_hook_end_to_end_b2_eos_stop_full_cap_coverage_context_and_population_openings():
    """`padding_for_committer` as `padding_hook` inside finalize (the pod call-site): the executed prefix from served tokens + declared EOS +
    declared LAG; r0 stops by EOS at its 2nd token (LAG 0 -> executed 2 of cap 5), r1 runs to the cap; the padded map covers the FULL cap
    population, its context (run root, every step root) is the run's, and openings sampled from the population -- padding leaves included --
    verify with the committer's verify()."""
    ids = _identities(["r0", "r1"])
    manifest = {"identities": ids, "program_digest": PD}
    requests = [{"request_id": "req-a", "index": 0, "max_tokens": CAP, "arrive_step": 0}, {"request_id": "req-b", "index": 1, "max_tokens": CAP, "arrive_step": 0}]
    served = {"req-a-1a2b3c4d": [17, 0], "req-b-5e6f7a8b": [5, 6, 7, 8, 9]}           # vLLM's internal id suffix; r0's 2nd served token is EOS 0
    executed = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"], 3: ["r1"], 4: ["r1"]}
    com = _committer()
    com.padding_hook = lambda c: PS.padding_for_committer(c, manifest=manifest, requests=requests, sampling={"max_tokens": CAP}, served_tokens=served,
                                                          stop={"eos": [0], "lag": 0})
    run = com.commit_offline(_executed(executed))
    assert com.padding_hook is None, "consumed by finalize"
    assert sorted(com._padding) == [2, 3, 4] and all(com._padding[s].appended for s in (2, 3, 4))
    assert len(run.steps) == CAP
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r1")
    # full cap population of both requests (r1 by the executed rows, r0 by 2 executed + 3 padded steps)
    req = [B.identity(d["program_digest"], d["request_id"], d["step"], d["invocation"], d["op_path"], d["output_member"], *d["element_range"], d["rank"])
           for d in ids if d["request_id"] == "r0" and d["step"] >= 2]
    cov = B.coverage_check(req, pm)
    assert cov["ok"] and cov["bound_n"] == 3 * len(OPS)
    assert B.map_context_check(pm, run) == (True, "PASS")
    # openings from the padding stratum, through the committer, against the run's original roots
    pad_entries = [e for e in pm["entries"] if e["activity"] == "bot-by-enablement"]
    sample = [(e["identity"]["step"], e["leaf_lo"]) for e in pad_entries[:6]]
    ops = com.open(sample)
    assert ops and all(com.verify(run, o) for o in ops) and all(o.identity["padding"] for o in ops)
    if hasattr(com, "leaf_retained"):                                   # a retaining committer (retain=host+exclude): a padding leaf is always held
        assert all(com.leaf_retained(s, i) for s, i in sample)
    # the hook refuses an unplaceable prefix by name (an EOS stop with no declared LAG) and synthesizes nothing
    com2 = _committer()
    com2.padding_hook = lambda c: PS.padding_for_committer(c, manifest=manifest, requests=requests, sampling={"max_tokens": CAP}, served_tokens=served,
                                                           stop={"eos": [0], "lag": None})
    with pytest.raises(RuntimeError, match="REFUSED"):
        com2.commit_offline(_executed(executed))


def test_the_padding_leaf_rule_is_the_committers_not_the_environment(monkeypatch):
    """The hook's padding steps take the committer's leaf rule: its GPU tree's declared `layout`, `host-pos-leaf` off the GPU tree."""
    monkeypatch.setenv("VERITY_LEAF_LAYOUT", "chunk-leaf-v2")
    kw = dict(manifest={"identities": _identities(["r0"]), "program_digest": PD}, sampling={"max_tokens": CAP}, stop={"eos": [0], "lag": 0},
              requests=[{"request_id": "req-a", "index": 0, "max_tokens": CAP, "arrive_step": 0}], served_tokens={"req-a": [17, 0]})
    host = PS.padding_for_committer(_committer(), **kw)
    assert sorted(ps.step for ps in host) == [2, 3, 4] and {ps.layout for ps in host} == {"host-pos-leaf"}
    gpu = NativeHostCommitter(None, run_id="run", chunk=CHUNK, hash_threads=1, program_digest=PDB, gpu_tree=True)
    assert {ps.layout for ps in PS.padding_for_committer(gpu, **kw)} == {"chunk-leaf-v1"}
    gpu.layout = "chunk-leaf-v2"
    v2 = PS.padding_for_committer(gpu, **kw)
    assert {ps.layout for ps in v2} == {"chunk-leaf-v2"} and all(PS.verify_opening(ps, 0) for ps in v2)


# ------------------------------------------------------------------------------------------------------------------ shared planes, rule (a)

LAYERS = 2
PLANE_WORDS = 32


def _shared_rows(requests: list[str], *, geometry: dict | None = None, members_geometry: dict | None = None) -> list[dict]:
    """The B>1 SHARED hidden launch rows of `build_global_manifest`: one per (engine step, layer), request `*`, the co-scheduled members listed."""
    out = []
    for s in range(CAP):
        for L in range(LAYERS):
            d = {"family": "fa2_hidden_m1_stream", "program_digest": PD, "request_id": "*", "requests": list(requests), "step": "*", "engine_step": s,
                 "invocation": 0, "op_path": f"fa2.m1/L{L}", "output_member": "m1", "element_range": [0, PLANE_WORDS], "rank": 0, "dtype": "u32"}
            if geometry is not None:
                d["geometry"] = dict(geometry)
            if members_geometry is not None:
                d["members_geometry"] = {k: dict(v) for k, v in members_geometry.items()}
            out.append(d)
    return out


def _executed_with_planes(present: dict[int, list[str]], *, heads: int | None = None) -> list[list[tuple[str, bytes]]]:
    """`heads`: the launch is a real decode one (`heads` slabs a sequence, one row each, NB 1 at the FA2 legacy tile), named and sized by
    its geometry over the step's present sequences; else the toy plane `m1` of PLANE_WORDS."""
    from verity_vllm.query.manifest.format import mat_total_words
    steps = _executed(present)
    for s, ts in zip(sorted(present), steps):
        hb = (heads or 0) * len(present[s])
        name, words = (f"g{hb}x1x1", mat_total_words(hb, 1, 1, 128, 64)) if heads else ("m1", PLANE_WORDS)
        ts.extend((f"fa2.m1/L{L}/{name}", bytes((3 + i + s + L) % 251 for i in range(words * 4))) for L in range(LAYERS))   # the launch plane, whole
    return steps


def test_f_r17b_64_rule_a_an_inactive_listed_member_of_a_shared_launch_plane_gets_its_own_bot_plane_at_mixed_and_idle_steps(tmp_path):
    """The probe: r0 EOS at its 2nd token, r1 at its 3rd, cap 5 -> step 2 MIXED (r1 launched alone),
    steps 3, 4 IDLE.  Rule (a): the plane of a listed member the executed prefix places outside the step is ⊥ by enablement -- one
    per-sequence ⊥ plane per (inactive member, layer): r0 at step 2 (2), both at steps 3 and 4 (4 + 4) = 10.  The pod's full-cap
    population (load_required_manifest as commit_delta loads it) is then COVERED on the shared class: 20 / 20, 0 missing, and the
    executed launch bindings are what they were (bound whole to the members the launch ran over).  The launch is a real decode one
    (4 heads a sequence): coverage holds it to its geometry over the members executing at the step."""
    import json
    from verity_vllm.query.manifest import format as RM
    words = RM.mat_total_words(4, 1, 1, 128, 64)                            # one member's share of the launch: its bot plane
    geo = {"HB": 8, "M": 1, "NB": 1, "BN": 128, "D": 64, "seqlen_k": CAP, "rows": 2, "n_seq": 2}
    ids = _identities(["r0", "r1"]) + [dict(r, output_member="g8x1x1", element_range=[0, RM.mat_total_words(8, 1, 1, 128, 64)])
                                       for r in _shared_rows(["r0", "r1"], geometry=geo)]
    manifest = {"schema": RM.SCHEMA, "program_digest": PD, "identities": ids, "manifest_digest": RM.manifest_digest(ids),
                "engine_steps": [{"engine_step": e, "rows": 2, "n_seq": 2, "members": [{"request_id": r, "step": e, "rows": 1} for r in ("r0", "r1")]}
                                 for e in range(CAP)]}
    requests = [{"request_id": "req-a", "index": 0, "max_tokens": CAP, "arrive_step": 0}, {"request_id": "req-b", "index": 1, "max_tokens": CAP, "arrive_step": 0}]
    served = {"req-a-1a2b": [17, 0], "req-b-5e6f": [5, 6, 0]}
    executed = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"]}
    com = _committer()
    com.padding_hook = lambda c: PS.padding_for_committer(c, manifest=manifest, requests=requests, sampling={"max_tokens": CAP}, served_tokens=served,
                                                          stop={"eos": [0], "lag": 0})
    run = com.commit_offline(_executed_with_planes(executed, heads=4))
    assert len(run.steps) == CAP and sorted(com._padding) == [2, 3, 4]
    planes = {s: sorted(m.name for m in ps.metas if m.name.startswith("fa2.m1/")) for s, ps in com._padding.items()}
    assert planes[2] == ["fa2.m1/L0/r0/g4x1x1/bot", "fa2.m1/L1/r0/g4x1x1/bot"], "the MIXED step: only the inactive member's plane is bot"
    assert planes[3] == planes[4] == ["fa2.m1/L0/r0/g4x1x1/bot", "fa2.m1/L0/r1/g4x1x1/bot", "fa2.m1/L1/r0/g4x1x1/bot", "fa2.m1/L1/r1/g4x1x1/bot"], \
        "IDLE steps: every member's"
    assert sum(len(v) for v in planes.values()) == 10
    # the bot plane is the plane's committed dtype in ITS lifted container (u32 -> lifted33/u64), canonical bot on every word
    for ps in com._padding.values():
        for m in ps.metas:
            if m.name.startswith("fa2.m1/"):
                assert m.dtype == "lifted33/u64" and m.committed_dtype == "u32" and m.elements == words and m.nbytes == words * 8
                body = bytes(ps.stream[m.stream_off - ps.stream_base:m.stream_off - ps.stream_base + m.nbytes])
                assert body == PS.bot_word("u32") * words
    rr = {0: [("r0", 0, 1), ("r1", 1, 2)], 1: [("r0", 0, 1), ("r1", 1, 2)], 2: [("r1", 0, 1)]}
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r0", request_rows=rr)
    # the pod's population under --padded-finalize: request_steps = arrival .. arrival + cap; the shared rows expand per listed member
    # (member `*`, held to the launch executing at the step; an inactive member to its bot plane)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    pop, hdr = B.load_required_manifest(str(path), PD, requests=["r0", "r1"], steps=CAP, ranks=[0], arrivals={"r0": 0, "r1": 0},
                                        request_steps={"r0": list(range(CAP)), "r1": list(range(CAP))})
    assert hdr["hidden_batch_rows_op_level"] == CAP * LAYERS
    shared_pop = [p for p in pop if str(p[4]).startswith("fa2.m1/")]
    assert len(shared_pop) == 2 * CAP * LAYERS == 20
    cov = B.coverage_check(shared_pop, pm)
    assert cov["ok"] and cov["bound_n"] == 20 and cov["missing_n"] == 0, cov.get("missing")
    # and the FULL population (per-request rows split by request_rows as commit_delta does + the shared class): covered
    cov_all = B.coverage_check(pop, pm)
    assert cov_all["ok"] and cov_all["missing_n"] == 0, cov_all.get("missing")
    # the map says which are which: the bot planes carry activity / shared_of / geometry_source; the launch planes bind to the members they ran over
    bot_planes = [e for e in pm["entries"] if e.get("activity") == "bot-by-enablement" and str(e["name"]).startswith("fa2.m1/")]
    assert len(bot_planes) == 10 and all(e["shared_of"]["requests"] == ["r0", "r1"] and e["geometry_source"] == "launch-share" for e in bot_planes)
    assert sorted((e["identity"]["request_id"], e["identity"]["step"]) for e in bot_planes) == sorted([("r0", 2)] * 2 + [("r0", 3), ("r1", 3)] * 2 + [("r0", 4), ("r1", 4)] * 2)
    launch = [e for e in pm["entries"] if e.get("activity") != "bot-by-enablement" and str(e["name"]).startswith("fa2.m1/")]
    assert sorted({(e["identity"]["request_id"], e["identity"]["step"]) for e in launch}) == [("r0", 0), ("r0", 1), ("r1", 0), ("r1", 1), ("r1", 2)]
    # coverage_over_manifest (the conversion's rule) counts per member: 10 bound by the executed launch, 10 by a bot plane, 0 missing
    c2 = PS.coverage_over_manifest(ids, pm)
    assert c2["ok"] and c2["shared"] == {**c2["shared"], "n": 10, "bound_n": 10, "missing_n": 0, "members_n": 20, "members_bound_executed_n": 10,
                                          "members_bound_bot_n": 10, "members_missing_n": 0}
    # the bot planes OPEN against the one run root like every other padding leaf
    sample = [(e["identity"]["step"], e["leaf_lo"]) for e in bot_planes]
    ops = com.open(sample)
    assert len(ops) == 10 and all(com.verify(run, o) for o in ops) and all(o.identity["padding"] for o in ops)
    assert B.map_context_check(pm, run) == (True, "PASS")
    # the overhead of record sees them
    # step 2: r0's 5 per-request rows + its 2 planes; steps 3, 4: both members' 10 rows + 4 planes each
    assert sum(len(ps.metas) for ps in com._padding.values()) == (5 + 2) + 2 * (10 + 4) == 35
    if getattr(com, "padding_stats", None) is not None:                # the union branch's committer measures the overhead (A14.3 (g))
        assert com.padding_stats["padding_tensors"] == 35 and com.padding_stats["padding_bytes"] >= 10 * words * 8


def test_f_r17b_64_the_bot_plane_is_sized_by_the_members_own_geometry_declared_first_else_the_launch_share():
    """`member_plane_identity`: `members_geometry[rid]` (the manifest declares the member's single-sequence geometry) wins; else the
    launch geometry reduced to one sequence (HB / n_seq -- the decode-swapped launch packs n_seq x KVH head blocks); a launch HB that is
    not a multiple of n_seq is refused by name; a row with neither sizes by its own width (fixtures)."""
    from verity_vllm.query.manifest.format import mat_total_words
    row = {"family": "fa2_hidden_m1_stream", "program_digest": PD, "request_id": "*", "requests": ["r0", "r1"], "step": "*", "engine_step": 7,
           "invocation": 0, "op_path": "fa2.m1/L3", "output_member": "g6x4x2", "element_range": [0, mat_total_words(6, 4, 2, 128, 64)], "rank": 0, "dtype": "u32",
           "geometry": {"HB": 6, "M": 4, "NB": 2, "BN": 128, "D": 64, "seqlen_k": 200, "rows": 2, "n_seq": 2}}
    m = PS.member_plane_identity(row, "r1", 7, arrival=3)
    assert m["geometry_source"] == "launch-share" and m["geometry"]["HB"] == 3 and m["geometry"]["n_seq"] == 1
    assert m["element_range"] == [0, mat_total_words(3, 4, 2, 128, 64)] and m["output_member"] == "r1/g3x4x2/bot"
    assert m["request_id"] == "r1" and m["engine_step"] == 7 and m["step"] == 4 and m["dtype"] == "u32" and m["shared_of"]["requests"] == ["r0", "r1"]
    assert PS.tensor_name(m["op_path"], m["output_member"]) == "fa2.m1/L3/r1/g3x4x2/bot"
    decl = dict(row, members_geometry={"r1": {"HB": 3, "M": 4, "NB": 1, "BN": 128, "D": 64}})
    m2 = PS.member_plane_identity(decl, "r1", 7, arrival=3)
    assert m2["geometry_source"] == "declared" and m2["element_range"] == [0, mat_total_words(3, 4, 1, 128, 64)] and m2["output_member"] == "r1/g3x4x1/bot"
    with pytest.raises(ValueError, match="not a multiple"):
        PS.member_plane_identity(dict(row, geometry=dict(row["geometry"], HB=5)), "r1", 7, arrival=0)
    bare = {k: v for k, v in row.items() if k != "geometry"}
    m3 = PS.member_plane_identity(bare, "r0", 7, arrival=0)
    assert m3["geometry_source"] == "row-width" and m3["element_range"] == row["element_range"] and m3["output_member"] == "r0/g6x4x2/bot"


def test_the_padding_entrys_step_is_the_engine_step_when_a_member_arrives_late():
    """The map's step axis is the ENGINE step (the loader rebases request-relative rows by arrival): a member arriving at engine step 1
    with cap 3 has request steps 0..2 = engine steps 1..3; stopped after its 1st token (LAG 0) its padding is at engine steps 2, 3 --
    entries keyed there, `request_step` beside."""
    import json
    cap = 3
    ids = []
    for rid, arr in (("r0", 0), ("r1", 1)):
        for s in range(cap):
            for op, mem, dt, n, fam in OPS[:2]:
                ids.append({"family": fam, "program_digest": PD, "request_id": rid, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                            "element_range": [0, n], "rank": 0, "engine_step": s + arr, "dtype": dt, **({"rows": 1} if fam == "input_token_ids" else {})})
    epr = {"ok": True, "lag": 0, "requests": {"r0": {"executed_prefix_len": cap, "cap": cap}, "r1": {"executed_prefix_len": 1, "cap": cap}}}
    from verity_vllm.commit.executed_prefix import request_steps_of_record
    steps = request_steps_of_record(epr, {"r0": 0, "r1": 1})
    assert steps == {"r0": [0, 1, 2], "r1": [1]}
    by_step = PS.padding_steps_of_record(ids, epr, {"r0": 0, "r1": 1})
    assert sorted(by_step) == [2, 3] and all(d["request_id"] == "r1" and d["engine_step"] == es and d["step"] == es - 1 for es, ds in by_step.items() for d in ds)
    ps = PS.padding_step_commit(3, by_step[3], program_digest=PDB, run_id="run", geo_digest=b"\0" * 32, chunk=CHUNK, layout="host-pos-leaf")
    ents = PS.padding_entries(ps, layout_version="host-pos-leaf")
    assert all(e["identity"]["step"] == 3 and e["engine_step"] == 3 and e["request_step"] == 2 for e in ents)


# ------------------------------------------------------------------------------------------------- the two pod refusals

def _b2_mixed_padded(tmp_path=None):
    """The pod fixture of the shared-planes test: r0 EOS at its 2nd token, r1 at its 3rd, cap 5 (LAG 0) -> step 2 MIXED, steps 3, 4 IDLE."""
    from verity_vllm.query.manifest import format as RM
    ids = _identities(["r0", "r1"]) + _shared_rows(["r0", "r1"])
    manifest = {"schema": RM.SCHEMA, "program_digest": PD, "identities": ids, "manifest_digest": RM.manifest_digest(ids)}
    requests = [{"request_id": "req-a", "index": 0, "max_tokens": CAP, "arrive_step": 0}, {"request_id": "req-b", "index": 1, "max_tokens": CAP, "arrive_step": 0}]
    served = {"req-a-1a2b": [17, 0], "req-b-5e6f": [5, 6, 0]}
    executed = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"]}
    com = _committer()
    com.padding_hook = lambda c: PS.padding_for_committer(c, manifest=manifest, requests=requests, sampling={"max_tokens": CAP}, served_tokens=served,
                                                          stop={"eos": [0], "lag": 0})
    # the two composition-reference members are [n_seq] id tensors on the pod (the row split of build_binding_map needs the shape); each step
    # root binds its layout (every meta's dtype and shape), so they are committed typed -- a meta changed after the commit no longer opens
    typed = {"model.embed_tokens/in": "int32", "runner.sampler/sampled_token_ids": "int64"}
    steps = _executed_with_planes(executed)
    for s, ts in zip(sorted(executed), steps):
        ts[:] = [(n, raw, typed[n], (len(executed[s]),)) if n in typed else (n, raw) for n, raw in ts]
    run = com.commit_offline(steps)
    rr = {0: [("r0", 0, 1), ("r1", 1, 2)], 1: [("r0", 0, 1), ("r1", 1, 2)], 2: [("r1", 0, 1)]}
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r0", request_rows=rr)
    hdr = {"engine_steps": [{"engine_step": e, "rows": 2, "n_seq": 2, "members": [{"request_id": r, "step": e, "rows": 1} for r in ("r0", "r1")]} for e in range(CAP)],
           "composition_reference": {"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                                     "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids"}}}
    rsteps = {"r0": [0, 1], "r1": [0, 1, 2]}
    return com, run, pm, hdr, rsteps, ids


def test_f_r17b_66_the_composition_check_tiles_the_executed_member_by_the_active_members_and_the_padding_is_its_own_row_space():
    """The r0 padding block, a tensor of its own with element_offset 0, is never tiled INTO r1's live member ('r1 rows [0,1) overlaps
    the previous block ending at 1'): the executed member is tiled by the ACTIVE
    members only (cursor == live rows), a bot-by-enablement entry is its own row space (one per absent request, declared rows), the UNION
    equals the declared cap composition (read unrestricted), and the four named refusals: double binding, padding inside the prefix,
    executed block past the stop, padding for no declared member."""
    import copy
    com, run, pm, hdr, rsteps, _ids = _b2_mixed_padded()
    comp = B.composition_check(hdr, pm, request_steps=rsteps)
    assert comp["ok"], comp["problems"]
    assert comp["checked_steps"] == CAP and comp["padded"]["steps_with_padding"] == 3 and comp["padded"]["padding_blocks"] == 1 + 2 + 2
    assert comp["padded"]["declared_composition_unrestricted"] and "executed_prefix_restriction" not in comp
    # the same map read by the pre-fix logic (padding tiled into the live member) is what the pod refused: shown by stripping `activity`
    naive = copy.deepcopy(pm)
    for e in naive["entries"]:
        e.pop("activity", None)
    bad = B.composition_check(hdr, naive, request_steps=rsteps)
    assert not bad["ok"] and any("overlaps" in str(p.get("overlaps")) for p in bad["problems"]), "the pod's refusal, reproduced"
    # (1) double binding: r1 has an executed block at step 2 AND a padding entry relabelled to it
    dbl = copy.deepcopy(pm)
    e2 = next(e for e in dbl["entries"] if e.get("activity") == "bot-by-enablement" and e["identity"]["step"] == 2 and e["identity"]["op_path"] == "model.embed_tokens")
    e2["identity"]["request_id"] = "r1"
    c1 = B.composition_check(hdr, dbl, request_steps=rsteps)
    assert not c1["ok"] and c1["problems"][0]["engine_step"] == 2 and any("double binding" in x for x in c1["problems"][0]["padding"]), c1["problems"][0]
    # (2) padding inside the executed prefix: the record says r0 executed step 2
    c2 = B.composition_check(hdr, pm, request_steps={"r0": [0, 1, 2], "r1": [0, 1, 2]})
    assert not c2["ok"] and any("inside the executed prefix" in x for x in c2["problems"][0]["padding"]), c2["problems"][0]
    # (3) an executed block past the stop: the record says r1 stopped before step 2
    c3 = B.composition_check(hdr, pm, request_steps={"r0": [0, 1], "r1": [0, 1]})
    assert not c3["ok"] and any("past the stop" in x for x in c3["problems"][0]["padding"]), c3["problems"][0]
    # (4) a declared member with neither an executed block nor a padding entry at an idle step
    gone = copy.deepcopy(pm)
    gone["entries"] = [e for e in gone["entries"] if not (e.get("activity") == "bot-by-enablement" and e["identity"]["step"] == 3
                                                          and e["identity"]["op_path"] == "model.embed_tokens" and e["identity"]["request_id"] == "r0")]
    c4 = B.composition_check(hdr, gone, request_steps=rsteps)
    assert not c4["ok"] and c4["problems"][0]["engine_step"] == 3 and c4["problems"][0]["missing"] == ["r0 x1"], c4["problems"][0]
    # (5) the sampled-ids slots: executed slots over the active members, the padding slots counted beside -- a padding slot relabelled fails
    smp = copy.deepcopy(pm)
    s3 = next(e for e in smp["entries"] if e.get("activity") == "bot-by-enablement" and e["identity"]["step"] == 3 and e["identity"]["op_path"] == "runner.sampler"
              and e["identity"]["request_id"] == "r0")
    s3["identity"]["request_id"] = "r1"
    c5 = B.composition_check(hdr, smp, request_steps=rsteps)
    assert not c5["ok"] and c5["problems"][0]["kind"] == "sampled-slots" and c5["problems"][0]["padding_slots"] == ["r1", "r1"], c5["problems"][0]


def test_the_inactive_stratum_is_the_padded_records_population_every_bot_entry_read_back_equals_the_bot_word_and_the_families_reconcile():
    """On a padded record the population is the FULL cap (never 'the Commit classified 0 not-executed, the population excluded
    8,274'); the identities the replay excluded at n_exec are the
    named stratum `inactive_by_enablement` (not_executed_by_rule n 0 on both sides, RENAMED, never dropped); every bot-by-enablement entry
    is read back through the committed reader (`<name>#ord<ordinal>`: a padding tensor beside an executed one of the same name at a mixed
    step) and checked word for word == bot_word(committed_dtype); the one-mechanism reconciliation compares the Commit's padding entries
    against that bucket family by family."""
    from collections import Counter
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.commit.executed_prefix import executed_prefix_of_record
    if not hasattr(OC, "ORDINAL_SEP"):
        pytest.skip("the ordinal-named committed reader (moe F-moe-r17-06, int-8) is not on this tree: a mixed step's padding tensor beside an executed one of the same name is unreadable here")
    com, run, pm, _hdr, _rsteps, _ids = _b2_mixed_padded()
    pads = [e for e in pm["entries"] if e.get("activity") == "bot-by-enablement"]
    fam = Counter(e["family"] for e in pads)
    assert set(fam) == {"input_token_ids", "instance_outputs", "sampled_token_ids", "fa2_hidden_m1_stream"} and sum(fam.values()) == 35
    assert fam["fa2_hidden_m1_stream"] == 10, "rule (a): r0 x2 layers at the mixed step + both x2 layers at the 2 idle steps"
    # the replay's population shape (sampled_replay 1650-1690 / 1861-1870): the per-request families are counted by identity; the BATCH-LEVEL
    # shared family counts a plane inactive only when EVERY listed member is past its stop (the 2 idle steps x 2 layers = 4 planes) and
    # states the (plane, past member) CELLS beside it (mixed 2 + idle 8 = 10) -- the number the Commit's per-member bot planes are
    pop_fam = dict(fam, fa2_hidden_m1_stream=4)
    totals = {"N_manifest": 2 * CAP * 5 + CAP * LAYERS, "N_excluded": sum(pop_fam.values()),
              "by_family": {f: {"N_manifest": 0, "N_required": 0, "N_excluded": n} for f, n in pop_fam.items()}}
    totals["by_family"]["fa2_hidden_m1_stream"]["plane_cells_not_executed"] = {"identities": 4, "cells": 10}
    sr = {"population": {"not_executed_by_rule": {"identities": sum(pop_fam.values()), "by_family": dict(pop_fam), "totals": totals, "scope": "scope text", "applied": True}},
          "not_executed_rule": {"n": sum(pop_fam.values()), "requests": {"r0": {"n_exec": 2, "a_r": 2, "cap": CAP}, "r1": {"n_exec": 3, "a_r": 3, "cap": CAP}}},
          "not_executed_by_family": dict(pop_fam)}
    epr = executed_prefix_of_record(["r0", "r1"], {"r0": [17, 0], "r1": [5, 6, 0]}, caps={"r0": CAP, "r1": CAP}, eos_of_record=[0], lag=0)
    rd = OC.OpenedReader(com, run)
    out = PS.padded_population(sr, pm, rd, epr)
    assert hashlib.sha256(json.dumps([out, sr], sort_keys=True).encode()).hexdigest() == \
        "a5d7a1f4f51030d83a6cbf083ef69ae44b34e81d91e486a70595a178a3aacd74", "the record block and the population it rewrites, byte for byte"
    assert out["ok"], out["problems"]
    assert out["padding_entries"] == out["checked_entries"] == 35 and out["unread_n"] == 0 and out["mismatches_n"] == 0
    assert out["checked_words"] == sum(int(e["width_bytes"]) // PS.LIFTED_CONTAINER[e["encoding"]["committed_dtype"]][1] for e in pads)
    assert all(d["commit_padding_entries"] == d["population_inactive"] for d in out["by_family"].values())
    # the shared-plane family's bot planes state HOW they were sized, on the record: this fixture's shared rows
    # carry neither members_geometry nor a launch geometry -> every plane `row-width` (the fixture-only fallback), counted and labelled, not 'declared'
    sizing = out["by_family"]["fa2_hidden_m1_stream"].pop("bot_plane_sizing")
    assert sizing["planes"] == 10 and sizing["geometry_source"] == {"row-width": 10} and sizing["all_declared"] is False
    assert sizing["plane_geometry"] == {"m1": 10} and sizing["words_per_plane"] == {str(PLANE_WORDS): 10} and sizing["rule"] == PS.SIZING_RULE
    assert out["by_family"]["fa2_hidden_m1_stream"] == {"commit_padding_entries": 10, "population_inactive": 10, "basis": "cells",
                                                         "population_inactive_plane_identities": 4, "population_planes_fully_inactive": 4,
                                                         "commit_kind": "shared-plane (rule a)"}
    assert out["by_family"]["instance_outputs"]["basis"] == "identities" and out["by_family"]["instance_outputs"]["commit_kind"] == "per-request"
    assert "bot_plane_sizing" not in out["by_family"]["instance_outputs"], "a per-request family has no planes to size"
    # a manifest that DECLARES members_geometry[rid] (reqval's ask) reads `declared` for every plane -> all_declared True; a mix is counted apart
    decl = [{"family": "fa2_hidden_m1_stream", "geometry_source": "declared", "identity": {"output_member": "r0/g3x4x1/bot", "element_range": [0, 12]}} for _ in range(3)]
    assert PS.bot_plane_sizing(decl) == {"planes": 3, "geometry_source": {"declared": 3}, "all_declared": True, "plane_geometry": {"g3x4x1": 3},
                                         "words_per_plane": {"12": 3}, "rule": PS.SIZING_RULE}
    mixed = decl + [{"family": "fa2_hidden_m1_stream", "geometry_source": "launch-share", "identity": {"output_member": "r1/g3x3x1/bot", "element_range": [0, 9]}}]
    assert PS.bot_plane_sizing(mixed)["geometry_source"] == {"declared": 3, "launch-share": 1} and PS.bot_plane_sizing(mixed)["all_declared"] is False
    assert PS.bot_plane_sizing([]) == {"planes": 0, "geometry_source": {}, "all_declared": False, "plane_geometry": {}, "words_per_plane": {}, "rule": PS.SIZING_RULE}
    assert out["n_exec"] == {"r0": {"commit": 2, "population": 2}, "r1": {"commit": 3, "population": 3}}
    # the population is rewritten in place: the class is RENAMED, not dropped
    assert sr["population"]["inactive_by_enablement"]["identities"] == 29 and sr["population"]["inactive_by_enablement"]["by_family"] == dict(pop_fam)
    assert sr["population"]["not_executed_by_rule"] == {**sr["population"]["not_executed_by_rule"], "identities": 0, "n": 0, "applied": False, "renamed_to": "inactive_by_enablement"}
    assert sr["not_executed_rule"]["n"] == 0 and sr["not_executed_by_family"] == {} and sr["inactive_by_family"] == dict(pop_fam)
    # a population that stated no cells for the shared family would compare planes (4) to bot planes (10) -- named
    no_cells = {"population": {"not_executed_by_rule": {"identities": 29, "by_family": dict(pop_fam)}}, "not_executed_rule": {"requests": sr["not_executed_rule"]["requests"]}}
    nc = PS.padded_population(no_cells, pm, rd, epr)
    assert nc["by_family"]["fa2_hidden_m1_stream"].pop("bot_plane_sizing")["geometry_source"] == {"row-width": 10}
    assert not nc["ok"] and nc["by_family"]["fa2_hidden_m1_stream"] == {"commit_padding_entries": 10, "population_inactive": 4, "basis": "identities", "commit_kind": "shared-plane (rule a)"}
    assert any("family fa2_hidden_m1_stream: the Commit padded 10 entries, the population names 4 inactive identities" == p for p in nc["problems"])
    # the mixed step's padding tensor `model.embed_tokens` sits beside r1's executed tensor of the SAME name: the bare-name read is refused
    # (two metas), the ordinal read returns the padding bytes -- and they are the bot word, not r1's committed id
    e2 = next(e for e in pads if e["identity"]["step"] == 2 and e["identity"]["op_path"] == "model.embed_tokens")
    assert e2["name"] == "model.embed_tokens/in/r0/bot", "a padding tensor is its OWN tensor: named apart from the executed member (first #4 pod run: 24/44 linked)"
    assert rd(2, e2["name"], 0, e2["width_bytes"]) == rd(2, f"{e2['name']}{OC.ORDINAL_SEP}{e2['tensor_ordinal']}", 0, e2["width_bytes"]) == PS.bot_word("i32")
    # ... and the EXECUTED member's bare name at the mixed step names ONE meta again: r1's committed id, what the C2 boundary linkage reads
    x2 = next(e for e in pm["entries"] if e.get("activity") == "active" and e["identity"]["step"] == 2 and e["identity"]["op_path"] == "model.embed_tokens")
    assert x2["name"] == "model.embed_tokens/in" and x2["identity"]["request_id"] == "r1"
    got = rd(2, x2["name"], 0, 4)
    assert got is not None and got != PS.bot_word("i32")[:4] and len(got) == 4
    # a padding entry whose committed word is NOT the bot word: named
    def tampered(step, name, lo, hi):
        got = rd(step, name, lo, hi)
        if got is not None and step == 3 and name.startswith("fa2.m1/L0/r0/"):
            return bytes(got[:8]) + b"\x01" + bytes(got[9:])
        return got
    bad = PS.padded_population({"population": {"not_executed_by_rule": {"identities": 29, "by_family": dict(pop_fam), "totals": totals}},
                                "not_executed_rule": {"requests": sr["not_executed_rule"]["requests"]}}, pm, tampered, epr)
    assert not bad["ok"] and bad["mismatches_n"] == 1 and bad["mismatches"][0]["first_bad_word"] == 1 and "NOT the bot word" in bad["problems"][0]
    # the population naming a different count for a family: the one-mechanism reconciliation disagrees by name
    fam2 = dict(pop_fam, instance_outputs=pop_fam["instance_outputs"] - 1)
    dis = PS.padded_population({"population": {"not_executed_by_rule": {"identities": 28, "by_family": fam2, "totals": totals}},
                                "not_executed_rule": {"requests": sr["not_executed_rule"]["requests"]}}, pm, rd, epr)
    assert not dis["ok"] and any("family instance_outputs" in p for p in dis["problems"])


# ------------------------------------------------------------------------------------------ at the committed word: the MoE planes past a stop

MOE_MEMBERS = [("router_ids", "i32", 8), ("router_w", "f32", 8), ("act", "bf16", 64)]     # OLMoE's moe_block_stream members per (layer, token): 8 routes


def _moe_identities(requests: list[str]) -> list[dict]:
    """The dense identities + a per-request `moe_block_stream` row per (step, member) of ONE MoE layer (the OLMoE B2 mixedstop manifest's shape:
    request-level rows `moe/L<l>` with element_range per request, A7 r17_1845)."""
    ids = _identities(requests)
    for rid in requests:
        for s in range(CAP):
            for mem, dt, n in MOE_MEMBERS:
                ids.append({"family": "moe_block_stream", "program_digest": PD, "request_id": rid, "step": s, "invocation": 0, "op_path": "moe/L0",
                            "output_member": mem, "element_range": [0, n], "rank": 0, "engine_step": s, "dtype": dt})
    return ids


def _executed_with_moe(present: dict[int, list[str]]) -> list[list[tuple[str, bytes]]]:
    steps = _executed(present)
    nb = {"i32": 4, "f32": 4, "bf16": 2}
    for s, ts in zip(sorted(present), steps):
        rows = len(present[s])
        for mem, dt, n in MOE_MEMBERS:
            # a live route: expert indices 0..7 (i32), weights, activations -- never a bot word
            if mem == "router_ids":
                payload = b"".join(int((i + s) % 64).to_bytes(4, "little") for i in range(n * rows))
            else:
                payload = bytes((23 + i + s) % 251 for i in range(n * nb[dt] * rows))
            ts.append((PS.tensor_name("moe/L0", mem), payload))
    return steps


def test_brief_7e_past_a_requests_stop_the_moe_route_planes_are_bot_words_and_a_smuggled_expert_index_fails_by_name():
    """An inactive continuation does not invent an expert route, at the COMMITTED word through the pod hook: on a B2 mixed stop
    (r0 stops by EOS at its 2nd token, LAG 0; r1 to the cap) the `moe_block_stream` planes of r0 past its stop are synthesized as bot-by-
    enablement tensors of their own (`moe/L0/router_ids/r0/bot` ...): router ids i32 -> lifted33/u64 bot, router weights f32 -> lifted33 bot,
    activations bf16 -> lifted17/u32 bot -- no expert index and no router weight exists as a committed value past the stop, only the bot word;
    r1's live route at the same mixed step is its own tensor by the bare name (ONE meta), untouched; the padding leaves open under the original
    root; a non-bot word in the route plane (an expert index smuggled into the inactive suffix) fails verify() and the word check by name.
    The seam: the moe planes' executed layouts at a mixed step beside r0's padding tensor -- the host committer half of it."""
    from verity_vllm.check import oracle_compare as OC
    ids = _moe_identities(["r0", "r1"])
    manifest = {"identities": ids, "program_digest": PD}
    requests = [{"request_id": "req-a", "index": 0, "max_tokens": CAP, "arrive_step": 0}, {"request_id": "req-b", "index": 1, "max_tokens": CAP, "arrive_step": 0}]
    served = {"req-a-1a2b3c4d": [17, 0], "req-b-5e6f7a8b": [5, 6, 7, 8, 9]}
    executed = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"], 3: ["r1"], 4: ["r1"]}
    com = _committer()
    com.padding_hook = lambda c: PS.padding_for_committer(c, manifest=manifest, requests=requests, sampling={"max_tokens": CAP}, served_tokens=served,
                                                          stop={"eos": [0], "lag": 0})
    run = com.commit_offline(_executed_with_moe(executed))
    assert sorted(com._padding) == [2, 3, 4] and all(com._padding[s].appended for s in (2, 3, 4)), "every padded step is MIXED: r1 live beside r0's bot"
    # the route planes past r0's stop: one bot tensor per (step, member), its own name, its lifted container, every word the bot word
    want = {"router_ids": ("lifted33/u64", PS.bot_word("i32")), "router_w": ("lifted33/u64", PS.bot_word("f32")), "act": ("lifted17/u32", PS.bot_word("bf16"))}
    for s in (2, 3, 4):
        ps = com._padding[s]
        moe = {m.name: m for m in ps.metas if m.name.startswith("moe/L0/")}
        assert sorted(moe) == ["moe/L0/act/r0/bot", "moe/L0/router_ids/r0/bot", "moe/L0/router_w/r0/bot"]
        for mem, dt, n in MOE_MEMBERS:
            m = moe[f"moe/L0/{mem}/r0/bot"]
            label, word = want[mem]
            assert m.dtype == label and m.block_kind == "padding" and m.nb == 1
            body = ps.stream[m.stream_off - ps.stream_base: m.stream_off - ps.stream_base + m.nbytes]
            assert len(body) == n * len(word) and set(body[i:i + len(word)] for i in range(0, len(body), len(word))) == {word}, f"{mem}: every word is bot"
    # r1's live route at the mixed step 2: the bare name names ONE meta -- r1's executed plane, expert indices, not bot words
    rd = OC.OpenedReader(com, run)
    live = rd(2, "moe/L0/router_ids", 0, 8 * 4)
    assert live is not None and len(live) == 32 and set(int.from_bytes(live[i:i + 4], "little") for i in range(0, 32, 4)) <= set(range(64))
    assert live[:8] != PS.bot_word("i32")
    # the padded map: r0's 9 moe bot entries (3 steps x 3 members) carry the family and the bot word; r1's 5 x 3 are active
    pm = PS.padded_binding_map_of_committer(com, run, PD, chunk=CHUNK, request_id="r1")
    moe_pad = [e for e in pm["entries"] if e.get("activity") == "bot-by-enablement" and e["identity"]["op_path"] == "moe/L0"]
    moe_live = [e for e in pm["entries"] if e.get("activity") == "active" and e["identity"]["op_path"] == "moe/L0"]
    assert len(moe_pad) == 3 * len(MOE_MEMBERS) and all(e["family"] == "moe_block_stream" and e["identity"]["request_id"] == "r0" for e in moe_pad)
    assert {e["identity"]["step"] for e in moe_pad} == {2, 3, 4} and len(moe_live) == CAP * len(MOE_MEMBERS)
    assert all(e["encoding"]["bot_word_hex"] == PS.bot_word(e["encoding"]["committed_dtype"]).hex() for e in moe_pad)
    suffix = [B.identity(d["program_digest"], d["request_id"], d["step"], d["invocation"], d["op_path"], d["output_member"], *d["element_range"], d["rank"])
              for d in ids if d["request_id"] == "r0" and d["step"] >= 2]
    assert B.coverage_check(suffix, pm)["bound_n"] == 3 * (len(OPS) + len(MOE_MEMBERS)), "r0's inactive suffix: dense + moe, every identity bound by enablement"
    assert _cov(ids, pm, "r1") == (CAP * (len(OPS) + len(MOE_MEMBERS)),) * 2, "r1 complete over the cap (executed rows bind r1 in this test map)"
    # the route planes open under the ORIGINAL root; a smuggled expert index (a non-bot word in the inactive route plane) fails
    ops = com.open([(e["identity"]["step"], e["leaf_lo"]) for e in moe_pad if e["identity"]["output_member"] == "router_ids"])
    assert len(ops) == 3 and all(com.verify(run, o) for o in ops) and all(o.identity["padding"] for o in ops)
    route = next(e for e in moe_pad if e["identity"]["step"] == 2 and e["identity"]["output_member"] == "router_ids")
    o = com.open([(2, route["leaf_lo"])])[0]
    t = next(t for t in o.identity["tensors"] if t["name"] == "moe/L0/router_ids/r0/bot")
    forged = bytearray(o.value)
    forged[t["first_byte"]:t["first_byte"] + 8] = (3).to_bytes(8, "little")            # expert 3, active-tagged: a route where none executed
    o.value = bytes(forged)
    assert not com.verify(run, o), "the original root refuses the invented route"
    if hasattr(OC, "ORDINAL_SEP"):
        # the inactive-stratum word check (padded_population's reader) sees the committed route plane as the bot word, and names a forgery
        got = rd(2, f"{route['name']}{OC.ORDINAL_SEP}{route['tensor_ordinal']}", 0, route["width_bytes"])
        assert got == PS.bot_word("i32") * 8
        def forging(step, name, lo, hi):
            g = rd(step, name, lo, hi)
            return ((3).to_bytes(8, "little") + g[8:]) if (g is not None and name.startswith("moe/L0/router_ids/r0/bot")) else g
        from verity_vllm.commit.executed_prefix import executed_prefix_of_record
        epr = executed_prefix_of_record(["r0", "r1"], {"r0": [17, 0], "r1": [5, 6, 7, 8, 9]}, caps={"r0": CAP, "r1": CAP}, eos_of_record=[0], lag=0)
        pads = [e for e in pm["entries"] if e.get("activity") == "bot-by-enablement"]
        from collections import Counter
        fam = Counter(e["family"] for e in pads)
        sr = {"population": {"not_executed_by_rule": {"identities": sum(fam.values()), "by_family": dict(fam), "applied": True}},
              "not_executed_rule": {"requests": {"r0": {"n_exec": 2}, "r1": {"n_exec": CAP}}}}
        bad = PS.padded_population(sr, pm, forging, epr)
        assert not bad["ok"] and bad["mismatches_n"] == 3 and all("moe/L0/router_ids/r0/bot" in str(m) for m in bad["mismatches"])
        # the honest record states the dtypes it read the bot words back through, per committed dtype, with the count
        sr2 = {"population": {"not_executed_by_rule": {"identities": sum(fam.values()), "by_family": dict(fam), "applied": True}},
               "not_executed_rule": {"requests": {"r0": {"n_exec": 2}, "r1": {"n_exec": CAP}}}}      # padded_population renames in place
        good = PS.padded_population(sr2, pm, rd, epr)
        assert good["ok"] and good["checked_entries"] == good["padding_entries"] == len(pads), good["problems"]
        bd = good["by_dtype"]
        assert bd["i32"]["container"] == bd["f32"]["container"] == "lifted33/u64" and bd["bf16"]["container"] == "lifted17/u32"
        assert bd["i32"]["bot_word_hex"] == PS.bot_word("i32").hex() and bd["bf16"]["lifted_width_bits"] == 17 and bd["i32"]["lifted_width_bits"] == 33
        assert "router_ids" in bd["i32"]["members"] and bd["f32"]["members"] == ["router_w"] and "act" in bd["bf16"]["members"]
        assert sum(d["checked_entries"] for d in bd.values()) == good["checked_entries"] and all(d["checked_entries"] == d["padding_entries"] for d in bd.values())


# ---------------------------------------------------------------------- fix (e): X-03 identity check on a PADDED run (the second #3 pod run, b542ef5fc)

MAN = {"modules": sorted({op.split("/", 1)[0] for op, *_ in OPS}), "attention_layers": [], "fa2_hidden_per_layer": False, "sampler_ids": 1,
       "moe_layers": [], "moe_tensors_per_layer": 0, "tp_sites": [], "tp_rank": None}


def test_fix_e_the_identity_check_reads_a_padding_only_step_as_the_identities_of_step_0_materialized_as_bot_and_a_mixed_step_as_unpadded():
    """The padding tensors have names of their own, so X-03 (4) -- the multiset of module-tensor NAMES equal in every step -- would read a
    padding-only step as one that dropped every module and produced unknown ones ('identity check: 21 step(s) differ from the expected
    manifest').  A padding tensor materializes an identity, it is not a module call: a padding-only step passes iff it materializes every identity
    NAME of step 0 as bot (set equality) and covers every expected module; a mixed step's executed metas are checked exactly as before."""
    ids = _identities(["r0"])
    executed = {0: ["r0"], 1: ["r0"]}
    com = _committer()

    def padding(c):
        return [PS.padding_step_commit(s, _ids_at(ids, "r0", s), program_digest=PDB, run_id="run", geo_digest=c._geo_digest, chunk=CHUNK,
                                       layout="host-pos-leaf") for s in range(len(executed), CAP)]
    com.commit_offline(_executed(executed), padding=padding)
    ic = com.identity_check(manifest=MAN)
    assert ic["ok"] and ic["violations_n"] == 0 and ic["steps_checked"] == CAP, ic
    assert ic["padding"]["padding_only_steps"] == [2, 3, 4] and ic["padding"]["steps_with_padding"] == 3 and ic["padding"]["padding_tensors"] == 3 * len(OPS)
    # an unpadded run reads as before: no padding block, ok
    plain = _committer()
    plain.commit_offline(_executed(executed))
    icp = plain.identity_check(manifest=MAN)
    assert icp["ok"] and "padding" not in icp
    # a padding-only step that DROPS one identity's bot still fails, by name (never a relabel: the padding is checked, not waved through)
    dropped = next(m for m in com._layouts[3] if m.name.startswith("logits_processor/"))
    com._layouts[3] = [m for m in com._layouts[3] if m is not dropped]
    bad = com.identity_check(manifest=MAN)
    assert not bad["ok"] and bad["violations_n"] == 1 and bad["violations"][0]["step"] == 3 and bad["violations"][0]["padding_only"] is True
    assert bad["violations"][0]["missing_modules"] == ["logits_processor"] and bad["violations"][0]["multiplicity_vs_step0"] == {"logits_processor": (0, 1)}
    # B2 mixed: r1 executed steps 2..4, r0's bot tensors appended -- the executed metas alone match step 0's multiset; the padding counted apart
    ids2 = _identities(["r0", "r1"])
    executed2 = {0: ["r0", "r1"], 1: ["r0", "r1"], 2: ["r1"], 3: ["r1"], 4: ["r1"]}
    com2 = _committer()

    def padding2(c):
        return [PS.padding_step_commit(s, _ids_at(ids2, "r0", s), program_digest=PDB, run_id="run", geo_digest=c._geo_digest, chunk=CHUNK,
                                       leaf_base=c._done[s].n_leaves, ordinal_base=len(c._layouts[s]), layout="host-pos-leaf") for s in range(2, CAP)]
    com2.commit_offline(_executed(executed2), padding=padding2)
    ic2 = com2.identity_check(manifest=MAN)
    assert ic2["ok"] and ic2["padding"] == {"steps_with_padding": 3, "padding_only_steps": [], "padding_tensors": 3 * len(OPS), "rule": ic2["padding"]["rule"]}, ic2
    # a mixed step that DUPLICATES an executed tensor still fails (4), as on an unpadded run
    com2._layouts[3] = list(com2._layouts[3]) + [next(m for m in com2._layouts[3] if m.name == "logits_processor")]
    bad2 = com2.identity_check(manifest=MAN)
    assert not bad2["ok"] and bad2["violations"][0]["step"] == 3 and bad2["violations"][0]["multiplicity_vs_step0"] == {"logits_processor": (2, 1)}
    assert bad2["violations"][0]["padding_only"] is False and bad2["violations"][0]["padding_tensors"] == len(OPS)
    # an MoE manifest, B2 SYNC stop (both requests inactive from step 2 -> PADDING-ONLY steps with 2 inactive requests): the
    # MoE per-layer count at a padding-only step is the number of distinct (layer, member) identities (3 = want), not 2 x 3
    man_moe = dict(MAN, moe_layers=[0], moe_tensors_per_layer=len(MOE_MEMBERS))
    ids3 = _moe_identities(["r0", "r1"])
    executed3 = {0: ["r0", "r1"], 1: ["r0", "r1"]}
    com3 = _committer()

    def padding3(c):
        out = []
        for s in range(2, CAP):
            idl = _ids_at(ids3, "r0", s) + _ids_at(ids3, "r1", s)
            out.append(PS.padding_step_commit(s, idl, program_digest=PDB, run_id="run", geo_digest=c._geo_digest, chunk=CHUNK, layout="host-pos-leaf"))
        return out
    com3.commit_offline(_executed_with_moe(executed3), padding=padding3)
    ic3 = com3.identity_check(manifest=man_moe)
    assert ic3["ok"] and ic3["padding"]["padding_only_steps"] == [2, 3, 4] and ic3["violations_n"] == 0, ic3
    assert ic3["padding"]["padding_tensors"] == 3 * 2 * (len(OPS) + len(MOE_MEMBERS))
    # and a padding-only step that drops one request's router plane still passes X-03's set read (the per-request count is padded_population's)
    # while dropping the WHOLE (layer, member) identity fails moe_layers_bad by name
    com3._layouts[3] = [m for m in com3._layouts[3] if not m.name.startswith("moe/L0/router_ids/")]
    bad3 = com3.identity_check(manifest=man_moe)
    assert not bad3["ok"] and bad3["violations"][0]["step"] == 3 and bad3["violations"][0]["moe_layers_bad"] == {0: 2}


def test_the_identity_check_counts_the_router_softmax_tap_as_one_more_moe_tensor_of_each_tapped_layer():
    """An OLMoE sm_120 config run refused every step with moe_layers_bad {L: 7}: under ROUTER_TAP the
    router-softmax source commits `moe/L<k>/router_softmax` beside the block's planes, and the per-layer count expected the block's alone."""
    man_moe = dict(MAN, moe_layers=[0], moe_tensors_per_layer=len(MOE_MEMBERS))
    executed = {0: ["r0"], 1: ["r0"]}
    steps = _executed_with_moe(executed)
    for s, ts in zip(sorted(executed), steps):
        ts.append((PS.tensor_name("moe/L0", "router_softmax"), bytes((7 + i + s) % 251 for i in range(4 * 33))))
    com = _committer()
    com.commit_offline(steps)
    bad = com.identity_check(manifest=man_moe)
    assert not bad["ok"] and bad["violations"][0]["moe_layers_bad"] == {0: len(MOE_MEMBERS) + 1}
    ok = com.identity_check(manifest=dict(man_moe, router_softmax_layers=[0]))
    assert ok["ok"] and ok["violations_n"] == 0, ok
    com._layouts[1] = [m for m in com._layouts[1] if not m.name.startswith("moe/L0/router_softmax")]
    dropped = com.identity_check(manifest=dict(man_moe, router_softmax_layers=[0]))
    assert not dropped["ok"] and dropped["violations"][0]["step"] == 1 and dropped["violations"][0]["moe_layers_bad"] == {0: len(MOE_MEMBERS)}
