"""A served window sealed as a deferred Commit's row dir (`serve_seal.seal_window`) and opened as PROVE-PATH opens it
(`deferred_rows.of_record`), and the registered advice compiler (`advice_windows.Capped`, PROVE-PATH's, with the cap ℓ).  CPU only."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_seal as ss

ROWS = {0: [("runner-a", 0, 3, 0, 3), ("runner-b", 3, 5, 0, 2), ("warmup", 5, 6, 0, 1)], 1: [("runner-a", 0, 1, 3, 3)]}
NAMES = {"runner-a": "alice", "runner-b": "cmpl-2"}


def _ledgers() -> dict[str, sc.Ledger]:
    return {"alice": sc.Ledger("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 4}}, "alice", streamed=[7, 9]),
            "cmpl-2": sc.Ledger("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, None, streamed=[8])}


def test_a_windows_requests_are_named_by_user_with_their_rows_and_first_step():
    leds = _ledgers()
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    assert [q["request_id"] for q in reqs] == ["alice", "cmpl-2"] and [q["index"] for q in reqs] == [0, 1]
    assert reqs[0] | {"ingress": None} == {"request_id": "alice", "index": 0, "arrive_step": 0, "prompt_token_ids": [1, 2, 3], "max_tokens": 4,
                                           "http_id": "cmpl-1", "engine_id": "e1", "ingress": None}
    assert rows == {0: [("alice", 0, 3), ("cmpl-2", 3, 5)], 1: [("alice", 0, 1)]}, "a row of no served request is in no request's rows"


def _sealed(tmp_path):
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows

    com, rc = commit_steps({0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]},
                           commit_scheme=step_rows.NAME)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    leds = _ledgers()
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    d = tmp_path / "windows" / "w5"
    s = ss.seal_window(com, rc, d, rec={"window": 5, "run_id": "t/w5", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                       request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t",
                       code="c0de", engine_of_record={"vllm": "test"})
    return rc, d, s


def test_a_sealed_rows_sha512_window_opens_as_a_deferred_commits_row_dir(tmp_path):
    from verity_vllm.pipeline import deferred_rows as DR

    rc, d, s = _sealed(tmp_path)
    assert s["commit_scheme"] == "rows-sha512" and len(rc.run_root) == DR.SERVED_ROOT and "error" not in s
    v = json.loads((d / "commit" / "verdict.json").read_text())
    assert v["run_root"] == rc.run_root.hex() and v["replay_deferred"]["bundles"][0]["manifest_sha256"] == s["manifest_sha256"]
    r = DR.of_record(s["bundle"], str(d))
    assert r.run.run_root == rc.run_root and r.ctx["served_window"]["window"] == 5 and r.ctx["code"] == "c0de"
    assert r.programs == {} and r.arrivals == {"alice": 0, "cmpl-2": 0} and r.ctx["tokens"] == {"alice": [7, 9], "cmpl-2": [8]}
    assert sorted(p.name for p in (d / "commit").iterdir()) == ["binding_map_p0.json", "replay_bundle_p0", "runs.jsonl", "verdict.json"]


#: the Commit of record of the bench row llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager (its ``source``):
#: its manifest's rows at engine steps 0 (four prefills) and 1 (four decodes), layers 0-1, its ``op_path_aliases``, and each
#: committed tensor of those steps in commit order with the identity the record's binding map bound it to
PARTS = Path(__file__).parent / "fixtures" / "llama32_1b_bi_eager_served_window_parts.json"
SERVED_ROWS = {0: [("runner-a", 0, 3, 0, 3), ("runner-b", 3, 5, 0, 2)], 1: [("runner-a", 0, 1, 3, 3), ("runner-b", 1, 2, 2, 2)]}
SERVED_NAMES = {"runner-a": "b0-1", "runner-b": "b0-2"}
_ITEM = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}


def _sealed_parts(tmp_path, with_manifest: bool):
    """The record's committed tensors of steps 0 and 1 (shapes cut to the served window's 5 and 2 token rows) committed and sealed
    as a served window of two requests, by a committer that carries the row's manifest (as `serve_seal.committer` leaves it) or not."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows

    fx = json.loads(PARTS.read_text())
    steps = {}
    for s, metas in fx["committed"].items():
        tokens = next(m["shape"][0] for m in metas if m["name"] == "model.embed_tokens/in")
        rows = max(hi for _r, _lo, hi, *_ in SERVED_ROWS[int(s)])
        steps[int(s)] = []
        for m in metas:
            shape = (rows, *([2] if len(m["shape"]) > 1 else [])) if m["shape"][0] == tokens else (8,)
            steps[int(s)].append((m["name"], bytes(_ITEM[m["dtype"]] * math.prod(shape)), m["ordinal"], m["dtype"], shape))
    com, rc = commit_steps(steps, commit_scheme=step_rows.NAME)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    if with_manifest:
        com.required_manifest = {k: fx[k] for k in ("identities", "op_path_aliases", "program_digest", "populations")}
    leds = {"b0-1": sc.Ledger("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 2}}, "b0-1", streamed=[7, 9]),
            "b0-2": sc.Ledger("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, "b0-2", streamed=[8, 6])}
    reqs, rows = ss.window_requests(SERVED_ROWS, SERVED_NAMES.get, leds.get)
    d = tmp_path / ("w5" if with_manifest else "w5-raw")
    s = ss.seal_window(com, rc, d, rec={"window": 5, "run_id": "t/w5", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                       request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t")
    return fx, d, s


def _store(d, s):
    from verity_vllm.check.replay.opening import CommittedStore
    from verity_vllm.pipeline import deferred_rows as DR

    r = DR.of_record(s["bundle"], str(d))
    bmap = json.loads((d / "commit" / "binding_map_p0.json").read_text())
    return r, bmap, CommittedStore.of_record(bmap, lambda *_a: None, r.manifest)


def test_a_served_windows_identities_are_what_the_rows_programs_read(tmp_path):
    """The failing read of PROVE-PATH on serve-openai's w5 (r20261024-114918-57a0), (b0-1, step 1, gate_up_proj/out), and the rest
    of what the row's Programs read: a Gemm's `out`, RoPE's `query` / `key` (layer 1 through the row's alias), an RMSNorm's
    `norm_scale` and an `fa2.m1` member, each bound exactly as the Commit of record bound the same tensor."""
    fx, d, s = _sealed_parts(tmp_path, with_manifest=True)
    r, bmap, store = _store(d, s)
    assert r.manifest == {"op_path_aliases": fx["op_path_aliases"]} and r.ctx["served_window"]["op_path_aliases"] == fx["op_path_aliases"]
    for rid in ("b0-1", "b0-2"):
        for step in (0, 1):
            for op, member in [("model.layers.0.mlp.gate_up_proj", "out"), ("model.layers.1.mlp.down_proj", "out"),
                               ("model.layers.0.self_attn.rotary_emb", "query"), ("model.layers.0.self_attn.rotary_emb", "key"),
                               ("model.layers.1.self_attn.rotary_emb", "query"), ("model.layers.1.self_attn.rotary_emb", "key"),
                               ("model.layers.0.input_layernorm", "norm_scale"), ("model.layers.1.post_attention_layernorm", "norm_scale"),
                               ("model.layers.1.input_layernorm", "0"), ("model.layers.1.input_layernorm", "1")]:
                assert store.has(rid, step, op, member), (rid, step, op, member)
        assert store.has(rid, 1, "fa2.m1/L0", "g32x4x3") and store.has(rid, 0, "fa2.m1/L1", "s0/g32x8x2/S")
    got = {(e["identity"]["step"], e["tensor_ordinal"]): [e["identity"]["op_path"], e["identity"]["output_member"], e["identity"]["invocation"]]
           for e in bmap["entries"] if e.get("kind") == "leaf"}
    want = {(int(st), m["ordinal"]): m["bound"] for st, ms in fx["committed"].items() for m in ms}
    assert got == want, "every committed tensor is bound to the identity the Commit of record bound it to"


def test_without_the_rows_manifest_a_window_binds_raw_names(tmp_path):
    """What r20261024-105002-1042's w5 sealed: no manifest, so a name is its identity, and the Programs' reads miss by name."""
    _fx, d, s = _sealed_parts(tmp_path, with_manifest=False)
    r, _bmap, store = _store(d, s)
    assert r.manifest is None and store.has("b0-1", 1, "model.layers.0.mlp.gate_up_proj", "0")
    assert not store.has("b0-1", 1, "model.layers.0.mlp.gate_up_proj", "out")
    assert not store.has("b0-1", 1, "model.layers.1.self_attn.rotary_emb", "query")


def test_a_committer_without_the_commit_of_records_parts_is_named():
    manifest = {"identities": [{"family": "norm_scales"}, {"family": "fa2_hidden_m1_stream"}, {"family": "instance_outputs"}]}
    com = SimpleNamespace(name="native_collect_v2_auto_nw_hp+rows-sha512+sampler_inputs")
    assert ss.parts_missing(com, manifest, hidden_required=True) == ["hidden_m1", "norm_scale", "plan"]
    com.name = "native_collect_v2_auto_nw_hp+rows-sha512+hidden_m1+norm_scale+plan"
    assert ss.parts_missing(com, manifest, hidden_required=True) == []
    assert ss.parts_missing(SimpleNamespace(name="c+plan"), {"identities": []}, hidden_required=False) == []


def _other_root(d):
    p = d / "commit" / "runs.jsonl"
    row = json.loads(p.read_text())
    row["commit"]["run_root"] = "00" * 64
    p.write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("tamper,why", [
    (_other_root, "run root of record"),
    (lambda d: (d / "commit" / "binding_map_p0.json").write_text("{}"), "binding map"),
    (lambda d: (d / "commit" / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": []}})), "names no bundle"),
])
def test_a_sealed_window_whose_record_is_not_its_bundles_is_refused(tmp_path, tamper, why):
    from verity_vllm.pipeline import deferred_rows as DR
    from verity_vllm.pipeline.cpu_replay import Refused

    _rc, d, s = _sealed(tmp_path)
    tamper(d)
    with pytest.raises(Refused, match=why):
        DR.of_record(s["bundle"], str(d))


# ---------------------------------------------------------------------------------------------------------------- the advice's cap

def test_the_registered_compiler_is_prove_paths_capped_compiler():
    """registration.json's compiler digest is the one PROVE-PATH's capped re-commit binds: `advice_windows.Capped`'s formula."""
    import hashlib

    from verity.ml.circuits.codec import canonical_json
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.predict import steps as ST

    assert ADV.CAP_BITS == AW.CAP_BITS == ST.CAP_BITS == 1 << 23 and AW.CAPPED == "verity/prove-path/capped-compiler/v0"
    c = ADV.registered_compiler("inner-compiler", "cd" * 64, cap_bits=4096)
    want = hashlib.sha512(canonical_json({"name": "verity/prove-path/capped-compiler/v0", "cap_bits": 4096,
                                          "compiler": {"name": "inner-compiler", "digest": "cd" * 64}})).hexdigest()
    assert isinstance(c, AW.Capped) and c.name == AW.CAPPED and c.digest == want
    assert c.digest == "2f470d2b6aadc53b707b0bd65820da4c55583eb18667eb3dce128aabb9dddab855473d52af758a630a2958e3555f7cafbc68e9238c0188cd9f8ef551abca2a9a"
    assert ADV.registered_compiler("inner-compiler", "cd" * 64, cap_bits=8192).digest != want
    assert ADV.registered_compiler("inner-compiler").cap_bits == ADV.CAP_BITS


@pytest.mark.parametrize("cap", [0, 7, True, "8"])
def test_a_cap_that_is_not_a_bit_count_is_refused(cap):
    with pytest.raises(ValueError, match="whole number of bits"):
        ADV.registered_compiler("c", digest="ab" * 64, cap_bits=cap)
