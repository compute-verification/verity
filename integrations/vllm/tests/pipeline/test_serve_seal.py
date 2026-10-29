"""A served window sealed as a deferred Commit's row dir (`serve_seal.seal_window`) and opened as PROVE-PATH opens it
(`deferred_rows.of_record`), and the registered registered compiler (`advice_windows.Capped`, PROVE-PATH's, with the cap ℓ).  CPU only."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_seal as ss
from verity_vllm.records.commit_run import CommitRun

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
    (run,) = CommitRun.load_lines(d / "commit" / "runs.jsonl")
    assert run.schema is None and run.arm == "instrumented" and run.tokens_equal is None and run.commit.run_root == rc.run_root.hex()
    assert "committed_not_retained" not in s and "committed_not_retained" not in json.loads(Path(s["bundle"], "MANIFEST.json").read_text())


#: a served step as `--retain-exclude hidden_m1` commits it: an attention interior between two module outputs
INTERIOR = [("model.layers.0.self_attn.qkv_proj/0", 300), ("fa2.m1/L0/s0/g32x8x1/S", None), ("model.layers.0.self_attn.o_proj/0", 200)]


def _interior_window(tmp_path, gap=None):
    """`INTERIOR` (the plane four leaves and 100 B) at step 0 and the sampler at step 1, committed under rows-sha512, step 0
    retained as the committer retains it with the hidden family excluded (`host_retention._HostRanges`, everything but the plane's
    leaf-aligned interior, or but ``gap`` as well), and sealed."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows
    from verity_vllm.commit.committer import host_retention as HR

    C = step_rows.ROW_BYTES
    spec = [(n, nb or 4 * C + 100) for n, nb in INTERIOR]
    stream = (torch.arange(sum(nb for _n, nb in spec), dtype=torch.int64) % 251).to(torch.uint8)
    tensors, off = [], 0
    for i, (name, nb) in enumerate(spec):
        tensors.append((name, bytes(stream[off:off + nb].numpy()), i, "int32" if ".m1/" in name else "bfloat16", (nb // 2,)))
        off += nb
    com, rc = commit_steps({0: tensors, 1: [("runner.sampler", b"\x01\x02\x03\x04")]}, commit_scheme=step_rows.NAME)
    assert com.chunk == C
    (lo, hi, fam), = HR._excluded_chunk_ranges(com._layouts[0], HR.HIDDEN_M1_FAMILIES, C)
    blk, used = com._gpu_blocks[0]
    kept = HR.outside(sorted([(lo, hi, fam), *([(*gap, "")] if gap else [])]), 0, -(-used // C) * C)
    com._gpu_blocks[0] = (HR._HostRanges([(a, b, blk[a:b]) for a, b in kept], used, {fam: hi - lo}, C), used)
    com.retain_exclude = set(HR.HIDDEN_M1_FAMILIES)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    leds = _ledgers()
    reqs, rows = ss.window_requests(ROWS, NAMES.get, leds.get)
    d = tmp_path / "windows" / "w6"

    def seal(by=ss.seal_window):
        return by(com, rc, d, rec={"window": 6, "run_id": "t/w6", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                  request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t")
    return SimpleNamespace(stream=stream, lo=lo, hi=hi, C=C, committed=used + com._gpu_blocks[1][1], d=d, seal=seal)


def test_a_window_committed_without_its_attention_interior_seals_it_as_committed_not_retained(tmp_path):
    """`--retain-exclude hidden_m1`: the run root is the one every byte was committed under, the bundle's store lacks only the
    plane's interior, its manifest (and the context, and the seal's line) name that as committed and not retained, and the window
    opens as PROVE-PATH opens it: a module output reads verified against the run root, a read inside the interior answers None."""
    from verity_vllm.commit.opened import OpenedReader
    from verity_vllm.pipeline import deferred_rows as DR, replay_bundle as RB

    w = _interior_window(tmp_path)
    s, n = w.seal(), w.hi - w.lo
    man = json.loads(Path(s["bundle"], "MANIFEST.json").read_text())
    got = man[RB.NOT_RETAINED]
    assert got == {"families": ["fa2_hidden_m1", "fa3_hidden_m1"], "bytes": n, "by_family": {"fa2_hidden_m1": n, "fa3_hidden_m1": 0},
                   "committed_bytes": w.committed, "retained_bytes": got["retained_bytes"]}
    assert s[RB.NOT_RETAINED] == got and RB.open_bundle(s["bundle"])["context"][RB.NOT_RETAINED] == got
    steps = json.loads(Path(s["bundle"], "store", "store.json").read_text())["steps"]
    assert got["retained_bytes"] == sum(b - a for st in steps for a, b in st["ranges"])
    assert Path(s["bundle"], "store", "raw", "s0.bin").stat().st_size == sum(b - a for a, b in steps[0]["ranges"])
    assert not any(a < w.hi and w.lo < b for a, b in steps[0]["ranges"]), "the interior is in no retained range"
    r = DR.of_record(s["bundle"], str(w.d))
    rd, o = OpenedReader(r.store, r.run), 300 + 4 * w.C + 100
    assert rd(0, "model.layers.0.self_attn.qkv_proj/0", 0, 300) == bytes(w.stream[0:300].numpy())
    assert rd(0, "model.layers.0.self_attn.o_proj/0", 0, 200) == bytes(w.stream[o:o + 200].numpy())
    assert rd(0, "fa2.m1/L0/s0/g32x8x1/S", 0, w.lo - 300) == bytes(w.stream[300:w.lo].numpy()), "the plane's boundary leaf is retained"
    assert rd(0, "fa2.m1/L0/s0/g32x8x1/S", w.lo - 300, w.lo - 300 + 64) is None
    assert rd.stats["not_retained"] == 1 and rd.stats["failed"] == 0


def test_a_committed_byte_not_retained_outside_the_excluded_interiors_is_refused_at_the_seal(tmp_path):
    from verity_vllm.commit import step_rows

    w = _interior_window(tmp_path, gap=(0, step_rows.ROW_BYTES))
    with pytest.raises(ValueError, match=r"not retained outside the excluded families' interiors .*step 0 \[0, "):
        w.seal()
    assert not (w.d / "commit" / "replay_bundle_p0").exists() and not (w.d / "commit" / "replay_bundle_p0.partial").exists()


#: the Commit of record of the bench row llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager (its ``source``):
#: its manifest's rows at engine steps 0 (four prefills) and 1 (four decodes), layers 0-1, its ``op_path_aliases``, and each
#: committed tensor of those steps in commit order with the identity the record's binding map bound it to
PARTS = Path(__file__).parent / "fixtures" / "llama32_1b_bi_eager_served_window_parts.json"
SERVED_ROWS = {0: [("runner-a", 0, 3, 0, 3), ("runner-b", 3, 5, 0, 2)], 1: [("runner-a", 0, 1, 3, 3), ("runner-b", 1, 2, 2, 2)]}
SERVED_NAMES = {"runner-a": "b0-1", "runner-b": "b0-2"}
_ITEM = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}


#: what the window's two steps take in: b0-1's prompt [1, 2, 3] and b0-2's [4, 5] at step 0, then each one's first sample
PORT = {0: ([1, 2, 3, 4, 5], [0, 1, 2, 0, 1]), 1: ([7, 8], [3, 2])}


def _i64(xs, T):
    return b"".join(int(x).to_bytes(8, "little") for x in [*xs, *[0] * (T - len(xs))])


def _sealed_parts(tmp_path, with_manifest: bool, port=None, embed_ids=None, name=None, kept=None, by=None):
    """The record's committed tensors of steps 0 and 1 (shapes cut to the served window's 5 and 2 token rows) committed and sealed
    as a served window of two requests, by a committer that carries the row's manifest (as `serve_seal.committer` leaves it) or not.
    With ``port`` ({step: (ids, positions)}) each step's stream starts with the step-input port and, as `serve_seal.single_ids`
    leaves the committer, the embedding's own ids aren't committed, unless ``embed_ids`` ({step: ids}) commits them beside it;
    its `served_record.SamplerInputsSource` kept the rows ``kept`` (`SERVED_ROWS` by default). Sealed by `serve_seal.seal_window`,
    or ``by``."""
    fx, seal = _parts_window(tmp_path, with_manifest, port=port, embed_ids=embed_ids, name=name, kept=kept)
    d, s = seal(by or ss.seal_window)
    return fx, d, s


def _parts_window(tmp_path, with_manifest: bool, port=None, embed_ids=None, name=None, kept=None):
    """`_sealed_parts`' window, committed, and how to seal it: ``seal(by)`` gives its row dir and what ``by`` returned."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows
    from verity_vllm.commit.served_record import PORT_NAMES, SamplerInputsSource, StepInputPort

    fx = json.loads(PARTS.read_text())
    T = 4096 // 8
    steps = {}
    for s, metas in fx["committed"].items():
        tokens = next(m["shape"][0] for m in metas if m["name"] == "model.embed_tokens/in")
        rows = max(hi for _r, _lo, hi, *_ in SERVED_ROWS[int(s)])
        steps[int(s)] = [(n, _i64(v, T), 1000 + j, "int64", (T,)) for j, (n, v) in enumerate(zip(PORT_NAMES, port[int(s)]))] if port else []
        for m in metas:
            shape = (rows, *([2] if len(m["shape"]) > 1 else [])) if m["shape"][0] == tokens else (8,)
            raw = bytes(_ITEM[m["dtype"]] * math.prod(shape))
            if m["name"] == "model.embed_tokens/in" and port:
                if embed_ids is None:
                    continue
                raw = b"".join(int(x).to_bytes(4, "little") for x in embed_ids[int(s)])
            steps[int(s)].append((m["name"], raw, m["ordinal"], m["dtype"], shape))
    com, rc = commit_steps(steps, commit_scheme=step_rows.NAME)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    if port:
        com.sources = [StepInputPort(com, T), SamplerInputsSource(com)]
        com.sources[1].rows = SERVED_ROWS if kept is None else kept
    if with_manifest:
        com.required_manifest = {k: fx[k] for k in ("identities", "op_path_aliases", "program_digest", "populations")}
    leds = {"b0-1": sc.Ledger("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 2}}, "b0-1", streamed=[7, 9]),
            "b0-2": sc.Ledger("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, "b0-2", streamed=[8, 6])}
    reqs, rows = ss.window_requests(SERVED_ROWS, SERVED_NAMES.get, leds.get)
    d = tmp_path / (name or ("w5" if with_manifest else "w5-raw"))

    def seal(by):
        return d, by(com, rc, d, rec={"window": 5, "run_id": "t/w5", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                     request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t")
    return fx, seal


def _store(d, s, opened: bool = False):
    """The sealed window opened as PROVE-PATH opens it: the record, its binding map, and the store its Programs read through, over the
    bundle's committed bytes verified against the run root (``opened``) or over no bytes."""
    from verity_vllm.check.replay.opening import CommittedStore
    from verity_vllm.commit.opened import OpenedReader
    from verity_vllm.pipeline import deferred_rows as DR

    r = DR.of_record(s["bundle"], str(d))
    bmap = json.loads((d / "commit" / "binding_map_p0.json").read_text())
    return r, bmap, CommittedStore.of_record(bmap, OpenedReader(r.store, r.run) if opened else (lambda *_a: None), r.manifest)


def test_a_served_windows_identities_are_what_the_rows_programs_read(tmp_path):
    """PROVE-PATH's read on serve-openai's w5 (b0-1, step 1, gate_up_proj/out), and the rest
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
    """A window sealed with no manifest: so a name is its identity, and the Programs' reads miss by name."""
    _fx, d, s = _sealed_parts(tmp_path, with_manifest=False)
    r, _bmap, store = _store(d, s)
    assert r.manifest is None and store.has("b0-1", 1, "model.layers.0.mlp.gate_up_proj", "0")
    assert not store.has("b0-1", 1, "model.layers.0.mlp.gate_up_proj", "out")
    assert not store.has("b0-1", 1, "model.layers.1.self_attn.rotary_emb", "query")


#: each request's token rows of each step (`SERVED_ROWS`)
TOKEN_ROWS = {("b0-1", 0): (0, 3), ("b0-2", 0): (3, 5), ("b0-1", 1): (0, 1), ("b0-2", 1): (1, 2)}


def _entries(bmap, rid, step, op, member):
    return [e for e in bmap["entries"] if e.get("kind") == "leaf"
            and [e["identity"][k] for k in ("request_id", "step", "op_path", "output_member")] == [rid, step, op, member]]


def test_a_sealed_windows_programs_read_the_ids_and_positions_at_the_ports_bytes(tmp_path):
    """The ids the row's Programs read (`model.embed_tokens/in`, the Embedding's input) are the port's bytes [0, 8T), each request's
    token rows of them, and the port's [8T, 16T) are the positions; the embedding's ids are committed nowhere else, and every other
    tensor binds as the Commit of record bound it. The row's RoPE reads no committed positions (its rows are ``query`` and ``key``:
    RoPE_v1's positions are the Program's static structure), so the port's positions bind as themselves, each request's rows."""
    from verity_vllm.commit.served_record import PORT_NAMES

    fx, d, s = _sealed_parts(tmp_path, True, port=PORT)
    r, bmap, store = _store(d, s, opened=True)
    T = 512
    assert r.ctx["served_window"]["step_input_port"]["binds_as"] == {"step.input/ids": ["model.embed_tokens", "in"]}
    assert {row["output_member"] for row in fx["identities"] if row["op_path"].endswith("rotary_emb")} == {"query", "key"}
    for (rid, step), (lo, hi) in TOKEN_ROWS.items():
        for (op, member), name, at in [(("model.embed_tokens", "in"), PORT_NAMES[0], 0), (("step.input", "positions"), PORT_NAMES[1], 8 * T)]:
            [e] = _entries(bmap, rid, step, op, member)
            assert (e["name"], e["stream_offset"], e["width_bytes"], e["element_offset"], e["identity"]["element_range"]) == (name, at, 8 * T, lo, [0, hi - lo])
        ids, pos = PORT[step]
        w, why = store.words(rid, step, "model.embed_tokens", "in", 0, hi - lo)
        assert w is not None and w.tolist() == ids[lo:hi], why
        assert store.words(rid, step, "step.input", "positions", 0, hi - lo)[0].tolist() == pos[lo:hi]
    got = {(e["identity"]["step"], e["tensor_ordinal"]): [e["identity"]["op_path"], e["identity"]["output_member"], e["identity"]["invocation"]]
           for e in bmap["entries"] if e.get("kind") == "leaf" and e["name"] not in PORT_NAMES}
    assert got == {(int(st), m["ordinal"]): m["bound"] for st, ms in fx["committed"].items() for m in ms if m["name"] != "model.embed_tokens/in"}


def test_a_port_byte_changed_is_what_the_programs_read(tmp_path):
    """b0-2's second prompt id 5 -> 6, one byte of step 0's stream (byte 32): the Embedding's input the Programs read changes with it,
    and so does the run root."""
    _fx, d, s = _sealed_parts(tmp_path, True, port=PORT)
    _fx, d2, s2 = _sealed_parts(tmp_path, True, port={**PORT, 0: ([1, 2, 3, 4, 6], PORT[0][1])}, name="w5-other")
    opened = [_store(dd, x, opened=True) for dd, x in ((d, s), (d2, s2))]
    assert [o[2].words("b0-2", 0, "model.embed_tokens", "in", 0, 2)[0].tolist() for o in opened] == [[4, 5], [4, 6]]
    assert [o[2].words("b0-1", 0, "model.embed_tokens", "in", 0, 3)[0].tolist() for o in opened] == [[1, 2, 3], [1, 2, 3]]
    assert opened[0][0].run.run_root != opened[1][0].run.run_root


@pytest.mark.parametrize("embed_ids", [{0: [1, 2, 3, 4, 9], 1: [7, 8]}, {0: PORT[0][0], 1: PORT[1][0]}], ids=["other-ids", "same-ids"])
def test_a_window_that_commits_the_models_ids_beside_the_port_is_refused(tmp_path, embed_ids):
    """A stream carrying the embedding's own ids beside the port's (a committer `single_ids` didn't leave, or a prover's) is refused
    at the seal whether or not they equal the port's: the row's Programs would read one of two copies of their input."""
    with pytest.raises(ValueError, match="one copy of \\[\\('model.embed_tokens', 'in'\\)\\].*step 0: model.embed_tokens/in"):
        _sealed_parts(tmp_path, True, port=PORT, embed_ids=embed_ids)


@pytest.mark.parametrize("port,kept,why", [
    ({**PORT, 1: ([7, 8], [3, 3])}, None,
     r"step 1: request b0-2's token row 1 \(its rows \[1, 2\) from position 2\) is at position 3, its row ran 2"),
    ({**PORT, 0: ([1, 2, 3, 4, 5], [0, 1, 1, 0, 1])}, None,
     r"step 0: request b0-1's token row 2 \(its rows \[0, 3\) from position 0\) is at position 1, its row ran 2"),
    ({**PORT, 0: ([1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 6])}, None, r"step 0: token 5, past the step's 5 tokens, is position 6, not 0"),
    (PORT, {0: SERVED_ROWS[0]}, r"step 1: no rows to hold its positions to"),
], ids=["decode-off-by-one", "prefill-off-by-one", "past-the-tokens", "no-rows"])
def test_a_window_whose_port_positions_are_not_what_its_rows_ran_is_refused(tmp_path, port, kept, why):
    """The eager Programs' RoPE positions are static, each request row's start..start+n-1 (the rows the window's served rows
    register); a port position other than those, or other than 0 past the step's tokens, is refused at the seal, by request."""
    with pytest.raises(ValueError, match=r"positions aren't the ones the window's rows ran at 1 step\(s\).*" + why):
        _sealed_parts(tmp_path, True, port=port, kept=kept)


def test_the_committer_commits_the_steps_ids_once_as_the_ports():
    """`single_ids` drops the embedding from the acquisition plan's input sites, so the committer acquires no ``model.embed_tokens/in``."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import committer

    com = committer(commit_scheme="rows-sha512")
    com.plan_inputs = {"model.embed_tokens", "model.layers.0.mlp.act_fn"}
    emb = torch.nn.Embedding(4, 2)
    assert com._acquires_input("model.embed_tokens", emb)
    ss.single_ids(com, json.loads(PARTS.read_text()))
    assert com.plan_inputs == {"model.layers.0.mlp.act_fn"} and not com._acquires_input("model.embed_tokens", emb)


def test_the_ports_identities_are_the_manifests_ids_and_compiled_positions():
    """On the eager row the port's ids are the Embedding's input and its positions nothing the Programs read; on the compiled path
    both are piece 0's inputs, and a tensor of the step under one of those names beside the port is a second copy."""
    from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort

    assert ss.port_identities(json.loads(PARTS.read_text())) == {PORT_NAMES[0]: ("model.embed_tokens", "in")}
    compiled = {"identities": [{"op_path": "c00.input", "output_member": m, "family": "served_kernel_values", "engine_step": 0}
                               for m in ("ids", "positions")] + [{"op_path": "logits_processor", "output_member": "out", "family": "served_kernel_values", "engine_step": 0}]}
    assert ss.port_identities(compiled) == {PORT_NAMES[0]: ("c00.input", "ids"), PORT_NAMES[1]: ("c00.input", "positions")}
    b = ss.served_binder(compiled, [0], [StepInputPort(SimpleNamespace(chunk=4096), 512)])
    assert b.bind(0, [*PORT_NAMES, "c00.input/ids"]) == [("c00.input", "ids", 0, "consumer-input"), ("c00.input", "positions", 0, "consumer-input"),
                                                          ("c00.input", "ids", 0, "unbound")]
    two = {"identities": [{"op_path": op, "output_member": "in", "family": "input_token_ids"} for op in ("model.embed_tokens", "model.mm_proj")]}
    with pytest.raises(ValueError, match="2 input-ids identities"):
        ss.port_identities(two)


def test_a_committer_without_the_commit_of_records_parts_is_named():
    manifest = {"identities": [{"family": "norm_scales"}, {"family": "fa2_hidden_m1_stream"}, {"family": "instance_outputs"}]}
    com = SimpleNamespace(name="native_collect_v2_auto_nw_hp+rows-sha512+sampler_inputs")
    assert ss.parts_missing(com, manifest, hidden_required=True) == ["hidden_m1", "norm_scale", "plan"]
    com.name = "native_collect_v2_auto_nw_hp+rows-sha512+hidden_m1+norm_scale+plan"
    assert ss.parts_missing(com, manifest, hidden_required=True) == []
    assert ss.parts_missing(SimpleNamespace(name="c+plan"), {"identities": []}, hidden_required=False) == []


def test_a_steps_stream_starts_with_its_input_ids_then_positions_padded_to_the_token_budget():
    """`served_record.StepInputPort` acquires the top model's ``input_ids`` and ``positions`` before any module of it commits,
    each T i64s with zeros past the step's N tokens; a call outside a step (a dummy run) commits nothing."""
    torch = pytest.importorskip("torch")
    from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort

    got: list[tuple[str, list[int]]] = []

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(64, 2)

        def forward(self, input_ids, positions):
            return self.embed_tokens(input_ids) + positions[:, None]

    model = Model()
    com = SimpleNamespace(model=model, chunk=64, _cur=None, _plan_shape=(3, 2), errors=[],
                          _acquire=lambda name, t, job: got.append((name, t.dtype, t.tolist())))
    model.embed_tokens.register_forward_pre_hook(lambda _m, a: com._cur and com._acquire("model.embed_tokens/in", a[0], com._cur))
    port = StepInputPort(com, 16)
    port.install()
    ids, pos = torch.tensor([5, 6, 7, 9], dtype=torch.int32), torch.tensor([0, 1, 4, 0])
    model(input_ids=ids, positions=pos)
    assert got == [], "a forward outside a step (vLLM's dummy runs) commits no port"
    com._cur = SimpleNamespace(step=0)
    model(input_ids=ids, positions=pos)
    assert [g[0] for g in got] == [*PORT_NAMES, "model.embed_tokens/in"] and com.errors == []
    assert got[0][1:] == (torch.int64, [5, 6, 7] + [0] * 13) and got[1][1:] == (torch.int64, [0, 1, 4] + [0] * 13)
    assert port.extra_step_bytes(None, None) == {"per_step_bytes": 256, "tensors_per_step": 2}
    assert port.describe() == {"names": list(PORT_NAMES), "T": 16, "IB": 8, "PB": 8, "ids_bytes": [0, 128], "positions_bytes": [128, 256]}
    port.uninstall()
    with pytest.raises(ValueError, match="multiple of 8"):
        StepInputPort(com, 12)


def test_a_sealed_windows_context_carries_its_served_rows_and_the_port_its_streams_start_with(tmp_path):
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import step_rows
    from verity_vllm.commit.served_record import PORT_NAMES, SamplerInputsSource, StepInputPort
    from verity_vllm.pipeline import deferred_rows as DR

    def i64(xs, T):
        return b"".join(int(x).to_bytes(8, "little") for x in [*xs, *[0] * (T - len(xs))])

    com0, _ = commit_steps({0: [("x", b"\0")]}, commit_scheme=step_rows.NAME)
    T = com0.chunk // 8
    steps = {0: [(PORT_NAMES[0], i64([1, 2, 3, 4, 5], T), 0, "int64", (T,)), (PORT_NAMES[1], i64([0, 1, 2, 0, 1], T), 1, "int64", (T,)),
                 ("model.embed_tokens/in", i64([1, 2, 3, 4, 5], 5), 2, "int64", (5,))],
             1: [(PORT_NAMES[0], i64([7, 8], T), 0, "int64", (T,)), (PORT_NAMES[1], i64([3, 2], T), 1, "int64", (T,))]}
    com, rc = commit_steps(steps, commit_scheme=step_rows.NAME)
    com.weights_tree, com.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    com.sources = [StepInputPort(com, T), SamplerInputsSource(com)]
    com.sources[1].rows = SERVED_ROWS
    served = {"digest": "ab" * 64, "statics": {"R": 4, "P": 2}}
    leds = {"b0-1": sc.Ledger("cmpl-1", "e1", [1, 2, 3], {"params": {"max_tokens": 2}}, "b0-1", streamed=[7, 9]),
            "b0-2": sc.Ledger("cmpl-2", "e2", [4, 5], {"params": {"max_tokens": 2}}, "b0-2", streamed=[8, 6])}
    reqs, rows = ss.window_requests(SERVED_ROWS, SERVED_NAMES.get, leds.get)
    d = tmp_path / "windows" / "w2"
    s = ss.seal_window(com, rc, d, rec={"window": 2, "run_id": "t/w2", "steps": 2, "passed": True, "served_rows": served}, reqs=reqs,
                       request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t")
    r = DR.of_record(s["bundle"], str(d))
    assert r.ctx["served_window"]["served_rows"] == served
    assert r.ctx["served_window"]["step_input_port"] == {"names": list(PORT_NAMES), "T": T, "IB": 8, "PB": 8,
                                                         "ids_bytes": [0, 8 * T], "positions_bytes": [8 * T, 16 * T]}
    assert s["port_layout"] == {"steps": 2, "at_head": 2, "first_other": None}
    com._layouts[1] = list(reversed(com._layouts[1]))
    assert ss.port_layout(com, ss.port_of(com))["first_other"]["head"][0][0] == PORT_NAMES[1]


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


# ------------------------------------------------------------------------------------------------------- the target φ reads

def test_the_served_target_carries_the_int8_scope_its_layers_ran(monkeypatch):
    """serve.json's target is the one predict's Build reads the int8 linears from (an exact-int8 checkpoint's Build refuses a
    target without them): `target_of` writes the scope read back from the served model's layers, none for a bf16 model, and
    refuses a mixed one."""
    import sys
    import types

    from verity_vllm.program.frontend.target_profile import TargetProfile

    vllm = types.ModuleType("vllm")
    vllm.envs = SimpleNamespace(VLLM_BATCH_INVARIANT=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", vllm.envs)
    h200 = TargetProfile.default().with_overrides(compute_capability=(9, 0), num_sms=132)
    monkeypatch.setattr(TargetProfile, "from_device", classmethod(lambda cls, device=0: h200))
    base = type("LinearMethodBase", (), {})
    int8 = type("Int8PlainLinearMethod", (base,), {"int8_arm": "plain"})()
    bf16 = type("UnquantizedLinearMethod", (base,), {})()

    def model(*methods):
        return SimpleNamespace(modules=lambda: iter([SimpleNamespace(quant_method=m) for m in methods]))

    got = ss.target_of(model(int8, int8))
    assert got == {"target": h200.with_overrides(int8_exact="linears").to_json(), "execution": "bi-eager"}
    assert got["target"]["int8_exact"] == "linears" and list(got["target"]["compute_capability"]) == [9, 0]
    assert ss.target_of(model(bf16)) == ss.target_of() == {"target": h200.to_json(), "execution": "bi-eager"}
    assert "int8_exact" not in ss.target_of(model(bf16))["target"]
    with pytest.raises(ValueError, match="int8_exact must be None"):
        ss.target_of(model(int8, bf16))


# ---------------------------------------------------------------------------------------------------------------- the advice's cap

def test_the_registered_compiler_is_prove_paths_capped_compiler():
    """registration.json's compiler digest is the one PROVE-PATH's capped re-commit binds: `advice_windows.Capped`'s formula."""
    import hashlib

    from verity.ml.circuits.codec import canonical_json
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.program.predict import steps as ST

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


# ------------------------------------------------------------------------------------------------- in the seal's own process

#: what a seal's result says of when and how long, which two seals of the same execution block needn't share
TIMES = ("manifest_sha256", "seal_s", "phases_s", "engine_s")


@pytest.fixture(scope="module")
def seal_process():
    from verity_vllm.pipeline.seal_process import SealProcess

    sp = SealProcess()
    yield sp
    assert sp.close()["exitcode"] == 0


def _written(d: Path) -> dict[str, Any]:
    """Every file under ``d``, by its path: its bytes, but the bundle's manifest less when it was sealed and the verdict less the
    manifest's sha256, as JSON."""
    out: dict[str, Any] = {}
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        rel = p.relative_to(d).as_posix()
        if p.name == "MANIFEST.json":
            out[rel] = {k: v for k, v in json.loads(p.read_text()).items() if k != "sealed_utc"}
        elif p.name == "verdict.json":
            out[rel] = v = json.loads(p.read_text())
            for b in v["replay_deferred"]["bundles"]:
                b.pop("manifest_sha256")
        else:
            out[rel] = p.read_bytes()
    return out


def _both(d: Path, seal: Any, by: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The execution block ``seal`` seals into ``d`` sealed by `seal_window`, then again by ``by``: what each wrote and returned."""
    inline = seal(ss.seal_window)[1]
    first = _written(d)
    shutil.rmtree(d)
    other = seal(by)[1]
    return first, _written(d), inline, other


@pytest.mark.parametrize("window", ["port", "interior"])
def test_an_execution_block_sealed_in_its_own_process_writes_what_the_serving_process_writes(tmp_path, seal_process, window):
    """`seal_process.SealProcess`: the store's files written by the serving process, the binding map, the checks, the store's
    store.json, the context, the manifest and the records by the seal's: the files `seal_window` writes, byte for byte but for
    when the manifest says it was sealed, and the result it returns. An execution block with the port and the row's manifest
    (its binder, its positions' check, its port layout), and one committed without its attention interior
    (`store_dump.not_retained`)."""
    if window == "port":
        _fx, seal = _parts_window(tmp_path, True, port=PORT)
        d = tmp_path / "w5"
    else:
        w = _interior_window(tmp_path)
        d, seal = w.d, lambda by: (w.d, w.seal(by))
    first, again, inline, other = _both(d, seal, seal_process.seal)
    assert first.keys() == again.keys() and [k for k in first if first[k] != again[k]] == []
    assert {k: v for k, v in inline.items() if k not in TIMES} == {k: v for k, v in other.items() if k not in TIMES}
    assert other["phases_s"].keys() == inline["phases_s"].keys() and other["engine_s"].keys() == {"handoff", "store_files"}
    assert json.loads((d / "commit" / "replay_bundle_p0" / "store" / "store.json").read_text())["steps"][0]["metas"]


def test_a_check_failed_in_the_seals_process_is_the_blocks_error_and_leaves_no_store(tmp_path, seal_process):
    """A check that fails in the seal's process fails the execution block by the same name `seal_window`'s does, and the store
    the serving process wrote meanwhile is discarded, as `seal_window` never writes it; the process seals the next block."""
    from verity_vllm.commit import step_rows
    from verity_vllm.pipeline.seal_process import SealFailed

    with pytest.raises(SealFailed, match=r"^ValueError: the step-input port's positions aren't .*its row ran 2"):
        _sealed_parts(tmp_path, True, port={**PORT, 1: ([7, 8], [3, 3])}, by=seal_process.seal)
    w = _interior_window(tmp_path, gap=(0, step_rows.ROW_BYTES))
    with pytest.raises(SealFailed, match=r"^ValueError: .*not retained outside the excluded families' interiors .*step 0 \[0, "):
        w.seal(seal_process.seal)
    for d in (tmp_path / "w5", w.d):
        assert not (d / "commit" / "replay_bundle_p0").exists() and not (d / "commit" / "replay_bundle_p0.partial").exists()
    _fx, _d, s = _sealed_parts(tmp_path, True, port=PORT, name="w7", by=seal_process.seal)
    assert "error" not in s and Path(s["bundle"], "MANIFEST.json").is_file()


def test_a_store_the_serving_process_fails_to_write_raises_what_seal_window_raises(tmp_path, seal_process):
    """The serving process's own failure to write the store, once the process's checks pass, is raised as `seal_window` raises it."""
    raised = []
    for by in (ss.seal_window, seal_process.seal):
        _fx, seal = _parts_window(tmp_path, True, port=PORT)
        (tmp_path / "w5" / "commit").mkdir(parents=True, exist_ok=True)
        (tmp_path / "w5" / "commit" / "replay_bundle_p0.partial").write_text("a file where the store's dir goes")
        with pytest.raises(NotADirectoryError) as e:
            seal(by)
        raised.append(str(e.value))
        shutil.rmtree(tmp_path / "w5")
    assert raised[0] == raised[1]
