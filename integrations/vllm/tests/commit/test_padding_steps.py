"""Padding steps: the inactive rows of a padded Commit are SYNTHESIZED by
enablement at finalize -- every identity the manifest requires of a request past its executed prefix of record is the canonical ⊥ of its
lifted width, chunked / folded / bound with the production step rule -- and the padded map binds RequiredValues(Lift(P)) completely.
One rule, two consumers (pod finalize, laptop conversion over a pulled record); this test drives the conversion over a synthetic record
laid out exactly like a pulled smollm2 record (manifest.json.gz, commit/binding_map_p0.json.gz, build_workload/workload_program.json,
match/capture/tokens.json).  Torch-free."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.commit import executed_prefix as EP
from verity_vllm.commit import padding_steps as PS
from verity_vllm.commit import scheme
from verity_vllm.commit.merkle import fold_path
from verity_vllm.commit.scheme import pos_leaf

PD = hashlib.sha256(b"padding-steps-test-program").hexdigest()
EOS = 0
CAP = 6
OPS = [("model.embed_tokens", "in", "i32", 1, "input_token_ids"), ("model.layers.0.input_layernorm", "out", "bf16", 64, "instance_outputs"),
       ("model.layers.0.self_attn.qkv_proj", "0", "bf16", 96, "instance_outputs"), ("fa2.m1/L0", "g3x3x1", "u32", 300, "fa2_hidden_m1_stream"),
       ("logits_processor", "out", "bf16", 128, "instance_outputs"), ("runner.sampler", "sampled_token_ids", "i64", 1, "sampled_token_ids")]


def _identities(requests: dict[str, int]) -> list[dict]:
    ids = []
    for rid, arrive in requests.items():
        for s in range(CAP):
            for op, mem, dt, n, fam in OPS:
                d = {"family": fam, "program_digest": PD, "request_id": rid, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                     "element_range": [0, n], "rank": 0, "engine_step": arrive + s, "dtype": dt}
                if fam == "input_token_ids":
                    d["rows"] = 1
                ids.append(d)
    ids.append({"family": "weights", "program_digest": PD, "request_id": "r0", "step": "*", "invocation": 0, "op_path": "model.embed_tokens.weight",
                "output_member": "weight", "element_range": [0, None], "rank": 0, "dtype": "bf16"})
    return ids


def _record(tmp: Path, *, served: dict[str, list[int]], lag: int, executed_steps: dict[str, list[int]], geo: bytes) -> Path:
    """A pulled record: the executed steps committed with the production rule (so their roots are real), the rest absent."""
    rd = tmp / "r17_test_record"
    (rd / "commit").mkdir(parents=True)
    (rd / "build_workload").mkdir()
    (rd / "match" / "capture").mkdir(parents=True)
    reqs = list(served)
    with gzip.open(rd / "manifest.json.gz", "wt") as f:
        json.dump({"identities": _identities({r: 0 for r in reqs})}, f)
    # executed steps: one packed stream per engine step over the requests present, bytes = deterministic non-⊥ payloads
    entries = []
    step_roots = {}
    steps = sorted({s for v in executed_steps.values() for s in v})
    for s in steps:
        present = [r for r in reqs if s in executed_steps[r]]
        metas, stream, off, k = [], [], 0, 0
        for op, mem, dt, n, fam in OPS:
            nb = {"i32": 4, "bf16": 2, "u32": 4, "i64": 8}[dt]
            rows = len(present)
            data = bytes((7 + i) % 251 for i in range(n * nb * rows))
            pad = (-off) % 256
            stream.append(b"\0" * pad)
            off += pad
            name = PS.tensor_name(op, mem)
            metas.append((k, name, dt, [rows, n], n * nb * rows, off))
            stream.append(data)
            off += n * nb * rows
            k += 1
        blob = b"".join(stream)
        leaves = [pos_leaf(blob[o:o + 256]) for o in range(0, len(blob), 256)]
        lv = scheme.fold_levels(leaves)
        ldig = hashlib.sha256(json.dumps([[m[0], m[1], m[2], m[3], m[4], 256, m[5] // 256, (m[4] + 255) // 256, m[5]] for m in metas], separators=(",", ":")).encode()).digest()
        root = scheme.step_root(bytes.fromhex(PD), PS.ctx_digest("test-run", s), geo, ldig, len(leaves), lv[-1][0])
        step_roots[s] = root
        for (k, name, dt, shape, nbytes, so), (op, mem, _dt, n, fam) in zip(metas, OPS):
            for i, r in enumerate(present):
                entries.append({"identity": {"program_digest": PD, "request_id": r, "step": s, "invocation": 0, "op_path": op, "output_member": mem,
                                             "element_range": [0, n], "rank": 0}, "kind": "leaf", "step_root": root.hex(), "layout": "chunk-leaf-v1",
                                "chunk": 256, "leaf_lo": so // 256, "leaf_hi": (so + nbytes - 1) // 256 + 1, "stream_offset": so, "width_bytes": nbytes,
                                "encoding": {"dtype": dt, "shape": shape}, "element_offset": i * n, "tensor_ordinal": k, "name": name})
    entries.append({"identity": {"program_digest": PD, "request_id": "*", "step": -1, "invocation": 0, "op_path": "weights", "output_member": "registered",
                                 "element_range": [0, 0], "rank": 0}, "kind": "registered-input", "digest": "ab" * 32})
    run_root = scheme.run_root(bytes.fromhex(PD), geo, [step_roots[s] for s in steps])
    with gzip.open(rd / "commit" / "binding_map_p0.json.gz", "wt") as f:
        json.dump({"schema": "binding-map/1", "rules_version": "r14-v1", "program_digest": PD, "run_root": run_root.hex(), "rank": 0,
                   "layout": "chunk-leaf-v1", "collector": {"variant": "test"}, "entries_n": len(entries), "digest": "0" * 64, "entries": entries}, f)
    (rd / "build_workload" / "workload_program.json").write_text(json.dumps({
        "lag": {"value": lag, "declared_explicitly": True}, "stop_rule": {"eos_token_ids": [EOS]},
        "requests": [{"request_id": r, "cap": CAP, "arrive_step": 0} for r in reqs],
        "request_ids": {"workload_request_ids": {r: f"uuid-{r}" for r in reqs}}}))
    (rd / "match" / "capture" / "tokens.json").write_text(json.dumps({"requests": [{"request_id": f"uuid-{r}", "generated_token_ids": served[r]} for r in reqs]}))
    return rd


@pytest.mark.parametrize("lag", [0, 1])
def test_b1_early_stop_the_padding_steps_are_the_cap_suffix_every_word_bot_one_root_over_cap_steps(tmp_path, lag):
    """#3 shape: r0 stops by EOS at served 3 of cap 6; executed prefix = 3 + LAG; padding steps = the rest; the padded map covers the
    manifest completely; the run root binds S = 6 steps; ⊥ words are canonical `1 || 0^w` in the lifted container; openings verify."""
    geo = hashlib.sha256(b"geo").digest()
    served = {"r0": [5, 6, EOS]}
    n_exec = 3 + lag
    rd = _record(tmp_path, served=served, lag=lag, executed_steps={"r0": list(range(n_exec))}, geo=geo)
    res = PS.padded_map_of_record(rd, run_id="test-run", geo_digest=geo)
    assert res["ok"], res.get("why")
    rep = res["report"]
    assert rep["steps"] == {"executed": n_exec, "padding": CAP - n_exec, "S": CAP}
    assert rep["executed_prefix"]["r0"]["executed_prefix_len"] == n_exec and rep["executed_prefix"]["r0"]["lag_steps"] == list(range(3, n_exec))
    cov = rep["coverage_over_manifest"]
    assert cov["before"]["missing_n"] == (CAP - n_exec) * len(OPS) and cov["padded"]["missing_n"] == 0 and cov["padded"]["ok"]
    assert rep["bot_words_canonical"] and all(rep["openings_checked"].values()) and rep["one_root_on_laptop"]
    # the bytes: every padding word is the lifted ⊥ of its committed dtype, the container the TRUE lifted width's (no split, A11)
    for ps in res["padding"]:
        assert not ps.appended
        for m in ps.metas:
            w, nb, label = PS.LIFTED_CONTAINER[m.committed_dtype]
            assert m.dtype == label and m.nbytes == m.elements * nb
            seg = ps.stream[m.stream_off:m.stream_off + m.nbytes]
            assert seg == (1 << w).to_bytes(nb, "little") * m.elements
        assert ps.n_leaves == len(ps.levels[0]) and PS.verify_opening(ps, ps.n_leaves - 1)
    assert PS.bot_word("bf16") == bytes.fromhex("00000100") and PS.bot_word("u32") == (1 << 32).to_bytes(8, "little") and len(PS.bot_word("i64")) == 16
    # one root over the cap: the record's step roots for the executed prefix + the padding roots, folded with the committer's rule
    pm = res["padded_map"]
    roots = {}
    for e in pm["entries"]:
        if e["kind"] == "leaf":
            roots.setdefault(e["identity"]["step"], set()).add(e["step_root"])
    assert sorted(roots) == list(range(CAP)) and all(len(v) == 1 for v in roots.values())
    assert pm["run_root"] == scheme.run_root(bytes.fromhex(PD), geo, [bytes.fromhex(next(iter(roots[s]))) for s in range(CAP)]).hex()
    # a CONVERSION never re-labels the record's Program: program_rules_version stays the record's (None
    # for an ordinary Program); the padding rule is named beside it (`conversion`, `padding_rules_version`)
    assert pm["schema"] == "padded-binding-map/1" and pm["program_rules_version"] is None and pm["padding_rules_version"] == "padded-v1"
    assert pm["conversion"].startswith("padding-steps/1")
    acts = {e["activity"] for e in pm["entries"]}
    assert acts == {"active", "bot-by-enablement", "registered-input"}
    # the block's domain is READ from the entry (block_kind / NB), never inferred from the activity label
    assert all((e["block_kind"], e["nb"]) == ("padding", 1) for e in pm["entries"] if e["activity"] == "bot-by-enablement")
    assert all((e["block_kind"], e["nb"]) == ("executed", 0) for e in pm["entries"] if e["activity"] == "active")
    assert sum(1 for e in pm["entries"] if e["activity"] == "bot-by-enablement") == (CAP - n_exec) * len(OPS)
    # the ONE fact the three consumers of the executed prefix read (rows_past_the_stop, coverage, sampled_replay's inactive
    # stratum): `activity` on EVERY entry; `encoding.committed_dtype` + `encoding.bot_word_hex` on every padding entry, and the hex IS the word
    assert all("activity" in e for e in pm["entries"])
    for e in pm["entries"]:
        if e["activity"] == "bot-by-enablement":
            assert e["encoding"]["bot_word_hex"] == PS.bot_word(e["encoding"]["committed_dtype"]).hex()
            assert e["identity"]["step"] >= n_exec and e["step_root"] is not None
        elif e["activity"] == "active":
            assert e["identity"]["step"] < n_exec and "bot_word_hex" not in e["encoding"]


def test_b2_mixed_stop_the_padding_is_appended_to_the_executed_steps_and_the_one_root_is_the_pods_refold(tmp_path):
    """#4 shape: r0 stops at served 2 (LAG 1 -> executed 3), r1 runs to the cap: steps 3..5 are MIXED (r1 executed them).  The padding
    of r0 at those steps is appended after the executed leaves (leaf ranks / stream offsets continue), its entries carry the padding
    subtree root and NO step root (the pulled record has no leaf digests to refold); `refold_step` over the executed leaf digests +
    the padding leaves gives the one step root the pod consumer publishes -- checked here against a full recomputation."""
    geo = hashlib.sha256(b"geo").digest()
    served = {"r0": [5, EOS], "r1": [1, 2, 3, 4, 5, 6]}
    rd = _record(tmp_path, served=served, lag=1, executed_steps={"r0": [0, 1, 2], "r1": list(range(CAP))}, geo=geo)
    res = PS.padded_map_of_record(rd, run_id="test-run", geo_digest=geo)
    assert res["ok"], res.get("why")
    rep = res["report"]
    assert rep["mixed_steps"] == [3, 4, 5] and rep["steps"] == {"executed": CAP, "padding": 3, "S": CAP} and not rep["one_root_on_laptop"]
    assert rep["coverage_over_manifest"]["padded"]["ok"] and rep["coverage_over_manifest"]["padded"]["missing_n"] == 0
    assert rep["coverage_over_manifest"]["before"]["missing_n"] == 3 * len(OPS)
    pm = res["padded_map"]
    assert pm["run_root"] is None and "refold" in pm["run_root_note"]
    bot = [e for e in pm["entries"] if e["activity"] == "bot-by-enablement"]
    assert len(bot) == 3 * len(OPS) and all(e["identity"]["request_id"] == "r0" and e["step_root"] is None and e["padding_subtree_root"] for e in bot)
    # appended after the executed leaves of the same step
    bm = json.load(gzip.open(rd / "commit" / "binding_map_p0.json.gz", "rt"))
    for s in (3, 4, 5):
        exec_hi = max(e["leaf_hi"] for e in bm["entries"] if e["kind"] == "leaf" and e["identity"]["step"] == s)
        assert min(e["leaf_lo"] for e in bot if e["identity"]["step"] == s) == exec_hi
    # the pod's refold = the fold over (executed leaf digests + padding leaves), bound with the layout over both -- recomputed from scratch
    ps = next(p for p in res["padding"] if p.step == 3)
    assert ps.appended and ps.leaf_base > 0
    exec_leaves = [hashlib.sha256(b"leaf%d" % i).digest() for i in range(ps.leaf_base)]         # the retained level-0 digests of step 3
    exec_metas = [PS.PaddingMeta(ordinal=0, name="x", dtype="bf16", shape=(1,), nbytes=ps.leaf_base * 256, chunk=256, first_leaf=0, n_leaves=ps.leaf_base, stream_off=0)]
    levels, root = PS.refold_step(exec_leaves, exec_metas, ps, program_digest=bytes.fromhex(PD), run_id="test-run", geo_digest=geo)
    # the padding leaves under the leaf rule of record (chunk-leaf-v1 = the GPU tree's CMT-2 chunk header): HB = step, M = the padding
    # BLOCK's padded words, NB = PADDING_BLOCK (1; executed leaves carry 0), launch_tag = step, chunk_index continuing from the executed
    # leaves, chunk_words 64, src_mask 0 -- recomputed here field by field, not through the module
    assert len(ps.stream) % 256 == 0

    class _L:
        HB, M, NB = 3, len(ps.stream) // 4, 1
    pad_leaves = [scheme.chunk_leaf(scheme.chunk_header(_L, 3, ps.leaf_base + i, 64, 0), ps.stream[i * 256:(i + 1) * 256]) for i in range(len(ps.stream) // 256)]
    assert pad_leaves == ps.levels[0] and pad_leaves != [pos_leaf(ps.stream[o:o + 256]) for o in range(0, len(ps.stream), 256)]
    all_leaves = exec_leaves + pad_leaves
    assert levels[-1][0] == scheme.fold(all_leaves)
    assert root == scheme.step_root(bytes.fromhex(PD), PS.ctx_digest("test-run", 3), geo, PS.layout_digest(exec_metas + ps.metas), len(all_leaves), scheme.fold(all_leaves))
    # a padding leaf opens against the REFOLDED step tree at its appended rank
    r = ps.leaf_base + 1
    path, k = [], r
    for lvl in levels[:-1]:
        sib = k ^ 1
        path.append(lvl[sib] if sib < len(lvl) else None)
        k >>= 1
    assert fold_path(all_leaves[r], r, path) == levels[-1][0]
    with pytest.raises(ValueError):
        PS.refold_step(exec_leaves[:-1], exec_metas, ps, program_digest=bytes.fromhex(PD), run_id="test-run", geo_digest=geo)


def test_the_rule_refuses_what_it_cannot_place_and_never_infers_the_lag(tmp_path):
    """No declared LAG on an EOS stop -> executed_prefix_of_record is not ok -> nothing is synthesized (ValueError by name); the
    conversion returns ok=False with the reason; an executed step outside the declared rule is a disagreement, not reconciled."""
    ids = _identities({"r0": 0})
    epr = EP.executed_prefix_of_record(["r0"], {"r0": [5, EOS]}, caps={"r0": CAP}, eos_of_record=[EOS], lag=None)
    assert not epr["ok"]
    with pytest.raises(ValueError, match="executed prefix of record not ok"):
        PS.padding_steps_of_record(ids, epr, {"r0": 0})
    ok = EP.executed_prefix_of_record(["r0"], {"r0": [5, EOS]}, caps={"r0": CAP}, eos_of_record=[EOS], lag=1)
    steps = PS.padding_steps_of_record(ids, ok, {"r0": 0})
    assert sorted(steps) == [3, 4, 5] and all(len(v) == len(OPS) for v in steps.values())
    geo = hashlib.sha256(b"geo").digest()
    rd = _record(tmp_path, served={"r0": [5, EOS]}, lag=0, executed_steps={"r0": [0, 1, 2]}, geo=geo)       # LAG 0 but 3 steps committed
    res = PS.padded_map_of_record(rd, run_id="test-run", geo_digest=geo)
    assert res["ok"] is False and "disagree" in res["why"]
    with pytest.raises(ValueError, match="no lifted container"):
        PS.padding_layout([dict(ids[0], dtype="complex128")])


#: sha256 of the conversion's outcomes over `_corpus` (`json.dumps(..., sort_keys=True, default=_plain)`, the run directory replaced),
#: messages included: a change to what the conversion decides or says changes it
OUTCOMES_SHA256 = "6f16a33b80b40734410c77c44e253efa9feea2718d5602512d81466673595688"
RECORDS = Path(__file__).parents[1] / "records" / "data"
B1 = {"served": {"r0": [5, 6, EOS]}, "lag": 1, "executed_steps": {"r0": [0, 1, 2, 3]}}
CAP_STOP = {"served": {"r0": [1, 2, 3, 4, 5, 6]}, "lag": 2, "executed_steps": {"r0": list(range(CAP))}}


def _edit(p: Path, change) -> None:
    d = json.loads(p.read_text())
    change(d)
    p.write_text(json.dumps(d))


def _lag_in_the_stop_rule_only(wp: dict) -> None:
    del wp["lag"]
    wp["stop_rule"].update(lag=1, eos_lag_rows=1)


def _one_process_capture(t: dict) -> None:
    t["requests"][0].update(arrive_step_realized=0, first_step=0, finish_step=3, finish_reason="stop", served_output_len=3, executed_prefix_len=4)
    t["lag"] = {"value": 1, "async_scheduling": True}


#: name -> (the record, a change to its workload Program, a change to its tokens.json (None: the record's; "absent": no file))
CASES = {
    "b1_lag0": (dict(B1, lag=0, executed_steps={"r0": [0, 1, 2]}), None, None),
    "b1_lag1": (B1, None, None),
    "b2_mixed": ({"served": {"r0": [5, EOS], "r1": [1, 2, 3, 4, 5, 6]}, "lag": 1, "executed_steps": {"r0": [0, 1, 2], "r1": list(range(CAP))}},
                 None, None),
    "lag_disagrees": (dict(B1, lag=0), None, None),
    "cap_stop": (CAP_STOP, None, None),
    "cap_stop_no_lag_block": (CAP_STOP, lambda wp: wp.pop("lag"), None),
    "no_lag_block": (B1, lambda wp: wp.pop("lag"), None),
    "lag_in_the_stop_rule_only": (B1, _lag_in_the_stop_rule_only, None),
    "lag_source": (B1, lambda wp: wp["lag"].update(source="manifest"), None),
    "no_stop_rule": (B1, lambda wp: wp.pop("stop_rule"), None),
    "eos_null": (B1, lambda wp: wp["stop_rule"].update(eos_token_ids=None), None),
    "ignore_eos": (B1, lambda wp: wp["stop_rule"].update(eos_token_ids=[], ignore_eos=True), None),
    "no_tokens_json": (B1, None, "absent"),
    "no_served_requests": (B1, None, lambda t: t.update(requests=[])),
    "unknown_request_id": (B1, None, lambda t: t["requests"][0].update(request_id="uuid-zz")),
    "one_process_capture": (B1, None, _one_process_capture),
}


def _plain(x: Any) -> Any:
    return x.hex() if isinstance(x, bytes) else dataclasses.asdict(x) if dataclasses.is_dataclass(x) else str(x)


def _corpus(tmp: Path) -> dict[str, Any]:
    geo = hashlib.sha256(b"geo").digest()
    out = {}
    for name, (kw, program, tokens) in CASES.items():
        rd = _record(tmp / name, geo=geo, **kw)
        if program is not None:
            _edit(rd / "build_workload" / "workload_program.json", program)
        if tokens == "absent":
            (rd / "match" / "capture" / "tokens.json").unlink()
        elif tokens is not None:
            _edit(rd / "match" / "capture" / "tokens.json", tokens)
        out[name] = PS.padded_map_of_record(rd, run_id="test-run", geo_digest=geo)
    return out


def test_the_conversion_reads_the_records_to_the_same_outcomes(tmp_path):
    blob = json.dumps(_corpus(tmp_path), sort_keys=True, default=_plain).replace(str(tmp_path), "<run>")
    assert hashlib.sha256(blob.encode()).hexdigest() == OUTCOMES_SHA256


def test_a_lag_declared_only_in_the_stop_rule_converts_as_one_in_the_lag_block(tmp_path):
    """G6 takes the LAG from the stop rule when the Program has no `lag` block (`DeclaredStop.declared_lag`), and so does the conversion."""
    got = _corpus(tmp_path)
    assert got["lag_in_the_stop_rule_only"]["ok"] and got["lag_in_the_stop_rule_only"]["report"]["lag"] == 1
    b1, stop_rule_only = (json.dumps(got[k], sort_keys=True, default=_plain) for k in ("b1_lag1", "lag_in_the_stop_rule_only"))
    assert stop_rule_only == b1


@pytest.mark.parametrize("record,field,change", [
    ("match/capture/tokens.json", "requests", lambda t: t.pop("requests")),
    ("match/capture/tokens.json", "requests[0].request_id", lambda t: t["requests"][0].pop("request_id")),
    ("match/capture/tokens.json", "requests[0].generated_token_ids", lambda t: t["requests"][0].pop("generated_token_ids")),
    ("build_workload/workload_program.json", "stop_rule.eos_token_ids", lambda wp: wp["stop_rule"].pop("eos_token_ids")),
    ("build_workload/workload_program.json", "lag.value", lambda wp: wp["lag"].pop("value")),
])
def test_a_record_missing_a_required_field_is_refused_by_name(tmp_path, record, field, change):
    geo = hashlib.sha256(b"geo").digest()
    rd = _record(tmp_path, geo=geo, **B1)
    _edit(rd / record, change)
    res = PS.padded_map_of_record(rd, run_id="test-run", geo_digest=geo)
    assert res["ok"] is False and res["why"] == (f"executed prefix of record not placeable: {rd / record}: field {field!r} is missing: "
                                                 "every writer of the record writes it")


def test_the_real_records_load_through_the_conversion(tmp_path):
    """The real served tokens and Program of `tests/records/data`, with the Program's request-id table built from its requests."""
    tokens, gp = (json.loads((RECORDS / n).read_text()) for n in ("served_tokens.json", "global_program.json"))
    (tmp_path / "match" / "capture").mkdir(parents=True)
    (tmp_path / "match" / "capture" / "tokens.json").write_bytes((RECORDS / "served_tokens.json").read_bytes())
    wp = dict(gp, request_ids={"workload_request_ids": {r["request_id"]: r["workload_request_id"] for r in gp["requests"]}})
    assert PS._served_tokens_of_record(tmp_path, wp) == {"r0": tokens["requests"][0]["generated_token_ids"]}
    assert len(tokens["requests"][0]["generated_token_ids"]) == 32
