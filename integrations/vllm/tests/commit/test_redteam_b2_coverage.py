"""Red-team (claim 3, the Commit's coverage contract): at B >= 2 two families can lose committed words while both coverage checks of record
-- `binding.coverage_check` over `load_required_manifest`'s population (runs.jsonl `commit.binding.coverage`) and the layouts-level
`manifest.format.coverage_check` (`commit.coverage`, what `tests/regression/checks/coverage.py` recomputes) -- report 0 missing.

* The hidden attention stream (`fa2.m1/L<k>`, `fa3.m1/L<k>`).  A batch-level hidden row is required op-level: the loader rewrites its
  member to `*` (binding.py `load_required_manifest`, "member = '*'") and `coverage_check` accepts any member of the op with no range
  (the `mem == "*"` branch); the layouts check accepts ">= n_seq distinct `s<i>` prefixes or any unsegmented member" and drops the range
  (`got, hi = numels, None`).  X-03 (`fa2_plane_classes.check_step`, a Commit rule) holds the plane NAMES to the declared class but never
  reads a tensor's size.  The replay leaves the stream value-unchecked (check/replay/coverage.py `QUERY_FAMILY_MECHANISM`: "C1 / C3"),
  so coverage is the one check that the stream was committed in full.  At B=1 the same truncation is missing in both checks.
* The MoE block-stream planes (`moe/L<k>/<plane>`).  A plane whose rows are not a multiple of the step's tokens is bound WHOLE to every
  request ("not a token multiple: bind whole, reported as an extra shape" -- nothing reports it), so a plane that carries only the first
  request's rows covers the second request's request-local range by containment; the layouts check compares a request-local range with
  the batched plane's size.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity_vllm.acquire import fa2_plane_classes as FP
from verity_vllm.commit import binding as B
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.query.manifest import format as MF

CHUNK = 64
PD = hashlib.sha256(b"program").hexdigest()
NH = KVH = 4
D = 64
LP = {"r0": 5, "r1": 3}                       # both prompts arrive at engine step 0: one prefill launch over two sequences
WIDTHS = dict(zip(FP.PLANES, (128, 128, 64, 8, 64, 4)))   # N, N, N/2, NB*8, NB*D, 4 at BN = 128, NB = 1 (native_collect.cpp `widths`)
MOE_H = 8


def _u32(shape: tuple[int, ...]) -> tuple[bytes, str, tuple[int, ...]]:
    return np.arange(int(np.prod(shape)), dtype=np.uint32).tobytes(), "uint32", shape


def _rows(lp: dict[str, int]) -> list[tuple[str, int, int]]:
    out, lo = [], 0
    for rid, n in lp.items():
        out.append((rid, lo, lo + n))
        lo += n
    return out


def _step(lp: dict[str, int], hidden: list[tuple[str, int]], moe_rows: int | None = None) -> list[tuple]:
    """One engine step's committed tensors: the per-token ids, one sampled id per sequence, the given hidden tensors (name, slabs
    actually staged), and optionally the MoE `sum` plane with `moe_rows` token rows."""
    tokens = sum(lp.values())
    out: list[tuple] = [("model.embed_tokens/in", bytes(range(tokens))), ("runner.sampler/sampled_token_ids", bytes(len(lp)))]
    for name, slabs in hidden:
        geo, plane = (name.split("/")[-2], name.split("/")[-1]) if name.count("/") == 4 else (name.split("/")[-1], None)
        _h, sl, nb = (int(x) for x in geo[1:].split("x")[:3])
        if plane is None:                                                       # a whole launch: mat_total_words of its geometry
            raw, dt, _ = _u32((MF.mat_total_words(_h, sl, nb, 128, D),))
            raw = raw[: len(raw) * slabs // _h]
            out.append((name, raw, dt, (len(raw) // 4,)))
        else:
            out.append((name, *_u32((slabs, sl, WIDTHS[plane]))))
    if moe_rows is not None:
        out.append(("moe/L0/sum", bytes(moe_rows * MOE_H), "uint8", (moe_rows, MOE_H)))
    return out


def _seq_planes(lp: dict[str, int]) -> list[str]:
    return [f"fa2.m1/L0/s{i}/g{NH}x{sl}x1/{p}" for i, sl in enumerate(lp.values()) for p in FP.PLANES]


def _manifest(tmp_path, lp: dict[str, int], *, hidden: bool = True, moe: bool = False) -> tuple[str, dict]:
    """The manifest `required_manifest` + `compose.compose_global` write for this step: per-request ids / sampled ids (and MoE `sum`),
    and at B >= 2 ONE batch-level hidden row (request "*", the co-scheduled `requests`, the padded launch geometry); at B=1 the request's
    own hidden row at its launch geometry."""
    ids: list[dict] = []
    for rid, n in lp.items():
        base = {"program_digest": PD, "request_id": rid, "step": 0, "engine_step": 0, "invocation": 0, "rank": 0, "rows": n}
        ids.append(dict(base, family="input_token_ids", op_path="model.embed_tokens", output_member="in", element_range=[0, n]))
        ids.append(dict(base, family="sampled_token_ids", op_path="runner.sampler", output_member="sampled_token_ids", element_range=[0, 1]))
        if moe:
            ids.append(dict(base, family="moe_block_stream", op_path="moe/L0", output_member="sum", element_range=[0, n * MOE_H]))
    if hidden:
        HB, M, NB, BN, gname = MF.fa2_geometry(max(lp.values()), max(lp.values()) * len(lp), NH, KVH, D, len(lp))
        row = {"family": "fa2_hidden_m1_stream", "program_digest": PD, "invocation": 0, "op_path": "fa2.m1/L0", "output_member": gname,
               "element_range": [0, MF.mat_total_words(HB, M, NB, BN, D)], "rank": 0, "dtype": "u32"}
        if len(lp) > 1:
            row |= {"request_id": "*", "requests": list(lp), "step": "*", "engine_step": 0}
        else:
            row |= {"request_id": next(iter(lp)), "step": 0, "engine_step": 0}
        ids.append(row)
    doc = {"schema": MF.SCHEMA, "global": len(lp) > 1, "addressing": "request-local", "program_digest": PD, "program_digest_kind": "workload",
           "manifest_digest": MF.manifest_digest(ids), "complete": True, "unmodelled": {},
           "requests_declared": [{"request_id": rid, "index": i, "prompt_len": n, "max_tokens": 1, "arrive_step": 0} for i, (rid, n) in enumerate(lp.items())],
           "engine_steps": ([{"engine_step": 0, "members": [{"request_id": rid, "step": 0, "rows": n} for rid, n in lp.items()],
                              "rows": sum(lp.values()), "n_seq": len(lp)}] if len(lp) > 1 else [0]),
           "composition_reference": {"input_token_ids": {"op_path": "model.embed_tokens", "output_member": "in", "per_row": 1},
                                     "sampled_token_ids": {"op_path": "runner.sampler", "output_member": "sampled_token_ids", "per_row": 1}},
           "identities": ids}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc))
    return str(p), doc


def _check(tmp_path, lp: dict[str, int], step: list[tuple], **man_kw) -> dict:
    """Both coverage checks of record on one committed step, plus the X-03 per-layer plane verdict (Build declarations present)."""
    c = NativeHostCommitter(None, run_id="run", chunk=CHUNK, hash_threads=1, program_digest=bytes.fromhex(PD))
    run = c.commit_offline([step])
    rows = {0: _rows(lp)} if len(lp) > 1 else None
    bmap = B.build_binding_map(c._layouts, run.steps, run.run_root, PD, request_rows=rows, chunk=CHUNK)
    path, doc = _manifest(tmp_path, lp, **man_kw)
    pop, hdr = B.load_required_manifest(path, PD, requests=list(lp), steps=1, ranks=[0], arrivals={r: 0 for r in lp})
    layouts = {s: [{"name": m.name, "shape": list(m.shape)} for m in ms] for s, ms in c._layouts.items()}
    names = [m.name for m in c._layouts[0]]
    x03 = FP.check_step([FP.AttnLayerDecl(0, NH, KVH, D)], FP.StepSequences(tuple(lp.values())), names, FP.LayoutPolicy("fa2", compact=True))
    return {"binding": B.coverage_check(pop, bmap), "layouts": MF.coverage_check(doc, layouts), "composition": B.composition_check(hdr, bmap),
            "x03": x03, "bmap": bmap}


def test_honest_b2_prefill_is_covered_and_passes_x03(tmp_path):
    r = _check(tmp_path, LP, _step(LP, [(n, NH) for n in _seq_planes(LP)]))
    assert r["binding"]["ok"] and r["binding"]["missing_n"] == 0, r["binding"]["missing"]
    assert r["layouts"]["ok"] and r["layouts"]["missing_n"] == 0, r["layouts"]["missing"]
    assert r["composition"]["ok"], r["composition"]
    assert r["x03"]["ok"], r["x03"]["layers_bad"]


def test_b1_hidden_launch_committed_short_is_missing_in_both_checks(tmp_path):
    lp = {"r0": 5}
    honest = _check(tmp_path, lp, _step(lp, [("fa2.m1/L0/g4x5x1", NH)]))
    assert honest["binding"]["ok"] and honest["layouts"]["ok"], (honest["binding"]["missing"], honest["layouts"]["missing"])
    short = _check(tmp_path, lp, _step(lp, [("fa2.m1/L0/g4x5x1", 1)]))      # one slab of four staged under the launch's name
    assert not short["binding"]["ok"] and short["binding"]["address_mismatch_n"] == 1
    assert not short["layouts"]["ok"] and short["layouts"]["missing_by_family"] == {"fa2_hidden_m1_stream": 1}


def test_b2_hidden_planes_committed_short_keep_their_names_so_x03_passes(tmp_path):
    r = _check(tmp_path, LP, _step(LP, [(n, 1) for n in _seq_planes(LP)]))   # one slab of NH per plane, every plane, both sequences
    assert r["x03"]["ok"], r["x03"]["layers_bad"]
    staged = sum(e["width_bytes"] for e in r["bmap"]["entries"] if e["name"].startswith("fa2.m1/") and e["identity"]["request_id"] == "r0")
    honest = sum(NH * sl * w * 4 for sl in LP.values() for w in WIDTHS.values())
    assert staged * NH == honest


def test_b2_moe_plane_committed_short_is_bound_whole_to_every_request(tmp_path):
    r = _check(tmp_path, LP, _step(LP, [(n, NH) for n in _seq_planes(LP)], moe_rows=LP["r0"]), moe=True)
    r1 = next(e for e in r["bmap"]["entries"] if e["name"] == "moe/L0/sum" and e["identity"]["request_id"] == "r1")
    assert r1["identity"]["element_range"] == [0, LP["r0"] * MOE_H] and r1["element_offset"] == 0   # r1's identity points at r0's rows


@pytest.mark.xfail(strict=True, reason="B>=2 coverage accepts a hidden launch op-level with no size, and binds a short MoE plane whole to every "
                                       "request: a short commit reports 0 missing in both coverage checks")
@pytest.mark.parametrize("mutation", ["hidden_planes_short", "hidden_one_word_whole", "moe_plane_short"])
def test_b2_short_commit_is_missing_in_coverage(tmp_path, mutation):
    planes = [(n, NH) for n in _seq_planes(LP)]
    if mutation == "hidden_planes_short":
        step, kw = _step(LP, [(n, 1) for n in _seq_planes(LP)]), {}
    elif mutation == "hidden_one_word_whole":
        step, kw = _step(LP, [("fa2.m1/L0/g1x1x1", 1)]), {}
        step[-1] = (step[-1][0], step[-1][1][:4], "uint32", (1,))
    else:
        step, kw = _step(LP, planes, moe_rows=LP["r0"]), {"moe": True}
    r = _check(tmp_path, LP, step, **kw)
    assert r["composition"]["ok"], r["composition"]
    assert not r["binding"]["ok"] or not r["layouts"]["ok"], (r["binding"]["missing_n"], r["layouts"]["missing_n"])


def test_b2_one_word_whole_member_is_caught_only_by_x03_declarations(tmp_path):
    step = _step(LP, [("fa2.m1/L0/g1x1x1", 1)])
    step[-1] = (step[-1][0], step[-1][1][:4], "uint32", (1,))
    r = _check(tmp_path, LP, step)
    assert not r["x03"]["ok"] and r["x03"]["layers_bad"][0]
    assert FP.uniform_rule(FP.counts_by_layer([s[0] for s in step]), [0]) == {}    # the no-declaration fallback passes it
