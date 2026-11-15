"""GLM-5.3's MTP draft-layer captures without a GPU (`glm53_mtp_eh_norm_capture_gpu`, `glm53_mtp_eh_proj_capture_gpu`,
`glm53_mtp_indexer_logits_capture_gpu`): each is a registered property; and each one's `main` end to end with the GPU stood in for
by the twin's words, through the spawned worker pool to the sealed record, the GO / NO-GO line and the exit status, so a slip
between the comparisons and the record shows here and not in the card's lease (the GPU runs are
`tests/properties/glm53_mtp_capture_card.sh NAME` on an H200)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import contextlib
import json
import sys
import types

import numpy as np
import pytest
from verity.definitions.numerics import dsa_norm_rows as NR
from verity.definitions.registry import bf16_gemm_check as BC

from tests.properties import glm53_mtp_eh_norm_capture_gpu as EN
from tests.properties import glm53_mtp_eh_proj_capture_gpu as EP
from tests.properties import glm53_mtp_indexer_logits_capture_gpu as IL
from verity_vllm.records.property_record import REGISTRY, digest_of, seal

CAPTURES = {"glm53_mtp_eh_norm_capture": EN, "glm53_mtp_eh_proj_capture": EP, "glm53_mtp_indexer_logits_capture": IL}
FAKE_PTX = (".version 8.7\n.target sm_90a\n.visible .entry _fused_eh_norm_kernel(\n)\n.reqntid 128\n{\n"
            "\tld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];\n\tshfl.sync.bfly.b32\t%r5, %r1, 16, 31, -1;\n"
            "\tdiv.full.f32 \t%r6, %r5, 0f45C00000;\n\tret;\n}\n\t.file\t1 \"/a/kernels.py\"\n")


def _bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    u = rng.standard_normal(shape).astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _stub_device(monkeypatch, X) -> None:
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})


def _stub_vllm_config(monkeypatch) -> None:
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)


@pytest.mark.parametrize("name", sorted(CAPTURES))
def test_each_capture_is_a_registered_property(name: str) -> None:
    X = CAPTURES[name]
    assert REGISTRY[name].schema == X.SCHEMA and f"{name}.json" == X.RECORD
    doc = seal({"schema": X.SCHEMA, "gate": {"ok": False}}, False)                     # the record a failed device gate leaves
    assert doc["ok"] is False and doc["digest"] == digest_of(doc)


# ------------------------------------------------------------------------------------------------------------------- eh_norm
def _eh_launch(d, o: dict[str, np.ndarray], flip: tuple[int, int] | None = None) -> dict[str, str]:
    """The twin's words as the kernel's: both halves of `fused_eh_norm` on these operands, one word flipped or none."""
    out = np.concatenate([NR.block_rmsnorm_rows(EN.zeroed(o["e"], o["pos"]), o["ew"], EN.EPS, EN.BLOCK, EN.WARPS)["out"],
                          NR.block_rmsnorm_rows(o["p"], o["hw"], EN.EPS, EN.BLOCK, EN.WARPS)["out"]], axis=1)
    if flip:
        out[flip] ^= 1
    return EN.DS._save(d, pos=o["pos"], e=o["e"], p=o["p"], ew=o["ew"], hw=o["hw"], out=out)


def test_an_eh_norm_job_finds_a_flipped_word_and_the_position_test(tmp_path) -> None:
    rng = np.random.default_rng(7)
    o = {"pos": np.array([0, 1 << 32, 7, 0], dtype=np.int64), "e": _bf16(rng, (4, EN.H)), "p": _bf16(rng, (4, EN.H)),
         "ew": _bf16(rng, (EN.H,)), "hw": _bf16(rng, (EN.H,))}
    good = EN.compare({"case": "served", "span": [0, 4], "sample": [], "paths": _eh_launch(tmp_path / "a", o)})
    assert {h: (r["rows"], r["words"], r["mismatch"]) for h, r in good["halves"].items()} == {h: (4, 4 * EN.H, 0) for h in EN.HALVES}
    # rows 0 and 3 are +0 only under the int64 test; row 1 (2^32: low word 0) is +0 under the low-word test alone
    refuted = good["halves"]["e"]["refuted"]
    assert refuted["position_ignored"] >= 2 * EN.H // 2 and refuted["low_word_only"] >= EN.H // 2
    bad = EN.compare({"case": "served", "span": [0, 4], "sample": [], "paths": _eh_launch(tmp_path / "b", o, flip=(2, EN.H + 5))})
    assert (bad["halves"]["e"]["mismatch"], bad["halves"]["p"]["mismatch"], bad["halves"]["p"]["rows_differ"]) == (0, 1, 1)
    first = bad["halves"]["p"]["first_mismatch"]
    assert (first["row"], first["col"]) == (2, 5) and first["kernel"] == first["want"] ^ 1


@pytest.mark.slow
def test_eh_norm_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU stood in for (the twin's words as the kernel's in a served and a probe launch of 64 tokens, a fake PTX
    in the Triton cache): the real probes and operands, the jobs through the spawned worker pool, the summary, the sealed record,
    the GO line and the exit status.  Slow (the reference evaluator of `MtpEhNorm_v1` at H 6144 on every sampled row): it is
    the only test that seals this capture's own document, which the card's run otherwise does first."""
    rec = EN.ptx_record(FAKE_PTX)
    for k, v in (("T", 64), ("ZERO_POS", (0, 3, 40, 63)), ("SAMPLE", (1, 62)), ("PTX_CODE_SHA256", rec["code_sha256"]),
                 ("PTX_COUNTS", rec["counts"]), ("SASS_COUNTS", {})):
        monkeypatch.setattr(EN, k, v)
    _stub_device(monkeypatch, EN)
    monkeypatch.setattr(EN.DS, "environment", lambda work: ({"TRITON_CACHE_DIR": str(work / "triton_cache")}, {"VLLM_USE_DEEP_GEMM": "0"}))
    monkeypatch.setattr(EN.DS, "served", lambda: {"VLLM_USE_DEEP_GEMM": True, "VLLM_BATCH_INVARIANT": False, "ok": True})
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__version__="stub"))

    def launch(outdir, o, case):
        return ({"case": case, "tokens": EN.T, "shape": [EN.T, 2 * EN.H], "dtype": "torch.bfloat16",
                 "strides": {"e": [EN.H, 1], "p": [EN.H, 1], "out": [2 * EN.H, 1]}, "checks": {"deterministic": True, "shape": True}},
                _eh_launch(outdir / "words" / case, o))

    real_ptx = EN.ptx

    def ptx(cache, outdir):
        (cache / "k").mkdir(parents=True)
        (cache / "k" / f"{EN.KERNEL}.ptx").write_text(FAKE_PTX)
        return real_ptx(cache, outdir)

    monkeypatch.setattr(EN, "launch", launch)
    monkeypatch.setattr(EN, "ptx", ptx)
    out = tmp_path / "out"
    assert EN.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 0
    doc = json.loads((out / EN.RECORD).read_text())
    assert doc["schema"] == EN.SCHEMA and doc["digest"] == digest_of(doc) and doc["ok"] is True
    assert doc["ptx"]["ok"] and doc["bind"]["ok"] and doc["triton"] == "stub" and doc["env_cleared"] == {"VLLM_USE_DEEP_GEMM": "0"}
    cases = doc["summary"]["cases"]
    assert {k: (t["rows"], t["words"], t["mismatch"], t["twin_mismatch"]) for k, t in cases.items()} == {
        f"{c}/{h}": (64, 64 * EN.H, 0, 0) for c in EN.CASES for h in EN.HALVES}
    probe_rows = max(p["rows"] for p in doc["probes"].values())
    assert cases["served/e"]["reference"] == 2 + len(EN.SPECIAL) + 4 + 1 and cases["probe/p"]["reference"] == 18 + probe_rows
    assert all(n >= EN.PER_ALT for alts in doc["summary"]["refuted"].values() for n in alts.values())
    assert {"position_ignored", "low_word_only"} <= set(doc["summary"]["refuted"]["e"])
    assert sorted(p.suffix for p in (out / "ptx").iterdir()) == [".ptx", ".sass"]
    printed = capsys.readouterr().out
    assert f"GLM53-MTP-EH-NORM GO {doc['digest']} served=True" in printed and "ptx=True bind=True" in printed


# ------------------------------------------------------------------------------------------------------------------- eh_proj
ROUTES = [[1, 3, None], [4, 6, 2], [7, 12, None]]


def _stub_eh_proj(monkeypatch, flip_m: int | None) -> dict:
    """`run`'s GPU pieces at K 256, N 8 and twelve token counts: the scan's routes as ROUTES, each launch the twin's words (the
    serial chain's, or the split's with its crafted probe), M `flip_m`'s first row with one word flipped."""
    for k, v in (("K", 256), ("N", 8), ("M_MAX", 12), ("LAUNCH_M", (1, 2, 5)), ("ROWS_ALL", 8), ("WINDOW", 2), ("CHUNK", 4),
                 ("SCAN_CHUNKS", ((1, 6), (7, 12)))):
        monkeypatch.setattr(EP, k, v)
    _stub_device(monkeypatch, EP)
    _stub_vllm_config(monkeypatch)

    def default_unquantized_gemm(layer, x, w, bias):
        raise AssertionError("the launches are stood in for")

    utils = types.ModuleType("vllm.model_executor.layers.utils")
    utils.dispatch_unquantized_gemm = lambda: default_unquantized_gemm
    for name in ("vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    seen: dict = {"scans": [], "launches": []}
    monkeypatch.setattr(EP.BG, "environment", lambda served_default, cache: ({"TRITON_CACHE_DIR": str(cache)}, {}))
    monkeypatch.setattr(EP.BG, "switch", lambda served_default: {"ok": True, "cublas_version": 120900})
    monkeypatch.setattr(EP.BG, "configuration", lambda served_default: {"ok": True})
    monkeypatch.setattr(EP.BG, "distributed", lambda work: None)
    monkeypatch.setattr(EP.BG, "Weights", lambda work, seed: None)

    def scan_chunk(a, outdir, lo, hi):
        seen["scans"].append((lo, hi))
        return {"ok": True, "returncode": 0, "cases": {"glm53-mtp/linear/mtp.eh_proj": {"classes": [[lo, hi]]}}}

    def route_table(scans, run_record):
        assert len(scans) == 2 and run_record == "this run"
        return {"routes": {EP.BG.route_key(EP.MODEL, EP.PHASE, EP.NAME): ROUTES}, "conditions": {"cublas_version": 120900}, "m_max": 12}

    w = _bf16(np.random.default_rng(3), (EP.N, EP.K))

    def launch(work, weights, gates, model, phase, name, M, seed, served_default, scan_split=None):
        assert (model, phase, name, served_default) == (EP.MODEL, EP.PHASE, EP.NAME, True)
        assert EP.BG.MODELS[model].linear == ((name, EP.K, EP.N),)                     # the shape the real launch reads
        seen["launches"].append((M, scan_split))
        x = _bf16(np.random.default_rng(seed), (M, EP.K))
        out = BC.linear_rows(x, w) if scan_split is None else BC.cublas_splitk_rows(x, w, scan_split, out="bf16")
        if flip_m == M:
            out[0, 1] ^= 1
        d = work / f"M{M}"
        paths, probe, probe_paths = EP.BG._save(d, x=x, w=w, out=out), None, None
        if scan_split:
            px, pw, differ = EP.BG.splitk_probe_words(EP.K, scan_split, "bf16", EP.N)
            probe = {"columns": EP.N, "discriminator": differ}
            probe_paths = EP.BG._save(d / "splitk_probe", x=px[None], w=pw, out=BC.cublas_splitk_rows(px[None], pw, scan_split, out="bf16"))
        rec = {"model": model, "phase": phase, "case": name, "M": M, "K": EP.K, "N": EP.N, "operands_ok": True, "deterministic": True,
               "us": 2.5, "splitk_probe": probe,
               "profile": {"ok": True, "launches": [{"name": "nvjet_stub"}], "reduction_kernels": ["reduce_stub"] if scan_split else []}
               | ({"split_k": scan_split} if scan_split else {})}
        return rec, {"model": model, "phase": phase, "case": name, "M": M, "K": EP.K, "N": EP.N, "kernel": None, "paths": paths,
                     "split_k": scan_split, "probe_paths": probe_paths}

    monkeypatch.setattr(EP, "scan_chunk", scan_chunk)
    monkeypatch.setattr(EP.BG, "route_table", route_table)
    monkeypatch.setattr(EP.BG, "launch", launch)
    return seen


def test_eh_proj_s_shape_is_in_the_sibling_capture_s_models_only_while_a_run_lasts() -> None:
    """Importing the capture leaves `glm53_bf16_gemm_capture_gpu.MODELS` the sibling's own, whose tests and captures read every
    model in it; a run or a scan process holds this shape there, and one that raises takes it out again."""
    before = dict(EP.BG.MODELS)
    assert EP.MODEL not in before
    with pytest.raises(RuntimeError, match="a phase that raises"), EP.registered():
        assert EP.BG.MODELS[EP.MODEL].linear == ((EP.NAME, EP.K, EP.N),)
        assert EP.BG.route_key(EP.MODEL, EP.PHASE, EP.NAME) == f"linear {EP.K} {EP.N}"
        raise RuntimeError("a phase that raises")
    assert before == EP.BG.MODELS


def test_eh_proj_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the scan's routes and the launches stood in for at a small shape: the launch list the routes give (LAUNCH_M, each
    route class's first M, the first M past ROWS_ALL, M_MAX), each launch at its route's split, the jobs through the spawned worker
    pool, the probe's hypotheses, the sealed record, the GO line and the exit status; then one flipped word is that launch's
    NO-GO and exit 1."""
    seen = _stub_eh_proj(monkeypatch, None)
    out = tmp_path / "out"
    assert EP.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 0
    assert seen == {"scans": [(1, 6), (7, 12)], "launches": [(1, None), (2, None), (4, 2), (5, 2), (9, None), (12, None)]}
    doc = json.loads((out / EP.RECORD).read_text())
    assert doc["schema"] == EP.SCHEMA and doc["digest"] == digest_of(doc) and doc["ok"] is True and not doc["errors"]
    assert doc["routes"] == ROUTES and doc["expected"] == [1, 2, 4, 5, 9, 12] and doc["scan"]["ok"] is True
    assert doc["selection"] == {"linear_gemm": "default_unquantized_gemm", "ok": True}
    assert list(doc["bind"]["bindings"].values()) == [True, True] and doc["bind"]["ok"]
    rows = {"M1": 1, "M2": 2, "M4": 4, "M5": 5, "M9": 6, "M12": 6}                    # past ROWS_ALL: three windows of two rows
    assert {k: (c["split"], c["totals"]["rows"], c["totals"]["words"], c["totals"]["mismatch"], c["ok"])
            for k, c in doc["cases"].items()} == {k: (2 if k in ("M4", "M5") else None, n, n * EP.N, 0, True) for k, n in rows.items()}
    assert all((c["hypothesis"] or {}).get("fits") == ([BC.CUBLAS_SPLITK_HYPOTHESIS] if c["split"] else None)
               for c in doc["cases"].values())
    assert doc["totals"]["rows"] == sum(rows.values()) and doc["totals"]["reference_coords"] > 0 and doc["totals"]["twin_mismatch"] == 0
    assert sorted(p.name for p in (out / "words").iterdir()) == sorted([*rows, "w.npy"])
    assert f"GLM53-MTP-EH-PROJ GO {doc['digest']}" in capsys.readouterr().out
    assert EP.MODEL not in EP.BG.MODELS
    _stub_eh_proj(monkeypatch, 5)
    assert EP.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / EP.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and doc["totals"]["mismatch"] == 1
    assert [k for k, c in doc["cases"].items() if not c["ok"]] == ["M5"]
    assert doc["cases"]["M5"]["first_mismatch"]["row"] == 0 and doc["cases"]["M5"]["first_mismatch"]["column"] == 1
    assert "GLM53-MTP-EH-PROJ NO-GO" in (printed := capsys.readouterr().out) and "no-go=['M5']" in printed


@pytest.mark.parametrize("splits", [2, 3, 4, 5])
def test_the_splitk_probe_separates_every_hypothesis_at_eh_proj_s_shape(splits: int) -> None:
    x, w, differ = EP.BG.splitk_probe_words(EP.K, splits, "bf16")
    assert differ.pop(BC.CUBLAS_SPLITK_HYPOTHESIS) == 0 and differ and all(n > 0 for n in differ.values())


# ------------------------------------------------------------------------------------------------------------ indexer logits
def _stub_indexer(monkeypatch, flip: str | None) -> None:
    """`run`'s GPU pieces at max_model_len 128: the capture's own operands, flattening and key table, every logit [0, seq_len)
    of every row the twin's word, entry 3 of case `flip` flipped."""
    monkeypatch.setattr(IL, "CASES", (("edges", (6, 7, 70, 0)), ("batch", (None, 0))))
    monkeypatch.setattr(IL, "MAX_MODEL_LEN", 128)
    _stub_device(monkeypatch, IL)
    monkeypatch.setattr(IL.DS, "environment", lambda work: ({}, {}))
    monkeypatch.setattr(IL.DS, "served", lambda: {"VLLM_USE_DEEP_GEMM": True, "VLLM_BATCH_INVARIANT": False, "has_deep_gemm": True, "ok": True})
    monkeypatch.setattr(IL, "selection", lambda: {"native_next_n_6": False, "supports_native_decode_6": False,
                                                  "supports_varlen": False, "ok": True})

    def launch(outdir, name, o, seed):
        lens, bt = IL.flattened(o["L"], o["bt"])
        assert bt.shape == (IL.NEXT_N * len(o["L"]), 2)
        k, ks, first = IL.key_table(o)
        req = np.repeat(np.arange(len(o["L"])), IL.NEXT_N)
        rr = np.repeat(np.arange(lens.size), lens).astype(np.int64)
        cc = np.concatenate([np.arange(n) for n in lens]).astype(np.int64)
        kk = first[req[rr]] + cc
        got, nonfinite = IL.DS.logit_rows(o["q"][rr], o["w"][rr], k[kk], ks[kk])
        assert not nonfinite.any()
        if name == flip:
            got[3] ^= 1
        paths = IL.DS._save(outdir / "words" / name, q=o["q"], w=o["w"], k=k, ks=ks, r=rr, c=kk, key=cc, got=got)
        rec = {"case": name, "requests": len(o["L"]), "rows": int(lens.size), "lengths": o["L"].tolist(), "entries": int(rr.size),
               "checks": {"flatten_lens": True, "flatten_bt": True, "decode_lens_one": True, "shape": True, "deterministic": True}}
        return rec, [{"phase": "logits", "case": name, "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [a, b]}
                     for a, b in IL.DS._chunks(rr.size, IL.DS.JOB_LOGITS)]

    monkeypatch.setattr(IL, "launch", launch)


def test_the_flattening_gives_each_request_s_six_rows_their_context_lengths() -> None:
    lens, bt = IL.flattened(np.array([6, 7, 3, 0], dtype=np.int64), np.arange(8, dtype=np.int32).reshape(4, 2))
    assert lens.tolist() == [1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    assert bt.tolist() == [[2 * (i // 6), 2 * (i // 6) + 1] for i in range(24)]


def test_indexer_logits_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the device, vLLM's selection and the launches stood in for at a small context: the capture's operands,
    flattening and key table, the jobs through the spawned worker pool, the sealed record, the GO line and the exit status; then
    one flipped logit is its case's mismatch, NO-GO and exit 1."""
    _stub_indexer(monkeypatch, None)
    out = tmp_path / "out"
    assert IL.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 0
    doc = json.loads((out / IL.RECORD).read_text())
    assert doc["schema"] == IL.SCHEMA and doc["digest"] == digest_of(doc) and doc["ok"] is True and not doc["errors"]
    assert [x["case"] for x in doc["launches"]] == ["edges", "batch"] and doc["launches"][0]["lengths"] == [6, 7, 70, 0]
    assert doc["bind"] == {"bindings": {"DsaMqaLogit_v3{H=32,D=128}": True}, "ok": True}
    edges = 21 + 27 + sum(range(65, 71))
    assert doc["cases"]["edges"] | {"reference": 2} == {"words": edges, "mismatch": 0, "unresolved": 0, "reference": 2,
                                                        "twin_mismatch": 0, "first_mismatch": None}
    assert doc["cases"]["batch"]["words"] == doc["launches"][1]["entries"] > 0 and doc["cases"]["batch"]["mismatch"] == 0
    assert f"GLM53-MTP-INDEXER-LOGITS GO {doc['digest']}" in capsys.readouterr().out
    _stub_indexer(monkeypatch, "edges")
    assert IL.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / IL.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False
    assert (doc["cases"]["edges"]["mismatch"], doc["cases"]["batch"]["mismatch"]) == (1, 0)
    first = doc["cases"]["edges"]["first_mismatch"]                                    # entry 3: row 2's first key
    assert (first["row"], first["key"]) == (2, 0) and first["kernel"] == first["want"] ^ 1
    assert "GLM53-MTP-INDEXER-LOGITS NO-GO" in capsys.readouterr().out
