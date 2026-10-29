import math
import types

import numpy as np
import pytest
import torch

import engine
import matmul
import methodology
import model as M
import report
import run
import spec
import steps


def evaluate(snap, ids, arms, seeds=(0,), calib=None, windows=None, **kw):
    windows = list(range(len(ids))) if windows is None else windows
    return engine.evaluate(snap, ids, windows, [spec.load(a) if isinstance(a, str) else a for a in arms], seeds, calib,
                           device="cpu", log=lambda s: None, hf_check=kw.pop("hf_check", False), **kw)


def ppl(nll: np.ndarray) -> float:
    return float(np.exp(nll.mean(dtype=np.float64)))


def float64_nll(snap, ids) -> np.ndarray:
    """The model in float64 with explicit GQA, causal softmax and RoPE: an independent reference for the bf16 stream."""
    cfg = M.Config.read(snap)
    W = M.Weights(snap)
    t = {k: v.double() for k, v in W.read(list(W.where)).items()}
    x = t["model.embed_tokens.weight"][torch.from_numpy(ids)]
    T = ids.shape[1]
    inv = M.inv_freq(cfg).double()
    f = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()

    def norm(h, w):
        return w * h / torch.sqrt((h * h).mean(-1, keepdim=True) + cfg.eps)

    def lin(p, n, h):
        y = h @ t[f"{p}{n}.weight"].T
        return y + t[f"{p}{n}.bias"] if f"{p}{n}.bias" in t else y

    for li in range(cfg.layers):
        p = f"model.layers.{li}."
        h = norm(x, t[p + "input_layernorm.weight"])
        q, k, v = (lin(p, f"self_attn.{n}_proj", h).view(len(ids), T, -1, cfg.head_dim).transpose(1, 2) for n in "qkv")
        q, k = q * cos + M.rotate_half(q) * sin, k * cos + M.rotate_half(k) * sin
        k, v = k.repeat_interleave(cfg.rep, 1), v.repeat_interleave(cfg.rep, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(len(ids), T, -1)
        x = x + lin(p, "self_attn.o_proj", a)
        h = norm(x, t[p + "post_attention_layernorm.weight"])
        x = x + lin(p, "mlp.down_proj", torch.nn.functional.silu(lin(p, "mlp.gate_proj", h)) * lin(p, "mlp.up_proj", h))
    logits = norm(x, t["model.norm.weight"]) @ t["lm_head.weight"].T
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), torch.from_numpy(ids)[:, 1:].reshape(-1),
                                             reduction="none").view(len(ids), -1).numpy()


@pytest.mark.parametrize("which", ["llama", "qwen"])
def test_the_bf16_stream_is_the_model(which, request, ids):
    snap = request.getfixturevalue(which)
    r = evaluate(snap, ids[:4], [])
    want = float64_nll(snap, ids[:4])
    assert r["nll"]["bf16/s0"].shape == (4, 63)
    assert abs(ppl(r["nll"]["bf16/s0"]) / ppl(want) - 1) < 1e-2
    assert 100 < ppl(want) < 1000


def passed(r):
    return {n: g["failed"] for n, g in r["gates"].items() if g["failed"]} == {}


@pytest.mark.parametrize("which", ["llama", "qwen"])
def test_the_adopted_design_passes_every_gate(which, request, ids, calib):
    snap = request.getfixturevalue(which)
    arms = ["int8-all", "adopted", "adopted-linears", "adopted-attention"]
    r = evaluate(snap, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    g = r["gates"]
    cfg = M.Config.read(snap)
    # per (arm with the adopted attention, layer, KV group): Q's peel and P's two blocks (key 0, then the rest)
    assert g["peel"]["checked"] == 2 * cfg.layers * cfg.kv_heads * 3
    assert g["int8_range"]["checked"] == g["accumulator"]["checked"] > 0 and g["product_exact"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for k in ("int8-all/s0", "adopted/s0", "adopted-linears/s0", "adopted-attention/s0"):
        assert k in r["nll"] and abs(ppl(r["nll"][k]) / bf - 1) < 0.05, k
        assert r["divergence"][k][-1] > 0
    assert 0 < r["attn_rel_err"]["adopted/s0"] < 0.1


def test_q_dithered_attention_passes_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-qkd"], calib=calib)
    assert passed(r), r["gates"]
    cfg = M.Config.read(llama)
    # P's two blocks per (layer, KV group); Q carries no peel
    assert r["gates"]["peel"]["checked"] == 2 * cfg.layers * cfg.kv_heads
    assert abs(ppl(r["nll"]["adopted-qkd/s0"]) / ppl(r["nll"]["bf16/s0"]) - 1) < 0.05


def test_dithered_weights_pass_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-bd", "adopted-qkd"], calib=calib)
    assert passed(r), r["gates"]
    assert abs(ppl(r["nll"]["adopted-bd/s0"]) / ppl(r["nll"]["bf16/s0"]) - 1) < 0.05
    assert not np.array_equal(r["nll"]["adopted-bd/s0"], r["nll"]["adopted-qkd/s0"])


def test_a_noised_head_passes_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-bd-lm", "adopted-bd", "co", "ncp", "cb", "co32c", "co32c-pv", "co32c-pv256"], calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in ("adopted-bd-lm", "co", "ncp", "cb", "co32c", "co32c-pv", "co32c-pv256"):
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.05, a
    assert not np.array_equal(r["nll"]["adopted-bd-lm/s0"], r["nll"]["adopted-bd/s0"])


def test_wide_dithers_pass_every_gate(llama, ids, calib):
    arms = ["adopted-bd-lm", "adopted-bd-lm-d2", "adopted-bd-lm-d4", "co32c-d4"]
    r = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in arms:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.1, a
    assert not np.array_equal(r["nll"]["adopted-bd-lm/s0"], r["nll"]["adopted-bd-lm-d2/s0"])


def test_linear_variants_pass_their_gates(llama, ids, calib):
    arms = [spec.parse({"linear": [{"step": "rotate", "kind": "h"}, {"step": "int8", "group": 32}]}, "grouped"),
            spec.parse({"linear": [{"step": "smooth", "alpha": 0.5}, {"step": "rotate"},
                                   {"step": "int8", "weight_top": 63, "gptq": True}]}, "gptq"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "act_top": 64}],
                        "attention": [{"step": "int8", "p_block": 32, "p_top": 119, "v_scale": "head", "k": "dither"},
                                      {"step": "peel", "operand": "p", "width": 8}]}, "mixed"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "weight": "dither", "weight_top": 63},
                                   {"step": "peel", "alphabet": "bit"}],
                        "attention": [{"step": "int8", "q": "dither", "k": "dither", "v": "dither"}]}, "dither-peel"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "density": 0.5}],
                        "attention": [{"step": "int8", "q": "dither", "q_top": 63, "k": "dither", "p": "dither",
                                       "p_sink": True, "v": "dither", "v_top": 63}]}, "thin-q-pv")]
    r = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in arms:
        assert abs(ppl(r["nll"][f"{a.name}/s0"]) / bf - 1) < 0.1, a.name


def test_seeds(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["int8-linears", "adopted-linears"], seeds=(0, 1), calib=calib)
    assert set(r["nll"]) == {"bf16/s0", "int8-linears/s0", "adopted-linears/s0", "adopted-linears/s1"}
    assert not np.array_equal(r["nll"]["adopted-linears/s0"], r["nll"]["adopted-linears/s1"])
    again = evaluate(llama, ids[:4], ["adopted-linears"], seeds=(0,), calib=calib)
    assert np.array_equal(again["nll"]["adopted-linears/s0"], r["nll"]["adopted-linears/s0"])
    assert np.array_equal(again["nll"]["bf16/s0"], r["nll"]["bf16/s0"])


def test_shards_are_one_run(llama, ids, calib):
    """Two GPUs' shards (contiguous whole groups of REF_BATCH windows) give each window the NLL one run gives it."""
    assert engine.shards(16, 2) == [list(range(8)), list(range(8, 16))]
    assert engine.shards(4, 2) == [[0, 1, 2, 3]] and engine.shards(10, 2) == [list(range(8)), [8, 9]]
    arms = ["adopted"]
    whole = evaluate(llama, ids, arms, calib=calib)
    parts = [evaluate(llama, ids[s], arms, calib=calib, windows=s) for s in engine.shards(len(ids), 2)]
    for k in whole["nll"]:
        assert np.array_equal(whole["nll"][k], np.concatenate([p["nll"][k] for p in parts])), k
    assert parts[1]["gates"]["peel"]["checked"] == 0 and parts[0]["gates"]["peel"]["checked"] > 0


def test_parts_merge_to_one_run(llama, ids, monkeypatch, tmp_path):
    """`merge` of the parts `eval --part I/N` writes gives the results.json one run over every window gives."""
    import json
    monkeypatch.setitem(methodology.CONTEXTS, ids.shape[1], len(ids))
    arms, wrec = run.arms_for(["int8-linears"]), {"sha256": "0" * 64}
    a = types.SimpleNamespace(model="llama-3.2-1b", set="wikitext2", ctx=ids.shape[1])

    def scored(w):  # HF's layers stand compared on the check window, so the gates pass without transformers
        r = evaluate(llama, ids[w], arms, windows=w)
        r["gates"]["hf_reference"] = {"checked": int(methodology.CHECK_WINDOW in w), "failed": 0, "first_failure": None}
        return r

    whole = scored(list(range(len(ids))))
    one = run.merge(a, arms, (0,), wrec, [whole], whole.pop("nll"), 0.0, 1)
    dirs = []
    for i, w in enumerate(engine.shards(len(ids), 2)):
        r, d = scored(w), tmp_path / f"part{i}"
        d.mkdir()
        np.savez(d / "part.npz", **{k.replace("/", "|"): v for k, v in r.pop("nll").items()})
        (d / "part.json").write_text(json.dumps(r))
        (d / "part-args.json").write_text(json.dumps({**vars(a), "spec": ["int8-linears"], "seeds": "0", "plugin": None,
                                                       "part": f"{i}/2", "windows": w, "wrec": wrec, "seconds": 0.0,
                                                       "run": f"r{i}"}))
        dirs.append(str(d))
    assert run.main(["merge", "--out", str(tmp_path / "merged"), *reversed(dirs)]) == 0
    got = json.loads((tmp_path / "merged" / "results.json").read_text())
    assert got["passed"] and got["env"]["run"] == "r0,r1"
    assert {k: v["seeds"] for k, v in got["arms"].items()} == {k: v["seeds"] for k, v in one["arms"].items()}
    with pytest.raises(SystemExit, match="every window once"):
        run.main(["merge", "--out", str(tmp_path / "half"), dirs[0]])


def _sabotage(monkeypatch, kind):
    reg = {k: dict(v) for k, v in spec.REGISTRY.items()}
    monkeypatch.setattr(spec, "REGISTRY", reg)
    if kind == "range":
        @spec.linear_step("overflow", phase="integer")
        def overflow(mm, nxt):
            return nxt(mm.replace(x=mm.x * 2))
        return spec.parse({"linear": [{"step": "int8"}, {"step": "overflow"}]}, "overflow")
    if kind == "peel":
        monkeypatch.setattr(steps, "fold_peel", lambda u, k: u @ k.reshape(*k.shape[:-1], -1, 32).sum(-2).transpose(-1, -2) + 1)
        return spec.load("adopted-attention")
    real = matmul.int_mm
    monkeypatch.setattr(matmul, "int_mm", lambda a, b: real(a, b) + (torch.arange(a.shape[0])[:, None] == 0).long())
    return spec.load("int8-linears")


@pytest.mark.parametrize("kind, gate", [("range", "int8_range"), ("peel", "peel"), ("product", "product_exact")])
def test_a_failed_gate_reports_nothing(kind, gate, monkeypatch, llama, ids, tmp_path):
    arm = _sabotage(monkeypatch, kind)
    r = evaluate(llama, ids[:4], [arm])
    assert r["gates"][gate]["failed"] > 0 and r["gates"][gate]["first_failure"]["where"]
    a = types.SimpleNamespace(model="llama-3.1-8b", set="c4", ctx=2048)
    nll = {k: np.zeros((16, 2047)) for k in r["nll"]}
    shard = {k: r[k] for k in ("gates", "divergence", "attn_rel_err")}
    res = run.merge(a, [arm], (0,), {"sha256": "x"}, [shard], nll, 1.0, 1)
    assert res["passed"] is False
    assert all("seeds" not in e for e in res["arms"].values())
    assert "GATES FAILED" in report.table(res) and gate in report.table(res)


def test_the_hf_reference_gate_must_run(llama, ids):
    """`run.py eval` fails a run whose check window's layers were never compared with HF's."""
    r = evaluate(llama, ids[:4], ["int8-linears"])
    a = types.SimpleNamespace(model="llama-3.1-8b", set="c4", ctx=2048)
    nll = {k: np.zeros((16, 2047)) for k in r["nll"]}
    res = run.merge(a, [spec.load("int8-linears")], (0,), {"sha256": "x"},
                    [{k: r[k] for k in ("gates", "divergence", "attn_rel_err")}], nll, 1.0, 1)
    assert res["passed"] is False and res["gates"]["hf_reference"]["failed"] == 1


def test_hf_parity(llama, qwen, ids):
    """The engine's bf16 layers are HF transformers' LlamaDecoderLayer / Qwen2DecoderLayer with SDPA, bit for bit."""
    pytest.importorskip("transformers")
    for snap in (llama, qwen):
        r = evaluate(snap, ids[:4], [], hf_check=True)
        g = r["gates"]["hf_reference"]
        assert g["checked"] == M.Config.read(snap).layers and g["failed"] == 0, g


def test_row_chunks_change_no_bit(llama, ids, calib, monkeypatch):
    """At 32K the s x s attention work and the loss run in chunks of query rows; the chunks change no bit, since every
    coin and scale is drawn per row or for the whole window, and the products are exact."""
    arms = ["adopted", "int8-all"]
    whole = evaluate(llama, ids[:4], arms, calib=calib)
    monkeypatch.setattr(M, "ROWS_WHOLE", 0)
    monkeypatch.setattr(M, "ROWS_BYTES", 16 * 2 * 64 * 8)
    monkeypatch.setattr(M, "LOSS_WHOLE", 16)
    monkeypatch.setattr(M, "LOSS_ROWS", 24)
    assert M.row_chunk(2, 64) == 16
    chunked = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(chunked), chunked["gates"]
    for k in whole["nll"]:
        assert np.array_equal(whole["nll"][k], chunked["nll"][k]), k


def test_the_readme_plugin_example_runs(llama, ids, monkeypatch, tmp_path):
    """README's "Adding an intervention" example, as written, is a step a spec names and the engine runs and gates."""
    monkeypatch.setattr(spec, "REGISTRY", {k: dict(v) for k, v in spec.REGISTRY.items()})
    code = run.README.read_text().split("## Adding an intervention")[1].split("~~~python\n")[1].split("~~~")[0]
    (tmp_path / "clip.py").write_text(code)
    run.load_plugins([str(tmp_path / "clip.py")])
    arm = spec.parse({"linear": [{"step": "clip", "sigmas": 1}, {"step": "int8"}]}, "clip")
    assert arm.linear[0].kw == {"sigmas": 1.0} and arm.scope == "linears"
    plain = spec.parse({"linear": [{"step": "int8"}]}, "unclipped")
    r = evaluate(llama, ids[:4], [arm, plain])
    assert passed(r) and r["gates"]["int8_range"]["checked"] > 0
    assert not np.array_equal(r["nll"]["clip/s0"], r["nll"]["unclipped/s0"])


def test_check_window_runs_the_costly_gates_only(llama, ids):
    r = evaluate(llama, ids[4:8], ["int8-attention"], windows=[4, 5, 6, 7])
    assert methodology.CHECK_WINDOW not in [4, 5, 6, 7]
    assert r["gates"]["product_exact"]["checked"] == 0 and r["gates"]["int8_range"]["checked"] > 0
