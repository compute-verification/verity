"""spot_proved's staging of the first layer's ``_C::rms_norm`` (`stage.rms_norm_instances`) and the MoE block's ``aten::add``
(`stage.add_instances`) on an in-memory capture, through the match gate: each real token row is staged, the norm on
`RMSNormCudaVector8_v1` at its launch block and the add on `ResidualAdd_v1`, which agrees with torch's f32 add rounded to
bf16 and fails on a flipped word; an add with an alpha is counted, not staged."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PADDED, TOKENS, N = 4, 3, 64


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16(f: np.ndarray) -> np.ndarray:
    u = np.asarray(f, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def _f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _cap(stage):
    C = stage.Capture
    t: dict = {}

    class _Cap:
        rows, real_rows, tokens, padded = C.rows, C.real_rows, C.tokens, C.padded

        def opened(self, e, weights=False):
            return True

        def tensor(self, r, ref):
            return t[ref["sha512"]] if "sha512" in ref else t[ref["param"]]

    def ref(name, a, param=False):
        t[name] = a
        return {"param" if param else "sha512": name, "shape": list(a.shape), "dtype": "torch.bfloat16"}

    rng = np.random.default_rng(9)
    x, o, w = (_bf16(rng.standard_normal(s).astype(np.float32)) for s in ((PADDED, N), (PADDED, N), (N,)))
    from verity.definitions.registry.cuda_rmsnorm import launch_block
    block = f'RMSNormCudaVector8_v1{{N={N},EPS={{"f64":"{float(1e-5).hex()}"}},BLOCK={launch_block(PADDED, N, fused=False)}}}'
    import a11  # on the path `stage.py` puts benchmarks/one_stage on
    spec = stage.block_fn(block)
    normed = np.stack([np.asarray([int(v) for v in a11.evaluate_units(spec, spec, [stage.words(x[i]), stage.words(w)], 1,
                                                                      4_000_000)["call_out"]], dtype=np.uint16)
                       for i in range(PADDED)])
    events = [{"seq": 0, "op": "_C::rms_norm", "layer": 0,
               "in": {"input": ref("x", x), "weight": ref("w", w, True), "epsilon": 1e-5, "result": ref("y0", x)},
               "out": {"result": ref("y", normed)}},
              {"seq": 1, "op": "aten::add", "layer": 3, "in": {"self": ref("a", x), "other": ref("b", o), "alpha": None},
               "out": {"ret0": ref("s", _bf16(_f32(x) + _f32(o)))}},
              {"seq": 2, "op": "aten::add", "layer": 3, "in": {"self": ref("a", x), "other": ref("b", o), "alpha": 2},
               "out": {"ret0": ref("s2", x)}},
              {"seq": 3, "op": "model_output", "layer": None, "in": {}, "out": {"hidden": {"shape": [PADDED, N]}}}]
    cap = _Cap()
    cap.ranks, cap.events, cap.meta, cap.only, cap.t = [0], {0: events}, {0: {"scheduled": {"a": TOKENS}}}, None, t
    return cap, block


def test_the_first_norm_and_the_moe_add_are_gated_per_real_token_row(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    cap, block = _cap(stage)
    unstaged: dict = {}
    norms, adds = list(stage.rms_norm_instances(cap)), list(stage.add_instances(cap, unstaged))
    assert [(i[0], i[1], i[2]) for i in norms] == [(f"RMSNormCudaVector8_N{N}", block, (0, 0, t)) for t in range(TOKENS)]
    assert [(i[0], i[2]) for i in adds] == [(f"ResidualAdd_N{N}", (0, 1, t)) for t in range(TOKENS)]
    assert unstaged["AtenAdd_other"]["calls"] == [{"rank": 0, "seq": 2}]
    for label, blk, key, args_of, want_of in norms + adds:
        g, _ = stage.gate_instance(stage.block_fn(blk), key, args_of, want_of, 4_000_000)
        assert g["agree"] is True, (label, key, g)
    cap.t["s"] = cap.t["s"].copy()
    cap.t["s"][1, 7] ^= 1
    _, blk, key, args_of, want_of = adds[1]
    assert stage.gate_instance(stage.block_fn(blk), key, args_of, want_of, 4_000_000)[0]["first_diff"] == 7
