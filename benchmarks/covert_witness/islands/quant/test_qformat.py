"""CWQUAN1 round trips on the CPU: every quantizer's codes rebuild θ + Q bit for bit, through the entropy coder.

    python islands/quant/test_qformat.py      (from benchmarks/covert_witness/; seconds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import prepare  # noqa: E402
import qformat as qf  # noqa: E402


def theta() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {"blocks.0.wq": torch.randn(96, 128, generator=g) * 0.02, "blocks.0.w_down": torch.randn(128, 192, generator=g) * 0.02,
            "blocks.0.attn_norm": torch.ones(128), "embed": torch.randn(50, 128, generator=g) * 0.02}


def test_round_trips() -> None:
    th = theta()
    g = torch.Generator().manual_seed(1)
    delta = {k: torch.randn(v.shape, generator=g) * 1e-3 for k, v in th.items()}
    delta["embed"][3:] = 0
    cases = {"rtn2": lambda x: qf.rtn(x, 2, 64), "rtn3": lambda x: qf.rtn(x, 3, 64), "rtn4": lambda x: qf.rtn(x, 4, 32),
             "rtn8": lambda x: qf.rtn(x, 8, 128), "sign": lambda x: qf.sign(x, 0), "tern": lambda x: qf.ternary(x, 1.0, 0)}
    for name, quant in cases.items():
        entries = {k: {"kind": "delta", "q": quant(d)} for k, d in delta.items()}
        blob = qf.encode(entries, th)
        got = qf.decode(th, blob)
        for k in th:
            want = (th[k].float() + entries[k]["q"].value()).to(th[k].dtype)
            assert prepare.same_bits(got[k], want), (name, k)
        err = sum(float(((got[k] - th[k] - delta[k]) ** 2).sum()) for k in th) / sum(float((d ** 2).sum()) for d in delta.values())
        n = sum(v.numel() for v in th.values())
        print(f"{name}: {len(blob)} bytes, {8 * len(blob) / n:.3f} bits/entry, relative error {err:.4f}")
    a, b = torch.randn(4, 128, generator=g), torch.randn(96, 4, generator=g) * 1e-3
    entries = {"blocks.0.wq": {"kind": "lora", "a": qf.rtn(a, 4, 64), "b": qf.rtn(b, 4, 4), "scale": 2.0},
               "blocks.0.attn_norm": {"kind": "full", "t": th["blocks.0.attn_norm"] * 1.01}}
    got = qf.decode(th, qf.encode(entries, th))
    want = prepare.lora_merge(th["blocks.0.wq"], entries["blocks.0.wq"]["a"].value(), entries["blocks.0.wq"]["b"].value(), 2.0)
    assert prepare.same_bits(got["blocks.0.wq"], want)
    assert prepare.same_bits(got["blocks.0.attn_norm"], th["blocks.0.attn_norm"] * 1.01)
    assert prepare.same_bits(got["embed"], th["embed"])
    zero = qf.rtn(torch.zeros(4, 64), 3, 64)
    assert float(zero.value().abs().sum()) == 0.0


def test_recompress() -> None:
    """Preset 9e holds the same payload in fewer bytes on sparse codes, and both decoders read it."""
    th = theta()
    g = torch.Generator().manual_seed(2)
    delta = {k: torch.randn(v.shape, generator=g) * (torch.rand(v.shape, generator=g) < 0.03) for k, v in th.items()}
    blob = qf.encode({k: {"kind": "delta", "q": qf.ternary(d, 0.5, 0)} for k, d in delta.items()}, th)
    small = qf.recompress(blob)
    assert len(small) < len(blob), (len(small), len(blob))
    got, want = qf.decode(th, small), qf.decode(th, blob)
    assert all(prepare.same_bits(got[k], want[k]) for k in th)
    got = prepare.decode(th, small, device="cpu")
    assert all(prepare.same_bits(got[k], want[k]) for k in th)


def test_refuses() -> None:
    th = theta()
    blob = qf.encode({"embed": {"kind": "delta", "q": qf.sign(torch.ones(50, 128))}}, th)
    for bad in (blob[:-3], b"CWQUAN2\n" + blob[8:]):
        try:
            qf.decode(th, bad)
        except (ValueError, Exception):
            continue
        raise AssertionError("decoded a damaged E")


if __name__ == "__main__":
    test_round_trips()
    test_recompress()
    test_refuses()
    print("ok")
