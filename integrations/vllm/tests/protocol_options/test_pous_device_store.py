"""`protocol_options.pous`'s GPU codec (``codec: gpu``) on the CPU: the device store with the scheme's reference codec standing in
for its GPU twin on CPU tensors.  It must give the reference store's storage root, leaves, blocks and tensors, drop the same
blocks for the negative control, and serve a model through the option's hooks; and the option refuses what it cannot run."""
from __future__ import annotations


import pytest

from verity_vllm import config
from verity_vllm import protocol_options as PO
from verity_vllm.protocol_options import pous

torch = pytest.importorskip("torch")

DENSE, BAND = "dense-chain/v1", "band-chain/d12/v1"
SALT = "ab" * 24


def _small(scheme: str):
    from verity_pous import band, dense
    return dense.DenseParams(8, 64 * 8) if scheme == DENSE else band.BandParams(16, 64 * 8, 12)


def _twin(scheme_id: str, params):
    """The reference as a stand-in twin: tags carry each row's segment id and the salt."""
    from verity_pous import schemes

    sch = schemes.get(scheme_id, params=params)

    def tags(salt, segs, device):
        return torch.tensor([[s, *salt] for s in segs], dtype=torch.int64)

    def rows(fn, x, t):
        out = []
        for g in range(x.shape[0]):
            s, salt = int(t[g, 0]), bytes(int(v) for v in t[g, 1:])
            out.append(fn(salt, s, x[g]))
        return torch.stack(out)

    def encode(w, t):
        return rows(lambda salt, s, seg: torch.frombuffer(bytearray(b"".join(sch.encode_segment(salt, s, seg.numpy().tobytes()))),
                                                          dtype=torch.uint8).view(seg.shape).clone(), w, t)

    def decode(c, t):
        return rows(lambda salt, s, seg: torch.frombuffer(bytearray(sch.decode_segment(salt, s, [bytes(r) for r in seg.numpy()])),
                                                          dtype=torch.uint8).view(seg.shape).clone(), c, t)
    return (encode, decode), tags


def _stores(scheme: str, own: dict, monkeypatch):
    monkeypatch.setattr(pous, "_base", lambda s: _small(scheme))
    weights = {"a": bytes(range(200)) * 3, "b": bytes(range(7, 250)) * 2, "c": bytes(range(90))}
    layout, sch, w, commitment = pous._commit(scheme, weights)
    twin, tags = _twin(scheme, sch.params)
    dev = pous._device_store(layout, sch, w, commitment, own, device="cpu", twin=twin, tags=tags, gate=lambda s, d: {"scheme": s})
    ref = pous._encode(layout, sch, w, commitment, None, bytes.fromhex(own["salt"]), float(own.get("drop_fraction", 0.0)),
                       own.get("drop_source"))
    return weights, dev, ref


@pytest.mark.parametrize("scheme", [DENSE, BAND])
def test_the_device_store_is_the_reference_store(scheme, monkeypatch):
    weights, dev, ref = _stores(scheme, {"salt": SALT}, monkeypatch)
    assert dev.salt == ref.salt and dev.salt_source.startswith("settings")
    assert dev.storage_root == ref.storage_root and dev.leaves == ref.leaves
    assert [dev.block(i) for i in range(dev.n_blocks)] == ref.blocks
    for name, raw in weights.items():
        assert dev.tensor(name, ((len(raw),), torch.uint8, "cpu")).numpy().tobytes() == raw == dev.decode(name)
    assert dev.stats["decodes"] == 2 * len(weights) and dev.stats["decoded_bytes"] > 0 and dev.stats["gate"] == {"scheme": scheme}


def test_the_negative_control_drops_the_same_blocks(monkeypatch):
    own = {"salt": SALT, "drop_fraction": 0.25, "drop_source": "cd" * 32}
    _, dev, ref = _stores(BAND, own, monkeypatch)
    assert dev.dropped == ref.dropped and len(dev.dropped) == round(0.25 * dev.n_blocks)
    assert all(dev.block(i) == bytes(len(ref.blocks[i])) for i in dev.dropped)
    assert dev.storage_root == ref.storage_root
    _, honest, _ = _stores(BAND, {"salt": SALT}, monkeypatch)
    assert honest.storage_root != dev.storage_root


class Linear:
    """A stand-in linear layer over a torch weight; its apply returns the weight it computes with."""

    def __init__(self, weight) -> None:
        self.weight = weight

        class Apply:
            def apply(self, layer, x, bias=None):
                return layer.weight.clone()
        self.quant_method = Apply()


class Model:
    def __init__(self) -> None:
        g = torch.Generator().manual_seed(3)
        a, b = (torch.randn(4, 10, generator=g).to(torch.bfloat16), torch.randn(3, 11, generator=g).to(torch.bfloat16))
        self.layers = {"q": Linear(a), "k": Linear(b), "head": Linear(a)}
        self.plain = {"q": a.clone(), "k": b.clone()}

    def named_modules(self):
        yield "", self
        yield from self.layers.items()

    def modules(self):
        return (m for _, m in self.named_modules())

    def named_parameters(self):
        return ((f"{n}.weight", m.weight) for n, m in self.layers.items())

    def named_buffers(self):
        return iter(())


def _ctx(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=config.parse_protocols(enabled), settings=settings or {}, **kw)


def test_codec_gpu_serves_through_the_hooks_and_answers_from_the_device_store(monkeypatch):
    from verity_pous import protocol

    model = Model()
    monkeypatch.setattr(pous, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(pous, "_base", lambda s: _small(DENSE))
    twin, tags = _twin(DENSE, None)
    real = pous._device_store

    def device_store(layout, sch, w, commitment, own, **kw):
        twin, tags = _twin(DENSE, sch.params)
        return real(layout, sch, w, commitment, own, device="cpu", twin=twin, tags=tags, gate=lambda s, d: {"scheme": s})
    monkeypatch.setattr(pous, "_device_store", device_store)
    comp = PO.install(model, _ctx(f"pous:{DENSE}", settings={"pous": {"codec": "gpu", "build_dir": "/unused", "salt": SALT}}))
    opt = comp.get("pous")
    assert all(layer.weight.numel() == 0 for layer in model.layers.values())                  # only C is held
    for name, layer in model.layers.items():
        want = model.plain["q" if name == "head" else name]
        assert torch.equal(layer.quant_method.apply(layer, None), want)                        # decoded on every use
    info = opt.info()
    assert info["codec"] == f"GPU twin of {DENSE}" and info["stats"]["decodes"] == 3 and info["responder"].startswith("in-process")
    c = opt.commitment()
    assert c["salt"] == SALT and len(opt.leaves()) == 32 * info["blocks"]
    s = opt.store
    vk = protocol.verifier_key(s.scheme, bytes.fromhex(c["commitment"]), s.salt, w=s.layout.pack(
        {n: t.contiguous().view(torch.uint8).numpy().tobytes() for n, t in model.plain.items()}))
    assert vk.root.hex() == c["storage_root"]
    v = protocol.Verifier(s.scheme, vk, protocol.challenge_key(b"audit", vk), k=8, research=True, clock=iter(range(0, 10**9, 1000)).__next__)
    assert v.run(opt.answer).accepted
    comp.close()
    assert all(torch.equal(model.layers[n].weight, model.plain["q" if n == "head" else n]) for n in model.layers)


def test_the_gpu_codec_refuses_what_it_cannot_run():
    got = lambda scheme, **own: pous.refusals(scheme, _ctx(f"pous:{scheme}", settings={"pous": own}))  # noqa: E731
    assert got(DENSE, codec="gpu", build_dir="/b") == ()
    assert any("no GPU twin of the scheme 'band-chain/d12/v1'" in r for r in got(BAND, codec="gpu", build_dir="/b"))
    assert any("needs build_dir" in r for r in got(DENSE, codec="gpu"))
    assert any("needs codec gpu" in r for r in got(DENSE, port=0))
    assert any("codec 'tpu'" in r for r in got(DENSE, codec="tpu"))
    assert any("salt" in r for r in got(DENSE, salt="00"))
    assert any("drop_fraction" in r for r in got(DENSE, drop_fraction=1.5))
