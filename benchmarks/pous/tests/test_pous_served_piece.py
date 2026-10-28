"""`served_piece.py`, PoUS's piece of the composed served run, end to end on the CPU over a tiny int8 checkpoint.

The checkpoint is a safetensors file of two decoder layers in the int8 pin's form (each linear int8 [N, K] beside its f32
[N, 1] scale, a bf16 embedding and norms) in an HF cache, held to a manifest the test writes; P2 runs at the small point's
1,024-bit blocks, 8 a segment (`pous._base`), so the adapter encodes it in milliseconds. The memory challenger's side is
`challenger_party.serve` in a thread, signed by the PoUS tests' test signer on a counter clock, and the prover answers it
over a socket pair, as `challenger_party.session` would across hosts. The answer is read back as the composed run reads it
(`served_verdicts.piece_row`, `block_record.leg`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import socket
import stat
import struct
import sys
from concurrent import futures
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.deploy.stream import Side
from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.ml.commitments import advice
from verity.ml.commitments.identity import canonical_json_bytes

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
REPO, REV = "verity-test/Tiny-INT8", "a1" * 20
NODE, AUDITOR = "gpu-node", "auditor-host"
COINS = bytes(range(64))
SMALL = P2Params(period.W, period.C0, 8)
TEST_PROTOCOL = mc.TEST_PREFIX + period.PROTOCOL
HIDDEN, FFN, LAYERS = 16, 32, 2
SHAPES = {"self_attn.q_proj": (HIDDEN, HIDDEN), "self_attn.k_proj": (8, HIDDEN), "self_attn.v_proj": (8, HIDDEN),
          "self_attn.o_proj": (HIDDEN, HIDDEN), "mlp.gate_proj": (FFN, HIDDEN), "mlp.up_proj": (FFN, HIDDEN),
          "mlp.down_proj": (HIDDEN, FFN)}
NAMES = {torch.int8: "I8", torch.float32: "F32", torch.bfloat16: "BF16"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SP = _load("pous_served_piece", ROOT / "benchmarks" / "pous" / "served_piece.py")
SV = _load("served_verdicts", ROOT / "benchmarks" / "one_stage" / "served_verdicts.py")
ts = _load("pous_test_signer", ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "pous_test_signer.py")
BR = _load("pous_served_block_record", ROOT / "demos" / "h200" / "interfaces" / "verity_e2e_demo" / "block_record.py")
pous = SP.pous


def _tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    out = {"model.embed_tokens.weight": torch.randn(64, HIDDEN, generator=g).to(torch.bfloat16),
           "model.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
        for name, (n, k) in SHAPES.items():
            out[p + name + ".weight"] = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g)
            out[p + name + ".weight_scale"] = torch.rand(n, 1, generator=g) + 0.5
    return out


def _safetensors(tensors: dict[str, torch.Tensor]) -> bytes:
    header, data = {}, bytearray()
    for k, t in tensors.items():
        b = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        header[k] = {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + bytes(data)


@dataclasses.dataclass
class Served:
    serve: Path
    hf: Path
    pins: Path
    out: Path
    private: Path
    tensors: dict[str, torch.Tensor]


@pytest.fixture
def served(tmp_path, monkeypatch) -> Served:
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL)
    tensors = _tensors()
    snap = tmp_path / "hf" / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / REV
    snap.mkdir(parents=True)
    blob = _safetensors(tensors)
    (snap / "model.safetensors").write_bytes(blob)
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"checkpoints": [{"repo": REPO, "revision": REV, "files": {
        "config.json": {"sha256": "00" * 32}, "model.safetensors": {"sha256": hashlib.sha256(blob).hexdigest()}}}]}))
    serve = tmp_path / "serve"
    serve.mkdir()
    (serve / "serve.json").write_text(json.dumps({"model": REPO, "revision": REV, "precision": "verity_int8"}))
    return Served(serve, tmp_path / "hf", pins, tmp_path / "pieces" / "pous", tmp_path / "private", tensors)


def _ticks(step: int = 1_000):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def _in_process_party(signer, log):
    """`challenger_party.session` with the challenger's side in a thread of this process (`challenger_party.serve`, the
    test signer, a counter clock) and the prover's over a socket pair (`challenger_party.answer`)."""
    calls = []

    def session(party, request, respond):
        a, b = socket.socketpair()

        def challenger():
            with b, b.makefile("rwb") as f:
                return CP.serve(f, request, signer=signer, log=log, coins=COINS, domain=period.SESSION_DOMAIN,
                                clock=_ticks())
        with futures.ThreadPoolExecutor(1) as pool:
            served_side = pool.submit(challenger)
            try:
                with a, a.makefile("rwb") as f:
                    rounds = CP.answer(f, request["block_bytes"], respond)
            finally:
                sid, signed = served_side.result()
        calls.append((party, request))
        return CP.report(sid, signed, signer.public_key) | {"host": party.host, "rounds": rounds - CP.WARMUPS}
    return session, calls


def _run(s: Served, tmp_path, monkeypatch, **kw):
    signer = ts.TestSigner(b"pous-served-piece", directory=tmp_path)
    session, calls = _in_process_party(signer, ts.session_log(tmp_path))
    monkeypatch.setattr(CP, "session", session)
    kw.setdefault("trusted_sha512", hashlib.sha512(signer.public_key).hexdigest())
    answer = SP.run(s.serve, 3, s.out, hf_home=s.hf, device="cpu", node=NODE,
                    party=Side(AUDITOR, ("ssh", AUDITOR, "challenger_party"), ("192.0.2.36", 47610)),
                    private=s.private, pins=s.pins, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, **kw)
    return answer, calls


def _public_text(s: Served) -> str:
    return "".join(p.read_text() for p in sorted(s.out.iterdir()))


def test_an_honest_store_is_accepted_in_the_answer_the_composed_run_reads(served, tmp_path, monkeypatch):
    """The served weights held as C*, the session answered from it, the verifier's tie and check: an accept, which the
    composed run's row and the block record read as PoUS's certificate (0 bits at the small point, which no certificate
    covers); the weights covered are the linears, the rest of the checkpoint and the serving state named outside."""
    types = pous.linear_types
    answer, calls = _run(served, tmp_path, monkeypatch)
    assert answer["verdict"] == "accept", answer["reason"]
    assert json.loads((served.out / "answer.json").read_text()) == answer and pous.linear_types is types
    row = SV.piece_row("space", served.out.parent)
    assert row["verdict"] == "accept" and row["owner"] == "memory-accounting" and row["protocol"] == TEST_PROTOCOL
    assert row["quantity"] == answer["public"] == {"certified": False, "eps": "0", "s_bits": 0}
    leg = BR.leg("pous", row, server=NODE, runs=[])
    assert leg["quantity"] == {"symbol": "S_cert", "value": 0, "unit": "bits", "certified": False}

    prof = answer["profile"]
    w_bytes = sum(n * k for n, k in SHAPES.values()) * LAYERS
    assert prof["weights"] | {"held": None} == {
        "model": REPO, "revision": REV, "covers": "linears", "linears": 4 * LAYERS, "bytes": w_bytes, "used_blocks": 44,
        "blocks": 48, "checkpoint_bytes": 7776, "share": str(Fraction(w_bytes, 7776)), "held": None}
    assert prof["outside"]["bytes"] == {"the embedding and the head": 2048, "the norms": 96, "the per-row scales": 1024}
    assert "KV cache" in prof["outside"]["serving"] and prof["window"] == 3
    assert prof["placement"][CP.PREMISE].startswith(f"holds: the memory challenger ran on {AUDITOR}")
    assert set(period.profile(period.point_for(1))) <= set(prof)                      # the run protocol's own profile
    party, request = calls[0]
    assert party.host == AUDITOR and request["node"] == NODE and request["blocks"] == 48

    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["key_pinned"] and doc["rounds_answered"] == period.K and doc["latency"]["late"] == 0


def test_the_answer_reveals_nothing_of_the_weights_and_its_commitment_opens_privately(served, tmp_path, monkeypatch):
    """The public files carry none of setup's published outputs (W's digest, the salt, C*'s root), the run's commitment or
    W's bytes; the answer's commitment is a hiding row commitment to the run's commitment, which the private record
    opens, in a directory only its owner reads."""
    answer, _ = _run(served, tmp_path, monkeypatch)
    run = json.loads((served.private / "run.json").read_text())
    opening = json.loads((served.private / "opening.json").read_text())
    published = run["published"]
    inner = hashlib.sha512(canonical_json_bytes(published)).hexdigest()
    assert run["answer"]["commitment"] == inner == opening["row"] and answer["commitment"] == opening["root"]
    text = _public_text(served)
    for secret in (published["commitment"], published["salt"], published["root"], inner,
                   served.tensors["model.layers.0.self_attn.q_proj.weight"].numpy().tobytes()[:24].hex()):
        assert secret not in text
    bind = SP.binding(REPO, REV, 3, None)
    assert opening["binding"] == bind.hex() == answer["profile"]["commitment"]["binding"]
    op = advice.Opening(opening["index"], bytes.fromhex(opening["row"]), bytes.fromhex(opening["salt"]),
                        tuple(bytes.fromhex(p) for p in opening["path"]))
    assert advice.verify(bytes.fromhex(answer["commitment"]), 1, 512, bind, op)
    assert not advice.verify(bytes.fromhex(answer["commitment"]), 1, 512, SP.binding(REPO, REV, 4, None), op)
    assert stat.S_IMODE(served.private.stat().st_mode) == 0o700
    assert (served.private / "digests.bin").stat().st_size == 48 * 64


def test_a_store_missing_blocks_is_rejected_with_its_reason(served, tmp_path, monkeypatch):
    """A prover that no longer holds most of C* answers the challenged blocks it dropped with zeros: the run's check
    refuses the session, and the answer is a reject with its reason, still under a hiding commitment."""
    held = pous.PousOption.answer
    dropped = set(range(8, 48))

    def lost(self, i):
        return bytes(self.store.params.block_bytes) if i in dropped else held(self, i)
    monkeypatch.setattr(pous.PousOption, "answer", lost)
    answer, calls = _run(served, tmp_path, monkeypatch)
    assert answer["verdict"] == "reject" and answer["reason"] and len(calls) == 1
    assert answer["public"] is None and answer["profile"] is None and len(bytes.fromhex(answer["commitment"])) == 64
    row = SV.piece_row("space", served.out.parent)
    assert row["verdict"] == "reject" and row["reason"] == answer["reason"]


def test_weights_that_are_not_the_pins_are_rejected_before_any_session(served, tmp_path, monkeypatch):
    doc = json.loads(served.pins.read_text())
    doc["checkpoints"][0]["files"]["model.safetensors"]["sha256"] = "11" * 32
    served.pins.write_text(json.dumps(doc))
    answer, calls = _run(served, tmp_path, monkeypatch)
    assert answer["verdict"] == "reject" and "not the pin's" in answer["reason"] and answer["commitment"] is None
    assert calls == [] and SV.check_answer(answer)


def test_a_weights_registration_of_another_model_is_rejected(served, tmp_path, monkeypatch):
    (served.serve / "weights_registration.json").write_text(json.dumps({"model": {"repo": "other/Model", "revision": REV}}))
    answer, calls = _run(served, tmp_path, monkeypatch)
    assert answer["verdict"] == "reject" and "the weights registration is of other/Model" in answer["reason"]
    assert calls == []


def test_a_session_signed_under_another_key_is_rejected(served, tmp_path, monkeypatch):
    answer, _ = _run(served, tmp_path, monkeypatch, trusted_sha512="00" * 64)
    assert answer["verdict"] == "reject" and answer["reason"]
    assert not json.loads((served.out / "pous.json").read_text())["key_pinned"]


def test_a_challenger_whose_side_does_not_start_leaves_the_period_undecided(served, tmp_path):
    """No retry and no exception: the answer says the session didn't complete, and why."""
    answer = SP.run(served.serve, 3, served.out, hf_home=served.hf, device="cpu", node=NODE, private=served.private,
                    party=Side(AUDITOR, (sys.executable, "-c", "import sys; sys.exit(3)"), ("127.0.0.1", 9)),
                    pins=served.pins, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    assert answer["verdict"] == "undecided" and answer["reason"].startswith(
        f"the memory challenger's session on {AUDITOR} did not complete")
    assert SV.piece_row("space", served.out.parent)["verdict"] == "undecided"


def test_a_serve_dir_without_serve_json_makes_no_answer(served, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert SP.main([str(empty), "0", str(served.out), "--hf-home", str(served.hf)]) == 2
    assert not (served.out / "answer.json").exists() and "no serve.json" in capsys.readouterr().err
