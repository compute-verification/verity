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
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
from concurrent import futures
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.deploy.stream import Side
from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous.schemes.p2 import POINT, P2Params
from verity.core.protocols.remote_bandwidth import p2_case, p2_lean
from verity.ml.commitments import advice
from verity.ml.commitments.identity import canonical_json_bytes

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
REPO, REV = "verity-test/Tiny-INT8", "a1" * 20
NODE, AUDITOR = "gpu-node", "auditor-host"
COINS = bytes(range(64))
SMALL = P2Params(period.W, period.C0, 8)
REGISTERED = P2Params(POINT[0], POINT[1], 4)
SOURCE = hashlib.sha512(b"the served piece's salt source").digest()
ENCODED: dict[tuple[bytes, int, bytes], tuple[bytes, dict]] = {}
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
fakes = _load("pous_fakes", ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "pous_fakes.py")
ts = _load("pous_test_signer", ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "pous_test_signer.py")
BR = _load("pous_served_block_record", ROOT / "examples" / "h200" / "interfaces" / "verity_e2e_demo" / "block_record.py")
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


def _in_process_party(signer, log, keep=None, record=lambda request: request):
    """`challenger_party.session` with the challenger's side in a thread of this process (`challenger_party.serve`, the
    test signer, a counter clock) and the prover's over a socket pair (`challenger_party.answer`); with `keep`, the
    challenger's device, where it keeps `record` of each completed session's request as `challenger_party.main` does."""
    calls = []

    def session(party, request, respond):
        CP.setup_of(request)
        a, b = socket.socketpair()

        def challenger():
            with b, b.makefile("rwb") as f:
                return CP.serve(f, request, signer=signer, log=log, coins=COINS, domain=period.SESSION_DOMAIN,
                                clock=fakes.Clock(step=1_000))
        with futures.ThreadPoolExecutor(1) as pool:
            served_side = pool.submit(challenger)
            try:
                with a, a.makefile("rwb") as f:
                    rounds = CP.answer(f, request["block_bytes"], respond)
            finally:
                sid, signed = served_side.result()
        if keep is not None:
            CP.keep(keep, sid, record(request))
        calls.append((party, request))
        return CP.report(sid, signed, signer.public_key) | {"host": party.host, "rounds": rounds - CP.WARMUPS}
    return session, calls


class Auditor:
    """The memory challenger's side in this process, as `challenger_party.main` serves it, on one device (a salt log and
    a session log) across a test's runs: a salt request answered by `challenger_party.give_salt` from ``draw``, a
    session refused before it listens as `designate` refuses it (the `RuntimeError` `Side.talk` raises), then served by
    `_in_process_party` with the device's salt log, which `_run` binds into `challenger_party.serve`, and kept in
    ``keep`` as ``record`` has it. With ``forget``, it serves the request without its audit, under an id of its coins;
    with ``claim`` too, it reports the audit's id."""

    def __init__(self, tmp_path: Path, forget: bool = False, claim: bool = False, keep: Path | None = None,
                 record=lambda request: request, draw=os.urandom) -> None:
        self.signer = ts.TestSigner(b"pous-served-piece", directory=tmp_path)
        device = Path(tempfile.mkdtemp(prefix="auditor-", dir=tmp_path))
        mc.SaltLog.create(device / "salts")
        self.salts, self.log = mc.SaltLog(device / "salts"), ts.session_log(tmp_path)
        self.forget, self.claim, self.draw = forget, claim, draw
        self.serve, self.calls = _in_process_party(self.signer, self.log, keep, record)
        self.sources = []

    def salt(self, party, audit, commitment):
        try:
            source = CP.give_salt(self.salts, {"audit": audit.hex(), "commitment": commitment.hex()}, self.draw)
        except CP.Refusal as e:
            raise RuntimeError(f"the memory challenger on {party.host} drew no salt: {e}") from None
        self.sources.append(source)
        return source

    def session(self, party, request, respond):
        try:
            CP.designate(request, self.salts, self.log)
        except CP.Refusal as e:
            raise RuntimeError(f"the side on {party.host} refused: {e}") from None
        if not self.forget:
            return self.serve(party, request, respond)
        audited = ("audit", "published", "params")
        report = self.serve(party, {k: v for k, v in request.items() if k not in audited}, respond)
        if not self.claim:
            return report
        pub = request["published"]
        sid = CP.session_id(bytes.fromhex(request["audit"]), bytes.fromhex(pub["commitment"]), bytes.fromhex(pub["salt"]))
        return report | {"session_id": sid.hex()}


def _run(s: Served, tmp_path, monkeypatch, auditor: Auditor | None = None, **kw):
    auditor = auditor or Auditor(tmp_path)
    monkeypatch.setattr(CP, "salt", auditor.salt)
    monkeypatch.setattr(CP, "session", auditor.session)
    monkeypatch.setattr(CP, "serve", functools.partial(CP.serve, salts=auditor.salts))
    kw.setdefault("trusted_sha512", hashlib.sha512(auditor.signer.public_key).hexdigest())
    answer = SP.run(s.serve, 3, s.out, hf_home=s.hf, device="cpu", node=NODE,
                    party=Side(AUDITOR, ("ssh", AUDITOR, "challenger_party"), ("192.0.2.36", 47610)),
                    private=s.private, pins=s.pins, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, **kw)
    return answer, auditor.calls


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
    assert request["k"] == SP.ROUNDS == 320 and prof["k"] == 320
    assert prof["placement"]["salt"].startswith(f"drawn by the memory challenger on {AUDITOR} from its OS")

    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["key_pinned"] and not {"latency", "rounds_answered", "warmups", "reported_host"} & set(doc)
    run = json.loads((served.private / "run.json").read_text())
    assert run["point"]["k"] == 320 and run["rounds_answered"] == 320 and run["latency"]["late"] == 0
    assert run["warmups"] == CP.WARMUPS and run["host"] == AUDITOR


def test_the_challenger_draws_the_salt_and_the_verdict_derives_the_session_id(served, tmp_path, monkeypatch):
    """Setup's salt is the one the memory challenger drew after the commitment and logged under the window's binding, and
    the session runs under the id derived from that binding, the commitment and the salt, which the verdict computes
    itself: the request names the binding and setup's published outputs, and the private record carries the id."""
    auditor = Auditor(tmp_path)
    answer, calls = _run(served, tmp_path, monkeypatch, auditor)
    assert answer["verdict"] == "accept", answer["reason"]
    run = json.loads((served.private / "run.json").read_text())
    published = run["published"]
    bind = SP.binding(REPO, REV, 3, None)
    commitment, salt = bytes.fromhex(published["commitment"]), bytes.fromhex(published["salt"])
    assert len(auditor.sources) == 1 and auditor.salts.get(bind) == (commitment, salt)
    assert salt == period.setup.draw_salt(auditor.sources[0], commitment)
    sid = CP.session_id(bind, commitment, salt)
    assert run["session_id"] == sid.hex() == run["session"]["commitment"]["session"] and sid in auditor.log
    request = calls[0][1]
    assert request["audit"] == bind.hex() and request["published"] == published
    assert sid.hex() not in _public_text(served)


def test_a_second_session_under_the_windows_binding_is_refused(served, tmp_path, monkeypatch):
    """A prover that runs the window again finds the challenger draws no second salt for its binding; one that reuses the
    first salt's source, so setup publishes the logged salt, finds the challenger runs no second session for it. Each
    is undecided with its reason, and no retry redraws."""
    auditor = Auditor(tmp_path)
    first, _ = _run(served, tmp_path, monkeypatch, auditor)
    assert first["verdict"] == "accept", first["reason"]
    again, calls = _run(served, tmp_path, monkeypatch, auditor)
    assert again["verdict"] == "undecided" and len(calls) == 1
    assert again["reason"].startswith(f"the memory challenger on {AUDITOR} drew no salt for the setup: ")
    assert "draws one salt an audit, and a retry never redraws it" in again["reason"]
    monkeypatch.setattr(auditor, "salt", lambda party, audit, commitment: auditor.sources[0])
    replay, calls = _run(served, tmp_path, monkeypatch, auditor)
    assert replay["verdict"] == "undecided" and len(calls) == 1
    assert replay["reason"].startswith(f"the memory challenger's session on {AUDITOR} did not complete: RuntimeError: "
                                       f"the side on {AUDITOR} refused: a session has run for audit")
    assert len(auditor.log.path.read_text().splitlines()) == 2


def test_a_session_whose_setup_salt_is_not_the_challengers_is_refused(served, tmp_path, monkeypatch):
    """A prover that takes the challenger's salt but salts setup from a source of its own publishes another salt: the
    challenger refuses the session, and the period is undecided."""
    auditor = Auditor(tmp_path)
    drawn = auditor.salt
    monkeypatch.setattr(auditor, "salt", lambda party, audit, commitment: (drawn(party, audit, commitment),
                                                                           b"the prover's own" * 4)[1])
    answer, calls = _run(served, tmp_path, monkeypatch, auditor)
    assert answer["verdict"] == "undecided" and calls == []
    assert answer["reason"].endswith(f"refused: setup's published outputs do not carry the commitment and salt the memory "
                                     f"challenger logged for audit {SP.binding(REPO, REV, 3, None).hex()}")
    assert auditor.log.path.read_text() == f"{mc.SESSION_LOG}\n"


def test_a_report_of_a_session_under_another_id_is_rejected(served, tmp_path, monkeypatch):
    """The verdict takes the session id from the window's binding and setup, never the report: a challenger that ran the
    session under an id of its coins is rejected for its report, and one that reports the window's id over that session
    is rejected by the run's check."""
    answer, calls = _run(served, tmp_path, monkeypatch, Auditor(tmp_path, forget=True))
    assert answer["verdict"] == "reject" and len(calls) == 1
    assert answer["reason"].startswith("the memory challenger's report is of a session under another id")
    lie, _ = _run(served, tmp_path, monkeypatch, Auditor(tmp_path, forget=True, claim=True))
    assert lie["verdict"] == "reject" and lie["reason"] == "the session does not run under the id the auditor designated"


def test_the_rounds_are_the_sessions_and_the_verdicts(served, tmp_path, monkeypatch):
    """`--rounds` sets the challenges the session asks for and the verdict checks: at 105 the run is checked there."""
    answer, calls = _run(served, tmp_path, monkeypatch, rounds=period.K)
    assert answer["verdict"] == "accept", answer["reason"]
    assert calls[0][1]["k"] == period.K and answer["profile"]["k"] == period.K
    assert json.loads((served.private / "run.json").read_text())["rounds_answered"] == period.K


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
    case = json.loads((served.private / "case.json").read_text())
    assert case["evidence"][:64] not in text and not (served.out / "case.json").exists()


def test_the_composites_piece_is_written_privately_with_the_runs_own_answer(served, tmp_path, monkeypatch):
    """The run's case (`period.case`) and the run protocol's answer on its bytes, which is the run's own, go beside the
    private record as the composite reads a piece: its commitment the digest of the case's, setup's published outputs;
    the public answer is that answer under its hiding commitment."""
    answer, _ = _run(served, tmp_path, monkeypatch)
    run = json.loads((served.private / "run.json").read_text())
    case = json.loads((served.private / "case.json").read_text())
    own = json.loads((served.private / "answer.json").read_text())
    assert own == run["answer"] and own == answer | {"commitment": own["commitment"], "profile": own["profile"]}
    assert sorted(case) == ["commitment", "evidence", "params"]
    assert json.loads(bytes.fromhex(case["commitment"])) == run["published"]
    assert own["commitment"] == hashlib.sha512(bytes.fromhex(case["commitment"])).hexdigest()
    assert SP.SC.pous_answer(case, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL) == own
    for f in ("case.json", "answer.json"):
        assert stat.S_IMODE((served.private / f).stat().st_mode) == 0o600


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
    assert json.loads((served.private / "answer.json").read_text())["verdict"] == "reject"


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
    """No retry and no exception: the answer says the challenger drew no salt for the setup, and why."""
    answer = SP.run(served.serve, 3, served.out, hf_home=served.hf, device="cpu", node=NODE, private=served.private,
                    party=Side(AUDITOR, (sys.executable, "-c", "import sys; sys.exit(3)"), ("127.0.0.1", 9)),
                    pins=served.pins, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    assert answer["verdict"] == "undecided" and answer["reason"].startswith(
        f"the memory challenger on {AUDITOR} drew no salt for the setup: ")
    assert SV.piece_row("space", served.out.parent)["verdict"] == "undecided"
    assert not (served.private / "case.json").exists()


def test_a_serve_dir_without_serve_json_makes_no_answer(served, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert SP.main([str(empty), "0", str(served.out), "--hf-home", str(served.hf)]) == 2
    assert not (served.out / "answer.json").exists() and "no serve.json" in capsys.readouterr().err


@pytest.fixture(scope="module")
def case_exe() -> Path:
    """`remote-audit-p2-protocol`, built in the remote audit's package under the difftest's build lock."""
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    p2_lean.BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(p2_lean.BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", p2_case.EXE], cwd=p2_lean.PACKAGE, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p2_lean.PACKAGE / ".lake" / "build" / "bin" / p2_case.EXE


@pytest.fixture
def registered(served, monkeypatch) -> Served:
    """`served` at PoUS's registered point, the one a P2 registration names (2,056-byte blocks, 4 a segment: the two
    layers' 8 linears in 2 segments, 2 units), with each segment's encode kept across the module's tests, so ``C*``,
    whose salt `_bandwidth`'s challenger draws from a fixed source, is encoded once."""
    monkeypatch.setattr(pous, "_base", lambda scheme: REGISTERED)
    S = pous._p2("store")
    encode = S.CpuEncoder.encode

    def kept(self, s, w_seg):
        key = (self.salt, s, hashlib.sha512(w_seg).digest())
        if key not in ENCODED:
            ENCODED[key] = encode(self, s, w_seg)
        return ENCODED[key]
    monkeypatch.setattr(S.CpuEncoder, "encode", kept)
    return served


def _bandwidth(tmp_path, record=lambda request: request, **kw):
    """`served_piece.run`'s bandwidth session: the party in the gate's shape (`p2_party --listen` from the tree with no
    site packages), its registration the challenger's record in ``parties/pous``, which the in-process challenger keeps
    as ``record`` has it, and a short session, which a toy party's flags hold it to."""
    party = (f"cd {ROOT} && exec {sys.executable} -S -E -m verity.core.protocols.remote_bandwidth.p2_party "
             f"--device {tmp_path / 'parties' / 'bw'} --listen 127.0.0.1:0 --pous-record {tmp_path / 'parties' / 'pous'} "
             f"--windows 2 --passes 2 --samples 4 --window-ns 1000000 --deadline-ns 1000000")
    auditor = Auditor(tmp_path, keep=tmp_path / "parties" / "pous", record=record, draw=lambda n: SOURCE[:n])
    return {"auditor": auditor, "bw_party": party, "bw_host": "127.0.0.1",
            "bw_request": SP.HB.request(NODE.encode(), windows=2, passes=2, samples=4, window_s=0.001,
                                        deadline_s=0.001)} | kw


def test_the_held_sessions_request_is_the_auditors_session():
    """The held session requests, by default, the session the auditor's party holds every request to
    (`p2_party.SESSION`), so V3's served run is never refused for its parameters."""
    assert SP.HB.request(NODE.encode()) == SP.HB.p2_party.request_of(NODE.encode(), **SP.HB.p2_party.SESSION)
    assert (SP.HB.WINDOWS, SP.HB.PASSES, SP.HB.SAMPLES, SP.HB.WINDOW_S, SP.HB.DEADLINE_S) == (6, 4, 1024, 20.0, 1.0)


def _jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.mark.slow
def test_the_bandwidth_session_is_over_the_store_pous_held_its_case_private_and_the_leans(registered, tmp_path,
                                                                                         monkeypatch, case_exe):
    """PoUS's session and then PoUB-P2's on the one ``C*`` the piece encoded and holds, the bandwidth party's
    registration the challenger's own record of the PoUS session: both accepted. The P2 case, written privately, is of
    PoUS's store (`p2_case.same_store` against the run's case), and answered as the session did (`p2_case.answer`, the
    piece judging without `--bw-case-exe`); `remote-audit-p2-protocol` refuses it, a toy party's session, naming each
    value off the auditor's (`p2_party.SESSION`). The public answer, which the composed run reads as the bandwidth
    piece's, names none of the store. It encodes ``C*`` at the registered point (about 25 s on the CPU), the only point
    a P2 registration names."""
    s = registered
    answer, calls = _run(s, tmp_path, monkeypatch, **_bandwidth(tmp_path))
    assert answer["verdict"] == "accept", answer["reason"]
    public = s.out.parent / "bandwidth"
    bw = json.loads((public / "answer.json").read_text())
    assert bw["verdict"] == "accept", bw["reason"]
    published = json.loads((s.private / "run.json").read_text())["published"]
    request = calls[0][1]
    setup = CP.setup_of(request)
    assert setup["root"] == request["binding"] == published["root"]
    assert setup["salt"] == published["salt"] and setup["commitment"] == published["commitment"]
    assert setup["params"] == {k: getattr(REGISTERED.with_segments(2), k) for k in CP.PARAMS}

    private = s.serve / "private" / "bandwidth"
    case = json.loads((private / p2_case.CASE).read_text())
    own = json.loads((private / p2_case.ANSWER).read_text())
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    for f in (p2_case.CASE, p2_case.ANSWER, "session.json", "opening.json"):
        assert stat.S_IMODE((private / f).stat().st_mode) == 0o600
    pous_case = json.loads((s.private / "case.json").read_text())
    assert p2_case.same_store(case, pous_case) is None and p2_case.store_root(case).hex() == published["root"]
    toy = SP.HB.p2_party.request_of(NODE.encode(), windows=2, passes=2, samples=4, window_ns=10 ** 6,
                                    deadline_ns=10 ** 6)
    assert case["params"]["request"] == toy and own["verdict"] == "accept"
    lean, = p2_case.serve(case_exe, [case])
    why = "; ".join(f"the request's {k} is not the auditor's ({toy[k]}, not {v})"
                    for k, v in SP.HB.p2_party.SESSION.items())
    assert lean == own | {"profile": None, "public": None, "reason": f"the request is refused: {why}", "verdict": "reject"}
    assert own["commitment"] == p2_case.digest(case).hex() and own["profile"]["units"] == 2
    assert _jsonl(tmp_path / "parties" / "bw" / "private" / "cases.jsonl") == [case]
    assert _jsonl(tmp_path / "parties" / "bw" / "answers.jsonl")[-1]["verdict"] == "accept"

    assert {k: v for k, v in bw["public"].items() if k not in ("gamma", "r_bytes_per_s")} == own["public"]
    assert "registration" not in bw["profile"] and bw["profile"]["units"] == 2
    opening = json.loads((private / "opening.json").read_text())
    assert opening["row"] == own["commitment"] and opening["root"] == bw["commitment"]
    op = advice.Opening(opening["index"], bytes.fromhex(opening["row"]), bytes.fromhex(opening["salt"]),
                        tuple(bytes.fromhex(p) for p in opening["path"]))
    assert advice.verify(bytes.fromhex(bw["commitment"]), 1, 512, SP.HB.binding(REPO, REV, 3), op)
    row = SV.piece_row("bandwidth", s.out.parent)
    assert row["verdict"] == "accept" and row["protocol"] == p2_case.PROTOCOL
    assert BR._quantity("bandwidth", row)[0] == {"symbol": "Q_cert", "value": bw["public"]["q_bytes"], "unit": "bytes"}
    assert bw["profile"]["gamma_terms"] and json.loads((s.out / "pous.json").read_text())["bandwidth"]["verdict"] == "accept"

    text = "".join(p.read_text() for d in (public, s.out) for p in sorted(d.iterdir()))
    digests = (s.private / "digests.bin").read_bytes()
    for secret in (published["root"], published["salt"], published["commitment"], own["commitment"],
                   case["commitment"][:64], digests[:64].hex(), digests[-64:].hex()):
        assert secret not in text


@pytest.mark.slow
def test_a_challenger_record_of_another_store_is_refused_and_no_case_is_made(registered, tmp_path, monkeypatch):
    """The challenger's record names a store other than the one the piece holds: the prover refuses its registration
    before any pass, the bandwidth answer is undecided naming that, and no case is written, while PoUS's stands."""
    other = hashlib.sha512(b"another store").hexdigest()

    def elsewhere(request):
        return request | {"binding": other, "published": request["published"] | {"root": other}}
    answer, _ = _run(registered, tmp_path, monkeypatch, **_bandwidth(tmp_path, record=elsewhere))
    assert answer["verdict"] == "accept", answer["reason"]
    bw = json.loads((registered.out.parent / "bandwidth" / "answer.json").read_text())
    assert bw["verdict"] == "undecided" and "the verifier's registration is not the held store's" in bw["reason"]
    assert bw["public"] is None and bw["commitment"] is None and SV.check_answer(bw)
    private = registered.serve / "private" / "bandwidth"
    assert not (private / p2_case.CASE).exists() and not (private / p2_case.ANSWER).exists()
    assert SV.piece_row("bandwidth", registered.out.parent)["verdict"] == "undecided"


def test_no_pous_session_leaves_the_bandwidth_piece_undecided_naming_why(served, tmp_path):
    """The challenger's side doesn't start: no store was bound, so no bandwidth session runs, and its answer says so."""
    SP.run(served.serve, 3, served.out, hf_home=served.hf, device="cpu", node=NODE, private=served.private,
           party=Side(AUDITOR, (sys.executable, "-c", "import sys; sys.exit(3)"), ("127.0.0.1", 9)),
           pins=served.pins, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, bw_party="false")
    bw = json.loads((served.out.parent / "bandwidth" / "answer.json").read_text())
    assert bw["verdict"] == "undecided" and bw["reason"].startswith("no PoUS session bound a store: the memory challenger")
    assert SV.piece_row("bandwidth", served.out.parent)["verdict"] == "undecided"
    assert not (served.serve / "private" / "bandwidth").exists()
