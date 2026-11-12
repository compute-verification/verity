"""`served_piece.py`, PoUS's piece of the composed served run, end to end on the CPU over a tiny int8 checkpoint.

The checkpoint is a safetensors file of two decoder layers in the int8 pin's form (each linear int8 [N, K] beside its f32
[N, 1] scale, a bf16 embedding and norms) in an HF cache, held to a manifest the test writes; `served_bf16` is the same
model in the BF16 pin's form, each linear bf16 [N, K] with no scale. P2 runs at the small point's
1,024-bit blocks, 8 a segment (`pous._base`), so the adapter encodes it in milliseconds. The memory challenger's side is
`challenger_party.serve` in a thread, signed by the PoUS tests' test signer on a counter clock, and the prover answers it
over a socket pair, as `challenger_session.session` would across hosts. The answer is read back as the composed run reads it
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
import re
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

from verity.configuration.deploy.stream import Side
from verity.protocols.pous.device import challenger_party as CP
from verity.protocols.pous.device import challenger_session as CS
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import period
from verity.protocols.pous.verifier.schemes.p2 import POINT, P2Params
from verity.protocols.poub.verifier import p2_case, p2_lean
from verity.commitments import advice
from verity.commitments.identity import canonical_json_bytes

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
REPO, REV = "verity-test/Tiny-INT8", "a1" * 20
PLAIN_REPO, PLAIN_REV = "verity-test/Tiny-BF16", "b2" * 20
NODE, AUDITOR = "gpu-node", "auditor-host"
COINS = bytes(range(64))
SMALL = P2Params(period.W, period.C0, 8)
REGISTERED = P2Params(POINT[0], POINT[1], 4)
SOURCE = hashlib.sha512(b"the served piece's salt source").digest()
ENCODED: dict[tuple[bytes, int, bytes], tuple[bytes, dict]] = {}
TEST_PROTOCOL = mc.TEST_PREFIX + period.PROTOCOL
RUN_EXE = "pous-run-protocol"
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
fakes = _load("pous_fakes", ROOT / "verity" / "protocols" / "pous" / "verifier" / "tests" / "pous_fakes.py")
ts = _load("pous_test_signer", ROOT / "verity" / "protocols" / "pous" / "verifier" / "tests" / "pous_test_signer.py")
BR = _load("pous_served_block_record", ROOT / "benchmarks" / "one_stage" / "interfaces" / "verity_e2e_demo" / "block_record.py")
pous = SP.pous


def _tensors(plain: bool = False) -> dict[str, torch.Tensor]:
    """The tiny checkpoint's tensors: each linear int8 beside its scale, or with ``plain`` bf16 and no scale."""
    g = torch.Generator().manual_seed(11)
    out = {"model.embed_tokens.weight": torch.randn(64, HIDDEN, generator=g).to(torch.bfloat16),
           "model.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
        for name, (n, k) in SHAPES.items():
            if plain:
                out[p + name + ".weight"] = torch.randn(n, k, generator=g).to(torch.bfloat16)
                continue
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


def _snapshot(tmp_path: Path, repo: str, rev: str, tensors: dict[str, torch.Tensor]) -> dict:
    """``tensors`` as ``repo``'s snapshot at ``rev`` in the HF cache under ``tmp_path``; its manifest entry."""
    snap = tmp_path / "hf" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    snap.mkdir(parents=True)
    blob = _safetensors(tensors)
    (snap / "model.safetensors").write_bytes(blob)
    return {"repo": repo, "revision": rev, "files": {
        "config.json": {"sha256": "00" * 32}, "model.safetensors": {"sha256": hashlib.sha256(blob).hexdigest()}}}


def _served(tmp_path: Path, monkeypatch, repo: str, rev: str, precision: str, plain: bool) -> Served:
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL)
    tensors = _tensors(plain)
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"checkpoints": [_snapshot(tmp_path, repo, rev, tensors)]}))
    serve = tmp_path / "serve"
    serve.mkdir()
    (serve / "serve.json").write_text(json.dumps({"model": repo, "revision": rev, "precision": precision}))
    return Served(serve, tmp_path / "hf", pins, tmp_path / "pieces" / "pous", tmp_path / "private", tensors)


@pytest.fixture
def served(tmp_path, monkeypatch) -> Served:
    return _served(tmp_path, monkeypatch, REPO, REV, "verity_int8", plain=False)


@pytest.fixture
def served_bf16(tmp_path, monkeypatch) -> Served:
    return _served(tmp_path, monkeypatch, PLAIN_REPO, PLAIN_REV, "pin", plain=True)


def _in_process_party(signer, log, keep=None, record=lambda request: request):
    """`challenger_session.session` with the challenger's side in a thread of this process (`challenger_party.serve`, the
    test signer, a counter clock) and the prover's over a socket pair (`challenger_session.answer`); with `keep`, the
    challenger's device, where it keeps `record` of each completed session's request as `challenger_party.main` does."""
    calls = []

    def session(party, request, respond):
        CS.setup_of(request)
        a, b = socket.socketpair()

        def challenger():
            with b, b.makefile("rwb") as f:
                return CP.serve(f, request, signer=signer, log=log, coins=COINS, domain=period.SESSION_DOMAIN,
                                clock=fakes.Clock(step=1_000))
        with futures.ThreadPoolExecutor(1) as pool:
            served_side = pool.submit(challenger)
            try:
                with a, a.makefile("rwb") as f:
                    rounds = CS.answer(f, request["block_bytes"], respond)
            finally:
                sid, signed = served_side.result()
        if keep is not None:
            CP.keep(keep, sid, record(request))
        calls.append((party, request))
        return CP.report(sid, signed, signer.public_key) | {"host": party.host, "rounds": rounds - CS.WARMUPS}
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
        sid = CS.session_id(bytes.fromhex(request["audit"]), bytes.fromhex(pub["commitment"]), bytes.fromhex(pub["salt"]))
        return report | {"session_id": sid.hex()}


def _run(s: Served, tmp_path, monkeypatch, auditor: Auditor | None = None, host: str = AUDITOR, **kw):
    auditor = auditor or Auditor(tmp_path)
    monkeypatch.setattr(CS, "salt", auditor.salt)
    monkeypatch.setattr(CS, "session", auditor.session)
    monkeypatch.setattr(CP, "serve", functools.partial(CP.serve, salts=auditor.salts))
    kw.setdefault("trusted_sha512", hashlib.sha512(auditor.signer.public_key).hexdigest())
    answer = SP.run(s.serve, 3, s.out, hf_home=s.hf, device="cpu", node=NODE,
                    party=Side(host, ("ssh", host, "challenger_party"), ("192.0.2.36", 47610)),
                    private=s.private, pins=[s.pins], schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, **kw)
    return answer, auditor.calls


def _gone(*args, **kw):
    raise AssertionError("the answers reach neither the challenger nor the store")


def _parts(s: Served, tmp_path, monkeypatch, auditor: Auditor | None = None, between=lambda: None, **kw):
    """`_run` in the two parts `served_piece.sh` runs: `sessions`, which leaves no answer, then ``between``, then
    `answers` with the challenger's sides and the adapter's setup out of reach."""
    auditor = auditor or Auditor(tmp_path)
    monkeypatch.setattr(CS, "salt", auditor.salt)
    monkeypatch.setattr(CS, "session", auditor.session)
    monkeypatch.setattr(CP, "serve", functools.partial(CP.serve, salts=auditor.salts))
    trusted = kw.pop("trusted_sha512", hashlib.sha512(auditor.signer.public_key).hexdigest())
    card = {k: kw.pop(k) for k in ("rounds", "bw_party", "bw_host", "bw_request") if k in kw}
    early = SP.sessions(s.serve, 3, s.out, hf_home=s.hf, device="cpu", node=NODE,
                        party=Side(AUDITOR, ("ssh", AUDITOR, "challenger_party"), ("192.0.2.36", 47610)),
                        private=s.private, pins=[s.pins], protocol=TEST_PROTOCOL, **card)
    assert early is None
    between()
    with monkeypatch.context() as m:
        for mod, name in ((CS, "salt"), (CS, "session"), (SP.PO, "install"), (SP.HB, "held")):
            m.setattr(mod, name, _gone)
        return SP.answers(s.serve, 3, s.out, trusted_sha512=trusted, private=s.private, schemes=ts.SCHEMES,
                          protocol=TEST_PROTOCOL, **kw)


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _stamps(out: Path) -> list[str]:
    """The events of the piece's timeline, ``out``'s ``stamps.txt``, in order: each line a UTC second, then the event."""
    lines = (out / "stamps.txt").read_text().splitlines()
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", line) for line in lines), lines
    return [line.split(" ", 1)[1] for line in lines]


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
    assert leg["quantity"] == {"symbol": "S0", "value": 0, "unit": "bits", "certified": False}

    prof = answer["profile"]
    w_bytes = sum(n * k for n, k in SHAPES.values()) * LAYERS
    assert prof["weights"] | {"held": None} == {
        "model": REPO, "revision": REV, "covers": "linears", "linears": 4 * LAYERS, "bytes": w_bytes, "used_blocks": 44,
        "blocks": 48, "checkpoint_bytes": 7776, "share": str(Fraction(w_bytes, 7776)), "held": None}
    assert prof["outside"]["bytes"] == {"the embedding and the head": 2048, "the norms": 96, "the per-row scales": 1024}
    assert "KV cache" in prof["outside"]["serving"] and prof["window"] == 3
    assert prof["placement"][CS.PREMISE].startswith("rehearsal: key not the auditor's pinned key, no bound; ")
    assert set(period.profile(period.point_for(1))) <= set(prof)                      # the run protocol's own profile
    party, request = calls[0]
    assert party.host == AUDITOR and request["node"] == NODE and request["blocks"] == 48
    assert request["k"] == SP.ROUNDS == 320 and prof["k"] == 320
    assert prof["placement"]["salt"].startswith(f"drawn by the memory challenger on {AUDITOR} from its OS")

    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["key_pinned"] and doc["key_pin"] == "run"
    assert not {"latency", "rounds_answered", "warmups", "reported_host"} & set(doc)
    assert doc["decided_by"] == {"program": "python", "host": socket.gethostname()} and "lean_s" not in doc
    run = json.loads((served.private / "run.json").read_text())
    assert doc["key_sha512"] == hashlib.sha512(bytes.fromhex(run["public_key"])).hexdigest()
    assert run["point"]["k"] == 320 and run["rounds_answered"] == 320 and run["latency"]["late"] == 0
    assert run["warmups"] == CS.WARMUPS and run["host"] == AUDITOR


@pytest.mark.parametrize("host", ["on-node-rehearsal", "vy-auditor-2"])
def test_a_run_that_pins_its_own_key_is_accepted_and_its_record_is_a_rehearsals(served, tmp_path, monkeypatch, host):
    """A run that passes its own challenger key's SHA-512 as the pin has its session accepted under it (`key_pinned`),
    and its record says the pin was the run's (`key_pin`) and that `trusted-host` is a rehearsal's, with no bound:
    the tree pins another key for vy-auditor-2 and none for any other name, so naming the auditor's host changes
    nothing."""
    answer, _ = _run(served, tmp_path, monkeypatch, host=host)
    assert answer["verdict"] == "accept", answer["reason"]
    where = answer["profile"]["placement"]
    assert where["kind"] == "own-host" and where[CS.PREMISE] == (
        "rehearsal: key not the auditor's pinned key, no bound; "
        f"the names say the memory challenger ran on {host}, not the prover's {NODE}")
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["key_pinned"] and doc["key_pin"] == "run" and doc["key_sha512"] != CS.AUDITOR_KEYS["vy-auditor-2"]
    assert SP.line(answer).endswith(f"; {CS.PREMISE} {where[CS.PREMISE]}")


def test_a_plain_bf16_checkpoint_is_held_as_its_linears_at_twice_the_bytes(served_bf16, tmp_path, monkeypatch):
    """The BF16 pin's form: each linear bf16 [N, K] with no scale beside it, so ``W`` is the same linears at two bytes an
    element, nothing outside it is a scale, and the session, the tie and the check run as over the int8 store."""
    s = served_bf16
    answer, calls = _run(s, tmp_path, monkeypatch)
    assert answer["verdict"] == "accept", answer["reason"]
    assert SV.piece_row("space", s.out.parent)["verdict"] == "accept"
    prof = answer["profile"]
    w_bytes = 2 * sum(n * k for n, k in SHAPES.values()) * LAYERS
    assert prof["weights"] | {"held": None} == {
        "model": PLAIN_REPO, "revision": PLAIN_REV, "covers": "linears", "linears": 4 * LAYERS, "bytes": w_bytes,
        "used_blocks": 80, "blocks": 80, "checkpoint_bytes": 11360, "share": str(Fraction(w_bytes, 11360)), "held": None}
    assert prof["outside"]["bytes"] == {"the embedding and the head": 2048, "the norms": 96}
    assert calls[0][1]["blocks"] == 80
    case = json.loads((s.private / "case.json").read_text())
    assert json.loads((s.private / "answer.json").read_text())["verdict"] == "accept"
    setup = json.loads(bytes.fromhex(case["evidence"]))["setup"]
    assert len(bytes.fromhex(setup["evidence"])) == 4 + 4 * 10 + 80 * (4 + SMALL.block_bytes)


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
    sid = CS.session_id(bind, commitment, salt)
    assert run["session_id"] == sid.hex() == run["session"]["commitment"]["session"] and sid in auditor.log
    request = calls[0][1]
    assert request["audit"] == bind.hex() and request["published"] == published
    assert sid.hex() not in _public_text(served)


def test_a_second_session_under_the_windows_binding_is_refused(served, tmp_path, monkeypatch):
    """A prover that runs the window again finds the challenger samples no second salt for its binding; one that reuses the
    first salt's source, so setup publishes the logged salt, finds the challenger runs no second session for it. Each
    is undecided with its reason, and no retry resamples."""
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


def test_the_sessions_are_kept_and_then_answered_with_no_card(served, tmp_path, monkeypatch):
    """The piece in its two parts. `sessions` answers nothing: it leaves the run's record so far, the table, ``C*`` as
    setup sent it and the summary so far, in the private directory alone. `answers` reads them back with no store, no
    challenger and no setup, and gives the answer one process gives, the case's line counted; ``C*`` and the summary
    so far are then gone, so nothing is answered twice."""
    s = served

    def kept():
        assert _names(s.private) == sorted(["digests.bin", "run.json", SP.HELD, SP.STORE])
        assert _names(s.out) == ["stamps.txt"]
        assert (s.private / SP.STORE).stat().st_size == 48 * SMALL.block_bytes
        assert stat.S_IMODE((s.private / SP.STORE).stat().st_mode) == 0o600
        assert not {"answer", "trusted"} & set(json.loads((s.private / "run.json").read_text()))
    answer = _parts(s, tmp_path, monkeypatch, between=kept)
    assert answer["verdict"] == "accept", answer["reason"]
    assert json.loads((s.out / "answer.json").read_text()) == answer
    assert SV.piece_row("space", s.out.parent)["verdict"] == "accept"
    assert _names(s.private) == ["answer.json", "case.json", "digests.bin", "opening.json", "run.json"]
    run = json.loads((s.private / "run.json").read_text())
    case = json.loads((s.private / "case.json").read_text())
    assert run["answer"] == json.loads((s.private / "answer.json").read_text()) and len(run["trusted"]) == 1
    assert run["answer"] == SP.SC.pous_answer(case, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    doc = json.loads((s.out / "pous.json").read_text())
    assert doc["case_bytes"] == (s.private / "case.json").stat().st_size == len(canonical_json_bytes(case)) + 1
    assert doc["key_pinned"] and doc["session_s"] >= 0 and doc["answer_s"] >= 0
    with pytest.raises(SP.Refusal, match="holds no sessions to answer"):
        SP.answers(s.serve, 3, s.out, private=s.private, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)


def test_the_pin_is_the_answers_own_never_one_the_sessions_kept(served, tmp_path, monkeypatch):
    """The trusted key is the one `answers` is given: under another pin than the session's key the kept session is
    rejected, and the record says no key was trusted."""
    answer = _parts(served, tmp_path, monkeypatch, trusted_sha512="00" * 64)
    assert answer["verdict"] == "reject" and answer["reason"]
    assert not json.loads((served.out / "pous.json").read_text())["key_pinned"]
    assert json.loads((served.private / "run.json").read_text())["trusted"] == []


def test_a_kept_store_that_is_not_the_points_blocks_is_no_answer(served, tmp_path, monkeypatch):
    """A kept ``C*`` cut short is refused before any case is built, and nothing is answered."""
    def cut():
        (served.private / SP.STORE).write_bytes((served.private / SP.STORE).read_bytes()[:-1])
    with pytest.raises(SP.Refusal, match=f"not C\\*'s 48 blocks of {SMALL.block_bytes}"):
        _parts(served, tmp_path, monkeypatch, between=cut)
    assert not (served.out / "answer.json").exists()


def test_kept_sessions_under_another_id_than_the_windows_are_not_answered(served, tmp_path, monkeypatch):
    """The answers derive the session's id again, from the window's binding and what setup published, and take neither
    from what was kept: the same sessions asked for as another window's, and then a kept record naming another id, are
    refused before anything is answered, and what was kept stays."""
    s = served

    def other():
        with pytest.raises(SP.Refusal, match="not this window's"):
            SP.answers(s.serve, 4, s.out, private=s.private, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
        run = json.loads((s.private / "run.json").read_text())
        (s.private / "run.json").write_text(json.dumps(run | {"session_id": "00" * 32}))
    with pytest.raises(SP.Refusal, match="under another binding or id than the ones window 3 of"):
        _parts(s, tmp_path, monkeypatch, between=other)
    assert _names(s.out) == ["stamps.txt"]
    assert _names(s.private) == sorted(["digests.bin", "run.json", SP.HELD, SP.STORE])


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


def test_the_pin_is_the_first_manifests_that_names_the_model(tmp_path):
    """The served model's pin is read from the manifests in order, so one that only another manifest pins is found, and
    a model none pins leaves no answer, naming each manifest read."""
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    entry = lambda repo, sha: {"repo": repo, "revision": REV, "files": {  # noqa: E731
        "config.json": {"size": 1}, "model.safetensors": {"sha256": sha}}}
    first.write_text(json.dumps({"checkpoints": [entry(REPO, "aa" * 32)]}))
    second.write_text(json.dumps({"checkpoints": [entry(REPO, "bb" * 32), entry(PLAIN_REPO, "cc" * 32)]}))
    assert SP._pin(REPO, REV, [first, second]) == {"model.safetensors": "aa" * 32}
    assert SP._pin(PLAIN_REPO, REV, [first, second]) == {"model.safetensors": "cc" * 32}
    with pytest.raises(SP.Refusal, match=f"first.json, second.json pin no {PLAIN_REPO}@{PLAIN_REV}"):
        SP._pin(PLAIN_REPO, PLAIN_REV, [first, second])
    assert [p.name for p in SP.PINS] == ["checkpoints-int8.json", "checkpoints.json"]


@pytest.mark.parametrize("plain, change, form", [
    (False, lambda t, p: t.pop(p + "self_attn.k_proj.weight_scale"), "self_attn.qkv_proj: I8 parts with 2 of 3 scales"),
    (False, lambda t, p: t.update({p + "mlp.down_proj.weight": t[p + "mlp.down_proj.weight"].to(torch.float32)}),
     "mlp.down_proj: F32 parts with 1 of 1 scales"),
    (False, lambda t, p: t.update({p + "mlp.up_proj.weight": t[p + "mlp.up_proj.weight"].to(torch.bfloat16)}),
     "mlp.gate_up_proj: BF16/I8 parts with 2 of 2 scales"),
    (True, lambda t, p: t.update({p + "self_attn.o_proj.weight_scale": torch.ones(HIDDEN, 1)}),
     "self_attn.o_proj: BF16 parts with 1 of 1 scales"),
])
def test_a_linear_that_is_neither_int8_with_its_scales_nor_plain_bf16_is_refused_by_name(tmp_path, plain, change, form):
    """A linear in a form no pin has: one of its parts' scales missing, another dtype, parts of two dtypes, or a bf16
    one with a scale beside it. The piece holds none of it and names the linear and what it found."""
    tensors = _tensors(plain)
    change(tensors, "model.layers.1.")
    entry = _snapshot(tmp_path, REPO, REV, tensors)
    snap = tmp_path / "hf" / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / REV
    ck, wrong = SP.checkpoint(snap, {"model.safetensors": entry["files"]["model.safetensors"]["sha256"]})
    assert wrong == []
    with pytest.raises(ValueError, match="neither an int8 linear with its scales nor a plain BF16 one") as e:
        SP.linears(ck, torch.device("cpu"))
    assert str(e.value).startswith("model.layers.1." + form)


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
                    pins=[served.pins], schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    assert answer["verdict"] == "undecided" and answer["reason"].startswith(
        f"the memory challenger on {AUDITOR} drew no salt for the setup: ")
    assert SV.piece_row("space", served.out.parent)["verdict"] == "undecided"
    assert not (served.private / "case.json").exists()


def test_a_serve_dir_without_serve_json_makes_no_answer(served, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert SP.main([str(empty), "0", str(served.out), "--hf-home", str(served.hf)]) == 2
    assert not (served.out / "answer.json").exists() and "no serve.json" in capsys.readouterr().err


def test_each_part_alone_makes_no_answer_where_the_piece_makes_none(served, tmp_path, capsys):
    """``--sessions`` and ``--answers`` (`served_piece.sh` runs one, then the other) exit 2 with nothing written where
    no answer can be made: a serve dir with no serve.json for the sessions' part, no sessions kept for the answers'.
    The two are never one call."""
    argv = [str(served.serve), "3", str(served.out), "--hf-home", str(served.hf), "--private", str(served.private)]
    assert SP.main([str(tmp_path / "nowhere"), "3", str(served.out), "--hf-home", str(served.hf), "--sessions"]) == 2
    assert "no serve.json" in capsys.readouterr().err
    assert SP.main([*argv, "--answers"]) == 2
    assert "holds no sessions to answer" in capsys.readouterr().err and not served.out.exists()
    with pytest.raises(SystemExit):
        SP.main([*argv, "--sessions", "--answers"])


def test_an_answer_the_sessions_part_leaves_already_is_never_answered_again(served, tmp_path, monkeypatch):
    """A challenger whose side doesn't start, after an earlier run of the same window kept its sessions: the sessions'
    part writes its undecided answer and returns it, what the earlier run kept is gone, and the answers' part has
    nothing to answer, so the earlier run's sessions are never answered as this one's."""
    auditor = Auditor(tmp_path)
    card = {"hf_home": served.hf, "device": "cpu", "node": NODE, "private": served.private, "pins": [served.pins],
            "protocol": TEST_PROTOCOL}
    with monkeypatch.context() as m:
        m.setattr(CS, "salt", auditor.salt)
        m.setattr(CS, "session", auditor.session)
        m.setattr(CP, "serve", functools.partial(CP.serve, salts=auditor.salts))
        assert SP.sessions(served.serve, 3, tmp_path / "earlier", **card,
                           party=Side(AUDITOR, ("ssh", AUDITOR, "challenger_party"), ("192.0.2.36", 47610))) is None
    assert (served.private / SP.HELD).exists() and (served.private / SP.STORE).exists()
    early = SP.sessions(served.serve, 3, served.out, **card,
                        party=Side(AUDITOR, (sys.executable, "-c", "import sys; sys.exit(3)"), ("127.0.0.1", 9)))
    assert early["verdict"] == "undecided" and json.loads((served.out / "answer.json").read_text()) == early
    assert not (served.private / SP.HELD).exists() and not (served.private / SP.STORE).exists()
    with pytest.raises(SP.Refusal, match="holds no sessions to answer"):
        SP.answers(served.serve, 3, served.out, private=served.private, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)


def _built(package: Path, exe: str) -> Path:
    """The Lean executable ``exe``, built in its ``package`` under the difftest's build lock."""
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    p2_lean.BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(p2_lean.BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", exe], cwd=package, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return package / ".lake" / "build" / "bin" / exe


@pytest.fixture(scope="module")
def case_exe() -> Path:
    """`remote-audit-p2-protocol`, built in the remote audit's package."""
    return _built(p2_lean.PACKAGE, p2_case.EXE)


def _script(directory: Path, name: str, body: str) -> Path:
    exe = directory / name
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return exe


@pytest.fixture(scope="module")
def run_exe(tmp_path_factory) -> Path:
    """`pous-run-protocol`, built in the run protocol's package, behind a script of its name that runs it under
    ``--test``: the protocol of the test signers' schemes, which the deployed one refuses."""
    real = _built(ROOT / "verity" / "protocols" / "pous" / "verifier" / "check", RUN_EXE)
    return _script(tmp_path_factory.mktemp("lean"), RUN_EXE, f'exec {real} --test "$@"\n')


@pytest.mark.slow
def test_with_its_lean_the_answer_is_the_leans_and_the_summary_names_it(served, tmp_path, monkeypatch, run_exe):
    """With ``--case-exe``, `pous-run-protocol` answers the case the piece holds, and its answer, the Python's but for
    the reason, is the one written: the summary names the program and this host under ``decided_by``, with the Lean's
    seconds, and the private answer is what the Lean says of the private case."""
    answer, _ = _run(served, tmp_path, monkeypatch, case_exe=str(run_exe))
    assert answer["verdict"] == "accept", answer["reason"]
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["decided_by"] == {"program": RUN_EXE, "host": socket.gethostname()}
    assert doc["lean_s"] >= 0 and "lean_peak_rss_kib" in doc and doc["key_pinned"]
    case = json.loads((served.private / "case.json").read_text())
    own = json.loads((served.private / "answer.json").read_text())
    assert period.serve(run_exe, [case]) == [own] == [json.loads((served.private / "run.json").read_text())["answer"]]
    assert own == SP.SC.pous_answer(case, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL)
    assert answer == own | {"commitment": answer["commitment"], "profile": answer["profile"]}
    row = SV.piece_row("space", served.out.parent)
    assert row["verdict"] == "accept" and row["protocol"] == TEST_PROTOCOL


def _fails(exe, cases, *args):
    raise subprocess.CalledProcessError(3, [str(exe), *args, "serve"])


REFUSAL = {"commitment": None, "profile": None, "protocol": "p", "public": None, "reason": "no", "verdict": "reject"}


@pytest.mark.parametrize("lean, reason", [
    (_fails, "doesn't answer the case: CalledProcessError"),
    (lambda exe, cases, *args: [], "doesn't answer the case: IndexError"),
    (lambda exe, cases, *args: [REFUSAL], "answers the case reject (no), its Python accept (None)")])
def test_a_lean_that_does_not_answer_as_the_python_leaves_the_window_undecided(served, tmp_path, monkeypatch, lean,
                                                                               reason):
    """A `pous-run-protocol` that fails, answers nothing, or answers the case otherwise than the Python (`period.serve`
    stood in for, so no program starts): no verdict is taken from either. The window is undecided, naming the failure
    or both answers, under the hiding commitment; no case is written for the composite; and the summary names the
    Python, whose answer this is. What the sessions kept stays, and once the Lean answers as the Python, the same
    sessions are answered with no new one, the Lean's answer the one written."""
    exe = tmp_path / RUN_EXE
    monkeypatch.setattr(period, "serve", lean)
    answer, calls = _run(served, tmp_path, monkeypatch, case_exe=str(exe))
    assert answer["verdict"] == "undecided" and answer["reason"].startswith(f"{exe} {reason}")
    assert answer["public"] is None and answer["profile"] is None and len(bytes.fromhex(answer["commitment"])) == 64
    assert SV.check_answer(answer) and SV.piece_row("space", served.out.parent)["verdict"] == "undecided"
    assert not (served.private / "case.json").exists() and not (served.private / "answer.json").exists()
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["decided_by"] == {"program": "python", "host": socket.gethostname()} and doc["lean_s"] >= 0
    run = json.loads((served.private / "run.json").read_text())
    assert run["answer"]["verdict"] == "undecided" and run["answer"]["reason"] == answer["reason"]

    assert (served.private / SP.STORE).exists() and (served.private / SP.HELD).exists()
    monkeypatch.setattr(period, "serve", lambda exe, cases, *args: [
        SP.SC.pous_answer(c, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL) for c in cases])
    again = SP.answers(served.serve, 3, served.out, trusted_sha512=hashlib.sha512(bytes.fromhex(run["public_key"])).hexdigest(),
                       private=served.private, schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, case_exe=str(exe))
    assert again["verdict"] == "accept" and len(calls) == 1
    assert SV.piece_row("space", served.out.parent)["verdict"] == "accept" and (served.private / "case.json").exists()
    assert json.loads((served.out / "pous.json").read_text())["decided_by"]["program"] == RUN_EXE
    assert not (served.private / SP.STORE).exists() and not (served.private / SP.HELD).exists()


@pytest.fixture
def registered(served, monkeypatch) -> Served:
    """`served` at PoUS's registered point, the one a P2 registration names (2,056-byte blocks, 4 a segment: the two
    layers' 8 linears in 2 segments, 2 units), with each segment's encode kept across the module's tests, so ``C*``,
    whose salt `_bandwidth`'s challenger samples from a fixed source, is encoded once."""
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
    party = (f"cd {ROOT} && exec {sys.executable} -S -E -m verity.protocols.poub.verifier.p2_party "
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
    PoUS's store (`p2_case.same_store` against the run's case), and `remote-audit-p2-protocol` answers it as the session
    did; the public answer, which the composed run reads as the bandwidth piece's, names none of the store. It encodes
    ``C*`` at the registered point (about 25 s on the CPU), the only point a P2 registration names."""
    s = registered
    answer, calls = _run(s, tmp_path, monkeypatch, **_bandwidth(tmp_path, bw_case_exe=str(case_exe)))
    assert answer["verdict"] == "accept", answer["reason"]
    public = s.out.parent / "bandwidth"
    bw = json.loads((public / "answer.json").read_text())
    assert bw["verdict"] == "accept", bw["reason"]
    published = json.loads((s.private / "run.json").read_text())["published"]
    request = calls[0][1]
    setup = CS.setup_of(request)
    assert setup["root"] == request["binding"] == published["root"]
    assert setup["salt"] == published["salt"] and setup["commitment"] == published["commitment"]
    assert setup["params"] == {k: getattr(REGISTERED.with_segments(2), k) for k in CS.PARAMS}

    private = s.serve / "private" / "bandwidth"
    case = json.loads((private / p2_case.CASE).read_text())
    own = json.loads((private / p2_case.ANSWER).read_text())
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    for f in (p2_case.CASE, p2_case.ANSWER, "session.json", "opening.json"):
        assert stat.S_IMODE((private / f).stat().st_mode) == 0o600
    pous_case = json.loads((s.private / "case.json").read_text())
    assert p2_case.same_store(case, pous_case) is None and p2_case.store_root(case).hex() == published["root"]
    assert p2_case.serve(case_exe, [case]) == [own] and own["verdict"] == "accept"
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
    assert not any((d / f).exists() for d, f in ((private, SP.HB.HELD), (s.private, SP.STORE), (s.private, SP.HELD)))
    assert _stamps(s.out) == [
        "the weights loaded", "setup done", "PoUS's session done", "PoUB's session start", "PoUB's session done",
        "PoUB's answer done", "PoUS's answer by the Python done", "PoUS's case written"]

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
           pins=[served.pins], schemes=ts.SCHEMES, protocol=TEST_PROTOCOL, bw_party="false")
    bw = json.loads((served.out.parent / "bandwidth" / "answer.json").read_text())
    assert bw["verdict"] == "undecided" and bw["reason"].startswith("no PoUS session bound a store: the memory challenger")
    assert SV.piece_row("bandwidth", served.out.parent)["verdict"] == "undecided"
    assert not (served.serve / "private" / "bandwidth").exists()
    doc = json.loads((served.out.parent / "bandwidth" / "bandwidth.json").read_text())
    assert doc["decided_by"] == {"program": "python", "host": socket.gethostname()}


def _bandwidth_files(s: Served) -> tuple[Path, dict, dict]:
    public = s.out.parent / "bandwidth"
    return (s.serve / "private" / "bandwidth", json.loads((public / "answer.json").read_text()),
            json.loads((public / "bandwidth.json").read_text()))


@pytest.mark.slow
def test_with_its_lean_the_bandwidth_answer_is_the_leans_and_the_summary_names_it(registered, tmp_path, monkeypatch,
                                                                                  case_exe):
    """With ``--bw-case-exe``, `remote-audit-p2-protocol` answers the case of the session over the held store, and its
    answer is the one written, privately and under the hiding commitment publicly, whatever the session's own party
    said of it: the summary names the program and this host under ``decided_by``, with the Lean's seconds, where a run
    without it names the Python. PoUS's answer is its own. Run in the piece's two parts: both sessions leave only what
    they kept, the P2 case not yet answered among it and nothing public but the timeline; the answers are then made
    with neither the store nor a party."""
    s = registered
    private = s.serve / "private" / "bandwidth"

    def kept():
        assert {SP.HB.SESSION, SP.HB.HELD} <= set(_names(private)) and p2_case.CASE not in _names(private)
        assert stat.S_IMODE((private / SP.HB.HELD).stat().st_mode) == 0o600
        assert _names(s.out) == ["stamps.txt"] and not (s.out.parent / "bandwidth").exists()
    answer = _parts(s, tmp_path, monkeypatch, between=kept, **_bandwidth(tmp_path, bw_case_exe=str(case_exe)))
    assert answer["verdict"] == "accept", answer["reason"]
    assert not (private / SP.HB.HELD).exists()
    private, bw, doc = _bandwidth_files(s)
    case = json.loads((private / p2_case.CASE).read_text())
    lean, = p2_case.serve(case_exe, [case])
    assert json.loads((private / p2_case.ANSWER).read_text()) == lean
    assert (bw["verdict"], bw["protocol"], bw["public"] is None) == (lean["verdict"], lean["protocol"],
                                                                    lean["public"] is None)
    assert doc["decided_by"] == {"program": p2_case.EXE, "host": socket.gethostname()} and doc["lean_s"] >= 0
    assert doc["verdict"] == lean["verdict"] and SV.piece_row("bandwidth", s.out.parent)["verdict"] == lean["verdict"]
    opening = json.loads((private / "opening.json").read_text())
    assert opening["row"] == lean["commitment"] and opening["root"] == bw["commitment"]


@pytest.mark.slow
def test_a_bandwidth_lean_that_does_not_answer_leaves_the_bandwidth_undecided(registered, tmp_path, monkeypatch):
    """A `remote-audit-p2-protocol` that fails on the case (`p2_case.serve` stood in for): the session's own answer is
    not taken in its place. The bandwidth answer is undecided naming the failure, no case is written, the summary names
    the Python, whose answer this is, and PoUS's answer stands."""
    exe = tmp_path / p2_case.EXE
    monkeypatch.setattr(p2_case, "serve", _fails)
    answer, _ = _run(registered, tmp_path, monkeypatch, **_bandwidth(tmp_path, bw_case_exe=str(exe)))
    assert answer["verdict"] == "accept", answer["reason"]
    private, bw, doc = _bandwidth_files(registered)
    assert bw["verdict"] == "undecided" and bw["reason"].startswith(f"{exe} doesn't answer the case: CalledProcessError")
    assert bw["public"] is None and bw["commitment"] is None and SV.check_answer(bw)
    assert not (private / p2_case.CASE).exists() and not (private / p2_case.ANSWER).exists()
    assert doc["decided_by"] == {"program": "python", "host": socket.gethostname()} and "lean_s" not in doc
    assert SV.piece_row("bandwidth", registered.out.parent)["verdict"] == "undecided"
    assert json.loads((registered.out / "pous.json").read_text())["bandwidth"]["verdict"] == "undecided"
    # the case its Lean has not answered stays, with what PoUS's answer needs, for another --answers
    assert (private / SP.HB.HELD).exists() and (registered.private / SP.STORE).exists()


@pytest.mark.slow
def test_a_bandwidth_lean_that_refuses_the_case_undecoded_is_the_answer(registered, tmp_path, monkeypatch):
    """A `remote-audit-p2-protocol` that decodes no commitment of the case refuses it with none, as `Protocol.answer`
    does (`p2_case.serve` stood in for): that refusal is the bandwidth answer, with no commitment to hide, and the
    summary names the Lean."""
    refusal = {"commitment": None, "profile": None, "protocol": p2_case.PROTOCOL, "public": None,
               "reason": "params: not the request's", "verdict": "reject"}
    exe = tmp_path / p2_case.EXE
    monkeypatch.setattr(p2_case, "serve", lambda exe, cases: [refusal])
    answer, _ = _run(registered, tmp_path, monkeypatch, **_bandwidth(tmp_path, bw_case_exe=str(exe)))
    assert answer["verdict"] == "accept", answer["reason"]
    private, bw, doc = _bandwidth_files(registered)
    assert bw["verdict"] == "reject" and bw["reason"].startswith("params: not the request's; ")
    assert bw["commitment"] is None and SV.check_answer(bw)
    assert json.loads((private / p2_case.ANSWER).read_text()) == refusal and not (private / "opening.json").exists()
    assert doc["decided_by"] == {"program": p2_case.EXE, "host": socket.gethostname()}
    assert SV.piece_row("bandwidth", registered.out.parent)["verdict"] == "reject"
