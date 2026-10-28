"""PoUS's piece of the composed served run (`benchmarks/one_stage/served_verdicts.sh`'s ``POUS_CMD``): one period of
PoUS over the served checkpoint's linear weights, held as P2's ``C*`` on the card the window served on, with the memory
challenger on the auditor's host; its core answer in ``DIR/answer.json``.

    served_piece.py SERVE_DIR WINDOW DIR --hf-home H [--p2-lib LIB] [--device cuda] [--node NAME] [--private P]
      [--rounds K] [--challenger-host NAME --challenger-address HOST:PORT --challenger-command ARGV] [--trusted-sha512 HEX]

`served_piece.sh` runs it under a lease of the card, as the composed run starts a piece (``bash -c CMD _ SERVE_DIR WINDOW
DIR``).

1. **The weights**: ``serve.json``'s model at its revision, the snapshot under ``--hf-home``, each of its safetensors files
   held to its sha256 in the int8 manifest (`PINS`), and the serve dir's weights registration, when it has one, naming the
   same model. ``W`` is each decoder layer's four linears as vLLM's Llama holds them (`MERGED`), int8 [N, K] beside f32
   [N, 1] scales, put on ``--device``: the weights the serve read from that snapshot, loaded again on the card after the
   window's serve ended. The rest of the checkpoint is outside ``W`` (the profile's ``outside``), and so is what serving
   adds: the KV cache's pages and the activations, which the weights-only obligation doesn't cover and an honest run
   doesn't need.
2. **Setup** by the vLLM adapter (`verity_vllm.protocols.pous`, ``pous:p2-16448/v4``, its registered point): the
   commitment to ``W``, then the salt, which the memory challenger draws after it from its own host's OS and logs under
   the window's binding (`binding`; `challenger_party.salt`, the gate's ``challenger_party`` verb on a salt request),
   ``C*`` encoded on the card (``libp2v1.so``, ``--p2-lib``), the memory challenger's table and the root. Setup's own
   inputs are kept as it sent them, for the tie. A challenger that draws no salt leaves the period undecided.
3. **The session**: the memory challenger's side started on its host by ``--challenger-command`` (the auditor gate's
   ``challenger_party`` verb), the prover connecting to ``--challenger-address`` and answering each of ``--rounds``
   indices (`ROUNDS` by default) from ``C*`` on the card (`challenger_party.session`). The request names the window's
   binding and setup's published outputs; the challenger refuses it unless they carry the salt it logged and the
   commitment it was drawn after, and runs one session a binding, under the id derived from them
   (`challenger_party.session_id`).
4. **The verdict**: the run protocol's case (`period.case`: setup's published outputs, the session, the table and setup's
   own inputs) under the session id the verdict derives itself from the window's binding and what setup published,
   never the report's (a report of a session under any other id is a reject), and its answer on that case's bytes
   (`served_case.pous_answer`, `period.answer`): the tie, the run's
   check under the challenger key that ``--trusted-sha512`` pins (a session signed under any other is refused), and what
   an accepted run certifies (`period.certify`): at `ROUNDS` challenges, ``⌊ρ·|C|⌋`` bits at ``2^-20``
   (`p2.wide320_certificate`); at 105, at ``1% + 2^-128``. The check takes about 2.3 minutes at the registered point at
   105 challenges or 320: the tie is its cost, not the rounds.
5. **Zero knowledge**: the answer's commitment is a hiding one, one ``hm96-sha512/row/v2`` row in a ``frame-v3-sha512``
   tree (`verity.ml.commitments.advice`) holding the run's commitment (the SHA-512 of what setup published) under
   `binding`, so the record fixes the run and reveals nothing of ``W``'s digest, the salt or ``C*``'s root. Its opening,
   setup's published outputs, the session and its id, the rounds' latencies, the trusted key, the table, and the case with
   the run protocol's own answer on it (``case.json``, ``answer.json``: the composite's piece, `served_case.write`) go to
   ``--private`` (by default
   ``SERVE_DIR/private/pous/w<WINDOW>``, 0700), never to ``DIR``. The profile carries the point, the certificate, the
   placement and the weights' public properties: their model, count and size.

Writes ``DIR/answer.json`` and ``DIR/pous.json`` (sizes and the phases' timings) and prints one ``POUS`` line.
Weights that aren't the pin's are a reject, and a session the challenger's side couldn't complete is undecided, each with
its reason. Exit 0 on accept, 1 on any other answer, 2 when no answer could be made (no serve.json, no snapshot).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import statistics
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from verity.core.deploy.stream import Side
from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous.schemes import p2 as P2
from verity.ml.commitments import advice
from verity.ml.commitments.identity import canonical_json_bytes
from verity_vllm import protocols as PO

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import served_case as SC

pous = PO.adapter("pous")
ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "integrations" / "vllm" / "manifests" / "checkpoints-int8.json"
ENTRY = f"pous:{pous.DEFAULT_SCHEME}"
#: each linear as vLLM's Llama holds it, from the checkpoint's tensors concatenated by rows in this order
MERGED = (("self_attn.qkv_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
          ("self_attn.o_proj", ("self_attn.o_proj",)),
          ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
          ("mlp.down_proj", ("mlp.down_proj",)))
SERVING = ("the KV cache's pages and the activations: serving state, which the weights-only obligation doesn't cover and "
           "an honest run doesn't need")
#: the auditor's host and how the card reaches its challenger: the gate's `challenger_party` verb under the card's key,
#: and the port the side listens on, which the security groups open from the card only
CHALLENGER_HOST = "vy-auditor-2"
CHALLENGER_ADDRESS = "10.96.0.36:47610"
CHALLENGER_COMMAND = ("ssh", "-i", "~/.ssh/vy-auditor2-pous", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                      "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "research@10.96.0.36",
                      "challenger_party")
#: the SHA-512 of the challenger's public key as its host made it (the gate's `party_key pous`)
TRUSTED_SHA512 = ("39cd635aea2e124a0fcae035e416b107de27955118914642bb8d9d06814a17a2"
                  "f73014a3b9229bf1293ce4703ccc52a5f3119d22789db6778f34e237dfa631bb")
BINDING_TAG = b"verity/pous/served-window/v1"
#: the period's challenges: the 320-read window's, at which an accepted run certifies at `2^-20`
ROUNDS = P2.CERT_K_WIDE320
DTYPES = {"I8": torch.int8, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
SIZES = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4, "I32": 4, "F64": 8, "I64": 8}


class Refusal(Exception):
    """No answer can be made: what the piece needs isn't there."""


class _Unused:
    """The layers' ``quant_method``, whose ``apply`` the adapter wraps; the piece runs no forward."""

    def apply(self, layer: Any, x: Any, bias: Any = None) -> Any:
        raise RuntimeError("the PoUS piece runs no forward")


class Linear:
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.quant_method = _Unused()


class Model:
    """The served model's linears, as the adapter reads a model: by module and by parameter."""

    def __init__(self, layers: dict[str, Linear]) -> None:
        self.layers = layers

    def named_modules(self):
        yield "", self
        yield from self.layers.items()

    def named_parameters(self):
        for n, m in self.layers.items():
            yield f"{n}.weight", m.weight
            yield f"{n}.weight_scale", m.weight_scale

    def named_buffers(self):
        return iter(())


def _linear_types() -> tuple[type, ...]:
    return (Linear,)


class NoSalt(Exception):
    """The memory challenger drew no salt for the window's setup."""


def _salt_source(party: Side, audit: bytes):
    """The salt's source: drawn by the memory challenger on ``party`` once the commitment exists, and logged there under
    the window's binding ``audit`` (`challenger_party.salt`)."""

    def source(commitment: bytes) -> bytes:
        try:
            return CP.salt(party, audit, commitment)
        except (OSError, RuntimeError, ValueError) as e:
            raise NoSalt(f"{type(e).__name__}: {e}") from e
    return source


@dataclass(frozen=True)
class Checkpoint:
    """The snapshot's tensors: each one's file, dtype, shape and byte range in that file's data."""

    files: dict[str, bytearray]
    tensors: dict[str, tuple[str, str, tuple[int, ...], int, int]]

    def bytes_of(self, name: str) -> int:
        _, _, _, a, b = self.tensors[name]
        return b - a

    def tensor(self, name: str) -> torch.Tensor:
        file, dtype, shape, a, b = self.tensors[name]
        if dtype not in DTYPES:
            raise ValueError(f"{name}: dtype {dtype} is not one the piece reads")
        return torch.frombuffer(bytearray(memoryview(self.files[file])[a:b]), dtype=DTYPES[dtype]).reshape(shape)


def _read(path: Path) -> tuple[bytearray, str]:
    """The file's bytes and their sha256, in one pass."""
    h, buf = hashlib.sha256(), bytearray()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
            buf += chunk
    return buf, h.hexdigest()


def _pin(repo: str, revision: str, pins: Path) -> dict[str, str]:
    """Each safetensors file's sha256 in the manifest's entry for ``repo`` at ``revision``."""
    for c in json.loads(pins.read_text())["checkpoints"]:
        if (c.get("repo"), c.get("revision")) == (repo, revision):
            return {f: e["sha256"] for f, e in c["files"].items() if f.endswith(".safetensors")}
    raise Refusal(f"{pins.name} pins no {repo}@{revision}")


def checkpoint(snapshot: Path, pin: Mapping[str, str]) -> tuple[Checkpoint, list[str]]:
    """The snapshot's safetensors files read and parsed, and why they aren't the pin's (empty when they are)."""
    files: dict[str, bytearray] = {}
    tensors: dict[str, tuple[str, str, tuple[int, ...], int, int]] = {}
    wrong: list[str] = []
    for name in sorted(pin):
        if not (snapshot / name).is_file():
            raise Refusal(f"the snapshot {snapshot} has no {name}")
        data, sha = _read(snapshot / name)
        if sha != pin[name]:
            wrong.append(f"{name} has sha256 {sha}, not the pin's {pin[name]}")
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + n])
        header.pop("__metadata__", None)
        files[name] = data
        for key, e in header.items():
            a, b = e["data_offsets"]
            tensors[key] = (name, e["dtype"], tuple(e["shape"]), 8 + n + a, 8 + n + b)
    return Checkpoint(files, tensors), wrong


def linears(ck: Checkpoint, device: torch.device) -> tuple[Model, dict[str, Any]]:
    """vLLM's linears of the checkpoint on ``device`` (`MERGED`), and the checkpoint's other tensors' bytes, by kind."""
    keys = set(ck.tensors)
    layers: dict[str, Linear] = {}
    used: set[str] = set()
    n = 1 + max((int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")), default=-1)
    for i in range(n):
        p = f"model.layers.{i}."
        for merged, parts in MERGED:
            names = [p + s + ".weight" for s in parts]
            missing = [k for k in names + [k + "_scale" for k in names] if k not in keys]
            if missing:
                raise ValueError(f"{p + merged}: the checkpoint has no {missing}")
            w = torch.cat([ck.tensor(k) for k in names]).contiguous()
            s = torch.cat([ck.tensor(k + "_scale") for k in names]).contiguous()
            if w.dtype != torch.int8 or s.dtype != torch.float32 or s.shape != (w.shape[0], 1):
                raise ValueError(f"{p + merged}: {w.dtype} {tuple(w.shape)} with {s.dtype} {tuple(s.shape)} scales is not "
                                 "an int8 linear")
            layers[p + merged] = Linear(w.to(device), s.to(device))
            used.update(names)
    outside: dict[str, int] = {}
    for k in sorted(keys - used):
        kind = ("the embedding and the head" if "embed" in k or "lm_head" in k else "the per-row scales"
                if k.endswith("_scale") else "the norms" if "norm" in k else "other tensors")
        outside[kind] = outside.get(kind, 0) + ck.bytes_of(k)
    return Model(layers), {"decoder_layers": n, "bytes": outside, "serving": SERVING}


def binding(model: str, revision: str, window: int, registration: str | None) -> bytes:
    """What the answer's hiding commitment is bound to: the served model, the window and its weights registration."""
    doc = {"model": model, "registration_sha512": registration, "revision": revision, "window": window}
    return hashlib.sha512(BINDING_TAG + canonical_json_bytes(doc)).digest()


def _no(verdict: str, reason: str, commitment: str | None = None, protocol: str = period.PROTOCOL) -> dict[str, Any]:
    return {"commitment": commitment, "profile": None, "protocol": protocol, "public": None, "reason": reason,
            "verdict": verdict}


def _latencies(session: R.Session, limit_ns: int) -> dict[str, Any]:
    ns = [r.receipt_ns - r.send_ns for r in session.signed.rounds]
    if not ns:
        return {"rounds": 0}
    return {"rounds": len(ns), "median_us": round(statistics.median(ns) / 1e3, 1), "max_us": round(max(ns) / 1e3, 1),
            "limit_us": round(limit_ns / 1e3, 1), "late": sum(x > limit_ns for x in ns)}


def _private(directory: Path, docs: Mapping[str, Any], digests: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    for name, doc in docs.items():
        (directory / name).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (directory / "digests.bin").write_bytes(digests)


def run(serve_dir: Path, window: int, out: Path, *, hf_home: Path, device: str = "cuda", p2_lib: str | None = None,
        node: str | None = None, party: Side | None = None, trusted_sha512: str = TRUSTED_SHA512,
        private: Path | None = None, pins: Path = PINS, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
        protocol: str = period.PROTOCOL, rounds: int = ROUNDS) -> dict[str, Any]:
    """The piece (module docstring): its answer, written with ``pous.json`` to ``out``; `Refusal` when none can be made.
    ``schemes`` and ``protocol`` are the signatures the verifier accepts and the id it answers under (a test signer's in
    tests), ``rounds`` the session's challenges."""
    if not (serve_dir / "serve.json").is_file():
        raise Refusal(f"{serve_dir} has no serve.json")
    serve = json.loads((serve_dir / "serve.json").read_text())
    repo, rev = serve.get("model"), serve.get("revision")
    if not (isinstance(repo, str) and isinstance(rev, str)):
        raise Refusal(f"serve.json names no model and revision ({repo!r}, {rev!r})")
    snapshot = hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not snapshot.is_dir():
        raise Refusal(f"no snapshot of {repo}@{rev} under {hf_home}")
    node = node or socket.gethostname()
    party = party or Side(CHALLENGER_HOST, tuple(os.path.expanduser(a) for a in CHALLENGER_COMMAND),
                          _address(CHALLENGER_ADDRESS))
    private = private or serve_dir / "private" / "pous" / f"w{window}"
    out.mkdir(parents=True, exist_ok=True)
    reg_file = serve_dir / "weights_registration.json"
    reg_sha = hashlib.sha512(reg_file.read_bytes()).hexdigest() if reg_file.is_file() else None
    bind = binding(repo, rev, window, reg_sha)
    summary: dict[str, Any] = {"window": window, "model": repo, "revision": rev, "node": node, "challenger": party.host,
                               "binding": bind.hex(), "registration_sha512": reg_sha}

    def done(answer: dict[str, Any], **more: Any) -> dict[str, Any]:
        (out / "answer.json").write_text(json.dumps(answer, indent=1, sort_keys=True) + "\n")
        (out / "pous.json").write_text(json.dumps(summary | more | {"verdict": answer["verdict"],
                                                                    "reason": answer["reason"]},
                                                  indent=1, sort_keys=True, default=str) + "\n")
        return answer

    t0 = time.monotonic()
    ck, wrong = checkpoint(snapshot, _pin(repo, rev, pins))
    if reg_sha is not None:
        reg = json.loads(reg_file.read_text()).get("model") or {}
        if (reg.get("repo"), reg.get("revision")) != (repo, rev):
            wrong.append(f"the weights registration is of {reg.get('repo')}@{reg.get('revision')}, not the served "
                         f"{repo}@{rev}")
    if wrong:
        return done(_no("reject", "the served weights are not the pinned checkpoint's: " + "; ".join(wrong), None, protocol))
    dev = torch.device(device)
    model, outside = linears(ck, dev)
    total = sum(ck.bytes_of(k) for k in ck.tensors)
    del ck
    load_s = round(time.monotonic() - t0, 3)

    types = pous.linear_types
    pous.linear_types = _linear_types
    try:
        ctx = PO.Context(stage="serve", enabled=PO.parse([ENTRY]), settings={"pous": {
            "salt_source": _salt_source(party, bind), "p2_lib": p2_lib, "experimental": True}})
        t = time.monotonic()
        try:
            comp = PO.install(model, ctx)
        except NoSalt as e:
            return done(_no("undecided", f"the memory challenger on {party.host} drew no salt for the setup: {e}"[:1000],
                            None, protocol))
        install_s = round(time.monotonic() - t, 3)
        try:
            opt = comp.get("pous")
            s = opt.store
            point = replace(s.point, k=rounds)
            inputs = (s.commitment, s.salt, s.segments())
            details = opt.details
            weights = {"model": repo, "revision": rev, "covers": pous.COVERS, "linears": details["layers"], "bytes": details["tensor_bytes"],
                       "used_blocks": details["used_blocks"], "blocks": details["blocks"], "checkpoint_bytes": total,
                       "share": str(Fraction(details["tensor_bytes"], total))}
            summary |= {"load_s": load_s, "install_s": install_s, "weights": weights, "outside": outside,
                        "details": details}
            published = s.published()
            sid = CP.session_id(bind, s.commitment, s.salt)
            t = time.monotonic()
            try:
                rep = CP.session(party, CP.request_of(node, point, s.root, audit=bind, published=published),
                                 opt.prover(()))
                key = bytes.fromhex(rep["public_key"])
                session = R.Session.from_json(rep["session"])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
                return done(_no("undecided", f"the memory challenger's session on {party.host} did not complete: "
                                f"{type(e).__name__}: {e}"[:1000], None, protocol), session_s=round(time.monotonic() - t, 3))
            session_s = round(time.monotonic() - t, 3)
            if rep.get("session_id") != sid.hex():
                return done(_no("reject", "the memory challenger's report is of a session under another id than the one "
                                "the window's binding and its setup fix (challenger_party.session_id)", None, protocol),
                            session_s=session_s)
            key_sha = hashlib.sha512(key).hexdigest()
            trusted = (key,) if key_sha == trusted_sha512 else ()
            t = time.monotonic()
            case = period.case(point, sid, trusted, published, s.digests, session,
                               period.setup_case(s.point.params, *inputs))
            del inputs
            inner = SC.pous_answer(case, schemes=schemes, protocol=protocol)
            answer_s = round(time.monotonic() - t, 3)
            row = bytes.fromhex(inner["commitment"])
            hidden = advice.register([row], 8 * len(row), bind)
            opening = hidden.open(0)
            _private(private, {
                "opening.json": {"binding": bind.hex(), "count": 1, "index": opening.index, "n_bits": 8 * len(row),
                                 "path": [p.hex() for p in opening.path], "root": hidden.root.hex(), "row": row.hex(),
                                 "salt": opening.salt.hex()},
                "run.json": {"answer": inner, "point": period.point_json(point), "published": published.to_json(),
                             "session": rep["session"], "session_id": sid.hex(), "trusted": [k.hex() for k in trusted],
                             "public_key": key.hex(), "host": rep.get("host"), "rounds_answered": rep.get("rounds"),
                             "warmups": rep.get("warmups"), "latency": _latencies(session, point.limit_ns)}},
                     s.digests)
            SC.write(private, case, inner)
            del case
            answer = inner | {"commitment": hidden.root.hex()}
            if inner["verdict"] == "accept":
                answer["profile"] = inner["profile"] | {
                    "commitment": {"scheme": "hm96-sha512/row/v2 in frame-v3-sha512 (verity.ml.commitments.advice)",
                                   "rows": 1, "n_bits": 8 * len(row), "binding": bind.hex(),
                                   "of": "the run protocol's commitment, the SHA-512 of setup's published outputs"},
                    "placement": CP.placement(node, party.host) | {
                        "salt": f"drawn by the memory challenger on {party.host} from its OS after the commitment, "
                                "logged under the window's binding, with the session's id derived from both"},
                    "weights": weights | {"held": "loaded from the served snapshot on the card the window served on, "
                                                  "after its serve ended"},
                    "outside": outside, "window": window}
            return done(answer, session_s=session_s, answer_s=answer_s, key_sha512=key_sha,
                        key_pinned=bool(trusted), private=str(private), public=inner["public"])
        finally:
            comp.close()
    finally:
        pous.linear_types = types


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host, int(port)


def line(answer: Mapping[str, Any]) -> str:
    pub = answer.get("public") or {}
    w = (answer.get("profile") or {}).get("weights") or {}
    if answer["verdict"] == "accept":
        return (f"POUS: accept: {w.get('model')}@{str(w.get('revision'))[:10]}'s {w.get('linears')} linears, "
                f"{w.get('bytes')} B of the checkpoint's {w.get('checkpoint_bytes')} in {w.get('used_blocks')} of "
                f"{w.get('blocks')} blocks; certifies {pub}; "
                f"{(answer['profile'].get('placement') or {}).get(CP.PREMISE)}")
    return f"POUS: {answer['verdict']}: {answer['reason']}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("serve_dir", type=Path)
    ap.add_argument("window", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hf-home", type=Path, required=True, help="the HF cache the serve read its snapshot from")
    ap.add_argument("--p2-lib", help="libp2v1.so's directory (benchmarks/torch_gpu/p2_stage/build.sh's lib/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--node", help="the prover's node, as the session names it (default: this host's name)")
    ap.add_argument("--private", type=Path, help="where the private record goes (default SERVE_DIR/private/pous/wK)")
    ap.add_argument("--rounds", type=int, default=ROUNDS, help=f"the session's challenges (default {ROUNDS})")
    ap.add_argument("--challenger-host", default=CHALLENGER_HOST)
    ap.add_argument("--challenger-address", default=CHALLENGER_ADDRESS, help="HOST:PORT the prover connects to")
    ap.add_argument("--challenger-command", help="the argv that starts the challenger's side, one shell-split line")
    ap.add_argument("--trusted-sha512", default=TRUSTED_SHA512, help="the SHA-512 of the challenger key the verifier trusts")
    a = ap.parse_args(argv)
    command = tuple(shlex.split(a.challenger_command)) if a.challenger_command else CHALLENGER_COMMAND
    party = Side(a.challenger_host, tuple(os.path.expanduser(x) for x in command), _address(a.challenger_address))
    try:
        answer = run(a.serve_dir, a.window, a.out, hf_home=a.hf_home, device=a.device, p2_lib=a.p2_lib, node=a.node,
                     party=party, trusted_sha512=a.trusted_sha512, private=a.private, rounds=a.rounds)
    except Refusal as e:
        print(f"POUS: no answer: {e}", file=sys.stderr, flush=True)
        return 2
    print(line(answer), flush=True)
    return 0 if answer["verdict"] == "accept" else 1


if __name__ == "__main__":
    sys.exit(main())
