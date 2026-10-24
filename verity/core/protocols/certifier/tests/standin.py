"""A stand-in for N2's proof of a link-window, and for the ingress tie's of an ingress link-window (`prove_tie`), for
tests: what C-Flock's prover behind the firewall would deliver, and a stand-in for the verifier's own run of C-Flock's
Lean verifier (`proofs.verify`), which reads each session's verdict from the session file the stand-in prover writes.
The proof's output word is the unit's on the prover's opening by each gate's reference (`LinkWindows.outputs`,
`IngressTie.outputs`), which the tests hold to the circuits: a Lean-accepted proof can't claim another. So the tests
check everything the verifier of record does around the proof: its run's findings, the public row opened to the
verifier's own copy, the output row, the certificates and the sync."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding

_INST = hm96.SHA512
SALTS = {"pub": b"\x51" * 192, "wit": b"\x52" * 192, "o0": b"\x53" * 192}


def _bc(data: bytes, bits: int, salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, sha512_row_v2_digest(data, bits), salt)


def prove(lw: SVC.LinkWindows, opening: SVC.Opening, draw: S.Draw, syncs: Sequence[Any], out: Path, *,
          program: str | None = None, accepted: bool = True, word: int | None = None, root: bytes | None = None,
          salts: dict[str, bytes] | None = None, units: Sequence[int] | None = None,
          advice: Any = None) -> SVC.Delivered:
    """The delivered proof of the drawn unit on `opening` under the registered root (or `root`), on an advised link of
    `advice`: the statement's public file, its one instance's rows committed under `SALTS` (``pub``, ``wit``, ``o0``),
    one session naming `program` (default: the link's registered Program) that the stand-in Lean `accepted`, and the
    circuit file. Where the unit takes no instance, the rows are zero and the output 0. `word` forges the output word,
    `salts` the salts delivered for ``pub`` and ``o0``, and `units` the public file's unit indices."""
    root = bytes.fromhex(draw.registration.record["roots"][0]["root"]) if root is None else root
    args = lw.arguments(opening, root, syncs, advice)
    rows = {r: lw.row(args, r) if args is not None else bytes(lw.bits(r) // 8) for r in SVC.ROWS}
    rows["o0"] = SVC.output_row(lw.outputs(opening, root, syncs, advice) if args is not None else 0)
    return _deliver(lw, rows, draw, out, program=program, accepted=accepted, word=word, salts=salts, units=units)


def _deliver(unit: Any, rows: dict[str, bytes], draw: S.Draw, out: Path, *, program: str | None, accepted: bool,
             word: int | None, salts: dict[str, bytes] | None, units: Sequence[int] | None) -> SVC.Delivered:
    """The statement's files for `rows` on `unit` (a `LinkWindows` or an `IngressTie`) and what the prover delivers."""
    out.mkdir(parents=True, exist_ok=True)
    public = out / "public.bin"
    header = {"units": {"indices": list(draw.units if units is None else units)}, "instances": 1}
    public.write_bytes(json.dumps(header).encode() + b"\n"
                       + b"".join(_bc(rows[p], unit.bits(p), SALTS[p]) for p in SVC.PORTS))
    session = out / "session-0"
    session.mkdir(exist_ok=True)
    (session / "session.json").write_text(json.dumps({"accepted": accepted, "units": [0],
                                                      "program": program or unit.partition["program"]}))
    circuit = out / "circuit.txt"
    circuit.write_text(f"stand-in circuit of {unit.unit.id}\n")
    delivered = {"pub": SALTS["pub"], "o0": SALTS["o0"], **(salts or {})}
    output = int.from_bytes(rows["o0"], "little") if word is None else word
    return SVC.Delivered(PF.Proof(circuit, public, (session,)), delivered, output)


def prove_tie(tie: Any, cells: Any, draw: S.Draw, view: Any, out: Path, *, program: str | None = None,
              accepted: bool = True, word: int | None = None, salts: dict[str, bytes] | None = None,
              units: Sequence[int] | None = None) -> SVC.Delivered:
    """The delivered proof of the drawn ingress-tie unit (`tie_service.IngressTie`) on the warden's `cells` under the
    registered root and the prover's `view` (`io_request.RequestView`), as `prove` delivers N2's: its output bit by
    IoTieRequest's reference (`io_request.reference`), which IO-TIE's tests hold to the circuit. Where the unit takes no
    instance of `view`, the rows are zero and the output 0."""
    root = bytes.fromhex(draw.registration.record["roots"][0]["root"])
    args = None if view is None else tie.arguments(cells, root, view)
    rows = {r: tie.row(args, r) if args is not None else bytes(tie.bits(r) // 8) for r in SVC.ROWS}
    rows["o0"] = SVC.output_row(tie.outputs(cells, root, view) if args is not None else 0)
    return _deliver(tie, rows, draw, out, program=program, accepted=accepted, word=word, salts=salts, units=units)


def circuit_sha512(delivered: SVC.Delivered) -> str:
    return PF.file_sha512(delivered.proof.circuit)


def lean(proof: PF.Proof, *, circuit_sha512: str, program: bytes, partition: dict,
         registered: dict | None = None) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """`proofs.verify`'s stand-in: its refusal of a circuit file other than the committed one, then each session's
    finding as its session file says (`prove`), with the units the public file's indices name."""
    assert registered is None and isinstance(program, bytes) and "program" in partition
    if PF.file_sha512(proof.circuit) != circuit_sha512:
        return None, ("circuit", "the statement's circuit file is not the one the auditor committed")
    out = []
    for s in proof.sessions:
        v = json.loads((s / "session.json").read_text())
        indices = json.loads(proof.public.read_bytes().split(b"\n", 1)[0])["units"]["indices"]
        out.append(Finding(str(s), v["accepted"], "" if v["accepted"] else "rejected", v["program"], {},
                           tuple(indices[u] for u in v["units"]) if v["accepted"] else ()))
    return out, None


def decide(lw: SVC.LinkWindows, stream: S.Stream, draw: S.Draw, opening: SVC.Opening, syncs: Sequence[Any], out: Path,
           *, certificates: Sequence[Any] | None = None, advice: Any = None, **kw: Any) -> S.Outcome:
    """The drawn link-window proved on `opening` (`prove`, with `kw`, on an advised link of `advice`), delivered and
    decided by the verifier of record on the proof, the certificates (by default the opening's) and `advice`, with the
    opening's in-the-clear check beside it as the debug comparison, which must agree with the proof's verdict when
    nothing is forged."""
    delivered = prove(lw, opening, draw, syncs, out, advice=advice, **kw)
    assert stream.deliver(draw, delivered.sessions()) is None
    certificates = [r.certificate for r in opening.records] if certificates is None else certificates
    outcome = lw.outcome(stream, draw, certificates, syncs, delivered, circuit_sha512=circuit_sha512(delivered),
                         opening=opening, advice=advice)
    verdict = stream.records[-1]["audit"]["verdicts"][SVC.VERIFIER]
    assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
    if not kw:
        assert verdict["clear"]["accepted"] == outcome.accepted
    return outcome
