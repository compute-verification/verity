"""One execution phase's network certificates through the proof system (N2): the network `Spec`, the ZK server's salted
commitment to the execution phase's certified records, and the check's verifier.

**What is committed.** One `Value`, `certificates`: four leaves under one salted root, the execution phase's two
certified records (`CertifiedRecord`), egress leaf 0 and ingress leaf 1, and their heads, egress leaf 2 and ingress leaf
3 (`rows`).
Each leaf is `hm96-sha512/row/v2` over its bytes, salted with its own fresh 192-byte salt, in a `frame-v3-sha512` tree
under one-stage's served domain, whose run is the link and the window. A record leaf's bytes are `encode`'s (the record
head, the attester's salt for it, then the certificate), and a head leaf's are the head's alone (`RecordHead.encode`).
A proof that reads only the head, such as the head check's, opens a head leaf in the head's two or three SHA-512
compressions instead of the certificate's 272. The certificates, salts
and heads are witness: the challenger sees the root, never a certificate, a salt or a head.

**Selection** is `All`: every execution phase is checked, so an accepted outcome is the exhaustive fact. A registration
is one execution phase, one proof block. Its `window` provenance is `{"kind": "served", "window": w}`, and the
challenger holds w to the registration's order in the stream (`TransferBlocks.register`), so the receipt's order is the
execution phase. The
network verdict of record is `TransferBlocks.network_verdict`, the AND over the link's execution phases with no ε or share
credit: it accepts only when every window of the link (its `NetworkAttesterProfile`'s) is registered in order and its outcome
of record accepted, so a rejected or late window, one with no outcome, and a gap in the registrations each
reject the link. A run's is the AND over its links'.

**The period sample** (`TransferBlocks.period_draw`) samples the execution phases instead. The prover first commits the
period: every window's salted root (`root`), which the challenger records before it reads a coin (coins with a `commit`
are first committed to the period's digest, as a remote challenger's is). It then samples `subset:K` of the n
windows, and only the drawn windows are registered, in window order, each held to its committed root
(`register(drawn=…)`), selected and proved. `network_verdict(drawn=…)` accepts when exactly the drawn windows are
registered, in order, each under its committed root, with an accepted outcome of record, and it states what the sample
established: except with probability δ, at most `wrong_units["bound"]` of the n committed windows are wrong (the
uniform K-subset's exact bound, one-stage's `wrong_units` over windows), so ε = bound / n at δ. The challenger's sample
is Python's
(`service.select_draw`), which records the coin bytes it read (`read`); the sample of record is the Lean
verifier's on those bytes (`lean_draw`: Flock's `verity-verify-proof draw --population n --k K --stream-file`, the Lean
`sample` the one-stage drivers run, `Flock.Draw.sampleOn`), which the network verdict runs itself, as a registration's own
sample is the Lean's `Select.select` on the bytes it read. The verdict accepts only when the windows registered and
proved are exactly that sample's, and otherwise refuses by name (`PERIOD_DRAWN`), whatever proof blocks the sample
object it is handed lists. Selection `All` (K None) samples every window and reads no coin, and K = n leaves nothing to
sample. The datacenter model (`benchmarks/datacenter`, a Python model) builds no
`TransferBlocks` outcomes, so its account doesn't read it, though `Account.sound` is the same AND over its own port
records.

**Public items**, each with its reason (`PUBLIC`): the population (one), the window (its index, the order), the
committed value's leaf count and domain, its salted root, the parameters digests, the network attester's schedule and
the link's windows (fixed before the run, bound by the spec's context), and the check's two public outputs, `verdict`
and `deadline`. A rejected execution phase shows only the service's `reject`: the violation's kind stays in the
verifier.

**The check** is `TransferBlocks.unit`, `NetworkTransferBlocks` v5{EGRESS, INGRESS}: each end's head check,
`head_check.NetworkHeadCheck_v2{L, T, COUNTS}` (EGRESS and INGRESS are those statics), a Boolean Definition over its
end's signed digest (`certificate_digest`: the end's record digest hidden under the attester's salt), its direction a
constant of the proof block (egress 0, ingress 1). The head opens at leaf 2 + direction under the registered root, a
hiding read; its link, direction and window are the execution phase's; its status is complete; its T, r,
B, τ_b and parameters digest are the end's declared grid's; and its record digest under the attester's salt is the
signed digest, so the certificate signs the head the check opened. The proof block's parameters are the two signed
digests (``egress``, ``ingress``) and each end's head check's others, ``<end>_<name>``; its output is two bits,
`OUTPUTS`: ``verdict``, the AND of the head checks, and ``deadline``, the AND of the two heads' deadline bits. The certificates are not in the proof block: the
verifier checks them on their public fields (`TransferBlocks.certified`).

In the proof block's Program each head check is one gate that names the Definition and its circuit digest
(`head_check.gate`, `Gate[NetworkHeadCheck_v2{L,T,COUNTS}@<digest>]_v2`), because the commitment gadgets' Boolean bodies
encode to about 140 MB and the proof system encodes the Program on every registration. The digest is core's
`program_digest` of the Definition's Program (`head_check.circuit_digest`), the one a Flock statement names a template
by, pinned in `head_check.circuit-check.json` and held to the Definition by circuit-check. So the Program's digest binds
the head check's circuit, and the spec, the stream and each registration bind to it: a proof of another circuit with the
same interface names another Program, which the verifier refuses (`TransferBlocks.proved`), and a proof's circuit
puts the Definition where its gate stands only once the digest is the gate's (`head_check.expand`). The digest is read
from the pin, never computed, since tracing the gadgets takes minutes: a size class no sidecar pins has a gate that
names none (`@unpinned`), and no proof of its proof block is accepted, while its verdict of record in the clear is
unchanged.

N3, the content check, a prototype evaluated in the clear and in no proof-system Spec, is retired, superseded by the step-indexed scan (`ingress_scan`, `egress`), which `verity/protocols/resource_accounting/network/prover/scan_zk.py`
proves in zero knowledge, outside this protocol.

**The proof.** The prover proves the sampled proof block as one Flock statement of one instance: rows ``pub`` (the proof
block's public parameters) and ``wit`` (each end's head, its leaf's salt, its two siblings and the attester's salt),
each the little-endian concatenation of its parameters' bits in whole u16 words (`TransferBlocks.layout`,
`TransferBlocks.row`), and the output row ``o0``, one u16 word holding the two output bits (`OUT_BITS`); each row committed
under a hiding salt. Its circuit is `TransferBlocks.proof_block`, the registered proof block with each gate the Definition
its pinned digest names, bound to the registered partition and Program (`verity/protocols/resource_accounting/network/prover/n2_zk.py` proves it
behind the ZK server).

Its verifier is `TransferBlocks.verify`, which decides on the delivered proof (`Delivered`) and public inputs
alone. It accepts when
- the verifier's own run of Flock's Lean verifier (`proofs.verify`, `verify --zk`) accepts every session on the circuit
  file the auditor committed, the sessions prove this link's registered proof block (`TransferBlocks.proved`: every gate
  names a pinned circuit digest and every session names the registered Program) and the sampled proof block;
- the proof's ``pub`` row opens, under the salt delivered, to the verifier's own copy (`public_arguments`: the registered
  root, the execution phase's public fields and both certificates' signed digests), and its ``o0`` row to the output
  word delivered, whose verdict and deadline bits are both 1;
- the certificates' check accepts (`certified`: each verifies under the network attester's public key in its certifier
  session over its public fields and its signed digest, and names the registered execution phase).

What the verifier receives: the registration (the salted root), the sample, the two certificates' public fields and
signatures, from the attester, which is the auditor's device; and from the prover the statement's
public file (each row's hiding commitment), its session files, the circuit file (held to the committed digest), the
salts of the ``pub`` and ``o0`` rows and the output word. Never the opening nor the ``wit`` row's salt. The certificates
are not a public item of the Spec: the service's public kinds (structure, outputs, coins, hiding commitments) have none
for a trusted device's signature over a hiding commitment, so they reach the verifier from the attester beside the
service, in the clear. That is a provisional exception to the four public kinds, not a fifth kind.

`TransferBlocks.clear` is the in-the-clear check on the opening, for a debug comparison only (`verify(opening=…)`, off by
default, which adds its accept bit and decides nothing) and the tests' reference: the opening reproduces the registered
root, each end's head check accepts (`head_check.reference`), the opening's certificates hold (`certified`) and both
heads met their deadline. The records' cells are the scan's (`network_attester.audit`
reads them in the clear), and the attester's host audit doesn't read them. `TransferBlocks.unproved_outcome` decides a
run that proves nothing (a rehearsal, the simulator) on it, so its verdict says ``"decided_by": "clear"`` and is never
zero-knowledge.

**The link** (`TransferBlocks(network_attester=NetworkAttesterProfile(…))`) is the network attester's (`network_attester.NetworkAttester`) own two records of
each execution phase: its responses on egress and its requests on ingress, on its one grid, certified under its
schedule's parameters digest (`network_attester.NetworkAttesterParams.digest`, which its deployment publishes) rather than a certifier's
(`certificate.parameters_digest`). The network attester is the auditor's device, and its code puts every bucket's
cells, r for each live connection and the rest padding (`network_attester.audit` reads them in the clear, the scan's proof blocks
in zero knowledge), so its counts are its schedule's: no occupancy profile holds them, the period's schedule terms
charge them (log2 #S_s and Σ_c log2(cap_c + 1)), and the network attester runs no shaper clock, so a window declares
no sync and the charge is 0. Each head check holds its end's head to the network attester's grid and digest and to its
signed digest. N2 takes no frames: the frames, their content and their order, are the scan's, and N2 says nothing of
them. A `TransferBlocks` without a `NetworkAttesterProfile` raises `ValueError`.

**The verdict of record is the Lean's**: `NetworkCheck.TransferBlocks.verdict`
(`verity/protocols/resource_accounting/network/verifier/lean`, `network-records verdicts`' ``link-window``, `lean_verdict`), switched by the one
line `OF_RECORD`. `verify` runs it and this module's `python_verdict` on every proof, records
whose verdict it is (`by`) and the other's accept bit (`reference`), and rejects an execution phase where they differ or
the Lean refuses the case. The Lean reads the network attester's parameters digest (`lean_case`'s ``network_attester``) for
both ends' grids and no sync, as `grid` and `python_verdict` do; its other kinds of link (`Kind.plain`, `Kind.served`)
are cases this module never sends. The gap list, what
`python_verdict` reads that the
Lean takes as an input rather than checks:
- the network attester's parameters digest (`NetworkAttesterProfile.digest`, which the network attester's deployment publishes
  as its schedule's, `network_attester.NetworkAttesterParams.digest`), which neither derives;
- what both take from the proof system and the lowering: the findings of the verifier's own run of Flock's Lean
  verifier (`proofs.verify`), the delivered sessions' SHA-512s, the registration and the sample, and the registered
  Program's and partition's digests and the gate ids (`P.partition`, `head_check.gate`), which the Lean holds the
  findings and pins to but doesn't derive.
The Lean reads an `LmsKey` only, and refuses (so `verify` rejects) any other key. `tie_service.IngressTie.verify`, the
ingress tie's verifier, is not ported and stays Python. Every proving run of N2 recorded before the switch replayed
through both to the verdict it recorded, and the properties of the Lean's verdict are
`security/Properties/Certifier/Check/TransferBlocks.lean`.

What that establishes, and what it doesn't: the outcome is accept exactly when the Lean verifier accepts a proof of the
registered proof block whose public row is the verifier's own and whose output word is 0b11 and the certificates verify;
the outcome record binds the registered root, the sample and the delivered sessions. The
verifier reads no witness. The proof opens the head leaves (2, 3) only, so the
record leaves (0, 1), which hold each certified record whole, are committed but read by nothing in the verdict, and
the certificates' signatures are checked
on the host from their public fields, not in the proof. Each registration is one execution phase, checked alone. The
frames, their content and their order within the counts, are not held: an accepted outcome says nothing of them.
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.circuits import boolean as BL
from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.commitments import hm96
from verity.commitments.identity import identity_digest_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.protocols.resource_accounting.network.verifier import head_check as HC, lean_audit as LA
from verity.protocols.resource_accounting.network.verifier.certificate import Certificate, CertifiedRecord, Direction, LmsKey, PublicKey
from verity.protocols.resource_accounting.network.verifier.grid import GridParams
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync, SyncSet
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import audit as OA, partition as P, proofs as PF, docket as R
from verity.protocols.proofs.spot.verifier.one_stage.proofs import Finding
from verity.protocols.proofs.spot.verifier.one_stage.verdict import Verdict

#: Whose verdict on an execution phase's proof is the record's: the Lean's (``"lean"``, `NetworkCheck.TransferBlocks`) or
#: this module's (``"python"``). Both run on every proof either way. Setting it to ``"python"`` is the one line that
#: switches it back.
OF_RECORD = "lean"

SPEC = "verity-network-certifier/service-spec/v0"
SESSION = "verity-network-certifier/session/v0"
STATEMENT = "verity-network-certifier/statement/v0"
VERDICT = "verity-network-certifier/verdict/v0"
NETWORK_VERDICT = "verity-network-certifier/network-verdict/v0"
PERIOD_DRAW = "verity-network-certifier/period-draw/v1"
#: the sampled network verdict's refusal by name: the windows registered and proved are not the period's sample of record
#: (`lean_draw` on the coins the period sample read), or that sample can't be made
PERIOD_DRAWN = "period-draw"
VALUE = "certificates"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "network-audit"
_INST = hm96.SHA512


_SIGNED = (("egress", HC.DIGEST), ("ingress", HC.DIGEST))
LEAVES = 4
#: the leaf of the head of the record at leaf i (0 egress, 1 ingress) is HEADS + i
HEADS = HC.HEADS
_ENDS = ("egress", "ingress")
#: the proof block's output, two bits in this order (`Check.public_outputs`)
OUTPUTS = ("verdict", "deadline")
#: the statement's rows: ``pub`` the proof block's public parameters, ``wit`` its witness (`TransferBlocks.layout`), then
#: its one output row ``o0``, the two output bits; each instance's commitments are in this order in a statement's public
#: file
ROWS = ("pub", "wit")
PORTS = ROWS + ("o0",)
#: the output row's length: Flock commits an output port as whole u16 words, so ``o0`` is one word, its bit 0 the
#: verdict and bit 1 the deadline, its other bits 0
OUT_BITS = 16
#: a head check's witness parameters; the rest of the proof block's parameters are public
_WITNESS = ("head", "salt", "path", "salt_c")
#: a head check's parameters the proof block fixes rather than takes: each end's direction is its own (egress 0, ingress
#: 1)
_FIXED = ("signed", "direction")


def _heads(S_) -> tuple[tuple[str, Any], ...]:
    """Each end's head check's parameters but its signed digest, which is the proof block's `egress` or `ingress`, and
    its direction, which the proof block fixes."""
    return tuple((f"{end}_{name}", t) for end, key in zip(_ENDS, (S_.EGRESS, S_.INGRESS))
                 for name, t in HC.gate(*key).params if name not in _FIXED)


@lru_cache(maxsize=None)
def _head_expanded(L: int, T: int, counts: bool):
    return HC.expand(HC.gate(L, T, counts), HC.unit(L, T, counts))


def _checks(B, S_, args, check=HC.gate) -> tuple[list[Coll], list[Coll], list[Any]]:
    """Each end's head check (`check`'s for its size class) over its end's signed digest (the proof block's `egress` or
    `ingress`) and its own direction, a constant; each end's head counts and its deadline byte's low bit, as the head
    check reads the head."""
    K = HC._Consts(B)
    oks, counts, deadline, rest = [], [], [], list(args[2:])
    for d, key in enumerate((S_.EGRESS, S_.INGRESS)):
        g = check(*key)
        names = [name for name, _ in g.signature()[0]]
        ends, rest = rest[:len(names) - len(_FIXED)], rest[len(names) - len(_FIXED):]
        ends.insert(names.index("direction"), HC._coll(Array(1, BL.BIT), K.bits(d, 1)))
        ends.insert(names.index("signed"), args[d])
        oks.append(B.call(g, *ends))
        sh = HC.Shape(*key[:2])
        head = HC._refs(ends[names.index("head")])
        counts.append(HC._coll(Array(sh.T, Array(4, HC.BYTE)), head[8 * sh.CNT:8 * sh.ROOT]))
        deadline.append(head[8 * sh.DL])
    return oks, counts, deadline


def _all(B, oks: list[Coll]) -> Coll:
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BL.BIT), bits))


def _outputs(B, oks: list[Coll], deadline: list[Any]) -> Coll:
    """The proof block's two output bits: the AND of its checks, then the AND of both heads' deadline bits."""
    settled = B.call(BL.and_gate(2), HC._coll(Array(2, BL.BIT), deadline))
    return Coll(_OUT, concat([_all(B, oks).refs, settled.refs]))


_OUT = Array(len(OUTPUTS), BL.BIT)


def _unit(check) -> CompositeDefinition:
    """N2's unit, each head check called as `check` for its size class."""

    def body(B, S_, *args):
        oks, _, deadline = _checks(B, S_, args, check)
        return _outputs(B, oks, deadline)

    return CompositeDefinition(
        "NetworkTransferBlocks", 5, ("EGRESS", "INGRESS"), lambda S_: (_SIGNED + _heads(S_), _OUT), body,
        doc="the proof block of network accounting: one execution phase, both directions, over the two certificates' signed "
            "digests; EGRESS and INGRESS are each end's head check's (L, T, COUNTS). Its two output bits are the "
            "verdict, both head checks, and the deadline, both heads' deadline bits", register=False)


_UNIT = _unit(HC.gate)
#: the same unit with each gate replaced by the Definition it names, as a proof's circuit lowers it
_PROOF_BLOCK = _unit(_head_expanded)


def unit(egress: tuple[int, int, int], ingress: tuple[int, int, int], *, proof: bool = False):
    """N2's proof block for heads of the size classes `egress` and `ingress`, each (L, T, COUNTS) (`head_check.unit`).
    With `proof`, the unit a proof's circuit lowers: the same id and signature, so the template the registered query
    names, with each gate the Definition it names once that Definition's circuit digest is the gate's pin
    (`head_check.expand`, which traces it: minutes a head check, and a `ValueError` for an unpinned size class)."""
    return bind(_PROOF_BLOCK if proof else _UNIT, EGRESS=tuple(egress), INGRESS=tuple(ingress))


PUBLIC = (
    S.Public("population", "structure", "one proof unit per registration: the link-window"),
    S.Public("window", "structure", "provenance: the link-window's index, held to the registration's order"),
    S.Public(f"leaves:{VALUE}", "structure", "four leaves, the egress and ingress certified records and their heads"),
    S.Public(f"domain:{VALUE}", "structure", "the served domain, which binds the check, the link and the window"),
    S.Public(VALUE, "commitment", "the link-window's certified records under one salted root"),
    S.Public("parameters", "structure", "the parameters digests (W1) the certificates sign, fixed before the run"),
    S.Public("verdict", "output", "the link-window's verdict, accept or reject; a violation's kind stays hidden"),
    S.Public("deadline", "output", "the certifier settled both records by the verdict deadline"),
    S.Public("warden", "structure", "the active warden's schedule (`warden.WardenParams`, the digest both ends' heads "
             "bind) and the link's windows, fixed before the run: the device's code puts every bucket's cells (r for "
             "each live connection), so its counts are the schedule's, which the period's schedule terms charge"),
)


def encode(record: CertifiedRecord) -> bytes:
    """The leaf's bytes: the record head (`RecordHead.encode`), the attester's salt (64 bytes), then the certificate,
    every field fixed-width or length-prefixed: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖
    u64be window ‖ signed digest (64) ‖ u32be len(signature) ‖ signature."""
    c = record.certificate
    link = c.link.encode("utf-8")
    return b"".join([record.head.encode(), record.salt, struct.pack(">H", len(c.key_id)), c.key_id,
                     struct.pack(">H", len(link)), link, struct.pack(">BQ", c.direction.code, c.window), c.digest,
                     struct.pack(">I", len(c.signature)), c.signature])


def rows(records: Sequence[CertifiedRecord]) -> list[bytes]:
    """The four leaves' bytes, in leaf order: each record's `encode`, then each record's head alone."""
    return [encode(r) for r in records] + [r.head.encode() for r in records]


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def _root_of(record: Any) -> Any:
    """An execution phase record's one salted root, None when it holds no such root."""
    roots = record.get("roots") if isinstance(record, Mapping) else None
    if isinstance(roots, list) and len(roots) == 1 and isinstance(roots[0], Mapping):
        return roots[0].get("root")
    return None


def lean_draw(k: int, n: int, read: bytes) -> dict[str, Any]:
    """The period's sample of record: Flock's Lean verifier (`proofs.executable`, which verifies the
    windows' proofs) sampling `subset:k` of n on `read`, the coin bytes the period sample read, in order
    (``verity-verify-proof draw --population n --k k --stream-file``, `Flock.Draw.sampleOn`, which `select_draw`'s sampler is
    byte for byte: `test_lean_verifier.test_unit_sample_samplers_are_the_exact_reduction` and `test_coins`); its ``DRAW``
    object, or ``{"refused": why}`` where the bytes run out or the Lean refuses or can't run."""
    try:
        with tempfile.TemporaryDirectory(prefix="verity-period-draw-") as d:
            (Path(d) / "coins.bin").write_bytes(read)
            p = subprocess.run([str(PF.executable()), "draw", "--population", str(n), "--k", str(k), "--stream-file",
                                str(Path(d) / "coins.bin")], capture_output=True, text=True)
    except (OSError, PF.LeanError) as e:
        return {"refused": f"verity-verify-proof draw can't run: {e}"}
    lines = [x[len("DRAW "):] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"refused": f"verity-verify-proof draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}"}
    try:
        return json.loads(lines[0])
    except ValueError:
        return {"refused": f"verity-verify-proof draw answered a DRAW line that is not JSON: {lines[0][:300]}"}


@dataclass(frozen=True)
class Delivered:
    """An execution phase's proof as the verifier receives it: the statement's proof files (`proofs.Proof`, its circuit
    file the auditor's own lowering of the proof block), and for the sampled proof block the salts of its public row and
    its output row (``pub``, ``o0``) and its output word, which the verifier opens against the public file's commitments
    to those rows. Never the witness row's salt."""

    proof: PF.Proof
    salts: Mapping[str, bytes]
    output: int

    def sessions(self) -> list[str]:
        """The delivered sessions' SHA-512s, their `session.json`s', in order."""
        return [PF.file_sha512(Path(s) / "session.json") for s in self.proof.sessions]


#: a row commitment's bytes in a statement's public file, ``b || c`` (`hm96.SHA512.commit_string`)
BC_BYTES = 128


def _commitments(public: Path, unit: int) -> dict[str, bytes] | None:
    """The row commitments a statement's public file holds for its one instance whose proof block index is `unit`, by
    port (`PORTS`, in their order after the header line): instance-major, or in a shared-row file (header
    ``shared_rows``, `verity_flock.circuit.write`) each port's table port-major, then every instance's refs into them
    (u32 LE, instance-major, the ports in order); None when no one instance is, or the file is short."""
    try:
        with open(public, "rb") as f:
            head = json.loads(f.readline())
            indices, shared = list(head["units"]["indices"]), head.get("shared_rows")
            if indices.count(unit) != 1:
                return None
            i, k = indices.index(unit), len(PORTS)
            if shared is None:
                f.seek(f.tell() + i * k * BC_BYTES)
                data = f.read(k * BC_BYTES)
            else:
                counts = [int(shared[p]) for p in PORTS]
                if set(shared) != set(PORTS) or min(counts) < 1:
                    return None
                body = f.read(sum(counts) * BC_BYTES + len(indices) * k * 4)
                if len(body) != sum(counts) * BC_BYTES + len(indices) * k * 4:
                    return None
                refs = body[sum(counts) * BC_BYTES + i * k * 4:][:k * 4]
                data, base = b"", 0
                for j, n in enumerate(counts):
                    r = int.from_bytes(refs[4 * j:4 * j + 4], "little")
                    if r >= n:
                        return None
                    data += body[(base + r) * BC_BYTES:(base + r + 1) * BC_BYTES]
                    base += n
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if len(data) != k * BC_BYTES:
        return None
    return {p: data[j * BC_BYTES:(j + 1) * BC_BYTES] for j, p in enumerate(PORTS)}


def _opens(data: bytes, bits: int, salt: Any, bc: bytes) -> bool:
    """Whether `bc` commits to the `bits`-bit row `data` under `salt`: hm96 over its sha512/row/v2 digest."""
    if not isinstance(salt, bytes) or len(salt) != _INST.salt_bytes:
        return False
    return _INST.commit_string(_INST.default_key, sha512_row_v2_digest(data, bits), salt) == bc


def output_row(word: int) -> bytes:
    """The output row's bytes for the output word `word` (`OUT_BITS`)."""
    return word.to_bytes(OUT_BITS // 8, "little")


def _opened(delivered: Delivered, unit: int, public: bytes | None, bits: int) -> int | None:
    """The proof block's output word, once the proof's public file holds the proof block's public row as `public` (the
    verifier's own copy, `bits` long) and its output row as the word delivered, each under the salt delivered; else
    None."""
    bcs, word, salts = _commitments(delivered.proof.public, unit), delivered.output, delivered.salts
    if public is None or bcs is None or type(word) is not int or not 0 <= word < 1 << len(OUTPUTS):
        return None
    if not (_opens(public, bits, salts.get("pub"), bcs["pub"])
            and _opens(output_row(word), OUT_BITS, salts.get("o0"), bcs["o0"])):
        return None
    return word


@dataclass(frozen=True)
class Hidden:
    """The ZK server's side of one execution phase's commitment, all witness: the egress and the ingress certified records,
    in leaf order, and one salt per leaf (`rows`'s order: the two records', then the two heads')."""

    window: int
    records: tuple[CertifiedRecord, CertifiedRecord]
    salts: tuple[bytes, bytes, bytes, bytes]


@dataclass(frozen=True)
class NetworkAttesterProfile:
    """A network attester link's profile (`TransferBlocks.network_attester`, the module docstring's "A network attester link"): the
    link's windows and the network attester's parameters digest (`network_attester.NetworkAttesterParams.digest`), which both ends' heads
    bind, both fixed before the run."""

    windows: int
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError(f"a network attester link has one or more windows, not {self.windows!r}")
        if type(self.digest) is not bytes or len(self.digest) != HC.DIGEST_BYTES:
            raise ValueError(f"a network certifier's parameters digest is {HC.DIGEST_BYTES} bytes")


@dataclass(frozen=True)
class TransferBlocks:
    """Network accounting's protocol on one link, every field public: the link, its egress and ingress grids, Σ_sync, the
    network attester's public key, the run that names the served domains, the attester session's id (the 32 bytes the
    challenger sampled for the attester's session, which every certificate signs), the proof deadline in seconds, and
    the network attester's profile (`network_attester`: its records, its key and session the attester's here, its counts its
    schedule's). It declares no sync, so `sync_set` is read only as the Lean case's ``sync_set``."""

    link: str
    params: GridParams
    sync_set: SyncSet
    key: PublicKey
    run: str
    certifier_session: bytes
    ingress: GridParams | None = None
    proof_deadline: float | None = None
    network_attester: NetworkAttesterProfile | None = None

    def __post_init__(self) -> None:
        if self.network_attester is None:
            raise ValueError("a link is the network attester's: TransferBlocks(network_attester=NetworkAttesterProfile(windows, digest))")
        if self.ingress_params != self.params:
            raise ValueError("a network certifier link's two ends are on the network certifier's one grid")

    @property
    def ingress_params(self) -> GridParams:
        return self.ingress or self.params

    def head_key(self, direction: int) -> tuple[int, int, int]:
        """The end's head check's size class (L, T, COUNTS): the link name's UTF-8 length, the end's buckets, and 0, no
        occupancy profile."""
        g = (self.params, self.ingress_params)[direction]
        return len(self.link.encode("utf-8")), g.buckets, 0

    def grid(self, direction: int) -> bytes:
        """The end's declared grid as its head encodes it (`head_check.grid_bytes`): the network attester's grid under
        its parameters digest (`NetworkAttesterProfile.digest`)."""
        return HC.grid_bytes((self.params, self.ingress_params)[direction], self.network_attester.digest)

    @property
    def windows(self) -> int:
        """The link's windows, which `network_verdict` takes by default: its `NetworkAttesterProfile`'s."""
        return self.network_attester.windows

    @cached_property
    def unit(self):
        """N2's proof block for this link (`unit`): its heads' size classes are the link's."""
        return unit(self.head_key(0), self.head_key(1))

    @cached_property
    def proof_block(self):
        """The proof block a proof's circuit lowers: `unit` with each gate the Definition it names (`unit(proof=True)`),
        the template of the registered query under the same id. The prover's and the auditor's lowering, never the
        registered Program, which holds the gates."""
        return unit(self.head_key(0), self.head_key(1), proof=True)

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    @cached_property
    def _program(self) -> Program:
        return P.population_program(self.unit, 1)

    @cached_property
    def _partition(self) -> dict[str, Any]:
        return P.partition(self._program, self.query)

    @property
    def partition(self) -> dict[str, Any]:
        """The registered partition object, which a statement proving the proof block binds (its digest and its
        ``program``)."""
        return self._partition

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({
            "format": SPEC, "link": self.link, "run": self.run, "key": self.key.key_id.hex(),
            "certifier_session": self.certifier_session.hex(), "charge": 0.0,
            "parameters": {"egress": self.grid(0)[-HC.DIGEST_BYTES:].hex(),
                           "ingress": self.grid(1)[-HC.DIGEST_BYTES:].hex()},
            "occupancy": [None, None], "warden": [self.network_attester.windows, self.network_attester.digest.hex()]})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query, public_outputs=OUTPUTS,
                        proof_deadline=self.proof_deadline)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, PUBLIC, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        """The challenger's stream for this link (`S.Stream`'s budget, delta, challenger, coins and clock)."""
        return S.Stream(self.spec, self.unit, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """One-stage's window domain (`registration.window_domain`) for the execution phase: run
        `<run>/<link>/<window>`."""
        binding = identity_digest_sha512(R.WINDOW_DOMAIN, {
            "program": self._partition["program"], "partition": P.digest(self._partition), "port": VALUE,
            "schema": SCHEMA, "leaves": LEAVES, "run": f"{self.run}/{self.link}/{window}"})
        return CommitmentDomain(binding, -1, RangeIndexedDomain(0, LEAVES), hash="sha512")

    def ports(self, window: int) -> dict[str, dict[str, Any]]:
        """The challenger's own copy of the execution phase's port: four leaves, under its domain."""
        return {VALUE: {"leaves": LEAVES, "domain": self.domain(window).domain_id.hex()}}

    @staticmethod
    def provenance(window: int) -> dict[str, Any]:
        return {"kind": "served", "window": window}

    # the ZK server

    def commit(self, window: int, egress: CertifiedRecord, ingress: CertifiedRecord,
               salts: Sequence[bytes] | None = None) -> Hidden:
        """The execution phase's opening. `salts`: four distinct salts, one per leaf, drawn once (default: fresh from the
        operating system)."""
        salts = tuple(salts) if salts is not None else tuple(_INST.fresh_salt() for _ in range(LEAVES))
        if len(salts) != LEAVES or len(set(salts)) != LEAVES or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"{LEAVES} distinct {_INST.salt_bytes}-byte salts")
        return Hidden(window, (egress, ingress), tuple(bytes(s) for s in salts))  # type: ignore[arg-type]

    def leaves(self, opening: Hidden) -> list[bytes]:
        """The salted leaves, which hide the records and their heads: `hm96-sha512/row/v2` over each of `rows`."""
        dom, key = self.domain(opening.window), _INST.default_key
        out = []
        for i, (row, salt) in enumerate(zip(rows(opening.records), opening.salts, strict=True)):
            bc = _INST.commit_string(key, sha512_row_v2_digest(row, 8 * len(row)), salt)
            out.append(row_leaf(dom, i, _INST.tree_leaf(key, bc), SCHEMA))
        return out

    def root(self, opening: Hidden) -> str:
        return tree_levels(self.domain(opening.window), self.leaves(opening))[-1][0].hex()

    def registration(self, opening: Hidden, previous: str | None = None) -> dict[str, Any]:
        """One-stage's record of the execution phase: its salted root and nothing of the certificates. `previous`: the
        digest of the stream's previous registration."""
        w = opening.window
        roots = [{"port": VALUE, "root": self.root(opening), **self.ports(w)[VALUE]}]
        return R.record(self._program, self.query, {"law": "subset", "k": 1}, S.SCHEME, roots, self.provenance(w),
                        previous=previous)

    def session(self, opening: Hidden, sample: S.Draw) -> str:
        """The delivered session's SHA-512: the registration, the sample and the salted leaves the opening reproduces."""
        return _sha512({"format": SESSION, "registration": sample.docket.digest, "draw": _sha512(dict(sample.draw)),
                        "leaves": [x.hex() for x in self.leaves(opening)]})

    # the challenger

    def register(self, stream: S.Stream, record: Mapping[str, Any],
                 drawn: Mapping[str, Any] | None = None) -> S.WindowCommitmentRecord:
        """`stream.register` with the execution phase's own ports, after holding the record's `window` provenance to exactly
        `{"kind": "served", "window": w}`: the service declares the provenance one public item and doesn't read it,
        and anything else there would reach the auditor. w is the registration's order in the stream, or with `drawn`,
        a period sample (`period_draw`) over this stream, the order-th sampled window, whose record's root must be the
        one the period committed. Raises `Refused` (`window`, `period`) otherwise."""
        order = len(stream.log)
        if drawn is None:
            window = order
        else:
            units = self._drawn(stream, len(drawn.get("roots") or ()) if isinstance(drawn, Mapping) else 0, drawn)
            if order >= len(units):
                raise S.Refused(Verdict(False, ("period",), {"period": {"drawn": units, "registered": order}}))
            window = units[order]
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        if drawn is not None and _root_of(record) != drawn["roots"][window]:
            raise S.Refused(Verdict(False, ("period",), {"period": {"window": window, "root": "not the committed one"}}))
        return stream.register(record, self.ports(window))

    # the prover's side: the proof block's arguments and outputs on the opening

    def head_checks(self, opening: Hidden, root: bytes) -> list[HC.Instance | None]:
        """Each end's head check on the opening under `root`, the registered root: the execution phase's public inputs with
        the end's grid and its certificate's signed digest, and as witness the head, its salt, its two siblings and the
        attester's salt for the record. None where the proof block takes no instance: a head whose length isn't the
        proof block's."""
        w, dom = opening.window, self.domain(opening.window)
        levels = tree_levels(dom, self.leaves(opening))
        out: list[HC.Instance | None] = []
        for d, record in enumerate(opening.records):
            L, T, _ = self.head_key(d)
            head, i = record.head.encode(), HEADS + d
            if len(head) != HC.Shape(L, T).HL:
                out.append(None)
                continue
            out.append(HC.Instance(root, dom.domain_id, self.link.encode("utf-8"), d, w, self.grid(d),
                                   None, record.certificate.digest, head,
                                   opening.salts[i], tuple(level[(i >> k) ^ 1] for k, level in enumerate(levels[:-1])),
                                   record.salt, T))
        return out

    @cached_property
    def layout(self) -> dict[str, tuple[str, ...]]:
        """The statement's two input rows (`ROWS`), each its parameters in the proof block's signature's order: ``pub``
        the public ones, ``wit`` each end's head, its leaf's salt, its two siblings and the attester's salt. A row is
        the little-endian concatenation of its parameters' bits, in whole u16 words, which each row fills exactly: every
        parameter is a multiple of 16 bits but each end's head, of 157 + L + 4T bytes, and the two heads together are.
        Raises `ValueError` for a proof block whose rows aren't whole words."""
        params, _ = self.unit.signature()
        out = {r: tuple(n for n, _ in params if ("_" in n and n.split("_", 1)[1] in _WITNESS) == (r == "wit"))
               for r in ROWS}
        for r, names in out.items():
            bits = sum(t.leaves for n, t in params if n in names)
            if bits % 16:
                raise ValueError(f"the unit's {r} row is {bits} bits, not whole u16 words")
        return out

    def bits(self, row: str) -> int:
        """A row's length in bits: `layout`'s, or the output row's (``o0``, `OUT_BITS`)."""
        if row == "o0":
            return OUT_BITS
        params, _ = self.unit.signature()
        return sum(t.leaves for n, t in params if n in self.layout[row])

    def _arguments(self, insts: Sequence[HC.Instance]) -> dict[str, list[int]]:
        out = {end: HC._digest_bits(inst.signed) for end, inst in zip(_ENDS, insts)}
        for end, inst in zip(_ENDS, insts):
            for (name, _), bits in zip(inst.unit.signature()[0], HC.arguments(inst)):
                if name not in _FIXED:
                    out[f"{end}_{name}"] = bits
        return out

    def arguments(self, opening: Hidden, root: bytes, syncs: Sequence[ClockSync] = ()) -> dict[str, list[int]] | None:
        """The unit's arguments on the opening under `root`, the registered root, by parameter, each its leaves' bits
        (each end's `head_check.arguments`): the prover's side. None where the unit takes no instance (`head_checks`).
        `syncs` is unread: the link declares none."""
        insts = self.head_checks(opening, root)
        if None in insts:
            return None
        return self._arguments(insts)

    def public_arguments(self, window: int, root: bytes, certificates: Sequence[Certificate],
                         syncs: Sequence[ClockSync] = ()) -> dict[str, list[int]] | None:
        """The unit's public arguments (`layout`'s ``pub``) as the verifier derives them from public fields alone: the
        registered `root`, the execution phase's domain, the link, the window, each end's grid and each end's certificate's
        signed digest. None where the unit takes no instance: not two certificates, or a signed digest of another
        length. `syncs` is unread: the link declares none."""
        if len(certificates) != 2:
            return None
        insts = []
        for d, c in enumerate(certificates):
            L, T, _ = self.head_key(d)
            if len(getattr(c, "digest", b"")) != HC.DIGEST_BYTES:
                return None
            insts.append(HC.Instance(root, self.domain(window).domain_id, self.link.encode("utf-8"), d, window,
                                     self.grid(d), None, c.digest,
                                     bytes(HC.Shape(L, T).HL), bytes(_INST.salt_bytes),
                                     (bytes(HC.DIGEST_BYTES), bytes(HC.DIGEST_BYTES)),
                                     bytes(64), T))
        args = self._arguments(insts)
        return {n: args[n] for n in self.layout["pub"]}

    def row(self, args: Mapping[str, Sequence[int]], row: str) -> bytes:
        """A row's bytes (`layout`): its parameters' bits in order, least significant first."""
        bits = [b for n in self.layout[row] for b in args[n]]
        return bytes(sum(b << j for j, b in enumerate(bits[k:k + 8])) for k in range(0, len(bits), 8))

    def outputs(self, opening: Hidden, root: bytes, syncs: Sequence[ClockSync] = ()) -> int:
        """The unit's output word on the opening under `root` by each gate's reference (`head_check.reference`), which
        the tests hold to the circuits: bit 0 the verdict, bit 1 the deadline. What the prover claims, and the debug
        comparison's. `syncs` is unread: the link declares none."""
        verdict = all(i is not None and HC.reference(i) for i in self.head_checks(opening, root))
        return int(verdict) | int(all(r.head.deadline for r in opening.records)) << 1

    def _registered(self, opening: Hidden, sample: S.Draw) -> str:
        reg = sample.docket.record
        if opening.window != reg["window"]["window"] or len(opening.records) != 2:
            raise ValueError("the opening is not the registered execution phase's")
        return reg["roots"][0]["root"]

    def gates(self) -> list[PrimitiveDefinition]:
        """The unit's gates: each end's head check (`head_check.gate`)."""
        return [HC.gate(*self.head_key(d)) for d in (0, 1)]

    def proved(self, findings: Sequence[Finding]) -> bool:
        """Whether `findings`, the verifier's run over a delivered proof of the proof block (`proofs.verify`),
        prove this link's proof block: every gate of it names its Definition's pinned circuit digest
        (`head_check.pinned`), and one or more sessions each accepted and named the link's registered Program (the
        partition object's `program`), which binds those digests. A proof of another circuit with the same interface
        names another Program, and a proof block with a size class no sidecar pins has no proof."""
        want = self._partition["program"]
        return all(HC.pinned(g) for g in self.gates()) and bool(findings) \
            and all(f.accepted and f.program == want for f in findings)

    @cached_property
    def program(self) -> bytes:
        """The registered Program's descriptor, the verifier's own copy, whose SHA-512 the partition object names."""
        return canonical_json(encode_program(self._program))

    def certified(self, window: int, certificates: Sequence[Certificate]) -> bool:
        """The certificates' check, on the host from their public fields alone: two certificates, each the attester's
        and verifying in its session over its fields and its signed digest (`Certificate.verify`), and each naming this
        link, its end (egress first) and the window. Each signed digest is the one its end's head check reads
        (`public_arguments`), which a proof holds to the head it opens."""
        return len(certificates) == 2 and all(
            isinstance(c, Certificate) and (c.link, c.direction.code, c.window) == (self.link, d, window)
            and c.verify(self.key, self.certifier_session) for d, c in enumerate(certificates))

    @staticmethod
    def _decided(sample: S.Draw, sessions: Sequence[str], accepted: bool, public: dict[str, Any]) -> dict[str, Any]:
        """A verdict's accept bit and public outputs, with the digest binding them to the registration and sessions."""
        digest = _sha512({"format": VERDICT, "registration": sample.docket.digest, "sessions": list(sessions),
                          "accepted": accepted, "public": public})
        return {"accepted": accepted, "digest": digest, "public": public}

    def python_verdict(self, sample: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                       delivered: Delivered, findings: Sequence[Finding] | None,
                       sessions: Sequence[str]) -> dict[str, Any]:
        """This module's verdict on the execution phase (`verify`'s conditions), on the findings of the verifier's own run
        of Flock's Lean verifier (None where `proofs.verify` refused): its accept bit, digest and public outputs."""
        reg = sample.docket.record
        w, root = reg["window"]["window"], bytes.fromhex(reg["roots"][0]["root"])
        proved = findings is not None and self.proved(findings) \
            and all(list(f.units) == list(sample.units) for f in findings)
        args = self.public_arguments(w, root, certificates, syncs)
        opened = _opened(delivered, sample.units[0], self.row(args, "pub") if args is not None else None,
                         self.bits("pub")) if proved else None
        word = opened if opened is not None else 0
        accepted = proved and opened is not None and word == 0b11 and self.certified(w, certificates)
        return self._decided(sample, sessions, accepted, {"verdict": int(accepted), "deadline": word >> 1 & 1})

    def lean_case(self, sample: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                  delivered: Delivered, findings: Sequence[Finding] | None,
                  sessions: Sequence[str]) -> dict[str, Any] | None:
        """`verify`'s inputs as `NetworkCheck.TransferBlocks.caseOf` reads them: the verifier's own structure (the Program's
        and partition's digests, the gate ids), the registration, the sample, the findings of its own run of Flock's
        Lean verifier (None where `proofs.verify` refused), the delivered public file, salts, output word and session
        SHA-512s, the certificates (each null unless a `Certificate` of well-typed fields) and each one's signed digest
        as `public_arguments` reads it, the declared syncs, Σ_sync, the key and session, each end's grid with no
        profile, and the network attester's parameters digest. None for a key the Lean doesn't read (not an `LmsKey`)."""
        if not isinstance(self.key, LmsKey):
            return None
        reg, p = sample.docket.record, self.key.params
        w = reg["window"]["window"]

        def cert(c: Any) -> dict[str, Any] | None:
            if not (isinstance(c, Certificate) and all(type(x) is bytes for x in (c.key_id, c.digest, c.signature))
                    and type(c.link) is str and isinstance(c.direction, Direction) and type(c.window) is int
                    and c.window >= 0):
                return None
            return {"key_id": c.key_id.hex(), "link": c.link, "direction": c.direction.code, "window": c.window,
                    "digest": c.digest.hex(), "signature": c.signature.hex()}

        def hexed(b: Any) -> str | None:
            return b.hex() if type(b) is bytes else None

        def digest(c: Any) -> str:
            d = getattr(c, "digest", b"")
            return bytes(d).hex() if isinstance(d, (bytes, bytearray)) else ""

        try:
            public = Path(delivered.proof.public).read_bytes().hex()
        except OSError:
            public = None
        word = delivered.output
        case: dict[str, Any] = {
            "kind": "warden", "link": self.link, "run": self.run, "window": w, "root": reg["roots"][0]["root"],
            "registration": sample.docket.digest, "units": list(sample.units), "program": self._partition["program"],
            "partition": P.digest(self._partition), "gates": [g.id for g in self.gates()],
            "findings": None if findings is None else [
                {"accepted": f.accepted, "program": f.program, "units": list(f.units)} for f in findings],
            "public": public, "salts": {k: hexed(delivered.salts.get(k)) for k in ("pub", "o0")},
            "output": word if type(word) is int and word >= 0 else None, "sessions": list(sessions),
            "certificates": [cert(c) for c in certificates], "digests": [digest(c) for c in certificates],
            "syncs": [[s.delta, s.rho.numerator, s.rho.denominator] for s in syncs],
            "sync_set": {"deltas": list(self.sync_set.deltas),
                         "rhos": [[r.numerator, r.denominator] for r in self.sync_set.rhos]},
            "lms": {"N": p.N, "W": p.W, "H": p.H, "toy": p.toy}, "key": self.key.public.hex(),
            "session": bytes(self.certifier_session).hex(),
            "egress": {"grid": LA._grid_json(self.params), "profile": None},
            "ingress": {"grid": LA._grid_json(self.ingress_params), "profile": None},
            "warden": self.network_attester.digest.hex()}
        return case

    def lean_verdict(self, sample: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                     delivered: Delivered, findings: Sequence[Finding] | None, sessions: Sequence[str],
                     server: LA.Server | None = None) -> dict[str, Any] | None:
        """The Lean's verdict on the execution phase (`NetworkCheck.TransferBlocks.verdict`, `network-records verdicts`'
        ``link-window``): its accept bit, digest and public outputs; None where `lean_case` has no case or the Lean
        refuses it. The Lean not built or not running raises."""
        case = self.lean_case(sample, certificates, syncs, delivered, findings, sessions)
        if case is None:
            return None
        try:
            out = (server or LA.SERVER).call("link-window", case)
        except ValueError:
            return None
        return {"accepted": out["accepted"], "digest": out["digest"], "public": out["public"]}

    def verify(self, sample: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
               delivered: Delivered, *, circuit_sha512: str, opening: Hidden | None = None) -> dict[str, Any]:
        """The verdict of record on the execution phase's delivered proof, from public inputs and the proof alone: accepted
        when
        - the verifier's own run of Flock's Lean verifier (`proofs.verify`, on the circuit file the auditor built,
          `circuit_sha512`) accepts every session, and they prove this link's registered proof block (`proved`) and the
          sampled proof block;
        - the proof's public row opens, under the salt delivered, to the verifier's own copy (`public_arguments`: the
          registered root, the public fields and both certificates' signed digests), and its output row to the word
          delivered (`_opened`), whose verdict bit and deadline bit are both 1;
        - the certificates' check accepts (`certified`). The link declares no sync: the Python reads nothing of
          `syncs`, which reach the Lean case alone.
        The verdict reports the public outputs, the proved deadline bit among them, and never a violation's kind.
        `opening` (the witness) is for a debug comparison only, off by default: given, the verdict also carries the
        in-the-clear check's accept bit (`clear`), which decides nothing.

        The Lean (`NetworkCheck.TransferBlocks`, `lean_verdict`) decides the same on every call beside this Python, and
        `OF_RECORD` names whose verdict is the record's (`by`); the other's accept bit is `reference`'s. Where the two
        differ in the accept bit, the public outputs or the digest, or the Lean refuses the case, the execution phase is
        rejected, with deadline 0."""
        findings, refused = PF.verify(delivered.proof, circuit_sha512=circuit_sha512, program=self.program,
                                      partition=self._partition)
        findings = None if refused is not None else findings
        sessions = delivered.sessions()
        python = self.python_verdict(sample, certificates, syncs, delivered, findings, sessions)
        lean = self.lean_verdict(sample, certificates, syncs, delivered, findings, sessions)
        by = OF_RECORD
        mine, other = (lean, python) if by == "lean" else (python, lean)
        if lean is None or any(lean[k] != python[k] for k in python):
            mine = self._decided(sample, sessions, False, {"verdict": 0, "deadline": 0})
        verdict = {"of_record": True, "decided_by": "proof", "by": by, **mine, "sessions": sessions,
                   "reference": {"accepted": None if other is None else other["accepted"]}}
        if opening is not None:
            verdict["clear"] = {"accepted": self.clear(opening, sample, syncs)["accepted"]}
        return verdict

    def clear(self, opening: Hidden, sample: S.Draw, syncs: Sequence[ClockSync] = ()) -> dict[str, Any]:
        """The in-the-clear check on the opening, the witness: the debug comparison `verify` makes on request and the
        tests' reference, never a verdict of record. Accepted when the opening reproduces the registered root under the
        execution phase's domain, both head checks accept (the module docstring), the opening's two certificates hold
        (`certified`) and both heads met their deadline. `syncs` is unread: the link declares none."""
        root = self._registered(opening, sample)
        egress, ingress = opening.records
        holds = self.root(opening) == root and all(
            i is not None and HC.reference(i) for i in self.head_checks(opening, bytes.fromhex(root))) \
            and self.certified(opening.window, [r.certificate for r in opening.records])
        public = {"verdict": int(holds), "deadline": int(egress.head.deadline and ingress.head.deadline)}
        return {"of_record": False, "accepted": holds and public["deadline"] == 1, "public": public}

    def statement(self, sample: S.Draw) -> dict[str, Any]:
        reg = sample.docket.record
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link,
                "window": reg["window"]["window"], "root": reg["roots"][0]["root"]}

    def outcome(self, stream: S.Stream, sample: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                delivered: Delivered, *, circuit_sha512: str, opening: Hidden | None = None) -> S.Outcome:
        """`stream.outcome` on the verifier's verdict (`verify`), with the registered root as the session's:
        the root the proof's public row names, which it opens only to the verifier's own copy."""
        verdict = self.verify(sample, certificates, syncs, delivered, circuit_sha512=circuit_sha512, opening=opening)
        return stream.outcome(sample, {VERIFIER: verdict}, self.statement(sample),
                              session_roots={VALUE: sample.docket.record["roots"][0]["root"]})

    def unproved_outcome(self, stream: S.Stream, sample: S.Draw, opening: Hidden,
                         syncs: Sequence[ClockSync]) -> S.Outcome:
        """`stream.outcome` for a run that proves nothing (a rehearsal, the simulator): on the in-the-clear check
        (`clear`), which reads the opening, so it is never zero-knowledge, and its
        verdict says ``"decided_by": "clear"``. The delivered session is the opening's (`session`), and its root the
        one the opening reproduces, which one-stage's audit holds to the registered root."""
        v = self.clear(opening, sample, syncs)
        sessions = [self.session(opening, sample)]
        digest = _sha512({"format": VERDICT, "registration": sample.docket.digest, "sessions": sessions,
                          "accepted": v["accepted"], "public": v["public"], "decided_by": "clear"})
        verdict = {**v, "of_record": True, "decided_by": "clear", "digest": digest, "sessions": sessions}
        return stream.outcome(sample, {VERIFIER: verdict}, self.statement(sample),
                              session_roots={VALUE: self.root(opening)})

    def _period(self, stream: S.Stream, windows: int | None) -> int:
        default = self.windows
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this link's: its spec isn't TransferBlocks.spec")
        n = default if windows is None else windows
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a link has one or more execution phases, not {n!r}")
        return n

    def _committed(self, stream: S.Stream, roots: Sequence[str]) -> str:
        """The period's digest, what the coins come after: this stream and the committed roots."""
        return _sha512({"format": PERIOD_DRAW, "stream": stream.digest, "roots": list(roots)})

    def period_draw(self, stream: S.Stream, roots: Sequence[str], k: int | None, coins: Any) -> dict[str, Any]:
        """The period's sample over its execution phases 0 … n − 1, after the prover committed them: `roots`, window w's
        salted root (`root`, hex) at w, n of them (by default the link's windows, which `network_verdict` takes). Every
        window for `k` None (`All`, reading no coins), else `subset:k`, 1 <= k <= n (the rule `rate:k/n`), on `coins`
        (a callable of a byte count, as a stream's); coins with a `commit` method are first committed to the
        period's digest (`commitment`), so the coins come after every commitment of the period. Call it before the
        stream's first registration. The object is `{format, stream, law, k, population, units, roots, commitment,
        read}`, `read` the coin bytes the sample read, in order (hex), from which the network verdict's Lean samples its
        own (`lean_draw`); `register` and `network_verdict` take it as `drawn`. Raises `ValueError` for a stream that
        isn't this link's or holds a registration, roots that aren't n SHA-512 hex digests, and k outside [1, n]."""
        n = self._period(stream, len(roots) if isinstance(roots, Sequence) and not isinstance(roots, str) else None)
        if stream.log:
            raise ValueError("the period is committed and drawn before the stream's first registration")
        roots = list(roots)
        if not all(isinstance(r, str) and len(r) == 128 and all(c in "0123456789abcdef" for c in r) for r in roots):
            raise ValueError("a period commits each window's salted root, a SHA-512 in lowercase hex")
        if k is not None and not (type(k) is int and 1 <= k <= n):
            raise ValueError(f"a period sample takes 1 <= K <= {n} of the execution phases, not {k!r}")
        commitment = self._committed(stream, roots)
        rule = S.All() if k is None else S.SamplingDesign(f"rate:{k}/{n}")
        if k is not None and callable(getattr(coins, "commit", None)):
            coins.commit(bytes.fromhex(commitment))
        read = bytearray()

        def reading(m: int) -> bytes:
            b = coins(m)
            read.extend(b)
            return b
        sample = S.select_draw(rule, S.selection_sampling_design(rule, n), n, reading)
        return {"format": PERIOD_DRAW, "stream": stream.digest, "law": "subset", "k": sample["k"], "population": n,
                "units": list(sample["units"]), "roots": roots, "commitment": commitment, "read": read.hex()}

    def _drawn(self, stream: S.Stream, n: int, drawn: Mapping[str, Any]) -> list[int]:
        """The sampled windows, once `drawn` is a period sample over this stream's n windows; else `ValueError`."""
        get = drawn.get if isinstance(drawn, Mapping) else (lambda _: None)
        units, k, roots, read = get("units"), get("k"), get("roots"), get("read")
        if not (isinstance(drawn, Mapping) and get("format") == PERIOD_DRAW and get("law") == "subset"
                and get("population") == n and type(k) is int and 1 <= k <= n and isinstance(units, list)
                and all(type(u) is int and 0 <= u < n for u in units) and units == sorted(set(units))
                and len(units) == k and isinstance(roots, list) and len(roots) == n and isinstance(read, str)
                and len(read) % 2 == 0 and all(c in "0123456789abcdef" for c in read)):
            raise ValueError(f"not a period sample of subset:K over this link's {n} windows")
        if get("stream") != stream.digest or get("commitment") != self._committed(stream, roots):
            raise ValueError("the period sample is not this stream's, or not over the period it committed")
        return list(units)

    @staticmethod
    def _of_record(n: int, drawn: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
        """The period's sample of record over its n windows, ascending, and None; or None and why there is none: every
        window at K = n, else `lean_draw` on the coins `drawn` read, which must be `subset:K` over n."""
        k = drawn["k"]
        if k == n:
            return list(range(n)), None
        got = lean_draw(k, n, bytes.fromhex(drawn["read"]))
        if not isinstance(got, Mapping) or "refused" in got:
            return None, str(got.get("refused") if isinstance(got, Mapping) else got)
        units = got.get("units")
        if not (got.get("law") == "subset" and got.get("k") == k and got.get("population") == n
                and isinstance(units, list) and all(type(u) is int and 0 <= u < n for u in units)
                and len(set(units)) == len(units) == k):
            return None, f"the Lean's sample is not subset:{k} over the {n} windows: {json.dumps(got)[:300]}"
        return sorted(units), None

    def network_verdict(self, stream: S.Stream, windows: int | None = None,
                        drawn: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The link's network verdict of record: the AND over its execution phases 0 … `windows` − 1 (by default the
        egress profile's windows, fixed before the run), with no ε and no share credit. It accepts only when, for every
        w, the stream's w-th registration is window w and its outcome of record accepted. A window registered out of
        order or not at all (a gap, or a run that stops short), a registration past the last window, a registration
        with no outcome yet, and a rejected or late outcome each reject the link; `refused` lists their windows.
        `records` is each window's outcome record SHA-512 (`S.Outcome.record`, None without one), and `digest` binds
        the verdict to the stream and those records. The windows are by default the `NetworkAttesterProfile`'s (`windows`).

        With `drawn`, the period sample (`period_draw`) over this stream's n windows, the stream's registrations must be
        exactly the drawn windows, in order, each under the root the period committed for it, and each needs an
        accepted outcome. An unsampled window has no registration and
        its record is None; a registration past the drawn windows is listed in `refused` as n, n + 1, … The verdict
        then carries `selection` (the sample) and `wrong_units`, what it established at the stream's δ: except with
        probability δ, an accepted link has at most `bound` wrong windows of the `of` committed, and `epsilon` is
        bound / of (the uniform K-subset's exact bound, one-stage's `wrong_units` over windows); `digest` binds the
        sample too.

        The sampled windows are the sample of record's, never the proof blocks `drawn` lists: the verdict runs
        `lean_draw` on `drawn`'s coins (`read`), and every window at K = n, which leaves nothing to sample. It refuses
        the link by name, `why[PERIOD_DRAWN]` with the Lean's sample (or why there is none), `drawn`'s proof blocks and
        the windows registered, when the Lean can't sample, when `drawn`'s proof blocks aren't the Lean's, and when a
        registered window isn't one the Lean sampled; a sampled window not registered is in `refused`. `selection` is
        the sample of record, ``by`` ``lean``
        (or ``all`` at K = n), and its units are the Lean's. Raises `ValueError` on a stream that isn't this link's,
        and on a sample that isn't this stream's period sample."""
        n = self._period(stream, windows)
        why: dict[str, Any] = {}
        if drawn is None:
            expected = list(range(n))
        else:
            claimed = self._drawn(stream, n, drawn)
            lean, no = self._of_record(n, drawn)
            expected = claimed if lean is None else lean
            by = None if lean is None else "all" if drawn["k"] == n else "lean"
            registered = [p.get("window") if isinstance(p := r.get("window"), Mapping) else None for r in stream.log]
            if lean is None or lean != claimed or any(w not in lean for w in registered):
                why[PERIOD_DRAWN] = {"lean": lean if lean is not None else {"refused": no}, "drawn": claimed,
                                     "registered": registered,
                                     "why": no or "the windows registered and proved are not the Lean verifier of "
                                                  "record's sample on the period's coins"}
        outcomes = {r["registration"]: r for r in stream.records}
        records: list[str | None] = [None] * n
        refused: list[int] = []
        for i in range(max(len(expected), len(stream.log))):
            reg = stream.log[i] if i < len(stream.log) else None
            w = expected[i] if i < len(expected) else n + i - len(expected)
            in_order = reg is not None and reg.get("window") == self.provenance(w)
            body = outcomes.get(R.digest(reg)) if in_order else None
            record = None if body is None else _sha512(body)
            if w < n:
                records[w] = record
            else:
                records.append(record)
            ok = (w < n and in_order and body is not None and body["symbol"] == S.ACCEPT
                  and (drawn is None or _root_of(reg) == drawn["roots"][w]))
            if not ok:
                refused.append(w)
        accepted = not refused and not why
        sampled = {} if drawn is None else {"drawn": dict(drawn), "of_record": expected, "why": why}
        digest = _sha512({"format": NETWORK_VERDICT, "stream": stream.digest, "windows": n, "records": records,
                          "accepted": accepted, **sampled})
        out = {"accepted": accepted, "windows": n, "refused": refused, "records": records, "digest": digest}
        if drawn is not None:
            bound = OA.core_sampling_design({"law": "subset", "k": drawn["k"]}, n).count_bound(stream.delta)
            out |= {"why": why,
                    "selection": {**{k: drawn[k] for k in ("law", "k", "population", "commitment")}, "units": expected,
                                  "by": by},
                    "wrong_units": {"delta": stream.delta, "of": n, "bound": bound, "epsilon": bound / n,
                                    "rule": "except with probability delta, an accepted link has at most `bound` wrong "
                                            "execution phases of the `of` its period committed"}}
        return out


__all__ = ["BC_BYTES", "Delivered", "HEADS", "LEAVES", "TransferBlocks", "OUTPUTS", "OUT_BITS", "Hidden", "PORTS",
           "PUBLIC", "ROWS", "SCHEMA", "VALUE", "NetworkAttesterProfile", "encode", "output_row", "rows", "unit"]
