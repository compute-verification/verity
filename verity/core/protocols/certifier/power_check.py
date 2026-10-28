"""The power certifier's proof unit, zero-knowledge: one interval's power certificate against its committed counts and
the public threshold (`power`). A Boolean Definition whose one output is the verdict bit; drawn intervals are proved
through the firewall like N2's link-windows, and the unit's Program holds it as one gate naming its pinned circuit digest
(`gate`, `head_check.gate_of`).

    PowerCheck_v1{L, J, DH, DC}
        (power: Digest, power_domain: Digest, power_pos: Array<DH>, channel: Array<L, Byte>, interval: Array<64>,
         interval_ms: Array<32>, parameters: Array<64, Byte>, counts_root: Digest, counts_domain: Digest,
         counts_pos: Array<DC>, signed: Digest, floors: Array<J, Array<48>>, gain: Array<24>, slack: Array<128>,
         head: Array<HL, Byte>, salt: Salt, path: Array<DH, Digest>, counts: Array<J, Array<8, Byte>>,
         counts_salt: Salt, counts_path: Array<DC, Digest>, salt_c: Digest) -> Value<1>
        HL = 91 + L, and Digest, Salt and Byte as `head_check`'s

**Statics: the size class.** L is the UTF-8 length of the channel's name, J the counted kinds (the declared operand
classes' MACs, compressions, forming units), DH and DC the depths of the trees the head and the counts open in. All are
public: the channel, the threshold's kinds and the commitments' shapes are protocol structure.

**Public inputs.** The power root and its domain id and the head's leaf position in it; the channel, the interval's index
(u64) and length Δ_ms (u32), and the meter's manifest digest (`power.Meter.digest`); the counts root and its domain id
and the counts' leaf position; `signed`, the digest the interval's certificate signs; and the threshold as `power`
quantizes it (`Threshold.quantized`): a floor per kind (u48), the gain (u24) and the slack (u128), every integer least
significant bit first.

**Witness.** The record head's bytes (`PowerHead.encode`), its hiding salt and path; the counts row
(`power.counts_row`), its hiding salt and path; and `salt_c`, the certifier's salt for the record.

**What 1 means.** (1) The head opens at `power_pos` of the power root and (2) the counts row at `counts_pos` of the
counts root (`NetworkHidingRead`: the sha512/row/v2 digest, hm96's hiding leaf, the frame-v3-sha512 read). (3) The head's
fields are the public inputs' (`PowerHeadFields`): its channel, interval, Δ_ms, status 0 (complete) and manifest digest.
(4) The threshold holds (`PowerThreshold`): gain · E_µJ · 2²⁰ ≤ Σ_j floor_j · N_j + slack, with E the head's energy and
N_j the counts. (5) The certificate signs this head: its record digest (`PowerRecordDigest`) salted by `salt_c` is
`signed` (`head_check.NetworkSignedDigest`). The head's sample count is free.

**Cost.** The threshold is two carry-save products and one comparison: 24·64 + J·48·64 partial products. The openings
and the record digest are SHA-512 and dominate (`size`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import and_reduce, compress_columns, ge_const_free
from verity.ml.circuits.boolean.trace import part as _part
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import root_ref
from verity.ml.commitments.identity import tagged_sha512, tagged_sha512_prefix
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.certificate import DIGEST_BYTES, certificate_digest
from verity.core.protocols.certifier.power import (ENERGY_BITS, FLOOR_WIDTH, GAIN_BITS, GAIN_WIDTH, RECORD_TAG,
                                                   SLACK_WIDTH, PowerStatus, passes_quantized)

BIT, BYTE, DIGEST, SALT = HC.BIT, HC.BYTE, HC.DIGEST, HC.SALT
COMPLETE = PowerStatus.COMPLETE.value
#: the circuit digests circuit-check pins (`pinned_digest`)
PINS = Path(__file__).with_name("power_check.circuit-check.json")


class Shape(SimpleNamespace):
    """A power head's byte offsets (`PowerHead.encode`) for a channel name of L bytes: the interval, Δ_ms, the status, the
    manifest digest, the energy and the sample count."""

    def __init__(self, L: int) -> None:
        if type(L) is not int or not 0 <= L < 1 << 16:
            raise ValueError(f"a power head has 0 <= L < 2^16, not L={L!r}")
        super().__init__(L=L, IV=2 + L, MS=10 + L, ST=14 + L, PAR=15 + L, E=79 + L, N=87 + L, HL=91 + L)


def _le(byte_list, n: int) -> list:
    """A u(8n) big-endian's bits, least significant first, from its n bytes (8 bits each, least significant first)."""
    return [byte_list[n - 1 - j // 8][j % 8] for j in range(8 * n)]


# --- the circuit: traced parts ----------------------------------------------------------------------------------------

def _sig_fields(S):
    sh = Shape(S.L)
    return ((("head", Array(sh.E, BYTE)), ("channel", Array(S.L, BYTE)), ("interval", Array(64, BIT)),
             ("interval_ms", Array(32, BIT)), ("parameters", Array(DIGEST_BYTES, BYTE))), BIT)


def _build_fields(C, S, *e):
    sh = Shape(S.L)
    head, channel = e[:sh.E], e[sh.E:sh.E + S.L]
    interval, interval_ms, parameters = e[sh.E + S.L], e[sh.E + S.L + 1], e[sh.E + S.L + 2:]

    def const(v: int) -> list[int]:
        return [v >> b & 1 for b in range(8)]

    want = [(0, const(S.L >> 8)), (1, const(S.L & 0xFF)), *((2 + i, channel[i]) for i in range(S.L)),
            *((sh.IV + i, interval[8 * (7 - i):8 * (8 - i)]) for i in range(8)),
            *((sh.MS + i, interval_ms[8 * (3 - i):8 * (4 - i)]) for i in range(4)), (sh.ST, const(COMPLETE)),
            *((sh.PAR + i, parameters[i]) for i in range(DIGEST_BYTES))]
    return [and_reduce(C, [HC._same(head[at][b], byte[b]) for at, byte in want for b in range(8)])]


PowerHeadFields = _part(
    "PowerHeadFields", ("L",), _sig_fields, _build_fields,
    "1 iff the power head's bytes before its energy are u16 L, the channel, the interval (u64 big-endian), Δ_ms (u32 "
    "big-endian), status 0 and the meter's manifest digest")


def _sig_threshold(S):
    return ((("energy", Array(8, BYTE)), ("counts", Array(S.J, Array(8, BYTE))),
             ("floors", Array(S.J, Array(FLOOR_WIDTH, BIT))), ("gain", Array(GAIN_WIDTH, BIT)),
             ("slack", Array(SLACK_WIDTH, BIT))), BIT)


def _build_threshold(C, S, *e):
    energy, counts = _le(e[:8], 8), e[8:8 + 8 * S.J]
    floors, gain, slack = e[8 + 8 * S.J:8 + 9 * S.J], e[8 + 9 * S.J], e[9 + 9 * S.J]
    shift = ENERGY_BITS - GAIN_BITS
    lhs_cols: list[list] = [[] for _ in range(shift + GAIN_WIDTH + 64)]
    for i, g in enumerate(gain):
        for j, x in enumerate(energy):
            lhs_cols[shift + i + j].append(C.AND(g, x))
    rhs_cols: list[list] = [[b] for b in slack]  # each product is under 2^(48 + 64), inside the slack's columns
    for k in range(S.J):
        n = _le(counts[8 * k:8 * k + 8], 8)
        for i, f in enumerate(floors[k]):
            for j, x in enumerate(n):
                rhs_cols[i + j].append(C.AND(f, x))
    lhs, rhs = compress_columns(C, lhs_cols), compress_columns(C, rhs_cols)
    return [ge_const_free(C, rhs, lhs, max(len(lhs), len(rhs)))]


PowerThreshold = _part(
    "PowerThreshold", ("J",), _sig_threshold, _build_threshold,
    "1 iff gain · E · 2^(40 − 20) ≤ Σ_j floor_j · N_j + slack, for the energy E and the J counts N_j (u64 big-endian) and "
    "the public floors (u48), gain (u24) and slack (u128), least significant bit first")


def _sig_recorddigest(S):
    return (("head", Array(S.HL, BYTE)),), DIGEST


def _body_recorddigest(B, S, head):
    K = HC._Consts(B)
    hb = HC._refs(head)
    prefix = [K.bits(b, 8) for b in tagged_sha512_prefix(RECORD_TAG, S.HL)]
    return HC._sha512(B, K, prefix + [hb[8 * k:8 * k + 8] for k in range(S.HL)])


PowerRecordDigest = CompositeDefinition(
    "PowerRecordDigest", 1, ("HL",), _sig_recorddigest, _body_recorddigest, conformance="exact-model-tested",
    doc="the record digest of an HL-byte power head (`PowerHead.digest`): tagged_sha512 of the power record tag and the "
        "head")


# --- the check --------------------------------------------------------------------------------------------------------

def _check_statics(S) -> Shape:
    if not all(type(x) is int for x in (S.J, S.DH, S.DC)) or S.J < 1 or not 1 <= S.DH <= 64 or not 1 <= S.DC <= 64:
        raise ValueError(f"a power check has J >= 1 and 1 <= DH, DC <= 64, not J={S.J!r}, DH={S.DH!r}, DC={S.DC!r}")
    return Shape(S.L)


def _params(S) -> list[tuple[str, object]]:
    sh = _check_statics(S)
    return [("power", DIGEST), ("power_domain", DIGEST), ("power_pos", Array(S.DH, BIT)), ("channel", Array(S.L, BYTE)),
            ("interval", Array(64, BIT)), ("interval_ms", Array(32, BIT)), ("parameters", Array(DIGEST_BYTES, BYTE)),
            ("counts_root", DIGEST), ("counts_domain", DIGEST), ("counts_pos", Array(S.DC, BIT)), ("signed", DIGEST),
            ("floors", Array(S.J, Array(FLOOR_WIDTH, BIT))), ("gain", Array(GAIN_WIDTH, BIT)),
            ("slack", Array(SLACK_WIDTH, BIT)),
            ("head", Array(sh.HL, BYTE)), ("salt", SALT), ("path", Array(S.DH, DIGEST)),
            ("counts", Array(S.J, Array(8, BYTE))), ("counts_salt", SALT), ("counts_path", Array(S.DC, DIGEST)),
            ("salt_c", DIGEST)]


def _sig_check(S):
    return tuple(_params(S)), BIT


def _body_check(B, S, *args):
    sh = _check_statics(S)
    a = dict(zip([k for k, _ in _params(S)], args))
    hb = HC._refs(a["head"])
    oks = [B.call(bind(HC.NetworkHidingRead, N_BITS=8 * sh.HL, D=S.DH), a["head"].reshape(Array(8 * sh.HL, BIT)),
                  a["salt"], a["power"], a["power_domain"], a["power_pos"], a["path"]),
           B.call(bind(HC.NetworkHidingRead, N_BITS=64 * S.J, D=S.DC), a["counts"].reshape(Array(64 * S.J, BIT)),
                  a["counts_salt"], a["counts_root"], a["counts_domain"], a["counts_pos"], a["counts_path"]),
           B.call(bind(PowerHeadFields, L=S.L), HC._coll(Array(sh.E, BYTE), hb[:8 * sh.E]), a["channel"],
                  a["interval"], a["interval_ms"], a["parameters"]),
           B.call(bind(PowerThreshold, J=S.J), HC._coll(Array(8, BYTE), hb[8 * sh.E:8 * sh.N]), a["counts"],
                  a["floors"], a["gain"], a["slack"]),
           B.call(HC.NetworkSignedDigest, B.call(bind(PowerRecordDigest, HL=sh.HL), a["head"]), a["salt_c"],
                  a["signed"])]
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


PowerCheck = CompositeDefinition(
    "PowerCheck", 1, ("L", "J", "DH", "DC"), _sig_check, _body_check, conformance="exact-model-tested",
    doc="1 iff the power head opens at power_pos of the power root and the counts at counts_pos of the counts root, the "
        "head's fields are the interval's, complete and the meter's, the threshold holds on its energy and the counts, "
        "and its record digest salted by salt_c is the signed digest; its sample count is free")

PowerCheck.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`gate`)


def unit(L: int, J: int, DH: int, DC: int):
    return bind(PowerCheck, L=L, J=J, DH=DH, DC=DC)


@lru_cache(maxsize=None)
def gate(L: int, J: int, DH: int, DC: int) -> PrimitiveDefinition:
    """`unit(L, J, DH, DC)` as one gate naming its pinned circuit digest (`head_check.gate_of`); a size class no sidecar
    pins names none, and no proof of it is accepted."""
    d = unit(L, J, DH, DC)
    return HC.gate_of(d, HC.pinned_digest(d, PINS), "the power certifier's check")


# --- the reference and the arguments ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """One interval's check: its arguments as bytes and ints (positions as leaf indices, the threshold quantized)."""

    power: bytes
    power_domain: bytes
    power_pos: int
    channel: bytes
    interval: int
    interval_ms: int
    parameters: bytes
    counts_root: bytes
    counts_domain: bytes
    counts_pos: int
    signed: bytes
    floors: tuple[int, ...]
    gain: int
    slack: int
    head: bytes
    salt: bytes
    path: tuple[bytes, ...]
    counts: bytes
    counts_salt: bytes
    counts_path: tuple[bytes, ...]
    salt_c: bytes

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (len(self.channel), len(self.floors), len(self.path), len(self.counts_path))

    @property
    def unit(self):
        return unit(*self.key)


def reference_parts(inst: Instance) -> dict[str, bool]:
    """Each part's output on the instance, from its arguments as the circuit reads them."""
    sh, h = Shape(len(inst.channel)), inst.head
    if len(h) != sh.HL:
        raise ValueError(f"the head is {len(h)} bytes, the unit's {sh.HL}")
    if len(inst.counts) != 8 * len(inst.floors):
        raise ValueError(f"the counts row is {len(inst.counts)} bytes, the unit's {8 * len(inst.floors)}")
    fields = (h[:2] == sh.L.to_bytes(2, "big") and h[2:sh.IV] == inst.channel
              and h[sh.IV:sh.MS] == inst.interval.to_bytes(8, "big") and h[sh.MS:sh.ST] == inst.interval_ms.to_bytes(4, "big")
              and h[sh.ST] == COMPLETE and h[sh.PAR:sh.E] == inst.parameters)
    counts = tuple(int.from_bytes(inst.counts[8 * k:8 * k + 8], "big") for k in range(len(inst.floors)))
    threshold = passes_quantized(int.from_bytes(h[sh.E:sh.N], "big"), counts, inst.floors, inst.gain, inst.slack)
    opens = root_ref(inst.power_domain, SCHEMA, HC.hiding_value(h, inst.salt), inst.power_pos, inst.path) == inst.power
    counts_open = root_ref(inst.counts_domain, SCHEMA, HC.hiding_value(inst.counts, inst.counts_salt), inst.counts_pos,
                           inst.counts_path) == inst.counts_root
    signed = certificate_digest(inst.salt_c, tagged_sha512(RECORD_TAG, h)) == inst.signed
    return {"opening": bool(opens), "counts opening": bool(counts_open), "fields": bool(fields),
            "threshold": threshold, "signed": signed}


def reference(inst: Instance) -> bool:
    """`PowerCheck_v1`'s output on the instance: every part's."""
    return all(reference_parts(inst).values())


def arguments(inst: Instance) -> list[list[int]]:
    """The unit's arguments for one instance in its signature's order, each its leaves' bits, 0 or 1."""
    L, J, DH, DC = inst.key
    flat = {"power": HC._digest_bits(inst.power), "power_domain": HC._digest_bits(inst.power_domain),
            "power_pos": HC._int_bits(inst.power_pos, DH), "channel": HC._bytes_bits(inst.channel),
            "interval": HC._int_bits(inst.interval, 64), "interval_ms": HC._int_bits(inst.interval_ms, 32),
            "parameters": HC._bytes_bits(inst.parameters), "counts_root": HC._digest_bits(inst.counts_root),
            "counts_domain": HC._digest_bits(inst.counts_domain), "counts_pos": HC._int_bits(inst.counts_pos, DC),
            "signed": HC._digest_bits(inst.signed),
            "floors": [b for f in inst.floors for b in HC._int_bits(f, FLOOR_WIDTH)],
            "gain": HC._int_bits(inst.gain, GAIN_WIDTH), "slack": HC._int_bits(inst.slack, SLACK_WIDTH),
            "head": HC._bytes_bits(inst.head), "salt": HC._digest_bits(inst.salt),
            "path": [b for s in inst.path for b in HC._digest_bits(s)], "counts": HC._bytes_bits(inst.counts),
            "counts_salt": HC._digest_bits(inst.counts_salt),
            "counts_path": [b for s in inst.counts_path for b in HC._digest_bits(s)],
            "salt_c": HC._digest_bits(inst.salt_c)}
    params, _ = inst.unit.signature()
    out = []
    for name, t in params:
        if len(flat[name]) != t.leaves:
            raise ValueError(f"argument {name} has {len(flat[name])} bits, the unit takes {t.leaves}")
        out.append(flat[name])
    return out


def evaluate(*insts: Instance) -> list[int]:
    """The unit's output on each instance, all of one unit, in one evaluation."""
    if len({i.key for i in insts}) != 1:
        raise ValueError("the instances are of one unit")
    args = [list(rows) for rows in zip(*(arguments(i) for i in insts))]
    return [int(b) for b in evaluate_bits(insts[0].unit, *args).reshape(-1).tolist()]


_GROUPS = {"NetworkHidingRead": "openings", "PowerHeadFields": "fields", "PowerThreshold": "threshold",
           "PowerRecordDigest": "signed", "NetworkSignedDigest": "signed"}


def size(target, memo: dict | None = None) -> dict:
    """The check's ANDs, in total and by part: the two openings, the fields, the threshold, the signed digest (the
    record digest and its salted digest) and the rest (the final AND)."""
    memo = {} if memo is None else memo
    out = dict.fromkeys(("openings", "fields", "threshold", "signed", "other"), 0)
    for nd in target.body.nodes:
        out[_GROUPS.get(getattr(nd.fn, "name", ""), "other")] += HC._times(nd) * HC.ands(nd.fn, memo)
    out["ands"] = sum(out.values())
    return out


__all__ = ["Instance", "PINS", "PowerCheck", "PowerHeadFields", "PowerRecordDigest", "PowerThreshold", "Shape",
           "arguments", "evaluate", "gate", "reference", "reference_parts", "size", "unit"]
