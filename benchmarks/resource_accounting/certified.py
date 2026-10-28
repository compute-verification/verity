"""What a point's arms certify, and each resource's covert residual.  Pure (stdlib and `verity`), so it runs on a CPU.

* **W_cert, compute.**  `Pouw.Guarantees.EndToEnd` (``Security/Definitions/Pouw/Guarantees/Game.lean``) bounds the chance that an
  audit of t draws accepts a transcript whose prover spent T < (1 − γ)(1 − ε_s)·ΣW_ref by (1 − ε_s)^t + ε_TT(q, N).  So
  W_cert = (1 − γ)(1 − ε_s)·ΣW_ref, in W1 units (one dense FP8 MAC on the device), with ΣW_ref the drawn statement's
  reference work over the credited cells (`clear_work.ticket_cell`; about 1.09 to 1.11 per MAC at the sweep's shapes).
  γ is looked up from the Lean-pinned price twins (`price_twins.lookup`) of the statement on the priced path the scheme's
  draw cites (`clear_work.draw_wref`), at both FP32 prices, the larger published; it is never typed.  Twins are stated
  at the cubes 8,192³ and 16,384³ and at each of Llama-3.1-8B's linears (n, k) and its layout (`GAMMA_PIN`), so
  every shape the sweep names is priced; a shape no twin states has W_cert unproved.  ε_s = 0.1% at t = K = 27,713 draws
  (`plan.RECORD_K`, (1 − ε_s)^K ≤ 2⁻⁴⁰).  A chain's credited rows are its screened rows less those the screen failed or
  left unsure, a lower bound on the rows the records credit.
* **Q_cert, bandwidth**: ``Bandwidth.SecurityProofs.AuditRO`` at the point the red team granted at 5761861fd (`Q_GRANT`,
  `granted_params`: `ro-blake3/v1`, `s = 1,849`, `T = 1.2 ms`, the verifier in the prover's process, so no RTT): an
  accepted window read at least `audit.bound`'s ingress of the store (`q_cert`), with S_chip the bandwidth harness's count
  of the device (`chip`: L2, registers, L1 and shared memory, and the constant and instruction caches beside them) and the
  side link at the harness's cap.  It holds for a store set up under `bandwidth.audit.draw_salt`'s device-bound salt (the PoUS
  harness's ``--bandwidth-salt``), which the granted verifier checks.  A chain run without windows certifies none.  A
  window is also reported on time at `window_params`' T, the RTT at P2's cap (`RTT_ALLOWANCE_CAP_NS`) under the rebuild
  floor, the T a verifier on another machine would grant.
* **S_cert, capacity**: the PoUS arm's storage profile (`storage`), when the harness issued one and it is certified: a
  server holding fewer than ⌊ρ·|C|⌋ bits passes the audit of record's k challenges within its limit with probability at most
  δ + 2^eps_log2, by the certificate's theorem (P2's is ``Pous.SecurityProofs.P2SlackFamilyUncond``), under its claims,
  with P2 itself experimental (the profile says why) and the deployed ring's theorem held.  S_cert is that bound in bytes;
  otherwise it is unproved, with why.
* **The covert residual** of a resource over an arm's chain time T is what the device could have done in T less what is
  certified: compute F·T − W_cert, with F the native FP8 chain's rate at the shape (W1 per second); bandwidth BW·T − Q_cert,
  BW the native read rate; capacity M − S_cert, M the device's memory.  It splits into honest overhead, native less the
  honest reference (F·T − ΣW_ref: what the protocol's own time costs; M − |C|, the memory the store leaves), and soundness
  slack, the reference less what is certified (ΣW_ref − W_cert; |C| − S_cert).  It is the weak form (`RESIDUAL`):
  ``CovertResidual`` is worst case over each audit's own coins, not a per-run bound.  A per-run compute residual holds only
  conditional on ``BudgetAbort`` for the deployed cost model (`PER_RUN`), so none is shown.  First fetches are charged to
  the bandwidth account, not compute (`FIRST_FETCH`).
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TWINS = ROOT / "benchmarks" / "pouw" / "harness" / "price_twins.py"
CLEAR = ROOT / "benchmarks" / "pouw" / "clear"
#: the panel line and version Pearl-C's arms run (`price_twins.statement` drops the hashing format)
SCHEME, LINE, VERSION = "pearl-c-sm120-v1-h3", "pearl-c-sm120", "v1-h3"
EPS_S = Fraction(1, 1000)
DELTA = Fraction(1, 2 ** 40)
#: the Lean domains of the twins stated at a linear's (n, k) and over a served layout's linears (`price_twins`' n, k
#: and layout keys), which price γ at the Llama linears
GAMMA_PIN = "pearlCDomainDevNK, pearlCDomainDevShapes"
#: the served layout whose twin is γ's max over its linears
LAYOUT = "llama31_8b"
#: the bandwidth protocol's red-team grant (with conditions), whose point `granted_params` is
Q_GRANT = "red-team grant at 5761861fd"
Q_THEOREM = "Bandwidth.SecurityProofs.AuditRO"
GRANTED_T_NS = 1_200_000
RESIDUAL = "residual, worst case over the audit's coins (CovertResidual's weak form)"
PER_RUN = ("a per-run compute residual holds only conditional on BudgetAbort for the deployed cost model "
           "(CovertResidualPerRun; BudgetAbort holds for constant-cost models and W1's operand refund breaks it), so "
           "none is shown")
FIRST_FETCH = ("first fetches are charged to the bandwidth account, not compute; a priced-sum column, "
               "compute + p·reads at p the device's DRAM rate in MAC units, waits for the priced lemma "
               "(w1Priced)")
S_BASIS = ("Pous.SecurityProofs.P2SlackFamilyUncond; P2 is experimental, and the deployed ring's theorem (EndToEndRing) "
           "is held")
NO_POUS = "no PoUS arm in this run"
SUMMARIZE = ROOT / "benchmarks" / "bandwidth" / "summarize.py"


def summarize():
    """The bandwidth harness's `summarize`, loaded by path: S_chip's count and the side link's cap."""
    if "bandwidth_summarize" not in sys.modules:
        spec = importlib.util.spec_from_file_location("bandwidth_summarize", SUMMARIZE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bandwidth_summarize"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["bandwidth_summarize"]


def chip(device: dict) -> dict:
    """S_chip of a device (its `l2_bytes`, `sms` and `regs_per_sm`), the bandwidth harness's count (`summarize.s_chip_bytes`:
    L2, registers, L1 and shared memory, and the red team's allowance for the caches and buffers beside them)."""
    s = summarize()
    return {"s_chip_bytes": s.s_chip_bytes(device), "formula": s.S_CHIP_FORMULA}


def window_params(tag: str):
    """A bandwidth window of `tag` at the proof's samples, with the RTT at P2's cap: `T + RTT` one ns under the rebuild floor."""
    from verity.core.protocols.bandwidth import audit
    from verity.core.protocols.pous.schemes.p2 import RTT_ALLOWANCE_CAP_NS
    return audit.Params(tag, audit.SAMPLES, audit.REGEN_FLOOR_NS - RTT_ALLOWANCE_CAP_NS - 1, rtt_allowance_ns=RTT_ALLOWANCE_CAP_NS)


def granted_params(tag: str):
    """A window of `tag` at the granted point: `s = 1,849`, `T = 1.2 ms`, and no RTT, the verifier in the prover's process."""
    from verity.core.protocols.bandwidth import audit
    return audit.Params(tag, audit.SAMPLES, GRANTED_T_NS)


def q_cert(tag: str, blocks: int, block_bytes: int, chip_bytes: int) -> dict:
    """Q_cert of one accepted window of `tag` at the granted point over a store of `blocks` blocks: `audit.bound`'s ingress
    with S_chip `chip_bytes` and the side link at the bandwidth harness's cap (`summarize.PCIE_GEN5_X16`, PCIe Gen 5 x16
    at its raw rate), and its error."""
    from verity.core.protocols.bandwidth import audit
    p = granted_params(tag)
    b = audit.bound(p, blocks, block_bytes, chip_bytes=chip_bytes, side_bytes_per_s=summarize().PCIE_GEN5_X16)
    return {"q_cert": b.ingress_bytes, "error": b.error, "error_log2": math.log2(b.error), "side_bytes": b.side_bytes,
            "chip_bytes": chip_bytes, "samples": p.samples, "limit_ns": p.limit_ns, "basis": Q_GRANT, "theorem": Q_THEOREM}

def price_twins():
    """The harness's `price_twins` module, loaded by path (benchmarks/pouw/harness is not a package)."""
    if "price_twins" not in sys.modules:
        spec = importlib.util.spec_from_file_location("price_twins", TWINS)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["price_twins"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["price_twins"]


def clear_work():
    """PoUW's work law, the clear audit's (`clear_work`; benchmarks/pouw/clear is not a package)."""
    if str(CLEAR) not in sys.path:
        sys.path.append(str(CLEAR))
    import clear_work
    return clear_work


def scheme():
    from verity.ml.protocols.pouw.schemes import pearl_c as C
    from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES
    from verity.core.protocols.pouw.serving import SchemeId
    sid = SchemeId.parse(SCHEME)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing)
    if s.name != SCHEME:
        raise ValueError(f"the scheme built for {SCHEME!r} names itself {s.name!r}")
    return s


def priced_path(s) -> str:
    """The priced path the scheme's draw cites, from its statement's name (``...Cast8p72...`` is ``cast-8.72``)."""
    at = clear_work().draw_wref(s)
    if at is None:
        raise ValueError(f"{s.name}: its draw cites no statement")
    m = re.search(r"Cast(8p72|32p06|16|32|8)", at.statement)
    if m is None:
        raise ValueError(f"{at.statement}: no cast in the statement's name")
    return price_twins().CASTS[m.group(1)]


def gamma(n: int | None, k: int | None, layout: str | None = None) -> dict:
    """γ for Pearl-C at (n, k), or of a served `layout` (`LAYOUT`): `price_twins.lookup`'s record (both prices, the larger
    published, what's missing), with the statement's cap held to the scheme's and K to ε_s and δ."""
    from verity.ml.protocols.pouw.circuit import plan
    P, s = price_twins(), scheme()
    stmt = P.statement(LINE, VERSION)
    if P.CAPS[stmt] != s.cap or clear_work().draw_wref(s).cap != s.cap:
        raise ValueError(f"{stmt}: the twins' cap {P.CAPS[stmt]} is not the scheme's {s.cap}")
    if plan.record_k(EPS_S, DELTA) != plan.RECORD_K:
        raise ValueError(f"K at ε_s = {EPS_S}, δ = {DELTA} is not plan.RECORD_K")
    out = P.lookup(P.load(), stmt, priced_path(s), n, k, layout=layout)
    out["draws"] = plan.RECORD_K
    return out


def wref_cell(n: int, k: int) -> Fraction:
    """One credited cell's reference work at (n, k), every channel kept: the drawn statement's (`ticket_cell`)."""
    from verity.core.protocols.pouw.protocol import Matmul, Shape
    from verity.ml.protocols.pouw.schemes import pearl_c_work as PW
    return clear_work().ticket_cell(scheme(), PW.UnitRecord(Matmul(0, Shape(16, k, n), 0), 1, k))


def compute(n: int, k: int, credited_rows: int, seconds: float, native_rate: float | None, g: dict | None = None) -> dict:
    """W_cert of a chain whose calls credited `credited_rows` rows in all (each n cells wide), over its time `seconds`, and
    the compute residual against `native_rate` (W1 per second; None where no native FP8 chain ran)."""
    g = gamma(n, k) if g is None else g
    wref = wref_cell(n, k) * n * credited_rows
    out = {"wref": float(wref), "gamma": g["published"], "gamma_at": g["published_at"], "pins": g.get("pins", []),
           "eps_s": str(EPS_S), "draws": g["draws"], "error": "(1 − ε_s)^K + ε_TT(q, N) ≤ 2⁻⁴⁰ + ε_TT(q, N)",
           "w_cert": None, "unproved": None}
    if g["published"] is None or g["pins"] != ["pinned"]:
        why = "; ".join(g["missing"]) or f"γ's twins are {', '.join(g['pins'])}, not pinned"
        out["unproved"] = f"no γ ({why})" if g["published"] is None else why
    else:
        out["w_cert"] = float((1 - Fraction(g["published"])) * (1 - EPS_S) * wref)
    out.update(residual(None if native_rate is None else native_rate * seconds, out["wref"], out["w_cert"]))
    return out


def residual(capacity: float | None, reference: float | None, certified: float | None) -> dict:
    """capacity − certified, split into honest overhead (capacity − reference) and soundness slack (reference −
    certified); each part None where a term is missing."""
    sub = lambda a, b: None if a is None or b is None else a - b
    return {"capacity": capacity, "residual": sub(capacity, certified), "honest_overhead": sub(capacity, reference),
            "soundness_slack": sub(reference, certified)}


def bandwidth(read_rate: float | None, seconds: float, bytes_read: float, certified: float = 0.0) -> dict:
    """The bandwidth residual: BW·T against the bytes the chain read and Q_cert, the bytes its accepted windows certified
    (none for a chain run without windows)."""
    capacity = None if read_rate is None else read_rate * seconds
    return {"q_cert": certified, "basis": Q_GRANT, "theorem": Q_THEOREM, **residual(capacity, bytes_read, certified)}


def capacity(memory_bytes: int | None, s_cert: float | None = None, why: str = NO_POUS, stored: float | None = None) -> dict:
    """The capacity residual: the device's memory against the honest store |C| (``stored``) and S_cert, in bytes."""
    return {"s_cert": s_cert, "unproved": None if s_cert is not None else why,
            **residual(None if memory_bytes is None else float(memory_bytes), stored, s_cert)}


def stored(result: dict) -> float | None:
    """|C| in bytes from the result's blocks at its registered scheme's block bits, for a result with no profile."""
    from verity.core.protocols.pous import schemes
    try:
        return result["blocks"] * schemes.get(result["scheme"], segments=result["segments"]).params.m / 8
    except (KeyError, TypeError, ValueError):
        return None


def storage(result: dict | None) -> dict:
    """S_cert from the PoUS harness's ``bench-result.json``: its storage profile's ``state_bound_bits`` as bytes when the
    harness issued a profile (every honest audit accepted) and it is certified; |C| is the profile's blocks at their bits."""
    if result is None:
        return {"s_cert": None, "stored": None, "blocks": None, "unproved": NO_POUS}
    prof = result.get("profile") or {}
    sp = prof.get("storage_profile") or {}
    cert = sp.get("certificate") or {}
    out = {"s_cert": None, "unproved": None, "blocks": sp.get("blocks") or result.get("blocks"),
           "stored": sp["blocks"] * sp["block_bits"] / 8 if sp.get("blocks") else stored(result),
           "theorem": cert.get("theorem"), "delta": cert.get("delta"), "eps_log2": cert.get("eps_log2"),
           "k": sp.get("k"), "limit_ns": sp.get("limit_ns"), "gaps": prof.get("gaps") or [],
           "experimental": prof.get("experimental")}
    audit = (result.get("metrics") or {}).get("audit") or {}
    if not prof:
        out["unproved"] = (f"no profile: {audit.get('audits_accepted')} of {audit.get('audits')} honest audits accepted"
                           if audit else "no profile: the harness ran no honest audit")
    elif not prof.get("certified") or sp.get("state_bound_bits") is None:
        out["unproved"] = "not certified: " + ("; ".join(out["gaps"] + list(sp.get("gaps") or [])) or "no certificate")
    else:
        out["s_cert"] = sp["state_bound_bits"] / 8
    return out
