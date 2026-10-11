"""What a point's arms certify, and each resource's covert residual.  Pure (stdlib and `verity`), so it runs on a CPU.

* **W_cert, compute.**  `Pouw.Guarantees.EndToEnd` (``Security/Models/Pouw/Guarantees/Game.lean``) bounds the chance that an
  audit of t draws accepts a transcript whose prover spent T < (1 − γ)(1 − ε_s)·ΣW_ref by (1 − ε_s)^t + ε_TT(q, N).  So
  W_cert = (1 − γ)(1 − ε_s)·ΣW_ref, in W1 units (one dense FP8 MAC on the device), with ΣW_ref the drawn statement's
  reference work over the credited cells (`pearl_c_work.ticket_cell`; about 1.09 to 1.11 per MAC at the sweep's shapes).
  γ is looked up from the Lean-pinned price twins (`price_twins.lookup`) of the statement on the priced path the scheme's
  draw cites (`pearl_c_work.draw_wref`), at both FP32 prices, the larger published; it is never typed.  Twins are stated
  at the cubes 8,192³ and 16,384³ only, so every other shape's W_cert is unproved.  ε_s = 0.1% at t = K = 27,713 draws
  (`plan.RECORD_K`, (1 − ε_s)^K ≤ 2⁻⁴⁰).  A chain's credited rows are its screened rows less those the screen failed or
  left unsure, a lower bound on the rows the records credit.
* **Q_cert, bandwidth**: no pinned theorem yet.
* **S_cert, capacity**: the PoUS arm's storage profile (`storage`), when the harness issued one and it is certified: a
  server holding fewer than ⌊ρ·|C|⌋ bits passes the audit of record's k challenges within its limit with probability at most
  δ + 2^eps_log2, by the certificate's theorem (P2's is ``Pous.SecurityProofs.P2SlackFamilyUncond``), under its claims,
  with P2 itself experimental (the profile says why).  S_cert is that bound in bytes; otherwise it is unproved, with why.
* **The covert residual** of a resource over an arm's chain time T is what the device could have done in T less what is
  certified: compute F·T − W_cert, with F the native FP8 chain's rate at the shape (W1 per second); bandwidth BW·T − Q_cert,
  BW the native read rate; capacity M − S_cert, M the device's memory.  It splits into honest overhead, native less the
  honest reference (F·T − ΣW_ref: what the protocol's own time costs; M − |C|, the memory the store leaves), and soundness
  slack, the reference less what is certified (ΣW_ref − W_cert; |C| − S_cert).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TWINS = ROOT / "benchmarks" / "pouw" / "harness" / "price_twins.py"
#: the panel line and version Pearl-C's arms run (`price_twins.statement` drops the hashing format)
SCHEME, LINE, VERSION = "pearl-c-sm120-v1-h3", "pearl-c-sm120", "v1-h3"
EPS_S = Fraction(1, 1000)
DELTA = Fraction(1, 2 ** 40)
UNPROVED_Q = "unproved: no bandwidth guarantee is pinned"
NO_POUS = "no PoUS arm in this run"


def price_twins():
    """The harness's `price_twins` module, loaded by path (benchmarks/pouw/harness is not a package)."""
    if "price_twins" not in sys.modules:
        spec = importlib.util.spec_from_file_location("price_twins", TWINS)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["price_twins"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["price_twins"]


def scheme():
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
    from verity.core.protocols.pouw.serving import SchemeId
    sid = SchemeId.parse(SCHEME)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing)
    if s.name != SCHEME:
        raise ValueError(f"the scheme built for {SCHEME!r} names itself {s.name!r}")
    return s


def priced_path(s) -> str:
    """The priced path the scheme's draw cites, from its statement's name (``...Cast8p72...`` is ``cast-8.72``)."""
    from verity.core.protocols.pouw.schemes import pearl_c_work as PW
    at = PW.draw_wref(s)
    if at is None:
        raise ValueError(f"{s.name}: its draw cites no statement")
    m = re.search(r"Cast(8p72|32p06|16|32|8)", at.statement)
    if m is None:
        raise ValueError(f"{at.statement}: no cast in the statement's name")
    return price_twins().CASTS[m.group(1)]


def gamma(n: int, k: int) -> dict:
    """γ for Pearl-C at (n, k): `price_twins.lookup`'s record (both prices, the larger published, what's missing), with the
    statement's cap held to the scheme's and K to ε_s and δ."""
    from verity.core.protocols.pouw.circuit import plan
    from verity.core.protocols.pouw.schemes import pearl_c_work as PW
    P, s = price_twins(), scheme()
    stmt = P.statement(LINE, VERSION)
    if P.CAPS[stmt] != s.cap or PW.draw_wref(s).cap != s.cap:
        raise ValueError(f"{stmt}: the twins' cap {P.CAPS[stmt]} is not the scheme's {s.cap}")
    if plan.record_k(EPS_S, DELTA) != plan.RECORD_K:
        raise ValueError(f"K at ε_s = {EPS_S}, δ = {DELTA} is not plan.RECORD_K")
    out = P.lookup(P.load(), stmt, priced_path(s), n, k)
    out["draws"] = plan.RECORD_K
    return out


def wref_cell(n: int, k: int) -> Fraction:
    """One credited cell's reference work at (n, k), every channel kept: the drawn statement's (`ticket_cell`)."""
    from verity.core.protocols.pouw.protocol import Matmul, Shape
    from verity.core.protocols.pouw.schemes import pearl_c_work as PW
    return PW.ticket_cell(scheme(), PW.UnitRecord(Matmul(0, Shape(16, k, n), 0), 1, k))


def compute(n: int, k: int, credited_rows: int, seconds: float, native_rate: float | None, g: dict | None = None) -> dict:
    """W_cert of a chain whose calls credited `credited_rows` rows in all (each n cells wide), over its time `seconds`, and
    the compute residual against `native_rate` (W1 per second; None where no native FP8 chain ran)."""
    g = gamma(n, k) if g is None else g
    wref = wref_cell(n, k) * n * credited_rows
    out = {"wref": float(wref), "gamma": g["published"], "gamma_at": g["published_at"], "pins": g.get("pins", []),
           "eps_s": str(EPS_S), "draws": g["draws"], "error": "(1 − ε_s)^K + ε_TT(q, N) ≤ 2⁻⁴⁰ + ε_TT(q, N)",
           "w_cert": None, "unproved": None}
    if g["published"] is None or g["pins"] != ["pinned"]:
        out["unproved"] = "; ".join(g["missing"]) or f"γ's twins are {', '.join(g['pins'])}, not pinned"
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


def bandwidth(read_rate: float | None, seconds: float, bytes_read: float) -> dict:
    """The bandwidth residual: BW·T against the bytes the chain read; Q_cert is unproved."""
    capacity = None if read_rate is None else read_rate * seconds
    return {"q_cert": None, "unproved": UNPROVED_Q, **residual(capacity, bytes_read, None)}


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
        return {"s_cert": None, "stored": None, "unproved": NO_POUS}
    prof = result.get("profile") or {}
    sp = prof.get("storage_profile") or {}
    cert = sp.get("certificate") or {}
    out = {"s_cert": None, "unproved": None, "stored": sp["blocks"] * sp["block_bits"] / 8 if sp.get("blocks") else stored(result),
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
