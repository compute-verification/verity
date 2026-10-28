import PousCheck.Wire

/-!
# What an accepted timed audit establishes: the certificate gate and the StorageProfile

`verity/core/protocols/pous/protocol.py` (`PousEstablished`, `TimedAudit`, `StorageProfile`) at the scheme
`schemes/p2.py` names (`P2`, `p2-<w>/v4`). P2 has one certificate, at its registered point `(w, c0, blocks per segment) =
(16448, 21065, 2^16)`: `Pous.SecurityProofs.P2SlackFamilyUncond`'s `λ = 128` row, `k = 105` challenges each answered
within no sequential root (`rounds = 0` calls of `ROOT_CALL_NS`), about `W` of 8 segments.

An audit of `k` challenges within `limit = Δ + RTT` is *covered* when the certificate does not refuse it (`refusals`:
`k` below the certificate's, or `limit` fitting a root) and neither does the spec (`deadlineRefusals`: `Δ` other than the
operating point's, an RTT allowance past its cap); `TimedAudit` runs no other audit unless asked for research.
It is *accepted* when its `FINAL` verdict is success and it is covered. The profile's `gaps` are the refusals and a segment
count other than the theorem's, and it is `certified` when there is a certificate and no gap.

`assess` reads the store from the session's manifest and `k`, `Δ` and `RTT` from its `FINAL` (on success `k` is the
record count): the profile is a function of the session's record.
-/

namespace PousCheck.Certificate

open Lean (Json JsonNumber)

/-- `protocol.PousEstablished`: `rho` and `delta` as fractions `(numerator, denominator)`. -/
structure PousEstablished where
  k : Nat
  rounds : Nat
  work : Nat
  callNs : Nat
  theoremName : String
  rho : Nat × Nat := (18, 19)
  delta : Nat × Nat := (1, 100)
  epsLog2 : Int := -128
  cliff : Option Nat := none
  segments : Option Nat := some 1
  hypotheses : List String := []
  claims : List String := []

/-- Half the fastest 16,448-bit root on the allowed hardware, 8 cores of one chiplet cooperating, in ns (`p2.ROOT_CALL_NS`). -/
def ROOT_CALL_NS : Nat := 854029
def POINT : Nat × Nat × Nat := (16448, 21065, 2 ^ 16)
def CERT_BLOCKS : Nat := 2 ^ 19
def CERT_K : Nat := 105
def EXPANDER : String := "sha512-ctr-v4"
def P2_CLAIMS : List String := ["m1-ideal-permutations", "sms-m1-instantiation", "seq-root", "random-oracle",
  "perfect-isolation", "cr/sha-512", "live-verifier", "designated-verifier", "trusted-host"]
/-- Why P2 is experimental (`p2.EXPERIMENTAL`): every profile of it says so. -/
def EXPERIMENTAL : String := "abstract-M1 result (pinned P2SlackFamilyUncond); the P2-M1 instantiation assumption is not " ++
  "accepted, and the XOR-then-reduce compression attack on it is open"

/-- The spec's operating point's `Δ` and its cap on the RTT allowance, in ns (`p2.DELTA_NS`, `p2.RTT_ALLOWANCE_CAP_NS`). -/
def DELTA_NS : Nat := 500000
def RTT_ALLOWANCE_CAP_NS : Nat := 354000

/-- `P2.deadline_refusals`: why the spec refuses this `Δ` and RTT allowance apart, beyond what the certificate reads from
their sum; none when it does not. -/
def deadlineRefusals (delta rtt : Nat) : List String :=
  (if delta != DELTA_NS then [s!"the spec's operating point sets Δ = {DELTA_NS} ns, not {delta}"] else []) ++
  if rtt > RTT_ALLOWANCE_CAP_NS then
    [s!"the RTT allowance of {rtt} ns is past the spec's cap of {RTT_ALLOWANCE_CAP_NS} ns"]
  else []

/-- `P2.certificate`: P2's certificate at `(w, c0, blocks per segment)`, only at the registered point. -/
def p2 (w c0 bps : Nat) : Option PousEstablished :=
  if (w, c0, bps) == POINT then
    some { k := CERT_K, rounds := 0, work := 2 ^ 20, callNs := ROOT_CALL_NS,
           theoremName := "Pous.SecurityProofs.P2SlackFamilyUncond", segments := some (CERT_BLOCKS / POINT.2.2),
           claims := P2_CLAIMS }
  else none

/-- The sequential calls of at least `callNs` that fit in `Δ + RTT`: the effective `D`. -/
def PousEstablished.effectiveRounds (c : PousEstablished) (limit : Nat) : Nat := limit / c.callNs

/-- `PousEstablished.refusals`: why `k` challenges within `limit` ns are outside the certificate; none when inside. -/
def PousEstablished.refusals (c : PousEstablished) (k limit : Nat) : List String :=
  let d := c.effectiveRounds limit
  (if k < c.k then [s!"k = {k} is below the certificate's {c.k}"] else []) ++
  match c.cliff with
  | some cliff =>
    if d > cliff then [s!"Δ + RTT = {limit} ns fits {d} calls of {c.callNs} ns, past the cliff at {cliff}"] else []
  | none =>
    if d > c.rounds then
      [s!"Δ + RTT = {limit} ns fits {d} calls of {c.callNs} ns; the certificate covers {c.rounds}, so Δ + RTT must " ++
       s!"stay under {(c.rounds + 1) * c.callNs} ns"]
    else []

def nat (n : Nat) : Json := .num (JsonNumber.fromNat n)
def opt (n : Option Nat) : Json := match n with | some x => nat x | none => .null
def strs (l : List String) : Json := .arr (l.map Json.str).toArray

def PousEstablished.toJson (c : PousEstablished) : Json := Json.mkObj [
  ("k", nat c.k), ("rounds", nat c.rounds), ("work", nat c.work), ("call_ns", nat c.callNs), ("theorem", .str c.theoremName),
  ("rho", .str s!"{c.rho.1}/{c.rho.2}"), ("delta", .str s!"{c.delta.1}/{c.delta.2}"),
  ("eps_log2", .num (JsonNumber.fromInt c.epsLog2)), ("cliff", opt c.cliff), ("segments", opt c.segments),
  ("hypotheses", strs c.hypotheses), ("claims", strs c.claims)]

/-- The wire's store as the scheme's (`P2Params`): `none` when `P2Params` refuses it. The wire's manifest already has `w` a
multiple of 32 in `(0, 16448]`, `c0 ≡ 1 mod 4` and positive, and `1 ≤ blocks per segment ≤ 2^16`; `P2Params` adds
`c0 < 2^16`. -/
def params (q : Wire.Params) : Option P2.Params :=
  if 1 ≤ q.c0 && q.c0 < 2 ^ 16 && q.w % 8 == 0 && q.w > 16 && q.blocksPerSegment ≤ 2 ^ 16 then
    some ⟨q.w, q.c0.toNat, q.blocksPerSegment, q.segments⟩
  else none

def _root_.PousCheck.P2.Params.name (s : P2.Params) : String := s!"p2-{s.w}/v4"

/-- `P2Params.manifest()`, in its keys' order. -/
def _root_.PousCheck.P2.Params.manifest (s : P2.Params) : List (String × Json) :=
  [("scheme", .str "p2"), ("w", nat s.w), ("c0", nat s.c0), ("profile", .str s.profile),
   ("blocks_per_segment", nat s.blocksPerSegment), ("segments", nat s.segments), ("expander", .str PousCheck.P2.EXPANDER)]

/-- The manifest as Python's `repr` of the dict writes it, for the refusal that names it. -/
def _root_.PousCheck.P2.Params.manifestRepr (s : P2.Params) : String :=
  "{" ++ ", ".intercalate (s.manifest.map fun (k, v) => s!"'{k}': " ++ match v with
    | .str x => s!"'{x}'"
    | j => j.compress) ++ "}"

/-- `TimedAudit`'s gate at `k`, `Δ` and the RTT allowance, for research (`research=True`, so it always has a profile):
the certificate it audits under, its refusals and the spec's, and its other gaps. -/
def gate (s : P2.Params) (k delta rtt : Nat) : Option PousEstablished × List String × List String :=
  match p2 s.w s.c0 s.blocksPerSegment with
  | none => (none, s!"{s.name} has no certificate at {s.manifestRepr}" :: deadlineRefusals delta rtt, [])
  | some c =>
    let uncovered := match c.segments with
      | some n => if s.segments != n then [s!"W has {s.segments} segments; {c.theoremName} is about {n}"] else []
      | none => []
    (some c, c.refusals k (delta + rtt) ++ deadlineRefusals delta rtt, uncovered)

/-- `StorageProfile.to_json()` of an audit of `k` challenges within `Δ + RTT` of the store. -/
def profileJson (s : P2.Params) (k delta rtt : Nat) : Json :=
  let limit := delta + rtt
  let (cert, refused, uncovered) := gate s k delta rtt
  let gaps := refused ++ uncovered
  Json.mkObj [
    ("schema", .str "verity-pous/storage-profile/v1"), ("scheme", .str s.name), ("params", Json.mkObj s.manifest),
    ("blocks", nat s.blocks), ("block_bits", nat s.w), ("k", nat k), ("limit_ns", nat limit),
    ("effective_rounds", opt (cert.map (·.effectiveRounds limit))),
    ("state_bound_bits", opt (cert.map fun c => s.blocks * s.w * c.rho.1 / c.rho.2)),
    ("certificate", match cert with | some c => c.toJson | none => .null),
    ("gaps", strs gaps), ("certified", .bool (cert.isSome && gaps.isEmpty)), ("experimental", .str EXPERIMENTAL)]

/-- What a session's record establishes: its `FINAL`'s verdict, whether the certificate covers its `k` and `Δ + RTT`,
whether it is accepted (both), and, when its verdict is success and `P2Params` takes the store, its StorageProfile. -/
def assess (q : Wire.Params) (final : ByteArray) : Json := Id.run do
  let .ok (verdict, _, records) := Wire.parseFinal final | return Json.mkObj [("verdict", .bool false)]
  if !verdict then return Json.mkObj [("verdict", .bool false), ("accepted", .bool false)]
  let some s := params q | return Json.mkObj [("verdict", .bool true), ("accepted", .bool false), ("profile", .null)]
  let body := final.extract 10 final.size
  let o := Wire.D + Wire.S + 4
  let (delta, rtt) := (Wire.field body o (o + 8), Wire.field body (o + 8) (o + 16))
  let (_, refused, _) := gate s records delta rtt
  let covered := refused.isEmpty
  return Json.mkObj [("verdict", .bool true), ("covered", .bool covered), ("accepted", .bool covered),
    ("profile", profileJson s records delta rtt)]

end PousCheck.Certificate
