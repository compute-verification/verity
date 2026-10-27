import Lean.Data.Json

/-!
# P2's certificate

`verity/core/protocols/pous/protocol.py` (`Certificate`) at the scheme `schemes/p2.py` names (`P2`, `p2-<w>/v4`). P2 has
one certificate, at its registered point `(w, c0, blocks per segment) = (16448, 21065, 2^16)`:
`Pous.SecurityProofs.P2SlackFamilyUncond`'s `λ = 128` row, `k = 105` challenges each answered within no sequential root
(`rounds = 0` calls of `ROOT_CALL_NS`), about `W` of 8 segments.

A session of `k` rounds within `limit = Δ + RTT` is *covered* when the certificate does not refuse it (`refusals`: `k`
below the certificate's, or `limit` fitting a root). `PousCheck.RunProtocol` reads the certificate at a run's point.
-/

namespace PousCheck.Certificate

open Lean (Json JsonNumber)

/-- `protocol.Certificate`: `rho` and `delta` as fractions `(numerator, denominator)`. -/
structure Certificate where
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
def P2_CLAIMS : List String := ["m1-ideal-permutations", "sms-m1-instantiation", "seq-root", "random-oracle",
  "perfect-isolation", "cr/sha-512", "live-verifier", "designated-verifier", "trusted-host"]
/-- Why P2 is experimental (`p2.EXPERIMENTAL`): every profile of it says so. -/
def EXPERIMENTAL : String := "abstract-M1 result (pinned P2SlackFamilyUncond); the P2-M1 instantiation assumption is not " ++
  "accepted, and the XOR-then-reduce compression attack on it is open"

/-- `P2.certificate`: P2's certificate at `(w, c0, blocks per segment)`, only at the registered point. -/
def p2 (w c0 bps : Nat) : Option Certificate :=
  if (w, c0, bps) == POINT then
    some { k := CERT_K, rounds := 0, work := 2 ^ 20, callNs := ROOT_CALL_NS,
           theoremName := "Pous.SecurityProofs.P2SlackFamilyUncond", segments := some (CERT_BLOCKS / POINT.2.2),
           claims := P2_CLAIMS }
  else none

/-- The sequential calls of at least `callNs` that fit in `Δ + RTT`: the effective `D`. -/
def Certificate.effectiveRounds (c : Certificate) (limit : Nat) : Nat := limit / c.callNs

/-- `Certificate.refusals`: why `k` challenges within `limit` ns are outside the certificate; none when inside. -/
def Certificate.refusals (c : Certificate) (k limit : Nat) : List String :=
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

def Certificate.toJson (c : Certificate) : Json := Json.mkObj [
  ("k", nat c.k), ("rounds", nat c.rounds), ("work", nat c.work), ("call_ns", nat c.callNs), ("theorem", .str c.theoremName),
  ("rho", .str s!"{c.rho.1}/{c.rho.2}"), ("delta", .str s!"{c.delta.1}/{c.delta.2}"),
  ("eps_log2", .num (JsonNumber.fromInt c.epsLog2)), ("cliff", opt c.cliff), ("segments", opt c.segments),
  ("hypotheses", strs c.hypotheses), ("claims", strs c.claims)]

end PousCheck.Certificate
