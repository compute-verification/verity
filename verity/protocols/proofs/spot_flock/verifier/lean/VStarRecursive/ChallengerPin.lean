/-
The challenger's pin (`Spot.ChallengerPin`) on SPOT's executable (`sampled`) and on a composite's SPOT member (`members`): a
case whose SPOT parameters, as SPOT parsed them, name at any level no challenger, a retired key or another key than the
pin's is refused by name (`reason`).

The window's root needs no second comparison: SPOT draws its coins (`Challenger.sampleCoins`) only from the ledger's open
whose root is the parameters' window root, in a ledger checked under the case's key (`auditedOpen`), withheld on that
root (`withheldOn`). Under the pinned key a root the challenger never opened is refused there.
-/
module

public import Spot.ChallengerPin
public import VStarRecursive.Sampled
meta import VStarRecursive.Sampled

@[expose] public section

namespace VStar.Recursive.ChallengerPin

open Spot.ChallengerPin (Pin keyReason keyReason_none pinnedB pinnedB_accept pinMain)

/-- The challenger key each level of `p` names: one level's, or a two-level case's boundary and interior. -/
def levels : Sampled.Params → List (Option Spot.Challenger.ChallengerKey)
  | .inl p => [p.base.challenger.map (·.1)]
  | .inr p => [p.boundary.challenger.map (·.1), p.interior.challenger.map (·.1)]

/-- Why `p` is refused under `pin`: the first level naming no challenger, a retired key or another key. -/
def reason (pin : Pin) (p : Sampled.Params) : Option String :=
  (levels p).findSome? (keyReason pin)

/-- SPOT's protocol under `pin`, refusing after its own parse. -/
def sampled (pin : Pin) : Core.BlobProtocol := pinnedB Sampled.protocol (reason pin)

/-- What SPOT under `pin` accepts, SPOT parsed to parameters whose every level names the pinned key, not retired: the
parameters its check is handed. -/
theorem sampled_accept (pin : Pin) (params commitment evidence : ByteArray) (blobs : Array ByteArray)
    (h : ((sampled pin).answer params commitment evidence blobs).verdict = .accept) :
    ∃ j p, Core.canonicalJson params = .ok j ∧ Sampled.paramsOfJson j = .ok p ∧
      ∀ k ∈ levels p, ∃ key, k = some key ∧ Spot.sha512Hex key.pub = pin.sha512 ∧
        pin.retired.contains (Flock.hexOf key.pub) = false := by
  obtain ⟨j, p, hj, hp, hw⟩ := pinnedB_accept Sampled.protocol (reason pin) params commitment evidence blobs h
  refine ⟨j, p, hj, hp, fun k hk => keyReason_none ?_⟩
  exact List.findSome?_eq_none_iff.mp hw k hk

/-- A composite's members `cs` with SPOT under `pin`. -/
def members (pin : Pin) (cs : List Core.BlobProtocol) : List Core.BlobProtocol :=
  cs.map fun P => if P.id == Sampled.PROTOCOL then sampled pin else P

theorem members_pinned (pin : Pin) (cs : List Core.BlobProtocol) :
    ∀ Q ∈ members pin cs, Q = sampled pin ∨ (Q ∈ cs ∧ Q.id ≠ Sampled.PROTOCOL) := by
  intro Q hQ
  simp only [members, List.mem_map] at hQ
  obtain ⟨P, hP, rfl⟩ := hQ
  by_cases h1 : P.id = Sampled.PROTOCOL
  · left; simp [h1]
  · right; simp [h1, hP]

/-- The command line of the composite `mk` builds over `cs`, its members under the pin `parse` reads ahead of it. -/
def compositeMain (mk : List Core.BlobProtocol → Core.BlobProtocol) (cs : List Core.BlobProtocol)
    (args : List String) : IO UInt32 :=
  pinMain (fun pin => (mk (members pin cs)).main) (mk cs).main args

end VStar.Recursive.ChallengerPin
