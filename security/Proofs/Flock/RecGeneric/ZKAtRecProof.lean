module

public import Proofs.Flock.RecGeneric.ZKAtRec
public import Proofs.Flock.RecGeneric.ZKSessionsProof
public import Proofs.Flock.RecGeneric.ZKHides

/-!
# Z1-F at `deployRec`'s proof

`deployRec`'s V\* sessions make up their layout's rounds (`recLayout_shaped`) wherever a session's rounds are the outer
verifier's and the `Hello`, since every V\* statement's rounds are its part's numbers' (`RecDeploy.numsOf_vstarAt`).
Then Z1 at sessions (`Flock.SecurityProofs.GenericRecursiveSessionsZK`) at hm96 leaves, whose hiding is Z2
(`Flock.SecurityProofs.Hm96Hides`), and at Flock's ZK server's session system at the recursive outer record, whose zero
knowledge is Z1-L (`Flock.SecurityProofs.GenericSessionsOfZKOut`) at `server_zkOutRec`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims

/-- The rounds of `m` of `deployRec`'s V\* sessions from part `i`, at a session system whose rounds are the outer
verifier's and the `Hello`: each part's numbers' rounds and the `Hello`. -/
theorem recLayout_rounds (S : ZkSess) (b : RecDeploy.Brings) (msOf : Msg → List (Msg × Out))
    (hk : ∀ xs, S.kS xs = RecDeploy.VO.k xs + 1) (x xO : Msg) :
    ∀ m i, (recLayout S b msOf).rounds x xO m i =
      ∑ s ∈ Finset.range m, ((RecDeploy.nums b.st x (i + s)).rounds + 1)
  | 0, _ => by simp [Sessions.rounds]
  | m + 1, i => by
    have e : RecDeploy.VO.k (RecDeploy.vstarAt b.st x (msOf xO) i) = (RecDeploy.nums b.st x i).rounds :=
      congrArg Claims.Recursion.Nums.rounds (RecDeploy.numsOf_vstarAt b.st x (msOf xO) i)
    have h1 : (recLayout S b msOf).rounds x xO (m + 1) i =
        S.kS (RecDeploy.vstarAt b.st x (msOf xO) i) + (recLayout S b msOf).rounds x xO m (i + 1) := rfl
    have hs : ∑ s ∈ Finset.range m, ((RecDeploy.nums b.st x (i + 1 + s)).rounds + 1) =
        ∑ s ∈ Finset.range m, ((RecDeploy.nums b.st x (i + (s + 1))).rounds + 1) :=
      Finset.sum_congr rfl fun s _ => by rw [Nat.add_assoc, Nat.add_comm 1 s]
    have h0 : (RecDeploy.nums b.st x (i + 0)).rounds = (RecDeploy.nums b.st x i).rounds := rfl
    rw [h1, hk, e, recLayout_rounds S b msOf hk x xO m (i + 1), hs, Finset.sum_range_succ']
    omega

/-- **`deployRec`'s V\* sessions make up their layout's rounds** at a session system whose rounds are the outer
verifier's and the `Hello`, at every outer statement. -/
theorem recLayout_shaped (S : ZkSess) (b : RecDeploy.Brings) (msOf : Msg → List (Msg × Out))
    (hk : ∀ xs, S.kS xs = RecDeploy.VO.k xs + 1) : (recLayout S b msOf).Shaped := by
  intro x xO
  rw [recLayout_rounds S b msOf hk x xO]
  exact Finset.sum_congr rfl fun s _ => by rw [Nat.zero_add]

end Claims.RecGeneric

open Claims Claims.RecGeneric Claims.SamplingGeneric in
set_option synthInstance.maxSize 100000 in
set_option synthInstance.maxHeartbeats 1000000 in
/-- **Z1-F at `deployRec`** (`Flock.Guarantees.GenericRecursiveZKAtRecServer`). -/
theorem Flock.SecurityProofs.GenericRecursiveZKAtRecServer : Flock.Guarantees.GenericRecursiveZKAtRecServer := by
  intro rel T vs t δ₁ ht hT1 hS hV hD hc hsc hT L₀ L hL wk pc fc Xs b msOf enc W F
  exact Flock.SecurityProofs.GenericRecursiveSessionsZK FlockZk.sha (Leaves.hm96 enc) _
    (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf) F _ (δ₁ : ℝ)
    (Flock.SecurityProofs.Hm96Hides enc δ₁ hT1)
    (Flock.SecurityProofs.GenericSessionsOfZKOut FlockZk.sha _ (FlockZk.roundsAt (RecDeploy.deployZkRec rel))
      (FlockZk.Tape T) L.toSrv.Chk _ _ _ Xs _
      (server_zkOutRec rel T vs t ht hT1 ENNReal.coe_ne_top hS hV hD hc hsc hT L₀ L hL wk pc fc))
    (recLayout_shaped (recServerSess rel T vs L wk pc fc t Xs) b msOf fun _ => rfl)
