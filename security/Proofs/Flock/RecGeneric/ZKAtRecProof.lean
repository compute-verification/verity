module

public import Proofs.Flock.RecGeneric.ZKAtRec
public import Proofs.Flock.RecGeneric.ZKSessionsProof
public import Proofs.Flock.RecGeneric.ZKHides

/-! Proof of `GenericRecursiveZKAtRecServer`, once `deployRec`'s V\* sessions make up their layout's rounds (`recLayout_shaped`). -/

@[expose] public section

namespace Claims.RecGeneric

open Claims

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

theorem recLayout_shaped (S : ZkSess) (b : RecDeploy.Brings) (msOf : Msg → List (Msg × Out))
    (hk : ∀ xs, S.kS xs = RecDeploy.VO.k xs + 1) : (recLayout S b msOf).Shaped := by
  intro x xO
  rw [recLayout_rounds S b msOf hk x xO]
  exact Finset.sum_congr rfl fun s _ => by rw [Nat.zero_add]

end Claims.RecGeneric

open Claims Claims.RecGeneric Claims.SamplingGeneric in
set_option synthInstance.maxSize 100000 in
set_option synthInstance.maxHeartbeats 1000000 in
theorem Flock.SecurityProofs.GenericRecursiveZKAtRecServer : Flock.Guarantees.GenericRecursiveZKAtRecServer := by
  intro rel T vs t δ₁ ht hT1 hS hV hD hc hsc hT L₀ L hL wk pc fc Xs b msOf enc W F
  exact Flock.SecurityProofs.GenericRecursiveSessionsZK FlockZk.sha (Leaves.hm96 enc) _
    (recLayout (recServerSess rel T vs L wk pc fc t Xs) b msOf) F _ (δ₁ : ℝ)
    (Flock.SecurityProofs.Hm96Hides enc δ₁ hT1)
    (Flock.SecurityProofs.GenericSessionsOfZKOut FlockZk.sha _ (FlockZk.roundsAt (RecDeploy.deployZkRec rel))
      (FlockZk.Tape T) L.toSrv.Chk _ _ _ Xs _
      (server_zkOutRec rel T vs t ht hT1 ENNReal.coe_ne_top hS hV hD hc hsc hT L₀ L hL wk pc fc))
    (recLayout_shaped (recServerSess rel T vs L wk pc fc t Xs) b msOf fun _ => rfl)
