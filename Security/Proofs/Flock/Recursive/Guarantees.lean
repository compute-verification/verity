import Proofs.Flock.Recursive.Statements
import Proofs.Flock.Soundness.ZK.Hiding

/-!
# Proofs of the recursive audit's guarantees

* `Flock.SecurityProofs.RecursiveSound : Flock.Guarantees.RecursiveSound` is `zk_sessions_recursive_read_off` with V*'s bridge
  (`Flock.Assumptions.VBridge`) taken apart, the inner verifier reading each round's message whole (`same` is equality,
  `Inner.accepts_of_rel_eq`).
* `Flock.SecurityProofs.RecursiveZK : Flock.Guarantees.RecursiveZK` is two hybrids: the outer session's simulator at
  the firewall's commitments (the premise, averaged over the auditor's tape, the developer's randomness and the salts),
  then T1 once per commitment (`ZK.ideal_leaves_swap` under `hash-derived-key`), whose salts the simulated outer session
  no longer reads.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.ZK
open Game
open scoped ENNReal

/-- **The recursive audit is sound.** -/
theorem RecursiveSound : Flock.Guarantees.RecursiveSound := by
  intro n L Reg I Cm S Vs Reg' Reg₂ hS hA3 zs dg N k Rw M hk hRw hM ρ hρ hr t' ht' qF qS qW qR σ x wmsg Sk cmt msg hbr qX
    hX q hCR εin hin
  obtain ⟨hb, hcmt, hdec, hfull⟩ := hbr
  exact zk_sessions_recursive_read_off hS hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg hb qX hX
    (fun _ _ a b => a = b) I.accepts_of_rel_eq Sk cmt msg hcmt hdec hfull q hCR hin

section ZK

/-- The firewall's commitments at the auditor's tape and the developer's randomness `ω` and the salts `y`. -/
def commits {Ωa Ωp : Type} {nh : ℕ} (msgs : Ωa → Ωp → Fin nh → ByteArray) (ω : Ωa × Ωp)
    (y : Fin nh → ZkView.Salt) : Fin nh → ByteArray :=
  fun i => Flock.Guarantees.firewallLeaf (msgs ω.1 ω.2 i) (y i)

/-- The salts and the outer session's randomness, exchanged. -/
def swapLast (A B C : Type) : (A × C) × B ≃ (A × B) × C where
  toFun w := ((w.1.1, w.2), w.1.2)
  invFun w := ((w.1.1, w.2), w.1.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- The developer's randomness, moved first. -/
def pullMid (A P S U : Type) : ((A × P) × S) × U ≃ P × ((A × U) × S) where
  toFun w := (w.1.1.2, ((w.1.1.1, w.2), w.1.2))
  invFun z := (((z.2.1.1, z.1), z.2.2), z.2.1.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **The recursive audit is zero knowledge.** -/
theorem RecursiveZK : Flock.Guarantees.RecursiveZK := by
  intro Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ nh msgs view sim εo hOuter hKey E
  -- the outer session, simulated at the firewall's commitments
  have h1 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, commits msgs w.1.1 w.1.2, view w.1.1.1 (commits msgs w.1.1 w.1.2) w.1.1.2 w.1.2 w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, commits msgs w.1.1 w.1.2, sim w.1.1.1 (commits msgs w.1.1 w.1.2) w.2)) + εo :=
    calc prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
          E (w.1.1.1, commits msgs w.1.1 w.1.2, view w.1.1.1 (commits msgs w.1.1 w.1.2) w.1.1.2 w.1.2 w.2))
        = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun r : Ωo =>
            E (ωy.1.1, commits msgs ωy.1 ωy.2, view ωy.1.1 (commits msgs ωy.1 ωy.2) ωy.1.2 ωy.2 r)) :=
          prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (r : Ωo) =>
            E (ωy.1.1, commits msgs ωy.1 ωy.2, view ωy.1.1 (commits msgs ωy.1 ωy.2) ωy.1.2 ωy.2 r)
      _ ≤ avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => (prCoin fun s : Ωs =>
            E (ωy.1.1, commits msgs ωy.1 ωy.2, sim ωy.1.1 (commits msgs ωy.1 ωy.2) s)) + εo) :=
          avg_mono fun ωy => hOuter ωy.1.1 ωy.1.2 ωy.2 fun v => E (ωy.1.1, commits msgs ωy.1 ωy.2, v)
      _ = avg (fun ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => prCoin fun s : Ωs =>
            E (ωy.1.1, commits msgs ωy.1 ωy.2, sim ωy.1.1 (commits msgs ωy.1 ωy.2) s)) + εo := by
          rw [avg_add, avg_const]
      _ = prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
            E (w.1.1.1, commits msgs w.1.1 w.1.2, sim w.1.1.1 (commits msgs w.1.1 w.1.2) w.2)) + εo :=
          congrArg (· + εo) (prCoin_prod_split fun (ωy : (Ωa × Ωp) × (Fin nh → ZkView.Salt)) (s : Ωs) =>
            E (ωy.1.1, commits msgs ωy.1 ωy.2, sim ωy.1.1 (commits msgs ωy.1 ωy.2) s)).symm
  -- T1 once per commitment: the simulated session reads the salts only through the commitments
  have h2 : prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, commits msgs w.1.1 w.1.2, sim w.1.1.1 (commits msgs w.1.1 w.1.2) w.2)) ≤
      prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, (fun i => Flock.Guarantees.idealLeaf (w.2 i)),
          sim w.1.1.1 (fun i => Flock.Guarantees.idealLeaf (w.2 i)) w.1.2)) + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
    refine (prCoin_congr_equiv (swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωs)
      (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, commits msgs w.1.1 w.1.2, sim w.1.1.1 (commits msgs w.1.1 w.1.2) w.2))).symm.trans_le ?_
    exact ideal_leaves_swap ZkView.My ZkView.saltHash hKey ZkView.leafH nh
      (fun (ω : (Ωa × Ωp) × Ωs) i => ZkView.bitsOf (Sha512.hash (msgs ω.1.1 ω.1.2 i)))
      (fun ω l => E (ω.1.1, l, sim ω.1.1 l ω.2))
  -- the developer's randomness is no longer read
  have h3 : prCoin (fun w : ((Ωa × Ωp) × Ωs) × (Fin nh → ZkView.Bits × ZkView.Salt) =>
        E (w.1.1.1, (fun i => Flock.Guarantees.idealLeaf (w.2 i)),
          sim w.1.1.1 (fun i => Flock.Guarantees.idealLeaf (w.2 i)) w.1.2)) =
      prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
        E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
          sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2)) :=
    calc _ = prCoin (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
          E (z.2.1.1, (fun i => Flock.Guarantees.idealLeaf (z.2.1.2 i)),
            sim z.2.1.1 (fun i => Flock.Guarantees.idealLeaf (z.2.1.2 i)) z.2.2)) :=
          prCoin_congr_equiv (pullMid Ωa Ωp Ωs (Fin nh → ZkView.Bits × ZkView.Salt))
            (fun z : Ωp × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
              E (z.2.1.1, (fun i => Flock.Guarantees.idealLeaf (z.2.1.2 i)),
                sim z.2.1.1 (fun i => Flock.Guarantees.idealLeaf (z.2.1.2 i)) z.2.2))
      _ = avg (fun _ : Ωp => prCoin fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
            E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
              sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2)) :=
          prCoin_prod_split fun (_ : Ωp) (w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
            E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
              sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2)
      _ = _ := avg_const _
  refine h1.trans ((add_le_add h2 (le_refl εo)).trans (le_of_eq ?_))
  rw [h3, add_right_comm]

end ZK

end Flock.SecurityProofs
