module

public import Proofs.Flock.Recursive.Statements
public import Proofs.Flock.Recursive.StopLeak
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Recursive.Compose
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# Proofs of the recursive audit's guarantees

* `Flock.SecurityProofs.RecursiveSound : Flock.Guarantees.RecursiveSound` (retired from the lock: the
  recursion's guarantee is `Flock.Guarantees.RecursiveKS`) is `zk_sessions_recursive_fork_read_off` with
  V*'s bridge
  (`Flock.Assumptions.VBridge`) taken apart, the inner verifier reading each round's message whole (`same` is equality,
  `Inner.accepts_of_rel_eq`). It reads no A3, so it proves `Flock.Guarantees.RecursiveSoundNoA3`
  (`Flock.SecurityProofs.RecursiveSoundNoA3`), and `RecursiveSound` from it.
* `Flock.SecurityProofs.RecursiveZK : Flock.Guarantees.RecursiveZK` is, for the whole view, two hybrids: the outer
  session's simulator at the ZK server's commitments (the premise, averaged over the auditor's tape, the developer's
  randomness and the salts), then T1 once per commitment (`ZK.ideal_leaves_swap` at the pinned key's hiding gap,
  `CROnly.hm96Hiding_gap`), whose salts the simulated outer session no longer reads. `recursiveZK_le` is the same at any
  hm96 hiding bound `δ`; `Proofs.Flock.Recursive.OsSalts` takes it to the salts the operating system gives (A3). A view
  released cut at a stop is then `prCoin_cut_le` (`Proofs.Flock.Recursive.StopLeak`) over the whole view's bound, and
  `recursiveZK_whole` is the guarantee at one stop, the whole view's bound.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.ZK
open Game
open scoped ENNReal

/-- **The recursive audit is sound, with no A3.** -/
theorem RecursiveSoundNoA3 : Flock.Guarantees.RecursiveSoundNoA3 := by
  intro n L Reg I Cm S Vs Reg' Reg₂ zs dg N k Rw M hk hRw hM ρ hρ hr σ x wmsg Sk cmt msg hbr εin hin
  obtain ⟨hb, hcmt, hdec, hfull⟩ := hbr
  exact zk_sessions_recursive_fork_read_off zs dg N hk hRw hM hρ hr σ x wmsg hb (fun _ _ a b => a = b)
    I.accepts_of_rel_eq Sk cmt msg hcmt hdec hfull hin

/-- **The recursive audit is sound.** -/
theorem RecursiveSound : Flock.Guarantees.RecursiveSound := by
  intro n L Reg I Cm S Vs Reg' Reg₂ _hA3 zs dg N k Rw M hk hRw hM ρ hρ hr σ x wmsg Sk cmt msg hbr εin hin
  exact RecursiveSoundNoA3 zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr hin

section ZK

/-- The ZK server's commitments at the auditor's tape and the developer's randomness `ω` and the salts `y`. -/
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

/-- **`RecursiveZK` at any hm96 hiding bound**: `Flock.Guarantees.RecursiveZK` with `Claims.Hm96Hiding` at `δ`
taken as a hypothesis, and `nh·δ` in place of `nh·hidingGap My saltHash`. -/
theorem recursiveZK_le {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo]
    [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ) (msgs : Ωa → Ωp → Fin nh → ByteArray)
    (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo δ : ℝ≥0∞}
    (hOuter : ∀ a p y (E : V → Prop),
      prCoin (fun r : Ωo => E (view a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) p y r))
        ≤ prCoin (fun s : Ωs => E (sim a (fun i => Flock.Guarantees.firewallLeaf (msgs a p i) (y i)) s)) + εo)
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Ωa × (Fin nh → ByteArray) × V → Prop) :
    prCoin (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
        E (w.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)),
          view w.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs w.1.1.1 w.1.1.2 i) (w.1.2 i)) w.1.1.2 w.1.2 w.2))
      ≤ prCoin (fun w : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
          E (w.1.1, (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)),
            sim w.1.1 (fun i => Flock.Guarantees.idealLeaf (w.1.2 i)) w.2))
        + εo + nh * δ := by
  -- the outer session, simulated at the ZK server's commitments
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
          sim w.1.1.1 (fun i => Flock.Guarantees.idealLeaf (w.2 i)) w.1.2)) + nh * δ := by
    refine (prCoin_congr_equiv (swapLast (Ωa × Ωp) (Fin nh → ZkView.Salt) Ωs)
      (fun w : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωs =>
        E (w.1.1.1, commits msgs w.1.1 w.1.2, sim w.1.1.1 (commits msgs w.1.1 w.1.2) w.2))).symm.trans_le ?_
    exact ideal_leaves_swap ZkView.My ZkView.saltHash hT1 ZkView.leafH nh
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

/-- **The recursive audit is zero knowledge, up to where it stops**, at the pinned key's hiding gap, where hm96 hiding
holds with no premise (`CROnly.hm96Hiding_gap`): the whole view's bound (`recursiveZK_le`), then the stop's count
(`prCoin_cut_le`), with the gaps added once. -/
theorem RecursiveZK : Flock.Guarantees.RecursiveZK := by
  intro Ωa Ωp Ωo Ωs V St Ωk W _ _ _ _ _ _ _ _ _ _ _ _ nh msgs view sim εo hOuter cut stop E
  refine (prCoin_cut_le (Flock.Guarantees.recView nh msgs view) (Flock.Guarantees.recIdeal nh sim) cut stop
    (fun E => (recursiveZK_le nh msgs view sim hOuter (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) E).trans
      (le_of_eq (add_assoc _ _ _))) E).trans (le_of_eq (add_assoc _ _ _).symm)

/-- **`RecursiveZK` at one stop is the whole view's bound**: at `St = Ωk = Unit` and `cut` the identity,
`Flock.Guarantees.RecursiveZK` gives `recursiveZK_le` at the pinned key's hiding gap. -/
theorem recursiveZK_whole (h : Flock.Guarantees.RecursiveZK) {Ωa Ωp Ωo Ωs V : Type} [Fintype Ωa] [Nonempty Ωa]
    [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs] (nh : ℕ)
    (msgs : Ωa → Ωp → Fin nh → ByteArray) (view : Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : Flock.Guarantees.OuterViewZK nh msgs view sim εo) (E : Ωa × (Fin nh → ByteArray) × V → Prop) :
    prCoin (fun w => E (Flock.Guarantees.recView nh msgs view w)) ≤
      prCoin (fun w => E (Flock.Guarantees.recIdeal nh sim w)) + εo +
        nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) := by
  have hu := h nh msgs view sim hOuter (fun (_ : Unit) (t : Ωa × (Fin nh → ByteArray) × V) => t)
    (fun _ : (((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo) × Unit => ()) E
  have hsnd : prCoin (fun u : Unit × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs) =>
      E (Flock.Guarantees.recIdeal nh sim u.2)) = prCoin (fun w => E (Flock.Guarantees.recIdeal nh sim w)) := by
    rw [prCoin_prod_split fun (_ : Unit) w => E (Flock.Guarantees.recIdeal nh sim w)]
    exact avg_const _
  rwa [prCoin_fst (fun w => E (Flock.Guarantees.recView nh msgs view w)), Fintype.card_unit, Nat.cast_one, one_mul,
    hsnd] at hu

end ZK

end Flock.SecurityProofs
