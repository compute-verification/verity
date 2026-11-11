module

public import Proofs.Pouw.DitherInstanceWitness

/-!
# Statement forms the witnesses rule is run on

Not for landing. These statements exist so that the Lean check's rules can be run on them and their verdicts read:
each `WorkSoundAt_form…` is `WorkSoundAt`'s statement under another name, so that the lock can give it a `witnesses`
entry of its own, with one witness varied and the other `WorkSoundAt`'s.

* `WorkSoundAt_formAt`: the nontrivial witness is `WorkSoundAt_nontrivial_at`, the failure at the honest instance with
  the premises under an `∃`.
* `WorkSoundAt_formAnd`: the nontrivial witness is `WorkSoundAt_nontrivial_and`, the same with the premises as a
  conjunct, `(premises) ∧ ¬ WorkBound …`.
* `WorkSoundAt_formNoHosts`: the honest witness is `WorkSoundAt_honest_noHosts`, which leaves out `HostsCounted`.
* `WorkSoundAt_formLam`: the honest witness is `WorkSoundAt_honest_lam`, which binds `DualOpt`'s credited work by
  `∃` where `WorkSoundAt` writes `1`.
* `WorkSoundAt_formNamed`: the honest witness is `WorkSoundAt_honest_named`, whose statement names the instance its
  proof uses, proof-only definitions among them, in place of binding it by `∃`.
-/

@[expose] public section

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance Pouw.Witness

open scoped NNRat

/-- The nontrivial statement at the honest instance with the premises as a conjunct. -/
theorem WorkSoundAt_nontrivial_and :
    ∃ (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) (D : Devices c) (G : Accel)
      (L : Layout (D.machine x) twRound) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G)
      (μ : G.Res → ℝ),
      (c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1) ∧
      ¬ WorkBound G twRound 2 (Obs := Unit) (fun _ _ _ _ => True) (fun _ _ _ => ()) Pd (fun _ => 1 / 2)
        (∑ r, μ r * (1 * Pd.hon r)) (fun _ => 1 / 2) (fun _ => True) := by
  obtain ⟨c, pb, hr, x, D, G, L, e, Pd, μ, h, hn⟩ := WorkSoundAt_nontrivial_at
  exact ⟨c, pb, hr, x, D, G, L, e, Pd, μ, h, hn⟩

/-- The honest statement without `HostsCounted`. -/
theorem WorkSoundAt_honest_noHosts :
    ∃ (c : Verity.Accel.Card) (pb : PubBest c) (x : Verity.Accel.Open) (D : Devices c) (G : Accel)
      (L : Layout (D.machine x) twRound) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G)
      (μ : G.Res → ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ CallsAtPub G e pb ∧
        G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ 1 := by
  obtain ⟨c, pb, _, x, D, G, L, e, Pd, μ, h1, h2, h3, -, h5, h6, h7, h8, h9⟩ := WorkSoundAt_honest
  exact ⟨c, pb, x, D, G, L, e, Pd, μ, h1, h2, h3, h5, h6, h7, h8, h9⟩

/-- The honest statement with `DualOpt`'s credited work bound by `∃`. -/
theorem WorkSoundAt_honest_lam :
    ∃ (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) (D : Devices c) (G : Accel)
      (L : Layout (D.machine x) twRound) (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res) (Pd : Period G)
      (μ : G.Res → ℝ) (lam : ℝ),
      c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ OnCard G c x D twRound L e ∧ HostsCounted D hr ∧
        CallsAtPub G e pb ∧ G.G1 ∧ G.G2 ∧ G.G3 ∧ Pd.DualOpt μ lam := by
  obtain ⟨c, pb, hr, x, D, G, L, e, Pd, μ, h⟩ := WorkSoundAt_honest
  exact ⟨c, pb, hr, x, D, G, L, e, Pd, μ, 1, h⟩

/-- The honest statement with its instance named: the H200's table, menu and host row, and the proof-only `s8Open`,
`devs`, `accelAt`, `idLayout` and `halfPd`. -/
theorem WorkSoundAt_honest_named :
    Verity.Accel.H200.card.Exact Verity.Accel.s8Open ∧ Verity.Assumptions.AccelProductForm Verity.Accel.s8Open ∧
      OnCard (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32)))
        Verity.Accel.H200.card Verity.Accel.s8Open (devs Verity.Accel.H200.card hostH200) twRound
        (idLayout Verity.Accel.H200.card Verity.Accel.s8Open hostH200) (Equiv.refl _) ∧
      HostsCounted (devs Verity.Accel.H200.card hostH200) hostH200 ∧
      CallsAtPub (D := devs Verity.Accel.H200.card hostH200)
        (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32))) (Equiv.refl _)
        pubBestH200 ∧
      (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32))).G1 ∧
      (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32))).G2 ∧
      (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32))).G3 ∧
      (halfPd (accelAt Verity.Accel.H200.card Verity.Accel.s8Open hostH200 (shaUnits 3400 152 (3608 / 32)))
        fun _ => 1).DualOpt (fun _ => 1) 1 :=
  honest_at Verity.Accel.H200.card h200_rates_pos hostH200 hostH200_pos pubBestH200 _ (List.Mem.head _)

/-- `WorkSoundAt`'s statement, to carry `WorkSoundAt_nontrivial_at` as its nontrivial witness. -/
theorem WorkSoundAt_formAt : Pouw.Guarantees.DitherInstance.WorkSoundAt := WorkSoundAt

/-- `WorkSoundAt`'s statement, to carry `WorkSoundAt_nontrivial_and` as its nontrivial witness. -/
theorem WorkSoundAt_formAnd : Pouw.Guarantees.DitherInstance.WorkSoundAt := WorkSoundAt

/-- `WorkSoundAt`'s statement, to carry `WorkSoundAt_honest_noHosts` as its honest witness. -/
theorem WorkSoundAt_formNoHosts : Pouw.Guarantees.DitherInstance.WorkSoundAt := WorkSoundAt

/-- `WorkSoundAt`'s statement, to carry `WorkSoundAt_honest_lam` as its honest witness. -/
theorem WorkSoundAt_formLam : Pouw.Guarantees.DitherInstance.WorkSoundAt := WorkSoundAt

/-- `WorkSoundAt`'s statement, to carry `WorkSoundAt_honest_named` as its honest witness. -/
theorem WorkSoundAt_formNamed : Pouw.Guarantees.DitherInstance.WorkSoundAt := WorkSoundAt

end Pouw.SecurityProofs.DitherInstance
