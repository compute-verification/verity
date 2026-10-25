import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Definitions.NetworkCertifier.Guarantees.Period
import Proofs.Accounting.PouwWork

/-!
# The network certifier's per-period guarantee from its period of sampled audits (proof)

`Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted`: on a run that realizes the network certifier's period of
`n` sampled audits (`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), the network audit's
`PeriodRun.Sound` holds at `n · (δ + ε_ks + ε_bind)`. The network certifier accepting with the claim failing means some
window fails `WindowGood` (`comm_iff`); the union bound over the windows sums each window's failure, which is at most
the sampled verifier's `PrBad` at `RLeak` (`window_le`), which `PeriodLeakAccepted`'s per-window conjunct bounds by
`κ + √(γ·Pr[B]) + Pr[B_ext]`, at most `δ + ε_ks + ε_bind` at each window's prover.

The per-window conjunct of `PeriodLeakAccepted` doesn't read the period's windows or the tap, so it is read off at the
period of no windows (`ksLin_of_periodLeak`). Its count of the secrets is what the claim at `K` stands for, and the
error doesn't use it.
-/

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge Finset

theorem pr_exists_le {β : Type} [Fintype β] {n : ℕ} (P : Fin n → β → Prop) :
    Pouw.pr (fun x => ∃ i, P i x) ≤ ∑ i, Pouw.pr (P i) := by
  classical
  unfold Pouw.pr
  rw [← Finset.sum_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hsub : (univ.filter fun x => ∃ i, P i x) ⊆ univ.biUnion (fun i => univ.filter (P i)) := fun x hx => by
    simp only [mem_filter, mem_biUnion, mem_univ, true_and] at hx ⊢
    exact hx
  have key : ((univ.filter fun x => ∃ i, P i x).card : ℝ) ≤ ∑ i, ((univ.filter (P i)).card : ℝ) := by
    exact_mod_cast (card_le_card hsub).trans card_biUnion_le
  convert key using 3
  exact Finset.filter_congr_decidable _ _ _

/-- `PeriodLeakAccepted`'s per-window conjunct, read off at the period of no windows. -/
theorem ksLin_of_periodLeak (hsrc : Guarantees.PeriodLeakAccepted) {Coin Col T : Type} [Fintype Coin]
    [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T)
    (hks : KS H D) (L Kₒ S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ)
    (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : Stmt → Finset ℕ) (Dj : ℕ) :
    KSLinOn H D.V (ScopeAcc D lay k)
      (RLeak H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
      (E D decode) (Claims.Sampling.κ D δ) (Claims.Sampling.γ D) (fun _ => 1) (Bext H D.V (E D decode)) := by
  let f : TapRecord Empty Unit (Fin 0 → List (Obs L Kₒ S)) :=
    ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ _ _ => funext fun c => c.elim⟩
  exact (hsrc (n := 0) (Conn := Empty) (Cell := Unit) (Srv := Unit) (Dev := Unit) (Z := Unit) H D hks L Kₒ S lay k δ
    decode hδ hdec reg Dj Fin.elim0 Fin.elim0 (fun i => i.elim0) Fin.elim0 Fin.elim0 Fin.elim0 (fun i => i.elim0)
    (fun _ c => c.elim) f () (fun _ p => tapCode f p.1 (fun c => c.elim)) (fun _ => rfl) (fun c => c.elim) ∅
    (fun _ h => h.elim) ∅ (fun _ => ⟨0, 0, Empty.elim, Empty.elim⟩) (fun _ => ()) (fun _ h => h.elim)
    (fun _ => ()) (fun _ h => h.elim) (fun _ => Fin.elim0) (fun _ => Fin.elim0)).1

/-- **The network certifier's per-period guarantee.** On a run that realizes the network certifier's period of `n`
sampled audits at the claim `c.K`, at the deployment `D` where `Claims.Sampling.KS` holds, `δ ≥ 0` and a decoder that
lists every trace within Flock's radius, where each window's prover has `κ − δ + √(γ·Pr[B]) ≤ ε_ks` and
`Pr[B_ext] ≤ ε_bind`, `PeriodLeakAccepted` gives the network audit's `PeriodRun.Sound` at `n · (δ + ε_ks + ε_bind)`. -/
theorem CommOfPeriodLeakAccepted (hsrc : Guarantees.PeriodLeakAccepted) {ι : Type} {d : Device ι} {α : Type}
    [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)]
    {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ}
    {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ}
    {Dj n : ℕ} (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (εks εbind : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind) :
    r.Sound c p .network (n * (δ + εks + εbind)) := by
  intro ξ
  have hlin := ksLin_of_periodLeak hsrc H D hks L Kₒ S lay k δ decode hδ hdec reg Dj
  calc Pouw.pr (fun x : Ω (r.own .network) =>
        r.accepts .network (Function.update ξ (r.own .network) x) ∧
          ¬ r.Holds c p .network (Function.update ξ (r.own .network) x))
      ≤ Pouw.pr (fun x : Ω (r.own .network) => ∃ i,
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) (R.op i)
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := by
        refine SecurityProofs.PouwWork.pr_mono fun x ⟨ha, hn⟩ => ?_
        have hn' : ¬ r.commWithin c.K (Function.update ξ (r.own .network) x) := hn
        rw [R.comm_iff] at hn'
        obtain ⟨i, hi⟩ := not_forall.mp hn'
        exact ⟨i, ha, hi⟩
    _ ≤ ∑ i, Pouw.pr (fun x : Ω (r.own .network) =>
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) (R.op i)
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := pr_exists_le _
    _ ≤ ∑ _i : Fin n, (δ + εks + εbind) := by
        refine Finset.sum_le_sum fun i _ => (R.window_le ξ i).trans ?_
        have h := hlin (R.stmt ξ i) (R.scope ξ i) (R.Tape ξ i) (R.prover ξ i)
        have he := herr ξ i
        have hb := hbind ξ i
        simp only [one_mul] at h
        linarith
    _ = n * (δ + εks + εbind) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Accounting.SecurityProofs.Bridge
