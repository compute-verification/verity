import Definitions.Accounting.Guarantees.JointBridge.CommOfPeriodLeakAccepted
import Proofs.NetworkCertifier.Period
import Proofs.Accounting.PouwWork

/-!
# The network certifier's per-period guarantee from its period of sampled audits (proof)

`Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted`: `comm_of_periodLeak` at network accounting's proof of
`PeriodLeakAccepted` (`NetTiming.periodLeak`). On a run that realizes the network certifier's period of `n` sampled
audits (`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), the network audit's `PeriodRun.Sound` holds at
`n · (δ + ε_ks + ε_bind + ε_tap)`. The network certifier accepting with the claim failing means some window fails
`WindowGood` (`comm_iff`); the union bound over the windows sums each window's failure, which is at most the sampled
verifier's `PrBad` at `RLeakTap` (`window_le`), which `PeriodLeakAccepted`'s per-window conjunct bounds by
`κ + √(γ·Pr[B]) + Pr[B_ext] + Pr[B_tap]`, at most `δ + ε_ks + ε_bind + ε_tap` at each window's prover.

The per-window conjunct of `PeriodLeakAccepted` reads neither the period's windows nor the network certifier device,
so it is read off at the period of no windows (`ksLin_of_periodLeak`). Its count of the secrets is what the claim at
`K` stands for, and the error doesn't use it. The network certifier's trust (`trusted-host`) is the realization's
`trusted`, which this proof doesn't read: `window_le` is the per-window bound its records give, and deriving it from
`PeriodLeakAccepted` at the deployed period and `trusted` is the follow-up.
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
    (hks : KS H D) (L Kₒ S : ℕ) (lay : AuditStatement → Layout L S) (k : AuditStatement → ℕ) (δ : ℝ)
    (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H) :
    ∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          (E D decode) m P ≤
        Claims.Sampling.κ D δ m + Real.sqrt (Claims.Sampling.γ D m * PrB H D.V P m) +
          (Bext H D.V (E D decode) Ω P m).Pr H + (Btap H D.V (E D decode) D.commit D.parse tops Ω P m).Pr H := by
  let f : TapRecord Empty Unit (Fin 0 → List (Obs L Kₒ S)) :=
    ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩
  exact (hsrc (n := 0) (Conn := Empty) (Cell := Unit) (Srv := Unit) (Dev := Unit) (Z := Unit) H D hks L Kₒ S lay k δ
    decode hδ hdec reg Dj Fin.elim0 Fin.elim0 (fun i => i.elim0) Fin.elim0 Fin.elim0 Fin.elim0 (fun i => i.elim0)
    (fun _ c => c.elim) f () (fun _ p => tapCode f p.1 (fun c => c.elim)) (fun _ => rfl) (fun c => c.elim) ∅
    (fun _ h => h.elim) (fun _ h => h.elim) ∅ (fun _ => ⟨0, 0, Empty.elim, Empty.elim⟩) (fun _ => ())
    (fun _ h => h.elim) (fun _ => ()) (fun _ h => h.elim) (fun _ => Fin.elim0) (fun _ => Fin.elim0) tops hb).1

/-- `Guarantees.Bridge.CommOfPeriodLeakAccepted` from `PeriodLeakAccepted`. -/
theorem comm_of_periodLeak (hsrc : Guarantees.PeriodLeakAccepted) {ι : Type} {d : Device ι} {α : Type}
    [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} [DecidableEq I] [∀ i, Fintype (Ω i)]
    {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ}
    {lay : AuditStatement → Layout L S} {k : AuditStatement → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : AuditStatement → Finset ℕ}
    {Dj n : ℕ} (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ decode reg Dj n) (hks : KS H D)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, Claims.Sampling.κ D δ (R.stmt ξ i) - δ +
      Real.sqrt (Claims.Sampling.γ D (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V (E D decode) (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V (E D decode) D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap) :
    r.Sound c p .network (n * (δ + εks + εbind + εtap)) := by
  intro ξ
  have hlin := ksLin_of_periodLeak hsrc H D hks L Kₒ S lay k δ decode hδ hdec reg Dj R.tops hb
  calc Pouw.pr (fun x : Ω (r.own .network) =>
        r.accepts .network (Function.update ξ (r.own .network) x) ∧
          ¬ r.Holds c p .network (Function.update ξ (r.own .network) x))
      ≤ Pouw.pr (fun x : Ω (r.own .network) => ∃ i,
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := by
        refine SecurityProofs.PouwWork.pr_mono fun x ⟨ha, hn⟩ => ?_
        have hn' : ¬ r.commWithin c.K (Function.update ξ (r.own .network) x) := hn
        rw [R.comm_iff] at hn'
        obtain ⟨i, hi⟩ := not_forall.mp hn'
        exact ⟨i, ha, hi⟩
    _ ≤ ∑ i, Pouw.pr (fun x : Ω (r.own .network) =>
          r.accepts .network (Function.update ξ (r.own .network) x) ∧
            ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops
              (R.stmt (Function.update ξ (r.own .network) x) i) (R.wit (Function.update ξ (r.own .network) x) i)
              (R.obs (Function.update ξ (r.own .network) x) i)) := pr_exists_le _
    _ ≤ ∑ _i : Fin n, (δ + εks + εbind + εtap) := by
        refine Finset.sum_le_sum fun i _ => (R.window_le ξ i).trans ?_
        have h := hlin (R.stmt ξ i) (R.scope ξ i) (R.Tape ξ i) (R.prover ξ i)
        have he := herr ξ i
        have hx := hbind ξ i
        have ht := htap ξ i
        linarith
    _ = n * (δ + εks + εbind + εtap) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **The network certifier's per-period guarantee**, at network accounting's proof of `PeriodLeakAccepted`. -/
theorem CommOfPeriodLeakAccepted : Guarantees.Bridge.CommOfPeriodLeakAccepted := by
  intro ι d α _ C I Ω _ _ Coin Col T _ _ _ _ _ r c p H D L Kₒ S lay k δ decode reg Dj n R
  exact comm_of_periodLeak NetTiming.periodLeak c p R

end Accounting.SecurityProofs.Bridge
