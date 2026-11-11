module

public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Core.Game.RewindingCanon

/-!
# The rewinding lemma at the committed trace, counting each bad run at one gate

`pr_offReg_le` at `Verity.Rewinding.prBad_le_canon` in place of `prBad_le`: a run's extractions open a gate of the
program off the committed trace with probability at most `√(N·Pr[conflict])`, `N` the program's gates, where
`pr_offReg_le` gives `N·√Pr[conflict]`; `pr_regHit_le_canon` carries it through `ksReg_of_sessionKS`'s first half,
so `εreg`'s `N·√Pr[B_conf]` can read `√(N·Pr[B_conf])`. The `_on` forms count only a set `S` of gates holding every
gate some run's extractions open.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The rewinding lemma at the committed trace, one gate a bad run**: at each tape, a run's extractions open a gate
of the program off the committed trace with probability at most `√(#S·Pr[conflict])`, at any set `S` of gates holding
every gate some run's extractions open. -/
theorem pr_offReg_le_canon_on (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (S : Finset (Fin (L.nG x))) (hS : ∀ rc g, (L.opensAt H P x ω rc.1 rc.2 g).isSome → g ∈ S) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (#S * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) := by
  classical
  have hcard : (0 : ℝ) < Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin (L.nG x)) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b) : ℝ) =
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b) *
          Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    intro g b
    rw [Pr_eq_card_of_mem (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b)
      (FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b)
      fun rc => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  have hpl : FlockSoundness.Rewinding.IsPlurality (L.opTable H P x ω) fun g => L.plur H P x ω g := by
    intro g v
    have h := le_plur (fun b => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      L.opensAt H P x ω rc.1 rc.2 g = some b) v
    have h' : (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g v) : ℝ) ≤
        #(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g (L.plur H P x ω g)) := by
      rw [hcnt, hcnt]
      exact mul_le_mul_of_nonneg_right h hcard.le
    exact_mod_cast h'
  have hbad := Verity.Rewinding.prBad_le_canon_on (op := L.opTable H P x ω) (fun g => L.plur H P x ω g) hpl S hS
  have hoff : Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      (#(FlockSoundness.Rewinding.bad (L.opTable H P x ω) fun g => L.plur H P x ω g) : ℝ) /
        Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    calc Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)
        ≤ Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => ∃ g,
            FlockSoundness.Rewinding.Off (L.opTable H P x ω) (fun g => L.plur H P x ω g) rc g) :=
          Pr_mono fun rc hrc => by
            obtain ⟨g, hg, b, hb, hne⟩ := hrc
            refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
            rw [L.regVal_eq_plur H P x ω hb]
            exact (hb.symm.trans h).symm
      _ = _ := Pr_eq_card_of_mem _ (FlockSoundness.Rewinding.bad (L.opTable H P x ω) fun g => L.plur H P x ω g)
          fun rc => by simp only [FlockSoundness.Rewinding.bad, mem_filter, mem_univ, true_and]
  have hconf : (#(FlockSoundness.Rewinding.conflict (L.opTable H P x ω)) : ℝ) /
      (Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) : ℝ) ^ 2 =
      Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr := by
    rw [Pr_eq_card_of_mem (fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) (FlockSoundness.Rewinding.conflict (L.opTable H P x ω))
        fun rr => by simp only [FlockSoundness.Rewinding.conflict, mem_filter, mem_univ, true_and]; exact Iff.rfl,
      Fintype.card_prod ((Fin (L.V.k x) → Coin) × L.SlotEC x) ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      Nat.cast_mul, sq]
  calc Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)
      ≤ _ := hoff
    _ ≤ _ := hbad
    _ = Real.sqrt (#S * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr) := by
      rw [hconf]

/-- `pr_offReg_le_canon_on` at every gate: `√(N·Pr[conflict])`, `N` the program's gates. -/
theorem pr_offReg_le_canon (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (L.nG x * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr) := by
  have h := L.pr_offReg_le_canon_on H P x ω univ fun _ g _ => mem_univ g
  rwa [card_univ, Fintype.card_fin] at h

set_option synthInstance.maxSize 1024 in
/-- `KSReg`'s first half from the sessions' bound, at `√(#S·Pr[B_conf])` where `εreg` has `N·√Pr[B_conf]`, at any set
`S` of gates holding every gate some run's extractions open: the proof of `ksReg_of_sessionKS` with
`pr_offReg_le_canon_on` in place of `pr_offReg_le`. -/
theorem pr_regHit_le_canon_on {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (S : Finset (Fin (L.nG x))) (hSo : ∀ ω rc g, (L.opensAt H P x ω rc.1 rc.2 g).isSome → g ∈ S) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + Real.sqrt (#S * (Bconf H L Ω P x).Pr H) +
        (Bconf H L Ω P x).Pr H := by
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hFlt : ∀ ω, ∀ v ∈ L.Freg H P x ω, v < L.nPU x := by
    intro ω v hv
    simp only [Freg, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hn : (0 : ℝ) ≤ #S := Nat.cast_nonneg _
  have hper : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selReg H P x ω) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        Real.sqrt (#S) * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) := by
    intro ω
    have hIO := (Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)).trans
        (add_le_add le_rfl ((L.pr_offReg_le_canon_on H P x ω S (hSo ω)).trans_eq (Real.sqrt_mul hn _)))
    refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω)).symm.trans_le (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
    obtain ⟨hacc, v, hvl, hvF⟩ := hrc
    obtain ⟨i, hi, u, hiu, huF⟩ := L.selReg_some H P x ω (hS.slots x rc.1) (hFlt ω) hvl hvF
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hiu u.isLt
    rcases reg_split hS hs hPt hin P ω rc.1 rc.2 hacc i u hiu huF i₀ u₀ hlead hu₀ with h | h | h
    · exact Or.inl ⟨hacc, i₀, by rw [leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)
  have hT := L.pr_unfound_le H P β hβ0 hβ (fun ω => L.leadSel x (L.selReg H P x ω))
    fun ω r => hκ ω r (L.leadSel x (L.selReg H P x ω))
  have hI := pr_inConf_le hbind hs P
  have hC : Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) ≤ Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    (Ex_sqrt_le _ fun _ => Pr_nonneg _).trans (Real.sqrt_le_sqrt (pr_conflict_le hbind hs P))
  refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hper).trans ?_)
  rw [Ex_add, Ex_add, Ex_mul_left, Real.sqrt_mul hn]
  linarith [mul_le_mul_of_nonneg_left hC (Real.sqrt_nonneg (#S : ℝ))]

set_option synthInstance.maxSize 1024 in
/-- `pr_regHit_le_canon_on` at every gate: `KSReg`'s first half at `√(N·Pr[B_conf])`. -/
theorem pr_regHit_le_canon {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + Real.sqrt (L.nG x * (Bconf H L Ω P x).Pr H) +
        (Bconf H L Ω P x).Pr H := by
  have h := pr_regHit_le_canon_on hbind hS hSK hsc Ω P univ fun _ _ g _ => mem_univ g
  rwa [card_univ, Fintype.card_fin] at h

end Compiled

end Claims.SamplingGeneric
