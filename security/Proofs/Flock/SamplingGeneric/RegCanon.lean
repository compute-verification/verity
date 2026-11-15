module

public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Core.Game.RewindingCanon
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.SamplingGeneric.RegTrace

/-!
# The rewinding lemma at the committed trace, counting each bad run at one gate

`pr_off_le_canon_on` is `pr_off_le` at `Verity.Rewinding.prBad_le_canon_on` in place of `prBad_le`, over any finite
space and a trace that is the table's plurality (`plurAt`) wherever a point opens. At one tape's runs,
`pr_offReg_le_canon`: a run's extractions open a gate of the program off the committed trace with probability at most
`√(N·Pr[conflict])`, `N` the program's gates, where `pr_offReg_le` gives `N·√Pr[conflict]`; `pr_regHit_le_canon`
carries it through `KSReg`'s first half (`event_le_of_off` at the committed trace, `SamplingGeneric/RegTrace.lean`),
so `εreg`'s `N·√Pr[B_conf]` can read `√(N·Pr[B_conf])`. The `_on` forms count only a set `S` of gates holding every
gate some run's extractions open; `pr_regHit_le_canon_pos` takes for `S` the positions the reveals open under the
statement's roots (`opensAt_isSome_mem`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope)

/-- **The rewinding lemma at a table's plurality, one position a bad point** (`Verity.Rewinding.prBad_le_canon_on`):
`pr_off_le` with `√(#T·Pr[conflict])` in place of `N·√Pr[conflict]`, at any set `T` of positions below `N` holding
every position a point opens. -/
theorem pr_off_le_canon_on {S : Type} [Fintype S] [Nonempty S] (N : ℕ) (op : S → ℕ → Option Bool)
    (tr : ℕ → Option Bool) (htr : ∀ a g b, op a g = some b → tr g = some (plurAt op g)) (T : Finset (Fin N))
    (hT : ∀ (a : S) (g : Fin N), (op a g).isSome → g ∈ T) :
    Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b) ≤
      Real.sqrt (#T * Pr fun aa : S × S =>
        ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g) := by
  classical
  have hcard : (0 : ℝ) < Fintype.card S := by exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin N) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b) : ℝ) =
        Pr (fun a : S => op a g = some b) * Fintype.card S := by
    intro g b
    rw [Pr_eq_card_of_mem (fun a : S => op a g = some b)
      (FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b)
      fun a => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  have hpl : FlockSoundness.Rewinding.IsPlurality (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g := by
    intro g v
    have h := le_plur (fun b => Pr fun a : S => op a g = some b) v
    have h' : (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g v) : ℝ) ≤
        #(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g (plurAt op g)) := by
      rw [hcnt, hcnt]
      exact mul_le_mul_of_nonneg_right h hcard.le
    exact_mod_cast h'
  have hbad := Verity.Rewinding.prBad_le_canon_on (op := fun (a : S) (g : Fin N) => op a g)
    (fun g => plurAt op g) hpl T hT
  have hoff : Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b) ≤
      (#(FlockSoundness.Rewinding.bad (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g) : ℝ) /
        Fintype.card S := by
    calc Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b)
        ≤ Pr (fun a : S => ∃ g,
            FlockSoundness.Rewinding.Off (fun (a : S) (g : Fin N) => op a g) (fun g => plurAt op g) a g) :=
          Pr_mono fun a ha => by
            obtain ⟨g, hg, b, hb, hne⟩ := ha
            refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
            rw [htr a g b hb]
            exact (hb.symm.trans h).symm
      _ = _ := Pr_eq_card_of_mem _
          (FlockSoundness.Rewinding.bad (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g)
          fun a => by simp only [FlockSoundness.Rewinding.bad, mem_filter, mem_univ, true_and]
  have hmem : ∀ aa : S × S, aa ∈ FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g) ↔
      ∃ g, FlockSoundness.Rewinding.ConflictAt (fun (a : S) (g : Fin N) => op a g) aa g :=
    fun aa => by simp only [FlockSoundness.Rewinding.conflict, mem_filter, mem_univ, true_and]
  have hconf : (#(FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g)) : ℝ) /
      (Fintype.card S : ℝ) ^ 2 =
      Pr fun aa : S × S => ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g := by
    rw [Pr_eq_card_of_mem
        (fun aa : S × S => ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g)
        (FlockSoundness.Rewinding.conflict (fun (a : S) (g : Fin N) => op a g)) hmem,
      Fintype.card_prod S S, Nat.cast_mul, sq]
  calc Pr (fun a : S => ∃ g < N, ∃ b, op a g = some b ∧ tr g ≠ some b)
      ≤ _ := hoff
    _ ≤ _ := hbad
    _ = Real.sqrt (#T * Pr fun aa : S × S =>
          ∃ g : Fin N, (op aa.1 g).isSome ∧ (op aa.2 g).isSome ∧ op aa.1 g ≠ op aa.2 g) := by
      rw [hconf]

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The rewinding lemma at the committed trace, one gate a bad run**: at each tape, a run's extractions open a gate
of the program off the committed trace with probability at most `√(#S·Pr[conflict])`, at any set `S` of gates holding
every gate some run's extractions open. -/
theorem pr_offReg_le_canon_on (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (S : Finset (Fin (L.nG x)))
    (hS : ∀ (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opensAt H P x ω rc.1 rc.2 g).isSome → g ∈ S) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (#S * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) :=
  pr_off_le_canon_on (L.nG x) (fun (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : ℕ) => L.opensAt H P x ω rc.1 rc.2 g)
    (L.regVal H P x ω) (fun _ _ _ h => L.regVal_eq_plur H P x ω h) S hS

/-- `pr_offReg_le_canon_on` at every gate: `√(N·Pr[conflict])`, `N` the program's gates. -/
theorem pr_offReg_le_canon (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (L.nG x * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr) := by
  have h := L.pr_offReg_le_canon_on H P x ω univ fun _ g _ => mem_univ g
  rwa [card_univ, Fintype.card_fin] at h

set_option synthInstance.maxSize 1024 in
/-- `KSReg`'s first half from the sessions' bound, at `√(#S·Pr[B_conf])` where `εreg` has `N·√Pr[B_conf]`, at any set
`S` of gates holding every gate some run's extractions open: `event_le_of_off` at the committed trace, as
`ksReg_of_sessionKS` is, with `pr_offReg_le_canon_on` in place of `pr_offReg_le`. -/
theorem pr_regHit_le_canon_on {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (S : Finset (Fin (L.nG x)))
    (hSo : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opensAt H P x ω rc.1 rc.2 g).isSome → g ∈ S) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + Real.sqrt (#S * (Bconf H L Ω P x).Pr H) +
        (Bconf H L Ω P x).Pr H := by
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hn : (0 : ℝ) ≤ #S := Nat.cast_nonneg _
  have hC : Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) ≤ Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    (Ex_sqrt_le _ fun _ => Pr_nonneg _).trans (Real.sqrt_le_sqrt (pr_conflict_le hbind hs P))
  have hOω : ∀ ω : Ω, Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (#S) * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr) := fun ω =>
    (L.pr_offReg_le_canon_on H P x ω S (hSo ω)).trans_eq (Real.sqrt_mul hn _)
  have hO : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffReg H P x q.1 q.2.1 q.2.2) ≤ Real.sqrt (#S) * Real.sqrt ((Bconf H L Ω P x).Pr H) :=
    calc Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
          L.OffReg H P x q.1 q.2.1 q.2.2)
        ≤ Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) :=
          Pr_mono fun _ h => h.2
      _ = Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) :=
          Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.OffReg H P x q.1 q.2.1 q.2.2)
      _ ≤ Ex (fun ω : Ω => Real.sqrt (#S) * Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
            ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) := Ex_mono hOω
      _ = Real.sqrt (#S) * Ex (fun ω : Ω => Real.sqrt (Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
            ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr)) := Ex_mul_left _ _
      _ ≤ Real.sqrt (#S) * Real.sqrt ((Bconf H L Ω P x).Pr H) :=
          mul_le_mul_of_nonneg_left hC (Real.sqrt_nonneg _)
  simp only [Freg_eq_FregAt]
  refine (event_le_of_off hS hSK hs hPt hin P (fun ω _ => L.regVal H P x ω) (pr_inConf_le hbind hs P) hO).trans ?_
  rw [Real.sqrt_mul hn]
  linarith

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

/-- **A run's extractions open only what the reveals open**: a gate `opensAt` opens is the position of an opening some
reveal makes under the statement's roots. -/
theorem opensAt_isSome_mem (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (Spos : Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ Spos)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (g : ℕ) (hg : (L.opensAt H P x ω r c g).isSome) : g ∈ Spos := by
  simp only [opensAt, hs] at hg
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  unfold Sampling.Wit.val at hb
  obtain ⟨o, ho, -⟩ := Option.map_eq_some_iff.1 hb
  have hpred := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpred
  obtain ⟨q, hq, hqo⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases hr : q.1 = s.roots
  · rw [if_pos hr] at hqo
    obtain rfl := Option.some.inj hqo
    rw [← hpred.1]
    unfold runExt at hq
    rw [Sampling.flatMapRP_eval] at hq
    obtain ⟨iu, -, hiu⟩ := List.mem_flatMap.1 hq
    split at hiu
    · unfold sessExt at hiu
      rw [(Claims.RP.eval_bind H _ _ _).1] at hiu
      split at hiu
      · simp [RP.eval] at hiu
      · rw [Sampling.flatMapRP_eval] at hiu
        obtain ⟨u', -, hu'⟩ := List.mem_flatMap.1 hiu
        exact hpos _ _ _ q (Sampling.checkRevealed_mem H _ L.commit _ q hu').1 hr
    · simp [RP.eval] at hiu
  · rw [if_neg hr] at hqo
    exact absurd hqo (by simp)

set_option synthInstance.maxSize 1024 in
/-- `pr_regHit_le_canon_on` at the positions the reveals open under the statement's roots: `KSReg`'s first half at
`√(#Spos·Pr[B_conf])`, wherever every reveal under `s.roots` opens a position of `Spos`. -/
theorem pr_regHit_le_canon_pos {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (Spos : Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ Spos) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + Real.sqrt (#Spos * (Bconf H L Ω P x).Pr H) +
        (Bconf H L Ω P x).Pr H := by
  have h := pr_regHit_le_canon_on hbind hS hSK hsc Ω P (univ.filter fun g : Fin (L.nG x) => g.val ∈ Spos)
    fun ω rc g hg => mem_filter.2 ⟨mem_univ _, opensAt_isSome_mem H L P x ω hs Spos hpos rc.1 rc.2 g hg⟩
  have hcard : #(univ.filter fun g : Fin (L.nG x) => g.val ∈ Spos) ≤ #Spos :=
    card_le_card_of_injOn Fin.val (by intro g hg; exact (mem_filter.1 hg).2) (by intro a _ b _ hab; exact Fin.ext hab)
  have hB : 0 ≤ (Bconf H L Ω P x).Pr H := by unfold Finder.Pr; exact Pr_nonneg _
  have hsq := Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right (Nat.cast_le.2 hcard : (_ : ℝ) ≤ _) hB)
  linarith

end Compiled

end Claims.SamplingGeneric
