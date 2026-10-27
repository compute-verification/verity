import Definitions.Accounting.Guarantees.JointLive
import Proofs.Core.Game.Prob

/-!
# Proof of the joint guarantee over a causal game

Each subprotocol's per-period guarantee over the causal game (`PeriodLive.Sound`) bounds its failure at every strategy
of the whole game, so at a strategy the union bound over the subprotocols, the periods, the wipes and the memory terms
sums the errors (`prob_exists_le`, `Verity.Game.prob_or_le`). Nothing here reads how the coins interleave: that is in
each subprotocol's own guarantee.
-/

namespace Accounting.SecurityProofs

open Accounting Verity Verity.Game

/-- **The union bound at a strategy**, over a finite family of events. -/
theorem Live.prob_exists_le {O ι : Type} (t : Finset ι) (p : ι → O → Prop) (G : Game O) (s : Strategy G) :
    prob (fun o => ∃ i ∈ t, p i o) G s ≤ ∑ i ∈ t, prob (p i) G s := by
  classical
  induction t using Finset.induction_on with
  | empty => simp [prob_false]
  | insert j t hj ih =>
      rw [Finset.sum_insert hj]
      calc prob (fun o => ∃ i ∈ insert j t, p i o) G s
          ≤ prob (fun o => p j o ∨ ∃ i ∈ t, p i o) G s :=
            prob_mono (fun o ⟨i, hi, hpi⟩ => by
              rcases Finset.mem_insert.1 hi with rfl | hi
              · exact Or.inl hpi
              · exact Or.inr ⟨i, hi, hpi⟩) G s
        _ ≤ prob (p j) G s + prob (fun o => ∃ i ∈ t, p i o) G s := prob_or_le _ _ G s
        _ ≤ prob (p j) G s + ∑ i ∈ t, prob (p i) G s := add_le_add le_rfl ih

/-- The union bound at a strategy, over a finite type. -/
theorem Live.prob_exists_univ_le {O ι : Type} [Fintype ι] (p : ι → O → Prop) (G : Game O) (s : Strategy G) :
    prob (fun o => ∃ i, p i o) G s ≤ ∑ i, prob (p i) G s :=
  (prob_mono (fun _ ⟨i, hi⟩ => ⟨i, Finset.mem_univ i, hi⟩) G s).trans (Live.prob_exists_le Finset.univ p G s)

section Opened

variable {C : Verity.Circuit} {n : ℕ}

/-- A wrong unit is unbound. -/
theorem wrong_subset_unbound (P : Verity.Partition C n) (opens : (g : Fin C.N) → BitVec (C.w g) → Prop)
    (X : C.Transcript) : P.wrong X ⊆ unbound P opens X := by
  classical
  intro u hu
  simp only [Verity.Partition.wrong, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  simp only [unbound, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inl hu

/-- Some transcript the openings open to: at each position an opened value where there is one. -/
theorem exists_opensTo (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) : ∃ X : C.Transcript, OpensTo opens X := by
  classical
  refine ⟨fun g => if h : ∃ v, opens g v then h.choose else 0, fun g hg => ?_⟩
  show opens g (if h : ∃ v, opens g v then h.choose else 0)
  split_ifs
  exact hg.choose_spec

/-- **More openings, with no two of one position, bind more.** If every opening of `opens` is one of `opens'` and
`opens'` has no two openings of one position, every transcript `opens'` opens to is one `opens` opens to, and binds
every unit it binds there: the integrity claim at `opens` gives it at `opens'`. A subprotocol's bound over the runs of
one strategy it plays (the other subprotocols' coins fixed) so carries to the openings of every run of the game. -/
theorem integrityHolds_mono (P : Verity.Partition C n) {opens opens' : (g : Fin C.N) → BitVec (C.w g) → Prop}
    (hle : ∀ g v, opens g v → opens' g v) (hd : ¬ DoubleOpen opens') {ε : ℝ} (h : IntegrityHolds P opens ε) :
    IntegrityHolds P opens' ε := by
  classical
  intro X hX
  have hX' : OpensTo opens X := fun g ⟨v, hv⟩ => by
    have hv' := hle g v hv
    have hx := hX g ⟨v, hv'⟩
    by_cases hxv : v = X g
    · exact hxv ▸ hv
    · exact absurd ⟨g, v, X g, hv', hx, hxv⟩ hd
  refine le_trans ?_ (h X hX')
  exact_mod_cast Finset.card_le_card fun u hu => by
    simp only [unbound, Finset.mem_filter, Finset.mem_univ, true_and] at hu ⊢
    rcases hu with hu | ⟨g, hg, hn⟩
    · exact Or.inl hu
    · exact Or.inr ⟨g, hg, fun ho => hn (hle g _ ho)⟩

end Opened

/-- **The joint guarantee over a period played as a causal game** (`Accounting.Guarantees.JointPeriodLive`). -/
theorem JointPeriodLive : Accounting.Guarantees.JointPeriodLive := by
  intro ι α _ d C Cm O r G c p δ h s
  calc prob (fun o => (∃ a, r.accepts a o ∧ ¬ r.Holds c p G s a o) ∧ ¬ r.Break G s o) G s
      ≤ prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c p G s a o ∧ ¬ r.Break G s o) G s :=
        prob_mono (fun o ⟨⟨a, ha, hh⟩, hb⟩ => ⟨a, ha, hh, hb⟩) G s
    _ ≤ ∑ a, prob (fun o => r.accepts a o ∧ ¬ r.Holds c p G s a o ∧ ¬ r.Break G s o) G s :=
        Live.prob_exists_univ_le _ G s
    _ ≤ ∑ a, ENNReal.ofReal (δ a) := Finset.sum_le_sum fun a _ => h a s

/-- **The joint guarantee over a horizon of periods played as one causal game**
(`Accounting.Guarantees.JointHorizonLive`). -/
theorem JointHorizonLive : Accounting.Guarantees.JointHorizonLive := by
  intro ι α _ d m C Cm O r G c p δ h s
  calc prob (fun o => ∃ j a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) p G s a o ∧ ¬ (r j).Break G s o) G s
      ≤ ∑ j, prob (fun o => ∃ a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) p G s a o ∧ ¬ (r j).Break G s o)
          G s := Live.prob_exists_univ_le _ G s
    _ ≤ ∑ j, ∑ a, prob (fun o => (r j).accepts a o ∧ ¬ (r j).Holds (c j) p G s a o ∧ ¬ (r j).Break G s o)
          G s := Finset.sum_le_sum fun j _ => Live.prob_exists_univ_le _ G s
    _ ≤ ∑ j, ∑ a, ENNReal.ofReal (δ j a) := Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun a _ => h j a s

/-- **The joint guarantee over a horizon of server-time blocks and their wipes, played as one causal game**
(`Accounting.Guarantees.JointHorizonBlocksLive`). -/
theorem JointHorizonBlocksLive : Accounting.Guarantees.JointHorizonBlocksLive := by
  intro ι α _ _ d B C Cm O S G c p f δ εWipe δPous h hW hM s
  have hA := JointHorizonLive ι α d B C Cm O S.block G c p δ h s
  have hW' : prob (fun o => ∃ b, (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives f o) G s ≤
      ∑ b, ENNReal.ofReal (εWipe b) :=
    (Live.prob_exists_univ_le _ G s).trans (Finset.sum_le_sum fun b _ => hW b s)
  have hM' : prob (fun o => ∃ b, (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm b o) G s ≤
      ∑ b, ENNReal.ofReal (δPous b) :=
    (Live.prob_exists_univ_le _ G s).trans (Finset.sum_le_sum fun b _ => hM b s)
  calc _ ≤ _ := prob_or_le _ _ G s
    _ ≤ _ := add_le_add le_rfl (prob_or_le _ _ G s)
    _ ≤ ∑ b, ∑ a, ENNReal.ofReal (δ b a) + (∑ b, ENNReal.ofReal (εWipe b) + ∑ b, ENNReal.ofReal (δPous b)) :=
        add_le_add hA (add_le_add hW' hM')
    _ = _ := (add_assoc _ _ _).symm

end Accounting.SecurityProofs
