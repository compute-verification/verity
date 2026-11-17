module

public import Proofs.Spot.Leakage
public import Proofs.Spot.Necessity
public import Proofs.WeightSecurity.Served

/-! `TheftServedTied`: `TheftServedReq` at a checked population whose run records each window's extracted trace. -/

@[expose] public section

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

theorem outOf_eq [Finite U] (h : P.Checked) {V : Type} (f : P.Fns V) {O : Type} (eg : O → S) {a σ : S → V}
    (hin : ∀ s ∈ P.input, σ s = a s) (hc : ∀ u, P.Correct f σ u) : P.outOf f eg a = fun o => σ (eg o) := by
  funext o
  unfold outOf
  split
  · next hr => exact congrFun ((P.determinate_of_checked h V f a).unique (Classical.choose_spec hr) ⟨hin, hc⟩) (eg o)
  · next hr => exact absurd ⟨σ, hin, hc⟩ hr

theorem mem_writesOf_down [Fintype U] {B : Finset U} {s : S} (hs : s ∈ P.writesOf (P.down B)) :
    ∃ u ∈ B, s ∈ P.writesOf (P.down {u}) := by
  classical
  obtain ⟨v, hv, hsv⟩ := mem_biUnion.1 hs
  obtain ⟨u, hu, huv⟩ := (mem_filter.1 hv).2
  exact ⟨u, hu, mem_biUnion.2 ⟨v, mem_filter.2 ⟨mem_univ _, u, mem_singleton_self u, huv⟩, hsv⟩⟩

end Population.Dataflow

namespace WeightSecurity

namespace Served.Traced

open Finset

variable {O I V Tick Adv S U R Q Θ Θ' : Type}

theorem count_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (L : ℕ) :
    ∀ l : List (Traced O (I → V) Tick Adv S V), count L (l.map (counted P f)) = count L (l.map toServed)
  | [] => rfl
  | w :: l => by
    simp only [List.map_cons, count, List.length_map]
    rw [count_counted P f L l]
    rfl

theorem accLen_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) :
    ∀ l : List (Traced O (I → V) Tick Adv S V), accLen (l.map (counted P f)) = accLen (l.map toServed)
  | [] => rfl
  | w :: l => by
    simp only [List.map_cons, accLen]
    rw [accLen_counted P f l]
    rfl

theorem reply_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (L : ℕ)
    (l : List (Traced O (I → V) Tick Adv S V)) (n : ℕ) :
    reply L (l.map (counted P f)) n = reply L (l.map toServed) n := by
  simp only [reply, count_counted, List.getElem?_map, Option.map_map]
  rfl

theorem view_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V)
    (Ou : Outsider R (Q × Adv) (Option ((O → I → V) × Tick)) Θ') (L : ℕ) (r : R)
    (l : List (Traced O (I → V) Tick Adv S V)) (n : ℕ) :
    view Ou L r (l.map (counted P f)) n = view Ou L r (l.map toServed) n := by
  simp only [view, reply_counted]

theorem card_changed_counted [Fintype U] [Fintype O] [Fintype I] (P : Population.Dataflow U S) (h : P.Checked)
    (f : P.Fns V) (eg : O × I → S) (inp : Θ → Q × Adv → S → V) (F : ℕ) (hF : P.FanOut eg F)
    (Ou : Outsider R (Q × Adv) (Option ((O → I → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Traced O (I → V) Tick Adv S V)) (ht : Tied eg L l) (hc : Covered P inp Ou L θ r l) (n : ℕ)
    (hn : n < count L (l.map toServed)) (w : Traced O (I → V) Tick Adv S V) (hw : l[n]? = some w)
    (ha : w.accepted) :
    (changed (popServed P f eg inp) Ou L θ r (l.map (counted P f)) n).card ≤ F * (P.incorrect f w.trace).card := by
  classical
  obtain ⟨σ, ⟨hin, hcor⟩, -⟩ :=
    P.determinate_of_checked h V f (inp θ ((Ou.query r (view Ou L r (l.map toServed) n)).1, w.adv))
  have hw' : (l.map (counted P f))[n]? = some (counted P f w) := by
    rw [List.getElem?_map, hw]
    rfl
  have hsub : changed (popServed P f eg inp) Ou L θ r (l.map (counted P f)) n ⊆
      (P.incorrect f w.trace).biUnion (P.unitsInReach eg) := by
    intro o ho
    simp only [changed, hw', mem_filter, mem_univ, true_and] at ho
    rw [view_counted] at ho
    have hne : (fun i => w.trace (eg (o, i))) ≠ fun i => σ (eg (o, i)) := by
      intro he
      apply ho
      show w.egress o =
        fun i => P.outOf f eg (inp θ ((Ou.query r (view Ou L r (l.map toServed) n)).1, w.adv)) (o, i)
      rw [ht n hn w hw ha, P.outOf_eq h f eg hin hcor]
      exact he
    obtain ⟨i, hi⟩ := Function.ne_iff.1 hne
    have hmem : eg (o, i) ∈ P.writesOf (P.down (P.incorrect f w.trace)) := by
      by_contra hs
      exact hi (P.eq_off_cone (P.covered_of_singleAssignment h.1) h.2 f
        (fun t ht' => (hc n hn w hw ha t ht').trans (hin t ht').symm) subset_rfl
        (fun u hu => absurd (hcor u) (mem_filter.1 hu).2) hs)
    obtain ⟨u, hu, hui⟩ := P.mem_writesOf_down hmem
    exact mem_biUnion.2 ⟨u, hu, mem_filter.2 ⟨mem_univ _, i, mem_inter.2 ⟨mem_image_of_mem eg (mem_univ _), hui⟩⟩⟩
  calc (changed (popServed P f eg inp) Ou L θ r (l.map (counted P f)) n).card
      ≤ ((P.incorrect f w.trace).biUnion (P.unitsInReach eg)).card := card_le_card hsub
    _ ≤ ∑ u ∈ P.incorrect f w.trace, (P.unitsInReach eg u).card := card_biUnion_le
    _ ≤ ∑ _u ∈ P.incorrect f w.trace, F := sum_le_sum fun u _ => hF u
    _ = F * (P.incorrect f w.trace).card := by rw [sum_const, smul_eq_mul, mul_comm]

theorem isolated_counted [Fintype U] [Fintype O] [Fintype I] (P : Population.Dataflow U S) (h : P.Checked) (f : P.Fns V)
    (eg : O × I → S) (inp : Θ → Q × Adv → S → V) (F : ℕ) (hF : P.FanOut eg F)
    (Ou : Outsider R (Q × Adv) (Option ((O → I → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Traced O (I → V) Tick Adv S V)) (ht : Tied eg L l) (hc : Covered P inp Ou L θ r l) :
    Isolated (popServed P f eg inp) F Ou L θ r (l.map (counted P f)) := by
  intro k hk
  rw [mul_sum]
  refine sum_le_sum fun n hn => ?_
  have hnA : n < accLen (l.map toServed) := by
    rw [← accLen_counted P f]
    exact lt_of_lt_of_le (mem_range.1 hn) hk
  have hnC : n < count L (l.map toServed) := lt_of_lt_of_le hnA (accLen_le_count L _)
  have hlen : n < l.length := by
    have := count_le_length L (l.map toServed)
    rw [List.length_map] at this
    omega
  have hw : l[n]? = some l[n] := List.getElem?_eq_getElem hlen
  have ha : l[n].accepted := accepted_of_lt_accLen (l.map toServed) n l[n].toServed hnA (by
    rw [List.getElem?_map, hw]
    rfl)
  have hinc : incorrectAt (l.map (counted P f)) n = (P.incorrect f l[n].trace).card := by
    simp [incorrectAt, hw, counted]
  rw [hinc]
  exact card_changed_counted P h f eg inp F hF Ou L θ r l ht hc n hnC l[n] hw ha

theorem tied_of_forall (eg : O × I → S) (L : ℕ) (l : List (Traced O (I → V) Tick Adv S V))
    (h : ∀ w ∈ l, w.egress = fun o i => w.trace (eg (o, i))) : Tied eg L l :=
  fun _ _ w hw _ => h w (List.mem_of_getElem? hw)

end Served.Traced

namespace SecurityProofs

open Verity.Game WeightSecurity.Assumptions Finset
open scoped ENNReal

theorem TheftServedTied : Guarantees.TheftServedTied := by
  intro Ω R Θ Q Adv O I V Tick Θ' U S _ _ _ _ _ _ _ _ _ _ _ _ _ P f eg inp gen Stolen A p Ou T L F M δ ηe ηa run
    hP hF hhon hsim hund hte hac
  have key := TheftServedReq (popServed P f eg inp) gen Stolen A p Ou T L F M δ (Served.Traced.countRun P f run)
    hhon hsim hund
  have hview : ∀ z : Ω × R, value (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
      (Served.Traced.countRun P f run (gen z.1) z.2) = value (fun l => Stolen (gen z.1)
        (Ou.out z.2 (Served.view Ou L z.2 (l.map Served.Traced.toServed) T))) (run (gen z.1) z.2) := fun z => by
    simp only [Served.Traced.countRun, value_map, Served.Traced.view_counted]
  have hiso : ∀ z : Ω × R, value (fun l => ¬ Served.Isolated (popServed P f eg inp) F Ou L (gen z.1) z.2 l)
      (Served.Traced.countRun P f run (gen z.1) z.2) ≤ ηe + ηa := fun z => by
    simp only [Served.Traced.countRun, value_map]
    refine (value_mono (q := fun l => ¬ Served.Traced.Tied eg L l ∨
      ¬ Served.Traced.Covered P inp Ou L (gen z.1) z.2 l) (fun l hl => ?_) _).trans
      ((value_or_le _ _ _).trans (add_le_add (hte _ _) (hac _ _)))
    by_cases ht : Served.Traced.Tied eg L l
    · by_cases hc : Served.Traced.Covered P inp Ou L (gen z.1) z.2 l
      · exact absurd (Served.Traced.isolated_counted P hP f eg inp F hF Ou L _ _ l ht hc) hl
      · exact Or.inr hc
    · exact Or.inl ht
  calc avg (fun z : Ω × R => value (fun l => Stolen (gen z.1)
          (Ou.out z.2 (Served.view Ou L z.2 (l.map Served.Traced.toServed) T))) (run (gen z.1) z.2))
      = avg (fun z : Ω × R => value (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
          (Served.Traced.countRun P f run (gen z.1) z.2)) := congrArg avg (funext fun z => (hview z).symm)
    _ ≤ _ := key
    _ ≤ _ := by
        rw [add_assoc _ ηe ηa]
        exact add_le_add le_rfl ((avg_mono hiso).trans (avg_le_const _))

namespace ToyTied

open WeightSecurity.ToyTied

theorem checked : pop.Checked := by
  refine ⟨fun s => ⟨fun hs => by simp [pop] at hs, fun _ => ⟨(), by simp [pop], fun _ _ => rfl⟩⟩, fun u h => ?_⟩
  cases h with
  | single hd =>
    obtain ⟨s, -, hs⟩ := hd
    simp [pop] at hs
  | tail _ hd =>
    obtain ⟨s, -, hs⟩ := hd
    simp [pop] at hs

theorem incorrect_honest : pop.incorrect fns (fun _ => 0) = ∅ := by
  ext u
  simp [Population.Dataflow.incorrect, Population.Dataflow.Correct, fns]

theorem rpow_bits :
    (2 : ℝ) ^ servedBits Unit Unit (Unit → Fin 16) Unit 2 (1 * 0 + (0 + 1) * Fintype.card Unit) = 144 := by
  rw [rpow_servedBits]
  norm_num [Fintype.card_unit, Fintype.card_fin, Fintype.card_fun]

end ToyTied

theorem TheftServedTiedHonest : Guarantees.TheftServedTiedHonest := by
  classical
  refine ⟨ToyTied.checked, fun _ => (card_le_univ _).trans_eq Fintype.card_unit,
    fun Pa _ => ToyServed.guess_le_of (popServed WeightSecurity.ToyTied.pop WeightSecurity.ToyTied.fns
      WeightSecurity.ToyTied.eg WeightSecurity.ToyTied.inp) (fun _ => rfl) Pa,
    fun _ => Set.mem_univ _, fun θ r => le_of_eq ?_, fun θ r => le_of_eq ?_, fun θ r => le_of_eq ?_⟩
  · simp only [Served.Traced.countRun, WeightSecurity.ToyTied.honestRun, value_map]
    rw [value_ret, ite_eq_right]
    rintro (⟨w, hw, -, hd⟩ | h)
    · simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at hw
      rcases hw with rfl | rfl <;> exact hd
    · simp [Served.lead, Served.Traced.counted, WeightSecurity.ToyTied.run, ToyTied.incorrect_honest] at h
  · unfold WeightSecurity.ToyTied.honestRun
    rw [value_ret, ite_eq_right]
    exact not_not.2 (Served.Traced.tied_of_forall _ _ _ fun w hw => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
      rcases hw with rfl | rfl <;> rfl)
  · unfold WeightSecurity.ToyTied.honestRun
    rw [value_ret, ite_eq_right]
    exact not_not.2 fun _ _ _ _ _ s hs => absurd hs (by simp [WeightSecurity.ToyTied.pop])

theorem TheftServedTiedNontrivial : Guarantees.TheftServedTiedNontrivial := by
  classical
  intro h
  have hsteal : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => WeightSecurity.ToyServed.Stolen
      (WeightSecurity.ToyServed.gen z.1) (WeightSecurity.ToyTied.thief.out z.2
        (Served.view WeightSecurity.ToyTied.thief 0 z.2 (l.map Served.Traced.toServed) 2)))
      (WeightSecurity.ToyTied.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 1 := by
    intro z
    unfold WeightSecurity.ToyTied.leakRun
    rw [value_ret, ite_eq_left]
    simp [WeightSecurity.ToyServed.Stolen, WeightSecurity.ToyServed.gen, WeightSecurity.ToyTied.thief,
      WeightSecurity.ToyTied.run, Served.view, Served.reply, Served.count, Outsider.view]
  simp only [hsteal, avg_const, ToyTied.rpow_bits, add_zero] at h
  have hlt : ENNReal.ofReal ((144 : ℝ) * (1 / 256)) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    norm_num
  exact absurd h (not_le.2 hlt)

end SecurityProofs

end WeightSecurity
