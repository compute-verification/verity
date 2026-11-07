module

public import Proofs.Flock.Claims.Sampling.InputSlot
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Claims.Sampling.Select.Misses
public import Proofs.Flock.Claims.Sampling.Session
public import Properties.SampledProofs.Draw
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import VCVio.Prelude.Core

/-!
# The draw law from the deployed draw (`Claims.Sampling.DrawLaw`)

* `Design.Pa_empty`: a well-formed design has `P_a(∅) = 1`.
* `allSlots`, `drawChecked`: every slot of a layout, and the deployed draw's checked slots, `Slots.select`'s or, where
  it is refused (the run rejects), every slot.
* The statement's partition is a population, then its program's input class (`InputSlot.withInputs`), and its design
  is `slotDesign`'s two rules, the population's first stage and the input class drawn `all`: on population slots `P_a`
  is the first stage's (`pa_of_pop`), and a set with an input slot has `P_a = 0` (`pa_input`). The input class is the
  program's, read on its own: an input is checked only through the drawn units that read it (`inputOk`).
* `checkedSlots`: the draw's population slots, then every input slot of the program, numbered after the population.
* `DrawSlots`: the deployed draw is `Slots.select`'s on a layout whose population has the strata sizes of the
  statement's and on its design, from uniform bytes of the coins, checking `checkedSlots`.
* `drawLaw_of_slots`: `DrawLaw` there, from `drawMisses`.
-/

@[expose] public section

open SampledProofs
open Lean (Json)

namespace Claims.Sampling

open Claims Finset Properties.SampledProofs Properties.SampledProofs.SecurityProofs

/-! ## `P_a(∅)` on a well-formed design -/

theorem window_empty (o n : ℕ) : window o n ∅ = ∅ := by simp [window]

theorem strataPa_empty : ∀ (l : List (ℕ × ℕ)) (o : ℕ), (∀ p ∈ l, p.2 ≤ p.1) → strataPa l o ∅ = 1
  | [], _, _ => rfl
  | (n, k) :: l, o, h => by
    have hk : k ≤ n := h _ List.mem_cons_self
    rw [strataPa, window_empty, Finset.card_empty, Nat.sub_zero,
      div_self (Nat.cast_ne_zero.2 (Nat.choose_pos hk).ne'), one_mul]
    exact strataPa_empty l _ fun p hp => h p (List.mem_cons_of_mem _ hp)

theorem OneStage.Pa_empty (ns : List ℕ) (st : OneStage) (h : st.WellFormed ns) : st.Pa ns ∅ = 1 := by
  cases st with
  | all => simp [OneStage.Pa]
  | subset K =>
    simp only [OneStage.Pa, Finset.card_empty, Nat.sub_zero]
    exact div_self (Nat.cast_ne_zero.2 (Nat.choose_pos h.2).ne')
  | bernoulli num den => simp [OneStage.Pa]
  | stratified K =>
    obtain ⟨hK1, hKN, hn⟩ := h
    refine strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨n, hnm, rfl⟩ := List.mem_map.1 hp
    have h1 := hn n hnm
    refine max_le h1 (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
    have : K * n ≤ ns.sum * n := Nat.mul_le_mul_right n hKN
    nlinarith
  | work K w f =>
    refine strataPa_empty _ 0 fun p hp => ?_
    obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
    exact min_le_left _ _

theorem classesPa_empty : ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (o : ℕ), rs.length = cs.length →
    (∀ p ∈ rs.zip cs, p.1.WellFormed (strataSizes p.2)) → classesPa rs cs o ∅ = 1
  | [], [], _, _, _ => rfl
  | r :: rs, c :: cs, o, hl, h => by
    rw [classesPa, window_empty, OneStage.Pa_empty _ r (h _ List.mem_cons_self), one_mul]
    exact classesPa_empty rs cs _ (by simpa using hl) fun p hp => h p (List.mem_cons_of_mem _ hp)
  | [], _ :: _, _, hl, _ => absurd hl (by simp)
  | _ :: _, [], _, hl, _ => absurd hl (by simp)

/-- **A well-formed design accepts the empty set surely**: `P_a(∅) = 1`. -/
theorem Design.Pa_empty (d : Design) (P : Partition) (h : d.WellFormed P) : d.Pa P ∅ = 1 :=
  classesPa_empty _ _ 0 h.1 h.2

/-! ## The deployed draw's checked slots -/

/-- Every slot of a layout, in order. -/
def allSlots (l : Slots.Layout) : Array ℕ := Array.range l.blocks.size

theorem mem_allSlots (l : Slots.Layout) (v : ℕ) : v ∈ allSlots l ↔ v < l.blocks.size := by
  simp [allSlots]

theorem slotPartition_blocks (l : Slots.Layout) : (slotPartition l).blocks = l.blocks.toList.map puGates := by
  simp [slotPartition, Partition.blocks, Slots.Layout.blocks, List.map_flatten, Function.comp_def]

/-- **The slots the deployed draw checks** on the bytes `src`: `Slots.select`'s, and every slot of the layout `l` where
it is refused, where the run rejects. -/
def drawChecked (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (l : Slots.Layout) (src : ByteArray) : Array ℕ :=
  match Slots.select rule d query table cm src with
  | .ok (_, c) => c
  | .error _ => allSlots l

/-- **The slots checked over the program `C`**: the population slots of the layout `l` among the draw's `c`, in slot
order, then every input slot of `C` (`InputSlot.inputsAt`), numbered after the population. -/
def checkedSlots (l : Slots.Layout) (c : Array ℕ) (C : Prog) : List ℕ :=
  (List.range (Slots.popOf l).length).filter (fun v => decide (v ∈ c)) ++
    (List.range (InputSlot.inputsAt C).length).map ((Slots.popOf l).length + ·)

theorem length_checkedSlots_le (l : Slots.Layout) (c : Array ℕ) (C : Prog) :
    (checkedSlots l c C).length ≤ (Slots.popOf l).length + (InputSlot.inputsAt C).length := by
  rw [checkedSlots, List.length_append, List.length_map, List.length_range]
  exact Nat.add_le_add_right (List.filter_sublist.length_le.trans (by simp)) _

/-- A population slot of the draw's is checked, at an index below the population's size. -/
theorem checkedSlots_pop {l : Slots.Layout} {c : Array ℕ} {C : Prog} {v : ℕ} (hv : v < (Slots.popOf l).length)
    (hvc : v ∈ c) : ∃ p < (Slots.popOf l).length, (checkedSlots l c C)[p]? = some v := by
  have hf : v ∈ (List.range (Slots.popOf l).length).filter (fun w => decide (w ∈ c)) :=
    List.mem_filter.2 ⟨List.mem_range.2 hv, decide_eq_true hvc⟩
  obtain ⟨p, hp⟩ := List.mem_iff_getElem?.1 hf
  have hpl := (List.getElem?_eq_some_iff.1 hp).1
  have hfl : ((List.range (Slots.popOf l).length).filter (fun w => decide (w ∈ c))).length ≤
      (Slots.popOf l).length :=
    List.filter_sublist.length_le.trans (by simp)
  exact ⟨p, by omega, by rw [checkedSlots, List.getElem?_append_left hpl]; exact hp⟩

/-- Every input slot of the program is checked, at an index below the slots'. -/
theorem checkedSlots_input {l : Slots.Layout} {c : Array ℕ} {C : Prog} {i : ℕ}
    (hi : i < (InputSlot.inputsAt C).length) :
    ∃ p < (Slots.popOf l).length + (InputSlot.inputsAt C).length,
      (checkedSlots l c C)[p]? = some ((Slots.popOf l).length + i) := by
  obtain ⟨f, hf⟩ : ∃ f, f = (List.range (Slots.popOf l).length).filter (fun w => decide (w ∈ c)) := ⟨_, rfl⟩
  have hfl : f.length ≤ (Slots.popOf l).length := by
    rw [hf]
    exact List.filter_sublist.length_le.trans (by simp)
  refine ⟨f.length + i, by omega, ?_⟩
  rw [checkedSlots, ← hf, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, List.getElem?_map,
    List.getElem?_range hi]
  rfl

/-! ## The statement's partition: a population, then the program's input class -/

theorem flatten_length_eq_sizes (c : List (List (List ℕ))) : c.flatten.length = (strataSizes c).sum := by
  simp [strataSizes, List.length_flatten]

/-- The layout's population slots are its first class's proof blocks. -/
theorem popOf_length (l : Slots.Layout) :
    (Slots.popOf l).length = ((slotPartition l).classes.headD []).flatten.length := by
  rcases l with ⟨⟨cl⟩⟩
  cases cl with
  | nil => rfl
  | cons c cs => simp [slotPartition, Slots.popOf, List.length_flatten, Function.comp_def]

/-- A population of the layout's population's strata sizes has as many proof blocks. -/
theorem popLength_of_sizes {l : Slots.Layout} {pop : List (List (List ℕ))}
    (h : strataSizes pop = strataSizes ((slotPartition l).classes.headD [])) :
    pop.flatten.length = (Slots.popOf l).length := by
  rw [popOf_length, flatten_length_eq_sizes, flatten_length_eq_sizes, h]

theorem popOf_length_le (l : Slots.Layout) : (Slots.popOf l).length ≤ l.blocks.size := by
  have h : ((slotPartition l).classes.headD []).flatten.length ≤ (slotPartition l).blocks.length := by
    unfold Partition.blocks
    generalize (slotPartition l).classes = cl
    cases cl with
    | nil => simp
    | cons c cs => simp [List.flatten_append]
  rw [popOf_length]
  exact h.trans (le_of_eq (by rw [slotPartition_blocks, List.length_map, Array.length_toList]))

theorem blocks_length_withInputs (pop : List (List (List ℕ))) (C : Prog) :
    (InputSlot.withInputs ⟨[pop]⟩ C).blocks.length = pop.flatten.length + (InputSlot.inputsAt C).length := by
  simp only [InputSlot.blocks_withInputs, List.length_append, List.length_map]
  simp [Partition.blocks]

theorem window_eq_empty {o n : ℕ} {B : Finset ℕ} (h : ∀ v ∈ B, v < o) : window o n B = ∅ := by
  simp only [window, Finset.image_eq_empty, Finset.filter_eq_empty_iff]
  rintro v hv ⟨hvo, -⟩
  have := h v hv
  omega

theorem window_ne_empty {o n v : ℕ} {B : Finset ℕ} (hv : v ∈ B) (h1 : o ≤ v) (h2 : v < o + n) :
    window o n B ≠ ∅ := by
  unfold window
  exact Finset.nonempty_iff_ne_empty.1 ⟨v - o, Finset.mem_image.2 ⟨v, Finset.mem_filter.2 ⟨hv, h1, h2⟩, rfl⟩⟩

theorem all_Pa_of_empty {ns : List ℕ} {W : Finset ℕ} (h : W = ∅) : OneStage.Pa ns .all W = 1 := by
  simp [OneStage.Pa, h]

theorem all_Pa_of_ne {ns : List ℕ} {W : Finset ℕ} (h : W ≠ ∅) : OneStage.Pa ns .all W = 0 := by
  simp [OneStage.Pa, h]

theorem classesPa_nil_rules (cs : List (List (List (List ℕ)))) (o : ℕ) (B : Finset ℕ) : classesPa [] cs o B = 1 := by
  cases cs <;> rfl

theorem classesPa_nil_classes (rs : List OneStage) (o : ℕ) (B : Finset ℕ) : classesPa rs [] o B = 1 := by
  cases rs <;> rfl

/-- **`P_a` on population slots**: `slotDesign`'s two rules, the population's first stage and the input class drawn
`all`, accept a set of the first class's slots with the first stage's probability on it. -/
theorem pa_of_pop (first : OneStage) {P : Partition} {c : List (List (List ℕ))} {cs : List (List (List (List ℕ)))}
    (hc : P.classes = c :: cs) {B : Finset ℕ} (hB : ∀ v ∈ B, v < c.flatten.length) :
    (⟨[first, .all]⟩ : Design).Pa P B = first.Pa (strataSizes c) (window 0 c.flatten.length B) := by
  unfold Design.Pa
  rw [hc]
  show first.Pa (strataSizes c) (window 0 c.flatten.length B) * classesPa [.all] cs (0 + c.flatten.length) B = _
  cases cs with
  | nil => rw [classesPa_nil_classes, mul_one]
  | cons c' cs =>
    show _ * (OneStage.Pa (strataSizes c') .all (window (0 + c.flatten.length) c'.flatten.length B) *
      classesPa [] cs (0 + c.flatten.length + c'.flatten.length) B) = _
    rw [classesPa_nil_rules, zero_add, window_eq_empty hB, all_Pa_of_empty rfl, mul_one, mul_one]

/-- **`P_a` of a set with an input slot is `0`**: the input class is drawn `all`. -/
theorem pa_input (first : OneStage) (pop : List (List (List ℕ))) (C : Prog) {B : Finset ℕ} {v : ℕ} (hv : v ∈ B)
    (h1 : pop.flatten.length ≤ v) (h2 : v < pop.flatten.length + (InputSlot.inputsAt C).length) :
    (⟨[first, .all]⟩ : Design).Pa (InputSlot.withInputs ⟨[pop]⟩ C) B = 0 := by
  show first.Pa (strataSizes pop) (window 0 pop.flatten.length B) *
    (OneStage.Pa (strataSizes (InputSlot.inputClass C)) .all
      (window (0 + pop.flatten.length) (InputSlot.inputClass C).flatten.length B) *
      classesPa [] [] (0 + pop.flatten.length + (InputSlot.inputClass C).flatten.length) B) = 0
  rw [InputSlot.inputClass_flatten, List.length_map, zero_add, all_Pa_of_ne (window_ne_empty hv h1 h2), zero_mul,
    mul_zero]

/-- **`slotDesign`'s design has two rules**, the population's first stage and the input class drawn `all`. -/
theorem slotDesign_two {rule : Select.Rule} {d : Program} {query : Json} {table : Option Json} {D : Design}
    (h : slotDesign rule d query table = .ok D) : ∃ first, D = ⟨[first, .all]⟩ := by
  obtain ⟨_, _, _, -, -, -, h⟩ := slotDesign_ok rule d query table D h
  rcases h with ⟨-, rfl⟩ | ⟨-, _, -, rfl⟩
  · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

/-- **The design is well formed on the statement's partition** where it is on the layout's: its first rule reads the
population's strata sizes, the layout's population's, and the input class is drawn `all`. -/
theorem wellFormed_withInputs {first : OneStage} {l : Slots.Layout} {pop : List (List (List ℕ))} (C : Prog)
    (hpop : strataSizes pop = strataSizes ((slotPartition l).classes.headD []))
    (h : (⟨[first, .all]⟩ : Design).WellFormed (slotPartition l)) :
    (⟨[first, .all]⟩ : Design).WellFormed (InputSlot.withInputs ⟨[pop]⟩ C) := by
  obtain ⟨hlen, hwf⟩ := h
  have h1 : first.WellFormed (strataSizes pop) := by
    rw [hpop]
    generalize (slotPartition l).classes = cl at hlen hwf ⊢
    cases cl with
    | nil => simp at hlen
    | cons c cs => exact hwf (first, c) (by simp)
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [InputSlot.withInputs, List.cons_append, List.nil_append, List.zip_cons_cons, List.zip_nil_right,
    List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact h1
  · trivial

/-- The proof units of a population, then the program's input class. -/
theorem nPU_withInputs {Coin Col T : Type} {D : Deploy Coin Col T} {x : Msg} {s : AuditStatement}
    {pop : List (List (List ℕ))} (hx : D.parse x = some s)
    (hP : D.partition s = some (InputSlot.withInputs ⟨[pop]⟩ s.prog)) :
    nPU D x = pop.flatten.length + (InputSlot.inputsAt s.prog).length := by
  rw [← blocks_length_withInputs]
  simp [nPU, hx, Deploy.part, hP]

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]

/-- **The deployed draw is `Slots.select`'s**: on a statement that parses and whose query evaluates, the verifier's own
program `d` (its parse of the statement's descriptor), query, selection rule, work table and closure map, whose layout
`l`'s population has the strata sizes of the statement's population `pop` and whose design is the statement's; the
statement's partition is `pop`, then its program's input class (`InputSlot.withInputs`); `L` bytes of the coins,
uniform when the coins are; and the checked slots the draw's population slots on them, then every input slot of the
program (`checkedSlots`). -/
def DrawSlots (D : Deploy Coin Col T) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt →
  ∃ (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (l : Slots.Layout) (pop : List (List (List ℕ))) (L : ℕ)
    (bytes : (Fin (D.V.k x) → Coin) → Fin L → Fin 256),
    Program.parse ⟨s.desc.toArray⟩ = .ok d ∧ Slots.layout d query = .ok l ∧
      Pt = InputSlot.withInputs ⟨[pop]⟩ s.prog ∧ strataSizes pop = strataSizes ((slotPartition l).classes.headD []) ∧
      slotDesign rule d query table = .ok s.design ∧
      (∀ E : (Fin L → Fin 256) → Prop, Pr (fun c => E (bytes c)) = Pr E) ∧
      ∀ c, D.checked x c = checkedSlots l (drawChecked rule d query table cm l (bytesOf (bytes c))) s.prog

/-- **The draw law from the deployed draw**: where the draw is `Slots.select`'s on uniform bytes (`DrawSlots`), a set
`B` with an input slot is never missed, since every input slot is checked, and has `P_a(B) = 0` (`pa_input`); on
population slots the miss probability at fixed first-stage coins averages at most the layout's `P_a(B)`
(`drawMisses`), which is the statement's (`pa_of_pop`), and a refused draw checks every slot, so it misses no nonempty
`B`; a well-formed design has `P_a(∅) = 1`. -/
theorem drawLaw_of_slots (D : Deploy Coin Col T) (h : DrawSlots D) : DrawLaw D := by
  intro x s Pt hx hP hin B hB
  obtain ⟨rule, d, query, table, cm, l, pop, L, bytes, -, hl, rfl, hpop, hds, hunif, hch⟩ := h x s Pt hx hP
  have hpn := popLength_of_sizes hpop
  have hnPU : nPU D x = (Slots.popOf l).length + (InputSlot.inputsAt s.prog).length := by
    rw [nPU_withInputs hx hP, hpn]
  rw [blocks_length_withInputs, hpn] at hB
  obtain ⟨first, hfirst⟩ := slotDesign_two hds
  by_cases hB0 : B = ∅
  · subst hB0
    refine ⟨fun _ => 1, fun _ => zero_le_one, fun r => Pr_le_one _, ?_⟩
    rw [Ex_const, Design.Pa_empty _ _ hin.2.2]
  by_cases hinp : ∃ v ∈ B, (Slots.popOf l).length ≤ v
  · obtain ⟨v, hvB, hvn⟩ := hinp
    have hvl : v < (Slots.popOf l).length + (InputSlot.inputsAt s.prog).length := Finset.mem_range.1 (hB hvB)
    refine ⟨fun _ => 0, fun _ => le_rfl,
      fun r => (Pr_congr fun dd => iff_false_intro fun hm => ?_).trans_le Pr_false.le, ?_⟩
    · obtain ⟨p, hp, hpv⟩ := checkedSlots_input (l := l)
        (c := drawChecked rule d query table cm l (bytesOf (bytes (forkOn (D.stage2 x) r dd)))) (C := s.prog)
        (i := v - (Slots.popOf l).length) (by omega)
      rw [Nat.add_sub_of_le hvn] at hpv
      exact hm v hvB ⟨p, by rw [hnPU]; exact hp⟩ (by rw [hch]; exact hpv)
    · rw [Ex_const, hfirst, pa_input first pop s.prog hvB (by omega) (by omega)]
  push_neg at hinp
  obtain ⟨v0, hv0⟩ := Finset.nonempty_iff_ne_empty.2 hB0
  let E : (Fin L → Fin 256) → Prop := fun src =>
    (Slots.select rule d query table cm (bytesOf src)).isOk = true ∧
      Misses B (checkedOf (Slots.select rule d query table cm (bytesOf src)))
  refine ⟨fun r => Pr fun dd => E (bytes (forkOn (D.stage2 x) r dd)), fun r => Pr_nonneg _,
    fun r => Pr_mono fun dd hm => ?_, ?_⟩
  · have hmiss : ∀ v ∈ B, v ∉ drawChecked rule d query table cm l (bytesOf (bytes (forkOn (D.stage2 x) r dd))) := by
      intro v hv hvc
      obtain ⟨p, hp, hpv⟩ := checkedSlots_pop (C := s.prog) (hinp v hv) hvc
      exact hm v hv ⟨p, by rw [hnPU]; omega⟩ (by rw [hch]; exact hpv)
    unfold drawChecked at hmiss
    split at hmiss
    · rename_i l' c hs
      obtain ⟨hl', draw, proved, read, -, rfl⟩ := slots_select_ok _ _ _ _ _ _ _ _ hs
      rw [hl] at hl'
      obtain rfl := Except.ok.inj hl'
      exact ⟨by rw [hs]; rfl, by simp only [hs, checkedOf]; exact hmiss⟩
    · exact absurd ((mem_allSlots _ _).2 ((hinp v0 hv0).trans_le (popOf_length_le l))) (hmiss v0 hv0)
  · have hav : Ex (fun r => Pr fun dd => E (bytes (forkOn (D.stage2 x) r dd))) = Pr fun c => E (bytes c) := by
      classical
      rw [← Ex_ind fun c => E (bytes c)]
      rw [← Ex_resample (fun _ => D.stage2 x) (fun _ _ => rfl) fun c => if E (bytes c) then 1 else 0]
      exact congrArg Ex (funext fun r => (Ex_ind _).symm)
    rw [hav, hunif]
    have hb := drawMisses rule d query table cm L B
    simp only [missBound, hl, hds] at hb
    refine hb.trans_eq ?_
    obtain ⟨c, cs, hc⟩ : ∃ c cs, (slotPartition l).classes = c :: cs := by
      cases hcl : (slotPartition l).classes with
      | nil =>
        have := hinp v0 hv0
        rw [popOf_length, hcl] at this
        simp at this
      | cons c cs => exact ⟨c, cs, rfl⟩
    have hcp : c.flatten.length = (Slots.popOf l).length := by rw [popOf_length, hc, List.headD_cons]
    have hsz : strataSizes pop = strataSizes c := by rw [hpop, hc, List.headD_cons]
    rw [hfirst, pa_of_pop first hc (fun v hv => by rw [hcp]; exact hinp v hv),
      pa_of_pop first (P := InputSlot.withInputs ⟨[pop]⟩ s.prog) (c := pop) (cs := [InputSlot.inputClass s.prog]) rfl
        (fun v hv => by rw [hpn]; exact hinp v hv), hsz, hcp, hpn]

end Claims.Sampling
