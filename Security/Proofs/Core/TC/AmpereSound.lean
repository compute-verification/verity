import Proofs.Core.TC.Sound
import Proofs.Core.TC.Ampere.Complete

/-!
# The step relation is the step semantics, on the Ampere/Ada pipeline

The theorems of `Verity.SecurityProofs.TC.Sound` for `AMPERE_BF16_M16N8K16` (two groups of eight products, a 25-bit
adder, exponent floor `-132`), whose parameters `ADA_BF16_M16N8K16` shares (`ada_eq_ampere`). Each step's two groups
chain inside the relation through the first group's output state, as they chain inside `tc_dot_total` through its value.
-/

open Verity.TC.Relation Verity.TC.Spec Verity.Assumptions Verity.TC.Proofs Verity.TC.Proofs.Ampere

namespace Verity.TC

theorem ada_eq_ampere : ADA_BF16_M16N8K16 = AMPERE_BF16_M16N8K16 := rfl

theorem zip_take {α β : Type} : ∀ (n : Nat) (a : List α) (b : List β), (a.zip b).take n = (a.take n).zip (b.take n)
  | 0, _, _ => by simp
  | _ + 1, [], _ => by simp
  | _ + 1, _ :: _, [] => by simp
  | n + 1, x :: a, y :: b => by simp [zip_take n a b]

theorem zip_drop {α β : Type} : ∀ (n : Nat) (a : List α) (b : List β), (a.zip b).drop n = (a.drop n).zip (b.drop n)
  | 0, _, _ => by simp
  | _ + 1, [], _ => by simp
  | _ + 1, _ :: _, [] => by simp
  | n + 1, x :: a, y :: b => by simp [zip_drop n a b]

/-- The product rows of an accepted witness are the ones `step_witness` builds. -/
theorem prods_eq_ampere (a b : List Nat) (ha : a.length = 16) (hb : b.length = 16) (ha16 : ∀ x ∈ a, x < 2 ^ 16)
    (hb16 : ∀ x ∈ b, x < 2 ^ 16) (prod : List ProdCols) (hl : prod.length = 16)
    (h : ∀ j (hj : j < prod.length), prodHolds AP (a[j]!) (b[j]!) prod[j]) :
    prod = (a.zip b).map (fun x => prodCols (-132) x.1 x.2) := by
  apply List.ext_getElem (by simp [hl, ha, hb])
  intro j h1 h2
  have hja : j < a.length := by omega
  have hjb : j < b.length := by omega
  have ea : a[j]! = a[j] := getElem!_pos a j hja
  have eb : b[j]! = b[j] := getElem!_pos b j hjb
  have := prodHolds_eq AP _ _ (ha16 _ (List.getElem_mem hja)) (hb16 _ (List.getElem_mem hjb)) _ (ea ▸ eb ▸ h j h1)
  rw [this]
  simp [List.getElem_zip, AP]

theorem words_mem {a b : List Nat} (ha16 : ∀ x ∈ a, x < 2 ^ 16) (hb16 : ∀ x ∈ b, x < 2 ^ 16) (n : Nat) :
    (∀ x ∈ (a.take n).zip (b.take n), x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16) ∧
      (∀ x ∈ (a.drop n).zip (b.drop n), x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16) := by
  constructor <;> intro x hx
  · exact ⟨ha16 _ (List.mem_of_mem_take (List.of_mem_zip hx).1), hb16 _ (List.mem_of_mem_take (List.of_mem_zip hx).2)⟩
  · exact ⟨ha16 _ (List.mem_of_mem_drop (List.of_mem_zip hx).1), hb16 _ (List.mem_of_mem_drop (List.of_mem_zip hx).2)⟩

theorem ampere_sem (acc : Nat) (a b : List Nat) (ha : a.length = 16) (hb : b.length = 16) :
    tcDotTotal AMPERE_BF16_M16N8K16 acc a b =
      packValue (groupStepTotal
        (groupStepTotal (fp32Value acc) (((a.take 8).zip (b.take 8)).map (fun x => bf16ProductTotal x.1 x.2)) 25 (-132))
        (((a.drop 8).zip (b.drop 8)).map (fun x => bf16ProductTotal x.1 x.2)) 25 (-132)) := by
  simp only [tcDotTotal, AMPERE_BF16_M16N8K16, runGroups]
  rw [← zipWith_eq_map_zip, ← zipWith_eq_map_zip, List.take_of_length_le (l := a.drop 8) (by simp [ha]),
    List.take_of_length_le (l := b.drop 8) (by simp [hb])]

/-- The step's value on the Ampere pipeline. -/
theorem ampere_step_core (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) (w : Witness)
    (h : StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w) :
    ∃ V, Admissible V ∧ tcDotTotal AMPERE_BF16_M16N8K16 acc a b = packValue V ∧
      stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups = stateOf (-132) V := by
  obtain ⟨hacc, hla, hlb, ha16, hb16⟩ := hin
  obtain ⟨⟨hpl, hgl, hterms⟩, hprod, hgroups⟩ := h
  rw [params_ampere] at hprod hgroups
  have hk : AMPERE_BF16_M16N8K16.k = 16 := rfl
  rw [hk] at hpl hla hlb
  have hl : w.groups.length = 2 := by rw [hgl]; rfl
  obtain ⟨g0, g1, hg⟩ : ∃ g0 g1, w.groups = [g0, g1] := by
    cases h : w.groups with
    | nil => rw [h] at hl; simp at hl
    | cons g0 t => cases t with
      | nil => rw [h] at hl; simp at hl
      | cons g1 t => cases t with
        | nil => exact ⟨g0, g1, rfl⟩
        | cons _ _ => rw [h] at hl; simp at hl
  have hg0 : g0.terms.length = 9 := by
    have := hterms 0 (by simp [hg]) (by simp [AMPERE_BF16_M16N8K16])
    simpa [hg, AMPERE_BF16_M16N8K16] using this
  have hg1 : g1.terms.length = 9 := by
    have := hterms 1 (by simp [hg]) (by simp [AMPERE_BF16_M16N8K16])
    simpa [hg, AMPERE_BF16_M16N8K16] using this
  have hpe := prods_eq_ampere a b hla hlb ha16 hb16 w.prod hpl hprod
  simp only [AMPERE_BF16_M16N8K16, hg, groupsHold] at hgroups
  obtain ⟨hgroup0, hgroup1, -⟩ := hgroups
  rw [hpe, ← List.map_take, zip_take] at hgroup0
  rw [hpe, ← List.map_drop, zip_drop, List.take_of_length_le (by simp [hla, hlb])] at hgroup1
  obtain ⟨hdec, hadm⟩ := Ampere.decode_eq acc hacc
  have hdec' : decodeState acc AMPERE_BF16_M16N8K16 = stateOf (-132) (fp32Value acc) := hdec
  have hdec2 : decodeState acc { groups := [8, 8], width := 25, zeroExponent := -132 } =
      stateOf (-132) (fp32Value acc) := hdec
  rw [hdec2] at hgroup0
  obtain ⟨hw0, hw1⟩ := words_mem ha16 hb16 8
  obtain ⟨hout0, hV0⟩ := Ampere.group_sound align (fp32Value acc) hadm ((a.take 8).zip (b.take 8))
    (by simp [hla, hlb]) hw0 g0 hg0 hgroup0
  rw [hout0] at hgroup1
  obtain ⟨hout1, hV1⟩ := Ampere.group_sound align _ hV0 ((a.drop 8).zip (b.drop 8)) (by simp [hla, hlb]) hw1 g1 hg1
    hgroup1
  refine ⟨_, hV1, ampere_sem acc a b hla hlb, ?_⟩
  rw [hdec', hg]
  simp only [stepOut]
  exact hout1

theorem ampere_step_sound (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) (w : Witness)
    (h : StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w) :
    stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups =
      decodeState (tcDotTotal AMPERE_BF16_M16N8K16 acc a b) AMPERE_BF16_M16N8K16 := by
  obtain ⟨V, hV, hsem, hout⟩ := ampere_step_core align acc a b hin w h
  rw [hout, hsem, Ampere.decode_pack _ hV]
  intro t ht hne
  rw [ht] at hV
  exact (hV.2 hne).2.1

theorem ampere_step_complete (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) :
    ∃ w, StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w := by
  obtain ⟨hacc, hla, hlb, ha16, hb16⟩ := hin
  have hk : AMPERE_BF16_M16N8K16.k = 16 := rfl
  rw [hk] at hla hlb
  obtain ⟨hw0, hw1⟩ := words_mem ha16 hb16 8
  obtain ⟨hdec, hadm⟩ := Ampere.decode_eq acc hacc
  have hl0 : ((a.take 8).zip (b.take 8)).length = 8 := by simp [hla, hlb]
  have hl1 : ((a.drop 8).zip (b.drop 8)).length = 8 := by simp [hla, hlb]
  obtain ⟨hc0, hgl0⟩ := Ampere.group_complete align (fp32Value acc) hadm _ hl0 hw0
  obtain ⟨hout0, hV0⟩ := Ampere.group_sound align (fp32Value acc) hadm _ hl0 hw0 _ hgl0 hc0
  obtain ⟨hc1, hgl1⟩ := Ampere.group_complete align _ hV0 _ hl1 hw1
  refine ⟨⟨(a.zip b).map (fun x => prodCols (-132) x.1 x.2),
    [Ampere.honestGroup (stateOf (-132) (fp32Value acc)) (((a.take 8).zip (b.take 8)).map (fun x => prodCols (-132) x.1 x.2)),
     Ampere.honestGroup (stateOf (-132) (groupStepTotal (fp32Value acc)
        (((a.take 8).zip (b.take 8)).map (fun x => bf16ProductTotal x.1 x.2)) 25 (-132)))
        (((a.drop 8).zip (b.drop 8)).map (fun x => prodCols (-132) x.1 x.2))]⟩, ?_, ?_, ?_⟩
  · refine ⟨by simp [hla, hlb, hk], rfl, ?_⟩
    intro i h h'
    have : i = 0 ∨ i = 1 := by simp at h; omega
    rcases this with rfl | rfl
    · simpa [AMPERE_BF16_M16N8K16] using hgl0
    · simpa [AMPERE_BF16_M16N8K16] using hgl1
  · intro j hj
    rw [params_ampere]
    have hj' : j < 16 := by simpa [hla, hlb] using hj
    have hja : j < a.length := by omega
    have hjb : j < b.length := by omega
    rw [getElem!_pos a j hja, getElem!_pos b j hjb]
    simp only [List.getElem_map, List.getElem_zip]
    exact prodHolds_prodCols AP _ _ (ha16 _ (List.getElem_mem hja)) (hb16 _ (List.getElem_mem hjb))
  · rw [params_ampere]
    simp only [AMPERE_BF16_M16N8K16, groupsHold]
    refine ⟨?_, ?_, trivial⟩
    · rw [← List.map_take, zip_take]
      have hdec2 : decodeState acc { groups := [8, 8], width := 25, zeroExponent := -132 } =
          stateOf (-132) (fp32Value acc) := hdec
      rw [hdec2]
      exact hc0
    · rw [← List.map_drop, zip_drop,
        List.take_of_length_le (l := ((a.drop 8).zip (b.drop 8)).map (fun x => prodCols (-132) x.1 x.2))
          (by simp [hla, hlb]), hout0]
      exact hc1

theorem ampere_step_iff (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) (out : State) :
    (∃ w, StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w ∧
        stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups = out) ↔
      out = decodeState (tcDotTotal AMPERE_BF16_M16N8K16 acc a b) AMPERE_BF16_M16N8K16 := by
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact ampere_step_sound align acc a b hin w hw
  · rintro rfl
    obtain ⟨w, hw⟩ := ampere_step_complete align acc a b hin
    exact ⟨w, hw, ampere_step_sound align acc a b hin w hw⟩

theorem ampere_step_hw (step : Nat → List Nat → List Nat → Nat) (hstep : GemmAmpereStep step)
    (align : Align) (acc : Nat) (a b : List Nat) (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) (w : Witness)
    (h : StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w) :
    stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups = decodeState (step acc a b) AMPERE_BF16_M16N8K16 := by
  rw [hstep acc a b hin]
  exact ampere_step_sound align acc a b hin w h

theorem ampere_step_lt (acc : Nat) (a b : List Nat) (hin : StepInputs AMPERE_BF16_M16N8K16 acc a b) :
    tcDotTotal AMPERE_BF16_M16N8K16 acc a b < 2 ^ 32 := by
  obtain ⟨w, hw⟩ := ampere_step_complete .div acc a b hin
  obtain ⟨V, hV, hsem, -⟩ := ampere_step_core .div acc a b hin w hw
  rw [hsem]; exact pack_lt V hV

/-- A verification unit on the Ampere pipeline: every witness chain the relation accepts from `decode_state(acc)` outputs
the decode of the semantics' accumulator after the unit's steps. -/
theorem ampere_unit_sound (align : Align) (acc : Nat) (hacc : acc < 2 ^ 32)
    (steps : List (List Nat × List Nat × Witness))
    (hin : ∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16))
    (out : State) (h : ChainHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) steps out) :
    out = decodeState (tcDotChainTotal AMPERE_BF16_M16N8K16 acc (steps.map (fun s => (s.1, s.2.1))))
      AMPERE_BF16_M16N8K16 := by
  induction steps generalizing acc with
  | nil => exact h
  | cons s rest ih =>
    obtain ⟨a, b, w⟩ := s
    obtain ⟨hs, hrest⟩ := h
    obtain ⟨hla, hlb, ha, hb⟩ := hin (a, b, w) (List.mem_cons_self ..)
    have hin' : StepInputs AMPERE_BF16_M16N8K16 acc a b := ⟨hacc, hla, hlb, ha, hb⟩
    rw [ampere_step_sound align acc a b hin' w hs] at hrest
    exact ih _ (ampere_step_lt acc a b hin') (fun s hs => hin s (List.mem_cons_of_mem _ hs)) hrest

/-- The same from the unit's constant initial state `zero_state`, which is `decode_state(+0)`. -/
theorem ampere_unit_sound_zero (align : Align) (steps : List (List Nat × List Nat × Witness))
    (hin : ∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16))
    (out : State) (h : ChainHolds AMPERE_BF16_M16N8K16 align (zeroState AMPERE_BF16_M16N8K16) steps out) :
    out = decodeState (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1))))
      AMPERE_BF16_M16N8K16 :=
  ampere_unit_sound align 0 (by decide) steps hin out h

end Verity.TC
