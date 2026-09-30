import Verity.TC.Spec
import Verity.TC.Relation
import Verity.Assumptions
import Verity.TC.Proofs.Step
import Verity.TC.Proofs.Words
import Verity.TC.Proofs.Complete

/-!
# The step relation is the step semantics

For the Hopper pipeline (`HOPPER_BF16_WGMMA_K16`, whose parameters `HOPPER_BF16_M16N8K16` shares), in either alignment
mode, over the integers:

* `hopper_step_sound`: every witness the relation accepts for the input state `decode_state(acc)` and the operand words
  `a`, `b` outputs `decode_state(tc_dot_total(acc, a, b))`;
* `hopper_step_complete`: the relation accepts some witness for every input in `tc_dot_total`'s domain;
* `hopper_step_iff`: together, the states the relation can output are exactly the one the semantics gives;
* `hopper_step_hw`: under `gemm-hopper-step`, the output state is the decode of the word the device writes.

A verification unit starts from `zero_state = decode_state(+0)` and feeds each step's output state to the next, so the
step theorem applies to every step of a unit in turn.
-/

open Verity.TC.Relation Verity.TC.Spec Verity.Assumptions Verity.TC.Proofs

namespace Verity.TC

theorem zipWith_eq_map_zip {α β γ : Type} (f : α → β → γ) : ∀ (a : List α) (b : List β),
    List.zipWith f a b = (a.zip b).map (fun x => f x.1 x.2)
  | [], _ => rfl
  | _ :: _, [] => rfl
  | x :: a, y :: b => by simp [zipWith_eq_map_zip f a b]

/-- The product rows of an accepted witness are the ones `step_witness` builds. -/
theorem prods_eq (a b : List Nat) (ha : a.length = 16) (hb : b.length = 16) (ha16 : ∀ x ∈ a, x < 2 ^ 16)
    (hb16 : ∀ x ∈ b, x < 2 ^ 16) (prod : List ProdCols) (hl : prod.length = 16)
    (h : ∀ j (hj : j < prod.length), prodHolds HP (a[j]!) (b[j]!) prod[j]) :
    prod = (a.zip b).map (fun x => prodCols (-133) x.1 x.2) := by
  apply List.ext_getElem (by simp [hl, ha, hb])
  intro j h1 h2
  have hja : j < a.length := by omega
  have hjb : j < b.length := by omega
  have ea : a[j]! = a[j] := getElem!_pos a j hja
  have eb : b[j]! = b[j] := getElem!_pos b j hjb
  have := prodHolds_eq HP _ _ (ha16 _ (List.getElem_mem hja)) (hb16 _ (List.getElem_mem hjb)) _ (ea ▸ eb ▸ h j h1)
  rw [this]
  simp [List.getElem_zip, HP]

theorem hopper_step_sound (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (w : Witness)
    (h : StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w) :
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups =
      decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16 := by
  obtain ⟨hacc, hla, hlb, ha16, hb16⟩ := hin
  obtain ⟨⟨hpl, hgl, hterms⟩, hprod, hgroups⟩ := h
  rw [params_hopper] at hprod hgroups
  have hk : HOPPER_BF16_WGMMA_K16.k = 16 := rfl
  rw [hk] at hpl hla hlb
  obtain ⟨g, hg⟩ : ∃ g, w.groups = [g] := by
    cases h : w.groups with
    | nil => simp [h, HOPPER_BF16_WGMMA_K16] at hgl
    | cons g gs => cases gs with
      | nil => exact ⟨g, rfl⟩
      | cons _ _ => simp [h, HOPPER_BF16_WGMMA_K16] at hgl
  have hg17 : g.terms.length = 17 := by
    have := hterms 0 (by simp [hg]) (by simp [HOPPER_BF16_WGMMA_K16])
    simpa [hg, HOPPER_BF16_WGMMA_K16] using this
  have hpe := prods_eq a b hla hlb ha16 hb16 w.prod hpl hprod
  simp only [HOPPER_BF16_WGMMA_K16, hg, groupsHold] at hgroups
  obtain ⟨hgroup, -⟩ := hgroups
  rw [List.take_of_length_le (by omega), hpe] at hgroup
  obtain ⟨hdec, hadm⟩ := decode_eq acc hacc
  have hdec' : decodeState acc HOPPER_BF16_WGMMA_K16 = stateOf (-133) (fp32Value acc) := hdec
  have hdec2 : decodeState acc { groups := [16], width := 26, zeroExponent := -133 } =
      stateOf (-133) (fp32Value acc) := hdec
  rw [hdec2] at hgroup
  rw [hdec']
  have hab : ∀ x ∈ a.zip b, x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16 := fun x hx =>
    ⟨ha16 _ (List.of_mem_zip hx).1, hb16 _ (List.of_mem_zip hx).2⟩
  obtain ⟨hout, hV⟩ := group_sound align (fp32Value acc) hadm (a.zip b) (by simp [hla, hlb]) hab g hg17 hgroup
  rw [hg]
  simp only [stepOut]
  rw [hout]
  have hsem : tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b =
      packValue (groupStepTotal (fp32Value acc) ((a.zip b).map (fun x => bf16ProductTotal x.1 x.2)) 26 (-133)) := by
    simp only [tcDotTotal, HOPPER_BF16_WGMMA_K16, runGroups]
    rw [List.take_of_length_le (by omega), List.take_of_length_le (by omega), zipWith_eq_map_zip]
  rw [hsem, decode_pack _ hV]
  intro t ht hne
  rw [ht] at hV
  exact (hV.2 hne).2.1

theorem hopper_step_complete (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) :
    ∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w := by
  obtain ⟨hacc, hla, hlb, ha16, hb16⟩ := hin
  have hk : HOPPER_BF16_WGMMA_K16.k = 16 := rfl
  rw [hk] at hla hlb
  have hab : ∀ x ∈ a.zip b, x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16 := fun x hx =>
    ⟨ha16 _ (List.of_mem_zip hx).1, hb16 _ (List.of_mem_zip hx).2⟩
  have hlen : (a.zip b).length = 16 := by simp [hla, hlb]
  obtain ⟨hdec, hadm⟩ := decode_eq acc hacc
  obtain ⟨hg, hgl⟩ := group_complete align (fp32Value acc) hadm (a.zip b) hlen hab
  refine ⟨⟨(a.zip b).map (fun x => prodCols (-133) x.1 x.2),
    [honestGroup (stateOf (-133) (fp32Value acc)) ((a.zip b).map (fun x => prodCols (-133) x.1 x.2))]⟩, ?_, ?_, ?_⟩
  · refine ⟨by simp [hlen, hk], rfl, ?_⟩
    intro i h h'
    have : i = 0 := by simp at h; omega
    subst this
    simpa [HOPPER_BF16_WGMMA_K16] using hgl
  · intro j hj
    rw [params_hopper]
    have hj' : j < 16 := by simpa [hlen] using hj
    have hja : j < a.length := by omega
    have hjb : j < b.length := by omega
    rw [getElem!_pos a j hja, getElem!_pos b j hjb]
    simp only [List.getElem_map, List.getElem_zip]
    exact prodHolds_prodCols HP _ _ (ha16 _ (List.getElem_mem hja)) (hb16 _ (List.getElem_mem hjb))
  · rw [params_hopper]
    simp only [HOPPER_BF16_WGMMA_K16, groupsHold]
    refine ⟨?_, trivial⟩
    rw [List.take_of_length_le (by simp [hlen])]
    have hdec2 : decodeState acc { groups := [16], width := 26, zeroExponent := -133 } =
        stateOf (-133) (fp32Value acc) := hdec
    rw [hdec2]
    exact hg

theorem hopper_step_iff (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (out : State) :
    (∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w ∧
        stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = out) ↔
      out = decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16 := by
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact hopper_step_sound align acc a b hin w hw
  · rintro rfl
    obtain ⟨w, hw⟩ := hopper_step_complete align acc a b hin
    exact ⟨w, hw, hopper_step_sound align acc a b hin w hw⟩

theorem hopper_step_hw (step : Nat → List Nat → List Nat → Nat) (hstep : GemmHopperStep step)
    (align : Align) (acc : Nat) (a b : List Nat) (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (w : Witness)
    (h : StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w) :
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = decodeState (step acc a b) HOPPER_BF16_WGMMA_K16 := by
  rw [hstep acc a b hin]
  exact hopper_step_sound align acc a b hin w h

end Verity.TC
