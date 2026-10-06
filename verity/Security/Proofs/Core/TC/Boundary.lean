import Proofs.Core.TC.AmpereSound
import Proofs.Core.TC.Proofs.Pack

/-!
# The unit boundary: the committed BF16 word

A verification unit starts from `zero_state`, chains its k16 steps (`ChainHolds`), and packs its final state
(`pack_relation`: the FP32 word, then `cvt.rn.bf16.f32`). `hopper_unit_bf16` and `ampere_unit_bf16`: every such unit the
relation accepts commits `cvt_rn_bf16_f32` of the semantics' accumulator, and its FP32 word is that accumulator.
`pack_sound` and `pack_complete` are the boundary alone.
-/

open Verity.TC.Relation Verity.TC.Spec Verity.TC.Proofs

namespace Verity.TC

/-- `pack_relation` is `pack_value` then `cvt_rn_bf16_f32` on a group's output, and it accepts some columns. -/
theorem pack_sound_complete (Z : Int) (V : Value) (hV : Admissible V) :
    (∀ c, packHolds (stateOf Z V) c → c.word = (packValue V : Int) ∧ c.bf16 = (cvtRnBf16F32 (packValue V) : Int)) ∧
      ∃ c, packHolds (stateOf Z V) c :=
  ⟨fun c h => Proofs.pack_sound Z V hV c h, ⟨_, Proofs.pack_complete Z V hV⟩⟩

theorem hopper_chain_value : ∀ (steps : List (List Nat × List Nat)) (acc : Nat) (V : Value), Admissible V →
    acc = packValue V →
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2, x < 2 ^ 16)) →
    ∃ V', Admissible V' ∧ tcDotChainTotal HOPPER_BF16_WGMMA_K16 acc steps = packValue V'
  | [], acc, V, hV, hacc, _ => ⟨V, hV, hacc⟩
  | (a, b) :: rest, acc, V, hV, hacc, hin => by
    obtain ⟨hla, hlb, ha, hb⟩ := hin (a, b) (List.mem_cons_self ..)
    have hin' : Assumptions.StepInputs HOPPER_BF16_WGMMA_K16 acc a b := ⟨by rw [hacc]; exact pack_lt V hV, hla, hlb, ha, hb⟩
    obtain ⟨w, hw⟩ := hopper_step_complete .div acc a b hin'
    obtain ⟨V', hV', hsem, -⟩ := hopper_step_core .div acc a b hin' w hw
    exact hopper_chain_value rest _ V' hV' hsem (fun s hs => hin s (List.mem_cons_of_mem _ hs))

theorem ampere_chain_value : ∀ (steps : List (List Nat × List Nat)) (acc : Nat) (V : Value), Admissible V →
    acc = packValue V →
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2, x < 2 ^ 16)) →
    ∃ V', Admissible V' ∧ tcDotChainTotal AMPERE_BF16_M16N8K16 acc steps = packValue V'
  | [], acc, V, hV, hacc, _ => ⟨V, hV, hacc⟩
  | (a, b) :: rest, acc, V, hV, hacc, hin => by
    obtain ⟨hla, hlb, ha, hb⟩ := hin (a, b) (List.mem_cons_self ..)
    have hin' : Assumptions.StepInputs AMPERE_BF16_M16N8K16 acc a b := ⟨by rw [hacc]; exact pack_lt V hV, hla, hlb, ha, hb⟩
    obtain ⟨w, hw⟩ := ampere_step_complete .div acc a b hin'
    obtain ⟨V', hV', hsem, -⟩ := ampere_step_core .div acc a b hin' w hw
    exact ampere_chain_value rest _ V' hV' hsem (fun s hs => hin s (List.mem_cons_of_mem _ hs))

theorem zero_admissible : Admissible (.fin ZERO) := by simp [Admissible, ZERO]

/-- A verification unit on the Hopper pipeline, from `zero_state` to the committed BF16 word: the relation's steps and
boundary accept only `cvt_rn_bf16_f32` of the semantics' accumulator, whose FP32 word is the accumulator itself. -/
theorem hopper_unit_bf16 (align : Align) (steps : List (List Nat × List Nat × Witness))
    (hin : ∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16))
    (out : State) (h : ChainHolds HOPPER_BF16_WGMMA_K16 align (zeroState HOPPER_BF16_WGMMA_K16) steps out)
    (c : PackCols) (hc : packHolds out c) :
    c.word = (tcDotChainTotal HOPPER_BF16_WGMMA_K16 0 (steps.map (fun s => (s.1, s.2.1))) : Int) ∧
      c.bf16 = (cvtRnBf16F32 (tcDotChainTotal HOPPER_BF16_WGMMA_K16 0 (steps.map (fun s => (s.1, s.2.1)))) : Int) := by
  have hout := hopper_unit_sound_zero align steps hin out h
  obtain ⟨V, hV, hW⟩ := hopper_chain_value (steps.map (fun s => (s.1, s.2.1))) 0 (.fin ZERO) zero_admissible rfl
    (fun s hs => by
      obtain ⟨s', hs', rfl⟩ := List.mem_map.mp hs
      exact hin s' hs')
  rw [hW] at hout ⊢
  rw [hout, decode_pack V hV (fun t ht hne => by rw [ht] at hV; exact (hV.2 hne).2.1)] at hc
  exact Proofs.pack_sound (-133) V hV c hc

/-- The same on the Ampere/Ada pipeline. -/
theorem ampere_unit_bf16 (align : Align) (steps : List (List Nat × List Nat × Witness))
    (hin : ∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16))
    (out : State) (h : ChainHolds AMPERE_BF16_M16N8K16 align (zeroState AMPERE_BF16_M16N8K16) steps out)
    (c : PackCols) (hc : packHolds out c) :
    c.word = (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1))) : Int) ∧
      c.bf16 = (cvtRnBf16F32 (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1)))) : Int) := by
  have hout := ampere_unit_sound_zero align steps hin out h
  obtain ⟨V, hV, hW⟩ := ampere_chain_value (steps.map (fun s => (s.1, s.2.1))) 0 (.fin ZERO) zero_admissible rfl
    (fun s hs => by
      obtain ⟨s', hs', rfl⟩ := List.mem_map.mp hs
      exact hin s' hs')
  rw [hW] at hout ⊢
  rw [hout, Ampere.decode_pack V hV (fun t ht hne => by rw [ht] at hV; exact (hV.2 hne).2.1)] at hc
  exact Proofs.pack_sound (-132) V hV c hc

end Verity.TC
