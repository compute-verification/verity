module

public import Proofs.Properties.InferenceOnly.RecordAligned
public import Proofs.Properties.InferenceOnly.RecordCircuit
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordInputs
public import Proofs.Properties.InferenceOnly.RecordRows
public import Proofs.Properties.InferenceOnly.RecordTemplates
public import Proofs.Properties.InferenceOnly.RecordWired

/-! Every record `read` accepts lays out a registered circuit (`form_laidOut`). -/

@[expose] public section

namespace InferenceOnlyCheck.Record.SecurityProofs

open Spot.RecordText
open Lean (Json)

theorem front_leaves {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    f.leaves = f.ts.map (leavesOf f.P (passedOn f.P)) := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rfl

end InferenceOnlyCheck.Record.SecurityProofs

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Lean (Json)
open InferenceOnlyCheck.Record (unwhole leavesOf passedOn)
open InferenceOnlyCheck.Record.SecurityProofs (form read_front finish_text front_text front_leaves inst_of_read
  program_of_read wiring_of_read slot_of_read declared_of_heldAt)

theorem unwhole_of_read {j tr : Json} {c : InferenceOnlyCheck.Record.Case} (h : InferenceOnlyCheck.Record.read j tr = .ok c) :
    Spot.RecordText.templatesOf c.f.P c.f.doc = .ok c.f.ts ∧
      unwhole c.f.ts (c.f.ts.map (InferenceOnlyCheck.Record.leavesOf c.f.P (passedOn c.f.P))) c.wired.wires = none := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  refine ⟨front_text hf, ?_⟩
  rw [← front_leaves hf]
  exact (finish_case hc).2.1

theorem served_of_read {j tr : Json} {c : InferenceOnlyCheck.Record.Case} (h : InferenceOnlyCheck.Record.read j tr = .ok c) :
    ∃ d, Served form c d := by
  have hP := readProg_of_read h
  obtain ⟨-, -, -, -, hown, -⟩ := program_of_read h
  obtain ⟨text, htext, hcuts, hwires⟩ := wiring_of_read h
  have hS := wired_of_read h
  obtain ⟨rs, hrs, hT⟩ := templates_of_read h
  obtain ⟨hts, hwh⟩ := unwhole_of_read h
  have hO := ownRows_holds hP hown hrs
  have hG := gatesOwn hP hrs
  obtain ⟨ρ, hρ⟩ := ranked hS
  obtain ⟨D, at_, hL⟩ := laysOut_exists c.f.P rs c.W c.wired hO (returnsOwn hP hrs hT hS) (unitsComputed hP hrs)
    (operandsOwn hP hown hrs hT hS) hG ρ hρ
  obtain ⟨θ, hθ, U, hN, hD⟩ := names_divides hL hO hG (fun r => (c.heldAt r).isSome) c.slot c.S
    (fun tid u sc hu => slot_of_unit h hrs hu)
  exact ⟨Verity.RegisteredCircuit.ofPartition D θ hθ c.S U, c.f.P, c.f.doc, text, c.wired, rs, at_,
    decode_of_read h, (inst_of_read h).2, htext, hcuts, hwires, slot_of_read h,
    fun r hd hh => declared_of_heldAt h hh, hS, hrs, hT, hO, aligned hP hown hrs hT hS hts hwh,
    filled hP hrs hT hS hts hwh, fun r hh => held_unwired h r hh, hL, hN, hD⟩

end InferenceOnlyCheck.Properties.Record.SecurityProofs

namespace InferenceOnlyCheck.Record.SecurityProofs

theorem form_laidOut : Properties.Record.LaidOut form := by
  intro j tr c h
  have h' : read j tr = .ok c := h
  exact Properties.Record.SecurityProofs.served_of_read h'

end InferenceOnlyCheck.Record.SecurityProofs
