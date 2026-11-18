module

public import Proofs.Properties.InferenceOnly.RecordAligned
public import Proofs.Properties.InferenceOnly.RecordCircuit
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordInputs
public import Proofs.Properties.InferenceOnly.RecordRows
public import Proofs.Properties.InferenceOnly.RecordTemplates
public import Proofs.Properties.InferenceOnly.RecordWired

/-!
# The record check's own form lays out a circuit

`form_laidOut` (`Properties.Record.LaidOut`): every record `read` accepts lays out a registered circuit
(`Properties.Record.Served`), clause by clause, under two hypotheses: every record `read` accepts is of a root with
no edge (`EdgeFreeReads`), so `own-rows` admits only roots whose arguments are runs of input leaves
(`ownRuns_of_edgeFree`), and the wiring has the front's templates (`FinishTemplates`). On a root with edges, every
unit's rows are its own input gates or its writers' outputs (`OwnOrEdgeRows`) where `own-rows` holds
(`OwnRowsEdges`, not yet proved); restating these proofs over it replaces `EdgeFreeReads`.

* The texts. The Program is the instance's, decoded (`decode_of_read`), and the record's text parses
  (`inst_of_read`); the wiring's cuts and wires, the slots and the reasons rows are held are the text's
  (`wiring_of_read`, `slot_of_read`, `declared_of_heldAt`).
* The wiring. It passes `Spot.Wires.check` (`wired_of_read`), the template query accepts its template ids and its
  templates are the Program's (`templates_of_read`); every unit's rows are its own input gates (`ownRows_holds`),
  so its own input gates or its writers' outputs (`ownOrEdgeRows_of_ownRows`);
  the wires pass the check's `unwhole` on the leaves the Program gives each template (`unwhole_of_read`), so every
  wire lands whole and every bit of a wired row is its writer's (`aligned`, `filled`); and no row the check holds has
  a writer (`held_unwired`).
* The circuit. What a unit returns and what its gates read are its own (`returnsOwn`, `operandsOwn`), a unit's gates
  are computed and every gate has its place (`unitsComputed`, `gatesOwn`), and the wires rank the units (`ranked`):
  so the Program's gates no wire covers are a circuit (`laysOut_exists`). θ is the gates of the rows held and the
  blocks are the slots (`names_divides`, on `slot_of_unit`); the registered circuit is that circuit with them
  (`Verity.RegisteredCircuit.ofPartition`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Spot.RecordText
open Lean (Json)

/-- `front` reads each template's leaves off its own Program. -/
theorem front_leaves {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    f.leaves = f.ts.map (leavesOf f.P (passedOn f.P)) := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rfl

end InferenceOnlyVerifier.Record.SecurityProofs

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open InferenceOnlyVerifier.Record (unwhole leavesOf passedOn)
open InferenceOnlyVerifier.Record.SecurityProofs (form read_front finish_text front_text front_leaves inst_of_read
  program_of_read wiring_of_read slot_of_read declared_of_heldAt)

/-- **The wires of a record `read` accepts pass the check's `unwhole`**, on the templates the text gives and the
leaves the Program gives each. -/
theorem unwhole_of_read {j tr : Json} {c : InferenceOnlyVerifier.Record.Case} (h : InferenceOnlyVerifier.Record.read j tr = .ok c) :
    Spot.RecordText.templatesOf c.f.P c.f.doc = .ok c.f.ts ∧
      unwhole c.f.P c.f.ts (c.f.ts.map (InferenceOnlyVerifier.Record.leavesOf c.f.P (passedOn c.f.P))) c.wired = none := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  refine ⟨front_text hf, ?_⟩
  rw [← front_leaves hf]
  exact (finish_case hc).2.1

/-- Rows that are their own input gates are their own input gates or their writers' outputs. -/
theorem ownOrEdgeRows_of_ownRows {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange}
    {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} (h : OwnRows P rs W) : OwnOrEdgeRows P rs W S :=
  ⟨fun r l g lo hi hl => Or.inl (h.1 r l g lo hi hl),
    fun r l lo hi r' l' lo' hi' g h1 h2 _ _ => h.2 r l lo hi r' l' lo' hi' g h1 h2⟩

/-- **On a wiring with no wire, `OwnOrEdgeRows` is `OwnRows`.** -/
theorem ownOrEdgeRows_iff {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {W : Spot.Wires.Wiring}
    {S : Spot.Wires.Summary} (hS : ∀ t u p k, S.writerOf t u p k = none) :
    OwnOrEdgeRows P rs W S ↔ OwnRows P rs W := by
  refine ⟨fun h => ⟨fun r l g lo hi hl => ?_, fun r l lo hi r' l' lo' hi' g h1 h2 => h.2 r l lo hi r' l' lo' hi' g h1 h2
    (hS _ _ _ _) (hS _ _ _ _)⟩, ownOrEdgeRows_of_ownRows⟩
  rcases h.1 r l g lo hi hl with hi' | ⟨l', lo', hi'', wt, wu, o, out, sc', hw, -⟩
  · exact hi'
  · rw [hS] at hw
    cases hw

/-- **A record `read` accepts of a root with no edge lays out a registered circuit**: the wired circuit of its
Program, with θ the gates of the rows the check holds and the blocks the slots. -/
theorem served_of_read (hFT : InferenceOnlyVerifier.Record.SecurityProofs.FinishTemplates) {j tr : Json}
    {c : InferenceOnlyVerifier.Record.Case} (h : InferenceOnlyVerifier.Record.read j tr = .ok c)
    (hNE : InferenceOnlyVerifier.Record.SecurityProofs.EdgeFree c.f.P) :
    ∃ d, Served form c d := by
  have hP := readProg_of_read h
  obtain ⟨-, -, -, -, hown, -⟩ := program_of_read h
  have hown := InferenceOnlyVerifier.Record.SecurityProofs.ownRuns_of_edgeFree hown hNE
  obtain ⟨text, htext, hcuts, hwires⟩ := wiring_of_read h
  have hS := wired_of_read h
  obtain ⟨rs, hrs, hT⟩ := templates_of_read hFT h
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
    fun r hd hh => declared_of_heldAt h hh, hS, hrs, hT, ownOrEdgeRows_of_ownRows hO, aligned hP hown hrs hT hS hts hwh,
    filled hP hrs hT hS hts hwh, fun r hh => held_unwired h r hh, hL, hN, hD⟩

end InferenceOnlyVerifier.Properties.Record.SecurityProofs

namespace InferenceOnlyVerifier.Record.SecurityProofs

/-- **Where `own-rows` holds, every unit's rows are its own input gates or its writers' outputs**, on the record's
wiring: the text's cuts and the edges of the Program's root (`Spot.RecordText.wiring`), the summary SPOT's check
gives it, and the ranges of its templates. Not yet proved; the record proofs restated over it replace
`EdgeFreeReads`. -/
def OwnRowsEdges : Prop :=
  ∀ (P : Flock.Program.Prog) (doc : Lean.Json) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary)
    (rs : Array Flock.TemplateQuery.TRange), ownRows P = true → Spot.RecordText.wiring P doc = .ok W →
    Spot.Wires.check W = .ok S →
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs →
    Properties.Record.OwnOrEdgeRows P rs W S

/-- **The record check's own form lays out a circuit** (`Properties.Record.LaidOut`), where every record it reads
is of a root with no edge. -/
theorem form_laidOut (hNE : EdgeFreeReads) (hFT : FinishTemplates) : Properties.Record.LaidOut form := by
  intro j tr c h
  have h' : read j tr = .ok c := h
  exact Properties.Record.SecurityProofs.served_of_read hFT h' (hNE j tr c h')

end InferenceOnlyVerifier.Record.SecurityProofs
