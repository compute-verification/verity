module

public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Record
public import Proofs.Flock.Soundness.Refine.WalkZeta
import all Lean.Data.Json.Basic

/-!
# The registration v2's parameters name, at the record

* **The parameters' program is the one their verifier parses** (`program_parse_of_paramsAt`): one-stage v2's parameters
  (`SampledProofs.OneStageZk.paramsOfJson`) take the verifier's own program bytes, `own.program`, as the canonical form
  of the parameters' `"program"` value, and `SampledProofs.OneStage.paramsParsed` parses the verifier's program from the
  canonical form of the same value, so `Program.parse own.program` is the verifier's program.
* **v2's slots are two classes** (`partSlots_classes`): the population, in the registration's strata, and the input
  class, at most one stratum. The population's strata have the sizes of the layout's first class.
* **The registration is v2's parameters'** (`regIs_of_setsUp`): at a session v2 sets up, the statement's descriptor and
  query are the parameters' program bytes and verifier query (`RegIs`, from `SetsUp.params`), so the descriptor parses
  as the parameters' verifier's program.
-/

@[expose] public section

namespace Verity.Sampled.Record

open Lean (Json)
open Claims.Sampling
open FlockSoundness.Refine (Ret throw_ne_ok)
open Verity.Sampled.Trees (bind_ok)

/-- `OneStage.paramsParsed` parses its verifier's program from the canonical form of the parameters' `"program"`. -/
theorem paramsParsed_program {j : Json} {b : _root_.SampledProofs.OneStage.Params}
    (h : _root_.SampledProofs.OneStage.paramsParsed j = .ok b) :
    ∃ s, Flock.canon (_root_.SampledProofs.getD j "program") = .ok s ∧
      _root_.SampledProofs.Program.parse s.toUTF8 = .ok b.verifier.program := by
  unfold _root_.SampledProofs.OneStage.paramsParsed at h
  extract_lets at h
  replace h := bind_ok h
  obtain ⟨_, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨pj, hpj, h⟩ := h
  replace h := bind_ok h
  obtain ⟨s, hs, h⟩ := h
  replace h := bind_ok h
  obtain ⟨prog, hprog, h⟩ := h
  have hpj' : pj = _root_.SampledProofs.getD j "program" := by
    generalize _root_.SampledProofs.getD j "program" = x at hpj ⊢
    cases x <;> cases hpj <;> rfl
  subst hpj'
  refine ⟨s, hs, ?_⟩
  suffices hb : b.verifier.program = prog by rw [hb]; exact hprog
  clear hprog hs
  revert h
  revert b
  change Ret _ (fun b : _root_.SampledProofs.OneStage.Params => b.verifier.program = prog)
  ret_walk_zeta

theorem oneStage_program {j : Json} {b : _root_.SampledProofs.OneStage.Params}
    (h : _root_.SampledProofs.OneStage.paramsOfJson j = .ok b) :
    ∃ s, Flock.canon (_root_.SampledProofs.getD j "program") = .ok s ∧
      _root_.SampledProofs.Program.parse s.toUTF8 = .ok b.verifier.program := by
  unfold _root_.SampledProofs.OneStage.paramsOfJson at h
  obtain ⟨b', hb', h⟩ := bind_ok h
  have hb : b = b' := by
    split at h
    · exact absurd h (throw_ne_ok _ _)
    · cases h
      rfl
  subst hb
  exact paramsParsed_program hb'

/-- One-stage v2's own program bytes parse as its verifier's program. -/
theorem zk_program {j : Json} {p : _root_.SampledProofs.OneStageZk.Params}
    (h : _root_.SampledProofs.OneStageZk.paramsOfJson j = .ok p) :
    _root_.SampledProofs.Program.parse p.own.program = .ok p.base.verifier.program := by
  unfold _root_.SampledProofs.OneStageZk.paramsOfJson at h
  repeat' first
    | exact absurd h (throw_ne_ok _ _)
    | (replace h := bind_ok h; obtain ⟨_, _, h⟩ := h)
    | (split at h)
  all_goals
    cases h
    have hc := ‹Flock.canon _ = Except.ok _›
    obtain ⟨s, hs, hp⟩ := oneStage_program ‹_root_.SampledProofs.OneStage.paramsOfJson _ = Except.ok _›
    rw [hc] at hs
    cases hs
    exact hp

/-- **The parameters' program is the one their verifier parses**: at parameters `ParamsAt θ p`, the verifier's own
program bytes parse as its verifier's program. -/
theorem program_parse_of_paramsAt {θ : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params} (h : ParamsAt θ p) :
    _root_.SampledProofs.Program.parse p.own.program = .ok p.base.verifier.program := by
  obtain ⟨_, -, x, -, -, hx⟩ := h
  obtain ⟨_, -, hx⟩ := bind_ok hx
  obtain ⟨_, hr⟩ := Binding.dispatch_params hx
  exact zk_program hr

theorem strataSizes_slotPartition (l : _root_.SampledProofs.Slots.Layout) :
    strataSizes ((_root_.Properties.SampledProofs.slotPartition l).classes.headD []) =
      (l.classes.toList.headD #[]).toList.map Array.size := by
  unfold _root_.Properties.SampledProofs.slotPartition strataSizes
  generalize l.classes.toList = cs
  cases cs <;> simp [Function.comp_def]

/-- **v2's slots are two classes**: the population in the registration's strata, whose sizes are those of the first
class of the registration's layout `l`, and the input class, at most one stratum. -/
theorem partSlots_classes (A : Live) {p : _root_.SampledProofs.Program} {l : _root_.SampledProofs.Slots.Layout}
    (hp : _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok p)
    (hl : _root_.SampledProofs.Slots.layout p A.qr = .ok l) :
    ∃ pop ins, (partSlots A).classes = [pop, ins] ∧ ins.length ≤ 1 ∧
      strataSizes pop = strataSizes ((_root_.Properties.SampledProofs.slotPartition l).classes.headD []) := by
  refine ⟨_, InputSlot.inputClass (Flock.Bridge.progOf A.Cc A.κ), rfl, ?_, ?_⟩
  · unfold InputSlot.inputClass
    split <;> simp
  · rw [strataSizes_slotPartition, ← Flock.Bridge.nsOf_eq hp hl]
    unfold strataSizes
    apply List.map_splitLengths_length
    exact (Flock.Bridge.nsOf_sum A.hpop).le.trans (by simp)

/-- **The statement's registration is v2's parameters'**: the statement's descriptor is the parameters' program bytes and
its query their verifier's. -/
def RegIs (A : Live) (p : _root_.SampledProofs.OneStageZk.Params) : Prop :=
  p.own.program = ⟨A.desc.toArray⟩ ∧ p.base.verifier.query = A.qr

/-- **At a session v2 sets up, the registration is v2's parameters'** (`RegIs`), and the statement's descriptor parses
as their verifier's program, at the parameters whose verifier's law the draw reads over the population. -/
theorem regIs_of_setsUp (A : Live) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A) :
    ∃ p, ParamsAt θ p ∧ RegIs A p ∧ lawAt p.base.verifier A.n = .ok A.law ∧
      _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨p, hp, -, -, hl, hd, hq⟩ := hsu.params
  refine ⟨p, hp, ⟨hd, hq⟩, hl, ?_⟩
  rw [← hd]
  exact program_parse_of_paramsAt hp

end Verity.Sampled.Record
