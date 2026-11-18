module

public import InferenceOnlyVerifier.RecordProtocol
public import Proofs.Properties.InferenceOnly.RecordComplies
public import Proofs.Properties.InferenceOnly.RecordProvenance
public import Proofs.Properties.InferenceOnly.RecordJson
import all Lean.Data.Json.FromToJson.Basic

/-!
# What the record protocol's acceptance means

`InferenceOnly.RecordProtocol.check`, the core protocol `verity/inference-only/record/v1`, reads an execution phase's
circuit record with the record check (`Record.read`) and holds its summary to the rate rule in the Program's gates
(`Record.rate`). Where it accepts (`RecordComplies`), on the instance and trusted input it builds from the case:

* the record check reads them, and the record lays out a circuit (`Served`);
* every circuit the record lays out meets the policy block by block at the params' `Γ` and `α`, so it complies
  (`InferenceOnly.CompliesRate`): `form_complies`, with the instance's `Γ` and `α` the params';
* every row the check counts as registered is a read or a table read of a parameter the log's first record
  registers, or a port of an opening whose reader's closure digest is one of the params' pins: `form_provenance`.

What it does not say: that a pinned reader returns the row it opens (`DigestPins`, `OpenedRowReturned`); that a wire
the record declares holds its writer's committed bits (`Vocab.WiresSound`); that a weight the record does not hold is
refused (it is counted as fresh).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json toJson)
open Flock.Program (Prog)
open InferenceOnly.RecordProtocol (Params Evidence instanceOf trustedOf check)
open InferenceOnlyVerifier.Record.SecurityProofs (form form_owns)
open InferenceOnlyVerifier.Record.SecurityProofs.Json (getObjVal?_mkObj)

/-- **The rate rule in the Program's gates on a case of the record protocol**: the record check reads the case's
instance and trusted input, the record lays out a circuit, every circuit it lays out meets the policy block by block
at `Γ` and `α = num / den`, and every row the check holds is registered or opened under a pin (`Provenance`'s
conclusion on that record). -/
def RecordHeld (p : Params) (c : InferenceOnlyVerifier.Protocol.Commitment) (e : Evidence) : Prop :=
  ∃ r : InferenceOnlyVerifier.Record.Case,
    InferenceOnlyVerifier.Record.read (instanceOf p e) (trustedOf p c) = .ok r ∧ (∃ d, Served form r d) ∧
    (∀ d, Served form r d →
      (∀ u, Meets d d.𝓘 p.gamma ((p.num : ℝ) / p.den) u) ∧
        InferenceOnly.CompliesRate p.gamma ((p.num : ℝ) / p.den) d) ∧
    ∀ (P : Prog) (pj : Json), Flock.Program.decode (form.inst r).program.toUTF8 = .ok P →
      Flock.JsonParse.parse (form.inst r).program = .ok pj →
      ∃ log run, TrustedRun (trustedOf p c) log run ∧ RunNames run P (form.inst r).record ∧
        ∀ row h, form.held r row = some h → Good form r P pj log run row h

/-- **Accepting a case of the record protocol gives the rate rule on its circuit**: every case
`InferenceOnly.RecordProtocol.check` accepts is `RecordHeld`. -/
def RecordComplies : Prop :=
  ∀ (p : Params) (c : InferenceOnlyVerifier.Protocol.Commitment) (e : Evidence) (pub : Unit) (prof : Json),
    check p c e = .accept pub prof → RecordHeld p c e

theorem natOf?_toJson (n : ℕ) : natOf? (toJson n) = some n := by
  simp [natOf?, toJson, Lean.JsonNumber.fromNat]

theorem check_accept {p : Params} {c : InferenceOnlyVerifier.Protocol.Commitment} {e : Evidence} {pub : Unit}
    {prof : Json} (h : check p c e = .accept pub prof) :
    ∃ r, InferenceOnlyVerifier.Record.read (instanceOf p e) (trustedOf p c) = .ok r ∧
      InferenceOnlyVerifier.Record.rate p.gamma p.num p.den (InferenceOnlyVerifier.Record.summarize r) = none := by
  simp only [check] at h
  cases hr : InferenceOnlyVerifier.Record.read (instanceOf p e) (trustedOf p c) with
  | error _ => simp [hr] at h
  | ok r =>
    refine ⟨r, rfl, ?_⟩
    cases hq : InferenceOnlyVerifier.Record.rate p.gamma p.num p.den (InferenceOnlyVerifier.Record.summarize r) with
    | none => rfl
    | some _ => simp [hr, hq] at h

/-- The instance the protocol builds carries the params' `Γ` and `α`. -/
theorem instance_fields (p : Params) (e : Evidence) :
    (instanceOf p e).getObjVal? "gamma" = .ok (toJson p.gamma) ∧
      (instanceOf p e).getObjVal? "alpha" = .ok (Json.arr #[toJson p.num, toJson p.den]) := by
  obtain ⟨record, order⟩ := e
  cases order <;>
    exact ⟨(getObjVal?_mkObj (by simp) _ _).2 (by simp), (getObjVal?_mkObj (by simp) _ _).2 (by simp)⟩

/-- A record the protocol's instance reads has the params' `Γ` and `α`. -/
theorem read_limits {p : Params} {c : InferenceOnlyVerifier.Protocol.Commitment} {e : Evidence}
    {r : InferenceOnlyVerifier.Record.Case}
    (h : InferenceOnlyVerifier.Record.read (instanceOf p e) (trustedOf p c) = .ok r) :
    (form.inst r).gamma = p.gamma ∧ (form.inst r).num = p.num ∧ (form.inst r).den = p.den := by
  obtain ⟨-, -, ⟨v, hv, hn⟩, ⟨a, b, hab, ha, hb⟩, -, -⟩ := form_owns _ _ r h
  obtain ⟨hg, hal⟩ := instance_fields p e
  rw [hg] at hv
  cases hv
  rw [hal] at hab
  cases hab
  rw [natOf?_toJson] at hn ha hb
  exact ⟨(Option.some.inj hn).symm, (Option.some.inj ha).symm, (Option.some.inj hb).symm⟩

theorem recordComplies : RecordComplies := by
  intro p c e pub prof h
  obtain ⟨r, hr, hq⟩ := check_accept h
  obtain ⟨hg, hn, hd⟩ := read_limits hr
  have hrate : form.rate (form.inst r).gamma (form.inst r).num (form.inst r).den (form.summarize r) = none := by
    rw [hg, hn, hd]
    exact hq
  obtain ⟨-, hserved, hall⟩ := form_complies _ _ r ⟨hr, hrate⟩
  refine ⟨r, hr, hserved, fun d hd' => ?_, fun P pj hP hpj => form_provenance _ _ r P pj hr hP hpj⟩
  have h2 := hall d hd'
  rw [hg, hn, hd] at h2
  exact h2

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
