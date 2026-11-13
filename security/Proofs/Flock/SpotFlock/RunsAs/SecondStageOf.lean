module

public import Proofs.Flock.SpotFlock.RunsAs.Round2
public import Proofs.Flock.SpotFlock.Facts

/-!
# The second stage's check gives the model's statement check, with what the design decides

`RunsAs.Round2`'s `stmtOk_of_check` takes two premises the second stage's check does not give: the templates' names
are distinct (`NamesDistinct`), and the first stage's interior record registers no value (`InputsNone`, from
`RegistersNone`). The model's design decides both on the auditor's copies (`two`, by `wiringOk`: `namesOnce`, and the
interior registration registering no value), so where the design evaluates (`TwoHolds`) neither is a premise:

* `namesDistinct_of_two`: the templates' names are distinct (`names_nodup_of_two`);
* `values_nil_of_two`: the interior registration registers no value;
* `registersNone_of_recordOk`: a record that names a registration's names registers no value where the registration
  registers none;
* `registersNone_of_two`, `inputsNone_of_two`: so the interior record of a run whose records name their registrations'
  names (`RecordsHeld`) registers none, and the second stage's `second-inputs` passes with nothing of the verifier's
  own;
* `stmtOk_of_secondStage`: the model's statement check from a passing second stage, with neither premise.

**Which designs these are true of.** The design evaluates only where the interior registration registers no value, so
`TwoHolds` is false at a design whose interior registration registers one, such as the replay block's `logits`; there
these lemmas say nothing, and the model's verifier accepts no run.
-/

@[expose] public section

namespace SpotFlock.RunsAs.Round2

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)
open Properties.Spot (bitsProgs)

/-! ## A record at a registration that registers no value -/

/-- A list read entry by entry as the empty list is empty. -/
theorem nil_of_mapM_nil {α β : Type} (f : α → Option β) : ∀ l : List α, l.mapM f = some [] → l = []
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons] at h
    cases hfa : f a with
    | none =>
      rw [hfa] at h
      cases h
    | some _ =>
      rw [hfa] at h
      cases hl : l.mapM f with
      | none =>
        rw [hl] at h
        cases h
      | some _ =>
        rw [hl] at h
        cases h

/-- Names that are all among no names are none. -/
theorem nil_of_all_contains_nil {ns : List String} (h : ns.all (fun x => ([] : List String).contains x) = true) :
    ns = [] := by
  cases ns with
  | nil => rfl
  | cons a t =>
    rw [List.all_cons] at h
    cases h

/-- **A record that names a registration's names registers no value where the registration registers none**
(`recordOk` at names with no values): its `registered` has no entry. -/
theorem registersNone_of_recordOk {rec : Json} {n : Names} (hv : n.values = []) (hr : recordOk rec n = true) :
    RegistersNone rec := by
  intro es hes e he v _
  unfold recordOk at hr
  simp only [Bool.and_eq_true] at hr
  obtain ⟨-, hr2⟩ := hr
  rw [hv] at hr2
  unfold namesEq at hr2
  cases hna : namesAt rec "registered" "value" with
  | none =>
    rw [hna] at hr2
    cases hr2
  | some ns =>
    rw [hna] at hr2
    simp only [Bool.and_eq_true] at hr2
    obtain ⟨⟨-, hall⟩, -⟩ := hr2
    have hns : ns = [] := nil_of_all_contains_nil hall
    subst hns
    unfold namesAt at hna
    rw [hes] at hna
    have hl : es.toList = [] := nil_of_mapM_nil _ es.toList hna
    have hmem : e ∈ es.toList := Array.mem_toList_iff.2 he
    rw [hl] at hmem
    cases hmem

/-! ## What the design decides -/

/-- Where the design evaluates at every statement, it evaluates at one. -/
theorem two_of_twoHolds {h : Held} (hT : TwoHolds h) :
    ∃ (s : AuditStatement) (d : Design.Two), two h s = some d := by
  have s : AuditStatement := AuditStatement.mk h.desc (bitsProgs h.ms) h.query (Design.mk []) []
  exact ⟨s, Option.isSome_iff_exists.1 (hT s)⟩

/-- **Where the design evaluates, the templates' names are distinct** (`names_nodup_of_two`). -/
theorem namesDistinct_of_two {h : Held} (hT : TwoHolds h) : NamesDistinct h := by
  obtain ⟨_, _, hd⟩ := two_of_twoHolds hT
  exact names_nodup_of_two h hd

/-- **Where the design evaluates, the interior registration registers no value**: the clause of the registered
wiring's check (`wiringOk`), read wherever it stands among the check's clauses. -/
theorem values_nil_of_two {h : Held} {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    h.interiorNames.values = [] := by
  have hw : wiringOk h = true := (scoped_wiring_of_two h hd).2
  unfold wiringOk at hw
  simp only [Bool.and_eq_true] at hw
  exact List.isEmpty_iff.1 (by simp only [hw])

/-- **Where the design evaluates, the interior record of a run whose records name their registrations' names
registers no value**: the design evaluates only where the interior registration registers none
(`values_nil_of_two`). -/
theorem registersNone_of_two {h : Held} (hT : TwoHolds h) {x m₁ : Msg} {rec : Json}
    (hm₁ : jsonMsg m₁ = some rec) (hRec : RecordsHeld h x m₁) : RegistersNone rec := by
  obtain ⟨_, _, hd⟩ := two_of_twoHolds hT
  have hv : h.interiorNames.values = [] := values_nil_of_two hd
  unfold RecordsHeld recordsOk at hRec
  rw [hm₁] at hRec
  cases hdx : decode x with
  | none =>
    rw [hdx] at hRec
    cases hRec
  | some _ =>
    rw [hdx] at hRec
    simp only [Bool.and_eq_true] at hRec
    exact registersNone_of_recordOk hv hRec.2

/-- **Where the design evaluates, the second stage's `second-inputs` passes with nothing of the verifier's own**
(`InputsNone`), at a running verifier that holds what the model holds of a run whose records name their
registrations' names. -/
theorem inputsNone_of_two {h : Held} (hT : TwoHolds h) {own : Spot.SecondStage.Own} {x m₁ : Msg} {st : Roots}
    {rec draw : Json} {c₀ : FlockDeploy.Coin} (hm₁ : jsonMsg m₁ = some rec)
    (hA : OwnAgrees h own st rec draw c₀) (hRec : RecordsHeld h x m₁) : InputsNone own := by
  have hi : own.interior = rec := congrArg Spot.SecondStage.Own.interior hA
  apply inputsNone_of_registersNone
  rw [hi]
  exact registersNone_of_two hT hm₁ hRec

/-! ## The statement check -/

/-- **A passing second stage gives the model's statement check** (`stmtOk`), where the design evaluates: the running
verifier holds what the model holds of the run (`OwnAgrees`), the statements the second stage checked are the session
statements' (`StmtsAre`), the run's records name their registrations' names (`RecordsHeld`), and each session
statement has its template's reference statement's rounds, is in scope and is the reference statement but for the
run's fields (`RoundsHeld`, `ScopeHeld`, `PinHeld`). It is `stmtOk_of_check` without its `InputsNone` and
`NamesDistinct` premises, which the design decides. -/
theorem stmtOk_of_secondStage (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (st : Roots) (rec draw : Json) (hx : decode x = some st)
    (hm₁ : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some draw) (hA : OwnAgrees h own st rec draw c₀)
    (hS : StmtsAre h m₂ ss) (hT : TwoHolds h) (hRec : RecordsHeld h x m₁) (hR : RoundsHeld h m₂)
    (hSc : ScopeHeld h m₂) (hP : PinHeld h m₂) (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) :
    stmtOk h x m₁ m₂ c₀ c₁ = true :=
  stmtOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS (inputsNone_of_two hT hm₁ hA hRec) hT
    (namesDistinct_of_two hT) hRec hR hSc hP hc

end SpotFlock.RunsAs.Round2
