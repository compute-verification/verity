module

public import Proofs.Flock.SpotFlock.RunsAs.Round2
public import Proofs.Flock.SpotFlock.Facts

/-! The second stage's check gives the model's statement check where the design evaluates (`stmtOk_of_secondStage`);
at a design whose interior registration registers a value, such as the replay block's `logits`, these say
nothing. -/

@[expose] public section

namespace SpotFlock.RunsAs.Round2

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)
open Properties.Spot (bitsProgs)

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

theorem nil_of_all_contains_nil {ns : List String} (h : ns.all (fun x => ([] : List String).contains x) = true) :
    ns = [] := by
  cases ns with
  | nil => rfl
  | cons a t =>
    rw [List.all_cons] at h
    cases h

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

theorem two_of_twoHolds {h : Held} (hT : TwoHolds h) :
    ∃ (s : AuditStatement) (d : Design.Two), two h s = some d := by
  have s : AuditStatement := AuditStatement.mk h.desc (bitsProgs h.ms) h.query (Design.mk []) []
  exact ⟨s, Option.isSome_iff_exists.1 (hT s)⟩

theorem namesDistinct_of_two {h : Held} (hT : TwoHolds h) : NamesDistinct h := by
  obtain ⟨_, _, hd⟩ := two_of_twoHolds hT
  exact names_nodup_of_two h hd

theorem values_nil_of_two {h : Held} {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    h.interiorNames.values = [] := by
  have hw : wiringOk h = true := (scoped_wiring_of_two h hd).2
  unfold wiringOk at hw
  simp only [Bool.and_eq_true] at hw
  exact List.isEmpty_iff.1 (by simp only [hw])

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

theorem inputsNone_of_two {h : Held} (hT : TwoHolds h) {own : Spot.SecondStage.Own} {x m₁ : Msg} {st : Roots}
    {rec draw : Json} {c₀ : FlockDeploy.Coin} (hm₁ : jsonMsg m₁ = some rec)
    (hA : OwnAgrees h own st rec draw c₀) (hRec : RecordsHeld h x m₁) : InputsNone own := by
  have hi : own.interior = rec := congrArg Spot.SecondStage.Own.interior hA
  apply inputsNone_of_registersNone
  rw [hi]
  exact registersNone_of_two hT hm₁ hRec

theorem stmtOk_of_secondStage (h : Held) (own : Spot.SecondStage.Own) (ss : Array Json) (x m₁ m₂ : Msg)
    (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (st : Roots) (rec draw : Json) (hx : decode x = some st)
    (hm₁ : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some draw) (hA : OwnAgrees h own st rec draw c₀)
    (hS : StmtsAre h m₂ ss) (hT : TwoHolds h) (hRec : RecordsHeld h x m₁) (hR : RoundsHeld h m₂)
    (hSc : ScopeHeld h m₂) (hP : PinHeld h m₂) (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) :
    stmtOk h x m₁ m₂ c₀ c₁ = true :=
  stmtOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS (inputsNone_of_two hT hm₁ hA hRec) hT
    (namesDistinct_of_two hT) hRec hR hSc hP hc

end SpotFlock.RunsAs.Round2
