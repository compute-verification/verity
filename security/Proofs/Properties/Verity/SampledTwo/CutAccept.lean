module

public import Proofs.Properties.Verity.SampledTwo.CutOrder
public import Proofs.Properties.Verity.SampledTwo.CutFirst
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Pouw.LiveSalt.Ledger

/-! The cut from the form's acceptance (`cutBound_accept`), with one premise about the device,
`PrintsInOrder`, which no theorem here proves; an earlier period's genuine signature is the forger's. -/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open scoped ENNReal

theorem evidence_of_caseOf {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} (h : caseOf s = .ok (p, c, e)) :
    ∃ b, VStar.Recursive.SampledTwo.evidenceOf b = .ok e := by
  unfold caseOf at h
  obtain ⟨p', -, h⟩ := bind_ok h
  obtain ⟨ce, hce, h⟩ := bind_ok h
  have he : ce.2 = e := by
    exact congrArg (fun x => x.2.2) (Except.ok.inj h)
  subst he
  unfold caseIn at hce
  obtain ⟨ev, -, hce⟩ := bind_ok hce
  split at hce
  · obtain ⟨c', -, hce⟩ := bind_ok hce
    obtain ⟨e', he', hce⟩ := bind_ok hce
    have hpair : (c', e') = ce := Except.ok.inj hce
    exact ⟨_, by rw [← hpair]; exact he'⟩
  · cases hce

private theorem evidence_coins {b : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence}
    (h : VStar.Recursive.SampledTwo.evidenceOf b = .ok e) : e.coins ≠ Json.null := by
  replace h := (VStar.Recursive.SampledTwo.evidenceOf_ok h).1
  unfold VStar.Recursive.SampledTwo.evidenceParsed at h
  obtain ⟨j₀, -, h⟩ := bind_ok h
  obtain ⟨j, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  dsimp only at h
  generalize Spot.getD j "coins" = cj at h
  cases cj
  case null =>
    simp only [↓reduceIte] at h
    obtain ⟨_, ht, -⟩ := bind_ok h
    cases ht
  all_goals
    simp only [Bool.false_eq_true, ↓reduceIte] at h
    split at h
    · split at h
      · obtain ⟨src, -, h⟩ := bind_ok h
        split at h
        · split at h
          · obtain ⟨_, -, h⟩ := bind_ok h
            obtain ⟨it, -, h⟩ := bind_ok h
            split at h
            · have he := Except.ok.inj h
              rw [← he]
              intro hx
              cases hx
            · cases h
          · cases h
        · cases h
      · cases h
    · cases h

theorem firstShown (θ : ByteArray) : FirstCoinsShown θ := by
  unfold FirstCoinsShown
  intro s _ _ p c e h1
  obtain ⟨b, hb⟩ := evidence_of_caseOf h1
  exact evidence_coins hb

theorem firstReads (θ : ByteArray) : FirstReads θ :=
  firstReads_of_shown θ (firstShown θ)

private theorem ordered_order {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    (h : VStar.Recursive.SampledTwo.stagesOrdered p e = .ok ())
    {k : Spot.Challenger.ChallengerKey} {label w₁ w₂ : String} (hk : p.audit.base.challenger = some (k, label))
    (h1 : p.boundary.verifier.windowRoot = some w₁) (h2 : p.audit.base.verifier.windowRoot = some w₂)
    (hne : w₁ ≠ w₂) :
    ∃ first second, Spot.getD e.coins "ledger" = .arr first ∧
      Spot.getD e.interior.base.coins "ledger" = .arr second ∧ Spot.Challenger.extendsHead first second = true ∧
      ∃ wins ms, Spot.Challenger.checked (.arr second) k = .ok (wins, ms) ∧
        ∃ en r, (wins.find? (·.root == w₁)).bind (·.ended) = some en ∧
          ((wins.find? (·.root == w₂)).bind fun w => (second.filterMap Spot.Challenger.ledgerEntry?).find? fun l =>
            l.kind == "reserve" && l.session == w.opened.session) = some r ∧ en.seq < r.seq := by
  have hb : (w₁ == w₂) = false := by simpa using hne
  unfold VStar.Recursive.SampledTwo.stagesOrdered at h
  rw [hk, h1, h2] at h
  simp only [hb, Bool.false_eq_true, ↓reduceIte] at h
  generalize hf : Spot.getD e.coins "ledger" = f at h
  generalize hs : Spot.getD e.interior.base.coins "ledger" = s at h
  cases f <;> cases s <;> first | (cases h; done) | skip
  case arr.arr first second =>
    dsimp only at h
    cases hx : Spot.Challenger.extendsHead first second
    · rw [hx] at h
      simp only [Bool.not_false, ↓reduceIte] at h
      obtain ⟨_, hthrow, -⟩ := bind_ok h
      cases hthrow
    · rw [hx] at h
      simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
      obtain ⟨⟨wins, ms⟩, hc, h⟩ := bind_ok h
      dsimp only at h
      split at h
      · cases h
      · rename_i heq
        obtain ⟨en, r, he, hr, hlt⟩ := VStar.Recursive.SampledTwo.orderReason_none heq
        exact ⟨first, second, rfl, rfl, hx, wins, ms, hc, en, r, he, hr, hlt⟩

theorem option_bind_ok {α β : Type} {o : Option α} {f : α → Option β} {b : β} (h : o.bind f = some b) :
    ∃ a, o = some a ∧ f a = some b := by
  cases o with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

theorem executionPhasesOf_some {θ : ByteArray} {p : VStar.Recursive.SampledTwo.Params} (hp : paramsAt θ = .ok p)
    {w₁ w₂ : String} (h : executionPhasesOf θ = some (w₁, w₂)) :
    p.boundary.verifier.windowRoot = some w₁ ∧ p.audit.base.verifier.windowRoot = some w₂ := by
  have hb : boundaryOf p = .ok p.boundary := rfl
  unfold executionPhasesOf at h
  rw [hp] at h
  cases h1 : p.boundary.verifier.windowRoot with
  | none => simp [Except.toOption, hb, h1] at h
  | some a =>
    cases h2 : p.audit.base.verifier.windowRoot with
    | none => simp [Except.toOption, hb, h2] at h
    | some b =>
      have h' : some (a, b) = some (w₁, w₂) := by
        rw [← h]
        simp [Except.toOption, hb, h1, h2]
      have hpair := Option.some.inj h'
      simp only [Prod.mk.injEq] at hpair
      obtain ⟨rfl, rfl⟩ := hpair
      exact ⟨rfl, rfl⟩

theorem executionPhasesOf_ne {θ : ByteArray} {w₁ w₂ : String} (h : executionPhasesOf θ = some (w₁, w₂)) : w₁ ≠ w₂ := by
  cases hp : paramsAt θ with
  | error err =>
    unfold executionPhasesOf at h
    rw [hp] at h
    simp [Except.toOption] at h
  | ok p =>
    obtain ⟨h₁, h₂⟩ := executionPhasesOf_some hp h
    obtain ⟨j, hj⟩ := paramsAt_form hp
    have hne := (VStar.Recursive.SampledTwo.paramsOfJson_ok hj).2
    rw [h₁, h₂] at hne
    exact fun hw => hne (congrArg some hw)

theorem win_lines {es : Array LedgerEntry} {key : ChallengerKey} {wins : Array Win}
    {ms : Array Spot.Challenger.PeriodRecord} (h : verifyLedger es key = .ok (wins, ms)) {w : String} {W : Win}
    (hf : wins.find? (·.root == w) = some W) :
    W.opened ∈ es.toList ∧ W.opened.kind = "open" ∧ openedRoot W.opened = w ∧
      ∀ en, W.ended = some en → en ∈ es.toList ∧ en.kind = "end" ∧ en.session = W.opened.session := by
  obtain ⟨v2, hI⟩ := Properties.Pouw.LiveSalt.Ledger.verifyLedger_inv h
  have hI : Properties.Pouw.LiveSalt.Ledger.LedgerInv es.toList wins v2 := hI
  obtain ⟨i, hi, hW⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hf)
  have hroot : W.root = w := by
    have hp := Array.find?_some hf
    exact beq_iff_eq.1 hp
  obtain ⟨hm, hk, hr, -⟩ := hI.win i hi
  rw [hW] at hm hk hr
  refine ⟨hm, hk, ?_, fun en he => ?_⟩
  · exact hr.symm.trans hroot
  · obtain ⟨h1, h2, h3, -⟩ := hI.ended i hi en (by rw [hW]; exact he)
    rw [hW] at h3
    exact ⟨h1, h2, h3⟩

theorem reserve_line {second : Array Json} {sid : String} {r : LedgerEntry}
    (h : (second.filterMap ledgerEntry?).find? (fun l => l.kind == "reserve" && l.session == sid) = some r) :
    r ∈ Properties.Pouw.LiveSalt.entries (Json.arr second) ∧ r.kind = "reserve" ∧ r.session = sid := by
  have hp := Array.find?_some h
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  have hm := Array.mem_of_find?_eq_some h
  refine ⟨?_, hp.1, hp.2⟩
  obtain ⟨line, hline, hle⟩ := Array.mem_filterMap.1 hm
  exact List.mem_filterMap.2 ⟨line, Array.mem_toList_iff.2 hline, hle⟩

theorem orderShown (θ : ByteArray) : OrderShown θ := by
  unfold OrderShown
  intro s hθ ha p c e h1 w₁ w₂ hw hne
  obtain ⟨p', c', e', pub, prof, h1', hacc⟩ := ha
  have hpce := Except.ok.inj (h1.symm.trans h1')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  have hpθ : paramsAt θ = .ok p := by
    rw [← hθ]
    exact paramsAt_of_caseOf h1
  obtain ⟨hw₁, hw₂⟩ := executionPhasesOf_some hpθ hw
  obtain ⟨k, label, hk, -⟩ := keyOf_some hpθ
  obtain ⟨first, second, -, hsec, -, wins, ms, hch, en, r, hen, hr, hlt⟩ :=
    ordered_order (VStar.Recursive.SampledTwo.accepted_ordered hacc) hk hw₁ hw₂ hne
  obtain ⟨es, hes, hvl⟩ := Properties.Pouw.LiveSalt.Ledger.checked_ok hch
  obtain ⟨W₁, hf₁, he₁⟩ := option_bind_ok hen
  obtain ⟨W₂, hf₂, hr₂⟩ := option_bind_ok hr
  obtain ⟨ho₁, hk₁, hr₁, hend⟩ := win_lines hvl hf₁
  obtain ⟨hen₁, hke, hse⟩ := hend en he₁
  obtain ⟨ho₂, hk₂, hroot₂, -⟩ := win_lines hvl hf₂
  obtain ⟨hrv, hkr, hsr⟩ := reserve_line hr₂
  have hmem : ∀ x : LedgerEntry, x ∈ es.toList →
      x ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger") := by
    intro x hx
    rw [hsec, hes]
    exact hx
  have hrv' : r ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger") := by
    rw [hsec]
    exact hrv
  exact ⟨W₁.opened, hmem _ ho₁, en, hmem _ hen₁, r, hrv', W₂.opened, hmem _ ho₂, hk₁, hr₁, hke, hse, hk₂, hroot₂,
    hkr, hsr, hlt⟩

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem cutBound_accept (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hw : Wired V k θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) →
      PrintsInOrder (dev o) w₁ w₂ o.log)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ ReadIsDevice (terms V dev θ F c σ) :=
  cutBound_order V k dev θ F c σ hw (firstReads θ) (orderShown θ) hP hs

end Verity.SampledTwo
