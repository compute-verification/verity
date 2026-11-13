module

public import Proofs.Properties.Verity.SampledTwo.SessionLogged
public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.SampledTwo.CutRead

/-!
# The interior's log on the play's, at the cut's read

`SessionLogged.interiorLogged_of` gives the sessions' coins' premise (`InteriorLogged`) from what the cut's read gives
of the interior's. This module reads that off the cut's own event, `ReadIsDevice`: the check read a log of both
sessions (`LogRead`) and every log it read is the device's (`LogIsDevice`).

* `readsLogs_some`: each step of the cut's read, where it read a log, and the logs the two sessions' reads make of
  the play's (`logsOf`).
* `secondRead_interior`: the cut's read of the interior's session (`secondRead`) is the read the sessions' coins are
  stated at (`interiorRead`), where the play's case is the one sent.
* `interiorLogged_read`: `InteriorLogged C ReadIsDevice`, where the play's case is the one sent and, on two windows,
  the first session's read shows no coin tree's `Hello` (`FirstTreeless`).
* `firstTreeless_of_plain`: `FirstTreeless`, where no line of the first session's log is a coin tree's on two
  windows.
* `heldSize`, `coinsRead_read`: the sessions' coins at the cut's read, with nothing assumed but that the case is the
  one sent and `FirstTreeless`.

The event has to be `ReadIsDevice`: `LogIsDevice` alone holds of any log on a play whose logs the cut didn't read, and
says nothing of the interior's there.

`FirstTreeless` is not proved here. The form's check refuses, on two windows, a first session's log that holds a
tree's line (`VStar.Recursive.SampledTwo.firstTreeless`), and `SessionTreeless.firstTreeless_of_check` proves
`FirstTreeless` from that refusal through `firstTreeless_of_plain`.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled.Binding (Read)
open Properties.Challenger Spot.Challenger

/-- **On two windows the first session samples no coin tree**: on a play the form accepts whose two sessions are on
two windows, the first session's releases as the cut reads them hold no coin tree's `Hello`. -/
def FirstTreeless (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e key d₁ ξ r₁ d₂, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key →
    firstDraws key p c e = some d₁ → drawCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) = .ok ξ →
    firstRead key p c e ξ = some r₁ → secondDraws p e = some d₂ → d₁.window ≠ d₂.window →
    ∀ r ∈ r₁, ∀ sp, r.frame ≠ .hello sp

theorem getD_nil_some {α : Type} {x : Option (List α)} (h : x.getD [] ≠ []) : ∃ L, x = some L := by
  cases x with
  | none => exact absurd rfl h
  | some L => exact ⟨L, rfl⟩

/-- **What the cut read, where it read a log** (`readsLogs`, `LogRead`): each step of its read, and the logs the two
sessions' reads make of the play's log. -/
theorem readsLogs_some {s : Verity.Sent} (h : readsLogs s ≠ []) :
    ∃ p c e key d₁ ξ c₂ d₂ ξ₂ r₁ r₂, caseOf s = .ok (p, c, e) ∧ keyOf s.params = some key ∧
      firstDraws key p c e = some d₁ ∧
      drawCoins d₁.key d₁.window d₁.label d₁.after (firstCoins e) = .ok ξ ∧
      interiorOf e = .ok c₂ ∧ secondDraws p e = some d₂ ∧
      Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧ firstRead key p c e ξ = some r₁ ∧
      secondRead p e (_root_.Core.Blobs.ofArray s.blobs) ξ₂ = some r₂ ∧
      readsLogs s = logsOf d₁.window d₂.window r₁ r₂ := by
  have h' := h
  unfold readsLogs at h'
  obtain ⟨L, hL⟩ := getD_nil_some h'
  have hE : readsLogs s = L := by
    unfold readsLogs
    rw [hL]
    rfl
  obtain ⟨⟨p, c, e⟩, h1, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨key, h2, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨d₁, h3, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨ξ, h4, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨c₂, h5, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨d₂, h6, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨ξ₂, h7, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨r₁, h8, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain ⟨r₂, h9, hL⟩ := Verity.Sampled.Binding.obind_some hL
  obtain rfl : logsOf d₁.window d₂.window r₁ r₂ = L := Option.some.inj hL
  exact ⟨p, c, e, key, d₁, ξ, c₂, d₂, ξ₂, r₁, r₂, toOption_ok h1, h2, h3, toOption_ok h4, toOption_ok h5, h6,
    toOption_ok h7, h8, h9, hE⟩

/-- **The cut's read of the interior's session is the read the sessions' coins are stated at** (`interiorRead`),
where the play's case is the one sent (`hC`, in the words of `StageCut`'s `CaseIsSent C`): wherever the case decodes,
the cut decodes what the prover sent to the same parameters, commitment and evidence, and the case's blobs are those
sent. -/
theorem secondRead_interior {C : CaseOf}
    (hC : ∀ o k p c e, C o = some k → Decoded k p c e →
      caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs)
    {o : Exchange} (hacc : FormAccepts C o)
    {l : List Release} (hl : interiorRead C o = some l) {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} (h1 : caseOf o.sent = .ok (p, c, e))
    {c₂ : Spot.OneStage.Commitment} (h5 : interiorOf e = .ok c₂) {ξ₂ : ByteArray}
    (h7 : Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂) :
    secondRead p e (_root_.Core.Blobs.ofArray o.sent.blobs) ξ₂ = some l := by
  obtain ⟨k, p', e', s, c₂', d, ξ₂', recs, lines, hk, hp', he', hs, hc₂', hd, hξ', hrecs, hlines, hR⟩ :=
    interiorRead_some hl
  obtain ⟨k', hk', p'', c'', e'', pub, prof, hdec, -⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain ⟨hcase, hblobs⟩ := hC o k p'' c'' e'' hk hdec
  obtain ⟨hp'', -, he''⟩ := hdec
  obtain rfl : p' = p'' := Except.ok.inj (hp'.symm.trans hp'')
  obtain rfl : e' = e'' := Except.ok.inj (he'.symm.trans he'')
  have hpce : (p, c, e) = (p', c'', e') := Except.ok.inj (h1.symm.trans hcase)
  obtain ⟨rfl, hce⟩ := Prod.mk.inj hpce
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hce
  have hi : interiorOf e = .ok c₂' := by
    unfold interiorOf
    rw [hs]
    exact hc₂'
  obtain rfl : c₂ = c₂' := Except.ok.inj (h5.symm.trans hi)
  obtain rfl : ξ₂ = ξ₂' := Except.ok.inj (h7.symm.trans hξ')
  have h6 : secondDraws p e = some d := by
    unfold secondDraws
    rw [h5]
    exact hd
  have hr : secondRecords (_root_.Core.Blobs.ofArray o.sent.blobs) e = some recs := by
    rw [← hblobs]
    exact hrecs
  unfold secondRead
  rw [h6, hr, hlines]
  exact hR

/-- **The interior's log on the play's, at the cut's read** (`InteriorLogged` at `ReadIsDevice`): on a play the form
accepts whose logs the cut read and found the device's, the interior's read is the play's log on one window, and the
play's log after the first session's releases on two, which sample and open no coin tree where the first session's
read shows no `Hello` (`FirstTreeless`, `readLog_treeless`). The play's case is the one sent (`hC`, as in
`secondRead_interior`). -/
theorem interiorLogged_read (C : CaseOf)
    (hC : ∀ o k p c e, C o = some k → Decoded k p c e →
      caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs)
    (hfree : FirstTreeless C) :
    InteriorLogged C ReadIsDevice := by
  intro o hacc hgood l hl
  obtain ⟨hread, hdev⟩ := hgood
  obtain ⟨p, c, e, key, d₁, ξ, c₂, d₂, ξ₂, r₁, r₂, h1, h2, h3, h4, h5, h6, h7, h8, h9, hE⟩ := readsLogs_some hread
  obtain rfl : r₂ = l := Option.some.inj (h9.symm.trans (secondRead_interior hC hacc hl h1 h5 h7))
  by_cases hw : d₁.window = d₂.window
  · have hm : r₂ ∈ readsLogs o.sent := by
      rw [hE, logsOf, ite_eq_left hw]
      exact List.mem_cons_of_mem _ List.mem_cons_self
    exact ⟨[], o.log, rfl, fun r hr => absurd hr List.not_mem_nil, hdev r₂ hm⟩
  · have hm : r₁ ++ r₂ ∈ readsLogs o.sent := by
      rw [hE, logsOf, ite_eq_right hw]
      exact List.mem_cons_self
    have hno := hfree o hacc p c e key d₁ ξ r₁ d₂ h1 h2 h3 h4 h8 h6 hw
    have hop : ∀ r ∈ r₁, openTree r.frame = none := by
      unfold firstRead at h8
      obtain ⟨d, -, h8⟩ := Verity.Sampled.Binding.obind_some h8
      obtain ⟨lines₁, -, h8⟩ := Verity.Sampled.Binding.obind_some h8
      exact readLog_treeless h8 hno
    refine interiorLogged_of (first := fun _ => r₁) (Good := fun o' => o' = o) ?_ ?_ o hacc rfl r₂ hl
    · intro o' _ ho' l' hl'
      subst ho'
      obtain rfl : l' = r₂ := Option.some.inj (hl'.symm.trans hl)
      exact .inr (hdev _ hm)
    · intro o' _ _ r hr
      exact ⟨hop r hr, hno r hr⟩

/-- **`FirstTreeless`, from a first session's log of no coin tree's line**: where, on a play the form accepts on two
windows, no line of the first session's log is a coin tree's (`Spot.Challenger.treeLine`), its read shows only fresh
coins (`readLog_plain`). The form's check refuses any other log on two windows, which is how
`SessionTreeless.firstTreeless_of_check` reaches this. -/
theorem firstTreeless_of_plain (C : CaseOf)
    (h : ∀ o, FormAccepts C o → ∀ p c e key d₁ d₂ ls, caseOf o.sent = .ok (p, c, e) →
      keyOf o.sent.params = some key → firstDraws key p c e = some d₁ → secondDraws p e = some d₂ →
      d₁.window ≠ d₂.window → firstLines e = .ok ls → ∀ l ∈ ls, Spot.Challenger.treeLine l = false) :
    FirstTreeless C := by
  intro o hacc p c e key d₁ ξ r₁ d₂ h1 h2 h3 _ h8 h6 hw r hr sp hsp
  unfold firstRead at h8
  obtain ⟨d, -, h8⟩ := Verity.Sampled.Binding.obind_some h8
  obtain ⟨ls, hls, h8⟩ := Verity.Sampled.Binding.obind_some h8
  obtain ⟨lab, hlab⟩ := readLog_plain h8 (h o hacc p c e key d₁ d₂ ls h1 h2 h3 h6 hw (toOption_ok hls)) r hr
  rw [hlab] at hsp
  cases hsp

/-! ## The sessions' coins at the cut's read -/

/-- **`membersHeld` returns a member for each statement** (`HeldSize`): the cut's `held_size`. -/
theorem heldSize : HeldSize := fun _ _ _ h => held_size h

/-- **The sessions' coins the check reads are the device's, at the cut's read** (`coinsRead_sessions` at
`ReadIsDevice`): on a play the form accepts whose logs the cut read and found the device's, wherever the check read a
coin for a session's round, the device's release for that round is that coin. The play's case is the one sent (`hC`,
as in `secondRead_interior`) and, on two windows, the first session's read shows no `Hello` (`FirstTreeless`). -/
theorem coinsRead_read (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (hdev : DevLocal A (locOf A tab) dev)
    (hC : ∀ o k p c e, C o = some k → Decoded k p c e →
      caseOf o.sent = .ok (p, c, e) ∧ k.blobs = _root_.Core.Blobs.ofArray o.sent.blobs)
    (hfree : FirstTreeless C) :
    CoinsReadSome A (formReading A C (withCoins A C tab dev Ms) dev) (FormAccepts C) ReadIsDevice .sessions :=
  coinsRead_sessions A C Ms tab dev ReadIsDevice hdev (interiorLogged_read C hC hfree)

end Verity.SampledTwo.SessionCoins
