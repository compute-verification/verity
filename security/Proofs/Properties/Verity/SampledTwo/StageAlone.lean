module

public import Proofs.Properties.Verity.SampledTwo.StageCut

/-!
# On two windows the first session's read holds none of the second stage's coins

`SecondAlone`, the premise `secondShown_of` takes on two windows, from the first session's roots: where the roots of
the session `drawCoins` reads for the first stage show no coin of the second stage's family after the interior
registration's digest (`FirstUnshown`), the cut's read of that session's log holds no such coins (`secondAlone_of`).
The read places a coin only at a release whose root line names its family and digest (`none_readLog`), and the roots
list every such line (`countFresh_zero`).

The form's check refuses a case whose first session shows such a coin (`interiorUnshown`), but reads that session from
the evidence's `coins` under the evidence's key, on the boundary's window against the interior's, in the family the
interior's segment names, after the interior record's digest; the cut reads it from `firstCoins` under the parameters'
key, against the interior audit's window and family, after the interior commitment's digest. `firstUnshown_of` gives
`FirstUnshown` from the check where those agree (`SessionMatches`, a named `Prop` taken as a hypothesis).
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Spot.Challenger
open Verity.Sampled (freshCoins readLog)

local instance lawfulBEqByteArrayAlone : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## The read places no coin the roots don't show -/

/-- A release that is no coin of `label` after `after` adds none to the log's coins of that family. -/
theorem freshCoins_cons_other {r : Verity.Release} {label : String} {after : ByteArray}
    (h : ∀ l, r.frame = .fresh l → ¬(l = label ∧ r.answers = after)) (rest : List Verity.Release) :
    freshCoins label after (r :: rest) = freshCoins label after rest := by
  unfold freshCoins
  rw [List.filterMap_cons]
  cases hr : r.frame with
  | fresh l =>
    have hn : (if l = label ∧ r.answers = after then some r.coins else none) = none := if_neg (h l hr)
    simp only [hn]
  | hello _ => rfl
  | opening _ _ _ => rfl

theorem none_readTail (label : String) (after : ByteArray) (records : List Json) (label' : String)
    (after' : ByteArray) :
    ∀ (ls : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      {log : List Verity.Release}, Verity.Sampled.readTail label after records ls ts pending ξ = some log →
      Verity.Sampled.countFresh label' after' (pending.toList ++ ls.filter fun l => !treeLine l) = 0 →
      freshCoins label' after' log = []
  | [], _, none, _, _, h, _ => by
    simp only [Verity.Sampled.readTail, Option.some.injEq] at h
    subst h
    rfl
  | [], _, some _, _, _, h, _ => by simp [Verity.Sampled.readTail] at h
  | l :: ls, ts, pending, ξ, log, h, hz => by
    unfold Verity.Sampled.readTail at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      rw [List.filter_cons_of_neg (by simp [ht])] at hz
      simp only [Option.bind_eq_bind] at h
      obtain ⟨⟨r, ts'⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hj⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at h
      obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      rw [Verity.Sampled.freshCoins_cons_step (Verity.Sampled.treeStep_frame hj)]
      exact none_readTail label after records label' after' ls ts' pending ξ hR hz
    · rw [ite_eq_right ht] at h
      rw [List.filter_cons_of_pos (by simp [ht])] at hz
      cases pending with
      | none => exact none_readTail label after records label' after' ls ts (some l) ξ h hz
      | some a =>
        simp only [Option.toList_some, List.cons_append, List.nil_append] at hz
        rw [Verity.Sampled.countFresh] at hz
        obtain ⟨hz0, hif⟩ := Nat.add_eq_zero_iff.1 hz
        have hc : Verity.Sampled.rootOf a ≠ some (label', after') := by
          intro e
          rw [if_pos e] at hif
          omega
        have hz' : Verity.Sampled.countFresh label' after'
            ((none : Option String).toList ++ ls.filter fun l => !treeLine l) = 0 := by
          simpa using hz0
        simp only [Option.bind_eq_bind] at h
        obtain ⟨⟨lab, root⟩, hra, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.pure_def] at h
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          rw [freshCoins_cons_other (fun l' hl => ?_)]
          · exact none_readTail label after records label' after' ls ts none _ hR hz'
          · rintro ⟨h1, h2⟩
            have hl' : label = l' := Verity.Frame.fresh.inj hl
            exact hc (by rw [hra, hm.1, hm.2, hl', h1, show after = after' from h2])
        · rw [ite_eq_right hm] at h
          obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          rw [freshCoins_cons_other (fun l' hl => ?_)]
          · exact none_readTail label after records label' after' ls ts none ξ hR hz'
          · rintro ⟨h1, h2⟩
            have hl' : lab = l' := Verity.Frame.fresh.inj hl
            exact hc (by rw [hra, hl', h1, show root = after' from h2])

/-- **The read places no coin the log's lines don't show**: where no root line of the log after its head names family
`label'` after `after'`, the read of the log (in any family) holds no coins of `label'` after `after'`. -/
theorem none_readLog (label : String) (after : ByteArray) (records : List Json) (ξ : ByteArray) (label' : String)
    (after' : ByteArray) {lines : List String} {log : List Verity.Release}
    (h : readLog label after records ξ lines = some log)
    (hz : Verity.Sampled.countFresh label' after' (lines.tail.filter fun l => !treeLine l) = 0) :
    freshCoins label' after' log = [] := by
  cases lines with
  | nil =>
    simp only [readLog, Option.some.injEq] at h
    subst h
    rfl
  | cons l ls =>
    simp only [readLog] at h
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at h
      simp only [Option.bind_eq_bind] at h
      obtain ⟨⟨r, ts⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hj⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at h
      obtain ⟨rest, hR, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      rw [Verity.Sampled.freshCoins_cons_step (Verity.Sampled.treeStep_frame hj)]
      exact none_readTail label after records label' after' ls ts none ξ hR (by simpa using hz)
    · rw [ite_eq_right ht] at h
      exact none_readTail label after records label' after' ls [] none ξ h (by simpa using hz)

/-- **Roots that don't show a coin count none**: where a session's released roots show no coin of `label` after
`after` (`interiorShown`), no root line of its log after the head names that family and digest. -/
theorem countFresh_zero {c : LiveCommitment} {log : List String} {roots : Roots} {label : String}
    {after : ByteArray} (hr : releasedRootsOf c log = .ok roots)
    (h : VStar.Recursive.SampledTwo.interiorShown roots label after = false) :
    Verity.Sampled.countFresh label after (log.tail.filter fun l => !treeLine l) = 0 := by
  rw [Verity.Sampled.countFresh_eq, ← Verity.Sampled.releasedRootsOf_ok hr label, List.length_eq_zero_iff,
    List.filter_eq_nil_iff]
  intro x hx hxa
  rw [beq_iff_eq] at hxa
  rw [hxa] at hx
  have hm : (Verity.Sampled.look roots label).contains after = true := List.contains_iff_mem.2 hx
  have h' : (Verity.Sampled.look roots label).contains after = false := h
  rw [h'] at hm
  cases hm

/-! ## The premise and the statement -/

/-- **On two windows the first session shows none of the second stage's coins**: on a play the form accepts whose two
sessions are on two windows, the roots of the first session `drawCoins` reads (from `firstCoins`, under the
parameters' key, on the boundary's window) show no coin of the second stage's family after the interior's digest. -/
def FirstUnshown (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e key d₁ d₂ cm roots, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key →
    firstDraws key p c e = some d₁ → secondDraws p e = some d₂ → d₁.window ≠ d₂.window →
    trustedSessionOf (Spot.getD (firstCoins e) "ledger") (Json.mkObj [("root", Json.str d₁.window)]) d₁.key
      (Spot.getD (firstCoins e) "pins") (Spot.getD (firstCoins e) "main") (Spot.getD (firstCoins e) "session") =
        .ok (cm, roots) →
    VStar.Recursive.SampledTwo.interiorShown roots d₂.label d₂.after = false

/-- **`SecondAlone` from the first session's roots.** -/
theorem secondAlone_of (C : CaseOf) (hU : FirstUnshown C) : SecondAlone C := by
  intro o ha p c e key d₁ ξ r₁ d₂ hcase hkey hd₁ hξ hr₁ hd₂ hw
  obtain ⟨lines, hlines, hrl⟩ := firstRead_some hd₁ hr₁
  obtain ⟨ls, hls, hlsl⟩ := firstLines_arr hlines
  obtain ⟨cm, roots, hts, -⟩ := Verity.Sampled.drawCoins_ok hξ
  have hsh := hU o ha p c e key d₁ d₂ cm roots hcase hkey hd₁ hd₂ hw hts
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedSessionOf_ok hts
  obtain ⟨_, hso⟩ := Verity.Sampled.sessionNamed_ok hsn
  obtain ⟨ls', lines', hlog', hls', hrr⟩ := Verity.Sampled.sessionOf_ok hso
  rw [hls] at hlog'
  cases hlog'
  rw [hlsl] at hls'
  have hl : lines' = lines := ((List.map_injective_iff.2 fun _ _ h => Json.str.inj h) hls').symm
  subst hl
  exact none_readLog _ _ _ _ _ _ hrl (countFresh_zero hrr hsh)

/-! ## From the form's check -/

/-- An accepted play's case passed the form's `interiorUnshown`. -/
theorem unshown_of_accepts {C : CaseOf} (hC : CaseIsSent C) {o : Verity.Exchange} (ha : FormAccepts C o)
    {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence}
    (hcase : caseOf o.sent = .ok (p, c, e)) : VStar.Recursive.SampledTwo.interiorUnshown p e = .ok () := by
  obtain ⟨k, hk, p', c', e', _, _, hd, hc⟩ := ha
  obtain ⟨_, _, _, _, _, _, _, _, _, hu, _, _⟩ := VStar.Recursive.SampledTwo.accepted_steps hc
  obtain ⟨hcase', -⟩ := hC o k p' c' e' hk hd
  have hpce := Except.ok.inj (hcase.symm.trans hcase')
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, -, rfl⟩ := hpce
  exact hu

/-- **The check's first session is the cut's**: on a play the form accepts, where the cut reads the first session
`drawCoins` reads and the interior's draw, the form's `interiorUnshown` reads the same session (the evidence's `coins`
name the same ledger, pins, main and session, and its key is the parameters'), the interior's window is the second
draw's, the interior's segment names the second draw's family, and the interior record's digest is the second
draw's. -/
def SessionMatches (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e key d₁ d₂ cm roots, caseOf o.sent = .ok (p, c, e) → keyOf o.sent.params = some key →
    firstDraws key p c e = some d₁ → secondDraws p e = some d₂ →
    trustedSessionOf (Spot.getD (firstCoins e) "ledger") (Json.mkObj [("root", Json.str d₁.window)]) d₁.key
      (Spot.getD (firstCoins e) "pins") (Spot.getD (firstCoins e) "main") (Spot.getD (firstCoins e) "session") =
        .ok (cm, roots) →
    p.interior.verifier.windowRoot = some d₂.window ∧
    Spot.getD (Spot.getD e.interior.base.coins "segment") "label" = .str d₂.label ∧
    Spot.Registration.digest e.record = .ok d₂.after ∧
    ChallengerKey.ofJson (Spot.getD e.coins "key") = .ok d₁.key ∧
    trustedSessionOf (Spot.getD e.coins "ledger") (Json.mkObj [("root", Json.str d₁.window)]) d₁.key
      (Spot.getD e.coins "pins") (Spot.getD e.coins "main") (Spot.getD e.coins "session") = .ok (cm, roots)

/-- **`FirstUnshown` from the form's check**, where it reads the cut's session (`SessionMatches`). -/
theorem firstUnshown_of (C : CaseOf) (hC : CaseIsSent C) (hM : SessionMatches C) : FirstUnshown C := by
  intro o ha p c e key d₁ d₂ cm roots hcase hkey hd₁ hd₂ hw hts
  obtain ⟨hwin, hlab, hdig, hk, hts'⟩ := hM o ha p c e key d₁ d₂ cm roots hcase hkey hd₁ hd₂ hts
  obtain ⟨bp, hb, hbw, -⟩ := firstDraws_window hd₁
  have hbp : bp = p.boundary := by
    unfold boundaryOf at hb
    first
      | exact (Except.ok.inj hb).symm
      | (cases hb; rfl)
      | (simp only [pure, Except.pure, Except.ok.injEq] at hb; exact hb.symm)
  subst hbp
  refine VStar.Recursive.SampledTwo.interiorUnshown_roots (unshown_of_accepts hC ha hcase) hbw ?_ hlab hdig hk hts'
  rw [hwin]
  exact fun h => hw (Option.some.inj h)

/-- **`SecondAlone` from the form's check**, where it reads the cut's session. -/
theorem secondAlone_of_matches (C : CaseOf) (hC : CaseIsSent C) (hM : SessionMatches C) : SecondAlone C :=
  secondAlone_of C (firstUnshown_of C hC hM)

end Verity.SampledTwo
