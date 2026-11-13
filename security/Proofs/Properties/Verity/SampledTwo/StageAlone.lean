module

public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Challenger.Coins
public import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# On two windows the first session's read holds none of the second stage's coins

`SecondAlone`, the premise `secondShown_of` takes on two windows, from the first session's roots: where the roots of
the session `drawCoins` reads for the first stage show no coin of the second stage's family after the interior
registration's digest (`FirstUnshown`), the cut's read of that session's log holds no such coins (`secondAlone_of`).
The read places a coin only at a release whose root line names its family and digest (`none_readLog`), and the roots
list every such line (`countFresh_zero`).

The form's check refuses a case whose first session shows such a coin (`interiorUnshown`). It reads that session
under the interior audit's challenger key, on the boundary's window against the interior audit's, in the interior
audit's family, as the cut does (`keyOf_challenger`, `secondDraws_parts`); it differs from the cut in that it reads
the digest of the interior record where the cut reads the interior commitment's. `firstUnshown_of` gives
`FirstUnshown` from the check where those agree (`SessionMatches`), and the form refuses unless they do
(`interiorDigestIs`, `INTERIOR_DIGEST`), so `SessionMatches` is the form's check (`sessionMatches_of`).

`FirstDrawn`, the premise `firstShown_of` takes, is the form's check (`firstDrawn_of`): the run of record's coin check
reads the cut's first sample, at the record the form holds to the commitment's digest (`digestIs`).
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

/-- **The parameters' key is the interior audit's challenger's**, on the parameters the case decodes. -/
theorem keyOf_challenger {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} {key : ChallengerKey} (hcase : caseOf s = .ok (p, c, e))
    (hkey : keyOf s.params = some key) : p.audit.base.challenger.map (·.1) = some key := by
  have hp : paramsAt s.params = .ok p := by
    unfold caseOf at hcase
    cases hq : paramsAt s.params with
    | error err => simp [hq, bind, Except.bind] at hcase
    | ok p' =>
      cases hc : caseIn s with
      | error err => simp [hq, hc, bind, Except.bind] at hcase
      | ok ce =>
        first
          | (simp only [hq, hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hcase
             rw [hcase.1])
          | (simp [hq, hc, bind, Except.bind, pure, Except.pure] at hcase
             obtain ⟨rfl, -, -⟩ := hcase
             rfl)
  unfold keyOf at hkey
  rw [hp] at hkey
  exact (Option.filter_eq_some_iff.1 hkey).1

/-- **The interior's sample is the interior audit's**: its key and family the interior audit's challenger's, its window
the interior audit's, after the interior commitment's digest. -/
theorem secondDraws_parts {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {d : Properties.Challenger.Draws} (h : secondDraws p e = some d) :
    ∃ c₂, interiorOf e = .ok c₂ ∧ p.audit.base.challenger = some (d.key, d.label) ∧
      p.audit.base.verifier.windowRoot = some d.window ∧ d.after = c₂.digest := by
  cases hi : interiorOf e with
  | error err =>
    unfold secondDraws at h
    rw [hi] at h
    simp [Except.toOption] at h
  | ok c₂ =>
    have h' := secondDraws_eq hi h
    unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource at h'
    rcases hch : p.audit.base.challenger with _ | ⟨key, label⟩ <;>
      rcases hw : p.audit.base.verifier.windowRoot with _ | w <;> simp only [hch, hw] at h' <;> cases h'
    refine ⟨c₂, ?_, ?_, ?_, rfl⟩ <;> first | rfl | assumption

/-- **The check's first session is the cut's**: on a play the form accepts, the evidence's `coins` name the ledger,
pins, main and session `firstCoins` names, and the interior record's digest is the interior commitment's. -/
def SessionMatches (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ p c e c₂, caseOf o.sent = .ok (p, c, e) → interiorOf e = .ok c₂ →
    Spot.Registration.digest e.record = .ok c₂.digest ∧
    ∀ f ∈ ["ledger", "pins", "main", "session"], Spot.getD (firstCoins e) f = Spot.getD e.coins f

/-- **`FirstUnshown` from the form's check**, where it reads the cut's session (`SessionMatches`). -/
theorem firstUnshown_of (C : CaseOf) (hC : CaseIsSent C) (hM : SessionMatches C) : FirstUnshown C := by
  intro o ha p c e key d₁ d₂ cm roots hcase hkey hd₁ hd₂ hw hts
  obtain ⟨c₂, hi₂, hch₂, hwin₂, hafter⟩ := secondDraws_parts hd₂
  obtain ⟨hdig, hf⟩ := hM o ha p c e c₂ hcase hi₂
  obtain ⟨bp, hb, hbw, hkey₁⟩ := firstDraws_window hd₁
  have hbp : bp = p.boundary := by
    unfold boundaryOf at hb
    first
      | exact (Except.ok.inj hb).symm
      | (cases hb; rfl)
  subst hbp
  have hk : d₁.key = d₂.key := by
    rw [hkey₁]
    have h1 := keyOf_challenger hcase hkey
    rw [hch₂] at h1
    exact (Option.some.inj h1).symm
  rw [hf "ledger" (by simp), hf "pins" (by simp), hf "main" (by simp), hf "session" (by simp), hk] at hts
  rw [hafter]
  refine VStar.Recursive.SampledTwo.interiorUnshown_roots (unshown_of_accepts hC ha hcase) hch₂ hbw ?_ hdig hts
  rw [hwin₂]
  exact fun h => hw (Option.some.inj h)

/-- **`SecondAlone` from the form's check**, where it reads the cut's session. -/
theorem secondAlone_of_matches (C : CaseOf) (hC : CaseIsSent C) (hM : SessionMatches C) : SecondAlone C :=
  secondAlone_of C (firstUnshown_of C hC hM)

/-- **The check's first session is the cut's**: `SessionMatches` from the form's check. On an accepted play the form
refuses unless the evidence's interior record has the interior commitment's digest (`interiorDigestIs`,
`INTERIOR_DIGEST`, read by `inputAt_interior`), and that commitment is the cut's (`interiorOf`); the cut's first
session is the evidence's `coins` (`firstCoins`). -/
theorem sessionMatches_of (C : CaseOf) (hC : CaseIsSent C) : SessionMatches C := by
  intro o ha p c e c₂ hcase hi₂
  obtain ⟨k, hk, p', c', e', _, _, hd, hc⟩ := ha
  obtain ⟨-, c₂', _, _, _, _, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hc
  obtain ⟨hcase', -⟩ := hC o k p' c' e' hk hd
  have hpce := Except.ok.inj (hcase'.symm.trans hcase)
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain ⟨⟨s, hs, hcm⟩, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have h2 : c₂' = c₂ := Except.ok.inj ((interiorOf_of hs hcm).symm.trans hi₂)
  subst h2
  exact ⟨(VStar.Recursive.SampledTwo.inputAt_interior hin).1, fun _ _ => rfl⟩

/-- **`SecondAlone` from the form's check.** -/
theorem secondAlone_of_sent (C : CaseOf) (hC : CaseIsSent C) : SecondAlone C :=
  secondAlone_of_matches C hC (sessionMatches_of C hC)

/-! ## The first stage's sample from the form's check -/

/-- The run of record's coin check reads the interior audit's key, the evidence's copy of main and `coins`, and the
boundary registration's family. -/
theorem inputOf_coinsRead (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment) (ξ ξ₂ : ByteArray)
    (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json)) :
    (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).key = p.audit.base.challenger.map (·.1) ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).main = Spot.getD e.coins "main" ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).coins = e.coins ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).family = p.boundary.challenger.map (·.2) ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).record = c.record ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).params = p.boundary ∧
      (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).source = ξ := by
  first
    | exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
    | (unfold VStar.Recursive.SampledTwo.inputOf
       split
       exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)
    | (unfold VStar.Recursive.SampledTwo.inputOf
       cases VStar.Recursive.SampledTwo.rootsSplit c.record e.interior.base.sessionRoots
       exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)

/-- **The first stage read the cut's sample**: `FirstDrawn` from the form's check. On an accepted play the run of record's
coin check gave no code, so either the first stage read no byte, or `drawCoins` read the evidence's `coins` under the
run's key, window and family after the digest of the record the run was given, and the bytes read start them
(`Coins.coinsReason_none`). The run's key, window and family are the cut's sample's (`inputOf`, `keyOf_challenger`), and
the form refuses unless the record the run is given has the commitment's digest (`digestIs`, `RECORD_DIGEST`), which
is the sample's. -/
theorem firstDrawn_of (C : CaseOf) (hC : CaseIsSent C) : FirstDrawn C := by
  intro o ha i read hi hread p c e key d ξ hcase hkey hd hξ
  obtain ⟨k, p', c', e', c₂, ξ₂, r, run, hk, hdec, hin, -, -, -⟩ := inputOn_accepts ha hi
  obtain ⟨hcase', -⟩ := hC o k p' c' e' hk hdec
  have hpce := Except.ok.inj (hcase'.symm.trans hcase)
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨hp, hc, he⟩ := hpce
  cases hp
  cases hc
  cases he
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, rfl, -, -, hdig⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain ⟨hik, him, hic, hif, hir, hip, his⟩ :=
    inputOf_coinsRead p { c with record := j₁ } e.source ξ₂ { e with record := j₂ } received₂ vs
  obtain ⟨_, _, _, _, -, -, -, hcc⟩ := hread
  have hcc' : Spot.Run.coinsCode (p.audit.base.challenger.map (·.1)) (Spot.getD e.coins "main") e.coins j₁
      p.boundary.verifier.windowRoot (p.boundary.challenger.map (·.2)) (e.source.extract 0 read) = none := by
    rw [hik, him, hic, hif, hir, hip, his] at hcc
    exact hcc
  rw [his]
  show Properties.Challenger.bitsOf (e.source.extract 0 read) <+: Properties.Challenger.bitsOf ξ
  have hK := keyOf_challenger hcase hkey
  unfold firstDraws at hd
  cases hb : boundaryOf p with
  | error _ => simp [hb, Except.toOption] at hd
  | ok bp =>
    cases hw : bp.verifier.windowRoot with
    | none => simp [hb, hw, Except.toOption] at hd
    | some w =>
      cases hl : firstLabel p with
      | none => simp [hb, hl, Except.toOption] at hd
      | some label =>
        have h' : some (⟨key, w, label, c.digest⟩ : Properties.Challenger.Draws) = some d := by
          rw [← hd]
          simp [hb, hw, hl, Except.toOption]
        obtain rfl := Option.some.inj h'
        have hbp : bp = p.boundary := by
          unfold boundaryOf at hb
          first | exact (Except.ok.inj hb).symm | (cases hb; rfl)
        subst hbp
        have hl' : p.boundary.challenger.map (·.2) = some label := hl
        have hξ' : drawCoins key w label c.digest e.coins = .ok ξ := hξ
        unfold Spot.Run.coinsCode at hcc'
        by_cases hr : (e.source.extract 0 read).isEmpty = true
        · have hz : (e.source.extract 0 read).size = 0 := by simpa [ByteArray.isEmpty] using hr
          have hn : Properties.Challenger.bitsOf (e.source.extract 0 read) = [] := by
            apply List.eq_nil_of_length_eq_zero
            rw [Verity.Sampled.length_bitsOf, hz]
          rw [hn]
          exact List.nil_prefix
        · rw [if_neg hr, hK, hw, hl', hdig] at hcc'
          cases hcr : coinsReason key (Spot.getD e.coins "main") e.coins w label c.digest
              (e.source.extract 0 read) with
          | some _ => simp [hcr] at hcc'
          | none =>
            obtain ⟨ξ₀, hξ₀, -, hpre, -⟩ := Properties.Challenger.Coins.coinsReason_none hcr
            have hξξ : ξ₀ = ξ := Except.ok.inj (hξ₀.symm.trans hξ')
            rw [← hξξ, ← hpre]
            exact bitsOf_extract_prefix _ _

end Verity.SampledTwo
