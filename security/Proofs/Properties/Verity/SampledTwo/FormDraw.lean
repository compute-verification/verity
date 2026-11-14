module

public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.FormStmts
public import Proofs.Properties.Challenger.CanonLine
public import Proofs.Properties.Challenger.CanonWf

/-!
# The sample each record prints, from the stage coins

`DrawAtFixed` (`FormStmts.lean`) says two accepted plays with one parameters and one log before the statements'
round hold, template by template, session records that print one sample. The model's check of the statements' round
holds each statement's sample to the verifier's own at the stage coins, by its canonical text
(`SpotFlock.sample_canon_of_stmtOk`). Here `DrawAtFixed` is a theorem where the two plays' statements pass that
check at one pair of stage coins (`StageCoinsOne`).

* **The statement holds the record's printed sample** (`stmtAt_record`): on an accepted play, template `t`'s
  statement in the statements' message decodes to a statement whose sample is the bytes its session's record's
  sample prints to (`SessionMsgs.drawBytes`), and that sample is printable (`SessionCoins.draw_good`).
* **The printed sample reads back with the record's sample's canonical text** (`canon_of_printed`): the parser
  reads a printable value's text back alike it (`Flock.PartCong.printed_readsBack`), and alike values have one
  canonical text.
* **A record's sample is fixed by its canonical text** (`recordDraw_of_canon`): it is `norm` of the record's
  `unit_draw` (`Verity.Sampled.recordDraw_raw`), a well-formed value `norm` fixes, and two such values with one
  canonical text are equivalent (`Properties.Challenger.CanonLine.canon_inj`), so equal
  (`FlockDeploy.eq_of_jsonEquiv_norm`).
* **`DrawAtFixed`** (`drawAtFixed_of_coins`), from the three and `StageCoinsOne`.

**Not established here**: `StageCoinsOne`, a hypothesis. The check reads each stage's bytes off the play's log
(`FirstShown`, `SecondShown`); that the bytes two plays with one log before the statements' round read start one
round-0 coin and one second stage's coins, and that each play's statements pass the model's check at those, is not
shown.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)

/-- **A record's sample is fixed by its canonical text**: two records' samples with one canonical text are one
value. -/
theorem recordDraw_of_canon {record record' : String} {dj dj' : Json} {s : String}
    (h : Flock.Witness.recordDraw record = some dj) (h' : Flock.Witness.recordDraw record' = some dj')
    (hs : Flock.canon dj = .ok s) (hs' : Flock.canon dj' = .ok s) : dj = dj' := by
  obtain ⟨_, d, -, -, -, rfl⟩ := Verity.Sampled.recordDraw_raw h
  obtain ⟨_, d', -, -, -, rfl⟩ := Verity.Sampled.recordDraw_raw h'
  exact FlockDeploy.eq_of_jsonEquiv_norm (FlockDeploy.norm_norm d) (FlockDeploy.norm_norm d')
    (Properties.Challenger.CanonLine.canon_inj
      (Properties.Challenger.CanonRead.sortedKeys_of_wf _ (FlockDeploy.wf_norm d))
      (Properties.Challenger.CanonRead.sortedKeys_of_wf _ (FlockDeploy.wf_norm d')) hs hs')

/-- **A record's printed sample, read back, has the record's sample's canonical text**: where the record's sample is
printable and a statement's sample `b` is the bytes it prints to, the value Flock's parser reads `b` as has the
canonical text of the record's sample. -/
theorem canon_of_printed {record : String}
    (hg : ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj) {b : ByteArray} {d : Json}
    {s : String} (hb : SessionMsgs.drawBytes record = some b) (hj : FlockDeploy.jsonOf "draw" b = .ok d)
    (hs : Flock.canon d = .ok s) : ∃ dj, Flock.Witness.recordDraw record = some dj ∧ Flock.canon dj = .ok s := by
  unfold SessionMsgs.drawBytes at hb
  cases hrd : Flock.Witness.recordDraw record with
  | none =>
    rw [hrd] at hb
    simp at hb
  | some dj =>
    rw [hrd, Option.map_some] at hb
    obtain ⟨P, hP, hsame⟩ := Flock.PartCong.printed_readsBack "draw" (hg dj hrd)
    rw [Option.some.inj hb, Option.mapM_some, hj] at hP
    have hPd : P = some d := (Except.ok.inj hP).symm
    refine ⟨dj, rfl, ?_⟩
    rcases hsame with he | ⟨a, c, ha, hc, hac⟩
    · rw [← Option.some.inj (hPd.symm.trans he)]
      exact hs
    · rw [Option.some.inj hc, ← FlockComplete.JsonRT.canon_same a c hac, ← Option.some.inj (hPd.symm.trans ha)]
      exact hs

/-- **Template `t`'s statement holds its session's record's printed sample**: on a play the form accepts whose log
the check read is the device's, the reading has a statements' message whose `t`-th statement decodes to a statement
whose sample is the bytes the record of the `t`-th of the statements' sessions prints its sample to
(`SessionMsgs.drawBytes`), and that sample is printable. The statement is the member's at the record's sample
(`SessionCoins.verified_at`, `SessionCoins.stmtAtCase_of_member`), which decodes from its bytes
(`SessionCoins.stmtRaw_decode`), and the form set the session up at that sample (`SessionCoins.draw_good`). -/
theorem stmtAt_record {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S)
    (hcount : StmtsCount A C Good) (hfit : SessionCoins.StmtsFit C Good) {o : Verity.Exchange} {k : Case}
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence} (hacc : FormAccepts C o)
    (hgood : Good o) (hk : C o = some k) (hf : SessionMsgs.formOf k = some (p, e)) {t : ℕ}
    (ht : t < A.h.ms.length) {fs : Spot.OneStageZk.RunFiles} {record : String}
    (hfs : SessionMsgs.filesAt e t = some fs) (hrec : SessionMsgs.recordAt k.blobs fs = some record) :
    ∃ m₂ w, S.stmts o = some m₂ ∧ FlockDeploy.decode (SpotFlock.stmtAt A.h m₂ t) = some w ∧
      w.draw = SessionMsgs.drawBytes record ∧
      ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj := by
  have hacc₀ := hacc
  obtain ⟨k', hk', hon⟩ := hacc₀
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain ⟨_, inp, -, hin, -, it, _, hit, -⟩ := input_of_accepts hon
  have hi : inputOn C o = some inp := by
    show (C o).bind inputOfCase = some inp
    rw [hk]
    exact hin
  obtain ⟨m₂, hm₂, -⟩ := hst o inp it hacc hi hit
  have hmsg : SessionMsgs.stmtsMsg k = some m₂ := by
    have h := hS o
    rw [hk] at h
    exact h.symm.trans hm₂
  obtain ⟨p', e', xs, hf', hxs, -⟩ := SessionMsgs.stmtsOf_stmtsMsg hmsg
  obtain ⟨rfl, rfl⟩ : p = p' ∧ e = e' := Prod.mk.inj (Option.some.inj (hf.symm.trans hf'))
  have hte : t < e.stmts.size := by
    rw [hcount o k p e hacc hgood hk hf]
    exact ht
  obtain ⟨hl, hb⟩ := hfit o k p e xs hacc hgood hk hf hxs
  have hx := SessionMsgs.stmtAt_stmtsMsg A.h hmsg hf hxs hl hb hte
  obtain ⟨m, circuitB, pubB, rd, record', spec, tabs, rest, proofs, -, hm, hc, hp, -, hrec', hsetup, -, -, hfw, hfd⟩ :=
    SessionCoins.verified_at hacc hk hf hfs
  have hrr : record' = record := Option.some.inj (hrec'.symm.trans hrec)
  rw [hrr] at hsetup hfd
  have hxe : SpotFlock.stmtAt A.h m₂ t =
      (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).encode :=
    Option.some.inj (hx.symm.trans (SessionCoins.stmtAtCase_of_member hf hm hfs hrec hc hp))
  have hdec := SessionCoins.stmtRaw_decode hfw hfd
  rw [← hxe] at hdec
  exact ⟨m₂, _, hm₂, hdec, rfl, fun dj hdj => SessionCoins.draw_good hdj hsetup⟩

/-- **The two plays' stage coins are one**: at two plays the check accepts whose logs are the device's, with one
parameters and one log before round `stage + 1`, the statements' messages the reading reads pass the model's check
of the statements' round (`SpotFlock.stmtOk`) at one round-0 coin and one second stage's coins, each at some run's
statement and round 1's message. -/
def StageCoinsOne (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e' m₂ m₂', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') →
    S.stmts o = some m₂ → S.stmts o' = some m₂' →
      ∃ (x x' m₁ m₁' : Msg) (c₀ : FlockDeploy.Coin) (c₁ : SpotFlock.Stage A.h),
        SpotFlock.stmtOk A.h x m₁ m₂ c₀ c₁ = true ∧ SpotFlock.stmtOk A.h x' m₁' m₂' c₀ c₁ = true

/-- **`DrawAtFixed` from the stage coins** (`StageCoinsOne`), where the reading's statements are the case's: each
play's `t`-th statement holds its record's printed sample (`stmtAt_record`), the model's check holds the two
statements' samples to one canonical text (`SpotFlock.sample_canon_of_stmtOk`), which is each record's sample's
(`canon_of_printed`), so the two records hold one sample (`recordDraw_of_canon`) and print the same bytes. -/
theorem drawAtFixed_of_coins {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    {L : CoinMap.Locate} (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S)
    (hcount : StmtsCount A C Good) (hfit : SessionCoins.StmtsFit C Good) (hone : StageCoinsOne A C S Good L) :
    DrawAtFixed A C Good L := by
  intro o o' k k' p e e' hA hG hA' hG' hk hk' hlog hf hf' t ht fs fs' record record' hfs hfs' hrec hrec'
  obtain ⟨m₂, w₀, hm₂, hw₀, hd₀, hg⟩ := stmtAt_record hS hst hcount hfit hA hG hk hf ht hfs hrec
  obtain ⟨m₂', w₀', hm₂', hw₀', hd₀', hg'⟩ := stmtAt_record hS hst hcount hfit hA' hG' hk' hf' ht hfs' hrec'
  obtain ⟨x, x', m₁, m₁', c₀, c₁, hok, hok'⟩ :=
    hone o o' k k' p e e' m₂ m₂' hA hG hA' hG' hk hk' hlog hf hf' hm₂ hm₂'
  obtain ⟨w, w', b, b', d, d', s, hw, hw', -, hb, hb', hj, hj', hs, hs'⟩ :=
    SpotFlock.sample_canon_of_stmtOk hok hok' ht
  have hbr : SessionMsgs.drawBytes record = some b := by
    rw [← hd₀, ← Option.some.inj (hw.symm.trans hw₀)]
    exact hb
  have hbr' : SessionMsgs.drawBytes record' = some b' := by
    rw [← hd₀', ← Option.some.inj (hw'.symm.trans hw₀')]
    exact hb'
  obtain ⟨dj, hdj, hc⟩ := canon_of_printed hg hbr hj hs
  obtain ⟨dj', hdj', hc'⟩ := canon_of_printed hg' hbr' hj' hs'
  unfold SessionMsgs.drawBytes
  rw [hdj, hdj', recordDraw_of_canon hdj hdj' hc hc']

end Verity.SampledTwo
