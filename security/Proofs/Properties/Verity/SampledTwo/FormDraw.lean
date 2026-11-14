module

public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.FormStmts
public import Proofs.Properties.Verity.SampledTwo.FormPrint
public import Proofs.Properties.Verity.SampledTwo.PortsOfParams
public import Proofs.Flock.Claims.CanonLine
public import Proofs.Flock.Claims.CanonWf

/-!
# The sample each record prints, from the stage coins

`DrawAtFixed` (`FormStmts.lean`) says two accepted plays with one parameters and one log before the statements'
round hold, template by template, session records that print one sample. The model's check of the statements' round
holds each statement's sample to the verifier's own at the stage coins, by its canonical text
(`SpotFlock.sample_canon_of_stmtOk`). Here `DrawAtFixed` is a theorem where the two plays' statements pass that
check at one pair of stage coins (`StageCoinsOne`), which they do where one pair of stage coins carries the bytes
both plays' samples read (`PlayCoinsOne`).

* **The statement holds the record's printed sample** (`stmtAt_record`): on an accepted play, template `t`'s
  statement in the statements' message decodes to its member's statement at the files the check read
  (`stmtAt_raw`), whose sample is the bytes its session's record's sample prints to (`SessionMsgs.drawBytes`), and
  that sample is printable (`SessionCoins.draw_good`).
* **The printed sample reads back with the record's sample's canonical text** (`canon_of_printed`): the parser
  reads a printable value's text back alike it (`Flock.PartCong.printed_readsBack`), and alike values have one
  canonical text.
* **A record's sample is fixed by its canonical text** (`recordDraw_of_canon`): it is `norm` of the record's
  `unit_draw` (`Verity.Sampled.recordDraw_raw`), a well-formed value `norm` fixes, and two such values with one
  canonical text are equivalent (`Properties.Challenger.CanonLine.canon_inj`), so equal
  (`FlockDeploy.eq_of_jsonEquiv_norm`).
* **`DrawAtFixed`** (`drawAtFixed_of_coins`), from the three and `StageCoinsOne`.
* **An accepted play's statements pass the model's check** (`stmtOk_form`) at any round-0 coin and second stage's
  coins that carry the bytes its run's samples read (`SpotFlock.RunsAs.PlayCoins`): the run of record is an accepted
  two-stage run at the held copies, whose round 2 gives the model's statement check
  (`SpotFlock.RunsAs.stmtOk_of_accepted`). Each registration lists each port once (`portsDistinctAt`), with no
  hypothesis: the parameters are read from the tree `Core.canonicalJson` returns, which is well formed.
* **`StageCoinsOne`** (`stageCoinsOne_of_play`), from `stmtOk_form` at each play and `PlayCoinsOne`.

**Not established here**: `PlayCoinsOne`, a hypothesis. The check reads each stage's bytes off the play's log
(`FirstShown`, `SecondShown`); that the bytes the samples of two plays with one log before the statements' round
read start one round-0 coin and one second stage's coins is not shown.
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

/-- **Template `t`'s statement is its member's, at the files the check read**: on a play the form accepts whose log
the check read is the device's, the reading has a statements' message whose `t`-th statement decodes to the statement
(`SessionMsgs.stmtRaw`) of the member the `t`-th of the statements' sessions' statement names
(`SessionMsgs.memberAt`), at the member's circuit and public files on the blobs, the blobs' lookup tables, the bytes
the session's record prints its sample to (`SessionMsgs.drawBytes`) and the member's registered reads; and that
sample is printable. The statement is the member's at the record's sample (`SessionCoins.verified_at`,
`SessionCoins.stmtAtCase_of_member`), which decodes from its bytes (`SessionCoins.stmtRaw_decode`), and the form set
the session up at that sample (`SessionCoins.draw_good`). -/
theorem stmtAt_raw {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S)
    (hcount : StmtsCount A C Good) (hfit : SessionCoins.StmtsFit C Good) {o : Verity.Exchange} {k : Case}
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence} (hacc : FormAccepts C o)
    (hgood : Good o) (hk : C o = some k) (hf : SessionMsgs.formOf k = some (p, e)) {t : ℕ}
    (ht : t < A.h.ms.length) {fs : Spot.OneStageZk.RunFiles} {record : String}
    (hfs : SessionMsgs.filesAt e t = some fs) (hrec : SessionMsgs.recordAt k.blobs fs = some record) :
    ∃ m₂ m circuitB pubB, S.stmts o = some m₂ ∧ SessionMsgs.memberAt p e t = some m ∧
      k.blobs.get m.own.pub "public file" = .ok pubB ∧
      FlockDeploy.decode (SpotFlock.stmtAt A.h m₂ t) =
        some (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
          (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)) ∧
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
  exact ⟨m₂, m, circuitB, pubB, hm₂, hm, hp, hdec, fun dj hdj => SessionCoins.draw_good hdj hsetup⟩

/-- **Template `t`'s statement holds its session's record's printed sample** (`stmtAt_raw`, its sample alone): the
`t`-th statement of the statements' message decodes to a statement whose sample is the bytes the record of the
`t`-th of the statements' sessions prints its sample to (`SessionMsgs.drawBytes`), and that sample is printable. -/
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
  obtain ⟨m₂, m, circuitB, pubB, hm₂, -, -, hdec, hg⟩ :=
    stmtAt_raw hS hst hcount hfit hacc hgood hk hf ht hfs hrec
  exact ⟨m₂, _, hm₂, hdec, rfl, hg⟩

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

/-! ## The statements' check at the play's coins -/

/-- **`PortsDistinctAt` holds**: `Core.canonicalJson` returns printable trees
(`Verity.CanonicalJson.canonicalJson_good`), which are well formed. -/
theorem portsDistinctAt (C : CaseOf) : PortsDistinctAt C :=
  portsDistinctAt_of (fun _ _ h => FormPrint.wf_of_good (Verity.CanonicalJson.canonicalJson_good h)) C

/-- **An accepted play's run of record and its records' messages**: the play has a run of record's input with an
interior, and the boundary's record and the interior's each have canonical bytes. -/
theorem records_of_accepts {C : CaseOf} {o : Verity.Exchange} (hA : FormAccepts C o) :
    ∃ i it x m₁, inputOn C o = some i ∧ i.interior = some it ∧ recordMsg i.record = some x ∧
      recordMsg it.record = some m₁ := by
  obtain ⟨k, hk, hon⟩ := hA
  obtain ⟨cm, i, -, hin, hrb, it, m₁, hit, hm₁⟩ := input_of_accepts hon
  obtain ⟨s, -, -, hs⟩ := VStar.Recursive.SampledTwo.readBack_ok hrb
  refine ⟨i, it, s.toUTF8.data.toList, m₁, ?_, hit, ?_, hm₁⟩
  · show (C o).bind inputOfCase = some i
    rw [hk]
    exact hin
  · simp only [recordMsg, hs]
    rfl

/-- **An accepted play's statements pass the model's statement check**, at any round-0 coin and second stage's coins
that carry the bytes its run's samples read (`SpotFlock.RunsAs.PlayCoins`): the play's run of record is an accepted
two-stage run (`acceptedTwo_of_formAccepts`) at the held copies (`heldAt_of_parts`), its records read back
(`recordsRead`) and name their registrations' names (`recordsAt_of_held`), and the reading's statements are the run's,
each with its template's rounds, in scope and its reference statement but for the run's fields
(`SpotFlock.RunsAs.stmtOk_of_accepted`). -/
theorem stmtOk_form {A : Live₂} {C : CaseOf} {S : SessionReading} (hparts : PartsHeldAt A C) (hof : HeldOfAt A C)
    (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S)
    {o : Verity.Exchange} {i : Spot.Run.Input} {it : Spot.Run.Interior} {x m₁ m₂ : Msg} (hA : FormAccepts C o)
    (hi : inputOn C o = some i) (hit : i.interior = some it) (hx : recordMsg i.record = some x)
    (hm₁ : recordMsg it.record = some m₁) (hm₂ : S.stmts o = some m₂) {c₀ : FlockDeploy.Coin}
    {c₁ : SpotFlock.Stage A.h} (hpc : SpotFlock.RunsAs.PlayCoins A.h i it c₀ c₁) :
    SpotFlock.stmtOk A.h x m₁ m₂ c₀ c₁ = true := by
  have htwo := twoHolds_at hof hA hi hit
  obtain ⟨run, hat⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨hrh, hrn⟩ := recordsAt_of_held htwo (recordsHeldAt_of hof (portsDistinctAt C)) o i it x m₁ hA hi hit hx hm₁
  obtain ⟨hjx, hjm⟩ := recordsRead C o i it x m₁ hA hi hit hx hm₁
  obtain ⟨m₂', hm₂', hsa⟩ := hst o i it hA hi hit
  rw [← Option.some.inj (hm₂.symm.trans hm₂')] at hsa
  exact SpotFlock.RunsAs.stmtOk_of_accepted hat (heldAt_of_parts hof hparts o i it hA hi hit) htwo
    (SpotFlock.RunsAs.Round2.namesDistinct_of_two htwo) hrh hrn hjx hjm (SpotFlock.RunsAs.stmtOk_of_check A.h) hsa
    (hrounds o m₂ hA hm₂) (hscope o m₂ hA hm₂) (hpin o m₂ hA hm₂) hpc

/-- **One pair of stage coins carries what both plays' samples read**: at two plays the check accepts whose logs are
the device's, with one parameters and one log before round `stage + 1`, some round-0 coin and second stage's coins
carry the bytes each play's run's samples read (`SpotFlock.RunsAs.PlayCoins`). -/
def PlayCoinsOne (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e' i i' it it', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k →
    C o' = some k' → Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') → inputOn C o = some i →
    i.interior = some it → inputOn C o' = some i' → i'.interior = some it' →
      ∃ (c₀ : FlockDeploy.Coin) (c₁ : SpotFlock.Stage A.h),
        SpotFlock.RunsAs.PlayCoins A.h i it c₀ c₁ ∧ SpotFlock.RunsAs.PlayCoins A.h i' it' c₀ c₁

/-- **`StageCoinsOne` from the plays' coins** (`PlayCoinsOne`): each play's statements pass the model's check at the
coins that carry what its samples read (`stmtOk_form`), at its own records' messages (`records_of_accepts`). -/
theorem stageCoinsOne_of_play {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    {L : CoinMap.Locate} (hparts : PartsHeldAt A C) (hof : HeldOfAt A C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S)
    (hcoins : PlayCoinsOne A C Good L) : StageCoinsOne A C S Good L := by
  intro o o' k k' p e e' m₂ m₂' hA hG hA' hG' hk hk' hlog hf hf' hm₂ hm₂'
  obtain ⟨i, it, x, m₁, hi, hit, hx, hm₁⟩ := records_of_accepts hA
  obtain ⟨i', it', x', m₁', hi', hit', hx', hm₁'⟩ := records_of_accepts hA'
  obtain ⟨c₀, c₁, hpc, hpc'⟩ :=
    hcoins o o' k k' p e e' i i' it it' hA hG hA' hG' hk hk' hlog hf hf' hi hit hi' hit'
  exact ⟨x, x', m₁, m₁', c₀, c₁, stmtOk_form hparts hof hst hrounds hscope hpin hA hi hit hx hm₁ hm₂ hpc,
    stmtOk_form hparts hof hst hrounds hscope hpin hA' hi' hit' hx' hm₁' hm₂' hpc'⟩

end Verity.SampledTwo
