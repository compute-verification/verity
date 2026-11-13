module

public import Proofs.Properties.Verity.SampledTwo.SessionFirst
public import Proofs.Properties.Verity.SampledTwo.SessionMsgs
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.CutPrints
public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Properties.Verity.SampledTwo.Accepts
public import Proofs.Flock.SpotFlock.OwnReads
public import Proofs.Flock.SpotFlock.AddressFacts.RowTable

/-!
# The statements' round

Round `stage + 1` of the form's reading is the session statements' message (`SessionReading.stmts`). The device
releases nothing for it: the verifier answers it with no coin. So the string an accepted play shows at it is one fixed
before any coin, the parameters' bytes, and what the round's message is beyond them is what the stage coins before the
round leave no choice in.

* **The model's check** (`SpotFlock.sample_of_stmtOk`, `sample_canon_of_stmtOk`): where round `stage + 1`'s check
  (`SpotFlock.stmtOk`) passes two statements' messages at one round-0 coin and one second stage's coins, template by
  template the two statements decode, agree in every field but the sample, the registered reads and the public file
  (`pinOk`), and state samples of one canonical text: each is held to the verifier's own sample (`second-sample`,
  `Spot.SecondStage.ownDraw`), which reads only the held copies and the stage coins.
* **The round shows the parameters** (`showsAt_form_stmts`): `Messages.ShowsAt` at round `stage + 1` of the form's
  reading, with the string the play's parameters and the digest the SHA-512 of the parameters `θ` every good play
  has. A play of a frame at `θ` has them (`Verity.Sampled.Binding.play_reaches`), so the plays kept may be those.
* **The round's message, from `StmtsFixed`** (`explainsOr_form_stmts`): `Messages.ExplainsOr` at round `stage + 1`
  with the blobs' pair (`Messages.blobsAlt`), where two accepted plays with one parameters' bytes and one log before
  the round read one statements' message or sent two blobs with one SHA-512 (`StmtsFixed`).
* **`StmtsFixed` at the form's own statements** (`stmtsFixed_of_each`): where the reading's statements are the ones
  the form's check verified each session at (`SessionMsgs.stmtsMsg`) and each case is the one sent (`CaseIsSent`),
  `StmtsFixed` follows from the statements' count (`StmtsCount`) and the same fact template by template
  (`StmtAtFixed`).

**Not established here**: `StmtsCount` and `StmtAtFixed`. Template `t`'s statement on a case
(`SessionMsgs.stmtAtCase`) is its member's, at the circuit and public files the parameters name, the blobs' lookup
tables and the session's record's sample as printed; its member is the one the `t`-th session's statement names
(`SessionMsgs.memberAt`). Between two accepted plays with one parameters' bytes the parameters' members are one list,
and a member's files are one or two blobs with one SHA-512. What is left is the member each statement names, the
tables and the sample: that the two plays' stage coins are one where their logs before the round are one, that the
statements name the same members in the same order where the sample is one, that a sample of one canonical text
prints to the same bytes, and that a statement decodes back to its fields, none of which is shown.
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)

variable {h : Held}

theorem getD_map_true {α : Type} {o : Option α} {f : α → Bool} (hf : (o.map f).getD false = true) :
    ∃ a, o = some a ∧ f a = true := by
  cases o with
  | none => simp at hf
  | some a => exact ⟨a, rfl, by simpa using hf⟩

/-- **Round 2's statement check holds a statement's sample to the verifier's own** (`second-sample`). -/
theorem sample_of_codesOk {own : Spot.SecondStage.Own} {t : ℕ} {S : Spot.SecondStage.Stmt}
    (hc : codesOk h own t S = true) :
    Spot.jeq S.draw (Spot.SecondStage.ownDraw own.sample.size (partAt h own.sample.size t) own.drawn) = true := by
  unfold codesOk at hc
  obtain ⟨-, -, -, -, -, hs, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  exact hs

/-- **Two statements `pinOk` passes at one template agree** in every field but the sample, the registered reads and
the public file. -/
theorem pin_eq {t : ℕ} {w w' : FlockDeploy.Raw} (hp : pinOk h t w = true) (hp' : pinOk h t w' = true) :
    { w with draw := none, registered := none, publicFile := .empty } =
      ({ w' with draw := none, registered := none, publicFile := .empty } : FlockDeploy.Raw) := by
  unfold pinOk at hp hp'
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, decide_eq_true_eq] at hp hp'
    exact hp.trans hp'.symm

/-- **Round `stage + 1`'s check reads template `t`'s statement**: where it passes, the run's statement and round 1's
record read, the second stage samples, the statement decodes and reads as round 2 reads one, round 2's statement check
passes it against what the verifier holds of the run (`codesOk` at `ownOf`), and it is the template's reference
statement but for the run's fields (`pinOk`). -/
theorem reads_of_stmtOk {x m₁ m₂ : Claims.Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec draw w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      codesOk h (ownOf h st rec draw c₀) t S = true ∧ pinOk h t w = true := by
  obtain ⟨s, hp⟩ : ∃ s, parse h x = some s := by
    cases hp : parse h x with
    | none => simp [stmtOk, hp] at hok
    | some s => exact ⟨s, rfl⟩
  obtain ⟨xs, hx⟩ : ∃ xs, stmtsOf m₂ = some xs := by
    cases hx : stmtsOf m₂ with
    | none => simp [stmtOk, hp, hx] at hok
    | some xs => exact ⟨xs, rfl⟩
  simp only [stmtOk, hp, hx, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hok
  have hr := (hok.2 t ht).2
  have hat : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by
    unfold stmtAt
    rw [hx]
  rw [← hat] at hr
  unfold readsOk at hr
  split at hr
  · simp only [Bool.and_eq_true] at hr
    obtain ⟨⟨-, hcodes⟩, hpin⟩ := hr
    obtain ⟨S, hS, hc⟩ := getD_map_true hcodes
    exact ⟨_, _, _, _, S, ‹decode x = some _›, ‹jsonMsg m₁ = some _›, ‹secondDraw h c₀ c₁ = some _›,
      ‹FlockDeploy.decode _ = some _›, hS, hc, hpin⟩
  · simp at hr

/-- **Round `stage + 1`'s check holds template `t`'s statement's sample to the verifier's own**: where it passes, the
second stage samples at the stage coins, and the statement's sample, read as JSON, has the canonical text of the
verifier's own sample for the template's part (`Spot.SecondStage.ownDraw`): the second sample's units among the part's
slots in the replay sample round 0's coin gives. -/
theorem sample_of_stmtOk {x m₁ m₂ : Claims.Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ draw w b d, secondDraw h c₀ c₁ = some draw ∧ FlockDeploy.decode (stmtAt h m₂ t) = some w ∧
      pinOk h t w = true ∧ w.draw = some b ∧ FlockDeploy.jsonOf "draw" b = .ok d ∧
      Spot.jeq d (Spot.SecondStage.ownDraw (h.reg.sample (src c₀)).size
        (partAt h (h.reg.sample (src c₀)).size t) ((Spot.Draw.proved draw).map Int.toNat)) = true := by
  obtain ⟨st, rec, draw, w, S, -, -, hd, hw, hS, hc, hpin⟩ := reads_of_stmtOk hok ht
  obtain ⟨b, hb, hj⟩ := AddressRow.stmtJsonOf_draw hS
  exact ⟨draw, w, b, S.draw, hd, hw, hpin, hb, hj, sample_of_codesOk hc⟩

/-- **Two statements' messages that pass round `stage + 1`'s check at one pair of stage coins state one sample,
template by template, and differ only in the run's fields**: the two statements of template `t` decode, are equal in
every field but the sample, the registered reads and the public file (`pinOk`), and their samples read as JSON values
of one canonical text. The run's statement and round 1's message may differ between the two. -/
theorem sample_canon_of_stmtOk {x x' m₁ m₁' m₂ m₂' : Claims.Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) (hok' : stmtOk h x' m₁' m₂' c₀ c₁ = true) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ w w' b b' d d' s, FlockDeploy.decode (stmtAt h m₂ t) = some w ∧
      FlockDeploy.decode (stmtAt h m₂' t) = some w' ∧
      { w with draw := none, registered := none, publicFile := .empty } =
        ({ w' with draw := none, registered := none, publicFile := .empty } : FlockDeploy.Raw) ∧
      w.draw = some b ∧ w'.draw = some b' ∧ FlockDeploy.jsonOf "draw" b = .ok d ∧
      FlockDeploy.jsonOf "draw" b' = .ok d' ∧ Flock.canon d = .ok s ∧ Flock.canon d' = .ok s := by
  obtain ⟨draw, w, b, d, hd, hw, hpin, hb, hj, hq⟩ := sample_of_stmtOk hok ht
  obtain ⟨draw', w', b', d', hd', hw', hpin', hb', hj', hq'⟩ := sample_of_stmtOk hok' ht
  rw [hd] at hd'
  obtain rfl := Option.some.inj hd'
  obtain ⟨s, hs, hown⟩ := ReadsJson.canon_of_jeq hq
  obtain ⟨s', hs', hown'⟩ := ReadsJson.canon_of_jeq hq'
  obtain rfl : s = s' := Except.ok.inj (hown.symm.trans hown')
  exact ⟨w, w', b, b', d, d', s, hw, hw', pin_eq hpin hpin', hb, hb', hj, hj', hs, hs'⟩

end SpotFlock

namespace Verity.SampledTwo

open Claims

namespace SessionMsgs

theorem mapM_congr_mem {α β : Type} {f g : α → Option β} :
    ∀ l : List α, (∀ a ∈ l, f a = g a) → l.mapM f = l.mapM g
  | [], _ => rfl
  | a :: l, hfg => by
    rw [List.mapM_cons, List.mapM_cons, hfg a (List.mem_cons.2 (Or.inl rfl)),
      mapM_congr_mem l fun b hb => hfg b (List.mem_cons.2 (Or.inr hb))]

/-- **The form's parameters and evidence of a case that decodes.** -/
theorem formOf_of_decoded {k : Case} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} (hd : Decoded k p c e) : formOf k = some (p, e) := by
  unfold formOf
  rw [hd.1, hd.2.2]
  rfl

/-- **The statements' message is a function of the statements' count and each template's statement.** -/
theorem stmtsMsg_congr {k k' : Case} {p p' : VStar.Recursive.SampledTwo.Params}
    {e e' : VStar.Recursive.SampledTwo.Evidence} (hf : formOf k = some (p, e)) (hf' : formOf k' = some (p', e'))
    (hn : e.stmts.size = e'.stmts.size) (hs : ∀ t < e.stmts.size, stmtAtCase k t = stmtAtCase k' t) :
    stmtsMsg k = stmtsMsg k' := by
  unfold stmtsMsg
  rw [hf, hf']
  show ((List.range e.stmts.size).mapM (stmtAtCase k)).bind (fun xs => some (encStmts xs)) =
    ((List.range e'.stmts.size).mapM (stmtAtCase k')).bind (fun xs => some (encStmts xs))
  rw [← hn, mapM_congr_mem (List.range e.stmts.size) fun t ht => hs t (List.mem_range.1 ht)]

/-- **Two plays that sent one parameters' bytes decode to one form's parameters**, where each case is the one sent. -/
theorem params_eq_of_sent {C : CaseOf} (hsent : CaseIsSent C) {o o' : Verity.Exchange} {k k' : Case}
    {p p' : VStar.Recursive.SampledTwo.Params} {c c' : Spot.OneStage.Commitment}
    {e e' : VStar.Recursive.SampledTwo.Evidence} (hk : C o = some k) (hk' : C o' = some k')
    (hd : Decoded k p c e) (hd' : Decoded k' p' c' e') (hp : o.params = o'.params) : p = p' := by
  have h₁ := paramsAt_of_caseOf (hsent o k p c e hk hd).1
  have h₂ := paramsAt_of_caseOf (hsent o' k' p' c' e' hk' hd').1
  have e₁ : o.sent.params = o'.sent.params := hp
  rw [e₁, h₂] at h₁
  exact (Except.ok.inj h₁).symm

end SessionMsgs

/-! ## The round shows the parameters -/

/-- **`Messages.ShowsAt` at the statements' round**: an accepted good play reads a statements' message there
(`StmtsRead`) and shows its parameters' bytes, whose SHA-512 is the digest: the SHA-512 of the parameters `θ` every
good play has. -/
theorem showsAt_form_stmts (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (θ : ByteArray)
    (hst : StmtsRead A C S) (hθ : ∀ o, FormAccepts C o → Good o → o.params = θ)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (hd : ∀ o, FormAccepts C o → Good o → Dg o (A.h.stage + 1) = some [Verity.sha512Msg θ.data.toList])
    (hr : ∀ o, FormAccepts C o → Good o → raw o (A.h.stage + 1) = some [o.params.data.toList]) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good Dg raw (A.h.stage + 1) := by
  intro o hA hG
  have hA₀ := hA
  obtain ⟨k, hk, hacc⟩ := hA₀
  obtain ⟨cm, i, -, hi, -, it, m₁, hit, -⟩ := input_of_accepts hacc
  have hin : inputOn C o = some i := by simp only [inputOn, hk, Option.bind_some, hi]
  obtain ⟨m₂, hm₂, -⟩ := hst o i it hA hin hit
  refine ⟨⟨m₂, (Messages.formMsg_stmt A C S dev o).trans hm₂⟩, [Verity.sha512Msg θ.data.toList],
    [o.params.data.toList], hd o hA hG, hr o hA hG, ?_⟩
  rw [hθ o hA hG]
  rfl

/-! ## The round's message -/

/-- **The statements' message is fixed by the parameters and the log before its round, or two blobs collide**: two
plays the check accepts whose logs are the device's, with one parameters' bytes and one log before round `stage + 1`,
read one statements' message, or sent two blobs, different, with one SHA-512 (`Messages.blobsAlt`). -/
def StmtsFixed (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) : Prop :=
  ∀ o o', FormAccepts C o → Good o → FormAccepts C o' → Good o' → o.params = o'.params →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
      S.stmts o = S.stmts o' ∨ ∃ p, Messages.blobsAlt o o' (A.h.stage + 1) = some p

/-- **`Messages.ExplainsOr` at the statements' round, from `StmtsFixed`**, with the string the play's parameters, the
context the log before the round and the pair the blobs'. -/
theorem explainsOr_form_stmts (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hfix : StmtsFixed A C S Good L) (raw : Verity.Exchange → ℕ → Option (List Msg))
    (hr : ∀ o, FormAccepts C o → Good o → raw o (A.h.stage + 1) = some [o.params.data.toList]) :
    Messages.ExplainsOr (formReading A C S dev) (FormAccepts C) Good raw (Messages.logBefore L) Messages.blobsAlt
      (A.h.stage + 1) := by
  intro o o' hA hG hA' hG' hc hraw
  rw [hr o hA hG, hr o' hA' hG'] at hraw
  have hp : o.params = o'.params :=
    ByteArray.ext (Array.ext' (List.cons.inj (Option.some.inj hraw)).1)
  rw [Messages.formMsg_stmt A C S dev o, Messages.formMsg_stmt A C S dev o']
  exact hfix o o' hA hG hA' hG' hp hc

/-! ## `StmtsFixed` at the form's own statements -/

/-- **An accepted play states one statement a template**: the evidence of its case holds as many statements as the
held copies have templates. -/
def StmtsCount (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k p e, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    e.stmts.size = A.h.ms.length

/-- **Each template's session statement is fixed by the parameters and the log before the statements' round, or two
blobs collide**: at two plays the check accepts whose logs are the device's, with one parameters' bytes and one log
before round `stage + 1`, every template's session statement on the two cases is one, or the plays sent two blobs,
different, with one SHA-512. -/
def StmtAtFixed (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    o.params = o'.params →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
      (∀ t < A.h.ms.length, SessionMsgs.stmtAtCase k t = SessionMsgs.stmtAtCase k' t) ∨
        ∃ p, Messages.blobsAlt o o' (A.h.stage + 1) = some p

/-- **`StmtsFixed` at the statements the form's check verified each session at** (`SessionMsgs.stmtsMsg`), where each
case is the one sent: from the statements' count and the same fact template by template. -/
theorem stmtsFixed_of_each (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hsent : CaseIsSent C)
    (hcount : StmtsCount A C Good) (heach : StmtAtFixed A C Good L) : StmtsFixed A C S Good L := by
  intro o o' hA hG hA' hG' hp hlog
  have hA₀ := hA
  have hA₀' := hA'
  obtain ⟨k, hk, pp, c, e, _, _, hdec, -⟩ := hA₀
  obtain ⟨k', hk', pp', c', e', _, _, hdec', -⟩ := hA₀'
  obtain rfl : pp = pp' := SessionMsgs.params_eq_of_sent hsent hk hk' hdec hdec' hp
  have hf := SessionMsgs.formOf_of_decoded hdec
  have hf' := SessionMsgs.formOf_of_decoded hdec'
  rcases heach o o' k k' hA hG hA' hG' hk hk' hp hlog with hall | hpair
  · refine Or.inl ?_
    have hn := hcount o k pp e hA hG hk hf
    have hn' := hcount o' k' pp e' hA' hG' hk' hf'
    rw [hS o, hS o', hk, hk']
    exact SessionMsgs.stmtsMsg_congr hf hf' (hn.trans hn'.symm) fun t ht => hall t (lt_of_lt_of_eq ht hn)
  · exact Or.inr hpair

end Verity.SampledTwo
