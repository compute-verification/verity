module

public import Proofs.Properties.Verity.SampledTwo.SessionFirst
public import Proofs.Properties.Verity.SampledTwo.SessionMsgs
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.CutPrints
public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Properties.Verity.SampledTwo.Accepts
public import Proofs.Flock.SpotFlock.OwnReads
public import Proofs.Flock.SpotFlock.AddressFacts.RowTable

/-! The statements' round, whose message is fixed by the parameters and the log before it; `StmtsCount` and
`StmtAtFixed` are hypotheses here. -/

@[expose] public section

namespace SpotFlock

open Lean (Json)

variable {h : Held}

theorem getD_map_true {α : Type} {o : Option α} {f : α → Bool} (hf : (o.map f).getD false = true) :
    ∃ a, o = some a ∧ f a = true := by
  cases o with
  | none => simp at hf
  | some a => exact ⟨a, rfl, by simpa using hf⟩

theorem sample_of_codesOk {own : Spot.SecondStage.Own} {t : ℕ} {S : Spot.SecondStage.Stmt}
    (hc : codesOk h own t S = true) :
    Spot.jeq S.draw (Spot.SecondStage.ownSample own.sample.size (partAt h own.sample.size t) own.drawn) = true := by
  unfold codesOk at hc
  obtain ⟨-, -, -, -, -, hs, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  exact hs

theorem pinOk_agree {t : ℕ} {w w' : FlockDeploy.Raw} (hp : pinOk h t w = true) (hp' : pinOk h t w' = true) :
    { w with draw := none, registered := none, publicFile := .empty } =
      ({ w' with draw := none, registered := none, publicFile := .empty } : FlockDeploy.Raw) := by
  unfold pinOk at hp hp'
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, decide_eq_true_eq] at hp hp'
    exact hp.trans hp'.symm

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

theorem sample_of_stmtOk {x m₁ m₂ : Claims.Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ draw w b d, secondDraw h c₀ c₁ = some draw ∧ FlockDeploy.decode (stmtAt h m₂ t) = some w ∧
      pinOk h t w = true ∧ w.draw = some b ∧ FlockDeploy.jsonOf "draw" b = .ok d ∧
      Spot.jeq d (Spot.SecondStage.ownSample (h.reg.sample (src c₀)).size
        (partAt h (h.reg.sample (src c₀)).size t) ((Spot.Draw.proved draw).map Int.toNat)) = true := by
  obtain ⟨st, rec, draw, w, S, -, -, hd, hw, hS, hc, hpin⟩ := reads_of_stmtOk hok ht
  obtain ⟨b, hb, hj⟩ := AddressRow.stmtJsonOf_sample hS
  exact ⟨draw, w, b, S.draw, hd, hw, hpin, hb, hj, sample_of_codesOk hc⟩

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
  exact ⟨w, w', b, b', d, d', s, hw, hw', pinOk_agree hpin hpin', hb, hb', hj, hj', hs, hs'⟩

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

theorem formOf_of_decoded {k : Case} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {e : VStar.Recursive.SampledTwo.Evidence} (hd : Decoded k p c e) : formOf k = some (p, e) := by
  unfold formOf
  rw [hd.1, hd.2.2]
  rfl

theorem stmtsMsg_congr {k k' : Case} {p p' : VStar.Recursive.SampledTwo.Params}
    {e e' : VStar.Recursive.SampledTwo.Evidence} (hf : formOf k = some (p, e)) (hf' : formOf k' = some (p', e'))
    (hn : e.stmts.size = e'.stmts.size) (hs : ∀ t < e.stmts.size, stmtAtCase k t = stmtAtCase k' t) :
    stmtsMsg k = stmtsMsg k' := by
  unfold stmtsMsg
  rw [hf, hf']
  show ((List.range e.stmts.size).mapM (stmtAtCase k)).bind (fun xs => some (encStmts xs)) =
    ((List.range e'.stmts.size).mapM (stmtAtCase k')).bind (fun xs => some (encStmts xs))
  rw [← hn, mapM_congr_mem (List.range e.stmts.size) fun t ht => hs t (List.mem_range.1 ht)]

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

def StmtsFixed (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) : Prop :=
  ∀ o o', FormAccepts C o → Good o → FormAccepts C o' → Good o' → o.params = o'.params →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
      S.stmts o = S.stmts o' ∨ ∃ p, Messages.blobsAlt o o' (A.h.stage + 1) = some p

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

def StmtsCount (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k p e, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    e.stmts.size = A.h.ms.length

def StmtAtFixed (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k → C o' = some k' →
    o.params = o'.params →
    Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
      (∀ t < A.h.ms.length, SessionMsgs.stmtAtCase k t = SessionMsgs.stmtAtCase k' t) ∨
        ∃ p, Messages.blobsAlt o o' (A.h.stage + 1) = some p

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
