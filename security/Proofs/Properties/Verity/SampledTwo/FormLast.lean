module

public import Proofs.Properties.Verity.SampledTwo.FormRecords
public import Proofs.Properties.Verity.Sampled.MemberForm.Evidence
public import Proofs.Properties.Verity.Sampled.Record.Accepts
public import Proofs.Flock.VBridge.PartS

/-!
# A session's last message, from the session's evidence

`LastFixed` (`FormRounds.lean`) says two accepted plays with one parameters' bytes, one log before a session's last
round and one evidence of that session read one last message, or sent two blobs with one SHA-512. Here it is a theorem
where each template's session statement is fixed (`StmtAtFixed`, `StmtsRound.lean`).

A session's messages on a case (`SessionMsgs.msgsAtCase`) are those of four parts: its statement, the setup the
statement gives, its record's text and its proof files. Between the two plays:

* **The statement** is one, or two blobs collide: `StmtAtFixed`, at the log before the statements' round, which is
  one where the log before a later round is one (`CoinMap.before_le`).
* **The setup** is the statement's, so it is one.
* **The record and the proof files** are the evidence's parts. A session's evidence is `evidenceBytes` of its
  record's bytes, then each rep file's name and bytes (`sessionEvidence_ok`), which decodes one way
  (`Verity.Sampled.MemberForm.evidenceBytes_inj`): the two sessions have one record's bytes and one list of names and
  files. The proof files are the files of the names the setup's rep streams give, each the first of its name in that
  list (`find_same`), so they are one list (`parts_same`).

That the session has evidence on the case is `SessionCoins.LogCloses`.

**Not established here**: `StmtAtFixed`, `SessionCoins.VerdictsAt` and `SessionCoins.LogCloses`, each a hypothesis,
shown elsewhere from hypotheses of their own. Nothing is said of a session's first message (`LinkFixed`).
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

/-- The log before a round of an empty log is empty and ends at no release. -/
theorem before_nil (L : Locate) (i : ℕ) (pre : List Verity.Release) : before L i pre [] = ([], none) := by
  simp [before]

/-- The log before a round of a log that holds a release is not the empty log that ends at no release. -/
theorem before_cons_ne_nil (L : Locate) {i : ℕ} {pre : List Verity.Release} {rl : Verity.Release}
    (log : List Verity.Release) : before L i pre (rl :: log) ≠ ([], none) := by
  intro h
  by_cases hc : ∃ n, locAt L pre rl.frame rl.answers = some n ∧ i ≤ n
  · rw [before_cons_pos L log hc] at h
    exact absurd (congrArg Prod.snd h) (by simp)
  · rw [before_cons_neg L log hc] at h
    exact absurd (congrArg Prod.fst h) (by simp)

/-- **Two logs with one log before round `i` have one log before every earlier round**: the log before round `j`
ends at the first release put on a round at least `j`, which is at or before the first put on a round at least `i`
where `j ≤ i`, and up to that one the two logs hold the same releases, and it has the same frame and bytes. -/
theorem before_le (L : Locate) {i j : ℕ} (hji : j ≤ i) : ∀ (pre log log' : List Verity.Release),
    before L i pre log = before L i pre log' → before L j pre log = before L j pre log'
  | _, [], [], _ => rfl
  | pre, [], rl' :: log', h => by
    rw [before_nil] at h
    exact absurd h.symm (before_cons_ne_nil L log')
  | pre, rl :: log, [], h => by
    rw [before_nil] at h
    exact absurd h (before_cons_ne_nil L log)
  | pre, rl :: log, rl' :: log', h => by
    by_cases hc : ∃ n, locAt L pre rl.frame rl.answers = some n ∧ i ≤ n
    · by_cases hc' : ∃ n, locAt L pre rl'.frame rl'.answers = some n ∧ i ≤ n
      · rw [before_cons_pos L log hc, before_cons_pos L log' hc'] at h
        have hfa : (rl.frame, rl.answers) = (rl'.frame, rl'.answers) := Option.some.inj (congrArg Prod.snd h)
        obtain ⟨n, hn, hin⟩ := hc
        have hj : ∃ n, locAt L pre rl.frame rl.answers = some n ∧ j ≤ n := ⟨n, hn, Nat.le_trans hji hin⟩
        have hj' : ∃ n, locAt L pre rl'.frame rl'.answers = some n ∧ j ≤ n := by
          rw [← (Prod.mk.inj hfa).1, ← (Prod.mk.inj hfa).2]
          exact hj
        rw [before_cons_pos L log hj, before_cons_pos L log' hj', hfa]
      · rw [before_cons_pos L log hc, before_cons_neg L log' hc'] at h
        exact absurd (congrArg Prod.fst h) (by simp)
    · by_cases hc' : ∃ n, locAt L pre rl'.frame rl'.answers = some n ∧ i ≤ n
      · rw [before_cons_neg L log hc, before_cons_pos L log' hc'] at h
        exact absurd (congrArg Prod.fst h) (by simp)
      · rw [before_cons_neg L log hc, before_cons_neg L log' hc'] at h
        obtain ⟨hrl, hA1⟩ := List.cons.inj (congrArg Prod.fst h)
        have hA2 := congrArg Prod.snd h
        subst hrl
        have ih := before_le L hji (pre ++ [rl]) log log' (Prod.ext hA1 hA2)
        by_cases hj : ∃ n, locAt L pre rl.frame rl.answers = some n ∧ j ≤ n
        · rw [before_cons_pos L log hj, before_cons_pos L log' hj]
        · rw [before_cons_neg L log hj, before_cons_neg L log' hj, ih]

end Verity.SampledTwo.CoinMap

namespace Verity.SampledTwo

open Claims
open FlockVBridge.PartS (mapM_forall₂)
open Verity.Sampled.Record (get_what)

/-! ## A session's evidence, part by part -/

/-- **The parts a session's evidence is of**: its record's bytes on the blobs and, for each rep file it names, in
order, the file's name in UTF-8 and its bytes on the blobs; under `2^64` parts, each under `2^64` bytes. -/
theorem sessionEvidence_ok {bs : Core.Blobs} {s : Spot.OneStageZk.RunFiles} {ev : ByteArray}
    (h : Spot.OneStageZk.sessionEvidence bs s = .ok ev) :
    ∃ (rb : ByteArray) (reps : List (Array ByteArray)), bs.get s.record s!"session {s.name}'s record" = .ok rb ∧
      List.Forall₂ (fun (fd : String × String) r => ∃ b, bs.get fd.2 s!"session {s.name}'s files: {fd.1}" = .ok b ∧
        r = #[fd.1.toUTF8, b]) s.reps.toList reps ∧
      (#[rb] ++ reps.toArray.flatten).size < 2 ^ 64 ∧ (∀ p ∈ #[rb] ++ reps.toArray.flatten, p.size < 2 ^ 64) ∧
      ev = Spot.OneStageZk.evidenceBytes (#[rb] ++ reps.toArray.flatten) := by
  unfold Spot.OneStageZk.sessionEvidence at h
  obtain ⟨rb, hrb, h⟩ := bind_ok h
  obtain ⟨reps, hreps, h⟩ := bind_ok h
  rw [Array.mapM_eq_mapM_toList] at hreps
  obtain ⟨rl, hrl, hreps⟩ := bind_ok hreps
  cases hreps
  try dsimp only at h
  by_cases hc : (decide (Spot.OneStageZk.WORD ≤ (#[rb] ++ rl.toArray.flatten).size) ||
      (#[rb] ++ rl.toArray.flatten).any fun part => decide (Spot.OneStageZk.WORD ≤ part.size)) = true
  · rw [ite_eq_left hc] at h
    cases h
  · rw [ite_eq_right hc] at h
    rw [Bool.or_eq_true, not_or] at hc
    obtain ⟨hc1, hc2⟩ := hc
    refine ⟨rb, rl, hrb, ?_, ?_, ?_, ?_⟩
    · refine List.Forall₂.imp ?_ (mapM_forall₂ hrl)
      intro fd r hr
      obtain ⟨b, hb, hr⟩ := bind_ok hr
      exact ⟨b, hb, (Except.ok.inj hr).symm⟩
    · exact Nat.lt_of_not_le (of_decide_eq_false (Bool.eq_false_iff.2 hc1))
    · intro p hp
      rw [Array.any_eq_true'] at hc2
      exact Nat.lt_of_not_le fun hle => hc2 ⟨p, hp, decide_eq_true hle⟩
    · exact (Except.ok.inj h).symm

/-- **One name's file is one in two sessions whose rep files' names and bytes are the same lists**: the first rep
file of the name is at one place in both, and the blobs' bytes under a digest don't depend on what a refusal would
name. -/
theorem find_same {bs bs' : Core.Blobs} {nm nm' n : String} :
    ∀ {l l' : List (String × String)} {rs rs' : List (Array ByteArray)},
      List.Forall₂ (fun (fd : String × String) r => ∃ b, bs.get fd.2 s!"session {nm}'s files: {fd.1}" = .ok b ∧
        r = #[fd.1.toUTF8, b]) l rs →
      List.Forall₂ (fun (fd : String × String) r => ∃ b, bs'.get fd.2 s!"session {nm'}'s files: {fd.1}" = .ok b ∧
        r = #[fd.1.toUTF8, b]) l' rs' →
      (rs.map Array.toList).flatten = (rs'.map Array.toList).flatten →
      ∀ {fd fd' : String × String} {w w' : String} {p p' : ByteArray},
        l.find? (·.1 == n) = some fd → l'.find? (·.1 == n) = some fd' →
          bs.get fd.2 w = .ok p → bs'.get fd'.2 w' = .ok p' → p = p'
  | _, _, _, _, .nil, _, _, _, _, _, _, _, _, hf, _, _, _ => by cases hf
  | _, _, _, _, .cons _ _, .nil, _, _, _, _, _, _, _, _, hf', _, _ => by cases hf'
  | fd₀ :: l, fd₀' :: l', _, _, .cons h t, .cons h' t', hfl, fd, fd', w, w', p, p', hf, hf', hg, hg' => by
    obtain ⟨b₀, hb₀, rfl⟩ := h
    obtain ⟨b₀', hb₀', rfl⟩ := h'
    simp only [List.map_cons, List.flatten_cons, List.cons_append, List.nil_append, List.cons.injEq] at hfl
    obtain ⟨hname, hb, hrest⟩ := hfl
    have hn : fd₀.1 = fd₀'.1 := String.toByteArray_inj.1 hname
    rw [List.find?_cons] at hf hf'
    by_cases hc : (fd₀.1 == n) = true
    · have hc' : (fd₀'.1 == n) = true := by rw [← hn]; exact hc
      rw [hc] at hf
      rw [hc'] at hf'
      obtain rfl : fd₀ = fd := Option.some.inj hf
      obtain rfl : fd₀' = fd' := Option.some.inj hf'
      have e := Except.ok.inj ((get_what hg (w' := s!"session {nm}'s files: {fd₀.1}")).symm.trans hb₀)
      have e' := Except.ok.inj ((get_what hg' (w' := s!"session {nm'}'s files: {fd₀'.1}")).symm.trans hb₀')
      rw [e, e', hb]
    · have hc' : ¬ (fd₀'.1 == n) = true := by rw [← hn]; exact hc
      rw [Bool.not_eq_true] at hc hc'
      rw [hc] at hf
      rw [hc'] at hf'
      exact find_same t t' hrest hf hf' hg hg'

/-- Two runs over one list give one list of results, where each element's two results are one. -/
theorem mapM_same {α β : Type} {f f' : α → Except String β} :
    ∀ {l : List α} {ys ys' : List β}, (∀ a ∈ l, ∀ b b', f a = .ok b → f' a = .ok b' → b = b') →
      l.mapM f = .ok ys → l.mapM f' = .ok ys' → ys = ys'
  | [], ys, ys', _, h, h' => by
    rw [List.mapM_nil] at h h'
    cases h
    cases h'
    rfl
  | a :: l, ys, ys', hall, h, h' => by
    rw [List.mapM_cons] at h h'
    obtain ⟨b, hb, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    obtain ⟨b', hb', h'⟩ := bind_ok h'
    obtain ⟨bs', hbs', h'⟩ := bind_ok h'
    cases h
    cases h'
    rw [hall a List.mem_cons_self b b' hb hb',
      mapM_same (fun a ha => hall a (List.mem_cons_of_mem _ ha)) hbs hbs']

/-- **One evidence, one record and one list of proof files**: two sessions with the same evidence's bytes, each on
its own blobs, have the same record's text and, at one setup, the same proof files. -/
theorem parts_same {bs bs' : Core.Blobs} {s s' : Spot.OneStageZk.RunFiles} {spec : _root_.Flock.SessionSpec}
    {ev : ByteArray} {record record' : String} {proofs proofs' : Array ByteArray}
    (hev : Spot.OneStageZk.sessionEvidence bs s = .ok ev) (hev' : Spot.OneStageZk.sessionEvidence bs' s' = .ok ev)
    (hr : SessionMsgs.recordAt bs s = some record) (hr' : SessionMsgs.recordAt bs' s' = some record')
    (hp : (Spot.OneStageZk.filesOf bs s (.ok (spec, #[], Lean.Json.null, Lean.Json.null))).toOption = some proofs)
    (hp' : (Spot.OneStageZk.filesOf bs' s' (.ok (spec, #[], Lean.Json.null, Lean.Json.null))).toOption =
      some proofs') : record = record' ∧ proofs = proofs' := by
  obtain ⟨rb, reps, hrb, hreps, hn, hsz, rfl⟩ := sessionEvidence_ok hev
  obtain ⟨rb', reps', hrb', hreps', hn', hsz', he⟩ := sessionEvidence_ok hev'
  have hparts := congrArg Array.toList (Verity.Sampled.MemberForm.evidenceBytes_inj hn hsz hn' hsz' he)
  simp only [Array.toList_append, Array.toList_flatten, List.cons_append, List.nil_append,
    List.cons.injEq] at hparts
  obtain ⟨hrbe, hfl⟩ := hparts
  refine ⟨?_, ?_⟩
  · unfold SessionMsgs.recordAt at hr hr'
    cases hg : (bs.get s.record "record").toOption with
    | none => rw [hg] at hr; cases hr
    | some x =>
      cases hg' : (bs'.get s'.record "record").toOption with
      | none => rw [hg'] at hr'; cases hr'
      | some x' =>
        rw [hg] at hr
        rw [hg'] at hr'
        have e := Except.ok.inj
          ((get_what (SpotFlock.ok_of_toOption hg) (w' := s!"session {s.name}'s record")).symm.trans hrb)
        have e' := Except.ok.inj
          ((get_what (SpotFlock.ok_of_toOption hg') (w' := s!"session {s'.name}'s record")).symm.trans hrb')
        rw [e] at hr
        rw [e', ← hrbe] at hr'
        exact Option.some.inj (hr.symm.trans hr')
  · have h := SpotFlock.ok_of_toOption hp
    have h' := SpotFlock.ok_of_toOption hp'
    unfold Spot.OneStageZk.filesOf at h h'
    obtain ⟨a, ha, h⟩ := bind_ok h
    obtain ⟨a', ha', h'⟩ := bind_ok h'
    cases ha
    cases ha'
    try dsimp only at h h'
    rw [Array.mapM_eq_mapM_toList] at h h'
    obtain ⟨pl, hpl, h⟩ := bind_ok h
    obtain ⟨pl', hpl', h'⟩ := bind_ok h'
    cases h
    cases h'
    refine congrArg List.toArray (mapM_same (fun a _ b b' hb hb' => ?_) hpl hpl')
    obtain ⟨a1, a2, w⟩ := a
    try dsimp only at hb hb'
    split at hb
    · rename_i f d hfind
      split at hb'
      · rename_i f' d' hfind'
        rw [← Array.find?_toList] at hfind hfind'
        exact find_same hreps hreps' hfl hfind hfind' hb hb'
      · cases hb'
    · cases hb

/-- **The evidence of the `t`-th of the statements' sessions is its files' on the case's blobs.** -/
theorem evidence_ok {C : CaseOf} {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} {t : ℕ} {fs : Spot.OneStageZk.RunFiles} {ev : ByteArray}
    (hk : C o = some k) (hf : SessionMsgs.formOf k = some (p, e)) (hfs : SessionMsgs.filesAt e t = some fs)
    (h : SessionCoins.evidenceAt C o t = some ev) : Spot.OneStageZk.sessionEvidence k.blobs fs = .ok ev := by
  have he := SessionMsgs.formOf_evidence hf
  unfold SessionMsgs.filesAt at hfs
  cases hse : Spot.OneStageZk.sessionEvidence k.blobs fs with
  | error w =>
    have hn : SessionCoins.evidenceAt C o t = none := by
      unfold SessionCoins.evidenceAt
      simp only [hk, he, hfs, hse, Option.bind_eq_bind, Option.bind_some, Except.toOption]
    rw [hn] at h
    cases h
  | ok ev₂ =>
    have hs : SessionCoins.evidenceAt C o t = some ev₂ := by
      unfold SessionCoins.evidenceAt
      simp only [hk, he, hfs, hse, Option.bind_eq_bind, Option.bind_some, Except.toOption]
    rw [hs] at h
    rw [Option.some.inj h]

/-! ## The last message -/

/-- **`LastFixed`, from the session's evidence**: where the reading's messages are the case's
(`SessionMsgs.msgAtCase`), each session is the parts the verdict accepted (`SessionCoins.VerdictsAt`), each session
has evidence on the case (`SessionCoins.LogCloses`) and each template's session statement is fixed (`StmtAtFixed`).
Two plays with one log before a session's last round have one log before the statements' round
(`CoinMap.before_le`), so one statement, or two blobs that collide; the statement gives the setup; and one evidence
gives one record and one list of proof files (`parts_same`). The session's messages are those parts'
(`SessionMsgs.msgsAtCase_of_parts`), so every one of them is the same, the last among them. -/
theorem lastFixed_of (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Verity.Exchange → Prop)
    (L : CoinMap.Locate) (tab : ℕ → String)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hv : SessionCoins.VerdictsAt A C Good tab) (hclose : SessionCoins.LogCloses A C Good L)
    (hfix : StmtAtFixed A C Good L) : LastFixed A C S Good L := by
  intro o o' t hA hG hA' hG' ht hp hlog hev
  have hA₀ := hA
  have hA₀' := hA'
  obtain ⟨k, hk, -⟩ := hA₀
  obtain ⟨k', hk', -⟩ := hA₀'
  obtain ⟨x, s, spec, st, record, proofs, ms, hparts, -, -, -, hk₀, hm⟩ := hv o k hA hG hk t ht
  obtain ⟨x', s', spec', st', record', proofs', ms', hparts', -, -, -, -, hm'⟩ := hv o' k' hA' hG' hk' t ht
  have hlog₁ : Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) := by
    have hle : A.h.stage + 1 ≤ SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) := by
      have := SpotFlock.stage_le_off A.h t
      omega
    exact CoinMap.before_le L hle [] o.log o'.log hlog
  rcases hfix o o' k k' hA hG hA' hG' hk hk' hp hlog₁ with hstmt | halt
  · have hparts₀ := hparts
    have hparts₀' := hparts'
    obtain ⟨p, e, fs, hf, hfs, hrec, hx, hsu, hpr⟩ := hparts₀
    obtain ⟨p', e', fs', hf', hfs', hrec', hx', hsu', hpr'⟩ := hparts₀'
    obtain rfl : x = x' := Option.some.inj (hx.symm.trans ((hstmt t ht).trans hx'))
    obtain rfl : spec = spec' := (Prod.mk.inj (Prod.mk.inj (Option.some.inj (hsu.symm.trans hsu'))).2).1
    obtain ⟨ev, hev₁, -⟩ := hclose o hA hG t ht (by omega)
    have hse := evidence_ok hk hf hfs hev₁
    have hse' := evidence_ok hk' hf' hfs' (hev.symm.trans hev₁)
    obtain ⟨rfl, rfl⟩ := parts_same hse hse' hrec hrec' hpr hpr'
    obtain rfl : ms = ms' := Option.some.inj (hm.symm.trans hm')
    refine Or.inl ?_
    rw [hM o t _, hM o' t _, hk, hk']
    show (SessionMsgs.msgsAtCase k t).bind (fun ms => ms[SpotFlock.k₀ A.h t - 1]?) =
      (SessionMsgs.msgsAtCase k' t).bind (fun ms => ms[SpotFlock.k₀ A.h t - 1]?)
    rw [SessionMsgs.msgsAtCase_of_parts hparts hm, SessionMsgs.msgsAtCase_of_parts hparts' hm']
  · exact Or.inr halt

end Verity.SampledTwo
