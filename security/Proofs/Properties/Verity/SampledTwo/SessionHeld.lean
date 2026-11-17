module

public import Proofs.Properties.Verity.SampledTwo.SessionCoins
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.LinkFirst
public import Std.Tactic.Do

/-! The form's acceptance holds the second stage's sessions to the interior's trees (`InteriorReads`,
`MembersAccepted`); `HeldSize` is a premise here. -/

@[expose] public section

set_option mvcgen.warning false

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Std.Do

theorem membersRun_held {p : Spot.OneStageZk.Params} {ms : Array Spot.OneStageZk.Member}
    {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence} {sts : Array Spot.OneStageZk.StatementFiles}
    {bs : Core.Blobs} {r : Spot.OneStageZk.Run} (h : Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) :
    ∃ held ts L, Spot.OneStageZk.membersHeld ms sts = .ok held ∧ Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      Spot.OneStageZk.linesOf e = .ok L ∧ Spot.OneStageZk.membersTreesHeld ts L (sts.zip held) bs = .ok () := by
  unfold Spot.OneStageZk.membersRun at h
  simp only [Verity.Sampled.bind_ok] at h
  obtain ⟨_, h, -⟩ := h
  unfold Spot.OneStageZk.membersStructure at h
  simp only [Verity.Sampled.bind_ok] at h
  obtain ⟨held, hheld, h⟩ := h
  split at h
  · simp only [Verity.Sampled.bind_ok] at h
    obtain ⟨_, hthrow, -⟩ := h
    cases hthrow
  · simp only [Verity.Sampled.bind_ok] at h
    obtain ⟨ts, hts, _, -, L, hL, _, hM, -⟩ := h
    exact ⟨held, ts, L, hheld, hts, hL, hM⟩

theorem sessionTables_circuit : Spot.OneStageZk.sessionTables = #[Flock.Tags.circuit.table] := by
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  simp only [Spot.OneStageZk.sessionTables, htag, Option.map_some, Option.getD_some]
  rfl

theorem rootBTables_one (p : Spot.OneStageZk.Own) (bs : Core.Blobs) (record : String) :
    ∃ n?, Spot.OneStageZk.rootBTables p bs record = #[(Flock.Tags.circuit.table, n?)] := by
  simp only [Spot.OneStageZk.rootBTables, sessionTables_circuit, Array.mapIdx_singleton]
  exact ⟨_, rfl⟩

theorem ownRootBTables_one (p : Spot.OneStageZk.Own) (bs : Core.Blobs) (record : String) :
    ∃ n?, Spot.OneStageZk.ownRootBTables p bs record = #[(Flock.Tags.circuit.table, n?)] := by
  unfold Spot.OneStageZk.ownRootBTables Spot.OneStageZk.rootBTablesAt
  simp only [sessionTables_circuit, Array.mapIdx_singleton]
  exact ⟨_, rfl⟩

def Named (bs : Core.Blobs) (s : Spot.OneStageZk.RunFiles)
    (x : String × Array (String × Option ℕ) × Json × String) : Prop :=
  Verity.Sampled.recordOn bs s = some x.2.2.1 ∧
    (∃ ev, Spot.OneStageZk.sessionEvidence bs s = .ok ev ∧ x.2.2.2 = Spot.sha512Hex ev) ∧
    ∃ n?, x.2.1 = #[(Flock.Tags.circuit.table, n?)]

theorem sessionEvidence_spec (bs : Core.Blobs) (s : Spot.OneStageZk.RunFiles) :
    ⦃⌜True⌝⦄ Spot.OneStageZk.sessionEvidence bs s ⦃⇓? r => ⌜Spot.OneStageZk.sessionEvidence bs s = .ok r⌝⦄ :=
  Verity.Sampled.ok_spec _

theorem named_step {bs : Core.Blobs} {l pre : List Spot.OneStageZk.RunFiles}
    {acc : Array (String × Array (String × Option ℕ) × Json × String)} {s : Spot.OneStageZk.RunFiles}
    {b ev : ByteArray} {t : String} {p : Spot.OneStageZk.Own}
    (hI : List.Forall₂ (Named bs) (l ++ pre) acc.toList)
    (hu : String.fromUTF8? b = some t) (hg : bs.get s.record s!"session {s.name}'s record" = .ok b)
    (hev : Spot.OneStageZk.sessionEvidence bs s = .ok ev) :
    List.Forall₂ (Named bs) (l ++ (pre ++ [s]))
      (acc.push (s.name, Spot.OneStageZk.ownRootBTables p bs t,
        (Flock.JsonParse.parse t).toOption.getD Json.null, Spot.sha512Hex ev)).toList := by
  rw [← List.append_assoc, Array.toList_push]
  refine List.rel_append hI (List.Forall₂.cons ⟨?_, ⟨ev, hev, rfl⟩, ownRootBTables_one p bs t⟩ .nil)
  simp [Verity.Sampled.recordOn, hg, hu, Except.toOption]

theorem reason_none {α : Type} {o x : Option α} (hx : o = x) (h : ∀ a, x = some a → False) : o = none := by
  subst hx
  cases o with
  | none => rfl
  | some a => exact (h a rfl).elim

-- `mvcgen` walks `membersTreesHeld`'s two loops, past the default heartbeats
set_option maxHeartbeats 1000000 in
theorem named_exact {ts : Array Spot.Challenger.LoggedTree} {L : List String}
    {sts : Array (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)} {bs : Core.Blobs}
    (h : Spot.OneStageZk.membersTreesHeld ts L sts bs = .ok ()) :
    ∃ named : Array (String × Array (String × Option ℕ) × Json × String),
      Spot.Challenger.membersTreeReason ts L named = none ∧
        List.Forall₂ (Named bs) (sts.toList.flatMap (·.1.sessions.toList)) named.toList := by
  refine Verity.Sampled.of_triple
    (Q := fun _ => ∃ named : Array (String × Array (String × Option ℕ) × Json × String),
      Spot.Challenger.membersTreeReason ts L named = none ∧
        List.Forall₂ (Named bs) (sts.toList.flatMap (·.1.sessions.toList)) named.toList) ?_ h
  clear h
  mvcgen [Spot.OneStageZk.membersTreesHeld, -Spec.throw_Except, Verity.Sampled.throw_spec,
    Verity.Sampled.blobsGet_spec, sessionEvidence_spec] invariants
  | inv1 => ⇓? ⟨cur, named⟩ => ⌜List.Forall₂ (Named bs) (cur.prefix.flatMap (·.1.sessions.toList)) named.toList⌝
  | inv2 pref _ _ _ _ _ => ⇓? ⟨cur, named⟩ =>
      ⌜List.Forall₂ (Named bs) (pref.flatMap (·.1.sessions.toList) ++ cur.prefix) named.toList⌝
  all_goals first
    | exact named_step ‹_› ‹_› ‹_› ‹_›
    | exact ⟨_, reason_none ‹_› ‹_›, ‹_›⟩
    | exact List.Forall₂.nil
    | simpa [List.flatMap_append] using ‹List.Forall₂ _ _ _›

theorem named_records {bs : Core.Blobs} {ss : List Spot.OneStageZk.RunFiles}
    {named : List (String × Array (String × Option ℕ) × Json × String)} (h : List.Forall₂ (Named bs) ss named) :
    ss.mapM (Verity.Sampled.recordOn bs) = some (named.map (·.2.2.1)) := by
  induction h with
  | nil => rfl
  | cons hx _ ih =>
    rw [List.mapM_cons, hx.1, ih]
    rfl

theorem forall₂_getElem? {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ {i : ℕ} {a : α}, l₁[i]? = some a →
      ∃ b, l₂[i]? = some b ∧ R a b
  | _, _, .nil, _, _, ha => by simp at ha
  | _, _, .cons h _, 0, _, ha => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at ha
    subst ha
    exact ⟨_, rfl, h⟩
  | _, _, .cons _ t, i + 1, _, ha => by
    simp only [List.getElem?_cons_succ] at ha ⊢
    exact forall₂_getElem? t ha

def HeldSize : Prop :=
  ∀ (ms : Array Spot.OneStageZk.Member) (sts : Array Spot.OneStageZk.StatementFiles)
    (held : Array Spot.OneStageZk.Member), Spot.OneStageZk.membersHeld ms sts = .ok held → held.size = sts.size

theorem zip_sessions {sts : Array Spot.OneStageZk.StatementFiles} {held : Array Spot.OneStageZk.Member}
    (h : held.size = sts.size) :
    (sts.zip held).toList.flatMap (·.1.sessions.toList) = sts.toList.flatMap (·.sessions.toList) := by
  have e : (sts.zip held).toList.map (·.1) = sts.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by simp [h])
  rw [← e, List.flatMap_map]

theorem source_some {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence}
    {ξ : ByteArray} (h : Spot.OneStageZk.coinsOf p c e = .ok ξ) :
    ∃ d, Properties.Challenger.oneStageZkSource p c = some d := by
  have hc : ∃ ξ', Spot.OneStage.coinsOf p.base c e.base = .ok (some ξ') := by
    unfold Spot.OneStageZk.coinsOf at h
    cases h1 : Spot.OneStage.coinsOf p.base c e.base with
    | error r => simp [h1, bind, Except.bind] at h
    | ok o =>
      cases o with
      | none => simp [h1, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
      | some ξ' => exact ⟨ξ', rfl⟩
  obtain ⟨ξ', hc⟩ := hc
  unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource
  unfold Spot.OneStage.coinsOf at hc
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at hc ⊢
  · split at hc <;> cases hc
  · split at hc <;> cases hc
  · cases hc
  · exact ⟨_, rfl⟩

theorem toOption_ok {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x with
  | error _ => cases h
  | ok b => exact congrArg Except.ok (Option.some.inj h)

theorem interiorRead_some {C : CaseOf} {o : Exchange} {l : List Release} (h : interiorRead C o = some l) :
    ∃ k p e s c₂ d ξ₂ recs lines, C o = some k ∧
      (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p ∧
      VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧ Flock.canon e.record = .ok s ∧
      Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂ ∧ Properties.Challenger.oneStageZkSource p.audit c₂ = some d ∧
      Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧
      (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs) = some recs ∧
      Verity.Sampled.linesOf e.interior = .ok lines ∧
      Verity.Sampled.readLog d.label d.after recs ξ₂ lines = some l := by
  unfold interiorRead at h
  obtain ⟨k, hk, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨p, hp, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨e, he, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨s, hs, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨c₂, hc₂, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨d, hd, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨ξ₂, hξ, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨recs, hrecs, h⟩ := Verity.Sampled.Binding.obind_some h
  obtain ⟨lines, hlines, h⟩ := Verity.Sampled.Binding.obind_some h
  exact ⟨k, p, e, s, c₂, d, ξ₂, recs, lines, hk, toOption_ok hp, toOption_ok he, toOption_ok hs, toOption_ok hc₂, hd,
    toOption_ok hξ, hrecs, toOption_ok hlines, h⟩

theorem interiorRead_of {C : CaseOf} {o : Exchange} {l : List Release} {k : Case}
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence} {s : String}
    {c₂ : Spot.OneStage.Commitment} {d : Properties.Challenger.Draws} {ξ₂ : ByteArray} {recs : List Json}
    {lines : List String} (hk : C o = some k)
    (hp : (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p)
    (he : VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e) (hs : Flock.canon e.record = .ok s)
    (hc₂ : Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂)
    (hd : Properties.Challenger.oneStageZkSource p.audit c₂ = some d)
    (hξ : Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂)
    (hrecs : (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs) = some recs)
    (hlines : Verity.Sampled.linesOf e.interior = .ok lines)
    (h : Verity.Sampled.readLog d.label d.after recs ξ₂ lines = some l) : interiorRead C o = some l := by
  unfold interiorRead
  simp only [hk, hp, he, hs, hc₂, hd, hξ, hrecs, hlines, Option.bind_eq_bind, Option.bind_some, Except.toOption]
  exact h

theorem sessionRecords_of {C : CaseOf} {o : Exchange} {k : Case} {e : VStar.Recursive.SampledTwo.Evidence}
    {recs : List Json} (hk : C o = some k) (he : VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e)
    (hrecs : (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs) = some recs) :
    sessionRecords C o = some recs := by
  unfold sessionRecords
  simp only [hk, he, hrecs, Option.bind_eq_bind, Option.bind_some, Except.toOption]

def MembersNamed (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l →
    ∃ (k : Case) (e : VStar.Recursive.SampledTwo.Evidence) (label : String) (after : ByteArray) (ξ : ByteArray)
      (lines : List String) (trees : Array Spot.Challenger.LoggedTree)
      (records : Array (String × Array (String × Option ℕ) × Json × String)),
      C o = some k ∧ VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧
      Spot.Challenger.coinTreesOf lines = .ok trees ∧
      Verity.Sampled.readLog label after (records.toList.map (·.2.2.1)) ξ lines = some l ∧
      Spot.Challenger.membersTreeReason trees lines records = none ∧
      List.Forall₂ (Named k.blobs) (e.statements.toList.flatMap (·.sessions.toList)) records.toList

theorem membersNamed (C : CaseOf) (hsz : HeldSize) : MembersNamed C := by
  intro o hacc l hl
  obtain ⟨k, p, e, s, c₂, d, ξ₂, recs, lines, hk, hp, he, hs, hc₂, hd, hξ, hrecs, hlines, hR⟩ := interiorRead_some hl
  obtain ⟨k', hk', p', c', e', pub, prof, ⟨hp', -, he'⟩, hck⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain rfl : p = p' := Except.ok.inj (hp.symm.trans hp')
  obtain rfl : e = e' := Except.ok.inj (he.symm.trans he')
  obtain ⟨-, c₂', ξ₂', r, i, run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨⟨s', hs', hc₂'⟩, hξ', hrun, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain rfl : s = s' := Except.ok.inj (hs.symm.trans hs')
  obtain rfl : c₂ = c₂' := Except.ok.inj (hc₂.symm.trans hc₂')
  obtain rfl : ξ₂ = ξ₂' := Except.ok.inj (hξ.symm.trans hξ')
  obtain ⟨held, ts, L, hheld, hts, hL, hM⟩ := membersRun_held hrun
  rw [Verity.Sampled.linesOf_check, hlines] at hL
  obtain rfl : lines = L := Except.ok.inj hL
  obtain ⟨named, hnone, hN⟩ := named_exact hM
  rw [zip_sessions (hsz _ _ _ hheld)] at hN
  have hrecs' := named_records hN
  rw [hrecs] at hrecs'
  obtain rfl : recs = named.toList.map (·.2.2.1) := Option.some.inj hrecs'
  obtain ⟨key, w, htt⟩ := Verity.Sampled.treesOf_ok hts
  obtain ⟨ls, lines', hlog, hls, hct⟩ := Verity.Sampled.trustedTreesOf_ok htt
  have hl' := Verity.Sampled.linesOf_of hlog hls
  rw [hlines] at hl'
  obtain rfl : lines = lines' := Except.ok.inj hl'
  exact ⟨k, e, d.label, d.after, ξ₂, lines, ts, named, hk, he, hct, hR, hnone, hN⟩

theorem membersAccepted {C : CaseOf} (h : MembersNamed C) : MembersAccepted C := by
  intro o hacc l hl
  obtain ⟨k, e, label, after, ξ, lines, trees, records, hk, he, hT, hR, hM, hN⟩ := h o hacc l hl
  exact ⟨label, after, ξ, lines, trees, records, hT, hR, hM, sessionRecords_of hk he (named_records hN)⟩

theorem accepted_ran {C : CaseOf} {o : Exchange} (hacc : FormAccepts C o) :
    ∃ (k : Case) (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence) (s : String)
      (c₂ : Spot.OneStage.Commitment) (ξ₂ : ByteArray) (r : Spot.OneStageZk.Run),
      C o = some k ∧ (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p ∧
      VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧ Flock.canon e.record = .ok s ∧
      Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂ ∧ Spot.OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧
      Spot.OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements k.blobs = .ok r := by
  obtain ⟨k, hk, p, c, e, pub, prof, ⟨hp, -, he⟩, hck⟩ := hacc
  obtain ⟨-, c₂, ξ₂, r, i, run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨⟨s, hs, hc₂⟩, hξ, hrun, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  exact ⟨k, p, e, s, c₂, ξ₂, r, hk, hp, he, hs, hc₂, hξ, hrun⟩

-- headroom: the read's lemmas below unify long statements, and one proof of this chain together with the form's
-- steps ran past the default heartbeats
set_option maxHeartbeats 1000000 in
theorem lines_read {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence}
    {ξ : ByteArray} {ts : Array Spot.Challenger.LoggedTree} (hξ : Spot.OneStageZk.coinsOf p c e = .ok ξ)
    (hts : Spot.OneStageZk.treesOf p.base e.base = .ok ts) :
    ∃ d lines, Properties.Challenger.oneStageZkSource p c = some d ∧ Verity.Sampled.linesOf e = .ok lines ∧
      Spot.Challenger.coinTreesOf lines = .ok ts ∧
      ∀ records : List Json, (Verity.Sampled.readLog d.label d.after records ξ lines).isSome := by
  obtain ⟨d, hd⟩ := source_some hξ
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c d hd e ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, htsn, -⟩ := Verity.Sampled.drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedChallengePhaseOf_ok htsn
  obtain ⟨_, hso⟩ := Verity.Sampled.challengePhaseNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := Verity.Sampled.challengePhaseOf_ok hso
  have hlines := Verity.Sampled.linesOf_of hlog hls
  obtain ⟨key, w, htt⟩ := Verity.Sampled.treesOf_ok hts
  obtain ⟨ls', lines', hlog', hls', hct⟩ := Verity.Sampled.trustedTreesOf_ok htt
  have hl' := Verity.Sampled.linesOf_of hlog' hls'
  rw [hlines] at hl'
  obtain rfl : lines = lines' := Except.ok.inj hl'
  refine ⟨d, lines, hd, hlines, hct, fun records => ?_⟩
  refine Verity.Sampled.readLog_isSome d.label d.after records ξ ?_ (Verity.Sampled.releasedRootsOf_pairsOk hrr)
  rw [Verity.Sampled.coinTreesOf_fold records hct]
  rfl

theorem interiorReads (C : CaseOf) (hsz : HeldSize) : InteriorReads C := by
  intro o hacc
  obtain ⟨k, p, e, s, c₂, ξ₂, r, hk, hp, he, hs, hc₂, hξ, hrun⟩ := accepted_ran hacc
  obtain ⟨held, ts, L, hheld, hts, -, hM⟩ := membersRun_held hrun
  obtain ⟨named, -, hN⟩ := named_exact hM
  rw [zip_sessions (hsz _ _ _ hheld)] at hN
  obtain ⟨d, lines, hd, hlines, -, hsome⟩ := lines_read hξ hts
  obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 (hsome (named.toList.map (·.2.2.1)))
  exact ⟨l, interiorRead_of (k := k) (p := p) (e := e) (s := s) (c₂ := c₂) (d := d) (ξ₂ := ξ₂) (lines := lines)
    hk hp he hs hc₂ hd hξ (named_records hN) hlines hl⟩

theorem closes_after {C : CaseOf} (h : MembersNamed C) :
    ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l →
      ∃ (k : Case) (e : VStar.Recursive.SampledTwo.Evidence) (tree : ℕ → ℕ), C o = some k ∧
        VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧
        (∀ t s, (e.statements.toList.flatMap (·.sessions.toList))[t]? = some s →
          ∃ ev Lc c L₂ st n, Spot.OneStageZk.sessionEvidence k.blobs s = .ok ev ∧ l = Lc ++ c :: L₂ ∧
            c.frame = .opening (tree t) st n ∧
            ((Verity.Sampled.helloSpecs Lc)[tree t]?.bind fun sp => sp.streams[st]?) =
              some Spot.Challenger.TREE_SESSION ∧
            openedOf Lc (tree t) st = 1 ∧ c.answers = Flock.Sha512.hash ev ∧
            ∀ y ∈ L₂, ∀ s' n', y.frame ≠ .opening (tree t) s' n') ∧
        (∀ pre r post, l = pre ++ r :: post → ∀ t, t < (e.statements.toList.flatMap (·.sessions.toList)).length →
          ∀ st n, r.frame = .opening (tree t) st n → ∀ q ∈ pre, ∀ t',
            t' < (e.statements.toList.flatMap (·.sessions.toList)).length → ∀ s' n',
              q.frame = .opening (tree t') s' n' → t' ≤ t) ∧
        ∀ pre r post, l = pre ++ r :: post → ∀ t s st n,
          (e.statements.toList.flatMap (·.sessions.toList))[t]? = some s → r.frame = .opening (tree t) st n →
            ∀ t' s', t' < t → (e.statements.toList.flatMap (·.sessions.toList))[t']? = some s' →
              ∃ ev q, Spot.OneStageZk.sessionEvidence k.blobs s' = .ok ev ∧ q ∈ pre ∧
                q.answers = Flock.Sha512.hash ev := by
  intro o hacc l hl
  obtain ⟨k, e, label, after, ξ, lines, trees, records, hk, he, hT, hR, hM, hN⟩ := h o hacc l hl
  obtain ⟨tree, h1, h2, h3⟩ := Verity.Sampled.MemberForm.members_read hT hR hM
  have hlen : (e.statements.toList.flatMap (·.sessions.toList)).length = records.size := by
    rw [hN.length_eq, Array.length_toList]
  refine ⟨k, e, tree, hk, he, fun t s hs => ?_,
    fun pre r post hpre t ht st n hr q hq t' ht' s' n' hq' =>
      h2 pre r post hpre t (lt_of_lt_of_eq ht hlen) st n hr q hq t' (lt_of_lt_of_eq ht' hlen) s' n' hq',
    fun pre r post hpre t s st n hs hr t' s' hlt hs' => ?_⟩
  · obtain ⟨x, hx, -, ⟨ev, hev, hxe⟩, -⟩ := forall₂_getElem? hN hs
    rw [Array.getElem?_toList] at hx
    obtain ⟨tr, -, -, -, -, Lc, c, L₂, st, n, hlog, hc, hname, hcount, hun, hlast⟩ := h1 t x hx
    exact ⟨ev, Lc, c, L₂, st, n, hev, hlog, hc, hname, hcount,
      Verity.Sampled.MemberForm.answers_of_hex hun hxe, hlast⟩
  · obtain ⟨x, hx, -⟩ := forall₂_getElem? hN hs
    obtain ⟨x', hx', -, ⟨ev, hev, hxe⟩, -⟩ := forall₂_getElem? hN hs'
    rw [Array.getElem?_toList] at hx hx'
    have ht : t < records.size := (Array.getElem?_eq_some_iff.1 hx).1
    obtain ⟨q, hq, hun⟩ := h3 pre r post hpre t ht st n hr t' x' hlt hx'
    exact ⟨ev, q, hev, hq, Verity.Sampled.MemberForm.answers_of_hex hun hxe⟩

theorem located_own_accepted (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) : LocatedOwn A C tab :=
  located_own A C tab (membersAccepted (membersNamed C hsz))

theorem located_one_accepted (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) :
    ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ n, (located (locOf A tab) n [] l).length ≤ 1 :=
  located_one A C tab (membersAccepted (membersNamed C hsz))

end Verity.SampledTwo.SessionCoins
