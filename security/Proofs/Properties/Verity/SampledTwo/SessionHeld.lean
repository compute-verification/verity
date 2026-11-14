module

public import Proofs.Properties.Verity.SampledTwo.SessionCoins
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.LinkFirst
public import Std.Tactic.Do

/-!
# The form's acceptance holds the second stage's sessions to the interior's trees

The sessions' coins take two facts about a play the two-stage form accepts as premises: the interior's log reads
(`InteriorReads`, which `sessionsAgree` takes), and the members' check held of what it read (`MembersAccepted`, which
`located_own` and `located_one` take). Both are proved here from the form's acceptance
(`VStar.Recursive.SampledTwo.accepted_steps`, `inputAt_ok`), through what `Spot.OneStageZk.membersRun` ran.

* **What `membersRun` ran** (`membersRun_held`): the statements' structure (`membersHeld`), the challenger's trees
  (`treesOf`), the interior's release log's lines (`linesOf`) and `membersTreesHeld` on them.
* **What `membersTreesHeld` named** (`Named`, `named_exact`): for the statements' sessions in order, each session's
  record as `Verity.Sampled.recordOn` reads it, the SHA-512 of its evidence (`sessionEvidence`), in hex, and its
  tables, one entry at the circuit statement's table (`rootBTables_one`); and `Spot.Challenger.membersTreeReason`
  accepted them. So the records it named are the sessions' records, in order (`named_records`).
* **The statements and their members** (`HeldSize`, `zip_sessions`): `membersTreesHeld` runs over the statements paired
  with their members, which are all the statements where `membersHeld` returns a member for each. That it does is
  `HeldSize`, a premise here: the cut's file proves it (`held_size`, as `SessionCut.heldSize`), and this file does not
  write it twice.
* **The sample's source** (`source_some`): where the interior's sample read coins, its parameters name a challenger and
  a window.
* **The members' check on the log the check read** (`MembersNamed`, `membersNamed`, `membersAccepted`), with what was
  named; and **the read** (`interiorReads`), the second stage's `Verity.Sampled.logRead_isSome`.
* **Each session's tree closes after its evidence's SHA-512, and the trees open in the sessions' order**
  (`closes_after`): in the log the check read, session `t`'s tree has an opening of the stream named the session's,
  after one opening of that stream, answering the bytes of the SHA-512 of session `t`'s evidence, with no opening of the
  tree after it; an opening of session `t`'s tree comes after openings of the trees of sessions at or before `t` only;
  and before any opening of a session's tree, each earlier session has an opening answering its evidence's SHA-512
  (`Verity.Sampled.MemberForm.members_read`).
* **The piece at the form's acceptance** (`located_own_accepted`, `located_one_accepted`): `LocatedOwn` and one
  release a round with `MembersAccepted` discharged; `SessionAgrees.lean` gives `SessionsAgree` with `InteriorReads`
  discharged (`sessionsAgree_accepted`).

**The earlier signature.** Nothing here reads a signature: the trees and the lines are the ones the check read under
the parameters' challenger key, and a strategy replaying an earlier period's genuine signature shows in the cut's
forger term, not here.
-/

@[expose] public section

set_option mvcgen.warning false

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Std.Do

/-! ## What `membersRun` ran -/

/-- **What an accepted `membersRun` ran before its sessions' verdicts**: the statements' members, the challenger's
trees, the release log's lines, and `membersTreesHeld` on the statements paired with their members. -/
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

/-! ## What `membersTreesHeld` named -/

/-- The session `Spot.OneStageZk.run` sets up has one table, the circuit statement's. -/
theorem sessionTables_circuit : Spot.OneStageZk.sessionTables = #[Flock.Tags.circuit.table] := by
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  simp only [Spot.OneStageZk.sessionTables, htag, Option.map_some, Option.getD_some]
  rfl

/-- **The tables a session's record is held at are one entry, at the circuit statement's table**, whatever the
member's own files, the blobs and the record's text: `Spot.OneStageZk.rootBTables` is the session's tables
(`sessionTables_circuit`), each with a `root_B` length or none. -/
theorem rootBTables_one (p : Spot.OneStageZk.Own) (bs : Core.Blobs) (record : String) :
    ∃ n?, Spot.OneStageZk.rootBTables p bs record = #[(Flock.Tags.circuit.table, n?)] := by
  simp only [Spot.OneStageZk.rootBTables, sessionTables_circuit, Array.mapIdx_singleton]
  exact ⟨_, rfl⟩

/-- **A session as `membersTreesHeld` names it**: the entry's record is the session's on the blobs
(`Verity.Sampled.recordOn`), its digest is the SHA-512 of the session's evidence (`sessionEvidence`), in hex, and its
tables are one entry, at the circuit statement's table (`rootBTables_one`). -/
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
      (acc.push (s.name, Spot.OneStageZk.rootBTables p bs t,
        (Flock.JsonParse.parse t).toOption.getD Json.null, Spot.sha512Hex ev)).toList := by
  rw [← List.append_assoc, Array.toList_push]
  refine List.rel_append hI (List.Forall₂.cons ⟨?_, ⟨ev, hev, rfl⟩, rootBTables_one p bs t⟩ .nil)
  simp [Verity.Sampled.recordOn, hg, hu, Except.toOption]

theorem reason_none {α : Type} {o x : Option α} (hx : o = x) (h : ∀ a, x = some a → False) : o = none := by
  subst hx
  cases o with
  | none => rfl
  | some a => exact (h a rfl).elim

-- `mvcgen` walks `membersTreesHeld`'s two loops, past the default heartbeats
set_option maxHeartbeats 1000000 in
/-- **What `membersTreesHeld` named and held**: an entry for each statement's session, in the statements' order, each
the session's record, its evidence's SHA-512 and its one table (`Named`), which `Spot.Challenger.membersTreeReason`
accepted against the trees and the lines. -/
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

/-- **The records `membersTreesHeld` named are the sessions' records, in order.** -/
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

/-! ## The statements and their members -/

/-- **`membersHeld` returns a member for each statement** (the cut's `held_size`). -/
def HeldSize : Prop :=
  ∀ (ms : Array Spot.OneStageZk.Member) (sts : Array Spot.OneStageZk.StatementFiles)
    (held : Array Spot.OneStageZk.Member), Spot.OneStageZk.membersHeld ms sts = .ok held → held.size = sts.size

/-- The sessions of the statements paired with as many members are the statements' sessions. -/
theorem zip_sessions {sts : Array Spot.OneStageZk.StatementFiles} {held : Array Spot.OneStageZk.Member}
    (h : held.size = sts.size) :
    (sts.zip held).toList.flatMap (·.1.sessions.toList) = sts.toList.flatMap (·.sessions.toList) := by
  have e : (sts.zip held).toList.map (·.1) = sts.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by simp [h])
  rw [← e, List.flatMap_map]

/-! ## The sample's source -/

/-- **Where the sample read coins, its parameters name what they are read under**: a challenger and a window. -/
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

/-! ## The read's steps -/

theorem toOption_ok {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x with
  | error _ => cases h
  | ok b => exact congrArg Except.ok (Option.some.inj h)

/-- **What the read's steps were, where it read** (`interiorRead`). -/
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

/-- **The read, from its steps.** -/
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

/-! ## The members' check on the log the check read -/

/-- **The members' check held of the log the check read, with what it named**: `MembersAccepted`, and each entry
`membersTreeReason` accepted is its session's record, its evidence's SHA-512 and its one table (`Named`), for the
statements' sessions in order. -/
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

/-- **The form's acceptance gives the members' check on the log the check read** (`MembersNamed`): the accepted check
ran `membersRun` on the interior's audit (`accepted_steps`, `inputAt_ok`), which read the trees and the lines the
read reads and held the statements' sessions' records to them (`membersRun_held`, `named_exact`). -/
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

/-- **`MembersAccepted`**, the premise of `located_own` and `located_one`, from `MembersNamed`. -/
theorem membersAccepted {C : CaseOf} (h : MembersNamed C) : MembersAccepted C := by
  intro o hacc l hl
  obtain ⟨k, e, label, after, ξ, lines, trees, records, hk, he, hT, hR, hM, hN⟩ := h o hacc l hl
  exact ⟨label, after, ξ, lines, trees, records, hT, hR, hM, sessionRecords_of hk he (named_records hN)⟩

/-! ## The read -/

/-- **What the accepted form ran on its case** (`VStar.Recursive.SampledTwo.accepted_steps`, `inputAt_ok`): the case
decodes, the interior's registration is the commitment of the evidence's record, the interior's sample read its coins,
and `membersRun` ran on the interior's audit with the statements. -/
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
/-- **One-stage v2's sample and its trees read one log's lines, and the read reads them**, as
`Verity.Sampled.logRead_isSome` shows of v2's own acceptance: the sample's source is the parameters' challenger and
window (`source_some`); the sample's read went through the challenger session's log, a list of lines whose pairs are
in order (`Verity.Sampled.sessionOf_ok`, `releasedRootsOf_pairsOk`); `treesOf` read the same lines as coin trees
(`trustedTreesOf_ok`, `coinTreesOf_fold`); so `Verity.Sampled.readLog` reads them, at any records. -/
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
  obtain ⟨_, hsn⟩ := Verity.Sampled.trustedSessionOf_ok htsn
  obtain ⟨_, hso⟩ := Verity.Sampled.sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := Verity.Sampled.sessionOf_ok hso
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

/-- **The form's acceptance reads the interior's log** (`InteriorReads`), as one-stage v2's acceptance reads its log:
the accepted check read the interior's coins and ran `membersRun` on its audit (`accepted_ran`), whose trees are of
the lines the sample read (`lines_read`), and a blob holds each statement's session's record (`named_exact`). -/
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

/-! ## Each session's tree closes after its evidence's SHA-512, and the trees open in the sessions' order -/

/-- **Each session's tree closes after its evidence's SHA-512, and the trees open in the sessions' order**: on a play
the form accepts, at a numbering `tree` of the sessions' trees as the read numbers trees, in the log the check read

* session `t`'s tree has an opening of its stream named the session's (`Spot.Challenger.TREE_SESSION`), after one
  opening of that stream, that answers the SHA-512 of session `t`'s evidence
  (`Spot.OneStageZk.sessionEvidence`), with no opening of the tree after it;
* an opening of session `t`'s tree comes after openings of the trees of sessions at or before `t` only; and
* before any opening of session `t`'s tree, each earlier session has an opening answering its own evidence's
  SHA-512.

It is `Verity.Sampled.MemberForm.members_read` at what `membersTreesHeld` named. -/
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

/-! ## The piece at the form's acceptance -/

/-- **Each session's coins are its own tree's** (`LocatedOwn`), at the form's acceptance. -/
theorem located_own_accepted (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) : LocatedOwn A C tab :=
  located_own A C tab (membersAccepted (membersNamed C hsz))

/-- **One release a round**, at the form's acceptance: the session-round locator puts at most one release of the log
the check read on each round. -/
theorem located_one_accepted (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) :
    ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ n, (located (locOf A tab) n [] l).length ≤ 1 :=
  located_one A C tab (membersAccepted (membersNamed C hsz))

end Verity.SampledTwo.SessionCoins
