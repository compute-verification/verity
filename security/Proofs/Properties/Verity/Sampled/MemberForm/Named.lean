module

public import Proofs.Properties.Verity.Sampled.MemberForm.Run

/-!
# What the per-member form's structure holds of each session's record

The per-member form's structure (`Spot.OneStageZk.membersStructure`) holds each statement's sessions to the trees
the challenger logged (`membersTreesHeld`), each session's record to its own tree less the round that closes the
session (`Spot.Challenger.memberTreeReason`). This module reads that off the verifier's code, below everything that
reads the device's log, so that the proofs about v2's read (`Binding.accepted_run`) hold of an accept of either
form:

* **The structure's parts** (`membersStructure_ok`): the statements' members, the sessions' names, the trees, the
  order, the log's lines and the sessions held to the trees.
* **What `membersTreesHeld` names each session** (`NamedOf`, `membersTreesHeld_ok`): its name, its tables at its
  member's own files and its record, the record parsed, and the SHA-512 of its evidence, each read from the blobs;
  and `Spot.Challenger.membersTreeReason` accepts those, in the statements' sessions' order. At statements of one
  session each, one a statement (`NamedAt`, `named_one`).
* **The records v2's read reads are those** (`records_named`): `recordOn` of each statement's session is the record
  `membersTreesHeld` parsed.
* **The tree less its last opening** (`find_pop`, `mem_pop`, `unclosed_opening`): an opening `t.unclosed` shows is
  one `t` shows, and the first of its stream and round there too.
* **A member's record at its tree** (`memberTreeReason_none`, `membersTreeReason_each`, `tree_round`): what
  `memberTreeReason` accepts, of each session `membersTreeReason` accepts; and each round the record gives coins of
  is one its tree shows opened.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger
open FlockSoundness.Refine (ebind_ok)

/-! ## The structure's parts -/

/-- **The per-member form's structure, part by part**: where `membersStructure` passes, the statements are members'
in order (`membersHeld`), their sessions' names are the evidence's in order, the challenger's trees read (`treesOf`),
the order refusal passed (`orderHeld`), the evidence's log reads as lines, and the statements' sessions hold to the
trees and those lines (`membersTreesHeld`). -/
theorem membersStructure_ok {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : _root_.Core.Blobs}
    (h : _root_.Spot.OneStageZk.membersStructure p ms c e sts bs = .ok ()) :
    ∃ held ts lines, _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      (sts.flatMap (·.sessions)).map (·.name) = e.base.sessions ∧
      _root_.Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧ _root_.Spot.OneStageZk.orderHeld p c e = .ok () ∧
      _root_.Spot.OneStageZk.linesOf e = .ok lines ∧
      _root_.Spot.OneStageZk.membersTreesHeld ts lines (sts.zip held) bs = .ok () := by
  unfold _root_.Spot.OneStageZk.membersStructure at h
  obtain ⟨held, hheld, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, ht, -⟩ := ebind_ok h
    cases ht
  rename_i hne
  obtain ⟨ts, hts, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hord, h⟩ := ebind_ok h
  obtain ⟨lines, hlines, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hth, -⟩ := ebind_ok h
  exact ⟨held, ts, lines, hheld, Classical.byContradiction fun hc => hne (bne_iff_ne.2 hc), hts, hord, hlines, hth⟩

/-! ## What `membersTreesHeld` names each session -/

/-- The statements' sessions, each with its statement's member, in order. -/
def sessionsOf (zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)) :
    List (_root_.Spot.OneStageZk.Member × _root_.Spot.OneStageZk.RunFiles) :=
  zs.flatMap fun z => z.1.sessions.toList.map fun s => (z.2, s)

/-- **What `membersTreesHeld` names the session `z.2` of the member `z.1`**, on the blobs `bs`: the session's name,
its tables with the `root_B` lengths its statement fixes on the member's own files and its record (`rootBTables`),
the record a blob holds as UTF-8, parsed (`null` where it doesn't parse), and the SHA-512, in hex, of the session's
evidence (`sessionEvidence`). -/
def NamedOf (bs : _root_.Core.Blobs) (z : _root_.Spot.OneStageZk.Member × _root_.Spot.OneStageZk.RunFiles)
    (x : String × Array (String × Option ℕ) × Json × String) : Prop :=
  ∃ blob record evidence, bs.get z.2.record s!"session {z.2.name}'s record" = .ok blob ∧
    String.fromUTF8? blob = some record ∧ _root_.Spot.OneStageZk.sessionEvidence bs z.2 = .ok evidence ∧
    x = (z.2.name, _root_.Spot.OneStageZk.rootBTables z.1.own bs record,
      (Flock.JsonParse.parse record).toOption.getD Json.null, sha512Hex evidence)

/-- **What `membersTreesHeld` names the one session of the statement `z.1`**, at its member `z.2`. -/
def NamedAt (bs : _root_.Core.Blobs) (z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)
    (x : String × Array (String × Option ℕ) × Json × String) : Prop :=
  ∃ s, z.1.sessions = #[s] ∧ NamedOf bs (z.2, s) x

theorem forall₂_append {β γ : Type} {R : β → γ → Prop} : ∀ {a₁ a₂ : List β} {b₁ b₂ : List γ},
    List.Forall₂ R a₁ b₁ → List.Forall₂ R a₂ b₂ → List.Forall₂ R (a₁ ++ a₂) (b₁ ++ b₂)
  | _, _, _, _, .nil, h => h
  | _, _, _, _, .cons h t, h' => .cons h (forall₂_append t h')

/-- Each element of the first of two lists related elementwise is related to one of the second. -/
theorem forall₂_mem {β γ : Type} {R : β → γ → Prop} : ∀ {l₁ : List β} {l₂ : List γ}, List.Forall₂ R l₁ l₂ →
    ∀ a ∈ l₁, ∃ b ∈ l₂, R a b
  | _, _, .nil, _, ha => by cases ha
  | _, _, .cons h t, a, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨_, List.mem_cons_self, h⟩
    · obtain ⟨b, hb, hr⟩ := forall₂_mem t a ha
      exact ⟨b, List.mem_cons_of_mem _ hb, hr⟩

/-- **The sessions `membersTreesHeld` holds to the trees**: where it passes on the trees `ts`, the lines `log` and
the statements `sts`, each with its member, `Spot.Challenger.membersTreeReason` accepts, against the trees and the
lines, what it names the statements' sessions in order (`NamedOf`). -/
theorem membersTreesHeld_ok {ts : Array LoggedTree} {log : List String}
    {sts : Array (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)} {bs : _root_.Core.Blobs}
    (h : _root_.Spot.OneStageZk.membersTreesHeld ts log sts bs = .ok ()) :
    ∃ named, membersTreeReason ts log named = none ∧
      List.Forall₂ (NamedOf bs) (sessionsOf sts.toList) named.toList := by
  unfold _root_.Spot.OneStageZk.membersTreesHeld at h
  dsimp only at h
  obtain ⟨named, hn, hrest⟩ := ebind_ok h
  refine ⟨named, ?_, ?_⟩
  · split at hrest
    case h_1 => cases hrest
    rename_i hne
    cases hw : membersTreeReason ts log named with
    | none => rfl
    | some w => exact (hne w hw).elim
  · rw [← Array.forIn_toList] at hn
    have hI := Trees.forIn_inv
      (I := fun pre named => List.Forall₂ (NamedOf bs) (sessionsOf pre) named.toList) ?_
      sts.toList [] #[] named .nil hn
    · simpa using hI
    · intro pre z named' res hI' hres
      obtain ⟨named2, hin, hres⟩ := ebind_ok hres
      cases hres
      refine ⟨_, rfl, ?_⟩
      rw [← Array.forIn_toList] at hin
      have hJ := Trees.forIn_inv (I := fun pre' named =>
          List.Forall₂ (NamedOf bs) (sessionsOf pre ++ pre'.map fun s => (z.2, s)) named.toList) ?_
        z.1.sessions.toList [] named' named2 (by simpa using hI') hin
      · have e : sessionsOf (pre ++ [z]) = sessionsOf pre ++ z.1.sessions.toList.map fun s => (z.2, s) := by
          simp [sessionsOf]
        rw [e]
        simpa using hJ
      · intro pre' s nm res' hI'' hres'
        obtain ⟨blob, hb, hres'⟩ := ebind_ok hres'
        split at hres'
        case h_2 => cases hres'
        rename_i record hrec
        obtain ⟨evidence, hev, hres'⟩ := ebind_ok hres'
        cases hres'
        refine ⟨_, rfl, ?_⟩
        rw [List.map_append, ← List.append_assoc, Array.toList_push]
        exact forall₂_append hI'' (.cons ⟨blob, record, evidence, hb, hrec, hev, rfl⟩ .nil)

/-- **At statements of one session each, `membersTreesHeld` names one session a statement.** -/
theorem named_one {bs : _root_.Core.Blobs} :
    ∀ {zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)}
      {l : List (String × Array (String × Option ℕ) × Json × String)},
      (∀ z ∈ zs, z.1.sessions.size = 1) → List.Forall₂ (NamedOf bs) (sessionsOf zs) l →
        List.Forall₂ (NamedAt bs) zs l
  | [], _, _, h => by
    cases h
    exact .nil
  | z :: zs, l, hone, h => by
    obtain ⟨s, hs⟩ : ∃ s, z.1.sessions = #[s] := ⟨_, eq_single (hone z List.mem_cons_self)⟩
    have e : sessionsOf (z :: zs) = (z.2, s) :: sessionsOf zs := by
      simp [sessionsOf, hs]
    rw [e] at h
    cases h with
    | cons hx ht => exact .cons ⟨s, hs, hx⟩ (named_one (fun z' hz' => hone z' (List.mem_cons_of_mem _ hz')) ht)

/-! ## The records v2's read reads -/

/-- **The records v2's read reads of the statements' sessions are the ones `membersTreesHeld` parsed**: `recordOn`
of each statement's one session is its record on the blobs, parsed. -/
theorem records_named {bs : _root_.Core.Blobs} :
    ∀ {zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)}
      {l : List (String × Array (String × Option ℕ) × Json × String)}, List.Forall₂ (NamedAt bs) zs l →
        ((zs.map (·.1)).flatMap (·.sessions.toList)).mapM (recordOn bs) = some (l.map (·.2.2.1))
  | _, _, .nil => rfl
  | _, _, .cons ⟨s, hs, blob, record, evidence, hb, hu, _, hx⟩ t => by
    have h1 : recordOn bs s = some ((Flock.JsonParse.parse record).toOption.getD Json.null) := by
      unfold recordOn
      rw [hb]
      simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some, hu]
      try rfl
    rw [List.map_cons, List.flatMap_cons, hs, List.toList_toArray, List.singleton_append, List.mapM_cons, h1,
      records_named t, hx]
    try rfl

/-! ## The tree less its last opening -/

/-- An array is its `pop` and what `pop` drops. -/
theorem pop_append {β : Type} (a : Array β) : ∃ r, a = a.pop ++ r := by
  refine ⟨(a.toList.drop (a.size - 1)).toArray, ?_⟩
  apply Array.ext'
  rw [Array.toList_append, Array.toList_pop, List.dropLast_eq_take, Array.length_toList]
  exact (List.take_append_drop _ _).symm

/-- What `find?` finds less the last, it finds. -/
theorem find_pop {β : Type} {p : β → Bool} {a : Array β} {o : β} (h : a.pop.find? p = some o) :
    a.find? p = some o := by
  obtain ⟨r, hr⟩ := pop_append a
  rw [hr, Array.find?_append, h]
  rfl

theorem mem_pop {β : Type} {a : Array β} {o : β} (h : o ∈ a.pop) : o ∈ a := by
  obtain ⟨r, hr⟩ := pop_append a
  rw [hr]
  exact Array.mem_append_left _ h

/-- **An opening a tree shows less its last opening, the tree shows.** -/
theorem unclosed_opening {t : LoggedTree} {name : String} {k : ℕ} {o : TreeOpening}
    (h : t.unclosed.opening name k = some o) : t.opening name k = some o :=
  find_pop (a := t.opened) h

/-- The one tree of `#[t]`. -/
theorem mem_one {t t' : LoggedTree} (h : t' ∈ #[t]) : t' = t := by
  simpa using h

/-! ## The record at its member's tree -/

variable {t : LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}

/-- **Each round a member's record gives coins of is one its tree shows opened**, with those coins' SHA-512
(`Trees.round_held`, at the one tree the member's check holds the record to). -/
theorem tree_round (h : sessionTreeReason #[t.unclosed] tables record = none) {stream : String} {k : ℕ}
    {cs : ByteArray} (hr : recordRound record stream k = some cs) :
    recordTree record = t.root ∧ ∃ o, t.opening stream k = some o ∧ o.coins = sha512Hex cs := by
  obtain ⟨t', ht', hroot, o, ho, hc⟩ := Trees.round_held h hr
  obtain rfl := mem_one ht'
  exact ⟨hroot, o, unclosed_opening ho, hc⟩

/-! ## What the members' check accepts of a session -/

/-- **What `memberTreeReason` accepts**: the record's tree is a logged tree, the last round the log shows opened of it
closes the session after the evidence, and the tree less that round is the session's as `sessionTreeReason` holds
one. -/
theorem memberTreeReason_none {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json} {evidence : String}
    (h : memberTreeReason trees tables record evidence = none) :
    ∃ t ∈ trees, getD (getD record "coin_tree") "root" = Json.str t.root ∧ ∃ o, t.opened.back? = some o ∧
      o.closes evidence = true ∧ sessionTreeReason #[t.unclosed] tables record = none := by
  unfold memberTreeReason at h
  split at h
  · cases h
  · rename_i t ht
    split at h
    · rename_i o ho
      split at h
      · rename_i hc
        exact ⟨t, Array.mem_of_find?_eq_some ht, Trees.strIs (Array.find?_some ht :), o, ho, hc, h⟩
      · cases h
    · cases h

/-- **Each session `membersTreeReason` accepts, `memberTreeReason` accepts.** -/
theorem membersTreeReason_each {trees : Array LoggedTree} {lines : List String}
    {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (h : membersTreeReason trees lines records = none) :
    ∀ x ∈ records, memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 = none := by
  intro x hx
  unfold membersTreeReason at h
  dsimp only at h
  split at h
  · rename_i y hy
    have hp : (memberTreeReason trees y.2.1 y.2.2.1 y.2.2.2).isSome = true := (Array.find?_some hy :)
    cases hm : memberTreeReason trees y.2.1 y.2.2.1 y.2.2.2 with
    | none =>
      rw [hm] at hp
      simp at hp
    | some w =>
      rw [hm] at h
      simp at h
  · rename_i hnone
    have hno := Array.find?_eq_none.1 hnone x hx
    cases hm : memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 with
    | none => rfl
    | some w =>
      exact absurd (show (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).isSome = true by simp [hm]) hno

end Verity.Sampled.MemberForm
