module

public import Proofs.Properties.Verity.Sampled.MemberForm.Run
public import Proofs.Properties.Verity.Sampled.MemberForm.Order
public import Proofs.Properties.Verity.Sampled.LinkFirst

/-!
# v2's read at an accept of the per-member form

The per-member form's structure (`Spot.OneStageZk.membersStructure`) holds, before its run, the statements'
sessions to the trees the challenger logged (`membersTreesHeld`) and the release log to its order (`orderHeld`).
This module reads off an accept of the form (`MembersAccept`) what the proofs about the device's log start from:

* **The structure's parts** (`membersStructure_ok`): the statements' members, the sessions' names, the trees, the
  order, the log's lines and the sessions held to the trees.
* **What `membersTreesHeld` names each session** (`NamedOf`, `membersTreesHeld_ok`): its name, its tables at its
  member's own files and its record, the record parsed, and the SHA-512 of its evidence, each read from the blobs;
  and `Spot.Challenger.membersTreeReason` accepts those, in the statements' sessions' order. At statements of one
  session each, one a statement (`NamedAt`, `named_one`).
* **The records v2's read reads are those** (`records_named`): `recordOn` of each statement's session is the record
  `membersTreesHeld` parsed.
* **v2's read at an accept** (`members_read`): where composite `k` accepted what the prover sent, v2's case in it
  accepts at the coins it read, the accept is one of the form and the log v2's check read is the device's, v2 read
  the release log's lines as releases whose reads are the device's log's, at the sessions' records; the lines are
  the trees `coinTreesOf` reads, which `membersTreeReason` accepts the sessions against; and the device's log is in
  order (`SampleLinkFirst`): the sample's coins before every opening, and the first opening of the session's stream.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger
open FlockSoundness.Refine (ebind_ok)
open Properties.Challenger (Draws)

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
        case h_2 =>
          first
            | cases hres'
            | (obtain ⟨_, hthrow, -⟩ := ebind_ok hres'
               cases hthrow)
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

/-! ## v2's read at an accept -/

/-- **v2's read at an accept of the per-member form**: where composite `k` accepted what the prover sent, v2's case
in it accepts at the coins it read (`V2Accepts`), the accept is one of the form (`MembersAccept`), and the log v2's
check read is the device's, v2 read the release log `lines` as the releases `lr` (`readLog`), whose reads are the
device's log's, at the records `membersTreesHeld` parsed; `coinTreesOf` reads `lines` as the trees `trees`, which
`membersTreeReason` accepts those sessions against; the statements are the members `held`'s, one session each as
`membersTreesHeld` names it (`NamedAt`); and the device's log is in order at the sample's family and digest
(`SampleLinkFirst`). -/
theorem members_read {k : Verity.Registration.Composite} {x : Verity.Exchange} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k x.sent)
    (hcase : V2Case k x.sent a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray x.sent.blobs) a b e p c ev ξ dr)
    {ms : Array _root_.Spot.OneStageZk.Member} {sts : Array _root_.Spot.OneStageZk.StatementFiles}
    {r : _root_.Spot.OneStageZk.Run}
    (hM : MembersAccept p c ξ ev (_root_.Core.Blobs.ofArray x.sent.blobs) ms sts r)
    (hl : Binding.Logged Binding.readsV2 x) :
    ∃ (lines : List String) (lr : List Verity.Release) (trees : Array LoggedTree)
      (held : Array _root_.Spot.OneStageZk.Member)
      (named : Array (String × Array (String × Option ℕ) × Json × String)),
      caseOf x.sent = .ok (p, c, ev) ∧ List.Forall₂ Binding.Read.Of (lr.map Binding.Read.ofV2) x.log ∧
      readLog dr.label c.digest (named.toList.map (·.2.2.1)) ξ lines = some lr ∧
      coinTreesOf lines = .ok trees ∧ membersTreeReason trees lines named = none ∧
      _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      List.Forall₂ (NamedAt (_root_.Core.Blobs.ofArray x.sent.blobs)) (sts.zip held).toList named.toList ∧
      SampleLinkFirst dr.label dr.after x.log := by
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, htsn, -⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedSessionOf_ok htsn
  obtain ⟨_, hso⟩ := sessionNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := sessionOf_ok hso
  have hL := linesOf_of hlog hls
  obtain ⟨hstruct, -⟩ := membersRun_ok hM.run hM.all
  obtain ⟨held, ts, lines₁, hheld, -, htr, hord, hL₁, hth⟩ := membersStructure_ok hstruct
  rw [linesOf_check, hL] at hL₁
  obtain rfl : lines = lines₁ := Except.ok.inj hL₁
  obtain ⟨key', w, htt⟩ := treesOf_ok htr
  obtain ⟨ls₂, lines₂, hlog₂, hls₂, hct⟩ := trustedTreesOf_ok htt
  have hL₂ := linesOf_of hlog₂ hls₂
  rw [hL] at hL₂
  obtain rfl : lines = lines₂ := Except.ok.inj hL₂
  obtain ⟨named, hreason, hnamed⟩ := membersTreesHeld_ok hth
  obtain ⟨hsz, hzip⟩ := membersHeld_zip hheld
  have hone := named_one (fun z hz => (hzip z hz).2.2) hnamed
  have hfst : (sts.zip held).toList.map (·.1) = sts.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by rw [Array.length_toList, Array.length_toList]; exact Nat.le_of_eq hsz)
  have hrs : recordsOn (_root_.Core.Blobs.ofArray x.sent.blobs) ev = some (named.toList.map (·.2.2.1)) := by
    unfold recordsOn
    rw [hM.sessions, Array.toList_flatMap, ← hfst]
    exact records_named hone
  have hcs := caseOf_of hS hcase hp hc hev
  have hread : logRead x.sent = readLog dr.label c.digest (named.toList.map (·.2.2.1)) ξ lines := by
    rw [logRead_eq hcs hch hξ hL, hrs, Option.bind_some]
  have hF : (treeFold (named.toList.map (·.2.2.1)) [] lines).isSome := by
    rw [coinTreesOf_fold _ hct]
    rfl
  obtain ⟨lr, hlr⟩ := Option.isSome_iff_exists.1
    (readLog_isSome dr.label c.digest (named.toList.map (·.2.2.1)) ξ hF (releasedRootsOf_pairsOk hrr))
  have hlogged := hl (lr.map Binding.Read.ofV2) (by
    show lr.map Binding.Read.ofV2 ∈ (logRead x.sent).map (·.map Binding.Read.ofV2)
    rw [hread, hlr]
    exact Option.mem_def.2 rfl)
  refine ⟨lines, lr, ts, held, named, hcs, hlogged, hlr, hct, hreason, hheld, hone, ?_⟩
  rw [hafter]
  exact sampleLinkFirst_congr (ofV2_frames hlogged) (sampleLinkFirst_readLog (orderHeld_ok hord hch hL) hlr)

end Verity.Sampled.MemberForm
