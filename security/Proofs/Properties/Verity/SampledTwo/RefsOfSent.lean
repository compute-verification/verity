module

public import Proofs.Properties.Verity.SampledTwo.HeldOfAtDesc
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Spot.Tables
public import Proofs.Flock.Bytes

/-!
# The reference statements of the blobs sent, digest by digest

`RefsAt C xs₀` (`HeldOfAtDesc`) asks that every play the form accepts on the case map `C` has blobs giving the
reference statements `xs₀`. A reference statement (`SpotFlock.HeldOf.refRawOf`) reads three things of the blobs: its
member's circuit file and public file, each the blob under a digest the parameters name (`Core.Blobs.get`), and the
pinned lookup tables the blobs hold (`Spot.OneStageZk.tablesOf`), each the first blob under a digest the verifier pins
(`Flock.MUFU_TABLES`). The blobs are the ones the prover sent (`CaseIsSent`), so each read is checked here against
its digest, with a collision as the second way out: a blob of the play and a blob of an array the statement is given
differ under one SHA-512 (`BlobsCollide`). No statement takes a premise on SHA-512, and no collision is under an `∃`
over plays: each conclusion is per accepted play (`RefsAtOr`, `HeldOfAtOr`), its pair a blob of that play and a blob
of the array given.

* **The files** are pinned: two plays' blobs give one file under one digest, or collide (`get_same`).
* **Each table's bytes** are pinned: where two plays' blobs both hold a pinned table, its two files are one or collide.
* **Which pinned tables the blobs hold** is pinned by no digest of its own. Under one list of table names on both
  plays' blobs the tables are the same or collide (`tables_same`), and so are the reference statements
  (`refStmts_same`). The names are pinned where both plays' blobs passed the tables' check at one registration
  (`Spot.OneStageZk.tablesHeld`: the registration's circuit file, under its digest, looks up exactly the pinned tables
  the blobs hold): `names_same`.

So at blobs `F₀` that give `xs₀` and passed the tables' check at one registration, each accepted play's blobs give
`xs₀` or collide with `F₀`, on a case map whose accepted plays all passed the check there (`TablesHeldAt`):
`refsAtOr_of_tables`, and `refsAt_of_held` at `F₀` an accepted play's blobs.

## What the form checks of the tables

The per-member form's run reads every member's statement, whichever members the sample holds
(`Spot.OneStageZk.membersRun_files`), and a member's statement begins with the tables' check at the member's own files
(`memberStatement_tables`). So every member of an accepted play's parameters has its circuit
file on the blobs, and the file looks up exactly the tables the blobs hold (`accepted_tables`), and on a case map
whose accepted plays decode one parameters (`DecodesAt`) every accepted play passed the check at template 0's member
(`tablesHeldAt_of_accept`). `refsAt_of_accept` is `RefsAtOr` from `CaseIsSent` and `DecodesAt` alone, and
`heldOfAt_or_collide` the held copies per play at the cases at one statement (`HeldOfAtOr`, `heldOfAtOr_at`).

## What this does not show

* `DecodesAt` at any case map, or that two members' circuit files look up the same tables.
-/

@[expose] public section

namespace Verity.SampledTwo

open SpotFlock.HeldOf (Unread refStmtsOf refRawOf memberOf memberRaw)
open FlockSoundness.Refine (ebind_ok mufu_nodup)

/-! ## Lists -/

private theorem mapM_some_congr {α β : Type} {f g : α → Option β} :
    ∀ (l : List α) {xs ys : List β}, l.mapM f = some xs → l.mapM g = some ys →
      (∀ a ∈ l, ∀ x y, f a = some x → g a = some y → x = y) → xs = ys
  | [], xs, ys, hf, hg, _ => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at hf hg
    rw [← hf, ← hg]
  | a :: l, xs, ys, hf, hg, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at hf hg
    obtain ⟨x, hx, xs', hxs, rfl⟩ := hf
    obtain ⟨y, hy, ys', hys, rfl⟩ := hg
    rw [h a List.mem_cons_self x y hx hy,
      mapM_some_congr l hxs hys fun b hb => h b (List.mem_cons_of_mem a hb)]

private theorem filterMap_congr_mem {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = g a) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.filterMap_cons, List.filterMap_cons, h a List.mem_cons_self,
      filterMap_congr_mem fun b hb => h b (List.mem_cons_of_mem a hb)]

/-- Two filters of one list whose kept entries carry their element's name, over elements of distinct names, are equal
where they keep the same names and agree on each element both keep. -/
private theorem filterMap_eq_of_names {τ : Type} {name : τ → String} {L : List τ}
    {f g : τ → Option (String × ByteArray)}
    (hinj : ∀ t ∈ L, ∀ t' ∈ L, name t = name t' → t = t')
    (hf : ∀ t x, f t = some x → x.1 = name t) (hg : ∀ t x, g t = some x → x.1 = name t)
    (hn : (L.filterMap f).map (·.1) = (L.filterMap g).map (·.1))
    (hb : ∀ t ∈ L, ∀ x y, f t = some x → g t = some y → x = y) :
    L.filterMap f = L.filterMap g := by
  refine filterMap_congr_mem fun t ht => ?_
  cases hft : f t with
  | none =>
    cases hgt : g t with
    | none => rfl
    | some y =>
      exfalso
      have hm : y.1 ∈ (L.filterMap g).map (·.1) :=
        List.mem_map.2 ⟨y, List.mem_filterMap.2 ⟨t, ht, hgt⟩, rfl⟩
      rw [← hn] at hm
      obtain ⟨x, hx, hxy⟩ := List.mem_map.1 hm
      obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hx
      have htt : t' = t := hinj t' ht' t ht (by rw [← hf t' x hft', hxy, hg t y hgt])
      rw [htt, hft] at hft'
      cases hft'
  | some x =>
    cases hgt : g t with
    | none =>
      exfalso
      have hm : x.1 ∈ (L.filterMap f).map (·.1) :=
        List.mem_map.2 ⟨x, List.mem_filterMap.2 ⟨t, ht, hft⟩, rfl⟩
      rw [hn] at hm
      obtain ⟨y, hy, hyx⟩ := List.mem_map.1 hm
      obtain ⟨t', ht', hgt'⟩ := List.mem_filterMap.1 hy
      have htt : t' = t := hinj t' ht' t ht (by rw [← hg t' y hgt', hyx, hf t x hft])
      rw [htt, hgt] at hgt'
      cases hgt'
    | some y => rw [hb t ht x y hft hgt]

/-! ## A collision among the blobs sent -/

/-- **Two of the blobs differ under one SHA-512**: a blob of `F` and a blob of `F'` that differ and have one
SHA-512. -/
def BlobsCollide (F F' : Array ByteArray) : Prop :=
  ∃ b ∈ F, ∃ b' ∈ F', b ≠ b' ∧ Flock.Sha512.hash b = Flock.Sha512.hash b'

/-! ## A file under a digest -/

/-- **The blob `Blobs.get` returns under a digest is one of the blobs, and its SHA-512 is the digest's bytes.** -/
theorem get_ofArray {F : Array ByteArray} {d w : String} {b : ByteArray}
    (h : (_root_.Core.Blobs.ofArray F).get d w = .ok b) :
    b ∈ F ∧ ∃ a, Flock.unhex d = .ok a ∧ Flock.Sha512.hash b = a := by
  unfold _root_.Core.Blobs.get at h
  obtain ⟨a, hd, h⟩ := ebind_ok h
  try dsimp only at h
  by_cases hc : (a.size != 64 || Flock.hexOf a != d) = true
  · rw [ite_eq_left hc] at h
    cases h
  · rw [ite_eq_right hc] at h
    cases hf : (_root_.Core.Blobs.ofArray F).entries.find? (·.1 == a) with
    | none =>
      rw [hf] at h
      cases h
    | some e =>
      rw [hf] at h
      obtain ⟨e1, e2⟩ := e
      have hb : e2 = b := by
        cases h
        rfl
      have h1 : (e1 == a) = true := by
        have h' := Array.find?_some hf
        exact h'
      have hm : (e1, e2) ∈ F.map fun b => (Flock.Sha512.hash b, b) := Array.mem_of_find?_eq_some hf
      obtain ⟨f, hfF, hfe⟩ := Array.mem_map.1 hm
      obtain ⟨he1, he2⟩ := Prod.mk.inj hfe
      rw [← hb, ← he2]
      exact ⟨hfF, a, hd, by rw [he1]; exact Verity.Sampled.beq_bytes.1 h1⟩

/-- **Two plays' blobs give one file under one digest, or collide.** -/
theorem get_same {F F' : Array ByteArray} {d w w' : String} {b b' : ByteArray}
    (h : (_root_.Core.Blobs.ofArray F).get d w = .ok b) (h' : (_root_.Core.Blobs.ofArray F').get d w' = .ok b') :
    b = b' ∨ BlobsCollide F F' := by
  obtain ⟨hb, a, ha, hh⟩ := get_ofArray h
  obtain ⟨hb', a', ha', hh'⟩ := get_ofArray h'
  have haa : a = a' := Except.ok.inj (ha.symm.trans ha')
  by_cases e : b = b'
  · exact .inl e
  · exact .inr ⟨b, hb, b', hb', e, by rw [hh, hh', haa]⟩

/-! ## The pinned tables -/

theorem tableFile_mem {sha : String} {F : Array ByteArray} {a : ByteArray}
    (h : Verity.Sampled.tableFile sha F = some a) : a ∈ F := by
  unfold Verity.Sampled.tableFile at h
  exact Array.mem_of_find?_eq_some h

/-- **Blobs that hold the pinned tables of one list of names hold the same tables, or collide**: each pinned table
both hold is the first blob under the table's pinned digest. -/
theorem tables_same {F F' : Array ByteArray}
    (hn : (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F)).map (·.1) =
      (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F')).map (·.1)) :
    _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) =
      _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F') ∨ BlobsCollide F F' := by
  by_cases hcol : BlobsCollide F F'
  · exact .inr hcol
  refine .inl ?_
  rw [Verity.Sampled.tablesOf_ofArray F, Verity.Sampled.tablesOf_ofArray F'] at hn ⊢
  refine filterMap_eq_of_names (name := fun t : Flock.MufuTable => t.name) (L := Flock.MUFU_TABLES)
    (fun t ht t' ht' e => List.inj_on_of_nodup_map mufu_nodup ht ht' e)
    (fun t x h => ?_) (fun t x h => ?_) hn (fun t _ x y hx hy => ?_)
  · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hx
    obtain ⟨a', ha', rfl⟩ := Option.map_eq_some_iff.1 hy
    have e : a = a' := by
      by_contra hne
      exact hcol ⟨a, tableFile_mem ha, a', tableFile_mem ha', hne,
        Properties.NetworkRecords.Bytes.hexOf_inj
          ((Verity.Sampled.tableFile_hash ha).trans (Verity.Sampled.tableFile_hash ha').symm)⟩
    rw [e]

/-! ## The reference statements -/

/-- **A template's reference statement, read**: its member, the member's circuit and public files on the blobs, and
the tables the blobs hold. -/
theorem refRawOf_some {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {r : FlockDeploy.Raw}
    (h : refRawOf p bs t = some r) :
    ∃ m circuit pub, memberOf p t = some m ∧ bs.get m.own.circuit "circuit" = .ok circuit ∧
      bs.get m.own.pub "public file" = .ok pub ∧
      r = memberRaw m circuit pub (_root_.Spot.OneStageZk.tablesOf bs) none
        (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8) := by
  simp only [refRawOf, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨m, hm, circuit, hc, pub, hp, rfl⟩ := h
  exact ⟨m, circuit, pub, hm, SpotFlock.HeldOf.ok_of_toOption hc, SpotFlock.HeldOf.ok_of_toOption hp, rfl⟩

/-- **Two plays' blobs that hold the pinned tables of one list of names give the same reference statements at one
parameters, or collide.** -/
theorem refStmts_same {p : VStar.Recursive.SampledTwo.Params} {F F' : Array ByteArray} {xs xs' : List Claims.Msg}
    (h : refStmtsOf p (_root_.Core.Blobs.ofArray F) = some xs)
    (h' : refStmtsOf p (_root_.Core.Blobs.ofArray F') = some xs')
    (hn : (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F)).map (·.1) =
      (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F')).map (·.1)) :
    xs = xs' ∨ BlobsCollide F F' := by
  by_cases hcol : BlobsCollide F F'
  · exact .inr hcol
  refine .inl ?_
  have ht := (tables_same hn).resolve_right hcol
  unfold refStmtsOf at h h'
  refine mapM_some_congr _ h h' fun t _ x y hx hy => ?_
  have hx' : (refRawOf p (_root_.Core.Blobs.ofArray F) t).map FlockDeploy.Raw.encode = some x := hx
  have hy' : (refRawOf p (_root_.Core.Blobs.ofArray F') t).map FlockDeploy.Raw.encode = some y := hy
  obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hx'
  obtain ⟨r', hr', rfl⟩ := Option.map_eq_some_iff.1 hy'
  obtain ⟨m, circuit, pub, hm, hc, hp, rfl⟩ := refRawOf_some hr
  obtain ⟨m', circuit', pub', hm', hc', hp', rfl⟩ := refRawOf_some hr'
  have hmm : m' = m := Option.some.inj (hm'.symm.trans hm)
  subst hmm
  rw [(get_same hc hc').resolve_right hcol, (get_same hp hp').resolve_right hcol, ht]

/-! ## What the form checks of the tables -/

/-- A member's statement begins with the tables' check at the member's own files. -/
private theorem memberStatement_tables {m : _root_.Spot.OneStageZk.Member} {slice : Lean.Json} {bs : Core.Blobs}
    {a : _root_.Spot.OneStageZk.StepAt} (h : m.statement slice bs = .ok a) :
    Properties.Spot.HoldsRegisteredTables m.own bs := by
  unfold _root_.Spot.OneStageZk.Member.statement at h
  obtain ⟨a', ha', -⟩ := ebind_ok h
  unfold _root_.Spot.OneStageZk.statementOf at ha'
  obtain ⟨⟨⟩, ht, -⟩ := ebind_ok ha'
  exact Properties.Spot.SecurityProofs.tablesHeld_ok ht

/-- **Every member of an accepted play's parameters, sampled or not, has its circuit file on the play's blobs, and
the file looks up exactly the pinned tables the blobs hold.** -/
theorem accepted_tables {C : CaseOf} {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence}
    (hA : FormAccepts C o) (hk : C o = some k) (hdec : Decoded k p c e) :
    ∀ m ∈ p.members, Properties.Spot.HoldsRegisteredTables m.own k.blobs := by
  obtain ⟨k', p', e', s, c₂, ξ₂, r, hk', hp', -, -, -, -, hrun⟩ := SessionCoins.accepted_ran hA
  have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
  subst hkk
  obtain ⟨g1, -, -⟩ := hdec
  have hpp : p' = p := Except.ok.inj (hp'.symm.trans g1)
  subst hpp
  obtain ⟨roots, -, hf⟩ := _root_.Spot.OneStageZk.membersRun_files hrun
  intro m hm
  obtain ⟨a, ha, -⟩ := hf m hm
  exact memberStatement_tables ha

theorem memberOf_mem {p : VStar.Recursive.SampledTwo.Params} {t : ℕ} {m : _root_.Spot.OneStageZk.Member}
    (h : memberOf p t = some m) : m ∈ p.members := by
  unfold memberOf at h
  obtain ⟨_, -, hf⟩ := Option.bind_eq_some_iff.1 h
  exact Array.mem_of_find?_eq_some hf

/-! ## `RefsAt`, or a collision -/

/-- **Every accepted play's blobs passed the tables' check at the registration `q`** (`tablesHeld`, as
`Properties.Spot.HoldsRegisteredTables` reads it): `q`'s circuit file is on the blobs and looks up exactly the pinned
tables the blobs hold. -/
def TablesHeldAt (C : CaseOf) (q : _root_.Spot.OneStageZk.Own) : Prop :=
  ∀ o k, FormAccepts C o → C o = some k → Properties.Spot.HoldsRegisteredTables q k.blobs

/-- **Every accepted play passed the tables' check at template 0's member**, on a case map whose accepted plays
decode one parameters (`accepted_tables`). -/
theorem tablesHeldAt_of_accept {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params}
    {m₀ : _root_.Spot.OneStageZk.Member} (hP : DecodesAt C P) (hm : memberOf P 0 = some m₀) :
    TablesHeldAt C m₀.own := by
  intro o k hA hk
  have hA' := hA
  obtain ⟨k', hk', p, c, e, -, -, hdec, -⟩ := hA'
  have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
  subst hkk
  have hpP : p = P := hP o _ p c e hA hk hdec
  subst hpP
  exact accepted_tables hA hk hdec m₀ (memberOf_mem hm)

/-- **Two plays' blobs that passed the tables' check at one registration hold the pinned tables of one list of names,
or collide**: the circuit file under the registration's digest is one file, or two blobs collide. -/
theorem names_same {q : _root_.Spot.OneStageZk.Own} {F F' : Array ByteArray}
    (h : Properties.Spot.HoldsRegisteredTables q (_root_.Core.Blobs.ofArray F))
    (h' : Properties.Spot.HoldsRegisteredTables q (_root_.Core.Blobs.ofArray F')) :
    (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F)).map (·.1) =
      (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F')).map (·.1) ∨ BlobsCollide F F' := by
  obtain ⟨cB, hc, hl⟩ := h
  obtain ⟨cB', hc', hl'⟩ := h'
  rcases get_same hc hc' with e | hcol
  · rw [e] at hl
    exact .inl (Except.ok.inj (hl.symm.trans hl'))
  · exact .inr hcol

theorem caseIsSent_at {C : CaseOf} (x : Claims.Msg) (h : CaseIsSent C) : CaseIsSent (atStmt C x) :=
  fun o k p c e hk hdec => h o k p c e (atStmt_some hk).1 hdec

/-! ## Per play: `RefsAt` or a collision with the blobs given -/

/-- **Each accepted play's blobs give the reference statements `xs₀`, or collide with the blobs `F₀`**: `RefsAt`
per play, the play's way out a collision between its own blobs and `F₀`, which the statement is given. -/
def RefsAtOr (C : CaseOf) (xs₀ : List Claims.Msg) (F₀ : Array ByteArray) : Prop :=
  ∀ o k p c e xs, FormAccepts C o → C o = some k → Decoded k p c e → refStmtsOf p k.blobs = some xs →
    xs = xs₀ ∨ BlobsCollide o.sent.blobs F₀

/-- **Each accepted play's held copies are the ones `heldOf` builds, or its blobs collide with the blobs `F₀`**:
`HeldOfAt` per play, the play's way out a collision between its own blobs and `F₀`, which the statement is given. -/
def HeldOfAtOr (A : Live₂) (C : CaseOf) (F₀ : Array ByteArray) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    (∃ u, SpotFlock.HeldOf.heldOf i it u = some A.h) ∨ BlobsCollide o.sent.blobs F₀

/-- **`RefsAtOr` at blobs `F₀` that give `xs₀` and passed the tables' check at the registration `q`**, on a case map
whose cases are the ones sent (`CaseIsSent`), whose accepted plays decode one parameters (`DecodesAt`) and passed the
tables' check at `q` (`TablesHeldAt`). -/
theorem refsAtOr_of_tables {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {q : _root_.Spot.OneStageZk.Own}
    {F₀ : Array ByteArray} {xs₀ : List Claims.Msg}
    (hS : CaseIsSent C) (hP : DecodesAt C P) (hT : TablesHeldAt C q)
    (hT₀ : Properties.Spot.HoldsRegisteredTables q (_root_.Core.Blobs.ofArray F₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray F₀) = some xs₀) : RefsAtOr C xs₀ F₀ := by
  intro o k p c e xs hA hk hdec hx
  have hpP : p = P := hP o k p c e hA hk hdec
  subst hpP
  have hb := (hS o k _ c e hk hdec).2
  have hh := hT o k hA hk
  rw [hb] at hx hh
  by_cases hne : BlobsCollide o.sent.blobs F₀
  · exact .inr hne
  exact .inl ((refStmts_same hx hx₀ ((names_same hh hT₀).resolve_right hne)).resolve_right hne)

/-- **Each accepted play's reference statements are the play `o₀`'s, or its blobs collide with `o₀`'s**, on a case
map whose cases are the ones sent (`CaseIsSent`), whose accepted plays decode one parameters (`DecodesAt`) and passed
the tables' check at one registration (`TablesHeldAt`), at the accepted play `o₀` the statement is given. -/
theorem refsAt_of_held {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {q : _root_.Spot.OneStageZk.Own}
    {o₀ : Verity.Exchange} {k₀ : Case} {c₀ : _root_.Spot.OneStage.Commitment}
    {e₀ : VStar.Recursive.SampledTwo.Evidence} {xs₀ : List Claims.Msg}
    (hS : CaseIsSent C) (hP : DecodesAt C P) (hT : TablesHeldAt C q)
    (hA₀ : FormAccepts C o₀) (hk₀ : C o₀ = some k₀) (hd₀ : Decoded k₀ P c₀ e₀)
    (hx₀ : refStmtsOf P k₀.blobs = some xs₀) : RefsAtOr C xs₀ o₀.sent.blobs := by
  have hb₀ := (hS o₀ k₀ _ c₀ e₀ hk₀ hd₀).2
  have hh₀ := hT o₀ k₀ hA₀ hk₀
  rw [hb₀] at hx₀ hh₀
  exact refsAtOr_of_tables hS hP hT hh₀ hx₀

/-- **Each accepted play's reference statements are the play `o₀`'s, or its blobs collide with `o₀`'s**, on a case
map whose cases are the ones sent (`CaseIsSent`) and whose accepted plays decode one parameters (`DecodesAt`). Where
the parameters name a template, every accepted play passed the tables' check at template 0's member
(`tablesHeldAt_of_accept`); where they name none, every play's reference statements are none. -/
theorem refsAt_of_accept {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {o₀ : Verity.Exchange} {k₀ : Case}
    {c₀ : _root_.Spot.OneStage.Commitment} {e₀ : VStar.Recursive.SampledTwo.Evidence} {xs₀ : List Claims.Msg}
    (hS : CaseIsSent C) (hP : DecodesAt C P)
    (hA₀ : FormAccepts C o₀) (hk₀ : C o₀ = some k₀) (hd₀ : Decoded k₀ P c₀ e₀)
    (hx₀ : refStmtsOf P k₀.blobs = some xs₀) : RefsAtOr C xs₀ o₀.sent.blobs := by
  by_cases hz : P.circuits.size = 0
  · intro o k p c e xs hA hk hdec hx
    refine .inl ?_
    have hpP : p = P := hP o k p c e hA hk hdec
    subst hpP
    have h1 := SpotFlock.HeldOf.refStmtsOf_length hx
    have h2 := SpotFlock.HeldOf.refStmtsOf_length hx₀
    rw [hz] at h1 h2
    rw [List.eq_nil_of_length_eq_zero h1, List.eq_nil_of_length_eq_zero h2]
  · obtain ⟨r, hr, -⟩ := SpotFlock.HeldOf.refStmtsOf_getElem? hx₀ (Nat.pos_of_ne_zero hz)
    obtain ⟨m₀, -, -, hm, -⟩ := refRawOf_some hr
    exact refsAt_of_held hS hP (tablesHeldAt_of_accept hP hm) hA₀ hk₀ hd₀ hx₀

/-- **`HeldOfAtOr` at the cases at the statement `x`**: `heldOfAt_at` per play, from `RefsAtOr` in place of
`RefsAt`. -/
theorem heldOfAtOr_at {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {xs₀ : List Claims.Msg} {F₀ : Array ByteArray} (u₀ : Unread) (hd : DescAt (atStmt C x))
    (hP : DecodesAt (atStmt C x) P) (hR : RefsAtOr (atStmt C x) xs₀ F₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAtOr A (atStmt C x) F₀ := by
  intro o i it hacc hi hit
  obtain ⟨k, hk, p, c, e, -, -, hdec, -⟩ := id hacc
  obtain ⟨xs, h, hx, hh⟩ := SpotFlock.HeldOf.heldOf_of_descOk u₀ (hd o k p c e i it hacc hk hdec hi hit)
  obtain rfl := hP o k p c e hacc hk hdec
  rcases hR o k p c e xs hacc hk hdec hx with rfl | hcol
  · obtain ⟨j₁, hj₁, hrec, hpar, hrule, hcl, hip, hrep, hcirc⟩ := fields_of_accept hacc hk hdec hi hit
    have hmsg : recordMsg i.record = some x := (inputOn_at hi).2
    rw [hrec] at hmsg
    have hjj : j₁ = j := Option.some.inj ((jsonMsg_of_readBack hj₁ hmsg).symm.trans hj)
    obtain rfl := hA i it h (by rw [hpar]) (hrec.trans hjj) hrule hcl (by rw [hip]) hrep hcirc hh
    exact .inl ⟨_, hh⟩
  · exact .inr hcol

/-- **Each accepted play at one statement has the held copies, or its blobs collide with the play `o₀`'s**:
`heldOfAtOr_at` with `RefsAtOr` from `refsAt_of_accept`, at the reference statements `xs₀` of the accepted play
`o₀` the statement is given. -/
theorem heldOfAt_or_collide {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {o₀ : Verity.Exchange} {k₀ : Case}
    {c₀ : _root_.Spot.OneStage.Commitment} {e₀ : VStar.Recursive.SampledTwo.Evidence} {xs₀ : List Claims.Msg}
    (u₀ : Unread) (hS : CaseIsSent C) (hd : DescAt (atStmt C x)) (hP : DecodesAt (atStmt C x) P)
    (hA₀ : FormAccepts (atStmt C x) o₀) (hk₀ : atStmt C x o₀ = some k₀)
    (hd₀ : Decoded k₀ P c₀ e₀) (hx₀ : refStmtsOf P k₀.blobs = some xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAtOr A (atStmt C x) o₀.sent.blobs :=
  heldOfAtOr_at u₀ hd hP (refsAt_of_accept (caseIsSent_at x hS) hP hA₀ hk₀ hd₀ hx₀) hj hA

/-! ## Anchored on files the statement is given -/

/-- **Each accepted play's reference statements are the files `B₀`'s, or its blobs collide with `B₀`**, on a case
map whose cases are the ones sent (`CaseIsSent`) and whose accepted plays decode the parameters `P` (`DecodesAt`), at
files `B₀` the statement is given that give the reference statements `xs₀` and passed the tables' check at template
0's member. No accepted play is read beyond the one at hand. -/
theorem refsAtOr_of_files {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {B₀ : Array ByteArray}
    {xs₀ : List Claims.Msg} (hS : CaseIsSent C) (hP : DecodesAt C P)
    (hT₀ : ∀ m₀, memberOf P 0 = some m₀ →
      Properties.Spot.HoldsRegisteredTables m₀.own (_root_.Core.Blobs.ofArray B₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray B₀) = some xs₀) : RefsAtOr C xs₀ B₀ := by
  by_cases hz : P.circuits.size = 0
  · intro o k p c e xs hA hk hdec hx
    refine .inl ?_
    have hpP : p = P := hP o k p c e hA hk hdec
    subst hpP
    have h1 := SpotFlock.HeldOf.refStmtsOf_length hx
    have h2 := SpotFlock.HeldOf.refStmtsOf_length hx₀
    rw [hz] at h1 h2
    rw [List.eq_nil_of_length_eq_zero h1, List.eq_nil_of_length_eq_zero h2]
  · obtain ⟨r, hr, -⟩ := SpotFlock.HeldOf.refStmtsOf_getElem? hx₀ (Nat.pos_of_ne_zero hz)
    obtain ⟨m₀, -, -, hm, -⟩ := refRawOf_some hr
    exact refsAtOr_of_tables hS hP (tablesHeldAt_of_accept hP hm) (hT₀ m₀ hm) hx₀

/-- **Each accepted play at one statement has the held copies, or its blobs collide with the files `B₀`**:
`heldOfAtOr_at` with `RefsAtOr` from `refsAtOr_of_files`, where `A.h` is the copy `heldOf` builds from the parameters
`P`, the record read from `x` and the reference statements `xs₀` the files `B₀` give (`CopyOf`). -/
theorem heldOfAtOr_of_files {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {B₀ : Array ByteArray} {xs₀ : List Claims.Msg} (u₀ : Unread) (hS : CaseIsSent C)
    (hd : DescAt (atStmt C x)) (hP : DecodesAt (atStmt C x) P)
    (hT₀ : ∀ m₀, memberOf P 0 = some m₀ →
      Properties.Spot.HoldsRegisteredTables m₀.own (_root_.Core.Blobs.ofArray B₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray B₀) = some xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAtOr A (atStmt C x) B₀ :=
  heldOfAtOr_at u₀ hd hP (refsAtOr_of_files (caseIsSent_at x hS) hP hT₀ hx₀) hj hA

end Verity.SampledTwo
