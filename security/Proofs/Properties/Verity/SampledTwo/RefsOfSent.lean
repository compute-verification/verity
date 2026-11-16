module

public import Proofs.Properties.Verity.SampledTwo.HeldOfAtDesc
public import Proofs.Properties.Verity.SampledTwo.StageCut
public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Spot.Tables
public import Proofs.Flock.Bytes

/-!
# The reference statements of the blobs sent, digest by digest

The held copies at an accepted play are built at the reference statements its blobs give (`HeldOfAtDesc`), so one
session of record's copies are every accepted play's only where each play's blobs give the same reference statements
`xs₀`. A reference statement (`SpotFlock.HeldOf.refRawOf`) reads three things of the blobs: its
member's circuit file and public file, each the blob under a digest the parameters name (`Core.Blobs.get`), and the
member's own lookup tables (`Spot.OneStageZk.ownTables`): the pinned tables its circuit file looks up, each the first
blob under a digest the verifier pins (`Flock.MUFU_TABLES`). The blobs are the ones the prover sent (`CaseIsSent`), so
each read is checked here against its digest, with a collision as the second way out: a blob of the play and a blob of
an array the statement is given differ under one SHA-512 (`BlobsCollide`). No statement takes a premise on SHA-512,
and no collision is under an `∃` over plays: each conclusion is per accepted play (`RefsAtOr`, `HeldOfAtOr`), its pair
a blob of that play and a blob of the array given.

* **The files** are pinned: two plays' blobs give one file under one digest, or collide (`get_same`).
* **Each table's bytes** are pinned: where two plays' blobs both hold a pinned table, its two files are one or collide.
* **Which pinned tables a member reads** is pinned by its circuit file, where the blobs hold every table it looks up
  (`Properties.Spot.HandedOwnTables`): two plays' blobs that do hand the member the same tables, or collide
  (`ownTables_same`, from `tablesAmong_same`), and so give the same reference statements (`refStmts_same`).

## What the form checks of the tables

The per-member form's run reads every member's statement, whichever members the sample holds
(`Spot.OneStageZk.membersRun_files`), and a member's statement hands the member its own tables only where the blobs
hold every table its circuit file looks up (`Properties.Spot.SecurityProofs.statement_tables`). So every member of an
accepted play's parameters has its tables on the play's blobs (`accepted_tables`), at every case map
(`tablesHeldAt_of_accept`). At blobs `F₀` that give `xs₀` and hold each member's tables, each accepted play's blobs
give `xs₀` or collide with `F₀` (`refsAtOr_of_tables`); `refsAtOr_of_files` is that at files the statement is given,
and `heldOfAtOr_of_files` the held copies per play at the cases at one statement (`HeldOfAtOr`, `heldOfAtOr_at`).

## What this does not show

* `DecodesAt` at any case map, or that the files the statement is given hold each member's tables.
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

private theorem tablesAmong_ofArray (names : List String) (G : Array ByteArray) :
    _root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray G) =
      Flock.MUFU_TABLES.filterMap fun t =>
        if names.contains t.name = true then (Verity.Sampled.tableFile t.sha512 G).map (fun b => (t.name, b))
        else none := by
  unfold _root_.Spot.OneStageZk.tablesAmong
  rw [Verity.Sampled.tablesOf_ofArray, List.filter_filterMap]
  refine List.filterMap_congr fun t _ => ?_
  cases Verity.Sampled.tableFile t.sha512 G <;> simp [Option.filter]

/-- **Blobs that hold the pinned tables among `names` under one list of names hold the same tables, or collide**:
each pinned table both hold is the first blob under the table's pinned digest. -/
theorem tablesAmong_same {names : List String} {F F' : Array ByteArray}
    (hn : (_root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray F)).map (·.1) =
      (_root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray F')).map (·.1)) :
    _root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray F) =
      _root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray F') ∨ BlobsCollide F F' := by
  by_cases hcol : BlobsCollide F F'
  · exact .inr hcol
  refine .inl ?_
  rw [tablesAmong_ofArray names F, tablesAmong_ofArray names F'] at hn ⊢
  refine filterMap_eq_of_names (name := fun t : Flock.MufuTable => t.name) (L := Flock.MUFU_TABLES)
    (fun t ht t' ht' e => List.inj_on_of_nodup_map mufu_nodup ht ht' e)
    (fun t x h => ?_) (fun t x h => ?_) hn (fun t _ x y hx hy => ?_)
  · split at h
    · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rfl
    · cases h
  · split at h
    · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rfl
    · cases h
  · by_cases hc : names.contains t.name = true
    · rw [if_pos hc] at hx hy
      obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hx
      obtain ⟨a', ha', rfl⟩ := Option.map_eq_some_iff.1 hy
      have e : a = a' := by
        by_contra hne
        exact hcol ⟨a, tableFile_mem ha, a', tableFile_mem ha', hne,
          Properties.NetworkRecords.Bytes.hexOf_inj
            ((Verity.Sampled.tableFile_hash ha).trans (Verity.Sampled.tableFile_hash ha').symm)⟩
      rw [e]
    · rw [if_neg hc] at hx
      cases hx

/-- **Tables handed to the registration `q` are its own tables** (`Spot.OneStageZk.ownTables`). -/
theorem ownTables_of_handed {q : _root_.Spot.OneStageZk.Own} {bs : Core.Blobs} {tables : List (String × ByteArray)}
    (h : Properties.Spot.HandedOwnTables q bs tables) : _root_.Spot.OneStageZk.ownTables q bs = .ok tables := by
  obtain ⟨c, names, hc, hl, rfl, -⟩ := h
  unfold _root_.Spot.OneStageZk.ownTables
  rw [hc]
  show (_root_.Spot.OneStageZk.lookupsOf c >>= fun names => pure (_root_.Spot.OneStageZk.tablesAmong names bs)) = _
  rw [hl]
  rfl

/-- **Two plays' blobs that each hold every table one registration's circuit file looks up hand it the same tables,
or collide**: the circuit file under the registration's digest is one file, or two blobs collide. -/
theorem ownTables_same {q : _root_.Spot.OneStageZk.Own} {F F' : Array ByteArray}
    {tables tables' : List (String × ByteArray)}
    (h : Properties.Spot.HandedOwnTables q (_root_.Core.Blobs.ofArray F) tables)
    (h' : Properties.Spot.HandedOwnTables q (_root_.Core.Blobs.ofArray F') tables') :
    tables = tables' ∨ BlobsCollide F F' := by
  obtain ⟨c, names, hc, hl, rfl, hn⟩ := h
  obtain ⟨c', names', hc', hl', rfl, hn'⟩ := h'
  rcases get_same hc hc' with rfl | hcol
  · obtain rfl : names = names' := Except.ok.inj (hl.symm.trans hl')
    exact tablesAmong_same (hn.trans hn'.symm)
  · exact .inr hcol

/-! ## The reference statements -/

/-- **A template's reference statement, read**: its member, the member's circuit and public files on the blobs, and
its own tables. -/
theorem refRawOf_some {p : VStar.Recursive.SampledTwo.Params} {bs : Core.Blobs} {t : ℕ} {r : FlockDeploy.Raw}
    (h : refRawOf p bs t = some r) :
    ∃ m circuit pub tables, memberOf p t = some m ∧ bs.get m.own.circuit "circuit" = .ok circuit ∧
      bs.get m.own.pub "public file" = .ok pub ∧ _root_.Spot.OneStageZk.ownTables m.own bs = .ok tables ∧
      r = memberRaw m circuit pub tables none
        (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8) := by
  simp only [refRawOf, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨m, hm, circuit, hc, pub, hp, tables, ht, rfl⟩ := h
  exact ⟨m, circuit, pub, tables, hm, SpotFlock.HeldOf.ok_of_toOption hc, SpotFlock.HeldOf.ok_of_toOption hp,
    SpotFlock.HeldOf.ok_of_toOption ht, rfl⟩

/-! ## The members -/

theorem memberOf_mem {p : VStar.Recursive.SampledTwo.Params} {t : ℕ} {m : _root_.Spot.OneStageZk.Member}
    (h : memberOf p t = some m) : m ∈ p.members := by
  unfold memberOf at h
  obtain ⟨_, -, hf⟩ := Option.bind_eq_some_iff.1 h
  exact Array.mem_of_find?_eq_some hf

/-- **The blobs hold every member's own tables**: each member of the parameters `p` is handed, on the blobs `bs`, every
pinned table its circuit file looks up (`Properties.Spot.HandedOwnTables`). -/
def OwnTablesHeld (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) : Prop :=
  ∀ m ∈ p.members, ∃ tables, Properties.Spot.HandedOwnTables m.own bs tables

/-- **Two plays' blobs that each hold every member's own tables give the same reference statements at one
parameters, or collide.** -/
theorem refStmts_same {p : VStar.Recursive.SampledTwo.Params} {F F' : Array ByteArray} {xs xs' : List Claims.Msg}
    (h : refStmtsOf p (_root_.Core.Blobs.ofArray F) = some xs)
    (h' : refStmtsOf p (_root_.Core.Blobs.ofArray F') = some xs')
    (hT : OwnTablesHeld p (_root_.Core.Blobs.ofArray F)) (hT' : OwnTablesHeld p (_root_.Core.Blobs.ofArray F')) :
    xs = xs' ∨ BlobsCollide F F' := by
  by_cases hcol : BlobsCollide F F'
  · exact .inr hcol
  refine .inl ?_
  unfold refStmtsOf at h h'
  refine mapM_some_congr _ h h' fun t _ x y hx hy => ?_
  have hx' : (refRawOf p (_root_.Core.Blobs.ofArray F) t).map FlockDeploy.Raw.encode = some x := hx
  have hy' : (refRawOf p (_root_.Core.Blobs.ofArray F') t).map FlockDeploy.Raw.encode = some y := hy
  obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hx'
  obtain ⟨r', hr', rfl⟩ := Option.map_eq_some_iff.1 hy'
  obtain ⟨m, circuit, pub, tables, hm, hc, hp, ht, rfl⟩ := refRawOf_some hr
  obtain ⟨m', circuit', pub', tables', hm', hc', hp', ht', rfl⟩ := refRawOf_some hr'
  have hmm : m' = m := Option.some.inj (hm'.symm.trans hm)
  subst hmm
  obtain ⟨ts, hts⟩ := hT _ (memberOf_mem hm)
  obtain ⟨ts', hts'⟩ := hT' _ (memberOf_mem hm)
  obtain rfl : tables = ts := Except.ok.inj (ht.symm.trans (ownTables_of_handed hts))
  obtain rfl : tables' = ts' := Except.ok.inj (ht'.symm.trans (ownTables_of_handed hts'))
  rw [(get_same hc hc').resolve_right hcol, (get_same hp hp').resolve_right hcol,
    (ownTables_same hts hts').resolve_right hcol]

/-! ## The tables' check at the accepted plays -/

/-- **Every member of an accepted play's parameters, sampled or not, has its own tables on the play's blobs.** -/
theorem accepted_tables {C : CaseOf} {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence}
    (hA : FormAccepts C o) (hk : C o = some k) (hdec : Decoded k p c e) : OwnTablesHeld p k.blobs := by
  obtain ⟨k', p', e', s, c₂, ξ₂, r, hk', hp', -, -, -, -, hrun⟩ := SessionCoins.accepted_ran hA
  have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
  subst hkk
  obtain ⟨g1, -, -⟩ := hdec
  have hpp : p' = p := Except.ok.inj (hp'.symm.trans g1)
  subst hpp
  obtain ⟨roots, -, hf⟩ := _root_.Spot.OneStageZk.membersRun_files hrun
  intro m hm
  obtain ⟨a, ha, -⟩ := hf m hm
  exact ⟨_, Properties.Spot.SecurityProofs.statement_tables ha⟩

/-- **Every accepted play's blobs hold its parameters' members' own tables.** -/
def TablesHeldAt (C : CaseOf) : Prop :=
  ∀ o k p c e, FormAccepts C o → C o = some k → Decoded k p c e → OwnTablesHeld p k.blobs

/-- **`TablesHeldAt` holds at every case map** (`accepted_tables`). -/
theorem tablesHeldAt_of_accept (C : CaseOf) : TablesHeldAt C :=
  fun _ _ _ _ _ hA hk hdec => accepted_tables hA hk hdec

theorem caseIsSent_at {C : CaseOf} (x : Claims.Msg) (h : CaseIsSent C) : CaseIsSent (atStmt C x) :=
  fun o k p c e hk hdec => h o k p c e (atStmt_some hk).1 hdec

/-! ## Per play: the reference statements, or a collision with the blobs given -/

/-- **Each accepted play's blobs give the reference statements `xs₀`, or collide with the blobs `F₀`**: at each play
the form accepts, the blobs the parameters' digests name give the reference statements `xs₀` (`refStmtsOf`), wherever
they give any, or the play's own blobs collide with `F₀`, which the statement is given. -/
def RefsAtOr (C : CaseOf) (xs₀ : List Claims.Msg) (F₀ : Array ByteArray) : Prop :=
  ∀ o k p c e xs, FormAccepts C o → C o = some k → Decoded k p c e → refStmtsOf p k.blobs = some xs →
    xs = xs₀ ∨ BlobsCollide o.sent.blobs F₀

/-- **Each accepted play's held copies are the ones `heldOf` builds, or its blobs collide with the blobs `F₀`**:
`HeldOfAt` per play, the play's way out a collision between its own blobs and `F₀`, which the statement is given. -/
def HeldOfAtOr (A : Live₂) (C : CaseOf) (F₀ : Array ByteArray) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    (∃ u, SpotFlock.HeldOf.heldOf i it u = some A.h) ∨ BlobsCollide o.sent.blobs F₀

/-- **`RefsAtOr` at blobs `F₀` that give `xs₀` and hold each member's own tables**, on a case map whose cases are the
ones sent (`CaseIsSent`), whose accepted plays decode one parameters (`DecodesAt`) and hold each member's own tables
(`TablesHeldAt`). -/
theorem refsAtOr_of_tables {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {F₀ : Array ByteArray}
    {xs₀ : List Claims.Msg} (hS : CaseIsSent C) (hP : DecodesAt C P) (hT : TablesHeldAt C)
    (hT₀ : OwnTablesHeld P (_root_.Core.Blobs.ofArray F₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray F₀) = some xs₀) : RefsAtOr C xs₀ F₀ := by
  intro o k p c e xs hA hk hdec hx
  have hpP : p = P := hP o k p c e hA hk hdec
  subst hpP
  have hb := (hS o k _ c e hk hdec).2
  have hh := hT o k _ c e hA hk hdec
  rw [hb] at hx hh
  exact refStmts_same hx hx₀ hh hT₀

/-- **`HeldOfAtOr` at the cases at the statement `x`**, where every accepted play there passes the descriptor check
(`DescAt`), decodes the parameters `P` (`DecodesAt`) and has blobs giving the reference statements `xs₀` or colliding
with `F₀` (`RefsAtOr`), and `A.h` is the copy `heldOf` builds from `P`, the record `j` the proved verifier's reader
reads from `x` and `xs₀` (`CopyOf`): the descriptor check gives a copy at the play's reference statements
(`HeldOf.heldOf_of_descOk`), and an accepted play at `x` carries `j` (`inputOn_at`, `jsonMsg_of_readBack`) and `P`'s
fields (`fields_of_accept`). -/
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


/-! ## Anchored on files the statement is given -/

/-- **Each accepted play's reference statements are the files `B₀`'s, or its blobs collide with `B₀`**, on a case
map whose cases are the ones sent (`CaseIsSent`) and whose accepted plays decode the parameters `P` (`DecodesAt`), at
files `B₀` the statement is given that give the reference statements `xs₀` and hold each member's own tables. No
accepted play is read beyond the one at hand. -/
theorem refsAtOr_of_files {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {B₀ : Array ByteArray}
    {xs₀ : List Claims.Msg} (hS : CaseIsSent C) (hP : DecodesAt C P)
    (hT₀ : OwnTablesHeld P (_root_.Core.Blobs.ofArray B₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray B₀) = some xs₀) : RefsAtOr C xs₀ B₀ :=
  refsAtOr_of_tables hS hP (tablesHeldAt_of_accept C) hT₀ hx₀

/-- **Each accepted play at one statement has the held copies, or its blobs collide with the files `B₀`**:
`heldOfAtOr_at` with `RefsAtOr` from `refsAtOr_of_files`, where `A.h` is the copy `heldOf` builds from the parameters
`P`, the record read from `x` and the reference statements `xs₀` the files `B₀` give (`CopyOf`). -/
theorem heldOfAtOr_of_files {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {B₀ : Array ByteArray} {xs₀ : List Claims.Msg} (u₀ : Unread) (hS : CaseIsSent C)
    (hd : DescAt (atStmt C x)) (hP : DecodesAt (atStmt C x) P)
    (hT₀ : OwnTablesHeld P (_root_.Core.Blobs.ofArray B₀))
    (hx₀ : refStmtsOf P (_root_.Core.Blobs.ofArray B₀) = some xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAtOr A (atStmt C x) B₀ :=
  heldOfAtOr_at u₀ hd hP (refsAtOr_of_files (caseIsSent_at x hS) hP hT₀ hx₀) hj hA

end Verity.SampledTwo
