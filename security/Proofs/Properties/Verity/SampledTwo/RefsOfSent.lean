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
its digest, with a collision as the second way out: two of the blobs sent differ under one SHA-512 (`BlobsCollide`,
`SentCollides`). No statement takes a premise on SHA-512.

* **The files** are pinned: two plays' blobs give one file under one digest, or collide (`get_same`).
* **Each table's bytes** are pinned: where two plays' blobs both hold a pinned table, its two files are one or collide.
* **Which pinned tables the blobs hold** is pinned by no digest of its own. Under one list of table names on both
  plays' blobs the tables are the same or collide (`tables_same`), and so are the reference statements
  (`refStmts_same`). The names are pinned where both plays' blobs passed the tables' check at one registration
  (`Spot.OneStageZk.tablesHeld`: the registration's circuit file, under its digest, looks up exactly the pinned tables
  the blobs hold): `names_same`.

So `RefsAt` holds at the reference statements of any one accepted play, or two of the blobs sent collide, on a case
map whose accepted plays all passed the tables' check at one registration (`TablesHeldAt`): `refsAt_of_held`, and
`heldOfAt_or_collide` for the held copies at the cases at one statement.

## What the form checks of the tables

The form runs the tables' check only in a member's statement (the first step of `statementOf`), and reads a member's
statement only where one of the evidence's statements names the member (`runMembers`). `accepted_tables` is that
check at an accepted play: each member a statement names has its circuit file on the blobs, and the file looks up
exactly the tables the blobs hold (`Properties.Spot.HoldsRegisteredTables`). So `TablesHeldAt C q` holds where every
accepted play names the registration `q` in a statement (`NamesAt`, `tablesHeldAt_of_names`).

## What this does not show

* `NamesAt` or `TablesHeldAt` at any case map. An accepted play whose evidence names no statement runs the tables'
  check at no member, and two accepted plays that name different members are held to two circuit files' lookups. The
  reference statement of every template reads the tables the blobs hold, whether or not a statement names its member.
  So the read no digest pins is which of the verifier's pinned tables a play's blobs hold, at a play that names no
  statement of a registration every other accepted play names.
* That an accepted play names a statement, or that two members' circuit files look up the same tables.
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

private theorem foldlM_step {σ α ε : Type} {f : σ → α → Except ε σ} :
    ∀ (l : List α) (s : σ) {r : σ}, l.foldlM f s = .ok r → ∀ z ∈ l, ∃ s₁ s₂, f s₁ z = .ok s₂
  | [], _, _, _, z, hz => by cases hz
  | a :: l, s, r, h, z, hz => by
    rw [List.foldlM_cons] at h
    obtain ⟨s', hs', h'⟩ := ebind_ok h
    rcases List.mem_cons.1 hz with rfl | hz'
    · exact ⟨s, s', hs'⟩
    · exact foldlM_step l s' h' z hz'

private theorem mapError_ok {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

/-! ## A collision among the blobs sent -/

/-- **Two of the blobs differ under one SHA-512**: a blob of `F` and a blob of `F'` that differ and have one
SHA-512. -/
def BlobsCollide (F F' : Array ByteArray) : Prop :=
  ∃ b ∈ F, ∃ b' ∈ F', b ≠ b' ∧ Flock.Sha512.hash b = Flock.Sha512.hash b'

/-- **Two of the blobs sent differ under one SHA-512**: two plays the form accepts on `C` sent blobs that collide. -/
def SentCollides (C : CaseOf) : Prop :=
  ∃ o o', FormAccepts C o ∧ FormAccepts C o' ∧ BlobsCollide o.sent.blobs o'.sent.blobs

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
    first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · rw [ite_eq_right hc] at h
    cases hf : (_root_.Core.Blobs.ofArray F).entries.find? (·.1 == a) with
    | none =>
      rw [hf] at h
      first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    | some e =>
      rw [hf] at h
      obtain ⟨e1, e2⟩ := e
      have hb : e2 = b := by
        first
          | (cases h; rfl)
          | (simp only at h; cases h; rfl)
          | (simp [bind, Except.bind, pure, Except.pure] at h; exact h)
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

/-- **A statement the per-member form stepped passed the tables' check at its member** (`tablesHeld`, the first step
of `statementOf`). -/
theorem stepMember_tables {slice : _root_.Spot.OneStageZk.Member → Lean.Json} {bs : Core.Blobs}
    {r r' : _root_.Spot.OneStageZk.Run}
    {z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member}
    (h : _root_.Spot.OneStageZk.stepMember slice bs r z = .ok r') :
    _root_.Spot.OneStageZk.tablesHeld z.2.own bs = .ok () := by
  unfold _root_.Spot.OneStageZk.stepMember at h
  have h := mapError_ok h
  unfold _root_.Spot.OneStageZk.runFrom at h
  obtain ⟨a, hst, -⟩ := ebind_ok h
  unfold _root_.Spot.OneStageZk.Member.statement at hst
  obtain ⟨a', ha', -⟩ := ebind_ok hst
  unfold _root_.Spot.OneStageZk.statementOf at ha'
  obtain ⟨⟨⟩, ht, -⟩ := ebind_ok ha'
  exact ht

/-- **The per-member form's run passed the tables' check at each member a statement names.** -/
theorem runMembers_tables {slice : _root_.Spot.OneStageZk.Member → Lean.Json}
    {ms : Array _root_.Spot.OneStageZk.Member} {sts : Array _root_.Spot.OneStageZk.StatementFiles}
    {want : Array ℕ} {bs : Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.runMembers slice ms sts want bs = .ok r) :
    ∃ held, _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      ∀ z ∈ (sts.zip held).toList, _root_.Spot.OneStageZk.tablesHeld z.2.own bs = .ok () := by
  unfold _root_.Spot.OneStageZk.runMembers at h
  obtain ⟨held, hheld, h⟩ := ebind_ok h
  obtain ⟨r', hfold, -⟩ := ebind_ok h
  rw [← Array.foldlM_toList] at hfold
  refine ⟨held, hheld, fun z hz => ?_⟩
  obtain ⟨r₁, r₂, hstep⟩ := foldlM_step _ _ hfold z hz
  exact stepMember_tables hstep

theorem membersRun_tables {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) :
    ∃ held, _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      ∀ z ∈ (sts.zip held).toList, _root_.Spot.OneStageZk.tablesHeld z.2.own bs = .ok () := by
  unfold _root_.Spot.OneStageZk.membersRun at h
  simp only [Verity.Sampled.bind_ok] at h
  obtain ⟨held, -, h⟩ := h
  split at h
  · simp only [Verity.Sampled.bind_ok] at h
    obtain ⟨_, hthrow, -⟩ := h
    cases hthrow
  · simp only [Verity.Sampled.bind_ok] at h
    obtain ⟨ts, -, _, -, L, -, _, -, _, -, _, -, _, -, roots, -, hrun⟩ := h
    exact runMembers_tables hrun

/-- **Each member a statement of an accepted play names has its circuit file on the play's blobs, and the file looks
up exactly the pinned tables the blobs hold.** -/
theorem accepted_tables {C : CaseOf} {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence}
    (hA : FormAccepts C o) (hk : C o = some k) (hdec : Decoded k p c e) :
    ∃ held, _root_.Spot.OneStageZk.membersHeld p.members e.statements = .ok held ∧
      ∀ z ∈ (e.statements.zip held).toList, Properties.Spot.HoldsRegisteredTables z.2.own k.blobs := by
  obtain ⟨k', p', e', s, c₂, ξ₂, r, hk', hp', he', -, -, -, hrun⟩ := SessionCoins.accepted_ran hA
  have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
  subst hkk
  obtain ⟨g1, -, g3⟩ := hdec
  have hpp : p' = p := Except.ok.inj (hp'.symm.trans g1)
  subst hpp
  have hee : e' = e := Except.ok.inj (he'.symm.trans g3)
  subst hee
  obtain ⟨held, hheld, ht⟩ := membersRun_tables hrun
  exact ⟨held, hheld, fun z hz => Properties.Spot.SecurityProofs.tablesHeld_ok (ht z hz)⟩

/-! ## `RefsAt`, or a collision -/

/-- **Every accepted play's blobs passed the tables' check at the registration `q`** (`tablesHeld`, as
`Properties.Spot.HoldsRegisteredTables` reads it): `q`'s circuit file is on the blobs and looks up exactly the pinned
tables the blobs hold. -/
def TablesHeldAt (C : CaseOf) (q : _root_.Spot.OneStageZk.Own) : Prop :=
  ∀ o k, FormAccepts C o → C o = some k → Properties.Spot.HoldsRegisteredTables q k.blobs

/-- **Every accepted play names the registration `q` in a statement**: one of its evidence's statements is held to a
member whose registration is `q`. -/
def NamesAt (C : CaseOf) (q : _root_.Spot.OneStageZk.Own) : Prop :=
  ∀ o k p c e held, FormAccepts C o → C o = some k → Decoded k p c e →
    _root_.Spot.OneStageZk.membersHeld p.members e.statements = .ok held →
    ∃ z ∈ (e.statements.zip held).toList, z.2.own = q

/-- **The form's check gives `TablesHeldAt` at a registration every accepted play names in a statement**
(`accepted_tables`). -/
theorem tablesHeldAt_of_names {C : CaseOf} {q : _root_.Spot.OneStageZk.Own} (hN : NamesAt C q) :
    TablesHeldAt C q := by
  intro o k hA hk
  have hA' := hA
  obtain ⟨k', hk', p, c, e, pub, prof, hdec, -⟩ := hA'
  have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
  subst hkk
  obtain ⟨held, hheld, ht⟩ := accepted_tables hA hk hdec
  obtain ⟨z, hz, hzq⟩ := hN o _ p c e held hA hk hdec hheld
  rw [← hzq]
  exact ht z hz

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

/-- **The reference statements of one accepted play are every accepted play's, or two of the blobs sent differ under
one SHA-512**, on a case map whose cases are the ones sent (`CaseIsSent`), whose accepted plays decode one parameters
(`DecodesAt`) and passed the tables' check at one registration (`TablesHeldAt`). -/
theorem refsAt_of_held {C : CaseOf} {P : VStar.Recursive.SampledTwo.Params} {q : _root_.Spot.OneStageZk.Own}
    {o₀ : Verity.Exchange} {k₀ : Case} {c₀ : _root_.Spot.OneStage.Commitment}
    {e₀ : VStar.Recursive.SampledTwo.Evidence} {xs₀ : List Claims.Msg}
    (hS : CaseIsSent C) (hP : DecodesAt C P) (hT : TablesHeldAt C q)
    (hA₀ : FormAccepts C o₀) (hk₀ : C o₀ = some k₀) (hd₀ : Decoded k₀ P c₀ e₀)
    (hx₀ : refStmtsOf P k₀.blobs = some xs₀) : RefsAt C xs₀ ∨ SentCollides C := by
  by_cases hcol : SentCollides C
  · exact .inr hcol
  refine .inl fun o k p c e xs hA hk hdec hx => ?_
  have hpP : p = P := hP o k p c e hA hk hdec
  subst hpP
  have hb := (hS o k _ c e hk hdec).2
  have hb₀ := (hS o₀ k₀ _ c₀ e₀ hk₀ hd₀).2
  have hh := hT o k hA hk
  have hh₀ := hT o₀ k₀ hA₀ hk₀
  rw [hb] at hx hh
  rw [hb₀] at hx₀ hh₀
  have hne : ¬ BlobsCollide o.sent.blobs o₀.sent.blobs := fun hc => hcol ⟨o, o₀, hA, hA₀, hc⟩
  exact (refStmts_same hx hx₀ ((names_same hh hh₀).resolve_right hne)).resolve_right hne

/-- **The held copies at the cases at one statement, or two of the blobs sent differ under one SHA-512**:
`heldOfAt_at` with `RefsAt` from `refsAt_of_held`, at the reference statements `xs₀` of one accepted play. -/
theorem heldOfAt_or_collide {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {q : _root_.Spot.OneStageZk.Own} {o₀ : Verity.Exchange} {k₀ : Case}
    {c₀ : _root_.Spot.OneStage.Commitment} {e₀ : VStar.Recursive.SampledTwo.Evidence} {xs₀ : List Claims.Msg}
    (u₀ : Unread) (hS : CaseIsSent C) (hd : DescAt (atStmt C x)) (hP : DecodesAt (atStmt C x) P)
    (hT : TablesHeldAt (atStmt C x) q) (hA₀ : FormAccepts (atStmt C x) o₀) (hk₀ : atStmt C x o₀ = some k₀)
    (hd₀ : Decoded k₀ P c₀ e₀) (hx₀ : refStmtsOf P k₀.blobs = some xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAt A (atStmt C x) ∨ SentCollides (atStmt C x) :=
  (refsAt_of_held (caseIsSent_at x hS) hP hT hA₀ hk₀ hd₀ hx₀).imp_left fun hR => heldOfAt_at u₀ hd hP hR hj hA

end Verity.SampledTwo
