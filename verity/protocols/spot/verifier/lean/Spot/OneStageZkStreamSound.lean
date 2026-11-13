/-
The lemmas `StreamSound` is proved from: what the stream computes on the blobs it holds is what `checkBlobs` computes
on all the blobs fed.

* `find?_sublist`, `Blobs.get_sublist`, `tablesOf_sublist`: a search of the blobs held finds what a search of all the
  blobs fed finds, where the blobs held are a sublist of those fed, their SHA-512 distinct, and the one entry the
  search can match is held;
* `verdictOf_split`, `StepKept.step_of`: a session's `step` is its kept pieces' step, its verdict split at the
  coin-tree key's freshness, the one piece that reads the sessions before it.
-/
module

public import Spot.OneStageZkStream

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk.Streamed

open Lean (Json)
open Flock (V)

/-- Two entries of a list whose first parts are distinct are one where their first parts are. -/
theorem key_unique {α β : Type} : ∀ {l : List (α × β)}, (l.map (·.1)).Nodup →
    ∀ {x y : α × β}, x ∈ l → y ∈ l → x.1 = y.1 → x = y
  | [], _, _, _, hx, _, _ => by cases hx
  | a :: l, hn, x, y, hx, hy, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 hx with hxa | hx' <;> rcases List.mem_cons.1 hy with hya | hy'
    · rw [hxa, hya]
    · subst hxa
      exact absurd (by rw [h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hy') hn.1
    · subst hya
      exact absurd (by rw [← h]; exact List.mem_map_of_mem (f := fun z : α × β => z.1) hx') hn.1
    · exact key_unique hn.2 hx' hy' h

/-- **A search of a sublist finds what a search of the list finds**, where at most one entry of the list matches and
every entry that matches is in the sublist. -/
theorem find?_sublist {α : Type} {p : α → Bool} {l₁ l₂ : List α} (hs : l₁.Sublist l₂)
    (hu : ∀ x ∈ l₂, ∀ y ∈ l₂, p x = true → p y = true → x = y) (hin : ∀ x ∈ l₂, p x = true → x ∈ l₁) :
    l₁.find? p = l₂.find? p := by
  cases h₂ : l₂.find? p with
  | none =>
    rw [List.find?_eq_none] at h₂ ⊢
    exact fun x hx => h₂ x (hs.subset hx)
  | some x =>
    have hx := List.mem_of_find?_eq_some h₂
    have hp := List.find?_some h₂
    cases h₁ : l₁.find? p with
    | none => exact absurd hp (List.find?_eq_none.mp h₁ x (hin x hx hp))
    | some y =>
      exact congrArg some (hu y (hs.subset (List.mem_of_find?_eq_some h₁)) x hx (List.find?_some h₁) hp)

/-- `==` on bytes is equality. -/
theorem bytes_beq {a b : ByteArray} : (a == b) = true ↔ a = b := by
  constructor
  · intro h
    exact ByteArray.ext (eq_of_beq (show (a.data == b.data) = true from h))
  · rintro rfl
    exact (show (a.data == a.data) = true from beq_self_eq_true _)

/-- **The blob held under a digest is the blob fed under it**, where the blobs held are a sublist of those fed, their
SHA-512 distinct, and the entry fed under the digest, if any, is held; a refusal of the digest is the same refusal. -/
theorem Blobs.get_sublist {held full : Core.Blobs} (hs : held.entries.toList.Sublist full.entries.toList)
    (hn : (full.entries.toList.map (·.1)).Nodup) {digest what : String}
    (hin : ∀ x ∈ full.entries.toList, Flock.hexOf x.1 = digest → x ∈ held.entries.toList) :
    held.get digest what = full.get digest what := by
  unfold Core.Blobs.get
  cases hu : Flock.unhex digest with
  | error e => rfl
  | ok d =>
    show (if (d.size != 64 || Flock.hexOf d != digest) = true then _ else _) =
      (if (d.size != 64 || Flock.hexOf d != digest) = true then _ else _)
    by_cases hc : (d.size != 64 || Flock.hexOf d != digest) = true
    · rw [if_pos hc, if_pos hc]
      rfl
    · rw [if_neg hc, if_neg hc]
      have hd : Flock.hexOf d = digest := by
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
        exact hc.2
      have hf : held.entries.find? (·.1 == d) = full.entries.find? (·.1 == d) := by
        rw [← Array.find?_toList, ← Array.find?_toList]
        refine find?_sublist hs (fun x hx y hy hpx hpy => key_unique hn hx hy ?_) (fun x hx hpx => hin x hx ?_)
        · exact (bytes_beq.mp hpx).trans (bytes_beq.mp hpy).symm
        · rw [bytes_beq.mp hpx, hd]
      rw [hf]

theorem filterMap_congr' {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.filterMap f = l.filterMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.filterMap_cons, List.filterMap_cons, h a List.mem_cons_self,
      filterMap_congr' fun x hx => h x (List.mem_cons_of_mem _ hx)]

/-- Two digests with one hex are one. -/
theorem hexOf_inj {a b : ByteArray} (h : Flock.hexOf a = Flock.hexOf b) : a = b := by
  have ha := Flock.Hex.unhex_hexOf a
  rw [h, Flock.Hex.unhex_hexOf b] at ha
  exact (Except.ok.inj ha).symm

/-- **The pinned tables held are the pinned tables fed**, where the blobs held are a sublist of those fed, their
SHA-512 distinct, and every blob fed under a pinned table's SHA-512 is held. -/
theorem tablesOf_sublist {held full : Core.Blobs} (hs : held.entries.toList.Sublist full.entries.toList)
    (hn : (full.entries.toList.map (·.1)).Nodup)
    (hin : ∀ x ∈ full.entries.toList, (∃ t ∈ Flock.MUFU_TABLES, Flock.hexOf x.1 = t.sha512) →
      x ∈ held.entries.toList) :
    tablesOf held = tablesOf full := by
  unfold tablesOf
  refine filterMap_congr' fun t ht => ?_
  rw [← Array.find?_toList, ← Array.find?_toList]
  refine congrArg _ (find?_sublist hs (fun x hx y hy hpx hpy => ?_) (fun x hx hpx => ?_))
  · rcases x with ⟨dx, bx⟩
    rcases y with ⟨dy, by_⟩
    dsimp only at hpx hpy
    exact key_unique hn hx hy (hexOf_inj ((beq_iff_eq.mp hpx).trans (beq_iff_eq.mp hpy).symm))
  · rcases x with ⟨dx, bx⟩
    dsimp only at hpx
    exact hin _ hx ⟨t, ht, beq_iff_eq.mp hpx⟩

/-- **`verdictOf` is the session's proof files, its key's freshness, then the rest** (`restOf`), the files read once
for the rest's proofs: the split `StepKept` keeps. -/
theorem verdictOf_split (I : Flock.Witness.Inputs) (scope ksFiles population : V Unit) (coinKeys : Array String)
    (record : String) (draw : Option Json) (st : SetUp) (files : V (Array ByteArray)) (coinKey : Option String) :
    verdictOf I scope ksFiles population coinKeys record draw st files coinKey =
      ((files >>= fun _ => pure ()) >>= fun _ =>
        (match coinKey with | some key => Flock.CoinTree.checkFresh coinKeys key | none => pure () : V Unit) >>=
          fun _ => files >>= restOf I scope ksFiles population record draw st) := by
  cases files <;> cases coinKey <;> rfl

/-- **A session's `step` is the step on what is kept of it**, after any run: the verdict's pieces made once, on the
blobs held, and only the coin-tree key's freshness read against the run. -/
theorem StepKept.step_of (a : StepAt) (bs : Core.Blobs) (s : RunFiles) (r : Run) :
    (StepKept.of a bs s).step r = a.step bs r s := by
  unfold StepKept.of StepAt.step OneStageZk.step
  cases hg : bs.get s.record s!"session {s.name}'s record" with
  | error e => rfl
  | ok rb =>
    simp only [ok_bind]
    cases hu : String.fromUTF8? rb with
    | none => rfl
    | some record =>
      dsimp only
      cases hd : drawFitsWhy WORD (Flock.Witness.recordDraw record) with
      | some what => rfl
      | none =>
        simp only [verdictOf_split]
        rfl

/-- **What the plan's distinctness check holds where it passes**: no digest is one `seen` holds, their hex is
distinct, and none is a pinned table's. -/
theorem distinctFrom_ok {tabs : List ByteArray} : ∀ {seen : Std.HashSet String} {l : List ByteArray},
    distinctFrom tabs seen l = .ok () →
      (∀ b ∈ l, seen.contains (Flock.hexOf b) = false) ∧ (l.map Flock.hexOf).Nodup ∧ ∀ b ∈ l, b ∉ tabs
  | _, [], _ => ⟨fun _ h => (nomatch h), List.nodup_nil, fun _ h => (nomatch h)⟩
  | seen, b :: bs, h => by
    simp only [distinctFrom] at h
    split at h
    · cases h
    rename_i hp
    split at h
    · cases h
    rename_i hs
    obtain ⟨hseen, hnd, hpin⟩ := distinctFrom_ok h
    have hne : ∀ b' ∈ bs, Flock.hexOf b ≠ Flock.hexOf b' ∧ seen.contains (Flock.hexOf b') = false := by
      intro b' hb'
      have := hseen b' hb'
      rw [Std.HashSet.contains_insert, Bool.or_eq_false_iff] at this
      refine ⟨fun he => ?_, this.2⟩
      rw [he, beq_self_eq_true] at this
      exact Bool.false_ne_true this.1.symm
    refine ⟨?_, ?_, ?_⟩
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact Bool.eq_false_iff.2 hs
      · exact (hne x hx).2
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨fun hm => ?_, hnd⟩
      obtain ⟨b', hb', he⟩ := List.mem_map.1 hm
      exact (hne b' hb').1 he.symm
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · intro hm
        exact hp (List.any_eq_true.2 ⟨x, hm, decide_eq_true rfl⟩)
      · exact hpin x hx

/-- Digests whose hex is distinct are distinct. -/
theorem nodup_of_hex {l : List ByteArray} (h : (l.map Flock.hexOf).Nodup) : l.Nodup :=
  List.Pairwise.of_map Flock.hexOf (fun _ _ hne he => hne (congrArg Flock.hexOf he)) h

theorem listMapM_congr {α β : Type} {f g : α → V β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.mapM f = l.mapM g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, List.mapM_cons, h a List.mem_cons_self,
      listMapM_congr fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem arrayMapM_congr {α β : Type} {f g : α → V β} {xs : Array α} (h : ∀ x ∈ xs, f x = g x) :
    xs.mapM f = xs.mapM g := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList,
    listMapM_congr fun x hx => h x (Array.mem_toList_iff.mp hx)]

/-- Two sets of blobs that answer alike for the digest `d`, under every name a refusal gives it. -/
abbrev Alike (b₁ b₂ : Core.Blobs) (d : String) : Prop := ∀ what, b₁.get d what = b₂.get d what

/-- Two sets of blobs that answer alike for a session's record and rep files. -/
abbrev SessionAlike (b₁ b₂ : Core.Blobs) (s : RunFiles) : Prop :=
  Alike b₁ b₂ s.record ∧ ∀ x ∈ s.reps, Alike b₁ b₂ x.2

theorem sessionEvidence_congr {b₁ b₂ : Core.Blobs} {s : RunFiles} (h : SessionAlike b₁ b₂ s) :
    sessionEvidence b₁ s = sessionEvidence b₂ s := by
  unfold sessionEvidence
  rw [h.1]
  congr 1
  funext record
  congr 1
  exact arrayMapM_congr fun x hx => by
    rcases x with ⟨f, d⟩
    dsimp only
    rw [h.2 _ hx]

theorem filesOf_congr {b₁ b₂ : Core.Blobs} {s : RunFiles} (h : SessionAlike b₁ b₂ s) (st : SetUp) :
    filesOf b₁ s st = filesOf b₂ s st := by
  unfold filesOf
  cases st with
  | error e => rfl
  | ok v =>
    obtain ⟨spec, rest⟩ := v
    dsimp only [ok_bind]
    refine arrayMapM_congr fun x _ => ?_
    obtain ⟨a, b, w⟩ := x
    dsimp only
    cases hf : s.reps.find? (·.1 == s!"{w.replace "/" "."}.bin") with
    | none => rfl
    | some y =>
      obtain ⟨n, d⟩ := y
      dsimp only
      rw [h.2 _ (Array.mem_of_find?_eq_some hf)]

theorem rootBTables_congr {b₁ b₂ : Core.Blobs} {p : Own} (hc : Alike b₁ b₂ p.circuit) (hp : Alike b₁ b₂ p.pub)
    (ht : tablesOf b₁ = tablesOf b₂) (record : String) : rootBTables p b₁ record = rootBTables p b₂ record := by
  unfold rootBTables inputsOf
  rw [hc, hp, ht]

theorem StepKept.of_congr {b₁ b₂ : Core.Blobs} {s : RunFiles} (h : SessionAlike b₁ b₂ s) (a : StepAt) :
    StepKept.of a b₁ s = StepKept.of a b₂ s := by
  unfold StepKept.of
  rw [h.1]
  simp only [filesOf_congr h]

/-- **What is kept of a session on the blobs held is what is kept of it on all the blobs**, where the two answer
alike for its record, its rep files, and its member's circuit and public files, and hold one set of pinned tables. -/
theorem SessionKept.of_congr {b₁ b₂ : Core.Blobs} {m : Member} {s : RunFiles} (h : SessionAlike b₁ b₂ s)
    (hc : Alike b₁ b₂ m.own.circuit) (hp : Alike b₁ b₂ m.own.pub) (ht : tablesOf b₁ = tablesOf b₂) (a : V StepAt) :
    SessionKept.of m a b₁ s = SessionKept.of m a b₂ s := by
  unfold SessionKept.of
  rw [h.1, sessionEvidence_congr h]
  simp only [StepKept.of_congr h, rootBTables_congr hc hp ht]

/-- A member's files on what is kept of its circuit file and program read the blobs for its public file alone. -/
theorem Member.filesOn_congr {b₁ b₂ : Core.Blobs} (m : Member) (hp : Alike b₁ b₂ m.own.pub) (tags0 : Flock.Tags)
    (tables : List (String × ByteArray)) (fk : FileKept) (pk : ProgramKept) (slice : Json) :
    m.filesOn tags0 tables fk pk slice b₁ = m.filesOn tags0 tables fk pk slice b₂ := by
  unfold Member.filesOn statementOn
  rw [hp]

/-- What `fileOf` keeps of a circuit file reads the blobs for its digest alone. -/
theorem fileOf_congr {b₁ b₂ : Core.Blobs} {d : String} (hd : Alike b₁ b₂ d) (tags0 : Flock.Tags)
    (tables : List (String × ByteArray)) : fileOf tags0 tables b₁ d = fileOf tags0 tables b₂ d := by
  unfold fileOf
  rw [hd]

/-- **A member's files' verdict on the blobs held is its files' verdict on all the blobs fed** (`Member.files`), where
the stream holds what the files' pass keeps of its circuit file on all the blobs, a program kept as `programOf` keeps
it, the tables fed, and its public file; the program it keeps after is kept so too. -/
theorem memberFiles_eq {C : Ctx} {S : State} {full : Core.Blobs} {m : Member} {tags0 : Flock.Tags}
    (ht0 : C.tags0 = some tags0) (htag : Flock.Tags.find? STATEMENT = some tags0)
    (htab : tablesOf S.blobs = tablesOf full)
    (hfile : ∃ h b, S.file = some (h, b, some (fileOf tags0 (tablesOf full) full m.own.circuit)))
    (hprog : ∀ pk, S.prog = some pk → pk = programOf pk.bytes) (hp : Alike S.blobs full m.own.pub) :
    (memberFiles C S m).1 = m.files (C.slice m) full ∧
      ∀ pk, (memberFiles C S m).2 = some pk → pk = programOf pk.bytes := by
  obtain ⟨h, b, hf⟩ := hfile
  unfold memberFiles
  rw [ht0]
  dsimp only
  rw [hf]
  dsimp only
  refine ⟨?_, ?_⟩
  · rw [htab, ProgramKept.next_eq hprog, Member.filesOn_congr m hp, ← Member.files_on htag]
  · intro pk hpk
    cases hpk
    rw [ProgramKept.next_eq hprog]
    rfl

/-- **What the stream keeps of every member's files, after the feed**: each member's files' verdict on all the blobs
fed, by its place. -/
def FilesKept (C : Ctx) (full : Core.Blobs) (S : State) : Prop :=
  S.files.size = C.ms.size ∧
    ∀ i (m : Member), C.ms[i]? = some m → S.files[i]? = some (m.files (C.slice m) full >>= fun _ => pure ())

/-- **What the stream keeps of every statement's session, after the feed**: what is kept of it on all the blobs fed,
at its member's files' verdict there, by its place. -/
def SessionsKept (C : Ctx) (full : Core.Blobs) (S : State) : Prop :=
  ∀ held, membersHeld C.ms C.sts = .ok held → ∀ k (st : StatementFiles) (m : Member) (s : RunFiles), C.sts[k]? = some st → held[k]? = some m →
    st.sessions[0]? = some s →
      S.sessions[k]? = some (some (SessionKept.of m (m.files (C.slice m) full) full s))

/-- **A statement's session at the finish is its session in `runMembers`' fold**, where the stream kept it as it reads
on all the blobs and the files' pass read its member's statement `a`. -/
theorem stepKept_eq {S : State} {full : Core.Blobs} {files : List (String × StepAt)} {st : StatementFiles}
    {m : Member} {s : RunFiles} {k : Nat} {slice : Json} {a : StepAt} {r : Run} (hs : st.sessions = #[s])
    (hk : S.sessions[k]? = some (some (SessionKept.of m (m.files slice full) full s)))
    (ha : m.files slice full = .ok a) (hst : statementAt files m = .ok a) :
    stepKept S r ((st, m), k) = stepMember (statementAt files) full r (st, m) := by
  unfold stepKept stepMember runFrom
  rw [hk, hst, ha, hs]
  simp only [SessionKept.of, Except.toOption, Option.map_some, StepKept.step_of, ok_bind]
  simp [← Array.foldlM_toList]

/-- **The fold's half of `StreamSound`, stated and not proved**: on a feed in the per-member stream's order, its
state after every step keeps every member's files and every statement's session as they read on all the blobs fed. -/
def StepsKept : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) ms sts plan (bs : List ByteArray),
    (membersStream p c e ms sts).plan = .ok plan → (membersStream p c e ms sts).Fed plan bs →
      FilesKept (Ctx.of p c e ms sts) (Core.Blobs.ofArray bs.toArray) ((membersStream p c e ms sts).steps bs) ∧
        SessionsKept (Ctx.of p c e ms sts) (Core.Blobs.ofArray bs.toArray) ((membersStream p c e ms sts).steps bs)

/-- **The finish's half of `StreamSound`, stated and not proved**: on a state that keeps every member's files and
every statement's session as they read on the blobs `full`, the finish is `checkBlobs` on them. -/
def FinishKept : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) ms sts (S : State) (full : Core.Blobs),
    p.members = some ms → e.statements = some sts →
      FilesKept (Ctx.of p c e ms sts) full S → SessionsKept (Ctx.of p c e ms sts) full S →
        finish (Ctx.of p c e ms sts) S = checkBlobs p c e full

/-- **`StreamSound` is its two halves**: every other form's stream is `checkBlobs` on its blobs (`keepAll_sound`), and
the per-member form's is where its fold keeps what it reads (`StepsKept`) and its finish reads it as `checkBlobs` does
(`FinishKept`). -/
theorem streamSound_of (hs : StepsKept) (hf : FinishKept) : StreamSound := by
  intro p c e plan bs hp hfed
  change (stream p c e).plan = .ok plan at hp
  change (stream p c e).Fed plan bs at hfed
  change (stream p c e).finish ((stream p c e).steps bs) = checkBlobs p c e (Core.Blobs.ofArray bs.toArray)
  match hm : p.members, he : e.statements with
  | some ms, some sts =>
    have hst : stream p c e = membersStream p c e ms sts := by simp only [stream, hm, he]
    rw [hst] at hp hfed ⊢
    obtain ⟨h1, h2⟩ := hs p c e ms sts plan bs hp hfed
    exact hf p c e ms sts _ _ hm he h1 h2
  | none, none =>
    have hst : stream p c e = keepAll p c e (oneNames p e) := by simp only [stream, hm, he]
    rw [hst]
    exact keepAll_sound _ _ _ _ _
  | some _, none =>
    have hst : stream p c e = keepAll p c e (pure #[]) := by simp only [stream, hm, he]
    rw [hst]
    exact keepAll_sound _ _ _ _ _
  | none, some _ =>
    have hst : stream p c e = keepAll p c e (pure #[]) := by simp only [stream, hm, he]
    rw [hst]
    exact keepAll_sound _ _ _ _ _

end Spot.OneStageZk.Streamed
