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
it under its partition, the tables fed, and its public file; the program it keeps after is kept so too. -/
theorem memberFiles_eq {C : Ctx} {S : State} {full : Core.Blobs} {m : Member} {tags0 : Flock.Tags}
    (ht0 : C.tags0 = some tags0) (htag : Flock.Tags.find? STATEMENT = some tags0)
    (htab : tablesOf S.blobs = tablesOf full)
    (hfile : ∃ h b, S.file = some (h, b, some (fileOf tags0 (tablesOf full) full m.own.circuit)))
    (hprog : ∀ pk, S.prog = some pk → pk = programOf pk.bytes pk.partition) (hp : Alike S.blobs full m.own.pub) :
    (memberFiles C S m).1 = m.files (C.slice m) full ∧
      ∀ pk, (memberFiles C S m).2 = some pk → pk = programOf pk.bytes pk.partition := by
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
    ∀ (i : Nat) (m : Member), C.ms[i]? = some m → S.files[i]? = some (m.files (C.slice m) full >>= fun _ => pure ())

/-- **What the stream keeps of every statement's session, after the feed**: what is kept of it on all the blobs fed,
at its member's files' verdict there, by its place. -/
def SessionsKept (C : Ctx) (full : Core.Blobs) (S : State) : Prop :=
  ∀ held, membersHeld C.ms C.sts = .ok held → ∀ (k : Nat) (st : StatementFiles) (m : Member) (s : RunFiles), C.sts[k]? = some st → held[k]? = some m →
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
  simp

/-- A fold over a list's entries with their places is the fold over its entries, where the step at each place is
the other step at its entry. -/
theorem foldlM_zipIdx {α β : Type} {f : β → α × Nat → V β} {g : β → α → V β} :
    ∀ (l : List α) (n : Nat) (r : β), (∀ (k : Nat) (x : α) (r : β), l[k]? = some x → f r (x, n + k) = g r x) →
      (l.zipIdx n).foldlM f r = l.foldlM g r
  | [], _, _, _ => rfl
  | x :: l, n, r, h => by
    have h0 := h 0 x r rfl
    rw [Nat.add_zero] at h0
    rw [List.zipIdx_cons, List.foldlM_cons, List.foldlM_cons, h0]
    congr 1
    funext r'
    exact foldlM_zipIdx l (n + 1) r' fun k y r'' hk => by
      rw [show n + 1 + k = n + (k + 1) by omega]
      exact h (k + 1) y r'' hk

/-- **Where every member's files read, the statement the files' pass read for a member is that member's**, the
first under its name, as `heldStep` finds a statement's member. -/
theorem statementAt_of {slice : Member → Json} {bs : Core.Blobs} :
    ∀ {ms : List Member} {files : List (String × StepAt)} {m : Member}, membersFiles slice bs ms = .ok files →
      ms.find? (·.name == m.name) = some m → ∃ a, m.files (slice m) bs = .ok a ∧ statementAt files m = .ok a
  | [], _, _, _, hf => by cases hf
  | m' :: ms, files, m, h, hf => by
    unfold membersFiles at h
    cases ha : m'.files (slice m') bs with
    | error e => rw [ha] at h; cases h
    | ok a =>
      rw [ha] at h
      cases hr : membersFiles slice bs ms with
      | error e => simp only [ok_bind, hr] at h; cases h
      | ok as =>
        simp only [ok_bind, hr] at h
        cases h
        rw [List.find?_cons] at hf
        cases hn : (m'.name == m.name) with
        | true =>
          simp only [hn] at hf
          cases hf
          exact ⟨a, ha, by simp [statementAt, List.lookup]; rfl⟩
        | false =>
          simp only [hn] at hf
          obtain ⟨b, hb, hs⟩ := statementAt_of hr hf
          refine ⟨b, hb, ?_⟩
          have hn' : (m.name == m'.name) = false :=
            beq_eq_false_iff_ne.2 fun he => beq_eq_false_iff_ne.1 hn he.symm
          unfold statementAt at hs ⊢
          simp only [List.lookup, hn'] at hs ⊢
          exact hs

/-- A loop that pushes each entry's value is the fold that pushes it. -/
theorem forIn_push {α γ : Type} {f : α → Array γ → V (ForInStep (Array γ))} {g : α → V γ} :
    ∀ (l : List α) (acc : Array γ), (∀ a ∈ l, ∀ r, f a r = (fun v => ForInStep.yield (r.push v)) <$> g a) →
      forIn l acc f = l.foldlM (fun r a => (r.push ·) <$> g a) acc
  | [], _, _ => rfl
  | a :: l, acc, h => by
    rw [List.forIn_cons, List.foldlM_cons, h a List.mem_cons_self]
    cases g a with
    | error e => rfl
    | ok v => exact forIn_push l (acc.push v) fun b hb r => h b (List.mem_cons_of_mem a hb) r

/-- A fold over the places `[i, i + l.length)` is the fold over `l`'s entries, where each place's value is its entry's. -/
theorem foldlM_range' {α γ : Type} {gL : Nat → V γ} {gR : α → V γ} :
    ∀ (l : List α) (i : Nat) (acc : Array γ), (∀ (k : Nat) (x : α), l[k]? = some x → gL (i + k) = gR x) →
      (List.range' i l.length).foldlM (fun r k => (r.push ·) <$> gL k) acc =
        l.foldlM (fun r x => (r.push ·) <$> gR x) acc
  | [], _, _, _ => rfl
  | x :: l, i, acc, h => by
    have h0 := h 0 x rfl
    rw [Nat.add_zero] at h0
    rw [List.length_cons, List.range'_succ, List.foldlM_cons, List.foldlM_cons, h0]
    cases gR x with
    | error e => rfl
    | ok v =>
      exact foldlM_range' l (i + 1) (acc.push v) fun k y hk => by
        rw [show i + 1 + k = i + (k + 1) by omega]
        exact h (k + 1) y hk

/-- **The trees' check on what was kept is `membersTreesHeld` on all the blobs**, where each statement has one session
and the stream kept it, by its place, as it reads on all the blobs. -/
theorem treesKept_eq {ts : Array Challenger.LoggedTree} {log : List String} {S : State} {full : Core.Blobs}
    {sts : Array (StatementFiles × Member)}
    (hk : ∀ (k : Nat) (st : StatementFiles) (m : Member), sts[k]? = some (st, m) →
      ∃ s a, st.sessions = #[s] ∧ S.sessions[k]? = some (some (SessionKept.of m a full s))) :
    treesKept ts log sts.size S = membersTreesHeld ts log sts full := by
  unfold treesKept membersTreesHeld
  dsimp only
  congr 1
  rw [Std.Legacy.Range.forIn_eq_forIn_range', ← Array.forIn_toList]
  rw [forIn_push (g := fun k => match S.sessions[k]? with
      | some (some sk) => sk.trees
      | _ => throw "a statement's session came in no blob")]
  · rw [forIn_push (g := fun x => match x.1.sessions[0]? with
        | some s => (SessionKept.of x.2 (throw "") full s).trees
        | none => throw "")]
    · have hn : (Std.Legacy.Range.size [0:sts.size]) = sts.toList.length := by
        simp [Std.Legacy.Range.size]
      simp only [hn]
      refine foldlM_range' sts.toList 0 #[] fun k x hx => ?_
      rw [Nat.zero_add]
      obtain ⟨st, m⟩ := x
      rw [Array.getElem?_toList] at hx
      obtain ⟨s, a, hs, hS⟩ := hk k st m hx
      simp only [hS, hs, List.getElem?_toArray, List.getElem?_cons_zero]
      rfl
    · intro x hx r
      obtain ⟨st, m⟩ := x
      obtain ⟨k, hk', hx'⟩ := List.getElem_of_mem hx
      have hx'' : sts[k]? = some (st, m) := by
        rw [← Array.getElem?_toList, List.getElem?_eq_getElem hk', hx']
      obtain ⟨s, a, hs, _⟩ := hk k st m hx''
      simp only [hs]
      rw [← Array.forIn_toList]
      simp only [List.toList_toArray, List.forIn_cons, List.forIn_nil, List.getElem?_toArray,
        List.getElem?_cons_zero, SessionKept.of]
      cases hg : full.get s.record s!"session {s.name}'s record" with
      | error e => rfl
      | ok b =>
        simp only [ok_bind]
        cases String.fromUTF8? b with
        | none => rfl
        | some record =>
          cases sessionEvidence full s <;> rfl
  · intro k _ r
    cases S.sessions[k]? with
    | none => rfl
    | some o =>
      cases o with
      | none => rfl
      | some sk => cases sk.trees <;> rfl

/-- Each member's files' verdict in turn is `membersFiles`' first refusal. -/
theorem files_forM {slice : Member → Json} {bs : Core.Blobs} :
    ∀ ms : List Member, (ms.map fun m => m.files (slice m) bs >>= fun _ => pure ()).forM id =
      (membersFiles slice bs ms >>= fun _ => pure ())
  | [] => rfl
  | m :: ms => by
    have ih := files_forM (slice := slice) (bs := bs) ms
    simp only [List.map_cons, List.forM_cons, membersFiles, id]
    cases m.files (slice m) bs with
    | error e => rfl
    | ok a =>
      simp only [ok_bind]
      rw [show ∀ l : List (V Unit), (pure () :: l).forM id = l.forM id from fun _ => rfl, ih]
      cases membersFiles slice bs ms <;> rfl

/-- What the stream kept of the members' files, as a list, where it kept each by its place. -/
theorem files_toList {C : Ctx} {full : Core.Blobs} {S : State} (h : FilesKept C full S) :
    S.files.toList = C.ms.toList.map fun m => m.files (C.slice m) full >>= fun _ => pure () := by
  apply List.ext_getElem?
  intro i
  rw [Array.getElem?_toList, List.getElem?_map, Array.getElem?_toList]
  cases hm : C.ms[i]? with
  | none =>
    have hi : C.ms.size ≤ i := Array.getElem?_eq_none_iff.1 hm
    simp only [Option.map_none]
    exact Array.getElem?_eq_none_iff.2 (h.1 ▸ hi)
  | some m => simp only [Option.map_some, h.2 i m hm]

/-- A statement `heldStep` holds has one session, and its member is the parameters' first under its name. -/
theorem heldStep_ok {ms acc held : Array Member} {st : StatementFiles} (h : heldStep ms acc st = .ok held) :
    ∃ m s, held = acc.push m ∧ st.sessions = #[s] ∧ ms.find? (·.name == m.name) = some m := by
  unfold heldStep at h
  split at h
  · cases h
  · rename_i name _
    split at h
    · cases h
    · rename_i m hm
      split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i hs
            cases h
            have h1 : st.sessions.size = 1 := by simpa using hs
            have hn : m.name = name := by simpa using Array.find?_some hm
            rcases hss : st.sessions with ⟨l⟩
            rw [hss] at h1
            cases l with
            | nil => simp at h1
            | cons s t =>
              cases t with
              | nil => exact ⟨m, s, rfl, rfl, hn ▸ hm⟩
              | cons _ _ => simp at h1

/-- What the statements' fold holds, from the members `acc` held before it: one member pushed a statement, the members
before kept, and each statement's session one and its member the parameters' first under its name. -/
theorem heldFold {ms : Array Member} :
    ∀ (l : List StatementFiles) (acc held : Array Member), l.foldlM (heldStep ms) acc = .ok held →
      held.size = acc.size + l.length ∧ (∀ k, k < acc.size → held[k]? = acc[k]?) ∧
        ∀ (k : Nat) (st : StatementFiles) (m : Member), l[k]? = some st → held[acc.size + k]? = some m →
          (∃ s, st.sessions = #[s]) ∧ ms.find? (·.name == m.name) = some m
  | [], acc, held, h => by
    cases h
    exact ⟨by simp, fun _ _ => rfl, fun k st m hk _ => nomatch hk⟩
  | st :: l, acc, held, h => by
    rw [List.foldlM_cons] at h
    cases h1 : heldStep ms acc st with
    | error e => rw [h1] at h; cases h
    | ok acc' =>
      rw [h1, ok_bind] at h
      obtain ⟨m, s, rfl, hs, hf⟩ := heldStep_ok h1
      obtain ⟨hsz, hpre, hrest⟩ := heldFold l _ held h
      refine ⟨by rw [hsz, Array.size_push, List.length_cons]; omega, fun k hk => ?_, fun k st' m' hk hm' => ?_⟩
      · rw [hpre k (by rw [Array.size_push]; omega), Array.getElem?_push_lt hk, Array.getElem?_eq_getElem hk]
      · cases k with
        | zero =>
          have he : held[acc.size]? = some m := by
            rw [hpre acc.size (by rw [Array.size_push]; omega), Array.getElem?_push_size]
          rw [Nat.add_zero, he] at hm'
          cases hm'
          cases hk
          exact ⟨⟨s, hs⟩, hf⟩
        | succ k =>
          exact hrest k st' m' hk (by rwa [Array.size_push, show acc.size + 1 + k = acc.size + (k + 1) by omega])

/-- **What `membersHeld` holds**: a member a statement, by place, each statement's session one and its member the
parameters' first under its name. -/
theorem membersHeld_ok {ms : Array Member} {sts : Array StatementFiles} {held : Array Member}
    (h : membersHeld ms sts = .ok held) :
    held.size = sts.size ∧ ∀ (k : Nat) (st : StatementFiles) (m : Member), sts[k]? = some st → held[k]? = some m →
      (∃ s, st.sessions = #[s]) ∧ ms.find? (·.name == m.name) = some m := by
  unfold membersHeld at h
  rw [← Array.foldlM_toList] at h
  obtain ⟨hsz, _, hk⟩ := heldFold sts.toList #[] held h
  refine ⟨by rw [hsz]; simp, fun k st m hst hm => hk k st m (by rwa [Array.getElem?_toList]) (by simpa using hm)⟩

/-- **The finish's run on what was kept is `membersRun` on all the blobs**, where the stream kept every member's files
and every statement's session as they read on them. -/
theorem runKept_eq {p : Params} {c : OneStage.Commitment} {e : Evidence} {ms : Array Member}
    {sts : Array StatementFiles} {S : State} {full : Core.Blobs}
    (hF : FilesKept (Ctx.of p c e ms sts) full S) (hS : SessionsKept (Ctx.of p c e ms sts) full S) :
    runKept (Ctx.of p c e ms sts) S = membersRun p ms c e sts full := by
  unfold runKept structureKept membersRun membersStructure runMembers
  cases hh : membersHeld ms sts with
  | error err => simp only [Ctx.of, hh]; rfl
  | ok held =>
    obtain ⟨_, hone⟩ := membersHeld_ok hh
    have hk : ∀ (k : Nat) (st : StatementFiles) (m : Member), (sts.zip held)[k]? = some (st, m) →
        ∃ s a, st.sessions = #[s] ∧ S.sessions[k]? = some (some (SessionKept.of m a full s)) := by
      intro k st m hz
      obtain ⟨hst, hm⟩ := Array.getElem?_zip_eq_some.1 hz
      obtain ⟨⟨s, hs⟩, _⟩ := hone k st m hst hm
      exact ⟨s, _, hs, hS held hh k st m s hst hm (by simp [hs])⟩
    have ht : ∀ ts log, treesKept ts log (sts.zip held).size S = membersTreesHeld ts log (sts.zip held) full :=
      fun _ _ => treesKept_eq hk
    simp only [Ctx.of, hh, ok_bind, ht]
    congr 1
    funext _
    have hfl := files_toList hF
    cases hr : rootsOf ms c.record with
    | error err => rfl
    | ok roots =>
      simp only [Ctx.of, hr, Except.toOption, Option.getD] at hfl hS
      simp only [ok_bind]
      cases sampledHeld held e.base.provedUnits with
      | error err => rfl
      | ok u =>
        simp only [ok_bind]
        rw [hfl, files_forM]
        cases hf : membersFiles (fun m => m.slice c.record roots) full ms.toList with
        | error err => rfl
        | ok files =>
          simp only [ok_bind]
          have hfold : Array.foldlM (stepKept S) ⟨#[], #[], #[], true⟩ (sts.zip held).zipIdx =
              Array.foldlM (stepMember (statementAt files) full) ⟨#[], #[], #[], true⟩ (sts.zip held) := by
            rw [← Array.foldlM_toList, ← Array.foldlM_toList, Array.toList_zipIdx]
            refine foldlM_zipIdx _ 0 _ fun k x r hx => ?_
            obtain ⟨st, m⟩ := x
            rw [Nat.zero_add]
            have hz : (sts.zip held)[k]? = some (st, m) := by rwa [Array.getElem?_toList] at hx
            obtain ⟨hst, hm⟩ := Array.getElem?_zip_eq_some.1 hz
            obtain ⟨⟨s, hs⟩, hfind⟩ := hone k st m hst hm
            have hsk := hS held hh k st m s hst hm (by simp [hs])
            obtain ⟨a, ha, hsa⟩ := statementAt_of hf (by rwa [Array.find?_toList])
            exact stepKept_eq hs hsk ha hsa
          rw [hfold]
          rfl

/-- **The fold's half of `StreamSound`**: on a feed in the per-member stream's order, its state after every step
keeps every member's files and every statement's session as they read on all the blobs fed. Stated here, and proved
in `Spot.OneStageZkStreamKept` (`stepsKept`). -/
def StepsKept : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) ms sts plan (bs : List ByteArray),
    (membersStream p c e ms sts).plan = .ok plan → (membersStream p c e ms sts).Fed plan bs →
      FilesKept (Ctx.of p c e ms sts) (Core.Blobs.ofArray bs.toArray) ((membersStream p c e ms sts).steps bs) ∧
        SessionsKept (Ctx.of p c e ms sts) (Core.Blobs.ofArray bs.toArray) ((membersStream p c e ms sts).steps bs)

/-- **The finish's half of `StreamSound`**: on a state that keeps every member's files and every statement's session
as they read on the blobs `full`, the finish is `checkBlobs` on them. -/
def FinishKept : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) ms sts (S : State) (full : Core.Blobs),
    p.members = some ms → e.statements = some sts →
      FilesKept (Ctx.of p c e ms sts) full S → SessionsKept (Ctx.of p c e ms sts) full S →
        finish (Ctx.of p c e ms sts) S = checkBlobs p c e full

/-- **The finish's half holds** (`runKept_eq`): the finish reads what was kept as `checkMembersAt` reads the blobs. -/
theorem finishKept : FinishKept := by
  intro p c e ms sts S full hm he hF hS
  unfold finish
  simp only [runKept_eq hF hS]
  simp only [checkBlobs, checkBlobsAt, hm, he, checkMembersAt]
  rfl

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

/-- **`StreamSound` rests on the fold's half alone**, the finish's being proved (`finishKept`). -/
theorem streamSound_of_steps (hs : StepsKept) : StreamSound := streamSound_of hs finishKept

end Spot.OneStageZk.Streamed
