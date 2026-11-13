/-
`StepsKept`, the fold's half of `StreamSound`, proved, so `StreamSound` holds with no hypothesis (`streamSound`).

On a feed in the per-member stream's order, the state after every step keeps every member's files and every
statement's session as they read on all the blobs fed. The proof, in the feed's order:

* `find?_subset`, `alike_of_mem`, `tablesOf_held`: the blobs held answer as all the blobs fed do. Both begin with
  the tables fed, the planned blobs held are among the planned blobs fed, whose SHA-512 are distinct and none a
  pinned table's, so a search needs no order of the blobs held;
* `membersHeld_named`, `stOfHeld_some`, `stOfHeld_of`: the statements' members, each the first under its name, their
  names distinct, and the place each name is kept under;
* `groupsOf_mem`, `groupsOf_covers`: `groupsOf`'s fold over a hash map, read as it is and replaced by nothing: each
  place in a group is a member naming the group's circuit file, and every member is in a group;
* `memberPlan_ok`, `groupPlan_ok`, `planOf_ok`: the plan's shape, a circuit file and then its members' blobs;
* `plan_named`: a statement's member is the one member the plan reads under its name, since a second would plan the
  session's record twice, which `distinctFrom` refuses;
* `tables_fold`, `close_fold`, `session_fold`, `member_fold`, `members_fold`, `group_fold`, `plan_fold`: the steps,
  each on the roles still to come (`Pos`), keeping `Inv`;
* `stepsKept`, `streamSound`.
-/
module

public import Spot.OneStageZkStreamSound

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk.Streamed

open Lean (Json)
open Flock (V)

/-! ## Searches of the blobs held -/

/-- A search of `sub` finds what a search of `r` finds, where every entry of `sub` is one of `r`, at most one entry of
`r` matches and every entry of `r` that matches is in `sub`. -/
theorem find?_subset {α : Type} {p : α → Bool} {sub r : List α} (hs : ∀ x ∈ sub, x ∈ r)
    (hu : ∀ x ∈ r, ∀ y ∈ r, p x = true → p y = true → x = y) (hin : ∀ x ∈ r, p x = true → x ∈ sub) :
    sub.find? p = r.find? p := by
  cases h₂ : r.find? p with
  | none =>
    rw [List.find?_eq_none] at h₂ ⊢
    exact fun x hx => h₂ x (hs x hx)
  | some x =>
    have hx := List.mem_of_find?_eq_some h₂
    have hp := List.find?_some h₂
    cases h₁ : sub.find? p with
    | none => exact absurd hp (List.find?_eq_none.mp h₁ x (hin x hx hp))
    | some y =>
      exact congrArg some (hu y (hs y (List.mem_of_find?_eq_some h₁)) x hx (List.find?_some h₁) hp)

/-- A search of `t ++ sub` finds what a search of `t ++ r` finds, where every entry of `sub` is one of `r` and no entry
of `r` matches. -/
theorem find?_append_none {α : Type} {p : α → Bool} {t sub r : List α} (hs : ∀ x ∈ sub, x ∈ r)
    (hr : ∀ x ∈ r, p x = false) : (t ++ sub).find? p = (t ++ r).find? p := by
  have h1 : sub.find? p = none := List.find?_eq_none.2 fun x hx => by simp [hr x (hs x hx)]
  have h2 : r.find? p = none := List.find?_eq_none.2 fun x hx => by simp [hr x hx]
  rw [List.find?_append, List.find?_append, h1, h2]

/-- **The blobs held answer for a digest as all the blobs fed do**, where both begin with the tables `t`, the planned
blobs held (`sub`) are among the planned blobs fed (`r`), those have distinct SHA-512, and the blob under the digest
is held. -/
theorem alike_of_mem {held full : Core.Blobs} {t sub r : List (ByteArray × ByteArray)}
    (hh : held.entries.toList = t ++ sub) (hf : full.entries.toList = t ++ r) (hs : ∀ x ∈ sub, x ∈ r)
    (hn : (r.map (·.1)).Nodup) {D b : ByteArray} {d : String} (hm : (D, b) ∈ sub) (hd : Flock.hexOf D = d) :
    Alike held full d := by
  intro what
  unfold Core.Blobs.get
  cases hu : Flock.unhex d with
  | error e => rfl
  | ok d' =>
    show (if (d'.size != 64 || Flock.hexOf d' != d) = true then _ else _) =
      (if (d'.size != 64 || Flock.hexOf d' != d) = true then _ else _)
    by_cases hc : (d'.size != 64 || Flock.hexOf d' != d) = true
    · simp only [hc, ↓reduceIte]
      rfl
    · simp only [hc, Bool.false_eq_true, ↓reduceIte]
      have hd' : Flock.hexOf d' = d := by
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
        exact hc.2
      have hDd : d' = D := hexOf_inj (hd'.trans hd.symm)
      have hfind : held.entries.find? (·.1 == d') = full.entries.find? (·.1 == d') := by
        rw [← Array.find?_toList, ← Array.find?_toList, hh, hf, List.find?_append, List.find?_append]
        refine congrArg _ (find?_subset hs (fun x hx y hy hpx hpy => key_unique hn hx hy ?_) (fun x hx hpx => ?_))
        · exact (bytes_beq.mp hpx).trans (bytes_beq.mp hpy).symm
        · have hxe : x = (D, b) := key_unique hn hx (hs _ hm) ((bytes_beq.mp hpx).trans hDd)
          rw [hxe]
          exact hm
      rw [hfind]

/-- A digest whose hex a pinned table names is one of `pinned`. -/
theorem mem_pinned_of_hex {D : ByteArray} {t : Flock.MufuTable} (ht : t ∈ Flock.MUFU_TABLES)
    (h : Flock.hexOf D = t.sha512) : D ∈ pinned := by
  unfold pinned
  refine List.mem_filterMap.2 ⟨t, ht, ?_⟩
  rw [← h, Flock.Hex.unhex_hexOf]
  rfl

/-- **The pinned tables held are the pinned tables fed**, where both begin with the tables `t`, the planned blobs held
are among the planned blobs fed, and no planned blob's SHA-512 is a pinned table's. -/
theorem tablesOf_held {held full : Core.Blobs} {t sub r : List (ByteArray × ByteArray)}
    (hh : held.entries.toList = t ++ sub) (hf : full.entries.toList = t ++ r) (hs : ∀ x ∈ sub, x ∈ r)
    (hp : ∀ x ∈ r, x.1 ∉ pinned) : tablesOf held = tablesOf full := by
  unfold tablesOf
  refine filterMap_congr' fun tb htb => ?_
  rw [← Array.find?_toList, ← Array.find?_toList, hh, hf]
  refine congrArg _ (find?_append_none hs fun x hx => ?_)
  rcases x with ⟨dx, bx⟩
  dsimp only
  exact Bool.eq_false_iff.2 fun hpx => hp _ hx (mem_pinned_of_hex htb (beq_iff_eq.mp hpx))

/-! ## The statements' members -/

/-- What `heldStep` holds of a statement it takes: one member pushed, the statement's one session, the member the
statement names, the parameters' first under its name, and no member held before it of that name. -/
theorem heldStep_named {ms acc held : Array Member} {st : StatementFiles} (h : heldStep ms acc st = .ok held) :
    ∃ m s, held = acc.push m ∧ st.sessions = #[s] ∧ st.member = some m.name ∧
      ms.find? (·.name == m.name) = some m ∧ ∀ x ∈ acc, x.name ≠ m.name := by
  unfold heldStep at h
  split at h
  · cases h
  · rename_i name hname
    split at h
    · cases h
    · rename_i m hm
      split at h
      · cases h
      · rename_i hany
        split at h
        · cases h
        · split at h
          · cases h
          · rename_i hs
            cases h
            have h1 : st.sessions.size = 1 := by simpa using hs
            have hn : m.name = name := by simpa using Array.find?_some hm
            have hno : ∀ x ∈ acc, x.name ≠ m.name := by
              intro x hx he
              have : acc.any (·.name == name) = true :=
                Array.any_eq_true'.2 ⟨x, hx, by simp [he, hn]⟩
              rw [hany] at this
              cases this
            rcases hss : st.sessions with ⟨l⟩
            rw [hss] at h1
            cases l with
            | nil => simp at h1
            | cons s t =>
              cases t with
              | nil => exact ⟨m, s, rfl, rfl, by rw [hname, hn], hn ▸ hm, hno⟩
              | cons _ _ => simp at h1

/-- What the statements' fold holds, from the members `acc` held before it: a member a statement, the members before
kept, each statement naming its member, the parameters' first under its name, in one session, and the members' names
distinct where those before were. -/
theorem heldFold_named {ms : Array Member} :
    ∀ (l : List StatementFiles) (acc held : Array Member), l.foldlM (heldStep ms) acc = .ok held →
      held.size = acc.size + l.length ∧ (∀ k, k < acc.size → held[k]? = acc[k]?) ∧
        (∀ (k : Nat) (st : StatementFiles) (m : Member), l[k]? = some st → held[acc.size + k]? = some m →
          (∃ s, st.sessions = #[s]) ∧ st.member = some m.name ∧ ms.find? (·.name == m.name) = some m) ∧
        ((acc.toList.map (·.name)).Nodup → (held.toList.map (·.name)).Nodup)
  | [], acc, held, h => by
    cases h
    refine ⟨by simp, fun _ _ => rfl, fun k st m hk _ => ?_, fun hn => hn⟩
    simp at hk
  | st :: l, acc, held, h => by
    rw [List.foldlM_cons] at h
    cases h1 : heldStep ms acc st with
    | error e => rw [h1] at h; cases h
    | ok acc' =>
      rw [h1, ok_bind] at h
      obtain ⟨m, s, rfl, hs, hmem, hf, hno⟩ := heldStep_named h1
      obtain ⟨hsz, hpre, hrest, hnd⟩ := heldFold_named l _ held h
      refine ⟨by rw [hsz, Array.size_push, List.length_cons]; omega, fun k hk => ?_, fun k st' m' hk hm' => ?_,
        fun hacc => hnd ?_⟩
      · rw [hpre k (by rw [Array.size_push]; omega), Array.getElem?_push_lt hk, Array.getElem?_eq_getElem hk]
      · cases k with
        | zero =>
          have he : held[acc.size]? = some m := by
            rw [hpre acc.size (by rw [Array.size_push]; omega), Array.getElem?_push_size]
          rw [Nat.add_zero, he] at hm'
          cases hm'
          cases hk
          exact ⟨⟨s, hs⟩, hmem, hf⟩
        | succ k =>
          exact hrest k st' m' hk (by rwa [Array.size_push, show acc.size + 1 + k = acc.size + (k + 1) by omega])
      · rw [Array.toList_push, List.map_append, List.nodup_append]
        refine ⟨hacc, by simp, ?_⟩
        intro a ha b hb
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 ha
        simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
        rw [hb]
        exact hno x (Array.mem_toList_iff.mp hx)

/-- **What `membersHeld` holds**: a member a statement, by place, each statement naming its member, the parameters'
first under its name, in one session, and the members' names distinct. -/
theorem membersHeld_named {ms : Array Member} {sts : Array StatementFiles} {held : Array Member}
    (h : membersHeld ms sts = .ok held) :
    held.size = sts.size ∧
      (∀ (k : Nat) (st : StatementFiles) (m : Member), sts[k]? = some st → held[k]? = some m →
        (∃ s, st.sessions = #[s]) ∧ st.member = some m.name ∧ ms.find? (·.name == m.name) = some m) ∧
      (held.toList.map (·.name)).Nodup := by
  unfold membersHeld at h
  rw [← Array.foldlM_toList] at h
  obtain ⟨hsz, _, hk, hnd⟩ := heldFold_named sts.toList #[] held h
  exact ⟨by rw [hsz]; simp, fun k st m hst hm => hk k st m (by rwa [Array.getElem?_toList]) (by simpa using hm),
    hnd (by simp)⟩

/-- Two members held under one name are at one place. -/
theorem held_place {held : Array Member} (hnd : (held.toList.map (·.name)).Nodup) {k k' : Nat} {m m' : Member}
    (hk : held[k]? = some m) (hk' : held[k']? = some m') (hn : m.name = m'.name) : k = k' := by
  have h1 : (held.toList.map (·.name))[k]? = some m.name := by
    rw [List.getElem?_map, Array.getElem?_toList, hk]; rfl
  have h2 : (held.toList.map (·.name))[k']? = some m.name := by
    rw [List.getElem?_map, Array.getElem?_toList, hk', hn]; rfl
  have hlt : k < (held.toList.map (·.name)).length := (List.getElem?_eq_some_iff.1 h1).1
  exact (List.Nodup.getElem?_inj hlt hnd).1 (h1.trans h2.symm)

/-! ## The plan's map from a member's name to its statement -/

/-- The statement that names each member held, by the member's name: `planOf`'s map. -/
def stOfHeld (held : Array Member) : Std.HashMap String Nat :=
  held.zipIdx.foldl (fun acc (m, i) => acc.insert m.name i) {}

theorem stOf_fold_some : ∀ (l : List (Member × Nat)) (acc : Std.HashMap String Nat) (n : String) (k : Nat),
    (l.foldl (fun acc (x : Member × Nat) => acc.insert x.1.name x.2) acc)[n]? = some k →
      acc[n]? = some k ∨ ∃ m, (m, k) ∈ l ∧ m.name = n
  | [], _, _, _, h => .inl h
  | x :: l, acc, n, k, h => by
    rw [List.foldl_cons] at h
    rcases stOf_fold_some l _ n k h with h1 | ⟨m, hm, hn⟩
    · rw [Std.HashMap.getElem?_insert] at h1
      by_cases hx : x.1.name = n
      · simp only [hx, beq_self_eq_true, ite_true] at h1
        cases h1
        exact .inr ⟨x.1, List.mem_cons_self, hx⟩
      · simp only [beq_iff_eq, hx, ite_false] at h1
        exact .inl h1
    · exact .inr ⟨m, List.mem_cons_of_mem _ hm, hn⟩

theorem stOf_fold_of : ∀ (l : List (Member × Nat)) (acc : Std.HashMap String Nat) (n : String) (k : Nat),
    (∀ x ∈ l, x.1.name = n → x.2 = k) → (acc[n]? = some k ∨ ∃ m, (m, k) ∈ l ∧ m.name = n) →
      (l.foldl (fun acc (x : Member × Nat) => acc.insert x.1.name x.2) acc)[n]? = some k
  | [], _, _, _, _, h => by
    rcases h with h | ⟨m, hm, _⟩
    · exact h
    · cases hm
  | x :: l, acc, n, k, hall, h => by
    rw [List.foldl_cons]
    refine stOf_fold_of l _ n k (fun y hy => hall y (List.mem_cons_of_mem _ hy)) ?_
    rw [Std.HashMap.getElem?_insert]
    by_cases hx : x.1.name = n
    · simp only [hx, beq_self_eq_true, ite_true]
      exact .inl (congrArg some (hall x List.mem_cons_self hx))
    · simp only [beq_iff_eq, hx, ite_false]
      rcases h with h | ⟨m, hm, hn⟩
      · exact .inl h
      · rcases List.mem_cons.1 hm with he | hm
        · rw [← he] at hx
          exact absurd hn hx
        · exact .inr ⟨m, hm, hn⟩

theorem stOfHeld_eq (held : Array Member) :
    stOfHeld held = held.toList.zipIdx.foldl (fun acc (x : Member × Nat) => acc.insert x.1.name x.2) {} := by
  unfold stOfHeld
  rw [← Array.foldl_toList, Array.toList_zipIdx]

/-- A name the map gives a statement is the name of the member held at that statement's place. -/
theorem stOfHeld_some {held : Array Member} {n : String} {k : Nat} (h : (stOfHeld held)[n]? = some k) :
    ∃ m, held[k]? = some m ∧ m.name = n := by
  rw [stOfHeld_eq] at h
  rcases stOf_fold_some _ _ n k h with h | ⟨m, hm, hn⟩
  · simp at h
  · refine ⟨m, ?_, hn⟩
    have := List.mem_zipIdx_iff_getElem?.1 hm
    simpa using this

/-- The map gives the member held at a place that place, where the members' names are distinct. -/
theorem stOfHeld_of {held : Array Member} (hnd : (held.toList.map (·.name)).Nodup) {k : Nat} {m : Member}
    (hk : held[k]? = some m) : (stOfHeld held)[m.name]? = some k := by
  rw [stOfHeld_eq]
  refine stOf_fold_of _ _ m.name k (fun x hx hn => ?_) (.inr ⟨m, ?_, rfl⟩)
  · have hx' : held[x.2]? = some x.1 := by
      have := List.mem_zipIdx_iff_getElem?.1 hx
      simpa using this
    exact held_place hnd hx' hk hn
  · exact List.mem_zipIdx_iff_getElem?.2 (by simpa using hk)

/-! ## The plan's groups -/

/-- One member into `groupsOf`'s fold: its place joins its circuit file's group, a new one where none is open. -/
def groupStep (acc : Array String × Std.HashMap String (Array Nat)) (x : Member × Nat) :
    Array String × Std.HashMap String (Array Nat) :=
  match acc.2[x.1.own.circuit]? with
  | some is => (acc.1, acc.2.insert x.1.own.circuit (is.push x.2))
  | none => (acc.1.push x.1.own.circuit, acc.2.insert x.1.own.circuit #[x.2])

/-- What `groupsOf`'s fold keeps: every group is in the order, each place in it has `P`, and every member seen is in
its circuit file's group. -/
theorem groups_fold {P : String → Nat → Prop} :
    ∀ (l : List (Member × Nat)) (acc : Array String × Std.HashMap String (Array Nat)) (seen : List (Member × Nat)),
      (∀ d is, acc.2[d]? = some is → d ∈ acc.1 ∧ ∀ i ∈ is, P d i) →
      (∀ x ∈ seen, ∃ is, acc.2[x.1.own.circuit]? = some is ∧ x.2 ∈ is) →
      (∀ x ∈ l, P x.1.own.circuit x.2) →
      (∀ d is, (l.foldl groupStep acc).2[d]? = some is → d ∈ (l.foldl groupStep acc).1 ∧ ∀ i ∈ is, P d i) ∧
        ∀ x, x ∈ seen ∨ x ∈ l → ∃ is, (l.foldl groupStep acc).2[x.1.own.circuit]? = some is ∧ x.2 ∈ is
  | [], acc, seen, h1, h2, _ => ⟨h1, fun x hx => by
      rcases hx with hx | hx
      · exact h2 x hx
      · cases hx⟩
  | x :: l, acc, seen, h1, h2, hP => by
    rw [List.foldl_cons]
    have hstep : (∀ d is, (groupStep acc x).2[d]? = some is → d ∈ (groupStep acc x).1 ∧ ∀ i ∈ is, P d i) ∧
        ∀ y ∈ x :: seen, ∃ is, (groupStep acc x).2[y.1.own.circuit]? = some is ∧ y.2 ∈ is := by
      unfold groupStep
      cases hc : acc.2[x.1.own.circuit]? with
      | some is0 =>
        dsimp only
        refine ⟨fun d is hd => ?_, fun y hy => ?_⟩
        · rw [Std.HashMap.getElem?_insert] at hd
          by_cases he : x.1.own.circuit = d
          · simp only [he, beq_self_eq_true, ite_true] at hd
            cases hd
            obtain ⟨hmem, hall⟩ := h1 d is0 (he ▸ hc)
            refine ⟨hmem, fun i hi => ?_⟩
            rcases Array.mem_push.1 hi with hi | hi
            · exact hall i hi
            · rw [hi, ← he]
              exact hP x List.mem_cons_self
          · simp only [beq_iff_eq, he, ite_false] at hd
            exact h1 d is hd
        · rw [Std.HashMap.getElem?_insert]
          rcases List.mem_cons.1 hy with rfl | hy
          · simp only [beq_self_eq_true, ite_true]
            exact ⟨_, rfl, Array.mem_push_self⟩
          · obtain ⟨isy, hisy, hmem⟩ := h2 y hy
            by_cases he : x.1.own.circuit = y.1.own.circuit
            · simp only [he, beq_self_eq_true, ite_true]
              rw [he, hisy] at hc
              cases hc
              exact ⟨_, rfl, Array.mem_push_of_mem _ hmem⟩
            · simp only [beq_iff_eq, he, ite_false]
              exact ⟨isy, hisy, hmem⟩
      | none =>
        dsimp only
        refine ⟨fun d is hd => ?_, fun y hy => ?_⟩
        · rw [Std.HashMap.getElem?_insert] at hd
          by_cases he : x.1.own.circuit = d
          · simp only [he, beq_self_eq_true, ite_true] at hd
            cases hd
            refine ⟨he ▸ Array.mem_push_self, fun i hi => ?_⟩
            have hi' : i = x.2 := by simpa using hi
            rw [hi', ← he]
            exact hP x List.mem_cons_self
          · simp only [beq_iff_eq, he, ite_false] at hd
            obtain ⟨hmem, hall⟩ := h1 d is hd
            exact ⟨Array.mem_push_of_mem _ hmem, hall⟩
        · rw [Std.HashMap.getElem?_insert]
          rcases List.mem_cons.1 hy with rfl | hy
          · simp only [beq_self_eq_true, ite_true]
            exact ⟨_, rfl, by simp⟩
          · obtain ⟨isy, hisy, hmem⟩ := h2 y hy
            by_cases he : x.1.own.circuit = y.1.own.circuit
            · rw [he, hisy] at hc
              cases hc
            · simp only [beq_iff_eq, he, ite_false]
              exact ⟨isy, hisy, hmem⟩
    obtain ⟨r1, r2⟩ := groups_fold l (groupStep acc x) (x :: seen) hstep.1 hstep.2
      (fun y hy => hP y (List.mem_cons_of_mem _ hy))
    refine ⟨r1, fun y hy => r2 y ?_⟩
    rcases hy with hy | hy
    · exact .inl (List.mem_cons_of_mem _ hy)
    · rcases List.mem_cons.1 hy with rfl | hy
      · exact .inl List.mem_cons_self
      · exact .inr hy

theorem groupsOf_eq (ms : Array Member) :
    groupsOf ms =
      (ms.toList.zipIdx.foldl groupStep (#[], {})).1.map fun d =>
        (d, (ms.toList.zipIdx.foldl groupStep (#[], {})).2.getD d #[]) := by
  unfold groupsOf
  rw [← Array.foldl_toList, Array.toList_zipIdx]
  rfl

/-- **Each place in a group of the plan is a member that names the group's circuit file.** -/
theorem groupsOf_mem {ms : Array Member} {g : String × Array Nat} (hg : g ∈ groupsOf ms) {i : Nat} (hi : i ∈ g.2) :
    ∃ m, ms[i]? = some m ∧ m.own.circuit = g.1 := by
  rw [groupsOf_eq] at hg
  obtain ⟨d, _, rfl⟩ := Array.mem_map.1 hg
  obtain ⟨h1, _⟩ := groups_fold (P := fun d i => ∃ m, ms[i]? = some m ∧ m.own.circuit = d) ms.toList.zipIdx (#[], {}) []
    (fun d is h => by simp at h) (fun x hx => by cases hx) (fun x hx => by
      have := List.mem_zipIdx_iff_getElem?.1 hx
      exact ⟨x.1, by simpa using this, rfl⟩)
  dsimp only at hi
  rw [Std.HashMap.getD_eq_getD_getElem?] at hi
  cases hd : (ms.toList.zipIdx.foldl groupStep (#[], {})).2[d]? with
  | none => rw [hd] at hi; simp at hi
  | some is =>
    rw [hd] at hi
    exact (h1 d is hd).2 i hi

/-- **Every member is in a group of the plan.** -/
theorem groupsOf_covers {ms : Array Member} {i : Nat} {m : Member} (hm : ms[i]? = some m) :
    ∃ g ∈ groupsOf ms, i ∈ g.2 := by
  obtain ⟨h1, h2⟩ := groups_fold (P := fun _ _ => True) ms.toList.zipIdx (#[], {}) []
    (fun d is h => by simp at h) (fun x hx => by cases hx) (fun _ _ => trivial)
  obtain ⟨is, his, hmem⟩ := h2 (m, i) (.inr (List.mem_zipIdx_iff_getElem?.2 (by simpa using hm)))
  refine ⟨(m.own.circuit, (ms.toList.zipIdx.foldl groupStep (#[], {})).2.getD m.own.circuit #[]), ?_, ?_⟩
  · rw [groupsOf_eq]
    exact Array.mem_map.2 ⟨m.own.circuit, (h1 _ is his).1, rfl⟩
  · dsimp only
    rw [Std.HashMap.getD_eq_getD_getElem?, his]
    exact hmem

/-! ## The plan's shape -/

/-- What a pass of `mapM` returns: each entry's value, in order. -/
theorem listMapM_ok {α β : Type} {f : α → V β} (d : β) : ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys →
    ys = l.map (fun x => (f x).toOption.getD d) ∧ ∀ x ∈ l, f x = .ok ((f x).toOption.getD d)
  | [], ys, h => by
    cases h
    exact ⟨rfl, fun _ hx => nomatch hx⟩
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok b =>
      rw [ha, ok_bind] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok bs =>
        rw [hl, ok_bind] at h
        cases h
        obtain ⟨h1, h2⟩ := listMapM_ok d hl
        refine ⟨by rw [List.map_cons, ha, h1]; rfl, fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · rw [ha]; rfl
        · exact h2 x hx

theorem arrayMapM_ok {α β : Type} {f : α → V β} (d : β) {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    ys.toList = xs.toList.map (fun x => (f x).toOption.getD d) ∧
      ∀ x ∈ xs.toList, f x = .ok ((f x).toOption.getD d) := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    cases h
    exact listMapM_ok d hl

/-- A digest read off its hex is the digest the hex names. -/
theorem digestBytes_ok {d what : String} {D : ByteArray} (h : digestBytes d what = .ok D) : Flock.hexOf D = d := by
  unfold digestBytes at h
  cases hu : Flock.unhex d with
  | error e => rw [hu] at h; cases h
  | ok b =>
    rw [hu, ok_bind] at h
    split at h
    · cases h
    · rename_i hc
      cases h
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
      exact hc.2

theorem blob_ok {d what : String} {r : Role} {y : ByteArray × Role} (h : blob d what r = .ok y) :
    digestBytes d what = .ok y.1 ∧ y.2 = r := by
  unfold blob at h
  cases hd : digestBytes d what with
  | error e => rw [hd] at h; cases h
  | ok D =>
    rw [hd] at h
    cases h
    exact ⟨rfl, rfl⟩

/-- A member's planned blobs as a list: `memberPlan`'s, none where it refuses. -/
def mpL (ms : Array Member) (sts : Array StatementFiles) (stOf : Std.HashMap String Nat) (i : Nat) :
    List (ByteArray × Role) :=
  ((memberPlan ms sts stOf i).toOption.getD #[]).toList

/-- A group's planned blobs as a list: `groupPlan`'s, none where it refuses. -/
def gpL (ms : Array Member) (sts : Array StatementFiles) (stOf : Std.HashMap String Nat) (g : String × Array Nat) :
    List (ByteArray × Role) :=
  ((groupPlan ms sts stOf g).toOption.getD #[]).toList

/-- **A member's planned blobs**: its public file, and where a statement names it, that statement's session's record
and rep files, each rep file of the session under one of them. -/
theorem memberPlan_ok {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat} {i : Nat}
    {seg : Array (ByteArray × Role)} (h : memberPlan ms sts stOf i = .ok seg) :
    ∃ m Dp, ms[i]? = some m ∧ Flock.hexOf Dp = m.own.pub ∧
      ((stOf[m.name]? = none ∧ seg.toList = [(Dp, .pub i none)]) ∨
        ∃ k s Dr reps, stOf[m.name]? = some k ∧ sts[k]?.bind (·.sessions[0]?) = some s ∧
          Flock.hexOf Dr = s.record ∧ seg.toList = (Dp, .pub i (some k)) :: (Dr, .record k) :: reps ∧
          (∀ e ∈ reps, ∃ j, e.2 = .rep k j) ∧ ∀ x ∈ s.reps, ∃ e ∈ reps, Flock.hexOf e.1 = x.2) := by
  unfold memberPlan at h
  cases hm : ms[i]? with
  | none => rw [hm] at h; cases h
  | some m =>
    rw [hm] at h
    dsimp only at h
    cases hp : blob m.own.pub s!"member {m.name}'s public file" (.pub i stOf[m.name]?) with
    | error e => rw [hp] at h; cases h
    | ok pub =>
      rw [hp, ok_bind] at h
      obtain ⟨hpd, hpr⟩ := blob_ok hp
      refine ⟨m, pub.1, rfl, digestBytes_ok hpd, ?_⟩
      cases hst : stOf[m.name]? with
      | none =>
        rw [hst] at h hpr
        cases h
        exact .inl ⟨rfl, by rw [← hpr]⟩
      | some k =>
        rw [hst] at h hpr
        dsimp only at h
        cases hs : sts[k]?.bind (·.sessions[0]?) with
        | none => rw [hs] at h; cases h
        | some s =>
          rw [hs] at h
          dsimp only at h
          cases hr : blob s.record s!"session {s.name}'s record" (.record k) with
          | error e => rw [hr] at h; cases h
          | ok record =>
            rw [hr, ok_bind] at h
            obtain ⟨hrd, hrr⟩ := blob_ok hr
            cases hreps : s.reps.zipIdx.mapM
                (fun x : (String × String) × Nat => blob x.1.2 s!"session {s.name}'s files: {x.1.1}" (.rep k x.2)) with
            | error e =>
              have : s.reps.zipIdx.mapM (fun x : (String × String) × Nat =>
                  match x with | ((f, rd), j) => blob rd s!"session {s.name}'s files: {f}" (Role.rep k j)) =
                    .error e := hreps
              rw [this] at h
              cases h
            | ok reps =>
              have : s.reps.zipIdx.mapM (fun x : (String × String) × Nat =>
                  match x with | ((f, rd), j) => blob rd s!"session {s.name}'s files: {f}" (Role.rep k j)) =
                    .ok reps := hreps
              rw [this, ok_bind] at h
              cases h
              obtain ⟨hl, hall⟩ := arrayMapM_ok (default : ByteArray × Role) hreps
              refine .inr ⟨k, s, record.1, reps.toList, rfl, hs, digestBytes_ok hrd, ?_, ?_, ?_⟩
              · rw [← hpr, ← hrr]
                simp
              · intro e he
                rw [hl] at he
                obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he
                have := hall x hx
                exact ⟨x.2, (blob_ok this).2⟩
              · intro x hx
                obtain ⟨j, hj⟩ := List.getElem_of_mem (Array.mem_toList_iff.2 hx)
                obtain ⟨hj, hxj⟩ := hj
                have hz : (x, j) ∈ s.reps.zipIdx.toList := by
                  rw [Array.toList_zipIdx]
                  exact List.mem_zipIdx_iff_getElem?.2 (List.getElem?_eq_some_iff.2 ⟨hj, hxj⟩)
                have hb := hall (x, j) hz
                refine ⟨_, ?_, digestBytes_ok (blob_ok hb).1⟩
                rw [hl]
                exact List.mem_map.2 ⟨(x, j), hz, rfl⟩

/-- **A group's planned blobs**: its circuit file, then each of its members' blobs, in order. -/
theorem groupPlan_ok {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat}
    {g : String × Array Nat} {seg : Array (ByteArray × Role)} (h : groupPlan ms sts stOf g = .ok seg) :
    ∃ D, Flock.hexOf D = g.1 ∧ seg.toList = (D, .circuit g.1) :: g.2.toList.flatMap (mpL ms sts stOf) ∧
      ∀ i ∈ g.2.toList, ∃ x, memberPlan ms sts stOf i = .ok x := by
  unfold groupPlan at h
  cases hc : blob g.1 "circuit" (.circuit g.1) with
  | error e => rw [hc] at h; cases h
  | ok c =>
    rw [hc, ok_bind] at h
    obtain ⟨hcd, hcr⟩ := blob_ok hc
    cases hm : g.2.mapM (memberPlan ms sts stOf) with
    | error e => rw [hm] at h; cases h
    | ok members =>
      rw [hm, ok_bind] at h
      cases h
      obtain ⟨hl, hall⟩ := arrayMapM_ok (#[] : Array (ByteArray × Role)) hm
      refine ⟨c.1, digestBytes_ok hcd, ?_, fun i hi => ⟨_, hall i hi⟩⟩
      rw [← hcr]
      simp only [Array.toList_append, Array.toList_flatten, hl, List.map_map]
      rfl

theorem mpL_of_ok {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat} {i : Nat}
    {seg : Array (ByteArray × Role)} (h : memberPlan ms sts stOf i = .ok seg) : mpL ms sts stOf i = seg.toList := by
  unfold mpL
  rw [h]
  rfl

theorem gpL_of_ok {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat}
    {g : String × Array Nat} {seg : Array (ByteArray × Role)} (h : groupPlan ms sts stOf g = .ok seg) :
    gpL ms sts stOf g = seg.toList := by
  unfold gpL
  rw [h]
  rfl

/-- **The plan's shape**: its statements' members are found, each group's blobs are planned, the plan is the groups'
blobs in order, and its SHA-512 are distinct and none a pinned table's. -/
theorem planOf_ok {ms : Array Member} {sts : Array StatementFiles} {P : Array (ByteArray × Role)}
    (h : planOf ms sts = .ok P) :
    ∃ held, membersHeld ms sts = .ok held ∧
      P.toList = (groupsOf ms).toList.flatMap (gpL ms sts (stOfHeld held)) ∧
      (∀ g ∈ (groupsOf ms).toList, ∃ seg, groupPlan ms sts (stOfHeld held) g = .ok seg) ∧
      (P.toList.map fun e => Flock.hexOf e.1).Nodup ∧ ∀ e ∈ P.toList, e.1 ∉ pinned := by
  unfold planOf at h
  cases hh : membersHeld ms sts with
  | error e => rw [hh] at h; cases h
  | ok held =>
    rw [hh, ok_bind] at h
    have h' : ((groupsOf ms).mapM (groupPlan ms sts (stOfHeld held)) >>= fun segs =>
        distinctFrom pinned {} (segs.flatten.toList.map (·.1)) >>= fun _ => pure segs.flatten) = .ok P := h
    cases hg : (groupsOf ms).mapM (groupPlan ms sts (stOfHeld held)) with
    | error e => rw [hg] at h'; cases h'
    | ok segs =>
      rw [hg, ok_bind] at h'
      cases hd : distinctFrom pinned {} (segs.flatten.toList.map (·.1)) with
      | error e => rw [hd] at h'; cases h'
      | ok u =>
        rw [hd, ok_bind] at h'
        cases h'
        obtain ⟨hl, hall⟩ := arrayMapM_ok (#[] : Array (ByteArray × Role)) hg
        obtain ⟨_, hnd, hpin⟩ := distinctFrom_ok hd
        refine ⟨held, rfl, ?_, fun g hg' => ⟨_, hall g hg'⟩, ?_, fun e he => hpin e.1 (List.mem_map_of_mem he)⟩
        · rw [Array.toList_flatten, hl, List.map_map, List.flatMap]
          rfl
        · rw [List.map_map] at hnd
          exact hnd

/-- Two places of a list whose entries' lists share an element are one place, where the lists joined repeat none. -/
theorem nodup_flatMap {α β : Type} {f : α → List β} : ∀ {l : List α}, (l.flatMap f).Nodup →
    ∀ {i j : Nat} {a b : α} {x : β}, l[i]? = some a → l[j]? = some b → x ∈ f a → x ∈ f b → i = j
  | [], _, _, _, _, _, _, hi, _, _, _ => by simp at hi
  | a0 :: l, hn, i, j, a, b, x, hi, hj, hxa, hxb => by
    rw [List.flatMap_cons, List.nodup_append] at hn
    obtain ⟨_, hn2, hdis⟩ := hn
    cases i with
    | zero =>
      cases j with
      | zero => rfl
      | succ j =>
        rw [List.getElem?_cons_zero] at hi
        cases hi
        rw [List.getElem?_cons_succ] at hj
        exact absurd rfl (hdis x hxa x (List.mem_flatMap.2 ⟨b, List.mem_of_getElem? hj, hxb⟩))
    | succ i =>
      cases j with
      | zero =>
        rw [List.getElem?_cons_zero] at hj
        cases hj
        rw [List.getElem?_cons_succ] at hi
        exact absurd rfl (hdis x hxb x (List.mem_flatMap.2 ⟨a, List.mem_of_getElem? hi, hxa⟩))
      | succ j =>
        rw [List.getElem?_cons_succ] at hi hj
        rw [nodup_flatMap hn2 hi hj hxa hxb]

/-- One entry's list repeats none, where the lists joined repeat none. -/
theorem nodup_flatMap_mem {α β : Type} {f : α → List β} : ∀ {l : List α}, (l.flatMap f).Nodup →
    ∀ {a : α}, a ∈ l → (f a).Nodup
  | [], _, _, ha => nomatch ha
  | a0 :: l, hn, a, ha => by
    rw [List.flatMap_cons, List.nodup_append] at hn
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hn.1
    · exact nodup_flatMap_mem hn.2.1 ha

/-- **A statement's member is the one member the plan reads under its name**: where the plan passes, a member whose
name a statement's place is kept under is the member `closeSession` finds for that statement. A second member under
the name would plan the session's record twice, which `distinctFrom` refuses. -/
theorem plan_named {ms : Array Member} {sts : Array StatementFiles} {P : Array (ByteArray × Role)}
    {held : Array Member} (hh : membersHeld ms sts = .ok held)
    (hP : P.toList = (groupsOf ms).toList.flatMap (gpL ms sts (stOfHeld held)))
    (hg : ∀ g ∈ (groupsOf ms).toList, ∃ seg, groupPlan ms sts (stOfHeld held) g = .ok seg)
    (hnd : (P.toList.map fun e => Flock.hexOf e.1).Nodup) {i : Nat} {m : Member} {k : Nat}
    (hm : ms[i]? = some m) (hk : (stOfHeld held)[m.name]? = some k) :
    held[k]? = some m ∧ (sts[k]?.bind (·.member)).bind (fun n => ms.find? (·.name == n)) = some m := by
  obtain ⟨hsize, hnamed, _⟩ := membersHeld_named hh
  obtain ⟨m', hk', hname⟩ := stOfHeld_some hk
  have hklt : k < sts.size := by
    have := (Array.getElem?_eq_some_iff.1 hk').1
    omega
  obtain ⟨_, hmem, hfind⟩ := hnamed k sts[k] m' (Array.getElem?_eq_getElem hklt) hk'
  suffices hmm : m' = m by
    subst hmm
    refine ⟨hk', ?_⟩
    rw [Array.getElem?_eq_getElem hklt]
    simp only [Option.bind_some, hmem, hfind]
  -- the two members' places, and their groups
  obtain ⟨i', hi'lt, hi'⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hfind)
  have hm' : ms[i']? = some m' := by rw [Array.getElem?_eq_getElem hi'lt, hi']
  obtain ⟨g, hgm, hig⟩ := groupsOf_covers hm
  obtain ⟨g', hgm', hig'⟩ := groupsOf_covers hm'
  obtain ⟨seg, hseg⟩ := hg g (Array.mem_toList_iff.2 hgm)
  obtain ⟨seg', hseg'⟩ := hg g' (Array.mem_toList_iff.2 hgm')
  obtain ⟨D, _, hsegl, hmp⟩ := groupPlan_ok hseg
  obtain ⟨D', _, hsegl', hmp'⟩ := groupPlan_ok hseg'
  obtain ⟨x, hx⟩ := hmp i (Array.mem_toList_iff.2 hig)
  obtain ⟨x', hx'⟩ := hmp' i' (Array.mem_toList_iff.2 hig')
  -- each plans the record of the statement's session
  have hrec : ∀ {i : Nat} {m : Member} {x : Array (ByteArray × Role)}, ms[i]? = some m →
      (stOfHeld held)[m.name]? = some k → memberPlan ms sts (stOfHeld held) i = .ok x →
        ∀ s, sts[k]?.bind (·.sessions[0]?) = some s → ∃ Dr, Flock.hexOf Dr = s.record ∧ Dr ∈ x.toList.map (·.1) := by
    intro i m x hm hk hx s hs
    obtain ⟨m₁, Dp, hm₁, _, hcase⟩ := memberPlan_ok hx
    rw [hm] at hm₁
    cases hm₁
    rcases hcase with ⟨hnone, _⟩ | ⟨k₁, s₁, Dr, reps, hk₁, hs₁, hDr, hxl, _⟩
    · rw [hk] at hnone
      cases hnone
    · rw [hk] at hk₁
      cases hk₁
      rw [hs] at hs₁
      cases hs₁
      exact ⟨Dr, hDr, by rw [hxl]; simp⟩
  have hk2 : (stOfHeld held)[m'.name]? = some k := by rw [hname]; exact hk
  cases hs : sts[k]?.bind (·.sessions[0]?) with
  | none =>
    obtain ⟨m₁, Dp, hm₁, _, hcase⟩ := memberPlan_ok hx
    rw [hm] at hm₁
    cases hm₁
    rcases hcase with ⟨hnone, _⟩ | ⟨k₁, s₁, _, _, hk₁, hs₁, _⟩
    · rw [hk] at hnone
      cases hnone
    · rw [hk] at hk₁
      cases hk₁
      rw [hs] at hs₁
      cases hs₁
  | some s =>
    obtain ⟨Dr, hDr, hin⟩ := hrec hm hk hx s hs
    obtain ⟨Dr', hDr', hin'⟩ := hrec hm' hk2 hx' s hs
    have hDD : Dr' = Dr := hexOf_inj (hDr'.trans hDr.symm)
    subst hDD
    -- the record's hex is in both members' planned blobs
    have hxi : s.record ∈ (mpL ms sts (stOfHeld held) i).map fun e => Flock.hexOf e.1 := by
      rw [mpL_of_ok hx, ← hDr]
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 hin
      exact List.mem_map.2 ⟨e, he, rfl⟩
    have hxi' : s.record ∈ (mpL ms sts (stOfHeld held) i').map fun e => Flock.hexOf e.1 := by
      rw [mpL_of_ok hx', ← hDr]
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 hin'
      exact List.mem_map.2 ⟨e, he, rfl⟩
    -- so in both groups' planned blobs
    have hgi : ∀ {g : String × Array Nat} {seg : Array (ByteArray × Role)} {D : ByteArray} {i : Nat},
        gpL ms sts (stOfHeld held) g = (D, Role.circuit g.1) :: g.2.toList.flatMap (mpL ms sts (stOfHeld held)) →
        i ∈ g.2 → s.record ∈ (mpL ms sts (stOfHeld held) i).map (fun e => Flock.hexOf e.1) →
          s.record ∈ (gpL ms sts (stOfHeld held) g).map fun e => Flock.hexOf e.1 := by
      intro g seg D i hgl hi hr
      rw [hgl, List.map_cons, List.map_flatMap]
      exact List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨i, Array.mem_toList_iff.2 hi, hr⟩)
    have hgl : gpL ms sts (stOfHeld held) g =
        (D, Role.circuit g.1) :: g.2.toList.flatMap (mpL ms sts (stOfHeld held)) := by rw [gpL_of_ok hseg, hsegl]
    have hgl' : gpL ms sts (stOfHeld held) g' =
        (D', Role.circuit g'.1) :: g'.2.toList.flatMap (mpL ms sts (stOfHeld held)) := by rw [gpL_of_ok hseg', hsegl']
    rw [hP, List.map_flatMap] at hnd
    obtain ⟨pg, hpg⟩ := List.getElem?_of_mem (Array.mem_toList_iff.2 hgm)
    obtain ⟨pg', hpg'⟩ := List.getElem?_of_mem (Array.mem_toList_iff.2 hgm')
    have hpp := nodup_flatMap hnd hpg hpg' (hgi (seg := seg) hgl hig hxi) (hgi (seg := seg') hgl' hig' hxi')
    subst hpp
    rw [hpg] at hpg'
    cases hpg'
    -- and in one group, at one place
    have hng := nodup_flatMap_mem hnd (Array.mem_toList_iff.2 hgm)
    rw [hgl, List.map_cons, List.map_flatMap, List.nodup_cons] at hng
    obtain ⟨p, hp⟩ := List.getElem?_of_mem (Array.mem_toList_iff.2 hig)
    obtain ⟨p', hp'⟩ := List.getElem?_of_mem (Array.mem_toList_iff.2 hig')
    have hpp := nodup_flatMap hng.2 hp hp' hxi hxi'
    subst hpp
    rw [hp] at hp'
    cases hp'
    rw [hm] at hm'
    cases hm'
    rfl

/-! ## The stream's steps -/

/-- A blob under its SHA-512, as the blobs fed hold it. -/
abbrev entry (b : ByteArray) : ByteArray × ByteArray := (Flock.Sha512.hash b, b)

/-- The stream's step on a blob fed. -/
abbrev feed1 (C : Ctx) (S : State) (b : ByteArray) : State := step C S (Flock.Sha512.hash b) b

theorem not_pinned {h : ByteArray} (hp : h ∉ pinned) : pinned.any (fun t => decide (t = h)) = false := by
  rw [List.any_eq_false]
  intro t ht
  simp only [decide_eq_true_eq]
  rintro rfl
  exact hp ht

theorem is_pinned {h : ByteArray} (hp : h ∈ pinned) : pinned.any (fun t => decide (t = h)) = true :=
  List.any_eq_true.2 ⟨h, hp, by simp⟩

/-- **The tables' steps**: before any planned blob, each pinned table fed joins the tables held. -/
theorem tables_fold (C : Ctx) : ∀ (pre : List ByteArray) (S : State), S.i = 0 →
    (∀ b ∈ pre, Flock.Sha512.hash b ∈ pinned) →
    pre.foldl (feed1 C) S = { S with tables := S.tables ++ (pre.map entry).toArray }
  | [], S, _, _ => by simp
  | b :: pre, S, hi, hp => by
    rw [List.foldl_cons]
    have hstep : feed1 C S b = { S with tables := S.tables.push (entry b) } := by
      unfold feed1 step
      simp only [hi, is_pinned (hp b List.mem_cons_self), and_self, ↓reduceIte]
    rw [hstep, tables_fold C pre { S with tables := S.tables.push (entry b) } hi
      (fun b hb => hp b (List.mem_cons_of_mem _ hb))]
    simp

/-- **Where the stream is in its plan**: the roles still to come are `todo`. -/
def Pos (S : State) (todo : List Role) : Prop := ∃ done, S.roles.toList = done ++ todo ∧ S.i = done.length

theorem Pos.get {S : State} {todo : List Role} (h : Pos S todo) : S.roles[S.i]? = todo.head? := by
  obtain ⟨done, hr, hi⟩ := h
  rw [← Array.getElem?_toList, hr, hi, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]
  cases todo <;> rfl

theorem Pos.next {S S' : State} {r : Role} {todo : List Role} (h : Pos S (r :: todo)) (hr : S'.roles = S.roles)
    (hi : S'.i = S.i + 1) : Pos S' todo := by
  obtain ⟨done, hr', hi'⟩ := h
  exact ⟨done ++ [r], by rw [hr, hr']; simp, by rw [hi, hi']; simp⟩

theorem step_circuit {C : Ctx} {S : State} {h b : ByteArray} {d : String} (hp : h ∉ pinned)
    (hr : S.roles[S.i]? = some (.circuit d)) :
    step C S h b =
      { S with i := S.i + 1,
               file := some (h, b, C.tags0.map fun tags0 =>
                 fileOf tags0 (tablesOf ⟨S.tables.push (h, b)⟩) ⟨S.tables.push (h, b)⟩ d),
               pub := none, cur := none, record := none, reps := #[] } := by
  unfold step
  simp only [not_pinned hp, Bool.false_eq_true, and_false, ↓reduceIte, hr]

/-- The state a member's files are read on, at its public file `bp`. -/
abbrev State.atPub (S : State) (bp : ByteArray) : State :=
  { S with i := S.i + 1, pub := some (entry bp), cur := none, record := none, reps := #[] }

/-- The state after the step on the public file `bp` of the member `m`, at the place `mi`. -/
abbrev State.afterPub (C : Ctx) (S : State) (bp : ByteArray) (mi : Nat) (st : Option Nat) (m : Member) : State :=
  { S.atPub bp with prog := (memberFiles C (S.atPub bp) m).2,
                    files := S.files.set! mi ((memberFiles C (S.atPub bp) m).1 >>= fun _ => pure ()),
                    cur := st.map fun _ => (memberFiles C (S.atPub bp) m).1 }

theorem step_pub {C : Ctx} {S : State} {b : ByteArray} {mi : Nat} {st : Option Nat} {m : Member}
    (hp : Flock.Sha512.hash b ∉ pinned) (hr : S.roles[S.i]? = some (.pub mi st)) (hm : C.ms[mi]? = some m) :
    feed1 C S b = S.afterPub C b mi st m := by
  unfold feed1 step
  simp only [not_pinned hp, Bool.false_eq_true, and_false, ↓reduceIte, hr]
  simp only [hm]

theorem step_record {C : Ctx} {S : State} {h b : ByteArray} {st : Nat} (hp : h ∉ pinned)
    (hr : S.roles[S.i]? = some (.record st)) :
    step C S h b = closeSession C st { S with i := S.i + 1, record := some (h, b), reps := #[] } := by
  unfold step
  simp only [not_pinned hp, Bool.false_eq_true, and_false, ↓reduceIte, hr]

theorem step_rep {C : Ctx} {S : State} {h b : ByteArray} {st j : Nat} (hp : h ∉ pinned)
    (hr : S.roles[S.i]? = some (.rep st j)) :
    step C S h b = closeSession C st { S with i := S.i + 1, reps := S.reps.push (h, b) } := by
  unfold step
  simp only [not_pinned hp, Bool.false_eq_true, and_false, ↓reduceIte, hr]

/-- A session stays open while its next rep file is still to come. -/
theorem close_open {C : Ctx} {S : State} {k j : Nat} (h : S.roles[S.i]? = some (.rep k j)) :
    closeSession C k S = S := by
  unfold closeSession
  rw [h]
  simp

/-- A session closes where the next blob is not one of its rep files: what is kept of it is `SessionKept.of` on the
blobs held, at the files' verdict of the member last read, for the first member under its statement's name. -/
theorem close_shut {C : Ctx} {S : State} {k : Nat} {s : RunFiles} {m : Member}
    (h : ∀ j, S.roles[S.i]? ≠ some (.rep k j)) (hs : C.sts[k]?.bind (·.sessions[0]?) = some s)
    (hm : (C.sts[k]?.bind (·.member)).bind (fun n => C.ms.find? (·.name == n)) = some m) :
    closeSession C k S =
      { S with sessions := S.sessions.set! k (some (SessionKept.of m (S.cur.getD (.error "")) S.blobs s)),
               record := none, reps := #[] } := by
  unfold closeSession
  simp only [hs, hm]
  split
  · rename_i k' j hr
    have hk : (k' == k) = false := by
      rw [beq_eq_false_iff_ne]
      rintro rfl
      exact absurd hr (h j)
    simp only [hk, Bool.false_eq_true, ↓reduceIte]
  · rfl

/-- No rep file comes next. -/
def NoRep (rs : List Role) : Prop := ∀ k j, rs.head? ≠ some (.rep k j)

/-- **A session's rep files' steps**: each joins the rep files held, and the session closes after the last. -/
theorem close_fold {C : Ctx} {R : List (ByteArray × ByteArray)} (hpin : ∀ x ∈ R, x.1 ∉ pinned) {k : Nat}
    {after : List Role} :
    ∀ (reps : List (ByteArray × Role)) (l : List ByteArray) (S : State),
      l.map Flock.Sha512.hash = reps.map (·.1) → (∀ e ∈ reps, ∃ j, e.2 = .rep k j) → (∀ b ∈ l, entry b ∈ R) →
      Pos S (reps.map (·.2) ++ after) →
      l.foldl (feed1 C) (closeSession C k S) =
          closeSession C k { S with i := S.i + l.length, reps := S.reps ++ (l.map entry).toArray } ∧
        Pos { S with i := S.i + l.length, reps := S.reps ++ (l.map entry).toArray } after
  | [], l, S, hl, _, _, hpos => by
    cases l with
    | nil => exact ⟨by simp, by simpa using hpos⟩
    | cons b l => simp at hl
  | e :: reps, l, S, hl, hrep, hR, hpos => by
    cases l with
    | nil => simp at hl
    | cons b l =>
      obtain ⟨j, hj⟩ := hrep e List.mem_cons_self
      rw [List.map_cons, List.cons_append, hj] at hpos
      have hget : S.roles[S.i]? = some (.rep k j) := hpos.get
      rw [List.map_cons, List.map_cons, List.cons.injEq] at hl
      have hb : Flock.Sha512.hash b ∉ pinned := hpin _ (hR b List.mem_cons_self)
      rw [List.foldl_cons, close_open hget]
      have hstep : feed1 C S b = closeSession C k { S with i := S.i + 1, reps := S.reps.push (entry b) } :=
        step_rep hb hget
      rw [hstep]
      obtain ⟨h1, h2⟩ := close_fold (C := C) hpin reps l { S with i := S.i + 1, reps := S.reps.push (entry b) } hl.2
        (fun e he => hrep e (List.mem_cons_of_mem _ he)) (fun b hb => hR b (List.mem_cons_of_mem _ hb))
        (hpos.next rfl rfl)
      have hi : S.i + (b :: l).length = S.i + 1 + l.length := by
        rw [List.length_cons]
        omega
      have hr : S.reps ++ ((b :: l).map entry).toArray = S.reps.push (entry b) ++ (l.map entry).toArray := by
        apply Array.ext'
        simp
      rw [hi, hr]
      exact ⟨h1, h2⟩

/-! ## A member's steps -/

/-- **What the planned blobs' steps rest on**: all the blobs fed (`full`) are the tables fed (`T`), then the planned
blobs (`R`), whose SHA-512 are distinct and none a pinned table's; the statement's tags are `Ctx.of`'s; and a member
whose name a statement's place is kept under is the member `closeSession` finds for that statement. -/
structure Env (C : Ctx) (stOf : Std.HashMap String Nat) (full : Core.Blobs) (T R : List (ByteArray × ByteArray)) :
    Prop where
  full_eq : full.entries.toList = T ++ R
  nodup : (R.map (·.1)).Nodup
  unpinned : ∀ x ∈ R, x.1 ∉ pinned
  tags : C.tags0 = Flock.Tags.find? STATEMENT
  named : ∀ (i : Nat) (m : Member) (k : Nat), C.ms[i]? = some m → stOf[m.name]? = some k →
    (C.sts[k]?.bind (·.member)).bind (fun n => C.ms.find? (·.name == n)) = some m

/-- Where the statement's tags are not known, a member's files read the blobs for its circuit file and the pinned
tables alone: they refuse after `tablesHeld`. -/
theorem files_none (htag : Flock.Tags.find? STATEMENT = none) {b₁ b₂ : Core.Blobs} (m : Member)
    (hc : Alike b₁ b₂ m.own.circuit) (ht : tablesOf b₁ = tablesOf b₂) (slice : Json) :
    m.files slice b₁ = m.files slice b₂ := by
  unfold Member.files Member.statement statementOf
  rw [tablesHeld_match, tablesHeld_match, hc, ht]
  simp only [htag]

/-- **A member's files' verdict at its public file's step is its files' verdict on all the blobs fed**, where the
stream holds the tables fed, the member's circuit file with what the files' pass keeps of it on all the blobs, and its
public file. -/
theorem memberFiles_right {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {S : State} {m : Member} {bc bp : ByteArray}
    (htab : S.tables.toList = T)
    (hfile : S.file = some (Flock.Sha512.hash bc, bc,
      C.tags0.map fun tags0 => fileOf tags0 (tablesOf full) full m.own.circuit))
    (hbc : entry bc ∈ R) (hbcd : Flock.hexOf (Flock.Sha512.hash bc) = m.own.circuit)
    (hpub : S.pub = some (entry bp)) (hbp : entry bp ∈ R) (hbpd : Flock.hexOf (Flock.Sha512.hash bp) = m.own.pub)
    (hrec : S.record = none) (hreps : S.reps = #[])
    (hprog : ∀ pk, S.prog = some pk → pk = programOf pk.bytes pk.partition) :
    (memberFiles C S m).1 = m.files (C.slice m) full ∧
      ∀ pk, (memberFiles C S m).2 = some pk → pk = programOf pk.bytes pk.partition := by
  have hbl : S.blobs.entries.toList = T ++ [entry bc, entry bp] := by
    unfold State.blobs
    simp [hfile, hpub, hrec, hreps, htab]
  have hsub : ∀ x ∈ [entry bc, entry bp], x ∈ R := by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hbc
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact hbp
      · cases hx
  have htabs : tablesOf S.blobs = tablesOf full := tablesOf_held hbl E.full_eq hsub E.unpinned
  have hpubA : Alike S.blobs full m.own.pub :=
    alike_of_mem hbl E.full_eq hsub E.nodup (List.mem_cons_of_mem _ List.mem_cons_self) hbpd
  have hcA : Alike S.blobs full m.own.circuit := alike_of_mem hbl E.full_eq hsub E.nodup List.mem_cons_self hbcd
  cases ht : C.tags0 with
  | none =>
    unfold memberFiles
    rw [ht]
    exact ⟨files_none (E.tags ▸ ht) m hcA htabs _, hprog⟩
  | some tags0 =>
    exact memberFiles_eq ht (E.tags ▸ ht) htabs ⟨_, _, by rw [hfile, ht]; rfl⟩ hprog hpubA

/-- **A session's steps**, its record and then its rep files: the session is kept as `SessionKept.of` keeps it on all
the blobs fed, where the stream holds the tables fed and its member's circuit and public files. -/
theorem session_fold {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {S : State} {i k : Nat} {m : Member}
    {s : RunFiles} {bc bp br : ByteArray} {lr : List ByteArray} {reps : List (ByteArray × Role)}
    {after : List Role} {fk : Option FileKept} {a : V StepAt}
    (hm : C.ms[i]? = some m) (hk : stOf[m.name]? = some k) (hs : C.sts[k]?.bind (·.sessions[0]?) = some s)
    (hpos : Pos S (Role.record k :: (reps.map (·.2) ++ after))) (hafter : NoRep after)
    (htab : S.tables.toList = T) (hfile : S.file = some (Flock.Sha512.hash bc, bc, fk))
    (hbc : entry bc ∈ R) (hbcd : Flock.hexOf (Flock.Sha512.hash bc) = m.own.circuit)
    (hpub : S.pub = some (entry bp)) (hbp : entry bp ∈ R) (hbpd : Flock.hexOf (Flock.Sha512.hash bp) = m.own.pub)
    (hcur : S.cur = some a) (hbr : entry br ∈ R) (hbrd : Flock.hexOf (Flock.Sha512.hash br) = s.record)
    (hlr : lr.map Flock.Sha512.hash = reps.map (·.1)) (hreps : ∀ e ∈ reps, ∃ j, e.2 = .rep k j)
    (hsreps : ∀ x ∈ s.reps, ∃ e ∈ reps, Flock.hexOf e.1 = x.2) (hlrR : ∀ b ∈ lr, entry b ∈ R) :
    (br :: lr).foldl (feed1 C) S =
        { S with i := S.i + 1 + lr.length, sessions := S.sessions.set! k (some (SessionKept.of m a full s)),
                 record := none, reps := #[] } ∧
      Pos { S with i := S.i + 1 + lr.length, sessions := S.sessions.set! k (some (SessionKept.of m a full s)),
                   record := none, reps := #[] } after := by
  have hstep : feed1 C S br = closeSession C k { S with i := S.i + 1, record := some (entry br), reps := #[] } :=
    step_record (E.unpinned _ hbr) hpos.get
  obtain ⟨hfold, hpos'⟩ := close_fold (C := C) E.unpinned reps lr
    { S with i := S.i + 1, record := some (entry br), reps := #[] } hlr hreps hlrR (hpos.next rfl rfl)
  have hshut := close_shut (C := C)
    (S := { S with i := S.i + 1 + lr.length, record := some (entry br), reps := #[] ++ (lr.map entry).toArray })
    (fun j => by rw [hpos'.get]; exact hafter k j) hs (E.named i m k hm hk)
  rw [List.foldl_cons, hstep, hfold, hshut]
  have hbl : (State.blobs
      { S with i := S.i + 1 + lr.length, record := some (entry br), reps := #[] ++ (lr.map entry).toArray }
        ).entries.toList = T ++ (entry bc :: entry bp :: entry br :: lr.map entry) := by
    unfold State.blobs
    simp [hfile, hpub, htab]
  have hsub : ∀ x ∈ entry bc :: entry bp :: entry br :: lr.map entry, x ∈ R := by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hbc
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact hbp
      · rcases List.mem_cons.1 hx with rfl | hx
        · exact hbr
        · obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hx
          exact hlrR b hb
  have hsa : SessionAlike (State.blobs
      { S with i := S.i + 1 + lr.length, record := some (entry br), reps := #[] ++ (lr.map entry).toArray })
        full s := by
    refine ⟨alike_of_mem hbl E.full_eq hsub E.nodup
      (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) hbrd, fun x hx => ?_⟩
    obtain ⟨e, he, hex⟩ := hsreps x hx
    have : e.1 ∈ lr.map Flock.Sha512.hash := by
      rw [hlr]
      exact List.mem_map_of_mem he
    obtain ⟨b, hb, hbe⟩ := List.mem_map.1 this
    refine alike_of_mem hbl E.full_eq hsub E.nodup (D := Flock.Sha512.hash b) (b := b)
      (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_map_of_mem hb)))) ?_
    rw [hbe]
    exact hex
  have hkept := SessionKept.of_congr (m := m) hsa
    (alike_of_mem hbl E.full_eq hsub E.nodup List.mem_cons_self hbcd)
    (alike_of_mem hbl E.full_eq hsub E.nodup (List.mem_cons_of_mem _ List.mem_cons_self) hbpd)
    (tablesOf_held hbl E.full_eq hsub E.unpinned) a
  have hget : S.cur.getD (.error "") = a := by
    rw [hcur]
    rfl
  dsimp only
  rw [hget, hkept]
  exact ⟨rfl, hpos'⟩

/-- **A member's steps**, its public file and then, where a statement names it, that statement's session's: its files'
verdict and its session are kept as they read on all the blobs fed, and nothing else kept changes. -/
theorem member_fold {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {S : State} {i : Nat} {m : Member}
    {bc : ByteArray} {after : List Role} {seg : Array (ByteArray × Role)} {l : List ByteArray}
    (hm : C.ms[i]? = some m) (hseg : memberPlan C.ms C.sts stOf i = .ok seg)
    (hl : l.map Flock.Sha512.hash = seg.toList.map (·.1)) (hlR : ∀ b ∈ l, entry b ∈ R)
    (hpos : Pos S (seg.toList.map (·.2) ++ after)) (hafter : NoRep after) (htab : S.tables.toList = T)
    (hfile : S.file = some (Flock.Sha512.hash bc, bc,
      C.tags0.map fun tags0 => fileOf tags0 (tablesOf full) full m.own.circuit))
    (hbc : entry bc ∈ R) (hbcd : Flock.hexOf (Flock.Sha512.hash bc) = m.own.circuit)
    (hprog : ∀ pk, S.prog = some pk → pk = programOf pk.bytes pk.partition) :
    Pos (l.foldl (feed1 C) S) after ∧ (l.foldl (feed1 C) S).tables = S.tables ∧
      (l.foldl (feed1 C) S).file = S.file ∧
      (∀ pk, (l.foldl (feed1 C) S).prog = some pk → pk = programOf pk.bytes pk.partition) ∧
      (l.foldl (feed1 C) S).files = S.files.set! i (m.files (C.slice m) full >>= fun _ => pure ()) ∧
      ((stOf[m.name]? = none ∧ (l.foldl (feed1 C) S).sessions = S.sessions) ∨
        ∃ k s, stOf[m.name]? = some k ∧ C.sts[k]?.bind (·.sessions[0]?) = some s ∧
          (l.foldl (feed1 C) S).sessions =
            S.sessions.set! k (some (SessionKept.of m (m.files (C.slice m) full) full s))) := by
  obtain ⟨m₁, Dp, hm₁, hDp, hcase⟩ := memberPlan_ok hseg
  rw [hm] at hm₁
  cases hm₁
  rcases hcase with ⟨hnone, hsegl⟩ | ⟨k, s, Dr, reps, hk, hs, hDr, hsegl, hreps, hsreps⟩
  · rw [hsegl] at hl hpos
    obtain ⟨bp, rfl, hbp⟩ : ∃ bp, l = [bp] ∧ Flock.Sha512.hash bp = Dp := by
      cases l with
      | nil => simp at hl
      | cons bp l =>
        cases l with
        | nil => exact ⟨bp, rfl, by simpa using hl⟩
        | cons _ _ => simp at hl
    have hbpR := hlR bp List.mem_cons_self
    have hmf := memberFiles_right E (S := S.atPub bp) (m := m) (bc := bc) (bp := bp) htab hfile hbc hbcd rfl hbpR
      (by rw [hbp]; exact hDp) rfl rfl hprog
    have hstep := step_pub (C := C) (E.unpinned _ hbpR) hpos.get hm
    rw [List.foldl_cons, List.foldl_nil, hstep]
    refine ⟨hpos.next rfl rfl, rfl, rfl, hmf.2, ?_, .inl ⟨hnone, rfl⟩⟩
    dsimp only
    rw [hmf.1]
  · rw [hsegl] at hl hpos
    obtain ⟨bp, br, lr, rfl, hbp, hbr, hlr⟩ : ∃ bp br lr, l = bp :: br :: lr ∧ Flock.Sha512.hash bp = Dp ∧
        Flock.Sha512.hash br = Dr ∧ lr.map Flock.Sha512.hash = reps.map (·.1) := by
      cases l with
      | nil => simp at hl
      | cons bp l =>
        cases l with
        | nil => simp at hl
        | cons br lr =>
          simp only [List.map_cons, List.cons.injEq] at hl
          exact ⟨bp, br, lr, rfl, hl.1, hl.2.1, hl.2.2⟩
    have hbpR := hlR bp List.mem_cons_self
    have hbrR := hlR br (List.mem_cons_of_mem _ List.mem_cons_self)
    have hmf := memberFiles_right E (S := S.atPub bp) (m := m) (bc := bc) (bp := bp) htab hfile hbc hbcd rfl hbpR
      (by rw [hbp]; exact hDp) rfl rfl hprog
    have hstep := step_pub (C := C) (E.unpinned _ hbpR) hpos.get hm
    rw [List.foldl_cons, hstep]
    obtain ⟨hfold, hpos'⟩ := session_fold E (bc := bc) (bp := bp) (br := br) (lr := lr) (reps := reps) (after := after)
      (S := S.afterPub C bp i (some k) m) hm hk hs (hpos.next rfl rfl) hafter htab hfile hbc hbcd rfl hbpR
      (by rw [hbp]; exact hDp) rfl hbrR (by rw [hbr]; exact hDr) hlr hreps hsreps
      (fun b hb => hlR b (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hb)))
    rw [hfold]
    refine ⟨hpos', rfl, rfl, hmf.2, ?_, .inr ⟨k, s, hk, hs, ?_⟩⟩
    · dsimp only
      rw [hmf.1]
    · dsimp only
      rw [hmf.1]

/-! ## The plan's steps -/

/-- **What the stream keeps between members**, the members at the places `D` read: the tables fed, a program kept as
`programOf` keeps it, and each of those members' files' verdict and its statement's session as they read on all the
blobs fed (`full`). -/
structure Inv (C : Ctx) (stOf : Std.HashMap String Nat) (full : Core.Blobs) (T : List (ByteArray × ByteArray))
    (D : Nat → Prop) (S : State) : Prop where
  tables : S.tables.toList = T
  prog : ∀ pk, S.prog = some pk → pk = programOf pk.bytes pk.partition
  fsize : S.files.size = C.ms.size
  ssize : S.sessions.size = C.sts.size
  files : ∀ (i : Nat) (m : Member), D i → C.ms[i]? = some m →
    S.files[i]? = some (m.files (C.slice m) full >>= fun _ => pure ())
  sessions : ∀ (i : Nat) (m : Member) (k : Nat) (s : RunFiles), D i → C.ms[i]? = some m → stOf[m.name]? = some k →
    C.sts[k]?.bind (·.sessions[0]?) = some s →
      S.sessions[k]? = some (some (SessionKept.of m (m.files (C.slice m) full) full s))

theorem Inv.mono {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs} {T : List (ByteArray × ByteArray)}
    {D D' : Nat → Prop} {S : State} (I : Inv C stOf full T D S) (h : ∀ i, D' i → D i) : Inv C stOf full T D' S :=
  ⟨I.tables, I.prog, I.fsize, I.ssize, fun i m hi => I.files i m (h i hi),
    fun i m k s hi => I.sessions i m k s (h i hi)⟩

/-- **A member's steps keep what is kept**, and add the member: its files' verdict by its place, and its statement's
session by the statement's. A member read before under the same statement is the same member (`Env.named`). -/
theorem Inv.member {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {D : Nat → Prop} {S S' : State} {i : Nat}
    {m : Member} (I : Inv C stOf full T D S) (hm : C.ms[i]? = some m) (htab : S'.tables = S.tables)
    (hprog : ∀ pk, S'.prog = some pk → pk = programOf pk.bytes pk.partition)
    (hfiles : S'.files = S.files.set! i (m.files (C.slice m) full >>= fun _ => pure ()))
    (hsess : (stOf[m.name]? = none ∧ S'.sessions = S.sessions) ∨
      ∃ k s, stOf[m.name]? = some k ∧ C.sts[k]?.bind (·.sessions[0]?) = some s ∧
        S'.sessions = S.sessions.set! k (some (SessionKept.of m (m.files (C.slice m) full) full s))) :
    Inv C stOf full T (fun x => D x ∨ x = i) S' := by
  have hilt : i < C.ms.size := (Array.getElem?_eq_some_iff.1 hm).1
  refine ⟨by rw [htab]; exact I.tables, hprog, by rw [hfiles]; simp [I.fsize], ?_, ?_, ?_⟩
  · rcases hsess with ⟨_, h⟩ | ⟨k, s, _, _, h⟩ <;> rw [h] <;> simp [I.ssize]
  · intro x mx hx hmx
    rw [hfiles, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
    by_cases hxi : i = x
    · subst hxi
      rw [hm] at hmx
      cases hmx
      simp [I.fsize, hilt]
    · simp only [hxi, ↓reduceIte]
      rcases hx with hx | hx
      · exact I.files x mx hx hmx
      · exact absurd hx.symm hxi
  · intro x mx kx sx hx hmx hkx hsx
    rcases hsess with ⟨hnone, h⟩ | ⟨k, s, hk, hs, h⟩
    · rw [h]
      rcases hx with hx | hx
      · exact I.sessions x mx kx sx hx hmx hkx hsx
      · subst hx
        rw [hm] at hmx
        cases hmx
        rw [hkx] at hnone
        cases hnone
    · rw [h, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
      have hklt : k < C.sts.size := by
        cases hsk : C.sts[k]? with
        | none =>
          rw [hsk] at hs
          cases hs
        | some _ => exact (Array.getElem?_eq_some_iff.1 hsk).1
      by_cases hkk : k = kx
      · subst hkk
        have hmm : mx = m := by
          have h1 := E.named x mx k hmx hkx
          rw [E.named i m k hm hk] at h1
          exact (Option.some.inj h1).symm
        subst hmm
        rw [hs] at hsx
        cases hsx
        simp [I.ssize, hklt]
      · simp only [hkk, ↓reduceIte]
        rcases hx with hx | hx
        · exact I.sessions x mx kx sx hx hmx hkx hsx
        · subst hx
          rw [hm] at hmx
          cases hmx
          rw [hk] at hkx
          cases hkx
          exact absurd rfl hkk

/-- No rep file comes first in the members' planned blobs. -/
theorem noRep_members {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat}
    {after : List Role} (hafter : NoRep after) : ∀ (is : List Nat),
      (∀ i ∈ is, ∃ seg, memberPlan ms sts stOf i = .ok seg) →
        NoRep ((is.flatMap (mpL ms sts stOf)).map (·.2) ++ after)
  | [], _ => by simpa using hafter
  | i :: is, h => by
    obtain ⟨seg, hseg⟩ := h i List.mem_cons_self
    obtain ⟨m, Dp, _, _, hcase⟩ := memberPlan_ok hseg
    intro k j
    rw [List.flatMap_cons, mpL_of_ok hseg]
    rcases hcase with ⟨_, hs⟩ | ⟨_, _, _, _, _, _, _, hs, _⟩ <;> rw [hs] <;> simp

/-- No rep file comes first in the groups' planned blobs. -/
theorem noRep_groups {ms : Array Member} {sts : Array StatementFiles} {stOf : Std.HashMap String Nat} :
    ∀ (gs : List (String × Array Nat)), (∀ g ∈ gs, ∃ seg, groupPlan ms sts stOf g = .ok seg) →
      NoRep ((gs.flatMap (gpL ms sts stOf)).map (·.2))
  | [], _ => fun k j => by simp
  | g :: gs, h => by
    obtain ⟨seg, hseg⟩ := h g List.mem_cons_self
    obtain ⟨D, _, hs, _⟩ := groupPlan_ok hseg
    intro k j
    rw [List.flatMap_cons, gpL_of_ok hseg, hs]
    simp

/-- **A group's members' steps**, the stream holding the group's circuit file `bc` with what the files' pass keeps of
it on all the blobs fed: every member of the group is read as on all the blobs fed. -/
theorem members_fold {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {bc : ByteArray} {d : String}
    {after : List Role} (hbc : entry bc ∈ R) (hbcd : Flock.hexOf (Flock.Sha512.hash bc) = d)
    (hafter : NoRep after) :
    ∀ (is : List Nat) (l : List ByteArray) (S : State) (D : Nat → Prop),
      (∀ i ∈ is, ∃ m, C.ms[i]? = some m ∧ m.own.circuit = d) →
      (∀ i ∈ is, ∃ seg, memberPlan C.ms C.sts stOf i = .ok seg) →
      l.map Flock.Sha512.hash = (is.flatMap (mpL C.ms C.sts stOf)).map (·.1) → (∀ b ∈ l, entry b ∈ R) →
      Pos S ((is.flatMap (mpL C.ms C.sts stOf)).map (·.2) ++ after) →
      S.file = some (Flock.Sha512.hash bc, bc, C.tags0.map fun tags0 => fileOf tags0 (tablesOf full) full d) →
      Inv C stOf full T D S →
        Pos (l.foldl (feed1 C) S) after ∧ Inv C stOf full T (fun x => D x ∨ x ∈ is) (l.foldl (feed1 C) S)
  | [], l, S, D, _, _, hl, _, hpos, _, I => by
    have hnil : l = [] := by simpa using hl
    subst hnil
    exact ⟨by simpa using hpos, I.mono fun i hi => hi.elim id fun h => nomatch h⟩
  | i :: is, l, S, D, hmem, hplan, hl, hlR, hpos, hfile, I => by
    obtain ⟨m, hm, hd⟩ := hmem i List.mem_cons_self
    obtain ⟨seg, hseg⟩ := hplan i List.mem_cons_self
    rw [List.flatMap_cons, mpL_of_ok hseg, List.map_append] at hl hpos
    obtain ⟨l₁, l₂, rfl, hl₁, hl₂⟩ := List.map_eq_append_iff.1 hl
    rw [List.append_assoc] at hpos
    have hnext : NoRep ((is.flatMap (mpL C.ms C.sts stOf)).map (·.2) ++ after) :=
      noRep_members hafter is fun i hi => hplan i (List.mem_cons_of_mem _ hi)
    obtain ⟨hpos₁, htab₁, hfile₁, hprog₁, hfiles₁, hsess₁⟩ := member_fold E hm hseg hl₁
      (fun b hb => hlR b (List.mem_append_left _ hb)) hpos hnext I.tables (by rw [hd]; exact hfile) hbc
      (hbcd.trans hd.symm) I.prog
    have I₁ := I.member E hm htab₁ hprog₁ hfiles₁ hsess₁
    rw [List.foldl_append]
    obtain ⟨hpos₂, I₂⟩ := members_fold E hbc hbcd hafter is l₂ _ _ (fun i hi => hmem i (List.mem_cons_of_mem _ hi))
      (fun i hi => hplan i (List.mem_cons_of_mem _ hi)) hl₂ (fun b hb => hlR b (List.mem_append_right _ hb)) hpos₁
      (hfile₁.trans hfile) I₁
    refine ⟨hpos₂, I₂.mono fun x hx => ?_⟩
    rcases hx with hx | hx
    · exact .inl (.inl hx)
    · rcases List.mem_cons.1 hx with rfl | hx
      · exact .inl (.inr rfl)
      · exact .inr hx

/-- **A group's steps**, its circuit file and then its members': what the circuit file's step keeps of the file is
what the files' pass keeps of it on all the blobs fed, and every member of the group is read as on all of them. -/
theorem group_fold {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) {g : String × Array Nat} {after : List Role}
    {l : List ByteArray} {S : State} {D : Nat → Prop} {seg : Array (ByteArray × Role)}
    (hseg : groupPlan C.ms C.sts stOf g = .ok seg)
    (hmem : ∀ i ∈ g.2.toList, ∃ m, C.ms[i]? = some m ∧ m.own.circuit = g.1)
    (hl : l.map Flock.Sha512.hash = (gpL C.ms C.sts stOf g).map (·.1)) (hlR : ∀ b ∈ l, entry b ∈ R)
    (hpos : Pos S ((gpL C.ms C.sts stOf g).map (·.2) ++ after)) (hafter : NoRep after)
    (I : Inv C stOf full T D S) :
    Pos (l.foldl (feed1 C) S) after ∧
      Inv C stOf full T (fun x => D x ∨ x ∈ g.2.toList) (l.foldl (feed1 C) S) := by
  obtain ⟨Dc, hDc, hsegl, hplan⟩ := groupPlan_ok hseg
  rw [gpL_of_ok hseg, hsegl] at hl hpos
  cases l with
  | nil => simp at hl
  | cons bc l =>
    rw [List.map_cons, List.map_cons, List.cons.injEq] at hl
    rw [List.map_cons, List.cons_append] at hpos
    have hbcR := hlR bc List.mem_cons_self
    have hstep : feed1 C S bc = _ := step_circuit (E.unpinned _ hbcR) hpos.get
    rw [List.foldl_cons, hstep]
    have hbl : (⟨S.tables.push (entry bc)⟩ : Core.Blobs).entries.toList = T ++ [entry bc] := by simp [I.tables]
    have hsub : ∀ x ∈ [entry bc], x ∈ R := by
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hbcR
      · cases hx
    have hbcd : Flock.hexOf (Flock.Sha512.hash bc) = g.1 := by
      rw [hl.1]
      exact hDc
    have hfk : ∀ tags0, fileOf tags0 (tablesOf ⟨S.tables.push (entry bc)⟩) ⟨S.tables.push (entry bc)⟩ g.1 =
        fileOf tags0 (tablesOf full) full g.1 := by
      intro tags0
      rw [tablesOf_held hbl E.full_eq hsub E.unpinned]
      exact fileOf_congr (alike_of_mem hbl E.full_eq hsub E.nodup List.mem_cons_self hbcd) _ _
    exact members_fold E hbcR hbcd hafter g.2.toList l _ D hmem hplan hl.2
      (fun b hb => hlR b (List.mem_cons_of_mem _ hb)) (hpos.next rfl rfl) (by simp only [hfk])
      ⟨I.tables, I.prog, I.fsize, I.ssize, I.files, I.sessions⟩

/-- **The planned blobs' steps**, group by group: every member of every group is read as on all the blobs fed. -/
theorem plan_fold {C : Ctx} {stOf : Std.HashMap String Nat} {full : Core.Blobs}
    {T R : List (ByteArray × ByteArray)} (E : Env C stOf full T R) :
    ∀ (gs : List (String × Array Nat)) (l : List ByteArray) (S : State) (D : Nat → Prop),
      (∀ g ∈ gs, ∃ seg, groupPlan C.ms C.sts stOf g = .ok seg) →
      (∀ g ∈ gs, ∀ i ∈ g.2.toList, ∃ m, C.ms[i]? = some m ∧ m.own.circuit = g.1) →
      l.map Flock.Sha512.hash = (gs.flatMap (gpL C.ms C.sts stOf)).map (·.1) → (∀ b ∈ l, entry b ∈ R) →
      Pos S ((gs.flatMap (gpL C.ms C.sts stOf)).map (·.2)) → Inv C stOf full T D S →
        Inv C stOf full T (fun x => D x ∨ ∃ g ∈ gs, x ∈ g.2.toList) (l.foldl (feed1 C) S)
  | [], l, S, D, _, _, hl, _, _, I => by
    have hnil : l = [] := by simpa using hl
    subst hnil
    exact I.mono fun i hi => hi.elim id fun ⟨g, hg, _⟩ => nomatch hg
  | g :: gs, l, S, D, hplan, hmem, hl, hlR, hpos, I => by
    obtain ⟨seg, hseg⟩ := hplan g List.mem_cons_self
    rw [List.flatMap_cons, List.map_append] at hl hpos
    obtain ⟨l₁, l₂, rfl, hl₁, hl₂⟩ := List.map_eq_append_iff.1 hl
    have hnext : NoRep ((gs.flatMap (gpL C.ms C.sts stOf)).map (·.2)) :=
      noRep_groups gs fun g hg => hplan g (List.mem_cons_of_mem _ hg)
    obtain ⟨hpos₁, I₁⟩ := group_fold E hseg (hmem g List.mem_cons_self) hl₁
      (fun b hb => hlR b (List.mem_append_left _ hb)) hpos hnext I
    rw [List.foldl_append]
    refine (plan_fold E gs l₂ _ _ (fun g hg => hplan g (List.mem_cons_of_mem _ hg))
      (fun g hg => hmem g (List.mem_cons_of_mem _ hg)) hl₂ (fun b hb => hlR b (List.mem_append_right _ hb)) hpos₁
      I₁).mono fun x hx => ?_
    rcases hx with hx | ⟨g', hg', hx⟩
    · exact .inl (.inl hx)
    · rcases List.mem_cons.1 hg' with rfl | hg'
      · exact .inl (.inr hx)
      · exact .inr ⟨g', hg', hx⟩

/-- **The fold's half of `StreamSound` holds**: on a feed in the per-member stream's order, the state after every step
keeps every member's files and every statement's session as they read on all the blobs fed. The tables fed join the
tables held (`tables_fold`); the planned blobs are read group by group (`plan_fold`), each blob once, since the plan's
SHA-512 are distinct and none a pinned table's (`planOf_ok`); every member is in a group (`groupsOf_covers`); and a
statement's member is the one member read under its name (`plan_named`). -/
theorem stepsKept : StepsKept := by
  intro p c e ms sts plan bs hplan hfed
  have hplan' : (planOf ms sts).map (·.map (·.1)) = .ok plan := hplan
  cases hP : planOf ms sts with
  | error err =>
    rw [hP] at hplan'
    cases hplan'
  | ok P =>
    rw [hP] at hplan'
    cases hplan'
    obtain ⟨held, hheld, hPl, hgs, hnd, hpin⟩ := planOf_ok hP
    obtain ⟨pre, rest, rfl, hpre, hrest⟩ := hfed
    have htabs : (membersStream p c e ms sts).tables = pinned := by
      unfold membersStream
      with_reducible rfl
    rw [htabs] at hpre
    have hpre' : ∀ b ∈ pre, Flock.Sha512.hash b ∈ pinned := fun b hb => hpre.subset (List.mem_map_of_mem hb)
    have hrest' : rest.map Flock.Sha512.hash = P.toList.map (·.1) := by simpa using hrest
    have hsteps : (membersStream p c e ms sts).steps (pre ++ rest) =
        rest.foldl (feed1 (Ctx.of p c e ms sts))
          { State.init (Ctx.of p c e ms sts) (P.map (·.2)) with tables := #[] ++ (pre.map entry).toArray } := by
      show (pre ++ rest).foldl (feed1 (Ctx.of p c e ms sts))
        (State.init (Ctx.of p c e ms sts) (((planOf ms sts).toOption.getD #[]).map (·.2))) = _
      rw [hP, List.foldl_append, tables_fold _ pre _ rfl hpre']
      rfl
    rw [hsteps]
    have E : Env (Ctx.of p c e ms sts) (stOfHeld held) (Core.Blobs.ofArray (pre ++ rest).toArray) (pre.map entry)
        (rest.map entry) :=
      { full_eq := by simp [Core.Blobs.ofArray]
        nodup := by
          rw [List.map_map]
          have hkeys : rest.map ((·.1) ∘ entry) = P.toList.map (·.1) := hrest'
          rw [hkeys]
          exact nodup_of_hex (by rw [List.map_map]; exact hnd)
        unpinned := by
          intro x hx
          obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hx
          have hbP : Flock.Sha512.hash b ∈ P.toList.map (·.1) := by
            rw [← hrest']
            exact List.mem_map_of_mem hb
          obtain ⟨e', he', hbe⟩ := List.mem_map.1 hbP
          have := hpin e' he'
          rw [hbe] at this
          exact this
        tags := rfl
        named := fun i m k hm hk => (plan_named hheld hPl hgs hnd hm hk).2 }
    have hPl' : P.toList = (groupsOf ms).toList.flatMap
        (gpL (Ctx.of p c e ms sts).ms (Ctx.of p c e ms sts).sts (stOfHeld held)) := hPl
    have I := plan_fold E (groupsOf ms).toList rest
      { State.init (Ctx.of p c e ms sts) (P.map (·.2)) with tables := #[] ++ (pre.map entry).toArray }
      (fun _ => False) hgs
      (fun g hg i hi => groupsOf_mem (Array.mem_toList_iff.1 hg) (Array.mem_toList_iff.1 hi))
      (by rw [← hPl']; exact hrest') (fun b hb => List.mem_map_of_mem hb)
      ⟨[], (by rw [← hPl']; simp [State.init]), rfl⟩
      { tables := by simp
        prog := fun pk h => nomatch h
        fsize := by simp [State.init]
        ssize := by simp [State.init]
        files := fun _ _ h => h.elim
        sessions := fun _ _ _ _ h => h.elim }
    have hcov : ∀ {i : Nat} {m : Member}, ms[i]? = some m → False ∨ ∃ g ∈ (groupsOf ms).toList, i ∈ g.2.toList := by
      intro i m hm
      obtain ⟨g, hg, hi⟩ := groupsOf_covers hm
      exact .inr ⟨g, Array.mem_toList_iff.2 hg, Array.mem_toList_iff.2 hi⟩
    refine ⟨⟨I.fsize, fun i m hm => I.files i m (hcov hm) hm⟩, ?_⟩
    intro held' hh' k st m s hst hmk hs0
    have hheq : held' = held := by
      have h2 : membersHeld ms sts = .ok held' := hh'
      rw [hheld] at h2
      exact (Except.ok.inj h2).symm
    subst hheq
    obtain ⟨_, hnamed, hndn⟩ := membersHeld_named hheld
    obtain ⟨_, _, hfind⟩ := hnamed k st m hst hmk
    obtain ⟨i, hilt, hi⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hfind)
    have hm : ms[i]? = some m := by rw [Array.getElem?_eq_getElem hilt, hi]
    exact I.sessions i m k s (hcov hm) hm (stOfHeld_of hndn hmk) (by
      have h2 : sts[k]? = some st := hst
      show sts[k]?.bind (·.sessions[0]?) = some s
      rw [h2]
      exact hs0)

/-- **The streamed check decides what the whole check decides**: `StreamSound`, with no hypothesis. The fold keeps what
the finish reads (`stepsKept`), and the finish reads it as `checkBlobs` reads all the blobs fed (`finishKept`). -/
theorem streamSound : StreamSound := streamSound_of_steps stepsKept

end Spot.OneStageZk.Streamed
