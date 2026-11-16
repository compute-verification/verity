module

public import Proofs.Flock.SpotFlock.AddressFacts.RowLeaves

/-!
# The derived wires are one a reader port

`AddressRow.wiresChecked h` (`RowLeaves`) asks two things of each wire of the auditor's copies: no link names the
wire's reader port, and the first wire at that port names the wire's writer port. This file proves the second of
every list of wires the design derives from the outer block's registered Definition (`SecondStage.replayWires`), so
that at such copies `wiresChecked` is its first clause alone (`wiresUnlinked`, `wiresChecked_of_unlinked`).

The wires are derived by `SecondStage.gather`, which folds the callers' shares into one entry a template port: a share
whose port an entry holds is merged into that entry, and any other is appended. So no two entries of what it returns
share a port (`gather_keys`), and the first wire at a wire's reader port is the wire itself.

The first clause is not a fact of the derivation. A wire's reader port and a link's port are both a template's
descriptor id and one of its parameters' names, or the name of one of its output fields, and the derivation gathers
the links and the wires apart. A registered program in which a template's parameter is named like an output field, or
two of its parameters share a name, can derive a link and a wire of one port.
-/

@[expose] public section

namespace SpotFlock

namespace AddressRow

/-! ## Lists with one entry a key -/

/-- Two entries of one key, in a list with no key twice, are one entry. -/
theorem eq_of_nodup_keys {α β : Type} {k : α → β} :
    ∀ {l : List α}, (l.map k).Nodup → ∀ {a b : α}, a ∈ l → b ∈ l → k a = k b → a = b
  | [], _, _, _, ha, _, _ => by simp at ha
  | c :: l, hl, a, b, ha, hb, hk => by
    rw [List.map_cons, List.nodup_cons] at hl
    rw [List.mem_cons] at ha hb
    rcases ha with ha | ha
    · rcases hb with hb | hb
      · rw [ha, hb]
      · exact absurd (List.mem_map.2 ⟨b, hb, by rw [← hk, ha]⟩) hl.1
    · rcases hb with hb | hb
      · exact absurd (List.mem_map.2 ⟨a, ha, by rw [hk, hb]⟩) hl.1
      · exact eq_of_nodup_keys hl.2 ha hb hk

/-- A list's keys are the same after an entry is replaced by one of its key. -/
theorem map_set_of_eq {α β : Type} {f : α → β} {a b : α} (hab : f b = f a) :
    ∀ (l : List α) (j : ℕ), l[j]? = some a → (l.set j b).map f = l.map f
  | [], _, h => by simp at h
  | c :: l, 0, h => by
    rw [List.getElem?_cons_zero] at h
    rw [List.set_cons_zero, List.map_cons, List.map_cons, hab, Option.some.inj h]
  | c :: l, j + 1, h => by
    rw [List.getElem?_cons_succ] at h
    rw [List.set_cons_succ, List.map_cons, List.map_cons, map_set_of_eq hab l j h]

/-- An array's keys are the same after an entry is replaced by one of its key. -/
theorem keys_set {α β : Type} {f : α → β} {xs : Array α} {j : ℕ} {a b : α} (ha : xs[j]? = some a)
    (hab : f b = f a) : (xs.set! j b).toList.map f = xs.toList.map f := by
  rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds]
  exact map_set_of_eq hab xs.toList j (by rw [Array.getElem?_toList]; exact ha)

/-- A list with no entry twice has none twice after a new entry is appended. -/
theorem nodup_concat {β : Type} {l : List β} {k : β} (hl : l.Nodup) (hk : k ∉ l) : (l ++ [k]).Nodup := by
  rw [List.nodup_append]
  refine ⟨hl, List.pairwise_singleton _ k, ?_⟩
  intro a ha b hb hab
  rw [List.mem_singleton] at hb
  rw [hab, hb] at ha
  exact hk ha

/-- The keys of an array's entries, each mapped, are the array's when the map keeps each entry's key. -/
theorem nodup_keys_map {α β γ : Type} {xs : Array α} {f : α → γ} {g : α → β} {k : β → γ}
    (h : (xs.toList.map f).Nodup) (hfk : ∀ a, k (g a) = f a) : ((xs.map g).toList.map k).Nodup := by
  have hm : (xs.map g).toList.map k = xs.toList.map f := by
    rw [Array.toList_map, List.map_map]
    exact List.map_congr_left fun a _ => hfk a
  rw [hm]
  exact h

/-- In an array with no key twice, the first entry a test takes is the entry `a`, when the test takes `a` and only
entries of `a`'s key. -/
theorem find?_of_keys {α β : Type} {k : α → β} {xs : Array α} {p : α → Bool} {a : α}
    (hk : (xs.toList.map k).Nodup) (ha : a ∈ xs) (hpa : p a = true) (hp : ∀ b, p b = true → k b = k a) :
    xs.find? p = some a := by
  cases hf : xs.find? p with
  | none => exact absurd hpa (Array.find?_eq_none.1 hf a ha)
  | some b =>
    exact congrArg some (eq_of_nodup_keys hk (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf))
      (Array.mem_toList_iff.2 ha) (hp b (Array.find?_some hf)))

/-- A fold in `Option` that returns keeps what each of its steps keeps. -/
theorem foldlM_keeps {α β : Type} {f : β → α → Option β} {Q : β → Prop}
    (hf : ∀ b a b', Q b → f b a = some b' → Q b') :
    ∀ (xs : List α) {b b' : β}, Q b → xs.foldlM f b = some b' → Q b'
  | [], b, b', hb, h => by
    rw [List.foldlM_nil] at h
    exact Option.some.inj h ▸ hb
  | a :: xs, b, b', hb, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨c, hc, h⟩ := Option.bind_eq_some_iff.1 h
    exact foldlM_keeps hf xs (hf b a c hb hc) h

/-! ## What `gather` returns -/

/-- **`gather` returns one entry a port**: no two entries of what it returns have one port. Each share is merged into
the entry that holds its port, where one does, and appended only where none does. -/
theorem gather_keys {α : Type} {port : α → String × String} {same : α → α → Bool} {units : Array ℕ}
    {xs : List (Spot.SecondStage.Share α)} {out : Array (α × ℕ × Array Spot.SecondStage.Run)}
    (h : Spot.SecondStage.gather port same units xs = some out) : (out.toList.map fun e => port e.1).Nodup := by
  unfold Spot.SecondStage.gather at h
  obtain ⟨g, hg, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  have hQ : (g.toList.map fun e => port e.1.edge).Nodup := by
    refine foldlM_keeps (Q := fun acc : Array (Spot.SecondStage.Share α × Array Spot.SecondStage.Run) =>
      (acc.toList.map fun e => port e.1.edge).Nodup) ?_ xs ?_ hg
    · intro acc x acc' hacc hstep
      try dsimp only at hstep
      split at hstep
      · rename_i j hj
        obtain ⟨e, he, hstep⟩ := Option.bind_eq_some_iff.1 hstep
        obtain ⟨y, rs⟩ := e
        obtain ⟨_, -, hstep⟩ := Option.bind_eq_some_iff.1 hstep
        have h' := Option.some.inj hstep
        subst h'
        have hk := keys_set (f := fun e : Spot.SecondStage.Share α × Array Spot.SecondStage.Run => port e.1.edge)
          (b := (y, Spot.SecondStage.pushRun rs x.run)) he rfl
        show ((acc.set! j (y, Spot.SecondStage.pushRun rs x.run)).toList.map fun e => port e.1.edge).Nodup
        rw [hk]
        exact hacc
      · rename_i hj
        have h' := Option.some.inj hstep
        subst h'
        show ((acc.push (x, #[x.run])).toList.map fun e => port e.1.edge).Nodup
        rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil]
        refine nodup_concat hacc ?_
        intro hmem
        obtain ⟨e, he, hek⟩ := List.mem_map.1 hmem
        have hne := Array.findIdx?_eq_none_iff.1 hj e (Array.mem_toList_iff.1 he)
        obtain ⟨y, rs⟩ := e
        exact ne_of_beq_false hne hek
    · exact List.nodup_nil
  have h' := Option.some.inj h
  subst h'
  exact nodup_keys_map hQ fun ⟨_, _⟩ => rfl

/-- The wires the design derives with their rows (`SecondStage.wireRuns`) are one a reader port. -/
theorem wireRuns_keys {defs : Array _root_.Flock.Program.Defn} {d : ℕ}
    {rs : Array (Spot.SecondStage.Wire × ℕ × Array Spot.SecondStage.Run)}
    (h : Spot.SecondStage.wireRuns defs d = some rs) : (rs.toList.map fun e => (e.1.template, e.1.port)).Nodup := by
  unfold Spot.SecondStage.wireRuns at h
  obtain ⟨D, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨ws, -, h⟩ := Option.bind_eq_some_iff.1 h
  exact gather_keys h

/-- The wires the design derives (`SecondStage.wiresOf`) are one a reader port. -/
theorem wiresOf_keys {defs : Array _root_.Flock.Program.Defn} {d : ℕ} {ws : Array Spot.SecondStage.Wire}
    (h : Spot.SecondStage.wiresOf defs d = some ws) : (ws.toList.map fun w => (w.template, w.port)).Nodup := by
  unfold Spot.SecondStage.wiresOf at h
  obtain ⟨rs, hrs, h⟩ := Option.map_eq_some_iff.1 h
  subst h
  exact nodup_keys_map (wireRuns_keys hrs) fun _ => rfl

/-- **The wires of the outer block's registered Definition are one a reader port**: no two wires
`SecondStage.replayWires` derives name one template's one port, and where it derives none there is no wire. -/
theorem replayWires_keys (P : _root_.Flock.Program.Prog) (key : String) :
    (((Spot.SecondStage.replayWires P key).getD #[]).toList.map fun w => (w.template, w.port)).Nodup := by
  cases hr : Spot.SecondStage.replayWires P key with
  | none => exact List.nodup_nil
  | some ws =>
    unfold Spot.SecondStage.replayWires at hr
    obtain ⟨d, -, hr⟩ := Option.bind_eq_some_iff.1 hr
    exact wiresOf_keys hr

/-! ## `wiresChecked` at derived wires -/

/-- **No link names a wire's reader port**, decided on the auditor's copies: the first clause of `wiresChecked`. -/
def wiresUnlinked (h : Held) : Bool :=
  (h.wires.getD #[]).all fun w => (heldLink h w.template w.port).isNone

/-- **At derived wires, `wiresChecked` is its first clause**: at a held code whose wires are those of a registered
Definition (`SecondStage.replayWires`), as the two-stage form's own copies are, each wire is one the second stage's
check reads once no link names a wire's reader port (`wiresUnlinked`). The first wire at a wire's reader port is the
wire itself, since the derivation gives one wire a port (`replayWires_keys`). -/
theorem wiresChecked_of_unlinked {h : Held} {P : _root_.Flock.Program.Prog} {key : String}
    (hw : h.wires = Spot.SecondStage.replayWires P key) (hu : wiresUnlinked h = true) : wiresChecked h = true := by
  have hk : ((h.wires.getD #[]).toList.map fun w => (w.template, w.port)).Nodup := by
    rw [hw]
    exact replayWires_keys P key
  unfold wiresUnlinked at hu
  unfold wiresChecked
  refine Array.all_eq_true_iff_forall_mem.2 fun w hmem => ?_
  have hfind : heldWire h w.template w.port = some w := by
    unfold heldWire
    exact find?_of_keys (k := fun w : Spot.SecondStage.Wire => (w.template, w.port)) hk hmem (by simp)
      (fun b hb => by simpa using hb)
  show ((heldLink h w.template w.port).isNone &&
    (heldWire h w.template w.port).any fun w' => w'.writer == w.writer && w'.wport == w.wport) = true
  rw [hfind, Bool.and_eq_true]
  exact ⟨Array.all_eq_true_iff_forall_mem.1 hu w hmem, by simp⟩

/-- `wiresChecked` says no more than `wiresUnlinked` and that the first wire at each wire's reader port names its
writer port: it gives `wiresUnlinked` at any held code. -/
theorem wiresUnlinked_of_checked {h : Held} (hc : wiresChecked h = true) : wiresUnlinked h = true := by
  unfold wiresChecked at hc
  unfold wiresUnlinked
  refine Array.all_eq_true_iff_forall_mem.2 fun w hmem => ?_
  have hw : ((heldLink h w.template w.port).isNone &&
      (heldWire h w.template w.port).any fun w' => w'.writer == w.writer && w'.wport == w.wport) = true :=
    Array.all_eq_true_iff_forall_mem.1 hc w hmem
  rw [Bool.and_eq_true] at hw
  exact hw.1

end AddressRow

end SpotFlock
