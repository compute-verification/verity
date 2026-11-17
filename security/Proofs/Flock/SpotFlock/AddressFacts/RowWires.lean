module

public import Proofs.Flock.SpotFlock.AddressFacts.RowLeaves

/-! The derived wires are one a reader port (`replayWires_keys`), so there `wiresChecked` is its first clause, which
is not a fact of the derivation: it can derive a link and a wire of one port. -/

@[expose] public section

namespace SpotFlock

namespace AddressRow

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

theorem map_set_of_eq {α β : Type} {f : α → β} {a b : α} (hab : f b = f a) :
    ∀ (l : List α) (j : ℕ), l[j]? = some a → (l.set j b).map f = l.map f
  | [], _, h => by simp at h
  | c :: l, 0, h => by
    rw [List.getElem?_cons_zero] at h
    rw [List.set_cons_zero, List.map_cons, List.map_cons, hab, Option.some.inj h]
  | c :: l, j + 1, h => by
    rw [List.getElem?_cons_succ] at h
    rw [List.set_cons_succ, List.map_cons, List.map_cons, map_set_of_eq hab l j h]

theorem keys_set {α β : Type} {f : α → β} {xs : Array α} {j : ℕ} {a b : α} (ha : xs[j]? = some a)
    (hab : f b = f a) : (xs.set! j b).toList.map f = xs.toList.map f := by
  rw [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds]
  exact map_set_of_eq hab xs.toList j (by rw [Array.getElem?_toList]; exact ha)

theorem nodup_concat {β : Type} {l : List β} {k : β} (hl : l.Nodup) (hk : k ∉ l) : (l ++ [k]).Nodup := by
  rw [List.nodup_append]
  refine ⟨hl, List.pairwise_singleton _ k, ?_⟩
  intro a ha b hb hab
  rw [List.mem_singleton] at hb
  rw [hab, hb] at ha
  exact hk ha

theorem nodup_keys_map {α β γ : Type} {xs : Array α} {f : α → γ} {g : α → β} {k : β → γ}
    (h : (xs.toList.map f).Nodup) (hfk : ∀ a, k (g a) = f a) : ((xs.map g).toList.map k).Nodup := by
  have hm : (xs.map g).toList.map k = xs.toList.map f := by
    rw [Array.toList_map, List.map_map]
    exact List.map_congr_left fun a _ => hfk a
  rw [hm]
  exact h

theorem find?_of_keys {α β : Type} {k : α → β} {xs : Array α} {p : α → Bool} {a : α}
    (hk : (xs.toList.map k).Nodup) (ha : a ∈ xs) (hpa : p a = true) (hp : ∀ b, p b = true → k b = k a) :
    xs.find? p = some a := by
  cases hf : xs.find? p with
  | none => exact absurd hpa (Array.find?_eq_none.1 hf a ha)
  | some b =>
    exact congrArg some (eq_of_nodup_keys hk (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf))
      (Array.mem_toList_iff.2 ha) (hp b (Array.find?_some hf)))

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

theorem wireRuns_keys {defs : Array _root_.Flock.Program.Defn} {d : ℕ}
    {rs : Array (Spot.SecondStage.Wire × ℕ × Array Spot.SecondStage.Run)}
    (h : Spot.SecondStage.wireRuns defs d = some rs) : (rs.toList.map fun e => (e.1.template, e.1.port)).Nodup := by
  unfold Spot.SecondStage.wireRuns at h
  obtain ⟨D, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨ws, -, h⟩ := Option.bind_eq_some_iff.1 h
  exact gather_keys h

theorem wiresOf_keys {defs : Array _root_.Flock.Program.Defn} {d : ℕ} {ws : Array Spot.SecondStage.Wire}
    (h : Spot.SecondStage.wiresOf defs d = some ws) : (ws.toList.map fun w => (w.template, w.port)).Nodup := by
  unfold Spot.SecondStage.wiresOf at h
  obtain ⟨rs, hrs, h⟩ := Option.map_eq_some_iff.1 h
  subst h
  exact nodup_keys_map (wireRuns_keys hrs) fun _ => rfl

theorem replayWires_keys (P : _root_.Flock.Program.Prog) (key : String) :
    (((Spot.SecondStage.replayWires P key).getD #[]).toList.map fun w => (w.template, w.port)).Nodup := by
  cases hr : Spot.SecondStage.replayWires P key with
  | none => exact List.nodup_nil
  | some ws =>
    unfold Spot.SecondStage.replayWires at hr
    obtain ⟨d, -, hr⟩ := Option.bind_eq_some_iff.1 hr
    exact wiresOf_keys hr

def wiresUnlinked (h : Held) : Bool :=
  (h.wires.getD #[]).all fun w => (heldLink h w.template w.port).isNone

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
