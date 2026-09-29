import FlockSoundness.ExecTemplatePlace

/-!
# A typed template's layout, from `Stmt.setupH` (1e for templates, T3)

`ExecTemplatePlace.lean` places a template's block rows from `TemplateLayout`, the accepted statement's layout facts. This
file derives them from what `Stmt.setupH` accepted for a template (`setupH_spec_typed`, `parseTyped_spec`,
`parse_facts_tmpl`, `check_facts_typed`, `delta_typed`) and what `Typed.read` built it from (`read_template`,
`templateOf_spec`, `blockOf_spec`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

/-! ## The class's unit is one of its layouts -/

theorem forIn_yield_push {α β : Type} {f : α → Array β → Except String (ForInStep (Array β))}
    (hf : ∀ a b s, f a b = .ok s → ∃ x, s = .yield (b.push x)) :
    ∀ (l : List α) (b v : Array β), forIn l b f = .ok v → v.size = b.size + l.length
  | [], b, v, h => by rw [List.forIn_nil] at h; cases h; simp
  | a :: l, b, v, h => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := except_bind_ok h
    obtain ⟨x, rfl⟩ := hf a b s hs
    have h' : forIn l (b.push x) f = .ok v := h
    have := forIn_yield_push hf l _ v h'
    simp only [Array.size_push, List.length_cons] at this ⊢
    omega

/-- `deriveAll` derives one `D` per layout. -/
theorem deriveAll_size {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (h : deriveAll types ls info words unit = .ok done) : done.size = ls.size := by
  unfold deriveAll at h
  obtain ⟨r, hr, h⟩ := except_bind_ok h
  cases h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
  have := forIn_yield_push (fun a b s hs => by
    dsimp only at hs
    split at hs <;> (obtain ⟨d, -, hs⟩ := except_bind_ok hs; cases hs; exact ⟨d, rfl⟩)) _ _ _ hr
  simpa [Std.Legacy.Range.size] using this

theorem deriveChecked_deriveAll {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (h : deriveChecked types ls info words unit = .ok done) : deriveAll types ls info words unit = .ok done := by
  unfold deriveChecked at h
  cases hd : deriveAll types ls info words unit with
  | error e => simp [hd, bind, Except.bind] at h
  | ok done' =>
    simp only [hd, bind, Except.bind] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · cases h; rfl

theorem const_mem_order (done : Array D) (u : D) : u.const ∈ order done u := by
  simp [order, segment]

/-- **The class's unit is one of its layouts**: `deriveChecked`'s order holds the unit's constant, one of its rows. -/
theorem unit_lt {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) : unit < ls.size := by
  have hsz := deriveAll_size (deriveChecked_deriveAll hd)
  by_contra hge
  have hu : done.getD unit default = default := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
  have := (order_sound hd 0).2.1 _ (const_mem_order done _)
  rw [hu] at this
  exact absurd this (by decide)

/-! ## The rows' wires are into the rows' circuits -/

theorem rowSlots_all (ports : Array Port) :
    ∀ w ∈ (HmRow.rowSlots ports).2.toList, ∀ e ∈ w.toList, e.1 < (HmRow.rowSlots ports).1 := by
  unfold HmRow.rowSlots
  simp only [HmRow.PER_SLOT]
  rw [← Array.forIn_toList]
  have := forIn_id_inv_tail (fun (_ : List Port) (s : ℕ × Array (Array (ℕ × ℕ))) =>
      ∀ w ∈ s.2.toList, ∀ e ∈ w.toList, ∃ k, e = (k / 3, k % 3) ∧ k < s.1)
    (f := fun p s => pure (ForInStep.yield (s.1 + p.chunks,
      s.2.push (Array.map (fun k => ((s.1 + k) / 3, (s.1 + k) % 3)) (Array.range p.chunks)))))
    (fun p L s hs => ⟨_, rfl, ?_⟩) ports.toList (0, #[]) (by simp)
  · intro w hw e he
    obtain ⟨k, rfl, hlt⟩ := this w hw e he
    show k / 3 < (_ + 3 - 1) / 3
    omega
  · intro w hw e he
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw
    rcases hw with hw | rfl
    · obtain ⟨k, hk, hlt⟩ := hs w hw e he
      exact ⟨k, hk, by simp only; omega⟩
    · simp only [Array.toList_map, Array.toList_range, List.mem_map, List.mem_range] at he
      obtain ⟨k, hk, rfl⟩ := he
      exact ⟨s.1 + k, rfl, by simp only; omega⟩

/-- **The rows' own wires are into the rows' circuits**: each chain's `cv` into a `sha512x3` slot, each row's hm96 `cv`
into its hm96 slot. -/
theorem rowWires_dst (ports : Array Port) (hp : ∀ p ∈ ports.toList, 0 < p.chunks) (sha hm : ℕ) :
    ∀ w ∈ (HmRow.rowWires ports sha hm).toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < (HmRow.rowSlots ports).1) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size) := by
  have hall := rowSlots_all ports
  have hsz := (rowSlots_spec ports hp).1
  unfold HmRow.rowWires
  generalize HmRow.rowSlots ports = r at hall hsz ⊢
  obtain ⟨nSha, where_⟩ := r
  simp only at hall hsz ⊢
  refine forIn_id_inv_range (P := fun out : Array Wire => ∀ w ∈ out.toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size))
    (fun p _ hpl out hout => ⟨_, rfl, ?_⟩) (by simp)
  have hw : where_[p]! ∈ where_.toList := getElem_mem_toList' hpl
  intro w hw'
  simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw'
  rcases hw' with hw' | rfl
  · revert w hw'
    refine forIn_id_inv_range (P := fun out : Array Wire => ∀ w ∈ out.toList,
        (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size))
      (fun k _ hk out hout => ⟨_, rfl, fun w hw' => ?_⟩) hout
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw'
    rcases hw' with hw' | rfl
    · exact hout w hw'
    · exact Or.inl ⟨rfl, hall _ hw (where_[p]!)[k]! (getElem_mem_toList' hk)⟩
  · exact Or.inr ⟨rfl, by rw [← hsz]; exact hpl⟩

/-- **Δ's other flat rows, for a template, are input bits of the rows' circuits' slots**: no leaf map or cut, and the
wires are the rows' own chains (`parse_facts_tmpl`). -/
theorem deltaIn_rows {tags : Tags} {t : Typed.Template} {c : Circuit} {tb : TypedBlock} (P : ParseFactsT tags t c)
    (C : CheckFactsT c tb) (hok : ∀ n, n = c.shaNet ∨ n = c.hmNet → NetOK (c.nets.getD n default).2) {x : ℕ}
    (hx : DeltaIn c x) :
    ∃ n, (n = c.shaNet ∨ n = c.hmNet) ∧ ∃ s off, AtSlot c n s off x ∧
      off < (c.nets.getD n default).2.inWords * Flock.WORD := by
  have hbit : ∀ n, (n = c.shaNet ∨ n = c.hmNet) → ∀ g < c.g, ∀ i < c.perVu.getD n 0,
      ∀ q ∈ (portCols (c.nets.getD n default).2.inGroups).toList, ∀ t < q.2,
      ∃ n', (n' = c.shaNet ∨ n' = c.hmNet) ∧ ∃ s off, AtSlot c n' s off (c.slotCol n g i + q.1 + t) ∧
        off < (c.nets.getD n' default).2.inWords * Flock.WORD := by
    intro n hn g hg i hi q hq t ht
    have hnl : n < c.nets.size := by rcases hn with rfl | rfl; exact P.shaNet; exact P.hmNet
    obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range n hnl)
    have hc := C.count _ hm _ hk
    have hsl := ((C.bound_slotLog _ hm).2 n hk).symm
    obtain ⟨-, hW, hports⟩ := hok n hn
    have hq' := hports q hq
    refine ⟨n, hn, g * c.perVu.getD n 0 + i, q.1 + t, ⟨by rw [hc]; exact Rows.mul_add_lt hg hi, ?_, ?_⟩, by omega⟩
    · rw [hsl]; omega
    · unfold Circuit.slotCol; omega
  obtain ⟨hws, hw0⟩ := rowSlots_spec c.ports P.ports
  rcases hx with ⟨g, hg, p, hp, t, ht, rfl⟩ | ⟨g, hg, p, hp, t, ht, rfl⟩ | ⟨k, hk, j, hj, t, ht, rfl⟩ |
    ⟨g, hg, e, he, t, ht, rfl⟩ | ⟨g, hg, w, hw, t, ht, rfl⟩
  · -- a row's first cv bit
    obtain ⟨h1, h2⟩ := hw0 _ (getElem_mem_toList' hp)
    have hsz := C.sha_in.1
    have hidx : 2 * ((HmRow.rowSlots c.ports).2[p]!)[0]!.2 < 6 := by omega
    refine hbit _ (Or.inl rfl) g hg _ (by rw [C.perVu_sha]; exact h1) _
      (getElem_mem_toList' (by rw [hsz]; exact hidx)) t ?_
    rw [C.sha_in.2 _ hidx, ite_eq_left (by simp)]
    exact ht
  · -- an hm96 slot's pad bit
    have hl : (portCols (c.nets.getD c.hmNet default).2.inGroups).toList.map (·.2) =
        [512, 1024, 8 * HmRow.SALT_BYTES] := by
      rw [← Array.toList_map, C.hm_in]
    have hsz : (portCols (c.nets.getD c.hmNet default).2.inGroups).size = 3 := by
      have := congrArg List.length hl; simpa using this
    have hw : (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.2 = 1024 := by
      rw [getElem!_pos _ 1 (by rw [hsz]; omega)]
      have := congrArg (fun l : List ℕ => l[1]?) hl
      simp only [List.getElem?_map, Array.getElem?_toList] at this
      rw [Array.getElem?_eq_getElem (by rw [hsz]; omega)] at this
      simpa using this
    refine hbit _ (Or.inr rfl) g hg p ?_ _ (getElem_mem_toList' (by rw [hsz]; omega)) t (by rw [hw]; exact ht)
    rw [C.perVu_hm, ← hws]
    exact lt_of_lt_of_le hp (le_nextPow2 _)
  · -- no leaf map
    rw [P.leavesIn, P.upv, Nat.mod_one] at hj
    exact absurd hj (by simp)
  · -- no leaf cut
    rw [P.leafCuts] at he
    cases he
  · -- the rows' own wires
    rw [P.wires] at hw
    have hwd := rowWires_dst c.ports P.ports c.shaNet c.hmNet w hw
    by_cases hq : w.dst.2.2 < (portCols (c.nets.getD w.dst.1 default).2.inGroups).size
    · have hqm := getD_mem_toList (d := default) hq
      rcases hwd with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · rw [h1] at hqm ht ⊢
        exact hbit _ (Or.inl rfl) g hg _ (by rw [C.perVu_sha]; exact h2) _ hqm t ht
      · rw [h1] at hqm ht ⊢
        refine hbit _ (Or.inr rfl) g hg _ ?_ _ hqm t ht
        rw [C.perVu_hm]
        exact lt_of_lt_of_le h2 (le_nextPow2 _)
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at ht
      exact absurd ht (show ¬ t < 0 by omega)

end FlockSoundness
