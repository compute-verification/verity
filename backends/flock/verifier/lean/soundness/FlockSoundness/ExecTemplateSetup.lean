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

/-! ## A template circuit's nets -/

/-- **A net built from derived rows fits its slot, and its constant row reads itself.** -/
theorem netOfD_fit {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) :
    n.constPos < 2 ^ n.unitLog ∧ 7 ≤ n.unitLog ∧ (n.unitLog ≤ 32 → ConstRow n) := by
  obtain ⟨hl, -, -, -, hcst, hin, hca, hcb, -⟩ := Typed.netOfD_ok h
  have hlt := (rowsA_lt h d.const).1 d.const (by rw [hca]; simp)
  refine ⟨by rw [hcst]; exact hlt, by rw [hl]; exact le_max_left _ _,
    fun hu => ⟨by rw [hcst]; exact hin, fun j => ?_, fun j => ?_⟩⟩
  · rw [(netOfD_sparse h hu _ j).1, hcst, hca]
    by_cases hj : j = d.const
    · subst hj; simp
    · simp [hj, Ne.symm hj]
  · rw [(netOfD_sparse h hu _ j).2, netOfD_lookup h, hcst, hcb]
    by_cases hj : j = d.const
    · subst hj; simp [lookupEntry]
    · simp [hj, Ne.symm hj, lookupEntry]

/-- **A template circuit's nets**: the root first, then the placed layouts' circuits, then the file's own text nets and
lookup slots. -/
theorem tmpl_nets {tags : Tags} {t : Typed.Template} {c : Circuit} (P : ParseFactsT tags t c) {i : ℕ}
    (hi : i < c.nets.size) :
    (i = 0 ∧ (c.nets.getD i default).2 = t.root) ∨
      (∃ k, ∃ hk : k < t.layouts.size, i = 1 + k ∧ (c.nets.getD i default).2 = t.layouts[k].2.1) ∨
      (t.layouts.size < i ∧ ((∃ lines, Net.parse lines = .ok (c.nets.getD i default).2) ∨
        ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok (c.nets.getD i default).2)) := by
  obtain ⟨more, hm, hmore⟩ := P.pre
  have hget : c.nets.toList[i]? = some (c.nets.getD i default) := by
    rw [Array.getElem?_toList, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]; rfl
  rw [hm] at hget
  simp only [Array.toList_append, List.append_assoc, Array.toList_map] at hget
  rw [show [("root", t.root)] = [("root", t.root)] from rfl, List.singleton_append] at hget
  rcases Nat.eq_zero_or_pos i with rfl | hpos
  · left
    refine ⟨rfl, ?_⟩
    rw [List.getElem?_cons_zero] at hget
    rw [← Option.some.inj hget]
  · right
    obtain ⟨k, rfl⟩ : ∃ k, i = k + 1 := ⟨i - 1, by omega⟩
    rw [List.getElem?_cons_succ] at hget
    by_cases hk : k < t.layouts.size
    · left
      refine ⟨k, hk, by omega, ?_⟩
      rw [List.getElem?_append_left (by simpa using hk), List.getElem?_map, Array.getElem?_toList,
        Array.getElem?_eq_getElem hk] at hget
      simp only [Option.map_some, Option.some.injEq] at hget
      rw [← hget]
    · right
      refine ⟨by omega, hmore _ ?_⟩
      rw [List.getElem?_append_right (by simp; omega)] at hget
      exact List.mem_of_getElem? hget

section template

variable {name : String} {done : Array D} {ls : Array Flock.Layout} {unit : ℕ} {ty : CircuitType}
  {held : Array Typed.Held} {t : Typed.Template}

/-- **Placed layout `k`'s circuit is its derivation's net**: its layout is one a part places, at place `k` among them, and
its net is `netOfD` of that layout's rows, but for a generated read's lookup. -/
theorem layout_at (htm : Typed.templateOf name done ls unit ty held = .ok t) {k : ℕ} (hk : k < t.layouts.size) :
    ∃ j, (∃ p ∈ (done.getD unit default).parts, p.1 = j) ∧
      ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf j = k ∧
      ∃ n0 nm, Typed.netOfD (done.getD j default) nm = .ok n0 ∧ { t.layouts[k].2.1 with lookup := none } = n0 ∧
        ((ls.getD j default).own = .derive → t.layouts[k].2.1.lookup = none) := by
  obtain ⟨digests, -, -, -, -, -, -, -, -, -, -, hsize, hent⟩ := Typed.templateOf_spec htm
  have hk' : k < ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).length := by
    rw [← hsize]; exact hk
  have hmem := List.getElem_mem hk'
  rw [List.mem_filter] at hmem
  obtain ⟨p, hp, hpj⟩ := Array.any_eq_true'.1 hmem.2
  obtain ⟨-, -, hnet⟩ := Typed.placedEntry_ok (hent k hk' hk)
  obtain ⟨n0, h0, heq, hder, -⟩ := Typed.placedNet_spec hnet
  refine ⟨_, ⟨p, hp, by simpa using hpj⟩, List.Nodup.idxOf_getElem (List.nodup_range.filter _) k hk', n0, _, h0, heq,
    hder⟩

/-- **Part `p`'s layout is placed**, at place `k` among the placed layouts, and its circuit is its derivation's net. -/
theorem part_at (htm : Typed.templateOf name done ls unit ty held = .ok t) {p : ℕ × ℕ}
    (hp : p ∈ (done.getD unit default).parts) :
    ∃ k, ∃ hk : k < t.layouts.size,
      ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf p.1 = k ∧
      ∃ n0 nm, Typed.netOfD (done.getD p.1 default) nm = .ok n0 ∧ { t.layouts[k].2.1 with lookup := none } = n0 ∧
        ((ls.getD p.1 default).own = .derive → t.layouts[k].2.1.lookup = none) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hmem, hsize, -⟩ := Typed.templateOf_spec htm
  have hin := hmem p (by simpa using hp)
  have hlt := List.idxOf_lt_length_iff.2 hin
  have hk : ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf p.1 <
      t.layouts.size := by rw [hsize]; exact hlt
  obtain ⟨j, ⟨p', hp', hpj⟩, hidx, rest⟩ := layout_at htm hk
  have hj : j = p.1 := (List.idxOf_inj (hpj ▸ hmem p' (by simpa using hp'))).1 hidx
  subst hj
  exact ⟨_, hk, rfl, rest⟩

/-! ## The block's parts -/

theorem toList_eq_map_range {α : Type} [Inhabited α] (xs : Array α) :
    xs.toList = (List.range xs.size).map fun i => xs.getD i default := by
  apply List.ext_getElem (by simp)
  intro n h1 h2
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  simp [Array.getD, show n < xs.size by simpa using h1]

/-- The parts on circuit `1 + k` in the block are the template's parts on placed layout `k`. -/
theorem blockOf_filter (t : Typed.Template) (k : ℕ) :
    ((HmRow.blockOf t).parts.filter (·.1 == 1 + k)).size = (t.parts.toList.filter (·.1 == k)).length := by
  rw [← Array.length_toList, Array.toList_filter]
  show (((Array.range t.parts.size).map fun i => (1 + (t.parts.getD i default).1, HmRow.slotOf t.parts i,
    (t.parts.getD i default).2)).toList.filter (·.1 == 1 + k)).length = _
  rw [Array.toList_map, Array.toList_range, List.filter_map, List.length_map, toList_eq_map_range t.parts,
    List.filter_map, List.length_map]
  congr 2
  funext i
  simp [Function.comp]

theorem slot_lt_perVu {c : Circuit} (C : CheckFactsT c (HmRow.blockOf t)) {i : ℕ} (hi : i < t.parts.size)
    (hk : t.parts[i].1 < t.layouts.size) :
    HmRow.slotOf t.parts i < c.perVu.getD (1 + t.parts[i].1) 0 := by
  obtain ⟨-, -, -, -, -, -, -, -, hrl1, hrl⟩ := HmRow.blockOf_spec t
  have hk' : t.parts[i].1 < (HmRow.blockOf t).rangeLogs.size := by rw [hrl1]; exact hk
  have he := C.rangeLogs _ (Array.getElem_mem_toList hk')
  rw [hrl _ hk hk'] at he
  rw [he.1, blockOf_filter]
  exact HmRow.slotOf_lt hi

theorem part_slot_inj {i i' : ℕ} (hi : i < t.parts.size) (hi' : i' < t.parts.size) (h1 : t.parts[i].1 = t.parts[i'].1)
    (h2 : HmRow.slotOf t.parts i = HmRow.slotOf t.parts i') : i = i' := by
  rcases Nat.lt_trichotomy i i' with h | h | h
  · exact absurd h2 (Nat.ne_of_lt (HmRow.slotOf_lt_of_lt h hi' h1))
  · exact h
  · exact absurd h2.symm (Nat.ne_of_lt (HmRow.slotOf_lt_of_lt h hi h1.symm))

theorem with_lookup_none {n : Net} (h : n.lookup = none) : { n with lookup := none } = n := by
  cases n; cases h; rfl

/-- **Every net of an accepted template circuit fits its slot, and its constant row reads itself**: the root and the
placed layouts' nets are their derivations' (`netOfD`), the others are text nets or lookup slots. -/
theorem tmpl_net_facts {tags : Tags} {c : Circuit} (P : ParseFactsT tags t c)
    (htm : Typed.templateOf name done ls unit ty held = .ok t)
    (hder : ∀ p ∈ (done.getD unit default).parts, (ls.getD p.1 default).own = .derive) (i : ℕ) :
    (c.nets.getD i default).2.constPos < 2 ^ (c.nets.getD i default).2.unitLog ∧
      (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
        ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2)) := by
  by_cases hi : i < c.nets.size
  · have key : ∀ n : Net, (c.nets.getD i default).2 = n →
        n.constPos < 2 ^ n.unitLog ∧ 7 ≤ n.unitLog ∧ (n.unitLog ≤ 32 → ConstRow n) →
        (c.nets.getD i default).2.constPos < 2 ^ (c.nets.getD i default).2.unitLog ∧
          (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
            ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2)) := by
      intro n hn hf; rw [hn]; exact ⟨hf.1, fun _ => hf.2⟩
    rcases tmpl_nets P hi with ⟨-, hr⟩ | ⟨k, hk, -, hl⟩ | ⟨-, hp | ⟨_, _, _, _, _, hb⟩⟩
    · obtain ⟨-, -, hroot, -⟩ := Typed.templateOf_spec htm
      exact key _ hr (netOfD_fit hroot)
    · obtain ⟨j, ⟨p, hp, rfl⟩, -, n0, nm, h0, heq, hdl⟩ := layout_at htm hk
      rw [with_lookup_none (hdl (hder p hp))] at heq
      exact key _ (hl.trans heq) (netOfD_fit h0)
    · obtain ⟨lines, hp⟩ := hp
      obtain ⟨h1, h2, h3⟩ := netOK_parsed hp
      exact key _ rfl ⟨h1.1, h2, h3⟩
    · obtain ⟨h1, h2, h3⟩ := netOK_lookup hb
      exact key _ rfl ⟨h1.1, h2, h3⟩
  · have hd : (c.nets.getD i default).2 = default := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
      rfl
    rw [hd]
    exact ⟨by decide, fun h => absurd h hi⟩

/-! ## The accepted template statement's layout -/

/-- **`TemplateLayout` for a template statement, from its parse and check.** For the circuit `parse` accepted for the
template `t` that `templateOf` built from the class's derivation. -/
theorem templateLayout_of {tags : Tags} {c : Circuit} (P : ParseFactsT tags t c) (C : CheckFactsT c (HmRow.blockOf t))
    (htm : Typed.templateOf name done ls unit ty held = .ok t)
    (hder : ∀ p ∈ (done.getD unit default).parts, (ls.getD p.1 default).own = .derive)
    (hrows : t.layouts.size < c.shaNet ∧ t.layouts.size < c.hmNet) {st : Stmt} (hst : st.c = c)
    (hpin : HmRow.pin c = .ok st.pin) (hda : st.da = (HmRow.delta c st.pin).1) (hdb : st.db = (HmRow.delta c st.pin).2)
    (hm : st.m = c.kLog + st.nbl) (hk : c.kLog ≤ 26) :
    TemplateLayout st (HmRow.blockOf t) done (done.getD unit default) := by
  subst hst
  obtain ⟨digests, -, hroot, hown, hinc, -, -, hbind, hcross, hparts, hmem, hsize, -⟩ := Typed.templateOf_spec htm
  obtain ⟨hbown, -, -, hbin, hbbind, hbcross, hbsize, hbpart, -, -⟩ := HmRow.blockOf_spec t
  obtain ⟨more, hpre, -⟩ := P.pre
  have hnsz : st.c.nets.size = 1 + t.layouts.size + more.length := by
    rw [← Array.length_toList, hpre]; simp; omega
  have hnet := tmpl_net_facts P htm hder
  have hpu : ∀ i (hi : i < t.parts.size) (hu : i < (done.getD unit default).parts.size),
      t.parts[i] = (((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf
        (done.getD unit default).parts[i].1, (done.getD unit default).parts[i].2) := by
    intro i hi hu
    simp only [hparts, Array.getElem_map]
  have hsha : ∀ n, n = st.c.shaNet ∨ n = st.c.hmNet → NetOK (st.c.nets.getD n default).2 := by
    intro n hn
    have hnl : n < st.c.nets.size := by rcases hn with rfl | rfl; exact P.shaNet; exact P.hmNet
    have hgt : t.layouts.size < n := by rcases hn with rfl | rfl; exact hrows.1; exact hrows.2
    rcases tmpl_nets P hnl with ⟨h0, -⟩ | ⟨k, hk', h1, -⟩ | ⟨-, ⟨lines, hp⟩ | ⟨_, _, _, _, _, hb⟩⟩
    · omega
    · omega
    · exact (netOK_parsed hp).1
    · exact (netOK_lookup hb).1
  obtain ⟨rp, hrp, ip, hkp, hcp, hpv⟩ := pin_spec hpin
  have hip : ip < st.c.nets.size := P.kinds rp hrp ip hkp
  have hsl' : rp.slotLog = (st.c.nets.getD ip default).2.unitLog := ((C.bound_slotLog rp hrp).2 ip hkp).symm
  have hslk : rp.slotLog ≤ st.c.kLog := by
    have h1 := (C.bound_slotLog rp hrp).1
    have h2 := count_mul_le_spanH rp
    have h3 : 2 ^ rp.slotLog ≤ rp.count * 2 ^ rp.slotLog := Nat.le_mul_of_pos_left _ hcp
    have h4 : 2 ^ rp.slotLog ≤ 2 ^ st.c.kLog := by unfold Range.start at h1; omega
    exact (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).1 h4
  obtain ⟨h7, hcr⟩ := (hnet ip).2 hip
  have hparts_size : t.parts.size = (done.getD unit default).parts.size := by rw [hparts, Array.size_map]
  exact {
    typed := P.typed
    disj := C.disj
    bound := fun r hr => (C.bound_slotLog r hr).1
    slotLog := fun r hr i hi => ((C.bound_slotLog r hr).2 i hi).symm
    count := C.count
    one_range := C.one_range
    fit := fun i => (hnet i).1
    unitNet := P.unitNet
    root_perVu := C.root_perVu
    root := by
      refine ⟨name, ?_⟩
      rw [hroot]
      obtain ⟨-, h⟩ := (tmpl_nets P (i := 0) (by omega)).resolve_right (by
        rintro (⟨k, -, h, -⟩ | ⟨h, -⟩) <;> omega)
      show Except.ok t.root = Except.ok (st.c.nets.getD st.c.unitNet default).2
      rw [P.root, h]
    own := by rw [hbown, hown]
    inCols := by rw [hbin, hinc]
    bind := by rw [hbbind, hbind]
    cross := by rw [hbcross, hcross]
    parts_size := by rw [hbsize, hparts_size]
    part := fun i hi hu => by
      have hti : i < t.parts.size := by rw [hparts_size]; exact hu
      rw [hbpart i hti hi]
      obtain ⟨kk, hkk, hidx, n0, nm, h0, heq, hdl⟩ := part_at htm (Array.getElem_mem hu)
      have hti1 : t.parts[i].1 = kk := by rw [hpu i hti hu]; exact hidx
      have hti2 : t.parts[i].2 = (done.getD unit default).parts[i].2 := by rw [hpu i hti hu]
      rw [with_lookup_none (hdl (hder _ (Array.getElem_mem hu)))] at heq
      have hlay : (st.c.nets.getD (1 + kk) default).2 = t.layouts[kk].2.1 := by
        rcases tmpl_nets P (i := 1 + kk) (by omega) with ⟨h, -⟩ | ⟨k', hk', h, hl⟩ | ⟨h, -⟩
        · omega
        · have : k' = kk := by omega
          subst this
          exact hl
        · omega
      refine ⟨hti2, by rw [hti1]; omega, by rw [hti1, P.root]; omega, by rw [hti1]; omega, by rw [hti1]; omega,
        slot_lt_perVu C hti (by rw [hti1]; exact hkk), nm, ?_⟩
      rw [hti1, hlay, heq]
      exact h0
    part_inj := fun i hi i' hi' h1 h2 => by
      have hti : i < t.parts.size := by rw [← hbsize]; exact hi
      have hti' : i' < t.parts.size := by rw [← hbsize]; exact hi'
      rw [hbpart i hti hi, hbpart i' hti' hi'] at h1 h2
      simp only at h1 h2
      exact part_slot_inj hti hti' (by omega) h2
    shaNet := ⟨P.shaNet, by rw [P.root]; omega⟩
    hmNet := ⟨P.hmNet, by rw [P.root]; omega⟩
    delta := by
      rw [hda, hdb]
      exact delta_typed st.c st.pin P.typed
    deltaIn := fun x hx => deltaIn_rows P C hsha hx
    pin := ⟨rp, hrp, ip, hkp, hcp, hpv, hcr (by omega)⟩
    kLog := by omega
    kLog6 := by omega
    m := hm }

end template

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **The accepted template statement's layout, from `Stmt.setupH`** (`tags.typed`, a template read). Its circuit carries
the class `k` (`Circuit.cls`), and `TemplateLayout` holds for the class's unit. Two hypotheses remain:
- `hder`, T3's scope: every part places a derived layout, with no table read;
- `hrows`: the rows' circuits come after the root and the placed layouts' circuits, which `parse` finds by name
  (asked of flock-verifier as a `checkTyped` check). -/
theorem setupH_templateLayout (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (hder : ∀ p ∈ (k.done.getD k.unit default).parts, (k.ls.getD p.1 default).own = .derive)
    (hrows : t.layouts.size < st.c.shaNet ∧ t.layouts.size < st.c.hmNet) :
    st.c.cls = some k ∧ st.c.typed = some (HmRow.blockOf t) ∧
      TemplateLayout st (HmRow.blockOf t) k.done (k.done.getD k.unit default) := by
  obtain ⟨hp, hpin, hda, hdb, hm, hk⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t', c', hrt, hpc, hc⟩ | ⟨net, c', hrn, -, -⟩
  · cases hrt
    have P := parse_facts_tmpl hpc
    obtain ⟨ls', hcheck⟩ := P.check
    have C := check_facts_typed P.typed hcheck
    obtain ⟨name, ty, htm⟩ := Typed.read_template hr
    refine ⟨by rw [hc], by rw [hc]; exact P.typed, ?_⟩
    have P' : ParseFactsT tags t st.c := by
      rw [hc]
      exact ⟨P.pre, P.ports, P.unitNet, P.root, P.shaNet, P.hmNet, P.kinds, ⟨ls', hcheck⟩, P.typed, P.upv,
        P.leavesIn, P.wires, P.leafCuts⟩
    have C' : CheckFactsT st.c (HmRow.blockOf t) := by
      rw [hc]
      exact ⟨C.bound_slotLog, C.disj, C.unit_lookup, C.count, C.one_range, C.root_perVu, C.rangeLogs, C.root_zero,
        C.sha_in, C.hm_in, C.perVu_sha, C.perVu_hm⟩
    exact templateLayout_of P' C' htm hder hrows rfl hpin hda hdb hm hk
  · cases hrn

/-- **1e's facts at every VU of an accepted template statement** (T3): for the class `k` its circuit carries, the block
rows at `Typed.col c tb g c` are the unit's block row there, in the model of the accepted statement. -/
theorem setupH_blockFacts (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (hder : ∀ p ∈ (k.done.getD k.unit default).parts, (k.ls.getD p.1 default).own = .derive)
    (hrows : t.layouts.size < st.c.shaNet ∧ t.layouts.size < st.c.hmNet)
    (U : UnitShape k.done (k.done.getD k.unit default))
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g : ℕ} (hg : g < st.c.g) {one : ℕ}
    (hone : (k.done.getD k.unit default).rows.size ≤ one) :
    BlockFacts ((setupH_templateLayout ht h hr hder hrows).2.2.model regs) (Typed.wordsIn k.held) k.done
      (k.done.getD k.unit default) one (Typed.col st.c (HmRow.blockOf t) g) :=
  (setupH_templateLayout ht h hr hder hrows).2.2.blockFacts k.hd (unit_lt k.hd) U regs hg hone

end FlockSoundness
