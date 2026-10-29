import Proofs.Flock.Soundness.ExecTemplatePlace

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

/-- **The unit's inputs are self rows with no Δ entry**: its own check (`checkLayout`, the unit's `Chk.rowIs`). -/
theorem unit_inputs {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) :
    ∀ x ∈ (done.getD unit default).inCols, (done.getD unit default).rows[x]? = some ([x], [x]) ∧
      (done.getD unit default).delta.find? (·.1 == x) = none := by
  have h := (deriveChecked_spec hd).1 unit (unit_lt hd)
  cases hown : (ls.getD unit default).own with
  | derive =>
    cases ht : (ls.getD unit default).type.bind types with
    | none => simp only [checkLayout, hown, ht, Bool.false_eq_true] at h
    | some t =>
      obtain ⟨-, -, hin, -, -, hins, -⟩ := checkLayout_spec h ht
      intro x hx
      obtain ⟨w, hw, rfl⟩ := Array.mem_iff_getElem.mp hx
      have := hins w (by omega)
      have hgd : (done.getD unit default).inCols.getD w 0 = (done.getD unit default).inCols[w] :=
        getD_of_getElem? (Array.getElem?_eq_getElem hw)
      rw [hgd] at this
      simp only [Chk.rowIs, beq_self_eq_true, Bool.not_true, Bool.false_or, Bool.and_eq_true, beq_iff_eq,
        Option.isNone_iff_eq_none] at this
      exact ⟨this.1.1, this.2⟩
  | gen sha n vb lo => exact absurd rfl (genLayout_spec h hown).1

/-- **`UnitShape` with its input field discharged**: the unit's inputs carry no Δ entry by its own check. -/
theorem unitShape_of {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done)
    (hcr : (done.getD unit default).rows.getD (done.getD unit default).const ([], []) =
      ([(done.getD unit default).const], [(done.getD unit default).const]))
    (ho : ∀ c ∈ order done (done.getD unit default), InUnit done (done.getD unit default) c)
    (hpr : ∀ p ∈ (done.getD unit default).parts, ∀ c < (done.getD p.1 default).size,
      (done.getD unit default).reads.find? (·.covers (p.2 + c)) =
        ((done.getD p.1 default).reads.find? (·.covers c)).map (·.shift p.2))
    (hcp : ∀ p ∈ (done.getD unit default).parts, ∀ c < (done.getD p.1 default).size, ∀ r,
      (done.getD p.1 default).reads.find? (·.covers c) = some r →
        ((done.getD p.1 default).rows.getD c ([], [])).2 = [] ∧
          ∀ l < 2 ^ r.lo, r.loTop + l < (done.getD p.1 default).size) :
    UnitShape done (done.getD unit default) :=
  ⟨hcr, ho, fun x hx => (unit_inputs hd x hx).2, hpr, hcp⟩

/-! ## The tables the class holds -/

/-- **A held table's words are its bytes, read as `u32`** (`hold`). -/
theorem hold_words {types : Array (String × CircuitType)} {tables : List (String × ByteArray)} {held : Array Typed.Held}
    (h : Typed.hold types tables = .ok held) :
    ∀ x ∈ held.toList, x.words = (Array.range (x.bytes.size / 4)).map fun i => rd32 x.bytes (4 * i) := by
  unfold Typed.hold at h
  obtain ⟨out, hout, h⟩ := except_bind_ok h
  cases h
  rw [← Array.forIn_toList] at hout
  refine forIn_except_inv (fun out : Array Typed.Held => ∀ x ∈ out.toList,
      x.words = (Array.range (x.bytes.size / 4)).map fun i => rd32 x.bytes (4 * i)) _ ?_ _ _ (by simp) hout
  intro a _ b hb s hs
  obtain ⟨_, t⟩ := a
  obtain ⟨o, ho, hs⟩ := except_bind_ok hs
  cases hs
  refine ⟨_, Or.inl rfl, ?_⟩
  rw [← Array.forIn_toList] at ho
  refine forIn_except_inv (fun out : Array Typed.Held => ∀ x ∈ out.toList,
      x.words = (Array.range (x.bytes.size / 4)).map fun i => rd32 x.bytes (4 * i)) _ ?_ _ _ hb ho
  intro e _ b hb s hs
  obtain ⟨sha, _, _⟩ := e
  dsimp only at hs
  split at hs
  · cases hs; exact ⟨_, Or.inl rfl, hb⟩
  · split at hs
    · split at hs
      · rename_i bytes _
        split at hs
        · simp [Functor.map, Except.map, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hs
        · simp only [Functor.map, Except.map, pure, Except.pure, Except.ok.injEq] at hs
          subst hs
          refine ⟨_, Or.inl rfl, fun x hx => ?_⟩
          simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hx
          rcases hx with hx | rfl
          · exact hb x hx
          · rfl
      · simp [Functor.map, Except.map, throw, throwThe, MonadExceptOf.throw] at hs
    · simp [Functor.map, Except.map, throw, throwThe, MonadExceptOf.throw] at hs

set_option maxHeartbeats 1000000 in
/-- **What `classOf` returns**: its class holds the tables `hold` read, and its type is the unit layout's. -/
theorem classOf_facts {mj : Lean.Json} {tables : List (String × ByteArray)} {kt : Typed.Class × CircuitType}
    (h : Typed.classOf mj tables = .ok kt) :
    (∃ types, Typed.hold types tables = .ok kt.1.held) ∧
      (kt.1.ls.getD kt.1.unit default).type.bind (Typed.typeIn kt.1.types) = some kt.2 := by
  unfold Typed.classOf at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  split at h
  swap
  · obtain ⟨_, hx, -⟩ := except_bind_ok h; cases hx
  simp (config := { zeta := false }) only [jp0] at h
  obtain ⟨kvs, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) types0 at h
  obtain ⟨s1, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) types1 at h
  obtain ⟨_, -, h⟩ := except_bind_ok h
  obtain ⟨held, hheld, h⟩ := except_bind_ok h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨unit, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  split at h
  · obtain ⟨_, hx, -⟩ := except_bind_ok h; cases hx
  simp (config := { zeta := false }) only [jp1] at h
  split at h
  swap
  · cases h
  rename_i ul hul
  split at h
  swap
  · cases h
  rename_i ut hut
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp5] at h
  split at h
  swap
  · cases h
  rename_i ty hty
  split at h
  · cases h
    refine ⟨⟨types1, hheld⟩, ?_⟩
    show (ls.getD unit default).type.bind (Typed.typeIn types1) = some ty
    rw [Array.getD_eq_getD_getElem?, hul, Option.getD_some, hut]
    exact hty
  · cases h

theorem classOf_held {mj : Lean.Json} {tables : List (String × ByteArray)} {kt : Typed.Class × CircuitType}
    (h : Typed.classOf mj tables = .ok kt) : ∃ types, Typed.hold types tables = .ok kt.1.held :=
  (classOf_facts h).1

/-- **A template read's block is `templateOf` on the unit layout's own type.** -/
theorem read_template_ty {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {k : Typed.Class}
    {t : Typed.Template} (h : Typed.read tags bytes tables = .ok (k, .inr t)) :
    ∃ name ty, Typed.templateOf name k.done k.ls k.unit ty k.held = .ok t ∧
      (k.ls.getD k.unit default).type.bind (Typed.typeIn k.types) = some ty := by
  unfold Typed.read at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  rename_i s _
  split at h
  · simp at h
  rename_i kt hkt
  have hty := (classOf_facts hkt).2
  split at h
  · split at h <;> simp at h
  · split at h
    · simp at h
    · rename_i v hv
      simp only [Except.ok.injEq, Prod.mk.injEq, Sum.inr.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨s.2.1, kt.2, hv, hty⟩

/-- **The class `Typed.read` returns holds its tables' words as their bytes, read as `u32`.** -/
theorem read_held {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {k : Typed.Class}
    {r : Net ⊕ Typed.Template} (h : Typed.read tags bytes tables = .ok (k, r)) :
    ∀ x ∈ k.held.toList, x.words = (Array.range (x.bytes.size / 4)).map fun i => rd32 x.bytes (4 * i) := by
  unfold Typed.read at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  split at h
  · simp at h
  rename_i kt hkt
  obtain ⟨types, hh⟩ := classOf_held hkt
  have hk : k = kt.1 := by
    split at h
    · split at h
      · simp at h
      · simp only [Except.ok.injEq, Prod.mk.injEq] at h; exact h.1.symm
    · split at h
      · simp at h
      · simp only [Except.ok.injEq, Prod.mk.injEq] at h; exact h.1.symm
  subst hk
  exact hold_words hh

/-! ## The rows' wires are into the rows' circuits -/

/-- **The rows' own wires are into the rows' circuits**: each chain's `cv` into a `sha512x3` slot, each row's hm96 `cv`
into its hm96 slot. -/
theorem rowWires_dst (ports : Array Port) (sha hm : ℕ) :
    ∀ w ∈ (HmRow.rowWires ports sha hm).toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < (HmRow.rowSlots ports).1) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size) := by
  have hall := rowSlots_all ports
  have hsz := rowSlots_spec ports
  unfold HmRow.rowWires
  generalize HmRow.rowSlots ports = r at hall hsz ⊢
  obtain ⟨nSha, where_⟩ := r
  simp only at hall hsz ⊢
  refine forIn_id_inv_range (P := fun out : Array Wire => ∀ w ∈ out.toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size))
    (fun p _ hpl out hout => ?_) (by simp)
  have hw : where_[p]! ∈ where_.toList := getElem_mem_toList' hpl
  have hchain : ∀ k, 1 ≤ k → k < (where_[p]!).size → ∀ out : Array Wire, (∀ w ∈ out.toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size)) →
      ∀ w ∈ (out.push ⟨(sha, (where_[p]!)[k]!.1, 2 * (where_[p]!)[k]!.2),
        (sha, (where_[p]!)[k - 1]!.1, (where_[p]!)[k - 1]!.2)⟩).toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size) := by
    intro k _ hk out hout w hw'
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw'
    rcases hw' with hw' | rfl
    · exact hout w hw'
    · exact Or.inl ⟨rfl, (hall _ hw (where_[p]!)[k]! (getElem_mem_toList' hk)).1⟩
  have hinner : ∀ k, 1 ≤ k → k < (where_[p]!).size → ∀ out : Array Wire, (∀ w ∈ out.toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size)) →
      ∃ out', (if (HmRow.chainStart ports[p]! k).isNone = true then
          ForInStep.yield (out.push ⟨(sha, (where_[p]!)[k]!.1, 2 * (where_[p]!)[k]!.2),
            (sha, (where_[p]!)[k - 1]!.1, (where_[p]!)[k - 1]!.2)⟩)
        else ForInStep.yield out) = ForInStep.yield out' ∧ ∀ w ∈ out'.toList,
      (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size) := by
    intro k hk1 hk out hout
    split
    · exact ⟨_, rfl, hchain k hk1 hk out hout⟩
    · exact ⟨_, rfl, hout⟩
  by_cases h0 : (0 < (where_[p]!).size && (HmRow.chainStart ports[p]! (where_[p]!).size).isNone) = true
  · refine ⟨_, by simp only [h0, ↓reduceIte]; rfl, fun w hw' => ?_⟩
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw'
    rcases hw' with hw' | rfl
    · revert w hw'
      exact forIn_id_inv_range (P := fun out : Array Wire => ∀ w ∈ out.toList,
          (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size))
        (fun k hk1 hk out hout => hinner k hk1 hk out hout) hout
    · exact Or.inr ⟨rfl, by rw [← hsz]; exact hpl⟩
  · exact ⟨_, by simp only [h0, Bool.false_eq_true, ↓reduceIte]; rfl, forIn_id_inv_range (P := fun out : Array Wire => ∀ w ∈ out.toList,
        (w.dst.1 = sha ∧ w.dst.2.1 < nSha) ∨ (w.dst.1 = hm ∧ w.dst.2.1 < ports.size))
      (fun k hk1 hk out hout => hinner k hk1 hk out hout) hout⟩

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
  have hws := rowSlots_spec c.ports
  have hall := rowSlots_all c.ports
  rcases hx with ⟨g, hg, p, hp, k, hk, i, hi, t, ht, rfl⟩ | ⟨g, hg, p, hp, i, hi, t, ht, rfl⟩ | ⟨k, hk, j, hj, t, ht, rfl⟩ |
    ⟨g, hg, e, he, t, ht, rfl⟩ | ⟨g, hg, w, hw, t, ht, rfl⟩
  · -- a bit of a row's chain
    obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hp) _ (getElem_mem_toList' hk)
    have hsz := C.sha_in.1
    have hidx : 2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + i < 6 := by omega
    refine hbit _ (Or.inl rfl) g hg _ (by rw [C.perVu_sha]; exact h1) _
      (getElem_mem_toList' (by rw [hsz]; exact hidx)) t ?_
    rw [C.sha_in.2 _ hidx, sha_width _ _ hi]
    exact ht
  · -- a bit of a row's hm96 slot
    obtain ⟨hil, hw⟩ := hm_width C.hm_in hi
    refine hbit _ (Or.inr rfl) g hg p ?_ _ (getElem_mem_toList' hil) t (by rw [hw]; exact ht)
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
    have hwd := rowWires_dst c.ports c.shaNet c.hmNet w hw
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

/-! ## A placed layout's net -/

theorem placedNet_derive {l : Flock.Layout} {d : D} {nm : String} {held : Array Typed.Held} {n : Net}
    (h : Typed.placedNet l d nm held = .ok n) (hown : l.own = .derive) : d.reads = #[] := by
  unfold Typed.placedNet at h
  rw [hown] at h
  simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · simp at h
  · rename_i hr
    simpa using hr

/-- **A generated layout's net carries its own read and held table**: `placedNet` checks the read is its record's. -/
theorem placedNet_gen {l : Flock.Layout} {d : D} {nm : String} {held : Array Typed.Held} {n : Net}
    (h : Typed.placedNet l d nm held = .ok n) {sha : String} {nb vb lo : ℕ} (hown : l.own = .gen sha nb vb lo) :
    ∃ r hd, d.reads.toList = [r] ∧ r.table = sha ∧ r.n = nb ∧ r.lo = lo ∧ held.find? (·.sha512 == sha) = some hd ∧
      n.lookup = some ⟨hd.bytes, r.n, r.lo, r.k, r.prod0, r.loTop⟩ := by
  unfold Typed.placedNet at h
  rw [hown] at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · rename_i r hr
    split at h
    · simp at h
    · rename_i hc
      split at h
      · rename_i hd hhd
        split at h
        · simp at h
        · rename_i n0 hn0
          simp only [Except.ok.injEq] at h
          subst h
          simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hc
          exact ⟨r, hd, hr, hc.1.1, hc.1.2, hc.2, hhd, rfl⟩
      · simp at h
  · simp at h

/-- **A generated layout's own check**: its table has `2^n` words, `lo < n`, and no read covers its constant. -/
theorem gen_facts {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {j : ℕ} (hj : j < ls.size) {sha : String} {nb vb lo : ℕ}
    (hown : (ls.getD j default).own = .gen sha nb vb lo) :
    ((words sha).getD #[]).size = 2 ^ nb ∧ lo < nb ∧
      (done.getD j default).reads.find? (·.covers (done.getD j default).const) = none := by
  have h := (deriveChecked_spec hd).1 j hj
  obtain ⟨-, hg⟩ := genLayout_spec h hown
  obtain ⟨-, -, hone, hws, -, hlt, -⟩ := genOk_facts hg
  refine ⟨hws, hlt, ?_⟩
  simp only [Chk.oneAt, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, Option.isNone_iff_eq_none] at hone
  exact hone.2

/-- **A placed layout's net fits its slot, and its constant row reads itself**, a generated read's lookup included. -/
theorem placedNet_fit {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
    {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {j : ℕ} (hj : j < ls.size) {nm : String}
    {held : Array Typed.Held} {n : Net} (h : Typed.placedNet (ls.getD j default) (done.getD j default) nm held = .ok n) :
    n.constPos < 2 ^ n.unitLog ∧ 7 ≤ n.unitLog ∧ (n.unitLog ≤ 32 → ConstRow n) := by
  obtain ⟨n0, h0, heq, hdl, -⟩ := Typed.placedNet_spec h
  subst heq
  obtain ⟨h1, h2, h3⟩ := netOfD_fit h0
  refine ⟨h1, h2, fun hu => ?_⟩
  obtain ⟨c1, c2, c3⟩ := h3 hu
  refine ⟨c1, c2, fun j' => ?_⟩
  have hc3 := c3 j'
  simp only [lookupEntry, add_zero] at hc3
  cases hown : (ls.getD j default).own with
  | derive => rw [hdl hown]; simpa [lookupEntry, Net.b] using hc3
  | gen sha nb vb lo =>
    obtain ⟨r, hdh, hr, -, -, -, -, hlk⟩ := placedNet_gen h hown
    obtain ⟨-, -, hnc⟩ := gen_facts hd hj hown
    have hcv : r.covers n.constPos = false := by
      have : (done.getD j default).const = n.constPos := (Typed.netOfD_ok h0).2.2.2.2.1.symm
      rw [← this]
      rw [← Array.find?_toList, hr] at hnc
      simpa using hnc
    rw [hlk, lookupEntry_uncovered hcv, add_zero]
    exact hc3

/-- **No read covers a template unit's own rows**: `templateOf` refuses one in its own region. -/
theorem templateOf_reads_past {name : String} {done : Array D} {ls : Array Flock.Layout} {unit : ℕ} {ty : CircuitType}
    {held : Array Typed.Held} {t : Typed.Template} (h : Typed.templateOf name done ls unit ty held = .ok t) :
    ∀ r ∈ (done.getD unit default).reads.toList, (done.getD unit default).size ≤ r.prod0 := by
  intro r hr
  unfold Typed.templateOf at h
  simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · simp at h
  · rename_i hany
    by_contra hlt
    exact hany (Array.any_eq_true'.2 ⟨r, by simpa using hr, by simpa using hlt⟩)


section template

variable {name : String} {done : Array D} {ls : Array Flock.Layout} {unit : ℕ} {ty : CircuitType}
  {held : Array Typed.Held} {t : Typed.Template}

/-- **Placed layout `k`'s circuit is its derivation's net**: its layout is one a part places, at place `k` among them, and
its net is `netOfD` of that layout's rows, but for a generated read's lookup. -/
theorem layout_at (htm : Typed.templateOf name done ls unit ty held = .ok t) {k : ℕ} (hk : k < t.layouts.size) :
    ∃ j, j < ls.size ∧ (∃ p ∈ (done.getD unit default).parts, p.1 = j) ∧
      ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf j = k ∧
      ∃ nm, Typed.placedNet (ls.getD j default) (done.getD j default) nm held = .ok t.layouts[k].2.1 := by
  obtain ⟨digests, -, -, -, -, -, -, -, -, -, -, hsize, hent⟩ := Typed.templateOf_spec htm
  have hk' : k < ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).length := by
    rw [← hsize]; exact hk
  have hmem := List.getElem_mem hk'
  rw [List.mem_filter] at hmem
  obtain ⟨p, hp, hpj⟩ := Array.any_eq_true'.1 hmem.2
  obtain ⟨-, -, hnet⟩ := Typed.placedEntry_ok (hent k hk' hk)
  exact ⟨_, List.mem_range.1 hmem.1, ⟨p, hp, by simpa using hpj⟩,
    List.Nodup.idxOf_getElem (List.nodup_range.filter _) k hk', _, hnet⟩

/-- **Part `p`'s layout is placed**, at place `k` among the placed layouts, and its circuit is its derivation's net. -/
theorem part_at (htm : Typed.templateOf name done ls unit ty held = .ok t) {p : ℕ × ℕ}
    (hp : p ∈ (done.getD unit default).parts) :
    ∃ k, ∃ hk : k < t.layouts.size,
      ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf p.1 = k ∧
      p.1 < ls.size ∧ ∃ nm, Typed.placedNet (ls.getD p.1 default) (done.getD p.1 default) nm held = .ok t.layouts[k].2.1 := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hmem, hsize, -⟩ := Typed.templateOf_spec htm
  have hin := hmem p (by simpa using hp)
  have hlt := List.idxOf_lt_length_iff.2 hin
  have hk : ((List.range ls.size).filter fun j => (done.getD unit default).parts.any (·.1 == j)).idxOf p.1 <
      t.layouts.size := by rw [hsize]; exact hlt
  obtain ⟨j, hjl, ⟨p', hp', hpj⟩, hidx, rest⟩ := layout_at htm hk
  have hj : j = p.1 := (List.idxOf_inj (hpj ▸ hmem p' (by simpa using hp'))).1 hidx
  subst hj
  exact ⟨_, hk, rfl, hjl, rest⟩

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

/-- **A part's placed net is a `PartNet`**: its derivation's rows, and a generated read's lookup from the table the
class holds for it. -/
theorem partNet_of {types : String → Option CircuitType} {info : String → Option TableInfo}
    (hd : deriveChecked types ls info (Typed.wordsIn held) unit = .ok done)
    (hheld : ∀ h ∈ held.toList, h.words = (Array.range (h.bytes.size / 4)).map fun i => rd32 h.bytes (4 * i))
    {j : ℕ} (hj : j < ls.size) {nm : String} {n : Net}
    (h : Typed.placedNet (ls.getD j default) (done.getD j default) nm held = .ok n) :
    PartNet (Typed.wordsIn held) (done.getD j default) n := by
  obtain ⟨n0, h0, heq, hdl, -⟩ := Typed.placedNet_spec h
  refine ⟨⟨nm, by rw [heq]; exact h0⟩, ?_⟩
  cases hown : (ls.getD j default).own with
  | derive => exact Or.inl ⟨hdl hown, placedNet_derive h hown⟩
  | gen sha nb vb lo =>
    obtain ⟨r, hdh, hr, htab, hn, hlo, hfind, hlk⟩ := placedNet_gen h hown
    obtain ⟨hws, hlt, -⟩ := gen_facts hd hj hown
    have hwr : (Typed.wordsIn held r.table).getD #[] = hdh.words := by
      simp only [Typed.wordsIn, htab, hfind, Option.map_some, Option.getD_some]
    refine Or.inr ⟨r, hdh.bytes, hr, hlk, ?_, ?_, by omega⟩
    · rw [hwr]; exact hheld hdh (by simpa using Array.mem_of_find?_eq_some hfind)
    · rw [htab, hn]; exact hws

/-- **Every net of an accepted template circuit fits its slot, and its constant row reads itself**: the root and the
placed layouts' nets are their derivations' (`netOfD`), the others are text nets or lookup slots. -/
theorem tmpl_net_facts {tags : Tags} {c : Circuit} (P : ParseFactsT tags t c)
    (htm : Typed.templateOf name done ls unit ty held = .ok t) {types : String → Option CircuitType}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)}
    (hd : deriveChecked types ls info words unit = .ok done) (i : ℕ) :
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
    · obtain ⟨j, hj, -, -, nm, hpl⟩ := layout_at htm hk
      exact key _ hl (placedNet_fit hd hj hpl)
    · obtain ⟨lines, hp⟩ := hp
      obtain ⟨h1, h2, h3⟩ := netOK_built (parse_built hp)
      exact key _ rfl ⟨h1.1, h2, h3⟩
    · obtain ⟨h1, h2, h3⟩ := netOK_lookup hb
      exact key _ rfl ⟨h1.1, h2, h3⟩
  · have hd : (c.nets.getD i default).2 = default := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
      rfl
    rw [hd]
    exact ⟨by decide, fun h => absurd h hi⟩

/-! ## An input's message bit is a column of the block -/

theorem nested_getElem! {α : Type} [Inhabited α] (xs : Array (Array α)) (p k : ℕ) :
    (xs[p]!)[k]! = default ∨ ∃ w ∈ xs.toList, (xs[p]!)[k]! ∈ w.toList := by
  by_cases hp : p < xs.size
  · rw [getElem!_pos xs p hp]
    by_cases hk : k < xs[p].size
    · exact Or.inr ⟨xs[p], Array.getElem_mem_toList hp, by rw [getElem!_pos _ k hk]; exact Array.getElem_mem_toList hk⟩
    · exact Or.inl (getElem!_neg _ k hk)
  · left
    rw [getElem!_neg xs p hp]
    exact getElem!_neg _ k (by show ¬ k < (#[] : Array α).size; simp)

/-- **The rows' circuits fit their slots and their input ports lie in their input rows**: they are the file's own text
nets or lookup slots, past the template's (`CheckFactsT.rows_after`). -/
theorem rows_netOK {tags : Tags} {t : Typed.Template} {c : Circuit} (P : ParseFactsT tags t c)
    (C : CheckFactsT c (HmRow.blockOf t)) : ∀ n, n = c.shaNet ∨ n = c.hmNet → NetOK (c.nets.getD n default).2 := by
  obtain ⟨-, -, -, -, -, -, -, -, hbrl, -⟩ := HmRow.blockOf_spec t
  have hrows : t.layouts.size < c.shaNet ∧ t.layouts.size < c.hmNet := by
    have := C.rows_after; rw [hbrl] at this; exact this
  intro n hn
  have hnl : n < c.nets.size := by rcases hn with rfl | rfl; exact P.shaNet; exact P.hmNet
  have hgt : t.layouts.size < n := by rcases hn with rfl | rfl; exact hrows.1; exact hrows.2
  rcases tmpl_nets P hnl with ⟨h0, -⟩ | ⟨k, hk', h1, -⟩ | ⟨-, ⟨lines, hp⟩ | ⟨_, _, _, _, _, hb⟩⟩
  · omega
  · omega
  · exact (netOK_built (parse_built hp)).1
  · exact (netOK_lookup hb).1

theorem starts_size (c : Circuit) : (Typed.starts c).size = c.ports.size := by
  unfold Typed.starts
  rw [← Array.foldl_toList]
  suffices h : ∀ (L : List Port) (acc : Array ℕ × ℕ),
      (L.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) acc).1.size = acc.1.size + L.length by
    simpa using h c.ports.toList (#[], 0)
  intro L
  induction L with
  | nil => intro acc; simp
  | cons p L ih => intro acc; rw [List.foldl_cons, ih]; simp only [Array.size_push, List.length_cons]; omega

/-- **The port a row word's leaf is in** (`copySrc`'s, as `HmRow.delta` finds it) is one of the file's ports, when it has
one. -/
theorem portAt_lt (c : Circuit) (hne : 0 < c.ports.size) (x : ℕ) :
    ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x)).getD 0 < c.ports.size := by
  rw [← starts_size c] at hne ⊢
  cases hf : (List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x) with
  | none => exact hne
  | some q => exact List.mem_range.1 (List.mem_reverse.1 (List.mem_of_find?_eq_some hf))

/-- **A message bit's column is in the block**: for a row of the file, any byte and bit, it is a column of a `sha512x3`
slot's message port or, past the row's chain, of its hm96 slot's `pad`. -/
theorem msgCol_lt {tags : Tags} {t : Typed.Template} {c : Circuit} (P : ParseFactsT tags t c)
    (C : CheckFactsT c (HmRow.blockOf t)) {g : ℕ} (hg : g < c.g) {p : ℕ} (hp : p < c.ports.size) (byte b : ℕ)
    (hb : b < 8) : Typed.msgCol c g p byte b < 2 ^ c.kLog := by
  have hws := rowSlots_spec c.ports
  have hall := rowSlots_all c.ports
  have hpad : ∀ n, (n = c.shaNet ∨ n = c.hmNet) → ∀ s < c.perVu.getD n 0,
      ∀ i < (portCols (c.nets.getD n default).2.inGroups).size,
      ((portCols (c.nets.getD n default).2.inGroups)[i]!).2 = 1024 →
      c.slotCol n g s + ((portCols (c.nets.getD n default).2.inGroups)[i]!).1 + 64 * (byte % 128 / 8) + 56 -
        8 * (byte % 128 % 8) + b < 2 ^ c.kLog := by
    intro n hn s hs i hi hq
    have hnl : n < c.nets.size := by rcases hn with rfl | rfl; exact P.shaNet; exact P.hmNet
    obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range _ hnl)
    have hc := C.count _ hm _ hk
    have hsl := ((C.bound_slotLog _ hm).2 _ hk).symm
    obtain ⟨-, hW, hports⟩ := rows_netOK P C _ hn
    have hqm := hports _ (getElem_mem_toList' hi)
    have h1 : 64 * (byte % 128 / 8) ≤ 960 := by omega
    have h2 : 8 * (byte % 128 % 8) ≤ 56 := by omega
    have hat : AtSlot c n (g * c.perVu.getD n 0 + s)
        (((portCols (c.nets.getD n default).2.inGroups)[i]!).1 + 64 * (byte % 128 / 8) +
          (56 - 8 * (byte % 128 % 8)) + b)
        (c.slotCol n g s + ((portCols (c.nets.getD n default).2.inGroups)[i]!).1 +
          64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + b) := by
      refine ⟨by rw [hc]; exact Rows.mul_add_lt hg hs, ?_, ?_⟩
      · rw [hsl]; omega
      · unfold Circuit.slotCol; omega
    exact hat.lt (fun r hr => (C.bound_slotLog r hr).1) hm
  have hpv := C.perVu_sha
  unfold Typed.msgCol
  rw [show HmRow.byteAt c.ports[p]! byte = ((HmRow.byteAt c.ports[p]! byte).1, byte % 128) from
    Prod.ext rfl (Typed.byteAt_snd _ _)]
  generalize (HmRow.byteAt c.ports[p]! byte).1 = kk
  generalize hrs : HmRow.rowSlots c.ports = rs at hall hws hpv
  obtain ⟨nSha, where_⟩ := rs
  simp only at hall hws hpv ⊢
  split
  · rename_i hk
    generalize hE : (where_[p]!)[kk]! = e
    obtain ⟨slot, sub⟩ := e
    have hent := hall _ (getElem_mem_toList' (by rw [hws]; exact hp)) (slot, sub)
      (by rw [← hE]; exact getElem_mem_toList' hk)
    simp only at hent ⊢
    obtain ⟨hsz, hsin⟩ := C.sha_in
    have hsub : 2 * sub + 1 < 6 := by omega
    refine hpad _ (Or.inl rfl) slot (by rw [hpv]; exact hent.1) _ (by rw [hsz]; exact hsub) ?_
    rw [hsin _ hsub, sha_width sub 1 (by omega)]
  · obtain ⟨hil, hw⟩ := hm_width C.hm_in (i := 1) (by omega)
    refine hpad _ (Or.inr rfl) p ?_ _ hil (by rw [hw])
    rw [C.perVu_hm]
    exact lt_of_lt_of_le hp (le_nextPow2 _)

theorem unit_inCols_size {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {ty : CircuitType}
    (hty : (ls.getD unit default).type.bind types = some ty) : (done.getD unit default).inCols.size = ty.inBits :=
  (checkLayout_spec ((deriveChecked_spec hd).1 unit (unit_lt hd)) hty).2.2.1

/-! ## The accepted template statement's layout -/

/-- **`TemplateLayout` for a template statement, from its parse and check.** For the circuit `parse` accepted for the
template `t` that `templateOf` built from the class's derivation. -/
theorem templateLayout_of {tags : Tags} {c : Circuit} (P : ParseFactsT tags t c) (C : CheckFactsT c (HmRow.blockOf t))
    (htm : Typed.templateOf name done ls unit ty held = .ok t) {types : String → Option CircuitType}
    {info : String → Option TableInfo} (hd : deriveChecked types ls info (Typed.wordsIn held) unit = .ok done)
    (hheld : ∀ h ∈ held.toList, h.words = (Array.range (h.bytes.size / 4)).map fun i => rd32 h.bytes (4 * i))
    {st : Stmt} (hst : st.c = c)
    (hpin : HmRow.pin c = .ok st.pin) (hda : st.da = (HmRow.delta c st.pin).1) (hdb : st.db = (HmRow.delta c st.pin).2)
    (hm : st.m = c.kLog + st.nbl) (hk : c.kLog ≤ 27) :
    TemplateLayout st (HmRow.blockOf t) (Typed.wordsIn held) done (done.getD unit default) := by
  subst hst
  obtain ⟨digests, -, hroot, hown, hinc, -, -, hbind, hcross, hparts, hmem, hsize, -⟩ := Typed.templateOf_spec htm
  obtain ⟨hbown, -, -, hbin, hbbind, hbcross, hbsize, hbpart, hbrl, -⟩ := HmRow.blockOf_spec t
  have hrows : t.layouts.size < st.c.shaNet ∧ t.layouts.size < st.c.hmNet := by
    have := C.rows_after; rw [hbrl] at this; exact this
  obtain ⟨more, hpre, -⟩ := P.pre
  have hnsz : st.c.nets.size = 1 + t.layouts.size + more.length := by
    rw [← Array.length_toList, hpre]; simp; omega
  have hnet := tmpl_net_facts P htm hd
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
    · exact (netOK_built (parse_built hp)).1
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
    reads_past := templateOf_reads_past htm
    parts_size := by rw [hbsize, hparts_size]
    part := fun i hi hu => by
      have hti : i < t.parts.size := by rw [hparts_size]; exact hu
      rw [hbpart i hti hi]
      obtain ⟨kk, hkk, hidx, hjl, nm, hpl⟩ := part_at htm (Array.getElem_mem hu)
      have hti1 : t.parts[i].1 = kk := by rw [hpu i hti hu]; exact hidx
      have hti2 : t.parts[i].2 = (done.getD unit default).parts[i].2 := by rw [hpu i hti hu]
      have hlay : (st.c.nets.getD (1 + kk) default).2 = t.layouts[kk].2.1 := by
        rcases tmpl_nets P (i := 1 + kk) (by omega) with ⟨h, -⟩ | ⟨k', hk', h, hl⟩ | ⟨h, -⟩
        · omega
        · have : k' = kk := by omega
          subst this
          exact hl
        · omega
      refine ⟨hti2, by rw [hti1]; omega, by rw [hti1, P.root]; omega, by rw [hti1]; omega, by rw [hti1]; omega,
        slot_lt_perVu C hti (by rw [hti1]; exact hkk), ?_⟩
      rw [hti1, hlay]
      exact partNet_of hd hheld hjl hpl
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

/-- **The accepted template statement's parse and check facts, from `Stmt.setupH`** (`tags.typed`, a template read): its
circuit carries the class `k` (`Circuit.cls`), and `ParseFactsT` and `CheckFactsT` hold for it. -/
theorem setupH_factsT (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t)) :
    st.c.cls = some k ∧ ParseFactsT tags t st.c ∧ CheckFactsT st.c (HmRow.blockOf t) := by
  obtain ⟨hp, -, -, -, -, -⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t', c', hrt, hpc, hc⟩ | ⟨net, c', hrn, -, -⟩
  · cases hrt
    have P := parse_facts_tmpl hpc
    obtain ⟨ls', hcheck⟩ := P.check
    have C := check_facts_typed P.typed hcheck
    refine ⟨by rw [hc], ?_, ?_⟩
    · rw [hc]
      exact ⟨P.pre, P.portWords, P.v1Segs, P.unitNet, P.root, P.shaNet, P.hmNet, P.kinds, ⟨ls', hcheck⟩,
        (by rw [← P.rowNets]; exact HmNets.check_congr rfl rfl rfl), P.typed, P.upv,
        P.leavesIn, P.wires, P.leafCuts, P.v1⟩
    · rw [hc]
      exact ⟨C.bound_slotLog, C.disj, C.unit_lookup, C.count, C.one_range, C.root_perVu, C.rangeLogs, C.root_zero,
        C.sha_in, C.hm_in, C.perVu_sha, C.perVu_hm, C.rows_after, C.ins⟩
  · cases hrn

/-- **The accepted template statement's layout, from `Stmt.setupH`** (`tags.typed`, a template read). Its circuit carries
the class `k` (`Circuit.cls`), and `TemplateLayout` holds for the class's unit, over the words of the tables it holds.
A part may place a derived layout or a generated read. -/
theorem setupH_templateLayout (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t)) :
    st.c.cls = some k ∧ st.c.typed = some (HmRow.blockOf t) ∧
      TemplateLayout st (HmRow.blockOf t) (Typed.wordsIn k.held) k.done (k.done.getD k.unit default) := by
  obtain ⟨-, hpin, hda, hdb, hm, hk⟩ := setupH_spec_typed ht h
  obtain ⟨hcls, P', C'⟩ := setupH_factsT ht h hr
  obtain ⟨name, ty, htm⟩ := Typed.read_template hr
  exact ⟨hcls, P'.typed, templateLayout_of P' C' htm k.hd (read_held hr) rfl hpin hda hdb hm hk⟩

/-- **The unit's shape, from `deriveChecked`**: its constant row and order (`unit_const_row`, `order_cols`), its
inputs (`unit_inputs`), and its parts' reads (`part_reads`, `callee_prod`). -/
theorem unitShape_of_class (k : Typed.Class) : UnitShape k.done (k.done.getD k.unit default) :=
  unitShape_of k.hd (unit_const_row k.hd) (order_cols k.hd) (fun _ hp _ hc => part_reads k.hd hp hc)
    (fun _ hp _ hc _ hr => callee_prod k.hd hp hc hr)

/-- **1e's facts at every template instance of an accepted template statement** (T3): for the class `k` its circuit
carries, the block rows at `Typed.col c tb g c` are the unit's block row there, in the model of the accepted statement.
The parts may be derived layouts or generated reads. -/
theorem setupH_blockFacts (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g : ℕ} (hg : g < st.c.g) {one : ℕ}
    (hone : (k.done.getD k.unit default).rows.size ≤ one) :
    BlockFacts ((setupH_templateLayout ht h hr).2.2.model regs) (Typed.wordsIn k.held) k.done
      (k.done.getD k.unit default) one (Typed.col st.c (HmRow.blockOf t) g) :=
  (setupH_templateLayout ht h hr).2.2.blockFacts k.hd (unit_lt k.hd) (unitShape_of_class k) regs hg hone

theorem count_flatMap_rows_pair {α : Type} (l : List α) (R S : α → ℕ) (y j : ℕ) :
    (l.flatMap fun e => [(R e, R e), (R e, S e)]).count (y, j) =
      ((l.filter fun e => R e = y).flatMap fun e => [R e, S e]).count j :=
  count_flatMap_rows l R (fun e => [R e, S e]) y j


theorem filter_range_eq (w : ℕ) : ∀ n, w < n → (List.range n).filter (fun i => decide (i = w)) = [w]
  | 0, h => absurd h (Nat.not_lt_zero _)
  | n + 1, h => by
    rw [List.range_succ, List.filter_append]
    by_cases hw : w < n
    · rw [filter_range_eq w n hw]
      simp [show n ≠ w by omega]
    · have hwn : w = n := by omega
      subst hwn
      rw [List.filter_eq_nil_iff.2 (fun i hi => by simp at hi; simp; omega)]
      simp


/-! ## The unit's inputs copy their message bits -/

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- **The typed tail at an input column's position**: that input's copy entries, `(i, i)` and `(i, copySrc g w)`. -/
theorem tail_count_input (hd : deriveChecked types ls info words unit = .ok done)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {g w : ℕ} (hg : g < st.c.g) (hw : w < 𝒖.inCols.size)
    (side : Bool) (j : ℕ) :
    (Typed.tail st.c tb side).count (Typed.col st.c tb g 𝒖.inCols[w], j) =
      [Typed.col st.c tb g 𝒖.inCols[w], Typed.copySrc st.c g w].count j := by
  have hul := unit_lt hd
  have hx : 𝒖.inCols[w] ∈ 𝒖.inCols := Array.getElem_mem hw
  obtain ⟨hxU, -⟩ := inCols_row hd hul U hx
  obtain ⟨hrow, hnone⟩ := unit_inputs hd _ hx
  have hnd : 𝒖.inCols.toList.Nodup := by
    have := (order_sound hd 0).1
    unfold order at this
    exact this.of_append_left
  unfold Typed.tail
  rw [count_flatMap_range _ _ hg (fun g' hg' hne => List.count_eq_zero.2 fun hm => by
    obtain ⟨x, hx', -, he⟩ := T.tailVU_rows hd hul U hm
    exact hne (T.col_inj hd hg' hg hx' hxU he.symm).1)]
  unfold Typed.tailVU
  rw [List.count_append, List.count_append]
  have hA := count_flatMap_rows (List.range tb.inCols.size) (fun w' => Typed.col st.c tb g tb.inCols[w']!)
    (fun w' => [Typed.col st.c tb g tb.inCols[w']!, Typed.copySrc st.c g w']) (Typed.col st.c tb g 𝒖.inCols[w]) j
  have hfil : ((List.range tb.inCols.size).filter fun w' =>
      Typed.col st.c tb g tb.inCols[w']! = Typed.col st.c tb g 𝒖.inCols[w]) = [w] := by
    have hsz : tb.inCols.size = 𝒖.inCols.size := by rw [T.inCols]
    rw [List.filter_congr (q := fun w' => decide (w' = w)) (fun w' hw' => by
      rw [List.mem_range] at hw'
      have hw'' : w' < 𝒖.inCols.size := by omega
      have hget : tb.inCols[w']! = 𝒖.inCols[w'] := by
        rw [getElem!_pos tb.inCols w' hw']; simp only [T.inCols]
      rw [hget]
      obtain ⟨hU', -⟩ := inCols_row hd hul U (Array.getElem_mem hw'')
      refine decide_eq_decide.2 ⟨fun h => ?_, fun h => by subst h; rfl⟩
      have := (T.col_inj hd hg hg hU' hxU h).2
      have h2 : 𝒖.inCols.toList[w']'(by simpa using hw'') = 𝒖.inCols.toList[w]'(by simpa using hw) := by
        simpa using this
      exact (List.Nodup.getElem_inj_iff hnd).1 h2)]
    rw [hsz]
    exact filter_range_eq w _ hw
  rw [count_flatMap_rows_pair]
  beta_reduce
  rw [hfil]
  have hB := count_flatMap_rows2 tb.bind.toList (fun rs => Typed.col st.c tb g rs.1) (fun rs => rs.2.toList)
    (Typed.col st.c tb g) (Typed.col st.c tb g 𝒖.inCols[w]) j
  have hC := count_flatMap_rows3 tb.cross.toList (fun e => Typed.col st.c tb g e.1)
    (fun e => (if side then e.2.1 else e.2.2).toList) (Typed.col st.c tb g) (Typed.col st.c tb g 𝒖.inCols[w]) j
  rw [hB, hC]
  rw [List.filter_congr (q := fun rs => decide (rs.1 = 𝒖.inCols[w])) (fun rs hrs => by
      obtain ⟨-, -, -, h1, -⟩ := T.bind_row hd hul hrs
      exact decide_eq_decide.2 ⟨fun h => (T.col_inj hd hg hg h1 hxU h).2, fun h => by rw [h]⟩),
    List.filter_congr (l := tb.cross.toList) (q := fun e => decide (e.1 = 𝒖.inCols[w])) (fun e he => by
      obtain ⟨h1, -⟩ := T.cross_row hd U he
      exact decide_eq_decide.2 ⟨fun h => (T.col_inj hd hg hg (Or.inl h1) hxU h).2, fun h => by rw [h]⟩)]
  rw [T.cross, crossOf_at, T.bind, bind_at _ _ (delta_nodup hd)]
  have hb0 : bindCols (Typed.col st.c tb g) 𝒖 𝒖.inCols[w] = [] := by unfold bindCols; rw [hnone]
  have hc0 : crossCols (Typed.col st.c tb g) 𝒖 side 𝒖.inCols[w] = [] := by
    unfold crossCols
    split
    · rename_i hlt
      rw [getD_of_getElem? hrow]
      have hfil' : [𝒖.inCols[w]].filter (· ≥ 𝒖.size) = [] := by
        rw [List.filter_eq_nil_iff]
        intro x hx
        simp only [List.mem_singleton] at hx
        subst hx
        simp only [ge_iff_le, decide_eq_true_eq, not_le]
        exact hlt
      cases side <;> simp only [Bool.false_eq_true, ↓reduceIte, hfil', List.map_nil]
    · rfl
  have hget : tb.inCols[w]! = 𝒖.inCols[w] := by
    rw [getElem!_pos tb.inCols w (by rw [T.inCols]; exact hw)]; simp only [T.inCols]
  rw [hb0, hc0, List.flatMap_singleton, hget]
  simp

/-- The block rows at a net slot's column, counted: the slot net's row there, placed; Δ's first-loop pairs there; and the
typed tail there. -/
theorem entry_raw (T : TemplateLayout st tb words done 𝒖) {n sl off y : ℕ} {d : D} (hn : n < st.c.nets.size)
    (hns : n ≠ st.c.shaNet) (hnh : n ≠ st.c.hmNet) (hat : AtSlot st.c n sl off y)
    (hnet : ∃ nm, Typed.netOfD d nm = .ok { (st.c.nets.getD n default).2 with lookup := none }) (LB : List ℕ)
    (hLB : ∀ y ∈ LB, y < 2 ^ (st.c.nets.getD n default).2.unitLog)
    (hB : (st.c.nets.getD n default).2.unitLog ≤ 32 → ∀ j', sparseEntry (st.c.nets.getD n default).2.b off j' +
      lookupEntry (st.c.nets.getD n default).2.lookup off j' = ((LB.count j' : ℕ) : ZMod 2)) (j : ℕ) :
    rangesEntry st (slotEntryA st) y j + deltaEntry st.da y j =
        (((((Typed.rowsA d).getD off #[]).toList.map ((st.c.rangeOf n).slot sl + ·)).count j +
          (constPairs st.c st.pin).count (y, j) + (Typed.tail st.c tb true).count (y, j) : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) y j + deltaEntry st.db y j =
        (((LB.map ((st.c.rangeOf n).slot sl + ·)).count j +
          (constPairs st.c st.pin).count (y, j) + (Typed.tail st.c tb false).count (y, j) : ℕ) : ZMod 2) := by
  obtain ⟨nm, hnet⟩ := hnet
  obtain ⟨hm, hk, -, hsl⟩ := T.range hn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le T.bound hm) T.kLog
  have hs : ∀ j', sparseEntry (st.c.nets.getD n default).2.a off j' = _ := fun j' => (netOfD_sparse hnet h32 off j').1
  have hla : ∀ y ∈ ((Typed.rowsA d).getD off #[]).toList, y < 2 ^ (st.c.nets.getD n default).2.unitLog :=
    (rowsA_lt hnet off).1
  obtain ⟨hda, hdb⟩ := T.delta_count (T.deltaIn_ne hn hns hnh hat) j
  have hre := fun e => rangesEntry_at T.disj hm hk hat e j
  refine ⟨?_, ?_⟩
  · rw [hre, hda]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
    rw [hs, count_slot _ (by rw [hsl]; exact hla) j]
    push_cast; ring
  · rw [hre, hdb]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
      lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
    rw [hB h32, count_slot _ (by rw [hsl]; exact hLB) j]
    push_cast; ring

theorem copy_sum (y s j : ℕ) :
    ((([y].count j + 0 + [y, s].count j : ℕ)) : ZMod 2) = (if j = s then 1 else 0) := by
  have h : [y].count j + 0 + [y, s].count j = [s].count j + 2 * [y].count j := by
    simp only [List.count_cons, List.count_nil]; ring
  rw [h]
  push_cast
  rw [show (2 : ZMod 2) = 0 from rfl, zero_mul, add_zero]
  by_cases hj : j = s
  · subst hj; simp
  · simp [hj, Ne.symm hj]

/-- **An input of template instance `g` copies its message bit**: the block rows at its position read exactly
`copySrc g w`, in `A` and in `B` (the input copies, over the unit's self input rows). -/
theorem input_rows (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    (U : UnitShape done 𝒖) {g w : ℕ} (hg : g < st.c.g) (hw : w < 𝒖.inCols.size) (j : ℕ) :
    rangesEntry st (slotEntryA st) (Typed.col st.c tb g 𝒖.inCols[w]) j +
        deltaEntry st.da (Typed.col st.c tb g 𝒖.inCols[w]) j = (if j = Typed.copySrc st.c g w then 1 else 0) ∧
      rangesEntry st (slotEntryB st) (Typed.col st.c tb g 𝒖.inCols[w]) j +
        deltaEntry st.db (Typed.col st.c tb g 𝒖.inCols[w]) j = (if j = Typed.copySrc st.c g w then 1 else 0) := by
  have hul := unit_lt hd
  have hx : 𝒖.inCols[w] ∈ 𝒖.inCols := Array.getElem_mem hw
  obtain ⟨hxU, hnc⟩ := inCols_row hd hul U hx
  obtain ⟨hrow, hnone⟩ := unit_inputs hd _ hx
  have htA := T.tail_count_input hd U hg hw true j
  have htB := T.tail_count_input hd U hg hw false j
  rcases T.at_inUnit hd hg hxU with ⟨hco, hat⟩ | ⟨i, hi, hu, hge, hlo, hhi, hat⟩
  · obtain ⟨nm, hr⟩ := T.root
    obtain ⟨-, -, -, -, hcst, -⟩ := Typed.netOfD_ok hr
    have hlk : (st.c.nets.getD st.c.unitNet default).2.lookup = none := netOfD_lookup hr
    have hr' : Typed.netOfD (Typed.rootOf 𝒖) nm = .ok { (st.c.nets.getD st.c.unitNet default).2 with lookup := none } := by
      rw [with_lookup_none hlk]; exact hr
    obtain ⟨hEA, hEB⟩ := T.entry_raw T.unitNet (Ne.symm T.shaNet.2) (Ne.symm T.hmNet.2) hat ⟨nm, hr'⟩
      ((Typed.rowsB (Typed.rootOf 𝒖)).getD 𝒖.inCols[w] #[]).toList (rowsA_lt hr _).2
      (fun h32 j' => by rw [hlk]; simp only [lookupEntry, add_zero]; exact (netOfD_sparse hr h32 _ j').2) j
    have hK := constPairs_count (pin := st.pin) T.disj T.slotLog T.fit (T.range T.unitNet).1 (T.range T.unitNet).2.1
      hat j
    have hK0 : (constPairs st.c st.pin).count (Typed.col st.c tb g 𝒖.inCols[w], j) = 0 := by
      rw [hK, ite_eq_right (fun h => hnc.1 (h.1.trans hcst))]
    have hgd : 𝒖.rows.getD 𝒖.inCols[w] ([], []) = ([𝒖.inCols[w]], [𝒖.inCols[w]]) := getD_of_getElem? hrow
    have hl : ∀ side : Bool, ((if side then (𝒖.rows.getD 𝒖.inCols[w] ([], [])).1
        else (𝒖.rows.getD 𝒖.inCols[w] ([], [])).2).filter (· < 𝒖.size)).map
          ((st.c.rangeOf st.c.unitNet).slot g + ·) = [Typed.col st.c tb g 𝒖.inCols[w]] := by
      intro side
      have hfx : [𝒖.inCols[w]].filter (· < 𝒖.size) = [𝒖.inCols[w]] := by
        rw [List.filter_singleton, decide_eq_true hco]; rfl
      rw [hgd, hat.2.2]
      cases side <;> simp only [Bool.false_eq_true, ↓reduceIte, hfx, List.map_cons, List.map_nil]
    refine ⟨hEA.trans ?_, hEB.trans ?_⟩
    · rw [rowsA_getD (d := Typed.rootOf 𝒖) hco, rootOf_rows, List.toList_toArray]
      rw [show ((𝒖.rows.getD 𝒖.inCols[w] ([], [])).1.filter (· < 𝒖.size)) =
        ((if true then (𝒖.rows.getD 𝒖.inCols[w] ([], [])).1 else (𝒖.rows.getD 𝒖.inCols[w] ([], [])).2).filter
          (· < 𝒖.size)) from rfl, hl true, hK0, htA]
      exact copy_sum _ _ _
    · rw [rowsB_getD (d := Typed.rootOf 𝒖) hco, rootOf_rows, List.toList_toArray]
      rw [show ((𝒖.rows.getD 𝒖.inCols[w] ([], [])).2.filter (· < 𝒖.size)) =
        ((if false then (𝒖.rows.getD 𝒖.inCols[w] ([], [])).1 else (𝒖.rows.getD 𝒖.inCols[w] ([], [])).2).filter
          (· < 𝒖.size)) from rfl, hl false, hK0, htB]
      exact copy_sum _ _ _
  · obtain ⟨-, hn, -, hns, hnh, -, hpn⟩ := T.part i hi hu
    have hnet' := hpn.1
    obtain ⟨nm, hnet⟩ := hnet'
    have hcst : (st.c.nets.getD tb.parts[i].1 default).2.constPos = (done.getD 𝒖.parts[i].1 default).const :=
      (Typed.netOfD_ok hnet).2.2.2.2.1
    have hpm : 𝒖.parts[i] ∈ 𝒖.parts := Array.getElem_mem hu
    have hcpr := U.callee_prod _ hpm
    have hK := constPairs_count (pin := st.pin) T.disj T.slotLog T.fit (T.range hn).1 (T.range hn).2.1 hat j
    have hy := hat.2.2
    generalize hb : 𝒖.parts[i].2 = b at hlo hhi hat hy hK
    generalize hcal : done.getD 𝒖.parts[i].1 default = d at hhi hcst hnet hpn hcpr
    generalize hc' : 𝒖.inCols[w] - b = c' at hat hy hK
    have hcd : c' < d.size := by omega
    have hceq : b + c' = 𝒖.inCols[w] := by omega
    obtain ⟨-, hK1, hO⟩ := part_rows hd hpm (c := c') (by rw [hcal]; exact hcd)
    rw [hcal, hb, hceq] at hK1 hO
    have hcc : c' ≠ d.const := fun h => by rw [hK1 h] at hnone; cases hnone
    have hK0 : (constPairs st.c st.pin).count (Typed.col st.c tb g 𝒖.inCols[w], j) = 0 := by
      rw [hK, ite_eq_right (fun h => hcc (h.1.trans hcst))]
    rcases hO hcc with ⟨s, hfs, -⟩ | ⟨-, hrow'⟩
    · rw [hfs] at hnone; cases hnone
    · rw [hrow] at hrow'
      have hsh := Option.some.inj hrow'
      generalize hr' : d.rows.getD c' ([], []) = row' at hsh
      obtain ⟨ra, rb⟩ := row'
      simp only [shiftRow, Prod.mk.injEq] at hsh
      obtain ⟨ha, hb'⟩ := hsh
      obtain ⟨a0, rfl, ha0⟩ := List.map_eq_singleton_iff.1 ha.symm
      obtain ⟨b0, rfl, hb0⟩ := List.map_eq_singleton_iff.1 hb'.symm
      have hnc : d.reads.find? (·.covers c') = none := by
        cases hrr : d.reads.find? (·.covers c') with
        | none => rfl
        | some r =>
          have := (hcpr c' hcd r hrr).1
          rw [hr'] at this
          simp at this
      obtain ⟨hEA, hEB⟩ := T.entry_raw hn hns hnh hat ⟨nm, hnet⟩ ((Typed.rowsB d).getD c' #[]).toList
        (rowsA_lt hnet c').2 (fun h32 j' => hpn.entryB_unc h32 hnc j') j
      have ha0' : a0 = c' := by omega
      have hb0' : b0 = c' := by omega
      refine ⟨hEA.trans ?_, hEB.trans ?_⟩
      · rw [rowsA_getD hcd, hr', hK0, htA]
        rw [show ([a0].toArray.toList.map ((st.c.rangeOf tb.parts[i].1).slot
          (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) + ·)) = [Typed.col st.c tb g 𝒖.inCols[w]] by
            rw [hy, ha0']; rfl]
        exact copy_sum _ _ _
      · rw [rowsB_getD hcd, hr', hK0, htB]
        rw [show ([b0].toArray.toList.map ((st.c.rangeOf tb.parts[i].1).slot
          (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) + ·)) = [Typed.col st.c tb g 𝒖.inCols[w]] by
            rw [hy, hb0']; rfl]
        exact copy_sum _ _ _

/-- **An input of template instance `g` is a copy row of its message bit**, in the model of the accepted statement (when
the message bit's position is in the block). A satisfying witness then carries the message bit there
(`CopyRow.block_eq`): a table class's `copy`. -/
theorem input_copy (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    (U : UnitShape done 𝒖) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g w : ℕ} (hg : g < st.c.g)
    (hw : w < 𝒖.inCols.size) (hs : Typed.copySrc st.c g w < 2 ^ st.c.kLog) :
    CopyRow (T.model regs)
      ⟨Typed.col st.c tb g 𝒖.inCols[w], T.col_lt hd hg (inCols_row hd (unit_lt hd) U (Array.getElem_mem hw)).1⟩
      ⟨Typed.copySrc st.c g w, hs⟩ := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) _ j.val + deltaEntry st.da _ j.val = _
    rw [(T.input_rows hd U hg hw j.val).1]
    exact ite_fin_val j hs
  · show rangesEntry st (slotEntryB st) _ j.val + deltaEntry st.db _ j.val = _
    rw [(T.input_rows hd U hg hw j.val).2]
    exact ite_fin_val j hs

end TemplateLayout

/-- **An accepted template statement's inputs read message bits inside the block**: input `w` of template instance `g`
copies `Typed.copySrc st.c g w`, a message bit of one of the file's rows (`msgCol_lt`). The file has a row, since the
instance's inputs are its rows' words (`checkTyped`) and the unit has input `w`. -/
theorem setupH_copySrc_lt {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t)) {g w : ℕ} (hg : g < st.c.g)
    (hw : w < (k.done.getD k.unit default).inCols.size) : Typed.copySrc st.c g w < 2 ^ st.c.kLog := by
  obtain ⟨-, P', C'⟩ := setupH_factsT ht h hr
  obtain ⟨name, ty, htm, hty⟩ := read_template_ty hr
  obtain ⟨-, -, -, -, -, hti, -⟩ := Typed.templateOf_spec htm
  obtain ⟨-, hbi, -⟩ := HmRow.blockOf_spec t
  have hne : 0 < st.c.ports.size := by
    by_contra h0
    have hp0 : st.c.ports = #[] := Array.eq_empty_of_size_eq_zero (by omega)
    have hins : ty.ins = #[] := by rw [← hti, ← hbi, C'.ins, hp0]; simp
    have hin := unit_inCols_size k.hd hty
    have : ty.inBits = 0 := by unfold CircuitType.inBits; rw [hins]; rfl
    omega
  unfold Typed.copySrc
  exact msgCol_lt P' C' hg (portAt_lt st.c hne _) _ _ (Nat.mod_lt _ (by norm_num))

/-- **The inputs of an accepted template statement copy their message bits** (a table class's `copy`): input `w` of
template instance `g` is a copy row of `Typed.copySrc st.c g w` in the model of the accepted statement. -/
theorem setupH_inputCopy {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g w : ℕ} (hg : g < st.c.g)
    (hw : w < (k.done.getD k.unit default).inCols.size) :
    CopyRow ((setupH_templateLayout ht h hr).2.2.model regs)
      ⟨Typed.col st.c (HmRow.blockOf t) g (k.done.getD k.unit default).inCols[w],
        (setupH_templateLayout ht h hr).2.2.col_lt k.hd hg
          (TemplateLayout.inCols_row k.hd (unit_lt k.hd) (unitShape_of_class k) (Array.getElem_mem hw)).1⟩
      ⟨Typed.copySrc st.c g w, setupH_copySrc_lt ht h hr hg hw⟩ :=
  (setupH_templateLayout ht h hr).2.2.input_copy k.hd (unitShape_of_class k) regs hg hw _

end FlockSoundness
