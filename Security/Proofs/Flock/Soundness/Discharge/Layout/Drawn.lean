import Proofs.Flock.Soundness.Discharge.Layout.HmRegion

set_option Elab.async false

/-!
# The drawn units' `b ‖ c`: the population's

A drawn statement's public file is `HmRow.drawn` of the registered population's (`setupH_drawn`): its instances are the
draw's proved units, and instance `r`'s `b ‖ c` is proved unit `D.proved[r]`'s in the population (`drawn_digest`, for a
file without shared rows, whose `b ‖ c` are `n · np` pieces of 128 bytes, `loadPublic_digests`; the draw's units are in
the population, `ofJson_lt`). So at a proved unit's `hm96` slot, block `o` and template instance `g` with `o · c.g + g = r`, a satisfying
witness holds the population's `b ‖ c` of unit `D.proved[r]` (`bc_drawn`): the registration's, fixed by the public file
before the draw.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Refine

set_option maxHeartbeats 2000000 in
/-- **A drawn statement's public file is the drawn units' over the population's**: setupH loads the population's public
file and takes `HmRow.drawn` of it at the draw. -/
theorem setupH_drawn {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st) :
    ∃ pop, HmRow.loadPublic tags st.c publicFile = .ok pop ∧ HmRow.drawn st.c pop d = .ok st.pub := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, ‹_›, ‹_›⟩

set_option maxHeartbeats 2000000 in
/-- **`HmRow.drawn`'s statement**: the draw parses (`D`), its population is the registered one, its instances are the
proved units, and its `b ‖ c` are the proved units' (a shared-row file keeps its tables). -/
theorem drawn_spec {c : Circuit} {pop pub : Public} {d : Lean.Json} (h : HmRow.drawn c pop d = .ok pub) :
    ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧ D.population = pop.n ∧ pub.n = D.proved.size ∧
      pub.tables = pop.tables.map (fun t => { t with refs := (D.proved.foldl
        (fun acc u => acc ++ t.refs.extract (u * c.ports.size) (u * c.ports.size + c.ports.size)) #[]) }) ∧
      pub.digests = if pop.tables.isSome then pop.digests else
        D.proved.foldl (fun acc u => acc ++ pop.digests.extract (u * c.ports.size) (u * c.ports.size + c.ports.size)) #[] := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, ‹_›, ?_, rfl, rfl, rfl⟩
    have hpop := ‹¬(_ != pop.n || pop.population.isSome) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at hpop
    exact hpop.1

theorem foldl_extract_toList {α : Type} (xs : Array α) (np : ℕ) : ∀ (L : List ℕ) (A : Array α),
    (L.foldl (fun acc u => acc ++ xs.extract (u * np) (u * np + np)) A).toList =
      A.toList ++ (L.map fun u => (xs.extract (u * np) (u * np + np)).toList).flatten
  | [], A => by simp
  | u :: L, A => by
      rw [List.foldl_cons, foldl_extract_toList xs np L, Array.toList_append]
      simp [List.append_assoc]

theorem getD_flatten_len {α : Type} {k : ℕ} {L : List (List α)} (hL : ∀ l ∈ L, l.length = k) (d : α) :
    ∀ (r j : ℕ), r < L.length → j < k → L.flatten.getD (r * k + j) d = (L.getD r []).getD j d := by
  induction L with
  | nil => intro r j hr; simp at hr
  | cons l L ih =>
      intro r j hr hj
      have hl : l.length = k := hL l List.mem_cons_self
      cases r with
      | zero =>
          simp only [List.flatten_cons, zero_mul, zero_add, List.getD_cons_zero]
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), ← List.getD_eq_getElem?_getD]
      | succ r =>
          simp only [List.flatten_cons, List.getD_cons_succ]
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hl]; nlinarith),
            ← List.getD_eq_getElem?_getD, hl, show (r + 1) * k + j - k = r * k + j by
              rw [Nat.succ_mul]; omega]
          exact ih (fun l' h' => hL l' (List.mem_cons_of_mem _ h')) r j (by simpa using hr) hj

theorem array_getD_toList {α : Type} (a : Array α) (i : ℕ) (d : α) : a.getD i d = a.toList.getD i d := by
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList]
  unfold Array.getD
  split <;> simp_all

/-- **The proved units' consecutive `np`-wide pieces, read at instance `r`'s position `p`**: the population's at unit
`L[r]`, when every proved unit's piece is inside the population's. -/
theorem foldl_extract_getD {α : Type} (xs : Array α) {np : ℕ} (L : Array ℕ) (hL : ∀ u ∈ L, u * np + np ≤ xs.size)
    (d : α) {r p : ℕ} (hr : r < L.size) (hp : p < np) :
    (L.foldl (fun acc u => acc ++ xs.extract (u * np) (u * np + np)) #[]).getD (r * np + p) d =
      xs.getD (L[r] * np + p) d := by
  rw [array_getD_toList, ← Array.foldl_toList, foldl_extract_toList, List.toList_toArray, List.nil_append,
    getD_flatten_len (k := np) _ d r p (by simpa using hr) hp]
  · have hm : (L.toList.map fun u => (xs.extract (u * np) (u * np + np)).toList).getD r [] =
        (xs.extract (L[r] * np) (L[r] * np + np)).toList := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hr]
      rfl
    have hle := hL L[r] (Array.getElem_mem hr)
    rw [hm, List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getElem?_extract, array_getD_toList,
      List.getD_eq_getElem?_getD, Array.getElem?_toList]
    simp only [show p < min (L[r] * np + np) xs.size - L[r] * np by omega, ↓reduceIte]
  · intro l hl
    simp only [List.mem_map] at hl
    obtain ⟨u, hu, rfl⟩ := hl
    have := hL u (by simpa using hu)
    simp only [Array.length_toList, Array.size_extract]
    omega

theorem extract_size {b : ByteArray} {i K O : ℕ} (hb : b.size = HmRow.BC_BYTES * K + O) (hi : i < K) :
    (b.extract (HmRow.BC_BYTES * i) (HmRow.BC_BYTES * i + HmRow.BC_BYTES)).size = 128 := by
  simp only [ByteArray.size_extract, HmRow.BC_BYTES] at hb ⊢
  omega

set_option maxHeartbeats 2000000 in
/-- **The population's `b ‖ c`**: a public file without shared rows holds `n · np` of them, each 128 bytes long (its
length check, then consecutive `BC_BYTES`-long pieces of its body). -/
theorem loadPublic_digests {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) (ht : pop.tables = none) :
    pop.digests.size = pop.n * c.ports.size ∧ ∀ x ∈ pop.digests, x.size = 128 := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [pure, Except.pure, Except.ok.injEq] at *
    subst_vars
  all_goals first
    | exact absurd ‹(some _ : Option Flock.Tables) = none› (Option.some_ne_none _)
    | skip
  all_goals
    have hbody := ‹¬((file.extract _ _).size != _) = true›
    simp only [bne_iff_ne, ne_eq, not_not, Option.map_none, Option.getD_none, Option.isSome_none,
      Bool.false_eq_true, ↓reduceIte, add_zero] at hbody
    refine ⟨by simp, ?_⟩
    intro x hx
    simp only [Array.mem_map, Array.mem_range, Option.map_none, Option.getD_none] at hx
    obtain ⟨i, hi, rfl⟩ := hx
    exact extract_size hbody hi

theorem lt_of_all_range {a : Array ℕ} {N : ℕ} {f : ℕ → Bool}
    (h : (List.range a.size).all (fun i => a[i]! < N && f i) = true) : ∀ u ∈ a, u < N := by
  intro u hu
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hu
  have := List.all_eq_true.1 h i (List.mem_range.2 hi)
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  simpa [getElem!_pos a i hi] using this.1

theorem of_not_not_true {b : Bool} (h : ¬ (!b) = true) : b = true := by simpa using h

theorem of_not_or_not_true {a b : Bool} (h : ¬ (a || !b) = true) : b = true := by
  cases a <;> cases b <;> simp_all

set_option maxHeartbeats 2000000 in
/-- **A draw's units are in its population**: `UnitDraw.ofJson` checks the drawn units, and a closure's units, against
`[0, population)`. -/
theorem ofJson_lt {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) :
    (∀ u ∈ D.units, u < D.population) ∧ ∀ u ∈ D.closure, u < D.population := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨lt_of_all_range (of_not_not_true ‹¬(!(List.range _).all _) = true›), ?_⟩
    have hcl := ‹(pure _ : V (Array ℕ)) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hcl
    subst hcl
    first
    | exact lt_of_all_range (of_not_or_not_true ‹¬((_ : Array ℕ).isEmpty || !(List.range _).all _) = true›)
    | simp

/-- A draw without a closure proves its drawn units, which are in its population. -/
theorem proved_lt {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) (hcl : D.closure = #[]) :
    ∀ u ∈ D.proved, u < D.population := by
  unfold Draw.UnitDraw.proved
  rw [hcl]
  exact (ofJson_lt h).1

/-- **Instance `r`'s `b ‖ c` in a drawn statement is its proved unit's in the population's** (a file without shared
rows). -/
theorem drawn_digest {c : Circuit} {pop pub : Public} {d : Lean.Json} {D : Draw.UnitDraw}
    (h : HmRow.drawn c pop d = .ok pub) (hD : Draw.UnitDraw.ofJson d = .ok D) (hpt : pop.tables = none)
    (hsz : pop.digests.size = pop.n * c.ports.size) (hlt : ∀ u ∈ D.proved, u < D.population)
    {r p : ℕ} (hr : r < D.proved.size) (hp : p < c.ports.size) :
    pub.digest r p c.ports.size = pop.digest D.proved[r] p c.ports.size := by
  obtain ⟨D', hD', hpopn, -, htab, hdig⟩ := drawn_spec h
  rw [hD] at hD'
  cases hD'
  have hpub : pub.tables = none := by rw [htab, hpt]; rfl
  unfold Public.digest
  rw [hpub, hpt, hdig, ite_eq_right (by simp [hpt])]
  dsimp only
  refine foldl_extract_getD pop.digests D.proved (fun u hu => ?_) _ hr hp
  have := hlt u hu
  rw [hsz, hpopn] at *
  nlinarith

/-- **A drawn statement's `hm96` chunk at block `o`, template instance `g`** is, at a proved unit, that unit's
population `b ‖ c`. -/
theorem bcChunk_drawn {c : Circuit} {pop pub : Public} {d : Lean.Json} {D : Draw.UnitDraw}
    (h : HmRow.drawn c pop d = .ok pub) (hD : Draw.UnitDraw.ofJson d = .ok D) (hpt : pop.tables = none)
    (hsz : pop.digests.size = pop.n * c.ports.size) (hlt : ∀ u ∈ D.proved, u < D.population)
    {o g p : ℕ} (hr : o * c.g + g < D.proved.size) (hp : p < c.ports.size) :
    bcChunk c pub o g p = pop.digest D.proved[o * c.g + g] p c.ports.size := by
  obtain ⟨D', hD', -, hn, -, -⟩ := drawn_spec h
  rw [hD] at hD'
  cases hD'
  unfold bcChunk Circuit.instOf
  rw [hn]
  simp only [hr, ↓reduceIte]
  exact drawn_digest h hD hpt hsz hlt hr hp

/-- A population unit's `b ‖ c` is 128 bytes (a file without shared rows). -/
theorem pop_digest_size {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) (hpt : pop.tables = none) {u p : ℕ} (hu : u < pop.n)
    (hp : p < c.ports.size) : (pop.digest u p c.ports.size).size = 128 := by
  obtain ⟨hsz, h128⟩ := loadPublic_digests h hpt
  unfold Public.digest
  rw [hpt]
  dsimp only
  have hi : u * c.ports.size + p < pop.digests.size := by rw [hsz]; nlinarith
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  exact h128 _ (Array.getElem_mem hi)

/-- **The `b ‖ c` a satisfying witness of a drawn statement holds at a proved unit's `hm96` slot** is the unit's `b ‖ c`
in the registered population's public file: at block `o`, template instance `g` (below `2^⌊log₂ c.g⌋`), input port `p`,
the instance `o · c.g + g` is proved unit `D.proved[o · c.g + g]`, and the 128 bytes at `bcAddr` are its population
`b ‖ c`. It holds in the model's statement at setupH's layout whose regions contain the executable's (`hregs`), for a
population file without shared rows (`hpt`) and a draw whose proved units are in the population (`hlt`; `proved_lt`
without a closure). -/
theorem bc_drawn {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st)
    {pop : Public} (hpop : HmRow.loadPublic tags st.c publicFile = .ok pop) (hpt : pop.tables = none)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hr : o.val * st.c.g + g < D.proved.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) =
      (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList := by
  obtain ⟨pop', hpop', hdr⟩ := setupH_drawn h
  rw [hpop] at hpop'
  cases hpop'
  have hsz := (loadPublic_digests hpop hpt).1
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := drawn_spec hdr
    rw [hD] at hD'
    cases hD'
    exact hpn
  rw [← bcChunk_drawn hdr hD hpt hsz hlt hr hp]
  refine bc_pinned L regs hregs (setupH_regions h) hz o hg hg2 hp fun g' hg' => ?_
  have hr' : o.val * st.c.g + g' < D.proved.size := by omega
  rw [bcChunk_drawn hdr hD hpt hsz hlt hr' hp]
  exact pop_digest_size hpop hpt (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp

end FlockSoundness.Discharge.Layout
