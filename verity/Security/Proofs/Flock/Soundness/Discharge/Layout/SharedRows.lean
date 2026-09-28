import Proofs.Flock.Soundness.Discharge.Layout.Drawn

set_option Elab.async false

/-!
# Shared-row populations

A shared-row population keeps its tables' rows as its `b ‖ c` and reads instance `i`'s row `p` at table `p`'s row
`refs[i · np + p]` (`Public.digest`); `HmRow.drawn` keeps the rows and takes the proved units' refs. So a drawn instance's
`b ‖ c` is its proved unit's in the population (`drawn_digestW`), and every one is 128 bytes (`pop_digest_sizeW`:
`HmRow.loadPublic` checks every ref against its table's count, and the tables' rows are `BC_BYTES`-long pieces of the
body, `loadPublic_shared`). `bc_drawnS` is `bc_drawn` without `pop.tables = none`: a registered value's rows are read
from a shared-row file's tables (`Flock.Registered.check` refuses any other), so a statement with registered reads
needs it. The shared-row lemmas keep their names in `Hidden`, where the hidden-output path first needed them.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open FlockSoundness.Discharge.Layout

/-! ## A table's rows -/

theorem foldl_push_prefix (cs : List ℕ) (A : Array ℕ) (s : ℕ) {i : ℕ} (hi : i < A.size) :
    (cs.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (A, s)).1.getD i 0 = A.getD i 0 := by
  induction cs generalizing A s with
  | nil => rfl
  | cons c cs ih =>
    rw [List.foldl_cons, ih (A.push s) (s + c) (by rw [Array.size_push]; omega)]
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, hi.ne, ↓reduceIte]

theorem le_foldl_add (cs : List ℕ) (s : ℕ) : s ≤ cs.foldl (· + ·) s := by
  induction cs generalizing s with
  | nil => exact le_rfl
  | cons c cs ih => exact le_trans (Nat.le_add_right s c) (ih (s + c))

theorem foldl_starts (cs : List ℕ) (A : Array ℕ) (s : ℕ) {p : ℕ} (hp : p < cs.length) :
    (cs.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (A, s)).1.getD (A.size + p) 0 +
      cs.getD p 0 ≤ cs.foldl (· + ·) s := by
  induction cs generalizing A s p with
  | nil => simp at hp
  | cons c cs ih =>
    rw [List.foldl_cons, List.foldl_cons]
    cases p with
    | zero =>
      rw [Nat.add_zero, foldl_push_prefix cs (A.push s) (s + c) (by rw [Array.size_push]; omega)]
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push_size, Option.getD_some, List.getD_cons_zero]
      exact le_foldl_add cs (s + c)
    | succ p =>
      have := ih (A.push s) (s + c) (p := p) (by simp at hp; omega)
      rw [Array.size_push, show A.size + 1 + p = A.size + (p + 1) by omega] at this
      simpa only [List.getD_cons_succ] using this

/-- **A row below table `p`'s count is among the tables' rows**: `starts[p] + x < Σ counts` for `x < counts[p]`. -/
theorem starts_lt {counts : Array ℕ} {p x : ℕ} (hx : x < counts[p]!) :
    (counts.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1.getD p 0 + x <
      counts.foldl (· + ·) 0 := by
  have hp : p < counts.size := by
    by_contra hn
    rw [getElem!_neg counts p hn] at hx
    exact Nat.not_lt_zero _ hx
  have hcp : counts[p]! = counts.toList.getD p 0 := by
    rw [getElem!_pos counts p hp, List.getD_eq_getElem _ _ (by simpa using hp), Array.getElem_toList]
  have := foldl_starts counts.toList #[] 0 (p := p) (by simpa using hp)
  rw [Array.foldl_toList, Array.foldl_toList, Array.size_empty, Nat.zero_add, ← hcp] at this
  omega

/-- `HmRow.loadPublic`'s shared-row branch: the body's length check and the refs' check against their tables' counts. -/
theorem tables_facts {body : ByteArray} {counts : Array ℕ} {n np K R S : ℕ} (f : ℕ → ℕ)
    (hbody : ¬(body.size != HmRow.BC_BYTES * K + R + S) = true)
    (hany : ¬((Array.range ((Array.range (n * np)).map f).size).any fun k =>
      decide (((Array.range (n * np)).map f)[k]! ≥ counts[k % np]!)) = true)
    (hK : K = counts.foldl (· + ·) 0) :
    (∀ x ∈ (Array.range K).map (fun i => body.extract (HmRow.BC_BYTES * i) (HmRow.BC_BYTES * i + HmRow.BC_BYTES)),
        x.size = 128) ∧
      ∀ t, some (⟨counts, (counts.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1,
          (Array.range (n * np)).map f⟩ : Tables) = some t →
        t.refs.size = n * np ∧ (∀ k < t.refs.size, t.refs[k]! < t.counts[k % np]!) ∧
        t.starts = (t.counts.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1 ∧
        ((Array.range K).map
          (fun i => body.extract (HmRow.BC_BYTES * i) (HmRow.BC_BYTES * i + HmRow.BC_BYTES))).size =
          t.counts.foldl (· + ·) 0 := by
  have hb : body.size = HmRow.BC_BYTES * K + R + S := by simpa using hbody
  refine ⟨fun x hx => ?_, fun t ht => ?_⟩
  · simp only [Array.mem_map, Array.mem_range] at hx
    obtain ⟨i, hi, rfl⟩ := hx
    simp only [ByteArray.size_extract, HmRow.BC_BYTES] at hb ⊢
    omega
  · obtain rfl := Option.some.inj ht
    dsimp only
    refine ⟨by simp, fun k hk => ?_, rfl, by simp [hK]⟩
    by_contra hc
    apply hany
    rw [Array.any_eq_true]
    refine ⟨k, by simpa using hk, ?_⟩
    simp only [Array.getElem_range, decide_eq_true_eq]
    omega

set_option maxHeartbeats 2000000 in
/-- **What `HmRow.loadPublic` reads of a population's `b ‖ c`**: each is 128 bytes, and a shared-row file's refs are one
per instance and port, each below its table's count, into tables of consecutive rows. -/
theorem loadPublic_shared {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) :
    (∀ x ∈ pop.digests, x.size = 128) ∧ ∀ t, pop.tables = some t →
      t.refs.size = pop.n * c.ports.size ∧ (∀ k < t.refs.size, t.refs[k]! < t.counts[k % c.ports.size]!) ∧
      t.starts = (t.counts.foldl (fun (acc : Array ℕ × ℕ) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1 ∧
      pop.digests.size = t.counts.foldl (· + ·) 0 := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [pure, Except.pure, Except.ok.injEq] at *
    subst_vars
  all_goals first
    | (refine ⟨?_, fun t ht => by cases ht⟩
       have hbody := ‹¬((file.extract _ _).size != _) = true›
       simp only [bne_iff_ne, ne_eq, not_not, Option.map_none, Option.getD_none, Option.isSome_none,
         Bool.false_eq_true, ↓reduceIte, add_zero] at hbody
       intro x hx
       simp only [Array.mem_map, Array.mem_range] at hx
       obtain ⟨i, hi, rfl⟩ := hx
       exact extract_size hbody hi)
    | exact tables_facts _ ‹_› ‹_› (by rfl)

/-- **A population unit's `b ‖ c` is 128 bytes**, with or without shared rows. -/
theorem pop_digest_sizeW {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) {u p : ℕ} (hu : u < pop.n) (hp : p < c.ports.size) :
    (pop.digest u p c.ports.size).size = 128 := by
  obtain ⟨h128, hT⟩ := loadPublic_shared h
  cases htab : pop.tables with
  | none => exact pop_digest_size h htab hu hp
  | some t =>
    obtain ⟨hrs, hlt, hst, hsz⟩ := hT t htab
    have hk : u * c.ports.size + p < t.refs.size := by rw [hrs]; nlinarith
    have hmod : (u * c.ports.size + p) % c.ports.size = p := by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hp]
    have hr := hlt _ hk
    rw [hmod] at hr
    have he : t.refs.getD (u * c.ports.size + p) 0 = t.refs[u * c.ports.size + p]! := by
      rw [getElem!_pos t.refs _ hk, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hk, Option.getD_some]
    have hi : t.starts.getD p 0 + t.refs.getD (u * c.ports.size + p) 0 < pop.digests.size := by
      rw [hsz, hst, he]
      exact starts_lt hr
    unfold Public.digest
    rw [htab]
    dsimp only
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
    exact h128 _ (Array.getElem_mem hi)

/-- **Instance `r`'s `b ‖ c` in a drawn statement is its proved unit's in the population's**, with or without shared
rows. -/
theorem drawn_digestW {tags : Tags} {c : Circuit} {file : ByteArray} {pop pub : Public} {d : Lean.Json}
    {D : Draw.UnitDraw} (hload : HmRow.loadPublic tags c file = .ok pop) (h : HmRow.drawn c pop d = .ok pub)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population) {r p : ℕ}
    (hr : r < D.proved.size) (hp : p < c.ports.size) :
    pub.digest r p c.ports.size = pop.digest D.proved[r] p c.ports.size := by
  cases htab : pop.tables with
  | none => exact drawn_digest h hD htab (loadPublic_digests hload htab).1 hlt hr hp
  | some t =>
    obtain ⟨hrs, -, -, -⟩ := (loadPublic_shared hload).2 t htab
    obtain ⟨D', hD', hpopn, -, hptab, hdig⟩ := drawn_spec h
    rw [hD] at hD'
    cases hD'
    rw [htab] at hptab hdig
    unfold Public.digest
    rw [hptab, htab, hdig]
    simp only [Option.map_some, Option.isSome_some, ↓reduceIte]
    congr 2
    refine foldl_extract_getD t.refs D.proved (fun u hu => ?_) 0 hr hp
    have := hlt u hu
    rw [hrs, ← hpopn]
    nlinarith

/-- **A drawn statement's `hm96` chunk at block `o`, template instance `g`** is, at a proved unit, that unit's
population `b ‖ c`, with or without shared rows. -/
theorem bcChunk_drawnW {tags : Tags} {c : Circuit} {file : ByteArray} {pop pub : Public} {d : Lean.Json}
    {D : Draw.UnitDraw} (hload : HmRow.loadPublic tags c file = .ok pop) (h : HmRow.drawn c pop d = .ok pub)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population) {o g p : ℕ}
    (hr : o * c.g + g < D.proved.size) (hp : p < c.ports.size) :
    bcChunk c pub o g p = pop.digest D.proved[o * c.g + g] p c.ports.size := by
  obtain ⟨D', hD', -, hn, -, -⟩ := drawn_spec h
  rw [hD] at hD'
  cases hD'
  unfold bcChunk Circuit.instOf
  rw [hn]
  simp only [hr, ↓reduceIte]
  exact drawn_digestW hload h hD hlt hr hp

end FlockSoundness.Discharge.Hidden

namespace FlockSoundness.Discharge.Layout

open Refine FlockSoundness.Discharge.Hidden

/-- **`bc_drawn` with or without shared rows**: the `b ‖ c` a satisfying witness of a drawn statement holds at a proved
unit's `hm96` slot is the unit's `b ‖ c` in the registered population's public file, at every row. -/
theorem bc_drawnS {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st)
    {pop : Public} (hpop : HmRow.loadPublic tags st.c publicFile = .ok pop)
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
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := drawn_spec hdr
    rw [hD] at hD'
    cases hD'
    exact hpn
  rw [← bcChunk_drawnW hpop hdr hD hlt hr hp]
  refine bc_pinned L regs hregs (setupH_regions h) hz o hg hg2 hp fun g' hg' => ?_
  have hr' : o.val * st.c.g + g' < D.proved.size := by omega
  rw [bcChunk_drawnW hpop hdr hD hlt hr' hp]
  exact pop_digest_sizeW hpop (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp

end FlockSoundness.Discharge.Layout
