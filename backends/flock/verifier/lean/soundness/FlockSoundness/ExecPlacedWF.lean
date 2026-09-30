import FlockSoundness.ExecTemplateClass
import FlockSoundness.ExecLookup
import FlockSoundness.Refine.Setup

/-!
# `PlacedWF` from the accepted statement (the layout facts `stmtOf_eq_placed` reads)

`Refine.stmtOf_eq_placed` needs three layout facts of a statement's nets and leaf circuit (`Refine.PlacedWF`). This file
derives them from what the verifier accepted:
- **Sparse columns inside their width.** Every matrix `Sparse.ofRows` builds refuses a column outside its width
  (`ofRows_col_lt`). The nets of an accepted typed template are the template's root and placed layouts (`netOfD`, and
  `placedNet`, which changes only the lookup), and the rest of the circuit's nets, each `Net.parse`'s or
  `Lookup.build`'s (`ParseFactsT.pre`); all of them hold `Sparse.ofRows` outputs. `Stmt.setupH` sets the leaf to
  `default`, whose matrices have no rows (`setupH_leaf_typed`).
- **A lookup inside its net.** `Lookup.build` refuses more than 32 output bits, and its low decoder's top block ends
  below its product rows (`build_placed`). A placed generated read's lookup is its read record's, whose low minterms the
  unit's part check keeps inside the callee (`partReadsOk`).

`setupH_placedWF` assembles them for an accepted typed template. One fact stays a hypothesis: every generated read of
the class has at most 32 output bits (`hk32`). `genOk` checks a read's bits against its layout's value bits, which the
table shapes bound. With `Refine.setupH_wf`, `setupH_templateTableClass_ofSetup` is then the template's class over the
refinement's `stmtOf`.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## `Sparse.ofRows` refuses a column outside its width -/

theorem rd32_lt (b : ByteArray) (o : ℕ) : rd32 b o < 2 ^ 32 := by
  unfold rd32
  have h0 := (b.get! o).toNat_lt
  have h1 := (b.get! (o + 1)).toNat_lt
  have h2 := (b.get! (o + 2)).toNat_lt
  have h3 := (b.get! (o + 3)).toNat_lt
  refine Nat.or_lt_two_pow (Nat.or_lt_two_pow (Nat.or_lt_two_pow ?_ ?_) ?_) ?_
  · omega
  · rw [Nat.shiftLeft_eq]; omega
  · rw [Nat.shiftLeft_eq]; omega
  · rw [Nat.shiftLeft_eq]; omega

theorem ofRows_width {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.width = width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, -, h⟩ := except_bind_ok h
  cases h
  rfl

/-- **Every stored column of a matrix `Sparse.ofRows` built is inside its width.** -/
theorem ofRows_col_lt {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    ∀ i t, i < m.rows → t < m.len[i]! → m.col i t < m.width := by
  intro i t hi ht
  rw [ofRows_width h]
  by_cases hw : width ≤ 2 ^ 32
  · obtain ⟨-, hs, hl, hc⟩ := ofRows_spec h
    have hrs : m.rows = rows.size := ofRows_rows h
    have hls : m.len.size = rows.size := by rw [← Array.length_toList, hl]; simp
    have hss : m.start.size = rows.size := by rw [← Array.length_toList, hs, rowStarts_length]; simp
    have hlen : m.len[i]! = m.len.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.len.size by omega]
    have hst : m.start[i]! = m.start.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.start.size by omega]
    have hsz : t < (m.row i).size := by simp only [Sparse.row, Array.size_map, Array.size_range]; omega
    have hcol : m.col i t = (m.row i)[t] := by
      simp only [Sparse.col, Sparse.row, Array.getElem_map, Array.getElem_range, hst]
    have hmem : m.col i t ∈ (rows.getD i #[]).toList := by
      rw [hcol, ← ofRows_row h hw i]
      exact Array.getElem_mem_toList _
    have hri : rows.getD i #[] = rows[i]'(by omega) := by
      simp [Array.getD_eq_getD_getElem?, show i < rows.size by omega]
    rw [hri] at hmem
    exact hc _ (Array.getElem_mem_toList _) _ hmem
  · have := rd32_lt m.data (m.start[i]! + 4 * t)
    unfold Sparse.col
    omega

/-- A net whose matrices `Sparse.ofRows` built has its columns inside their widths. -/
theorem netCols_of_ofRows {net : Net} {u : ℕ} (ha : ∃ ra, Sparse.ofRows ra (2 ^ u) = .ok net.a)
    (hb : ∃ rb, Sparse.ofRows rb (2 ^ u) = .ok net.b) :
    (∀ i t, i < net.a.rows → t < net.a.len[i]! → net.a.col i t < net.a.width) ∧
      (∀ i t, i < net.b.rows → t < net.b.len[i]! → net.b.col i t < net.b.width) :=
  ⟨let ⟨_, h⟩ := ha; ofRows_col_lt h, let ⟨_, h⟩ := hb; ofRows_col_lt h⟩

/-! ## `Lookup.build`'s lookup is inside its net -/

/-- `Lookup.decode` only adds rows. -/
theorem decode_mono (fuel : ℕ) : ∀ (bd : Lookup.Builder) (bits : List ℕ),
    bd.a.size ≤ (Lookup.decode bd bits fuel).1.a.size := by
  induction fuel with
  | zero => intro bd bits; unfold Lookup.decode; split <;> first | exact Nat.le_refl _ | omega
  | succ fuel ih =>
    intro bd bits
    unfold Lookup.decode
    split
    · exact Nat.le_refl _
    · exact Nat.le_refl _
    · rename_i heq _
      obtain rfl := Nat.succ.inj heq
      dsimp only
      exact foldl_inv (fun s : Lookup.Builder × Array Lookup.Form => bd.a.size ≤ s.1.a.size)
        (fun s x hs => by simp only [Lookup.row, Array.size_push]; omega) (Nat.le_trans (ih _ _) (ih _ _)) rfl

/-- **The low decoder's top block**: for `lo ≥ 2` bits, `decode` returns the first row of the top block it pushed last,
`2^lo` rows. -/
theorem decode_top {lo : ℕ} (hlo : 2 ≤ lo) (bd : Lookup.Builder) :
    (Lookup.decode bd (List.range lo) lo).2.2 + 2 ^ lo = (Lookup.decode bd (List.range lo) lo).1.a.size := by
  obtain ⟨k, rfl⟩ : ∃ k, lo = k + 1 := ⟨lo - 1, by omega⟩
  unfold Lookup.decode
  split
  · rename_i heq
    have := congrArg List.length heq
    simp at this
    omega
  · omega
  · rename_i heq _
    obtain rfl := Nat.succ.inj heq
    dsimp only
    symm
    refine (foldl_count (fun s : Lookup.Builder × Array Lookup.Form => s.1.a.size) (k := 1) (fun s x => ?_) _ _).trans ?_
    · simp [Lookup.row]
    · simp

/-- **What `Lookup.build` builds, for `PlacedWF`**: its matrices are `Sparse.ofRows`'s at its slot size, and its
lookup has at most 32 output bits and its low minterms inside the `B` width. -/
theorem build_placed {name : String} {table : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build name table n lo bits = .ok net) :
    (∃ ra, Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a) ∧ (∃ rb, Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b) ∧
      ∀ lk, net.lookup = some lk → lk.loTop + 2 ^ lk.lo ≤ net.b.width ∧ lk.bits ≤ 32 := by
  unfold Lookup.build at h
  obtain ⟨hc, h⟩ := ite_throw_ok_not h
  have hc' : 2 ≤ lo ∧ bits ≤ 32 := by simp at hc; omega
  obtain ⟨s1, hs1, h⟩ := except_bind_ok h
  dsimp only at h
  rcases hd1 : Lookup.decode s1 (List.range lo) lo with ⟨bd1, loM, loTop⟩
  rw [hd1] at h
  rcases hd2 : Lookup.decode bd1 ((List.range (n - lo)).map (lo + ·)) (n - lo) with ⟨bd2, hiM, x⟩
  rw [hd2] at h
  dsimp only at h
  obtain ⟨s2, hs2, h⟩ := except_bind_ok h
  obtain ⟨s3, hs3, h⟩ := except_bind_ok h
  obtain ⟨s4, hs4, h⟩ := except_bind_ok h
  obtain ⟨hc2, h⟩ := ite_throw_ok_not h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨a, ha, h⟩ := except_bind_ok h
  obtain ⟨b, hb, h⟩ := except_bind_ok h
  cases h
  have htop := decode_top hc'.1 s1
  rw [hd1] at htop
  dsimp only at htop
  have hm2 : bd1.a.size ≤ bd2.a.size := by
    have := decode_mono (n - lo) bd1 ((List.range (n - lo)).map (lo + ·)); rwa [hd2] at this
  have e1a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs1
  have e1b := forIn_range_pure (fun r : Lookup.Builder => r.b.size) (fun i r => by split <;> simp) hs1
  simp only [Array.size_empty] at e1a e1b
  have hb1 : bd1.a.size = bd1.b.size := by
    have := decode_sizes lo s1 (List.range lo) (by omega); rwa [hd1] at this
  have hb2 : bd2.a.size = bd2.b.size := by
    have := decode_sizes (n - lo) bd1 ((List.range (n - lo)).map (lo + ·)) hb1; rwa [hd2] at this
  have e2a := forIn_range_nested (fun r : Lookup.Builder => r.a.size) (fun i r => by simp) hs2
  have e2b := forIn_range_nested (fun r : Lookup.Builder => r.b.size) (fun i r => by simp) hs2
  have e3 := @while_pad s2.a.size _ _ ⟨Nat.le_refl _, by omega⟩ hs3
  have e4a := forIn_range_pure (fun r : Lookup.Builder => r.a.size) (fun i r => by split <;> simp) hs4
  refine ⟨⟨_, ha⟩, ⟨_, hb⟩, fun lk hlk => ?_⟩
  cases hlk
  dsimp only
  refine ⟨?_, hc'.2⟩
  rw [ofRows_width hb]
  have hu := le_two_pow_log2ceil (s4.a.size + 1)
  have hp : 2 ^ log2ceil (s4.a.size + 1) ≤ 2 ^ max 7 (log2ceil (s4.a.size + 1)) :=
    Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
  simp only [Array.size_push] at hu hp ⊢
  omega

/-! ## The leaf of an accepted statement -/

set_option maxHeartbeats 4000000 in
/-- **`Stmt.setupH` sets the leaf to `default`** for a typed statement: its compression circuit's matrices have no rows. -/
theorem setupH_leaf_typed {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    st.leaf = default := by
  unfold Stmt.setupH at h
  simp (config := { zeta := false }) only [ht, ↓reduceIte] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨c, hc, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
  obtain ⟨f0, h⟩ : ∃ f0, jpA f0 = .ok st := by
    split at h
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · split at h <;> (obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩)
  simp (config := { zeta := false }) only [jpA] at h
  split at h
  swap
  · cases h
  (have hh := except_bind_ok h; clear h; obtain ⟨pop, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h
  obtain ⟨pub, h⟩ : ∃ pub, jpB pub = .ok st := by
    split at h <;> (obtain ⟨p, -, h⟩ := except_bind_ok h; exact ⟨p, h⟩)
  simp (config := { zeta := false }) only [jpB] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h
      split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨f, -, h⟩ := except_bind_ok h
        exact ⟨f, h⟩
      · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, hpin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  rfl

/-! ## The nets of an accepted template statement -/

/-- A net `netOfD` built holds `Sparse.ofRows` outputs at its slot size, and has no lookup. -/
theorem netOfD_placed {d : DeriveAll.D} {name : String} {n : Net} (h : Typed.netOfD d name = .ok n) :
    (∃ ra, Sparse.ofRows ra (2 ^ n.unitLog) = .ok n.a) ∧ (∃ rb, Sparse.ofRows rb (2 ^ n.unitLog) = .ok n.b) ∧
      n.unitLog = max 7 (log2ceil d.size) ∧ n.lookup = none := by
  obtain ⟨h1, h2, h3, -, -, -, -, -, -, -, -, -, h13, -⟩ := Net.ofRows_ok h
  exact ⟨⟨_, h2⟩, ⟨_, h3⟩, h1, h13⟩

/-- **A template's placed layout net** is `netOfD`'s, with a generated read's lookup taken from its read record. -/
theorem placedNet_placed {l : Flock.Layout} {d : DeriveAll.D} {nm : String} {held : Array Typed.Held} {n : Net}
    (h : Typed.placedNet l d nm held = .ok n) :
    (∃ ra, Sparse.ofRows ra (2 ^ n.unitLog) = .ok n.a) ∧ (∃ rb, Sparse.ofRows rb (2 ^ n.unitLog) = .ok n.b) ∧
      n.unitLog = max 7 (log2ceil d.size) ∧
      ∀ lk, n.lookup = some lk → ∃ r ∈ d.reads.toList, lk.loTop = r.loTop ∧ lk.lo = r.lo ∧ lk.bits = r.k := by
  obtain ⟨n0, h0, heq, hdl, hgen⟩ := Typed.placedNet_spec h
  obtain ⟨⟨ra, ha⟩, ⟨rb, hb⟩, hu, -⟩ := netOfD_placed h0
  have ha' : n.a = n0.a := by rw [← heq]
  have hb' : n.b = n0.b := by rw [← heq]
  have hu' : n.unitLog = n0.unitLog := by rw [← heq]
  refine ⟨⟨ra, by rw [ha', hu']; exact ha⟩, ⟨rb, by rw [hb', hu']; exact hb⟩, hu'.trans hu, fun lk hlk => ?_⟩
  cases hown : l.own with
  | derive => rw [hdl hown] at hlk; cases hlk
  | gen sha nb vb lo =>
    obtain ⟨r, hd, hr, -, hl⟩ := hgen sha nb vb lo hown
    rw [hl] at hlk
    cases hlk
    exact ⟨r, by rw [hr]; exact List.mem_singleton_self r, rfl, rfl, rfl⟩

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **Every net of an accepted typed template** holds `Sparse.ofRows` outputs at its slot size, and its lookup, if any,
is inside the net with at most 32 output bits, given that every generated read of the class has at most 32. -/
theorem setupH_nets_placed (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (hk32 : ∀ j, ∀ r ∈ (k.done.getD j default).reads.toList, r.k ≤ 32) :
    ∀ x ∈ st.c.nets.toList, (∃ ra, Sparse.ofRows ra (2 ^ x.2.unitLog) = .ok x.2.a) ∧
      (∃ rb, Sparse.ofRows rb (2 ^ x.2.unitLog) = .ok x.2.b) ∧
      ∀ lk, x.2.lookup = some lk → lk.loTop + 2 ^ lk.lo ≤ x.2.b.width ∧ lk.bits ≤ 32 := by
  obtain ⟨hp, -, -, -, -, -⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t', c', hrt, hpc, hc⟩ | ⟨net, c', hrn, -, -⟩
  swap
  · cases hrn
  cases hrt
  obtain ⟨more, hnets, hmore⟩ := (parse_facts_tmpl hpc).pre
  obtain ⟨name, ty, htm⟩ := Typed.read_template hr
  obtain ⟨digests, -, hroot, -, -, -, -, -, -, -, -, hsize, hent⟩ := Typed.templateOf_spec htm
  have hparts := (Types.Dag.partsChecked_spec (Types.Dag.deriveChecked_parts k.hd)).2.2.2.2.2.2.2
  intro x hx
  rw [hc, hnets] at hx
  simp only [Array.toList_append, List.mem_append, Array.toList_map, List.mem_map, List.mem_cons,
    List.mem_nil_iff, or_false] at hx
  rcases hx with (rfl | ⟨e, he, rfl⟩) | hx
  · obtain ⟨ha, hb, -, hl⟩ := netOfD_placed hroot
    exact ⟨ha, hb, fun lk hlk => by rw [hl] at hlk; cases hlk⟩
  · obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem (Array.mem_toList_iff.1 he)
    have hi' : i < ((List.range k.ls.size).filter fun j => (k.done.getD k.unit default).parts.any (·.1 == j)).length := by
      rw [← hsize]; exact hi
    have hpe := hent i hi' hi
    unfold Typed.placedEntry at hpe
    obtain ⟨v, hv, he'⟩ := except_bind_ok hpe
    have hlv : t.layouts[i] = _ := (Except.ok.inj he').symm
    rw [hlv]
    obtain ⟨ha, hb, hu, hl⟩ := placedNet_placed hv
    refine ⟨ha, hb, fun lk hlk => ?_⟩
    obtain ⟨r0, hr0, hlo, hlo', hbits⟩ := hl lk hlk
    set j := ((List.range k.ls.size).filter fun j => (k.done.getD k.unit default).parts.any (·.1 == j))[i]
    have hjm : j ∈ (List.range k.ls.size).filter fun j => (k.done.getD k.unit default).parts.any (·.1 == j) :=
      List.getElem_mem _
    obtain ⟨p, hp, hpj⟩ := Array.any_eq_true'.1 (List.mem_filter.1 hjm).2
    have hpr := hparts p hp
    unfold DeriveAll.partReadsOk at hpr
    simp only [Bool.and_eq_true, Array.all_eq_true_iff_forall_mem, decide_eq_true_eq] at hpr
    rw [beq_iff_eq] at hpj
    rw [hpj] at hpr
    have hsz := hpr.1 r0 (Array.mem_toList_iff.1 hr0)
    have hw : (2 : ℕ) ^ log2ceil (k.done.getD j default).size ≤ 2 ^ max 7 (log2ceil (k.done.getD j default).size) :=
      Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
    have hl2 := le_two_pow_log2ceil (k.done.getD j default).size
    obtain ⟨rb, hrb⟩ := hb
    rw [ofRows_width hrb, hu, hlo, hlo', hbits]
    exact ⟨by omega, hk32 j r0 hr0⟩
  · rcases hmore x hx with ⟨lines, hl⟩ | ⟨nm, tb, n, lo, bits, hb⟩
    · obtain ⟨hlk, -, -, -, -, ra, rb, -, -, ha, hb⟩ := parse_spec hl
      exact ⟨⟨ra, ha⟩, ⟨rb, hb⟩, fun lk h' => by rw [hlk] at h'; cases h'⟩
    · exact build_placed hb

/-- **`PlacedWF` for an accepted typed template**, given that every generated read of its class has at most 32 output
bits: every net holds `Sparse.ofRows` outputs (so its columns are inside their widths), every lookup is inside its net,
and the leaf is `default`. -/
theorem setupH_placedWF (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t))
    (hk32 : ∀ j, ∀ r ∈ (k.done.getD j default).reads.toList, r.k ≤ 32) : Refine.PlacedWF st := by
  have hn := setupH_nets_placed ht h hr hk32
  have hleaf := setupH_leaf_typed ht h
  have hnet : ∀ n (hlt : n < st.c.nets.size), st.c.nets.getD n default = st.c.nets[n] := by
    intro n hlt
    simp [Array.getD_eq_getD_getElem?, hlt]
  have hdef : ∀ n, ¬ n < st.c.nets.size → st.c.nets.getD n default = default := by
    intro n hlt
    simp [Array.getD_eq_getD_getElem?, hlt]
  refine ⟨fun n i t hi ht' => ?_, fun n i t hi ht' => ?_, fun i t hi _ => ?_, fun i t hi _ => ?_, fun n lk hlk => ?_⟩
  · by_cases hlt : n < st.c.nets.size
    · rw [hnet n hlt] at hi ht' ⊢
      obtain ⟨⟨_, ha⟩, -, -⟩ := hn _ (Array.getElem_mem_toList hlt)
      exact ofRows_col_lt ha i t hi ht'
    · rw [hdef n hlt] at hi
      exact absurd hi (Nat.not_lt_zero i)
  · by_cases hlt : n < st.c.nets.size
    · rw [hnet n hlt] at hi ht' ⊢
      obtain ⟨-, ⟨_, hb⟩, -⟩ := hn _ (Array.getElem_mem_toList hlt)
      exact ofRows_col_lt hb i t hi ht'
    · rw [hdef n hlt] at hi
      exact absurd hi (Nat.not_lt_zero i)
  · rw [hleaf] at hi
    exact absurd hi (Nat.not_lt_zero i)
  · rw [hleaf] at hi
    exact absurd hi (Nat.not_lt_zero i)
  · by_cases hlt : n < st.c.nets.size
    · rw [hnet n hlt] at hlk ⊢
      exact (hn _ (Array.getElem_mem_toList hlt)).2.2 lk hlk
    · rw [hdef n hlt] at hlk
      cases hlk

/-- **A typed template's table class over the refinement's `stmtOf`, from `Stmt.setupH`**: `StmtWF` is
`Refine.setupH_wf`'s, and `PlacedWF` is `setupH_placedWF`'s, given that every generated read of the class has at most 32
output bits. -/
def setupH_templateTableClass_ofSetup (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
    (hr : Typed.read tags cf tables = .ok (k, .inr t)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hk32 : ∀ j, ∀ r ∈ (k.done.getD j default).reads.toList, r.k ≤ 32) :
    TableClass (Refine.stmtOf st (Refine.setupH_wf h).1 regs) :=
  setupH_templateTableClass_stmtOf ht h hr regs (Refine.setupH_wf h).1 (setupH_placedWF ht h hr hk32)

end FlockSoundness
