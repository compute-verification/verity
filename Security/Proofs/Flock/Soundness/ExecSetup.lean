import Proofs.Flock.Soundness.ExecCircuit
import Proofs.Flock.Soundness.ExecCheck
import Proofs.Flock.Soundness.ExecLookup

open Flock FlockLevel3

namespace FlockSoundness

/-- **What `Stmt.setupH` accepted** (an untyped `hm96-sha512/row/v1` statement, `tags.typed = false`, which `setupH`
reads with `HmRow.parse` rather than `parseTyped`): the circuit `HmRow.parse` accepted, the pin
`HmRow.pin` gave, Δ from `HmRow.delta`, `m = k_log + nbl`, `k_log ≤ 27`, and the range `checkInRange` accepted
(k_log, the region count and `m_pts`). -/
theorem setupH_spec {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = false) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    HmRow.parse tags cf tables = .ok st.c ∧ HmRow.pin st.c = .ok st.pin ∧
      st.da = (HmRow.delta st.c st.pin).1 ∧ st.db = (HmRow.delta st.c st.pin).2 ∧
      st.m = st.c.kLog + st.nbl ∧ st.c.kLog ≤ 27 ∧ checkInRange st.c.kLog st.regions.size st.mPts = .ok () := by
  unfold Stmt.setupH at h
  simp (config := { zeta := false }) only [ht, Bool.false_eq_true, ↓reduceIte] at h
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
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h)
  obtain ⟨pub, h⟩ : ∃ pub, jpB pub = .ok st := by
    split at h <;> (obtain ⟨p, -, h⟩ := except_bind_ok h; exact ⟨p, h⟩)
  simp (config := { zeta := false }) only [jpB] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h)
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h)
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
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h)
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, hpin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨⟨⟩, hin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) h2 digest sigma at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  refine ⟨hc, hpin, rfl, rfl, rfl, ?_, hin⟩
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hk
  exact Nat.le_of_not_lt hk.1


/-! ## Nets, ranges and the pin -/

theorem kind_beq_net {k : Kind} {n : ℕ} (h : (k == Kind.net n) = true) : k = Kind.net n := by
  cases k with
  | net i =>
    have h' : (i == n) = true := h
    rw [beq_iff_eq.1 h']
  | comp => cases h
  | mask => cases h

/-- **The pin `HmRow.pin` gives**: the first slot's constant of a net range with a slot. -/
theorem pin_spec {c : Circuit} {p : ℕ} (h : HmRow.pin c = .ok p) :
    ∃ r ∈ c.ranges.toList, ∃ i, r.kind = .net i ∧ 0 < r.count ∧
      p = r.start + (c.nets.getD i default).2.constPos := by
  unfold HmRow.pin at h
  dsimp only at h
  split at h
  · rename_i r hr
    split at h
    · rename_i i hi
      by_cases hc : (r.count == 0) = true
      · rw [ite_eq_left hc] at h; obtain ⟨_, h0, _⟩ := except_bind_ok h; cases h0
      · rw [ite_eq_right hc] at h
        obtain ⟨-, h⟩ := ite_throw_ok' h
        cases h
        refine ⟨r, Array.mem_def.1 (Array.mem_of_find?_eq_some hr), i, hi, ?_, rfl⟩
        simp only [beq_iff_eq] at hc
        omega
    · obtain ⟨_, h0, _⟩ := except_bind_ok h; cases h0
  · obtain ⟨_, h0, _⟩ := except_bind_ok h; cases h0

/-- A net with exactly one range: `rangeOf` finds it. -/
theorem rangeOf_spec {c : Circuit} {n : ℕ} (h : (c.ranges.filter (·.kind == Kind.net n)).size = 1) :
    c.rangeOf n ∈ c.ranges.toList ∧ (c.rangeOf n).kind = .net n ∧
      c.rangeOf n = (c.ranges.filter (·.kind == Kind.net n))[0]! := by
  have hl : (c.ranges.toList.filter (·.kind == Kind.net n)).length = 1 := by
    rw [← Array.toList_filter, Array.length_toList, h]
  obtain ⟨r, hr⟩ := List.length_eq_one_iff.1 hl
  have hf : c.ranges.find? (·.kind == Kind.net n) = some r := by
    rw [← Array.find?_toList, ← List.head?_filter, hr]; rfl
  have hmem : r ∈ c.ranges.toList.filter (·.kind == Kind.net n) := by rw [hr]; simp
  rw [List.mem_filter] at hmem
  have hro : c.rangeOf n = r := by unfold Circuit.rangeOf; rw [hf]; rfl
  have harr : c.ranges.filter (·.kind == Kind.net n) = #[r] := Array.ext' (by rw [Array.toList_filter, hr])
  refine ⟨hro ▸ hmem.1, hro ▸ kind_beq_net hmem.2, ?_⟩
  rw [hro, harr]
  rfl

/-! ## Port lists and the rows' slots -/

theorem inner_snd : ∀ (L : List ℕ) (out : Array (ℕ × ℕ)) (a : ℕ),
    ((forIn L (out, a) (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList.map (·.2))
      = out.toList.map (·.2) ++ L
  | [], out, a => by simp; rfl
  | w :: L, out, a => by
    rw [List.forIn_cons]
    show ((forIn L (out.push (a, w), a + w) _ : Id _).1.toList.map (·.2)) = _
    rw [inner_snd L]
    simp
theorem portCols_single (g : Group) : (portCols #[g]).toList.map (·.2) = g.bits.toList := by
  unfold portCols
  simp only
  rw [← Array.forIn_toList]
  simp only [List.forIn_cons, List.forIn_nil]
  rw [← Array.forIn_toList]
  show ((forIn g.bits.toList (#[], g.col * WORD)
    (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList.map (·.2)) = _
  rw [inner_snd]
  simp
theorem rowSlots_spec (ports : Array Port) : (HmRow.rowSlots ports).2.size = ports.size := by
  unfold HmRow.rowSlots
  simp only [HmRow.PER_SLOT]
  rw [← Array.forIn_toList]
  have := forIn_id_inv_tail (fun (L : List Port) (s : ℕ × Array (Array (ℕ × ℕ))) =>
      s.2.size + L.length = ports.size)
    (f := fun p s => pure (ForInStep.yield (s.1 + p.chunks,
      s.2.push (Array.map (fun k => ((s.1 + k) / 3, (s.1 + k) % 3)) (Array.range p.chunks)))))
    (fun p L s hs => ⟨_, rfl, ?_⟩) ports.toList (0, #[]) (by simp)
  · exact this
  · simp only [List.length_cons, Array.size_push] at hs ⊢; omega

theorem rowSlots_all (ports : Array Port) :
    ∀ w ∈ (HmRow.rowSlots ports).2.toList, ∀ e ∈ w.toList, e.1 < (HmRow.rowSlots ports).1 ∧ e.2 < 3 := by
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
    show k / 3 < (_ + 3 - 1) / 3 ∧ k % 3 < 3
    omega
  · intro w hw e he
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hw
    rcases hw with hw | rfl
    · obtain ⟨k, hk, hlt⟩ := hs w hw e he
      exact ⟨k, hk, by simp only; omega⟩
    · simp only [Array.toList_map, Array.toList_range, List.mem_map, List.mem_range] at he
      obtain ⟨k, hk, rfl⟩ := he
      exact ⟨s.1 + k, rfl, by simp only; omega⟩

/-- The width of a `sha512x3` slot's input `2 s + i` of a row's chain: its `cv`'s 512 bits, or its message's 1024. -/
theorem sha_width (s i : ℕ) (hi : i < 2) :
    (if (2 * s + i) % 2 == 0 then 512 else 1024) = 512 * (i + 1) := by
  rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl
  · simp
  · have : (2 * s + 1) % 2 = 1 := by omega
    simp [this]

/-- The width of an hm96 slot's input `i` of a row: its `cv`'s 512 bits, or its `pad`'s 1024. -/
theorem hm_width {gs : Array Group} (h : (portCols gs).map (·.2) = #[512, 1024, 8 * HmRow.SALT_BYTES]) {i : ℕ}
    (hi : i < 2) : i < (portCols gs).size ∧ ((portCols gs)[i]!).2 = 512 * (i + 1) := by
  have hl : (portCols gs).toList.map (·.2) = [512, 1024, 8 * HmRow.SALT_BYTES] := by rw [← Array.toList_map, h]
  have hsz : (portCols gs).size = 3 := by have := congrArg List.length hl; simpa using this
  refine ⟨by omega, ?_⟩
  rw [getElem!_pos _ i (by omega)]
  have := congrArg (fun l : List ℕ => l[i]?) hl
  simp only [List.getElem?_map, Array.getElem?_toList] at this
  rw [Array.getElem?_eq_getElem (by omega)] at this
  rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl <;> simpa using this


theorem inner_append : ∀ (L : List ℕ) (out : Array (ℕ × ℕ)) (a : ℕ),
    ((forIn L (out, a) (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList) =
      out.toList ++
        ((forIn L (#[], a) (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList)
  | [], out, a => by
    show out.toList = out.toList ++ (#[] : Array (ℕ × ℕ)).toList
    simp
  | w :: L, out, a => by
    rw [List.forIn_cons, List.forIn_cons]
    show ((forIn L (out.push (a, w), a + w) _ : Id _).1.toList) =
      out.toList ++ ((forIn L (#[].push (a, w), a + w) _ : Id _).1.toList)
    rw [inner_append L (out.push (a, w)), inner_append L (#[].push (a, w))]
    simp

/-- Each group's ports, in order: `portCols` lists every group's ports as it would alone. -/
theorem portCols_toList (gs : Array Group) :
    (portCols gs).toList = gs.toList.flatMap fun g => (portCols #[g]).toList := by
  have hsingle : ∀ g : Group, (portCols #[g]).toList = ((forIn g.bits.toList (#[], g.col * WORD)
      (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList) := by
    intro g
    unfold portCols
    simp only
    rw [← Array.forIn_toList]
    simp only [List.forIn_cons, List.forIn_nil]
    rw [← Array.forIn_toList]
    rfl
  simp only [hsingle]
  unfold portCols
  simp only
  rw [← Array.forIn_toList]
  refine (forIn_id_append (fun g : Group => ((forIn g.bits.toList (#[], g.col * WORD)
      (fun w s => pure (ForInStep.yield (s.1.push (s.2, w), s.2 + w))) : Id _).1.toList)) _
    (fun g _ d => ⟨_, rfl, ?_⟩) #[]).trans (by simp)
  rw [← Array.forIn_toList]
  exact inner_append _ _ _

theorem mem_portCols {gs : Array Group} {g : Group} (hg : g ∈ gs.toList) {q : ℕ × ℕ}
    (hq : q ∈ (portCols #[g]).toList) : q ∈ (portCols gs).toList := by
  rw [portCols_toList]
  exact List.mem_flatMap.2 ⟨g, hg, hq⟩

theorem getElem_mem_toList' {α : Type} [Inhabited α] {xs : Array α} {i : ℕ} (hi : i < xs.size) :
    xs[i]! ∈ xs.toList := by
  rw [getElem!_pos xs i hi]
  exact Array.mem_def.1 (Array.getElem_mem hi)

theorem getD_mem_toList {α : Type} {xs : Array α} {i : ℕ} {d : α} (hi : i < xs.size) :
    xs.getD i d ∈ xs.toList := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  exact Array.mem_def.1 (Array.getElem_mem hi)

/-! ## Every net of an accepted circuit -/

/-- A net's constant and input rows fit its slot, and its input ports lie in its input rows. -/
def NetOK (net : Net) : Prop :=
  net.constPos < 2 ^ net.unitLog ∧ net.inWords * WORD ≤ 2 ^ net.unitLog ∧
    ∀ q ∈ (portCols net.inGroups).toList, q.1 + q.2 ≤ net.inWords * WORD

theorem netOK_built {net : Net} (h : Built net) :
    NetOK net ∧ 7 ≤ net.unitLog ∧ (net.unitLog ≤ 32 → ConstRow net) :=
  let ⟨h1, h2, h3⟩ := built_fit h
  ⟨⟨h1, h2, built_ports h⟩, h3, built_constRow h⟩

theorem netOK_lookup {nm : String} {tb : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build nm tb n lo bits = .ok net) :
    NetOK net ∧ 7 ≤ net.unitLog ∧ (net.unitLog ≤ 32 → ConstRow net) := by
  obtain ⟨-, hig, hiw, hn, hu, hl, hc⟩ := build_facts h
  have h1 : 2 ^ log2ceil net.useful ≤ 2 ^ net.unitLog :=
    Nat.pow_le_pow_right (by norm_num) (by rw [hl]; exact le_max_right _ _)
  have h7 : 7 ≤ net.unitLog := by rw [hl]; exact le_max_left _ _
  have h128 : 2 ^ 7 ≤ 2 ^ net.unitLog := Nat.pow_le_pow_right (by norm_num) h7
  have := le_two_pow_log2ceil net.useful
  have hW : net.inWords * WORD = 128 := by rw [hiw]; rfl
  refine ⟨⟨by omega, by rw [hW]; exact h128, ?_⟩, h7, hc⟩
  intro q hq
  rw [hig] at hq
  obtain ⟨g, hg, hle⟩ := portCols_le _ q hq
  simp only [List.mem_singleton] at hg
  subst hg
  rw [hW]
  simp only [List.sum_cons, List.sum_nil] at hle
  have : (0 : ℕ) * WORD = 0 := Nat.zero_mul _
  omega

/-- **Every net fits its slot**, and reads itself at its constant when it has a slot and fits `rd32`'s words: what
`nets_ok` gives a flat circuit and `parseTmpl_nets_ok` a templated one. -/
abbrev NetsFit (c : Circuit) : Prop :=
  ∀ i, NetOK (c.nets.getD i default).2 ∧ (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
    ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2))

theorem nets_ok {tags : Tags} {c : Circuit} (P : ParseFacts tags c) (i : ℕ) :
    NetOK (c.nets.getD i default).2 ∧ (i < c.nets.size → 7 ≤ (c.nets.getD i default).2.unitLog ∧
      ((c.nets.getD i default).2.unitLog ≤ 32 → ConstRow (c.nets.getD i default).2)) := by
  by_cases hi : i < c.nets.size
  · have hmem : c.nets.getD i default ∈ c.nets.toList := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
      exact Array.mem_def.1 (Array.getElem_mem hi)
    rcases P.nets _ hmem with hp | ⟨_, _, _, _, _, hb⟩
    · exact ⟨(netOK_built hp).1, fun _ => (netOK_built hp).2⟩
    · exact ⟨(netOK_lookup hb).1, fun _ => (netOK_lookup hb).2⟩
  · have hd : (c.nets.getD i default).2 = default := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]
      rfl
    rw [hd]
    refine ⟨⟨by decide, by decide, fun q hq => ?_⟩, fun h => absurd h hi⟩
    have : portCols (default : Net).inGroups = #[] := rfl
    rw [this] at hq
    cases hq

/-- **The unit is a built net**: `HmRow.parse`'s nets are nets `Net.ofRows` built and lookup slots, and `HmRow.check`
refuses a lookup slot as the unit. -/
theorem unit_built {tags : Tags} {c : Circuit} (P : ParseFacts tags c) (C : CheckFacts c) : Built c.unit := by
  have hmem : c.nets.getD c.unitNet default ∈ c.nets.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem P.unitNet, Option.getD_some]
    exact Array.mem_def.1 (Array.getElem_mem P.unitNet)
  rcases P.nets _ hmem with h | ⟨_, _, _, _, _, hb⟩
  · exact h
  · have h1 := (build_facts hb).1
    have h2 : (c.nets.getD c.unitNet default).2.lookup.isSome = false := C.unit_lookup
    rw [h1] at h2
    cases h2

/-! ## Δ's later pairs are input bits -/

theorem unit_range_facts_of {c : Circuit} (hu : c.unitNet < c.nets.size) (C : CheckFacts c) :
    c.unitRange ∈ c.ranges.toList ∧ c.unitRange.kind = .net c.unitNet ∧ c.unitRange.count = c.g * c.upv := by
  obtain ⟨hm, hk, h0⟩ := rangeOf_spec (C.one_range _ hu)
  refine ⟨hm, hk, ?_⟩
  show (c.rangeOf c.unitNet).count = _
  rw [h0]
  exact C.unit_range.2

theorem unit_range_facts {tags : Tags} {c : Circuit} (P : ParseFacts tags c) (C : CheckFacts c) :
    c.unitRange ∈ c.ranges.toList ∧ c.unitRange.kind = .net c.unitNet ∧ c.unitRange.count = c.g * c.upv :=
  unit_range_facts_of P.unitNet C

theorem perVu_unit {c : Circuit} (C : CheckFacts c) (hur : c.unitRange ∈ c.ranges.toList)
    (hk : c.unitRange.kind = .net c.unitNet) (hcount : c.unitRange.count = c.g * c.upv) (hg : 0 < c.g) :
    c.perVu.getD c.unitNet 0 = c.upv := by
  have := C.count _ hur _ hk
  rw [hcount] at this
  exact (Nat.eq_of_mul_eq_mul_left hg this).symm

/-- **Every pair Δ writes after its first loop is at an input bit of a net slot**, for a circuit whose nets' ports lie in
their input rows and whose unit, `sha512x3` and `hm96` nets are among its nets. -/
theorem deltaIn_inBit_of {c : Circuit}
    (hports : ∀ n, ∀ q ∈ (portCols (c.nets.getD n default).2.inGroups).toList,
      q.1 + q.2 ≤ (c.nets.getD n default).2.inWords * WORD)
    (hu : c.unitNet < c.nets.size) (hs : c.shaNet < c.nets.size) (hh : c.hmNet < c.nets.size) (C : CheckFacts c)
    {x : ℕ} (hx : DeltaIn c x) : InBit c x := by
  have hrange : ∀ n < c.nets.size, c.rangeOf n ∈ c.ranges.toList ∧ (c.rangeOf n).kind = .net n ∧
      (c.rangeOf n).count = c.g * c.perVu.getD n 0 := by
    intro n hn
    obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range n hn)
    exact ⟨hm, hk, C.count _ hm _ hk⟩
  have hws := rowSlots_spec c.ports
  have hall := rowSlots_all c.ports
  have hbit : ∀ n < c.nets.size, ∀ g < c.g, ∀ i < c.perVu.getD n 0,
      ∀ q ∈ (portCols (c.nets.getD n default).2.inGroups).toList, ∀ t < q.2,
      InBit c (c.slotCol n g i + q.1 + t) := by
    intro n hn g hg i hi q hq t ht
    obtain ⟨hm, hk, hc⟩ := hrange n hn
    refine ⟨n, g * c.perVu.getD n 0 + i, q.1 + t, hm, hk, ?_, ?_, by unfold Circuit.slotCol; omega⟩
    · rw [hc]; exact Rows.mul_add_lt hg hi
    · have := hports n q hq; omega
  rcases hx with ⟨g, hg, p, hp, k, hk, i, hi, t, ht, rfl⟩ | ⟨g, hg, p, hp, i, hi, t, ht, rfl⟩ |
    ⟨k, hk, j, hj, t, ht, rfl⟩ | ⟨g, hg, e, he, t, ht, rfl⟩ | ⟨g, hg, w, hw, t, ht, rfl⟩
  · -- a bit of a row's chain
    obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hp) _ (getElem_mem_toList' hk)
    have hsz := C.sha_in.1
    have hidx : 2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + i < 6 := by omega
    refine hbit _ hs g hg _ (by rw [C.perVu_sha]; exact h1) _ (getElem_mem_toList' (by rw [hsz]; exact hidx)) t ?_
    rw [C.sha_in.2 _ hidx, sha_width _ _ hi]
    exact ht
  · -- a bit of a row's hm96 slot
    obtain ⟨hil, hw⟩ := hm_width C.hm_in hi
    refine hbit _ hh g hg p ?_ _ (getElem_mem_toList' hil) t (by rw [hw]; exact ht)
    rw [C.perVu_hm, ← hws]
    exact lt_of_lt_of_le hp (le_nextPow2 _)
  · -- a unit's leaf bit
    obtain ⟨hum, huk, huc⟩ := unit_range_facts_of hu C
    have hupv : 0 < c.upv := by
      rcases Nat.eq_zero_or_pos c.upv with h0 | h0
      · rw [huc, h0, Nat.mul_zero] at hk; omega
      · exact h0
    have hmod := Nat.mod_lt k hupv
    have hli : c.leavesIn.getD (k % c.upv) #[] = c.leavesIn[k % c.upv]! := by
      have hlt : k % c.upv < c.leavesIn.size := by rw [C.leaves.1]; exact hmod
      rw [getElem!_pos c.leavesIn (k % c.upv) hlt, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt,
        Option.getD_some]
    rw [hli, C.leaves.2 _ hmod] at hj
    have hsingle := portCols_single (c.unit.inGroups.getD 0 default)
    have hlen : (portCols #[c.unit.inGroups.getD 0 default]).size = (c.unit.inGroups.getD 0 default).bits.size := by
      have := congrArg List.length hsingle
      simpa using this
    have hjq : j < (portCols #[c.unit.inGroups.getD 0 default]).size := by rw [hlen]; exact hj
    have hq2 : (portCols #[c.unit.inGroups.getD 0 default])[j].2 = 16 := by
      have := congrArg (fun l : List ℕ => l[j]?) hsingle
      simp only [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hjq,
        Array.getElem?_eq_getElem hj, Option.map_some] at this
      rw [Option.some.inj this]
      exact C.unit_groups.2 _ (Array.mem_def.1 (Array.getElem_mem hj))
    have hq1 : ((portCols #[c.unit.inGroups.getD 0 default]).map (·.1)).getD j 0 =
        (portCols #[c.unit.inGroups.getD 0 default])[j].1 := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_getElem hjq]
      rfl
    have hg0 : c.unit.inGroups.getD 0 default ∈ c.unit.inGroups.toList :=
      getD_mem_toList (Nat.pos_of_ne_zero C.unit_groups.1)
    have hqm := mem_portCols hg0 (Array.mem_def.1 (Array.getElem_mem hjq))
    have hle := hports c.unitNet _ hqm
    rw [hq1]
    refine ⟨c.unitNet, k, (portCols #[c.unit.inGroups.getD 0 default])[j].1 + t, hum, huk, hk, ?_,
      by rw [Nat.add_assoc]; rfl⟩
    have : (portCols #[c.unit.inGroups.getD 0 default])[j].1 + (portCols #[c.unit.inGroups.getD 0 default])[j].2 ≤
      c.unit.inWords * WORD := hle
    omega
  · -- a unit's leaf-cut bit
    obtain ⟨hum, huk, huc⟩ := unit_range_facts_of hu C
    obtain ⟨he1, he2⟩ := C.cuts e he
    exact hbit _ hu g hg _ (by rw [perVu_unit C hum huk huc (by omega)]; exact he1) _
      (getElem_mem_toList' he2) t ht
  · -- a wire's destination bit
    obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
    exact hbit _ hd1 g hg _ hd2 _ (getD_mem_toList hd3) t ht

/-- **Every pair Δ writes after its first loop is at an input bit of a net slot.** -/
theorem deltaIn_inBit {tags : Tags} {c : Circuit} (P : ParseFacts tags c) (C : CheckFacts c) {x : ℕ}
    (hx : DeltaIn c x) : InBit c x :=
  deltaIn_inBit_of (fun n => (nets_ok P n).1.2.2) P.unitNet P.shaNet P.hmNet C hx


/-! ## The accepted statement's layout -/

/-- A range with a slot, inside the block, has slots no larger than the block. -/
theorem slotLog_le {c : Circuit} (C : CheckFacts c) {r : Range} (hr : r ∈ c.ranges.toList) (hc : 0 < r.count) :
    r.slotLog ≤ c.kLog := by
  have h1 := (C.bound_slotLog r hr).1
  have h2 := count_mul_le_spanH r
  have h3 : 2 ^ r.slotLog ≤ r.count * 2 ^ r.slotLog := Nat.le_mul_of_pos_left _ hc
  have : 2 ^ r.slotLog ≤ 2 ^ c.kLog := by omega
  exact (Nat.pow_le_pow_iff_right (by norm_num)).1 this

/-- **A statement with the circuit `HmRow.parse` accepted has the layout the placement reads** (`Layout`): from its nets,
`HmRow.check`'s layout, `HmRow.pin` and `HmRow.delta`, for any circuit `HmRow.parse` accepted without a typed template
(an untyped statement's, or a typed flat class's). -/
theorem layout_of {tags : Tags} {st : Stmt} (P : ParseFacts tags st.c) (hpin : HmRow.pin st.c = .ok st.pin)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    (hm : st.m = st.c.kLog + st.nbl) (hk : st.c.kLog ≤ 27) : Layout st := by
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  obtain ⟨hum, huk, huc⟩ := unit_range_facts P C
  have hU := unit_built P C
  obtain ⟨hUlk, -, -, hUin, -⟩ := built_spec hU
  obtain ⟨rp, hrp, ip, hkp, hcp, hpv⟩ := pin_spec hpin
  have hip : ip < st.c.nets.size := P.kinds rp hrp ip hkp
  have hsl := slotLog_le C hrp hcp
  have hsl' : rp.slotLog = (st.c.nets.getD ip default).2.unitLog := ((C.bound_slotLog rp hrp).2 ip hkp).symm
  obtain ⟨h7, hcr⟩ := (nets_ok P ip).2 hip
  exact {
    disj := C.disj
    bound := fun r hr => (C.bound_slotLog r hr).1
    slotLog := fun r hr i hi => ((C.bound_slotLog r hr).2 i hi).symm
    fit := fun i => ⟨(nets_ok P i).1.1, (nets_ok P i).1.2.1⟩
    unit_mem := hum
    unit_kind := huk
    unit_count := huc
    unit_built := by rw [lunit_flat P.tmpl]; exact hU
    unit_at := UnitAt.flat P.tmpl hU hUin
    unit_lookup := hUlk
    unit_in := hUin
    unit_const := built_constRow hU
    pin := ⟨rp, hrp, ip, hkp, hcp, hpv, hcr (by omega)⟩
    delta_b := by rw [hdb, hda, delta_snd _ _ P.typed]
    delta := by
      obtain ⟨rest, h1, h2⟩ := delta_split st.c st.pin P.typed
      exact ⟨rest, by rw [hda]; exact h1, fun p hp => deltaIn_inBit P C (h2 p hp)⟩
    kLog := by omega
    kLog6 := by omega
    m := hm }

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **The untyped statement `Stmt.setupH` accepted has the layout the placement reads** (`Layout`): from `HmRow.parse`'s nets,
`HmRow.check`'s layout, `HmRow.pin` and `HmRow.delta`. -/
theorem setupH_layout (ht : tags.typed = false) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    Layout st :=
  let ⟨hp, hpin, hda, hdb, hm, hk, _⟩ := setupH_spec ht h
  layout_of (parse_facts hp) hpin hda hdb hm hk

/-- **The lowering's placement, from the executable's accept step.** When `Stmt.setupH` accepts a statement, every
template instance `g` has its `units_per_vu` unit slots' rows (the unit net's rows stacked, `Rows.stack`) placed at
`ur.slot (g·upv + u)`, the shared constant at the pin, in the model statement of the level-0 matrices the verifier folds
(`placedA`, `placedB`). -/
theorem placement_of_setupH (ht : tags.typed = false) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g : ℕ} (hg : g < st.c.g) :
    Placement ((setupH_layout ht h).model regs) ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv)
      ((setupH_layout ht h).col regs hg) :=
  (setupH_layout ht h).placement regs hg

/-- **A drawn unit's place, from the executable's accept step** (`UnitPlace`, which `loweringSoundB_of_place` and
the `_placed` forms take): template instance `g`'s stacked unit slots in block `o` of the accepted statement, given the unit's
instance in the audit circuit, whose inputs on one gate carry one value there (`Layout.Aliased`). -/
noncomputable def unitPlace_of_setupH (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hal : (setupH_layout ht h).Aliased regs inst o hg) :
    UnitPlace P ((setupH_layout ht h).model regs) u :=
  (setupH_layout ht h).unitPlace regs inst o hg hal

end FlockSoundness
