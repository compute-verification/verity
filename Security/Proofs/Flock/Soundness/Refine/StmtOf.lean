import Proofs.Flock.Soundness.Refine.Realizes
import Proofs.Flock.Level3.Folded

/-!
# The statement decoding, part 1: the executable's layout is a block R1CS (PROTOCOL.md §10, §16.1)

The executable never holds `A₀` and `B₀`: its lincheck fold (`Stmt.fold`) reads them off the circuit's layout. `stmtOf st`
is the model statement those matrices make. Entry `(i, c)` of `A₀` or `B₀` counts, mod 2:
* the covering range's slot matrix at `(i mod 2^sl, c mod 2^sl)`, when row `i` is in column `c`'s slot. The covering
  range is the last in the layout that covers `c`, since the fold overwrites (`lastCover`). A mask slot is the identity
  on both sides;
* for a lookup slot, on the B side, the product rows its table's words send to `c` (`lkCnt`);
* the pairs `(i, c)` of Δ_A or Δ_B.

`ofCircuit_fold`: the fold `Setup.ofCircuit` hands the verifier realizes `stmtOf st` (`FoldRealizes`, R3), given the
layout facts `StmtWF`: `6 ≤ k_log ≤ m`, the pin inside the block, and every range's slots between 2^6 and 2^k_log. The
regions are a parameter here.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

/-! ### The matrices -/

/-- The occurrences of column `c` in row `i` of a sparse matrix: what `foldT` adds at `c` (none past its rows or its
width). -/
def spCnt (m : Sparse) (i c : ℕ) : ℕ :=
  if i < m.rows ∧ c < m.width then ((Finset.range m.len[i]!).filter fun t => m.col i t = c).card else 0

/-- The product rows a lookup table sends to column `c` (§16.5): row `prod0 + j·2^(n−lo) + h` for each set bit `j < bits`
of word `(h, l)` with `loTop + l = c`. -/
def lkCnt (lk : LookupTable) (i c : ℕ) : ℕ :=
  ((Finset.range (2 ^ (lk.n - lk.lo)) ×ˢ Finset.range (2 ^ lk.lo) ×ˢ Finset.range 32).filter fun x =>
    lk.loTop + x.2.1 = c ∧ (Lookup.word lk (2 ^ lk.lo) (2 ^ lk.bits - 1) x.1 x.2.1).testBit x.2.2 = true ∧
      x.2.2 < lk.bits ∧ lk.prod0 + x.2.2 * 2 ^ (lk.n - lk.lo) + x.1 = i).card

/-- A slot type's A side at local row `i` and column `c`. -/
def slotA (st : Stmt) : Kind → ℕ → ℕ → ℕ
  | .comp => spCnt st.leaf.comp.a
  | .net j => spCnt (st.c.nets.getD j default).2.a
  | .mask => fun i c => if i = c then 1 else 0

/-- A lookup slot's table rows, inside the B matrix's width. -/
def lkSide (b : Sparse) : Option LookupTable → ℕ → ℕ → ℕ
  | some lk => fun i c => if c < b.width then lkCnt lk i c else 0
  | none => fun _ _ => 0

/-- A slot type's B side at local row `i` and column `c`; a lookup slot adds its table's product rows. -/
def slotB (st : Stmt) : Kind → ℕ → ℕ → ℕ
  | .comp => spCnt st.leaf.comp.b
  | .net j => fun i c => spCnt (st.c.nets.getD j default).2.b i c +
      lkSide (st.c.nets.getD j default).2.b (st.c.nets.getD j default).2.lookup i c
  | .mask => fun i c => if i = c then 1 else 0

/-- The last range of `rs` whose slots cover column `c`: the one whose base the fold leaves at `c`. -/
def lastCover : List Range → ℕ → Option Range
  | [], _ => none
  | r :: rs, c => (lastCover rs c).or (if covers r c then some r else none)

/-- The ranges' share of entry `(i, c)`: the covering range's slot side, when row `i` is in column `c`'s slot. -/
def rangeCnt (side : Kind → ℕ → ℕ → ℕ) (rs : List Range) (i c : ℕ) : ℕ :=
  match lastCover rs c with
  | none => 0
  | some r => if i / 2 ^ r.slotLog = c / 2 ^ r.slotLog then side r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) else 0

theorem rangesVal_lastCover (st : Stmt) (α : F128) (lw ρ : Array F128) :
    ∀ (rs : List Range) (c : ℕ) (v : F128), rangesVal st α lw ρ rs c v =
      match lastCover rs c with
      | none => v
      | some r => (eHiFor ρ r)[c / 2 ^ r.slotLog]! * (baseFor st α lw ρ r)[c % 2 ^ r.slotLog]!
  | [], _, _ => rfl
  | r :: rs, c, v => by
      rw [rangesVal, rangesVal_lastCover st α lw ρ rs c, lastCover]
      cases lastCover rs c with
      | some r' => rfl
      | none => by_cases h : covers r c <;> simp [h]

theorem lastCover_spec : ∀ (rs : List Range) (c : ℕ) (r : Range), lastCover rs c = some r → r ∈ rs ∧ covers r c
  | [], _, _, h => by cases h
  | r₀ :: rs, c, r, h => by
      rw [lastCover] at h
      cases h' : lastCover rs c with
      | some r' =>
          rw [h'] at h
          simp only [Option.some_or, Option.some.injEq] at h
          subst h
          obtain ⟨hm, hc⟩ := lastCover_spec rs c _ h'
          exact ⟨List.mem_cons_of_mem _ hm, hc⟩
      | none =>
          rw [h'] at h
          by_cases hc : covers r₀ c
          · simp only [hc, ite_true, Option.none_or, Option.some.injEq] at h
            subst h
            exact ⟨List.mem_cons_self, hc⟩
          · simp [hc] at h

/-! ### Counting in characteristic 2 -/

theorem cast_zmod (n : ℕ) : (((n : ZMod 2).val : ℕ) : 𝔽) = (n : 𝔽) := by
  rw [ZMod.val_natCast, CharP.cast_eq_mod 𝔽 2 n]

theorem sum_ite_card (s : Finset ℕ) (p : ℕ → Prop) [DecidablePred p] (x : 𝔽) :
    (∑ t ∈ s, if p t then x else 0) = ((s.filter p).card : 𝔽) * x := by
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]

/-- Δ's pairs at column `c`, as a sum over rows of their counts. -/
theorem delta_sum (f : ℕ → 𝔽) (N c : ℕ) (hf : ∀ i, N ≤ i → f i = 0) : ∀ L : List (ℕ × ℕ),
    (L.map fun ic => if ic.2 = c then f ic.1 else 0).sum = ∑ i ∈ Finset.range N, (L.count (i, c) : 𝔽) * f i
  | [] => by simp
  | x :: L => by
      rw [List.map_cons, List.sum_cons, delta_sum f N c hf L]
      simp only [List.count_cons, Nat.cast_add, add_mul, Finset.sum_add_distrib]
      rw [add_comm]
      congr 1
      obtain ⟨a, b⟩ := x
      simp only [beq_iff_eq, Prod.mk.injEq]
      by_cases hb : b = c
      · subst hb
        simp only [and_true]
        rw [Finset.sum_eq_single a (fun i _ hi => by simp [Ne.symm hi]) (fun ha => by
          rw [hf a (by simpa using ha)]; simp)]
        simp
      · simp [hb]

/-! ### One slot -/

theorem foldBHigh_size (lk : LookupTable) (nh nlo wmask : ℕ) (e : Array F128) :
    ∀ (fuel h : ℕ) (acc : Array F128), (Lookup.foldBHigh lk nh nlo wmask e fuel h acc).size = acc.size := by
  intro fuel; induction fuel with
  | zero => intro h acc; rfl
  | succ fuel ih => intro h acc; simp only [Lookup.foldBHigh]; rw [ih, foldBLow_size]

theorem foldB_size (lk : LookupTable) (e acc0 : Array F128) : (Lookup.foldB lk e acc0).size = acc0.size :=
  foldBHigh_size _ _ _ _ _ _ _ _

theorem of_getElem!_oob {a : Array F128} {i : ℕ} (h : a.size ≤ i) : GF128.of a[i]! = 0 := by
  rw [getElem!_neg a i (by omega)]; exact of_zero

/-- A slot's base at any column: each side is zero past its matrix's width. -/
theorem baseOf_get' (α : F128) (eLo : Array F128) (a b : Sparse) (lk : Option LookupTable) (sl c : ℕ)
    (hc : c < 2 ^ sl) :
    GF128.of (Stmt.baseOf α eLo a b lk sl)[c]! =
      GF128.of α * (if c < a.width then sparseT a eLo c else 0) +
        (if c < b.width then sparseT b eLo c + lookupTerm eLo lk c else 0) := by
  have hA : GF128.of (a.foldT eLo)[c]! = if c < a.width then sparseT a eLo c else 0 := by
    split_ifs with ha
    · exact foldT_get a eLo c ha
    · exact of_getElem!_oob (by rw [foldT_size]; omega)
  unfold Stmt.baseOf
  rw [getElem!_pos _ c (by simpa using hc), Array.getElem_map, Array.getElem_range, of_add, of_mul, hA]
  refine congrArg (fun x : 𝔽 => GF128.of α * (if c < a.width then sparseT a eLo c else 0) + x) ?_
  split_ifs with hb
  · cases lk with
    | none => dsimp only; simp only [lookupTerm, add_zero]; exact foldT_get b eLo c hb
    | some t =>
        dsimp only
        rw [foldB_get t eLo _ c (by rw [foldT_size]; exact hb), foldT_get b eLo c hb]; rfl
  · cases lk with
    | none => dsimp only; exact of_getElem!_oob (by rw [foldT_size]; omega)
    | some t => dsimp only; exact of_getElem!_oob (by rw [foldB_size, foldT_size]; omega)

/-- When the fold succeeds, every range's slot type folds: `baseFor` is its `slotBase`. -/
theorem foldRanges_slotBase (st : Stmt) (α : F128) (lw ρ : Array F128) : ∀ (rs : List Range)
    (bases : Array ((Kind × ℕ) × Array F128)) (out res : Array F128),
    st.foldRanges α lw ρ rs bases out = .ok res →
    (∀ kb ∈ bases, st.slotBase α lw ρ kb.1 = .ok kb.2) →
    ∀ r ∈ rs, st.slotBase α lw ρ (r.kind, r.slotLog) = .ok (baseFor st α lw ρ r) := by
  intro rs
  induction rs with
  | nil => intro _ _ _ _ _ r hr; cases hr
  | cons r rs ih =>
    intro bases out res h hinv r' hr'
    simp only [Stmt.foldRanges] at h
    cases hf : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      rw [hf] at h
      simp only [bind, Except.bind, pure, Except.pure] at h
      have hkb := Array.find?_some hf
      have hmem := Array.mem_of_find?_eq_some hf
      have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true hkb
      rcases List.mem_cons.1 hr' with rfl | hr'
      · rw [baseFor, ← hkey, hinv kb hmem]
      · exact ih bases _ res h hinv r' hr'
    | none =>
      rw [hf] at h
      cases hb : st.slotBase α lw ρ (r.kind, r.slotLog) with
      | error e => rw [hb] at h; simp [bind, Except.bind] at h
      | ok b =>
        rw [hb] at h
        simp only [bind, Except.bind, pure, Except.pure] at h
        have hinv' : ∀ kb ∈ bases.push ((r.kind, r.slotLog), b), st.slotBase α lw ρ kb.1 = .ok kb.2 := by
          intro kb hkb
          rcases Array.mem_push.mp hkb with h1 | h1
          · exact hinv kb h1
          · rw [h1]; exact hb
        rcases List.mem_cons.1 hr' with rfl | hr'
        · rw [baseFor, hb]
        · exact ih _ _ res h hinv' r' hr'

theorem eqAt_vecOf (n : ℕ) (ρ : Array F128) (h : ρ.size = n) (b : ℕ) :
    Model.eqAt (vecOf n ρ) b = FlockLevel3.eqAt (fun j : Fin ρ.size => GF128.of ρ[j.val]!) b := by
  subst h; rfl

theorem pow_K (K : ℕ) (hK : 6 ≤ K) : 2 ^ K = 2 ^ (K - 6) * 64 := by
  rw [show (64 : ℕ) = 2 ^ 6 from rfl, ← pow_add]; congr 1; omega

/-- The model's weight of row `i` is the executable's `e(i)`. -/
theorem e_model (lw ρ : Array F128) (K : ℕ) (hK : 6 ≤ K) (hρ : ρ.size = K - 6) (i : ℕ) (hi : i < 2 ^ K) :
    fOf lw[i % 64]! * Model.eqAt (vecOf (K - 6) ρ) (i / 64) = GF128.of (eIn lw ρ i) := by
  rw [eIn, of_mul, eqAt_vecOf (K - 6) ρ hρ, ← eqTable_eqAt ρ (i / 64) ?_]
  rw [hρ, Nat.div_lt_iff_lt_mul (by norm_num), ← pow_K K hK]; exact hi

/-- Past the block, `e` is zero. -/
theorem eIn_oob (lw ρ : Array F128) (K : ℕ) (hK : 6 ≤ K) (hρ : ρ.size = K - 6) (i : ℕ) (hi : 2 ^ K ≤ i) :
    GF128.of (eIn lw ρ i) = 0 := by
  rw [eIn, of_mul, of_getElem!_oob (a := eqTable ρ) ?_, mul_zero]
  rw [eqTable_size, hρ, Nat.le_div_iff_mul_le (by norm_num), ← pow_K K hK]; exact hi

/-- The rows of one slot: `i = q·2^sl + i'`. -/
theorem sum_slot {M : Type} [AddCommMonoid M] (K sl q : ℕ) (hsl : sl ≤ K) (hq : q < 2 ^ (K - sl)) (g : ℕ → M) :
    ∑ i ∈ Finset.range (2 ^ K), (if i / 2 ^ sl = q then g i else 0) =
      ∑ i' ∈ Finset.range (2 ^ sl), g (q * 2 ^ sl + i') := by
  rw [← Finset.sum_filter]
  have hp : 0 < 2 ^ sl := Nat.two_pow_pos _
  have hK : 2 ^ K = 2 ^ (K - sl) * 2 ^ sl := by rw [← pow_add]; congr 1; omega
  apply Finset.sum_nbij' (fun i => i % 2 ^ sl) (fun i' => q * 2 ^ sl + i')
  · intro i _; simpa using Nat.mod_lt _ hp
  · intro i' hi'
    rw [Finset.mem_range] at hi'
    rw [Finset.mem_filter, Finset.mem_range]
    refine ⟨?_, ?_⟩
    · have : (q + 1) * 2 ^ sl ≤ 2 ^ (K - sl) * 2 ^ sl := Nat.mul_le_mul_right _ hq
      rw [hK]; nlinarith
    · rw [mul_comm, Nat.mul_add_div hp, Nat.div_eq_of_lt hi', add_zero]
  · intro i hi
    rw [Finset.mem_filter] at hi
    rw [← hi.2, mul_comm]; exact Nat.div_add_mod i (2 ^ sl)
  · intro i' hi'
    rw [Finset.mem_range] at hi'
    rw [mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hi']
  · intro i hi
    rw [Finset.mem_filter] at hi
    rw [← hi.2, mul_comm, Nat.div_add_mod i (2 ^ sl)]

theorem range_min (a b : ℕ) : Finset.range (min a b) = (Finset.range b).filter (· < a) := by
  ext i; simp only [Finset.mem_range, Finset.mem_filter]; omega

/-- A slot matrix's side at column `c`, scaled into the block: `Σ_{i'} cnt(i', c)·e(q·2^sl + i')`. -/
theorem sp_scale (lw ρ : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ ρ.size)
    (q : ℕ) (hq : q < 2 ^ (ρ.size - (sl - 6))) (m : Sparse) (c : ℕ) :
    GF128.of (eqTable (ρ.extract (sl - 6) ρ.size))[q]! *
        (if c < m.width then sparseT m (Stmt.tensor lw (ρ.extract 0 (sl - 6))) c else 0) =
      ∑ i ∈ Finset.range (2 ^ sl), (spCnt m i c : 𝔽) * GF128.of (eIn lw ρ (q * 2 ^ sl + i)) := by
  split_ifs with hc
  · rw [sparseT_scale lw ρ hlw sl hsl hslr q hq m c, range_min, Finset.sum_filter]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [sum_ite_card]
    unfold spCnt
    by_cases hi : i < m.rows
    · rw [ite_eq_left hi, ite_eq_left ⟨hi, hc⟩]
    · rw [ite_eq_right hi, ite_eq_right (fun h => hi h.1)]; simp
  · simp [spCnt, hc]

theorem of_zero' : GF128.of 0 = 0 := GF128.zero_eq.symm

theorem getD_of_lt {a : Array F128} {i : ℕ} (h : i < a.size) : a.getD i 0 = a[i]! := by
  rw [getElem!_pos a i h]; simp [Array.getD, h]

theorem getD_of_ge {a : Array F128} {i : ℕ} (h : a.size ≤ i) : a.getD i 0 = 0 := by
  simp [Array.getD, show ¬ i < a.size by omega]

/-- A lookup slot's table rows at column `c`, scaled into the block. -/
theorem lk_scale (lw ρ : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ ρ.size)
    (q : ℕ) (hq : q < 2 ^ (ρ.size - (sl - 6))) (b : Sparse) (lk : Option LookupTable) (c : ℕ) :
    GF128.of (eqTable (ρ.extract (sl - 6) ρ.size))[q]! *
        (if c < b.width then lookupTerm (Stmt.tensor lw (ρ.extract 0 (sl - 6))) lk c else 0) =
      ∑ i ∈ Finset.range (2 ^ sl), (lkSide b lk i c : 𝔽) * GF128.of (eIn lw ρ (q * 2 ^ sl + i)) := by
  cases lk with
  | none => simp [lookupTerm, lkSide]
  | some t =>
    split_ifs with hc
    · set eLo := Stmt.tensor lw (ρ.extract 0 (sl - 6)) with heLo
      set eh := GF128.of (eqTable (ρ.extract (sl - 6) ρ.size))[q]! with heh
      set E : ℕ → 𝔽 := fun i => GF128.of (eIn lw ρ (q * 2 ^ sl + i)) with hE
      set nh := 2 ^ (t.n - t.lo)
      set T := Finset.range nh ×ˢ Finset.range (2 ^ t.lo) ×ˢ Finset.range 32 with hT
      let idx : ℕ × ℕ × ℕ → ℕ := fun x => t.prod0 + x.2.2 * nh + x.1
      let P : ℕ × ℕ × ℕ → Prop := fun x =>
        t.loTop + x.2.1 = c ∧ (Lookup.word t (2 ^ t.lo) (2 ^ t.bits - 1) x.1 x.2.1).testBit x.2.2 = true ∧
          x.2.2 < t.bits
      have hsz : eLo.size = 2 ^ sl := tensor_size lw ρ hlw sl hsl hslr
      have hterm : ∀ i, eh * GF128.of (eLo.getD i 0) = if i < 2 ^ sl then E i else 0 := by
        intro i
        split_ifs with hi
        · rw [getD_of_lt (by rw [hsz]; exact hi)]
          exact slot_scale lw ρ hlw sl hsl hslr q i hq hi
        · rw [getD_of_ge (by rw [hsz]; omega), of_zero', mul_zero]
      have hL : eh * lookupTerm eLo (some t) c =
          ∑ x ∈ T, if P x ∧ idx x < 2 ^ sl then E (idx x) else 0 := by
        simp only [lookupTerm, wordSum, Lookup.ej]
        refine (Finset.mul_sum _ _ _).trans ?_
        rw [hT, Finset.sum_product]
        refine Finset.sum_congr rfl fun h _ => ?_
        refine (Finset.mul_sum _ _ _).trans ?_
        rw [Finset.sum_product]
        refine Finset.sum_congr rfl fun l _ => ?_
        by_cases hl : t.loTop + l = c
        · rw [ite_eq_left hl]
          refine (Finset.mul_sum _ _ _).trans ?_
          refine Finset.sum_congr rfl fun j _ => ?_
          by_cases hb : (Lookup.word t (2 ^ t.lo) (2 ^ t.bits - 1) h l).testBit j = true
          · by_cases hj : j < t.bits
            · simp only [hb, ite_true, hj, hterm, P, idx, hl, true_and]; rfl
            · simp [hb, hj, P, of_zero']
          · simp [hb, P]
        · rw [ite_eq_right hl]
          simp [P, hl]
      have hR : ∀ i ∈ Finset.range (2 ^ sl), (lkSide b (some t) i c : 𝔽) * E i =
          ∑ x ∈ (T.filter fun x => P x ∧ idx x < 2 ^ sl).filter (fun x => idx x = i), E (idx x) := by
        intro i hi
        rw [Finset.mem_range] at hi
        rw [Finset.sum_congr rfl fun x hx => by rw [(Finset.mem_filter.1 hx).2], Finset.sum_const, nsmul_eq_mul]
        congr 2
        simp only [lkSide, hc, ite_true, lkCnt, Finset.filter_filter]
        congr 1
        refine Finset.filter_congr fun x _ => ?_
        simp only [P, idx]
        constructor
        · rintro ⟨h1, h2, h3, h4⟩; exact ⟨⟨⟨h1, h2, h3⟩, by rw [h4]; exact hi⟩, h4⟩
        · rintro ⟨⟨⟨h1, h2, h3⟩, _⟩, h4⟩; exact ⟨h1, h2, h3, h4⟩
      rw [hL, Finset.sum_congr rfl hR, Finset.sum_fiberwise_of_maps_to (g := idx) (fun x hx => by
        simpa using (Finset.mem_filter.1 hx).2.2), Finset.sum_filter]
    · simp [lkSide, hc]

/-- A unit or compression slot's base at column `c`, scaled into the block: its rows' counts times their weights. -/
theorem slot_sides (lw ρ : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ ρ.size)
    (q : ℕ) (hq : q < 2 ^ (ρ.size - (sl - 6))) (α : F128) (a b : Sparse) (lk : Option LookupTable) (c : ℕ)
    (hc : c < 2 ^ sl) :
    GF128.of (eqTable (ρ.extract (sl - 6) ρ.size))[q]! *
        GF128.of (Stmt.baseOf α (Stmt.tensor lw (ρ.extract 0 (sl - 6))) a b lk sl)[c]! =
      ∑ i ∈ Finset.range (2 ^ sl), GF128.of (eIn lw ρ (q * 2 ^ sl + i)) *
        (GF128.of α * (spCnt a i c : 𝔽) + ((spCnt b i c + lkSide b lk i c : ℕ) : 𝔽)) := by
  rw [baseOf_get' α _ a b lk sl c hc]
  have hsplit : (if c < b.width then sparseT b (Stmt.tensor lw (ρ.extract 0 (sl - 6))) c +
      lookupTerm (Stmt.tensor lw (ρ.extract 0 (sl - 6))) lk c else 0) =
      (if c < b.width then sparseT b (Stmt.tensor lw (ρ.extract 0 (sl - 6))) c else 0) +
        (if c < b.width then lookupTerm (Stmt.tensor lw (ρ.extract 0 (sl - 6))) lk c else 0) := by
    split_ifs <;> simp
  have e1 : GF128.of α * (GF128.of (eqTable (ρ.extract (sl - 6) ρ.size))[q]! *
      (if c < a.width then sparseT a (Stmt.tensor lw (ρ.extract 0 (sl - 6))) c else 0)) =
      ∑ i ∈ Finset.range (2 ^ sl), GF128.of α * ((spCnt a i c : 𝔽) * GF128.of (eIn lw ρ (q * 2 ^ sl + i))) := by
    rw [sp_scale lw ρ hlw sl hsl hslr q hq a c]; exact Finset.mul_sum _ _ _
  rw [hsplit, mul_add, mul_add, mul_left_comm, e1, sp_scale lw ρ hlw sl hsl hslr q hq b c,
    lk_scale lw ρ hlw sl hsl hslr q hq b lk c, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  push_cast; ring

/-- The covering range's slot at column `c`, for each slot type. -/
theorem slot_part (st : Stmt) (α : F128) (lw ρ : Array F128) (hlw : lw.size = 64) (r : Range)
    (hsl : 6 ≤ r.slotLog) (hslr : r.slotLog - 6 ≤ ρ.size) (c : ℕ)
    (hq : c / 2 ^ r.slotLog < 2 ^ (ρ.size - (r.slotLog - 6)))
    (hbase : st.slotBase α lw ρ (r.kind, r.slotLog) = .ok (baseFor st α lw ρ r)) :
    GF128.of ((eHiFor ρ r)[c / 2 ^ r.slotLog]! * (baseFor st α lw ρ r)[c % 2 ^ r.slotLog]!) =
      ∑ i ∈ Finset.range (2 ^ r.slotLog), GF128.of (eIn lw ρ (c / 2 ^ r.slotLog * 2 ^ r.slotLog + i)) *
        (GF128.of α * (slotA st r.kind i (c % 2 ^ r.slotLog) : 𝔽) + (slotB st r.kind i (c % 2 ^ r.slotLog) : 𝔽)) := by
  have hc' : c % 2 ^ r.slotLog < 2 ^ r.slotLog := Nat.mod_lt _ (Nat.two_pow_pos _)
  unfold eHiFor
  revert hbase
  cases hk : r.kind with
  | comp =>
      intro hbase
      simp only [Stmt.slotBase] at hbase
      split_ifs at hbase with hs
      · cases hbase
      · simp only [pure, Except.pure, Except.ok.injEq] at hbase
        rw [← hbase, of_mul, slot_sides lw ρ hlw _ hsl hslr _ hq α _ _ none _ hc']
        simp [slotA, slotB, lkSide]
  | net j =>
      intro hbase
      simp only [Stmt.slotBase] at hbase
      split_ifs at hbase with hs
      · cases hbase
      · simp only [pure, Except.pure, Except.ok.injEq] at hbase
        rw [← hbase, of_mul, slot_sides lw ρ hlw _ hsl hslr _ hq α _ _ _ _ hc']
        simp only [slotA, slotB, Nat.cast_add]
  | mask =>
      intro hbase
      simp only [Stmt.slotBase, pure, Except.pure, Except.ok.injEq] at hbase
      rw [← hbase, mask_term lw ρ hlw _ hsl hslr α c hq]
      rw [Finset.sum_eq_single (c % 2 ^ r.slotLog) (fun i _ hi => by simp [slotA, slotB, hi])
        (fun h => absurd (Finset.mem_range.2 hc') h)]
      simp only [slotA, slotB, ite_true, Nat.cast_one, mul_one, Nat.div_add_mod' c]
      ring

theorem div_lt_block (K sl c : ℕ) (hsl : sl ≤ K) (hc : c < 2 ^ K) : c / 2 ^ sl < 2 ^ (K - sl) := by
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← pow_add, Nat.sub_add_cancel hsl]; exact hc

/-- The ranges' share of the fold at column `c`: the covering range's slot, on the rows of `c`'s slot. -/
theorem ranges_part (st : Stmt) (hk6 : 6 ≤ st.c.kLog)
    (hslots : ∀ r ∈ st.c.ranges.toList, 6 ≤ r.slotLog ∧ r.slotLog ≤ st.c.kLog)
    (α : F128) (lw ρ v : Array F128) (hlw : lw.size = 64) (hρ : ρ.size = st.c.kLog - 6)
    (hv : st.fold α lw ρ = .ok v) (c : ℕ) (hc : c < 2 ^ st.c.kLog) :
    GF128.of (rangesVal st α lw ρ st.c.ranges.toList c 0) =
      ∑ i ∈ Finset.range (2 ^ st.c.kLog), GF128.of (eIn lw ρ i) *
        (GF128.of α * (rangeCnt (slotA st) st.c.ranges.toList i c : 𝔽) +
          (rangeCnt (slotB st) st.c.ranges.toList i c : 𝔽)) := by
  have hfr : ∃ res, st.foldRanges α lw ρ st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) = .ok res := by
    simp only [Stmt.fold] at hv
    split_ifs at hv with hck
    · cases hv
    cases hr : st.foldRanges α lw ρ st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) with
    | error e => rw [hr] at hv; cases hv
    | ok res => exact ⟨res, rfl⟩
  obtain ⟨res, hres⟩ := hfr
  have hsb := foldRanges_slotBase st α lw ρ _ #[] _ res hres (by simp)
  rw [rangesVal_lastCover]
  unfold rangeCnt
  cases hlc : lastCover st.c.ranges.toList c with
  | none => simp [of_zero']
  | some r =>
      obtain ⟨hmem, _⟩ := lastCover_spec _ _ _ hlc
      obtain ⟨h6, hK⟩ := hslots r hmem
      have hq : c / 2 ^ r.slotLog < 2 ^ (st.c.kLog - r.slotLog) := div_lt_block _ _ c hK hc
      have hq' : c / 2 ^ r.slotLog < 2 ^ (ρ.size - (r.slotLog - 6)) := by
        rw [hρ, show st.c.kLog - 6 - (r.slotLog - 6) = st.c.kLog - r.slotLog by omega]; exact hq
      dsimp only
      rw [slot_part st α lw ρ hlw r h6 (by omega) c hq' (hsb r hmem)]
      have hite : ∀ i, GF128.of (eIn lw ρ i) *
          (GF128.of α * ((if i / 2 ^ r.slotLog = c / 2 ^ r.slotLog then
              slotA st r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) else 0 : ℕ) : 𝔽) +
            ((if i / 2 ^ r.slotLog = c / 2 ^ r.slotLog then
              slotB st r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) else 0 : ℕ) : 𝔽)) =
          if i / 2 ^ r.slotLog = c / 2 ^ r.slotLog then GF128.of (eIn lw ρ i) *
            (GF128.of α * (slotA st r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) : 𝔽) +
              (slotB st r.kind (i % 2 ^ r.slotLog) (c % 2 ^ r.slotLog) : 𝔽)) else 0 := by
        intro i; split_ifs <;> simp
      rw [Finset.sum_congr rfl fun i _ => hite i, sum_slot _ _ _ hK hq]
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [Finset.mem_range] at hi
      rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hi]

/-! ### The statement -/

/-- The pairs `(i, c)` of Δ. -/
def dCnt (d : Array (ℕ × ℕ)) (i c : ℕ) : ℕ := d.toList.count (i, c)

/-- `A₀` read off the executable's layout: the ranges' A sides and Δ_A, mod 2. -/
def matA (st : Stmt) (i c : ℕ) : ZMod 2 := (rangeCnt (slotA st) st.c.ranges.toList i c + dCnt st.da i c : ℕ)

/-- `B₀` read off the executable's layout: the ranges' B sides, lookup tables included, and Δ_B, mod 2. -/
def matB (st : Stmt) (i c : ℕ) : ZMod 2 := (rangeCnt (slotB st) st.c.ranges.toList i c + dCnt st.db i c : ℕ)

/-- The layout facts that reading the fold as matrices needs. -/
structure StmtWF (st : Stmt) : Prop where
  hk6 : 6 ≤ st.c.kLog
  hk : st.c.kLog ≤ st.m
  pin : st.pin < 2 ^ st.c.kLog
  slots : ∀ r ∈ st.c.ranges.toList, 6 ≤ r.slotLog ∧ r.slotLog ≤ st.c.kLog

/-- **The model statement of an executable statement** (§16.1): its `m`, `k_log` and pin, the matrices its layout
makes, and the given regions. -/
def stmtOf (st : Stmt) (h : StmtWF st) (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    Model.Statement where
  m := st.m
  kLog := st.c.kLog
  hk := h.hk
  hk6 := h.hk6
  pin := ⟨st.pin, h.pin⟩
  A₀ := fun i c => matA st i.val c.val
  B₀ := fun i c => matB st i.val c.val
  regions := regions

/-- **The circuit fold realizes `stmtOf`** (§10, §16.1). When the executable's fold succeeds on `α`, `lw` and `ρ`, entry
`c` is `Σ_i e_i·(α·A₀[i, c] + B₀[i, c])` for the matrices `stmtOf` reads off the layout. -/
theorem stmtOf_fold (st : Stmt) (h : StmtWF st) (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    FoldRealizes st.circuitFold (stmtOf st h regions) where
  kLog := rfl
  pin := rfl
  fold := by
    intro α lw ρ v hlw hρ hv col
    have hρ' : ρ.size = st.c.kLog - 6 := hρ
    have hcol : col.val < 2 ^ st.c.kLog := col.isLt
    have hv' : st.fold α lw ρ = .ok v := hv
    have hE0 : ∀ i, 2 ^ st.c.kLog ≤ i → GF128.of (eIn lw ρ i) = 0 := fun i hi => eIn_oob lw ρ _ h.hk6 hρ' i hi
    show GF128.of v[col.val]! = ∑ i : Fin (2 ^ st.c.kLog), fOf lw[i.val % 64]! *
      Model.eqAt (vecOf (st.c.kLog - 6) ρ) (i.val / 64) *
        (fOf α * ((matA st i.val col.val).val : 𝔽) + ((matB st i.val col.val).val : 𝔽))
    rw [fold_get st α lw ρ v hv' col.val hcol, ranges_part st h.hk6 h.slots α lw ρ v hlw hρ' hv' col.val hcol,
      Fin.sum_univ_eq_sum_range (fun i => fOf lw[i % 64]! * Model.eqAt (vecOf (st.c.kLog - 6) ρ) (i / 64) *
        (fOf α * ((matA st i col.val).val : 𝔽) + ((matB st i col.val).val : 𝔽))) (2 ^ st.c.kLog),
      delta_sum (fun i => GF128.of α * GF128.of (eIn lw ρ i)) (2 ^ st.c.kLog) col.val
        (fun i hi => by rw [hE0 i hi, mul_zero]),
      delta_sum (fun i => GF128.of (eIn lw ρ i)) (2 ^ st.c.kLog) col.val hE0,
      ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [e_model lw ρ _ h.hk6 hρ' i hi]
    unfold matA matB dCnt
    rw [cast_zmod, cast_zmod]
    push_cast
    ring
  folded := by
    intro α β lw ρ ts w hw
    obtain ⟨v, hv, hw'⟩ := folded_eq st α β lw ρ ts w hw
    exact ⟨v, hv, fold_size st α lw ρ v hv, hw'⟩

/-- The fold `Setup.ofCircuit` hands the verifier realizes `stmtOf`. -/
theorem ofCircuit_fold (st : Stmt) (h : StmtWF st) (regions : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    FoldRealizes (Setup.ofCircuit st).fold (stmtOf st h regions) :=
  stmtOf_fold st h regions

end FlockSoundness.Refine
