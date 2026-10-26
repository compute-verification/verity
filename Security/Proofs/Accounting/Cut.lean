import Proofs.Accounting.Gates
import Models.Accounting.Guarantees.Cut

/-!
# The min-cut bound, proved

`MinCut` is `EncodeChainHardness` applied to the cut circuit (`cut C D`). The cut circuit keeps every gate at its
position, so a gate in `D` reads the same operand indices, and by induction over the gates it takes `C`'s value on
every gate of `D` and its boundary (`cut_val`) and `C`'s derived interval there (`cut_intervals`). From those it is
restricted (`cut_proven`: `D`'s conditions are `C`'s, a boundary input and a constant ask nothing; `cut_operandBits`: a
boundary gate's derived interval lies in its declared one, so it has no more bits), its inputs and budgeted gates but
gathers hold their declared intervals wherever `C`'s do (`cut_declaredHold`: a boundary input's value lies in its
derived interval, `window_inv`), its inputs are the codec's (`cut_codecInputs`: a static boundary gate takes one value
on every draw, `static_val`), and it computes the same words on the same draws. Its run inputs' bits are the boundary's
dynamic bits and `D`'s run inputs' (`cut_runBits`). `Whole` and `Units` compare the cut circuit's units with `C`'s.
-/

namespace Accounting.SecurityProofs.Cut

open Accounting.Gates Accounting.Assumptions Accounting.SecurityProofs.Gates

variable {ι κ : Type}

/-! ## Renamed gates -/

lemma map_args (e : ι → κ) (g : Gate ι) : (g.map e).args = g.args := rfl

lemma map_bits (e : ι → κ) (g : Gate ι) : (g.map e).bits = g.bits := rfl

lemma map_isDerived (e : ι → κ) (op : Op ι) : (op.map e).IsDerived ↔ op.IsDerived := by
  cases op <;> simp [Op.map, Op.IsDerived]

lemma map_eq_affine {e : ι → κ} {op : Op ι} {c₀ : ℤ} {cs : List ℤ} (h : op.map e = .affine c₀ cs) :
    op = .affine c₀ cs := by
  cases op <;> simp_all [Op.map]

lemma map_eq_mma {e : ι → κ} {op : Op ι} {a b : Int8Fmt} {k : ℕ} {s : Bool} (h : op.map e = .mma a b k s) :
    op = .mma a b k s := by
  cases op <;> simp_all [Op.map]

lemma map_eq_budgeted {e : ι → κ} {op : Op ι} {cls : Budget} {f : List ℤ → ℤ} (h : op.map e = .budgeted cls f) :
    op = .budgeted cls f := by
  cases op <;> simp_all [Op.map]

lemma map_eq_input {e : ι → κ} {op : Op ι} {b : Binding} {s : κ} (h : op.map e = .input b s) :
    ∃ s₀, op = .input b s₀ ∧ s = e s₀ := by
  cases op <;> simp_all [Op.map]

lemma map_derive (e : ι → κ) (g : Gate ι) (ivs : List Iv) : (g.map e).derive ivs = g.derive ivs := by
  obtain ⟨op, args, lo, hi, step, prim, site, commit⟩ := g
  cases op <;> rfl

lemma map_inputBits (e : ι → κ) (g : Gate ι) (b : Binding) : (g.map e).inputBits b = g.inputBits b := by
  obtain ⟨op, args, lo, hi, step, prim, site, commit⟩ := g
  cases op <;> rfl

lemma map_raw (e : ι → κ) (g : Gate ι) (x : ι → ℤ) (x' : κ → ℤ) (hx : ∀ s, x' (e s) = x s) (vs vs' : List ℤ)
    (hv : ∀ j ∈ g.args, vs'.getD j 0 = vs.getD j 0) : (g.map e).raw x' vs' = g.raw x vs := by
  have ho : (g.map e).operands vs' = g.operands vs := List.map_congr_left hv
  unfold Gate.raw
  rw [ho]
  obtain ⟨op, args, lo, hi, step, prim, site, commit⟩ := g
  cases op <;> simp [Gate.map, Op.map, Op.raw, hx]

lemma map_value (e : ι → κ) (g : Gate ι) (x : ι → ℤ) (x' : κ → ℤ) (hx : ∀ s, x' (e s) = x s) (vs vs' : List ℤ)
    (hv : ∀ j ∈ g.args, vs'.getD j 0 = vs.getD j 0) : (g.map e).value x' vs' = g.value x vs := by
  unfold Gate.value
  rw [map_raw e g x x' hx vs vs' hv]
  obtain ⟨op, args, lo, hi, step, prim, site, commit⟩ := g
  cases op with
  | mma a b k s => cases s <;> rfl
  | _ => rfl

/-! ## Scans read before a position -/

lemma intervals_length (C : Dag ι) : (intervals C).length = C.length := by
  have := intervals_take_length C C.length le_rfl
  rwa [List.take_length] at this

lemma getD_values_take_eq (C : Dag ι) (x : ι → ℤ) (n j : ℕ) (hn : n ≤ C.length) :
    (values (C.take n) x).getD j 0 = if j < n then val C x j else 0 := by
  split_ifs with hj
  · rw [getD_values_take C x j (by omega) n hj hn, val_eq C x j (by omega)]
  · exact List.getD_eq_default _ _ (by rw [values_take_length C x n hn]; omega)

lemma getD_intervals_take_eq (C : Dag ι) (n j : ℕ) (hn : n ≤ C.length) :
    (intervals (C.take n)).getD j (0, 0) = if j < n then (intervals C).getD j (0, 0) else (0, 0) := by
  split_ifs with hj
  · exact getD_intervals_take C n j hj hn
  · exact List.getD_eq_default _ _ (by rw [intervals_take_length C n hn]; omega)

lemma val_of_le (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : C.length ≤ i) : val C x i = 0 :=
  List.getD_eq_default _ _ (by rw [values_length]; exact h)

/-! ## The cut circuit's gates -/

lemma cut_length (C : Dag ι) (D : Finset ℕ) : (cut C D).length = C.length := List.length_mapIdx

lemma cut_getElem (C : Dag ι) (D : Finset ℕ) (i : ℕ) (h : i < (cut C D).length) :
    (cut C D)[i] = cutGate C D i (C[i]'(by rwa [cut_length] at h)) := List.getElem_mapIdx

lemma boundary_not_mem {C : Dag ι} {D : Finset ℕ} {j : ℕ} (h : j ∈ boundary C D) : j ∉ D := by
  simp only [boundary, Finset.mem_filter] at h
  exact h.2.1

lemma boundary_lt {C : Dag ι} {D : Finset ℕ} {j : ℕ} (h : j ∈ boundary C D) : j < C.length := by
  simp only [boundary, Finset.mem_filter, Finset.mem_range] at h
  exact h.1

lemma mem_boundary {C : Dag ι} {D : Finset ℕ} {i j : ℕ} (hi : i ∈ D) (hlt : i < C.length) (hj : j ∈ C[i].args)
    (hjl : j < C.length) (hjD : j ∉ D) : j ∈ boundary C D := by
  simp only [boundary, Finset.mem_filter, Finset.mem_range]
  refine ⟨hjl, hjD, i, hi, ?_⟩
  rw [List.getElem?_eq_getElem hlt]
  simpa using hj

/-- An operand of a gate in `D` is in `D`, on its boundary, or past the circuit. -/
lemma arg_cases {C : Dag ι} {D : Finset ℕ} {i j : ℕ} (hi : i ∈ D) (hlt : i < C.length) (hj : j ∈ C[i].args)
    (hjl : j < C.length) : j ∈ D ∨ j ∈ boundary C D := by
  by_cases hjD : j ∈ D
  · exact Or.inl hjD
  · exact Or.inr (mem_boundary hi hlt hj hjl hjD)

/-! ## The cut circuit computes `C`'s values on `D` and its boundary -/

theorem cut_val (C : Dag ι) (D : Finset ℕ) (x : ι → ℤ) :
    ∀ i, (i ∈ D ∨ i ∈ boundary C D) → val (cut C D) (cutInput C x) i = val C x i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
  intro hi
  by_cases hlt : i < C.length
  swap
  · rw [val_of_le _ _ i (by rw [cut_length]; omega), val_of_le C x i (by omega)]
  have hlt' : i < (cut C D).length := by rw [cut_length]; exact hlt
  rw [val_eq (cut C D) _ i hlt', cut_getElem]
  rcases hi with hD | hb
  · rw [cutGate, ite_eq_left hD, val_eq C x i hlt]
    refine map_value Sum.inl C[i] x (cutInput C x) (fun _ => rfl) _ _ fun j hj => ?_
    rw [getD_values_take_eq (cut C D) _ i j (by rw [cut_length]; omega), getD_values_take_eq C x i j hlt.le]
    split_ifs with hji
    · exact ih j hji (arg_cases hD hlt hj (by omega))
    · rfl
  · rw [cutGate, ite_eq_right (boundary_not_mem hb), ite_eq_left hb]
    simp [Gate.value, Gate.raw, Op.raw, Op.out, cutInput]

lemma cut_args_values (C : Dag ι) (D : Finset ℕ) (x : ι → ℤ) {i : ℕ} (hD : i ∈ D) (hlt : i < C.length) :
    ∀ j ∈ C[i].args, (values ((cut C D).take i) (cutInput C x)).getD j 0 = (values (C.take i) x).getD j 0 := by
  intro j hj
  rw [getD_values_take_eq (cut C D) _ i j (by rw [cut_length]; omega), getD_values_take_eq C x i j hlt.le]
  split_ifs with hji
  · exact cut_val C D x j (arg_cases hD hlt hj (by omega))
  · rfl

theorem cut_intervals (C : Dag ι) (D : Finset ℕ) :
    ∀ i, (i ∈ D ∨ i ∈ boundary C D) → (intervals (cut C D)).getD i (0, 0) = (intervals C).getD i (0, 0) := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
  intro hi
  by_cases hlt : i < C.length
  swap
  · rw [List.getD_eq_default _ _ (by rw [intervals_length, cut_length]; omega),
      List.getD_eq_default _ _ (by rw [intervals_length]; omega)]
  have hlt' : i < (cut C D).length := by rw [cut_length]; exact hlt
  rw [getD_intervals_self (cut C D) i hlt', cut_getElem]
  rcases hi with hD | hb
  · rw [cutGate, ite_eq_left hD, map_derive, getD_intervals_self C i hlt]
    refine derive_congr _ _ _ fun j hj => ?_
    rw [getD_intervals_take_eq (cut C D) i j (by rw [cut_length]; omega), getD_intervals_take_eq C i j hlt.le]
    split_ifs with hji
    · exact ih j hji (arg_cases hD hlt hj (by omega))
    · rfl
  · rw [cutGate, ite_eq_right (boundary_not_mem hb), ite_eq_left hb]
    simp [Gate.derive]

lemma cut_args_intervals (C : Dag ι) (D : Finset ℕ) {i : ℕ} (hD : i ∈ D) (hlt : i < C.length) :
    ∀ j ∈ C[i].args, (intervals ((cut C D).take i)).getD j (0, 0) = (intervals (C.take i)).getD j (0, 0) := by
  intro j hj
  rw [getD_intervals_take_eq (cut C D) i j (by rw [cut_length]; omega), getD_intervals_take_eq C i j hlt.le]
  split_ifs with hji
  · exact cut_intervals C D j (arg_cases hD hlt hj (by omega))
  · rfl

/-! ## A static gate takes one value wherever the registered inputs do -/

theorem static_val (C : Dag ι) (x x' : ι → ℤ)
    (hreg : ∀ s, (∃ g ∈ C, g.op = .input .registered s) → x s = x' s) :
    ∀ i, dynAt C i = false → val C x i = val C x' i := by
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
  intro hd
  by_cases hlt : i < C.length
  swap
  · rw [val_of_le C x i (by omega), val_of_le C x' i (by omega)]
  rw [val_eq C x i hlt, val_eq C x' i hlt]
  by_cases hin : ∃ b s, C[i].op = .input b s
  · obtain ⟨b, s, hop⟩ := hin
    rw [dynAt_eq C i hlt] at hd
    simp only [dynStep, hop] at hd
    have hb : b = .registered := by cases b <;> simp_all
    subst hb
    have hx := hreg s ⟨C[i], List.getElem_mem hlt, hop⟩
    simp [Gate.value, Gate.raw, hop, Op.raw, Op.out, hx]
  · simp only [not_exists] at hin
    refine value_eq_of_operands C[i] x x' _ _ hin fun j hj => ?_
    rw [getD_values_take_eq C x i j hlt.le, getD_values_take_eq C x' i j hlt.le]
    split_ifs with hji
    · exact ih j hji (args_static C i hlt hin hd j hj hji)
    · rfl

/-! ## The cut circuit is restricted, holds its declarations and has the codec's inputs -/

lemma bits_le_of_within (a : Iv) (lo hi : ℤ) (h : a.Within lo hi) : a.bits ≤ Nat.clog 2 (hi + 1 - lo).toNat :=
  Nat.clog_mono_right 2 (Int.toNat_le_toNat (by obtain ⟨h1, h2⟩ := h; omega))

/-- A gate's derived interval has no more bits than its declared one. -/
lemma intervals_bits_le (C : Dag ι) (hP : Proven C) (j : ℕ) (hj : j < C.length) :
    ((intervals C).getD j (0, 0)).bits ≤ C[j].bits := by
  by_cases hd : C[j].op.IsDerived
  · rw [getD_intervals_self C j hj]
    exact bits_le_of_within _ _ _ ((hP j hj).1 hd)
  · rw [getD_intervals_decl C j hj hd]
    rfl

theorem cut_proven (C : Dag ι) (D : Finset ℕ) (hP : Proven C) : Proven (cut C D) := by
  intro i hi
  have hlt : i < C.length := by rwa [cut_length] at hi
  obtain ⟨h1, h2, h3⟩ := hP i hlt
  dsimp only
  rw [cut_getElem]
  by_cases hD : i ∈ D
  · rw [cutGate, ite_eq_left hD]
    have hiv := cut_args_intervals C D hD hlt
    refine ⟨fun hd => ?_, fun c₀ cs hc => h2 c₀ cs (map_eq_affine hc), fun fa fb k sat hm => ?_⟩
    · rw [map_derive, derive_congr _ _ _ hiv]
      exact h1 ((map_isDerived _ _).mp hd)
    · obtain ⟨ha, hlo, hhi, hA, hB⟩ := h3 fa fb k sat (map_eq_mma hm)
      refine ⟨ha, hlo, hhi, fun j hj => ?_, fun j hj => ?_⟩
      · rw [hiv j (List.mem_of_mem_tail (List.mem_of_mem_take hj))]
        exact hA j hj
      · rw [hiv j (List.mem_of_mem_tail (List.mem_of_mem_drop hj))]
        exact hB j hj
  · rw [cutGate, ite_eq_right hD]
    by_cases hb : i ∈ boundary C D
    · rw [ite_eq_left hb]
      refine ⟨fun h => ?_, fun _ _ h => ?_, fun _ _ _ _ h => ?_⟩
      · simp [Op.IsDerived] at h
      · simp at h
      · simp at h
    · rw [ite_eq_right hb]
      refine ⟨fun _ => ?_, fun _ _ h => ?_, fun _ _ _ _ h => ?_⟩
      · simp [Gate.derive, Iv.Within]
      · simp at h
      · simp at h

theorem cut_operandBits (C : Dag ι) (D : Finset ℕ) (B : ℕ) (hP : Proven C) (hB : OperandBits B C) :
    OperandBits B (cut C D) := by
  intro g hg cls f hop
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hg
  have hlt : i < C.length := by rwa [cut_length] at hi
  rw [cut_getElem] at hop ⊢
  by_cases hD : i ∈ D
  · rw [cutGate, ite_eq_left hD] at hop ⊢
    refine le_trans (List.sum_le_sum fun j hj => ?_) (hB C[i] (List.getElem_mem hlt) cls f (map_eq_budgeted hop))
    by_cases hj' : j < C.length
    · rw [List.getElem?_eq_getElem (by rw [cut_length]; exact hj'), List.getElem?_eq_getElem hj']
      simp only [Option.map_some, Option.getD_some]
      rw [cut_getElem]
      rcases arg_cases hD hlt hj hj' with hjD | hjb
      · rw [cutGate, ite_eq_left hjD]
        rfl
      · rw [cutGate, ite_eq_right (boundary_not_mem hjb), ite_eq_left hjb]
        exact intervals_bits_le C hP j hj'
    · rw [List.getElem?_eq_none (by rw [cut_length]; omega), List.getElem?_eq_none (by omega)]
      rfl
  · rw [cutGate, ite_eq_right hD] at hop
    by_cases hb : i ∈ boundary C D
    · rw [ite_eq_left hb] at hop
      simp at hop
    · rw [ite_eq_right hb] at hop
      simp at hop

theorem cut_declaredHold (C : Dag ι) (D : Finset ℕ) (x : ι → ℤ) (hP : Proven C) (hH : DeclaredHold C x) :
    DeclaredHold (cut C D) (cutInput C x) := by
  intro i hi hnd
  have hlt : i < C.length := by rwa [cut_length] at hi
  rw [cut_getElem] at hnd ⊢
  by_cases hD : i ∈ D
  · rw [cutGate, ite_eq_left hD] at hnd ⊢
    rw [map_raw Sum.inl C[i] x (cutInput C x) (fun _ => rfl) _ _ (cut_args_values C D x hD hlt)]
    exact hH i hlt fun h => hnd ((map_isDerived _ _).mpr h)
  · rw [cutGate, ite_eq_right hD] at hnd ⊢
    by_cases hb : i ∈ boundary C D
    · rw [ite_eq_left hb] at hnd ⊢
      have hw := window_inv C x hP hH C.length le_rfl i
      rw [List.take_length] at hw
      exact ⟨hw.1, hw.2⟩
    · rw [ite_eq_right hb] at hnd
      exact absurd trivial hnd

theorem cut_codecInputs (F : Field) {k : ℕ} (C : Dag ι) (D : Finset ℕ) (m : ℕ → ℕ) (x : Draw k → ι → ℤ)
    (hc : CodecInputs F C (draws F k) m x) :
    CodecInputs F (cut C D) (draws F k) m fun d => cutInput C (x d) := by
  obtain ⟨hreg, hcoin⟩ := hc
  refine ⟨fun s ⟨g, hg, hop⟩ d hd d' hd' => ?_, fun s ⟨g, hg, hop⟩ => ?_⟩
  · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hg
    have hlt : i < C.length := by rwa [cut_length] at hi
    rw [cut_getElem] at hop
    by_cases hD : i ∈ D
    · rw [cutGate, ite_eq_left hD] at hop
      obtain ⟨s₀, hs₀, rfl⟩ := map_eq_input hop
      exact hreg s₀ ⟨C[i], List.getElem_mem hlt, hs₀⟩ d hd d' hd'
    · rw [cutGate, ite_eq_right hD] at hop
      by_cases hb : i ∈ boundary C D
      · rw [ite_eq_left hb] at hop
        obtain ⟨hbind, rfl⟩ := Op.input.inj hop
        have hst : dynAt C i = false := by
          show (dynamic C).getD i false = false
          revert hbind
          cases (dynamic C).getD i false <;> simp
        exact static_val C (x d) (x d') (fun s₀ hs₀ => hreg s₀ hs₀ d hd d' hd') i hst
      · rw [ite_eq_right hb] at hop
        simp at hop
  · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hg
    have hlt : i < C.length := by rwa [cut_length] at hi
    rw [cut_getElem] at hop
    by_cases hD : i ∈ D
    · rw [cutGate, ite_eq_left hD] at hop
      obtain ⟨s₀, hs₀, rfl⟩ := map_eq_input hop
      exact hcoin s₀ ⟨C[i], List.getElem_mem hlt, hs₀⟩
    · rw [cutGate, ite_eq_right hD] at hop
      by_cases hb : i ∈ boundary C D
      · rw [ite_eq_left hb] at hop
        obtain ⟨hbind, -⟩ := Op.input.inj hop
        revert hbind
        cases (dynamic C).getD i false <;> simp
      · rw [ite_eq_right hb] at hop
        simp at hop

/-! ## The cut circuit's run-input bits -/

lemma sum_map_eq_range {β : Type} (L : List β) (f : β → ℕ) :
    (L.map f).sum = ∑ i ∈ Finset.range L.length, (L[i]?.map f).getD 0 := by
  induction L with
  | nil => simp
  | cons a L ih =>
    rw [List.map_cons, List.sum_cons, ih, List.length_cons, Finset.sum_range_succ']
    simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.map_some, Option.getD_some]
    omega

theorem cut_runBits (C : Dag ι) (D : Finset ℕ) : runBits (cut C D) = boundaryBits C D + runBitsIn C D := by
  unfold runBits
  rw [sum_map_eq_range, cut_length]
  have hpt : ∀ i ∈ Finset.range C.length, ((cut C D)[i]?.map (Gate.inputBits .run)).getD 0 =
      (if i ∈ boundary C D then
        (if (dynamic C).getD i false then ((intervals C).getD i (0, 0)).bits else 0) else 0) +
      (if i ∈ D then (C[i]?.map (Gate.inputBits .run)).getD 0 else 0) := by
    intro i hi
    have hlt := Finset.mem_range.mp hi
    rw [List.getElem?_eq_getElem (by rw [cut_length]; exact hlt), List.getElem?_eq_getElem hlt]
    simp only [Option.map_some, Option.getD_some]
    rw [cut_getElem]
    by_cases hD : i ∈ D
    · rw [cutGate, ite_eq_left hD, ite_eq_left hD, ite_eq_right fun h => boundary_not_mem h hD, zero_add, map_inputBits]
    · rw [cutGate, ite_eq_right hD, ite_eq_right hD, add_zero]
      by_cases hb : i ∈ boundary C D
      · rw [ite_eq_left hb, ite_eq_left hb]
        cases hdy : (dynamic C).getD i false <;> simp [Gate.inputBits, Gate.bits, Gate.size, Iv.bits]
      · rw [ite_eq_right hb, ite_eq_right hb]
        rfl
  have hbd : (Finset.range C.length).filter (· ∈ boundary C D) = boundary C D := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_range]
    exact ⟨fun h => h.2, fun h => ⟨boundary_lt h, h⟩⟩
  have h1 : (∑ i ∈ Finset.range C.length, if i ∈ boundary C D then
      (if (dynamic C).getD i false then ((intervals C).getD i (0, 0)).bits else 0) else 0) = boundaryBits C D := by
    rw [← Finset.sum_filter, hbd, boundaryBits]
  rw [Finset.sum_congr rfl hpt, Finset.sum_add_distrib, h1, runBitsIn, Finset.sum_filter]

/-! ## The min-cut bound -/

theorem MinCut : Accounting.Guarantees.Cut.MinCut := by
  intro F k B E εnum εden ι C D m x words out hE hε hk hp hp4 hR hc hnd hlt hW hout
  obtain ⟨hP, hB⟩ := hR
  have hR' : Restricted B (cut C D) := ⟨cut_proven C D hP, cut_operandBits C D B hP hB⟩
  have hW' : ComputesWords F k (cut C D) (fun d => cutInput C (x d)) m words out εnum εden := by
    obtain ⟨good, hgood, hmore, henc⟩ := hW
    refine ⟨good, hgood, hmore, fun d hd => ?_⟩
    obtain ⟨hdh, hval⟩ := henc d hd
    refine ⟨cut_declaredHold C D (x d) hP hdh, fun i hi l hl => ?_⟩
    rw [cut_val C D (x d) (out i l) (Or.inl (hout i hi l hl))]
    exact hval i hi l hl
  have := hE hε hk hp hp4 (cut C D) m (fun d => cutInput C (x d)) words out hR'
    (cut_codecInputs F C D m x hc) hnd hlt hW'
  rwa [cut_runBits] at this

end Accounting.SecurityProofs.Cut
