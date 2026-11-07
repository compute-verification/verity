module

public import Proofs.Flock.Bridge.FilePart.Block

/-!
# `uniform`'s proof blocks, and who reads a source

`blockOf_uniform`: `Block`'s blocks at `Placed.uniform`. `reads_uniform`: on `Placed.insOf`'s source-indexed input
gates, unit `v` reads gate `2 + v·B + b` exactly when some row's input column has source `b`. `stack_reads`: an input of
stacked copies is read exactly when its template input is.
-/

@[expose] public section

namespace Flock.Bridge.FilePart

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge.Placed

/-! ## `uniform`'s circuit and partition -/

section uniform

variable {k : ℕ} {R : Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one} {n : ℕ}
  (outs : Finset (Fin (gates k R n)))

theorem size_uniform (g : Fin ((uniform k R ins one hone n).circuit outs).N) :
    gateSize (((uniform k R ins one hone n).circuit outs).op g) = if g.val < k then 1 else 3 := by
  by_cases h : g.val < k
  · rw [if_pos h]
    show gateSize ((uniform k R ins one hone n).op g) = 1
    rw [op_uniform_input n g h]
    rfl
  · rw [if_neg h]
    obtain ⟨i, a, b, -, ho, -⟩ := op_uniform_row (ins := ins) (one := one) (hone := hone) n g (by omega)
    show gateSize ((uniform k R ins one hone n).op g) = 3
    rw [ho]
    rfl

theorem isInput_uniform (g : Fin ((uniform k R ins one hone n).circuit outs).N) :
    (((uniform k R ins one hone n).circuit outs).op g).isInput = true ↔ g.val < k := by
  constructor
  · intro hi
    by_contra h
    obtain ⟨i, a, b, -, ho, -⟩ := op_uniform_row (ins := ins) (one := one) (hone := hone) n g (by omega)
    change ((uniform k R ins one hone n).op g).isInput = true at hi
    rw [ho] at hi
    cases hi
  · intro h
    show ((uniform k R ins one hone n).op g).isInput = true
    rw [op_uniform_input n g h]
    rfl

/-- **`uniform`'s proof blocks**: unit `u`'s rows' `3·nComp` lowered gates from `k + 3·nComp·u`, then the gates
`2 + u·B + b` of the sources `b ∈ S` it owns, where `k = 2 + n·B` and unit `v` reads input gate `2 + v·B + b` exactly
when `b` is a source some row reads (`Rd`). -/
theorem blockOf_uniform (κ : Fin ((uniform k R ins one hone n).circuit outs).N → Option Bool) {B : ℕ}
    (hk : k = 2 + n * B) (hκ : ∀ g : Fin ((uniform k R ins one hone n).circuit outs).N, g.val < k →
      (κ g = none ↔ 2 ≤ g.val))
    (Rd : ℕ → Prop)
    (hreads : ∀ (g : Fin ((uniform k R ins one hone n).circuit outs).N) (v : Fin n), 2 ≤ g.val → g.val < k →
      ((∃ h, ((uniform k R ins one hone n).partition outs).unit h = some v ∧
        g ∈ (((uniform k R ins one hone n).circuit outs).op h).args) ↔ (g.val - 2) / B = v.val ∧ Rd ((g.val - 2) % B)))
    (S : List ℕ) (hS : S.Pairwise (· < ·)) (hSm : ∀ b, b ∈ S ↔ b < B ∧ Rd b) (u : Fin n) :
    blockOf ((uniform k R ins one hone n).partition outs) κ u =
      (List.range (3 * R.nComp)).map (k + 3 * R.nComp * u.val + ·) ++ S.map (2 + u.val * B + ·) := by
  unfold blockOf
  rw [unitGates_of ((uniform k R ins one hone n).partition outs) (gates_eq k R n) (size_uniform outs)
      (unit_uniform_val n) u,
    insOf_of ((uniform k R ins one hone n).partition outs) κ hk (le_gates k R n) (size_uniform outs)
      (isInput_uniform outs) hκ Rd hreads S hS hSm u]

end uniform

/-! ## Who reads a source gate -/

section reads

variable {R : Rows} {n B : ℕ} {src : Fin R.nIn → Option (Fin B)}

theorem colVal_insOf {v : ℕ} (hv : v < n) (c : Fin R.w) {g : ℕ} (hg2 : 2 ≤ g) (hgk : g < inGates n B) :
    colVal (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src) (oneGate n B) v c = g ↔
      ∃ h : c.val < R.nIn, ∃ b : Fin B, src ⟨c.val, h⟩ = some b ∧ g = 2 + v * B + b.val := by
  unfold colVal
  by_cases h : c.val < R.nIn
  · simp only [h, ↓reduceDIte, FlockSoundness.Discharge.Placed.insOf, hv, exists_true_left]
    cases hs : src ⟨c.val, h⟩ with
    | none => simp [srcGate, zeroGate]; omega
    | some b =>
      simp only [srcGate, Option.some.injEq]
      constructor
      · intro he; exact ⟨b, rfl, he.symm⟩
      · rintro ⟨b', rfl, he⟩; exact he.symm
  · rw [dif_neg h]
    constructor
    · intro he
      exfalso
      split at he
      · show False
        have : inGates n B ≤ g := by rw [← he]; omega
        omega
      · simp [oneGate] at he; omega
    · rintro ⟨h', -⟩; exact absurd h' h

theorem reads_uniform (outs : Finset (Fin (gates (inGates n B) R n)))
    (g : Fin ((uniform (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src) (oneGate n B)
      (insOf_ne_one n B src) n).circuit outs).N) (v : Fin n) (hg2 : 2 ≤ g.val) (hgk : g.val < inGates n B) :
    (∃ h, ((uniform (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src) (oneGate n B)
        (insOf_ne_one n B src) n).partition outs).unit h = some v ∧
      g ∈ (((uniform (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src) (oneGate n B)
        (insOf_ne_one n B src) n).circuit outs).op h).args) ↔
      (g.val - 2) / B = v.val ∧ ∃ i : Fin R.nComp, ∃ c ∈ R.a i ++ R.b i, ∃ hc : c.val < R.nIn,
        (src ⟨c.val, hc⟩).map Fin.val = some ((g.val - 2) % B) := by
  have hG := gates_eq (inGates n B) R n
  constructor
  · rintro ⟨h, hu, hg⟩
    have hv := unit_uniform_val (ins := FlockSoundness.Discharge.Placed.insOf n B src) (one := oneGate n B)
      (hone := insOf_ne_one n B src) n h
    change (uniform _ R _ _ _ n).unit h = some v at hu
    rw [hu, Option.map_some] at hv
    split at hv
    · cases hv
    · rename_i hk
      have hv' := (Option.some.inj hv)
      obtain ⟨i, a, b, -, ho, ha, hb⟩ := op_uniform_row (ins := FlockSoundness.Discharge.Placed.insOf n B src)
        (one := oneGate n B) (hone := insOf_ne_one n B src) n h (by omega)
      change g ∈ ((uniform _ R _ _ _ n).op h).args at hg
      rw [ho] at hg
      have hm : g.val ∈ (R.a i ++ R.b i).map (colVal (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src)
          (oneGate n B) ((h.val - inGates n B) / R.nComp)) := by
        rw [List.map_append, ← ha, ← hb, ← List.map_append]
        exact List.mem_map_of_mem hg
      obtain ⟨c, hc, hcv⟩ := List.mem_map.1 hm
      rw [← hv'] at hcv
      obtain ⟨hcn, b', hs, hgb⟩ := (colVal_insOf v.isLt c hg2 hgk).1 hcv
      have hb' := b'.isLt
      refine ⟨?_, i, c, hc, hcn, ?_⟩
      · rw [hgb, show 2 + v.val * B + b'.val - 2 = b'.val + v.val * B by omega,
          Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hb', Nat.zero_add]
      · rw [hs, hgb, show 2 + v.val * B + b'.val - 2 = b'.val + v.val * B by omega, Nat.add_mul_mod_self_right,
          Nat.mod_eq_of_lt hb']
        rfl
  · rintro ⟨hdiv, i, c, hc, hcn, hs⟩
    obtain ⟨b', hs', hb'⟩ := Option.map_eq_some_iff.1 hs
    have hB : 0 < B := Nat.pos_of_ne_zero fun h0 => by
      have := (b'.isLt); omega
    have hgv : g.val = 2 + v.val * B + b'.val := by
      have := Nat.div_add_mod' (g.val - 2) B
      rw [hdiv, ← hb'] at this
      omega
    have hcv := (colVal_insOf (src := src) v.isLt c hg2 hgk).2 ⟨hcn, b', hs', hgv⟩
    have hi := i.isLt
    have hv := v.isLt
    have hvn : v.val * R.nComp + R.nComp ≤ n * R.nComp := by
      rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hv
    let h : Fin ((uniform (inGates n B) R (FlockSoundness.Discharge.Placed.insOf n B src) (oneGate n B)
      (insOf_ne_one n B src) n).circuit outs).N := ⟨inGates n B + v.val * R.nComp + i.val, by
        show _ < gates (inGates n B) R n; omega⟩
    have hdivh : (h.val - inGates n B) / R.nComp = v.val := by
      show (inGates n B + v.val * R.nComp + i.val - inGates n B) / R.nComp = v.val
      rw [show inGates n B + v.val * R.nComp + i.val - inGates n B = i.val + v.val * R.nComp by omega,
        Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, Nat.zero_add]
    have hmodh : (h.val - inGates n B) % R.nComp = i.val := by
      show (inGates n B + v.val * R.nComp + i.val - inGates n B) % R.nComp = i.val
      rw [show inGates n B + v.val * R.nComp + i.val - inGates n B = i.val + v.val * R.nComp by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    have hkh : ¬ h.val < inGates n B := by show ¬ inGates n B + v.val * R.nComp + i.val < inGates n B; omega
    refine ⟨h, ?_, ?_⟩
    · have hu := unit_uniform_val (ins := FlockSoundness.Discharge.Placed.insOf n B src) (one := oneGate n B)
        (hone := insOf_ne_one n B src) n h
      rw [if_neg hkh, hdivh] at hu
      obtain ⟨w, hw, hwv⟩ := Option.map_eq_some_iff.1 hu
      change (uniform _ R _ _ _ n).unit h = some v
      rw [hw, show w = v from Fin.ext hwv]
    · obtain ⟨i', a, b, hi', ho, ha, hb⟩ := op_uniform_row (ins := FlockSoundness.Discharge.Placed.insOf n B src)
        (one := oneGate n B) (hone := insOf_ne_one n B src) n h (by omega)
      obtain rfl : i' = i := Fin.ext (by rw [← hi', hmodh])
      change g ∈ ((uniform _ R _ _ _ n).op h).args
      rw [ho]
      have hm : g.val ∈ (a ++ b).map Fin.val := by
        rw [List.map_append, ha, hb, ← List.map_append, hdivh]
        exact List.mem_map.2 ⟨c, hc, hcv⟩
      obtain ⟨g', hg', hgg⟩ := List.mem_map.1 hm
      exact (Fin.ext hgg : g' = g) ▸ hg'

end reads

/-! ## Which stacked inputs a row reads -/

theorem stackCol_val_lt (T : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (c : Fin T.w) (hc : c.val < T.nIn) :
    (T.stackCol k hu c).val = u * T.nIn + c.val := by
  simp [Rows.stackCol, hc]

theorem stackCol_val_ge (T : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (c : Fin T.w) (hc : ¬ c.val < T.nIn) :
    k * T.nIn ≤ (T.stackCol k hu c).val := by
  unfold Rows.stackCol
  rw [dif_neg hc]
  split <;> simp

/-- **A stacked input is read when its template input is**: input `x` of `k` stacked copies is copy `x / nIn`'s input
`x % nIn`, which some stacked row reads exactly when some template row reads it. -/
theorem stack_reads (T : Rows) (k x : ℕ) (hx : x < k * T.nIn) :
    (∃ i : Fin (T.stack k).nComp, ∃ c ∈ (T.stack k).a i ++ (T.stack k).b i, c.val = x) ↔
      ∃ j : Fin T.nComp, ∃ c ∈ T.a j ++ T.b j, c.val = x % T.nIn := by
  have hn : 0 < T.nIn := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega
  constructor
  · rintro ⟨i, c, hc, rfl⟩
    change c ∈ T.stackRow k T.a i ++ T.stackRow k T.b i at hc
    unfold Rows.stackRow at hc
    by_cases hj : i.val % (T.nComp + 1) = 0
    · rw [dif_pos hj, dif_pos hj] at hc
      have hc' : c.val = k * T.nIn + k * (T.nComp + 1) := by
        rcases List.mem_append.1 hc with h | h <;> rw [List.mem_singleton.1 h]
      omega
    · have hmod := Nat.mod_lt i.val (show 0 < T.nComp + 1 by omega)
      rw [dif_neg hj, dif_neg hj, ← List.map_append] at hc
      obtain ⟨c₀, hc₀, rfl⟩ := List.mem_map.1 hc
      by_cases h0 : c₀.val < T.nIn
      · refine ⟨_, c₀, hc₀, ?_⟩
        rw [stackCol_val_lt T k _ c₀ h0]
        have h1 : (i.val / (T.nComp + 1) * T.nIn + c₀.val) % T.nIn = c₀.val := by
          rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h0]
        exact h1.symm
      · have := stackCol_val_ge T k (T.stack_copy_lt k i.isLt) c₀ h0
        omega
  · rintro ⟨j, c₀, hc₀, hcv⟩
    have hu : x / T.nIn < k := (Nat.div_lt_iff_lt_mul hn).2 hx
    have hj := j.isLt
    have hc0 : c₀.val < T.nIn := by rw [hcv]; exact Nat.mod_lt _ hn
    have hik : (x / T.nIn) * (T.nComp + 1) + (j.val + 1) < k * (T.nComp + 1) :=
      Rows.mul_add_lt hu (by omega)
    let i : Fin (T.stack k).nComp := ⟨(x / T.nIn) * (T.nComp + 1) + (j.val + 1), hik⟩
    have himod : i.val % (T.nComp + 1) = j.val + 1 := by
      show ((x / T.nIn) * (T.nComp + 1) + (j.val + 1)) % (T.nComp + 1) = j.val + 1
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    have hidiv : i.val / (T.nComp + 1) = x / T.nIn := by
      show ((x / T.nIn) * (T.nComp + 1) + (j.val + 1)) / (T.nComp + 1) = x / T.nIn
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
    refine ⟨i, T.stackCol k (T.stack_copy_lt k i.isLt) c₀, ?_, ?_⟩
    · change _ ∈ T.stackRow k T.a i ++ T.stackRow k T.b i
      unfold Rows.stackRow
      rw [dif_neg (by omega), dif_neg (by omega), ← List.map_append]
      have hjj : (⟨i.val % (T.nComp + 1) - 1, by omega⟩ : Fin T.nComp) = j := Fin.ext (by simp only; omega)
      rw [hjj]
      exact List.mem_map_of_mem hc₀
    · rw [stackCol_val_lt T k _ c₀ hc0, hidiv, hcv]
      exact Nat.div_add_mod' x T.nIn

end Flock.Bridge.FilePart
