module

public import Properties.Spot.Bits

/-! # Bits' layout as the claims' program (`Properties.Spot.Bits`)

`bitsGate` and `ofBGate` are inverse and keep the gates read, constructor by constructor. Bits lowers its gates in
order, gate `g` to `gateSize kp g` gates (`lower_length`) after the `off kp g` lowered before it (`off_of_le`), so of
`bitsProg kp κ`'s `off kp kp.size` gates (`bitsProg_length`), the one at `posOf kp g` is the last of gate `g`'s
(`bitsProg_gate`).
-/

@[expose] public section

open _root_.Spot

namespace Properties.Spot.SecurityProofs

theorem ofBGate_bitsGate (g : Bits.LGate) : ofBGate (bitsGate g) = g := by
  cases g <;> rfl

theorem bitsGate_ofBGate (g : Claims.Sampling.BGate) : bitsGate (ofBGate g) = g := by
  cases g <;> rfl

theorem bitsGate_injective : Function.Injective bitsGate :=
  Function.LeftInverse.injective ofBGate_bitsGate

/-- An `LGate` and its claims' gate read the same gates. -/
theorem bitsGate_reads (g : Bits.LGate) : (bitsGate g).reads = g.reads := by
  cases g <;> rfl

theorem bitsProg_outs (kp : Bits.KeyProg) (κ : ℕ → Option Bool) :
    (bitsProg kp κ).outs = kp.outs.map (Bits.posOf kp) := rfl

private theorem sum_range_succ (h : ℕ → ℕ) (m : ℕ) :
    ((List.range (m + 1)).map h).sum = ((List.range m).map h).sum + h m := by
  simp [List.range_succ]

private theorem sum_range_mono (h : ℕ → ℕ) {a b : ℕ} (hab : a ≤ b) :
    ((List.range a).map h).sum ≤ ((List.range b).map h).sum := by
  induction b with
  | zero =>
    obtain rfl : a = 0 := by omega
    exact Nat.le_refl _
  | succ b ih =>
    rcases Nat.lt_or_ge a (b + 1) with h' | h'
    · have := ih (by omega)
      rw [sum_range_succ]
      omega
    · obtain rfl : a = b + 1 := by omega
      exact Nat.le_refl _

/-- **Of the first `n` lists flattened in order, list `g < n`'s element `i`** sits after the lists before it. -/
private theorem getElem?_flatMap_range {α : Type} (f : ℕ → List α) {n g i : ℕ} (hg : g < n)
    (hi : i < (f g).length) :
    ((List.range n).flatMap f)[((List.range g).map fun x => (f x).length).sum + i]? = (f g)[i]? := by
  induction n with
  | zero => omega
  | succ m ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rcases Nat.lt_or_ge g m with h | h
    · have hle := sum_range_mono (fun x => (f x).length) (show g + 1 ≤ m by omega)
      rw [sum_range_succ] at hle
      rw [List.getElem?_append_left (by rw [List.length_flatMap]; omega), ih h]
    · obtain rfl : g = m := by omega
      rw [List.getElem?_append_right (by rw [List.length_flatMap]; omega), List.length_flatMap,
        Nat.add_sub_cancel_left]

private theorem filter_lt_range {m g : ℕ} (hg : g ≤ m) :
    (List.range m).filter (fun x => decide (x < g)) = List.range g := by
  induction m with
  | zero =>
    obtain rfl : g = 0 := by omega
    rfl
  | succ m ih =>
    rcases Nat.lt_or_ge g (m + 1) with h | h
    · rw [List.range_succ, List.filter_append, ih (by omega)]
      simp [show ¬ m < g by omega]
    · obtain rfl : g = m + 1 := by omega
      exact List.filter_eq_self.2 fun x hx => by simpa using List.mem_range.1 hx

/-- Gate `g` lowers to `gateSize kp g` gates. -/
theorem lower_length (kp : Bits.KeyProg) (κ : ℕ → Option Bool) (g : ℕ) :
    (Bits.lower kp κ g).length = Bits.gateSize kp g := by
  unfold Bits.lower Bits.gateSize
  cases kp.row? g with
  | some r => rfl
  | none => cases κ g <;> rfl

/-- `off kp g` at a gate of `kp` is the lowered length of the gates before it. -/
theorem off_of_le (kp : Bits.KeyProg) {g : ℕ} (hg : g ≤ kp.size) :
    Bits.off kp g = ((List.range g).map (Bits.gateSize kp)).sum := by
  unfold Bits.off
  rw [filter_lt_range hg]

/-- **`bitsProg`'s gates** are `off kp kp.size`. -/
theorem bitsProg_length (kp : Bits.KeyProg) (κ : ℕ → Option Bool) :
    (bitsProg kp κ).gates.length = Bits.off kp kp.size := by
  simp only [bitsProg, Bits.prog, List.length_map, List.length_flatMap, lower_length, off_of_le kp (Nat.le_refl _)]

/-- **Gate `g` sits at its position**: of `bitsProg kp κ`'s gates, the one at `Bits.posOf kp g` is the last of gate
`g`'s lowered gates, under `bitsGate`. -/
theorem bitsProg_gate (kp : Bits.KeyProg) (κ : ℕ → Option Bool) {g : ℕ} (hg : g < kp.size) :
    (bitsProg kp κ).gates[Bits.posOf kp g]? = ((Bits.lower kp κ g)[Bits.gateSize kp g - 1]?).map bitsGate := by
  have hpos : 0 < Bits.gateSize kp g := by
    unfold Bits.gateSize
    split <;> omega
  have hoff : Bits.off kp g = ((List.range g).map fun x => (Bits.lower kp κ x).length).sum := by
    rw [off_of_le kp (Nat.le_of_lt hg)]
    simp only [lower_length]
  simp only [bitsProg, Bits.prog, Bits.posOf, List.getElem?_map]
  rw [show Bits.off kp g + Bits.gateSize kp g - 1 = Bits.off kp g + (Bits.gateSize kp g - 1) by omega, hoff,
    getElem?_flatMap_range _ hg (by rw [lower_length]; omega)]

end Properties.Spot.SecurityProofs
