import Properties.NetworkScan.Flattens
import Flock.Sigma

/-!
# What Bits' check of a circuit file gives

`SampledProofs.Bits.check` lays a circuit file out as a template of a Program, or says why it doesn't. Where it finds
nothing wrong (`facts`), the file has one template instance a VU (`Facts.upv`) and one output leaf map an instance
(`Facts.leaves`), each of as many words as the return of the Definition `find?` finds at the template's key
(`Facts.ret`, `SampledProofs.Bits.retWords`), and every returned output bit sits on a computed row of the unit net
(`Facts.outRows`), as `SampledProofs.Bits.outGate` reads the rows (`Rows.ofNet`, whose split of the columns is the
check's: `ofNet_nIn`, `ofNet_nComp`).

A type's bits (`SampledProofs.Bits.bitsOf`) are its leaves' widths summed, `Flock.Sigma.bits` (`bits_bitsOf`): an
array's its length times its element's, a value's its width (`bits_array`, `bits_value`). A pair's return words are its
two fields' (`retWords_pair`).
-/

open SampledProofs.Bits

namespace Properties.NetworkScan.Check

/-! ## A type's bits -/

theorem foldl_add_eq : ∀ (l : List ℕ) (a : ℕ), l.foldl (· + ·) a = a + l.sum
  | [], a => by simp
  | x :: l, a => by rw [List.foldl_cons, foldl_add_eq l, List.sum_cons]; omega

theorem bits_bitsOf (t : Flock.Program.Ty) : bitsOf t = Flock.Sigma.bits t := by
  unfold bitsOf
  rw [← Array.foldl_toList, foldl_add_eq, Flock.Sigma.widths_sum, Nat.zero_add]

theorem bits_value (w : ℕ) : Flock.Sigma.bits (.value w) = w := by
  simp [Flock.Sigma.bits]

theorem bits_array (n : ℕ) (e : Flock.Program.Ty) : Flock.Sigma.bits (.array n e) = n * Flock.Sigma.bits e := by
  simp [Flock.Sigma.bits]

theorem retWords_pair (a b : Flock.Program.Ty) : retWords (.tuple #[a, b]) = wordsOf a + wordsOf b := rfl

/-! ## The check's facts -/

theorem ofNet_nIn (net : Flock.Net) : (Rows.ofNet net).nIn = min (net.inWords * Flock.WORD) net.constPos := rfl

theorem ofNet_nComp (net : Flock.Net) :
    (Rows.ofNet net).nComp = net.constPos - min (net.inWords * Flock.WORD) net.constPos := rfl

theorem orElse_eq_none {α : Type} {x : Option α} {y : Unit → Option α} (h : HOrElse.hOrElse x y = none) :
    x = none ∧ y () = none := by
  cases x with
  | none => exact ⟨rfl, h⟩
  | some a => cases h

theorem ite_eq_none {α : Type} {b : Bool} {x : α} (h : (if b = true then some x else none) = none) : b = false := by
  cases b
  · rfl
  · simp at h

theorem ite_none {α : Type} {p : Prop} {inst : Decidable p} {x : α} {r : Option α}
    (h : @ite _ p inst (some x) r = none) : ¬ p ∧ r = none := by
  split at h
  · cases h
  · rename_i hp
    exact ⟨hp, h⟩

/-- **What Bits' check gives the proofs** of the circuit file `c` laid out as template `tmpl` of the Program `P`: one
template instance a VU, one output leaf map an instance, each of the words of the return of the Definition `find?`
finds at `tmpl`, and every returned output bit on a computed row of the unit net. -/
structure Facts (c : Flock.Circuit) (P : Flock.Program.Prog) (tmpl : String) : Prop where
  upv : c.upv = 1
  leaves : c.leavesOut.size = c.upv
  ret : ∀ d, P.defs.find? (·.key == tmpl) = some d → ∀ x ∈ c.leavesOut, x.size = retWords d.ret
  outRows : ∀ k < c.upv, ∀ j < (c.leavesOut.getD k #[]).size, ∀ t < 16,
    (Rows.ofNet c.unit).nIn ≤ outCol c j t ∧ outCol c j t < (Rows.ofNet c.unit).nIn + (Rows.ofNet c.unit).nComp

/-- **Bits' check, passed, gives its facts.** -/
theorem facts {c : Flock.Circuit} {P : Flock.Program.Prog} {tmpl : String} (h : check c P tmpl = none) :
    Facts c P tmpl := by
  unfold check at h
  extract_lets nIn nComp gi go outs wired cut pc unwired at h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨hleaves, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  generalize unwired = uw at h
  rcases uw with _ | ⟨k, p⟩
  rotate_left
  · exact absurd h (Option.some_ne_none _)
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨-, h⟩ := ite_none h
  obtain ⟨hrows, h⟩ := ite_none h
  obtain ⟨hdef, hone⟩ := orElse_eq_none h
  refine ⟨bne_eq_false_iff_eq.1 (ite_eq_none hone), ?_, ?_, ?_⟩
  · apply Classical.byContradiction
    intro hne
    apply hleaves
    rw [Bool.or_eq_true, Bool.or_eq_true]
    exact Or.inl (Or.inr (bne_iff_ne.2 hne))
  · intro d hd x hx
    rw [hd] at hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨-, hdef⟩ := ite_none hdef
    obtain ⟨hsz, -⟩ := ite_none hdef
    exact Classical.byContradiction fun hne => hsz (Array.any_eq_true'.2 ⟨x, hx, bne_iff_ne.2 hne⟩)
  · intro k hk j hj t ht
    rw [ofNet_nIn, ofNet_nComp]
    have h3 := List.any_eq_false.1 (Bool.eq_false_iff.2 hrows) k (List.mem_range.2 hk)
    have h4 := List.any_eq_false.1 (Bool.eq_false_iff.2 h3) j (List.mem_range.2 hj)
    have h5 := List.any_eq_false.1 (Bool.eq_false_iff.2 h4) t (List.mem_range.2 ht)
    simpa only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq] using h5

end Properties.NetworkScan.Check
