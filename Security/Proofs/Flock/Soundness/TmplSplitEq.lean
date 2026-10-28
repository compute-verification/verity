import Proofs.Flock.Soundness.TmplSplitWalk
import Proofs.Flock.Soundness.TmplFoldLayout

/-!
# The split fold is the template fold

`foldSplit` walks the root's class at chunk origin 0 and sums the table's entries into the 64 slots; with the root
reading no inputs that sum is `tabExp`, which `walk_ok` values as the root copy's reference value
(`foldSplit_refVal`), `fold_refVal`'s too. So the two folds agree entry by entry (`foldSplit_eq_fold`), and with
`fold_eq_foldRef_wf` the split fold is the reference (`foldSplit_eq_foldRef_wf`). Beside the lane counts `fold` already
needs, the split fold needs the root's chunk, `2^log2ceil(size)`, to hold 64 slots inside the point.
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

theorem getD_modify_ne (xs : Array F128) (f : F128 → F128) {i j : ℕ} (h : i ≠ j) :
    (xs.modify i f).getD j 0 = xs.getD j 0 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_modify, if_neg h]

/-- `fold_val` under an invariant every step keeps. -/
theorem fold_val_inv {κ δ : Type} [BEq κ] [Hashable κ] (I : δ → Prop) (V : δ → GF128) (step : δ → κ → F128 → δ)
    (G : κ → F128 → GF128) (hI : ∀ d k v, I d → I (step d k v)) (h : ∀ d k v, I d → V (step d k v) = V d + G k v)
    (m : HashMap κ F128) (init : δ) (hi : I init) :
    I (m.fold step init) ∧ V (m.fold step init) = V init + (m.toList.map fun e => G e.1 e.2).sum := by
  rw [HashMap.fold_eq_foldl_toList]
  generalize m.toList = l
  induction l generalizing init with
  | nil => simp [hi]
  | cons e l ih =>
    rw [List.foldl_cons]
    obtain ⟨h1, h2⟩ := ih _ (hI _ e.1 e.2 hi)
    exact ⟨h1, by rw [h2, h _ _ _ hi, List.map_cons, List.sum_cons, add_assoc]⟩

theorem hsum_of_not_mem {κ : Type} [BEq κ] [LawfulBEq κ] [Hashable κ] [LawfulHashable κ] [DecidableEq κ]
    (m : HashMap κ F128) (f : κ → GF128) (h : ∀ x, ¬ x ∈ m) : hsum m f = 0 := by
  rw [hsum_superset m f ∅ fun x hx => absurd hx (h x), Finset.sum_empty]

/-- **The 64 slots `foldSplit` sums** from a table with no open inputs, at chunk origin 0: its `tabExp`, plus the
constant's term. -/
theorem split_val {pt : FoldPt} {L s : ℕ} (hs : s < 64) (o : Tab) (iv : ℕ → ℕ → GF128) (hU : ∀ x, ¬ x ∈ o.U)
    {stepP : Array F128 → ℕ × ℕ × ℕ → F128 → Array F128} {stepC : Array F128 → ℕ × ℕ → F128 → Array F128}
    (hsP : ∀ p cx cy s' v, stepP p (cx, cy, s') v = p.modify s' (· + v * eqTail pt.rhoT L cx * eqTail pt.rT L cy))
    (hsC : ∀ p cx s' v, stepC p (cx, s') v = p.modify s' (· + v * eqTail pt.rhoT L cx)) (q : ℕ) (c : F128) :
    of (addAt (o.PC.fold stepC (o.P.fold stepP (Array.replicate 64 0))) q c)[s]! =
      tabExp pt s L 0 iv o + if q % 64 = s then of c else 0 := by
  obtain ⟨hz1, hv1⟩ := fold_val_inv (fun p : Array F128 => p.size = 64) (fun p => of (p.getD s 0)) stepP
    (fun k v => of v * if k.2.2 = s then hiE pt L 0 k.1 * hiG pt L 0 k.2.1 else 0)
    (fun d k v hd => by
      obtain ⟨cx, cy, s'⟩ := k
      simp only [hsP, Array.size_modify, hd])
    (fun d k v hd => by
      obtain ⟨cx, cy, s'⟩ := k
      simp only [hsP]
      by_cases h : s' = s
      · rw [h, getD_modify d _ (show s < d.size by rw [hd]; exact hs), if_pos rfl, if_pos rfl]
        simp only [hiE, hiG, Nat.zero_add, of_add, of_mul]
        ring
      · rw [getD_modify_ne _ _ h, if_neg h, mul_zero, add_zero])
    o.P (Array.replicate 64 0) (by simp)
  obtain ⟨hz2, hv2⟩ := fold_val_inv (fun p : Array F128 => p.size = 64) (fun p => of (p.getD s 0)) stepC
    (fun k v => of v * if k.2 = s then hiE pt L 0 k.1 else 0)
    (fun d k v hd => by
      obtain ⟨cx, s'⟩ := k
      simp only [hsC, Array.size_modify, hd])
    (fun d k v hd => by
      obtain ⟨cx, s'⟩ := k
      simp only [hsC]
      by_cases h : s' = s
      · rw [h, getD_modify d _ (show s < d.size by rw [hd]; exact hs), if_pos rfl, if_pos rfl]
        simp only [hiE, Nat.zero_add, of_add, of_mul]
      · rw [getD_modify_ne _ _ h, if_neg h, mul_zero, add_zero])
    o.PC _ hz1
  rw [getElem!_getD _ (by rw [addAt_size, hz2]; exact hs), addAt_get _ _ _ (by rw [hz2]; exact hs), hv2, hv1]
  unfold tabExp
  rw [hsum_of_not_mem o.U _ hU, add_zero]
  unfold hsum
  rw [getD_replicate, ← GF128.zero_eq, zero_add]

theorem stOk_empty (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s : ℕ) : StOk t lays pt s {} :=
  ⟨fun _ _ h => by simp [HashMap.get?_eq_getElem?] at h, fun _ _ _ _ h => by simp [HashMap.get?_eq_getElem?] at h⟩

/-- **The split fold** at entry `s < 64` is `fold_refVal`'s value: the root copy's reference value plus the constant's
row, while the root's chunk holds 64 slots inside the point and the root reads no inputs. -/
theorem foldSplit_refVal {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P : Array F128} {w1 w2 w3 : ℕ}
    (hP : foldSplit t lays alpha lw rho r = .ok (P, w1, w2, w3)) (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    (hroot : t.root < t.defs.size) (hins : (t.defs.getD t.root default).ins = 0)
    (hal : 6 ≤ log2ceil (lays.getD t.root default).size)
    (hr1 : log2ceil (lays.getD t.root default).size - 6 ≤ rho.size)
    (hr2 : log2ceil (lays.getD t.root default).size - 6 ≤ r.size) {s : ℕ} (hs : s < 64) :
    of P[s]! =
      refVal t lays { alpha, lw, rho, r, qc := posOf t lays ((t.defs.getD t.root default).cols - 1) } s (t.defs.size + 1)
          t.root 0 1 [] +
        if posOf t lays ((t.defs.getD t.root default).cols - 1) % 64 = s then
          of ((alpha + 1) * FoldPt.e { alpha, lw, rho, r, qc := posOf t lays ((t.defs.getD t.root default).cols - 1) }
              (posOf t lays ((t.defs.getD t.root default).cols - 1)) *
            FoldPt.g { alpha, lw, rho, r, qc := posOf t lays ((t.defs.getD t.root default).cols - 1) }
              (posOf t lays ((t.defs.getD t.root default).cols - 1)))
        else 0 := by
  unfold foldSplit at hP
  obtain ⟨⟨o, st⟩, ho, hP⟩ := except_bind_ok hP
  cases pure_ok hP
  obtain ⟨-, hU, hv, -⟩ := walk_ok hs rfl rfl hlay hcall _ _ _ _ _ _ _ _ _ ho (stOk_empty _ _ _ _) (by omega)
    ⟨by omega, by simpa using hr1, by simpa using hr2, fun _ k hk => by
      obtain rfl : k = 0 := by simpa using hk
      simp⟩
  have h1 := hv (by omega) 0 fun _ => []
  rw [show (#[0] : Array ℕ).size = 1 from rfl, Finset.sum_range_one] at h1
  simp only [show (#[0] : Array ℕ).getD 0 0 = 0 from rfl, pow_zero, zero_mul, add_zero, mul_zero, Nat.zero_add] at h1
  rw [← h1]
  exact split_val hs o _ (fun x hx => absurd (hU x hx).1 (by rw [hins]; exact Nat.not_lt_zero _))
    (by intros; rfl) (by intros; rfl) _ _

/-- **The split fold is the template fold**, entry by entry. -/
theorem foldSplit_eq_fold {t : Tmpl} {lays : Array Lay} {alpha : F128} {lw rho r P P' : Array F128} {w w1 w2 w3 : ℕ}
    (hS : foldSplit t lays alpha lw rho r = .ok (P, w1, w2, w3)) (hP : fold t lays alpha lw rho r = .ok (P', w))
    (hlay : LayOk t lays)
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d)
    (hroot : t.root < t.defs.size) (hins : (t.defs.getD t.root default).ins = 0)
    (hf1 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + rho.size) (hf2 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + r.size)
    (hal : 6 ≤ log2ceil (lays.getD t.root default).size)
    (hr1 : log2ceil (lays.getD t.root default).size - 6 ≤ rho.size)
    (hr2 : log2ceil (lays.getD t.root default).size - 6 ≤ r.size) {s : ℕ} (hs : s < 64) :
    of P[s]! = of P'[s]! := by
  rw [foldSplit_refVal hS hlay hcall hroot hins hal hr1 hr2 hs, fold_refVal hP hlay hins hf1 hf2 hs]

/-- **The split fold is the reference** on `layoutM`'s layouts of a well-formed template. -/
theorem foldSplit_eq_foldRef_wf {t : Tmpl} {mode : Mode} {lays : Array Lay} {alpha : F128}
    {lw rho r P P' P'' : Array F128} {w w1 w2 w3 : ℕ} (hwf : t.wellFormed = true) (hL : layoutM t mode = .ok lays)
    (hS : foldSplit t lays alpha lw rho r = .ok (P, w1, w2, w3)) (hP : fold t lays alpha lw rho r = .ok (P', w))
    (hR : foldRef t lays alpha lw rho r = .ok P'')
    (hf1 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + rho.size) (hf2 : lanes t lays (t.defs.size + 1) t.root ≤ 6 + r.size)
    (hal : 6 ≤ log2ceil (lays.getD t.root default).size)
    (hr1 : log2ceil (lays.getD t.root default).size - 6 ≤ rho.size)
    (hr2 : log2ceil (lays.getD t.root default).size - 6 ≤ r.size) {s : ℕ} (hs : s < 64) :
    of P[s]! = of P''[s]! := by
  have hlay : LayOk t lays :=
    ⟨fun d i j hi => wf_bound hwf d i j hi, layouts_lanes (layoutM_layouts hL) fun d hd i hi => (wf_call hwf d hd i hi).2⟩
  have hcall : ∀ d i, i < (t.defs.getD d default).calls.size →
      ((t.defs.getD d default).calls.getD i default).def_ < d := fun d i hi => by
    by_cases hd : d < t.defs.size
    · exact (wf_call hwf d hd i hi).1
    · rw [getD_out hd] at hi
      first | exact absurd hi (Nat.not_lt_zero _) | simp at hi
  rw [foldSplit_eq_fold hS hP hlay hcall (wf_tmpl hwf).1 (wf_tmpl hwf).2.1 hf1 hf2 hal hr1 hr2 hs,
    fold_eq_foldRef_wf hwf hL hP hR hf1 hf2 hs]

end FlockSoundness.TmplFold
