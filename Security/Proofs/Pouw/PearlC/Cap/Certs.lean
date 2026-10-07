import Mathlib.Tactic.Linarith
import Proofs.Pouw.PearlC.FreezeP
import Proofs.Pouw.PearlC.Cap.Grid

/-!
# Freeze's certificates on the device's step, and the cap circuit's interval test

Piece 1 of the cap circuit's soundness (`capTile_sound`): its debit (A) flags are `SkipP.jointFlagsP` for **any**
certificates that pass its checks, so no uniqueness of the certificates is needed.

* **The port** (`Freeze.lean`'s linear replay over `SkipP`): `cert_groupP`, `keepsFastP`, `keepsFastP_eq`,
  `jointAtFastP`, `jointFlagsFastP` and `jointFlagsFastP_eq` are `Freeze`'s `cert_group`, `keepsFast`, `keepsFast_eq`,
  `jointAtFast`, `jointFlagsFast` and `jointFlagsFast_eq` with the atom a parameter `p : Pipeline`. The certificates
  themselves read only `promote` and `rnAdd`, never the atom, so `Freeze.Cert`, `Freeze.certChain`,
  `Freeze.certChain_good` and `Freeze.certChain_getElem` apply as they are.
* **No fallback** (`no_fallback`, `keepsFastP_ofIdx`): with `c` and `d` the FP32 neighbours of `lo` and `hi`
  (`Cert.ofIdx`, `Grid.lean`'s `gridQ` indices), an FP32 total is in `[lo, hi]`, at most `c` or at least `d`, so
  `keepsFast` never falls back to the later promotions: a mask keeps iff its promoted total is finite and in
  `[lo, hi]` (`keepsCut`), which is the circuit's test.
* **The circuit's checks** (`certCut`, `cap.py`'s `_word_body`): per group but the last, `lo` and `hi` promoted with the
  next group's word land in the next interval, and `pred(lo)` and `succ(hi)` promoted land strictly outside it; the last
  interval is the output. They give `Freeze.certChain` of the neighbours' certificates (`certCut_certChain`).
* **The conclusion** (`jointAtCut_eq`): when `certCut` passes, the flags by the circuit's test are `jointFlagsP`.
* `promote_zero_total` and `promote_zero_word` are the two promotions the circuit leaves out: into the total `+0` of
  group 0, and of the all-skipped group's word `+0`.
-/

namespace Pouw.PearlC.Cap

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.Skip Pouw.PearlC.Accum Pouw.PearlC.SkipP
  Pouw.PearlC.FreezeP
open Pouw.PearlC.Freeze (Cert certChain certChain_good certChain_getElem)

variable (p : Pipeline)

/-! ## Freeze's linear replay over `SkipP` -/

/-- A checked certificate list certifies each group's later promotions, under any atom. -/
theorem cert_groupP (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) (hi : i < gs.length)
    (h : certChain ((gs.map (groupWordP p)).drop 1) zs (totalP p gs) = true) (hz : i < zs.length) :
    zs[i].Good (laterWordsP p gs i)
      ((laterWordsP p gs i).foldl promote (promote (inTotalP p gs i) (groupWordP p gs[i]))) := by
  have g := certChain_getElem _ zs _ h i hz
  have e : ((gs.map (groupWordP p)).drop 1).drop i = laterWordsP p gs i := by
    rw [laterWordsP, List.drop_drop, List.map_drop, Nat.add_comm]
  rw [e, total_splitP p gs i hi] at g
  exact g

/-- **The fast later-promotion test** under `p`: a promoted total in the certificate's interval keeps the output, one at
or past its bounds does not, and only one strictly between falls back to the later promotions. -/
def keepsFastP (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool) : Bool :=
  match promote t (wordInP p g m) with
  | some x => if z.lo ≤ x ∧ x ≤ z.hi then true else if x ≤ z.c ∨ z.d ≤ x then false else keepsP p g t ws m
  | none => keepsP p g t ws m

theorem keepsFastP_eq (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool)
    (hz : z.Good ws (ws.foldl promote (promote t (groupWordP p g)))) : keepsFastP p z g t ws m = keepsP p g t ws m := by
  unfold keepsFastP
  split
  · rename_i x hx
    split_ifs with h1 h2
    · simp [keepsP, hx, hz.1 x h1.1 h1.2]
    · rcases h2 with h2 | h2
      · simp [keepsP, hx, hz.2.1 x h2]
      · simp [keepsP, hx, hz.2.2 x h2]
    · rfl
  · rfl

/-- The joint flags of a group by the fast test, under `p`. -/
def jointAtFastP (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) : List Bool :=
  (List.range g.length).map fun j => (allMasks g.length).any fun m => m.getD j false && keepsFastP p z g t ws m

/-- **The joint flags by the linear replay** under `p`: the certificates `zs`, one per group, are checked once
(`certChain`); on success each group's masked totals are tested against its certificate, and on failure the test is
`jointFlagsP`'. -/
def jointFlagsFastP (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) : List Bool :=
  if h : certChain ((gs.map (groupWordP p)).drop 1) zs (totalP p gs) = true ∧ i < zs.length then
    jointAtFastP p zs[i] (gs.getD i []) (inTotalP p gs i) (laterWordsP p gs i)
  else jointFlagsP p gs i

/-- **The linear replay computes debit (A)'s flags** under any atom, for any certificates. -/
theorem jointFlagsFastP_eq (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) (hi : i < gs.length) :
    jointFlagsFastP p gs zs i = jointFlagsP p gs i := by
  unfold jointFlagsFastP
  split
  · rename_i h
    have hz := cert_groupP p gs zs i hi h.1 h.2
    rw [jointFlagsP, jointAtFastP, jointAtP, List.getD_eq_getElem _ _ hi]
    congr 1; funext j; congr 1; funext m; rw [keepsFastP_eq p _ _ _ _ _ hz]
  · rfl

/-! ## The circuit's certificates: `lo` and `hi`, with their FP32 neighbours -/

/-- The certificate of an interval `[gridQ b.1, gridQ b.2]` of FP32 values, with `c` and `d` the FP32 values just below
`lo` and just above `hi` (`cap.py`'s `_Cmp.pred` and `_Cmp.succ`). -/
def Cert.ofIdx (b : ℤ × ℤ) : Cert := ⟨gridQ b.1, gridQ b.2, gridQ (b.1 - 1), gridQ (b.2 + 1)⟩

/-- **`keepsFast` never falls back on an FP32 total** when `c = pred(lo)` and `d = succ(hi)`: no FP32 value lies
strictly between `c` and `lo`, or between `hi` and `d`. -/
theorem no_fallback (b : ℤ × ℤ) {x : ℚ} (hx : OnGrid x) :
    ((Cert.ofIdx b).lo ≤ x ∧ x ≤ (Cert.ofIdx b).hi) ∨ x ≤ (Cert.ofIdx b).c ∨ (Cert.ofIdx b).d ≤ x := by
  simp only [Cert.ofIdx]
  by_cases h1 : x < gridQ b.1
  · exact Or.inr (Or.inl (le_pred_of_lt hx h1))
  · by_cases h2 : gridQ b.2 < x
    · exact Or.inr (Or.inr (succ_le_of_lt hx h2))
    · exact Or.inl ⟨not_lt.1 h1, not_lt.1 h2⟩

/-- **The circuit's test of a mask**: its promoted total is finite and in `[lo, hi]`. -/
def keepsCut (b : ℤ × ℤ) : Option ℚ → Bool
  | some x => decide (gridQ b.1 ≤ x ∧ x ≤ gridQ b.2)
  | none => false

/-- **The fast test is the circuit's** for a certificate of neighbours, on a total from the grid, when the output is
finite: an FP32 total never falls back, and a non-finite one keeps nothing. -/
theorem keepsFastP_ofIdx (b : ℤ × ℤ) (g : List AtomIn) {t : Option ℚ} (ht : GridOpt t) (ws : List Nat)
    (m : List Bool) (hout : ws.foldl promote (promote t (groupWordP p g)) ≠ none) :
    keepsFastP p (Cert.ofIdx b) g t ws m = keepsCut b (promote t (wordInP p g m)) := by
  unfold keepsFastP
  split
  · rename_i x hx
    have hxg : OnGrid x := gridOpt_promote ht _ x hx
    rw [hx]
    simp only [keepsCut]
    by_cases h1 : gridQ b.1 ≤ x ∧ x ≤ gridQ b.2
    · simp [Cert.ofIdx, h1]
    · have h2 : x ≤ gridQ (b.1 - 1) ∨ gridQ (b.2 + 1) ≤ x := (no_fallback b hxg).resolve_left h1
      simp [Cert.ofIdx, h1, h2]
  · rename_i hx
    simp only [keepsP, hx, foldl_promote_none, keepsCut, decide_eq_false_iff_not]
    exact fun e => hout e.symm

/-- **The circuit's certificate checks** (`cap.py`'s `_word_body`), one interval per group by grid index, the last the
output's: for each group but the last, with `v` the next group's word, `lo` and `hi` promoted land in the next interval
and `pred(lo)` and `succ(hi)` promoted land strictly outside it. -/
def certCut : List Nat → List (ℤ × ℤ) → Option ℚ → Bool
  | [], [b], some o => decide (gridQ b.1 = o ∧ gridQ b.2 = o)
  | w :: ws, b :: b' :: bs, out =>
    (match wordQ w with
      | some v => decide (gridQ b'.1 ≤ rnAdd (gridQ b.1) v ∧ rnAdd (gridQ b.2) v ≤ gridQ b'.2 ∧
          rnAdd (gridQ (b.1 - 1)) v < gridQ b'.1 ∧ gridQ b'.2 < rnAdd (gridQ (b.2 + 1)) v)
      | none => false) && certCut ws (b' :: bs) out
  | _, _, _ => false

/-- **The circuit's checks give Freeze's**: an FP32 total strictly below `lo'` is at most `pred(lo')`, and one strictly
above `hi'` at least `succ(hi')`. -/
theorem certCut_certChain (ws : List Nat) (bs : List (ℤ × ℤ)) (out : Option ℚ) (h : certCut ws bs out = true) :
    certChain ws (bs.map Cert.ofIdx) out = true := by
  induction ws generalizing bs with
  | nil =>
    match bs, out, h with
    | [b], some o, h =>
      simp only [certCut, decide_eq_true_eq] at h
      obtain ⟨h1, h2⟩ := h
      simp only [List.map_cons, List.map_nil, certChain, Cert.ofIdx, decide_eq_true_eq]
      refine ⟨h1.ge, h2.le, ?_, ?_⟩
      · rw [← h1]; exact gridQ_lt_iff.2 (by omega)
      · rw [← h2]; exact gridQ_lt_iff.2 (by omega)
  | cons w ws ih =>
    match bs, h with
    | b :: b' :: bs, h =>
      simp only [certCut, Bool.and_eq_true] at h
      obtain ⟨hw, hrest⟩ := h
      simp only [List.map_cons, certChain, Bool.and_eq_true]
      refine ⟨?_, by simpa using ih (b' :: bs) hrest⟩
      cases hv : wordQ w with
      | none => rw [hv] at hw; simp at hw
      | some v =>
        rw [hv] at hw
        simp only [decide_eq_true_eq] at hw ⊢
        obtain ⟨h1, h2, h3, h4⟩ := hw
        have gv := wordQ_onGrid hv
        simp only [Cert.ofIdx]
        refine ⟨h1, h2, le_pred_of_lt (rnAdd_onGrid (onGrid_gridQ _) gv) h3,
          succ_le_of_lt (rnAdd_onGrid (onGrid_gridQ _) gv) h4⟩

theorem certCut_length (ws : List Nat) (bs : List (ℤ × ℤ)) (out : Option ℚ) (h : certCut ws bs out = true) :
    bs.length = ws.length + 1 := by
  induction ws generalizing bs with
  | nil =>
    match bs, out, h with
    | [_], some _, _ => rfl
  | cons w ws ih =>
    match bs, h with
    | b :: b' :: bs, h =>
      simp only [certCut, Bool.and_eq_true] at h
      have := ih (b' :: bs) h.2
      simp only [List.length_cons] at this ⊢
      omega

theorem certCut_out (ws : List Nat) (bs : List (ℤ × ℤ)) (out : Option ℚ) (h : certCut ws bs out = true) :
    out ≠ none := by
  induction ws generalizing bs with
  | nil =>
    match bs, out, h with
    | [_], some _, _ => simp
  | cons w ws ih =>
    match bs, h with
    | b :: b' :: bs, h =>
      simp only [certCut, Bool.and_eq_true] at h
      exact ih (b' :: bs) h.2

/-- The joint flags of a group by the circuit's test against the interval `b`. -/
def jointAtCut (b : ℤ × ℤ) (g : List AtomIn) (t : Option ℚ) : List Bool :=
  (List.range g.length).map fun j => (allMasks g.length).any fun m => m.getD j false && keepsCut b (promote t (wordInP p g m))

theorem gridOpt_inTotalP (gs : List (List AtomIn)) (i : Nat) : GridOpt (inTotalP p gs i) := by
  rw [inTotalP, ← foldl_promote_wordsP]
  exact gridOpt_foldl_promote _ (gridOpt_some (gridQ_zero ▸ onGrid_gridQ 0))

/-- **The circuit's flags are debit (A)'s**: when the circuit's certificate checks pass, whatever the intervals, every
group's flags by its interval test are `SkipP.jointFlagsP`. -/
theorem jointAtCut_eq (gs : List (List AtomIn)) (bs : List (ℤ × ℤ))
    (h : certCut ((gs.map (groupWordP p)).drop 1) bs (totalP p gs) = true) (i : Nat) (hi : i < gs.length) :
    jointAtCut p (bs.getD i (0, 0)) (gs.getD i []) (inTotalP p gs i) = jointFlagsP p gs i := by
  have hlen := certCut_length _ _ _ h
  simp only [List.length_drop, List.length_map] at hlen
  have hb : i < bs.length := by omega
  have hz : i < (bs.map Cert.ofIdx).length := by simpa using hb
  have hc := cert_groupP p gs (bs.map Cert.ofIdx) i hi (certCut_certChain _ _ _ h) hz
  have hout : (laterWordsP p gs i).foldl promote (promote (inTotalP p gs i) (groupWordP p gs[i])) ≠ none := by
    rw [← total_splitP p gs i hi]; exact certCut_out _ _ _ h
  rw [jointFlagsP, jointAtP, jointAtCut, List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hb]
  congr 1; funext j; congr 1; funext m
  rw [← keepsFastP_ofIdx p bs[i] gs[i] (gridOpt_inTotalP p gs i) (laterWordsP p gs i) m hout]
  simp only [List.getElem_map] at hc
  rw [keepsFastP_eq p _ _ _ _ _ hc]

/-! ## The promotions the circuit leaves out -/

/-- Promoting into the total `+0` gives the word's own value (group 0's masked words, and the chain's first total). -/
theorem promote_zero_total (w : Nat) : promote (some 0) w = wordQ w := by
  unfold promote
  cases hx : wordQ w with
  | none => rfl
  | some x => simp [rnAdd, rn_exact (wordQ_representable hx)]

/-- Promoting the word `+0` leaves an FP32 total unchanged (the all-skipped group's word). -/
theorem promote_zero_word {t : ℚ} (ht : OnGrid t) : promote (some t) 0 = some t := by
  have h0 : wordQ 0 = some 0 := by simp [wordQ, fp32Value, Value.finite?, Term.value, Term.zero]
  simp only [promote, h0, rnAdd, add_zero, Option.some.injEq]
  exact rn_exact (onGrid_representable ht)

end Pouw.PearlC.Cap
