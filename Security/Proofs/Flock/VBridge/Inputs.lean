import Proofs.Flock.VBridge.RecOpen
import Proofs.Flock.Soundness.Refine.Arith

/-!
# V*'s verifier-side words: the shape, the openings' coefficients and their running sums

The words the auditor computes for V* (the plan's piece F), as Lean functions of the inner statement and the rep's
coins:

* `repShape`: the shape `verifyRep` decodes a rep's proof at (Verify.lean:40), from the `Setup`'s fields, the fast100
  schedule and the extra claims `st.extra` gives at the session's points. The Algebra's `verifierRows` and
  `structureOf` take it as their `sh`.
* `coefAt`: V*'s `coef` at an opening, `κ·eq(α)[j]` (`rec_outer.coefficients`): `κ[r][l]·e` at `r · LANES + l` for the
  final check's components `r = 0, 1` and the level's lanes `l < LANES`, with `e = eq(α)[j]` (`Flock.eqTable`).
* `chainAcc`: `acc`'s first two elements after a level's openings in order, each opening `RecOpen` at `coefAt`, whose
  `acc_out` begins with `consTerm` at `k = 0, 1` (`sound_recOpen`).

`chainAcc_level`: the chain telescopes. Over the openings `j < q` of a level, at their weights `eq(α)[j]` and rows
`rows[j]`, the running sum after the openings plus the one before is `Σ_l κ[k][l] · Σ_j eq(α)[j]·rows[j][l]`: the
level's component `k` of the final check's consistency sums, the Algebra's `kapSum` term on `consAt rows q α`.
-/

namespace FlockVBridge

open Flock FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The shape -/

/-- **The shape `verifyRep` decodes a rep's proof at** (Verify.lean:40), from the `Setup`'s fields, the schedule `sch`
(`fast100 st.m`) and the extra claims `extras` (`st.extra sess.points`). -/
def repShape (st : Setup) (sch : Schedule) (extras : Array Claim) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes }

/-! ## The openings' coefficients -/

/-- **V*'s `coef` at an opening whose `eq(α)` weight is `e`** (`coef = κ·eq(α)[j]`): `κ[r][l]·e` at `r · LANES + l`, for
the final check's components `r = 0, 1` and the level's lanes `l < LANES`. -/
def coefAt (kap : Array (Array F128)) (LANES : ℕ) (e : F128) : Array F128 :=
  Array.ofFn (n := LANES) (fun l => (kap[0]!)[l.1]! * e) ++ Array.ofFn (n := LANES) (fun l => (kap[1]!)[l.1]! * e)

theorem coefAt_get (kap : Array (Array F128)) (LANES : ℕ) (e : F128) {k l : ℕ} (hk : k < 2) (hl : l < LANES) :
    (coefAt kap LANES e)[k * LANES + l]! = (kap[k]!)[l]! * e := by
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
  · simp [coefAt, getElem!_def, Array.getElem?_append, hl]
  · simp [coefAt, getElem!_def, hl]

theorem fOf_foldl_add (g : ℕ → F128) : ∀ (n : ℕ) (x : F128),
    fOf (((List.range n).map g).foldl (· + ·) x) = fOf x + ∑ l ∈ Finset.range n, fOf (g l)
  | 0, x => by simp
  | n + 1, x => by
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, fOf_add, fOf_foldl_add g n x, Finset.sum_range_succ, add_assoc]

/-- An opening's consistency term at `coefAt`: `a[k] + Σ_l κ[k][l]·e·row[l]`. -/
theorem fOf_consTerm_coefAt (LANES : ℕ) (kap : Array (Array F128)) (e : F128) (row a : Array F128) {k : ℕ}
    (hk : k < 2) : fOf (consTerm LANES row a (coefAt kap LANES e) k) =
      fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * (fOf e * fOf row[l]!) := by
  rw [consTerm, fOf_foldl_add]
  refine congrArg (fOf a[k]! + ·) (Finset.sum_congr rfl fun l hl => ?_)
  rw [Finset.mem_range] at hl
  rw [coefAt_get kap LANES e hk hl, fOf_mul, fOf_mul, mul_assoc]

/-! ## The running sums -/

theorem sum_map_range (f : ℕ → 𝔽) : ∀ q, ((List.range q).map f).sum = ∑ j ∈ Finset.range q, f j
  | 0 => by simp
  | q + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_map_range f q, Finset.sum_range_succ]
    simp

/-- `acc`'s first two elements after V*'s openings `os` in order, from `a`: each opening, its `eq(α)` weight and its
row, is `RecOpen` at `coefAt`, whose `acc_out` begins with `consTerm` at `k = 0, 1`. -/
def chainAcc (LANES : ℕ) (kap : Array (Array F128)) : List (F128 × Array F128) → Array F128 → Array F128
  | [], a => a
  | o :: os, a => chainAcc LANES kap os
      #[consTerm LANES o.2 a (coefAt kap LANES o.1) 0, consTerm LANES o.2 a (coefAt kap LANES o.1) 1]

theorem chainAcc_sum (LANES : ℕ) (kap : Array (Array F128)) :
    ∀ (os : List (F128 × Array F128)) (a : Array F128) {k : ℕ}, k < 2 →
      fOf (chainAcc LANES kap os a)[k]! =
        fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * (os.map fun o => fOf o.1 * fOf o.2[l]!).sum
  | [], a, k, _ => by simp [chainAcc]
  | o :: os, a, k, hk => by
    rw [chainAcc, chainAcc_sum LANES kap os _ hk]
    have h : (#[consTerm LANES o.2 a (coefAt kap LANES o.1) 0, consTerm LANES o.2 a (coefAt kap LANES o.1) 1] :
        Array F128)[k]! = consTerm LANES o.2 a (coefAt kap LANES o.1) k := by
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl
    rw [h, fOf_consTerm_coefAt LANES kap o.1 o.2 a hk, add_assoc, ← Finset.sum_add_distrib]
    refine congrArg (fOf a[k]! + ·) (Finset.sum_congr rfl fun l _ => ?_)
    rw [List.map_cons, List.sum_cons, mul_add]

/-- **A level's running sums telescope to its consistency sums.** Over a level's openings `j < q` in order, at their
weights `eq(α)[j]` (`Flock.eqTable`) and rows `rows[j]`, each `RecOpen` at `coefAt κ`, the running sum's component
`k < 2` after the openings plus the one before is `Σ_l κ[k][l] · Σ_j eq(α)[j]·rows[j][l]`: the final check's component
`k` on the level's consistency sums, `kapSum`'s term on `consAt rows q α`. -/
theorem chainAcc_level (LANES : ℕ) (kap : Array (Array F128)) (rows : Array (Array F128)) (q : ℕ)
    (alpha a : Array F128) {k : ℕ} (hk : k < 2) :
    fOf (chainAcc LANES kap ((List.range q).map fun j => ((eqTable alpha)[j]!, rows[j]!)) a)[k]! + fOf a[k]! =
      ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * ∑ j ∈ Finset.range q, fOf (eqTable alpha)[j]! * fOf (rows[j]!)[l]! := by
  rw [chainAcc_sum LANES kap _ a hk, add_comm (fOf a[k]!) _, add_assoc, CharTwo.add_self_eq_zero, add_zero]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [List.map_map, sum_map_range]
  rfl

end FlockVBridge
