import Proofs.Flock.Claims.Complete.Zk.Ligerito
import Proofs.Flock.Claims.Complete.Replay.LigeritoCons
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigPad

/-!
# Level 0's masked query identity

The masked verifier (`Zk.ligeritoWith`) checks level 0's opened rows (`encodeRowsZk`, `2^lanes + 4` lanes, each padded
by `padF`) as the sum of three terms per query `j`, weighted by `eq(α)[j]`: the first `2^lanes` lanes folded by the lane
challenges (`consE0`), rep `r`'s extra pair weighted by `ρ`, and the padding's codeword `padAt` at the prover's `ybar`
(`extraE`). On the honest prover's rows and `ybar`, the paddings cancel, and the sum is the plain one at
`y₁' = f + ρ_0 R_{2r} + ρ_1 R_{2r+1}`, `f` the packed table folded at the lane challenges: `Σ_c y₁'(c)·V(c)` for the
consistency weight `V = consW` (`consE0_honestZk`), as `consE_fold` is for an unmasked level.

* `embed_padF`: lane `l`'s padding `padF` is `padAt` on its coefficients, which lie in `F` (`padAt_c1`).
* `padAt_sum`: `padAt` at position `p` is linear in `ybar`, `Σ_i ybar_i·pe(p, i)` (`padAt_eq`).
* `row_comb`: at one opened row, the three terms are `Enc(y₁')` there: each lane's padding is `Σ_i μ_l[i]·pe(p, i)`,
  and `ybar`'s is the same combination of them, so the two cancel in characteristic 2.
-/

namespace FlockComplete

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.ZkExec FlockComplete.Replay

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace L0

/-! ## Sums of scaled words -/

theorem gOf_foldl_smul (a : ℕ → F256) (b : ℕ → F128) : ∀ (n : ℕ) (acc : F256),
    gOf ((List.range n).foldl (fun acc l => acc + (a l).smul (b l)) acc) =
      gOf acc + ∑ l ∈ Finset.range n, gOf (a l) * FlockLevel3.embed (fOf (b l)) := by
  intro n
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, gOf_add, ih, gOf_smul,
        Finset.sum_range_succ, add_assoc]

theorem fOf_sumF (n : ℕ) (f : ℕ → F128) : fOf (sumF n f) = ∑ x ∈ Finset.range n, fOf (f x) := Zc.of_sumF n f

/-! ## A lane's padding -/

theorem f128_zero_mul (a : F128) : 0 * a = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, zero_mul])

theorem f128_zero_add_zero : (0 : F128) + 0 = 0 := fOf_injective (by rw [fOf_add, fOf_zero, add_zero])

theorem padStep_c1 (ws : Array F128) (ybar : Array F256) (hy : ∀ i : ℕ, (ybar[i]!).c1 = 0) (s : Array F128 × F256)
    (hs : s.2.c1 = 0) (i : ℕ) : (padStep ws ybar s i).2.c1 = 0 := by
  unfold padStep
  split
  · show s.2.c1 + (ybar[i]!).c1 * _ = 0
    rw [hs, hy, f128_zero_mul, f128_zero_add_zero]
  · show s.2.c1 + (ybar[i]!).c1 * _ = 0
    rw [hs, hy, f128_zero_mul, f128_zero_add_zero]

theorem foldl_padStep_c1 (ws : Array F128) (ybar : Array F256) (hy : ∀ i : ℕ, (ybar[i]!).c1 = 0) :
    ∀ (l : List ℕ) (s : Array F128 × F256), s.2.c1 = 0 → (l.foldl (padStep ws ybar) s).2.c1 = 0
  | [], _, hs => hs
  | i :: l, s, hs => foldl_padStep_c1 ws ybar hy l _ (padStep_c1 ws ybar hy s hs i)

/-- `padAt` of coefficients in `F` lies in `F`. -/
theorem padAt_c1 (ws : Array F128) (logMsg : ℕ) (ybar : Array F256) (hy : ∀ i : ℕ, (ybar[i]!).c1 = 0) :
    (Zk.padAt ws logMsg ybar).c1 = 0 := by
  rw [padAt_foldl]
  show (_ : F256).c1 * _ = 0
  rw [foldl_padStep_c1 ws ybar hy _ _ rfl, f128_zero_mul]

theorem ofF_c1 (mu : Array F128) (i : ℕ) : ((mu.map F256.ofF)[i]!).c1 = 0 := by
  by_cases hi : i < mu.size
  · rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map]
    rfl
  · rw [getElem!_neg _ i (by simpa using hi)]
    rfl

/-- **A lane's padding** is `padAt` on its coefficients. -/
theorem embed_padF (ws : Array F128) (logMsg : ℕ) (mu : Array F128) :
    FlockLevel3.embed (fOf (padF ws logMsg mu)) = gOf (Zk.padAt ws logMsg (mu.map F256.ofF)) := by
  rw [gOf_limbs (Zk.padAt ws logMsg (mu.map F256.ofF)), padAt_c1 _ _ _ (ofF_c1 mu), fOf_zero, map_zero, zero_mul,
    add_zero]
  rfl

/-- The padding's codeword at position `p` and coefficient `i`: `c·X̂_i(ω_p) + X̂_{2^cols + i}(ω_p)`. -/
noncomputable def pe (cols p i : ℕ) : 𝕂 :=
  execArith.embed (fOf Zk.PAD_C * execArith.Xhat i (cols + 1) (execArith.omega p) +
    execArith.Xhat (2 ^ cols + i) (cols + 1) (execArith.omega p))

/-- **`padAt` is linear in `ybar`**: at position `p`, `Σ_i ybar_i·pe(p, i)`. -/
theorem padAt_sum (p cols tPad : ℕ) (ybar : Array F256) (hy : ybar.size = tPad) (ht : tPad ≤ 2 ^ cols) :
    gOf (Zk.padAt (whats (svTable (cols + 1)) p (cols + 1)) cols ybar) =
      ∑ i ∈ Finset.range tPad, gOf ybar[i]! * pe cols p i := by
  rw [padAt_eq p cols tPad ybar hy ht]
  unfold Model.padCorr
  exact Fin.sum_univ_eq_sum_range (fun i => gOf ybar[i]! * pe cols p i) tPad

/-! ## An opened row -/

/-- Entry `l` of level 0's padded row at `q`: lane `l`'s encoding there plus its padding. -/
theorem encodeRowsZk_get (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q l : ℕ}
    (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES) :
    ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]! =
      sumF (2 ^ lv.cols) (fun c => (if l < 2 ^ lv.lanes then fun c => g[c * 2 ^ lv.lanes + l]! else
        extra (l - 2 ^ lv.lanes)) c * (xhats (svTable (lv.cols + 1)) q lv.cols)[c]!) +
      padF (whats (svTable (lv.cols + 1)) q (lv.cols + 1)) lv.cols ((Array.range tPad).map (lanePad l)) := by
  unfold encodeRowsZk
  simp only
  rw [getElem!_range_map _ hq, getElem!_range_map _ hl, getElem!_range_map _ hl]

/-- A lane's encoding at `q`, in the model's field. -/
noncomputable def enc (cols q : ℕ) (v : ℕ → F128) : 𝕂 :=
  ∑ c ∈ Finset.range (2 ^ cols), FlockLevel3.embed (fOf (v c)) *
    FlockLevel3.embed (fOf (xhats (svTable (cols + 1)) q cols)[c]!)

/-- A lane's padding at `q`, in the model's field. -/
noncomputable def padv (cols q tPad : ℕ) (mu : ℕ → F128) : 𝕂 :=
  ∑ i ∈ Finset.range tPad, FlockLevel3.embed (fOf (mu i)) * pe cols q i

theorem entry_eq (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q l : ℕ}
    (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES) (ht : tPad ≤ 2 ^ lv.cols) :
    FlockLevel3.embed (fOf ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]!) =
      enc lv.cols q (if l < 2 ^ lv.lanes then fun c => g[c * 2 ^ lv.lanes + l]! else extra (l - 2 ^ lv.lanes)) +
        padv lv.cols q tPad (lanePad l) := by
  rw [encodeRowsZk_get lv tPad g extra lanePad hq hl, fOf_add, map_add, embed_padF,
    padAt_sum q lv.cols tPad _ (by simp) ht, fOf_sumF, map_sum]
  unfold enc padv
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun c _ => by rw [fOf_mul, map_mul]) ?_
  refine Finset.sum_congr rfl fun i hi => ?_
  have hi' := Finset.mem_range.1 hi
  rw [getElem!_pos _ i (by simpa using hi'), Array.getElem_map, Array.getElem_map, Array.getElem_range, gOf_ofF]

/-- **The padding's combination** the honest prover sends: `ybar_i = Σ_l eq(r̄)_l μ_l[i] + ρ_0 μ_{W+2r}[i] +
ρ_1 μ_{W+2r+1}[i]` (`ligeritoPZk`'s, at `W = 2^lanes` and `ef = eq(r̄)`). -/
def ybarOf (ef : Array F256) (rho : F256 × F256) (rep W tPad : ℕ) (lanePad : ℕ → ℕ → F128) : Array F256 :=
  (Array.range tPad).map fun i =>
    (List.range W).foldl (fun acc l => acc + (ef[l]!).smul (lanePad l i))
      (rho.1.smul (lanePad (W + 2 * rep) i) + rho.2.smul (lanePad (W + 2 * rep + 1) i))

theorem gOf_ybarOf (ef : Array F256) (rho : F256 × F256) (rep W tPad : ℕ) (lanePad : ℕ → ℕ → F128) {i : ℕ}
    (hi : i < tPad) :
    gOf (ybarOf ef rho rep W tPad lanePad)[i]! =
      gOf rho.1 * FlockLevel3.embed (fOf (lanePad (W + 2 * rep) i)) +
        gOf rho.2 * FlockLevel3.embed (fOf (lanePad (W + 2 * rep + 1) i)) +
        ∑ l ∈ Finset.range W, gOf ef[l]! * FlockLevel3.embed (fOf (lanePad l i)) := by
  unfold ybarOf
  rw [getElem!_range_map _ hi, gOf_foldl_smul, gOf_add, gOf_smul, gOf_smul]

theorem pad_reorder {K : Type} [CommSemiring K] (T W : ℕ) (a b : K) (e A B pe : ℕ → K) (M : ℕ → ℕ → K) :
    ∑ i ∈ Finset.range T, (a * A i + b * B i + ∑ l ∈ Finset.range W, e l * M l i) * pe i =
      a * ∑ i ∈ Finset.range T, A i * pe i + b * ∑ i ∈ Finset.range T, B i * pe i +
        ∑ l ∈ Finset.range W, e l * ∑ i ∈ Finset.range T, M l i * pe i := by
  simp only [add_mul, Finset.sum_add_distrib, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm (s := Finset.range T) (t := Finset.range W)]
  simp only [mul_assoc]

theorem enc_reorder {K : Type} [CommSemiring K] (C W : ℕ) (a b : K) (e u v xh : ℕ → K) (G : ℕ → ℕ → K) :
    ∑ c ∈ Finset.range C, (∑ l ∈ Finset.range W, e l * G c l + a * u c + b * v c) * xh c =
      ∑ l ∈ Finset.range W, e l * ∑ c ∈ Finset.range C, G c l * xh c + a * ∑ c ∈ Finset.range C, u c * xh c +
        b * ∑ c ∈ Finset.range C, v c * xh c := by
  simp only [add_mul, Finset.sum_add_distrib, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm (s := Finset.range C) (t := Finset.range W)]
  simp only [mul_assoc]

/-- **One opened row, combined**: the first `2^lanes` lanes folded by `ef`, rep `r`'s pair weighted by `ρ`, and the
padding's codeword at the prover's `ybar` give `Enc(y₁')` at the row's position: the paddings cancel. -/
theorem row_comb (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) (ef : Array F256)
    (rep : ℕ) (rho : F256 × F256) {q : ℕ} (hq : q < 2 ^ lv.d) (hrep : rep < 2) (ht : tPad ≤ 2 ^ lv.cols) :
    gOf ((List.range (2 ^ lv.lanes)).foldl (fun acc l =>
        acc + F256.smul ef[l]! ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]!) 0) +
      gOf (rho.1.smul ((encodeRowsZk lv tPad g extra lanePad)[q]!)[2 ^ lv.lanes + 2 * rep]! +
        rho.2.smul ((encodeRowsZk lv tPad g extra lanePad)[q]!)[2 ^ lv.lanes + 2 * rep + 1]! +
        Zk.padAt (whats (svTable (lv.cols + 1)) q (lv.cols + 1)) lv.cols
          (ybarOf ef rho rep (2 ^ lv.lanes) tPad lanePad)) =
      ∑ c ∈ Finset.range (2 ^ lv.cols),
        (∑ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! * FlockLevel3.embed (fOf g[c * 2 ^ lv.lanes + l]!) +
          gOf rho.1 * FlockLevel3.embed (fOf (extra (2 * rep) c)) +
          gOf rho.2 * FlockLevel3.embed (fOf (extra (2 * rep + 1) c))) *
        FlockLevel3.embed (fOf (xhats (svTable (lv.cols + 1)) q lv.cols)[c]!) := by
  have hE : Zk.EXTRA_LANES = 4 := rfl
  have two : ∀ x : 𝕂, x + x = 0 := fun x => CharTwo.add_self_eq_zero x
  have hfold := gOf_foldl_smul (fun l => ef[l]!) (fun l => ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]!)
    (2 ^ lv.lanes) 0
  rw [hfold, gOf_zero, zero_add, gOf_add, gOf_add, gOf_smul, gOf_smul,
    padAt_sum q lv.cols tPad _ (by simp [ybarOf]) ht]
  have hl : ∀ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! *
      FlockLevel3.embed (fOf ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]!) =
      gOf ef[l]! * enc lv.cols q (fun c => g[c * 2 ^ lv.lanes + l]!) +
        gOf ef[l]! * padv lv.cols q tPad (lanePad l) := by
    intro l hl
    have hl' := Finset.mem_range.1 hl
    rw [entry_eq lv tPad g extra lanePad hq (by rw [hE]; omega) ht, ite_eq_left hl', mul_add]
  rw [Finset.sum_congr rfl hl, Finset.sum_add_distrib,
    entry_eq lv tPad g extra lanePad hq (by rw [hE]; omega) ht,
    entry_eq lv tPad g extra lanePad hq (by rw [hE]; omega) ht, ite_eq_right (by omega), ite_eq_right (by omega),
    show 2 ^ lv.lanes + 2 * rep - 2 ^ lv.lanes = 2 * rep by omega,
    show 2 ^ lv.lanes + 2 * rep + 1 - 2 ^ lv.lanes = 2 * rep + 1 by omega]
  have hy : ∑ i ∈ Finset.range tPad, gOf (ybarOf ef rho rep (2 ^ lv.lanes) tPad lanePad)[i]! * pe lv.cols q i =
      gOf rho.1 * padv lv.cols q tPad (lanePad (2 ^ lv.lanes + 2 * rep)) +
        gOf rho.2 * padv lv.cols q tPad (lanePad (2 ^ lv.lanes + 2 * rep + 1)) +
        ∑ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! * padv lv.cols q tPad (lanePad l) := by
    rw [Finset.sum_congr rfl fun i hi => by
      rw [gOf_ybarOf ef rho rep _ tPad lanePad (Finset.mem_range.1 hi)]]
    exact pad_reorder _ _ _ _ _ _ _ _ _
  have hR : ∑ c ∈ Finset.range (2 ^ lv.cols),
      (∑ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! * FlockLevel3.embed (fOf g[c * 2 ^ lv.lanes + l]!) +
          gOf rho.1 * FlockLevel3.embed (fOf (extra (2 * rep) c)) +
          gOf rho.2 * FlockLevel3.embed (fOf (extra (2 * rep + 1) c))) *
        FlockLevel3.embed (fOf (xhats (svTable (lv.cols + 1)) q lv.cols)[c]!) =
      ∑ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! * enc lv.cols q (fun c => g[c * 2 ^ lv.lanes + l]!) +
        gOf rho.1 * enc lv.cols q (extra (2 * rep)) + gOf rho.2 * enc lv.cols q (extra (2 * rep + 1)) :=
    enc_reorder _ _ _ _ _ _ _ _ _
  rw [hy, hR]
  linear_combination two (gOf rho.1 * padv lv.cols q tPad (lanePad (2 ^ lv.lanes + 2 * rep)) +
    gOf rho.2 * padv lv.cols q tPad (lanePad (2 ^ lv.lanes + 2 * rep + 1)) +
    ∑ l ∈ Finset.range (2 ^ lv.lanes), gOf ef[l]! * padv lv.cols q tPad (lanePad l))

/-! ## The queries, batched -/

/-- **Level 0's masked consistency value on the honest opening**: the rows the masked prover committed (`encodeRowsZk`
of the packed table `g`), opened at `pos`, with the prover's `ybar`, give `Σ_c y₁'(c)·V(c)` for any K-table `y` whose
entries are `y₁' = f + ρ_0 R_{2r} + ρ_1 R_{2r+1}`, `f` the table `g` folded at the lane challenges `ch`. -/
theorem consE0_honestZk (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) (pos : Array ℕ)
    (alpha : Array F128) (ch : Array F256) (rep : ℕ) (rho : F256 × F256) (y : Array F256)
    (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (hpos : ∀ j < pos.size, pos[j]! < 2 ^ lv.d)
    (hrep : rep < 2) (ht : tPad ≤ 2 ^ lv.cols)
    (hy : ∀ c < 2 ^ lv.cols, Lg.gv y c = FlockSoundness.foldLoL (ch.toList.map gOf) (Lg.gv (embedF g)) c +
      gOf rho.1 * FlockLevel3.embed (fOf (extra (2 * rep) c)) +
      gOf rho.2 * FlockLevel3.embed (fOf (extra (2 * rep + 1) c))) :
    gOf (consE0 (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos.size (2 ^ lv.lanes) alpha ch +
        extraE (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos (2 ^ lv.lanes) rep lv.cols rho alpha
          (ybarOf (eqTableK ch) rho rep (2 ^ lv.lanes) tPad lanePad)) =
      ∑ c ∈ Finset.range (2 ^ lv.cols), Lg.gv y c * Lg.gv (embedF (consW lv.cols pos alpha)) c := by
  have hrow : ∀ j < pos.size, (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!)[j]! =
      (encodeRowsZk lv tPad g extra lanePad)[pos[j]!]! := fun j hj => by
    rw [getElem!_pos _ j (by simpa using hj), Array.getElem_map, ← getElem!_pos pos j hj]
  unfold consE0 extraE
  rw [← List.range_eq_range', gOf_add, gOf_foldl_cons, gOf_foldl_cons, gOf_zero, zero_add, zero_add,
    ← Finset.sum_add_distrib]
  have hY : ∀ c < 2 ^ lv.cols,
      ∑ l ∈ Finset.range (2 ^ lv.lanes), gOf (eqTableK ch)[l]! * FlockLevel3.embed (fOf g[c * 2 ^ lv.lanes + l]!) +
        gOf rho.1 * FlockLevel3.embed (fOf (extra (2 * rep) c)) +
        gOf rho.2 * FlockLevel3.embed (fOf (extra (2 * rep + 1) c)) = Lg.gv y c := by
    intro c hc
    rw [hy c hc, FlockSoundness.foldLoL_eq_sum, List.length_map, Array.length_toList, hch]
    refine congrArg (· + _) (congrArg (· + _) (Finset.sum_congr rfl fun l hl => ?_))
    have hl' := Finset.mem_range.1 hl
    have hlt : l + 2 ^ lv.lanes * c < g.size := by
      rw [hg, pow_add]
      nlinarith
    rw [eqTableK_eqList _ _ (by rw [hch]; exact hl'), Lg.gv_embedF _ hlt,
      show l + 2 ^ lv.lanes * c = c * 2 ^ lv.lanes + l by ring]
  have hj : ∀ j ∈ Finset.range pos.size,
      gOf ((List.range (2 ^ lv.lanes)).foldl (fun acc l => acc + F256.smul (eqTableK ch)[l]!
          ((pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!)[j]!)[l]!) 0) *
          execArith.embed (fOf (eqTable alpha)[j]!) +
        gOf (rho.1.smul ((pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!)[j]!)[2 ^ lv.lanes + 2 * rep]! +
          rho.2.smul ((pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!)[j]!)[2 ^ lv.lanes + 2 * rep + 1]! +
          Zk.padAt (whats (svTable (lv.cols + 1)) pos[j]! (lv.cols + 1)) lv.cols
            (ybarOf (eqTableK ch) rho rep (2 ^ lv.lanes) tPad lanePad)) *
          execArith.embed (fOf (eqTable alpha)[j]!) =
      ∑ c ∈ Finset.range (2 ^ lv.cols), Lg.gv y c *
        (FlockLevel3.embed (fOf (eqTable alpha)[j]!) *
          FlockLevel3.embed (fOf (xhats (svTable (lv.cols + 1)) pos[j]! lv.cols)[c]!)) := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [← add_mul, hrow j hj', row_comb lv tPad g extra lanePad (eqTableK ch) rep rho (hpos j hj') hrep ht]
    refine (Finset.sum_mul _ _ _).trans (Finset.sum_congr rfl fun c hc => ?_)
    rw [hY c (Finset.mem_range.1 hc)]
    show _ * _ * FlockLevel3.embed _ = _
    ring
  rw [Finset.sum_congr rfl hj, Finset.sum_comm]
  refine Finset.sum_congr rfl fun c hc => ?_
  have hc' := Finset.mem_range.1 hc
  refine (Finset.mul_sum _ _ _).symm.trans ?_
  rw [Lg.gv_embedF _ (by rw [Lg.consW_size]; exact hc')]
  refine congrArg (Lg.gv y c * ·) ?_
  unfold consW
  rw [getElem!_range_map _ hc', fOf_sumF, map_sum]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' := Finset.mem_range.1 hj
  rw [fOf_mul, map_mul, getElem!_pos (pos.map _) j (by simpa using hj'), Array.getElem_map,
    ← getElem!_pos pos j hj']

/-! ## The claim, step by step -/

theorem gOf_foldl_mul (a b : Array F256) : ∀ (n : ℕ) (acc : F256),
    gOf ((List.range n).foldl (fun acc c => acc + a[c]! * b[c]!) acc) =
      gOf acc + ∑ c ∈ Finset.range n, gOf a[c]! * gOf b[c]! := by
  intro n
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, gOf_add, ih, gOf_mul,
        Finset.sum_range_succ, add_assoc]

theorem gOf_dotK (a b : Array F256) :
    gOf (dotK a b) = ∑ c ∈ Finset.range a.size, Lg.gv b c * Lg.gv a c := by
  unfold dotK
  rw [gOf_foldl_mul, gOf_zero, zero_add]
  exact Finset.sum_congr rfl fun c _ => mul_comm _ _

/-- **The lane phase's close**: the blinded claim `T' = ⟨w, y₁'⟩` and the message of `y₁'` split against the weight
lifted by `u`, on tables of `2^n` entries. -/
theorem sync_laneClose (w y : Array F256) (n : ℕ) (hw : w.size = 2 ^ n) (hy : y.size = 2 ^ n) :
    Lg.Sync (dotK w y) (Poly.ofMsg (roundPoly (embedF (splitK y)) (liftU w)) (dotK w y)) (embedF (splitK y))
      (liftU w) n := by
  refine Lg.Sync.ofMsg (by rw [Lg.embedF_size, (Lg.splitK_spec _).1, hy, pow_succ]; ring)
    (by rw [(Lg.liftU_spec _).1, hw, pow_succ]; ring) ?_
  rw [Lg.sum_split _ _ n hy hw, gOf_dotK, hw]

/-- **Level 0's masked consistency claim**: on the split of `y₁'`, the claim gains `b·(consE0 + extraE)` at the
honest opening, and the weight `b·u^bit·V`. -/
theorem Sync.closeZk {s : F256} {q : Poly} {w : Array F256} {N : ℕ} (lv : Level) (tPad : ℕ) (g : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (pos : Array ℕ) (alpha : Array F128) (ch : Array F256) (rep : ℕ)
    (rho : F256 × F256) (y : Array F256) (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols))
    (hpos : ∀ j < pos.size, pos[j]! < 2 ^ lv.d) (hrep : rep < 2) (ht : tPad ≤ 2 ^ lv.cols) (hn : lv.cols = N)
    (hys : y.size = 2 ^ N)
    (hy : ∀ c < 2 ^ lv.cols, Lg.gv y c = FlockSoundness.foldLoL (ch.toList.map gOf) (Lg.gv (embedF g)) c +
      gOf rho.1 * FlockLevel3.embed (fOf (extra (2 * rep) c)) +
      gOf rho.2 * FlockLevel3.embed (fOf (extra (2 * rep + 1) c)))
    (h : Lg.Sync s q (embedF (splitK y)) w N) (b : F128) :
    Lg.Sync (s + (consE0 (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos.size (2 ^ lv.lanes) alpha
          ch + extraE (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos (2 ^ lv.lanes) rep lv.cols
          rho alpha (ybarOf (eqTableK ch) rho rep (2 ^ lv.lanes) tPad lanePad)).smul b)
      (q.addScaled (Poly.ofMsg (roundPoly (embedF (splitK y)) (liftU (embedF (consW lv.cols pos alpha))))
        (consE0 (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos.size (2 ^ lv.lanes) alpha ch +
          extraE (pos.map fun p => (encodeRowsZk lv tPad g extra lanePad)[p]!) pos (2 ^ lv.lanes) rep lv.cols rho
            alpha (ybarOf (eqTableK ch) rho rep (2 ^ lv.lanes) tPad lanePad))) b)
      (embedF (splitK y)) (addScaled w (liftU (embedF (consW lv.cols pos alpha))) b) N := by
  have hv : (embedF (consW lv.cols pos alpha)).size = 2 ^ N := by rw [Lg.embedF_size, Lg.consW_size, hn]
  refine h.add _ _ ?_ b
  rw [Lg.sum_split _ _ N hys hv, consE0_honestZk lv tPad g extra lanePad pos alpha ch rep rho y hch hg hpos hrep ht hy,
    hn]

end L0

end FlockComplete
