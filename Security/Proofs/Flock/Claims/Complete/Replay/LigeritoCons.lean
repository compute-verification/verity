import Proofs.Flock.Claims.Complete.Replay.LigeritoSync

/-!
# The consistency value on an honest opening

A level's consistency value (`consE`) reads the opened codeword rows, folded at the level's challenges and batched by
`eq(α)`. On rows the honest prover committed (`encodeRows` of its F-table `g`), it is the sum of `g` folded at the
challenges against the prover's consistency weight `V = consW` (`consE_honest`): row `pos_j`'s lane `ℓ` is
`Σ_c g[c·2^lanes + ℓ]·X̂_c(ω_{pos_j})`, so folding the lanes folds `g`'s low variables (`foldLoL`), and batching the
rows by `eq(α)` batches the `X̂_c(ω_{pos_j})` into `V(c)`. The prover's iterated `foldK` is the model's `foldLoL`
(`gv_foldKs`).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-- A K-table folded at each challenge in turn, lowest variable first. -/
def foldKs (ch : List F256) (a : Array F256) : Array F256 := ch.foldl (fun a r => foldK r a) a

theorem foldKs_size : ∀ (ch : List F256) (a : Array F256), (foldKs ch a).size = a.size / 2 ^ ch.length
  | [], a => by simp [foldKs]
  | r :: rs, a => by
      rw [show foldKs (r :: rs) a = foldKs rs (foldK r a) from rfl, foldKs_size rs, foldK_size, List.length_cons,
        pow_succ, Nat.div_div_eq_div_mul, mul_comm]

/-- **The prover's folds are the model's**: entry `y` of the iterated fold is `foldLoL` of the table's entries. -/
theorem gv_foldKs : ∀ (ch : List F256) (a : Array F256) (y : ℕ), y < a.size / 2 ^ ch.length →
    gv (foldKs ch a) y = foldLoL (ch.map gOf) (gv a) y
  | [], a, y, _ => by simp [foldKs, foldLoL]
  | r :: rs, a, y, hy => by
      rw [show foldKs (r :: rs) a = foldKs rs (foldK r a) from rfl, List.map_cons, foldLoL,
        gv_foldKs rs (foldK r a) y (by rw [foldK_size, Nat.div_div_eq_div_mul, ← pow_succ']; exact hy)]
      refine foldLoL_congr _ _ _ y fun x hx => ?_
      rw [List.length_map] at hx ⊢
      rw [List.length_cons, pow_succ'] at hy
      have h1 : y < a.size / 2 / 2 ^ rs.length := by rw [Nat.div_div_eq_div_mul]; exact hy
      have h2 : (y + 1) * 2 ^ rs.length ≤ a.size / 2 := (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).1 h1
      exact gv_foldK r a (by nlinarith)

theorem cons_reorder {K : Type} [CommSemiring K] (J L C : ℕ) (e : ℕ → K) (G xv : ℕ → ℕ → K) (A : ℕ → K) :
    ∑ j ∈ Finset.range J, (∑ l ∈ Finset.range L, e l * ∑ c ∈ Finset.range C, G c l * xv j c) * A j =
      ∑ c ∈ Finset.range C, (∑ l ∈ Finset.range L, e l * G c l) * ∑ j ∈ Finset.range J, A j * xv j c := by
  calc _ = ∑ j ∈ Finset.range J, ∑ l ∈ Finset.range L, ∑ c ∈ Finset.range C, e l * (G c l * xv j c) * A j := by
        simp only [Finset.mul_sum, Finset.sum_mul]
    _ = ∑ l ∈ Finset.range L, ∑ j ∈ Finset.range J, ∑ c ∈ Finset.range C, e l * (G c l * xv j c) * A j :=
        Finset.sum_comm
    _ = ∑ l ∈ Finset.range L, ∑ c ∈ Finset.range C, ∑ j ∈ Finset.range J, e l * (G c l * xv j c) * A j :=
        Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ c ∈ Finset.range C, ∑ l ∈ Finset.range L, ∑ j ∈ Finset.range J, e l * (G c l * xv j c) * A j :=
        Finset.sum_comm
    _ = ∑ c ∈ Finset.range C, ∑ j ∈ Finset.range J, ∑ l ∈ Finset.range L, e l * (G c l * xv j c) * A j :=
        Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = _ := by
        simp only [Finset.mul_sum, Finset.sum_mul]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

/-- A committed row: lane `ℓ` of the codeword at `q`. -/
theorem encodeRows_get (lv : Level) (g : Array F128) {q : ℕ} (hq : q < 2 ^ lv.d) :
    (encodeRows lv g)[q]! = (Array.range (2 ^ lv.lanes)).map fun l =>
      sumF (2 ^ lv.cols) fun c => g[c * 2 ^ lv.lanes + l]! * (xhats (svTable (lv.cols + 1)) q lv.cols)[c]! := by
  unfold encodeRows
  exact getElem!_range_map _ hq

/-- **The consistency value on an honest opening**: the rows the prover committed at level `lv`, opened at `pos` and
folded at the level's challenges `ch`, batched by `eq(α)`, give `g` folded at `ch` against the consistency weight. -/
theorem consE_honest (lv : Level) (g : Array F128) (pos : Array ℕ) (alpha : Array F128) (ch : Array F256)
    (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (hpos : ∀ j < pos.size, pos[j]! < 2 ^ lv.d) :
    gOf (consE (pos.map fun p => (encodeRows lv g)[p]!) pos.size alpha ch) =
      ∑ c ∈ Finset.range (2 ^ lv.cols), foldLoL (ch.toList.map gOf) (gv (embedF g)) c *
        FlockLevel3.embed (fOf (consW lv.cols pos alpha)[c]!) := by
  set xh := fun j : ℕ => xhats (svTable (lv.cols + 1)) pos[j]! lv.cols
  set rows := pos.map fun p => (encodeRows lv g)[p]!
  have hrow : ∀ j < pos.size, rows[j]! = (Array.range (2 ^ lv.lanes)).map
      fun l => sumF (2 ^ lv.cols) fun c => g[c * 2 ^ lv.lanes + l]! * (xh j)[c]! := by
    intro j hj
    rw [getElem!_pos _ j (by simpa [rows] using hj), Array.getElem_map, ← getElem!_pos pos j hj]
    exact encodeRows_get lv g (hpos j hj)
  unfold consE
  rw [gOf_foldl_cons, gOf_zero, zero_add]
  have hL : ∀ j ∈ Finset.range pos.size,
      gOf ((List.range rows[j]!.size).foldl (fun acc l => acc + F256.smul (eqTableK ch)[l]! (rows[j]!)[l]!) 0) *
        execArith.embed (fOf (eqTable alpha)[j]!) =
      (∑ l ∈ Finset.range (2 ^ lv.lanes), Model.eqList (ch.toList.map gOf) l *
        ∑ c ∈ Finset.range (2 ^ lv.cols),
          FlockLevel3.embed (fOf g[c * 2 ^ lv.lanes + l]!) * FlockLevel3.embed (fOf (xh j)[c]!)) *
        FlockLevel3.embed (fOf (eqTable alpha)[j]!) := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [gOf_foldl_row, gOf_zero, zero_add, hrow j hj', Array.size_map, Array.size_range]
    refine congrArg (· * _) (Finset.sum_congr rfl fun l hl => ?_)
    have hl' := Finset.mem_range.1 hl
    rw [eqTableK_eqList _ _ (by rw [hch]; exact hl'), getElem!_range_map _ hl']
    show _ * FlockLevel3.embed (FlockLevel3.GF128.of _) = _
    rw [Zc.of_sumF, map_sum]
    exact congrArg (_ * ·) (Finset.sum_congr rfl fun c _ => by
      show FlockLevel3.embed (fOf (_ * _)) = _
      rw [fOf_mul, map_mul])
  rw [Finset.sum_congr rfl hL, cons_reorder]
  refine Finset.sum_congr rfl fun c hc => ?_
  have hc' := Finset.mem_range.1 hc
  congr 1
  · rw [foldLoL_eq_sum, List.length_map, Array.length_toList, hch]
    refine Finset.sum_congr rfl fun l hl => ?_
    have hl' := Finset.mem_range.1 hl
    have hlt : l + 2 ^ lv.lanes * c < g.size := by
      rw [hg, pow_add]
      nlinarith
    rw [gv_embedF _ hlt, show l + 2 ^ lv.lanes * c = c * 2 ^ lv.lanes + l by ring]
  · unfold consW
    rw [getElem!_range_map _ hc']
    show _ = FlockLevel3.embed (FlockLevel3.GF128.of _)
    rw [Zc.of_sumF, map_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    show _ = FlockLevel3.embed (fOf (_ * _))
    rw [fOf_mul, map_mul, getElem!_pos (pos.map _) j (by simpa using hj'), Array.getElem_map,
      ← getElem!_pos pos j hj']

theorem consW_size (cols : ℕ) (pos : Array ℕ) (alpha : Array F128) : (consW cols pos alpha).size = 2 ^ cols := by
  simp [consW]

/-- **The consistency value is the folded table's sum** against the consistency weight, the prover's fold of its
F-table at the level's challenges. -/
theorem consE_fold (lv : Level) (g : Array F128) (pos : Array ℕ) (alpha : Array F128) (ch : Array F256)
    (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (hpos : ∀ j < pos.size, pos[j]! < 2 ^ lv.d) :
    gOf (consE (pos.map fun p => (encodeRows lv g)[p]!) pos.size alpha ch) =
      ∑ x ∈ Finset.range (2 ^ lv.cols),
        gv (foldKs ch.toList (embedF g)) x * gv (embedF (consW lv.cols pos alpha)) x := by
  rw [consE_honest lv g pos alpha ch hch hg hpos]
  refine Finset.sum_congr rfl fun x hx => ?_
  have hx' := Finset.mem_range.1 hx
  rw [gv_foldKs _ _ _ (by rw [embedF_size, hg, Array.length_toList, hch, pow_add, Nat.mul_div_cancel_left _
      (Nat.two_pow_pos _)]; exact hx'), gv_embedF _ (by rw [consW_size]; exact hx')]

/-- **A level's consistency claim**, on the split of the prover's fold `f` of the level's F-table at its challenges:
the claim gains `b·consE`, the weight `b·u^bit·V`. -/
theorem Sync.close {s : F256} {q : Poly} {w : Array F256} {n : ℕ} (lv : Level) (g : Array F128) (pos : Array ℕ)
    (alpha : Array F128) (ch : Array F256) (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols))
    (hpos : ∀ j < pos.size, pos[j]! < 2 ^ lv.d) (hn : lv.cols = n)
    (h : Sync s q (embedF (splitK (foldKs ch.toList (embedF g)))) w n) (b : F128) :
    Sync (s + (consE (pos.map fun p => (encodeRows lv g)[p]!) pos.size alpha ch).smul b)
      (q.addScaled (Poly.ofMsg (roundPoly (embedF (splitK (foldKs ch.toList (embedF g))))
        (liftU (embedF (consW lv.cols pos alpha)))) (consE (pos.map fun p => (encodeRows lv g)[p]!) pos.size alpha ch))
        b)
      (embedF (splitK (foldKs ch.toList (embedF g)))) (addScaled w (liftU (embedF (consW lv.cols pos alpha))) b) n := by
  have hf : (foldKs ch.toList (embedF g)).size = 2 ^ n := by
    rw [foldKs_size, embedF_size, hg, Array.length_toList, hch, pow_add, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _),
      hn]
  have hv : (embedF (consW lv.cols pos alpha)).size = 2 ^ n := by rw [embedF_size, consW_size, hn]
  refine h.add _ _ ?_ b
  rw [sum_split _ _ n hf hv, consE_fold lv g pos alpha ch hch hg hpos, hn]

end Lg

end FlockComplete.Replay
