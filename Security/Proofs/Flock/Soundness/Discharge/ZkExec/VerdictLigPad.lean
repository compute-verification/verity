import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig
import Proofs.Flock.Soundness.NovelBasis
import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled

/-!
# Level 0's padded column, as `Zk.ligerito` computes it

The model's padded level-0 column (`ZkLink.padColumnO`) at an opened row is three terms: the witness lanes folded by the
lane challenges, rep `r`'s extra pair weighted by `ρ`, and the padding's codeword `Model.padCorr` at `ȳ`. The executable
computes the same three at each query `j` of level 0 (`Zk.ligerito`): the fold over the first `2^lanes` entries in
`Zk.queryLevel0`, then `ρ₁·row[2^lanes + 2r] + ρ₂·row[2^lanes + 2r + 1] + padAt(…)` in its extra loop.

* `padAt_eq`: the executable's `Zk.padAt` is `Model.padCorr` at `c₀ = PAD_C`: its `xs` are the novel-basis monomials
  `X̂_i(ω_p)` (`padFold_inv`, from `X̂_i = X̂_{i − 2^⌊log₂ i⌋}·Ŵ_{⌊log₂ i⌋}`), and `X̂_{2^cols + j} = X̂_j·Ŵ_cols` for
  `j < 2^cols`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine FlockLevel3 Model

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The novel basis -/

theorem Xhat_lift {L c : ℕ} (hc : c < 2 ^ L) (ω : 𝔽) :
    ∀ L', L ≤ L' → execArith.Xhat c L' ω = execArith.Xhat c L ω := by
  intro L' hL
  induction L', hL using Nat.le_induction with
  | base => rfl
  | succ n hn ih =>
      rw [Xhat_succ_low execArith n c (lt_of_lt_of_le hc (Nat.pow_le_pow_right (by norm_num) hn)) ω, ih]

/-- `X̂_i = X̂_{i − 2^⌊log₂ i⌋}·Ŵ_{⌊log₂ i⌋}`, the executable's recursion for the padding's monomials. -/
theorem Xhat_top {L i : ℕ} (h0 : 0 < i) (hi : i < 2 ^ L) (ω : 𝔽) :
    execArith.Xhat i L ω = execArith.Xhat (i - 2 ^ i.log2) L ω * execArith.What i.log2 ω := by
  have hle : 2 ^ i.log2 ≤ i := Nat.log2_self_le (by omega)
  have hlt : i < 2 ^ (i.log2 + 1) := Nat.lt_log2_self
  have htop : i.log2 < L := (Nat.log2_lt (by omega)).2 hi
  have hrest : i - 2 ^ i.log2 < 2 ^ i.log2 := by rw [pow_succ] at hlt; omega
  have e : execArith.Xhat i (i.log2 + 1) ω = execArith.Xhat (2 ^ i.log2 + (i - 2 ^ i.log2)) (i.log2 + 1) ω := by
    rw [Nat.add_sub_cancel' hle]
  rw [Xhat_lift hlt ω L (by omega), e, Xhat_succ_high execArith i.log2 _ hrest ω, Xhat_lift hrest ω L (by omega),
    mul_comm]

theorem Xhat_zero (L : ℕ) (ω : 𝔽) : execArith.Xhat 0 L ω = 1 := by
  unfold Arith.Xhat
  simp

/-! ## The padding's codeword -/

/-- One step of `Zk.padAt`'s loop: the next monomial (after the first), and its term of the sum. -/
def padStep (ws : Array F128) (ybar : Array F256) (s : Array F128 × F256) (i : ℕ) : Array F128 × F256 :=
  if i > 0 then
    (s.1.push (s.1[i - 2 ^ i.log2]! * ws[i.log2]!),
      s.2 + ybar[i]!.smul (s.1.push (s.1[i - 2 ^ i.log2]! * ws[i.log2]!))[i]!)
  else (s.1, s.2 + ybar[i]!.smul s.1[i]!)

theorem padAt_foldl (ws : Array F128) (logMsg : ℕ) (ybar : Array F256) :
    Zk.padAt ws logMsg ybar =
      ((List.range' 0 ybar.size).foldl (padStep ws ybar) (#[1], 0)).2.smul (Zk.PAD_C + ws[logMsg]!) := by
  unfold Zk.padAt
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', range_size]
  rw [show (fun (i : ℕ) (s : Array F128 × F256) =>
      if i > 0 then (pure (ForInStep.yield (s.1.push (s.1[i - 2 ^ i.log2]! * ws[i.log2]!),
        s.2 + ybar[i]!.smul (s.1.push (s.1[i - 2 ^ i.log2]! * ws[i.log2]!))[i]!)) : Id _)
      else pure (ForInStep.yield (s.1, s.2 + ybar[i]!.smul s.1[i]!))) =
      fun i s => pure (ForInStep.yield (padStep ws ybar s i)) by
    funext i s
    unfold padStep
    split <;> rfl]
  rw [List.forIn_pure_yield_eq_foldl]
  rfl

/-- `Zk.padAt`'s loop after `n` steps: its monomials are `X̂_k(ω)` and its sum is `Σ_{k<n} ȳ_k·X̂_k(ω)`, when its
`Ŵ`s are `Ŵ_b(ω)`. -/
theorem padFold_inv (ws : Array F128) (ybar : Array F256) (L : ℕ) (ω : 𝔽)
    (hws : ∀ b < L, fOf ws[b]! = execArith.What b ω) : ∀ n, n ≤ 2 ^ L →
      ((List.range' 0 n).foldl (padStep ws ybar) (#[1], 0)).1.size = max n 1 ∧
      (∀ k < max n 1, fOf ((List.range' 0 n).foldl (padStep ws ybar) (#[1], 0)).1[k]! = execArith.Xhat k L ω) ∧
      gOf ((List.range' 0 n).foldl (padStep ws ybar) (#[1], 0)).2 =
        ∑ k ∈ Finset.range n, gOf ybar[k]! * embed (execArith.Xhat k L ω) := by
  intro n
  induction n with
  | zero =>
      intro _
      refine ⟨rfl, fun k hk => ?_, by simp [gOf_zero]⟩
      obtain rfl : k = 0 := by omega
      rw [Xhat_zero]
      simp [fOf_one]
  | succ n ih =>
      intro hn
      obtain ⟨hsz, hxs, hacc⟩ := ih (by omega)
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
      generalize (List.range' 0 n).foldl (padStep ws ybar) (#[1], 0) = s at hsz hxs hacc
      unfold padStep
      by_cases h0 : n > 0
      · rw [ite_eq_left_of_eq_true _ _ (eq_true h0)]
        have hsz' : s.1.size = n := by rw [hsz]; omega
        have hn2 : n - 2 ^ n.log2 < n := by
          have := Nat.log2_self_le (show n ≠ 0 by omega); have := Nat.one_le_two_pow (n := n.log2); omega
        have hpush : (s.1.push (s.1[n - 2 ^ n.log2]! * ws[n.log2]!))[n]! = s.1[n - 2 ^ n.log2]! * ws[n.log2]! := by
          rw [getElem!_pos _ n (by simp [hsz'])]
          simp [Array.getElem_push, hsz']
        have hlast : fOf (s.1.push (s.1[n - 2 ^ n.log2]! * ws[n.log2]!))[n]! = execArith.Xhat n L ω := by
          rw [hpush, fOf_mul, hxs _ (by omega), hws _ ((Nat.log2_lt (by omega)).2 (by omega)),
            ← Xhat_top h0 (by omega)]
        refine ⟨by simp [hsz'], fun k hk => ?_, ?_⟩
        · rcases Nat.lt_or_ge k n with hk' | hk'
          · have : (s.1.push (s.1[n - 2 ^ n.log2]! * ws[n.log2]!))[k]! = s.1[k]! := by
              rw [getElem!_pos _ k (by simp [hsz']; omega), getElem!_pos _ k (by omega)]
              simp [Array.getElem_push, hsz', hk']
            rw [this]
            exact hxs k (by omega)
          · obtain rfl : k = n := by omega
            exact hlast
        · rw [gOf_add, hacc, gOf_smul, hlast, Finset.sum_range_succ]
      · rw [ite_eq_right_of_eq_false _ _ (eq_false h0)]
        obtain rfl : n = 0 := by omega
        refine ⟨hsz, hxs, ?_⟩
        rw [gOf_add, hacc, gOf_smul, hxs 0 (by omega)]
        simp

/-- **The padding's codeword** at position `p` (live PROTOCOL.md §7): the executable's `Zk.padAt` on level 0's `Ŵ`s at
`ω_p` is `Model.padCorr` at `c₀ = PAD_C`, for `t ≤ 2^cols` and a level of `2^(cols + 1)` positions. -/
theorem padAt_eq (p cols tPad : ℕ) (ybar : Array F256) (hy : ybar.size = tPad) (ht : tPad ≤ 2 ^ cols) :
    gOf (Zk.padAt (whats (svTable (cols + 1)) p (cols + 1)) cols ybar) =
      padCorr execArith (cols + 1) cols tPad (fOf Zk.PAD_C) (fun i => gOf ybar[i.val]!) p := by
  have hws : ∀ b < cols, fOf (whats (svTable (cols + 1)) p (cols + 1))[b]! = execArith.What b (execArith.omega p) :=
    fun b hb => whats_get (cols + 1) p (cols + 1) b le_rfl (by omega)
  obtain ⟨-, -, hacc⟩ := padFold_inv (whats (svTable (cols + 1)) p (cols + 1)) ybar cols (execArith.omega p) hws
    tPad ht
  rw [padAt_foldl, gOf_smul, hy, hacc, fOf_add, whats_get (cols + 1) p (cols + 1) cols le_rfl (by omega)]
  unfold padCorr
  dsimp only
  rw [Fin.sum_univ_eq_sum_range (fun j => gOf ybar[j]! * execArith.embed (fOf Zk.PAD_C * execArith.Xhat j (cols + 1)
    (execArith.omega p) + execArith.Xhat (2 ^ cols + j) (cols + 1) (execArith.omega p))) tPad]
  refine (Finset.sum_mul (Finset.range tPad) (fun k => gOf ybar[k]! * embed (execArith.Xhat k cols (execArith.omega p)))
    (embed (fOf Zk.PAD_C + What cols (omega p)))).trans (Finset.sum_congr rfl fun j hj => ?_)
  rw [Finset.mem_range] at hj
  rw [Xhat_succ_low execArith cols j (by omega), Xhat_succ_high execArith cols j (by omega), mul_assoc]
  show _ * (embed _ * embed _) = _ * embed _
  rw [← map_mul]
  congr 2
  show _ = _ * _ + FlockLevel3.What cols (FlockLevel3.omega p) * _
  ring

/-! ## Level 0's consistency value -/

/-- Level 0's consistency value as `Zk.queryLevel0` folds it: query `j`'s row, its first `W` entries folded by
`eq_K(fold)`, weighted by `eq(α)[j]`. -/
def consE0 (rows : Array (Array F128)) (q W : ℕ) (alpha : Array F128) (fold : Array F256) : F256 :=
  (List.range q).foldl (fun e j =>
    e + F256.smul ((List.range W).foldl
      (fun acc l => acc + F256.smul (eqTableK fold)[l]! (rows[j]!)[l]!) 0) (eqTable alpha)[j]!) 0

/-- The extra terms `Zk.ligerito` adds to level 0's consistency value: query `j`'s rep-`r` pair weighted by `ρ` and
the padding's codeword at its position, weighted by `eq(α)[j]`. -/
def extraE (rows : Array (Array F128)) (pos : Array ℕ) (W rep cols : ℕ) (rho : F256 × F256) (alpha : Array F128)
    (ybar : Array F256) : F256 :=
  (List.range' 0 pos.size).foldl (fun extra j => extra +
    (rho.1.smul (rows[j]!)[W + 2 * rep]! + rho.2.smul (rows[j]!)[W + 2 * rep + 1]! +
      Zk.padAt (whats (svTable (cols + 1)) pos[j]! (cols + 1)) cols ybar).smul (eqTable alpha)[j]!) 0

/-- **Level 0's padded consistency value** (live PROTOCOL.md §7): the model's batched padded column, at openings
whose rows are the proof's (`hR`, rows `2^k₀ + 4` wide), is the executable's fold of the first `2^k₀` entries plus its
extra terms, at `ρ` the executable's pair, `c₀ = PAD_C` and a level of `2^(cols + 1)` positions. -/
theorem consPad_eq (rows : Array (Array F128)) (R : ℕ → List 𝔽) (pos : Array ℕ) (alpha : Array F128)
    (fold : Array F256) (k0 rep cols tPad : ℕ) (rho : F256 × F256) (ρ : Fin 2 → 𝕂) (ybar : Array F256)
    (hk : fold.size = k0) (hrows : ∀ j < pos.size, rows[j]!.size = 2 ^ k0 + 4)
    (hR : ∀ j < pos.size, R j = rows[j]!.toList.map fOf) (halpha : pos.size ≤ 2 ^ alpha.size) (hrep : rep < 2)
    (hρ0 : ρ 0 = gOf rho.1) (hρ1 : ρ 1 = gOf rho.2) (hy : ybar.size = tPad) (ht : tPad ≤ 2 ^ cols) :
    ((pos.toList.zipIdx).map fun pj => execArith.embed (eqList (alpha.toList.map fOf) pj.2) *
        ZkLink.padColumnO execArith (cols + 1) cols tPad (fOf Zk.PAD_C) (fold.toList.map gOf) ρ
          (extraLane k0 2 rep) (fun i => gOf ybar[i.val]!) (R pj.2) pj.1).sum =
      gOf (consE0 rows pos.size (2 ^ k0) alpha fold + extraE rows pos (2 ^ k0) rep cols rho alpha ybar) := by
  have hz : (pos.toList.zipIdx).map (fun pj => execArith.embed (eqList (alpha.toList.map fOf) pj.2) *
      ZkLink.padColumnO execArith (cols + 1) cols tPad (fOf Zk.PAD_C) (fold.toList.map gOf) ρ
        (extraLane k0 2 rep) (fun i => gOf ybar[i.val]!) (R pj.2) pj.1) =
      (List.range pos.size).map fun j => execArith.embed (eqList (alpha.toList.map fOf) j) *
        ZkLink.padColumnO execArith (cols + 1) cols tPad (fOf Zk.PAD_C) (fold.toList.map gOf) ρ
          (extraLane k0 2 rep) (fun i => gOf ybar[i.val]!) (R j) pos[j]! := by
    refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
    simp only [List.getElem_map, List.getElem_zipIdx, List.getElem_range, Nat.zero_add, Array.getElem_toList]
    rw [getElem!_pos pos i (by simpa using h₂)]
  unfold consE0 extraE
  rw [← List.range_eq_range', hz, FlockLevel3.list_sum_range, gOf_add, gOf_foldl_cons, gOf_foldl_cons, gOf_zero, zero_add, zero_add,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j hj => ?_
  rw [Finset.mem_range] at hj
  have hsz := hrows j hj
  rw [hR j hj, ← eqTable_eqList alpha j (by omega), gOf_foldl_row, gOf_zero, zero_add, gOf_add, gOf_add, gOf_smul,
    gOf_smul, padAt_eq pos[j]! cols tPad ybar hy ht]
  unfold ZkLink.padColumnO foldColumn
  rw [Fin.sum_univ_two, hρ0, hρ1, List.length_map, Array.length_toList, hk]
  have hl : ∀ l ∈ Finset.range (2 ^ k0), eqList (fold.toList.map gOf) l *
      execArith.embed ((rows[j]!.toList.map fOf).getD l 0) =
      gOf (eqTableK fold)[l]! * execArith.embed (fOf (rows[j]!)[l]!) := fun l hl' => by
    rw [Finset.mem_range] at hl'
    rw [eqTableK_eqList fold l (by rw [hk]; exact hl'), getD_map_fOf _ l (by omega)]
  rw [Finset.sum_congr rfl hl]
  have e0 : (extraLane k0 2 rep 0) = 2 ^ k0 + 2 * rep := rfl
  have e1 : (extraLane k0 2 rep 1) = 2 ^ k0 + 2 * rep + 1 := rfl
  rw [e0, e1, getD_map_fOf _ _ (by omega), getD_map_fOf _ _ (by omega)]
  simp only [show ∀ x, FlockLevel3.embed x = execArith.embed x from fun _ => rfl]
  ring

end FlockSoundness.Discharge.ZkExec
