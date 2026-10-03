import FlockSoundness.Discharge.ZkExec.RunsLig
import FlockSoundness.NovelBasis
import FlockSoundness.Discharge.ZkLink.Compiled

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

end FlockSoundness.Discharge.ZkExec
