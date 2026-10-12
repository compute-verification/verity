import Proofs.Pous.P2.Rounds
import Proofs.Pous.P2.RingTrunc

/-!
# Falsification tests of `P2RingRoundSecure`, at every block count from `2^19` to `2^25`

`RingTight`'s and `RingTrunc`'s responders, each at round `0` of a one-challenge audit at block count `N`:

* recomputation (`recSeq`, `RingTight`'s responder: no state, both roots recomputed) is always right, so the
  per-round floor is `p = 0` unless the bad event may take all the mass (`round_tight`, at `D ≥ 32,894`, `Q ≥ 2`,
  `L ≥ 65,789`);
* keeping `m` blocks without their low `t` bits within `S₀` passes round `0` with probability at least `m/N`, so
  `m/N ≤ 1 − p + η N 1` (`round_trunc`, at `D ≥ 2`, `Q ≥ 2`, `L ≥ 2^(t+1)`).

Both hold at every `N`, as `tight_p16448` does; `not_meetsOn_b19_k88`'s numbers are `2^19`'s, and here `m` and `t` are
hypotheses, with `m·(16448 − t) ≤ S₀` at the `N` in question.
-/

namespace PousP2Rounds

open Pous Pous.P2.Assumptions PousP2Ring PousP2p PousThm1
open scoped ENNReal

section Tight

variable {B w p b : ℕ} [NeZero B] [NeZero p]

/-- `RingTight`'s responder: keep the state, recompute the challenged block from the payload. -/
noncomputable def recSeq (B w p b : ℕ) [NeZero B] [NeZero p] (n D Q L : ℕ) (hD : 2 * (n + 1) ≤ D) (hQ : 2 ≤ Q)
    (hL : 4 * n + 5 ≤ L) : SeqResponderOn (p2ring B w p b) ((p2ring B w p b).stateBound Params.ρ) D Q L :=
  ⟨fun W _ _ σ i => progMap (fun c => (c, σ)) (recBlock n ((p + 1) / 4) (pad B w b W i) i),
    fun _ _ _ _ _ => ringBounded_progMap _ (ringBounded_mono (recBlock_bounded _ _ _ _) hD hQ hL)⟩

theorem recSeq_accepts (hp : p.Prime) (h4 : p % 4 = 3) (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w) (n : ℕ)
    (hn : (p + 1) / 4 < 2 ^ n) {D Q L k : ℕ} (hD : 2 * (n + 1) ≤ D) (hQ : 2 ≤ Q) (hL : 4 * n + 5 ≤ L)
    (x : Outcome (p2ring B w p b) k) :
    seqAcceptsOn (p2ring B w p b) k (fun _ => false) (fun _ _ _ _ _ => false) (recSeq B w p b n D Q L hD hQ hL) x := by
  show seqAnswers _ (fun _ σ i => progMap (fun c => (c, σ))
    (recBlock n ((p + 1) / 4) (pad B w b (fun _ => false) i) i)) 0 _ _ = _
  refine (seqAnswers_keep _ (fun _ i => recBlock n ((p + 1) / 4) (pad B w b (fun _ => false) i) i) 0 _ _).trans ?_
  refine List.map_ofFn.trans ?_
  exact congrArg List.ofFn (funext fun j => recBlock_run hp h4 hp2 n hn x.1 _ (pad_lt hb hp2 _ _))

end Tight

/-- **Recomputation refutes every positive per-round floor**, at every block count from `2^19` to `2^25`: if `p` is
prime, `P2RingRoundSecure p η D Q L` at `D ≥ 32,894`, `Q ≥ 2` and `L ≥ 65,789` forces `p = 0` at any `N` where
`η N 1 < 1`. -/
theorem round_tight (hP : PrimeP16448) (p : ℝ≥0∞) (η : ℕ → ℕ → ℝ≥0∞) (D Q L : ℕ) (hD : 32894 ≤ D) (hQ : 2 ≤ Q)
    (hL : 65789 ≤ L) (h : P2RingRoundSecure p η D Q L) (N : ℕ) [NeZero N] (h1 : 2 ^ 19 ≤ N) (h2 : N ≤ 2 ^ 25)
    (hη : η N 1 < 1) : p = 0 := by
  let A₂ := recSeq N 16448 (2 ^ 16448 - 21065) 16432 16446 D Q L (by omega) hQ (by omega)
  obtain ⟨bad, hbad, hstep⟩ := h N h1 h2 _ 1 (fun _ => false) (fun _ _ _ _ _ => false) (ofSeq A₂)
  have hs := hstep 0 one_pos le_rfl
  have hall : ∀ x, varPassesOn _ 1 1 (fun _ => false) (fun _ _ _ _ _ => false) (ofSeq A₂) x := fun x =>
    (varPassesOn_ofSeq_iff _ _ _ A₂ x).2 (recSeq_accepts hP p16448_mod_four
      (two_pow_le_of_le (by norm_num) hyps_16448.1) hyps_16448.2.1 16446 p16448_root_bits _ _ _ x)
  have e1 : pr (fun x => varPassesOn _ 1 (0 + 1) (fun _ => false) (fun _ _ _ _ _ => false) (ofSeq A₂) x ∧ ¬ bad x) =
      pr (fun x => ¬ bad x) := congrArg pr (funext fun x => propext ⟨fun hx => hx.2, fun hx => ⟨hall x, hx⟩⟩)
  have e0 : pr (fun x => varPassesOn _ 1 0 (fun _ => false) (fun _ _ _ _ _ => false) (ofSeq A₂) x ∧ ¬ bad x) =
      pr (fun x => ¬ bad x) :=
    congrArg pr (funext fun x => propext ⟨fun hx => hx.2, fun hx => ⟨varPassesOn_zero _ _ _ _ x, hx⟩⟩)
  rw [e1, e0] at hs
  have hpos : pr (fun x => ¬ bad x) ≠ 0 := by
    intro h0
    have hc := pr_le_and_not_add (fun _ => True) bad
    rw [pr_eq_one fun _ => trivial] at hc
    simp only [true_and] at hc
    rw [h0, zero_add] at hc
    exact absurd (hc.trans hbad) (not_le.2 hη)
  have hfin : pr (fun x => ¬ bad x) ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (pr_le_one' _)
  by_contra hp
  have hlt : (1 - p) * pr (fun x => ¬ bad x) < pr (fun x => ¬ bad x) := by
    conv_rhs => rw [← one_mul (pr fun x => ¬ bad x)]
    exact ENNReal.mul_lt_mul_left hpos hfin (ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hp)
  exact absurd hs (not_le.2 hlt)

/-- **Truncation bounds the per-round floor**, at every block count from `2^19` to `2^25`: if `p` is prime, keeping
`m ≤ N` blocks without their low `t` bits within `S₀` gives `m/N ≤ 1 − p + η N 1` under
`P2RingRoundSecure p η D Q L` at `D ≥ 2`, `Q ≥ 2` and `L ≥ 2^(t+1)`. At `2^19`, `t = 166` and `m = 501,757` this
refutes `p = 0.043` with `η ≤ 2^-130`. -/
theorem round_trunc (hP : PrimeP16448) (p : ℝ≥0∞) (η : ℕ → ℕ → ℝ≥0∞) (D Q L t : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q)
    (ht : t ≤ 16448) (hL : 2 ^ (t + 1) ≤ L) (h : P2RingRoundSecure p η D Q L) (N : ℕ) [NeZero N]
    (h1 : 2 ^ 19 ≤ N) (h2 : N ≤ 2 ^ 25) (m : ℕ) (hm : m ≤ N)
    (hS : m * (16448 - t) ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) :
    (m : ℝ≥0∞) / N ≤ (1 - p) + η N 1 := by
  let A₂ : SeqResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432)
      ((p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) D Q L :=
    ⟨(truncSeq (b := 16432) t hS).step, fun W pp j σ i =>
      ringBounded_mono ((truncSeq (b := 16432) t hS).bounded W pp j σ i) hD hQ hL⟩
  have hacc := trunc_seq_accepted (b := 16432) hP p16448_mod_four hyps_16448.2.1 1 ht hm hS
  obtain ⟨bad, hbad, hstep⟩ := h N h1 h2 _ 1 (fun _ => false) (truncPre t hm) (ofSeq A₂)
  have hs := hstep 0 one_pos le_rfl
  calc (m : ℝ≥0∞) / N = ((m : ℝ≥0∞) / N) ^ 1 := (pow_one _).symm
    _ ≤ pr (seqAcceptsOn _ 1 (fun _ => false) (truncPre t hm) A₂) := hacc
    _ = pr (varPassesOn _ 1 1 (fun _ => false) (truncPre t hm) (ofSeq A₂)) :=
        congrArg pr (funext fun x => propext (varPassesOn_ofSeq_iff _ _ _ A₂ x).symm)
    _ ≤ pr (fun x => varPassesOn _ 1 (0 + 1) (fun _ => false) (truncPre t hm) (ofSeq A₂) x ∧ ¬ bad x) + pr bad :=
        pr_le_and_not_add _ _
    _ ≤ (1 - p) * pr (fun x => varPassesOn _ 1 0 (fun _ => false) (truncPre t hm) (ofSeq A₂) x ∧ ¬ bad x) +
          η N 1 := add_le_add hs hbad
    _ ≤ (1 - p) * 1 + η N 1 := add_le_add (mul_le_mul' le_rfl (pr_le_one' _)) le_rfl
    _ = _ := by rw [mul_one]

/-- **At `2^19` blocks, truncation refutes `p = 0.043` from `L = 2^167`**: keeping `501,757` blocks without their low
`166` bits (`16,282` bits a block, within `S₀ = 8,169,621,180`) passes a round with probability
`501757/2^19 ≈ 0.95703 > 0.957 + 2^-130`. So `P2RingRoundSecure (43/1000) η D Q L` is false at `D ≥ 2`, `Q ≥ 2`,
`L ≥ 2^167` whenever `η (2^19) 1 ≤ 2^-130`, if `p` is prime. -/
theorem round_trunc_b19 (hP : PrimeP16448) (η : ℕ → ℕ → ℝ≥0∞) (D Q L : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q)
    (hL : 2 ^ 167 ≤ L) (hη : η (2 ^ 19) 1 ≤ (2⁻¹ : ℝ≥0∞) ^ 130) : ¬ P2RingRoundSecure (43 / 1000) η D Q L := by
  intro h
  have hS : 501757 * (16448 - 166) ≤ (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ := by
    rw [stateBound_p2ring_b19]
    norm_num
  have hb := (round_trunc hP _ η D Q L 166 hD hQ (by norm_num) hL h (2 ^ 19) le_rfl (by norm_num) 501757
    (by norm_num) hS).trans (add_le_add le_rfl hη)
  have hm : ((501757 : ℕ) : ℝ≥0∞) / ((2 ^ 19 : ℕ) : ℝ≥0∞) = ENNReal.ofReal (501757 / 2 ^ 19) := by
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast (2 ^ 19), ← ENNReal.ofReal_div_of_pos (by norm_num)]
    norm_num
  have hr : (1 - 43 / 1000 : ℝ≥0∞) + (2⁻¹ : ℝ≥0∞) ^ 130 = ENNReal.ofReal (957 / 1000 + (2⁻¹ : ℝ) ^ 130) := by
    rw [one_sub_043, ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]
  rw [hm, hr, ENNReal.ofReal_le_ofReal_iff (by positivity)] at hb
  norm_num at hb

end PousP2Rounds
