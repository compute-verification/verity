import Definitions.Pouw.Assumptions.Dither

/-!
# SAK(matmul) from a pointwise work premise: what is proved, and the residual

`Definitions.Pouw.Assumptions.Dither.SAKMatmul` bounds a words program's moment
`E_σ[1_{d ∉ Deg ∧ pass} · e^{η (pub − cost)₊}] ≤ 1`. It mixes two things: what a program must spend to get a fraction of
the round's words right (an algorithmic statement about exact matrix products in 𝓜), and how the round's check and its
rate turn a fraction of wrong words into a probability of failing (sampling, which is proved here).

This file proves the sampling half for the no-fold round and states the algorithmic half as one pointwise premise,
`SAKResidual`: outside a set of the round's dithers of probability at most `κdeg B`, a program that exchanges no bits
and reads at most `B` bits of advice costs at least the fraction of the words it gets right times `pub`, less a luck
allowance `lam`. Then:

* `wordsMoment_le_one`: a moment is at most 1 once, at every dither outside `Deg`, the check's pass probability times
  `e^{η · shortfall}` is at most 1 (averaging over the dither);
* `Detects`, `detects_iid`: a check of `s` cells drawn independently and uniformly passes words that are right on a set
  `S` of cells with probability `(|S|/N)^s`;
* `rate_iid`, `rate_iid_sharp`: at that check, `(1 − f)^s e^{x f} ≤ 1` for every fraction `f` wrong exactly when
  `x ≤ s`, `x = η · pub`;
* `rate_bern`, `rate_bern_sharp`: for a round checked with probability `π` (and its `s` cells then), `x ≤ π` suffices,
  and past `−log(1 − π)` a program that skips the whole round wins;
* `sakMatmul_of_residual`, `sakMatmul_of_residual_iid`: `SAKResidual` with a check that detects at `g` and a rate
  with `g(1 − f) e^{η f pub} ≤ 1` gives `SAKMatmul` at `pub − lam`.

So the trust that `SAKMatmul` asks of programs is `SAKResidual`'s, at the cost of the luck allowance `lam` in the
headline's slack, and the rate condition is a theorem about the check. Nothing here is a new premise of a guarantee:
`SAKResidual` is a proof-only definition, a hypothesis of these lemmas.
-/

namespace Pouw.SakLb

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

section Averaging

theorem prob_eq_card_subtype {Ω : Type*} [Fintype Ω] (p : Ω → Prop) [Fintype {x // p x}] :
    prob p = (Fintype.card {x // p x} : ℝ) / Fintype.card Ω := by
  classical
  unfold prob
  congr 2
  rw [Fintype.card_subtype]

theorem prob_nonneg {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : 0 ≤ prob p := by
  unfold prob; positivity

theorem prob_le_one {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : prob p ≤ 1 := by
  classical
  unfold prob
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast card_filter_le _ _

open Classical in
/-- **The moment from the dither's view**: if at every dither `d` outside `Deg` the check passes the program's words
with probability `p_d` and `p_d · e^{η · shortfall_d} ≤ 1`, the moment is at most 1. -/
theorem wordsMoment_le_one (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pub η : ℝ) (a : P.Inst) {St : Type}
    (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Deg : P.Draw → Prop)
    (h : ∀ d, ¬ Deg d → prob (fun c => P.ok a d c (G.Mc.eval Mw (s, d))) *
      Real.exp (η * roundShortfall pub (G.cost μ Mw (s, d))) ≤ 1) :
    wordsMoment G P μ pub η a s Mw Deg ≤ 1 := by
  unfold wordsMoment avg
  apply div_le_one_of_le₀ _ (by positivity)
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul]
  calc _ ≤ ∑ _d : P.Draw, (Fintype.card P.Chk : ℝ) := Finset.sum_le_sum fun d _ => ?_
    _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  by_cases hd : Deg d
  · simp [hd]
  · have hp := h d hd
    unfold prob at hp
    simp only [hd, not_false_eq_true, true_and]
    rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    rcases Nat.eq_zero_or_pos (Fintype.card P.Chk) with hc | hc
    · have : (univ.filter fun c => P.ok a d c (G.Mc.eval Mw (s, d))).card = 0 := by
        apply Nat.eq_zero_of_le_zero
        calc _ ≤ (univ : Finset P.Chk).card := card_filter_le _ _
          _ = 0 := by rw [Finset.card_univ, hc]
      simp [this]
    · have hc' : (0 : ℝ) < Fintype.card P.Chk := by exact_mod_cast hc
      rw [div_mul_eq_mul_div, div_le_one hc'] at hp
      convert hp using 3

end Averaging

section Cells

variable (k : NoFold)

/-- The no-fold round's cells: a row, a column and a segment. -/
abbrev Cell : Type := Fin k.n × Fin k.n' × Fin k.nb

/-- The cells where the words `T` are the noised product's at the dither `d`. -/
def right (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr) : Finset (Cell k) :=
  univ.filter fun x => T x.1 x.2.1 x.2.2 = noFoldWord k (k.noise a d) x.1 x.2.1 x.2.2

/-- **The fraction of the words that are right**: `|right| / (n n' nb)`. -/
noncomputable def rightFrac (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr) : ℝ :=
  ((right k a d T).card : ℝ) / Fintype.card (Cell k)

theorem rightFrac_nonneg (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr) :
    0 ≤ rightFrac k a d T := by
  unfold rightFrac; positivity

theorem rightFrac_le_one (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr) :
    rightFrac k a d T ≤ 1 := by
  unfold rightFrac
  rcases Nat.eq_zero_or_pos (Fintype.card (Cell k)) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast card_le_univ _

/-- The honest words are right everywhere. -/
theorem rightFrac_honest (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (hN : 0 < Fintype.card (Cell k)) :
    rightFrac k a d (noFoldHonest k a d) = 1 := by
  have h : right k a d (noFoldHonest k a d) = univ := by
    ext x; simp [right, noFoldHonest]
  rw [rightFrac, h, card_univ]
  exact div_self (Nat.cast_pos.mpr hN).ne'

/-- **The check detects at `g`**: for every set `S` of cells, the check's coin selects only cells of `S` with
probability at most `g (|S| / N)`. -/
def Detects (g : ℝ → ℝ) : Prop :=
  ∀ S : Finset (Cell k), prob (fun c : k.Chk => ∀ q, k.cells c q ∈ S) ≤ g ((S.card : ℝ) / Fintype.card (Cell k))

/-- **A check of `s` cells drawn independently and uniformly** (the check's coin is the `s` cells) detects at `x^s`. -/
theorem detects_iid (e : k.Chk ≃ (Fin k.s → Cell k)) (he : ∀ c q, k.cells c q = e c q) :
    Detects k fun x => x ^ k.s := by
  classical
  intro S
  rw [prob_eq_card_subtype]
  have h1 : Fintype.card {c : k.Chk // ∀ q, k.cells c q ∈ S} = S.card ^ k.s := by
    rw [Fintype.card_congr (e.subtypeEquiv (q := fun f => ∀ q, f q ∈ S) (fun c => by simp [he])),
      Fintype.card_congr Equiv.subtypePiEquivPi]
    simp
  have h2 : Fintype.card k.Chk = Fintype.card (Cell k) ^ k.s := by
    rw [Fintype.card_congr e]; simp
  rw [h1, h2]; push_cast; rw [div_pow]

end Cells

section Rate

/-- **The rate for an iid check**: `x ≤ s` gives `(1 − f)^s e^{x f} ≤ 1` at every fraction `f` of wrong words. -/
theorem rate_iid (s : ℕ) (x : ℝ) (hx : x ≤ s) (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) :
    (1 - f) ^ s * Real.exp (x * f) ≤ 1 := by
  have h1 : (1 - f) ^ s ≤ Real.exp (-f) ^ s :=
    pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-f)]) s
  rw [← Real.exp_nat_mul] at h1
  calc (1 - f) ^ s * Real.exp (x * f) ≤ Real.exp (s * -f) * Real.exp (x * f) :=
        mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
    _ = Real.exp (f * (x - s)) := by rw [← Real.exp_add]; ring_nf
    _ ≤ Real.exp 0 := Real.exp_le_exp.mpr (mul_nonpos_of_nonneg_of_nonpos hf0 (by linarith))
    _ = 1 := Real.exp_zero

/-- **`x ≤ s` is sharp for an iid check**: past it, a program that skips a small fraction of the words has moment
above 1 at a rate `x = η · pub`. -/
theorem rate_iid_sharp (s : ℕ) (hs : 1 ≤ s) (x : ℝ) (hx : (s : ℝ) < x) :
    ∃ f : ℝ, 0 < f ∧ f ≤ 1 ∧ 1 < (1 - f) ^ s * Real.exp (x * f) := by
  have hs' : (1 : ℝ) ≤ s := by exact_mod_cast hs
  have hx0 : 0 < x := by linarith
  set f := (x - s) / (2 * s * x) with hf
  have hf0 : 0 < f := by rw [hf]; apply div_pos <;> nlinarith
  have hsf : s * f = (x - s) / (2 * x) := by rw [hf]; field_simp
  have hsf1 : s * f ≤ 1 / 2 := by
    rw [hsf, div_le_div_iff₀ (by positivity) (by norm_num)]; nlinarith
  have hf1 : f ≤ 1 := by nlinarith
  refine ⟨f, hf0, hf1, ?_⟩
  have hb : 1 - s * f ≤ (1 - f) ^ s := by
    have := one_add_mul_le_pow (a := -f) (by linarith) s
    simpa [sub_eq_add_neg, mul_neg] using this
  have he : 1 + x * f ≤ Real.exp (x * f) := by linarith [Real.add_one_le_exp (x * f)]
  have hxf : x * f * (1 - s * f) - s * f > 0 := by
    have : x * f * (1 - s * f) - s * f = f * ((x - s) - x * (s * f)) := by ring
    rw [this, hsf]
    have : x * ((x - s) / (2 * x)) = (x - s) / 2 := by field_simp
    rw [this]; apply mul_pos hf0; linarith
  calc 1 < (1 - s * f) * (1 + x * f) := by nlinarith
    _ ≤ (1 - f) ^ s * Real.exp (x * f) :=
        mul_le_mul hb he (by nlinarith) (by positivity)

/-- **The rate for a round checked with probability `π`**: a round whose check, selected with probability `π`, then
draws `s ≥ 1` cells independently detects at `1 − π + π x^s`, and `x ≤ π` gives moment at most 1 at every
fraction `f` of wrong words. -/
theorem rate_bern (s : ℕ) (hs : 1 ≤ s) (π x : ℝ) (hπ0 : 0 ≤ π) (hx : x ≤ π) (f : ℝ) (hf0 : 0 ≤ f)
    (hf1 : f ≤ 1) : (1 - π + π * (1 - f) ^ s) * Real.exp (x * f) ≤ 1 := by
  have h1 : (1 - f) ^ s ≤ 1 - f := pow_le_of_le_one (by linarith) (by linarith) (by omega)
  have h2 : 1 - π + π * (1 - f) ^ s ≤ Real.exp (-(π * f)) := by
    nlinarith [Real.add_one_le_exp (-(π * f)), mul_le_mul_of_nonneg_left h1 hπ0]
  calc (1 - π + π * (1 - f) ^ s) * Real.exp (x * f) ≤ Real.exp (-(π * f)) * Real.exp (x * f) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = Real.exp (f * (x - π)) := by rw [← Real.exp_add]; ring_nf
    _ ≤ Real.exp 0 := Real.exp_le_exp.mpr (mul_nonpos_of_nonneg_of_nonpos hf0 (by linarith))
    _ = 1 := Real.exp_zero

/-- **Skipping the whole round** wins once `x > −log(1 − π)`: a round checked with probability `π < 1` passes a
program with no word right with probability `1 − π`, and its moment `(1 − π) e^x` exceeds 1. So for `s ≥ 2` the
binding necessary condition is about `π`, not `π s`. -/
theorem rate_bern_sharp (s : ℕ) (hs : 1 ≤ s) (π x : ℝ) (hπ1 : π < 1) (hx : -Real.log (1 - π) < x) :
    1 < (1 - π + π * (1 - 1) ^ s) * Real.exp (x * 1) := by
  have h0 : (0 : ℝ) < 1 - π := by linarith
  rw [sub_self, zero_pow (by omega), mul_zero, add_zero, mul_one]
  have : Real.exp (-Real.log (1 - π)) = (1 - π)⁻¹ := by rw [Real.exp_neg, Real.exp_log h0]
  calc 1 = (1 - π) * Real.exp (-Real.log (1 - π)) := by rw [this, mul_inv_cancel₀ h0.ne']
    _ < (1 - π) * Real.exp x := mul_lt_mul_of_pos_left (Real.exp_lt_exp.mpr hx) h0

end Rate

section Residual

/-- **SAK's residual** (proposed, not a premise of any guarantee): in 𝓜 at its device premises, on every registrable
instance, every words program of the no-fold round that exchanges no bits with outside machines, on any advice of at
most `B` bits fixed before the round's coins, costs at least the fraction of the words it gets right times `pub`, less
the luck allowance `lam`, at every dither outside a set `Deg` of probability at most `κdeg B`:
`rightFrac · pub − lam ≤ cost_μ`.

In plain words: no program gets a fraction `φ` of a sampling block's noised prefix sums right for less than `φ` of
the public best's priced cost, but for `lam` (the words a program can guess) and on all but `κdeg B` of the dithers.
It speaks of every program: bilinear schemes (what `Proofs.Core.Bilinear` bounds), table lookups, advice and
precomputation. It is falsified by any program on the card, with any advice, that gets a fraction `φ` of the words
right on most dithers at a measured priced cost below `φ · pub − lam`. -/
def SAKResidual (G : Accel) (k : NoFold) (μ : G.Res → ℝ) (Reg : Set (noFoldRound k).Inst)
    (pub lam : (noFoldRound k).Inst → ℝ) (κdeg : ℕ → ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ a ∈ Reg, ∀ (B : ℕ) (St : Type), (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    ∀ (s : St) (Mw : G.Mc.Alg (St × (noFoldRound k).Draw) (noFoldRound k).Tr), (∀ i, G.ext Mw i = 0) →
      ∃ Deg : (noFoldRound k).Draw → Prop, prob Deg ≤ κdeg B ∧
        ∀ d, ¬ Deg d → rightFrac k a d (G.Mc.eval Mw (s, d)) * pub a - lam a ≤ G.cost μ Mw (s, d)

/-- The no-fold check passes exactly where every checked cell is right. -/
theorem ok_iff_mem_right (k : NoFold) (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (c : k.Chk)
    (T : (noFoldRound k).Tr) : (noFoldRound k).ok a d c T ↔ ∀ q, k.cells c q ∈ right k a d T := by
  simp [noFoldRound, right]

/-- **One dither**: a program right on a fraction `r` of the words at cost at least `r · pub − lam`, under a check
that detects at `g` and a rate with `g(1 − f) e^{η f pub} ≤ 1`, has `pass · e^{η · shortfall} ≤ 1` at `pub − lam`. -/
theorem pointwise (k : NoFold) (g : ℝ → ℝ) (hdet : Detects k g) (pub lam η cost : ℝ) (hpub : 0 ≤ pub) (hη : 0 ≤ η)
    (hrate : ∀ f : ℝ, 0 ≤ f → f ≤ 1 → g (1 - f) * Real.exp (η * (f * pub)) ≤ 1)
    (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr)
    (hcost : rightFrac k a d T * pub - lam ≤ cost) :
    prob (fun c => (noFoldRound k).ok a d c T) * Real.exp (η * roundShortfall (pub - lam) cost) ≤ 1 := by
  set r := rightFrac k a d T with hr
  have hr0 := rightFrac_nonneg k a d T
  have hr1 := rightFrac_le_one k a d T
  have hpass : prob (fun c => (noFoldRound k).ok a d c T) ≤ g (1 - (1 - r)) := by
    have hf : (fun c => (noFoldRound k).ok a d c T) = fun c : k.Chk => ∀ q, k.cells c q ∈ right k a d T :=
      funext fun c => propext (ok_iff_mem_right k a d c T)
    rw [hf, sub_sub_cancel]
    exact hdet (right k a d T)
  have hsh : roundShortfall (pub - lam) cost ≤ (1 - r) * pub := by
    unfold roundShortfall
    apply max_le (mul_nonneg (by linarith) hpub)
    linarith
  have hexp : Real.exp (η * roundShortfall (pub - lam) cost) ≤ Real.exp (η * ((1 - r) * pub)) :=
    Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hsh hη)
  have hg := hrate (1 - r) (by linarith) (by linarith)
  have hg0 : 0 ≤ g (1 - (1 - r)) := le_trans (prob_nonneg _) hpass
  calc _ ≤ g (1 - (1 - r)) * Real.exp (η * ((1 - r) * pub)) :=
        mul_le_mul hpass hexp (Real.exp_pos _).le hg0
    _ ≤ 1 := hg

/-- **`SAKMatmul` from its residual**: `SAKResidual` at `pub` and `lam`, a check that detects at `g` and a rate with
`g(1 − f) e^{η f pub} ≤ 1` at every fraction `f` give `SAKMatmul` at `pub − lam`. -/
theorem sakMatmul_of_residual (G : Accel) (k : NoFold) (μ : G.Res → ℝ) (Reg : Set (noFoldRound k).Inst)
    (pub lam η : (noFoldRound k).Inst → ℝ) (κdeg : ℕ → ℝ) (g : ℝ → ℝ)
    (hres : SAKResidual G k μ Reg pub lam κdeg) (hdet : Detects k g)
    (hrate : ∀ a ∈ Reg, ∀ f : ℝ, 0 ≤ f → f ≤ 1 → g (1 - f) * Real.exp (η a * (f * pub a)) ≤ 1)
    (hpub : ∀ a ∈ Reg, 0 ≤ pub a) (hη : ∀ a ∈ Reg, 0 ≤ η a) :
    SAKMatmul G (noFoldRound k) μ Reg (fun a => pub a - lam a) η κdeg := by
  intro g1 g2 p1 p2 a ha B St hSt s Mw hext
  obtain ⟨Deg, hDeg, hcost⟩ := hres g1 g2 p1 p2 a ha B St hSt s Mw hext
  refine ⟨Deg, hDeg, wordsMoment_le_one G _ μ _ _ a s Mw Deg fun d hd => ?_⟩
  exact pointwise k g hdet (pub a) (lam a) (η a) _ (hpub a ha) (hη a ha) (hrate a ha) a d _ (hcost d hd)

/-- **`SAKMatmul` from its residual at an iid check**: with the check's `s` cells drawn independently and uniformly,
the rate `η · pub ≤ s` suffices (and is sharp, `rate_iid_sharp`). -/
theorem sakMatmul_of_residual_iid (G : Accel) (k : NoFold) (μ : G.Res → ℝ) (Reg : Set (noFoldRound k).Inst)
    (pub lam η : (noFoldRound k).Inst → ℝ) (κdeg : ℕ → ℝ)
    (hres : SAKResidual G k μ Reg pub lam κdeg)
    (e : k.Chk ≃ (Fin k.s → Cell k)) (he : ∀ c q, k.cells c q = e c q)
    (hrate : ∀ a ∈ Reg, η a * pub a ≤ k.s) (hpub : ∀ a ∈ Reg, 0 ≤ pub a) (hη : ∀ a ∈ Reg, 0 ≤ η a) :
    SAKMatmul G (noFoldRound k) μ Reg (fun a => pub a - lam a) η κdeg :=
  sakMatmul_of_residual G k μ Reg pub lam η κdeg _ hres (detects_iid k e he)
    (fun a ha f hf0 hf1 => by
      have := rate_iid k.s (η a * pub a) (hrate a ha) f hf0 hf1
      simpa [sub_sub_cancel, mul_comm, mul_left_comm, mul_assoc] using this)
    hpub hη

/-- **Honest**: the honest words meet the residual's inequality at a cost of at least `pub − lam`, so the residual
asks nothing of the honest program past its own cost. -/
theorem residual_honest (k : NoFold) (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (pub lam cost : ℝ)
    (hN : 0 < Fintype.card (Cell k)) (hcost : pub - lam ≤ cost) :
    rightFrac k a d (noFoldHonest k a d) * pub - lam ≤ cost := by
  rw [rightFrac_honest k a d hN, one_mul]; exact hcost

/-- **Nontrivial**: the residual's inequality is no bound on a program that gets nothing right (`rightFrac = 0`
asks only `cost ≥ −lam`), so the moment bound comes from the check: at a rate past the check's (`x > s`), some
fraction of skipped words beats moment 1 (`rate_iid_sharp`), and `pointwise` needs its rate hypothesis. -/
theorem residual_nontrivial (s : ℕ) (hs : 1 ≤ s) :
    ¬ ∀ f : ℝ, 0 ≤ f → f ≤ 1 → (1 - f) ^ s * Real.exp ((s + 1) * f) ≤ 1 := by
  intro h
  obtain ⟨f, hf0, hf1, hlt⟩ := rate_iid_sharp s hs (s + 1) (by linarith)
  linarith [h f hf0.le hf1]

end Residual

end Pouw.SakLb
