import Proofs.Pous.P2.Rows

/-!
# The abstract model `m1p` on `[0, p)`: `Meets` for `Pous.m1p`, at any payload width `b`

P2 permutes `Z_p` with `p = 2^w − c₀`, not `Bits w`. The abstract model `Pous.m1p B w p b`
(`Security/Definitions/Pous/Model/M1p.lean`) draws `B` independent uniform permutations of `Fin p`; they act on
`w`-bit strings through the numeric value and fix the tail `[p, 2^w)`. Each block carries `b` payload bits,
zero-extended to `w` bits. These are theorems about `m1p` at P2's parameters, not about the concrete P2's
square–mask–square maps. This file proves, with no hypothesis beyond what the rows discharge:

* `PousP2p.p2_meets_m1p_b19`: `2^19` blocks at `w = 16,448`, `p = 2^16448 − 21065`, `Q = 2^20`, `k = 88`, for every
  `b` with `20·16448 ≤ 21·b` and `b ≤ 16447` (instances `b = 16447` and `b = 16432`, 2,054 bytes);
* `PousP2p.family`: every `2^8 ≤ B ≤ 2^23` and every such `b`, at the same `w`, `p`, `Q`, `k`;
* `PousP2p.p2_meets_m1p_8192`: `2^24` blocks at `w = 8,192`, `p = 2^8192 − 9345`, `Q = 2^20`, `k = 90`, for every
  `b` with `20·8192 ≤ 21·b` and `b ≤ 8191` (instances `8191` and `8176`);
* `PousP2p.not_meets_m1p_b19_k85`: raw storage breaks `k = 85` at `2^19` blocks, for every `p` and `b`.

The route:

1. `PousGen.count_main`: A2's lazy-sampling count over permutations of any finite type `X`, giving
   `|X|^g` in place of `2^(ℓ·g)`.
2. `PousP2p.lemmaA_m1p`: Lemma A for `m1p` at `X = Fin p`, with bound `gammaP = 2^S·C(B, g)·Σ_{h ≤ g}
   C((g+1)Q, h) / p^g`, whenever `2^b ≤ p ≤ 2^w`. A query in the tail is answered by itself (`trP`), and a
   responder's output is read mod `p`. The bound does not depend on `b`.
3. `PousP2p.gammaP_le`: `2^(w·g) ≤ 2·p^g` when `p ≥ 2^w − c` and `2·B·c ≤ 2^w` (Bernoulli), so `gammaP` is at most
   A2's bound on `Bits w` with one more state bit. The certificate machinery of
   `Security/Proofs/Pous/P2/Rows.lean` then applies at `S + 1` (`meets_m1p_of_row`); `S = ⌊(18/19)·B·w⌋` does not
   depend on `b` either.
4. `m1p` is correct (`b ≤ w`, `p ≤ 2^w`), `|C| = B·w ≥ n = B·b`, and within the `21/20` space bound
   (`20·w ≤ 21·b`). No payload and no code block lies in the tail (`pad_lt`, `m1p_code`), and the code blocks are
   uniform on `[0, p)` (`m1p_code_uniform`).

The generic count (`PousGen`) is `Security/Proofs/Pous/P2/Count.lean`, the certificate
plumbing (`PousP2`, up to `meets_m1_of_row`) is `Security/Proofs/Pous/P2/Rows.lean`, and
A3′ is `PousTrackA.a3_prime_lam`. This module's own part is `PousP2p`.
-/

/-! ## M1 on P2's exact domain `[0, p)`: `Pous.m1p` -/

namespace PousP2p

open Pous Pous.Assumptions PousP2 Finset
open scoped ENNReal

/-! ### Bits as numbers -/

theorem bitsVal_lt {w : ℕ} (x : Bits w) : bitsVal x < 2 ^ w := (bitsEquiv w x).isLt

theorem bitsVal_bitsOfNat {w n : ℕ} (h : n < 2 ^ w) : bitsVal (bitsOfNat w n) = n := by
  simp [bitsVal, bitsOfNat, Nat.mod_eq_of_lt h]

theorem bitsOfNat_bitsVal {w : ℕ} (x : Bits w) : bitsOfNat w (bitsVal x) = x := by
  unfold bitsOfNat bitsVal
  rw [Equiv.symm_apply_eq]
  ext
  simp [Nat.mod_eq_of_lt (bitsEquiv w x).isLt]

section permAct

variable {w p : ℕ} (π : Equiv.Perm (Fin p))

theorem permAct_of_lt {x : Bits w} (h : bitsVal x < p) : permAct π x = bitsOfNat w (π ⟨bitsVal x, h⟩) := by
  simp only [permAct, h, ↓reduceDIte]

theorem permAct_of_ge {x : Bits w} (h : ¬ bitsVal x < p) : permAct π x = x := by
  simp only [permAct, h, ↓reduceDIte]

theorem bitsVal_permAct_of_lt (hp : p ≤ 2 ^ w) {x : Bits w} (h : bitsVal x < p) :
    bitsVal (permAct π x) = π ⟨bitsVal x, h⟩ := by
  rw [permAct_of_lt π h, bitsVal_bitsOfNat ((π _).isLt.trans_le hp)]

/-- The action of `π` undoes the action of `π⁻¹`, on the moved values and on the tail alike. -/
theorem permAct_symm (hp : p ≤ 2 ^ w) (y : Bits w) : permAct π (permAct π.symm y) = y := by
  by_cases h : bitsVal y < p
  · have h1 := bitsVal_permAct_of_lt π.symm hp h
    have h2 : bitsVal (permAct π.symm y) < p := h1 ▸ (π.symm _).isLt
    rw [permAct_of_lt π h2]
    have : (⟨bitsVal (permAct π.symm y), h2⟩ : Fin p) = π.symm ⟨bitsVal y, h⟩ := Fin.ext h1
    rw [this, Equiv.apply_symm_apply]
    exact bitsOfNat_bitsVal y
  · rw [permAct_of_ge π.symm h, permAct_of_ge π h]

end permAct

/-! ### The model: correctness, sizes, and the tail -/

/-- Block `j` of the payload `W` (`b` bits) zero-extended to `w` bits: the value `m1p` encodes at block `j`. -/
def pad (B w b : ℕ) (W : Bits (B * b)) (j : Fin B) : Bits w :=
  bitsOfNat w (bitsVal ((codeEquiv B b).symm W j))

theorem bitsVal_pad {B w b : ℕ} (hbw : 2 ^ b ≤ 2 ^ w) (W : Bits (B * b)) (j : Fin B) :
    bitsVal (pad B w b W j) = bitsVal ((codeEquiv B b).symm W j) :=
  bitsVal_bitsOfNat ((bitsVal_lt _).trans_le hbw)

/-- **No payload block lands in the tail**: an extended payload has value `< 2^b ≤ p`. -/
theorem pad_lt {B w p b : ℕ} (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w) (W : Bits (B * b)) (j : Fin B) :
    bitsVal (pad B w b W j) < p := by
  rw [bitsVal_pad (hb.trans hp2)]
  exact (bitsVal_lt _).trans_le hb

theorem le_of_two_pow_le {b w : ℕ} (h : 2 ^ b ≤ 2 ^ w) : b ≤ w := by
  by_contra hlt
  exact absurd h (not_le.2 (Nat.pow_lt_pow_right (by norm_num) (not_le.1 hlt)))

theorem m1p_correct (B w p b : ℕ) [NeZero B] (hbw : b ≤ w) (hp : p ≤ 2 ^ w) : (m1p B w p b).Correct := by
  intro W ω r
  change codeEquiv B b (fun j => bitsOfNat b
    (bitsVal (permAct (ω j) (permAct (ω j).symm (pad B w b W j))))) = W
  have h : ∀ j, bitsOfNat b (bitsVal (pad B w b W j)) = (codeEquiv B b).symm W j := by
    intro j
    exact (congrArg (bitsOfNat b) (bitsVal_pad (Nat.pow_le_pow_right two_pos hbw) W j)).trans (bitsOfNat_bitsVal _)
  simp only [permAct_symm _ hp, h]
  exact (codeEquiv B b).apply_symm_apply W

theorem m1p_codeHoldsW (B w p b : ℕ) [NeZero B] (hbw : b ≤ w) : (m1p B w p b).CodeHoldsW :=
  Nat.mul_le_mul_left B hbw

theorem m1p_spaceBound (B w p b : ℕ) [NeZero B] (hsp : 20 * w ≤ 21 * b) :
    (m1p B w p b).SpaceBound Params.expansion := by
  change (((B * w + 0 : ℕ)) : ℚ) ≤ 21 / 20 * ((B * b : ℕ) : ℚ)
  rw [Nat.add_zero]
  push_cast
  have h : (20 : ℚ) * w ≤ 21 * b := by exact_mod_cast hsp
  have hB : (0 : ℚ) ≤ B := Nat.cast_nonneg _
  nlinarith

/-- **No code block lands in the tail**: block `j` of the code has value `Π_j⁻¹(pad_j) < p`. -/
theorem m1p_code (B w p b : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w) (W : Bits (B * b))
    (ω : (permsp B w p).Ω) (j : Fin B) :
    bitsVal (((m1p B w p b).enc W ω ()).2 j) = (ω j).symm ⟨bitsVal (pad B w b W j), pad_lt hb hp2 W j⟩ :=
  bitsVal_permAct_of_lt (ω j).symm hp2 (pad_lt hb hp2 W j)

/-- `Π_j(x) = y` holds for a `1/p` fraction of the permutation tuples. -/
theorem card_eval_eq (B p : ℕ) (j : Fin B) (x y : Fin p) :
    (univ.filter fun ω : PousGen.Om B (Fin p) => ω j x = y).card * p = Fintype.card (PousGen.Om B (Fin p)) := by
  classical
  have hR : ∀ v : Fin p, v ∉ PousGen.R ([] : List (PousGen.Ent B (Fin p))) j := fun v h => by
    simpa using PousGen.mem_R.1 h
  have hc : ∀ v : Fin p, (PousGen.cons [((j, x), v)]).card = (univ.filter fun ω : PousGen.Om B (Fin p) =>
      ω j x = y).card := by
    intro v
    have e : PousGen.cons [((j, x), y)] = univ.filter fun ω : PousGen.Om B (Fin p) => ω j x = y := by
      rw [PousGen.cons_cons, PousGen.cons_nil]
    rw [← e]
    exact le_antisymm (PousGen.card_cons_le [] j x (hR v) (hR y)) (PousGen.card_cons_le [] j x (hR y) (hR v))
  have h := PousGen.card_cons_split ([] : List (PousGen.Ent B (Fin p))) j x
  rw [PousGen.cons_nil, card_univ] at h
  rw [h]
  simp only [hc, sum_const, card_univ, Fintype.card_fin, smul_eq_mul, mul_comm]

/-- **The code blocks are uniform on `[0, p)`**: each value `v < p` is block `j` of the code with probability
`1/p`, and by `m1p_code` no value `≥ p` is. -/
theorem m1p_code_uniform (B w p b : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w)
    (W : Bits (B * b)) (j : Fin B) (v : Fin p) :
    pr (fun ω : (permsp B w p).Ω => bitsVal (((m1p B w p b).enc W ω ()).2 j) = v) = (p : ℝ≥0∞)⁻¹ := by
  classical
  have hp0 : 0 < p := lt_of_lt_of_le (Nat.two_pow_pos _) hb
  have hiff : ∀ ω : (permsp B w p).Ω, bitsVal (((m1p B w p b).enc W ω ()).2 j) = v ↔
      ω j v = ⟨bitsVal (pad B w b W j), pad_lt hb hp2 W j⟩ := by
    intro ω
    rw [m1p_code B w p b hb hp2, ← Fin.ext_iff, Equiv.symm_apply_eq, eq_comm]
  have hcard := card_eval_eq B p j v ⟨bitsVal (pad B w b W j), pad_lt hb hp2 W j⟩
  obtain ⟨c, hc⟩ : ∃ c, (univ.filter fun ω : PousGen.Om B (Fin p) =>
      ω j v = ⟨bitsVal (pad B w b W j), pad_lt hb hp2 W j⟩).card = c := ⟨_, rfl⟩
  rw [hc] at hcard
  have hc0 : c ≠ 0 := by
    rintro rfl
    exact Fintype.card_ne_zero (α := PousGen.Om B (Fin p)) (by rw [← hcard, zero_mul])
  rw [pr_def]
  have e1 : (univ.filter fun ω : (permsp B w p).Ω => bitsVal (((m1p B w p b).enc W ω ()).2 j) = v).card = c := by
    rw [← hc]; exact PousGen.card_filter_congr _ _ _ _ hiff
  have e2 : Fintype.card (permsp B w p).Ω = p * c := by rw [mul_comm]; exact hcard.symm
  rw [e1, e2, Nat.cast_mul, ENNReal.div_eq_inv_mul,
    ENNReal.mul_inv (Or.inl (by exact_mod_cast hp0.ne')) (Or.inl (ENNReal.natCast_ne_top _)), mul_assoc,
    ENNReal.inv_mul_cancel (by exact_mod_cast hc0) (ENNReal.natCast_ne_top _), mul_one]

/-! ### Query trees over `[0, p)` -/

section Trans

variable {B w p : ℕ} {α : Type}

/-- A query tree over `permsp B w p` as one over `Fin B × Fin p`: a query at a value `< p` is asked, and a query in
the tail is answered by itself without asking. -/
def trP : PousGen.Tree (Fin B × Bits w) (Bits w) α → PousGen.Tree (Fin B × Fin p) (Fin p) α
  | .ret a => .ret a
  | .ask q k => if h : bitsVal q.2 < p then .ask (q.1, ⟨bitsVal q.2, h⟩) (fun y => trP (k (bitsOfNat w y)))
      else trP (k q.2)

theorem run_trP (ω : PousGen.Om B (Fin p)) (t : PousGen.Tree (Fin B × Bits w) (Bits w) α) :
    (trP t).run (PousGen.ans ω) = t.run (fun q => permAct (ω q.1) q.2) := by
  induction t with
  | ret a => rfl
  | ask q k ih =>
    by_cases h : bitsVal q.2 < p
    · simp only [trP, h, ↓reduceDIte, PousGen.Tree.run, PousGen.ans]
      rw [ih, permAct_of_lt _ h]
    · simp only [trP, h, ↓reduceDIte, PousGen.Tree.run]
      rw [ih, permAct_of_ge _ h]

theorem work_trP (ω : PousGen.Om B (Fin p)) (t : PousGen.Tree (Fin B × Bits w) (Bits w) α) :
    (trP t).work (PousGen.ans ω) ≤ t.work (fun q => permAct (ω q.1) q.2) := by
  induction t with
  | ret a => simp [trP, PousGen.Tree.work]
  | ask q k ih =>
    by_cases h : bitsVal q.2 < p
    · simp only [trP, h, ↓reduceDIte, PousGen.Tree.work, PousGen.ans]
      rw [permAct_of_lt _ h]
      exact Nat.add_le_add_right (ih _) 1
    · simp only [trP, h, ↓reduceDIte, PousGen.Tree.work]
      rw [permAct_of_ge _ h]
      exact (ih _).trans (Nat.le_succ _)

end Trans

/-! ### Lemma A on `[0, p)` -/

/-- Lemma A's bound on a domain of `p` points: `2^S · C(B, g) · Σ_{h ≤ g} C((g+1)·Q, h) / p^g`. -/
noncomputable def gammaP (B p S Q g : ℕ) : ℝ≥0∞ :=
  2 ^ S * (B.choose g) * (∑ h ∈ Finset.range (g + 1), (((g + 1) * Q).choose h : ℝ≥0∞)) / (p : ℝ≥0∞) ^ g

/-- **Lemma A for `m1p`** (good event `True`), from `PousGen.count_main` at `X = Fin p`. A responder's output `o`
for block `i` is read as the point `o mod p` of `[0, p)`: when `o` is block `i` of the code, that point is
`Π_i⁻¹(pad_i)`, since no code block lies in the tail. -/
theorem lemmaA_m1p (B w p b S D Q g : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w) :
    LemmaA (m1p B w p b) S D Q g (gammaP B p S Q g) (fun _ _ => True) := by
  intro W A₁ A₂
  classical
  have hp0 : 0 < p := lt_of_lt_of_le (Nat.two_pow_pos _) hb
  let pp0 : Bits (m1p B w p b).t := ((m1p B w p b).enc W (fun _ => Equiv.refl _) ()).1
  have hpp : ∀ ω, ((m1p B w p b).enc W ω ()).1 = pp0 := fun _ => rfl
  let fromP : Bits w → Fin p := fun o => ⟨bitsVal o % p, Nat.mod_lt _ hp0⟩
  let u : Fin B → Fin p := fun j => ⟨bitsVal (pad B w b W j), pad_lt hb hp2 W j⟩
  let pt : Bits S → Fin B → PousGen.Tree (Fin B × Fin p) (Fin p) (Fin p) := fun σ i =>
    (trP (PousGen.toTree (A₂.prog W pp0 σ i))).bind fun o => .ret (fromP o)
  have hpt : ∀ σ i (ω : PousGen.Om B (Fin p)), (pt σ i).work (PousGen.ans ω) ≤ Q := by
    intro σ i ω
    simp only [pt, PousGen.Tree.work_bind, PousGen.Tree.work, add_zero]
    refine (work_trP ω _).trans ?_
    exact (PousGen.work_toTree _ _).trans_le (A₂.bounded W pp0 σ i ω).2
  have hrun : ∀ σ i (ω : PousGen.Om B (Fin p)), (pt σ i).run (PousGen.ans ω) =
      fromP ((A₂.prog W pp0 σ i).run ((permsp B w p).answer ω)) := by
    intro σ i ω
    simp only [pt, PousGen.Tree.run_bind, PousGen.Tree.run]
    exact congrArg fromP ((run_trP ω _).trans (PousGen.run_toTree (M := permsp B w p) _ _))
  let σf : PousGen.Om B (Fin p) → Bits S := fun ω => A₁ W pp0 ((m1p B w p b).enc W ω ()).2 ω
  have hN := PousGen.count_main S Q g pt hpt (fun _ => ⟨0, hp0⟩) u σf
  rw [Fintype.card_fin] at hN
  have hfrac := PousGen.frac_le (Fintype.card_pos (α := PousGen.Om B (Fin p))) (pow_pos hp0 g) hN
  have key : ∀ (ω : PousGen.Om B (Fin p)) (j : Fin B),
      (A₂.prog W pp0 (σf ω) j).run ((permsp B w p).answer ω) = ((m1p B w p b).enc W ω ()).2 j →
      ω j ((pt (σf ω) j).run (PousGen.ans ω)) = u j := by
    intro ω j hj
    have e1 : (pt (σf ω) j).run (PousGen.ans ω) = fromP (((m1p B w p b).enc W ω ()).2 j) :=
      (hrun (σf ω) j ω).trans (congrArg fromP hj)
    have e2 : fromP (((m1p B w p b).enc W ω ()).2 j) = (ω j).symm (u j) := Fin.ext
      ((congrArg (· % p) (m1p_code B w p b hb hp2 W ω j)).trans (Nat.mod_eq_of_lt (Fin.isLt _)))
    calc ω j ((pt (σf ω) j).run (PousGen.ans ω)) = ω j ((ω j).symm (u j)) := congrArg (ω j) (e1.trans e2)
      _ = u j := Equiv.apply_symm_apply _ _
  have hsub : ∀ ω : PousGen.Om B (Fin p),
      ((univ : Finset (Fin B)).filter fun i => (A₂.prog W pp0 (σf ω) i).run ((permsp B w p).answer ω) =
        ((m1p B w p b).enc W ω ()).2 i) ⊆
      (univ.filter fun i => ω i ((pt (σf ω) i).run (PousGen.ans ω)) = u i) :=
    fun ω i hi => mem_filter.2 ⟨mem_univ _, key ω i (mem_filter.1 hi).2⟩
  let E' : PousGen.Om B (Fin p) → Prop := fun ω =>
    g ≤ (univ.filter fun i => ω i ((pt (σf ω) i).run (PousGen.ans ω)) = u i).card
  refine le_trans (pr_mono (Q := fun x : (permsp B w p).Ω × Unit => E' x.1) fun x hx => ?_) ?_
  · have h1 := hx.2
    exact h1.trans (card_le_card (hsub x.1))
  refine (PousThm1.pr_equiv (Equiv.prodPUnit (PousGen.Om B (Fin p))) E').trans_le ?_
  rw [pr_def]
  refine hfrac.trans (le_of_eq ?_)
  unfold gammaP PousGen.Ssum
  push_cast
  rfl

/-! ### From `p^g` to `2^(w·g)`: one bit -/

/-- `2^(w·g) ≤ 2·p^g` when `p ≥ 2^w − c` and `2·g·c ≤ 2^w` (Bernoulli's inequality). -/
theorem bernoulli_pow (w p c g : ℕ) (hc : 2 ^ w ≤ p + c) (hgc : 2 * g * c ≤ 2 ^ w) :
    2 ^ (w * g) ≤ 2 * p ^ g := by
  rcases Nat.eq_zero_or_pos g with rfl | hg
  · simp
  have hW : (0 : ℝ) < 2 ^ w := by positivity
  have hcR : (2 : ℝ) * g * c ≤ 2 ^ w := by exact_mod_cast hgc
  have hpR : (2 : ℝ) ^ w ≤ p + c := by exact_mod_cast hc
  have hg1 : (1 : ℝ) ≤ g := by exact_mod_cast hg
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have hcW : (c : ℝ) / 2 ^ w ≤ 1 / 2 := by
    rw [div_le_iff₀ hW]; nlinarith
  have ha : (-2 : ℝ) ≤ -((c : ℝ) / 2 ^ w) := by linarith
  have h1 := one_add_mul_le_pow ha g
  have h2 : (1 : ℝ) / 2 ≤ 1 + g * -((c : ℝ) / 2 ^ w) := by
    have : (g : ℝ) * ((c : ℝ) / 2 ^ w) ≤ 1 / 2 := by
      rw [← mul_div_assoc, div_le_iff₀ hW]; linarith
    linarith
  have h3 : 0 ≤ 1 + -((c : ℝ) / 2 ^ w) := by linarith
  have h4 : 1 + -((c : ℝ) / 2 ^ w) ≤ p / 2 ^ w := by
    rw [le_div_iff₀ hW, add_mul, one_mul, neg_mul, div_mul_cancel₀ _ hW.ne']
    linarith
  have h5 := pow_le_pow_left₀ h3 h4 g
  rw [div_pow, ← pow_mul] at h5
  have hWg : (0 : ℝ) < 2 ^ (w * g) := by positivity
  have h6 := h2.trans (h1.trans h5)
  rw [le_div_iff₀ hWg] at h6
  have : (2 : ℝ) ^ (w * g) ≤ 2 * (p : ℝ) ^ g := by linarith
  exact_mod_cast this

/-- Lemma A's bound on `[0, p)` is at most A2's bound on `Bits w` with one more state bit. -/
theorem gammaP_le (B w p c S Q g : ℕ) (hp0 : 0 < p) (hc : 2 ^ w ≤ p + c) (hgc : 2 * g * c ≤ 2 ^ w) :
    gammaP B p S Q g ≤ gammaA2 B w (S + 1) Q g := by
  have h := bernoulli_pow w p c g hc hgc
  have e1 : gammaP B p S Q g = ((2 ^ S * B.choose g * PousGen.Ssum ((g + 1) * Q) g : ℕ) : ℝ≥0∞) /
      ((p ^ g : ℕ) : ℝ≥0∞) := by
    unfold gammaP PousGen.Ssum; push_cast; rfl
  have e2 : gammaA2 B w (S + 1) Q g = ((2 ^ (S + 1) * B.choose g * PousGen.Ssum ((g + 1) * Q) g : ℕ) : ℝ≥0∞) /
      ((2 ^ (w * g) : ℕ) : ℝ≥0∞) := by
    unfold gammaA2 PousGen.Ssum; push_cast; rfl
  rw [e1, e2]
  refine PousGen.frac_le (pow_pos hp0 _) (pow_pos two_pos _) ?_
  calc 2 ^ S * B.choose g * PousGen.Ssum ((g + 1) * Q) g * 2 ^ (w * g)
      ≤ 2 ^ S * B.choose g * PousGen.Ssum ((g + 1) * Q) g * (2 * p ^ g) := Nat.mul_le_mul_left _ h
    _ = _ := by ring

/-! ### `Meets` for `m1p` from a row -/

/-- **`m1p` meets the requirement, given a certificate row with one extra state bit.** The hypotheses: the domain
`2^b ≤ p ≤ 2^w` (every payload in `[0, p)`; it gives `b ≤ w`); `p ≥ 2^w − c` with `2·B·c ≤ 2^w` (so
`p^g ≥ 2^(w·g)/2` for every `g ≤ B`); `20·w ≤ 21·b` (the space bound); and `P2Row` at `S + 1`, where `S` is `m1p`'s
state bound. -/
theorem meets_m1p_of_row (B w p b c Q k : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w)
    (hc : 2 ^ w ≤ p + c) (hBc : 2 * B * c ≤ 2 ^ w) (hsp : 20 * w ≤ 21 * b) (hk : 1 ≤ k)
    (hrow : P2Row B w ((m1p B w p b).stateBound Params.ρ + 1) Q k) (D : ℕ) :
    Meets (m1p B w p b) .sequential D Q k Params.εMax := by
  obtain ⟨g, -, hgB, hpow, hlog⟩ := hrow
  have hp0 : 0 < p := lt_of_lt_of_le (Nat.two_pow_pos _) hb
  have hbw : b ≤ w := le_of_two_pow_le (hb.trans hp2)
  refine ⟨le_rfl, m1p_correct B w p b hbw hp2, m1p_codeHoldsW B w p b hbw, m1p_spaceBound B w p b hsp, ?_⟩
  intro W A₁ A₂
  have hgc : 2 * g * c ≤ 2 ^ w := le_trans (by gcongr) hBc
  have hΓ : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaP B p ((m1p B w p b).stateBound Params.ρ) Q g ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 :=
    (mul_le_mul_right (gammaP_le B w p c _ Q g hp0 hc hgc) _).trans (hist_gamma_le B w _ Q g k hk hgB hlog)
  have h := PousTrackA.a3_prime_lam (m1p B w p b) _ D Q k g 128 _ (fun _ _ => True)
    (lemmaA_m1p B w p b ((m1p B w p b).stateBound Params.ρ) D Q g hb hp2) hΓ W A₁ A₂
  refine h.trans ?_
  simp only [not_true_eq_false, pr_false, add_zero]
  have hB0 : (0 : ℝ) < B := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne B)
  have hδ : ((((g - 1 : ℕ) : ℝ≥0∞) / (m1p B w p b).B) ^ k) ≤ ENNReal.ofReal Params.δ := by
    change ((((g - 1 : ℕ) : ℝ≥0∞) / (B : ℝ≥0∞)) ^ k) ≤ ENNReal.ofReal ((1 / 100 : ℚ) : ℝ)
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    have := (Rat.cast_le (K := ℝ)).2 hpow
    push_cast at this ⊢
    exact this
  exact add_le_add hδ le_rfl

/-! ### Rows at P2's recorded primes -/

open Real in
theorem logb_two_pow_nat (n : ℕ) : logb 2 (((2 ^ n : ℕ)) : ℝ) = n := by
  rw [Nat.cast_pow, Nat.cast_ofNat, logb_pow, logb_self_eq_one one_lt_two, mul_one]

/-- `m1p`'s state bound is `m1`'s: `⌊(18/19)·B·w⌋`. -/
theorem m1p_stateBound_le (B w p b : ℕ) [NeZero B] :
    (((m1p B w p b).stateBound Params.ρ : ℕ) : ℝ) ≤ 18 / 19 * (B : ℝ) * w := by
  have h := Nat.floor_le (show (0 : ℚ) ≤ Params.ρ * ((B * w : ℕ) : ℚ) by simp only [Params.ρ]; positivity)
  have h' : (((m1p B w p b).stateBound Params.ρ : ℕ) : ℚ) ≤ 18 / 19 * ((B : ℚ) * w) := by
    simpa [Scheme.stateBound, Params.ρ, m1p] using h
  have h'' := (Rat.cast_le (K := ℝ)).2 h'
  push_cast at h''
  linarith

open Real in
/-- **The uniform certificate with one extra state bit.** At `w = 16,448`, `T = 2^20` and `k = 88`, the row at
`S + 1` holds with `g = ⌊0.94901·B⌋ + 1` for every `B ≥ 2` with `B ≤ 2^n`, `88·n + 131 ≤ 5·B` and
`S ≤ (18/19)·16448·B`: `PousA2Rows.row_uniform` with its constant `130` raised to `131`. -/
theorem row_uniform (B n S : ℕ) (hB2 : 2 ≤ B) (hBn : B ≤ 2 ^ n) (hslack : 88 * n + 131 ≤ 5 * B)
    (hS : (S : ℝ) ≤ 18 / 19 * (B : ℝ) * 16448) :
    P2Row B 16448 (S + 1) (2 ^ 20) 88 := by
  obtain ⟨g, hg⟩ : ∃ g, g = ⌊(94901 / 100000 : ℚ) * B⌋₊ + 1 := ⟨_, rfl⟩
  have hc0 : 0 ≤ (94901 / 100000 : ℚ) * B := by positivity
  have hB0 : (0 : ℚ) < B := by exact_mod_cast (by omega : 0 < B)
  have hgB : g ≤ B := by
    have : ⌊(94901 / 100000 : ℚ) * B⌋₊ < B := (Nat.floor_lt hc0).2 (by linarith)
    omega
  refine ⟨g, by omega, hgB, ?_, ?_⟩
  · have hle : ((g - 1 : ℕ) : ℚ) / B ≤ 94901 / 100000 := by
      rw [div_le_iff₀ hB0, show g - 1 = ⌊(94901 / 100000 : ℚ) * B⌋₊ by omega]
      exact Nat.floor_le hc0
    calc (((g - 1 : ℕ) : ℚ) / B) ^ 88 ≤ (94901 / 100000 : ℚ) ^ 88 := pow_le_pow_left₀ (by positivity) hle 88
      _ ≤ 1 / 100 := by norm_num
  · refine (logHistGammaM1_le _ _ _ _ _ _ hB2 (by norm_num) (by omega) hgB (by norm_num)).trans ?_
    rw [logb_two_pow_nat]
    have hlogB : logb 2 (B : ℝ) ≤ n := by
      have := logb_le_logb_of_le one_lt_two (by exact_mod_cast (by omega : 0 < B))
        (show (B : ℝ) ≤ ((2 ^ n : ℕ) : ℝ) by exact_mod_cast hBn)
      rwa [logb_two_pow_nat] at this
    have hgc : (94901 / 100000 : ℚ) * B < ((⌊(94901 / 100000 : ℚ) * B⌋₊ : ℕ) : ℚ) + 1 := Nat.lt_floor_add_one _
    have hgR : (94901 / 100000 : ℝ) * B < g := by
      have := (Rat.cast_lt (K := ℝ)).2 hgc
      push_cast [hg] at this ⊢
      linarith
    have hl := log_two_gt_d9
    have hL : ((g : ℝ) + 1) / log 2 ≤ ((g : ℝ) + 1) * (14427 / 10000) := by
      rw [div_le_iff₀ (by positivity)]
      have h1 : (1 : ℝ) ≤ 14427 / 10000 * log 2 := by norm_num at hl ⊢; linarith
      calc ((g : ℝ) + 1) = ((g : ℝ) + 1) * 1 := (mul_one _).symm
        _ ≤ ((g : ℝ) + 1) * (14427 / 10000 * log 2) := mul_le_mul_of_nonneg_left h1 (by positivity)
        _ = _ := by ring
    have hsl : (88 : ℝ) * n + 131 ≤ 5 * B := by exact_mod_cast hslack
    have hBR : (0 : ℝ) ≤ B := Nat.cast_nonneg _
    push_cast at hL ⊢
    linarith

/-- `2^(w−1) ≤ 2^w − c` from `c ≤ 2^(w−1)`, without evaluating `2^w`. -/
theorem pow_sub_ge (w c : ℕ) (hw : 1 ≤ w) (hc : c ≤ 2 ^ (w - 1)) : 2 ^ (w - 1) ≤ 2 ^ w - c := by
  obtain ⟨v, rfl⟩ : ∃ v, w = v + 1 := ⟨w - 1, by omega⟩
  rw [Nat.add_sub_cancel] at hc ⊢
  rw [pow_succ]
  generalize 2 ^ v = X at *
  omega

/-- `c ≤ 2^a ≤ 2^b` for `a ≤ b`, without evaluating `2^b`. -/
theorem le_two_pow_of (c a b : ℕ) (hc : c ≤ 2 ^ a) (hab : a ≤ b) : c ≤ 2 ^ b :=
  hc.trans (Nat.pow_le_pow_right two_pos hab)

/-- **Satisfiability of the domain hypotheses at the 16,448-bit prime** `p = 2^16448 − 21065` (by `norm_num`, `omega`
and monotonicity of `2^·`, no `native_decide`): `2^16447 ≤ p ≤ 2^16448`, `2^16448 ≤ p + 21065`,
`2·2^23·21065 ≤ 2^16448`. -/
theorem hyps_16448 : 2 ^ (16448 - 1) ≤ 2 ^ 16448 - 21065 ∧ 2 ^ 16448 - 21065 ≤ 2 ^ 16448 ∧
    2 ^ 16448 ≤ (2 ^ 16448 - 21065) + 21065 ∧ 2 * 2 ^ 23 * 21065 ≤ 2 ^ 16448 :=
  ⟨pow_sub_ge 16448 21065 (by norm_num) (le_two_pow_of 21065 15 (16448 - 1) (by norm_num) (by norm_num)),
    Nat.sub_le _ _, (Nat.sub_add_cancel (le_two_pow_of 21065 15 16448 (by norm_num) (by norm_num))).ge,
    le_two_pow_of _ 39 16448 (by norm_num) (by norm_num)⟩

/-- **Satisfiability of the domain hypotheses at the 8,192-bit prime** `p = 2^8192 − 9345` (no `native_decide`). -/
theorem hyps_8192 : 2 ^ (8192 - 1) ≤ 2 ^ 8192 - 9345 ∧ 2 ^ 8192 - 9345 ≤ 2 ^ 8192 ∧
    2 ^ 8192 ≤ (2 ^ 8192 - 9345) + 9345 ∧ 2 * 2 ^ 24 * 9345 ≤ 2 ^ 8192 :=
  ⟨pow_sub_ge 8192 9345 (by norm_num) (le_two_pow_of 9345 14 (8192 - 1) (by norm_num) (by norm_num)),
    Nat.sub_le _ _, (Nat.sub_add_cancel (le_two_pow_of 9345 14 8192 (by norm_num) (by norm_num))).ge,
    le_two_pow_of _ 39 8192 (by norm_num) (by norm_num)⟩

/-- **Satisfiability of the width hypotheses** (by `norm_num`): at `w = 16,448`, `b = 16447` (one pad bit) and
`b = 16432` (2,054 bytes) satisfy `20·16448 ≤ 21·b` and `b ≤ 16447`; at `w = 8,192`, `b = 8191` and `b = 8176`
(1,022 bytes) satisfy `20·8192 ≤ 21·b` and `b ≤ 8191`. The least widths in range are `15665` and `7802`. -/
theorem widths : (20 * 16448 ≤ 21 * 16447 ∧ 16447 ≤ 16447) ∧ (20 * 16448 ≤ 21 * 16432 ∧ 16432 ≤ 16447) ∧
    (20 * 8192 ≤ 21 * 8191 ∧ 8191 ≤ 8191) ∧ (20 * 8192 ≤ 21 * 8176 ∧ 8176 ≤ 8191) ∧
    (20 * 16448 ≤ 21 * 15665 ∧ ¬ 20 * 16448 ≤ 21 * 15664) ∧ (20 * 8192 ≤ 21 * 7802 ∧ ¬ 20 * 8192 ≤ 21 * 7801) := by
  norm_num

/-- All hypotheses of `meets_m1p_of_row` other than the row also hold at small sizes, by `decide`: `w = 21`,
`p = 2^21 − 1`, `b = 20`, `c = 1`, `B = 2^8`. -/
theorem hyps_small : 2 ^ 20 ≤ 2 ^ 21 - 1 ∧ 2 ^ 21 - 1 ≤ 2 ^ 21 ∧ 2 ^ 21 ≤ (2 ^ 21 - 1) + 1 ∧
    2 * 2 ^ 8 * 1 ≤ 2 ^ 21 ∧ 20 * 21 ≤ 21 * 20 := by
  decide

/-- `2^b ≤ p` from `b ≤ w − 1` and `2^(w−1) ≤ p`. -/
theorem two_pow_le_of_le {b w p : ℕ} (hb : b ≤ w - 1) (hlo : 2 ^ (w - 1) ≤ p) : 2 ^ b ≤ p :=
  (Nat.pow_le_pow_right two_pos hb).trans hlo

/-- **The family on P2's exact 16,448-bit domain.** For every block count `2^8 ≤ B ≤ 2^23` and every payload width
`b` with `20·16448 ≤ 21·b` and `b ≤ 16447`, `m1p B 16448 (2^16448 − 21065) b` meets the requirement with sequential
reveal, `2^20` queries per answer, any depth, `88` challenges and `ε = 2^-128`. -/
theorem family : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, Meets (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 Params.εMax := by
  intro B _ h0 h1 b hsp hb D
  obtain ⟨hlo, hhi, hc, hBc⟩ := hyps_16448
  refine meets_m1p_of_row B 16448 _ b 21065 (2 ^ 20) 88 (two_pow_le_of_le hb hlo) hhi hc ?_ hsp (by norm_num) ?_ D
  · exact le_trans (by gcongr) hBc
  · have hS := m1p_stateBound_le B 16448 (2 ^ 16448 - 21065) b
    rw [Nat.cast_ofNat] at hS
    by_cases h9 : B ≤ 2 ^ 9
    · exact row_uniform B 9 _ (by omega) h9 (by omega) hS
    · exact row_uniform B 23 _ (by omega) h1 (by omega) hS

/-- **The MVP instance on P2's exact domain** (`2^19` blocks), for every payload width in range, from the family. -/
theorem p2_meets_m1p_b19 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, Meets (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 Params.εMax :=
  family (2 ^ 19) (by norm_num) (by norm_num)

/-- The `2^19` row at one pad bit per block, `b = 16447`. -/
theorem p2_meets_m1p_b19_w16447 :
    ∀ D, Meets (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16447) .sequential D (2 ^ 20) 88 Params.εMax :=
  p2_meets_m1p_b19 16447 widths.1.1 widths.1.2

/-- The `2^19` row at the MVP's byte-aligned packing, `b = 16432` (2,054 bytes per block). -/
theorem p2_meets_m1p_b19_w16432 :
    ∀ D, Meets (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432) .sequential D (2 ^ 20) 88 Params.εMax :=
  p2_meets_m1p_b19 16432 widths.2.1.1 widths.2.1.2

/-- The family's upper end, `2^23` (the `P2MeetsM1Uncond` block count), at the MVP's packing. -/
theorem p2_meets_m1p_b23_w16432 :
    ∀ D, Meets (m1p (2 ^ 23) 16448 (2 ^ 16448 - 21065) 16432) .sequential D (2 ^ 20) 88 Params.εMax :=
  family (2 ^ 23) (by norm_num) (by norm_num) 16432 widths.2.1.1 widths.2.1.2

/-- `S = ⌊(18/19)·2^24·8192⌋`, whatever `p` and `b`. -/
theorem stateBound_m1p_8192 (p b : ℕ) : (m1p (2 ^ 24) 8192 p b).stateBound Params.ρ = 130205324341 := by
  rw [Scheme.stateBound, Nat.floor_eq_iff (mul_nonneg (by norm_num [Params.ρ]) (Nat.cast_nonneg _))]
  change ((130205324341 : ℕ) : ℚ) ≤ 18 / 19 * ((2 ^ 24 * 8192 : ℕ) : ℚ) ∧
    18 / 19 * ((2 ^ 24 * 8192 : ℕ) : ℚ) < (130205324341 : ℕ) + 1
  norm_num

open Real in
/-- **The certificate at `w = 8,192` with one extra state bit**: `g = 15,938,000`, as in `PousP2.row_8192`. -/
theorem row_8192p : P2Row (2 ^ 24) 8192 (130205324341 + 1) (2 ^ 20) 90 := by
  refine ⟨15938000, by norm_num, by norm_num, by norm_num, ?_⟩
  refine (logHistGammaM1_le _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)).trans ?_
  rw [logb_two_pow_nat, logb_two_pow_nat]
  have hl := log_two_gt_d9
  have h : (15938001 : ℝ) / log 2 ≤ 15938001 / 0.6931471803 :=
    div_le_div_of_nonneg_left (by norm_num) (by norm_num) hl.le
  push_cast at h ⊢
  linarith

/-- **The 8,192-bit row on P2's exact domain**: `2^24` blocks, `p = 2^8192 − 9345`, `90` challenges, for every
payload width `b` with `20·8192 ≤ 21·b` and `b ≤ 8191`. -/
theorem p2_meets_m1p_8192 : ∀ b, 20 * 8192 ≤ 21 * b → b ≤ 8191 →
    ∀ D, Meets (m1p (2 ^ 24) 8192 (2 ^ 8192 - 9345) b) .sequential D (2 ^ 20) 90 Params.εMax := by
  intro b hsp hb D
  obtain ⟨hlo, hhi, hc, hBc⟩ := hyps_8192
  refine meets_m1p_of_row (2 ^ 24) 8192 _ b 9345 (2 ^ 20) 90 (two_pow_le_of_le hb hlo) hhi hc hBc hsp
    (by norm_num) ?_ D
  rw [stateBound_m1p_8192]
  exact row_8192p

/-- The 8,192-bit row at `b = 8191` and at the byte-aligned `b = 8176` (1,022 bytes per block). -/
theorem p2_meets_m1p_8192_instances :
    (∀ D, Meets (m1p (2 ^ 24) 8192 (2 ^ 8192 - 9345) 8191) .sequential D (2 ^ 20) 90 Params.εMax) ∧
    (∀ D, Meets (m1p (2 ^ 24) 8192 (2 ^ 8192 - 9345) 8176) .sequential D (2 ^ 20) 90 Params.εMax) :=
  ⟨p2_meets_m1p_8192 8191 widths.2.2.1.1 widths.2.2.1.2, p2_meets_m1p_8192 8176 widths.2.2.2.1.1 widths.2.2.2.1.2⟩

/-! ### Non-vacuity: raw storage breaks `k = 85` -/

/-- `S = ⌊(18/19)·2^19·16448⌋`, whatever `p` and `b`. -/
theorem stateBound_m1p_b19 (p b : ℕ) : (m1p (2 ^ 19) 16448 p b).stateBound Params.ρ = 8169621180 := by
  rw [Scheme.stateBound, Nat.floor_eq_iff (mul_nonneg (by norm_num [Params.ρ]) (Nat.cast_nonneg _))]
  change ((8169621180 : ℕ) : ℚ) ≤ 18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) ∧
    18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) < (8169621180 : ℕ) + 1
  norm_num

/-- **`k = 85` fails at `2^19` blocks, for every `p` and every payload width `b`.** Storing `⌊S/w⌋ = 496,693` of the
`2^19` code blocks raw (`Pous.raw_seq_accepted`) passes 85 sequential challenges with probability
`(496693/2^19)^85 > 1% + 2^-128`, for every `D` and `Q`. So `k = 88` is within three of what raw storage forces. -/
theorem not_meets_m1p_b19_k85 (p b D Q : ℕ) :
    ¬ Meets (m1p (2 ^ 19) 16448 p b) .sequential D Q 85 Params.εMax := by
  rintro ⟨-, -, -, -, hsec⟩
  have hS : 496693 * (m1p (2 ^ 19) 16448 p b).ℓ ≤ (m1p (2 ^ 19) 16448 p b).stateBound Params.ρ := by
    rw [stateBound_m1p_b19]
    change 496693 * 16448 ≤ 8169621180
    norm_num
  have hm : 496693 ≤ (m1p (2 ^ 19) 16448 p b).B := by
    change 496693 ≤ 2 ^ 19
    norm_num
  have h := (raw_seq_accepted (m1p (2 ^ 19) 16448 p b) hS hm D Q 85 (fun _ => false)).trans (hsec _ _ _)
  have hL : (((496693 : ℕ) : ℝ≥0∞) / ((m1p (2 ^ 19) 16448 p b).B : ℝ≥0∞)) ^ 85 =
      ENNReal.ofReal (((496693 : ℝ) / 2 ^ 19) ^ 85) := by
    change (((496693 : ℕ) : ℝ≥0∞) / ((2 ^ 19 : ℕ) : ℝ≥0∞)) ^ 85 = _
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast (2 ^ 19), ← ENNReal.ofReal_div_of_pos (by norm_num),
      ← ENNReal.ofReal_pow (by positivity)]
    norm_num
  have hR : ENNReal.ofReal Params.δ + Params.εMax = ENNReal.ofReal (1 / 100 + (2⁻¹ : ℝ) ^ 128) := by
    rw [ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]
    simp only [Params.δ, Params.εMax, Params.secParam]
    norm_num
  rw [hL, hR, ENNReal.ofReal_le_ofReal_iff (by positivity)] at h
  norm_num at h

/-- **The family fails at `k = 85`**: its `2^19` instance fails `Meets` at `85` challenges, at every width and every
`D` and `Q` (`not_meets_m1p_b19_k85`). -/
theorem family_k85_fails : ∃ (B : ℕ) (_ : NeZero B), 2 ^ 8 ≤ B ∧ B ≤ 2 ^ 23 ∧
    ∀ b D Q, ¬ Meets (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential D Q 85 Params.εMax :=
  ⟨2 ^ 19, inferInstance, by norm_num, by norm_num, not_meets_m1p_b19_k85 _⟩

end PousP2p
