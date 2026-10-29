import Definitions.Pous.Assumptions.P2
import Proofs.Pous.P2.RingBasic

/-!
# P2 in the ring model: recomputing a block breaks the requirement (tightness)

A responder that keeps no state recomputes the challenged block from `W`: it asks for the block's two keys, adds the
offset to the payload's label, takes the first root, applies `σ_K` locally and takes the second root. A root of `y` is
`y^((p+1)/4)` by the right-to-left binary method (`powAux`, one round `[acc·y, y·y]` per bit of the exponent), and one
more round squares it to fix the sign locally (`rootSel`: by Euler's criterion the square is `±y`). With `n` bits of
exponent the responder spends `2(n + 1)` multiplication rounds, `2` key queries and `4n + 5` ring queries, and it is
always right (`recBlock_run`). So `MeetsOn` fails at every `D ≥ 2(n + 1)` (`tight`), at P2's modulus `D ≥ 32,894`
(`tight_p16448`).
-/

namespace PousP2Ring

open Pous PousP2p PousThm1 Pous.P2.Assumptions
open scoped ENNReal

/-! ### The programs -/

section Programs

variable {B w p : ℕ} [NeZero p]

/-- `acc·y^e` over `n` bits of `e`, right to left: one round `[acc·y, y·y]` per bit, keeping `acc·y` when the bit is
set. -/
def powAux : ℕ → Bits w → Bits w → ℕ → OracleAlgorithm (ringRO B w p) (Bits w)
  | 0, acc, _, _ => .ret acc
  | n + 1, acc, y, e => .round [RingQry.mul acc y, RingQry.mul y y] fun r =>
      powAux n (if e % 2 = 1 then nthAns r 0 else acc) (nthAns r 1) (e / 2)

/-- The root's sign, chosen locally from `r` and `s = r²`: of `r` and `−r`, the one with an even label when `s` is
`a`, and the one with an odd label otherwise. -/
def rootSel (p : ℕ) {w : ℕ} (a r s : Bits w) : Bits w :=
  if bitsVal s = bitsVal a then (if bitsVal r % 2 = 0 then r else negL p r)
  else (if bitsVal r % 2 = 1 then r else negL p r)

/-- Square `r` and choose the sign. -/
def rootTail (a r : Bits w) : OracleAlgorithm (ringRO B w p) (Bits w) :=
  .round [RingQry.mul r r] fun r₂ => .ret (rootSel p a r (nthAns r₂ 0))

/-- `ρ(a)`: ask for `1`, raise `a` to `e` over `n` bits, then square and choose the sign. -/
def rootProg (n e : ℕ) (a : Bits w) : OracleAlgorithm (ringRO B w p) (Bits w) :=
  .round [RingQry.one] fun r₀ => pbind (powAux n (nthAns r₀ 0) a e) (rootTail a)

/-- **Recomputing block `i`** from its payload label `v`: both keys, `v + τ`, the first root, `σ_K`, the second
root. -/
def recBlock (n e : ℕ) (v : Bits w) (i : Fin B) : OracleAlgorithm (ringRO B w p) (Bits w) :=
  .round [RingQry.key i .mask, RingQry.key i .offset] fun r₁ =>
    .round [RingQry.add v (nthAns r₁ 1)] fun r₂ =>
      pbind (rootProg n e (nthAns r₂ 0)) fun y => rootProg n e (sigmaK p (nthAns r₁ 0) y)

end Programs

/-! ### Their cost, on every path -/

section Cost

variable {B w p : ℕ} [NeZero p] (f : (ringRO B w p).Qry → (ringRO B w p).Ans)

theorem powAux_cost : ∀ (n : ℕ) (acc y : Bits w) (e : ℕ),
    (powAux n acc y e : OracleAlgorithm (ringRO B w p) _).mulDepth f = n ∧
      (powAux n acc y e : OracleAlgorithm (ringRO B w p) _).keyWork f = 0 ∧
      (powAux n acc y e : OracleAlgorithm (ringRO B w p) _).ringWork f = 2 * n
  | 0, _, _, _ => ⟨rfl, rfl, rfl⟩
  | n + 1, acc, y, e => by
      obtain ⟨h1, h2, h3⟩ := powAux_cost n (if e % 2 = 1 then f (.mul acc y) else acc) (f (.mul y y)) (e / 2)
      refine ⟨?_, ?_, ?_⟩
      · change 1 + OracleAlgorithm.mulDepth f (powAux n (if e % 2 = 1 then f (.mul acc y) else acc) (f (.mul y y)) (e / 2)) = _
        omega
      · change 0 + OracleAlgorithm.keyWork f (powAux n (if e % 2 = 1 then f (.mul acc y) else acc) (f (.mul y y)) (e / 2)) = _
        omega
      · change 2 + OracleAlgorithm.ringWork f (powAux n (if e % 2 = 1 then f (.mul acc y) else acc) (f (.mul y y)) (e / 2)) = _
        omega

theorem rootProg_cost (n e : ℕ) (a : Bits w) :
    (rootProg n e a : OracleAlgorithm (ringRO B w p) _).mulDepth f = n + 1 ∧
      (rootProg n e a : OracleAlgorithm (ringRO B w p) _).keyWork f = 0 ∧
      (rootProg n e a : OracleAlgorithm (ringRO B w p) _).ringWork f = 2 * n + 2 := by
  obtain ⟨h1, h2, h3⟩ := powAux_cost (B := B) (p := p) f n (f .one) a e
  refine ⟨?_, ?_, ?_⟩
  · change 0 + OracleAlgorithm.mulDepth f (pbind (powAux n (f .one) a e) (rootTail a)) = _
    rw [mulDepth_pbind, h1]
    change 0 + (n + (1 + 0)) = _
    omega
  · change 0 + OracleAlgorithm.keyWork f (pbind (powAux n (f .one) a e) (rootTail a)) = _
    rw [keyWork_pbind, h2]
    rfl
  · change 1 + OracleAlgorithm.ringWork f (pbind (powAux n (f .one) a e) (rootTail a)) = _
    rw [ringWork_pbind, h3]
    change 1 + (2 * n + (1 + 0)) = _
    omega

theorem recBlock_cost (n e : ℕ) (v : Bits w) (i : Fin B) :
    (recBlock n e v i : OracleAlgorithm (ringRO B w p) _).mulDepth f = 2 * (n + 1) ∧
      (recBlock n e v i : OracleAlgorithm (ringRO B w p) _).keyWork f = 2 ∧
      (recBlock n e v i : OracleAlgorithm (ringRO B w p) _).ringWork f = 4 * n + 5 := by
  let y₀ := (rootProg n e (f (.add v (f (.key i .offset)))) : OracleAlgorithm (ringRO B w p) _).run f
  obtain ⟨a1, a2, a3⟩ := rootProg_cost (B := B) (p := p) f n e (f (.add v (f (.key i .offset))))
  obtain ⟨b1, b2, b3⟩ := rootProg_cost (B := B) (p := p) f n e (sigmaK p (f (.key i .mask)) y₀)
  refine ⟨?_, ?_, ?_⟩
  · change 0 + (0 + OracleAlgorithm.mulDepth f (pbind (rootProg n e (f (.add v (f (.key i .offset)))))
      fun y => rootProg n e (sigmaK p (f (.key i .mask)) y))) = _
    rw [mulDepth_pbind, a1, b1]
    omega
  · change 2 + (0 + OracleAlgorithm.keyWork f (pbind (rootProg n e (f (.add v (f (.key i .offset)))))
      fun y => rootProg n e (sigmaK p (f (.key i .mask)) y))) = _
    rw [keyWork_pbind, a2, b2]
  · change 0 + (1 + OracleAlgorithm.ringWork f (pbind (rootProg n e (f (.add v (f (.key i .offset)))))
      fun y => rootProg n e (sigmaK p (f (.key i .mask)) y))) = _
    rw [ringWork_pbind, a3, b3]
    omega

theorem ringBounded_mono {α : Type} {P : OracleAlgorithm (ringRO B w p) α} {D Q L D' Q' L' : ℕ} (h : P.RingBounded D Q L)
    (hD : D ≤ D') (hQ : Q ≤ Q') (hL : L ≤ L') : P.RingBounded D' Q' L' := fun ω =>
  ⟨(h ω).1.trans hD, (h ω).2.1.trans hQ, (h ω).2.2.trans hL⟩

/-- Recomputing a block costs `2(n + 1)` multiplication rounds, `2` key queries and `4n + 5` ring queries. -/
theorem recBlock_bounded (n e : ℕ) (v : Bits w) (i : Fin B) :
    (recBlock n e v i : OracleAlgorithm (ringRO B w p) _).RingBounded (2 * (n + 1)) 2 (4 * n + 5) := fun ω => by
  obtain ⟨h1, h2, h3⟩ := recBlock_cost (B := B) (p := p) ((ringRO B w p).answer ω) n e v i
  exact ⟨h1.le, h2.le, h3.le⟩

end Cost

/-! ### They compute the roots and the block -/

section Run

variable {B w p : ℕ} [NeZero p]

theorem powAux_run (hp2 : p ≤ 2 ^ w) (ω : (ringRO B w p).Ω) :
    ∀ (n : ℕ) (x y : ZMod p) (e : ℕ), e < 2 ^ n →
      (powAux n (lab w ω.1 x) (lab w ω.1 y) e : OracleAlgorithm (ringRO B w p) _).run ((ringRO B w p).answer ω) =
        lab w ω.1 (x * y ^ e)
  | 0, x, y, e, he => by
      obtain rfl : e = 0 := by simpa using he
      rw [pow_zero, mul_one]
      rfl
  | n + 1, x, y, e, he => by
      have hrun : (powAux (n + 1) (lab w ω.1 x) (lab w ω.1 y) e : OracleAlgorithm (ringRO B w p) _).run
          ((ringRO B w p).answer ω) =
          (powAux n (if e % 2 = 1 then (ringRO B w p).answer ω (.mul (lab w ω.1 x) (lab w ω.1 y)) else lab w ω.1 x)
            ((ringRO B w p).answer ω (.mul (lab w ω.1 y) (lab w ω.1 y))) (e / 2) : OracleAlgorithm (ringRO B w p) _).run
            ((ringRO B w p).answer ω) := rfl
      have hif : (if e % 2 = 1 then lab w ω.1 (x * y) else lab w ω.1 x) = lab w ω.1 (x * y ^ (e % 2)) := by
        rcases Nat.mod_two_eq_zero_or_one e with h | h <;> simp [h]
      have he2 : e / 2 < 2 ^ n := by
        rw [Nat.div_lt_iff_lt_mul two_pos, ← pow_succ]
        exact he
      rw [hrun, answer_mul ω hp2, answer_mul ω hp2, hif, powAux_run hp2 ω n _ _ _ he2]
      congr 1
      conv_rhs => rw [← Nat.mod_add_div e 2]
      rw [pow_add, pow_mul, sq, mul_assoc]

omit [NeZero p] in
/-- Euler's criterion at `p ≡ 3 mod 4`: `(y^((p+1)/4))² = ±y`. -/
theorem sq_root_pow (hp : p.Prime) (h4 : p % 4 = 3) (y : ZMod p) :
    y ^ ((p + 1) / 4) * y ^ ((p + 1) / 4) = y ∨ y ^ ((p + 1) / 4) * y ^ ((p + 1) / 4) = -y := by
  have := Fact.mk hp
  by_cases hy : y = 0
  · left
    rw [hy, zero_pow (by omega), mul_zero]
  · rw [← pow_add, show (p + 1) / 4 + (p + 1) / 4 = p / 2 + 1 by omega, pow_succ]
    rcases ZMod.pow_div_two_eq_neg_one_or_one p hy with h | h
    · left
      rw [h, one_mul]
    · right
      rw [h, neg_one_mul]

theorem sqE_eq_of_rhoE {φ : OddLabeling p} (hp : p.Prime) (h4 : p % 4 = 3) {x y : ZMod p} (h : sqE φ x = y) :
    lab w φ x = lab w φ (rhoE φ y) := by
  rw [rhoE_eq hp h4 φ h]

/-- `rootSel` picks the root `ρ(y)` from `r` with `r² = ±y`. -/
theorem rootSel_lab (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (φ : OddLabeling p) (y r : ZMod p)
    (h : r * r = y ∨ r * r = -y) :
    rootSel p (lab w φ y) (lab w φ r) (lab w φ (r * r)) = lab w φ (rhoE φ y) := by
  have hodd := odd_of_mod_four h4
  unfold rootSel
  rw [bitsVal_lab φ hp2 (r * r), bitsVal_lab φ hp2 y, bitsVal_lab φ hp2 r]
  by_cases hA : r * r = y
  · rw [ite_eq_left (by rw [hA])]
    split_ifs with hpar
    · refine sqE_eq_of_rhoE hp h4 ?_
      unfold sqE
      rw [ite_eq_right (by omega), hA]
    · rw [negL_lab φ hp2 hodd]
      have hr0 : r ≠ 0 := by
        rintro rfl
        rw [lab_zero φ hodd] at hpar
        exact hpar rfl
      have := parity_neg φ hodd hr0
      refine sqE_eq_of_rhoE hp h4 ?_
      unfold sqE
      rw [ite_eq_right (by omega), neg_mul_neg, hA]
  · have hB : r * r = -y := h.resolve_left hA
    have hne : ¬ ((φ.1 (r * r) : ℕ) = φ.1 y) := fun h' => hA (φ.1.injective (Fin.ext h'))
    rw [ite_eq_right hne]
    have hr0 : r ≠ 0 := by
      rintro rfl
      rw [mul_zero] at hB
      exact hA (by rw [mul_zero, neg_eq_zero.1 hB.symm])
    split_ifs with hpar
    · refine sqE_eq_of_rhoE hp h4 ?_
      unfold sqE
      rw [ite_eq_left hpar, hB, neg_neg]
    · rw [negL_lab φ hp2 hodd]
      have := parity_neg φ hodd hr0
      refine sqE_eq_of_rhoE hp h4 ?_
      unfold sqE
      rw [ite_eq_left (by omega), neg_mul_neg, hB, neg_neg]

/-- `rootProg` computes `ρ`, on every primitive, once the exponent `(p+1)/4` fits in `n` bits. -/
theorem rootProg_run (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (n : ℕ) (hn : (p + 1) / 4 < 2 ^ n)
    (ω : (ringRO B w p).Ω) (y : ZMod p) :
    (rootProg n ((p + 1) / 4) (lab w ω.1 y) : OracleAlgorithm (ringRO B w p) _).run ((ringRO B w p).answer ω) =
      lab w ω.1 (rhoE ω.1 y) := by
  change (pbind (powAux n ((ringRO B w p).answer ω .one) (lab w ω.1 y) ((p + 1) / 4)) (rootTail (lab w ω.1 y))).run
    ((ringRO B w p).answer ω) = _
  rw [run_pbind, answer_one, powAux_run hp2 ω n 1 y _ hn, one_mul]
  change rootSel p (lab w ω.1 y) (lab w ω.1 (y ^ ((p + 1) / 4)))
    ((ringRO B w p).answer ω (.mul (lab w ω.1 (y ^ ((p + 1) / 4))) (lab w ω.1 (y ^ ((p + 1) / 4))))) = _
  rw [answer_mul ω hp2]
  exact rootSel_lab hp h4 hp2 ω.1 y _ (sq_root_pow hp h4 y)

/-- **Recomputation is always right**: `recBlock` returns block `i` of the code, for a payload label below `p`. -/
theorem recBlock_run (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (n : ℕ) (hn : (p + 1) / 4 < 2 ^ n)
    (ω : (ringRO B w p).Ω) (i : Fin B) {v : Bits w} (hv : bitsVal v < p) :
    (recBlock n ((p + 1) / 4) v i : OracleAlgorithm (ringRO B w p) _).run ((ringRO B w p).answer ω) =
      encBlock ω.1 (ω.2 i).1 (ω.2 i).2 v := by
  obtain ⟨m, rfl⟩ : ∃ m, lab w ω.1 m = v := ⟨_, lab_eltD ω.1 hv⟩
  change (pbind (rootProg n ((p + 1) / 4) ((ringRO B w p).answer ω (.add (lab w ω.1 m)
      ((ringRO B w p).answer ω (.key i .offset))))) fun y =>
      rootProg n ((p + 1) / 4) (sigmaK p ((ringRO B w p).answer ω (.key i .mask)) y)).run
    ((ringRO B w p).answer ω) = _
  rw [run_pbind, answer_mask, answer_offset, answer_add ω hp2, rootProg_run hp h4 hp2 n hn]
  have hs := sigmaK_lt (K := (ω.2 i).1) (lab_lt ω.1 hp2 (rhoE ω.1 (m + ω.1.1.symm (ω.2 i).2)))
  obtain ⟨z, hz⟩ : ∃ z, lab w ω.1 z = sigmaK p (ω.2 i).1 (lab w ω.1 (rhoE ω.1 (m + ω.1.1.symm (ω.2 i).2))) :=
    ⟨_, lab_eltD ω.1 hs⟩
  rw [← hz, rootProg_run hp h4 hp2 n hn, encBlock, eltD_lab ω.1 hp2, ← hz, eltD_lab ω.1 hp2]

end Run

/-! ### Tightness -/

/-- **Tightness (non-vacuity) of P2 in the ring model.** For every prime `p ≡ 3 mod 4` with `2^b ≤ p ≤ 2^w`, and `n`
with `(p+1)/4 < 2^n`: a responder that keeps no state and recomputes each challenged block (`recBlock`) within
`2(n + 1)` multiplication rounds, `2` key queries and `4n + 5` ring queries is always right, so `p2ring` fails `MeetsOn`
in the sequential audit at every `D ≥ 2(n + 1)`, `Q ≥ 2`, `L ≥ 4n + 5`, `k` and `ε`. At P2's modulus this is
`D ≥ 32,894` (`tight_p16448`). No `(D, L)` at which `MeetsOn` holds is proved; at P2's parameters and `k ≤ 88` it
fails from `D = 2` once `L ≥ 2^30` (`not_meetsOn_b19_k88`). -/
theorem tight (B w p b : ℕ) [NeZero B] [NeZero p] (hp : p.Prime) (h4 : p % 4 = 3) (hb : 2 ^ b ≤ p)
    (hp2 : p ≤ 2 ^ w) (n : ℕ) (hn : (p + 1) / 4 < 2 ^ n) (D Q L k : ℕ) (hD : 2 * (n + 1) ≤ D) (hQ : 2 ≤ Q)
    (hL : 4 * n + 5 ≤ L) (ε : ℝ≥0∞) : ¬ MeetsOn (p2ring B w p b) .sequential D Q L k ε := by
  rintro ⟨hε, -, -, -, hsec⟩
  let A₂ : SeqResponderOn (p2ring B w p b) ((p2ring B w p b).stateBound Params.ρ) D Q L :=
    ⟨fun W _ _ σ i => progMap (fun c => (c, σ)) (recBlock n ((p + 1) / 4) (pad B w b W i) i),
      fun _ _ _ _ _ => ringBounded_progMap _ (ringBounded_mono (recBlock_bounded _ _ _ _) hD hQ hL)⟩
  have h1 : pr (seqAcceptsOn (p2ring B w p b) k (fun _ => false) (fun _ _ _ _ _ => false) A₂) = 1 :=
    pr_eq_one fun x => by
      show seqAnswers _ (fun _ σ i => progMap (fun c => (c, σ))
        (recBlock n ((p + 1) / 4) (pad B w b (fun _ => false) i) i)) 0 _ _ = _
      refine (seqAnswers_keep _ (fun _ i => recBlock n ((p + 1) / 4) (pad B w b (fun _ => false) i) i) 0 _ _).trans ?_
      refine List.map_ofFn.trans ?_
      exact congrArg List.ofFn (funext fun j => recBlock_run hp h4 hp2 n hn x.1 _ (pad_lt hb hp2 _ _))
  have h := (h1.symm.le.trans (hsec _ _ A₂)).trans (add_le_add le_rfl hε)
  have hR : ENNReal.ofReal Params.δ + Params.εMax = ENNReal.ofReal (1 / 100 + (2⁻¹ : ℝ) ^ 128) := by
    rw [ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]
    simp only [Params.δ, Params.εMax, Params.secParam]
    norm_num
  rw [hR, ← ENNReal.ofReal_one, ENNReal.ofReal_le_ofReal_iff (by positivity)] at h
  norm_num at h

set_option exponentiation.threshold 20000 in
theorem p16448_mod_four : (2 ^ 16448 - 21065) % 4 = 3 := by decide +kernel

set_option exponentiation.threshold 20000 in
theorem p16448_root_bits : (2 ^ 16448 - 21065 + 1) / 4 < 2 ^ 16446 := by decide +kernel

/-- **Tightness at P2's modulus** `p = 2^16448 − 21065`: recomputation (`2·16,447 = 32,894` multiplication rounds,
`2` key queries, `65,789` ring queries per answer) breaks `MeetsOn` for every `B`, every payload width `b ≤ 16,447`,
`k` and `ε`. -/
theorem tight_p16448 (hP : PrimeP16448) (B : ℕ) [NeZero B] (b : ℕ) (hb : b ≤ 16447) (D Q L k : ℕ)
    (hD : 32894 ≤ D) (hQ : 2 ≤ Q) (hL : 65789 ≤ L) (ε : ℝ≥0∞) :
    ¬ MeetsOn (p2ring B 16448 (2 ^ 16448 - 21065) b) .sequential D Q L k ε :=
  tight B 16448 _ b hP p16448_mod_four (two_pow_le_of_le hb hyps_16448.1) hyps_16448.2.1 16446 p16448_root_bits
    D Q L k hD hQ hL ε

end PousP2Ring
