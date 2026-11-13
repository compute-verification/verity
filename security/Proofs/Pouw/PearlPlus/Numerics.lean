module

public import Evaluate.Fp32
public import Evaluate.TC.Dots
public import Definitions.Pouw.PearlC.RowPrims
public import Proofs.Pouw.PearlC.CoreOps
public import Proofs.Properties.Core.Evaluate

/-!
# Pearl+'s bridge, the numerics: the check's FP32 add, square root and BF16 step are the semantics'

`PouwCheck.PearlPlus.Words` runs core's evaluator: `Evaluate.Fp32.f32AddV2` in the chain's promotion, the row statistics'
butterfly and sum and the peel's `B` factor, `Evaluate.Fp32.sqrtRn` for the row's `ρ`, and `Evaluate.TC.hopperBf16Dot16`
for the peel's four K16 steps. The semantics reads `TileCircuit.f32Add` (with `PeelExact.addWord?`, its rounding on finite
words), `RowCircuit.sqrtWord` and `PeelExact.k16Step`. Each statement says the two agree wherever the check reads the
result, which it refuses when it is not finite.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.Fp8Atom Pouw.PearlC

/-- **The check's FP32 add is the semantics'**: where `Evaluate.Fp32.f32AddV2 a b` is a finite word, `a` and `b` are
finite and it is `TileCircuit.f32Add a b`, so `PeelExact.addWord? a b` is `some` of it. -/
def Fp32AddIsTheSpec : Prop :=
  ∀ a b : ℕ, (wordQ (Evaluate.Fp32.f32AddV2 a b)).isSome →
    (wordQ a).isSome ∧ (wordQ b).isSome ∧ Evaluate.Fp32.f32AddV2 a b = TileCircuit.f32Add a b

/-- **The check's square root is the semantics'**: where `Evaluate.Fp32.sqrtRn a` is a finite word, it is
`RowCircuit.sqrtWord a` (both `sqrt.rn.f32`; they differ only in the NaN they write). -/
def SqrtIsTheSpec : Prop :=
  ∀ a : ℕ, (wordQ (Evaluate.Fp32.sqrtRn a)).isSome → Evaluate.Fp32.sqrtRn a = RowCircuit.sqrtWord a

/-- **The check's BF16 step is `k16Step`**: `Evaluate.TC.hopperBf16Dot16` on a finite FP32 accumulator word below `2^32`
(the domain `Verity.TC.Spec` is transcribed on; it reads the sign as `acc >>> 31 = 1`, unmasked) and 16 pairs of finite
BF16 words is `PeelExact.k16Step` on the same words. -/
def Bf16DotIsTheStep : Prop :=
  ∀ (acc : ℕ) (a b : List ℕ), (wordQ acc).isSome → acc < 2 ^ 32 → a.length = 16 → b.length = 16 →
    (∀ x ∈ a, x < 2 ^ 16 ∧ PeelExact.bf16Exp x ≠ 255) → (∀ y ∈ b, y < 2 ^ 16 ∧ PeelExact.bf16Exp y ≠ 255) →
    Evaluate.TC.hopperBf16Dot16 acc a b = PeelExact.k16Step acc (fun l => a.getD l.val 0) (fun l => b.getD l.val 0)

/-! ## The add -/

namespace Numerics

theorem wordQ_nanPtx : wordQ Verity.Fp32.F32_NAN_PTX = none := by decide +kernel

theorem wordQ_infWord (s : Bool) : wordQ (Verity.Fp32.infWord s) = none := by cases s <;> decide +kernel

theorem wordQ_nan : wordQ Evaluate.Fp32.F32_NAN = none := by decide +kernel

/-- A finite sum has finite addends. -/
theorem addRn_fin {x y : Verity.Fp32.Decoded} (h : (wordQ (x.addRn Verity.Fp32.F32_NAN_PTX 1 y)).isSome) :
    ∃ sa ma sb mb, x = .fin sa ma ∧ y = .fin sb mb := by
  cases x <;> cases y <;> first
    | exact ⟨_, _, _, _, rfl, rfl⟩
    | simp only [Verity.Fp32.Decoded.addRn, apply_ite, wordQ_nanPtx, wordQ_infWord, ite_self, Option.isSome_none,
        Bool.false_eq_true] at h

theorem wordQ_of_decode {u : ℕ} {s : Bool} {m : ℕ} (hu : Verity.Fp32.decode u = .fin s m) : ∃ x, wordQ u = some x := by
  by_cases he : (u >>> 23) % 256 = 255
  · rcases CoreOps.decode_nonfin he with h | ⟨t, h⟩ <;> rw [hu] at h <;> cases h
  · exact ⟨_, PeelExactProofs.wordQ_fields he⟩

end Numerics

open Numerics in
theorem fp32AddIsTheSpec : Fp32AddIsTheSpec := by
  intro a b h
  rw [show Evaluate.Fp32.f32AddV2 = Verity.Fp32.f32AddV2 from Properties.Evaluator.Fp32.f32AddV2] at h ⊢
  obtain ⟨sa, ma, sb, mb, hda, hdb⟩ := addRn_fin h
  obtain ⟨x, hx⟩ := wordQ_of_decode hda
  obtain ⟨y, hy⟩ := wordQ_of_decode hdb
  refine ⟨by rw [hx]; rfl, by rw [hy]; rfl, ?_⟩
  obtain ⟨Ma, hMa, rfl⟩ := CoreOps.decode_fin hx
  obtain ⟨Mb, hMb, rfl⟩ := CoreOps.decode_fin hy
  unfold Verity.Fp32.f32AddV2 TileCircuit.f32Add
  rw [hMa, hMb, hx, hy]
  simp only [Verity.Fp32.Decoded.addRn]
  rw [CoreOps.roundSum_eq _ _ _ _ Nat.one_pos]
  simp only [Nat.cast_one, div_one, CoreOps.val_eq_zero]

/-! ## The square root -/

namespace Numerics

theorem isqrtBits_spec : ∀ (k m r : ℕ), r * r ≤ m → m < (r + 2 ^ k) * (r + 2 ^ k) →
    RowCircuit.isqrtBits k m r * RowCircuit.isqrtBits k m r ≤ m ∧
      m < (RowCircuit.isqrtBits k m r + 1) * (RowCircuit.isqrtBits k m r + 1)
  | 0, m, r, h1, h2 => by simpa [RowCircuit.isqrtBits] using And.intro h1 h2
  | k + 1, m, r, h1, h2 => by
    simp only [RowCircuit.isqrtBits]
    apply isqrtBits_spec
    · split
      · assumption
      · exact h1
    · split
      · rw [Nat.add_assoc, ← Nat.two_mul, ← Nat.pow_succ']; exact h2
      · next h => exact Nat.lt_of_not_le h

/-- `RowCircuit.isqrt` is `⌊√m⌋` below `2^64`. -/
theorem isqrt_row {m : ℕ} (hm : m < 2 ^ 64) : RowCircuit.isqrt m = Nat.sqrt m :=
  Nat.eq_sqrt.2 (isqrtBits_spec 32 m 0 (by simp) (by simpa [← Nat.pow_add] using hm))

theorem div_le_of_gt {n x s : ℕ} (hs : n < (s + 1) * (s + 1)) (hx : s + 1 ≤ x) : n / x ≤ s := by
  apply Nat.le_of_lt_succ
  rw [Nat.div_lt_iff_lt_mul (by omega)]
  exact Nat.lt_of_lt_of_le hs (Nat.mul_le_mul_left _ hx)

theorem le_newton {n x s : ℕ} (hs : s * s ≤ n) (hx : s ≤ x) : s ≤ (x + n / x) / 2 := by
  rcases Nat.eq_zero_or_pos x with rfl | hx0
  · omega
  have h : s - (x - s) ≤ n / x := by
    rw [Nat.le_div_iff_mul_le hx0]
    refine Nat.le_trans ?_ hs
    obtain ⟨a, rfl⟩ : ∃ a, x = s + a := ⟨x - s, by omega⟩
    rw [show s + a - s = a by omega]
    rcases Nat.le_total a s with ha | ha
    · obtain ⟨b, rfl⟩ : ∃ b, s = b + a := ⟨s - a, by omega⟩
      rw [show b + a - a = b by omega]
      have : b * (b + a + a) ≤ (b + a) * (b + a) := by
        simp only [Nat.mul_add, Nat.add_mul, Nat.mul_comm a b]; omega
      exact this
    · rw [show s - a = 0 by omega, Nat.zero_mul]; exact Nat.zero_le _
  omega

theorem isqrt_go {n s : ℕ} (hs : s * s ≤ n) (hs' : n < (s + 1) * (s + 1)) :
    ∀ fuel x, s ≤ x → x - s < 2 ^ fuel → Evaluate.Fp32.isqrt.go n fuel x = s
  | 0, x, h1, h2 => by simp only [Evaluate.Fp32.isqrt.go]; simp at h2; omega
  | fuel + 1, x, h1, h2 => by
    simp only [Evaluate.Fp32.isqrt.go]
    have hy := le_newton hs h1
    rcases Nat.eq_or_lt_of_le h1 with rfl | hlt
    · rw [ite_eq_right (by omega)]
    · have hq := div_le_of_gt hs' hlt
      rw [ite_eq_left (by omega)]
      apply isqrt_go hs hs' fuel _ hy
      rw [Nat.pow_succ] at h2
      omega

/-- `Evaluate.Fp32.isqrt` is `⌊√n⌋`: Newton's iteration from above, the error at least halving each step. -/
theorem isqrt_eval (n : ℕ) : Evaluate.Fp32.isqrt n = Nat.sqrt n := by
  have hs := Nat.sqrt_le n
  have hs' := Nat.lt_succ_sqrt n
  generalize Nat.sqrt n = s at hs hs'
  unfold Evaluate.Fp32.isqrt
  split
  · next h =>
    subst h
    have : s ≤ 0 := (Nat.mul_self_le_mul_self_iff (m := s) (n := 0)).1 (by rw [Nat.zero_mul]; exact hs)
    omega
  · next h =>
    have hn : n < 2 ^ (n.log2 + 1) := Nat.lt_log2_self
    have hx : (1 <<< (n.log2 / 2 + 1)) * (1 <<< (n.log2 / 2 + 1)) = 2 ^ (2 * (n.log2 / 2 + 1)) := by
      rw [Nat.one_shiftLeft, ← Nat.pow_add]; congr 1; omega
    have hle : 2 ^ (n.log2 + 1) ≤ 2 ^ (2 * (n.log2 / 2 + 1)) := Nat.pow_le_pow_right (by omega) (by omega)
    have hsx : s ≤ 1 <<< (n.log2 / 2 + 1) := by
      apply Nat.le_of_lt
      apply Nat.mul_self_lt_mul_self_iff.1
      omega
    apply isqrt_go hs hs' _ _ hsx
    have : 1 <<< (n.log2 / 2 + 1) < 2 ^ (n.log2 + 4) := by
      rw [Nat.one_shiftLeft]; exact Nat.pow_lt_pow_right (by omega) (by omega)
    omega

/-- `roundRn _ n 2` for `n` strictly between `2^K·P` and `2^K·(P + 1)`, `P` of at least 25 bits: the word of `P·2^(K−1)`
rounded with a sticky bit, which depends on `n` only through `P`. -/
theorem roundRn_two_eq (s : Bool) {n K P : ℕ} (hK : 1 ≤ K) (hP : 2 ^ 24 ≤ P) (hlo : 2 ^ K * P < n)
    (hhi : n < 2 ^ K * (P + 1)) :
    Verity.Fp32.roundRn s n 2 = Verity.Fp32.zeroWord s ||| (if ((P.log2 + K - 24) <<< 23) + (P / 2 ^ (P.log2 - 23) +
      if 2 ^ (P.log2 - 24) ≤ P % 2 ^ (P.log2 - 23) then 1 else 0) ≥ 0x7F800000 then 0x7F800000
      else ((P.log2 + K - 24) <<< 23) + (P / 2 ^ (P.log2 - 23) +
        if 2 ^ (P.log2 - 24) ≤ P % 2 ^ (P.log2 - 23) then 1 else 0)) := by
  have hP0 : P ≠ 0 := by omega
  have hL1 : 2 ^ P.log2 ≤ P := Nat.log2_self_le hP0
  have hL2 : P < 2 ^ (P.log2 + 1) := Nat.lt_log2_self
  have hL24 : 24 ≤ P.log2 := (Nat.le_log2 hP0).2 hP
  generalize P.log2 = L at hL1 hL2 hL24 ⊢
  have hA : 0 < 2 ^ K := Nat.two_pow_pos K
  have hlo' : 2 ^ (L + K) ≤ 2 ^ K * P := by
    rw [Nat.pow_add, Nat.mul_comm]; exact Nat.mul_le_mul_left _ hL1
  have hhi' : 2 ^ K * (P + 1) ≤ 2 ^ (L + K) * 2 := by
    have h2 : 2 ^ (L + K) * 2 = 2 ^ K * 2 ^ (L + 1) := by
      rw [← Nat.pow_succ, ← Nat.pow_add]; congr 1; omega
    rw [h2]; exact Nat.mul_le_mul_left _ hL2
  have hn2 : n / 2 ≠ 0 := by omega
  have hlog : (n / 2).log2 = L + K - 1 := by
    apply Nat.le_antisymm
    · have : (n / 2).log2 < L + K := (Nat.log2_lt hn2).2 ((Nat.div_lt_iff_lt_mul (by omega)).2 (by omega))
      omega
    · apply (Nat.le_log2 hn2).2
      apply (Nat.le_div_iff_mul_le (by omega)).2
      have h2 : 2 ^ (L + K - 1) * 2 = 2 ^ (L + K) := by rw [← Nat.pow_succ]; congr 1; omega
      rw [h2]; omega
  have hT : (2 : ℕ) <<< (L + K - 1 - 23) = 2 ^ K * 2 ^ (L - 23) := by
    rw [Nat.shiftLeft_eq, ← Nat.pow_succ', ← Nat.pow_add]; congr 1; omega
  have hdiv : n / 2 ^ K = P := Nat.div_eq_of_lt_le (by rw [Nat.mul_comm]; omega) (by rw [Nat.mul_comm]; omega)
  have hq : n / (2 ^ K * 2 ^ (L - 23)) = P / 2 ^ (L - 23) := by rw [← Nat.div_div_eq_div_mul, hdiv]
  have hr : n % (2 ^ K * 2 ^ (L - 23)) = n % 2 ^ K + 2 ^ K * (P % 2 ^ (L - 23)) := by rw [Nat.mod_mul, hdiv]
  have he1 : 0 < n % 2 ^ K := by
    have := Nat.div_add_mod n (2 ^ K); rw [hdiv] at this; omega
  have he2 : n % 2 ^ K < 2 ^ K := Nat.mod_lt _ hA
  have hB : 2 ^ (L - 23) = 2 * 2 ^ (L - 24) := by rw [← Nat.pow_succ']; congr 1; omega
  have hcond : (2 * (n % (2 ^ K * 2 ^ (L - 23))) > 2 ^ K * 2 ^ (L - 23) ∨
      (2 * (n % (2 ^ K * 2 ^ (L - 23))) = 2 ^ K * 2 ^ (L - 23) ∧ P / 2 ^ (L - 23) % 2 = 1)) ↔
      2 ^ (L - 24) ≤ P % 2 ^ (L - 23) := by
    rw [hr, hB]
    generalize P % (2 * 2 ^ (L - 24)) = c
    generalize 2 ^ (L - 24) = B
    generalize n % 2 ^ K = e at he1 he2
    rw [Nat.mul_left_comm (2 ^ K) 2 B]
    rcases Nat.lt_or_ge c B with h | h
    · have : 2 ^ K * (c + 1) ≤ 2 ^ K * B := Nat.mul_le_mul_left _ h
      rw [Nat.mul_succ] at this
      omega
    · have : 2 ^ K * B ≤ 2 ^ K * c := Nat.mul_le_mul_left _ h
      omega
  unfold Verity.Fp32.roundRn
  simp only [hlog, hT, hq]
  rw [show (L + K - 1 - 23) = L + K - 24 by omega]
  by_cases hc : 2 ^ (L - 24) ≤ P % 2 ^ (L - 23)
  · rw [ite_eq_left (hcond.2 hc), ite_eq_left hc]
  · rw [ite_eq_right (fun h => hc (hcond.1 h)), ite_eq_right hc, Nat.add_zero]

theorem roundRn_two_congr (s : Bool) {n₁ n₂ K P : ℕ} (hK : 1 ≤ K) (hP : 2 ^ 24 ≤ P) (h1 : 2 ^ K * P < n₁)
    (h1' : n₁ < 2 ^ K * (P + 1)) (h2 : 2 ^ K * P < n₂) (h2' : n₂ < 2 ^ K * (P + 1)) :
    Verity.Fp32.roundRn s n₁ 2 = Verity.Fp32.roundRn s n₂ 2 := by
  rw [roundRn_two_eq s hK hP h1 h1', roundRn_two_eq s hK hP h2 h2']

theorem four_pow (k : ℕ) : (4 : ℕ) ^ k = 2 ^ k * 2 ^ k := by rw [← mul_pow]; norm_num

theorem sqrt_bounds (m D : ℕ) :
    Nat.sqrt m * 2 ^ D ≤ Nat.sqrt (m * 4 ^ D) ∧ Nat.sqrt (m * 4 ^ D) < (Nat.sqrt m + 1) * 2 ^ D := by
  have hs := Nat.sqrt_le m
  have hs' := Nat.lt_succ_sqrt m
  constructor
  · rw [Nat.le_sqrt, four_pow]
    calc Nat.sqrt m * 2 ^ D * (Nat.sqrt m * 2 ^ D) = Nat.sqrt m * Nat.sqrt m * (2 ^ D * 2 ^ D) := by ring
      _ ≤ m * (2 ^ D * 2 ^ D) := Nat.mul_le_mul_right _ hs
  · rw [Nat.sqrt_lt, four_pow]
    calc m * (2 ^ D * 2 ^ D) < (Nat.sqrt m + 1) * (Nat.sqrt m + 1) * (2 ^ D * 2 ^ D) :=
          Nat.mul_lt_mul_of_pos_right hs' (by positivity)
      _ = (Nat.sqrt m + 1) * 2 ^ D * ((Nat.sqrt m + 1) * 2 ^ D) := by ring

theorem sqrt_exact {m D : ℕ} (hm : Nat.sqrt m * Nat.sqrt m = m) :
    Nat.sqrt (m * 4 ^ D) = Nat.sqrt m * 2 ^ D := by
  conv_lhs => rw [← hm, four_pow, show Nat.sqrt m * Nat.sqrt m * (2 ^ D * 2 ^ D) =
    Nat.sqrt m * 2 ^ D * (Nat.sqrt m * 2 ^ D) by ring]
  exact Nat.sqrt_eq _

theorem sqrt_inexact {m D : ℕ} (hm : Nat.sqrt m * Nat.sqrt m ≠ m) :
    Nat.sqrt (m * 4 ^ D) * Nat.sqrt (m * 4 ^ D) < m * 4 ^ D := by
  refine lt_of_le_of_ne (Nat.sqrt_le _) fun heq => hm ?_
  generalize Nat.sqrt (m * 4 ^ D) = s at heq
  have hdvd : 2 ^ D ∣ s := by
    have : (2 ^ D) ^ 2 ∣ s ^ 2 := ⟨m, by rw [sq, sq, heq, four_pow]; ring⟩
    exact (Nat.pow_dvd_pow_iff two_ne_zero).1 this
  obtain ⟨k, rfl⟩ := hdvd
  have hkk : k * k = m := by
    apply Nat.eq_of_mul_eq_mul_right (show 0 < 2 ^ D * 2 ^ D by positivity)
    rw [← four_pow, ← heq, four_pow]; ring
  rw [← hkk, Nat.sqrt_eq]

theorem bracket {r s X : ℕ} (h1 : r * X ≤ s) (h2 : s < (r + 1) * X) :
    X * 2 * r < 2 * s + 1 ∧ 2 * s + 1 < X * 2 * (r + 1) ∧ X * 2 * r < (2 * r + 1) * X ∧
      (2 * r + 1) * X < X * 2 * (r + 1) := by
  have hX : 0 < X := by
    rcases Nat.eq_zero_or_pos X with rfl | h
    · simp at h2
    · exact h
  refine ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

/-- **The root of a positive finite word**: with `M = m₀·2^(max e 1 − 1)` units and `sqrtWord`'s `m₁`, `h`, `t`, `m`,
`M·2^149 = m·4^D` for `D = h − t + 149`, so `⌊√(M·2^149)⌋` is `r·2^D` when `r = ⌊√m⌋` is exact and lies strictly between
`r·2^D` and `(r + 1)·2^D` otherwise, where `roundRn _ _ 2` is constant (`roundRn_two_congr`). -/
theorem sqrt_main (e mm M m₀ m₁ t m : ℕ) (h : ℤ) (he : e < 255) (hmm : mm < 2 ^ 23) (h0 : e ≠ 0 ∨ mm ≠ 0)
    (hm₀ : m₀ = mm + (if e = 0 then 0 else 2 ^ 23)) (hM : M = m₀ * 2 ^ (max e 1 - 1))
    (hm₁ : m₁ = if max e 1 % 2 = 1 then 2 * m₀ else m₀)
    (hh : h = (((max e 1 : ℕ) : ℤ) - (if max e 1 % 2 = 1 then 151 else 150)) / 2)
    (ht : t = (54 - bitLength m₁) / 2 + 1) (hm : m = m₁ * 4 ^ t) :
    Evaluate.Fp32.roundRn false (2 * Evaluate.Fp32.isqrt (M <<< 149) +
      if Evaluate.Fp32.isqrt (M <<< 149) * Evaluate.Fp32.isqrt (M <<< 149) < M <<< 149 then 1 else 0) 2 =
    f32Word (((RowCircuit.isqrt m : ℚ) + if RowCircuit.isqrt m * RowCircuit.isqrt m = m then 0 else 1 / 2) *
      2 ^ (h - (t : ℤ))) := by
  rw [show Evaluate.Fp32.roundRn = Verity.Fp32.roundRn from Properties.Evaluator.Fp32.roundRn]
  generalize hE : max e 1 = E at hM hm₁ hh
  have hE1 : 1 ≤ E := by omega
  have hm₀pos : 0 < m₀ := by rw [hm₀]; split_ifs <;> omega
  have hm₀lt : m₀ < 2 ^ 24 := by rw [hm₀]; split_ifs <;> omega
  have hm₁pos : 0 < m₁ := by rw [hm₁]; split_ifs <;> omega
  have hm₁lt : m₁ < 2 ^ 25 := by rw [hm₁]; split_ifs <;> omega
  have hL1 : 2 ^ m₁.log2 ≤ m₁ := Nat.log2_self_le (by omega)
  have hL2 : m₁ < 2 ^ (m₁.log2 + 1) := Nat.lt_log2_self
  have hL25 : m₁.log2 < 25 := (Nat.log2_lt (by omega)).2 hm₁lt
  have hbl : bitLength m₁ = m₁.log2 + 1 := by unfold bitLength; rw [if_neg (by omega)]
  rw [hbl] at ht
  generalize m₁.log2 = L at hL1 hL2 hL25 ht
  have h4 : ∀ k : ℕ, (4 : ℕ) ^ k = 2 ^ (2 * k) := fun k => by rw [pow_mul]; norm_num
  have hmlo : 2 ^ 54 ≤ m := by
    rw [hm, h4]
    calc 2 ^ 54 ≤ 2 ^ (L + 2 * t) := Nat.pow_le_pow_right (by norm_num) (by omega)
      _ = 2 ^ L * 2 ^ (2 * t) := pow_add _ _ _
      _ ≤ m₁ * 2 ^ (2 * t) := Nat.mul_le_mul_right _ hL1
  have hmhi : m < 2 ^ 64 := by
    rw [hm, h4]
    calc m₁ * 2 ^ (2 * t) < 2 ^ (L + 1) * 2 ^ (2 * t) := Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
      _ = 2 ^ (L + 1 + 2 * t) := (pow_add _ _ _).symm
      _ ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) (by omega)
  rw [isqrt_row hmhi]
  have hr27 : 2 ^ 27 ≤ Nat.sqrt m := Nat.le_sqrt.2 (by rw [← pow_add]; exact hmlo)
  obtain ⟨D, hD⟩ : ∃ D : ℕ, (D : ℤ) = h - t + 149 :=
    ⟨(h - t + 149).toNat, Int.toNat_of_nonneg (by rw [hh]; split_ifs <;> omega)⟩
  have hkey : M <<< 149 = m * 4 ^ D := by
    rw [Nat.shiftLeft_eq, hM, hm, h4, h4, mul_assoc, ← pow_add, mul_assoc, ← pow_add]
    by_cases hodd : E % 2 = 1
    · rw [if_pos hodd] at hm₁ hh
      rw [hm₁, mul_comm 2 m₀, mul_assoc, ← pow_succ']
      congr 2; omega
    · rw [if_neg hodd] at hm₁ hh
      rw [hm₁]; congr 2; omega
  have hz : h - (t : ℤ) = (D : ℤ) - ((149 : ℕ) : ℤ) := by push_cast; omega
  rw [hz, zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast, isqrt_eval, hkey]
  obtain ⟨hs1, hs2⟩ := sqrt_bounds m D
  have hr0 : 0 < Nat.sqrt m := lt_of_lt_of_le (by norm_num) hr27
  by_cases hrr : Nat.sqrt m * Nat.sqrt m = m
  · have hsq : Nat.sqrt m * 2 ^ D * (Nat.sqrt m * 2 ^ D) = m * 4 ^ D := by
      conv_rhs => rw [← hrr]
      rw [four_pow]; ring
    have hnl : ¬ (Nat.sqrt m * 2 ^ D * (Nat.sqrt m * 2 ^ D) < m * 4 ^ D) := by rw [hsq]; exact lt_irrefl _
    rw [sqrt_exact hrr, if_neg hnl, if_pos hrr]
    have hN : 0 < 2 * (Nat.sqrt m * 2 ^ D) + 0 := by have := Nat.mul_pos hr0 (Nat.two_pow_pos D); omega
    rw [CoreOps.roundRn_f32Word false hN two_pos]
    refine congrArg f32Word ?_
    rw [if_neg Bool.false_ne_true]; push_cast; ring
  · have hlt := sqrt_inexact (D := D) hrr
    rw [if_pos hlt, if_neg hrr]
    have hN : 0 < (2 * Nat.sqrt m + 1) * 2 ^ D := by positivity
    have hval : ((Nat.sqrt m : ℚ) + 1 / 2) * (2 ^ D / 2 ^ 149) = (if false = true then -1 else 1) *
        ((((2 * Nat.sqrt m + 1) * 2 ^ D : ℕ) : ℚ) / ((2 : ℕ) : ℚ) / 2 ^ 149) := by
      rw [if_neg Bool.false_ne_true]; push_cast; ring
    rw [hval, ← CoreOps.roundRn_f32Word false hN two_pos]
    obtain ⟨b1, b2, b3, b4⟩ := bracket hs1 hs2
    exact roundRn_two_congr false (K := D + 1) (P := Nat.sqrt m) (by omega)
      (le_trans (Nat.pow_le_pow_right (by norm_num) (by norm_num)) hr27)
      (by rw [pow_succ]; exact b1) (by rw [pow_succ]; exact b2) (by rw [pow_succ]; exact b3)
      (by rw [pow_succ]; exact b4)

end Numerics

open Numerics in
theorem sqrtIsTheSpec : SqrtIsTheSpec := by
  intro a hfin
  have hd := Properties.Evaluator.Fp32.decode_toCore a
  rw [CoreOps.decode_fields] at hd
  by_cases he : (a >>> 23) % 256 = 255
  · exfalso
    rw [if_pos he] at hd
    have hnf : Evaluate.Fp32.sqrtRn a = Evaluate.Fp32.F32_NAN ∨ Evaluate.Fp32.sqrtRn a = 0x7F800000 := by
      unfold Evaluate.Fp32.sqrtRn
      cases hE : Evaluate.Fp32.decode a with
      | nan => exact .inl rfl
      | inf s => cases s; exacts [.inr rfl, .inl rfl]
      | fin s m => rw [hE] at hd; split_ifs at hd <;> cases hd
    rcases hnf with h | h <;> rw [h] at hfin
    · rw [wordQ_nan] at hfin; exact absurd hfin (by decide)
    · rw [CoreOps.wordQ_inf] at hfin; exact absurd hfin (by decide)
  rw [if_neg he] at hd
  have hmm : a % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  have hdE : Evaluate.Fp32.decode a = .fin (PeelExact.signBit a)
      ((a % 2 ^ 23 + (if (a >>> 23) % 256 = 0 then 0 else 2 ^ 23)) * 2 ^ (max ((a >>> 23) % 256) 1 - 1)) := by
    cases hE : Evaluate.Fp32.decode a with
    | nan => rw [hE] at hd; split_ifs at hd <;> cases hd
    | inf s => rw [hE] at hd; split_ifs at hd <;> cases hd
    | fin s m =>
      rw [hE] at hd
      by_cases h0 : (a >>> 23) % 256 = 0
      · rw [if_pos h0] at hd
        cases hd
        rw [if_pos h0, h0]; simp
      · rw [if_neg h0] at hd
        cases hd
        rw [if_neg h0, Nat.max_eq_left (show 1 ≤ (a >>> 23) % 256 by omega), CoreOps.or_hidden hmm,
          Nat.shiftLeft_eq]
        congr 1; ring
  have hu1 : ((a % 2 ^ 32) >>> 23) % 256 = (a >>> 23) % 256 := by omega
  have hu2 : a % 2 ^ 32 % 2 ^ 23 = a % 2 ^ 23 := by omega
  have hsb : PeelExact.signBit a = true ↔ 2 ^ 31 ≤ a % 2 ^ 32 := by
    unfold PeelExact.signBit; rw [beq_iff_eq]; omega
  have h1 : ¬ ((a % 2 ^ 32) >>> 23) % 256 = 255 := by omega
  by_cases hz : (a >>> 23) % 256 = 0 ∧ a % 2 ^ 23 = 0
  · have h2 : a % 2 ^ 32 % 2 ^ 31 = 0 := by omega
    have hL : Evaluate.Fp32.sqrtRn a = if PeelExact.signBit a = true then 0x80000000 else 0 := by
      unfold Evaluate.Fp32.sqrtRn
      rw [hdE, hz.1, hz.2]
      cases PeelExact.signBit a <;> rfl
    rw [hL]
    show _ = (if ((a % 2 ^ 32) >>> 23) % 256 = 255 then _ else if a % 2 ^ 32 % 2 ^ 31 = 0 then _ else _)
    rw [if_neg h1, if_pos h2]
    by_cases hs : PeelExact.signBit a = true
    · rw [if_pos hs]; have := hsb.1 hs; omega
    · rw [if_neg hs]; have : ¬ 2 ^ 31 ≤ a % 2 ^ 32 := fun h => hs (hsb.2 h); omega
  have hM0 : (a % 2 ^ 23 + (if (a >>> 23) % 256 = 0 then 0 else 2 ^ 23)) * 2 ^ (max ((a >>> 23) % 256) 1 - 1) ≠ 0 := by
    apply Nat.mul_ne_zero
    · split_ifs <;> omega
    · positivity
  have h2 : ¬ a % 2 ^ 32 % 2 ^ 31 = 0 := by omega
  by_cases hs : PeelExact.signBit a = true
  · exfalso
    have hL : Evaluate.Fp32.sqrtRn a = Evaluate.Fp32.F32_NAN := by
      unfold Evaluate.Fp32.sqrtRn; rw [hdE, hs]; exact if_neg hM0
    rw [hL, wordQ_nan] at hfin; exact absurd hfin (by decide)
  have hs' : PeelExact.signBit a = false := by simpa using hs
  have h3 : ¬ 2 ^ 31 ≤ a % 2 ^ 32 := fun h => hs (hsb.2 h)
  unfold Evaluate.Fp32.sqrtRn
  rw [hdE, hs']
  show (if _ = 0 then 0 else _) = (if ((a % 2 ^ 32) >>> 23) % 256 = 255 then _ else if a % 2 ^ 32 % 2 ^ 31 = 0 then _
    else if 2 ^ 31 ≤ a % 2 ^ 32 then _ else _)
  rw [if_neg hM0, if_neg h1, if_neg h2, if_neg h3]
  exact sqrt_main (((a % 2 ^ 32) >>> 23) % 256) (a % 2 ^ 32 % 2 ^ 23) _ _ _ _ _ _ (by omega)
    (Nat.mod_lt _ (by norm_num)) (by omega) rfl (by rw [hu1, hu2]) rfl rfl rfl rfl

/-! ## The BF16 step on an accumulator word below `2^32` -/

namespace Numerics

open PeelExact


/-- Core's term as the Definitions' (the same three fields). -/
def termOf (t : Verity.TC.Spec.Term) : Pouw.Fp8Atom.Term := ⟨t.negative, t.exponent, t.magnitude⟩

/-- Core's value as the Definitions'. -/
def valueOf : Verity.TC.Spec.Value → Pouw.Fp8Atom.Value
  | .nan => .nan
  | .inf s => .inf s
  | .fin t => .fin (termOf t)

theorem packValue_eq (v : Verity.TC.Spec.Value) :
    Verity.TC.Spec.packValue v = Pouw.Fp8Atom.packValue (valueOf v) := by
  cases v with
  | nan => rfl
  | inf s => rfl
  | fin t =>
    show Verity.TC.Spec.packFp32 t = Pouw.Fp8Atom.packTerm (termOf t)
    unfold Verity.TC.Spec.packFp32 Pouw.Fp8Atom.packTerm termOf Verity.TC.Spec.FP32_MAX_EXPONENT
    dsimp only
    by_cases h : t.magnitude &&& 8388608 = 0 <;> simp [h]

theorem fp32Value_eq {w : ℕ} (hw : w < 2 ^ 32) :
    valueOf (Verity.TC.Spec.fp32Value w) = Pouw.Fp8Atom.fp32Value w := by
  have h1 : (w >>> 23) &&& 0xFF = (w >>> 23) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  have h2 : w &&& 0x7FFFFF = w % 2 ^ 23 := Nat.and_two_pow_sub_one_eq_mod _ 23
  have h3 : decide (w >>> 31 = 1) = ((w >>> 31) % 2 == 1) := by
    rw [Nat.shiftRight_eq_div_pow]
    have : w / 2 ^ 31 < 2 := by omega
    rcases (by omega : w / 2 ^ 31 = 0 ∨ w / 2 ^ 31 = 1) with h | h <;> rw [h] <;> rfl
  unfold Verity.TC.Spec.fp32Value Verity.TC.Spec.fp32Class Verity.TC.Spec.fp32Term Pouw.Fp8Atom.fp32Value
  simp only [h1, h2, h3]
  by_cases he : (w >>> 23) % 256 = 255
  · by_cases hm : w % 8388608 = 0
    · simp [he, hm, valueOf, h3]
    · simp [he, hm, valueOf]
  · by_cases h0 : (w >>> 23) % 256 = 0
    · by_cases hm : w % 8388608 = 0
      · simp [he, h0, hm, valueOf, termOf, Verity.TC.Spec.ZERO, Pouw.Fp8Atom.Term.zero]
      · simp [he, h0, hm, valueOf, termOf, Verity.TC.Spec.FP32_MIN_NORMAL_EXPONENT, h3]
    · simp [he, h0, valueOf, termOf, h3]

theorem sign_eq (a b : ℕ) : decide ((a ^^^ b) >>> 15 &&& 1 = 1) = (bf16Sign a ^^ bf16Sign b) := by
  rw [Nat.and_one_is_mod, Nat.shiftRight_xor_distrib]
  unfold bf16Sign
  rcases Nat.mod_two_eq_zero_or_one (a >>> 15) with ha | ha <;>
    rcases Nat.mod_two_eq_zero_or_one (b >>> 15) with hb | hb <;> simp [Nat.xor_mod_two_eq_one, ha, hb] <;>
      omega

theorem class_nan (a : ℕ) : Verity.TC.Spec.bf16Class a = .nan ↔ bf16IsNaN a = true := by
  have h1 : (a >>> 7) &&& 0xFF = (a >>> 7) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  have h2 : a &&& 0x7F = a % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
  unfold Verity.TC.Spec.bf16Class bf16IsNaN bf16Exp bf16Man
  simp only [h1, h2]
  by_cases hE : (a >>> 7) % 256 = 255 <;> by_cases hM : a % 128 = 0 <;> by_cases h0 : (a >>> 7) % 256 = 0 <;>
    simp [hE, hM, h0]

theorem class_inf (a : ℕ) : Verity.TC.Spec.bf16Class a = .inf ↔ bf16IsInf a = true := by
  have h1 : (a >>> 7) &&& 0xFF = (a >>> 7) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  have h2 : a &&& 0x7F = a % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
  unfold Verity.TC.Spec.bf16Class bf16IsInf bf16Exp bf16Man
  simp only [h1, h2]
  by_cases hE : (a >>> 7) % 256 = 255 <;> by_cases hM : a % 128 = 0 <;> by_cases h0 : (a >>> 7) % 256 = 0 <;>
    simp [hE, hM, h0]

theorem class_zero (a : ℕ) : Verity.TC.Spec.bf16Class a = .zero ↔ bf16IsZero a = true := by
  have h1 : (a >>> 7) &&& 0xFF = (a >>> 7) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  have h2 : a &&& 0x7F = a % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
  unfold Verity.TC.Spec.bf16Class bf16IsZero bf16Exp bf16Man
  simp only [h1, h2]
  by_cases hE : (a >>> 7) % 256 = 255 <;> by_cases hM : a % 128 = 0 <;> by_cases h0 : (a >>> 7) % 256 = 0 <;>
    simp [hE, hM, h0]

theorem sig_eq (a : ℕ) :
    (a &&& 0x7F ||| (if (a >>> 7) &&& 0xFF ≠ 0 then 0x80 else 0)) = bf16Sig a := by
  have h1 : (a >>> 7) &&& 0xFF = (a >>> 7) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  have h2 : a &&& 0x7F = a % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
  unfold bf16Sig bf16Exp bf16Man
  rw [h1, h2]
  by_cases h : (a >>> 7) % 256 = 0
  · simp [h]
  · simp only [h, ne_eq, not_false_eq_true, ite_true, ite_false]
    have hm : a % 128 < 2 ^ 7 := Nat.mod_lt _ (by decide)
    have := Nat.shiftLeft_add_eq_or_of_lt hm 1
    rw [Nat.one_shiftLeft] at this
    show (a % 128 ||| 2 ^ 7) = 2 ^ 7 + a % 128
    rw [Nat.or_comm, ← this]

theorem product_eq (a b : ℕ) : termOf (Verity.TC.Spec.bf16Product a b) = bf16Product a b := by
  have hx : ∀ x : ℕ, (if (x >>> 7) &&& 0xFF ≠ 0 then (x >>> 7) &&& 0xFF else 1) = bf16ExpOr1 x := by
    intro x
    have h1 : (x >>> 7) &&& 0xFF = (x >>> 7) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
    unfold bf16ExpOr1 bf16Exp; rw [h1]; by_cases h : (x >>> 7) % 256 = 0 <;> simp [h] <;> omega
  unfold Verity.TC.Spec.bf16Product bf16Product
  simp only [sig_eq, hx, sign_eq]
  by_cases h : (bf16Sig a * bf16Sig b) <<< 9 = 0 <;>
    simp [h, termOf, Verity.TC.Spec.ZERO, Pouw.Fp8Atom.Term.zero]

theorem productTotal_eq (a b : ℕ) :
    valueOf (Verity.TC.Spec.bf16ProductTotal a b) = bf16ProductTotal a b := by
  unfold Verity.TC.Spec.bf16ProductTotal bf16ProductTotal
  simp only [class_nan, class_inf, class_zero, Bool.or_eq_true, sign_eq]
  cases bf16IsNaN a <;> cases bf16IsNaN b <;> cases bf16IsInf a <;> cases bf16IsInf b <;> cases bf16IsZero a <;>
    cases bf16IsZero b <;> simp [valueOf, sign_eq, product_eq]

theorem any_isNaN (vs : List Verity.TC.Spec.Value) :
    (vs.map valueOf).any Pouw.Fp8Atom.Value.isNaN = vs.any Verity.TC.Spec.Value.isNaN := by
  induction vs with
  | nil => rfl
  | cons v vs ih => cases v <;> simp [ih, valueOf, Pouw.Fp8Atom.Value.isNaN, Verity.TC.Spec.Value.isNaN]

theorem any_isInf (s : Bool) (vs : List Verity.TC.Spec.Value) :
    (vs.map valueOf).any (Pouw.Fp8Atom.Value.isInf s) = (vs.filterMap Verity.TC.Spec.Value.infSign?).contains s := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    cases v with
    | inf n =>
      simp only [List.map_cons, List.any_cons, List.filterMap_cons, valueOf, Pouw.Fp8Atom.Value.isInf,
        Verity.TC.Spec.Value.infSign?, List.contains_cons, ih]
      cases n <;> cases s <;> rfl
    | _ =>
      simp only [List.map_cons, List.any_cons, List.filterMap_cons, valueOf, Pouw.Fp8Atom.Value.isInf,
        Verity.TC.Spec.Value.infSign?, Bool.false_or, ih]

theorem finite_eq (vs : List Verity.TC.Spec.Value) :
    (vs.map valueOf).filterMap Pouw.Fp8Atom.Value.finite? = (vs.filterMap Verity.TC.Spec.Value.term?).map termOf := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    cases v <;> simp only [List.map_cons, List.filterMap_cons, valueOf, Pouw.Fp8Atom.Value.finite?,
      Verity.TC.Spec.Value.term?, ih]

theorem foldl_add {α : Type} (g : α → Int) (l : List α) (c : Int) :
    l.foldl (fun acc x => acc + g x) c = c + (l.map g).sum := by
  induction l generalizing c with
  | nil => simp
  | cons x l ih => simp [List.foldl, ih, Int.add_assoc]

theorem foldl_congr_add {α : Type} (g : α → Int) (l : List α) (f : Int → α → Int)
    (hf : ∀ acc x, f acc x = acc + g x) : l.foldl f 0 = (l.map g).sum := by
  have : f = fun acc x => acc + g x := funext fun acc => funext fun x => hf acc x
  subst this
  rw [foldl_add]; simp

theorem groupSum_eq (ts : List Verity.TC.Spec.Term) :
    termOf (Verity.TC.Spec.groupSum ts 26 (-133)) = Pouw.Fp8Atom.groupSum 26 (-133) (ts.map termOf) := by
  unfold Verity.TC.Spec.groupSum Pouw.Fp8Atom.groupSum Pouw.Fp8Atom.alignedSum
  generalize hE : Pouw.Fp8Atom.maxExp (-133) (ts.map termOf) = E
  have hE' : ts.foldl (fun mx (t : Verity.TC.Spec.Term) => if t.magnitude ≠ 0 ∧ t.exponent > mx then t.exponent else mx)
      (-133) = E := by
    rw [← hE, Pouw.Fp8Atom.maxExp, List.foldl_map]; rfl
  simp only [hE']
  rw [foldl_congr_add (fun t => Pouw.Fp8Atom.aligned 26 E (termOf t)) ts _ (fun acc t => by
    simp only [Pouw.Fp8Atom.aligned, Pouw.Fp8Atom.toAdder, termOf, Verity.TC.Spec.FP32_SIGNIFICAND_WIDTH]
    by_cases h0 : t.magnitude = 0 <;> by_cases hn : t.negative = true <;> simp [h0, hn, Int.sub_eq_add_neg])]
  rw [show (ts.map fun t => Pouw.Fp8Atom.aligned 26 E (termOf t)) = (ts.map termOf).map (Pouw.Fp8Atom.aligned 26 E)
    by rw [List.map_map]; rfl]
  generalize ((ts.map termOf).map (Pouw.Fp8Atom.aligned 26 E)).sum = T
  unfold Pouw.Fp8Atom.normalize Pouw.Fp8Atom.fromAdder
  simp only [Verity.TC.Spec.FP32_SIGNIFICAND_WIDTH, Verity.TC.Spec.FP32_MIN_NORMAL_EXPONENT,
    show ∀ n, Verity.TC.bitLength n = Pouw.Fp8Atom.bitLength n from fun _ => rfl]
  by_cases hT : T = 0
  · simp [hT, termOf, Verity.TC.Spec.ZERO, Pouw.Fp8Atom.Term.zero]
  simp only [hT, ite_false]
  generalize Pouw.Fp8Atom.bitLength T.natAbs = bl
  simp only [apply_ite termOf]
  rfl

theorem groupStepTotal_eq (acc : Verity.TC.Spec.Value) (ps : List Verity.TC.Spec.Value) :
    valueOf (Verity.TC.Spec.groupStepTotal acc ps 26 (-133)) =
      Pouw.Fp8Atom.groupStepTotal 26 (-133) (valueOf acc) (ps.map valueOf) := by
  unfold Verity.TC.Spec.groupStepTotal Pouw.Fp8Atom.groupStepTotal
  dsimp only
  rw [show valueOf acc :: ps.map valueOf = (acc :: ps).map valueOf from rfl, any_isNaN, any_isInf, any_isInf,
    finite_eq, ← groupSum_eq]
  generalize (acc :: ps).any Verity.TC.Spec.Value.isNaN = A
  generalize Verity.TC.Spec.groupSum ((acc :: ps).filterMap Verity.TC.Spec.Value.term?) 26 (-133) = t
  generalize (acc :: ps).filterMap Verity.TC.Spec.Value.infSign? = L
  cases A
  · cases L with
    | nil =>
      simp only [List.contains_nil, Bool.false_eq_true, and_self, Bool.and_self, ite_false, Bool.false_eq_true]
      rw [apply_ite valueOf]; rfl
    | cons s L =>
      cases s <;> cases h1 : L.contains true <;> cases h2 : L.contains false <;>
        simp only [List.contains_cons, h1, h2] <;> rfl
  · rfl

theorem products_eq (a b : List ℕ) (ha : a.length = 16) (hb : b.length = 16) :
    (List.zipWith Verity.TC.Spec.bf16ProductTotal a b).map valueOf =
      List.ofFn fun l : Fin 16 => bf16ProductTotal (a.getD l.val 0) (b.getD l.val 0) := by
  apply List.ext_getElem
  · simp [ha, hb]
  · intro i h1 h2
    have hi : i < 16 := by simpa using h2
    simp only [List.getElem_map, List.getElem_zipWith, List.getElem_ofFn, productTotal_eq,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show i < a.length by omega),
      List.getElem?_eq_getElem (show i < b.length by omega), Option.getD_some]

theorem k16_bounded (acc : ℕ) (a b : List ℕ) (hacc : acc < 2 ^ 32) (ha : a.length = 16) (hb : b.length = 16) :
    Verity.TC.Spec.tcDotTotal Verity.TC.HOPPER_BF16_WGMMA_K16 acc a b =
      k16Step acc (fun l => a.getD l.val 0) (fun l => b.getD l.val 0) := by
  unfold Verity.TC.Spec.tcDotTotal k16Step
  simp only [Verity.TC.HOPPER_BF16_WGMMA_K16, hopperBf16K16, Verity.TC.Spec.runGroups]
  rw [List.take_of_length_le (by omega), List.take_of_length_le (by omega), packValue_eq, groupStepTotal_eq,
    fp32Value_eq hacc, products_eq a b ha hb]

end Numerics

open Numerics in
/-- The BF16 step on any 16 word pairs and an accumulator word below `2^32`, finite or not. -/
theorem bf16Dot_of_lt (acc : ℕ) (a b : List ℕ) (hacc : acc < 2 ^ 32) (ha : a.length = 16) (hb : b.length = 16) :
    Evaluate.TC.hopperBf16Dot16 acc a b =
      PeelExact.k16Step acc (fun l => a.getD l.val 0) (fun l => b.getD l.val 0) := by
  rw [show Evaluate.TC.hopperBf16Dot16 = _ from Properties.Evaluator.TC.hopperBf16Dot16]
  exact k16_bounded acc a b hacc ha hb

theorem bf16DotIsTheStep : Bf16DotIsTheStep :=
  fun acc a b _ hacc ha hb _ _ => bf16Dot_of_lt acc a b hacc ha hb

end Pouw.PearlPlus.Bridge
