import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Proofs.Pouw.PearlC.HonestWords
import Proofs.Pouw.Copy

/-!
# Guessing a `1 × 1 × k` unit's two output words at W1

On the earlier domain (`pearlCDomainDevOld`, which admits
a single `1 × 1 × k` unit), a W1 program that guesses the unit's ticket word and useful word costs `2·fetch` and wins with
probability at least `Pr[E]/2^64`, above `εPearlC 0 1 = 2^−128`. This file proves it, and what the narrowed domain
(`64 ≤ n ∧ 64 ∣ n`, `pearlCDomainDev`) does to that program. It is analysis: it states and lists no assumption.

**The program** (`guessProg t y`): two free moves, constant word 0 to ticket word `(0, 0, 0)` and constant word 1 to
useful word `(0, 0, 0)`. The adversary `guessAdv` has no preprocessing and supplies activations `a`.
- `guess_cost`: its W1 cost (`W1.w1Priced`) is exactly `2·fetch` on every device, workload, state and salt: the moves
  cost 0 and the run fetches the two constants (`guess_fetched`, `guess_read`); `guess_bounded`: it is bounded by
  `2·fetch` and `0` queries.
- `GuessEvent P k w a θ`: on `(H, s)`, unit `(1, 1, k)` with weights `w` and activations `a` passes `actOK`, `capOK`
  and `weightOK`, its two reference words are FP32 words' values, and its credit exceeds `θ`.
- `wins_of_event`: on that event, the program whose constants are those two words wins TT_OUT's game at `θ`; so
  `exists_guess_ge`: some constants `t, y < 2^32` win with probability at least `Pr[GuessEvent]/2^64`
  (`exists_guess_gt_eps`: above `εPearlC 0 1 = 2^−128` once `Pr[GuessEvent] > 2^−64`), and
  `pr_guessEvent_le`: TT_OUT at W1 on any domain containing the unit gives `Pr[GuessEvent at 2·fetch/(1−γ₀)] ≤
  2^64·ε(0, 1)`.

**(a) The refutation** on the old domain, `¬ TTOutW1 dv salt P (pearlCDomainDevOld d) (1/400) εPearlC`, which is
`TTOutPearlCDevW1` (rev1: `TTOutPearlCDevRev1W1`) as stated before the floor on `n`:
- `not_ttOutPearlCDevW1_old`, `not_ttOutPearlCDevRev1W1_old`: at any record, once `Pr[GuessEvent] > 2^−64`;
- `not_ttOutW1_devAt_old`, `not_ttOutW1Rev1_devAt_old`: at `devAt ⟨[32], wd, z⟩ G pp` with `wd ≥ 24` (sm_120's step
  has 26), the representability conjuncts hold always (`HonestWords`), so the hypotheses are deterministic rules on `a`
  and `w`, `2·fetch/(1 − 1/400) < (1 − ρ)·creditDev`, and `Pr[capOK] > 2^−64`;
- `not_ttOutW1_sm120v1_old` (and rev1, and v2): at `Prices.sm120` and `k = 128` (credit 9,480; rev1 and v2 9,472), for
  `fetch ≤ 2000` and `ρ ≤ 1/2`.
At the H100 (`devH100`, a 14-bit adder) the general forms apply, with the representability conjuncts kept in the event.

**(b) The narrowed domain** proves the program no attack in two exact senses, and no more:
- `oneUnit_not_new`: its workload is outside `pearlCDomainDev`, so TT_OUT's quantifier never reaches it;
- `guess_win_zero_pearlCDev` (and rev1): on every workload in `pearlCDomainDev`, a draw on which the same program wins
  TT_OUT's game at any `T ≥ 0` has a unit whose reference ticket and useful words `(0, j)`, `0 < j < 64`, are all `+0`:
  126 words the program never wrote. How rare such draws are is a property of the noise, which this file does not bound.
`ttOutPearlCDevW1_of_old`: the refuted form implies the narrowed one, so the narrowing only weakens the assumption.
-/

namespace Pouw.PearlC.TTOutW1Item1

open Finset Pouw.Fp8Atom Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-! ## The program -/

/-- One unit of shape `1 × 1 × k`, weight id 0, weights `w`. -/
def oneUnit (k : ℕ) (w : Codes) : Workload := ⟨⟨1, fun _ => ⟨1, 1, k⟩, fun _ => 0⟩, fun _ => w⟩

/-- Constant word 0 to ticket word `(0, 0, 0)` and constant word 1 to useful word `(0, 0, 0)`: three nodes of text and two
constants, within W1's bound. -/
def guessProg (t y : ℕ) : W1.Listing :=
  ⟨fun _ => [t, y], fun _ => .op (.mov (W1.ticketAt 0 0 0) (W1.constAt 0)) (.op (.mov (W1.usefulAt 0 0 0) (W1.constAt 1)) .done),
    fun _ => by norm_num [W1.Code.size]⟩

/-- The guessing adversary: no preprocessing, `guessProg t y` online, activations `a` for every unit. -/
noncomputable def guessAdv (dv : W1.W1Device) (salt : S → List ℕ) (a : Codes) (t y : ℕ) :
    Adversary (W1.w1Priced dv salt : CostModel Q R S) :=
  ⟨.done [], guessProg t y, fun _ => a⟩

/-- `pearlCDomainDev` as stated before the floor on `n`: `n` only positive. -/
def pearlCDomainDevOld (d : PearlCDevice) (L : UnitLayout) : Prop :=
  ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n ∧ 0 < (L.shape u).k ∧ 128 ∣ (L.shape u).k ∧
    (d.G = 0 ∨ 32 * d.G ∣ (L.shape u).k) ∧ (L.shape u).k ≤ 2 ^ 16

/-! ## Its run -/

theorem tk_ne_us (u i j : ℕ) : W1.ticketAt u i j ≠ W1.usefulAt 0 0 0 := by
  intro h
  have h1 := congrArg Prod.fst h
  change (6 : ℕ) = 7 at h1
  omega

theorem us_ne_tk (u i j : ℕ) : W1.usefulAt u i j ≠ W1.ticketAt 0 0 0 := by
  intro h
  have h1 := congrArg Prod.fst h
  change (7 : ℕ) = 6 at h1
  omega

theorem c1_ne_tk : W1.constAt 1 ≠ W1.ticketAt 0 0 0 := by
  intro h
  have h1 := congrArg Prod.fst h
  change (3 : ℕ) = 6 at h1
  omega

theorem act_ne_tk (u i l : ℕ) : W1.actAt u i l ≠ W1.ticketAt 0 0 0 := by
  intro h
  have h1 := congrArg Prod.fst h
  change (4 : ℕ) = 6 at h1
  omega

theorem act_ne_us (u i l : ℕ) : W1.actAt u i l ≠ W1.usefulAt 0 0 0 := by
  intro h
  have h1 := congrArg Prod.fst h
  change (4 : ℕ) = 7 at h1
  omega

theorem pair_ne {u i j : ℕ} (hj : j ≠ 0) : Nat.pair u (Nat.pair i j) ≠ Nat.pair 0 (Nat.pair 0 0) := by
  intro h
  have h1 := congrArg (fun n => (Nat.unpair (Nat.unpair n).2).2) h
  simp only [Nat.unpair_pair] at h1
  exact hj h1

theorem ticketAt_ne {u i j : ℕ} (hj : j ≠ 0) : W1.ticketAt u i j ≠ W1.ticketAt 0 0 0 :=
  fun h => pair_ne hj (congrArg Prod.snd h)

theorem usefulAt_ne {u i j : ℕ} (hj : j ≠ 0) : W1.usefulAt u i j ≠ W1.usefulAt 0 0 0 :=
  fun h => pair_ne hj (congrArg Prod.snd h)

section Run

variable (dv : W1.W1Device) (sl : List ℕ) (t y : ℕ) (U : Workload) (act : ℕ → Codes) (σ : State)

/-- The run reads the two constants, both pre-salt. -/
theorem guess_fetched : (W1.final dv sl (guessProg t y) U act σ).fetched = [W1.constAt 0, W1.constAt 1] := rfl

/-- A move is free. -/
theorem price_mov (d a : W1.Addr) : W1.Instr.price dv false (.mov d a) = 0 := by
  simp [W1.Instr.price, W1.Instr.base, W1.Instr.dynamic]

/-- The moves are free. -/
theorem guess_cost0 : (W1.final dv sl (guessProg t y) U act σ).cost = 0 := by
  show (0 : ℚ) + W1.Instr.price dv false (.mov (W1.ticketAt 0 0 0) (W1.constAt 0)) +
    W1.Instr.price dv false (.mov (W1.usefulAt 0 0 0) (W1.constAt 1)) = 0
  rw [price_mov, price_mov]
  norm_num

/-- Two distinct words fetched. -/
theorem guess_read : W1.w1Read dv sl (guessProg t y) U act σ = 2 := by
  rw [W1.w1Read, guess_fetched]
  decide

/-- **The program's W1 cost is exactly `2·fetch`.** -/
theorem guess_cost (salt : S → List ℕ) (H : Q → R) (s : S) :
    (W1.w1Priced dv salt : CostModel Q R S).cost (guessProg t y) H U act σ s = 2 * (dv.prices.fetch : ℝ) := by
  show (((W1.final dv (salt s) (guessProg t y) U act σ).cost +
    dv.prices.fetch * (W1.w1Read dv (salt s) (guessProg t y) U act σ : ℚ) : ℚ) : ℝ) = _
  rw [guess_cost0, guess_read]
  push_cast
  ring

theorem guess_bounded (salt : S → List ℕ) (a : Codes) :
    (guessAdv (Q := Q) (R := R) dv salt a t y).Bounded U (2 * (dv.prices.fetch : ℝ)) 0 := by
  intro H s
  refine ⟨(guess_cost dv t y U _ _ salt H s).le, ?_⟩
  exact Nat.le_refl 0

/-- The final memory: the two moves on the initial memory. -/
theorem guess_mem : (W1.final dv sl (guessProg t y) U act σ).mem =
    Function.update (Function.update (W1.init sl (guessProg t y) U act σ) (W1.ticketAt 0 0 0)
        (W1.w32 (W1.init sl (guessProg t y) U act σ (W1.constAt 0)))) (W1.usefulAt 0 0 0)
      (W1.w32 (Function.update (W1.init sl (guessProg t y) U act σ) (W1.ticketAt 0 0 0)
        (W1.w32 (W1.init sl (guessProg t y) U act σ (W1.constAt 0))) (W1.constAt 1))) := rfl

theorem init_const0 : W1.init sl (guessProg t y) U act σ (W1.constAt 0) = W1.w32 t := rfl

theorem init_const1 : W1.init sl (guessProg t y) U act σ (W1.constAt 1) = W1.w32 y := rfl

theorem guess_mem_of_ne {z : W1.Addr} (h1 : z ≠ W1.ticketAt 0 0 0) (h2 : z ≠ W1.usefulAt 0 0 0) :
    (W1.final dv sl (guessProg t y) U act σ).mem z = W1.init sl (guessProg t y) U act σ z := by
  rw [guess_mem, Function.update_of_ne h2, Function.update_of_ne h1]

theorem guess_ticket : (W1.final dv sl (guessProg t y) U act σ).mem (W1.ticketAt 0 0 0) = W1.w32 t := by
  rw [guess_mem, Function.update_of_ne (tk_ne_us 0 0 0), Function.update_self, init_const0]
  simp only [W1.w32, Nat.mod_mod]

theorem guess_useful : (W1.final dv sl (guessProg t y) U act σ).mem (W1.usefulAt 0 0 0) = W1.w32 y := by
  rw [guess_mem, Function.update_self, Function.update_of_ne c1_ne_tk, init_const1]
  simp only [W1.w32, Nat.mod_mod]

theorem guess_ticket_zero {u i j : ℕ} (hj : j ≠ 0) :
    (W1.final dv sl (guessProg t y) U act σ).mem (W1.ticketAt u i j) = 0 := by
  rw [guess_mem_of_ne dv sl t y U act σ (ticketAt_ne hj) (tk_ne_us u i j)]
  rfl

theorem guess_useful_zero {u i j : ℕ} (hj : j ≠ 0) :
    (W1.final dv sl (guessProg t y) U act σ).mem (W1.usefulAt u i j) = 0 := by
  rw [guess_mem_of_ne dv sl t y U act σ (us_ne_tk u i j) (usefulAt_ne hj)]
  rfl

theorem init_act (a : Codes) (u i l : ℕ) :
    W1.init sl (guessProg t y) U (fun _ => a) σ (W1.actAt u i l) = W1.w32 (a i l) := by
  simp [W1.init, W1.actAt]

/-- The activations stay as supplied when they are 32-bit words. -/
theorem guess_act (a : Codes) (ha : ∀ i l, a i l < 2 ^ 32) (u i l : ℕ) :
    (W1.final dv sl (guessProg t y) U (fun _ => a) σ).mem (W1.actAt u i l) = a i l := by
  rw [guess_mem_of_ne dv sl t y U _ σ (act_ne_tk u i l) (act_ne_us u i l), init_act, W1.w32,
    Nat.mod_eq_of_lt (ha i l)]

/-- The committed transcript of a declared unit. -/
theorem guess_transcript (salt : S → List ℕ) (a : Codes) (H : Q → R) (s : S) {u : ℕ} (hu : u < U.layout.N) :
    (guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s u =
      some ⟨fun i l => if i < (U.layout.shape u).m ∧ l < (U.layout.shape u).k then
          (W1.final dv (salt s) (guessProg t y) U (fun _ => a) []).mem (W1.actAt u i l) else a i l,
        fun i j => (W1.final dv (salt s) (guessProg t y) U (fun _ => a) []).mem (W1.ticketAt u i j),
        fun i j => (W1.final dv (salt s) (guessProg t y) U (fun _ => a) []).mem (W1.usefulAt u i j)⟩ := by
  show W1.readout U (fun _ => a) (W1.final dv (salt s) (guessProg t y) U (fun _ => a) []).mem u = _
  exact if_pos hu

end Run

/-! ## Its win -/

/-- FP32's value reads the low 32 bits. -/
theorem wordQ_w32 (v : ℕ) : wordQ (W1.w32 v) = wordQ v := by
  have h0 : ((v % 2 ^ 32) >>> 31) % 2 = (v >>> 31) % 2 := by
    simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; omega
  have h1 : ((v % 2 ^ 32) >>> 23) % 256 = (v >>> 23) % 256 := by
    simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; omega
  have h2 : v % 2 ^ 32 % 2 ^ 23 = v % 2 ^ 23 := by simp only [Nat.reducePow]; omega
  unfold W1.w32 Pouw.Fp8Atom.wordQ Pouw.Fp8Atom.fp32Value
  simp only [h0, h1, h2]

/-- **The event** on `(H, s)`: unit `(1, 1, k)` with weights `w` and activations `a` passes the activation, cap and
weight rules, its ticket and useful reference values `(0, 0)` are FP32 words' values, and its credit exceeds `θ`. -/
def GuessEvent (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ) (x : (Q → R) × S) : Prop :=
  P.actOK ⟨1, 1, k⟩ a ∧ P.capOK x.1 x.2 (oneUnit k w) 0 a ∧ P.weightOK ⟨1, 1, k⟩ w 0 ∧
    (∃ t, wordQ t = P.ticket x.1 x.2 (oneUnit k w) 0 a 0 0) ∧ (∃ y, wordQ y = P.useful x.1 x.2 (oneUnit k w) 0 a 0 0) ∧
    θ < P.credit x.1 x.2 (oneUnit k w) 0 a

/-- **The win** of `guessAdv … t y` on `(H, s)`, as TT_OUT's game scores it at `θ = T/(1 − γ₀)`. -/
def GuessWins (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ) (t y : ℕ)
    (x : (Q → R) × S) : Prop :=
  θ < ∑ u ∈ correctSet P (oneUnit k w) x.1 x.2 ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) x.1 x.2),
    P.credit x.1 x.2 (oneUnit k w) u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) x.1 x.2) u)

/-- On the event, the program whose constants are the two reference words wins. -/
theorem wins_of_event {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S} {k : ℕ} {w a : Codes} {θ : ℝ}
    {t y : ℕ} {H : Q → R} {s : S} (ha : ∀ i l, a i l < 2 ^ 32) (hact : P.actOK ⟨1, 1, k⟩ a)
    (hcap : P.capOK H s (oneUnit k w) 0 a) (hwt : P.weightOK ⟨1, 1, k⟩ w 0)
    (ht : wordQ t = P.ticket H s (oneUnit k w) 0 a 0 0) (hy : wordQ y = P.useful H s (oneUnit k w) 0 a 0 0)
    (hθ : θ < P.credit H s (oneUnit k w) 0 a) : GuessWins dv salt P k w a θ t y (H, s) := by
  have hτ : (guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) H s 0 =
      some ⟨a, fun i j => (W1.final dv (salt s) (guessProg t y) (oneUnit k w) (fun _ => a) []).mem (W1.ticketAt 0 i j),
        fun i j => (W1.final dv (salt s) (guessProg t y) (oneUnit k w) (fun _ => a) []).mem (W1.usefulAt 0 i j)⟩ := by
    rw [guess_transcript dv t y (oneUnit k w) salt a H s Nat.one_pos]
    congr 2
    funext i l
    split_ifs <;> first | rfl | exact guess_act dv (salt s) t y (oneUnit k w) [] a ha 0 i l
  have hC : Correct P (oneUnit k w) H s 0
      ⟨a, fun i j => (W1.final dv (salt s) (guessProg t y) (oneUnit k w) (fun _ => a) []).mem (W1.ticketAt 0 i j),
        fun i j => (W1.final dv (salt s) (guessProg t y) (oneUnit k w) (fun _ => a) []).mem (W1.usefulAt 0 i j)⟩ := by
    refine ⟨hact, hcap, fun j hj => ?_, fun i hi j hj => ?_⟩
    · obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
      exact hwt
    · obtain rfl : i = 0 := Nat.lt_one_iff.1 hi
      obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
      exact ⟨(congrArg wordQ (guess_ticket dv (salt s) t y (oneUnit k w) (fun _ => a) [])).trans
          ((wordQ_w32 t).trans ht),
        (congrArg wordQ (guess_useful dv (salt s) t y (oneUnit k w) (fun _ => a) [])).trans ((wordQ_w32 y).trans hy)⟩
  have hK : correctSet P (oneUnit k w) H s ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) H s) =
      {0} := by
    ext u
    simp only [correctSet, Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]
    constructor
    · rintro ⟨hu, -⟩
      exact Nat.lt_one_iff.1 hu
    · rintro rfl
      exact ⟨Nat.one_pos, _, hτ, hC⟩
  have hA : actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) H s) 0 = a := by
    simp only [actOf, hτ]
  show θ < ∑ u ∈ correctSet P (oneUnit k w) H s ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) H s),
    P.credit H s (oneUnit k w) u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript (oneUnit k w) H s) u)
  rw [hK, Finset.sum_singleton, hA]
  exact hθ

theorem w32_lt (v : ℕ) : W1.w32 v < 2 ^ 32 := by
  unfold W1.w32
  exact Nat.mod_lt _ (by norm_num)

/-- The counting below runs over `range N ×ˢ range N` for an abstract `N` and is used at `N = 2^32`: reducing a
membership in a literal `range (2^32)` unfolds `List.range` and exceeds the recursion limit. -/
theorem mem_range_prod {N t y : ℕ} (ht : t < N) (hy : y < N) : (t, y) ∈ range N ×ˢ range N :=
  Finset.mem_product.2 ⟨Finset.mem_range.2 ht, Finset.mem_range.2 hy⟩

/-- On the event, one of the constant pairs below `N` wins. -/
theorem event_mem {N : ℕ} (hN : ∀ v, W1.w32 v < N) {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S}
    {k : ℕ} {w a : Codes} {θ : ℝ} (ha : ∀ i l, a i l < 2 ^ 32) {x : (Q → R) × S} (hx : GuessEvent P k w a θ x) :
    ∃ p ∈ range N ×ˢ range N, GuessWins dv salt P k w a θ p.1 p.2 x := by
  obtain ⟨H, s⟩ := x
  obtain ⟨hact, hcap, hwt, ⟨t, ht⟩, ⟨y, hy⟩, hθ⟩ := hx
  exact ⟨(W1.w32 t, W1.w32 y), mem_range_prod (hN t) (hN y),
    wins_of_event ha hact hcap hwt ((wordQ_w32 t).trans ht) ((wordQ_w32 y).trans hy) hθ⟩

/-- At a step `⟨[32], wd, z⟩` with `wd ≥ 24`, the event is the rules, the cap and the credit bound: the reference words
are always FP32 words' values (`HonestWords`), and within the cap the credit is at least `(1 − ρ)·creditDev`. -/
theorem guessEvent_devAt {wd : ℕ} (hw : 24 ≤ wd) (z : ℤ) (G : ℕ) (pp : Prices) (sem : PearlCSem Q R S) (ρ : ℚ)
    {k : ℕ} (hk : k ≤ 2 ^ 16) {w a : Codes} {θ : ℝ} (hop : sem.OperandOK 1 k a) (hrow : sem.RowOK 1 k w 0)
    (hθ : θ < (((1 - ρ) * creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ : ℚ) : ℝ)) (x : (Q → R) × S)
    (hx : (pearlCProtocolDev (devAt ⟨[32], wd, z⟩ G pp) sem ρ).capOK x.1 x.2 (oneUnit k w) 0 a) :
    GuessEvent (pearlCProtocolDev (devAt ⟨[32], wd, z⟩ G pp) sem ρ) k w a θ x := by
  have hk32 : ((oneUnit k w).layout.shape 0).k / 32 < 2 ^ 24 := by
    show k / 32 < 2 ^ 24
    exact lt_of_le_of_lt ((Nat.div_le_self k 32).trans hk) (by norm_num)
  obtain ⟨ht, hy⟩ := HonestWords.honestWords_devRev1K hw z G pp sem ρ 0 x.1 x.2 (oneUnit k w) 0 a hk32 0 0
  refine ⟨hop, hx, hrow, ht, hy, ?_⟩
  have hc : debitDev (devAt ⟨[32], wd, z⟩ G pp) (sem.unitDebitDev (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a) ≤
      ρ * creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ := hx
  have hq : (1 - ρ) * creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ ≤ creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ -
      debitDev (devAt ⟨[32], wd, z⟩ G pp) (sem.unitDebitDev (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a) := by
    linarith
  show θ < ((creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ -
    debitDev (devAt ⟨[32], wd, z⟩ G pp) (sem.unitDebitDev (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a) : ℚ) : ℝ)
  exact hθ.trans_le (by exact_mod_cast hq)

/-- `guessEvent_devAt` under rev1. -/
theorem guessEvent_devAtRev1 {wd : ℕ} (hw : 24 ≤ wd) (z : ℤ) (G : ℕ) (pp : Prices) (sem : PearlCSem Q R S) (ρ : ℚ)
    {k : ℕ} (hk : k ≤ 2 ^ 16) {w a : Codes} {θ : ℝ} (hop : sem.OperandOK 1 k a) (hrow : sem.RowOK 1 k w 0)
    (hθ : θ < (((1 - ρ) * creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ : ℚ) : ℝ)) (x : (Q → R) × S)
    (hx : (pearlCProtocolDevRev1 (devAt ⟨[32], wd, z⟩ G pp) sem ρ).capOK x.1 x.2 (oneUnit k w) 0 a) :
    GuessEvent (pearlCProtocolDevRev1 (devAt ⟨[32], wd, z⟩ G pp) sem ρ) k w a θ x := by
  have hk32 : ((oneUnit k w).layout.shape 0).k / 32 < 2 ^ 24 := by
    show k / 32 < 2 ^ 24
    exact lt_of_le_of_lt ((Nat.div_le_self k 32).trans hk) (by norm_num)
  obtain ⟨ht, hy⟩ := HonestWords.honestWords_devRev1K hw z G pp sem ρ 0 x.1 x.2 (oneUnit k w) 0 a hk32 0 0
  refine ⟨hop, hx, hrow, ht, hy, ?_⟩
  have hc : sem.unitDebitRev1 (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a ≤
      ρ * creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ := hx
  have hq : (1 - ρ) * creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ ≤
      creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ -
        sem.unitDebitRev1 (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a := by
    linarith
  show θ < ((creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ -
    sem.unitDebitRev1 (devAt ⟨[32], wd, z⟩ G pp) x.1 x.2 (oneUnit k w) 0 a : ℚ) : ℝ)
  exact hθ.trans_le (by exact_mod_cast hq)

/-! ## Its probability -/

section Prob

variable [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem pr_event_le_sum {N : ℕ} (hN : ∀ v, W1.w32 v < N) (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ) (ha : ∀ i l, a i l < 2 ^ 32) :
    pr (GuessEvent P k w a θ) ≤ ∑ p ∈ range N ×ˢ range N, pr (GuessWins dv salt P k w a θ p.1 p.2) :=
  calc pr (GuessEvent P k w a θ)
      ≤ pr (fun x => ∃ p ∈ range N ×ˢ range N, GuessWins dv salt P k w a θ p.1 p.2 x) :=
        pr_mono fun _ hx => event_mem hN ha hx
    _ ≤ _ := Pouw.Proofs.pr_exists_le_sum _ _

theorem exists_guess_ge_of {N : ℕ} (hN : ∀ v, W1.w32 v < N) (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ) (ha : ∀ i l, a i l < 2 ^ 32) :
    ∃ t < N, ∃ y < N, pr (GuessEvent P k w a θ) / ((N : ℝ) * N) ≤ pr (GuessWins dv salt P k w a θ t y) := by
  have hN0 : 0 < N := lt_of_le_of_lt (Nat.zero_le _) (hN 0)
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hN0.ne'
  have hM : (N : ℝ) * N ≠ 0 := mul_ne_zero hN' hN'
  have h3 : ∑ _p ∈ range N ×ˢ range N, pr (GuessEvent P k w a θ) / ((N : ℝ) * N) = pr (GuessEvent P k w a θ) := by
    rw [Finset.sum_const, Finset.card_product, Finset.card_range, nsmul_eq_mul, Nat.cast_mul, mul_comm,
      div_mul_eq_mul_div, mul_div_assoc, div_self hM, mul_one]
  have hS : (range N ×ˢ range N).Nonempty := ⟨(0, 0), mem_range_prod hN0 hN0⟩
  obtain ⟨p, hp, h⟩ := Finset.exists_le_of_sum_le hS (h3.trans_le (pr_event_le_sum hN dv salt P k w a θ ha))
  rw [Finset.mem_product, Finset.mem_range, Finset.mem_range] at hp
  exact ⟨p.1, hp.1, p.2, hp.2, h⟩

/-- **The success probability**: some constants `t, y < 2^32` win with probability at least `Pr[GuessEvent]/2^64`. -/
theorem exists_guess_ge (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ)
    (ha : ∀ i l, a i l < 2 ^ 32) :
    ∃ t < 2 ^ 32, ∃ y < 2 ^ 32, pr (GuessEvent P k w a θ) / 2 ^ 64 ≤ pr (GuessWins dv salt P k w a θ t y) := by
  obtain ⟨t, ht, y, hy, h⟩ := exists_guess_ge_of w32_lt dv salt P k w a θ ha
  have e : ((2 ^ 32 : ℕ) : ℝ) * ((2 ^ 32 : ℕ) : ℝ) = 2 ^ 64 := by norm_num
  rw [e] at h
  exact ⟨t, ht, y, hy, h⟩

/-- **The program beats `εPearlC`**: when `Pr[GuessEvent] > 2^−64`, some constants win with probability above
`εPearlC 0 1 = 2^−128`. -/
theorem exists_guess_gt_eps (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (k : ℕ) (w a : Codes) (θ : ℝ)
    (ha : ∀ i l, a i l < 2 ^ 32) (hE : 1 / 2 ^ 64 < pr (GuessEvent P k w a θ)) :
    ∃ t < 2 ^ 32, ∃ y < 2 ^ 32, εPearlC 0 1 < pr (GuessWins dv salt P k w a θ t y) := by
  obtain ⟨t, ht, y, hy, h⟩ := exists_guess_ge dv salt P k w a θ ha
  refine ⟨t, ht, y, hy, lt_of_lt_of_le ?_ h⟩
  have e : εPearlC 0 1 = 1 / 2 ^ 64 / 2 ^ 64 := by norm_num [εPearlC]
  rw [e]
  exact div_lt_div_of_pos_right hE (by positivity)

/-- **TT_OUT at W1 bounds the event**: on any domain containing the unit, `Pr[GuessEvent at 2·fetch/(1−γ₀)]` is at most
`2^64·ε(0, 1)`. -/
theorem pr_guessEvent_le {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S} {D : UnitLayout → Prop} {γ₀ : ℝ}
    {ε : ℕ → ℕ → ℝ} {k : ℕ} {w a : Codes} (hD : D (oneUnit k w).layout) (hf : 0 ≤ dv.prices.fetch)
    (ha : ∀ i l, a i l < 2 ^ 32) (h : TTOutW1 dv salt P D γ₀ ε) :
    pr (GuessEvent P k w a (2 * (dv.prices.fetch : ℝ) / (1 - γ₀))) ≤ 2 ^ 64 * ε 0 1 := by
  have hU : (oneUnit k w).InDomain P D := ⟨by show (1 : ℕ) ≤ 2 ^ 64; norm_num, hD⟩
  have hT : (0 : ℝ) ≤ 2 * (dv.prices.fetch : ℝ) := by
    have : (0 : ℝ) ≤ dv.prices.fetch := by exact_mod_cast hf
    linarith
  have hW : ∀ p ∈ range (2 ^ 32) ×ˢ range (2 ^ 32),
      pr (GuessWins dv salt P k w a (2 * (dv.prices.fetch : ℝ) / (1 - γ₀)) p.1 p.2) ≤ ε 0 1 :=
    fun p _ => h (oneUnit k w) hU _ hT 0 (Nat.zero_le _) (guessAdv dv salt a p.1 p.2)
      (guess_bounded dv p.1 p.2 (oneUnit k w) salt a)
  calc pr (GuessEvent P k w a (2 * (dv.prices.fetch : ℝ) / (1 - γ₀)))
      ≤ ∑ p ∈ range (2 ^ 32) ×ˢ range (2 ^ 32),
          pr (GuessWins dv salt P k w a (2 * (dv.prices.fetch : ℝ) / (1 - γ₀)) p.1 p.2) :=
        pr_event_le_sum w32_lt dv salt P k w a _ ha
    _ ≤ ∑ _p ∈ range (2 ^ 32) ×ˢ range (2 ^ 32), ε 0 1 := Finset.sum_le_sum hW
    _ = 2 ^ 64 * ε 0 1 := by
        rw [Finset.sum_const, Finset.card_product, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

/-- **The refutation, in general**: if the event has probability above `2^64·ε(0, 1)`, TT_OUT at W1 fails on any domain
containing the unit. -/
theorem not_ttOutW1_of_guess {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S} {D : UnitLayout → Prop}
    {γ₀ : ℝ} {ε : ℕ → ℕ → ℝ} {k : ℕ} {w a : Codes} (hD : D (oneUnit k w).layout) (hf : 0 ≤ dv.prices.fetch)
    (ha : ∀ i l, a i l < 2 ^ 32) (hE : 2 ^ 64 * ε 0 1 < pr (GuessEvent P k w a (2 * (dv.prices.fetch : ℝ) / (1 - γ₀)))) :
    ¬ TTOutW1 dv salt P D γ₀ ε :=
  fun h => not_le.2 hE (pr_guessEvent_le hD hf ha h)

/-! ## (a) Pearl-C on the old domain -/

theorem oneUnit_old {d : PearlCDevice} {k : ℕ} (w : Codes) (hk0 : 0 < k) (h128 : 128 ∣ k)
    (hG : d.G = 0 ∨ 32 * d.G ∣ k) (hk : k ≤ 2 ^ 16) : pearlCDomainDevOld d (oneUnit k w).layout :=
  fun _ _ => ⟨Nat.one_pos, Nat.one_pos, hk0, h128, hG, hk⟩

theorem eps_guess : (2 : ℝ) ^ 64 * εPearlC 0 1 = 1 / 2 ^ 64 := by
  norm_num [εPearlC]

/-- **TT_OUT(1/400) at W1 on the old domain fails** at any record once `Pr[GuessEvent] > 2^−64`. -/
theorem not_ttOutPearlCDevW1_old {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S}
    {ρ : ℚ} {k : ℕ} {w a : Codes} (hk0 : 0 < k) (h128 : 128 ∣ k) (hG : d.G = 0 ∨ 32 * d.G ∣ k) (hk : k ≤ 2 ^ 16)
    (hf : 0 ≤ dv.prices.fetch) (ha : ∀ i l, a i l < 2 ^ 32)
    (hE : 1 / 2 ^ 64 < pr (GuessEvent (pearlCProtocolDev d sem ρ) k w a (2 * (dv.prices.fetch : ℝ) / (1 - 1 / 400)))) :
    ¬ TTOutW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCDomainDevOld d) (1 / 400) εPearlC :=
  not_ttOutW1_of_guess (oneUnit_old w hk0 h128 hG hk) hf ha (eps_guess.trans_lt hE)

/-- **TT_OUT(1/400) rev1 at W1 on the old domain fails** at any record once `Pr[GuessEvent] > 2^−64`. -/
theorem not_ttOutPearlCDevRev1W1_old {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} {k : ℕ} {w a : Codes} (hk0 : 0 < k) (h128 : 128 ∣ k)
    (hG : d.G = 0 ∨ 32 * d.G ∣ k) (hk : k ≤ 2 ^ 16) (hf : 0 ≤ dv.prices.fetch) (ha : ∀ i l, a i l < 2 ^ 32)
    (hE : 1 / 2 ^ 64 <
      pr (GuessEvent (pearlCProtocolDevRev1 d sem ρ) k w a (2 * (dv.prices.fetch : ℝ) / (1 - 1 / 400)))) :
    ¬ TTOutW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevOld d) (1 / 400) εPearlC :=
  not_ttOutW1_of_guess (oneUnit_old w hk0 h128 hG hk) hf ha (eps_guess.trans_lt hE)

/-- **The refutation at a step `⟨[32], wd, z⟩` with `wd ≥ 24`**: activations and weights that pass the rules, a fetch
price below `(1 − 1/400)·(1 − ρ)·creditDev/2`, and the cap holding with probability above `2^−64`. -/
theorem not_ttOutW1_devAt_old {dv : W1.W1Device} {salt : S → List ℕ} {wd : ℕ} (hw : 24 ≤ wd) (z : ℤ) (G : ℕ)
    (pp : Prices) (sem : PearlCSem Q R S) (ρ : ℚ) {k : ℕ} {w a : Codes} (hk0 : 0 < k) (h128 : 128 ∣ k)
    (hG : G = 0 ∨ 32 * G ∣ k) (hk : k ≤ 2 ^ 16) (hf : 0 ≤ dv.prices.fetch) (ha : ∀ i l, a i l < 2 ^ 32)
    (hop : sem.OperandOK 1 k a) (hrow : sem.RowOK 1 k w 0)
    (hθ : 2 * (dv.prices.fetch : ℝ) / (1 - 1 / 400) <
      (((1 - ρ) * creditDev (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ : ℚ) : ℝ))
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDev (devAt ⟨[32], wd, z⟩ G pp) sem ρ).capOK x.1 x.2 (oneUnit k w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDev (devAt ⟨[32], wd, z⟩ G pp) sem ρ)
      (pearlCDomainDevOld (devAt ⟨[32], wd, z⟩ G pp)) (1 / 400) εPearlC :=
  not_ttOutPearlCDevW1_old hk0 h128 hG hk hf ha
    (hcap.trans_le (pr_mono fun x hx => guessEvent_devAt hw z G pp sem ρ hk hop hrow hθ x hx))

/-- `not_ttOutW1_devAt_old` under rev1. -/
theorem not_ttOutW1Rev1_devAt_old {dv : W1.W1Device} {salt : S → List ℕ} {wd : ℕ} (hw : 24 ≤ wd) (z : ℤ) (G : ℕ)
    (pp : Prices) (sem : PearlCSem Q R S) (ρ : ℚ) {k : ℕ} {w a : Codes} (hk0 : 0 < k) (h128 : 128 ∣ k)
    (hG : G = 0 ∨ 32 * G ∣ k) (hk : k ≤ 2 ^ 16) (hf : 0 ≤ dv.prices.fetch) (ha : ∀ i l, a i l < 2 ^ 32)
    (hop : sem.OperandOK 1 k a) (hrow : sem.RowOK 1 k w 0)
    (hθ : 2 * (dv.prices.fetch : ℝ) / (1 - 1 / 400) <
      (((1 - ρ) * creditDevRev1 (devAt ⟨[32], wd, z⟩ G pp) ⟨1, 1, k⟩ : ℚ) : ℝ))
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDevRev1 (devAt ⟨[32], wd, z⟩ G pp) sem ρ).capOK x.1 x.2 (oneUnit k w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDevRev1 (devAt ⟨[32], wd, z⟩ G pp) sem ρ)
      (pearlCDomainDevOld (devAt ⟨[32], wd, z⟩ G pp)) (1 / 400) εPearlC :=
  not_ttOutPearlCDevRev1W1_old hk0 h128 hG hk hf ha
    (hcap.trans_le (pr_mono fun x hx => guessEvent_devAtRev1 hw z G pp sem ρ hk hop hrow hθ x hx))

theorem credit_sm120v1 : creditDev (devSm120v1 Prices.sm120) ⟨1, 1, 128⟩ = 9480 := by
  norm_num [creditDev, devSm120v1, devAt, Prices.sm120, Params.pi]

theorem creditRev1_sm120v1 : creditDevRev1 (devSm120v1 Prices.sm120) ⟨1, 1, 128⟩ = 9472 := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devSm120v1, devAt, Prices.sm120, Params.pi]

theorem credit_sm120v2 : creditDev (devSm120v2 Prices.sm120) ⟨1, 1, 128⟩ = 9472 := by
  norm_num [creditDev, devSm120v2, devAt, Prices.sm120, Params.pi]

theorem creditRev1_sm120v2 : creditDevRev1 (devSm120v2 Prices.sm120) ⟨1, 1, 128⟩ = 9472 := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devSm120v2, devAt, Prices.sm120, Params.pi]

/-- At `fetch ≤ 2000`, `ρ ≤ 1/2` and a credit of at least 9,472, the program's `2·fetch/(1 − 1/400) ≈ 4,010` is below
`(1 − ρ)·credit ≥ 4,736`. -/
theorem guess_cost_lt {f ρ c : ℚ} (hf : f ≤ 2000) (hρ : ρ ≤ 1 / 2) (hc : 9472 ≤ c) :
    2 * (f : ℝ) / (1 - 1 / 400) < (((1 - ρ) * c : ℚ) : ℝ) := by
  have e : 2 * f / (1 - 1 / 400) = 800 / 399 * f := by ring
  have h1 : 4736 ≤ (1 - ρ) * c := by nlinarith
  have h : 2 * f / (1 - 1 / 400) < (1 - ρ) * c := by
    rw [e]
    linarith
  have h' : ((2 * f / (1 - 1 / 400) : ℚ) : ℝ) < (((1 - ρ) * c : ℚ) : ℝ) := Rat.cast_lt.2 h
  have e' : ((2 * f / (1 - 1 / 400) : ℚ) : ℝ) = 2 * (f : ℝ) / (1 - 1 / 400) := by
    push_cast
    ring1
  exact e'.symm.trans_lt h'

/-- **sm_120 v1 at `Prices.sm120`, `k = 128`**: the old-domain TT_OUT at W1 fails for any fetch price up to 2,000, any
cap `ρ ≤ 1/2`, activations and a weight row that pass the rules, and a cap that holds with probability above `2^−64`. -/
theorem not_ttOutW1_sm120v1_old {dv : W1.W1Device} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ} {w a : Codes}
    (hf0 : 0 ≤ dv.prices.fetch) (hf : dv.prices.fetch ≤ 2000) (hρ : ρ ≤ 1 / 2) (ha : ∀ i l, a i l < 2 ^ 32)
    (hop : sem.OperandOK 1 128 a) (hrow : sem.RowOK 1 128 w 0)
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDev (devSm120v1 Prices.sm120) sem ρ).capOK x.1 x.2 (oneUnit 128 w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDev (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevOld (devSm120v1 Prices.sm120)) (1 / 400) εPearlC := by
  have hc : (9472 : ℚ) ≤ creditDev (devSm120v1 Prices.sm120) ⟨1, 1, 128⟩ := by rw [credit_sm120v1]; norm_num
  exact not_ttOutW1_devAt_old (wd := 26) (by norm_num) (-133) 4 Prices.sm120 sem ρ (by norm_num) (by norm_num)
    (Or.inr (by norm_num)) (by norm_num) hf0 ha hop hrow (guess_cost_lt hf hρ hc) hcap

/-- `not_ttOutW1_sm120v1_old` under rev1. -/
theorem not_ttOutW1Rev1_sm120v1_old {dv : W1.W1Device} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    {w a : Codes} (hf0 : 0 ≤ dv.prices.fetch) (hf : dv.prices.fetch ≤ 2000) (hρ : ρ ≤ 1 / 2)
    (ha : ∀ i l, a i l < 2 ^ 32) (hop : sem.OperandOK 1 128 a) (hrow : sem.RowOK 1 128 w 0)
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ).capOK x.1 x.2 (oneUnit 128 w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevOld (devSm120v1 Prices.sm120)) (1 / 400) εPearlC := by
  have hc : (9472 : ℚ) ≤ creditDevRev1 (devSm120v1 Prices.sm120) ⟨1, 1, 128⟩ := creditRev1_sm120v1.ge
  exact not_ttOutW1Rev1_devAt_old (wd := 26) (by norm_num) (-133) 4 Prices.sm120 sem ρ (by norm_num) (by norm_num)
    (Or.inr (by norm_num)) (by norm_num) hf0 ha hop hrow (guess_cost_lt hf hρ hc) hcap

/-- sm_120 v2 (no promotion) at `Prices.sm120`, `k = 128`. -/
theorem not_ttOutW1_sm120v2_old {dv : W1.W1Device} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ} {w a : Codes}
    (hf0 : 0 ≤ dv.prices.fetch) (hf : dv.prices.fetch ≤ 2000) (hρ : ρ ≤ 1 / 2) (ha : ∀ i l, a i l < 2 ^ 32)
    (hop : sem.OperandOK 1 128 a) (hrow : sem.RowOK 1 128 w 0)
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDev (devSm120v2 Prices.sm120) sem ρ).capOK x.1 x.2 (oneUnit 128 w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDev (devSm120v2 Prices.sm120) sem ρ)
      (pearlCDomainDevOld (devSm120v2 Prices.sm120)) (1 / 400) εPearlC := by
  have hc : (9472 : ℚ) ≤ creditDev (devSm120v2 Prices.sm120) ⟨1, 1, 128⟩ := credit_sm120v2.ge
  exact not_ttOutW1_devAt_old (wd := 26) (by norm_num) (-133) 0 Prices.sm120 sem ρ (by norm_num) (by norm_num)
    (Or.inl rfl) (by norm_num) hf0 ha hop hrow (guess_cost_lt hf hρ hc) hcap

/-- sm_120 v2 under rev1. -/
theorem not_ttOutW1Rev1_sm120v2_old {dv : W1.W1Device} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    {w a : Codes} (hf0 : 0 ≤ dv.prices.fetch) (hf : dv.prices.fetch ≤ 2000) (hρ : ρ ≤ 1 / 2)
    (ha : ∀ i l, a i l < 2 ^ 32) (hop : sem.OperandOK 1 128 a) (hrow : sem.RowOK 1 128 w 0)
    (hcap : 1 / 2 ^ 64 < pr fun x : (Q → R) × S =>
      (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120) sem ρ).capOK x.1 x.2 (oneUnit 128 w) 0 a) :
    ¬ TTOutW1 dv salt (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120) sem ρ)
      (pearlCDomainDevOld (devSm120v2 Prices.sm120)) (1 / 400) εPearlC := by
  have hc : (9472 : ℚ) ≤ creditDevRev1 (devSm120v2 Prices.sm120) ⟨1, 1, 128⟩ := creditRev1_sm120v2.ge
  exact not_ttOutW1Rev1_devAt_old (wd := 26) (by norm_num) (-133) 0 Prices.sm120 sem ρ (by norm_num) (by norm_num)
    (Or.inl rfl) (by norm_num) hf0 ha hop hrow (guess_cost_lt hf hρ hc) hcap

end Prob

/-! ## (b) The narrowed domain -/

/-- **The program's workload is outside the narrowed domain.** -/
theorem oneUnit_not_new (d : PearlCDevice) (k : ℕ) (w : Codes) : ¬ pearlCDomainDev d (oneUnit k w).layout := by
  intro h
  have h64 : 64 ≤ 1 := (h 0 Nat.one_pos).2.1.1
  omega

theorem pearlCDomainDevOld_of_new {d : PearlCDevice} {L : UnitLayout} (h : pearlCDomainDev d L) : pearlCDomainDevOld d L :=
  fun u hu => by
    obtain ⟨h1, ⟨h2, -⟩, h3⟩ := h u hu
    exact ⟨h1, by omega, h3⟩

theorem ttOutW1_mono_domain [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {dv : W1.W1Device}
    {salt : S → List ℕ} {P : Protocol Q R S} {D D' : UnitLayout → Prop} {γ₀ : ℝ} {ε : ℕ → ℕ → ℝ}
    (hD : ∀ L, D' L → D L) (h : TTOutW1 dv salt P D γ₀ ε) : TTOutW1 dv salt P D' γ₀ ε :=
  fun U hU => h U ⟨hU.1, hD _ hU.2⟩

/-- The refuted form implies the narrowed one: the narrowing only weakens the assumption. -/
theorem ttOutPearlCDevW1_of_old [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {dv : W1.W1Device}
    {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S} {ρ : ℚ}
    (h : TTOutW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCDomainDevOld d) (1 / 400) εPearlC) :
    TTOutPearlCDevW1 dv salt d sem ρ :=
  ttOutW1_mono_domain (fun _ => pearlCDomainDevOld_of_new) h

theorem ttOutPearlCDevRev1W1_of_old [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {dv : W1.W1Device}
    {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S} {ρ : ℚ}
    (h : TTOutW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevOld d) (1 / 400) εPearlC) :
    TTOutPearlCDevRev1W1 dv salt d sem ρ :=
  ttOutW1_mono_domain (fun _ => pearlCDomainDevOld_of_new) h

/-- A correct unit of the program's transcript has reference value `+0` at every word `(i, j)`, `j ≠ 0`, of its shape:
the program never writes one. -/
theorem guess_zero_words {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S} {U : Workload} {a : Codes}
    {t y : ℕ} {H : Q → R} {s : S} {u : ℕ}
    (hu : u ∈ correctSet P U H s ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s)) {i j : ℕ}
    (hi : i < (U.layout.shape u).m) (hj : j < (U.layout.shape u).n) (hj0 : j ≠ 0) :
    P.ticket H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) i j = some 0 ∧
      P.useful H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) i j = some 0 := by
  simp only [correctSet, Finset.mem_filter, Finset.mem_range] at hu
  obtain ⟨huN, c, hc, hC⟩ := hu
  obtain ⟨-, -, -, hw⟩ := hC
  have hc' := Option.some.inj (hc.symm.trans (guess_transcript dv t y U salt a H s huN))
  have hA : actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u = c.act := by
    simp only [actOf, hc]
  rw [hA]
  have ht0 : c.ticket i j = 0 := by
    rw [hc']
    exact guess_ticket_zero dv (salt s) t y U (fun _ => a) [] hj0
  have hu0 : c.useful i j = 0 := by
    rw [hc']
    exact guess_useful_zero dv (salt s) t y U (fun _ => a) [] hj0
  obtain ⟨h1, h2⟩ := hw i hi j hj
  rw [ht0, Freeze.wordQ_zero] at h1
  rw [hu0, Freeze.wordQ_zero] at h2
  exact ⟨h1.symm, h2.symm⟩

/-- **On a domain of units with `m > 0` and `n ≥ 64`**, a draw on which the program wins at any `θ ≥ 0` has a unit whose
reference words `(0, j)`, `0 < j < 64`, ticket and useful, are all `+0`. -/
theorem guess_win_zero {dv : W1.W1Device} {salt : S → List ℕ} {P : Protocol Q R S} {U : Workload} {a : Codes}
    {t y : ℕ} {H : Q → R} {s : S} {θ : ℝ}
    (hn : ∀ u < U.layout.N, 0 < (U.layout.shape u).m ∧ 64 ≤ (U.layout.shape u).n) (hθ : 0 ≤ θ)
    (hwin : θ < ∑ u ∈ correctSet P U H s ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s),
      P.credit H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u)) :
    ∃ u < U.layout.N, ∀ j, 0 < j → j < 64 →
      P.ticket H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 ∧
        P.useful H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 := by
  obtain ⟨u, hu⟩ : (correctSet P U H s ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s)).Nonempty := by
    by_contra h
    rw [Finset.not_nonempty_iff_eq_empty] at h
    rw [h, Finset.sum_empty] at hwin
    linarith
  have huN : u < U.layout.N := by
    have h := hu
    simp only [correctSet, Finset.mem_filter, Finset.mem_range] at h
    exact h.1
  exact ⟨u, huN, fun j hj0 hj =>
    guess_zero_words hu (hn u huN).1 (lt_of_lt_of_le hj (hn u huN).2) (by omega)⟩

/-- **(b) for Pearl-C at any record**: on a workload in the narrowed domain, a draw on which the program wins TT_OUT's game
at `T ≥ 0` has a unit with 126 reference words `+0` (63 ticket and 63 useful, row 0, columns 1 to 63). -/
theorem guess_win_zero_pearlCDev {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S}
    {ρ : ℚ} {U : Workload} (hD : pearlCDomainDev d U.layout) {a : Codes} {t y : ℕ} {H : Q → R} {s : S} {T : ℝ}
    (hT : 0 ≤ T)
    (hwin : T / (1 - 1 / 400) < ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s
        ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s),
      (pearlCProtocolDev d sem ρ).credit H s U u (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u)) :
    ∃ u < U.layout.N, ∀ j, 0 < j → j < 64 →
      (pearlCProtocolDev d sem ρ).ticket H s U u
          (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 ∧
        (pearlCProtocolDev d sem ρ).useful H s U u
          (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 :=
  guess_win_zero (fun u hu => ⟨(hD u hu).1, (hD u hu).2.1.1⟩) (div_nonneg hT (by norm_num)) hwin

/-- `guess_win_zero_pearlCDev` under rev1. -/
theorem guess_win_zero_pearlCDevRev1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} {U : Workload} (hD : pearlCDomainDev d U.layout) {a : Codes} {t y : ℕ}
    {H : Q → R} {s : S} {T : ℝ} (hT : 0 ≤ T)
    (hwin : T / (1 - 1 / 400) < ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) U H s
        ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s),
      (pearlCProtocolDevRev1 d sem ρ).credit H s U u
        (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u)) :
    ∃ u < U.layout.N, ∀ j, 0 < j → j < 64 →
      (pearlCProtocolDevRev1 d sem ρ).ticket H s U u
          (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 ∧
        (pearlCProtocolDevRev1 d sem ρ).useful H s U u
          (actOf ((guessAdv (Q := Q) (R := R) dv salt a t y).transcript U H s) u) 0 j = some 0 :=
  guess_win_zero (fun u hu => ⟨(hD u hu).1, (hD u hu).2.1.1⟩) (div_nonneg hT (by norm_num)) hwin

end Pouw.PearlC.TTOutW1Item1
