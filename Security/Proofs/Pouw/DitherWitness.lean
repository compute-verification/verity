import Proofs.Pouw.Dither

/-!
# The headline's witness (RT-LEAN L7): the words side

A machine in which the words cost computation and cheap programs exist, and where `SAKMatmul` holds by the check's
sampling rather than by the machine's prices.

* `tm`, `ta`: a toy machine over three resources (`0` the tensor cores' MAC units, `1` the hashing units, `2` the
  rest). A program is any function with its use, and the toy's one law is that a program which outputs words uses a
  MAC unit for each word that is some dither's honest word: words come only from a "compute word" step. The program
  that outputs no honest word is free, and one that computes a single word pays for one.
* `tround`: the toy round's words side, with two words, one dither value per word, and a check that draws one word
  (`s = 1 < 2`) and passes where it is the honest one. The hash side is a parameter.
* `sak_toy`: `SAKMatmul` at `c = 1` and `η = s = 1` against the honest words' cost `2 μ₀`. A program charged for `j`
  words passes with probability at most `j/2` and has shortfall at most `1 − j/2`, and `x e^{1−x} ≤ 1`.
-/

namespace Pouw.SecurityProofs.DitherWitness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

/-- The toy round's dither: one of two values per word. -/
abbrev TDraw : Type := Fin 2 → Fin 2

/-- The toy round's words. -/
abbrev TTr : Type := Fin 2 → ℕ

/-- The honest words at the dither `d`: `1` or `2` each, so a word that is neither is no dither's. -/
def honest (d : TDraw) : TTr := fun i => (d i).val + 1

/-- The words of `T` that are some dither's honest word: the machine charges a MAC unit for each. -/
def charged (T : TTr) : Finset (Fin 2) := univ.filter fun i => T i = 1 ∨ T i = 2

theorem card_charged_le (T : TTr) : ((charged T).card : ℝ) ≤ 2 := by
  have h := card_le_univ (charged T)
  rw [Fintype.card_fin] at h
  exact_mod_cast h

/-- **A toy program**: what it computes, its use of the three resources (nonnegative), and the toy's law: a program
that outputs words uses a MAC unit for each word that is some dither's honest word. -/
structure TAlg (I O : Type) where
  run : I → O
  use : I → Fin 3 → ℝ
  use_nonneg : ∀ x r, 0 ≤ use x r
  mac : ∀ (h : O = TTr) (x : I), ((charged (cast h (run x))).card : ℝ) ≤ use x 0

/-- Any function, at the use `u` plus two units of every resource (which pays for any two words). -/
def TAlg.ofFun {I O : Type} (f : I → O) (u : I → Fin 3 → ℝ) (hu : ∀ x r, 0 ≤ u x r) : TAlg I O where
  run := f
  use x r := 2 + u x r
  use_nonneg x r := by linarith [hu x r]
  mac h x := by linarith [card_charged_le (cast h (f x)), hu x 0]

/-- The toy machine. Its `time` is zero (the headline reads `Accel.use`, not `time`), and composing or routing
programs pays two units of every resource, the most any program's words can cost. -/
def tm : Machine where
  Alg I O := TAlg I O
  eval f := f.run
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := TAlg.ofFun (g.run ∘ f.run) (fun x r => f.use x r + g.use (f.run x) r)
    fun x r => add_nonneg (f.use_nonneg x r) (g.use_nonneg _ r)
  pair f g := TAlg.ofFun (fun x => (f.run x, g.run x)) (fun x r => f.use x r + g.use x r)
    fun x r => add_nonneg (f.use_nonneg x r) (g.use_nonneg x r)
  fst := TAlg.ofFun Prod.fst (fun _ _ => 0) fun _ _ => le_rfl
  snd := TAlg.ofFun Prod.snd (fun _ _ => 0) fun _ _ => le_rfl
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- The toy accelerator: three resources at rate 1, a run's wall time its total use, nothing imported. -/
noncomputable def ta : Accel where
  Mc := tm
  Res := Fin 3
  rate _ := 1
  mem := 0
  use f x r := TAlg.use f x r
  held _ _ := 0
  wall f x := ∑ r, TAlg.use f x r
  ext _ _ := 0
  P1 := True
  P2 := True

theorem ta_G1 : ta.G1 := fun _ _ => by
  show (0 : ℝ) + 0 ≤ 0
  norm_num

theorem ta_G2 : ta.G2 := by
  refine ⟨fun _ => one_pos, fun {I O} f x => ⟨le_rfl, fun r => ⟨TAlg.use_nonneg f x r, ?_⟩⟩⟩
  show TAlg.use f x r ≤ 1 * ∑ r, TAlg.use f x r
  rw [one_mul]
  exact single_le_sum (f := fun r => TAlg.use f x r) (fun r _ => TAlg.use_nonneg f x r) (mem_univ r)

/-- **The toy round, words side**: no instance, one dither value per word, and a check that draws one of the two
words (`s = 1`) and passes where it is the honest word. The oracle, the commitment and the answer are parameters. -/
def tround (Orc : Type) [Fintype Orc] (Root : Type) (commit : Orc → TTr → Root) (Ans : Type)
    (accept : Orc → Unit → TDraw → Fin 2 → Root → Ans → Prop) : RoundProto where
  Inst := Unit
  Draw := TDraw
  Chk := Fin 2
  Orc := Orc
  Tr := TTr
  Root := Root
  commit := commit
  ok _ d c T := T c = honest d c
  Ans := Ans
  accept := accept

/-- `n e^{1 − n/2} ≤ 2` on `[0, 2]`: `x e^{1−x} ≤ 1` on `[0, 1]`, from `x ≤ e^{x−1}`. -/
theorem mul_exp_le_two (n : ℝ) : n * Real.exp (1 - n / 2) ≤ 2 := by
  have h := mul_le_mul_of_nonneg_right (Real.add_one_le_exp (n / 2 - 1)) (Real.exp_pos (1 - n / 2)).le
  rw [← Real.exp_add, show n / 2 - 1 + (1 - n / 2) = 0 by ring, Real.exp_zero, sub_add_cancel] at h
  calc n * Real.exp (1 - n / 2) = 2 * (n / 2 * Real.exp (1 - n / 2)) := by ring
    _ ≤ 2 * 1 := mul_le_mul_of_nonneg_left h two_pos.le
    _ = 2 := mul_one 2

/-- **`SAKMatmul` holds on the toy round by sampling** at `c = 1`, `η = 1` and any import weight, against the honest
words' cost `2 μ₀` (two words, a MAC unit each). -/
theorem sak_toy (Orc : Type) [Fintype Orc] (Root : Type) (commit : Orc → TTr → Root) (Ans : Type)
    (accept : Orc → Unit → TDraw → Fin 2 → Root → Ans → Prop) (μ : Fin 3 → ℝ) (hμ : ∀ r, 0 ≤ μ r) (hμ0 : 0 < μ 0)
    (β : ℝ) : SAKMatmul ta (tround Orc Root commit Ans accept) μ 1 1 β (2 * μ 0) := by
  intro _ _ _ _ a St s Mw
  have hmem : ∀ (d : TDraw) (c : Fin 2), TAlg.run Mw (s, d) c = honest d c → c ∈ charged (TAlg.run Mw (s, d)) := by
    intro d c h
    simp only [charged, mem_filter, mem_univ, true_and, h, honest]
    have := (d c).isLt
    omega
  have hsf : ∀ d : TDraw, roundShortfall 1 β (2 * μ 0) (ta.cost μ Mw (s, d)) (ta.ext Mw (s, d)) ≤
      1 - ((charged (TAlg.run Mw (s, d))).card : ℝ) / 2 := by
    intro d
    have hS := card_charged_le (TAlg.run Mw (s, d))
    have hcost : μ 0 * ((charged (TAlg.run Mw (s, d))).card : ℝ) ≤ ta.cost μ Mw (s, d) :=
      calc μ 0 * ((charged (TAlg.run Mw (s, d))).card : ℝ) ≤ μ 0 * TAlg.use Mw (s, d) 0 :=
            mul_le_mul_of_nonneg_left (TAlg.mac Mw rfl (s, d)) (hμ 0)
        _ ≤ ∑ r, μ r * TAlg.use Mw (s, d) r :=
            single_le_sum (f := fun r => μ r * TAlg.use Mw (s, d) r)
              (fun r _ => mul_nonneg (hμ r) (TAlg.use_nonneg Mw (s, d) r)) (mem_univ 0)
    unfold roundShortfall
    refine max_le (by linarith) ?_
    show 1 - (ta.cost μ Mw (s, d) + β * 0) / (2 * μ 0) ≤ _
    rw [mul_zero, add_zero, sub_le_sub_iff_left, div_le_div_iff₀ two_pos (mul_pos two_pos hμ0)]
    calc ((charged (TAlg.run Mw (s, d))).card : ℝ) * (2 * μ 0) =
          2 * (μ 0 * ((charged (TAlg.run Mw (s, d))).card : ℝ)) := by ring
      _ ≤ 2 * ta.cost μ Mw (s, d) := by linarith
      _ = ta.cost μ Mw (s, d) * 2 := by ring
  unfold wordsMoment avg
  refine (div_le_one (Nat.cast_pos.mpr (Fintype.card_pos (α := TDraw × Fin 2)))).mpr ?_
  calc _ ≤ ∑ x : TDraw × Fin 2, (if x.2 ∈ charged (TAlg.run Mw (s, x.1)) then
          Real.exp (1 - ((charged (TAlg.run Mw (s, x.1))).card : ℝ) / 2) else 0) := by
        refine sum_le_sum fun x _ => ?_
        split_ifs with h1 h2 h3
        · rw [one_mul]
          exact Real.exp_le_exp.mpr (hsf x.1)
        · exact absurd (hmem x.1 x.2 h1) h2
        · exact (Real.exp_pos _).le
        · exact le_rfl
    _ = ∑ d : TDraw, ((charged (TAlg.run Mw (s, d))).card : ℝ) *
          Real.exp (1 - ((charged (TAlg.run Mw (s, d))).card : ℝ) / 2) := by
        rw [Fintype.sum_prod_type]
        refine sum_congr rfl fun d _ => ?_
        dsimp only
        rw [sum_ite_mem, univ_inter, sum_const, nsmul_eq_mul]
    _ ≤ ∑ _d : TDraw, (2 : ℝ) := sum_le_sum fun d _ => mul_exp_le_two _
    _ = (Fintype.card (TDraw × Fin 2) : ℝ) := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul]
        norm_num

end Pouw.SecurityProofs.DitherWitness
