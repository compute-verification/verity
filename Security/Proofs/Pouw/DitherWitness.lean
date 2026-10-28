import Proofs.Pouw.Dither
import Proofs.Pouw.ShaRealize

/-!
# `WorkSound`'s premises hold together

The headline isn't vacuous: on the witness machine, round and period below, every premise of
`Definitions.Pouw.Guarantees.Dither.WorkSound` holds at once, at `k = 2` rounds and the non-uniform prices
`μ = (2, 1, 1)`, and the conclusion's bounds say something: `ε_A = 1/7 + ln 2 / 2 < 1` and `κ_A = 1/2 < 1` for every
prover (`witness`). `witness_freed` is the headline's conclusion there, from `WorkSound`.

* `tm`, `tw`: the witness machine over three resources (`0` MAC units, `1` hash units, `2` the rest). A program is
  any function with its use and its oracle calls, each call a step that uses a hash unit (G3, `callUse = (0, 1, 0)`).
  The machine's laws (`TAlg`) charge it a MAC unit for each word cell it outputs that varies over its inputs, a hash
  unit for each root cell that varies, and a round's work (dither in, state and root out) both for each root cell
  that varies. A cell that is constant over every input is free: a program can guess a word or a root cell at no cost.
  No law reads an answer's output, so answers are free.
* `twNF`, `twRound`: the witness round, a no-fold round with two words (`n = 2`, `n' = nb = L = 1`). The noise is the
  word: word `i` is the dither's `d i`, one of 16 values, so a guessed word passes its check with probability
  `p = 1/16`. The check draws one of the two words (`s = 1`). The root is transparent: root cell `i` is word `i`
  reduced into `Fin 17` (`dec`), and a cell opens where its root cell reads back as the word (`enc`). There is no
  oracle (`Orc = Unit`), so the commitment is perfectly binding.
* The period (`twPd`): credited use `hon = (4, 4, 2)`, two honest rounds whose work uses `(2, 2, 1)` and makes two
  oracle calls and whose answer uses nothing (`honest`, timed use `hon`), `T = 8`, and one other job per resource
  worth its use at `μ`, so `μ` is the period's dual price (`twDual`). The cap on a run's calls is `Q = 8`, the
  period's hash units (`twCap`); the honest prover makes 4. A round's honest cost is
  `H = H_mm + H_sha + H_o = 4 + 2 + 1` and the public bests are the honest ones (`pub_mm = 4`, `pub_sha = 2`), so
  `C = 14`.

**The rate and free guesses.** A words program charged for `j` varying words costs at least `2 j` at `μ`, falls
short of `pub_mm = 4` by at most `4 − 2 j`, and passes the check with probability at most `(j + (2 − j) p)/2`. At rate
`η_mm` a unit of priced cost its moment is at most 1 for every `j` iff `e^{2 η_mm} (1 + p)/2 ≤ 1` (`j = 1`) and
`e^{4 η_mm} p ≤ 1` (`j = 0`), so `η_mm ≤ ln(2/(1 + p))/2`: about `0.316` at `p = 1/16`, and `ln(4/3)/2` for one-bit
words (`p = 1/2`). A commitment program charged for `j` varying root cells falls short of `pub_sha = 2` by at most
`2 − j`, with the same pass probability, so `η_sha ≤ ln(2/(1 + p))`. `WorkSound` reads them at `η_mm = η H/H_mm = 7η/4`
and `η_sha = η H/H_sha = 7η/2`, and the witness proves them at `η = 1/7` (`η_mm = 1/4`, `η_sha = 1/2`), where both
moments are at most `e^{(2 − j)/2} (32 + 240 j)/512 ≤ 1` (`exp_mul_le`). The luck term is `ln(1/δ)/(η C) = ln 2 / 2`
at `δ = 1/2`.

**What the witness doesn't show.**
* Its laws are global: a program is charged for a cell that varies anywhere on its inputs, so a cell that varies with
  the advice but not with the dither is charged too. The moment bounds use only that an uncharged cell is constant in
  the dither at fixed advice.
* No dither is degenerate (`κ_dmm = κ_dsha = 0` at every advice size): the noise is the whole word, so no advice
  predicts it (W2's degenerate set is empty here).
* `ExtractAdv` holds at `ξ = 0`, `κ_x = 0` and advice of `|Adv|` bits for every prover because the commitment is
  transparent; that is not `ShaRealize` on a random oracle.
* No program exchanges bits (`ext = 0`), so `β = K = 0`, and the verdict is the rounds' checks (`κ_S = 0`).
* The oracle is trivial (`Orc = Unit`), so the calls are counted and priced but read nothing, and `ShaRealize`'s
  `OracleCalls` and realization over a random oracle aren't exercised here. Any table `eo : Unit ≃ (X → D)` is a
  single point, so at `D = Unit` the realization's walk clause is vacuous and `WorkSoundNoFold`'s `κ`, with its
  `(q² + …)/|D|`, is at least 1; that guarantee has no witness here.
-/

namespace Pouw.SecurityProofs.DitherWitness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

section AlgModel

/-- A root cell: a word reduced into 17 values. 17 is odd, so no product with the roots' type (`289` values) has as
many values as the words' type (`2^64`), which the laws read. -/
abbrev RCell : Type := Fin 17

/-- The witness round's words, one per cell `(i, j, b)` with `j b : Fin 1`. -/
abbrev TTr : Type := Fin 2 → Fin 1 → Fin 1 → Word

/-- The witness round's roots, one cell per word. -/
abbrev TRoot : Type := Fin 2 → Fin 1 → Fin 1 → RCell

/-- The witness round's dither: one of 16 values per word. -/
abbrev TDraw : Type := Fin 2 → Fin 16

open Classical in
/-- The cells of `f`'s output that vary over its inputs. -/
noncomputable def nonconst {X α : Type} (f : X → Fin 2 → Fin 1 → Fin 1 → α) : Finset (Fin 2) :=
  univ.filter fun i => ∃ x y, f x i 0 0 ≠ f y i 0 0

open Classical in
theorem mem_nonconst {X α : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {c : Fin 2} :
    c ∈ nonconst f ↔ ∃ x y, f x c 0 0 ≠ f y c 0 0 := by
  simp [nonconst]

theorem card_le_two (s : Finset (Fin 2)) : (s.card : ℝ) ≤ 2 := by
  have h := card_le_univ s
  rw [Fintype.card_fin] at h
  exact_mod_cast h

theorem const_of_not_mem {X α : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {c : Fin 2} (hc : c ∉ nonconst f)
    (x y : X) : f x c 0 0 = f y c 0 0 := by
  by_contra h
  exact hc (mem_nonconst.2 ⟨x, y, h⟩)

theorem nonconst_mono {X Y α β : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {g : Y → Fin 2 → Fin 1 → Fin 1 → β}
    (h : ∀ c x y, f x c 0 0 ≠ f y c 0 0 → ∃ x' y', g x' c 0 0 ≠ g y' c 0 0) : nonconst f ⊆ nonconst g := by
  intro c hc
  obtain ⟨x, y, hxy⟩ := mem_nonconst.1 hc
  exact mem_nonconst.2 (h c x y hxy)

/-- **A witness program**: what it computes, its use of the three resources (nonnegative), its oracle calls (each
uses a hash unit), and the machine's three laws, each read where the program's type has the shape it is about:
* `words`: a program that outputs words uses a MAC unit for each word cell that varies over its inputs;
* `roots`: a program that outputs a root uses a hash unit for each root cell that varies;
* `work`: a round's work (oracle, state and dither in; state and root out) uses a MAC unit and a hash unit for each
  root cell that varies: its root's words are computed and hashed. -/
structure TAlg (I O : Type) where
  run : I → O
  use : I → Fin 3 → ℝ
  use_nonneg : ∀ x r, 0 ≤ use x r
  calls : I → ℝ
  calls_nonneg : ∀ x, 0 ≤ calls x
  calls_le : ∀ x, calls x ≤ use x 1
  words : ∀ (h : O = TTr) (x : I), ((nonconst fun y => cast h (run y)).card : ℝ) ≤ use x 0
  roots : ∀ (h : O = TRoot) (x : I), ((nonconst fun y => cast h (run y)).card : ℝ) ≤ use x 1
  work : ∀ (S : Type), I = (Unit × (S × TDraw)) → ∀ (h : O = (S × TRoot)) (x : I),
    ((nonconst fun y => (cast h (run y)).2).card : ℝ) ≤ use x 0 ∧
      ((nonconst fun y => (cast h (run y)).2).card : ℝ) ≤ use x 1

/-- Any function, at the use `u` plus two units of every resource (which pays for any two cells). -/
def TAlg.ofFun {I O : Type} (f : I → O) (u : I → Fin 3 → ℝ) (hu : ∀ x r, 0 ≤ u x r) : TAlg I O where
  run := f
  use x r := 2 + u x r
  use_nonneg x r := by linarith [hu x r]
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le x := by linarith [hu x 1]
  words h x := by linarith [card_le_two (nonconst fun y => cast h (f y)), hu x 0]
  roots h x := by linarith [card_le_two (nonconst fun y => cast h (f y)), hu x 1]
  work _ _ h x := ⟨by linarith [card_le_two (nonconst fun y => (cast h (f y)).2), hu x 0],
    by linarith [card_le_two (nonconst fun y => (cast h (f y)).2), hu x 1]⟩

/-- The witness machine. Its `time` is zero (the headline reads `Accel.use`, not `time`), and composing or routing
programs pays two units of every resource, the most any program's cells can cost. -/
abbrev tm : AlgModel where
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

/-- The witness accelerator: three resources at rate 1, a run's wall time its total use, nothing exchanged, and an
oracle call a step that uses one hash unit. -/
noncomputable abbrev tw : Accel where
  Mc := tm
  Res := Fin 3
  rate _ := 1
  mem := 0
  use f x r := TAlg.use f x r
  held _ _ := 0
  wall f x := ∑ r, TAlg.use f x r
  ext _ _ := 0
  calls f x := TAlg.calls f x
  callUse r := if r = 1 then 1 else 0
  P1 := True
  P2 := True

theorem tw_G1 : tw.G1 := fun _ _ => by
  show (0 : ℝ) + 0 ≤ 0
  norm_num

theorem tw_G2 : tw.G2 := by
  refine ⟨fun _ => one_pos, fun {I O} f x => ⟨le_rfl, fun r => ⟨TAlg.use_nonneg f x r, ?_⟩⟩⟩
  show TAlg.use f x r ≤ 1 * ∑ r, TAlg.use f x r
  rw [one_mul]
  exact single_le_sum (f := fun r => TAlg.use f x r) (fun r _ => TAlg.use_nonneg f x r) (mem_univ r)

theorem tw_G3 : tw.G3 := by
  intro I O f x
  refine ⟨TAlg.calls_nonneg f x, fun (r : Fin 3) => ?_⟩
  show TAlg.calls f x * (if r = 1 then 1 else 0) ≤ TAlg.use f x r
  split_ifs with h
  · rw [mul_one, h]
    exact TAlg.calls_le f x
  · rw [mul_zero]
    exact TAlg.use_nonneg f x r

theorem cost_ge (μ : Fin 3 → ℝ) (hμ : ∀ r, 0 ≤ μ r) {I O : Type} (f : TAlg I O) (x : I) (r : Fin 3) :
    μ r * f.use x r ≤ tw.cost μ f x :=
  single_le_sum (f := fun r => μ r * f.use x r) (fun r _ => mul_nonneg (hμ r) (f.use_nonneg x r)) (mem_univ r)

theorem card_ttr : Nat.card TTr = 18446744073709551616 := by
  simp [Nat.card_eq_fintype_card, ZMod.card]

theorem card_troot : Nat.card TRoot = 289 := by
  simp [Nat.card_eq_fintype_card]

theorem card_tdraw : Nat.card TDraw = 256 := by
  simp [Nat.card_eq_fintype_card]

theorem card_unit' : Nat.card Unit = 1 := by
  simp [Nat.card_eq_fintype_card]

theorem ttr_ne_troot : TTr ≠ TRoot := fun e => by
  have := congrArg Nat.card e
  rw [card_ttr, card_troot] at this
  omega

theorem ttr_ne_prod (S : Type) : TTr ≠ (S × TRoot) := fun e => by
  have := congrArg Nat.card e
  rw [Nat.card_prod, card_ttr, card_troot] at this
  omega

end AlgModel

section Round

/-- A word reduced into a root cell: its value where it is below 17, else 0. -/
def dec (w : Word) : RCell := if h : w.val < 17 then ⟨w.val, h⟩ else 0

/-- A root cell read back as a word. -/
def enc (v : RCell) : Word := (v.val : Word)

theorem enc_dec (v : Fin 16) : enc (dec ((v : ℕ) : Word)) = ((v : ℕ) : Word) := by
  have hv : ((v : ℕ) : Word).val = v := by
    rw [ZMod.val_natCast]
    exact Nat.mod_eq_of_lt (lt_trans v.isLt (by norm_num))
  have hlt : ((v : ℕ) : Word).val < 17 := by rw [hv]; exact lt_trans v.isLt (by norm_num)
  unfold dec
  split_ifs
  show ((((v : ℕ) : Word).val : ℕ) : Word) = _
  rw [hv]

/-- **The witness round's knobs**: two words (`n = 2`, `n' = nb = L = 1`), the noise the word itself (word `i` is the
dither's `d i`), a check of one of the two words (`s = 1`), no oracle, and the transparent root: each word reduced into
`Fin 17`, a cell opening where its root cell reads back as the word. -/
def twNF : NoFold where
  n := 2
  n' := 1
  nb := 1
  L := 1
  Draw := TDraw
  noise _ d := (fun i _ => ((d i : ℕ) : Word), fun _ _ => 1)
  s := 1
  Chk := Fin 2
  cells c _ := (c, 0, 0)
  Orc := Unit
  Root := TRoot
  merkle _ T i j b := dec (T i j b)
  Path := Unit
  opens _ r c v _ := enc (r c.1 c.2.1 c.2.2) = v

/-- The witness round. -/
abbrev twRound : RoundProto := noFoldRound twNF

theorem twWord (a : twRound.Inst) (d : TDraw) (i : Fin 2) (j b : Fin 1) :
    noFoldWord twNF (twNF.noise a d) i j b = ((d i : ℕ) : Word) := by
  show (∑ t : Fin 1, if t.val < (b.val + 1) * 1 then ((d i : ℕ) : Word) * 1 else 0) = _
  simp

theorem twOk_iff (a : twRound.Inst) (d : TDraw) (c : Fin 2) (T : TTr) :
    twRound.ok a d c T ↔ T c 0 0 = ((d c : ℕ) : Word) := by
  show (∀ _ : Fin 1, T c 0 0 = noFoldWord twNF (twNF.noise a d) c (0 : Fin 1) (0 : Fin 1)) ↔ _
  rw [twWord]
  exact ⟨fun h => h 0, fun h _ => h⟩

theorem twOpens_iff (o : Unit) (r : TRoot) (c : Fin 2) (T : TTr) :
    twRound.opens o r c T ↔ enc (r c 0 0) = T c 0 0 := by
  show (∀ _ : Fin 1, ∃ _ : Unit, enc (r c 0 0) = T c 0 0) ↔ _
  exact ⟨fun h => (h 0).elim fun _ h => h, fun h _ => ⟨(), h⟩⟩

theorem twAccept_iff (o : Unit) (a : twRound.Inst) (d : TDraw) (c : Fin 2) (r : TRoot)
    (y : Fin 1 → Word × Unit) :
    twRound.accept o a d c r y ↔ enc (r c 0 0) = (y 0).1 ∧ (y 0).1 = ((d c : ℕ) : Word) := by
  show (∀ q : Fin 1, enc (r c 0 0) = (y q).1 ∧
    (y q).1 = noFoldWord twNF (twNF.noise a d) c (0 : Fin 1) (0 : Fin 1)) ↔ _
  rw [twWord]
  exact ⟨fun h => h 0, fun h q => by rw [Subsingleton.elim q 0]; exact h⟩

end Round

section Sampling

theorem prob_le_zero {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p ≤ 0 := by
  classical
  unfold prob
  rw [filter_false_of_mem fun ω _ => h ω, card_empty, Nat.cast_zero, zero_div]

theorem word_inj {a b : Fin 16} (h : ((a : ℕ) : Word) = ((b : ℕ) : Word)) : a = b := by
  have := congrArg ZMod.val h
  rw [ZMod.val_natCast, ZMod.val_natCast, Nat.mod_eq_of_lt (lt_trans a.isLt (by norm_num)),
    Nat.mod_eq_of_lt (lt_trans b.isLt (by norm_num))] at this
  exact Fin.ext this

/-- A free guess: a word constant in the dither is the dither's word at a cell on at most 16 of the 256 dithers. -/
theorem card_match_le (F : TDraw → Word) (hF : ∀ d d', F d = F d') (c : Fin 2) :
    (univ.filter fun d : TDraw => F d = ((d c : ℕ) : Word)).card ≤ 16 := by
  have key : ∀ d ∈ univ.filter (fun d : TDraw => F d = ((d c : ℕ) : Word)),
      ∀ d' ∈ univ.filter (fun d : TDraw => F d = ((d c : ℕ) : Word)), d c = d' c := by
    intro d hd d' hd'
    rw [mem_filter] at hd hd'
    exact word_inj (hd.2.symm.trans ((hF d d').trans hd'.2))
  have hinj : Set.InjOn (fun d : TDraw => d (c + 1)) (univ.filter fun d : TDraw => F d = ((d c : ℕ) : Word)) := by
    intro d hd d' hd' h
    have hc := key d (mem_coe.1 hd) d' (mem_coe.1 hd')
    funext i
    fin_cases c <;> fin_cases i <;> first | exact hc | exact h
  calc _ ≤ (univ : Finset (Fin 16)).card := card_le_card_of_injOn _ (fun _ _ => by simp) hinj
    _ = 16 := by simp

theorem exp_half_lt : Real.exp (1 / 2) < 32 / 17 := by
  have h : Real.exp (1 / 2) * Real.exp (1 / 2) = Real.exp 1 := by rw [← Real.exp_add]; norm_num
  nlinarith [Real.exp_one_lt_d9, Real.exp_pos (1 / 2)]

/-- The sampling bound's arithmetic: `e^{(2 − j)/2} (32 + 240 j) ≤ 512` for `j ≤ 2`. -/
theorem exp_mul_le (j : ℕ) (hj : j ≤ 2) : Real.exp ((2 - j) / 2) * (32 + 240 * j) ≤ 512 := by
  obtain rfl | rfl | rfl : j = 0 ∨ j = 1 ∨ j = 2 := by omega
  · have := Real.exp_one_lt_d9
    rw [show ((2 : ℝ) - ((0 : ℕ) : ℝ)) / 2 = 1 by norm_num]
    push_cast
    linarith
  · rw [show ((2 : ℝ) - ((1 : ℕ) : ℝ)) / 2 = 1 / 2 by norm_num]
    push_cast
    linarith [exp_half_lt]
  · rw [show ((2 : ℝ) - ((2 : ℕ) : ℝ)) / 2 = 0 by norm_num, Real.exp_zero]
    push_cast
    norm_num

/-- **The moment bound both public-best premises reduce to on the witness round.** A test on the round's coins that
passes only where `F d c` is the dither's word at the checked cell `c`, with `F` constant in the dither outside the
cells `σ`, and weighs a pass at most `e^{(2 − |σ|)/2}`, averages at most 1. -/
theorem moment_le_one (F : TDraw → Fin 2 → Word) (σ : Finset (Fin 2))
    (hσ : ∀ c ∉ σ, ∀ d d', F d c = F d' c) (g : TDraw × Fin 2 → ℝ)
    (hg : ∀ x, g x ≤ if F x.1 x.2 = ((x.1 x.2 : ℕ) : Word) then Real.exp ((2 - σ.card) / 2) else 0) :
    avg g ≤ 1 := by
  have hE := (Real.exp_pos ((2 - σ.card) / 2)).le
  have hj : σ.card ≤ 2 := by simpa using card_le_univ σ
  have hc : ∀ c : Fin 2, ((univ.filter fun d : TDraw => F d c = ((d c : ℕ) : Word)).card : ℝ) ≤
      16 + if c ∈ σ then 240 else 0 := by
    intro c
    split_ifs with h
    · have h1 := card_le_univ (univ.filter fun d : TDraw => F d c = ((d c : ℕ) : Word))
      have h256 : Fintype.card TDraw = 256 := by simp
      rw [h256] at h1
      have : ((univ.filter fun d : TDraw => F d c = ((d c : ℕ) : Word)).card : ℝ) ≤ 256 := by exact_mod_cast h1
      linarith
    · have h1 := card_match_le (fun d => F d c) (hσ c h) c
      have : ((univ.filter fun d : TDraw => F d c = ((d c : ℕ) : Word)).card : ℝ) ≤ 16 := by exact_mod_cast h1
      linarith
  unfold avg
  rw [div_le_one (Nat.cast_pos.mpr (Fintype.card_pos (α := TDraw × Fin 2)))]
  calc ∑ x, g x ≤ ∑ x : TDraw × Fin 2,
        if F x.1 x.2 = ((x.1 x.2 : ℕ) : Word) then Real.exp ((2 - σ.card) / 2) else 0 :=
        sum_le_sum fun x _ => hg x
    _ = ∑ c : Fin 2, Real.exp ((2 - σ.card) / 2) *
          ((univ.filter fun d : TDraw => F d c = ((d c : ℕ) : Word)).card : ℝ) := by
        rw [Fintype.sum_prod_type_right]
        refine sum_congr rfl fun c _ => ?_
        dsimp only
        rw [← sum_filter, sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ c : Fin 2, Real.exp ((2 - σ.card) / 2) * (16 + if c ∈ σ then 240 else 0) :=
        sum_le_sum fun c _ => mul_le_mul_of_nonneg_left (hc c) hE
    _ = Real.exp ((2 - σ.card) / 2) * (32 + 240 * σ.card) := by
        rw [← mul_sum, sum_add_distrib, sum_const, sum_ite_mem, univ_inter, sum_const, card_univ,
          Fintype.card_fin]
        simp only [nsmul_eq_mul]
        push_cast
        ring
    _ ≤ 512 := exp_mul_le σ.card hj
    _ = Fintype.card (TDraw × Fin 2) := by simp

end Sampling

section PeriodDevices

/-- The prices: a MAC unit at 2, a hash unit and a unit of the rest at 1. -/
noncomputable def twμ (r : Fin 3) : ℝ := if r = 0 then 2 else 1

theorem twμ_nonneg (r : Fin 3) : 0 ≤ twμ r := by
  unfold twμ
  split_ifs <;> norm_num

/-- The period's credited use: two honest rounds of `(2, 2, 1)`. -/
noncomputable def twHon (r : Fin 3) : ℝ := if r = 2 then 2 else 4

/-- **The witness period**: `T = 8` seconds of the three resources at rate 1, one other job per resource worth its
use at `μ`, and the credited use `hon` worth `C = μ · hon = 14`. -/
noncomputable def twPd : PeriodDevices tw where
  T := 8
  J := Fin 3
  val := twμ
  needs j r := if j = r then 1 else 0
  hon := twHon
  rv := 14

/-- `μ` is the witness period's dual price at one credited unit: the honest strategy fills what the credited rounds
leave with the other jobs. -/
theorem twDual : twPd.DualOpt twμ 1 := by
  refine ⟨twμ_nonneg, fun (j r : Fin 3) => ?_, fun (j : Fin 3) => ?_, fun j => 8 - twHon j, ⟨fun j => ?_, fun r => ?_⟩, ?_⟩
  · show (0 : ℝ) ≤ if j = r then 1 else 0
    split_ifs <;> norm_num
  · show twμ j ≤ ∑ r : Fin 3, twμ r * if j = r then 1 else 0
    simp
  · show (0 : ℝ) ≤ 8 - twHon j
    unfold twHon
    split_ifs <;> norm_num
  · show 1 * twHon r + ∑ j : Fin 3, (8 - twHon j) * (if j = r then 1 else 0) ≤ 1 * 8
    simp
  · show ∑ j : Fin 3, twμ j * (8 - twHon j) = ∑ r : Fin 3, twμ r * (1 * 8 - 1 * twHon r)
    simp

/-- A round's honest matmul cost at `μ` (two words, a MAC unit each), its public best's too. -/
noncomputable def twHmm (_ : twRound.Inst) : ℝ := 4

/-- A round's honest hashing cost at `μ` (two root cells, a hash unit each), its public best's too. -/
noncomputable def twHsha (_ : twRound.Inst) : ℝ := 2

/-- The rest of a round's honest cost at `μ`. -/
noncomputable def twHo (_ : twRound.Inst) : ℝ := 1

/-- The rate a unit of priced cost: `η_mm = 1/4`, `η_sha = 1/2`. -/
noncomputable def twη : ℝ := 1 / 7

/-- The confidence. -/
noncomputable def twδ : ℝ := 1 / 2

/-- The cap on a run's oracle calls: the period's hash units over a call's use of them. -/
noncomputable def twQ : ℝ := 8

theorem twCap : ∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r := by
  refine ⟨1, ?_, ?_⟩
  · show (0 : ℝ) < if (1 : Fin 3) = 1 then 1 else 0
    norm_num
  · show (1 : ℝ) * 8 ≤ twQ * if (1 : Fin 3) = 1 then 1 else 0
    norm_num [twQ]

/-- The registered instances: the witness round's noise reads none, so any will do. -/
def twReg (_ : Fin 2) : twRound.Inst := (fun _ _ => 0, fun _ _ => 0)

theorem twCost :
    ∑ w : Fin 2, (twHmm (twReg w) + twHsha (twReg w) + twHo (twReg w)) = ∑ r, twμ r * twPd.hon r := by
  show ∑ _w : Fin 2, ((4 : ℝ) + 2 + 1) = ∑ r : Fin 3, twμ r * twHon r
  norm_num [Fin.sum_univ_three, twμ, twHon]

end PeriodDevices

section Premises

/-- **The witness verdict**: the rounds' checks on the transcript. The devices read nothing. -/
def twVerdict : Acceptance twRound 2 Unit := fun o x t _ =>
  ∀ w, twRound.accept o (twReg w) (x w).1 (x w).2 (t w).1 (t w).2

/-- The devices' readings: nothing. -/
def twDev : Readings tw twRound 2 Unit := fun _ _ _ => ()

theorem twSAK : SAKMatmul tw twRound twμ Set.univ twHmm
    (fun a => twη * (twHmm a + twHsha a + twHo a) / twHmm a) (fun _ => 0) := by
  intro _ _ _ _ a _ B St _ s Mw _
  refine ⟨fun _ => False, prob_le_zero fun _ h => h, ?_⟩
  unfold wordsMoment
  refine moment_le_one (fun d c => (TAlg.run Mw (s, d) : TTr) c 0 0) (nonconst (TAlg.run Mw))
    (fun c hc d d' => const_of_not_mem hc (s, d) (s, d')) _ fun x => ?_
  have hσ : ((nonconst (TAlg.run Mw)).card : ℝ) ≤ TAlg.use Mw (s, x.1) 0 := TAlg.words Mw rfl (s, x.1)
  have hcost := cost_ge twμ twμ_nonneg Mw (s, x.1) 0
  have hcard := card_le_two (nonconst (TAlg.run Mw))
  have hμ0 : twμ 0 = 2 := by simp [twμ]
  rw [hμ0] at hcost
  split_ifs with h1 h2
  · simp only [twη, twHmm, twHsha, twHo, roundShortfall]
    apply Real.exp_le_exp.2
    have key : max 0 (4 - tw.cost twμ Mw (s, x.1)) ≤ 4 - 2 * ((nonconst (TAlg.run Mw)).card : ℝ) :=
      max_le (by linarith) (by linarith)
    linarith
  · exact absurd ((twOk_iff _ _ _ _).1 h1.2) h2
  · exact (Real.exp_pos _).le
  · exact le_rfl

theorem twShaSpeed : ShaSpeed tw twRound 2 twReg twμ twHsha
    (fun a => twη * (twHmm a + twHsha a + twHo a) / twHsha a) (fun _ => 0) (fun _ => 0) := by
  intro _ _ _ _ _ A B Adv _ adv _ Mw Mh _ _ _
  refine prob_le_zero fun ox ⟨w, hw⟩ => hw ⟨fun _ => False, prob_le_zero fun _ h => h, ?_⟩
  unfold commitMoment
  refine moment_le_one
    (fun d c => enc ((TAlg.run (Mh w) (ox.1, (adv ox.1 ox.2 w, d), TAlg.run (Mw w) (adv ox.1 ox.2 w, d)) : TRoot)
      c 0 0))
    (nonconst (TAlg.run (Mh w))) (fun c hc d d' => congrArg enc (const_of_not_mem hc _ _)) _ fun x => ?_
  have hσ : ((nonconst (TAlg.run (Mh w))).card : ℝ) ≤
      TAlg.use (Mh w) (ox.1, (adv ox.1 ox.2 w, x.1), TAlg.run (Mw w) (adv ox.1 ox.2 w, x.1)) 1 :=
    TAlg.roots (Mh w) rfl _
  have hcost :=
    cost_ge twμ twμ_nonneg (Mh w) (ox.1, (adv ox.1 ox.2 w, x.1), TAlg.run (Mw w) (adv ox.1 ox.2 w, x.1)) 1
  have hcard := card_le_two (nonconst (TAlg.run (Mh w)))
  have hμ1 : twμ 1 = 1 := by simp [twμ]
  rw [hμ1, one_mul] at hcost
  split_ifs with h1 h2
  · simp only [twη, twHmm, twHsha, twHo, roundShortfall]
    apply Real.exp_le_exp.2
    have key : max 0 (2 - tw.cost twμ (Mh w) (ox.1, (adv ox.1 ox.2 w, x.1),
        TAlg.run (Mw w) (adv ox.1 ox.2 w, x.1))) ≤ 2 - ((nonconst (TAlg.run (Mh w))).card : ℝ) :=
      max_le (by linarith) (by linarith)
    linarith
  · exact absurd (((twOpens_iff _ _ _ _).1 h1.2.2).trans ((twOk_iff _ _ _ _).1 h1.2.1)) h2
  · exact (Real.exp_pos _).le
  · exact le_rfl

theorem twOutsource : OutsourceWithinEgress tw twμ 0 := by
  intro _ _ _ _ I O f
  exact ⟨f, fun i => ⟨rfl, rfl, by simp⟩⟩

theorem twService : ServiceSound tw twRound 2 twReg twVerdict twDev 0 := fun _ =>
  prob_le_zero fun _ h => h.2 h.1

theorem twImport : ImportWithinCertifier tw twRound 2 twVerdict twDev 0 := fun _ _ _ _ => by
  show ∑ _b : Fin 2, ((0 : ℝ) + 0) ≤ 0
  simp

/-- The extractor's advice at round `w`: the coins of the rounds before `w`, which is all the round's state reads
besides the oracle (`Unit` here). -/
abbrev twAdv : Type := Fin 2 → Option (TDraw × Fin 2)

/-- The advice at round `w`: the earlier rounds' coins. -/
def past (_ : Unit) (x : Fin 2 → TDraw × Fin 2) (w : Fin 2) : twAdv := fun v => if v < w then some (x v) else none

/-- The coins the advice records, the later ones filled in arbitrarily. -/
def fill (p : twAdv) : Fin 2 → TDraw × Fin 2 := fun v => (p v).getD default

/-- The advice size every prover's extraction reads. -/
def twBadv (_ : Prover tw twRound 2) : ℕ := Fintype.card twAdv

/-- The extraction's walk and log: none. -/
def twξ (_ : Prover tw twRound 2) (_ : Fin 3) : ℝ := 0

theorem adv_inj : ∃ e : twAdv → Fin (2 ^ Fintype.card twAdv), Function.Injective e :=
  ⟨fun a => Fin.castLE (Nat.lt_two_pow_self).le (Fintype.equivFin twAdv a),
    (Fin.castLE_injective _).comp (Fintype.equivFin twAdv).injective⟩

theorem fill_past (o : Unit) (x : Fin 2 → TDraw × Fin 2) (w v : Fin 2) (hv : (v : ℕ) < w) :
    fill (past o x w) v = x v := by
  simp [fill, past, show v < w from hv]

/-- Round `w`'s root, recomputed from the advice and the dither: the prover's own work. -/
noncomputable def rootOf (A : Prover tw twRound 2) (w : Fin 2) (p : twAdv × TDraw) : TRoot :=
  (TAlg.run (A.work w) ((), A.state () (fill p.1) w, p.2)).2

theorem rootOf_past (A : Prover tw twRound 2) (o : Unit) (x : Fin 2 → TDraw × Fin 2) (w : Fin 2) :
    rootOf A w (past o x w, (x w).1) = A.root o x w := by
  cases o
  show (TAlg.run (A.work w) ((), A.state () (fill (past () x w)) w, (x w).1)).2 =
    (TAlg.run (A.work w) ((), A.state () x w, (x w).1)).2
  rw [Pouw.SecurityProofs.ShaRealize.state_congr A () w fun v hv => fill_past () x w v hv]

/-- The root cells round `w`'s work varies: the MAC and hash units the machine charges it. -/
noncomputable def cw (A : Prover tw twRound 2) (w : Fin 2) : ℕ :=
  (nonconst fun y => (TAlg.run (A.work w) y).2).card

theorem mw_card_le (A : Prover tw twRound 2) (w : Fin 2) :
    (nonconst fun (p : twAdv × TDraw) (i : Fin 2) (_ _ : Fin 1) => enc (rootOf A w p i 0 0)).card ≤ cw A w :=
  card_le_card (nonconst_mono fun _ x y hxy =>
    ⟨((), A.state () (fill x.1) w, x.2), ((), A.state () (fill y.1) w, y.2), fun he => hxy (congrArg enc he)⟩)

theorem mh_card_le (A : Prover tw twRound 2) (w : Fin 2) :
    (nonconst fun p : Unit × (twAdv × TDraw) × TTr => rootOf A w p.2.1).card ≤ cw A w :=
  card_le_card (nonconst_mono fun _ x y hxy =>
    ⟨((), A.state () (fill x.2.1.1) w, x.2.1.2), ((), A.state () (fill y.2.1.1) w, y.2.1.2), hxy⟩)

theorem mh_shape (S : Type) (hI : (Unit × (twAdv × TDraw) × TTr) = (Unit × (S × TDraw)))
    (hO : TRoot = (S × TRoot)) : False := by
  have h1 := congrArg Nat.card hI
  have h2 := congrArg Nat.card hO
  rw [Nat.card_prod, Nat.card_prod, Nat.card_prod, Nat.card_prod, Nat.card_prod, card_unit', card_tdraw,
    card_ttr] at h1
  rw [Nat.card_prod, card_troot] at h2
  omega

/-- **The extraction's words program** at round `w`: the words read off the prover's own root, which its work
computes from the advice and the dither. It is charged a MAC unit for each root cell the work varies. -/
noncomputable def mwX (A : Prover tw twRound 2) (w : Fin 2) : TAlg (twAdv × TDraw) TTr where
  run p i _ _ := enc (rootOf A w p i 0 0)
  use _ r := if r = 0 then (cw A w : ℝ) else 0
  use_nonneg _ r := by split_ifs <;> simp
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := by simp
  words h _ := by
    have := mw_card_le A w
    simp only [cast_eq]
    simpa using this
  roots h _ := absurd h ttr_ne_troot
  work S _ h _ := absurd h (ttr_ne_prod S)

/-- **The extraction's commitment program** at round `w`: the prover's own root, from the advice and the dither. It
is charged a hash unit for each root cell the work varies. -/
noncomputable def mhX (A : Prover tw twRound 2) (w : Fin 2) : TAlg (Unit × (twAdv × TDraw) × TTr) TRoot where
  run p := rootOf A w p.2.1
  use _ r := if r = 1 then (cw A w : ℝ) else 0
  use_nonneg _ r := by split_ifs <;> simp
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := by simp
  words h _ := absurd h.symm ttr_ne_troot
  roots h _ := by
    have := mh_card_le A w
    simp only [cast_eq]
    simpa using this
  work S hI h _ := (mh_shape S hI h).elim

theorem twExtract : ExtractAdv tw twRound 2 twReg twQ twBadv twξ (fun _ => 0) := by
  intro _ _ _ _ _ A
  refine ⟨twAdv, past, mwX A, mhX A, adv_inj, fun o x x' w hx => ?_, fun o x w => rootOf_past A o x w,
    fun o x _ r => ?_, fun o x w => ?_, ?_⟩
  · funext v
    simp only [past]
    split_ifs with h
    · rw [hx v h]
    · rfl
  · show ∑ w, (TAlg.use (mwX A w) (past o x w, (x w).1) r +
        TAlg.use (mhX A w) (o, (past o x w, (x w).1), TAlg.run (mwX A w) (past o x w, (x w).1)) r) ≤
      ∑ w, TAlg.use (A.work w) (A.workIn o x w) r + 0
    rw [add_zero]
    refine sum_le_sum fun w _ => ?_
    have hw : (cw A w : ℝ) ≤ TAlg.use (A.work w) (A.workIn o x w) 0 ∧
        (cw A w : ℝ) ≤ TAlg.use (A.work w) (A.workIn o x w) 1 :=
      TAlg.work (A.work w) A.St rfl rfl (A.workIn o x w)
    have h2 := TAlg.use_nonneg (A.work w) (A.workIn o x w) r
    show (if r = 0 then (cw A w : ℝ) else 0) + (if r = 1 then (cw A w : ℝ) else 0) ≤
      TAlg.use (A.work w) (A.workIn o x w) r
    fin_cases r
    · simpa using hw.1
    · simpa using hw.2
    · simpa using h2
  · show (0 : ℝ) + 0 ≤ 0
    norm_num
  · refine prob_le_zero fun ox ⟨hacc, _, w, hw⟩ => hw ⟨?_, ?_⟩
    · have ha := (twAccept_iff _ _ _ _ _ _).1 (hacc w)
      refine (twOk_iff _ _ _ _).2 ?_
      exact (congrArg (fun r : TRoot => enc (r (ox.2 w).2 0 0)) (rootOf_past A ox.1 ox.2 w)).trans
        (ha.1.trans ha.2)
    · refine (twOpens_iff _ _ _ _).2 ?_
      exact (congrArg (fun r : TRoot => enc (r (ox.2 w).2 0 0)) (rootOf_past A ox.1 ox.2 w)).symm

end Premises

section Honest

/-- The honest words at the dither `d`: word `i` is `d i`. -/
def hw (d : TDraw) : TTr := fun i _ _ => ((d i : ℕ) : Word)

/-- The honest round's work: its two words (a MAC unit each), their root (a hash unit each) and a unit of the rest.
Its state is the words. -/
noncomputable def hWork : TAlg (Unit × TTr × TDraw) (TTr × TRoot) where
  run p := (hw p.2.2, fun i j b => dec (hw p.2.2 i j b))
  use _ r := if r = 2 then 1 else 2
  use_nonneg _ r := by split_ifs <;> norm_num
  calls _ := 2
  calls_nonneg _ := by norm_num
  calls_le _ := by norm_num
  words _ _ := (card_le_two _).trans (by simp)
  roots _ _ := (card_le_two _).trans (by simp)
  work _ _ _ _ := ⟨(card_le_two _).trans (by simp), (card_le_two _).trans (by simp)⟩

theorem card_ans : Nat.card (Fin 1 → Word × Unit) = 4294967296 := by
  simp [Nat.card_eq_fintype_card, ZMod.card]

/-- Sampling block `b`'s honest answer: the checked word, from the state after the block's work, at the block's own
check coin among the period's. It outputs neither words nor a root, so no law charges it. -/
noncomputable def hAnswer (b : Fin 2) : TAlg (Unit × TTr × (Fin 2 → TDraw × Fin 2)) (Fin 1 → Word × Unit) where
  run p := fun _ => (p.2.1 (p.2.2 b).2 0 0, ())
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_ttr]; omega)
  roots h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_troot]; omega)
  work S _ h _ := absurd (congrArg Nat.card h) (by rw [card_ans, Nat.card_prod, card_troot]; omega)

/-- **The honest prover**: every sampling block computes the words and commits them, keeps the words as its state,
and opens the checked one. -/
noncomputable def honest : Prover tw twRound 2 where
  St := TTr
  start := hw fun _ => 0
  work _ := hWork
  absorb _ s _ := s
  answer b := hAnswer b

theorem honest_accepted (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.Accepted twReg o x := fun w =>
  (twAccept_iff _ _ _ _ _ _).2 ⟨enc_dec ((x w).1 (x w).2), rfl⟩

theorem honest_timedUse (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.timedUse o x = twPd.hon := by
  funext r
  show ∑ _b : Fin 2, (if r = 2 then (1 : ℝ) else 2) = twHon r
  unfold twHon
  split_ifs <;> norm_num

theorem honest_calls (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.calls o x ≤ twQ := by
  show ∑ _b : Fin 2, (2 : ℝ) ≤ twQ
  norm_num [twQ]

end Honest

section Witness

/-- The headline's error at the witness, prover by prover (`WorkSound`'s `ε_A`): `1/7 + ln 2 / 2`. -/
noncomputable def twε (A : Prover tw twRound 2) : ℝ :=
  epsWorkSound (fun w => twHmm (twReg w) + twHsha (twReg w) + twHo (twReg w)) (fun w => twHmm (twReg w))
      (fun w => twHsha (twReg w)) twη twδ 0 0 +
    (∑ r, twμ r * twξ A r) / ∑ w, (twHmm (twReg w) + twHsha (twReg w) + twHo (twReg w))

/-- The headline's probability at the witness, prover by prover (`WorkSound`'s `κ_A`): `δ = 1/2`. -/
noncomputable def twκ (_ : Prover tw twRound 2) : ℝ := twδ + 0 + 0 + 0 + ((2 : ℕ) : ℝ) * (0 + 0)

theorem twε_lt (A : Prover tw twRound 2) : twε A < 1 := by
  have h2 := Real.log_two_lt_d9
  simp only [twε, epsWorkSound, twHmm, twHsha, twHo, twη, twδ, twξ, mul_zero, sum_const_zero, zero_div, add_zero,
    Fin.sum_univ_two, one_div_one_div]
  rw [div_lt_one (by norm_num)]
  linarith

theorem twκ_lt (A : Prover tw twRound 2) : twκ A < 1 := by
  norm_num [twκ, twδ]

/-- **The witness statement**: every premise of `WorkSound`, in its order, at the witness
machine `tw`, round `twRound` and period `twPd`, with `k = 2`, `Reg` every instance, `μ = (2, 1, 1)`,
`H_mm = pub_mm = 4`, `H_sha = pub_sha = 2`, `H_o = 1`, `η = 1/7`, `δ = 1/2`, `β = K = κ_S = 0`, the cap `Q = 8`,
`κ_dmm = κ_dsha = 0`, and for every prover `B_adv = |Adv|`, `ξ = 0`, `κ_x = 0` and `κ_sha = 0`; then the
conclusion's bounds are nontrivial for every prover, `ε_A < 1` and `κ_A < 1`; and the honest prover passes on every
oracle and coins, its rounds' timed use is the period's whole honest use, and its oracle calls are within the cap. -/
def WitnessHolds : Prop :=
  (∀ w, twReg w ∈ (Set.univ : Set twRound.Inst)) ∧ twPd.DualOpt twμ 1 ∧
    (∀ a ∈ (Set.univ : Set twRound.Inst),
      0 < twHmm a ∧ 0 < twHsha a ∧ 0 ≤ twHo a ∧ twHmm a ≤ twHmm a ∧ twHsha a ≤ twHsha a) ∧
    ∑ w, (twHmm (twReg w) + twHsha (twReg w) + twHo (twReg w)) = ∑ r, twμ r * twPd.hon r ∧
    0 < twη ∧ 0 < twδ ∧ twδ ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (∀ A r, 0 ≤ twξ A r) ∧
    (∀ _B : ℕ, (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0) ∧ (∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r) ∧
    tw.G1 ∧ tw.G2 ∧ tw.G3 ∧ tw.P1 ∧ tw.P2 ∧
    SAKMatmul tw twRound twμ Set.univ twHmm (fun a => twη * (twHmm a + twHsha a + twHo a) / twHmm a)
      (fun _ => 0) ∧
    ShaSpeed tw twRound 2 twReg twμ twHsha (fun a => twη * (twHmm a + twHsha a + twHo a) / twHsha a)
      (fun _ => 0) (fun _ => 0) ∧
    OutsourceWithinEgress tw twμ 0 ∧ ServiceSound tw twRound 2 twReg twVerdict twDev 0 ∧
    ImportWithinCertifier tw twRound 2 twVerdict twDev 0 ∧
    ExtractAdv tw twRound 2 twReg twQ twBadv twξ (fun _ => 0) ∧
    (∀ A, twε A < 1 ∧ twκ A < 1) ∧
    ∀ o x, honest.Passes twVerdict twDev o x ∧ honest.timedUse o x = twPd.hon ∧ honest.calls o x ≤ twQ

theorem witness : WitnessHolds :=
  ⟨fun _ => Set.mem_univ _, twDual,
    fun _ _ => ⟨by norm_num [twHmm], by norm_num [twHsha], by norm_num [twHo], le_rfl, le_rfl⟩, twCost,
    by norm_num [twη], by norm_num [twδ], by norm_num [twδ], le_rfl, le_rfl, fun _ _ => le_rfl,
    fun _ => ⟨le_rfl, le_rfl⟩, twCap, tw_G1, tw_G2, tw_G3, trivial, trivial, twSAK, twShaSpeed, twOutsource,
    twService, twImport, twExtract, fun A => ⟨twε_lt A, twκ_lt A⟩,
    fun o x => ⟨honest_accepted o x, honest_timedUse o x, honest_calls o x⟩⟩

/-- **The headline at the witness**: `WorkSound`'s conclusion from `witness`. -/
theorem witness_freed :
    FreedSound tw twRound 2 twVerdict twDev twPd 1 twε (∑ r, twμ r * (1 * twPd.hon r)) twκ := by
  obtain ⟨hreg, hdual, hinst, hC, hη, hδ, hδ1, hβ, hK, hξ, hdeg, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout,
    hsvc, himp, hext, -, -⟩ := witness
  exact Pouw.SecurityProofs.Dither.WorkSound tw twRound 2 Set.univ twReg Unit twVerdict twDev twPd twμ twHmm twHsha
    twHo twHmm twHsha twη twδ 0 0 0 twQ (fun _ => 0) (fun _ => 0) twBadv twξ (fun _ => 0) (fun _ => 0) hreg hdual
    hinst hC hη hδ hδ1 hβ hK hξ hdeg hcap hG1 hG2 hG3 hP1 hP2 hsak hsha hout hsvc himp hext

end Witness

end Pouw.SecurityProofs.DitherWitness
