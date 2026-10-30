import Definitions.Pouw.Witness
import Proofs.Pouw.Dither
import Proofs.Pouw.ShaRealize

/-!
# `WorkSound`'s premises hold together

The headline isn't vacuous: on `Definitions.Pouw.Witness`'s machine `tw`, protocol `twRound` and period `twPd`, every
premise of `Definitions.Pouw.Guarantees.Dither.WorkSound` holds at once, at `k = 2` sampling blocks and the non-uniform
prices `μ = (2, 1, 1)`, and the conclusion's bounds say something: `ε_A = 1/7 + ln 2 / 2 < 1` and `κ_A = 1/2 < 1` for
every prover (`witness`). `witness_freed` is the headline's conclusion there, from `WorkSound`.

* The machine's premises: G1–G3 (`tw_G1`, `tw_G2`, `tw_G3`), and a program's cost at `μ` at least each resource's
  part (`cost_ge`).
* The period: `μ` is its dual price (`twDual`), the cap `Q = 8` is its capacity of the hash units, which each call
  uses (`twCap`), and the blocks' honest costs sum to `C = 14` (`twCost`).
* The verdict reads every block's answer within the period, so `AnswersFit` holds with every check coin selected
  (`twAnswersFit`), and the answers' calls are under the cap.

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
  predicts it (the degenerate set is empty here).
* `ExtractAdv` holds at `ξ = 0`, `κ_x = 0` and advice of `|Adv|` bits for every prover because the commitment is
  transparent; that is not `ShaRealize` on a random oracle.
* No program exchanges bits (`ext = 0`), so `β = K = 0`, and the verdict implies the checks (`κ_S = 0`).
* The commit phase is free: no law reads an instance, so a commit phase that adapts its instance to the earlier
  blocks' coins is charged nothing, and the instance changes no check (the noise reads none).
* The oracle is trivial (`Orc = Unit`), so the calls are counted and priced but read nothing, and `ShaRealize`'s
  `OracleCalls` and realization over a random oracle aren't exercised here. Any table `eo : Unit ≃ (X → D)` is a
  single point, so at `D = Unit` the realization's walk clause is vacuous and `WorkSoundNoFold`'s `κ`, with its
  `(q² + …)/|D|`, is at least 1; that guarantee has no witness here.
-/

namespace Pouw.SecurityProofs.DitherWitness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Witness Finset

section Machine

theorem const_of_not_mem {X α : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {c : Fin 2} (hc : c ∉ nonconst f)
    (x y : X) : f x c 0 0 = f y c 0 0 := by
  by_contra h
  exact hc (mem_nonconst.2 ⟨x, y, h⟩)

theorem nonconst_mono {X Y α β : Type} {f : X → Fin 2 → Fin 1 → Fin 1 → α} {g : Y → Fin 2 → Fin 1 → Fin 1 → β}
    (h : ∀ c x y, f x c 0 0 ≠ f y c 0 0 → ∃ x' y', g x' c 0 0 ≠ g y' c 0 0) : nonconst f ⊆ nonconst g := by
  intro c hc
  obtain ⟨x, y, hxy⟩ := mem_nonconst.1 hc
  exact mem_nonconst.2 (h c x y hxy)

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

end Machine

section Round

theorem enc_dec (v : Fin 16) : enc (dec ((v : ℕ) : Word)) = ((v : ℕ) : Word) := by
  have hv : ((v : ℕ) : Word).val = v := by
    rw [ZMod.val_natCast]
    exact Nat.mod_eq_of_lt (lt_trans v.isLt (by norm_num))
  have hlt : ((v : ℕ) : Word).val < 17 := by rw [hv]; exact lt_trans v.isLt (by norm_num)
  unfold dec
  split_ifs
  show ((((v : ℕ) : Word).val : ℕ) : Word) = _
  rw [hv]

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

section Jobs

theorem twμ_nonneg (r : Fin 3) : 0 ≤ twμ r := by
  unfold twμ
  split_ifs <;> norm_num

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

theorem twCap : ∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r := by
  refine ⟨1, ?_, ?_⟩
  · show (0 : ℝ) < if (1 : Fin 3) = 1 then 1 else 0
    norm_num
  · show (1 : ℝ) * 8 ≤ twQ * if (1 : Fin 3) = 1 then 1 else 0
    norm_num [twQ]

theorem twCost :
    ∑ w : Fin 2, (twHmm w + twHsha w + twHo w) = ∑ r, twμ r * twPd.hon r := by
  show ∑ _w : Fin 2, ((4 : ℝ) + 2 + 1) = ∑ r : Fin 3, twμ r * twHon r
  norm_num [Fin.sum_univ_three, twμ, twHon]

end Jobs

section Premises

theorem twSAK (w : Fin 2) : SAKMatmul tw twRound twμ (twReg w) (fun _ => twHmm w)
    (fun _ => twη * (twHmm w + twHsha w + twHo w) / twHmm w) (fun _ => 0) := by
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

theorem twSAKSha : SAKSha tw twRound 2 twReg twμ twHsha
    (fun w => twη * (twHmm w + twHsha w + twHo w) / twHsha w) (fun _ => 0) (fun _ => 0) := by
  intro _ _ _ _ _ A B Adv _ adv _ Mw Mh _ _ _
  refine prob_le_zero fun ox ⟨w, _, hw⟩ => hw ⟨fun _ => False, prob_le_zero fun _ h => h, ?_⟩
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

theorem twService : ProofSystemSound tw twRound 2 twReg twVerdict twDev 0 := fun _ =>
  prob_le_zero fun _ h => h.2 fun w => ⟨Set.mem_univ _, h.1.1 w⟩

theorem twImport : ImportWithinCertifier tw twRound 2 twVerdict twDev 0 := fun _ _ _ _ => by
  show ∑ _b : Fin 2, ((0 : ℝ) + 0 + 0) ≤ 0
  simp

theorem twAnswersFit : AnswersFit tw twRound 2 twVerdict twDev twSel twPd := fun _ _ _ hv b _ r => hv.2 b r

/-- The advice at round `w`: the earlier rounds' coins. -/
def past (_ : Unit) (x : Fin 2 → TDraw × Fin 2) (w : Fin 2) : twAdv := fun v => if v < w then some (x v) else none

/-- The coins the advice records, the later ones filled in arbitrarily. -/
def fill (p : twAdv) : Fin 2 → TDraw × Fin 2 := fun v => (p v).getD default

theorem adv_inj : ∃ e : twAdv → Fin (2 ^ Fintype.card twAdv), Function.Injective e :=
  ⟨fun a => Fin.castLE (Nat.lt_two_pow_self).le (Fintype.equivFin twAdv a),
    (Fin.castLE_injective _).comp (Fintype.equivFin twAdv).injective⟩

theorem fill_past (o : Unit) (x : Fin 2 → TDraw × Fin 2) (w v : Fin 2) (hv : (v : ℕ) < w) :
    fill (past o x w) v = x v := by
  simp [fill, past, show v < w from hv]

/-- The state sampling block `w`'s work starts from, recomputed from the advice: the prover's own commit phase. -/
noncomputable def stOf (A : Prover tw twRound 2) (w : Fin 2) (p : twAdv) : A.St :=
  (TAlg.run (A.commit w) ((), A.state () (fill p) w)).1

/-- Sampling block `w`'s root, recomputed from the advice and the dither: the prover's own commit phase and work. -/
noncomputable def rootOf (A : Prover tw twRound 2) (w : Fin 2) (p : twAdv × TDraw) : TRoot :=
  (TAlg.run (A.work w) ((), stOf A w p.1, p.2)).2

theorem rootOf_past (A : Prover tw twRound 2) (o : Unit) (x : Fin 2 → TDraw × Fin 2) (w : Fin 2) :
    rootOf A w (past o x w, (x w).1) = A.root o x w := by
  cases o
  show (TAlg.run (A.work w) ((), (TAlg.run (A.commit w) ((), A.state () (fill (past () x w)) w)).1, (x w).1)).2 =
    (TAlg.run (A.work w) ((), (TAlg.run (A.commit w) ((), A.state () x w)).1, (x w).1)).2
  rw [Pouw.SecurityProofs.ShaRealize.state_congr A () w fun v hv => fill_past () x w v hv]

/-- The root cells round `w`'s work varies: the MAC and hash units the machine charges it. -/
noncomputable def cw (A : Prover tw twRound 2) (w : Fin 2) : ℕ :=
  (nonconst fun y => (TAlg.run (A.work w) y).2).card

theorem mw_card_le (A : Prover tw twRound 2) (w : Fin 2) :
    (nonconst fun (p : twAdv × TDraw) (i : Fin 2) (_ _ : Fin 1) => enc (rootOf A w p i 0 0)).card ≤ cw A w :=
  card_le_card (nonconst_mono fun _ x y hxy =>
    ⟨((), stOf A w x.1, x.2), ((), stOf A w y.1, y.2), fun he => hxy (congrArg enc he)⟩)

theorem mh_card_le (A : Prover tw twRound 2) (w : Fin 2) :
    (nonconst fun p : Unit × (twAdv × TDraw) × TTr => rootOf A w p.2.1).card ≤ cw A w :=
  card_le_card (nonconst_mono fun _ x y hxy =>
    ⟨((), stOf A w x.2.1.1, x.2.1.2), ((), stOf A w y.2.1.1, y.2.1.2), hxy⟩)

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

theorem twExtract : ExtractAdv tw twRound 2 twReg twSel twQ twBadv twξ (fun _ => 0) := by
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
      ∑ w, (TAlg.use (A.commit w) (A.commitIn o x w) r + TAlg.use (A.work w) (A.workIn o x w) r) + 0
    rw [add_zero]
    refine sum_le_sum fun w _ => ?_
    have hw : (cw A w : ℝ) ≤ TAlg.use (A.work w) (A.workIn o x w) 0 ∧
        (cw A w : ℝ) ≤ TAlg.use (A.work w) (A.workIn o x w) 1 :=
      TAlg.work (A.work w) A.St rfl rfl (A.workIn o x w)
    have h2 := TAlg.use_nonneg (A.work w) (A.workIn o x w) r
    have h3 := TAlg.use_nonneg (A.commit w) (A.commitIn o x w) r
    show (if r = 0 then (cw A w : ℝ) else 0) + (if r = 1 then (cw A w : ℝ) else 0) ≤
      TAlg.use (A.commit w) (A.commitIn o x w) r + TAlg.use (A.work w) (A.workIn o x w) r
    split_ifs with h0 h1 h1
    · exact absurd (h0.symm.trans h1) (by decide)
    · subst h0
      linarith [hw.1]
    · subst h1
      linarith [hw.2]
    · linarith
  · show (0 : ℝ) + 0 ≤ 0
    norm_num
  · refine prob_le_zero fun ox ⟨hacc, _, w, hw⟩ => hw ⟨?_, ?_⟩
    · have ha := (twAccept_iff _ _ _ _ _ _).1 (hacc w).2
      refine (twOk_iff _ _ _ _).2 ?_
      exact (congrArg (fun r : TRoot => enc (r (ox.2 w).2 0 0)) (rootOf_past A ox.1 ox.2 w)).trans
        (ha.1.trans ha.2)
    · refine (twOpens_iff _ _ _ _).2 ?_
      exact (congrArg (fun r : TRoot => enc (r (ox.2 w).2 0 0)) (rootOf_past A ox.1 ox.2 w)).symm

end Premises

section Honest

theorem honest_accepted (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.Accepted twReg o x := fun w =>
  ⟨Set.mem_univ _, (twAccept_iff _ _ _ _ _ _).2 ⟨enc_dec ((x w).1 (x w).2), rfl⟩⟩

theorem honest_passes (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.Passes twVerdict twDev o x :=
  ⟨fun w => (honest_accepted o x w).2, fun _ _ => show (0 : ℝ) ≤ 1 * 8 by norm_num⟩

theorem honest_timedUse (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.timedUse o x = twPd.hon := by
  funext r
  show ∑ _b : Fin 2, ((0 : ℝ) + if r = 2 then (1 : ℝ) else 2) = twHon r
  unfold twHon
  split_ifs <;> norm_num

theorem honest_calls (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.calls o x ≤ twQ := by
  show ∑ _b : Fin 2, ((0 : ℝ) + 2) ≤ twQ
  norm_num [twQ]

theorem honest_withinCap (o : Unit) (x : Fin 2 → TDraw × Fin 2) : honest.WithinCap twSel twQ o x :=
  ⟨honest_calls o x, fun _ _ => show (0 : ℝ) ≤ twQ by norm_num [twQ]⟩

end Honest

section Witness

theorem twε_lt (A : Prover tw twRound 2) : twε A < 1 := by
  have h2 := Real.log_two_lt_d9
  simp only [twε, epsWorkSound, twHmm, twHsha, twHo, twη, twδ, twξ, mul_zero, sum_const_zero, zero_div, add_zero,
    Fin.sum_univ_two, one_div_one_div]
  rw [div_lt_one (by norm_num)]
  linarith

theorem twκ_lt (A : Prover tw twRound 2) : twκ A < 1 := by
  norm_num [twκ, twδ]

/-- **The witness statement**: every premise of `WorkSound`, in its order, at the witness
machine `tw`, protocol `twRound` and period `twPd`, with `k = 2` sampling blocks, every instance registrable for each
(`twReg`), `μ = (2, 1, 1)`, `H_mm = pub_mm = 4`, `H_sha = pub_sha = 2`, `H_o = 1` a block, `η = 1/7`, `δ = 1/2`,
`β = K = κ_S = 0`, the cap `Q = 8`, every check coin selected (`twSel`), `κ_dmm = κ_dsha = 0`, and for every prover
`B_adv = |Adv|`, `ξ = 0`, `κ_x = 0` and `κ_sha = 0`; then the conclusion's bounds are nontrivial for every prover,
`ε_A < 1` and `κ_A < 1`; and the honest prover passes on every oracle and coins, its sampling blocks' timed use is the
period's whole honest use, and its run is within the cap, its answers' calls included. -/
def WitnessHolds : Prop :=
  twPd.DualOpt twμ 1 ∧
    (∀ w, 0 < twHmm w ∧ 0 < twHsha w ∧ 0 ≤ twHo w ∧ twHmm w ≤ twHmm w ∧ twHsha w ≤ twHsha w) ∧
    ∑ w, (twHmm w + twHsha w + twHo w) = ∑ r, twμ r * twPd.hon r ∧
    0 < twη ∧ 0 < twδ ∧ twδ ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (∀ A r, 0 ≤ twξ A r) ∧
    (∀ _B : ℕ, (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0) ∧ (∃ r, 0 < tw.callUse r ∧ twPd.cap r ≤ twQ * tw.callUse r) ∧
    tw.G1 ∧ tw.G2 ∧ tw.G3 ∧ tw.P1 ∧ tw.P2 ∧
    (∀ w, SAKMatmul tw twRound twμ (twReg w) (fun _ => twHmm w)
      (fun _ => twη * (twHmm w + twHsha w + twHo w) / twHmm w) (fun _ => 0)) ∧
    SAKSha tw twRound 2 twReg twμ twHsha (fun w => twη * (twHmm w + twHsha w + twHo w) / twHsha w)
      (fun _ => 0) (fun _ => 0) ∧
    OutsourceWithinEgress tw twμ 0 ∧ ProofSystemSound tw twRound 2 twReg twVerdict twDev 0 ∧
    ImportWithinCertifier tw twRound 2 twVerdict twDev 0 ∧ AnswersFit tw twRound 2 twVerdict twDev twSel twPd ∧
    ExtractAdv tw twRound 2 twReg twSel twQ twBadv twξ (fun _ => 0) ∧
    (∀ A, twε A < 1 ∧ twκ A < 1) ∧
    ∀ o x, honest.Passes twVerdict twDev o x ∧ honest.timedUse o x = twPd.hon ∧ honest.WithinCap twSel twQ o x

theorem witness : WitnessHolds :=
  ⟨twDual, fun _ => ⟨by norm_num [twHmm], by norm_num [twHsha], by norm_num [twHo], le_rfl, le_rfl⟩, twCost,
    by norm_num [twη], by norm_num [twδ], by norm_num [twδ], le_rfl, le_rfl, fun _ _ => le_rfl,
    fun _ => ⟨le_rfl, le_rfl⟩, twCap, tw_G1, tw_G2, tw_G3, trivial, trivial, twSAK, twSAKSha, twOutsource,
    twService, twImport, twAnswersFit, twExtract, fun A => ⟨twε_lt A, twκ_lt A⟩,
    fun o x => ⟨honest_passes o x, honest_timedUse o x, honest_withinCap o x⟩⟩

/-- **The headline at the witness**: `WorkSound`'s conclusion from `witness`. -/
theorem witness_freed :
    FreedSound tw twRound 2 twVerdict twDev twPd 1 twε (∑ r, twμ r * (1 * twPd.hon r)) twκ := by
  obtain ⟨hdual, hinst, hC, hη, hδ, hδ1, hβ, hK, hξ, hdeg, hcap, hG1, hG2, hG3, hP1, hP2, hsak, hsha, hout,
    hsvc, himp, hans, hext, -, -⟩ := witness
  exact Pouw.SecurityProofs.Dither.WorkSound tw twRound 2 twReg Prop twVerdict twDev twPd twμ twHmm twHsha
    twHo twHmm twHsha twη twδ 0 0 0 twQ (fun _ => 0) (fun _ => 0) twSel twBadv twξ (fun _ => 0) (fun _ => 0) hdual
    hinst hC hη hδ hδ1 hβ hK hξ hdeg hcap hG1 hG2 hG3 hP1 hP2 hsak hsha hout hsvc himp hans hext

end Witness

end Pouw.SecurityProofs.DitherWitness
