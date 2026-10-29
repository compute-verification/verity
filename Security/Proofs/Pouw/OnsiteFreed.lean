import Definitions.Pouw.Guarantees.OnsiteFreed
import Proofs.Pouw.Dither
import Proofs.Pouw.DitherWitness

/-!
# The onsite PoUW protocol's freed compute: the proof

* `OnsiteFreed`: the guarantee `Definitions.Pouw.Guarantees.OnsiteFreed`. The outsourcing premise turns the prover
  into an exchange-free one with the same transcript, which the work check accepts wherever it accepts the prover. Off
  the remote floor's bad event that one spent at least `F`, so the prover spent at least `F − β · ext`, and off the
  certifier's bad event `ext ≤ K`. `FreedOfFloor` turns the floor `F − β K` into freed compute, and the union bound
  adds the two events' `κ + δ`.
* `certifiedImport_of_within`: `ImportWithinCertifier` is `CertifiedImport` at `δ = 0`.
* `outsourcePeriod_of_perProgram`: the per-program `OutsourceWithinEgress` implies its period form, so
  `OutsourceWithinEgressPeriod` asks no more of the model. Each program is replaced by its exchange-free one; the state
  path, and so the transcript, is unchanged, and the answers' exchanged bits are nonnegative (G2).
* `OnsiteFreed_honest`, `OnsiteFreed_nontrivial`: at `Definitions.Pouw.Witness`'s machine `tw`, round `twRound`
  and period `twPd` (`T = 8`, `μ = (2, 1, 1)`, `C = 14`), with the work check the witness's sampling blocks'
  checks (`twWork`). Every premise holds there: the remote floor at `F = (1 − ε_W) C`, `ε_W = 1/7 + ln 2 / 2`
  `WorkSound`'s error at the witness, from `WorkSound`'s conclusion there (`witness_freed`) at the prover with every
  answer at no use (`fitAnswers`, which passes `WorkSound`'s devices' reading) by filling each resource the prover
  leaves with its other job; the outsourcing premise at `β = 0` from the per-program one; the certifier's count
  at `K = 0` and `δ = 0`, since the witness machine exchanges nothing. The onsite verdict accepts the honest prover,
  and the bound is `ε = ε_W < 1` at `κ + δ = 1/2`. At verdicts that accept everything the conclusion is false: a
  prover that does nothing leaves the period to other work worth 46 against at most `32 + ε C < 46`.

**What the witness doesn't show.** The witness machine exchanges no bits, so `β = K = 0` and the certifier's check
refuses nothing (`twNet`); the onsite statement's `β K` term and the certifier's refusals aren't exercised.
-/

namespace Pouw.SecurityProofs

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

section Guarantee

theorem floor_of_max {C X F e : ℝ} (hC : 0 < C) (h : F - e ≤ X) : (1 - max 0 (C - F + e) / C) * C ≤ X := by
  rw [sub_mul, div_mul_cancel₀ _ hC.ne', one_mul]
  linarith [le_max_right 0 (C - F + e)]

/-- **An accepted onsite period frees at most `ε C`** (`Definitions.Pouw.Guarantees.OnsiteFreed`). -/
theorem OnsiteFreed : Pouw.Guarantees.OnsiteFreed := by
  intro G P k Obs vw vn dev Pd μ F β K κ δ hdual hC hβ hG1 hG2 hP1 hP2 hfloor hout himp A
  obtain ⟨hμ, hneeds, hval, z', hfit', hslack⟩ := hdual
  obtain ⟨A', hfree, hA'⟩ := hout hG1 hG2 hP1 hP2 A
  have hf := hfloor hG1 hG2 hP1 hP2 A' hfree
  have hi := himp A
  refine le_trans (prob_mono fun ox h => ?_) ((prob_or_le _ _).trans (add_le_add hf hi))
  obtain ⟨⟨hw, hn⟩, z, hfit, hnot⟩ := h
  by_cases hK : A.ext ox.1 ox.2 ≤ K
  · left
    obtain ⟨htr, hcost⟩ := hA' ox.1 ox.2
    refine ⟨show vw ox.1 ox.2 (A'.transcript ox.1 ox.2) () by rw [htr]; exact hw, not_le.mp fun hge => hnot ?_⟩
    have hext : β * A.ext ox.1 ox.2 ≤ β * K := mul_le_mul_of_nonneg_left hK hβ
    exact Pouw.SecurityProofs.Dither.FreedOfFloor G Pd 1 _ (A.timedUse ox.1 ox.2) μ z z' hμ hneeds hval zero_le_one
      (div_nonneg (le_max_left _ _) hC.le) hfit' hslack hfit (floor_of_max hC (by linarith))
  · exact Or.inr ⟨hn, not_le.mp hK⟩

/-- **The certifier's sure count is its count at `δ = 0`**: `ImportWithinCertifier` gives `CertifiedImport` with no
error. -/
theorem certifiedImport_of_within {G : Accel} {P : RoundProto} {k : ℕ} {Obs : Type} {verdict : Verdict P k Obs}
    {dev : Readings G P k Obs} {K : ℝ} (h : ImportWithinCertifier G P k verdict dev K) :
    CertifiedImport G P k verdict dev K 0 := fun A =>
  (prob_mono fun ox hox => (not_le.mpr hox.2 (h A ox.1 ox.2 hox.1) : False)).trans Dither.prob_false.le

end Guarantee

section PerProgram

variable {G : Accel} {P : RoundProto} {k : ℕ}

/-- `A` with its programs replaced: the same state type, start and absorption, and the commit phases `fc`, work `fw`
and answers `fa`. -/
def withPrograms (A : Prover G P k) (fc : Fin k → G.Mc.Alg (P.Orc × A.St) (A.St × P.Inst))
    (fw : Fin k → G.Mc.Alg (P.Orc × A.St × P.Draw) (A.St × P.Root))
    (fa : Fin k → G.Mc.Alg (P.Orc × A.St × (Fin k → P.Draw × P.Chk)) P.Ans) : Prover G P k where
  St := A.St
  start := A.start
  commit := fc
  work := fw
  absorb := A.absorb
  answer := fa

/-- A period's priced timed work is its sampling blocks' commit phases' and work's priced cost. -/
theorem timed_cost (A : Prover G P k) (μ : G.Res → ℝ) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) :
    ∑ r, μ r * A.timedUse o x r =
      ∑ b, (G.cost μ (A.commit b) (A.commitIn o x b) + G.cost μ (A.work b) (A.workIn o x b)) := by
  simp only [Prover.timedUse, Accel.cost, mul_sum, mul_add, sum_add_distrib]
  congr 1 <;> exact sum_comm

variable (A : Prover G P k) {fc : Fin k → G.Mc.Alg (P.Orc × A.St) (A.St × P.Inst)}
  {fw : Fin k → G.Mc.Alg (P.Orc × A.St × P.Draw) (A.St × P.Root)}
  {fa : Fin k → G.Mc.Alg (P.Orc × A.St × (Fin k → P.Draw × P.Chk)) P.Ans}

theorem withPrograms_state (hc : ∀ b i, G.Mc.eval (fc b) i = G.Mc.eval (A.commit b) i)
    (hw : ∀ b i, G.Mc.eval (fw b) i = G.Mc.eval (A.work b) i) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) :
    ∀ n, (withPrograms A fc fw fa).state o x n = A.state o x n := by
  intro n
  induction n with
  | zero => rfl
  | succ b ih =>
    simp only [Prover.state]
    split_ifs with hb
    · rw [ih]
      show A.absorb ⟨b, hb⟩ (G.Mc.eval (fw ⟨b, hb⟩)
          (o, (G.Mc.eval (fc ⟨b, hb⟩) (o, A.state o x b)).1, (x ⟨b, hb⟩).1)).1 (x ⟨b, hb⟩).2 = _
      rw [hc, hw]
    · exact ih

theorem withPrograms_commitIn (hc : ∀ b i, G.Mc.eval (fc b) i = G.Mc.eval (A.commit b) i)
    (hw : ∀ b i, G.Mc.eval (fw b) i = G.Mc.eval (A.work b) i) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (b : Fin k) :
    (withPrograms A fc fw fa).commitIn o x b = A.commitIn o x b :=
  congrArg (fun s : A.St => (o, s)) (withPrograms_state A (fa := fa) hc hw o x b)

theorem withPrograms_workIn (hc : ∀ b i, G.Mc.eval (fc b) i = G.Mc.eval (A.commit b) i)
    (hw : ∀ b i, G.Mc.eval (fw b) i = G.Mc.eval (A.work b) i) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (b : Fin k) :
    (withPrograms A fc fw fa).workIn o x b = A.workIn o x b := by
  show (o, (G.Mc.eval (fc b) ((withPrograms A fc fw fa).commitIn o x b)).1, (x b).1) =
    (o, (G.Mc.eval (A.commit b) (A.commitIn o x b)).1, (x b).1)
  rw [withPrograms_commitIn A (fa := fa) hc hw o x b, hc]

/-- Programs that compute what `A`'s do send `A`'s transcript. -/
theorem withPrograms_transcript (hc : ∀ b i, G.Mc.eval (fc b) i = G.Mc.eval (A.commit b) i)
    (hw : ∀ b i, G.Mc.eval (fw b) i = G.Mc.eval (A.work b) i)
    (ha : ∀ b i, G.Mc.eval (fa b) i = G.Mc.eval (A.answer b) i) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) :
    (withPrograms A fc fw fa).transcript o x = A.transcript o x := by
  funext b
  show ((G.Mc.eval (fc b) ((withPrograms A fc fw fa).commitIn o x b)).2,
      (G.Mc.eval (fw b) ((withPrograms A fc fw fa).workIn o x b)).2,
      G.Mc.eval (fa b) (o, (G.Mc.eval (fw b) ((withPrograms A fc fw fa).workIn o x b)).1, x)) =
    ((G.Mc.eval (A.commit b) (A.commitIn o x b)).2, (G.Mc.eval (A.work b) (A.workIn o x b)).2,
      G.Mc.eval (A.answer b) (o, (G.Mc.eval (A.work b) (A.workIn o x b)).1, x))
  rw [withPrograms_commitIn A (fa := fa) hc hw, withPrograms_workIn A (fa := fa) hc hw, hc, hw, ha]

/-- Replacing only the answers leaves the timed use, which no answer is in. -/
theorem withPrograms_timedUse (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (r : G.Res) :
    (withPrograms A A.commit A.work fa).timedUse o x r = A.timedUse o x r := by
  unfold Prover.timedUse
  refine sum_congr rfl fun b _ => ?_
  show G.use (A.commit b) ((withPrograms A A.commit A.work fa).commitIn o x b) r +
      G.use (A.work b) ((withPrograms A A.commit A.work fa).workIn o x b) r = _
  rw [withPrograms_commitIn A (fc := A.commit) (fw := A.work) (fa := fa) (fun _ _ => rfl) (fun _ _ => rfl),
    withPrograms_workIn A (fc := A.commit) (fw := A.work) (fa := fa) (fun _ _ => rfl) (fun _ _ => rfl)]

/-- **The per-program price of an exchanged bit implies its period form**: each sampling block's
commit phase, work and answer replaced by its exchange-free one, the prover sends the same transcript, and its timed
work costs at most `β` a bit its commit phases and work exchanged more, which is at most `β` a bit its programs
exchanged (an answer's bits are a count, G2). So `OutsourceWithinEgressPeriod` asks no more of the model than
`OutsourceWithinEgress`. -/
theorem outsourcePeriod_of_perProgram (G : Accel) (P : RoundProto) (k : ℕ) (μ : G.Res → ℝ) (β : ℝ) (hβ : 0 ≤ β) :
    OutsourceWithinEgress G μ β → OutsourceWithinEgressPeriod G P k μ β := by
  intro h hG1 hG2 hP1 hP2 A
  choose fc hfc using fun b => h hG1 hG2 hP1 hP2 (A.commit b)
  choose fw hfw using fun b => h hG1 hG2 hP1 hP2 (A.work b)
  choose fa hfa using fun b => h hG1 hG2 hP1 hP2 (A.answer b)
  have hc : ∀ b i, G.Mc.eval (fc b) i = G.Mc.eval (A.commit b) i := fun b i => (hfc b i).1
  have hw : ∀ b i, G.Mc.eval (fw b) i = G.Mc.eval (A.work b) i := fun b i => (hfw b i).1
  refine ⟨withPrograms A fc fw fa, fun b c i j => ⟨(hfc b c).2.1, (hfw b i).2.1, (hfa b j).2.1⟩,
    fun o x => ⟨withPrograms_transcript A hc hw (fun b j => (hfa b j).1) o x, ?_⟩⟩
  rw [timed_cost, timed_cost]
  have hS : ∑ b, (G.ext (A.commit b) (A.commitIn o x b) + G.ext (A.work b) (A.workIn o x b)) ≤ A.ext o x :=
    sum_le_sum fun b _ => le_add_of_nonneg_right (hG2.2 (A.answer b) (A.answerIn o x b)).1
  have hβS := mul_le_mul_of_nonneg_left hS hβ
  calc ∑ b, (G.cost μ ((withPrograms A fc fw fa).commit b) ((withPrograms A fc fw fa).commitIn o x b) +
        G.cost μ ((withPrograms A fc fw fa).work b) ((withPrograms A fc fw fa).workIn o x b))
      = ∑ b, (G.cost μ (fc b) (A.commitIn o x b) + G.cost μ (fw b) (A.workIn o x b)) := by
        refine sum_congr rfl fun b _ => ?_
        show G.cost μ (fc b) ((withPrograms A fc fw fa).commitIn o x b) +
            G.cost μ (fw b) ((withPrograms A fc fw fa).workIn o x b) = _
        rw [withPrograms_commitIn A (fa := fa) hc hw o x b, withPrograms_workIn A (fa := fa) hc hw o x b]
    _ ≤ ∑ b, ((G.cost μ (A.commit b) (A.commitIn o x b) + β * G.ext (A.commit b) (A.commitIn o x b)) +
          (G.cost μ (A.work b) (A.workIn o x b) + β * G.ext (A.work b) (A.workIn o x b))) :=
        sum_le_sum fun b _ => add_le_add ((hfc b _).2.2) ((hfw b _).2.2)
    _ = ∑ b, (G.cost μ (A.commit b) (A.commitIn o x b) + G.cost μ (A.work b) (A.workIn o x b)) +
          β * ∑ b, (G.ext (A.commit b) (A.commitIn o x b) + G.ext (A.work b) (A.workIn o x b)) := by
        rw [mul_sum, ← sum_add_distrib]
        exact sum_congr rfl fun b _ => by ring
    _ ≤ ∑ b, (G.cost μ (A.commit b) (A.commitIn o x b) + G.cost μ (A.work b) (A.workIn o x b)) +
          β * A.ext o x := by linarith

end PerProgram

namespace OnsiteFreedWitness

open Pouw.Witness Pouw.SecurityProofs.DitherWitness

/-- The work check at the witness: every sampling block's check on the transcript at its committed instance. It reads
no device. -/
def twWork : Verdict twRound 2 Unit := fun o x t _ =>
  ∀ w, twRound.accept o (t w).1 (x w).1 (x w).2 (t w).2.1 (t w).2.2

/-- The network certifier's check at the witness: the witness machine exchanges nothing, so there is nothing to
refuse. -/
def twNet : Verdict twRound 2 Prop := fun _ _ _ _ => True

/-- The remote floor at the witness: `(1 − ε_W) C`, with `ε_W` `WorkSound`'s error there. -/
noncomputable def twF : ℝ := (1 - twε honest) * 14

theorem twμ0 : twμ 0 = 2 := by norm_num [twμ]

theorem twμ1 : twμ 1 = 1 := by norm_num [twμ]

theorem twμ2 : twμ 2 = 1 := by norm_num [twμ]

theorem twHon0 : twHon 0 = 4 := by norm_num [twHon]

theorem twHon1 : twHon 1 = 4 := by norm_num [twHon]

theorem twHon2 : twHon 2 = 2 := by norm_num [twHon]

theorem twC : ∑ r, twμ r * (1 * twPd.hon r) = 14 := by
  change ∑ r : Fin 3, twμ r * (1 * twHon r) = 14
  rw [Fin.sum_univ_three, twμ0, twμ1, twμ2, twHon0, twHon1, twHon2]
  norm_num

theorem priced3 (X : Fin 3 → ℝ) : ∑ r, twμ r * X r = 2 * X 0 + X 1 + X 2 := by
  rw [Fin.sum_univ_three, twμ0, twμ1, twμ2]
  ring

theorem value3 (W : ℝ) (z : Fin 3 → ℝ) : twPd.value W z = W * 14 + (2 * z 0 + z 1 + z 2) := by
  show W * 14 + ∑ j : Fin 3, twμ j * z j = _
  rw [priced3]

theorem twε_eq (A : Prover tw twRound 2) : twε A = 1 / 7 + Real.log 2 / 2 := by
  simp only [twε, epsWorkSound, twHmm, twHsha, twHo, twη, twδ, twξ, mul_zero, sum_const_zero, zero_div, add_zero,
    Fin.sum_univ_two, one_div_one_div]
  ring

theorem twF_lt : twF < 8 := by
  have h2 := Real.log_two_gt_d9
  rw [twF, twε_eq]
  linarith

theorem twF_pos : 0 < twF := by
  have h2 := Real.log_two_lt_d9
  rw [twF, twε_eq]
  linarith

/-- The onsite bound's error at the witness is `WorkSound`'s there. -/
theorem twEps : max 0 (∑ r, twμ r * (1 * twPd.hon r) - twF + 0 * 0) / ∑ r, twμ r * (1 * twPd.hon r) =
    twε honest := by
  have h2 := Real.log_two_gt_d9
  rw [twC, twF, twε_eq, max_eq_right]
  · ring
  · linarith

theorem one_le_twμ (r : Fin 3) : 1 ≤ twμ r := by
  unfold twμ
  split_ifs <;> norm_num

theorem timedUse_nonneg (A : Prover tw twRound 2) (o : Unit) (x : Fin 2 → TDraw × Fin 2) (r : Fin 3) :
    0 ≤ A.timedUse o x r :=
  sum_nonneg fun b _ =>
    add_nonneg ((tw_G2.2 (A.commit b) (A.commitIn o x b)).2 r).1 ((tw_G2.2 (A.work b) (A.workIn o x b)).2 r).1

/-- An honest strategy's value at the witness period is at most 32: each resource's capacity of 8 less what the
credited part takes, at `μ`, plus the credit `14 W'`, which is the credited part's use at `μ`. -/
theorem honest_le {W' : ℝ} {z' : Fin 3 → ℝ} (h : twPd.Fits (fun r => W' * twPd.hon r) z') :
    twPd.value W' z' ≤ 32 := by
  have e : ∀ r : Fin 3, W' * twHon r + z' r ≤ 8 := fun r => by
    have hr : W' * twHon r + ∑ j : Fin 3, z' j * (if j = r then 1 else 0) ≤ 1 * 8 := h.2 r
    simpa using hr
  have h0 := e 0
  have h1 := e 1
  have h2 := e 2
  rw [twHon0] at h0
  rw [twHon1] at h1
  rw [twHon2] at h2
  rw [value3]
  linarith

/-- **The witness period's arithmetic**: a credited part `X` worth less than `twF` at `μ`, with each resource's rest
filled by its job, is worth more than any honest strategy plus `ε_W C`. -/
theorem fill_not_freed (X : Fin 3 → ℝ) (hX : ∀ r, 0 ≤ X r) (hlt : 2 * X 0 + X 1 + X 2 < twF) {ε : ℝ}
    (hε : ε = twε honest) :
    ∃ z, twPd.Fits X z ∧ ¬ twPd.FreedWithin 1 z ε (∑ r, twμ r * (1 * twPd.hon r)) := by
  have hle : ∀ r, X r ≤ ∑ r, twμ r * X r := fun r =>
    (le_mul_of_one_le_left (hX r) (one_le_twμ r)).trans
      (single_le_sum (f := fun r => twμ r * X r) (fun r _ => mul_nonneg (zero_le_one.trans (one_le_twμ r)) (hX r))
        (mem_univ r))
  have hX8 : ∀ r, X r ≤ 8 := fun r => by
    have h := hle r
    rw [priced3] at h
    linarith [twF_lt]
  refine ⟨fun j => 8 - X j, ⟨fun j => sub_nonneg.2 (hX8 j), fun r => ?_⟩, ?_⟩
  · show X r + ∑ j : Fin 3, (8 - X j) * (if j = r then 1 else 0) ≤ 1 * 8
    simp
  · rintro ⟨W', z', -, hfit, hval⟩
    have hv := honest_le hfit
    have e1 : twPd.value 1 (fun j => 8 - X j) = 46 - (2 * X 0 + X 1 + X 2) := by
      show 1 * 14 + ∑ j : Fin 3, twμ j * (8 - X j) = _
      rw [priced3]
      ring
    rw [twC, hε, twε_eq] at hval
    rw [twF, twε_eq] at hlt
    linarith

/-- An answer at no use, with `f`'s outputs. A witness answer outputs neither words nor a root, so no law charges
it. -/
noncomputable def freeAnswer {I : Type} (f : TAlg I (Fin 1 → Word × Unit)) : TAlg I (Fin 1 → Word × Unit) where
  run := f.run
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_ttr]; omega)
  roots h _ := absurd (congrArg Nat.card h) (by rw [card_ans, card_troot]; omega)
  work S _ h _ := absurd (congrArg Nat.card h) (by rw [card_ans, Nat.card_prod, card_troot]; omega)

/-- `A` with every answer at no use: the same transcript and timed use, and every answer runs within the period. -/
noncomputable def fitAnswers (A : Prover tw twRound 2) : Prover tw twRound 2 :=
  withPrograms A A.commit A.work fun b => freeAnswer (A.answer b)

theorem fitAnswers_transcript (A : Prover tw twRound 2) (o : twRound.Orc)
    (x : Fin 2 → twRound.Draw × twRound.Chk) : (fitAnswers A).transcript o x = A.transcript o x :=
  withPrograms_transcript A (fc := A.commit) (fw := A.work) (fa := fun b => freeAnswer (A.answer b))
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) o x

theorem fitAnswers_timedUse (A : Prover tw twRound 2) (o : twRound.Orc)
    (x : Fin 2 → twRound.Draw × twRound.Chk) : (fitAnswers A).timedUse o x = A.timedUse o x :=
  funext (withPrograms_timedUse A (fa := fun b => freeAnswer (A.answer b)) o x)

theorem fitAnswers_dev (A : Prover tw twRound 2) (o : Unit) (x : Fin 2 → TDraw × Fin 2) :
    twDev (fitAnswers A) o x :=
  fun _ _ => show (0 : ℝ) ≤ 1 * 8 by norm_num

/-- **The remote floor at the witness**: from `WorkSound`'s conclusion there (`witness_freed`), at the prover with
every answer at no use, which sends the same transcript, has the same timed use and passes the devices' reading. A
prover whose timed work costs less than `(1 − ε_W) C` at `μ` uses each resource below its capacity (`μ ≥ 1`), and the
rest of each resource, filled with its other job, is worth more than any honest strategy's other work plus
`ε_W C`. -/
theorem twFloor : RemoteWorkFloor tw twRound 2 twWork twμ twF (1 / 2) := by
  intro _ _ _ _ A _
  refine le_trans (prob_mono fun ox h => ?_) ((witness_freed (fitAnswers A)).trans (by norm_num [twκ, twδ]))
  obtain ⟨hp, hlt⟩ := h
  have hlt3 : ∑ r : Fin 3, twμ r * A.timedUse ox.1 ox.2 r < twF := hlt
  rw [priced3] at hlt3
  have hε : twε (fitAnswers A) = twε honest := by rw [twε_eq, twε_eq]
  have hw : twWork ox.1 ox.2 ((fitAnswers A).transcript ox.1 ox.2) () := by
    rw [fitAnswers_transcript]
    exact hp
  refine ⟨⟨hw, fitAnswers_dev A ox.1 ox.2⟩, ?_⟩
  rw [fitAnswers_timedUse]
  exact fill_not_freed (A.timedUse ox.1 ox.2) (timedUse_nonneg A ox.1 ox.2) hlt3 hε

/-- A quiet sampling block's work: a constant state and root, at no use. -/
noncomputable def quietWork : TAlg (Unit × Unit × TDraw) (Unit × TRoot) where
  run _ := ((), fun _ _ _ => 0)
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words _ _ := by simp [nonconst]
  roots _ _ := by simp [nonconst]
  work _ _ _ _ := by simp [nonconst]

/-- A quiet answer: a constant, at no use. -/
noncomputable def quietAnswer : TAlg (Unit × Unit × (Fin 2 → TDraw × Fin 2)) (Fin 1 → Word × Unit) where
  run _ := fun _ => (0, ())
  use _ _ := 0
  use_nonneg _ _ := le_rfl
  calls _ := 0
  calls_nonneg _ := le_rfl
  calls_le _ := le_rfl
  words _ _ := by simp [nonconst]
  roots _ _ := by simp [nonconst]
  work _ _ _ _ := by simp [nonconst]

/-- **The quiet prover**: it does nothing, so its sampling blocks use nothing. -/
noncomputable def quiet : Prover tw twRound 2 where
  St := Unit
  start := ()
  commit _ := constAlg ((), twInst)
  work _ := quietWork
  absorb _ s _ := s
  answer _ := quietAnswer

theorem quiet_timedUse (o : Unit) (x : Fin 2 → TDraw × Fin 2) (r : Fin 3) : quiet.timedUse o x r = 0 := by
  show ∑ _b : Fin 2, ((0 : ℝ) + 0) = 0
  simp

theorem prob_eq_one {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  unfold prob
  rw [filter_true_of_mem fun ω _ => h ω, card_univ, div_self (by exact_mod_cast Fintype.card_ne_zero)]

end OnsiteFreedWitness

section Witnesses

open Pouw.Witness Pouw.SecurityProofs.DitherWitness Pouw.SecurityProofs.OnsiteFreedWitness

/-- **`OnsiteFreed`'s honest witness**: at `Definitions.Pouw.Witness`'s machine, round and period,
every premise of `OnsiteFreed` holds: the dual price, `C = 14 > 0`, `β = 0`, G1, G2, P1 and P2, the remote floor at
`F = (1 − ε_W) C` and `κ = 1/2` (from `WorkSound` there), the period form of the outsourcing premise (from the
per-program one) and the certifier's count at `K = 0` and `δ = 0`. The bound says something (`ε = ε_W < 1`,
`κ + δ = 1/2`), and the onsite verdict accepts the honest prover, which exchanges nothing, on every oracle and
coins. -/
theorem OnsiteFreed_honest :
    twPd.DualOpt twμ 1 ∧ 0 < ∑ r, twμ r * (1 * twPd.hon r) ∧ (0 : ℝ) ≤ 0 ∧
      tw.G1 ∧ tw.G2 ∧ tw.P1 ∧ tw.P2 ∧
      RemoteWorkFloor tw twRound 2 twWork twμ twF (1 / 2) ∧
      OutsourceWithinEgressPeriod tw twRound 2 twμ 0 ∧ CertifiedImport tw twRound 2 twNet twDev 0 0 ∧
      max 0 (∑ r, twμ r * (1 * twPd.hon r) - twF + 0 * 0) / ∑ r, twμ r * (1 * twPd.hon r) < 1 ∧
      (1 / 2 + 0 : ℝ) < 1 ∧
      honest.ExchangeFree ∧ ∀ o x, honest.Passes (onsiteVerdict twWork twNet) twDev o x := by
  refine ⟨twDual, by rw [twC]; norm_num, le_rfl, tw_G1, tw_G2, trivial, trivial, twFloor,
    outsourcePeriod_of_perProgram tw twRound 2 twμ 0 le_rfl twOutsource,
    certifiedImport_of_within fun _ _ _ _ => ?_,
    by rw [twEps]; exact twε_lt honest, by norm_num, fun _ _ _ _ => ⟨rfl, rfl, rfl⟩,
    fun o x => ⟨fun w => (honest_accepted o x w).2, trivial⟩⟩
  show ∑ _b : Fin 2, ((0 : ℝ) + 0 + 0) ≤ 0
  simp

/-- **`OnsiteFreed`'s nontrivial witness**: its conclusion at the honest witness's instance
(`ε = ε_W`, `C = 14`, `κ + δ = 1/2`) is false at verdicts that accept everything. The quiet prover is accepted on
every oracle and coins and leaves the period to other work worth 46, against at most `32 + ε C < 46` for any honest
strategy, so the event has probability 1. -/
theorem OnsiteFreed_nontrivial :
    ¬ FreedSound tw twRound 2 (onsiteVerdict (fun _ _ _ _ => True) (fun _ _ _ _ => True)) twDev twPd 1
      (fun _ => max 0 (∑ r, twμ r * (1 * twPd.hon r) - twF + 0 * 0) / ∑ r, twμ r * (1 * twPd.hon r))
      (∑ r, twμ r * (1 * twPd.hon r)) (fun _ => 1 / 2 + 0) := by
  intro h
  have : Nonempty (twRound.Orc × (Fin 2 → twRound.Draw × twRound.Chk)) :=
    (inferInstance : Nonempty (Unit × (Fin 2 → TDraw × Fin 2)))
  have h1 := h quiet
  rw [prob_eq_one] at h1
  · norm_num at h1
  intro ox
  refine ⟨⟨trivial, trivial⟩, fill_not_freed (quiet.timedUse ox.1 ox.2) (timedUse_nonneg quiet ox.1 ox.2) ?_ twEps⟩
  have h0 := quiet_timedUse ox.1 ox.2 0
  have h1 := quiet_timedUse ox.1 ox.2 1
  have h2 := quiet_timedUse ox.1 ox.2 2
  linarith [twF_pos]

end Witnesses

end Pouw.SecurityProofs
