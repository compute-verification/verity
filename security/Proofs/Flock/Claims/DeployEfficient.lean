module

public import Proofs.Flock.Claims.GS.RadiusDeploy
public import Proofs.Flock.Claims.ERepCost
public import Proofs.Flock.Claims.DeployCode

/-! Flock's efficiency claim (`Flock.Efficient`) at both deployed records. -/

@[expose] public section

namespace Claims

open Polynomial

theorem Flock.Efficient_of_size {Coin F T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype F]
    [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Flock.Deploy Coin (List F) T)
    (size : List (ℕ × List F) → ℕ) (q a : Polynomial ℕ) (hl : Flock.ListWithin D.len D.enc D.θ D.decode)
    (hs : ∀ os, (D.dec os).2 ≤ q.eval (size os)) (hlen : ∀ os, (D.decode os).length ≤ q.eval (size os))
    (ha : ∀ os, size os ≤ a.eval (Flock.wordLen os)) : Flock.Efficient H D (q.comp a) := by
  have hq : ∀ os, q.eval (size os) ≤ (q.comp a).eval (Flock.wordLen os) := fun os => by
    rw [eval_comp]
    exact GS.eval_mono_nat q (ha os)
  exact Flock.Efficient_of_decoder H D _ hl (fun os => (hs os).trans (hq os)) fun os => (hlen os).trans (hq os)

theorem Flock.le_eval_wordLen {F : Type} [Fintype F] (size : List (ℕ × List F) → ℕ) (c : ℕ)
    (h : ∀ os, size os ≤ c * (os.length + 1)) (os : List (ℕ × List F)) :
    size os ≤ (C c * (Polynomial.X + 1)).eval (Flock.wordLen os) := by
  simp only [eval_mul, eval_C, eval_add, eval_X, eval_one]
  exact (h os).trans (Nat.mul_le_mul_left c (Nat.add_le_add_right (Flock.length_le_wordLen os) 1))

theorem Flock.not_efficient_of_nil {Coin F T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype F]
    [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Flock.Deploy Coin (List F) T) (p : Polynomial ℕ)
    (hp : p.eval 0 < (D.dec []).2) : ¬ Flock.Efficient H D p := fun hE => by
  have h := hE.2.2.1 []
  simp only [Flock.wordLen, List.map_nil, List.sum_nil] at h
  omega

end Claims

namespace FlockDeploy

open Flock Polynomial

local notation "𝔽" => FlockLevel3.GF128

theorem decode_plain (rel : Readable.Parsed → Readable.Relation) :
    (deployPlainTrim rel).decode = Claims.GS.plainDecodeRD :=
  funext Claims.GS.plainCountedRD_fst

theorem decode_zk (rel : Readable.Parsed → Readable.Relation) : (deployZkTrim rel).decode = Claims.GS.zkDecodeRD :=
  funext Claims.GS.zkCountedRD_fst

theorem listWithin_plain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployPlain rel).len (deployPlain rel).enc (deployPlain rel).θ Claims.GS.plainDecodeRD :=
  Claims.GS.listWithin_plainDecodeRD id fun _ _ => rfl

theorem listWithin_zk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployZk rel).len (deployZk rel).enc (deployZk rel).θ Claims.GS.zkDecodeRD :=
  Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl

theorem listWithin_plainTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployPlainTrim rel).len (deployPlainTrim rel).enc (deployPlainTrim rel).θ
      Claims.GS.plainDecodeRD :=
  Claims.GS.listWithin_plainDecodeRD trimCol trimCol_getD

theorem listWithin_zkTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployZkTrim rel).len (deployZkTrim rel).enc (deployZkTrim rel).θ Claims.GS.zkDecodeRD :=
  Claims.GS.listWithin_zkDecodeRD trimCol trimCol_getD

noncomputable def plainStepPoly : Polynomial ℕ :=
  (C 14 * (Claims.GS.flockPolyR + Polynomial.X)).comp (C (Claims.GS.plainSizeRD []) * (Polynomial.X + 1))

noncomputable def zkStepPoly : Polynomial ℕ :=
  (C 15 * (Claims.GS.zkPolyR + Polynomial.X)).comp (C (Claims.GS.zkSizeRD []) * (Polynomial.X + 1))

theorem efficient_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployPlainTrim rel) plainStepPoly := by
  refine Claims.Flock.Efficient_of_size H (deployPlainTrim rel) Claims.GS.plainSizeRD _ _ ?_
    (fun os => (Claims.GS.plainDecodeRD_complexity os).2) ?_
    (Claims.Flock.le_eval_wordLen _ _ Claims.GS.plainSizeRD_le)
  · rw [decode_plain]
    exact listWithin_plainTrim rel
  · rw [decode_plain]
    exact Claims.GS.plainDecodeRD_length_le

theorem efficient_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployZkTrim rel) zkStepPoly := by
  refine Claims.Flock.Efficient_of_size H (deployZkTrim rel) Claims.GS.zkSizeRD _ _ ?_
    (fun os => (Claims.GS.zkDecodeRD_complexity os).2) ?_
    (Claims.Flock.le_eval_wordLen _ _ Claims.GS.zkSizeRD_le)
  · rw [decode_zk]
    exact listWithin_zkTrim rel
  · rw [decode_zk]
    exact Claims.GS.zkDecodeRD_length_le

theorem not_efficient_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (p : Polynomial ℕ) (hp : p.eval 0 = 0) : ¬ Claims.Flock.Efficient H (deployPlainTrim rel) p :=
  Claims.Flock.not_efficient_of_nil H _ p (by rw [hp]; exact Claims.GS.plainCountedRD_nil_pos)

theorem not_efficient_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (p : Polynomial ℕ) (hp : p.eval 0 = 0) : ¬ Claims.Flock.Efficient H (deployZkTrim rel) p :=
  Claims.Flock.not_efficient_of_nil H _ p (by rw [hp]; exact Claims.GS.zkCountedRD_nil_pos)

theorem erepRuns_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployPlainTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployPlainTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

theorem erepRuns_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.ZkTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployZkTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployZkTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

end FlockDeploy
