module

public import Flock.Net
public import Properties.Flock.Circuit.Rows
public import Proofs.Flock.EndToEndRegistered.Inputs
public import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
public import Proofs.Flock.Soundness.Discharge.ZkLink.CompiledLink
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
public import Proofs.Flock.Soundness.Instance
import Proofs.Core.Game.Expect
import Proofs.Core.Game.Prob
import Proofs.Flock.Soundness.CROnly.Link
import Proofs.Flock.Soundness.ExtractWait
import Proofs.Flock.Soundness.Refine.Walk
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# `EndToEndRegistered`'s premises hold together (its honest witness)

`Flock.SecurityProofs.EndToEndRegistered_honest` concludes, at one instance, each premise of
`Flock.Guarantees.EndToEndRegistered` that no named assumption states, and
`Flock.SecurityProofs.EndToEndRegisteredDrawn_honest` the same of `EndToEndRegisteredDrawn`, which takes the same four:

* `Built`: the one-row unit net `unitNet`, which `Net.ofRows` builds (a constant row and one output word);
* `TagsOkZHJ`: `verify --zk`'s inputs at `verity/flock-circuit/tmpl`'s tags (`Flock.Tags.circuitTmpl`), the templated
  statement the deployed verifier serves;
* `LinkCRZC` at a run's cost `2^256`, for every prover: a link finder's cost is at least one run's
  (`le_finderCost_cTZC`), so `ecr/sha-512`'s bound is at least 1 (`crExpected_of_le`);
* `SessCRZ` at budgets `2^257`, for every prover: `cr/sha-512`'s bound `q²/2^513` is then 2 (`finder_cr_top`).

The collision premises are hypotheses on the prover's finders at the budgets the bound is read at; at a deployed budget
they are `ecr/sha-512` and `cr/sha-512` (`Claims.ExpectedCR`), which no theorem concludes. These budgets show that
neither is a premise that can never hold.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- **`ecr/sha-512` at a finder whose every run costs at least `2^256`**: its bound, expected cost over `2^256`, is at
least 1. -/
theorem crExpected_of_le {β Dh : Type} {H : List UInt8 → Dh} {g : Game β} {s : Strategy g}
    {out : β → Option (List UInt8 × List UInt8)} {cost : β → ℝ} (h : ∀ b, (2 : ℝ) ^ 256 ≤ cost b) :
    Assumptions.SHA512CRExpected H g s out cost := by
  unfold Assumptions.SHA512CRExpected
  have h1 := ENNReal.toReal_mono ENNReal.one_ne_top
    (Verity.Game.prob_le_one (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s)
  rw [ENNReal.toReal_one] at h1
  have h2 := Verity.Game.expect_mono (f := fun _ => (2 : ℝ) ^ 256) h g s
  rw [Verity.Game.expect_const] at h2
  rw [show (2 : ℝ) ^ (256 : ℝ) = 2 ^ 256 by norm_num, le_div_iff₀ (by positivity)]
  have h3 := mul_le_mul_of_nonneg_right h1 (by positivity : (0 : ℝ) ≤ 2 ^ 256)
  rw [one_mul] at h3
  linarith

theorem one_le_crBound : (1 : ℝ) ≤ (2 ^ 257) ^ 2 / 2 ^ (513 : ℝ) := by
  rw [show (2 : ℝ) ^ (513 : ℝ) = 2 ^ 513 by norm_num, ← pow_mul, le_div_iff₀ (by positivity), one_mul]
  exact pow_le_pow_right₀ (by norm_num) (by norm_num)

/-- **`cr/sha-512` at budget `2^257` holds at every finder**: its bound `q²/2^513` is 2. -/
theorem finder_cr_top {Dh : Type} (H : List UInt8 → Dh) (Fd : Finder Pair) : Fd.CR H (2 ^ 257) := fun _ =>
  (ENNReal.toReal_le_of_le_ofReal zero_le_one
    ((Verity.Game.prob_le_one _ _ _).trans ENNReal.ofReal_one.ge)).trans one_le_crBound

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **`SessCRZ` at budgets `2^257`**, at every session strategy. -/
theorem sessCRZ_top (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    SessCRZ A Hs T pre post cap₀ σ (2 ^ 257) (2 ^ 257) :=
  ⟨fun _ _ _ _ => finder_cr_top _ _, fun _ _ => finder_cr_top _ _, fun _ _ => finder_cr_top _ _,
    finder_cr_top _ _, finder_cr_top _ _⟩

end Session

section Link

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

omit [Fintype K] in
/-- **A link finder costs at least one run** (`t'`): its first trial plays the fresh run. -/
theorem le_finderCost_cTZC (c : vb.Pos) {t' : ℝ} (ht : 0 ≤ t')
    (x : TrialOutZC F L D Sl × List (TrialOutZC F L D Sl)) :
    t' ≤ finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t') x := by
  classical
  have hc : ∀ e : TrialOutZC F L D Sl, t' ≤ cTZC planZ R t' e := fun e => by
    unfold cTZC
    split_ifs
    · have := List.sum_nonneg (l := e.2.2.map (waitCost (accωZC planZ R e.1))) fun y hy => by
        obtain ⟨b, -, rfl⟩ := List.mem_map.mp hy
        exact waitCost_nonneg _ b
      nlinarith [mul_nonneg ht this]
    · simp
  have h2 : 0 ≤ if (rdTZC A Hs P planZ tab vb R τ c x.1).card = 1 then
      loopCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t') x.2 else 0 := by
    split_ifs
    · exact CROnly.loopCost_nonneg _ (fun e => ht.trans (hc e)) _
    · exact le_rfl
  unfold finderCost
  linarith [hc x.1]

omit [Fintype K] in
/-- **`LinkCRZC` at a run's cost `2^256`**, at every prover. -/
theorem linkCRZC_top (k Rw M : ℕ) : LinkCRZC A Hs P planZ tab vb R τ k Rw M (2 ^ 256) :=
  fun c _ => crExpected_of_le (le_finderCost_cTZC A Hs P planZ tab vb R τ c (by positivity))

end Link

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.ZkReg.Witness

/-- **The witness's unit net**: one useful row, the constant, and one output group of one word. -/
def unitNet : Flock.Net :=
  match Flock.Net.ofRows "unit" 1 0 #[] #[{ col := 0, words := 1, bits := #[1] }] #[#[0]] #[#[0]] with
  | .ok net => net
  | .error _ => default

theorem unitNet_built : Built unitNet :=
  ⟨"unit", 1, 0, #[], #[{ col := 0, words := 1, bits := #[1] }], #[#[0]], #[#[0]], by kernel_rfl⟩

end FlockSoundness.Discharge.ZkReg.Witness

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession StrictCR

/-- **`EndToEndRegistered`'s premises hold together.** The unit net is built, the inputs carry the templated
statement's tags, and every prover's link and session finders satisfy the collision premises at budgets `2^256` and
`2^257`. -/
theorem _root_.Flock.SecurityProofs.EndToEndRegistered_honest :
    Built ZkReg.Witness.unitNet ∧ ZkHidden.TagsOkZHJ ZkReg.Witness.ins ∧
      (∀ {mPts : ℕ} {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
          (planZ : Finset (Fin n) → Reg → List (TabZK FlockLevel3.GF128 mPts))
          (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) {L : Law n}
          {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
          (vb : ValueBindingZ execArith P planZ tab decode Hc) (R : Reg)
          (τ : Cont L Reg (batchedSessionZC execArith ZkExec.hs512 planZ) R) (k Rw M : ℕ),
        LinkCRZC execArith ZkExec.hs512 P planZ tab vb R τ k Rw M (2 ^ 256)) ∧
      ∀ {mPts : ℕ} (T : TabZK FlockLevel3.GF128 mPts)
        (pre post : List ((Fin 2 → Fin mPts → FlockLevel3.GF128) → Game (OutZC FlockLevel3.GF128 Refine.Dig (List UInt8))))
        (cap₀ : Cap Refine.Dig) (σ : Strategy (sessAtZ pre post (tabAfterZC execArith ZkExec.hs512 T cap₀))),
        SessCRZ execArith ZkExec.hs512 T pre post cap₀ σ (2 ^ 257) (2 ^ 257) :=
  ⟨ZkReg.Witness.unitNet_built, ZkReg.Witness.ins_tagsOk, by intros; apply linkCRZC_top, by intros; apply sessCRZ_top⟩

/-- **`EndToEndRegisteredDrawn`'s premises hold together**: they are `EndToEndRegistered`'s. -/
theorem _root_.Flock.SecurityProofs.EndToEndRegisteredDrawn_honest :
    Built ZkReg.Witness.unitNet ∧ ZkHidden.TagsOkZHJ ZkReg.Witness.ins ∧
      (∀ {mPts : ℕ} {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
          (planZ : Finset (Fin n) → Reg → List (TabZK FlockLevel3.GF128 mPts))
          (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) {L : Law n}
          {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
          (vb : ValueBindingZ execArith P planZ tab decode Hc) (R : Reg)
          (τ : Cont L Reg (batchedSessionZC execArith ZkExec.hs512 planZ) R) (k Rw M : ℕ),
        LinkCRZC execArith ZkExec.hs512 P planZ tab vb R τ k Rw M (2 ^ 256)) ∧
      ∀ {mPts : ℕ} (T : TabZK FlockLevel3.GF128 mPts)
        (pre post : List ((Fin 2 → Fin mPts → FlockLevel3.GF128) → Game (OutZC FlockLevel3.GF128 Refine.Dig (List UInt8))))
        (cap₀ : Cap Refine.Dig) (σ : Strategy (sessAtZ pre post (tabAfterZC execArith ZkExec.hs512 T cap₀))),
        SessCRZ execArith ZkExec.hs512 T pre post cap₀ σ (2 ^ 257) (2 ^ 257) :=
  Flock.SecurityProofs.EndToEndRegistered_honest

end FlockSoundness.Discharge.ZkLink
