import Proofs.Flock.Claims.KS.Extend
import Proofs.Flock.Claims.KS.PlainLeafKS

/-!
# KS at the untrimmed plain instance

The plain verifier's KS at its trimmed record holds with no premise (`plain_KS`), and does not give it at the
untrimmed instance (`FlockDeploy.deployPlain`), whose decoders the sampling claims read
(`Flock.TrimBridge.hole_agree`). The untrimmed instance's KS follows from the same full-width table bound
(`plain_tableBoundθAt_all`) once a table with unopened positions extends (`Claims.Flock.Extend`), which holds at the
plain code by the argument `Proofs.Flock.Claims.KS.Extend` gives at any code, the masked one's (`zk_extendJ`) at the
masked code:

* `plain_reg`, `plain_inj`, `plain_card`, `plain_avoid`: the plain code's regime one agreement short of its radius
  (`n ≥ 2^10` at rate `1/2`, `Claims.GS.reg_of_rate`), its distinct points and its size below `|GF(2^128)|`, and so
  the first lane to fill with (`Claims.GS.exists_avoid`).
* `wide_width`, `plain_fillable`, `plain_extend`: `Extend` at `deployPlain rel`, at every relation.
* `plain_tableBoundθ_orZero`, `plain_KS_untrim`: so the table bound at tables with unopened positions
  (`Claims.Flock.tableBoundθ_of_extend`), and `Flock.KS` at the untrimmed plain instance at SHA-512 at the deployed
  relation with no premise (`plain_KS_of_tableBound`).
-/

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain code's regime one agreement short of its radius**, at every `m` served: `n = 2^d ≥ 2^10` and rate
`1/2`. -/
theorem plain_reg (i : Claims.Flock.PlainIdx) (t : ℕ)
    (ht : (1 - (1 - Real.sqrt ((2 : ℝ) ^ Claims.Flock.plainCols i / 2 ^ Claims.Flock.plainLen i) - 1 / 200)) *
      ((2 ^ Claims.Flock.plainLen i : ℕ) : ℝ) - 1 ≤ t) :
    ((Claims.Flock.flockPlainLevelZero i).k + 1) * 2 ^ Claims.Flock.plainLen i < t * t := by
  show (2 ^ Claims.Flock.plainCols i + 1) * 2 ^ Claims.Flock.plainLen i < t * t
  have hs := Claims.Flock.plainShape_table i
  have hn : 1024 ≤ 2 ^ Claims.Flock.plainLen i :=
    le_trans (by norm_num) (Nat.pow_le_pow_right (by norm_num) (show 10 ≤ Claims.Flock.plainLen i by omega))
  have hk : 2 ^ Claims.Flock.plainLen i ≤ 2 * 2 ^ Claims.Flock.plainCols i := by
    rw [hs.2.2.2.1, pow_succ]
    omega
  refine Claims.GS.reg_of_rate hn hk (le_of_eq_of_le ?_ ht)
  push_cast
  ring

/-- **The plain code's points are distinct** at each index's `2^d` positions. -/
theorem plain_inj (i : Claims.Flock.PlainIdx) :
    Set.InjOn (Claims.Flock.flockPlainLevelZero i).dom {q | q < 2 ^ Claims.Flock.plainLen i} :=
  FlockSoundness.execArith_correct.omega_injective (Claims.Flock.plainLen i)
    ((Claims.Flock.plainShape_table i).2.2.2.2.1.trans (by norm_num))

/-- **The indices' codeword lengths summed stay below `|GF(2^128)|`**: each is at most `2^40`. -/
theorem plain_card : ∑ i : Claims.Flock.PlainIdx, 2 ^ Claims.Flock.plainLen i < Fintype.card 𝔽 := by
  rw [FlockLevel3.card_F]
  calc ∑ i : Claims.Flock.PlainIdx, 2 ^ Claims.Flock.plainLen i ≤ ∑ _i : Claims.Flock.PlainIdx, 2 ^ 40 :=
        Finset.sum_le_sum fun i _ =>
          Nat.pow_le_pow_right (by norm_num) (Claims.Flock.plainShape_table i).2.2.2.2.1
    _ < 2 ^ 128 := by simp

/-- **A first lane no plain trace one agreement short of its radius has at `p`**, at any word `W`, read at the
untrimmed plain instance (`Claims.GS.exists_avoid` at the deployed plain code). -/
theorem plain_avoid (rel : Readable.Parsed → Readable.Relation) (W : ℕ → List 𝔽) (p : ℕ) :
    ∃ c : 𝔽, ∀ M : Claims.Flock.PlainTrace,
      (1 - (deployPlain rel).θ M) * ((deployPlain rel).len M : ℝ) ≤
        (((Finset.univ : Finset (Fin ((deployPlain rel).len M))).filter fun q =>
          (deployPlain rel).enc M q.val = W q.val).card : ℝ) + 1 →
      ((deployPlain rel).enc M p)[0]? ≠ some c :=
  Claims.GS.exists_avoid Claims.Flock.flockPlainLevelZero (fun _ => pow_pos (by norm_num) _)
    (fun i => 2 ^ Claims.Flock.plainLen i)
    (fun i => 1 - Real.sqrt ((2 : ℝ) ^ Claims.Flock.plainCols i / 2 ^ Claims.Flock.plainLen i) - 1 / 200)
    plain_inj plain_reg plain_card W p

/-- **A column of the code's width is a width**, at every nonempty word: the zero column's length is positive. -/
theorem wide_width (zk : Bool) (x : Claims.Msg) (h : 0 < wordLen zk x) : 0 < (zeroCol zk x).length := by
  unfold zeroCol
  split
  · simp only [List.length_replicate]
    positivity
  · rename_i heq
    unfold wordLen at h
    rw [heq] at h
    exact absurd h (lt_irrefl 0)

/-- **Every position of the plain word can be filled**: `Claims.Flock.fillable_of_avoid` at the plain code's first
lane (`plain_avoid`) and the zero column's width (`wide_width`). -/
theorem plain_fillable (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Fillable (deployPlain rel) (fun x _ => wordLen false x) (fun x _ => wide false x) :=
  Claims.Flock.fillable_of_avoid _ _ _ (plain_avoid rel) fun x _ h => ⟨_, wide_width false x h, fun _ hu => hu⟩

/-- **`Extend` at the untrimmed plain instance**, at every relation (`Claims.Flock.extend_of_fill`). -/
theorem plain_extend (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Extend (deployPlain rel) (wordLen false) (wide false) :=
  Claims.Flock.extend_of_fill (plain_fillable rel)

/-- **The plain table bound at tables with unopened positions**, at SHA-512 at the deployed relation: the bound at
full-width tables at the statements some run accepts (`plain_tableBoundθAt_all`) and `Extend` (`plain_extend`). -/
theorem plain_tableBoundθ_orZero :
    Claims.Flock.TableBoundθ sha512 (deployPlain Readable.relation) (wordLen false)
      (Claims.Flock.orZero (wide false)) :=
  Claims.Flock.tableBoundθ_of_extend sha512 _ _ _ (fun x u hN h => wide_ne_zero false x u hN h)
    (plain_colsOk sha512 _) (plain_extend _)
    (plain_tableBoundθ_of_acc Readable.relation (wide false) fun _ _ _ _ _ hno _ _ _ => plain_tableBoundθAt_all hno)

/-- **Plain knowledge soundness at the untrimmed instance**, at SHA-512 at the deployed relation, with no premise:
`plain_KS_of_tableBound` at `plain_tableBoundθ_orZero`. -/
theorem plain_KS_untrim : Claims.Flock.KS sha512 (deployPlain Readable.relation) :=
  plain_KS_of_tableBound Readable.relation (plain_hone _) (plain_colsOk sha512 _) (plain_conflictTied _)
    plain_tableBoundθ_orZero

end FlockDeploy.KS
