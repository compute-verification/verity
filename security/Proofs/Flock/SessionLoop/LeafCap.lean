module

public import Proofs.Flock.SessionLoop.AnyErrBound
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import all Init.Data.Nat.Power2.Basic

/-!
# The deployed masked session's leaf count is at most a constant

`Loop.anyErr`'s hiding terms read the session's pre-final leaf count (`RefSession.nLeaves`), which `Loop.anyErr_le`
takes a cap of as a hypothesis. At the deployed session's tables (`Masked.tabC`) the count is bounded by the table
count times a constant, from what `Zk.schedule` accepts, and the table count by the verifier's cap.

* `ZkHidden.zkLen0_all`: level 0's codeword of `Zk.schedule m` has at most `2^23` symbols at every `m` it accepts,
  one check per `m` (`decide +kernel`), as `zkSchedFacts_all` is.
* `ZkHidden.ZkTab.card_leafPos_le`: a `--zk` table's `Coined` tables have at most `leafCap = 2^23 + 2^14 + 2`
  pre-final leaves. Level 0's opened and unopened positions are its codeword's (`zkSchedFacts_all`: the pad count is
  below the codeword's length); each of the two reps' pads positions are the inner code's, at most `2^13` (`len_le`);
  and the two pads message leaves.
* `Masked.nL_le`: so the session's leaves at `x` are at most `jOf x * leafCap`.
* `Masked.jOf_le_max`: the session has at most `Tags.MAX_TABLES` tables (`ZkTables.tables_of_setup`).
* `Loop.nLeaves_tableData_le`, **`Loop.nLeaves_dataE_le`**: `nL_le` at `CoinLayer.session` of any plans, and of the
  deployed session's (`Masked.dataE`).
* **`Loop.nLeaves_dataE_le_max`**: so the deployed session has at most `Tags.MAX_TABLES * leafCap` leaves at every
  statement, `2^29 + 2^20 + 2^7`.
* **`Loop.anyErr_le_dataE`**: `Loop.anyErr_le` at that constant, with no leaf count left free.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine ZkLink

/-- **The leaf cap of one `--zk` table**: level 0's codeword positions, two reps' pads positions and two pads message
leaves, at the largest sizes `Zk.schedule` accepts (`m = 35`, where level 0's codeword has `2^23` symbols). -/
def leafCap : ℕ := 2 ^ 23 + 2 ^ 14 + 2

/-- Whether level 0's codeword of `Zk.schedule m` has at most `2^23` symbols. -/
def zkLen0 (m : ℕ) : Bool :=
  match Zk.schedule m with
  | .ok (s, _) => decide (((schedOf s).levels.headD default).logLen ≤ 23)
  | .error _ => true

set_option maxRecDepth 100000 in
theorem zkLen0_all : ∀ m, 25 ≤ m → m ≤ 35 → zkLen0 m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **A `--zk` table's `Coined` tables have at most `leafCap` pre-final leaves.** -/
theorem ZkTab.card_leafPos_le (P : Composed.LigProver) {st : Flock.Stmt} (y : ZkTab st) :
    Fintype.card (ZkView.LeafPos (Composed.coinedOf P y.table st).z) ≤ leafCap := by
  obtain ⟨h1, h2⟩ := schedule_range y.hz
  have hf := zkSchedFacts_all _ h1 h2
  simp only [zkSchedFacts, y.hz, Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, b3⟩, -⟩, -⟩, hk0⟩ := hf
  have b1 := zkLen0_all _ h1 h2
  simp only [zkLen0, y.hz, decide_eq_true_eq] at b1
  have hlen := len_le h2 y.wf.1.hk hk0
  have hle : ((layOf st y.sch).len + Zk.K_PAD).nextPowerOfTwo ≤ 2 ^ 10 :=
    Nat.nextPowerOfTwo_le
      (by
        show (Zk.PadLayout.new st.m st.c.kLog y.sch.k0).len + Zk.K_PAD ≤ 2 ^ 10
        simp only [Zk.K_PAD]
        omega)
      ⟨10, rfl⟩
  have hN : Composed.padNOf y.table st ≤ 2 ^ 13 := by
    show ((layOf st y.sch).len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE ≤ 2 ^ 13
    simp only [Zk.INNER_LOG_INV_RATE]
    omega
  have ht : y.tPad ≤ 2 ^ ((schedOf y.sch).levels.headD default).logLen := (Nat.le_add_left _ _).trans b3
  have hT : y.tPad + (2 ^ ((schedOf y.sch).levels.headD default).logLen - y.tPad) ≤ 2 ^ 23 :=
    (Nat.add_sub_of_le ht).trans_le (Nat.pow_le_pow_right (by norm_num) b1)
  have hP : Zk.Q + (Composed.padNOf y.table st - Zk.Q) ≤ 2 ^ 13 := by
    simp only [Zk.Q]
    omega
  have hc : Fintype.card (ZkView.LeafPos (Composed.coinedOf P y.table st).z) =
      (y.tPad + (2 ^ ((schedOf y.sch).levels.headD default).logLen - y.tPad)) +
        (2 * (Zk.Q + (Composed.padNOf y.table st - Zk.Q)) + 2) := by
    show Fintype.card (Fin (y.tPad + (2 ^ ((schedOf y.sch).levels.headD default).logLen - y.tPad)) ⊕
      (Fin 2 × Fin (Zk.Q + (Composed.padNOf y.table st - Zk.Q))) ⊕ Fin 2) = _
    simp only [Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  rw [hc]
  calc _ ≤ 2 ^ 23 + (2 * 2 ^ 13 + 2) := Nat.add_le_add hT (Nat.add_le_add_right (Nat.mul_le_mul_left 2 hP) 2)
    _ = leafCap := by norm_num [leafCap]

end FlockSoundness.Discharge.ZkHidden

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.Discharge

/-- **The deployed masked session's leaves at `x` are at most `leafCap` a table.** -/
theorem nL_le (P : Composed.LigProver) (x : Msg) : nL P x ≤ jOf x * ZkHidden.leafCap := by
  unfold nL
  rw [Fintype.card_sigma]
  refine (Finset.sum_le_sum fun j _ => (zkTab x j).card_leafPos_le P).trans ?_
  simp

/-- **The masked session has at most `Tags.MAX_TABLES` tables**: a statement of the masked scope that sets up sets up
at most that many (`ZkTables.tables_of_setup`), and `jOf` is `0` elsewhere. -/
theorem jOf_le_max (x : Msg) : jOf x ≤ _root_.Flock.Tags.MAX_TABLES := by
  unfold jOf
  split_ifs with h
  · rcases Nat.eq_zero_or_pos (stsOf x).size with h0 | h0
    · rw [h0]
      exact Nat.zero_le _
    · obtain ⟨s, spec, hset⟩ := setupOf_of_pos h0
      exact (FlockComplete.ZkTables.tables_of_setup h.1 hset).2.1
  · exact Nat.zero_le _

end Claims.ZKMal.Masked

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal
open FlockSoundness.Discharge

/-- **The session's leaf count at the deployed tables**, whatever the round and tape plans: at most `leafCap` a
table. -/
theorem nLeaves_tableData_le (nr : Msg → ℕ → ℕ) (P : Composed.LigProver) (Q : Masked.RoundPlan (Masked.tabC P))
    (U : Masked.TapePlan (Masked.tabC P)) (x : Msg) :
    (CoinLayer.session nr (Masked.tableData P Q U)).nLeaves x ≤ Masked.jOf x * ZkHidden.leafCap :=
  Masked.nL_le P x

/-- **The deployed session's leaf count is at most `leafCap` a table**: the cap `Loop.anyErr_le` takes, at
`L = jOf x * leafCap`. -/
theorem nLeaves_dataE_le (x : Msg) :
    (CoinLayer.session loopNr Masked.dataE).nLeaves x ≤ Masked.jOf x * ZkHidden.leafCap :=
  nLeaves_tableData_le loopNr _ _ _ x

/-- **The deployed session's leaf count is at most `Tags.MAX_TABLES * leafCap`** at every statement: the cap
`Loop.anyErr_le` takes, at a constant `L`. -/
theorem nLeaves_dataE_le_max (x : Msg) :
    (CoinLayer.session loopNr Masked.dataE).nLeaves x ≤ _root_.Flock.Tags.MAX_TABLES * ZkHidden.leafCap :=
  (nLeaves_dataE_le x).trans (Nat.mul_le_mul_right _ (Masked.jOf_le_max x))

/-- **`anyErr` at the deployed tables is at most `errAt` at `L = Tags.MAX_TABLES * leafCap`**: `anyErr_le` with no
leaf count left free. -/
theorem anyErr_le_dataE (vs : (CoinLayer.session loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (t : ℕ) {x : Msg} {w : Flock.Wit}
    {Θ : Type} {Vs : VStar Θ} {θ : Θ} {q : ℕ} {δ₁ : ℝ} (hδ : hidingGap ZkView.My ZkView.saltHash ≤ δ₁)
    (hB : Budget Masked.dataE vs L₀ real x w Vs θ q) (o : Bool) :
    anyErr Masked.dataE vs L₀ real t x w Θ Vs θ o ≤
      errAt t (_root_.Flock.Tags.MAX_TABLES * ZkHidden.leafCap) q δ₁ o :=
  anyErr_le Masked.dataE vs L₀ real t (nLeaves_dataE_le_max x) hδ hB o

end Claims.ZKServer.Loop
