import FlockSoundness.Discharge.ZkEveryCoin.Defs
import FlockSoundness.Discharge.Composed.Headline

/-!
# One `--zk` session is zero knowledge at every coin vector

`Composed.zk_session_view` holds at the covered coin vectors only. At every coin vector, with the verifier's coins uniform:

* where the prover refuses (`¬ Admits`, `refused_coins`), it stops, and what it sent is a prefix of its view
  (`Table.abort`). The refusal reads only the coins, which are public, so the simulator aborts at exactly those coins and
  outputs the honest prefix on the dummy witness (`simAll`). Lemma A at every coin vector (`table_prefinal_indep_pub`) and
  T1 on every pre-final leaf bound the difference by `2·|LeafPos|·δ₁` (`Table.table_abort_hm96`), with no hypothesis on
  the coins: every leaf is unopened at a refusal, so T1 is paid on all of them, and Lemma B's re-randomization of the
  opened rows (`PadOnto`, which `CoinedOK` gives only at covered coins) isn't needed;
* where it doesn't refuse and the coins are covered, Lemma B (`zkViewAt_of_ok`), within `2·|Hid|·δ₁ ≤ 2·|LeafPos|·δ₁`;
* where it doesn't refuse and a zerocheck outer coin is `1` (`Admits ∧ ¬Covered`), nothing: the view there is not
  simulated, and the statement charges the probability of those coins, at most `(m − 13)·2^-128` per rep
  (`uncovered_le`).

* `zkViewAll_of_ok`: `ZkViewAll` at a `CoinedSetup` from `CoinedOK`, `hash-derived-key` and `PadNonvanishing`.
* `uncovered_le`: at a draw's tables, `Pr_c[Admits c ∧ ¬Covered c] ≤ 2·(m − 13)·2^-128`.
* `zk_session_view_all`: at every draw `verify --zk` sets up, `ZkViewAll` at `2^-193` with that bound.
-/

namespace FlockSoundness.Discharge.ZkEveryCoin

open Game Audit Exec Refine Flock FlockLevel3 Model ZkSession ZkLink ZkExec DecodeZL Composed ZkView
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

local notation "𝔽" => FlockLevel3.GF128

/-- **Zero knowledge at every coin vector at a `CoinedSetup` from its facts**: Lemma B at the covered coins
(`zkViewAt_of_ok`), the refusal at the refused ones (`Table.table_abort_hm96`, which reads nothing of `CoinedOK`), and
the coins it doesn't refuse with an outer coin `1` charged. -/
theorem zkViewAll_of_ok (C : CoinedSetup) [Fintype (CoinsZK C.Rest)]
    (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (hPad : Assumptions.Zk.PadNonvanishing40) :
    ZkViewAll C ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  sorry

/-- **The coins zero knowledge doesn't cover, but the prover doesn't refuse**, at a draw's tables: a zerocheck outer coin
`1` in some rep, each of the `m − 13` outer coins of each of the two reps uniform over F128. -/
theorem uncovered_le {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    prCoin (fun c : CoinsZK (coinedOf Tz st).Rest => (coinedOf Tz st).Admits c ∧ ¬ (coinedOf Tz st).Covered c)
      ≤ 2 * (st.m - 13 : ℕ) * (2 : ℝ≥0∞) ^ (-128 : ℤ) := by
  sorry

variable {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session is zero knowledge at every coin vector.** At every draw `verify --zk` sets up, with the
verifier's coins uniform: for every event on the coins and what the honest prover sends (its view, or the prefix of it
before it stops at a refused coin), at any valid witness whose registered values are the public ones, its probability
and its probability under the simulator (`S_shvzk`, or the honest prefix on the dummy witness where the prover refuses,
at any `w0`) are within `2·|LeafPos|·2^-193 + 2·(m − 13)·2^-128`, both ways. The first term is T1's on every leaf; the
second is the coins with a zerocheck outer coin `1` (`CoinedSetup.Covered`), the only ones where the view is not
simulated. -/
theorem zk_session_view_all (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (hPad : Assumptions.Zk.PadNonvanishing40) :
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAll (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
      prCoin (fun c : CoinsZK (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Rest =>
          (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Admits c ∧
            ¬ (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Covered c)
        ≤ 2 * (y.x.st.m - 13 : ℕ) * (2 : ℝ≥0∞) ^ (-128 : ℤ) := by
  sorry

end FlockSoundness.Discharge.ZkEveryCoin
