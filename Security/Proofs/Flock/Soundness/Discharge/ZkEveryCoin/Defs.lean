import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.ZK.Abort

/-!
# One `--zk` session at every coin vector: what the prover sends and what the simulator outputs

`Composed.ZkViewAt` is the view statement at the coin vectors the honest prover doesn't refuse (`CoinedSetup.Admits`).
Here the statement at every coin vector, with the verifier's coins uniform and drawn independently of the prover's
randomness:

* `sentAt`: the guarded messages a table sent before the session's first failing check (`Session.sent`, one table).
* `outExe`: **the honest prover's output** at a coin vector: its view (`viewExe`) where it doesn't refuse
  (`CoinedSetup.Admits`, `refused_coins`), and where it does, the prefix of that view it sends before it stops
  (`Table.abort`: `at_open` returns `Err` before any `ŝ` leaves, and so on for `ρ` and `β`; at a zerocheck outer coin `1`
  it stops earlier still, and what it sent is a function of that cut).
* `simAll`: **the simulator**: `S_shvzk` (`simExe`) where the prover doesn't refuse, and the honest prefix on the dummy
  witness with the public region values (`Table.pubAbort`) where it does. Which branch is taken is a function of the
  coins, which are public, so the simulator aborts exactly when the prover does.
* `ZkViewAll`: every event on the coins and the output has, averaged over uniform coins, probabilities within
  `2·|LeafPos|·δ₁` under the prover and the simulator, both ways.
* `instFintypeF128`, `instFintypeF256`, `instFintypeRestZK`, `instFintypeCoinsZK` (and `instFintypeRestCoinedOf` at a draw's tables): the uniform distribution on a
  draw's coins.
-/

namespace FlockSoundness.Discharge.ZkEveryCoin

open Game Composed ZkView Model Flock
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The table's guarded messages sent before the session's first failing check, at coin vector `c`. -/
noncomputable def sentAt (C : CoinedSetup) (c : CoinsZK C.Rest) : (C.T c).tab.Sent :=
  ZK.Session.sent (fun _ : Fin 1 => (C.T c).tab) (fun _ => (C.T c).rep) 0

open Classical in
/-- **The honest prover's output at coin vector `c`**, on witness `w`, randomness `ω` and hidden salts `ys`: its view where
it doesn't refuse `c`, and otherwise the prefix of that view it sends before it stops. -/
noncomputable def outExe (C : CoinedSetup) (c : CoinsZK C.Rest) (w : Witness C.S.m) (ω : (C.T c).tab.Rand)
    (ys : (C.T c).tab.Hid → ZkView.Salt) : (C.T c).tab.View ⊕ (C.T c).tab.Cut :=
  if C.Admits c then .inl ((C.T c).viewExe w ω ys) else .inr ((C.T c).tab.abort (sentAt C c) ((C.T c).viewExe w ω ys))

open Classical in
/-- **The simulator's output at coin vector `c`**, on the dummy witness `w0`, randomness `σ` and hidden salts `ys`:
`S_shvzk` where the prover doesn't refuse `c`, and otherwise the honest prefix on `w0`, at `S_shvzk`'s dummy randomness,
with the public region values. -/
noncomputable def simAll (C : CoinedSetup) (c : CoinsZK C.Rest) (w0 : Witness C.S.m) (σ : (C.T c).tab.SimRand)
    (ys : (C.T c).tab.Hid → ZkView.Salt) : (C.T c).tab.View ⊕ (C.T c).tab.Cut :=
  if C.Admits c then .inl ((C.T c).simExe w0 σ ys)
  else .inr ((C.T c).tab.pubAbort (sentAt C c) ((C.T c).viewExe w0 ((C.T c).tab.dummyOf σ) ys))

/-- **Zero knowledge at a `CoinedSetup`, at every coin vector, at `δ₁` per leaf**: for any valid witness `w` whose
registered values are the public ones (at every coin vector's link points), any dummy witness `w0` and any event `E` on
the coins and the output, the probability under uniform coins `c` and the honest prover's randomness (`outExe`) and
under uniform coins and the simulator's (`simAll`) are within `2·|LeafPos|·δ₁`, both ways.
The coins are drawn independently of either party's randomness, so each probability is the average over `c` of the
probability at `c`. -/
def ZkViewAll (C : CoinedSetup) [Fintype (CoinsZK C.Rest)] (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (w w0 : Witness C.S.m), C.Valid w → (∀ c, (C.T c).tab.sReg w = (C.T c).tab.pubReg) →
    ∀ E : ∀ c : CoinsZK C.Rest, (C.T c).tab.View ⊕ (C.T c).tab.Cut → Prop,
      avg (fun c => prCoin fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
          ≤ avg (fun c => prCoin fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
              E c (simAll C c w0 p.1 p.2))
            + 2 * Fintype.card (LeafPos C.z) * δ₁
        ∧ avg (fun c => prCoin fun p : (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
              E c (simAll C c w0 p.1 p.2))
          ≤ avg (fun c => prCoin fun p : (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) => E c (outExe C c w p.1 p.2))
            + 2 * Fintype.card (LeafPos C.z) * δ₁

/-! ## The uniform distribution on a draw's coins -/

/-- F128 as its two 64-bit words. -/
def f128Equiv : Flock.F128 ≃ Fin (2 ^ 64) × Fin (2 ^ 64) :=
  ⟨fun a => (a.lo.toFin, a.hi.toFin), fun p => ⟨UInt64.ofFin p.1, UInt64.ofFin p.2⟩, fun _ => rfl, fun _ => rfl⟩

instance instFintypeF128 : Fintype Flock.F128 := Fintype.ofEquiv _ f128Equiv.symm

/-- F256 as its two F128 coordinates. -/
def f256Equiv : Flock.F256 ≃ Flock.F128 × Flock.F128 := ⟨fun a => (a.c0, a.c1), fun p => ⟨p.1, p.2⟩, fun _ => rfl, fun _ => rfl⟩

instance instFintypeF256 : Fintype Flock.F256 := Fintype.ofEquiv _ f256Equiv.symm

variable {q0 L0 k0 m kLog nCl nOod mP R : ℕ} {qs : ℕ → ℕ} in
/-- `RestZK` as the product of its fields. -/
def restEquiv : RestZK q0 L0 k0 m kLog nCl nOod mP R qs ≃
    ((Fin 2 → Fin mP → F128) ×
      (Fin 2 → Fin q0 → Fin L0) ×
      (Fin 2 → Fin 6 → F128) ×
      (Fin 2 → Fin (m - 13) → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → Fin (m - 6) → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → Fin (kLog - 6) → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → Fin nCl → Fin 7 → F128) ×
      (Fin 2 → Fin nCl → F128) ×
      (Fin 2 → Fin nOod → Fin (m - 7) → F128) ×
      (Fin 2 → Fin nOod → F128) ×
      (Fin 2 → Fin k0 → F256) ×
      (Fin 2 → (j : Fin R) → Fin 2 → Fin (m - 6 - k0 - 3 * j.val) → F128) ×
      (Fin 2 → Fin R → Fin 2 → F128) ×
      (Fin 2 → Fin (Nat.log2 (qs 0 - 1) + 1) → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → Fin R → Fin 4 → F256) ×
      (Fin 2 → (i : Fin R) → Fin (qs (i.val + 1)) → F128) ×
      (Fin 2 → (i : Fin R) → Fin (Nat.log2 (qs (i.val + 1) - 1) + 1) → F128) ×
      (Fin 2 → Fin R → F128) ×
      (Fin 2 → F128) ×
      (Fin 2 → Fin 10 → F128) ×
      (Fin 2 → Fin Zk.Q → F128)) where
  toFun r := (r.link, r.q0s, r.rSkip, r.rOuter, r.zc, r.rho, r.lcAlpha, r.lcBeta, r.ts, r.rPrime, r.rpp, r.rsGamma, r.oodPt, r.ood, r.lane, r.ligZ, r.ligZB, r.ligAlpha0, r.ligB0, r.ligFold, r.ligQ, r.ligAlpha, r.ligB, r.zeta, r.gammas, r.qP)
  invFun p := ⟨p.1, p.2.1, p.2.2.1, p.2.2.2.1, p.2.2.2.2.1, p.2.2.2.2.2.1, p.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

instance instFintypeRestZK {q0 L0 k0 m kLog nCl nOod mP R : ℕ} {qs : ℕ → ℕ} :
    Fintype (RestZK q0 L0 k0 m kLog nCl nOod mP R qs) :=
  Fintype.ofEquiv _ restEquiv.symm

/-- `CoinsZK` as the product of its fields. -/
def coinsEquiv (Rest : Type) : CoinsZK Rest ≃ (Fin 2 → F128) × (Fin 2 → F256 × F256) × Rest where
  toFun c := (c.beta, c.rho, c.rest)
  invFun p := ⟨p.1, p.2.1, p.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

instance instFintypeCoinsZK (Rest : Type) [Fintype Rest] : Fintype (CoinsZK Rest) :=
  Fintype.ofEquiv _ (coinsEquiv Rest).symm

instance instFintypeRestCoinedOf (P : LigProver) {mPts : ℕ} (Tz : ZkSession.TabZK FlockLevel3.GF128 mPts) (st : Stmt) :
    Fintype (coinedOf P Tz st).Rest :=
  inferInstanceAs (Fintype (RestOf Tz st))

end FlockSoundness.Discharge.ZkEveryCoin
