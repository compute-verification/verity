import Proofs.Flock.Claims.Recursion.NonVacuousRows
import Proofs.Flock.Claims.Recursion.RowDLevels

/-!
# `TargetDMerkle` from the honest rep's equations

`TargetDMerkle` (`NonVacuousRows`, the `DecompB` assembly's row D leaf) asks, at every decision and rep, for query coins
`cf` from a round `k` at which the rep's decided proof passes `MerkleOk` at the standard hashes, and from which every
run of its verifier rows draws its positions. `targetDMerkle_of_repP` gives it at `cf` the rep's recorded rounds
(`cfOf`) and `k` Ligerito's first round (`lgStart`), from the honest rep's three equations alone (HONESTIV's):

* `h0`: the level-0 commitment is a `commitP` of the rep's table,
* `hrun`: the decided proof is the output of the honest rep `repP`, run from a transcript at round `0`,
* `hcoin`: the replies' coins at the queried rounds are the rep's recorded ones.

`MerkleOk` is `merkleOk_of_repP` at `Sha512.hash`, with the plain scheme's 64-byte widths (`Rows.plain_widths_std`,
row G's `merkle`), carried to the recorded coins by `merkleOk_congr`; the positions are `DecAt.htie`.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)
open FlockVBridge.Family (cfOf lgStart)
open FlockComplete (repP asksS commitP Commit Trace)
open FlockComplete.Replay.Lg (qOff)

namespace Top

/-- **`TargetDMerkle` from the honest rep's equations**: wherever each decided rep's proof is a returning honest rep's
(`hrun`) on a `commitP` of its table (`h0`) at the deployed hashes, and its replies' coins at the queried rounds are its
recorded ones (`hcoin`). -/
theorem targetDMerkle_of_repP {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}
    {acc : Msg → List Msg → List Coin → ℕ → Array F128}
    (hon : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass)
      (i : ℕ), i < D.sts.size → ∀ r : Fin 2,
      ∃ (resp : ℕ → Msg) (tr : Trace) (tbl : Array F128) (cm0 : Commit) (salt : ℕ → ℕ → ByteArray)
        (nonce : ℕ → ℕ → UInt64) (s0 : ℕ → ByteArray) (t : Transcript) (l : Array Msg) (t' : Transcript)
        (l' : Array Msg),
        commitP (Hashes.ofSha512 Sha512.hash : Hashes Flock.V) (D.files cls i).st.merkle
            ((D.files cls i).sch.levels.getD 0 default) tbl s0 = .ok cm0 ∧
          t.k = 0 ∧
          ((repP (asksS resp) (Hashes.ofSha512 Sha512.hash) (D.files cls i).st (D.files cls i).sch tr tbl cm0
            (D.files cls i).extras salt nonce).run t).go l = (.ok ((D.files cls i).proof r, t'), l') ∧
          ∀ j < (D.files cls i).sch.levels.size, ∀ u < ((D.files cls i).sch.levels.getD j default).q,
            FlockComplete.Replay.Lg.cfOf resp l.size
                (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) u =
              cfOf (FlockVBridge.Family.roundsOf (D.files cls i) r)
                (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) u) :
    TargetDMerkle rd P acc := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hG := D.rowG cls (acc x ms cs) hst hf hx hi
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  obtain ⟨resp, tr, tbl, cm0, salt, nonce, s0, t, l, t', l', h0, ht, hrun, hcoin⟩ := hon x ms cs hA D cls i hi r
  obtain ⟨h22, -⟩ := FlockVBridge.Family.fast100_range hG.fast
  have hM := FlockComplete.Replay.merkleOk_of_repP resp Sha512.hash _ _ tr tbl cm0 _ salt nonce hG.fast (by omega)
    (by rw [hG.merkle]; exact Rows.plain_widths_std.1) (by rw [hG.merkle]; exact Rows.plain_widths_std.2) h0 t ht l
    hrun
  have hN : 3 + ((D.files cls i).st.m - 6) + 3 + ((D.files cls i).st.fold.kLog - 6) +
      (2 + (D.files cls i).extras.size) + 1 = lgStart (D.sh cls i) (D.files cls i).extras.size := by
    unfold lgStart
    rw [hG.foldKLog]
    rfl
  rw [hN, hG.merkle] at hM
  exact ⟨cfOf (FlockVBridge.Family.roundsOf (D.files cls i) r), lgStart (D.sh cls i) (D.files cls i).extras.size,
    FlockComplete.Replay.Lg.merkleOk_congr hM hcoin, D.htie cls (acc x ms cs) hG hti r⟩

end Top

end RecDeploy
