import Proofs.Flock.Claims.ZK.Deployed
import Proofs.Flock.Claims.ZK.Worker

/-!
# The switched prover at the session's tables

`Worker` proves Lemma B at a worker's clear values for one table. Here it is at the `J` tables of a `--zk` session, in
the form the session's models read (`Deployed`):

* `Coined.swExe`: a `--zk` table's switched view at a worker's values `v`, with the executable's hidden leaves
  (`Table.Hm96.swViewR`).
* `zkViewAtJ_sw`: at `J` `CoinedSetup`s, at coins each admits and valid witnesses, for any values the workers supply, the
  joint switched view and the tables' `S_shvzk` on any dummy are within `Σ_j 2·|Hid_j|·δ₁` both ways, as
  `ZkHidden.zkViewAtJ_of_ok` has it for the honest prover.
* `Proto.TabModel.SwLink`, `Proto.TabModel.shvzk_sw`: where the prover goes on, a prover whose messages are `lift` of the
  tables' switched views at the workers' values meets `SHVZK` against `simOut`, the honest prover's simulator, within
  `2·n·δ₁`, as `TabModel.shvzk` has it at `TabModel.Link`.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (CoinedSetup CoinsZK)
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A `--zk` table's switched view** at a worker's clear values `v`, with the executable's hidden leaves: `v`'s view
where its rows hold, else the honest prover's on the moved pads (`Table.Hm96.swViewR`). -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swExe {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (ω : T.tab.Rand)
    (ys : T.tab.Hid → ZkView.Salt) : T.tab.View :=
  T.tab.toBase (T.hm96.onVals.swViewR (w, v) (w, T.tab.vals) ω ys)

set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
/-- **Lemma B at the workers' values, at `J` `CoinedSetup`s**: at any coin vector of each table that the honest prover
doesn't refuse, valid witnesses `w j` and any clear values `v j` the workers supply, the joint switched view and the
simulators' at any `w0 j`, each table's randomness independent, are within `Σ_j 2·|Hid_j|·δ₁` both ways
(`Table.worker_shvzk_hm96` at each table, `prCoin_pi_close`). -/
theorem zkViewAtJ_sw {J : ℕ} (C : Fin J → CoinedSetup)
    (hC : ∀ j, Composed.CoinedOK FlockSoundness.execArith (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) (C j))
    (hreg : ∀ j c w, (C j).Valid w → ((C j).T c).tab.sReg w = ((C j).T c).tab.pubReg)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (c : ∀ j, CoinsZK (C j).Rest)
    (hc : ∀ j, (C j).Admits (c j)) {Out : Type} (enc : (∀ j, ((C j).T (c j)).tab.View) → Out)
    (w w0 : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (hv : ∀ j, (C j).Valid (w j))
    (v : ∀ j, ((C j).T (c j)).tab.Vals) (E : Out → Prop) :
    FlockSoundness.Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).swExe (w j) (v j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ FlockSoundness.Game.prCoin
          (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E (enc fun j => ((C j).T (c j)).swExe (w j) (v j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
  have hreg := fun j => hreg j (c j) (w j) (hv j)
  have one : ∀ j (E' : ((C j).T (c j)).tab.View → Prop),
      FlockSoundness.Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E' (((C j).T (c j)).swExe (w j) (v j) p.1 p.2))
        ≤ FlockSoundness.Game.prCoin
            (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E' (((C j).T (c j)).simExe (w0 j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ FlockSoundness.Game.prCoin
            (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
              E' (((C j).T (c j)).simExe (w0 j) p.1 p.2))
        ≤ FlockSoundness.Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E' (((C j).T (c j)).swExe (w j) (v j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
    intro j E'
    obtain ⟨hL, ht, hS, hT⟩ := hC j
    obtain ⟨⟨hpos, hlt⟩, -, hrank, hGp, hW, hpads, hβ, hinner⟩ := hT (c j) (hc j)
    have hpad : ((C j).T (c j)).tab.PadOnto :=
      FlockSoundness.ZK.padOnto_M1 ((C j).T (c j)).tab FlockSoundness.execArith FlockSoundness.execArith_correct (C j).cols hL ht
        hS (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) ((C j).pos (c j)) hpos hlt hGp (FlockSoundness.ZK.padNonvanishing40 _ hL)
    simp only [ZkView.Coined.simExe_eq]
    exact ((C j).T (c j)).tab.worker_shvzk_hm96 ((C j).T (c j)).hm96 hT1 hrank hpad hW hpads hβ (w j) (hreg j)
      (hinner (w j) (hv j) (hreg j)) (v j) (w0 j) E'
  have : Nonempty ZkView.Salt := ⟨0⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((ZkHidden.rand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((ZkHidden.simRand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  exact ⟨FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).swExe (w j) (v j) p.1 p.2) (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').1) fun v => E (enc v),
    FlockSoundness.ZK.prCoin_pi_close
      (Ω := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (Ω' := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt))
      (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2) (fun j p => ((C j).T (c j)).swExe (w j) (v j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').2) fun v => E (enc v)⟩

namespace Proto

/-- **Where the prover goes on at `c`, its messages are `lift` of the tables' switched views at the workers' values
`vs`** (`Coined.swExe`), at valid table witnesses, on randomness read through a map with fibers of one size. -/
def TabModel.SwLink {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (vs : ∀ j, ((M.C j).T (M.cz j)).tab.Vals) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧ (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    ∀ r, G.realVec H x w ν r c =
      M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs j) (M.eR r j).1 (M.eR r j).2

/-- **`SHVZK` of the switched prover at a coin vector where it goes on** (`zkViewAtJ_sw`, within `2·n·δ₁`): its
messages at `c` and `simOut`, the honest prover's simulator, on any tape read through a map with fibers of one size,
whatever values the workers supply. -/
theorem TabModel.shvzk_sw {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (vs : ∀ j, ((M.C j).T (M.cz j)).tab.Vals) (hL : M.SwLink H ν c w vs) (hOK : M.OK) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (eS : Ω → M.SimAt) {kS : ℕ} (heS : ∀ p, Nat.card {s // eS s = p} = kS) {n : ℕ}
    (hn : M.size ≤ n) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (M.simOut (eS s)))| ≤ 2 * n * δ₁.toReal := by
  obtain ⟨hv, heR, hR⟩ := hL
  obtain ⟨hset, hA⟩ := hOK
  have hb := zkViewAtJ_sw M.C (fun j => (hset j).ok.1) (fun j => (hset j).ok.2) hT1 M.cz hA M.lift (M.wt w) M.wt₀ hv
    vs E
  have : Nonempty M.SimAt := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, ((M.C j).T (M.cz j)).tab.Rand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)) :=
    ⟨M.eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, ((M.C j).T (M.cz j)).tab.Rand ×
          (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt) =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).swExe (M.wt w j) (vs j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber M.eR heR]
    simp only [hR]
  have eSim : Pr (fun s : Ω => E (M.simOut (eS s))) =
      (FlockSoundness.Game.prCoin fun p : M.SimAt =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).simExe (M.wt₀ j) (p j).1 (p j).2)).toReal := by
    simp only [TabModel.simOut]
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eS heS]
  have hle : ∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁ ≤ 2 * n * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hn
  have htop : (2 * n * δ₁ : ENNReal) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁) ≠ ⊤ :=
    ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁).toReal ≤ 2 * n * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.2
  rw [eReal, eSim, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
