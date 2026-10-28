import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server
import Proofs.Flock.Claims.ZK.Capacity

/-!
# Zero knowledge at the reference session, at any values pad-blind workers supply

`Switched` has Lemmas A, B and C at the switched prover: the honest prover at the workers' values where their rows hold,
else the honest prover on moved pads. Here that prover is defined at the reference session, it meets `Claims.Flock.ZKAt`
with the honest prover's simulator, and the server that runs pad-blind workers is bounded by it:

* `RefSession.SwVals`: the values the workers supply, each table's at each statement, witness, nonce and coin vector:
  what a pad-blind worker (`ZkServer.BWorker`) sees, never the server's tape. `RefSession.swP vs`: `refP` with its
  messages the switched prover's at `vs` (`swMsgAt`), its dummy, coins, tape and opening check `refP`'s.
* `viewPre_swExe`: the switched prover's real views' pre-final part is its cut view at the executable's leaves, as
  `viewPre_viewExe` has it at the honest prover, so `swP`'s last message and its round messages agree.
* `Proto.simulatorE_msg`: `S_GK` never reads the prover's messages, so `RefSession.swPS vs` (`refP` with the models'
  simulator and `swP vs`'s messages) has `swP vs`'s prover and `refP`'s simulator.
* `RefSession.swPS_shvzk`: its `SHVZK` against `refP`'s simulator, from `TCoinModel.shvzk_sw` at `refP`'s models
  typed at `swP` (`toSw`, `swCoinModel_eq`); Lemma C before the last coin is `swPreModel`'s. Both rest on `Complete`,
  `DummyReg`, `Causal`, `TapeOK`, `Tables` and `SwCausal vs`, the switched round encoding's causality.
  **`Flock.zk_deployZk_sw`**: `Claims.Flock.ZKAt` at `swP vs`'s prover with `Flock.zkAt_deployZk_ref`'s simulator, `ε_zk`
  and `γ`: one simulator for every `vs`.
* `RefSession.honestVals`, each table's own values: `swP_vals` (`swP` there is `refP`) and `swCausal_vals` (`Causal`
  gives `SwCausal` there), so `Flock.zkAt_deployZk_ref_of_sw` recovers `Flock.zkAt_deployZk_ref` from `zk_deployZk_sw`.
* **`Flock.zk_server_blind`**: for every pad-blind worker and any values `vs` whose round encoding is causal, the server
  that defers every stop to the last round (`ZkServer.nextD`), whose frame at the worker's values with its stand-ins in
  after a failed round is `swP vs`'s message wherever it releases one (`ZkServer.RefD`), releases, on every event, at
  most twice the probability at the simulator that guesses whether the session stops (`leaky1`) plus `ε_zk + γ·Fork`,
  with `Flock.zk_server_ref`'s simulator and no term for the masked checks: one bit a session.
* `Flock.zk_deployZk_sw_on`, `Flock.zk_server_blind_on`: the same at a statement of a class `A`, with `TapeOK` asked for
  on `A` only (`RefSession.TapeOKOn`), at the record whose parser refuses the rest (`deployOn`); both and
  `Flock.zk_deployZk_sw` are `Flock.zkAt_sw_of` at a record. `Flock.zk_server_blind_deployed`: `zk_server_blind_on` at
  the deployed coin layer (`CoinLayer.session`) and the pinned key's hiding gap, under the tables' tape read with fibers
  of one size on `A`.
-/

namespace FlockSoundness.ZK.Table

set_option smartUnfolding false in
/-- At the honest values, the view with its hidden leaves given is the table's. -/
theorem toBase_viewWith {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
    (T : Table F K W Salt (B × Salt) D Hi VL) (w : W) (ω : T.Rand) (L : T.Hid → D) :
    T.toBase (T.onVals.viewWith (w, T.vals) ω L) = T.viewWith w ω L := by
  have h := T.toBase_view w ω
  simp only [viewWith]
  rw [← h]
  rfl

set_option smartUnfolding false in
/-- At the honest values, the view with real leaves is the table's. -/
theorem Hm96.toBase_viewR {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
    {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    T.toBase (M.onVals.viewR (w, T.vals) ω ys) = M.viewR w ω ys := by
  have hd : M.onVals.dig (w, T.vals) ω = M.dig w ω := by
    funext i
    rcases i with q | ⟨r, k⟩ <;> rfl
  unfold Hm96.viewR
  rw [hd]
  exact T.toBase_viewWith w ω _

/-- `shift` from a witness to itself moves nothing. -/
theorem shift_self {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) (p : W)
    (ω : T.Rand) : T.shift p p ω = ω := by
  have h : T.padShift p p ω.1 ω.2.1.2.1 = 0 := by
    funext r
    simp only [padShift, sub_self, mul_zero, neg_zero]
    rfl
  simp only [shift, h, add_zero]

/-- **At the honest values the switched pre-final view is the honest one.** -/
theorem swPre_vals {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : Table F K W Salt Seed D Hi VL) (w : W) (ω : T.Rand) : T.swPre w T.vals ω = T.preView w ω := by
  unfold swPre
  split_ifs
  · exact T.toBasePre_preView w ω
  · exact congrArg (T.preView w) (T.onVals.shift_self (w, T.vals) ω)

end FlockSoundness.ZK.Table

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## A table's switched view before the last coin -/

set_option maxRecDepth 100000 in
/-- The executable's view of a `--zk` table at any clear values is hm96's. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_viewExe {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (ω : T.tab.Rand)
    (ys : T.tab.Hid → ZkView.Salt) : T.onVals.viewExe (w, v) ω ys = T.hm96.onVals.viewR (w, v) ω ys := by
  refine (ZkView.Coined.viewExe_eq T.onVals (w, v) ω ys).trans ?_
  unfold FlockSoundness.ZK.Table.Hm96.viewR
  congr 1

/-- **The switched prover's pre-final leaf** at a position, under salt `y`: `hm96-sha512/v1`'s leaf of
`Coined.swContent`. -/
noncomputable def _root_.FlockSoundness.Discharge.ZkView.Coined.swExeLeaf {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : ZkView.LeafPos z) (y : ZkView.Salt) : ByteArray :=
  _root_.Flock.Hm96.Default512.leaf (T.swContent w v b p) (ZkView.saltBytes y)

set_option maxRecDepth 100000 in
/-- **The switched prover's real view's pre-final part is its pre-final view at the executable's leaves**: at the table
randomness whose revealed salts are the leaves' (`baseWith`) and whose hidden leaves' salts are theirs (`hidOf`),
whatever the ideal seeds `u` (`viewPre_viewExe` at `v` where the rows hold, at the moved base elsewhere). -/
theorem viewPre_swExe {K W Hi VL : Type} [Field K] {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : ZkView.Base z Hi) (y : ZkView.LeafPos z → ZkView.Salt)
    (u : ZkView.LeafPos z → ZkView.Bits × ZkView.Salt) :
    viewPre T (T.swExe w v (T.randOf (baseWith T b y) u) (hidOf T y)) =
      T.swPreWith w v b fun p => T.swExeLeaf w v b p (y p) := by
  by_cases hr : T.tab.onVals.RowsAt (w, v) b.1 b.2.1.2.1
  · have hr' : T.tab.onVals.RowsAt (w, v) (T.randOf (baseWith T b y) u).1
        (T.randOf (baseWith T b y) u).2.1.2.1 := hr
    have e1 : T.swExe w v (T.randOf (baseWith T b y) u) (hidOf T y) =
        T.tab.toBase (T.onVals.viewExe (w, v) (T.randOf (baseWith T b y) u) (hidOf T y)) :=
      (congrArg T.tab.toBase ((FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals)
        (T.randOf (baseWith T b y) u) (hidOf T y)).trans (ite_eq_left hr'))).trans
        (congrArg T.tab.toBase (T.onVals_viewExe w v _ _).symm)
    have h0 : T.tab.swPre w v (T.randOf b 0) =
        T.tab.toBasePre (T.tab.onVals.preView (w, v) (T.randOf b 0)) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_left hr
    have hc : T.swContent w v b = T.onVals.content (w, v) b := by
      unfold ZkView.Coined.swContent
      exact ite_eq_left hr
    have e2 : T.swPreWith w v b (fun p => T.swExeLeaf w v b p (y p)) =
        T.tab.toBasePre (T.onVals.preWith (w, v) b fun p => T.onVals.exeLeaf (w, v) b p (y p)) := by
      rw [ZkView.Coined.swPreWith, h0]
      simp only [ZkView.Coined.swExeLeaf, hc]
      rfl
    rw [e1, e2, ← viewPre_viewExe T.onVals (w, v) b y u]
    rfl
  · have hr' : ¬ T.tab.onVals.RowsAt (w, v) (T.randOf (baseWith T b y) u).1
        (T.randOf (baseWith T b y) u).2.1.2.1 := hr
    have e1 : T.swExe w v (T.randOf (baseWith T b y) u) (hidOf T y) =
        T.viewExe w (T.randOf (baseWith T (T.shiftB w v b) y) u) (hidOf T y) :=
      (congrArg T.tab.toBase ((FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals)
        (T.randOf (baseWith T b y) u) (hidOf T y)).trans (ite_eq_right hr'))).trans
        ((T.hm96.toBase_viewR w (T.randOf (baseWith T (T.shiftB w v b) y) u) (hidOf T y)).trans
          (ZkView.Coined.viewExe_eq T w _ _).symm)
    have h0 : T.tab.swPre w v (T.randOf b 0) = T.tab.preView w (T.randOf (T.shiftB w v b) 0) := by
      unfold FlockSoundness.ZK.Table.swPre
      exact ite_eq_right hr
    have hc : T.swContent w v b = T.content w (T.shiftB w v b) := by
      unfold ZkView.Coined.swContent
      exact ite_eq_right hr
    have e2 : T.swPreWith w v b (fun p => T.swExeLeaf w v b p (y p)) =
        T.preWith w (T.shiftB w v b) fun p => T.exeLeaf w (T.shiftB w v b) p (y p) := by
      rw [ZkView.Coined.swPreWith, h0]
      simp only [ZkView.Coined.swExeLeaf, hc]
      rfl
    rw [e1, e2, viewPre_viewExe T w (T.shiftB w v b) y u]

/-! ## The switched prover at the honest values -/

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.shiftB_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) :
    T.shiftB w T.tab.vals b = b := by
  have h : T.tab.onVals.padShift (w, T.tab.vals) (w, T.tab.vals) b.1 b.2.1.2.1 = 0 := by
    funext r
    simp only [FlockSoundness.ZK.Table.padShift, sub_self, mul_zero, neg_zero]
    rfl
  unfold ZkView.Coined.shiftB
  simp only [h]
  exact congrArg (fun s => (b.1, b.2.1, s, b.2.2.2)) (add_zero b.2.2.1)

set_option smartUnfolding false in
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_col0At_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) (p : Fin (z.nS + z.nU)) :
    T.onVals.col0At (w, T.tab.vals) b p = T.col0At w b p :=
  rfl

set_option smartUnfolding false in
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_colPAt_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) (r : Fin 2)
    (p : Fin (z.nQ + z.nPU)) : T.onVals.colPAt (w, T.tab.vals) b r p = T.colPAt w b r p :=
  rfl

set_option smartUnfolding false in
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_tau_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) (r : Fin 2) :
    (T.onVals.tab.view (w, T.tab.vals) (T.onVals.randOf b 0)).tau r = (T.tab.view w (T.randOf b 0)).tau r :=
  rfl

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.onVals_content_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) :
    T.onVals.content (w, T.tab.vals) b = T.content w b := by
  funext p
  rcases p with p | ⟨r, p⟩ | r <;>
    simp only [ZkView.Coined.content, ZkView.Coined.onVals_col0At_vals, ZkView.Coined.onVals_colPAt_vals,
      ZkView.Coined.onVals_tau_vals]

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi) :
    T.swContent w T.tab.vals b = T.content w b := by
  unfold ZkView.Coined.swContent
  split_ifs
  · exact T.onVals_content_vals w b
  · rw [T.shiftB_vals]

theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swPreWith_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (b : ZkView.Base z Hi)
    (lv : ZkView.LeafPos z → ByteArray) : T.swPreWith w T.tab.vals b lv = T.preWith w b lv := by
  unfold ZkView.Coined.swPreWith ZkView.Coined.preWith
  rw [T.tab.swPre_vals]

/-- **At the honest values the switched view is the honest one.** -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swExe_vals {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (ω : T.tab.Rand) (ys : T.tab.Hid → ZkView.Salt) :
    T.swExe w T.tab.vals ω ys = T.viewExe w ω ys := by
  have e : T.hm96.onVals.swViewR (w, T.tab.vals) (w, T.tab.vals) ω ys = T.hm96.onVals.viewR (w, T.tab.vals) ω ys := by
    refine (FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, T.tab.vals) (w, T.tab.vals) ω ys).trans ?_
    split_ifs
    · rfl
    · exact congrArg (fun ω' => T.hm96.onVals.viewR (w, T.tab.vals) ω' ys) (T.tab.onVals.shift_self _ ω)
  unfold ZkView.Coined.swExe
  rw [e]
  exact (T.hm96.toBase_viewR w ω ys).trans (ZkView.Coined.viewExe_eq T w ω ys).symm

theorem swCutAt_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) :
    swCutAt C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω lv = cutAt C cz wt ω lv := by
  funext j
  simp only [swCutAt, cutAt, ZkView.Coined.swPreWith_vals]

theorem swDigOf_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) :
    swDigOf C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω = digOf C cz wt ω := by
  funext l
  simp only [swDigOf, digOf, ZkView.Coined.swContent_vals]

theorem swExeOf_vals {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (ω : ∀ j, ZkView.Base (C j).z (C j).Hi)
    (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) :
    swExeOf C cz wt (fun j => ((C j).T (cz j)).tab.vals) ω y = exeOf C cz wt ω y := by
  funext l
  simp only [swExeOf, exeOf, ZkView.Coined.swContent_vals]
  rfl

/-! ## The simulator never reads the prover's messages -/

theorem Proto.estSP_msg {W : Type} (G : Proto W) (m : Msg → W → G.Nonce → G.Rand → List G.Coin → OC Msg) (x : Msg)
    (ν : G.Nonce) : ∀ b k (rs : Fin b → G.Rand),
    ({ G with msg := m } : Proto W).estSP x ν b k rs = G.estSP x ν b k rs
  | b, 0, rs => by cases b <;> rfl
  | 0, _ + 1, _ => rfl
  | b + 1, k + 1, rs => by
      simp only [Proto.estSP]
      congr 1
      funext p
      rw [Proto.estSP_msg G m x ν b]

theorem Proto.rewSP_msg {W : Type} (G : Proto W) (m : Msg → W → G.Nonce → G.Rand → List G.Coin → OC Msg) (x : Msg)
    (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ₀ : List (Msg × Out)) : ∀ n L (ss : Fin L → G.SimRand x),
    ({ G with msg := m } : Proto W).rewSP x ν c τ₀ n L ss = G.rewSP x ν c τ₀ n L ss
  | 0, _, _ => rfl
  | _ + 1, 0, _ => rfl
  | n + 1, L + 1, ss => by
      simp only [Proto.rewSP]
      congr 1
      funext p
      rw [Proto.rewSP_msg G m x ν c τ₀ n L]

theorem Proto.dRunSP_msg {W : Type} (G : Proto W) (m : Msg → W → G.Nonce → G.Rand → List G.Coin → OC Msg) :
    ({ G with msg := m } : Proto W).dRunSP = G.dRunSP := rfl

/-- **`S_GK` reads the dummy-witness prover, the coins and the SHVZK simulator, never the prover's messages**: `G` with
any other `msg` has `G`'s simulator. -/
theorem Proto.simulatorE_msg {W : Type} (G : Proto W)
    (m : Msg → W → G.Nonce → G.Rand → List G.Coin → OC Msg) (t M : ℕ) :
    ({ G with msg := m } : Proto W).simulatorE t M = G.simulatorE t M := by
  unfold Proto.simulatorE
  simp only [Proto.estSP_msg, Proto.rewSP_msg, Proto.dRunSP_msg]
  congr 1
  funext x c
  congr 1
  funext p
  rcases p with ⟨τ, _ | cs⟩ <;> rfl

/-! ## The switched prover at the reference session -/

namespace RefSession

/-- **The values the workers supply**: at each statement, witness, nonce and coin vector, each table's clear values
(`Table.Vals`, read from the mask slot and the extra lanes). Its arguments are what a pad-blind worker
(`ZkServer.BWorker`) sees: the statement, the witness, the shared part of the tape (the nonce) and the coins, never the
server's tape, which holds the pads. -/
abbrev SwVals (S : RefSession) : Type :=
  (x : Msg) → Flock.Wit → S.Nonce → (c : Fin (refK x) → S.Coin) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals

/-- **The switched prover's cut pre-final view at coins `c`**, on table witnesses `wt`, values `v` and tape `t`, every
pre-final leaf the executable's under its own salt (`cutOf` at the honest prover). -/
noncomputable def swCutOf (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (v : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Cut :=
  swCutAt (S.C x) (S.cz x c) wt v (S.eP x t).1 (swExeOf (S.C x) (S.cz x c) wt v (S.eP x t).1 (S.eP x t).2)

/-- The tables' switched views at coins `c` (`Coined.swExe`). -/
noncomputable def swViewsAt (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (v : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) : ∀ j, ((S.C x j).T (S.cz x c j)).tab.View :=
  fun j => ((S.C x j).T (S.cz x c j)).swExe (wt j) (v j) (S.randAt x c t j).1 (S.randAt x c t j).2

open Classical in
/-- **The switched prover's message after the coins `cs`** at the values `v`, as `msgAt` has the honest prover's. -/
noncomputable def swMsgAt (S : RefSession) (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m)
    (v : (c : Fin (refK x) → S.Coin) → ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals) (t : S.Tape) (cs : List S.Coin) :
    Msg :=
  if cs.length < refK x then S.enc x (S.pad cs) cs.length (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)
  else if S.Admits x (S.pad cs) then S.fin x (S.pad cs) (S.swViewsAt x (S.pad cs) wt (v (S.pad cs)) t)
  else S.off x (S.pad cs) (S.swCutOf x (S.pad cs) wt (v (S.pad cs)) t)

/-- **The switched reference prover** at the workers' values `vs`: `refP` with its messages on each witness
`swMsgAt`; its dummy, coins, nonce, tape and opening check `refP`'s. -/
noncomputable def swP (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.refP with msg := fun x w ν t cs => .ret (S.swMsgAt x (S.wt x w) (vs x w ν) t cs) }

noncomputable instance instFintypeSwPCoin (S : RefSession) (vs : S.SwVals) : Fintype (S.swP vs).Coin := S.finCoin

/-- **The switched round encoding reads the coins before `i` only**, whatever each pre-final leaf's pair, at every
witness and nonce (`Causal` at the switched prover). -/
def SwCausal (S : RefSession) (vs : S.SwVals) : Prop :=
  ∀ (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (ω : ∀ j, ZkView.Base (S.C x j).z (S.C x j).Hi)
    (p : ZkView.Leaves (fun j => (S.C x j).z) → ZkView.Bits × ByteArray) (c c' : Fin (refK x) → S.Coin) (i : ℕ),
    (∀ j : Fin (refK x), j.val < i → c j = c' j) →
    S.enc x c i (swCutAt (S.C x) (S.cz x c) (S.wt x w) (vs x w ν c) ω fun l =>
        ZkView.leafH (swDigOf (S.C x) (S.cz x c) (S.wt x w) (vs x w ν c) ω l + (p l).1, (p l).2)) =
      S.enc x c' i (swCutAt (S.C x) (S.cz x c') (S.wt x w) (vs x w ν c') ω fun l =>
        ZkView.leafH (swDigOf (S.C x) (S.cz x c') (S.wt x w) (vs x w ν c') ω l + (p l).1, (p l).2))

/-- `preModel` at `swP`. -/
noncomputable def swPreModel (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (ν : S.Nonce) :
    (S.swP vs).TPreModel x where
  J := S.J x
  C := S.C x
  cz a := S.cz x (S.coinsOf a)
  out a σ i := S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1) σ)) (ansHist a i)
  wt := S.wt x
  wt₀ := S.wt₀ x
  eR := S.eP x
  kR := S.kP x

open Classical in
/-- `coinModel` at `swP`. -/
noncomputable def swCoinModel (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.swP vs).TCoinModel x :=
  if S.Admits x c then
    .inl { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           lift := fun V => (List.range (refK x + 1)).map fun j =>
             if j < refK x then S.enc x c j (S.cutV x c V) else S.fin x c V
           eR := S.randAt x c, kR := S.kT x c }
  else
    .inr { J := S.J x, C := S.C x, cz := S.cz x c, wt := S.wt x, wt₀ := S.wt₀ x
           out := fun σ => (List.range (refK x + 1)).map fun j => if j < refK x then S.enc x c j σ else S.off x c σ
           eR := S.eP x, kR := S.kP x }

/-- **The switched real views' pre-final part, cut, is the cut view the switched round messages encode**
(`viewPre_swExe`). -/
theorem cutV_swViewsAt (S : RefSession) (x : Msg) (c : Fin (refK x) → S.Coin)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (v : ∀ j, ((S.C x j).T (S.cz x c j)).tab.Vals)
    (t : S.Tape) : S.cutV x c (S.swViewsAt x c wt v t) = S.swCutOf x c wt v t := by
  funext j
  simp only [cutV, swViewsAt, randAt, viewPre_swExe]
  rfl

/-- **Round `i`'s switched encoding reads the coins before `i` only** (`SwCausal` at the executable's leaves). -/
theorem enc_swCutOf (S : RefSession) (vs : S.SwVals) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausal vs)
    (t : S.Tape) (c c' : Fin (refK x) → S.Coin) (i : ℕ) (h : ∀ k : Fin (refK x), k.val < i → c k = c' k) :
    S.enc x c i (S.swCutOf x c (S.wt x w) (vs x w ν c) t) = S.enc x c' i (S.swCutOf x c' (S.wt x w) (vs x w ν c') t) := by
  have e := hc x w ν (S.eP x t).1 (fun l => (ZkView.My ((S.eP x t).2 l), ZkView.saltHash ((S.eP x t).2 l))) c c' i h
  unfold swCutOf
  rw [swExeOf_eq, swExeOf_eq]
  convert e using 4

open Classical in
/-- **The switched prover's messages at the prefixes of a coin vector** (`SwCausal`). -/
theorem swMsgAt_take (S : RefSession) (vs : S.SwVals) (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausal vs)
    (t : S.Tape) (c : Fin (refK x) → S.Coin) (j : ℕ) (hj : j < refK x + 1) :
    S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      if j < refK x then S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t)
      else if S.Admits x c then S.fin x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t)
      else S.off x c (S.swCutOf x c (S.wt x w) (vs x w ν c) t) := by
  have hl : ((List.ofFn c).take j).length = j := by simp only [List.length_take, List.length_ofFn]; omega
  unfold swMsgAt
  rw [hl]
  by_cases h : j < refK x
  · rw [ite_eq_left h, ite_eq_left h]
    exact S.enc_swCutOf vs x w ν hc t _ c j fun k hk => S.pad_take c j k hk
  · rw [ite_eq_right h, ite_eq_right h]
    obtain rfl : j = refK x := by omega
    rw [S.pad_ofFn c]

/-- **Before the last coin, the switched prover sends the switched round encoding at the coins the answers open**
(`SwCausal`). -/
theorem swPre_msg (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (hc : S.SwCausal vs) (t : S.Tape) (a : Fin (refK x + 1) → Out) (i : Fin (refK x + 1)) :
    S.refP.pmG H x ν (fun cs => .ret (S.swMsgAt x (S.wt x w) (vs x w ν) t cs)) (ansHist a i) =
      S.refP.pmG H x ν (fun _ => .ret (S.enc x (S.coinsOf a) (i.val - 1)
        (S.swCutOf x (S.coinsOf a) (S.wt x w) (vs x w ν (S.coinsOf a)) t))) (ansHist a i) := by
  unfold Proto.pmG
  congr 2
  refine strat_congr_gen _ _ _ _ _ _ _ fun hne => ?_
  have hi : 0 < i.val := by
    rcases Nat.eq_zero_or_pos i.val with h0 | h0
    · exact absurd (by simp [ansHist, h0]) hne
    · exact h0
  have hl : ((ansHist a i).tail.map fun p => S.decOf p.2).length = i.val - 1 := by
    simp only [List.length_map, List.length_tail, ansHist, List.length_ofFn]
  show OC.ret (S.swMsgAt x (S.wt x w) (vs x w ν) t ((ansHist a i).tail.map fun p => S.decOf p.2)) = OC.ret _
  unfold swMsgAt
  rw [hl, ite_eq_left (by omega),
    S.enc_swCutOf vs x w ν hc t _ (S.coinsOf a) (i.val - 1) fun k hk => S.pad_ansHist a i k hk]

theorem swPreModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) : (S.swPreModel vs H x ν).SwLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x hx).1 p _,
    ⟨fun a => vs x w ν (S.coinsOf a), ?_, fun t a => funext fun i => S.swPre_msg vs H x w ν hsc t a i⟩,
    fun t a => funext fun i => S.pre_msg H x ν hc (S.wt₀ x) t a i⟩
  · intro wt ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact hc x wt ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))
  · intro ω p a a' i h
    show S.refP.pmG H x ν _ (ansHist a i) = S.refP.pmG H x ν _ (ansHist a' i)
    rw [ansHist_congr a a' i h]
    refine congrArg (fun g => S.refP.pmG H x ν g (ansHist a' i)) (funext fun _ => congrArg OC.ret ?_)
    exact hsc x w ν ω p (S.coinsOf a) (S.coinsOf a') (i.val - 1) fun k hk =>
      congrArg (fun o => S.decOf (msgOf o)) (h k.succ (Fin.lt_def.mpr (by simp only [Fin.val_succ]; omega)))

theorem swCoinModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    (hV : S.Complete R) (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOKOn A) (x : Msg)
    (hx : A x) (w : Flock.Wit) (hR : R x w) (ν : S.Nonce) (c : Fin (refK x) → S.Coin) :
    (S.swCoinModel vs x c).SwLink H ν c w := by
  unfold swCoinModel
  split_ifs with hA
  · refine ⟨vs x w ν c, hV x w hR, fun p => (hT x hx).2 c p _, fun (t : S.Tape) => ?_⟩
    show (List.range (refK x + 1)).map (fun j => S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutV x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t))
        else S.fin x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t)
    refine List.map_congr_left fun j hj => ?_
    rw [S.swMsgAt_take vs x w ν hsc t c j (List.mem_range.1 hj), S.cutV_swViewsAt, ite_eq_left hA]
  · refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x hx).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
      fun (t : S.Tape) => ?_⟩
    · show (List.range (refK x + 1)).map (fun j => S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t)
          else S.off x c (S.swCutOf x c (S.wt x w) (vs x w ν c) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.swMsgAt_take vs x w ν hsc t c j (List.mem_range.1 hj), ite_eq_right hA]
    · show (List.range (refK x + 1)).map (fun j => S.msgAt x (S.wt₀ x) t ((List.ofFn c).take j)) =
        (List.range (refK x + 1)).map fun j =>
          if j < refK x then S.enc x c j (S.cutOf x c (S.wt₀ x) t) else S.off x c (S.cutOf x c (S.wt₀ x) t)
      refine List.map_congr_left fun j hj => ?_
      rw [S.msgAt_take x hc _ t c j (List.mem_range.1 hj), ite_eq_right hA]

theorem swTablesOK (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (hT : S.Tables) :
    (S.swP vs).TablesOK (fun x ν => S.swPreModel vs H x ν) fun x _ c => S.swCoinModel vs x c := by
  refine ⟨fun x _ => hT x, fun x _ c => ?_⟩
  show (S.swCoinModel vs x c).OK
  unfold swCoinModel
  split_ifs with h
  · exact ⟨hT x, h⟩
  · exact hT x

theorem swCoinModel_size (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    (S.swCoinModel vs x c).size ≤ S.nLeaves x := by
  unfold swCoinModel
  split_ifs
  · show ∑ j, Fintype.card ((S.C x j).T (S.cz x c j)).tab.Hid ≤ Fintype.card (ZkView.Leaves fun j => (S.C x j).z)
    rw [Fintype.card_sigma]
    exact Finset.sum_le_sum fun j _ => Fintype.card_le_of_injective _ (hidPos_injective ((S.C x j).T (S.cz x c j)))
  · exact le_rfl

/-- **The honest values**: each table's own (`Table.vals`). -/
noncomputable def honestVals (S : RefSession) : S.SwVals := fun x _ _ c j => ((S.C x j).T (S.cz x c j)).tab.vals

theorem swMsgAt_vals (S : RefSession) (x : Msg) (w : Flock.Wit) (ν : S.Nonce)
    (wt : ∀ j, FlockSoundness.Model.Witness (S.C x j).S.m) (t : S.Tape) (cs : List S.Coin) :
    S.swMsgAt x wt (S.honestVals x w ν) t cs = S.msgAt x wt t cs := by
  have hc : ∀ c, S.swCutOf x c wt (S.honestVals x w ν c) t = S.cutOf x c wt t := fun c => by
    unfold swCutOf cutOf honestVals
    rw [swExeOf_vals, swCutAt_vals]
  have hv : ∀ c, S.swViewsAt x c wt (S.honestVals x w ν c) t = S.viewsAt x c wt t := fun c => by
    funext j
    exact ZkView.Coined.swExe_vals _ _ _ _
  unfold swMsgAt msgAt
  rw [hc, hv]

/-- **At the honest values the switched prover is `refP`.** -/
theorem swP_vals (S : RefSession) : S.swP S.honestVals = S.refP := by
  have h : (fun x w (ν : S.Nonce) (t : S.Tape) cs => OC.ret (S.swMsgAt x (S.wt x w) (S.honestVals x w ν) t cs)) =
      S.refP.msg :=
    funext fun x => funext fun w => funext fun ν => funext fun t => funext fun cs =>
      congrArg OC.ret (S.swMsgAt_vals x w ν (S.wt x w) t cs)
  exact congrArg (fun m => ({ S.refP with msg := m } : Proto Flock.Wit)) h

/-- **At the honest values `SwCausal` is `Causal`.** -/
theorem swCausal_vals (S : RefSession) (hc : S.Causal) : S.SwCausal S.honestVals := by
  intro x w ν ω p c c' i h
  unfold honestVals
  rw [swDigOf_vals, swDigOf_vals, swCutAt_vals, swCutAt_vals]
  exact hc x (S.wt x w) ω p c c' i h

/-! ## The switched prover with the honest prover's simulator -/

/-- **The switched reference prover with `refP`'s simulator**: `refP` with the models' simulator (`withSim` at
`coinModel`), its messages `swP vs`'s. Its prover is `swP vs`'s and its simulator `refP`'s. -/
noncomputable def swPS (S : RefSession) (vs : S.SwVals) : Proto Flock.Wit :=
  { S.refP.withSim (fun x _ c => S.coinModel x c) with msg := (S.swP vs).msg }

noncomputable instance instFintypeSwPSCoin (S : RefSession) (vs : S.SwVals) : Fintype (S.swPS vs).Coin := S.finCoin

theorem swPS_prover (S : RefSession) (vs : S.SwVals) : (S.swPS vs).prover = (S.swP vs).prover := rfl

theorem swPS_simulatorE (S : RefSession) (vs : S.SwVals) (t M : ℕ) :
    (S.swPS vs).simulatorE t M = (S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t M :=
  Proto.simulatorE_msg _ _ t M

/-- `refP`'s model at a coin vector, typed at `swP`. -/
noncomputable def toSw (S : RefSession) (vs : S.SwVals) (x : Msg) :
    S.refP.TCoinModel x → (S.swP vs).TCoinModel x
  | .inl M => .inl { J := M.J, C := M.C, cz := M.cz, wt := M.wt, wt₀ := M.wt₀, lift := M.lift, eR := M.eR,
                     kR := M.kR }
  | .inr M => .inr { J := M.J, C := M.C, cz := M.cz, out := M.out, wt := M.wt, wt₀ := M.wt₀, eR := M.eR,
                     kR := M.kR }

theorem swCoinModel_eq (S : RefSession) (vs : S.SwVals) (x : Msg) (c : Fin (refK x) → S.Coin) :
    S.swCoinModel vs x c = S.toSw vs x (S.coinModel x c) := by
  unfold swCoinModel coinModel
  split_ifs <;> rfl

/-- **`SHVZK` of the switched prover at a coin vector against `refP`'s model's simulator** (`TCoinModel.shvzk_sw` at
`toSw`). -/
theorem toSw_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (x : Msg) (ν : S.Nonce)
    (c : Fin (refK x) → S.Coin) (w : Flock.Wit) {n : ℕ} (E : List Msg → Prop) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (sim : Ω → List Msg) :
    ∀ (m : S.refP.TCoinModel x) (ev : Ω → m.SimAt) (kS : ℕ), (∀ p, Nat.card {s // ev s = p} = kS) →
      (∀ s, sim s = ((m.simAt (ev s) ν c).eval H).1) → (S.toSw vs x m).SwLink H ν c w → (S.toSw vs x m).OK →
      (S.toSw vs x m).size ≤ n →
      |Pr (fun r : S.Tape => E ((S.swP vs).realVec H x w ν r c)) - Pr (fun s : Ω => E (sim s))| ≤ 2 * n * δ₁.toReal
  | .inl M, ev, kS, hev, hsim, hL, hOK, hn =>
      Proto.TCoinModel.shvzk_sw (G := S.swP vs) H hT1 hδ₁ ν c w E sim (S.toSw vs x (.inl M)) ev kS hev hsim hL hOK hn
  | .inr M, ev, kS, hev, hsim, hL, hOK, hn =>
      Proto.TCoinModel.shvzk_sw (G := S.swP vs) H hT1 hδ₁ ν c w E sim (S.toSw vs x (.inr M)) ev kS hev hsim hL hOK hn

/-- **`SHVZK` of `swPS vs`**: the switched prover's messages against `refP`'s simulator, within `2·|L|·δ₁`, on a
relation whose statements are in `A`, where the tape is read with fibers of one size (`TapeOKOn A`). -/
theorem swPS_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (A : Msg → Prop)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOKOn A) (hRA : ∀ x w, R x w → A x)
    (hS : S.Tables) : (S.swPS vs).SHVZK H R fun x => 2 * S.nLeaves x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := Proto.TCoinModel.instNonemptySimAt (S.coinModel x c)
  have e := S.swCoinModel_eq vs x c
  have hL := S.swCoinModel_link vs H R A hV hD hc hsc hT x (hRA x w hR) w hR ν c
  have hOK : (S.swCoinModel vs x c).OK := (S.swTablesOK vs H hS).2 x ν c
  have hn := S.swCoinModel_size vs x c
  rw [e] at hL hOK hn
  exact S.toSw_shvzk vs H hT1 hδ₁ x ν c w E (Ω := (S.swPS vs).SimRand x)
    (fun s => (((S.swPS vs).sim x ν c s).eval H).1) (S.coinModel x c) (fun s => s (ν, c)) _
    (fun p => Proto.eval_fiber_card
      (β := fun q : S.Nonce × (Fin (refK x) → S.Coin) => (S.coinModel x q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) hL hOK hn

end RefSession

/-- **Flock's zero knowledge at the switched reference prover, at a masked record whose statements are in `A`**: the
common proof of `Flock.zk_deployZk_sw` (at `deployZk`, every statement) and `Flock.zk_deployZk_sw_on` (at `deployOn`). -/
theorem Flock.zkAt_sw_of {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (hzk : D.zk = true)
    (S : RefSession) (vs : S.SwVals) (hk : ∀ x, D.V.k x + 1 = (S.swPS vs).k x) (A : Msg → Prop) (t : ℕ) (ht : 0 < t)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R D.parse)) (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs)
    (hT : S.TapeOKOn A) (hRA : ∀ x w, Flock.R D.parse x w → A x) :
    Claims.Flock.ZKAt H D (S.swP vs).prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  have hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : S.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (refK x + 1)) (E : (Fin (refK x + 1) → Msg) → Prop),
      |Pr (fun r : S.Tape => E (FlockSoundness.ZK.run ((S.swPS vs).preStrat H x ν ((S.swPS vs).msg x w ν)) V r)) -
        Pr (fun r : S.Tape => E (FlockSoundness.ZK.run ((S.swPS vs).preStrat H x ν ((S.swPS vs).dmsg x ν)) V r))| ≤
        2 * S.nLeaves x * δ₁.toReal := fun x w hR ν V E =>
    (S.swPreModel vs H x ν).runs_close_sw H hT1 hδ₁ ν w
      (S.swPreModel_link vs H _ A hV hD hc hsc hT x (hRA x w hR) w hR ν) ((S.swTablesOK vs H hS).1 x ν) V E
  rw [← S.swPS_prover vs, ← S.swPS_simulatorE vs]
  exact Flock.zkR_gkE_hm96_bind H D (S.swPS vs) t ht hzk hk δ₁ S.nLeaves S.nLeaves hA
    (S.swPS_shvzk vs H _ A hT1 hδ₁ hV hD hc hsc hT hRA hS) (S.binding H)

/-- **Flock's zero knowledge at the switched reference prover, with the honest prover's simulator**: `Claims.Flock.ZKAt`
at `P_zk = swP vs`'s prover, at any values `vs` the workers supply whose round encoding is causal, and the simulator of
`Flock.zkAt_deployZk_ref` (`refP`'s), at its `ε_zk` and `γ`, under `Claims.Hm96Hiding` and the session's five Props. One
simulator serves every `vs`. -/
theorem Flock.zk_deployZk_sw (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) (S.swP vs).prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zkAt_sw_of H _ rfl S vs (fun x => (deployZk_proto_k rel (S.swPS vs) (fun _ => rfl) x).symm) (fun _ => True)
    t ht hT1 hδ₁ hS hV hD hc hsc ((S.tapeOKOn_iff).2 hT) fun _ _ _ => trivial

/-- **The same on a class of statements**: `Claims.Flock.ZKAt` at the record whose parser refuses every statement
outside `A` (`deployOn`), with `TapeOK` asked for only on `A` (`RefSession.TapeOKOn`). -/
theorem Flock.zk_deployZk_sw_on (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (vs : S.SwVals)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w) (hD : S.DummyReg)
    (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOKOn A) :
    Claims.Flock.ZKAt H (deployOn (FlockDeploy.deployZk rel) A) (S.swP vs).prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zkAt_sw_of H (deployOn (FlockDeploy.deployZk rel) A) rfl S vs
    (fun x => (deployZk_proto_k rel (S.swPS vs) (fun _ => rfl) x).symm) A t ht hT1 hδ₁
    hS (fun x w hR => hV x w ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR)) hD hc hsc hT
    fun x w hR => ((r_deployOn (FlockDeploy.deployZk rel) A x w).1 hR).1

/-- **At the honest values `Flock.zk_deployZk_sw` is `Flock.zkAt_deployZk_ref`**: the switched prover is `refP`
(`RefSession.swP_vals`) and `SwCausal` is `Causal` (`RefSession.swCausal_vals`). -/
theorem Flock.zkAt_deployZk_ref_of_sw (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (t : ℕ) (ht : 0 < t)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hT : S.TapeOK) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) S.refP.prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  have h := Flock.zk_deployZk_sw H rel S S.honestVals t ht hT1 hδ₁ hS hV hD hc (S.swCausal_vals hc) hT
  rwa [S.swP_vals] at h

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at pad-blind workers, at `flock-circuit --zk`'s reference session**: the server defers every
stop to the session's last round (`ZkServer.nextD`, live PROTOCOL.md §10.1: a failed check stops the session at its
`Finish`, never before), putting in its own stand-in `fill` for a masked value once a round failed. For every pad-blind
worker `bw`, handed the nonce and the verifier's answers and never the server's tape, and any values `vs` whose round
encoding is causal, if the deferred server's frame is `swP vs`'s message wherever it releases one (`ZkServer.RefD`), it
releases, on every event, at most twice the probability at the simulator that guesses whether the session stops
(`leaky1`, at `Flock.zk_server_ref`'s simulator, `refP`'s), plus `ε_zk + γ·Fork`, under `Claims.Hm96Hiding` and the
session's five Props, with no term for the masked checks. The stop leaks one bit a session; over a stream of `m`
sessions with at most `b` stops, `log₂ ∑_{i ≤ b} C(m, i)` bits (`stream_leak_deferred`). -/
theorem Flock.zk_server_blind (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (fill : Msg → ℕ → ℕ → Flock.Wit → S.refP.Nonce × S.refP.Rand → List (Msg × Msg) → srv.Val)
    (bw : srv.BWorker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hRef : srv.RefD H (S.swP vs).prover fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) x w) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × srv.Chk =>
        Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
          Ev (srv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) := by
  have h := srv.zk_deferred fill H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (S.swP vs).prover _ _ _ _
    (Flock.zk_deployZk_sw H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT).2 x w hR (srv.lift bw) hRef.own Vs θ hk Ev
  exact h

/-- **The ZK server at pad-blind workers on a class of statements**: `Flock.zk_server_blind` at a statement in `A`,
with `TapeOK` and the lowering's completeness asked for only on `A` (`RefSession.TapeOKOn`), as a tape of bounded size
can meet it where `A` bounds the tables. -/
theorem Flock.zk_server_blind_on (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession) (vs : S.SwVals)
    (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w) (hD : S.DummyReg)
    (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOKOn A) (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (fill : Msg → ℕ → ℕ → Flock.Wit → S.refP.Nonce × S.refP.Rand → List (Msg × Msg) → srv.Val)
    (bw : srv.BWorker) (x : Msg) (hx : A x) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hRef : srv.RefD H (S.swP vs).prover fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) x w) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × srv.Chk =>
        Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
          Ev (srv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) := by
  have h := srv.zk_deferred fill H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (S.swP vs).prover _ _ _ _
    (Flock.zk_deployZk_sw_on H rel S vs A t ht hT1 hδ₁ hS hV hD hc hsc hT).2 x w
    ((r_deployOn (FlockDeploy.deployZk rel) A x w).2 ⟨hx, hR⟩) (srv.lift bw) hRef.own Vs θ hk Ev
  exact h

/-- **The ZK server at pad-blind workers at the deployed coin layer**: `Flock.zk_server_blind_on` at the reference
session of `coin-tree/hm96-sha512/v2` (`CoinLayer.session`, `zkDeployed`'s), at the pinned key's hiding gap, with the
tape asked for only as the tables' tape read with fibers of one size on `A` (`RefSession.tapeOKOn_of_tape`). -/
theorem Flock.zk_server_blind_deployed (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (vs : (CoinLayer.session nr T).SwVals) (A : Msg → Prop) [DecidablePred A] (t : ℕ) (ht : 0 < t)
    (hS : (CoinLayer.session nr T).Tables)
    (hV : (CoinLayer.session nr T).Complete fun x w => A x ∧ Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hD : (CoinLayer.session nr T).DummyReg) (hc : (CoinLayer.session nr T).Causal)
    (hsc : (CoinLayer.session nr T).SwCausal vs) (hT : ∀ x, A x → FibersEq (T.tape x))
    (srv : ZkServer Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (fill : Msg → ℕ → ℕ → Flock.Wit → (CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand →
      List (Msg × Msg) → srv.Val)
    (bw : srv.BWorker) (x : Msg) (hx : A x) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hRef : srv.RefD H ((CoinLayer.session nr T).swP vs).prover fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) x w) {Θ : Type} (Vs : VStar Θ)
    (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand) × srv.Chk =>
        Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 (((CoinLayer.session nr T).refP.withSim
              fun x _ c => (CoinLayer.session nr T).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
          Ev (srv.timed x ((leaky1 (((CoinLayer.session nr T).refP.withSim
              fun x _ c => (CoinLayer.session nr T).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
          (8 * (CoinLayer.session nr T).nLeaves x *
              (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
            8 * t * (CoinLayer.session nr T).nLeaves x *
              (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
            2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H (((CoinLayer.session nr T).swP vs).prover w) Vs θ x).Pr H) :=
  Flock.zk_server_blind_on H rel (CoinLayer.session nr T) vs A t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc
    ((CoinLayer.session nr T).tapeOKOn_of_tape A hT) srv fill bw x hx w hR hRef Vs θ hk Ev

end Claims.ZKServer
