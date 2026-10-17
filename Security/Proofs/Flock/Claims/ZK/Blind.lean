import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.Server

/-!
# Zero knowledge at the reference session, at any values pad-blind workers supply

`Switched` has Lemmas A, B and C at the switched prover: the honest prover at the workers' values where their rows hold,
else the honest prover on moved pads. Here that prover is defined at the reference session, it meets `Claims.Flock.ZK`
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
  **`Flock.zk_deployZk_sw`**: `Claims.Flock.ZK` at `swP vs`'s prover with `Flock.zk_deployZk_ref`'s simulator, `ε_zk`
  and `γ`: one simulator for every `vs`.
* **`Flock.zk_server_blind`**: for every pad-blind worker and any values `vs` whose round encoding is causal, the server
  whose frame at the worker's values is `swP vs`'s message where the run is live (`RefLive` at `withHonest`, `srcOf`)
  releases, on every event, at most `D.V.k x + 2` times the guessing simulator's probability plus `ε_zk + γ·Fork`, with
  `Flock.zk_server_ref`'s simulator and no term for the masked checks.
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

theorem swPreModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w)
    (ν : S.Nonce) : (S.swPreModel vs H x ν).SwLink H ν w := by
  refine ⟨hV x w hR, fun a j => hD x (S.coinsOf a) j, ?_, fun p => (hT x).1 p _,
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

theorem swCoinModel_link (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) (hV : S.Complete R)
    (hD : S.DummyReg) (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOK) (x : Msg) (w : Flock.Wit) (hR : R x w)
    (ν : S.Nonce) (c : Fin (refK x) → S.Coin) : (S.swCoinModel vs x c).SwLink H ν c w := by
  unfold swCoinModel
  split_ifs with hA
  · refine ⟨vs x w ν c, hV x w hR, fun p => (hT x).2 c p _, fun (t : S.Tape) => ?_⟩
    show (List.range (refK x + 1)).map (fun j => S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j)) =
      (List.range (refK x + 1)).map fun j =>
        if j < refK x then S.enc x c j (S.cutV x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t))
        else S.fin x c (S.swViewsAt x c (S.wt x w) (vs x w ν c) t)
    refine List.map_congr_left fun j hj => ?_
    rw [S.swMsgAt_take vs x w ν hsc t c j (List.mem_range.1 hj), S.cutV_swViewsAt, ite_eq_left hA]
  · refine ⟨hV x w hR, fun j => hD x c j, fun p => (hT x).1 p _, ⟨vs x w ν c, fun (t : S.Tape) => ?_⟩,
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

/-- **`SHVZK` of `swPS vs`**: the switched prover's messages against `refP`'s simulator, within `2·|L|·δ₁`. -/
theorem swPS_shvzk (S : RefSession) (vs : S.SwVals) (H : Msg → Msg) (R : Rel Flock.Wit) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hV : S.Complete R) (hD : S.DummyReg)
    (hc : S.Causal) (hsc : S.SwCausal vs) (hT : S.TapeOK) (hS : S.Tables) :
    (S.swPS vs).SHVZK H R fun x => 2 * S.nLeaves x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := Proto.TCoinModel.instNonemptySimAt (S.coinModel x c)
  have e := S.swCoinModel_eq vs x c
  have hL := S.swCoinModel_link vs H R hV hD hc hsc hT x w hR ν c
  have hOK : (S.swCoinModel vs x c).OK := (S.swTablesOK vs H hS).2 x ν c
  have hn := S.swCoinModel_size vs x c
  rw [e] at hL hOK hn
  exact S.toSw_shvzk vs H hT1 hδ₁ x ν c w E (Ω := (S.swPS vs).SimRand x)
    (fun s => (((S.swPS vs).sim x ν c s).eval H).1) (S.coinModel x c) (fun s => s (ν, c)) _
    (fun p => Proto.eval_fiber_card
      (β := fun q : S.Nonce × (Fin (refK x) → S.Coin) => (S.coinModel x q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) hL hOK hn

end RefSession

/-- **Flock's zero knowledge at the switched reference prover, with the honest prover's simulator**: `Claims.Flock.ZK`
at `P_zk = swP vs`'s prover, at any values `vs` the workers supply whose round encoding is causal, and the simulator of
`Flock.zk_deployZk_ref` (`refP`'s), at its `ε_zk` and `γ`, under `Claims.Hm96Hiding` and the session's five Props. One
simulator serves every `vs`. -/
theorem Flock.zk_deployZk_sw (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) (S.swP vs).prover
      ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
      (fun x => 8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t)
      fun _ => 4 * t * 3 ^ t := by
  obtain ⟨_, _, Tr, nb, root, parse, pos, idx, n, depth, dc, hchk, hdec⟩ := S.coinCheck
  have hA : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → ∀ (ν : S.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (refK x + 1)) (E : (Fin (refK x + 1) → Msg) → Prop),
      |Pr (fun r : S.Tape => E (FlockSoundness.ZK.run ((S.swPS vs).preStrat H x ν ((S.swPS vs).msg x w ν)) V r)) -
        Pr (fun r : S.Tape => E (FlockSoundness.ZK.run ((S.swPS vs).preStrat H x ν ((S.swPS vs).dmsg x ν)) V r))| ≤
        2 * S.nLeaves x * δ₁.toReal := fun x w hR ν V E =>
    (S.swPreModel vs H x ν).runs_close_sw H hT1 hδ₁ ν w (S.swPreModel_link vs H _ hV hD hc hsc hT x w hR ν)
      ((S.swTablesOK vs H hS).1 x ν) V E
  rw [← S.swPS_prover vs, ← S.swPS_simulatorE vs]
  exact Flock.zkR_gkE_hm96_deployZk H rel (S.swPS vs) t ht (fun _ => rfl) δ₁ S.nLeaves S.nLeaves hA
    (S.swPS_shvzk vs H _ hT1 hδ₁ hV hD hc hsc hT hS) Tr nb root parse pos idx n depth dc hchk hdec

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The ZK server at pad-blind workers, at `flock-circuit --zk`'s reference session**: for every pad-blind worker
`bw`, handed the nonce and the verifier's answers and never the server's tape, and any values `vs` whose round encoding
is causal, the server whose frame at the worker's values is `swP vs`'s message where the run is live releases, on every
event, at most `D.V.k x + 2` times the probability at the guessing simulator of `Flock.zk_server_ref`'s simulator
(`refP`'s), plus `ε_zk + γ·Fork`, under `Claims.Hm96Hiding` and the session's five Props, with no term for the masked
checks. -/
theorem Flock.zk_server_blind (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (S : RefSession) (vs : S.SwVals) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hS : S.Tables)
    (hV : S.Complete (Flock.R (FlockDeploy.deployZk rel).parse)) (hD : S.DummyReg) (hc : S.Causal)
    (hsc : S.SwCausal vs) (hT : S.TapeOK) (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand))
    (bw : srv.BWorker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hRef : (srv.withHonest (srv.srcOf (srv.lift bw))).RefLive H (S.swP vs).prover (srv.lift bw) x w) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : (S.refP.Nonce × S.refP.Rand) × srv.Chk =>
        Ev (srv.timed x (runStar H (srv.prover (srv.lift bw) w) Vs x ω θ).msgs)) ≤
      ((FlockDeploy.deployZk rel).V.k x + 1 + 1) *
        (Pr (fun c : (leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
          Ev (srv.timed x ((leaky ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
          (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) := by
  have h := srv.zk_server_own H (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (S.swP vs).prover _ _ _ _
    (Flock.zk_deployZk_sw H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT).2 x w hR (srv.lift bw) hRef Vs θ hk Ev
  push_cast at h ⊢
  exact h

end Claims.ZKServer
