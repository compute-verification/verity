module

public import Proofs.Flock.Claims.ZK.Refine
public import Proofs.Flock.Claims.ZK.Split
public import Proofs.Flock.Claims.ZK.Switched
public import Proofs.Flock.Claims.ZK.LinkFft

/-! The link round at the Lean recheck. -/

@[expose] public section

namespace Flock.ZkServer.Link

theorem flat_eq : flat = FlockComplete.flat := rfl

theorem layer_eq (d : Nat) (lf : Nat → ByteArray) :
    ∀ h, layer d lf h = FlockComplete.layerOf (fun l r => Sha512.hash (l ++ r)) d lf h
  | 0 => rfl
  | h + 1 => by rw [layer, FlockComplete.layerOf, layer_eq d lf h]

theorem layerOf_congr (nd : ByteArray → ByteArray → ByteArray) (d : Nat) {lf lf' : Nat → ByteArray}
    (hl : ∀ q, q < 2 ^ d → lf q = lf' q) : ∀ h, FlockComplete.layerOf nd d lf h = FlockComplete.layerOf nd d lf' h
  | 0 => by
    simp only [FlockComplete.layerOf]
    exact Array.map_congr_left fun q hq => hl q (Array.mem_range.1 hq)
  | h + 1 => by rw [FlockComplete.layerOf, FlockComplete.layerOf, layerOf_congr nd d hl h]

theorem leaves_getElem! (lv : Level) (tPad : Nat) (i : In) {q : Nat} (hq : q < 2 ^ lv.d) :
    (leaves lv tPad i)[q]! = leaf ((rows lv tPad i.g i.extra i.lanePad)[q]!) (i.salt q) := by
  have hk : 2 ^ (lv.d - min lv.d CHUNK) * 2 ^ min lv.d CHUNK = 2 ^ lv.d := by
    rw [← Nat.pow_add, Nat.sub_add_cancel (Nat.min_le_left _ _)]
  simp only [leaves, parRange_eq, hk]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq]

theorem cap_eq (lv : Level) (tPad : Nat) (i : In) (lf : Nat → ByteArray)
    (hl : ∀ q, q < 2 ^ lv.d → lf q = leaf ((rows lv tPad i.g i.extra i.lanePad)[q]!) (i.salt q)) :
    cap lv tPad i = FlockComplete.layerOf (fun l r => Sha512.hash (l ++ r)) lv.d lf (lv.d - lv.cap) := by
  show layer lv.d (fun q => (leaves lv tPad i)[q]!) (lv.d - lv.cap) = _
  rw [layer_eq]
  exact layerOf_congr _ _ (fun q hq => (leaves_getElem! lv tPad i hq).trans (hl q hq).symm) _

theorem root_eq (i : In) {s : _root_.Flock.Schedule} {tPad : Nat} (hz : Zk.schedule i.wideLeaf i.m = .ok (s, tPad)) :
    root i = flat (cap (s.levels.getD 0 default) tPad i) := by
  unfold root
  rw [hz]

theorem sumF_congr {n : Nat} {f f' : Nat → F128} (h : ∀ c, c < n → f c = f' c) :
    FlockComplete.sumF n f = FlockComplete.sumF n f' := by
  unfold FlockComplete.sumF
  exact List.foldl_ext _ _ _ fun a c hc => by simp only [h c (List.mem_range.1 hc)]

theorem rows_congr (lv : Level) (tPad : Nat) (g : Array F128) {extra extra' lanePad lanePad' : Nat → Nat → F128}
    (he : ∀ e c, c < 2 ^ lv.cols → extra e c = extra' e c)
    (hp : ∀ l i, l < 2 ^ lv.lanes + Zk.EXTRA_LANES → i < tPad → lanePad l i = lanePad' l i) :
    rows lv tPad g extra lanePad = rows lv tPad g extra' lanePad' := by
  rw [rows_eq, rows_eq]
  unfold FlockComplete.encodeRowsZk
  have hm : ((Array.range (2 ^ lv.lanes + Zk.EXTRA_LANES)).map fun l => (Array.range tPad).map (lanePad l)) =
      (Array.range (2 ^ lv.lanes + Zk.EXTRA_LANES)).map fun l => (Array.range tPad).map (lanePad' l) :=
    Array.map_congr_left fun l hl => Array.map_congr_left fun i hi =>
      hp l i (Array.mem_range.1 hl) (Array.mem_range.1 hi)
  simp only [hm]
  refine Array.map_congr_left fun q _ => Array.map_congr_left fun l _ => ?_
  by_cases h : l < 2 ^ lv.lanes
  · simp only [h, ↓reduceIte]
  · simp only [h, ↓reduceIte]
    refine congrArg (· + _) ?_
    exact sumF_congr fun c hc => by simp only [he _ c hc]

end Flock.ZkServer.Link

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open CoinLayer

theorem swExeOf_inl {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) (j : Fin J)
    (p : Fin ((C j).z.nS + (C j).z.nU)) :
    swExeOf C cz wt v ω y ⟨j, .inl p⟩ = exeOf C cz wt ω y ⟨j, .inl p⟩ := by
  show _root_.Flock.Hm96.Default512.leaf (((C j).T (cz j)).swContent (wt j) (v j) (ω j) (.inl p)) _ =
    _root_.Flock.Hm96.Default512.leaf (((C j).T (cz j)).content (wt j) (ω j) (.inl p)) _
  rw [ZkView.Coined.swContent_inl]

namespace Masked

theorem leafAt_swCutAt (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (tabC P x j).S.m) (v : ∀ j, ((tabC P x j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (lv : ZkView.Leaves (fun j => (tabC P x j).z) → ByteArray) (j : Fin (jOf x)) {q : ℕ}
    (hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU) :
    leafAt P x cz (swCutAt (tabC P x) cz wt v ω lv) j q = lv ⟨j, .inl ⟨q, hq⟩⟩ := by
  unfold leafAt
  rw [dite_eq_left_of_eq_true (eq_true hq)]
  show (match ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with
    | .inl a => lv ⟨j, .inl (((tabC P x j).T (cz j)).s0 (.inl a))⟩
    | .inr u => lv ⟨j, .inl (((tabC P x j).T (cz j)).s0 (.inr u))⟩) = _
  rcases h : ((tabC P x j).T (cz j)).s0.symm ⟨q, hq⟩ with a | u
  · have e : ((tabC P x j).T (cz j)).s0 (.inl a) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]
  · have e : ((tabC P x j).T (cz j)).s0 (.inr u) = ⟨q, hq⟩ := by rw [← h, Equiv.apply_symm_apply]
    simp only [e]

/-- **The deployed session's table data**: the plan `czExe`, `encExe` stopped where a refusal is seen, `finExe`, at
`PE` and the executable's tape plan `UE` (`Flock.zk_deployZk_masked_exe_split`'s). -/
noncomputable abbrev dataE : TableData :=
  tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE

/-- **What the ZK server hands `Flock.ZkServer.Link` for table `j`** at the tape `t`: the table's `m`, its packed table
at the witness's traces with the tape's mask words, and the tape's extra lanes, lane paddings and level-0 salts at the
sizes level 0 of the table's `--zk` schedule takes (`Level0.sampleSize`): each extra lane's first `2^cols` words, each of
the `2^lanes + 4` lanes' first `t_pad` padding coefficients and the salts of the `2^d` tree positions, and `0` or empty
past them, as `Level0.toIn` reads a level 0 sampled at those sizes. -/
noncomputable def linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (j : Fin (jOf x)) :
    _root_.Flock.ZkServer.Link.In where
  wideLeaf := (stAt x j).wideLeaf
  m := (stAt x j).m
  g := (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
    (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask)))[j.val]!
  extra e c := if c < 2 ^ (lv0 x j).cols then (exeTape t).extra j.val e c else 0
  lanePad l i := if l < 2 ^ (lv0 x j).lanes + _root_.Flock.Zk.EXTRA_LANES ∧ i < (zkTab x j).tPad then
    (exeTape t).lanePad j.val l i else 0
  salt q := if q < 2 ^ (lv0 x j).d then (exeTape t).toTape.salt j.val 0 0 q else ByteArray.empty

theorem rows_linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (j : Fin (jOf x)) :
    _root_.Flock.ZkServer.Link.rows (lv0 x j) (zkTab x j).tPad (linkIn x w t j).g (linkIn x w t j).extra
        (linkIn x w t j).lanePad =
      _root_.Flock.ZkServer.Link.rows (lv0 x j) (zkTab x j).tPad (linkIn x w t j).g ((exeTape t).extra j.val)
        ((exeTape t).lanePad j.val) :=
  _root_.Flock.ZkServer.Link.rows_congr _ _ _ (fun _ _ hc => if_pos hc) fun _ _ hl hi => if_pos ⟨hl, hi⟩

theorem salt_linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (j : Fin (jOf x)) {q : ℕ}
    (hq : q < 2 ^ (lv0 x j).d) : (linkIn x w t j).salt q = (exeTape t).toTape.salt j.val 0 0 q :=
  if_pos hq

noncomputable def linkMsg (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) : Msg :=
  if (FlockDeploy.setupOf true x).isNone then [] else
  0 :: FlockDeploy.linkBody (List.ofFn fun j => _root_.Flock.ZkServer.Link.root (linkIn x w t j))
    (List.replicate (jOf x) ByteArray.empty)

theorem root_linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals) (j : Fin (jOf x)) :
    _root_.Flock.ZkServer.Link.root (linkIn x w t j) =
      FlockComplete.flat (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x j).d
        (leafAt PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
          (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) j)
        ((lv0 x j).d - (lv0 x j).cap)) := by
  obtain ⟨-, -, -, -, hd, -⟩ := tabC_z PE x j
  have hl : ∀ q, q < 2 ^ (lv0 x j).d →
      leafAt PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
        (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) j q =
      _root_.Flock.ZkServer.Link.leaf ((_root_.Flock.ZkServer.Link.rows (lv0 x j) (zkTab x j).tPad
        (linkIn x w t j).g (linkIn x w t j).extra (linkIn x w t j).lanePad)[q]!) ((linkIn x w t j).salt q) :=
    fun q hq => by
      have hq' : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU := lt_of_lt_of_eq hq hd.symm
      refine (leafAt_swCutAt PE x cz (wtOf PE x w) v (UE.tape x t).1
        (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1) j hq').trans
        ((swExeOf_inl (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1 j ⟨q, hq'⟩).trans
        ((exeOf_inl PE (TapeLayout.exe PE) exeTape (agrees0_exe PE) w t cz j ⟨q, hq'⟩).trans ?_))
      rw [rows_linkIn, salt_linkIn x w t j hq, _root_.Flock.ZkServer.Link.rows_eq] <;> rfl
  rw [_root_.Flock.ZkServer.Link.root_eq (linkIn x w t j) (zkTab x j).hz]
  show _root_.Flock.ZkServer.Link.flat (_root_.Flock.ZkServer.Link.cap (lv0 x j) (zkTab x j).tPad (linkIn x w t j)) = _
  rw [_root_.Flock.ZkServer.Link.flat_eq, _root_.Flock.ZkServer.Link.cap_eq (lv0 x j) (zkTab x j).tPad _ _ hl] <;> rfl

theorem enc0_swCutAt (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals) :
    enc0 PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
      (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) = linkMsg x w t := by
  unfold enc0 linkMsg
  simp only [root_linkIn x w t cz v] <;> rfl

/-- Round `0` sees no refusal, so a stopped plan's round `0` is the plan's. -/
theorem encStop_zero {C : Tabs} {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (C x j).Rest}
    (enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg)
    (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (cut : ∀ j, ((C x j).T (cz x c j)).tab.Cut) :
    encStop enc x c 0 cut = enc x c 0 cut :=
  ite_eq_right (refused_zero c)

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace LeanRelease

/-- A named hypothesis, one of the transport's trusts, narrower than `LinkRoundEq`: what `ZkServer::new`
hands the Lean code and how `ZkServer::root` frames the roots it answers, never how a root is computed. -/
def LinkIs (nr : Msg → ℕ → ℕ)
    (L : LeanRelease Claims.Flock.Wit ((session nr dataE).refP.Nonce × (session nr dataE).refP.Rand)) (x : Msg)
    (w : Claims.Flock.Wit) : Prop :=
  ∀ (ν : (session nr dataE).Nonce) (t : (session nr dataE).Tape) (m₀ ρ₀ : Msg), 0 < refK x →
    L.refusesCoins x (ν, t) [(m₀, ρ₀)] = false → L.link x w (ν, t) [(m₀, ρ₀)] = linkMsg x w t

/-- **`LinkRoundEq` at the deployed session from the Lean recheck** (`LinkIs`), whatever values the workers
supply. -/
theorem linkRoundEq_of_linkIs (nr : Msg → ℕ → ℕ) (vs : (session nr dataE).SwVals)
    (L : LeanRelease Claims.Flock.Wit ((session nr dataE).refP.Nonce × (session nr dataE).refP.Rand)) (x : Msg)
    (w : Claims.Flock.Wit) (hL : LinkIs nr L x w) : LinkRoundEq (session nr dataE) vs L x w := by
  intro ν t m₀ ρ₀ hk hr
  have e := enc0_swCutAt x w t (czExe PE x ((session nr dataE).pad [] : Fin (refK x) → FlockDeploy.Coin))
    (vs x w ν ((session nr dataE).pad [] : Fin (refK x) → FlockDeploy.Coin) ((session nr dataE).seenOf x t))
  rw [hL ν t m₀ ρ₀ hk hr, ← e]
  unfold RefSession.swMsgAt
  rw [List.length_nil, if_pos hk]
  first
  | exact ((encStop_zero (encExe PE) x _ _).trans (encExe_zero PE x _ _)).symm
  | (show _ = encStop (encExe PE) x ((session nr dataE).pad []) 0 _
     rw [encStop_zero, encExe_zero])

end LeanRelease

end Claims.ZKServer
