import Proofs.Flock.Claims.ZK.Refine
import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.Switched
import Proofs.Flock.Claims.ZK.LinkFft

/-!
# The link round at the Lean recheck

`Flock.Firewall.Link.root` is the code of record of the firewall's level-0 recheck: each table's
level-0 root, committed again from the host's witness, the table's level-0 hiding and its level-0 salts: the root of
record, which the session's link carries and `Firewall::new` holds the device's root to (live `PROTOCOL.md` §10.2; Rust
commits nothing of level 0 that leaves). Here that function is held to the session's link.

* `Flock.Firewall.Link.root_eq`, `cap_eq`, `leaves_getElem!`, `layer_eq`: the recheck's root is
  `FlockComplete.flat` of `FlockComplete.layerOf`'s cap over `hm96-sha512/v1`'s leaves of `encodeRowsZk`'s rows under the
  salts (`rows_eq`, `LinkFft`: the FFT's rows are those rows); its tasks (`parRange`) split one map (`Link.parRange_eq`).
* `Coined.swContent_inl`, `swExeOf_inl`, `Masked.leafAt_swCutAt`: the switched prover's level-0 leaves are the honest
  prover's, whatever values the workers supply: level 0 reads neither the values nor the moved pads.
* `Masked.linkIn`, `Masked.linkMsg`: what the firewall hands `Flock.Firewall.Link` for table `j` at the tape (the packed
  table of the witness's traces at the tape's mask words, and the tape's extra lanes, lane paddings and level-0 salts),
  and the link message of the roots it answers. That message is round `0`'s encoding of the switched cut, at every
  coin vector and any values the workers supply (`Masked.enc0_swCutAt`, from `exeOf_inl` at `agrees0_exe`).
* `LeanRelease.LinkIs`: the firewall's link is `linkMsg`, a named trust narrower than `LinkRoundEq`. It is a fact about
  what the transport hands `Flock.Firewall.Link` and how it frames the roots, not about how a root is computed.
* **`LeanRelease.linkRoundEq_of_linkIs`**: `LinkRoundEq` at the deployed session (`Masked.dataE`) from `LinkIs`.
-/

namespace Flock.Firewall.Link

theorem flat_eq : flat = FlockComplete.flat := rfl

/-- **The recheck's tree is `layerOf`'s** with SHA-512 nodes. -/
theorem layer_eq (d : Nat) (lf : Nat → ByteArray) :
    ∀ h, layer d lf h = FlockComplete.layerOf (fun l r => Sha512.hash (l ++ r)) d lf h
  | 0 => rfl
  | h + 1 => by rw [layer, FlockComplete.layerOf, layer_eq d lf h]

/-- `layerOf` reads its leaves below `2^d` only. -/
theorem layerOf_congr (nd : ByteArray → ByteArray → ByteArray) (d : Nat) {lf lf' : Nat → ByteArray}
    (hl : ∀ q, q < 2 ^ d → lf q = lf' q) : ∀ h, FlockComplete.layerOf nd d lf h = FlockComplete.layerOf nd d lf' h
  | 0 => by
    simp only [FlockComplete.layerOf]
    exact Array.map_congr_left fun q hq => hl q (Array.mem_range.1 hq)
  | h + 1 => by rw [FlockComplete.layerOf, FlockComplete.layerOf, layerOf_congr nd d hl h]

/-- **Leaf `q` of the recheck** is row `q`'s `hm96-sha512/v1` leaf under salt `q`. -/
theorem leaves_getElem! (lv : Level) (tPad : Nat) (i : In) {q : Nat} (hq : q < 2 ^ lv.d) :
    (leaves lv tPad i)[q]! = leaf ((rows lv tPad i.g i.extra i.lanePad)[q]!) (i.salt q) := by
  have hk : 2 ^ (lv.d - min lv.d CHUNK) * 2 ^ min lv.d CHUNK = 2 ^ lv.d := by
    rw [← Nat.pow_add, Nat.sub_add_cancel (Nat.min_le_left _ _)]
  simp only [leaves, parRange_eq, hk]
  rw [FlockSoundness.Refine.getElem!_range_map _ hq]

/-- **The recheck's cap** is `layerOf`'s over any leaves that are its leaves below `2^d`. -/
theorem cap_eq (lv : Level) (tPad : Nat) (i : In) (lf : Nat → ByteArray)
    (hl : ∀ q, q < 2 ^ lv.d → lf q = leaf ((rows lv tPad i.g i.extra i.lanePad)[q]!) (i.salt q)) :
    cap lv tPad i = FlockComplete.layerOf (fun l r => Sha512.hash (l ++ r)) lv.d lf (lv.d - lv.cap) := by
  show layer lv.d (fun q => (leaves lv tPad i)[q]!) (lv.d - lv.cap) = _
  rw [layer_eq]
  exact layerOf_congr _ _ (fun q hq => (leaves_getElem! lv tPad i hq).trans (hl q hq).symm) _

/-- **The recheck's root** at a table whose `--zk` schedule is `s`: the flat cap at its level 0. -/
theorem root_eq (i : In) {s : _root_.Flock.Schedule} {tPad : Nat} (hz : Zk.schedule i.m = .ok (s, tPad)) :
    root i = flat (cap (s.levels.getD 0 default) tPad i) := by
  unfold root
  rw [hz]

end Flock.Firewall.Link

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open CoinLayer

/-- **At a level-0 position the switched prover's leaf commits the honest prover's column**: level 0 reads the mask
slot's lanes and the lanes' paddings, which the values and the moved pads leave as they are. -/
theorem _root_.FlockSoundness.Discharge.ZkView.Coined.swContent_inl {K W Hi VL : Type} [Field K]
    {z : ZkView.Sizes} (T : ZkView.Coined K W Hi VL z) (w : W) (v : T.tab.Vals) (b : ZkView.Base z Hi)
    (p : Fin (z.nS + z.nU)) : T.swContent w v b (.inl p) = T.content w b (.inl p) := by
  unfold ZkView.Coined.swContent
  by_cases h : T.tab.onVals.RowsAt (w, v) b.1 b.2.1.2.1
  · rw [if_pos h] <;> rfl
  · rw [if_neg h] <;> rfl

/-- **The executable's level-0 leaves at the switched prover are the honest prover's.** -/
theorem swExeOf_inl {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (y : ZkView.Leaves (fun j => (C j).z) → ZkView.Salt) (j : Fin J)
    (p : Fin ((C j).z.nS + (C j).z.nU)) :
    swExeOf C cz wt v ω y ⟨j, .inl p⟩ = exeOf C cz wt ω y ⟨j, .inl p⟩ := by
  show _root_.Flock.Hm96.Default512.leaf (((C j).T (cz j)).swContent (wt j) (v j) (ω j) (.inl p)) _ =
    _root_.Flock.Hm96.Default512.leaf (((C j).T (cz j)).content (wt j) (ω j) (.inl p)) _
  rw [ZkView.Coined.swContent_inl]

namespace Masked

/-- **A switched cut's level-0 leaf at tree position `q` is the leaf at `q`**, whatever the coins' tree
(`leafAt_cutAt` at the switched prover). -/
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

/-- **What the firewall hands `Flock.Firewall.Link` for table `j`** at the tape `t`: the table's `m`, its packed table
at the witness's traces with the tape's mask words, and the tape's extra lanes, lane paddings and level-0 salts. -/
noncomputable def linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (j : Fin (jOf x)) :
    _root_.Flock.Firewall.Link.In where
  m := (stAt x j).m
  g := (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
    (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask)))[j.val]!
  extra := (exeTape t).extra j.val
  lanePad := (exeTape t).lanePad j.val
  salt := (exeTape t).toTape.salt j.val 0 0

/-- **The link message of the recheck's roots**: `Flock.Firewall.Link`'s root of each table, framed as `enc0` frames its
caps; the empty message where the statement sets up no session. -/
noncomputable def linkMsg (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) : Msg :=
  if (FlockDeploy.setupOf true x).isNone then [] else
  0 :: FlockDeploy.linkBody (List.ofFn fun j => _root_.Flock.Firewall.Link.root (linkIn x w t j))
    (List.replicate (jOf x) ByteArray.empty)

/-- **Table `j`'s recheck root is its level-0 cap at the switched cut**, at every coin vector and any values the workers
supply: each leaf below `2^d` is the executable's leaf of `encodeRowsZk`'s row under the tape's salt (`exeOf_inl`),
which the switched prover's is (`swExeOf_inl`). -/
theorem root_linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals) (j : Fin (jOf x)) :
    _root_.Flock.Firewall.Link.root (linkIn x w t j) =
      FlockComplete.flat (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x j).d
        (leafAt PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
          (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) j)
        ((lv0 x j).d - (lv0 x j).cap)) := by
  obtain ⟨-, -, -, -, hd, -⟩ := tabC_z PE x j
  have hl : ∀ q, q < 2 ^ (lv0 x j).d →
      leafAt PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
        (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) j q =
      _root_.Flock.Firewall.Link.leaf ((_root_.Flock.Firewall.Link.rows (lv0 x j) (zkTab x j).tPad
        (linkIn x w t j).g (linkIn x w t j).extra (linkIn x w t j).lanePad)[q]!) ((linkIn x w t j).salt q) :=
    fun q hq => by
      have hq' : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU := lt_of_lt_of_eq hq hd.symm
      refine (leafAt_swCutAt PE x cz (wtOf PE x w) v (UE.tape x t).1
        (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1) j hq').trans
        ((swExeOf_inl (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1 j ⟨q, hq'⟩).trans
        ((exeOf_inl PE (TapeLayout.exe PE) exeTape (agrees0_exe PE) w t cz j ⟨q, hq'⟩).trans ?_))
      rw [_root_.Flock.Firewall.Link.rows_eq] <;> rfl
  rw [_root_.Flock.Firewall.Link.root_eq (linkIn x w t j) (zkTab x j).hz]
  show _root_.Flock.Firewall.Link.flat (_root_.Flock.Firewall.Link.cap (lv0 x j) (zkTab x j).tPad (linkIn x w t j)) = _
  rw [_root_.Flock.Firewall.Link.flat_eq, _root_.Flock.Firewall.Link.cap_eq (lv0 x j) (zkTab x j).tPad _ _ hl] <;> rfl

/-- **The recheck's link message is round `0`'s encoding of the switched cut**, at every coin vector and any values the
workers supply. -/
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

/-- **The firewall's link is the recheck's link message**: after `Hello`'s answer, where the protocol has coins and the
frame takes the answer's, the link is `linkMsg`, `Flock.Firewall.Link.root`'s roots of the tables the transport hands
it. A named hypothesis, one of the transport's trusts, narrower than `LinkRoundEq`: what `flock-circuit`'s
`Firewall::new` hands the Lean code (the host's packed table, the extra lanes, the lane paddings and the level-0 salts
on its tape, to the session loop's `Level0` or the Lean finish's `table`) and how `Firewall::root` frames the roots it
answers, never how a root is computed. -/
def LinkIs (nr : Msg → ℕ → ℕ)
    (L : LeanRelease Claims.Flock.Wit ((session nr dataE).refP.Nonce × (session nr dataE).refP.Rand)) (x : Msg)
    (w : Claims.Flock.Wit) : Prop :=
  ∀ (ν : (session nr dataE).Nonce) (t : (session nr dataE).Tape) (m₀ ρ₀ : Msg), 0 < refK x →
    L.refusesCoins x (ν, t) [(m₀, ρ₀)] = false → L.link x w (ν, t) [(m₀, ρ₀)] = linkMsg x w t

/-- **`LinkRoundEq` at the deployed session from the Lean recheck**: where the firewall's link is `Link.root`'s
link message (`LinkIs`), it is the session's switched link at the tape, whatever values the workers supply
(`enc0_swCutAt`: round `0` sees no refusal, `encStop_zero`, and is `enc0`, `encExe_zero`). -/
theorem linkRoundEq_of_linkIs (nr : Msg → ℕ → ℕ) (vs : (session nr dataE).SwVals)
    (L : LeanRelease Claims.Flock.Wit ((session nr dataE).refP.Nonce × (session nr dataE).refP.Rand)) (x : Msg)
    (w : Claims.Flock.Wit) (hL : LinkIs nr L x w) : LinkRoundEq (session nr dataE) vs L x w := by
  intro ν t m₀ ρ₀ hk hr
  have e := enc0_swCutAt x w t (czExe PE x ((session nr dataE).pad [] : Fin (refK x) → FlockDeploy.Coin))
    (vs x w ν ((session nr dataE).pad []))
  rw [hL ν t m₀ ρ₀ hk hr, ← e]
  unfold RefSession.swMsgAt
  rw [List.length_nil, if_pos hk]
  first
  | exact ((encStop_zero (encExe PE) x _ _).trans (encExe_zero PE x _ _)).symm
  | (show _ = encStop (encExe PE) x ((session nr dataE).pad []) 0 _
     rw [encStop_zero, encExe_zero])

end LeanRelease

end Claims.ZKServer
