import Proofs.Flock.Claims.ZK.Round1Enc
import Proofs.Flock.Claims.Complete.ZkTables.Round2

/-!
# Round `2` of the masked session

The masked session's third body is table `0`'s rep `0` outer zerocheck point squeeze (`rOuter`, `m − 13` coins), taken
right after its first squeeze with nothing absorbed in between: the header of `m − 13` coins of kind `2` alone, or the
session stops after the link message where that table's extra claims at the link points throw. It reads no value, no
commitment and no coin of the session beyond round `0`'s (which decides whether the extra claims throw).

* `Masked.liveRun_round2`: round `2` of `liveRun` at a live statement, in those two cases.
* `Masked.enc2`: round `2`'s encoding, that header where the extra claims at round `0`'s link points return and `[]`
  where they throw;
* `Masked.enc2_causal`: it reads round `0`'s coin only (`RoundPlan.Causal` at round `2`);
* `Masked.honest_round2_exe`: round `2`'s `hstep`: the honest body is `enc2`'s, or the session stops within the
  openings and `enc2` is `[]`, at every tape and coin vector;
* `Flock.zk_deployZk_masked_exe_round2_gap`: so `Flock.zk_deployZk_masked_exe_round1_gap` at a plan whose round `2` is
  `enc2` asks `hstep` after the third round only.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem getD0_of_two (ops : List (Msg × Msg)) (hl : ops.length = 2) :
    (ops.map fun p => coinOfD p.2).getD 0 [] = coinOfD ops[0]!.2 := by
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.1 hl
  rfl

/-- **Table `0`'s size at a live statement** is at least `14`, so its outer zerocheck point has `m − 13 ≠ 0` coins. -/
theorem tabs0_m {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    14 ≤ (FlockDeploy.tabsOf (stsOf x))[0]!.m := by
  obtain ⟨h0, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by unfold jOf; exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hj : 0 < jOf x := by rw [hJ]; exact h0
  have htm : (FlockDeploy.tabsOf (stsOf x))[0]!.m = (stAt x ⟨0, hj⟩).m := by
    rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using h0)]
    simp [FlockDeploy.tabsOf, _root_.Flock.Setup.ofCircuit]
    rfl
  have h25 : 25 ≤ (stAt x ⟨0, hj⟩).m := (hall 0 h0).1
  rw [htm]
  omega

/-- **Round `2` of `liveRun` at a live statement**: either table `0`'s extra claims at the link points throw and the
session stops after the link message, or they return and the session's third body is table `0` rep `0`'s outer
zerocheck point squeeze, the header of `m − 13` coins of kind `2` alone. -/
theorem liveRun_round2 (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) {x : Msg} (hx : Live x)
    (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)) (hl : ops.length = 2) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    ((∃ e, (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun j => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + j)) =
            .error e) ∧
        (∃ e, (liveRun x w ω ops spec).1 = .error e) ∧ (liveRun x w ω ops spec).2.size ≤ ops.length) ∨
      ((∃ extras, (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun j => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + j)) =
            .ok extras) ∧
        ops.length < (liveRun x w ω ops spec).2.size ∧
        (liveRun x w ω ops spec).2[ops.length]! =
          Claims.toMsg (_root_.Flock.Transcript.framed ⟨#[], 0, ByteArray.empty⟩ 2
            ((FlockDeploy.tabsOf (stsOf x))[0]!.m - 13))) := by
  obtain ⟨h0, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hm := tabs0_m hx hsx
  obtain ⟨schs, hs, -⟩ := schedsPZk_live hx
  have hms : ∀ i < (FlockDeploy.tabsOf (stsOf x)).size,
      (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  obtain ⟨cms, hcm, -, -⟩ := FlockComplete.commitsPZk_caps (FlockDeploy.tabsOf (stsOf x)) schs
    (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
      (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask))) ω hms
  have hc : (FlockComplete.commitsPZk (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) (FlockDeploy.tabsOf (stsOf x))
      schs (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
        (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask))) ω :
      _root_.Flock.V _) = .ok cms := by
    rw [hH]
    exact hcm
  have hrs := repStreams_head spec (by omega)
  have hrun : liveRun x w ω ops spec = (FlockComplete.sessionPZk
      (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
      (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) spec (FlockDeploy.tabsOf (stsOf x))
      (FlockComplete.MaskRank.masksOf (stsOf x))
      (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)) ω).go #[] := by
    rw [hH]
    rfl
  rw [hrun, hl]
  rw [← getD0_of_two ops hl]
  generalize FlockDeploy.tabsOf (stsOf x) = tabs at hs hc hcm hm ⊢
  generalize FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask) = trs at hc hcm ⊢
  cases he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD 0 [])
        (p * spec.mPts + j)) with
  | error e =>
    refine .inl ⟨⟨e, rfl⟩, ?_⟩
    have hg := @FlockComplete.sessionPZk_go_round1_err H (fun k => (ops.map fun p => coinOfD p.2).getD k []) spec
      tabs (FlockComplete.MaskRank.masksOf (stsOf x)) trs ω _ _ hs hc _ _ hrs e
    rw [hg he]
    exact ⟨⟨e, rfl⟩, by simp⟩
  | ok extras =>
    refine .inr ⟨⟨extras, rfl⟩, ?_⟩
    obtain ⟨ic, hi⟩ := FlockComplete.JsonRT.innerCommitP_ok _root_.Flock.Sha512.hash
      (FlockComplete.padsOf ω 0 0 (_root_.Flock.Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0))
      (ω.innerSalt 0 0)
    have hi' : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 (Claims.hOf H))
        (FlockComplete.padsOf ω 0 0 (_root_.Flock.Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0))
        (ω.innerSalt 0 0) : _root_.Flock.V _) = .ok ic := by
      rw [hH]
      exact hi
    have hg := FlockComplete.sessionPZk_go_round2 H (fun k => (ops.map fun p : Msg × Msg => coinOfD p.2).getD k [])
      spec tabs hm (FlockComplete.MaskRank.masksOf (stsOf x)) trs ω hs hc hrs he hi'
    exact hg

/-- **Round `2`'s encoding**: table `0`'s rep `0` outer zerocheck point squeeze, `honestPR`'s third body
(`liveRun_round2`), the header of `m − 13` coins of kind `2` alone; `[]` where table `0`'s extra claims at the link
points of round `0`'s coin throw, or where the statement sets up no session. -/
noncomputable def enc2 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) : Msg :=
  match FlockDeploy.setupOf true x with
  | none => []
  | some (_, spec, _) =>
    if 0 < jOf x then
      match (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun i => slotAt c 0 (p * spec.mPts + i)) with
      | .error _ => []
      | .ok _ => 0 :: Claims.toMsg (_root_.Flock.Transcript.framed ⟨#[], 0, ByteArray.empty⟩ 2
          ((FlockDeploy.tabsOf (stsOf x))[0]!.m - 13))
    else []

/-- **Round `2` reads round `0`'s coin only** (`RoundPlan.Causal` at round `2`). -/
theorem enc2_causal (x : Msg) (c c' : Fin (refK x) → FlockDeploy.Coin)
    (hcc : ∀ k : Fin (refK x), k.val < 1 → c k = c' k) : enc2 x c = enc2 x c' := by
  have hs : ∀ e, slotAt c 0 e = slotAt c' 0 e := slotAt_congr hcc (by norm_num)
  unfold enc2
  simp only [hs]

/-- **Round `2`'s `hstep`**: after two openings, `liveRun` sends past them and `enc2` is its third body, tagged `0`,
or it stops after the link message (table `0`'s extra claims at the link points throw) and `enc2` is `[]`, at every
tape and coin vector. -/
theorem honest_round2_exe (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) {x : Msg} (hx : Live x)
    (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)) (hl : ops.length = 2)
    (hK : ops.length < refK x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    (ops.length < (liveRun x w ω ops spec).2.size ∧
        enc2 x (padOps ops) = 0 :: (liveRun x w ω ops spec).2[ops.length]!) ∨
      ((∃ e, (liveRun x w ω ops spec).1 = .error e) ∧ (liveRun x w ω ops spec).2.size ≤ ops.length ∧
        enc2 x (padOps ops) = []) := by
  obtain ⟨h0, -, -, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by unfold jOf; exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hj : 0 < jOf x := by rw [hJ]; exact h0
  have hpts : ((Array.range spec.points).map fun p => (Array.range spec.mPts).map fun i =>
        slotAt (padOps (x := x) ops) 0 (p * spec.mPts + i)) =
      (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun i =>
        FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + i) := by
    refine Array.ext (by simp) fun p h1 _ => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    refine Array.ext (by simp) fun i h1' _ => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    rw [slotAt_padOps_lt ops (by omega) (by omega)
      (linkPt_lt hx hsx (by simpa using h1) (by simpa using h1')), getD0_of_two ops hl]
  have henc : enc2 x (padOps ops) =
      match (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun i => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + i)) with
      | .error _ => []
      | .ok _ => 0 :: Claims.toMsg (_root_.Flock.Transcript.framed ⟨#[], 0, ByteArray.empty⟩ 2
          ((FlockDeploy.tabsOf (stsOf x))[0]!.m - 13)) := by
    unfold enc2
    simp only [hsx, ite_eq_left hj]
    rw [hpts]
  rcases liveRun_round2 H hH hx w ω ops hl hsx with ⟨⟨e, he⟩, hrun, hn⟩ | ⟨⟨extras, hex⟩, hn, hb⟩
  · exact .inr ⟨hrun, hn, by rw [henc, he]⟩
  · exact .inl ⟨hn, by rw [henc, hex, hb]⟩

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a plan that stops where a
refusal is seen, with rounds `1` and `2` proved**: `Flock.zk_deployZk_masked_exe_round1_gap` at a plan whose round `2`
is `enc2`, its `hstep` there `honest_round2_exe`. Left: `hstep` after the third round, the plan's `Causal`, its round
`0` and the last round where every table admits. -/
theorem Flock.zk_deployZk_masked_exe_round2_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (h1 : ∀ x c cut, enc x c 1 cut = enc1 P x c (cz x c) cut)
    (h2 : ∀ x c cut, enc x c 2 cut = enc2 x c)
    (t : ℕ) (ht : 0 < t) (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz)
    (hp : PointReads cz)
    (hstep : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), 2 < ops.length →
      ops.length < refK x → ¬ Refused x (padOps ops) ops.length →
      ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
        (ops.length < (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2.size ∧
          enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
            0 :: (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2[ops.length]!) ∨
        ((∃ e, (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).1 = .error e) ∧
          (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk τ) ops spec).2.size ≤ ops.length ∧
          enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) = []))
    (hfin : (session nr (tableData P (RoundPlan.stop cz (encStop enc) fin)
      (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).HonestFinOn Live RZk H coinOfD
        fun ω => (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr
        (tableData P (RoundPlan.stop cz (encStop enc) fin) (TapePlan.ofLayout (TapeLayout.exe P) exeTape)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_exe_round1_gap H hH nr P h0 h1 t ht hc ho hr hp
    (fun x hx w hR τ ops hpos hl hs s spec hsx => by
      rcases Nat.lt_or_ge 2 ops.length with h | h
      · exact hstep x hx w hR τ ops h hl hs s spec hsx
      · have hl2 : ops.length = 2 := by omega
        have e : enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
            enc2 x (padOps ops) := by
          rw [hl2]
          exact h2 _ _ _
        rw [e]
        exact honest_round2_exe H hH hx w _ ops hl2 hl hsx)
    hfin

end Claims.ZKMal
