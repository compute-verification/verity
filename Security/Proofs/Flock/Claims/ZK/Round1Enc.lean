import Proofs.Flock.Claims.ZK.Round1Root

/-!
# Round `1` of the masked session: its encoding

The masked session's second body is table `0`'s rep `0` first zerocheck squeeze, of the domain's transcript with the
digest, the level-0 cap and the pads' root absorbed, or it stops after the link message where that table's extra
claims at the link points throw (`Masked.liveRun_round1`). Here:

* `Masked.enc1`: round `1`'s encoding, that body of the cut's level-0 leaves and pads leaves by tree position, at the
  link points of round `0`'s coin (`slotAt c 0`), and `[]` where the extra claims there throw;
* `Masked.enc1_causal`: it reads round `0`'s coin only (`RoundPlan.Causal` at round `1`);
* `Masked.honest_round1_exe`: at the executable's layout, round `1`'s `hstep` (`Masked.hlive_of_sent`): the honest
  body is `enc1`'s, or the session stops within the openings and `enc1` is `[]`, at every coin vector
  (`Masked.cap_exe`, `Masked.innerRoot_exe`, the link points below `2^16`: `Masked.linkPt_lt`);
* `Flock.zk_deployZk_masked_exe_round1_gap`: so `Flock.zk_deployZk_masked_exe_sent_gap` at a plan whose round `1` is
  `enc1` asks `hstep` after the second round only.
-/

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge CoinLayer

/-- **Table `j`'s level-0 cap at the executable's layout is the model's**: `commitsPZk`'s cap of table `j` is
`layerOf`'s of the cut's level-0 leaves by tree position (`leafAt`), at every coin vector (`exeOf_inl`). -/
theorem cap_exe (P : Composed.LigProver) {x : Msg} (hx : Live x) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → ExeCell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) {schs cms}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    (hcm : (FlockComplete.commitsPZk (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockDeploy.tabsOf (stsOf x)) schs (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
        (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).mask)))
      ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) : _root_.Flock.V _) = .ok cms)
    (j : Fin (jOf x)) :
    cms[j.val]!.cap = FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x j).d
      (leafAt P x cz (cutAt (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
        (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) j)
      ((lv0 x j).d - (lv0 x j).cap) := by
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : jOf x = (stsOf x).size := by unfold jOf; exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
  rw [hs] at hs'
  cases hs'
  have hms : ∀ i < (FlockDeploy.tabsOf (stsOf x)).size,
      (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  obtain ⟨cms', hcm', -, hcaps⟩ := FlockComplete.commitsPZk_caps (FlockDeploy.tabsOf (stsOf x)) schs
    (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
      (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).mask)))
    ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) hms
  rw [hcm] at hcm'
  cases hcm'
  have hts : (FlockDeploy.tabsOf (stsOf x)).size = jOf x := by rw [hJ]; simp [FlockDeploy.tabsOf]
  rw [hcaps j.val (by rw [hts]; exact j.isLt), hschs j]
  refine FlockComplete.layerOf_congr _ _ (fun q hq => ?_) _
  obtain ⟨-, -, -, -, hd, -⟩ := tabC_z P x j
  have hq' : q < (tabC P x j).z.nS + (tabC P x j).z.nU := by rw [hd]; exact hq
  rw [leafAt_cutAt P x cz _ _ _ j hq', exeOf_inl P (TapeLayout.exe P) exeTape (agrees0_exe P) w t cz j ⟨q, hq'⟩]
  rfl

/-- **The link points are below `2^16`**: at a live statement, point `p < 2` of the link's `m_pts ≤ 59` coordinates
reads element `p·m_pts + i < 118` of round `0`'s coin. -/
theorem linkPt_lt {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    {p i : ℕ} (hp : p < spec.points) (hi : i < spec.mPts) : p * spec.mPts + i < 65536 := by
  obtain ⟨h0, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨hpts, hmp⟩ :=
    HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hsx)
  have hJ : jOf x = (stsOf x).size := by unfold jOf; exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hj : 0 < jOf x := by rw [hJ]; exact h0
  have hwf := (zkTab x ⟨0, hj⟩).wf.2
  have e : (stAt x ⟨0, hj⟩).mPts = spec.mPts := hmp 0 h0
  have hm : (stAt x ⟨0, hj⟩).m ≤ 35 := (hall 0 h0).2.1
  have hPT : _root_.Flock.PT_LOCAL = 24 := rfl
  have hmP : spec.mPts ≤ 59 := by
    have := hwf.mPts
    have := hwf.hm
    omega
  have hp1 : p ≤ 1 := by omega
  have : p * spec.mPts ≤ spec.mPts := by
    calc p * spec.mPts ≤ 1 * spec.mPts := Nat.mul_le_mul_right _ hp1
      _ = spec.mPts := Nat.one_mul _
  omega

/-- **Round `1`'s encoding**: table `0`'s rep `0` first zerocheck squeeze, `honestPR`'s second body
(`liveRun_round1`), of the domain's transcript with the table's digest, its level-0 cap and its rep `0` pads root
absorbed, each built from the cut's leaves by tree position (`leafAt`, `leafAtP`) with SHA-512 nodes; `[]` where table
`0`'s extra claims at the link points of round `0`'s coin throw, or where the statement sets up no session. -/
noncomputable def enc1 (P : Composed.LigProver) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin)
    (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (cut : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) : Msg :=
  match FlockDeploy.setupOf true x with
  | none => []
  | some (_, spec, _) =>
    if h : 0 < jOf x then
      match (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun i => slotAt c 0 (p * spec.mPts + i)) with
      | .error _ => []
      | .ok _ => 0 :: Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.ZcZ.t₁
          (FlockDeploy.tabsOf (stsOf x))[0]!.digest
          (FlockComplete.flat (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x ⟨0, h⟩).d
            (leafAt P x cz cut ⟨0, h⟩) ((lv0 x ⟨0, h⟩).d - (lv0 x ⟨0, h⟩).cap)))
          (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r))
            (Nat.log2 (Composed.padNOf (zkTab x ⟨0, h⟩).table (stAt x ⟨0, h⟩))) (leafAtP P x cz cut ⟨0, h⟩ 0)
            (Nat.log2 (Composed.padNOf (zkTab x ⟨0, h⟩).table (stAt x ⟨0, h⟩))))[0]!
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6)
    else []

/-- **Round `1` reads round `0`'s coin only** (`RoundPlan.Causal` at round `1`): its encoding of the cut at any coins,
at the simulator's leaves `H(dig ⊕ b, c)`, is the same at coin vectors that agree at round `0`. -/
theorem enc1_causal (P : Composed.LigProver) (x : Msg) (c c' : Fin (refK x) → FlockDeploy.Coin)
    (hcc : ∀ k : Fin (refK x), k.val < 1 → c k = c' k) (cz cz' : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (wt : ∀ j, Model.Witness (tabC P x j).S.m) (ω : ∀ j, ZkView.Base (tabC P x j).z (tabC P x j).Hi)
    (p : ZkView.Leaves (fun j => (tabC P x j).z) → ZkView.Bits × ByteArray) :
    enc1 P x c cz (cutAt (tabC P x) cz wt ω fun l => ZkView.leafH (digOf (tabC P x) cz wt ω l + (p l).1, (p l).2)) =
      enc1 P x c' cz' (cutAt (tabC P x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz' wt ω l + (p l).1, (p l).2)) := by
  have hs : ∀ e, slotAt c 0 e = slotAt c' 0 e := slotAt_congr hcc (by norm_num)
  have h : ∀ j, leafAt P x cz (cutAt (tabC P x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz wt ω l + (p l).1, (p l).2)) j =
      leafAt P x cz' (cutAt (tabC P x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz' wt ω l + (p l).1, (p l).2)) j := fun j => funext fun q => by
    by_cases hq : q < (tabC P x j).z.nS + (tabC P x j).z.nU
    · rw [leafAt_cutAt P x cz wt ω _ j hq, leafAt_cutAt P x cz' wt ω _ j hq, digOf_inl P x cz cz' wt ω j]
    · simp only [leafAt, hq, ↓reduceDIte]
  have hP : ∀ j r, leafAtP P x cz (cutAt (tabC P x) cz wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz wt ω l + (p l).1, (p l).2)) j r =
      leafAtP P x cz' (cutAt (tabC P x) cz' wt ω fun l =>
        ZkView.leafH (digOf (tabC P x) cz' wt ω l + (p l).1, (p l).2)) j r := fun j r => funext fun q => by
    by_cases hq : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU
    · rw [leafAtP_cutAt P x cz wt ω _ j r hq, leafAtP_cutAt P x cz' wt ω _ j r hq, digOf_pads P x cz cz' wt ω j]
    · simp only [leafAtP, hq, ↓reduceDIte]
  unfold enc1
  simp only [hs, h, hP]

theorem coinOfD_head (ops : List (Msg × Msg)) (hl : ops.length = 1) {i : ℕ} :
    FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD 0 []) i = FlockComplete.coinOf (coinOfD ops[0]!.2) i := by
  rw [List.getD_eq_getElem _ _ (by simp [hl]), List.getElem_map, getElem!_pos ops 0 (by omega)]

/-- **Round `1`'s `hstep` at the executable's layout**: after one opening, `liveRun` sends past it and `enc1` of the
cut view is its second body, tagged `0`, or it stops after the link message (table `0`'s extra claims at the link
points throw) and `enc1` is `[]`, at every coin vector. -/
theorem honest_round1_exe (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) (P : Composed.LigProver)
    {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (ops : List (Msg × Msg)) (hl : ops.length = 1)
    (hK : ops.length < refK x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    (ops.length < (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) ops spec).2.size ∧
        enc1 P x (padOps ops) cz (cutAt (tabC P x) cz (wtOf P x w)
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
            ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) =
          0 :: (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) ops spec).2[ops.length]!) ∨
      ((∃ e, (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) ops spec).1 = .error e) ∧
        (liveRun x w ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) ops spec).2.size ≤ ops.length ∧
        enc1 P x (padOps ops) cz (cutAt (tabC P x) cz (wtOf P x w)
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
            ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) = []) := by
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
      (linkPt_lt hx hsx (by simpa using h1) (by simpa using h1')), coinOfD_head ops hl]
  obtain ⟨schs, cms, hs, hcm, hcase⟩ := liveRun_round1 H hH hx w _ ops hl hsx
  have henc : enc1 P x (padOps ops) cz (cutAt (tabC P x) cz (wtOf P x w)
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
        (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) =
      match (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun i => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + i)) with
      | .error _ => []
      | .ok _ => 0 :: Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.ZcZ.t₁
          (FlockDeploy.tabsOf (stsOf x))[0]!.digest
          (FlockComplete.flat (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lv0 x ⟨0, hj⟩).d
            (leafAt P x cz (cutAt (tabC P x) cz (wtOf P x w)
              ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
              (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
                ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) ⟨0, hj⟩)
            ((lv0 x ⟨0, hj⟩).d - (lv0 x ⟨0, hj⟩).cap)))
          (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r))
            (Nat.log2 (Composed.padNOf (zkTab x ⟨0, hj⟩).table (stAt x ⟨0, hj⟩)))
            (leafAtP P x cz (cutAt (tabC P x) cz (wtOf P x w)
              ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
              (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
                ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) ⟨0, hj⟩ 0)
            (Nat.log2 (Composed.padNOf (zkTab x ⟨0, hj⟩).table (stAt x ⟨0, hj⟩))))[0]!
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6) := by
    unfold enc1
    simp only [hsx, dite_eq_left hj]
    rw [hpts]
  rcases hcase with ⟨⟨e, he⟩, hrun, hn⟩ | ⟨extras, ic, hex, hic, hn, hb⟩
  · refine .inr ⟨hrun, hn, ?_⟩
    rw [henc, he]
  · refine .inl ⟨hn, ?_⟩
    have hj' : 0 < (stsOf x).size := h0
    have htm : (FlockDeploy.tabsOf (stsOf x))[0]!.m = (stAt x ⟨0, hj⟩).m := by
      rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using hj')]
      simp [FlockDeploy.tabsOf, _root_.Flock.Setup.ofCircuit]
      rfl
    have htk : (FlockDeploy.tabsOf (stsOf x))[0]!.kLog = (stAt x ⟨0, hj⟩).c.kLog := by
      rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using hj')]
      simp [FlockDeploy.tabsOf, _root_.Flock.Setup.ofCircuit]
      rfl
    obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
    rw [hs] at hs'
    cases hs'
    rw [htm, htk, show schs[0]!.1.k0 = (zkTab x ⟨0, hj⟩).sch.k0 by rw [hschs ⟨0, hj⟩]] at hic
    rw [henc, hex, hb, cap_exe P hx hsx w t cz hs hcm ⟨0, hj⟩, innerRoot_exe P w t cz ⟨0, hj⟩ 0 hic]

end Claims.ZKMal.Masked

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge CoinLayer Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at the executable's layout and a plan that stops where a
refusal is seen, with round `1` proved**: `Flock.zk_deployZk_masked_exe_sent_gap` at a plan whose round `1` is `enc1`,
its `hstep` there `honest_round1_exe`. Left: `hstep` after the second round, the plan's `Causal`, its round `0` and the
last round where every table admits. -/
theorem Flock.zk_deployZk_masked_exe_round1_gap (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash)
    (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    {cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest}
    {enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg}
    {fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg}
    (h0 : ∀ x c cut, enc x c 0 cut = enc0 P x (cz x c) cut) (h1 : ∀ x c cut, enc x c 1 cut = enc1 P x c (cz x c) cut)
    (t : ℕ) (ht : 0 < t) (hc : (RoundPlan.stop cz enc fin).Causal) (ho : OuterReads cz) (hr : CoinReads cz)
    (hp : PointReads cz)
    (hstep : ∀ x, Live x → ∀ w, RZk x w → ∀ τ (ops : List (Msg × Msg)), 1 < ops.length →
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
  Flock.zk_deployZk_masked_exe_sent_gap H hH nr P h0 t ht hc ho hr hp
    (fun x hx w hR τ ops hpos hl hs s spec hsx => by
      rcases Nat.lt_or_ge 1 ops.length with h | h
      · exact hstep x hx w hR τ ops h hl hs s spec hsx
      · have hl1 : ops.length = 1 := by omega
        have e : enc x (padOps ops) ops.length ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) =
            enc1 P x (padOps ops) (cz x (padOps ops)) ((session nr (tableData P (RoundPlan.stop cz enc fin)
              (TapePlan.ofLayout (TapeLayout.exe P) exeTape))).cutOf x (padOps ops) (wtOf P x w) τ) := by
          rw [hl1]
          exact h1 _ _ _
        rw [e]
        exact honest_round1_exe H hH P hx w τ (cz x (padOps ops)) ops hl1 hl hsx)
    hfin

end Claims.ZKMal
