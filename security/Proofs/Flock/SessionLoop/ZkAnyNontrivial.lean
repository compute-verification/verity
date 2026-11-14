module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.SessionLoop.Leak
public import Proofs.Flock.SessionLoop.LeafCap
public import Proofs.Flock.SessionLoop.AnyErrBound
public import Proofs.Flock.SessionLoop.RepSrcSat
public import Proofs.Flock.SessionLoop.FinPiopRet
public import Proofs.Flock.SessionLoop.Concrete
public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Flock.SessionLoop.Concrete.LiveNets
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.ZK.FrameSched
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Claims.Prob

/-!
# `SessionLoopZKAnyDeployed`'s conclusion can fail at the deployed session

`Flock.Guarantees.SessionLoopZKAnyDeployed` concludes `Loop.ZkAnyAt` at the deployed tables `Masked.dataE` and the
session loop's own link and `Finish` (`withFiles (withLink L₀ src) src`): the session loop's release meets
`Claims.ZKOut` within `Loop.anyErr`. That conclusion fails at a level-0 source that refuses at one witness and samples
the tape's level 0 at another (`Loop.leakSrc`): the link carries an empty root at the one and a root that is not empty
at the other, so its length tells the two apart (`Loop.not_zkOut_link`), at the concrete statement, wherever the error
against `Loop.helloRej` sums below `1` over the two.

The statement takes no hiding premise: its error is stated at hm96's hiding gap at the pinned key, whatever that gap is.
That the error is below `1` there rests on the gap being at most `2^-193`, the pinned key's hiding
(`FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96`, the witness's one hypothesis), with the
session's leaves within 64 times the per-table leaf limit (`Loop.nLeaves_dataE_le_max`).

* `Loop.pLog_unP_helloRej`, `Loop.fork_unP_helloRej`: against `helloRej` the unswitched prover sends its `Hello` and its
  first message without a SHA-512 call before `helloRej` rejects, so its collision finder finds nothing.
* `Loop.root_size_pos`: a level 0 the `--zk` schedule samples at `25 ≤ m ≤ 35` has a root that is not empty.
* `Loop.length_linkOf`: the link's length is its tables' roots' sizes past a length of the table count.
* `Loop.length_linkOf_leak_nil`, `Loop.length_linkOf_leak_true`: the link is short at `⟨[]⟩` and long at `⟨[true]⟩`.
* `Loop.anyErr_leak_lt`: at `t = 2` the error at each witness is `24·n_L·δ₁ + 2·3^{-2}`, and the two sum below `1`.
* `Flock.SecurityProofs.SessionLoopZKAnyDeployed_nontrivial`: `ZkAnyAt` fails.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal

/-- **A release of constant fields**, at any tape. -/
def constRel (R : Type) : LeanRelease Flock.Wit R where
  rep _ _ := 0
  lay _ _ := default
  slot _ _ := none
  draws _ _ := .empty
  prog _ _ := {}
  coinsOf _ _ := #[]
  fresh _ _ := .error ""
  device _ _ _ _ _ := []
  computed _ _ _ _ := default
  cross _ _ _ _ _ _ := false
  scheduled _ _ _ _ _ := []
  coins _ _ := 0
  sched _ _ := 0
  stop := []
  hello _ _ := []
  helloChk _ _ := .error ""
  req _ _ := ("", 0)
  link _ _ _ _ := []
  files _ _ _ _ := ("", #[])
  finStops _ _ _ _ := false
  last _ := 0

/-- **A level-0 source that refuses at the witness without hidden values**, and samples the tape's level 0 at every
other witness (`tapeSrcHi`). -/
noncomputable def leakSrc (x : Msg) (w : Flock.Wit) (ρ : Tp) (k : ℕ) :
    _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray) :=
  if w.hid = [] then .error "no hidden values" else tapeSrcHi x w ρ k

/-- A worker that never answers, at the session loop's release on `leakSrc`. -/
def leakWk : (LeanRelease.ofLoopLink (constRel Tp) leakSrc (realOf leakSrc)).toSrv.Worker := fun _ _ _ _ _ => none

/-- **The unswitched prover makes no SHA-512 call against `helloRej`**: it sends its `Hello`, then its first message
on no opening, and `helloRej` rejects the second. -/
theorem pLog_unP_helloRej (S : RefSession) (vs : S.SwVals) (w : Flock.Wit) (x : Msg)
    (ω : (S.unP vs).Nonce × (S.unP vs).Rand) :
    (runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x ω ()).pLog = [] := by
  simp [runStar, helloRej, stepWith, Proto.prover, RefSession.unP, strat, allOk, ocBind, OC.eval]

/-- **The unswitched prover's collision finder against `helloRej` finds nothing.** -/
theorem fork_unP_helloRej (S : RefSession) (vs : S.SwVals) (w : Flock.Wit) (x : Msg) :
    (Fork CoinOpen.shaH ((S.unP vs).prover w) helloRej () x).Pr CoinOpen.shaH = 0 := by
  have h : ∀ s, (Fork CoinOpen.shaH ((S.unP vs).prover w) helloRej () x).out s = none := fun s => by
    show findColl ((runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x (s.1, s.2.1) ()).pLog ++
      (runStar CoinOpen.shaH ((S.unP vs).prover w) helloRej x (s.1, s.2.2) ()).pLog) = none
    rw [pLog_unP_helloRej, pLog_unP_helloRej]
    rfl
  unfold Finder.Pr
  rw [Claims.Pr_congr (q := fun _ => False) fun s => by simp [h s]]
  exact Claims.Pr_false

/-- **A Merkle layer above the leaves flattens to bytes**: `d − c ≥ 1` layers up, it is `2^c` digests. -/
theorem flat_layer_pos (d c : ℕ) (lf : ℕ → ByteArray) (hc : c < d) :
    0 < (_root_.Flock.Firewall.Link.flat (_root_.Flock.Firewall.Link.layer d lf (d - c))).size := by
  obtain ⟨k, hk⟩ : ∃ k, d - c = k + 1 := ⟨d - c - 1, by omega⟩
  have hmem : ∀ b ∈ _root_.Flock.Firewall.Link.layer d lf (d - c), b.size = 64 := by
    intro b hb
    rw [hk] at hb
    simp only [_root_.Flock.Firewall.Link.layer, Array.mem_map] at hb
    obtain ⟨q, -, rfl⟩ := hb
    exact _root_.Flock.Sha512.hash_size _
  have hsz : (_root_.Flock.Firewall.Link.layer d lf (d - c)).size = 2 ^ c := by
    rw [Sched.layer_size, Nat.pow_div (Nat.sub_le d c) (by norm_num)]
    congr 1
    omega
  have hf : (_root_.Flock.Firewall.Link.flat (_root_.Flock.Firewall.Link.layer d lf (d - c))).size =
      64 * (_root_.Flock.Firewall.Link.layer d lf (d - c)).size :=
    Sched.flat_size _ hmem
  rw [hf, hsz]
  positivity

/-- **A level 0 the `--zk` schedule samples has a root**: at `25 ≤ m ≤ 35` the level's root sits strictly below its
depth, so the root holds at least one digest. -/
theorem root_size_pos {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35) {g : Array _root_.Flock.F128} {d : ByteArray}
    {l : _root_.Flock.Firewall.Level0.Level0} (h : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l) :
    0 < l.root.size := by
  obtain ⟨hm, -, ⟨s, hz, hlv⟩, -, -, hroot⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok h
  have hz' : _root_.Flock.Zk.schedule l.m = .ok (s, l.tPad) := by rw [hm]; exact hz
  have hlt : l.lv.cap < l.lv.d := by
    rw [hlv]
    exact FlockComplete.JsonRT.zkSchedule_cap0_lt h25 h35 hz
  rw [hroot, _root_.Flock.Firewall.Level0.root_of l s hz' hlv]
  exact flat_layer_pos l.lv.d l.lv.cap (fun q => (_root_.Flock.Firewall.Link.leaves l.lv l.tPad l.toIn)[q]!) hlt

/-- A level 0 the loop samples is the one built at its table's `m` from what the source gave. -/
theorem built_of_level0Of {R : Type}
    {src : Msg → Flock.Wit → R → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {sts : Array _root_.Flock.Stmt} {j : ℕ} {l : _root_.Flock.Firewall.Level0.Level0}
    (h : level0Of src x w ρ sts j = .ok l) : ∃ g d, _root_.Flock.Firewall.Level0.ofDraws sts[j]!.m g d = .ok l := by
  unfold level0Of at h
  cases hsrc : src x w ρ (2 * j) with
  | error e => rw [hsrc] at h; cases h
  | ok v =>
    obtain ⟨g, d, hi⟩ := v
    rw [hsrc] at h
    exact ⟨g, d, h⟩

/-- **The link's length**: a tag, two list lengths, each table's root after its length, and each table's empty public
words after theirs. -/
theorem length_linkOf {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) :
    (linkOf src x w ρ).length = 17 + 16 * sts.size +
      ∑ j : Fin sts.size, (((level0Of src x w ρ sts j.val).toOption.map
        (fun l : _root_.Flock.Firewall.Level0.Level0 => l.root)).getD ByteArray.empty).size := by
  unfold linkOf
  rw [hs]
  simp only [List.length_cons, FlockDeploy.linkBody, FlockDeploy.encList, FlockDeploy.encBytes, FlockDeploy.encNat,
    List.length_append, List.length_map, List.length_range, List.length_flatMap, List.map_ofFn, List.sum_ofFn,
    List.length_ofFn, List.length_replicate, List.map_replicate, List.sum_replicate, Function.comp_apply,
    Array.length_toList, ByteArray.size_data, ByteArray.size_empty, Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  ring

/-- `leakSrc` refuses at the witness without hidden values. -/
theorem leakSrc_nil (x : Msg) (ρ : Tp) (k : ℕ) : leakSrc x ⟨[]⟩ ρ k = .error "no hidden values" := by
  simp [leakSrc]

/-- `leakSrc` is `tapeSrcHi` at the witness `⟨[true]⟩`. -/
theorem leakSrc_true (x : Msg) (ρ : Tp) (k : ℕ) : leakSrc x ⟨[true]⟩ ρ k = tapeSrcHi x ⟨[true]⟩ ρ k := by
  simp [leakSrc]

theorem level0Of_leakSrc_nil (x : Msg) (ρ : Tp) (sts : Array _root_.Flock.Stmt) (j : ℕ) :
    level0Of leakSrc x ⟨[]⟩ ρ sts j = .error "no hidden values" := by
  unfold level0Of
  rw [leakSrc_nil]
  rfl

theorem level0Of_leakSrc_true (x : Msg) (ρ : Tp) (sts : Array _root_.Flock.Stmt) (j : ℕ) :
    level0Of leakSrc x ⟨[true]⟩ ρ sts j = level0Of tapeSrcHi x ⟨[true]⟩ ρ sts j := by
  unfold level0Of
  rw [leakSrc_true]

/-- **At `⟨[]⟩` the link is short**: every root is empty, so the link is its length fields alone. -/
theorem length_linkOf_leak_nil {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (ρ : Tp) :
    (linkOf leakSrc x ⟨[]⟩ ρ).length ≤ 17 + 16 * sts.size := by
  rw [length_linkOf _ _ _ _ hs, Finset.sum_eq_zero fun j _ => by rw [level0Of_leakSrc_nil]; rfl]
  omega

/-- **A link is long where table `0`'s root is not empty.** -/
theorem length_linkOf_of_root {R : Type}
    {src : Msg → Flock.Wit → R → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} {ρ : R} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (h0 : 0 < sts.size)
    {l : _root_.Flock.Firewall.Level0.Level0} (hl : level0Of src x w ρ sts 0 = .ok l) (hpos : 0 < l.root.size) :
    17 + 16 * sts.size < (linkOf src x w ρ).length := by
  rw [length_linkOf _ _ _ _ hs]
  have hle : (((level0Of src x w ρ sts 0).toOption.map
        (fun l : _root_.Flock.Firewall.Level0.Level0 => l.root)).getD ByteArray.empty).size ≤
      ∑ j : Fin sts.size, (((level0Of src x w ρ sts j.val).toOption.map
          (fun l : _root_.Flock.Firewall.Level0.Level0 => l.root)).getD ByteArray.empty).size :=
    Finset.single_le_sum (f := fun j : Fin sts.size => (((level0Of src x w ρ sts j.val).toOption.map
        (fun l : _root_.Flock.Firewall.Level0.Level0 => l.root)).getD ByteArray.empty).size)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ ⟨0, h0⟩)
  rw [hl] at hle
  change l.root.size ≤ _ at hle
  omega

/-- **At `⟨[true]⟩` table `0`'s level 0 is sampled**: `leakSrc` is `tapeSrcHi` there (`level0Src_tapeSrcHi`). -/
theorem level0Of_leak_true_ok {x : Msg} {sts : Array _root_.Flock.Stmt} (hsts : Masked.stsOf x = sts)
    (hj : 0 < Masked.jOf x) (ν : (CoinLayer.session loopNr Masked.dataE).refP.Nonce)
    (t : (CoinLayer.session loopNr Masked.dataE).refP.Rand) :
    ∃ l, level0Of leakSrc x ⟨[true]⟩ (ν, t) sts 0 = .ok l := by
  obtain ⟨l, hl, -⟩ := Level0Src.linkLevel0 (level0Src_tapeSrcHi x ⟨[true]⟩) ν t ⟨0, hj⟩
  subst hsts
  refine ⟨l, ?_⟩
  rw [level0Of_leakSrc_true]
  exact hl

/-- **At `⟨[true]⟩` the link is long**: table `0`'s root is not empty (`root_size_pos`). -/
theorem length_linkOf_leak_true {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (h0 : 0 < sts.size)
    (hm : ∀ j (hj : j < sts.size), 25 ≤ sts[j].m ∧ sts[j].m ≤ 35) (hj : 0 < Masked.jOf x)
    (ν : (CoinLayer.session loopNr Masked.dataE).refP.Nonce) (t : (CoinLayer.session loopNr Masked.dataE).refP.Rand) :
    17 + 16 * sts.size < (linkOf leakSrc x ⟨[true]⟩ (ν, t)).length := by
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  have hm0 : 25 ≤ sts[0]!.m ∧ sts[0]!.m ≤ 35 := by
    rw [getElem!_pos sts 0 h0]
    exact hm 0 h0
  obtain ⟨l, hl⟩ := level0Of_leak_true_ok hsts hj ν t
  obtain ⟨g, d, hd⟩ := built_of_level0Of hl
  exact length_linkOf_of_root hs h0 hl (root_size_pos hm0.1 hm0.2 hd)

/-- The concrete statement's tables: at least one, each at `25 ≤ m ≤ 35`. -/
theorem concrete_tables :
    0 < FlockDeploy.concreteSetup.2.size ∧ ∀ j (hj : j < FlockDeploy.concreteSetup.2.size),
      25 ≤ FlockDeploy.concreteSetup.2[j].m ∧ FlockDeploy.concreteSetup.2[j].m ≤ 35 := by
  obtain ⟨h0, -, -, htabs⟩ := FlockComplete.ZkTables.tables_of_setup FlockDeploy.concreteStmt_live.1
    (FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild)
  exact ⟨h0, fun j hj => ⟨(htabs j hj).1, (htabs j hj).2.1⟩⟩

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer Claims.ZKServer.Loop
open scoped ENNReal

/-- **The error at each witness is below `1/2`** at `t = 2` against `helloRej`, where hm96 hides at `2^-193` at the
pinned key: `24·n_L·δ₁ + 2·3^{-2}`, with `n_L` within 64 times the per-table leaf limit (`Loop.nLeaves_dataE_le_max`)
and no collision found. -/
theorem _root_.Claims.ZKServer.Loop.anyErr_leak_lt
    (hKey : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash
      ((2 : ℝ≥0∞) ^ (-193 : ℤ))) (w₁ w₂ : Flock.Wit) :
    Loop.anyErr Masked.dataE Loop.finS.honestVals
        (LeanRelease.withFiles (LeanRelease.withLink (Loop.constRel Loop.Tp) Loop.leakSrc) Loop.leakSrc)
        (Loop.realOf Loop.leakSrc) 2 FlockDeploy.concreteStmt w₁ Unit Loop.helloRej () false +
      Loop.anyErr Masked.dataE Loop.finS.honestVals
        (LeanRelease.withFiles (LeanRelease.withLink (Loop.constRel Loop.Tp) Loop.leakSrc) Loop.leakSrc)
        (Loop.realOf Loop.leakSrc) 2 FlockDeploy.concreteStmt w₂ Unit Loop.helloRej () false < 1 := by
  have _ : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96 := hKey
  have hL : ((CoinLayer.session Loop.loopNr Masked.dataE).nLeaves FlockDeploy.concreteStmt : ℝ) ≤
      64 * (2 ^ 23 + 2 ^ 14 + 2) := by
    have := Loop.nLeaves_dataE_le_max FlockDeploy.concreteStmt
    simp only [_root_.Flock.Tags.MAX_TABLES, FlockSoundness.Discharge.ZkHidden.leafCap] at this
    exact_mod_cast this
  have hδ : FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
      FlockSoundness.Discharge.ZkView.saltHash ≤ (2 : ℝ) ^ (-193 : ℤ) := by
    have h2 : ((2 : ℝ≥0∞) ^ (-193 : ℤ)).toReal = (2 : ℝ) ^ (-193 : ℤ) := by
      rw [← ENNReal.coe_ofNat, ← ENNReal.coe_zpow (by norm_num), ENNReal.coe_toReal, NNReal.coe_zpow,
        NNReal.coe_ofNat]
    rw [← h2]
    exact FlockSoundness.CROnly.hidingGap_le_of_hiding hKey
      (ENNReal.zpow_ne_top (by norm_num) ENNReal.ofNat_ne_top _)
  have hδ0 := FlockSoundness.CROnly.hidingGap_nonneg FlockSoundness.Discharge.ZkView.My
    FlockSoundness.Discharge.ZkView.saltHash
  have hnδ : ((CoinLayer.session Loop.loopNr Masked.dataE).nLeaves FlockDeploy.concreteStmt : ℝ) *
      FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash ≤
        64 * (2 ^ 23 + 2 ^ 14 + 2) * (2 : ℝ) ^ (-193 : ℤ) :=
    mul_le_mul hL hδ hδ0 (by norm_num)
  have hc : 64 * (2 ^ 23 + 2 ^ 14 + 2) * (2 : ℝ) ^ (-193 : ℤ) < 1 / 100 := by norm_num [zpow_neg]
  have hF : ∀ w, (Fork CoinOpen.shaH
      (((Loop.anyStops Masked.dataE
        (LeanRelease.withFiles (LeanRelease.withLink (Loop.constRel Loop.Tp) Loop.leakSrc) Loop.leakSrc)
        (Loop.realOf Loop.leakSrc)).unP Loop.finS.honestVals).prover w)
      Loop.helloRej () FlockDeploy.concreteStmt).Pr CoinOpen.shaH = 0 :=
    fun w => Loop.fork_unP_helloRej _ _ w _
  simp only [Loop.anyErr, Bool.false_eq_true, ↓reduceIte, Nat.cast_ofNat]
  rw [hF w₁, hF w₂]
  linarith [hnδ.trans hc.le]

/-- **`SessionLoopZKAnyDeployed`'s conclusion fails at the deployed session**: at the deployed tables, the session
loop's own link and `Finish` on the source `leakSrc`, the concrete statement under the relation of every pair and the
witnesses `⟨[]⟩` and `⟨[true]⟩`, `ZkAnyAt` fails at `t = 2`. Against `helloRej` the link carries the empty root at the
first and a root that is not empty at the second, and the error at each is `24·n_L·δ₁ + 2·3^{-2}`, below `1/2` where
hm96 hides at `2^-193` at the pinned key (`HashDerivedKeyHm96`) and the session's leaves are within 64 times the
per-table leaf limit. -/
theorem SessionLoopZKAnyDeployed_nontrivial (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) :
    ¬ Loop.ZkAnyAt Masked.dataE Loop.finS.honestVals
      (LeanRelease.withFiles (LeanRelease.withLink (Loop.constRel Loop.Tp) Loop.leakSrc) Loop.leakSrc)
      (Loop.realOf Loop.leakSrc) Loop.leakWk (fun _ _ => True) 2 := fun h =>
  Loop.not_zkOut_link Masked.dataE
    (LeanRelease.withFiles (LeanRelease.withLink (Loop.constRel Loop.Tp) Loop.leakSrc) Loop.leakSrc)
    (Loop.realOf Loop.leakSrc) Loop.leakWk
    (FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild)
    (refK_pos (FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild))
    (w₁ := ⟨[]⟩) (w₂ := ⟨[true]⟩) (fun m => m.length ≤ 17 + 16 * FlockDeploy.concreteSetup.2.size)
    (fun ρ _ => Loop.length_linkOf_leak_nil
      (FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild) ρ)
    (fun ρ _ => not_le.2 (Loop.length_linkOf_leak_true
      (FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild) Loop.concrete_tables.1
      Loop.concrete_tables.2 (by rw [FlockDeploy.concreteStmt_jOf]; norm_num) ρ.1 ρ.2))
    (fun _ _ => True) trivial trivial _ _ (Loop.anyErr_leak_lt hKey _ _) h.1

end Flock.SecurityProofs
