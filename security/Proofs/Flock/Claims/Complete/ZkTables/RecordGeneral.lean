module

public import Proofs.Flock.Claims.Complete.ZkTables.RecordSetup
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Merkle
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The masked record's length from the setup alone, at public outputs and at hidden ones

`RecordSetup` proves the facts the length bound reads at one table under tags with hidden outputs. Here the other
branch: under tags with public outputs (`Stmt.setupH`), the verifier's public inputs are refused (`setupTables_own_none`),
so the setup is `setupTables_ok`'s; its table has `25 ≤ m ≤ 35` (`Zk.shapeOk`), the `hm96` leaf (`setupH_hm96`),
`k_log ≤ m`, at most `2^10` regions and `m_pts ≤ 64` (`setupH_spec`), a keyed coin tree (`setupH_coinTree`) and a `Hello`
that reads back (`setupOne_stmt`). `setup_facts` is the two together, and `recordZk_size_le`,
`recordFitsZkAt_of_text` take `hs`, `h1` and the text alone.
-/

@[expose] public section

open private setupOne_stmt from Proofs.Flock.Claims.Complete.ZkTables.StmtOne

open _root_.Flock

namespace FlockComplete.ZkTables

open Flock Transcript FlockDeploy With FlockComplete JsonRT FlockSoundness.Discharge FlockComplete.Replay
open Lean (Json)

/-- **No statement with public outputs and the hm96 rows is typed.** -/
theorem tags_public_untyped : ∀ t ∈ Tags.all ++ Tags.templated, t.hm96Rows = true → t.hiddenOutputs = false →
    t.typed = false := by
  decide +kernel

/-- Under `--zk` a statement `Zk.shapeOk` passes is of `25 ≤ m ≤ 35` (`queries0`). -/
theorem m_range_of_shapeOk {st : Flock.Stmt} (h : Zk.shapeOk st = .ok ()) : 25 ≤ st.m ∧ st.m ≤ 35 := by
  refine ⟨Readable.m25_of_shapeOk h, ?_⟩
  unfold Zk.shapeOk at h
  obtain ⟨_, hq, -⟩ := FlockSoundness.except_bind_ok h
  unfold Zk.queries0 at hq
  by_contra hm
  split at hq <;> first | (cases hq; done) | (simp at *; omega)

open _root_.Flock FlockSoundness.Refine in
set_option maxHeartbeats 4000000 in
/-- **A statement `Stmt.setupH` returns keeps the coin tree its setup was given.** -/
theorem setupH_coinTree {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : st.coinTree = coins.tree? := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

open _root_.Flock FlockSoundness.Refine in
set_option maxHeartbeats 4000000 in
/-- **A session with public outputs takes no public inputs of its own** (`Stmt.setupTables` refuses them). -/
theorem setupTables_own_none {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program own : Option ByteArray}
    {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : tags.hiddenOutputs = false)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) : own = none := by
  by_contra hn
  obtain ⟨p, rfl⟩ := Option.ne_none_iff_exists'.1 hn
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp [hh, throw_bind', pure_bind] at h

/-- **The facts the length bound reads, from the setup alone**, at one table under tags with public outputs. -/
theorem setup_facts_public {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) (hh : s.read.tags.hiddenOutputs = false) :
    ∃ (st : Flock.Stmt) (t : String) (ct : CoinTree.Spec) (own : Json) (hz : String), sts = #[st] ∧
      spec.tables = #[t] ∧ spec.coinTree = some ct ∧ Flock.JsonParse.parse spec.hello = .ok own ∧
      canon (JsonRT.helloJ own) = .ok hz ∧ 25 ≤ st.m ∧ st.m ≤ 35 ∧ st.merkle = MerkleScheme.hm96Sha512 ∧
      st.c.kLog ≤ st.m ∧ st.regions.size ≤ 1024 ∧ spec.mPts = st.mPts ∧ st.mPts ≤ 64 ∧ spec.points ≤ 2 := by
  obtain ⟨hi, hset⟩ := setupOf_some hs
  have hr := Ins.setup_read hset
  have hzk : s.read.zk = true := readOf_zk (x := x) (by unfold readOf; rw [hi]; rfl)
  have hm96 := Ins.setup_hm96Rows hset
  have e1 : s.read.tags.hm96Rows = s.read.I.tags.hm96Rows := by
    simp [Readable.Read.tags, ZkExec.zkTags, Zk.tags, hzk]
  have e2 : s.read.tags.merkleLeaf = s.read.I.tags.merkleLeaf := by
    simp [Readable.Read.tags, ZkExec.zkTags, Zk.tags, hzk]
  have e3 : s.read.tags.hiddenOutputs = s.read.I.tags.hiddenOutputs := by
    simp [Readable.Read.tags, ZkExec.zkTags, Zk.tags, hzk]
  have e4 : s.read.tags.typed = s.read.I.tags.typed := by
    simp [Readable.Read.tags, ZkExec.zkTags, Zk.tags, hzk]
  obtain ⟨w, hw, hwz, hwi⟩ := insOf_ins hi
  obtain ⟨t, hbn, hIt⟩ := Raw.ins_tags hwi
  have hl : s.read.tags.merkleLeaf = "hm96-sha512/v1" := by
    rw [e2, hIt]
    exact byName_hm96 hbn (Raw.ins_plain hwi hbn hwz) (by rw [← hIt, ← e1]; exact hm96)
  have hty : s.read.tags.typed = false := by
    rw [e4, hIt]
    exact tags_public_untyped t (mem_of_byName hbn) (by rw [← hIt, ← e1]; exact hm96)
      (by rw [← hIt, ← e3]; exact hh)
  have hco : s.read.coins = .tree (Zk.coinSpec (s.read.I.tags.tableNames s.read.I.count)) := by
    simp [Readable.Read.coins, ZkExec.zkCoins, hzk]
  have hshape := Claims.Bridge.Custody.setup_shapeOk hset hzk
  unfold Readable.Read.setup at hr
  have hown := setupTables_own_none hh hr
  rw [hown] at hr
  rcases Exec.setupTables_ok hh hr with ⟨-, st, hd, hspec, hsts⟩ | ⟨hc, hsz, -⟩
  swap
  · omega
  subst hspec
  subst hsts
  have hms := Exec.setupH_hm96 hl hd
  have hct := setupH_coinTree hd
  rw [hco] at hct
  obtain ⟨-, -, -, -, hm, -, hrange⟩ := FlockSoundness.setupH_spec hty hd
  have hrange' := (checkInRange_ok _ _ _).1 hrange
  have hsh := Readable.array_forM_ok hshape 0 (by simp)
  obtain ⟨h25, h35⟩ := m_range_of_shapeOk hsh
  have hd' := (HiddenExec.setupOne_public hh).trans hd
  obtain ⟨-, -, -, hhello⟩ := setupOne_stmt hd'
  rw [hco] at hhello
  obtain ⟨m, kvs, hown', -, -, -, hg⟩ := hello_ok hhello
  obtain ⟨hz, hc, -⟩ := FlockComplete.JsonRT.canon_ok _ hg
  exact ⟨st, st.tags.table, _, .obj m, hz, rfl, rfl, hct, hown', hc, h25, h35, hms, by omega, hrange'.2.1, rfl,
    hrange'.2.2, Nat.le_refl 2⟩

/-- **The facts the length bound reads, from the setup alone**, at one table. -/
theorem setup_facts {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) :
    ∃ (st : Flock.Stmt) (t : String) (ct : CoinTree.Spec) (own : Json) (hz : String), sts = #[st] ∧
      spec.tables = #[t] ∧ spec.coinTree = some ct ∧ Flock.JsonParse.parse spec.hello = .ok own ∧
      canon (JsonRT.helloJ own) = .ok hz ∧ 25 ≤ st.m ∧ st.m ≤ 35 ∧ st.merkle = MerkleScheme.hm96Sha512 ∧
      st.c.kLog ≤ st.m ∧ st.regions.size ≤ 1024 ∧ spec.mPts = st.mPts ∧ st.mPts ≤ 64 ∧ spec.points ≤ 2 := by
  cases hh : s.read.tags.hiddenOutputs
  · exact setup_facts_public hs h1 hh
  · exact setup_facts_hidden hs h1 hh

variable {H : Claims.Msg → Claims.Msg}

/-- **The honest masked record weighs at most `2^8` its spec's text and `2^40`**, at one table of a masked session whose
setup is `x`'s, whatever the witness, tape and coins. -/
theorem recordZk_size_le (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1)
    {w : Claims.Flock.Wit} {ω : TapeZk} {k : ℕ} {r : Fin k → Coin} {record : String} {proofs : Array ByteArray}
    (hrun : ((sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run.sim H
      (fun j => (Claims.coinSeq r j).bytes) 0).2 = .ok (record, proofs)) :
    record.toUTF8.size ≤ 2 ^ 8 * specText spec + 2 ^ 40 := by
  obtain ⟨st, t, ct, own, hz, rfl, ht, hct, hown, hc, h25, h35, hms, hkm, hreg, hpm, hmp, hpt⟩ := setup_facts hs h1
  obtain ⟨bs, h2, -⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf #[st])
    (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω #[]
  have h3 := congrArg Prod.snd h2
  simp only [List.size_toArray, List.length_nil] at h3
  rw [hrun] at h3
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun' : (sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) spec
      #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[] =
      (.ok (record, proofs), ((sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H))
        spec #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[]).2) := by
    rw [← htabs, h3]
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  exact recordZk_size_of h64 ht hct hown hc st h25 h35 hms hkm hreg hpm hmp hpt _ _ ω r hrun'

/-- **The honest masked record fits its length word** at one table of a masked session whose setup is `x`'s, where its
spec's text is under `2^55`. -/
theorem recordFitsZkAt_of_text (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1)
    (ht : specText spec < 2 ^ 55) : RecordFitsZkAt H x spec sts := by
  obtain ⟨st, t, ct, own, hz, rfl, htb, hct, hown, hc, h25, h35, hms, hkm, hreg, hpm, hmp, hpt⟩ := setup_facts hs h1
  exact recordFitsZk_of_text h64 htb hct hown hc st h25 h35 hms hkm hreg hpm hmp hpt ht

end FlockComplete.ZkTables
