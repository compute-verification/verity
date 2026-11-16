module

public import Proofs.Flock.Claims.Complete.ZkTables.HelloSize
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The one-table masked session's setup, from `setupOf true x` alone

At a statement `x` whose masked setup returns one table (`setupOf true x = some (s, spec, sts)`, `sts.size = 1`), the
table's statement `st` has the facts the length bound reads (`setup_facts`): `25 ≤ m ≤ 35` (`Zk.shapeOk`, `Spec.m35`),
the `hm96` leaf (the tags' leaf scheme, which `Tags.byName?` pins at `hm96Rows`), `k_log ≤ m`, at most `2^10` regions
and `m_pts ≤ 64` (`checkInRange`), a keyed coin tree (`zkCoins` is a tree, and `setupHidden` keeps it:
`setupHidden_coinTree`) and a `Hello` that reads back (`hello_ok`). It holds for a statement with hidden outputs, at any
public inputs.
-/

@[expose] public section

open _root_.Flock

namespace FlockDeploy

theorem insOf_ins {zk : Bool} {x : Claims.Msg} {s : Ins} (h : insOf zk x = some s) :
    ∃ w, decode x = some w ∧ w.zk = zk ∧ w.ins = .ok s := by
  unfold insOf at h
  obtain ⟨w, hw, h2⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h2 with hz
  · cases h2
  · change w.ins.toOption = some s at h2
    refine ⟨w, hw, not_not.1 hz, ?_⟩
    cases hi : w.ins with
    | error e => rw [hi] at h2; cases h2
    | ok s' =>
      rw [hi] at h2
      cases h2
      rfl

theorem Raw.ins_tags {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    ∃ t, Tags.byName? w.statement = some t ∧ s.read.I.tags = t ∧ (w.zk = true → t.plainLeaves = false) := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, by assumption, rfl, by assumption⟩) | (exfalso; contradiction)

/-- A masked setup's statement has the hm96 rows (`Ins.setup` refuses any other). -/
theorem Ins.setup_hm96Rows {s : Ins} {p : SessionSpec × Array Stmt} (h : s.setup = .ok p) :
    s.read.tags.hm96Rows = true := by
  unfold Ins.setup at h
  by_contra hn
  have hf : s.read.tags.hm96Rows = false := by simpa using hn
  simp [hf, throw_bind'] at h

end FlockDeploy

namespace FlockComplete.ZkTables

open Flock Transcript FlockDeploy With FlockComplete JsonRT FlockSoundness.Discharge FlockComplete.Replay
open Lean (Json)

/-- Among the named tags, those with the hm96 rows commit with `hm96-sha512/v1`, but the plain-leaves ones. -/
theorem tags_hm96_or : ∀ t ∈ Tags.all ++ Tags.templated, t.hm96Rows = true →
    t.merkleLeaf = "hm96-sha512/v1" ∨ t.name = "verity/flock-circuit+plain-leaves" := by
  decide +kernel

theorem plainLeaves_circuitPlainLeaves : Tags.circuitPlainLeaves.plainLeaves = true := by
  simp [Tags.plainLeaves, Tags.circuitPlainLeaves, Tags.withPlainLeaves]

/-- A statement of `Tags.byName?` with the hm96 rows and no plain leaves commits with `hm96-sha512/v1`. -/
theorem byName_hm96 {n : String} {t : Tags} (h : Tags.byName? n = some t) (hp : t.plainLeaves = false)
    (h9 : t.hm96Rows = true) : t.merkleLeaf = "hm96-sha512/v1" := by
  have hmem : t ∈ Tags.all ++ Tags.templated := by
    unfold Tags.byName? at h
    cases hf : Tags.find? n with
    | some t' =>
      rw [hf] at h
      change some t' = some t at h
      cases h
      unfold Tags.find? at hf
      exact List.mem_append_left _ (List.mem_of_find?_eq_some hf)
    | none =>
      rw [hf] at h
      change Tags.templated.find? (·.name == n) = some t at h
      exact List.mem_append_right _ (List.mem_of_find?_eq_some h)
  rcases tags_hm96_or t hmem h9 with h1 | h1
  · exact h1
  · exfalso
    have hn : t.name = n := Tags.byName?_name h
    have e : Tags.byName? "verity/flock-circuit+plain-leaves" = some Tags.circuitPlainLeaves := by rfl
    rw [← hn, h1, e] at h
    have : Tags.circuitPlainLeaves = t := Option.some.inj h
    rw [← this, plainLeaves_circuitPlainLeaves] at hp
    cases hp

open _root_.Flock FlockSoundness.Refine in
set_option maxHeartbeats 4000000 in
/-- **A statement `Stmt.setupHidden` returns keeps the coin tree its setup was given.** -/
theorem setupHidden_coinTree {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program own : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.coinTree = coins.tree? := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

/-- **The facts the length bound reads, from the setup alone**: at one table under tags with hidden outputs, at any public
inputs. -/
theorem setup_facts {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) (hh : s.read.tags.hiddenOutputs = true) :
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
  obtain ⟨w, hw, hwz, hwi⟩ := insOf_ins hi
  obtain ⟨t, hbn, hIt, hpl⟩ := Raw.ins_tags hwi
  have hl : s.read.tags.merkleLeaf = "hm96-sha512/v1" := by
    rw [e2, hIt]
    exact byName_hm96 hbn (hpl hwz) (by rw [← hIt, ← e1]; exact hm96)
  have hco : s.read.coins = .tree (Zk.coinSpec (s.read.I.tags.tableNames s.read.I.count)) := by
    simp [Readable.Read.coins, ZkExec.zkCoins, hzk]
  have hshape := Claims.Bridge.Custody.setup_shapeOk hset hzk
  unfold Readable.Read.setup at hr
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hr with ⟨-, st, hd, hspec, hsts⟩ | ⟨hc, hsz, -⟩
  swap
  · omega
  subst hspec
  subst hsts
  have hsp := (FlockSoundness.Discharge.Hidden.setupHidden_specP hd).choose_spec
  have hms := FlockSoundness.Discharge.Hidden.setupHidden_hm96P hl hd
  have hct := setupHidden_coinTree hd
  rw [hco] at hct
  have hrange := (checkInRange_ok _ _ _).1 hsp.inRange
  have hsh := Readable.array_forM_ok hshape 0 (by simp)
  have h25 := Readable.m25_of_shapeOk hsh
  have hhello := hsp.hello
  rw [hco] at hhello
  obtain ⟨m, kvs, hown, -, -, -, hg⟩ := hello_ok hhello
  obtain ⟨hz, hc, -⟩ := FlockComplete.JsonRT.canon_ok _ hg
  refine ⟨st, st.tags.table, _, .obj m, hz, rfl, rfl, hct, hown, hc, h25, hsp.m35, hms, ?_, hrange.2.1, rfl, ?_,
    Nat.le_refl 2⟩
  · have := hsp.m
    omega
  · have := hsp.mPts
    omega

variable {H : Claims.Msg → Claims.Msg}

/-- **The honest masked record weighs at most `2^8` its spec's text and `2^40`**, at one table of a session whose setup
is `x`'s (hidden outputs, any public inputs): whatever the witness, tape and coins. -/
theorem recordZk_size_le (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1)
    (hh : s.read.tags.hiddenOutputs = true) {w : Claims.Flock.Wit} {ω : TapeZk} {k : ℕ} {r : Fin k → Coin}
    {record : String} {proofs : Array ByteArray}
    (hrun : ((sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run.sim H
      (fun j => (Claims.coinSeq r j).bytes) 0).2 = .ok (record, proofs)) :
    record.toUTF8.size ≤ 2 ^ 8 * specText spec + 2 ^ 40 := by
  obtain ⟨st, t, ct, own, hz, rfl, ht, hct, hown, hc, h25, h35, hms, hkm, hreg, hpm, hmp, hpt⟩ :=
    setup_facts hs h1 hh
  obtain ⟨bs, h2, -⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf #[st])
    (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω #[]
  have h3 := congrArg Prod.snd h2
  simp only [List.size_toArray, List.length_nil, Array.size_empty] at h3
  rw [hrun] at h3
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun' : (sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) spec
      #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[] =
      (.ok (record, proofs), ((sessionPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H))
        spec #[Setup.ofCircuit st] (MaskRank.masksOf #[st]) (tracesOf #[st] w) ω).go #[]).2) := by
    rw [← htabs, h3]
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  exact recordZk_size_of h64 ht hct hown hc st h25 h35 hms hkm hreg hpm hmp hpt _ _ ω r hrun'

/-- **The honest masked record fits its length word** at one table of a session whose setup is `x`'s (hidden outputs,
any public inputs), where its spec's text is under `2^55`. -/
theorem recordFitsZkAt_of_text (h64 : Claims.Out64 H) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1)
    (hh : s.read.tags.hiddenOutputs = true) (ht : specText spec < 2 ^ 55) : RecordFitsZkAt H x spec sts := by
  obtain ⟨st, t, ct, own, hz, rfl, htb, hct, hown, hc, h25, h35, hms, hkm, hreg, hpm, hmp, hpt⟩ :=
    setup_facts hs h1 hh
  exact recordFitsZk_of_text h64 htb hct hown hc st h25 h35 hms hkm hreg hpm hmp hpt ht

end FlockComplete.ZkTables
