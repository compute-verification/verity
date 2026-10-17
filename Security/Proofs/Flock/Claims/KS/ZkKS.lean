import Proofs.Flock.Claims.KS.ZkBound
import Proofs.Flock.Claims.KS.PlainKS
import Proofs.Flock.Soundness.Discharge.ZkLink.M1
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements

/-!
# The deployed masked verifier's KS from the live sessions' bound at one table

`PlainKS` at `--zk`. `zk_KS_of_live`: `Flock.KS` at SHA-512 at the trimmed record `deployZkTrim` at the deployed
relation, from two premises:

* `ZkOne`: every statement some run accepts parses with one table (`zk_KS_of_table`'s `hone`). The masked scope admits
  several tables (`InScope`'s `zk ∨ tables ≤ 1`), so this holds of a masked deployment only where no run accepts a
  statement of several;
* `ZkLive x` at every statement some run accepts: at C-Flock ZK's padded table there (`zkTab`), every prover has
  escapes at injective fork rounds and, at each tape and each table of full-width columns, live strategies whose verdict
  bounds the deployed runs accepting on the table, a decoder right for that verdict, and escapes' events bounding their
  `Adv`s and self-clash (`ZkLiveAt` at `hs512`, the hypothesis `zk_tableBoundθAt_of_live` reads).

Everything else is proved here at a statement some run accepts with one table, which parses in scope and sets up:
table 0's layout and link layout (`zk_table0`), `Zk.schedule`'s schedule and `t_pad` at its `m` (`zk_schedule_ok`), the
padded table `zkTab` at them with the executable's padding constant, pads layout and inner code (`DrawSetupZK.tabAt`'s
fields, without a draw), that `zkTab` is at C-Flock ZK's parameters (`zkTab_m1`), and that a table that does not extract
is not committed to it (`zk_not_committed`, `zkTab_committed`). `zk_tableBoundθ_live` is the bound at full-width
columns, the form the trimmed record reads (`zk_KS_trim`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

/-- **`Zk.schedule` accepts every `m` from 25 to 35**, at `zkSched m`. -/
theorem zk_schedule_ok {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35) :
    Zk.schedule m = .ok (Claims.Flock.zkSched m) := by
  have h := (Claims.Flock.zkShape_table ⟨m - 25, by omega⟩).1
  have hi : Claims.Flock.ZkIdx.m ⟨m - 25, by omega⟩ = m := by
    simp only [Claims.Flock.ZkIdx.m]
    omega
  rw [hi] at h
  unfold Claims.Flock.zkSched
  cases hs : Zk.schedule m with
  | ok p => rfl
  | error e => rw [hs] at h; simp [Except.toBool] at h

theorem zkSched_of {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule m = .ok (sch, tPad)) :
    Claims.Flock.zkSched m = (sch, tPad) := by
  unfold Claims.Flock.zkSched
  rw [hz]

/-- `codeAt`'s points at `N` columns, distinct below `2^64` (`DrawSetupZK.innerDom`). -/
def innerDomOf (N : ℕ) (hN : N < 2 ^ 64) : Fin N ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin N, j.val + 1 < 2 ^ 64 := fun j => by
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- The inner code's length at a statement `Zk.schedule` accepts, of `k_log ≤ 27`: a power of two, at most `2^13`. -/
theorem innerN_pow_zk {st : Stmt} {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) : ∃ b, Discharge.ZkExec.innerN (layOf st sch) = 2 ^ b ∧ b ≤ 13 := by
  obtain ⟨hshape, hmsch, -⟩ := schedule_zk hz
  obtain ⟨-, hm35, -⟩ := schedule_spec hz
  have h0 : sch.k0 ≤ 35 := by
    have := hshape.2.2.1
    simp only [Refine.schedOf] at this hmsch
    omega
  exact innerN_pow _ (layLen_le hm35 hk h0)

theorem innerN_lt_zk {st : Stmt} {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) : Discharge.ZkExec.innerN (layOf st sch) < 2 ^ 64 := by
  obtain ⟨b, hb, hb13⟩ := innerN_pow_zk hz hk
  rw [hb]
  exact Nat.pow_lt_pow_right (by norm_num) (by omega)

theorem innerN_dvd_zk {st : Stmt} {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) : Discharge.ZkExec.innerN (layOf st sch) ∣ 2 ^ 64 := by
  obtain ⟨b, hb, hb13⟩ := innerN_pow_zk hz hk
  rw [hb]
  exact Nat.pow_dvd_pow 2 (by omega)

/-- **C-Flock ZK's padded table at a statement**, the fields of `DrawSetupZK.tabAt` at layout `L`'s model statement
at its regions: `Zk.schedule`'s schedule and `t_pad` at its `m`, the padding constant `PAD_C`, two extra lanes, link
layout `PT_LOCAL`, the pads at `PadLayout.new m kLog k0`'s positions, and the inner code `verifyInner` reads (`innerN`
columns, column `j` at `j + 1`, `K_PAD` padding symbols, `Q` queries at `w.lo mod innerN`, the unique-decoding
radius). -/
noncomputable def zkTab {st : Stmt} (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule st.m = .ok (sch, tPad)) (hk : st.c.kLog ≤ 27) {mPts : ℕ}
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL mPts) : TabZK 𝔽 mPts where
  S := L.model (Refine.regionsOf st)
  sch := Refine.schedOf sch
  hshape := (schedule_zk hz).1
  hmsch := (schedule_zk hz).2.1
  hm := by
    have := (schedule_spec hz).1
    show 13 ≤ st.m
    omega
  t := tPad
  hpad := (schedule_zk hz).2.2
  c₀ := padC
  e := 2
  ptLocal := PT_LOCAL
  hlay := hlay
  np := (layOf st sch).len
  pads := padsOfLay (layOf st sch)
  padsA := padsAOfLay (layOf st sch)
  tr := tripleOfLay (layOf st sch)
  N := Discharge.ZkExec.innerN (layOf st sch)
  dom := innerDomOf _ (innerN_lt_zk hz hk)
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ (lt_of_lt_of_le (by simp [Zk.K_PAD]) (innerN_ge _))
  hpos := fun D => prCoin_innerPos_le _ _ (innerN_dvd_zk hz hk) D
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code (innerDomOf _ (innerN_lt_zk hz hk)) ((layOf st sch).len + Zk.K_PAD) :
      Set (Fin (Discharge.ZkExec.innerN (layOf st sch)) → 𝔽))
  hL := innerN_ge _
  hδ := le_rfl

/-- **`zkTab` is at C-Flock ZK's parameters**: `m1Schedule` and `m1Pad` at `m`, two extra lanes, the shape `acct`,
at most `2^13` inner columns, 168 queries, a radius of at least `7/16`, and the deployed pads domain. -/
theorem zkTab_m1 {st : Stmt} (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule st.m = .ok (sch, tPad)) (hk : st.c.kLog ≤ 27) (spec : SessionSpec)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) :
    (zkTab L hz hk hlay).sch = Accounting.m1Schedule st.m ∧ (zkTab L hz hk hlay).t = Accounting.m1Pad st.m ∧
      (zkTab L hz hk hlay).e = 2 ∧ (zkTab L hz hk hlay).S.shape spec.mPts = acct spec st ∧
      (zkTab L hz hk hlay).N ≤ 2 ^ 13 ∧ (zkTab L hz hk hlay).q = 168 ∧ 7 / 16 ≤ (zkTab L hz hk hlay).δ ∧
      (zkTab L hz hk hlay).N ≤
        Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0) := by
  obtain ⟨hsch, hpad⟩ := schedule_m1 hz
  obtain ⟨b, hb, hb13⟩ := innerN_pow_zk hz hk
  have h8 : 8 * ((layOf st sch).len + Zk.K_PAD) ≤ Discharge.ZkExec.innerN (layOf st sch) := by
    have := Nat.le_nextPowerOfTwo ((layOf st sch).len + Zk.K_PAD)
    rw [Discharge.ZkExec.innerN, Zk.INNER_LOG_INV_RATE, show (2 : ℕ) ^ 3 = 8 by norm_num]
    omega
  have : NeZero ((layOf st sch).len + Zk.K_PAD) := ⟨by simp [Zk.K_PAD]⟩
  refine ⟨hsch, hpad, rfl, shape_model L spec, ?_, rfl, relUDR_ge_m1 (innerDomOf _ (innerN_lt_zk hz hk)) h8, ?_⟩
  · show Discharge.ZkExec.innerN (layOf st sch) ≤ 2 ^ 13
    rw [hb]
    exact Nat.pow_le_pow_right (by norm_num) hb13
  · rw [zkSched_of hz]
    exact le_rfl

/-- **`zkTab` commits to what `zk_not_committed` speaks of**: table 0's model statement, `zkSched`'s schedule and
padding, and `PAD_C`. -/
theorem zkTab_committed {st : Stmt} (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule st.m = .ok (sch, tPad)) (hk : st.c.kLog ≤ 27) {mPts : ℕ}
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL mPts) (C : Model.Oracle 𝔽) :
    (zkTab L hz hk hlay).Committed execArith C ↔
      CommittedPad execArith (L.model (Refine.regionsOf st)) (Refine.schedOf (Claims.Flock.zkSched st.m).1)
        (Claims.Flock.zkSched st.m).2 padC C := by
  rw [zkSched_of hz]
  rfl

/-- **The live sessions' bound at a masked statement**: `ZkLiveAt` at the deployed hashes (`hs512`) and tables of
full-width columns, at `zkTab`, wherever the setup has the one table `st`, at its every layout, link layout at
`PT_LOCAL`, and `Zk.schedule`'s schedule. -/
def ZkLive (x : Claims.Msg) : Prop :=
  ∀ (spec : SessionSpec) (st : Stmt) (_ : tables true x = [(spec, st)]) (L : Layout st) (sch : Flock.Schedule)
    (tPad : ℕ) (hz : Zk.schedule st.m = .ok (sch, tPad)) (hk : st.c.kLog ≤ 27)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    ZkLiveAt execArith Discharge.ZkExec.hs512 (wide true) x (zkTab L hz hk hlay)

/-- The readable statement's `k_log` is table 0's. -/
theorem parse_kLog {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk Readable.relation x = some s)
    {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts))
    (h0 : 0 < sts.size) : s.kLog = sts[0].c.kLog := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
  subst hpst
  simp only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]

/-- **Table 0 of a masked statement in scope**: its layout and its link layout at `PT_LOCAL` (`plain_table0`). -/
theorem zk_table0 {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size) :
    ∃ L : Layout (sts[0]'h0), (L.model (Refine.regionsOf (sts[0]'h0))).LinkLayout PT_LOCAL spec.mPts := by
  obtain ⟨p, hp, rfl, hP, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, ht, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at ht hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  obtain ⟨-, L, hwf, hr⟩ := Readable.tableAt (Readable.tablesOf_of_parses hP) ht hpi' h0
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  have hpts := (Discharge.HiddenExec.setupTables_pointsAny hsr).2 0 h0
  refine ⟨L, (Refine.stmtOf_linkLayout _ hwf hr).1, ?_⟩
  show PT_LOCAL + ((p.sts[0]'h0).m - (p.sts[0]'h0).c.kLog) ≤ spec.mPts
  have h1 := hr.mPts
  have h2 := hr.hm
  omega

/-- **The masked table bound at a statement in scope of one table, from the live sessions' bound**: at tables of
full-width columns, where the statement parses in scope and sets up one table. -/
theorem zk_tableBoundθAt_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h1 : sts.size = 1)
    (hl : ZkLive x) :
    Claims.Flock.TableBoundθAt sha512 (deployZk Readable.relation) (wordLen true) (wide true) x := by
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have hkl := parse_kLog hs hset h0
  have h25 : 25 ≤ (sts[0]'h0).m := by have := hin.1; simp only [↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  have hk : (sts[0]'h0).c.kLog ≤ 27 := by have := hin.2.2.2.1; omega
  obtain ⟨L, hlay⟩ := zk_table0 hs hin hset h0
  obtain ⟨sch, tPad, hz⟩ : ∃ sch tPad, Zk.schedule (sts[0]'h0).m = .ok (sch, tPad) :=
    ⟨_, _, zk_schedule_ok h25 h35⟩
  have ht := tables_of_one hset h1
  obtain ⟨hsch, hpad, he, hS, hN, hq, hδ, hNp⟩ := zkTab_m1 L hz hk spec hlay
  exact zk_tableBoundθAt_of_live execArith execArith_correct Discharge.ZkExec.hs512 Readable.relation (wide true) ht
    (zkTab L hz hk hlay) h25 h35 hsch hpad he hS hN hq hδ hNp
    (fun C hw hg hc => zk_not_committed hs hin hset h1 L C hw hg ((zkTab_committed L hz hk hlay _).1 hc))
    (hl spec _ ht L sch tPad hz hk hlay)

/-- **One table at every statement some masked run accepts** (`zk_KS_of_table`'s `hone`). The masked scope admits
several tables (`InScope`'s `zk ∨ tables ≤ 1`), so this is a premise on which statements a run accepts, which a
deployment that serves statements of several tables does not meet. -/
def ZkOne : Prop :=
  ∀ x, (∃ s, (deployZk Readable.relation).parse x = some s ∧ s.tables = 1) ∨
    Claims.Flock.NoAcc sha512 (deployZk Readable.relation).V x

/-- **The masked table bound at full-width columns, from the live sessions' bound** at the statements some run
accepts, each of one table. -/
theorem zk_tableBoundθ_live (hone : ZkOne) (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLive x) :
    Claims.Flock.TableBoundθ sha512 (deployZk Readable.relation) (wordLen true) (wide true) :=
  zk_tableBoundθ_of_acc Readable.relation (wide true) fun x s _ _ _ hno hs hin hset => by
    have h1 : s.tables = 1 := by
      rcases hone x with ⟨s', hs', h1⟩ | hna
      · obtain rfl : s = s' := Option.some.inj (hs.symm.trans hs')
        exact h1
      · exact absurd hna hno
    obtain ⟨p, hp, -, -, hsz, -⟩ := parsed_of_parse hs
    obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
    exact zk_tableBoundθAt_live hs hin hset (by rw [← hpst, hsz, h1]) (hl x hno)

/-- **The deployed masked verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, from one table at
every statement some run accepts and the live sessions' bound there. -/
theorem zk_KS_of_live (hone : ZkOne) (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLive x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim Readable.relation hone (zk_tableBoundθ_live hone hl)

end FlockDeploy.KS
