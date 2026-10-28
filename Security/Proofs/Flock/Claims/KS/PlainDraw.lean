import Proofs.Flock.Claims.KS.PlainEvent

/-!
# The plain verdict at the statement's draw

`plainVerdict` reads flock-verify's acceptance of what `wrP` writes, and `wrP` asks for a record whose draw is the
statement's as the same term (`RecordsLiveAtP`), since `Exec.Accepts` sets the session up at the record's own draw.
`Deploy.V false` sets the session up at the statement's draw and runs `verify` on the record against that setup; its
checks compare the two draws only up to key order (`drawEq_ne`). So the verdict here reads the setup at the statement's
draw `d`, and no draw in the record is read, as the masked side does (`Composed.AcceptsAt`):

* `setupAtD`, `AcceptsAtD`: `Exec.setupOf` and `Exec.Accepts` at a draw value rather than a record's
  (`setupOf_eq_setupAtD`, `accepts_iff_acceptsAtD`).
* `RecordsLiveAtD`, `TxNoOffAt`: `RecordsLiveAtP` and `TxNoOff` at the setup at `d`, without the record's draw.
* `wrPD`, `recordsLiveAtD_wrPD`, `acceptsAtD_wrPD`: the writer at `d` (`wrP` without the draw clause).
* `setupAtD_plainInputs`, `recordsLiveAtD_of_checks`, `acceptsAtD_of_decide`: an accepting decision of `Deploy.V false`
  is accepted at the setup at its statement's draw and records its live outcome there, with no premise on the record's
  draw (`setupOf_plainInputs`, `recordsLiveAtP_of_checks` and `accepts_of_decide_plain` without `hdraw`).
* `plainVerdictAt`, `plainVerdictAt_of_run`: `plainVerdict` at `wrPD`, which every accepting run of `V false` with no
  column off `C` has at its live outcome, on every draw-carrying record (`plainVerdict_of_run` without its premise
  that the record carry the statement's draw verbatim).
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Refine (ptsOf shapeOf)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`verify`'s setup at a draw value**: `Exec.setupOf` at the draw `dr` itself rather than a record's
(`setupOf_eq_setupAtD`). -/
def setupAtD (I : Exec.Inputs) (dr : Option Json) : _root_.Flock.V (SessionSpec × Array Stmt) :=
  Stmt.setupTables I.tags I.circuitFile I.publicFile I.coins I.count I.tables dr I.partition I.program I.work
    I.stratified I.closure

theorem setupOf_eq_setupAtD (I : Exec.Inputs) (record : String) :
    Exec.setupOf I record = setupAtD I (Exec.recordDraw record) := rfl

/-- **flock-verify accepts at the session it sets up at the draw `dr`**: `Exec.Accepts` with the setup at `dr` for the
setup at the record's draw. -/
def AcceptsAtD (I : Exec.Inputs) (dr : Option Json) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupAtD I dr = .ok (spec, sts) ∧
    _root_.Flock.verify spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

theorem accepts_iff_acceptsAtD (I : Exec.Inputs) (t : Exec.Transcript) :
    Exec.Accepts I t ↔ AcceptsAtD I (Exec.recordDraw t.record) t := Iff.rfl

/-- **`RecordsLiveAtP` at the setup at the draw `d`**: where `verify` accepts the transcript `t` at the session it sets
up at `d`, its proof files are the outcome `o`'s, and the session its record decodes to there has `o`'s link points
and, at each of `o`'s tables, `o`'s `root_B` and rep streams' rounds. -/
def RecordsLiveAtD (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) (t : Exec.Transcript) :
    Prop :=
  AcceptsAtD I (some d) t →
    t.proofs = o.proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

/-- **No column off `C` in a transcript, at the setup at the draw `d`**: `TxNoOff` with the setup at `d` for the setup
at the record's draw. -/
def TxNoOffAt (I : Exec.Inputs) (d : Json) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) : Prop :=
  ∀ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : RunRecord) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) → Record.decode spec t.record = .ok sess →
    _root_.Flock.fast100 st.m = some sch →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]!)) p.val = some u ∧ u ≠ C p

open Classical in
/-- **The record writer at the draw `d`**: at the live outcome `o`, a transcript `verify` accepts at the setup at `d`
that records `o` there (`RecordsLiveAtD`), where one exists, else a session without proof files, which `verify`
refuses (`verify_nil_plain`). -/
noncomputable def wrPD (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) : Exec.Transcript :=
  if h : ∃ t, AcceptsAtD I (some d) t ∧ RecordsLiveAtD I d o t then h.choose else ⟨"", #[]⟩

theorem not_acceptsAtD_nil (I : Exec.Inputs) (dr : Option Json) (record : String) :
    ¬ AcceptsAtD I dr ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verify_nil_plain _ _ _ h

/-- **`wrPD` records the live session at the setup at `d`.** -/
theorem recordsLiveAtD_wrPD (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) :
    RecordsLiveAtD I d o (wrPD I d o) := by
  unfold wrPD
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsAtD_nil _ _ _)

/-- **`verify` accepts what `wrPD` writes** wherever it accepts some transcript at the setup at `d` that records the
outcome there. -/
theorem acceptsAtD_wrPD {I : Exec.Inputs} {d : Json} {J mPts : ℕ} {o : Tables.LiveOutJ J mPts} {t : Exec.Transcript}
    (hA : AcceptsAtD I (some d) t) (hR : RecordsLiveAtD I d o t) : AcceptsAtD I (some d) (wrPD I d o) := by
  have hex : ∃ t, AcceptsAtD I (some d) t ∧ RecordsLiveAtD I d o t := ⟨t, hA, hR⟩
  unfold wrPD
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

/-- **`verify`'s setup at the statement's draw is `Deploy.V`'s**, where the statement is plain and carries no
`--public-inputs`: `setupOf_plainInputs` with no record. -/
theorem setupAtD_plainInputs {r : Readable.Read} {spec : SessionSpec} {sts : Array Stmt} {d : Json}
    (hs : r.setup = .ok (spec, sts)) (hzk : r.zk = false) (hpi : r.publicInputs = none) (hd : r.draw = some d) :
    setupAtD r.I (some d) = .ok (spec, sts) := by
  rw [← hs]
  simp only [setupAtD, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, hzk, hpi, hd,
    Bool.false_eq_true, ↓reduceIte]

/-- **Custody gives `RecordsLiveAtD`**: where `Deploy.V false`'s checks pass on a plain statement that sets up at its
draw `d` with no `--public-inputs`, the last message's record and proof files are a record of the run's live outcome
(`viewOf`) at the setup at `d`, at any `J` of the session's tables, whatever draw the record carries
(`recordsLiveAtP_of_checks` without `hdraw`). -/
theorem recordsLiveAtD_of_checks {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = false)
    (hpi : s.read.publicInputs = none) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size)
    (hc : checks false s spec x link c₁ mids f = .ok ()) :
    RecordsLiveAtD s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupAtD_plainInputs (Ins.setup_read hs) hzk hpi hd] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

/-- **An accepting decision of `Deploy.V false` at SHA-512 is an accepting `verify` at the setup at its statement's
draw**, on a record of its live outcome there, whatever draw the record carries (`accepts_of_decide_plain` without
`hdraw`). -/
theorem acceptsAtD_of_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} {J mPts : ℕ} {d : Json}
    (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    AcceptsAtD s.read.I (some d) ⟨f.record, f.proofs⟩ ∧
      RecordsLiveAtD s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  exact ⟨⟨spec, sts, setupAtD_plainInputs (Ins.setup_read hs) hzk hpi hd, hv⟩,
    recordsLiveAtD_of_checks hs hzk hpi hd hJ hc⟩

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain verdict at the draw `d`**: `verify` accepts what `wrPD` writes for the outcome at the setup at `d`, the
outcome's root is `rootB₀`'s, and the deployed plain reader looks up no column off `C` in that transcript there
(`TxNoOffAt`). `plainVerdict` with the setup at `d` for the setup at the record's draw. -/
def plainVerdictAt (I : Exec.Inputs) (d : Json) {N mPts : ℕ} (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray)
    (o : Tables.LiveOutJ 1 mPts) : Prop :=
  AcceptsAtD I (some d) (wrPD I d o) ∧ o.rootB[0]! = rootB₀[0]! ∧ TxNoOffAt I d C (wrPD I d o)

/-- **The deployed event gives the verdict at the draw**: an accepting run of `V false` at SHA-512 decides on its
messages and coins, and at its statement's draw `d`, if the run's reader looks up no column off `C` at table 0,
`plainVerdictAt` holds at the run's live outcome (`viewOf`, at any `mPts`) and its link's root, whatever draw its record
carries: `verify` accepts the last message at the setup at `d` (`acceptsAtD_of_decide`), so it accepts what `wrPD`
writes, which records the run's streams and proof files there, whose columns are the reader's (`cols_plain`). -/
theorem plainVerdictAt_of_run (mPts : ℕ) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {N : ℕ} (C : Fin N → List 𝔽)
    (h : (Claims.run sha512 P (V false) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ d, s.read.draw = some d →
        (¬ ∃ (p : Fin N) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) →
        plainVerdictAt s.read.I d C link.1.toArray (viewOf 1 mPts spec link c₁ f) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := acc_decide h
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, fun d hd hno => ?_⟩
  have hacc := run_acc_next sha512 P (V false) x ω r h
  replace hacc : ((next false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu⟩ := setupOf_some (act_decide hact)
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD s.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verify_ok_tables hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  obtain ⟨hA₀, hR₀⟩ := acceptsAtD_of_decide (J := 1) (mPts := mPts) hact hacc hd hJ
  have hA := acceptsAtD_wrPD hA₀ hR₀
  refine ⟨hA, rfl, ?_⟩
  intro spec' st sess sch s₀ s₁ hset' hdec' hsch hs₀ hs₁
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD s.read.I d (viewOf 1 mPts spec link c₁ f) hA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hset'))
  obtain ⟨-, hrs⟩ := hall spec #[st] sess hset' hdec'
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω r).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω r).vLog.map Prod.fst)) spec
      (tabsOf #[st]) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := cols_plain (st := st) _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hsch
    (HiddenExec.setupTables_pointsAny hsetA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  rw [hpr]
  show ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow _
    (Claims.Flock.plainRepColsOf _ y₀.rounds f.proofs[0]! ++ Claims.Flock.plainRepColsOf _ y₁.rounds f.proofs[1]!))
      p.val = some u ∧ u ≠ C p
  rw [plainRepColsOf_congr _ _ (hco 0 y₀ hr₀), plainRepColsOf_congr _ _ (hco 1 y₁ hr₁), ← getD_empty,
    ← getD_empty, ← hcols]
  exact hno

end FlockDeploy.KS
