import Proofs.Flock.Claims.KS.ZkDecodeW
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# The deployed masked event at `wrS`'s verdict

`PlainEvent` under `--zk`, at table `j`: on an accepting run of `Deploy.V --zk` whose reader looks up no column of table
`j` off `C`, the verdict the witness-lane decoders are right at on (`AccTxJ`, `decodesTableW_deploy`) holds at the run's
live outcome (`viewOf`), so what is left of a mixture bound at that verdict is the transport of the run's coins to the
live session's:

* `cols_zk`: an accepted masked run's reader looks up table `j`'s columns at the shape and level 0 `ZkTxNoOff` reads,
  in its live session's rep streams `2j`, `2j + 1` and proof files.
* `accTxJ_of_run`: an accepting run of `V true` at SHA-512 decides, and on a statement of draw `d`, a run whose reader
  looks up no column of table `j` off `C` has `AccTxJ` at table `j` at its live outcome and its link's root, at any
  `J ≤` the statement's tables above `j`: `verify --zk` accepts what `wrS` writes for it at the setup at `d`
  (`acceptsAt_wrS`), and that transcript records the run's streams and proof files (`RecordsLiveAtS`), whose columns
  are the reader's (`cols_zk`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

namespace Gen

variable [S : ZkSetup]

/-- **An accepted masked run's reader looks up table `j`'s columns at `ZkTxNoOff`'s shape**: where the run's last
message `f` sets up at `(spec, sts)` with two link points, the masked verdict accepts its record and proof files at any
hashes and `liveOf` reads the record, the reader's table-`j` columns are `zkRepColsOf` at `shapeZKn` with
`2 · |regions|` extra claims, on the live session's rep streams `2j`, `2j + 1` and proof files, below level 0's `2^d`
of table `j`'s `Zk.schedule`. -/
theorem cols_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    cols true seen j = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.zkRepColsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j) #[]) (f.proofs.getD (2 * j) ByteArray.empty) ++
        Claims.Flock.zkRepColsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j + 1) #[]) (f.proofs.getD (2 * j + 1) ByteArray.empty)) := by
  obtain ⟨hsz, hall⟩ := rep_zk hs hv hlv
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hf', hrep, -, -⟩ := hall (2 * j) (by omega)
  obtain ⟨-, extras, -, hex, -⟩ := zkVerifyRepWith_rounds hs hf' hrep
  have hj' : sts[2 * j / 2]! = sts[j] := by
    rw [show 2 * j / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hzs : Claims.Flock.zkSched (Setup.ofCircuit sts[j]).m = (sch, tPad) := by
    simp only [Claims.Flock.zkSched]
    rw [show (Setup.ofCircuit sts[j]).m = sts[j].m from rfl, hsch]
  have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit sts[j]).m (Setup.ofCircuit sts[j]).kLog (2 + extras.size)
      (Setup.ofCircuit sts[j]).merkle.digestLen (saltLenOf true (Setup.ofCircuit sts[j])) =
      ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [Claims.Flock.zkProofShape, hzs, ZkExec.shapeZKn, saltLenOf, hx, ite_true]
  have hrun : runOf true seen j = some
      { m := (Setup.ofCircuit sts[j]).m, kLog := (Setup.ofCircuit sts[j]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[j]).merkle.digestLen, saltLen := saltLenOf true (Setup.ofCircuit sts[j]),
        reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                 (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, Array.getElem?_eq_getElem hj, hlv, hex, Except.toOption]
  unfold cols
  rw [hrun]
  simp only [↓reduceIte, Claims.Flock.zkRunCols, Claims.Flock.ZkRun.shape, hsh, Claims.Flock.zkLevel0, hzs,
    List.flatMap_cons, List.flatMap_nil, List.append_nil]

end Gen

/-- **An accepted masked run's reader looks up table `j`'s columns at `ZkTxNoOff`'s shape**: where the run's last
message `f` sets up at `(spec, sts)` with two link points, the masked verdict accepts its record and proof files at any
hashes and `liveOf` reads the record, the reader's table-`j` columns are `zkRepColsOf` at `shapeZKn` with
`2 · |regions|` extra claims, on the live session's rep streams `2j`, `2j + 1` and proof files, below level 0's `2^d`
of table `j`'s `Zk.schedule`. -/
theorem cols_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    cols true seen j = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.zkRepColsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j) #[]) (f.proofs.getD (2 * j) ByteArray.empty) ++
        Claims.Flock.zkRepColsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j + 1) #[]) (f.proofs.getD (2 * j + 1) ByteArray.empty)) :=
  Gen.cols_zk (S := S₀) (hs := hs) (seen := seen) (f := f) (ins := ins) (spec := spec) (sts := sts) (lv := lv) (j := j)
    (sch := sch) (tPad := tPad) (hf := hf) (hset := hset) (hlv := hlv) (hv := hv) (hj := hj) (hsch := hsch) (hp2 := hp2)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed masked event gives `AccTxJ`**: an accepting run of `V true` at SHA-512 decides on its messages and
coins, and on a statement of draw `d`, at any `J ≤` the statement's tables above `j`, if the run's reader looks up no
column of table `j` off `C`, `AccTxJ` holds at table `j` at the run's live outcome (`viewOf`, at any `mPts`) and its
link's root: `verify --zk` accepts what `wrS` writes for it at the setup at `d` (`acceptsAt_wrS`), and that transcript
records the run's streams and proof files (`recordsLiveAtS_wrS`), whose columns are the reader's (`cols_zk`). -/
theorem accTxJ_of_run (mPts : ℕ) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {N : ℕ} (C : Fin N → List 𝔽) (j : ℕ)
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ d {J : ℕ}, s.read.draw = some d → J ≤ sts.size → j < J →
        (¬ ∃ (p : Fin N) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧ u ≠ C p) →
        AccTxJ s.read.I d j link.1.toArray C (viewOf J mPts spec link c₁ f) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := acc_decide h
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, fun d J hd hJ hjJ hno => ?_⟩
  have hacc := run_acc_next sha512 P (V true) x ω r h
  replace hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc
  obtain ⟨hins, hsu⟩ := setupOf_some (act_decide hact)
  obtain ⟨hvr, -⟩ := decision_runs hsu hrun
  have hv' : Zk.verifyWithM (hashesIn ((Claims.run sha512 P (V true) x ω r).vLog.map Prod.fst)) spec
      (tabsOf sts) f.record f.proofs = .ok () := verdict_within hvr
  have hJ' : J ≤ spec.tables.size := by
    obtain ⟨-, -, -, hsz, -⟩ := zkVerifyWithM_tables _ hv'
    simp only [tabsOf, Array.size_map] at hsz
    omega
  have hA := acceptsAt_wrS (mPts := mPts) hact hacc hd hJ'
  refine ⟨hA, ?_, rfl⟩
  intro spec' sts' sess sch tPad s₀ s₁ hj hset' hdec' hsch hs₀ hs₁
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hsA := setupAt_of_setup hsu hzk (publicInputs_none_of_checks hc)
  rw [hd] at hsA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsA.symm.trans hset'))
  obtain ⟨hpr, hall⟩ := recordsLiveAtS_wrS s.read.I d (viewOf J mPts spec link c₁ f) hA
  obtain ⟨-, hrs⟩ := hall spec sts sess hsA hdec'
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨j, hjJ⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨j, hjJ⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V true) x ω r).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  have hcols := cols_zk _ hf (by rw [hx]; exact act_decide hact) hlv hv' hj hsch
    (HiddenExec.setupTables_pointsAny hsA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (q : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[q]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD q #[])[i]?.map (fun rd : Round => rd.coins) := by
    intro q y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS s.read.I d (viewOf J mPts spec link c₁ f)).proofs = f.proofs := hpr
  rw [hpr', zkRepColsOf_congr _ _ (hco (2 * j) _ hr₀), zkRepColsOf_congr _ _ (hco (2 * j + 1) _ hr₁),
    ← getD_empty, ← getD_empty, ← hcols]
  exact hno

end Gen

/-- **The deployed masked event gives `AccTxJ`**: an accepting run of `V true` at SHA-512 decides on its messages and
coins, and on a statement of draw `d`, at any `J ≤` the statement's tables above `j`, if the run's reader looks up no
column of table `j` off `C`, `AccTxJ` holds at table `j` at the run's live outcome (`viewOf`, at any `mPts`) and its
link's root: `verify --zk` accepts what `wrS` writes for it at the setup at `d` (`acceptsAt_wrS`), and that transcript
records the run's streams and proof files (`recordsLiveAtS_wrS`), whose columns are the reader's (`cols_zk`). -/
theorem accTxJ_of_run (mPts : ℕ) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {N : ℕ} (C : Fin N → List 𝔽) (j : ℕ)
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ d {J : ℕ}, s.read.draw = some d → J ≤ sts.size → j < J →
        (¬ ∃ (p : Fin N) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧ u ≠ C p) →
        AccTxJ s.read.I d j link.1.toArray C (viewOf J mPts spec link c₁ f) :=
  Gen.accTxJ_of_run (S := S₀) (mPts := mPts) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (N := N) (C := C) (j := j)
    (h := h)

end FlockDeploy.KS
