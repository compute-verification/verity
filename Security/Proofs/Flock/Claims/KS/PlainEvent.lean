import Proofs.Flock.Claims.KS.PlainWrP
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# The deployed event at `wrP`'s verdict

`PlainLiveWrPAt`'s first clause bounds the deployed runs that accept with no column off `C` by the live strategies'
`plainVerdict`. On each such run that event holds at the run's live outcome (`viewOf`), so what is left of the clause
is the transport of the run's coins to the live session's:

* `cols_plain`: an accepted plain run's reader looks up table 0's columns at the shape and level 0 `TxNoOff` reads, in
  its live session's rep streams and its proof files.
* `plainVerdict_of_run`: an accepting run of `V false` at SHA-512 decides, and on a statement of draw `d` (which the
  record's draw then is, as a value: `recordDraw_of_decide`), a run whose reader looks up no column off `C` has
  `plainVerdict` at its live outcome and its link's root.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **An accepted plain run's reader looks up table 0's columns at `TxNoOff`'s shape**: where the run's last message
`f` sets up at `(spec, sts)` with its first table `st` and two link points, the plain verdict accepts its record and
proof files at any hashes and `liveOf` reads the record, the reader's table-0 columns are `plainRepColsOf` at
`shapeOf` with `2 · |regions|` extra claims, on the live session's first two rep streams and proof files, below level
0's `2^d` of `st`'s `fast100` schedule. -/
theorem cols_plain (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {st : Stmt} {sch : Schedule} (hf : lastFinal seen = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hst : sts[0]? = some st)
    (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) :
    cols false seen 0 = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty)) := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0' : sts[0]! = sts[0] := getElem!_pos sts 0 h0
  obtain ⟨hsz, hall⟩ := rep_plain hs hv hlv
  have hk : 0 < lv.streams.size := by omega
  obtain ⟨sess, -, -, -, hpts, hpn, -⟩ := hall 0 hk
  obtain ⟨extras, -, -, hex, -⟩ := queryRound_plain hs hv hlv hk (tab := Setup.ofCircuit sts[0])
    (by rw [Nat.zero_div, h0'])
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOf false seen 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                 (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, hst, hlv, hex, Except.toOption]
  unfold cols
  rw [hrun]
  simp only [Bool.false_eq_true, ↓reduceIte, Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape,
    Claims.Flock.plainLevel0, hsched, FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons, List.flatMap_nil,
    List.append_nil]

/-- **The deployed event gives `wrP`'s verdict**: an accepting run of `V false` at SHA-512 decides on its messages and
coins, and on a statement of draw `d`, if the run's reader looks up no column off `C` at table 0, `plainVerdict` holds
at the run's live outcome (`viewOf`, at any `mPts`) and its link's root: the checks refuse `--public-inputs`
(`publicInputs_none_of_checks`), flock-verify accepts what `wrP` writes for the run (`accepts_wrP`), and that
transcript records the run's streams and proof files (`RecordsLiveAtP`), whose columns are the reader's
(`cols_plain`). -/
theorem plainVerdict_of_run (mPts : ℕ) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {N : ℕ} (C : Fin N → List 𝔽)
    (h : (Claims.run sha512 P (V false) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ d, s.read.draw = some d →
        (¬ ∃ (p : Fin N) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) →
        plainVerdict s.read.I d C link.1.toArray (viewOf 1 mPts spec link c₁ f) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := acc_decide h
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, fun d hd hno => ?_⟩
  have hacc := run_acc_next sha512 P (V false) x ω r h
  replace hacc : ((next false x ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  have hdraw := recordDraw_of_decide hact hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu⟩ := setupOf_some (act_decide hact)
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetE : Exec.setupOf s.read.I f.record = .ok (spec, sts) :=
    setupOf_plainInputs (Ins.setup_read hsu) hzk hpi hdraw
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verify_ok_tables hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  have hA := accepts_wrP (mPts := mPts) (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) (S := ∅) (R := ()) hact hacc
    hpi hd hJ
  refine ⟨hA, rfl, ?_⟩
  intro spec' st sess sch s₀ s₁ hset' hdec' hsch hs₀ hs₁
  obtain ⟨hdr, hpr, hall⟩ := (recordsLiveJ_iff _ _ _ _).1
    (recordsLiveJ_wrP (mPts := mPts) s.read.I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1)) ∅ ()
    (viewOf 1 mPts spec link c₁ f) hA
  have e : Exec.setupOf s.read.I (wrP s.read.I (fun _ : Finset (Fin 0) => d) (fun _ _ => 1) ∅ ()
      (viewOf 1 mPts spec link c₁ f)).record = Exec.setupOf s.read.I f.record := by
    unfold Exec.setupOf
    rw [hdr, hdraw, hd]
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetE.symm.trans (e.symm.trans hset')))
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
    (HiddenExec.setupTables_pointsAny hsetE).1
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
