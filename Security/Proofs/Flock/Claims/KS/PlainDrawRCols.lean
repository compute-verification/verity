import Proofs.Flock.Claims.KS.PlainDrawJCols
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.Recursion.NextR

/-!
# The plain verdict at the draw from an accepting `nextR` step, at its reader's columns

`plainVerdictAtJK_of_nextJ_cols` at the deployed reader's last step (`FlockDeploy.nextR`, decision `decisionSF`).
`nextR` decides on what `decR` reads: the last message's statement and value (`rd`), and the reps' proofs read from
the opened rounds (`readPs`), not proof files the message carries. Its verdict (`decisionAtS_evalJ`) is `nextJ`'s at
the `nextJ` final `finalOfR H spec sts c₁ f`, and differs from `nextJ`'s at `f`'s parts in two places only:

* the proof files are the read proofs' encodings, `filesAt (tabsOf sts) (pointsOf spec c₁) f.ps`, for the message's;
* the value is `jAt H spec sts c₁ mids f`, `coinSubJ` of `shaSubJ spec ds f.j` with `ds` those files' digests at
  `hOf H`, for `coinSubJ` of the message's value.

The checks after the verdict read `⟨f.stmt, f.j, #[]⟩`, which is `checksAtJ` at `finalOfR` (`checksAtJ_finalOfR`):
they read the statement and the draw, and `shaSubJ` keeps the draw. So `nextJ`'s lemmas at a parsed value apply at
that value and those files unchanged (`checksAtJ_ok`, `decodeJWithK_live`, `liveOfJ_points`,
`recordsLiveAtDJK_of_checksJ`, `txNoOffAtJK_of_records`):

* `decideR_sha512`: an accepting `nextR` step at SHA-512 is `decideJ_sha512` at `finalOfR sha512`.
* `runOfR`, `colsR`: the column reader at a `nextR` decision (`runOfJ`, `colsJ`), with `liveOfJ` at
  `jAt sha512` and the read proofs' files, which are what `nextR`'s verdict reads.
* `colsR_plain`, `txNoOffAtJK_of_colsR`, `plainVerdictAtJK_of_nextR_cols`: `colsJ_plain`, `txNoOffAtJK_of_colsJ` and
  `plainVerdictAtJK_of_nextJ_cols` there.
* `VR`, `plainVerdictAtJK_of_runR`: the run-level form, at the rounds of `V false` with the steps `nextR rd`.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-- **The `nextJ` final a `nextR` decision's verdict reads**: the statement, the value with the read proofs' digests at
`hOf H` substituted, and those proofs' files. -/
def finalOfR (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (f : FinalP) :
    FinalJ :=
  ⟨f.stmt, shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map (FlockComplete.dgOf spec (Claims.hOf H)))
    f.j, filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩

theorem jAt_finalOfR (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    coinSubJ spec c₁ mids (finalOfR H spec sts c₁ f).j = jAt H spec sts c₁ mids f := rfl

/-- **`nextR`'s checks are `checksAtJ` at `finalOfR`** (`checksAtJ_shaSubJ`). -/
theorem checksAtJ_finalOfR (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalP) (v : Json) :
    checksAtJ s spec x link c₁ mids (finalOfR H spec sts c₁ f) v =
      checksAtJ s spec x link c₁ mids ⟨f.stmt, f.j, #[]⟩ v :=
  checksAtJ_shaSubJ s spec x link c₁ mids f.stmt _ f.j _ v

/-- `nextR` decides only on a statement that sets up plain, at its setup (`actJ_decide`). -/
theorem actR_setupOf {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    setupOf false x = some (s, spec, sts) := by
  obtain ⟨⟨s', spec', sts'⟩, _, _, _, _, _, _, hv, -, -, -, -, -, hd⟩ := decR_some (actR_decide h)
  simp only [DecR.mk.injEq] at hd
  obtain ⟨rfl, rfl, rfl, -⟩ := hd
  exact hv

end FlockDeploy

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open Lean (Json)

/-- **An accepting `nextR` step at SHA-512**: the statement sets up at `(spec, sts)`, plain and without
`--public-inputs`, `verifyJWithMK id` at SHA-512 accepts `jAt sha512` and the read proofs' files there, and `checksAtJ`
passes there (`nextR_acc_iff`, `decisionAtS_evalJ`; `decideJ_sha512` at `finalOfR sha512`). -/
theorem decideR_sha512 {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) :
    verifyJWithMK id With.std spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) = .ok () ∧
      checksAtJ s spec x link c₁ mids ⟨f.stmt, f.j, #[]⟩ (jAt sha512 spec sts c₁ mids f) = .ok () ∧
      s.setup = .ok (spec, sts) ∧ s.read.zk = false ∧ s.read.publicInputs = none := by
  obtain ⟨spec', sts', hsu, hd⟩ := (nextR_acc_iff sha512 ha).1 hacc
  obtain ⟨hins, hs⟩ := setupOf_some (actR_setupOf ha)
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at hsu
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsu)
  obtain ⟨⟨⟩, hv, hc⟩ := FlockSoundness.Refine.ebind_ok
    ((decisionAtS_evalJ sha512 s spec sts x link c₁ mids (stmt := f.stmt) (j := f.j) (ps := f.ps)).symm.trans hd)
  have e : (_root_.Flock.Hashes.ofSha512 (Claims.hOf sha512) : _root_.Flock.Hashes _root_.Flock.V) = With.std := by
    rw [hOf_sha512]
  rw [e, verifyJWithM_eq_verifyJWithMK] at hv
  exact ⟨hv, hc, hs, readOf_zk (x := x) (by simp [readOf, hins]), publicInputs_none_of_check (checksAtJ_scope hc)⟩

end Claims.Bridge.PlainDraw

namespace FlockDeploy

open Flock

/-- **What the column reader takes of a `nextR` decision's table `j`** (`runOfJ` at `nextR`'s verdict): the table's
`m`, `kLog`, claim count and Merkle widths at the link points `liveOfJ` reads at `jAt sha512`, and its two reps' rounds
there and the read proofs' files. None where the step does not decide. -/
noncomputable def runOfR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (j : ℕ) : Option Claims.Flock.ZkRun :=
  match actR rd x as cs with
  | .decide _ spec sts _ c₁ mids f => do
    let st ← sts[j]?
    let tab := Setup.ofCircuit st
    let lv ← (liveOfJ spec (jAt sha512 spec sts c₁ mids f)).toOption
    let extras ← (tab.extra lv.points).toOption
    pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
           saltLen := saltLenOf false tab,
           reps := [(lv.streams.getD (2 * j) #[],
                     (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j) ByteArray.empty),
                    (lv.streams.getD (2 * j + 1) #[],
                     (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j + 1) ByteArray.empty)] }
  | _ => none

/-- **The opened level-0 columns of a `nextR` decision's table `j`** (`colsJ` at `nextR`). -/
noncomputable def colsR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (j : ℕ) : List (ℕ × List FlockLevel3.GF128) :=
  match runOfR rd x as cs j with
  | some r => Claims.Flock.plainRunCols r
  | none => []

/-- **The deployed reader's rounds**: `V false`'s round count, with the steps `nextR rd`. -/
noncomputable def VR (rd : Claims.Msg → Option FinalP) : Claims.Verifier Coin where
  k := (FlockDeploy.V false).k
  next := nextR rd

end FlockDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A `nextR` step's reader looks up table 0's columns at `TxNoOffAtJK`'s shape** (`colsJ_plain` at `nextR`): where
the step decides at `(spec, sts)` with first table `st` and two link points, `liveOfJ` reads `jAt sha512` and the
verdict accepts it with the read proofs' files, the reader's table-0 columns are `plainRepColsOf` at `shapeOf` with
`2 · |regions|` extra claims, on `liveOfJ`'s first two rep streams and those files, below level 0's `2^d`. -/
theorem colsR_plain {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {st : Stmt} {sch : Schedule}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv)
    (hv : verifyJWithMK id With.std spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) = .ok ())
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) :
    colsR rd x as cs 0 = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 1 ByteArray.empty)) := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨sess, schs, hdec, -, -, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
  have h0' : 0 < (tabsOf sts).size := by simpa [tabsOf] using h0
  obtain ⟨_, _, hf, hrep, -⟩ := hall 0 h0'
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit sts[0] := tabsOf_getElem h0'
  simp only [ht0] at hf hrep
  obtain ⟨-, extras, -, -, hex, -⟩ := verifyRepWith_rounds With.std rfl hf hrep
  obtain ⟨-, hpts, -, -⟩ := decodeJWithK_live id With.std hdec hlv
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, liveOfJ_points spec c₁ mids _ hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOfR rd x as cs 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 0
                    ByteArray.empty),
                 (lv.streams.getD 1 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 1
                    ByteArray.empty)] } := by
    unfold runOfR
    simp [ha, hst, hlv, hex, Except.toOption]
  unfold colsR
  rw [hrun]
  simp only [Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape, Claims.Flock.plainLevel0, hsched,
    FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons, List.flatMap_nil, List.append_nil]

/-- **The step's reader gives the reader's premise at its parsed session** (`txNoOffAtJK_of_colsJ` at `nextR`): on an
accepting `nextR` step at SHA-512 whose statement carries the draw `d`, if the step's reader looks up no column off
`C` at table 0 (`colsR … 0`), the deployed plain reader looks up none in the session the verdict reads, `jAt sha512`
with the read proofs' files, at the setup at `d` (`TxNoOffAtJK id`). -/
theorem txNoOffAtJK_of_colsR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} {d : Json} {N : ℕ} (C : Fin N → List 𝔽)
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    TxNoOffAtJK id s.read.I d C
      ⟨jAt sha512 spec sts c₁ mids f, FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩ := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideR_sha512 ha hacc
  have hsetA : setupAtD s.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd
  intro spec' st sess sch s₀ s₁ hset hdec hsch hs₀ hs₁
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hset))
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  have hcols := colsR_plain (st := st) ha hlv hv (by simp) hsch (HiddenExec.setupTables_pointsAny hsetA).1
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyJWithMK_tables_cap id With.std hv
  have htab : 0 < spec.tables.size := by rw [← htsz]; exact hpos
  obtain ⟨-, -, -, hstr⟩ := decodeJWithK_live id With.std hdec hlv
  have hco : ∀ (r : ℕ), r < 2 → ∀ y : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some y →
      ∀ k : ℕ, y.rounds[k]?.map (fun a : Round => a.coins) =
        (lv.streams.getD r #[])[k]?.map (fun a : Round => a.coins) := by
    intro r hr y hy k
    obtain ⟨hq, hw⟩ := Claims.Bridge.Custody.repStreams_getElem spec (i := 0) (r := r) htab hr
    obtain ⟨y', hy', hyr⟩ := hstr _ hq
    rw [hw] at hy'
    obtain rfl : y' = y := Option.some.inj (hy'.symm.trans hy)
    have e := forall₂_coins (fun _ _ h => h.2.1) hyr k
    simp only [Array.getElem?_toList] at e
    rw [e, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp
  show ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow _
    (Claims.Flock.plainRepColsOf _ s₀.rounds (FlockDeploy.filesAt (tabsOf #[st]) (pointsOf spec c₁) f.ps)[0]! ++
      Claims.Flock.plainRepColsOf _ s₁.rounds (FlockDeploy.filesAt (tabsOf #[st]) (pointsOf spec c₁) f.ps)[1]!))
      p.val = some u ∧ u ≠ C p
  rw [plainRepColsOf_congr _ _ (hco 0 (by omega) s₀ hs₀), plainRepColsOf_congr _ _ (hco 1 (by omega) s₁ hs₁),
    ← getD_empty, ← getD_empty, ← hcols]
  exact hno

/-- **An accepting `nextR` step gives the verdict at the draw, at its reader's columns** (`plainVerdictAtJK_of_nextJ`
and `plainVerdictAtJK_of_nextJ_cols` at `nextR`): at SHA-512, on a step whose statement carries the draw `d`, if the
step's reader looks up no column off `C` at table 0 (`colsR … 0`), `plainVerdictAtJK id` holds at the step's live
outcome (`viewOfJ` at `jAt sha512` and the read proofs' files, at any `mPts`) and its link's root. `verifyJWithMK id`
accepts that session at the setup at `d` (`decideR_sha512`), which records the outcome (`recordsLiveAtDJK_of_checksJ`
at `finalOfR sha512`), so it accepts what `wrPDJK` writes, and the reader's premise (`txNoOffAtJK_of_colsR`) moves
there (`txNoOffAtJK_of_records`). -/
theorem plainVerdictAtJK_of_nextR_cols (mPts : ℕ) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} {d : Json}
    {N : ℕ} (C : Fin N → List 𝔽) (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    plainVerdictAtJK id s.read.I d C link.1.toArray
      (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideR_sha512 ha hacc
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyJWithMK_tables_cap id With.std hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  have hc' : checksAtJ s spec x link c₁ mids (finalOfR sha512 spec sts c₁ f) (jAt sha512 spec sts c₁ mids f) =
      .ok () := (checksAtJ_finalOfR sha512 s spec sts x link c₁ mids f _).trans hc
  have hA₀ : AcceptsAtDJK id s.read.I (some d)
      ⟨jAt sha512 spec sts c₁ mids f, FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩ :=
    ⟨spec, sts, setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd, hv⟩
  have hR₀ := recordsLiveAtDJK_of_checksJ id (J := 1) (mPts := mPts) hsu hzk hpi hd hJ hc'
  have hA := acceptsAtDJK_wrPDJK hA₀ hR₀
  exact ⟨hA, rfl, txNoOffAtJK_of_records hA₀ hR₀ hA (recordsLiveAtDJK_wrPDJK _ _ _ _)
    (txNoOffAtJK_of_colsR C ha hacc hd hno)⟩

/-- **An accepted run of the deployed reader decides, and gives the verdict at the draw at its reader's columns**: a
run of `VR rd` that accepts at SHA-512 decides on its messages and coins (`actR`), and wherever the decided statement
carries the draw `d` and the step's reader looks up no column off `C` at table 0, `plainVerdictAtJK id` holds at the
step's live outcome (`plainVerdictAtJK_of_nextR_cols`). -/
theorem plainVerdictAtJK_of_runR (mPts : ℕ) (rd : Claims.Msg → Option FinalP) {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ (d : Json), s.read.draw = some d → ∀ {N : ℕ} (C : Fin N → List 𝔽),
        (¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x
          ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) 0)
            p.val = some u ∧ u ≠ C p) →
        plainVerdictAtJK id s.read.I d C link.1.toArray
          (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
            (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) := by
  have ha := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha ⊢
  replace ha : ((nextR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs).eval sha512).1 = .acc := ha
  cases hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    exact ⟨s, spec, sts, link, c₁, mids, f, rfl, fun d hd _ C hno =>
      plainVerdictAtJK_of_nextR_cols mPts C hact ha hd hno⟩

end FlockDeploy.KS
