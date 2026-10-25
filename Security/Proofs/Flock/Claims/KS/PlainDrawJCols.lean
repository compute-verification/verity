import Proofs.Flock.Claims.KS.PlainDrawJRun

/-!
# The reader's premise at a `nextJ` step, from the step's reader's columns

`plainVerdictAtJK_of_nextJ` takes the reader's premise at the step's own parsed session (`TxNoOffAtJK` at
`⟨coinSubJ spec c₁ mids f.j, f.proofs⟩`), which reads the session the element-level decoder makes of the value. The
run's reader reads the step itself: `runOf` and `cols` take the last message's record text and `liveOf`'s streams there,
and `nextJ`'s verdict reads the coin-substituted value, so its reader reads `liveOfJ` at that value:

* `runOfJ`, `colsJ`: `runOf` and `cols` (plain) at a `nextJ` decision, with `liveOfJ` at the substituted value for
  `liveOf` at the record.
* `colsJ_plain`: where the verdict accepts the substituted value, the reader's table-0 columns are `plainRepColsOf` at
  `TxNoOffAtJK`'s shape on `liveOfJ`'s first two rep streams and the proof files (`cols_plain` at a parsed value). The
  link carries the first coin's points (`liveOfJ_points`), `spec.points` of them, so the extra claims number
  `2 · |regions|`.
* `txNoOffAtJK_of_colsJ`: on an accepting step at SHA-512 whose statement carries the draw `d`, a reader that looks up
  no column off `C` gives `TxNoOffAtJK`: the decoder's rep streams have `liveOfJ`'s coins (`decodeJWithK_live`).
* `plainVerdictAtJK_of_nextJ_cols`: `plainVerdictAtJK_of_nextJ` with that premise for the reader's.
-/

namespace FlockDeploy

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **What the column reader takes of a `nextJ` decision's table `j`** (`runOf` at a parsed value): the table's `m`,
`kLog`, claim count and Merkle widths at the link points `liveOfJ` reads at the substituted value, and its two reps'
rounds there and proof files. None where the step does not decide. -/
def runOfJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) (j : ℕ) :
    Option Claims.Flock.ZkRun :=
  match actJ rd x as cs with
  | .decide _ spec sts _ c₁ mids f => do
    let st ← sts[j]?
    let tab := Setup.ofCircuit st
    let lv ← (liveOfJ spec (coinSubJ spec c₁ mids f.j)).toOption
    let extras ← (tab.extra lv.points).toOption
    pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
           saltLen := saltLenOf false tab,
           reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                    (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] }
  | _ => none

/-- **The opened level-0 columns of a `nextJ` decision's table `j`** (`cols false` at a parsed value). -/
def colsJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) (j : ℕ) :
    List (ℕ × List 𝔽) :=
  match runOfJ rd x as cs j with
  | some r => Claims.Flock.plainRunCols r
  | none => []

end FlockDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A `nextJ` step's reader looks up table 0's columns at `TxNoOffAtJK`'s shape**: where the step decides at
`(spec, sts)` with first table `st` and two link points, `liveOfJ` reads the substituted value and the verdict accepts
it with the proof files, the reader's table-0 columns are `plainRepColsOf` at `shapeOf` with `2 · |regions|` extra
claims, on `liveOfJ`'s first two rep streams and the proof files, below level 0's `2^d` (`cols_plain` at a parsed
value). -/
theorem colsJ_plain {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} {lv : Live} {st : Stmt} {sch : Schedule}
    (ha : actJ rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : liveOfJ spec (coinSubJ spec c₁ mids f.j) = .ok lv)
    (hv : verifyJWithMK id With.std spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs = .ok ())
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) :
    colsJ rd x as cs 0 = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          (f.proofs.getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          (f.proofs.getD 1 ByteArray.empty)) := by
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
    rw [extraClaims_size (st := sts[0]) hex, liveOfJ_points spec c₁ mids f.j hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOfJ rd x as cs 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                 (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
    unfold runOfJ
    simp [ha, hst, hlv, hex, Except.toOption]
  unfold colsJ
  rw [hrun]
  simp only [Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape, Claims.Flock.plainLevel0, hsched,
    FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons, List.flatMap_nil, List.append_nil]

/-- **The step's reader gives the reader's premise at its parsed session**: on an accepting `nextJ` step at SHA-512
whose statement carries the draw `d`, if the step's reader looks up no column off `C` at table 0 (`colsJ … 0`), the
deployed plain reader looks up none in the step's own parsed session at the setup at `d` (`TxNoOffAtJK id`). The
decoder's table-0 rep streams have `liveOfJ`'s coins at the substituted value (`decodeJWithK_live`), and those are the
reader's columns (`colsJ_plain`). -/
theorem txNoOffAtJK_of_colsJ {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {d : Json} {N : ℕ} (C : Fin N → List 𝔽)
    (ha : actJ rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextJ rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsJ rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    TxNoOffAtJK id s.read.I d C ⟨coinSubJ spec c₁ mids f.j, f.proofs⟩ := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideJ_sha512 ha hacc
  have hsetA : setupAtD s.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd
  intro spec' st sess sch s₀ s₁ hset hdec hsch hs₀ hs₁
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hset))
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  have hcols := colsJ_plain (st := st) ha hlv hv (by simp) hsch (HiddenExec.setupTables_pointsAny hsetA).1
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
    (Claims.Flock.plainRepColsOf _ s₀.rounds f.proofs[0]! ++ Claims.Flock.plainRepColsOf _ s₁.rounds f.proofs[1]!))
      p.val = some u ∧ u ≠ C p
  rw [plainRepColsOf_congr _ _ (hco 0 (by omega) s₀ hs₀), plainRepColsOf_congr _ _ (hco 1 (by omega) s₁ hs₁),
    ← getD_empty, ← getD_empty, ← hcols]
  exact hno

/-- **An accepting `nextJ` step gives the verdict at the draw, at its reader's columns**: `plainVerdictAtJK_of_nextJ`
with the step's reader looking up no column off `C` at table 0 (`colsJ … 0`) for the premise at its parsed session
(`txNoOffAtJK_of_colsJ`). -/
theorem plainVerdictAtJK_of_nextJ_cols (mPts : ℕ) {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {d : Json}
    {N : ℕ} (C : Fin N → List 𝔽) (ha : actJ rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextJ rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsJ rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    plainVerdictAtJK id s.read.I d C link.1.toArray
      (viewOfJ 1 mPts spec link c₁ (coinSubJ spec c₁ mids f.j) f.proofs) :=
  plainVerdictAtJK_of_nextJ mPts C ha hacc hd (txNoOffAtJK_of_colsJ C ha hacc hd hno)

end FlockDeploy.KS
