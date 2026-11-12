module

public import Proofs.Flock.Claims.Recursion.VStarNets
public import Proofs.Flock.Claims.Recursion.VStarTraceOk
public import Proofs.Flock.VBridge.Family.UnitName
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.List.SplitOn.Basic

/-!
# A staged part's circuit has three nets (`nets_three`)

`traceOk_deployed` takes `c.nets.size = 3`. It holds at every staged part: the setup's `checkStmt` holds the circuit
file to V\*'s (`checkCircuit` runs `checkFile`, so `circuitFile sv f holo = .ok circuitB`: `checkFile_ok`) and refuses
lookup tables (`tables = []`: `checkStmt_tables`), and the staged circuit is `HmRow.parse` of that file
(`table0`, `circuitOf`), whose nets `VStarNets.lean` counts (`nets_of_file`).
-/

@[expose] public section

namespace RecDeploy.VStarNets

open _root_.Flock
open Flock.HmNets (Laid shaLaid hmLaid)
open FlockVBridge.Names
open Claims (Msg)
open RecDeploy.VStarBuild (partAt progOf)
open RecDeploy.VStarTrace (TraceOk)
open RecDeploy.VStarTraceOk (RowsOk)

/-- `checkStmt` at an inner shape refuses lookup tables. -/
theorem checkStmt_tables {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) : tables = [] := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · rename_i ht
      simpa using ht

/-- **A staged part's circuit has three nets**: the unit, `sha512x3` and `hm96`. The setup's `checkStmt` holds the
part's circuit file to V\*'s (`checkFile_ok`) and refuses tables (`checkStmt_tables`), and its circuit is
`HmRow.parse`'s of that file (`table0`), the circuit `partAt` reads (`partCircuit_staged`). -/
theorem nets_three {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) : c.nets.size = 3 := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  obtain ⟨p, r', _, hx, ht, hpi, hr', _, h0, hchk⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  have hflat := FlockSoundness.Discharge.Recursion.checkParsed_flat (hchk 0 h0).2.2
  obtain ⟨tags', P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0 hflat
  rw [← hpc]
  obtain ⟨ms, i, spec, sts, rfl, hsu, hck⟩ := hS
  have hpx := parse_of_setupO hsu
  rw [hx] at hpx
  cases hpx
  have h0' : 0 < sts.size := h0
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r0, -, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hC0 := checks_stmt hC 0 h0'
  obtain ⟨j, f, sv0, hj, -, hcc⟩ := checkStmt_circuit hC0
  cases hj
  obtain ⟨hck', hfile, -, -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hcc
  obtain ⟨_, _, _, hname, _⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  have hcf := FlockSoundness.Discharge.Recursion.checkFile_ok hfile
  have htb := checkStmt_tables hC0
  have hty : i.base.read.tags.typed = false := ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1.2.1
  have hpi' := publicInputs_none_of_scope hck
  have htab : i.base.read.tablesOf = some sts := by
    unfold Flock.Readable.Read.tablesOf
    rw [RecIns.setup_read hset]
    cases hz : i.base.read.zk
    · simp
    · simp [RecIns.setup_shapeOk hset hz]
  obtain ⟨st0, hst0, hc0, -⟩ := Flock.Readable.table0 htab hty hpi'
  rw [Array.getElem?_eq_getElem h0', Option.some.injEq] at hst0
  subst hst0
  have hparse : HmRow.parse i.base.read.tags i.base.read.I.circuitFile [] = .ok sts[0].c := by
    rw [← htb]
    unfold ProvedScope.circuitOf at hc0
    split at hc0
    · obtain ⟨_, _, -, -, -, -, hor⟩ := FlockSoundness.Discharge.Hidden.parseCommit_untyped hty hc0
      have hi : i.base.read.tags.isTmpl = false := by
        rcases hor with ⟨hi, -⟩ | ⟨tu, -, -, htu⟩
        · exact hi
        · exact absurd (htu.symm.trans hflat) (by simp)
      rw [FlockSoundness.parseCommit_flat hi] at hc0
      exact (FlockSoundness.Discharge.HiddenExec.parse_untyped hty hc0).1
    · exact hc0
  exact nets_of_file hcf hparse hname P.unitNet

/-- **`TraceOk` at a staged part, given the port inputs' rows** (`traceOk_deployed` with `nets_three`). -/
theorem traceOk_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {src : VStar.In.Src} {pin : VStar.In.PartIn}
    (hi : VStar.In.partIn src (progOf c) sv = .ok pin) (hR : RowsOk c pin) : TraceOk c pin :=
  VStarTraceOk.traceOk_deployed hS hp hi (nets_three hS hp) hR

theorem partAt_lt {x : Msg} {s : ℕ} {sv : VStar.Stmt} {c : Circuit} (hp : partAt x s = some (sv, c)) :
    s < RecDeploy.nS x := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  unfold RecDeploy.nS
  rw [hr]
  cases hq : r.w.parts[s]? with
  | none => rw [hq] at hc; simp at hc
  | some q => exact (List.getElem?_eq_some_iff.1 hq).1

/-- **`TraceOk` at every part of a staged statement** (`HpEarly.TraceOkAt`'s body): each part `s < nS x` staged at
`xOf s`, and the port inputs' rows sized (`RowsOk`). -/
theorem traceOkAt_staged {I : VStarBuild.InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {h : List (Msg × Msg)} {stg : Stage} {xOf : ℕ → Msg} (hS : ∀ s < RecDeploy.nS x, StagedPart stg s x (xOf s))
    (hR : ∀ s st c src pin, partAt x s = some (st, c) → VStarBuild.srcOf I x w ρ h = some src →
      VStar.In.partIn src (progOf c) st = .ok pin → RowsOk c pin) :
    ∀ s st c src pin, partAt x s = some (st, c) → VStarBuild.srcOf I x w ρ h = some src →
      VStar.In.partIn src (progOf c) st = .ok pin → TraceOk c pin :=
  fun s st c src pin hp hsrc hi => traceOk_staged (hS s (partAt_lt hp)) hp hi (hR s st c src pin hp hsrc hi)

end RecDeploy.VStarNets
