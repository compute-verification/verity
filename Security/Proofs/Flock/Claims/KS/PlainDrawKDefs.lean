import Proofs.Flock.Claims.KS.PlainDraw
import Proofs.Flock.Claims.Recursion.RepErase
import Proofs.Flock.Soundness.Discharge.Exec.VerifyK

/-!
# The plain verdict at the statement's draw, at `VK keep`: the definitions

`PlainDraw`'s predicates at every `keep`, with no facts about them: flock-verify's acceptance is `verifyWithMK keep` at
SHA-512 (`With.std`), the record decodes through `Record.decodeWithK keep`, and a recorded round is the live outcome's
up to its digest (`Round.erase`), since at `keep := fun _ => true` the decoder drops S12's comparison of a retained
round's bytes with its recorded digest. At `keep := id` each is `PlainDraw`'s.

* `AcceptsAtDK`, `RecordsLiveAtDK`, `TxNoOffAtK`, `wrPDK`: `AcceptsAtD`, `RecordsLiveAtD`, `TxNoOffAt` and `wrPD`
  through `keep`.
* `plainVerdictAtK`: `plainVerdictAt` at `wrPDK`.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Refine (ptsOf shapeOf)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify accepts at the setup at the draw `dr`, its digest comparisons read through `keep`**: `AcceptsAtD`
with `verifyWithMK keep` at SHA-512 for `verify`. -/
def AcceptsAtDK (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupAtD I dr = .ok (spec, sts) ∧
    verifyWithMK keep With.std spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

/-- **`RecordsLiveAtD` through `keep`, up to the rounds' digests**: where `verifyWithMK keep` accepts the transcript `t`
at the session it sets up at `d`, its proof files are the outcome `o`'s, and the session its record decodes to through
`keep` there has `o`'s link points and, at each of `o`'s tables, `o`'s `root_B` and rep streams' rounds up to their
digests (`Round.erase`). -/
def RecordsLiveAtDK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts)
    (t : Exec.Transcript) : Prop :=
  AcceptsAtDK keep I (some d) t →
    t.proofs = o.proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decodeWithK keep With.std spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList.map Round.erase = (o.reps j r).map Round.erase

/-- **`TxNoOffAt` through `keep`**: no column off `C` at the session the record decodes to through `keep`. -/
def TxNoOffAtK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N : ℕ} (C : Fin N → List 𝔽) (t : Exec.Transcript) :
    Prop :=
  ∀ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : RunRecord) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) → Record.decodeWithK keep With.std spec t.record = .ok sess →
    _root_.Flock.fast100 st.m = some sch →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]!)) p.val = some u ∧ u ≠ C p

open Classical in
/-- **The record writer at the draw `d`, through `keep`**: at the live outcome `o`, a transcript `verifyWithMK keep`
accepts at the setup at `d` that records `o` there up to the rounds' digests (`RecordsLiveAtDK`), where one exists,
else a session without proof files, which the verdict refuses at every `keep`. -/
noncomputable def wrPDK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) :
    Exec.Transcript :=
  if h : ∃ t, AcceptsAtDK keep I (some d) t ∧ RecordsLiveAtDK keep I d o t then h.choose else ⟨"", #[]⟩

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain verdict at the draw `d`, through `keep`**: `verifyWithMK keep` accepts what `wrPDK` writes for the
outcome at the setup at `d`, the outcome's root is `rootB₀`'s, and the deployed plain reader looks up no column off
`C` in that transcript there (`TxNoOffAtK`). `plainVerdictAt` at every `keep`. -/
def plainVerdictAtK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N mPts : ℕ} (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (o : Tables.LiveOutJ 1 mPts) : Prop :=
  AcceptsAtDK keep I (some d) (wrPDK keep I d o) ∧ o.rootB[0]! = rootB₀[0]! ∧ TxNoOffAtK keep I d C (wrPDK keep I d o)

end FlockDeploy.KS
