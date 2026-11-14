module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP
public import Proofs.Flock.Soundness.Discharge.Exec.Pts
public import Flock.Desc

/-!
# Scratch probe, not for landing: the statements of a member's reference setup without its session

Statements only, each proof left out. This file is on a scratch branch and never goes to a branch that lands.

A member the sample holds no unit of has no session, so nothing runs its setup. Its reference setup is taken at the
sample of no unit (`emptyDraw`), where the setup reads the member's public file only in loading it, in the header's
`frame_v3` and in the roots' recomputation (`HmOut.checkPublic`); every other step reads the circuit file, the tags,
the lookup tables, the partition object and the program, which a sibling (a statement of the same circuit side at
another public file) shares.

* `setupHidden_sibling`: the setup at the member's public file and the sample of no unit, from a sibling's setup of the
  least count of link points and the member's public file's own checks.
* `y₀_of_sibling`: the same in the shape of the window's reference setup (`DrawSetupZHJP` at `mPtsOf g 0`).
* `y₀_of_filesHeld`: the same on the registration's files check (`StepAt.filesHeld`) and a sibling's session
  (`sessionAt`), leaving as hypotheses exactly what the files check does not hold.
* `drawSetupZHJP_subsingleton`: a reference setup at given inputs, sampled proof blocks and sample is unique.
-/

@[expose] public section

set_option Elab.async false

namespace Verity.Sampled.SetupProbe

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Recursion.Witness (toExec)

/-- The sample of no unit over a population of `N` units, as `Draw.UnitDraw.toJson` prints a subset sample. -/
def emptyDraw (N : ℕ) : Json := Flock.Draw.UnitDraw.toJson ⟨.subset 0, N, #[], #[]⟩

/-- **The setup at a sibling's public file, at the sample of no unit.** `h₁`, `hz₁`: a statement of the same tags,
circuit file, coins, lookup tables, partition object and program set up at some public file `pf₁` and sample `d₁`
and passed `Zk.shapeOk`; `hcls`: at the least count of link points. `hpop`, `hkey`, `hpub`: the public file `pf₂`
loads, its header names no public inputs, and its `frame_v3` and roots are its rows' (`HmOut.checkPublic` on the
population's file). Then the statement sets up at `pf₂` and the sample of no unit, passes `Zk.shapeOk`, and is of the
sibling's circuit and count of link points. -/
theorem setupHidden_sibling {tags : Flock.Tags} {cf pf₁ pf₂ : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {partition : Option Json} {program : Option ByteArray} {d₁ : Json}
    {st₁ : Flock.Stmt} {pop : Flock.Public}
    (h₁ : Flock.Stmt.setupHidden tags cf pf₁ coins tables (some d₁) partition program none = .ok st₁)
    (hz₁ : Flock.Zk.shapeOk st₁ = .ok ()) (hcls : st₁.nbl = 3)
    (hpop : Flock.HmOut.loadPublic tags st₁.c pf₂ = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : Flock.HmOut.checkPublic st₁.c pop = .ok ()) :
    ∃ st₂, Flock.Stmt.setupHidden tags cf pf₂ coins tables (some (emptyDraw pop.n)) partition program none = .ok st₂ ∧
      Flock.Zk.shapeOk st₂ = .ok () ∧ st₂.c = st₁.c ∧ st₂.nbl = 3 := by
  sorry

open FlockSoundness.Discharge.Composed (inputsJ nTab) in
/-- **A member's reference setup from a sibling's**, in the shape `y₀_of_setup` gives a sampled member's: at one
table, the inputs `I` set up a `--zk` table at the sample of no unit and `mPtsOf g 0` points, where the same inputs at
another public file `pf₁` set up at some sample at the least count of link points and `I`'s public file passes its
own checks. -/
theorem y₀_of_sibling (n : ℕ) {I : FlockSoundness.Discharge.Exec.Inputs} {D : Flock.Desc} {pf₁ : ByteArray}
    {d₁ : Json} {st₁ : Flock.Stmt} {pop : Flock.Public} (hc1 : I.count = 1) (ht : I.tags.typed = false)
    (hD : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D)
    (hst₁ : Flock.Stmt.setupHidden (inputsJ { I with publicFile := pf₁ } 0).tags
      (inputsJ { I with publicFile := pf₁ } 0).circuitFile (inputsJ { I with publicFile := pf₁ } 0).publicFile
      (inputsJ { I with publicFile := pf₁ } 0).coins (inputsJ { I with publicFile := pf₁ } 0).tables (some d₁)
      (inputsJ { I with publicFile := pf₁ } 0).partition (inputsJ { I with publicFile := pf₁ } 0).program none
        = .ok st₁)
    (hshape₁ : Flock.Zk.shapeOk st₁ = .ok ()) (hcls : st₁.nbl = 3)
    (hpop : Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) st₁.c I.publicFile = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : Flock.HmOut.checkPublic st₁.c pop = .ok ()) :
    ∃ (S₀ : Finset (Fin n)) (d₀ : Json),
      Nonempty (FlockSoundness.Discharge.ZkHidden.DrawSetupZHJP (inputsJ I 0) none
        (FlockSoundness.Discharge.Exec.mPtsOf D.unroll.g (0 / nTab I)) S₀ d₀) := by
  sorry

open FlockSoundness.Discharge.Composed (inputsJ nTab) in
/-- **A member's reference setup on the registration's files check.** `hfiles`: the member's statement passed the
files check the registration runs on every member, sampled or not (`StepAt.filesHeld`). `hsess`, `hcls`: a session of
the same own inputs, tags, circuit file and blobs at another public file `pubB₁` set up (`sessionAt`, which runs
`Zk.shapeOk`) at the least count of link points. `hpop` names the member's public file as it loads, which `hfiles`
holds that it does; `hkey` and `hpub` are what the files check does not hold. -/
theorem y₀_of_filesHeld (n : ℕ) (p : _root_.Spot.OneStageZk.Own) (tags0 : Flock.Tags)
    (circuitB pubB pubB₁ : ByteArray) (bs : _root_.Core.Blobs) (rd rd₁ : _root_.Spot.OneStageZk.Reads)
    (population : Flock.V Unit) {D : Flock.Desc} {d₁ : Json} {spec₁ : Flock.SessionSpec} {st₁ : Flock.Stmt}
    {pop : Flock.Public}
    (hfiles : (_root_.Spot.OneStageZk.StepAt.ofOwn p tags0 circuitB pubB bs rd population).filesHeld = .ok ())
    (ht : tags0.typed = false)
    (hD : Flock.Desc.parse
      (FlockSoundness.Discharge.ZkExec.zkTags (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)))
      (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)).circuitFile
      (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)).tables = .ok D)
    (hsess : _root_.Spot.OneStageZk.sessionAt (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB₁ bs) rd₁
      (some d₁) = .ok (spec₁, #[st₁]))
    (hcls : st₁.nbl = 3)
    (hpop : Flock.HmOut.loadPublic
      (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)) st₁.c pubB = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : Flock.HmOut.checkPublic st₁.c pop = .ok ()) :
    ∃ (S₀ : Finset (Fin n)) (d₀ : Json),
      Nonempty (FlockSoundness.Discharge.ZkHidden.DrawSetupZHJP
        (inputsJ (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)) 0) none
        (FlockSoundness.Discharge.Exec.mPtsOf D.unroll.g
          (0 / nTab (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)))) S₀ d₀) := by
  sorry

/-- **A reference setup is unique**: at given inputs, public inputs, count of link points, sampled proof blocks and
sample, any two are equal, each field a value the setup, the sample's decoding or the schedule returns. -/
theorem drawSetupZHJP_subsingleton {n : ℕ} (I' : FlockSoundness.Discharge.Exec.Inputs) (own : Option ByteArray)
    (mPts : ℕ) (S : Finset (Fin n)) (d : Json) :
    Subsingleton (FlockSoundness.Discharge.ZkHidden.DrawSetupZHJP I' own mPts S d) := by
  sorry

end Verity.Sampled.SetupProbe
