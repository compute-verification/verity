module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Steps.NextJ

/-! # What the plain KS proofs read of `Claims.Recursion.NextP`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- **The prover's last message with the proofs decoded**: the statement, the record's parsed value and each rep's
proof, in `spec.repStreams`' order. -/
structure FinalP where
  stmt : Claims.Msg
  j : Json
  ps : Array Flock.Proof

/-- The shape rep `k`'s proof file is read at, at the link points `pts`: table `k / 2`'s, on its `fast100` schedule,
with its extra claims at `pts` (`Flock.repFileShape`). -/
def shapeAt (tabs : Array Flock.Setup) (pts : Array (Array F128)) (k : ℕ) : Flock.Shape :=
  Flock.repFileShape tabs[k / 2]! ((fast100 tabs[k / 2]!.m).getD default)
    (match tabs[k / 2]!.extra pts with
     | .ok e => e.size
     | .error _ => 0)

/-- The proofs' files: each proof's one encoding at its shape (`Wire.proofFile`). -/
def filesAt (tabs : Array Flock.Setup) (pts : Array (Array F128)) (ps : Array Flock.Proof) : Array ByteArray :=
  ps.mapIdx fun k p => Wire.proofFile (shapeAt tabs pts k) p

/-- **The decision at a session with the proofs decoded**: `decisionAtJ` at the proofs' files, at the live first coin's
link points. -/
def decisionAtP (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    Claims.OCV Unit :=
  decisionAtJ s spec sts x link c₁ mids ⟨f.stmt, f.j, filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩

end FlockDeploy
