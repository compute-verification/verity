import Definitions
import Models
import Properties.Pouw
import Properties.Pouw.Check
import Properties.Pouw.Commit
import Properties.Pouw.LiveSalt
import Properties.Pouw.Period
import Properties.Pouw.Strata
import Properties.Pouw.Credit
import Properties.Nci
import Properties.Nci.Threads
import Properties.CertifierDevice
import Properties.Pous
import Properties.Pous.TimedAudit
import Properties.Pous.MemoryChallenger
import Properties.Pous.EndToEnd
import Properties.Pous.Period
import Properties.NetworkCertifier
import Properties.NetworkCertifier.LinkWindow
import Properties.CertifierNetwork
import Properties.CertifierNetwork.Runs
import Properties.Identity
import Properties.NciDesc
import Properties.NciDesc.Verdict
import Properties.NciDesc.Cover
import Properties.Nci.Baseline
import Properties.SampledProofs
import Properties.SampledProofs.Draw
import Properties.SampledProofs.Halt
import Properties.Evaluate
import Properties.NetworkRecords
import Properties.Accounting.Requirement
import Properties.Accounting.Compose
import Properties.RemoteAudit
import Properties.RemoteAudit.Coins
import Properties.RemoteAudit.Pooled
import Properties.Accounting.Served
import Properties.Accounting.DeclaredCircuit
import Properties.RemoteAudit.Line
import Properties.RemoteAudit.Schedule
import Properties.RemoteAudit.Shared
import Properties.RemoteAudit.Live
import Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Core
import Proofs.Nci
import Proofs.Pous
import Proofs.Pouw
import Proofs.NetworkCertifier
import Proofs.Flock
import Proofs.Accounting
import Proofs.Hash
import Proofs.Population
import Proofs.Properties.Pouw
import Proofs.Properties.Pouw.Check
import Proofs.Properties.Pouw.Commit
import Proofs.Properties.Pouw.LiveSalt
import Proofs.Properties.Pouw.LiveSalt.Ledger
import Proofs.Properties.Pouw.LiveSalt.Canon
import Proofs.Properties.Pouw.LiveSalt.Coin
import Proofs.Properties.Pouw.Strata
import Proofs.Properties.Nci
import Proofs.Properties.CertifierDevice
import Proofs.Properties.Pous
import Proofs.Properties.Pous.TimedAudit
import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.Pous.Run
import Proofs.Properties.NetworkCertifier
import Proofs.Properties.CertifierNetwork
import Proofs.Properties.CertifierNetwork.Runs
import Proofs.Properties.Identity
import Proofs.Properties.Nci.DescVerdict
import Proofs.Properties.Nci.DescAgree
import Proofs.Properties.Nci.Walk
import Proofs.Properties.Nci.Resolve
import Proofs.Properties.Nci.BodiesOk
import Proofs.Properties.Nci.Width
import Proofs.Properties.Nci.Novel
import Proofs.Properties.Nci.BaselineSteps
import Proofs.Properties.Nci.Baseline
import Proofs.Properties.Nci.OperandsResolve
import Proofs.Properties.Nci.Agrees
import Proofs.Properties.SampledProofs
import Proofs.Properties.SampledProofs.Halt
import Proofs.Properties.Evaluate
import Proofs.Properties.NetworkRecords
import Proofs.Properties.Accounting.Requirement
import Proofs.Properties.RemoteAudit
import Proofs.Properties.RemoteAudit.Coins
import Proofs.Properties.RemoteAudit.Frozen
import Proofs.Properties.RemoteAudit.Order
import Proofs.Properties.RemoteAudit.Pooled
import Proofs.Properties.Accounting.Served
import Proofs.Properties.RemoteAudit.Line
import Proofs.Properties.RemoteAudit.Schedule
import Proofs.Properties.RemoteAudit.Shared
import Proofs.Properties.RemoteAudit.SharedLink
import Proofs.Properties.RemoteAudit.Live
import Proofs.Properties.Nci.ReadsExact
import Proofs.Properties.Nci.BodiesOk.ReadsDecoded

/-! Scratch: for each declaration of a `Definitions.*` module, the declarations of the workspace that use it. -/

open Lean

def generated (s : String) : Bool :=
  s.startsWith "_" || s.startsWith "match_" || s.startsWith "proof_" || s.startsWith "eq_" ||
  ["mk", "rec", "recOn", "casesOn", "noConfusion", "noConfusionType", "below", "brecOn", "binductionOn", "ibelow",
   "ctorIdx", "inj", "injEq", "sizeOf_spec"].contains s

partial def parentOf : Name → Name
  | n@(.str p s) => if generated s && !p.isAnonymous then parentOf p else n
  | n@(.num p _) => if p.isAnonymous then n else parentOf p
  | n => n

def ours (m : Name) : Bool :=
  let s := m.toString
  s.startsWith "Definitions" || s.startsWith "Properties" || s.startsWith "Proofs"

def kindOf : ConstantInfo → String
  | .defnInfo _ => "def" | .thmInfo _ => "theorem" | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec" | .opaqueInfo _ => "opaque" | .axiomInfo _ => "axiom" | .quotInfo _ => "quot"

#eval show CoreM Unit from do
  let env ← getEnv
  let mods := env.header.moduleNames
  let modOf (c : Name) : Option Name := (env.getModuleIdxFor? c).map fun i => mods[i.toNat]!
  -- used Definitions declaration (parent) -> users (parent, module)
  let mut uses : Std.HashMap Name (Std.HashSet (Name × Name)) := {}
  let mut kinds : Std.HashMap Name (String × Name) := {}
  for (c, info) in env.constants.map₁.toList do
    let some um := modOf c | continue
    if !ours um then continue
    let pc := parentOf c
    if um.toString.startsWith "Definitions" && pc == c then
      kinds := kinds.insert c (kindOf info, um)
    for u in info.getUsedConstantsAsSet.toList do
      let some dm := modOf u | continue
      if !dm.toString.startsWith "Definitions" then continue
      let pu := parentOf u
      if pu == pc then continue
      uses := uses.insert pu ((uses.getD pu {}).insert (pc, um))
  for (c, (k, m)) in kinds.toList do
    let all := (uses.getD c {}).toList
    let defsUsers := all.filter fun (_, um) => um.toString.startsWith "Definitions"
    let others := all.filter fun (_, um) => !um.toString.startsWith "Definitions"
    let us := defsUsers.take 150 ++ others.take 20
    let line := us.foldl (fun acc (p, um) => acc ++ s!"\t{um}:{p}")
      s!"DEFUSE\t{c}\t{k}\t{m}\t{defsUsers.length}\t{others.length}"
    IO.println line
