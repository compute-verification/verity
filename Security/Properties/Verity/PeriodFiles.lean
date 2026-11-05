import Properties.Verity.Game
import CorePeriodFiles

/-!
# A composite's executable with the period's circuit files read by path

A composite with the network certifier's period (onsite, and remote with the network certifier, with PoUW or without)
is stated over its verifier of record (`verifier`), which finds the circuit files the period lists among its blobs
(`NetworkCheck.Period.blobProtocol`). Its executable can instead read them by path
(`check P C E BLOB ... --circuits CIRCUIT ...`, `Core.PeriodFiles.main`), and its network certifier's period is then
`NetworkCheck.Period.readProtocol` on what it read (`byPath`).

`CircuitFilesByPathAgreeAt k` ties the two at composite `k`: where the executable accepts the blobs `bs` with the
circuit files `cs` read by path, the verifier of record gives the same answer on the blobs `bs ++ cs`, unless two of
those byte strings collide under SHA-512. `CircuitFilesByPathAgree` is that at all three composites.
`byPath` holds what the executable reads as the files' bytes: `NetworkCheck.PeriodFiles.read` reads the file at each
path as `NetworkCheck.Period.ofRead` holds its bytes. Remote, which has no network certifier's period, has no such
executable.
-/

namespace Verity

open Core

/-- **Composite `k`'s verifier with `net` as its network certifier's period**, as its executable composes it; remote's,
which has none, is its verifier of record. -/
def verifierWith : Verity.Registration.Composite → BlobProtocol → BlobProtocol
  | .onsite, net => Onsite.protocolWith net
  | .remote, _ => Remote.protocol
  | .remoteNetwork, net => RemoteNetwork.protocolWith net
  | .remoteNetworkNoPouw, net => RemoteNetworkNoPouw.protocolWith net

/-- **Composite `k`'s executable's answer with the period's circuit files read by path** (`Core.PeriodFiles.main`): on
the parameters `p`, the commitment `c`, the evidence `e` and the blobs `bs`, with the circuit files' bytes `cs`, held as
`NetworkCheck.PeriodFiles.read` holds them (`NetworkCheck.Period.ofRead`) with as many scan files as the network
certifier's own parameters list (`Core.PeriodFiles.certifierParams`, `NetworkCheck.Period.scanCount`). -/
def byPath (k : Verity.Registration.Composite) (p c e : ByteArray) (bs cs : Array ByteArray) : Answer :=
  let n := NetworkCheck.Period.scanCount (Core.PeriodFiles.certifierParams ((canonicalJson p).toOption.getD .null))
  (verifierWith k (NetworkCheck.Period.readProtocol n (NetworkCheck.Period.ofRead n cs)).withoutBlobs).answer p c e bs

/-- **Where composite `k`'s executable accepts with the period's circuit files read by path, its verifier of record
gives the same answer with them among the blobs**: where the executable accepts the blobs `bs` with the circuit files
`cs` read by path (`byPath`), the verifier of record answers the same on the blobs `bs ++ cs`, or two of the byte
strings `bs ++ cs` collide under SHA-512 (`Hash.CollisionIn`). -/
def CircuitFilesByPathAgreeAt (k : Verity.Registration.Composite) : Prop :=
  ∀ (p c e : ByteArray) (bs cs : Array ByteArray), (byPath k p c e bs cs).verdict = .accept →
    (verifier k).answer p c e (bs ++ cs) = byPath k p c e bs cs ∨
      Hash.CollisionIn sha512 ((bs ++ cs).toList.map (·.data.toList))

/-- **`CircuitFilesByPathAgreeAt` at every composite with the network certifier's period**: onsite, and remote with the
network certifier, with PoUW or without. -/
def CircuitFilesByPathAgree : Prop :=
  ∀ (k : Verity.Registration.Composite), k ≠ .remote → CircuitFilesByPathAgreeAt k

namespace PeriodFiles

/-- The sync set of one value, δ 0 and ρ 1. -/
def emptySync : NetworkCertifier.SyncSet where
  deltas := [0]
  rhos := [1]
  deltas_nonempty := by simp
  rhos_nonempty := by simp
  rhos_pos := by
    intro ρ h
    rw [List.mem_singleton] at h
    subst h
    decide
  deltas_nodup := by simp
  rhos_nodup := by simp

/-- A network certifier's period of no link-window and no audit, in the clock form, under the deployed LMS
parameters. -/
def emptyParams : NetworkCheck.Period.Params where
  audits := #[]
  cells := #[]
  first := 0
  jitter := 0
  link := ""
  lms := Lms.deployed
  key := .empty
  session := .empty
  program := ""
  circuit := none
  form := .clock emptySync
  windows := 0

/-- `emptyParams`' commitment: no audit and no link-window registered. -/
def emptyCommitment : NetworkCheck.Period.Commitment := ⟨#[], #[], .empty⟩

/-- `emptyParams`' evidence: no audit's answer and no link-window's. -/
def emptyEvidence : NetworkCheck.Period.Evidence := ⟨#[], #[]⟩

end PeriodFiles

end Verity
