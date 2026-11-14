module

public import Spot.OneStageZk

/-!
# v2 holds the session's lookup tables to the registered circuit's (`Spot.OneStageZk.tablesHeld`)

The parameters of the one-stage audit's v2 (`Spot.OneStageZk`) register its circuit file by its SHA-512, and
the circuit names the lookup tables its session reads: its META's `lookups` (`OneStageZk.lookupsOf`). v2 refuses blobs
whose pinned MUFU tables (`OneStageZk.tablesOf`, the tables `inputsOf` hands `verity-verify-proof`) are other than those
before it sets up a session (`tablesHeld`, the first statement of `run`), so wherever the one statement's check
accepts, the session's table set is the registered circuit's.

In the per-member form the blobs are one set a window, and its members' circuits may look up different tables. There
each member's statement (`Spot.OneStageZk.Member.statement`) is handed the blobs' pinned tables its own circuit looks
up (`OneStageZk.ownTables`) and refuses where the blobs lack one of them, so a member's inputs hold its own circuit's
tables and no other, whatever other pinned table the blobs hold.

* `TablesRegistered`: on blobs `tablesHeld` accepts, the registered circuit file is among the blobs and the pinned
  tables the blobs hold are, by name, the ones it looks up (`HoldsRegisteredTables`); on blobs on which v2's check
  at any coins (`checkBlobsAt`) accepts, that holds of the verifier's own files where the parameters name no members,
  and where they name members the blobs hold each table each member's circuit looks up, sampled or not
  (`AcceptHoldsTables`); and where a member's statement reads, its circuit file is among the blobs and the lookup
  tables its inputs hold are the blobs' pinned tables among the names it looks up, every one of those names
  (`HandedOwnTables`).

It takes no premise.
-/

@[expose] public section

open Spot

namespace Properties.Spot

/-- **The blobs hold the registered circuit's lookup tables and no other pinned table**: the circuit file the verifier's
own files `p` name is among the blobs, and the pinned MUFU tables the blobs hold (`OneStageZk.tablesOf`) are, in
`Flock.MUFU_TABLES`' order, the ones it looks up (`OneStageZk.lookupsOf`). -/
def HoldsRegisteredTables (p : OneStageZk.Own) (bs : Core.Blobs) : Prop :=
  ∃ circuitB, bs.get p.circuit "circuit" = .ok circuitB ∧
    OneStageZk.lookupsOf circuitB = .ok ((OneStageZk.tablesOf bs).map (·.1))

/-- **The tables `tables` are the blobs' pinned tables the circuit of `p` looks up, every one of them**: the circuit
file the files `p` name is among the blobs, `tables` are the pinned MUFU tables the blobs hold among the names it looks
up (`OneStageZk.tablesAmong`, in `Flock.MUFU_TABLES`' order), and their names are exactly those names: the blobs hold
each table the circuit looks up, and `tables` hold no table it does not look up, whatever else the blobs hold. -/
def HandedOwnTables (p : OneStageZk.Own) (bs : Core.Blobs) (tables : List (String × ByteArray)) : Prop :=
  ∃ circuitB names, bs.get p.circuit "circuit" = .ok circuitB ∧ OneStageZk.lookupsOf circuitB = .ok names ∧
    tables = OneStageZk.tablesAmong names bs ∧ tables.map (·.1) = names

/-- **What an accept holds of the blobs' lookup tables, in the form the parameters `p` name**: where they name no
members, the blobs hold the registered tables of the verifier's own files (`p.own`) and no other pinned table; where
they name members, the blobs hold each table each member's circuit looks up (`HandedOwnTables`, at the tables its
statement is handed), a member the sample missed as one it holds, whatever pinned table of another member's circuit
they hold besides. -/
def AcceptHoldsTables (p : OneStageZk.Params) (bs : Core.Blobs) : Prop :=
  match p.members with
  | none => HoldsRegisteredTables p.own bs
  | some ms => ∀ m ∈ ms, ∃ tables, HandedOwnTables m.own bs tables

/-- **v2 holds the lookup tables a session reads to its registered circuit's**: on blobs its refusal (`tablesHeld`)
accepts, the pinned tables are the registered circuit's lookups; on blobs its check accepts at any coins, they are in
the form the parameters name (`AcceptHoldsTables`): the one statement's circuit's, or each table each member's circuit
looks up; and where a member's statement of the per-member form reads, the lookup tables its inputs hold are its own
circuit's among the blobs' (`HandedOwnTables`). -/
def TablesRegistered : Prop :=
  (∀ (p : OneStageZk.Own) (bs : Core.Blobs), OneStageZk.tablesHeld p bs = .ok () → HoldsRegisteredTables p bs) ∧
    (∀ (p : OneStageZk.Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : OneStageZk.Evidence) (bs : Core.Blobs)
      (pub : OneStage.Public) (prof : Core.Integrity),
      OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof → AcceptHoldsTables p bs) ∧
    ∀ (m : OneStageZk.Member) (slice : Lean.Json) (bs : Core.Blobs) (a : OneStageZk.StepAt),
      m.statement slice bs = .ok a → HandedOwnTables m.own bs a.I.tables

/-- A digest under which the witnesses' blobs hold their circuit file, whatever its bytes: `Core.Blobs` pairs each file
with the digest it is found under. -/
def circuitDigest : ByteArray := ⟨Array.replicate 64 0x11⟩

/-- The verifier's own files naming the circuit file under `circuitDigest` and no public file. -/
def witnessOwn : OneStageZk.Own :=
  { programJ := Lean.Json.null, program := .empty, partition := Lean.Json.null, circuit := String.ofList (List.replicate 128 '1'),
    pub := "", registered := none }

/-- `rcp`'s pinned SHA-512 (`Flock.MUFU_TABLES`), as bytes. -/
def rcpDigest : ByteArray :=
  ⟨#[0x8a, 0x31, 0x7a, 0xdb, 0xcc, 0x07, 0x8b, 0xf7, 0x0a, 0xe0, 0x19, 0x59, 0xb6, 0x79, 0xd4, 0x9f,
     0x97, 0xcd, 0x9c, 0x8a, 0x93, 0xd6, 0xf3, 0x9b, 0x73, 0x25, 0xc9, 0x3d, 0x0f, 0xd5, 0x50, 0xfa,
     0x43, 0x96, 0xa4, 0xc3, 0xa0, 0x56, 0x8d, 0x23, 0x87, 0xc2, 0xff, 0x2f, 0x33, 0xeb, 0x76, 0x45,
     0x3b, 0x33, 0xa7, 0xb8, 0x89, 0x0d, 0x97, 0x75, 0x2d, 0x86, 0xcc, 0x1d, 0x97, 0x1a, 0xea, 0xcc]⟩

/-- A circuit file whose META looks up `rcp`. -/
def honestCircuit : ByteArray := "circuit\nMETA {\"lookups\":{\"rcp\":{}}}\n".toUTF8

/-- A circuit file whose META looks up no table. -/
def tableCircuit : ByteArray := "circuit\nMETA {}\n".toUTF8

/-- Blobs as the served path hands v2: a circuit file looking up `rcp` and `rcp`'s table, under its pinned SHA-512. -/
def honestBlobs : Core.Blobs := ⟨#[(circuitDigest, honestCircuit), (rcpDigest, ByteArray.empty)]⟩

/-- Blobs holding a circuit file that looks up no table, and `rcp`'s table under its pinned SHA-512. -/
def tableBlobs : Core.Blobs := ⟨#[(circuitDigest, tableCircuit), (rcpDigest, ByteArray.empty)]⟩

end Properties.Spot
