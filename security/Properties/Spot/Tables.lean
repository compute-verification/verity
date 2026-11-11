module

public import Spot.OneStageZk

/-!
# v2 holds the session's lookup tables to the registered circuit's (`Spot.OneStageZk.tablesHeld`)

The parameters of the one-stage audit's v2 (`Spot.OneStageZk`) register its circuit file by its SHA-512, and
the circuit names the lookup tables its session reads: its META's `lookups` (`OneStageZk.lookupsOf`). v2 refuses blobs
whose pinned MUFU tables (`OneStageZk.tablesOf`, the tables `inputsOf` hands `flock-verify`) are other than those
before it sets up a session (`tablesHeld`, the first statement of `run`), so wherever its check accepts, the session's
table set is the registered circuit's.

* `TablesRegistered`: on blobs `tablesHeld` accepts, and on blobs on which v2's check at any coins (`checkBlobsAt`)
  accepts, the registered circuit file is among the blobs and the pinned tables the blobs hold are, by name, the ones it
  looks up (`HoldsRegisteredTables`).

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

/-- **v2 holds the blobs' lookup tables to the registered circuit's**: on blobs its refusal (`tablesHeld`) accepts, and
on blobs its check accepts at any coins, the pinned tables are the registered circuit's lookups. -/
def TablesRegistered : Prop :=
  (∀ (p : OneStageZk.Own) (bs : Core.Blobs), OneStageZk.tablesHeld p bs = .ok () → HoldsRegisteredTables p bs) ∧
    ∀ (p : OneStageZk.Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : OneStageZk.Evidence) (bs : Core.Blobs)
      (pub : OneStage.Public) (prof : Core.Integrity),
      OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof → HoldsRegisteredTables p.own bs

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
