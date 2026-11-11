module

public import Proofs.Flock.SessionLoop.Concrete.Fwd
public import Proofs.Flock.SessionLoop.ConcreteNet
public import Properties.Flock.Nets
public import Flock
public import VStar.Unit
public import VStar.Compose
public import Std.Data.String.ToNat
import Flock.HmRow
import Flock.Net
import Flock.HmNets
import all Init.Data.Array.QSort.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import Flock.Canon
import Flock.JsonParse
import Flock.RecOpen
import Flock.RecResiduals
import Flock.RecSparse
import Flock.RecOpening
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Subslice
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.FindPos
import all Init.Data.String.Termination
import all Init.Data.String.ToSlice
import all Init.Data.String.Bootstrap
import all Init.Data.String.Iter
import all Init.Data.String.Iterator
import all Init.Data.String.Substring
import all Init.Data.String.Modify
import all Init.Data.String.Extra
import all Init.Data.String.Legacy

/-! # The concrete statement's circuit file, and what `HmRow.parse` makes of it

The circuit file of the statement `concreteStmt` (`Concrete/Stmt.lean`): V\*'s file layout (`RecDeploy.VStarNets.fileLs`)
with the one-unit net `copy16` (`unitLaid`, 16 input words copied to 16 output words), a META line (`metaText`) and the
verifier's own `sha512x3` and `hm96` nets as parameters `ls` and `lh`, so that no term here asks the kernel to build
them. `parse_cW` walks `HmRow.parse` on its bytes (`bytesOf`) to the circuit `cW`, at hypotheses that the statement's
proof discharges at the open guarantee that the two nets build (`Properties.Flock.Nets.FixedNetsBuild`).

The walk is `fwd_def` (`Concrete/Fwd.lean`); its rules are the lemmas below, each a fact about one step of the parse.
-/

@[expose] public section
open Flock RecDeploy.VStarNets FlockVBridge.Names
open Properties.Flock.Nets (built sparseOf)
open Lean (Json)

set_option maxHeartbeats 0 -- the walks below step through `HmRow.parse` and leave closed checks to the kernel
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete

def metaChars : List Char :=
  ['{', '"', 'f', 'o', 'r', 'm', 'a', 't', '"', ':', '"', 'f', 'l', 'o', 'c', 'k', '-', 'c', 'i', 'r', 'c', 'u', 'i', 't',
   '"', ',', '"', 's', 't', 'a', 't', 'e', 'm', 'e', 'n', 't', '"', ':', '"', 'v', 'e', 'r', 'i', 't', 'y', '/', 'f', 'l',
   'o', 'c', 'k', '-', 'c', 'i', 'r', 'c', 'u', 'i', 't', '"', ',', '"', 's', 'c', 'h', 'e', 'm', 'e', '"', ':', '"', 'f',
   'r', 'a', 'm', 'e', '-', 'v', '3', '-', 's', 'h', 'a', '5', '1', '2', '/', 'h', 'm', '9', '6', '-', 's', 'h', 'a', '5',
   '1', '2', '"', ',', '"', 'r', 'o', 'w', '_', 's', 'c', 'h', 'e', 'm', 'a', '"', ':', '"', 'h', 'm', '9', '6', '-', 's',
   'h', 'a', '5', '1', '2', '/', 'r', 'o', 'w', '/', 'v', '1', '"', ',', '"', 'p', 'o', 'r', 't', 's', '"', ':', '[', '[',
   '"', 'x', '"', ',', '6', '4', ',', '1', ']', ']', ',', '"', 'r', 'o', 'w', 's', '"', ':', '[', '{', '"', 'p', 'r', 'e',
   'f', 'i', 'x', '"', ':', '"', '7', '6', '6', '5', '7', '2', '6', '9', '7', '4', '7', '9', '2', 'f', '7', '3', '6', '8',
   '6', '1', '3', '5', '3', '1', '3', '2', '2', 'd', '7', '2', '6', 'f', '7', '7', '2', 'f', '7', '6', '3', '1', '0', '0',
   '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '4', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '"', ',',
   '"', 'm', 'i', 'd', 's', 't', 'a', 't', 'e', '"', ':', '[', '"', '4', '1', '9', '9', 'd', 'c', '1', 'e', '3', 'a', '1',
   '0', '2', '5', 'e', 'f', '"', ',', '"', 'a', '8', '0', 'd', 'a', '3', '3', '0', '0', '9', 'a', '5', '3', 'd', 'f', 'b',
   '"', ',', '"', 'e', '2', '2', 'a', 'c', '9', '2', '0', 'c', 'b', '4', 'e', 'f', '1', 'e', '1', '"', ',', '"', '5', '4',
   'a', '9', '1', '7', 'b', '5', 'e', '1', 'c', '9', 'e', '4', 'e', 'f', '"', ',', '"', '0', '8', '6', 'b', '1', '4', '2',
   '3', '7', '0', '3', '8', 'b', '3', '1', '4', '"', ',', '"', 'f', 'e', 'e', '5', '5', '4', 'c', 'e', 'e', 'f', '1', 'b',
   '8', 'e', '9', 'e', '"', ',', '"', 'a', '2', 'd', 'c', '2', 'a', 'e', '8', 'd', '9', '9', '7', '3', 'b', '0', 'f', '"',
   ',', '"', '1', '3', 'f', '4', '4', '4', 'b', '0', '5', '0', '1', 'f', '9', '7', 'c', 'a', '"', ']', ',', '"', 'p', 'a',
   'd', '"', ':', '"', '8', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
   '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '8', '0', '0', '"', '}', ']', ',',
   '"', 'r', 'o', 'w', '_', 'k', 'e', 'y', '_', 's', 'h', 'a', '5', '1', '2', '"', ':', '"', 'a', 'f', 'c', '5', 'a', '6',
   '0', 'c', '8', '5', 'd', '8', 'a', 'a', '7', '5', 'c', 'b', '4', 'c', '7', 'f', '7', '2', '6', '5', '8', '8', 'd', '2',
   '9', '9', 'a', '1', '7', '1', '2', 'e', '1', 'c', '6', 'a', '6', '2', '1', 'f', '0', '9', 'c', 'a', '6', '4', '8', '0',
   '9', 'a', '5', 'f', 'e', 'd', 'd', '7', '4', '4', 'a', '0', 'f', '6', 'b', '0', '2', '0', '1', 'd', '0', '5', 'e', '8',
   '2', 'b', '2', '9', 'c', 'd', '5', 'b', '2', '7', '4', 'e', '6', 'c', '2', '9', '6', '7', '0', 'c', 'd', '4', 'e', '4',
   'a', 'b', '1', '0', '8', '3', '1', '9', '1', '1', 'f', 'b', '1', 'f', '9', 'e', '0', 'c', '7', '0', '1', '9', 'd', 'f',
   '2', '2', '"', ',', '"', 'k', '_', 'l', 'o', 'g', '"', ':', '2', '2', ',', '"', 'v', 'u', 's', '_', 'p', 'e', 'r', '_',
   'b', 'l', 'o', 'c', 'k', '"', ':', '1', ',', '"', 'u', 'n', 'i', 't', 's', '_', 'p', 'e', 'r', '_', 'v', 'u', '"', ':',
   '1', ',', '"', 'm', 'a', 's', 'k', '_', 'l', 'o', 'g', '"', ':', '1', '4', ',', '"', 'r', 'a', 'n', 'g', 'e', 's', '"',
   ':', '[', '{', '"', 't', 'y', 'p', 'e', '"', ':', '"', 'u', 'n', 'i', 't', '"', ',', '"', 'n', 'e', 't', '"', ':', '"',
   'h', 'm', '9', '6', '"', ',', '"', 's', 'l', 'o', 't', '_', 'l', 'o', 'g', '"', ':', '1', '8', ',', '"', 'p', 'o', 's',
   '0', '"', ':', '0', ',', '"', 'c', 'o', 'u', 'n', 't', '"', ':', '1', '}', ',', '{', '"', 't', 'y', 'p', 'e', '"', ':',
   '"', 'u', 'n', 'i', 't', '"', ',', '"', 'n', 'e', 't', '"', ':', '"', 's', 'h', 'a', '5', '1', '2', 'x', '3', '"', ',',
   '"', 's', 'l', 'o', 't', '_', 'l', 'o', 'g', '"', ':', '1', '8', ',', '"', 'p', 'o', 's', '0', '"', ':', '1', ',', '"',
   'c', 'o', 'u', 'n', 't', '"', ':', '1', '}', ',', '{', '"', 't', 'y', 'p', 'e', '"', ':', '"', 'u', 'n', 'i', 't', '"',
   ',', '"', 'n', 'e', 't', '"', ':', '"', 'u', 'n', 'i', 't', '"', ',', '"', 's', 'l', 'o', 't', '_', 'l', 'o', 'g', '"',
   ':', '9', ',', '"', 'p', 'o', 's', '0', '"', ':', '1', '0', '2', '4', ',', '"', 'c', 'o', 'u', 'n', 't', '"', ':', '1',
   '}', ',', '{', '"', 't', 'y', 'p', 'e', '"', ':', '"', 'm', 'a', 's', 'k', '"', ',', '"', 's', 'l', 'o', 't', '_', 'l',
   'o', 'g', '"', ':', '1', '4', ',', '"', 'p', 'o', 's', '0', '"', ':', '6', '4', ',', '"', 'c', 'o', 'u', 'n', 't', '"',
   ':', '1', '}', ']', ',', '"', 'l', 'e', 'a', 'v', 'e', 's', '_', 'i', 'n', '"', ':', '[', '[', '0', ']', ']', ',', '"',
   'l', 'e', 'a', 'v', 'e', 's', '_', 'o', 'u', 't', '"', ':', '[', '[', '0', ']', ']', ',', '"', 'd', 'u', 'm', 'm', 'y',
   '"', ':', '{', '"', 'b', 'c', '"', ':', '[', '"', '0', 'e', '9', '0', '3', '9', '3', '8', '1', 'b', 'b', 'b', '6', 'a',
   '2', '2', 'd', '1', 'a', '1', '7', 'c', 'c', 'c', 'd', '3', '7', '8', '8', '8', '7', 'b', '1', '2', '4', '4', 'e', '9',
   'c', '6', '8', 'e', '6', '1', '7', '7', '8', '8', '8', '3', '0', '8', '2', '5', 'f', '6', '5', '8', 'b', '1', '6', '0',
   'b', 'e', '1', '2', '4', 'f', '7', '1', '6', 'c', '4', '8', '3', '0', '5', 'a', '7', '4', '0', 'd', '6', '5', 'f', 'f',
   'c', '2', '5', '0', '4', 'd', '7', 'd', '7', '4', '0', '0', '0', '9', '9', 'e', '7', '8', '6', 'c', 'c', '8', '8', 'b',
   '9', '9', '5', '3', '0', '5', '3', '3', '4', 'a', '9', '0', '1', '1', '9', '0', 'd', 'f', '8', '7', 'a', '4', 'f', 'a',
   'b', '0', '6', '0', '4', '7', '4', 'd', 'e', 'c', '4', 'e', '8', '3', '0', '6', '2', '2', 'e', '6', '8', '7', '4', 'd',
   '3', '4', '7', '2', '3', '6', '2', '7', '4', '0', 'd', 'b', 'd', '9', '7', 'c', '1', '5', '5', '4', 'b', '9', '4', 'a',
   '9', 'f', '5', '7', '2', '6', 'd', 'f', '8', '8', '7', '5', '8', '4', '3', '4', '4', 'b', '1', '7', '5', 'a', '3', '8',
   'b', '4', 'a', 'e', 'a', '5', '7', 'b', '8', '7', '3', '6', 'b', 'b', '1', '1', 'f', 'e', 'c', '4', 'd', '8', '7', '5',
   '5', 'c', '4', '2', '4', '2', 'c', '0', 'c', '7', 'b', '9', '9', 'b', '2', '3', '9', 'c', '7', 'c', '1', '6', 'd', '3',
   '3', 'c', '"', ']', ',', '"', 'o', 'u', 't', 'p', 'u', 't', 's', '"', ':', '[', '0', ']', '}', ',', '"', 'o', 'u', 't',
   '_', 'p', 'o', 'r', 't', 's', '"', ':', '[', '[', '"', 'o', 'u', 't', '"', ',', '1', ']', ']', ',', '"', 'p', 'e', 'r',
   '_', 'v', 'u', '"', ':', '{', '"', 'h', 'm', '9', '6', '"', ':', '1', ',', '"', 's', 'h', 'a', '5', '1', '2', 'x', '3',
   '"', ':', '1', ',', '"', 'u', 'n', 'i', 't', '"', ':', '1', '}', ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', '[',
   '"', 'h', 'm', '9', '6', '"', ',', '0', ',', '0', ',', '"', 's', 'h', 'a', '5', '1', '2', 'x', '3', '"', ',', '0', ',',
   '0', ']', ']', ',', '"', 'l', 'e', 'a', 'f', '_', 's', 'c', 'h', 'e', 'm', 'e', '"', ':', '{', '"', 'i', 'd', '"', ':',
   '"', 'h', 'm', '9', '6', '-', 's', 'h', 'a', '5', '1', '2', '/', 'v', '1', '"', ',', '"', 'i', 'n', 'n', 'e', 'r', '"',
   ':', '"', 's', 'h', 'a', '5', '1', '2', ' ', 'c', 'o', 'l', 'u', 'm', 'n', ' ', 'd', 'i', 'g', 'e', 's', 't', '"', ',',
   '"', 's', 'a', 'l', 't', '_', 'b', 'y', 't', 'e', 's', '"', ':', '1', '9', '2', ',', '"', 's', 'a', 'l', 't', '_', 's',
   'o', 'u', 'r', 'c', 'e', '"', ':', '"', 'c', 'h', 'a', 'c', 'h', 'a', '2', '0', ' ', 'e', 'x', 'p', 'a', 'n', 's', 'i',
   'o', 'n', ' ', 'o', 'f', ' ', 'a', ' ', 'f', 'r', 'e', 's', 'h', ' ', '2', '5', '6', '-', 'b', 'i', 't', ' ', 'o', 's',
   ' ', 'k', 'e', 'y', ' ', 'p', 'e', 'r', ' ', 'p', 'r', 'o', 'o', 'f', ',', ' ', 'n', 'e', 'v', 'e', 'r', ' ', 'r', 'e',
   'u', 's', 'e', 'd', ' ', 'o', 'r', ' ', 's', 't', 'o', 'r', 'e', 'd', '"', ',', '"', 'k', 'e', 'y', '"', ':', '"', 'd',
   'e', 'f', 'a', 'u', 'l', 't', '_', 'k', 'e', 'y', ',', ' ', 'a', ' ', 's', 't', 'a', 't', 'e', 'm', 'e', 'n', 't', ' ',
   'c', 'o', 'n', 's', 't', 'a', 'n', 't', ' ', 'f', 'o', 'r', ' ', 'e', 'v', 'e', 'r', 'y', ' ', 't', 'r', 'e', 'e', '"',
   ',', '"', 'k', 'e', 'y', '_', 's', 'h', 'a', '5', '1', '2', '"', ':', '"', 'a', 'f', 'c', '5', 'a', '6', '0', 'c', '8',
   '5', 'd', '8', 'a', 'a', '7', '5', 'c', 'b', '4', 'c', '7', 'f', '7', '2', '6', '5', '8', '8', 'd', '2', '9', '9', 'a',
   '1', '7', '1', '2', 'e', '1', 'c', '6', 'a', '6', '2', '1', 'f', '0', '9', 'c', 'a', '6', '4', '8', '0', '9', 'a', '5',
   'f', 'e', 'd', 'd', '7', '4', '4', 'a', '0', 'f', '6', 'b', '0', '2', '0', '1', 'd', '0', '5', 'e', '8', '2', 'b', '2',
   '9', 'c', 'd', '5', 'b', '2', '7', '4', 'e', '6', 'c', '2', '9', '6', '7', '0', 'c', 'd', '4', 'e', '4', 'a', 'b', '1',
   '0', '8', '3', '1', '9', '1', '1', 'f', 'b', '1', 'f', '9', 'e', '0', 'c', '7', '0', '1', '9', 'd', 'f', '2', '2', '"',
   '}', '}']

def metaText : String := String.ofList metaChars

def unitRow (a : Bool) (i : Nat) : Array Nat :=
  if i < 16 then #[i] else if 128 ≤ i && i < 144 then (if a then #[i - 128] else #[256])
  else if i == 256 then #[256] else #[]

def unitLaid : HmNets.Laid :=
  { useful := 257, constPos := 256, inGroups := #[⟨0, 1, #[16]⟩], outGroups := #[⟨1, 1, #[16]⟩],
    ra := (Array.range 257).map (unitRow true), rb := (Array.range 257).map (unitRow false) }

def mjC : Json := (JsonParse.parseList metaChars).toOption.getD .null

theorem meta_parse : JsonParse.parseList metaChars = .ok mjC := by kernel_rfl'

def t0 : Tags := Tags.circuitEb90718f
def coins0 : Coins := .tree (Zk.coinSpec (t0.tableNames 1))
def tz : Tags := Zk.tags t0 14 1

def lines (ls lh : HmNets.Laid) : List String := fileLs "copy16" metaText unitLaid ls lh
def bytesOf (ls lh : HmNets.Laid) : ByteArray := (joinNl (lines ls lh)).toUTF8

theorem startsWith_append (a b : String) : (a ++ b).startsWith a = true :=
  String.startsWith_string_iff.2 ⟨b.toList, by simp⟩

theorem drop_append (a b : String) : ((a ++ b).drop a.length).toString = b := by
  rw [String.Slice.toString_eq]
  apply String.toList_inj.1
  rw [String.toList_copy_drop, String.toList_append, ← String.length_toList, List.drop_left]

theorem trim_of (s : String) (h : s.toList.getLast?.any Char.isWhitespace = false) :
    s.trimAsciiEnd.toString = s := by
  show (String.Slice.dropEndWhile s.toSlice Char.isWhitespace).copy = s
  rw [String.Slice.dropEndWhile_eq_self, String.copy_toSlice]
  rw [String.Slice.endsWith_bool_eq_getLast?, String.copy_toSlice]
  exact h

theorem getD_toArray {α} (L : List α) (i : Nat) (d : α) : L.toArray.getD i d = L.getD i d := by
  simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]
theorem getElem!_toArray (L : List String) (i : Nat) : L.toArray[i]! = L.getD i "" := by
  rw [getElem!_str, getD_toArray]

theorem lines1 (ls lh : HmNets.Laid) : (lines ls lh).getD 1 "" = "META " ++ metaText := rfl

theorem starts_meta : ("META " ++ metaText).startsWith "META " = true := startsWith_append _ _
theorem drop_meta : (("META " ++ metaText).drop 5).toString = metaText := drop_append "META " metaText

theorem forIn_yield {β : Type} {f : Unit → β → Except String (ForInStep β)} {b b' : β} {r : Except String β}
    (h : f () b = .ok (.yield b')) (hr : forIn Lean.Loop.mk b' f = r) : forIn Lean.Loop.mk b f = r := by
  change Lean.Loop.forIn Lean.Loop.mk b f = r
  rw [Lean.Loop.forIn_eq_of_monadTail, h]; exact hr
theorem forIn_done {β : Type} {f : Unit → β → Except String (ForInStep β)} {b b' : β}
    (h : f () b = .ok (.done b')) : forIn Lean.Loop.mk b f = .ok b' := by
  change Lean.Loop.forIn Lean.Loop.mk b f = .ok b'
  rw [Lean.Loop.forIn_eq_of_monadTail, h]; rfl

theorem getD3 (a b c d : String) : (List.toArray [a, b, c]).toList.getD 2 d = c := rfl
def lsJ : Lean.Json := match J.get mjC "leaf_scheme" "" with | .ok j => j | .error _ => .null
def pinJ : Lean.Json := tz.pinnedLeafScheme.getD .null
theorem bne_self (s : String) : (s != s) = false := by simp
theorem toNat?_toString (n : Nat) : (toString n).toNat? = some n := Nat.toNat?_repr n

theorem splitOnAux_eq (s sep : String) (b i j : String.Pos.Raw) (r : List String) :
    s.splitOnAux sep b i j r =
      if i.atEnd s = true then (b.extract s i :: r).reverse
      else if (i.get s == j.get sep) = true then
        if (j.next sep).atEnd sep = true then
          s.splitOnAux sep (i.next s) (i.next s) 0 (b.extract s ((i.next s).unoffsetBy (j.next sep)) :: r)
        else s.splitOnAux sep b (i.next s) (j.next sep) r
      else s.splitOnAux sep b ((i.unoffsetBy j).next s) 0 r := String.splitOnAux.eq_1 s sep b i j r

theorem split_l : "copy16".splitOn "/l" = ["copy16"] := by
  unfold String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])
theorem split_m : "copy16".splitOn "/m" = ["copy16"] := by
  unfold String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])
theorem split_n : "copy16".splitOn "/n" = ["copy16"] := by
  unfold String.splitOn
  simp (config := {decide := true}) only [ite_false]
  repeat (rw [splitOnAux_eq]; simp (config := {decide := true}) only [ite_false])

theorem open_none : RecOpen.params "copy16" = none := by unfold RecOpen.params; rw [split_l]
theorem res_none : RecResiduals.params "copy16" = none := by unfold RecResiduals.params; rw [split_m]
theorem sparse_none : RecSparse.params "copy16" = none := by unfold RecSparse.params; rw [split_n]
theorem opening_none : RecOpening.params "copy16" = none := by unfold RecOpening.params; rw [split_m]
theorem unit_eq (c : Circuit) : c.unit = (c.nets.getD c.unitNet default).2 := rfl

theorem lines_if (L : List String) :
    (if ((L ++ [""]).toArray.back? == some "") = true then (L ++ [""]).toArray.pop else (L ++ [""]).toArray) =
      L.toArray := by
  simp

theorem metaChars_last : metaChars.getLast? = some '}' := by decide +kernel

theorem trim_meta : ("META " ++ metaText).trimAsciiEnd.toString = "META " ++ metaText := by
  apply trim_of
  simp only [String.toList_append, metaText, String.toList_ofList, List.getLast?_append, metaChars_last]
  rfl

theorem parse_meta : JsonParse.parse metaText = .ok mjC := by
  rw [JsonParse.parse_eq, metaText, String.toList_ofList]; exact meta_parse

theorem noNl_meta : NoNl metaText := by
  unfold NoNl metaText; rw [String.toList_ofList]; decide +kernel

theorem noNl_copy16 : NoNl "copy16" := by unfold NoNl; decide

theorem unit_lines : VStar.netLines unitLaid = 258 := rfl

theorem netLs_length (name : String) (l : HmNets.Laid) : (netLs name l).length = VStar.netLines l := by
  simp [netLs, rowsLs, VStar.netLines]; omega

section
variable (ls lh : HmNets.Laid) (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
include hls hlh

theorem lines_eq : lines ls lh =
    ["flock-circuit copy16", "META " ++ metaText] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs "copy16" unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) := by
  simp only [lines, fileLs, secLs, VStar.SHA_NET, VStar.HM_NET, unit_lines, hls, hlh]
  simp

theorem size_lines : (lines ls lh).toArray.size = 503051 := by
  rw [lines_eq ls lh hls hlh]
  simp only [List.size_toArray, List.length_append, List.length_cons, netLs_length, unit_lines, hls, hlh]
  rfl

end

theorem getD_mid {α : Type} (A T R : List α) (h d : α) (i : Nat) (hi : i = A.length) :
    (A ++ (h :: T) ++ R).getD i d = h := by
  subst hi; simp [List.getD_eq_getElem?_getD, List.append_assoc]

theorem extract_mid {α : Type} (A T R : List α) (a b : Nat) (ha : a = A.length) (hb : b = A.length + T.length) :
    (A ++ T ++ R).toArray.extract a b = T.toArray := by
  subst ha hb
  rw [List.extract_toArray]
  simp [List.append_assoc]

theorem circuit_unit : "CIRCUIT unit " ++ toString 258 = "CIRCUIT" ++ " " ++ "unit" ++ " " ++ toString 258 := by
  simp only [String.append_assoc]; rfl
theorem circuit_sha : "CIRCUIT sha512x3 " ++ toString 256002 = "CIRCUIT" ++ " " ++ "sha512x3" ++ " " ++ toString 256002 := by
  simp only [String.append_assoc]; rfl
theorem circuit_hm : "CIRCUIT hm96 " ++ toString 246786 = "CIRCUIT" ++ " " ++ "hm96" ++ " " ++ toString 246786 := by
  simp only [String.append_assoc]; rfl

abbrev PL : List String := ["flock-circuit copy16", "META " ++ metaText]
abbrev cU : String := "CIRCUIT unit " ++ toString 258
abbrev cS : String := "CIRCUIT sha512x3 " ++ toString 256002
abbrev cH : String := "CIRCUIT hm96 " ++ toString 246786

section
variable (ls lh : HmNets.Laid) (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
include hls hlh

theorem line_unit (i : Nat) (hi : i = 2) : (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "unit" ++ " " ++ toString 258 := by
  have e : lines ls lh = PL ++ (cU :: netLs "copy16" unitLaid) ++ ((cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh)) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi])).trans circuit_unit

theorem line_sha (i : Nat) (hi : i = 261) :
    (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "sha512x3" ++ " " ++ toString 256002 := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid)) ++ (cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh) := by
    rw [lines_eq ls lh hls hlh]
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi, netLs_length, unit_lines])).trans circuit_sha

theorem line_hm (i : Nat) (hi : i = 256264) :
    (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "hm96" ++ " " ++ toString 246786 := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ (cS :: netLs "sha512x3" ls)) ++ (cH :: netLs "hm96" lh) ++ [] := by
    rw [lines_eq ls lh hls hlh, List.append_nil]
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi, netLs_length, unit_lines, hls])).trans circuit_hm

theorem ext_unit (a b : Nat) (ha : a = 3) (hb : b = 261) :
    (lines ls lh).toArray.extract a b = (netLs "copy16" unitLaid).toArray := by
  have e : lines ls lh = (PL ++ [cU]) ++ netLs "copy16" unitLaid ++ ((cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh)) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha]) (by simp [hb, netLs_length, unit_lines])

theorem ext_sha (a b : Nat) (ha : a = 262) (hb : b = 256264) :
    (lines ls lh).toArray.extract a b = (netLs "sha512x3" ls).toArray := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ [cS]) ++ netLs "sha512x3" ls ++ (cH :: netLs "hm96" lh) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha, netLs_length, unit_lines]) (by simp [hb, netLs_length, unit_lines, hls])

theorem ext_hm (a b : Nat) (ha : a = 256265) (hb : b = 503051) :
    (lines ls lh).toArray.extract a b = (netLs "hm96" lh).toArray := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ (cS :: netLs "sha512x3" ls) ++ [cH]) ++ netLs "hm96" lh ++ [] := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha, netLs_length, unit_lines, hls]) (by simp [hb, netLs_length, unit_lines, hls, hlh])
end

def sparse9 (r : Array (Array Nat)) : Sparse := (Sparse.ofRows r (2 ^ 9)).toOption.getD default
def UN : Net := { built "copy16" 257 ⟨0, 1, #[16]⟩ ⟨1, 1, #[16]⟩ 1 (sparse9 unitLaid.ra) (sparse9 unitLaid.rb) with unitLog := 9 }

theorem unit_netOf : HmNets.netOf "copy16" unitLaid = .ok UN := by kernel_rfl'

theorem tok_copy16 : FlockDeploy.ConcreteNet.Tok "copy16" := ⟨by decide, by simp [NoWs]⟩

theorem parse_unit : Net.parse (netLs "copy16" unitLaid).toArray = .ok UN :=
  FlockDeploy.ConcreteNet.parse_netLs tok_copy16 (by simp [unitLaid]) (by simp [unitLaid]) unit_netOf

theorem tokens3 (a b c : String) (ha : (a != "") = true) (hb : (b != "") = true) (hc : (c != "") = true)
    (na : a.toList.all (fun c => !c.isWhitespace) = true) (nb : b.toList.all (fun c => !c.isWhitespace) = true)
    (nc : c.toList.all (fun c => !c.isWhitespace) = true) : tokens (a ++ " " ++ b ++ " " ++ c) = #[a, b, c] := by
  have nw : ∀ s : String, s.toList.all (fun c => !c.isWhitespace) = true → NoWs s := fun s h c hc => by
    simpa using List.all_eq_true.1 h c hc
  exact tokens_three a b c (by simpa using ha) (by simpa using hb) (by simpa using hc) (nw a na) (nw b nb) (nw c nc)

theorem hutf (ls lh : HmNets.Laid) : String.fromUTF8? (bytesOf ls lh) = some (joinNl (lines ls lh)) :=
  fromUTF8?_toUTF8 _
theorem hsplit (ls lh : HmNets.Laid) : (joinNl (lines ls lh)).splitOn "\n" = lines ls lh ++ [""] :=
  splitOn_join _ (fileLs_noNl _ _ _ _ _ noNl_copy16 noNl_meta)

abbrev shaB (A B : Sparse) : Net := built "sha512x3" 256001 ⟨0, 64, HmNets.SHA_IN⟩ ⟨1984, 16, HmNets.SHA_OUT⟩ 64 A B
abbrev hmB (A B : Sparse) : Net := built "hm96" 246785 ⟨0, 32, HmNets.HM_IN⟩ ⟨1920, 8, HmNets.HM_OUT⟩ 32 A B

fwd_def cW parse_cW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    HmRow.parse tz (bytesOf ls lh) [] by
  [hH, hutf ls lh, hsplit ls lh, lines_if _, getD_toArray, getElem!_toArray, lines1, trim_meta, starts_meta, drop_meta,
    parse_meta, trim_of, size_lines ls lh hls hlh, line_unit ls lh hls hlh, line_sha ls lh hls hlh,
    line_hm ls lh hls hlh, tokens3, getD3, toNat?_toString, ext_unit ls lh hls hlh, ext_sha ls lh hls hlh,
    ext_hm ls lh hls hlh, parse_unit, hps, hph, forIn_yield, forIn_done, hc1, hc2, bne_self, hck, unit_eq, open_none,
    res_none, sparse_none, opening_none]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse]
  shallow [metaText]

end FlockDeploy.Concrete
