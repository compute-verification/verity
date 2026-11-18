/-
The verifier's pin of the challenger's key: the SHA-512 of its public key, which the auditor takes from its own checkout
(`--challenger-key-sha512 H`, with `--challenger-retired FILE`, a `challenger-key/v1` file's `retired` list), never from
the case. One pin serves every key the challenger signs under: SPOT's coins (`VStar.Recursive.ChallengerPin`) and PoUW's
draw coin, a period's `key` (`periodReason`, `pouw-period`). A key that is none, retired or other than the pin's is
refused by name (`keyReason`). A run with no pin says so first, on stderr (`UNPINNED`), its answer line unchanged.
-/
module

public import CoreProtocol.Cli
public import Spot.ChallengerRecord
public import Spot.Json

@[expose] public section

namespace Spot.ChallengerPin

open Lean (Json)
open Flock (V)

/-- The pin: `sha512`, lowercase hex of 64 bytes; `retired`, the retired public keys as lowercase hex. -/
structure Pin where
  sha512 : String
  retired : Array String := #[]

def NONE : String :=
  "challenger-key-none: the case's parameters name no challenger key, and the verifier pins one " ++
    "(--challenger-key-sha512)"

def RETIRED (h : String) : String :=
  s!"challenger-key-retired: the case's challenger key (public key SHA-512 {h}) is one the verifier's key file retires"

def OTHER (h pin : String) : String :=
  s!"challenger-key-not-pinned: the case's challenger key (public key SHA-512 {h}) is not the verifier's pin {pin}"

def UNREAD : String :=
  "challenger-key-unread: the case's parameters don't parse to the challenger key the verifier pins"

def UNPINNED : String := "challenger's key not pinned: the case names it"

/-- The retired list is read before the pin, so a pin the key file retires is refused too. -/
def keyReason (pin : Pin) : Option Challenger.ChallengerKey → Option String
  | none => some NONE
  | some k =>
    let h := sha512Hex k.pub
    if pin.retired.contains (Flock.hexOf k.pub) then some (RETIRED h)
    else if h == pin.sha512 then none
    else some (OTHER h pin.sha512)

def PERIOD : String := "verity/pouw/period/v1"

/-- `keyReason` on a PoUW period's parameters `j`: its `key`, the challenger's that signs its draw coin. -/
def periodReason (pin : Pin) (j : Json) : Option String :=
  match getD j "key" with
  | .null => some NONE
  | k => match Challenger.ChallengerKey.ofJson k with
    | .ok k => keyReason pin (some k)
    | .error _ => some UNREAD

/-- `main` on `args`, saying first on stderr that no pin is taken. -/
def unpinned (main : List String → IO UInt32) (args : List String) : IO UInt32 := do
  unless args == ["id"] do IO.eprintln UNPINNED
  main args

/-- `P`, refusing parameters `why` names before its own parse reads them. -/
def guard (P : Core.Protocol) (why : Json → Option String) : Core.Protocol :=
  { P with params := fun j => do
      if let some r := why j then throw r
      P.params j }

/-- The public keys a `challenger-key/v1` file's `retired` list names, as lowercase hex. -/
def retiredOf (path : String) : IO (Array String) := do
  let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  match getD j "retired" with
  | .arr xs => xs.mapM fun x => match getD x "pubkey" with
    | .str s => if isHex s then pure s else throw (IO.userError s!"{path}: a retired pubkey is not lowercase hex")
    | _ => throw (IO.userError s!"{path}: a retired key names no pubkey")
  | .null => pure #[]
  | _ => throw (IO.userError s!"{path}: its retired list is not a list")

/-- `--challenger-key-sha512 H [--challenger-retired FILE] REST`: the pin and `REST`; no pin without the flag. -/
def parse : List String → IO (Option Pin × List String)
  | "--challenger-key-sha512" :: h :: rest => do
    unless h.length == 128 && isHex h do
      throw (IO.userError "--challenger-key-sha512 takes lowercase hex of 64 bytes")
    match rest with
    | "--challenger-retired" :: f :: rest => pure (some { sha512 := h, retired := ← retiredOf f }, rest)
    | rest => pure (some { sha512 := h }, rest)
  | rest => pure (none, rest)

/-- A PoUW period's command line (`Core.Protocol.main`), under the pin `parse` reads ahead of it. -/
def periodMain (P : Core.Protocol) (args : List String) : IO UInt32 := do
  match ← parse args with
  | (some pin, rest) => (guard P (periodReason pin)).main rest
  | (none, rest) => unpinned P.main rest

end Spot.ChallengerPin
