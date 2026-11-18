/-
The verifier's pin of the challenger's key: the SHA-512 of its public key, which the auditor takes from its own checkout
(`--challenger-key-sha512 H`, with `--challenger-retired FILE`, a `challenger-key/v1` file's `retired` list), never from
the case. A case's parameters whose level names no challenger, a retired key or any other key than the pin's are refused
by name (`reason`), in SPOT's executable (`sampled`) and in the composite, over its SPOT component's parameters
(`compositeReason`).

The window's root needs no second comparison: SPOT draws its coins (`Challenger.sampleCoins`) only from the ledger's open
whose root is the parameters' window root, in a ledger checked under the case's key (`auditedOpen`), withheld on that
root (`withheldOn`). Under the pinned key a root the challenger never opened is refused there.
-/
module

public import VStarRecursive.Sampled
meta import VStarRecursive.Sampled

@[expose] public section

namespace VStar.Recursive.ChallengerPin

open Lean (Json)
open Flock (V)
open Spot

/-- The pin: `sha512`, lowercase hex of 64 bytes; `retired`, the retired public keys as lowercase hex. -/
structure Pin where
  sha512 : String
  retired : Array String := #[]

def NONE : String :=
  "challenger-key-none: a level of the case's parameters names no challenger, and the verifier pins one " ++
    "(--challenger-key-sha512)"

def RETIRED (h : String) : String :=
  s!"challenger-key-retired: the case's challenger key (public key SHA-512 {h}) is one the verifier's key file retires"

def OTHER (h pin : String) : String :=
  s!"challenger-key-not-pinned: the case's challenger key (public key SHA-512 {h}) is not the verifier's pin {pin}"

/-- The challenger key each level of `p` names: one level's, or a two-level case's boundary and interior. -/
def levels : Sampled.Params → List (Option Challenger.ChallengerKey)
  | .inl p => [p.base.challenger.map (·.1)]
  | .inr p => [p.boundary.challenger.map (·.1), p.interior.challenger.map (·.1)]

def keyReason (pin : Pin) : Option Challenger.ChallengerKey → Option String
  | none => some NONE
  | some k =>
    let h := sha512Hex k.pub
    if h == pin.sha512 then none
    else if pin.retired.contains (Flock.hexOf k.pub) then some (RETIRED h)
    else some (OTHER h pin.sha512)

/-- Why `p` is refused under `pin`: the first level naming no challenger, a retired key or another key. -/
def reason (pin : Pin) (p : Sampled.Params) : Option String :=
  (levels p).findSome? (keyReason pin)

/-- `reason` on SPOT's parameters `j`; none where they don't parse, which the protocol refuses itself. -/
def jsonReason (pin : Pin) (j : Json) : Option String :=
  match Sampled.paramsOfJson j with
  | .ok p => reason pin p
  | .error _ => none

/-- `jsonReason` on each SPOT component of a composite's parameters `{"components":[{"id","params"},...]}`. -/
def compositeReason (pin : Pin) (j : Json) : Option String :=
  match getD j "components" with
  | .arr xs => xs.toList.findSome? fun x =>
    if getD x "id" == Json.str Sampled.PROTOCOL then jsonReason pin (getD x "params") else none
  | _ => none

/-- `P`, refusing parameters `why` names before its own parse reads them. -/
def guarded (P : Core.BlobProtocol) (why : Json → Option String) : Core.BlobProtocol :=
  { P with params := fun j => do
      if let some r := why j then throw r
      P.params j }

/-- SPOT's protocol under `pin`. -/
def sampled (pin : Pin) : Core.BlobProtocol := guarded Sampled.protocol (jsonReason pin)

/-- A composite under `pin`, over its SPOT component. -/
def composite (pin : Pin) (P : Core.BlobProtocol) : Core.BlobProtocol := guarded P (compositeReason pin)

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

/-- The command line of a composite `P` (`Core.BlobProtocol.main`), under the pin `parse` reads ahead of it. -/
def compositeMain (P : Core.BlobProtocol) (args : List String) : IO UInt32 := do
  match ← parse args with
  | (some pin, rest) => (composite pin P).main rest
  | (none, rest) => P.main rest

end VStar.Recursive.ChallengerPin
