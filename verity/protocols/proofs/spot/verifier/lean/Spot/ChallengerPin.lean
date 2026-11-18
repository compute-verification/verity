/-
The verifier's pin of the challenger's key: the SHA-512 of its public key, which the auditor takes from its own checkout
(`--challenger-key-sha512 H`, with `--challenger-retired FILE`, a `challenger-key/v1` file's `retired` list), never from
the case. One pin serves every key the challenger signs under: SPOT's coins (`VStar.Recursive.ChallengerPin`) and PoUW's
draw coin, a period's `key` (`PouwVerifier.Period.pinned`). A key that is none, retired or other than the pin's is
refused by name (`keyReason`). A run with no pin says so first, on stderr (`UNPINNED`), its answer line unchanged.

The pin reads the parameters the protocol itself parsed (`pinned`): the value it tests is the value the protocol's check
is handed, so the key it pins is the key the check is handed (`pinned_accept`).
-/
module

public import CoreProtocol.Blobs
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

/-- `keyReason` on a PoUW period's `key` as its parameters hold it: the challenger's that signs its draw coin. -/
def periodReason (pin : Pin) : Json → Option String
  | .null => some NONE
  | k => match Challenger.ChallengerKey.ofJson k with
    | .ok k => keyReason pin (some k)
    | .error _ => some UNREAD

/-- `P`, refusing what `why` names in the parameters `P` parsed, after its own parse and before anything else. -/
def pinned (P : Core.Protocol) (why : P.Params → Option String) : Core.Protocol :=
  { P with params := fun j => match P.params j with
      | .error e => .error e
      | .ok p => match why p with
        | some r => .error r
        | none => .ok p }

/-- `pinned` on a protocol that reads blobs. -/
def pinnedB (P : Core.BlobProtocol) (why : P.Params → Option String) : Core.BlobProtocol :=
  { P with params := (pinned P.toProtocol why).params }

theorem pinned_params {P : Core.Protocol} {why : P.Params → Option String} {j : Json} {p : P.Params}
    (h : (pinned P why).params j = .ok p) : P.params j = .ok p ∧ why p = none := by
  simp only [pinned] at h
  split at h
  · cases h
  · rename_i q hq
    split at h
    · cases h
    · rename_i hw
      cases h
      exact ⟨hq, hw⟩

/-- What `Q` accepts, its parameters parsed. -/
theorem accept_params (Q : Core.Protocol) (params commitment evidence : ByteArray)
    (h : (Q.answer params commitment evidence).verdict = .accept) :
    ∃ j p, Core.canonicalJson params = .ok j ∧ Q.params j = .ok p := by
  unfold Core.Protocol.answer at h
  split at h
  · simp at h
  · rename_i p hp
    cases hj : Core.canonicalJson params with
    | error e => rw [hj] at hp; cases hp
    | ok j => rw [hj] at hp; exact ⟨j, p, rfl, hp⟩

/-- What `pinned P why` accepts, `P` parsed to parameters `why` names nothing in, and those are what its check read. -/
theorem pinned_accept (P : Core.Protocol) (why : P.Params → Option String) (params commitment evidence : ByteArray)
    (h : ((pinned P why).answer params commitment evidence).verdict = .accept) :
    ∃ j p, Core.canonicalJson params = .ok j ∧ P.params j = .ok p ∧ why p = none := by
  obtain ⟨j, p, hj, hp⟩ := accept_params (pinned P why) params commitment evidence h
  obtain ⟨h1, h2⟩ := pinned_params (P := P) (j := j) (p := p) hp
  exact ⟨j, p, hj, h1, h2⟩

theorem pinnedB_accept (P : Core.BlobProtocol) (why : P.Params → Option String)
    (params commitment evidence : ByteArray) (blobs : Array ByteArray)
    (h : ((pinnedB P why).answer params commitment evidence blobs).verdict = .accept) :
    ∃ j p, Core.canonicalJson params = .ok j ∧ P.params j = .ok p ∧ why p = none := by
  obtain ⟨j, p, hj, hp⟩ := accept_params ((pinnedB P why).on blobs) params commitment evidence h
  obtain ⟨h1, h2⟩ := pinned_params (P := P.toProtocol) (why := why) (j := j) (p := p) hp
  exact ⟨j, p, hj, h1, h2⟩

theorem keyReason_none {pin : Pin} {x : Option Challenger.ChallengerKey} (h : keyReason pin x = none) :
    ∃ k, x = some k ∧ sha512Hex k.pub = pin.sha512 ∧ pin.retired.contains (Flock.hexOf k.pub) = false := by
  cases x with
  | none => simp [keyReason] at h
  | some k =>
    simp only [keyReason] at h
    split at h
    · cases h
    · rename_i hr
      split at h
      · rename_i heq; exact ⟨k, rfl, beq_iff_eq.mp heq, by simpa using hr⟩
      · cases h

theorem periodReason_none {pin : Pin} {k : Json} (h : periodReason pin k = none) :
    ∃ key, Challenger.ChallengerKey.ofJson k = .ok key ∧ sha512Hex key.pub = pin.sha512 ∧
      pin.retired.contains (Flock.hexOf key.pub) = false := by
  unfold periodReason at h
  split at h
  · cases h
  · split at h
    · rename_i key hk
      obtain ⟨k', hk', hs, hr⟩ := keyReason_none h
      cases hk'
      exact ⟨key, hk, hs, hr⟩
    · cases h

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

/-- `main` on `args`, saying first on stderr that no pin is taken. -/
def unpinned (main : List String → IO UInt32) (args : List String) : IO UInt32 := do
  unless args == ["id"] do IO.eprintln UNPINNED
  main args

/-- A command line under the pin `parse` reads ahead of it: `pin`'s main under a pin, `none`'s, announced, without. -/
def pinMain (pin : Pin → List String → IO UInt32) (none : List String → IO UInt32) (args : List String) :
    IO UInt32 := do
  match ← parse args with
  | (some p, rest) => pin p rest
  | (Option.none, rest) => unpinned none rest

end Spot.ChallengerPin
