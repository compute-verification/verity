module

public import Definitions.Core.Hash

/-!
# The auditor's devices as parties of a play

An audit's coins and signed logs come from the auditor's own devices at the datacenter or beside it: SPOT's live
challenger, the memory challengers and the network certifiers (`Role`). Each holds a registered LMS key (`AuditorPin`),
and the deployment pins, before the monitoring period, every device's key and the sessions it designated for each role
(`AuditorPins`): a check trusts a signature only under a pinned key and coins only from a designated session, never a
key or a session the prover's parameters name.

In a play, a device is a party (`AuditorParty L`, over the type `L` of its log): its pin, the coin families it releases
from, the bits it samples at the close, what it signs before any coin (`first`) and at the close (`last`), and its
signature on its `i`-th message (`sign`). `signs` lists what it signs in a play, `signatures` pairs each with its
signature. A memory challenger's pin names the network certifier whose marks it reads (`AuditorPin.marks`), the auditor's
wiring, which the deployment pins with the keys (`AuditorPins.Wired`).
-/

@[expose] public section

namespace Verity

/-- **An auditor device's role**: which checks read what it signs. -/
inductive Role where
  /-- SPOT's live challenger: sampled proofs' coins and PoUW's salts -/
  | challenger
  /-- the memory challenger: PoUS's challenges, PoUB's samples, the wipe's labels and reads -/
  | memoryChallenger
  /-- a network certifier: the rows it stamps -/
  | networkCertifier
  deriving DecidableEq, Repr

/-- **An auditor device's pin**: its role, its registered public key, the LMS parameters `(N, W, H)` it is under; for
a memory challenger the key of the network certifier whose marks it reads and the servers whose memory it is pinned to
challenge; and for a network certifier the link it is pinned for, its key's session and the start of the stretch of
that link's link-windows its key signs in that session, which runs up to the next pin's start for the link. A link's
entries, one pin each, are its certifier pins in their order. -/
structure AuditorPin where
  role : Role
  key : Hash.Msg
  scheme : ℕ × ℕ × ℕ
  marks : Option Hash.Msg := none
  link : String := ""
  session : Hash.Msg := []
  start : ℕ := 0
  servers : List Hash.Msg := []

/-- **The deployment's pins for one monitoring period**: its devices' pins, SPOT's live challenger first, and the
sessions the auditor designated for each role, all fixed before the prover's first move. -/
structure AuditorPins where
  keys : List AuditorPin
  sessions : Role → List Hash.Msg

namespace AuditorPins

/-- The keys pinned for role `r`, the only keys a check of that role trusts. -/
def keysOf (P : AuditorPins) (r : Role) : List Hash.Msg := (P.keys.filter (·.role = r)).map (·.key)

/-- **The pins are wired**: only a memory challenger names a mark source, each memory challenger names one, and it is
a pinned network certifier's key. -/
def Wired (P : AuditorPins) : Prop :=
  ∀ p ∈ P.keys, match p.marks with
    | some k => p.role = .memoryChallenger ∧ k ∈ P.keysOf .networkCertifier
    | none => p.role ≠ .memoryChallenger

end AuditorPins

/-- **An auditor device as a party of a play**, over its log type `L`: its pin, the coin families it releases from, the
bits it samples at the close, what it signs before any coin (`first`, from the request) and at the close (`last`, from
the request, its log, the close's bytes and its sample), and `sign i m`, its signature on `m` as its `i`-th message. -/
structure AuditorParty (L : Type) where
  pin : AuditorPin
  families : List (String × ℕ)
  draws : ℕ
  first : ByteArray → List Hash.Msg
  last : ByteArray → L → ByteArray → List Bool → List Hash.Msg
  sign : ℕ → Hash.Msg → Hash.Msg

namespace AuditorParty

variable {L : Type}

/-- Everything the party signs in a play: what it signs before any coin, then at the close. -/
def signs (D : AuditorParty L) (a : ByteArray) (log : L) (b : ByteArray) (ρ : List Bool) : List Hash.Msg :=
  D.first a ++ D.last a log b ρ

/-- The party's signatures in a play: each message it signs, with its signature as the message's place among them. -/
def signatures (D : AuditorParty L) (a : ByteArray) (log : L) (b : ByteArray) (ρ : List Bool) :
    List (Hash.Msg × Hash.Msg) :=
  (D.signs a log b ρ).mapIdx fun i m => (m, D.sign i m)

end AuditorParty

end Verity
