import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Sigma

/-!
# Scratch: auditor-trusted devices as parties of the play

Design probe only, outside every Lake library; elaborate with `lake env lean scratch/devices/Devices.lean` from
`security/`. It checks that the proposed signatures elaborate and that the play with no further party is the play of
`main` up to the new `Exchange` field. The stubs stand in for `Hash.Msg` (`Definitions/Core/Hash.lean`),
`Flock.CoinTree.Spec`, `Hash.Assumptions.LmsForger` and `Verity.Game` (copied from `Definitions/Core/Game.lean`).
The tree frames of `releases` are cut to `.ret []`: their shape is unchanged by the proposal.
-/

namespace Hash
abbrev Msg := List UInt8
end Hash

namespace Hash.Assumptions
structure LmsForger where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  key : Msg
  signed : S → List Msg
  out : S → Option (Msg × Msg)
  cost : S → ℕ
end Hash.Assumptions

namespace Flock.CoinTree
structure Spec where
  streams : Array ℕ
  roundsLog : ℕ
  block : ℕ
end Flock.CoinTree

namespace Verity

inductive Game (α : Type) : Type 1 where
  | ret : α → Game α
  | send (M : Type) : (M → Game α) → Game α
  | coin (C : Type) [Fintype C] [Nonempty C] : (C → Game α) → Game α

namespace Game
variable {α β : Type}
def bind : Game α → (α → Game β) → Game β
  | .ret a, f => f a
  | .send M k, f => .send M fun m => bind (k m) f
  | @Game.coin _ C iF iN k, f => @Game.coin _ C iF iN fun c => bind (k c) f
def map (f : α → β) (g : Game α) : Game β := g.bind fun a => .ret (f a)
def Strategy : Game α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × Strategy (k m)
  | @Game.coin _ C _ _ k => (c : C) → Strategy (k c)
end Game

/-! ## Proposed `security/Definitions/Core/Party.lean` (imports `Definitions.Core.Hash`)

`Device`, `Challenger`, `Session`, `Record`, `Signature` and `Ledger` are guarded concepts
(`tests/test_lean_vocabulary.py`), and `Device` is reserved for `Accounting.Device`'s move into Core, so the
auditor-trusted device is a `Party` and its registered key a `Pin`. -/

/-- The role of an auditor-trusted device: which checks read what it signs. -/
inductive Role where
  /-- SPOT's live challenger: sampled proofs' coins and PoUW's salts -/
  | challenger
  /-- the memory challenger: PoUS's challenges, PoUB-P2's draws, the wipe's labels and reads -/
  | memoryChallenger
  /-- a network certifier: the rows it stamps -/
  | networkCertifier
  deriving DecidableEq, Repr

/-- **An auditor-trusted device's pin**: its role and its registered LMS public key. -/
structure Pin where
  role : Role
  key : Hash.Msg

/-- **The deployment's pins for one monitoring period**: its devices' keys, SPOT's live challenger first, and the
sessions the auditor designated for each role, all fixed before the prover's first move. -/
structure Pins where
  keys : List Pin
  sessions : Role → List Hash.Msg

/-- The keys pinned for role `r`: the only keys a check of that role trusts. -/
def Pins.keysOf (P : Pins) (r : Role) : List Hash.Msg := (P.keys.filter (·.role = r)).map (·.key)

/-- **An auditor-trusted device as a party of a play**, over its log type `L`: its pin, the coin families it releases
from, the bits it samples at the close, what it signs before any coin (`first`, from the request) and at the close
(`last`, from the request, its log, the close's bytes and its sample), and `sign i m`, its signature on `m` as its
`i`-th. -/
structure Party (L : Type) where
  pin : Pin
  families : List (String × ℕ)
  draws : ℕ
  first : ByteArray → List Hash.Msg
  last : ByteArray → L → ByteArray → List Bool → List Hash.Msg
  sign : ℕ → Hash.Msg → Hash.Msg

namespace Party
variable {L : Type}
def signs (D : Party L) (a : ByteArray) (log : L) (b : ByteArray) (ρ : List Bool) : List Hash.Msg :=
  D.first a ++ D.last a log b ρ
def signatures (D : Party L) (a : ByteArray) (log : L) (b : ByteArray) (ρ : List Bool) :
    List (Hash.Msg × Hash.Msg) :=
  (D.signs a log b ρ).mapIdx fun i m => (m, D.sign i m)
end Party

/-! ## `Properties/Verity/Game.lean`, as proposed -/

inductive Frame where
  | fresh (label : String)
  | hello (spec : Flock.CoinTree.Spec)
  | opening (i s n : ℕ)

structure Release where
  frame : Frame
  answers : ByteArray
  bits : List Bool

def takeCoin (l : String) : List (String × ℕ) → Option (List (String × ℕ))
  | [] => none
  | (l', n) :: fs =>
    if l' = l then (match n with | 0 => none | n + 1 => some ((l', n) :: fs)) else (takeCoin l fs).map ((l', n) :: ·)

def maxTreeLines : ℕ := 2 ^ 23
def frames (fs : List (String × ℕ)) : ℕ := maxTreeLines + (fs.map Prod.snd).sum

/-- `releases` unchanged but for the cut tree frames. -/
def releases : ℕ → List (String × ℕ) → Game (List Release)
  | 0, _ => .ret []
  | k + 1, fs => .send (Option (Frame × ByteArray)) fun
    | none => .ret []
    | some (.fresh l, b) =>
      match takeCoin l fs with
      | none => .ret []
      | some fs' => .coin (Fin 512 → Bool) fun x => (releases k fs').map fun log => ⟨.fresh l, b, List.ofFn x⟩ :: log
    | some (.hello _, _) => .ret []
    | some (.opening _ _ _, _) => .ret []

/-- **SPOT's live challenger as a party**: `Signer`'s four fields keep their names, so `D.first`, `D.draws`, `D.last`
and `D.sign` read as on `main`; `D.signs` and `D.signatures` now resolve to `Party`'s. -/
abbrev Signer := Party (List Release)

/-- The further parties' coin families, in order. -/
def Parties.families (Ds : List Signer) : List (String × ℕ) := (Ds.map (·.families)).flatten

/-- A release is party `P`'s: a coin of one of its families. Tree frames are only SPOT's live challenger's. -/
def Signer.owns (P : Signer) (r : Release) : Bool :=
  match r.frame with
  | .fresh l => P.families.any (·.1 == l)
  | _ => false

/-- Party `P`'s own releases in the play's log, the lines it prints and signs. -/
def Signer.logOf (P : Signer) (log : List Release) : List Release := log.filter P.owns

/-- SPOT's live challenger's releases: the log without the further parties'. At no further party it is the log
itself, by `rfl`, so the play with no further party is `main`'s. -/
def spotLog : List Signer → List Release → List Release
  | [], log => log
  | Ds@(_ :: _), log => log.filter fun r => !Ds.any (·.owns r)

/-- `Exchange` gains `signedBy`: each further party's signatures, in the parties' order. `log` is still every
release, `signed` still SPOT's live challenger's. -/
structure Exchange where
  params : ByteArray
  files : Array ByteArray
  commitment : ByteArray
  request : ByteArray
  log : List Release
  close : ByteArray
  drawn : List Bool
  signed : List (Hash.Msg × Hash.Msg)
  signedBy : List (List (Hash.Msg × Hash.Msg))
  evidence : ByteArray
  blobs : Array ByteArray

/-- The further parties sample nothing at the close (their coins are family coins, released one by one), so the close's
one coin stays SPOT's live challenger's and `Coupling.play_couples` keeps its last coordinate. -/
def finish (D : Signer) (Ds : List Signer) (θ : ByteArray) (F : Array ByteArray) (c a : ByteArray)
    (log : List Release) : Game Exchange :=
  .send ByteArray fun b => .coin (Fin D.draws → Bool) fun ρ => .send (ByteArray × Array ByteArray) fun e =>
    .ret ⟨θ, F, c, a, log, b, List.ofFn ρ, D.signatures a (spotLog Ds log) b (List.ofFn ρ),
      Ds.map (fun P => P.signatures a (P.logOf log) b []), e.1, e.2⟩

/-- The play: the further parties' families first, so at `Ds = []` the families are `fs` by `rfl`. -/
def play (fs : List (String × ℕ)) (D : Signer) (Ds : List Signer) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) : Game Exchange :=
  .send ByteArray fun a =>
    (releases (frames (Parties.families Ds ++ fs)) (Parties.families Ds ++ fs)).bind (finish D Ds θ F c a)

/-- `main`'s play (`Game.lean:511-512`, `finish` at 505-507), with the new field empty. -/
def playMain (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    Game Exchange :=
  .send ByteArray fun a => (releases (frames fs) fs).bind fun log =>
    .send ByteArray fun b => .coin (Fin D.draws → Bool) fun ρ => .send (ByteArray × Array ByteArray) fun e =>
      .ret ⟨θ, F, c, a, log, b, List.ofFn ρ, D.signatures a log b (List.ofFn ρ), [], e.1, e.2⟩

/-- **With no further party the play is `main`'s, definitionally**: the SPOT tree's proofs see the same game. -/
example (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    play fs D [] θ F c = playMain fs D θ F c := rfl

/-! ## The composite verifier, pinned -/

/-- Stand-ins for `Registration.Composite`, `Subprotocol`, `Sent` and the compiled verifier's acceptance. -/
inductive Composite | onsite | remote
inductive Subprotocol | sampled | pouw | inferenceOnly | pous | poub | network | wipe
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray
  blobs : Array ByteArray
opaque compiledAccepts : Composite → List (Role × ByteArray) → List (Role × ByteArray) → Sent → Bool

/-- What the compiled verifier is given: the pins as bytes (code never requires `security/`). -/
def Pins.wire (P : Pins) : List (Role × ByteArray) × List (Role × ByteArray) :=
  (P.keys.map fun p => (p.role, ⟨p.key.toArray⟩),
    [Role.challenger, .memoryChallenger, .networkCertifier].flatMap fun r =>
      (P.sessions r).map fun s => (r, ⟨s.toArray⟩))

/-- **`Accepts` reads the pins** (`Game.lean:206-207` today reads `verifier k` alone): the compiled composite
refuses every component whose parameters name a key or a session the pins don't. -/
def Accepts (P : Pins) (k : Composite) (s : Sent) : Prop := compiledAccepts k P.wire.1 P.wire.2 s = true

def Exchange.sent (o : Exchange) : Sent := ⟨o.params, o.commitment, o.evidence, o.files ++ o.blobs⟩

/-! ## `Vocab`'s changed fields (`Game.lean:652-746`), alone -/

structure VocabDevices where
  families : ByteArray → List (String × ℕ)
  /-- entry 0, SPOT's live challenger, unchanged -/
  signer : ByteArray → Signer
  /-- the further parties, each with its own families, fixed by the parameters before any coin -/
  devices : ByteArray → List Signer
  /-- the deployment's pins: not a function of the parameters -/
  pins : Pins
  /-- every party is the pinned one, at every parameters: a θ naming another key reaches no party -/
  pinned : ∀ θ, (signer θ :: devices θ).map (·.pin) = pins.keys
  /-- the further parties' labels are none of SPOT's, so `spotLog` drops none of SPOT's releases -/
  disjoint : ∀ θ, ∀ l ∈ (Parties.families (devices θ)).map (·.1), l ∉ (families θ).map (·.1)
  coinsRead : Subprotocol → Sent → Option (List Release)
  /-- the party whose signed log subprotocol `a`'s check reads (0 is SPOT's live challenger); none for the policy -/
  reads : Subprotocol → Option (Fin pins.keys.length)
  forger : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Game.Strategy (play (families θ) (signer θ) (devices θ) θ F c) → Hash.Assumptions.LmsForger
  /-- each forger is against the pinned key of the party its check reads, never a key θ names -/
  forger_pinned : ∀ a θ F c σ i, reads a = some i → (forger a θ F c σ).key = (pins.keys.get i).key
  signatures : ByteArray → ℕ
  /-- every party signs at most `Q` messages in a play -/
  signs_le : ∀ θ, ∀ P ∈ signer θ :: devices θ, ∀ a log b ρ, (P.signs a log b ρ).length ≤ signatures θ

namespace VocabDevices
variable (V : VocabDevices)

def game (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) : Game Exchange :=
  play (V.families θ) (V.signer θ) (V.devices θ) θ F c

/-- The log party `i` printed and signed in the play that ended with `o`. -/
def logAt (o : Exchange) : ℕ → List Release
  | 0 => spotLog (V.devices o.params) o.log
  | i + 1 => ((V.devices o.params)[i]?.map (·.logOf o.log)).getD []

/-- **`Logged`** (`Game.lean:837`): the log `a`'s check read is the log of the party it reads, no longer the whole log. -/
def Logged (a : Subprotocol) (o : Exchange) : Prop :=
  ∀ log ∈ V.coinsRead a o.sent, log = match V.reads a with
    | some i => V.logAt o i.val
    | none => o.log

/-- The verdict reads the pins. -/
def Verdict (k : Composite) (o : Exchange) : Prop := Accepts V.pins k o.sent

end VocabDevices

/-! ## `Instance` (`Proofs/Properties/Verity/Sampled/Instance.lean:163-173`) -/

/-- The deployment's pins enter `base`, so #9410's `Instance.target_of_base` stays definitional: `vocab` replaces
only SPOT's names. `session` is taken (SPOT's `Live` session), so the field is `pinned`. -/
structure InstanceDevices where
  base : VocabDevices
  /-- the deployment's pins, registered before the period -/
  registeredPins : Pins
  pins_eq : base.pins = registeredPins
  /-- SPOT's key is the first pin, the challenger's (`Device.pubOf` and `Binding.keyOf` read this, not θ) -/
  spot_first : ∃ p ps, registeredPins.keys = p :: ps ∧ p.role = .challenger

/-! ## `Coupling.UpFrontAll` (`Properties/Verity/Coupling.lean:74`) -/

def UpFront (fs : List (String × ℕ)) : Type := (i : Fin fs.length) → Fin (fs.get i).2 → Fin 512 → Bool

def UpFrontAll (fs : List (String × ℕ)) (D : Signer) (Ds : List Signer) : Type :=
  UpFront (Parties.families Ds ++ fs) × (Fin D.draws → Bool)

example (fs : List (String × ℕ)) (D : Signer) : UpFrontAll fs D [] = (UpFront fs × (Fin D.draws → Bool)) := rfl

end Verity

/-! ## PoUB-P2's signed case: the contract `P2Protocol.check` builds against -/

namespace RemoteAudit.P2Protocol

/-- P2's parameters gain the memory challenger's trusted keys and the designated session; the composite refuses them
unless they are the pins' (`Pins.keysOf .memoryChallenger`, `Pins.sessions .memoryChallenger`). -/
structure Params where
  notBeforeNs : Nat
  request : String
  session : ByteArray
  trusted : Array ByteArray

/-- The memory challenger's signed close over its P2 lines: its key, the line count, the lines' SHA-512 chain, the
LMS signature on `closeMessage`. -/
structure SignedClose where
  publicKey : ByteArray
  lines : Nat
  chain : ByteArray
  signature : ByteArray

/-- The message the memory challenger signs at the close, domain-separated from SPOT's `closeMessage`. -/
opaque closeMessage (ident session : ByteArray) (lines : Nat) (chain : ByteArray) : ByteArray

/-- One P2 line as the memory challenger logged it: the root it answered, its 64-byte coin, the openings. -/
structure Line where
  root : ByteArray
  coin : ByteArray
  openings : Array ByteArray

structure Evidence where
  log : Array Line
  close : SignedClose

opaque chainOf : Array Line → ByteArray
opaque lmsVerify : ByteArray → ByteArray → ByteArray → Bool

/-- **The signature check P2's `check` runs before its replay**: the close is under a trusted key, over the designated
session, the log's line count and chain. -/
def signedOk (p : Params) (e : Evidence) : Bool :=
  p.trusted.contains e.close.publicKey && e.close.lines == e.log.size && e.close.chain == chainOf e.log &&
    lmsVerify e.close.publicKey (closeMessage (e.close.publicKey.extract 0 16) p.session e.log.size e.close.chain)
      e.close.signature

end RemoteAudit.P2Protocol

namespace Verity

/-- **P2's `coinsRead`**: its lines as the memory challenger's releases of its P2 family, which `Logged .poub` equates
with the memory challenger's `logAt`. -/
def p2Release (label : String) (l : RemoteAudit.P2Protocol.Line) : Release :=
  ⟨.fresh label, l.root, l.coin.toList.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit i⟩

end Verity
