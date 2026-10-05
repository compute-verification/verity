/-
The run's wire: who the parties are, which side each is on, and every message that passes between them.

Eight parties are trusted code, four on each side. The auditor's: the coin server (the only one facing the network), the
record (an append-only log of what the coin server received and issued), the verifier and the profile. The developer's
boundary, on the CPU tier between the GPU rack and the network: the proxy (salts and hashes every message leaving the
GPUs), the outer ZK layer (masks and salts for the outer session), the recheck (recomputes the outer proof's level-0
commitment) and the send gate (the boundary's only exit). Everything else is a `Worker` or the network, the untrusted
environment, whose every output is the adversary's.
-/
import Flock.Draw
import Flock.Hm96

namespace Run

/-- A SHA-512 digest, 64 bytes. -/
abbrev Digest := ByteArray
/-- An hm96-sha512/v1 salt, 192 bytes. -/
abbrev Salt := ByteArray

inductive Side where
  | auditor
  | boundary
  deriving DecidableEq, Repr, Inhabited

inductive Party where
  | coinServer
  | record
  | verifier
  | profile
  | proxy
  | zkLayer
  | recheck
  | sendGate
  deriving DecidableEq, Repr, Inhabited

def Party.side : Party → Side
  | .coinServer | .record | .verifier | .profile => .auditor
  | .proxy | .zkLayer | .recheck | .sendGate => .boundary

def Party.all : List Party := [.coinServer, .record, .verifier, .profile, .proxy, .zkLayer, .recheck, .sendGate]

/-- The GPU rack's untrusted workers: the inner prover, the outer prover, and the store that keeps the inner proof
between the phases. -/
inductive Worker where
  | innerProver
  | outerProver
  | store
  deriving DecidableEq, Repr, Inhabited

inductive Node where
  | party (p : Party)
  | worker (w : Worker)
  /-- whoever reads the run's result: the profile's only receiver -/
  | outside
  deriving DecidableEq, Repr, Inhabited

/-- Behind the boundary: its four parties and the GPU rack. The egress premise is that the rack reaches nothing else. -/
def Node.inside : Node → Bool
  | .party p => p.side == .boundary
  | .worker _ => true
  | .outside => false

/-- What a commitment and the coins that answer it are for. -/
inductive Slot where
  | registration
  | inner (batch round : Nat)
  /-- round 0 is the outer session's `Hello`, answered with the coin tree's root -/
  | outer (round : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- The record's entries, in the order the coin server logged them. -/
inductive Entry where
  | committed (slot : Slot) (hs : Array Digest)
  | drew (d : Flock.Draw.UnitDraw)
  | issued (slot : Slot) (coins : ByteArray)
  /-- an outer message the send gate released, which the coin server answered -/
  | outer (round : Nat) (msg : ByteArray)
  | finished (record : String) (proofs : Array ByteArray)
  deriving Inhabited

/-- What verification established (`verity.proofs.profile.IntegrityProfile`): the facts checked in full, the
verifier's law over the units of one level of the partition hierarchy, the units it drew, and the claim with its error:
at most `threshold` wrong units at `level`, except with probability `δ = deltaNum / deltaDen`. -/
structure IntegrityProfile where
  program : Digest
  vstar : Digest
  exhaustive : Array String
  level : String
  law : Flock.Draw.Law
  population : Nat
  drawn : Array Nat
  threshold : Nat
  deltaNum : Nat
  deltaDen : Nat
  deriving Inhabited

inductive Body where
  /-- GPU → proxy: the registration roots (the registered values' Merkle tree tops) -/
  | roots (rs : Array Digest)
  /-- GPU → proxy: an inner prover's round message -/
  | innerMsg (batch round : Nat) (msg : ByteArray)
  /-- proxy → send gate → coin server: the salted hashes (hm96 leaves) of a slot's message, the only part of the
  registration and inner phases that leaves the boundary -/
  | commit (slot : Slot) (hs : Array Digest)
  /-- coin server → proxy → GPU: the proof units drawn after the registration commitment -/
  | draw (d : Flock.Draw.UnitDraw)
  /-- coin server → proxy → GPU: a slot's coins, drawn after its commitment -/
  | coins (slot : Slot) (c : ByteArray)
  /-- proxy → ZK layer, recheck, outer prover: V*'s witness, every committed row with its salt (never leaves) -/
  | witness (rows : Array ByteArray) (salts : Array Salt)
  /-- ZK layer → outer prover, recheck, send gate: the outer session's masks and salts (never leave) -/
  | masks (m : ByteArray) (salts : Array Salt)
  /-- outer prover → send gate: a candidate outer message; send gate → coin server: the released one -/
  | outer (round : Nat) (msg : ByteArray)
  /-- send gate → recheck: a level-0 commitment to recompute; recheck → send gate: the same, recomputed and equal -/
  | level0 (root : Digest)
  /-- outer prover → send gate → coin server: the outer session's record and proofs -/
  | finish (record : String) (proofs : Array ByteArray)
  /-- coin server → record -/
  | log (e : Entry)
  /-- record → verifier: the closed record -/
  | closed (log : Array Entry)
  /-- verifier → profile -/
  | verdict (ok : Bool) (why : String) (log : Array Entry)
  /-- profile → outside -/
  | profile (p : Option IntegrityProfile)
  deriving Inhabited

structure Frame where
  src : Node
  dst : Node
  /-- the sender's count of frames to `dst` before this one -/
  seq : Nat
  body : Body
  deriving Inhabited

/-- A frame that crosses the boundary outward: from inside to a node that isn't. -/
def Frame.leaves (f : Frame) : Bool := f.src.inside && !f.dst.inside

end Run
