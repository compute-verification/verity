/-
The outer statement both sides check: V*'s public words, built from the run's public data (the proxy's hashes, the coin
server's coins and draw, and the comb computed from the public inner circuit at those coins), and `Flock.Zk.verify` on
the outer session with V*'s setup pinned by digest. The verifier runs it on the closed record; the send gate runs the
same function on what it is about to release, so a proof that leaves the boundary is one the verifier accepts.
-/
import Run.Agreement

namespace Run

/-- The run's public data and the outer session's result, as the record keeps them. -/
structure Parsed where
  reg : Array Digest := #[]
  draw : Option Flock.Draw.UnitDraw := none
  /-- each inner slot's hashes and the coins that answered them, in order -/
  inner : Array (Slot × Array Digest × ByteArray) := #[]
  /-- each outer round's released message and the opening that answered it -/
  outer : Array (Nat × ByteArray × ByteArray) := #[]
  record : String := ""
  proofs : Array ByteArray := #[]
  finished : Bool := false
  deriving Inhabited

namespace Statement

/-- The lincheck's comb: the public inner circuit's matrices folded at the inner coins (`Setup.fold`; in circuit,
`CombEntry`/`CombChunk`/`CombColumn`). Computed by whoever checks; no message carries it. -/
def comb (inner : Flock.Setup) (coins : Array ByteArray) : Array Flock.F128 := sorry

/-- V*'s public words: the registration's and every inner round's hashes, the coins answering them, the draw, and the
comb, in V*'s layout (`rec/inner-transcript/v0`'s `public.json`, as words). -/
def vstarPublic (A : Agreement) (p : Parsed) : ByteArray := sorry

/-- The outer record's `coin_tree` is the one the coin server committed at `Hello`: its root and key are the ones that
answered round 0 (`Flock.Zk.verify` checks the record's coins against its own `coin_tree`, not against the server). -/
def coinsIssued (p : Parsed) : Except String Unit := sorry

/-- The outer check: V*'s setup is the agreement's digest, the public words are the record's, `Flock.Zk.verify`
accepts, and the outer coins are the coin server's. -/
def verify (A : Agreement) (p : Parsed) : Except String Unit := do
  if A.vstarSetup.digest != A.vstar then throw "V*: the setup is not the agreement's digest"
  let pub := vstarPublic A p
  let tab : Flock.Setup :=
    { A.vstarSetup with publics := fun w => if w == pub then pure () else throw "S8: V*'s public words are not the run's" }
  Flock.Zk.verify A.outer #[tab] p.record p.proofs
  coinsIssued p

end Statement

end Run
