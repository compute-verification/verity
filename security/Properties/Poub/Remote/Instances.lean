module

public import Poub.Codec
public import Poub.Receipts
public import Poub.Window
public import Poub.Session

/-!
# The bandwidth audit's honest instances

The deployed codec (`honestCodec`), the zero block's payload and the root a value climbs to along a path of zero
siblings (`zeroPayload`, `zeroRoot`), and the one-block session the window's, the sample's and the appends' honest
witnesses are at: the registration `reg1`, its window 0 of one pass over the whole store (`window0`), the session's
parameters (`params1`) and a state whose store ends past its registration (`overrun`), at which `appendsInOrder`'s
nontrivial witness is. They are stated here, over the verifier's own functions, so those witnesses' statements read
nothing of the proofs tree; the proofs that the session accepts are `Proofs.Properties.Poub.Remote.RecordAccepts`'.
-/

@[expose] public section

open Poub

namespace Properties.Poub

/-- A path of `log2ceil count` siblings, each 64 zero bytes. -/
def zeroPath (d : Flock.FV3.Sha512.Domain) : Array ByteArray := Array.replicate (Flock.log2ceil d.count) (zeros 64)

/-- The root `value` at `position` climbs to along `zeroPath`. -/
def zeroRoot (d : Flock.FV3.Sha512.Domain) (position : Nat) (value : ByteArray) : ByteArray :=
  d.climb position SCHEMA value (zeroPath d)

/-- The zero block's decoding under `k`. -/
def zeroPayload (k : Codec.Key) : ByteArray :=
  match Codec.decode (zeros Codec.BLOCK_BYTES) 0 k none with
  | .ok (some p) => p
  | _ => .empty

/-- The deployed codec: p1024, three rounds, masked. -/
def honestCodec : Codec.Codec := ⟨Codec.P1024, 3, true⟩

end Properties.Poub

namespace Properties.Poub.Line.Honest

open _root_.Poub

/-- The receipt of the zero block under seed `s`: position 0's leaf, the plan's only one. -/
def leaf1 (reg : Registration) (s : ByteArray) : ByteArray :=
  (receipter reg.receipt s reg.rows).receipt 0 (zeros Codec.BLOCK_BYTES)

/-- The receipt tree's root of a pass over the one-block store under seed `s`. -/
def root1 (reg : Registration) (s : ByteArray) : ByteArray :=
  zeroRoot (treeDomain s (Plan.whole 1)) 0 (hidden (leaf1 reg s) (zeros SALT_BYTES))

/-- Window 0 under challenge `c`, one pass over the whole store, chained to the registration. -/
def window1 (reg : Registration) (c root : ByteArray) : WindowRecord :=
  ⟨0, c, reg.digest, #[⟨root, Plan.whole 1, none⟩]⟩

/-- One window of one pass, one sample of one receipt; no deadline, no stated length, no pace. -/
def params1 : _root_.Poub.Session.Params :=
  { deadlineNs := 0, windows := 1, samples := 1, minPasses := 1, windowNs := 0, lifetime := 1 }

/-- The one-block registration of node `zeros 1` under the zero codec seed and the deployed codec, sketch-sha512
receipts of 16 rows, its payload tree holding `zeroPayload` at block 0. -/
def reg1 : Registration :=
  let key := Codec.key (zeros Codec.SEED_BYTES) honestCodec
  let pd := Flock.FV3.Sha512.Domain.make (Poub.sha512 (PAYLOAD_DOMAIN ++ frameBytes (zeros 1))) 0 0 1
  { node := zeros 1, blocks := 1, payloadRoot := zeroRoot pd 0 (hidden (zeroPayload key) (zeros SALT_BYTES)),
    codecSeed := zeros Codec.SEED_BYTES, codec := honestCodec, receipt := .sketch, rows := 16, key }

end Properties.Poub.Line.Honest

namespace Properties.Poub

open Properties.Poub.Line.Honest

/-- `RecordAccepts`'s window 0 under challenge `zeros 64`. -/
def window0 : WindowRecord := window1 reg1 (zeros 64) (root1 reg1 (seed (zeros 64) 0 reg1.digest))

/-- A state of the one-block registration whose store ends at block 2, with no appends. -/
def overrun : _root_.Poub.Session.State :=
  { _root_.Poub.Session.State.init params1 reg1 #[] #[] 0 with extent := 2 }

end Properties.Poub
