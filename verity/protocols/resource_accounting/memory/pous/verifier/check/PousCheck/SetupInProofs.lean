module

public import CoreProtocol.Protocol
public import PousCheck.SetupProtocol

/-!
# PoUS's setup as a sampled proof about the store

`verity/pous/p2/v4/setup-in-proofs` is the setup check that reads no block of `C*`. The store's blocks are committed and
their root registered; the auditor's challenger then samples `SAMPLE` of them on its live coins (`samplingDesign`); each sampled
block is proved to decode to the registered rows of `W` at its place, one instance of the setup unit
`PousP2SetupCheck_v1` a block (`verity/protocols/accounting/pous/prover/window.py`). The check takes the parameters
and the commitment as `SetupProtocol` does, and in place of `C*` the claim (`Claim`, the Python `setup_proof.Claim`'s
last four fields):

~~~text
evidence  {"law":"rate:8828/B"|null,"outcome":hex,"store":hex,"unit":hex}   canonical JSON
~~~

It decides the parameters' validity, the salt's size and that the claim's sampling design is `samplingDesign q`. Reading SPOT's
outcome, that the sampled audit accepted over the unit `unit` at that design on the store's root, is not built: until it
is, the check rejects every claim, naming the goal. On acceptance it publishes `SetupProtocol.Public` with the store's
root as `root`, so the timed audit's input keeps its form.
-/

@[expose] public section

namespace PousCheck.SetupInProofs

open Lean (Json)
open Flock (V)

def ID : String := "verity/pous/p2/v4/setup-in-proofs"

def GOAL : String := "memory-accounting/pous-setup-as-proof"

/-- The blocks the challenger samples: a uniform `8,828`-subset misses `⌊B/100⌋ + 1` wrong blocks with probability at
most `0.99^8828 ≤ 2^-128`, at any store size. -/
def SAMPLE : Nat := 8828

/-- The sampling design over the store's `q.blocks` blocks: `rate:SAMPLE/B`, or every block (`none`) when there are no
more than `SAMPLE`. -/
def samplingDesign (q : P2.Params) : Option String :=
  if q.blocks ≤ SAMPLE then none else some s!"rate:{SAMPLE}/{q.blocks}"

/-- What the check reads in place of `C*`: the store's registered root, the setup unit's Program digest, the sampling
design and SPOT's outcome record. -/
structure Claim where
  store : ByteArray
  unit : ByteArray
  samplingDesign : Option String
  outcome : ByteArray

def claim (b : ByteArray) : V Claim := do
  let j ← Core.canonicalJson b
  let ctx := "the setup's claim"
  let samplingDesign ← match j.getObjVal? "law" with
    | .ok .null => pure none
    | .ok (.str s) => pure (some s)
    | _ => throw s!"{ctx}: law is a string or null"
  return ⟨← Flock.unhex (← Flock.J.str j "store" ctx), ← Flock.unhex (← Flock.J.str j "unit" ctx), samplingDesign,
    ← Flock.unhex (← Flock.J.str j "outcome" ctx)⟩

/-- The setup check in proofs. -/
def check (q : P2.Params) (c : SetupProtocol.Commitment) (cl : Claim) : Core.Outcome SetupProtocol.Public Unit :=
  match q.validate with
  | .error m => .reject s!"params: {m}"
  | .ok _ =>
    if c.salt.size != 24 then .reject "the salt is 24 bytes, drawn by the verifier after W is committed"
    else if cl.samplingDesign != samplingDesign q then .reject s!"the sampling design is {samplingDesign q}, not {cl.samplingDesign}"
    else .reject s!"the sampled audit's check is not built: {GOAL}"

def protocol : Core.Protocol where
  id := ID
  Params := P2.Params
  Commitment := SetupProtocol.Commitment
  Evidence := Claim
  Public := SetupProtocol.Public
  Profile := Unit
  params := SetupProtocol.params
  commitment := SetupProtocol.commitment
  evidence := claim
  digest c := c.w ++ c.salt
  check := check
  publicJson := SetupProtocol.Public.toJson
  profileJson _ := Json.mkObj [("mode", .str "in-proofs")]

end PousCheck.SetupInProofs
