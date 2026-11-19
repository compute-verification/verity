module

public import CoreProtocol.Protocol
public import PousVerifier.SetupProtocol

/-!
# PoUS's setup as a sampled proof about the store

`verity/pous/p2/v4/setup-in-proofs` is the setup check that reads no block of `C*`. The setup is one circuit `C`, the
setup unit `PousP2SetupCheck_v1` over every block of the store, one proof block a block
(`verity/protocols/resource_accounting/memory/pous/prover/window.py`, `Setup`); the store's blocks and `W`'s rows are
its registered values, committed before the run, and the proof system samples `SAMPLE` of the blocks on its live coins
after that (`samplingDesign`). The check takes the parameters and the commitment as `SetupProtocol` does, and in place of
`C*` the claim (`Claim`, the Python `setup_proof.Claim`'s last three fields):

~~~text
evidence  {"law":"rate:8828/B"|null,"outcome":hex,"store":hex}   canonical JSON
~~~

It decides the parameters' validity, the salt's size and that the claim's sampling design is `samplingDesign q`. Its
acceptance is to be the door's acceptance of the sampled blocks of `C`, and two places stop it, each refused by name:
the salt, the one value the verifier supplies to `C` (`pous.setup.salt/v1`), is bound to `C` by no supplied input yet
(`notBound`), and the door's statement of a block is not derived (`Flock.OneCircuit.notDerived`, at `Spot.OneStageZkStatement.restrict`). On
acceptance it publishes `SetupProtocol.Public` with the store's root as `root`, so the timed audit's input keeps its form.
-/

@[expose] public section

namespace PousVerifier.SetupInProofs

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

/-- What the check reads in place of `C*`: the store's registered root, the sampling design and SPOT's outcome record. -/
structure Claim where
  store : ByteArray
  samplingDesign : Option String
  outcome : ByteArray

def claim (b : ByteArray) : V Claim := do
  let j ← Core.canonicalJson b
  let ctx := "the setup's claim"
  let samplingDesign ← match j.getObjVal? "law" with
    | .ok .null => pure none
    | .ok (.str s) => pure (some s)
    | _ => throw s!"{ctx}: law is a string or null"
  return ⟨← Flock.unhex (← Flock.J.str j "store" ctx), samplingDesign, ← Flock.unhex (← Flock.J.str j "outcome" ctx)⟩

/-- Where `C`'s one supplied value, the salt, waits on the supplied input: the Python `setup_proof.NotBound`'s message. Each
unit builds its base from the salt and its block's constants, and its slot is a constant of the unit in `C`. -/
def notBound : String :=
  "pous setup: no supplied input yet: salt = pous.setup.salt/v1, the memory attester's coin drawn after W is " ++
    "committed, the one value the verifier supplies; unit u builds its base expand_prefix(params, salt) || " ++
    "u64be(u / S) || u64be(u mod S) inside, its block's u64be(u / S) || u64be(u mod S) and its slot u mod n " ++
    "constants of the unit in C, and its ok a public output held to 1"

/-- The setup check in proofs: the door's acceptance of the sampled blocks of the setup's `C`, refused where `C`'s
supplied value is bound to nothing (`notBound`). -/
def check (q : P2.Params) (c : SetupProtocol.Commitment) (cl : Claim) : Core.Outcome SetupProtocol.Public Unit :=
  match q.validate with
  | .error m => .reject s!"params: {m}"
  | .ok _ =>
    if c.salt.size != 24 then .reject "the salt is 24 bytes, drawn by the verifier after W is committed"
    else if cl.samplingDesign != samplingDesign q then .reject s!"the sampling design is {samplingDesign q}, not {cl.samplingDesign}"
    else .reject notBound

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

end PousVerifier.SetupInProofs
