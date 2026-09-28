import MemoryChallenger.Protocol

/-!
# The verdict on a record, as its caller holds it

`memory_challenger.verdict(signed, commitment, openings, expected, trusted=trusted, schemes=schemes)` on its inputs as JSON,
the whole `Verdict` back. This is what `benchmarks/pous/memory_challenger/live.py` judges each session with, and what a
replay of its records reads (`verdict.py` beside this package is the Python call). The input is one object:

* `signed`: `SignedSession.to_json()`;
* `commitment`: `SessionCommitment.to_json()`, the commitment published before the session;
* `openings`: `[CoinOpening.to_json(), …]`;
* `expected`: `{"<i>": hex, …}`, `block_digest` of block `i` of `C` for the blocks the caller has it for (the blocks the
  rounds name are enough: `verdict` reads no other);
* `trusted`: the memory challengers' public keys, hex;
* `schemes`: the ids of the accepted signature schemes, each one this port verifies (`KNOWN_SCHEMES`): `[LMS]` is
  `memory_challenger.SCHEMES`, the test signers' ids `pous_test_signer.SCHEMES`.

Each part is exactly the canonical JSON of what it decodes to (`exactly`), as `to_json` writes it: a field missing or
extra, a byte string not lowercase hex, a negative integer or a header constant not the protocol's is refused, and so
is a scheme id this port doesn't verify. The output is `{"accepted", "late", "reasons", "wrong"}`, the reasons the
Python's word for word.
-/

namespace MemoryChallenger.Judge

open Lean (Json)
open Flock

/-- `verdict`'s inputs, decoded. -/
structure Record where
  signed : SignedSession
  commitment : SessionCommitment
  openings : Array CoinOpening
  expected : Digests
  trusted : Array ByteArray
  schemes : Array String

def Record.toJson (r : Record) : Json :=
  Json.mkObj [("commitment", r.commitment.toJson), ("expected", r.expected.toJson),
    ("openings", .arr (r.openings.map CoinOpening.toJson)), ("schemes", .arr (r.schemes.map Json.str)),
    ("signed", r.signed.toJson), ("trusted", .arr (r.trusted.map fun b => .str (hexOf b)))]

def Record.ofJson (j : Json) : V Record := do
  let ctx := "record"
  pure { signed := ← SignedSession.ofJson (← J.get j "signed" ctx),
         commitment := ← SessionCommitment.ofJson (← J.get j "commitment" ctx),
         openings := ← objs j "openings" ctx CoinOpening.ofJson,
         expected := ← Digests.ofJson j "expected" ctx,
         trusted := ← hexArr j "trusted" ctx,
         schemes := ← (← J.arr j "schemes" ctx).mapM (J.asStr · s!"{ctx}.schemes") }

/-- The schemes a record names, each looked up among those this port verifies. -/
def schemesOf (ids : Array String) : V Schemes :=
  ids.toList.mapM fun id => match KNOWN_SCHEMES.lookup id with
    | some p => pure (id, p)
    | none => throw s!"record.schemes: {pyRepr id} is not a scheme this port verifies"

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj [("accepted", .bool v.accepted), ("late", Lean.toJson v.late), ("reasons", .arr (v.reasons.map Json.str)),
    ("wrong", Lean.toJson v.wrong)]

/-- The verdict on a record. -/
def judge (r : Record) : V Verdict := do
  let schemes ← schemesOf r.schemes
  pure (verdict schemes r.trusted.toList r.expected.find r.signed r.commitment r.openings)

def verdictFn (j : Json) : V Json := do
  pure (Verdict.toJson (← judge (← exactly Record.ofJson Record.toJson j)))

end MemoryChallenger.Judge
