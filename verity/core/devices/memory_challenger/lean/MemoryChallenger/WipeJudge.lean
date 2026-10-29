import MemoryChallenger.Judge
import MemoryChallenger.StreamedWipe

/-!
# The streamed wipe's verdict on a record, as its caller holds it

`streamed_wipe.verdict(signed, commitment, trusted=trusted, schemes=schemes)` on its inputs as JSON, the whole `Verdict`
back: what `wipe_verdict.py` beside this package sends `streamed-wipe-verdict`. The input is one object:

* `signed`: `SignedWipe.to_json()`;
* `commitment`: `WipeCommitment.to_json()`, the commitment published when the session opened;
* `trusted`: the memory challengers' public keys, hex;
* `schemes`: the ids of the accepted signature schemes, each one this port verifies (`KNOWN_SCHEMES`).

The input is exactly the canonical JSON of what it decodes to (`exactly`), as `to_json` writes it: a field missing or
extra, a byte string not lowercase hex, a header constant not the protocol's, parameters `WipeParams` refuses, or a scheme
id this port doesn't verify is refused. Integers are any sign where the Python's dataclasses hold any (a batch record's
fields, the fill's times, the egress), naturals in the parameters. The output is `{"accepted", "egress_mib", "reasons"}`,
the reasons the Python's word for word.
-/

namespace MemoryChallenger.StreamedWipe

open Lean (Json)
open Flock

private def hexJ (b : ByteArray) : Json := .str (hexOf b)
private def natJ (n : Nat) : Json := Lean.toJson n
private def intJ (n : Int) : Json := Lean.toJson n

/-- `WipeCommitment.to_json()`: its header. -/
def Commitment.toJson (c : Commitment) : Json :=
  let p := c.params
  Json.mkObj [("protocol", .str PROTOCOL), ("records_tree", .str RECORDS_TREE), ("server", hexJ c.server),
    ("session", hexJ c.session), ("labels", natJ p.labels), ("label_bytes", natJ LABEL_BYTES), ("k", natJ p.k),
    ("batch", natJ p.batch), ("limit_ns", natJ p.limitNs), ("tiers", .arr (p.tiers.map fun (t, b) => .arr #[.str t, natJ b])),
    ("clock", .str CLOCK)]

def Batch.toJson (r : Batch) : Json :=
  Json.mkObj [("b", intJ r.b), ("first", intJ r.first), ("count", intJ r.count), ("send_ns", intJ r.sendNs),
    ("receipt_ns", intJ r.receiptNs), ("wrong", intJ r.wrong), ("digest", hexJ r.digest)]

def Signed.toJson (s : Signed) : Json :=
  Json.mkObj [("commitment", s.commitment.toJson), ("ended", .str s.ended), ("fill_start_ns", intJ s.fillStartNs),
    ("fill_end_ns", intJ s.fillEndNs), ("batches", .arr (s.batches.map Batch.toJson)), ("egress_bits", intJ s.egressBits),
    ("scheme", .str s.scheme), ("public_key", hexJ s.publicKey), ("signature", hexJ s.signature)]

private def intField (j : Json) (k ctx : String) : V Int := do J.asInt (← J.get j k ctx) s!"{ctx}.{k}"

def Commitment.ofJson (j : Json) : V Commitment := do
  let ctx := "commitment"
  let tiers ← (← J.arr j "tiers" ctx).mapM fun t => do
    let a ← J.asArr t s!"{ctx}.tiers"
    if a.size != 2 then throw s!"{ctx}.tiers: a tier is [name, bytes]"
    pure (← J.asStr a[0]! s!"{ctx}.tiers", ← J.asNat a[1]! s!"{ctx}.tiers")
  let p : Params := { labels := ← J.nat j "labels" ctx, k := ← J.nat j "k" ctx, batch := ← J.nat j "batch" ctx,
                      limitNs := ← J.nat j "limit_ns" ctx, tiers }
  match p.check with
  | .ok () => pure { server := ← hexField j "server" ctx, session := ← hexField j "session" ctx, params := p }
  | .error e => throw s!"{ctx}: {e}"

def Batch.ofJson (j : Json) : V Batch := do
  let ctx := "batch"
  pure { b := ← intField j "b" ctx, first := ← intField j "first" ctx, count := ← intField j "count" ctx,
         sendNs := ← intField j "send_ns" ctx, receiptNs := ← intField j "receipt_ns" ctx, wrong := ← intField j "wrong" ctx,
         digest := ← hexField j "digest" ctx }

def Signed.ofJson (j : Json) : V Signed := do
  let ctx := "signed"
  pure { commitment := ← Commitment.ofJson (← J.get j "commitment" ctx), ended := ← J.str j "ended" ctx,
         fillStartNs := ← intField j "fill_start_ns" ctx, fillEndNs := ← intField j "fill_end_ns" ctx,
         batches := ← objs j "batches" ctx Batch.ofJson, egressBits := ← intField j "egress_bits" ctx,
         scheme := ← J.str j "scheme" ctx, publicKey := ← hexField j "public_key" ctx,
         signature := ← hexField j "signature" ctx }

/-- `verdict`'s inputs, decoded. -/
structure JudgeInputs where
  signed : Signed
  commitment : Commitment
  trusted : Array ByteArray
  schemes : Array String

def JudgeInputs.toJson (r : JudgeInputs) : Json :=
  Json.mkObj [("commitment", r.commitment.toJson), ("schemes", .arr (r.schemes.map Json.str)), ("signed", r.signed.toJson),
    ("trusted", .arr (r.trusted.map hexJ))]

def JudgeInputs.ofJson (j : Json) : V JudgeInputs := do
  let ctx := "record"
  pure { signed := ← Signed.ofJson (← J.get j "signed" ctx), commitment := ← Commitment.ofJson (← J.get j "commitment" ctx),
         trusted := ← hexArr j "trusted" ctx,
         schemes := ← (← J.arr j "schemes" ctx).mapM (J.asStr · s!"{ctx}.schemes") }

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj [("accepted", .bool v.accepted), ("egress_mib", intJ v.egressMib), ("reasons", .arr (v.reasons.map Json.str))]

/-- The verdict on a record. -/
def judge (r : JudgeInputs) : V Verdict := do
  pure (verdict (← Judge.schemesOf r.schemes) r.trusted.toList r.signed r.commitment)

def verdictFn (j : Json) : V Json := do
  pure (Verdict.toJson (← judge (← exactly JudgeInputs.ofJson JudgeInputs.toJson j)))

end MemoryChallenger.StreamedWipe
