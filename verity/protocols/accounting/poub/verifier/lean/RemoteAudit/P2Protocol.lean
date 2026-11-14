module

public import CoreProtocol.Cli
public import CoreProtocol.Protocol
public import CoreProtocol.Ratio
public import Lms.Sign
public import MemoryChallenger.Verdict
public import RemoteAudit.P2Close
public import RemoteAudit.Verifier
meta import CoreProtocol.Cli
meta import CoreProtocol.Ratio
meta import Lms.Sign
meta import MemoryChallenger.Verdict
meta import RemoteAudit.P2Close
meta import RemoteAudit.Verifier

/-!
# A session over PoUS's P2 store as a core protocol

PoUB-P2's session (`P2Session`, as the verifier's process decides it live, `p2_party.py`) behind the core interface
(`Core.Protocol`), so that onsite can take it as its bandwidth component: its registered store is PoUS's `C*`, so the
store it audits is the one PoUS's setup published. Its bytes are canonical JSON (`p2_case.py` writes them):

* params `{"not_before_ns": n, "request": R, "session": hex, "trusted": [hex, …]}`: the verifier's clock floor, the
  prover's request R, the session id the auditor designated for the memory challenger that ran the party, and the
  memory challengers' keys the check trusts, which never come from the evidence. R is refused unless it is for the
  registration's node and restates the auditor's session, `P2Session.SESSION`, as it is (`P2Session.requestOf`),
  naming each value that isn't: no digest binds R, so the session's parameters are that constant's, never R's, and a
  toy party's case is refused;
* the commitment: the auditor's registration file, its fields and its plan (`P2Session.heldOf`); its digest
  `Commitment.digest`, SHA-512 of `DIGEST_DOMAIN` and those bytes, so it binds the plan as well as the registration;
* the evidence `{"close": {"chain", "lines", "public_key", "signature"}, "log": [...]}`. The log is every request the
  verifier's process answered of the session after it opened, in the order it answered them, as `Verifier.handle`
  takes its `p2_*` ops, each line's bytes as the prover sent them, hex: `{"op": "p2_event", "event", "coin"}` (a
  start, a boundary or a close with the coin the challenger released for it, `Verifier.eventOn 2 1`, whole: its
  `COIN_BYTES`, of which the process read the first `Codec.COIN_BYTES`), `{"op": "p2_root", "t", "i", "now", "line"}`
  (`P2Session.receive`), `{"op": "p2_openings", "t", "line"}` (`P2Session.verify`) and `{"op": "p2_fail"}`
  (`P2Session.fail`). The close is the memory challenger's signature over the whole log (`P2Close`): its key, the
  log's length, the log's chain (`P2Close.chainOf`, over each entry's canonical JSON) and its LMS signature on
  `P2Close.closeMessage`, the memory challenger role's message of the params' session and P2's raw record. The log
  is the verifier's own: its clock's readings and its coins never come from the prover;
* the check, in order: the close (`closeRefusal`) is under a trusted key, counts the log's lines, holds the log's
  chain and verifies under S1's LMS at its deployed parameters (`Lms.deployed`, `MemoryChallenger.LMS`) on the params'
  session; every coin is the challenger's `COIN_BYTES` (`coinRefusal`); then the log replayed from the session's first
  state (`P2Session.init`), each coin read at its first `Codec.COIN_BYTES`. Each refusal is named. The replay rejects
  when an entry is a request the process would have answered with an error, and with the failure that rejected the
  session; it is undecided while a window is unverified; it accepts once every window is.

On acceptance it publishes `{"eps", "q_bytes", "span_ns"}`: the credited reads (`P2Session.credit`, the answer's),
their span, and the error they hold at, ε = δ + S/2^64 for S the draws, `RemoteAudit.Protocol.certify`'s terms (not
the sketch's term at units, `SketchBoundP2`, which is drafted). The profile is the rest of the session's answer.
No property states yet what this check accepts; `p2_case.py`'s tests hold it to the live process's answer.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace RemoteAudit.P2Protocol

open Lean (Json JsonNumber ToJson toJson)
open Flock

def ID : String := P2Session.PROTOCOL

def DIGEST_DOMAIN : ByteArray := "verity/remote-bandwidth/p2/commitment/v0".toUTF8.push 0

/-- A coin as the log carries it: the challenger's whole coin, whose first `Codec.COIN_BYTES` the replay reads. -/
def COIN_BYTES : Nat := 64

/-- The draws a thread checks (`P2Session.checkOpeningsChunked`), which changes no verdict. -/
def CHUNK : Nat := 64

private def natJ (n : Nat) : Json := .num (JsonNumber.fromNat n)

/-- `j` is an object with exactly the keys `want`, in sorted order. -/
private def keys (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj kvs => if kvs.keys == want then pure () else throw s!"{ctx}: the keys must be {want}, not {kvs.keys}"
  | _ => throw s!"{ctx}: not an object"

private def hexOf? (j : Json) (ctx : String) : V ByteArray := do
  match unhex (← J.asStr j ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}: {e}"

private def hexAt (j : Json) (k ctx : String) : V ByteArray := do hexOf? (← J.get j k ctx) s!"{ctx}.{k}"

structure Params where
  notBeforeNs : Nat
  request : Json
  session : ByteArray
  trusted : Array ByteArray

def decodeParams (j : Json) : V Params := do
  keys j ["not_before_ns", "request", "session", "trusted"] "params"
  pure { notBeforeNs := ← J.nat j "not_before_ns" "params"
         request := ← J.get j "request" "params"
         session := ← hexAt j "session" "params"
         trusted := ← (← J.arr j "trusted" "params").mapM (hexOf? · "params.trusted") }

structure Commitment where
  held : P2Session.Held
  bytes : ByteArray

def decodeCommitment (b : ByteArray) : V Commitment := do
  let _ ← Core.canonicalJson b
  pure ⟨← P2Session.heldOf b, b⟩

def Commitment.digest (c : Commitment) : ByteArray := sha512 (DIGEST_DOMAIN ++ c.bytes)

/-- One request of the verifier's process, as the log records it. -/
inductive Op where
  | event (e : Session.Event) (coin : Option ByteArray)
  | root (t i : Int) (now : Nat) (line : ByteArray)
  | openings (t : Int) (line : ByteArray)
  | fail
  deriving Inhabited

def opOf (j : Json) : V Op := do
  let ctx := "log"
  match ← J.str j "op" ctx with
  | "p2_event" =>
    keys j ["coin", "event", "op"] ctx
    let e ← Difftest.eventOf (← J.get j "event" ctx)
    let coin ← match ← J.get j "coin" ctx with
      | .null => pure none
      | c => some <$> hexOf? c s!"{ctx}.coin"
    pure (.event e coin)
  | "p2_root" =>
    keys j ["i", "line", "now", "op", "t"] ctx
    pure (.root (← J.asInt (← J.get j "t" ctx) s!"{ctx}.t") (← J.asInt (← J.get j "i" ctx) s!"{ctx}.i")
      (← J.nat j "now" ctx) (← hexOf? (← J.get j "line" ctx) s!"{ctx}.line"))
  | "p2_openings" =>
    keys j ["line", "op", "t"] ctx
    pure (.openings (← J.asInt (← J.get j "t" ctx) s!"{ctx}.t") (← hexOf? (← J.get j "line" ctx) s!"{ctx}.line"))
  | "p2_fail" =>
    keys j ["op"] ctx
    pure .fail
  | op => throw s!"{ctx}: no op {op}"

/-- One entry of the log: its bytes, the entry's canonical JSON, which the close's chain reads, and the request they
are. -/
structure Line where
  bytes : ByteArray
  op : Op
  deriving Inhabited

def lineOf (j : Json) : V Line := do pure ⟨(← canon j).toUTF8, ← opOf j⟩

/-- The memory challenger's close over the log: its key, the log's length and chain, and its LMS signature on
`P2Close.closeMessage`. -/
structure SignedClose where
  publicKey : ByteArray
  lines : Nat
  chain : ByteArray
  signature : ByteArray

def closeOf (j : Json) : V SignedClose := do
  keys j ["chain", "lines", "public_key", "signature"] "close"
  pure { publicKey := ← hexAt j "public_key" "close"
         lines := ← J.nat j "lines" "close"
         chain := ← hexAt j "chain" "close"
         signature := ← hexAt j "signature" "close" }

structure Evidence where
  log : Array Line
  close : SignedClose

def decodeEvidence (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  keys j ["close", "log"] "evidence"
  pure ⟨← (← J.arr j "log" "evidence").mapM lineOf, ← closeOf (← J.get j "close" "evidence")⟩

def UNTRUSTED : String := "the close is signed by a key that is not a trusted memory challenger's"
def UNCHAINED : String := "the close's chain is not the log's"
def UNSIGNED : String := "the close's signature does not verify on the params' session"

/-- Why the close doesn't sign the log under `lp`, none when it does: its key isn't trusted, it counts other lines than
the log's, its chain isn't the log's (`P2Close.chainOf`), or its signature doesn't verify on `P2Close.closeMessage` of
the params' session
(`MemoryChallenger.lmsVerify`, the check PoUS's sessions and the wipe run). -/
def closeRefusal (lp : Lms.Params) (p : Params) (e : Evidence) : Option String :=
  let c := e.close
  if !p.trusted.contains c.publicKey then some UNTRUSTED
  else if c.lines != e.log.size then some s!"the close counts {c.lines} lines, not the log's {e.log.size}"
  else if c.chain != P2Close.chainOf (e.log.map (·.bytes)) then some UNCHAINED
  else match MemoryChallenger.lmsVerify lp c.publicKey p.session (P2Close.closeRecord e.log.size c.chain)
      c.signature with
    | .ok true => none
    | _ => some UNSIGNED

/-- The close signs the log under `lp` (`closeRefusal`). -/
def signedOk (lp : Lms.Params) (p : Params) (e : Evidence) : Bool := (closeRefusal lp p e).isNone

/-- Why a coin of the log isn't the challenger's whole coin, none when every coin is. -/
def coinRefusal (e : Evidence) : Option String := Id.run do
  for h : k in [0:e.log.size] do
    if let .event _ (some c) := e.log[k].op then
      if c.size ≠ COIN_BYTES then
        return some s!"log entry {k}: a coin is the challenger's {COIN_BYTES} bytes, not {c.size}"
  return none

/-- One entry of the log through session `s`, as the verifier's process answers it (`Verifier.handle`'s `p2_*` ops),
its coin read at its first `Codec.COIN_BYTES`: the session after it, or the error the process would have answered. -/
def step (held : P2Session.Held) (s : Session.State) : Op → V Session.State
  | .event e coin =>
    match e with
    | .start _ | .boundary _ | .close .. => do
      let (_, s) ← Verifier.eventOn 2 1 s e (coin.map (·.extract 0 Codec.COIN_BYTES))
      pure s
    | _ => throw "a P2 session's events are a start, a boundary or a close: a root or openings is a line"
  | .root t i now line => pure (P2Session.receive held s t i line now).2
  | .openings t line => pure (P2Session.verify CHUNK held s t line).2
  | .fail => pure (P2Session.fail s).2

/-- The session after the log, from its first state on `SESSION`'s parameters, which the request restates, with no coin
yet released. -/
def replay (p : Params) (c : Commitment) (e : Evidence) : V Session.State := do
  let params ← match P2Session.requestOf c.held P2Session.SESSION p.request with
    | .ok x => pure x
    | .error why => throw s!"the request is refused: {why}"
  let mut s := P2Session.init c.held params (Array.replicate (2 * params.windows) ByteArray.empty) p.notBeforeNs
  for k in [0:e.log.size] do
    match step c.held s e.log[k]!.op with
    | .ok s' => s := s'
    | .error why => throw s!"log entry {k}: {why}"
  pure s

/-- The public outputs: the error, the reads certified in bytes, and the span they are over in nanoseconds. -/
structure Public where
  eps : Core.Ratio
  qBytes : Nat
  spanNs : Nat

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("eps", .str p.eps.toString), ("q_bytes", natJ p.qBytes), ("span_ns", natJ p.spanNs)]

/-- What an accepted session certifies (module docstring), with its profile. -/
def certify (held : P2Session.Held) (sm : Session.Summary) : Public × Json :=
  let c := P2Session.credit sm
  let draws := c.draws.toList.sum
  (⟨(Core.Ratio.mk' (2 ^ 24 + draws) (2 ^ 64)).getD ⟨1, 1⟩, c.qBytes, c.spanNs⟩,
   Json.mkObj [("bound", c.bound), ("delta", .str (toString P2Session.DELTA)), ("draws", toJson c.draws),
     ("plan_units", natJ held.plan.length), ("receipts", natJ c.receipts),
     ("registration", .str (hexOf held.reg.digest)), ("unit_bytes", natJ P2.UNIT_BYTES),
     ("units", natJ held.reg.units), ("windows", natJ sm.windows.size), ("wrong", natJ c.wrong)])

/-- The check under the LMS parameters `lp` (module docstring); `check` is it under S1's. -/
def checkAt (lp : Lms.Params) (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Public Json :=
  match closeRefusal lp p e <|> coinRefusal e with
  | some why => .reject why
  | none =>
    match replay p c e with
    | .error why => .reject why
    | .ok s =>
      match s.failed, s.summary with
      | some f, _ =>
        let draw := match f.at? with
          | some (i, q) => s!", draw [{i}, {q}]"
          | none => ""
        .reject s!"the session was rejected: {f.failure.name} at window {f.window}{draw}"
      | none, none => .undecided "the session has a window not yet verified"
      | none, some sm => let (pub, prof) := certify c.held sm; .accept pub prof

def check : Params → Commitment → Evidence → Core.Outcome Public Json := checkAt Lms.deployed

def protocol : Core.Protocol where
  id := ID
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Public
  Profile := Json
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := Commitment.digest
  check := check
  publicJson := Public.toJson
  profileJson := id

#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason ==
  some "params: params: the keys must be [not_before_ns, request, session, trusted], not []"

/-! The close's refusals on a one-line log, signed at a toy LMS key (N = 8, height 1), so the guards sign quickly; the
check runs at `Lms.deployed`. -/

private def toy : Lms.Params := ⟨8, 1, 1⟩
private def toySession : ByteArray := ByteArray.mk (Array.replicate 32 1)
private def otherSession : ByteArray := ByteArray.mk (Array.replicate 32 2)
private def toyLog : Array Line := #[⟨"{\"op\":\"p2_fail\"}".toUTF8, .fail⟩]

/-- `closeRefusal` at `toy` on `log` and a close of `toyLog` whose signature is the toy key's on `signedOn` over
`record` (P2's close record when none), edited by `edit`, its key trusted when `trust`. -/
private def refusal (signedOn : ByteArray) (trust : Bool) (edit : SignedClose → SignedClose := id)
    (log : Array Line := toyLog) (record : Option ByteArray := none) : Option (Option String) := do
  let k ← (Lms.keygen toy (ByteArray.mk (Array.replicate 8 7))).toOption
  let chain := P2Close.chainOf (toyLog.map (·.bytes))
  let m ← (MemoryChallenger.memoryChallengerMessage k.I signedOn
    (record.getD (P2Close.closeRecord toyLog.size chain))).toOption
  let s ← (Lms.signAt k 0 m).toOption
  let close := edit ⟨k.publicKey, toyLog.size, chain, s.encode⟩
  pure (closeRefusal toy ⟨0, .null, toySession, if trust then #[close.publicKey] else #[]⟩ ⟨log, close⟩)

/-- A PoUS session record and a wipe record of the toy session, each a digest. -/
private def pousRecord : Option ByteArray :=
  (MemoryChallenger.message { (default : MemoryChallenger.SessionCommitment) with session := toySession }
    MemoryChallenger.COMPLETE #[]).toOption
private def wipeRecord : Option ByteArray :=
  (MemoryChallenger.StreamedWipe.message
    { (default : MemoryChallenger.StreamedWipe.Commitment) with session := toySession }
    MemoryChallenger.StreamedWipe.COMPLETE .empty 0 0 #[]).toOption

private def flipLast (b : ByteArray) : ByteArray := b.set! (b.size - 1) (b[b.size - 1]! ^^^ 1)

#guard refusal toySession true == some none
#guard refusal toySession false == some (some UNTRUSTED)
#guard refusal toySession true (fun c => { c with lines := 2 }) ==
  some (some "the close counts 2 lines, not the log's 1")
#guard refusal toySession true (log := toyLog.push toyLog[0]!) ==
  some (some "the close counts 1 lines, not the log's 2")
#guard refusal toySession true (fun c => { c with chain := flipLast c.chain }) == some (some UNCHAINED)
#guard refusal otherSession true == some (some UNSIGNED)
#guard refusal toySession true (fun c => { c with signature := flipLast c.signature }) == some (some UNSIGNED)
#guard pousRecord.map (·.size) == some 64 && wipeRecord.map (·.size) == some 64
#guard (pousRecord.bind fun r => refusal toySession true (record := some r)) == some (some UNSIGNED)
#guard (wipeRecord.bind fun r => refusal toySession true (record := some r)) == some (some UNSIGNED)
#guard hexOf (P2Close.closeRecord 3 (ByteArray.mk (Array.replicate 64 0xab))) ==
  "766572697479" ++ "2f72656d6f74652d62616e647769647468" ++ "2f70322f636c6f73652f763000" ++ "0000000000000003" ++
    String.join (List.replicate 64 "ab")

private def coinLog (n : Nat) : Evidence :=
  ⟨#[⟨ByteArray.empty, .event (.start 0) (some (.mk (Array.replicate n 3)))⟩, ⟨ByteArray.empty, .fail⟩],
   ⟨ByteArray.empty, 0, ByteArray.empty, ByteArray.empty⟩⟩

#guard coinRefusal (coinLog COIN_BYTES) == none
#guard coinRefusal (coinLog Codec.COIN_BYTES) == some "log entry 0: a coin is the challenger's 64 bytes, not 32"

end RemoteAudit.P2Protocol
