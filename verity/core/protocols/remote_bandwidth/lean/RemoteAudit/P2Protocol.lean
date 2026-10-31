import CoreProtocol.Cli
import CoreProtocol.Ratio
import RemoteAudit.Verifier

/-!
# A session over PoUS's P2 store as a core protocol

PoUB-P2's session (`P2Session`, as the verifier's process decides it live, `p2_party.py`) behind the core interface
(`Core.Protocol`), so that onsite can take it as its bandwidth component: its registered store is PoUS's `C*`, so the
store it audits is the one PoUS's setup published. Its bytes are canonical JSON (`p2_case.py` writes them):

* params `{"not_before_ns": n, "request": R}`: the verifier's clock floor and the prover's request R, refused unless
  R is for the registration's node and restates the auditor's session, `P2Session.SESSION`, as it is
  (`P2Session.requestOf`), naming each value that isn't: no digest binds R, so the session's parameters are that
  constant's, never R's, and a toy party's case is refused;
* the commitment: the auditor's registration file, its fields and its plan (`P2Session.heldOf`); its digest
  `Commitment.digest`, SHA-512 of `DIGEST_DOMAIN` and those bytes, so it binds the plan as well as the registration;
* the evidence `{"log": [...]}`: every request the verifier's process answered of the session after it opened, in the
  order it answered them, as `Verifier.handle` takes its `p2_*` ops, each line's bytes as the prover sent them, hex:
  `{"op": "p2_event", "event", "coin"}` (a start, a boundary or a close with the coin the challenger released for it,
  `Verifier.eventOn 2 1`), `{"op": "p2_root", "t", "i", "now", "line"}` (`P2Session.receive`), `{"op": "p2_openings",
  "t", "line"}` (`P2Session.verify`) and `{"op": "p2_fail"}` (`P2Session.fail`). The log is the verifier's own: its
  clock's readings and its coins never come from the prover;
* the check: the log replayed from the session's first state (`P2Session.init`). It rejects when an entry doesn't
  decode or the process would have answered it with an error, and with the failure that rejected the session; it is
  undecided while a window is unverified; it accepts once every window is.

On acceptance it publishes `{"eps", "q_bytes", "span_ns"}`: the credited reads (`P2Session.credit`, the answer's),
their span, and the error they hold at, ε = δ + S/2^64 for S the draws, `RemoteAudit.Protocol.certify`'s terms (not
the sketch's term at units, `SketchBoundP2`, which is drafted). The profile is the rest of the session's answer.
No property states yet what this check accepts; `p2_case.py`'s tests hold it to the live process's answer.
-/

namespace RemoteAudit.P2Protocol

open Lean (Json JsonNumber ToJson toJson)
open Flock

def ID : String := P2Session.PROTOCOL

def DIGEST_DOMAIN : ByteArray := "verity/remote-bandwidth/p2/commitment/v0".toUTF8.push 0

/-- The draws a thread checks (`P2Session.checkOpeningsChunked`), which changes no verdict. -/
def CHUNK : Nat := 64

private def natJ (n : Nat) : Json := .num (JsonNumber.fromNat n)

/-- `j` is an object with exactly the keys `want`, in sorted order. -/
private def keys (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj kvs => if kvs.keys == want then pure () else throw s!"{ctx}: the keys must be {want}, not {kvs.keys}"
  | _ => throw s!"{ctx}: not an object"

structure Params where
  notBeforeNs : Nat
  request : Json

def decodeParams (j : Json) : V Params := do
  keys j ["not_before_ns", "request"] "params"
  pure ⟨← J.nat j "not_before_ns" "params", ← J.get j "request" "params"⟩

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

private def hexOf? (j : Json) (ctx : String) : V ByteArray := do
  match unhex (← J.asStr j ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}: {e}"

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

structure Evidence where
  log : Array Op

def decodeEvidence (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  keys j ["log"] "evidence"
  pure ⟨← (← J.arr j "log" "evidence").mapM opOf⟩

/-- One entry of the log through session `s`, as the verifier's process answers it (`Verifier.handle`'s `p2_*` ops):
the session after it, or the error the process would have answered. -/
def step (held : P2Session.Held) (s : Session.State) : Op → V Session.State
  | .event e coin =>
    match e with
    | .start _ | .boundary _ | .close .. => do
      let (_, s) ← Verifier.eventOn 2 1 s e coin
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
    match step c.held s e.log[k]! with
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

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Public Json :=
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
  some "params: params: the keys must be [not_before_ns, request], not []"

end RemoteAudit.P2Protocol
