import CoreProtocol.Cli
import PouwCheck.Credit

/-!
# A PoUW period as a `Core.Protocol`: `verity/pouw/period/v1`

Onsite's work component (`Core.Onsite.components`). A period is the windows the auditor's ledger lists between the
period's open and its close, and it earns all or nothing: `Credit.periodCredit` credits it only when every one of its
windows is credited (`Credit.window`), and a refused window refuses the period. `pouw-period` runs it through
`Core.Protocol.main`, and `tests/test_period_difftest.py` holds it to its Python, `verity/core/protocols/pouw/period_check.py`.

**Params**, canonical JSON with exactly these keys, the verifier's own:

    {"key": challenger key, "main": [commit, …], "name": period}

the challenger key the verifier pins (`SampledProofs.Challenger.ChallengerKey`), its own copy of main's first-parent
history (newest first) and the name of the period it credits, a string.

**Commitment**, canonical JSON: the auditor's ledger, the list of the lines the challenger signed (`ledgerEntry?`). Its
period's open commits the period's circuit, its draw K, its table rows T, its weights' root and model and its fixed
length before any coin of the period, and each window's session its roots before their coins. Its digest is
SHA-512(`commitmentTag` ‖ 0 ‖ the commitment's bytes).

**Evidence**, canonical JSON `{"pins": pins, "windows": [window, …]}`: the challenger host's published pins log, and each
window's evidence in the close's order, `Credit.PERIOD_KEYS` (its identity, session, salt, manifest, served rows, Commit
record, round log, draw, descriptor, registered entries, each drawn unit's proof outcome and its live evidence).

**The check** is `Credit.periodCredit` on them: it accepts with the period's credit, or rejects with the first refusal,
`code: why`. It never leaves a period undecided.

**Public**, on acceptance: `Credit.PeriodCredit.toJson`, each window's credited calls, its credited served work, the K
the period's open committed, the δ (`Credit.DELTA`) and each window's ε at it.

**Profile**: the period's name and its number of windows.

What it doesn't check: each drawn unit's statement and each round's live statement, whose proof outcomes the evidence
carries as the verifier of record's verdicts (`PouwCheck.Verified` and `PouwCheck.LiveVerified` judge them, with
C-Flock's Lean verifier run outside).
-/

namespace PouwCheck.Period

open Lean (Json toJson)
open Flock (V)

def protocolId : String := "verity/pouw/period/v1"
def commitmentTag : String := "verity/pouw/period/ledger/v1"
def paramsKeys : List String := ["key", "main", "name"]
def evidenceKeys : List String := ["pins", "windows"]

structure Params where
  key : Json
  main : Json
  name : String

structure Commitment where
  ledger : Json
  bytes : ByteArray

structure Evidence where
  pins : Json
  windows : Json

structure Profile where
  period : String
  windows : Nat

def paramsOf (j : Json) : V Params := do
  unless keys j == some paramsKeys do throw s!"keys are not {", ".intercalate paramsKeys}"
  let some name := str? (get j "name") | throw "name is not a string"
  pure { key := get j "key", main := get j "main", name }

def commitmentOf (b : ByteArray) : V Commitment := do
  let ledger ← Core.canonicalJson b
  let .arr _ := ledger | throw "the auditor's ledger is not a list of its lines"
  pure { ledger, bytes := b }

def Commitment.digest (c : Commitment) : ByteArray :=
  Flock.Sha512.hash (commitmentTag.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys j == some evidenceKeys do throw s!"keys are not {", ".intercalate evidenceKeys}"
  pure { pins := get j "pins", windows := get j "windows" }

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Credit.PeriodCredit Profile :=
  match Credit.periodCredit p.key c.ledger e.pins p.main (.str p.name) e.windows with
  | .ok pc => .accept pc { period := p.name, windows := pc.windows.length }
  | .error (code, why) => .reject s!"{code}: {why}"

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("period", .str pr.period), ("windows", toJson pr.windows)]

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Credit.PeriodCredit
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest := Commitment.digest
  check := check
  publicJson := Credit.PeriodCredit.toJson
  profileJson := profileJson

-- every input gets a verdict: parameters, a commitment or evidence of another shape is refused, naming the input
#guard (protocol.answer "{}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason == some "params: keys are not key, main, name"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":1}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason ==
  some "params: name is not a string"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "{}".toUTF8 "{}".toUTF8).reason ==
  some "commitment: the auditor's ledger is not a list of its lines"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason ==
  some "evidence: keys are not pins, windows"
-- well-formed inputs reach `periodCredit`, whose refusal is the period's
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "[]".toUTF8
  "{\"pins\":[],\"windows\":7}".toUTF8).reason == some s!"period: {Credit.WINDOWS_FORM}"

end PouwCheck.Period
