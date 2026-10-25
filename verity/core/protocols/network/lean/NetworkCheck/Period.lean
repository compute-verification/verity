import NetworkCheck.Check
import CoreProtocol.Profile

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

Onsite's communication component (`Core.Onsite.components`). A period is one node's link over `m` certifier windows
(N2's link-windows, `service.LinkWindows`, Selection `All`) beside the period's compute audits (sampled proofs'
one-stage, one answer per registered window of the workload). It accepts when every link-window was proved and both its
ends and its clock were certified, and every compute audit accepted; it then publishes `K`, the bits a covert sender in
the server can move through the wardens in the period, as `NetTiming.Guarantees.PeriodLeakAccepted` counts them.

**Params**, JSON with exactly these keys (and each nested object exactly its own):

    {"audits": [{"free": f, "steps": k}, …], "cells": [cap, …], "first": w₀, "jitter": Dj, "key": hex(I ‖ T[1]),
     "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool}, "program": hex, "session": hex(session),
     "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": m}

each compute audit's layout terms (`free`, the free inputs it doesn't register, `#(freeInputs \ reg)`; `steps`, its
horizon `k`), each warden connection's cell cap over the period (`cap_c`), the session's window the period starts at
(`w₀`: its link-windows are the session's `w₀ … w₀ + m − 1`, so one certifier session spans many periods), the
observations' jitter bound `Dj ≥ 1`, the
network certifier's LMS public key and parameters and the certifier session (32 bytes) every certificate binds, the link
(at most 65,535 UTF-8 bytes), the SHA-512 of N2's registered Program for the link (the partition object's `program`,
`LinkWindows.partition`), Σ_sync, and the period's `m ≥ 1` link-windows.

**Commitment**, canonical JSON `{"audits": [registration, …], "windows": [registration, …]}`: one-stage's registration
record of each compute audit and of each link-window, in order, each of format `verity/registration/v1` or `v2`. Its
digest is `tagged_sha512("verity-network-certifier/period/v0", the commitment's bytes)`, and each registration's is
one-stage's, SHA-512(format ‖ 0 ‖ canonical record) (`registration.digest`).

**Evidence**, canonical JSON:

    {"audits": [answer, …],
     "windows": [{"answer": answer, "certificates": [certificate, certificate],
                  "clock": {"certificate": certificate, "sync": [δ, num, den]}}, …]}
    certificate = {"digest": hex, "direction": "egress" | "ingress", "key_id": hex, "link": s, "signature": hex,
                   "window": w}

each audit's and each link-window's one-stage answer (`Core.Answer.ofJson`), and per link-window the two certificates
(`certificate.Certificate`, egress first) and the clock certificate (`certificate.ClockCertificate`) the certifier, the
auditor's device, issued for it.

**The check**, in order; any rejection rejects, then any undecided answer leaves the period undecided:

* the commitment registers the parameters' audits and `m` link-windows, and the evidence answers each;
* link-window `w` (the `i`-th, `w = w₀ + i`): its registration names the link's Program and the provenance
  `{"kind": "served", "window": w}`;
  both certificates are the key's, name the link, their end and `w`, and verify under LMS over `certificate_message` in
  the session; the clock certificate is the key's egress certificate of the link at `w`, signs `clock_digest(sync)`
  and verifies, and its sync is in Σ_sync (`advice`); and the answer is one-stage's for that registration's digest,
  accepted with the profile of one unit drawn `subset:1`: the link-window was proved (N2's verifier of record,
  `LinkWindows.verify`, is the answer's verifier of record);
* audit `i`: the answer is one-stage's for its registration's digest, accepted with one level drawn `subset:k`.

**Public**, on acceptance: `K` and its terms, each a ceiling of `log2` of `PeriodLeakAccepted`'s factor:

    {"K": K, "audits": [{"delta", "drawn": k, "faulty_bits", "faulty_sets", "free", "jitter", "units": N,
                         "wrong_units": b}, …],
     "cells": Σ_c ⌈log2 (cap_c + 1)⌉, "eps": Σ_i δ_i, "schedules": ⌈log2 #Σ_sync^m⌉, "windows": m}

with `eps` the union of the audits' δ, an exact ratio (`Core.Ratio.toString`): except with probability at most `eps`,
every audit's wrong units are within its bound, and so K bounds the period's leak;

with `b` the audit's bound on wrong units at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (the possible faulty sets of a uniform `subset:k` draw: every set of at most `b` units),
`faulty_bits` = `16 · b` (a proof unit's outputs are at most 16 bits), `jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of
the terms. The schedules term charges each link-window's sync, the one the clock certificate signs.

**Profile**: the link, each window's certified sync `[δ, num, den]`, and `m`.

What it doesn't check: the proof's public row against the certificates' digests and the output word, which N2's
verifier of record checks inside the answer (`LinkWindows.verify`, with C-Flock's Lean verifier on the `--zk` proof);
the link-window registration's domain; and that the audits' answers are the sampled-proofs component's own, which the
composition's theorem is to bind.
-/

namespace NetworkCheck.Period

open Lean (Json toJson)
open Flock (V)
open NetworkCertifier (ClockSync)
open NetworkCheck.Check (Certificate inSyncSet ratOf syncJson syncOf)

def protocolId : String := "verity/certifier/v0"
def commitmentTag : String := "verity-network-certifier/period/v0"
def clockTag : String := "verity-network-certifier/clock/v1"
def oneStage : String := "verity/sampled-proofs/one-stage/v1"
def formats : List String := ["verity/registration/v1", "verity/registration/v2"]
/-- A proof unit's outputs are at most 16 bits. -/
def outBits : Nat := 16
def unitLevel : String := "unit"

/-- A compute audit's layout terms: its free inputs it doesn't register, and its horizon. -/
structure Audit where
  free : Nat
  steps : Nat

structure Params where
  audits : Array Audit
  cells : Array Nat
  first : Nat
  jitter : Nat
  link : String
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  program : String
  syncSet : NetworkCertifier.SyncSet
  windows : Nat

/-- A registration record and its one-stage digest. -/
structure Registration where
  record : Json
  digest : ByteArray

structure Commitment where
  audits : Array Registration
  windows : Array Registration
  bytes : ByteArray

structure Clock where
  sync : ClockSync
  certificate : Certificate

structure Window where
  answer : Core.Answer
  certificates : Array Certificate
  clock : Clock

structure Evidence where
  audits : Array Core.Answer
  windows : Array Window

/-- What an accepted audit establishes: its units, its draw, its δ, and its bound on wrong units. -/
structure Facts where
  units : Nat
  drawn : Nat
  delta : Core.Ratio
  wrong : Nat

/-! ## Decoding -/

private def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

private def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

def isDigest (s : String) : Bool := s.length = 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

def syncSetOf (j : Json) : V NetworkCertifier.SyncSet := do
  exactly j ["deltas", "rhos"] "sync_set"
  let deltas := (← (← Flock.J.arr j "deltas" "sync_set").mapM (Flock.J.asNat · "sync_set.deltas")).toList
  let rhos := (← (← Flock.J.arr j "rhos" "sync_set").mapM (ratOf · "sync_set.rhos")).toList
  if h : deltas ≠ [] ∧ rhos ≠ [] ∧ (∀ ρ ∈ rhos, 0 < ρ) ∧ deltas.Nodup ∧ rhos.Nodup then
    pure ⟨deltas, rhos, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩
  else throw "sync_set: non-empty, each delta and each rho listed once"

def paramsKeys : List String :=
  ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "session", "sync_set", "windows"]

def paramsOf (j : Json) : V Params := do
  exactly j paramsKeys "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM fun a => do
    exactly a ["free", "steps"] "an audit"
    pure ({ free := ← Flock.J.nat a "free" "an audit", steps := ← Flock.J.nat a "steps" "an audit" } : Audit)
  let cells ← (← Flock.J.arr j "cells" "the object").mapM (Flock.J.asNat · "cells")
  let first ← Flock.J.nat j "first" "the object"
  let jitter ← Flock.J.nat j "jitter" "the object"
  unless 1 ≤ jitter do throw "jitter: the observations' jitter bound is at least 1"
  let link ← Flock.J.str j "link" "the object"
  unless link.toUTF8.size < 2 ^ 16 do throw "link: a name is at most 65,535 UTF-8 bytes"
  let l ← Flock.J.get j "lms" "the object"
  exactly l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let session ← Flock.unhex (← Flock.J.str j "session" "the object")
  unless session.size = sessionBytes do throw s!"a certifier session id is {sessionBytes} bytes, not {session.size}"
  let program ← Flock.J.str j "program" "the object"
  unless isDigest program do throw "program: not a SHA-512 digest (128 lowercase hex digits)"
  let syncSet ← syncSetOf (← Flock.J.get j "sync_set" "the object")
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one link-window"
  pure { audits, cells, first, jitter, link, lms, key, session, program, syncSet, windows }

/-- `registration.digest`: SHA-512(format ‖ 0 ‖ the canonical record), the record's own format the tag. -/
def registrationOf (j : Json) (ctx : String) : V Registration := do
  let format ← Flock.J.str j "format" ctx
  unless formats.contains format do throw s!"{ctx}: format is not {" or ".intercalate formats}"
  let body ← Flock.canon j
  pure ⟨j, Flock.Sha512.hash (format.toUTF8 ++ ByteArray.mk #[0] ++ body.toUTF8)⟩

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (registrationOf · "an audit's registration")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (registrationOf · "a link-window's registration")
  pure { audits, windows, bytes := b }

def certificateOf (j : Json) (ctx : String) : V Certificate := do
  exactly j ["digest", "direction", "key_id", "link", "signature", "window"] ctx
  let direction ← Direction.ofValue (← Flock.J.str j "direction" ctx)
  pure { keyId := ← Flock.unhex (← Flock.J.str j "key_id" ctx), link := ← Flock.J.str j "link" ctx, direction,
         window := ← Flock.J.nat j "window" ctx, digest := ← Flock.unhex (← Flock.J.str j "digest" ctx),
         signature := ← Flock.unhex (← Flock.J.str j "signature" ctx) }

def windowOf (j : Json) (ctx : String) : V Window := do
  exactly j ["answer", "certificates", "clock"] ctx
  let answer ← Core.Answer.ofJson (← Flock.J.get j "answer" ctx) s!"{ctx}.answer"
  let certificates ← (← Flock.J.arr j "certificates" ctx).mapM (certificateOf · s!"{ctx}.certificates")
  let c ← Flock.J.get j "clock" ctx
  exactly c ["certificate", "sync"] s!"{ctx}.clock"
  pure { answer, certificates,
         clock := { sync := ← syncOf (← Flock.J.get c "sync" s!"{ctx}.clock"),
                    certificate := ← certificateOf (← Flock.J.get c "certificate" s!"{ctx}.clock") s!"{ctx}.clock" } }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (Core.Answer.ofJson · "an audit's answer")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (windowOf · "a link-window")
  pure { audits, windows }

/-! ## The check -/

/-- `Certificate.verify` under the parameters' key in their session: the certificate is the key's (its I) and its LMS
signature verifies over `certificate_message` of its fields; a message that can't be formed or a signature that doesn't
decode is false. -/
def signs (p : Params) (c : Certificate) : Bool :=
  c.keyId == p.key.extract 0 16 &&
    match certificateMessage c.keyId p.session c.link c.direction c.window c.digest,
        Lms.Signature.decode p.lms c.signature with
    | .ok m, .ok sig => (Lms.verify p.lms p.key m sig).toOption == some true
    | _, _ => false

/-- `clock_digest`: `tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)` of the sync, ρ = p/q
in lowest terms; `none` past a u64. -/
def clockDigest (s : ClockSync) : Option ByteArray :=
  (do
    let d ← uBE 8 s.delta "δ"
    let n ← uBE 8 s.rho.num "p"
    let q ← uBE 8 s.rho.den "q"
    Identity.taggedSha512 clockTag (d ++ n ++ q)).toOption

/-- A certificate of the link's end `d` at window `w`, the key's and verifying. -/
def certifies (p : Params) (d : Direction) (w : Nat) (c : Certificate) : Bool :=
  c.direction == d && c.link == p.link && c.window == w && signs p c

/-- A finding against the period: a rejection, or what leaves it undecided. -/
abbrev Found := Except (Core.Verdict × String)

def refuse {α} (r : String) : Found α := .error (.reject, r)

/-- One-stage's answer for the registration: its reject is a rejection, its undecided leaves the period undecided, and its
acceptance gives the profile. -/
def answered (what : String) (reg : Registration) (a : Core.Answer) : Found (Core.Integrity × Json) := do
  if a.protocol != oneStage then refuse s!"{what}: protocol {a.protocol}, not one-stage's {oneStage}"
  else if a.digest != some reg.digest then refuse s!"{what}: the answer is for another registration"
  else match a.verdict, a.published, a.profile with
    | .accept, some pub, some prof =>
      match Core.Integrity.ofJson prof with
      | .ok i => pure (i, pub)
      | .error e => refuse s!"{what}: {e}"
    | .undecided, _, _ => .error (.undecided, s!"{what}: service: {a.reason.getD ""}")
    | _, _, _ => refuse s!"{what}: service: {a.reason.getD ""}"

def served (w : Nat) : Json := Json.mkObj [("kind", "served"), ("window", toJson w)]

/-- The canonical text of a value read from canonical bytes, which `Flock.canon` always writes. -/
def canonText (j : Json) : Option String := (Flock.canon j).toOption

/-- Link-window `w`'s findings, in the order the module docstring gives. -/
def windowCheck (p : Params) (w : Nat) (reg : Registration) (x : Window) : Found Unit := do
  let what := s!"window {w}"
  unless reg.record.getObjValD "program" == .str p.program &&
      canonText (reg.record.getObjValD "window") == canonText (served w) do
    refuse s!"{what}: registration: not the link's Program at window {w}"
  unless x.certificates.size == 2 && (x.certificates.zip #[Direction.egress, .ingress]).all
      (fun (c, d) => certifies p d w c) do
    refuse s!"{what}: certificates"
  let c := x.clock.certificate
  unless certifies p .egress w c && clockDigest x.clock.sync == some c.digest do refuse s!"{what}: clock"
  unless inSyncSet p.syncSet x.clock.sync do refuse s!"{what}: advice"
  let (prof, _) ← answered what reg x.answer
  unless prof.levels.map (fun l => (l.name, l.size, l.draw)) == #[(unitLevel, 1, some (Core.Draw.subset 1))] &&
      prof.drawn == #[(unitLevel, 1)] do
    refuse s!"{what}: profile: not the link-window's one unit proved"

/-- Audit `i`'s findings, and on acceptance its facts. -/
def auditCheck (i : Nat) (reg : Registration) (a : Core.Answer) : Found Facts := do
  let what := s!"audit {i}"
  let (prof, pub) ← answered what reg a
  let #[l] := prof.levels | refuse s!"{what}: profile: not one level"
  let some (.subset k) := l.draw | refuse s!"{what}: profile: not a subset draw"
  match Flock.J.nat pub "bound" "public" with
  | .ok b => pure { units := l.size, drawn := k, delta := prof.delta, wrong := b }
  | .error e => refuse s!"{what}: {e}"

def findingOf {α : Type} : Found α → Option (Core.Verdict × String)
  | .error x => some x
  | .ok _ => none

/-- The first rejection among the findings, else the first undecided one. -/
def firstFinding (fs : List (Option (Core.Verdict × String))) : Option (Core.Verdict × String) :=
  match fs.findSome? fun f => f.filter (·.1 == .reject) with
  | some r => some r
  | none => fs.findSome? id

/-! ## The bound -/

/-- `⌈log2 x⌉`, 0 for x ≤ 1. -/
def clog2 (x : Nat) : Nat := if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1

/-- `Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units. -/
def atMost (n b : Nat) : Nat := Id.run do
  let mut c := 1
  let mut s := 1
  for j in [0:min b n] do
    c := c * (n - j) / (j + 1)
    s := s + c
  return s

structure AuditTerms where
  facts : Facts
  free : Nat
  faultySets : Nat
  faultyBits : Nat
  jitter : Nat

def AuditTerms.bits (t : AuditTerms) : Nat := t.free + t.faultySets + t.faultyBits + t.jitter

structure Public where
  audits : Array AuditTerms
  cells : Nat
  schedules : Nat
  windows : Nat

def Public.K (x : Public) : Nat := x.schedules + x.cells + (x.audits.map (·.bits)).foldl (· + ·) 0

/-- `a + b`, exactly. -/
def ratioAdd (a b : Core.Ratio) : Core.Ratio :=
  (Core.Ratio.mk' (a.num * b.den + b.num * a.den) (a.den * b.den)).getD a

/-- The union of the audits' δ: the probability, at most, that some audit's bound fails, so that K fails to bound the
period's leak. -/
def Public.eps (x : Public) : Core.Ratio := x.audits.foldl (fun r t => ratioAdd r t.facts.delta) ⟨0, 1⟩

/-- The bound's terms from the audits' layout terms and facts, the cells' caps, the jitter bound, `#Σ_sync` and the
link-windows. -/
def terms (audits : Array Audit) (facts : Array Facts) (cells : Array Nat) (jitter syncs windows : Nat) : Public :=
  { audits := (audits.zip facts).map fun (a, f) =>
      { facts := f, free := a.free, faultySets := clog2 (atMost f.units f.wrong), faultyBits := outBits * f.wrong,
        jitter := clog2 (jitter ^ a.steps) }
    cells := (cells.map fun cap => clog2 (cap + 1)).foldl (· + ·) 0
    schedules := clog2 (syncs ^ windows)
    windows }

structure Profile where
  link : String
  syncs : Array ClockSync
  windows : Nat

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Public Profile :=
  if c.windows.size ≠ p.windows then
    .reject s!"the commitment registers {c.windows.size} link-windows, not the period's {p.windows}"
  else if c.audits.size ≠ p.audits.size then
    .reject s!"the commitment registers {c.audits.size} audits, not the parameters' {p.audits.size}"
  else if e.windows.size ≠ c.windows.size then
    .reject s!"evidence: {e.windows.size} link-windows for {c.windows.size} registered"
  else if e.audits.size ≠ c.audits.size then
    .reject s!"evidence: {e.audits.size} audits' answers for {c.audits.size} registered"
  else
    let ws := (c.windows.zip e.windows).zipIdx.map fun ((reg, x), i) => windowCheck p (p.first + i) reg x
    let as := (c.audits.zip e.audits).zipIdx.map fun ((reg, a), i) => auditCheck i reg a
    match firstFinding (ws.toList.map findingOf ++ as.toList.map findingOf) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      .accept (terms p.audits (as.filterMap (·.toOption)) p.cells p.jitter
          (p.syncSet.deltas.length * p.syncSet.rhos.length) p.windows)
        { link := p.link, syncs := e.windows.map (·.clock.sync), windows := p.windows }

def publicJson (x : Public) : Json :=
  Json.mkObj [
    ("K", toJson x.K),
    ("audits", .arr <| x.audits.map fun t => Json.mkObj [
      ("delta", .str t.facts.delta.toString), ("drawn", toJson t.facts.drawn), ("faulty_bits", toJson t.faultyBits),
      ("faulty_sets", toJson t.faultySets), ("free", toJson t.free), ("jitter", toJson t.jitter),
      ("units", toJson t.facts.units), ("wrong_units", toJson t.facts.wrong)]),
    ("cells", toJson x.cells), ("eps", .str x.eps.toString), ("schedules", toJson x.schedules),
    ("windows", toJson x.windows)]

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("link", .str pr.link), ("syncs", .arr (pr.syncs.map syncJson)), ("windows", toJson pr.windows)]

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Public
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson := publicJson
  profileJson := profileJson

#guard clog2 0 == 0 && clog2 1 == 0 && clog2 2 == 1 && clog2 3 == 2 && clog2 4 == 2 && clog2 5 == 3
#guard atMost 10 0 == 1 && atMost 10 1 == 11 && atMost 10 2 == 56 && atMost 4 9 == 16 && atMost 0 3 == 1
-- an audit of 10 units drawn subset:3 at δ = 1/10 (at most 4 wrong), 7 cells on one connection and none on another,
-- 2 syncs over 3 link-windows, and jitter below 2 over 3 steps
#guard (terms #[⟨0, 3⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K ==
  3 + 3 + (0 + 9 + 64 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨0, 1⟩, ⟨0, 1⟩] #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 },
  { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }] #[] 1 1 1).eps.toString == "10001/1000000"
#guard (terms #[] #[] #[] 1 1 1).eps.toString == "0"

end NetworkCheck.Period
