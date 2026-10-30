import RemoteAudit.P2Protocol
import Properties.RemoteAudit.P2Line

/-!
# PoUB-P2's subprotocol accepts only a session whose every window verified (`RemoteAudit.P2Protocol`)

Onsite's bandwidth subprotocol at PoUB-P2 (`Verity.component .onsite .poub`) is `RemoteAudit.P2Protocol.protocol`
(`verity/remote-bandwidth/p2/v0`), the remote audit over PoUS's P2 store: its check replays the verifier's log of a
session (`RemoteAudit.P2Protocol.replay`: the prover's request read against the held registration,
`P2Session.requestOf`, the session's first state, `P2Session.init`, and each logged request through `step`) and
accepts once the session has every window verified, publishing `certify` of its record: the reads its windows credit
under the pooled bound and their error. PoUB-P2's line of record (`Properties.RemoteAudit.P2Line.LineHeld`) is stated
on the session the check replays (`P2Line.Verified`); the two properties join them:

* **AcceptIsVerified**, the refinement: what the check accepts is a request that is the held registration's node's, at
  the session parameters it reads, and a log that replays from the session's first state to a state with no failure,
  which ended accepted with a record `sm`; and the public outputs are `certify`'s of `sm` on the held store.
* **LineHeldAtSubprotocol**: `P2Line.LineHeld` with the subprotocol's acceptance in the event in place of `Verified`.
  On the same node, trace, records and premises, and the bytes `a ω`, `b ω` and `e ω` the subprotocol is given at each
  outcome (the parameters, the commitment and the evidence), the probability that the subprotocol accepts bytes that
  play the record of `ω` on the node (`Plays`) while the network certifiers sign nothing but the verifier's messages
  entering the node and the record's credit claims more bits read than the trace's passes read is at most `LineHeld`'s
  bound, term for term.

`Plays` is the subprotocol's acceptance (`Properties.Composition.DecodedAccepted` unfolded at
`RemoteAudit.P2Protocol.protocol`, whose decoders are `decodeParams`, `decodeCommitment` and `decodeEvidence`) with what
picks out the record the line reads: the parameters decode to the record's clock floor and request, the request reads
as the node's session parameters, the commitment holds the node's store and the log is the record's. It is in the
event, not a hypothesis: a log that is not the record's is an outcome the bound counts nothing for.

`LineHeldAtSubprotocol acc ver sum` is stated at a verifier, and its guarantee at the subprotocol's acceptance `Plays`,
the verdict `P2Line.Verified` the trace reads and the record `P2Line.Replay.summary` the credit reads, so that its
nontrivial witness can replace all three by a verifier that accepts everything (`traceBy`, `paceBy` and `ReadBy` are
`P2Line`'s `ofRecord`, `recordPace` and `ReadAt` at a verdict and a record).

The honest instance its witnesses read is the auditor's session (`p2_party.SESSION`: two windows of 1 s, one pass
each, 1,024 openings a window, an allowance of 1 s) on a store of PoUS's: one segment of 2^16 blocks, 2^14 units, a
plan over all of them (`honestHeld`), with no request answered yet (`honestEvidence`).
-/

namespace Properties.RemoteAudit.P2Subprotocol

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.P2Line

/-- **The P2 subprotocol's check accepts only a session that verified every window.** Whenever
`RemoteAudit.P2Protocol.check` accepts parameters `p`, a commitment `c` (the held registration and plan) and a log `e`
with public outputs `pub`: the request in `p` is the held registration's node's, at session parameters `params`
(`P2Session.requestOf`); the log replays from the session's first state to a state `s` (`replay`) with no failure,
which ended accepted with a record `sm`; and `pub` is what `certify` publishes of `sm` on the held store. -/
def AcceptIsVerified : Prop :=
  ∀ (p : RemoteAudit.P2Protocol.Params) (c : RemoteAudit.P2Protocol.Commitment)
    (e : RemoteAudit.P2Protocol.Evidence) (pub : RemoteAudit.P2Protocol.Public) (prof : Lean.Json),
    RemoteAudit.P2Protocol.check p c e = .accept pub prof →
    ∃ (params : RemoteAudit.Session.Params) (s : RemoteAudit.Session.State) (sm : RemoteAudit.Session.Summary),
      RemoteAudit.P2Session.requestOf c.held p.request = .ok params ∧ RemoteAudit.P2Protocol.replay p c e = .ok s ∧
      s.failed = none ∧ s.outcome = some "accepted" ∧ s.summary = some sm ∧
      pub = (RemoteAudit.P2Protocol.certify c.held sm).1

/-- The honest node's id, two bytes (`6e30`). -/
def honestNode : ByteArray := ⟨#[0x6e, 0x30]⟩

/-- A store of PoUS's at its registered point: one segment of 2^16 blocks (2^14 units), its commitment, salt and vk
root of the right sizes, and sketch-sha512's 64 rows. -/
def honestReg : RemoteAudit.P2.Registration where
  node := honestNode
  blocksPerSegment := 65536
  segments := 1
  commitment := ⟨Array.replicate 64 0⟩
  salt := ⟨Array.replicate 24 0⟩
  root := ⟨Array.replicate 64 0⟩
  rows := 64

/-- What the verifier holds of the honest store: its registration and the plan over all its units. -/
def honestHeld : RemoteAudit.P2Session.Held where
  reg := honestReg
  plan := RemoteAudit.Plan.whole honestReg.units

/-- The auditor's session (`p2_party.SESSION`) requested for the honest node. -/
def honestRequest : Lean.Json :=
  Lean.Json.mkObj [("deadline_ns", (1000000000 : Nat)), ("node", .str "6e30"), ("passes", (1 : Nat)),
    ("samples", (1024 : Nat)), ("window_ns", (1000000000 : Nat)), ("windows", (2 : Nat))]

/-- The session parameters `honestRequest` reads as: two windows of 1 s, one pass each, 1,024 openings a window, an
allowance of 1 s, and the lifetime cap of 30 windows. -/
def honestSessionParams : RemoteAudit.Session.Params where
  deadlineNs := 1000000000
  windows := 2
  samples := 1024
  minPasses := 1
  windowNs := 1000000000
  lifetime := 30

/-- The honest session's parameters: the request, from clock 0. -/
def honestParams : RemoteAudit.P2Protocol.Params := ⟨0, honestRequest⟩

/-- The honest store's commitment. The check reads only what it holds (`held`), not the bytes it was read from. -/
def honestCommitment : RemoteAudit.P2Protocol.Commitment := ⟨honestHeld, ByteArray.empty⟩

/-- The log of the honest session before its first request is answered. -/
def honestEvidence : RemoteAudit.P2Protocol.Evidence := ⟨#[]⟩

/-- **The subprotocol accepts bytes that play the record on the node.** The parameters `a` (canonical JSON), the
commitment `b` and the evidence `e` decode under the subprotocol's decoders, its check accepts them
(`Properties.Composition.DecodedAccepted` at `RemoteAudit.P2Protocol.protocol`), the parameters are the record's clock
floor and request, which reads as the node's session parameters against the store it holds, the commitment holds the
node's store, and the log is the record's. -/
def Plays (node : Audited) (r : Replay) (a b e : ByteArray) : Prop :=
  ∃ (c : RemoteAudit.P2Protocol.Commitment) (ev : RemoteAudit.P2Protocol.Evidence),
    (Core.canonicalJson a >>= RemoteAudit.P2Protocol.decodeParams) = .ok ⟨r.notBeforeNs, r.request⟩ ∧
    RemoteAudit.P2Protocol.decodeCommitment b = .ok c ∧ c.held = node.held ∧
    RemoteAudit.P2Protocol.decodeEvidence e = .ok ev ∧ ev.log = r.log ∧
    RemoteAudit.P2Session.requestOf node.held r.request = .ok node.params ∧
    ∃ pub prof, RemoteAudit.P2Protocol.check ⟨r.notBeforeNs, r.request⟩ c ev = .accept pub prof

/-- The windows of the record a verifier records of the session as `sum node r`, none where it records none:
`Replay.windows` at `sum = Replay.summary`. -/
def windowsBy (sum : Audited → Replay → Option RemoteAudit.Session.Summary) (node : Audited) (r : Replay) :
    Array RemoteAudit.Session.WindowSummary :=
  match sum node r with
  | some sm => sm.windows
  | none => #[]

/-- `ofRecord` at a verifier's verdict `ver`: the trace over the store's units, accepting where `ver` does. -/
def traceBy {Ω : Type} (T : Trace Ω) (node : Audited) (ver : Audited → Replay → Prop) (rec : Ω → Replay) : Trace Ω :=
  { Line.overBlocks T node.blocks with accepts := fun ω => ver node (rec ω) }

/-- `recordPace` of the windows a verifier's record `sum` gives. -/
def paceBy {Ω : Type} (sum : Audited → Replay → Option RemoteAudit.Session.Summary) (node : Audited)
    (rec : Ω → Replay) : Pace Ω where
  credited := node.windows - 1
  first ω g := P2Line.passesBefore (windowsBy sum node (rec ω)) g
  released ω g := (windowsBy sum node (rec ω))[min g (node.windows - 1)]!.releasedNs
  ended ω g := P2Line.ended (windowsBy sum node (rec ω)) g

/-- `ReadAt` of a verifier's record `sum`: if it records the session as `sm`, `8·q_bytes` of `sm`'s credit is at most
the bits of the blocks its `P` passes read on the node, `UNIT_BYTES` bytes a block. -/
def ReadBy (sum : Audited → Replay → Option RemoteAudit.Session.Summary) {Ω : Type} (T : Trace Ω) (ω : Ω)
    (node : Audited) (r : Replay) : Prop :=
  ∀ sm, sum node r = some sm →
    (8 * (RemoteAudit.P2Session.credit sm).qBytes : ℝ) ≤
      (8 * RemoteAudit.P2.UNIT_BYTES : ℝ) * T.reads ω (P2Line.passes sm)

/-- **PoUB-P2's line of record, in bits read, at the subprotocol.** `P2Line.LineHeld`'s node, trace, records and
premises verbatim, one coin for each of the session's windows (`Audited.draws`), and the bytes the subprotocol is given
at each outcome (`a`, `b`, `e`: the parameters, the commitment and the evidence): the probability that the subprotocol
accepts bytes that play the record of `ω` on the node while the network certifiers sign nothing but the verifier's
messages entering the node over the credited passes and the record's credit claims more bits read than the passes read
is at most `δs + δnet + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`, `LineHeld`'s bound at the credit's δ
(`P2Session.DELTA`, 2^-40). It is stated at a verifier: `acc` for the subprotocol's acceptance in the event, `ver` for
the trace's verdict and `sum` for the record the credit and its paced session read. The guarantee is it at `Plays`,
`P2Line.Verified` and `P2Line.Replay.summary`. -/
def LineHeldAtSubprotocol (acc : Audited → Replay → ByteArray → ByteArray → ByteArray → Prop)
    (ver : Audited → Replay → Prop) (sum : Audited → Replay → Option RemoteAudit.Session.Summary) : Prop :=
  ∀ (node : Audited) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks Line.receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Replay)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs δnet : ℝ)
    (a b e : Coins node.draws κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (traceBy T node ver r)) (I : Ingress (Coins node.draws κ X)),
    Paced (traceBy T node ver r) (paceBy sum node r) → Covers (traceBy T node ver r) →
    SketchBoundAt (traceBy T node ver r) O δs → Covered (traceBy T node ver r) O I →
    IngressCounted (traceBy T node ver r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => acc node (r ω) (a ω) (b ω) (e ω) ∧
        ¬ (Accounting.Remote.Held.Quiet (traceBy T node ver r) I.counted ω →
          ReadBy sum (traceBy T node ver r) ω node (r ω))) ≤
      δs + δnet + ((RemoteAudit.P2Session.DELTA : ℝ) +
        ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (RemoteAudit.P2Session.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (RemoteAudit.P2Session.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end Properties.RemoteAudit.P2Subprotocol
