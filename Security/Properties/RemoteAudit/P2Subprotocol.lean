import RemoteAudit.P2Protocol

/-!
# PoUB-P2's subprotocol accepts only a session whose every window verified (`RemoteAudit.P2Protocol`)

Onsite's bandwidth subprotocol at PoUB-P2 (`Verity.component .onsite .poub`) is `RemoteAudit.P2Protocol.protocol`
(`verity/remote-bandwidth/p2/v0`), the remote audit over PoUS's P2 store: its check replays the verifier's log of a
session (`RemoteAudit.P2Protocol.replay`: the prover's request read against the held registration,
`P2Session.requestOf`, the session's first state, `P2Session.init`, and each logged request through `step`) and
accepts once the session has every window verified, publishing `certify` of its record: the reads its windows credit
under the pooled bound and their error.

* **AcceptIsVerified**, the refinement: what the check accepts is a request that is the held registration's node's, at
  the session parameters it reads, and a log that replays from the session's first state to a state with no failure,
  which ended accepted with a record `sm`; and the public outputs are `certify`'s of `sm` on the held store.

The honest instance its witnesses read is the auditor's session (`p2_party.SESSION`: two windows of 1 s, one pass
each, 1,024 openings a window, an allowance of 1 s) on a store of PoUS's: one segment of 2^16 blocks, 2^14 units, a
plan over all of them (`honestHeld`), with no request answered yet (`honestEvidence`).
-/

namespace Properties.RemoteAudit.P2Subprotocol

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

end Properties.RemoteAudit.P2Subprotocol
