module

public import Definitions.NetworkCertifier.Warden

/-!
# The response cap: each response at most its request's `max_tokens` tokens

Definitions only. A run may register a response cap `cap` with its schedule (`warden.WardenParams.max_tokens_cap`,
which the schedule's parameters digest, and so every record head, binds), in the id egress form, where each data cell
is one id event, a token, and the end of the stream holds only the trailer (`io_tie`). The network certifier then reads
each request's `max_tokens` from its first ingress cell, at the place `active.id_request` fixes for it
(`active.id_request_max_tokens`): `some m`, or `none` where the field isn't a u32 there. A request of `none`, or of
`m > cap`, is refused at ingress: none of its bytes reaches the server, and its lane fails at once (reason `cap`, the
client's). Every other's response is held to its `m` tokens: a lane fails at its data cell `m` (reason `limit`, the
server's), as at a late cell.

`Arrivals.capped` is that hold on what reaches the network certifier: each data cell past the first `m` is late at its
bucket, so the network certifier's code (`certifierLane`) fails the lane there. A refused request's lane is held to no
data cell (`admitted`); the Python certifier fails it at once even when the server sends nothing, which sends no data
cell either. `cappedCode` is the network certifier's code under the cap: `certifierCode` on the arrivals so held, so
every statement about `certifierCode` at arbitrary arrivals (the tap, its record, the leak bounds) holds of it.
-/

@[expose] public section

namespace NetTiming

/-- Whether a cell the client receives is data. -/
def CellView.isData {Cell : Type} : CellView Cell → Bool
  | .data _ => true
  | _ => false

/-- The data cells among a lane's cells: in the id egress form, its response's tokens. -/
def dataCells {Cell : Type} (v : List (ℕ × CellView Cell)) : ℕ :=
  v.countP fun x => x.2.isData

/-- The arrivals `a` held to `m` data cells on a lane whose cell `j` is sent in bucket `t j`: each data cell past the
first `m` is late at its bucket. -/
def Arrivals.capped {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) (m : ℕ) : Arrivals Cell where
  stream := a.stream
  ready j := if m ≤ j ∧ j < a.stream.length then t j + 1 else a.ready j

/-- The data cells a request is held to under the cap `cap`: its `max_tokens` `m` when it read as `some m` with
`m ≤ cap`, else none (refused at ingress). -/
def admitted (cap : ℕ) : Option ℕ → ℕ
  | some m => if m ≤ cap then m else 0
  | none => 0

/-- **The network certifier's code under a response cap** `cap`, each connection `c`'s request read as `mt c`:
`certifierCode` on each connection's arrivals held to its admitted `max_tokens`. -/
def cappedCode {κ Cell : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ) (arr : κ → Arrivals Cell) :
    κ → List (ℕ × CellView Cell) :=
  certifierCode s fun c => (arr c).capped (s.bucket c) (admitted cap (mt c))

end NetTiming
