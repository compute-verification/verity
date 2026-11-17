module

public import Definitions.Certifier.Cells

/-! The response cap: each response at most its request's `max_tokens` tokens. -/

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

/-- **The network attester's code under a response cap** `cap`, each connection `c`'s request read as `mt c`:
`certifierCode` on each connection's arrivals held to its admitted `max_tokens`. -/
def cappedCode {κ Cell : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ) (arr : κ → Arrivals Cell) :
    κ → List (ℕ × CellView Cell) :=
  certifierCode s fun c => (arr c).capped (s.bucket c) (admitted cap (mt c))

end NetTiming
