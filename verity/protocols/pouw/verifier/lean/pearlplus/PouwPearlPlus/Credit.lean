module

public import PouwPearlPlus.Rule
public import PouwPearlPlus.Run

/-!
# The run's credit at Pearl's assumption (`verify_run.credit_at_assumption`)

At Pearl's Assumption 1 (`pearl_kw`), a tile that passes the rule cost its credited cells' MACs, (1 − ρ) of them at the
scheme's cap. Over every pass's records, a record whose tiles credit any cell is one unit: its MACs are its credited cells
at its kept width |K|, its ticket weight is those cells at the cell's ticket `wref(1, |K|, n)/n` (`clear_work.ticket_cell`
at the H100's record), and its MACs a ticket are |K|·n/`wref(1, |K|, n)`. Of t draws by ticket weight, X_O landed on tiles
over their inert-work bound, which are credited nothing (`over_cap` `uncredited`; under `reject` an accepted run has
X_O = 0). Except with δ, the tiles that are wrong or over the bound hold at most ε of the tickets' weight, where ε is the
least share such a set can hold and still draw at most X_O times with chance δ (`epsilon`; at X_O = 0, (1 − ε)^t ≤ δ). The
most MACs that weight can hold, filled from the units with the most MACs a ticket, are taken off:

    credited = ⌊(1 − ρ) · max(macs − wrong, 0)⌋

Every quantity is exact (`Rat`), as the Python's `Fraction`s are, ε among them: one more than the least k with
Σ_{i ≤ X_O} C(t, i) (k/10⁹)^i (1 − k/10⁹)^(t − i) ≤ δ, over 10⁹.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

/-- δ, the draw's miss probability the run is credited at (`verify_run.DELTA`). -/
def DELTA : Rat := mkRat 1 100

/-- Σ_{i ≤ x} C(t, i) · k^i · (g − k)^(t − i): g^t times the chance that t draws at k/g land at most x times. -/
def missMass (t x k g : Nat) : Nat := Id.run do
  let mut c := 1
  let mut s := 0
  for i in [0:min x t + 1] do
    s := s + c * k ^ i * (g - k) ^ (t - i)
    c := c * (t - i) / (i + 1)
  return s

/-- `verify_run.epsilon`: the least ε on the 10⁻⁹ grid at which t draws land on a set of ε of the tickets' weight at most
`over` times with chance at most δ (at `over` = 0, (1 − ε)^t ≤ δ), plus 10⁻⁹. -/
def epsilon (t over : Nat) (delta : Rat) : Rat := Id.run do
  let g : Nat := 10 ^ 9
  let ok (k : Nat) : Bool := (missMass t over k g : Int) * delta.den ≤ delta.num * (g : Int) ^ t
  let mut lo := 0
  let mut hi := g
  for _ in [0:31] do
    if lo < hi then
      let mid := (lo + hi) / 2
      if ok mid then hi := mid else lo := mid + 1
  return mkRat (lo + 1 : Nat) g

structure Credit where
  credited : Nat
  creditedTiles : Nat
  macs : Nat
  rho : Rat
  epsilon : Rat
  overCapDraws : Nat

/-- `credit_at_assumption(passes, t, δ, over)`: `over` is X_O, how many of the t draws landed on a tile over its inert-work
bound. -/
def creditAtAssumption (passes : List Pass) (t over : Nat) (delta : Rat) : Credit := Id.run do
  let mut macs := 0
  let mut tiles := 0
  let mut units : Array (Rat × Rat) := #[]
  for p in passes do
    for r in p.records do
      let cells := (p.scheme.tiles r.matmul.m r.matmul.n).map fun (rows, cols) =>
        let (cr, cc) := r.credited rows cols
        cr.length * cc.length
      let total := cells.foldl (· + ·) 0
      if total != 0 then
        let ticket := mkRat (wref 1 r.kKept r.matmul.n : Int) r.matmul.n
        macs := macs + total * r.kKept
        tiles := tiles + (cells.filter (· != 0)).length
        units := units.push ((r.kKept : Rat) / ticket, (total : Rat) * ticket)
  let eps := epsilon t over delta
  let rho := mkRat Rule.CAP_NUM Rule.CAP_DEN
  let mut budget := eps * units.foldl (fun s (_, w) => s + w) 0
  let mut wrong : Rat := 0
  for (ratio, w) in units.qsort (fun a b => b.1 < a.1) do
    let take := min budget w
    wrong := wrong + take * ratio
    budget := budget - take
  let credited := ((1 - rho) * max ((macs : Rat) - wrong) 0).floor.toNat
  return { credited, creditedTiles := tiles, macs, rho, epsilon := eps, overCapDraws := over }

end PouwCheck.PearlPlus
