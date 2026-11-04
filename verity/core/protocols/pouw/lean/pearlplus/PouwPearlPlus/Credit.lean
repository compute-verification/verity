import PouwPearlPlus.Rule
import PouwPearlPlus.Run

/-!
# The run's credit at Pearl's assumption (`verify_run.credit_at_assumption`)

At Pearl's Assumption 1 (`pearl_kw`), a tile that passes the rule cost its credited cells' MACs, (1 − ρ) of them at the
scheme's cap. Over every pass's records, a record whose tiles credit any cell is one unit: its MACs are its credited cells
at its kept width |K|, its ticket weight is those cells at the cell's ticket `wref(1, |K|, n)/n` (`clear_work.ticket_cell`
at the H100's record), and its MACs a ticket are |K|·n/`wref(1, |K|, n)`. t accepted draws by ticket weight leave at most
ε of the tickets' weight on wrong tiles except with δ (`epsilon`); the most MACs that weight can hold, filled from the units
with the most MACs a ticket, are taken off:

    credited = ⌊(1 − ρ) · max(macs − wrong, 0)⌋

Every quantity is exact (`Rat`), as the Python's `Fraction`s are. ε is the one the Python reaches through floating point,
`⌈−expm1(ln δ / t) · 10⁹⌉ + 1` over 10⁹: here it is exact, one more than the least k with (1 − k/10⁹)^t ≤ δ, which is the
Python's unless the true (1 − δ^(1/t)) · 10⁹ lies within the float's rounding of an integer (none does for t < 400 or
t ∈ {1,000, 2,000, 4,096, 10,000} at δ = 1/100, 1/1000, 1/20, 10⁻⁶ or 2⁻⁴⁰).
-/

namespace PouwCheck.PearlPlus

/-- δ, the draw's miss probability the run is credited at (`verify_run.DELTA`). -/
def DELTA : Rat := mkRat 1 100

/-- `verify_run.epsilon`: the least ε on the 10⁻⁹ grid with (1 − ε)^t ≤ δ, plus 10⁻⁹. -/
def epsilon (t : Nat) (delta : Rat) : Rat := Id.run do
  let g : Nat := 10 ^ 9
  let ok (k : Nat) : Bool := ((g - k : Nat) : Int) ^ t * delta.den ≤ delta.num * (g : Int) ^ t
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

/-- `credit_at_assumption(passes, t, δ)`. -/
def creditAtAssumption (passes : List Pass) (t : Nat) (delta : Rat) : Credit := Id.run do
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
  let eps := epsilon t delta
  let rho := mkRat Rule.CAP_NUM Rule.CAP_DEN
  let mut budget := eps * units.foldl (fun s (_, w) => s + w) 0
  let mut wrong : Rat := 0
  for (ratio, w) in units.qsort (fun a b => b.1 < a.1) do
    let take := min budget w
    wrong := wrong + take * ratio
    budget := budget - take
  let credited := ((1 - rho) * max ((macs : Rat) - wrong) 0).floor.toNat
  return { credited, creditedTiles := tiles, macs, rho, epsilon := eps }

end PouwCheck.PearlPlus
