import Flock.RecCoins
import VStar.Stage

/-!
# V*'s per-part port inputs: what `verity_flock.rec_vstage.stage` writes to each part's `inst.bin`

Each V* statement's prover file (`circuit.write` with shared rows) holds, per port of its circuit in order (input ports,
then output rows), a table of rows (u16 words), one 192-byte salt per table row, and the table row each instance reads. This
module computes those tables from what the stager reads of the inner session (`Src`), part by part:

* `levelIn`: a Ligerito level's `RecOpen{LANES, H}` statement, `L<l>` (`rec_outer.stage`): `row` and `aux` (the opening's
  row and siblings, `rec_open.pack`) under the prover's fresh salts, `acc` and `acc_out` (the running sums' rows either side
  of the opening, `rec_outer.Chain`) under their registered salts, `coef` and `dirs` (the verifier's values) under
  `verifierSalt`, `out` (the top node the opening climbs to, `rec_outer.TopValue`) under the firewall's salt;
* `algebraIn`: a part of the algebra, `alg/p<j>` (`rec_vstage._stage_part`, `rec_residuals.write`): each `m<i>` the inner
  prover's rows of round `i` under its salts (rep `r` row `r`), `acc0` and `acc1` the running sums before each rep's first
  opening and after its last, `v` the verifier's values under `verifierSalt`, `out` zero rows under `zeroSalt`;
* `capIn`: a level's `RecCap`, `cap` or `cap/L<l>` (`rec_cap.stage`): `top` every node of the level's tops under the
  firewall's salts, `out` zero rows under `zeroSalt`.

`partIn` dispatches on the statement (`VStar.Stmt`). The difftest (`core_vstar`'s `vstar-difftest`, `part_in`) holds each to
the stager's `inst.bin`.
-/

namespace VStar.In

open Flock

/-- One table row of a port as `inst.bin` carries it: its u16 words and its 192-byte salt. -/
structure Row where
  words : Array Nat
  salt : ByteArray
  deriving Inhabited

/-- A port's input (`circuit.write` with shared rows): its table rows in order, and the row each instance reads. -/
structure PortIn where
  rows : Array Row
  refs : Array Nat
  deriving Inhabited

/-- A part's port inputs by name, in its circuit's port order: input ports, then output rows. -/
abbrev PartIn := Array (String × PortIn)

/-- One inner Ligerito opening (`rec_vstar.Query`, plain leaves): its row's elements, its siblings bottom up, its position. -/
structure Query where
  rep : Nat
  level : Nat
  j : Nat
  pos : Nat
  row : Array F128
  sibs : Array ByteArray
  deriving Inhabited

/-- What `rec_vstage.stage` reads of the inner session, decoded: the prover's side (the message rounds' rows and salts, the
openings, the tops with the firewall's salts, the running sums' salts, the fresh salts of the openings' private rows) and
the verifier's (each rep's κ and eq(α) by level, and its v). -/
structure Src where
  /-- round `i` (`rec_algebra.rounds`), rep `r`: the inner prover's row and the salt it committed it under (`rec-c<i>`) -/
  msgs : Array (Array Row)
  /-- both reps' openings (`rec_vstar.queries`) -/
  openings : Array Query
  /-- rep, level: each node of the rep's top (64 bytes) with the firewall's salt (`rec_vstar.tops`) -/
  tops : Array (Array (Array (ByteArray × ByteArray)))
  /-- the running sums' `N + 1` salts (`rec_vstage.acc_salts`) -/
  accSalts : Array ByteArray
  /-- rep, level, component `r`, lane `l`: the final check's κ at the rep's coins (`verifier_rows`' info) -/
  kappa : Array (Array (Array (Array F128)))
  /-- rep, level, query `j`: eq(α) at the level (`verifier_rows`' info `queries[level]`'s second) -/
  eqA : Array (Array (Array F128))
  /-- each rep's v (`verifier_rows`, the comb included) -/
  vs : Array (Array F128)
  /-- the prover's fresh salt of statement directory `dir`, port `port`, table row `i` (a level's `row` and `aux`) -/
  fresh : String → String → Nat → ByteArray

/-- `rec_open.elements`: whole 8-word groups as F elements, `lo` then `hi`. -/
def elements (ws : Array Nat) : Array F128 :=
  (Array.range (ws.size / 8)).map fun i =>
    let w := fun k => (ws[8 * i + k]!).toUInt64
    ⟨w 0 ||| (w 1 <<< 16) ||| (w 2 <<< 32) ||| (w 3 <<< 48), w 4 ||| (w 5 <<< 16) ||| (w 6 <<< 32) ||| (w 7 <<< 48)⟩

/-- `rec_open.words_of`: bytes two to a little-endian word. -/
def wordsOf (b : ByteArray) : Array Nat := (Array.range (b.size / 2)).map fun i => (b.get! (2 * i)).toNat + 256 * (b.get! (2 * i + 1)).toNat

/-- `rec_open.aux_words`: whole 64-word rows for `H` siblings of 32 words, at least one. -/
def auxWords (H : Nat) : Nat := max 1 ((32 * H + 63) / 64) * 64

/-- `rec_open.pack`'s `aux`: the siblings' words bottom up, then zero words up to `auxWords`. -/
def aux (sibs : Array ByteArray) : Array Nat := RecCoins.padTo (sibs.flatMap wordsOf) (auxWords sibs.size)

/-- `rec_open.acc_ref`: `acc` with `Σ_l coef[r · L + l] · row[l]` added to its element `r`, its other words as they were. -/
def accRef (acc : Array Nat) (coef row : Array F128) : Array Nat :=
  let a := elements (acc.extract 0 16)
  let L := row.size
  let t := (Array.range 2).map fun r => (Array.range L).foldl (fun s l => s + coef[r * L + l]! * row[l]!) a[r]!
  RecCoins.words t ++ acc.extract 16 acc.size

/-- Query `o`'s coefficients (`rec_outer.coefficients` at its rep's κ and eq(α)): κ[level][0]'s then κ[level][1]'s, each
times eq(α)[j]. -/
def coefOf (src : Src) (o : Query) : Array F128 :=
  let kap := src.kappa[o.rep]![o.level]!
  let e := src.eqA[o.rep]![o.level]![o.j]!
  kap[0]!.map (· * e) ++ kap[1]!.map (· * e)

/-- `(rep, level, j)` before `(rep', level', j')`, lexicographically. -/
def before (a b : Query) : Bool :=
  a.rep < b.rep || (a.rep == b.rep && (a.level < b.level || (a.level == b.level && a.j < b.j)))

def same (a b : Query) : Bool := a.rep == b.rep && a.level == b.level && a.j == b.j

/-- The chain's order (`rec_outer.chain_order`): every opening, rep-major, then by level and query. -/
def order (src : Src) : Array Query := src.openings.qsort before

/-- The running sums' rows (`rec_outer.Chain.build`): row 0 zero, row `p + 1` opening `p`'s `acc_out` on row `p`. -/
def chainRows (src : Src) : Array (Array Nat) :=
  (order src).foldl (fun rows o => rows.push (accRef rows.back! (coefOf src o) o.row)) #[Array.replicate 64 0]

/-- Query `o`'s position in the chain. -/
def at? (src : Src) (o : Query) : Option Nat := (order src).findIdx? (same o)

/-- The rows before a rep's first opening and after its last (`rec_outer.Chain.rep_ends`). -/
def repEnds (src : Src) (rep : Nat) : Except String (Nat × Nat) := do
  let ps := (List.range (order src).size).filter fun p => (order src)[p]!.rep == rep
  match ps.min?, ps.max? with
  | some a, some b => pure (a, b + 1)
  | _, _ => throw s!"rep {rep} has no openings"

def sameTop (a b : Array (ByteArray × ByteArray)) : Bool :=
  a.size == b.size && (a.zip b).all fun (x, y) => x.1.data == y.1.data && x.2.data == y.2.data

/-- A level's tops as one registered value (`rec_outer.TopValue.build`): the distinct tops of the reps in rep order, one run
of rows each (level 0's `root_B`, both reps' one tree, is one run), with their salts, and each rep's first row. -/
structure Tops where
  rows : Array (Array Nat)
  salts : Array ByteArray
  first : Array Nat

def topsAt (src : Src) (level : Nat) : Except String Tops := do
  let mut runs : Array (Array (ByteArray × ByteArray)) := #[]
  let mut first : Array Nat := #[]
  for rep in [0:src.tops.size] do
    let some t := src.tops[rep]![level]? | throw s!"rep {rep} has no top at level {level}"
    unless t.all (·.1.size == 64) do throw s!"rep {rep}'s top at level {level}: nodes of 64 bytes"
    let k := (runs.findIdx? (sameTop t)).getD runs.size
    if k == runs.size then runs := runs.push t
    first := first.push ((runs.extract 0 k).foldl (· + ·.size) 0)
  pure { rows := runs.flatMap fun t => t.flatMap fun (n, _) => (Firewall.topRows n).toArray,
         salts := runs.flatMap fun t => t.map (·.2), first }

/-- A table of `rows` under `salts`, instance `i` reading row `i`. -/
def ownRows (rows : Array (Array Nat)) (salts : Array ByteArray) : PortIn :=
  { rows := (rows.zip salts).map fun (w, s) => { words := w, salt := s }, refs := Array.range rows.size }

/-- Level `level`'s `RecOpen{LANES, H}` statement over its `n` openings of both reps, in rep and query order
(`rec_outer.stage`, `rec_vstage.stage`'s `L<level>`); `prog` its program's SHA-512 hex (`partition.program_sha512`), under
which the verifier registers `coef` and `dirs`. -/
def levelIn (src : Src) (prog : String) (level LANES H n : Nat) : Except String PartIn := do
  let dir := s!"L{level}"
  let qs := (src.openings.filter (·.level == level)).qsort before
  unless qs.size == n do throw s!"{dir}: {qs.size} openings, not {n}"
  unless qs.all fun q => q.row.size == LANES && q.sibs.size == H do
    throw s!"{dir}: an opening is not of {LANES} lanes and {H} siblings"
  let chain := chainRows src
  let some ps := qs.mapM (at? src) | throw s!"{dir}: an opening not in the chain"
  let tops ← topsAt src level
  let tp := qs.map fun q => tops.first[q.rep]! + q.pos / 2 ^ H
  unless tp.all (· < tops.rows.size) do throw s!"{dir}: an opening's top node is past the level's tops"
  let ids := Array.range n
  pure #[("row", ownRows (qs.map (RecCoins.words ·.row)) (ids.map (src.fresh dir "row"))),
    ("aux", ownRows (qs.map (aux ·.sibs)) (ids.map (src.fresh dir "aux"))),
    ("acc", ownRows (ps.map (chain[·]!)) (ps.map (src.accSalts[·]!))),
    ("coef", ownRows (qs.map fun q => RecCoins.words (coefOf src q)) (ids.map (RecCoins.verifierSalt prog RecCoins.COEF_VALUE))),
    ("dirs", ownRows (qs.map fun q => RecCoins.dirsRow q.pos H) (ids.map (RecCoins.verifierSalt prog RecCoins.DIRS_VALUE))),
    ("out", ownRows (tp.map (tops.rows[·]!)) (tp.map (tops.salts[·]!))),
    ("acc_out", ownRows (ps.map (chain[· + 1]!)) (ps.map (src.accSalts[· + 1]!)))]

/-- A part of the algebra over both reps (`rec_vstage._stage_part`, `rec_residuals.write`): its row ports `ports` (`m<i>`,
`acc0`, `acc1`, as `rec_residuals.parts` names them), its v elements `v` (indices into each rep's v), `prog` its program's
SHA-512 hex, under which the verifier registers `v`. -/
def algebraIn (src : Src) (prog : String) (ports : List (String × Nat)) (v : List Nat) : Except String PartIn := do
  let chain := chainRows src
  let (a0, a1) ← repEnds src 0
  let (b0, b1) ← repEnds src 1
  let reg := fun (ps : Array Nat) => ownRows (ps.map (chain[·]!)) (ps.map (src.accSalts[·]!))
  let mut out : PartIn := #[]
  for (nm, _) in ports do
    if nm == "acc0" then out := out.push (nm, reg #[a0, b0])
    else if nm == "acc1" then out := out.push (nm, reg #[a1, b1])
    else
      let some i := RecVStar.mIndex nm | throw s!"alg: a row port {nm}, not m<i>, acc0 or acc1"
      let some rs := src.msgs[i]? | throw s!"alg: no message round {i}"
      unless rs.size == 2 do throw s!"alg: round {i} has {rs.size} rows, not one a rep"
      out := out.push (nm, { rows := rs, refs := #[0, 1] })
  let vw := RecResiduals.portWordsOf v.length
  let vrows := (Array.range 2).map fun r => RecCoins.padTo (RecCoins.words (v.toArray.map (src.vs[r]![·]!))) vw
  out := out.push ("v", ownRows vrows ((Array.range 2).map (RecCoins.verifierSalt prog RecCoins.V_VALUE)))
  pure (out.push ("out", ownRows (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)
    ((Array.range 2).map (RecCoins.zeroSalt RecCoins.ZERO_VALUE))))

/-- Level `level`'s `RecCap` over its `N` top nodes (`rec_cap.stage`): `top` every node of the level's tops (both reps'), `out`
zero rows. -/
def capIn (src : Src) (level N : Nat) : Except String PartIn := do
  let tops ← topsAt src level
  unless tops.rows.size == N do throw s!"cap at level {level}: {tops.rows.size} nodes, not {N}"
  pure #[("top", ownRows tops.rows tops.salts),
    ("out", ownRows (RecCoins.zeroRows N VStar.CAP_OUT_WORDS) ((Array.range N).map (RecCoins.zeroSalt RecCoins.CAP_ZERO_VALUE)))]

/-- Statement `st`'s port inputs, `prog` its program's SHA-512 hex: a level's, a part of the algebra's or a cap's. -/
def partIn (src : Src) (prog : String) : Stmt → Except String PartIn
  | .recOpen level LANES H n => levelIn src prog level LANES H n
  | .inner _ _ p => algebraIn src prog p.S.ports p.v
  | .cap level N => capIn src level N
  | st => throw s!"{st.dir}: no input builder for this statement"

/-- A row's words as bits, each word's 16 least significant first. -/
def wordBits (ws : Array Nat) : List Bool := ws.toList.flatMap fun w => (List.range 16).map w.testBit

end VStar.In
