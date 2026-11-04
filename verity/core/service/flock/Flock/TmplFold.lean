/-
The lincheck fold of a templated unit under layout (iv), lenient, from its description. No verifier path reads
it yet.

For a unit slot of `2^sl` bits, `Stmt.partialRanges` needs `p = project (baseOf α e_lo A B none sl) (eqTable r)`, with
`e_lo = lw ⊗ eq(ρ, ·)`, `ρ` and `r` of `sl − 6` coordinates, and `A`, `B` the unit's rows at their committed positions:
`p[s] = Σ_{entries (x, y)} c · e(π x) · [π y mod 64 = s] · eq(r, ⌊π y / 64⌋)` with `e(q) = lw[q mod 64] · eq(ρ, ⌊q/64⌋)`,
`c` α for A's entries and 1 for B's, and `π` the layout's position. `foldRef` is that sum over the unrolled rows, the
reference. `fold` computes it once per call path (each run of copies counted once), never per copy.

A Definition's copy placed under `q = a + B · (h + p)` (`B = 2^β`, `a < B` the outer runs' lane bits, `h` its base)
is a *class*. While `β < 6` the class keeps its copies' low parts `a` explicitly (fewer than 64) and adds to `p` as it
goes. From `β ≥ 6` on, `q mod 64 = a mod 64` and `⌊q/64⌋`'s bits split at `β − 6`, so `e(q) = E(a) · eq(ρ[β−6 ..], h + p)`
and the slot's weight `G_s(a) · eq(r[β−6 ..], h + p)` likewise: the class sums its entries with the high factors only
(`s`, and `c` for entries reading the constant), and its caller scales them by `Σ_a E(a) G_s(a)` and `Σ_a E(a)`. A run of
`count` copies in `M = 2^m` lanes adds `m` bits to `a`, whose sum over the copies is `laneSum` (1 for a full run, over
GF(2^128)). An entry reading a Definition's input is left open (`u`), as its `E` sum, and paired with the column its
caller binds, copy by copy.

Cost: per call path, its Definition's own entries and its calls' bound inputs times their copies, each an `eq` of at
most `sl` coordinates and a position read through the call tree; `work` counts the entries and binding terms.
-/
import Flock.TmplLayout

namespace Flock.Tmpl

/-- `eq(r[lo .. lo + n], u)`: `Π_{j < n}` of `r[lo + j]` where bit `j` of `u` is set, else `r[lo + j] + 1` (a coordinate
past `r` is 0); 0 if `u ≥ 2^n`. -/
def eqAt (r : Array F128) (lo n u : Nat) : F128 :=
  if u ≥ 2 ^ n then 0
  else Nat.fold n (fun j _ acc => acc * (if u.testBit j then r.getD (lo + j) 0 else r.getD (lo + j) 0 + 1)) 1

/-- `eq(r[lo ..], ·)` for every `lo ≤ |r|` in two halves: `eq(r[lo ..], u) = lo-table[u mod 2^h] · hi-table[⌊u/2^h⌋]`,
`h` half of `|r| − lo`. -/
def halves (r : Array F128) : Array (Nat × Array F128 × Array F128) :=
  (Array.range (r.size + 1)).map fun lo =>
    let h := (r.size - lo) / 2
    (h, (Array.range (2 ^ h)).map (eqAt r lo h), (Array.range (2 ^ (r.size - lo - h))).map (eqAt r (lo + h) (r.size - lo - h)))

/-- `eq(r[lo ..], u)` from `halves r`. -/
def eqTail (T : Array (Nat × Array F128 × Array F128)) (lo u : Nat) : F128 :=
  let (h, a, b) := T.getD lo (0, #[], #[])
  a.getD (u % 2 ^ h) 0 * b.getD (u / 2 ^ h) 0

/-- `Σ_{j < count} Π_{b < m} f b (bit b of j)`, for `count ≤ 2^m`, digit by digit from the top. -/
def laneSum (f : Nat → Bool → F128) : Nat → Nat → F128
  | 0, count => if count == 0 then 0 else 1
  | m + 1, count =>
    if count ≤ 2 ^ m then f m false * laneSum f m count
    else f m false * (List.range m).foldl (fun acc b => acc * (f b false + f b true)) 1 +
      f m true * laneSum f m (count - 2 ^ m)

/-- What the fold reads: the point (`α`, `lw` of 64, `ρ` and `r` of `sl − 6`) and the constant's position. -/
structure FoldPt where
  alpha : F128
  lw : Array F128
  rho : Array F128
  r : Array F128
  qc : Nat
  rhoT : Array (Nat × Array F128 × Array F128) := halves rho
  rT : Array (Nat × Array F128 × Array F128) := halves r

def FoldPt.e (pt : FoldPt) (q : Nat) : F128 := pt.lw.getD (q % 64) 0 * eqTail pt.rhoT 0 (q / 64)

def FoldPt.g (pt : FoldPt) (q : Nat) : F128 := eqTail pt.rT 0 (q / 64)

def addAt (p : Array F128) (q : Nat) (v : F128) : Array F128 := p.modify (q % 64) (· + v)

/-- A class's fold: explicit (`β < 6`), its share `p` of the 64 entries and per input and low part its `E` sum `u`;
summed (`β ≥ 6`), its high-factor sums `s` (own entries) and `c` (entries reading the constant), and per input its
high-factor `E` sum `u[t][0]`. -/
structure Out where
  p : Array F128
  s : F128
  c : F128
  u : Array (Array F128)
  work : Nat

/-- A child class's share: its `p`, and its `s` and `c` scaled by `ls` and `lc` (its copies' lane sums). -/
def Out.add (o ch : Out) (ls lc : F128) : Out :=
  { o with p := (o.p.zip ch.p).map (fun (a, b) => a + b), s := o.s + ls * ch.s, c := o.c + lc * ch.c,
           work := o.work + ch.work }

/-- A reference `v` of a Definition of `cols` columns and `ins` inputs, its `E` weight `w` for its row in class copy
`k`: an own column adds `w` times its slot weight (`gOwn`), an input opens, the constant adds `w` to `c` (summed) or
`w · g(qc)` at `qc` (explicit). -/
def Out.ref (o : Out) (pt : FoldPt) (summed : Bool) (cols ins v k : Nat) (w : F128) (gOwn : Nat → Out → Out) : Out :=
  if v < cols then gOwn v o
  else if v < cols + ins then { o with u := o.u.modify (v - cols) (·.modify k (· + w)) }
  else if summed then { o with c := o.c + w }
  else { o with p := addAt o.p pt.qc (w * pt.g pt.qc) }

/-- **The fold of Definition `d`'s class** `(β, h)`, its low parts `as` while explicit (`fuel` bounds the depth). -/
def foldDef (t : Tmpl) (lays : Array Lay) (pt : FoldPt) : Nat → Nat → Nat → Nat → Array Nat → V Out
  | 0, _, _, _, _ => throw "template fold: calls nest deeper than the definitions"
  | fuel + 1, d, β, h, as => do
    let D := t.defs.getD d default
    let L := lays.getD d default
    let summed := 6 ≤ β
    let B := 2 ^ β
    let nk := if summed then 1 else as.size
    let pos := fun (y : Nat) => posIn t lays (t.defs.size + 1) d y
    let qOf := fun (k p : Nat) => as.getD k 0 + B * (h + p)
    let eHi := fun (p : Nat) => eqTail pt.rhoT (β - 6) (h + p)
    let gHi := fun (p : Nat) => eqTail pt.rT (β - 6) (h + p)
    -- an own column `y` read with weight `w` by class copy `k`
    let own := fun (k : Nat) (w : F128) (y : Nat) (o : Out) =>
      if summed then { o with s := o.s + w * gHi (pos y) }
      else
        let q := qOf k (pos y)
        { o with p := addAt o.p q (w * pt.g q) }
    let mut o : Out := ⟨Array.replicate 64 0, 0, 0, Array.replicate D.ins (Array.replicate nk 0), 0⟩
    for (x, ra, rb) in D.rows do
      for k in [0:nk] do
        let ex := if summed then eHi (pos x) else pt.e (qOf k (pos x))
        for (cs, coef) in [(ra, pt.alpha), (rb, (1 : F128))] do
          for v in cs do
            o := o.ref pt summed D.cols D.ins v k (coef * ex) (own k (coef * ex))
        o := { o with work := o.work + ra.size + rb.size }
    for i in [0:D.calls.size] do
      let c := D.calls.getD i default
      let pl := L.places.getD i default
      let bind := fun (j : Nat) (o : Out) (val : Nat → Nat → F128) =>
        ((c.bound id j).zipIdx.foldl (fun o (v, t') =>
          (List.range nk).foldl (fun o k => o.ref pt summed D.cols D.ins v k (val t' k) (own k (val t' k))) o) o)
      if pl.ms == 1 then
        if summed && c.count != 1 then throw "template fold: copies side by side below the root"
        for j in [0:c.count] do
          let ch ← foldDef t lays pt fuel c.def_ β (h + pl.nb + j * pl.cs) as
          o := o.add ch 1 1
          o := { o with work := o.work + c.segs.foldl (fun n (s : Seg) => n + s.reps * s.len) 0 * nk }
          o := bind j o fun t' k => (ch.u.getD t' #[]).getD k 0
      else
        let M := pl.ms
        let m := Nat.log2 M
        if (h + pl.nb) % M != 0 then throw "template fold: a run of lanes is not aligned to its lanes"
        let h' := (h + pl.nb) / M
        let ins := (t.defs.getD c.def_ default).ins
        o := { o with work := o.work + ins * c.count * nk }
        if summed then
          let ch ← foldDef t lays pt fuel c.def_ (β + m) h' #[]
          let rb := fun (b : Nat) => pt.rho.getD (β - 6 + b) 0
          let tb := fun (b : Nat) => pt.r.getD (β - 6 + b) 0
          let lrr := laneSum (fun b bit => if bit then rb b * tb b else (rb b + 1) * (tb b + 1)) m c.count
          let lr := laneSum (fun b bit => if bit then rb b else rb b + 1) m c.count
          o := o.add ch lrr lr
          for j in [0:c.count] do
            let ej := eqAt pt.rho (β - 6) m j
            o := bind j o fun t' _ => ej * (ch.u.getD t' #[]).getD 0 0
        else if β + m < 6 then
          let as' := (Array.range (nk * c.count)).map fun i => as.getD (i / c.count) 0 + B * (i % c.count)
          let ch ← foldDef t lays pt fuel c.def_ (β + m) h' as'
          o := o.add ch 0 0
          for j in [0:c.count] do
            o := bind j o fun t' k => (ch.u.getD t' #[]).getD (k * c.count + j) 0
        else
          let β' := β + m
          let ch ← foldDef t lays pt fuel c.def_ β' h' #[]
          let eLow := fun (a : Nat) => pt.lw.getD (a % 64) 0 * eqAt pt.rho 0 (β' - 6) (a / 64)
          let gLow := fun (a : Nat) => eqAt pt.r 0 (β' - 6) (a / 64)
          o := o.add ch 0 0
          let mut p := o.p
          for k in [0:nk] do
            for j in [0:c.count] do
              let a := as.getD k 0 + B * j
              p := addAt (addAt p a (eLow a * gLow a * ch.s)) pt.qc (eLow a * ch.c * pt.g pt.qc)
          o := { o with p, work := o.work + nk * c.count }
          for j in [0:c.count] do
            o := bind j o fun t' k => eLow (as.getD k 0 + B * j) * (ch.u.getD t' #[]).getD 0 0
    return o

/-- **The unit's fold**, `project (baseOf α (lw ⊗ eq(ρ, ·)) A B none sl) (eqTable r)` on its rows at their positions under
`lays`, from the description, and the fold's work. -/
def fold (t : Tmpl) (lays : Array Lay) (alpha : F128) (lw rho r : Array F128) : V (Array F128 × Nat) := do
  let qc := posOf t lays ((t.defs.getD t.root default).cols - 1)
  let pt : FoldPt := { alpha, lw, rho, r, qc }
  let o ← foldDef t lays pt (t.defs.size + 1) t.root 0 0 #[0]
  return (addAt o.p qc ((alpha + 1) * pt.e qc * pt.g qc), o.work)

/-- **The reference**: the same sum over the unrolled rows, each at its position (O(unrolled)). -/
def foldRef (t : Tmpl) (lays : Array Lay) (alpha : F128) (lw rho r : Array F128) : V (Array F128) := do
  let acc ← t.rows
  let π := posMap t lays
  let pt : FoldPt := { alpha, lw, rho, r, qc := 0 }
  let mut p := Array.replicate 64 (0 : F128)
  for x in [0:acc.ra.size] do
    let ex := pt.e (π.getD x 0)
    for v in acc.ra.getD x #[] do p := addAt p (π.getD v 0) (alpha * ex * pt.g (π.getD v 0))
    for v in acc.rb.getD x #[] do p := addAt p (π.getD v 0) (ex * pt.g (π.getD v 0))
  return p

end Flock.Tmpl
