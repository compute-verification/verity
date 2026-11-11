module

public import Spot.Interval

/-!
# Rankin's factor, bounded above (`Spot.FactorUpper`)

Rankin's bound on the two-stage count (`Claims.Sampling.AdmissibleTwoRankin`) multiplies one factor per replay block,
`G_ρ(θ) = Σ_{U ⊆ ρ} 2^{b(U)} (1 − q(1 − e_ρ(U)))^θ` (`Claims.Sampling.rankinFactor`), and its patterns' form one
patterns' factor (`rankinFactorPatterns`). Neither is evaluated directly: a replay block holds up to `10^11` proof
blocks. This module bounds each above, in exact arithmetic, from the replay block's cells (`Cell`: the second stage's
`k` and its proof blocks by committed bits), and returns `ln` of the bound, which `G_ρ` itself, as large as
`2^{10^13}`, can't be:

* `lnFactorUpper cs P`: at least `ln G_ρ(θ)` (`Properties.Spot.FactorUpperSound`);
* `lnFactorUpperPatterns cs reads N S P`: at least `ln` of the patterns' factor, when at most `N` requests read the
  replay block's proof blocks, each serving `S` bits and reading at most `reads`, and no two read one proof block
  (`FactorUpperPatternsSound`);
* `lnFactorUpperCapped cs S P`: at least `ln` of the capped factor, each set's bits capped at `S`
  (`FactorUpperCappedSound`).

The bound is `optimize.two_stage_exfil`'s, term by term:

1. a cell's escape is at most `e^{−k m/n}` (`0` once `k ≥ n` and `m ≥ 1`), so a set `U` passes with probability at most
   `e^{−h(y(U))}`, `y(U) = Σ_{v∈U} k_v/n_v` over `U`'s proof blocks outside the cells sampled whole and
   `h(y) = −ln(1 − q(1 − e^{−y}))`;
2. `h` is increasing and concave, so on `[y_j, y_{j+1}]` it is at least its chord through rationals below it at the two
   ends (`chordsOf`, `hLower`), and past the last breakpoint at least its value there: each chord's weight
   `e^{−θ(a_j + s_j y(U))}` is a product over `U`'s proof blocks, and the sum over `U` a product over the groups of
   proof blocks of one cell and width (`Group`), `∏_g (1 + 2^{b_g} w_g)^{n_g}`;
3. a set meeting a cell sampled whole passes with probability `1 − q`;
4. in the patterns' form, `y(U) ≥ Σ_r y(U_r)` over the requests, so each chord's weight is a product over them, and the
   sum over patterns `(1 + Light + Heavy)^N` (`requestFac`): a request's set of at least `S` bits serves at most `2^S`
   outputs and has `y ≥ y_S = min_g r_g max(1, S/b_g)` (`heavyY`, a breakpoint), and its sets of at most `S` bits are
   summed (`lightPatterns`) by the lesser of two bounds: each group's sets of at most `⌊S/b_g⌋` of its proof blocks,
   `Σ_{1≤j≤m} C(n, j) x^j`, bounded by `C(n, j) ≤ n^j/j!` and Stirling's `ln m! ≥ m ln m − m + ½ ln(2πm)` (`lnHead`);
   and `2^S Σ_{1≤j≤S/b_min} W^j/j!`, `W = Σ_v w_v`;
5. in the capped form, a set's `2^{min(b(U), S)}` is at most the product of `2^{min(b_g(U), S)}` over the groups, so
   each group's sum is the lesser of `(1 + 2^{b_g} w_g)^{n_g}` and its sets of at most `m = ⌊S/b_g⌋` proof blocks plus
   `2^S` times the rest (`servedFac`), the rest at most `(1 + w_g)^{n_g} − 1` and, at `z = n_g w_g < m + 2`, at most
   `z^{m+1}/(m+1)! · 1/(1 − z/(m+2))` (`heavyTail`); the whole sum is also at most `2^S (∏_g (1 + w_g)^{n_g} − 1)`.

Every quantity is carried as an upper bound on its logarithm, and a factor `1 + L` as bounds on both `ln(1 + L)` and
`ln L` (`Fac`), so that a product of factors each barely above `1` keeps its size. The precision `p` (and `θ`'s, for
`h`) only decides how tight the bound is and how long it takes: the bound holds at every choice. Rankin's `θ` and the
breakpoints are the caller's (`optimize.two_stage_exfil` picks `θ`; `chords` is `optimize.CHORDS`), and it holds at
every `θ ≥ 0` and every increasing list of breakpoints from `0`.
-/

@[expose] public section

namespace Spot.FactorUpper

open Spot.Interval

/-! ## Logarithms -/

/-- At least `e^x`: `2^-p` once `x ≤ −p`, since `e^{−p} ≤ 2^{−p}`, else `expUpper`. `expUpper` works to `e^x`'s
integer bits, so callers keep `x` small. -/
def expUp (x : Rat) (p : Nat) : Rat := if x ≤ -(p : Rat) then 1 / 2 ^ p else expUpper x p

/-- At most `e^x`: `0` once `x ≤ −p`, else `expLower`, which would work to `e^{−x}`'s integer bits. -/
def expLo (x : Rat) (p : Nat) : Rat := if x ≤ -(p : Rat) then 0 else expLower x p

/-- Past this, `e^λ` isn't computed: `e^32` exceeds every `ln(1 + L)` a replay block's factors reach. -/
def expCap : Rat := 32

/-- The largest of `ts`, `none` for none. -/
def maxOf : List Rat → Option Rat
  | [] => none
  | t :: ts => some (ts.foldl max t)

/-- At least `ln Σ_{t ∈ ts} e^t`, from the largest; `none` for no terms, an empty sum. -/
def lse (ts : List Rat) (p : Nat) : Option Rat :=
  (maxOf ts).map fun m => m + lnUpper ((ts.map fun t => expUp (t - m) p).sum) p

/-- At most `ln Σ_{t ∈ ts} e^t`, from the largest; `none` for no terms. -/
def lseLower (ts : List Rat) (p : Nat) : Option Rat :=
  (maxOf ts).map fun m =>
    let z := (ts.map fun t => expLo (t - m) p).sum
    if 0 < z then max m (m + lnLower z p) else m

/-- At least `ln(1 + e^x)`: `e^x` itself once `x ≤ −24`, within a factor `1 + e^{−24}`. -/
def softplusUp (x : Rat) (p : Nat) : Rat :=
  if x ≤ -24 then expUp x p
  else
    let m := max 0 x
    m + lnUpper (expUp (-m) p + expUp (x - m) p) p

/-- At least `ln(1 + L)` for `L ≤ e^l` (`L = 0` at `none`). -/
def lnOnePlus (l : Option Rat) (p : Nat) : Rat :=
  match l with
  | none => 0
  | some l => softplusUp l p

/-- The smaller of two bounds on one logarithm, `none` (the quantity is `0`) beating any. -/
def omin : Option Rat → Option Rat → Option Rat
  | some a, some b => some (min a b)
  | _, _ => none

/-- The bound on a product's logarithm: their sum, `none` (a factor `0`) at either. -/
def oadd : Option Rat → Option Rat → Option Rat
  | some a, some b => some (a + b)
  | _, _ => none

/-- At least `ln(e^X − 1)` for `0 ≤ X ≤ P` and `X ≤ e^{lP}`, `none` when `P ≤ 0`, where `X = 0`: `e^X − 1` is at most
`e^P (1 − e^{−P})` and at most `X e^X`, the latter alone once `P ≤ 2^-20`. -/
def logExpm1 (P lP : Rat) (p : Nat) : Option Rat :=
  if P ≤ 0 then none
  else if P ≤ 1 / 2 ^ 20 then some (P + lP)
  else some (P + min lP (lnUpper (1 - expLo (-P) p) p))

/-- At least `X` for `X ≤ f` and `X ≤ e^λ`: the smaller, `e^λ` bounded only up to `expCap`. -/
def minExp (f lam : Rat) (p : Nat) : Rat := if lam ≤ expCap then min f (expUp lam p) else f

/-- At most `ln m!`, by Stirling's `m! ≥ √(2πm) (m/e)^m` and `π > 3.14`. -/
def lnFactLower (m p : Nat) : Rat :=
  if m = 0 then 0 else m * lnLower m p - m + (lnLower (2 * m) p + lnLower (314 / 100) p) / 2

/-! ## Factors -/

/-- **A factor `1 + L`, `L ≥ 0`, as two logarithms**: `1 + L ≤ e^f`, and `L ≤ e^l` (`L = 0` at `none`). -/
structure Fac where
  f : Rat
  l : Option Rat
  deriving Repr, Inhabited

/-- `1 + x` for `x ≤ e^{lx}`. -/
def Fac.one (lx : Rat) (p : Nat) : Fac := ⟨softplusUp lx p, some lx⟩

/-- A product of factors: `∏ (1 + L_i) = 1 + L` with `ln(1 + L) ≤ Σ f_i`, and `L = e^X − 1` at `X ≤ Σ L_i`. -/
def Fac.prod (fs : List Fac) (p : Nat) : Fac :=
  let f := (fs.map Fac.f).sum
  match lse (fs.filterMap Fac.l) p with
  | none => ⟨f, none⟩
  | some lam => ⟨f, logExpm1 (minExp f lam p) lam p⟩

/-- A factor's `n`-th power, with `lnN ≥ ln n`. -/
def Fac.pow (F : Fac) (n : Nat) (lnN : Rat) (p : Nat) : Fac :=
  match F.l with
  | none => ⟨n * F.f, none⟩
  | some l =>
    if n = 0 then ⟨0, none⟩
    else
      let lam := l + lnN
      ⟨n * F.f, logExpm1 (minExp (n * F.f) lam p) lam p⟩

/-- At least `ln Σ_{1≤j≤m} z^j/j!` for `z > 0` with `lzLo ≤ ln z ≤ lz`, `lm ≥ ln m` and `lf ≤ ln m!`; `none` at `m = 0`,
an empty sum. It is at most `e^z − 1` and at most `(1 + z)^m − 1`, and once `z > m` at most its last term times
`min(m, 1/(1 − m/z))`, each term at most `m/z` of the next. -/
def lnHead (m : Nat) (lm lf lz lzLo : Rat) (p : Nat) : Option Rat :=
  if m = 0 then none
  else
    let whole := if lz ≤ expCap then logExpm1 (expUp lz p) lz p else some (m * softplusUp lz p)
    if lm < lzLo then
      let e := expUp (lm - lzLo) p
      let last := m * lz - lf + (if e < 1 then min lm (-lnLower (1 - e) p) else lm)
      some (match whole with
        | some w => min w last
        | none => last)
    else whole

/-! ## A replay block -/

/-- **A replay block's cell**: its proof blocks in one stratum, as `(count, bits)` pairs (`parts`), and the second
stage's `k` there. -/
structure Cell where
  k : Nat
  parts : List (Nat × Nat)
  deriving Repr, Inhabited

/-- The proof blocks of `parts` committing `b` bits. -/
def countIn (parts : List (Nat × Nat)) (b : Nat) : Nat := ((parts.filter fun x => x.2 == b).map Prod.fst).sum

/-- The cell's proof blocks, `n`. -/
def Cell.size (c : Cell) : Nat := (c.parts.map Prod.fst).sum

/-- The cell's proof blocks committing `b` bits. -/
def Cell.count (c : Cell) (b : Nat) : Nat := countIn c.parts b

/-- The rate `k/n` the second stage samples the cell's proof blocks at, `none` when it samples it whole (`k ≥ n`). -/
def Cell.rate (c : Cell) : Option Rat := if c.size ≤ c.k then none else some ((c.k : Rat) / (c.size : Rat))

/-- `l` without repeats, each kept at its last place. -/
def dedup : List Nat → List Nat
  | [] => []
  | x :: xs => if xs.contains x then dedup xs else x :: dedup xs

/-- **A group**: `n` proof blocks committing `b` bits each, in stratum `t`, whose second stage samples each at rate
`rate`, `none` when it samples the cell whole; with the logarithms its sums read, computed once: `lnNLo ≤ ln n ≤ lnN`,
and at `m = ⌊S/b⌋`, `lm ≥ ln m` and `lf ≤ ln m!`. -/
structure Group where
  t : Nat
  n : Nat
  b : Nat
  rate : Option Rat
  lnN : Rat
  lnNLo : Rat
  lm : Rat
  lf : Rat
  deriving Repr, Inhabited

/-- The group's rate, `0` when its cell is sampled whole. -/
def Group.r (g : Group) : Rat := g.rate.getD 0

/-- The group's stratum and width. -/
def Group.key (g : Group) : Nat × Nat := (g.t, g.b)

/-- The `(stratum, width)` pairs of `reads`, `reads[t]`'s `(count, bits)` in stratum `t` of the cells `cs`, with a
proof block or more, each once. -/
def keysOf (cs : List Cell) (reads : List (List (Nat × Nat))) : List (Nat × Nat) :=
  (List.range cs.length).flatMap fun t =>
    ((dedup ((reads.getD t []).map Prod.snd)).filter fun b => 0 < countIn (reads.getD t []) b).map (t, ·)

/-- **The groups of `reads`** at the rates of the cells `cs`, one per `(stratum, width)`, its logarithms at `S` and
precision `p`. -/
def readGroups (cs : List Cell) (reads : List (List (Nat × Nat))) (S p : Nat) : List Group :=
  (keysOf cs reads).map fun k =>
    let n := countIn (reads.getD k.1 []) k.2
    ⟨k.1, n, k.2, (cs.getD k.1 default).rate, lnUpper n p, lnLower n p, lnUpper ((S / k.2 : Nat) : Rat) p,
      lnFactLower (S / k.2) p⟩

/-- The replay block's groups. -/
def groupsOf (cs : List Cell) (S p : Nat) : List Group := readGroups cs (cs.map Cell.parts) S p

/-! ## The parameters -/

/-- **The bound's parameters**: the first stage's rate `q` on the replay block, Rankin's `θ`, the chords' breakpoints
`ys` and the precision `p`. -/
structure Params where
  q : Rat
  θ : Rat
  ys : List Rat
  p : Nat
  deriving Repr, Inhabited

/-- `optimize.CHORDS`: the breakpoints `two_stage_exfil`'s chords of `h` are drawn at. -/
def chords : List Rat := [0, 1 / 256, 1 / 16, 1 / 4, 1, 2, 3, 4, 5, 6, 8]

/-- `ys` strictly increasing. -/
def increasing : List Rat → Bool
  | x :: y :: ys => decide (x < y) && increasing (y :: ys)
  | _ => true

/-- `ys` with `y` in its place, unless it is one of them. -/
def insertY (y : Rat) : List Rat → List Rat
  | [] => [y]
  | x :: xs => if y < x then y :: x :: xs else if y = x then x :: xs else x :: insertY y xs

/-- The parameters the bound is computed at: `0 ≤ q ≤ 1`, `θ ≥ 0`, and breakpoints increasing from `0`. -/
def Params.valid (P : Params) : Bool :=
  decide (0 ≤ P.q) && decide (P.q ≤ 1) && decide (0 ≤ P.θ) && decide (P.ys.head? = some 0) && increasing P.ys

/-- The precision `h` is bounded at: `p`, and `θ`'s bits, since `θ` multiplies its error. -/
def Params.pθ (P : Params) : Nat := P.p + P.θ.ceil.toNat.log2 + 1

/-- **`ln 2` at precision `p`**, computed once: `lo ≤ ln 2 ≤ up`. -/
structure Ln2 where
  up : Rat
  lo : Rat
  deriving Repr, Inhabited

/-- `ln 2`'s bounds at `p`. -/
def Ln2.of (p : Nat) : Ln2 := ⟨lnUpper 2 p, lnLower 2 p⟩

/-! ## The chords of `h` -/

/-- At most `h(y) = −ln(1 − q(1 − e^{−y}))`. -/
def hLower (q y : Rat) (p : Nat) : Rat := -lnUpper (1 - q + q * expUp (-y) p) p

/-- The chords `(s_j, a_j)` of `h` between consecutive breakpoints, through `hLower` at each: `a_j + s_j y ≤ h(y)` on
`[y_j, y_{j+1}]`, as `h` is concave. -/
def chordsOf (q : Rat) (p : Nat) : List Rat → List (Rat × Rat)
  | y0 :: y1 :: ys =>
    let h0 := hLower q y0 p
    let s := (hLower q y1 p - h0) / (y1 - y0)
    (s, h0 - s * y0) :: chordsOf q p (y1 :: ys)
  | _ => []

/-- At least `ln((1 − q)^θ)`, `none` at `q = 1 < θ`'s `0`. -/
def lnPassWhole (q θ : Rat) (p : Nat) : Option Rat :=
  if θ = 0 then some 0 else if q < 1 then some (θ * lnUpper (1 - q) p) else none

/-- **The bound from its sums**: `ln(1 + Σ e^T)` over the terms `T`, at least the factor's logarithm when, at every
slope `s`, `e^{term s}` is at least the sum over the nonempty patterns of the sets avoiding the cells sampled whole of
`2^{bits} e^{−θ s y}`, `y` the least `y(U)` of the pattern, and `e^{whole}` at least the sum over the patterns meeting
them of `2^{bits} (1 − q)^θ`: each such `y` is on a chord, `h(y) ≥ a_j + s_j y`, or past the last breakpoint,
`h(y) ≥ h(y_J)`. -/
def bound (P : Params) (term : Rat → Option Rat) (whole : Option Rat) : Option Rat :=
  let chordTerms := (chordsOf P.q P.pθ P.ys).filterMap fun c => (term c.1).map (-(P.θ * c.2) + ·)
  let tail := P.ys.getLast?.bind fun y => (term 0).map (-(P.θ * hLower P.q y P.pθ) + ·)
  lse (0 :: chordTerms ++ [tail, whole].filterMap id) P.p

/-! ## The sums over sets -/

/-- `(1 + 2^b w)^n` at `w = e^{lw}`: a group's sum over its sets of `2^{b(U)} ∏_{v∈U} w`. -/
def wholeFac (L : Ln2) (g : Group) (lw : Rat) (p : Nat) : Fac := (Fac.one (g.b * L.up + lw) p).pow g.n g.lnN p

/-- The group's sets of at most `S` bits, `1 + Σ_{1≤j≤m} C(n, j) x^j` at `x = 2^b w` and `m = ⌊S/b⌋`: the whole sum
when the group holds at most `m`, else the least of it and `lnHead`'s bound by `C(n, j) ≤ n^j/j!`. -/
def cappedFac (L : Ln2) (g : Group) (lw : Rat) (S p : Nat) : Fac :=
  let W := wholeFac L g lw p
  if g.b = 0 ∨ g.n ≤ S / g.b then W
  else
    let l := omin W.l (lnHead (S / g.b) g.lm g.lf (g.lnN + g.b * L.up + lw) (g.lnNLo + g.b * L.lo + lw) p)
    ⟨min W.f (lnOnePlus l p), l⟩

/-- **The sets' sum**: at least `ln Σ_{∅ ≠ U} 2^{b(U)} ∏_{v∈U} w_v` over the sets of the groups' proof blocks, each
group `g` at weight `w = e^{lw}`, `∏_g (1 + 2^{b_g} w_g)^{n_g} − 1`. -/
def lightSets (L : Ln2) (gs : List (Group × Rat)) (p : Nat) : Option Rat :=
  (Fac.prod (gs.map fun x => wholeFac L x.1 x.2 p) p).l

/-- **The light sets' sum**: at least `ln Σ_{∅ ≠ U, b(U) ≤ S} 2^{b(U)} ∏_{v∈U} w_v`, the lesser of: the product of the
groups' capped sums (`cappedFac`); and, `b_min > 0` the narrowest group's bits, `2^S Σ_{1≤j≤S/b_min} W^j/j!` at
`W = Σ_v w_v`, a set of at most `S` bits holding at most `S/b_min` proof blocks. -/
def lightPatterns (L : Ln2) (gs : List (Group × Rat)) (S p : Nat) : Option Rat :=
  let prod := (Fac.prod (gs.map fun x => cappedFac L x.1 x.2 S p) p).l
  let bmin := (gs.map fun x => x.1.b).foldl min (S + 1)
  if bmin = 0 then prod
  else
    match lse (gs.map fun x => x.1.lnN + x.2) p, lseLower (gs.map fun x => x.1.lnNLo + x.2) p with
    | some lW, some lWLo =>
      omin prod ((lnHead (S / bmin) (lnUpper ((S / bmin : Nat) : Rat) p) (lnFactLower (S / bmin) p) lW lWLo p).map
        ((S : Rat) * L.up + ·))
    | _, _ => prod

/-- The groups sampled at a rate, and the groups sampled whole. -/
def split (gs : List Group) : List Group × List Group :=
  (gs.filter fun g => g.rate.isSome, gs.filter fun g => g.rate.isNone)

/-! ## Rankin's factor -/

/-- **At least `ln G_ρ(θ)`**, Rankin's factor of the replay block with cells `cs` (`Claims.Sampling.rankinFactor`):
at slope `s`, the sets' sum at `w = e^{−θ s r}`, and the sets meeting a cell sampled whole,
`(1 − q)^θ ∏_{F} (1 + 2^b)^n (∏_{I} (1 + 2^b)^n − 1)`. `none` when the parameters are not valid. -/
def lnFactorUpper (cs : List Cell) (P : Params) : Option Rat :=
  if P.valid then
    let L := Ln2.of P.p
    let (fin, inf) := split (groupsOf cs 0 P.p)
    let whole := if inf.isEmpty then none
      else oadd (lnPassWhole P.q P.θ P.p) ((Fac.prod (inf.map (wholeFac L · 0 P.p)) P.p).l.map
        ((Fac.prod (fin.map (wholeFac L · 0 P.p)) P.p).f + ·))
    bound P (fun s => lightSets L (fin.map fun g => (g, -(P.θ * s * g.r))) P.p) whole
  else none

/-! ## The patterns' factor -/

/-- The least `y` of a request's heavy set avoiding the cells sampled whole, `y_S = min_g r_g max(1, S/b_g)` over the
groups (of nonzero bits, when `S > 0`); `none` when the groups hold fewer than `S` bits, so that there is none. -/
def heavyY (fin : List Group) (S : Nat) : Option Rat :=
  if (fin.map fun g => g.n * g.b).sum < S then none
  else
    match fin.filterMap fun g => if g.b = 0 ∧ 0 < S then none else some (g.r * max 1 ((S : Rat) / (g.b : Rat))) with
    | [] => none
    | y :: ys => some (ys.foldl min y)

/-- **A request's factor at slope `s`**, `θs = θ s`: `1 + Σ_{A light} 2^{b(A)} e^{−θ s y(A)} + 2^S e^{−θ s y_S}`, over
its classes avoiding the cells sampled whole: none, light, heavy. -/
def requestFac (L : Ln2) (fin : List Group) (yS : Option Rat) (θs : Rat) (S p : Nat) : Fac :=
  let light := lightPatterns L (fin.map fun g => (g, -(θs * g.r))) S p
  let l := lse ([light, yS.map fun y => (S : Rat) * L.up - θs * y].filterMap id) p
  ⟨lnOnePlus l p, l⟩

/-- At least `ln` of the patterns' count meeting a cell sampled whole, `N` requests in all: with `T` a request's classes
meeting one (`2^S` heavy, the light sets meeting one `∏_{F} (1 + L_g) (∏_{I} (1 + L_g) − 1)`) and `A` all its classes,
the lesser of `N T A^{N−1}` (one of the requests meets one) and `A^N`. -/
def touchCount (L : Ln2) (fin inf : List Group) (N S p : Nat) : Option Rat :=
  if N = 0 then none
  else
    let finF := Fac.prod (fin.map (cappedFac L · 0 S p)) p
    let infF := Fac.prod (inf.map (cappedFac L · 0 S p)) p
    let heavy : Option Rat :=
      if ((fin ++ inf).map fun g => g.n * g.b).sum < S then none else some ((S : Rat) * L.up)
    let lA := (lse ([some (finF.f + infF.f), heavy].filterMap id) p).getD 0
    (lse ([infF.l.map (finF.f + ·), heavy].filterMap id) p).map fun lT =>
      min (lnUpper N p + lT + (N - 1 : Nat) * lA) (N * lA)

/-- **At least `ln` of the patterns' factor** of the replay block with cells `cs`
(`Claims.Sampling.rankinFactorPatterns`), its proof blocks read by at most `N` requests, each serving `S` bits and
reading at most `reads[t]`'s `(count, bits)` in stratum `t`, no two reading one proof block: as `y(U) ≥ Σ_r y(U_r)`, a
chord's weight is a product over the requests, and the sum over the patterns of the sets avoiding the cells sampled
whole `(requestFac)^N − 1`, with `y_S` a breakpoint so that a heavy request's weight is `h(y_S)`'s; and the patterns
meeting one are `touchCount`, at `(1 − q)^θ`. -/
def lnFactorUpperPatterns (cs : List Cell) (reads : List (List (Nat × Nat))) (N S : Nat) (P : Params) :
    Option Rat :=
  if P.valid then
    let L := Ln2.of P.p
    let (fin, inf) := split (readGroups cs reads S P.p)
    let yS := heavyY fin S
    let lnN := lnUpper N P.p
    let whole := if inf.isEmpty then none else oadd (lnPassWhole P.q P.θ P.p) (touchCount L fin inf N S P.p)
    let P' := match yS with
      | some y => if ({ P with ys := insertY y P.ys } : Params).valid then { P with ys := insertY y P.ys } else P
      | none => P
    bound P' (fun s => ((requestFac L fin yS (P.θ * s) S P.p).pow N lnN P.p).l) whole
  else none

/-! ## The capped factor -/

/-- At least `ln Σ_{j>m} C(n, j) w^j`, a group's sets of more than `m = ⌊S/b⌋` of its proof blocks at `w = e^{lw}`:
`(1 + w)^n − 1`, and once `z = n w < m + 2` also `z^{m+1}/(m+1)! · 1/(1 − z/(m+2))`, as `C(n, j) ≤ n^j/j!` and each
term is at most `z/(m+2)` of the one before. -/
def heavyTail (g : Group) (lw : Rat) (S p : Nat) : Option Rat :=
  let grow := ((Fac.one lw p).pow g.n g.lnN p).l
  let m := S / g.b
  let e := expUp (g.lnN + lw - lnLower ((m + 2 : Nat) : Rat) p) p
  if e < 1 then
    omin grow (some ((m + 1 : Nat) * (g.lnN + lw) - (g.lf + lnLower ((m + 1 : Nat) : Rat) p) - lnLower (1 - e) p))
  else grow

/-- **A group's served sum**: at least `ln Σ_U 2^{min(b(U), S)} ∏_{v∈U} w` over its sets at `w = e^{lw}`, the lesser
of `(1 + 2^b w)^n` (`wholeFac`) and, once a set can hold more than `S` bits, its sets of at most `⌊S/b⌋` proof blocks
(`cappedFac`) plus `2^S` times the rest (`heavyTail`). -/
def servedFac (L : Ln2) (g : Group) (lw : Rat) (S p : Nat) : Fac :=
  let W := wholeFac L g lw p
  if g.b = 0 ∨ g.n ≤ S / g.b then W
  else
    let l := omin W.l (lse ([(cappedFac L g lw S p).l,
      (heavyTail g lw S p).map ((S : Rat) * L.up + ·)].filterMap id) p)
    ⟨min W.f (lnOnePlus l p), l⟩

/-- **The served sets' sum**: at least `ln Σ_{∅ ≠ U} 2^{min(b(U), S)} ∏_{v∈U} w_v` over the sets of the groups' proof
blocks, each group `g` at weight `w = e^{lw}`: the lesser of the product of the groups' `servedFac`, as
`min(a + b, S) ≤ min(a, S) + min(b, S)`, and `2^S (∏_g (1 + w_g)^{n_g} − 1)`. -/
def servedSets (L : Ln2) (gs : List (Group × Rat)) (S p : Nat) : Option Rat :=
  omin (Fac.prod (gs.map fun x => servedFac L x.1 x.2 S p) p).l
    ((Fac.prod (gs.map fun x => (Fac.one x.2 p).pow x.1.n x.1.lnN p) p).l.map ((S : Rat) * L.up + ·))

/-- **At least `ln G^S_ρ(θ)`**, Rankin's capped factor of the replay block with cells `cs`
(`Claims.Sampling.rankinFactorCapped`), each set's bits capped at `S`: at slope `s`, the served sets' sum at
`w = e^{−θ s r}`, and the sets meeting a cell sampled whole, `(1 − q)^θ` times the lesser of
`∏_F G_g (∏_I G_g − 1)` over the groups' served sums at `w = 1` and `2^S 2^{n_F} (2^{n_I} − 1)`. `none` when the
parameters are not valid. -/
def lnFactorUpperCapped (cs : List Cell) (S : Nat) (P : Params) : Option Rat :=
  if P.valid then
    let L := Ln2.of P.p
    let (fin, inf) := split (groupsOf cs S P.p)
    let whole := if inf.isEmpty then none
      else oadd (lnPassWhole P.q P.θ P.p) (omin
        ((Fac.prod (inf.map (servedFac L · 0 S P.p)) P.p).l.map
          ((Fac.prod (fin.map (servedFac L · 0 S P.p)) P.p).f + ·))
        ((Fac.prod (inf.map fun g => (Fac.one 0 P.p).pow g.n g.lnN P.p) P.p).l.map
          ((S : Rat) * L.up + (fin.map fun g => (g.n : Rat)).sum * L.up + ·)))
    bound P (fun s => servedSets L (fin.map fun g => (g, -(P.θ * s * g.r))) S P.p) whole
  else none

end Spot.FactorUpper
