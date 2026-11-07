module

public import SampledProofs.Interval

/-!
# Rankin's factor, bounded above (`SampledProofs.FactorUpper`)

Rankin's bound on the two-stage count (`Claims.Sampling.AdmissibleTwoRankin`) multiplies one factor per replay block,
`G_ρ(θ) = Σ_{U ⊆ ρ} 2^{b(U)} (1 − q(1 − e_ρ(U)))^θ` (`Claims.Sampling.rankinFactor`), and its patterns' form one patterns'
factor (`rankinFactorPatterns`). Neither is evaluated directly: a replay block holds up to `10^11` proof blocks. This
module bounds each above, in exact arithmetic, from the replay block's cells (`Cell`: the second stage's `k` and its
proof blocks by committed bits), and returns `ln` of the bound, which `G_ρ` itself, as large as `2^{10^13}`, can't be:

* `lnFactorUpper cs P`: at least `ln G_ρ(θ)` (`Properties.SampledProofs.FactorUpperSound`);
* `lnFactorUpperPatterns cs S P`: at least `ln` of the patterns' factor, when at most one request, serving `S` bits,
  reads the replay block's proof blocks (`FactorUpperPatternsSound`).

The bound is `optimize.two_stage_exfil`'s, term by term:

1. a cell's escape is at most `e^{−k m/n}` (`0` once `k ≥ n` and `m ≥ 1`), so a set `U` passes with probability at most
   `e^{−h(y(U))}`, `y(U) = Σ_{v∈U} k_v/n_v` over `U`'s proof blocks outside the cells sampled whole and
   `h(y) = −ln(1 − q(1 − e^{−y}))`;
2. `h` is increasing and concave, so on `[y_j, y_{j+1}]` it is at least its chord through rationals below it at the two
   ends (`chordsOf`, `hLower`), and past the last breakpoint at least its value there: each chord's weight
   `e^{−θ(a_j + s_j y(U))}` is a product over `U`'s proof blocks, and the sum over `U` a product over the groups of
   proof blocks of one cell and width (`Group`), `∏_g (1 + 2^{b_g} w_g)^{n_g}`;
3. a set meeting a cell sampled whole passes with probability `1 − q`;
4. in the patterns' form, a set of at least `S` bits serves at most `2^S` outputs and has
   `y ≥ y_S = min_g r_g max(1, S/b_g)` (`Form.heavy`), and the sets of at most `S` bits are summed (`lightPatterns`) by
   the least of three bounds: each group's sets of at most `⌊S/b_g⌋` of its proof blocks, `Σ_{1≤j≤m} C(n, j) x^j`,
   bounded by `C(n, j) ≤ n^j/j!` and Stirling's `ln m! ≥ m ln m − m + ½ ln(2πm)` (`lnHead`); `2^S (∏_v (1 + w_v) − 1)`;
   and `2^S Σ_{1≤j≤S/b_min} W^j/j!`, `W = Σ_v w_v`.

Every quantity is carried as an upper bound on its logarithm, and a factor `1 + L` as bounds on both `ln(1 + L)` and
`ln L` (`Fac`), so that a product of factors each barely above `1` keeps its size. The precision `p` (and `θ`'s, for
`h`) only decides how tight the bound is and how long it takes: the bound holds at every choice. Rankin's `θ` and the
breakpoints are the caller's (`optimize.two_stage_exfil` picks `θ`; `chords` is `optimize.CHORDS`), and it holds at
every `θ ≥ 0` and every increasing list of breakpoints from `0`.
-/

@[expose] public section

namespace SampledProofs.FactorUpper

open SampledProofs.Interval

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

/-- The cell's proof blocks, `n`. -/
def Cell.size (c : Cell) : Nat := (c.parts.map Prod.fst).sum

/-- The cell's proof blocks committing `b` bits. -/
def Cell.count (c : Cell) (b : Nat) : Nat := ((c.parts.filter fun x => x.2 == b).map Prod.fst).sum

/-- `l` without repeats, each kept at its last place. -/
def dedup : List Nat → List Nat
  | [] => []
  | x :: xs => if xs.contains x then dedup xs else x :: dedup xs

/-- **A group**: `n` proof blocks committing `b` bits each, in a cell whose second stage samples each at rate `k/n_t`
(`rate`), `none` when the cell is sampled whole (`k ≥ n_t`); with the logarithms its sums read, computed once:
`lnNLo ≤ ln n ≤ lnN`, and at `m = ⌊S/b⌋`, `lm ≥ ln m` and `lf ≤ ln m!`. -/
structure Group where
  n : Nat
  b : Nat
  rate : Option Rat
  lnN : Rat
  lnNLo : Rat
  lm : Rat
  lf : Rat
  deriving Repr, Inhabited

/-- The group of `n` proof blocks of `b` bits at `rate`, its logarithms at `S` and precision `p`. -/
def Group.of (n b : Nat) (rate : Option Rat) (S p : Nat) : Group :=
  ⟨n, b, rate, lnUpper n p, lnLower n p, lnUpper (S / b) p, lnFactLower (S / b) p⟩

/-- The cell's groups, one per width, each nonempty. -/
def Cell.groups (c : Cell) (S p : Nat) : List Group :=
  let r : Option Rat := if c.size ≤ c.k then none else some ((c.k : Rat) / (c.size : Rat))
  ((dedup (c.parts.map Prod.snd)).filter fun b => 0 < c.count b).map fun b => Group.of (c.count b) b r S p

/-- The replay block's groups. -/
def groupsOf (cs : List Cell) (S p : Nat) : List Group := cs.flatMap (Cell.groups · S p)

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

/-- **The light sets' sum**: at least `ln Σ_{∅ ≠ U, b(U) ≤ S} 2^{b(U)} ∏_{v∈U} w_v`, the least of: the product of the
groups' capped sums (`cappedFac`); `2^S (∏_v (1 + w_v) − 1)`; and, `b_min` the narrowest group's bits,
`2^S Σ_{1≤j≤S/b_min} W^j/j!` at `W = Σ_v w_v`, a set of at most `S` bits holding at most `S/b_min` proof blocks. -/
def lightPatterns (L : Ln2) (gs : List (Group × Rat)) (S p : Nat) : Option Rat :=
  let cap := (S : Rat) * L.up
  let prod := (Fac.prod (gs.map fun x => cappedFac L x.1 x.2 S p) p).l
  let grow := ((Fac.prod (gs.map fun x => (Fac.one x.2 p).pow x.1.n x.1.lnN p) p).l).map (cap + ·)
  let bmin := (gs.map fun x => x.1.b).foldl min (S + 1)
  let head : Option Rat :=
    if bmin = 0 then prod
    else
      match lse (gs.map fun x => x.1.lnN + x.2) p, lseLower (gs.map fun x => x.1.lnNLo + x.2) p with
      | some lW, some lWLo =>
        (lnHead (S / bmin) (lnUpper (S / bmin) p) (lnFactLower (S / bmin) p) lW lWLo p).map (cap + ·)
      | _, _ => none
  omin prod (omin grow head)

/-- **The two forms**: the sets' factor `G_ρ`, or the patterns' at a request serving `S` bits. -/
inductive Form where
  | sets
  | patterns (S : Nat)
  deriving Repr, Inhabited

/-- The `S` the form's groups read: `0` in the sets' form. -/
def Form.served : Form → Nat
  | .sets => 0
  | .patterns S => S

/-- The form's sum over sets of the groups' proof blocks, each group at its weight's logarithm. -/
def Form.light (F : Form) (L : Ln2) (gs : List (Group × Rat)) (p : Nat) : Option Rat :=
  match F with
  | .sets => lightSets L gs p
  | .patterns S => lightPatterns L gs S p

/-- The form's factor of a group at weight `1`: every set of it, or its sets of at most `S` bits. -/
def Form.fac (F : Form) (L : Ln2) (g : Group) (p : Nat) : Fac :=
  match F with
  | .sets => wholeFac L g 0 p
  | .patterns S => cappedFac L g 0 S p

/-- At least `ln Σ_{U meets I} c(U)` over the sets meeting the cells sampled whole (`inf`), `c(U)` the form's charge:
`∏_{F} (1 + L_g) (∏_{I} (1 + L_g) − 1)`, and in the patterns' form also `2^S 2^{|F|} (2^{|I|} − 1)`. -/
def Form.whole (F : Form) (L : Ln2) (fin inf : List Group) (p : Nat) : Option Rat :=
  let byGroup := (Fac.prod (inf.map (F.fac L · p)) p).l.map
    ((Fac.prod (fin.map (F.fac L · p)) p).f + ·)
  match F with
  | .sets => byGroup
  | .patterns S =>
    let nI := (inf.map Group.n).sum
    let count := ((fin.map Group.n).sum : Rat) * L.up
    omin byGroup (((Fac.one 0 p).pow nI (lnUpper nI p) p).l.map (S * L.up + count + ·))

/-- At least `ln(2^S e^{−θ h(y_S)})`, the sets of at least `S` bits, in the patterns' form, when the groups hold at
least `S` bits: `y_S = min_g r_g max(1, S/b_g)` over the groups not sampled whole (and of nonzero bits, when `S > 0`),
and `h(∞)` when there is none. -/
def Form.heavy (F : Form) (L : Ln2) (gs : List Group) (P : Params) : Option Rat :=
  match F with
  | .sets => none
  | .patterns S =>
    if (gs.map fun g => g.n * g.b).sum < S then none
    else
      let ys := gs.filterMap fun g =>
        if g.b = 0 ∧ 0 < S then none else g.rate.map fun r => r * max 1 ((S : Rat) / (g.b : Rat))
      let cap := (S : Rat) * L.up
      match ys with
      | y :: ys => some (cap - P.θ * hLower P.q (ys.foldl min y) P.pθ)
      | [] => (lnPassWhole P.q P.θ P.p).map (cap + ·)

/-! ## The bound -/

/-- **At least `ln` of the form's factor**: `ln(1 + Σ e^T)` over the terms `T`, each at least the logarithm of a sum
over the nonempty sets: one per chord, the tail past the last breakpoint, the sets meeting a cell sampled whole, and the
patterns' heavy sets. `none` when the parameters are not valid. -/
def lnFactorUpperOf (F : Form) (cs : List Cell) (P : Params) : Option Rat :=
  if P.valid then
    let L := Ln2.of P.p
    let gs := groupsOf cs F.served P.p
    let fin := gs.filterMap fun g => g.rate.map (g, ·)
    let inf := gs.filter fun g => g.rate.isNone
    let chordTerms := (chordsOf P.q P.pθ P.ys).filterMap fun c =>
      (F.light L (fin.map fun x => (x.1, -(P.θ * c.1 * x.2))) P.p).map (-(P.θ * c.2) + ·)
    let tail := match P.ys.getLast? with
      | some y => (F.light L (fin.map fun x => (x.1, 0)) P.p).map (-(P.θ * hLower P.q y P.pθ) + ·)
      | none => none
    let whole := if inf.isEmpty then none
      else oadd (lnPassWhole P.q P.θ P.p) (F.whole L (fin.map Prod.fst) inf P.p)
    let terms := chordTerms ++ [tail, whole, F.heavy L gs P].filterMap id
    lse (0 :: terms) P.p
  else none

/-- **At least `ln G_ρ(θ)`**, Rankin's factor of the replay block with cells `cs` (`Claims.Sampling.rankinFactor`). -/
def lnFactorUpper (cs : List Cell) (P : Params) : Option Rat := lnFactorUpperOf .sets cs P

/-- **At least `ln` of the patterns' factor** of the replay block with cells `cs`, its proof blocks read by at most one
request, serving `S` bits (`Claims.Sampling.rankinFactorPatterns`). -/
def lnFactorUpperPatterns (cs : List Cell) (S : Nat) (P : Params) : Option Rat := lnFactorUpperOf (.patterns S) cs P

end SampledProofs.FactorUpper
