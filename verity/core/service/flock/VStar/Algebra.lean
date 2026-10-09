import Std.Data.HashMap
import Flock.Schedule
import VStar.InnerRepCheck

/-!
# V*'s algebra: `GfResiduals`' S of a rep and `InnerRepCheck`'s (`verity_flock.rec_algebra.structure`, `rec_residuals.spec`)

`rec_algebra.run` is the inner verifier's algebra on one rep, written once over values; on symbolic messages it yields each
residual as `Σ t[k]·k + c + Σ q[(a, b)]·a·b`, and `_build` reads `GfResiduals`' S from which terms, products and constants
each residual has, never from their coefficients (a term stays as a structural slot when its coefficient is zero). So the
structure is a function of the shape alone, and `run` here tracks only that support: a value is `none` (a Python int, which
the symbolic run only ever combines with a message as zero) or a `Sym`, and every operation of `run` (`^`, `sm`, `kc`,
`kf`, `fk`, `kadd`, `cst`, `mv`) is a union of supports. Terms are named by their `_order` index: the message elements in
`rounds`' order, then the running sums (`ACC_NAMES`), then each claim's ring-switch transpose `rs.t` (128 a claim), then
the comb (64). The consistency sums (`cons`) are kept apart, as `_build` reads them apart.

`run`'s refusals on a shape are kept: no `fast100` schedule (`m ∉ [22, 35]`), fewer than two claims, and the Ligerito
dimension checks. Below `k_log = 6` Python's `run` reads past the first claim's point (an `IndexError` in
`basis_for_claim`); that is refused too.
-/

namespace VStar.Algebra

open Flock (Schedule)

/-- `rec_algebra.Shape` without the Merkle widths, which the structure doesn't read. -/
structure Shape where
  m : Nat
  kLog : Nat
  claims : Nat
  deriving Repr, Inhabited

/-- A message element or running sum as `rec_algebra.rounds` and `ACC_NAMES` name it. -/
inductive Name where
  | ab (i : Nat) | c (i : Nat) | g1 (i : Nat) | ginf (i : Nat) | fa | fb | q1 (j : Nat) | qinf (j : Nat) | zp (l : Nat)
  | rs (k v : Nat) | T | ood (i : Nat) | msg (i h c : Nat) | yr (j : Nat) | acc0 (r : Nat) | acc1 (r : Nat)
  deriving BEq, Hashable, Repr, Inhabited

/-- The shape's schedule, or `run`'s refusal of it. -/
def Shape.sched (sh : Shape) : Except String Schedule := do
  let some s := Flock.fast100 sh.m | throw s!"m = {sh.m}: no fast100 schedule"
  if sh.kLog < 6 then throw s!"k_log = {sh.kLog}: below the skip's 6 variables (rec_algebra.run reads past the first claim's point)"
  if sh.claims < 2 then throw s!"{sh.claims} claims, {2 + (sh.claims - 2)} given"
  pure s

def msgNames (i : Nat) : Array Name := #[.msg i 0 0, .msg i 0 1, .msg i 1 0, .msg i 1 1]

/-- `rec_algebra.rounds`: the rep's message elements by the round that absorbs them, in protocol order. -/
def rounds (sh : Shape) (s : Schedule) : Array (Array Name) := Id.run do
  let R := s.levels.size - 1
  let mut out : Array (Array Name) := #[(Array.range 64).map Name.ab ++ (Array.range 64).map Name.c]
  for i in [0:sh.m - 6] do out := out.push #[.g1 i, .ginf i]
  out := out.push #[.fa, .fb]
  for j in [0:sh.kLog - 6] do out := out.push #[.q1 j, .qinf j]
  out := out.push ((Array.range 64).map Name.zp)
  for k in [0:sh.claims] do out := out.push ((Array.range 128).map (Name.rs k))
  let mut mi := 0
  out := out.push #[.T]
  out := out.push #[.ood 0]
  let mut oi := 1
  for _ in [0:s.k0 + 1] do
    out := out.push (msgNames mi)
    mi := mi + 1
  for _ in [0:2] do
    out := out.push (#[.ood oi] ++ msgNames mi)
    oi := oi + 1
    mi := mi + 1
  out := out.push (msgNames mi)
  mi := mi + 1
  for i in [0:R] do
    if i + 1 < R then
      for _ in [0:4] do
        out := out.push (msgNames mi)
        mi := mi + 1
      for _ in [0:2] do
        out := out.push (#[.ood oi] ++ msgNames mi)
        oi := oi + 1
        mi := mi + 1
      out := out.push (msgNames mi)
      mi := mi + 1
    else
      for _ in [0:3] do
        out := out.push (msgNames mi)
        mi := mi + 1
      out := out.push (msgNames mi ++ (Array.range (2 <<< s.ext)).map Name.yr)
      mi := mi + 1
  return out

def ACC_NAMES : Array Name := #[.acc0 0, .acc0 1, .acc1 0, .acc1 1]

/-! ## Supports -/

/-- The union of two ascending arrays without repeats. -/
def union {α : Type} [Ord α] [Inhabited α] (a b : Array α) : Array α := Id.run do
  let mut out := Array.mkEmpty (a.size + b.size)
  let mut i := 0
  let mut j := 0
  for _ in [0:a.size + b.size] do
    if i < a.size then
      if j < b.size then
        match compare a[i]! b[j]! with
        | .lt =>
          out := out.push a[i]!
          i := i + 1
        | .gt =>
          out := out.push b[j]!
          j := j + 1
        | .eq =>
          out := out.push a[i]!
          i := i + 1
          j := j + 1
      else
        out := out.push a[i]!
        i := i + 1
    else if j < b.size then
      out := out.push b[j]!
      j := j + 1
  return out

instance instOrdNatPair : Ord (Nat × Nat) := lexOrd

/-- A symbolic value's support: its terms (by `_order` index, ascending), its consistency sums `(level, lane)`, its
products of messages `(a, b)` (by `_order` index) and whether it has a constant. -/
structure Sym where
  t : Array Nat := #[]
  cons : Array (Nat × Nat) := #[]
  q : Array (Nat × Nat) := #[]
  c : Bool := false
  deriving Inhabited, BEq

/-- A value of `run`: `none` is a Python int. -/
abbrev Val := Option Sym

/-- `^` (and `sm`'s scaling, which keeps the support). -/
def vx : Val → Val → Val
  | none, y => y
  | some a, none => some a
  | some a, some b => some { t := union a.t b.t, cons := union a.cons b.cons, q := union a.q b.q, c := a.c || b.c }

def vxAll (xs : Array Val) : Val := xs.foldl vx none

/-- A value of K, `(c0, c1)`. -/
abbrev KV := Val × Val

def kadd (a b : KV) : KV := (vx a.1 b.1, vx a.2 b.2)

/-- `kc`: a constant of K times a K value; both components read both of the value's. -/
def kc (x : KV) : KV := (vx x.1 x.2, vx x.1 x.2)

/-! ## `run` on supports -/

/-- The shape, its schedule and the `_order` index of every name. -/
structure Ctx where
  sh : Shape
  s : Schedule
  rounds : Array (Array Name)
  idx : Std.HashMap Name Nat
  /-- the names': `rounds`' elements and `ACC_NAMES` -/
  base : Nat

def Ctx.make (sh : Shape) : Except String Ctx := do
  let s ← sh.sched
  let rds := VStar.Algebra.rounds sh s
  let names := rds.flatten ++ ACC_NAMES
  let mut idx : Std.HashMap Name Nat := {}
  for i in [0:names.size] do idx := idx.insert names[i]! i
  pure { sh, s, rounds := rds, idx, base := names.size }

def Ctx.ix (cx : Ctx) (n : Name) : Except String Nat :=
  match cx.idx[n]? with
  | some i => pure i
  | none => throw s!"rec_algebra: no element {repr n} in the rep's rounds"

/-- `rs.t[k][u]`'s index. -/
def Ctx.rst (cx : Ctx) (k u : Nat) : Nat := cx.base + 128 * k + u

/-- `comb[l]`'s index. -/
def Ctx.comb (cx : Ctx) (l : Nat) : Nat := cx.base + 128 * cx.sh.claims + l

def Ctx.var (cx : Ctx) (n : Name) : Except String Val := do pure (some { t := #[← cx.ix n] })

def Ctx.vars (cx : Ctx) (ns : Array Name) : Except String Val := do pure (vxAll (← ns.mapM cx.var))

/-- Ligerito's running state: the sum `s`, the quadratic `q`, how many messages and OOD values it has read, the OOD
contexts' `(dimension, split)` and the consistency contexts' `(columns, start)`. -/
structure LState where
  s : KV := (none, none)
  q : KV × KV × KV := ((none, none), (none, none), (none, none))
  mi : Nat := 0
  oi : Nat := 0
  oods : Array (Nat × Option Nat) := #[]
  conss : Array (Nat × Nat) := #[]

abbrev LM := StateT LState (Except String)

/-- The schedule's message and OOD counts: `k0 + 8 + 7 (R - 1)` and `1 + 2 R`. -/
def Ctx.counts (cx : Ctx) : Nat × Nat :=
  let R := cx.s.levels.size - 1
  (cx.s.k0 + 8 + 7 * (R - 1), 1 + 2 * R)

def nextMsg (cx : Ctx) : LM (KV × KV) := do
  let st ← get
  if st.mi ≥ cx.counts.1 then throw "ligerito: the messages run out"
  set { st with mi := st.mi + 1 }
  let i := st.mi
  pure ((← cx.var (.msg i 0 0), ← cx.var (.msg i 0 1)), (← cx.var (.msg i 1 0), ← cx.var (.msg i 1 1)))

def nextOod (cx : Ctx) : LM Val := do
  let st ← get
  if st.oi ≥ cx.counts.2 then throw "ligerito: the OOD values run out"
  set { st with oi := st.oi + 1 }
  cx.var (.ood st.oi)

def ofMsg (mm : KV × KV) (s : KV) : KV × KV × KV := (mm.1, kadd s mm.2, mm.2)

def challenge (cx : Ctx) : LM Unit := do
  let (a0, a1, a2) := (← get).q
  modify fun st => { st with s := kadd (kadd a0 (kc a1)) (kc a2) }
  let mm ← nextMsg cx
  modify fun st => { st with q := ofMsg mm st.s }

def addScaled (p : KV × KV × KV) : LM Unit :=
  modify fun st => { st with q := (kadd st.q.1 p.1, kadd st.q.2.1 p.2.1, kadd st.q.2.2 p.2.2) }

def oodIntro (cx : Ctx) (dim split : Nat) : LM Unit := do
  let y ← nextOod cx
  let mm ← nextMsg cx
  addScaled (ofMsg mm (y, none))
  modify fun st => { st with s := kadd st.s (y, none), oods := st.oods.push (dim, some split) }

/-- `query`'s consistency term: the level's sums over its lanes, both components (`kf` of each). -/
def query (cx : Ctx) (li foldLen : Nat) : Except String KV := do
  let some lv := cx.s.levels[li]? | throw s!"ligerito: no level {li}"
  if 2 ^ foldLen != 2 ^ lv.lanes then throw s!"ligerito: a fold of {2 ^ foldLen} for {2 ^ lv.lanes} lanes"
  let e : Val := some { cons := (Array.range (2 ^ lv.lanes)).map (li, ·) }
  pure (e, e)

/-- `run`'s Ligerito section from `t_in`, then `final_weights`' and `run`'s dimension checks: the final sum `s`. -/
def ligerito (cx : Ctx) (tIn : Val) : LM KV := do
  let s := cx.s
  let R := s.levels.size - 1
  modify fun st => { st with s := (tIn, none) }
  let y ← nextOod cx
  modify fun st => { st with s := kadd st.s (y, none), oods := st.oods.push (s.n, none) }
  let mm ← nextMsg cx
  modify fun st => { st with q := ofMsg mm st.s }
  for _ in [0:s.k0] do challenge cx
  let mut split := s.n - s.k0 + 1
  oodIntro cx split 0
  oodIntro cx split 0
  let e ← query cx 0 s.k0
  let mm ← nextMsg cx
  addScaled (ofMsg mm e)
  modify fun st => { st with s := kadd st.s e, conss := st.conss.push (s.n - s.k0, 0) }
  let mut ext := 0
  for i in [0:R] do
    if split < 4 then throw "ligerito: split_dim < k"
    for _ in [0:4] do challenge cx
    ext := split - 4
    if i + 1 < R then
      split := ext + 1
      oodIntro cx split (i + 1)
      oodIntro cx split (i + 1)
      let e ← query cx (i + 1) 4
      let mm ← nextMsg cx
      addScaled (ofMsg mm e)
      modify fun st => { st with s := kadd st.s e, conss := st.conss.push (ext, i + 1) }
    else
      let e ← query cx R 4
      modify fun st => { st with s := kadd st.s e, conss := st.conss.push (ext, R) }
  let st ← get
  if st.mi != cx.counts.1 || st.oi != cx.counts.2 then throw "ligerito: leftover messages"
  let orig := s.k0 + 3 * R
  if orig + ext != s.n then throw "ligerito: |orig| + ext != n"
  for (dim, sp) in st.oods do
    let fixed ← match sp with
      | none => pure orig
      | some sp => if sp < R then pure (4 + 3 * (R - 1 - sp)) else throw s!"ligerito: no level {sp}"
    if dim != fixed + ext then throw "ligerito: ood dimension"
  for (cols, start) in st.conss do
    if 3 * (R - start) + ext != cols then throw "ligerito: consistency dimension"
  if 2 <<< s.ext != 2 <<< ext then throw "ligerito: |yr|"
  pure st.s

/-- `run`'s residuals on symbolic messages, as supports: zerocheck, lincheck, one per claim (its ring switch), Ligerito's
final check (c0, c1), and the absorbed T minus the batched sum. -/
def supports (cx : Ctx) : Except String (Array Sym) := do
  let sh := cx.sh
  let pc ← cx.vars ((Array.range 64).map Name.c)
  let mut claim := vx (← cx.vars ((Array.range 64).map Name.ab ++ (Array.range 64).map Name.c)) pc
  for i in [0:sh.m - 6] do
    let g1 ← cx.var (.g1 i)
    let ginf ← cx.var (.ginf i)
    claim := vx (vx (vx claim g1) g1) ginf
  let mut out : Array Val := #[vx claim (some { q := #[(← cx.ix .fa, ← cx.ix .fb)] })]
  let mut lc := vx (vx (← cx.var .fa) (← cx.var .fb)) (some { c := true })
  for j in [0:sh.kLog - 6] do
    let q1 ← cx.var (.q1 j)
    let qinf ← cx.var (.qinf j)
    let e0 := vx lc q1
    lc := vx (vx qinf (vx (vx e0 q1) qinf)) e0
  let mut acc : Val := none
  for l in [0:64] do acc := vx acc (some { q := #[(cx.comb l, ← cx.ix (.zp l))] })
  out := out.push (vx acc lc)
  let w ← cx.vars ((Array.range 64).map Name.zp)
  let values : Array Val := #[w, pc] ++ Array.replicate (sh.claims - 2) (some { c := true })
  let mut T : Val := none
  for k in [0:sh.claims] do
    out := out.push (vx (← cx.vars ((Array.range 128).map (Name.rs k))) (values.getD k none))
    T := vx T (some { t := (Array.range 128).map (cx.rst k) })
  let tIn ← cx.var .T
  let (s, _) ← (ligerito cx tIn).run {}
  let yr ← cx.vars ((Array.range (2 <<< cx.s.ext)).map Name.yr)
  out := out ++ #[vx yr s.1, vx yr s.2, vx tIn T]
  pure (out.map (·.getD {}))

/-! ## `_build` -/

structure BState where
  ops : Array (List Term) := #[]
  opOf : Std.HashMap Nat Nat := {}
  /-- the comb's operands, `(operand, l)`: placed after every coefficient once the coefficients are counted -/
  combOps : Array (Nat × Nat) := #[]
  nv : Nat := 0

abbrev BM := StateT BState (Except String)

/-- `_build`'s `var_op`: the operand of a message element, a ring switch's transpose or the comb, made on first use. -/
def varOp (cx : Ctx) (k : Nat) : BM Nat := do
  if let some i := (← get).opOf[k]? then return i
  let i := (← get).ops.size
  let t ← if k < cx.base then pure (Term.w k)
    else if k < cx.rst cx.sh.claims 0 then do
      pure (Term.wt (← cx.ix (.rs ((k - cx.base) / 128) 0)) ((k - cx.base) % 128))
    else if k < cx.comb 64 then do
      modify fun st => { st with combOps := st.combOps.push (i, k - cx.comb 0) }
      pure (Term.v 0)
    else throw s!"rec_algebra: no element of index {k}"
  modify fun st => { st with ops := st.ops.push [t], opOf := st.opOf.insert k i }
  pure i

/-- One residual's products and added terms, its coefficients' and constant's v slots taken in order. -/
def buildRes (cx : Ctx) (ri : Nat) (f : Sym) : BM Res := do
  let fin := 2 + cx.sh.claims
  let mut add : Array Term := #[]
  let mut mulp : Array (Nat × Nat) := #[]
  if !f.cons.isEmpty then
    unless ri == fin || ri == fin + 1 do throw s!"residual {ri}: a consistency sum outside Ligerito's final check"
    add := add ++ #[.w (cx.base - 2 + (ri - fin)), .w (cx.base - 4 + (ri - fin))]
  for k in f.t do
    let st ← get
    set { st with ops := st.ops.push [.v st.nv], nv := st.nv + 1 }
    let a := st.ops.size
    mulp := mulp.push (a, ← varOp cx k)
  for (a, b) in f.q do
    let oa ← varOp cx a
    let ob ← varOp cx b
    mulp := mulp.push (oa, ob)
  if f.c then
    let st ← get
    set { st with nv := st.nv + 1 }
    add := add.push (.v st.nv)
  pure { mul := mulp.toList, add := add.toList }

/-- `GfResiduals`' S from the residuals' supports (`_build`). -/
def build (cx : Ctx) (lins : Array Sym) : Except String GfS := do
  let go : BM (Array Res) := do
    let mut res := #[]
    for ri in [0:lins.size] do res := res.push (← buildRes cx ri lins[ri]!)
    pure res
  let (res, st) ← go.run {}
  let mut ops := st.ops
  for (i, l) in st.combOps do ops := ops.set! i [.v (st.nv + l)]
  pure { nw := cx.base, nv := st.nv + 64, ops := ops.toList, res := res.toList }

/-- `rec_algebra.structure(sh)`'s S. -/
def structure_ (sh : Shape) : Except String GfS := do
  let cx ← Ctx.make sh
  build cx (← supports cx)

/-- `rec_residuals.spec(sh)`: the structure, a row port per round that absorbs elements, and the running sums' two. -/
def spec (sh : Shape) : Except String RepS := do
  let cx ← Ctx.make sh
  let gf ← build cx (← supports cx)
  let mut ports : Array (String × Nat) := #[]
  for i in [0:cx.rounds.size] do ports := ports.push (s!"m{i}", cx.rounds[i]!.size)
  let S : RepS := { gf, ports := ports.toList ++ [("acc0", 2), ("acc1", 2)] }
  checkRepS S (whole := true)
  pure S

end VStar.Algebra
