import Flock.Ligerito

/-!
# V*'s algebra: `GfResiduals`' structure and its verifier rows

V*'s algebra unit, `InnerRepCheck_v1{S}`, computes `GfResiduals_v1{S.gf}` on the rep's message elements (`w`) and the
verifier's coefficients (`v`): one residual per algebraic check of the inner verifier, zero exactly when the check
holds. This file restates how Python builds that unit's `S` and `v` (`verity_flock.rec_algebra`, at rec-step2
`b6139d9b6`; the citations `rec_algebra.py:N` are its lines), operation for operation, and `GfResiduals`' semantics
(`gf2k.residuals`).

* `run` is the inner verifier's algebra written once over values (`Val`): over `F128` it is the reference, over the
  symbolic values `Lin` it yields each residual's coefficients. The coin-only functions it calls (`lagrange`,
  `combWeights`, `eqTable`, `positions`, `basis`, `svTable`, `whats`, `cff`) are the Lean verifier's, which Python's
  restate; `finalWeights` is the verifier's §13.6 loop (`Flock.ligerito`) as a function.
* `build` reads `GfResiduals`' `S`, the coefficient slots' values and the final check's coefficients on the consistency
  sums (`kappa`) off the symbolic residuals (`rec_algebra._build`).
* `structureOf sh` is `S` at coins all zero (`rec_algebra.structure`: the structure depends on the rounds the verifier
  draws, not on the coins); `verifierRows` is `v` and the coin-derived points at the rep's coins, and refuses when the
  structure there is not `structureOf sh` (`rec_algebra.verifier_rows`).
* `residuals` is `GfResiduals`' semantics: per residual, its added terms and the products of its operand pairs.

The verifier computes the S of each `rec-residuals/…` unit it checks with `structureOf` (`Flock.RecResiduals`).
-/

namespace Flock.RecAlgebra

open Flock

/-! ## Coins -/

/-- Where the verifier's coins come from (`rec_algebra.CoinSource`, rec_algebra.py:229–249): `round k n` is round `k`'s coins
when the verifier draws `n` of them, `drawnAll k` that every recorded round is drawn once `k` are. -/
structure CoinSource where
  round : Nat → Nat → Except String (Array F128)
  drawnAll : Nat → Bool

/-- A session record's coins by round (`CoinSource(rounds)`, rec_algebra.py:232–242). A recorded round of no coins is no
draw (§6.2), so it is skipped (`CoinSource.__init__`): z_partial's round in a holo session absorbs elements and draws none. -/
def CoinSource.ofRounds (recorded : Array (Array F128)) : CoinSource :=
  let rounds := recorded.filter (·.size ≠ 0)
  { round := fun k n => match rounds[k]? with
      | some r => if r.size = n then .ok r else .error s!"round {k}: {n} coins drawn, the record has {r.size}"
      | none => .error s!"round {k}: {n} coins drawn, the record has no round"
    drawnAll := fun k => k == rounds.size }

/-- Coins in whatever rounds the verifier draws, all zero (`_dummy_coins`, rec_algebra.py:884–901, whose coins are
random: the structure depends on the rounds, not on the coins). -/
def CoinSource.zeros : CoinSource := ⟨fun _ n => .ok (Array.replicate n 0), fun _ => true⟩

/-- The coins drawn so far, read from a source. -/
abbrev CM := StateT Nat (ReaderT CoinSource (Except String))

/-- `CoinSource.vec` (rec_algebra.py:235–242): a draw of `n` coins is one round of `n`; none is no round (§6.2). -/
def coinVec (n : Nat) : CM (Array F128) := do
  if n = 0 then return #[]
  let src ← read
  let k ← get
  match src.round k n with
  | .ok r => set (k + 1); return r
  | .error e => throw e

/-- A computation that can fail, in `CM`. -/
def liftE {A : Type} : Except String A → CM A
  | .ok a => pure a
  | .error e => throw e

/-- `CoinSource.one` (rec_algebra.py:244–245). -/
def coinOne : CM F128 := do return (← coinVec 1)[0]!

/-- `CoinSource.k2` (rec_algebra.py:247–249): two coins as a K element. -/
def coinK2 : CM F256 := do
  let v ← coinVec 2
  return ⟨v[0]!, v[1]!⟩

/-! ## Values -/

/-- A message element's name (`rec_algebra.rounds`, `symbolic`, `ACC_NAMES`; rec_algebra.py:743–809): zerocheck's
`round1_ab`, `round1_c`, rounds and `v_A, v_B`; lincheck's rounds and `z_partial`; a holo shape's `h`; each claim's ring
switch; Ligerito's `T`, OOD values, messages `(u0, u2)` and `yr`; the running sums before and after the rep's openings; and
the symbolic values w does not hold: each ring switch's transpose, the comb (in a holo shape, the operand `comb[l] + h[l]`,
Python's `("combh", l)`) and the levels' consistency sums. -/
inductive Key where
  | ab (i : Nat)
  | c (i : Nat)
  | g1 (i : Nat)
  | ginf (i : Nat)
  | fa
  | fb
  | q1 (j : Nat)
  | qinf (j : Nat)
  | zp (l : Nat)
  | h (l : Nat)
  | rs (k v : Nat)
  | T
  | ood (i : Nat)
  | msg (i h c : Nat)
  | yr (j : Nat)
  | acc0 (r : Nat)
  | acc1 (r : Nat)
  | rsT (k u : Nat)
  | comb (l : Nat)
  | cons (level l : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- A dict of coefficients, in insertion order. -/
abbrev Terms (κ : Type) := List (κ × F128)

/-- `t[k] = t.get(k, 0) ^ v` (rec_algebra.py:487–488): a key's coefficient plus `v`, the key appended when new. -/
def addAt {κ : Type} [DecidableEq κ] : Terms κ → κ → F128 → Terms κ
  | [], k, v => [(k, v)]
  | (k', x) :: r, k, v => if k' = k then (k', x + v) :: r else (k', x) :: addAt r k v

/-- Two dicts' coefficients summed key by key (rec_algebra.py:486–491). -/
def mergeT {κ : Type} [DecidableEq κ] (a b : Terms κ) : Terms κ := b.foldl (fun acc kv => addAt acc kv.1 kv.2) a

/-- A symbolic F value (`rec_algebra.Lin`, rec_algebra.py:470–499): `Σ t[k]·k + c + Σ q[(a, b)]·a·b` over message
elements. `c` is `none` when the value has no constant term; a term or product keeps its slot when its coefficient is
zero. -/
structure Lin where
  t : Terms Key := []
  c : Option F128 := none
  q : Terms (Key × Key) := []
  deriving Inhabited, Repr

namespace Lin

/-- A message element (`var`, rec_algebra.py:792). -/
def var (k : Key) : Lin := { t := [(k, 1)] }

/-- `Lin.__xor__` (rec_algebra.py:481–493). -/
def xor (a b : Lin) : Lin where
  t := mergeT a.t b.t
  c := match a.c, b.c with
    | x, none => x
    | none, some y => some y
    | some x, some y => some (x + y)
  q := mergeT a.q b.q

/-- `Lin.scale` (rec_algebra.py:497–499). -/
def scale (s : F128) (a : Lin) : Lin where
  t := a.t.map fun kv => (kv.1, s * kv.2)
  c := a.c.map (s * ·)
  q := a.q.map fun kv => (kv.1, s * kv.2)

/-- `mv` on symbolic values (rec_algebra.py:507–514): two message elements, each alone with coefficient 1, give their
product. -/
def mv (x y : Lin) : Except String Lin :=
  match x, y with
  | ⟨[(kx, a)], none, []⟩, ⟨[(ky, b)], none, []⟩ =>
    if a = 1 ∧ b = 1 then .ok { q := [((kx, ky), 1)] } else .error "a product of two messages takes two messages"
  | _, _ => .error "a product of two messages takes two messages"

end Lin

/-- What `run` computes with (rec_algebra.py:468–543): Python's `0`, `^`, `sm` (a coin-derived constant times a value),
`mv` (a message times a message) and `cst` (a constant). -/
class Val (α : Type) where
  zero : α
  add : α → α → α
  sm : F128 → α → α
  mv : α → α → Except String α
  cst : F128 → α

/-- The reference: values are F's elements. -/
instance instValF128 : Val F128 := ⟨0, (· + ·), (· * ·), fun x y => .ok (x * y), id⟩

/-- Symbolic values (`symbolic`, `cst = lambda v: Lin(c=v)`, rec_algebra.py:582). -/
instance instValLin : Val Lin := ⟨{}, Lin.xor, Lin.scale, Lin.mv, fun v => { c := some v }⟩

open Val (zero add sm mv cst)

variable {α : Type} [Val α]

/-- `dot` (rec_algebra.py:517–521). -/
def dotV (cs : Array F128) (xs : Array α) : α := (cs.zip xs).foldl (fun acc cx => add acc (sm cx.1 cx.2)) zero

/-- A K value, `(c0, c1)`. -/
abbrev KV (α : Type) := α × α

/-- `kc` (rec_algebra.py:524–526): a constant of K times a K value, `(x0 c0 + x⁻¹ x1 c1, x0 c1 + x1 (c0 + c1))`. -/
def kc (c : F256) (x : KV α) : KV α :=
  (add (sm c.c0 x.1) (sm (F256.xinv * c.c1) x.2), add (sm c.c1 x.1) (sm (c.c0 + c.c1) x.2))

/-- `kf` (rec_algebra.py:529–531): a constant of K times an F value. -/
def kf (c : F256) (x : α) : KV α := (sm c.c0 x, sm c.c1 x)

/-- `fk` (rec_algebra.py:534–536): a constant of F times a K value. -/
def fk (b : F128) (x : KV α) : KV α := (sm b x.1, sm b x.2)

/-- `kadd` of two (rec_algebra.py:539–543). -/
def kadd (x y : KV α) : KV α := (add x.1 y.1, add x.2 y.2)

/-- `kadd(*xs)` (rec_algebra.py:539–543). -/
def kaddAll (xs : List (KV α)) : KV α := xs.foldl kadd (zero, zero)

/-- A K value with no `u` part (`(y, 0)`). -/
def kofF (y : α) : KV α := (y, zero)

/-! ## The rep's messages -/

/-- One rep's message elements as values (`rec_algebra.Messages`, rec_algebra.py:555–576). `rsT[k][u]` is claim `k`'s
ring switch's transpose `t[u]`; `cons level ea` the level's consistency sums `Σ_j ea[j] · row_j[l]`; `comb` the circuit
fold (in a holo shape, the public part's fold plus `h`, as `run` folds them); `tAtom` Ligerito's `T` as the prover absorbed it (`none`: the verifier's own, the batched sum). -/
structure Messages (α : Type) where
  ab : Array α
  c : Array α
  zc : Array (α × α)
  fa : α
  fb : α
  lc : Array (α × α)
  zp : Array α
  rs : Array (Array α)
  rsT : Array (Array α)
  ood : Array α
  msgs : Array (KV α × KV α)
  yr : Array α
  comb : Array α
  cons : Nat → Array F128 → Array α
  tAtom : Option α := none

/-- A claim handed to the opening (`rec_algebra.Claim`, rec_algebra.py:546–552): value, 64 skip weights, `m − 6`
coordinates. -/
structure AClaim (α : Type) where
  value : α
  skip : Array F128
  x : Array F128

/-! ## `run`: the inner verifier's algebra over values -/

/-- The zerocheck's coin-derived points (`info.update(z=, rho=, r=, v_c=)`, rec_algebra.py:600). -/
structure ZcPts (α : Type) where
  z : F128
  rho : Array F128
  r : Array F128
  vc : α

/-- The zerocheck's residual rounds (rec_algebra.py:593–598): from round `i`, the claim and the challenges so far. -/
def zcRounds (r : Array F128) : List (α × α) → Nat → α → Array F128 → CM (α × Array F128)
  | [], _, claim, rho => return (claim, rho)
  | (g1, ginf) :: rest, i, claim, rho => do
    let g0 := sm (1 + r[i]!).inv (add claim (sm r[i]! g1))
    let rr ← coinOne
    zcRounds r rest (i + 1) (add (add (sm (1 + rr) g0) (sm rr g1)) (sm (rr * (1 + rr)) ginf)) (rho.push rr)

/-- §9.2, the zerocheck (rec_algebra.py:587–600): its residual `claim + v_A·v_B`, and its points. -/
def zerocheck (m : Nat) (M : Messages α) : CM (α × ZcPts α) := do
  let _ ← coinVec 6
  let r := pinned ++ (← coinVec (m - 13))
  let z ← coinOne
  let pc := dotV (lagrange nodesL den6Inv z) M.c
  let claim := add (dotV (combWeights z) (M.ab.zipWith add M.c)) pc
  let (claim, rho) ← zcRounds r M.zc.toList 0 claim #[]
  let ab ← liftE (mv M.fa M.fb)
  return (add claim ab, ⟨z, rho, r, pc⟩)

/-- The lincheck's fold points (`fold=dict(...)`, rec_algebra.py:616): `α`, `β`, the skip weights at the zerocheck's
`z`, `ρ_in` and the round challenges. -/
structure FoldPts where
  alpha : F128
  beta : F128
  lw : Array F128
  rhoIn : Array F128
  ts : Array F128
  deriving Inhabited, Repr

/-- The lincheck's coin-derived points (rec_algebra.py:616–617). -/
structure LcPts (α : Type) where
  fold : FoldPts
  w : α
  rPrime : F128
  rIn : Array F128

/-- The lincheck's rounds (rec_algebra.py:605–609): the claim and the challenges so far. -/
def lcRounds : List (α × α) → α → Array F128 → CM (α × Array F128)
  | [], claim, ts => return (claim, ts)
  | (q1, qinf) :: rest, claim, ts => do
    let t ← coinOne
    let e0 := add claim q1
    lcRounds rest (add (add (sm (t * t) qinf) (sm t (add (add e0 q1) qinf))) e0) (ts.push t)

/-- `Σ_l comb[l] · z_partial[l]` (rec_algebra.py:610–612). -/
def combDot : List (α × α) → α → Except String α
  | [], acc => .ok acc
  | (cl, zl) :: rest, acc => do combDot rest (add acc (← mv cl zl))

/-- §10, the lincheck (rec_algebra.py:601–617): its residual `Σ_l comb[l]·z_partial[l] + claim`, and its points. -/
def lincheck (kLog : Nat) (M : Messages α) (zc : ZcPts α) : CM (α × LcPts α) := do
  let alpha ← coinOne
  let beta ← coinOne
  let (claim, ts) ← lcRounds M.lc.toList (add (add (sm alpha M.fa) M.fb) (cst beta)) #[]
  let acc ← liftE (combDot (M.comb.zip M.zp).toList zero)
  let rPrime ← coinOne
  let w := dotV (skipWeights rPrime) M.zp
  return (add acc claim, ⟨⟨alpha, beta, skipWeights zc.z, zc.rho.extract 0 (kLog - 6), ts⟩, w, rPrime, ts.reverse⟩)

/-- The claims handed to the opening (rec_algebra.py:621–622): the lincheck's, the zerocheck's (its value the one V*
takes for `v_C`) and the statement's extra claims, whose values are constants. -/
def claimsOf (m kLog : Nat) (zc : ZcPts α) (lc : LcPts α) (extras : Array Claim) : Array (AClaim α) :=
  #[⟨lc.w, skipWeights lc.rPrime, lc.rIn ++ zc.rho.extract (kLog - 6) (m - 6)⟩, ⟨zc.vc, skipWeights zc.z, zc.r⟩] ++
    extras.map fun e => ⟨cst e.value, e.skip, e.x⟩

/-- Ring switching's coin-derived points (rec_algebra.py:631–635): the batched sum `T`, the batching coins, and the
switched claims. -/
structure RsPts (α : Type) where
  T : α
  gamma : Array F128
  sws : Array Switched

/-- One claim's ring-switch residual and its `(β_k, r'', z)` (rec_algebra.py:624–630). The weights test the high half as
`Flock.ringSwitch` does (`v / 64 == 1`, Python's `v >= 64`). -/
def rsClaim (M : Messages α) (k : Nat) (cl : AClaim α) : CM (α × α × Array F128 × Array F128) := do
  let x0 := cl.x[0]!
  let wv := (Array.range 128).map fun v => cl.skip[v % 64]! * (if v / 64 == 1 then x0 else 1 + x0)
  let out := add (dotV wv M.rs[k]!) cl.value
  let rpp ← coinVec 7
  return (out, dotV (eqTable rpp) M.rsT[k]!, rpp, cl.x.extract 1 cl.x.size)

/-- The claims' ring switches from claim `k` (rec_algebra.py:624–630). -/
def rsClaims (M : Messages α) : List (AClaim α) → Nat → CM (Array α × Array α × Array (Array F128 × Array F128))
  | [], _ => return (#[], #[], #[])
  | cl :: rest, k => do
    let (out, beta, rpp, z) ← rsClaim M k cl
    let (outs, betas, pts) ← rsClaims M rest (k + 1)
    return (#[out] ++ outs, #[beta] ++ betas, #[(rpp, z)] ++ pts)

/-- §11, §12.2–§12.3, ring switching (rec_algebra.py:620–636): one residual per claim, and the batched sum. -/
def ringSwitch (M : Messages α) (claims : Array (AClaim α)) : CM (Array α × RsPts α) := do
  let (outs, betas, pts) ← rsClaims M claims.toList 0
  let gamma ← coinVec claims.size
  let sws := (Array.range claims.size).map fun k =>
    ({ gamma := gamma[k]!, rpp := pts[k]!.1, z := pts[k]!.2 } : Switched)
  return (outs, ⟨dotV gamma betas, gamma, sws⟩)

/-! ### Ligerito (§13) -/

/-- Ligerito's state (`st`, `msgs`, `oods`, `ood_ctx`, `cons_ctx`; rec_algebra.py:639–654): the running sum `s`, the
round polynomial `q`, the messages and OOD values read, and the OOD and consistency contexts. -/
structure LSt (α : Type) where
  s : KV α
  q : KV α × KV α × KV α
  msg : Nat := 0
  ood : Nat := 0
  oods : Array OodCtx := #[]
  conss : Array ConsCtx := #[]

abbrev LM (α : Type) := StateT (LSt α) CM

/-- `next(msgs)` (rec_algebra.py:639). -/
def nextMsg (M : Messages α) : LM α (KV α × KV α) := do
  let st ← get
  match M.msgs[st.msg]? with
  | some mm => set { st with msg := st.msg + 1 }; return mm
  | none => throw "ligerito: sumcheck messages exhausted"

/-- `next(oods)` (rec_algebra.py:639). -/
def nextOod (M : Messages α) : LM α α := do
  let st ← get
  match M.ood[st.ood]? with
  | some y => set { st with ood := st.ood + 1 }; return y
  | none => throw "ligerito: ood_values exhausted"

/-- `of_msg` (rec_algebra.py:642–643): the round polynomial of a message at the running sum. -/
def ofMsg (mm : KV α × KV α) (s : KV α) : KV α × KV α × KV α := (mm.1, kadd s mm.2, mm.2)

/-- `challenge` (rec_algebra.py:645–650). -/
def challenge (M : Messages α) : LM α F256 := do
  let rk ← coinK2
  let st ← get
  let (a0, a1, a2) := st.q
  set { st with s := kadd (kadd a0 (kc rk a1)) (kc (rk * rk) a2) }
  let mm ← nextMsg M
  modify fun st => { st with q := ofMsg mm st.s }
  return rk

/-- `[challenge() for _ in range(n)]`. -/
def challenges (M : Messages α) : Nat → LM α (Array F256)
  | 0 => return #[]
  | n + 1 => do
    let rk ← challenge M
    let rest ← challenges M n
    return #[rk] ++ rest

/-- `add_scaled` (rec_algebra.py:652–653). -/
def addScaled (p : KV α × KV α × KV α) (b : F128) : LM α Unit :=
  modify fun st => { st with q := (kadd st.q.1 (fk b p.1), kadd st.q.2.1 (fk b p.2.1), kadd st.q.2.2 (fk b p.2.2)) }

/-- `ood_intro` (rec_algebra.py:657–664). -/
def oodIntro (M : Messages α) (dim split : Nat) : LM α Unit := do
  let zz ← coinVec dim
  let y ← nextOod M
  let mm ← nextMsg M
  let b ← coinOne
  addScaled (ofMsg mm (kofF y)) b
  modify fun st => { st with s := kadd st.s (kofF (sm b y)), oods := st.oods.push ⟨zz, b, some split⟩ }

/-- `query` (rec_algebra.py:666–673): a level's positions, its consistency coins and its enforced sum
`e = Σ_l eq(fold)[l] · cons(level, eq(α))[l]`. Python draws `(q − 1).bit_length()` consistency coins, which is
`log2 (q − 1) + 1` for every `q ≥ 2`. -/
def query (sch : Schedule) (M : Messages α) (i : Nat) (fold : Array F256) : LM α (Array Nat × Array F128 × KV α) := do
  let lv := sch.levels[i]!
  let pos := positions lv (← coinVec lv.q)
  let al ← coinVec (Nat.log2 (lv.q - 1) + 1)
  let ea := (eqTable al).extract 0 lv.q
  let e := kaddAll (((eqTableK fold).zip (M.cons i ea)).toList.map fun fa => kf fa.1 fa.2)
  return (pos, al, e)

/-- A level's queries, its message and its batching coin, folded into the running sum and the round polynomial
(rec_algebra.py:685–690, 702–707). -/
def queryFold (sch : Schedule) (M : Messages α) (i cols : Nat) (fold : Array F256) : LM α Unit := do
  let (pos, al, e) ← query sch M i fold
  let mm ← nextMsg M
  let b ← coinOne
  addScaled (ofMsg mm e) b
  modify fun st => { st with s := kadd st.s (fk b e), conss := st.conss.push ⟨cols, pos, al, b, i⟩ }

/-- The levels after the first (rec_algebra.py:693–712), from level `i` with the split dimension `split`: each level's
four challenges, then its OOD samples and queries, or at the last level its final queries. -/
def levelsLoop (sch : Schedule) (M : Messages α) (R : Nat) :
    Nat → Nat → Nat → Array (Array F256) → Nat → LM α (Array (Array F256) × Nat)
  | 0, _, _, levels, ext => return (levels, ext)
  | fuel + 1, i, split, levels, _ => do
    if split < 4 then throw "ligerito: split_dim < k"
    let chs ← challenges M 4
    let levels := levels.push chs
    let ext := split - 4
    if i + 1 < R then
      oodIntro M (ext + 1) (i + 1)
      oodIntro M (ext + 1) (i + 1)
      queryFold sch M (i + 1) ext chs
      levelsLoop sch M R fuel (i + 1) (ext + 1) levels ext
    else
      let (pos, al, e) ← query sch M R chs
      let b ← coinOne
      modify fun st => { st with s := kadd st.s (fk b e), conss := st.conss.push ⟨ext, pos, al, b, R⟩ }
      levelsLoop sch M R fuel (i + 1) split levels ext

/-- Ligerito up to its final check (rec_algebra.py:674–712): the level-0 OOD sample, the initial challenges, level 0's
OOD samples and queries, then the levels. Returns the initial challenges, the levels' challenges and `ext`. -/
def ligeritoRun (sch : Schedule) (M : Messages α) : LM α (Array F256 × Array (Array F256) × Nat) := do
  let n := sch.n
  let zz ← coinVec n
  let y ← nextOod M
  let b ← coinOne
  modify fun st => { st with s := kadd st.s (kofF (sm b y)), oods := st.oods.push ⟨zz, b, none⟩ }
  let mm ← nextMsg M
  modify fun st => { st with q := ofMsg mm st.s }
  let initCh ← challenges M sch.k0
  let split := n - sch.k0 + 1
  oodIntro M split 0
  oodIntro M split 0
  queryFold sch M 0 (n - sch.k0) initCh
  let (levels, ext) ← levelsLoop sch M (sch.levels.size - 1) (sch.levels.size - 1) 0 split #[] 0
  return (initCh, levels, ext)

/-- §13.6's `res[y]` for `y < 2^ext` (`rec_algebra.final_weights`, rec_algebra.py:395–455): the batched basis at the
folded point, the OOD terms and the consistency terms, all functions of the coins and the claims' points. This is the
verifier's final-check loop (`Flock.ligerito`) as a function, which Python's restates; like the verifier it takes the
levels loop's `ext`, where Python reads `sch.ext` (the same number: each level's columns are the last one's less 3). -/
def finalWeights (sch : Schedule) (ext : Nat) (initCh : Array F256) (levels : Array (Array F256)) (oods : Array OodCtx)
    (conss : Array ConsCtx) (sws : Array Switched) : Except String (Array F256) := do
  let n := sch.n
  let orig := levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh
  if orig.size + ext ≠ n then throw "ligerito: |orig| + ext != n"
  let heads := levels.map (fun l => l[0]!)
  let prodFrom := fun (a : Nat) => (heads.extract a heads.size).foldl (fun acc c => acc * cff c) (1 : F256)
  let pi0 := prodFrom 0
  let bres := basis sws orig ext
  if bres.size ≠ 2 ^ ext then throw "ligerito: basis residual length"
  let res := bres.map (· * pi0)
  let res ← oods.foldlM (init := res) fun res o => do
    let (fixed, scale) := match o.split with
      | none => (orig, pi0)
      | some s => (levels.extract s levels.size |>.foldl
          (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFrom (s + 1))
    if o.z.size ≠ fixed.size + ext then throw "ligerito: ood dimension"
    let head := (Array.range fixed.size).foldl (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fixed[t]!))
      (1 : F256)
    let c := (head * scale).smul o.beta
    return (Array.range (2 ^ ext)).map fun y =>
      let tail := (Array.range ext).foldl (fun acc j =>
        let zj := o.z[fixed.size + j]!
        acc * (if (y >>> j) % 2 == 1 then zj else (1 : F128) + zj)) (1 : F128)
      res[y]! + c.smul tail
  let sv := svTable (n + 1)
  conss.foldlM (init := res) fun res cs => do
    let fixed := (levels.extract cs.start levels.size).foldl (fun acc l => acc ++ l.extract 1 l.size) #[]
    if fixed.size + ext ≠ cs.cols then throw "ligerito: consistency dimension"
    let ea := eqTable cs.alpha
    let scale := prodFrom cs.start
    let add := (Array.range cs.pos.size).foldl (init := Array.replicate (2 ^ ext) (0 : F256)) fun add j =>
      let wh := whats sv cs.pos[j]! cs.cols
      let pj := (Array.range fixed.size).foldl
        (fun acc i => acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + wh[i]!))) (1 : F256)
      let coef := pj.smul ea[j]!
      (Array.range (2 ^ ext)).map fun y =>
        let tj := (Array.range ext).foldl (fun acc i => if (y >>> i) % 2 == 1 then acc * wh[fixed.size + i]! else acc)
          (1 : F128)
        add[y]! + coef.smul tj
    return (Array.range (2 ^ ext)).map fun y => res[y]! + (add[y]! * scale).smul cs.beta

/-- Ligerito's coin-derived points (`info.update(levels=, init_ch=, res=, queries=)`, rec_algebra.py:722): the
challenges, the final weights, and each level's query positions and `eq(α)` over them. -/
structure LgPts where
  initCh : Array F256
  levels : Array (Array F256)
  res : Array F256
  queries : Array (Array Nat × Array F128)
  oods : Array OodCtx
  conss : Array ConsCtx

/-- §13 Ligerito (rec_algebra.py:637–721): from the running sum `(T, 0)` (`tIn`: the absorbed `T`, or the verifier's
own), the final check's two residuals (K's `c0`, `c1`), and its points. -/
def ligerito (sch : Schedule) (M : Messages α) (tIn : α) (sws : Array Switched) : CM (Array α × LgPts) := do
  let ((initCh, levels, ext), st) ←
    (ligeritoRun sch M).run { s := kofF tIn, q := ((zero, zero), (zero, zero), (zero, zero)) }
  if !(← read).drawnAll (← get) then throw "recorded rounds were never drawn"
  if st.msg ≠ M.msgs.size ∨ st.ood ≠ M.ood.size then throw "ligerito: leftover messages"
  let res ← liftE (finalWeights sch ext initCh levels st.oods st.conss sws)
  if M.yr.size ≠ 2 <<< ext then throw "ligerito: |yr|"
  let lhs := kaddAll ((Array.range res.size).toList.map fun j =>
    kc res[j]! (M.yr.getD (2 * j) zero, M.yr.getD (2 * j + 1) zero))
  let fin := kadd lhs st.s
  return (#[fin.1, fin.2],
    ⟨initCh, levels, res, st.conss.map (fun c => (c.pos, (eqTable c.alpha).extract 0 c.pos.size)), st.oods, st.conss⟩)

/-- The coin-derived points of a run (`run`'s info, rec_algebra.py:585–722). -/
structure Info (α : Type) where
  zc : ZcPts α
  lc : LcPts α
  rs : RsPts α
  lg : LgPts

/-- `run` (rec_algebra.py:579–723): the rep's residuals (zerocheck, lincheck, one per claim, Ligerito's `c0` and `c1`,
the absorbed `T` minus the batched sum) and the coin-derived points beside them. -/
def run (sh : Shape) (M : Messages α) (extras : Array Claim) : CM (Array α × Info α) := do
  if sh.claims ≠ 2 + extras.size then throw s!"{sh.claims} claims, {2 + extras.size} given"
  let (r0, zc) ← zerocheck sh.m M
  let (r1, lc) ← lincheck sh.kLog M zc
  let (rsr, rs) ← ringSwitch M (claimsOf sh.m sh.kLog zc lc extras)
  let tIn := M.tAtom.getD rs.T
  let (lgr, lg) ← ligerito sh.sched M tIn rs.sws
  return (#[r0, r1] ++ rsr ++ lgr ++ #[add tIn rs.T], ⟨zc, lc, rs, lg⟩)

/-- `fold_points` (rec_algebra.py:924–928): the lincheck's fold points, which the coins decide before any message or
extra claim is read. -/
def foldPoints (sh : Shape) (M : Messages α) : CM FoldPts := do
  let (_, zc) ← zerocheck sh.m M
  let (_, lc) ← lincheck sh.kLog M zc
  return lc.fold

/-! ## The rounds, w and the symbolic messages -/

/-- A message's four elements (`msg()`, rec_algebra.py:760–762). -/
def msgNames (i : Nat) : Array Key := #[.msg i 0 0, .msg i 0 1, .msg i 1 0, .msg i 1 1]

/-- Ligerito's rounds from the message `mi` and the OOD value `oi` (rec_algebra.py:767–772). -/
def ligRounds (sch : Schedule) : Array (Array Key) := Id.run do
  let R := sch.levels.size - 1
  let mut mi := 0
  let mut oi := 0
  let mut out : Array (Array Key) := #[#[.T], #[.ood oi]]
  oi := oi + 1
  for _ in [0:sch.k0 + 1] do
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
      out := out.push (msgNames mi ++ (Array.range (2 <<< sch.ext)).map .yr)
      mi := mi + 1
  return out

/-- A holo shape's round of `h`, absorbed after `z_partial` and before `r′` is drawn (`H_ROUND`). -/
def hRound (sh : Shape) : Array (Array Key) := if sh.holo then #[(Array.range 64).map .h] else #[]

/-- The rep's message elements by the round that absorbs them, in protocol order (`rec_algebra.rounds`,
rec_algebra.py:748–772, and `H_ROUND`); rounds that absorb no element are not listed. -/
def rounds (sh : Shape) : Array (Array Key) :=
  #[(Array.range 64).map .ab ++ (Array.range 64).map .c] ++
    (Array.range (sh.m - 6)).map (fun i => #[.g1 i, .ginf i]) ++ #[#[.fa, .fb]] ++
    (Array.range (sh.kLog - 6)).map (fun j => #[.q1 j, .qinf j]) ++ #[(Array.range 64).map .zp] ++ hRound sh ++
    (Array.range sh.claims).map (fun k => (Array.range 128).map (.rs k)) ++ ligRounds sh.sched

/-- The running sums' elements w reads (`ACC_NAMES`, rec_algebra.py:745). -/
def accNames : Array Key := #[.acc0 0, .acc0 1, .acc1 0, .acc1 1]

/-- w's names (`_w_names`, rec_algebra.py:812–813): the rounds' elements, then the running sums. -/
def wNames (sh : Shape) : Array Key := (rounds sh).flatten ++ accNames

/-- An element's slot in w (`wi`, rec_algebra.py:834). -/
def wi (sh : Shape) (k : Key) : Option Nat := (wNames sh).idxOf? k

/-- The order of `_build`'s terms (`_order`, rec_algebra.py:816–822): w's names, then the ring switches' transposes,
then the comb. -/
def order (sh : Shape) (k : Key) : Option Nat :=
  let base := (wNames sh).size
  match k with
  | .rsT k u => some (base + 128 * k + u)
  | .comb l => some (base + 128 * sh.claims + l)
  | k => wi sh k

/-- The rep's messages as symbolic values (`symbolic`, rec_algebra.py:790–808). -/
def symbolic (sh : Shape) : Messages Lin :=
  let var := Lin.var
  let s := sh.sched
  let R := s.levels.size - 1
  { ab := (Array.range 64).map (var ∘ .ab)
    c := (Array.range 64).map (var ∘ .c)
    zc := (Array.range (sh.m - 6)).map fun i => (var (.g1 i), var (.ginf i))
    fa := var .fa
    fb := var .fb
    lc := (Array.range (sh.kLog - 6)).map fun j => (var (.q1 j), var (.qinf j))
    zp := (Array.range 64).map (var ∘ .zp)
    rs := (Array.range sh.claims).map fun k => (Array.range 128).map (var ∘ .rs k)
    rsT := (Array.range sh.claims).map fun k => (Array.range 128).map (var ∘ .rsT k)
    ood := (Array.range (1 + 2 * R)).map (var ∘ .ood)
    msgs := (Array.range (s.k0 + 8 + 7 * (R - 1))).map fun i =>
      ((var (.msg i 0 0), var (.msg i 0 1)), (var (.msg i 1 0), var (.msg i 1 1)))
    yr := (Array.range (2 <<< s.ext)).map (var ∘ .yr)
    comb := (Array.range 64).map (var ∘ .comb)
    cons := fun level _ => (Array.range (2 ^ (s.levels[level]!).lanes)).map (var ∘ .cons level)
    tAtom := some (var .T) }

/-! ## `GfResiduals`' structure -/

/-- A term of `GfResiduals`' S (`["w", i]`, `["v", j]`, `["wt", i, u]`; `gf2k._term_value`). -/
inductive Term where
  | w (i : Nat)
  | v (j : Nat)
  | wt (i u : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- A residual of S: its products of operands and its added terms. -/
structure Res where
  mul : Array (Nat × Nat)
  add : Array Term
  deriving DecidableEq, Repr, Inhabited

/-- `GfResiduals_v1{S}`'s S: w's and v's sizes, the operands (each a sum of terms) and the residuals. -/
structure GfS where
  nw : Nat
  nv : Nat
  ops : Array (Array Term)
  res : Array Res
  deriving DecidableEq, Repr, Inhabited

/-- A v slot's name (`Layout.v`, rec_algebra.py:862, 871, 876). -/
inductive VName where
  | coef (ri : Nat) (k : Key)
  | const (ri : Nat)
  | comb (l : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- `_build`'s state (rec_algebra.py:837–841). -/
structure BSt where
  ops : Array (Array Term) := #[]
  opOf : List (Key × Nat) := []
  vNames : Array VName := #[]
  vVals : Array F128 := #[]
  kappa : Array (Array (Array F128))

/-- A message element's term (`var_op`, rec_algebra.py:847): its element of w, its ring switch's transpose (`wt`), or
the comb's slot `comb0 + l` of v. A consistency sum is no operand: Python's `wi[k]` fails on it, as w holds none. -/
def atomTerm (sh : Shape) (comb0 : Nat) : Key → Except String Term
  | .rsT k u => match wi sh (.rs k 0) with
    | some i => .ok (.wt i u)
    | none => .error "build: no such ring switch"
  | .comb l => .ok (.v (comb0 + l))
  | .cons _ _ => .error "build: a consistency sum as an operand"
  | k => match wi sh k with
    | some i => .ok (.w i)
    | none => .error "build: an element w does not hold"

/-- A computation that can fail, in `_build`'s state. -/
def liftB {A : Type} : Except String A → StateT BSt (Except String) A
  | .ok a => pure a
  | .error e => throw e

/-- A message element's operand (`var_op`, rec_algebra.py:847, and its `combh`): its term, and in a holo shape the
comb's `h[l]` after the comb's slot (`[["v", comb0 + l], ["w", wi[("h", l)]]]`). -/
def atomOp (sh : Shape) (comb0 : Nat) (k : Key) : Except String (Array Term) := do
  let t ← atomTerm sh comb0 k
  match k, sh.holo with
  | .comb l, true => match wi sh (.h l) with
    | some i => .ok #[t, .w i]
    | none => .error "build: no h"
  | _, _ => .ok #[t]

/-- `var_op` (rec_algebra.py:843–848): the operand of a message element, made when first needed. Python places the
comb's slots after every coefficient's by patching their operands at the end; here `comb0`, the number of coefficient
slots, is counted first. -/
def varOp (sh : Shape) (comb0 : Nat) (k : Key) : StateT BSt (Except String) Nat := do
  let st ← get
  match st.opOf.lookup k with
  | some i => return i
  | none =>
    let op ← liftB (atomOp sh comb0 k)
    set { st with opOf := st.opOf ++ [(k, st.ops.size)], ops := st.ops.push op }
    return st.ops.size

/-- The sort key of a term (`order.__getitem__`). -/
def orderOf (sh : Shape) (k : Key) : Except String Nat :=
  match order sh k with
  | some o => .ok o
  | none => .error "build: an element outside the order"

/-- Is this a consistency sum. -/
def Key.isCons : Key → Bool
  | .cons _ _ => true
  | _ => false

/-- A list sorted by the order of its keys (`sorted(..., key=order.__getitem__)`). -/
def sortBy {β : Type} (o : β → Except String Nat) (xs : List β) : Except String (List β) := do
  let keyed ← xs.mapM fun x => do return ((← o x), x)
  return (keyed.mergeSort fun a b => decide (a.1 ≤ b.1)).map (·.2)

/-- `kappa[k[1]][ri - fin][k[2]] = f.t[k]` (rec_algebra.py:857–858), as a sum into `kappa`: the keys are distinct, so
each entry is set once from zero, as Python sets it. A sum outside `kappa` is Python's `IndexError`. -/
def kapAdd (r : Nat) : List (Key × F128) → StateT BSt (Except String) Unit
  | [] => pure ()
  | (k, x) :: rest => do
    match k with
    | .cons level l =>
      let st ← get
      if level < st.kappa.size ∧ r < st.kappa[level]!.size ∧ l < st.kappa[level]![r]!.size then
        set { st with kappa := st.kappa.modify level (·.modify r (·.modify l (· + x))) }
      else throw "build: a consistency sum outside kappa"
    | _ => pure ()
    kapAdd r rest

/-- A residual's coefficient slots and their products (rec_algebra.py:860–865). -/
def coefOps (sh : Shape) (comb0 ri : Nat) : List (Key × F128) → StateT BSt (Except String) (Array (Nat × Nat))
  | [] => return #[]
  | (k, x) :: rest => do
    let st ← get
    set { st with vNames := st.vNames.push (.coef ri k), vVals := st.vVals.push x,
                  ops := st.ops.push #[.v st.vNames.size] }
    let o ← varOp sh comb0 k
    return #[(st.ops.size, o)] ++ (← coefOps sh comb0 ri rest)

/-- A residual's products of two messages (rec_algebra.py:866–869). -/
def prodOps (sh : Shape) (comb0 ri : Nat) : List ((Key × Key) × F128) → StateT BSt (Except String) (Array (Nat × Nat))
  | [] => return #[]
  | ((a, b), x) :: rest => do
    if x ≠ 1 then throw s!"residual {ri}: a product of messages with a coefficient other than 1"
    let oa ← varOp sh comb0 a
    let ob ← varOp sh comb0 b
    return #[(oa, ob)] ++ (← prodOps sh comb0 ri rest)

/-- A residual's running sums (rec_algebra.py:855–859): none without consistency sums; else, in Ligerito's final check
only, its coefficients on them go into `kappa` and the running sums after and before the openings are added. -/
def accTerms (sh : Shape) (fin ri : Nat) (cons : List (Key × F128)) : StateT BSt (Except String) (Array Term) :=
  if cons.isEmpty then pure #[]
  else if ri ≠ fin ∧ ri ≠ fin + 1 then throw s!"residual {ri}: a consistency sum outside Ligerito's final check"
  else do
    kapAdd (ri - fin) cons
    match wi sh (.acc1 (ri - fin)), wi sh (.acc0 (ri - fin)) with
    | some i1, some i0 => pure #[Term.w i1, .w i0]
    | _, _ => throw "build: no running sums"

/-- A residual's constant (rec_algebra.py:870–873): a v slot holding it. -/
def constTerm (ri : Nat) : Option F128 → StateT BSt (Except String) (Array Term)
  | some c => do
    let st ← get
    set { st with vNames := st.vNames.push (.const ri), vVals := st.vVals.push c }
    pure #[Term.v st.vNames.size]
  | none => pure #[]

/-- `_build`'s pass over residual `ri` (rec_algebra.py:851–874). -/
def buildRes (sh : Shape) (fin comb0 ri : Nat) (f : Lin) : StateT BSt (Except String) Res := do
  let acc ← accTerms sh fin ri (f.t.filter (·.1.isCons))
  let cm ← coefOps sh comb0 ri (← liftB (sortBy (fun kx => orderOf sh kx.1) (f.t.filter (!·.1.isCons))))
  let pm ← prodOps sh comb0 ri
    (← liftB (sortBy (fun abx => do return (← orderOf sh abx.1.1) * 2 ^ 32 + (← orderOf sh abx.1.2)) f.q))
  let c ← constTerm ri f.c
  return ⟨cm ++ pm, acc ++ c⟩

/-- `_build`'s passes over the residuals from `ri`. -/
def buildAll (sh : Shape) (fin comb0 : Nat) : List Lin → Nat → StateT BSt (Except String) (Array Res)
  | [], _ => return #[]
  | f :: rest, ri => do
    let r ← buildRes sh fin comb0 ri f
    let rs ← buildAll sh fin comb0 rest (ri + 1)
    return #[r] ++ rs

/-- What `_build` reads off the symbolic residuals (rec_algebra.py:830–881): S, the v slots' names and the
coefficient slots' values, and `kappa[level][r][l]`, the final check's component `r`'s coefficient on the level's
consistency sum `l`. -/
structure Built where
  S : GfS
  vNames : Array VName
  vVals : Array F128
  kappa : Array (Array (Array F128))

/-- A residual's coefficient slots: its terms but the consistency sums, and its constant. -/
def slotsOf (f : Lin) : Nat := (f.t.filter (!·.1.isCons)).length + (if f.c.isSome then 1 else 0)

/-- `_build` (rec_algebra.py:830–881). The comb's slots follow the `comb0` coefficient slots, which the passes fill. -/
def build (sh : Shape) (lins : Array Lin) : Except String Built := do
  let kappa := sh.sched.levels.map fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))
  let comb0 := (lins.toList.map slotsOf).sum
  let (res, st) ← (buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa }
  if st.vNames.size ≠ comb0 ∨ st.vVals.size ≠ comb0 then throw "build: the coefficient slots miscounted"
  let vNames := st.vNames ++ (Array.range 64).map .comb
  return ⟨⟨(wNames sh).size, vNames.size, st.ops, res⟩, vNames, st.vVals, st.kappa⟩

/-- The symbolic residuals at a coin source (`_residual_lins`, rec_algebra.py:825–827). -/
def residualLins (sh : Shape) (coins : CoinSource) (extras : Array Claim) : Except String (Array Lin × Info Lin) :=
  ((run sh (symbolic sh) extras).run' 0).run coins

/-- Extra claims of the statement's number, all zero (`_extras_like`, rec_algebra.py:904–909, whose are random). -/
def extrasLike (sh : Shape) : Array Claim :=
  Array.replicate (sh.claims - 2) { value := 0, skip := Array.replicate 64 0, x := Array.replicate (sh.m - 6) 0 }

/-- `GfResiduals_v1{S}`'s S, a function of the shape alone (`rec_algebra.structure`, rec_algebra.py:915–921). -/
def structureOf (sh : Shape) : Except String GfS := do
  let (lins, _) ← residualLins sh CoinSource.zeros (extrasLike sh)
  return (← build sh lins).S

/-- `GfResiduals`' v port at the rep's coins (the comb last), the coin-derived points and `kappa`
(`rec_algebra.verifier_rows`, rec_algebra.py:931–940); refused when the structure moved with the coins. -/
def verifierRows (sh : Shape) (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128) :
    Except String (Array F128 × Info Lin × Array (Array (Array F128))) := do
  let (lins, info) ← residualLins sh (CoinSource.ofRounds rounds) extras
  let b ← build sh lins
  if some b.S ≠ (structureOf sh).toOption then throw "the structure moved with the coins"
  return (b.vVals ++ comb, info, b.kappa)

/-! ## `GfResiduals`' semantics -/

/-- A term's value (`gf2k._term_value`): w's element, v's element, or the element whose bit `j` is bit `u` of
`w[i + j]` (written as `Flock.ringSwitch` writes its transpose). -/
def termValue (w v : Array F128) : Term → F128
  | .w i => w[i]!
  | .v j => v[j]!
  | .wt i u => (Array.range 128).foldl (fun acc v => if (w[i + v]!).bit u then acc + F128.monomial v else acc) 0

/-- An operand's value: the sum of its terms. -/
def opValue (S : GfS) (w v : Array F128) (k : Nat) : F128 := (S.ops[k]!).foldl (fun x t => x + termValue w v t) 0

/-- A residual's value: its added terms, then its products. -/
def resValue (S : GfS) (w v : Array F128) (r : Res) : F128 :=
  r.mul.foldl (fun x ab => x + opValue S w v ab.1 * opValue S w v ab.2) (r.add.foldl (fun x t => x + termValue w v t) 0)

/-- `GfResiduals{S}` on F elements (`gf2k.residuals`). -/
def residuals (S : GfS) (w v : Array F128) : Except String (Array F128) :=
  if w.size ≠ S.nw ∨ v.size ≠ S.nv then .error s!"ports of {w.size} and {v.size} elements, S wants {S.nw} and {S.nv}"
  else .ok (S.res.map (resValue S w v))

end Flock.RecAlgebra
