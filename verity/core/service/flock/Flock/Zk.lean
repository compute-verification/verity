/-
`flock-circuit --zk` (live PROTOCOL.md §1–§7, M1 with M2's committed coins): the verifier. The zerocheck and lincheck are
replayed on masked values, every check an affine constraint on the committed pads `h` (`Lin`); the opening's two masked
claims' ring-switch checks and level 0's lane phase become constraints too; the inner proof shows that the pads satisfy every
constraint at once (`verifyInner`). The session around the reps is `Flock.verify`'s, with the keyed coin tree.
-/
import Flock.Verify
import Flock.ZkLigerito

namespace Flock.Zk

open Transcript
open Lean (Json)

def LABEL_PADS : String := "verity-flock-zk-pads-v0"
def LABEL_INNER : String := "verity-flock-zk-inner-v0"
/-- The inner code's padding symbols per row. -/
def K_PAD : Nat := 192
/-- The inner proof's opened columns. -/
def Q : Nat := 168
def INNER_LOG_INV_RATE : Nat := 3
def TAU_SALT_BYTES : Nat := 192
def TAU_TAG : ByteArray := "verity/flock-zk/tau/v1".toUTF8.push 0
/-- The claim points a table's rank check covers (ab and c, for both reps), and the mask words beyond `128 ·` that. -/
def RANK_POINTS : Nat := 4
def RANK_SLACK : Nat := 64
/-- The least mask slot `--zk` takes, `2^13` bits per block (the deployed circuits' `mask_log` is 14): the bound on the
honest prover's rank check refusing a coin, its completeness loss, covers no smaller slot. -/
def SLOT_LOG_MIN : Nat := 13

/-- Level 0's query count `q_0` (live PROTOCOL.md §7, upstream's `zk_queries0`): 277, 242, 229 at `m = 25, 26, 27`, else the
least `q_0 ≥ 218` (fast100's) with `q_0 log2(1/(√ρ' + 1/50)) ≥ 100`, `ρ' = (L + 2 q_0) / 2L`, `L = 2^{m − 7 − k_0}` (219 at
`m = 28..31`, 218 at `32..35`, each at least 0.02 bits from the threshold, so the table is exact); none at `m ≤ 24`. -/
def queries0 (m : Nat) : V Nat :=
  if m ≤ 24 then throw s!"m = {m}: --zk needs m >= 25 (Ligerito level 0's padding outgrows its lanes at rate 1/2)"
  else if m == 25 then pure 277
  else if m == 26 then pure 242
  else if m == 27 then pure 229
  else if m ≤ 31 then pure 219
  else if m ≤ 35 then pure 218
  else throw s!"m = {m}: no fast100 schedule"

/-- The `--zk` schedule: fast100's with level 0's `q_0` queries (restratified, its cap the first stratum), and `t_pad = 2 q_0`. -/
def schedule (m : Nat) : V (Schedule × Nat) := do
  let q0 ← queries0 m
  let some sch := fast100 m | throw s!"m = {m}: no fast100 schedule"
  let l0 := sch.levels[0]!
  let strata := decompose q0 l0.d
  return ({ sch with levels := sch.levels.set! 0 { l0 with q := q0, strata, cap := strata.getD 0 0 } }, 2 * q0)

/-- Where each pad sits (upstream's `Layout`, live PROTOCOL.md §2): the sent values in proof-field order (zerocheck, lincheck,
level 0's OOD value, lane-round messages and the pair's sums), then `h_ab` and the triple `x, y, w`. -/
structure PadLayout where
  nMlv : Nat
  nLc : Nat
  nOod0 : Nat
  k0 : Nat
  deriving Inhabited

namespace PadLayout

def new (m kLog k0 : Nat) : PadLayout := ⟨m - 6, kLog - 6, 1, k0⟩
def r1ab (_ : PadLayout) (i : Nat) : Nat := i
def r1c (_ : PadLayout) (i : Nat) : Nat := 64 + i
def ml (_ : PadLayout) (j t : Nat) : Nat := 128 + 2 * j + t
def fa (l : PadLayout) : Nat := 128 + 2 * l.nMlv
def fb (l : PadLayout) : Nat := l.fa + 1
def fc (l : PadLayout) : Nat := l.fa + 2
def lc (l : PadLayout) (j t : Nat) : Nat := l.fc + 1 + 2 * j + t
def zp (l : PadLayout) (i : Nat) : Nat := l.lc l.nLc 0 + i
def lgOod (l : PadLayout) (i : Nat) : Nat := l.zp 64 + i
def lgMsg (l : PadLayout) (j t : Nat) : Nat := l.zp 64 + l.nOod0 + 4 * j + t
def lgE (l : PadLayout) (i c : Nat) : Nat := l.zp 64 + l.nOod0 + 4 * l.k0 + 2 * i + c
def hab (l : PadLayout) : Nat := l.zp 64 + l.nOod0 + 4 * l.k0 + 4
def tx (l : PadLayout) : Nat := l.hab + 1
def ty (l : PadLayout) : Nat := l.hab + 2
def tw (l : PadLayout) : Nat := l.hab + 3
def len (l : PadLayout) : Nat := l.tw + 1

end PadLayout

/-- `c + Σ a_i h_i` (characteristic 2: a masked value `v' = v + h_i` is the unmasked `v = v' + h_i`). -/
structure Lin where
  c : F128
  a : Array F128
  deriving Inhabited

namespace Lin

def konst (c : F128) (s : Nat) : Lin := ⟨c, Array.replicate s 0⟩
/-- The unmasked value behind masked `v` under pad `i`. -/
def masked (v : F128) (i s : Nat) : Lin := ⟨v, (Array.replicate s 0).set! i 1⟩
def add (x y : Lin) : Lin := ⟨x.c + y.c, (Array.range (min x.a.size y.a.size)).map fun i => x.a[i]! + y.a[i]!⟩
def scale (x : Lin) (k : F128) : Lin := ⟨x.c * k, x.a.map (· * k)⟩
def addConst (x : Lin) (k : F128) : Lin := { x with c := x.c + k }
/-- `Σ w_i l_i` over `min |ws| |ls|` terms, on `s` pads. -/
def combo (ws : Array F128) (ls : Array Lin) (s : Nat) : Lin := Id.run do
  let mut c : F128 := 0
  let mut a : Array F128 := Array.replicate s 0
  for i in [0:min ws.size ls.size] do
    let w := ws[i]!
    let l := ls[i]!
    c := c + w * l.c
    for k in [0:min s l.a.size] do
      a := a.set! k (a[k]! + w * l.a[k]!)
  return ⟨c, a⟩

end Lin

/-- An affine form over K by coordinates (`x = x_0 + x_1 U`). -/
abbrev Lin2 := Lin × Lin

def l2masked (v : F256) (i0 i1 s : Nat) : Lin2 := (Lin.masked v.c0 i0 s, Lin.masked v.c1 i1 s)
def l2konst (v : F256) (s : Nat) : Lin2 := (Lin.konst v.c0 s, Lin.konst v.c1 s)
def l2add (x y : Lin2) : Lin2 := (x.1.add y.1, x.2.add y.2)
/-- `k · x` for a public `k`: `x_0 k + x_1 (k U)`, coordinate by coordinate. -/
def l2scale (x : Lin2) (k : F256) : Lin2 :=
  let m1 := k * F256.u
  ((x.1.scale k.c0).add (x.2.scale m1.c0), (x.1.scale k.c1).add (x.2.scale m1.c1))

/-- What the masked algebraic phase hands the opening: the two z-claims (their values unused), their values as affine
forms, and the constraints so far (each `Lin = 0`). -/
structure Replayed where
  claims : Array Claim
  abValue : Lin
  cValue : Lin
  cons : Array (String × Lin)
  deriving Inhabited

/-- Live PROTOCOL.md §1 and §4: §9.1's binding, then §9.2 and §10 on the masked proof, absorbing exactly what they absorb and
drawing exactly their coins, with the pads' root absorbed after the zerocheck's label and every message-dependent check a
constraint. -/
def replay (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) : TM Replayed := do
  let s := lay.len
  let m := st.m
  let zc := p.zerocheck
  label "flock-r1cs-v0"
  absorbBytes st.digest
  absorbBytes (p.cap.foldl (· ++ ·) ByteArray.empty)
  label "flock-zerocheck-v0"
  label LABEL_PADS
  absorbBytes root
  let _rSkip ← sampleVec 6
  let rOuter ← sampleVec (m - 13)
  let r := pinned ++ rOuter
  absorbSlice zc.round1ab
  absorbSlice zc.round1c
  let z ← sample
  let r1c := (Array.range 64).map fun i => Lin.masked zc.round1c[i]! (lay.r1c i) s
  let r1ab := (Array.range 64).map fun i => Lin.masked zc.round1ab[i]! (lay.r1ab i) s
  let computedC := Lin.combo (lagrange nodesL den6Inv z) r1c s
  let fc := Lin.masked zc.finalC lay.fc s
  let mut cons : Array (String × Lin) := #[("zerocheck c_eval", computedC.add fc)]
  let comb := (Array.range 64).map fun i => r1ab[i]!.add r1c[i]!
  let mut running := (Lin.combo (combWeights z) comb s).add computedC
  let mut rho : Array F128 := #[]
  for j in [0:m - 6] do
    let (m1, minf) := zc.rounds[j]!
    let req := r[j]!
    let g1 := Lin.masked m1 (lay.ml j 0) s
    let ginf := Lin.masked minf (lay.ml j 1) s
    let g0 := (running.add (g1.scale req)).scale (1 + req).inv
    absorbF m1
    absorbF minf
    let rr ← sample
    rho := rho.push rr
    running := ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))
  let a := Lin.masked zc.finalA lay.fa s
  let b := Lin.masked zc.finalB lay.fb s
  -- a · b = a'b' + a' h_b + b' h_a + h_ab
  let prod : Lin := ⟨zc.finalA * zc.finalB, (((Array.replicate s 0).set! lay.fb zc.finalA).set! lay.fa zc.finalB).set! lay.hab 1⟩
  cons := cons.push ("zerocheck final a·b", running.add prod)
  absorbF zc.finalA
  absorbF zc.finalB
  -- lincheck
  let kLog := st.kLog
  label "flock-lincheck-v0"
  let alpha ← sample
  let beta ← sample
  running := ((a.scale alpha).add b).addConst beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (e1v, einfv) := p.lincheck.rounds[j]!
    absorbF e1v
    absorbF einfv
    let rr ← sample
    let e1 := Lin.masked e1v (lay.lc j 0) s
    let einf := Lin.masked einfv (lay.lc j 1) s
    let e0 := running.add e1
    let c1 := (e0.add e1).add einf
    running := ((einf.scale (rr * rr)).add (c1.scale rr)).add e0
    ts := ts.push rr
  let combv ← match st.fold.folded alpha beta (skipWeights z) (rho.extract 0 (kLog - 6)) ts with
    | .ok v => pure v
    | .error err => throw err
  check (combv.size == 64) "lincheck: the circuit fold has the wrong width"
  absorbSlice p.lincheck.zPartial
  let zp := (Array.range 64).map fun i => Lin.masked p.lincheck.zPartial[i]! (lay.zp i) s
  cons := cons.push ("lincheck final", running.add (Lin.combo combv zp s))
  let rPrime ← sample
  let w := Lin.combo (skipWeights rPrime) zp s
  let zcOut : ZcOut := { z, rho, rRest := r, vA := 0, vB := 0, vC := 0 }
  let lcOut : LcOut := { w := 0, rPrime, rIn := ts.reverse }
  return { claims := abcClaims m kLog zcOut lcOut, abValue := w, cValue := fc, cons }

/-- §12.2's claim check `⟨w, s_hat_v⟩` on claim `c`'s point (its value is masked, so it is a constraint, §4). -/
def claimCheck (c : Claim) (sv : Array F128) : F128 :=
  let x0 := c.x[0]!
  dot ((Array.range 128).map fun v => c.skip[v % 64]! * (if v / 64 == 1 then x0 else 1 + x0)) sv

/-- Level 0's lane phase as constraints (live PROTOCOL.md §7): the claim `target + Σ β_i y_i` through the `k_0` lane rounds'
quadratics `u_0 + r (claim + u_2) + r² u_2`, then `claim + ρ_0 e_0 + ρ_1 e_1 = T'`, on the masked OOD value, lane-round
messages and pair's sums. -/
def ligeritoConstraints (lay : PadLayout) (rec : L0Rec) (lg : LigeritoProof) (msgs : Level0Msgs) : V (Array (String × Lin)) := do
  let s := lay.len
  if rec.ood.size != lay.nOod0 || rec.challenges.size != lay.k0 || lg.messages.size < lay.k0 then
    throw "zk: level 0's record is not the layout's"
  let mut claim0 := Lin.konst rec.target s
  for i in [0:rec.ood.size] do
    let (beta, _) := rec.ood[i]!
    claim0 := claim0.add ((Lin.masked lg.oodValues[i]! (lay.lgOod i) s).scale beta)
  let mut claim : Lin2 := (claim0, Lin.konst 0 s)
  for j in [0:lay.k0] do
    let r := rec.challenges[j]!
    let (u0v, u2v) := lg.messages[j]!
    let u0 := l2masked u0v (lay.lgMsg j 0) (lay.lgMsg j 1) s
    let u2 := l2masked u2v (lay.lgMsg j 2) (lay.lgMsg j 3) s
    claim := l2add (l2add u0 (l2scale (l2add claim u2) r)) (l2scale u2 (r * r))
  let e0 := l2masked msgs.e0 (lay.lgE 0 0) (lay.lgE 0 1) s
  let e1 := l2masked msgs.e1 (lay.lgE 1 0) (lay.lgE 1 1) s
  let lhs := l2add (l2add claim (l2scale e0 rec.rho.1)) (l2add (l2scale e1 rec.rho.2) (l2konst msgs.tPrime s))
  return #[("ligerito lane phase (c0)", lhs.1), ("ligerito lane phase (c1)", lhs.2)]

/-- The triple's constraints for the sacrifice coin `ε` and the opened `ρ, σ` (live PROTOCOL.md §4). -/
def triple (lay : PadLayout) (eps rho sigma : F128) : Array (String × Lin) :=
  let s := lay.len
  let z := Array.replicate s (0 : F128)
  #[("triple ρ", ⟨rho, (z.set! lay.fa eps).set! lay.tx 1⟩),
    ("triple σ", ⟨sigma, (z.set! lay.fb 1).set! lay.ty 1⟩),
    ("triple sacrifice", ⟨rho * sigma, (((z.set! lay.hab eps).set! lay.tw 1).set! lay.tx sigma).set! lay.ty rho⟩)]

/-- `τ`'s commitment: the `hm96-sha512/v1` leaf of `SHA-512(TAU_TAG ‖ τ)` under `salt`. -/
def tauCommitment (tau : F128) (salt : ByteArray) : ByteArray :=
  Hm96.Default512.leaf (Sha512.hash (TAU_TAG ++ tau.toBytes)) salt

/-- The inner code at column `j`: `Σ y_i X^i` at `X = j + 1` (an element of the polynomial basis). -/
def codeAt (y : Array F128) (j : Nat) : F128 :=
  let x : F128 := ⟨(j + 1).toUInt64, 0⟩
  y.foldr (fun c acc => acc * x + c) 0

/-- Live PROTOCOL.md §5: the inner proof that the committed pads satisfy `cons` and the triple's constraints. `finalC`, the
proof's `final_c'`, is a constant of two of `cons` that nothing before absorbs; it is absorbed before `ε`. -/
def verifyInner (lay : PadLayout) (pr : InnerProof) (finalC : F128) (cons : Array (String × Lin)) : TM Unit := do
  let msg := lay.len + K_PAD
  let n := msg.nextPowerOfTwo * 2 ^ INNER_LOG_INV_RATE
  check (pr.y.size == msg && pr.opened.size == Q && pr.salts.size == Q && pr.paths.size == Q && pr.root.size == 64
    && pr.tauCommit.size == 64 && pr.tauSalt.size == TAU_SALT_BYTES) "zk inner: malformed proof"
  label LABEL_INNER
  absorbF finalC
  let eps ← sample
  absorbSlice #[pr.rho, pr.sigma]
  let cons := cons ++ triple lay eps pr.rho pr.sigma
  let gammas ← sampleVec cons.size
  let l := Lin.combo gammas (cons.map (·.2)) lay.len
  absorbBytes pr.tauCommit
  let beta ← sample
  check (!beta.isZero) "zk inner: degenerate coin β = 0"
  absorbSlice pr.y
  let qs := (← sampleVec Q).map fun w => w.lo.toNat % n
  check (tauCommitment pr.tau pr.tauSalt == pr.tauCommit) "zk inner: τ does not open its commitment"
  check (dot l.a pr.y == l.c + beta * pr.tau) "zk inner: the batched constraint fails (⟨c, Y⟩ ≠ t + β τ)"
  let depth := Nat.log2 n
  for k in [0:Q] do
    let j := qs[k]!
    let (a, b) := pr.opened[k]!
    check (codeAt pr.y j == a + beta * b) s!"zk inner: column {j} is not consistent with Y"
    let path := pr.paths[k]!
    check (path.size == depth) "zk inner: a path of the wrong depth"
    let leaf := Hm96.Default512.leaf (Sha512.hash (a.toBytes ++ b.toBytes)) pr.salts[k]!
    check (climb MerkleScheme.hm96Sha512 leaf j path.toList == pr.root) s!"zk inner: column {j}'s path does not reach the pads' root"

/-- Table `i`'s rep `r` (`Flock.verifyRep` for a `--zk` proof): the masked replay, the opening with the masked claims'
checks as constraints, level 0's hiding, the inner proof; returns the proof's flattened cap. -/
def verifyRep (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : Nat) (sess : Session) (i r : Nat) (bytes : ByteArray) :
    V ByteArray := do
  let t := spec.tables[i]!
  let name := s!"{t}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes
  let shape : Shape := { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let zp ← decode shape bytes
  let p := zp.proof
  if p.params != pcsParams st.merkle sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
  if zp.inner.root.size != 64 then throw "zk: the pads' root is not 64 bytes"
  let lay := PadLayout.new st.m st.kLog sch.k0
  let prog : TM Unit := do
    let rp ← replay st lay p zp.inner.root
    let ab := rp.claims[0]!
    let c := rp.claims[1]!
    let ccAb := claimCheck ab p.ringSwitches[0]!
    let ccC := claimCheck c p.ringSwitches[1]!
    let (T, sws) ← ringSwitch st.m (#[{ ab with value := ccAb }, { c with value := ccC }] ++ extras) p.ringSwitches
    let rec ← ligerito st.merkle sch T p.cap p.ligerito (basis sws) ⟨tPad, r, zp.level0⟩
    let lg ← match ligeritoConstraints lay rec p.ligerito zp.level0 with
      | .ok v => pure v
      | .error e => throw e
    verifyInner lay zp.inner p.zerocheck.finalC
      (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lg)
  let ((), tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- The verdict on a `--zk` session (`Flock.verify` with `verifyRep` above): `.ok ()` is Accept. Its coins are the keyed
coin tree's (M2), so `spec` must name one. -/
def verify (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) : V Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  if spec.coinTree.isNone then throw "zk: the session's coins are not the keyed coin tree"
  let sess ← Record.decode spec record
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => schedule st.m
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (spec.dg b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, (sch, tPad)) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRep spec st sch tPad sess i 0 proofs[2 * i]!
    let c1 ← verifyRep spec st sch tPad sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

/-- **The statement avoids the mask range `r`** (D12): no pair of Δ (`da`, `db`) has a row or a column in `r`'s words, and
every region, at each of `r`'s words, fixes an in-block word bit (`i + 7 < k_log`, not free) that the word's index does not
have. The mask slot's uniform words hide every non-region `ŝ_v` (live PROTOCOL.md §6), so the zero-knowledge proof needs
them untouched by Δ and by every region; no proof derives it from the verifier's other checks. -/
def maskFree (st : Stmt) (r : Range) : Bool :=
  let lo := r.start / 128
  let off := fun (i : Nat) => !(decide (lo ≤ i / 128) && decide (i / 128 < lo + 2 ^ (r.slotLog - 7)))
  st.da.all (fun p => off p.1 && off p.2) && st.db.all (fun p => off p.1 && off p.2) &&
  st.regions.all fun reg => (List.range (2 ^ (r.slotLog - 7))).all fun j =>
    (List.range (st.c.kLog - 7)).any fun i => !reg.free.contains (i + 7) &&
      (((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1) != (lo + j).testBit i)

/-- A statement `--zk` can prove (upstream's `zk_shape_ok`): level 0's query count, a mask slot of at least
`2^SLOT_LOG_MIN` bits, and mask words enough for the prover's rank check (live PROTOCOL.md §6) to be reachable. Checked
when the statement is set up, as upstream does before `Hello`. And, which upstream does not check, Δ and the regions
avoid the mask slot (`maskFree`, D12). -/
def shapeOk (st : Stmt) : V Unit := do
  let _ ← queries0 st.m
  let some r := st.c.ranges.find? (·.kind == Kind.mask) | throw "zk: the circuit has no mask slot"
  if r.slotLog < SLOT_LOG_MIN then
    throw s!"slot_log = {r.slotLog}: --zk needs a mask slot of slot_log >= {SLOT_LOG_MIN}"
  let words := 2 ^ st.nbl * 2 ^ (r.slotLog - 7)
  if words < 128 * RANK_POINTS + RANK_SLACK then
    throw s!"{words} mask words: the rank check needs {128 * RANK_POINTS} for its {RANK_POINTS} claim points, and {RANK_SLACK} more"
  if !maskFree st r then
    throw "zk: a pair of Δ or a region meets the mask slot (D12: the zero-knowledge proof covers no such statement)"

/-- META `mask_log` of a circuit file (14 when absent), which the identity names. It reads the file's second line, the
META line, and nothing past it, so it costs that line's bytes whatever the file's size; a file that isn't UTF-8 past it
`HmOut.parse` refuses. -/
def maskLogOf (circuit : ByteArray) : Nat :=
  match findByte circuit 10 with
  | none => 14
  | some e1 =>
    let e2 := (findByte circuit 10 (e1 + 1)).getD circuit.size
    match String.fromUTF8? (circuit.extract (e1 + 1) e2) with
    | none => 14
    | some line =>
      match Json.parse (line.drop 5).toString with
      | .ok j => (j.getObjValAs? Nat "mask_log").toOption.getD 14
      | .error _ => 14

private def s (x : String) : Json := Json.str x

/-- The identity's `hooks` for a `--zk` statement (upstream's `zk_identity`): what is masked, and how; `tables` for a session
of several tables; `os` when the prover's every random value is the OS's bytes (`Tags.osMasks`), else its pads and every
table's draws are ChaCha20 streams of its seed. -/
def hooks (maskLog : Nat) (tables : Bool) (os : Bool := false) : Json :=
  let pads := if os then "os bytes, getrandom" else "prover ChaCha20 stream 4"
  let draws := if os then "os bytes of its own (getrandom)" else "ChaCha20 streams at purpose | j << 32"
  let base := [("zk", Json.bool true), ("stage", s "m2-prototype (committed verifier coins, CPU)"),
    ("compiler", s "veil (ePrint 2026/683 §4.1, Remark 4.4) over gf(2^128)"),
    ("verifier_coins", s s!"committed before the prover's first message ({CoinTree.SCHEME}): the prover's fresh nonce in hello; the root and the verifier's session key in its answer; every coin opened against them, or the prover stops"),
    ("pads", s s!"zerocheck and lincheck: every sent value + a pad ({pads}); pads committed before the zerocheck's first coin"),
    ("pads_commitment", s s!"rs over gf(2^128), {K_PAD} padding symbols, rate 1/{2 ^ INNER_LOG_INV_RATE}, {Q} queries, hm96-sha512/v1 leaves, sha512 nodes"),
    ("product", s "h_ab = h_a h_b committed; sacrificed against a committed triple (x, y, x y)"),
    ("inner_proof", s "one batched <c, h> = t by a masked dot product (tau = <c, mu> committed by an hm96 leaf, opened after the last coin; Y = h + beta mu)"),
    ("ring_switch", s s!"non-region s_hat_v hidden by the mask slot's uniform words (2^{maskLog} bits per block); the prover refuses coins whose weights do not span"),
    ("ligerito_level0", s "veil figs. 8-9: every lane padded by (W_n + c) mu with t_pad = 2 q_0 uniform coefficients (c = x); four padded uniform lanes, rep r folding its own pair 2r, 2r + 1 after its lane rounds (rho_0, rho_1 F128-independent); the ood value, the lane-round messages and the pair's sums padded; T' and ybar sent (uniform)"),
    ("ligerito_level0_queries", s "q_0 = 277, 242, 229 at m = 25, 26, 27, else the least q_0 >= fast100's with q_0 log2(1/(sqrt(rho') + 1/50)) >= 100; rho' = (L + 2 q_0) / 2L; refused below 100 bits and at m <= 24"),
    ("refused_coins", Json.arr #[s "beta = 0", s "mask-slot weights of rank < 128 k over the claims so far", s "rho_0, rho_1 dependent over F128",
      s "a zerocheck outer coin r_j = 1"]),
    ("not_yet_masked", Json.arr #[])]
  Json.mkObj (if tables then base ++ [("tables", s s!"per table: {draws} and hm96 salt trees from id j << 32 (table 0's are a one-table session's); a rank check per table; every table's level-0 root in one commit; every proof after the last coin of every stream (the barrier)")] else base)

/-- A statement's tags under `--zk` in a session of `count` tables: the identity's `hooks` are `hooks` (naming the prover's
randomness as `t.osMasks` says); its `proof_class` stays upstream's (`NON_ZK_PROOF`). -/
def tags (t : Tags) (maskLog count : Nat) : Tags :=
  { t with identity := t.identity.setObjVal! "hooks" (hooks maskLog (count > 1) t.osMasks) }

/-- M2's committed coin schedule for a session's tables (upstream's `coin_spec_of`): the session's stream, then every table's
two reps, 512 rounds of at most 512 coins each. -/
def coinSpec (tables : Array String) : CoinTree.Spec :=
  ⟨#["session"] ++ tables.flatMap (fun t => #[s!"{t}/rep0", s!"{t}/rep1"]), 9, 512⟩

end Flock.Zk
