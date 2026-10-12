/-
PROTOCOL.md §13–§14: the dense Ligerito opening and its Merkle caps.
-/
import Flock.Opening
import Flock.Schedule
import Flock.Merkle

namespace Flock

open Transcript

/-- §13.5: stratified positions from query coins; reads only the low `d − c_j` bits of `coin.lo`. -/
def positions (lv : Level) (coins : Array F128) : Array Nat := Id.run do
  let mut out := #[]
  let mut j := 0
  for c in lv.strata do
    for s in [0:2 ^ c] do
      let lo := (coins[j]!).lo.toNat % 2 ^ (lv.d - c)
      out := out.push ((s <<< (lv.d - c)) ||| lo)
      j := j + 1
  return out

/-- The first `k` subspace-polynomial steps from `x`: `s ← s² + sv[j]·s` for `j < k` (§13.4). -/
def sIter (sv : Array F128) : Nat → F128 → F128
  | 0, x => x
  | k + 1, x => let y := sIter sv k x; y * y + sv[k]! * y

/-- `s_i(v_i)` for `i < n`: the subspace polynomials' normalisers (§13.4). -/
def svTable : Nat → Array F128
  | 0 => #[]
  | n + 1 => let sv := svTable n; sv.push (sIter sv n (F128.monomial n))

/-- From step `i` with `s = s_i(ω)`: push `s · sv[i]⁻¹` and step, `fuel` times. -/
def whatsGo (sv : Array F128) : Nat → Nat → F128 → Array F128 → Array F128
  | 0, _, _, out => out
  | fuel + 1, i, s, out => whatsGo sv fuel (i + 1) (s * s + sv[i]! * s) (out.push (s * (sv[i]!).inv))

/-- `Ŵ_i(ω_q)` for `i < n`. -/
def whats (sv : Array F128) (q : Nat) (n : Nat) : Array F128 := whatsGo sv n 0 ⟨q.toUInt64, 0⟩ #[]

/-- `eq` over K points (LSB first). -/
def eqTableK (r : Array F256) : Array F256 := Id.run do
  let mut t : Array F256 := #[1]
  for rj in r do
    let a := rj + 1
    t := t.map (· * a) ++ t.map (· * rj)
  return t

/-- A round polynomial `a0 + a1 X + a2 X²` over K. -/
structure Poly where
  a0 : F256
  a1 : F256
  a2 : F256
  deriving Inhabited

def Poly.ofMsg (msg : F256 × F256) (s : F256) : Poly := ⟨msg.1, s + msg.2, msg.2⟩
def Poly.eval (p : Poly) (r : F256) : F256 := p.a0 + p.a1 * r + p.a2 * r * r
def Poly.addScaled (p q : Poly) (b : F128) : Poly := ⟨p.a0 + q.a0.smul b, p.a1 + q.a1.smul b, p.a2 + q.a2.smul b⟩

structure OodCtx where
  z : Array F128
  beta : F128
  split : Option Nat
  deriving Inhabited

structure ConsCtx where
  cols : Nat
  pos : Array Nat
  alpha : Array F128
  beta : F128
  start : Nat
  deriving Inhabited

/-- The Ligerito verifier's explicit state. -/
structure LState where
  s : F256
  q : Poly
  msg : Nat
  ood : Nat
  qn : Nat
  cn : Nat
  cbn : Nat
  oods : Array OodCtx
  conss : Array ConsCtx
  deriving Inhabited

abbrev LM := StateT LState TM

def lift {α} (x : TM α) : LM α := StateT.lift x

def nextMsg (p : LigeritoProof) : LM (F256 × F256) := do
  let st ← get
  match p.messages[st.msg]? with
  | none => throw "ligerito: sumcheck messages exhausted"
  | some m =>
    set { st with msg := st.msg + 1 }
    lift (absorbK m.1)
    lift (absorbK m.2)
    return m

def nextOod (p : LigeritoProof) : LM F128 := do
  let st ← get
  match p.oodValues[st.ood]? with
  | none => throw "ligerito: ood_values exhausted"
  | some v => set { st with ood := st.ood + 1 }; return v

def nonceFrom (xs : Array UInt64) (i : Nat) (what : String) : LM UInt64 :=
  match xs[i]? with
  | none => throw s!"ligerito: {what} nonces exhausted"
  | some v => pure v

def claimPow (p : LigeritoProof) : LM Unit := do
  let st ← get
  let n ← nonceFrom p.claimNonces st.cbn "claim"
  set { st with cbn := st.cbn + 1 }
  lift (pow n)

def challenge (p : LigeritoProof) : LM F256 := do
  let r ← lift sampleK
  modify fun st => { st with s := st.q.eval r }
  let m ← nextMsg p
  modify fun st => { st with q := Poly.ofMsg m st.s }
  return r

/-- An out-of-domain sample with an intro message (levels ≥ 1). -/
def oodIntro (p : LigeritoProof) (dim : Nat) (split : Nat) : LM Unit := do
  let z ← lift (sampleVec dim)
  let y ← nextOod p
  lift (absorbF y)
  let m ← nextMsg p
  let intro := Poly.ofMsg m (F256.ofF y)
  claimPow p
  let b ← lift sample
  modify fun st => { st with q := st.q.addScaled intro b, s := st.s + (F256.ofF y).smul b, oods := st.oods.push ⟨z, b, some split⟩ }

/-- Queries on a level, its consistency coins, its Merkle opening, and the enforced sum `e`. -/
def queryLevel (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray) (rows : Array (Array F128)) (sibs : Array ByteArray)
    (salts : Array ByteArray) (fold : Array F256) : LM (Array Nat × Array F128 × F256) := do
  let st ← get
  let qnonce ← nonceFrom p.queryNonces st.qn "query"
  let cnonce ← nonceFrom p.consistencyNonces st.cn "consistency"
  set { st with qn := st.qn + 1, cn := st.cn + 1 }
  lift (pow qnonce)
  let coins ← lift (sampleVec lv.q)
  let pos := positions lv coins
  lift (pow cnonce)
  let alpha ← lift (sampleVec (Nat.log2 (lv.q - 1) + 1))
  match merkleCheck ms cap lv.cap lv.d pos rows sibs salts (2 ^ lv.lanes) with
  | .ok () => pure ()
  | .error e => throw e
  let ea := eqTable alpha
  let ef := eqTableK fold
  let mut e : F256 := 0
  for j in [0:lv.q] do
    let row := rows[j]!
    let mut acc : F256 := 0
    for l in [0:row.size] do acc := acc + (ef[l]!).smul (row[l]!)
    e := e + acc.smul (ea[j]!)
  return (pos, alpha, e)

/-- `1 + c·(1 + u)` (§13.6). -/
def cff (c : F256) : F256 := 1 + c * (1 + F256.u)

/-- §13.3–§13.6. `basisAt orig ext` is `b̂(orig ‖ bits(y))` for `y < 2^ext`. -/
def ligerito (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) : TM Unit := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  check (p.initialCap == cap) "ligerito: initial_cap differs from the commitment's cap"
  label "flock-ligerito-basis-f256-split-v0"
  absorbF T
  absorbBytes (cap.foldl (· ++ ·) ByteArray.empty)
  let prog : LM Unit := do
    -- level 0 out-of-domain sample (no message)
    let z ← lift (sampleVec n)
    let y ← nextOod p
    lift (absorbF y)
    claimPow p
    let b ← lift sample
    modify fun st => { st with s := st.s + (F256.ofF y).smul b, oods := st.oods.push ⟨z, b, none⟩ }
    let m0 ← nextMsg p
    modify fun st => { st with q := Poly.ofMsg m0 st.s }
    let mut initCh : Array F256 := #[]
    for _ in [0:sch.k0] do
      initCh := initCh.push (← challenge p)
    let mut splitDim := n - sch.k0 + 1
    let cap1 := p.recursiveCaps[0]!
    lift (absorbBytes (cap1.foldl (· ++ ·) ByteArray.empty))
    oodIntro p splitDim 0
    oodIntro p splitDim 0
    let (pos0, al0, e0) ← queryLevel ms p (lv 0) cap p.initialProof.rows p.initialProof.siblings p.initialProof.salts initCh
    let m ← nextMsg p
    claimPow p
    let b0 ← lift sample
    modify fun st => { st with q := st.q.addScaled (Poly.ofMsg m e0) b0, s := st.s + e0.smul b0,
                               conss := st.conss.push ⟨n - sch.k0, pos0, al0, b0, 0⟩ }
    let mut prevCap := cap1
    let mut levels : Array (Array F256) := #[]
    let mut ext := 0
    for i in [0:R] do
      let k := 4
      if splitDim < k then throw "ligerito: split_dim < k"
      let mut chs : Array F256 := #[]
      for _ in [0:k] do
        chs := chs.push (← challenge p)
      levels := levels.push chs
      ext := splitDim - k
      if i + 1 < R then
        splitDim := ext + 1
        let capN := p.recursiveCaps[i + 1]!
        lift (absorbBytes (capN.foldl (· ++ ·) ByteArray.empty))
        oodIntro p splitDim (i + 1)
        oodIntro p splitDim (i + 1)
        let pr := p.recursiveProofs[i]!
        let (pos, al, e) ← queryLevel ms p (lv (i + 1)) prevCap pr.rows pr.siblings pr.salts chs
        let mm ← nextMsg p
        claimPow p
        let bb ← lift sample
        modify fun st => { st with q := st.q.addScaled (Poly.ofMsg mm e) bb, s := st.s + e.smul bb,
                                   conss := st.conss.push ⟨ext, pos, al, bb, i + 1⟩ }
        prevCap := capN
        if splitDim < 4 then throw "ligerito: split_dim < lanes"
      else
        let st ← get
        if st.msg != p.messages.size then throw "ligerito: leftover sumcheck messages"
        if p.finalProof.yr.size != 2 ^ (ext + 1) then throw "ligerito: |yr|"
        for w in p.finalProof.yr do lift (absorbF w)
        let (pos, al, e) ← queryLevel ms p (lv R) prevCap p.finalProof.rows p.finalProof.siblings p.finalProof.salts chs
        claimPow p
        let bb ← lift sample
        modify fun st => { st with s := st.s + e.smul bb, conss := st.conss.push ⟨ext, pos, al, bb, R⟩ }
    -- §13.6 final check
    let st ← get
    let orig := levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh
    if orig.size + ext != n then throw "ligerito: |orig| + ext ≠ n"
    let heads := levels.map (fun l => l[0]!)
    let prodFrom := fun (a : Nat) => (heads.extract a heads.size).foldl (fun acc c => acc * cff c) (1 : F256)
    let pi0 := prodFrom 0
    let bres := basisAt orig ext
    if bres.size != 2 ^ ext then throw "ligerito: basis residual length"
    let mut res : Array F256 := bres.map (· * pi0)
    let sv := svTable (n + 1)
    for o in st.oods do
      let (fixed, scale) := match o.split with
        | none => (orig, pi0)
        | some s => (levels.extract s levels.size |>.foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFrom (s + 1))
      if o.z.size != fixed.size + ext then throw "ligerito: ood dimension"
      let head := (Array.range fixed.size).foldl (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fixed[t]!)) (1 : F256)
      let c := (head * scale).smul o.beta
      res := (Array.range (2 ^ ext)).map fun y =>
        let tail := (Array.range ext).foldl (fun acc j =>
          let zj := o.z[fixed.size + j]!
          acc * (if (y >>> j) % 2 == 1 then zj else (1 : F128) + zj)) (1 : F128)
        res[y]! + c.smul tail
    for cs in st.conss do
      let fixed := (levels.extract cs.start levels.size).foldl (fun acc l => acc ++ l.extract 1 l.size) #[]
      if fixed.size + ext != cs.cols then throw "ligerito: consistency dimension"
      let ea := eqTable cs.alpha
      let scale := prodFrom cs.start
      let mut add : Array F256 := Array.replicate (2 ^ ext) 0
      for j in [0:cs.pos.size] do
        let wh := whats sv cs.pos[j]! cs.cols
        let pj := (Array.range fixed.size).foldl (fun acc i => acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + wh[i]!))) (1 : F256)
        let coef := pj.smul (ea[j]!)
        add := (Array.range (2 ^ ext)).map fun y =>
          let tj := (Array.range ext).foldl (fun acc i => if (y >>> i) % 2 == 1 then acc * wh[fixed.size + i]! else acc) (1 : F128)
          add[y]! + coef.smul tj
      res := (Array.range (2 ^ ext)).map fun y => res[y]! + (add[y]! * scale).smul cs.beta
    let yr := p.finalProof.yr
    let lhs := (Array.range (2 ^ ext)).foldl (fun acc y => acc + res[y]! * ⟨yr[2 * y]!, yr[2 * y + 1]!⟩) (0 : F256)
    if lhs != st.s then throw "ligerito: final check failed"
    if st.ood != p.oodValues.size || st.qn != p.queryNonces.size || st.cn != p.consistencyNonces.size
        || st.cbn != p.claimNonces.size then throw "ligerito: leftover proof lists"
  let init : LState := ⟨F256.ofF T, ⟨0, 0, 0⟩, 0, 0, 0, 0, 0, #[], #[]⟩
  let _ ← prog.run init
  return ()

end Flock
