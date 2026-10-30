/-
Live PROTOCOL.md §7 (`--zk`): Ligerito with level 0's hiding (VEIL Figs. 8–9), as the verifier sees it. Every lane's message
is padded by `(Ŵ_n + c)·μ` with `t_pad` uniform coefficients `μ` and `c = x`; four padded uniform lanes follow the witness's,
a pair per rep. Rep `r` folds its own pair `ρ_0 R_{2r} + ρ_1 R_{2r+1}` after its lane rounds, sends the pair's sums `e`
before `ρ` and the blinded claim `T'` and the padding's combination `ybar` after it, and its level-0 rows are checked against
`ybar`'s codeword. The lane phase's claim is masked, so its checks are not made here: `ligerito` returns what `Flock.Zk`
turns into constraints (`L0Rec`), and from `T'` on it is §13's verifier.
-/
import Flock.Ligerito
import Flock.ZkProof

namespace Flock.Zk

open Transcript

def LABEL_OPEN : String := "verity-flock-zk-l0-open-v0"
def LABEL_CLOSE : String := "verity-flock-zk-l0-close-v0"
/-- The padding polynomial's constant `c` (the element `x`, outside GF(2)). -/
def PAD_C : F128 := ⟨2, 0⟩

/-- What a hiding level 0 hands the Ligerito verifier: the padding per lane, the rep whose pair the rows are checked against,
and the proof's hiding messages. -/
structure Level0 where
  tPad : Nat
  rep : Nat
  msgs : Level0Msgs
  deriving Inhabited

/-- The lane phase as the verifier saw it (upstream's `Level0ZkRecord`): the target, `(β, y)` per level-0 OOD sample (`y` as
sent), the lane rounds' challenges and the pair's coins `ρ`. -/
structure L0Rec where
  target : F128
  ood : Array (F128 × F128)
  challenges : Array F256
  rho : F256 × F256
  deriving Inhabited

/-- `ρ_0 R_0 + ρ_1 R_1` is uniform on K for uniform F lanes exactly when `ρ_0, ρ_1` are independent over F. -/
def rhoIndependent (rho : F256 × F256) : Bool := !(rho.1.c0 * rho.2.c1 + rho.1.c1 * rho.2.c0).isZero

/-- The padding's part of the folded level-0 row at position `q`, `Enc(c·ybar ‖ ybar)(q)` coordinate by coordinate: with
`Ŵ_j = Ŵ_j(ω_q)` (`ws`, `j ≤ logMsg`) and `X_i = Π_{j ∈ bits(i)} Ŵ_j` the novel basis, `Σ_i ybar_i X_i · (c + Ŵ_{logMsg})`
(`X_{l + i} = X_i Ŵ_{logMsg}` for `i < l = 2^logMsg`). -/
def padAt (ws : Array F128) (logMsg : Nat) (ybar : Array F256) : F256 := Id.run do
  let mut xs : Array F128 := #[1]
  let mut acc : F256 := 0
  for i in [0:ybar.size] do
    if i > 0 then
      let top := Nat.log2 i
      xs := xs.push (xs[i - 2 ^ top]! * ws[top]!)
    acc := acc + (ybar[i]!).smul xs[i]!
  return acc.smul (PAD_C + ws[logMsg]!)

/-- Level 0's queries (`queryLevel`) on rows of `2^lanes + EXTRA_LANES` elements: the enforced sum folds the first `2^lanes`. -/
def queryLevel0 (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray) (fold : Array F256) :
    LM (Array Nat × Array F128 × F256) := do
  let st ← get
  let qnonce ← nonceFrom p.queryNonces st.qn "query"
  let cnonce ← nonceFrom p.consistencyNonces st.cn "consistency"
  set { st with qn := st.qn + 1, cn := st.cn + 1 }
  lift (pow qnonce)
  let coins ← lift (sampleVec lv.q)
  let pos := positions lv coins
  lift (pow cnonce)
  let alpha ← lift (sampleVec (Nat.log2 (lv.q - 1) + 1))
  let rows := p.initialProof.rows
  match merkleCheck ms cap lv.cap lv.d pos rows p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + EXTRA_LANES) with
  | .ok () => pure ()
  | .error e => throw e
  let ea := eqTable alpha
  let ef := eqTableK fold
  let mut e : F256 := 0
  for j in [0:lv.q] do
    let row := rows[j]!
    let mut acc : F256 := 0
    for l in [0:2 ^ lv.lanes] do acc := acc + (ef[l]!).smul (row[l]!)
    e := e + acc.smul (ea[j]!)
  return (pos, alpha, e)

/-- §13.3–§13.6 with level 0's hiding `zl`: `Flock.ligerito` but for the lane phase's tail (the pair's sums, `ρ`, `T'`,
`ybar`), level 0's wider rows and their extra terms (rep `zl.rep`'s pair and the padding's codeword). Returns the lane phase's
record for the constraints. -/
def ligerito (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) (zl : Level0) : TM L0Rec := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  check (p.initialCap == cap) "ligerito: initial_cap differs from the commitment's cap"
  check (zl.msgs.ybar.size == zl.tPad && zl.rep < EXTRA_LANES / 2) "ligerito: level 0's hiding messages have the wrong shape"
  label "flock-ligerito-basis-f256-split-v0"
  absorbF T
  absorbBytes (cap.foldl (· ++ ·) ByteArray.empty)
  label LABEL_OPEN
  let prog : LM L0Rec := do
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
    let mut rho : F256 × F256 := (0, 0)
    for j in [0:sch.k0] do
      let r ← lift sampleK
      initCh := initCh.push r
      modify fun st => { st with s := st.q.eval r }
      if j + 1 == sch.k0 then
        -- the lane phase's claim is masked: the pair's sums, its coins, then the blinded claim takes over
        lift (absorbK zl.msgs.e0)
        lift (absorbK zl.msgs.e1)
        rho := (← lift sampleK, ← lift sampleK)
        if !rhoIndependent rho then throw "ligerito: ρ_0, ρ_1 dependent over F128"
        lift (label LABEL_CLOSE)
        lift (absorbK zl.msgs.tPrime)
        lift (absorbSlice (zl.msgs.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]))
        modify fun st => { st with s := zl.msgs.tPrime }
      let m ← nextMsg p
      modify fun st => { st with q := Poly.ofMsg m st.s }
    let mut splitDim := n - sch.k0 + 1
    let cap1 := p.recursiveCaps[0]!
    lift (absorbBytes (cap1.foldl (· ++ ·) ByteArray.empty))
    oodIntro p splitDim 0
    oodIntro p splitDim 0
    let lanes := 2 ^ (lv 0).lanes
    let (pos0, al0, e0) ← queryLevel0 ms p (lv 0) cap initCh
    -- level 0's rows also carry rep `r`'s pair and the padding's codeword
    let sv0 := svTable ((lv 0).cols + 1)
    let ea0 := eqTable al0
    let mut extra : F256 := 0
    for j in [0:pos0.size] do
      let row := p.initialProof.rows[j]!
      let pad := padAt (whats sv0 pos0[j]! ((lv 0).cols + 1)) (lv 0).cols zl.msgs.ybar
      extra := extra + (rho.1.smul row[lanes + 2 * zl.rep]! + rho.2.smul row[lanes + 2 * zl.rep + 1]! + pad).smul ea0[j]!
    let e0 := e0 + extra
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
    -- level 0 has one out-of-domain sample, the first
    return { target := T, ood := #[((st.oods[0]!).beta, p.oodValues[0]!)], challenges := initCh, rho }
  let init : LState := ⟨F256.ofF T, ⟨0, 0, 0⟩, 0, 0, 0, 0, 0, #[], #[]⟩
  let (rec, _) ← prog.run init
  return rec

end Flock.Zk
