/-
The `--zk` verifier (`Flock.Zk`) over the hash oracle (`Flock.With.Hashes`): `Zk.verifyWithM` is `Flock.Zk.verify` with
every hash call the oracle's, the inner proof's `τ` commitment and its pads' Merkle paths included, and `Zk.verifyWith h`
runs it in `V` with SHA-512 replaced by `h`. The copies keep the originals' text.
-/
import Flock.With.Verify
import Flock.Zk
import Flock.JsonParse

namespace Flock.Zk

open Transcript

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- `replay` with the oracle's coins. -/
def replayWith (hs : Hashes M) (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) : TMW M Replayed := do
  let s := lay.len
  let m := st.m
  let zc := p.zerocheck
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes st.digest)
  liftTM (absorbBytes (p.cap.foldl (· ++ ·) ByteArray.empty))
  liftTM (label "flock-zerocheck-v0")
  liftTM (label LABEL_PADS)
  liftTM (absorbBytes root)
  let _rSkip ← sampleVecWith hs 6
  let rOuter ← sampleVecWith hs (m - 13)
  let r := pinned ++ rOuter
  liftTM (absorbSlice zc.round1ab)
  liftTM (absorbSlice zc.round1c)
  let z ← sampleWith hs
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
    liftTM (absorbF m1)
    liftTM (absorbF minf)
    let rr ← sampleWith hs
    rho := rho.push rr
    running := ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))
  let a := Lin.masked zc.finalA lay.fa s
  let b := Lin.masked zc.finalB lay.fb s
  -- a · b = a'b' + a' h_b + b' h_a + h_ab
  let prod : Lin := ⟨zc.finalA * zc.finalB, (((Array.replicate s 0).set! lay.fb zc.finalA).set! lay.fa zc.finalB).set! lay.hab 1⟩
  cons := cons.push ("zerocheck final a·b", running.add prod)
  liftTM (absorbF zc.finalA)
  liftTM (absorbF zc.finalB)
  -- lincheck
  let kLog := st.kLog
  liftTM (label "flock-lincheck-v0")
  let alpha ← sampleWith hs
  let beta ← sampleWith hs
  running := ((a.scale alpha).add b).addConst beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (e1v, einfv) := p.lincheck.rounds[j]!
    liftTM (absorbF e1v)
    liftTM (absorbF einfv)
    let rr ← sampleWith hs
    let e1 := Lin.masked e1v (lay.lc j 0) s
    let einf := Lin.masked einfv (lay.lc j 1) s
    let e0 := running.add e1
    let c1 := (e0.add e1).add einf
    running := ((einf.scale (rr * rr)).add (c1.scale rr)).add e0
    ts := ts.push rr
  let combv ← match st.fold.folded alpha beta (skipWeights z) (rho.extract 0 (kLog - 6)) ts with
    | .ok v => pure v
    | .error err => throw err
  liftTM (check (combv.size == 64) "lincheck: the circuit fold has the wrong width")
  liftTM (absorbSlice p.lincheck.zPartial)
  let zp := (Array.range 64).map fun i => Lin.masked p.lincheck.zPartial[i]! (lay.zp i) s
  cons := cons.push ("lincheck final", running.add (Lin.combo combv zp s))
  let rPrime ← sampleWith hs
  let w := Lin.combo (skipWeights rPrime) zp s
  let zcOut : ZcOut := { z, rho, rRest := r, vA := 0, vB := 0, vC := 0 }
  let lcOut : LcOut := { w := 0, rPrime, rIn := ts.reverse }
  return { claims := abcClaims m kLog zcOut lcOut, abValue := w, cValue := fc, cons }

/-- `tauCommitment` through the oracle. -/
def tauCommitmentWith (hs : Hashes M) (tau : F128) (salt : ByteArray) : M ByteArray := do
  Hm96.Default512.leafWith hs (← hs.sha512 (TAU_TAG ++ tau.toBytes)) salt

/-- `verifyInner` with the oracle's coins, `τ` commitment and Merkle paths. -/
def verifyInnerWith (hs : Hashes M) (lay : PadLayout) (pr : InnerProof) (finalC : F128) (cons : Array (String × Lin)) : TMW M Unit := do
  let msg := lay.len + K_PAD
  let n := msg.nextPowerOfTwo * 2 ^ INNER_LOG_INV_RATE
  liftTM (check (pr.y.size == msg && pr.opened.size == Q && pr.salts.size == Q && pr.paths.size == Q && pr.root.size == 64
    && pr.tauCommit.size == 64 && pr.tauSalt.size == TAU_SALT_BYTES) "zk inner: malformed proof")
  liftTM (label LABEL_INNER)
  liftTM (absorbF finalC)
  let eps ← sampleWith hs
  liftTM (absorbSlice #[pr.rho, pr.sigma])
  let cons := cons ++ triple lay eps pr.rho pr.sigma
  let gammas ← sampleVecWith hs cons.size
  let l := Lin.combo gammas (cons.map (·.2)) lay.len
  liftTM (absorbBytes pr.tauCommit)
  let beta ← sampleWith hs
  liftTM (check (!beta.isZero) "zk inner: degenerate coin β = 0")
  liftTM (absorbSlice pr.y)
  let qs := (← sampleVecWith hs Q).map fun w => w.lo.toNat % n
  let tc ← (tauCommitmentWith hs pr.tau pr.tauSalt : M ByteArray)
  liftTM (check (tc == pr.tauCommit) "zk inner: τ does not open its commitment")
  liftTM (check (dot l.a pr.y == l.c + beta * pr.tau) "zk inner: the batched constraint fails (⟨c, Y⟩ ≠ t + β τ)")
  let depth := Nat.log2 n
  for k in [0:Q] do
    let j := qs[k]!
    let (a, b) := pr.opened[k]!
    liftTM (check (codeAt pr.y j == a + beta * b) s!"zk inner: column {j} is not consistent with Y")
    let path := pr.paths[k]!
    liftTM (check (path.size == depth) "zk inner: a path of the wrong depth")
    let x ← hs.sha512 (a.toBytes ++ b.toBytes)
    let leaf ← (Hm96.Default512.leafWith hs x pr.salts[k]! : M ByteArray)
    let top ← (climbWith (MerkleOps.hm96With hs) leaf j path.toList : M ByteArray)
    liftTM (check (top == pr.root) s!"zk inner: column {j}'s path does not reach the pads' root")

/-- `queryLevel0` with the oracle's coins and Merkle hashes. -/
def queryLevel0With (hs : Hashes M) (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray) (fold : Array F256) :
    LMW M (Array Nat × Array F128 × F256) := do
  let st ← get
  let qnonce ← liftLM (nonceFrom p.queryNonces st.qn "query")
  let cnonce ← liftLM (nonceFrom p.consistencyNonces st.cn "consistency")
  set { st with qn := st.qn + 1, cn := st.cn + 1 }
  liftLM (lift (pow qnonce))
  let coins ← liftW (sampleVecWith hs lv.q)
  let pos := positions lv coins
  liftLM (lift (pow cnonce))
  let alpha ← liftW (sampleVecWith hs (Nat.log2 (lv.q - 1) + 1))
  let rows := p.initialProof.rows
  (merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d pos rows p.initialProof.siblings p.initialProof.salts
    (2 ^ lv.lanes + EXTRA_LANES) : M Unit)
  let ea := eqTable alpha
  let ef := eqTableK fold
  let mut e : F256 := 0
  for j in [0:lv.q] do
    let row := rows[j]!
    let mut acc : F256 := 0
    for l in [0:2 ^ lv.lanes] do acc := acc + (ef[l]!).smul (row[l]!)
    e := e + acc.smul (ea[j]!)
  return (pos, alpha, e)

/-- `Zk.ligerito` with the oracle's coins and Merkle hashes. -/
def ligeritoWith (hs : Hashes M) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) (zl : Level0) : TMW M L0Rec := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  liftTM (check (p.initialCap == cap) "ligerito: initial_cap differs from the commitment's cap")
  liftTM (check (zl.msgs.ybar.size == zl.tPad && zl.rep < EXTRA_LANES / 2) "ligerito: level 0's hiding messages have the wrong shape")
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (cap.foldl (· ++ ·) ByteArray.empty))
  liftTM (label LABEL_OPEN)
  let prog : LMW M L0Rec := do
    -- level 0 out-of-domain sample (no message)
    let z ← liftW (sampleVecWith hs n)
    let y ← liftLM (nextOod p)
    liftLM (lift (absorbF y))
    liftLM (claimPow p)
    let b ← liftW (sampleWith hs)
    modify fun st => { st with s := st.s + (F256.ofF y).smul b, oods := st.oods.push ⟨z, b, none⟩ }
    let m0 ← liftLM (nextMsg p)
    modify fun st => { st with q := Poly.ofMsg m0 st.s }
    let mut initCh : Array F256 := #[]
    let mut rho : F256 × F256 := (0, 0)
    for j in [0:sch.k0] do
      let r ← liftW (sampleKWith hs)
      initCh := initCh.push r
      modify fun st => { st with s := st.q.eval r }
      if j + 1 == sch.k0 then
        -- the lane phase's claim is masked: the pair's sums, its coins, then the blinded claim takes over
        liftLM (lift (absorbK zl.msgs.e0))
        liftLM (lift (absorbK zl.msgs.e1))
        rho := (← liftW (sampleKWith hs), ← liftW (sampleKWith hs))
        if !rhoIndependent rho then throw "ligerito: ρ_0, ρ_1 dependent over F128"
        liftLM (lift (label LABEL_CLOSE))
        liftLM (lift (absorbK zl.msgs.tPrime))
        liftLM (lift (absorbSlice (zl.msgs.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[])))
        modify fun st => { st with s := zl.msgs.tPrime }
      let m ← liftLM (nextMsg p)
      modify fun st => { st with q := Poly.ofMsg m st.s }
    let mut splitDim := n - sch.k0 + 1
    let cap1 := p.recursiveCaps[0]!
    liftLM (lift (absorbBytes (cap1.foldl (· ++ ·) ByteArray.empty)))
    oodIntroWith hs p splitDim 0
    oodIntroWith hs p splitDim 0
    let lanes := 2 ^ (lv 0).lanes
    let (pos0, al0, e0) ← queryLevel0With hs ms p (lv 0) cap initCh
    -- level 0's rows also carry rep `r`'s pair and the padding's codeword
    let sv0 := svTable ((lv 0).cols + 1)
    let ea0 := eqTable al0
    let mut extra : F256 := 0
    for j in [0:pos0.size] do
      let row := p.initialProof.rows[j]!
      let pad := padAt (whats sv0 pos0[j]! ((lv 0).cols + 1)) (lv 0).cols zl.msgs.ybar
      extra := extra + (rho.1.smul row[lanes + 2 * zl.rep]! + rho.2.smul row[lanes + 2 * zl.rep + 1]! + pad).smul ea0[j]!
    let e0 := e0 + extra
    let m ← liftLM (nextMsg p)
    liftLM (claimPow p)
    let b0 ← liftW (sampleWith hs)
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
        chs := chs.push (← challengeWith hs p)
      levels := levels.push chs
      ext := splitDim - k
      if i + 1 < R then
        splitDim := ext + 1
        let capN := p.recursiveCaps[i + 1]!
        liftLM (lift (absorbBytes (capN.foldl (· ++ ·) ByteArray.empty)))
        oodIntroWith hs p splitDim (i + 1)
        oodIntroWith hs p splitDim (i + 1)
        let pr := p.recursiveProofs[i]!
        let (pos, al, e) ← queryLevelWith hs ms p (lv (i + 1)) prevCap pr.rows pr.siblings pr.salts chs
        let mm ← liftLM (nextMsg p)
        liftLM (claimPow p)
        let bb ← liftW (sampleWith hs)
        modify fun st => { st with q := st.q.addScaled (Poly.ofMsg mm e) bb, s := st.s + e.smul bb,
                                   conss := st.conss.push ⟨ext, pos, al, bb, i + 1⟩ }
        prevCap := capN
        if splitDim < 4 then throw "ligerito: split_dim < lanes"
      else
        let st ← get
        if st.msg != p.messages.size then throw "ligerito: leftover sumcheck messages"
        if p.finalProof.yr.size != 2 ^ (ext + 1) then throw "ligerito: |yr|"
        for w in p.finalProof.yr do liftLM (lift (absorbF w))
        let (pos, al, e) ← queryLevelWith hs ms p (lv R) prevCap p.finalProof.rows p.finalProof.siblings p.finalProof.salts chs
        liftLM (claimPow p)
        let bb ← liftW (sampleWith hs)
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

/-- `Zk.verifyRep` with the oracle's hashes. -/
def verifyRepWith (hs : Hashes M) (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : Nat) (sess : RunRecord) (i r : Nat) (bytes : ByteArray) :
    M ByteArray := do
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
  let prog : TMW M Unit := do
    let rp ← replayWith hs st lay p zp.inner.root
    let ab := rp.claims[0]!
    let c := rp.claims[1]!
    let ccAb := claimCheck ab p.ringSwitches[0]!
    let ccC := claimCheck c p.ringSwitches[1]!
    let (T, sws) ← ringSwitchWith hs st.m (#[{ ab with value := ccAb }, { c with value := ccC }] ++ extras) p.ringSwitches
    let rec ← ligeritoWith hs st.merkle sch T p.cap p.ligerito (basis sws) ⟨tPad, r, zp.level0⟩
    let lg ← match ligeritoConstraints lay rec p.ligerito zp.level0 with
      | .ok v => pure v
      | .error e => throw e
    verifyInnerWith hs lay zp.inner p.zerocheck.finalC
      (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lg)
  let ((), tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- `Zk.verify` with every hash call the oracle's: `.ok ()` is Accept. -/
def verifyWithM (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  if spec.coinTree.isNone then throw "zk: the session's coins are not the keyed coin tree"
  let sess ← Record.decodeWith hs spec record
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => schedule st.m
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (← spec.dgWith hs b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, (sch, tPad)) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch tPad sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch tPad sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

/-- `Zk.verifyWithM` at the record's parsed value `j` (`Record.decodeJWith`); at a record that parses to `j` it is
`Zk.verifyWithM` (`verifyWithM_of_parse`). -/
def verifyJWithM (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json) (proofs : Array ByteArray) :
    M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  if spec.coinTree.isNone then throw "zk: the session's coins are not the keyed coin tree"
  let sess ← Record.decodeJWith hs spec j
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => schedule st.m
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (← spec.dgWith hs b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, (sch, tPad)) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch tPad sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch tPad sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

theorem verifyWithM_of_parse [LawfulMonad M] (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup)
    {record : String} {j : Lean.Json} (proofs : Array ByteArray) (h : JsonParse.parse record = .ok j) :
    verifyWithM hs spec tabs record proofs = verifyJWithM hs spec tabs j proofs := by
  simp only [verifyWithM, verifyJWithM, Record.decodeWith_of_parse hs spec h]

/-- The `--zk` verifier with SHA-512 replaced by `h`, every other hash the deployed one; `Zk.verifyWith Sha512.hash` is
`Zk.verify` (`Flock.Zk.verify_eq_verifyWith`). -/
def verifyWith (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : V Unit :=
  verifyWithM (Hashes.ofSha512 h) spec tabs record proofs

end Flock.Zk
