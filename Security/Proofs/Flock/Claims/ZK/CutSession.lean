import Proofs.Flock.Claims.Complete.Zk.Session
import Proofs.Flock.Claims.Complete.Sim

/-!
# The masked session from the values it sends

`sessionPZk` with every value it computes from the witness or the tape read instead from a source beside it: each
rep's masked values by pad index, its switched vectors, its restarted table `y₁'` and `ȳ`, levels ≥ 1's salts, the
triple's openings, `τ`'s commitment, `Y`, and the leaves of level 0's and the pads' trees (`RepSrc`, `TabSrc`). The
coins, the public data and every check on them are the session's: the extra claims, the circuit fold, the rank check,
the pair `ρ₀, ρ₁` and `β`. The inner proof's batched constraint on the pads is not checked (the honest pads pass it).
Levels ≥ 1 run as the session runs them, from `y₁'` and their salts. What only the final message reads (level 0's
opened rows and salts, the pads' opened columns and salts, `τ` and its salt) is read from the source too, and nothing
before it reads it.

* `zerocheckCut`, `lincheckCut`, `ringSwitchCut`, `ligeritoCut`, `innerCut`: the rep's phases.
* `commit0Cut`, `padsTreeCut`: level 0's commitment and a rep's pads commitment from their leaves.
* `piopPZk`, `piopCut`: a rep's zerocheck, lincheck, rank check and ring switch, the session's and from the source.
* `repCut`, `sessionCut`: a rep and the session from the source. `repMix k`: a rep whose first `k` phases are the
  session's and the rest the source's, the hybrids between `repCut` and `repPZk`.
* `asksCut coin n`: the sends that answer round `k` with `coin k` and stop right after the send at round `n`.
* `eraseC`, `eraseL`, `eraseIC`, `eraseZ`: a commitment, an opening, a pads commitment and a rep's proof with what only the
  final message reads forgotten.
-/

namespace FlockComplete

open Flock Transcript

/-- **What one rep's messages read beyond the coins and the public data**: the masked value `v + h_i` at each pad
index `i` (`masked`), claim `k`'s switched vector (`sv`), the restarted table `y₁'` and `ȳ`, levels ≥ 1's salts by level
and position, the triple's openings `ρ_in`, `σ_in`, `τ`'s commitment, `Y = (h ‖ pad_h) + β (μ ‖ pad_μ)` and the pads
tree's leaf at each tree position; for the final message, `τ`, its salt, and each pads column and its salt by tree
position. -/
structure RepSrc where
  masked : Nat → F128
  sv : Nat → Array F128
  y1 : Array F256
  ybar : Array F256
  hiSalt : Nat → Nat → ByteArray
  rhoIn : F128
  sigmaIn : F128
  tauCom : ByteArray
  y : Array F128
  padLeaf : Nat → ByteArray
  tau : F128
  tauSalt : ByteArray
  ch : Nat → F128
  cm : Nat → F128
  padSalt : Nat → ByteArray
  deriving Inhabited

/-- **What one table's messages read beyond the coins**: level 0's leaf at each tree position, and for the final
message its row and salt there; each rep's source. -/
structure TabSrc where
  leaf0 : Nat → ByteArray
  row0 : Nat → Array F128
  salt0 : Nat → ByteArray
  rep : Nat → RepSrc
  deriving Inhabited

/-! ## What only the final message reads -/

/-- A commitment without its rows and salts. -/
def eraseC (cm : Commit) : Commit :=
  { cm with rows := cm.rows.map fun _ => #[], salts := cm.salts.map fun _ => ByteArray.empty }

/-- An opening without its rows and salts. -/
def eraseLP (op : LevelProof) : LevelProof :=
  { op with rows := op.rows.map fun _ => #[], salts := op.salts.map fun _ => ByteArray.empty }

/-- A Ligerito proof without level 0's opened rows and salts. -/
def eraseL (lg : LigeritoProof) : LigeritoProof := { lg with initialProof := eraseLP lg.initialProof }

/-- A pads commitment without its columns and salts. -/
def eraseIC (ic : InnerCommit) : InnerCommit :=
  { ic with ch := ic.ch.map fun _ => 0, cm := ic.cm.map fun _ => 0, salts := ic.salts.map fun _ => ByteArray.empty }

/-- An inner proof without `τ`, its salt, and the opened columns and their salts. -/
def eraseIP (ip : Zk.InnerProof) : Zk.InnerProof :=
  { ip with tau := 0, tauSalt := ByteArray.empty, opened := ip.opened.map fun _ => (0, 0),
            salts := ip.salts.map fun _ => ByteArray.empty }

/-- A rep's proof without what only the final message reads. -/
def eraseZ (p : Zk.ZkProof) : Zk.ZkProof :=
  { p with proof := { p.proof with ligerito := eraseL p.proof.ligerito }, inner := eraseIP p.inner }

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-! ## A rep's phases -/

/-- **The masked zerocheck** sending the masked values `mk` at their pad indices. -/
def zerocheckCut (io : Asks M) (m : Nat) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : Nat → F128) :
    TMW M (ZerocheckProof × ZcOut) := do
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes digest)
  liftTM (absorbBytes cap)
  liftTM (label "flock-zerocheck-v0")
  liftTM (label Zk.LABEL_PADS)
  liftTM (absorbBytes root)
  let _rSkip ← sampleVecP io 6
  let rOuter ← sampleVecP io (m - 13)
  let r := pinned ++ rOuter
  let r1ab' := (Array.range 64).map fun i => mk (lay.r1ab i)
  let r1c' := (Array.range 64).map fun i => mk (lay.r1c i)
  liftTM (absorbSlice r1ab')
  liftTM (absorbSlice r1c')
  let z ← sampleP io
  let mut rounds : Array (F128 × F128) := #[]
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let g1' := mk (lay.ml i 0)
    let ginf' := mk (lay.ml i 1)
    liftTM (absorbF g1')
    liftTM (absorbF ginf')
    let rr ← sampleP io
    rho := rho.push rr
    rounds := rounds.push (g1', ginf')
  liftTM (absorbF (mk lay.fa))
  liftTM (absorbF (mk lay.fb))
  return (⟨r1ab', r1c', rounds, mk lay.fa, mk lay.fb, mk lay.fc⟩, { z, rho, rRest := r, vA := 0, vB := 0, vC := 0 })

/-- **The masked lincheck** sending the masked values `mk` at their pad indices; it stops where the circuit fold
fails, as the session does. -/
def lincheckCut (io : Asks M) (c : CircuitFold) (_m : Nat) (lay : Zk.PadLayout) (mk : Nat → F128) (zc : ZcOut) :
    TMW M (LincheckProof × LcOut) := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  liftTM (label "flock-lincheck-v0")
  let alpha ← sampleP io
  let _beta ← sampleP io
  let _full ← match c.fold alpha (skipWeights zc.z) rhoIn with
    | .ok v => pure v
    | .error err => throw err
  let mut rounds : Array (F128 × F128) := #[]
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let q1' := mk (lay.lc j 0)
    let qinf' := mk (lay.lc j 1)
    liftTM (absorbF q1')
    liftTM (absorbF qinf')
    let t ← sampleP io
    ts := ts.push t
    rounds := rounds.push (q1', qinf')
  let zp' := (Array.range 64).map fun i => mk (lay.zp i)
  liftTM (absorbSlice zp')
  let rPrime ← sampleP io
  return (⟨rounds, zp'⟩, { w := 0, rPrime, rIn := ts.reverse })

/-- **The ring switch** sending claim `k`'s switched vector `sv k`. -/
def ringSwitchCut (io : Asks M) (sv : Nat → Array F128) (claims : Array Claim) :
    TMW M (Array (Array F128) × F128 × Array Switched) := do
  let n := claims.size
  liftTM (label "flock-pcs-open-batch-v0")
  let mut ss : Array (Array F128) := #[]
  let mut betas : Array F128 := #[]
  let mut perClaim : Array (Array F128 × Array F128) := #[]
  for k in [0:n] do
    let c := claims[k]!
    let s := sv k
    liftTM (label "flock-ring-switch-v0")
    liftTM (absorbSlice s)
    let rpp ← sampleVecP io 7
    let t := (Array.range 128).map fun u =>
      (Array.range 128).foldl (fun acc v => if (s[v]!).bit u then acc + F128.monomial v else acc) (0 : F128)
    betas := betas.push (dot (eqTable rpp) t)
    perClaim := perClaim.push (rpp, c.x.extract 1 c.x.size)
    ss := ss.push s
  let gamma ← sampleVecP io n
  let T := dot gamma betas
  let sws := (Array.range n).map fun k => { gamma := gamma[k]!, rpp := (perClaim[k]!).1, z := (perClaim[k]!).2 : Switched }
  return (ss, T, sws)

/-- **A rep's zerocheck, lincheck, rank check and ring switch**, as `repPZk` runs them before Ligerito. -/
def piopPZk (io : Asks M) (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List Nat) :
    TMW M (ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched) := do
  let (zp, zc) ← zerocheckPZk io st.m st.digest cap root lay h tr
  let (lp, lc) ← lincheckPZk io st.fold st.m lay h tr zc
  rankCheckP st.m st.kLog prior words
  let (ss, T, sws) ← ringSwitchP io tr (abcClaims st.m st.kLog zc lc ++ extras)
  return (zp, lp, ss, T, sws)

/-- **`piopPZk` from the source**: the masked values and switched vectors read from `src`. -/
def piopCut (io : Asks M) (st : Setup) (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List Nat) :
    TMW M (ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched) := do
  let (zp, zc) ← zerocheckCut io st.m st.digest cap root lay src.masked
  let (lp, lc) ← lincheckCut io st.fold st.m lay src.masked zc
  rankCheckP st.m st.kLog prior words
  let (ss, T, sws) ← ringSwitchCut io src.sv (abcClaims st.m st.kLog zc lc ++ extras)
  return (zp, lp, ss, T, sws)

/-- **The masked Ligerito opening from the source**: level 0's out-of-domain value, lane messages and pair's sums are the
masked values at `lgOod`, `lgMsg`, `lgE`; after the pair's coins (refused where dependent, as the session does) the
restart sends `T' = ⟨W, y₁'⟩`, `ȳ` and goes on from `y₁'`; levels ≥ 1 are committed under `src.hiSalt`; every nonce is
`0`. -/
def ligeritoCut (io : Asks M) (hs : Hashes M) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (b : Array F128) (lay : Zk.PadLayout) (src : RepSrc) :
    TMW M (LigeritoProof × Zk.Level0Msgs × Zk.L0Rec) := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  let nonce : Nat → Nat → UInt64 := fun _ _ => 0
  let salt := src.hiSalt
  let mm : Nat → F256 × F256 := fun j =>
    (⟨src.masked (lay.lgMsg j 0), src.masked (lay.lgMsg j 1)⟩, ⟨src.masked (lay.lgMsg j 2), src.masked (lay.lgMsg j 3)⟩)
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (flat cm0.cap))
  liftTM (label Zk.LABEL_OPEN)
  let prog : LP M (Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof × Zk.Level0Msgs ×
      Array F256 × (F256 × F256) × F128 × F128) := do
    let z ← liftT (sampleVecP io n)
    let y := src.masked (lay.lgOod 0)
    modify fun st => { st with oods := st.oods.push y }
    liftT (liftTM (absorbF y))
    powP nonce 2
    let b0 ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (embedF (eqTable z)) b0 }
    sendMsg (mm 0)
    let mut initCh : Array F256 := #[]
    let mut rho : F256 × F256 := (0, 0)
    let mut zl : Zk.Level0Msgs := default
    for j in [0:sch.k0] do
      let r ← liftT (sampleKP io)
      initCh := initCh.push r
      modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
      if j + 1 == sch.k0 then
        let st ← get
        let e0 : F256 := ⟨src.masked (lay.lgE 0 0), src.masked (lay.lgE 0 1)⟩
        let e1 : F256 := ⟨src.masked (lay.lgE 1 0), src.masked (lay.lgE 1 1)⟩
        liftT (liftTM (absorbK e0))
        liftT (liftTM (absorbK e1))
        let rh0 ← liftT (sampleKP io)
        let rh1 ← liftT (sampleKP io)
        rho := (rh0, rh1)
        if !Zk.rhoIndependent rho then throw "PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128"
        liftT (liftTM (label Zk.LABEL_CLOSE))
        let tP := dotK st.w src.y1
        liftT (liftTM (absorbK tP))
        liftT (liftTM (absorbSlice (src.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[])))
        zl := ⟨e0, e1, tP, src.ybar⟩
        modify fun st => { st with tab := embedF (splitK src.y1), w := liftU st.w }
        let st ← get
        sendMsg (roundPoly st.tab st.w)
      else
        sendMsg (mm (j + 1))
    let mut splitDim := n - sch.k0 + 1
    let g1 := (← get).tab.map F256.c0
    let cm1 ← liftT (StateT.lift (commitP hs ms (lv 1) g1 (salt 1)))
    liftT (liftTM (absorbBytes (flat cm1.cap)))
    oodIntroP io nonce g1 splitDim
    oodIntroP io nonce g1 splitDim
    let (op0, v0) ← queryP io nonce ms (lv 0) cm0
    sendMsg (roundPoly (embedF g1) (liftU (embedF v0)))
    powP nonce 2
    let bq ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (liftU (embedF v0)) bq }
    let mut prevCm := cm1
    let mut caps : Array (Array ByteArray) := #[cm1.cap]
    let mut rproofs : Array LevelProof := #[]
    let mut fin : FinalProof := default
    for i in [0:R] do
      let last := i + 1 == R
      let mut f : Array F256 := #[]
      for j in [0:4] do
        if j + 1 == 4 then
          let r ← liftT (sampleKP io)
          modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
          f := (← get).tab
          modify fun st => { st with tab := embedF (splitK st.tab), w := liftU st.w }
          let st ← get
          sendMsg (roundPoly st.tab st.w)
        else
          let _ ← challengeP io false
      let ext := splitDim - 4
      if !last then
        splitDim := ext + 1
        let g := splitK f
        let cmN ← liftT (StateT.lift (commitP hs ms (lv (i + 2)) g (salt (i + 2))))
        liftT (liftTM (absorbBytes (flat cmN.cap)))
        caps := caps.push cmN.cap
        oodIntroP io nonce g splitDim
        oodIntroP io nonce g splitDim
        let (op, v) ← queryP io nonce ms (lv (i + 1)) prevCm
        rproofs := rproofs.push op
        sendMsg (roundPoly (embedF g) (liftU (embedF v)))
        powP nonce 2
        let bb ← liftT (sampleP io)
        modify fun st => { st with w := addScaled st.w (liftU (embedF v)) bb }
        prevCm := cmN
      else
        let yr := f.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]
        for x in yr do liftT (liftTM (absorbF x))
        let (op, _) ← queryP io nonce ms (lv R) prevCm
        fin := { yr, rows := op.rows, siblings := op.siblings, salts := op.salts }
        powP nonce 2
        let _ ← liftT (sampleP io)
    return (caps, rproofs, fin, op0, zl, initCh, rho, b0, y)
  let ((caps, rproofs, fin, op0, zl, initCh, rho, b0, y), st) ← prog.run { tab := #[], w := embedF b }
  return ({ initialCap := cm0.cap, initialProof := op0, recursiveCaps := caps, recursiveProofs := rproofs,
            finalProof := fin, messages := st.msgs, queryNonces := st.qn, oodValues := st.oods, claimNonces := st.cbn,
            consistencyNonces := st.cn }, zl, { target := T, ood := #[(b0, y)], challenges := initCh, rho })

/-- **The inner proof from the source**: the triple's openings, `τ`'s commitment and `Y` read from `src`, `β` refused at
`0` as the session does; the batched constraint is not checked. -/
def innerCut (io : Asks M) (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src : RepSrc)
    (cons : Array (String × Zk.Lin)) : TMW M Zk.InnerProof := do
  let n := ic.ch.size
  liftTM (label Zk.LABEL_INNER)
  liftTM (absorbF finalC)
  let eps ← sampleP io
  let rho := src.rhoIn
  let sigma := src.sigmaIn
  liftTM (absorbSlice #[rho, sigma])
  let cons := cons ++ Zk.triple lay eps rho sigma
  let _gammas ← sampleVecP io cons.size
  liftTM (absorbBytes src.tauCom)
  let beta ← sampleP io
  if beta.isZero then throw "PROVER-STOPPED: degenerate coin β = 0"
  liftTM (absorbSlice src.y)
  let qs := (← sampleVecP io Zk.Q).map fun w => w.lo.toNat % n
  return { root := ic.root, rho, sigma, tauCommit := src.tauCom, tauSalt := src.tauSalt, tau := src.tau, y := src.y,
           opened := qs.map fun j => (ic.ch[j]!, ic.cm[j]!), salts := qs.map fun j => ic.salts[j]!,
           paths := qs.map fun j => (Array.range (Nat.log2 n)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]! }

/-! ## The commitments from their leaves -/

/-- **Level 0's commitment from its leaves**: `commitRowsP`'s tree on the leaves `src.leaf0` by tree position, with the
rows and salts `src.row0`, `src.salt0`. -/
def commit0Cut (hs : Hashes M) (ms : MerkleScheme) (lv : Level) (src : TabSrc) : M Commit := do
  let ops := ms.opsWith hs
  let leaves := (Array.range (2 ^ lv.d)).map src.leaf0
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:lv.d - lv.cap] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => ops.node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨(Array.range (2 ^ lv.d)).map src.row0, (Array.range (2 ^ lv.d)).map src.salt0, layers, cur⟩

/-- **A rep's pads commitment from its leaves**: `innerCommitP`'s tree on the `n` leaves `src.padLeaf` by tree
position, with the columns and salts `src.ch`, `src.cm`, `src.padSalt`. -/
def padsTreeCut (hs : Hashes M) (n : Nat) (src : RepSrc) : M InnerCommit := do
  let leaves := (Array.range n).map src.padLeaf
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:Nat.log2 n] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => (MerkleOps.hm96With hs).node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨(Array.range n).map src.ch, (Array.range n).map src.cm, (Array.range n).map src.padSalt, layers, cur[0]!⟩

/-! ## A rep and the session -/

/-- **One masked rep from the source** (`repPZk`'s phases with `piopCut`, `ligeritoCut`, `innerCut`): the replay and
the constraints as the session builds them. -/
def repCut (io : Asks M) (hs : Hashes M) (st : Setup) (sch : Schedule) (_rep : Nat) (dom : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List Nat) :
    TMW M Zk.ZkProof := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let (zp, lp, ss, T, sws) ← piopCut io st lay src (flat cm0.cap) ic.root extras prior words
  let (lg, zl, rec) ← ligeritoCut io hs st.merkle sch T cm0 (basisVals sws (st.m - 7)) lay src
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  let inner ← innerCut io lay ic zp.finalC src
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc)
  return { proof := p, level0 := zl, inner }

/-- **A rep with its first `k` phases the session's and the rest from the source** (`k = 0`: none; `1`: the PIOP; `2`:
and Ligerito; `3`: and the inner proof), level 0's commitment `cm0` and the pads commitment `ic` the session's, and
what only the final message reads forgotten (`eraseC` before the source's Ligerito, `eraseL` after the session's,
`eraseZ` at the end). Between `repMix k` and `repMix (k + 1)` one phase changes. -/
def repMix (k : Nat) (io : Asks M) (hs : Hashes M) (st : Setup) (sch : Schedule) (tPad rep : Nat) (dom : ByteArray)
    (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64)
    (prior : Array (List F128)) (words : List Nat) (src : RepSrc) : TMW M Zk.ZkProof := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let (zp, lp, ss, T, sws) ← if k = 0 then piopCut io st lay src (flat cm0.cap) ic.root extras prior words
    else piopPZk io st lay pads.h tr (flat cm0.cap) ic.root extras prior words
  let (lg, zl, rec) ← if k ≤ 1 then ligeritoCut io hs st.merkle sch T (eraseC cm0) (basisVals sws (st.m - 7)) lay src
    else do
      let (lg, zl, rec) ← ligeritoPZk io hs st.merkle sch T cm0 t (basisVals sws (st.m - 7)) lay pads.h rep tPad extra
        lanePad salt nonce
      pure (eraseL lg, zl, rec)
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  let cons := rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++
    lgc
  let inner ← if k ≤ 2 then innerCut io lay ic zp.finalC src cons else innerP io hs lay pads ic zp.finalC cons
  return eraseZ { proof := p, level0 := zl, inner }

/-- **The masked session from the sources** (`sessionPZk`'s order): the `--zk` schedules, level 0's commitments from
their leaves, the link, the link's points, then the rep streams, each with its pads commitment from its leaves; the
record and the proof files. -/
def sessionCut (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List Nat))
    (src : Nat → TabSrc) : M (String × Array ByteArray) := do
  let schs ← tabs.mapM fun st => (monadLift (Zk.schedule st.m) : M (Schedule × Nat))
  let cms ← (Array.range tabs.size).mapM fun i =>
    commit0Cut hs tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)
  let rootB := cms.map fun c => flat c.cap
  let c1 ← io.link rootB (tabs.map fun _ => ByteArray.empty)
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let sch := schs[i]!.1
    let extras ← (monadLift (st.extra points) : M _)
    let lay := Zk.PadLayout.new st.m st.kLog sch.k0
    let ic ← padsTreeCut hs ((lay.len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE) ((src i).rep r)
    let dom := (spec.domain spec.tables[i]! r).toUTF8
    let (p, t) ← (repCut io hs st sch r dom cms[i]! extras ic ((src i).rep r) (priorOf st streams r) words[i]!).run
      (Transcript.init #[] dom)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockSoundness.Discharge.ZkView.encZk (shapeZk st sch extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

/-- **The sends that stop right after round `n`'s**: each send is recorded as `asksS` records it; round `k < n` is
answered by `coin k`, and the send at round `n` throws once recorded. -/
def asksCut (coin : Nat → Nat → F128) (n : Nat) : Asks PS where
  link rootB publics := do
    let l ← get
    set (l.push (FlockDeploy.linkBody rootB.toList publics.toList))
    if n ≤ l.size then throw "past the openings" else pure (coin l.size)
  round c := do
    let l ← get
    set (l.push (Claims.toMsg c))
    if n ≤ l.size then throw "past the openings" else pure (coin l.size)

/-- **`asksCut` reads only the coins before round `n`.** -/
theorem asksCut_congr {coin coin' : Nat → Nat → F128} {n : Nat} (h : ∀ k < n, coin k = coin' k) :
    asksCut coin n = asksCut coin' n := by
  unfold asksCut
  congr 1
  · funext rootB publics
    congr 1
    funext l
    by_cases hn : n ≤ l.size
    · simp only [hn, ↓reduceIte]
    · simp only [hn, ↓reduceIte, h l.size (by omega)]
  · funext c
    congr 1
    funext l
    by_cases hn : n ≤ l.size
    · simp only [hn, ↓reduceIte]
    · simp only [hn, ↓reduceIte, h l.size (by omega)]

end FlockComplete
