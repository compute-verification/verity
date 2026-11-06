/-
§3 and §9–§14 over the hash oracle (`Flock.With.Hashes`): `verifyWithM` is `Flock.verify` with every hash call the oracle's,
in a verdict monad `M`, and `verifyWith h` runs it in `V` with SHA-512 replaced by `h`. The copies keep the originals' text;
what hashes nothing is the compiled verifier's own, lifted (`liftTM`, `liftLM`).
-/
import Flock.With.Transcript
import Flock.Verify
import Flock.JsonParse

namespace Flock

open Transcript

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- `bindAndZerocheck` with the oracle's coins. -/
def bindAndZerocheckWith (hs : Hashes M) (m : Nat) (digest : ByteArray) (cap : ByteArray) (p : ZerocheckProof) :
    TMW M ZcOut := do
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes digest)
  liftTM (absorbBytes cap)
  liftTM (label "flock-zerocheck-v0")
  let _rSkip ← sampleVecWith hs 6
  let rOuter ← sampleVecWith hs (m - 13)
  let r := pinned ++ rOuter
  liftTM (absorbSlice p.round1ab)
  liftTM (absorbSlice p.round1c)
  let z ← sampleWith hs
  let pc := dot (lagrange nodesL den6Inv z) p.round1c
  liftTM (check (pc == p.finalC) "zerocheck: CEvalMismatch")
  let comb := (Array.range 64).map fun i => p.round1ab[i]! + p.round1c[i]!
  let pcomb := dot (combWeights z) comb
  let mut claim := pcomb + pc
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let (g1, ginf) := p.rounds[i]!
    let req := r[i]!
    let g0 := (claim + req * g1) * (1 + req).inv
    liftTM (absorbF g1)
    liftTM (absorbF ginf)
    let rr ← sampleWith hs
    rho := rho.push rr
    claim := g0 * (1 + rr) + g1 * rr + ginf * rr * (1 + rr)
  liftTM (check (claim == p.finalA * p.finalB) "zerocheck: SumcheckFinalFailed")
  liftTM (absorbF p.finalA)
  liftTM (absorbF p.finalB)
  return { z, rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC }

/-- `lincheck` with the oracle's coins. -/
def lincheckWith (hs : Hashes M) (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) : TMW M LcOut := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  liftTM (label "flock-lincheck-v0")
  let alpha ← sampleWith hs
  let beta ← sampleWith hs
  let mut claim := alpha * zc.vA + zc.vB + beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (q1, qinf) := p.rounds[j]!
    liftTM (absorbF q1)
    liftTM (absorbF qinf)
    let t ← sampleWith hs
    ts := ts.push t
    let e0 := claim + q1
    let c1 := e0 + q1 + qinf
    claim := qinf * t * t + c1 * t + e0
  let comb ← match c.folded alpha beta (skipWeights zc.z) rhoIn ts with
    | .ok v => pure v
    | .error err => throw err
  liftTM (check (comb.size == 64) "lincheck: the circuit fold has the wrong width")
  liftTM (absorbSlice p.zPartial)
  liftTM (check (dot comb p.zPartial == claim) "lincheck: ConsistencyFailed (sumcheck-final)")
  let rPrime ← sampleWith hs
  let w := dot (skipWeights rPrime) p.zPartial
  return { w, rPrime, rIn := ts.reverse }

/-- `ringSwitch` with the oracle's coins. -/
def ringSwitchWith (hs : Hashes M) (_m : Nat) (claims : Array Claim) (rs : Array (Array F128)) :
    TMW M (F128 × Array Switched) := do
  let n := claims.size
  liftTM (label "flock-pcs-open-batch-v0")
  let mut betas : Array F128 := #[]
  let mut perClaim : Array (Array F128 × Array F128) := #[]
  for k in [0:n] do
    let c := claims[k]!
    let s := rs[k]!
    liftTM (label "flock-ring-switch-v0")
    liftTM (absorbSlice s)
    let x0 := c.x[0]!
    let w := (Array.range 128).map fun v => c.skip[v % 64]! * (if v / 64 == 1 then x0 else 1 + x0)
    liftTM (check (dot w s == c.value) s!"opening: ring-switch claim {k} mismatch")
    let rpp ← sampleVecWith hs 7
    -- t[u]: bit v of t[u] is bit u of s[v]
    let t := (Array.range 128).map fun u =>
      (Array.range 128).foldl (fun acc v => if (s[v]!).bit u then acc + F128.monomial v else acc) (0 : F128)
    betas := betas.push (dot (eqTable rpp) t)
    perClaim := perClaim.push (rpp, c.x.extract 1 c.x.size)
  let gamma ← sampleVecWith hs n
  let T := dot gamma betas
  let sws := (Array.range n).map fun k => { gamma := gamma[k]!, rpp := (perClaim[k]!).1, z := (perClaim[k]!).2 : Switched }
  return (T, sws)

/-- The Ligerito verifier's state over `M`. -/
abbrev LMW (M : Type → Type) := StateT LState (TMW M)

def liftW {α : Type} (x : TMW M α) : LMW M α := StateT.lift x

/-- A Ligerito action that hashes nothing, over `M`. -/
def liftLM {α : Type} (x : LM α) : LMW M α := fun s t => monadLift (x s t)

/-- `challenge` with the oracle's coins. -/
def challengeWith (hs : Hashes M) (p : LigeritoProof) : LMW M F256 := do
  let r ← liftW (sampleKWith hs)
  modify fun st => { st with s := st.q.eval r }
  let m ← liftLM (nextMsg p)
  modify fun st => { st with q := Poly.ofMsg m st.s }
  return r

/-- `oodIntro` with the oracle's coins. -/
def oodIntroWith (hs : Hashes M) (p : LigeritoProof) (dim : Nat) (split : Nat) : LMW M Unit := do
  let z ← liftW (sampleVecWith hs dim)
  let y ← liftLM (nextOod p)
  liftLM (lift (absorbF y))
  let m ← liftLM (nextMsg p)
  let intro := Poly.ofMsg m (F256.ofF y)
  liftLM (claimPow p)
  let b ← liftW (sampleWith hs)
  modify fun st => { st with q := st.q.addScaled intro b, s := st.s + (F256.ofF y).smul b, oods := st.oods.push ⟨z, b, some split⟩ }

/-- `queryLevel` with the oracle's coins and Merkle hashes. -/
def queryLevelWith (hs : Hashes M) (ms : MerkleScheme) (p : LigeritoProof) (lv : Level) (cap : Array ByteArray)
    (rows : Array (Array F128)) (sibs : Array ByteArray) (salts : Array ByteArray) (fold : Array F256) :
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
  (merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d pos rows sibs salts (2 ^ lv.lanes) : M Unit)
  let ea := eqTable alpha
  let ef := eqTableK fold
  let mut e : F256 := 0
  for j in [0:lv.q] do
    let row := rows[j]!
    let mut acc : F256 := 0
    for l in [0:row.size] do acc := acc + (ef[l]!).smul (row[l]!)
    e := e + acc.smul (ea[j]!)
  return (pos, alpha, e)

/-- `ligerito` with the oracle's coins and Merkle hashes. -/
def ligeritoWith (hs : Hashes M) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → Nat → Array F256) : TMW M Unit := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  liftTM (check (p.initialCap == cap) "ligerito: initial_cap differs from the commitment's cap")
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (cap.foldl (· ++ ·) ByteArray.empty))
  let prog : LMW M Unit := do
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
    for _ in [0:sch.k0] do
      initCh := initCh.push (← challengeWith hs p)
    let mut splitDim := n - sch.k0 + 1
    let cap1 := p.recursiveCaps[0]!
    liftLM (lift (absorbBytes (cap1.foldl (· ++ ·) ByteArray.empty)))
    oodIntroWith hs p splitDim 0
    oodIntroWith hs p splitDim 0
    let (pos0, al0, e0) ← queryLevelWith hs ms p (lv 0) cap p.initialProof.rows p.initialProof.siblings p.initialProof.salts initCh
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
  let init : LState := ⟨F256.ofF T, ⟨0, 0, 0⟩, 0, 0, 0, 0, 0, #[], #[]⟩
  let _ ← prog.run init
  return ()

/-- `verifyRep` with the oracle's hashes. -/
def verifyRepWith (hs : Hashes M) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : RunRecord) (i r : Nat)
    (bytes : ByteArray) : M ByteArray := do
  let t := spec.tables[i]!
  let name := s!"{t}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes
  let shape : Shape := { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let p ← decodeProof shape bytes
  if p.params != pcsParams st.merkle sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
  let prog : TMW M Unit := do
    let zc ← bindAndZerocheckWith hs st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let lc ← lincheckWith hs st.fold zc p.lincheck
    let (T, sws) ← ringSwitchWith hs st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    ligeritoWith hs st.merkle sch T p.cap p.ligerito (basis sws)
  let ((), tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- The shape `verifyRepWith` decodes a rep's proof file at, with `n` extra claims. -/
def repFileShape (st : Setup) (sch : Schedule) (n : Nat) : Shape :=
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes
  { m := st.m, kLog := st.kLog, claims := 2 + n, sched := sch, hashLen := st.merkle.digestLen, saltLen }

/-- **`verifyRepWith` at the decoded proof `p`**: everything after the decode, on the rep's stream with the statement's
extra claims: the pinned PCS parameters (R7/F2), §9–§14, every recorded round consumed (S15/R2) and the cap the committed
`root_B` (S16/R1). -/
def verifyRepPWith (hs : Hashes M) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : RunRecord) (i r : Nat)
    (stream : StreamRec) (extras : Array Claim) (p : Proof) : M ByteArray := do
  let t := spec.tables[i]!
  let name := s!"{t}/rep{r}"
  if p.params != pcsParams st.merkle sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
  let prog : TMW M Unit := do
    let zc ← bindAndZerocheckWith hs st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let lc ← lincheckWith hs st.fold zc p.lincheck
    let (T, sws) ← ringSwitchWith hs st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    ligeritoWith hs st.merkle sch T p.cap p.ligerito (basis sws)
  let ((), tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- **`verifyRepWith` splits at the decode**: the rep's stream (S11) and the extra claims, the proof file decoded at
`repFileShape`, then `verifyRepPWith` at the decoded proof. -/
theorem verifyRepWith_split (hs : Hashes M) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : RunRecord)
    (i r : Nat) (bytes : ByteArray) :
    verifyRepWith hs spec st sch sess i r bytes = (do
      let name := s!"{spec.tables[i]!}/rep{r}"
      let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
      let extras ← st.extra sess.points
      let p ← decodeProof (repFileShape st sch extras.size) bytes
      verifyRepPWith hs spec st sch sess i r stream extras p) := rfl

/-- `verify` with every hash call the oracle's: `.ok ()` is Accept. -/
def verifyWithM (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decodeWith hs spec record
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (← spec.dgWith hs b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

/-- `verifyWithM` at the record's parsed value `j` (`Record.decodeJWith`); at a record that parses to `j` it is
`verifyWithM` (`verifyWithM_of_parse`). -/
def verifyJWithM (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (j : Lean.Json) (proofs : Array ByteArray) :
    M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decodeJWith hs spec j
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (← spec.dgWith hs b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

theorem verifyWithM_of_parse [LawfulMonad M] (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup)
    {record : String} {j : Lean.Json} (proofs : Array ByteArray) (h : JsonParse.parse record = .ok j) :
    verifyWithM hs spec tabs record proofs = verifyJWithM hs spec tabs j proofs := by
  simp only [verifyWithM, verifyJWithM, Record.decodeWith_of_parse hs spec h]

/-- The verifier with SHA-512 replaced by `h`, every other hash the deployed one; `verifyWith Sha512.hash` is `verify`
(`Flock.verify_eq_verifyWith`). -/
def verifyWith (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : V Unit :=
  verifyWithM (Hashes.ofSha512 h) spec tabs record proofs

end Flock
