import Proofs.Flock.Claims.Complete.Prover.Piop

/-!
# The honest prover's Ligerito opening

PROTOCOL.md §13–§14 from the prover's side. Ligerito proves `Σ_y t(y)·W(y) = s` for the committed table `t` and a weight
`W` that starts as the batched basis `b` (`basisVals`) with `s = T`; every claim the verifier adds (an out-of-domain
value, a consistency sum) adds its weight to `W`, scaled by its coin.

* **Commitments** (`commitP`, §13.1, §14). A level's F-table `g` of `2^(lanes + cols)` entries has lanes
  `M[ℓ][c] = g[c·2^lanes + ℓ]`, codeword `C[q][ℓ] = Σ_c M[ℓ][c]·X̂_c(ω_q)` for `q < 2^d` (the verifier's `whats`), leaf `q`
  the scheme's leaf of row `C[q]` under its salt, and the Merkle cap the `2^cap` nodes `d − cap` levels up. An opening
  (`openP`) is the rows at the query positions, their siblings leaf side first, and their salts.
* **Sumcheck** (§13.2). The table and the weight are K-tables, folded at each challenge on the low variable
  (`foldK`); a message is `(q(0), q`'s leading coefficient`)` for `q(X) = Σ_{x'} t(X, x')·W(X, x')` (`roundPoly`).
* **Splits.** After a level's last challenge the folded K-table `f = f₀ + u·f₁` is split into the next level's F-table
  `g(b, x) = f_b(x)` (`splitK`, `b` the new low variable), and the weight into `W'(b, x) = u^b·W(x)` (`liftU`), so
  `Σ g·W' = Σ f·W`; the next message is `g`'s round polynomial in `b`.
* **Claims.** An out-of-domain sample `z` at a level adds `β·eq(z, ·)` with value `g̃(z)` (`mleF`); level `i`'s queries
  add `β·u^b·V` on the next level's table, `V(c) = Σ_j eq(α)[j]·X̂_c(ω_{pos_j})` (`consW`).
* **The last level** sends `yr`, the folded K-table as `F` pairs. The message read after its last challenge enters no
  equation (§13.3); the prover sends upstream's plain fold message there, the round polynomial of the folded, unsplit
  K-table and weight in their next variable (`recursive_prover_with_basis_impl`, whose last round at the last level is
  `sumcheck.fold`), not the split coordinate's.
-/

namespace FlockComplete

open Flock Transcript

/-- **A level's commitment**: its codeword rows, the leaves' salts, the Merkle levels from the leaves to the cap, and the
cap. -/
structure Commit where
  rows : Array (Array F128)
  salts : Array ByteArray
  layers : Array (Array ByteArray)
  cap : Array ByteArray
  deriving Inhabited

/-- `X̂_c(ω_q)` for `c < 2^cols`: the products of `Ŵ_i(ω_q)` over the bits of `c`. -/
def xhats (sv : Array F128) (q cols : Nat) : Array F128 :=
  let wh := whats sv q cols
  (List.range cols).foldl (fun t i => t ++ t.map (· * wh[i]!)) #[1]

/-- **The codeword** of an F-table at a level: `C[q][ℓ] = Σ_c g[c·2^lanes + ℓ]·X̂_c(ω_q)`. -/
def encodeRows (lv : Level) (g : Array F128) : Array (Array F128) :=
  let lanes := 2 ^ lv.lanes
  let sv := svTable (lv.cols + 1)
  (Array.range (2 ^ lv.d)).map fun q =>
    let xh := xhats sv q lv.cols
    (Array.range lanes).map fun l => sumF (2 ^ lv.cols) fun c => g[c * lanes + l]! * xh[c]!

/-- The cap of a level's flattened entries. -/
def flat (cap : Array ByteArray) : ByteArray := cap.foldl (· ++ ·) ByteArray.empty

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **A level's commitment** of the F-table `g` with the leaves' salts `salt q`, each fitted to the scheme's salt
size. -/
def commitP (hs : Hashes M) (ms : MerkleScheme) (lv : Level) (g : Array F128) (salt : Nat → ByteArray) : M Commit := do
  let ops := ms.opsWith hs
  let rows := encodeRows lv g
  let salts := (Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q)
  let leaves ← (Array.range (2 ^ lv.d)).mapM fun q => ops.leaf rows[q]! salts[q]!
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:lv.d - lv.cap] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => ops.node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨rows, salts, layers, cur⟩

/-- **A level's opening** at `pos`: the rows, each query's `d − cap` siblings leaf side first, and the salts. -/
def openP (ms : MerkleScheme) (cm : Commit) (lv : Level) (pos : Array Nat) : LevelProof :=
  { rows := pos.map fun p => cm.rows[p]!
    siblings := pos.foldl (fun acc p => acc ++ (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!) #[]
    salts := if ms.saltBytes = 0 then #[] else pos.map fun p => cm.salts[p]! }

/-- A K-table's fold at `r` on its low variable. -/
def foldK (r : F256) (a : Array F256) : Array F256 :=
  (Array.range (a.size / 2)).map fun k => a[2 * k]! + r * (a[2 * k + 1]! + a[2 * k]!)

/-- **A sumcheck message**: `(q(0), q`'s leading coefficient`)` for `q(X) = Σ_{x'} t(X, x')·W(X, x')`. -/
def roundPoly (t w : Array F256) : F256 × F256 :=
  let h := t.size / 2
  ((List.range h).foldl (fun acc k => acc + t[2 * k]! * w[2 * k]!) 0,
   (List.range h).foldl (fun acc k => acc + (t[2 * k]! + t[2 * k + 1]!) * (w[2 * k]! + w[2 * k + 1]!)) 0)

/-- **The split** of a K-table: `g[2x + b] = f[x].c_b`. -/
def splitK (f : Array F256) : Array F128 := f.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]

/-- The weight on a split table: `W'[2x + b] = u^b·W[x]`. -/
def liftU (w : Array F256) : Array F256 := w.foldl (fun acc v => (acc.push v).push (F256.u * v)) #[]

/-- An F-table as a K-table. -/
def embedF (g : Array F128) : Array F256 := g.map F256.ofF

/-- `W + β·V` entrywise. -/
def addScaled (w v : Array F256) (b : F128) : Array F256 :=
  (Array.range w.size).map fun i => w[i]! + (v[i]!).smul b

/-- **The multilinear extension** of an F-table at an F point: `Σ_x g[x]·eq(z, x)`. -/
def mleF (g : Array F128) (z : Array F128) : F128 :=
  let e := eqTable z
  sumF g.size fun x => g[x]! * e[x]!

/-- **The consistency weight** of a level's queries: `V(c) = Σ_j eq(α)[j]·X̂_c(ω_{pos_j})` for `c < 2^cols`. -/
def consW (cols : Nat) (pos : Array Nat) (alpha : Array F128) : Array F128 :=
  let ea := eqTable alpha
  let sv := svTable (cols + 1)
  let xs := pos.map fun p => xhats sv p cols
  (Array.range (2 ^ cols)).map fun c => sumF pos.size fun j => ea[j]! * (xs[j]!)[c]!

/-- What the prover keeps through the opening: the K-table being folded, the weight, and the proof's lists so far. -/
structure LigState where
  tab : Array F256
  w : Array F256
  msgs : Array (F256 × F256) := #[]
  oods : Array F128 := #[]
  qn : Array UInt64 := #[]
  cn : Array UInt64 := #[]
  cbn : Array UInt64 := #[]
  deriving Inhabited

/-- The prover's Ligerito actions: its transcript and its state. -/
abbrev LP (M : Type → Type) := StateT LigState (TMW M)

def liftT {α : Type} (x : TMW M α) : LP M α := StateT.lift x

/-- Send the next sumcheck message. -/
def sendMsg (m : F256 × F256) : LP M Unit := do
  modify fun st => { st with msgs := st.msgs.push m }
  liftT (liftTM (absorbK m.1))
  liftT (liftTM (absorbK m.2))

/-- A grinding nonce onto list `kind` (`0` query, `1` consistency, `2` claim), absorbed as a message: `0`, the only one the
verifier accepts (R4), so the tape's nonces are unread. -/
def powP (_nonce : Nat → Nat → UInt64) (kind : Nat) : LP M Unit := do
  let st ← get
  set (match kind with
    | 0 => { st with qn := st.qn.push 0 }
    | 1 => { st with cn := st.cn.push 0 }
    | _ => { st with cbn := st.cbn.push 0 })
  liftT (liftTM (pow 0))

/-- A challenge: fold the table and the weight at it; then the next message, after splitting when `split`. -/
def challengeP (io : Asks M) (split : Bool) : LP M F256 := do
  let r ← liftT (sampleKP io)
  modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
  if split then modify fun st => { st with tab := embedF (splitK st.tab), w := liftU st.w }
  let st ← get
  sendMsg (roundPoly st.tab st.w)
  return r

/-- An out-of-domain sample on the current F-table `g` with an intro message (levels ≥ 1). -/
def oodIntroP (io : Asks M) (nonce : Nat → Nat → UInt64) (g : Array F128) (dim : Nat) : LP M Unit := do
  let z ← liftT (sampleVecP io dim)
  let y := mleF g z
  modify fun st => { st with oods := st.oods.push y }
  liftT (liftTM (absorbF y))
  let e := embedF (eqTable z)
  sendMsg (roundPoly (embedF g) e)
  powP nonce 2
  let b ← liftT (sampleP io)
  modify fun st => { st with w := addScaled st.w e b }

/-- A level's queries: the query and consistency coins, the opening, and the consistency weight `V`. -/
def queryP (io : Asks M) (nonce : Nat → Nat → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    LP M (LevelProof × Array F128) := do
  powP nonce 0
  let coins ← liftT (sampleVecP io lv.q)
  let pos := positions lv coins
  powP nonce 1
  let alpha ← liftT (sampleVecP io (Nat.log2 (lv.q - 1) + 1))
  return (openP ms cm lv pos, consW lv.cols pos alpha)

/-- **The Ligerito prover** (§13.3) on the table `t` committed at level 0 as `cm0`, for the claim `Σ t·b = T`: the
recursive levels' commitments are salted from `salt l` (level `l ≥ 1`), the nonces read from `nonce`. -/
def ligeritoP (io : Asks M) (hs : Hashes M) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64) : TMW M LigeritoProof := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (flat cm0.cap))
  let prog : LP M (Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof) := do
    -- level 0's out-of-domain sample (no message)
    let z ← liftT (sampleVecP io n)
    let y := mleF t z
    modify fun st => { st with oods := st.oods.push y }
    liftT (liftTM (absorbF y))
    powP nonce 2
    let b0 ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (embedF (eqTable z)) b0 }
    let st ← get
    sendMsg (roundPoly st.tab st.w)
    for j in [0:sch.k0] do
      let _ ← challengeP io (j + 1 == sch.k0)
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
          let d ← get
          f := d.tab
          modify fun st => { st with tab := embedF (splitK st.tab), w := liftU st.w }
          let st ← get
          -- nothing reads the last level's split: it sends upstream's plain fold message, of the unsplit table
          sendMsg (if last then roundPoly d.tab d.w else roundPoly st.tab st.w)
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
    return (caps, rproofs, fin, op0)
  let ((caps, rproofs, fin, op0), st) ← prog.run { tab := embedF t, w := embedF b }
  return { initialCap := cm0.cap, initialProof := op0, recursiveCaps := caps, recursiveProofs := rproofs,
           finalProof := fin, messages := st.msgs, queryNonces := st.qn, oodValues := st.oods, claimNonces := st.cbn,
           consistencyNonces := st.cn }

end FlockComplete
