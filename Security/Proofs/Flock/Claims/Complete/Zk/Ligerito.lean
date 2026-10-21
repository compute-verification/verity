import Proofs.Flock.Claims.Complete.Prover.Ligerito
import Proofs.Flock.Claims.Complete.Zk.Tape
import Flock.With.Zk

/-!
# The masked prover's Ligerito opening

Live PROTOCOL.md §7 (VEIL Figs. 8–9) from the prover's side: `ligeritoP` with level 0's hiding, as `Zk.ligeritoWith`
reads it.

* **Level 0's commitment** (`encodeRowsZk`, `commitPZk`): each row carries the table's `2^lanes` witness lanes and four
  uniform extra lanes `R_e` (the tape's), and every lane `l` is padded by `(c + Ŵ_cols)·Σ_{i < t_pad} μ_l[i]·X_i`
  (`padF`, `Zk.padAt` on F, `c = PAD_C`): lane `l`'s entry at position `q` is `Enc(lane)(ω_q) + padF(μ_l)(ω_q)`. Both
  reps share it.
* **The lane phase** (`ligeritoPZk`): after `LABEL_OPEN`, the level-0 out-of-domain value and the first `k_0` sumcheck
  messages are sent masked (`lgOod`, `lgMsg`; `maskMsg`). After the last lane challenge, rep `r` sends its pair's sums
  `e_i = ⟨W, R_{2r+i}⟩` masked (`lgE`), draws `ρ_0, ρ_1`, refuses dependent ones (as upstream does), and after
  `LABEL_CLOSE` sends the blinded claim `T' = ⟨W, y₁'⟩` for `y₁' = f + ρ_0 R_{2r} + ρ_1 R_{2r+1}` and the padding's
  combination `ybar = Σ_l eq(r̄)_l μ_l + ρ_0 μ_{L+2r} + ρ_1 μ_{L+2r+1}`; the opening goes on from `y₁'` as
  `ligeritoP` from the split table, the last level's last message upstream's plain fold message (the folded, unsplit
  table's round polynomial). The row check then cancels the padding: the verifier adds `padAt(ybar)`.
* It returns the proof, level 0's hiding messages, and the lane phase's record (`L0Rec`, the verifier's), from which
  the prover builds the lane phase's constraints (`Zk.ligeritoConstraints`).
-/

namespace FlockComplete

open Flock Transcript

/-- Lane `l`'s padding at a position: `(c + Ŵ_logMsg)·Σ_i μ[i]·X_i` at the position's `Ŵ`s `ws` (`Zk.padAt` on F). -/
def padF (ws : Array F128) (logMsg : Nat) (mu : Array F128) : F128 := (Zk.padAt ws logMsg (mu.map F256.ofF)).c0

/-- **Level 0's padded codeword**: the witness lanes of the F-table `g`, then the extra lanes `extra e`, each entry the
lane's encoding plus its padding (lane `l`'s coefficients `lanePad l`, `t_pad` of them). -/
def encodeRowsZk (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) : Array (Array F128) :=
  let lanes := 2 ^ lv.lanes
  let sv := svTable (lv.cols + 1)
  let mus := (Array.range (lanes + Zk.EXTRA_LANES)).map fun l => (Array.range tPad).map (lanePad l)
  (Array.range (2 ^ lv.d)).map fun q =>
    let xh := xhats sv q lv.cols
    let ws := whats sv q (lv.cols + 1)
    (Array.range (lanes + Zk.EXTRA_LANES)).map fun l =>
      let v : Nat → F128 := if l < lanes then fun c => g[c * lanes + l]! else extra (l - lanes)
      sumF (2 ^ lv.cols) (fun c => v c * xh[c]!) + padF ws lv.cols mus[l]!

/-- The sum `Σ_c a[c]·b[c]` of two K-tables. -/
def dotK (a b : Array F256) : F256 := (List.range a.size).foldl (fun acc c => acc + a[c]! * b[c]!) 0

/-- A sumcheck message masked by the lane round `j`'s pads (`lgMsg j 0` to `lgMsg j 3`, coordinate by coordinate). -/
def maskMsg (lay : Zk.PadLayout) (h : Array F128) (j : Nat) (m : F256 × F256) : F256 × F256 :=
  (m.1 + ⟨h[lay.lgMsg j 0]!, h[lay.lgMsg j 1]!⟩, m.2 + ⟨h[lay.lgMsg j 2]!, h[lay.lgMsg j 3]!⟩)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **A level's commitment of given rows**: `commitP` once the codeword is computed. -/
def commitRowsP (hs : Hashes M) (ms : MerkleScheme) (lv : Level) (rows : Array (Array F128)) (salt : Nat → ByteArray) :
    M Commit := do
  let ops := ms.opsWith hs
  let salts := (Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q)
  let leaves ← (Array.range (2 ^ lv.d)).mapM fun q => ops.leaf rows[q]! salts[q]!
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:lv.d - lv.cap] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => ops.node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨rows, salts, layers, cur⟩

/-- **Level 0's masked commitment** of the packed table `g`, with the table's extra lanes and paddings. -/
def commitPZk (hs : Hashes M) (ms : MerkleScheme) (lv : Level) (tPad : Nat) (g : Array F128)
    (extra lanePad : Nat → Nat → F128) (salt : Nat → ByteArray) : M Commit :=
  commitRowsP hs ms lv (encodeRowsZk lv tPad g extra lanePad) salt

/-- **The masked Ligerito prover** on the table `t` committed at level 0 as `cm0` (`commitPZk`), for the claim
`Σ t·b = T`, as rep `rep` with the pads `h`; the rest as `ligeritoP`. -/
def ligeritoPZk (io : Asks M) (hs : Hashes M) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : Nat) (extra lanePad : Nat → Nat → F128)
    (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64) :
    TMW M (LigeritoProof × Zk.Level0Msgs × Zk.L0Rec) := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  let lanes := 2 ^ (lv 0).lanes
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (flat cm0.cap))
  liftTM (label Zk.LABEL_OPEN)
  let prog : LP M (Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof × Zk.Level0Msgs ×
      Array F256 × (F256 × F256) × F128 × F128) := do
    -- level 0's out-of-domain sample (no message), its value masked
    let z ← liftT (sampleVecP io n)
    let y := mleF t z + h[lay.lgOod 0]!
    modify fun st => { st with oods := st.oods.push y }
    liftT (liftTM (absorbF y))
    powP nonce 2
    let b0 ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (embedF (eqTable z)) b0 }
    let st ← get
    sendMsg (maskMsg lay h 0 (roundPoly st.tab st.w))
    let mut initCh : Array F256 := #[]
    let mut rho : F256 × F256 := (0, 0)
    let mut zl : Zk.Level0Msgs := default
    for j in [0:sch.k0] do
      let r ← liftT (sampleKP io)
      initCh := initCh.push r
      modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
      if j + 1 == sch.k0 then
        -- the pair's sums, its coins, the blinded claim and the padding's combination
        let st ← get
        let r0 := (Array.range st.tab.size).map (extra (2 * rep))
        let r1 := (Array.range st.tab.size).map (extra (2 * rep + 1))
        let e0 := dotK st.w (embedF r0) + ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩
        let e1 := dotK st.w (embedF r1) + ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩
        liftT (liftTM (absorbK e0))
        liftT (liftTM (absorbK e1))
        let rh0 ← liftT (sampleKP io)
        let rh1 ← liftT (sampleKP io)
        rho := (rh0, rh1)
        if !Zk.rhoIndependent rho then throw "PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128"
        liftT (liftTM (label Zk.LABEL_CLOSE))
        let y1 := (Array.range st.tab.size).map fun c => st.tab[c]! + rh0.smul r0[c]! + rh1.smul r1[c]!
        let tP := dotK st.w y1
        liftT (liftTM (absorbK tP))
        let ef := eqTableK initCh
        let ybar := (Array.range tPad).map fun i =>
          (List.range lanes).foldl (fun acc l => acc + (ef[l]!).smul (lanePad l i))
            (rh0.smul (lanePad (lanes + 2 * rep) i) + rh1.smul (lanePad (lanes + 2 * rep + 1) i))
        liftT (liftTM (absorbSlice (ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[])))
        zl := ⟨e0, e1, tP, ybar⟩
        modify fun st => { st with tab := embedF (splitK y1), w := liftU st.w }
        let st ← get
        sendMsg (roundPoly st.tab st.w)
      else
        let st ← get
        sendMsg (maskMsg lay h (j + 1) (roundPoly st.tab st.w))
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
    return (caps, rproofs, fin, op0, zl, initCh, rho, b0, y)
  let ((caps, rproofs, fin, op0, zl, initCh, rho, b0, y), st) ← prog.run { tab := embedF t, w := embedF b }
  return ({ initialCap := cm0.cap, initialProof := op0, recursiveCaps := caps, recursiveProofs := rproofs,
            finalProof := fin, messages := st.msgs, queryNonces := st.qn, oodValues := st.oods, claimNonces := st.cbn,
            consistencyNonces := st.cn }, zl, { target := T, ood := #[(b0, y)], challenges := initCh, rho })

end FlockComplete
