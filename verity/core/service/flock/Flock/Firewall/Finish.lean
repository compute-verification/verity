/-
# The firewall's final round (live PROTOCOL.md §10.1): a firewalled rep's proof file, computed by the firewall itself

At a firewalled rep's end the proof file that leaves is the firewall's own, this code's (`flock-circuit` sends it, in
`zk_finish_fw`, after its end check `Firewall::finish`): the zerocheck and lincheck sent as the values that passed the firewall, masked by the rep's pads, `final_c`
the one those values fix; the ring switch, the masked Ligerito opening and the levels above it computed from the host's
witness, the table's level-0 hiding and the rep's salts; and the inner proof on the rep's pads. Here that file is code the
firewall runs, the analysis's masked prover (`Flock.Firewall.Finish`'s copies, `FinishProver`) on what the firewall holds:

* `rowsFft`, `commitFw`: a level ≥ 1's commitment of an F-table, its codeword by the additive FFT (`Link.fft`) on each lane
  (`FlockComplete.commitP`'s, which evaluates each entry on its own), its leaves `hm96-sha512/v1`'s in tasks.
* `commit0`: level 0's commitment of the table the transport hands it (`Link.In`, the table `Link.parse` reads;
  `FlockComplete.commitPZk`'s). `commitOf`: the same commitment read off the table's `Level0`, which the session
  loop drew and committed (`Level0.ofDraws_ok`: its rows, layers and root are `Link`'s of its own inputs).
* `zerocheckFw`, `lincheckFw`: the masked zerocheck and lincheck sending the passed values `v_i` as `v_i + h_i`
  (`FlockComplete.zerocheckCut`, `lincheckCut` at `masked i = v_i + h_i`), `final_c`'s too: the inner proof holds the
  passed `final_c` to `P_c(ζ)` of the passed `round1_c` (`Zk.replay`'s constraint).
* `switchVec`: claim `k`'s switched vector from the packed table's bits (`FlockComplete.switchVec` on the trace).
* `repFw`: the rep (`FlockComplete.repPZk`'s order: the PIOP, `repPiop`; the ring switch on `abcClaims ++ extras` and the
  masked Ligerito, `repOpening`; the verifier's own replay and the ring-switch and lane-phase constraints, `repCons`; the
  inner proof), its proof file
  `encZk` at `shapeZk`; a constraint that fails on the rep's pads (`failingOn`) stops it, and no file leaves. The rank check is the Lean frame's (`Frame.maskRank`), which stops the rep before this runs.
* `asksOf`: a rep's rounds answered by the coins the stream answered them, round by round.
* `liveAsks`, `openLive`, `innerLive`: the rep's rounds past its recorded coins sent to the firewall as they come, each
  waiting for the coins the stream answered it: the opening's (`repPiop` and `repOpening` at zero values and pads), the
  rounds the Lean frame holds the device's opening rounds to, and the inner proof's (`repCons` and `innerP`), the
  firewall's own (live PROTOCOL.md §10.1). Only the `Asks` is live: the code that computes each round is `repFw`'s.

* `serve`: the requests, on the statement of `flock-verify verify --zk`: each table's level 0, each rep's salts for levels
  ≥ 1, which this code draws itself from the operating system at the rep's start and hands the prover (`salts`), and
  each rep's proof file on them. `flock-firewall finish` serves them on its standard input (`cmd`, `Witness.setupOf`);
  the session loop on its frames port (`serveLoop`, `Hello.Setup`), reading each table's level 0 from its own `Level0`s.
-/
import Flock.Firewall.FinishProver
import Flock.Firewall.Frame
import Flock.Firewall.Hello
import Flock.Firewall.Level0
import Flock.Firewall.Link
import Flock.Witness

namespace Flock.Firewall.Finish

open Flock Transcript

/-! ## The commitments -/

/-- **A level's codeword** by the FFT, one task to a lane: lane `l`'s words `g[c·2^lanes + l]` as one polynomial in the
novel basis over `cols + 1` bits, its upper half zero (`Link.coeffs` with no padding), evaluated on all `2^d` points. -/
def rowsFft (lv : Level) (g : Array F128) : Array (Array F128) :=
  let lanes := 2 ^ lv.lanes
  let tw := Link.twiddles (svTable (lv.cols + 1)) lv.d (lv.cols + 1)
  let es := ((Array.range lanes).map fun l =>
    Task.spawn fun _ => Link.fft tw lv.d 0 (lv.cols + 1) (Link.coeffs lv.cols (fun c => g[c * lanes + l]!) #[])).map
    Task.get
  (Array.range (2 ^ lv.d)).map fun q => (Array.range lanes).map fun l => es[l]![q]!

/-- **A level's codeword**: by the FFT where the domain, wider than the columns, fits in a word; else `encodeRows`. -/
def rowsOf (lv : Level) (g : Array F128) : Array (Array F128) :=
  if lv.cols < lv.d ∧ lv.d ≤ 64 then rowsFft lv g else encodeRows lv g

/-- The Merkle levels over `leaves`, `k` of them above the leaves: each node `SHA-512(l ‖ r)`; and the top level. -/
def treeOf (leaves : Array ByteArray) (k : Nat) : Array (Array ByteArray) × Array ByteArray := Id.run do
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:k] do
    cur := (Array.range (cur.size / 2)).map fun i => Sha512.hash (cur[2 * i]! ++ cur[2 * i + 1]!)
    layers := layers.push cur
  return (layers, cur)

/-- **A commitment of given rows** under the `hm96-sha512/v1` leaves, salt `q` fitted to 192 bytes, `2^CHUNK` rows to a
task. -/
def commitRows (lv : Level) (rows : Array (Array F128)) (salt : Nat → ByteArray) : Commit :=
  let salts := (Array.range (2 ^ lv.d)).map fun q => fitBytes 192 (salt q)
  let k := min lv.d Link.CHUNK
  let leaves := Link.parRange (fun q => MerkleScheme.hm96Sha512.leaf rows[q]! salts[q]!) (2 ^ (lv.d - k)) (2 ^ k)
  let (layers, cap) := treeOf leaves (lv.d - lv.cap)
  ⟨rows, salts, layers, cap⟩

/-- **A level ≥ 1's commitment** of the F-table `g` with the leaves' salts `salt q` (`FlockComplete.commitP`'s). -/
def commitFw (lv : Level) (g : Array F128) (salt : Nat → ByteArray) : Commit := commitRows lv (rowsOf lv g) salt

/-- **Level 0's commitment** of the table `i` at level `lv` with `t_pad` paddings (`FlockComplete.commitPZk`'s): the rows
`Link.rows`, under the table's level-0 salts. -/
def commit0 (lv : Level) (tPad : Nat) (i : Link.In) : Commit :=
  commitRows lv (Link.rows lv tPad i.g i.extra i.lanePad) i.salt

/-- **Level 0's commitment as the session loop keeps it** (`Level0`): its rows, its salts fitted to 192 bytes, its layers
from the leaves to the cap, and the cap, the top layer; `commit0`'s of the table `l.toIn` where `Level0.ofDraws_ok` holds,
read off rather than committed again. -/
def commitOf (l : Level0.Level0) : Commit :=
  ⟨l.rows, (Array.range (2 ^ l.lv.d)).map (fun q => fitBytes 192 (l.salt q)), l.layers, l.layers.back?.getD #[]⟩

/-! ## The rep -/

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The masked zerocheck** sending the masked values `mk` at their pad indices (`FlockComplete.zerocheckCut`). -/
def zerocheckFw (io : Asks M) (m : Nat) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : Nat → F128) :
    TMW M (ZerocheckProof × ZcOut) := do
  liftTM (label "flock-r1cs-v0")
  liftTM (absorbBytes digest)
  liftTM (absorbBytes cap)
  liftTM (label "flock-zerocheck-v0")
  liftTM (label Zk.LABEL_PADS)
  liftTM (absorbBytes root)
  let _rSkip ← sampleVecP io 6
  let rOuter ← sampleVecP io (m - 13)
  let r := Flock.pinned ++ rOuter
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

/-- **The masked lincheck** sending the masked values `mk` at their pad indices; it stops where the circuit fold fails,
as the session does. -/
def lincheckFw (io : Asks M) (c : CircuitFold) (lay : Zk.PadLayout) (mk : Nat → F128) (zc : ZcOut) :
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

/-- Claim `c`'s ring-switch vector on the packed table `g`: `s[v] = Σ_y eq(x[1..], y)·bit_v(g[y])`. -/
def switchVec (g : Array F128) (c : Claim) : Array F128 :=
  let eY := eqTable (c.x.extract 1 c.x.size)
  (Array.range 128).map fun v => sumF eY.size fun y => if (g[y]!).bit v then eY[y]! else 0

namespace Fast

/-- `Link.parRange` specialized at each call (`parRangeS_eq`): each task's loop runs `f` inline, so no entry calls a
closure or touches the reference count of anything the tasks share. -/
@[specialize] def parRangeS {α : Type} (f : Nat → α) (a b : Nat) : Array α :=
  ((Array.range a).map fun i => Task.spawn fun _ => (Array.range b).map fun j => f (i * b + j)).foldl
    (fun acc t => acc ++ t.get) #[]

theorem parRangeS_eq {α : Type} (f : Nat → α) (a b : Nat) : parRangeS f a b = (Array.range (a * b)).map f :=
  Link.parRange_eq f a b

/-- `(Array.range n).map f` in tasks of `2^12` entries. -/
@[specialize] def parGen (f : Nat → F128) (n : Nat) : Array F128 :=
  let a := n / 4096
  parRangeS f a 4096 ++ (Array.range (n - a * 4096)).map fun j => f (a * 4096 + j)

theorem parGen_eq (f : Nat → F128) (n : Nat) : parGen f n = (Array.range n).map f := by
  conv => rhs; rw [← Nat.add_sub_cancel' (Nat.div_mul_le_self n 4096)]
  rw [Array.range_add, Array.map_append, Array.map_map, parGen, parRangeS_eq]
  rfl

theorem range_map_getElem! (f : F128 → F128) (t : Array F128) :
    (Array.range t.size).map (fun i => f t[i]!) = t.map f := by
  apply Array.ext
  · simp
  · intro i h1 _
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_eq t (by simpa using h1)]

/-- `t.map f` in tasks of `2^12` entries. -/
@[specialize] def parMap (f : F128 → F128) (t : Array F128) : Array F128 := parGen (fun i => f t[i]!) t.size

theorem parMap_eq (f : F128 → F128) (t : Array F128) : parMap f t = t.map f := by
  rw [parMap, parGen_eq, range_map_getElem!]

/-- `parGen f` with each task building its own `ctx ()` first, which `f` reads. What a task builds is its own, so
reading it touches no reference count other tasks touch: entries of one table shared by every task are counted
atomically by all of them at once. -/
@[specialize] def parGenWith {β : Type} (ctx : Unit → β) (f : β → Nat → F128) (n : Nat) : Array F128 :=
  let a := n / 4096
  ((Array.range a).map fun i => Task.spawn fun _ =>
      let c := ctx (); (Array.range 4096).map fun j => f c (i * 4096 + j)).foldl (fun acc t => acc ++ t.get) #[] ++
    (let c := ctx (); (Array.range (n - a * 4096)).map fun j => f c (a * 4096 + j))

theorem parGenWith_eq {β : Type} (ctx : Unit → β) (f : β → Nat → F128) (n : Nat) :
    parGenWith ctx f n = parGen (f (ctx ())) n := rfl

/-- `mulMap` in tasks, each with its own table of `r`, expanded from one `nibTable r` they share. -/
def mulMapPar (t : Array F128) (r : F128) : Array F128 :=
  if t.size < 2048 then parMap (· * r) t else
    let nb := F128.nibTable r
    parGenWith (fun _ => F128.expandNib nb) (fun tb i => F128.mulBy tb t[i]!) t.size

theorem mulMapPar_eq (t : Array F128) (r : F128) : mulMapPar t r = t.map (· * r) := by
  unfold mulMapPar
  split
  · exact parMap_eq _ t
  · rw [parGenWith_eq]
    show parMap (F128.mulBy (F128.expandNib (F128.nibTable r))) t = _
    rw [parMap_eq, F128.expandNib_nibTable]
    exact Array.map_congr_left fun v _ => F128.mulBy_mulTable r v

/-- `eqTable` with each step in tasks: bit 1's half by `mulMapPar`, then bit 0's as bit 1's plus the table
(`F128.mul_add_one`), as `eqStepFast` computes it. Not a `@[csimp]` of `eqTable`, which the verifier's compiled path
reaches (`verifier` in `lean-audit.json`): only the fast paths below call it. -/
def eqTablePar (r : Array F128) : Array F128 :=
  r.foldl (fun t rj => let hi := mulMapPar t rj; parGen (fun i => hi[i]! + t[i]!) t.size ++ hi) #[1]

theorem eqTablePar_eq (r : Array F128) : eqTablePar r = eqTable r := by
  unfold eqTablePar eqTable
  congr
  funext t rj
  rw [mulMapPar_eq]
  dsimp only
  rw [parGen_eq, eqStep, ← range_map_getElem! (· * (rj + 1))]
  congr 1
  apply Array.map_congr_left
  intro i hi
  have hi := Array.mem_range.1 hi
  rw [getElem!_eq _ (by simpa using hi), Array.getElem_map, F128.mul_add_one, getElem!_eq t hi]

/-- `switchVec`'s sums over the words `lo, …, lo + n − 1`: byte `j`'s buckets (`byteBuckets`), then
`s[8j + i] = Σ_{b < 256, bit_i(b)} buckets_j[b]`. -/
def switchPart (g e : Array F128) (lo n : Nat) : Array F128 :=
  let bs := (Array.range 16).map fun j => byteBuckets g e j lo n
  (Array.range 128).map fun v => sumF 256 fun b => if b.testBit (v % 8) then (bs[v / 8]!)[b]! else 0

theorem switchPart_get (g e : Array F128) (lo n : Nat) {v : Nat} (hv : v < 128) :
    (switchPart g e lo n)[v]! = sumF n fun y => if (g[lo + y]!).bit v then e[lo + y]! else 0 := by
  unfold switchPart
  rw [getElem!_range_map _ hv, getElem!_range_map _ (show v / 8 < 16 by omega)]
  have h := (byteBuckets_go g e (v / 8) lo (by omega) (show v % 8 < 8 by omega) n).2
  rwa [show 8 * (v / 8) + v % 8 = v by omega] at h

/-- `switchVec` as compiled code computes it: `switchPart` of each `2^14` words in a task, the parts added. -/
def switchVecFast (g : Array F128) (c : Claim) : Array F128 :=
  let eY := eqTable (c.x.extract 1 c.x.size)
  let a := eY.size / 16384
  let ps := ((Array.range a).map fun i => Task.spawn fun _ => switchPart g eY (16384 * i) 16384).map Task.get
  let r := switchPart g eY (16384 * a) (eY.size - 16384 * a)
  (Array.range 128).map fun v => sumF a (fun i => (ps[i]!)[v]!) + r[v]!

theorem switchVec_eq_fast (g : Array F128) (c : Claim) : switchVec g c = switchVecFast g c := by
  unfold switchVec switchVecFast
  generalize eqTable (c.x.extract 1 c.x.size) = e
  apply Array.map_congr_left
  intro v hv
  have hv := Array.mem_range.1 hv
  rw [switchPart_get _ _ _ _ hv]
  have hp : (sumF (e.size / 16384) fun i =>
      (((Array.range (e.size / 16384)).map fun i => Task.spawn fun _ => switchPart g e (16384 * i) 16384).map
        Task.get)[i]![v]!) =
      sumF (e.size / 16384) fun i => sumF 16384 fun y => if (g[16384 * i + y]!).bit v then e[16384 * i + y]! else 0 :=
    sumF_congr fun i hi => by
      rw [Array.map_map, getElem!_range_map _ hi]
      exact switchPart_get _ _ _ _ hv
  rw [hp]
  conv => lhs; rw [← Nat.add_sub_cancel' (Nat.mul_div_le e.size 16384)]
  rw [sumF_add, sumF_blocks]

theorem tasks_get {α β : Type} (f : α → β) (xs : Array α) :
    (xs.map fun x => Task.spawn fun _ => f x).map Task.get = xs.map f := by
  rw [Array.map_map]
  rfl

/-- `basisVals` as compiled code computes it: each claim's `γ_k·Φ_k(eq(z_k, y))` from the byte tables of
`γ_k·eq(rpp_k, ·)` (`mul_sumF`, `byteTable_sum`), in tasks of `2^12` points. Each claim's tables are a task of their own,
`eqTable` serial in it: `eqTablePar`'s tasks inside them only contend for the same cores. -/
def basisValsFast (sws : Array Switched) (l : Nat) : Array F128 :=
  let tabs := (sws.map fun sw => Task.spawn fun _ =>
    (eqTable sw.z, byteTable ((eqTable sw.rpp).map (sw.gamma * ·)))).map Task.get
  let k := min l 12
  parRangeS (fun y => tabs.foldl (fun acc (ez, tb) =>
      let v := ez[y]!
      acc + sumF 16 fun j => tb[256 * j + v.byte j]!) 0) (2 ^ (l - k)) (2 ^ k)

theorem basisVals_eq_fast (sws : Array Switched) (l : Nat) : basisVals sws l = basisValsFast sws l := by
  unfold basisVals basisValsFast
  rw [tasks_get fun sw : Switched => (eqTable sw.z, byteTable ((eqTable sw.rpp).map (sw.gamma * ·))), parRangeS_eq,
    ← Nat.pow_add, Nat.sub_add_cancel (Nat.min_le_left _ _)]
  apply Array.map_congr_left
  intro y _
  simp only [Array.foldl_map]
  congr 1
  funext acc sw
  rw [← byteTable_sum, mul_sumF]
  refine congrArg (acc + ·) (sumF_congr fun u _ => ?_)
  rw [getElem!_map_mul]
  split
  · rfl
  · exact F128.mul_zero _

end Fast

/-- Compiled code runs `switchVecFast` and `basisValsFast`; the kernel and every proof see `switchVec` and `basisVals`.
`@[csimp]` rewrites only code compiled after it, so it stays above `repOpening`. -/
@[csimp] theorem switchVec_eq_switchVecFast : @switchVec = @Fast.switchVecFast :=
  funext fun g => funext fun c => Fast.switchVec_eq_fast g c

@[csimp] theorem basisVals_eq_basisValsFast : @basisVals = @Fast.basisValsFast :=
  funext fun sws => funext fun l => Fast.basisVals_eq_fast sws l

/-- `c + Σ a_i h_i` at the pads `h`, over `min |a| |h|` terms (`zk_veil::Lin::eval`). -/
def linAt (l : Zk.Lin) (h : Array F128) : F128 :=
  (Array.range (min l.a.size h.size)).foldl (fun acc i => acc + l.a[i]! * h[i]!) l.c

/-- The names of the constraints `cons` that fail on the pads `h` (`zk_veil::failing`). -/
def failingOn (cons : Array (String × Zk.Lin)) (h : Array F128) : Array String :=
  (cons.filter fun c => linAt c.2 h != 0).map (·.1)

/-- **A rep's PIOP**: the masked zerocheck and lincheck sending `mk i` at pad index `i`, the pads committed under `root`;
the claims they leave, `abcClaims ++ extras`, are their coins' alone. -/
def repPiop (io : Asks M) (st : Setup) (lay : Zk.PadLayout) (cap root : ByteArray) (extras : Array Claim)
    (mk : Nat → F128) : TMW M (ZerocheckProof × LincheckProof × Array Claim) := do
  let (zp, zc) ← zerocheckFw io st.m st.digest cap root lay mk
  let (lp, lc) ← lincheckFw io st.fold lay mk zc
  return (zp, lp, abcClaims st.m st.kLog zc lc ++ extras)

/-- **A rep's opening** after its PIOP: the ring switch on `claims` and the Ligerito opening of the packed table `g`
committed as `cm0`, level 0 masked by the pads `h`, levels ≥ 1 committed by `commit` under `salt`, every grinding nonce
`0`. -/
def repOpening (io : Asks M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (st : Setup)
    (sch : Flock.Schedule) (lay : Zk.PadLayout) (tPad rep : Nat) (g : Array F128) (cm0 : Commit) (claims : Array Claim)
    (h : Array F128) (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) :
    TMW M (Array (Array F128) × LigeritoProof × Zk.Level0Msgs × Zk.L0Rec) := do
  let (ss, T, sws) ← ringSwitchP io (fun k => switchVec g claims[k]!) claims
  let (lg, zl, rec) ← ligeritoPZk io commit st.merkle sch T cm0 g (basisVals sws (st.m - 7)) lay h rep tPad extra
    lanePad salt (fun _ _ => 0)
  return (ss, lg, zl, rec)

/-- **A rep up to its inner proof**, as rep `rep` of a stream whose transcript starts at `dom`: its PIOP on the passed
values `vals` (by pad index) masked by the rep's pads `pads` committed as `ic`, its opening (`repOpening`) on the
statement's extra claims `extras` and the table's extra lanes and paddings, then the verifier's own replay of them and the
ring-switch and lane-phase constraints. Returns the proof, level 0's hiding messages and the constraints the inner proof
batches. -/
def repCons (io : Asks M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (st : Setup)
    (sch : Flock.Schedule) (tPad rep : Nat) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (vals : Array F128) (extra lanePad : Nat → Nat → F128)
    (salt : Nat → Nat → ByteArray) : TMW M (Proof × Zk.Level0Msgs × Array (String × Zk.Lin)) := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let (zp, lp, claims) ← repPiop io st lay (flat cm0.cap) ic.root extras fun i => vals[i]! + pads.h[i]!
  let (ss, lg, zl, rec) ← repOpening io commit st sch lay tPad rep g cm0 claims pads.h extra lanePad salt
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  return (p, zl, rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc)

/-- **The firewall's proof of one rep**: `repCons`, then the inner proof on the rep's pads and the constraints it returns.
A constraint that fails on the pads stops the rep before its inner proof, and no file leaves (`zk_finish_fw`'s
`constraints_failing_on_pads`, which stops the session at its `Finish`). -/
def repFw (io : Asks M) (hs : Hashes M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (st : Setup)
    (sch : Flock.Schedule) (tPad rep : Nat) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (vals : Array F128) (extra lanePad : Nat → Nat → F128)
    (salt : Nat → Nat → ByteArray) : TMW M Zk.ZkProof := do
  let (p, zl, cons) ← repCons io commit st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt
  let fails := failingOn cons pads.h
  if !fails.isEmpty then throw s!"{Frame.STOP}the constraints {", ".intercalate fails.toList} fail on the rep's pads"
  let inner ← innerP io hs (Zk.PadLayout.new st.m st.kLog sch.k0) pads ic p.zerocheck.finalC cons
  return { proof := p, level0 := zl, inner }

/-- **A rep's rounds answered by its coins**: round `k` (counted from `0` in the state) by `coins[k]`, which must hold the
round's coin count (its squeeze header's last eight bytes). -/
def asksOf (coins : Array (Array F128)) : Asks (StateT Nat V) where
  link _ _ := throw "finish: a link message inside a rep"
  round c := do
    let k ← get
    let some cs := coins[k]? | throw s!"finish: round {k} past the rep's {coins.size} rounds of coins"
    let n := (u64OfLE c (c.size - 8)).toNat
    if cs.size != n then throw s!"finish: round {k} squeezes {n} coins where the rep's coins have {cs.size}"
    set (k + 1)
    return fun i => cs[i]!

/-! ## What the firewall hands it -/

/-- **One table's state**: its statement, its `--zk` schedule and `t_pad`, the packed table, its extra lanes and lane
paddings, and level 0's commitment. -/
structure Tab where
  st : Setup
  sch : Flock.Schedule
  tPad : Nat
  g : Array F128
  extra : Nat → Nat → F128
  lanePad : Nat → Nat → F128
  cm0 : Commit

/-- **What one rep's proof reads beyond its table**: its draws (`Frame.Draws`, the Lean frame's), the values that passed
the firewall by pad index, the extra claims' coordinates, levels ≥ 1's salts by level (`192` bytes a leaf, `hiSaltsOf`'s),
and each round's coins. -/
structure RepIn where
  draws : Frame.Draws
  vals : Array F128
  xs : Array (Array F128)
  hiSalts : Array ByteArray
  coins : Array (Array F128)

/-- **What the session loop holds of a rep**, which its finish holds the rep's requests to (`RepIn.differs`): the draws its
frame took at the rep's start, the values the frame took by pad index, the extra claims at the session's link points, and
the coins the stream answered each of the rep's rounds so far. -/
structure Own where
  draws : Frame.Draws
  vals : Array F128
  xs : Array (Array F128)
  coins : Array (Array F128)

/-- **Where the request `q` is not what the loop holds of the rep (`o`)**, or none: its draws, its passed values, its extra
claims or its coins. `final_c`'s slot (`lay.fc`) is the request's: the passed values fix it, and where it is another, the
rep's `zerocheck c_eval` constraint fails on the loop's pads and the rep stops (`repFw`, `failingOn`). -/
def RepIn.differs (lay : Zk.PadLayout) (q : RepIn) (o : Own) : Option String :=
  let (d, e) := (q.draws, o.draws)
  if !(d.h == e.h && d.mu == e.mu && d.padH == e.padH && d.padMu == e.padMu) then some "its pads are not the frame's draws"
  else if !(d.tauSalt == e.tauSalt && d.salts == e.salts) then some "its salts are not the frame's draws"
  else if !(q.vals.size == o.vals.size && (Array.range q.vals.size).all fun i => i == lay.fc || q.vals[i]! == o.vals[i]!) then
    some "its passed values are not the ones the frame took"
  else if q.xs != o.xs then some "its extra claims are not the ones at the session's link points"
  else if q.coins != o.coins then some "its coins are not the ones the stream answered"
  else none

/-- A rep's pads from its draws (`FlockComplete.padsOf`'s: `h_ab` and `w` set by `Draws.ofBytes`). -/
def padsOf (d : Frame.Draws) : PadsZk :=
  { h := d.h, mu := d.mu, padH := d.padH, padMu := d.padMu, tauSalt := fitBytes Zk.TAU_SALT_BYTES d.tauSalt }

/-- The extra claims at their coordinates: the prover reads a claim's coordinates alone. -/
def claimsOf (xs : Array (Array F128)) : Array Claim := xs.map fun x => { value := 0, skip := #[], x }

/-- The `192`-byte salt at position `q` of `b`. -/
def saltAt (b : ByteArray) (q : Nat) : ByteArray := b.extract (192 * q) (192 * q + 192)

/-- **The rep's proof file**: `repFw` on the table `t` and the rep's inputs `q`, answered by `q`'s coins, every round of
which it must use; its `encZk` at `shapeZk`. -/
def proofFile (t : Tab) (rep : Nat) (dom : ByteArray) (q : RepIn) : V ByteArray := do
  let pads := padsOf q.draws
  let ic ← (innerCommitP (Hashes.ofSha512 Sha512.hash) pads (saltAt q.draws.salts) : V InnerCommit)
  let extras := claimsOf q.xs
  let hs : Hashes (StateT Nat V) := Hashes.ofSha512 Sha512.hash
  let commit : Level → Array F128 → (Nat → ByteArray) → StateT Nat V Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)
  let ((p, _), k) ← ((repFw (asksOf q.coins) hs commit t.st t.sch t.tPad rep dom t.g t.cm0 extras pads ic q.vals t.extra
    t.lanePad salt).run (Transcript.init #[] dom)).run 0
  if k != q.coins.size then throw s!"finish: the rep took {k} of its {q.coins.size} rounds of coins"
  return encZk (shapeZk t.st t.sch extras.size) p

/-! ## The request's bytes -/

/-- A request's bytes and how far they are read. -/
abbrev Rd := StateT Nat (ReaderT ByteArray V)

def take (n : Nat) (what : String) : Rd ByteArray := do
  let o ← get
  let b ← read
  if o + n > b.size then throw s!"finish: the request ends inside {what}"
  set (o + n)
  return b.extract o (o + n)

def u64 (what : String) : Rd Nat := do
  let b ← take 8 what
  return (u64OfLE b 0).toNat

def wordsOf (n : Nat) (what : String) : Rd (Array F128) := do
  let b ← take (16 * n) what
  return Link.wordsAt b 0 n

/-- `n` words after their `u64` count. -/
def vecOf (what : String) : Rd (Array F128) := do
  let n ← u64 what
  wordsOf n what

/-- **A rep's request** for the table `t`, every byte little-endian: the draws (`Frame.drawBytes` of the layout: `h`, `μ`,
the rows' padding, `τ`'s salt, the pads tree's salts), the passed values (`lay.zp 64` words, by pad index), the
extra claims (a count, then each as a count and its words), and the coins (a round count, then each round as a count
and its words); nothing after. Levels ≥ 1's salts are `hiSalts`, the ones drawn here for the rep (`hiSaltsOf`). -/
def parseRep (t : Tab) (hiSalts : Array ByteArray) (b : ByteArray) : V RepIn := do
  let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
  let rd : Rd RepIn := do
    let draws := Frame.Draws.ofBytes lay (← take (Frame.drawBytes lay) "the draws")
    let vals ← wordsOf (lay.zp 64) "the passed values"
    let nx ← u64 "the extra claims"
    let mut xs := #[]
    for _ in [0:nx] do xs := xs.push (← vecOf "an extra claim")
    let nk ← u64 "the coins"
    let mut coins := #[]
    for _ in [0:nk] do coins := coins.push (← vecOf "a round's coins")
    if (← get) != b.size then throw s!"finish: {b.size - (← get)} bytes after the rep's request"
    return { draws, vals, xs, hiSalts, coins }
  (rd.run' 0).run b

/-- **A table's state** from a table's request bytes (`Link.parse`) at the statement `st`. -/
def tabOf (st : Setup) (b : ByteArray) : V Tab := do
  if st.merkle.id != MerkleScheme.hm96Sha512.id then throw s!"finish: a {st.merkle.id} statement: --zk commits hm96-sha512/v1"
  let (sch, tPad) ← Zk.schedule st.m
  let i ← Link.parse st.m b
  return { st, sch, tPad, g := i.g, extra := i.extra, lanePad := i.lanePad
           cm0 := commit0 (sch.levels.getD 0 default) tPad i }

/-- **A table's state** from its level 0 as the session loop keeps it (`Level0`) at the statement `st`: the loop's
witness, hiding and commitment (`commitOf`), at level 0 of `st`'s `--zk` schedule. -/
def tabOfLevel0 (st : Setup) (l : Level0.Level0) : V Tab := do
  if st.merkle.id != MerkleScheme.hm96Sha512.id then throw s!"finish: a {st.merkle.id} statement: --zk commits hm96-sha512/v1"
  if l.m != st.m then throw s!"finish: a level 0 drawn at m = {l.m} for a table at m = {st.m}"
  let (sch, tPad) ← Zk.schedule st.m
  if l.tPad != tPad then throw s!"finish: a level 0 with t_pad {l.tPad} for a table whose schedule's is {tPad}"
  let i := l.toIn
  return { st, sch, tPad, g := l.g, extra := i.extra, lanePad := i.lanePad, cm0 := commitOf l }

/-- **A rep's salts for levels ≥ 1** of the table `t`, drawn from `get`: `192·2^d` bytes for each level `1` to `R` of its
schedule, one read a level. The prover's trees at those levels take them (`salts`), so they never come from it. -/
def hiSaltsOf (get : USize → IO ByteArray) (t : Tab) : IO (Array ByteArray) :=
  (Array.range (t.sch.levels.size - 1)).mapM fun j => get (192 * 2 ^ (t.sch.levels.getD (j + 1) default).d).toUSize

/-- The salts as they go to the prover: their count, then each level's byte count and bytes, every count a `u64le`. -/
def saltsBytes (s : Array ByteArray) : ByteArray :=
  s.foldl (fun b x => b ++ le64 x.size.toUInt64 ++ x) (le64 s.size.toUInt64)

/-! ## The live rep -/

/-- A refusal of the finish's pure code, as the live rep's. -/
instance liftV : MonadLift V (ExceptT String IO) := ⟨fun x => ExceptT.mk (pure x)⟩

/-- What a live rep runs in: the coins of every round it has taken, over `IO`, with its refusals. -/
abbrev LiveM := StateT (Array (Array F128)) (ExceptT String IO)

/-- **A rep's rounds, live** on `ch`: round `k` (the rounds taken so far) by `coins[k]` while the rep's recorded coins last,
which must hold the round's coin count (its squeeze header's last eight bytes), as `asksOf`. Past them, with `live`, the
round goes to the firewall as `round N LEN` followed by its `LEN` bytes, `N` its coin count, and is answered `coins N`
followed by the `16 N` bytes of the coins the stream answered it; any other answer, or a round past them without `live`,
ends the rep. -/
def liveAsks (ch : Frame.Chan) (coins : Array (Array F128)) (live : Bool) : Asks LiveM where
  link _ _ := throwThe String "finish: a link message inside a rep"
  round c := do
    let got ← get
    let k := got.size
    let n := (u64OfLE c (c.size - 8)).toNat
    let cs ← match coins[k]?, live with
      | some cs, _ => pure cs
      | none, false => throwThe String s!"finish: round {k} past the rep's {coins.size} rounds of coins"
      | none, true => do
        ch.write ((s!"round {n} {c.size}\n").toUTF8 ++ c)
        ch.flush
        let line := (← ch.getLine).replace "\n" ""
        if line != s!"coins {n}" then
          throwThe String s!"finish: round {k}, of {n} coins, answered `{line}`"
        pure (Link.wordsAt (← ch.readExact (16 * n)) 0 n)
    if cs.size != n then throwThe String s!"finish: round {k} squeezes {n} coins where the rep's coins have {cs.size}"
    set (got.push cs)
    return fun i => cs[i]!

/-- The constraints as they go to the firewall: their count, then each as its name's byte count and bytes, its constant and
its coefficients (a count and the words), every count a `u64le`. -/
def consBytes (cons : Array (String × Zk.Lin)) : ByteArray :=
  cons.foldl (fun b (n, l) => b ++ wBytes n.toUTF8 ++ l.c.toBytes ++ wVec F128.toBytes l.a) (wLen cons.size)

/-- **A rep's opening, live**: `repFw`'s own PIOP (`repPiop`) on its recorded coins `coins` (the algebraic phase's, every
one of which it must take) and its own opening (`repOpening`) on `abcClaims ++ extras` (the extra claims at `xs`) with every
value and pad zero, levels ≥ 1 under the rep's `hiSalts`, each round sent live (`liveAsks`): the device's opening, its
level-0 values unmasked, which the Lean frame holds the device's rounds to and masks on release. Nothing it reads in the
PIOP but the coins shapes a round it sends. -/
def openLive (ch : Frame.Chan) (t : Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128))
    (hiSalts : Array ByteArray) (coins : Array (Array F128)) : LiveM Unit := do
  let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
  let commit : Level → Array F128 → (Nat → ByteArray) → LiveM Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (hiSalts.getD (l - 1) ByteArray.empty)
  let prog : TMW LiveM Unit := do
    let (_, _, claims) ← repPiop (liveAsks ch coins false) t.st lay (flat t.cm0.cap) ByteArray.empty (claimsOf xs)
      fun _ => 0
    let k := (← getThe (Array (Array F128))).size
    if k != coins.size then throwThe String s!"finish: the rep's PIOP took {k} of its {coins.size} rounds of coins"
    let _ ← repOpening (liveAsks ch coins true) commit t.st t.sch lay t.tPad rep t.g t.cm0 claims
      (Array.replicate lay.len 0) t.extra t.lanePad salt
  let _ ← prog.run (Transcript.init #[] dom)

/-- **A rep's inner proof, live**: `repFw` on the request `q` but for its stop, its `repCons` on `q`'s recorded coins (every
one of which it must take), then its constraints sent to the firewall as `cons LEN` followed by their `LEN` bytes
(`consBytes`), then `repFw`'s inner proof (`innerP`) on them, each round sent live (`liveAsks`). Where `repFw` stops on a
constraint that fails on the pads, this speaks on schedule (`proofFile` then refuses the rep's file, and the session
stops at its `Finish`). -/
def innerLive (ch : Frame.Chan) (t : Tab) (rep : Nat) (dom : ByteArray) (q : RepIn) : LiveM Unit := do
  let pads := padsOf q.draws
  let ic ← (monadLift (innerCommitP (Hashes.ofSha512 Sha512.hash) pads (saltAt q.draws.salts) : V InnerCommit) : LiveM InnerCommit)
  let hs : Hashes LiveM := Hashes.ofSha512 Sha512.hash
  let commit : Level → Array F128 → (Nat → ByteArray) → LiveM Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)
  let prog : TMW LiveM Unit := do
    let (p, _, cons) ← repCons (liveAsks ch q.coins false) commit t.st t.sch t.tPad rep dom t.g t.cm0 (claimsOf q.xs)
      pads ic q.vals t.extra t.lanePad salt
    let k := (← getThe (Array (Array F128))).size
    if k != q.coins.size then throwThe String s!"finish: the rep's PIOP and opening took {k} of its {q.coins.size} rounds of coins"
    let b := consBytes cons
    ch.write ((s!"cons {b.size}\n").toUTF8 ++ b)
    ch.flush
    let _ ← innerP (liveAsks ch q.coins true) hs (Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) pads ic p.zerocheck.finalC
      cons
  let _ ← prog.run (Transcript.init #[] dom)

/-- **A rep's inner proof, live, kept**: `innerLive`, returning `repFw`'s proof where no constraint fails on the pads
(`innerLiveK_discard`): `repFw` on `q` with every coin this took, which is the proof `proofFile` computes on a `rep`
request with those inputs and coins (`Kept`). -/
def innerLiveK (ch : Frame.Chan) (t : Tab) (rep : Nat) (dom : ByteArray) (q : RepIn) : LiveM (Option Zk.ZkProof) := do
  let pads := padsOf q.draws
  let ic ← (monadLift (innerCommitP (Hashes.ofSha512 Sha512.hash) pads (saltAt q.draws.salts) : V InnerCommit) : LiveM InnerCommit)
  let hs : Hashes LiveM := Hashes.ofSha512 Sha512.hash
  let commit : Level → Array F128 → (Nat → ByteArray) → LiveM Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)
  let prog : TMW LiveM (Option Zk.ZkProof) := do
    let (p, zl, cons) ← repCons (liveAsks ch q.coins false) commit t.st t.sch t.tPad rep dom t.g t.cm0 (claimsOf q.xs)
      pads ic q.vals t.extra t.lanePad salt
    let k := (← getThe (Array (Array F128))).size
    if k != q.coins.size then throwThe String s!"finish: the rep's PIOP and opening took {k} of its {q.coins.size} rounds of coins"
    let b := consBytes cons
    ch.write ((s!"cons {b.size}\n").toUTF8 ++ b)
    ch.flush
    let inner ← innerP (liveAsks ch q.coins true) hs (Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) pads ic p.zerocheck.finalC
      cons
    return if (failingOn cons pads.h).isEmpty then some { proof := p, level0 := zl, inner } else none
  return (← prog.run (Transcript.init #[] dom)).1

instance instLawfulEST {ε σ : Type} : LawfulMonad (EST ε σ) := LawfulMonad.mk' (EST ε σ)
  (id_map := fun x => by funext s; show EST.bind x (fun a => EST.pure (id a)) s = x s; unfold EST.bind EST.pure; cases x s <;> rfl)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x f g => by
    funext s; show EST.bind (EST.bind x f) g s = EST.bind x (fun a => EST.bind (f a) g) s
    unfold EST.bind; cases x s <;> rfl)

instance instLawfulEIO {ε : Type} : LawfulMonad (EIO ε) := inferInstanceAs (LawfulMonad (EST ε IO.RealWorld))

/-- `innerLiveK` is `innerLive` with its result kept: the same rounds, the same bytes to the firewall, the same coins. -/
theorem innerLiveK_discard (ch : Frame.Chan) (t : Tab) (rep : Nat) (dom : ByteArray) (q : RepIn) :
    (do let _ ← innerLiveK ch t rep dom q) = innerLive ch t rep dom q := by
  unfold innerLiveK innerLive
  simp only [StateT.run_bind, map_bind, bind_pure_comp]
  congr 1; funext ic; congr 1; funext a; congr 1; funext b
  have h : ∀ {α} (X : TMW LiveM α) s,
      (fun _ => ()) <$> Prod.fst <$> X.run s = (fun _ => ()) <$> ((fun _ => ()) <$> X).run s := by
    intros; simp only [StateT.run_map, Functor.map_map]
  rw [h]; congr 2
  split <;> simp only [map_bind, Functor.map_map]

/-- **What a rep's `inner` request leaves for its `rep` request**: table `i`'s rep `r`, its request `q`, every round's
coins it took, and `repFw`'s proof on them (`innerLiveK`). -/
structure Kept where
  i : Nat
  r : Nat
  q : RepIn
  coins : Array (Array F128)
  proof : Zk.ZkProof

/-- Whether `k` is `repFw`'s proof on table `i`'s rep `r` at the request `q`: everything of `q` that `proofFile` reads is
`k`'s, and `q`'s coins are every coin `k`'s rep took. -/
def Kept.answers (k : Kept) (i r : Nat) (q : RepIn) : Bool :=
  k.i == i && k.r == r && q.coins == k.coins && q.vals == k.q.vals && q.xs == k.q.xs && q.hiSalts == k.q.hiSalts &&
    q.draws.h == k.q.draws.h && q.draws.mu == k.q.draws.mu && q.draws.padH == k.q.draws.padH &&
    q.draws.padMu == k.q.draws.padMu && q.draws.tauSalt == k.q.draws.tauSalt && q.draws.salts == k.q.draws.salts

/-- **A rep's opening request**, every byte little-endian: the extra claims (a count, then each as a count and its words)
and the algebraic phase's coins (a round count, then each round as a count and its words), as `parseRep` reads them;
nothing after. -/
def parseOpen (b : ByteArray) : V (Array (Array F128) × Array (Array F128)) := do
  let rd : Rd (Array (Array F128) × Array (Array F128)) := do
    let nx ← u64 "the extra claims"
    let mut xs := #[]
    for _ in [0:nx] do xs := xs.push (← vecOf "an extra claim")
    let nk ← u64 "the coins"
    let mut coins := #[]
    for _ in [0:nk] do coins := coins.push (← vecOf "a round's coins")
    if (← get) != b.size then throw s!"finish: {b.size - (← get)} bytes after the opening's request"
    return (xs, coins)
  (rd.run' 0).run b

/-! ## Serving it -/

/-- **Why table `i`'s rep `r`'s request on the table `t` is refused**, where the session loop holds the rep (`own`, by the
table's level-0 root): it holds none, or `differs` of what it holds; none where no loop holds it. -/
def ownWhy (own : Nat → Nat → ByteArray → IO (Option (Except String Own))) (i r : Nat) (t : Tab)
    (differs : Own → Option String) : IO (Option String) := do
  match ← own i r (flat t.cm0.cap) with
  | none => return none
  | some (.error e) => return some s!"finish: {e}"
  | some (.ok o) => return (differs o).map fun why => s!"finish: table {i} rep {r}'s request: {why}"

/-- **The finish's requests** on `ch`, at the session's statement (`spec`, its tables' `setups`), each a line `HEAD LEN`
followed by `LEN` bytes, answered `ok N` followed by `N` bytes:
* `table I LEN`: table `I`'s level-0 data (`Link.parse`'s, `tabOf`), kept; the answer is its level-0 root.
* `level0 I LEN`: the root of a level 0 that `level0Of` holds (the session loop's), kept as table `I` (`tabOfLevel0`);
  the answer is its root.
* `salts I R 0`: rep `R`'s salts for levels ≥ 1 on table `I`, drawn from `get` once (`hiSaltsOf`), kept for its proof
  file and handed to `keep` with the table's level-0 root; the answer is them (`saltsBytes`). A rep's salts are drawn
  once: asking again is refused.
* `open I R LEN`: rep `R`'s opening on table `I`, after its salts, on the extra claims and the algebraic phase's coins
  (`parseOpen`): each of its rounds goes to the firewall as it comes and waits for its coins (`openLive`, `liveAsks`), the
  rounds the Lean frame holds the device's opening rounds to; answered `ok 0` after the last.
* `inner I R LEN`: rep `R`'s request (`parseRep`) with the coins of its PIOP and opening: its constraints, then its inner
  proof's rounds, live (`innerLive`); answered `ok 0` after the last. Its proof file is then `rep`'s on every round's coins.
* `rep I R LEN`: rep `R`'s request (`parseRep`) on table `I`, after its salts; the answer is its proof file
  (`proofFile`), encoded from the proof the rep's last `inner` request computed where that request's inputs and coins
  are this one's (`Kept.answers`). Another rep's requests may come between the two. The file is handed to `file` with
  the table's level-0 root before it is answered, and where `file` refuses it, so does this.
Where the session loop holds the rep (`own`, by table, rep and the table's level-0 root), an `open`, `inner` or `rep`
request that is not what it holds is refused (`RepIn.differs`; an opening's extra claims and coins): the draws, the
passed values, the extra claims and the coins are the loop's own, whatever the host sends.
With `liveOnly` (the session loop's finish, `serveLoop`), a table comes only as the loop's level 0 (`level0`, never
`table`), and a rep's proof file is only the one its last `inner` request computed live (`Kept.answers`): a `rep` request
no such proof answers is refused, never computed afresh from the request.
Anything else, or a request it refuses, is answered `err WHY` and ends it, with exit code `1`. -/
def serve (ch : Frame.Chan) (get : USize → IO ByteArray) (spec : SessionSpec) (setups : Array Setup)
    (level0Of : ByteArray → IO (Option Level0.Level0))
    (keep : ByteArray → Nat → Array ByteArray → IO Unit := fun _ _ _ => pure ())
    (file : ByteArray → Nat → ByteArray → IO (Option String) := fun _ _ _ => pure none)
    (own : Nat → Nat → ByteArray → IO (Option (Except String Own)) := fun _ _ _ => pure none)
    (liveOnly : Bool := false) : IO UInt32 := do
  let refuse := fun (why : String) => do
    ch.write (s!"err {why.replace "\n" " "}\n").toUTF8
    ch.flush
    return (1 : UInt32)
  let answer := fun (b : ByteArray) => do
    ch.write (s!"ok {b.size}\n").toUTF8
    ch.write b
    ch.flush
  let mut tabs : Array (Option Tab) := Array.replicate setups.size none
  let mut hi : Array (Array (Option (Array ByteArray))) := Array.replicate setups.size #[none, none]
  let mut kept : Array Kept := #[]
  repeat
    let line := (← ch.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["table", i, len] | ["level0", i, len] =>
      let (some i, some len) := (i.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let some st := setups[i]? | return (← refuse s!"table {i} of the session's {setups.size}")
      let b ← ch.readExact len
      if liveOnly && line.startsWith "table " then
        return (← refuse s!"finish: table {i}'s level-0 data from the host, where the session loop's finish takes only its own level 0")
      let t : V Tab ← if line.startsWith "table " then pure (tabOf st b) else do
        match ← level0Of b with
        | some l => pure (tabOfLevel0 st l)
        | none => pure (.error s!"finish: table {i}'s root is no level 0's the session loop drew")
      match t with
      | .ok t =>
        tabs := tabs.set! i (some t)
        hi := hi.set! i #[none, none]
        kept := kept.filter (·.i != i)
        answer (flat t.cm0.cap)
      | .error e => return (← refuse e)
    | ["salts", i, r, len] =>
      let (some i, some r, some len) := (i.toNat?, r.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let _ ← ch.readExact len
      let some (some t) := tabs[i]? | return (← refuse s!"rep {r}'s salts of table {i} before its table")
      if r > 1 then return (← refuse s!"rep {r}: a table has reps 0 and 1")
      if ((hi[i]!)[r]!).isSome then return (← refuse s!"table {i} rep {r}'s salts for levels ≥ 1 were drawn already")
      let s ← hiSaltsOf get t
      hi := hi.set! i ((hi[i]!).set! r (some s))
      keep (flat t.cm0.cap) r s
      answer (saltsBytes s)
    | ["rep", i, r, len] =>
      let (some i, some r, some len) := (i.toNat?, r.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let b ← ch.readExact len
      let some (some t) := tabs[i]? | return (← refuse s!"a rep of table {i} before its table")
      let some name := spec.tables[i]? | return (← refuse s!"table {i} of the session's {spec.tables.size}")
      if r > 1 then return (← refuse s!"rep {r}: a table has reps 0 and 1")
      let some s := (hi[i]!)[r]! | return (← refuse s!"table {i} rep {r} before its salts for levels ≥ 1")
      let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
      if let .ok q := parseRep t s b then
        if let some why ← ownWhy own i r t (q.differs lay) then return (← refuse why)
      let pf : V ByteArray := parseRep t s b >>= fun q => match kept.find? (·.answers i r q) with
        | some k => pure (encZk (shapeZk t.st t.sch (claimsOf q.xs).size) k.proof)
        | none =>
          if liveOnly then
            .error s!"finish: table {i} rep {r}'s proof file is only the one its inner request computed live, and none answers this request"
          else proofFile t r (spec.domain name r).toUTF8 q
      kept := kept.filter fun k => !(k.i == i && k.r == r)
      match pf with
      | .ok p =>
        if let some why ← file (flat t.cm0.cap) r p then return (← refuse why)
        answer p
      | .error e => return (← refuse e)
    | ["open", i, r, len] | ["inner", i, r, len] =>
      let (some i, some r, some len) := (i.toNat?, r.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let b ← ch.readExact len
      let some (some t) := tabs[i]? | return (← refuse s!"a rep of table {i} before its table")
      let some name := spec.tables[i]? | return (← refuse s!"table {i} of the session's {spec.tables.size}")
      if r > 1 then return (← refuse s!"rep {r}: a table has reps 0 and 1")
      let some s := (hi[i]!)[r]! | return (← refuse s!"table {i} rep {r} before its salts for levels ≥ 1")
      let dom := (spec.domain name r).toUTF8
      let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
      let why ← if line.startsWith "open " then
          match parseOpen b with
          | .ok (xs, coins) => ownWhy own i r t fun o =>
            if xs != o.xs then some "its extra claims are not the ones at the session's link points"
            else if coins != o.coins then some "its coins are not the ones the stream answered" else none
          | .error _ => pure none
        else match parseRep t s b with
          | .ok q => ownWhy own i r t (q.differs lay)
          | .error _ => pure none
      if let some why := why then return (← refuse why)
      kept := kept.filter fun k => !(k.i == i && k.r == r)
      let run : V (LiveM (Option Kept)) := if line.startsWith "open " then
          (parseOpen b).map fun (xs, coins) => do openLive ch t r dom xs s coins; return none
        else (parseRep t s b).map fun q => do
          let z ← innerLiveK ch t r dom q
          let coins ← getThe (Array (Array F128))
          return z.map fun proof => ({ i, r, q, coins, proof } : Kept)
      match run with
      | .error e => return (← refuse e)
      | .ok m =>
        match ← (m.run' #[]).run with
        | .ok k =>
          kept := kept ++ k.toArray
          answer ByteArray.empty
        | .error e => return (← refuse e)
    | _ => return (← refuse s!"a request `{line}`")
  return 0

/-- **`flock-firewall finish --circuit F --public F [--statement ID] [--session-tables J] [--tables DIR] [--partition F]
[--program F] [--draw F]`**: the firewall's proof files, `serve`d on stdin and stdout, on the statement `flock-verify verify
--zk` sets up (`Witness.setupOf`, at the `unit_draw` of the JSON object in `--draw`), every salt drawn from the operating
system. With no session loop beside it, it holds no level 0 of its own: a table comes as `table I LEN`. -/
def cmd (args : List String) : IO UInt32 := do
  let ch ← Frame.stdio
  let refuse := fun (why : String) => do
    ch.write (s!"err {why.replace "\n" " "}\n").toUTF8
    ch.flush
    return (1 : UInt32)
  let (some cf, some pf) := (Witness.argOf args "--circuit", Witness.argOf args "--public")
    | refuse "usage: flock-firewall finish --circuit F --public F [--statement ID] [--session-tables J] [--tables DIR] [--partition F] [--program F] [--draw F]"
  let some tags := Tags.find? ((Witness.argOf args "--statement").getD Tags.circuit.name)
    | refuse s!"unknown statement; known: {Tags.all.map (·.name)}"
  let some count := ((Witness.argOf args "--session-tables").getD "1").toNat? | refuse "--session-tables J: J is a number"
  let (partition, program) ← match ← Witness.bindingsOf args with
    | .ok b => pure b
    | .error e => return (← refuse e)
  let draw ← match Witness.argOf args "--draw" with
    | some f => IO.FS.readFile f
    | none => pure "{}"
  let I : Witness.Inputs := { tags, circuitFile := ← IO.FS.readBinFile cf, publicFile := ← IO.FS.readBinFile pf, count,
                              tables := ← Witness.readTables args, partition, program }
  let (spec, sts) ← match Witness.setupOf I draw with
    | .ok s => pure s
    | .error e => return (← refuse s!"setup: {e}")
  serve ch IO.getRandomBytes spec (sts.map Setup.ofCircuit) fun _ => pure none

/-- **The session loop's finish**, a connection to its frames port whose first line is `line`: `finish LEN` followed by
the `LEN` bytes of the session's draw (`Hello.drawOf`'s JSON, `null` for none) sets the session's statement up as the
loop's `Hello` does (`Hello.Setup.setupAt`), is answered `ok 0`, and then `serve`s, every salt drawn from the operating
system and each `level0 I LEN` read from the loop's own level 0 of that root (`Level0.Registry`, `l0s`), each rep's salts
for levels ≥ 1 handed to `keep`, the loop's frames' copy, and each rep's proof file to `file`, the proof the loop sends
(`Session.relay`), each rep's requests held to what the loop holds of it (`own`, `Frames.ownOf`). -/
def serveLoop (ch : Frame.Chan) (S : Hello.Setup) (specs : Hello.Specs) (l0s : IO.Ref Level0.Registry)
    (keep : ByteArray → Nat → Array ByteArray → IO Unit) (file : ByteArray → Nat → ByteArray → IO (Option String))
    (own : Nat → Nat → ByteArray → IO (Option (Except String Own))) (line : String) : IO Unit := do
  let say := fun (s : String) => do
    ch.write (s ++ "\n").toUTF8
    ch.flush
  match (line.replace "\n" "").splitOn " " with
  | ["finish", len] =>
    let some len := len.toNat? | say s!"err a request `{line.trim}`"
    let text := (String.fromUTF8? (← ch.readExact len)).getD "?"
    match Hello.drawOf text with
    | .error e => say s!"err {e.replace "\n" " "}"
    | .ok draw =>
      match ← S.setupAt specs draw with
      | .error e => say s!"err setup: {e.replace "\n" " "}"
      | .ok (spec, sts) =>
        say "ok 0"
        let _ ← serve ch IO.getRandomBytes spec (sts.map Setup.ofCircuit)
          (fun r => do return ((← l0s.get).slots.find? (·.l0.root == r)).map (·.l0)) keep file own true
  | _ => say s!"err a request `{line.trim}`"

end Flock.Firewall.Finish
