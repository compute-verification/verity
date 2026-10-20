/-
# The firewall's final round (live PROTOCOL.md §10.1): a firewalled rep's proof file, computed by the firewall itself

At a firewalled rep's end the proof file that leaves is the firewall's own (`flock-circuit`'s `Firewall::finish` and
`zk_finish`): the zerocheck and lincheck sent as the values that passed the firewall, masked by the rep's pads, `final_c`
the one those values fix; the ring switch, the masked Ligerito opening and the levels above it computed from the host's
witness, the table's level-0 hiding and the rep's salts; and the inner proof on the rep's pads. Here that file is code the
firewall runs, the analysis's masked prover (`Flock.Firewall.Finish`'s copies, `FinishProver`) on what the firewall holds:

* `rowsFft`, `commitFw`: a level ≥ 1's commitment of an F-table, its codeword by the additive FFT (`Link.fft`) on each lane
  (`FlockComplete.commitP`'s, which evaluates each entry on its own), its leaves `hm96-sha512/v1`'s in tasks.
* `commit0`: level 0's commitment of the table the transport hands it (`Link.In`, the same table `flock-firewall link`
  commits; `FlockComplete.commitPZk`'s).
* `zerocheckFw`, `lincheckFw`: the masked zerocheck and lincheck sending the passed values `v_i` as `v_i + h_i`
  (`FlockComplete.zerocheckCut`, `lincheckCut` at `masked i = v_i + h_i`), `final_c`'s too: the inner proof holds the
  passed `final_c` to `P_c(ζ)` of the passed `round1_c` (`Zk.replay`'s constraint).
* `switchVec`: claim `k`'s switched vector from the packed table's bits (`FlockComplete.switchVec` on the trace).
* `repFw`: the rep (`FlockComplete.repPZk`'s order: the PIOP, the ring switch on `abcClaims ++ extras`, the masked
  Ligerito, the verifier's own replay, the ring-switch and lane-phase constraints, the inner proof), its proof file
  `encZk` at `shapeZk`. The rank check is the Lean frame's (`Frame.maskRank`), which stops the rep before this runs.
* `asksOf`: a rep's rounds answered by the coins the stream answered them, round by round.

`flock-firewall finish` serves it (`cmd`) on the statement of `flock-verify verify --zk` (`Witness.setupOf`).
-/
import Flock.Firewall.FinishProver
import Flock.Firewall.Frame
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
`flock-firewall link` commits (`Link.rows`), under the table's level-0 salts. -/
def commit0 (lv : Level) (tPad : Nat) (i : Link.In) : Commit :=
  commitRows lv (Link.rows lv tPad i.g i.extra i.lanePad) i.salt

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

/-- **The firewall's proof of one rep** of the packed table `g` committed as `cm0`, as rep `rep` of a stream whose
transcript starts at `dom`: the passed values `vals` (by pad index) masked by the rep's pads `pads` committed as `ic`, the
statement's extra claims `extras`, the table's extra lanes and paddings, levels ≥ 1 committed by `commit` under `salt`,
every grinding nonce `0`. -/
def repFw (io : Asks M) (hs : Hashes M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (st : Setup)
    (sch : Flock.Schedule) (tPad rep : Nat) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (vals : Array F128) (extra lanePad : Nat → Nat → F128)
    (salt : Nat → Nat → ByteArray) : TMW M Zk.ZkProof := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let mk : Nat → F128 := fun i => vals[i]! + pads.h[i]!
  let (zp, zc) ← zerocheckFw io st.m st.digest (flat cm0.cap) ic.root lay mk
  let (lp, lc) ← lincheckFw io st.fold lay mk zc
  let claims := abcClaims st.m st.kLog zc lc ++ extras
  let (ss, T, sws) ← ringSwitchP io (fun k => switchVec g claims[k]!) claims
  let (lg, zl, rec) ← ligeritoPZk io commit st.merkle sch T cm0 g (basisVals sws (st.m - 7)) lay pads.h rep tPad extra
    lanePad salt (fun _ _ => 0)
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  let inner ← innerP io hs lay pads ic zp.finalC
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc)
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
the firewall by pad index, the extra claims' coordinates, levels ≥ 1's salts by level (`192` bytes a leaf), and each
round's coins. -/
structure RepIn where
  draws : Frame.Draws
  vals : Array F128
  xs : Array (Array F128)
  hiSalts : Array ByteArray
  coins : Array (Array F128)

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
extra claims (a count, then each as a count and its words), levels ≥ 1's salts (`192·2^d` bytes each, levels `1` to `R`),
and the coins (a round count, then each round as a count and its words); nothing after. -/
def parseRep (t : Tab) (b : ByteArray) : V RepIn := do
  let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
  let R := t.sch.levels.size - 1
  let rd : Rd RepIn := do
    let draws := Frame.Draws.ofBytes lay (← take (Frame.drawBytes lay) "the draws")
    let vals ← wordsOf (lay.zp 64) "the passed values"
    let nx ← u64 "the extra claims"
    let mut xs := #[]
    for _ in [0:nx] do xs := xs.push (← vecOf "an extra claim")
    let mut hiSalts := #[]
    for l in [1:R + 1] do hiSalts := hiSalts.push (← take (192 * 2 ^ (t.sch.levels.getD l default).d) s!"level {l}'s salts")
    let nk ← u64 "the coins"
    let mut coins := #[]
    for _ in [0:nk] do coins := coins.push (← vecOf "a round's coins")
    if (← get) != b.size then throw s!"finish: {b.size - (← get)} bytes after the rep's request"
    return { draws, vals, xs, hiSalts, coins }
  (rd.run' 0).run b

/-- **A table's state** from `flock-firewall link`'s request bytes (`Link.parse`) at the statement `st`. -/
def tabOf (st : Setup) (b : ByteArray) : V Tab := do
  if st.merkle.id != MerkleScheme.hm96Sha512.id then throw s!"finish: a {st.merkle.id} statement: --zk commits hm96-sha512/v1"
  let (sch, tPad) ← Zk.schedule st.m
  let i ← Link.parse st.m b
  return { st, sch, tPad, g := i.g, extra := i.extra, lanePad := i.lanePad
           cm0 := commit0 (sch.levels.getD 0 default) tPad i }

/-! ## `flock-firewall finish` -/

/-- **`flock-firewall finish --circuit F --public F [--statement ID] [--session-tables J] [--tables DIR] [--partition F]
[--program F] [--draw F]`**: the firewall's proof files, served on stdin and stdout, on the statement `flock-verify verify
--zk` sets up (`Witness.setupOf`, at the `unit_draw` of the JSON object in `--draw`). `table I LEN` followed by `LEN` bytes
of table `I`'s level-0 data (`Link.parse`'s) keeps the table and is answered `ok N` followed by the `N` bytes of its
level-0 root. `rep I R LEN` followed by the `LEN` bytes of rep `R`'s request (`parseRep`) on table `I` is answered
`ok N` followed by the `N` bytes of the rep's proof file (`proofFile`). Anything else, or a request it refuses, is
answered `err WHY` and ends it. -/
def cmd (args : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let refuse := fun (why : String) => do
    stdout.putStrLn s!"err {why.replace "\n" " "}"
    stdout.flush
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
  let setups := sts.map Setup.ofCircuit
  let mut tabs : Array (Option Tab) := Array.replicate setups.size none
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["table", i, len] =>
      let (some i, some len) := (i.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let some st := setups[i]? | return (← refuse s!"table {i} of the session's {setups.size}")
      let b ← Coins.readExact "finish" stdin len
      match tabOf st b with
      | .ok t =>
        tabs := tabs.set! i (some t)
        let r := flat t.cm0.cap
        stdout.putStrLn s!"ok {r.size}"
        stdout.write r
        stdout.flush
      | .error e => return (← refuse e)
    | ["rep", i, r, len] =>
      let (some i, some r, some len) := (i.toNat?, r.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let b ← Coins.readExact "finish" stdin len
      let some (some t) := tabs[i]? | return (← refuse s!"a rep of table {i} before its table")
      let some name := spec.tables[i]? | return (← refuse s!"table {i} of the session's {spec.tables.size}")
      if r > 1 then return (← refuse s!"rep {r}: a table has reps 0 and 1")
      match parseRep t b >>= proofFile t r (spec.domain name r).toUTF8 with
      | .ok p =>
        stdout.putStrLn s!"ok {p.size}"
        stdout.write p
        stdout.flush
      | .error e => return (← refuse e)
    | _ => return (← refuse s!"a request `{line}`")
  return 0

end Flock.Firewall.Finish
