import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg

/-!
# `verify --zk --public-inputs F`, its statement and its public gates

A statement with hidden outputs may name public input ports (PROTOCOL §16.10, `HmIn`): unit inputs that are no row but
public words the verifier recomputes, which the units read from `in` regions over whole words of every unit slot's leaf
group, with no Δ copy. `flock-verify verify --zk --public-inputs F` passes its own recomputation `F` to every table's
`Stmt.setupHidden` (`Stmt.setupTables … (some own)`), which takes the public file's words only when they equal `F`'s
(`HmIn.checkOwn`). Without `F` a statement with public input ports is refused, so `ZkExec.AcceptsZK` (at
`publicInputs := none`) is empty on such circuits; these are its events with `F`:

* `setupOfP`, `AcceptsZKP`, `LiveAcceptsZKP`: `Exec.setupOf`, `AcceptsZK` and `LiveAcceptsZK` with `own`;
* `AcceptsZKRP`, `LiveAcceptsZKRP`: the same under `--registered`, as `ZkReg.AcceptsZKR`.

What `Stmt.setupHidden` checks of them (`SpecP`, `setupHidden_specP` in `Spec.lean`): `Hidden.Spec`'s checks, with
`HmIn`'s public file, Δ (no copy into a public leaf) and regions (`HmOut.regions ++ HmIn.regions`), the public words
the verifier's own (`OwnWords`, from `checkOwn`), and the public leaf maps' layout (`LeavesPublic`).

The units' program with its public gates (`keyProgP`): `Placed.keyProg`'s gates, every input on the gate of its Δ key
(`Aliased.keyIdx`) except a public leaf's bit, which is a gate of its own (`keyIdxP`). Under `keyProg` a public leaf,
a zero leaf of `leaves_in`, reads the shared zero gate, which is not what the unit reads. `pubBit` is a public gate's
value, the bit of the verifier's own word (`ownWord`), and `pubOnes`, `pubZeros` the public gates at 1 and at 0.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Lean (Json)
open Flock
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.ZkExec (zkInputs)
open FlockSoundness.Discharge.Layout (keyB keyN)

/-! ## The events -/

/-- **The session's statements with the verifier's own public inputs**: `Exec.setupOf` with `own` as
`Stmt.setupTables`'s `publicInputs` (`--public-inputs F`, `Main.buildSession`). -/
def setupOfP (I : Inputs) (own : ByteArray) (record : String) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTables I.tags I.circuitFile I.publicFile I.coins I.count I.tables (recordDraw record) I.partition I.program
    I.work I.stratified I.closure (some own)

/-- **flock-verify `--zk --public-inputs F` accepts the transcript**, `own` F's bytes: the session's statements set up
from `zkInputs I`, the record's draw and `own`, every table passes `Zk.shapeOk`, and `Flock.Zk.verify` accepts the
record and the proof files. -/
def AcceptsZKP (I : Inputs) (own : ByteArray) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOfP (zkInputs I) own t.record = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
    Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

/-- **flock-verify `--zk --registered R --public-inputs F` accepts the transcript**: `AcceptsZKP` with the session
passing the registered check against `reg` (`Registered.checkSession`, as `Main.buildSession` runs it), as
`ZkReg.AcceptsZKR`. -/
def AcceptsZKRP (I : Inputs) (reg : Array Registered.Port) (own : ByteArray) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOfP (zkInputs I) own t.record = .ok (spec, sts) ∧
    Registered.checkSession sts I.publicFile (recordDraw t.record) I.count (some reg) = .ok () ∧
    sts.forM Flock.Zk.shapeOk = .ok () ∧ Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- flock-verify `--zk --public-inputs F` accepts the live audit's transcript. -/
def LiveAcceptsZKP (I : Inputs) (own : ByteArray) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKP I own o.2.2

/-- flock-verify `--zk --registered R --public-inputs F` accepts the live audit's transcript. -/
def LiveAcceptsZKRP (I : Inputs) (reg : Array Registered.Port) (own : ByteArray)
    (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKRP I reg own o.2.2

/-! ## The verifier's own words and the public leaf maps -/

/-- **Word `x` of the verifier's own recomputation**: u16 LE at bytes `2x`, `2x + 1` of the body after `F`'s header
line (instance-major, the ports in order, as `HmIn.checkOwn` reads it); 0 without a header line. -/
def ownWord (own : ByteArray) (x : ℕ) : ℕ :=
  match findByte own 10 with
  | some nl =>
    let body := own.extract (nl + 1) own.size
    (body.get! (2 * x)).toNat + 256 * (body.get! (2 * x + 1)).toNat
  | none => 0

/-- **The public file's words are the verifier's own**: every word of the population's public words (`checkOwn`'s
`words`, instance-major, `pi.words` per instance) is `ownWord own` at its index. -/
def OwnWords (own : ByteArray) (pi : HmIn.Ports) (pop : Public) : Prop :=
  ∀ x < (pop.population.map (·.1)).getD pop.n * pi.words,
    ((pop.population.map (·.2.2)).getD pop.outputs).getD x 0 = ownWord own x

/-- **The public leaf maps' layout** (`HmIn.parse`'s checks): one map per unit copy, as long as its `leaves_in`, each
leaf `-1` or a public word, and a public leaf a zero leaf of `leaves_in`; the unit range unpacked and its slot bits
beside an in region's within `PT_LOCAL`; some leaf public; the in regions' blocks inside the leaf group; and no row
leaf inside a word that holds a public leaf. -/
structure LeavesPublic (c : Circuit) (pi : HmIn.Ports) : Prop where
  size : pi.leaves.size = c.upv
  maps : ∀ u < c.upv, (pi.leaves.getD u #[]).size = (c.leavesIn.getD u #[]).size ∧
    ∀ j < (pi.leaves.getD u #[]).size, -1 ≤ (pi.leaves.getD u #[]).getD j (-1) ∧
      (pi.leaves.getD u #[]).getD j (-1) < pi.words ∧
      (0 ≤ (pi.leaves.getD u #[]).getD j (-1) → (c.leavesIn.getD u #[]).getD j (-1) = -1)
  unpacked : c.unitRange.packed = false
  slots : 7 + log2ceil c.unitRange.count ≤ PT_LOCAL
  some_word : (HmIn.words c pi).isEmpty = false
  blocks : ∀ b ∈ (HmIn.blocks (HmIn.words c pi)).toList, (c.unit.inGroups.getD 0 default).col ≤ b.1 ∧
    b.1 + 2 ^ b.2 ≤ (c.unit.inGroups.getD 0 default).col + (c.unit.inGroups.getD 0 default).words
  noRow : ∀ u < c.upv, ∀ j < (c.leavesIn.getD u #[]).size,
    (HmIn.words c pi).binSearchContains ((c.unit.inGroups.getD 0 default).col + j / 8) (· < ·) = true →
    (pi.leaves.getD u #[]).getD j (-1) < 0 → (c.leavesIn.getD u #[]).getD j (-1) < 0

/-! ## The units' program with its public gates -/

/-- **The public leaf bit a VU-stacked input reads**: input `x` is bit `t` of leaf `p` of unit copy `x / W`'s leaf
group (`W = inWords · WORD`, the leaf and bit `Aliased.srcKey` reads), and that leaf is public word `q` of its
instance (`pi.leaves`): `some (q, t)`; `none` for any other input. -/
def pubLeaf (c : Circuit) (pi : HmIn.Ports) (x : ℕ) : Option (ℕ × ℕ) :=
  let W := c.unit.inWords * WORD
  let pc := Flock.portCols c.unit.inGroups
  let p := FlockSoundness.portOf pc (x % W)
  if x % W ∈ FlockSoundness.portBits c.unit.inGroups ∧ p < (c.unit.inGroups.getD 0 default).bits.size then
    let q := (pi.leaves.getD (x / W) #[]).getD p (-1)
    if 0 ≤ q then some (q.toNat, x % W - pc[p]!.1) else none
  else none

/-- **An input's gate in `keyProgP`**: a public leaf's bit on its own gate; any other input on the gate of the first
input with its Δ key that is no public leaf's bit (`Aliased.keyIdx` among the others), the zero's gate at the zero. -/
def keyIdxP (c : Circuit) (pi : HmIn.Ports) (m : ℕ) (x : Fin m) : Option (Fin m) :=
  if (pubLeaf c pi x.val).isSome then some x
  else if Aliased.srcKey c x.val = .zero then none
  else (List.finRange m).find? fun i =>
    (pubLeaf c pi i.val).isNone && decide (Aliased.srcKey c i.val = Aliased.srcKey c x.val)

/-- **The circuit's program with its public gates**: `Placed.keyProg`'s units and gates, every input on `keyIdxP`'s
gate. -/
noncomputable abbrev keyProgP (c : Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports) :
    FlockSoundness.Prog (keyN c hU n) n :=
  Placed.circuitProg c hU n (keyB c hU) (keyIdxP c pi (keyB c hU))

/-- **A public gate's value**: input gate `2 + u·B + x` of `keyProgP`, at a public leaf's bit `(q, t)` of input `x`
(`pubLeaf`), is bit `t` of the verifier's own word `q` of instance `u` (`ownWord own (u · pi.words + q)`); `none` at
every other gate. -/
noncomputable def pubBit (c : Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports) (own : ByteArray)
    (g : Fin (keyN c hU n)) : Option Bool :=
  if 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU then
    match pubLeaf c pi ((g.val - 2) % keyB c hU) with
    | some (q, t) => some (Nat.testBit (ownWord own ((g.val - 2) / keyB c hU * pi.words + q)) t)
    | none => none
  else none

/-- The public gates the verifier's own words set to 1. -/
noncomputable def pubOnes (c : Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports) (own : ByteArray) :
    Finset (Fin (keyN c hU n)) :=
  @Finset.filter _ (fun g => pubBit c hU n pi own g = some true) (Classical.decPred _) Finset.univ

/-- The public gates the verifier's own words set to 0. -/
noncomputable def pubZeros (c : Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports) (own : ByteArray) :
    Finset (Fin (keyN c hU n)) :=
  @Finset.filter _ (fun g => pubBit c hU n pi own g = some false) (Classical.decPred _) Finset.univ

/-! ## What `Stmt.setupHidden` checked, with the verifier's public inputs -/

/-- **What `Stmt.setupHidden … (some own)` accepted on a circuit with public input ports `pi`**: `Hidden.Spec`'s
checks, with the public file `HmIn.loadPublic` loaded (at a draw, `HmIn.drawn` of it), whose words `HmIn.checkOwn`
found the verifier's own (`OwnWords`); Δ from `HmIn.delta` (`HmOut.delta`'s without a public leaf's entries); the
regions `HmOut.regions ++ HmIn.regions` (the public leaves' `in` regions); and the public leaf maps' layout
(`LeavesPublic`). -/
structure SpecP (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Json) (own : ByteArray) (pi : HmIn.Ports) (st : Stmt) : Prop where
  tagsEq : st.tags = tags
  leaf : st.leaf = default
  parse : HmOut.parse tags cf tables = .ok st.c
  inParse : HmIn.parse st.c = .ok pi
  ports : pi.ports.isEmpty = false
  leavesPublic : LeavesPublic st.c pi
  merkle : MerkleScheme.find? st.c.leafScheme = some st.merkle
  pub : ∃ pop, HmIn.loadPublic tags st.c pi pf = .ok pop ∧ HmIn.checkOwn pi pop (some own) = .ok () ∧
    OwnWords own pi pop ∧ (draw = none → st.pub = pop) ∧ ∀ d, draw = some d → HmIn.drawn st.c pi pop d = .ok st.pub
  checkPublic : HmOut.checkPublic st.c st.pub = .ok ()
  n : st.n = st.pub.n
  blocksReal : st.blocksReal = (st.n + st.c.g - 1) / st.c.g
  nbl : st.nbl = max 3 (log2ceil (max st.blocksReal 1))
  m : st.m = st.c.kLog + st.nbl
  kLog : st.c.kLog ≤ 27
  m35 : st.m ≤ 35
  mPts : st.mPts = PT_LOCAL + st.nbl
  pin : HmRow.pin st.c = .ok st.pin
  da : st.da = (HmIn.delta st.c pi st.pin).1
  db : st.db = (HmIn.delta st.c pi st.pin).2
  regions : ∃ ro ri, HmOut.regions st.c st.pub = .ok ro ∧ HmIn.regions st.c pi st.pub = .ok ri ∧
    st.regions = ro ++ ri
  regionWords : HmRow.checkRegionWords st.c st.da st.db st.regions = .ok ()
  inRange : checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok ()
  sigma : st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha)
  hello : helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello

end FlockSoundness.Discharge.ZkPubIn
