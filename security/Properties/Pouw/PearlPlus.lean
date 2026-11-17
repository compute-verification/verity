module

public import PouwPearlPlus
public import Definitions.Pouw.PearlC.Unscale
public import Properties.Pouw.Commit

/-! The properties of Pearl+'s executable run verdict (`PouwCheck.PearlPlus.check`), each proved in
`Proofs.Properties.Pouw.PearlPlus`; none states what the window record's tie establishes. -/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-- The fixed operand's noise the replay uses for tile `tid` of pass `p`: its fixed operand's, once per pass and fixed operand. -/
def noiseOf (p : Pass) (tid : Nat) : Reject FixedNoise :=
  match p.tile? tid with
  | some (pos, _, _) =>
    let u := p.records[pos]!.matmul
    fixedNoise p.salt ((p.weightRoots.lookup u.weight).getD ByteArray.empty) u.k
  | none => throw "tile does not match the verifier's own enumeration"

/-- **What `plan` establishes**: the run's passes are `load`'s of its manifests under the epoch's coins, all of one salt
and, unless the input is a test instance's, each at the served instance, `k_min` `K_MIN` and `TILE × TILE` tiles; its commitment is `runCommitment` of them, its source is the coin record's where the verifier holds one (whose commitment is this run's)
and else the source given, and its tiles are the sample's, keyed by the source and the commitment, then each pass's
exclusion tiles. -/
def PlanMeans : Prop :=
  ∀ (i : Input) (pl : Plan), plan i = .ok pl →
    passesDistinct i.manifests = none ∧ i.manifests.mapM (load i.coins) = .ok pl.passes ∧
    (∀ p ∈ pl.passes, p.salt = pl.passes[0]!.salt) ∧
    (i.testInstance = false → ∀ p ∈ pl.passes, p.scheme.kMin = K_MIN ∧ p.scheme.tileM = TILE ∧ p.scheme.tileN = TILE) ∧
    pl.root = runCommitment pl.passes ∧
    (∀ h, i.held = some h → h.commitment = pl.root ∧ pl.source = h.source ∧ (i.source = none ∨ i.source = some h.source)) ∧
    (i.held = none → i.source = some pl.source) ∧
    ∃ key drawn, PouwCheck.PearlPlus.drawKey pl.source pl.root = .ok key ∧ PouwCheck.PearlPlus.draw pl.passes key i.draws = .ok drawn ∧
      pl.todo = todo pl.passes drawn ∧ pl.drawn = drawn.length ∧ pl.picks = drawn

/-- **What `check` accepts**: a plan, every one of whose tiles the prover opened and `checkTile` passed, on the fixed
operand's noise, or found over its inert-work bound with every other check passed where the run's
`over_cap` is `uncredited`; the verdicts are those tiles', in the plan's order, each ok under that rule. -/
def AcceptMeans : Prop :=
  ∀ (i : Input) (opener : Nat → Nat → Except String TileOpening), (check i opener).accept = true →
    ∃ pl, plan i = .ok pl ∧ (check i opener).plan = some pl ∧ (check i opener).overInertBound = i.overInertBound ∧
      (check i opener).verdicts.map (fun v => (v.pass, v.tile)) = pl.todo ∧
      (∀ v ∈ (check i opener).verdicts, v.ok i.overInertBound = true) ∧
      ∀ pi tid, (pi, tid) ∈ pl.todo → ∃ p o, pl.passes[pi]? = some p ∧ opener pi tid = .ok o ∧
        (checkTile p (noiseOf p tid) tid o = .pass ∨
          i.overInertBound = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overInertBound why)

/-- **A run the verdict refuses before any tile is opened carries no commitment**: a refusal of `plan` (`after-coin`, a
pass that doesn't load, passes that share a call) answers with no plan and no verdicts, under the run's `over_cap` rule and
with the tie where the refusal came after it. -/
def RefusedBeforeTheCoin : Prop :=
  ∀ (i : Input) (opener : Nat → Nat → Except String TileOpening) (r : Refusal), plan i = .error r →
    check i opener = { accept := false, reason := some r.reason, refused := r.refused, overInertBound := i.overInertBound, tie := r.tie }

/-- **What a loaded pass's records are**: each keeps `UnitRecord`'s own rule, together they keep `check_records` (every
unit at its fixed operand's split, one set of excluded and of voluntary B rows per fixed operand, V-EX's count caps ⌈m/64⌉ of A and
⌈n/64⌉ of B), and they are the ones whose root the pass published, under its salt and fixed operands' root. -/
def LoadedGivesUp : Prop :=
  ∀ (coins : CoinCommitment) (man : Lean.Json) (p : Pass), load coins man = .ok p →
    (∀ r ∈ p.records, r.reason = none ∧ r.voluntaryA.length ≤ voluntaryCap r.matmul.m ∧
      r.voluntaryB.length ≤ voluntaryCap r.matmul.n) ∧
    ∃ splits, checkRecords splits p.records = none ∧ recordsRoot p.salt p.weightsRoot splits p.records = p.recordsRoot

/-- The tile's operands and checked values, as `checkTile` forms them from its openings. -/
def formed (p : Pass) (wn : Reject FixedNoise) (rows cols : List Nat) (u : Matmul) (o : TileOpening) :
    Reject (Ops × Checked) := do
  let ops ← encode p.salt ((p.weightRoots.lookup u.weight).getD ByteArray.empty) (← wn) u.index u.k rows
    (o.a.map (openedValue ·.value)) cols (o.b.map (openedValue ·.value))
  pure (ops, ← checked ops)

/-- **What a tile that passes, or is only over its inert-work bound, establishes**: the tile is the verifier's own
(`tile?`); its A and B rows and its leaf open, at the tile's positions, against the call's A root, the fixed operand's root and
the pass's transcript root (hm96 under SHA-512); its words formed from those rows under the pass's salt and the fixed
operand's noise are checked and hash to the opened leaf; its rows keep the per-row rules (`Rule.rowsFailure`); its
credited cells' replayed inert work is within the bound where it passes and over it, with that reason, where it is
`overInertBound` (`Rule.inertBoundCheck`); and on a call with y, the tile's real rows open against the y root and each word is the
unscale of the tile's U (`Rule.yReason`). -/
def TileAcceptMeans : Prop :=
  ∀ (p : Pass) (wn : Reject FixedNoise) (tid : Nat) (o : TileOpening) (bound : Option String),
    checkTile p wn tid o = (match bound with | none => .pass | some why => .overInertBound why) →
    ∃ pos rows cols ops ch, p.tile? tid = some (pos, rows, cols) ∧
      let r := p.records[pos]!
      let u := r.matmul
      o.a.map (·.position) = rows ∧ o.b.map (·.position) = cols ∧ o.leaf.position = tid ∧
      o.a.all (opens (rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false) p.rootsA[pos]! ROW_SCHEMA) ∧
      o.b.all (opens (rowsDomain (weightBinding u.weight) (-2) u.n HM96 false)
        ((p.weightRoots.lookup u.weight).getD ByteArray.empty) ROW_SCHEMA) ∧
      opens (transcriptDomain p.salt p.weightsRoot p.count HM96) p.transcriptRoot TILE_SCHEMA o.leaf ∧
      formed p wn rows cols u o = .ok (ops, ch) ∧ leaf ch = openedValue o.leaf.value ∧
      Rule.rowsFailure r rows cols ops = none ∧ Rule.inertBoundCheck p.scheme r rows cols ops = .ok bound ∧
      ∀ yRoot, p.yRoots[pos]! = some yRoot → ∃ ys, o.y = .ok ys ∧
        ys.map (·.position) = rows.filter (· < r.realRows) ∧
        ys.all (opens (rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false) yRoot ROW_SCHEMA) ∧
        Rule.yReason rows cols ops ch (ys.map fun y => words32 (openedValue y.value)) = none

/-- **The inert-work bound a passed tile keeps** (`Rule.checkOpened`'s `inertBoundFailure`): on its credited rows `cr` and
columns `cc`, the replayed inert work, 40 per flagged atom-word, 32 per later +0 promotion and fs·|cc|/n per forming
element of a credited row (`Rule.rowElements`: salt-dead, or in an atom block every word of its row flags;
`tileInertWorkDev`'s elems) and one per product of a word that no salt makes count (`Rule.wordMend`: a dead pair or a
forced zero, from both rows' ends, `Rule.rowEnds`; rev1's `wordMend`), is at most the inert-work share ρ = 1/400 of
`creditOf(1, |K|, n)/n` per credited cell, as a tile and on each credited A row (four times that at a census width).
`Rule.inertWorkUnits` is n times the chain's and the forming's inert work, so with M the mends the tile's test is
400·(n·(D + M) + fs·|cc|·E) ≤ creditOf(1, |K|, n)·|cr|·|cc|. -/
def TileKeepsTheInertBound : Prop :=
  ∀ (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops), Rule.checkOpened s r rows cols ops = none →
    let (cr, cc) := r.credited rows cols
    cr ≠ [] → cc ≠ [] →
    ∃ (words : List (List (Array Bool × Nat))) (elems : List Nat)
      (endsA endsB : List (Array Nat × Array Nat × Array Bool)),
      cr.mapM (fun i => cc.mapM fun j => Rule.wordFlags ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]!) = .ok words ∧
      (cr.zip words).mapM (fun (iw : Nat × List (Array Bool × Nat)) =>
        Rule.rowElements ops.a.alpha[rows.idxOf iw.1]! ops.a.beta[rows.idxOf iw.1]! ops.a.x[rows.idxOf iw.1]!
          (iw.2.map Prod.fst)) = .ok elems ∧
      cr.mapM (fun i => Rule.rowEnds ops.a.alpha[rows.idxOf i]! ops.a.beta[rows.idxOf i]! ops.a.x[rows.idxOf i]!) =
        .ok endsA ∧
      cc.mapM (fun j => Rule.rowEnds ops.b.alpha[cols.idxOf j]! ops.b.beta[cols.idxOf j]! ops.b.x[cols.idxOf j]!) =
        .ok endsB ∧
      let credit := creditOf 1 r.kKept r.matmul.n
      let rowInertWork := ((words.zip elems).zip endsA).map
        fun (we : (List (Array Bool × Nat) × Nat) × (Array Nat × Array Nat × Array Bool)) =>
          Rule.inertWorkUnits r.matmul.n cc.length (we.1.1.foldl (fun t w => t + (w.1.filter id).size) 0)
            (we.1.1.foldl (fun t w => t + w.2) 0) we.1.2 + r.matmul.n * endsB.foldl (fun t eb => t + Rule.wordMend we.2 eb) 0
      rowInertWork.foldl (· + ·) 0 * Rule.INERT_BOUND_DEN ≤ Rule.INERT_BOUND_NUM * credit * cr.length * cc.length ∧
      ∀ d ∈ rowInertWork, d * Rule.INERT_BOUND_DEN ≤
        Rule.INERT_BOUND_NUM * credit * cc.length * (if s.censusKs.contains r.matmul.k then 4 else 1)

/-- **The unscale the check holds y to is the served y's Definition**: wherever `Rule.yOf` gives a word (each step finite),
it is `Pouw.PearlC.unscale` of the same U and α (`PearlCUnscale_v1` at `BIAS` off), and the check's copy is that
Definition everywhere. -/
def YIsTheUnscale : Prop :=
  (∀ u alphaA alphaB, Rule.unscale u alphaA alphaB = Pouw.PearlC.unscale u alphaA alphaB) ∧
  ∀ u alphaA alphaB y, Rule.yOf u alphaA alphaB = some y → y = Pouw.PearlC.unscale u alphaA alphaB

/-- **The work an accepted run publishes is its credit at Pearl's assumption**: over the plan's passes, at the run's samples,
X_O (how many of them landed on a tile over its inert-work bound, `overInertBoundDraws`) and δ = 1/100 (`creditAtAssumption`),
crediting nothing on a test instance's input (`testCredit`); a refused or rejected run publishes none. -/
def WorkIsTheCredit : Prop :=
  (∀ (i : Input) (opener : Nat → Nat → Except String TileOpening),
    (check i opener).accept = true → ∃ pl, plan i = .ok pl ∧
      (check i opener).work =
        some (testCredit i.testInstance (creditAtAssumption pl.passes i.draws (overInertBoundDraws pl (check i opener).verdicts) DELTA))) ∧
  (∀ (i : Input) (opener : Nat → Nat → Except String TileOpening),
    (check i opener).accept = false → (check i opener).work = none)

/-- The messages `sha512/row-seg/v1` hashes for the row `row`: each 1024-byte segment's under its first block, then the outer
one over their digests. -/
def segHashed (row : ByteArray) : List ByteArray :=
  let l := 8 * row.size
  let segs := (Flock.HmRow.segShape l).1
  let inner := (List.range segs.size).map fun j =>
    Flock.HmRow.segmentPrefix j segs.size segs[j]!.1 ++ row.extract (1024 * j) (1024 * j + (segs[j]!.1 + 7) / 8)
  inner ++ [inner.foldl (fun acc m => acc ++ Flock.Sha512.hash m) (Flock.HmRow.rowPrefixSeg l)]

/-- What SHA-512 hashes when `RowsDomain.verify` checks the opening `o` in the hm96 tree `d`, from the value up: the
preimage's row digest's messages (none in a tile tree, whose preimage is the digest), the hm96 leaf's salt-digest and leaf
inputs, the tree leaf's frame, and the frame of each node on its path. -/
def rowsHashed (d : RowsDomain) (o : LeafOpening) : List ByteArray :=
  let pre := o.value.extract 0 (o.value.size - HM96_SALT)
  let y := o.value.extract (o.value.size - HM96_SALT) o.value.size
  let x := if d.digests then pre else Flock.HmRow.segDigest (8 * pre.size) pre
  let lf := Flock.Hm96.Default512.leaf x y
  (if d.digests then [] else segHashed pre) ++
    [Flock.Hm96.Default512.saltPrefix ++ y,
     Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString x y,
     frameBytes "leaf" [d.frame.id, Flock.FV3.uint o.position, Flock.FV3.uint o.position, Flock.HmRow.ROW_SCHEMA_SEG.toUTF8,
       lf]] ++
    Properties.Pouw.Commit.pathFrames d.frame o.position (d.frame.leaf o.position Flock.HmRow.ROW_SCHEMA_SEG lf) o.path

/-- **An opening binds its position's value, or SHA-512 collides on the run's own messages**: two openings of one position
of one hm96 tree, both accepted against its root, open one value, or two different messages, one hashed in checking each
opening, have one SHA-512 digest. -/
def OpeningBinds : Prop :=
  ∀ binding owner count digests (d : RowsDomain) (root : ByteArray) (schema : String) (o o' : LeafOpening),
    rowsDomain binding owner count HM96 digests = .ok d → o.position = o'.position →
    opens (.ok d) root schema o = true → opens (.ok d) root schema o' = true →
      o.value = o'.value ∨
        ∃ x ∈ rowsHashed d o, ∃ y ∈ rowsHashed d o', x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

end Properties.Pouw.PearlPlus
