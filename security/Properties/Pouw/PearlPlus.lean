module

public import PouwPearlPlus
public import Definitions.Pouw.PearlC.Unscale
public import Properties.Pouw.Commit

/-!
# Pearl+'s run verdict: what its Lean code (`PouwCheck.PearlPlus.check`) does

The properties of the executable verdict (`verity/protocols/pouw/verifier/lean/pearlplus/`, the port of
`benchmarks/pouw/pearl_c_vllm/verify_run.verify_passes`), stated against `check` and the functions it runs, for every input:
the epoch's coin commitment, the retained passes' manifests, the number of draws, the verifier's coin (a source, or its
record of the coin) and the prover's openings. Each has one shape, the input, the acceptance, then what it establishes.
`Proofs.Properties.Pouw.PearlPlus` proves every one of them, on Lean core, each as its name in lower camel case
(`WorkIsTheCredit` as `workIsTheCredit`).

What they compose into is the remote theorem's useful-work part: on an accepted run, under Pearl's Assumption 1 at the H100
record per drawn tile (`TTOutTilePearlCH100Rev1`), the run's credit is at most its cost up to the draw's ε at δ, or a
SHA-512 break on the run's trees. These properties are the code's side of it:

* `AcceptMeans`: an accepted run is a plan (`PlanMeans`) whose every drawn and exclusion tile passes `checkTile`, or,
  under `over_cap` `uncredited` (CMP-311 option A, the default), is over its inert-work bound with every other check
  passed and is credited nothing.
* `PlanMeans`: the plan's passes are the loaded passes, all of one salt and, but on a test instance's input, of the served
  instance (`k_min` 1,024, 64 × 64 tiles), its commitment theirs, its source the coin record's when there is one (retention and give-up fixed before
  the coin, `after-coin`), and its tiles the draw's over their tickets, keyed by the source and the commitment.
* `LoadedGivesUp`: a loaded pass's records are its published root's, and V-EX's give-up is within its caps.
* `TileAcceptMeans`: a tile that passes, or is only over its inert-work bound, opens against the pass's roots, its
  checked words are the committed leaf's, its rows keep the per-row rules, its credited cells' replayed inert work, P3's
  forming term included, is within the inert-work share ρ = 1/400 of their credit (as a tile and per credited A row)
  where it passes and over it where it doesn't, and its served y rows are the unscale of its U.
* `YIsTheUnscale`: the unscale the check holds y to is `Pouw.PearlC.unscale`, the served y's Definition
  (`PearlCUnscale_v1` at `BIAS` off).
* `WorkIsTheCredit`: the work an accepted run publishes is its credit at Pearl's assumption (`creditAtAssumption`) over
  the plan's passes, at the run's draws, the draws over the inert-work bound (X_O) and δ = 1/100; a test instance's
  credits nothing.
* `OpeningBinds`: two openings of one position of a tree both accepted open one value, or two different messages hashed
  in checking them (`rowsHashed`) collide under SHA-512.

The window record's tie is ported (`tieOf`, `PouwPearlPlus.Tie`, which `plan` runs before the draw), but no property
here states what it establishes on an accepted run, that W registers the run's calls and fixed operands: that is open.

**The device record.** The premise is read at the H100's record, `devH100` (`Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1`,
and at W1 `TTOutTilePearlCH100Rev1W1` on `W1Device.h100`), and the run it is applied to served on an H200. Every field of
that record the premise reads is the H200's, so the properties are stated at it and there is no H200 instance:

* the promotion period G = 4 (`d.G`, which is all the tile domain `pearlCDomainDev` reads): the scheme's own, and the H200
  serves the one scheme, `pearl-c-h100-v1-h3`;
* the prices (`Prices.h100`: an FP32 add 32, a BF16 MAC 2, `Costs.adopted`) and W1's (`W1Prices.h100`: an E4M3 or S8 MAC 1,
  a BF16 MAC 2, FP32 32, a conversion 64, INT32 64), per tensor-core clock: the H200 is the same GH100 at sm_90a, the same
  4,096 E4M3 MACs per SM a clock and half that in BF16 (`Core.Accel.H200`), and its data sheet's FP8 and BF16 rates are the
  H100 SXM's;
* the steps (`hopperE4m3K32` at G = 4, the BF16 k16 peel, `PeelExact`, `Skip.unitFlags`, `SaltDead`) and W1's fragments
  (`h100Frags`: wgmma E4M3 m64nNk32 and BF16 m64nNk16, N = 64 to 256, each priced whole): the same wgmma on the same die,
  and the verifier recomputes an H200 run's drawn tiles bit for bit at `HOPPER_E4M3_K32`;
* W1's `sha` and `fetch` prices are not the record's but the cost model's parameters, taken at the H200's measured ones
  (they differ from the H100's in its memory, which W1 does not otherwise price, and the bound only weakens as they grow).
-/

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

/-! ## The run -/

/-- **What `plan` establishes**: the run's passes are `load`'s of its manifests under the epoch's coins, all of one salt
and, unless the input is a test instance's, each at the served instance, `k_min` `K_MIN` and `TILE × TILE` tiles; its commitment is `runCommitment` of them, its source is the coin record's where the verifier holds one (whose commitment is this run's)
and else the source given, and its tiles are the draw's, keyed by the source and the commitment, then each pass's
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
    ∃ pl, plan i = .ok pl ∧ (check i opener).plan = some pl ∧ (check i opener).overCap = i.overCap ∧
      (check i opener).verdicts.map (fun v => (v.pass, v.tile)) = pl.todo ∧
      (∀ v ∈ (check i opener).verdicts, v.ok i.overCap = true) ∧
      ∀ pi tid, (pi, tid) ∈ pl.todo → ∃ p o, pl.passes[pi]? = some p ∧ opener pi tid = .ok o ∧
        (checkTile p (noiseOf p tid) tid o = .pass ∨
          i.overCap = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overCap why)

/-- **A run the verdict refuses before any tile is opened carries no commitment**: a refusal of `plan` (`after-coin`, a
pass that doesn't load, passes that share a call) answers with no plan and no verdicts, under the run's `over_cap` rule and
with the tie where the refusal came after it. -/
def RefusedBeforeTheCoin : Prop :=
  ∀ (i : Input) (opener : Nat → Nat → Except String TileOpening) (r : Refusal), plan i = .error r →
    check i opener = { accept := false, reason := some r.reason, refused := r.refused, overCap := i.overCap, tie := r.tie }

/-! ## A pass's records and give-up (V-EX) -/

/-- **What a loaded pass's records are**: each keeps `UnitRecord`'s own rule, together they keep `check_records` (every
unit at its fixed operand's split, one set of excluded and of voluntary B rows per fixed operand, V-EX's count caps ⌈m/64⌉ of A and
⌈n/64⌉ of B), and they are the ones whose root the pass published, under its salt and fixed operands' root. -/
def LoadedGivesUp : Prop :=
  ∀ (coins : CoinCommitment) (man : Lean.Json) (p : Pass), load coins man = .ok p →
    (∀ r ∈ p.records, r.reason = none ∧ r.voluntaryA.length ≤ voluntaryCap r.matmul.m ∧
      r.voluntaryB.length ≤ voluntaryCap r.matmul.n) ∧
    ∃ splits, checkRecords splits p.records = none ∧ recordsRoot p.salt p.weightsRoot splits p.records = p.recordsRoot

/-! ## A tile -/

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
`overCap` (`Rule.capCheck`); and on a call with y, the tile's real rows open against the y root and each word is the
unscale of the tile's U (`Rule.yReason`). -/
def TileAcceptMeans : Prop :=
  ∀ (p : Pass) (wn : Reject FixedNoise) (tid : Nat) (o : TileOpening) (bound : Option String),
    checkTile p wn tid o = (match bound with | none => .pass | some why => .overCap why) →
    ∃ pos rows cols ops ch, p.tile? tid = some (pos, rows, cols) ∧
      let r := p.records[pos]!
      let u := r.matmul
      o.a.map (·.position) = rows ∧ o.b.map (·.position) = cols ∧ o.leaf.position = tid ∧
      o.a.all (opens (rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false) p.rootsA[pos]! ROW_SCHEMA) ∧
      o.b.all (opens (rowsDomain (weightBinding u.weight) (-2) u.n HM96 false)
        ((p.weightRoots.lookup u.weight).getD ByteArray.empty) ROW_SCHEMA) ∧
      opens (transcriptDomain p.salt p.weightsRoot p.count HM96) p.transcriptRoot TILE_SCHEMA o.leaf ∧
      formed p wn rows cols u o = .ok (ops, ch) ∧ leaf ch = openedValue o.leaf.value ∧
      Rule.rowsFailure r rows cols ops = none ∧ Rule.capCheck p.scheme r rows cols ops = .ok bound ∧
      ∀ yRoot, p.yRoots[pos]! = some yRoot → ∃ ys, o.y = .ok ys ∧
        ys.map (·.position) = rows.filter (· < r.realRows) ∧
        ys.all (opens (rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false) yRoot ROW_SCHEMA) ∧
        Rule.yReason rows cols ops ch (ys.map fun y => words32 (openedValue y.value)) = none

/-- **The inert-work bound a passed tile keeps** (`Rule.checkOpened`'s `capFailure`): on its credited rows `cr` and
columns `cc`, the replayed inert work, 40 per flagged atom-word, 32 per later +0 promotion and fs·|cc|/n per forming
element of a credited row (`Rule.rowElements`: salt-dead, or in an atom block every word of its row flags;
`tileDebitDev`'s elems), is at most the inert-work share ρ = 1/400 of `creditOf(1, |K|, n)/n` per credited cell, as a
tile and on each credited A row (four times that at a census width). `Rule.debitUnits` is n times the inert work, so the
tile's test is 400·(n·D + fs·|cc|·E) ≤ creditOf(1, |K|, n)·|cr|·|cc|. -/
def TileKeepsTheCap : Prop :=
  ∀ (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops), Rule.checkOpened s r rows cols ops = none →
    let (cr, cc) := r.credited rows cols
    cr ≠ [] → cc ≠ [] →
    ∃ (words : List (List (Array Bool × Nat))) (elems : List Nat),
      cr.mapM (fun i => cc.mapM fun j => Rule.wordFlags ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]!) = .ok words ∧
      (cr.zip words).mapM (fun (iw : Nat × List (Array Bool × Nat)) =>
        Rule.rowElements ops.a.alpha[rows.idxOf iw.1]! ops.a.beta[rows.idxOf iw.1]! ops.a.x[rows.idxOf iw.1]!
          (iw.2.map Prod.fst)) = .ok elems ∧
      let credit := creditOf 1 r.kKept r.matmul.n
      let rowDebit := (words.zip elems).map fun (we : List (Array Bool × Nat) × Nat) =>
        Rule.debitUnits r.matmul.n cc.length (we.1.foldl (fun t w => t + (w.1.filter id).size) 0)
          (we.1.foldl (fun t w => t + w.2) 0) we.2
      rowDebit.foldl (· + ·) 0 * Rule.CAP_DEN ≤ Rule.CAP_NUM * credit * cr.length * cc.length ∧
      ∀ d ∈ rowDebit, d * Rule.CAP_DEN ≤
        Rule.CAP_NUM * credit * cc.length * (if s.censusKs.contains r.matmul.k then 4 else 1)

/-! ## The served y and the work -/

/-- **The unscale the check holds y to is the served y's Definition**: wherever `Rule.yOf` gives a word (each step finite),
it is `Pouw.PearlC.unscale` of the same U and α (`PearlCUnscale_v1` at `BIAS` off), and the check's copy is that
Definition everywhere. -/
def YIsTheUnscale : Prop :=
  (∀ u alphaA alphaB, Rule.unscale u alphaA alphaB = Pouw.PearlC.unscale u alphaA alphaB) ∧
  ∀ u alphaA alphaB y, Rule.yOf u alphaA alphaB = some y → y = Pouw.PearlC.unscale u alphaA alphaB

/-- **The work an accepted run publishes is its credit at Pearl's assumption**: over the plan's passes, at the run's draws,
X_O (how many of them landed on a tile over its inert-work bound, `overCapDraws`) and δ = 1/100 (`creditAtAssumption`),
crediting nothing on a test instance's input (`testCredit`); a refused or rejected run publishes none. -/
def WorkIsTheCredit : Prop :=
  (∀ (i : Input) (opener : Nat → Nat → Except String TileOpening),
    (check i opener).accept = true → ∃ pl, plan i = .ok pl ∧
      (check i opener).work =
        some (testCredit i.testInstance (creditAtAssumption pl.passes i.draws (overCapDraws pl (check i opener).verdicts) DELTA))) ∧
  (∀ (i : Input) (opener : Nat → Nat → Except String TileOpening),
    (check i opener).accept = false → (check i opener).work = none)

/-! ## The trees -/

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
