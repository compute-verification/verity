import PouwPearlPlus

/-!
# Pearl+'s run verdict: what its Lean code (`PouwCheck.PearlPlus.check`) does

The properties of the executable verdict (`verity/core/protocols/pouw/lean/pearlplus/`, the port of
`benchmarks/pouw/pearl_c_vllm/verify_run.verify_passes`), stated against `check` and the functions it runs, for every input:
the epoch's coin commitment, the retained passes' manifests, the number of draws, the verifier's coin (a source, or its
record of the coin) and the prover's openings. Each has one shape, the input, the acceptance, then what it establishes.
None is proved yet: `Proofs.Properties.Pouw.PearlPlus` is to prove every one, on Lean core.

What they compose into is the remote theorem's useful-work part: on an accepted run, under Pearl's Assumption 1 at the H100
record per drawn tile (`TTOutTilePearlCH100Rev1`), the run's credit is at most its cost up to the draw's ε at δ, or a
SHA-512 break on the run's trees. These properties are the code's side of it:

* `AcceptMeans`: an accepted run is a plan (`PlanMeans`) whose every drawn and exclusion tile passes `checkTile`.
* `PlanMeans`: the plan's passes are the loaded passes, its commitment theirs, its source the coin record's when there is one
  (retention and give-up fixed before the coin, `after-coin`), and its tiles the draw's over their tickets, keyed by the
  source and the commitment.
* `LoadedGivesUp`: a loaded pass's records are its published root's, and V-EX's give-up is within its caps.
* `TileAcceptMeans`: a tile that passes opens against the pass's roots, its checked words are the committed leaf's, its
  rows keep the per-row rules, its credited cells' replayed debit is within ρ = 1/400 of their credit (as a tile and per
  credited A row), and its served y rows are the unscale of its U.
* `OpeningBinds`: two openings of one position of a tree both accepted name one value, or SHA-512 collides.

The credit at Pearl's assumption (`credit_at_assumption`) and the window record's tie (`tie`) are not ported.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-- The fixed operand's noise the replay uses for tile `tid` of pass `p`: its weight's, once per pass and weight. -/
def noiseOf (p : Pass) (tid : Nat) : Reject WeightNoise :=
  match p.tile? tid with
  | some (pos, _, _) =>
    let u := p.records[pos]!.matmul
    weightNoise p.salt ((p.weightRoots.lookup u.weight).getD ByteArray.empty) u.k
  | none => throw "tile does not match the verifier's own enumeration"

/-! ## The run -/

/-- **What `plan` establishes**: the run's passes are `load`'s of its manifests under the epoch's coins, its commitment is
`runCommitment` of them, its source is the coin record's where the verifier holds one (whose commitment is this run's)
and else the source given, and its tiles are the draw's, keyed by the source and the commitment, then each pass's
exclusion tiles. -/
def PlanMeans : Prop :=
  ∀ (i : Input) (pl : Plan), plan i = .ok pl →
    passesDistinct i.manifests = none ∧ i.manifests.mapM (load i.coins) = .ok pl.passes ∧
    pl.root = runCommitment pl.passes ∧
    (∀ h, i.held = some h → h.commitment = pl.root ∧ pl.source = h.source ∧ (i.source = none ∨ i.source = some h.source)) ∧
    (i.held = none → i.source = some pl.source) ∧
    ∃ key drawn, PouwCheck.PearlPlus.drawKey pl.source pl.root = .ok key ∧ PouwCheck.PearlPlus.draw pl.passes key i.draws = .ok drawn ∧
      pl.todo = todo pl.passes drawn ∧ pl.drawn = drawn.length

/-- **What `check` accepts**: a plan, every one of whose tiles the prover opened and `checkTile` passed, on the fixed
operand's noise of its weight; the verdicts are those tiles', in the plan's order, each passed. -/
def AcceptMeans : Prop :=
  ∀ (i : Input) (opener : Nat → Nat → Except String TileOpening), (check i opener).accept = true →
    ∃ pl, plan i = .ok pl ∧ (check i opener).plan = some pl ∧
      (check i opener).verdicts.map (fun v => (v.pass, v.tile)) = pl.todo ∧
      (∀ v ∈ (check i opener).verdicts, v.reason = none) ∧
      ∀ pi tid, (pi, tid) ∈ pl.todo → ∃ p o, pl.passes[pi]? = some p ∧ opener pi tid = .ok o ∧
        checkTile p (noiseOf p tid) tid o = none

/-- **A run the verdict refuses before any tile is opened carries no commitment**: a refusal of `plan` (`after-coin`, a
pass that doesn't load, passes that share a call) answers with no plan and no verdicts. -/
def RefusedBeforeTheCoin : Prop :=
  ∀ (i : Input) (opener : Nat → Nat → Except String TileOpening) (r : Refusal), plan i = .error r →
    check i opener = { accept := false, reason := some r.reason, refused := r.refused }

/-! ## A pass's records and give-up (V-EX) -/

/-- **What a loaded pass's records are**: each keeps `UnitRecord`'s own rule, together they keep `check_records` (every
unit at its weight's split, one set of excluded and of voluntary B rows per weight, V-EX's count caps ⌈m/64⌉ of A and
⌈n/64⌉ of B), and they are the ones whose root the pass published, under its salt and weights' root. -/
def LoadedGivesUp : Prop :=
  ∀ (coins : CoinCommitment) (man : Lean.Json) (p : Pass), load coins man = .ok p →
    (∀ r ∈ p.records, r.reason = none ∧ r.voluntaryA.length ≤ voluntaryCap r.matmul.m ∧
      r.voluntaryB.length ≤ voluntaryCap r.matmul.n) ∧
    ∃ splits, checkRecords splits p.records = none ∧ recordsRoot p.salt p.weightsRoot splits p.records = p.recordsRoot

/-! ## A tile -/

/-- The tile's operands and checked values, as `checkTile` forms them from its openings. -/
def formed (p : Pass) (wn : Reject WeightNoise) (rows cols : List Nat) (u : Matmul) (o : TileOpening) :
    Reject (Ops × Checked) := do
  let ops ← encode p.salt ((p.weightRoots.lookup u.weight).getD ByteArray.empty) (← wn) u.index u.k rows
    (o.a.map (openedValue ·.value)) cols (o.b.map (openedValue ·.value))
  pure (ops, ← checked ops)

/-- **What a passed tile establishes**: the tile is the verifier's own (`tile?`); its A and B rows and its leaf open, at the
tile's positions, against the call's A root, the weight's root and the pass's transcript root (hm96 under SHA-512); its
words formed from those rows under the pass's salt and the fixed operand's noise are checked and hash to the opened leaf;
its rows keep the per-row rules and its credited cells' replayed debit is within the cap (`Rule.checkOpened`); and on a
call with y, the tile's real rows open against the y root and each word is the unscale of the tile's U
(`Rule.yReason`). -/
def TileAcceptMeans : Prop :=
  ∀ (p : Pass) (wn : Reject WeightNoise) (tid : Nat) (o : TileOpening), checkTile p wn tid o = none →
    ∃ pos rows cols ops ch, p.tile? tid = some (pos, rows, cols) ∧
      let r := p.records[pos]!
      let u := r.matmul
      o.a.map (·.position) = rows ∧ o.b.map (·.position) = cols ∧ o.leaf.position = tid ∧
      o.a.all (opens (rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false) p.rootsA[pos]! ROW_SCHEMA) ∧
      o.b.all (opens (rowsDomain (weightBinding u.weight) (-2) u.n HM96 false)
        ((p.weightRoots.lookup u.weight).getD ByteArray.empty) ROW_SCHEMA) ∧
      opens (transcriptDomain p.salt p.weightsRoot p.count HM96) p.transcriptRoot TILE_SCHEMA o.leaf ∧
      formed p wn rows cols u o = .ok (ops, ch) ∧ leaf ch = openedValue o.leaf.value ∧
      Rule.checkOpened p.scheme r rows cols ops = none ∧
      ∀ yRoot, p.yRoots[pos]! = some yRoot →
        o.y.map (·.position) = rows.filter (· < r.realRows) ∧
        o.y.all (opens (rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false) yRoot ROW_SCHEMA) ∧
        Rule.yReason rows cols ops ch (o.y.map fun y => words32 (openedValue y.value)) = none

/-- **The cap a passed tile keeps** (`Rule.checkOpened`'s `capFailure`): on its credited rows `cr` and columns `cc`, the
replayed debit, 40 per flagged atom-word and 32 per later +0 promotion, is at most ρ = 1/400 of `creditOf(1, |K|, n)/n` per
credited cell, as a tile and on each credited A row (four times that at a census width). -/
def TileKeepsTheCap : Prop :=
  ∀ (s : Scheme) (r : UnitRecord) (rows cols : List Nat) (ops : Ops), Rule.checkOpened s r rows cols ops = none →
    let (cr, cc) := r.credited rows cols
    cr ≠ [] → cc ≠ [] →
    ∃ words : List (List (Nat × Nat)),
      cr.mapM (fun i => cc.mapM fun j => Rule.wordDebit ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]!) = .ok words ∧
      let credit := creditOf 1 r.kKept r.matmul.n
      let rowDebit := words.map fun ws => Rule.debitUnits (ws.foldl (· + ·.1) 0) (ws.foldl (· + ·.2) 0)
      rowDebit.foldl (· + ·) 0 * (Rule.CAP_DEN * r.matmul.n) ≤ Rule.CAP_NUM * credit * cr.length * cc.length ∧
      ∀ d ∈ rowDebit, d * (Rule.CAP_DEN * r.matmul.n) ≤
        Rule.CAP_NUM * credit * cc.length * (if s.censusKs.contains r.matmul.k then 4 else 1)

/-! ## The trees -/

/-- **An opening binds its position's value, or SHA-512 collides**: two openings of one position of one hm96 tree, both
accepted against its root, open one value, or two distinct messages have one compiled SHA-512. -/
def OpeningBinds : Prop :=
  ∀ (d : Except String RowsDomain) (root : ByteArray) (schema : String) (o o' : LeafOpening), o.position = o'.position →
    opens d root schema o = true → opens d root schema o' = true →
      o.value = o'.value ∨ ∃ x y : ByteArray, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

end Properties.Pouw.PearlPlus
