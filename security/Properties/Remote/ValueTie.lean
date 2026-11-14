module

public import Properties.Verity.Game
public import Properties.Pouw.PearlPlusTie

/-!
# The value tie at remote: a served linear's output is PoUW's gates, but for the wrong tiles

As served, a linear layer sits outside the registered circuit: each Pearl+ call's y enters the circuit's blocks as
input gates that are not registered, so the inference-only policy counts every bit of it as fresh input. Pearl+ holds
that y, at the registration record W's entries, to the call's function of the call's A rows and the weight's rows
(`Properties.Pouw.PearlPlus.yCell`, `TileTiedAtW`). The tie puts each such linear back as gates and reads the policy
on that circuit. These are statements only: none is proved here, and none is a guarantee yet. The statements about
Pearl+ are `Properties.Pouw.PearlPlus`'s, read by name and restated nowhere here.

* **The tied circuit** (`Verity.ValueTie.Tying`, with its blocks and its untied gates in `Properties.Verity.Tied`).
  The served circuit with each cell's y computed by the tie's gates from the call's A row, served gates, and the
  weight's row, added registered inputs. No circuit Definition of the call is in Lean, so the tied circuit is a
  parameter held to one property (`Tying.AtW`): at each cell it ties, a real cell of the run whose call and weight W
  registers, its gates compute `yCell` under the pass's salt and the root W registers for the weight.
* **The composite step** (`Phases.YIsTheGates`, `Phases.FleetYIsTheGates`). Over a monitoring period's execution
  phases, each remote's play, on the period's coins: the phases of a sequence up to some phase are accepted by remote's
  compiled verifier on a W that registers the weights while the claim fails on them, with probability at most the
  error. The claim (`Phases.TiedUpTo`) is, at each of those phases, on the transcript SPOT's extractor reads
  off the prover (`Commit.TiedBy`): every served y of a tile that the run's table ties at W's entries, bound in that
  transcript with its A row, holds the tied gates' output; and where the registered file's circuit complies with the
  inference-only policy, the tied circuit's blocks hold `Γ` on their registered rows with the untied cells' y counted
  as fresh input (`Tying.CompliesBut`). And over those phases
  ∏ (1 − wrong share) is at least 1 − ε, the wrong share (`Properties.Pouw.PearlPlus.Run.wrongShare`) the ticket weight
  of the tiles the table does not tie. `Phases.FleetYIsTheGates` is the same claim pooled across sequences, at
  `FleetYAtWIsTheCall`'s threshold.
* **The error** (`tieError`), a sum of named terms: PoUW's δ, wherever (1 − ε)^t ≤ δ at `t` samples a phase, once for
  the sequence however many phases it has; two SHA-512 collision terms, each a finder's budget over 2^256 where the
  finder meets `ecr/sha-512` (`Claims.ExpectedCR`, the form of `Verity.Vocab.finderPr_le_crBound`): the adaptive
  opener's (`Properties.Pouw.PearlPlus.Runs.finder`, PoUW's own) and the tie's (`Phases.finder`); and SPOT's term
  (`Phases.spotMiss`), the probability that a counted phase holds a served y that SPOT binds and that is not
  shown at W's entries, which SPOT's binding error bounds, once a phase.
* **The premises**: `AcceptedIsHeldToW` and `OpeningBinds` (`Shared`), PoUW's period statement
  (`PeriodYAtWIsTheCall`, or `FleetYAtWIsTheCall`) with its two conditions on the runs (`Runs.CommittedBeforeCoin`,
  `Runs.DrawIsTickets`), `ecr/sha-512` at the two finders, and what makes the period one (`Phases.WellFormed`). Uniform
  coins are the probability's own, and the challenger's log and its signature enter through `Runs.DrawIsTickets`.

**Two commitments, one value.** A served y is under SPOT's commitment, which the extractor's transcript is
read from, and under Pearl+'s, whose values the run's table holds (`Properties.Pouw.PearlPlus.Run.table`). The claim is
on the first and PoUW's statement on the second. They meet at W: `Phases.shown` is the rows of W's trees that SPOT's
openings show, a fixed construction applied to the prover as the extractor is. A y is shown at W's entries
(`Tying.ShownAt`) when those rows open W's entries for its cell to its served value, its served A row and the weight's
row. Where a shown row and the table's row of one position differ, two openings of one of W's roots are accepted and
SHA-512 collides on the messages checking them (`OpeningBinds`): the tie's finder (`Phases.finder`) outputs the first
such collision. That a y SPOT binds is shown at W's entries is SPOT's to establish, by the
extractor's openings and the public layout that puts a y gate's registered position at W's entry for its cell;
`Phases.spotMiss` is the probability that it fails, and no premise here.

**One sequence, or pooled.** The period's phases, their sequences and their order are the runs'
(`Properties.Pouw.PearlPlus.Runs`): a node's runs of record in time order are one sequence, and phases that overlap in
time are sequences of their own, pooled and never multiplied. `Phases.YIsTheGates` counts one sequence's accepted
prefix to any end; `Phases.FleetYIsTheGates` pools the prefixes to ends fixed before the coins.

**What is not tied**, and counts as fresh input:

* the y of a cell that is not credited, of a tile in the wrong share, or that is not shown at W's entries: the tie
  computes it, and the claim reads it nowhere, so it counts as fresh input wherever its served y is not the tie's
  output;
* the y of a dropped round's call or of a version-1 W (`Properties.Pouw.PearlPlus`'s list): no tied circuit that is
  the run's calls at W's entries computes it, so it stays an input gate that is not registered. A phase accepted on a
  version-1 W is not counted, and ends its sequence's prefix as a refusal does;
* a y that no unit bound in the extractor's transcript reads, or whose A row holds a gate no such unit reads or writes
  (`Commit.BoundAt`): SPOT's own claim (`Verity.Vocab.SpotSound`) bounds the unbound units' share.

The statement does not say the A rows are the model's activations: the tied gates read the served gates the layout
names. The wrong share is of ticket weight, the work a tile is credited, not of y bits.
-/

@[expose] public section

namespace Verity.ValueTie

open Accounting Core

/-! ## The tied circuit at W's entries -/

/-- **The cell `c` is a real cell of the run `R`, at W's entries.** Its pass `p`, the call's record `r`, a row below
the call's real rows and a column below its width; and W registers the call's A rows, the weight's B rows and the
call's y (`Properties.Pouw.PearlPlus.Registers`) under the roots `rootA`, `rootB` and `yRoot` and the call's domains
`da`, `db` and `dy`, at the pass's salt: the entries `TieMeans` says W registers, as `CellAtW` reads them. -/
def CellAt (R : Properties.Pouw.PearlPlus.Run) (c : Cell) (p : PouwCheck.PearlPlus.Pass)
    (r : PouwCheck.PearlPlus.UnitRecord) (da db dy : PouwCommit.RowsDomain) (rootA rootB yRoot : ByteArray) : Prop :=
  R.passes[c.1]? = some p ∧ p.records[c.2.1]? = some r ∧ c.2.2.1 < r.realRows ∧ c.2.2.2 < r.matmul.n ∧
    PouwCommit.rowsDomain (PouwCommit.activationBinding p.salt r.matmul.index) r.matmul.index r.matmul.m
      PouwCheck.PearlPlus.HM96 false = .ok da ∧
    PouwCommit.rowsDomain (PouwCommit.weightBinding r.matmul.weight) (-2) r.matmul.n
      PouwCheck.PearlPlus.HM96 false = .ok db ∧
    PouwCommit.rowsDomain (PouwCheck.PearlPlus.yBinding p.salt r.matmul.index) r.matmul.index r.realRows
      PouwCheck.PearlPlus.HM96 false = .ok dy ∧
    Properties.Pouw.PearlPlus.Registers R.w
      s!"pearl-a/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length c.1}/{c.2.1}"
      rootA r.matmul.m r.matmul.k da ∧
    Properties.Pouw.PearlPlus.Registers R.w s!"pearl-b/{r.matmul.weight}" rootB r.matmul.n r.matmul.k db ∧
    Properties.Pouw.PearlPlus.Registers R.w
      s!"pearl-y/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length c.1}/{c.2.1}"
      yRoot r.realRows r.matmul.n dy

/-- **The cell `c` is credited in a tile the run `R`'s table ties at W's entries**: a credited cell
(`UnitRecord.credited`) of a tile of its pass and call, and that tile tied (`Properties.Pouw.PearlPlus.TileTiedAtW`). -/
def CreditedTied (R : Properties.Pouw.PearlPlus.Run) (c : Cell) : Prop :=
  ∃ (p : PouwCheck.PearlPlus.Pass) (tile : ℕ) (rows cols : List ℕ),
    R.passes[c.1]? = some p ∧ p.tile? tile = some (c.2.1, rows, cols) ∧
    c.2.2.1 ∈ (p.records[c.2.1]!.credited rows cols).1 ∧ c.2.2.2 ∈ (p.records[c.2.1]!.credited rows cols).2 ∧
    Properties.Pouw.PearlPlus.TileTiedAtW R.w R.rounds R.passes R.table (c.1, tile)

/-- **The row `e` opens W's value `v` at position `i`** under the domain `d` and the root `root`: it is a row of that
value, domain and root, and its opening is of position `i` and opens against the root. -/
def OpensAt (e : Properties.Pouw.PearlPlus.RowAtW) (v : String) (d : PouwCommit.RowsDomain) (root : ByteArray)
    (i : ℕ) : Prop :=
  e.value = v ∧ e.domain = .ok d ∧ e.root = root ∧ e.opening.position = i ∧
    PouwCheck.PearlPlus.opens (.ok d) root PouwCommit.ROW_SCHEMA e.opening = true

namespace Tying

variable {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Cut C} (T : Tying C θ L)

/-- **The tied circuit is the run's calls at W's entries: its gates compute the product Pearl+ proved.** Each cell it
ties is a real cell of the run `R` at W's entries (`CellAt`), and in every transcript of the tied circuit whose added
gates are evaluated on the weights' registered rows, the tie's output at the cell's y gates is, as the word its tree
stores, `yCell` of the cell's served A row and its weight's row under the pass's salt and the root W registers for the
weight: the unscale of the tile, one function from a call's x and weight rows to its y. -/
def AtW (R : Properties.Pouw.PearlPlus.Run) : Prop :=
  ∀ c ∈ T.cells, ∃ (p : PouwCheck.PearlPlus.Pass) (r : PouwCheck.PearlPlus.UnitRecord)
      (da db dy : PouwCommit.RowsDomain) (rootA rootB yRoot : ByteArray),
    CellAt R c p r da db dy rootA rootB yRoot ∧
    ∀ X' : T.tied.D.Trace, T.Evaluated X' → T.HoldsWeights X' →
      Properties.Pouw.PearlPlus.yCell p.salt rootB r.matmul.index r.matmul.k c.2.2.1 c.2.2.2
          (T.aRow (T.served X') c) (T.bRow c) =
        some (L.word c (T.gatesOut X'))

/-- **The cell `c` is shown at W's entries by the rows `S`**, in the served transcript `X` and the run `R`: it is a
real cell of the run at W's entries (`CellAt`), and rows of `S` open W's y entry at the cell's row to a row whose word
at the cell's column is the cell's served word, W's A entry at the cell's row to the cell's served A row, and W's B
entry at the cell's column to the weight's row. -/
def ShownAt (R : Properties.Pouw.PearlPlus.Run) (S : List Properties.Pouw.PearlPlus.RowAtW) (X : C.Trace)
    (c : Cell) : Prop :=
  ∃ (p : PouwCheck.PearlPlus.Pass) (r : PouwCheck.PearlPlus.UnitRecord)
      (da db dy : PouwCommit.RowsDomain) (rootA rootB yRoot : ByteArray),
    CellAt R c p r da db dy rootA rootB yRoot ∧
    (∃ e ∈ S,
      OpensAt e s!"pearl-y/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length c.1}/{c.2.1}" dy yRoot c.2.2.1 ∧
      (PouwCheck.PearlPlus.words32 (PouwCheck.PearlPlus.openedValue e.opening.value))[c.2.2.2]? =
        some (L.word c X)) ∧
    (∃ e ∈ S,
      OpensAt e s!"pearl-a/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length c.1}/{c.2.1}" da rootA c.2.2.1 ∧
      PouwCheck.PearlPlus.openedValue e.opening.value = T.aRow X c) ∧
    ∃ e ∈ S, OpensAt e s!"pearl-b/{r.matmul.weight}" db rootB c.2.2.2 ∧
      PouwCheck.PearlPlus.openedValue e.opening.value = T.bRow c

end Tying

/-! ## An execution phase at remote -/

variable {ι : Type} {dev : Device ι} {α : Type} [Fintype α]

/-- **An execution phase's play at remote, as the one vocabulary reads it**: the parameters, the committed files and the
commitment, the prover's strategy in the game after them (`Verity.Vocab.game`), and the tied circuit of the committed
circuit with its registered inputs at the vocabulary's cut (`Verity.Vocab.cut`), whose y no registered input is. -/
structure Commit (V : Vocab dev α) where
  params : ByteArray
  files : Array ByteArray
  commitment : ByteArray
  strategy : Game.Strategy (V.game params files commitment)
  /-- the tied circuit of the phase's committed circuit -/
  tying : Tying (V.circuitOf params files) (V.theta params files) (V.cut params files)

namespace Commit

variable {V : Vocab dev α} (c : Commit V)

/-- **The served transcript**: the one SPOT's extractor reads off the phase's strategy (`Vocab.extract`). -/
noncomputable def served : (V.circuitOf c.params c.files).Trace :=
  V.extract c.params c.files c.commitment c.strategy

/-- **Gate `g` is bound in the served transcript**: it is a committed input or output of a unit that is not unbound
there (`Accounting.unbound`: correct in it, with every committed value the commitment opens to its value there). -/
def BoundAt (g : Fin (V.circuitOf c.params c.files).N) : Prop :=
  ∃ u : Fin (V.units c.params c.files),
    u ∉ unbound (V.partsOf c.params c.files) (V.Opens c.params c.files c.commitment) c.served ∧
    g ∈ (V.partsOf c.params c.files).io u

/-- **The cell `e` is one the claim reads**, at the run `R`: it is credited in a tile the run's table ties at W's
entries (`CreditedTied`), and its y gates and the gates of its A row are bound in the served transcript. -/
def Reads (R : Properties.Pouw.PearlPlus.Run) (e : Cell) : Prop :=
  CreditedTied R e ∧ (∀ g ∈ (V.cut c.params c.files).yAt e, c.BoundAt g) ∧ ∀ a ∈ c.tying.row e, c.BoundAt a

/-- **The phase is tied by the run `R`**: the served transcript extends to one of the tied circuit, the added gates
evaluated on the weights' registered rows, in which

* every y the claim reads (`Reads`: of a cell the tie ties, credited in a tile the run's table ties at W's entries,
  bound in the served transcript with the gates of its A row) holds the tied gates' output; and
* where the registered file's circuit complies with the inference-only policy at the published `Γ` and `α`
  (`Verity.Vocab.declaration`), the tied circuit does with each block held to `Γ` on its registered rows and the
  untied cells' y counted as fresh input (`Tying.CompliesBut`). -/
def TiedBy (R : Properties.Pouw.PearlPlus.Run) : Prop :=
  ∃ X' : c.tying.tied.D.Trace, c.tying.Extends c.served X' ∧ c.tying.Evaluated X' ∧
    (∀ e ∈ c.tying.cells, c.Reads R e → ∀ g ∈ (V.cut c.params c.files).yAt e, c.tying.IsGates X' g) ∧
    (Nci.CompliesRate (V.gates c.params) (V.fresh c.params) (V.declaration c.params c.files) →
      c.tying.CompliesBut (V.gates c.params) (V.fresh c.params) (c.tying.untiedCells X'))

/-- **A y the claim reads is not shown at W's entries** by the rows `S`, at the run `R` (`Tying.ShownAt` in the served
transcript). -/
def Unshown (R : Properties.Pouw.PearlPlus.Run) (S : List Properties.Pouw.PearlPlus.RowAtW) : Prop :=
  ∃ e ∈ c.tying.cells, c.Reads R e ∧ ¬ c.tying.ShownAt R S c.served e

end Commit

/-- **The run Pearl+'s check reads in what the prover sent remote**, the openings it takes and the tiles its plan
samples: the three byte strings decode, Pearl+'s case decodes under the parameters remote names for it, its plan
answers, the run `R` is the plan's passes with the registration record the commitment names, at its own table, the
openings `O` are the ones the evidence's blobs give (`PouwCheck.PearlPlus.Protocol.opener`), which the prover sends
after the coin, and the sample is the plan's. -/
def PearlPlusRun (s : Sent) (R : Properties.Pouw.PearlPlus.Run) (O : Properties.Pouw.PearlPlus.Opener)
    (drawn : Option (List (ℕ × ℕ))) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components .remote) ps ev,
    x.2.2.id = PouwCheck.PearlPlus.Protocol.protocolId ∧
    ∃ (p : PouwCheck.PearlPlus.Protocol.Params) (cm : PouwCheck.PearlPlus.Protocol.Commitment)
        (e : PouwCheck.PearlPlus.Protocol.Evidence) (pl : PouwCheck.PearlPlus.Plan),
      (canonicalJson x.2.1.params.toUTF8 >>= PouwCheck.PearlPlus.Protocol.paramsOf) = .ok p ∧
      PouwCheck.PearlPlus.Protocol.commitmentOf x.2.2.commitment = .ok cm ∧
      PouwCheck.PearlPlus.Protocol.evidenceOf x.2.2.evidence = .ok e ∧
      PouwCheck.PearlPlus.plan (PouwCheck.PearlPlus.Protocol.inputOf p cm) = .ok pl ∧
      R = Properties.Pouw.PearlPlus.runOf (PouwCheck.PearlPlus.Protocol.inputOf p cm) pl.passes cm.window R.table ∧
      O = PouwCheck.PearlPlus.Protocol.opener cm e (Blobs.ofArray s.blobs) pl ∧
      drawn = some pl.picks

/-- **The shared statements the composite step reads beside PoUW's period statement**: an accepted verdict is a run
held to W, and an opening binds its position's value or SHA-512 collides. -/
def Shared : Prop :=
  Properties.Pouw.PearlPlus.AcceptedIsHeldToW ∧ Properties.Pouw.PearlPlus.OpeningBinds

/-- **The tie's error**, a sum of named terms: PoUW's `δ`; the adaptive opener's SHA-512 collision term, the budget `q`
of PoUW's finder over 2^256; the tie's, the budget `q'` of the tie's finder over 2^256; and SPOT's term `miss`. -/
noncomputable def tieError (δ : ℝ) (q q' : ℕ) (miss : ℝ) : ℝ :=
  δ + (q : ℝ) / 2 ^ (256 : ℝ) + (q' : ℝ) / 2 ^ (256 : ℝ) + miss

/-! ## A monitoring period -/

/-- **A monitoring period of `n` execution phases at remote on its coins**, one coin a phase, each field any function
of the coins: phase `k`'s play as the vocabulary reads it (`commit`) and how it ended (`out`); the phases' Pearl+ runs
with their tables, samples and openings, in their sequences (`runs`); and the rows of W's trees that SPOT's openings
show (`shown`), a fixed construction applied to the prover as `Verity.Vocab.extract` is. -/
structure Phases (V : Vocab dev α) (Ω : Type) (n : ℕ) where
  commit : (Fin n → Ω) → Fin n → Commit V
  out : (Fin n → Ω) → Fin n → Exchange
  /-- the phases' Pearl+ runs, the ones PoUW's period statement is about -/
  runs : Properties.Pouw.PearlPlus.Runs Ω n
  /-- the rows of W's trees that SPOT's openings show in each phase -/
  shown : (Fin n → Ω) → Fin n → List Properties.Pouw.PearlPlus.RowAtW

namespace Phases

variable {V : Vocab dev α} {Ω : Type} {n : ℕ} (P : Phases V Ω n)

/-- **Phase `k` is accepted at remote** on the period's coins `x`: remote's compiled verifier accepts what the prover
sent, for files that passed the commit phase's check. -/
def Accepted (x : Fin n → Ω) (k : Fin n) : Prop :=
  Accepts .remote (P.out x k).sent ∧ V.Registered (P.out x k).params (P.out x k).files

/-- **Phase `k` is counted** on the period's coins `x`: it is accepted at remote, and its run is held to a W that
registers the weights (`Properties.Pouw.PearlPlus.Run.HeldToW`, a version-2 W). -/
def Counted (x : Fin n → Ω) (k : Fin n) : Prop :=
  P.Accepted x k ∧ (P.runs.run x k).HeldToW

/-- **Phase `k` and every earlier phase of its sequence are counted** on the period's coins `x`. -/
def CountedUpTo (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∀ w : Fin n, P.runs.seq w = P.runs.seq k → w ≤ k → P.Counted x w

/-- Each phase's play ends with the parameters, the committed files and the commitment it is read at. -/
def Plays : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n), (P.out x k).params = (P.commit x k).params ∧
    (P.out x k).files = (P.commit x k).files ∧ (P.out x k).commitment = (P.commit x k).commitment

/-- **The runs are the plays' runs.** On coins on which phase `k` is accepted at remote, its run is the run Pearl+'s
check read, at its own table, its openings are the ones the check took and its sample the plan's (`PearlPlusRun`), and
its table holds what those openings open (`Properties.Pouw.PearlPlus.Run.Holds`). A table that does so on every such
coin exists for every prover: the prover's own opening of each row on one coin on which it is opened. -/
def RunsAreThePlays : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n), P.Accepted x k →
    PearlPlusRun (P.out x k).sent (P.runs.run x k) (P.runs.opener x k) (P.runs.drawn x k) ∧
    ∀ d : List (ℕ × ℕ), P.runs.drawn x k = some d → (P.runs.run x k).Holds (P.runs.opener x k) d

/-- **Each counted phase's tied circuit is its run's calls at W's entries** (`Tying.AtW`). -/
def TiedAtW : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n), P.Counted x k → (P.commit x k).tying.AtW (P.runs.run x k)

/-- **What makes the period one**: each play ends at what it is read at, the runs are the plays', and each tied
circuit is its run's calls at W's entries. -/
def WellFormed : Prop := P.Plays ∧ P.RunsAreThePlays ∧ P.TiedAtW

/-- **The messages SHA-512 hashes in checking the shown rows against the tables'**: for each phase and each row of W's
trees that SPOT's openings show there, where the phase's table holds an opening of the row's position, the
messages checking the two hash (`Properties.Pouw.PearlPlus.rowsHashed`). -/
def hashed (x : Fin n → Ω) : List ByteArray :=
  (List.finRange n).flatMap fun k =>
    (P.shown x k).flatMap fun e =>
      match e.domain, (P.runs.run x k).table e.value e.opening.position with
      | .ok d, some o' =>
        Properties.Pouw.PearlPlus.rowsHashed d e.opening ++ Properties.Pouw.PearlPlus.rowsHashed d o'
      | _, _ => []

/-- **The tie's finder**: on the period's coins, the first SHA-512 collision (`Properties.Pouw.PearlPlus.collisionIn`)
among the messages checking the shown rows and the tables' hash (`hashed`), its cost declared at the budget `q` for the
period's whole SHA-512 work, SPOT's openings included, as `Properties.Pouw.PearlPlus.Runs.finder`'s is. -/
noncomputable def finder [Fintype Ω] [Nonempty Ω] (q : ℕ) : _root_.Claims.Finder where
  S := Fin n → Ω
  out x := (Properties.Pouw.PearlPlus.collisionIn (P.hashed x)).map Properties.Pouw.PearlPlus.msgPair
  cost _ := q

/-- **Phase `k` is tied** on the period's coins `x`: by its run (`Commit.TiedBy`). -/
def TiedAt (x : Fin n → Ω) (k : Fin n) : Prop := (P.commit x k).TiedBy (P.runs.run x k)

/-- **Phase `k` and every earlier phase of its sequence are tied** on the period's coins `x`. -/
def TiedPrefix (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∀ w : Fin n, P.runs.seq w = P.runs.seq k → w ≤ k → P.TiedAt x w

/-- **The tie's claim up to phase `k`** at ε, on the period's coins `x`: phase `k` and every earlier phase of its
sequence are tied, and their ∏ (1 − wrong share) (`Properties.Pouw.PearlPlus.Runs.tiedUpTo`) is at least 1 − ε, each
wrong share the ticket weight of the phase's tiles that its table does not tie at W's entries, over its tickets'
total. -/
def TiedUpTo (ε : ℝ) (x : Fin n → Ω) (k : Fin n) : Prop :=
  P.TiedPrefix x k ∧ 1 - ε ≤ P.runs.tiedUpTo x k

/-- **SPOT's term over the phases `K`**: the probability, over the period's coins, that some phase of `K` is counted
while a served y the claim reads there is not shown at W's entries (`Commit.Unshown`). SPOT's binding error bounds it,
once a phase of `K`. -/
noncomputable def spotMiss [Fintype Ω] (K : Finset (Fin n)) : ℝ :=
  _root_.Pouw.pr fun x : Fin n → Ω =>
    ∃ k ∈ K, P.Counted x k ∧ (P.commit x k).Unshown (P.runs.run x k) (P.shown x k)

/-- **The value tie at remote, over one sequence of a monitoring period's execution phases.** Under the shared
statements (`Shared`, `PeriodYAtWIsTheCall`), for a period that is one (`WellFormed`) whose runs are each committed
before their phase's coin (`Runs.CommittedBeforeCoin`) and whose samples are each `t` tickets on their phase's coin
(`Runs.DrawIsTickets`), ε, δ with (1 − ε)^t ≤ δ, and budgets `q` and `q'` at which PoUW's finder and the tie's meet
`ecr/sha-512`: for each sequence `a`, over the period's coins, some phase of `a` and every earlier one of `a` are
counted while the tie's claim up to it fails (`TiedUpTo`), with probability at most the tie's error at `δ`, the two
budgets and SPOT's term over `a`'s phases, however many phases the period has. It counts the accepted prefix alone. -/
def YIsTheGates [Fintype Ω] [Nonempty Ω] (t q q' : ℕ) (ε δ : ℝ) : Prop :=
  Shared → Properties.Pouw.PearlPlus.PeriodYAtWIsTheCall → P.WellFormed →
    P.runs.CommittedBeforeCoin → P.runs.DrawIsTickets t → 0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    _root_.Claims.ExpectedCR sha512Msg (P.runs.finder q) → _root_.Claims.ExpectedCR sha512Msg (P.finder q') →
    ∀ a : ℕ,
      _root_.Pouw.pr (fun x : Fin n → Ω =>
          ∃ k : Fin n, P.runs.seq k = a ∧ P.CountedUpTo x k ∧ ¬ P.TiedUpTo ε x k) ≤
        tieError δ q q' (P.spotMiss (Finset.univ.filter fun k => P.runs.seq k = a))

/-- **The value tie at remote, pooled convexly across sequences.** Under the shared statements (`Shared`,
`FleetYAtWIsTheCall`), for a period as in `YIsTheGates`, prefix ends `ends` fixed before its coins and 0 < δ: over the
period's coins, some nonempty `K ⊆ ends` has every prefix counted while some phase of those prefixes is not tied or
(∏ over `K` of the prefix's ∏ (1 − wrong share))^t is below (|K| δ / |ends|)^|K|, `FleetYAtWIsTheCall`'s threshold,
with probability at most the tie's error at `δ`, the two budgets and SPOT's term over every phase. -/
def FleetYIsTheGates [Fintype Ω] [Nonempty Ω] (ends : Finset (Fin n)) (t q q' : ℕ) (δ : ℝ) : Prop :=
  Shared → Properties.Pouw.PearlPlus.FleetYAtWIsTheCall → P.WellFormed →
    P.runs.CommittedBeforeCoin → P.runs.DrawIsTickets t → 0 < δ →
    _root_.Claims.ExpectedCR sha512Msg (P.runs.finder q) → _root_.Claims.ExpectedCR sha512Msg (P.finder q') →
    _root_.Pouw.pr (fun x : Fin n → Ω =>
        ∃ K : Finset (Fin n), K ⊆ ends ∧ K.Nonempty ∧ (∀ k ∈ K, P.CountedUpTo x k) ∧
          ¬ ((∀ k ∈ K, P.TiedPrefix x k) ∧
            ((K.card : ℝ) * δ / (ends.card : ℝ)) ^ K.card ≤ (∏ k ∈ K, P.runs.tiedUpTo x k) ^ t)) ≤
      tieError δ q q' (P.spotMiss Finset.univ)

end Phases

end Verity.ValueTie
