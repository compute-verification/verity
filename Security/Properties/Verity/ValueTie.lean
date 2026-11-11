import Properties.Verity.Game
import Properties.Pouw.PearlPlusTie

/-!
# The value tie at remote: a served linear's output is PoUW's gates, but for the wrong tiles

As served, a linear layer sits outside the registered circuit: each Pearl+ call's y enters the circuit's blocks as
input gates that are not registered, so the inference-only policy counts every bit of it as fresh input. Pearl+ holds
that y, at the window record W's entries, to the call's function of the call's A rows and the weight's rows
(`Properties.Pouw.PearlPlus.yCell`, `TileTiedAtW`). The tie puts each such linear back as gates and reads the policy
on that circuit. These are statements only: none is proved here, and none is a guarantee yet.

* **The tied circuit** (`Tying`). The served circuit with each credited cell's y gate computed by added gates from the
  call's A row, served gates, and the weight's row, added registered inputs. No circuit Definition of the call is in
  Lean, so the tied circuit is a parameter held to two properties (`Tying.AtW`): its gates at a cell compute `yCell`
  under the pass's salt and the root W registers for the weight, and the commitment opens its y gates, its A row gates
  and its weight rows at W's entries for the cell, the entries `TieMeans` says W registers.
* **The composite step** (`Period.ValueTie`). Over a period of one node's windows, each remote's play, on the period's
  coins: some window and every earlier one are accepted by remote's compiled verifier while the claim fails on them,
  with probability at most the error. The claim (`Period.TiedUpTo`) is, at each of those windows, on the transcript
  sampled proofs' extractor reads off the prover (`Commit.TiedBy`): every served y of a tile that the prover's
  reference opening ties at W's entries, bound in that transcript with its A row, holds the tied gates' output; and the
  policy's verdict on the tied circuit bounds each block's fresh input with the other y's counted as fresh input
  (`Tying.CompliesBut`). And over those windows ∏ (1 − wrong share) is at least 1 − ε, the wrong share
  (`Properties.Pouw.PearlPlus.Run.wrongShare`) the ticket weight of the tiles the reference opening does not tie.
* **The error** (`Period.error`), three terms: PoUW's δ, wherever (1 − ε)^t ≤ δ at `t` draws a window, once for the
  period however many windows it has; the SHA-512 collision term of the tie's finder at a budget of SHA-512 calls
  (`Claims.ecrBound`, the one break term); and SPOT's binding error, a named parameter, once a window.
* **The premises** (`Shared`, `Period.WellFormed`, and the two conditions `PeriodYAtWIsTheCall` reads): the shared
  statements about Pearl+ (`TieMeans`, `AcceptedIsHeldToW`, `PeriodYAtWIsTheCall`, `OpeningBinds`), sampled proofs'
  target at remote, and what makes the period one: each window's reference run committed before its coin, its draw `t`
  tickets on its coin, and the tied circuit laid out at W's entries. Uniform coins are the probability's own, and the
  challenger's log and its signature enter through `Runs.DrawIsTickets` and SPOT's term.

**The prover's openings are adaptive.** Remote reads a tile's opening from the evidence, sent after the coin. The wrong
share is read at reference openings (`Period.ref`), a function of the earlier windows' coins: a tile's reference
opening is one the prover sends for that tile in an accepted play of the window, and there is one wherever some coin
makes the verdict open the tile and accept (`Period.RefIsThePlays`). A tile's check does not read the coin, so a window
accepted on its coin is accepted at its reference openings, and PoUW's term takes no collision. Two openings of one
position that open two values are the finder's: the reference opening against the prover's on another coin, or against
the value sampled proofs bind.

**One node, in time order, or pooled.** `Period.ValueTie` takes the order as a premise
(`Runs.CommittedBeforeCoin`: a window's run is a function of the earlier windows' coins) and counts the accepted
prefix. Windows it does not hold for, one node's that overlap in time or several nodes', pool convexly instead
(`Period.ValueTiePooled`): each run a function of every coin but its own, all windows accepted, and the weighted mean
of the wrong shares at most ε, at any convex weights fixed beforehand.

**What is not tied**, and stays fresh input in the tied circuit:

* the y of a tile in the wrong share, and the y of a cell that is not credited, of a dropped round's call or of a
  version-1 W (`Properties.Pouw.PearlPlus`'s list);
* a y, or a y whose A row holds a gate, that no unit bound in the extractor's transcript reads or writes
  (`Commit.BoundAt`): sampled proofs' claim bounds the unbound units' share;
* whether the A rows are the model's activations: the tied gates read the served gates the layout names.

The share is of ticket weight, the work a tile is credited, not of y bits.

**Constructions the proof defines.** The reference openings and the finder are fields of the period
(`Period.ref`, `Period.finder`), as the extractor and the finders are fields of `Verity.Vocab`: fixed constructions
applied to the prover, never chosen per prover, and the statement is a target at them.
-/

namespace Verity.ValueTie

open Accounting Core
open scoped ENNReal

/-! ## The tied circuit -/

/-- A cell of a served run: the pass, the tile of the pass that holds it, the call's position in the pass, and the row
and the column of the call. -/
structure Cell where
  pass : ℕ
  tile : ℕ
  pos : ℕ
  row : ℕ
  col : ℕ

/-- **A tied circuit of the served circuit `C` with registered inputs `θ`.** The served gates are gates of the tied
circuit of the same width (`emb`). The served y gates (`y`), input gates of `C` that are not registered, are computed in
the tied circuit; every other served gate is as served: an input gate stays one, registered or not as it was, and a
computed gate reads the same gates and applies the same operation. The gates the tied circuit adds compute each y from
its call's A row, served gates (`row`), and the weight's row, added input gates, which are registered and hold
`weights`. `cell` names the cell a y gate holds, `aRow` and `bRow` the bytes of the cell's A row and of its weight's
row as their trees store them, the A row read from the tied circuit's values at `row` alone, and `word` a y gate's value
as the word its tree stores. -/
structure Tying (C : Verity.Circuit) (θ : Finset (Fin C.N)) where
  /-- the tied circuit, with its registered inputs -/
  tied : Verity.Declaration
  emb : Fin C.N → Fin tied.D.N
  emb_inj : Function.Injective emb
  emb_w : ∀ g, tied.D.w (emb g) = C.w g
  /-- the served y gates the tied circuit computes -/
  y : Finset (Fin C.N)
  y_fresh : ∀ g ∈ y, C.isInput g = true ∧ g ∉ θ
  y_gate : ∀ g ∈ y, tied.D.isInput (emb g) = false
  isInput_emb : ∀ g, g ∉ y → tied.D.isInput (emb g) = C.isInput g
  reads_emb : ∀ g, C.isInput g = false → tied.D.reads (emb g) = (C.reads g).map ⟨emb, emb_inj⟩
  op_emb : ∀ g, C.isInput g = false → ∀ v : tied.D.Transcript,
    (tied.D.op (emb g) v).cast (emb_w g) = C.op g fun a => (v (emb a)).cast (emb_w a)
  outputs_emb : ∀ o ∈ C.outputs, emb o ∈ tied.D.outputs
  theta_emb : ∀ g, emb g ∈ tied.θ ↔ g ∈ θ
  /-- an input gate the tied circuit adds is registered -/
  added_registered : ∀ h, (∀ g, emb g ≠ h) → tied.D.isInput h = true → h ∈ tied.θ
  /-- the registered values of the added input gates, the weights' rows; read at those gates alone -/
  weights : tied.D.Transcript
  /-- the cell a served y gate holds -/
  cell : Fin C.N → Cell
  /-- the served gates of the A row of a y gate's cell -/
  row : Fin C.N → Finset (Fin C.N)
  /-- the bytes of the A row of a y gate's cell, as its tree stores the row -/
  aRow : tied.D.Transcript → Fin C.N → ByteArray
  aRow_reads : ∀ (X X' : tied.D.Transcript) (g : Fin C.N), (∀ a ∈ row g, X (emb a) = X' (emb a)) →
    aRow X g = aRow X' g
  /-- the bytes of the weight's row of a y gate's cell, as its tree stores the row -/
  bRow : Fin C.N → ByteArray
  /-- a y gate's value as the 32-bit word its tree stores -/
  word : (g : Fin C.N) → BitVec (C.w g) → ℕ

namespace Tying

variable {C : Verity.Circuit} {θ : Finset (Fin C.N)} (T : Tying C θ)

/-- The tied circuit's transcript `X'` holds the weights' registered rows at the input gates the tied circuit adds. -/
def HoldsWeights (X' : T.tied.D.Transcript) : Prop :=
  ∀ h, (∀ g, T.emb g ≠ h) → T.tied.D.isInput h = true → X' h = T.weights h

/-- **`X'` is the served transcript `X` in the tied circuit**: it holds `X`'s value at every served gate, and the
weights' registered rows at the added input gates. -/
def Extends (X : C.Transcript) (X' : T.tied.D.Transcript) : Prop :=
  (∀ g, (X' (T.emb g)).cast (T.emb_w g) = X g) ∧ T.HoldsWeights X'

/-- **The added gates are evaluated in `X'`**: every computed gate the tied circuit adds holds its operation's value on
`X'`. With `Extends` it fixes `X'`, the served transcript with the linears' interiors filled in. -/
def Evaluated (X' : T.tied.D.Transcript) : Prop :=
  ∀ h, (∀ g, T.emb g ≠ h) → T.tied.D.isInput h = false → X' h = T.tied.D.op h X'

/-- **The served gate `g` holds the tied gates' output in `X'`**: its value is its tied gate's operation on `X'`. -/
def IsGates (X' : T.tied.D.Transcript) (g : Fin C.N) : Prop :=
  X' (T.emb g) = T.tied.D.op (T.emb g) X'

open Classical in
/-- **The served y's that are not the gates' output in `X'`**: the y's that count as fresh input. -/
noncomputable def untied (X' : T.tied.D.Transcript) : Finset (Fin C.N) :=
  T.y.filter fun g => ¬ T.IsGates X' g

/-- The bits of the served y's `B` that the gates `R` of the tied circuit hold or read. -/
def wrongBits (B : Finset (Fin C.N)) (R : Finset (Fin T.tied.D.N)) : ℕ :=
  T.tied.D.width (B.map ⟨T.emb, T.emb_inj⟩ ∩ (R ∪ T.tied.D.into R))

/-- **The fresh input the gates `R` of the tied circuit read, the served y's `B` counted as fresh input**, in bits:
`R`'s novel reads in the tied circuit (`Nci.novelReads`), and the y's of `B` that `R` holds or reads. -/
def freshInput (B : Finset (Fin C.N)) (R : Finset (Fin T.tied.D.N)) : ℕ :=
  Nci.novelReads T.tied R + T.wrongBits B R

/-- **The inference-only policy on the tied circuit, the served y's `B` counted as fresh input**: the tied circuit
divides into blocks of at most `Γ` gates, each reading at most `a` bits of fresh input a gate and the y's of `B` it
holds or reads. It is `Nci.Complies Γ a` on the tied circuit with those y's added to both sides, so what it says beyond
the policy's verdict is which y's are in `B`. -/
def CompliesBut (Γ : ℕ) (a : ℝ) (B : Finset (Fin C.N)) : Prop :=
  ∃ (n : ℕ) (P : Verity.Partition T.tied.D n), ∀ u,
    T.tied.D.size (P.gates u) ≤ Γ ∧
      (T.freshInput B (P.gates u) : ℝ) ≤ a * T.tied.D.size (P.gates u) + T.wrongBits B (P.gates u)

/-- **The tied circuit is the run's calls at W's entries**, under the openings `opensTo` of the served commitment.
Each served y gate `g` holds a credited cell of a tile of a pass of the run `R`, and with the pass's salt and the
entries W registers for the cell's call and weight (`Properties.Pouw.PearlPlus.Registers`: the call's A rows, the
weight's B rows and the call's y, under the call's domains, as `TieMeans` says W registers them):

* the tied gates compute the call's function: in every transcript of the tied circuit whose added gates are evaluated
  on the weights' registered rows, the tied gate's output at `g` is, as a word, `yCell` of the cell's A row and its
  weight's row under the salt and the weight's registered root;
* a value the commitment opens `g` to is the word, at the cell's column, of a row that opens W's y entry at the cell's
  row;
* values the commitment opens the cell's A row gates to are a row that opens W's A entry at the cell's row;
* the weight's row is a row that opens W's B entry at the cell's column. -/
def AtW (opensTo : (g : Fin C.N) → BitVec (C.w g) → Prop) (R : Properties.Pouw.PearlPlus.Run) : Prop :=
  ∀ g ∈ T.y, ∃ (p : PouwCheck.PearlPlus.Pass) (r : PouwCheck.PearlPlus.UnitRecord) (rows cols : List ℕ)
      (da db dy : PouwCommit.RowsDomain) (rootA rootB yRoot : ByteArray),
    R.passes[(T.cell g).pass]? = some p ∧ p.records[(T.cell g).pos]? = some r ∧
    p.tile? (T.cell g).tile = some ((T.cell g).pos, rows, cols) ∧
    (T.cell g).row ∈ (r.credited rows cols).1 ∧ (T.cell g).col ∈ (r.credited rows cols).2 ∧
    PouwCommit.rowsDomain (PouwCommit.activationBinding p.salt r.matmul.index) r.matmul.index r.matmul.m
      PouwCheck.PearlPlus.HM96 false = .ok da ∧
    PouwCommit.rowsDomain (PouwCommit.weightBinding r.matmul.weight) (-2) r.matmul.n
      PouwCheck.PearlPlus.HM96 false = .ok db ∧
    PouwCommit.rowsDomain (PouwCheck.PearlPlus.yBinding p.salt r.matmul.index) r.matmul.index r.realRows
      PouwCheck.PearlPlus.HM96 false = .ok dy ∧
    Properties.Pouw.PearlPlus.Registers R.w
      s!"pearl-a/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length (T.cell g).pass}/{(T.cell g).pos}"
      rootA r.matmul.m r.matmul.k da ∧
    Properties.Pouw.PearlPlus.Registers R.w s!"pearl-b/{r.matmul.weight}" rootB r.matmul.n r.matmul.k db ∧
    Properties.Pouw.PearlPlus.Registers R.w
      s!"pearl-y/{Properties.Pouw.PearlPlus.roundOf R.rounds R.passes.length (T.cell g).pass}/{(T.cell g).pos}"
      yRoot r.realRows r.matmul.n dy ∧
    (∀ X' : T.tied.D.Transcript, T.Evaluated X' → T.HoldsWeights X' →
      Properties.Pouw.PearlPlus.yCell p.salt rootB r.matmul.index r.matmul.k (T.cell g).row (T.cell g).col
          (T.aRow X' g) (T.bRow g) =
        some (T.word g ((T.tied.D.op (T.emb g) X').cast (T.emb_w g)))) ∧
    (∀ v : BitVec (C.w g), opensTo g v → ∃ oy : PouwCheck.PearlPlus.LeafOpening,
      PouwCheck.PearlPlus.opens (.ok dy) yRoot PouwCommit.ROW_SCHEMA oy = true ∧ oy.position = (T.cell g).row ∧
      (PouwCheck.PearlPlus.words32 (PouwCheck.PearlPlus.openedValue oy.value))[(T.cell g).col]! = T.word g v) ∧
    (∀ X' : T.tied.D.Transcript, (∀ a ∈ T.row g, opensTo a ((X' (T.emb a)).cast (T.emb_w a))) →
      ∃ oa : PouwCheck.PearlPlus.LeafOpening,
        PouwCheck.PearlPlus.opens (.ok da) rootA PouwCommit.ROW_SCHEMA oa = true ∧ oa.position = (T.cell g).row ∧
        PouwCheck.PearlPlus.openedValue oa.value = T.aRow X' g) ∧
    ∃ ob : PouwCheck.PearlPlus.LeafOpening,
      PouwCheck.PearlPlus.opens (.ok db) rootB PouwCommit.ROW_SCHEMA ob = true ∧ ob.position = (T.cell g).col ∧
      PouwCheck.PearlPlus.openedValue ob.value = T.bRow g

end Tying

/-! ## A window at remote -/

variable {ι : Type} {dev : Device ι} {α : Type} [Fintype α]

/-- **A window's play at remote, as the one vocabulary reads it**: the parameters, the committed files and the
commitment, the prover's strategy in the game after them (`Verity.Vocab.game`), and the tied circuit of the committed
circuit with its registered inputs. -/
structure Commit (V : Vocab dev α) where
  params : ByteArray
  files : Array ByteArray
  commitment : ByteArray
  strategy : Game.Strategy (V.game params files commitment)
  /-- the tied circuit of the window's committed circuit -/
  tying : Tying (V.circuitOf params files) (V.theta params (V.circuitFile params files))

namespace Commit

variable {V : Vocab dev α} (c : Commit V)

/-- **The served transcript**: the one sampled proofs' extractor reads off the window's strategy (`Vocab.extract`). -/
noncomputable def served : (V.circuitOf c.params c.files).Transcript :=
  V.extract c.params c.files c.commitment c.strategy

/-- **Gate `g` is bound in the served transcript**: it is a committed input or output of a unit that is not unbound
there (`Accounting.unbound`: correct in it, with every committed value the commitment opens to its value there). -/
def BoundAt (g : Fin (V.circuitOf c.params c.files).N) : Prop :=
  ∃ u : Fin (V.units c.params c.files),
    u ∉ unbound (V.partsOf c.params c.files) (V.Opens c.params c.files c.commitment) c.served ∧
    g ∈ (V.partsOf c.params c.files).io u

/-- **The window is tied by the run `R`**: the served transcript extends to one of the tied circuit, the added gates
evaluated on the weights' registered rows, in which

* every served y of a tile that `R`'s openings tie at W's entries (`Properties.Pouw.PearlPlus.TileTiedAtW`), bound in
  the served transcript with the gates of its A row, holds the tied gates' output; and
* where the tied circuit complies with the inference-only policy at the published `Γ` and `α`, it does with the y's
  that are not the gates' output counted as fresh input (`Tying.CompliesBut`). -/
def TiedBy (R : Properties.Pouw.PearlPlus.Run) : Prop :=
  ∃ X' : c.tying.tied.D.Transcript, c.tying.Extends c.served X' ∧ c.tying.Evaluated X' ∧
    (∀ g ∈ c.tying.y,
      Properties.Pouw.PearlPlus.TileTiedAtW R.w R.rounds R.passes R.opener
          ((c.tying.cell g).pass, (c.tying.cell g).tile) →
        c.BoundAt g → (∀ a ∈ c.tying.row g, c.BoundAt a) → c.tying.IsGates X' g) ∧
    (Nci.Complies (V.gates c.params) (V.fresh c.params) c.tying.tied →
      c.tying.CompliesBut (V.gates c.params) (V.fresh c.params) (c.tying.untied X'))

end Commit

/-- **The run Pearl+'s check reads in what the prover sent remote**, and the tiles its plan draws: the three byte
strings decode, Pearl+'s case decodes under the parameters remote names for it, its plan answers, and the run is the
plan's passes with the window record the commitment names and the openings the evidence's blobs give
(`PouwCheck.PearlPlus.Protocol.opener`), which the prover sends after the coin. -/
def PearlPlusRun (s : Sent) (R : Properties.Pouw.PearlPlus.Run) (drawn : List (ℕ × ℕ)) : Prop :=
  ∃ ps c ev, Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components .remote) ps ev,
    x.2.2.id = PouwCheck.PearlPlus.Protocol.protocolId ∧
    ∃ (p : PouwCheck.PearlPlus.Protocol.Params) (cm : PouwCheck.PearlPlus.Protocol.Commitment)
        (e : PouwCheck.PearlPlus.Protocol.Evidence) (pl : PouwCheck.PearlPlus.Plan),
      (canonicalJson x.2.1.params.toUTF8 >>= PouwCheck.PearlPlus.Protocol.paramsOf) = .ok p ∧
      PouwCheck.PearlPlus.Protocol.commitmentOf x.2.2.commitment = .ok cm ∧
      PouwCheck.PearlPlus.Protocol.evidenceOf x.2.2.evidence = .ok e ∧
      PouwCheck.PearlPlus.plan (PouwCheck.PearlPlus.Protocol.inputOf p cm) = .ok pl ∧
      R = Properties.Pouw.PearlPlus.runOf (PouwCheck.PearlPlus.Protocol.inputOf p cm) pl.passes cm.window
        (PouwCheck.PearlPlus.Protocol.opener cm e (Blobs.ofArray s.blobs) pl) ∧
      drawn = pl.picks

/-- **The shared statements the tie reads**: what Pearl+'s tie to W establishes, that an accepted verdict is a run held
to W, that an opening binds its position's value or SHA-512 collides, and sampled proofs' target at remote, whose
extractor's transcript the claim is on. -/
def Shared (V : Vocab dev α) : Prop :=
  Properties.Pouw.PearlPlus.TieMeans ∧ Properties.Pouw.PearlPlus.AcceptedIsHeldToW ∧
    Properties.Pouw.PearlPlus.OpeningBinds ∧ V.SampledProofsSound .remote

/-! ## A period of windows -/

/-- **A period of `n` windows at remote on its coins**, one coin a window, each field any function of the coins:
window `k`'s play as the vocabulary reads it (`commit`) and how it ended (`out`); the windows' runs at the prover's
reference openings (`ref`), which the wrong share is read at; and the tie's collision finder (`finder`). -/
structure Period (V : Vocab dev α) (Ω : Type) (n : ℕ) where
  commit : (Fin n → Ω) → Fin n → Commit V
  out : (Fin n → Ω) → Fin n → Exchange
  /-- the windows' runs at the prover's reference openings -/
  ref : Properties.Pouw.PearlPlus.Runs Ω n
  /-- the tie's collision finder, a fixed construction applied to the prover -/
  finder : Claims.Finder

namespace Period

variable {V : Vocab dev α} {Ω : Type} {n : ℕ} (P : Period V Ω n)

/-- **Window `k` is accepted at remote** on the period's coins `x`: remote's compiled verifier accepts what the prover
sent, for files that passed the commit phase's check. -/
def Accepted (x : Fin n → Ω) (k : Fin n) : Prop :=
  Accepts .remote (P.out x k).sent ∧ V.Registered (P.out x k).params (P.out x k).files

/-- **Window `k` and every earlier one are accepted at remote** on the period's coins `x`. -/
def AcceptedUpTo (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∀ w : Fin n, w ≤ k → P.Accepted x w

/-- Each window's play ends with the parameters, the committed files and the commitment it is read at. -/
def Plays : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n), (P.out x k).params = (P.commit x k).params ∧
    (P.out x k).files = (P.commit x k).files ∧ (P.out x k).commitment = (P.commit x k).commitment

/-- **The reference runs are the plays' runs at the prover's own openings.**

* On coins on which window `k` is accepted, its reference run is the run Pearl+'s check read but for its openings, and
  its draw is the plan's.
* A tile's reference opening is one the prover sends: on some coin of the window, the others held, the window is
  accepted, the verdict opens the tile, and the evidence's opening of it is the reference opening.
* There is one wherever there can be: a tile that the verdict opens in some accepted play of the window has a
  reference opening. -/
def RefIsThePlays : Prop :=
  (∀ (x : Fin n → Ω) (k : Fin n), P.Accepted x k →
    ∃ (R : Properties.Pouw.PearlPlus.Run) (drawn : List (ℕ × ℕ)), PearlPlusRun (P.out x k).sent R drawn ∧
      P.ref.drawn x k = some drawn ∧ P.ref.run x k = { R with opener := (P.ref.run x k).opener }) ∧
  ∀ (x : Fin n → Ω) (k : Fin n) (pi tid : ℕ),
    (∀ o, (P.ref.run x k).opener pi tid = .ok o →
      ∃ (ω : Ω) (R : Properties.Pouw.PearlPlus.Run) (drawn : List (ℕ × ℕ)),
        P.Accepted (Function.update x k ω) k ∧ PearlPlusRun (P.out (Function.update x k ω) k).sent R drawn ∧
        (pi, tid) ∈ PouwCheck.PearlPlus.todo R.passes drawn ∧ R.opener pi tid = .ok o) ∧
    ((∃ (ω : Ω) (R : Properties.Pouw.PearlPlus.Run) (drawn : List (ℕ × ℕ)),
        P.Accepted (Function.update x k ω) k ∧ PearlPlusRun (P.out (Function.update x k ω) k).sent R drawn ∧
        (pi, tid) ∈ PouwCheck.PearlPlus.todo R.passes drawn) →
      ∃ o, (P.ref.run x k).opener pi tid = .ok o)

/-- **Each window's tied circuit is its reference run's calls at W's entries** (`Tying.AtW`), under the openings of
the window's commitment (`Vocab.Opens`). -/
def TiedAtW : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n),
    (P.commit x k).tying.AtW
      (V.Opens (P.commit x k).params (P.commit x k).files (P.commit x k).commitment) (P.ref.run x k)

/-- **What makes the period one**: each play ends at what it is read at, the reference runs are the plays' at the
prover's own openings, and each tied circuit is its reference run's calls at W's entries. -/
def WellFormed : Prop := P.Plays ∧ P.RefIsThePlays ∧ P.TiedAtW

/-- **Window `k` is tied** on the period's coins `x`: by its reference run (`Commit.TiedBy`). -/
def TiedAt (x : Fin n → Ω) (k : Fin n) : Prop := (P.commit x k).TiedBy (P.ref.run x k)

/-- **The tie's claim up to window `k`** at ε, on the period's coins `x`: window `k` and every earlier one are tied,
and their ∏ (1 − wrong share) is at least 1 − ε, each wrong share the ticket weight of the window's tiles that its
reference openings do not tie at W's entries, over its tickets' total. -/
def TiedUpTo (ε : ℝ) (x : Fin n → Ω) (k : Fin n) : Prop :=
  (∀ w : Fin n, w ≤ k → P.TiedAt x w) ∧
    1 - ε ≤ (∏ w : Fin n, (if w ≤ k then 1 - (P.ref.run x w).wrongShare else 1))

/-- **The tie's error**, three terms: PoUW's `δ` for the period; the SHA-512 collision term of the tie's finder at the
budget `q` of SHA-512 calls (`Claims.ecrBound`: `q / 2^256` where the finder meets `ecr/sha-512` within `q` expected
calls, no bound elsewhere); and SPOT's binding error `spot`, once a window. -/
noncomputable def error (δ q : ℝ) (spot : ℝ≥0∞) : ℝ≥0∞ :=
  ENNReal.ofReal δ + Claims.ecrBound sha512Msg P.finder q + (n : ℝ≥0∞) * spot

/-- **The value tie at remote, over a period of one node's windows in time order.** Under the shared statements
(`Shared`, `PeriodYAtWIsTheCall`), for a period that is one (`WellFormed`) whose reference runs are each committed before
their window's coin (`Runs.CommittedBeforeCoin`) and whose draws are each `t` tickets on their window's coin
(`Runs.DrawIsTickets`), and ε, δ with (1 − ε)^t ≤ δ: over the period's coins, some window and every earlier one are
accepted at remote while the tie's claim up to it fails (`TiedUpTo`), with probability at most the tie's error at `δ`,
the budget `q` and SPOT's binding error `spot`, whatever `n` is. It counts the accepted prefix alone. -/
def ValueTie [Fintype Ω] (t : ℕ) (ε δ q : ℝ) (spot : ℝ≥0∞) : Prop :=
  Shared V → Properties.Pouw.PearlPlus.PeriodYAtWIsTheCall → P.WellFormed →
    P.ref.CommittedBeforeCoin → P.ref.DrawIsTickets t → 0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    ENNReal.ofReal (_root_.Pouw.pr fun x : Fin n → Ω => ∃ k : Fin n, P.AcceptedUpTo x k ∧ ¬ P.TiedUpTo ε x k) ≤
      P.error δ q spot

/-- **Each window's reference run is committed apart from its coin**: it is one function of every other window's coin.
No order among the windows is read, and a draw may read every coin. -/
def CommittedApart : Prop :=
  ∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, v ≠ k → x v = x' v) → P.ref.run x k = P.ref.run x' k

/-- **The value tie at remote, pooled over windows in no order**: one node's windows that overlap in time, or several
nodes'. Under the shared statements (`Shared`, the run's `YAtWIsTheCall`), for a period that is one (`WellFormed`) whose
reference runs are each committed apart from their window's coin (`CommittedApart`) and whose draws are each `t`
tickets on their window's coin, convex weights `lam` fixed beforehand, and ε, δ with (1 − ε)^t ≤ δ: over the period's
coins, every window is accepted at remote while some window is not tied or the weighted mean of the wrong shares is
above ε, with probability at most the tie's error. -/
def ValueTiePooled [Fintype Ω] (t : ℕ) (ε δ q : ℝ) (spot : ℝ≥0∞) (lam : Fin n → ℝ) : Prop :=
  Shared V → Properties.Pouw.PearlPlus.YAtWIsTheCall → P.WellFormed →
    P.CommittedApart → P.ref.DrawIsTickets t → 0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    (∀ k, 0 ≤ lam k) → (∑ k, lam k) = 1 →
    ENNReal.ofReal (_root_.Pouw.pr fun x : Fin n → Ω => (∀ k : Fin n, P.Accepted x k) ∧
        ¬ ((∀ k : Fin n, P.TiedAt x k) ∧ (∑ k, lam k * (P.ref.run x k).wrongShare) ≤ ε)) ≤
      P.error δ q spot

end Period

end Verity.ValueTie
