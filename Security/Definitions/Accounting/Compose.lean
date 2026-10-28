import Definitions.Accounting.TS
import Definitions.Accounting.Gates.Model
import Definitions.Accounting.Device
import Definitions.Pouw.Window

/-!
# PoUW as a compile pass over an arbitrary declared circuit (trusted)

The vocabulary of `Definitions.Accounting.Guarantees.Compose`. The
compile pass `C → C′` replaces each credited product of a declared circuit by its kind's template, with its noise drawn
after its operands are committed, one coin a round. The service then draws one uniform subset of the
window's grid-padded tiles, one a proof unit, and proves each drawn tile with the generation of its own noise.

Each round's coin is `live` (the verifier's, released after the round's served values are committed) or `seeded`
(a seed: SHA-512 of the window's salt, the unit's position and its operands' roots, Fiat–Shamir). The live statements
(`CircuitWork`, `CircuitCredit`) describe the designed protocol: PoUW's code today has one salt a window and derives
the noise in the scheme (`verity/core/protocols/pouw/coins.py`'s `LABELS`, `PROTOCOL.md`), so every window it runs is
seeded, and its credit is the seeded statements' (`CircuitWorkSeeded`, `CircuitCreditSeeded`), at a grinding premium
per seed group (GRIND, one seed a call), once its seeds name what each round reads (`SeedNamed`, which
today's code doesn't meet).

It has two layers. The logical layer, from `Declared` to `WorkAt`, is the circuit, its rounds, its
proof units, the templates' luck, the service's draw and what an accepted window establishes about the run's committed
words; it reads no machine model. The physical layer, from `creditOf` to `certOf`, is the tensor–scalar model on a
data sheet, which prices those words as time on a declared device.

The logical layer:

* `CoinMode`, `Declared`: the circuit as the IR gives it, a `Accounting.Gates.IntDag` (memory accounting's form, with its
  wiring), each gate's Definition call site, each call site's round (the coin its noise reads: one coin a round),
  each round's coin mode (live or seeded), the coin before which the service commits each value under
  a root (`Declared.commitRound`, where the proofs' layer cuts it, and `LayerCommitted` holds it to the IR's cut, or
  to the draw for a computed value only its own unit reads),
  which values each round's seed preimage names (`Declared.seedNames`, which `SeedNamed` reads), and each gate's
  proof unit (an
  output coordinate, a K-chunk committing its running sum, which no served kind is cut into, or a unit
  of glue). `Declared.Wf`: the call sites, rounds and units in range, one unit's gates in
  one round, each coin input read only in its own round, and one unit's words of one kind.
* `Declared.binding`, `Declared.Reads`, `Declared.Writes`: each gate's input binding, which gates read which, and the
  values the prover writes (every gate but a coin input).
* `isWord`, `wordGates`, `Kind`, `kindAt`: the circuit's committed words of served products (`T_t`), and each gate's
  kind, computed from the IR, never asserted: its primitive, its place's statics (what `Gates.roleOf` reads) and
  whether each operand depends on a run input (an activation) or not (a weight), a receipt word's read off its tile's
  operand rows. So a gather-fed `QKᵀ` and a weight matmul of one shape are different kinds.
* `unitGates`, `unitWords`, `Declared.roundOf`, `Declared.stageOf`, `Declared.kindOf`: a unit's gates and words, its
  round, its stage counted from the last round (as `SplitStages` counts) and its words' kind (none for glue).
* `coinMark`, `Declared.firstRead`, `Declared.cutOf`, `Declared.Late`, `Declared.UnitLate`, `Declared.Reaches`: each
  value's cut, computed from the IR (W1): a registered input before the first coin, a run input (decode feedback among
  them) before its first reader's round's coin (where `LayerCommitted` says the service commits it), and every
  other value (glue, the KV cache) with its writing gate's round, before that round's coin or, when it depends on it,
  after; a read is late when its operand's cut is past the reader's coin; a unit writes another late when one of its
  gates is read late by one of the other's, and its late cone is closed under that per unit, as
  `DrawnLayer.Late` is.
* `LayerCommitted`, `Declared.atDraw`, `Declared.atIR`: the service commits every value the prover writes by its IR
  cut, or a computed value only its own unit reads (a receipt) by the draw (protocol structure, in `Scope`, which
  today's code doesn't meet); the circuit with every value committed at the
  draw, where the seeded statements check binding, and with every value cut at the IR, where they count cones.
* `LuckForm`, `Scheme`, `Scheme.LuckProved`, `Scheme.SeedPriced`, `Scheme.foldLuck`: per kind, the floor its
  template's noise proves at the worst operands (`0` where its noise proves none), whether it is
  credited, its skip pattern's span, the form of its template's proved operand luck, the three lucks a live window is
  held to: (H-μ)'s point mass `μ`, the receipt's B′ luck `r_B` and the whole-unit guess bound, and the rates its
  seeded gains are proved at: its template's folded one (GRIND) and its whole skip's (from a whole-tile luck a seed
  query, which no ruling grants); no credited
  kind reading two activations (H1 + fx1's attention) under seeded coins. A folded unit's luck is `μ + (1 − μ)·r_B` (`7/8` at `μ = ¾`, `r_B = ½`,
  `Accounting.Guarantees.TS.LuckOfReceipt`).
* `creditedWords`, `wordsOf`, `floorOf`, `kindCount`: the circuit's credited words, `N_w` (W3: from the IR, not a
  parameter), its floor `♭ = Σ_words ♭_{kind}` and its words per kind.
* `tileWords`, `tileWeight`, `tileStage`, `tileKind`, `tileLuck`, `tileLuck0`: the service's `N_p` grid-padded tiles
  (`verity/core/protocols/pouw/window.py`): unit `t` for `t < units`, and fillers of no words past them.
* `tileCone`, `shareCone`, `zeroCone`, `kindF`: a tile's late cone's weight, which the service's share
  charges a skipped tile, its detected weight (its cone when it carries words), the cone weight on tiles of no words
  (W4, `0` for every Program COMPILE accepts), and `F` per kind: the most tiles one late cone of a kind holds (`1` for
  every kind served: its units hold whole K).
* `SplitStages`, `modeAt`, `sameAt`, `SplitStages.LuckAt`, `EveryStage`, `share`, `stageShare`, `tileMode`: a window's
  stages, one a round, with a unit's mode `Accounting.TS.Mode` (paid; folded, its product skipped and its
  committed words folded; or whole, product and fold skipped), each stage's tiles' deficient words right or not
  (`right`) and their checked values honest or not (`same`, which a receipt-lucky skipped tile meets); and the modes as
  the window's proofs read them, which `DrawnLayer.Good` and `UnitFixed` take.
* `JointRun`, `JointRun.share`, `SplitStages.joint`, `jointMode`: a window's modes and checked values as functions of
  every round's coin at once, which the seeded statements take (a cheater may choose a round's skips after querying
  the oracle for later rounds' seeds); a split-stage window is one, and the proofs read its modes as `tileMode` does.
* `tileGame`, `PerTile`: a tile's stage as the ideal game split at the receipt (`ReceiptGame`), in each skipped mode,
  and a claim of every credited tile at its stage, whatever the past: what `OperandHMu` and `ReceiptLuckAt` and
  `Assumptions.Compose.GuessLuckAt` are taken of.
* `jointGame`, `serviceGame`, `drawOf`: the window as one round on coins `A × (Equiv.Perm (Fin N) × Ξ)`: the rounds'
  coins, then the service's one uniform `K`-subset (`Pouw.Window.subsetDraw`, its `Equiv.Perm`) and the proofs' coins
  `Ξ`. Its share is `a·(folded weight) + b·(whole weight)` at the weights it is given, and it passes when every drawn
  tile's checked value is honest; `serviceGame` is it at a split-stage window's joint run.
* `Declared.slots`, `ProofRun`, `Declared.layer`: the window's proofs as an `Accounting.TS.DrawnLayer` over the IR's
  gates as positions, its static half defined from the IR (each tile's slots its unit's gates and their operands, each
  position's writer the unit of its gate, each position's cut its IR cut or the service's commit coin, whichever is
  later, each tile's coin its round's, the draw the
  service's) and only its run-dependent half (`ProofRun`: what was accepted and opened, each unit's relation and
  rightness) abstract.
* `StageCuts`, `StageCoins`, `StageSlots`, `StageComplete`: what the derived layer meets (`Declared.layer`'s lemmas in
  `Proofs.Accounting.Compose`): each value the service commits by its IR cut bound there (every one at
  `Declared.atIR`), each tile's noise at its round's coin (W2), its slots and writers inside the IR, and every
  one of the IR's there (every operand a slot, every written position a writer), so under `LayerCommitted` the
  layer's late cones are the IR's.
* `SplitRateAt`, `SplitRate`: the per-stage condition at given `ν, ν₀`, and for some, from which a rate `Λ` bounds the
  window's saving per unit of detection.
* `Declared.Live`, `Scope`, `rOf`, `WorkAt`: every round holding a credited tile live, the logical parameters' ranges
  and the service's commit order (`LayerCommitted`), the logical layer's bound `Λ·ln(1/δ_s) + min(a, b)·D₀` with `D₀`
  the cone weight on tiles of no words, and its conclusion on an accepted run: all but a share `d` of the committed
  words fixed right and all but `d₀ ≤ d` folded, with `a·(d − d₀) + b·d₀ < r`.
* `Declared.Seeded`, `SeedNamed`, `ScopeSeeded`, `setAt`, `redraw`, `tileRound`, `roundGain`, `roundWords`,
  `roundRate`, `roundCone`, `seedTheta`, `wholeTheta`, `roundPremium`, `premiumOf`, `rOfSeeded`: the seeded
  statements' scope (every round holding a credited tile seeded, one call site a round, so a round is a seed group,
  and every value a seeded round reads named by its seed's preimage or registered before the salt,
  which today's code doesn't meet), one oracle answer of the window redrawn, a tile's round, a round's gain at an
  exchange rate on the whole window's coins (GRIND's `L − κ·U`), each seed group's words, folded and whole rates,
  weight and grinding premium `π* = ln(T_q/δ_g)/θ` from the IR, and the seeded logical bound
  `max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K) + a·Π_F + b·Π_0 + min(a, b)·D₀`, the whole skips' premium `Π_0` at their own
  rate.

The physical layer:

* `creditOf`, `epsFlat`, `epsMu`, `aOf`, `bOf`, `epsOf`: `T_cred = B(♭)/f_max`, NOTATION's `ε_♭` and `ε_μ` at the
  circuit's floor, the time a folded and a whole share of `1` cost (`κ`, and `κ + (1 − μ_f)·N_w·φ*/B_♭`), and
  `ε = ε_♭ + ε_μ + r` at the logical layer's bound `r`.
* `Fits`, `ModelScope`: the model is the circuit's on the data sheet; the model's ranges.
* `Certificate`, `Certificate.Holds`, `certOf`: the interface the physical layer's ledger consumes:
  work of each kind in a device window (`Accounting.Window`, its ticks), its credit, `ε`, `δ`, `γ_♭ = T(H)/T_cred`
  (W1: `♭` reported, not pinned) and the coin mode `ε` is for (`live` from `CircuitCredit`, `seeded` from
  `CircuitCreditSeeded`), so the issuer of composed certificates, which doesn't exist yet, can refuse the live `ε`
  for a seeded window. The ledger, not this statement, decides how much of a device's time the certificates it
  admits may claim.
-/

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

/-! ## The declared circuit: its rounds, its proof units and its words -/

/-- **A round's coin mode**: `live`, a verifier's coin released after the round's served values are committed (a
per-round coin, or a per-stage prefill coin), or `seeded`, derived from the window's salt and commitments
by a hash (Fiat–Shamir, `fiat-shamir`), as PoUW's code derives every noise today. -/
inductive CoinMode
  | live
  | seeded
  deriving DecidableEq, Repr

/-- **A declared circuit**: its gates in the IR's canonical order (`Accounting.Gates.IntDag`: each gate's operation,
operands, interval, primitive, place and whether it is committed), and as the IR binds them:
* `call i`: the Definition call site, of `calls`, that made gate `i`, numbered in the circuit's order;
* `round c`: call site `c`'s round, of `rounds`: its noise reads that round's coin, released after the round's operands
  are committed (one coin a round; one salt for the window, COMPILE);
* `coinMode r`: round `r`'s coin's mode, live or seeded (`CoinMode`). The live statements need every round that holds a
  credited tile live (`Declared.Live`, in `Scope`); the seeded statements need every one seeded, one call site a round
  (`Declared.Seeded`, in `ScopeSeeded`);
* `commitRound ℓ`: the coin, of the window's `rounds` (`rounds`: the service's draw), before which the service commits
  value `ℓ` under a root that every opening of it is checked against (live: a round's root, before that round's coin).
  The proofs' layer cuts `ℓ` there or at its IR cut, whichever is later (`Declared.layer`). The live statements take
  it no later than the IR cut, or the draw for a computed value only its own unit reads (`LayerCommitted`, in
  `Scope`); the seeded statements read the circuit at the draw
  (`Declared.atDraw`) for binding and at the IR (`Declared.atIR`) for cones, so they read no `commitRound`;
* `seedNames r ℓ`: whether round `r`'s seed preimage names a root that value `ℓ` is a leaf under, every opening of it
  checked against that root (`SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B))`, whose salt names
  the weights' root). The seeded statements need every value a seeded round reads named (`SeedNamed`, in
  `ScopeSeeded`); the live ones read none of it;
* `unit i`: gate `i`'s proof unit, of `units`, the unit the service draws and proves alone (an output
  coordinate, or a K-chunk committing its running sum, which no served kind is cut into; glue between
  products is in units of its own). -/
structure Declared (ι : Type) where
  dag : Gates.IntDag ι
  calls : ℕ
  call : ℕ → ℕ
  rounds : ℕ
  round : ℕ → ℕ
  coinMode : ℕ → CoinMode
  commitRound : ℕ → ℕ
  seedNames : ℕ → ℕ → Bool
  units : ℕ
  unit : ℕ → ℕ

/-- **Gate `i`'s input binding**: `some b` for an input bound at `b` (`Gates.Binding`: registered, a coin or at run
time), and `none` for any other gate or past the circuit. -/
def Declared.binding {ι : Type} (Cd : Declared ι) (i : ℕ) : Option Gates.Binding :=
  (Cd.dag[i]?).bind fun g => match g.op with
    | .input b _ => some b
    | _ => none

/-- **Gate `j` reads gate `i`**: `i` is one of `j`'s operands. -/
def Declared.Reads {ι : Type} (Cd : Declared ι) (j i : ℕ) : Prop :=
  (Cd.dag[j]?).any (fun g => g.args.contains i) = true

/-- **Gate `ℓ` is a value the prover writes**: a gate of the circuit other than a coin input, which is the verifier's
(or the seed's) and no unit writes. -/
def Declared.Writes {ι : Type} (Cd : Declared ι) (ℓ : ℕ) : Prop :=
  ℓ < Cd.dag.length ∧ Cd.binding ℓ ≠ some .coin

/-- Whether each gate's value depends on a run input, in order: a `run` input, or a gate that reads such a value.
Unlike `Gates.dynamic` it leaves coins out, so it tells an activation (a request's tokens and what they reach) from a
weight or a codec's parameters. -/
def runDep {ι : Type} (C : Gates.IntDag ι) : List Bool :=
  C.foldl (fun ds g => ds ++ [match g.op with
    | .input b _ => b == .run
    | _ => g.args.any fun j => ds.getD j false]) []

/-- **A committed word**: a committed gate on a served product's main chain, its word `T_t` (S8-PORT's COMMIT layout:
the final `T_1`, or the prefix `T_1 … T_{K/L}`). -/
def isWord {ι : Type} (g : Gates.IntGate ι) : Bool := g.commit && g.role.isServed

/-- The circuit's committed words, as gate indices in the circuit's order. -/
def wordGates {ι : Type} (Cd : Declared ι) : List ℕ :=
  (List.range Cd.dag.length).filter fun i => Cd.dag[i]?.any isWord

/-- **A product's kind**, read off the IR: its primitive's id, its place's statics (the step zeroed: `K`, `L` and the
tail groups, what `Gates.roleOf` reads) and whether its `a` and `b` operands depend on a run input. -/
structure Kind where
  prim : String
  place : Gates.Place
  actA : Bool
  actB : Bool
  deriving DecidableEq

/-- A place's statics: an `S8Dot_v1` body's `K`, `L` and tail groups, its step zeroed; an H1 + fx1 tile's receipt
word's dot kind and `K`, its word and operand rows zeroed. -/
def statics : Gates.Place → Gates.Place
  | .dot K L tail _ => .dot K L tail 0
  | .receipt d K _ _ _ => .receipt d K 0 [] []
  | .other => .other

/-- An s8-family step's `k`, and `0` for any other operation. -/
def mmaK {ι : Type} : Gates.IntOp ι → ℕ
  | .mma _ _ k _ => k
  | _ => 0

/-- The gates of gate `g`'s `a` and `b` operands: a receipt word's tile's operand rows (`Gates.Place.receipt`), and
otherwise `args[1 … k]` and `args[k+1 …]`. -/
def operandGates {ι : Type} (g : Gates.IntGate ι) : List ℕ × List ℕ :=
  match g.place with
  | .receipt _ _ _ a b => (a, b)
  | _ => (g.args.tail.take (mmaK g.op), g.args.tail.drop (mmaK g.op))

/-- Gate `g`'s kind in circuit `C`: its primitive, its place's statics, and whether any of its `a` operands and any
of its `b` operands (`operandGates`) depends on a run input. -/
def kindOfGate {ι : Type} (C : Gates.IntDag ι) (g : Gates.IntGate ι) : Kind :=
  ⟨g.prim, statics g.place, (operandGates g).1.any fun j => (runDep C).getD j false,
    (operandGates g).2.any fun j => (runDep C).getD j false⟩

/-- **Gate `i`'s kind** (`kindOfGate`), and an empty one past the circuit's gates. -/
def kindAt {ι : Type} (Cd : Declared ι) (i : ℕ) : Kind :=
  (Cd.dag[i]?).elim ⟨"", .other, false, false⟩ (kindOfGate Cd.dag)

/-- **The circuit is well formed**: every gate's call site and unit are in range, every call site's round is, one
unit's gates are in one round, each coin input's round is the round of every gate that reads it (so of the unit that
reads it: a round's noise reads that round's coin, which `coinMark` and `StageCoins` read), and one unit's words are of
one kind. -/
def Declared.Wf {ι : Type} (Cd : Declared ι) : Prop :=
  (∀ i, i < Cd.dag.length → Cd.call i < Cd.calls ∧ Cd.unit i < Cd.units) ∧
    (∀ c, c < Cd.calls → Cd.round c < Cd.rounds) ∧
    (∀ i j, i < Cd.dag.length → j < Cd.dag.length → Cd.unit i = Cd.unit j →
      Cd.round (Cd.call i) = Cd.round (Cd.call j)) ∧
    (∀ i j, i < Cd.dag.length → j < Cd.dag.length → Cd.binding i = some .coin → Cd.Reads j i →
      Cd.round (Cd.call i) = Cd.round (Cd.call j)) ∧
    ∀ i ∈ wordGates Cd, ∀ j ∈ wordGates Cd, Cd.unit i = Cd.unit j → kindAt Cd i = kindAt Cd j

/-- Unit `u`'s gates, in the circuit's order. -/
def unitGates {ι : Type} (Cd : Declared ι) (u : ℕ) : List ℕ :=
  (List.range Cd.dag.length).filter fun i => Cd.unit i == u

/-- Unit `u`'s committed words, in the circuit's order. -/
def unitWords {ι : Type} (Cd : Declared ι) (u : ℕ) : List ℕ :=
  (wordGates Cd).filter fun i => Cd.unit i == u

/-- **Unit `u`'s round**: its gates' (one, `Declared.Wf`), and `0` for a unit of no gates. -/
def Declared.roundOf {ι : Type} (Cd : Declared ι) (u : ℕ) : ℕ :=
  (unitGates Cd u).head?.elim 0 fun i => Cd.round (Cd.call i)

/-- **Unit `u`'s stage**: the rounds after its own, so the circuit's last round is stage `0`, as `SplitStages`
counts. -/
def Declared.stageOf {ι : Type} (Cd : Declared ι) (u : ℕ) : ℕ := Cd.rounds - 1 - Cd.roundOf u

/-- **Unit `u`'s kind**: its words' (one, `Declared.Wf`), and none for a unit of no words (glue). -/
def Declared.kindOf {ι : Type} (Cd : Declared ι) (u : ℕ) : Option Kind := (unitWords Cd u).head?.map (kindAt Cd)

/-! ## Cuts and late cones, from the IR -/

/-- **The coins each gate's value depends on**, in order: one past the round of the latest coin input it reads,
transitively (a coin input of a call site is its round's coin), and `0` for a value that reads no coin. -/
def coinMark {ι : Type} (Cd : Declared ι) : List ℕ :=
  Cd.dag.foldl (fun ms g => ms ++ [match g.op with
    | .input .coin _ => Cd.round (Cd.call ms.length) + 1
    | _ => (g.args.map fun j => ms.getD j 0).foldl max 0]) []

/-- The round of the first gate that reads gate `i`, and `rounds` (the draw) when none does. -/
def Declared.firstRead {ι : Type} (Cd : Declared ι) (i : ℕ) : ℕ :=
  ((List.range Cd.dag.length).find? fun j => (Cd.dag[j]?).any fun g => g.args.contains i).elim Cd.rounds
    fun j => Cd.round (Cd.call j)

/-- **Gate `i`'s cut**, from the IR (W1): the coin before which its value is committed, as an index of the window's
`rounds` coins (the draw is `rounds`). With one coin a round, a round's coin is released after the round's served values
are committed, and its noised products are computed after it:
* a registered input (a weight, a codec's parameters) is bound before the first coin, `0`;
* a run input (a request's tokens, decode feedback) is bound after the coins it follows, so it may be any function of
  the coins released before its cut, which is the coin of the first round that reads it: the service commits it under
  that round's root, before its coin (`LayerCommitted`, at `Declared.commitRound`);
* any other value (a product's served output, glue, the KV cache) is committed with its writing gate's round, before
  that round's coin, or after it when it depends on it (`coinMark`: a noised word, a receipt, glue fed by one); one
  that only its own unit reads (a receipt) the service may commit later, by the draw (`LayerCommitted`). -/
def Declared.cutOf {ι : Type} (Cd : Declared ι) (i : ℕ) : ℕ :=
  match Cd.binding i with
  | some .registered => 0
  | some .run => Cd.firstRead i
  | _ => max (Cd.round (Cd.call i)) ((coinMark Cd).getD i 0)

/-- **Gate `j` reads gate `i` after its own coin**: `i` is one of `j`'s operands, and `i`'s cut is past the coin of
`j`'s round, so `i`'s value can be chosen after `j`'s noise is drawn: a noised word read in its own round (a K-chunk's
running sum), or glue fed by one (a fused call site, W4). A value committed before `j`'s coin (a served value of `j`'s
round or an earlier one, a registered or run input) is fixed when `j`'s noise is drawn, whether it is right or not, so
it is no edge. -/
def Declared.Late {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop :=
  Cd.Reads j i ∧ Cd.round (Cd.call j) < Cd.cutOf i

/-- **Unit `u` writes unit `v` a value after `v`'s coin**: a gate of `v` reads a value of `u`'s late (`Declared.Late`,
`Declared.Writes`: a coin input is no unit's value, so it is no edge). -/
def Declared.UnitLate {ι : Type} (Cd : Declared ι) (u v : ℕ) : Prop :=
  ∃ i j, Cd.Writes i ∧ Cd.unit i = u ∧ Cd.unit j = v ∧ Cd.Late i j

/-- **Unit `u'` is in unit `u`'s late cone**: `u` itself, and every unit reached from it along late
writes (`Declared.UnitLate`, transitively). It is closed per unit, as `Accounting.TS.DrawnLayer.Late` is: a skipped unit
can leave every gate of it wrong, so a late read of any one of them reaches the reader's whole unit. -/
def Declared.Reaches {ι : Type} (Cd : Declared ι) : ℕ → ℕ → Prop := Relation.ReflTransGen Cd.UnitLate

/-- **The service commits by the IR's cuts** (protocol structure, which today's code doesn't meet).
Every value `ℓ` the prover writes is committed under a root, which every opening of it is checked against,
before coin `Declared.commitRound ℓ` of the window's `rounds`, and that coin is no later than its IR cut
(`Declared.cutOf ℓ`), or, for a computed value (no input) that only gates of its own unit read, no later than the draw
(`rounds`). The proofs' layer cuts `ℓ` at the later of the two (`Declared.layer`), so a value of the first case has its
IR cut there (`StageCuts`, `Proofs.Accounting.Compose.layer_stageCuts`), and the live statements take it in `Scope`.

The second case is the ring of depth `D` (a receipt of step `s` under the root of round `s + D − 1`, so its off-node
live coins, about 100 ms a round trip, are hidden behind `D` steps: about 104 ms a token at `D = 2`, 0.09 ms at
`D = 6`). A value read only inside its own unit is in that unit's late cone and in no other unit's:
a layer late edge it makes ends in its own unit, which that unit's cone already holds (`reaches_of_slot`,
`Proofs.Accounting.Compose.ServiceCovers`), so committing it late leaves no reader whose skip goes uncharged, and the
draw, at coin `rounds`, still follows every commitment. It is a computed value, its gates' function of what its unit
reads, which the unit's proof checks when the unit is drawn and whose skip its cone pays for when it isn't. An input
gets no such case: a run input committed after its reader's coin lets the prover choose
it against the reader's noise (`A = −b` cancels a one-bit dither), even when only its own unit reads it.

It is a fact of the declared circuit, not a premise about the cheater. A service that commits later than it allows
fails it, and then `Scope` fails and the live statements credit nothing. A `Declared` whose `commitRound` claims an
earlier coin than the code commits at meets it, and then its layer cuts a value before the code binds it: an opening
moved after that cut is `DrawnLayer.MovedIn`, so the failure lands in the binding premise (`MovedIn ≤ δ_bind`), which
is false at any small `δ_bind` against a cheater that moves it. Neither case is `ServiceCovers`'s: it takes this as a
hypothesis. It is also the order the binding card needs: `δ_bind` bounds an opening that differs from a root fixed
before the cut, and nothing for a value committed after it; the layer cuts a value of the second case at its commit
coin, after its root.

On the live path the order is: the registered inputs (weights, a codec's parameters) before the first coin; each
round's served values, and each run input (a request's tokens, decode feedback) under the root of its first reader's
round, before that round's coin; the values that read the coin and that other units read (a noised word another
unit's gate reads, glue fed by one) before the next round's; and a receipt, which only its own unit reads, under any
later round's root up to the draw's. A run input committed after its reader's coin lets the prover choose tokens or
decode feedback to match the reader's noise: that breaks (H-μ)'s salt-independence (E3) for the reader.

Today's code has a coin a round, and nothing shows that it meets this order. A live window's coins are its salt coin, its
draw coin and one round coin a round (`verity/core/protocols/pouw/window.py`'s `live_families`). Round r's coin opens
once, after the round root the challenger's signed log shows it received (`core_check.rounds`), and round r's noise key
is that coin's and that root's (`window.round_key`). Each round's root commits that round's rows of every A value
before its coin (`credit.a_link`). A seeded window's coins are still one salt a window and the draw
(`verity/core/protocols/pouw/coins.py`'s `LABELS = ("salt", "draw")`), with no per-round coin or root, and the salt opens
before the activations are committed. No code checks this order itself: that every value the served
Program writes is committed under the root of the round its `commitRound` names. A served Program's commit coins reach
`Declared` only through `Properties.Compose.UnitsOfLayout`, which is unproved. The seeded statements don't read it: they
read the circuit at the draw (`Declared.atDraw`) and at the IR (`Declared.atIR`, which meets it), and take its seeded
mirror, `SeedNamed`. -/
def LayerCommitted {ι : Type} (Cd : Declared ι) : Prop :=
  ∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ ≤ Cd.cutOf ℓ ∨
    (Cd.binding ℓ = none ∧ Cd.commitRound ℓ ≤ Cd.rounds ∧ ∀ j, Cd.Reads j ℓ → Cd.unit j = Cd.unit ℓ)

/-- **The circuit committed at the draw**: every value's commit coin the service's live draw (`rounds`), so
`Declared.layer` cuts every position there, after every round's coin. The seeded statements check binding on it
(`DrawnLayer.MovedIn`, `Of`, `JointSound`): under seeded coins a round's coin is the oracle's answer at a preimage the
cheater chooses, so it can query any round's seed, then choose what it commits, and nothing it writes is bound before
the draw. -/
def Declared.atDraw {ι : Type} (Cd : Declared ι) : Declared ι := { Cd with commitRound := fun _ => Cd.rounds }

/-- **The circuit cut at the IR**: every value's commit coin `0`, so `Declared.layer` cuts each position at its IR cut
(`Declared.cutOf`), and `LayerCommitted` holds. The seeded statements count cones on it (`UnitFixed`,
`ServiceCovers`): its late edges are the IR's, and neither reads a committed value, only cuts, slots, writers and
the run's modes. Its cuts are the seeded protocol's commitments only where each seeded round's seed names what the
round reads (`SeedNamed`, in `ScopeSeeded`). -/
def Declared.atIR {ι : Type} (Cd : Declared ι) : Declared ι := { Cd with commitRound := fun _ => 0 }

/-- **Every value a seeded round reads by its IR cut is named by the round's seed** (protocol structure, which today's
code doesn't meet; the mirror of `LayerCommitted`). Each value `ℓ` the prover writes that
a gate `j` of a seeded round reads, its IR cut at or before that round (not `Declared.Late`), is a registered input,
committed before the salt (whose derivation names the weights' root, `PouwCommit.saltOf`), or a leaf under a root the
round's seed preimage names (`Declared.seedNames`). It is every value read, not only a product's operands: run inputs
(a request's tokens, decode feedback), earlier rounds' outputs and the round's own glue count.

The seeded statements count cones on the IR's cuts (`Declared.atIR`), and `UnitFixed` there is E1(c): a value
committed before a unit's coin is fixed when its noise is drawn. Under seeded coins a round's coin is the oracle's
answer at its preimage, so a value is committed before it only when the preimage names it: re-chosen after the seed,
it changes the query, a new try that `premiumOf` prices, and for each query it is fixed. A value no preimage names can
be chosen after the seed, to make a paid unit's checked words dead against its noise, and then `UnitFixed` on the
IR's layer is false and the statements credit nothing. This puts that condition in `ScopeSeeded`, where a reader sees
it, as `Scope` carries `LayerCommitted`.

Today's code doesn't meet it: each unit's seed is `seeds.PouwSeed` of the salt, its position and its seed group's
`pouw.roots` slot, a Program input that nothing ties to the unit's operand rows, and no seed names a run input or an
earlier round's output, so only registered inputs are named (`Properties.Compose.UnitsOfLayout`'s seed conjunct). -/
def SeedNamed {ι : Type} (Cd : Declared ι) : Prop :=
  ∀ ℓ j, Cd.Writes ℓ → Cd.Reads j ℓ → Cd.coinMode (Cd.round (Cd.call j)) = .seeded → ¬ Cd.Late ℓ j →
    Cd.binding ℓ = some .registered ∨ Cd.seedNames (Cd.round (Cd.call j)) ℓ = true

/-! ## Schemes: floors and luck per kind -/

/-- **The form of a template's proved operand luck**, at a skip pattern's word span:
* `cube bind`: `¾` while the span is within the receipt's binding `bind` (`Accounting.Guarantees.TS.LuckCaseACube`,
  `OperandLuckCube`, for degree-2 cube operands; `Pouw.Guarantees.Receipt.DuoWindowNoKernel` binds four
  words), and `1` past it (`DuoLongRunKernel`: no receipt binds every length);
* `coins b Z`: `Z(span)·C(b, ⌊b/2⌋)/2^b` for `b` fresh bits at the fibre `Z(span)` the receipt has at that span
  (`Accounting.Guarantees.HMu.CoinPointMass`, `ReceiptLuckOfCoins`);
* `pinned`: `0`, a value the proof pins (`Accounting.Guarantees.FillerLuck.PinnedFillLuckAt`);
* `unproved`: `1`. -/
inductive LuckForm
  | cube (bind : ℕ)
  | coins (b : ℕ) (Z : ℕ → ℕ)
  | pinned
  | unproved

/-- A luck form's value at a skip pattern's word span. -/
noncomputable def LuckForm.value : LuckForm → ℕ → ℝ
  | .cube bind, span => if span ≤ bind then 3 / 4 else 1
  | .coins b Z, span => (Z span : ℝ) * ((b.choose (b / 2) : ℝ) / 2 ^ b)
  | .pinned, _ => 0
  | .unproved, _ => 1

/-- **A scheme's templates**, per kind: one committed word's floor `♭_k` over the classes `K`, proved at the worst
operands (`0` where the noise proves none, as one-sided noise proves none), whether the kind is credited
(a word of an uncredited kind counts in no `N_w` and no floor), the span a skip pattern has before the receipt's next
checkpoint (W6), the form of the operand luck its template proves, and the three lucks the window is held to:
* `mu`: (H-μ)'s point mass at a folded unit's deficient words (`OperandHMu`, `¾` in the headlines);
* `rB`: the receipt's luck when they are not all right (`ReceiptLuckAt`, HMU's B′, `½`);
* `guess`: a unit skipped whole keeps its honest checked value only by a guess of its fold state
  (`Assumptions.Compose.GuessLuckAt`, a condition at this value: `½`, B′'s, for the live line as ruled,
  where `2^-64` for a 64-bit state was proposed and isn't granted);
* `seedRate`: the rate `θ_k`, per committed word of the circuit, at which a seed group of kind `k`'s template gain at
  `κ = 4` has moment at most `1` over its seed (GRIND's `Accounting.Guarantees.TS.TemplateMoment`: `1/(256·r)` per
  product, times a word's products), and `0` where none is proved, which charges a group its whole weight;
* `wholeRate`: the rate `θ₀_k`, per committed word, at which a seed group of kind `k`'s whole-skip gain at `κ₀` has
  moment at most `1` over its seed. From the guess bound it is `x₀/w_k`, `x₀` the largest per-tile rate with
  `guess·e^{x₀} + (1 − guess)·e^{−κ₀·x₀} ≤ 1` (`44.34` nats a tile at `2^-64`, `κ₀ = 0.092`; a premium of about `2.8`
  tiles a seed group at `T_q = 2^128`, whatever `K` is; at `guess = ½` it needs `κ₀ > 1` and is below `ln 2`, about
  `177` tiles a seed group), and `0` where none is proved. No ruling grants a whole-tile luck per seed query, so a
  value here is a condition the seeded statements' numbers are labelled by. It is not `seedRate`: at the
  folded rate a tile's whole-skip moment is `guess·e^{K/8}` (`e^{467}` at `K = 4096`, past `1` from `K = 355`), so
  `seedRate` may stand here only under `Assumptions.Compose.WholeEntropyAt` at `K/8` nats a tile.
  The statements don't check the value: a rate above what the luck supports makes `SeedLuckAt` false, and
  nothing is credited.
Under live coins H1 + fx1's attention matmuls are credited at their floors: `QKᵀ` draws a
coin per entry on both operands; `P·V`'s `P` carries only the period-32 word `U` and `V` a coin `b` per entry, and its
floor is `U·b`'s, a cell per `k`, which holds under any salt-independent advice while the kernel commits a prefix word
at most every 32 keys (at one every `C` keys it is `32·k/C`). Under seeded coins they are uncredited
(`Scheme.SeedPriced`, in `ScopeSeeded`): GRIND prices no premium for a kind reading two activations, and at a head's
seed group it would be 37.5% of `QKᵀ` and 75% of `P·V`. -/
structure Scheme (K : Type) where
  floor : Kind → Labelled K
  credited : Kind → Bool
  span : Kind → ℕ
  form : Kind → LuckForm
  mu : Kind → ℝ
  rB : Kind → ℝ
  guess : Kind → ℝ
  seedRate : Kind → ℝ
  wholeRate : Kind → ℝ

/-- **The luck is at least what each template proves, and in range**: per kind, (H-μ)'s point mass is at least its
form's value at the kind's span, so `OperandHMu` asks no more of the cheater's view than the template's own coins give,
the receipt's luck lies in `[0, 1]` (`LuckOfReceipt` needs `r_B ≤ 1`) and the guess bound is `≥ 0`. -/
def Scheme.LuckProved {K : Type} (Sc : Scheme K) : Prop :=
  ∀ k, (Sc.form k).value (Sc.span k) ≤ Sc.mu k ∧ 0 ≤ Sc.rB k ∧ Sc.rB k ≤ 1 ∧ 0 ≤ Sc.guess k

/-- **Every credited kind's seeded premium is priced**: no credited kind reads two activations (`Kind.actA` and
`Kind.actB`), so H1 + fx1's `QKᵀ` and `P·V` are uncredited under seeded coins, where GRIND prices no premium for them.
The seeded statements take it (`ScopeSeeded`); a scheme for a seeded window sets `credited` false on those kinds. -/
def Scheme.SeedPriced {K : Type} (Sc : Scheme K) : Prop :=
  ∀ k, Sc.credited k = true → ¬(k.actA = true ∧ k.actB = true)

/-- **A folded unit's luck**: `μ + (1 − μ)·r_B`, its deficient words right at `μ` or, when not, its receipt kept at
`r_B` (`Accounting.Guarantees.TS.LuckOfReceipt`): `7/8` at `μ = ¾`, `r_B = ½`. -/
noncomputable def Scheme.foldLuck {K : Type} (Sc : Scheme K) (k : Kind) : ℝ := Sc.mu k + (1 - Sc.mu k) * Sc.rB k

/-- The circuit's credited words: its committed words of a credited kind, in order. -/
def creditedWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : List ℕ :=
  (wordGates Cd).filter fun i => Sc.credited (kindAt Cd i)

/-- **The circuit's committed words** `N_w`, read off the IR (W3): its credited words. -/
def wordsOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : ℕ := (creditedWords Sc Cd).length

/-- **The circuit's floor** `♭ = Σ_words ♭_{kind}`, over its credited words, part by part. -/
noncomputable def floorOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : Labelled K where
  nw k := ((creditedWords Sc Cd).map fun i => (Sc.floor (kindAt Cd i)).nw k).sum
  nf k := ((creditedWords Sc Cd).map fun i => (Sc.floor (kindAt Cd i)).nf k).sum
  nx k := ((creditedWords Sc Cd).map fun i => (Sc.floor (kindAt Cd i)).nx k).sum
  ns k := ((creditedWords Sc Cd).map fun i => (Sc.floor (kindAt Cd i)).ns k).sum

/-- The circuit's committed words of kind `k`: the work of that kind a certificate reports. -/
def kindCount {ι : Type} (Cd : Declared ι) (k : Kind) : ℕ :=
  ((wordGates Cd).filter fun i => decide (kindAt Cd i = k)).length

/-! ## The service's grid-padded tiles -/

/-- Tile `t`'s committed words: unit `t`'s credited words, and `0` for glue, a unit of an uncredited kind or a filler
(`t ≥ units`: `window.py` pads the draw to one grid level with fillers). -/
def tileWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℕ :=
  ((unitWords Cd t.val).filter fun i => Sc.credited (kindAt Cd i)).length

/-- A tile's work weight: its words over the circuit's. -/
noncomputable def tileWeight {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (tileWords Sc Cd Np t : ℝ) / (wordsOf Sc Cd : ℝ)

/-- A tile's stage: its unit's. -/
def tileStage {ι : Type} (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℕ := Cd.stageOf t.val

/-- A tile's kind: its unit's words', none for glue or a filler. -/
def tileKind {ι : Type} (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : Option Kind := Cd.kindOf t.val

/-- A tile's luck when folded: its kind's `foldLuck` when credited, and `1` otherwise. -/
noncomputable def tileLuck {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (tileKind Cd Np t).elim 1 fun k => if Sc.credited k then Sc.foldLuck k else 1

/-- A tile's luck when whole: its kind's guess bound when credited, and `1` otherwise. -/
noncomputable def tileLuck0 {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (tileKind Cd Np t).elim 1 fun k => if Sc.credited k then Sc.guess k else 1

open Classical in
/-- **A tile's late cone's weight**, from the IR: the weights of the tiles its unit reaches along late
writes (`Declared.Reaches`), itself included. A skipped tile can leave every word of its cone adaptive, and no other;
the service's share charges each skipped tile this (`serviceGame` in `CircuitWork`). -/
noncomputable def tileCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  ∑ t' : Fin Np, if Cd.Reaches t.val t'.val then tileWeight Sc Cd Np t' else 0

/-- **A tile's detected weight**: its late cone's weight when it carries words, and `0` for a tile of no words, whose
skip no luck bounds. The service's rate is computed at these weights (`CircuitWork`'s `ρ`, stage share and luck
premises), and the cones of tiles of no words, which no draw detects, are `zeroCone`. -/
noncomputable def shareCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  if tileWords Sc Cd Np t = 0 then 0 else tileCone Sc Cd Np t

/-- **`D₀`, the cone weight on tiles of no words** (W4): glue and uncredited units whose late cones carry work. A
cheater may skip them all at no risk, so the logical layer's bound adds it at the smaller cost (`rOf`: `WorkAt` lets
the excess sit in either share). It is computed, not assumed, and it is `0` for every Program COMPILE accepts: the salt
reaches only the units' seeds (`salt-flow`), no noise reaches `z`, and the receipt is a sink only the Program's outputs
read, so no glue's value depends on a coin and no unit of no words writes another late. It is nonzero only for a
circuit COMPILE refuses, such as a hypothetical fused chain whose glue reads a noised word: there one kind's late cones
of `F` tiles put `(F − 1)/2` times that kind's share on glue, `D₀ = 1.5%` at `F = 4` and a `1%` share, charged `0.33%`
at `a = 0.218`. -/
noncomputable def zeroCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) : ℝ :=
  ∑ t : Fin Np, if tileWords Sc Cd Np t = 0 then tileCone Sc Cd Np t else 0

open Classical in
/-- **`F` per kind**: the most tiles one late cone of a tile of kind `k` holds, from the IR. It is `1`
for every kind served: its units hold whole K and `z` reads no noise, so no unit reaches another late. Only
a coordinate cut into K-chunks, which COMPILE doesn't make, would have `F > 1` (its running sums late within their
round). At a tile weight bound `w_max` per tile, `ρ = 1/(F·r)`, `F` the largest `F_k` and `r = (N_p/N)·(w_max/w̄)`,
meets `CircuitWork`'s `ρ·N_p·shareCone ≤ 1`. -/
noncomputable def kindF {ι : Type} (Cd : Declared ι) (Np : ℕ) (k : Kind) : ℕ :=
  (Finset.univ.filter fun t : Fin Np => tileKind Cd Np t = some k).sup fun t =>
    (Finset.univ.filter fun t' : Fin Np => Cd.Reaches t.val t'.val).card

/-! ## The window: stages, one a round, with B1's split, and the service's draw -/

/-- **A window's stages, one a round**: per stage, each unit's mode (`Accounting.TS.Mode`: paid, folded or whole),
whether its deficient words are all right (`right`) and whether its checked values (the values its proof pins: a warp
checkpoint, a fold state) are the honest ones (`same`), as functions of the stage's coin `C` (the stage's outputs are
committed after it is released), then the rest of the window, which may depend on it. A unit belongs to one stage,
`stg u`, counted from the last (stage `0`). A skipped unit whose receipt is lucky is `same` with wrong words. -/
inductive SplitStages (C V : Type) : ℕ → Type
  | done : SplitStages C V 0
  | stage {m : ℕ} (mode : C → V → Mode) (right same : C → V → Prop) (next : C → SplitStages C V m) :
      SplitStages C V (m + 1)

namespace SplitStages

variable {C V : Type}

/-- Unit `u`'s mode along the stages' coins: its stage's, and paid when no stage holds it. -/
def modeAt (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Paths C m → V → Mode
  | _, .done, _, _ => .paid
  | _, .stage (m := m) mode _ _ next, x, u => if stg u = m then mode x.1 u else modeAt stg (next x.1) x.2 u

/-- Unit `u`'s checked values are honest along the stages' coins: its stage says so. -/
def sameAt (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Paths C m → V → Prop
  | _, .done, _, _ => True
  | _, .stage (m := m) _ _ same next, x, u => if stg u = m then same x.1 u else sameAt stg (next x.1) x.2 u

/-- **The luck per unit at its stage's coin**: at every stage, whatever the past, a unit of the stage that is folded
keeps its honest checked values with probability at most `p u` times that of being folded, and one that is whole with
`p0 u`. -/
def LuckAt [Fintype C] (p p0 : V → ℝ) (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Prop
  | _, .done => True
  | _, .stage (m := m) mode _ same next =>
      (∀ u, stg u = m →
        prob (fun c => mode c u = .folded ∧ same c u) ≤ p u * prob (fun c => mode c u = .folded) ∧
          prob (fun c => mode c u = .whole ∧ same c u) ≤ p0 u * prob (fun c => mode c u = .whole)) ∧
        ∀ c, LuckAt p p0 stg (next c)

/-- **`P` at every stage, whatever the past**: at each stage `j` of the tree, of its modes, rightness and checked
values. -/
def EveryStage (P : ℕ → (C → V → Mode) → (C → V → Prop) → (C → V → Prop) → Prop) :
    {m : ℕ} → SplitStages C V m → Prop
  | _, .done => True
  | _, .stage (m := m) mode right same next => P m mode right same ∧ ∀ c, EveryStage P (next c)

end SplitStages

/-- **A window's run on its rounds' coins jointly**: each tile's mode and whether its checked values are honest, as
functions of every round's coin `A` at once. A split-stage window is one (`SplitStages.joint`), its stage `j`'s modes
reading only the coins up to it. The seeded statements take any: under seeded coins each round's coin is the oracle's
answer at a preimage the cheater chooses, so it may choose round `r`'s operands and skips after querying the oracle for
every later round's seed, and no stage order binds it. -/
structure JointRun (A V : Type) where
  mode : A → V → Mode
  same : A → V → Prop

/-- The work in mode `s` on the rounds' coins `a`, at work weights `w`. -/
noncomputable def JointRun.share {A V : Type} [Fintype V] (R : JointRun A V) (w : V → ℝ) (a : A) (s : Mode) : ℝ :=
  ∑ u, if R.mode a u = s then w u else 0

/-- **A split-stage window as a joint run**: each unit's mode and checked values its own stage's along the rounds'
coins (`SplitStages.modeAt`, `SplitStages.sameAt`). -/
def SplitStages.joint {C V : Type} {M : ℕ} (stg : V → ℕ) (W : SplitStages C V M) : JointRun (Paths C M) V :=
  ⟨W.modeAt stg, W.sameAt stg⟩

/-- **A joint run's modes as the window's proofs read them**: on a path of the window's one service round, tile `u`'s
mode at the rounds' coins. `Accounting.TS.DrawnLayer.Good` and `Accounting.Assumptions.TS.UnitFixed` read these. -/
def jointMode {A B : Type} {Np : ℕ} (R : JointRun A (Fin Np)) (x : Paths (A × B) 1) (u : Fin 1 × Fin Np) : Mode :=
  R.mode (coinAt x 0).1 u.2

/-- **The run's modes as the window's proofs read them**, for a split-stage window: tile `u`'s mode is its stage's
along the rounds' coins (`jointMode` of `SplitStages.joint`). -/
def tileMode {ι C B : Type} {Np : ℕ} (Cd : Declared ι) (W : SplitStages C (Fin Np) Cd.rounds)
    (x : Paths (Paths C Cd.rounds × B) 1) (u : Fin 1 × Fin Np) : Mode :=
  jointMode (W.joint (tileStage Cd Np)) x u

/-- **Tile `t`'s stage in mode `s`, split at the receipt** (`ReceiptGame`, one round on the stage's coin): its share is
`1` where the tile is in mode `s` and `0` elsewhere, it passes where its checked values are honest (`same`, so it is
`Pinned`), and its deficient words are all right where `right` says so. -/
noncomputable def tileGame {C V : Type} (mode : C → V → Mode) (right same : C → V → Prop) (s : Mode) (t : V) :
    ReceiptGame C 1 :=
  .round (fun c => if mode c t = s then 1 else 0) (fun c => same c t) (fun c => right c t) (fun c => same c t)
    fun _ => .done

/-- **A claim of every credited tile at its stage, whatever the past**: at each stage of the window, `P k G G₀` for each
tile of the stage of a credited kind `k`, `G` its stage folded and `G₀` whole (`tileGame`). -/
def PerTile {ι K C : Type} {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι) (W : SplitStages C (Fin Np) M)
    (P : Kind → ReceiptGame C 1 → ReceiptGame C 1 → Prop) : Prop :=
  W.EveryStage fun j mode right same => ∀ t : Fin Np, tileStage Cd Np t = j → ∀ k, tileKind Cd Np t = some k →
    Sc.credited k = true → P k (tileGame mode right same .folded t) (tileGame mode right same .whole t)

/-- Stage `j`'s work share at work weights `w`. -/
noncomputable def stageShare {V : Type} [Fintype V] (stg : V → ℕ) (w : V → ℝ) (j : ℕ) : ℝ :=
  ∑ v, if stg v = j then w v else 0

/-- **The window as one round of the service**: on the rounds' coins `A`, the permutation whose first `K` images are the
drawn tiles (`Pouw.Window.subsetDraw`) and the proofs' coins `Ξ`, its share is `a` per unit of folded work and `b` per
unit of whole work of the joint run `R`, and it passes when every drawn tile's checked values are honest. -/
noncomputable def jointGame {A Ξ : Type} {N : ℕ} (R : JointRun A (Fin N)) (w : Fin N → ℝ) (a b : ℝ) (K : ℕ) :
    RoundGame (A × (Equiv.Perm (Fin N) × Ξ)) 1 :=
  .round (fun ω => a * R.share w ω.1 .folded + b * R.share w ω.1 .whole)
    (fun ω => ∀ v ∈ Pouw.Window.subsetDraw K ω.2.1, R.same ω.1 v) fun _ => .done

/-- **The window as one round of the service**, for a split-stage window: `jointGame` at its joint run. -/
noncomputable def serviceGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w : Fin N → ℝ) (a b : ℝ) (K : ℕ) : RoundGame (Paths C M × (Equiv.Perm (Fin N) × Ξ)) 1 :=
  jointGame (W.joint stg) w a b K

/-- **The service's draws**: draw `i` is the permutation's image of `i`, so the `K` draws are `subsetDraw K`. -/
def drawOf {N K : ℕ} {Ξ : Type} (hK : K ≤ N) (β : Equiv.Perm (Fin N) × Ξ) (i : Fin K) : Fin N :=
  β.1 (Fin.castLE hK i)

/-! ## The window's proofs, their static half from the IR -/

/-- **Tile `t`'s slots**, from the IR: unit `t`'s gates (its outputs, which its proof opens), then each one's operands,
in the circuit's order (a position may repeat). -/
def Declared.slots {ι : Type} (Cd : Declared ι) (t : ℕ) : List ℕ :=
  unitGates Cd t ++ (unitGates Cd t).flatMap fun j => ((Cd.dag[j]?).map Gates.IntGate.args).getD []

/-- **The window's proofs' run-dependent half**, on the window's one service round over the rounds' coins and the
draw `B`: whether the proof of the `i`-th drawn tile was accepted and the values its opening gives its slots, and each
tile's relation (the public Program its proof passes) and rightness on its slots' values, as `DrawnLayer`'s fields of
those names. Everything else of the layer is the IR's (`Declared.layer`). -/
structure ProofRun (C B : Type) (m s Np : ℕ) (V : Type) where
  accepted : Paths (Paths C m × B) 1 → Fin 1 → Fin s → Prop
  opened : Paths (Paths C m × B) 1 → Fin 1 → Fin s → ℕ → V
  rel : Paths (Paths C m × B) 1 → Fin 1 → Fin Np → (ℕ → V) → Prop
  right : Paths (Paths C m × B) 1 → Fin 1 → Fin Np → (ℕ → V) → Prop

/-- **The window's proofs as the IR defines them** (`DrawnLayer`, at one service round): positions are the
circuit's gates and slots are numbered, and
* `draw`: the service's (`drawOf`, the first `K_d` images of a uniform permutation of the `N_p` tiles);
* `pos t k`: tile `t`'s `k`-th slot (`Declared.slots`), its first slot past the list (no position for a tile of no
  gates: `Cd.dag.length`, which no unit writes);
* `coin t`: its unit's round (`Declared.roundOf`, in range under `Declared.Wf`);
* `cut ℓ`: gate `ℓ`'s IR cut (`Declared.cutOf`) or the coin the service commits it before (`Declared.commitRound`),
  whichever is later, the draw past the last round: the layer binds a value no earlier than the service does (at
  `Declared.atDraw` every cut is the draw, at `Declared.atIR` every cut the IR's);
* `src ℓ`: the unit of gate `ℓ` for a value the prover writes (`Declared.Writes`), none for a coin input or past the
  circuit;
and the run-dependent rest is `pr`'s. A cheater can't change any static field: they are the IR's, fixed before the
run. -/
def Declared.layer {ι C Ξ V : Type} (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V) :
    DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) 1 Cd.rounds Kd (Fin Np) ℕ ℕ V where
  draw := drawOf hK
  pos := fun _ t k => (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length)
  coin := fun _ t => ⟨min (Cd.roundOf t.val) (Cd.rounds - 1),
    Nat.lt_of_le_of_lt (Nat.min_le_right _ _) (Nat.sub_lt hr Nat.one_pos)⟩
  cut := fun ℓ => (0, ⟨min (max (Cd.cutOf ℓ) (Cd.commitRound ℓ)) Cd.rounds,
    Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩)
  src := fun ℓ => if h : ℓ < Cd.dag.length ∧ Cd.binding ℓ ≠ some .coin ∧ Cd.unit ℓ < Np then
    some (0, ⟨Cd.unit ℓ, h.2.2⟩) else none
  accepted := pr.accepted
  opened := pr.opened
  rel := pr.rel
  right := pr.right

/-- **Each position committed by its IR cut is bound there** (B1 for the window, at the writing
gate's round, W1): each value `ℓ` the prover writes whose commit coin is no later than its IR cut (`LayerCommitted`'s
first case) is bound before coin `Declared.cutOf ℓ` of the window's `rounds` (the draw past the last). A value opened
one way for one drawn tile and another for a tile that reads it, after the reader's coin, is then a binding failure
(`DrawnLayer.MovedIn`). `Declared.layer` meets it, so at `Declared.atIR` for every value. A value of
`LayerCommitted`'s second case, committed past its IR cut, is bound at its commit coin, and only its own unit reads
it. -/
def StageCuts {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ ≤ Cd.cutOf ℓ → ((pf.cut ℓ).2 : ℕ) = min (Cd.cutOf ℓ) Cd.rounds

/-- **Each tile's noise reads its round's coin** (W2: one coin a round): the layer's coin for tile `t` is
its unit's round, the coin `Declared.Late` reads. `Declared.layer` meets it under `Declared.Wf`. -/
def StageCoins {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ t : Fin N, ((pf.coin 0 t : Fin Cd.rounds) : ℕ) = Cd.roundOf t.val

/-- **Each tile's slots and each position's writer lie inside the IR**: a position with a writer is a value the prover
writes, a gate of the writer's unit, and each slot of tile `t` is one of `t`'s own gates or an operand of one of them
(`Declared.slots`), or no position when `t` has no gates. With `StageCuts`, `StageCoins` and `LayerCommitted` it puts
the layer's late cones (`DrawnLayer.Late`) inside the IR's (`Declared.Reaches`), so a share charging each skipped tile
its `tileCone` covers every tile that isn't good (`ServiceCovers`). `Declared.layer` meets it by definition. -/
def StageSlots {ι C B V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) ℕ ℕ V) :
    Prop :=
  (∀ ℓ u, pf.src ℓ = some u → Cd.Writes ℓ ∧ Cd.unit ℓ = u.2.val) ∧
    ∀ (t : Fin N) (k : ℕ), pf.pos 0 t k ∈ Cd.slots t.val ∨ (Cd.slots t.val = [] ∧ pf.pos 0 t k = Cd.dag.length)

/-- **Every one of the IR's slots and writers is the layer's** (the completeness half `StageSlots` leaves out): every
value the prover writes has its unit as its writer, and every gate of tile `t` and every operand of one is one of `t`'s
slots. Without it the layer could miss a late edge, and `DrawnLayer.Good` would hold for a tile whose words read a late,
skipped writer, which `UnitFixed` can't meet. `Declared.layer` meets it under `Declared.Wf` and `units ≤ N`. -/
def StageComplete {ι C B V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) ℕ ℕ V) :
    Prop :=
  (∀ ℓ, Cd.Writes ℓ → ∃ u, pf.src ℓ = some u ∧ u.2.val = Cd.unit ℓ) ∧
    ∀ (t : Fin N) (ℓ : ℕ), ℓ ∈ Cd.slots t.val → ∃ k, pf.pos 0 t k = ℓ

/-- **The per-stage rate condition at `ν, ν₀`** for `K` draws at weight bound `ρ` (each tile is at least `ρ` times its
work share of the `N` tiles), luck `p` folded and `p0` whole, costs `a` and `b` per unit of share and stage shares at
most `w_max`: `ν, ν₀ ≥ 0`, and for every stage's folded work `W_F`, whole work `W_0` and their lucky parts `L_F`, `L_0`,
`exp((a·W_F + b·W_0)/Λ)·(1 − ρ·((W_F − L_F) + (W_0 − L_0)))^K ≤ 1 + ν·(L_F − p·W_F) + ν₀·(L_0 − p0·W_0)`. A rate
quoted at a `Λ` is proved only with its `ν, ν₀` as terms (`SplitRateExp`), not as floats: at the sharp `Λ` the
condition is tight to rounding. -/
def SplitRateAt (K : ℕ) (ρ p p0 a b wmax Λ ν ν0 : ℝ) : Prop :=
  0 ≤ ν ∧ 0 ≤ ν0 ∧ ∀ WF W0 LF L0 : ℝ, 0 ≤ LF → LF ≤ WF → 0 ≤ L0 → L0 ≤ W0 → WF + W0 ≤ wmax →
    Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * ((WF - LF) + (W0 - L0))) ^ K ≤
      1 + ν * (LF - p * WF) + ν0 * (L0 - p0 * W0)

/-- **The per-stage rate condition**: `SplitRateAt` for some `ν, ν₀`. -/
def SplitRate (K : ℕ) (ρ p p0 a b wmax Λ : ℝ) : Prop := ∃ ν ν0 : ℝ, SplitRateAt K ρ p p0 a b wmax Λ ν ν0

/-! ## The logical layer's conclusion -/

/-- **Every round that holds a credited tile is live**: each unit with a credited word has its round's coin `live`
(`Declared.coinMode`). The live statements need it: a seeded round's noise is a hash of what the prover committed, so
the cheater can try seeds, which the seeded statements price (`premiumOf`). -/
def Declared.Live {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : Prop :=
  ∀ u, u < Cd.units → (unitWords Cd u).any (fun i => Sc.credited (kindAt Cd i)) = true →
    Cd.coinMode (Cd.roundOf u) = .live

/-- **The logical parameters' ranges**: the circuit well formed, every round holding a credited tile live
(`Declared.Live`: a window with a seeded one is the seeded statements', `ScopeSeeded`), `δ_s > 0`, every kind's
luck at least its template's proved value and in range, and the service committing by the IR's cuts, a computed value
only its own unit reads by the draw (`LayerCommitted`, protocol structure, which today's code doesn't meet: a live
window it serves is out of scope). -/
def Scope {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (δs : ℝ) : Prop :=
  Cd.Wf ∧ Cd.Live Sc ∧ 0 < δs ∧ Sc.LuckProved ∧ LayerCommitted Cd

/-- **Every round that holds a credited tile is seeded, and a seeded round is one call site** (one seed
a call): each unit with a credited word has its round's coin `seeded`, and two gates of a seeded round are of one
call site, so a seeded round is one seed group, whose coin is the oracle's answer at that call's preimage and whose
grinding premium `roundPremium` prices. The mirror of `Declared.Live`. -/
def Declared.Seeded {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : Prop :=
  (∀ u, u < Cd.units → (unitWords Cd u).any (fun i => Sc.credited (kindAt Cd i)) = true →
    Cd.coinMode (Cd.roundOf u) = .seeded) ∧
  ∀ i j, i < Cd.dag.length → j < Cd.dag.length → Cd.round (Cd.call i) = Cd.round (Cd.call j) →
    Cd.coinMode (Cd.round (Cd.call i)) = .seeded → Cd.call i = Cd.call j

/-- **The seeded statements' ranges**: the circuit well formed, every round holding a credited tile seeded and each
seeded round one call site (`Declared.Seeded`), no credited kind whose premium GRIND hasn't priced
(`Scheme.SeedPriced`: H1 + fx1's attention uncredited), `δ_s > 0`, and every value a seeded round reads by its IR cut
named by the round's seed (`SeedNamed`, protocol structure, which today's code doesn't meet: a seeded window it
serves is out of scope until its seeds name what each round reads). The seeded statements read the templates' luck
through GRIND's premise (`Assumptions.Compose.SeedLuckAt`), not through `Scheme.LuckProved`'s `μ`. -/
def ScopeSeeded {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (δs : ℝ) : Prop :=
  Cd.Wf ∧ Cd.Seeded Sc ∧ Sc.SeedPriced ∧ 0 < δs ∧ SeedNamed Cd

/-- **The logical layer's bound** `r = Λ·ln(1/δ_s) + min(a, b)·D₀`: the service's budget at `δ_s`, and the cone weight
on tiles of no words (`zeroCone`, W4, `0` for every Program COMPILE accepts), which no draw detects, at the smaller
cost: `WorkAt` takes the skipped share split any way into folded and whole, so the zero-word excess goes in whichever
share costs less. -/
noncomputable def rOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (a b Λ δs : ℝ) : ℝ :=
  Λ * Real.log (1 / δs) + min a b * zeroCone Sc Cd Np

/-! ## The seeded statements' premium, per seed group from the IR -/

/-- The rounds' coins `x` with round `r`'s replaced by `c`. -/
def setAt {C : Type} : {n : ℕ} → Paths C n → Fin n → C → Paths C n
  | 0, _, r, _ => r.elim0
  | _ + 1, x, ⟨0, _⟩, c => (c, x.2)
  | _ + 1, x, ⟨r + 1, h⟩, c => (x.1, setAt x.2 ⟨r, Nat.lt_of_succ_lt_succ h⟩ c)

/-- **One oracle answer of the window redrawn**: the window's oracle answers `a` (round `r`'s seed queries' answers
its `r`-th coin, `Fin T_q → B`) with query `q` of round `r` answered `b`, every other answer of every round as `a`
has it. GRIND's per-query moment (`Assumptions.Compose.SeedLuckAt`) is over this one answer. -/
def redraw {B : Type} {M Tq : ℕ} (a : Paths (Fin Tq → B) M) (r : Fin M) (q : Fin Tq) (b : B) :
    Paths (Fin Tq → B) M :=
  setAt a r (Function.update (coinAt a r) q b)

/-- A tile's round: its unit's (`Declared.roundOf`), the seed group whose seed its noise reads. -/
def tileRound {ι : Type} (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℕ := Cd.roundOf t.val

open Classical in
/-- **Round `r`'s gain at exchange rate `κ`** (GRIND's `L − κ·U`, on the whole window's coins `a`): the tiles of round
`r` (`rd v = r`) in mode `s` of the joint run `R`, each `w v` when its checked values are honest (`same`, lucky) and
`−κ·w v` when not. Its modes and checked values may read every round's coin, so a cheater that picks round `r`'s skips
after querying later rounds' seeds is in it. GRIND bounds it per seed group by the premium `π*` except with
probability `δ_g + δ_b`, off the oracle's `ε_id` (`Assumptions.Compose.SeedLuckAt`, `SeedExtractedAt`). -/
noncomputable def roundGain {A V : Type} [Fintype V] (rd : V → ℕ) (r : ℕ) (R : JointRun A V) (w : V → ℝ) (s : Mode)
    (κ : ℝ) (a : A) : ℝ :=
  ∑ v, if rd v = r ∧ R.mode a v = s then w v * ((1 + κ) * (if R.same a v then 1 else 0) - κ) else 0

/-- Round `r`'s credited words: the circuit's credited words whose call site is in round `r` (one seed group under
`Declared.Seeded`). -/
def roundWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (r : ℕ) : List ℕ :=
  (creditedWords Sc Cd).filter fun i => Cd.round (Cd.call i) == r

/-- **Round `r`'s rate per committed word** at a rate per kind (`Scheme.seedRate` folded, `Scheme.wholeRate` whole):
the least of its credited words' kinds' (one kind for a seed group of one call site), and `0` for a round of no
credited words. -/
noncomputable def roundRate {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (rate : Kind → ℝ) (r : ℕ) : ℝ :=
  match roundWords Sc Cd r with
  | [] => 0
  | i :: l => l.foldl (fun m j => min m (rate (kindAt Cd j))) (rate (kindAt Cd i))

/-- **Round `r`'s detected weight**: the `shareCone` of its tiles, the most of the window's share its seed group can
leave lucky. -/
noncomputable def roundCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (r : ℕ) : ℝ :=
  ∑ t : Fin Np, if Cd.roundOf t.val = r then shareCone Sc Cd Np t else 0

/-- **Round `r`'s folded rate per unit of share**: `N_w` times its `Scheme.seedRate` per word, the `θ_r` at which
`SeedLuckAt` takes GRIND's template moment for round `r`'s folded tiles. -/
noncomputable def seedTheta {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (r : ℕ) : ℝ :=
  (wordsOf Sc Cd : ℝ) * roundRate Sc Cd Sc.seedRate r

/-- **Round `r`'s whole rate per unit of share**: `N_w` times its `Scheme.wholeRate` per word, the `θ₀_r` at which
`SeedLuckAt` takes the guess bound's moment for round `r`'s whole tiles. -/
noncomputable def wholeTheta {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (r : ℕ) : ℝ :=
  (wordsOf Sc Cd : ℝ) * roundRate Sc Cd Sc.wholeRate r

/-- **Round `r`'s grinding premium** at rate `θ` (GRIND's `π* = ln(T·M/δ)/θ` at `M = 1`), per unit of
share: `ln(T_q/δ_g)/θ_r` for `T_q` oracle calls at the round's rate `θ_r` (`seedTheta` folded, `wholeTheta` whole), at
most the round's own weight (`roundCone`), and the whole weight where no rate is proved (`θ_r = 0`). Computed from the
IR per seed group, so a small site pays what it costs and no cover is assumed. -/
noncomputable def roundPremium {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np Tq : ℕ) (δg : ℝ) (θ : ℕ → ℝ)
    (r : ℕ) : ℝ :=
  if 0 < θ r then min (Real.log (Tq / δg) / θ r) (roundCone Sc Cd Np r) else roundCone Sc Cd Np r

/-- **The window's grinding premium** at rate `θ`, `Π = Σ_r π*_r`, over its rounds (its seed groups under
`Declared.Seeded`). -/
noncomputable def premiumOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np Tq : ℕ) (δg : ℝ) (θ : ℕ → ℝ) : ℝ :=
  ((List.range Cd.rounds).map (roundPremium Sc Cd Np Tq δg θ)).sum

/-- **The seeded logical layer's bound**
`r = max(5a, (1 + κ₀)·b)·ln(1/δ_s)/(ρ·K_d) + a·Π_F + b·Π_0 + min(a, b)·D₀`, `Π_F` the premium at the folded rates
(`seedTheta`) and `Π_0` at the whole ones (`wholeTheta`). Per round, GRIND's premise (`Assumptions.Compose.SeedLuckAt`)
at the template's `κ = 4` (`TemplateMoment`'s point) for folded tiles and at `κ₀` (a whole-tile luck a seed query,
which no ruling grants, so the bound's value is conditional on it) for whole ones gives, except with probability `δ_g + δ_b` each and the oracle's `ε_id` once (`SeedExtractedAt`), lucky folded share
`L_F < 4·U_F + π_F` and lucky whole share `L_0 < κ₀·U_0 + π_0`, `U` the share whose checked values aren't honest. An
accepted draw of `K_d` tiles then has `ρ·K_d·(U_F + U_0) < ln(1/δ_s)` except with probability `δ_s`, so
`a·(L_F + U_F) + b·(L_0 + U_0)` is below `r`, with `D₀` (`zeroCone`) as in `rOf`. No `SplitRate` is read: the live
draw prices only the unlucky share, and the premiums the lucky excess. -/
noncomputable def rOfSeeded {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np Kd Tq : ℕ) (a b κ0 ρ δs δg : ℝ) : ℝ :=
  max (a * (1 + 4)) (b * (1 + κ0)) * Real.log (1 / δs) / (ρ * Kd) +
    a * premiumOf Sc Cd Np Tq δg (seedTheta Sc Cd) + b * premiumOf Sc Cd Np Tq δg (wholeTheta Sc Cd) +
    min a b * zeroCone Sc Cd Np

/-- **What an accepted window establishes about its run** (the logical layer): of its `N_w` committed words,
all but a share `d_F + d₀` are fixed right (`fixed`) and all but `d₀` are folded (`folded`), the skipped share split
into the folded units' `d_F` and the whole units' `d₀` (B1), with `a·d_F + b·d₀ < r`. At `a = b = 1` it says the
words fixed right are more than a share `1 − r` of the work. -/
def WorkAt (Nw a b r fixed folded : ℝ) : Prop :=
  ∃ dF d0 : ℝ, 0 ≤ dF ∧ 0 ≤ d0 ∧ (1 - (dF + d0)) * Nw ≤ fixed ∧ (1 - d0) * Nw ≤ folded ∧ a * dF + b * d0 < r

/-! ## The physical layer: the tensor–scalar model on a data sheet -/

section Credit

variable {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- **The credit**: `T_cred = B(♭)/f_max`, the circuit's floor on the model's machine. -/
noncomputable def creditOf (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) : ℝ := Z.credit (floorOf Sc Cd)

/-- `ε_♭ = ε_skip + ε_cert + ε_route` at the circuit's floor. -/
noncomputable def epsFlat (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ :=
  Z.epsSkip (floorOf Sc Cd) lam + Z.epsCert (floorOf Sc Cd) lam + Z.epsRoute (floorOf Sc Cd) lam

/-- `ε_μ = ε_mand` at the circuit's floor. -/
noncomputable def epsMu (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ :=
  Z.epsMand (floorOf Sc Cd) lam

/-- **A folded share's time**, per unit of share: `κ(π) = N_w·w*(π)/B_♭`, the words it skips. -/
noncomputable def aOf (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ :=
  Z.kappa (floorOf Sc Cd) lam

/-- **A whole share's time**, per unit of share: `κ(π) + (1 − μ_f)·N_w·φ*(π)/B_♭`, its words and its fold. -/
noncomputable def bOf (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) : ℝ :=
  aOf Sc Cd Z lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd)

/-- **The circuit's `ε`** at the logical layer's bound `r`: `ε_♭ + ε_μ + r`. For a live window,
`r = Λ·ln(1/δ_s) + min(a, b)·D₀` (`rOf`), where `Λ` bounds its time saved per unit of detection (`ServiceBudget`). -/
noncomputable def epsOf (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) (r : ℝ) : ℝ :=
  epsFlat Sc Cd Z lam + epsMu Sc Cd Z lam + r

/-- **The model is the circuit's on sheet `Sh`**: its machine is the sheet's, and its committed words are the circuit's
(`N_w = wordsOf`). -/
def Fits {P : Type} [DecidableEq P] (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P)
    (Z : Model K (SheetRow P) T L U F X) : Prop :=
  Z.G = Sh.machine ∧ Z.Nw = (wordsOf Sc Cd : ℝ)

/-- **The model's ranges**: `π` a certificate, the model well formed at the circuit's floor, and SCALAR-ROUTE's floors
at the ½ the data sheet gives (`q, q_T ≤ ½`, `b_T ≤ 0`). -/
def ModelScope (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) : Prop :=
  IsCert lam ∧ Z.Wf (floorOf Sc Cd) ∧ Z.q ≤ 1 / 2 ∧ Z.qT ≤ 1 / 2 ∧ Z.bT ≤ 0

end Credit

/-- **A credit certificate** (the interface the physical layer's ledger consumes): the prover system
did work `units k` of each kind `k` in the device window `window` (`Accounting.Window`: `len` ticks from `start`),
worth `credit` seconds on the declared machine, short by at most `eps` of it except with probability `delta`; `gamma` is
honest's time over the credit, `γ_♭ = T(H)/T_cred`, which says what the floor `♭` is against the honest kernel (W1);
`coins` is the coin mode `eps` is for: `live` from `CircuitCredit`, every round holding a credited tile live, and
`seeded` from `CircuitCreditSeeded`, with its grinding premium. The issuer of composed certificates, which doesn't exist
yet, refuses a live `eps` for a seeded window.
It is what the physical layer's ledger consumes, never a credit by itself: the ledger admits a set of certificates on
one device only if the credit inside every interval of its ticks fits in it, so one device's time is credited once. -/
structure Certificate where
  window : Accounting.Window
  units : Kind → ℕ
  credit : ℝ
  eps : ℝ
  delta : ℝ
  gamma : ℝ
  coins : CoinMode

/-- **The certificate holds for a play**: its accepted runs take less than `(1 − eps)·credit` with probability at
most `delta`. -/
def Certificate.Holds {Ω : Type} [Fintype Ω] {S : ℕ} {K T L U : Type} (c : Certificate) (P : Play Ω S K T L U) :
    Prop :=
  prob (fun x => P.accepts x ∧ (P.run x).run.time < (1 - c.eps) * c.credit) ≤ c.delta

/-- **The circuit's certificate** in device window `win`: its words per kind, `T_cred`, `ε` at the logical layer's
bound `r`, the given `δ`, `γ_♭` against honest kernel `H`, and the coin mode it is issued for (`.live` in
`CircuitCredit`, `.seeded` in `CircuitCreditSeeded`). -/
noncomputable def certOf {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (H : Honest K)
    (lam : R → ℝ) (r δ : ℝ) (win : Accounting.Window) (coins : CoinMode) : Certificate where
  window := win
  units := kindCount Cd
  credit := creditOf Sc Cd Z
  eps := epsOf Sc Cd Z lam r
  delta := δ
  gamma := Z.gammaFloor H (floorOf Sc Cd)
  coins := coins

end Accounting.Compose
