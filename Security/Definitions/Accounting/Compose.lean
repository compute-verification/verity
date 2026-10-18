import Definitions.Accounting.TS
import Definitions.Accounting.Gates.Model
import Definitions.Accounting.Device
import Definitions.Pouw.Window

/-!
# PoUW as a compile pass over an arbitrary declared circuit (trusted)

The vocabulary of `Models.Accounting.Guarantees.Compose`
(note:compute-accounting/20261018T1230Z-draft-theorem-compose-r3), restated after RT-COMPOSE r2
(note:compute-accounting/20261018T1200Z-draft-rt-compose-r2) under decisions 107, 109–114, 117, 119 and 122. The
compile pass `C → C′` replaces each credited product of a declared circuit by its kind's template, with its noise drawn
after its operands are committed, one coin a round (decision 122). The service then draws one uniform subset of the
window's grid-padded tiles, one a proof unit, and proves each drawn tile with the generation of its own noise.

Decision 113 splits it in two layers. The logical layer, from `Declared` to `WorkAt`, is the circuit, its rounds, its
proof units, the templates' luck, the service's draw and what an accepted window establishes about the run's committed
words; it reads no machine model. The physical layer, from `creditOf` to `certOf`, is the tensor–scalar model on a
data sheet, which prices those words as time on a declared device.

The logical layer:

* `Declared`: the circuit as the IR gives it, a `Accounting.Gates.Dag` (memory accounting's form, with its wiring),
  each gate's Definition call site, each call site's round (the coin its noise reads: one coin a round, decision 122)
  and each gate's proof unit (decision 107: an output coordinate, a K-chunk of 2048 committing its running sum, or a
  unit of glue). `Declared.Wf`: the call sites, rounds and units in range, and one unit's gates in one round and its
  words of one kind.
* `isWord`, `wordGates`, `Kind`, `kindAt`: the circuit's committed words of served products (`T_t`), and each gate's
  kind, computed from the IR, never asserted: its primitive, its site's statics (what `Gates.roleOf` reads) and
  whether each operand depends on a run input (an activation) or not (a weight). So a gather-fed `QKᵀ` and a weight
  matmul of one shape are different kinds.
* `unitGates`, `unitWords`, `Declared.roundOf`, `Declared.stageOf`, `Declared.kindOf`: a unit's gates and words, its
  round, its stage counted from the last round (as `SplitStages` counts) and its words' kind (none for glue).
* `Declared.cutOf`, `Declared.Late`, `Declared.Reaches`: each value's cut, computed from the IR (W1): a registered input
  before the first coin, and every other value (glue, a run input, the KV cache, decode feedback) with its writing
  gate's round, before the next round's coin; a read is late when its operand's cut is past the reader's coin; a unit
  reaches another along late reads, through any gate.
* `LuckForm`, `Scheme`, `Scheme.LuckProved`, `Scheme.foldLuck`: per kind, the floor its template's noise proves at the
  worst operands (decision 109: `0` where its noise proves none), whether it is credited, its skip pattern's span, the
  form of its template's proved operand luck, and the three lucks the window is held to: (H-μ)'s point mass `μ`, the
  receipt's B′ luck `r_B` and the whole-unit guess bound. A folded unit's luck is `μ + (1 − μ)·r_B` (`7/8` at
  `μ = ¾`, `r_B = ½`, `Accounting.Guarantees.TS.LuckOfReceipt`).
* `creditedWords`, `wordsOf`, `floorOf`, `kindCount`: the circuit's credited words, `N_w` (W3: from the IR, not a
  parameter), its floor `♭ = Σ_words ♭_{kind}` and its words per kind.
* `tileWords`, `tileWeight`, `tileStage`, `tileKind`, `tileLuck`, `tileLuck0`: the service's `N_p` grid-padded tiles
  (`verity/core/protocols/pouw/window.py`): unit `t` for `t < units`, and fillers of no words past them.
* `tileCone`, `shareCone`, `zeroCone`, `kindF`: a tile's late cone's weight (decision 119), its share weight (its cone
  when it carries words), the cone weight on tiles of no words (W4), and `F` per kind: the most tiles one late cone of
  a kind holds (`1` for a whole coordinate, `K/2048` for K-chunks of 2048).
* `Mode`, `SplitStages`, `modeAt`, `sameAt`, `SplitStages.LuckAt`, `EveryStage`, `share`, `stageShare`: a window's
  stages, one a round, with a unit paid, folded (its product skipped and its committed words folded) or whole (product
  and fold skipped), each stage's tiles' deficient words right or not (`right`) and their checked values honest or not
  (`same`, which a receipt-lucky skipped tile meets).
* `tileGame`, `PerTile`: a tile's stage as the ideal game split at the receipt (`ReceiptGame`), in each skipped mode,
  and a claim of every credited tile at its stage, whatever the past: what #2482's `OperandHMu` and `ReceiptLuckAt` and
  `Assumptions.Compose.GuessLuckAt` are taken of.
* `serviceGame`, `drawOf`: the window as one round on coins `Paths C M × (Equiv.Perm (Fin N) × Ξ)`: the rounds' coins,
  then the service's one uniform `K`-subset (`Pouw.Window.subsetDraw`, its `Equiv.Perm`) and the proofs' coins `Ξ`.
  Its share is `a·(folded weight) + b·(whole weight)` at the weights it is given, and it passes when every drawn tile's
  checked value is honest.
* `StageCuts`, `StageCoins`: the window's proofs (`Accounting.TS.DrawnLayer` over the IR's gates as positions) bind
  each position at its IR cut, and each tile's noise reads its round's coin (W2).
* `SplitRate`: the per-stage condition from which a rate `Λ` bounds the window's saving per unit of detection.
* `Scope`, `rOf`, `WorkAt`: the logical parameters' ranges, the logical layer's bound `Λ·ln(1/δ_s) + max(a, b)·D₀` with
  `D₀` the cone weight on tiles of no words, and its conclusion on an accepted run: all but a share `d` of the
  committed words fixed right and all but `d₀ ≤ d` folded, with `a·(d − d₀) + b·d₀ < r`.

The physical layer:

* `creditOf`, `epsFlat`, `epsMu`, `aOf`, `bOf`, `epsOf`: `T_cred = B(♭)/f_max`, NOTATION's `ε_♭` and `ε_μ` at the
  circuit's floor, the time a folded and a whole share of `1` cost (`κ`, and `κ + (1 − μ_f)·N_w·φ*/B_♭`), and
  `ε = ε_♭ + ε_μ + r` at the logical layer's bound `r`.
* `Fits`, `ModelScope`: the model is the circuit's on the data sheet; the model's ranges.
* `Certificate`, `Certificate.Holds`, `certOf`: decision 111's interface, which the physical layer's ledger consumes:
  work of each kind in a device window (`Accounting.Window`, its ticks), its credit, `ε`, `δ` and `γ_♭ = T(H)/T_cred`
  (W1: `♭` reported, not pinned). The ledger, not this statement, decides how much of a device's time the certificates
  it admits may claim.
-/

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

/-! ## The declared circuit: its rounds, its proof units and its words -/

/-- **A declared circuit**: its gates in the IR's canonical order (`Accounting.Gates.Dag`: each gate's operation,
operands, interval, primitive, site and whether it is committed), and as the IR binds them:
* `call i`: the Definition call site, of `calls`, that made gate `i`, numbered in the circuit's order;
* `round c`: call site `c`'s round, of `rounds`: its noise reads that round's coin, released after the round's operands
  are committed (one coin a round, decision 122; one salt for the window, COMPILE);
* `unit i`: gate `i`'s proof unit, of `units`, the unit the service draws and proves alone (decision 107: an output
  coordinate, or a K-chunk of 2048 committing its running sum; glue between products is in units of its own). -/
structure Declared (ι : Type) where
  dag : Gates.Dag ι
  calls : ℕ
  call : ℕ → ℕ
  rounds : ℕ
  round : ℕ → ℕ
  units : ℕ
  unit : ℕ → ℕ

/-- Whether each gate's value depends on a run input, in order: a `run` input, or a gate that reads such a value.
Unlike `Gates.dynamic` it leaves coins out, so it tells an activation (a request's tokens and what they reach) from a
weight or a codec's parameters. -/
def runDep {ι : Type} (C : Gates.Dag ι) : List Bool :=
  C.foldl (fun ds g => ds ++ [match g.op with
    | .input b _ => b == .run
    | _ => g.args.any fun j => ds.getD j false]) []

/-- **A committed word**: a committed gate on a served product's main chain, its word `T_t` (S8-PORT's COMMIT layout:
the final `T_1`, or the prefix `T_1 … T_{K/L}`). -/
def isWord {ι : Type} (g : Gates.Gate ι) : Bool := g.commit && g.role.isServed

/-- The circuit's committed words, as gate indices in the circuit's order. -/
def wordGates {ι : Type} (Cd : Declared ι) : List ℕ :=
  (List.range Cd.dag.length).filter fun i => Cd.dag[i]?.any isWord

/-- **A product's kind**, read off the IR: its primitive's id, its site's statics (the step zeroed: `K`, `L` and the
tail groups, what `Gates.roleOf` reads) and whether its `a` and `b` operands depend on a run input. -/
structure Kind where
  prim : String
  site : Gates.Site
  actA : Bool
  actB : Bool
  deriving DecidableEq

/-- A site's statics: an `S8Dot_v1` body's `K`, `L` and tail groups, its step zeroed. -/
def statics : Gates.Site → Gates.Site
  | .dot K L tail _ => .dot K L tail 0
  | .other => .other

/-- An s8-family step's `k`, and `0` for any other operation. -/
def mmaK {ι : Type} : Gates.Op ι → ℕ
  | .mma _ _ k _ => k
  | _ => 0

/-- Gate `g`'s kind in circuit `C`: its primitive, its site's statics, and whether any of its `a` operands
(`args[1 … k]`) and any of its `b` operands (`args[k+1 …]`) depends on a run input. -/
def kindOfGate {ι : Type} (C : Gates.Dag ι) (g : Gates.Gate ι) : Kind :=
  ⟨g.prim, statics g.site, (g.args.tail.take (mmaK g.op)).any fun j => (runDep C).getD j false,
    (g.args.tail.drop (mmaK g.op)).any fun j => (runDep C).getD j false⟩

/-- **Gate `i`'s kind** (`kindOfGate`), and an empty one past the circuit's gates. -/
def kindAt {ι : Type} (Cd : Declared ι) (i : ℕ) : Kind :=
  (Cd.dag[i]?).elim ⟨"", .other, false, false⟩ (kindOfGate Cd.dag)

/-- **The circuit is well formed**: every gate's call site and unit are in range, every call site's round is, one
unit's gates are in one round, and one unit's words are of one kind. -/
def Declared.Wf {ι : Type} (Cd : Declared ι) : Prop :=
  (∀ i, i < Cd.dag.length → Cd.call i < Cd.calls ∧ Cd.unit i < Cd.units) ∧
    (∀ c, c < Cd.calls → Cd.round c < Cd.rounds) ∧
    (∀ i j, i < Cd.dag.length → j < Cd.dag.length → Cd.unit i = Cd.unit j →
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

/-- **Gate `i`'s cut**: the coin before which its value is committed, as an index of the window's `rounds` coins (the
draw is `rounds`). A registered input (a weight, a codec's parameters) is bound before the first coin, `0`. Every other
value is committed with its writing gate's round, before the next round's coin (W1): a gate's output, glue among them,
and a run input (a request's tokens, the KV cache, decode feedback), which is bound after the coins, so not before the
first. -/
def Declared.cutOf {ι : Type} (Cd : Declared ι) (i : ℕ) : ℕ :=
  match (Cd.dag[i]?).map (·.op) with
  | some (.input .registered _) => 0
  | _ => Cd.round (Cd.call i) + 1

/-- **Gate `j` reads gate `i` after its own coin**: `i` is one of `j`'s operands, and `i`'s cut is past the coin of
`j`'s round, so `i`'s value can be chosen after `j`'s noise is drawn. Within a round every read is late (a K-chunk's
running sum, glue between products of one round); a value committed in an earlier round, or registered, is fixed when
`j`'s noise is drawn, whether it is right or not, so it is no edge. -/
def Declared.Late {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop :=
  (Cd.dag[j]?).any (fun g => g.args.contains i) = true ∧ Cd.round (Cd.call j) < Cd.cutOf i

/-- **Unit `u'` is in unit `u`'s late cone**: a gate of `u'` is reached from a gate of `u` along late reads
(`Declared.Late`, transitively, through gates of any unit); `u` itself when it has a gate. -/
def Declared.Reaches {ι : Type} (Cd : Declared ι) (u u' : ℕ) : Prop :=
  ∃ i j, i < Cd.dag.length ∧ Cd.unit i = u ∧ Cd.unit j = u' ∧ Relation.ReflTransGen Cd.Late i j

/-! ## Schemes: floors and luck per kind -/

/-- **The form of a template's proved operand luck**, at a skip pattern's word span:
* `cube bind`: `¾` while the span is within the receipt's binding `bind` (`Accounting.Guarantees.TS.LuckCaseACube`,
  `OperandLuckCube`, for degree-2 cube operands; #2447's `Pouw.Guarantees.Receipt.DuoWindowNoKernel` binds four
  words), and `1` past it (#2447's `DuoLongRunKernel`: no receipt binds every length);
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
operands (decision 109; `0` where the noise proves none, as one-sided noise proves none), whether the kind is credited
(a word of an uncredited kind counts in no `N_w` and no floor), the span a skip pattern has before the receipt's next
checkpoint (W6), the form of the operand luck its template proves, and the three lucks the window is held to:
* `mu`: (H-μ)'s point mass at a folded unit's deficient words (#2482's `OperandHMu`, `¾` in the headlines);
* `rB`: the receipt's luck when they are not all right (#2482's `ReceiptLuckAt`, HMU's B′, card 8b00536a, `½`);
* `guess`: a unit skipped whole keeps its honest checked value only by a guess of its fold state
  (`Assumptions.Compose.GuessLuckAt`, `2^-64` for a 64-bit state).
H1's attention matmuls are credited at their floors with no special case (decision 117, ATTN-NOISE r2): `QKᵀ` draws a
coin per entry on both operands; `P·V`'s `P` carries only the period-32 word `U` and `V` a coin `b` per entry, and its
floor is `U·b`'s, a cell per `k`, which holds under any salt-independent advice while the kernel commits a prefix word at
most every 32 keys (at one every `C` keys it is `32·k/C`). -/
structure Scheme (K : Type) where
  floor : Kind → Labelled K
  credited : Kind → Bool
  span : Kind → ℕ
  form : Kind → LuckForm
  mu : Kind → ℝ
  rB : Kind → ℝ
  guess : Kind → ℝ

/-- **The luck is at least what each template proves, and in range**: per kind, (H-μ)'s point mass is at least its
form's value at the kind's span, so `OperandHMu` asks no more of the cheater's view than the template's own coins give,
the receipt's luck lies in `[0, 1]` (`LuckOfReceipt` needs `r_B ≤ 1`) and the guess bound is `≥ 0`. -/
def Scheme.LuckProved {K : Type} (Sc : Scheme K) : Prop :=
  ∀ k, (Sc.form k).value (Sc.span k) ≤ Sc.mu k ∧ 0 ≤ Sc.rB k ∧ Sc.rB k ≤ 1 ∧ 0 ≤ Sc.guess k

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
/-- **A tile's late cone's weight** (decision 119), from the IR: the weights of the tiles its unit reaches along late
reads (`Declared.Reaches`), itself included. A skipped tile can leave every word of its cone adaptive, and no other. -/
noncomputable def tileCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  ∑ t' : Fin Np, if Cd.Reaches t.val t'.val then tileWeight Sc Cd Np t' else 0

/-- **A tile's share weight**: its late cone's weight when it carries words, and `0` for a tile of no words, whose
skip no luck bounds (`zeroCone` charges it). -/
noncomputable def shareCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  if tileWords Sc Cd Np t = 0 then 0 else tileCone Sc Cd Np t

/-- **`D₀`, the cone weight on tiles of no words** (W4): glue and uncredited units whose late cones carry work. A
cheater may skip them all at no risk, so the logical layer's bound adds it at the larger cost (`rOf`); a circuit whose
glue reaches no product late has `D₀ = 0`. -/
noncomputable def zeroCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) : ℝ :=
  ∑ t : Fin Np, if tileWords Sc Cd Np t = 0 then tileCone Sc Cd Np t else 0

open Classical in
/-- **`F` per kind** (decision 107): the most tiles one late cone of a tile of kind `k` holds, from the IR: `1` for a
unit that is a whole coordinate, which reaches no other late, and `K/2048` for a coordinate cut in K-chunks of 2048,
whose running sums are late within their round. At a tile weight bound `w_max` per tile, `ρ = 1/(F·r)`, `F` the largest
`F_k` and `r = (N_p/N)·(w_max/w̄)`, meets `CircuitWork`'s `ρ·N_p·shareCone ≤ 1`. -/
noncomputable def kindF {ι : Type} (Cd : Declared ι) (Np : ℕ) (k : Kind) : ℕ :=
  (Finset.univ.filter fun t : Fin Np => tileKind Cd Np t = some k).sup fun t =>
    (Finset.univ.filter fun t' : Fin Np => Cd.Reaches t.val t'.val).card

/-! ## The window: stages, one a round, with B1's split, and the service's draw -/

/-- A unit's mode in a cheater's run: paid; folded (its product skipped, its committed words folded); or whole (its
product and its fold skipped). -/
inductive Mode
  | paid
  | folded
  | whole
  deriving DecidableEq

/-- **A window's stages, one a round**: per stage, each unit's mode, whether its deficient words are all right
(`right`) and whether its checked values (the values its proof pins: a warp checkpoint, a fold state) are the honest
ones (`same`), as functions of the stage's coin `C` (the stage's outputs are committed after it is released), then the
rest of the window, which may depend on it. A unit belongs to one stage, `stg u`, counted from the last (stage `0`). A
skipped unit whose receipt is lucky is `same` with wrong words. -/
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

/-- The work in mode `s` along the stages' coins `x`, at work weights `w`. -/
noncomputable def share [Fintype V] {M : ℕ} (stg : V → ℕ) (W : SplitStages C V M) (w : V → ℝ) (x : Paths C M)
    (s : Mode) : ℝ :=
  ∑ u, if W.modeAt stg x u = s then w u else 0

end SplitStages

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

/-- **The window as one round of the service**: on the rounds' coins, the permutation whose first `K` images are the
drawn tiles (`Pouw.Window.subsetDraw`) and the proofs' coins `Ξ`, its share is `a` per unit of folded work and `b` per
unit of whole work, and it passes when every drawn tile's checked values are honest. -/
noncomputable def serviceGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w : Fin N → ℝ) (a b : ℝ) (K : ℕ) : RoundGame (Paths C M × (Equiv.Perm (Fin N) × Ξ)) 1 :=
  .round (fun ω => a * W.share stg w ω.1 .folded + b * W.share stg w ω.1 .whole)
    (fun ω => ∀ v ∈ Pouw.Window.subsetDraw K ω.2.1, W.sameAt stg ω.1 v) fun _ => .done

/-- **The service's draws**: draw `i` is the permutation's image of `i`, so the `K` draws are `subsetDraw K`. -/
def drawOf {N K : ℕ} {Ξ : Type} (hK : K ≤ N) (β : Equiv.Perm (Fin N) × Ξ) (i : Fin K) : Fin N :=
  β.1 (Fin.castLE hK i)

/-- **Each position is bound at its IR cut** (RT-I8's B1 for the window, decision 114, at the writing gate's round,
W1): the window's proofs' positions are the circuit's gates, and gate `ℓ`'s value is bound before coin
`Declared.cutOf ℓ` of the window's `rounds` (the draw past the last). A value opened one way for one drawn tile and
another for a tile that reads it, after the reader's coin, is then a binding failure (`DrawnLayer.MovedIn`). Protocol
structure: each round's values are committed before its next coin. -/
def StageCuts {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ ℓ, ((pf.cut ℓ).2 : ℕ) = min (Cd.cutOf ℓ) Cd.rounds

/-- **Each tile's noise reads its round's coin** (W2: one coin a round, decision 122, one salt for the window): the
layer's coin for tile `t` is its unit's round, the coin `Declared.Late` reads. Protocol structure, as `StageCuts` is:
the cones are computed at the coin each tile's noise really reads, and the proof doesn't read it. -/
def StageCoins {ι C B Q V : Type} {s N : ℕ} (Cd : Declared ι) (pf : DrawnLayer C B 1 Cd.rounds s (Fin N) Q ℕ V) :
    Prop :=
  ∀ t : Fin N, ((pf.coin 0 t : Fin Cd.rounds) : ℕ) = Cd.roundOf t.val

/-- **The per-stage rate condition** for `K` draws at weight bound `ρ` (each tile is at least `ρ` times its work share
of the `N` tiles), luck `p` folded and `p0` whole, costs `a` and `b` per unit of share and stage shares at most
`w_max`: there are `ν, ν₀ ≥ 0` such that for every stage's folded work `W_F`, whole work `W_0` and their lucky parts
`L_F`, `L_0`,
`exp((a·W_F + b·W_0)/Λ)·(1 − ρ·((W_F − L_F) + (W_0 − L_0)))^K ≤ 1 + ν·(L_F − p·W_F) + ν₀·(L_0 − p0·W_0)`. -/
def SplitRate (K : ℕ) (ρ p p0 a b wmax Λ : ℝ) : Prop :=
  ∃ ν ν0 : ℝ, 0 ≤ ν ∧ 0 ≤ ν0 ∧ ∀ WF W0 LF L0 : ℝ, 0 ≤ LF → LF ≤ WF → 0 ≤ L0 → L0 ≤ W0 → WF + W0 ≤ wmax →
    Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * ((WF - LF) + (W0 - L0))) ^ K ≤
      1 + ν * (LF - p * WF) + ν0 * (L0 - p0 * W0)

/-! ## The logical layer's conclusion -/

/-- **The logical parameters' ranges**: the circuit well formed, `δ_s > 0`, and every kind's luck at least its
template's proved value and in range. -/
def Scope {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (δs : ℝ) : Prop := Cd.Wf ∧ 0 < δs ∧ Sc.LuckProved

/-- **The logical layer's bound** `r = Λ·ln(1/δ_s) + max(a, b)·D₀`: the service's budget at `δ_s`, and the cone weight
on tiles of no words (`zeroCone`, W4), which no draw detects, at the larger cost. -/
noncomputable def rOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (a b Λ δs : ℝ) : ℝ :=
  Λ * Real.log (1 / δs) + max a b * zeroCone Sc Cd Np

/-- **What an accepted window establishes about its run** (decision 113's logical layer): of its `N_w` committed words,
all but a share `d_F + d₀` are fixed right (`fixed`) and all but `d₀` are folded (`folded`), the skipped share split
into the folded units' `d_F` and the whole units' `d₀` (B1), with `a·d_F + b·d₀ < r`. At `a = b = 1` it says the
words fixed right are more than a share `1 − r` of the work. -/
def WorkAt (Nw a b r fixed folded : ℝ) : Prop :=
  ∃ dF d0 : ℝ, 0 ≤ dF ∧ 0 ≤ d0 ∧ (1 - (dF + d0)) * Nw ≤ fixed ∧ (1 - d0) * Nw ≤ folded ∧ a * dF + b * d0 < r

/-! ## The physical layer: the tensor–scalar model on a data sheet -/

section Credit

variable {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- **The credit**: `T_cred = B(♭)/f_max`, the circuit's floor on the model's machine (decision 100). -/
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

/-- **The circuit's `ε`** at the logical layer's bound `r`: `ε_♭ + ε_μ + r`. For a window,
`r = Λ·ln(1/δ_s) + max(a, b)·D₀` (`rOf`), where `Λ` bounds its time saved per unit of detection (`ServiceBudget`). -/
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

/-- **A credit certificate** (decision 111's interface, at the physical layer under decision 113): the prover system
did work `units k` of each kind `k` in the device window `window` (`Accounting.Window`: `len` ticks from `start`),
worth `credit` seconds on the declared machine, short by at most `eps` of it except with probability `delta`; `gamma` is
honest's time over the credit, `γ_♭ = T(H)/T_cred`, which says what the floor `♭` is against the honest kernel (W1).
It is what the physical layer's ledger consumes, never a credit by itself: the ledger admits a set of certificates on
one device only if the credit inside every interval of its ticks fits in it, so one device's time is credited once. -/
structure Certificate where
  window : Accounting.Window
  units : Kind → ℕ
  credit : ℝ
  eps : ℝ
  delta : ℝ
  gamma : ℝ

/-- **The certificate holds for a play**: its accepted runs take less than `(1 − eps)·credit` with probability at
most `delta`. -/
def Certificate.Holds {Ω : Type} [Fintype Ω] {S : ℕ} {K T L U : Type} (c : Certificate) (P : Play Ω S K T L U) :
    Prop :=
  prob (fun x => P.accepts x ∧ (P.run x).run.time < (1 - c.eps) * c.credit) ≤ c.delta

/-- **The circuit's certificate** in device window `win`: its words per kind, `T_cred`, `ε` at the logical layer's
bound `r`, the given `δ`, and `γ_♭` against honest kernel `H`. -/
noncomputable def certOf {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (H : Honest K)
    (lam : R → ℝ) (r δ : ℝ) (win : Accounting.Window) : Certificate where
  window := win
  units := kindCount Cd
  credit := creditOf Sc Cd Z
  eps := epsOf Sc Cd Z lam r
  delta := δ
  gamma := Z.gammaFloor H (floorOf Sc Cd)

end Accounting.Compose
