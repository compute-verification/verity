import Definitions.Accounting.TS
import Definitions.Accounting.Gates.Model
import Definitions.Accounting.Device
import Definitions.Pouw.Window

/-!
# PoUW as a compile pass over an arbitrary declared circuit (trusted)

The vocabulary of `Models.Accounting.Guarantees.Compose`
(note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2), restated after RT-COMPOSE
(note:compute-accounting/20261018T0501Z-draft-rt-compose) and RT-I8 (note:compute-accounting/20261018T0728Z-draft-rt-i8)
under decisions 107, 109–114, 117 and 119. The compile pass `C → C′`
replaces each credited unit of a declared circuit by its kind's template, with its noise drawn after its operands are
committed, one seed a Definition call site (decisions 101 and 102). The service then draws one uniform subset of the
window's grid-padded tiles and proves each drawn tile with the generation of its own noise.

Decision 113 splits it in two layers. The logical layer, from `Declared` to `WorkAt`, is the circuit, its units, the
templates' proved luck, the service's draw and what an accepted window establishes about the run's committed words; it
reads no machine model. The physical layer, from `creditOf` to `certOf`, is the tensor–scalar model on a data sheet,
which prices those words as time on a declared device.

The logical layer:

* `Declared`: the circuit as the IR gives it, a `Accounting.Gates.Dag` (memory accounting's form, with its wiring), and
  each gate's Definition call site; a call site is a subcircuit, the stage its seed is released at. `Declared.Wf`: every
  gate's call site is one of the circuit's.
* `runDep`, `isUnit`, `unitGates`, `Declared.N`, `Declared.gateOf`: the circuit's units, its committed words of served
  products, in order (decision 107: an output coordinate, or its chunk `T_t` of `L` products).
* `Kind`, `kindOfGate`, `Declared.kindOf`: a unit's kind, computed from the IR, never asserted: its primitive, its
  site's statics (what `Gates.roleOf` reads) and whether each operand depends on a run input (an activation) or not (a
  weight). So a gather-fed `QKᵀ` and a weight matmul of one shape are different kinds.
* `Declared.stageOf`: a unit's stage, counted from the circuit's last call site, as `CoinStages` counts.
* `LuckForm`, `Scheme`, `Scheme.LuckProved`: per kind, the floor its template's noise proves at the worst operands
  (decision 109: every matmul is credited at that floor, a parameter, `0` where its noise proves none), the committed
  words that floor prices, the skip pattern's word span before a checkpoint (W6, #2447), and per mode its luck and the
  form of the Lean guarantee that bounds it (B2: `Scheme.luck` is a field, at least its template's proved value).
* `wordsOf`, `weightOf`, `floorOf`, `kindCount`: the circuit's committed words, a unit's work weight, the circuit's
  floor `♭ = Σ_u ♭_{kind u}` and its units per kind.
* `Declared.unitAt`, `tileWords`, `tileWeight`, `tileStage`, `tileLuck`, `tileLuck0`: the service's `Np` grid-padded
  tiles (`verity/core/protocols/pouw/window.py`): the circuit's units first, then fillers of no words and weight `0`,
  which the work excludes. A unit of weight `0` has luck `1`, so no premise asks anything of it.
* `Mode`, `SplitStages`, `modeAt`, `rightAt`, `SplitStages.LuckAt`, `share`, `stageShare`: a window's stages under
  per-call-site coins, with B1's split: a unit is paid, folded (its product skipped and its committed words folded, so
  a drawn one passes when its receipt is lucky) or whole (product and fold skipped, so a drawn one passes only when its
  guessed fold state is right).
* `serviceGame`, `drawOf`: the window as one round on coins `Paths C M × (Equiv.Perm (Fin N) × Ξ)`: the call sites'
  coins, then the service's one uniform `K`-subset (`Pouw.Window.subsetDraw`, its `Equiv.Perm`) and the proofs' coins
  `Ξ` (W4: `DrawnLayer.JointSound` needs them). Its share is `a·(folded weight) + b·(whole weight)` at the weights it is
  given, the tiles' cones (`tileCone`) in the theorem, and it passes when every drawn tile is right.
* `StageCuts`, `FeedsAfter`, `coneWt`, `tileCone`, `TileSound`, `ConeGood`, `PaidSound`: the window's proofs
  (`Accounting.TS.DrawnLayer`, values per committed position) bind each position before the coin after the call site
  that wrote it (RT-I8's B1, decision 114); a wrong tile spreads only along positions committed after the reader's
  coin, so a tile is good when every tile upstream along those edges is sound, and the share charges a skipped tile its
  cone (decision 119, RT-I8 r2's `covers_of_cone`); the modes are the run's: a paid tile is sound.
* `SplitRate`: the per-stage condition from which a rate `Λ` bounds the window's saving per unit of detection.
* `Scope`, `WorkAt`: the logical parameters' ranges, and the logical layer's conclusion on an accepted run: all but a
  share `d` of the committed words fixed right and all but `d₀ ≤ d` folded, with `a·(d − d₀) + b·d₀ < r`.

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

/-! ## The declared circuit and its units -/

/-- **A declared circuit**: its gates in the IR's canonical order (`Accounting.Gates.Dag`: each gate's operation,
operands, interval, primitive, site and whether it is committed), and the Definition call site each gate belongs to,
as the IR binds it: gate `i` was made by call `call i` of `calls`, numbered in the circuit's order. A call site is a
subcircuit: one noise seed, released after its operands are committed. -/
structure Declared (ι : Type) where
  dag : Gates.Dag ι
  calls : ℕ
  call : ℕ → ℕ

/-- Every gate's call site is one of the circuit's `calls`. -/
def Declared.Wf {ι : Type} (Cd : Declared ι) : Prop := ∀ i, i < Cd.dag.length → Cd.call i < Cd.calls

/-- Whether each gate's value depends on a run input, in order: a `run` input, or a gate that reads such a value.
Unlike `Gates.dynamic` it leaves coins out, so it tells an activation (a request's tokens and what they reach) from a
weight or a codec's parameters. -/
def runDep {ι : Type} (C : Gates.Dag ι) : List Bool :=
  C.foldl (fun ds g => ds ++ [match g.op with
    | .input b _ => b == .run
    | _ => g.args.any fun j => ds.getD j false]) []

/-- **A unit**: a committed gate on a served product's main chain, its committed word `T_t` (decision 107: an output
coordinate, or its `t`-th chunk of `L` products when the prefix `T_1 … T_{K/L}` is committed). -/
def isUnit {ι : Type} (g : Gates.Gate ι) : Bool := g.commit && g.role.isServed

/-- The circuit's units, as gate indices in the circuit's order. -/
def unitGates {ι : Type} (Cd : Declared ι) : List ℕ :=
  (List.range Cd.dag.length).filter fun i => Cd.dag[i]?.any isUnit

/-- The circuit's number of units. -/
def Declared.N {ι : Type} (Cd : Declared ι) : ℕ := (unitGates Cd).length

/-- Unit `v`'s gate. -/
def Declared.gateOf {ι : Type} (Cd : Declared ι) (v : Fin Cd.N) : ℕ := (unitGates Cd).getD v.val 0

/-- **A unit's kind**, read off the IR: its primitive's id, its site's statics (the step zeroed: `K`, `L` and the tail
groups, what `Gates.roleOf` reads) and whether its `a` and `b` operands depend on a run input. -/
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

/-- **Unit `v`'s kind**, its gate's (`kindOfGate`). -/
def Declared.kindOf {ι : Type} (Cd : Declared ι) (v : Fin Cd.N) : Kind :=
  (Cd.dag[Cd.gateOf v]?).elim ⟨"", .other, false, false⟩ (kindOfGate Cd.dag)

/-- **Unit `v`'s stage**: the call sites after its own, so the circuit's last call site is stage `0`. -/
def Declared.stageOf {ι : Type} (Cd : Declared ι) (v : Fin Cd.N) : ℕ := Cd.calls - 1 - Cd.call (Cd.gateOf v)

/-! ## Schemes: floors, words and proved luck per kind -/

/-- **The form of a template's proved luck**, at a skip pattern's word span:
* `cube bind`: `¾` while the span is within the receipt's binding `bind` (`Accounting.Guarantees.TS.LuckCaseACube`,
  `OperandLuckCube`, for degree-2 cube operands; #2447's `Pouw.Guarantees.Receipt.DuoWindowNoKernel` binds four
  words), and `1` past it (#2447's `DuoLongRunKernel`: no receipt binds every length);
* `coins b Z`: `Z(span)·C(b, ⌊b/2⌋)/2^b` for `b` fresh bits at the fibre `Z(span)` the receipt has at that span
  (`Accounting.Guarantees.HMu.CoinPointMass`, `ReceiptLuckOfCoins`);
* `pinned`: `0`, a value the proof pins (`Accounting.Guarantees.FillerLuck.PinnedFillLuckAt`, not yet on main);
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

/-- **A scheme's templates**, per kind: one unit's floor `♭_k` over the classes `K`, proved at the worst operands
(decision 109; `0` where the noise proves none, as one-sided noise proves none; decision 117: H1's attention matmuls
draw a fresh coin per entry on both operands, so their floor reads no operand and they are credited at it), the
committed words that floor prices (`0` with it), the span a skip pattern has before the receipt's next checkpoint
(W6), and for each mode, folded and whole, the form of the luck its template proves and the luck the window is held to
(`luck`, `luck0`). -/
structure Scheme (K : Type) where
  floor : Kind → Labelled K
  words : Kind → ℕ
  span : Kind → ℕ
  form : Kind → LuckForm
  form0 : Kind → LuckForm
  luck : Kind → ℝ
  luck0 : Kind → ℝ

/-- **The luck is at least what each template proves**: per kind and mode, the luck the window is held to is at least
its form's value at the kind's span, so (H-μ) (`Assumptions.Compose.HMu`) asks no more of the cheater's view than the
template's own coins give. -/
def Scheme.LuckProved {K : Type} (Sc : Scheme K) : Prop :=
  ∀ k, (Sc.form k).value (Sc.span k) ≤ Sc.luck k ∧ (Sc.form0 k).value (Sc.span k) ≤ Sc.luck0 k

/-- The circuit's committed words that its floor prices: `N_w = Σ_u words(kind u)`. -/
def wordsOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : ℕ := ∑ v : Fin Cd.N, Sc.words (Cd.kindOf v)

/-- **A unit's work weight**: its kind's words over the circuit's. -/
noncomputable def weightOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (v : Fin Cd.N) : ℝ :=
  (Sc.words (Cd.kindOf v) : ℝ) / (wordsOf Sc Cd : ℝ)

/-- **The circuit's floor** `♭ = Σ_u ♭_{kind u}`, part by part. -/
noncomputable def floorOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) : Labelled K where
  nw k := ∑ v : Fin Cd.N, (Sc.floor (Cd.kindOf v)).nw k
  nf k := ∑ v : Fin Cd.N, (Sc.floor (Cd.kindOf v)).nf k
  nx k := ∑ v : Fin Cd.N, (Sc.floor (Cd.kindOf v)).nx k
  ns k := ∑ v : Fin Cd.N, (Sc.floor (Cd.kindOf v)).ns k

/-- The circuit's units of kind `k`: the work of that kind a certificate reports. -/
def kindCount {ι : Type} (Cd : Declared ι) (k : Kind) : ℕ :=
  (Finset.univ.filter fun v : Fin Cd.N => Cd.kindOf v = k).card

/-! ## The service's grid-padded tiles -/

/-- Tile `t` of `Np`: unit `t` of the circuit when `t < N`, and a filler otherwise (`window.py` pads the work to one
grid level with fillers, which weigh `0`). -/
def Declared.unitAt {ι : Type} (Cd : Declared ι) {Np : ℕ} (t : Fin Np) : Option (Fin Cd.N) :=
  if h : t.val < Cd.N then some ⟨t.val, h⟩ else none

/-- A tile's committed words: its unit's kind's, and `0` for a filler. -/
def tileWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (Cd.unitAt t).elim 0 fun v => (Sc.words (Cd.kindOf v) : ℝ)

/-- A tile's work weight: its unit's, and `0` for a filler. -/
noncomputable def tileWeight {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (Cd.unitAt t).elim 0 (weightOf Sc Cd)

/-- A tile's stage: its unit's, and `0` for a filler. -/
def tileStage {ι : Type} (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℕ := (Cd.unitAt t).elim 0 Cd.stageOf

/-- A tile's luck when folded: its kind's `luck` when it carries work, and `1` otherwise. -/
noncomputable def tileLuck {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (Cd.unitAt t).elim 1 fun v => if 0 < weightOf Sc Cd v then Sc.luck (Cd.kindOf v) else 1

/-- A tile's luck when whole: its kind's `luck0` when it carries work, and `1` otherwise. -/
noncomputable def tileLuck0 {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) : ℝ :=
  (Cd.unitAt t).elim 1 fun v => if 0 < weightOf Sc Cd v then Sc.luck0 (Cd.kindOf v) else 1

/-! ## The window: stages with B1's split, and the service's draw -/

/-- A unit's mode in a cheater's run: paid; folded (its product skipped, its committed words folded); or whole (its
product and its fold skipped). -/
inductive Mode
  | paid
  | folded
  | whole
  deriving DecidableEq

/-- **A window's stages under per-call-site coins**: per stage, each unit's mode and whether its committed values are
right, as functions of the stage's coin `C` (the stage's outputs are committed after it is released), then the rest of
the window, which may depend on it. A unit belongs to one stage, `stg u`, counted from the last (stage `0`). -/
inductive SplitStages (C V : Type) : ℕ → Type
  | done : SplitStages C V 0
  | stage {m : ℕ} (mode : C → V → Mode) (right : C → V → Prop) (next : C → SplitStages C V m) :
      SplitStages C V (m + 1)

namespace SplitStages

variable {C V : Type}

/-- Unit `u`'s mode along the stages' coins: its stage's, and paid when no stage holds it. -/
def modeAt (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Paths C m → V → Mode
  | _, .done, _, _ => .paid
  | _, .stage (m := m) mode _ next, x, u => if stg u = m then mode x.1 u else modeAt stg (next x.1) x.2 u

/-- Unit `u`'s committed values are right along the stages' coins: its stage says so. -/
def rightAt (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Paths C m → V → Prop
  | _, .done, _, _ => True
  | _, .stage (m := m) _ right next, x, u => if stg u = m then right x.1 u else rightAt stg (next x.1) x.2 u

/-- **The luck per unit at its stage's coin**: at every stage, whatever the past, a unit of the stage that is folded is
right with probability at most `p u` times that of being folded, and one that is whole with `p0 u`. -/
def LuckAt [Fintype C] (p p0 : V → ℝ) (stg : V → ℕ) : {m : ℕ} → SplitStages C V m → Prop
  | _, .done => True
  | _, .stage (m := m) mode right next =>
      (∀ u, stg u = m →
        prob (fun c => mode c u = .folded ∧ right c u) ≤ p u * prob (fun c => mode c u = .folded) ∧
          prob (fun c => mode c u = .whole ∧ right c u) ≤ p0 u * prob (fun c => mode c u = .whole)) ∧
        ∀ c, LuckAt p p0 stg (next c)

/-- The work in mode `s` along the stages' coins `x`, at work weights `w`. -/
noncomputable def share [Fintype V] {M : ℕ} (stg : V → ℕ) (W : SplitStages C V M) (w : V → ℝ) (x : Paths C M)
    (s : Mode) : ℝ :=
  ∑ u, if W.modeAt stg x u = s then w u else 0

end SplitStages

/-- Stage `j`'s work share at work weights `w`. -/
noncomputable def stageShare {V : Type} [Fintype V] (stg : V → ℕ) (w : V → ℝ) (j : ℕ) : ℝ :=
  ∑ v, if stg v = j then w v else 0

/-- **The window as one round of the service**: on the call sites' coins, the permutation whose first `K` images are
the drawn tiles (`Pouw.Window.subsetDraw`) and the proofs' coins `Ξ`, its share is `a` per unit of folded work and `b`
per unit of whole work, and it passes when every drawn tile's committed values are right. -/
noncomputable def serviceGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M)
    (w : Fin N → ℝ) (a b : ℝ) (K : ℕ) : RoundGame (Paths C M × (Equiv.Perm (Fin N) × Ξ)) 1 :=
  .round (fun ω => a * W.share stg w ω.1 .folded + b * W.share stg w ω.1 .whole)
    (fun ω => ∀ v ∈ Pouw.Window.subsetDraw K ω.2.1, W.rightAt stg ω.1 v) fun _ => .done

/-- **The service's draws**: draw `i` is the permutation's image of `i`, so the `K` draws are `subsetDraw K`. -/
def drawOf {N K : ℕ} {Ξ : Type} (hK : K ≤ N) (β : Equiv.Perm (Fin N) × Ξ) (i : Fin K) : Fin N :=
  β.1 (Fin.castLE hK i)

/-- **Each position is bound before the coin after the call site that wrote it** (RT-I8's B1 for the window, decision
114): a position of the circuit's inputs before the first call site's coin, and a tile's output before the coin of the
call site after its own, the draw after the last. Stage `j`, counted from the last, is coin `M − 1 − j` of the window's
`M`, so the coin after it is `M − j`. A value opened one way for the tile that wrote it and another for a tile that
reads it, after the reader's coin, is then a binding failure (`DrawnLayer.MovedIn`). -/
def StageCuts {C B Q Pos V : Type} {M s N : ℕ} (stg : Fin N → ℕ) (pf : DrawnLayer C B 1 M s (Fin N) Q Pos V) :
    Prop :=
  ∀ ℓ, ((pf.cut ℓ).2 : ℕ) = (pf.src ℓ).elim 0 fun u => M - stg u.2

/-- **Tile `u` feeds tile `v` after `v`'s coin** (decision 119): `v` reads a position `u` writes that is committed after
the coin `v`'s noise reads, its call site's, coin `M − 1 − stg v` of the window's `M`. Only along these edges can a
wrong tile leave a later one's words adaptive: a position committed before a tile's coin is fixed when its noise is
drawn. Under `StageCuts` they are the edges within a call site, decision 107's `K`-chunk running sums. -/
def FeedsAfter {C B Q Pos V : Type} {M s N : ℕ} (stg : Fin N → ℕ) (pf : DrawnLayer C B 1 M s (Fin N) Q Pos V)
    (u v : Fin N) : Prop :=
  ∃ k, pf.src (pf.pos 0 v k) = some (0, u) ∧ M - 1 - stg v < ((pf.cut (pf.pos 0 v k)).2 : ℕ)

open Classical in
/-- **Tile `u`'s cone weight** along edges `R`: the weight of every tile `u` reaches, itself included. -/
noncomputable def coneWt {N : ℕ} (R : Fin N → Fin N → Prop) (w : Fin N → ℝ) (u : Fin N) : ℝ :=
  ∑ v, if Relation.ReflTransGen R u v then w v else 0

/-- **A tile's share weight** (decision 119): its cone's work weight along `FeedsAfter`, what skipping it can leave
adaptive. It is at most `F` times its own weight when a call site's chains have at most `F` tiles, each of at most its
weight (`F` the chunks per coordinate: `1` for a whole coordinate, `4` at `K = 8192`). -/
noncomputable def tileCone {ι K C B Q Pos V : Type} {s Np : ℕ} (Sc : Scheme K) (Cd : Declared ι)
    (pf : DrawnLayer C B 1 Cd.calls s (Fin Np) Q Pos V) : Fin Np → ℝ :=
  coneWt (FeedsAfter (tileStage Cd Np) pf) (tileWeight Sc Cd Np)

/-- **Tile `t` is sound on `x`**: its slots are held at their committed values, and it is right at them. -/
def TileSound {C B Q Pos V : Type} [Fintype C] [Fintype B] [Nonempty V] {M s N : ℕ}
    (pf : DrawnLayer C B 1 M s (Fin N) Q Pos V) (x : Paths (Paths C M × B) 1) (t : Fin N) : Prop :=
  (∀ k, pf.Held x (pf.pos 0 t k)) ∧ pf.right x 0 t (pf.view x 0 t)

/-- **Tile `t` is good on `x`** (decision 119's `Good`, drawn at each coin boundary): every tile upstream of it along
`FeedsAfter`, itself included, is sound. #2466's `DrawnLayer.Good` spreads along every edge until it restates `Good`
this way. -/
def ConeGood {C B Q Pos V : Type} [Fintype C] [Fintype B] [Nonempty V] {M s N : ℕ} (stg : Fin N → ℕ)
    (pf : DrawnLayer C B 1 M s (Fin N) Q Pos V) (x : Paths (Paths C M × B) 1) (t : Fin N) : Prop :=
  ∀ u, Relation.ReflTransGen (FeedsAfter stg pf) u t → TileSound pf x u

/-- **The modes are the run's**: a tile paid along the window's coins is sound, so a tile that isn't is folded or
whole, and the service's share charges it its cone, which holds every tile that isn't good (`ConeGood`). It asks
nothing of a tile upstream: a cheater that skips one tile and computes the rest honestly meets it. -/
def PaidSound {C B Q Pos V : Type} [Fintype C] [Fintype B] [Nonempty V] {M s N : ℕ} (stg : Fin N → ℕ)
    (W : SplitStages C (Fin N) M) (pf : DrawnLayer C B 1 M s (Fin N) Q Pos V) : Prop :=
  ∀ x t, W.modeAt stg (coinAt x 0).1 t = .paid → TileSound pf x t

/-- **The per-stage rate condition** for `K` draws at weight bound `ρ` (each tile is at least `ρ` times its work share
of the `N` tiles), luck `p` folded and `p0` whole, costs `a` and `b` per unit of share and stage shares at most
`w_max`: there are `ν, ν₀ ≥ 0` such that for every stage's folded work `W_F`, whole work `W_0` and their right parts
`L_F`, `L_0`,
`exp((a·W_F + b·W_0)/Λ)·(1 − ρ·((W_F − L_F) + (W_0 − L_0)))^K ≤ 1 + ν·(L_F − p·W_F) + ν₀·(L_0 − p0·W_0)`. -/
def SplitRate (K : ℕ) (ρ p p0 a b wmax Λ : ℝ) : Prop :=
  ∃ ν ν0 : ℝ, 0 ≤ ν ∧ 0 ≤ ν0 ∧ ∀ WF W0 LF L0 : ℝ, 0 ≤ LF → LF ≤ WF → 0 ≤ L0 → L0 ≤ W0 → WF + W0 ≤ wmax →
    Real.exp ((a * WF + b * W0) / Λ) * (1 - ρ * ((WF - LF) + (W0 - L0))) ^ K ≤
      1 + ν * (LF - p * WF) + ν0 * (L0 - p0 * W0)

/-! ## The logical layer's conclusion -/

/-- **The logical parameters' ranges**: the circuit's call sites well formed, `δ_s > 0`, and every kind's luck at least
its template's proved value. -/
def Scope {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (δs : ℝ) : Prop := Cd.Wf ∧ 0 < δs ∧ Sc.LuckProved

/-- **What an accepted window establishes about its run** (decision 113's logical layer): of its `N_w` committed words,
all but a share `d_F + d₀` are fixed right (`fixed`) and all but `d₀` are folded (`folded`), the skipped share split
into the folded units' `d_F` and the whole units' `d₀` (B1), with `a·d_F + b·d₀ < r`. At `a = b = 1` it says the
units right are more than a share `1 − r` of the work. -/
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

/-- **The circuit's `ε`** at the logical layer's bound `r`: `ε_♭ + ε_μ + r`. For a window, `r = Λ·ln(1/δ_s)`, where
`Λ` bounds its time saved per unit of detection (`ServiceBudget`). -/
noncomputable def epsOf (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (lam : R → ℝ) (r : ℝ) : ℝ :=
  epsFlat Sc Cd Z lam + epsMu Sc Cd Z lam + r

/-- **The model is the circuit's on sheet `Sh`**: its machine is the sheet's, and its committed words are the units'
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

/-- **The circuit's certificate** in device window `win`: its units per kind, `T_cred`, `ε` at `r = Λ·ln(1/δ_s)`, the
given `δ`, and `γ_♭` against honest kernel `H`. -/
noncomputable def certOf {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Sc : Scheme K) (Cd : Declared ι) (Z : Model K R T L U F X) (H : Honest K)
    (lam : R → ℝ) (Λ δs δ : ℝ) (win : Accounting.Window) : Certificate where
  window := win
  units := kindCount Cd
  credit := creditOf Sc Cd Z
  eps := epsOf Sc Cd Z lam (Λ * Real.log (1 / δs))
  delta := δ
  gamma := Z.gammaFloor H (floorOf Sc Cd)

end Accounting.Compose
