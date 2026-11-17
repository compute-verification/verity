module

public import Definitions.Compose.Compose
public import Definitions.ResourceAccounting.Gates.Declared
public import Definitions.Pouw.Game.Deadline
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! A served window's public shape, its rounds and forms, the coin each served value is committed before, and what a registered circuit must be to be the served window (`Tie`). -/

@[expose] public section

namespace ResourceAccounting.Served

open ResourceAccounting.Compose

structure Shape where
  steps : ℕ
  rows : ℕ
  tileRows : ℕ
  tileCols : ℕ
  linears : List (ℕ × ℕ)
  deriving DecidableEq, Repr

namespace Shape

variable (L : Shape)

def calls : ℕ := L.linears.length

/-- Linear `w`'s `K`, and `0` past the linears. -/
def kOf (w : ℕ) : ℕ := (L.linears.getD w (0, 0)).1

/-- Linear `w`'s `N`, and `0` past the linears. -/
def nOf (w : ℕ) : ℕ := (L.linears.getD w (0, 0)).2

def unitsOf (w : ℕ) : ℕ := (L.rows / L.tileRows) * (L.nOf w / L.tileCols)

def off (w : ℕ) : ℕ := ((List.range w).map L.unitsOf).sum

/-- The units of one step, `U`. -/
def perStep : ℕ := L.off L.calls

def base (s w : ℕ) : ℕ := s * L.perStep + L.off w

/-- Every one of the window's units is present whatever the requests are. -/
def population : ℕ := L.steps * L.perStep

def stepOf (c : ℕ) : ℕ := c / L.calls

def linearOf (c : ℕ) : ℕ := c % L.calls

/-- A unit's rows and columns divide the slot rows and every linear's `N`, as the H1 tile's lanes need. -/
def Wf : Prop :=
  0 < L.steps ∧ 0 < L.calls ∧ 0 < L.tileRows ∧ 0 < L.tileCols ∧ 4 ∣ L.tileRows ∧ 8 ∣ L.tileCols ∧
    L.tileRows ∣ L.rows ∧
    ∀ w, w < L.calls → 0 < L.nOf w ∧ L.tileCols ∣ L.nOf w ∧ 32 ∣ L.kOf w ∧ 0 < L.kOf w

end Shape

/-- One decoder layer of Llama-3.1-8B's linears in call order: `qkv` (`K` 4096, `N` 4096 + 2·1024), `o`, `gate_up`
(`N` 2·14336) and `down`. -/
def llamaLayer : List (ℕ × ℕ) := [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)]

def llama31_8b : Shape where
  steps := 18
  rows := 64
  tileRows := 32
  tileCols := 32
  linears := (List.replicate 32 llamaLayer).flatten

/-- `beside`: receipts over the committed operands beside the served GEMM's `y`; `zTree`: the fused kernel serves
`dequant(z)` and commits `z`, which receipt units open; `yOpen`: receipt units open the Commit's `y` and check
`dequant(z) = y`; `perCall`: a round a call, whose units output the served `y`. -/
inductive Form
  | beside
  | zTree
  | yOpen
  | perCall
  deriving DecidableEq, Repr

def Form.rounds (L : Shape) : Form → ℕ
  | .perCall => L.steps * L.calls
  | _ => L.steps

def Form.roundOf (L : Shape) : Form → ℕ → ℕ
  | .perCall, c => c
  | _, c => L.stepOf c

/-- The rounds past its own after which form `F` commits an `outZ` or `outY` word: one in `perCall`, whose `y` reads its
round's coin, none in the others. -/
def Form.outLag : Form → ℕ
  | .perCall => 1
  | _ => 0

/-- **A served value's class**:
* `registered`: a row of a linear's B table, or its scale, under the weights' root, the registration root;
* `act`: a row of a call's A table, its slot's s8 row and scale (the empty row in a slot no request is assigned to),
  or a request entry its round admits: a run input under its round's root. Certified serving releases a window's
  requests after its salt frame, so a request entry can't be committed before the salt;
* `pub`: a public input of a unit's Program that no request or product brings, which the verifier derives (the unit's
  position, from the shape; its round's roots, from the round log): a run input committed before the coin of every
  round whose gates read it (`Tie.pub`), never a word Compose credits (`isWord`), and no product's operand or scale,
  since `ProductOf` and `DequantOf` take `act` and `registered` words only. An H1 unit's `position` and `roots` ports
  are of it. Its salt is not: the salt is its round's `coin`, sampled after the root it is keyed by;
* `receipt`: a receipt word of a call's unit;
* `check`: a word that its unit's proof holds to `1` (`PouwH1TileZy_v1`'s `ok`, which the verifier reads through a row
  of known content): computed, committed after its round's coin and by the sample (`Tie.checked`), and read by no gate
  of another unit. `ProductOf`, `DequantOf` and `ValueChecked`'s distinctness read the inputs on which every check
  word of the unit is `1` (`Held`). Under `ValueChecked` it reads no computed gate of another unit, directly or
  through computed gates, and neither does an `outZ` or `outY` word;
* `outZ`: a word of the exact product `z` of a call's unit, committed in a row table (`zTree`, and under `ValueChecked`
  `perCall`): a word the unit computes, or in `zTree` a word of the unit's record, the run input the served kernel
  commits under its round's root, which a `check` word of the unit holds to the unit's product (`Tie.outWords`);
* `outY`: a word of the served `y = dequant(z)` of a call's unit, committed in a row table, the words the Commit's `y`
  root is over (`perCall`, and under `ValueChecked` `zTree`): computed by the unit, or in `zTree` a word of its record,
  as `outZ`. Under `ValueChecked` each is `dq` of an `outZ` word, never a raw product;
* `served`: a word of the Commit's served output `y` of a call, which its unit opens (`yOpen` only);
* `inner`: a value no opening reads, a proof block's own wire (`z` in `beside` and `yOpen`, and the unit's own `z` and
  `y` beside a record it checks);
* `coin`: a round's coin input, the unit's salt port.
A call's admit words have no class: their always-proved units are outside the per-call Program (the module
docstring). -/
inductive Cls
  | registered
  | act
  | pub
  | receipt
  | check
  | outZ
  | outY
  | served
  | inner
  | coin
  deriving DecidableEq, Repr

/-- A receipt's is the earliest it may be committed at, the next round's: `Tie.receipt` lets it go later, by the
sample. `inner` and `coin` values are committed nowhere. -/
def commitCoin (F : Form) : Cls → ℕ → ℕ
  | .registered, _ => 0
  | .act, r => r
  | .pub, r => r
  | .outZ, r => r + F.outLag
  | .outY, r => r + F.outLag
  | .served, r => r
  | .receipt, r => r + 1
  | .check, r => r + 1
  | .inner, _ => 0
  | .coin, _ => 0

/-- A read, not a dependence: what ties the word's value to the product is `ValueChecked`. -/
def Form.Links {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) : Form → ℕ → Prop
  | .beside, _ => True
  | .zTree, i => ∃ j, j < Cd.dag.length ∧ (cls j = .outZ ∨ cls j = .outY) ∧ Cd.unit j = Cd.unit i
  | .yOpen, i => ∃ g j, g < Cd.dag.length ∧ Cd.unit g = Cd.unit i ∧ Cd.Reads g j ∧ cls j = .served
  | .perCall, i => ∃ j, j < Cd.dag.length ∧ (cls j = .outZ ∨ cls j = .outY) ∧ Cd.unit j = Cd.unit i

/-- `zTree` has the link only under `fused`, a premise on code the prover runs that no run-time check reaches. It is an
equality of values: useful-work credit needs it, and it doesn't give that credit. -/
def Form.ServedLink (fused : Prop) : Form → Prop
  | .beside => False
  | .zTree => fused
  | .yOpen => True
  | .perCall => True

/-- `noised` is reachability: the coin input is in the gate's cone, not that the gate's value depends on it. No code
checks the Program facts among its fields, and its commit coins are the code's only if the verifier checks the run's
round log. -/
structure Tie {ι : Type} (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) : Prop where
  calls : Cd.calls = L.steps * L.calls
  rounds : Cd.rounds = F.rounds L
  round : ∀ c, c < Cd.calls → Cd.round c = F.roundOf L c
  inRange : ∀ i, i < Cd.dag.length → Cd.call i < Cd.calls
  live : ∀ r, r < Cd.rounds → Cd.coinMode r = .live
  coin : ∀ i, i < Cd.dag.length → (cls i = .coin ↔ Cd.binding i = some .coin)
  registered : ∀ i, i < Cd.dag.length → cls i = .registered → Cd.binding i = some .registered
  act : ∀ i, i < Cd.dag.length → (cls i = .act ∨ cls i = .served) →
    Cd.binding i = some .run ∧ ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i
  pub : ∀ i, i < Cd.dag.length → cls i = .pub →
    Cd.binding i = some .run ∧ (Cd.dag[i]?).any isWord = false ∧ Cd.commitRound i ≤ Cd.rounds ∧
      ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.commitRound i ≤ Cd.round (Cd.call j)
  computed : ∀ i, i < Cd.dag.length →
    (cls i = .receipt ∨ cls i = .check ∨ cls i = .inner) → Cd.binding i = none
  outWords : ∀ i, i < Cd.dag.length → (cls i = .outZ ∨ cls i = .outY) →
    Cd.binding i = none ∨ (F = .zTree ∧ Cd.binding i = some .run ∧
      ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.call j = Cd.call i ∧ Cd.unit j = Cd.unit i)
  committed : ∀ i, i < Cd.dag.length → Cd.binding i = none →
    (cls i = .inner ↔ (Cd.dag[i]?).any (fun g => g.commit) = false)
  noised : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ ((cls i = .outZ ∨ cls i = .outY) ∧ F = .perCall)) →
    Cd.round (Cd.call i) + 1 ≤ (coinMark Cd).getD i 0
  forms : ∀ i, i < Cd.dag.length → ((cls i = .outZ ∨ cls i = .outY) → F = .zTree ∨ F = .perCall) ∧
    (cls i = .served → F = .yOpen)
  linked : ∀ i, i < Cd.dag.length → cls i = .receipt → F.Links Cd cls i
  commit : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt → cls i ≠ .check →
    cls i ≠ .pub → Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i))
  receipt : ∀ i, i < Cd.dag.length → cls i = .receipt →
    F.roundOf L (Cd.call i) + 1 ≤ Cd.commitRound i ∧ Cd.commitRound i ≤ Cd.rounds
  checked : ∀ i, i < Cd.dag.length → cls i = .check → Cd.commitRound i ≤ Cd.rounds
  readers : ∀ i, i < Cd.dag.length → (cls i = .receipt ∨ cls i = .check) →
    ∀ j, j < Cd.dag.length → Cd.Reads j i → Cd.unit j = Cd.unit i
  inner : ∀ i, i < Cd.dag.length → cls i = .inner → Cd.commitRound i = Cd.cutOf i

def TieCommits : Prop :=
  ∀ (ι : Type) (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls), Tie L F Cd cls → LayerCommitted Cd

def TieLive : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls),
    Tie L F Cd cls → Cd.Live Sc

/-- Reachability, not dependence: `j`'s value may ignore `i`'s (`0·i`). -/
def Feeds {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop := Relation.ReflTransGen (fun a b => Cd.Reads b a) i j

/-- Inputs stop the walk: no input is in another gate's computed cone. -/
def ComputedFeeds {ι : Type} (Cd : Declared ι) (i j : ℕ) : Prop :=
  Relation.ReflTransGen
    (fun a b => Cd.Reads b a ∧ a < Cd.dag.length ∧ Cd.binding a = none ∧ Cd.binding b = none) i j

/-- Gate `i` is declared an s8 word, `[−128, 127]`: the type of an A row's and a B row's words. -/
def S8 {ι : Type} (Cd : Declared ι) (i : ℕ) : Prop :=
  (Cd.dag[i]?).any (fun g => decide (g.lo = -128 ∧ g.hi = 127)) = true

/-- **Unit `u`'s checks hold on input `x`**: every `check` word of the unit is `1`, as its proof holds it when the unit
is sampled (an accepted proof of its Program, whose `ok` row the verifier reads at known content). With no check word in
the unit it holds of every input. -/
def Held {ι : Type} (Cd : Declared ι) (cls : ℕ → Cls) (u : ℕ) (x : ι → ℤ) : Prop :=
  ∀ k, k < Cd.dag.length → cls k = .check → Cd.unit k = u → Gates.val Cd.dag x k = 1

/-- A function-of statement: it holds of any gates with that value. `S8` keeps the typed inputs the served rows' words,
and `Nodup` keeps the `K` products distinct, so one product's work is never credited as `K`. -/
def ProductOf {ι : Type} (L : Shape) (Cd : Declared ι) (cls : ℕ → Cls) (z : ℕ) (as bs : List ℕ) : Prop :=
  as.length = L.kOf (L.linearOf (Cd.call z)) ∧ bs.length = as.length ∧ as.Nodup ∧ bs.Nodup ∧
    (∀ a ∈ as, a < Cd.dag.length ∧ cls a = .act ∧ Cd.call a = Cd.call z ∧ S8 Cd a) ∧
    (∀ b ∈ bs, b < Cd.dag.length ∧ cls b = .registered ∧ S8 Cd b) ∧
    ∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit z) x →
      Gates.val Cd.dag x z = (List.zipWith (· * ·) (as.map (Gates.val Cd.dag x)) (bs.map (Gates.val Cd.dag x))).sum

/-- A function-of statement: a `y` that reads `z` and takes another product's value fails it. That `dq` is the dequant
the served code computes is the Definition's review. -/
def DequantOf {ι : Type} (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) (y z sa sw : ℕ) : Prop :=
  sa < Cd.dag.length ∧ cls sa = .act ∧ Cd.call sa = Cd.call z ∧ ¬ S8 Cd sa ∧
    sw < Cd.dag.length ∧ cls sw = .registered ∧ ¬ S8 Cd sw ∧
    ∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit y) x →
      Gates.val Cd.dag x y = dq [Gates.val Cd.dag x z, Gates.val Cd.dag x sa, Gates.val Cd.dag x sw]

/-- The credited work is the receipts' noised GEMM on the served operands, not necessarily the computation that produced
the served bytes: no check can tell a plain GEMM beside the noised one from a `z` peeled from it. `Feeds` is
reachability; that a receipt is the noised product is its public Program's. -/
def ValueChecked {ι : Type} (L : Shape) (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) : Prop :=
  ∃ (ops : ℕ → List ℕ × List ℕ) (src : ℕ → ℕ × ℕ × ℕ),
    (∀ z, z < Cd.dag.length → cls z = .outZ → ProductOf L Cd cls z (ops z).1 (ops z).2) ∧
    (∀ z z', z < Cd.dag.length → z' < Cd.dag.length → cls z = .outZ → cls z' = .outZ →
      (∀ x, Gates.Typed Cd.dag x → Held Cd cls (Cd.unit z) x → Held Cd cls (Cd.unit z') x →
        Gates.val Cd.dag x z = Gates.val Cd.dag x z') → z = z') ∧
    (∀ y, y < Cd.dag.length → cls y = .outY →
      (src y).1 < Cd.dag.length ∧ cls (src y).1 = .outZ ∧ Cd.unit (src y).1 = Cd.unit y ∧
        DequantOf dq Cd cls y (src y).1 (src y).2.1 (src y).2.2) ∧
    (∀ r, r < Cd.dag.length → cls r = .receipt →
      ∃ y, y < Cd.dag.length ∧ cls y = .outY ∧ Cd.unit y = Cd.unit r ∧
        Cd.commitRound (src y).1 ≤ Cd.rounds ∧ Cd.commitRound y ≤ Cd.rounds ∧
        ∀ t ∈ (ops (src y).1).1 ++ (ops (src y).1).2, Feeds Cd t r) ∧
    (∀ x, Gates.Typed Cd.dag x → ∃ x', Gates.Typed Cd.dag x' ∧ (∀ u, Held Cd cls u x') ∧
      ∀ i g b s, Cd.dag[i]? = some g → g.op = .input b s → cls i ≠ .outZ → cls i ≠ .outY → x' s = x s) ∧
    ∀ w, w < Cd.dag.length → (cls w = .check ∨ cls w = .outZ ∨ cls w = .outY) →
      ∀ g, ComputedFeeds Cd g w → Cd.unit g = Cd.unit w

/-- Necessary, not sufficient: no credit statement reads it, it is of the registered circuit and not of the prover's
kernel, and no gate checks it. -/
def CreditTie {ι : Type} (L : Shape) (F : Form) (dq : List ℤ → ℤ) (Cd : Declared ι) (cls : ℕ → Cls) : Prop :=
  Tie L F Cd cls ∧ ValueChecked L dq Cd cls

def Llama31Shape : Prop :=
  llama31_8b.calls = 128 ∧ llama31_8b.perStep = 86016 ∧ llama31_8b.population = 1548288 ∧
    Form.rounds llama31_8b .perCall = 2304 ∧ llama31_8b.Wf

/-- Every window of the salt is timed from the salt's release, so what a window spends before its own start is inside
the interval. There is no seeded form: a window whose noise the salt fixes before its A rows are committed isn't tied. -/
structure EpochTie {ι Q R S : Type} (L : Shape) (F : Form) (W : ℕ) (Cd : ℕ → Declared ι) (cls : ℕ → ℕ → Cls)
    (release : ℝ) (run : ℕ → Pouw.TimedRun Q R S) : Prop where
  tie : ∀ w, w < W → Tie L F (Cd w) (cls w)
  salted : ∀ w, w < W → (run w).ts = release

/-- Compose's bound reads the largest round's share of the words, not the count, so `1/M` bounds every share only when
the rounds are equal, and `perCall`'s are not. -/
def Llama31Rounds : Prop :=
  (∀ F : Form, F ≠ .perCall → Form.rounds llama31_8b F = 18) ∧ Form.rounds llama31_8b .perCall = 2304

end ResourceAccounting.Served
