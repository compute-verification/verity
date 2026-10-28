import Definitions.Pouw.PearlC.Tile
import Definitions.Core.Partition

/-!
# The tile check as a circuit, and what it computes (trusted; L3's statement)

A proof system's accepted run on a tile (`TileGame.good_of_accept`) has to end on a good or blank tile of `τ`. Sampled
proofs gives less: every drawn proof unit of the tile-check Program is correct in the witness `X`, and `X` holds the
registered values. L3 is the step between: the tile check computes the good-tile predicate. It is stated here over a
**tile check** (`TileCheck`), the Program as a core circuit for each unit shape (`Verity.Circuit`, cut into proof units
by a `Verity.Partition`), with

* `out sh rows cols`: the wires a tile is checked at, its words' C̃ and U (under hiding, its tile digest). The tile's
  proof covers the units of their cone (`Verity.Partition.cone`): in `pc8` and `pc4`, the tile unit, the row units of
  its rows and the row units of its columns, `plan.tile_layout`'s closure;
* `pub sh`, the public input wires, at `anchorsOf H s U u act`: the anchors E_A, E_B, F_A, F_B and G of unit `u`, which
  the verifier derives from public seeds and so never reads from the prover. That the verifier's public inputs are
  these is L8, outside L3;
* `anchorsOK H s U u act`: the verifier's own check of those anchors, before it reads the witness. `pc8`'s refuses a
  noise line with a NaN code, as Python's verifier does (`fp8_to_f32` refuses one); the honest draw never makes one,
  which is completeness, not L3;
* `reg sh`, the registered wires, at `rowsOf U u c`: the values unit `u`'s commitment `c` gives them, its activation
  rows, its weight's rows and its tiles' checked words (under hiding, their digests). These are `valueAt`, the values
  `τ`'s commitment binds. Some are outputs of a unit (a tile's words or digest), so the agreement sampled proofs gives
  has to cover registered outputs, not only the circuit's inputs (`Verity.Circuit.InputsAgree`);
* `Collides`: the event outside which the registered values bind the words the circuit computes. It is `False` for the
  plain Program, whose checked words are registered wires. The hiding Program registers a tile by its TurboSHAKE128 leaf
  (`pouw/tile-h0/v1`), so there it is the circuit's words differing from `c`'s under the same leaf.

The check also asks each of the tile's weight rows to pass the protocol's `weightOK`, as `check_tile`'s `admissible`
does and as `TileGood` asks. It decides that on the rows it opens against the weight's root, which are `U`'s, so no
prover supplies it.

The cap, `TileGood`'s first conjunct, reads only the committed activations; its weight rows, the second, are the check's
own last conjunct; the words read the circuit's outputs. So L3 is the cap and the words (`TileCheck.Computes`,
`TileCheck.ComputesCap`, `TileCheck.ComputesWords`). L3 is exact: the Program
recomputes A′, B̃, P_A, P_B, C̃ and U from the anchors and checks no projection, so the tile check has no soundness error
of its own (L7). `εPearlC` stays TT_OUT's, and a proof system's error stays in its `η`.

A tile check says something only when it is the deployed Program: a circuit nobody tied to the Program meets L3 with
any `op`s. The tie is the Program's descriptor digest and a test against `verity.ir`, beside the instance.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- A tile check: the tile-check Program as a core circuit for each unit shape, its partition into proof units, the
wires a tile is checked at, the public wires with the anchors the verifier derives and its check of them, the registered
wires with the values a commitment gives them, and the event outside which those values bind the circuit's words (see
the module docstring). -/
structure TileCheck (Q R S : Type) where
  C : Shape → Verity.Circuit
  n : Shape → ℕ
  part : (sh : Shape) → Verity.Partition (C sh) (n sh)
  out : (sh : Shape) → Finset ℕ → Finset ℕ → Finset (Fin (C sh).N)
  pub : (sh : Shape) → Finset (Fin (C sh).N)
  reg : (sh : Shape) → Finset (Fin (C sh).N)
  anchorsOf : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes → (C (U.layout.shape u)).CommitLog
  anchorsOK : (Q → R) → S → Workload → ℕ → Codes → Prop
  rowsOf : (U : Workload) → (u : ℕ) → Commit → (C (U.layout.shape u)).CommitLog
  Collides : (U : Workload) → (u : ℕ) → Finset ℕ → Finset ℕ → Commit → (C (U.layout.shape u)).CommitLog → Prop

/-- The registered values of unit `u` in `τ`: its commitment's, and zero for a blank unit, whose tiles are never
checked. -/
def TileCheck.valueAt (TC : TileCheck Q R S) (U : Workload) (τ : CommitLog) (u : ℕ) :
    (TC.C (U.layout.shape u)).CommitLog :=
  match τ u with
  | some c => TC.rowsOf U u c
  | none => fun _ => 0

/-- **Tile `g`'s check accepts `X`** for the commitment `c` of its unit: every unit of the cone of the tile's checked
wires is correct in `X`, `X` holds the anchors on the public wires, the anchors pass the verifier's check
(`TC.anchorsOK`), `X` holds `c`'s values on the registered wires, and each of the tile's weight rows passes the rules
(`P.weightOK`). The check decides the anchors' check from the public seeds and the last from the weight rows it opens,
which are `U`'s, never from `X`. -/
def TileCheck.Accepts (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (H : Q → R) (s : S)
    (U : Workload) (g : ℕ) (c : Commit) (X : (TC.C (U.layout.shape ((TR.tiling U.layout).unit g))).CommitLog) : Prop :=
  (∀ v ∈ (TC.part _).cone (TC.out _ ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g)),
      (TC.part _).Correct X v) ∧
    (∀ w ∈ TC.pub _, X w = TC.anchorsOf H s U ((TR.tiling U.layout).unit g) c.act w) ∧
    TC.anchorsOK H s U ((TR.tiling U.layout).unit g) c.act ∧
    (∀ w ∈ TC.reg _, X w = TC.rowsOf U ((TR.tiling U.layout).unit g) c w) ∧
    ∀ j ∈ (TR.tiling U.layout).cols g,
      P.weightOK (U.layout.shape ((TR.tiling U.layout).unit g)) (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j

/-- The words of a good tile: each ticket and useful word of a credited row has the reference value. `TileGood` is the
cap, the tile's weight rows and these. -/
def TileWordsGood (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S) (g : ℕ) (c : Commit) :
    Prop :=
  ∀ i ∈ (TR.tiling U.layout).rows g, TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) c.act i →
    ∀ j ∈ (TR.tiling U.layout).cols g,
      wordQ (c.ticket i j) = P.ticket H s U ((TR.tiling U.layout).unit g) c.act i j ∧
        wordQ (c.useful i j) = P.useful H s U ((TR.tiling U.layout).unit g) c.act i j

/-- **L3: the tile check computes the good-tile predicate.** On every in-domain workload, for every oracle, salt, tile
`g` and commitment `c` of `g`'s unit, a witness that `g`'s check accepts makes `g` good under `c`, unless the
registered values fail to bind the circuit's words (`Collides`). On a transcript `τ` whose unit `(TR.tiling U.layout).unit
g` is `c`, `g` is then in `goodTiles`; a tile of a blank unit is in `blankTiles` whatever the witness. -/
def TileCheck.Computes (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

/-- **L3's cap**: an accepted tile's committed activations keep its replayed debit within the cap. The Program has no
debit or cap yet (the replay shortcut, `PROTOCOL.md`), so its check leaves the activations' debit free: this holds of it
only once a cap unit returns the cap's verdict on a public wire, as under hiding it must, since the verifier never sees
the activations. -/
def TileCheck.ComputesCap (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X → TR.capOK H s U g c.act

/-- **L3's words**: an accepted tile's checked words have the reference values on its credited rows, unless the
registered values fail to bind the circuit's words. -/
def TileCheck.ComputesWords (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (g : ℕ), g < (TR.tiling U.layout).NT → ∀ (c : Commit) X,
    TC.Accepts P TR H s U g c X →
      TileWordsGood P TR U H s g c ∨
        TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X

end Pouw.PearlC
