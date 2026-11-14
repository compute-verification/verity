module

public import Properties.Pouw.Period
public import Mathlib.Data.Finset.Sort

/-!
# The instance `h1CheckedCreditPeriod`'s witnesses are stated at

`Proofs.Pouw.Period.Witnesses` proves `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod`'s `honest` and
`nontrivial` witnesses (`lean-audit.json`'s `witnesses`), and `Proofs.Pouw.Period.Honest` that the instance meets every
premise (`h1_meets`); the instance they read is here, so their statements read only this tree, the spec and the code.

* The twin's circuit word, scheme and anatomy (`Accounting.SecurityProofs.ComposeMonitoringPeriod.Honest`): one
  committed `S8Dot_v1` word, every kind credited with its operand luck pinned, `r_B = ½` and guess `½`, one word fixed.
* The deployed check's numbers at the all-protocols PoUW leg (`Accept`): the pinned H1 tile's check item, its
  parameters `h1Params`, the loop's honest registration `h1Reg` and one-stage's answer accepting it, `h1Evidence`.
* H1's honest live-salt period (`HonestH1`): 128 committed words of no operands, one a unit, on `N_p = 144` tiles, one
  live round; its windows registering `h1Reg` and answered with the sample's units at `K_d = 24`; its salt the round's
  coin, its openings in `Unit`; and its plays: `play`, accepted on every path, `checkedPlay`, accepted where the deployed
  check accepts every window, and `acceptAll`, the verifier that accepts everything, fixing no word.
-/

@[expose] public section

namespace Accounting.SecurityProofs.ComposeMonitoringPeriod.Honest

open Accounting.TS Accounting.Compose

/-- The word: step `0` of an `S8Dot_v1` body of `K = L = 32` (its `T_1`, a served step), committed. -/
def word : Accounting.Gates.IntGate Unit where
  op := .const 0
  args := []
  lo := 0
  hi := 0
  step := 0
  prim := "S8Dot_v1"
  place := .dot 32 32 [] 0
  commit := true

/-- Every kind credited, its operand luck pinned (`0`), its receipt luck and guess bound `½`. -/
noncomputable def scheme : Scheme Unit where
  floor := fun _ => ⟨fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩
  credited := fun _ => true
  span := fun _ => 0
  form := fun _ => .pinned
  mu := fun _ => 0
  rB := fun _ => 1 / 2
  guess := fun _ => 1 / 2
  seedRate := fun _ => 0
  wholeRate := fun _ => 0

/-- The honest run's anatomy: one fixed word a window. -/
noncomputable def anatomy : Anatomy Unit Unit Unit Unit where
  run := ⟨fun _ => 0, 0, 0⟩
  nword := fun _ _ => 0
  nfold := fun _ => 0
  nfill := fun _ => 0
  words := fun _ => 0
  fixed := 1

end Accounting.SecurityProofs.ComposeMonitoringPeriod.Honest

namespace Properties.Pouw.Period.SecurityProofs.Accept

open PouwCheck
open Lean (Json)
open PouwWindow (Work Sampling)

/-- The pinned H1 tile (`core_check.TILES`' salted `PouwH1Tile_v1`, `K = 64`, `R = 4`, `C = 8`). -/
def h1Tile : Tile := tiles[4]

/-- The leg's check item (`h1_check`): the pinned tile's template and work per tile (`64·32`), the scheme and the
entries bound 1024, no coin mode. -/
def h1Check : CheckItem :=
  { entries := 1024, scheme := PouwCheck.scheme, template := h1Tile.template, tileWork := h1Tile.tileWork }

/-- The leg's check parameters (`h1_params`) in the period's context `context`. -/
def h1Params (context : String) : Params where
  check := h1Check
  context := context
  delta := ⟨1, 1000000⟩
  eps := ⟨1, 2⟩
  gridBits := PouwWindow.gridBits
  query := h1Tile.query
  lo := 1
  hi := 65536
  maxTokens := [2, 2, 2, 2]

/-- What the loop's honest registration fixes: level 39 (128 credited tiles, padded to 144), the sampling item at
`K′ = 24`, and no `W` root. -/
def h1Fixed : Fixed := ⟨Work.at 3 39, ⟨toRat ⟨1, 2⟩, toRat ⟨1, 1000000⟩, 3, 24⟩, none⟩

/-- Its registered items: the verifier's own at level 39 and the fixed length's. -/
def h1ItemList : List (String × Json) :=
  [("check", h1Check.toJson), ("length", PouwWindow.lengthItem [2, 2, 2, 2]), ("root_W", rootWJson none),
   ("sampling", h1Fixed.sampling.toJson), ("work", h1Fixed.work.itemJson h1Check.tileWork)]

def h1Items : Json := Json.mkObj h1ItemList

/-- The loop's honest registration in context `context`, for any record bytes, format, Program and partition: a served
window registering `h1Items`, the population 144 sampled `subset:24`, the check's query and scheme, and no roots. -/
def h1Reg (context format program partition : String) (bytes : ByteArray) : WindowCommitmentRecord where
  bytes := bytes
  format := format
  program := program
  query := h1Tile.query
  partition := partition
  scheme := PouwCheck.scheme
  population := 144
  law := subsetLaw 24
  roots := #[]
  window := ⟨"served", context, h1Items, none, none⟩

/-- The profile the rule fixes for it: the registered Program over one level, the partition's 144 units sampled
`subset:24`, at δ = 10⁻⁶, 24 sampled. -/
def h1Profile (program partition : String) : Core.Integrity where
  programs := #[program]
  prescribed := #[]
  linked := true
  covered := #[]
  levels := #[{ name := unitLevel, query := some partition, size := 144, draw := some (.subset 24) }]
  delta := ⟨1, 1000000⟩
  drawn := #[(unitLevel, 24)]

/-- One-stage's answer accepting registration `c` with the rule's profile and a `subset:24` sample of `units` over 144. -/
def h1Evidence (c : WindowCommitmentRecord) (units : Array Nat) : Evidence where
  protocol := oneStage
  commitment := some (Flock.hexOf c.digest)
  said := .accept (h1Profile c.program c.partition) { law := .subset 24, population := 144, units }

end Properties.Pouw.Period.SecurityProofs.Accept

namespace Properties.Pouw.Period.SecurityProofs.HonestH1

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs.Accept
open PouwCheck (Params)
open Accounting.TS Accounting.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposeMonitoringPeriod

/-! ## The circuit -/

/-- 128 committed words of no operands, one a unit, one live round. -/
def circuit : Declared Unit where
  dag := List.replicate 128 Honest.word
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun _ => 0
  seedNames := fun _ _ => false
  units := 128
  unit := fun i => i

theorem hr1 : 0 < circuit.rounds := Nat.one_pos

theorem hK24 : 24 ≤ 144 := by decide

/-! ## The period, its windows and its proofs -/

/-- A window's coins: its round's coin, the sample's permutation and the proofs' (none). -/
abbrev Ω1 : Type := Paths Bool circuit.rounds × (Equiv.Perm (Fin 144) × Unit)

/-- A window's honest stages: every tile paid, its words right and its checked values honest. -/
def stages : SplitStages Bool (Fin 144) 1 :=
  .stage (fun _ _ => .paid) (fun _ _ => True) (fun _ _ => True) fun _ => .done

/-- The honest period of `n` windows: every tile live, each window's stages `stages`, whatever the past. -/
def period : (n : ℕ) → ExecutionPhasePlay Bool Ω1 (Fin 144) circuit.rounds n
  | 0 => .done
  | n + 1 => .window (fun _ => true) stages fun _ => period n

/-- The check's parameters: `h1Params`. -/
def prm : Params := h1Params ""

/-- Every window's registration: the loop's honest one. -/
def reg : PouwCheck.WindowCommitmentRecord := h1Reg "" "" "" "" ByteArray.empty

/-- The units of the model's sample at permutation coin `β`, ascending. -/
noncomputable def unitsOf (β : Equiv.Perm (Fin 144) × Unit) : Array ℕ :=
  (Finset.sort (Finset.univ.image fun i : Fin 24 => (drawOf hK24 β i).val) (· ≤ ·)).toArray

/-- The period's windows as the check reads them: the honest registration, and the service's answer accepting the
units of the window's sample. -/
noncomputable def windows (n : ℕ) : Executions Ω1 n where
  reg := fun _ _ => reg
  ev := fun x w => h1Evidence reg (unitsOf (coinAt x w).2)

/-- The service's openings, in `Unit`. -/
def opened {n : ℕ} : Paths Ω1 n → Fin n → ℕ → ℕ → Unit := fun _ _ _ _ => ()

/-- A salt's bytes from a round coin. -/
def enc (b : Bool) : ByteArray := ⟨#[if b then 1 else 0]⟩

/-- Each window's salt: its round coin's. -/
def salt {n : ℕ} (x : Paths Ω1 n) (w : Fin n) : ByteArray := enc (coinAt (coinAt x w).1 ⟨0, hr1⟩)

/-- The staged and the flat statements: every opening meets them. -/
def staged : ByteArray → Fin 144 → (ℕ → Unit) → Prop := fun _ _ _ => True

def flat : ByteArray → Fin 144 → (ℕ → Unit) → Prop := fun _ _ _ => True

/-- The period's proofs under the check. -/
noncomputable def run (n : ℕ) : Outcomes Bool (Equiv.Perm (Fin 144) × Unit) circuit.rounds 24 144 n Unit :=
  checkedRun hK24 prm (windows n) opened (stagedAt salt staged) (flatAtCoin hr1 enc flat)

noncomputable def layer (n : ℕ) :
    SampledLayer Bool (Equiv.Perm (Fin 144) × Unit) n circuit.rounds 24 (Fin 144) ℕ (Fin n × ℕ) Unit :=
  checkedLayer circuit hr1 hK24 prm (windows n) opened (stagedAt salt staged) (flatAtCoin hr1 enc flat)

/-- The honest run's anatomy: the 128 words fixed. -/
noncomputable def anatomy : Anatomy Unit Unit Unit Unit := { Honest.anatomy with fixed := 128 }

/-- The play: the period's ideal game at `liveCone`, accepted on every path. -/
noncomputable def play (n : ℕ) : Play Ω1 n Unit Unit Unit Unit where
  game := monitoringPeriodGame (tileStage circuit 144) (liveCone Honest.scheme circuit 144) 1 1 24 (period n)
  accepts := fun _ => True
  run := fun _ => anatomy

/-! ## The witnesses' plays -/

/-- The uniform measure's instance on `64` windows' coins: instance search unfolds `Paths` 64 levels deep and fails. -/
instance (priority := high) fintypePaths64 : Fintype (Paths Ω1 64) := Paths.instFintype Ω1 64

/-- **The deployed check accepts the period**: `PouwCheck.protocol.check` accepts every window on the coins `x`. -/
def checkedAccepts (n : ℕ) (x : Paths Ω1 n) : Prop :=
  ∀ w, Accepts prm ((windows n).reg x w) ((windows n).ev x w)

/-- The honest play, accepted where the deployed check accepts every window. -/
noncomputable def checkedPlay (n : ℕ) : Play Ω1 n Unit Unit Unit Unit :=
  { play n with accepts := checkedAccepts n }

/-- **The verifier that accepts everything**: the honest play's game, accepted on every path, fixing no word. -/
noncomputable def acceptAll (n : ℕ) : Play Ω1 n Unit Unit Unit Unit :=
  { game := (play n).game, accepts := fun _ => True, run := fun _ => { anatomy with fixed := 0 } }

end Properties.Pouw.Period.SecurityProofs.HonestH1
