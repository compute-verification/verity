import Mathlib.Data.Rat.Floor
import Definitions.Pouw.Lead.Gap
import Definitions.Pouw.Game.Defs

/-!
# The lead in PoUW's game (trusted definitions)

The lead's calls as an instance of PoUW's γ-gap game (`Pouw.Gγ`): every unit is a call at one layout `L` of the lead,
at most `N₀` of them, with the data `κ U u` of unit `u` of the workload `U`.

* **The salt** (`Salt`) is every unit's coins, uniform and drawn after preprocessing, as the certifier draws a call's
  coins after the call's tables are fixed.
* **A program** (`LeadProg`) is a strategy for each unit, among those the node's processors run (`can`), chosen as any
  function of the other units' coins: a unit's tables are fixed before its own coins, never before the others'.
* **The run** (`costModel`) commits unit `u` when the lead's verifier accepts its call (`Accepted`), as an ideal
  commitment that is correct by construction (`unitCommit`), and leaves a refused call blank. It costs the units'
  priced W1 work (`unitWork`), makes no oracle call, and hashes with `sha` what each unit's execution hashed and what
  the honest commitment of its words hashes (`unitHashed`), the messages the lead's `Break` reads.
* **The protocol** (`protocol`) compares no checked value, so a unit is correct exactly when its call is accepted, and
  declares `W_ref = ⌊32 · honestRegs⌋`, the honest priced work, for every unit.
* **The gap** `gamma` is `1 − 32 · (honestRegs − gapFixed) / W_ref`, so `(1 − gamma) · W_ref` is the work below which
  `G_tail` bounds a call's acceptance, and the error `eta` is `N · etaLead`, a union bound over the units.
-/

namespace Pouw.Lead.Game

open Pouw.Dimension.H100

variable (L : Pouw.Lead.Layout) (N₀ : ℕ)

/-- The salt: every unit's coins. -/
abbrev Salt : Type := Fin N₀ → Coins L

/-- The coins of every unit but `i`. -/
def rest {L : Pouw.Lead.Layout} {N₀ : ℕ} (s : Salt L N₀) (i : Fin N₀) : {j : Fin N₀ // j ≠ i} → Coins L :=
  fun j => s j

/-- **A program**: a strategy for each unit, a function of the other units' coins, each one the node's processors
run. -/
structure LeadProg (can : Strategy L → Prop) where
  /-- Unit `i`'s strategy, given the other units' coins. -/
  σ : (i : Fin N₀) → ({j : Fin N₀ // j ≠ i} → Coins L) → Strategy L
  /-- Each is one the node's processors run. -/
  can_σ : ∀ i r, can (σ i r)

/-- Unit `i`'s strategy at the salt `s`. -/
def LeadProg.play {L : Pouw.Lead.Layout} {N₀ : ℕ} {can : Strategy L → Prop} (p : LeadProg L N₀ can) (s : Salt L N₀)
    (i : Fin N₀) : Strategy L :=
  p.σ i (rest s i)

variable {L N₀}

open Classical in
/-- Unit `u`'s commitment at the salt `s`: when its call is accepted, an ideal one, correct by construction; else a
blank. Past `N₀`, a blank. -/
noncomputable def unitCommit (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop} (p : LeadProg L N₀ can)
    (U : Pouw.Workload) (s : Salt L N₀) (u : ℕ) : Option Pouw.Commit :=
  if h : u < N₀ then
    if Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩) then
      some ⟨0, fun _ => 0, mm (U.layout.shape u).k 0 (U.weight (U.layout.wid u))⟩
    else none
  else none

/-- Unit `u`'s priced W1 work at the salt `s`; zero past `N₀`. -/
def unitWork (prc : Prices100) {can : Strategy L → Prop} (p : LeadProg L N₀ can) (s : Salt L N₀) (u : ℕ) : ℕ :=
  if h : u < N₀ then (p.play s ⟨u, h⟩).work prc (s ⟨u, h⟩) else 0

/-- What unit `u`'s execution hashed at the salt `s` and what the honest commitment of its words hashes. -/
def unitHashed (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop} (p : LeadProg L N₀ can)
    (U : Pouw.Workload) (s : Salt L N₀) (u : ℕ) : List Hash.Msg :=
  if h : u < N₀ then
    ((p.play s ⟨u, h⟩).exec (s ⟨u, h⟩)).hashed ++ (C.msgs sha (word L (κ U u) (s ⟨u, h⟩)) ++ C.aMsgs)
  else []

variable (L N₀)

/-- **The lead's cost model** in PoUW's game, over any oracle `Q → R`, which it never reads. -/
noncomputable def costModel (Q R : Type) (prc : Prices100) (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) (can : Strategy L → Prop) :
    Pouw.CostModel Q R (Salt L N₀) where
  Prog := LeadProg L N₀ can
  run p _ U _ _ s := unitCommit sha C κ p U s
  cost p _ U _ _ s := ∑ u ∈ Finset.range U.layout.N, (unitWork prc p s u : ℝ)
  calls _ _ _ _ _ _ := 0
  sha := sha
  hashed p _ U _ _ s := (List.range U.layout.N).flatMap (unitHashed sha C κ p U s)

/-- **The lead's protocol** in PoUW's game: no checked value is compared, and every unit's `W_ref` is
`⌊32 · honestRegs⌋`. -/
def protocol (Q R : Type) (C : Pouw.Lead.Commit L) : Pouw.Protocol Q R (Salt L N₀) where
  Y _ _ _ _ _ := fun _ => 0
  checkIdx _ := ∅
  refChecked _ _ _ _ _ := fun _ => 0
  refOut _ _ U u A := mm (U.layout.shape u).k A (U.weight (U.layout.wid u))
  Wref _ _ := ⌊32 * honestRegs L C⌋₊

/-- The domain: at most `N₀` units. -/
def domain : Pouw.Layout → Prop := fun Lay => Lay.N ≤ N₀

variable {N₀}

/-- The gap: `(1 − gamma) · W_ref = 32 · (honestRegs − gapFixed)`. -/
noncomputable def gamma (C : Pouw.Lead.Commit L) : ℝ :=
  1 - ((32 * (honestRegs L C - gapFixed L C) : ℚ) : ℝ) / (⌊32 * honestRegs L C⌋₊ : ℝ)

/-- The error at `N` units: `N · etaLead`. -/
noncomputable def eta : ℕ → ℕ → ℝ := fun _ N => N * etaLead L

end Pouw.Lead.Game
