import Models.Accounting.Guarantees.Stacking

/-!
# A declared circuit's space and bandwidth from an accepted audit (trusted)

Daniel's goal in one statement, `DeclaredCircuitRequirement`: for any declared circuit `W` that passes the stacking
property, any prover, and an accepted audit at the deployed parameters (sketch receipts, a store per pass, the
append-coin rule), except with probability `δ_G` the prover system read, from the blocks of `W`'s values each pass's
step reads, `W`'s credited bits there less an explicit share `ε` of the store. It is a layer-1 bound in decision 111's
sense (a cryptographic lower bound on the prover system, parametric in the system's resources a pass); which device
held what (the node's boundary, HBM against host DRAM) is attribution, stated apart. It composes the stacked remote
audit in its deployed sketch form (`StackedSessionConservationSketchDeployed`), on a layout the prover may steer after
the coins, with the carriers' cost read from the property's caps (`Gates.Property`, under
`Accounting.Assumptions.EncodeChainHardness`) and the audit linked to `W`'s placed requirement (A1's
`Definitions/Accounting/Requirement.lean`) pass by pass, each pass and window on the devices' clocks. Its proof is in
`Security/Proofs/Accounting/DeclaredCircuit.lean`. -/

namespace Accounting.Guarantees

open Accounting
open Pous (pr)
open scoped ENNReal

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the deployed parameters.**

*The audit.* `StackedSessionConservationSketchDeployed`'s session: `n ≥ 7` windows of `P` passes, sketch receipts at
16 rows over tiles of 128 sectors, 512-bit tags, 2,773 openings a window, and a store per pass made by the append-coin
rule (`RateSketch.steeredStore`: the blocks of `T` appends of `M` positions, each append's choices `choose c ω a κ_<a`
fixed before its own coin keys its blocks). Pass `j` of window `w` certifies the `N` blocks at `layout c y`, where `c` is
the session's coins before the pass (`RateSketch.SketchPrefix`) and `y` the codec and every append's coin, so the layout
may follow inputs chosen after those coins: it reads `y` through at most `a` bits (`Accounting.Assumptions.SteeredLayout`).

*The prover system.* Its pass devices are verbatim (`rate-sketch-verbatim`). The system's resources a pass, within
the pass's deadline `τ`, are the class's parameters: at most `S` bits kept besides the store, at most `L_r·τ` bits
taken from outside it, at most `q_r·τ` oracle queries over its grinding candidates
(`Accounting.Assumptions.SketchHashBudget`), and at most `X_r·τ` bits rebuilt besides the sectors of a carrier of the
declared circuit `Wc`'s over at most `life` of the audit `Aud`'s windows (`Accounting.Assumptions.CircuitCarriers`).
`G` holds at every value of them; what they are on a node's devices is layer 2 (data sheets, named assumptions).

*The circuit.* `Wc` is in the restricted class (`Gates.Restricted B`) and passes the stacking property at the caps
`caps` on `Aud` (`Gates.Property`): in each window its budgeted units are within `c = caps.total` a byte `Aud`
certifies, and its run inputs within `R_in = caps.runBytes` bytes. A window certifies at most its passes' stores,
`P·N·4096` bytes.

*The requirement.* The prover chooses everything on `W`'s side after every coin: the placed execution `p x` (each
value's device `leaf` and the tick `ticks d s` at which device `d` begins engine step `s`), the layout of values in
blocks `ℓ x`, the groups `grp x` (a KV value's sequence, layer and plane), the step `bind x w j` each pass is bound to,
strictly increasing over the passes (#2378), and the store position `addr x k` of each block `k`. On device `d`'s
clock pass `j` of window `w` is the `τ` ticks from the one at which `d` begins the pass's step
(`Requirement.Placement.passTicks`), window `w` its passes' ticks (`Requirement.Placement.windowTicks`), and no device
tick is in two windows or two passes of one (`Requirement.Placement.Deadlines`). At each bound step `t` the layout packs
the live values and the values read into blocks of 32,768 bits, the values `t` reads from incomplete blocks lie in one
block a group, and the blocks of `W`'s credited reads at `t` are at distinct tiles of the pass's store. The pass's tiles
of `W` are those tiles, and its credit is the larger of `W`'s credited live bits in those blocks (held) and the bits `t`
reads less a block a group (moved):

  `credit t = max (bitsAt (credited t) (blocks (creditedReads t))) (readBits t − |grp(reads t \ creditedReads t)|·32768)`.

*The bound.* With

  `C = 8·life·R_in + ⌈1024·life·c·P·N·4096/E⌉`, the carriers' bits a pass,
  `γ = (S + (L_r·τ − ρ·256) + X_r·τ + C + k·32768)/(N·32768)`,

the probability that every opening passes while the passes' distinct sectors read from their tiles of `W` fall short of
the credit over the bound steps by more than `(0.0015 + γ)` of the session's `n·P·N·32768` bits, or some window's fall
short of its passes' credit by more than `(0.01 + γ)` of its `P·N·32768`, is at most

  `δ_G = 2·2^-40 + n·ε + n·P·G·N/2^512 + n·P·(2^a·(q_r·τ/2^96)^(k+1) + δ)`.

**The steering.** The layout reads no window coin, so the draws' term and the signs' rank term (`n·ε`) don't see it,
and the tag-guess term holds at every store. The query term is a correct digest of a tile with a sector the device
never obtained, which needs the pass's blocks uniform: key binding gives that per pass at `2^a` times the uniform
probability, `δ` once. A pass short by `k` more tiles than the slack needs `k + 1` such digests at distinct unobtained
sectors, at `2^-96` each, so `k` tiles of slack a pass trade for `k·(96 − log₂(q_r·τ))` bits of steering: about 63
bits a tile at `q_r·τ = 2^33`, the steering costing about 520 bits of `γ`'s numerator a bit. At `a = k = 0` the bound
is `StackedSessionConservationSketchDeployed`'s.

**The reading.** The sectors counted are distinct addresses of the store read in a pass, so the prover system held
those tiles at that pass, in the window's ticks on each device's clock, and the bound steps' credit is `W`'s: held,
`W`'s live bits in the blocks it read then, and moved, what it read then. The shortfall is a share of the store:
relative to `W` it is `ε_W = (0.0015 + γ)·n·P·N·32768/Σ credit`. What the game doesn't hold, the code does: that `Wc`
(the gate model's) and `W` (the requirement's wiring) are one circuit (requirement.py and census/v1 read one IR), that
the blocks at `addr` carry `W`'s credited values (`Requirement.Carries`: the memory-map unit, and the appends' digests
by `remote_bandwidth`'s Replay), that each pass's plan is its step's (#2378), and that the pass ran in its ticks (a
certifier stamping step records and pass roots on one clock). A credit across circuits needs their credited blocks
disjoint, and a device tick credited to a pass here and to another window elsewhere is joint accounting's. Which of the
node's devices held the tiles, and that the bits taken from outside the store are every bit crossing the node's
boundary, are attribution's (`certifier-coverage`, `trusted-nebius-host` and the tiers' bandwidths at the deadline). A
node that registers `R` times pays `δ_G` once a registration.

Proof: `Accounting.SecurityProofs.DeclaredCircuitRequirement`: each carrier's inputs are at most `8·life·R_in` and its
encodes at most `life·c·P·N·4096` units (`OfCircuit` with the property), so every pass rebuilds at most `X_r·τ + C`
bits; an unpaid window at slack `k` tiles a pass has a rank-poor sign matrix, a guessed tag (at every store), or a pass
with `k + 1` correct digests of unobtained sectors; `KeyBinding` at the whole layout, with a union over the `2^a`
layouts inside the uniform store, bounds that pass at `2^a·(q_r·τ/2^96)^(k+1) + δ`; and a pass's reads outside its
tiles of `W` are at most a tile's bits for each other tile, while its credit is at most a tile's bits for each of its
tiles of `W`.

Frozen (no-function, hash): `StackedSessionConservationSketchDeployed`'s game on a store per pass whose layout reads
the codec and the append coins through at most `a` bits, with carriers of a declared circuit passing the stacking
property within its caps over their life, its shortfall read pass by pass against the circuit's credited reads at the
step each pass is bound to, each pass and window on the devices' clocks, in the random-oracle and ideal encoding
models, under the per-pass hash budget, `EncodeChainHardness`, `KeyBinding`, `CircuitCarriers` and `SteeredLayout`. -/
def DeclaredCircuitRequirement : Prop :=
  ∀ (n P N ρ S q G Xr Lr qr τ k a life E B εnum εden T M : ℕ) (ε δ : ℝ≥0∞) {ιW : Type} (Wc : Gates.Dag ιW)
    (caps : Gates.Caps) (Aud : Gates.Audit) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (layout : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S (Lr * τ) q 16 G 2773)
    {V K Dv : Type} [DecidableEq V] [DecidableEq K] (W : Requirement.Wiring V)
    (p : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Placement Dv V)
    (ℓ : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Layout V)
    (grp : RateSketch.StackedCoins Ω Ca T n P N ρ G → V → K)
    (bind : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin n → Fin P → ℕ)
    (addr : RateSketch.StackedCoins Ω Ca T n P N ρ G → ℕ → Fin T × Fin M),
    7 ≤ n → 0 < P → 0 < N → 0 < E →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.EncodeChainHardness Gates.p1024 3 B E εnum εden →
    (∀ w η j pre, Assumptions.SketchHashBudget ((A.window w η).device j pre) (qr * τ)) →
    Assumptions.CircuitCarriers A (Xr * τ) life B εnum εden Wc Aud →
    Assumptions.SteeredLayout a layout →
    Gates.Restricted B Wc → Gates.Property caps Aud Wc → (∀ v < Aud.windows, Aud.reads v ≤ P * N * 4096) →
    (∀ x, (p x).Deadlines (bind x) τ ∧ (∀ w j w' j', w < w' ∨ w = w' ∧ j < j' → bind x w j < bind x w' j') ∧
      ∀ w j,
        let e := (p x).toExecution
        Requirement.Packs W (ℓ x) 32768 (Requirement.live W e (bind x w j)) ∧
        Requirement.Packs W (ℓ x) 32768 (Requirement.reads W e (bind x w j)) ∧
        Requirement.OneBlockEach (ℓ x) (grp x)
          (Requirement.reads W e (bind x w j) \ Requirement.creditedReads W e (ℓ x) (bind x w j)) ∧
        Set.InjOn (addr x) ↑(Requirement.blocks (ℓ x) (Requirement.creditedReads W e (ℓ x) (bind x w j))) ∧
        ∀ b ∈ Requirement.blocks (ℓ x) (Requirement.creditedReads W e (ℓ x) (bind x w j)),
          addr x b ∈ Set.range (layout (RateSketch.SketchPrefix.of x.2 w j) x.1)) →
    let C := 8 * life * caps.runBytes + ⌈(1024 * life * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    let credit := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      let e := (p x).toExecution
      max (Requirement.bitsAt W (ℓ x) (Requirement.credited W e (ℓ x) t)
          (Requirement.blocks (ℓ x) (Requirement.creditedReads W e (ℓ x) t)))
        (Requirement.readBits W e t -
          ((Requirement.reads W e t \ Requirement.creditedReads W e (ℓ x) t).image (grp x)).card * 32768)
    let tiles := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) (j : Fin P) =>
      Finset.univ.filter fun i : Fin N => layout (RateSketch.SketchPrefix.of x.2 w j) x.1 i ∈
        (Requirement.blocks (ℓ x) (Requirement.creditedReads W (p x).toExecution (ℓ x) (bind x w j))).image (addr x)
    let store := RateSketch.programmed (RateSketch.steeredStore enc choose layout)
    pr (fun x : RateSketch.StackedCoins Ω Ca T n P N ρ G =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctOn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ)) : ℝ) <
            ((∑ w, ∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.0015 + γ) * (n * P * N * 32768) ∨
          ∃ w, (((A.win w x.2).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ) : ℝ) <
            ((∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.01 + γ) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ a * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ)

end Accounting.Guarantees
