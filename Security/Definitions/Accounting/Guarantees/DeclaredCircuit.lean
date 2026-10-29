import Definitions.Accounting.Guarantees.Stacking

/-!
# A declared circuit's space and bandwidth from an accepted audit (trusted)

The goal in one statement, `DeclaredCircuitRequirement`: for any declared circuit `W` that passes the stacking
property, any prover, and an accepted audit at the deployed parameters (sketch receipts, a store per pass, the
append-coin rule), except with probability `δ_G` the prover system read, from the appended blocks of `W`'s values each
pass's step reads, `W`'s credited bits there less an explicit share `ε` of the store. It is a layer-1 bound in decision
111's sense (a cryptographic lower bound on the prover system, parametric in the system's resources a pass); which device
held what (the node's boundary, HBM against host DRAM) is attribution, stated apart. It composes the stacked remote
audit in its deployed sketch form (`StackedSessionConservationSketchDeployed`), on a layout the prover may steer after
the coins, with the carriers' cost read from the property's caps (`Gates.Property`, under
`Accounting.Assumptions.SquaringFloor` and `Accounting.Assumptions.ChainSum`) and the audit linked to `W`'s placed
requirement (A1's
`Definitions/Accounting/Requirement.lean`) pass by pass, each pass and window on the devices' clocks. It credits `W`'s
appended blocks only: a registered block's key is fixed for the registration, so nothing bounds its carriers, and the
weights and the filler are the registration audit's to count. Its theorem is
`Accounting.SecurityProofs.DeclaredCircuitRequirement`, in `Security/Proofs/Accounting/DeclaredCircuit/Requirement.lean`
with its pieces beside it.
-/

namespace Accounting.Guarantees

open Accounting
open Pous (pr)
open scoped ENNReal

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the deployed parameters.**

*The audit.* `StackedSessionConservationSketchDeployed`'s session: `n ≥ 7` windows of `P` passes, sketch receipts at
16 rows over tiles of 128 sectors, 512-bit tags, 2,773 openings a window (the 1 s window's count; a longer window's
audit instantiates `G` by verifying a uniform 2,773 of its draws, and a paced session's when each group draws at least
2,773), and a store per pass made by the append-coin rule (`RateSketch.steeredStore`: the blocks of `T` appends of `M`
positions, each append's choices `choose c ω a κ_<a` fixed before its own coin keys its blocks). `P` is fixed before
the session's coins, and the code holds a window to it only in a paced session, whose windows are its groups of
at most `k` passes (the audit refuses a group's root `k + 1`, `order`, in Python and Lean): a paced session reads `G` at
`P = k`, a group of fewer passes as one padded with passes that hold no tile of `W`.
The code caps no unpaced window's passes yet, so a
KV-credited session meets `G` when it is paced. Pass `j` of window `w` certifies the `N` blocks at `layout c y`, where
`c` is the session's coins before the pass (`RateSketch.SketchPrefix`) and `y` the codec and every append's coin, so
the layout may follow inputs chosen after those coins: it reads `y` only through the `a`-bit summary `steer c y`
(`Accounting.Assumptions.SteeredLayout`). That holds of every layout at `a = |Y|`; what makes `a` small is the schedule
check at a cap fixed before the coins. `SteeredLayout` fixes one `a` at every coin, and
`Properties.RemoteAudit.steered_of_checked` gives it at `a = a₀ + 1` when, at every prefix and every append coin, the
check accepts a schedule of at most `a₀` bits holding the plan the pass sent, the model's layout of a pass being the
verifier's own (`Properties.RemoteAudit.sessionLayout`: the positions of the session's appends and the plan the pass
sent, where the verifier checks each opened block, `Properties.RemoteAudit.CodeLayoutTiles` and `OpensAtPosition`). So
`a₀` is the steering cap the prover commits before the first coin
(`audit.Params.schedule_cap`, credited as `audit.Ledger.steering`), over which the check refuses a schedule, not
the bits of the one schedule a session published: a prover whose schedule is 200 bits on most coins and 300 where it
steers would meet `SteeredLayout` only at 301 (which the
committed cap fixes). `Properties.DeclaredCircuit.ScheduledRequirement` is `G` with `SteeredLayout` so discharged, on
the sessions the check accepts at every pass, with `2^(a + (cap + 1))` in the query term
(`Properties.DeclaredCircuit.SecurityProofs.scheduledRequirement`). A layout that follows a
coin at a counted schedule is refused (`Properties.RemoteAudit.coinLayout_not_scheduled`, `ScheduleFixesLayout`), so
`a` does not collapse to `|Y|`. Each append has its own coin, uniform and independent of the others', so a re-key,
which appends a block again, draws a fresh one. Each append's blocks are read within `L` windows of the window
`joined x t` it joined (`RateSketch.LifetimeCapped`, the remote audit's `lifetime`). The remote audit's `L` is a life
of `D = 30` s in whole windows of length `T_w`, `L = max(L_min, ⌊D/T_w⌋)` (`audit.LIFETIME_NS`, `lifetime_windows`),
with `L_min` 2 when the prover defers residency and 1 otherwise; a paced session counts its groups at their
nominal length.

*The prover system.* Its pass devices are verbatim (`rate-sketch-verbatim`), and they are told their pass's summary:
`A` has a session for each value of the summary (`RateSketch.SketchSummarySession`), and pass `j` of window `w` runs
the devices of session `steer c y`, so a device may follow the layout the prover steered. The system's resources a
pass, within the pass's deadline `τ`, are the class's parameters: at most `S` bits kept besides the store, at most
`L_r·τ` bits taken from outside it, at most `q_r·τ` oracle queries over its grinding candidates at every summary
(`Accounting.Assumptions.SketchHashBudget`), and at most `X_r·τ` bits of the pass's appended tiles rebuilt besides the
sectors of a carrier of the declared circuit `Wc`'s over at most `L` of the audit `Aud`'s windows, the lifetime cap
(`Accounting.Assumptions.CircuitCarriers`, on the pass's store, under the lifetime cap). Its rebuilds of registered
tiles are not bounded. `G` holds at every value of them; what they are on a node's devices is layer 2 (data sheets,
named assumptions).

*The circuit.* `Wc` is in the restricted class (`Gates.Restricted B`) and passes the stacking property at the caps
`caps` on `Aud` (`Gates.Property`): in each window its nonlinear units (its budgeted units and its data products,
`Gates.nonlinear`) are within `c = caps.total` a byte `Aud` certifies, its data products within `caps.products` a
byte, and its run inputs within that window's `R_in,v = caps.runBytes v` bytes. A window certifies at most its
passes' stores, `P·N·4096` bytes. An encode of a word costs `E` nonlinear units, with `E ≤ 3·S·e_s` (`S = 1,021`
squarings a root) under `Accounting.Assumptions.SquaringFloor` at `e_s` nonlinear units a squaring and
`Accounting.Assumptions.ChainSum` at the deployed codec (p1024, 3 rounds, `ε ≥ 2^−64`).

*The requirement.* The prover chooses everything on `W`'s side after every coin: the placed execution `p x` (each
value's device `leaf` and the tick `ticks d s` at which device `d` begins engine step `s`), the layout of values in
blocks `ℓ x` (with each block's first encoded read: deferred residency credits a block from it), the groups `grp x` (a
KV value's sequence, layer, plane and page parity), the step `bind x w j` each pass is bound to,
strictly increasing over the passes, the memory map `σ x`, each pass's plan `plan x w j` (its blocks), and the
store position `addr x k` of each block `k`. On device `d`'s clock pass `j` of window `w` is the `τ` ticks from the one
at which `d` begins the pass's step (`Requirement.Placement.passTicks`), window `w` its passes' ticks
(`Requirement.Placement.windowTicks`), and no device tick is in two windows or two passes of one
(`Requirement.Placement.Deadlines`). At each bound step `t` the memory map holds `W`'s credited blocks
(`Requirement.StoreMatches`: every block holding a credited value is resident, the live values are packed into blocks
of 32,768 bits, and each credited value is in its block's payload), the pass's plan covers `W`'s reads
(`Requirement.PlanMatches`: every complete block `t` reads is planned, and the values read are packed), the values `t`
reads from incomplete blocks lie in one block a group, and the plan's blocks are at distinct positions of the pass's
store. The links are one-sided: other blocks may be resident or planned (filler, dead values, another workload's), and
they earn `W` nothing. `W`'s owned blocks at `t` are the blocks of its credited reads at an append after the
registration (`(addr x k).1 ≠ 0`). Registered blocks (append `0`: the weights, the filler) earn `W` nothing here,
whatever it reads from them, because their keys are fixed for the registration and their carriers are not bounded. The
pass's tiles of `W` are the tiles of its owned blocks, and its credit is the larger of `W`'s credited live bits in them
(held) and the bits `t` reads from appended blocks less a block a group (moved):

  `owned t = {k ∈ blocks (creditedReads t) | (addr k).1 ≠ 0}`,
  `credit t = max (bitsAt (credited t) (owned t))`
  `  (bits {v ∈ reads t | (addr (block v)).1 ≠ 0} − |grp(reads t \ creditedReads t)|·32768)`.

*The bound.* With

  `R_L = max_I Σ_{v∈I} R_in,v` over the sets `I` of at most `L` of `Aud`'s windows, the run-input bytes the caps
  allow a carrier's life,
  `C = 8·R_L + ⌈1024·L·c·P·N·4096/E⌉`, the carriers' bits a pass,
  `γ = (S + (L_r·τ − ρ·256) + X_r·τ + C + k·32768)/(N·32768)`,

the probability that every opening passes while the passes' distinct sectors read from their tiles of `W` fall short of
the credit over the bound steps by more than `(0.0015 + γ)` of the session's `n·P·N·32768` bits, or some window's fall
short of its passes' credit by more than `(0.01 + γ)` of its `P·N·32768`, is at most

  `δ_G = 2·2^-40 + n·ε + n·P·G·N/2^512 + n·P·(2^a·(q_r·τ/2^96)^(k+1) + δ)`.

**The steering.** The layout reads no window coin, so the draws' term and the signs' rank term (`n·ε`) don't see it,
and the tag-guess term holds at every store. The query term is a correct digest of a tile with a sector the device
never obtained, which needs the pass's blocks uniform: key binding gives that per pass at `2^a` times the uniform
probability, `δ` once. A pass short by `k` more tiles than the slack needs `k + 1` distinct unobtained positions, tile
sectors or the stream's, among its correct digests, at `2^-96` each, so `k` tiles of slack a pass trade for
`k·(96 − log₂(q_r·τ))` bits of steering: about 63 bits a tile at `q_r·τ = 2^33` (41 at the stated cap of `2^55`), the
steering costing about 520 bits of `γ`'s numerator a bit. The devices are told the
summary, so a device may aim at the layout the prover steered to. Each summary fixes both the layout and the devices,
so the union over the `2^a` summaries covers them together: an adaptive device costs nothing beyond `2^a`. At
`a = k = 0` the bound is `StackedSessionConservationSketchDeployed`'s.

**The reading.** The sectors counted are the distinct addresses of the store each pass read from device memory, within
the pass's ticks on each device's clock, and the bound is on their sum, over the session and over each window's `P`
passes. So what it says the prover system held is an average over a window's passes, about the window's mean credit a
pass (`f·N` tiles at a credited share `f` of the store), not what any one pass held: a pass may read less while another
of its window reads more, and a sector read was in device memory when it was read, not through the pass. The bound
steps' credit is `W`'s in its appended blocks: held, its credited live bits in the appended blocks a step reads, and
moved, what the step reads from appended blocks. The shortfall is a share of the store: relative to `W` it is
`ε_W = (0.0015 + γ)·n·P·N·32768/Σ credit`. What the game doesn't hold, the code does: that `Wc` (the gate model's) and
`W` (the requirement's wiring) are one circuit (requirement.py and census/v1 read one IR), that the memory map and the
plans hold `W`'s credited blocks (`StoreMatches`, `PlanMatches`: `core_requirement`'s one-sided check, and the
memory-map and plan units), that the payloads at `addr` are the store's blocks, each at its own append (the appends'
digests, by `remote_bandwidth`'s Replay), that each pass's plan is its step's, and that the pass ran in its
ticks (a certifier stamping step records and pass roots on one clock). A credit across circuits needs their credited blocks
disjoint, and a device tick credited to a pass here and to another window elsewhere is joint accounting's. Which of the
node's devices held the tiles, and that the bits taken from outside the store are every bit crossing the node's
boundary, are attribution's (`certifier-coverage`, `trusted-host` and the tiers' bandwidths at the deadline). A
node that registers `R` times pays `δ_G` once a registration.

`G` credits nothing once `γ ≥ 1 − 0.0015`, because a pass's credit is at most its `N` tiles' bits and the event's
threshold is then negative. The carrier term grows with the carrier's life, `128·L·P·c/E` of the store with `P·N·4096`
bytes certified a window and `c = caps.total`, so `G` is useful only when the life `L·T_w` is short: at caps
(c = 0.442 for 8B, 0.0918 for 70B TP8, 61 ms passes) `ε` a session is 10% at a life of 35 s for 8B and 171 s for 70B.
At remote_bandwidth's deployed life of 30 s it is 8.62% for 8B and 2.57% for 70B at 1 s windows (`L = 30`, the same
492 passes as a paced session of 12 groups of `k = 82` a 60 s stretch at `L = 6`). At the stated 60 s window
`⌊D/T_w⌋ = 0`, so `L = L_min`: 16.25% and 4.16% at `L = 1`, and 31.52% and 7.33% at `L = 2` for a prover that defers
residency, as the vLLM serving store does (the earlier
60 windows of 60 s put it past 100% for both). These are `G`'s at a paced session whose groups each draw at least
2,773 openings, which no deployed KV-credited session is yet. An honest run meets every hypothesis but the world's, with
its credit at least the store's bits, so the bound says something whenever `0.0015 + γ < 1`, at a decode schedule and
at a prefill. The decode witness (`Accounting.SecurityProofs.DeclaredCircuitWitness`) decodes one token a step a
sequence, each page's coin back two steps after it completes, with page-parity groups. The prefill witness
(`Accounting.SecurityProofs.DeclaredCircuitWitnessPrefill`) is a chunked prefill and then the first decode step, with a
group a page, so a chunk may complete several pages while their coins are out; its credit is exactly its tiles' bits,
and both of `G`'s thresholds are positive at the 30 s life for 8B and for 70B (`life30_8b`, `life30_70b`).

Proof (`Accounting.SecurityProofs.DeclaredCircuitRequirement`): under the lifetime cap each carrier's inputs are at
most `8·R_L` and its encodes at most `L·c·P·N·4096` nonlinear units (`OfCircuit` with the property), so every pass
rebuilds at
most `X_r·τ + C` bits of its appended tiles, and its tiles of `W` are appended tiles; the conservation runs on a pass's
tiles of `W` alone (their sectors held, read, linked or rebuilt), so the rebuilds of registered tiles, which nothing
bounds, never enter it; an unpaid window at slack `k` tiles a pass has a rank-poor sign matrix, a guessed tag (at every
store and every device), or a pass with `k + 1` correct digests behind distinct unobtained positions of its tiles of
`W` and the stream, at most one sector a tile; `KeyBinding` at the whole layout, with a union inside the uniform store
over the `2^a` summaries, each fixing the layout `slot c s` and the devices of session `s` (`KeyBinding.pr_steered_le`),
bounds that pass at `2^a·(q_r·τ/2^96)^(k+1) + δ`; and a pass's credit is at most a tile's bits for each of its tiles of
`W`.

Frozen (no-function, hash): `StackedSessionConservationSketchDeployed`'s game on a store per pass whose layout reads
the codec and the append coins through an `a`-bit summary the pass's devices are told, each append's blocks read
within `L` windows of the one it joined, with the rebuilds of appended tiles charged to carriers of a declared circuit
passing the stacking property within its caps over `L` windows, its shortfall read pass by pass against the circuit's
credited reads in appended blocks at the step each pass is bound to, with the memory map and plans holding its
credited blocks (one-sided), each pass and window on the devices' clocks, in the random-oracle and ideal encoding
models, under the per-pass hash budget, `SquaringFloor`, `ChainSum`, `KeyBinding`, `CircuitCarriers` and
`SteeredLayout`. -/
def DeclaredCircuitRequirement : Prop :=
  ∀ (n P N ρ S q G Xr Lr qr τ k a L E es B εnum εden T M : ℕ) (ε δ : ℝ≥0∞) {ιW : Type} (Wc : Gates.Dag ιW)
    (caps : Gates.Caps) (Aud : Gates.Audit) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (layout : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (steer : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (joined : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin T → ℕ)
    (A : RateSketch.SketchSummarySession a n P N 128 ρ 512 S (Lr * τ) q 16 G 2773)
    {V K Dv : Type} [DecidableEq V] [DecidableEq K] (W : Requirement.Wiring V)
    (p : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Placement Dv V)
    (ℓ : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Layout V)
    (grp : RateSketch.StackedCoins Ω Ca T n P N ρ G → V → K)
    (bind : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin n → Fin P → ℕ)
    (σ : RateSketch.StackedCoins Ω Ca T n P N ρ G → Requirement.Store)
    (plan : RateSketch.StackedCoins Ω Ca T n P N ρ G → Fin n → Fin P → Finset ℕ)
    (addr : RateSketch.StackedCoins Ω Ca T n P N ρ G → ℕ → Fin T × Fin M),
    7 ≤ n → 0 < P → 0 < N → 0 < E →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.SquaringFloor Gates.p1024 es εnum εden → Assumptions.ChainSum Gates.p1024 3 B es εnum εden →
    E ≤ 3 * Gates.p1024.squarings * es → εden ≤ 2 ^ 64 * εnum →
    (∀ s w η j pre, Assumptions.SketchHashBudget (((A.session s).window w η).device j pre) (qr * τ)) →
    Assumptions.CircuitCarriers (fun c y => A.session (steer c y)) (RateSketch.steeredStore enc choose layout) layout
      joined (Xr * τ) L B εnum εden Wc Aud →
    Assumptions.SteeredLayout layout steer →
    RateSketch.LifetimeCapped layout joined L →
    Gates.Restricted B Wc → Gates.Property caps Aud Wc → (∀ v < Aud.windows, Aud.reads v ≤ P * N * 4096) →
    (∀ x, (p x).Deadlines (bind x) τ ∧ (∀ w j w' j', w < w' ∨ w = w' ∧ j < j' → bind x w j < bind x w' j') ∧
      ∀ w j,
        let e := (p x).toExecution
        Requirement.StoreMatches W e (ℓ x) (σ x) 32768 (bind x w j) ∧
        Requirement.PlanMatches W e (ℓ x) (plan x w j) 32768 (bind x w j) ∧
        Requirement.OneBlockEach (ℓ x) (grp x)
          (Requirement.reads W e (bind x w j) \ Requirement.creditedReads W e (ℓ x) (bind x w j)) ∧
        Set.InjOn (addr x) ↑(plan x w j) ∧
        ∀ b ∈ plan x w j, addr x b ∈ Set.range (layout (RateSketch.SketchPrefix.of x.2 w j) x.1)) →
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    let owned := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      (Requirement.blocks (ℓ x) (Requirement.creditedReads W (p x).toExecution (ℓ x) t)).filter
        fun b => ((addr x b).1 : ℕ) ≠ 0
    let credit := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      let e := (p x).toExecution
      max (Requirement.bitsAt W (ℓ x) (Requirement.credited W e (ℓ x) t) (owned x t))
        (Requirement.bits W ((Requirement.reads W e t).filter fun v => ((addr x ((ℓ x).block v)).1 : ℕ) ≠ 0) -
          ((Requirement.reads W e t \ Requirement.creditedReads W e (ℓ x) t).image (grp x)).card * 32768)
    let tiles := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) (j : Fin P) =>
      Finset.univ.filter fun i : Fin N => layout (RateSketch.SketchPrefix.of x.2 w j) x.1 i ∈
        (owned x (bind x w j)).image (addr x)
    let store := RateSketch.programmed (RateSketch.steeredStore enc choose layout)
    let win := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) =>
      A.win w x.2 fun j => steer (RateSketch.SketchPrefix.of x.2 w j) x.1
    pr (fun x : RateSketch.StackedCoins Ω Ca T n P N ρ G =>
      (∀ w t, (x.2 w).2.2 t ∈ (win x w).correctOn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ)) : ℝ) <
            ((∑ w, ∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.0015 + γ) * (n * P * N * 32768) ∨
          ∃ w, (((win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ) : ℝ) <
            ((∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.01 + γ) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ a * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ)

end Accounting.Guarantees
