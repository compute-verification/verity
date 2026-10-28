import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.PearlC.Accum
import Definitions.Pouw.PearlC.PeelExactP
import Definitions.Pouw.PearlC.SkipP
import Definitions.Pouw.PearlC.FormingPDefs
import Definitions.Pouw.PearlC.Sm120

/-!
# Pearl-C at a device record (trusted)

`pearlCProtocol` is Pearl-C on the H100: the ticket chain is `hopperStep` promoted every 4 atoms, U is `PeelExact`'s
Hopper BF16 order, the debit's flags are `Skip`'s, the salt-dead predicate is P3's at the Hopper atom-mix, and the costs
are H100 W1 prices. A device record makes each of them a parameter, as the scheme's device record does:
- `PearlCDevice`: the promotion period `G` (0: no promotion, one chain over all of k), the W1 prices, and the device's
  maps (the ticket chain, U, the debit's flags, and P3's salt-dead predicate). The FP8 lane's parametric twins over a
  `Pipeline` supply the maps for a captured step: the chain, `Skip`'s flags, `PeelExact`, and `SaltDead` at the device's
  atom-mix.
- `pearlCProtocolDev` and `pearlCTilesDev`: the named protocol and its audit tiles at a device record. The domain rules
  (`PearlCSem.OperandOK`) are E4M3's, and `sem` still carries the noise, the forming chain and the scales.
- `devH100`: today's H100 record. `pearlCProtocolDev devH100 = pearlCProtocol · Costs.adopted`
  (`pearlCProtocolDev_h100`, `DeviceGamma.lean`), so TT_OUT at the H100 record is the granted `TTOutPearlC`.

- `devAt p G pr`: the record at any captured FP8 step `p` (a `Pipeline`), from the FP8 lane's twins: `Accum.chainAt`,
  `PeelExactP.peelExactValP`, `SkipP.unitFlagsAt` and `FormingP.saltDeadP`. At `hopperE4m3K32`, 4 and `Prices.h100` it
  is `devH100` (`devAt_hopper`, `DeviceGamma.lean`).
- `devSm120v1 pr`: `pearl-c-sm120` v1, the RTX PRO 6000's E4M3 step `⟨[32], 26, −133⟩` (`sm120E4m3`) promoted every 4
  atoms, at prices `pr`. Its named assumption takes `sem.fc = FormingP.atomMixP sm120E4m3`, the sm_120 atom-mix.
- `devSm120v2 pr`: `pearl-c-sm120` v2, the same step without promotion (`G = 0`). Its flags are `SkipP.unitFlagsAt`'s
  at `G = 0`, the pure chain's lone skips (`SkipP.pureFlagsP`), not `SkipP.unitFlagsP · 0`, which would judge one group
  over all of k.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- A device's W1 prices, in units of one dense FP8 multiply-accumulate: one FP32 add on the CUDA cores per element (the
promotion's add, and any post-add), one BF16 multiply-accumulate on the tensor cores (the peel), and the forming per
activation element (`fs` credited, `qa` not). A tensor-core MAC's price is a share of its whole instruction: the cost
model charges an MMA its full fragment shape (on sm_120, 4,096 units for an E4M3 `m16n8k32`), never per output word. -/
structure Prices where
  add : ℚ
  bf16 : ℚ
  costs : Costs

/-- The H100's prices (Rounds 10–11): an FP32 add costs 32, a BF16 MAC 2, and §5's forming `Costs.adopted`. -/
def Prices.h100 : Prices := ⟨32, 2, Costs.adopted⟩

/-- **A device record** for Pearl-C: the promotion period `G` (0: none), the prices, and the device's maps on a unit's
noisy codes (A′, B̃): the ticket words C̃, the useful words U from the codes, F's codes and the peel's noise rows, the
debit's flags per word and atom, and P3's salt-dead predicate at the rows' scales. -/
structure PearlCDevice where
  G : ℕ
  prices : Prices
  ticket : ∀ {m n k : ℕ}, Matrix (Fin m) (Fin k) Code → Matrix (Fin n) (Fin k) Code → Matrix (Fin m) (Fin n) (Option ℚ)
  useful : ∀ {m n k : ℕ}, Matrix (Fin m) (Fin k) Code → Matrix (Fin n) (Fin k) Code → Matrix (Fin k) (Fin 32) Code →
    Matrix (Fin k) (Fin 32) Code → Matrix (Fin m) (Fin 32) ℕ → Matrix (Fin n) (Fin 32) ℕ → Matrix (Fin m) (Fin n) (Option ℚ)
  flags : ∀ {m n k : ℕ}, Matrix (Fin m) (Fin k) Code → Matrix (Fin n) (Fin k) Code → Fin m → Fin n → ℕ → Bool
  saltDead : ((k : ℕ) → (Fin k → ℚ) → ℚ) → ((k : ℕ) → (Fin k → ℚ) → ℚ) → (k : ℕ) → (Fin k → ℚ) → Fin k → Prop

/-- The H100 record: `ctChain` promoted every 4 atoms, `PeelExact`'s U, `Skip`'s flags and P3's `SaltDead`. -/
def devH100 : PearlCDevice where
  G := 4
  prices := Prices.h100
  ticket := fun A B => ctChain 4 A B
  useful := fun A B FA FB beA beB => PeelExact.peelExactVal A B FA FB beA beB
  flags := fun A B => Skip.unitFlags A B
  saltDead := SaltDead.saltDead

/-- **The record at a captured FP8 step `p`**, promotion period `G` and prices `pr`: the FP8 lane's twins at `p`. -/
def devAt (p : Pipeline) (G : ℕ) (pr : Prices) : PearlCDevice where
  G := G
  prices := pr
  ticket := fun A B => Accum.chainAt p G A B
  useful := fun A B FA FB beA beB => PeelExactP.peelExactValP p G A B FA FB beA beB
  flags := fun A B => SkipP.unitFlagsAt p G A B
  saltDead := FormingP.saltDeadP p

/-- **`pearl-c-sm120` v1's record**: the RTX PRO 6000's E4M3 `mma.sync` step `sm120E4m3 = ⟨[32], 26, −133⟩`, promoted
every 4 atoms, at prices `pr`; the named rows take `Prices.sm120` (`DevicePrices`). A tensor-core instruction is priced
by its whole fragment: an E4M3 `m16n8k32` costs 4,096 units and a BF16 `m16n8k16` 2,048 MACs at 2, however few of its
outputs a program uses (`w1-complete/sm120`). -/
def devSm120v1 (pr : Prices) : PearlCDevice := devAt sm120E4m3 4 pr

/-- **`pearl-c-sm120` v2's record**: the same E4M3 step `sm120E4m3` without promotion (`G = 0`), at prices `pr`. The
ticket is the pure chain over all of k (`AccumProofs.chainAt_zero_eq_pure`), and the debit's flags are the pure chain's
lone skips (`SkipP.pureFlagsP`, `FreezePPure.unitFlagsAt_zero_iff`). -/
def devSm120v2 (pr : Prices) : PearlCDevice := devAt sm120E4m3 0 pr

/-- The credited work of one unit at a device: the chain `m·k·n`, its promotion adds (`m·k·n/(32·G)` FP32 adds at the
device's price, none without promotion), the fused `A′·F_B` (`m·k·r`), the peel (depth `2r` in BF16, `2·r·m·n` MACs at the
device's BF16 price), and the salt-dependent forming (`fs·m·k`). At the H100 record it is `creditOf`. -/
def creditDev (d : PearlCDevice) (s : Shape) : ℚ :=
  let mkn : ℚ := (s.m : ℚ) * s.k * s.n
  mkn + (if d.G = 0 then 0 else mkn * d.prices.add / (32 * d.G)) + (s.m : ℚ) * s.k * Params.pi.r +
    2 * d.prices.bf16 * Params.pi.r * s.m * s.n + (d.prices.costs.fs : ℚ) * s.m * s.k

/-- The debit at a device: per flagged atom-word its 32 MACs and its share of the promotion's add, and `fs` per element. -/
def debitDev (d : PearlCDevice) (x : Debit) : ℚ :=
  (32 + (if d.G = 0 then 0 else d.prices.add / d.G)) * x.atoms + (d.prices.costs.fs : ℚ) * x.elems

/-- The honest reference's work at a device: the credited work plus the forming that depends on the activations alone. -/
def wrefDev (d : PearlCDevice) (s : Shape) : ℚ := creditDev d s + (d.prices.costs.qa : ℚ) * s.m * s.k

variable {Q R S : Type}

/-- U at a device, as `PearlCSem.unitUseful` with the device's map. -/
noncomputable def PearlCSem.unitUsefulDev (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) : Matrix (Fin (U.layout.shape u).m) (Fin (U.layout.shape u).n) (Option ℚ) :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.noise H s U u act
  let AB := sem.unitChain H s U u act
  let be := fun (β : ℚ) (e : Fin Params.pi.r → Code) (p : Fin 32) => codeBf16 (lineCodes β e p)
  d.useful AB.1 AB.2 (fun l p => z.FA l p) (fun l p => z.FB l p)
    (fun i => be (sem.beta sh.k (X i)) (z.EA i)) (fun j => be (sem.beta sh.k (W j)) (z.EB j))

open Classical in
/-- The unit's replayed debit at a device, as `PearlCSem.unitDebit` with the device's flags and salt-dead predicate. -/
noncomputable def PearlCSem.unitDebitDev (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) : Debit :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  ⟨∑ i : Fin sh.m, ∑ j : Fin sh.n, ((range (sh.k / 32)).filter fun t => fl i j t = true).card,
   (univ.filter fun e : Fin sh.m × Fin sh.k => d.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨ ¬ Skip.needed fl e.1 e.2).card⟩

/-- **Pearl-C's protocol at a device record**, with the debit cap `ρ`. -/
noncomputable def pearlCProtocolDev (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S where
  ticket := fun H s U u act => atNat (d.ticket (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2)
  useful := fun H s U u act => atNat (sem.unitUsefulDev d H s U u act)
  actOK := fun sh act => sem.OperandOK sh.m sh.k act
  weightOK := fun sh w j => sem.RowOK sh.n sh.k w j
  credit := fun H s U u act =>
    ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u act) : ℚ) : ℝ)
  capOK := fun H s U u act => debitDev d (sem.unitDebitDev d H s U u act) ≤ ρ * creditDev d (U.layout.shape u)
  Wref := fun L u => (wrefDev d (L.shape u) : ℝ)

/-- Pearl-C's shape domain at a device: positive `m`, `n` a positive multiple of 64, k a whole number of 128-deep
kernel tiles and, when the record promotes, of promotion groups, and `k ≤ 2^16`. Without promotion (`G = 0`) the group
condition is void: read as `32·0 ∣ k` it would force `k = 0` and leave the domain empty. At the H100 record it is
`pearlCDomain Params.pi` with `n`'s floor (`pearlCDomainDev_h100`).

The floor on `n` keeps out the units whose output words are cheaper to guess than TT_OUT's `εPearlC` allows. The binding
attack is partial guessing: compute most of a unit's words and guess the rest. Under W1 a winning strategy must save the
unit's `n·c` (the weight-side forming and `B̃·F_A`, online and uncredited) plus `γ₀` of its credit, and the cheapest
such saving guesses about 96% of one row: 124 words at `n = 64` on the H100 (123 on sm_120), about 995 at `n = 512`.
Each guessed word then needs only about 1.04 bits of salt-dependent min-entropy at `n = 64`, and about 0.13 at
`n = 512`, which TT_OUT already assumes. `64 ∣ n` matters as much as `64 ≤ n`: a `1 × (n mod 64)` edge tile is guessed
whole. `m` stays free, so a one-row unit is in the domain (`pearlCDomainDev_split`). -/
def pearlCDomainDev (d : PearlCDevice) (L : Layout) : Prop :=
  ∀ u < L.N, 0 < (L.shape u).m ∧ (64 ≤ (L.shape u).n ∧ 64 ∣ (L.shape u).n) ∧ 0 < (L.shape u).k ∧
    128 ∣ (L.shape u).k ∧ (d.G = 0 ∨ 32 * d.G ∣ (L.shape u).k) ∧ (L.shape u).k ≤ 2 ^ 16

/-- The domain restricted to one shape. -/
def pearlCDomainDevAt (d : PearlCDevice) (s : Shape) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, L.shape u = s

/-- The domain restricted to one linear, `n` and `k` fixed and `m` free: its units at any mix of batch sizes. -/
def pearlCDomainDevNK (d : PearlCDevice) (n k : ℕ) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, (L.shape u).n = n ∧ (L.shape u).k = k

/-- The domain restricted to a set of linears: every unit's `(n, k)` in `nks`, `m` free. A served step's layout holds
every linear of its model at once, each at any mix of batch sizes. -/
def pearlCDomainDevShapes (d : PearlCDevice) (nks : Finset (ℕ × ℕ)) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, ((L.shape u).n, (L.shape u).k) ∈ nks

/-- Llama-3.1-8B's linears `(n, k)`: qkv, o, gate_up, down. -/
def llama31_8bLinears : Finset (ℕ × ℕ) := {(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)}

/-- Qwen3-8B's linears `(n, k)`: qkv, o, gate_up, down. -/
def qwen3_8bLinears : Finset (ℕ × ℕ) := {(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)}

open Classical in
/-- A tile's replayed debit at a device, as `PearlCSem.tileDebit` with the device's flags, salt-dead predicate and prices.
The forming part is P3's per-tile rule ("The per-tile cap"). -/
noncomputable def PearlCSem.tileDebitDev (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  let atoms := ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
    ((range (sh.k / 32)).filter fun t => fl e.1 e.2 t = true).card
  let elems := ((univ : Finset (Fin sh.m × Fin sh.k)).filter fun e => (e.1 : ℕ) ∈ rows ∧
    (d.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨ ∀ j : Fin sh.n, (j : ℕ) ∈ cols → fl e.1 j ((e.2 : ℕ) / 32) = true)).card
  debitDev d ⟨atoms, 0⟩ + (d.prices.costs.fs : ℚ) * (cols.card / sh.n) * elems

/-- **Pearl-C's audit tiles at a device record**, as `pearlCTiles` (credit on the credited rows), with `creditDev` and
`wrefDev`. -/
noncomputable def pearlCTilesDev (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S where
  tiling := auditTiling 64 64
  rowOK := fun sh act i => sem.RowOK sh.m sh.k act i
  credit := fun _ _ U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    (((1 - ρ) * creditDev d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)
  capOK := fun H s U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    sem.tileDebitDev d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
      ρ * creditDev d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g)
  Wref := fun L g =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefDev d sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
  Wcred := fun L g act =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefDev d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)

end Pouw.PearlC
