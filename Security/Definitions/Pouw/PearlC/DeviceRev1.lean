import Definitions.Pouw.PearlC.Device

/-!
# rev1: Pearl-C's credit without the promotions into a +0 total (trusted; definitions only)

The granted credit counts `k/(32·G)` FP32 promotion adds per word, `T_g = RNE(T_{g−1} + S_g)` from `T_0 = +0`. A
promotion into a +0 total returns `S_g` itself (`Defs.promote` on `0`; the scheme's `chain` reads `−0` as `+0`), so a
program can skip it. rev1 takes every promotion into a +0 total out of the credit:
* **the first of every word** (`T_0 = +0` always): out of the credit and out of `W_ref` (`firstAddDev`), since the
  honest kernel needn't execute it either;
* **any later one** (`T_{g−1} = 0` for `g ≥ 2`): debited by replay (`laterZeroPromos`), since only the chain's values
  show it. It is rare: at most 3.9·10⁻⁶ of later promotions on the H100 atom and none measured on sm_120,
  so it moves no γ. `W_ref` keeps it: the honest kernel can't skip it
  without a per-word test that costs as much as the add.

The maps, the domain and the cap's form are the device record's. The debit gains the later +0 promotions and is
otherwise `debitDev`, which still gives each flagged atom-word `add/G` of its group's promotion, the first group's
included. That over-debits by at most `firstAddDev` per unit, so it only lowers the credit and tightens the cap.

* `firstAddDev`, `creditDevRev1`, `wrefDevRev1`: the first add of every word at the device's add price, and the credit
  and `W_ref` without it. `creditOfRev1` and `wrefOfRev1` are the same at `Params` and `Costs` (H100 prices).
* `laterZeroPromos`, `PearlCSem.unitDebitRev1`, `PearlCSem.tileDebitRev1`: the later +0 promotions of a word, read off the
  record's own ticket map on column prefixes (`T_g` is the chain over the first `32·G·g` columns), and the unit and
  tile debits with them.
* `pearlCProtocolDevRev1`, `pearlCTilesDevRev1`: `pearlCProtocolDev` and `pearlCTilesDev` with rev1's credit, debit and
  `W_ref`, in the credit, the cap and `W_ref` alike.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The first promotion add of every word at the device's add price: `m·n` adds, none without promotion. -/
def firstAddDev (d : PearlCDevice) (s : Shape) : ℚ := if d.G = 0 then 0 else d.prices.add * s.m * s.n

/-- **rev1's credited work of one unit at a device**: `creditDev` less the first promotion add of every word, so
`k/(32·G) − 1` adds per word on the domain. -/
def creditDevRev1 (d : PearlCDevice) (s : Shape) : ℚ := creditDev d s - firstAddDev d s

/-- rev1's honest reference at a device: its credited work plus the forming that depends on the activations alone. -/
def wrefDevRev1 (d : PearlCDevice) (s : Shape) : ℚ := creditDevRev1 d s + (d.prices.costs.qa : ℚ) * s.m * s.k

/-- The first promotion add of every word at H100 prices (an FP32 add costs 32), as `creditOf` prices its adds. -/
def firstAddOf (p : Params) (s : Shape) : ℚ := if p.G = 0 then 0 else 32 * s.m * s.n

/-- **rev1's credited work of one unit** at `Params` and `Costs` (H100 prices): `creditOf` less the first add. -/
def creditOfRev1 (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - firstAddOf p s

/-- rev1's honest reference at `Params` and `Costs`. -/
def wrefOfRev1 (p : Params) (c : Costs) (s : Shape) : ℚ := creditOfRev1 p c s + (c.qa : ℚ) * s.m * s.k

/-- The first `k'` columns of a matrix of codes. -/
def prefixCols {p k : ℕ} (M : Matrix (Fin p) (Fin k) Code) (k' : ℕ) (h : k' ≤ k) : Matrix (Fin p) (Fin k') Code :=
  fun i l => M i (Fin.castLE h l)

open Classical in
/-- **The later promotions into a +0 total** of word `(i, j)`: the `g ∈ [1, k/(32·G))` whose total after `g` groups, the
record's ticket map on the first `32·G·g` columns, is zero, so that promotion `g + 1` adds into `+0`. None without
promotion. -/
noncomputable def laterZeroPromos (d : PearlCDevice) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) : ℕ :=
  if d.G = 0 then 0 else
    ((Ico 1 (k / (32 * d.G))).filter fun g =>
      ∃ h : 32 * d.G * g ≤ k, d.ticket (prefixCols A _ h) (prefixCols B _ h) i j = some 0).card

variable {Q R S : Type}

/-- rev1's replayed debit of a unit: `debitDev` of `unitDebitDev`, plus its later promotions into a +0 total at the
add price. -/
noncomputable def PearlCSem.unitDebitRev1 (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (act : Codes) : ℚ :=
  let AB := sem.unitChain H s U u act
  debitDev d (sem.unitDebitDev d H s U u act) + d.prices.add * ∑ i, ∑ j, (laterZeroPromos d AB.1 AB.2 i j : ℚ)

open Classical in
/-- rev1's replayed debit of a tile: `tileDebitDev`, plus the later promotions into a +0 total of the tile's words. -/
noncomputable def PearlCSem.tileDebitRev1 (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  sem.tileDebitDev d H s U u rows cols act + d.prices.add *
    ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      (laterZeroPromos d AB.1 AB.2 e.1 e.2 : ℚ)

/-- **Pearl-C's protocol at a device record under rev1**, with the debit cap `ρ`: `pearlCProtocolDev` with
`creditDevRev1`, `unitDebitRev1` and `wrefDevRev1`. -/
noncomputable def pearlCProtocolDevRev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S where
  ticket := fun H s U u act => atNat (d.ticket (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2)
  useful := fun H s U u act => atNat (sem.unitUsefulDev d H s U u act)
  actOK := fun sh act => sem.OperandOK sh.m sh.k act
  weightOK := fun sh w j => sem.RowOK sh.n sh.k w j
  credit := fun H s U u act =>
    ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s U u act : ℚ) : ℝ)
  capOK := fun H s U u act => sem.unitDebitRev1 d H s U u act ≤ ρ * creditDevRev1 d (U.layout.shape u)
  Wref := fun L u => (wrefDevRev1 d (L.shape u) : ℝ)

/-- **Pearl-C's audit tiles at a device record under rev1**: `pearlCTilesDev` with `creditDevRev1`, `tileDebitRev1` and
`wrefDevRev1`. -/
noncomputable def pearlCTilesDevRev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S where
  tiling := auditTiling 64 64
  rowOK := fun sh act i => sem.RowOK sh.m sh.k act i
  credit := fun _ _ U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    (((1 - ρ) * creditDevRev1 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)
  capOK := fun H s U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    sem.tileDebitRev1 d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
      ρ * creditDevRev1 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g)
  Wref := fun L g =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefDevRev1 d sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
  Wcred := fun L g act =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefDevRev1 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)

end Pouw.PearlC
