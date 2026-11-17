module

public import Definitions.Pouw.PearlC.Device
public import Definitions.Pouw.PearlC.Mend

/-! rev1, Pearl-C's credit without the promotions into a +0 total, which a program can skip. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The first promotion add of every word at the device's add price: `m·n` adds, none without promotion. -/
def firstAddDev (d : PearlCDevice) (s : Shape) : ℚ := if d.G = 0 then 0 else d.prices.add * s.m * s.n

/-- **rev1's credited work of one unit at a device**: `creditDev` less the first promotion add of every word, so
`k/(32·G) − 1` adds per word on the domain. -/
def creditDevRev1 (d : PearlCDevice) (s : Shape) : ℚ := creditDev d s - firstAddDev d s

/-- rev1's honest reference at a device: its credited work plus the forming that depends on the activations alone. -/
def wrefDevRev1 (d : PearlCDevice) (s : Shape) : ℚ := creditDevRev1 d s + (d.prices.costs.qa : ℚ) * s.m * s.k

/-- At H100 prices, where an FP32 add costs 32. -/
def firstAddOf (p : Params) (s : Shape) : ℚ := if p.G = 0 then 0 else 32 * s.m * s.n

def creditOfRev1 (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - firstAddOf p s

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

open Classical in
/-- **A word's products that no salt makes count**, of the A row `x` and the B row `w` at the scales `alpha` and
`beta`: the elements `l` that are a dead pair (salt-dead in both rows) or a forced zero (cut at both ends of the support
on the device's step, promoted every `G` atoms). -/
noncomputable def wordMend (d : PearlCDevice) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) {k : ℕ} (x w : Fin k → ℚ) : ℕ :=
  ((univ : Finset (Fin k)).filter fun l => (d.saltDead alpha beta k x l ∧ d.saltDead alpha beta k w l) ∨
    Mend.ForcedZero d.pipe d.G alpha beta k x w l).card

variable {Q R S : Type}

/-- rev1's replayed debit of a unit: `inertWorkDev` of `unitInertWorkDev`, plus its later promotions into a +0 total at the
add price and its words' `wordMend` at one MAC each. -/
noncomputable def PearlCSem.unitInertWorkRev1 (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  inertWorkDev d (sem.unitInertWorkDev d H s U u act) + d.prices.add * ∑ i, ∑ j, (laterZeroPromos d AB.1 AB.2 i j : ℚ) +
    ∑ i, ∑ j, (wordMend d sem.alpha sem.beta (wordsVal sh.m sh.k act i)
      (wordsVal sh.n sh.k (U.fixed (U.layout.fixedId u)) j) : ℚ)

open Classical in
/-- rev1's replayed debit of a tile: `tileInertWorkDev`, plus the later promotions into a +0 total of the tile's words and
their `wordMend` at one MAC each. -/
noncomputable def PearlCSem.tileInertWorkRev1 (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  sem.tileInertWorkDev d H s U u rows cols act + d.prices.add *
    ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      (laterZeroPromos d AB.1 AB.2 e.1 e.2 : ℚ) +
    ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      (wordMend d sem.alpha sem.beta (wordsVal sh.m sh.k act e.1)
        (wordsVal sh.n sh.k (U.fixed (U.layout.fixedId u)) e.2) : ℚ)

/-- **Pearl-C's protocol at a device record under rev1**, with the debit cap `ρ`: `pearlCProtocolDev` with
`creditDevRev1`, `unitInertWorkRev1` and `wrefDevRev1`. -/
noncomputable def pearlCProtocolDevRev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S where
  ticket := fun H s U u act => atNat (d.ticket (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2)
  useful := fun H s U u act => atNat (sem.unitUsefulDev d H s U u act)
  actOK := fun sh act => sem.OperandOK sh.m sh.k act
  fixedOK := fun sh w j => sem.RowOK sh.n sh.k w j
  credit := fun H s U u act =>
    ((creditDevRev1 d (U.layout.shape u) - sem.unitInertWorkRev1 d H s U u act : ℚ) : ℝ)
  inertBoundOK := fun H s U u act => sem.unitInertWorkRev1 d H s U u act ≤ ρ * creditDevRev1 d (U.layout.shape u)
  Wref := fun L u => (wrefDevRev1 d (L.shape u) : ℝ)

/-- **Pearl-C's audit tiles at a device record under rev1**: `pearlCTilesDev` with `creditDevRev1`, `tileInertWorkRev1` and
`wrefDevRev1`. -/
noncomputable def pearlCTilesDevRev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S where
  tiling := auditTiling 64 64
  rowOK := fun sh act i => sem.RowOK sh.m sh.k act i
  credit := fun _ _ U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    (((1 - ρ) * creditDevRev1 d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)
  inertBoundOK := fun H s U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    sem.tileInertWorkRev1 d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
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
