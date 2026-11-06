import Definitions.Pouw.Game.Defs
import Definitions.Pouw.PearlC.Instance

/-!
# The per-tile cap and the sampled audit (trusted)

The verifier opens drawn tiles, not units: `verity.protocols.accounting.work.pouw.audit` draws `t` tiles independently, each with probability
`W_ref(tile)/W`, and recomputes each drawn tile from its openings (its activation rows and weight rows over the full k).
So the debit cap has to be decided on a tile, and γ has to hold for a verifier that checks drawn tiles only. This
file states both; `TileGamma` proves γ from them.

- **A tile** is the output words `rows g × cols g` of one unit (`Tiling`). The named tiling is the audit's grid of
  `64 × 64` words (`auditTiling`).
- **Credited rows.** A tile's rows whose activation row passes Π's rules (`PearlCSem.RowOK`, `PearlCSem.passRows`) are
  credited. A row that fails them (a massive activation the fixed `S_ℓ` doesn't cover) stays in its unit, computed and
  bound, and earns no credit: the plan's one per-row rule. The verifier checks
  that an excluded row fails the rules, which is what makes it uncredited here.
- **A good tile** (`TileGood`): its replayed debit on its credited rows is within the cap, each of its weight rows
  passes the rules (`weightOK`), and each ticket and useful word of its credited rows is correct. A weight row is never
  excluded: a tile with a failing one is not good, and earns nothing. Each part reads only the tile's openings:
  - the rules are per row;
  - P1's flags of a word read only the word's two rows (`Skip.wordFlags`), so the chain debit, the flagged atoms summed
    over the words, is replayed from the openings;
  - the forming debit reads the tile's rows and its words' flags.
- **The credit** of a good tile is fixed at `(1 − ρ)·creditOf` times its credited rows' share of its unit's words
  (`pearlCTiles`), whatever its debit within the cap. The kernel credits an accepted tile this way.
- **The draw's weights** are every tile's full `W_ref`, fixed by the layout; the certified work counts credited rows only
  (`TileRules.Wcred`).
- **Blank units** are the ones the prover leaves uncommitted. Their tiles are never checked and never credited.
- **`GγSampled`:** for every adversary, the probability that the audit accepts and the credited work, less `ε_s` of the
  layout's, exceeds `T/(1 − γ)` is at most `(1 − ε_s)^t + η(q, N)`. The draw is independent of the oracle and the salt,
  so its source must be fixed only after the transcript's commitment.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- A layout's audit tiles: tile `g < NT` is the output words `rows g × cols g` of unit `unit g`. -/
structure Tiling where
  NT : ℕ
  unit : ℕ → ℕ
  rows : ℕ → Finset ℕ
  cols : ℕ → Finset ℕ

/-- The audit's grid: each unit's `m × n` words in blocks of `tm × tn`, clipped at the edges; the units in order, and a
unit's blocks by block row, then block column. The order does not enter γ: every tile is drawn by its own weight. -/
def gridTiles (tm tn : ℕ) (L : Layout) : List (ℕ × ℕ × ℕ) :=
  (List.range L.N).flatMap fun u =>
    (List.range (((L.shape u).m + tm - 1) / tm)).flatMap fun bi =>
      (List.range (((L.shape u).n + tn - 1) / tn)).map fun bj => (u, bi, bj)

/-- The tiling of the audit's `tm × tn` grid. -/
def auditTiling (tm tn : ℕ) (L : Layout) : Tiling where
  NT := (gridTiles tm tn L).length
  unit := fun g => ((gridTiles tm tn L).getD g (0, 0, 0)).1
  rows := fun g =>
    let b := (gridTiles tm tn L).getD g (0, 0, 0)
    Ico (b.2.1 * tm) (min ((b.2.1 + 1) * tm) (L.shape b.1).m)
  cols := fun g =>
    let b := (gridTiles tm tn L).getD g (0, 0, 0)
    Ico (b.2.2 * tn) (min ((b.2.2 + 1) * tn) (L.shape b.1).n)

/-- A tiled output-ticket protocol: the verifier's tiling of a layout, the activation rule per row, and per tile its
credit, its cap (its replayed debit within the cap), its `W_ref` (the draw's weight, from the layout) and `Wcred`, the
`W_ref` of its credited rows given the committed activations. -/
structure TileRules (Q R S : Type) where
  tiling : Layout → Tiling
  rowOK : Shape → Codes → ℕ → Prop
  credit : (Q → R) → S → Workload → ℕ → Codes → ℝ
  capOK : (Q → R) → S → Workload → ℕ → Codes → Prop
  Wref : Layout → ℕ → ℝ
  Wcred : Layout → ℕ → Codes → ℝ

variable {Q R S : Type}

/-- A good tile of a committed unit: its replayed debit is within the cap, each of its weight rows (its columns' rows of
its unit's weight) passes the rules, and each ticket and useful word of its credited rows (the activation rows that
pass the rules) has the reference FP32 value. Excluded rows are bound but uncredited, so their words don't enter γ. -/
def TileGood (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S) (g : ℕ) (c : Commit) :
    Prop :=
  TR.capOK H s U g c.act ∧
    (∀ j ∈ (TR.tiling U.layout).cols g,
      P.weightOK (U.layout.shape ((TR.tiling U.layout).unit g))
        (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j) ∧
    ∀ i ∈ (TR.tiling U.layout).rows g, TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) c.act i →
      ∀ j ∈ (TR.tiling U.layout).cols g,
        wordQ (c.ticket i j) = P.ticket H s U ((TR.tiling U.layout).unit g) c.act i j ∧
          wordQ (c.useful i j) = P.useful H s U ((TR.tiling U.layout).unit g) c.act i j

open Classical in
/-- The good tiles of `τ`. -/
noncomputable def goodTiles (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Transcript) : Finset ℕ :=
  (range (TR.tiling U.layout).NT).filter fun g =>
    ∃ c, τ ((TR.tiling U.layout).unit g) = some c ∧ TileGood P TR U H s g c

/-- A good tile under U-only binding (the `-h1` format's U-only messages): `TileGood` without its ticket conjunct. Its
replayed debit is within the cap, each of its weight rows passes the rules and each useful word of its credited rows has
the reference FP32 value; its ticket words are not read. -/
def TileGoodU (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S) (g : ℕ) (c : Commit) :
    Prop :=
  TR.capOK H s U g c.act ∧
    (∀ j ∈ (TR.tiling U.layout).cols g,
      P.weightOK (U.layout.shape ((TR.tiling U.layout).unit g))
        (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j) ∧
    ∀ i ∈ (TR.tiling U.layout).rows g, TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) c.act i →
      ∀ j ∈ (TR.tiling U.layout).cols g,
        wordQ (c.useful i j) = P.useful H s U ((TR.tiling U.layout).unit g) c.act i j

open Classical in
/-- The good tiles of `τ` under U-only binding. -/
noncomputable def goodTilesU (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Transcript) : Finset ℕ :=
  (range (TR.tiling U.layout).NT).filter fun g =>
    ∃ c, τ ((TR.tiling U.layout).unit g) = some c ∧ TileGoodU P TR U H s g c

open Classical in
/-- The tiles of the units `τ` leaves blank (declared uncredited): the audit never checks them, and they earn nothing. -/
noncomputable def blankTiles (TR : TileRules Q R S) (U : Workload) (τ : Transcript) : Finset ℕ :=
  (range (TR.tiling U.layout).NT).filter fun g => τ ((TR.tiling U.layout).unit g) = none

/-- The work `τ` claims: the `W_ref` of the credited rows of the tiles of the units it commits. -/
noncomputable def claimedWref (TR : TileRules Q R S) (U : Workload) (τ : Transcript) : ℝ :=
  ∑ g ∈ range (TR.tiling U.layout).NT \ blankTiles TR U τ,
    TR.Wcred U.layout g (actOf τ ((TR.tiling U.layout).unit g))

/-- The layout's `W_ref`, tile by tile. -/
def tileWrefTotal (TR : TileRules Q R S) (L : Layout) : ℝ := ∑ g ∈ range (TR.tiling L).NT, TR.Wref L g

/-- `AdmitsRef` with the reference's cost bounded tile by tile. -/
def AdmitsRefTile (CM : CostModel Q R S) (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop)
    (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ tileWrefTotal TR U.layout ∧
      CM.calls prog H U act [] s ≤ qref U

/-- **The γ-gap property under the sampled audit.** For every in-domain workload, budget `T`, oracle bound `q` and
adversary, every ticket count `w` proportional to the tiles' `W_ref` (`verity.protocols.accounting.work.pouw.audit`'s `tickets`), `t` independent
draws and `ε_s ≤ 1`: the audit accepts (every drawn tile is good, or of a blank unit) while `T` is below `1 − γ` of the
claimed work (the credited rows' `W_ref`) less `ε_s` of the layout's, with probability at most `(1 − ε_s)^t + η(q, N)`. -/
def GγSampled [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AuditAccepts (goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - γ) * (claimedWref TR U (A.transcript U x.1 x.2.1) - εs * tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + η q U.layout.N

/-- **The γ-gap property under the sampled audit with U-only binding**: `GγSampled` with the audit accepting a drawn tile
when it is good under U-only binding (`goodTilesU`) or of a blank unit. -/
def GγSampledU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AuditAccepts (goodTilesU P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - γ) * (claimedWref TR U (A.transcript U x.1 x.2.1) - εs * tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + η q U.layout.N

/-! ## Pearl-C's tiles -/

open Classical in
/-- A tile's credited rows: those whose activation row passes Π's rules. An excluded row stays in its unit, computed and
bound, and earns no credit. -/
noncomputable def PearlCSem.passRows (sem : PearlCSem Q R S) (sh : Shape) (act : Codes) (rows : Finset ℕ) : Finset ℕ :=
  rows.filter fun i => sem.RowOK sh.m sh.k act i

/-- A tile's share of its unit's words, `|rows|·|cols|/(m·n)`. Every term of `creditOf` and `wrefOf` is per word or per
activation row, so a tile carries this share of each: the per-row terms are split evenly over the unit's columns. -/
def tileShare (sh : Shape) (rows cols : Finset ℕ) : ℚ := (rows.card * cols.card : ℚ) / (sh.m * sh.n)

open Classical in
/-- **A tile's replayed debit** in H100 units, on the unit's noisy codes and flags (`unitChain`, `Skip.unitFlags`):
- **the chain:** `debitOf` of the tile's flagged atom-words, its words' share of the unit's `atoms` (additive over words,
  and each word's flags read only its two rows);
- **the forming, P3's per-tile rule** ("The per-tile cap"): each A′ element of
  the tile's rows that is salt-dead, or that no word of the tile needs (every word `(i, j)` with `j` in the tile's
  columns has the atom containing it flagged), at `fs` times the tile's column share `|cols|/n`, the share of the
  element's forming that the tile's credit carries. Needed is judged on the tile's own words, which over-debits against
  the unit and is sound; the shares over a row band's tiles sum to 1, so an element no word of the unit needs is debited
  once in all, and the tiles' forming debits add up to at least the unit's. -/
noncomputable def PearlCSem.tileDebit (sem : PearlCSem Q R S) (c : Costs) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := Skip.unitFlags AB.1 AB.2
  let atoms := ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
    ((range (sh.k / 32)).filter fun t => fl e.1 e.2 t = true).card
  let elems := ((univ : Finset (Fin sh.m × Fin sh.k)).filter fun e => (e.1 : ℕ) ∈ rows ∧
    (SaltDead.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨ ∀ j : Fin sh.n, (j : ℕ) ∈ cols → fl e.1 j ((e.2 : ℕ) / 32) = true)).card
  debitOf Params.pi c ⟨atoms, 0⟩ + (c.fs : ℚ) * (cols.card / sh.n) * elems

/-- **Pearl-C's tiles** on the audit's `64 × 64` grid, at costs `c` and cap `ρ`:
- `rowOK`: `PearlCSem.RowOK`, Π's rules on one activation row;
- `credit`: `(1 − ρ)·creditOf` times the credited rows' share, fixed;
- `capOK`: the replayed debit on the credited rows (`tileDebit`) is at most `ρ·creditOf` times their share;
- `Wref`: `wrefOf` times the tile's share, as `verity.protocols.accounting.work.pouw.audit`'s `tickets` weigh a tile once its `wref` is `wrefOf`;
- `Wcred`: `wrefOf` times the credited rows' share. -/
noncomputable def pearlCTiles (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) : TileRules Q R S where
  tiling := auditTiling 64 64
  rowOK := fun sh act i => sem.RowOK sh.m sh.k act i
  credit := fun _ _ U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    (((1 - ρ) * creditOf Params.pi c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)
  capOK := fun H s U g act =>
    let tl := auditTiling 64 64 U.layout
    let sh := U.layout.shape (tl.unit g)
    sem.tileDebit c H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
      ρ * creditOf Params.pi c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g)
  Wref := fun L g =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefOf Params.pi c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
  Wcred := fun L g act =>
    let tl := auditTiling 64 64 L
    let sh := L.shape (tl.unit g)
    ((wrefOf Params.pi c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)

end Pouw.PearlC
