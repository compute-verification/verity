import Proofs.Pouw.PearlPlus.Draw
import Proofs.Pouw.PearlPlus.Debit

/-!
# Pearl+'s bridge, part (c): a passed tile is within the game's inert-work bound

The bound the game reads is `TileGood`'s first conjunct, `capOK` of `pearlCTilesDevRev1 devH100` at the deployed
semantics and `ρ = 1/400`: the tile's replayed inert work (`tileDebitRev1`: its flagged atom-words, its later promotions
into a +0 total and its forming) on its rows that pass the rules and all its columns is at most `ρ·creditDevRev1` times
their share. Acceptance doesn't read it: `Draw` puts a sampled tile with good words in `goodDeclTiles` within the bound
and in `overCapTiles` over it. The credit does: the tolerant audit's deduction grows with the samples on
`overCapTiles`, and a passed tile within the bound keeps those to the samples the check finds over its own bound.

The check's bound (`Rule.capCheck`) sums the chain's inert work over the record's credited rows and columns
(`UnitRecord.credited`: real, neither excluded nor voluntary) and has no forming term. Two rules the check doesn't apply
yet close the difference, each stated here as the check's:
* the band rule (`bandCredited`): a credited row counts only if its 64-row band holds no voluntary A row, and a credited
  column only if its 64-column block holds no excluded or voluntary B row. A tile that keeps a row and a column carries
  a ticket (`CarriesATicket`). Such a tile has no voluntary row and no excluded or voluntary column, its filler rows are
  zero and fail the rules, and its excluded rows fail them, so the check's credited rows are the tile's `passRows` and
  its credited columns all its columns;
* the forming rule (`formingCapOK`): the check's bound with `64·|cc|` added to `n` times the replayed debit per forming
  element. A tile meets it at `E` forming elements (`MeetsTheFormingRule`) when the words its openings form do;
  `TilePassGivesTheCap` takes it at the tile's own (`formingElems`: the A′ elements of its rows that are salt-dead or
  that every column's word flags).

* `tilePassGivesTheCap`: a passed tile that carries a ticket and meets the forming rule is within the bound. The forming
  rule bounds the replayed debit and the forming, the replayed debit is the game's chain (`debitIsTheReplay`,
  `replayReadsTheCodes`) on rows that pass the rules exactly when the game's do (`RowsAreTheRules`), the forming is the
  rest of the game's inert work (`tileDebitRev1_split`), and the check's credit per credited row,
  `creditOf(1, kKept, n)`, is at most `creditDevRev1`'s (`credit_le`), with equality at `kKept = k`.
* `tilePassGivesTheChainCap`: on today's check (`TileKeepsTheCap`, the forming rule at no forming element), the chain's
  part of the game's inert work (`tileChainDebit`) is within the bound.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset

/-- **The band rule's credited rows and columns** of a tile: `UnitRecord.credited`'s, less every row whose 64-row band
holds a voluntary A row and every column whose 64-column block holds an excluded or voluntary B row. The check doesn't
apply it yet: its own credited set replaces this definition when it does. -/
def bandCredited (r : UnitRecord) (rows cols : List ℕ) : List ℕ × List ℕ :=
  ((r.credited rows cols).1.filter fun i => !r.voluntaryA.any (· / 64 == i / 64),
   (r.credited rows cols).2.filter fun j => !(r.excludedB ++ r.voluntaryB).any (· / 64 == j / 64))

/-- Tile `tid` of pass `p` carries a ticket under the band rule: it keeps a credited row and a credited column. -/
def CarriesATicket (p : Pass) (tid : ℕ) : Prop :=
  ∀ pos rows cols, p.tile? tid = some (pos, rows, cols) →
    (bandCredited p.records[pos]! rows cols).1 ≠ [] ∧ (bandCredited p.records[pos]! rows cols).2 ≠ []

open Classical in
/-- The chain's part of `tileDebitRev1`: `debitDev` of the tile's flagged atom-words, and its words' later promotions
into a +0 total at the add price. -/
noncomputable def tileChainDebit {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  debitDev d ⟨∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      ((range (sh.k / 32)).filter fun t => fl e.1 e.2 t = true).card, 0⟩ +
    d.prices.add * ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      (laterZeroPromos d AB.1 AB.2 e.1 e.2 : ℚ)

open Classical in
/-- A tile's forming elements: the A′ elements of its rows that are salt-dead or that every column's word flags. -/
noncomputable def formingElems {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℕ :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  ((univ : Finset (Fin sh.m × Fin sh.k)).filter fun e => (e.1 : ℕ) ∈ rows ∧
    (d.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨
      ∀ j : Fin sh.n, (j : ℕ) ∈ cols → fl e.1 j ((e.2 : ℕ) / 32) = true)).card

/-- The forming part of `tileDebitRev1`: `fs` per forming element, at the tile's share of the columns. -/
noncomputable def tileFormingDebit {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  (d.prices.costs.fs : ℚ) * (cols.card / (U.layout.shape u).n) * formingElems sem d H s U u rows cols act

/-- A tile's replayed inert work is its chain's and its forming's. -/
theorem tileDebitRev1_split {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) :
    sem.tileDebitRev1 d H s U u rows cols act =
      tileChainDebit sem d H s U u rows cols act + tileFormingDebit sem d H s U u rows cols act := by
  unfold PearlCSem.tileDebitRev1 PearlCSem.tileDebitDev tileChainDebit tileFormingDebit formingElems
  exact add_right_comm _ _ _

/-- **The forming rule**: the check's inert-work bound with a forming term, on the band rule's credited cells. The
cells' replayed debit, summed as `TileKeepsTheCap` sums it, counts `n` times, and each of `E` forming elements `64·|cc|`:
64 is the forming price of an element, and the element is formed once for the unit's `n` columns. The total is at most
`creditOf(1, kKept, n)` per credited cell at `ρ = CAP_NUM/CAP_DEN`. The check doesn't apply it yet. -/
def formingCapOK (r : UnitRecord) (rows cols : List ℕ) (ops : Words.Ops) (E : ℕ) : Prop :=
  let (cr, cc) := bandCredited r rows cols
  ∃ words : List (List (ℕ × ℕ)),
    cr.mapM (fun i => cc.mapM fun j => Rule.wordDebit ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]!) =
      .ok words ∧
    Rule.CAP_DEN * (r.matmul.n * (words.map fun ws => Rule.debitUnits (ws.foldl (· + ·.1) 0)
      (ws.foldl (· + ·.2) 0)).foldl (· + ·) 0 + 64 * cc.length * E) ≤
      Rule.CAP_NUM * PouwCheck.PearlPlus.creditOf 1 r.kKept r.matmul.n * cr.length * cc.length

/-- Tile `tid` of pass `p` meets the forming rule at `E` forming elements: `formingCapOK` holds on the words its
openings `o` form. -/
def MeetsTheFormingRule (p : Pass) (tid : ℕ) (o : TileOpening) (E : ℕ) : Prop :=
  ∀ pos rows cols ops ch, p.tile? tid = some (pos, rows, cols) →
    Properties.Pouw.PearlPlus.formed p (Properties.Pouw.PearlPlus.noiseOf p tid) rows cols p.records[pos]!.matmul o =
      .ok (ops, ch) → formingCapOK p.records[pos]! rows cols ops E

/-- **(c) A passed tile's chain is within the inert-work bound**: a tile as in `TilePassGivesTheCap` that carries a
ticket has the chain's part of its replayed inert work (`tileChainDebit`) within the game's bound at `ρ = 1/400`. -/
def TilePassGivesTheChainCap : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCommit.CoinCommitment) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.wid ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).weightOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.weight (U.layout.wid ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c → CarriesATicket p tid →
    checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass →
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      let P := (semH3 saltOf rootOf : PearlCSem Q R S).passRows sh c.act (tl.rows g)
      tileChainDebit (semH3 saltOf rootOf) devH100 H s U (tl.unit g) P (tl.cols g) c.act ≤
        1 / 400 * creditDevRev1 devH100 sh * tileShare sh P (tl.cols g)

/-- **(c) A passed tile is within the inert-work bound**: a tile of a pass `load` gives, which is audit tile `g` of `U`,
under the semantics' salt and weight root, whose weight rows all pass the rules (`TileGood`'s second conjunct, as (d)
takes it), whose openings are its unit's commitment `c`, that carries a ticket under the band rule, that meets the
forming rule at its forming elements, and that `checkTile` passes on its weight's noise, has its replayed inert work
within the game's bound at `ρ = 1/400`. -/
def TilePassGivesTheCap : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCommit.CoinCommitment) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.wid ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).weightOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.weight (U.layout.wid ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c → CarriesATicket p tid →
    MeetsTheFormingRule p tid o (formingElems (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g)
      ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
        ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act) →
    checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass →
      (tilesH3 saltOf rootOf (1 / 400) : Pouw.PearlC.TileRules Q R S).capOK H s U g c.act

end Pouw.PearlPlus.Bridge

namespace Pouw.PearlPlus.Cap

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset

/-! ## A tile's rows and columns -/

theorem tile_nodup {p : Pass} {tid pos : ℕ} {rows cols : List ℕ} (h : p.tile? tid = some (pos, rows, cols)) :
    rows.Nodup ∧ cols.Nodup := by
  obtain ⟨hp, x, hx, -, hg⟩ := Draw.tile?_some h
  have h1 : (Draw.grid p.scheme p.records[pos])[x]? = some (rows, cols) := by
    rw [List.getElem?_eq_getElem hx, ← getElem!_pos (Draw.grid p.scheme p.records[pos]) x hx, hg]
  rw [Draw.grid, Draw.tiles_getElem?] at h1
  split at h1
  · simp only [Option.some.injEq, Prod.mk.injEq] at h1
    obtain ⟨rfl, rfl⟩ := h1
    exact ⟨Draw.block_nodup _ _ _, Draw.block_nodup _ _ _⟩
  · cases h1

theorem rows_band {L : Pouw.PearlC.Layout} {g i i' : ℕ} (hi : i ∈ (auditTiling 64 64 L).rows g)
    (hi' : i' ∈ (auditTiling 64 64 L).rows g) : i / 64 = i' / 64 := by
  have h1 := Finset.mem_Ico.1 hi
  have h2 := Finset.mem_Ico.1 hi'
  have h1' := lt_of_lt_of_le h1.2 (min_le_left _ _)
  have h2' := lt_of_lt_of_le h2.2 (min_le_left _ _)
  omega

theorem cols_band {L : Pouw.PearlC.Layout} {g j j' : ℕ} (hj : j ∈ (auditTiling 64 64 L).cols g)
    (hj' : j' ∈ (auditTiling 64 64 L).cols g) : j / 64 = j' / 64 := by
  have h1 := Finset.mem_Ico.1 hj
  have h2 := Finset.mem_Ico.1 hj'
  have h1' := lt_of_lt_of_le h1.2 (min_le_left _ _)
  have h2' := lt_of_lt_of_le h2.2 (min_le_left _ _)
  omega

/-- A tile in one 64-row band that keeps a row under the band rule has no voluntary A row. -/
theorem band_vol {r : UnitRecord} {rows cols : List ℕ} (hband : ∀ i ∈ rows, ∀ i' ∈ rows, i / 64 = i' / 64)
    (h : (Bridge.bandCredited r rows cols).1 ≠ []) (i : ℕ) (hi : i ∈ rows) : r.voluntaryA.contains i = false := by
  obtain ⟨i0, hi0⟩ := List.exists_mem_of_ne_nil _ h
  simp only [Bridge.bandCredited, UnitRecord.credited, List.mem_filter] at hi0
  obtain ⟨⟨hi0r, -⟩, hno⟩ := hi0
  cases hc : r.voluntaryA.contains i
  · rfl
  · have hv : i ∈ r.voluntaryA := List.contains_iff_mem.1 hc
    have : r.voluntaryA.any (· / 64 == i0 / 64) = true :=
      List.any_eq_true.2 ⟨i, hv, by simp [hband i hi i0 hi0r]⟩
    simp [this] at hno

/-- A tile in one 64-column block that keeps a column under the band rule has no excluded or voluntary B row. -/
theorem band_colsB {r : UnitRecord} {rows cols : List ℕ} (hband : ∀ j ∈ cols, ∀ j' ∈ cols, j / 64 = j' / 64)
    (h : (Bridge.bandCredited r rows cols).2 ≠ []) (j : ℕ) (hj : j ∈ cols) :
    r.excludedB.contains j = false ∧ r.voluntaryB.contains j = false := by
  obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil _ h
  simp only [Bridge.bandCredited, UnitRecord.credited, List.mem_filter] at hj0
  obtain ⟨⟨hj0c, -⟩, hno⟩ := hj0
  have key : ∀ l : List ℕ, (∀ a ∈ l, a ∈ r.excludedB ++ r.voluntaryB) → l.contains j = true → False := by
    intro l hl hc
    have hv := hl j (List.contains_iff_mem.1 hc)
    have : (r.excludedB ++ r.voluntaryB).any (· / 64 == j0 / 64) = true :=
      List.any_eq_true.2 ⟨j, hv, by simp [hband j hj j0 hj0c]⟩
    simp [this] at hno
  constructor
  · cases hc : r.excludedB.contains j
    · rfl
    · exact (key _ (fun a ha => List.mem_append_left _ ha) hc).elim
  · cases hc : r.voluntaryB.contains j
    · rfl
    · exact (key _ (fun a ha => List.mem_append_right _ ha) hc).elim

/-- On a tile in one 64-row band and one 64-column block that keeps a row and a column under the band rule, the band
rule keeps all of `UnitRecord.credited`. -/
theorem band_eq {r : UnitRecord} {rows cols : List ℕ} (hbandR : ∀ i ∈ rows, ∀ i' ∈ rows, i / 64 = i' / 64)
    (hbandC : ∀ j ∈ cols, ∀ j' ∈ cols, j / 64 = j' / 64) (hA : (Bridge.bandCredited r rows cols).1 ≠ [])
    (hB : (Bridge.bandCredited r rows cols).2 ≠ []) : Bridge.bandCredited r rows cols = r.credited rows cols := by
  obtain ⟨i0, hi0⟩ := List.exists_mem_of_ne_nil _ hA
  obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil _ hB
  simp only [Bridge.bandCredited, List.mem_filter] at hi0 hj0
  obtain ⟨hi0c, hi0n⟩ := hi0
  obtain ⟨hj0c, hj0n⟩ := hj0
  have hi0r : i0 ∈ rows := (List.mem_filter.1 hi0c).1
  have hj0r : j0 ∈ cols := (List.mem_filter.1 hj0c).1
  unfold Bridge.bandCredited
  refine Prod.ext (List.filter_eq_self.2 fun i hi => ?_) (List.filter_eq_self.2 fun j hj => ?_)
  · try dsimp only
    rw [hbandR i (List.mem_filter.1 hi).1 i0 hi0r]
    exact hi0n
  · try dsimp only
    rw [hbandC j (List.mem_filter.1 hj).1 j0 hj0r]
    exact hj0n

/-! ## What a passed tile's check keeps of its filler and excluded rows -/

theorem rowsFailure_filler {r : UnitRecord} {rows cols : List ℕ} {ops : Words.Ops}
    (h : Rule.rowsFailure r rows cols ops = none) {x : ℕ} (hx : x < rows.length) (hf : r.realRows ≤ rows[x]) :
    (ops.a.x[x]!.any fun w => w != 0) = false := by
  unfold Rule.rowsFailure at h
  dsimp only [Id.run, bind, pure] at h
  split at h
  · rename_i r' heq
    have hs := Draw.forIn_some ?_ heq
    · obtain ⟨m, rfl⟩ := hs
      cases h
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
  · rename_i heq
    have hall := Draw.forIn_none _ _ ?_ heq
    rotate_left
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
    have hmem := hall ("A", rows[x], ops.a, x, Rule.Kind.filler) (List.mem_append_left _ (List.mem_map.2
      ⟨(rows[x], x), List.mem_zipIdx_iff_getElem?.2 (by simp [hx]), by
        dsimp only
        rw [ite_eq_left (show rows[x] ≥ r.realRows from hf)]⟩))
    dsimp only at hmem
    simp only [show (Rule.Kind.filler == Rule.Kind.filler) = true from rfl, ↓reduceIte] at hmem
    split at hmem
    · cases hmem
    · rename_i hz
      exact Bool.eq_false_iff.2 hz

theorem rowsFailure_excluded {r : UnitRecord} {rows cols : List ℕ} {ops : Words.Ops}
    (h : Rule.rowsFailure r rows cols ops = none) {x : ℕ} (hx : x < rows.length) (hlt : rows[x] < r.realRows)
    (hex : r.excludedA.contains rows[x] = true) :
    Rule.rowPasses ops.a.rho[x]! ops.a.alpha[x]! ops.a.x[x]! = false := by
  unfold Rule.rowsFailure at h
  dsimp only [Id.run, bind, pure] at h
  split at h
  · rename_i r' heq
    have hs := Draw.forIn_some ?_ heq
    · obtain ⟨m, rfl⟩ := hs
      cases h
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
  · rename_i heq
    have hall := Draw.forIn_none _ _ ?_ heq
    rotate_left
    · intro a
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩
    have hmem := hall ("A", rows[x], ops.a, x, Rule.Kind.excluded) (List.mem_append_left _ (List.mem_map.2
      ⟨(rows[x], x), List.mem_zipIdx_iff_getElem?.2 (by simp [hx]), by
        dsimp only
        rw [ite_eq_right (Nat.not_le.2 hlt), ite_eq_left hex]⟩))
    dsimp only at hmem
    simp only [show (Rule.Kind.excluded == Rule.Kind.filler) = false from rfl,
      show (Rule.Kind.excluded == Rule.Kind.voluntary) = false from rfl,
      show (Rule.Kind.excluded == Rule.Kind.credited) = false from rfl,
      show (Rule.Kind.excluded == Rule.Kind.excluded) = true from rfl] at hmem
    simp only [Bool.false_eq_true, ↓reduceIte, Bool.true_and, Bool.false_and, Bool.false_or] at hmem
    split at hmem
    · cases hmem
    · split at hmem
      · cases hmem
      · rename_i hrp
        exact Bool.eq_false_iff.2 hrp

/-- A tile whose rows keep the rules and that is within the check's bound passes `check_opened`. -/
theorem checkOpened_none {s : Scheme} {r : UnitRecord} {rows cols : List ℕ} {ops : Words.Ops}
    (hr : Rule.rowsFailure r rows cols ops = none) (hc : Rule.capCheck s r rows cols ops = .ok none) :
    Rule.checkOpened s r rows cols ops = none := by
  simp only [Rule.checkOpened, Rule.capFailure, hr, hc, Option.orElse_none]

theorem zero_of_any {x : Array ℕ} (h : (x.any fun w => w != 0) = false) (l : ℕ) : x[l]! = 0 := by
  by_cases hl : l < x.size
  · have h1 := Array.any_eq_false.1 h l hl
    rw [getElem!_pos x l hl]
    simpa using h1
  · rw [getElem!_neg x l hl]
    rfl

/-- A row of zeros isn't live. -/
theorem liveRow_zero {x : Array ℕ} (hk : 0 < x.size) (hz : ∀ l : ℕ, x[l]! = 0) : Rule.liveRow x = false := by
  have hs : Rule.sq300 0 = 0 := by decide
  rw [Bridge.liveRow_eq]
  simp [hz, hs]
  omega

/-! ## The check's replayed debit, summed -/

theorem mapM_ok_eq {ε α β : Type} {f : α → Except ε β} {g : α → β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → (∀ a ∈ l, ∀ b, f a = .ok b → b = g a) → l' = l.map g
  | [], l', h, _ => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl
  | a :: l, l', h, hg => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := Assembly.bind_ok h
    obtain ⟨bs, hbs, h⟩ := Assembly.bind_ok h
    cases h
    rw [List.map_cons]
    exact congrArg₂ List.cons (hg a List.mem_cons_self b0 h0)
      (mapM_ok_eq hbs fun a' ha' b hb => hg a' (List.mem_cons_of_mem _ ha') b hb)

theorem foldl_add_eq : ∀ (L : List ℕ) (a : ℕ), L.foldl (· + ·) a = a + L.sum
  | [], a => by simp
  | b :: L, a => by rw [List.foldl_cons, foldl_add_eq L, List.sum_cons, Nat.add_assoc]

theorem foldl_fst_eq : ∀ (L : List (ℕ × ℕ)) (a : ℕ), L.foldl (· + ·.1) a = a + (L.map Prod.fst).sum
  | [], a => by simp
  | b :: L, a => by rw [List.foldl_cons, foldl_fst_eq L, List.map_cons, List.sum_cons, Nat.add_assoc]

theorem foldl_snd_eq : ∀ (L : List (ℕ × ℕ)) (a : ℕ), L.foldl (· + ·.2) a = a + (L.map Prod.snd).sum
  | [], a => by simp
  | b :: L, a => by rw [List.foldl_cons, foldl_snd_eq L, List.map_cons, List.sum_cons, Nat.add_assoc]

theorem debitUnits_eq (a z : ℕ) : Rule.debitUnits a z = 40 * a + 32 * z := by
  show (32 + 32 / 4) * a + 32 * z = 40 * a + 32 * z
  norm_num

/-- The check's tile debit, on words whose debits are `(F i j, Z i j)`, is `40·F + 32·Z` summed over its cells. -/
theorem debit_sum (cr cc : List ℕ) (hcr : cr.Nodup) (hcc : cc.Nodup) (F Z : ℕ → ℕ → ℕ) :
    ((cr.map fun i => cc.map fun j => (F i j, Z i j)).map fun ws =>
      Rule.debitUnits (ws.foldl (· + ·.1) 0) (ws.foldl (· + ·.2) 0)).foldl (· + ·) 0 =
      ∑ i ∈ cr.toFinset, ∑ j ∈ cc.toFinset, (40 * F i j + 32 * Z i j) := by
  rw [foldl_add_eq, Nat.zero_add, List.map_map, ← List.sum_toFinset _ hcr]
  refine Finset.sum_congr rfl fun i _ => ?_
  show Rule.debitUnits ((cc.map fun j => (F i j, Z i j)).foldl (· + ·.1) 0)
    ((cc.map fun j => (F i j, Z i j)).foldl (· + ·.2) 0) = _
  rw [foldl_fst_eq, foldl_snd_eq, Nat.zero_add, Nat.zero_add, List.map_map, List.map_map, debitUnits_eq,
    ← List.sum_toFinset _ hcc, ← List.sum_toFinset _ hcc, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  rfl

/-! ## The game's chain debit, cell by cell -/

open Classical in
/-- Cell `(i, j)`'s flagged atoms at the H100 record, `0` off the unit. -/
noncomputable def cellF {Q R S : Type} (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Pouw.PearlC.Workload) (u : ℕ)
    (act : Codes) (i j : ℕ) : ℕ :=
  if h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n then
    ((range ((U.layout.shape u).k / 32)).filter fun t =>
      devH100.flags (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 ⟨i, h.1⟩ ⟨j, h.2⟩ t = true).card
  else 0

/-- Cell `(i, j)`'s later promotions into a +0 total at the H100 record, `0` off the unit. -/
noncomputable def cellZ {Q R S : Type} (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Pouw.PearlC.Workload) (u : ℕ)
    (act : Codes) (i j : ℕ) : ℕ :=
  if h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n then
    laterZeroPromos devH100 (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 ⟨i, h.1⟩ ⟨j, h.2⟩
  else 0

theorem sum_fin_filter {β : Type} [AddCommMonoid β] {m n : ℕ} (P C : Finset ℕ) (hP : ∀ i ∈ P, i < m)
    (hC : ∀ j ∈ C, j < n) {inst : DecidablePred fun e : Fin m × Fin n => (e.1 : ℕ) ∈ P ∧ (e.2 : ℕ) ∈ C}
    (F : Fin m × Fin n → β) :
    ∑ e ∈ @Finset.filter _ (fun e : Fin m × Fin n => (e.1 : ℕ) ∈ P ∧ (e.2 : ℕ) ∈ C) inst univ, F e =
      ∑ i ∈ P, ∑ j ∈ C, if h : i < m ∧ j < n then F (⟨i, h.1⟩, ⟨j, h.2⟩) else 0 := by
  rw [← Finset.sum_product']
  symm
  refine Finset.sum_bij'
    (fun q hq => ((⟨q.1, hP _ (Finset.mem_product.1 hq).1⟩ : Fin m), (⟨q.2, hC _ (Finset.mem_product.1 hq).2⟩ : Fin n)))
    (fun e _ => ((e.1 : ℕ), (e.2 : ℕ))) ?_ ?_ ?_ ?_ ?_
  · intro q hq
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact Finset.mem_product.1 hq
  · intro e he
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at he
    exact Finset.mem_product.2 he
  · intro q _
    rfl
  · intro e _
    rfl
  · intro q hq
    rw [dite_eq_left ⟨hP _ (Finset.mem_product.1 hq).1, hC _ (Finset.mem_product.1 hq).2⟩]

/-- At the H100 record a tile's chain debit is `40` per flagged atom and `32` per later +0 promotion, summed over its
cells. -/
theorem tileChainDebit_cells {Q R S : Type} (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Pouw.PearlC.Workload)
    (u : ℕ) (act : Codes) (P C : Finset ℕ) (hP : ∀ i ∈ P, i < (U.layout.shape u).m)
    (hC : ∀ j ∈ C, j < (U.layout.shape u).n) :
    Bridge.tileChainDebit sem devH100 H s U u P C act =
      ((∑ i ∈ P, ∑ j ∈ C, (40 * cellF sem H s U u act i j + 32 * cellZ sem H s U u act i j) : ℕ) : ℚ) := by
  have hG : devH100.G = 4 := rfl
  have hadd : devH100.prices.add = 32 := rfl
  simp only [Bridge.tileChainDebit, debitDev, hG, hadd, ite_eq_right (show (4 : ℕ) ≠ 0 by decide), Nat.cast_sum,
    Nat.cast_zero, mul_zero, add_zero]
  rw [sum_fin_filter P C hP hC, sum_fin_filter P C hP hC, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j hj => ?_
  have h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n := ⟨hP i hi, hC j hj⟩
  simp only [cellF, cellZ, dite_eq_left h]
  push_cast
  ring

/-! ## The credit -/

theorem creditDevRev1_h100 (sh : Pouw.PearlC.Shape) :
    creditDevRev1 devH100 sh = 5 / 4 * sh.m * sh.k * sh.n + 96 * sh.m * sh.k + 96 * sh.m * sh.n := by
  have hG : devH100.G = 4 := rfl
  have hadd : devH100.prices.add = 32 := rfl
  have hbf : devH100.prices.bf16 = 2 := rfl
  have hfs : devH100.prices.costs.fs = 64 := rfl
  have hr : Params.pi.r = 32 := rfl
  simp only [creditDevRev1, creditDev, firstAddDev, hG, hadd, hbf, hfs, hr, ite_eq_right (show (4 : ℕ) ≠ 0 by decide)]
  push_cast
  ring

/-- Four times the check's credit per credited row, at a kept width within a whole number of windows, is at most
`5nk + 384k + 384n`, four times `creditDevRev1`'s per row. -/
theorem credit_nat {K k n : ℕ} (hK : K ≤ k) (hk : k % 128 = 0) (hk0 : 0 < k) :
    4 * PouwCheck.PearlPlus.creditOf 1 K n ≤ 5 * (n * k) + 384 * k + 384 * n := by
  simp only [PouwCheck.PearlPlus.creditOf, PouwCheck.PearlPlus.wref, PouwCheck.PearlPlus.FADD,
    PouwCheck.PearlPlus.WINDOW, PouwCheck.PearlPlus.R, PouwCheck.PearlPlus.BF16_MAC, PouwCheck.PearlPlus.PEEL_K,
    PouwCheck.PearlPlus.FORMING, Nat.one_mul, Nat.mul_one]
  rw [Nat.mul_add, Nat.mul_left_comm n 32]
  have h1 : 128 * (n * (K / 128)) ≤ n * K := by
    have := Nat.mul_le_mul_left n (Nat.mul_div_le K 128)
    rwa [Nat.mul_left_comm n 128] at this
  have h2 : n * K ≤ n * k := Nat.mul_le_mul_left n hK
  have h3 : n * 128 ≤ n * k := Nat.mul_le_mul_left n (by omega)
  omega

theorem credit_le (sh : Pouw.PearlC.Shape) {K : ℕ} (hK : K ≤ sh.k) (hk : sh.k % 128 = 0) (hk0 : 0 < sh.k) :
    (sh.m : ℚ) * (PouwCheck.PearlPlus.creditOf 1 K sh.n : ℚ) ≤ creditDevRev1 devH100 sh := by
  rw [creditDevRev1_h100]
  have h : (4 * PouwCheck.PearlPlus.creditOf 1 K sh.n : ℚ) ≤ 5 * (sh.n * sh.k) + 384 * sh.k + 384 * sh.n := by
    exact_mod_cast credit_nat hK hk hk0
  have hm : (0 : ℚ) ≤ sh.m := Nat.cast_nonneg _
  nlinarith [mul_le_mul_of_nonneg_left h hm]

theorem bound_glue {D c Cr a b m n : ℚ} (hm : 0 < m) (hn : 0 < n) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hD : D * (400 * n) ≤ c * a * b) (hc : m * c ≤ Cr) : D ≤ 1 / 400 * Cr * (a * b / (m * n)) := by
  rw [show 1 / 400 * Cr * (a * b / (m * n)) = Cr * (a * b) / (400 * m * n) by ring,
    le_div_iff₀ (mul_pos (mul_pos (by norm_num) hm) hn)]
  calc D * (400 * m * n) = m * (D * (400 * n)) := by ring
    _ ≤ m * (c * a * b) := mul_le_mul_of_nonneg_left hD hm.le
    _ = (m * c) * (a * b) := by ring
    _ ≤ Cr * (a * b) := mul_le_mul_of_nonneg_right hc (mul_nonneg ha hb)

theorem bound_glue_forming {D f E c Cr a b m n : ℚ} (hm : 0 < m) (hn : 0 < n) (ha : 0 ≤ a) (hb : 0 ≤ b) (hf : f = 64)
    (hD : 400 * (n * D + 64 * b * E) ≤ c * a * b) (hc : m * c ≤ Cr) :
    D + f * (b / n) * E ≤ 1 / 400 * Cr * (a * b / (m * n)) := by
  subst hf
  have hn1 : n * n⁻¹ = 1 := mul_inv_cancel₀ hn.ne'
  have h : (D + 64 * (b / n) * E) * (400 * n) = 400 * (n * D + 64 * b * E) :=
    calc (D + 64 * (b / n) * E) * (400 * n) = 400 * (n * D) + 400 * (64 * b * E) * (n * n⁻¹) := by ring
      _ = 400 * (n * D + 64 * b * E) := by rw [hn1]; ring
  exact bound_glue hm hn ha hb (by rw [h]; exact hD) hc

end Pouw.PearlPlus.Cap

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset

/-- **(c) at `E` forming elements** from what a passed tile establishes (`TileAcceptMeans`), what a loaded pass's
records keep (`LoadedGivesUp`), the per-row rules (`RowsAreTheRules`) and the replay (`debitIsTheReplay`,
`replayReadsTheCodes`): a tile as in `TilePassGivesTheCap` that meets the forming rule at `E` has its chain's inert work
and `fs` per forming element, at its share of the columns, within the game's bound. -/
theorem tile_bound (hT : Properties.Pouw.PearlPlus.TileAcceptMeans) (hL : Properties.Pouw.PearlPlus.LoadedGivesUp)
    (hR : RowsAreTheRules) {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCommit.CoinCommitment) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit) (E : ℕ)
    (hload : PouwCheck.PearlPlus.load coins man = .ok p) (hof : TileOf p tid U g) (hsalt : bytesNat p.salt = saltOf s)
    (hroot : bytesNat ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD
      ByteArray.empty) = rootOf (U.layout.wid ((auditTiling 64 64 U.layout).unit g)))
    (hwOK : ∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).weightOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.weight (U.layout.wid ((auditTiling 64 64 U.layout).unit g))) j)
    (hopen : OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c) (hticket : CarriesATicket p tid)
    (hforming : MeetsTheFormingRule p tid o E)
    (hpass : checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass) :
    tileChainDebit (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g)
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act +
      (devH100.prices.costs.fs : ℚ) * (((auditTiling 64 64 U.layout).cols g).card /
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n) * E ≤
      1 / 400 * creditDevRev1 devH100 (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
        tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
          ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
            ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) := by
  obtain ⟨pos, rows, cols, htile, -, hunit, -, hk, hn, hwid, hrows, hcols⟩ := hof
  obtain ⟨pos', rows', cols', ops, ch, htile', hrest⟩ := hT p _ tid o none hpass
  simp only [htile, Option.some.injEq, Prod.mk.injEq] at htile'
  obtain ⟨rfl, rfl, rfl⟩ := htile'
  try dsimp only at hrest
  obtain ⟨-, -, -, -, -, -, hform, -, hchk, -, -⟩ := hrest
  obtain ⟨hp, -⟩ := Draw.tile?_some htile
  have hrec : p.records[pos]! ∈ p.records := by rw [getElem!_pos p.records pos hp]; exact List.getElem_mem hp
  obtain ⟨-, -, hkept0, hkept⟩ := Draw.reason_none ((hL coins man p hload).1 _ hrec).1
  obtain ⟨hKmax, hK128⟩ := Assembly.record_domain htile (.inl hpass)
  obtain ⟨wn', hwn, henc, -⟩ := Assembly.formed_ok hform
  rw [Assembly.noiseOf_eq htile, ← hk, ← hwid] at hwn
  obtain ⟨hoa, hob, -, hact, hwt, -⟩ := hopen pos rows cols htile
  rw [← hunit, ← hk, ← hwid, hoa, hob] at henc
  have hk0 : 0 < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k := by rw [hk]; omega
  have hKk : p.records[pos]!.kKept ≤ (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k := by
    rw [hk]; exact hkept
  have hk128 : (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k % 128 = 0 := by rw [hk]; exact hK128
  have hKmax' : (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k ≤ 2 ^ 16 := by rw [hk]; exact hKmax
  obtain ⟨hrowsnd, hcolsnd⟩ := Cap.tile_nodup htile
  have hrowsm : ∀ i ∈ rows, i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m := fun i hi =>
    Assembly.rows_lt (by rw [hrows]; exact List.mem_toFinset.2 hi)
  have hcolsn : ∀ j ∈ cols, j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n := fun j hj =>
    Assembly.cols_lt (by rw [hcols]; exact List.mem_toFinset.2 hj)
  obtain ⟨hbA, hbB⟩ := hticket pos rows cols htile
  have hbandR : ∀ i ∈ rows, ∀ i' ∈ rows, i / 64 = i' / 64 := fun i hi i' hi' =>
    Cap.rows_band (L := U.layout) (g := g) (by rw [hrows]; exact List.mem_toFinset.2 hi)
      (by rw [hrows]; exact List.mem_toFinset.2 hi')
  have hbandC : ∀ j ∈ cols, ∀ j' ∈ cols, j / 64 = j' / 64 := fun j hj j' hj' =>
    Cap.cols_band (L := U.layout) (g := g) (by rw [hcols]; exact List.mem_toFinset.2 hj)
      (by rw [hcols]; exact List.mem_toFinset.2 hj')
  have hvolA := Cap.band_vol hbandR hbA
  have hB := Cap.band_colsB hbandC hbB
  have hcc : (p.records[pos]!.credited rows cols).2 = cols := by
    simp only [UnitRecord.credited]
    exact List.filter_eq_self.2 fun j hj => by
      simp only [(hB j hj).1, (hB j hj).2, Bool.not_false, Bool.and_self]
  have hrow : ∀ x (hx : x < rows.length),
      (ops.a.x[x]!).size = (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k ∧
      (Rule.rowPasses ops.a.rho[x]! ops.a.alpha[x]! ops.a.x[x]! = true ↔
        (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
          (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act rows[x]) := by
    intro x hx
    have him := hrowsm _ (List.getElem_mem hx)
    obtain ⟨hparse, s', β, hst⟩ := Draw.encode_rowA (by simp) henc x hx
    rw [getElem!_pos _ x (by simpa using hx), List.getElem_map, Draw.rowBytesOf_eq] at hparse
    obtain ⟨hsz, hwords⟩ := Draw.parseRow_wordBytes hparse
    have hW : ∀ l < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k,
        (ops.a.x[x]!)[l]! = c.act rows[x] l ∧ c.act rows[x] l < 2 ^ 32 ∧ (wordQ (c.act rows[x] l)).isSome := by
      intro l hl
      have h32 := hact rows[x] (List.getElem_mem hx) l hl
      obtain ⟨hxl, hfin⟩ := hwords l hl
      rw [Nat.mod_eq_of_lt h32] at hxl
      refine ⟨hxl, h32, ?_⟩
      rw [hxl, word_and_ff] at hfin
      have hne : (c.act rows[x] l >>> 23) % 256 ≠ 255 := by simpa using hfin
      rw [Pouw.PearlC.PeelExactProofs.wordQ_fields hne]
      rfl
    exact ⟨hsz, @hR Q R S saltOf rootOf _ _ c.act rows[x] ops.a.x[x]! s' _ _ β him (by omega) (by omega) (by omega)
      hsz hW hst⟩
  have hcr_iff : ∀ i, i ∈ (p.records[pos]!.credited rows cols).1 ↔
      i ∈ rows ∧ (semH3 saltOf rootOf : PearlCSem Q R S).RowOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act i := by
    intro i
    simp only [UnitRecord.credited, List.mem_filter]
    constructor
    · rintro ⟨hi, hc⟩
      obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
      exact ⟨hi, (hrow x hx).2.1 (Draw.rowsFailure_credited hchk hx hc)⟩
    · rintro ⟨hi, hok⟩
      obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
      have hlt : rows[x] < p.records[pos]!.realRows := by
        by_contra hge
        have hz := Cap.rowsFailure_filler hchk hx (Nat.le_of_not_lt hge)
        have hlive := Cap.liveRow_zero (by rw [(hrow x hx).1]; exact hk0) (Cap.zero_of_any hz)
        have h2 := (hrow x hx).2.2 hok
        rw [Rule.rowPasses, hlive, Bool.and_false] at h2
        cases h2
      have hex : p.records[pos]!.excludedA.contains rows[x] = false := by
        cases hc : p.records[pos]!.excludedA.contains rows[x]
        · rfl
        · have h1 := Cap.rowsFailure_excluded hchk hx hlt hc
          have h2 := (hrow x hx).2.2 hok
          rw [h1] at h2
          cases h2
      exact ⟨hi, by simp only [decide_eq_true hlt, hex, hvolA _ hi, Bool.not_false, Bool.and_self]⟩
  have hcrnd : (p.records[pos]!.credited rows cols).1.Nodup := hrowsnd.filter _
  have hcrne : (p.records[pos]!.credited rows cols).1 ≠ [] := by
    obtain ⟨i0, hi0⟩ := List.exists_mem_of_ne_nil _ hbA
    exact List.ne_nil_of_mem (List.mem_filter.1 hi0).1
  have hccne : cols ≠ [] := by
    obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil _ hbB
    have := List.ne_nil_of_mem (List.mem_filter.1 hj0).1
    rwa [hcc] at this
  obtain ⟨cr, hcrdef⟩ : ∃ cr, (p.records[pos]!.credited rows cols).1 = cr := ⟨_, rfl⟩
  rw [hcrdef] at hcrnd hcrne hcr_iff
  have hpair : p.records[pos]!.credited rows cols = (cr, cols) := Prod.ext hcrdef hcc
  have hF := hforming pos rows cols ops ch htile hform
  unfold formingCapOK at hF
  rw [Cap.band_eq hbandR hbandC hbA hbB, hpair] at hF
  try dsimp only at hF
  obtain ⟨words, hw, hcap⟩ := hF
  have hCC := replayReadsTheCodes saltOf rootOf H s U ((auditTiling 64 64 U.layout).unit g) c.act p.salt
    ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) wn' rows cols
    ops hsalt hroot hKmax' hrowsm hcolsn hact hwt
    (fun j hj => hwOK j (by rw [hcols]; exact List.mem_toFinset.2 hj)) hwn henc
  have hcellv : ∀ i ∈ cr, ∀ j ∈ cols, ∀ v : ℕ × ℕ,
      Rule.wordDebit ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]! = .ok v →
      v = (Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j,
        Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j) := by
    intro i hi j hj v hv
    obtain ⟨hir, hok⟩ := (hcr_iff i).1 hi
    have him := hrowsm i hir
    have hjn := hcolsn j hj
    have hx := List.idxOf_lt_length_of_mem hir
    have hz := List.idxOf_lt_length_of_mem hj
    have hcodes := hCC (rows.idxOf i) (cols.idxOf j) ⟨i, him⟩ ⟨j, hjn⟩
      (by rw [List.getElem?_eq_getElem hx, List.getElem_idxOf hx])
      (by rw [List.getElem?_eq_getElem hz, List.getElem_idxOf hz]) hok
    rw [hcodes] at hv
    obtain ⟨f, z⟩ := v
    obtain ⟨hf, hz'⟩ := debitIsTheReplay _ _ (⟨i, him⟩ : Fin _) (⟨j, hjn⟩ : Fin _) f z hv
    have h : i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m ∧
        j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n := ⟨him, hjn⟩
    rw [Cap.cellF, Cap.cellZ, dite_eq_left h, dite_eq_left h, hf, hz']
    rfl
  have hwords := Cap.mapM_ok_eq
    (g := fun i => cols.map fun j =>
      (Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j,
        Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j)) hw
    fun i hi ws hws => Cap.mapM_ok_eq hws fun j hj v hv => hcellv i hi j hj v hv
  rw [hwords, Cap.debit_sum cr cols hcrnd hcolsnd
    (Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act)
    (Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act), ← hn] at hcap
  simp only [Rule.CAP_NUM, Rule.CAP_DEN, Nat.one_mul] at hcap
  have hP : cr.toFinset = (semH3 saltOf rootOf : PearlCSem Q R S).passRows
      (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act ((auditTiling 64 64 U.layout).rows g) := by
    ext i
    rw [List.mem_toFinset, hcr_iff]
    simp only [PearlCSem.passRows, Finset.mem_filter, hrows, List.mem_toFinset]
  have hPm : ∀ i ∈ cr.toFinset, i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m := fun i hi =>
    hrowsm i ((hcr_iff i).1 (List.mem_toFinset.1 hi)).1
  have hCn : ∀ j ∈ cols.toFinset, j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n := fun j hj =>
    hcolsn j (List.mem_toFinset.1 hj)
  obtain ⟨i0, hi0⟩ := List.exists_mem_of_ne_nil _ hcrne
  obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil _ hccne
  have hm0 : (0 : ℚ) < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m :=
    Nat.cast_pos.2 (Nat.lt_of_le_of_lt (Nat.zero_le _) (hrowsm i0 ((hcr_iff i0).1 hi0).1))
  have hn0 : (0 : ℚ) < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n :=
    Nat.cast_pos.2 (Nat.lt_of_le_of_lt (Nat.zero_le _) (hcolsn j0 hj0))
  have hcred := Cap.credit_le (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) hKk hk128 hk0
  rw [← hP, hcols, Cap.tileChainDebit_cells (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act
    _ _ hPm hCn, tileShare, List.toFinset_card_of_nodup hcrnd, List.toFinset_card_of_nodup hcolsnd]
  exact Cap.bound_glue_forming hm0 hn0 (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by rw [show devH100.prices.costs.fs = 64 from rfl]; norm_num) (by exact_mod_cast hcap) hcred

/-- Today's check meets the forming rule at no forming element, on a passed tile that carries a ticket. -/
theorem meets_zero (hT : Properties.Pouw.PearlPlus.TileAcceptMeans) (hK : Properties.Pouw.PearlPlus.TileKeepsTheCap)
    {p : Pass} {tid : ℕ} {o : TileOpening} {U : Pouw.PearlC.Workload} {g : ℕ} (hof : TileOf p tid U g)
    (hticket : CarriesATicket p tid) (hpass : checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass) :
    MeetsTheFormingRule p tid o 0 := by
  intro pos rows cols ops ch htile hform
  obtain ⟨pos₀, rows₀, cols₀, htile₀, -, -, -, -, -, -, hrows, hcols⟩ := hof
  have h0 := htile.symm.trans htile₀
  simp only [Option.some.injEq, Prod.mk.injEq] at h0
  obtain ⟨rfl, rfl, rfl⟩ := h0
  obtain ⟨pos', rows', cols', ops', ch', htile', hrest⟩ := hT p _ tid o none hpass
  simp only [htile, Option.some.injEq, Prod.mk.injEq] at htile'
  obtain ⟨rfl, rfl, rfl⟩ := htile'
  try dsimp only at hrest
  obtain ⟨-, -, -, -, -, -, hform', -, hrowsF, hcapC, -⟩ := hrest
  have h1 := hform.symm.trans hform'
  simp only [Except.ok.injEq, Prod.mk.injEq] at h1
  obtain ⟨rfl, rfl⟩ := h1
  obtain ⟨hbA, hbB⟩ := hticket pos rows cols htile
  have hbandR : ∀ i ∈ rows, ∀ i' ∈ rows, i / 64 = i' / 64 := fun i hi i' hi' =>
    Cap.rows_band (L := U.layout) (g := g) (by rw [hrows]; exact List.mem_toFinset.2 hi)
      (by rw [hrows]; exact List.mem_toFinset.2 hi')
  have hbandC : ∀ j ∈ cols, ∀ j' ∈ cols, j / 64 = j' / 64 := fun j hj j' hj' =>
    Cap.cols_band (L := U.layout) (g := g) (by rw [hcols]; exact List.mem_toFinset.2 hj)
      (by rw [hcols]; exact List.mem_toFinset.2 hj')
  have hbe := Cap.band_eq hbandR hbandC hbA hbB
  have hK' := hK p.scheme p.records[pos]! rows cols ops (Cap.checkOpened_none hrowsF hcapC)
  have hne1 : (p.records[pos]!.credited rows cols).1 ≠ [] := by rw [← hbe]; exact hbA
  have hne2 : (p.records[pos]!.credited rows cols).2 ≠ [] := by rw [← hbe]; exact hbB
  unfold formingCapOK
  rw [hbe]
  revert hK' hne1 hne2
  generalize p.records[pos]!.credited rows cols = q
  obtain ⟨cr, cc⟩ := q
  try dsimp only
  intro hK' hne1 hne2
  obtain ⟨words, hw, hcap, -⟩ := hK' hne1 hne2
  exact ⟨words, hw, Eq.trans_le (by ring) hcap⟩

/-- **(c)'s chain part** from `TileAcceptMeans`, `LoadedGivesUp`, the check's cap (`TileKeepsTheCap`) and
`RowsAreTheRules`, with the replay (`debitIsTheReplay`, `replayReadsTheCodes`): `tile_bound` at no forming element. -/
theorem tilePassGivesTheChainCap (hT : Properties.Pouw.PearlPlus.TileAcceptMeans)
    (hL : Properties.Pouw.PearlPlus.LoadedGivesUp) (hK : Properties.Pouw.PearlPlus.TileKeepsTheCap)
    (hR : RowsAreTheRules) : TilePassGivesTheChainCap := by
  intro Q R S saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket hpass
  have h := tile_bound hT hL hR saltOf rootOf H s coins man p tid o U g c 0 hload hof hsalt hroot hwOK hopen
    hticket (meets_zero hT hK hof hticket hpass) hpass
  simpa only [Nat.cast_zero, mul_zero, add_zero] using h

/-- **(c)** from `TileAcceptMeans`, `LoadedGivesUp` and `RowsAreTheRules`, with the replay (`debitIsTheReplay`,
`replayReadsTheCodes`): `tile_bound` at the tile's forming elements, whose forming term is the rest of
`tileDebitRev1`. -/
theorem tilePassGivesTheCap (hT : Properties.Pouw.PearlPlus.TileAcceptMeans)
    (hL : Properties.Pouw.PearlPlus.LoadedGivesUp) (hR : RowsAreTheRules) : TilePassGivesTheCap := by
  intro Q R S saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket hforming hpass
  show (semH3 saltOf rootOf : PearlCSem Q R S).tileDebitRev1 devH100 H s U ((auditTiling 64 64 U.layout).unit g)
      ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
        ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act ≤
    1 / 400 * creditDevRev1 devH100 (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
      tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g)
  rw [tileDebitRev1_split]
  exact tile_bound hT hL hR saltOf rootOf H s coins man p tid o U g c _ hload hof hsalt hroot hwOK hopen hticket
    hforming hpass

end Pouw.PearlPlus.Bridge
