module

public import Proofs.Pouw.PearlPlus.Draw
public import Proofs.Pouw.PearlPlus.Debit
public import Proofs.Pouw.PearlC.Cap.SaltDead
public import Proofs.Pouw.PearlC.Records

/-!
# Pearl+'s bridge, part (c): a passed tile is within the game's inert-work bound

The bound the game reads is `TileGood`'s first conjunct, `capOK` of `pearlCTilesDevRev1 devH100` at the deployed
semantics and `ρ = 1/400`: the tile's replayed inert work (`tileDebitRev1`: its flagged atom-words, its later promotions
into a +0 total, its forming and its words' mends) on its rows that pass the rules and all its columns is at most
`ρ·creditDevRev1` times their share. Acceptance doesn't read it: `Draw` puts a sampled tile with good words in
`goodDeclTiles` within the bound and in `overCapTiles` over it. The credit does: the tolerant audit's deduction grows
with the samples on `overCapTiles`, and a passed tile within the bound keeps those to the samples the check finds over
its own bound.

The check's bound (`Rule.capCheck`, as `TileKeepsTheCap` states it) is on the record's credited rows and columns under
the band rule (`UnitRecord.credited`: a real row neither excluded nor in a 64-row band with a voluntary A row, and a
column whose 64-column block holds no excluded or voluntary B row). It counts `n` times each credited word's replayed
inert work and mends (`Rule.wordMend`) and `64·|cc|` per forming element of a credited row (`Rule.rowElements`). A tile
that keeps a row and a column carries a ticket (`CarriesATicket`). Such a tile has no voluntary row and no excluded or
voluntary column, its filler rows are zero and fail the rules, and its excluded rows fail them, so the check's credited
rows are the tile's `passRows` and its credited columns all its columns.

* `tilePassGivesTheCap`: a passed tile that carries a ticket is within the bound. The check's inert work per word is
  the game's chain (`wordFlagsIsTheReplay`, `codes_eq`) on rows that pass the rules exactly when the game's do
  (`RowsAreTheRules`).
  The check's element count covers the game's forming elements (`formingElems_le`, `row_forming`): an element the game
  finds salt-dead the check finds salt-dead (`saltDeadP_iff_ends_hopper` and the check's float arithmetic,
  `saltDead_of_ends`), and an element whose block every column's word flags the check counts by those flags. The
  check's mends cover the game's (`MendSim.wordMend_le`, `MendSim.row_ends`): its end codes are the game's `endRow`
  (`endCodes_eq`), and its cut flags on them the game's `chainCuts` (`MendSim.wordCuts_eq`). The forming and the mends
  are the rest of the game's inert work (`tileDebitRev1_split`), and the check's credit per credited row,
  `creditOf(1, kKept, n)`, is at most `creditDevRev1`'s (`credit_le`), with equality at `kKept = k`.
* `tilePassGivesTheChainCap`: the chain's part of the game's inert work (`tileChainDebit`) is within the bound.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset

/-- Tile `tid` of pass `p` carries a ticket under the band rule: it keeps a credited row and a credited column. -/
def CarriesATicket (p : Pass) (tid : ℕ) : Prop :=
  ∀ pos rows cols, p.tile? tid = some (pos, rows, cols) →
    (p.records[pos]!.credited rows cols).1 ≠ [] ∧ (p.records[pos]!.credited rows cols).2 ≠ []

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

open Classical in
/-- The mend part of `tileDebitRev1`: its words' `wordMend` at one MAC each. -/
noncomputable def tileMendDebit {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (U : Pouw.PearlC.Workload)
    (u : ℕ) (rows cols : Finset ℕ) (act : Codes) : ℚ :=
  let sh := U.layout.shape u
  ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
    (wordMend d sem.alpha sem.beta (wordsVal sh.m sh.k act e.1)
      (wordsVal sh.n sh.k (U.fixed (U.layout.fixedId u)) e.2) : ℚ)

/-- A tile's replayed inert work is its chain's, its forming's and its mends'. -/
theorem tileDebitRev1_split {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) :
    sem.tileDebitRev1 d H s U u rows cols act =
      tileChainDebit sem d H s U u rows cols act + tileFormingDebit sem d H s U u rows cols act +
        tileMendDebit sem d U u rows cols act := by
  unfold PearlCSem.tileDebitRev1 PearlCSem.tileDebitDev tileChainDebit tileFormingDebit formingElems tileMendDebit
  exact congrArg (· + _) (add_right_comm _ _ _)

/-- **(c) A passed tile's chain is within the inert-work bound**: a tile as in `TilePassGivesTheCap` has the chain's
part of its replayed inert work (`tileChainDebit`) within the game's bound at `ρ = 1/400`. -/
def TilePassGivesTheChainCap : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCheck.PearlPlus.EpochCoins) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).fixedOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c → CarriesATicket p tid →
    checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass →
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      let P := (semH3 saltOf rootOf : PearlCSem Q R S).passRows sh c.act (tl.rows g)
      tileChainDebit (semH3 saltOf rootOf) devH100 H s U (tl.unit g) P (tl.cols g) c.act ≤
        1 / 400 * creditDevRev1 devH100 sh * tileShare sh P (tl.cols g)

/-- **(c) A passed tile is within the inert-work bound**: a tile of a pass `load` gives, which is audit tile `g` of `U`,
under the semantics' salt and fixed operand's root, whose fixed-operand rows all pass the rules (`TileGood`'s second conjunct, as (d)
takes it), whose openings are its unit's commitment `c`, that carries a ticket under the band rule, and that
`checkTile` passes on its fixed operand's noise, has its replayed inert work within the game's bound at `ρ = 1/400`. -/
def TilePassGivesTheCap : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCheck.PearlPlus.EpochCoins) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).fixedOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c → CarriesATicket p tid →
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

/-- A tile in one 64-row band whose credited rows aren't empty has no voluntary A row in its band. -/
theorem band_anyA {r : UnitRecord} {rows cols : List ℕ} (hband : ∀ i ∈ rows, ∀ i' ∈ rows, i / 64 = i' / 64)
    (h : (r.credited rows cols).1 ≠ []) (i : ℕ) (hi : i ∈ rows) :
    r.voluntaryA.any (· / UnitRecord.BAND == i / UnitRecord.BAND) = false := by
  obtain ⟨i0, hi0⟩ := List.exists_mem_of_ne_nil _ h
  simp only [UnitRecord.credited, List.mem_filter] at hi0
  obtain ⟨hi0r, hc⟩ := hi0
  have he : i / UnitRecord.BAND = i0 / UnitRecord.BAND := hband i hi i0 hi0r
  rw [he]
  cases hv : r.voluntaryA.any (· / UnitRecord.BAND == i0 / UnitRecord.BAND)
  · rfl
  · rw [hv] at hc
    simp at hc

/-- A tile in one 64-column block whose credited columns aren't empty credits all its columns. -/
theorem credited_cols {r : UnitRecord} {rows cols : List ℕ} (hband : ∀ j ∈ cols, ∀ j' ∈ cols, j / 64 = j' / 64)
    (h : (r.credited rows cols).2 ≠ []) : (r.credited rows cols).2 = cols := by
  obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil _ h
  simp only [UnitRecord.credited, List.mem_filter] at hj0 ⊢
  obtain ⟨hj0c, hj0n⟩ := hj0
  refine List.filter_eq_self.2 fun j hj => ?_
  try dsimp only
  have he : j / UnitRecord.BAND = j0 / UnitRecord.BAND := hband j hj j0 hj0c
  rw [he]
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

/-! ## A row's statistics -/

/-- Each row `formV1` forms keeps its words, and its ρ, α and β are `s5_stats`' of them. -/
theorem formV1_stats {rows eRows f : Array (Array ℕ)} {out : Words.Formed} (h : Words.formV1 rows eRows f = .ok out) :
    ∀ j < (rows.zip eRows).size, out.x[j]! = (rows.zip eRows)[j]!.1 ∧
      ∃ s, Words.s5Stats out.x[j]! = .ok (s, out.rho[j]!, out.alpha[j]!, out.beta[j]!) := by
  unfold Words.formV1 at h
  obtain ⟨r, hr, h⟩ := Assembly.bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at hr
  have := Draw.forIn_inv_idx (fun n (o : Words.Formed) => o.x.size = n ∧ o.rho.size = n ∧ o.alpha.size = n ∧
      o.beta.size = n ∧ ∀ j < n, o.x[j]! = (rows.zip eRows).toList[j]!.1 ∧
        ∃ s, Words.s5Stats o.x[j]! = .ok (s, o.rho[j]!, o.alpha[j]!, o.beta[j]!)) _ _ 0 _ _
    ⟨rfl, rfl, rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩ ?_ hr
  · intro j hj
    obtain ⟨-, -, -, -, hall⟩ := this
    have hj' : j < (rows.zip eRows).toList.length := by simpa using hj
    obtain ⟨h1, h2⟩ := hall j (by omega)
    refine ⟨?_, h2⟩
    rw [h1, getElem!_pos _ j hj', getElem!_pos _ j hj]
    simp
  · intro n a s r ha hs hr
    obtain ⟨x, e⟩ := a
    obtain ⟨st, hst, hr⟩ := Assembly.bind_ok hr
    obtain ⟨fst, rho, alpha, beta⟩ := st
    obtain ⟨ep, -, hr⟩ := Assembly.bind_ok hr
    obtain ⟨codes, -, hr⟩ := Assembly.bind_ok hr
    simp only [pure, Except.pure, Except.ok.injEq] at hr
    subst hr
    obtain ⟨hx, hrho, halpha, hbeta, hall⟩ := hs
    refine ⟨_, rfl, by simp [hx], by simp [hrho], by simp [halpha], by simp [hbeta], fun j hj => ?_⟩
    simp only [Nat.zero_add] at hx hrho halpha hbeta hall hj ha ⊢
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | hj
    · rw [Draw.push_get!_lt _ _ (by omega), Draw.push_get!_lt _ _ (by omega), Draw.push_get!_lt _ _ (by omega),
        Draw.push_get!_lt _ _ (by omega)]
      exact hall j hj
    · have g1 : (s.x.push x)[j]! = x := by rw [hj, ← hx]; exact Draw.push_get!_eq _ _
      have g2 : (s.rho.push rho)[j]! = rho := by rw [hj, ← hrho]; exact Draw.push_get!_eq _ _
      have g3 : (s.alpha.push alpha)[j]! = alpha := by rw [hj, ← halpha]; exact Draw.push_get!_eq _ _
      have g4 : (s.beta.push beta)[j]! = beta := by rw [hj, ← hbeta]; exact Draw.push_get!_eq _ _
      rw [g1, g2, g3, g4]
      refine ⟨?_, fst, hst⟩
      rw [List.getElem!_eq_getElem?_getD, hj, ha]
      rfl

/-- The A rows `encode` forms are the parsed openings, and their ρ, α and β are `s5_stats`' of them. -/
theorem encode_stats {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k : ℕ} {rows : List ℕ}
    {aRows : List ByteArray} {cols : List ℕ} {bRows : List ByteArray} {ops : Words.Ops}
    (ha : aRows.length = rows.length) (h : Words.encode salt rootB wn index k rows aRows cols bRows = .ok ops) :
    ∀ x < rows.length, Words.parseRow aRows[x]! k = .ok ops.a.x[x]! ∧
      ∃ s, Words.s5Stats ops.a.x[x]! = .ok (s, ops.a.rho[x]!, ops.a.alpha[x]!, ops.a.beta[x]!) := by
  unfold Words.encode at h
  obtain ⟨ea, hea, h⟩ := Assembly.bind_ok h
  obtain ⟨xa, hxa, h⟩ := Assembly.bind_ok h
  obtain ⟨a, hfa, h⟩ := Assembly.bind_ok h
  obtain ⟨eb, -, h⟩ := Assembly.bind_ok h
  obtain ⟨xb, -, h⟩ := Assembly.bind_ok h
  obtain ⟨b, -, h⟩ := Assembly.bind_ok h
  obtain ⟨pa, -, h⟩ := Assembly.bind_ok h
  obtain ⟨pb, -, h⟩ := Assembly.bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  intro x hx
  have sea := (Assembly.array_mapM_ok _ _ _ hea).1
  obtain ⟨sxa, gxa⟩ := Assembly.array_mapM_ok _ _ _ hxa
  simp only [List.size_toArray, List.length_zip] at sea sxa
  have hz : x < (xa.zip ea).size := by simp only [Array.size_zip]; omega
  obtain ⟨h1, h2⟩ := formV1_stats hfa x hz
  refine ⟨?_, h2⟩
  rw [h1, getElem!_pos (xa.zip ea) x hz, Array.getElem_zip]
  have := gxa x (by simp; omega)
  rw [getElem!_pos _ x (by simp; omega), List.getElem_toArray, getElem!_pos xa x (by omega)] at this
  rw [getElem!_pos aRows x (by omega)]
  exact this

/-- The B rows `encode` forms are the parsed openings, and their ρ, α and β are `s5_stats`' of them. -/
theorem encode_stats_b {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k : ℕ} {rows : List ℕ}
    {aRows : List ByteArray} {cols : List ℕ} {bRows : List ByteArray} {ops : Words.Ops}
    (hb : bRows.length = cols.length) (h : Words.encode salt rootB wn index k rows aRows cols bRows = .ok ops) :
    ∀ z < cols.length, Words.parseRow bRows[z]! k = .ok ops.b.x[z]! ∧
      ∃ s, Words.s5Stats ops.b.x[z]! = .ok (s, ops.b.rho[z]!, ops.b.alpha[z]!, ops.b.beta[z]!) := by
  unfold Words.encode at h
  obtain ⟨ea, -, h⟩ := Assembly.bind_ok h
  obtain ⟨xa, -, h⟩ := Assembly.bind_ok h
  obtain ⟨a, -, h⟩ := Assembly.bind_ok h
  obtain ⟨eb, heb, h⟩ := Assembly.bind_ok h
  obtain ⟨xb, hxb, h⟩ := Assembly.bind_ok h
  obtain ⟨b, hfb, h⟩ := Assembly.bind_ok h
  obtain ⟨pa, -, h⟩ := Assembly.bind_ok h
  obtain ⟨pb, -, h⟩ := Assembly.bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  intro z hz
  have seb := (Assembly.array_mapM_ok _ _ _ heb).1
  obtain ⟨sxb, gxb⟩ := Assembly.array_mapM_ok _ _ _ hxb
  simp only [List.size_toArray] at seb sxb
  have hz' : z < (xb.zip eb).size := by simp only [Array.size_zip]; omega
  obtain ⟨h1, h2⟩ := formV1_stats hfb z hz'
  refine ⟨?_, h2⟩
  rw [h1, getElem!_pos (xb.zip eb) z hz', Array.getElem_zip]
  have := gxb z (by simp; omega)
  rw [getElem!_pos _ z (by simp; omega), List.getElem_toArray, getElem!_pos xb z (by omega)] at this
  rw [getElem!_pos bRows z (by omega)]
  exact this

/-! ## The check's replayed inert work, summed -/

theorem mapM_ok_zip {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → ∀ q ∈ l.zip l', f q.1 = .ok q.2
  | [], l', _, q, hq => by simp at hq
  | a :: l, l', h, q, hq => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := Assembly.bind_ok h
    obtain ⟨bs, hbs, h⟩ := Assembly.bind_ok h
    cases h
    rw [List.zip_cons_cons] at hq
    rcases List.mem_cons.1 hq with rfl | hq
    · exact h0
    · exact mapM_ok_zip hbs q hq

theorem mapM_sum_eq {ε α β : Type} {f : α → Except ε β} {g : α → ℕ} {g' : β → ℕ} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → (∀ a ∈ l, ∀ b, f a = .ok b → g' b = g a) →
      (l'.map g').sum = (l.map g).sum
  | [], l', h, _ => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | a :: l, l', h, hg => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := Assembly.bind_ok h
    obtain ⟨bs, hbs, h⟩ := Assembly.bind_ok h
    cases h
    simp only [List.map_cons, List.sum_cons]
    rw [hg a List.mem_cons_self b0 h0, mapM_sum_eq hbs fun a' ha' b hb => hg a' (List.mem_cons_of_mem _ ha') b hb]

theorem mapM_sum_le {ε α : Type} {f : α → Except ε ℕ} {g : α → ℕ} :
    ∀ {l : List α} {l' : List ℕ}, l.mapM f = .ok l' → (∀ a ∈ l, ∀ b, f a = .ok b → g a ≤ b) →
      (l.map g).sum ≤ l'.sum
  | [], l', h, _ => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | a :: l, l', h, hg => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := Assembly.bind_ok h
    obtain ⟨bs, hbs, h⟩ := Assembly.bind_ok h
    cases h
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (hg a List.mem_cons_self b0 h0)
      (mapM_sum_le hbs fun a' ha' b hb => hg a' (List.mem_cons_of_mem _ ha') b hb)

theorem map_zip_fst {α β : Type} (g : α → ℕ) :
    ∀ (l : List α) (l' : List β), l.length = l'.length → (l.zip l').map (fun q => g q.1) = l.map g
  | [], _, _ => by simp
  | _ :: _, [], h => by simp at h
  | a :: l, b :: l', h => by
    simp only [List.zip_cons_cons, List.map_cons, map_zip_fst g l l' (by simpa using h)]

theorem foldl_add_eq : ∀ (L : List ℕ) (a : ℕ), L.foldl (· + ·) a = a + L.sum
  | [], a => by simp
  | b :: L, a => by rw [List.foldl_cons, foldl_add_eq L, List.sum_cons, Nat.add_assoc]

theorem foldl_map_eq {α : Type} (f : α → ℕ) : ∀ (L : List α) (a : ℕ), L.foldl (fun t w => t + f w) a = a + (L.map f).sum
  | [], a => by simp
  | b :: L, a => by rw [List.foldl_cons, foldl_map_eq f L, List.map_cons, List.sum_cons, Nat.add_assoc]

theorem debitUnits_eq (n c a z e : ℕ) : Rule.debitUnits n c a z e = n * (40 * a + 32 * z) + 64 * c * e := by
  show n * ((32 + 32 / 4) * a + 32 * z) + 64 * c * e = _
  norm_num

/-- The check's inert work on a tile, summed as `TileKeepsTheCap` sums it, is `n` times its words' `40·F + 32·Z` and
`64·|cc|` per element. -/
theorem rowDebit_sum (n c : ℕ) (ws : List (List (Array Bool × ℕ))) (es : List ℕ) (h : ws.length = es.length) :
    ((ws.zip es).map fun (we : List (Array Bool × ℕ) × ℕ) => Rule.debitUnits n c
      (we.1.foldl (fun t w => t + (w.1.filter id).size) 0) (we.1.foldl (fun t w => t + w.2) 0) we.2).foldl
        (· + ·) 0 =
      n * (ws.map fun w => 40 * (w.map fun v => (v.1.filter id).size).sum + 32 * (w.map fun v => v.2).sum).sum +
        64 * c * es.sum := by
  rw [foldl_add_eq, Nat.zero_add]
  induction ws generalizing es with
  | nil =>
    cases es with
    | nil => simp
    | cons _ _ => simp at h
  | cons w ws ih =>
    cases es with
    | nil => simp at h
    | cons e es =>
      simp only [List.zip_cons_cons, List.map_cons, List.sum_cons]
      rw [ih es (by simpa using h), debitUnits_eq, foldl_map_eq (fun v : Array Bool × ℕ => (v.1.filter id).size),
        foldl_map_eq (fun v : Array Bool × ℕ => v.2)]
      ring

/-- The check's inert work with its mends, as `TileKeepsTheCap` sums it: `rowDebit_sum` and `n` times every credited
word's `wordMend`. -/
theorem rowDebit_sum_mend (n c : ℕ) (ws : List (List (Array Bool × ℕ))) (es : List ℕ)
    (eA eB : List (Array ℕ × Array ℕ × Array Bool)) (h : ws.length = es.length) (h' : es.length = eA.length) :
    (((ws.zip es).zip eA).map fun (we : (List (Array Bool × ℕ) × ℕ) × (Array ℕ × Array ℕ × Array Bool)) =>
      Rule.debitUnits n c (we.1.1.foldl (fun t w => t + (w.1.filter id).size) 0)
        (we.1.1.foldl (fun t w => t + w.2) 0) we.1.2 + n * eB.foldl (fun t eb => t + Rule.wordMend we.2 eb) 0).foldl
        (· + ·) 0 =
      n * (ws.map fun w => 40 * (w.map fun v => (v.1.filter id).size).sum + 32 * (w.map fun v => v.2).sum).sum +
        64 * c * es.sum + n * (eA.map fun ea => (eB.map (Rule.wordMend ea)).sum).sum := by
  rw [foldl_add_eq, Nat.zero_add]
  induction ws generalizing es eA with
  | nil =>
    cases es with
    | nil =>
      cases eA with
      | nil => simp
      | cons _ _ => simp at h'
    | cons _ _ => simp at h
  | cons w ws ih =>
    cases es with
    | nil => simp at h
    | cons e es =>
      cases eA with
      | nil => simp at h'
      | cons ea eA =>
        simp only [List.zip_cons_cons, List.map_cons, List.sum_cons]
        rw [ih es eA (by simpa using h) (by simpa using h'), debitUnits_eq,
          foldl_map_eq (fun v : Array Bool × ℕ => (v.1.filter id).size), foldl_map_eq (fun v : Array Bool × ℕ => v.2),
          foldl_map_eq (Rule.wordMend ea)]
        ring

theorem sum_le_zip {α β : Type} (g : α → ℕ) (h : β → ℕ) :
    ∀ (L : List α) (L' : List β), L.length = L'.length → (∀ q ∈ L.zip L', g q.1 ≤ h q.2) →
      (L.map g).sum ≤ (L'.map h).sum
  | [], [], _, _ => le_rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: L, b :: L', hl, hq => by
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (hq (a, b) (by simp))
      (sum_le_zip g h L L' (by simpa using hl) fun q hq' => hq q (by simp [hq']))

/-- Over distinct rows and columns, per-row sums of `40·F + 32·Z` are its sum over the cells. -/
theorem cells_sum (cr cc : List ℕ) (hcr : cr.Nodup) (hcc : cc.Nodup) (F Z : ℕ → ℕ → ℕ) :
    (cr.map fun i => 40 * (cc.map (F i)).sum + 32 * (cc.map (Z i)).sum).sum =
      ∑ i ∈ cr.toFinset, ∑ j ∈ cc.toFinset, (40 * F i j + 32 * Z i j) := by
  rw [← List.sum_toFinset _ hcr]
  refine Finset.sum_congr rfl fun i _ => ?_
  try dsimp only
  rw [← List.sum_toFinset _ hcc, ← List.sum_toFinset _ hcc, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  try rfl

/-! ## The game's chain inert work, cell by cell -/

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

/-- At the H100 record a tile's chain inert work is `40` per flagged atom and `32` per later +0 promotion, summed over
its cells. -/
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

open Classical in
/-- Cell `(i, j)`'s mends at the H100 record, `0` off the unit. -/
noncomputable def cellM {Q R S : Type} (sem : PearlCSem Q R S) (U : Pouw.PearlC.Workload) (u : ℕ) (act : Codes)
    (i j : ℕ) : ℕ :=
  if h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n then
    wordMend devH100 sem.alpha sem.beta (wordsVal (U.layout.shape u).m (U.layout.shape u).k act ⟨i, h.1⟩)
      (wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.fixed (U.layout.fixedId u)) ⟨j, h.2⟩)
  else 0

/-- A tile's mends are its cells'. -/
theorem tileMendDebit_cells {Q R S : Type} (sem : PearlCSem Q R S) (U : Pouw.PearlC.Workload) (u : ℕ) (act : Codes)
    (P C : Finset ℕ) (hP : ∀ i ∈ P, i < (U.layout.shape u).m) (hC : ∀ j ∈ C, j < (U.layout.shape u).n) :
    Bridge.tileMendDebit sem devH100 U u P C act = ((∑ i ∈ P, ∑ j ∈ C, cellM sem U u act i j : ℕ) : ℚ) := by
  simp only [Bridge.tileMendDebit]
  rw [sum_fin_filter P C hP hC]
  push_cast
  refine Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => ?_
  have h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n := ⟨hP i hi, hC j hj⟩
  simp only [cellM, dif_pos h]

/-! ## The game's forming elements, row by row -/

open Classical in
/-- Row `i`'s forming elements in the game: its elements that are salt-dead or whose atom block every column of `C`
flags. -/
noncomputable def rowGame {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) {m n k : ℕ}
    (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (act : Codes) (C : Finset ℕ) (i : ℕ) : ℕ :=
  ((range k).filter fun l => ∃ (hi : i < m) (hl : l < k),
    d.saltDead sem.alpha sem.beta k (wordsVal m k act ⟨i, hi⟩) ⟨l, hl⟩ ∨
      ∀ j : Fin n, (j : ℕ) ∈ C → d.flags A B ⟨i, hi⟩ j (l / 32) = true).card

/-- A tile's forming elements are its rows'. -/
theorem formingElems_le {Q R S : Type} (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (P C : Finset ℕ) (act : Codes) :
    Bridge.formingElems sem d H s U u P C act ≤
      ∑ i ∈ P, rowGame sem d (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 act C i := by
  classical
  unfold Bridge.formingElems
  dsimp only
  refine (Finset.card_eq_sum_card_fiberwise
    (f := fun e : Fin (U.layout.shape u).m × Fin (U.layout.shape u).k => (e.1 : ℕ)) (t := P) ?_).trans_le ?_
  · intro e he
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at he ⊢
    exact he.1
  · refine Finset.sum_le_sum fun i _ => ?_
    unfold rowGame
    refine Finset.card_le_card_of_injOn (fun e => (e.2 : ℕ)) ?_ ?_
    · intro e he
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at he ⊢
      obtain ⟨⟨-, hc⟩, rfl⟩ := he
      exact ⟨Finset.mem_range.2 e.2.isLt, e.1.isLt, e.2.isLt, hc⟩
    · intro e he e' he' h
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at he he'
      exact Prod.ext (Fin.ext (he.2.trans he'.2.symm)) (Fin.ext h)

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

theorem bound_glue_forming {D f E E' c Cr a b m n : ℚ} (hm : 0 < m) (hn : 0 < n) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hf : f = 64) (hE : E' ≤ E) (hD : 400 * (n * D + 64 * b * E) ≤ c * a * b) (hc : m * c ≤ Cr) :
    D + f * (b / n) * E' ≤ 1 / 400 * Cr * (a * b / (m * n)) := by
  subst hf
  have h0 : 0 ≤ 64 * (b / n) := mul_nonneg (by norm_num) (div_nonneg hb hn.le)
  have h1 : D + 64 * (b / n) * E' ≤ D + 64 * (b / n) * E := by linarith [mul_le_mul_of_nonneg_left hE h0]
  refine h1.trans ?_
  have hn1 : n * n⁻¹ = 1 := mul_inv_cancel₀ hn.ne'
  have h : (D + 64 * (b / n) * E) * (400 * n) = 400 * (n * D + 64 * b * E) :=
    calc (D + 64 * (b / n) * E) * (400 * n) = 400 * (n * D) + 400 * (64 * b * E) * (n * n⁻¹) := by ring
      _ = 400 * (n * D + 64 * b * E) := by rw [hn1]; ring
  exact bound_glue hm hn ha hb (by rw [h]; exact hD) hc

theorem bound_glue_mend {D f E E' M M' c Cr a b m n : ℚ} (hm : 0 < m) (hn : 0 < n) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hf : f = 64) (hE : E' ≤ E) (hM : M' ≤ M) (hD : 400 * (n * D + 64 * b * E + n * M) ≤ c * a * b)
    (hc : m * c ≤ Cr) : D + f * (b / n) * E' + M' ≤ 1 / 400 * Cr * (a * b / (m * n)) := by
  have h : 400 * (n * (D + M) + 64 * b * E) = 400 * (n * D + 64 * b * E + n * M) := by ring
  have h' := bound_glue_forming (D := D + M) hm hn ha hb hf hE (by rw [h]; exact hD) hc
  linarith

end Pouw.PearlPlus.Cap

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus Finset

namespace FormingSim

/-! ## The check's forming count covers the game's -/

theorem step_acc (p : Pouw.Fp8Atom.Pipeline) {a a' : ℕ}
    (h : Pouw.Fp8Atom.fp32Value a = Pouw.Fp8Atom.fp32Value a') (l l' : List Code) : p.step a l l' = p.step a' l l' := by
  unfold Pouw.Fp8Atom.Pipeline.step
  rw [h]

/-- The forming's accumulator, `f32(α·x)` in the check's arithmetic, is the game's `f32Word (α·x)` as the atom reads
it: the two differ at most in a zero's sign. -/
theorem mul_fp32Value {αw w : ℕ} {y : ℚ} (hα : AccumProofs.StepWord αw) (hw : w < 2 ^ 32) (hy : wordQ w = some y) :
    Pouw.Fp8Atom.fp32Value (Verity.Fp32.f32MulV2 αw w) =
      Pouw.Fp8Atom.fp32Value (f32Word (PearlC.wordVal αw * y)) := by
  have hops := CoreOps.pc8OpsAgree_coreOps
  have hαq := WholeCircuit.stepWord_wordQ hα
  have hlt : Verity.Fp32.f32MulV2 αw w % 2 ^ 32 = Verity.Fp32.f32MulV2 αw w :=
    Nat.mod_eq_of_lt (PeelSim.mulV2_lt _ _)
  by_cases hs : PearlC.wordVal αw * y = 0 → PeelExact.signBit αw = PeelExact.signBit w
  · have h : Verity.Fp32.f32MulV2 αw w % 2 ^ 32 = f32Word (PearlC.wordVal αw * y) :=
      hops.mul αw w _ y hα.1 hw hαq hy hs
    rw [← h, hlt]
  · obtain ⟨h0, hne⟩ := not_imp.1 hs
    have h : Verity.Fp32.f32MulV2 αw w % 2 ^ 32 = 0x80000000 := hops.mulZero αw w _ y hα.1 hw hαq hy h0 hne
    rw [hlt] at h
    rw [h, h0, show f32Word 0 = 0 by simp [f32Word]]
    rfl

/-- A step the check takes and casts, where it is the game's step, is the game's code. -/
theorem step_code {acc : ℕ} {a b : List ℕ} {D c : ℕ} (P : ℕ) (hP : Words.atom acc a b = P)
    (hPstep : ∃ acc' l l', P = hopperE4m3K32.step acc' l l') (hD : Rule.step acc a b = .ok D)
    (hc : Words.toFp8 D = .ok c) :
    c = (e4m3Word P).val ∧ e4m3 (PearlC.wordVal P) = e4m3Word P ∧ (e4m3Word P).isNaN = false := by
  obtain ⟨rfl, hf⟩ := DebitSim.step_ok hD
  obtain ⟨-, rfl⟩ := WordsSim.toFp8_ok hc
  rw [hP] at hf ⊢
  obtain ⟨acc', l, l', rfl⟩ := hPstep
  have hS : AccumProofs.StepWord (hopperE4m3K32.step acc' l l') :=
    ⟨AccumProofs.step_lt _ _ _ _, AccumProofs.step_ne_negZero _ _ _ _, WordsSim.finite?_of hf⟩
  refine ⟨?_, RowCircuit.e4m3_wordVal (fun h => absurd h hf) (fun h0 => by
    rw [AccumProofs.wordQ_zero_word hS h0]; rfl), ?_⟩
  · rw [Properties.Evaluator.Casts.f32ToE4m3Sat]
    exact CoreOps.pc8OpsAgree_coreOps.cast _
  · rw [← RowCircuit.f32Word_wordVal hS]
    exact WholeCircuit.e4m3Word_f32Word_notNaN _

theorem endsAgree_of {c d : Code} (h : c = d ∨ (c.val % 128 = 0 ∧ d.val % 128 = 0)) :
    Rule.endsAgree c.val d.val = true := by
  rcases h with rfl | ⟨hc, hd⟩
  · simp [Rule.endsAgree]
  · have hc' := c.isLt
    have hd' := d.isLt
    have h1 : c.val = 0 ∨ c.val = 128 := by omega
    have h2 : d.val = 0 ∨ d.val = 128 := by omega
    rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2 <;> rw [h1, h2] <;> decide

/-- The game's step at one end of the support (`minus` clear: `F` at `+11/4`; set: at `−11/4`) for an element `y` of a
row at scales `αw` and `βw`. -/
def endStep (αw βw : ℕ) (y : ℚ) (minus : Bool) : ℕ :=
  hopperE4m3K32.step (f32Word (PearlC.wordVal αw * y))
    (List.ofFn (lineCodes (PearlC.wordVal βw) (Pouw.PearlC.SaltDead.flatLine fun _ => false)))
    (List.ofFn (Pouw.PearlC.SaltDead.flatLineF fun _ => minus))

/-- The check's code `c` is the cast of the step word `P`, a finite code. -/
def CastOf (c P : ℕ) : Prop :=
  c = (e4m3Word P).val ∧ e4m3 (PearlC.wordVal P) = e4m3Word P ∧ (e4m3Word P).isNaN = false

/-- **The check's end codes are the game's**: on a row whose α and β are step words, β positive, where `flat_code` and
`end_codes` are values, the check's codes at the support's two ends are the casts of the game's steps there. -/
theorem endCodes_eq {αw βw cb w : ℕ} {c : ℕ × ℕ} {y : ℚ} (hα : AccumProofs.StepWord αw)
    (hβ : AccumProofs.StepWord βw)
    (hβpos : 0 < PearlC.wordVal βw) (hw : w < 2 ^ 32) (hy : wordQ w = some y) (hcb : Rule.flatCode βw = .ok cb)
    (hc : Rule.endCodes αw cb w = .ok c) :
    CastOf c.1 (endStep αw βw y false) ∧ CastOf c.2 (endStep αw βw y true) := by
  have eMul : Evaluate.Fp32.f32MulV2 = Verity.Fp32.f32MulV2 := Properties.Evaluator.Fp32.f32MulV2
  have eWid : Evaluate.Casts.e4m3ToF32 = Verity.Prims.e4m3ToF32 := Properties.Evaluator.Casts.e4m3ToF32
  have eCast : Evaluate.Casts.f32ToE4m3Sat = Verity.Prims.f32ToE4m3Sat := Properties.Evaluator.Casts.f32ToE4m3Sat
  have hops := CoreOps.pc8OpsAgree_coreOps
  unfold Rule.flatCode at hcb
  obtain ⟨wf, hwf, hcb⟩ := WordsSim.bind_ok hcb
  obtain ⟨-, rfl⟩ := WordsSim.ofFp8_ok hwf
  obtain ⟨-, hcb⟩ := WordsSim.toFp8_ok hcb
  rw [eCast, eMul, eWid] at hcb
  have hfl : ∀ t : Fin 32, Pouw.PearlC.SaltDead.flatLine (fun _ => false) t = e4m3 44 := fun _ => rfl
  have h44 : (e4m3 (44 : ℚ)).val = Rule.E_FLAT := by decide +kernel
  have hnan44 : (e4m3 (44 : ℚ)).isNaN = false := by decide +kernel
  have hcbq : ∀ q : Fin Params.pi.r,
      cb = (lineCodes (PearlC.wordVal βw) (Pouw.PearlC.SaltDead.flatLine fun _ => false) q).val := by
    intro q
    refine Eq.trans ?_ (WholeCircuit.eTilde8_line hops hβ hβpos (e := Pouw.PearlC.SaltDead.flatLine fun _ => false) q
      hnan44)
    have henc : RowCircuit.encCodes (r := Params.pi.r) (Pouw.PearlC.SaltDead.flatLine fun _ => false) q =
        (e4m3 (44 : ℚ)).val := WholeCircuit.encCodes_of_lt q.isLt
    rw [hcb, RowCircuit.eTilde8, henc, PeelSim.eval_e4m3ToF32,
      Nat.mod_eq_of_lt (PeelSim.e4m3ToF32_lt _), StatsSim.eval_mul, Nat.mod_eq_of_lt (PeelSim.mulV2_lt _ _),
      PeelSim.eval_toE4m3, Nat.mod_eq_of_lt (PeelSim.toE4m3_lt _), h44]
  have hcb256 : cb < 256 := by
    rw [hcb]
    exact lt_of_lt_of_eq (PeelSim.toE4m3_lt _) (by norm_num)
  have hEl : (List.replicate 32 cb).map TileCircuit.code =
      List.ofFn (lineCodes (PearlC.wordVal βw) (Pouw.PearlC.SaltDead.flatLine fun _ => false)) := by
    rw [List.map_replicate, ← List.ofFn_const]
    congr 1
    funext t
    exact Fin.ext ((Nat.mod_eq_of_lt hcb256).trans (hcbq t))
  have hFp : (List.replicate 32 Rule.F_PLUS).map TileCircuit.code =
      List.ofFn (Pouw.PearlC.SaltDead.flatLineF fun _ => false) := by
    rw [List.map_replicate, ← List.ofFn_const]
    congr 1
    funext t
    rw [Pouw.PearlC.Cap.flatLineF_eq]
    rfl
  have hFm : (List.replicate 32 Rule.F_MINUS).map TileCircuit.code =
      List.ofFn (Pouw.PearlC.SaltDead.flatLineF fun _ => true) := by
    rw [List.map_replicate, ← List.ofFn_const]
    congr 1
    funext t
    rw [Pouw.PearlC.Cap.flatLineF_eq]
    rfl
  have hacc := mul_fp32Value hα hw hy
  have hstep : ∀ (F : List ℕ) (f : Fin 32 → Code), F.map TileCircuit.code = List.ofFn f →
      Words.atom (Evaluate.Fp32.f32MulV2 αw w) (List.replicate 32 cb) F =
        hopperE4m3K32.step (f32Word (PearlC.wordVal αw * y))
          (List.ofFn (lineCodes (PearlC.wordVal βw) (Pouw.PearlC.SaltDead.flatLine fun _ => false))) (List.ofFn f) := by
    intro F f hF
    have h1 : Words.atom (Evaluate.Fp32.f32MulV2 αw w) (List.replicate 32 cb) F =
        hopperE4m3K32.step (Evaluate.Fp32.f32MulV2 αw w) ((List.replicate 32 cb).map TileCircuit.code)
          (F.map TileCircuit.code) := atomIsTheStep _ _ _
    rw [h1, hEl, hF, eMul]
    exact step_acc _ hacc _ _
  unfold Rule.endCodes at hc
  obtain ⟨acc, hacc', hc⟩ := WordsSim.bind_ok hc
  obtain ⟨rfl, -⟩ := WordsSim.finite_ok hacc'
  try dsimp only at hc
  obtain ⟨Dp, hDp, hc⟩ := WordsSim.bind_ok hc
  obtain ⟨cp, hcp, hc⟩ := WordsSim.bind_ok hc
  obtain ⟨Dm, hDm, hc⟩ := WordsSim.bind_ok hc
  obtain ⟨cm, hcm, hc⟩ := WordsSim.bind_ok hc
  obtain rfl := WordsSim.pure_ok hc
  exact ⟨step_code _ (hstep _ _ hFp) ⟨_, _, _, rfl⟩ hDp hcp, step_code _ (hstep _ _ hFm) ⟨_, _, _, rfl⟩ hDm hcm⟩

/-- Where the support's two ends agree in the game (`saltDeadP_iff_ends_hopper`'s `EndsAgree`), the check's end codes
agree (`ends_agree`). -/
theorem endsAgree_of_endCodes {αw βw cb w : ℕ} {c : ℕ × ℕ} {y : ℚ} (hα : AccumProofs.StepWord αw)
    (hβ : AccumProofs.StepWord βw)
    (hβpos : 0 < PearlC.wordVal βw) (hw : w < 2 ^ 32) (hy : wordQ w = some y) (hcb : Rule.flatCode βw = .ok cb)
    (hc : Rule.endCodes αw cb w = .ok c)
    (hE : Pouw.PearlC.Cap.EndsAgree hopperE4m3K32 (PearlC.wordVal αw) y (PearlC.wordVal βw)) :
    Rule.endsAgree c.1 c.2 = true := by
  obtain ⟨⟨hcp', hEp, hnp⟩, ⟨hcm', hEm, hnm⟩⟩ := endCodes_eq hα hβ hβpos hw hy hcb hc
  have hE' : codeVal (e4m3Word (endStep αw βw y false)) = codeVal (e4m3Word (endStep αw βw y true)) := by
    rw [← hEp, ← hEm]
    exact hE
  rw [hcp', hcm']
  exact endsAgree_of ((Pouw.PearlC.Cap.codeVal_eq_iff _ _ hnp hnm).1 hE')

/-- **An element the game finds salt-dead the check finds salt-dead**: on a row whose α and β are step words, β
positive, where `flat_code` and `salt_dead` are values, the support's two ends agree in the game
(`saltDeadP_iff_ends_hopper`'s `EndsAgree`) only where the check's `ends_agree` holds. -/
theorem saltDead_of_ends {αw βw cb w : ℕ} {d : Bool} {y : ℚ} (hα : AccumProofs.StepWord αw)
    (hβ : AccumProofs.StepWord βw)
    (hβpos : 0 < PearlC.wordVal βw) (hw : w < 2 ^ 32) (hy : wordQ w = some y) (hcb : Rule.flatCode βw = .ok cb)
    (hd : Rule.saltDead αw cb w = .ok d)
    (hE : Pouw.PearlC.Cap.EndsAgree hopperE4m3K32 (PearlC.wordVal αw) y (PearlC.wordVal βw)) : d = true := by
  unfold Rule.saltDead at hd
  obtain ⟨c, hc, hd⟩ := WordsSim.bind_ok hd
  obtain rfl := WordsSim.pure_ok hd
  exact endsAgree_of_endCodes hα hβ hβpos hw hy hcb hc hE

/-- `row_elements` counts at least every element of a property each of whose elements is in an atom block every word
flags or is salt-dead in the check. -/
theorem rowElements_covers {αw βw k e : ℕ} {x : Array ℕ} {flags : List (Array Bool)} (Pl : ℕ → Prop)
    {inst : DecidablePred Pl} (hre : Rule.rowElements αw βw x flags = .ok e) (hk : k % 32 = 0) (hx : x.size = k)
    (hfl : flags ≠ []) (hsz : ∀ f ∈ flags, f.size = k / 32)
    (hP : ∀ l < k, Pl l → (∀ f ∈ flags, f[l / 32]! = true) ∨
      ∀ cb d, Rule.flatCode βw = .ok cb → Rule.saltDead αw cb x[l]! = .ok d → d = true) :
    (@Finset.filter _ Pl inst (Finset.range k)).card ≤ e := by
  have hb : (flags.head?.map (·.size)).getD 0 = k / 32 := by
    cases flags with
    | nil => exact absurd rfl hfl
    | cons f fs => simp [hsz f (by simp)]
  unfold Rule.rowElements at hre
  dsimp only at hre
  split at hre
  · simp only [pure, Except.pure, Except.ok.injEq] at hre
    subst hre
    rw [hb]
    calc (@Finset.filter _ Pl inst (Finset.range k)).card ≤ (Finset.range k).card := Finset.card_filter_le _ _
      _ = k := Finset.card_range k
      _ = 32 * (k / 32) := by omega
  · obtain ⟨cb, hcb, hre⟩ := WordsSim.bind_ok hre
    obtain ⟨dead, hdead, hre⟩ := WordsSim.bind_ok hre
    have he := WordsSim.pure_ok hre
    obtain ⟨-, hdA⟩ := WordsSim.mapM_ok hdead
    rw [← he, DebitSim.length_filter_range, hx]
    refine Finset.card_le_card fun l hl => ?_
    simp only [Finset.mem_filter] at hl ⊢
    have hlk : l < k := Finset.mem_range.1 hl.1
    refine ⟨hl.1, ?_⟩
    rw [Bool.or_eq_true]
    rcases hP l hlk hl.2 with hf | hs
    · left
      rw [hb, getElem!_pos _ (l / 32) (by simp; omega), Array.getElem_map, Array.getElem_range, List.all_eq_true]
      exact hf
    · right
      exact hs cb dead[l]! hcb (hdA l (by omega))

/-- **A row's forming elements in the game are within the check's count**: on a credited row of `k` finite words (a
whole number of promotion groups within the domain) that passes the rules, whose `s5_stats` are a value, where every
column's word flags the blocks the game's flags do and `row_elements` is a value, the game's forming elements of the
row (`rowGame`) are at most `row_elements`'. -/
theorem row_forming {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) {m n k i e s' ρw αw βw : ℕ}
    {A : Matrix (Fin m) (Fin k) Code} {B : Matrix (Fin n) (Fin k) Code} {act : Codes} {C : Finset ℕ}
    {xw : Array ℕ} {flags : List (Array Bool)} (hi : i < m) (hk : k % 128 = 0) (hk0 : 0 < k) (hkK : k ≤ 2 ^ 16)
    (hxs : xw.size = k) (hW : ∀ l < k, xw[l]! = act i l ∧ act i l < 2 ^ 32 ∧ (wordQ (act i l)).isSome)
    (hst : Words.s5Stats xw = .ok (s', ρw, αw, βw))
    (hR : (semH3 saltOf rootOf : PearlCSem Q R S).RowOK m k act i) (hfl : flags ≠ [])
    (hfs : ∀ f ∈ flags, f.size = k / 32)
    (hF : ∀ l < k, (∀ j : Fin n, (j : ℕ) ∈ C → devH100.flags A B ⟨i, hi⟩ j (l / 32) = true) →
      ∀ f ∈ flags, f[l / 32]! = true)
    (hre : Rule.rowElements αw βw xw flags = .ok e) :
    Cap.rowGame (semH3 saltOf rootOf : PearlCSem Q R S) devH100 A B act C i ≤ e := by
  obtain ⟨-, hα, hβ⟩ := statsAreTheRow k xw s' ρw αw βw hxs hk0 (by omega) (by omega)
    (fun l hl => by rw [(hW l hl).1]; exact ⟨(hW l hl).2.1, (hW l hl).2.2⟩) hst
  have hrv : rowVals k xw = WholeCircuit.rowQ k (act i) :=
    funext fun l => by simp only [rowVals, WholeCircuit.rowQ, (hW l l.isLt).1]
  rw [hrv] at hα hβ
  have h8 := WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
    (semH3 saltOf rootOf : PearlCSem Q R S).noise hi hR.1 (hR.2 hi).2
  obtain ⟨hαS, hβS, hβpos⟩ := h8.2
  rw [← hα] at hαS
  rw [← hβ] at hβS hβpos
  unfold Cap.rowGame
  refine rowElements_covers _ hre (by omega) hxs hfl hfs fun l hl hPl => ?_
  obtain ⟨hi', hl', hP⟩ := hPl
  rcases hP with hsd | hflg
  · right
    intro cb d hcb hd
    rw [(hW l hl).1] at hd
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 (hW l hl).2.2
    have hE0 := (Pouw.PearlC.Cap.saltDeadP_iff_ends_hopper (semH3 saltOf rootOf : PearlCSem Q R S).alpha
      (semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi'⟩) ⟨l, hl'⟩).1 hsd
    have hαv : (semH3 saltOf rootOf : PearlCSem Q R S).alpha k (wordsVal m k act ⟨i, hi'⟩) = PearlC.wordVal αw := by
      rw [hα]
      rfl
    have hβv : (semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi'⟩) = PearlC.wordVal βw := by
      rw [hβ]
      rfl
    have hyv : wordsVal m k act ⟨i, hi'⟩ ⟨l, hl'⟩ = y := by
      show (wordQ (act i l)).getD 0 = y
      simp [hy]
    rw [hαv, hβv, hyv] at hE0
    exact saltDead_of_ends hαS hβS hβpos (hW l hl).2.1 hy hcb hd hE0
  · left
    exact hF l hl hflg

end FormingSim

namespace MendSim

/-! ## The check's mends cover the game's -/

theorem aligned_toDef (w : ℕ) (E : ℤ) (t : Evaluate.TC.Fp8Atom.Term) :
    Evaluate.TC.Fp8Atom.aligned w E t = Pouw.Fp8Atom.aligned w E (termToDef t) := rfl

/-- One group's cut flags at an exponent, on the evaluator's model and on the Definitions'. -/
theorem cutMap_toDef (w : ℕ) (E : ℤ) (L : List (Fin 256 × Fin 256)) :
    (L.map fun x => Evaluate.TC.Fp8Atom.aligned w E (Evaluate.TC.Fp8Atom.product x.1 x.2) == 0) =
      L.map fun x => Pouw.Fp8Atom.aligned w E (Pouw.Fp8Atom.product x.1 x.2) == 0 :=
  List.map_congr_left fun x _ => by rw [aligned_toDef, product_toDef]

/-- The check's cut flags (`Rule.cutFold`, on the evaluator's grouped-sum model) are the Definitions' (`Mend.cutFold`),
on every pipeline, incoming value and operand codes. -/
theorem cutFold_toDef (p : Evaluate.TC.Fp8Atom.Pipeline) (gs : List ℕ) (v : Evaluate.TC.Fp8Atom.Value)
    (ab : List (Fin 256 × Fin 256)) :
    Rule.cutFold p gs v ab = Mend.cutFold (pipelineToDef p) gs (valueToDef v) ab := by
  induction gs generalizing v ab with
  | nil => rfl
  | cons size gs ih =>
    have hps : ((ab.take size).map fun x => Evaluate.TC.Fp8Atom.productTotal x.1 x.2).map valueToDef =
        (ab.take size).map fun x => Pouw.Fp8Atom.productTotal x.1 x.2 := by
      rw [List.map_map]
      exact List.map_congr_left fun x _ => productTotal_toDef x.1 x.2
    have hfin : ∀ L : List Evaluate.TC.Fp8Atom.Value,
        (L.map valueToDef).filterMap Pouw.Fp8Atom.Value.finite? =
          (L.filterMap Evaluate.TC.Fp8Atom.Value.finite?).map termToDef := by
      intro L
      rw [List.filterMap_map, List.map_filterMap]
      congr 1
      funext v
      cases v <;> rfl
    have hE : Evaluate.TC.Fp8Atom.maxExp p.zeroExp
        ((v :: (ab.take size).map fun x => Evaluate.TC.Fp8Atom.productTotal x.1 x.2).filterMap
          Evaluate.TC.Fp8Atom.Value.finite?) =
        Pouw.Fp8Atom.maxExp p.zeroExp
          ((valueToDef v :: (ab.take size).map fun x => Pouw.Fp8Atom.productTotal x.1 x.2).filterMap
            Pouw.Fp8Atom.Value.finite?) := by
      rw [maxExp_toDef, ← hfin, List.map_cons, hps]
    show (ab.take size).map (fun x => Evaluate.TC.Fp8Atom.aligned p.width
          (Evaluate.TC.Fp8Atom.maxExp p.zeroExp
            ((v :: (ab.take size).map fun x => Evaluate.TC.Fp8Atom.productTotal x.1 x.2).filterMap
              Evaluate.TC.Fp8Atom.Value.finite?))
          (Evaluate.TC.Fp8Atom.product x.1 x.2) == 0) ++
        Rule.cutFold p gs (Evaluate.TC.Fp8Atom.groupStepTotal p.width p.zeroExp v
          ((ab.take size).map fun x => Evaluate.TC.Fp8Atom.productTotal x.1 x.2)) (ab.drop size) =
      (ab.take size).map (fun x => Pouw.Fp8Atom.aligned p.width
          (Pouw.Fp8Atom.maxExp p.zeroExp
            ((valueToDef v :: (ab.take size).map fun x => Pouw.Fp8Atom.productTotal x.1 x.2).filterMap
              Pouw.Fp8Atom.Value.finite?))
          (Pouw.Fp8Atom.product x.1 x.2) == 0) ++
        Mend.cutFold (pipelineToDef p) gs (Pouw.Fp8Atom.groupStepTotal p.width p.zeroExp (valueToDef v)
          ((ab.take size).map fun x => Pouw.Fp8Atom.productTotal x.1 x.2)) (ab.drop size)
    rw [ih, groupStepTotal_toDef, hps, hE, cutMap_toDef]

/-- The check's cut flags of one H100 atom step are `Mend.cuts` at `hopperE4m3K32` on its operands' codes. -/
theorem cuts_eq (acc : ℕ) (a b : List ℕ) :
    Rule.cuts acc a b = Mend.cuts hopperE4m3K32 acc (a.map TileCircuit.code) (b.map TileCircuit.code) :=
  (cutFold_toDef Evaluate.TC.Fp8Atom.hopperE4m3K32 Evaluate.TC.Fp8Atom.hopperE4m3K32.groups
    (Evaluate.TC.Fp8Atom.fp32Value acc)
    ((a.map Evaluate.TC.Fp8Atom.code).zip (b.map Evaluate.TC.Fp8Atom.code))).trans (by
      rw [fp32Value_toDef]
      rfl)

/-- Two folds from one state agree where their steps agree at each position. -/
theorem foldl_rel {α β σ : Type} (f : σ → α → σ) (g : σ → β → σ) :
    ∀ (L : List α) (L' : List β) (s : σ), L.length = L'.length →
      (∀ i (h : i < L.length) (h' : i < L'.length) (s : σ), f s L[i] = g s L'[i]) → L.foldl f s = L'.foldl g s
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, hl, _ => by simp at hl
  | _ :: _, [], _, hl, _ => by simp at hl
  | a :: L, b :: L', s, hl, h => by
    have h0 : f s a = g s b := h 0 (Nat.succ_pos _) (Nat.succ_pos _) s
    rw [List.foldl_cons, List.foldl_cons, h0]
    exact foldl_rel f g L L' _ (by simpa using hl) fun i hi hi' s =>
      h (i + 1) (Nat.succ_lt_succ hi) (Nat.succ_lt_succ hi') s

/-- One atom of the check's chain of cuts is the game's, on operands that are the game's atom's codes. -/
theorem cutStep_eq (st : ℕ × ℕ × List Bool) (x : List ℕ × List ℕ) (y : AtomIn)
    (h1 : x.1.map TileCircuit.code = List.ofFn y.1) (h2 : x.2.map TileCircuit.code = List.ofFn y.2) :
    Rule.cutStep st x = Mend.chainStep hopperE4m3K32 4 st y := by
  have hstep : ∀ w, Words.atom w x.1 x.2 = hopperE4m3K32.step w (List.ofFn y.1) (List.ofFn y.2) := fun w =>
    (atomIsTheStep w x.1 x.2).trans (by rw [h1, h2])
  have hcuts : ∀ w, Rule.cuts w x.1 x.2 = Mend.cuts hopperE4m3K32 w (List.ofFn y.1) (List.ofFn y.2) := fun w =>
    (cuts_eq w x.1 x.2).trans (by rw [h1, h2])
  show (st.1 + 1, Words.atom (if st.1 % 4 = 0 then 0 else st.2.1) x.1 x.2,
      st.2.2 ++ Rule.cuts (if st.1 % 4 = 0 then 0 else st.2.1) x.1 x.2) =
    (st.1 + 1, hopperE4m3K32.step (if st.1 % 4 = 0 then 0 else st.2.1) (List.ofFn y.1) (List.ofFn y.2),
      st.2.2 ++ Mend.cuts hopperE4m3K32 (if st.1 % 4 = 0 then 0 else st.2.1) (List.ofFn y.1) (List.ofFn y.2))
  rw [hstep, hcuts]

/-- **The check's cut flags of a word are the game's**: on two rows of `K` codes, `Rule.wordCuts` is `Mend.chainCuts`
at the H100 (`hopperE4m3K32`, promoted every 4 atoms) of the rows' atoms. -/
theorem wordCuts_eq {K : ℕ} {a b : Array ℕ} {A B : Fin K → Code} (ha : a.size = K) (hb : b.size = K)
    (hA : ∀ t (ht : t < K), a[t]! = (A ⟨t, ht⟩).val) (hB : ∀ t (ht : t < K), b[t]! = (B ⟨t, ht⟩).val) :
    Rule.wordCuts a b = Mend.chainCuts hopperE4m3K32 4 (rowAtoms A B) := by
  have hpt : ∀ i (h : i < ((List.range (K / 32)).map fun t =>
        (Words.slice a (32 * t) 32, Words.slice b (32 * t) 32)).length) (h' : i < (rowAtoms A B).length)
      (st : ℕ × ℕ × List Bool),
      Rule.cutStep st (((List.range (K / 32)).map fun t =>
        (Words.slice a (32 * t) 32, Words.slice b (32 * t) 32))[i]) =
        Mend.chainStep hopperE4m3K32 4 st ((rowAtoms A B)[i]) := by
    intro i hi hi' st
    have hiK : i < K / 32 := by simpa using hi
    have hn : 32 * i + 32 ≤ K := by omega
    simp only [List.getElem_map, List.getElem_range, rowAtoms, List.getElem_ofFn]
    exact cutStep_eq st _ _ (WordsSim.slice_code (s := 32 * i) (n := i) ha hA rfl hn)
      (WordsSim.slice_code (s := 32 * i) (n := i) hb hB rfl hn)
  show (((List.range (a.size / 32)).map fun t => (Words.slice a (32 * t) 32, Words.slice b (32 * t) 32)).foldl
      Rule.cutStep (0, 0, [])).2.2 = ((rowAtoms A B).foldl (Mend.chainStep hopperE4m3K32 4) (0, 0, [])).2.2
  rw [ha, foldl_rel _ _ _ _ _ (by simp [rowAtoms]) hpt]

/-- The check's ends `e` of a row (`Rule.rowEnds`: its plus-end codes, its minus-end codes and its salt-dead flags) are
the game's for the row `x` of `k` elements: the codes are `Mend.endRow`'s at each end of the support, and every element
the game finds salt-dead is flagged. -/
def EndsOf {Q R S : Type} (sem : PearlCSem Q R S) {k : ℕ} (x : Fin k → ℚ) (e : Array ℕ × Array ℕ × Array Bool) :
    Prop :=
  e.1.size = k ∧ e.2.1.size = k ∧ e.2.2.size = k ∧
    (∀ l (hl : l < k), e.1[l]! = (Mend.endRow hopperE4m3K32 sem.alpha sem.beta k x false ⟨l, hl⟩).val) ∧
    (∀ l (hl : l < k), e.2.1[l]! = (Mend.endRow hopperE4m3K32 sem.alpha sem.beta k x true ⟨l, hl⟩).val) ∧
    ∀ l (hl : l < k), devH100.saltDead sem.alpha sem.beta k x ⟨l, hl⟩ → e.2.2[l]! = true

theorem getElem!_map {α β : Type} [Inhabited α] [Inhabited β] (f : α → β) (xs : Array α) {l : ℕ}
    (hl : l < xs.size) : (xs.map f)[l]! = f xs[l]! := by
  rw [getElem!_pos _ l (by rw [Array.size_map]; exact hl), Array.getElem_map, getElem!_pos xs l hl]

/-- **A row's ends in the check are the game's**: on a row of `k` finite words (a whole number of promotion groups
within the domain) that passes the rules, whose `s5_stats` are a value, where `row_ends` is a value, the check's end
codes are the game's `endRow` at each end (`endCodes_eq`) and every element the game finds salt-dead is flagged
(`endsAgree_of_endCodes`). -/
theorem row_ends {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) {m k i s' ρw αw βw : ℕ} {act : Codes}
    {xw : Array ℕ} {e : Array ℕ × Array ℕ × Array Bool} (hi : i < m) (hk : k % 128 = 0) (hk0 : 0 < k)
    (hkK : k ≤ 2 ^ 16) (hxs : xw.size = k)
    (hW : ∀ l < k, xw[l]! = act i l ∧ act i l < 2 ^ 32 ∧ (wordQ (act i l)).isSome)
    (hst : Words.s5Stats xw = .ok (s', ρw, αw, βw))
    (hR : (semH3 saltOf rootOf : PearlCSem Q R S).RowOK m k act i) (hre : Rule.rowEnds αw βw xw = .ok e) :
    EndsOf (semH3 saltOf rootOf : PearlCSem Q R S) (wordsVal m k act ⟨i, hi⟩) e := by
  obtain ⟨-, hα, hβ⟩ := statsAreTheRow k xw s' ρw αw βw hxs hk0 (by omega) (by omega)
    (fun l hl => by rw [(hW l hl).1]; exact ⟨(hW l hl).2.1, (hW l hl).2.2⟩) hst
  have hrv : rowVals k xw = WholeCircuit.rowQ k (act i) :=
    funext fun l => by simp only [rowVals, WholeCircuit.rowQ, (hW l l.isLt).1]
  rw [hrv] at hα hβ
  have h8 := WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
    (semH3 saltOf rootOf : PearlCSem Q R S).noise hi hR.1 (hR.2 hi).2
  obtain ⟨hαS, hβS, hβpos⟩ := h8.2
  rw [← hα] at hαS
  rw [← hβ] at hβS hβpos
  have hαv : (semH3 saltOf rootOf : PearlCSem Q R S).alpha k (wordsVal m k act ⟨i, hi⟩) = PearlC.wordVal αw := by
    rw [hα]
    rfl
  have hβv : (semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi⟩) = PearlC.wordVal βw := by
    rw [hβ]
    rfl
  unfold Rule.rowEnds at hre
  obtain ⟨cb, hcb, hre⟩ := WordsSim.bind_ok hre
  obtain ⟨ends, hends, hre⟩ := WordsSim.bind_ok hre
  obtain rfl := WordsSim.pure_ok hre
  obtain ⟨hsz, hall⟩ := WordsSim.mapM_ok hends
  have hel : ∀ l (hl : l < k), ∃ y, wordQ xw[l]! = some y ∧ wordsVal m k act ⟨i, hi⟩ ⟨l, hl⟩ = y ∧
      xw[l]! < 2 ^ 32 := by
    intro l hl
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 (hW l hl).2.2
    refine ⟨y, by rw [(hW l hl).1]; exact hy, ?_, by rw [(hW l hl).1]; exact (hW l hl).2.1⟩
    show (wordQ (act i l)).getD 0 = y
    simp [hy]
  have hrow : ∀ (minus : Bool) (l : ℕ) (hl : l < k) (y : ℚ), wordsVal m k act ⟨i, hi⟩ ⟨l, hl⟩ = y →
      Mend.endRow hopperE4m3K32 (semH3 saltOf rootOf : PearlCSem Q R S).alpha
          (semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi⟩) minus ⟨l, hl⟩ =
        e4m3 (PearlC.wordVal (FormingSim.endStep αw βw y minus)) := by
    intro minus l hl y hy
    show e4m3 (PearlC.wordVal (hopperE4m3K32.step
        (f32Word ((semH3 saltOf rootOf : PearlCSem Q R S).alpha k (wordsVal m k act ⟨i, hi⟩) *
          wordsVal m k act ⟨i, hi⟩ ⟨l, hl⟩))
        (List.ofFn (lineCodes ((semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi⟩))
          (Pouw.PearlC.SaltDead.flatLine fun _ => false)))
        (List.ofFn (Pouw.PearlC.SaltDead.flatLineF fun _ => minus)))) =
      e4m3 (PearlC.wordVal (hopperE4m3K32.step (f32Word (PearlC.wordVal αw * y))
        (List.ofFn (lineCodes (PearlC.wordVal βw) (Pouw.PearlC.SaltDead.flatLine fun _ => false)))
        (List.ofFn (Pouw.PearlC.SaltDead.flatLineF fun _ => minus))))
    rw [hαv, hβv, hy]
  unfold EndsOf
  refine ⟨?_, ?_, ?_, fun l hl => ?_, fun l hl => ?_, fun l hl hsd => ?_⟩
  · show (ends.map _).size = k
    rw [Array.size_map, hsz, hxs]
  · show (ends.map _).size = k
    rw [Array.size_map, hsz, hxs]
  · show (ends.map _).size = k
    rw [Array.size_map, hsz, hxs]
  · obtain ⟨y, hy, hyv, h32⟩ := hel l hl
    obtain ⟨⟨hc, hE, -⟩, -⟩ := FormingSim.endCodes_eq hαS hβS hβpos h32 hy hcb (hall l (by omega))
    show (ends.map _)[l]! = _
    rw [getElem!_map _ _ (by omega), hrow false l hl y hyv, hE]
    exact hc
  · obtain ⟨y, hy, hyv, h32⟩ := hel l hl
    obtain ⟨-, ⟨hc, hE, -⟩⟩ := FormingSim.endCodes_eq hαS hβS hβpos h32 hy hcb (hall l (by omega))
    show (ends.map _)[l]! = _
    rw [getElem!_map _ _ (by omega), hrow true l hl y hyv, hE]
    exact hc
  · obtain ⟨y, hy, hyv, h32⟩ := hel l hl
    have hE0 := (Pouw.PearlC.Cap.saltDeadP_iff_ends_hopper (semH3 saltOf rootOf : PearlCSem Q R S).alpha
      (semH3 saltOf rootOf : PearlCSem Q R S).beta k (wordsVal m k act ⟨i, hi⟩) ⟨l, hl⟩).1 hsd
    rw [hαv, hβv, hyv] at hE0
    show (ends.map _)[l]! = true
    rw [getElem!_map _ _ (by omega)]
    exact FormingSim.endsAgree_of_endCodes hαS hβS hβpos h32 hy hcb (hall l (by omega)) hE0

/-- **A word's mends in the game are within the check's**: where the two rows' ends in the check are the game's
(`EndsOf`), the game's `wordMend` at the H100 record, its dead pairs and its forced zeros, is at most the check's
`word_mend`. The forced zeros are the same products (`wordCuts_eq`); the check may flag more elements salt-dead. -/
theorem wordMend_le {Q R S : Type} (sem : PearlCSem Q R S) {k : ℕ} {x w : Fin k → ℚ}
    {ea eb : Array ℕ × Array ℕ × Array Bool} (ha : EndsOf sem x ea) (hb : EndsOf sem w eb) :
    wordMend devH100 sem.alpha sem.beta x w ≤ Rule.wordMend ea eb := by
  classical
  obtain ⟨ha1, ha2, ha3, hap, ham, had⟩ := ha
  obtain ⟨hb1, hb2, -, hbp, hbm, hbd⟩ := hb
  have hcp := wordCuts_eq (A := Mend.endRow hopperE4m3K32 sem.alpha sem.beta k x false)
    (B := Mend.endRow hopperE4m3K32 sem.alpha sem.beta k w false) ha1 hb1 hap hbp
  have hcm := wordCuts_eq (A := Mend.endRow hopperE4m3K32 sem.alpha sem.beta k x true)
    (B := Mend.endRow hopperE4m3K32 sem.alpha sem.beta k w true) ha2 hb2 ham hbm
  show _ ≤ ((List.range ea.2.2.size).filter fun l => (ea.2.2[l]! && eb.2.2[l]!) ||
    ((Rule.wordCuts ea.1 eb.1).getD l false && (Rule.wordCuts ea.2.1 eb.2.1).getD l false)).length
  rw [DebitSim.length_filter_range, ha3, hcp, hcm]
  unfold wordMend
  refine Finset.card_le_card_of_injOn (fun l : Fin k => (l : ℕ)) ?_ ?_
  · intro l hl
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_range] at hl ⊢
    refine ⟨l.isLt, ?_⟩
    rw [Bool.or_eq_true, Bool.and_eq_true, Bool.and_eq_true]
    rcases hl with ⟨h1, h2⟩ | h
    · exact .inl ⟨had l l.isLt h1, hbd l l.isLt h2⟩
    · exact .inr h
  · intro l _ l' _ h
    exact Fin.ext h

end MendSim

open DebitSim in
/-- **A word's codes**: where `k` is whole promotion groups, the check's A′ and B̃ rows of a word whose rows pass the
rules are `unitChain`'s codes at `semH3`'s noise, whose lines and bases are the check's (`seedsAreTheSpec`,
`lineIsTheSpec`, `fixedNoiseIsTheSpec`). -/
theorem codes_eq {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (U : Pouw.PearlC.Workload) (u : ℕ) (act : Codes) (salt rootB : ByteArray) (wn : Words.FixedNoise)
    (rows cols : List ℕ) (ops : Words.Ops) (hsalt : bytesNat salt = saltOf s)
    (hroot : bytesNat rootB = rootOf (U.layout.fixedId u)) (hk : (U.layout.shape u).k ≤ 2 ^ 16)
    (hk128 : (U.layout.shape u).k % 128 = 0) (hact : ∀ i ∈ rows, ∀ l < (U.layout.shape u).k, act i l < 2 ^ 32)
    (hwt : ∀ j ∈ cols, ∀ l < (U.layout.shape u).k, U.fixed (U.layout.fixedId u) j l < 2 ^ 32)
    (hW : ∀ j ∈ cols, (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).n (U.layout.shape u).k
      (U.fixed (U.layout.fixedId u)) j)
    (hwn : Words.fixedNoise salt rootB (U.layout.shape u).k = .ok wn)
    (henc : Words.encode salt rootB wn u (U.layout.shape u).k rows (rows.map (rowBytesOf act (U.layout.shape u).k))
      cols (cols.map (rowBytesOf (U.fixed (U.layout.fixedId u)) (U.layout.shape u).k)) = .ok ops)
    (x z : ℕ) (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n) (hxi : rows[x]? = some i.val)
    (hyj : cols[z]? = some j.val)
    (hR : (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape u).m (U.layout.shape u).k act i) :
    ops.a.codes[x]! =
        Array.ofFn (fun l => (((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).1 i l).val) ∧
      ops.b.codes[z]! =
        Array.ofFn (fun l => (((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U u act).2 j l).val) := by
  obtain ⟨hx, hxi'⟩ := List.getElem?_eq_some_iff.1 hxi
  obtain ⟨hy, hyj'⟩ := List.getElem?_eq_some_iff.1 hyj
  have hWj := hW _ (List.mem_of_getElem? hyj)
  obtain ⟨hseed, -, -, hF⟩ := fixedNoiseIsTheSpec salt rootB _ wn hwn
  have hsd : bytesNat wn.seedB = NoiseSha512.seedB (saltOf s) (rootOf (U.layout.fixedId u)) := by
    rw [hseed, hsalt, hroot]
  obtain ⟨-, -, hAB⟩ := encode_codes (z := (semH3 saltOf rootOf : PearlCSem Q R S).noise H s U u act)
    (wt := U.fixed (U.layout.fixedId u)) hk hact hwt
    (fun l hl t => by rw [(hF l hl t).1, hsd]; rfl) (fun l hl t => by rw [(hF l hl t).2, hsd]; rfl)
    (fun i e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 0 0 0 Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, (seedsAreTheSpec).2.2, hsalt, hroot, WordsSim.bytesNat_rowBytesOf]
      rfl)
    (fun j e he => by
      obtain ⟨h1, h2⟩ := lineIsTheSpec _ 1 0 j.val Words.NORM_E 256 e (by norm_num) (by norm_num) (Or.inl ⟨rfl, rfl⟩) he
      refine ⟨h1, fun t => ?_⟩
      rw [h2 t, hsd]
      rfl)
    hwn henc i j hx (by rw [getElem!_pos rows x hx]; exact hxi') hy (by rw [getElem!_pos cols z hy]; exact hyj')
  obtain ⟨hA, hB⟩ := hAB (Nat.dvd_of_mod_eq_zero (by omega))
    (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
      (semH3 saltOf rootOf : PearlCSem Q R S).noise i.isLt hR.1 (hR.2 i.isLt).2)
    (WholeCircuit.row8OK_of_floor CoreOps.pc8OpsAgree_coreOps hopperE4m3K32
      (semH3 saltOf rootOf : PearlCSem Q R S).noise j.isLt hWj.1 (hWj.2 j.isLt).2)
  obtain ⟨hU1, hU2⟩ := unitChain_h3 saltOf rootOf H s U u act
  rw [hA, hB, hU1, hU2]
  exact ⟨rfl, rfl⟩

/-- **(c)** from what a passed tile establishes (`TileAcceptMeans`), the check's inert-work bound (`TileKeepsTheCap`),
what a loaded pass's records keep (`load_facts`), the per-row rules (`RowsAreTheRules`), the replay
(`wordFlagsIsTheReplay`, `codes_eq`), the forming count (`formingElems_le`, `row_forming`) and the mends
(`MendSim.row_ends`, `MendSim.wordMend_le`): a tile as in `TilePassGivesTheCap` has its chain's inert work, its forming
and its mends within the game's bound. -/
theorem tile_bound (hT : Properties.Pouw.PearlPlus.TileAcceptMeans) (hK : Properties.Pouw.PearlPlus.TileKeepsTheCap)
    (hR : RowsAreTheRules) {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCheck.PearlPlus.EpochCoins) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit)
    (hload : PouwCheck.PearlPlus.load coins man = .ok p) (hof : TileOf p tid U g) (hsalt : bytesNat p.salt = saltOf s)
    (hroot : bytesNat ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD
      ByteArray.empty) = rootOf (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)))
    (hwOK : ∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).fixedOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) j)
    (hopen : OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c) (hticket : CarriesATicket p tid)
    (hpass : checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = .pass) :
    tileChainDebit (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g)
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act +
      tileFormingDebit (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g)
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act +
      tileMendDebit (semH3 saltOf rootOf : PearlCSem Q R S) devH100 U ((auditTiling 64 64 U.layout).unit g)
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act ≤
      1 / 400 * creditDevRev1 devH100 (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
        tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
          ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
            ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) := by
  obtain ⟨pos, rows, cols, htile, -, hunit, -, hk, hn, hwid, hrows, hcols⟩ := hof
  obtain ⟨pos', rows', cols', ops, ch, htile', hrest⟩ := hT p _ tid o none hpass
  simp only [htile, Option.some.injEq, Prod.mk.injEq] at htile'
  obtain ⟨rfl, rfl, rfl⟩ := htile'
  try dsimp only at hrest
  obtain ⟨-, -, -, -, -, -, hform, -, hchk, hcapC, -⟩ := hrest
  obtain ⟨hp, -⟩ := Draw.tile?_some htile
  have hrec : p.records[pos]! ∈ p.records := by rw [getElem!_pos p.records pos hp]; exact List.getElem_mem hp
  obtain ⟨-, -, hkept0, hkept⟩ := Draw.reason_none ((Properties.Pouw.PearlPlus.load_facts hload).1 _ hrec)
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
  have hvolA := Cap.band_anyA hbandR hbA
  have hcc := Cap.credited_cols hbandC hbB
  have hrow : ∀ x (hx : x < rows.length),
      (ops.a.x[x]!).size = (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k ∧
      (∀ l < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k,
        (ops.a.x[x]!)[l]! = c.act rows[x] l ∧ c.act rows[x] l < 2 ^ 32 ∧ (wordQ (c.act rows[x] l)).isSome) ∧
      (∃ s', Words.s5Stats ops.a.x[x]! = .ok (s', ops.a.rho[x]!, ops.a.alpha[x]!, ops.a.beta[x]!)) ∧
      (Rule.rowPasses ops.a.rho[x]! ops.a.alpha[x]! ops.a.x[x]! = true ↔
        (semH3 saltOf rootOf : PearlCSem Q R S).RowOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
          (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act rows[x]) := by
    intro x hx
    have him := hrowsm _ (List.getElem_mem hx)
    obtain ⟨hparse, s', hst⟩ := Cap.encode_stats (by simp) henc x hx
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
    exact ⟨hsz, hW, ⟨s', hst⟩, @hR Q R S saltOf rootOf _ _ c.act rows[x] ops.a.x[x]! s' _ _ _ him (by omega)
      (by omega) (by omega) hsz hW hst⟩
  have hcol : ∀ z (hz : z < cols.length),
      (ops.b.x[z]!).size = (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k ∧
      (∀ l < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k,
        (ops.b.x[z]!)[l]! = U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) cols[z] l ∧
          U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) cols[z] l < 2 ^ 32 ∧
          (wordQ (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) cols[z] l)).isSome) ∧
      ∃ s', Words.s5Stats ops.b.x[z]! = .ok (s', ops.b.rho[z]!, ops.b.alpha[z]!, ops.b.beta[z]!) := by
    intro z hz
    obtain ⟨hparse, s', hst⟩ := Cap.encode_stats_b (by simp) henc z hz
    rw [getElem!_pos _ z (by simpa using hz), List.getElem_map, Draw.rowBytesOf_eq] at hparse
    obtain ⟨hsz, hwords⟩ := Draw.parseRow_wordBytes hparse
    refine ⟨hsz, fun l hl => ?_, s', hst⟩
    have h32 := hwt cols[z] (List.getElem_mem hz) l hl
    obtain ⟨hxl, hfin⟩ := hwords l hl
    rw [Nat.mod_eq_of_lt h32] at hxl
    refine ⟨hxl, h32, ?_⟩
    rw [hxl, word_and_ff] at hfin
    have hne : (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g)) cols[z] l >>> 23) % 256 ≠ 255 := by
      simpa using hfin
    rw [Pouw.PearlC.PeelExactProofs.wordQ_fields hne]
    rfl
  have hcr_iff : ∀ i, i ∈ (p.records[pos]!.credited rows cols).1 ↔
      i ∈ rows ∧ (semH3 saltOf rootOf : PearlCSem Q R S).RowOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act i := by
    intro i
    simp only [UnitRecord.credited, List.mem_filter]
    constructor
    · rintro ⟨hi, hc⟩
      obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
      exact ⟨hi, (hrow x hx).2.2.2.1 (Draw.rowsFailure_credited hchk hx (banded_row hc))⟩
    · rintro ⟨hi, hok⟩
      obtain ⟨x, hx, rfl⟩ := List.getElem_of_mem hi
      have hlt : rows[x] < p.records[pos]!.realRows := by
        by_contra hge
        have hz := Cap.rowsFailure_filler hchk hx (Nat.le_of_not_lt hge)
        have hlive := Cap.liveRow_zero (by rw [(hrow x hx).1]; exact hk0) (Cap.zero_of_any hz)
        have h2 := (hrow x hx).2.2.2.2 hok
        rw [Rule.rowPasses, hlive, Bool.and_false] at h2
        cases h2
      have hex : p.records[pos]!.excludedA.contains rows[x] = false := by
        cases hc : p.records[pos]!.excludedA.contains rows[x]
        · rfl
        · have h1 := Cap.rowsFailure_excluded hchk hx hlt hc
          have h2 := (hrow x hx).2.2.2.2 hok
          rw [h1] at h2
          cases h2
      exact ⟨hi, by simp only [decide_eq_true hlt, hex, hvolA _ hi, Bool.not_false, Bool.and_self]⟩
  have hcrnd : (p.records[pos]!.credited rows cols).1.Nodup := hrowsnd.filter _
  have hccne : cols ≠ [] := by rw [← hcc]; exact hbB
  have hcrne := hbA
  obtain ⟨cr, hcrdef⟩ : ∃ cr, (p.records[pos]!.credited rows cols).1 = cr := ⟨_, rfl⟩
  rw [hcrdef] at hcrnd hcrne hcr_iff
  have hpair : p.records[pos]!.credited rows cols = (cr, cols) := Prod.ext hcrdef hcc
  have hK' := hK p.scheme p.records[pos]! rows cols ops (Cap.checkOpened_none hchk hcapC)
  rw [hpair] at hK'
  try dsimp only at hK'
  obtain ⟨words, elems, endsA, endsB, hw, he, hEA, hEB, hcap, -⟩ := hK' hcrne hccne
  have hCC := codes_eq saltOf rootOf H s U ((auditTiling 64 64 U.layout).unit g) c.act p.salt
    ((p.weightRoots.lookup (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) wn' rows cols
    ops hsalt hroot hKmax' hk128 hact hwt (fun j hj => hwOK j (by rw [hcols]; exact List.mem_toFinset.2 hj)) hwn henc
  have hcell : ∀ i ∈ cr, ∀ j ∈ cols, ∀ v : Array Bool × ℕ,
      Rule.wordFlags ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]! = .ok v →
      (v.1.filter id).size = Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j ∧
      v.2 = Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j ∧
      ∀ (him : i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m)
        (hjn : j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n),
        v.1.size = (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k / 32 ∧
        ∀ t < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k / 32, v.1[t]! =
          Skip.unitFlags
            ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U ((auditTiling 64 64 U.layout).unit g) c.act).1
            ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U ((auditTiling 64 64 U.layout).unit g) c.act).2
            ⟨i, him⟩ ⟨j, hjn⟩ t := by
    intro i hi j hj v hv
    obtain ⟨hir, hok⟩ := (hcr_iff i).1 hi
    have him := hrowsm i hir
    have hjn := hcolsn j hj
    have hx := List.idxOf_lt_length_of_mem hir
    have hz := List.idxOf_lt_length_of_mem hj
    obtain ⟨hA, hB⟩ := hCC (rows.idxOf i) (cols.idxOf j) ⟨i, him⟩ ⟨j, hjn⟩
      (by rw [List.getElem?_eq_getElem hx, List.getElem_idxOf hx])
      (by rw [List.getElem?_eq_getElem hz, List.getElem_idxOf hz]) hok
    rw [hA, hB] at hv
    obtain ⟨fl, z⟩ := v
    obtain ⟨hfs, hft, hfc, hzz⟩ := wordFlagsIsTheReplay _ _ (⟨i, him⟩ : Fin _) (⟨j, hjn⟩ : Fin _) hv
    have h : i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m ∧
        j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n := ⟨him, hjn⟩
    refine ⟨?_, ?_, fun _ _ => ⟨hfs, hft⟩⟩
    · rw [Cap.cellF, dite_eq_left h]
      exact hfc
    · rw [Cap.cellZ, dite_eq_left h]
      exact hzz
  have hlw : words.length = cr.length := (WordsSim.listMapM_ok hw).1
  have hle : elems.length = cr.length := by
    rw [(WordsSim.listMapM_ok he).1, List.length_zip, hlw, Nat.min_self]
  have hlen : words.length = elems.length := hlw.trans hle.symm
  have hsumW : (words.map fun w => 40 * (w.map fun v => (v.1.filter id).size).sum +
      32 * (w.map fun v => v.2).sum).sum =
      ∑ i ∈ cr.toFinset, ∑ j ∈ cols.toFinset,
        (40 * Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j +
          32 * Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i j) :=
    (Cap.mapM_sum_eq
      (g := fun i => 40 * (cols.map
          (Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i)).sum +
        32 * (cols.map (Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i)).sum)
      (g' := fun w => 40 * (w.map fun v => (v.1.filter id).size).sum + 32 * (w.map fun v => v.2).sum) hw
      fun i hi ws hws => by
        have h1 := Cap.mapM_sum_eq
          (g := Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i)
          (g' := fun v : Array Bool × ℕ => (v.1.filter id).size) hws fun j hj v hv => (hcell i hi j hj v hv).1
        have h2 := Cap.mapM_sum_eq
          (g := Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act i)
          (g' := fun v : Array Bool × ℕ => v.2) hws fun j hj v hv => (hcell i hi j hj v hv).2.1
        show 40 * (ws.map fun v => (v.1.filter id).size).sum + 32 * (ws.map fun v => v.2).sum = _
        rw [h1, h2]).trans
      (Cap.cells_sum cr cols hcrnd hcolsnd
        (Cap.cellF (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act)
        (Cap.cellZ (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act))
  have hE : (cr.map fun i => Cap.rowGame (semH3 saltOf rootOf : PearlCSem Q R S) devH100
      ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U ((auditTiling 64 64 U.layout).unit g) c.act).1
      ((semH3 saltOf rootOf : PearlCSem Q R S).unitChain H s U ((auditTiling 64 64 U.layout).unit g) c.act).2
      c.act cols.toFinset i).sum ≤ elems.sum := by
    rw [← Cap.map_zip_fst _ cr words hlw.symm]
    refine Cap.mapM_sum_le he fun q hq e' he' => ?_
    obtain ⟨i, ws⟩ := q
    try dsimp only at he' ⊢
    have hiw : cols.mapM (fun j => Rule.wordFlags ops.a.codes[rows.idxOf i]! ops.b.codes[cols.idxOf j]!) =
        .ok ws := Cap.mapM_ok_zip hw (i, ws) hq
    have hi := (List.of_mem_zip hq).1
    obtain ⟨hir, hok⟩ := (hcr_iff i).1 hi
    have him := hrowsm i hir
    have hx := List.idxOf_lt_length_of_mem hir
    obtain ⟨hsz, hW, ⟨s', hst⟩, -⟩ := hrow _ hx
    rw [List.getElem_idxOf hx] at hW
    have hwsl : ws.length = cols.length := (WordsSim.listMapM_ok hiw).1
    refine FormingSim.row_forming saltOf rootOf him hk128 hk0 hKmax' hsz hW hst hok ?_ ?_ ?_ he'
    · rw [Ne, List.map_eq_nil_iff, ← List.length_eq_zero_iff, hwsl, List.length_eq_zero_iff]
      exact hccne
    · intro f hf
      obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hf
      obtain ⟨j, hj, hvj⟩ := Assembly.mapM_ok_mem hiw v hv
      exact ((hcell i hi j hj v hvj).2.2 him (hcolsn j hj)).1
    · intro l hl hall f hf
      obtain ⟨v, hv, rfl⟩ := List.mem_map.1 hf
      obtain ⟨j, hj, hvj⟩ := Assembly.mapM_ok_mem hiw v hv
      have hjn := hcolsn j hj
      obtain ⟨-, hft⟩ := (hcell i hi j hj v hvj).2.2 him hjn
      rw [hft (l / 32) (by omega)]
      exact hall ⟨j, hjn⟩ (List.mem_toFinset.2 hj)
  have hFE : formingElems (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g) cr.toFinset
      cols.toFinset c.act ≤ elems.sum :=
    (Cap.formingElems_le (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g) cr.toFinset
      cols.toFinset c.act).trans (by rw [List.sum_toFinset _ hcrnd]; exact hE)
  have hlA : endsA.length = cr.length := (WordsSim.listMapM_ok hEA).1
  have hlB : endsB.length = cols.length := (WordsSim.listMapM_ok hEB).1
  have hendA : ∀ i ea, (i, ea) ∈ cr.zip endsA →
      ∀ him : i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m,
        MendSim.EndsOf (semH3 saltOf rootOf : PearlCSem Q R S)
          (wordsVal (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
            (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act ⟨i, him⟩) ea := by
    intro i ea hq him
    obtain ⟨hir, hok⟩ := (hcr_iff i).1 (List.of_mem_zip hq).1
    have hx := List.idxOf_lt_length_of_mem hir
    obtain ⟨hsz, hW, ⟨s', hst⟩, -⟩ := hrow _ hx
    rw [List.getElem_idxOf hx] at hW
    have hre : Rule.rowEnds ops.a.alpha[rows.idxOf i]! ops.a.beta[rows.idxOf i]! ops.a.x[rows.idxOf i]! = .ok ea :=
      Cap.mapM_ok_zip hEA (i, ea) hq
    exact MendSim.row_ends saltOf rootOf him hk128 hk0 hKmax' hsz hW hst hok hre
  have hendB : ∀ j eb, (j, eb) ∈ cols.zip endsB →
      ∀ hjn : j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n,
        MendSim.EndsOf (semH3 saltOf rootOf : PearlCSem Q R S)
          (wordsVal (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n
            (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k
            (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) ⟨j, hjn⟩) eb := by
    intro j eb hq hjn
    have hj := (List.of_mem_zip hq).1
    have hz := List.idxOf_lt_length_of_mem hj
    obtain ⟨hsz, hW, s', hst⟩ := hcol _ hz
    rw [List.getElem_idxOf hz] at hW
    have hre : Rule.rowEnds ops.b.alpha[cols.idxOf j]! ops.b.beta[cols.idxOf j]! ops.b.x[cols.idxOf j]! = .ok eb :=
      Cap.mapM_ok_zip hEB (j, eb) hq
    have hRj : (semH3 saltOf rootOf : PearlCSem Q R S).RowOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k
        (U.fixed (U.layout.fixedId ((auditTiling 64 64 U.layout).unit g))) j :=
      hwOK j (by rw [hcols]; exact List.mem_toFinset.2 hj)
    exact MendSim.row_ends saltOf rootOf hjn hk128 hk0 hKmax' hsz hW hst hRj hre
  have hM : (∑ i ∈ cr.toFinset, ∑ j ∈ cols.toFinset,
      Cap.cellM (semH3 saltOf rootOf : PearlCSem Q R S) U ((auditTiling 64 64 U.layout).unit g) c.act i j) ≤
      (endsA.map fun ea => (endsB.map (Rule.wordMend ea)).sum).sum := by
    have hin : ∀ i, (∑ j ∈ cols.toFinset,
        Cap.cellM (semH3 saltOf rootOf : PearlCSem Q R S) U ((auditTiling 64 64 U.layout).unit g) c.act i j) =
        (cols.map (Cap.cellM (semH3 saltOf rootOf : PearlCSem Q R S) U ((auditTiling 64 64 U.layout).unit g)
          c.act i)).sum := fun i => List.sum_toFinset _ hcolsnd
    rw [List.sum_toFinset _ hcrnd]
    simp only [hin]
    refine Cap.sum_le_zip _ _ cr endsA hlA.symm fun q hq => ?_
    obtain ⟨i, ea⟩ := q
    have him := hrowsm i ((hcr_iff i).1 (List.of_mem_zip hq).1).1
    have hA := hendA i ea hq him
    show (cols.map (Cap.cellM (semH3 saltOf rootOf : PearlCSem Q R S) U ((auditTiling 64 64 U.layout).unit g)
      c.act i)).sum ≤ (endsB.map (Rule.wordMend ea)).sum
    refine Cap.sum_le_zip _ _ cols endsB hlB.symm fun q' hq' => ?_
    obtain ⟨j, eb⟩ := q'
    have hjn := hcolsn j (List.of_mem_zip hq').1
    have h : i < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m ∧
        j < (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).n := ⟨him, hjn⟩
    show Cap.cellM (semH3 saltOf rootOf : PearlCSem Q R S) U ((auditTiling 64 64 U.layout).unit g) c.act i j ≤
      Rule.wordMend ea eb
    simp only [Cap.cellM, dif_pos h]
    exact MendSim.wordMend_le _ hA (hendB j eb hq' hjn)
  rw [Cap.rowDebit_sum_mend _ _ words elems endsA endsB hlen (hle.trans hlA.symm), hsumW] at hcap
  simp only [Rule.CAP_NUM, Rule.CAP_DEN, Nat.one_mul] at hcap
  rw [← hn] at hcap
  have hD' := (Nat.mul_comm _ _).trans_le hcap
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
  unfold tileFormingDebit
  rw [← hP, hcols, Cap.tileChainDebit_cells (semH3 saltOf rootOf) H s U ((auditTiling 64 64 U.layout).unit g) c.act
    _ _ hPm hCn, Cap.tileMendDebit_cells (semH3 saltOf rootOf : PearlCSem Q R S) U
    ((auditTiling 64 64 U.layout).unit g) c.act _ _ hPm hCn, tileShare, List.toFinset_card_of_nodup hcrnd,
    List.toFinset_card_of_nodup hcolsnd]
  exact Cap.bound_glue_mend hm0 hn0 (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by rw [show devH100.prices.costs.fs = 64 from rfl]; norm_num) (Nat.cast_le.2 hFE) (Nat.cast_le.2 hM)
    (by exact_mod_cast hD') hcred

/-- **(c)'s chain part** from `TileAcceptMeans`, the check's cap (`TileKeepsTheCap`) and `RowsAreTheRules`:
`tile_bound` less its forming and its mends, which are nonnegative. -/
theorem tilePassGivesTheChainCap (hT : Properties.Pouw.PearlPlus.TileAcceptMeans)
    (hK : Properties.Pouw.PearlPlus.TileKeepsTheCap) (hR : RowsAreTheRules) : TilePassGivesTheChainCap := by
  intro Q R S saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket hpass
  have h := tile_bound hT hK hR saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket
    hpass
  have h0 : 0 ≤ tileFormingDebit (semH3 saltOf rootOf) devH100 H s U ((auditTiling 64 64 U.layout).unit g)
      ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
        ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act := by
    unfold tileFormingDebit
    positivity
  have h1 : 0 ≤ tileMendDebit (semH3 saltOf rootOf : PearlCSem Q R S) devH100 U ((auditTiling 64 64 U.layout).unit g)
      ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
        ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act := by
    unfold tileMendDebit
    exact Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _
  exact le_trans (le_add_of_nonneg_right h0) (le_trans (le_add_of_nonneg_right h1) h)

/-- **(c)** from `TileAcceptMeans`, the check's cap (`TileKeepsTheCap`) and `RowsAreTheRules`: `tile_bound`, whose
chain, forming and mends are `tileDebitRev1`. -/
theorem tilePassGivesTheCap (hT : Properties.Pouw.PearlPlus.TileAcceptMeans)
    (hK : Properties.Pouw.PearlPlus.TileKeepsTheCap) (hR : RowsAreTheRules) : TilePassGivesTheCap := by
  intro Q R S saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket hpass
  show (semH3 saltOf rootOf : PearlCSem Q R S).tileDebitRev1 devH100 H s U ((auditTiling 64 64 U.layout).unit g)
      ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
        ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g) c.act ≤
    1 / 400 * creditDevRev1 devH100 (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
      tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        ((semH3 saltOf rootOf : PearlCSem Q R S).passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) c.act
          ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g)
  rw [tileDebitRev1_split]
  exact tile_bound hT hK hR saltOf rootOf H s coins man p tid o U g c hload hof hsalt hroot hwOK hopen hticket hpass

end Pouw.PearlPlus.Bridge
