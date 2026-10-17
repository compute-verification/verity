import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.VBridge.OpeningUnit
import Proofs.Flock.VBridge.Sparse.Laid
import Proofs.Flock.VBridge.Sparse.Unit

/-!
# V*'s parts at the sparse shape

At the sparse shape (`Form.Sparse`, candidate 3's statements `RecVStar.candidate3Statements`) V* stages, besides the
dense shape's parts, R's levels, R's opening parts and the sparse phase. Each part is a table of one unit the verifier
rebuilds itself and `HmRow.parse` checks (`ParseFacts`), so each part's `PartChecks` is its unit's semantics
(`VStarParts.lean`'s method: `laidRows_of_satisfies`, then the unit's laid-out theorem) and links that are not the
circuit's. The parts share three units:

* **rec-open** (`climbAt_of_satisfies`): the table's Ligerito levels (`.recOpen`, `rec-open/l<L>-h<H>`, `RecOpen.params`),
  as at the dense shape, and R's levels (`.recOpenStacked`, the same net under `rec-open-stacked/l<L>-h<H>`,
  `RecOpen.stackedParams`, `RecOpen.check_ok_stacked`: `climbStacked_of_satisfies`). `laid_unit`.
* **rec-residuals** (`resCols_of_check`): the rep's algebra (`.inner`, `rec-residuals/m-k-c-p`, `RecResiduals.part`), as
  at the dense shape, and R's opening parts (`.opening`, `rec-opening/m-c-p`, the residuals net at `RecOpening.part`,
  `check_opening`: `openingCols_of_satisfies`). `laid_part`.
* **rec-sparse** (`sparseOuts_of_satisfies`), new: the sparse phase (`.sparse`, `rec-sparse/n-e-s-t`, `check_sparse`).
  `laid_sparse`, at the `T_LOG ≤ 16` the check enforces: a copy whose input columns hold the message ports' elements,
  `v`'s and the slots' types (`SparseIn`) holds `outOf`'s residuals on its output columns (`SparseOuts`).

The rest:

* **The checks the parser made** (`parseFacts_of_tablesOf`, `parseFacts_of_check`): every table of an untyped parsed
  statement is a circuit `HmRow.parse` returned, so `RecOpen.check`, `RecResiduals.check`, `RecSparse.check` and
  `RecOpening.check` accepted it. The links here carry only the unit's name; the check is the parser's.
* **A part's links at one statement** (`ClimbAt`, `ResAt`, `SparseAt`; together `PartAt`): at a V* statement `xO` of `x`,
  a table whose unit is named for one of the units above, one of its copies, the input tie (the copy's input columns
  hold the unit's inputs) and the output tie (the unit's outputs there give a goal at the witness and the messages).
  `PartLinksTo s` is `PartAt` wherever the V* statement shows the coins `c`, its goal at `c` or `PartChecks`' fallback
  (`PartColl`: two of the part's recorded calls collide), so a part may be of a different kind at different compiled
  statements. `PartLinks s` has the goal the inner check `chk s`; `LogLinks s` has the goal that the calls the inner
  check makes (`logs s`) are among the part's recorded calls, which at a rec-open part are the climb's.
* **The input tie off the columns** (`resAt_of_reads`, `sparseAt_of_reads`): a copy's input columns hold the elements
  they read (`inCols_readElems`, `sparseIn_reads` where no types are stacked, `T_LOG = 0` as in scope), so the input tie
  is left only in the output tie's premise.
* **`PartChecks` and `PartLogs` at every part** (`goal_of_links`; `partChecks_of_links`, `partLogs_of_links`):
  `PartLinks s` gives `PartChecks s` and `LogLinks s` gives `PartLogs s`, the goal where the part's calls don't collide.
  M's dense-shape links, whose output ties give the inner check outright, give `PartLinks` (`partLinks_of_resLinks`,
  `partLinks_of_openLinks`). A part whose inner check makes no call has `PartLogs` outright (`partLogs_of_nil`).
* **`VStarAccepts` at the sparse shape** (`vstarAccepts_sparse_of_links`): `vstarAccepts_sparse` with every part's
  `PartChecks` from its links; `vstarPartsStaging_sparse_of_links` is `VStarPartsStaging` so.
-/

namespace Flock.Readable

/-! ## The checks the verifier's parser made, at every table -/

section ParseFacts

open FlockSoundness FlockSoundness.Discharge Exec

/-- **Every table the verifier parses passed `HmRow.parse`**: at an untyped statement without public input words, each
table's circuit is one `HmRow.parse` returned (on the public-output path and the hidden one), with what it accepted
(`ParseFacts`). -/
theorem parseFacts_of_tablesOf {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) {j : ℕ} (hj : j < sts.size) : ∃ tags, ParseFacts tags sts[j].c := by
  obtain ⟨spec, hset⟩ := setup_of_tablesOf hs
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt r.tags r.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs] at hd
  cases hh : r.tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    exact ⟨_, parse_facts (setupH_spec ht' hd).1⟩
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    exact ⟨_, parse_facts (Hidden.parse_untyped ht' (Hidden.setupHidden_spec' hd).parse).1⟩

end ParseFacts

/-- **Every table passed `HmRow.parse`, at `Flock.check`**: wherever `Flock.check` holds at the deployed parser, the
statement parses (`relS_of_check`), so at an untyped statement without public input words each table's circuit is
one `HmRow.parse` returned. -/
theorem parseFacts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) :
    ∃ tags, FlockSoundness.ParseFacts tags p.sts[t].c := by
  obtain ⟨p', hx', hp', -⟩ := relS_of_check h
  rw [hx] at hx'
  cases hx'
  exact parseFacts_of_tablesOf (tablesOf_of_parses hp') ht hpi htl

/-! ## The units' semantics at a satisfied table -/

section Templates

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar
open FlockVBridge.Sparse (SparseIn sparseInBits sparseBase outOf resArr shOf vBaseS)

/-- The laid-out sparse-phase unit's input rows: its ports' words, 16 bits each. -/
theorem inWords_sparseLaid {sh : Flock.RecSparse.Shape} {name : String} {net : Flock.Net}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf (sparseInBits sh) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw]
  simp only [Flock.RecSparse.unitLaid, layout_inGroups]
  simp [inRowsOf, sparseInBits]

/-- **A rec-sparse copy's outputs** at `laid_sparse`'s offsets: from `sparseBase`, `outOf`'s residuals at the message
ports' elements `wA`, `v`'s `vA` and the slots' types `τA`, 128 bits each, and with `T > 1` from bit 512 each type with
its low `T_LOG` bits cleared. -/
def SparseOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) (wA vA : Array Flock.F128) (τA : Array ℕ) : Prop :=
  (∀ r (hr : r < (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1.size), ∀ k < 128,
    v (sparseBase sh + (128 * r + k)) = FlockVBridge.bitOf (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1[r] k) ∧
  (sh.stacked = true → ∀ s < sh.S, ∀ b < 16,
    v (sparseBase sh + (512 + (16 * s + b))) = (τA[s]! - τA[s]! % 2 ^ sh.tLog).testBit b)

/-- **The sparse phase: a satisfied table's copies compute `outOf`.** Let `RecSparse.check` accept the table's circuit,
its unit named for a shape `sh` (`RecSparse.params`). In every block, at every VU and copy, if the copy's input
columns hold `wA` on the message ports, `vA` on `v`'s and with `T > 1` the slots' types `τA` on `tau`'s (`SparseIn`),
its output columns hold `outOf`'s residuals there (`SparseOuts`): `laid_sparse` at the verifier's net
(`check_sparse`), whose shape has `T_LOG ≤ 16`. -/
theorem sparseOuts_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : Flock.RecSparse.check st.c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params st.c.unit.name = some sh) (wA vA : Array Flock.F128) (τA : Array ℕ)
    (hin : SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!)
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    SparseOuts sh (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      wA vA τA := by
  obtain ⟨hv, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hc hp
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName sh) (Flock.RecSparse.unitLaid sh) = .ok st.c.unit := hn
  have hL := laidRows_of_satisfies L regs hz ho hg hu hn'
  rw [inWords_sparseLaid hn'] at hL
  simp only [Flock.RecSparse.Shape.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
  exact FlockVBridge.Sparse.laid_sparse sh hv.2 _ hL wA vA τA hin

/-- **A rec-open net under any name: a satisfied table's copies climb.** `climb_of_satisfies` for a unit the verifier
built from `RecOpen.unitLaid LANES H` under the name `name`, at `H ≤ 32`: a copy whose input columns hold a row, its
siblings, the running sums, the coefficients and the position's bits (`OpenIns`) holds the climb from the row's leaf
and the consistency term on its output columns (`OpenOuts`). -/
theorem climbAt_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {LANES H : ℕ} (hH : H ≤ 32) {name : String}
    (hl : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok st.c.unit)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) LANES H
      r a cf rest idx ss) :
    OpenOuts (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) ms LANES H
      r a cf rest idx ss := by
  obtain ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩ := hin
  have hL := laidRows_of_satisfies L regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

/-- **R's levels: a satisfied table's copies climb.** Let `RecOpen.check` accept the table's circuit, its unit named
`rec-open-stacked/l<LANES>-h<H>` (`RecOpen.stackedParams`): it is the rec-open net under that name
(`RecOpen.check_ok_stacked`), so `climbAt_of_satisfies`. -/
theorem climbStacked_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hrec : Flock.RecOpen.check st.c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.stackedParams st.c.unit.name = some (LANES, H))
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) LANES H
      r a cf rest idx ss) :
    OpenOuts (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) ms LANES H
      r a cf rest idx ss := by
  obtain ⟨hvalid, hl⟩ := Flock.RecOpen.check_ok_stacked hrec hp
  simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hvalid
  exact climbAt_of_satisfies L regs hz ho hg hu hvalid.2 hl ms hms hin

/-- **R's opening parts: a satisfied table's copies compute the part's residuals.** Let `RecOpening.check` accept the
table's circuit, its unit named for part `j` of R's opening at a shape `sh` (`RecOpening.params`, `RecOpening.shape?`),
part `j` being `S` with row ports of `ns` elements and a `v` of `nv` (`RecOpening.part`). In every block, at every VU
and copy, if the copy's input columns hold `W` on the row ports and `vv` on `v`'s (`InCols`), zero past `ns`'s total
and `nv`, its output columns hold each of S's residuals at `W`, `vv` (`PartCols`): `laid_part` at the verifier's
residuals net (`check_opening`). -/
theorem openingCols_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : RecOpening.check st.c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params st.c.unit.name = some (m, cl, j)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh j = .ok (S, ns, nv))
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    PartCols S W vv (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (opening_small hpart) _ hL W vv hw0 hv0 hin

/-! ## At Flock's relation, at the deployed parser -/

/-- **A rec-sparse part's copies compute `outOf` at `Flock.check`** (`sparseOuts_of_satisfies` at every table, with
the parser's `RecSparse.check`). -/
theorem sparseOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {sh : Flock.RecSparse.Shape} (hp : Flock.RecSparse.params p.sts[t].c.unit.name = some sh)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (wA vA : Array Flock.F128) (τA : Array ℕ)
    (hin : SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (copyOf p t htl w o g u)) :
    SparseOuts sh (copyOf p t htl w o g u) wA vA τA := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, P⟩ := parseFacts_of_check h hx ht hpi htl
  exact sparseOuts_of_satisfies L regs hz ho hg hu P.recSparse hp wA vA τA hin

/-- **A rec-open or R-level part's copies climb at `Flock.check`** (`climb_of_satisfies` or
`climbStacked_of_satisfies` at every table, with the parser's `RecOpen.check`): a table whose unit is named for
`LANES` and `H` under either name. -/
theorem climbAt_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, H))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyOf p t htl w o g u) LANES H r a cf rest idx ss) :
    OpenOuts (copyOf p t htl w o g u) ms LANES H r a cf rest idx ss := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, P⟩ := parseFacts_of_check h hx ht hpi htl
  rcases hp with hp | hp
  · obtain ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩ := hin
    exact climb_of_satisfies L regs hz ho hg hu rfl ms hms P.recNet hp r a cf rest idx ss hr ha hcf hrest hss hrow
      haux hacc hcoef hdir
  · exact climbStacked_of_satisfies L regs hz ho hg hu P.recNet hp ms hms hin

/-- **A table of residuals: the plain algebra's part `j`, or R's opening's**: by its unit's name, the circuit `c` is
part `j` of a plain shape's algebra (`RecResiduals`) or of R's opening at a shape (`RecOpening`), `S` with row ports of
`ns` elements and a `v` of `nv`. -/
def ResPart (c : Flock.Circuit) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : Prop :=
  (∃ (m k cl j : ℕ) (sh : Flock.Shape), RecResiduals.params c.unit.name = some (m, k, cl, j) ∧
    RecResiduals.shape? m k cl = some sh ∧ RecResiduals.holoOf c = .ok false ∧
    RecResiduals.part sh j = .ok (S, ns, nv)) ∨
  ∃ (m cl j : ℕ) (sh : Flock.Shape), RecOpening.params c.unit.name = some (m, cl, j) ∧
    RecOpening.shape? m cl = some sh ∧ RecOpening.part sh j = .ok (S, ns, nv)

/-- **A table of residuals' copies compute them at `Flock.check`** (`partCols_of_satisfies` or
`openingCols_of_satisfies` at every table, with the parser's `RecResiduals.check` or `RecOpening.check`). -/
theorem resCols_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : ResPart p.sts[t].c S ns nv)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyOf p t htl w o g u)) :
    PartCols S W vv (copyOf p t htl w o g u) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, P⟩ := parseFacts_of_check h hx ht hpi htl
  rcases hpart with ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩ | ⟨m, cl, j, sh, hp, hsh, hpart⟩
  · exact partCols_of_satisfies L regs hz ho hg hu P.recResiduals hp hsh hh hpart W vv hw0 hv0 hin
  · exact openingCols_of_satisfies L regs hz ho hg hu P.recOpening hp hsh hpart W vv hw0 hv0 hin

/-! ## A rec-sparse copy's inputs off its columns -/

theorem ofFn_readElems {col : ℕ → ℕ} {n : ℕ} {v : ℕ → Bool} (e : ℕ) :
    (Array.ofFn fun i : Fin n => readElems col n v i)[e]! = readElems col n v e := by
  by_cases he : e < n
  · rw [getElem!_pos _ _ (by simpa using he)]
    simp
  · rw [getElem!_neg _ _ (by simpa using he), readElems_ge (by omega)]
    rfl

/-- The message ports' elements a rec-sparse copy's columns `v` hold (`readElems` where `SparseIn` puts them). -/
def sparseW (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Array Flock.F128 :=
  Array.ofFn fun e : Fin ((Flock.RecSparse.messagePorts sh).map (·.2)).sum =>
    readElems (rowCol (Flock.RecSparse.messagePorts sh) 0) ((Flock.RecSparse.messagePorts sh).map (·.2)).sum v e

/-- The elements a rec-sparse copy's columns `v` hold on `v`'s port. -/
def sparseV (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Array Flock.F128 :=
  Array.ofFn fun e : Fin (Flock.RecSparse.nv sh) => readElems (vBaseS sh + ·) (Flock.RecSparse.nv sh) v e

/-- **A rec-sparse copy's input columns hold the elements they read**, at a shape that stacks no types (`T_LOG = 0`):
any columns `v` hold `sparseW` on the message ports and `sparseV` on `v`'s (`SparseIn`, with no types). -/
theorem sparseIn_reads (sh : Flock.RecSparse.Shape) (hst : sh.stacked = false) (v : ℕ → Bool) :
    SparseIn sh ((sparseW sh v)[·]!) ((sparseV sh v)[·]!) ((#[] : Array ℕ)[·]!) v := by
  have hW : (fun e => (sparseW sh v)[e]!) =
      readElems (rowCol (Flock.RecSparse.messagePorts sh) 0) ((Flock.RecSparse.messagePorts sh).map (·.2)).sum v :=
    funext ofFn_readElems
  have hV : (fun e => (sparseV sh v)[e]!) = readElems (vBaseS sh + ·) (Flock.RecSparse.nv sh) v :=
    funext ofFn_readElems
  refine ⟨fun q hq => ?_, fun q hq => ?_, fun h => absurd h (by rw [hst]; decide), fun e he => ?_, fun e he => ?_⟩
  · rw [hW]
    exact (wbit_readElems hq).symm
  · rw [hV]
    exact (wbit_readElems hq).symm
  · exact (ofFn_readElems e).trans (readElems_ge he)
  · exact (ofFn_readElems e).trans (readElems_ge he)

end Templates

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar

/-! ## A part's links at one V* statement -/

section Links

variable {Coin Col T : Type}

/-- **A rec-open part's links at one V\* statement** (the table's Ligerito levels or R's levels): `OpenLinks`' body at
the compiled statement `x` and the V* statement `xO`, at a unit named for `LANES` lanes and height `Ht` under
`rec-open/…` or `rec-open-stacked/…`, without the check the parser makes (`parseFacts_of_check`), its output tie giving
the goal `G` at the witness and the agreeing messages. -/
def ClimbAt (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
      (_root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∨
        _root_.Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, Ht)) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
          ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
            (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
            Flock.Readable.OpenIns (Flock.Readable.copyOf p t ht w o g u) LANES Ht r a cf rest idx ss ∧
            (Flock.Readable.OpenOuts (Flock.Readable.copyOf p t ht w o g u) msch LANES Ht r a cf rest idx ss →
              G w ms)

/-- **A residuals part's links at one V\* statement** (the rep's algebra or R's opening parts): `ResLinks`' body at
`x` and `xO`, at a table of residuals (`ResPart`), without the check the parser makes, its output tie giving the goal
`G`. -/
def ResAt (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      Flock.Readable.ResPart p.sts[t].c S ns nv ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
          ∃ W vv : ℕ → _root_.Flock.F128, (∀ e, ns.sum ≤ e → W e = 0) ∧ (∀ e, nv ≤ e → vv e = 0) ∧
            InCols (partRep ns nv) W vv (Flock.Readable.copyOf p t ht w o g u) ∧
            (PartCols S W vv (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → G w ms)

/-- **The sparse phase's links at one V\* statement** (a code fact about V*'s staging, the layout and the goal at part
`s`, none of it the circuit's semantics): at `x` and `xO`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words one of whose tables has a
  unit named for a shape `sh` (`RecSparse.params`);
* the input tie: the statement names a copy of that table's unit (block `o`, VU `g`, copy `u`) whose input columns, on
  every witness and every list of inner messages that agrees with the messages the part holds, hold the message
  ports' elements `wA`, `v`'s `vA` and with `T > 1` the slots' types `τA` (`SparseIn`);
* the output tie: `outOf`'s residuals there on that copy's output columns (`SparseOuts`) give the goal `G` at the
  witness and the messages. -/
def SparseAt (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ sh : _root_.Flock.RecSparse.Shape,
      _root_.Flock.RecSparse.params p.sts[t].c.unit.name = some sh ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
          ∃ (wA vA : Array _root_.Flock.F128) (τA : Array ℕ),
            FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (Flock.Readable.copyOf p t ht w o g u) ∧
            (Flock.Readable.SparseOuts sh (Flock.Readable.copyOf p t ht w o g u) wA vA τA → G w ms)

/-- **A part's links at one V\* statement**: the part is a table of one of the three units there, with its links
(`ClimbAt`, `ResAt`, `SparseAt`), each output tie giving the goal `G`. -/
def PartAt (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ClimbAt decode L s x xO G ∨ ResAt decode L s x xO G ∨ SparseAt decode L s x xO G

/-- **`PartChecks`' and `PartLogs`' fallback at part `s`**: two of the part's recorded calls collide. -/
def PartColl (L : Layout) (s : ℕ) (x : Msg) (w : Flock.Wit) : Prop := (findColl (L.calls x s w)).isSome = true

/-- **Part `s`'s links toward a goal** (a code fact about V*'s staging and the layout at part `s`): wherever a V*
statement `xO` of a compiled statement `x` of the shape shows the coins `c` for part `s`, the part's links hold there
(`PartAt`), as a table of whichever unit V* stages it as at `x`, each output tie giving the goal `G x c` at the witness
and the messages, or the fallback (`PartColl`). -/
def PartLinksTo (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop) (s : ℕ) :
    Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    PartAt decode L s x xO fun w ms => G x c w ms ∨ PartColl L s x w

/-- **Part `s`'s links** toward its inner check `chk s` (`PartLinksTo`). -/
def PartLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  PartLinksTo D decode L shape shows (fun x c _ ms => chk s x c ms) s

/-- **Part `s`'s log links** toward its inner check's calls (`logs s`) being among the calls it records at the witness
(`PartLinksTo`). At a rec-open part these are the climb's evaluations. -/
def LogLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartLinksTo D decode L shape shows (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s

/-- **The goal from a part's links at one V\* statement**: at the deployed parser, wherever the part's outer relation
holds at `xO`, its links there (`PartAt`) give their goal on every list of inner messages that agrees with what it
holds: the named copy computes its unit's outputs at its inputs (`climbAt_of_check`, `resCols_of_check`,
`sparseOuts_of_check`), which the output tie reads. -/
theorem goal_of_partAt {DO : Flock.Deploy Coin Col T} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode) {L : Layout} {s : ℕ} {x xO : Msg}
    {w : Flock.Wit} {G : Flock.Wit → List Msg → Prop} (hck : Flock.check DO.parse xO w = true)
    (hA : PartAt decode L s x xO G) {ms : List Msg}
    (hms : ∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) : G w ms := by
  rw [hDO] at hck
  rcases hA with hA | hA | hA
  · obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hp, o, g, u, ho, hg, hu, hlink⟩ := hA
    obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, hout⟩ := hlink w ms hms
    exact hout (Flock.Readable.climbAt_of_check hck hx ht hpi htl hp ho hg hu msch hmsch hin)
  · obtain ⟨p, hx, ht, hpi, t, htl, S, ns, nv, hpart, o, g, u, ho, hg, hu, hlink⟩ := hA
    obtain ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms hms
    exact hout (Flock.Readable.resCols_of_check hck hx ht hpi htl hpart ho hg hu W vv hw0 hv0 hin)
  · obtain ⟨p, hx, ht, hpi, t, htl, sh, hp, o, g, u, ho, hg, hu, hlink⟩ := hA
    obtain ⟨wA, vA, τA, hin, hout⟩ := hlink w ms hms
    exact hout (Flock.Readable.sparseOuts_of_check hck hx ht hpi htl hp ho hg hu wA vA τA hin)

/-- **The goal from a part's links, or the fallback**: at the deployed parser, wherever part `s`'s outer relation holds
at a V* statement `xO` of a compiled statement `x` of the shape, at a witness whose recorded calls don't collide its
links' goal holds at every coins `xO` shows and every agreeing list of messages (`goal_of_partAt`); at one whose calls
do, the fallback holds. -/
theorem goal_of_links {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {decode : Msg → Option Flock.Readable.Read}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop} {s : ℕ}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartLinksTo D decode L shape shows G s) (x : Msg) (st : Stmt) (hst : D.parse x = some st)
    (hf : shape st.form) (xO : Msg) (w : Flock.Wit) (hck : Flock.check DO.parse xO w = true) :
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) → G x c w ms) ∨ PartColl L s x w := by
  by_cases hfb : PartColl L s x w
  · exact Or.inr hfb
  · exact Or.inl fun c hc ms hms => (goal_of_partAt hDO hck (h x st hst hf xO c hc) hms).resolve_right hfb

/-- **`PartChecks` from a part's links**: at the deployed parser (`Readable.parseOf relation decode`), part `s`'s
links (`PartLinks`) give `PartChecks s` (`goal_of_links`). -/
theorem partChecks_of_links {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s :=
  fun x st hst hf xO w hck _ => goal_of_links hDO h x st hst hf xO w hck

/-- **`PartLogs` from a part's log links**: at the deployed parser, part `s`'s log links (`LogLinks`) give
`PartLogs s` (`goal_of_links`). -/
theorem partLogs_of_links {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : LogLinks D decode L shape shows logs s) : PartLogs H D DO L shape shows logs s :=
  fun x st hst hf xO w hck _ => goal_of_links hDO h x st hst hf xO w hck

/-! ## The dense shape's links, and the input tie off the columns -/

/-- **An algebra part's links are its links at every V\* statement** (`ResLinks` gives `PartLinks`). -/
theorem partLinks_of_resLinks {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (h : ResLinks D decode L shape shows chk s) :
    PartLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, -, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    h x st hst hf xO c hc
  exact Or.inr (Or.inl ⟨p, hx, ht, hpi, t, htl, S, ns, nv, Or.inl ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩, o, g, u, ho,
    hg, hu, fun w ms hms =>
      let ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms hms
      ⟨W, vv, hw0, hv0, hin, fun h => Or.inl (hout h)⟩⟩)

/-- **A rec-open part's links are its links at every V\* statement** (`OpenLinks` gives `PartLinks`). -/
theorem partLinks_of_openLinks {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (h : OpenLinks D decode L shape shows chk s) :
    PartLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, -, hp, o, g, u, ho, hg, hu, hlink⟩ := h x st hst hf xO c hc
  exact Or.inl ⟨p, hx, ht, hpi, t, htl, LANES, Ht, Or.inl hp, o, g, u, ho, hg, hu, fun w ms hms =>
    let ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, hout⟩ := hlink w ms hms
    ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, fun h => Or.inl (hout h)⟩⟩

/-- **A residuals part's links, read off the copy's columns**: `ResAt` with the input tie's elements those the named
copy's columns hold (`rowsRead`, `vRead`; `inCols_readElems`). What is left of the input tie sits in the output tie's
premise. -/
theorem resAt_of_reads {decode : Msg → Option Flock.Readable.Read} {L : Layout} {s : ℕ} {x xO : Msg}
    {G : Flock.Wit → List Msg → Prop} {p : Flock.Readable.Parsed} (hx : Flock.Readable.parseBytes decode xO = some p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ}
    {nv : ℕ} (hpart : Flock.Readable.ResPart p.sts[t].c S ns nv) {o g u : ℕ}
    (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (hout : ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
      PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
        (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
        (Flock.Readable.copyOf p t htl w o g u) (partBase S ns nv) → G w ms) :
    ResAt decode L s x xO G :=
  ⟨p, hx, ht, hpi, t, htl, S, ns, nv, hpart, o, g, u, ho, hg, hu, fun w ms hms =>
    ⟨Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u),
      Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u),
      fun _ he => Flock.Readable.readElems_ge he, fun _ he => Flock.Readable.readElems_ge he,
      Flock.Readable.inCols_readElems ns nv _, hout w ms hms⟩⟩

/-- **The sparse phase's links, read off the copy's columns**, at a shape that stacks no types (`T_LOG = 0`, as in
scope): `SparseAt` with the input tie's elements those the named copy's columns hold (`sparseW`, `sparseV`;
`sparseIn_reads`). What is left of the input tie sits in the output tie's premise. -/
theorem sparseAt_of_reads {decode : Msg → Option Flock.Readable.Read} {L : Layout} {s : ℕ} {x xO : Msg}
    {G : Flock.Wit → List Msg → Prop} {p : Flock.Readable.Parsed} (hx : Flock.Readable.parseBytes decode xO = some p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) {sh : _root_.Flock.RecSparse.Shape}
    (hp : _root_.Flock.RecSparse.params p.sts[t].c.unit.name = some sh) (hst : sh.stacked = false) {o g u : ℕ}
    (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (hout : ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
      Flock.Readable.SparseOuts sh (Flock.Readable.copyOf p t htl w o g u)
        (Flock.Readable.sparseW sh (Flock.Readable.copyOf p t htl w o g u))
        (Flock.Readable.sparseV sh (Flock.Readable.copyOf p t htl w o g u)) #[] → G w ms) :
    SparseAt decode L s x xO G :=
  ⟨p, hx, ht, hpi, t, htl, sh, hp, o, g, u, ho, hg, hu, fun w ms hms =>
    ⟨_, _, _, Flock.Readable.sparseIn_reads sh hst _, hout w ms hms⟩⟩

end Links

/-! ## At the sparse shape -/

section Sparse

variable {Coin W Col T : Type} [Inhabited Coin]

/-- **`VStarAccepts` at the sparse shape, from the parts' links**: `vstarAccepts_sparse` with every part's `PartChecks`
from its links (`partChecks_of_links`), at the deployed parser. -/
theorem vstarAccepts_sparse_of_links {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartLinks D decode L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens D DO L Form.Sparse opening cmt s) (hS : Staged H D I L Form.Sparse shows cmt)
    (hD : Decomp H D IV Form.Sparse chk) : VStarAccepts H D I DO Form.Sparse :=
  vstarAccepts_sparse hR (fun s => partChecks_of_links hDO (hC s)) hO hS hD

/-- **V\*'s parts at the sparse shape, from the parts' links**: `VStarPartsStaging` at `Form.Sparse` with every part's
`PartChecks` from its links. -/
theorem vstarPartsStaging_sparse_of_links {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hC : ∀ s, PartLinks D decode L Form.Sparse shows chk s) (hO : ∀ s, PartOpens D DO L Form.Sparse opening cmt s)
    (hL : ∀ s, PartLogs H D DO L Form.Sparse shows logs s) (hV : VstarShows D I L Form.Sparse decC shows cmt)
    (hD : Decomp H D I.V Form.Sparse chk) (hLD : LogDecomp H D I.V Form.Sparse logs) (hH : Held D I L Form.Sparse) :
    VStarPartsStaging D I DO H L opening decC Form.Sparse :=
  ⟨shows, chk, cmt, logs, fun s => partChecks_of_links hDO (hC s), hO, hL, hV, hD, hLD, hH⟩

end Sparse

end Claims.Recursion.VStar
