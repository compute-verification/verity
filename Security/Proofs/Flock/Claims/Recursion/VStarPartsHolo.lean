import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.Claims.Recursion.DeployReg
import Proofs.Flock.VBridge.OpeningUnit
import Proofs.Flock.VBridge.Sparse.Laid
import Proofs.Flock.VBridge.Sparse.Unit

/-!
# V*'s parts at the holo shape

At a holo inner shape (`Form.Holo`: the inner circuits private, `i.holo`) V*'s statements (`RecVStar.statements i`) are
`VStar.statementsHolo`'s, then candidate 3's (`candidate3Statements i`, empty unless the shape names a sparse phase):

* the levels' openings, `recOpen l (2^lanes) (d − cap) q`, as at the dense shape;
* the algebra's parts, `inner j name part` at `Algebra.specHolo`: h's round sits after z_partial and the comb operands
  read it, and each part's circuit file carries META `holo` (`circuitFile`'s `holo && s matches .inner ..`), so
  `RecResiduals.check` builds its net at `{ sh with holo := true }`;
* with the sparse phase: the RecSparse unit (`sparse sh hRound`), R's levels (`recOpenStacked`, units named
  `rec-open-stacked/l<L>-h<H>`) and R's opening's parts (`opening m c j S ports nv`, `rec-opening/m<m>-c<c>-p<j>`).

Every one is a unit the verifier rebuilds itself, so a copy's semantics at `Flock.check` follows M's method
(`VStarParts.lean`): `layout_of_check`, `laidRows_of_satisfies` at the net the check accepted, then the template's own
lemma. Which template each part uses:

| holo part | net the check accepts | check lemma | semantics | as at dense? |
|---|---|---|---|---|
| levels' `recOpen` | `RecOpen.unitLaid L H` | `RecOpen.check_ok` | `laid_unit` | yes (`climb_of_check`) |
| algebra parts | `RecResiduals.unitLaid S ns nv` | `check_part_at`, `holo` true | `laid_part` | net yes, flag new |
| R's levels | `RecOpen.unitLaid L H` | `RecOpen.check_ok_stacked` | `laid_unit` | net yes, name new |
| R's opening parts | `RecResiduals.unitLaid S ns nv` | `check_opening` | `laid_part` | net yes, check new |
| RecSparse | `RecSparse.unitLaid sh` | `check_sparse` | `laid_sparse` | new |

* **Templates** (`Flock.Readable`): `partCols_of_check_at` (an algebra part at either holo flag, so the holo parts'
  `h` round needs nothing beyond the flag: `laid_part` holds for any `S`, `ns`, `nv`), `openOuts_of_net` and
  `openOuts_of_check_any` (a level's climb under either name), `partCols_of_check_opening` (R's opening parts), and
  `sparseCopyOuts_of_check` (the RecSparse unit, with `inWords_sparse`).
* **Links** (`Claims.Recursion.VStar`): what is not the circuit's semantics, per kind of part, at one compiled
  statement, V* statement, coins and the part's witness: `LevelAt` (either level kind, M's `OpenLinks` body),
  `ResReadsAt` (an algebra part at either flag, M's `ResReads` body), `OpeningReadsAt`, `SparseCopyAt`; `PartKindAt` is
  their disjunction. `PartsLinks s` asks it wherever part `s`'s outer relation holds with true recorded calls and its
  coins are shown; `PartsLinksOr s` allows `PartChecks`' fallback instead, a collision among the part's recorded calls
  (`findColl`) or a tie with the setup's out-of-oracle evaluations at `xO` (`SetupColl`, which `Br` finds: card
  f93bbff1). `partChecks_of_partsLinks` and `partChecks_of_partsLinksOr` give `PartChecks s` at any shape;
  `partChecks_holo` at every part of the holo shape. M's dense links are instances (`partsLinks_of_reads`,
  `partsLinks_of_open`).
* **`PartLogs`** (`partLogs_of_shown_nil`, `partLogs_of_logLinks`, `partLogs_holo`): the algebra, opening and sparse
  parts' checks log no call, so `logs s = []` wherever their coins are shown (`LogsNil`); a level's check logs the
  climb's calls, which its links for its calls put among the part's recorded calls (`LevelLogsAt`, `PartsLogLinks`,
  either level name; M's `OpenLogLinks` is an instance, `partsLogLinks_of_open`).
* **`PartOpens`** (`partOpens_holo`): M's `partOpens_of_climbs`, which holds at any part of any shape, so the holo
  parts need nothing of their own: each part opens what it holds to the digest its public file names (`HeldOpens`),
  and the setup climbs that digest and the round's commit string to one registered root (`RegClimbs`); two such
  climbs from different values collide, at `Br`'s setup evaluations or among the recorded calls.
* **At the holo shape**: `vstarAccepts_holo_of_links` (`VStarAcceptsR`), and the bundles `vstarParts_holo_of_links` and
  `vstarPartsStaging_holo_of_links`, what `codeFacts_of_parts`, `ksA_deployed_parts` and `KS_of_parts` read, with each
  part a holo part (`HoloPart`, the counterpart of M's `DensePart`: `PartsLinksOr`, and `LogsNil` or `PartsLogLinks`;
  `holoPart_of_open` from M's level links, `holoPart_of_nil` at a part without calls). At
  `deployRec` over the outer record `deployZkRec relation` (`vstarAccepts_holo_deployRec`, `vstarParts_holo_deployRec`,
  `vstarPartsStaging_holo_deployRec`), the outer record parses at `parseOf relation readOfO` by definition, and
  `deployRec` at the layout's readers reads through them (`RecDeploy.readsThrough_deployRec`).
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

/-! ## The algebra parts at either holo flag -/

/-- **An algebra part's copies compute its residuals, at the statement's holo flag**: `partCols_of_satisfies` with the
flag the circuit's META gives (`holoOf`), the net `RecResiduals.check` built at `{ sh with holo }` (`check_part_at`).
At `holo = true` it is the holo shape's algebra parts, whose `h` round is one of the part's row ports like any other. -/
theorem partCols_of_satisfies_at {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : RecResiduals.check st.c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params st.c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf st.c = .ok holo)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part { sh with holo } j = .ok (S, ns, nv))
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    PartCols S W vv (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part_at hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (part_small hpart) _ hL W vv hw0 hv0 hin

/-- **An algebra part's copies compute its residuals at `Flock.check`, at either holo flag**
(`partCols_of_satisfies_at` at every table). -/
theorem partCols_of_check_at {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecResiduals.check p.sts[t].c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf p.sts[t].c = .ok holo)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part { sh with holo } j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyOf p t htl w o g u)) :
    PartCols S W vv (copyOf p t htl w o g u) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact partCols_of_satisfies_at L (flat_of_check h hx ht hpi htl) regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

/-! ## The levels, under either name -/

/-- **A level's copies climb, at any name the verifier built its net under**: `climb_of_satisfies` with the net taken
as given, so R's levels (`rec-open-stacked/…`) are the levels' (`rec-open/…`) template under another name. -/
theorem openOuts_of_net {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {LANES H : ℕ} {name : String}
    (hvalid : Flock.RecOpen.valid LANES H = true)
    (hl : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok st.c.unit)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) LANES H r
      a cf rest idx ss) :
    OpenOuts (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) ms LANES H r a
      cf rest idx ss := by
  obtain ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩ := hin
  have hH : H ≤ 32 := by
    simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hvalid
    exact hvalid.2
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

/-- **What `RecOpen.check` accepted, under either name**: a unit named `rec-open/l<L>-h<H>` (`params`, the levels) or
`rec-open-stacked/l<L>-h<H>` (`stackedParams`, R's levels) has a valid `L` and `H`, and is the net the verifier builds
at them under that name. -/
theorem recOpen_check_ok_any {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams c.unit.name = some (LANES, H)) :
    ∃ name, Flock.RecOpen.valid LANES H = true ∧
      Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok c.unit := by
  rcases hp with hp | hp
  · exact ⟨_, Flock.RecOpen.check_ok h hp⟩
  · exact ⟨_, Flock.RecOpen.check_ok_stacked h hp⟩

/-- **A level's copies climb at `Flock.check`, under either name** (`openOuts_of_net` at every table, the net from
`recOpen_check_ok_any`). -/
theorem openOuts_of_check_any {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (hrec : Flock.RecOpen.check p.sts[t].c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H) ∨
      Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, H))
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {r a cf : Array Flock.F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (hin : OpenIns (copyOf p t htl w o g u) LANES H r a cf rest idx ss) :
    OpenOuts (copyOf p t htl w o g u) ms LANES H r a cf rest idx ss := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨_, hvalid, hl⟩ := recOpen_check_ok_any hrec hp
  exact openOuts_of_net L (flat_of_check h hx ht hpi htl) regs hz ho hg hu hvalid hl ms hms hin

/-! ## R's opening parts -/

/-- **R's opening parts' copies compute their residuals at `Flock.check`**: where `RecOpening.check` accepted the
table's circuit as part `j` of R's opening at a shape it accepts, the net is `RecResiduals.unitLaid` at that part
(`check_opening`), the algebra parts' template, so `laid_part` holds at it (`opening_small`). -/
theorem partCols_of_check_opening {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecOpening.check p.sts[t].c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params p.sts[t].c.unit.name = some (m, cl, j)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyOf p t htl w o g u)) :
    PartCols S W vv (copyOf p t htl w o g u) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_opening hc hp hsh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L (flat_of_check h hx ht hpi htl) regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (opening_small hpart) _ hL W vv hw0 hv0 hin

/-! ## The RecSparse unit -/

/-- The verifier's RecSparse net's input words are its ports' (`sparseInBits`), under any name. -/
theorem inWords_sparse {name : String} {sh : Flock.RecSparse.Shape} {net : Flock.Net}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf (FlockVBridge.Sparse.sparseInBits sh) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw]
  simp only [Flock.RecSparse.unitLaid, layout_inGroups]
  simp [inRowsOf, FlockVBridge.Sparse.sparseInBits]

/-- **A RecSparse copy's outputs** (`laid_sparse`'s conclusion): from `sparseBase`, `outOf`'s residuals at `Defs`'
residuals, 128 bits each, and with `T > 1` from bit 512 each slot's type with its low `T_LOG` bits cleared. -/
def SparseCopyOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) (wA vA : Array Flock.F128) (τA : Array ℕ) : Prop :=
  (∀ r (hr : r < (FlockVBridge.Sparse.outOf (FlockVBridge.Sparse.shOf sh)
      (FlockVBridge.Sparse.resArr (FlockVBridge.Sparse.shOf sh) wA vA) wA vA τA).1.size), ∀ k < 128,
    v (FlockVBridge.Sparse.sparseBase sh + (128 * r + k)) =
      FlockVBridge.bitOf (FlockVBridge.Sparse.outOf (FlockVBridge.Sparse.shOf sh)
        (FlockVBridge.Sparse.resArr (FlockVBridge.Sparse.shOf sh) wA vA) wA vA τA).1[r] k) ∧
  (sh.stacked = true → ∀ s < sh.S, ∀ b < 16,
    v (FlockVBridge.Sparse.sparseBase sh + (512 + (16 * s + b))) = (τA[s]! - τA[s]! % 2 ^ sh.tLog).testBit b)

/-- **The RecSparse unit's copies compute `outOf` at `Flock.check`**: where `RecSparse.check` accepted the table's
circuit at the shape its unit's name gives, the net is `RecSparse.unitLaid sh` (`check_sparse`) at a valid shape
(`T_LOG ≤ 16`), so a copy whose input columns hold the messages, `v` and the types (`SparseIn`) holds `laid_sparse`'s
outputs. -/
theorem sparseCopyOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : Flock.RecSparse.check p.sts[t].c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params p.sts[t].c.unit.name = some sh)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (wA vA : Array Flock.F128) (τA : Array ℕ)
    (hin : FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (copyOf p t htl w o g u)) :
    SparseCopyOuts sh (copyOf p t htl w o g u) wA vA τA := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨hv, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hc hp
  have hL := laidRows_of_satisfies L (flat_of_check h hx ht hpi htl) regs hz ho hg hu (l := Flock.RecSparse.unitLaid sh) hn
  rw [inWords_sparse hn] at hL
  have ht16 : sh.tLog ≤ 16 := by
    simp only [Flock.RecSparse.Shape.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    exact hv.2
  exact FlockVBridge.Sparse.laid_sparse sh ht16 _ hL wA vA τA hin

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar

section Facts

variable {Coin W Col T : Type}

/-- A list of inner messages agrees with the messages part `s` holds off its witness `w`. -/
def Agrees (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) (ms : List Msg) : Prop :=
  ∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m

/-! ## Each kind of part's links, at one V* statement -/

/-- **A level's links** at the compiled statement `x`, part `s`, V* statement `xO`, coins `c` and the part's witness
`w` (M's `OpenLinks` body at `w`, under either level name): `xO` decodes to an untyped statement without public input
words one of whose tables `RecOpen.check` accepts as a level (`rec-open/…`) or as one of R's (`rec-open-stacked/…`); a
copy of its unit whose input columns hold a row, siblings, sums, coefficients and position (`OpenIns`) on every list of
messages agreeing with `w`; and the climb and consistency term on its outputs (`OpenOuts`) give the inner check. -/
def LevelAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
      _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
      (_root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∨
        _root_.Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, Ht)) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
            (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
            Flock.Readable.OpenIns (Flock.Readable.copyOf p t ht w o g u) LANES Ht r a cf rest idx ss ∧
            (Flock.Readable.OpenOuts (Flock.Readable.copyOf p t ht w o g u) msch LANES Ht r a cf rest idx ss →
              chk s x c ms)

/-- **An algebra part's links, at the statement's holo flag** (M's `ResReads` body with the flag `holoOf` gives):
`xO` decodes to an untyped statement without public input words one of whose tables `RecResiduals.check` accepts as
part `j` of a shape's algebra at holo flag `holo`, `S` with row ports of `ns` elements and a `v` of `nv`; and S's
residuals at the elements a copy of its unit reads off `w` (`rowsRead`, `vRead`) give the inner check on every list of
messages agreeing with `w`. At the holo shape `holo = true`. -/
def ResReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (holo : Bool)
      (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
      _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo ∧
      _root_.Flock.RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

/-- **An R opening part's links**: `ResReadsAt` with `RecOpening.check` accepting the table as part `j` of R's
opening at `(m, cl)`. -/
def OpeningReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
      (nv : ℕ),
      _root_.Flock.RecOpening.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecOpening.params p.sts[t].c.unit.name = some (m, cl, j) ∧
      _root_.Flock.RecOpening.shape? m cl = some sh ∧ _root_.Flock.RecOpening.part sh j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

/-- **The RecSparse part's links**: `xO` decodes to an untyped statement without public input words one of whose
tables `RecSparse.check` accepts at the shape `sh` its unit's name gives; a copy of its unit whose input columns hold
messages `wA`, `v` elements `vA` and types `τA` (`SparseIn`) off `w`, on every list of messages agreeing with `w`; and
its outputs there (`SparseCopyOuts`) give the inner check. -/
def SparseCopyAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ sh : _root_.Flock.RecSparse.Shape,
      _root_.Flock.RecSparse.check p.sts[t].c = .ok () ∧ _root_.Flock.RecSparse.params p.sts[t].c.unit.name = some sh ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ ms : List Msg, Agrees L x s w ms →
          ∃ (wA vA : Array _root_.Flock.F128) (τA : Array ℕ),
            FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (Flock.Readable.copyOf p t ht w o g u) ∧
            (Flock.Readable.SparseCopyOuts sh (Flock.Readable.copyOf p t ht w o g u) wA vA τA → chk s x c ms)

/-- **Some kind of part's links hold** at `x`, `s`, `xO`, `c`, `w`: a level's (either name), an algebra part's (either
flag), an R opening part's or the RecSparse part's. -/
def PartKindAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) (w : Flock.Wit) : Prop :=
  LevelAt decode L chk x s xO c w ∨ ResReadsAt decode L chk x s xO c w ∨ OpeningReadsAt decode L chk x s xO c w ∨
    SparseCopyAt decode L chk x s xO c w

/-- **Part `s`'s links** (a code fact about V*'s staging, the layout and the inner check at part `s`, none of it the
circuit's semantics): wherever the part's outer relation holds at a V* statement `xO` of a compiled statement of the
shape, with true recorded calls, and `xO` shows the coins `c` for part `s`, some kind of part's links hold there at the
part's witness (`PartKindAt`). -/
def PartsLinks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) → ∀ c : ℕ → Coin, shows s x xO c →
      PartKindAt decode L chk x s xO c w

/-- **Part `s`'s links, or a collision the finders charge** (`PartsLinks` with `PartChecks`' fallback): at each
accepted witness with true recorded calls, either some kind of part's links hold at every coins `xO` shows for part
`s`, or two of the part's recorded calls collide (`findColl`), or one of them ties with an evaluation the setup makes at
`xO` outside the oracle (`SetupColl`, which `Br` finds at the run: card f93bbff1). The registration ties (`v`, the zero
output rows, the levels' `coef`, `dirs` and top) are where a link can fail off a recorded collision. -/
def PartsLinksOr (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (∀ c : ℕ → Coin, shows s x xO c → PartKindAt decode L chk x s xO c w) ∨ (findColl (L.calls x s w)).isSome ∨
        SetupColl H D xO (L.calls x s w)

/-! ## `PartChecks` from the links -/

/-- **One kind of part's links and the outer relation give the inner check**: at the deployed parser, wherever
`Flock.check` holds at `xO` and `w`, the links at `xO` give the inner check on every agreeing list of messages, by the
kind's template (`climb_of_check`'s `openOuts_of_check_any`, `partCols_of_check_at`, `partCols_of_check_opening`,
`sparseCopyOuts_of_check`). -/
theorem chk_of_kindAt {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {x : Msg} {s : ℕ} {xO : Msg} {c : ℕ → Coin} {w : Flock.Wit}
    (hck : Flock.check (Flock.Readable.parseOf Flock.Readable.relation decode) xO w = true)
    (hK : PartKindAt decode L chk x s xO c w) {ms : List Msg} (hms : Agrees L x s w ms) : chk s x c ms := by
  rcases hK with hK | hK | hK | hK
  · obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, hout⟩ := hlink ms hms
    exact hout (Flock.Readable.openOuts_of_check_any hck hx ht hpi htl ho hg hu hrec hp msch hmsch hin)
  · obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, holo, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
      hlink⟩ := hK
    exact hlink ms hms (Flock.Readable.partCols_of_check_at hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, m, cl, j, sh, S, ns, nv, hrc, hp, hsh, hpart, o, g, u, ho, hg, hu, hlink⟩ := hK
    exact hlink ms hms (Flock.Readable.partCols_of_check_opening hck hx ht hpi htl hrc hp hsh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, sh, hrc, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨wA, vA, τA, hin, hout⟩ := hlink ms hms
    exact hout (Flock.Readable.sparseCopyOuts_of_check hck hx ht hpi htl hrc hp ho hg hu wA vA τA hin)

/-- **`PartChecks` at any part, from its links**: at the deployed parser (`Readable.parseOf relation decode`), part
`s`'s links (`PartsLinks`) give `PartChecks s` at any shape, its inner check holding on every agreeing list of messages
with no appeal to a collision (`chk_of_kindAt`). -/
theorem partChecks_of_partsLinks {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLinks H D DO decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck htrue
  have hK := hR x st hst hf xO w hck htrue
  rw [hDO] at hck
  exact Or.inl fun c hc ms hms => chk_of_kindAt hck (hK c hc) hms

/-- **`PartChecks` at any part, from its links or a collision the finders charge** (`PartsLinksOr`): off the
collision the links give the inner check (`chk_of_kindAt`), and the collision is `PartChecks`' own fallback. -/
theorem partChecks_of_partsLinksOr {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLinksOr H D DO decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck htrue
  rcases hR x st hst hf xO w hck htrue with hK | hK
  · rw [hDO] at hck
    exact Or.inl fun c hc ms hms => chk_of_kindAt hck (hK c hc) hms
  · exact Or.inr hK

/-- Part links are part links or a collision. -/
theorem partsLinksOr_of_links {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ}
    (h : PartsLinks H D DO decode L shape shows chk s) : PartsLinksOr H D DO decode L shape shows chk s :=
  fun x st hst hf xO w hck htrue => Or.inl (h x st hst hf xO w hck htrue)

/-- **M's algebra links are part links** (`ResReads`, the plain flag): a shape `RecResiduals.shape?` gives is plain
(`shape?_holo`), so its part at `holo := false` is its own. -/
theorem partsLinks_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReads D decode L shape shows chk s) :
    PartsLinks H D DO decode L shape shows chk s := by
  intro x st hst hf xO w _ _ c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  have e : { sh with holo := false } = sh := by
    have h0 := shape?_holo hsh
    cases sh
    simp only at h0
    subst h0
    rfl
  have hpart' : _root_.Flock.RecResiduals.part { sh with holo := false } j = .ok (S, ns, nv) := by
    rw [e]
    exact hpart
  exact Or.inr (Or.inl ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, false, S, ns, nv, hrc, hp, hsh, hh, hpart', o, g, u,
    ho, hg, hu, hlink w⟩)

/-- **M's level links are part links** (`OpenLinks`, a unit named `rec-open/…`). -/
theorem partsLinks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : OpenLinks D decode L shape shows chk s) :
    PartsLinks H D DO decode L shape shows chk s := by
  intro x st hst hf xO w _ _ c hc
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  exact Or.inl ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, Or.inl hp, o, g, u, ho, hg, hu, hlink w⟩

/-- **`PartChecks` at every part of the holo shape**: `partChecks_of_partsLinksOr` at `Form.Holo`. -/
theorem partChecks_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ∀ s, PartsLinksOr H D DO decode L Form.Holo shows chk s) : ∀ s, PartChecks H D DO L Form.Holo shows chk s :=
  fun s => partChecks_of_partsLinksOr hDO (hR s)

/-! ## `PartLogs` -/

/-- **Part `s`'s inner check logs no call where its coins are shown** (a code fact about the inner check's calls at
part `s`): at the compiled statements of the shape and the V* statements that show part `s`'s coins, `logs s` is empty
on every list of messages. -/
def LogsNil (D : Deploy Coin) (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin),
    shows s x xO c → ∀ ms, logs s x c ms = []

/-- **`PartLogs` at a part whose inner check logs no call where its coins are shown** (`LogsNil`). -/
theorem partLogs_of_shown_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} (h : LogsNil D shape shows logs s) :
    PartLogs H D DO L shape shows logs s := fun x st hst hf xO _ _ _ => Or.inl fun c hc ms _ e he => by
  rw [h x st hst hf xO c hc ms] at he
  cases he

/-- **A level's links for its calls** at `x`, `s`, `xO`, `c` and the part's witness `w`: a level's links (`LevelAt`,
either name) with, for the inner check, that every call part `s`'s check logs at `c` (`logs s`, the climb's SHA-512
evaluations) is among the calls the part records at `w`. -/
def LevelLogsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin)
    (w : Flock.Wit) : Prop :=
  LevelAt decode L (fun s x c ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) x s xO c w

/-- **A level part's links for its calls, or a collision the finders charge** (a code fact about the level's staging,
the layout and its climb's SHA-512 gadgets): wherever the part's outer relation holds at a V* statement `xO` of a
compiled statement of the shape, with true recorded calls, a level's links for its calls hold at every coins `xO`
shows for part `s` (`LevelLogsAt`), or two recorded calls collide, or one ties with the setup at `xO`. -/
def PartsLogLinks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (∀ c : ℕ → Coin, shows s x xO c → LevelLogsAt decode L logs x s xO c w) ∨ (findColl (L.calls x s w)).isSome ∨
        SetupColl H D xO (L.calls x s w)

/-- **`PartLogs` at a level part, from the shipped circuit's semantics**: at the deployed parser, a level's links for
its calls (`PartsLogLinks`) give `PartLogs s`: the outer relation makes the copy climb under either name
(`openOuts_of_check_any`, through `chk_of_kindAt`), which the output tie reads as the check's calls recorded. -/
theorem partLogs_of_logLinks {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLogLinks H D DO decode L shape shows logs s) : PartLogs H D DO L shape shows logs s := by
  intro x st hst hf xO w hck htrue
  rcases hR x st hst hf xO w hck htrue with hK | hK
  · rw [hDO] at hck
    exact Or.inl fun c hc ms hms =>
      chk_of_kindAt (chk := fun s x c ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) hck (Or.inl (hK c hc)) hms
  · exact Or.inr hK

/-- **M's level links for calls are level log links** (`OpenLogLinks`, a unit named `rec-open/…`). -/
theorem partsLogLinks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (hR : OpenLogLinks D decode L shape shows logs s) : PartsLogLinks H D DO decode L shape shows logs s := by
  intro x st hst hf xO w _ _
  refine Or.inl fun c hc => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  exact ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, Or.inl hp, o, g, u, ho, hg, hu, hlink w⟩

/-- **`PartLogs` at every part of the holo shape**: each part either logs no call where its coins are shown (the
algebra parts, R's opening parts and the RecSparse part: residuals only, `LogsNil`), or is a level, under either
name, whose links for its calls hold (`PartsLogLinks`). -/
theorem partLogs_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : ∀ s, LogsNil D Form.Holo shows logs s ∨ PartsLogLinks H D DO decode L Form.Holo shows logs s) :
    ∀ s, PartLogs H D DO L Form.Holo shows logs s := fun s =>
  (h s).elim partLogs_of_shown_nil (partLogs_of_logLinks hDO)

/-! ## `PartOpens` -/

/-- **`PartOpens` at a part that holds no message**: no round's message is held, so there is nothing to open. -/
theorem partOpens_of_none {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening}
    {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (h : ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ i, L.msgAt x s i = none) :
    PartOpens H D DO L shape shows opening cmt s := by
  intro x st hst hf xO w _ _
  exact Or.inl fun _ _ i _ m hm => by simp [Layout.held, h x st hst hf i] at hm

/-- **`PartOpens` at every part of the holo shape, through registered climbs**: M's `partOpens_of_climbs` (any part
of any shape) at `Form.Holo`: each part opens what it holds to the digest its public file names (`HeldOpens`), and the
setup climbs that digest and the round's commit string to one registered root (`RegClimbs`). -/
theorem partOpens_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s) : ∀ s, PartOpens H D DO L Form.Holo shows opening cmt s :=
  fun s => partOpens_of_climbs (hO s) (hC s)

/-! ## A part at the holo shape -/

/-- **Part `s` at the holo shape** (the holo counterpart of M's `DensePart`): its links for its check or a collision
the finders charge (`PartsLinksOr`, any kind of part), and for its calls either none where its coins are shown (the
algebra, opening and sparse parts, `LogsNil`) or a level's links for them (`PartsLogLinks`). -/
def HoloPart (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T)
    (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartsLinksOr H D DO decode L Form.Holo shows chk s ∧
    (LogsNil D Form.Holo shows logs s ∨ PartsLogLinks H D DO decode L Form.Holo shows logs s)

/-- **A holo level named `rec-open/…`, from M's dense links at the holo shape** (`OpenLinks`, `OpenLogLinks`). -/
theorem holoPart_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (hC : OpenLinks D decode L Form.Holo shows chk s) (hL : OpenLogLinks D decode L Form.Holo shows logs s) :
    HoloPart H D DO decode L shows chk logs s :=
  ⟨partsLinksOr_of_links (partsLinks_of_open hC), Or.inr (partsLogLinks_of_open hL)⟩

/-- **A holo part whose check makes no SHA-512 call** (M's `partLogs_of_nil` premise): its links for its check and
`logs s` empty everywhere. -/
theorem holoPart_of_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (hC : PartsLinksOr H D DO decode L Form.Holo shows chk s) (h0 : ∀ x c ms, logs s x c ms = []) :
    HoloPart H D DO decode L shows chk logs s :=
  ⟨hC, Or.inl fun x _ _ _ _ c _ ms => h0 x c ms⟩

/-! ## `VStarAcceptsR` and V*'s parts at the holo shape -/

/-- **`VStarAcceptsR` at the holo shape from the parts' links**: `vstarAccepts_holo` with every part's `PartChecks`
from its links or a collision (`partChecks_holo`) and its `PartOpens` through registered climbs (`partOpens_holo`). -/
theorem vstarAccepts_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartsLinksOr H D DO decode L Form.Holo shows chk s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D IV Form.Holo chk) :
    VStarAcceptsR H D I DO Form.Holo :=
  vstarAccepts_holo hR hVL (partChecks_holo hDO hC) (partOpens_holo hO hRC) hS hD

/-- **V\*'s parts at the holo shape, from the parts' links** (`VStarParts` at `Form.Holo`, what `codeFacts_of_parts`
and `ksA_deployed_parts` read there): with every part a holo part (`HoloPart`), its `PartChecks` from its links
(`partChecks_holo`), its `PartOpens` through registered climbs (`partOpens_holo`), and its `PartLogs` nil where its
coins are shown or from a level's links for its calls (`partLogs_holo`), the inner verifier's calls theirs or public
ones the V* setup evaluates (`LogDecompPub`, `PubReg`). -/
theorem vstarParts_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {pub : Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, HoloPart H D DO decode L shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D I.V Form.Holo chk)
    (hLD : LogDecompPub H D I.V Form.Holo logs pub) (hPR : PubReg H D Form.Holo pub) (hH : HeldPub D I L Form.Holo) :
    VStarParts H D I DO L opening Form.Holo :=
  ⟨shows, chk, cmt, logs, pub, partChecks_holo hDO fun s => (hP s).1, partOpens_holo hO hRC,
    partLogs_holo hDO fun s => (hP s).2, hS, hD, hLD, hPR, hH⟩

/-- **V\*'s parts at the holo shape with the staging off the run, from the parts' links** (`VStarPartsStaging` at
`Form.Holo`, what `KS_of_parts` reads there): `vstarParts_holo_of_links` with the staging's fact about `D.vstar`
(`VstarShows`) in place of `Staged`. -/
theorem vstarPartsStaging_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {pub : Msg → List (Msg × Msg)} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, HoloPart H D DO decode L shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H D DO L Form.Holo shows cmt dig s)
    (hV : VstarShows D I L Form.Holo decC shows cmt) (hD : Decomp H D I.V Form.Holo chk)
    (hLD : LogDecompPub H D I.V Form.Holo logs pub) (hPR : PubReg H D Form.Holo pub) (hH : HeldPub D I L Form.Holo) :
    VStarPartsStaging D I DO H L opening decC Form.Holo :=
  ⟨shows, chk, cmt, logs, pub, partChecks_holo hDO fun s => (hP s).1, partOpens_holo hO hRC,
    partLogs_holo hDO fun s => (hP s).2, hV, hD, hLD, hPR, hH⟩

end Facts

/-! ## At `deployRec` -/

section Deployed

variable {W : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W) (ivScope : Msg → Bool) (b : RecDeploy.Brings)

/-- **`VStarAcceptsR` at the holo shape at `deployRec`**, over the outer record `deployZkRec relation`: where
`deployRec` brings the layout's readers it reads through them (`RecDeploy.readsThrough_deployRec`), its V* statements
number `nS` (`RecDeploy.vstarLen_deployRec`), the outer record parses at `parseOf relation readOfO` by definition, and
the rest is `vstarAccepts_holo_of_links`: the parts' links, each or a collision the finders charge (`Be`'s among the
recorded calls, `Br`'s with the setup `b.setup` lists), the openings through registered climbs, the staging and the
decomposition. -/
theorem vstarAccepts_holo_deployRec {L : Layout} (hb : b.rd = RecDeploy.readersOf L)
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hC : ∀ s, PartsLinksOr H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L Form.Holo shows chk s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L Form.Holo shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) :
    VStarAcceptsR H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) Form.Holo :=
  vstarAccepts_holo_of_links rfl (RecDeploy.readsThrough_deployRec H I.V ivScope b L hb)
    (RecDeploy.vstarLen_deployRec I.V ivScope b) hC hO hRC hS hD

/-- **V\*'s parts at the holo shape at `deployRec`** (`ksA_deployed_parts`' holo premise, at `relO := relation`):
`vstarParts_holo_of_links` with the outer record `deployZkRec relation`'s parser by definition. -/
theorem vstarParts_holo_deployRec {L : Layout} {opening : Opening}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, HoloPart H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L Form.Holo shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) Form.Holo pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L Form.Holo) :
    VStarParts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      Form.Holo :=
  vstarParts_holo_of_links rfl hP hO hRC hS hD hLD hPR hH

/-- **V\*'s parts at the holo shape at `deployRec`, with the staging off the run** (`KS_of_parts`' holo premise
there): `vstarPartsStaging_holo_of_links` with the outer record `deployZkRec relation`'s parser by definition. -/
theorem vstarPartsStaging_holo_deployRec {L : Layout} {opening : Opening} {decC : Msg → Msg → FlockDeploy.Coin}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, HoloPart H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation)
      RecDeploy.readOfO L shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo shows cmt dig s)
    (hV : VstarShows (RecDeploy.deployRec I.V ivScope b) I L Form.Holo decC shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) {pub : Msg → List (Msg × Msg)}
    (hLD : LogDecompPub H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo logs pub)
    (hPR : PubReg H (RecDeploy.deployRec I.V ivScope b) Form.Holo pub)
    (hH : HeldPub (RecDeploy.deployRec I.V ivScope b) I L Form.Holo) :
    VStarPartsStaging (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) H L
      opening decC Form.Holo :=
  vstarPartsStaging_holo_of_links rfl hP hO hRC hV hD hLD hPR hH

end Deployed

end Claims.Recursion.VStar
