import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.Claims.Recursion.Deploy
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
| algebra parts, holo | `RecResiduals.unitLaid S ns nv` | `check_part_at` (`holo := true`) | `laid_part` | net yes, flag new |
| R's levels | `RecOpen.unitLaid L H` | `RecOpen.check_ok_stacked` | `laid_unit` | net yes, name new |
| R's opening parts | `RecResiduals.unitLaid S ns nv` | `check_opening` | `laid_part` | net yes, check new |
| RecSparse | `RecSparse.unitLaid sh` | `check_sparse` | `laid_sparse` | new |

* **Templates** (`Flock.Readable`): `partCols_of_check_at` (an algebra part at either holo flag, so the holo parts'
  `h` round needs nothing beyond the flag: `laid_part` holds for any `S`, `ns`, `nv`), `openOuts_of_net` and
  `openOuts_of_check_any` (a level's climb under either name), `partCols_of_check_opening` (R's opening parts), and
  `sparseOuts_of_check` (the RecSparse unit, with `inWords_sparse`).
* **Links** (`Claims.Recursion.VStar`): what is not the circuit's semantics, per kind of part, at one compiled
  statement, V* statement and coins: `ClimbAt` (either level kind, M's `OpenLinks` body), `ResReadsAt` (an algebra part
  at either flag, M's `ResReads` body), `OpeningReadsAt`, `SparseCopyAt`; `PartKindAt` is their disjunction and
  `PartsLinks s` asks it wherever part `s`'s coins are shown. `partChecks_of_links` gives `PartChecks s` at any shape;
  `partChecks_holo` at every part of the holo shape. M's dense links are instances (`partsLinks_of_reads`,
  `partsLinks_of_open`).
* **`PartLogs`** (`partLogs_of_shown_nil`, `partLogs_holo`): the algebra, opening and sparse parts' checks log no call,
  so `logs s = []` wherever their coins are shown; a level's check logs the climb's calls, which its recorded calls
  must hold (`ClimbLogs`).
* **`PartOpens`** (`partOpens_of_rows`): the part opens, through recorded calls, the row commitments its circuit
  computes (`RowsOpen`), and those are the commit strings the V* statement shows (`RegTies`, the registration, which
  rests on SHA-512 over the setup's out-of-oracle evaluations: card f93bbff1's finder).
* **At the holo shape** (`vstarAccepts_holo_of_links`), and at `deployRec` (`vstarAccepts_holo_deployRec`), where
  `deployRec` at `⟨L.msgs, L.calls⟩` reads through `L` (`readsThrough_deployRec`) and the outer record
  `deployZkRec relation` parses at `parseOf relation readOfO` by definition.
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

/-! ## The algebra parts at either holo flag -/

/-- **An algebra part's copies compute its residuals, at the statement's holo flag**: `partCols_of_satisfies` with the
flag the circuit's META gives (`holoOf`), the net `RecResiduals.check` built at `{ sh with holo }` (`check_part_at`).
At `holo = true` it is the holo shape's algebra parts, whose `h` round is one of the part's row ports like any other. -/
theorem partCols_of_satisfies_at {st : Flock.Stmt} (L : FlockSoundness.Layout st)
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
  have hL := laidRows_of_satisfies L regs hz ho hg hu hn
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
  exact partCols_of_satisfies_at L regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

/-! ## The levels, under either name -/

/-- **A level's copies climb, at any name the verifier built its net under**: `climb_of_satisfies` with the net taken
as given, so R's levels (`rec-open-stacked/…`) are the levels' (`rec-open/…`) template under another name. -/
theorem openOuts_of_net {st : Flock.Stmt} (L : FlockSoundness.Layout st)
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
  have hL := laidRows_of_satisfies L regs hz ho hg hu hl
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
  exact openOuts_of_net L regs hz ho hg hu hvalid hl ms hms hin

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
  have hL := laidRows_of_satisfies L regs hz ho hg hu hn
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
def SparseOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) (wA vA : Array Flock.F128) (τA : Array ℕ) : Prop :=
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
theorem sparseOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : Flock.RecSparse.check p.sts[t].c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params p.sts[t].c.unit.name = some sh)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (wA vA : Array Flock.F128) (τA : Array ℕ)
    (hin : FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (copyOf p t htl w o g u)) :
    SparseOuts sh (copyOf p t htl w o g u) wA vA τA := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  obtain ⟨hv, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse hc hp
  have hL := laidRows_of_satisfies L regs hz ho hg hu (l := Flock.RecSparse.unitLaid sh) hn
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

/-- **A level's links** at the compiled statement `x`, part `s`, V* statement `xO` and coins `c` (M's `OpenLinks` body,
under either level name): `xO` decodes to an untyped statement without public input words one of whose tables
`RecOpen.check` accepts as a level (`rec-open/…`) or as one of R's (`rec-open-stacked/…`); a copy of its unit whose
input columns hold a row, siblings, sums, coefficients and position (`OpenIns`) on every witness and agreeing list of
messages; and the climb and consistency term on its outputs (`OpenOuts`) give the inner check. -/
def ClimbAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
      _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
      (_root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∨
        _root_.Flock.RecOpen.stackedParams p.sts[t].c.unit.name = some (LANES, Ht)) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms →
          ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
            (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
            Flock.Readable.OpenIns (Flock.Readable.copyOf p t ht w o g u) LANES Ht r a cf rest idx ss ∧
            (Flock.Readable.OpenOuts (Flock.Readable.copyOf p t ht w o g u) msch LANES Ht r a cf rest idx ss →
              chk s x c ms)

/-- **An algebra part's links, at the statement's holo flag** (M's `ResReads` body with the flag `holoOf` gives):
`xO` decodes to an untyped statement without public input words one of whose tables `RecResiduals.check` accepts as
part `j` of a shape's algebra at holo flag `holo`, `S` with row ports of `ns` elements and a `v` of `nv`; and S's
residuals at the elements a copy of its unit reads (`rowsRead`, `vRead`) give the inner check on every witness and
agreeing list of messages. At the holo shape `holo = true`. -/
def ResReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (holo : Bool)
      (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ),
      _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
      _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo ∧
      _root_.Flock.RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

/-- **An R opening part's links**: `ResReadsAt` with `RecOpening.check` accepting the table as part `j` of R's
opening at `(m, cl)`. -/
def OpeningReadsAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ (m cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
      (nv : ℕ),
      _root_.Flock.RecOpening.check p.sts[t].c = .ok () ∧
      _root_.Flock.RecOpening.params p.sts[t].c.unit.name = some (m, cl, j) ∧
      _root_.Flock.RecOpening.shape? m cl = some sh ∧ _root_.Flock.RecOpening.part sh j = .ok (S, ns, nv) ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms →
          PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
            (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

/-- **The RecSparse part's links**: `xO` decodes to an untyped statement without public input words one of whose
tables `RecSparse.check` accepts at the shape `sh` its unit's name gives; a copy of its unit whose input columns hold
messages `wA`, `v` elements `vA` and types `τA` (`SparseIn`) on every witness and agreeing list of messages; and its
outputs there (`SparseOuts`) give the inner check. -/
def SparseCopyAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∃ t, ∃ ht : t < p.sts.size, ∃ sh : _root_.Flock.RecSparse.Shape,
      _root_.Flock.RecSparse.check p.sts[t].c = .ok () ∧ _root_.Flock.RecSparse.params p.sts[t].c.unit.name = some sh ∧
      ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
        ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms →
          ∃ (wA vA : Array _root_.Flock.F128) (τA : Array ℕ),
            FlockVBridge.Sparse.SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) (Flock.Readable.copyOf p t ht w o g u) ∧
            (Flock.Readable.SparseOuts sh (Flock.Readable.copyOf p t ht w o g u) wA vA τA → chk s x c ms)

/-- **Some kind of part's links hold** at `x`, `s`, `xO`, `c`: a level's (either name), an algebra part's (either
flag), an R opening part's or the RecSparse part's. -/
def PartKindAt (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (x : Msg) (s : ℕ) (xO : Msg) (c : ℕ → Coin) : Prop :=
  ClimbAt decode L chk x s xO c ∨ ResReadsAt decode L chk x s xO c ∨ OpeningReadsAt decode L chk x s xO c ∨
    SparseCopyAt decode L chk x s xO c

/-- **Part `s`'s links** (a code fact about V*'s staging, the layout and the inner check at part `s`, none of it the
circuit's semantics): wherever a V* statement `xO` of a compiled statement of the shape shows the coins `c` for part
`s`, some kind of part's links hold there (`PartKindAt`). -/
def PartsLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    PartKindAt decode L chk x s xO c

/-! ## `PartChecks` from the links -/

/-- **One kind of part's links and the outer relation give the inner check**: at the deployed parser, wherever
`Flock.check` holds at `xO` and `w`, the links at `xO` give the inner check on every agreeing list of messages, by the
kind's template (`climb_of_check`'s `openOuts_of_check_any`, `partCols_of_check_at`, `partCols_of_check_opening`,
`sparseOuts_of_check`). -/
theorem chk_of_kindAt {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {x : Msg} {s : ℕ} {xO : Msg} {c : ℕ → Coin} {w : Flock.Wit}
    (hck : Flock.check (Flock.Readable.parseOf Flock.Readable.relation decode) xO w = true)
    (hK : PartKindAt decode L chk x s xO c) {ms : List Msg} (hms : Agrees L x s w ms) : chk s x c ms := by
  rcases hK with hK | hK | hK | hK
  · obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, hin, hout⟩ := hlink w ms hms
    exact hout (Flock.Readable.openOuts_of_check_any hck hx ht hpi htl ho hg hu hrec hp msch hmsch hin)
  · obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, holo, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
      hlink⟩ := hK
    exact hlink w ms hms (Flock.Readable.partCols_of_check_at hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, m, cl, j, sh, S, ns, nv, hrc, hp, hsh, hpart, o, g, u, ho, hg, hu, hlink⟩ := hK
    exact hlink w ms hms (Flock.Readable.partCols_of_check_opening hck hx ht hpi htl hrc hp hsh hpart ho hg hu
      (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u))
      (fun _ he => Flock.Readable.readElems_ge he) (fun _ he => Flock.Readable.readElems_ge he)
      (Flock.Readable.inCols_readElems ns nv _))
  · obtain ⟨p, hx, ht, hpi, t, htl, sh, hrc, hp, o, g, u, ho, hg, hu, hlink⟩ := hK
    obtain ⟨wA, vA, τA, hin, hout⟩ := hlink w ms hms
    exact hout (Flock.Readable.sparseOuts_of_check hck hx ht hpi htl hrc hp ho hg hu wA vA τA hin)

/-- **`PartChecks` at any part, from its links**: at the deployed parser (`Readable.parseOf relation decode`), part
`s`'s links (`PartsLinks`) give `PartChecks s` at any shape, its inner check holding on every agreeing list of messages
with no appeal to a collision (`chk_of_kindAt`). -/
theorem partChecks_of_links {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : PartsLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck _
  rw [hDO] at hck
  exact Or.inl fun c hc ms hms => chk_of_kindAt hck (hR x st hst hf xO c hc) hms

/-- **M's algebra links are part links** (`ResReads`, the plain flag): a shape `RecResiduals.shape?` gives is plain
(`shape?_holo`), so its part at `holo := false` is its own. -/
theorem partsLinks_of_reads {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReads D decode L shape shows chk s) :
    PartsLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
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
    ho, hg, hu, hlink⟩)

/-- **M's level links are part links** (`OpenLinks`, a unit named `rec-open/…`). -/
theorem partsLinks_of_open {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : OpenLinks D decode L shape shows chk s) :
    PartsLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  exact Or.inl ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, Or.inl hp, o, g, u, ho, hg, hu, hlink⟩

/-- **`PartChecks` at every part of the holo shape**: `partChecks_of_links` at `Form.Holo`. -/
theorem partChecks_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ∀ s, PartsLinks D decode L Form.Holo shows chk s) : ∀ s, PartChecks H D DO L Form.Holo shows chk s :=
  fun s => partChecks_of_links hDO (hR s)

/-! ## `PartLogs` -/

/-- **`PartLogs` at a part whose inner check logs no call where its coins are shown**: `partLogs_of_nil` with `logs s`
empty only at the compiled statements of the shape and the V* statements that show part `s`'s coins. -/
theorem partLogs_of_shown_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    (h : ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin),
      shows s x xO c → ∀ ms, logs s x c ms = []) :
    PartLogs H D DO L shape shows logs s := fun x st hst hf xO _ _ _ => Or.inl fun c hc ms _ e he => by
  rw [h x st hst hf xO c hc ms] at he
  cases he

/-- **A level's logged calls are recorded** (a code fact about the layout's SHA-512 evaluations at a level): wherever
the outer relation holds at a V* statement of a compiled statement of the shape, with true recorded calls, the calls
part `s`'s inner check logs at the coins the statement shows (the climb's: the leaf's hash and each node's) are among
the calls the part records (`L.shaAt`, the climb gadget's SHA-512 rows), or two recorded calls collide. This is
`PartLogs s`, named for the parts whose check logs calls. -/
def ClimbLogs (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartLogs H D DO L shape shows logs s

/-- **`PartLogs` at every part of the holo shape**: each part either logs no call where its coins are shown (the
algebra parts, R's opening parts and the RecSparse part: residuals only), or is a level whose logged climb calls are
recorded (`ClimbLogs`). -/
theorem partLogs_holo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    (h : ∀ s, (∀ (x : Msg) (st : Stmt), D.parse x = some st → Form.Holo st.form → ∀ (xO : Msg) (c : ℕ → Coin),
      shows s x xO c → ∀ ms, logs s x c ms = []) ∨ ClimbLogs H D DO L Form.Holo shows logs s) :
    ∀ s, PartLogs H D DO L Form.Holo shows logs s := fun s =>
  (h s).elim partLogs_of_shown_nil id

/-! ## `PartOpens` -/

/-- **`PartOpens` at a part that holds no message**: no round's message is held, so there is nothing to open. -/
theorem partOpens_of_none {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout} {shape : Form → Prop}
    {opening : Opening} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (h : ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ i, L.msgAt x s i = none) :
    PartOpens D DO L shape opening cmt s := by
  intro x st hst hf xO w _ i _ m hm
  simp [Layout.held, h x st hst hf i] at hm

/-- **Part `s` opens, through calls it records, the row commitments its circuit computes** (a code fact about part
`s`'s row nets and the layout): wherever the outer relation holds at a V* statement of a compiled statement of the
shape, each round's message the part holds opens under `opening` the commit string `bc x s w i` (the `b ‖ c` its hm96
rows compute off its witness), through calls among those it records. -/
def RowsOpen (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop) (opening : Opening)
    (bc : Msg → ℕ → Flock.Wit → ℕ → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ cs, cs ⊆ L.calls x s w ∧ opening.opens (bc x s w i) m cs

/-- **The registration: part `s`'s row commitments are the commit strings its V\* statement shows** (a code fact about
the setup's registered reads, `RecIns.setup`): wherever the outer relation holds at a V* statement `xO` of a compiled
statement of the shape, at each round whose message the part holds, the `b ‖ c` its rows compute is the round's commit
string `xO` shows (`cmt x i xO`, the registered `rec-c<i>`). The setup's tree over the registered rows is recomputed
out of the oracle, so this binds by SHA-512's collision resistance over those evaluations (card f93bbff1). -/
def RegTies (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (bc : Msg → ℕ → Flock.Wit → ℕ → Msg) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ i < L.rounds x, (L.msgAt x s i).isSome → bc x s w i = cmt x i xO

/-- **`PartOpens` from the row commitments and the registration**: the part opens its rows' `b ‖ c` (`RowsOpen`), which
is the round's commit string its V* statement shows (`RegTies`). -/
theorem partOpens_of_rows {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout} {shape : Form → Prop}
    {opening : Opening} {bc : Msg → ℕ → Flock.Wit → ℕ → Msg} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (hR : RowsOpen D DO L shape opening bc s) (hT : RegTies D DO L shape bc cmt s) :
    PartOpens D DO L shape opening cmt s := by
  intro x st hst hf xO w hck i hi m hm
  have hs : (L.msgAt x s i).isSome := by
    cases h : L.msgAt x s i with
    | none => simp [Layout.held, h] at hm
    | some _ => rfl
  rw [← hT x st hst hf xO w hck i hi hs]
  exact hR x st hst hf xO w hck i hi m hm

/-! ## `VStarAccepts` at the holo shape -/

/-- **`VStarAccepts` at the holo shape from the parts' links**: `vstarAccepts_holo` with every part's `PartChecks` from
its links (`partChecks_holo`) and its `PartOpens` from its rows and the registration (`partOpens_of_rows`). -/
theorem vstarAccepts_holo_of_links [Inhabited Coin] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read} {bc : Msg → ℕ → Flock.Wit → ℕ → Msg}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartsLinks D decode L Form.Holo shows chk s)
    (hRows : ∀ s, RowsOpen D DO L Form.Holo opening bc s) (hT : ∀ s, RegTies D DO L Form.Holo bc cmt s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D IV Form.Holo chk) : VStarAccepts H D I DO Form.Holo :=
  vstarAccepts_holo hR (partChecks_holo hDO hC) (fun s => partOpens_of_rows (hRows s) (hT s)) hS hD

end Facts

/-! ## At `deployRec` -/

/-- **`deployRec` at the layout's readers reads through them**: its inner view is `IV`'s run of `b.rd.msgs` at the
statement's inner part (`xI_deployRec`), and its calls are `b.rd.calls`' (`Deploy.lean`), by definition. -/
theorem readsThrough_deployRec (H : Msg → Msg) (IV : Verifier FlockDeploy.Coin) (ivScope : Msg → Bool)
    (b : RecDeploy.Brings) {L : Layout} (hm : b.rd.msgs = L.msgs) (hc : b.rd.calls = L.calls) :
    ReadsThrough H (RecDeploy.deployRec IV ivScope b) IV L.msgs L.calls := by
  refine ⟨fun x r ws => ?_, fun x ws => ?_⟩
  · show RecDeploy.innerView H IV b.rd x r ws = _
    unfold RecDeploy.innerView
    rw [hm]
    rfl
  · show (ws.mapIdx fun s w => b.rd.calls x s w).flatten = _
    rw [hc]

/-- **`VStarAccepts` at the holo shape at `deployRec`**, over the outer record `deployZkRec relation`: the readers are
the layout's (`readsThrough_deployRec`), the outer record parses at `parseOf relation readOfO` by definition, and the
rest is `vstarAccepts_holo_of_links`: the parts' links, rows and registration, the staging and the decomposition. -/
theorem vstarAccepts_holo_deployRec {W : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W)
    (ivScope : Msg → Bool) (b : RecDeploy.Brings) {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    {bc : Msg → ℕ → Flock.Wit → ℕ → Msg} (hm : b.rd.msgs = L.msgs) (hc : b.rd.calls = L.calls)
    (hC : ∀ s, PartsLinks (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L Form.Holo shows chk s)
    (hRows : ∀ s, RowsOpen (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo opening bc s)
    (hT : ∀ s, RegTies (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Holo bc cmt s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L Form.Holo shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V Form.Holo chk) :
    VStarAccepts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) Form.Holo :=
  vstarAccepts_holo_of_links rfl (readsThrough_deployRec H I.V ivScope b hm hc) hC hRows hT hS hD

end Claims.Recursion.VStar
