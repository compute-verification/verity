import Proofs.Flock.Claims.Recursion.VStar
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.VBridge.ResUnit
import Proofs.Flock.VBridge.Verifier

/-!
# V*'s parts at the Claims relation: a unit copy's semantics, and the algebra parts

`PartChecks s`, `PartOpens s` and `PartLogs s` (`VStar.lean`) read V* part `s`'s outer relation as
`Flock.check DO.parse xO w = true`. At the deployed parser (`Readable.parseOf relation decode`) that is the verifier's
parsed statement with every table's `Layout.model` satisfied by its share of the hidden values (`relS_layout`). This
file takes that to the circuit V* ships, which the verifier rebuilds itself: `RecVStar.checkFile` compares the part's
circuit file byte for byte with `RecVStar.circuitFile`, and `RecResiduals.check` makes the unit its own net.

* **Every template** (`laidRows_of_satisfies`): in every block, every unit copy's laid-out rows (`LaidRows`, the rows
  the verifier built the unit's net from) hold on the hidden values at the copy's columns (`copyCol`). It is the
  placement (`Layout.placement`, `rowsSat_of_satisfies`), one copy of the stack (`sat_stack`, `stack_one`) and the
  net's rows as its laid-out rows (`laidRows_ofNet`); nothing in it reads the template.
* **The algebra parts** (`partCols_of_satisfies`): where `RecResiduals.check` accepted the table's circuit as part `j`
  of a shape's algebra, a copy whose input columns hold `W` on the row ports and `vv` on `v`'s (`InCols`) holds each
  of the part's residuals at `W`, `vv` on its output columns (`PartCols`): `laid_part` at the verifier's net
  (`check_part`).
* **The rec-open parts** (`climb_of_satisfies`), the dense shape's first statements, one per Merkle level: where
  `RecOpen.check` accepted the table's circuit, a copy whose input columns hold a row, its siblings, `acc`, the
  coefficients and the position's bits holds the climb from the row's leaf and the consistency term on its output
  columns: `laid_unit` at the verifier's net (`RecOpen.check_ok`).
* **At the relation** (`relS_of_check`, `layout_of_check`, `partCols_of_check`, `climb_of_check`): `Flock.check` at
  the deployed parser gives every table's layout and satisfied model, so the above at every table, block, VU and
  copy.
* **`PartChecks` at an algebra part** (`partChecks_of_res`): the circuit's semantics above, with the links that are not
  the circuit's (`ResLinks`): the staging (the V* statement the part's coins are shown on is an accepted algebra part),
  the input tie (a copy's input columns hold the messages the part holds and the coins' coefficients) and the output
  tie (the part's residuals there give its inner check).
* **The input tie's elements** (`inCols_readElems`, `ResReads`, `partChecks_of_reads`): a copy's input columns always
  hold the elements they read (`rowsRead`, `vRead`), so `ResReads` drops `ResLinks`' existential: S's residuals at
  the elements the copy reads give the inner check.
* **`PartChecks` at a rec-open part** (`partChecks_of_open`): the same, with the rec-open links (`OpenLinks`): an
  accepted rec-open table, a copy whose inputs hold the row, siblings, sums, coefficients and position (`OpenIns`), and
  the climb and consistency term on its outputs (`OpenOuts`) giving the inner check.
* **`PartLogs` at an algebra part** (`partLogs_of_nil`): its inner check, the part's residuals, makes no SHA-512 call,
  so at `logs s = []` the part records every call its check makes.
* **`PartLogs` at a rec-open part** (`partLogs_of_open`): the same circuit semantics as `partChecks_of_open`, with the
  links' output tie putting the climb's SHA-512 evaluations among the part's recorded calls (`OpenLogLinks`).
* **`PartOpens` through registered climbs** (`partOpens_of_climbs`, at any part of any shape): a part's hm96 row opens
  each message it holds to the `b ‖ c` its public file names (`HeldOpens`), and the V* setup climbs that digest and the
  round's commit string to one registered root (`RegClimbs`) outside the oracle. Two climbs from different values to
  one value collide (`chain_coll`, wherever each level's frame keeps the value climbing whole, `StepsApart`), and every
  input they hash is one the setup evaluates or one the part records, so the collision is a registered tie `Br` finds
  or one among the recorded calls (`coll_of_chains`).
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

/-! ## A unit copy's columns off the hidden values -/

/-- **A unit copy's columns off the hidden values**: column `j` of the copy at in-block base `base`, in block `o` of
table `c` of `J` (`zOf`'s interleave of the tables' bits), for `j` up to the copy's constant `top`; `false` past it. -/
def copyCol (J c : ℕ) (w : Claims.Flock.Wit) (kLog o base top j : ℕ) : Bool :=
  if j ≤ top then w.hid.getD ((o * 2 ^ kLog + (base + j)) * J + c) false else false

theorem bit_toZ (b : Bool) : FlockSoundness.bit (toZ b) = b := by cases b <;> rfl

/-- **Every template: a satisfied table's unit copies hold their laid-out rows.** Let a table's model statement
(`Layout.model`, at any regions) be satisfied by the table's share of the hidden values (`zOf`), and the verifier have
built the table's unit from `l` (`netOf`). Then in every block `o`, at every VU `g` and copy `u`, `l`'s rows hold at the
copy's columns (`copyCol`), from the end of the unit's input rows. -/
theorem laidRows_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {name : String} {l : Flock.HmNets.Laid}
    (hl : Flock.HmNets.netOf name l = .ok st.c.unit) :
    LaidRows l (st.c.unit.inWords * Flock.WORD)
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos) := by
  have hpl := L.placement regs hg
  have hs := rowsSat_of_satisfies hpl hz ⟨o, ho⟩
  have hs1 := FlockVBridge.sat_stack _ _ hu hs
  have hone := FlockVBridge.stack_one _ _ hu hs
  beta_reduce at hone
  rw [hpl.one_eq hz ⟨o, ho⟩] at hone
  have hl' : Flock.HmNets.netOf name l = .ok (FlockSoundness.lunit st.c) := by
    rw [FlockSoundness.lunit_flat hfl]; exact hl
  have hL := FlockVBridge.laidRows_ofNet L.unit_built hl' (L.lunitLog_le (L.slot_lt hg hu)) hs1
    (hone.trans (by simp [FlockSoundness.bit]))
  rw [L.unit_at.in_eq] at hL
  have hw : (unitRows st L.unit_built).w = st.c.unit.constPos + 1 := by
    show _ + _ + 1 = _
    rw [L.unitRows_nIn_nComp, FlockSoundness.lunit_flat hfl]
  have hw' : (Rows.ofNet L.unit_built).w = st.c.unit.constPos + 1 := hw
  convert hL using 1
  funext j
  unfold FlockVBridge.colVal copyCol
  split_ifs
  · have hj : st.c.unitRange.slot (g * st.c.upv + u) + j < 2 ^ (L.model regs).kLog := by
      have := L.hbM regs hg u hu ⟨j, by omega⟩
      rwa [FlockSoundness.upos_flat hfl] at this
    have e : L.col regs hg ((unitRows st L.unit_built).stackCol st.c.upv hu ⟨j, by omega⟩) = ⟨_, hj⟩ := by
      apply Fin.ext
      rw [L.col_stackCol regs hg hu, FlockSoundness.upos_flat hfl]
    exact (bit_toZ _).symm.trans
      (congrArg (fun i => FlockSoundness.bit ((L.model regs).block (zOf J w c st.m) ⟨o, ho⟩ i)) e.symm)
  · omega
  · omega
  · rfl

/-- **The algebra parts: a satisfied table's copies compute the part's residuals.** Let `RecResiduals.check` accept
the table's circuit, its unit named for part `j` of a shape `sh` the verifier accepts (plain), part `j` being `S` with
row ports of `ns` elements and a `v` of `nv`. In every block, at every VU and copy, if the copy's input columns hold
`W` on the row ports and `vv` on `v`'s (`InCols`), `W` and `vv` zero past `ns`'s total and `nv`, its output columns
hold each of S's residuals at `W`, `vv`, 128 bits each, then zeros (`PartCols`). -/
theorem partCols_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) (hc : RecResiduals.check st.c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params st.c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf st.c = .ok false)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part sh j = .ok (S, ns, nv))
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv
      (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)) :
    PartCols S W vv (copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
      (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (part_small hpart) _ hL W vv hw0 hv0 hin

/-- **The rec-open parts: a satisfied table's copies climb.** Let `RecOpen.check` accept the table's circuit, its unit
named for `LANES` and `H` (`params`). In every block, at every VU and copy, whose columns are `v` (`copyCol`): if the
input columns hold a row `r`, `H` siblings `ss`, `acc`'s first elements `a` and the rest `rest`, the coefficients `cf`
and the position's bits `idx`, the output columns hold the verifier of record's climb from the row's leaf
(`Flock.climb`) and 64 zero bytes, then the consistency term at `k = 0, 1` and `rest`: `laid_unit` at the verifier's
net (`RecOpen.check_ok`). -/
theorem climb_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st) (hfl : st.c.tmpl = none)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {J c : ℕ} {w : Claims.Flock.Wit}
    (hz : (L.model regs).Satisfies (zOf J w c st.m)) {o g u : ℕ} (ho : o < 2 ^ (st.m - st.c.kLog))
    (hg : g < st.c.g) (hu : u < st.c.upv) {v : ℕ → Bool}
    (hv : v = copyCol J c w st.c.kLog o (st.c.unitRange.slot (g * st.c.upv + u)) st.c.unit.constPos)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check st.c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params st.c.unit.name = some (LANES, H))
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96)
    (hss : ss.length = H) (hrow : FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i])
    (hacc : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    FlockVBridge.ColsHold v
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
        (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
          rest)) := by
  subst hv
  obtain ⟨hvalid, hl⟩ := Flock.RecOpen.check_ok hrec hp
  have hH : H ≤ 32 := by
    simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hvalid
    exact hvalid.2
  have hL := laidRows_of_satisfies L hfl regs hz ho hg hu hl
  rw [FlockVBridge.inWords_unit hl] at hL
  exact FlockVBridge.laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

/-! ## At Flock's relation, at the deployed parser -/

/-- **`Flock.check` at the deployed parser**: the statement bytes decode to a statement the verifier's parser returns
(`Parses`), at whose readable form the witness satisfies `RelS`. -/
theorem relS_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) :
    ∃ p, parseBytes decode x = some p ∧ Parses p ∧ Claims.Flock.RelS (toReadable relation p) w := by
  unfold Claims.Flock.check parseOf at h
  rcases hp : parseBytes decode x with _ | p
  · rw [hp] at h
    exact absurd h Bool.false_ne_true
  · rw [hp] at h
    have h' : decide (Claims.Flock.RelS (toReadable relation p) w) = true := h
    refine ⟨p, rfl, ?_, of_decide_eq_true h'⟩
    unfold parseBytes at hp
    obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hp
    exact parses_of_parse hr

/-- **Every table's layout and model at `Flock.check`**: at an untyped parsed statement without the verifier's own
public input words, each table has the verifier's layout, and its model statement (at the regions the soundness proof
reads) is satisfied by the table's share of the hidden values (`relS_layout`). -/
theorem layout_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size) :
    ∃ (L : FlockSoundness.Layout p.sts[t]) (regs : List (Model.Region p.sts[t].c.kLog (p.sts[t].m - p.sts[t].c.kLog))),
      (L.model regs).Satisfies (zOf p.sts.size w t p.sts[t].m) := by
  obtain ⟨p', hx', hp', hR⟩ := relS_of_check h
  rw [hx] at hx'
  cases hx'
  obtain ⟨L, hz⟩ := (relS_layout hp' ht hpi w).1 hR t htl
  exact ⟨L, _, hz⟩

/-- **Every table is flat at `Flock.check`**: at an untyped statement without public input words, each table's circuit
is one `HmRow.parse` returned (`tableAt`), which carries no template. -/
theorem flat_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hrec : Flock.HmOut.recUnit p.sts[t].c.unit.name = true) : p.sts[t].c.tmpl = none := by
  obtain ⟨p', hx', hp', -⟩ := relS_of_check h
  rw [hx] at hx'
  cases hx'
  obtain ⟨hc, -⟩ := tableAt (tablesOf_of_parses hp') ht hpi htl
  unfold ProvedScope.circuitOf at hc
  split at hc
  · rcases FlockSoundness.Discharge.Hidden.committed_of_commit hc with hc | hc
    · exact (FlockSoundness.parse_facts (FlockSoundness.Discharge.Hidden.parse_untyped ht hc).1).tmpl
    · rw [FlockSoundness.TmplValue.parseTmpl_notRec hc] at hrec
      cases hrec
  · exact (FlockSoundness.parse_facts hc).tmpl

theorem recUnit_open {n : String} {L H : ℕ}
    (h : Flock.RecOpen.params n = some (L, H) ∨ Flock.RecOpen.stackedParams n = some (L, H)) :
    Flock.HmOut.recUnit n = true := by
  rcases h with h | h <;> simp [Flock.HmOut.recUnit, h]

theorem recUnit_residuals {n : String} {q : ℕ × ℕ × ℕ × ℕ} (h : Flock.RecResiduals.params n = some q) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

theorem recUnit_sparse {n : String} {sh : Flock.RecSparse.Shape} (h : Flock.RecSparse.params n = some sh) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

theorem recUnit_opening {n : String} {q : ℕ × ℕ × ℕ} (h : Flock.RecOpening.params n = some q) :
    Flock.HmOut.recUnit n = true := by
  simp [Flock.HmOut.recUnit, h]

/-- **An algebra part's copies compute its residuals at `Flock.check`** (`partCols_of_satisfies` at every table):
wherever `Flock.check` holds at the deployed parser, at an untyped parsed statement without public input words, a
table `RecResiduals.check` accepted as part `j` of a shape's algebra holds, at every copy whose input columns hold `W`
and `vv`, the part's residuals at `W`, `vv` on the copy's output columns. -/
theorem partCols_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    (hc : RecResiduals.check p.sts[t].c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j)) {sh : Flock.Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf p.sts[t].c = .ok false)
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecResiduals.part sh j = .ok (S, ns, nv))
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (W vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → W e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InCols (partRep ns nv) W vv (copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)) :
    PartCols S W vv (copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) (partBase S ns nv) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact partCols_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_residuals hp)) regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

/-- **A rec-open part's copies climb at `Flock.check`** (`climb_of_satisfies` at every table): wherever `Flock.check`
holds at the deployed parser, at an untyped parsed statement without public input words, a table `RecOpen.check`
accepted under `LANES` and `H` holds, at every copy whose input columns hold a row, its siblings, `acc`, the
coefficients and the position's bits, the climb from the row's leaf and the consistency term on its output columns. -/
theorem climb_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    {v : ℕ → Bool} (hv : v = copyCol p.sts.size t w p.sts[t].c.kLog o
      (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check p.sts[t].c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, H))
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96)
    (hss : ss.length = H) (hrow : FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i])
    (hacc : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    FlockVBridge.ColsHold v
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
        (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
          rest)) := by
  obtain ⟨L, regs, hz⟩ := layout_of_check h hx ht hpi htl
  exact climb_of_satisfies L (flat_of_check h hx ht hpi htl (recUnit_open (.inl hp))) regs hz ho hg hu hv ms hms hrec hp r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc
    hcoef hdir

/-- The columns of copy `u` of VU `g` in block `o` of table `t`, off the hidden values (`copyCol`). -/
def copyOf (p : Parsed) (t : ℕ) (ht : t < p.sts.size) (w : Claims.Flock.Wit) (o g u : ℕ) : ℕ → Bool :=
  copyCol p.sts.size t w p.sts[t].c.kLog o (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u))
    p.sts[t].c.unit.constPos

/-- The elements columns `v` hold where `col` puts their bits: element `e`'s bit `t` at `col (128 e + t)`, zero past
`n` elements. -/
def readElems (col : ℕ → ℕ) (n : ℕ) (v : ℕ → Bool) (e : ℕ) : Flock.F128 :=
  if e < n then FlockVBridge.ofBits fun t => v (col (128 * e + t)) else 0

theorem wbit_readElems {col : ℕ → ℕ} {n : ℕ} {v : ℕ → Bool} {q : ℕ} (hq : q < 128 * n) :
    FlockVBridge.wbit (readElems col n v) q = v (col q) := by
  have hn : q / 128 < n := by omega
  unfold FlockVBridge.wbit readElems
  simp only [hn, ↓reduceIte]
  rw [FlockVBridge.bitOf_ofBits _ (Nat.mod_lt _ (by norm_num)), Nat.div_add_mod]

theorem readElems_ge {col : ℕ → ℕ} {n : ℕ} {v : ℕ → Bool} {e : ℕ} (he : n ≤ e) : readElems col n v e = 0 := by
  simp [readElems, show ¬e < n by omega]

/-- The elements columns `v` hold on a part's row ports (`ns`'s total). -/
def rowsRead (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : ℕ → Flock.F128 :=
  readElems (FlockVBridge.OfVStar.rowCol (partRep ns nv).ports 0) ns.sum v

/-- The elements columns `v` hold on a part's `v` port (`nv` elements). -/
def vRead (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : ℕ → Flock.F128 :=
  readElems (FlockVBridge.OfVStar.vBase (partRep ns nv) + ·) nv v

/-- **An algebra part's input columns always hold elements**: any columns `v` hold, on a part's row ports and `v`'s
port (`InCols`), the elements they read there (`readElems`), which are zero past the ports' sizes. So the input tie
asks only that those elements be the held messages' and the coins' coefficients. -/
theorem inCols_readElems (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) :
    InCols (partRep ns nv) (rowsRead ns nv v) (vRead ns nv v) v :=
  ⟨fun _ hq => (wbit_readElems hq).symm, fun _ hq => (wbit_readElems hq).symm⟩

/-- **A rec-open copy's inputs** at `laid_unit`'s offsets: the columns `v` hold a row `r` of `LANES` elements, `H`
64-byte siblings `ss`, the running sums `a` and 96 bytes `rest`, the `2 LANES` coefficients `cf` and the low `H` bits
of the position `idx`. -/
def OpenIns (v : ℕ → Bool) (LANES H : ℕ) (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ)
    (ss : List ByteArray) : Prop :=
  r.size = LANES ∧ a.size = 2 ∧ cf.size = 2 * LANES ∧ rest.size = 96 ∧ ss.length = H ∧
    FlockVBridge.ColsHold v 0 (Flock.MerkleScheme.rowBytes r) ∧
    (∀ i (hi : i < ss.length), ss[i].size = 64 ∧ FlockVBridge.ColsHold v (128 * LANES + 512 * i) ss[i]) ∧
    FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2)) (Flock.MerkleScheme.rowBytes a ++ rest) ∧
    FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024) (Flock.MerkleScheme.rowBytes cf) ∧
    ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h

/-- **A rec-open copy's outputs**: from `outStart`, the climb from the row's leaf through `ss` under `ms` and 64 zero
bytes, then the consistency term at `k = 0, 1` and `rest`. -/
def OpenOuts (v : ℕ → Bool) (ms : Flock.MerkleScheme) (LANES H : ℕ) (r a cf : Array Flock.F128) (rest : ByteArray)
    (idx : ℕ) (ss : List ByteArray) : Prop :=
  FlockVBridge.ColsHold v
    (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run Flock.HmNets.St.empty).2
      (Flock.RecOpen.wordPorts (FlockVBridge.unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
    ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
      (Flock.MerkleScheme.rowBytes #[FlockVBridge.consTerm LANES r a cf 0, FlockVBridge.consTerm LANES r a cf 1] ++
        rest))

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar

section Facts

variable {Coin Col T : Type}

/-- **An algebra part's links** (a code fact about V*'s staging, the layout and the inner check at part `s`, none of it
the circuit's semantics): wherever a V* statement `xO` of a compiled statement `x` of the shape shows the coins `c` for
part `s`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words one of whose tables
  `RecResiduals.check` accepts as part `j` of a shape's algebra (plain), `S` with row ports of `ns` elements and a `v`
  of `nv`;
* the input tie: the statement names a copy of that table's unit (block `o`, VU `g`, copy `u`) whose input columns, on
  every witness and every list of inner messages that agrees with the messages the part holds, hold elements `W` on the
  row ports and coefficients `vv` on `v`'s (`InCols`), zero past `ns`'s total and `nv`;
* the output tie: S's residuals at `W`, `vv` on that copy's output columns (`PartCols`) give the inner check `chk s`. -/
def ResLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ W vv : ℕ → _root_.Flock.F128, (∀ e, ns.sum ≤ e → W e = 0) ∧ (∀ e, nv ≤ e → vv e = 0) ∧
              InCols (partRep ns nv) W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) ∧
              (PartCols S W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
                (partBase S ns nv) → chk s x c ms)

/-- **`PartChecks` at an algebra part, from the shipped circuit's semantics**: at the deployed parser
(`Readable.parseOf relation decode`), an algebra part's links (`ResLinks`) give `PartChecks s`, its inner check holding
on every agreeing list of messages with no appeal to a collision. The outer relation makes the named copy compute S's
residuals at its inputs (`partCols_of_check`), which the output tie reads as the inner check. -/
theorem partChecks_of_res {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  obtain ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.partCols_of_check hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu W vv hw0 hv0 hin)

/-- **An algebra part's links, read off the copy's columns**: `ResLinks` with the input tie's elements those the named
copy's columns hold (`readElems`, which `InCols` always admits: `inCols_readElems`). The staging is as in `ResLinks`;
what is left of the input tie sits in the output tie's premise: S's residuals at the elements the copy reads give the
inner check, which they do once those elements are the held messages' and the coins' coefficients. -/
def ResReads (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

theorem resLinks_of_reads {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReads D decode L shape shows chk s) :
    ResLinks D decode L shape shows chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  refine ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
    fun w ms hms => ?_⟩
  exact ⟨Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    fun _ he => Flock.Readable.readElems_ge he, fun _ he => Flock.Readable.readElems_ge he,
    Flock.Readable.inCols_readElems ns nv _, hlink w ms hms⟩

theorem partChecks_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResReads D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s :=
  partChecks_of_res hDO (resLinks_of_reads hR)

/-- **A rec-open part's links** (a code fact about V*'s staging, the layout and the inner check at part `s`, none of
it the circuit's semantics): wherever a V* statement `xO` of a compiled statement `x` of the shape shows the coins `c`
for part `s`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words one of whose tables
  `RecOpen.check` accepts, its unit named for `LANES` lanes and height `Ht` (`RecOpen.params`);
* the input tie: the statement names a copy of that table's unit (block `o`, VU `g`, copy `u`) whose input columns, on
  every witness and every list of inner messages that agrees with the messages the part holds, hold a row, its
  siblings, the running sums, the coefficients and the position's bits (`OpenIns`), under a Merkle scheme whose node
  is SHA-512 of its two sides;
* the output tie: the climb and the consistency term on that copy's output columns (`OpenOuts`) give the inner check
  `chk s`. -/
def OpenLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → chk s x c ms)

/-- **`PartChecks` at a rec-open part, from the shipped circuit's semantics**: at the deployed parser
(`Readable.parseOf relation decode`), a rec-open part's links (`OpenLinks`) give `PartChecks s`, its inner check
holding on every agreeing list of messages with no appeal to a collision. The outer relation makes the named copy
climb from the row's leaf and compute the consistency term (`climb_of_check`), which the output tie reads as the inner
check. -/
theorem partChecks_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLinks D decode L shape shows chk s) : PartChecks H D DO L shape shows chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

/-- **`PartLogs` at a part whose inner check logs no call**: an algebra part's inner check (its residuals) makes no
SHA-512 call, so at `logs s = []` every call it logs is recorded. -/
theorem partLogs_of_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} (h : ∀ x c ms, logs s x c ms = []) :
    PartLogs H D DO L shape shows logs s := fun _ _ _ _ _ _ _ _ => Or.inl fun _ _ _ _ e he => by
  rw [h] at he
  cases he

/-- **A rec-open part's links for its calls** (`OpenLinks` with the inner check's calls for the inner check): the
staging and the input tie as in `OpenLinks`, and the output tie: the climb and the consistency term on the copy's
output columns (`OpenOuts`) put every call the part's inner check makes (`logs s`, the climb's SHA-512 evaluations)
among the calls the part records, which read the copy's SHA-512 gadgets. -/
def OpenLogLinks (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → ∀ e ∈ logs s x c ms, e ∈ L.calls x s w)

/-- **`PartLogs` at a rec-open part, from the shipped circuit's semantics**: at the deployed parser, a rec-open part's
links for its calls (`OpenLogLinks`) give `PartLogs s` with no appeal to a collision. The outer relation makes the
named copy climb from the row's leaf (`climb_of_check`), which the output tie reads as the check's calls recorded. -/
theorem partLogs_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLogLinks D decode L shape shows logs s) : PartLogs H D DO L shape shows logs s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hms => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms hms
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

end Facts

/-! ## Openings through registered climbs -/

/-- **A climb at `H`** from `a`: each step hashes its encoding of the value so far (`f a`, a frame-v3 node's frame of
the value with its sibling, level and index). -/
def chainH (H : Msg → Msg) : List (Msg → Msg) → Msg → Msg
  | [], a => a
  | f :: fs, a => chainH H fs (H (f a))

/-- The inputs a climb hashes, in order. -/
def chainIns (H : Msg → Msg) : List (Msg → Msg) → Msg → List Msg
  | [], _ => []
  | f :: fs, a => f a :: chainIns H fs (H (f a))

/-- **Two climbs' steps keep their values apart**: at each level, the two steps' encodings of two values are equal only
where the values are (a frame-v3 node holds the value climbing whole, at its level's side). -/
def StepsApart (fs gs : List (Msg → Msg)) : Prop :=
  List.Forall₂ (fun f g => ∀ u v, f u = g v → u = v) fs gs

/-- **Two climbs from different values to one value collide**: at the last level where their values differ, they hash
two different inputs to one output. -/
theorem chain_coll (H : Msg → Msg) {fs gs : List (Msg → Msg)} (h : StepsApart fs gs) :
    ∀ {a b : Msg}, a ≠ b → chainH H fs a = chainH H gs b →
      ∃ u ∈ chainIns H fs a, ∃ v ∈ chainIns H gs b, u ≠ v ∧ H u = H v := by
  induction h with
  | nil => intro a b hab heq; exact absurd heq hab
  | @cons f g fs gs hfg _ ih =>
    intro a b hab heq
    simp only [chainH] at heq
    by_cases hH : H (f a) = H (g b)
    · exact ⟨f a, by simp [chainIns], g b, by simp [chainIns], fun e => hab (hfg a b e), hH⟩
    · obtain ⟨u, hu, v, hv, huv⟩ := ih hH heq
      exact ⟨u, by simp [chainIns, hu], v, by simp [chainIns, hv], huv⟩

/-- **Two climbs' collision is one the finders charge**: where every input either climb hashes is one the V* setup
evaluates at `xO` or one of the part's recorded calls, which are true evaluations, a collision between the climbs is a
registered tie (`SetupColl`, `Br`'s) or one among the recorded calls (`Be`'s). -/
theorem coll_of_chains {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {xO : Msg} {cs : List (Msg × Msg)}
    (htrue : ∀ e ∈ cs, H e.1 = e.2) {fs gs : List (Msg → Msg)} {a b : Msg}
    (hin : ∀ u ∈ chainIns H fs a ++ chainIns H gs b, u ∈ D.setup xO ∨ (u, H u) ∈ cs)
    (h : ∃ u ∈ chainIns H fs a, ∃ v ∈ chainIns H gs b, u ≠ v ∧ H u = H v) :
    (findColl cs).isSome ∨ SetupColl H D xO cs := by
  obtain ⟨u, hu, v, hv, hne, hH⟩ := h
  have hsu : ∀ {y}, y ∈ D.setup xO → (y, H y) ∈ setupCalls H D [xO] := fun hy => by
    simp only [setupCalls, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_map]
    exact ⟨_, hy, rfl⟩
  rcases hin u (List.mem_append_left _ hu) with hu' | hu'
  · refine Or.inr (findCollIn_isSome (e := (u, H u)) (e' := (v, H v)) (hsu hu') ?_ (fun e => hne e.symm) hH.symm)
    rcases hin v (List.mem_append_right _ hv) with hv' | hv'
    · exact List.mem_append_left _ (hsu hv')
    · exact List.mem_append_right _ hv'
  · rcases hin v (List.mem_append_right _ hv) with hv' | hv'
    · exact Or.inr (findCollIn_isSome (e := (v, H v)) (e' := (u, H u)) (hsu hv') (List.mem_append_right _ hu') hne hH)
    · exact Or.inl (findColl_isSome_mem H cs htrue u v (List.mem_map.2 ⟨_, hu', rfl⟩) (List.mem_map.2 ⟨_, hv', rfl⟩)
        ⟨hne, hH⟩)

/-- **Part `s` opens what it holds to a public digest** (a code fact about part `s`'s circuit and the layout):
wherever the part's outer relation holds at a V* statement `xO` of a compiled statement `x` of the shape that shows
coins for part `s`, each inner round's message the part holds opens under `opening` the digest `dig x i xO` that `xO`'s
public file names for that round (the `b ‖ c` the part's hm96 row computes), through calls the part records. -/
def HeldOpens {Coin Col T : Type} (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (opening : Opening) (dig : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ c, shows s x xO c → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ cs, cs ⊆ L.calls x s w ∧ opening.opens (dig x i xO) m cs

/-- **The setup climbs each held round's digest and commit string to one root** (a code fact about the V* setup at
part `s`, its registered reads: `Registered.opens` at the digest, and the root the setup rebuilds from the round's
commit string): wherever the part's outer relation holds at a V* statement `xO` of a compiled statement `x` of the
shape that shows coins for part `s`, at each inner round the part holds a message at, the digest `dig x i xO` and the
commit string `cmt x i xO` climb to one value through steps that keep their values apart (`StepsApart`), and every
input either climb hashes is one the setup evaluates at `xO` (`D.setup xO`) or one the part records. -/
def RegClimbs {Coin Col T : Type} (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt dig : Msg → ℕ → Msg → Msg) (s : ℕ) :
    Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ c, shows s x xO c → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ fs gs : List (Msg → Msg), StepsApart fs gs ∧ chainH H fs (dig x i xO) = chainH H gs (cmt x i xO) ∧
        ∀ u ∈ chainIns H fs (dig x i xO) ++ chainIns H gs (cmt x i xO), u ∈ D.setup xO ∨ (u, H u) ∈ L.calls x s w

/-- **`PartOpens` through registered climbs**, at any part of any shape: a held message opens the round's digest
through recorded calls (`HeldOpens`); where that digest is the round's commit string, it opens the commit string, and
where it isn't, the setup's two climbs to one root collide (`chain_coll`), a registered tie or a collision among the
recorded calls (`coll_of_chains`). -/
theorem partOpens_of_climbs {Coin Col T : Type} {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {opening : Opening}
    {cmt dig : Msg → ℕ → Msg → Msg} {s : ℕ} (hO : HeldOpens D DO L shape shows opening dig s)
    (hC : RegClimbs H D DO L shape shows cmt dig s) : PartOpens H D DO L shape shows opening cmt s := by
  intro x st hst hf xO w hck htrue
  by_cases hcoll : (findColl (L.calls x s w)).isSome ∨ SetupColl H D xO (L.calls x s w)
  · exact Or.inr hcoll
  refine Or.inl fun c hc i hi m hm => ?_
  obtain ⟨cs, hcs, ho⟩ := hO x st hst hf xO w hck c hc i hi m hm
  by_cases hd : dig x i xO = cmt x i xO
  · exact ⟨cs, hcs, hd ▸ ho⟩
  · obtain ⟨fs, gs, hap, heq, hin⟩ := hC x st hst hf xO w hck c hc i hi m hm
    exact absurd (coll_of_chains htrue hin (chain_coll H hap hd heq)) hcoll

end Claims.Recursion.VStar
