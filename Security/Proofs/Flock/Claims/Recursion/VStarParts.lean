import Proofs.Flock.Claims.Recursion.VStar
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.Matches
import Proofs.Flock.VBridge.ResUnit

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
* **At the relation** (`relS_of_check`, `layout_of_check`, `partCols_of_check`): `Flock.check` at the deployed parser
  gives every table's layout and satisfied model, so the above at every table, block, VU and copy.
* **`PartChecks` at an algebra part** (`partChecks_of_res`): the circuit's semantics above, with the links that are not
  the circuit's (`ResLinks`): the staging (the V* statement the part's coins are shown on is an accepted algebra part),
  the input tie (a copy's input columns hold the messages the part holds and the coins' coefficients) and the output
  tie (the part's residuals there give its inner check).
* **`PartLogs` at an algebra part** (`partLogs_of_nil`): its inner check, the part's residuals, makes no SHA-512 call,
  so at `logs s = []` the part records every call its check makes.
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
theorem laidRows_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
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
  have hL := FlockVBridge.laidRows_ofNet L.unit_built hl (L.unitLog_le (L.slot_lt hg hu)) hs1
    (hone.trans (by simp [FlockSoundness.bit]))
  have hw : (unitRows st L.unit_built).w = st.c.unit.constPos + 1 := by
    show _ + _ + 1 = _
    rw [L.unitRows_nIn_nComp]
  have hw' : (Rows.ofNet L.unit_built).w = st.c.unit.constPos + 1 := hw
  convert hL using 1
  funext j
  unfold FlockVBridge.colVal copyCol
  split_ifs
  · have e := stackPlace_stackCol (L.model regs) (unitRows st L.unit_built) st.c.upv
      (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) hu ⟨j, by omega⟩
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
theorem partCols_of_satisfies {st : Flock.Stmt} (L : FlockSoundness.Layout st)
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
  have hL := laidRows_of_satisfies L regs hz ho hg hu hn
  rw [inWords_unitLaid hn] at hL
  exact laid_part S ns nv (part_small hpart) _ hL W vv hw0 hv0 hin

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
  exact partCols_of_satisfies L regs hz ho hg hu hc hp hsh hh hpart W vv hw0 hv0 hin

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

/-- **`PartLogs` at a part whose inner check logs no call**: an algebra part's inner check (its residuals) makes no
SHA-512 call, so at `logs s = []` every call it logs is recorded. -/
theorem partLogs_of_nil {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ} (h : ∀ x c ms, logs s x c ms = []) :
    PartLogs H D DO L shape shows logs s := fun _ _ _ _ _ _ _ _ => Or.inl fun _ _ _ _ e he => by
  rw [h] at he
  cases he

end Facts

end Claims.Recursion.VStar
