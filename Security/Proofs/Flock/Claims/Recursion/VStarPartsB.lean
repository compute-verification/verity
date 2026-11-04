import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.Claims.Recursion.VStarBound

/-!
# V*'s algebra and rec-open parts at the bound layout (option (d))

`VStarParts.lean` takes a part's outer relation to the circuit V\* ships at `VStar.Layout` and `VStar.PartChecks`
(`partChecks_of_res`, `partChecks_of_reads`, `partChecks_of_open`). The circuit's semantics there (`partCols_of_check`,
`climb_of_check`) read no layout, so they carry over unchanged; what changes is the input tie, which here reads the
messages the bound layout's check quantifies over (`LayoutB.Agrees`): those that agree with the part's holdings, its
pieces, the commit strings its statement shows (`LayoutB.CmtAgrees`) and its shared rows (`LayoutB.RowsRead`).

* **`PartChecksB` at an algebra part** (`partChecksB_of_res`): `ResLinksB`, `ResLinks` at `LayoutB` with the input tie
  over agreeing messages.
* **The input tie's elements** (`ResReadsB`, `resLinksB_of_reads`, `partChecksB_of_reads`): `ResReads` at `LayoutB`.
* **`PartChecksB` at a rec-open part** (`partChecksB_of_open`): `OpenLinksB`, `OpenLinks` at `LayoutB`.

`PartOpensB` needs no port of `VStar.partOpens_of_climbs`: at the deployed layout it is proved outright
(`RecDeploy.partOpensB_recRead`, through `partOpensB_layoutBTops`), and every route at `layoutBTops` drops it. The
climbs' argument also escapes to a setup collision (`SetupColl`), which `PartOpensB`'s escape doesn't admit.
-/

namespace Claims.Recursion.VStarB

open Claims
open FlockVBridge.OfVStar

/-- **The messages part `s`'s check reads agree with it** (what `PartChecksB` quantifies over), off its witness `w` at
its V\* statement `xO`: at each round it holds whole, at each of its pieces at its offset, at the commit strings `xO`
shows at every round it doesn't fix before the last (`LayoutB.CmtAgrees`), and at the shared rows it holds where the
messages carry them (`LayoutB.RowsRead`). -/
def LayoutB.Agrees (L : LayoutB) (cmt : Msg → ℕ → Msg → Msg) (x : Msg) (s : ℕ) (w : Flock.Wit) (xO : Msg)
    (ms : List Msg) : Prop :=
  (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
    (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) ∧
    L.CmtAgrees cmt x xO ms ∧ L.RowsRead x s w ms

section Facts

variable {Coin Col T : Type}

/-- **An algebra part's links at the bound layout** (`VStar.ResLinks` at `LayoutB`; a code fact about V*'s staging,
the layout and the inner check at part `s`, none of it the circuit's semantics): wherever a V* statement `xO` of a
compiled statement `x` of the shape shows the coins `c` for part `s`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words one of whose tables
  `RecResiduals.check` accepts as part `j` of a shape's algebra (plain), `S` with row ports of `ns` elements and a `v`
  of `nv`;
* the input tie: the statement names a copy of that table's unit (block `o`, VU `g`, copy `u`) whose input columns, on
  every witness and every list of inner messages that agrees with the part (`LayoutB.Agrees`), hold elements `W` on
  the row ports and coefficients `vv` on `v`'s (`InCols`), zero past `ns`'s total and `nv`;
* the output tie: S's residuals at `W`, `vv` on that copy's output columns (`PartCols`) give the inner check `chk s`. -/
def ResLinksB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            ∃ W vv : ℕ → _root_.Flock.F128, (∀ e, ns.sum ≤ e → W e = 0) ∧ (∀ e, nv ≤ e → vv e = 0) ∧
              InCols (partRep ns nv) W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) ∧
              (PartCols S W vv (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos)
                (partBase S ns nv) → chk s x c ms)

/-- **`PartChecksB` at an algebra part, from the shipped circuit's semantics** (`VStar.partChecks_of_res` at
`LayoutB`): at the deployed parser (`Readable.parseOf relation decode`), an algebra part's links (`ResLinksB`) give
`PartChecksB s`, its inner check holding on every agreeing list of messages with no appeal to a collision. The outer
relation makes the named copy compute S's residuals at its inputs (`partCols_of_check`), which the output tie reads as
the inner check. -/
theorem partChecksB_of_res {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResLinksB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hhd hpc hcm hrr => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  obtain ⟨W, vv, hw0, hv0, hin, hout⟩ := hlink w ms ⟨hhd, hpc, hcm, hrr⟩
  rw [hDO] at hck
  exact hout (Flock.Readable.partCols_of_check hck hx ht hpi htl hrc hp hsh hh hpart ho hg hu W vv hw0 hv0 hin)

/-- **An algebra part's links at the bound layout, read off the copy's columns** (`VStar.ResReads` at `LayoutB`):
`ResLinksB` with the input tie's elements those the named copy's columns hold (`readElems`, which `InCols` always
admits: `inCols_readElems`). S's residuals at the elements the copy reads give the inner check, which they do once
those elements are the agreeing messages' and the coins' coefficients. -/
def ResReadsB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ (m k cl j : ℕ) (sh : _root_.Flock.Shape) (S : _root_.Flock.RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ),
        _root_.Flock.RecResiduals.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, j) ∧
        _root_.Flock.RecResiduals.shape? m k cl = some sh ∧ _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok false ∧
        _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            PartCols S (Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t ht w o g u))
              (Flock.Readable.copyOf p t ht w o g u) (partBase S ns nv) → chk s x c ms

theorem resLinksB_of_reads {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (hR : ResReadsB D decode L shape shows cmt chk s) :
    ResLinksB D decode L shape shows cmt chk s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    hR x st hst hf xO c hc
  refine ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, hrc, hp, hsh, hh, hpart, o, g, u, ho, hg, hu,
    fun w ms hms => ?_⟩
  exact ⟨Flock.Readable.rowsRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    Flock.Readable.vRead ns nv (Flock.Readable.copyOf p t htl w o g u),
    fun _ he => Flock.Readable.readElems_ge he, fun _ he => Flock.Readable.readElems_ge he,
    Flock.Readable.inCols_readElems ns nv _, hlink w ms hms⟩

/-- **`PartChecksB` at an algebra part, from the elements its copy reads** (`VStar.partChecks_of_reads` at
`LayoutB`). -/
theorem partChecksB_of_reads {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ResReadsB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s :=
  partChecksB_of_res hDO (resLinksB_of_reads hR)

/-- **A rec-open part's links at the bound layout** (`VStar.OpenLinks` at `LayoutB`; a code fact about V*'s staging,
the layout and the inner check at part `s`, none of it the circuit's semantics): wherever a V* statement `xO` of a
compiled statement `x` of the shape shows the coins `c` for part `s`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words one of whose tables
  `RecOpen.check` accepts, its unit named for `LANES` lanes and height `Ht` (`RecOpen.params`);
* the input tie: the statement names a copy of that table's unit (block `o`, VU `g`, copy `u`) whose input columns, on
  every witness and every list of inner messages that agrees with the part (`LayoutB.Agrees`), hold a row, its
  siblings, the running sums, the coefficients and the position's bits (`OpenIns`), under a Merkle scheme whose node
  is SHA-512 of its two sides;
* the output tie: the climb and the consistency term on that copy's output columns (`OpenOuts`) give the inner check
  `chk s`. -/
def OpenLinksB (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      ∃ t, ∃ ht : t < p.sts.size, ∃ LANES Ht : ℕ,
        _root_.Flock.RecOpen.check p.sts[t].c = .ok () ∧
        _root_.Flock.RecOpen.params p.sts[t].c.unit.name = some (LANES, Ht) ∧
        ∃ o g u, o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) ∧ g < p.sts[t].c.g ∧ u < p.sts[t].c.upv ∧
          ∀ (w : Flock.Wit) (ms : List Msg), L.Agrees cmt x s w xO ms →
            ∃ (msch : _root_.Flock.MerkleScheme) (r a cf : Array _root_.Flock.F128) (rest : ByteArray) (idx : ℕ)
              (ss : List ByteArray), (∀ l r, msch.node l r = _root_.Flock.Sha512.hash (l ++ r)) ∧
              Flock.Readable.OpenIns (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) LANES Ht r a cf rest
                idx ss ∧
              (Flock.Readable.OpenOuts (Flock.Readable.copyCol p.sts.size t w p.sts[t].c.kLog o
                (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u)) p.sts[t].c.unit.constPos) msch LANES Ht r a cf
                rest idx ss → chk s x c ms)

/-- **`PartChecksB` at a rec-open part, from the shipped circuit's semantics** (`VStar.partChecks_of_open` at
`LayoutB`): at the deployed parser, a rec-open part's links (`OpenLinksB`) give `PartChecksB s`, its inner check
holding on every agreeing list of messages with no appeal to a collision. The outer relation makes the named copy
climb from the row's leaf and compute the consistency term (`climb_of_check`), which the output tie reads as the inner
check. -/
theorem partChecksB_of_open {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : OpenLinksB D decode L shape shows cmt chk s) : PartChecksB H D DO L shape shows cmt chk s := by
  intro x st hst hf xO w hck _
  refine Or.inl fun c hc ms hhd hpc hcm hrr => ?_
  obtain ⟨p, hx, ht, hpi, t, htl, LANES, Ht, hrec, hp, o, g, u, ho, hg, hu, hlink⟩ := hR x st hst hf xO c hc
  obtain ⟨msch, r, a, cf, rest, idx, ss, hmsch, ⟨hr, ha, hcf, hrest, hss, hrow, haux, hacc, hcoef, hdir⟩, hout⟩ :=
    hlink w ms ⟨hhd, hpc, hcm, hrr⟩
  rw [hDO] at hck
  exact hout (Flock.Readable.climb_of_check hck hx ht hpi htl ho hg hu rfl msch hmsch hrec hp r a cf rest idx ss hr
    ha hcf hrest hss hrow haux hacc hcoef hdir)

end Facts

end Claims.Recursion.VStarB
