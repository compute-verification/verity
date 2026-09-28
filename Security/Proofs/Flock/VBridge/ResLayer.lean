import Proofs.Flock.VBridge.AlgHolds

/-!
# A part's residuals at the columns its rows read, its outputs in the layer

`decode_check` asks every row-port column of the verifier's `rec-residuals` unit for a registered bit of `w`. A column
no computed row reads (`ReadCol`) is no wire of the unit (the copy's rows read only through `a` and `b`), so nothing
reaches it from acceptance, and nothing needs to: the rows' values elsewhere do not depend on it (`sat_of_read`). Here
the row ports are asked for `w` only at the columns some computed row reads, and the outputs are read in the layer
(`copyVal`) rather than at registered reads (`RegOut`).

* `sat_of_read`: column values agreeing with satisfying ones at every read column and every computed one satisfy the
  rows. `rowCol_inj`, `rowCol_lt_nIn`: the row ports' columns are distinct input columns of the verifier's unit.
* `unit_partR`, `decode_resUnitR`, `decode_checkR`, `decode_partR`: `unit_part`, `decode_resUnit`, `decode_check`,
  `decode_part` with the row ports given at the read columns only (`InColsR`): the copy's columns, with each unread row
  column set to its bit of `w`, still satisfy the rows, and agree with the copy's past the inputs.
* `ResZeroAtL`, `partCols_of_resZeroAtL`, `algebraHolds_of_decodeL`: `ResZeroAt` with the row ports registered at the
  read columns and the outputs 0 in the layer, and `AlgebraHolds` from it as from `ResZeroAt`.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open Flock.RecResiduals (portSizes groups partS)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.PartS (wIdx)
open FlockVBridge.Compiled (RepData AlgebraHolds)
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-! ## The columns the rows read -/

/-- Some computed row of `R` reads column `col`. -/
def ReadCol (R : Rows) (col : ℕ) : Prop := ∃ (i : Fin R.nComp) (x : Fin R.w), x ∈ R.a i ++ R.b i ∧ x.val = col

/-- **The rows hold at values that agree at every read and computed column.** -/
theorem sat_of_read {R : Rows} {z z' : Fin R.w → Bool} (hz : R.Sat z)
    (h : ∀ x : Fin R.w, (ReadCol R x.val ∨ R.nIn ≤ x.val) → z' x = z x) : R.Sat z' := by
  intro i
  have hab : ∀ x ∈ R.a i ++ R.b i, z' x = z x := fun x hx => h x (Or.inl ⟨i, x, hx, rfl⟩)
  rw [h (R.comp i) (Or.inr (by simp [Rows.comp])), hz i,
    xorSum_congr fun x hx => (hab x (List.mem_append_left _ hx)).symm,
    xorSum_congr fun x hx => (hab x (List.mem_append_right _ hx)).symm]

theorem le_rowCol : ∀ (ps : List (String × ℕ)) (k q : ℕ), k ≤ rowCol ps k q
  | [], _, _ => le_refl _
  | p :: ps, k, q => by
    simp only [rowCol]
    split
    · omega
    · have := le_rowCol ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2)
      omega

/-- The row ports' columns are distinct. -/
theorem rowCol_inj : ∀ (ps : List (String × ℕ)) (k q q' : ℕ), q < 128 * (ps.map (·.2)).sum →
    q' < 128 * (ps.map (·.2)).sum → rowCol ps k q = rowCol ps k q' → q = q'
  | [], _, _, _, hq, _, _ => by simp at hq
  | p :: ps, k, q, q', hq, hq', h => by
    simp only [List.map_cons, List.sum_cons] at hq hq'
    have hp := le_portWords p.2
    simp only [rowCol] at h
    by_cases h1 : q < 128 * p.2 <;> by_cases h2 : q' < 128 * p.2 <;> simp only [h1, h2, ↓reduceIte] at h
    · omega
    · have := le_rowCol ps (k + 16 * VStar.portWordsOf p.2) (q' - 128 * p.2)
      omega
    · have := le_rowCol ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2)
      omega
    · have := rowCol_inj ps _ _ _ (by omega) (by omega) h
      omega

/-- **The row ports' columns are input columns** of the verifier's `rec-residuals` unit: below `innerIn`, which its
input rows round up (`le_groupWords`). -/
theorem rowCol_lt_nIn {c : Flock.Circuit} (hU : Built c.unit) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) {q : ℕ} (hq : q < 128 * ns.sum) :
    rowCol (partRep ns nv).ports 0 q < (Rows.ofNet hU).nIn := by
  have hlt := rowCol_lt (partRep ns nv).ports 0 q (by rw [ports_snd]; exact hq)
  have hN : (partIn ns nv).foldl (· + ·) 0 = innerIn (partRep ns nv) := by
    rw [partIn, foldl_replicate, innerIn_partRep]
    exact Nat.mul_comm _ _
  have hi := innerIn_eq (partRep ns nv)
  unfold vBase at hi
  rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
  refine lt_of_lt_of_le ?_ (Flock.HmNets.le_groupWords _)
  rw [hN]
  omega

/-- `InCols` with the row ports asked only at the columns some computed row of `R` reads. -/
def InColsR (R : Rows) (S : VStar.RepS) (w vv : ℕ → F128) (v : ℕ → Bool) : Prop :=
  (∀ q < 128 * S.gf.nw, ReadCol R (rowCol S.ports 0 q) → v (rowCol S.ports 0 q) = wbit w q) ∧
    ∀ q < 128 * S.gf.nv, v (vBase S + q) = wbit vv q

/-- **The verifier's laid-out `rec-residuals` unit computes its residuals from its read columns** (`laid_part`): when
its rows hold at `z` with 1 at the constant, and `z` holds `w` at the row ports' columns some computed row reads and `vv`
on `v`'s, the output columns hold S's residuals. `z` with each unread row column set to its bit of `w` still satisfies
the rows (`sat_of_read`) and agrees with `z` past the inputs. -/
theorem laid_partR {net : Flock.Net} (hB : Built net) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ)
    (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS) {name : String}
    (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok net) (hu : net.unitLog ≤ 32)
    (hlt : ∀ q < 128 * ns.sum, rowCol (partRep ns nv).ports 0 q < (Rows.ofNet hB).nIn)
    {z : Fin (Rows.ofNet hB).w → Bool} (hz : (Rows.ofNet hB).Sat z) (hone : z (Rows.ofNet hB).one = true)
    (w vv : ℕ → F128) (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InColsR (Rows.ofNet hB) (partRep ns nv) w vv (colVal z)) :
    PartCols S w vv (colVal z) (partBase S ns nv) := by
  classical
  obtain ⟨Q, hQ⟩ : ∃ Q : ℕ → Prop, ∀ x, Q x ↔ x < (Rows.ofNet hB).nIn ∧ ¬ ReadCol (Rows.ofNet hB) x ∧
      ∃ q, q < 128 * ns.sum ∧ rowCol (partRep ns nv).ports 0 q = x := ⟨_, fun _ => Iff.rfl⟩
  obtain ⟨z', hz'⟩ : ∃ z' : Fin (Rows.ofNet hB).w → Bool, ∀ x, z' x =
      if h : Q x.val then wbit w (Classical.choose ((hQ _).1 h).2.2) else z x := ⟨_, fun _ => rfl⟩
  have hzz : ∀ x : Fin (Rows.ofNet hB).w, ¬ Q x.val → z' x = z x := fun x h => by rw [hz', dite_eq_right h]
  have hzq : ∀ (x : Fin (Rows.ofNet hB).w) (h : Q x.val), z' x = wbit w (Classical.choose ((hQ _).1 h).2.2) :=
    fun x h => by rw [hz', dite_eq_left h]
  have hcol : ∀ x, ¬ Q x → colVal z' x = colVal z x := by
    intro x hx
    unfold colVal
    split_ifs with h
    · exact hzz _ hx
    · rfl
  have hs' : (Rows.ofNet hB).Sat z' := sat_of_read hz fun x hx => hzz x fun hq =>
    hx.elim ((hQ _).1 hq).2.1 fun h => absurd ((hQ _).1 hq).1 (Nat.not_lt.2 h)
  have hone' : z' (Rows.ofNet hB).one = true := by
    rw [hzz (Rows.ofNet hB).one fun hq => absurd ((hQ _).1 hq).1 (by simp [Rows.one])]
    exact hone
  have hbase : (Rows.ofNet hB).nIn ≤ partBase S ns nv := by
    rw [Rows.ofNet_nIn, inWords_unitLaid hn]
    exact (min_le_left _ _).trans (Decode.inRows_le_outStart _ _ _)
  have hout : ∀ t, ¬ Q (partBase S ns nv + t) := fun t hq =>
    absurd ((hQ _).1 hq).1 (Nat.not_lt.2 (hbase.trans (Nat.le_add_right _ _)))
  have hL := laidRows_ofNet hB hn hu hs' hone'
  rw [inWords_unitLaid hn] at hL
  have hP := laid_part S ns nv hres _ hL w vv hw0 hv0 ⟨fun q hq => ?_, fun q hq => ?_⟩
  · exact ⟨fun r hr k hk => (hcol _ (hout _)).symm.trans (hP.1 r hr k hk),
      fun x hx1 hx2 => (hcol _ (hout _)).symm.trans (hP.2 x hx1 hx2)⟩
  · by_cases hr : ReadCol (Rows.ofNet hB) (rowCol (partRep ns nv).ports 0 q)
    · rw [hcol (rowCol (partRep ns nv).ports 0 q) fun hq' => ((hQ _).1 hq').2.1 hr]
      exact hin.1 q hq hr
    · have hq' : Q (rowCol (partRep ns nv).ports 0 q) := (hQ _).2 ⟨hlt q hq, hr, q, hq, rfl⟩
      have hxw : rowCol (partRep ns nv).ports 0 q < (Rows.ofNet hB).w := by
        have := hlt q hq
        simp only [Rows.w]
        omega
      unfold colVal
      rw [dite_eq_left hxw, hzq ⟨_, hxw⟩ hq']
      obtain ⟨hq0, he⟩ := Classical.choose_spec ((hQ _).1 hq').2.2
      exact congrArg (wbit w)
        (rowCol_inj _ 0 _ _ (by rw [ports_snd]; exact hq0) (by rw [ports_snd]; exact hq) he)
  · have hnq : ¬ Q (vBase (partRep ns nv) + q) := fun hQq => by
      obtain ⟨-, -, q', hq', he⟩ := (hQ _).1 hQq
      have := rowCol_lt (partRep ns nv).ports 0 q' (by rw [ports_snd]; exact hq')
      unfold vBase at he
      omega
    rw [hcol _ hnq]
    exact hin.2 q hq

/-- **A copy of the verifier's `rec-residuals` unit computes its residuals from its read columns** (`laid_partR` at
`sat_copy`), as `unit_part` from all its row columns. -/
theorem unit_partR {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (tr : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : tr (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS)
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok net) (hu : net.unitLog ≤ 32)
    (hlt : ∀ q < 128 * ns.sum, rowCol (partRep ns nv).ports 0 q < (Rows.ofNet hB).nIn)
    (w vv : ℕ → F128) (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hin : InColsR (Rows.ofNet hB) (partRep ns nv) w vv (copyVal I tr hj)) :
    PartCols S w vv (copyVal I tr hj) (partBase S ns nv) := by
  obtain ⟨hs, h1⟩ := sat_copy I tr hj
  exact laid_partR hB S ns nv hres hn hu hlt hs (h1.trans hone) w vv hw0 hv0 hin

section Decode

open FlockVBridge.Decode

variable {mPts nU J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nU} (zo : ZkOuter mPts nU J Reg' c hU hscope Lv)

/-- **`decode_resUnit` from the read row columns**: the row ports registered bits of `w` only at the columns some
computed row reads (`unit_partR` at `fill`). -/
theorem decode_resUnitR {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (hres : 8 * S.res.length ≤ RecResiduals.RES_OUT_WORDS)
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) (hu : c.unit.unitLog ≤ 32)
    (w vv : ℕ → Flock.F128) (hw0 : ∀ e, ns.sum ≤ e → w e = 0) (hv0 : ∀ e, nv ≤ e → vv e = 0)
    (hw : ∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit w q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit vv q)) :
    PartCols S w vv (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  have h1 : fill zo bits ((keyInst hU nU (Layout.outsOf c hU nU) u).wire ((Rows.ofNet hU).stack c.upv).one) = true := by
    rw [fill_one zo hA _ hj, keyInst_one]
    exact hone
  obtain ⟨hr, hz⟩ := unit_partR hU (keyInst hU nU (Layout.outsOf c hU nU) u) (fill zo bits) hj h1 S ns nv hres hn hu
    (fun q hq => rowCol_lt_nIn hU hn hq) w vv hw0 hv0
    ⟨fun q hq hrd => copyVal_regBit zo bits hj (hw q hq hrd), fun q hq => copyVal_regBit zo bits hj (hv q hq)⟩
  have hbase : (Rows.ofNet hU).nIn ≤ partBase S ns nv := by
    rw [Layout.ofNet_nIn_eq, inWords_unitLaid hn]
    exact inRows_le_outStart _ _ _
  refine ⟨fun r hr' k hk => ?_, fun x hx1 hx2 => ?_⟩
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hr r hr' k hk)
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hz x hx1 hx2)

/-- **`decode_check` from the read row columns** (`decode_resUnitR`). -/
theorem decode_checkR {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok false)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128)
    (w : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ ns.sum) (hV : V.size ≤ nv)
    (hw : ∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * nv, RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub V (vOf sh rounds extras comb)) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vOf sh rounds extras comb V[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨S', ns', nv', hpart', hn⟩ := check_part hc hp hsh hh
  rw [hpart] at hpart'
  cases hpart'
  exact ⟨sub wI w, sub V (vOf sh rounds extras comb), sub_lt wI w, sub_lt V _,
    decode_resUnitR zo hA u hj hone S ns nv (part_small hpart) hn hu _ _ (sub_ge w hwI) (sub_ge _ hV) hw hv⟩

/-- **`decode_part` from the read row columns** (`decode_checkR` at `part_sizes`). -/
theorem decode_partR {bits : Fin (Layout.keyN c hU nU) → Bool} (hA : zo.RegAgree bits) (u : Fin nU) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true)
    (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) {S0 : RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0) (hg : groups S0 (portSizes sh) = .ok gs)
    (hpg : p < gs.size) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part sh p = .ok (S, ns, nv)) (hu : c.unit.unitLog ≤ 32)
    (hh : RecResiduals.holoOf c = .ok false)
    (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128) (w : ℕ → Flock.F128)
    (hw : ∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
      RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q))
    (hv : ∀ q < 128 * nv,
      RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) :
    partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S ∧
    ∃ w' v' : ℕ → Flock.F128,
      (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
      (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
      PartCols S w' v' (copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj) (partBase S ns nv) := by
  obtain ⟨hS', -, -⟩ := part_eq hS hg hpg hpart
  obtain ⟨hwI, hV⟩ := part_sizes hS hg hpg hpart
  exact ⟨hS', decode_checkR zo hA u hj hone hc hp hsh hpart hu hh rounds extras comb w _ _ hwI.le hV.le hw hv⟩

/-- **Copy `j` of unit `u` is part `p` of the groups `gs`, its outputs 0 in the layer**: `ResZeroAt`'s facts, with the
row ports registered bits of `w` only at the columns some computed row of the unit reads (`ReadCol`), and every output
column of the part (`16 · RES_OUT_WORDS` bits) a 0 of copy `j` in the layer (`copyVal`) in place of a registered 0
(`RegOut`). -/
def ResZeroAtL (bits : Fin (Layout.keyN c hU nU) → Bool) (u : Fin nU) (j : ℕ) (sh : Shape)
    (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (p : ℕ) (w : ℕ → F128) (rounds : Array (Array F128))
    (extras : Array Claim) (comb : Array F128) : Prop :=
  ∃ (hpg : p < gs.size) (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nU) u) = true ∧
    RecResiduals.check c = .ok () ∧ RecResiduals.holoOf c = .ok false ∧ c.unit.unitLog ≤ 32 ∧
    ∃ m k cl, RecResiduals.params c.unit.name = some (m, k, cl, p) ∧ RecResiduals.shape? m k cl = some sh ∧
    ∃ S ns nv, RecResiduals.part sh p = .ok (S, ns, nv) ∧
      (∀ q < 128 * ns.sum, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 q) →
        RegBit zo u j (rowCol (partRep ns nv).ports 0 q) (wbit (sub (wIdx (portSizes sh) gs[p].2.1) w) q)) ∧
      (∀ q < 128 * nv,
        RegBit zo u j (vBase (partRep ns nv) + q) (wbit (sub gs[p].2.2 (vOf sh rounds extras comb)) q)) ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS,
        copyVal (keyInst hU nU (Layout.outsOf c hU nU) u) bits hj (partBase S ns nv + t) = false

/-- **A part's residuals, zero, from its unit's copy in a good layer** (`decode_partR`), as `partCols_of_resZeroAt`. -/
theorem partCols_of_resZeroAtL {bits : Fin (Layout.keyN c hU nU) → Bool} (hG : zo.Good bits) {u : Fin nU} {j : ℕ}
    {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : structureOf sh = .ok S0)
    (hg : groups S0 (portSizes sh) = .ok gs) {p : ℕ} {w : ℕ → F128} {rounds : Array (Array F128)}
    {extras : Array Claim} {comb : Array F128} (h : ResZeroAtL zo bits u j sh gs p w rounds extras comb) :
    ∃ hpg : p < gs.size, ∃ S', partS S0 (portSizes sh) gs[p].1 gs[p].2.1 gs[p].2.2 = .ok S' ∧
      ∃ (w' v' : ℕ → F128) (col : ℕ → Bool) (base : ℕ),
        (∀ k (hk : k < (wIdx (portSizes sh) gs[p].2.1).size), w' k = w (wIdx (portSizes sh) gs[p].2.1)[k]) ∧
        (∀ k (hk : k < gs[p].2.2.size), v' k = vOf sh rounds extras comb gs[p].2.2[k]) ∧
        PartCols S' w' v' col base ∧ ∀ t < 128 * S'.res.length, col (base + t) = false := by
  obtain ⟨hpg, hj, hone, hc, hh, hu, m, k, cl, hp, hsh, S, ns, nv, hpart, hw, hv, hz⟩ := h
  obtain ⟨hS', w', v', hw', hv', hcols⟩ :=
    decode_partR zo hG.2 u hj hone hc hp hsh hS hg hpg hpart hu hh rounds extras comb w hw hv
  have hsmall := part_small hpart
  exact ⟨hpg, S, hS', w', v', _, _, hw', hv', hcols, fun t ht => hz t (by omega)⟩

end Decode

section Sessions

open FlockVBridge.Decode

variable {T : ℕ} {Vs : Fin T → VStmt} {Reg' : Type}

/-- **`AlgebraHolds` from each part's unit, its outputs read in the layer**: `algebraHolds_of_decode` at
`ResZeroAtL`. -/
theorem algebraHolds_of_decodeL (zs : (s : Fin T) → (Vs s).Outer Reg') (L : (s : Fin T) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (L s)) {sh : Shape} {d : RepData} {S : RecAlgebra.GfS} (hS : structureOf sh = .ok S)
    (hbits : S.BitsOk) {v : Array F128} {info kappa}
    (hvr : verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa))
    (hw : d.w.size = S.nw) (hc : d.comb.size = 64) (hv : v.size = S.nv)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S (portSizes sh) = .ok gs)
    (sess : ℕ → Fin T) (un : (p : ℕ) → Fin (Vs (sess p)).nv) (cp : ℕ → ℕ)
    (h : ∀ p < gs.size,
      ResZeroAtL (zs (sess p)) (L (sess p)) (un p) (cp p) sh gs p (el d.w) d.rounds d.extras d.comb) :
    AlgebraHolds sh d := by
  refine algebraHolds_of_parts hS hbits hvr hw hc hv hg fun p hp => ?_
  obtain ⟨_, S', hS', w', v', col, base, hw', hv', hcols, h0⟩ :=
    partCols_of_resZeroAtL (zs (sess p)) (hG (sess p)) hS hg (h p hp)
  rw [vOf_eq hvr] at hv'
  exact ⟨S', hS', w', v', col, base, hw', hv', hcols, h0⟩

end Sessions

end FlockVBridge.OfVStar
