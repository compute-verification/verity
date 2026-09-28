import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Layout.OutSite
import Proofs.Flock.Soundness.Discharge.Layout.OutShape

set_option Elab.async false

/-!
# The units' committed gates at `keyProg`'s places: the population's returned outputs

`keyProg_valPinned` takes `hout`: at a unit's committed gate, a satisfying witness holds the registration's bit. Here
that bit is `outBitOf`: the gate's column in its unit's instance is a template instance copy `u'`'s computed column
`outCol j t` (column `16 j + t` of the unit's returned-output group; `OutColAt`, read back by `outDecode`), and the bit
is bit `t` of the population's output word `lo[j]` of the unit, for the copy's leaf's outputs `lo`. `keyProg_hout`
proves `hout` for it, in an untyped circuit whose unit returns the outputs (`c.outNet = c.unitNet`), from:

* the stacked column's place in the block (`stackPlace_val_comp`, `col_val_comp`): the template instance's slot
  `vu · upv + u'` at column `outCol j t`, which is region "out"'s `outAddr` (`out_site`, at each drawn unit's site,
  `BcSite`);
* `OutsOut`: every committed gate a unit computes is on such a column (a condition on the circuit's outputs `outs`: the
  returned outputs' bits, and nothing else a unit computes);
* `OutShape`, which the parse checked (`outShape_of_parse`): the output group's words are a power of two, and each
  copy's leaf's outputs are 16-bit ports, words below `nOut`, that fit in them.

`keyRowsOut` is `keyRows` with `outBitOf` for the registration's bit and `keyProg_hout` for its `hout`.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model Audit Binding Placed

/-! ## A stacked computed column's place -/

/-- **Copy `u'`'s computed column `c0`, stacked and placed**, is `base u' + c0`. -/
theorem stackPlace_val_comp (S : Model.Statement) (R : FlockSoundness.Rows) (k : ℕ) (base : ℕ → ℕ)
    (hb : ∀ u < k, base u + (R.nIn + R.nComp) < 2 ^ S.kLog) {u' c0 : ℕ} (hu' : u' < k) (h1 : R.nIn ≤ c0)
    (h2 : c0 < R.nIn + R.nComp) (x : Fin (R.stackW k))
    (hx : x.val = k * R.nIn + (u' * (R.nComp + 1) + (1 + (c0 - R.nIn)))) :
    (stackPlace S R k base hb x).val = base u' + c0 := by
  have hc0 : c0 < R.w := by simp only [FlockSoundness.Rows.w]; omega
  have e : x = R.stackCol k hu' ⟨c0, hc0⟩ := by
    apply Fin.ext
    simp only [FlockSoundness.Rows.stackCol, show ¬ c0 < R.nIn by omega, h2, ↓reduceDIte]
    exact hx
  rw [e, stackPlace_stackCol]
  rfl

/-- **An accepted statement's template instance column of copy `u'`'s computed column `c0`** is column `c0` of the unit
slot `vu · upv + u'`. -/
theorem col_val_comp {T : Model.Statement} (a : Placed.Accepted T) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' c0 : ℕ} (hu' : u' < a.st.c.upv) (h1 : (unitRows a.st a.layout.unit_built).nIn ≤ c0)
    (h2 : c0 < (unitRows a.st a.layout.unit_built).nIn + (unitRows a.st a.layout.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.layout.unit_built).nIn +
      (u' * ((unitRows a.st a.layout.unit_built).nComp + 1) + (1 + (c0 - (unitRows a.st a.layout.unit_built).nIn)))) :
    (a.col hg x).val = a.st.c.unitRange.slot (g * a.st.c.upv + u') + c0 := by
  unfold Placed.Accepted.col
  rw [Fin.val_cast]
  exact stackPlace_val_comp _ _ _ _ _ hu' h1 h2 x hx

/-! ## The returned outputs' columns -/

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- Column `16 j + t` of the unit's returned-output group, in the unit's rows. -/
def outCol (j t : ℕ) : ℕ := (c.unit.outGroups.getD 0 default).col * WORD + (16 * j + t)

/-- **Stacked column `x` of a template instance is a returned output's bit**: copy `u'`'s computed column `outCol j t`,
for output `j` of the copy's leaf and bit `t < 16`. -/
def OutColAt (x u' j t : ℕ) : Prop :=
  u' < c.upv ∧ j < (c.leavesOut.getD u' #[]).size ∧ t < 16 ∧ (Rows.ofNet hU).nIn ≤ outCol c j t ∧
    outCol c j t < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp ∧
    x = c.upv * (Rows.ofNet hU).nIn + (u' * ((Rows.ofNet hU).nComp + 1) + (1 + (outCol c j t - (Rows.ofNet hU).nIn)))

/-- The copy, output and bit a stacked computed column is on (`OutColAt` read back). -/
noncomputable def outDecode (x : ℕ) : ℕ × ℕ × ℕ :=
  ((x - c.upv * (Rows.ofNet hU).nIn) / ((Rows.ofNet hU).nComp + 1),
    ((Rows.ofNet hU).nIn + ((x - c.upv * (Rows.ofNet hU).nIn) % ((Rows.ofNet hU).nComp + 1) - 1) -
      (c.unit.outGroups.getD 0 default).col * WORD) / 16,
    ((Rows.ofNet hU).nIn + ((x - c.upv * (Rows.ofNet hU).nIn) % ((Rows.ofNet hU).nComp + 1) - 1) -
      (c.unit.outGroups.getD 0 default).col * WORD) % 16)

variable {c hU} in
theorem outDecode_of {x u' j t : ℕ} (h : OutColAt c hU x u' j t) : outDecode c hU x = (u', j, t) := by
  obtain ⟨-, -, ht, h1, h2, hx⟩ := h
  obtain ⟨hd, hm⟩ := div_mod_copy (u := u') (show 1 + (outCol c j t - (Rows.ofNet hU).nIn) < (Rows.ofNet hU).nComp + 1
    by omega)
  have hv : x - c.upv * (Rows.ofNet hU).nIn = u' * ((Rows.ofNet hU).nComp + 1) + (1 + (outCol c j t - (Rows.ofNet hU).nIn))
    := by omega
  unfold outDecode
  rw [hv, hd, hm]
  have e : (Rows.ofNet hU).nIn + (1 + (outCol c j t - (Rows.ofNet hU).nIn) - 1) -
      (c.unit.outGroups.getD 0 default).col * WORD = 16 * j + t := by
    unfold outCol at h1 ⊢; omega
  rw [e]
  simp only [Prod.mk.injEq]
  exact ⟨trivial, by omega, by omega⟩

variable {Reg : Type} (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)

open Classical in
/-- **The registration's bit at a unit's computed gate**: bit `t` of the population's output word `lo[j]` of the unit,
where the gate's column in the unit's instance is copy `u'`'s output `j`, bit `t` (`outDecode`), and `lo` the copy's
leaf's outputs. -/
noncomputable def outBitOf (R : Reg) (g : Fin (keyN c hU n)) : Bool :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      ((popOf R).outputs.getD (u.val * c.outWords +
        (c.leavesOut.getD (outDecode c hU (Classical.choose h).val).1 #[])[(outDecode c hU (Classical.choose h).val).2.1]!)
          0).testBit (outDecode c hU (Classical.choose h).val).2.2
    else false
  | none => false

/-- **The committed gates a unit computes are returned outputs' bits** (a condition on the circuit's outputs `outs`):
each column of the unit's instance on one is a returned output's bit (`OutColAt`). -/
def OutsOut : Prop :=
  ∀ u g, g ∈ ((keyProg c hU n).partition outs).io u → (keyProg c hU n).unit g = some u →
    ∀ x : Fin ((keyProg c hU n).rows u).w, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g →
      ∃ u' j t, OutColAt c hU x.val u' j t

/-- **The unit range's count is a power of two** in a circuit `HmRow.parse` accepted whose unit returns the outputs. -/
theorem unitRange_pow2 {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) (ho : c.outNet = c.unitNet)
    (hcount : c.unitRange.count = c.g * c.upv) : 2 ^ c.unitRange.count.log2 = c.unitRange.count := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hp := unit_count_pow2 P.typed hls ho
  rw [C.unit_range.2] at hp
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  rw [hcount]
  exact hp.2

section places

variable {c hU n outs popOf} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

include hparse in
/-- **`hout` at `keyProg`'s places**: in an untyped circuit whose unit returns the outputs, at each drawn unit's site
(`BcSite`, as `keyProg_bcPinned` takes it), a satisfying witness holds at every committed gate the unit computes the
registration's bit `outBitOf`, the population's returned output's, when those gates are returned outputs' bits
(`OutsOut`); the shape region "out" packs is the parse's (`outShape_of_parse`). -/
theorem keyProg_hout (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) :
    ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((keyProg c hU n).partition outs).io u,
      (keyProg c hU n).unit g = some u →
        bit (Binding.wit plan tab S R u M (gateAt (keyPlace hparse outs) S R u g)) = outBitOf c hU n outs popOf R g := by
  intro S R u hu M hM g hg hgu
  have hc' := exists_col_of_io ((keyPlace hparse outs) S R u hu).inst hg
  have hc : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g := hc'
  obtain ⟨u', j, t, hoc⟩ := houts u g hg hgu (Classical.choose hc') (Classical.choose_spec hc')
  obtain ⟨hu', hj, ht, h1, h2, hxv⟩ := hoc
  have hob : outBitOf c hU n outs popOf R g =
      ((popOf R).outputs.getD (u.val * c.outWords + (c.leavesOut.getD u' #[])[j]!) 0).testBit t := by
    unfold outBitOf
    rw [hgu]
    dsimp only
    rw [dif_pos hc]
    have e : outDecode c hU (Classical.choose hc).val = (u', j, t) := outDecode_of ⟨hu', hj, ht, h1, h2, hxv⟩
    rw [e]
  have hcc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hR : unitRows (si.acc S R u hu).st (si.acc S R u hu).layout.unit_built = Rows.ofNet hU :=
    Placed.ofNet_eq hcc _ hU
  have hcol := col_val_comp (si.acc S R u hu) (si.hvu S R u hu)
    (Fin.cast (congrArg FlockSoundness.Rows.w (Sites.rowsOf_keyProg hU hparse S R u hu)) (Classical.choose hc'))
    (u' := u') (c0 := outCol c j t) (by rw [hcc]; exact hu') (by rw [hR]; exact h1) (by rw [hR]; exact h2)
    (by rw [hR, Fin.val_cast, hcc]; exact hxv)
  have hA : (si.acc S R u hu).st.c.unitRange.slot (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') + outCol c j t =
      outAddr (si.acc S R u hu).st.c (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') j t := by
    rw [hcc]
    simp only [outAddr, outCol, Range.slot, Range.start]
    ring
  rw [hA] at hcol
  have hlt : outAddr (si.acc S R u hu).st.c (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') j t <
      2 ^ (si.acc S R u hu).st.c.kLog := by
    rw [← hcol]
    exact lt_of_lt_of_eq (Fin.isLt _) (si.acc S R u hu).kLog_eq.symm
  have hga : gateAt (keyPlace hparse outs) S R u g = siteAddr (si.acc S R u hu) (si.o S R u hu)
      (outAddr (si.acc S R u hu).st.c (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') j t) := by
    unfold gateAt
    rw [dif_pos hu, dif_pos hc']
    unfold siteAddr
    congr 1
    apply Fin.ext
    have hin : (inBlockOr (si.acc S R u hu).st.c.kLog
        (outAddr (si.acc S R u hu).st.c (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') j t)).val =
        outAddr (si.acc S R u hu).st.c (si.vu S R u hu * (si.acc S R u hu).st.c.upv + u') j t := by
      simp only [inBlockOr, hlt, ↓reduceDIte]
    exact hcol.trans hin.symm
  rw [hga, hob]
  have hpar := hparse S R u hu
  rw [← hcc] at hpar
  have hcount := (si.acc S R u hu).layout.unit_count
  have hpow := unitRange_pow2 hpar (by rw [hcc]; exact ho) hcount
  have hk := (si.acc S R u hu).layout.slot_lt (si.hvu S R u hu) (show u' < (si.acc S R u hu).st.c.upv by
    rw [hcc]; exact hu')
  obtain ⟨hw, hsh⟩ := outShape_of_parse (hparse S R u hu) hU ho
  obtain ⟨hbl, hfit⟩ := hsh u' hu'
  have := out_site (si.acc S R u hu) (hsite S R u hu M hM) (by rw [hcc]; exact ho) (parse_facts hpar).typed hM
    (u' := u') (j := j) (t := t) (by rw [hcc]; exact hu') hk (by rw [hpow]; exact hk) (by rw [hcc]; exact hw)
    (by rw [hcc]; exact fun j hj => (hbl j hj).1) (by rw [hcc]; exact fun j hj => (hbl j hj).2)
    (by rw [hcc]; exact hfit) (by rw [hcc]; exact hj) ht
  rw [this, hcc]

variable (rowW : Fin c.ports.size → ℕ) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ)

variable (outs popOf) in
/-- **`rs` at `keyProg`'s places, with the returned outputs**: `keyRows` with the registration's bit `outBitOf`, its
`hout` from `keyProg_hout`. -/
noncomputable def keyRowsOut (enc : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (enc_inj : Function.Injective enc) (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (encO_inj : Function.Injective encO) (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW) (houts : OutsOut c hU n outs)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)}) :
    Binding.HmRows (keyLayout rowW hparse outs saltCol popOf (outBitOf c hU n outs popOf) encO) :=
  keyRows rowW hparse outs saltCol popOf (outBitOf c hU n outs popOf) enc enc_inj encO encO_inj hsite hkeys hc1 hz1
    (keyProg_hout hparse ho hsite houts)

end places

end FlockSoundness.Discharge.Layout
