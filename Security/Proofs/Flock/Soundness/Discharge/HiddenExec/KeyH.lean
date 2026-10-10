import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.Layout.Keyed
import Proofs.Flock.Soundness.Discharge.Layout.PinsX

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in hidden-output tables

`Layout.keyReaders` (`Layout/KeyProg.lean`) at `SitesH.progPlaces_keyProg`, for the hidden-output statement. Its rows
are every port of the circuit: the input rows, as `Layout.keyRowOf` reads them, and the output rows `nIn + jo`, which
hold the returned outputs. So a unit's computed committed gate, which `Layout.keyReaders` reads as a single gate whose
bit region "out" pins (`Layout.outBitOf`), is here a bit of an output row whose `b ‖ c` the public file registers:

* `outBitRow`, `outRowOf`, `rowOfH`: the gate on copy `u'`'s output `j`, bit `t` (`Layout.OutColAt`, read back by
  `Layout.outDecode`) is bit `16 k + t` of output row `nIn + jo`, where the copy's output word `lo[j]` is word `k` of
  output port `jo` (`outPortOf`);
* `keyReadersH`: `Layout.Readers.ofRows` at those rows, their message columns (`keyRowAddrH`), `b ‖ c` (`keyBcAddrH`)
  and salt (`keySaltAddrH`) at each drawn unit's site (`siteAddrH`), the registration's `b ‖ c` the population's public
  file's (`popOf R`), and single gates with `Layout.keyGate (fun _ _ => false)`;
* `keyProgH_copies`: Δ's copies, from `copy_siteH` at an input row and `out_copyH` at an output row;
* `keyProgH_bcPinned`: each row's `b ‖ c` at a drawn unit's site is the population's (`bc_site_ofH`, at a `BcSiteH`);
* `keyProgH_valPinned`: the single gates are the constant and the zero (`Layout.single_gate`, with every computed
  committed gate an output row's bit, `OutsOut`), so they hold 1 and 0 (`constCols_keyProgH`, `zeroCols_keyProgH`);
* `encPortH`: the bytes a row port's hash reads for a value, `Layout.encSeg` at a `sha512/row-seg/v1` port and
  `rowEncP` under the port's own prefix (`Hidden.rowPre`) at any other, with its extractor (`extPortH_spec`);
* `keyLayoutH`, `keyRowsH` (`encPortH` at each row, `bitCode` elsewhere, `Layout.Readers.rowsX`) and
  `keyRowsH_computes`, `HmRowComputes` at them (`bc_site_commitH`, `keyRowH_enc`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-! ## Output rows' bits -/

variable (c) in
/-- The output word of copy `d.1`'s output `d.2.1`. -/
def outWordOf (d : ℕ × ℕ × ℕ) : ℕ := (c.leavesOut.getD d.1 #[])[d.2.1]!

variable (c) in
/-- **An output row's bit**: copy `d.1`'s output `d.2.1`, bit `d.2.2`, of unit `u` is bit `16 k + d.2.2` of `u`'s output
row `nIn + jo`, where the copy's output word is word `k` of output port `jo` (`outPortOf`); none past the row. -/
def outBitRow (u : Fin n) (d : ℕ × ℕ × ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  if hq : HmOut.nIn c + outPortOf c (outWordOf c d) < c.ports.size then
    if hk : 16 * (outWordOf c d - Hidden.outAt c (outPortOf c (outWordOf c d))) + d.2.2 < portRowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

theorem outBitRow_spec {u u' : Fin n} {d : ℕ × ℕ × ℕ} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outBitRow c u d = some (u', ⟨q, k⟩)) :
    u' = u ∧ q.val = HmOut.nIn c + outPortOf c (outWordOf c d) ∧
      k.val = 16 * (outWordOf c d - Hidden.outAt c (outPortOf c (outWordOf c d))) + d.2.2 := by
  unfold outBitRow at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨hu, hs⟩ := h
      subst hu
      cases hs
      exact ⟨rfl, rfl, rfl⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- **An output word's bit is an output row's**: at an output word of the circuit (`outWords`), in a circuit whose
output rows hold their ports' words (`HmOut.checkRows`), each of its 16 bits is a bit of an output row. -/
theorem outBitRow_ne_none (hcr : HmOut.checkRows c = .ok ()) (hrf : Hidden.Any.RowsFitS c) (u : Fin n)
    {d : ℕ × ℕ × ℕ} (hw : outWordOf c d < c.outWords) (ht : d.2.2 < 16) : outBitRow c u d ≠ none := by
  obtain ⟨hjo, -, hk⟩ := outPortOf_spec c hw
  have hle := checkRows_words hcr hrf.1.1 hrf.bits16 hjo
  have hq := (hrf.1.2 _ hjo).1
  have hpw : portRowW c ⟨_, hq⟩ = 16 * (c.ports.getD (HmOut.nIn c + outPortOf c (outWordOf c d)) default).words := by
    unfold portRowW
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    rfl
  unfold outBitRow
  rw [dite_eq_left hq, dite_eq_left (by rw [hpw]; omega)]
  exact Option.some_ne_none _

/-- **A unit's own column on a gate it computes** is a computed column, of that unit. -/
theorem unit_wire (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    (h : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u') :
    u' = u ∧ ((keyProg c hU n).rows u).nIn ≤ x.val := by
  by_cases hcomp : ∃ i, x = ((keyProg c hU n).rows u).comp i
  · obtain ⟨i, rfl⟩ := hcomp
    have hm : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire
        (((keyProg c hU n).rows u).comp i)) = some u := (Const.Prog.instC (keyProg c hU n) outs u).mem i
    rw [hm] at h
    refine ⟨(Option.some.inj h).symm, ?_⟩
    simp only [FlockSoundness.Rows.comp]
    omega
  · have hn : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none :=
      Layout.keyProg_reads c hU n outs u x fun i e => hcomp ⟨i, e⟩
    rw [hn] at h
    cases h

variable (c hU n) in
open Classical in
/-- **The output-row bit of a unit's computed gate**: the bit (`outBitRow`) of the returned output whose column of the
gate's unit's instance holds the gate (`Layout.outDecode`). -/
noncomputable def outRowOf (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      outBitRow c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

theorem outRowOf_wire (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    (hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u)
    (hx : ((keyProg c hU n).rows u).nIn ≤ x.val) :
    outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
      outBitRow c u (outDecode c hU x.val) := by
  have hc : ∃ x', (Const.Prog.instC (keyProg c hU n) outs u).wire x' =
      (Const.Prog.instC (keyProg c hU n) outs u).wire x := ⟨x, rfl⟩
  have hch : Classical.choose hc = x := by
    rcases (Const.Prog.instC (keyProg c hU n) outs u).shared _ x (Classical.choose_spec hc) with he | ⟨-, h2⟩
    · exact he
    · omega
  unfold outRowOf
  rw [hgu]
  dsimp only
  rw [dite_eq_left hc, hch]

theorem outRowOf_wire_some (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u ∧
      ((keyProg c hU n).rows u).nIn ≤ x.val ∧ outBitRow c u (outDecode c hU x.val) = some (u', ⟨q, k⟩) := by
  rcases hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | u''
  · unfold outRowOf at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hx⟩ := unit_wire outs x hgu
    exact ⟨rfl, hx, (outRowOf_wire outs x hgu hx).symm.trans h⟩

variable (c hU n) in
/-- **The row bits of `keyProg` at a hidden-output statement**: an input row's bit (`Layout.keyRowOf`), else an output
row's (`outRowOf`). -/
noncomputable def rowOfH (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match keyRowOf c hU n (portRowW c) g with
  | some r => some r
  | none => outRowOf c hU n outs g

theorem rowOfH_of_key (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    {r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)} (h : keyRowOf c hU n (portRowW c) g = some r) :
    rowOfH c hU n outs g = some r := by
  unfold rowOfH
  rw [h]

theorem rowOfH_of_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : keyRowOf c hU n (portRowW c) g = none) : rowOfH c hU n outs g = outRowOf c hU n outs g := by
  unfold rowOfH
  rw [h]

theorem rowOfH_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)} (h : rowOfH c hU n outs g = none) :
    keyRowOf c hU n (portRowW c) g = none ∧ outRowOf c hU n outs g = none := by
  rcases hk : keyRowOf c hU n (portRowW c) g with _ | r
  · exact ⟨rfl, (rowOfH_of_none outs hk).symm.trans h⟩
  · rw [rowOfH_of_key outs hk] at h
    cases h

/-- **A row bit of a unit's own instance is that unit's row**. -/
theorem rowOfH_wire_unit (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : rowOfH c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) : u' = u := by
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · rw [rowOfH_of_none outs hk] at h
    exact (outBitRow_spec (outRowOf_wire_some outs x h).2.2).1
  · rw [rowOfH_of_key outs hk] at h
    rw [h] at hk
    exact (Layout.keyRowOf_wire outs x hk).1

/-! ## Output-row copies at an accepted statement -/

/-- **A returned output's column holds its output row's bit**: in a satisfying witness of an accepted statement on `c`
whose unit returns the outputs and whose rows are covered (`Hidden.Any.RowsFitS`), at block `o` and template instance `g`, the
template instance's column on copy `u'`'s output `j`, bit `t` (`Layout.OutColAt`) holds the witness's bit at the message
column of its output row's bit (`outBitRow`, `out_copyH`). -/
theorem out_copy_key {T : Model.Statement} (a : AcceptedH T) (hce : a.st.c = c) (ho : c.outNet = c.unitNet)
    (hrf : Hidden.Any.RowsFitS c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (hb : outBitRow c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRow_spec hb
  obtain ⟨hu', hj, ht, h1, h2, hx⟩ := hoc
  have P := parse_facts a.parse
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : a.st.c.leavesOut.getD u' #[] = a.st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf a.st.c (u', j, t) = (a.st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf a.st.c (u', j, t) < a.st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hjo, hle, hkk⟩ := outPortOf_spec a.st.c hwlt
  have hw : Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t))) +
      (outWordOf a.st.c (u', j, t) - Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t)))) =
      (a.st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyH a ho hrf hz o hg x hu' hj ht h1 h2 hx hjo hkk hw
  rw [hq, hk]
  exact this

/-! ## The rows' encodings -/

/-- **The bytes row port `p`'s hash reads for a value**: `Layout.encSeg` at its bits for a `sha512/row-seg/v1` port,
`rowEncP` under its own first block (`Hidden.rowPre` at its `v2`) for any other. -/
def encPortH {W : ℕ} (p : Port) : Layout.BitVal W → List UInt8 :=
  if p.segs.isSome then Layout.encSeg p.bits else Layout.rowEncP (Hidden.rowPre p.v2)

/-- `encPortH`'s extractor: `Layout.extSeg` at a `sha512/row-seg/v1` port, where `encPortH` is not injective. -/
def extPortH {W : ℕ} (p : Port) (v v' : Layout.BitVal W) : List UInt8 × List UInt8 :=
  if p.segs.isSome then Layout.extSeg p.bits v v' else (encPortH p v, encPortH p v')

theorem extPortH_spec {W : ℕ} (p : Port) (v v' : Layout.BitVal W) (hne : v ≠ v') (h : encPortH p v = encPortH p v') :
    Collides (extPortH p v v') := by
  unfold extPortH
  unfold encPortH at h
  split_ifs at h ⊢ with hs
  · exact Layout.extSeg_spec _ v v' hne h
  · exact absurd (Layout.rowEncP_inj (Hidden.size_rowPre _) h) hne

variable (c) in
/-- **`keyProg`'s commit strings' encodings**: `encPortH` at row `q` of every unit, `bitCode` at a single gate. -/
def encKeyH {W N : ℕ} : (Fin n × Fin c.ports.size) ⊕ Fin N → Layout.BitVal W → List UInt8
  | .inl (_, q) => encPortH c.ports[q.val]!
  | .inr _ => Layout.bitCode

variable (c) in
def extKeyH {W N : ℕ} : (Fin n × Fin c.ports.size) ⊕ Fin N → Layout.BitVal W → Layout.BitVal W → List UInt8 × List UInt8
  | .inl (_, q) => extPortH c.ports[q.val]!
  | .inr _ => fun v v' => (Layout.bitCode v, Layout.bitCode v')

variable (c) in
theorem extKeyH_spec {W N : ℕ} (q : (Fin n × Fin c.ports.size) ⊕ Fin N) (v v' : Layout.BitVal W) (hne : v ≠ v')
    (h : encKeyH c q v = encKeyH c q v') : Collides (extKeyH c q v v') := by
  rcases q with ⟨_, q⟩ | _
  · exact extPortH_spec _ v v' hne h
  · change Layout.bitCode v = Layout.bitCode v' at h
    exact absurd (Layout.bitCode_inj h) hne

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesH plan tab)

/-- **Bit `k` of row `q` at a drawn unit's site**: message bit `k % 8` of byte `k / 8` (`Typed.msgCol`) at the unit's
template instance, in its block. -/
noncomputable def keyRowAddrH (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (k : ℕ) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrH (si.acc S R u h) (si.o S R u h)
      (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val (k / 8) (k % 8))
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s `b ‖ c` at a drawn unit's site**: its `hm96` slot's output (`Layout.bcAddr`) at the unit's template
instance. -/
noncomputable def keyBcAddrH (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (j : Fin 128)
    (t : Fin 8) : Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrH (si.acc S R u h) (si.o S R u h) (Layout.bcAddr (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s salt at a drawn unit's site**: its `hm96` slot's salt input (`Hm.saltCol`) at the unit's template
instance. -/
noncomputable def keySaltAddrH (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      siteAddrH (si.acc S R u h) (si.o S R u h) (Hm.saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n)))

/-- `SitesH.progPlaces_keyProg`'s places. -/
noncomputable abbrev keyPlaceH := (SitesH.progPlaces_keyProg hU hparse outs).derived.place

variable (popOf : Reg → Public)

/-- **The readers at `keyProg`'s sites in hidden-output tables**: every row `(u, q)` of `16 · words` bits at its message
columns (`keyRowAddrH`), its `b ‖ c` at the `hm96` output (`keyBcAddrH`) and salt at `Hm.saltCol` (`keySaltAddrH`), the
registration's `b ‖ c` the population's public file's (`popOf R`), and single gates with `Layout.keyGate` at no output
bit. -/
noncomputable def keyReadersH : Layout.Readers ((keyProg c hU n).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceH hparse outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfH c hU n outs)
    (keyRowAddrH si) (keySaltAddrH si (hU := hU)) (keyBcAddrH si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ _ => false)

variable (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitS c)
  (houts : OutsOut c hU n outs)

include hparse ho hrf houts in
/-- **Δ's copies at `keyProg`'s sites in hidden-output tables**: a column on an input row's bit holds the witness's bit
at its key's message column (`copy_siteH`); a column on an output row's bit, a returned output's (`OutsOut`), holds the
bit at the output row's message column (`out_copy_key`). -/
theorem keyProgH_copies : (keyReadersH hparse outs popOf).Copies (keyPlaceH hparse outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesH.rowsOf_keyProg hU hparse S R u hu
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · have ho' : outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (rowOfH_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOf_wire_some outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copy_key (si.acc S R u hu) hc ho hrf hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfH_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := Layout.keyRowOf_wire outs x hr'
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_siteH (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hk' hin'
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this

include hparse in
/-- **The rows' `b ‖ c` at `keyProg`'s sites in hidden-output tables is the registration's**: at each row a drawn unit
reads, input or output, the bytes at its `hm96` output are the population's `b ‖ c` of the unit (`bc_site_ofH`), at
every site whose rank is the unit (`BcSiteH`, at the sites whose table some witness satisfies). -/
theorem keyProgH_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersH hparse outs popOf).BcPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  rw [rowOfH_wire_unit outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofH (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrH
  simp only [hu, ↓reduceDIte]
  exact this

include hparse ho hrf houts in
/-- **The single gates at `keyProg`'s sites in hidden-output tables are the constant and the zero**: when every input a
row reads has the zero or a row bit as its key (`KeysRow`), a committed gate that is no row's bit is the zero, the
constant, or a computed gate (`Layout.single_gate`); a computed one is a returned output's (`OutsOut`), so an output
row's bit (`outBitRow_ne_none`). So the gates hold 0 and 1 (`zeroCols_keyProgH`, `constCols_keyProgH`). -/
theorem keyProgH_valPinned (hkeys : KeysRow c hU (portRowW c)) :
    (keyReadersH hparse outs popOf).ValPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (keyPlaceH hparse outs) (constCols_keyProgH hU hparse outs)
    (zeroCols_keyProgH hU hparse outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  change bit _ = Layout.keyGate (c := c) (hU := hU) (n := n) (fun _ _ => false) R g
  obtain ⟨hk, hno⟩ := rowOfH_none outs hr
  rcases Layout.single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, ↓reduceIte]
  · exfalso
    subst hx
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
    have hno' : outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none := hno
    rw [outRowOf_wire outs x hm (unit_wire outs x hm).2, Layout.outDecode_of hoc] at hno'
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hcr : HmOut.checkRows c = .ok () := hc ▸ (si.acc S R u hu).checkRows
    have P := parse_facts (hparse S R u hu)
    obtain ⟨ls, hls⟩ := P.check
    obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
    obtain ⟨hu', hj, ht, -⟩ := hoc
    have hget : c.leavesOut.getD u'' #[] = c.leavesOut[u'']! := by
      rw [Hidden.getElem!_eq_getD']; rfl
    have hwlt : outWordOf c (u'', j, t) < c.outWords := by
      rw [show outWordOf c (u'', j, t) = (c.leavesOut.getD u'' #[])[j] from getElem!_pos _ j hj]
      refine (hlo u'' hu').2 _ ?_
      rw [← hget]
      exact Array.getElem_mem_toList _
    exact outBitRow_ne_none hcr hrf u hwlt ht hno'

/-- **`lay` at `keyProg`'s places in hidden-output tables**: the readers' layout (`Layout.Readers.layout`), with
`bitCode` for the single gates and Δ's copies from `keyProgH_copies`. -/
noncomputable def keyLayoutH :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlaceH hparse outs)) :=
  (keyReadersH hparse outs popOf).layout (keyPlaceH hparse outs) Layout.bitCode
    (keyProgH_copies hparse outs popOf ho hrf houts)

/-- **`rs` at `keyProg`'s places in hidden-output tables**: the readers' rows each under its own encoding
(`Layout.Readers.rowsX`), `encPortH` at each row (the bytes `sha512/row/v1`, `/row/v2` or `/row-seg/v1` hashes), from
`keyProgH_copies`, `keyProgH_bcPinned` and `keyProgH_valPinned`. -/
noncomputable def keyRowsH
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    Binding.HmRowsX (keyLayoutH hparse outs popOf ho hrf houts) :=
  (keyReadersH hparse outs popOf).rowsX (keyPlaceH hparse outs) (encKeyH c) (extKeyH c) (extKeyH_spec c)
    Layout.bitCode Layout.bitCode_inj (keyProgH_copies hparse outs popOf ho hrf houts)
    (keyProgH_bcPinned hparse outs popOf hsite) (keyProgH_valPinned hparse outs popOf ho hrf houts hkeys)

/-- **The bytes a row's hash reads at `keyProg`'s places in hidden-output tables** (`Layout.keyRow_enc`): at a drawn
unit `u`'s site, the value a message holds at row `q` of unit `u'`, `16 · words` bits, has `encPortH` the bytes the row
hashes (`Hidden.Any.rowMsg`) for the `2 · words` bytes the witness holds at the unit's template instance's message
columns, in its block. -/
theorem keyRowH_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) (hrw : portRowW c q = 16 * c.ports[q.val]!.words)
    (hb : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words) :
    encPortH c.ports[q.val]! ((keyLayoutH hparse outs popOf ho hrf houts).row S R u M (.inl (u', q))) =
      Hidden.Any.rowMsg c.ports[q.val]!
        (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddrH (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val)))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hbytes : (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t =>
      (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) => keyRowAddrH si S R u q k.val)
        ⟨8 * j.val + t.val, by omega⟩) =
      List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
        (siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
    refine List.ext_getElem (by simp) fun j h1 h2 => ?_
    simp only [List.getElem_ofFn]
    congr 1
    funext t
    have hj : (8 * j + t.val) / 8 = j := by omega
    have ht : (8 * j + t.val) % 8 = t.val := by omega
    show bit (Binding.wit plan tab S R u M (keyRowAddrH si S R u q (8 * j + t.val))) = _
    simp only [keyRowAddrH, hu, ↓reduceDIte, hj, ht]
  show encPortH (W := Finset.univ.sup (portRowW c) + 1) c.ports[q.val]!
    (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
      Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
        keyRowAddrH si S R u q k.val)) = _
  unfold encPortH Hidden.Any.rowMsg
  split_ifs with hs
  · rw [hb, ← hrw, Layout.encSeg_row hrw hle, hbytes]
  · rw [Layout.rowEncP_row _ hrw hle, hbytes]

/-- **`HmRowComputes` at `keyProg`'s rows in hidden-output tables** (`hHm` for `keyRowsH`): at every drawn unit's
committed wire held by a row, input or output, in a satisfying witness of the unit's table, the `b ‖ c` at the row's
`hm96` output is the commit string of the row's bytes (`rowEncP`) and the salt the witness holds at `Hm.saltCol`
(`bc_site_commitH`, `keyRowH_enc`). -/
theorem keyRowsH_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsH hparse outs popOf ho hrf houts hsite hkeys).Computes := by
  unfold Binding.HmRowsX.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutH hparse outs popOf ho hrf houts).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * c.ports[q.val]!.words := by
      rw [getElem!_pos c.ports q.val q.isLt]
      rfl
    have hbq : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words :=
      hrf.bits16 _ (by rw [getElem!_pos c.ports q.val q.isLt]; exact Array.getElem_mem_toList q.isLt)
    have hb := bc_site_commitH (si.acc S R u hu) (by rw [hce]; exact ho) (by rw [hce]; exact hrf) hM
      (si.o S R u hu) (si.hvu S R u hu) hq
    have hpq : (si.acc S R u hu).st.c.ports[q.val]! = c.ports[q.val]! := by rw [hce]
    rw [hpq] at hb
    show Binding.bytesAt _ (keyBcAddrH si S R u q) = Binding.commitString
      (encPortH c.ports[q.val]! ((keyLayoutH hparse outs popOf ho hrf houts).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddrH si S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddrH si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddrH (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddrH, hu, dite_true]
    have e2 : keySaltAddrH si S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddrH, hu, dite_true]
    rw [keyRowH_enc hparse outs popOf ho hrf houts hu M u' q hwq hbq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.HiddenExec
