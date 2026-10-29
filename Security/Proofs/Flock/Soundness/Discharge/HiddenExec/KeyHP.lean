import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteHP
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.Layout.OutsP

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProgP`'s places in hidden-output tables at public inputs

`KeyH.lean` at `SitesHP.progPlaces_keyProgP`, whose units read the gates of their keys with public inputs
(`Placed.keyProgP`). Its rows are the input rows (`Layout.keyRowOfP`) and the output rows (`outRowOfP`), as `KeyH`'s;
its single gates are the constant, the zero, and each unit's public gates (`Layout.keyPubOf`), whose bit is the unit's
public input word's in the population's public file (`pubBitOf`):

* `pub_siteHP`: at a drawn unit's site (`BcSiteHP`), a stacked unit input whose key is bit `t` of public word `q` holds
  bit `t` of word `q` of the proved unit's public inputs (`pub_block`, `Hidden.inVal_drawnP`);
* `rowOfHP`, `keyReadersHP`: `Layout.Readers.ofRows` at those rows, at each drawn unit's site (`siteAddrHP`), and single
  gates with `Layout.keyGateP (pubBitOf pi popOf) (fun _ _ => false)`; a public gate is no row's bit (`rowOfHP_pub`);
* `keyProgHP_copies`: Δ's copies, from `copy_siteHP` at an input row (its key a message bit, so off a public leaf) and
  `out_copyHP` at an output row (`out_copy_keyP`);
* `keyProgHP_bcPinned`: each row's `b ‖ c` at a drawn unit's site is the population's (`bc_site_ofHP`);
* `keyProgHP_gate`, `keyProgHP_valPinned`: the single gates are the constant, the zero and the public gates
  (`Layout.single_gateP`, with every computed committed gate an output row's bit, `OutsOutP`), holding 1, 0
  (`constCols_keyProgHP`, `zeroCols_keyProgHP`) and the public input words' bits (`pub_siteHP`);
* `keyLayoutHP`, `keyRowsHP` and `keyRowsHP_computes`, `HmRowComputes` at them (`bc_site_commitHP`, `keyRowHP_enc`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN portRowW OutColAt outDecode OutsOutP KeysRowP keyRowOfP keyPubOf keyGateP)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {pi : HmIn.Ports}

/-! ## Public gates at a site -/

/-- **A drawn unit's input at a public leaf holds its public word's bit**: in a satisfying witness of an accepted
statement at public inputs, at a site (`BcSiteHP`: block `o`, template instance `g`, proved unit `u`), a stacked unit
input whose key is bit `t` of public word `q` holds bit `t` of word `q` of `u`'s public inputs in the population's
public file (`pub_block`, `Hidden.inVal_drawnP`). -/
theorem pub_siteHP {T : Model.Statement} {a : AcceptedHP T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ}
    (hs : BcSiteHP a pop o g u) {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) (x : Fin a.rows.w)
    (hx : x.val < a.rows.nIn) {q t : ℕ} (hk : Aliased.srcKeyP a.st.c a.pi x.val = .pub q t) :
    bit (T.block z o (a.col hg x)) = (pop.outputs.getD (u * a.pi.words + q) 0).testBit t := by
  obtain ⟨d, D, hd, hD, hlt, hpop, -, hu⟩ := hs
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, hi, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  dsimp only at hd hpop hu hg x hx hk ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  have hpub : 0 ≤ Aliased.pubLeaf st.c pi x.val := by
    by_contra hn
    rw [Aliased.srcKeyP_of_neg (c := st.c) (pi := pi) (x := x.val) (by omega)] at hk
    exact Aliased.KeyP.noConfusion hk
  obtain ⟨hpa, -⟩ := Hidden.parse_untyped ht (s.parseF hi)
  have P := parse_facts hpa
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  obtain ⟨ro, ri, -, hRi, hst⟩ := s.regions
  have hri : ∀ r ∈ ri.toList, r ∈ st.regions.toList := fun r hr' => by
    rw [hst, Array.toList_append]; exact List.mem_append_right _ hr'
  obtain ⟨p, t', hkey, hq, hb⟩ := pub_block L regs hregs hRi hri (ports_of_pub s.ports hpub) C hg x hx hpub hz o
  rw [hk] at hkey
  injection hkey with hqq htt
  subst hqq htt
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← L.unitRows_nIn]; exact hx
  have hW : 0 < st.c.unit.inWords * WORD :=
    Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx'; omega
  have hdx : x.val / (st.c.unit.inWords * WORD) < st.c.upv := (Nat.div_lt_iff_lt_mul hW).2 hx'
  obtain ⟨hdv, hmd⟩ := slot_div_mod (g := g) hdx
  have hkc : g * st.c.upv + x.val / (st.c.unit.inWords * WORD) < st.c.unitRange.count := by
    rw [L.unit_count]
    nlinarith [Nat.mul_le_mul_right st.c.upv (Nat.succ_le_of_lt hg)]
  have hq' : (pi.leaves.getD ((g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) % st.c.upv) #[]).getD p (-1) =
      Aliased.pubLeaf st.c pi x.val := by
    rw [hmd]; exact hq
  have hiv := Hidden.inVal_drawnP s hpop hD hlt (b := o.val) hkc (by rw [hdv]) hr hq' hpub
  simp only [AcceptedHP.col, Fin.cast_eq_self]
  rw [hb, hiv]

/-- **The registration's bit at a public gate**: bit `t` of word `q` of unit `u`'s public inputs in the population's
public file. -/
def pubBitOf {Reg : Type} (pi : HmIn.Ports) (popOf : Reg → Public) (R : Reg) (u : Fin n) (q t : ℕ) : Bool :=
  ((popOf R).outputs.getD (u.val * pi.words + q) 0).testBit t

/-- `keyGateP` at a public gate is `pubBit`'s. -/
theorem keyGateP_pub {Reg : Type} (pubBit : Reg → Fin n → ℕ → ℕ → Bool) (outBit : Reg → Fin (keyN c hU n) → Bool)
    (R : Reg) {g : Fin (keyN c hU n)} {u : Fin n} {q t : ℕ} (h : keyPubOf c hU n pi g = some (u, q, t)) :
    keyGateP c hU n pi pubBit outBit R g = pubBit R u q t := by
  obtain ⟨h2, -⟩ := Layout.keyPubOf_spec h
  have h1 : g ≠ Layout.oneG := fun e => by
    rw [e] at h2
    simp [Layout.oneG, Placed.oneGate] at h2 <;> omega
  have h0 : g ≠ Layout.zeroG := fun e => by
    rw [e] at h2
    simp [Layout.zeroG, Placed.zeroGate] at h2 <;> omega
  simp only [keyGateP, h1, h0, h, ↓reduceIte]

/-! ## Output rows' bits -/

/-- **A unit's own column on a gate it computes** is a computed column, of that unit (`unit_wire` at `keyProgP`). -/
theorem unit_wireP (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProgP c pi hU n).rows u).w)
    (h : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u') :
    u' = u ∧ ((keyProgP c pi hU n).rows u).nIn ≤ x.val := by
  by_cases hcomp : ∃ i, x = ((keyProgP c pi hU n).rows u).comp i
  · obtain ⟨i, rfl⟩ := hcomp
    have hm : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire
        (((keyProgP c pi hU n).rows u).comp i)) = some u := (Const.Prog.instC (keyProgP c pi hU n) outs u).mem i
    rw [hm] at h
    refine ⟨(Option.some.inj h).symm, ?_⟩
    simp only [FlockSoundness.Rows.comp]
    omega
  · have hn : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = none :=
      Layout.keyProgP_reads c hU n pi outs u x fun i e => hcomp ⟨i, e⟩
    rw [hn] at h
    cases h

variable (c hU n pi) in
open Classical in
/-- **The output-row bit of a unit's computed gate** of `keyProgP` (`outRowOf` at `keyProgP`). -/
noncomputable def outRowOfP (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (keyProgP c pi hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x = g then
      outBitRow c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

theorem outRowOfP_wire (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProgP c pi hU n).rows u).w)
    (hgu : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u)
    (hx : ((keyProgP c pi hU n).rows u).nIn ≤ x.val) :
    outRowOfP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) =
      outBitRow c u (outDecode c hU x.val) := by
  have hc : ∃ x', (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x' =
      (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x := ⟨x, rfl⟩
  have hch : Classical.choose hc = x := by
    rcases (Const.Prog.instC (keyProgP c pi hU n) outs u).shared _ x (Classical.choose_spec hc) with he | ⟨-, h2⟩
    · exact he
    · omega
  unfold outRowOfP
  rw [hgu]
  dsimp only
  rw [dite_eq_left hc, hch]

theorem outRowOfP_wire_some (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n}
    (x : Fin ((keyProgP c pi hU n).rows u).w) {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outRowOfP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u ∧
      ((keyProgP c pi hU n).rows u).nIn ≤ x.val ∧ outBitRow c u (outDecode c hU x.val) = some (u', ⟨q, k⟩) := by
  rcases hgu : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) with _ | u''
  · unfold outRowOfP at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hx⟩ := unit_wireP outs x hgu
    exact ⟨rfl, hx, (outRowOfP_wire outs x hgu hx).symm.trans h⟩

variable (c hU n pi) in
/-- **The row bits of `keyProgP` at a hidden-output statement**: an input row's bit (`Layout.keyRowOfP`), else an
output row's (`outRowOfP`). -/
noncomputable def rowOfHP (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match keyRowOfP c hU n pi (portRowW c) g with
  | some r => some r
  | none => outRowOfP c hU n pi outs g

theorem rowOfHP_of_key (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    {r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)} (h : keyRowOfP c hU n pi (portRowW c) g = some r) :
    rowOfHP c hU n pi outs g = some r := by
  unfold rowOfHP
  rw [h]

theorem rowOfHP_of_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : keyRowOfP c hU n pi (portRowW c) g = none) : rowOfHP c hU n pi outs g = outRowOfP c hU n pi outs g := by
  unfold rowOfHP
  rw [h]

theorem rowOfHP_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : rowOfHP c hU n pi outs g = none) :
    keyRowOfP c hU n pi (portRowW c) g = none ∧ outRowOfP c hU n pi outs g = none := by
  rcases hk : keyRowOfP c hU n pi (portRowW c) g with _ | r
  · exact ⟨rfl, (rowOfHP_of_none outs hk).symm.trans h⟩
  · rw [rowOfHP_of_key outs hk] at h
    cases h

/-- **A public gate is no row's bit**: its key is a public word's bit, so it is no input row's (`keyRowOfP`), and it is
an input gate of `keyProgP` (`Layout.unit_uniform_input`), so no unit computes it (`outRowOfP`). -/
theorem rowOfHP_pub (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)} {u : Fin n} {q t : ℕ}
    (h : keyPubOf c hU n pi g = some (u, q, t)) : rowOfHP c hU n pi outs g = none := by
  obtain ⟨h2, hlt, -, hk⟩ := Layout.keyPubOf_spec h
  have hkr : keyRowOfP c hU n pi (portRowW c) g = none := by
    unfold keyRowOfP
    simp only [h2, hlt, and_self, ↓reduceDIte, hk]
  rw [rowOfHP_of_none outs hkr]
  have hun : (keyProgP c pi hU n).unit g = none :=
    Layout.unit_uniform_input (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdxP c pi (Layout.keyB c hU)))
      (one := Placed.oneGate n (Layout.keyB c hU))
      (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (Aliased.keyIdxP c pi (Layout.keyB c hU))) n g hlt
  unfold outRowOfP
  rw [hun]

/-- **A row bit of a unit's own instance is that unit's row**. -/
theorem rowOfHP_wire_unit (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n}
    (x : Fin ((keyProgP c pi hU n).rows u).w) {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : rowOfHP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u := by
  rcases hk : keyRowOfP c hU n pi (portRowW c) ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) with _ | r
  · rw [rowOfHP_of_none outs hk] at h
    exact (outBitRow_spec (outRowOfP_wire_some outs x h).2.2).1
  · rw [rowOfHP_of_key outs hk] at h
    rw [h] at hk
    exact (Layout.keyRowOfP_wire outs x hk).1

/-! ## Output-row copies at an accepted statement -/

/-- **A returned output's column holds its output row's bit** at public inputs (`out_copy_key`, through
`out_copyHP`). -/
theorem out_copy_keyP {T : Model.Statement} (a : AcceptedHP T) (hce : a.st.c = c) (ho : c.outNet = c.unitNet)
    (hrf : Hidden.Any.RowsFitS c)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (hb : outBitRow c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrHP a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
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
  have er : unitRows a.st a.L.unit_built = Rows.ofNet hU := Placed.ofNet_eq (lunit_flat P.tmpl) _ hU
  have := out_copyHP a ho hrf hz o hg x hu' hj ht (by rw [er]; exact h1) (by rw [er]; exact h2) (by rw [er]; exact hx)
    hjo hkk hw
  rw [hq, hk]
  exact this

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesHP plan tab)

/-- **Bit `k` of row `q` at a drawn unit's site** at public inputs (`keyRowAddrH`). -/
noncomputable def keyRowAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (k : ℕ) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h)
      (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val (k / 8) (k % 8))
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s `b ‖ c` at a drawn unit's site** at public inputs (`keyBcAddrH`). -/
noncomputable def keyBcAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (j : Fin 128)
    (t : Fin 8) : Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h) (Layout.bcAddr (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s salt at a drawn unit's site** at public inputs (`keySaltAddrH`). -/
noncomputable def keySaltAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      siteAddrHP (si.acc S R u h) (si.o S R u h) (Hm.saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (hpi : HmIn.parse c = .ok pi)
  (outs : Finset (Fin (keyN c hU n)))

/-- `SitesHP.progPlaces_keyProgP`'s places. -/
noncomputable abbrev keyPlaceHP := (SitesHP.progPlaces_keyProgP hU hparse hpi outs).derived.place

variable (popOf : Reg → Public)

/-- **The readers at `keyProgP`'s sites in hidden-output tables**: `keyReadersH`'s rows (`rowOfHP`) at public inputs,
and single gates with `Layout.keyGateP`, a public gate's bit its public input word's (`pubBitOf`). -/
noncomputable def keyReadersHP : Layout.Readers ((keyProgP c pi hU n).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceHP hparse hpi outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _)
    (rowOfHP c hU n pi outs) (keyRowAddrHP si) (keySaltAddrHP si (hU := hU)) (keyBcAddrHP si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (keyGateP c hU n pi (pubBitOf pi popOf) fun _ _ => false)

variable (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitS c)
  (houts : OutsOutP c hU n pi outs)

include hparse hpi ho hrf houts in
/-- **Δ's copies at `keyProgP`'s sites in hidden-output tables**: a column on an input row's bit, its key a message bit
and so off a public leaf, holds the witness's bit at its key's message column (`copy_siteHP`); a column on an output
row's bit holds the bit at the output row's message column (`out_copy_keyP`). -/
theorem keyProgHP_copies : (keyReadersHP hparse hpi outs popOf).Copies (keyPlaceHP hparse hpi outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hq := (si.acc S R u hu).pi_eq (hparse S R u hu) hpi
  have hrw := SitesHP.rowsOf_keyProgP (pi := pi) hU hparse S R u hu
  rcases hk : keyRowOfP c hU n pi (portRowW c) ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) with _ | r
  · have ho' : outRowOfP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := (rowOfHP_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOfP_wire_some outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copy_keyP (si.acc S R u hu) hc ho hrf hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrHP, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOfP c hU n pi (portRowW c) ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfHP_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := Layout.keyRowOfP_wire outs x hr'
    have hneg' : Aliased.pubLeaf c pi x.val < 0 := by
      by_contra hn
      obtain ⟨t, ht⟩ := Aliased.srcKeyP_of_nonneg (c := c) (pi := pi) (x := x.val) (by omega)
      rw [ht] at hkey
      exact Aliased.KeyP.noConfusion hkey
    have hneg : Aliased.pubLeaf (si.acc S R u hu).st.c (si.acc S R u hu).pi x.val < 0 := by
      rw [hc, hq]; exact hneg'
    rw [Aliased.srcKeyP_of_neg hneg'] at hkey
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact Aliased.KeyP.src.inj hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_siteHP (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hneg hk' hin'
    simp only [keyRowAddrHP, hu, ↓reduceDIte]
    exact this

include hparse hpi in
/-- **The rows' `b ‖ c` at `keyProgP`'s sites in hidden-output tables is the registration's**: at each row a drawn
unit reads, the bytes at its `hm96` output are the population's `b ‖ c` of the unit (`bc_site_ofHP`, at a
`BcSiteHP`). -/
theorem keyProgHP_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse hpi outs popOf).BcPinned (P := (keyProgP c pi hU n).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse hpi outs) S R u hu).inst hg
  rw [rowOfHP_wire_unit outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofHP (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrHP
  simp only [hu, ↓reduceDIte]
  exact this

include hparse hpi ho hrf houts in
/-- **A single gate at a drawn unit's site holds the registration's bit**: when every input a row reads has a key with
public inputs (`KeysRowP`), a committed gate that is no row's bit is the zero, the constant, a public gate of the unit,
or a computed gate (`Layout.single_gateP`); a computed one is a returned output's (`OutsOutP`), so an output row's bit.
So the gate holds 0, 1 (`zeroCols_keyProgHP`, `constCols_keyProgHP`) or, at a public gate, the unit's public input
word's bit (`pub_siteHP`, at each drawn unit's site, `BcSiteHP`). -/
theorem keyProgHP_gate (hkeys : KeysRowP c hU pi (portRowW c))
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) {M : Binding.Msg plan tab S R u}
    (hM : Binding.Sat plan tab S R u M) {g : Fin (keyN c hU n)} (hg : g ∈ ((keyProgP c pi hU n).partition outs).io u)
    (hr : rowOfHP c hU n pi outs g = none) :
    bit (Binding.wit plan tab S R u M (Layout.gateAt (keyPlaceHP hparse hpi outs) S R u g)) =
      keyGateP c hU n pi (pubBitOf pi popOf) (fun _ _ => false) R g := by
  have hpub := Layout.gate_pub (keyPlaceHP hparse hpi outs) (constCols_keyProgHP hU hparse hpi outs)
    (zeroCols_keyProgHP hU hparse hpi outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse hpi outs) S R u hu).inst hg
  obtain ⟨hk, hno⟩ := rowOfHP_none outs hr
  rcases Layout.single_gateP outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨q, t, hpq⟩ | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [keyGateP, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [keyGateP, ↓reduceIte]
  · rw [keyGateP_pub _ _ R hpq]
    subst hx
    obtain ⟨-, hxn, hkey⟩ := Layout.keyPubOf_wire outs x hpq
    rw [← Layout.block_gateAt (keyPlaceHP hparse hpi outs) hu hM x]
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq := (si.acc S R u hu).pi_eq (hparse S R u hu) hpi
    have hrw := SitesHP.rowsOf_keyProgP (pi := pi) hU hparse S R u hu
    have hk' : Aliased.srcKeyP (si.acc S R u hu).st.c (si.acc S R u hu).pi x.val = .pub q t := by
      rw [hc, hq]; exact hkey
    have := pub_siteHP (hsite S R u hu M hM) hM (si.hvu S R u hu) (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x)
      (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw)) hk'
    rw [hq] at this
    exact this
  · exfalso
    subst hx
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
    have hno' : outRowOfP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = none := hno
    rw [outRowOfP_wire outs x hm (unit_wireP outs x hm).2, Layout.outDecode_of hoc] at hno'
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

include hparse hpi ho hrf houts in
/-- **The single gates at `keyProgP`'s sites in hidden-output tables hold the registration's bits**
(`keyProgHP_gate`). -/
theorem keyProgHP_valPinned (hkeys : KeysRowP c hU pi (portRowW c))
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse hpi outs popOf).ValPinned (P := (keyProgP c pi hU n).partition outs) :=
  Layout.ofRows_valPinned _ fun _ _ _ hu _ hM _ hg hr =>
    keyProgHP_gate hparse hpi outs popOf ho hrf houts hkeys hsite hu hM hg hr

/-- **`lay` at `keyProgP`'s places in hidden-output tables**: the readers' layout, with `bitCode` for the single gates
and Δ's copies from `keyProgHP_copies`. -/
noncomputable def keyLayoutHP :
    Binding.Layout ((keyProgP c pi hU n).partition outs) plan tab
      (((keyProgP c pi hU n).partition outs).placeDecoder plan tab (keyPlaceHP hparse hpi outs)) :=
  (keyReadersHP hparse hpi outs popOf).layout (keyPlaceHP hparse hpi outs) Layout.bitCode
    (keyProgHP_copies hparse hpi outs popOf ho hrf houts)

/-- **`rs` at `keyProgP`'s places in hidden-output tables**: the readers' rows each under its own encoding, `encPortH`
at each row, from `keyProgHP_copies`, `keyProgHP_bcPinned` and `keyProgHP_valPinned`. -/
noncomputable def keyRowsHP
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRowP c hU pi (portRowW c)) :
    Binding.HmRowsX (keyLayoutHP hparse hpi outs popOf ho hrf houts) :=
  (keyReadersHP hparse hpi outs popOf).rowsX (keyPlaceHP hparse hpi outs) (encKeyH c) (extKeyH c) (extKeyH_spec c)
    Layout.bitCode Layout.bitCode_inj (keyProgHP_copies hparse hpi outs popOf ho hrf houts)
    (keyProgHP_bcPinned hparse hpi outs popOf hsite)
    (keyProgHP_valPinned hparse hpi outs popOf ho hrf houts hkeys hsite)

/-- **The bytes a row's hash reads at `keyProgP`'s places in hidden-output tables** (`keyRowH_enc`). -/
theorem keyRowHP_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) (hrw : portRowW c q = 16 * c.ports[q.val]!.words)
    (hb : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words) :
    encPortH c.ports[q.val]! ((keyLayoutHP hparse hpi outs popOf ho hrf houts).row S R u M (.inl (u', q))) =
      Hidden.Any.rowMsg c.ports[q.val]!
        (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddrHP (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val)))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hbytes : (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t =>
      (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) => keyRowAddrHP si S R u q k.val)
        ⟨8 * j.val + t.val, by omega⟩) =
      List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
        (siteAddrHP (si.acc S R u hu) (si.o S R u hu)
          (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
    refine List.ext_getElem (by simp) fun j h1 h2 => ?_
    simp only [List.getElem_ofFn]
    congr 1
    funext t
    have hj : (8 * j + t.val) / 8 = j := by omega
    have ht : (8 * j + t.val) % 8 = t.val := by omega
    show bit (Binding.wit plan tab S R u M (keyRowAddrHP si S R u q (8 * j + t.val))) = _
    simp only [keyRowAddrHP, hu, ↓reduceDIte, hj, ht]
  show encPortH (W := Finset.univ.sup (portRowW c) + 1) c.ports[q.val]!
    (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
      Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
        keyRowAddrHP si S R u q k.val)) = _
  unfold encPortH Hidden.Any.rowMsg
  split_ifs with hs
  · rw [hb, ← hrw, Layout.encSeg_row hrw hle, hbytes]
  · rw [Layout.rowEncP_row _ hrw hle, hbytes]

/-- **`HmRowComputes` at `keyProgP`'s rows in hidden-output tables** (`keyRowsH_computes` at public inputs, through
`bc_site_commitHP` and `keyRowHP_enc`). -/
theorem keyRowsHP_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRowP c hU pi (portRowW c)) :
    (keyRowsHP hparse hpi outs popOf ho hrf houts hsite hkeys).Computes := by
  unfold Binding.HmRowsX.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutHP hparse hpi outs popOf ho hrf houts).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * c.ports[q.val]!.words := by
      rw [getElem!_pos c.ports q.val q.isLt]
      rfl
    have hbq : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words :=
      hrf.bits16 _ (by rw [getElem!_pos c.ports q.val q.isLt]; exact Array.getElem_mem_toList q.isLt)
    have hupv : 0 < (si.acc S R u hu).st.c.upv := by
      have P := parse_facts (si.acc S R u hu).parse
      obtain ⟨ls, hls⟩ := P.check
      exact Hm.upv_pos (check_facts P.typed hls) (by rw [hce]; exact ho)
    have hb := bc_site_commitHP (si.acc S R u hu) hupv (by rw [hce]; exact hrf) hM
      (si.o S R u hu) (si.hvu S R u hu) hq
    have hpq : (si.acc S R u hu).st.c.ports[q.val]! = c.ports[q.val]! := by rw [hce]
    rw [hpq] at hb
    show Binding.bytesAt _ (keyBcAddrHP si S R u q) = Binding.commitString
      (encPortH c.ports[q.val]! ((keyLayoutHP hparse hpi outs popOf ho hrf houts).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddrHP si S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddrHP si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddrHP (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddrHP, hu, dite_true]
    have e2 : keySaltAddrHP si S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddrHP (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddrHP, hu, dite_true]
    rw [keyRowHP_enc hparse hpi outs popOf ho hrf houts hu M u' q hwq hbq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.HiddenExec
