import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.Inst.Keys

set_option Elab.async false

/-!
# The program of the whole template instance at hidden-output sites

`Inst/Prog.lean` and `Inst/Keys.lean` at `AcceptedH` (hidden-output tables), with `KeyH.lean`'s output rows read at
`instProg`'s composite columns:

* `wiredDeltaH`: Δ at an input bit a wire fills is its flat twin's, `outPairs` writing nothing there: their message
  bits are at the rows' `sha512x3` and `hm96` slots, no other net's (`outPairs_ne_net`). So the hidden statement places
  the template instance's rows (`AcceptedH.placement_inst`, from `placement_instOf`);
* `inst_blockH`: a composite input carries its flat twin's copy position (`copyAt_blockH`), so composite inputs with one
  key carry one value (`AcceptedH.inst_aliased`), a zero-keyed one 0 (`AcceptedH.inst_zero`), and a message-keyed one its
  row's bit (`copy_site_instH`);
* `SitesH.progPlaces_instProg`, `constCols_instProgH`, `zeroCols_instProgH`: `ProgPlaces`, `hConst` and `hZero` from
  the sites and the parse, at output rows that fit;
* `OutColAtI`, `outBitRowO`, `instOutRowOf`, `instRowOfH`: a composite column holding bit `t` of output word `o` at every
  template instance (`HmOut.outCol`, where the unit or a tail computes it) is bit `16 k + t` of output row `nIn + jo`,
  where `o` is word `k` of output port `jo`; `out_copy_instH` is its copy (`Hidden.Any.msg_copyF`);
* `outsOfI`, `outsOutI_outsOfI`: the program's outputs on those columns, which every computed committed gate is on;
* `instReadersH`, `instProgH_copies`, `instProgH_bcPinned`, `instProgH_valPinned` (every composite input a row reads
  has a row key, `InstKeysRow c (portRowW c)`), `instLayoutH`, `instRowsH` and `instRowsH_computes`: `lay`, `rs` and
  `HmRowComputes` at `instProg`'s places.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed Compose FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Layout (portRowW)
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt)

/-! ## Δ at the input bits a wire fills -/

/-- **No output pair of Δ is at another net's slot**, at output rows that fit: every output row's message bit is an
input bit of a `sha512x3` or `hm96` slot (`Hidden.Any.msg_at`). -/
theorem outPairs_ne_net {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
    (hrows : Hidden.Any.OutRowsP st.c) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {n s₀ b : ℕ}
    (hn : n < st.c.nets.size) (hsha : n ≠ st.c.shaNet) (hhm : n ≠ st.c.hmNet) (hs₀ : s₀ < (st.c.rangeOf n).count)
    (hb : b < 2 ^ (st.c.rangeOf n).slotLog) : q.1 ≠ (st.c.rangeOf n).slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  obtain ⟨hp, -, ht8⟩ := hrows.bound hj hk ht
  have R := Hm.RowRanges.ofL L F.parse F.check
  obtain ⟨n', s, off, hn', hat, -⟩ := Hidden.Any.msg_at R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hne : n ≠ n' := by rcases hn' with rfl | rfl <;> assumption
  intro he
  exact Hm.atSlot_ne L F (R.rowNet_lt hn') hn hne hat ⟨hs₀, hb, he⟩

/-- **Δ at the input bits a wire fills** (`WiredDelta`), at a Δ that is `HmRow.delta`'s, then `outPairs`: its flat
twin's (`delta_wired`), at output rows that fit. -/
theorem wiredDelta_out {tags : Tags} {st : Stmt} (Lf : FlockSoundness.Layout (Hidden.flatTwin st))
    (F : Aliased.UnitFacts tags (Hidden.flatTwin st)) (hrows : Hidden.Any.OutRowsP st.c)
    (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray) : WiredDelta st := by
  intro n g i o s hn hsha hhm hg hi hf j
  have P : ParseFacts tags st.c := F.parse
  have C : CheckFacts st.c := F.check
  have hflat := delta_wired Lf F hn hsha hhm hg hi hf j
  obtain ⟨ho, -⟩ := feed_some hf
  obtain ⟨p, tb, hp, htb, hpt⟩ := Aliased.mem_portBits ho
  have hs₀ : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := net_slot_lt Lf C hn hg hi
  have hsl : (st.c.rangeOf n).slotLog = (netOf st.c n).unitLog := (net_range Lf C hn).2.2.2
  have hq : (portCols (netOf st.c n).inGroups)[p]!.1 + (portCols (netOf st.c n).inGroups)[p]!.2 ≤ netIn st.c n :=
    (nets_ok P n).1.2.2 _ (getElem_mem_toList' hp)
  have hfit : netIn st.c n ≤ netW st.c n := (Lf.fit n).2
  have hb : o < 2 ^ (st.c.rangeOf n).slotLog := by
    rw [hsl]
    show o < netW st.c n
    omega
  rw [hda, Array.toList_append, List.count_append, List.toList_toArray,
    List.count_eq_zero.2 fun hm => outPairs_ne_net Lf F hrows hm hn hsha hhm hs₀ hb (pos_eq st.c g n i o),
    Nat.add_zero]
  exact hflat

/-- **Δ at the input bits a wire fills, in an accepted hidden-output statement without its pinned copies**
(`wiredDelta_out` at `Hidden.unpin`, whose Δ is `HmRow.delta`'s, then `outPairs`, by definition). -/
theorem wiredDeltaH {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hrows : Hidden.Any.OutRowsP st.c) : WiredDelta (Hidden.unpin st) :=
  have FL := flat_layout ht h
  have FU := flat_unitFacts ht h
  wiredDelta_out (st := Hidden.unpin st) FL FU hrows rfl

/-! ## A composite input's value -/

section block

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (Lf : FlockSoundness.Layout (Hidden.flatTwin st))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  (hR : UnitRowsFlat st)

include hR in
/-- **A composite input carries what its stacked unit input's copy position carries in the flat twin**
(`Aliased.copyPos`), in every block of a satisfying witness of the hidden statement (`copyAt_blockH`). -/
theorem inst_blockH (P : ParseFacts tags st.c) (C : CheckFacts st.c) {g : ℕ} (hg : g < st.c.g)
    (x : Fin (instRows st.c).w) (hx : x.val < (instRows st.c).nIn) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o
        (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g) (inst_hbOf L regs P C hg) x) =
      (L.model regs).block z o (Aliased.copyPos Lf regs F hg ((inX st.c).getD x.val 0)) := by
  have hx' : x.val < (inCols st.c).length := hx
  obtain ⟨he, hm⟩ := inCols_getD hx'
  have hX : (inX st.c).getD x.val 0 < st.c.upv * (st.c.unit.inWords * WORD) := inX_lt hm
  have hW : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hX; omega
  have hg0 : 0 < st.c.g := by omega
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count hg0
  have hle : (inCols st.c).length ≤ (order st.c).length := (inv_order st.c).1.length_le
  unfold Aliased.copyPos
  rw [dite_eq_left hX]
  refine copyAt_blockH L Lf regs F hR _ (Nat.mod_lt _ hW) _ ?_ hz o
  have hv : (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g)
      (inst_hbOf L regs P C hg) x).val = pos st.c g ((order st.c).getD x.val 0) :=
    (composePlace_val (S := L.model regs) (L := logical st.c) (h := logical_ordered st.c) (pos := pos st.c g)
      (hb := inst_hbOf L regs P C hg) x).trans (ite_eq_left (by show x.val < (order st.c).length; omega))
  rw [hv, order_getD_in hx', he]
  unfold inCode
  rw [pos_eq, hpv]
  rfl

end block

namespace AcceptedH

variable {T : Model.Statement} (a : AcceptedH T)

/-- The order's columns are in the block (`inst_hbOf`, from the flat twin's parse and check). -/
theorem instHb {g : ℕ} (hg : g < a.st.c.g) :
    ∀ ℓ < (logical a.st.c).order.length, pos a.st.c g ((logical a.st.c).order.getD ℓ 0) < 2 ^ (a.L.model a.regs).kLog :=
  inst_hbOf a.L a.regs a.unitFacts.parse a.unitFacts.check hg

/-- **Template instance `g`'s composite columns in the block**: `composePlace` at `pos`, the shared constant at the
pin. -/
noncomputable def instCol {g : ℕ} (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq
    (composePlace (a.L.model a.regs) (logical a.st.c) (logical_ordered a.st.c) (pos a.st.c g) (a.instHb hg) x)

theorem instCol_val {g : ℕ} (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) (hx : x.val < (order a.st.c).length) :
    (a.instCol hg x).val = pos a.st.c g ((order a.st.c).getD x.val 0) := by
  unfold instCol
  have hlen : x.val < (logical a.st.c).order.length := hx
  rw [Fin.val_cast]
  exact (composePlace_val (S := a.L.model a.regs) (L := logical a.st.c) (h := logical_ordered a.st.c)
    (pos := pos a.st.c g) (hb := a.instHb hg) x).trans (ite_eq_left hlen)

/-- **The accepted statement places the template instance's rows at every template instance**, at output rows that
fit (`placement_instOf`, `wiredDeltaH`). -/
theorem placement_inst (hrows : Hidden.Any.OutRowsP a.st.c) {g : ℕ} (hg : g < a.st.c.g) :
    Placement T (instRows a.st.c) (a.instCol hg) :=
  Placed.Accepted.placement_cast a.hT
    (placement_instOf a.L a.regs a.unitFacts.parse a.unitFacts.check (wiredDeltaH a.ht a.h hrows) hg)

/-- **Composite inputs with one key carry one value**, at template instance `g`, in every block of a satisfying
witness. -/
theorem inst_aliased {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) (x x' : Fin (instRows a.st.c).w)
    (hx : x.val < (instRows a.st.c).nIn) (hx' : x'.val < (instRows a.st.c).nIn)
    (hk : keyOf a.st.c x.val = keyOf a.st.c x'.val) {z : Model.Witness T.m} (hz : T.Satisfies z) :
    T.block z o (a.instCol hg x) = T.block z o (a.instCol hg x') := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  simp only [instCol, Fin.cast_eq_self]
  have F := flat_unitFacts ht h
  have Lf := flat_layout ht h
  rw [inst_blockH L Lf regs F hR F.parse F.check hg x hx hz o, inst_blockH L Lf regs F hR F.parse F.check hg x' hx' hz o]
  obtain ⟨-, hm⟩ := inCols_getD (c := st.c) hx
  obtain ⟨-, hm'⟩ := inCols_getD (c := st.c) hx'
  rcases Aliased.copyPos_of_key Lf regs F hg (inX_lt hm) (inX_lt hm') hk with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_blockH L Lf regs F hs hz o, zeroPos_blockH L Lf regs F hs' hz o]

/-- **A zero-keyed composite input carries 0**, at template instance `g`, in every block of a satisfying witness. -/
theorem inst_zero {g : ℕ} (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) (hx : x.val < (instRows a.st.c).nIn)
    (hk : keyOf a.st.c x.val = .zero) {z : Model.Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) :
    T.block z o (a.instCol hg x) = 0 := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  simp only [instCol, Fin.cast_eq_self]
  have F := flat_unitFacts ht h
  have Lf := flat_layout ht h
  rw [inst_blockH L Lf regs F hR F.parse F.check hg x hx hz o]
  obtain ⟨-, hm⟩ := inCols_getD (c := st.c) hx
  obtain ⟨s, hs, he⟩ := Zero.copyPos_zero Lf regs F hg (inX_lt hm) ((Zero.srcKey_zero_iff (inX_lt hm)).1 hk)
  rw [he]
  exact zeroPos_blockH L Lf regs F hs hz o

end AcceptedH

/-- **A drawn unit's message-keyed composite input holds its row's bit**: in a satisfying witness of an accepted
hidden-output statement, at block `o` and template instance `g`, the composite input whose Δ key is message bit `bit` of
byte `byte` of row `q` holds the witness's bit at that message column (`Typed.msgCol`), when the column is in the block.
`copy_siteH` at the template instance's rows. -/
theorem copy_site_instH {T : Model.Statement} (a : AcceptedH T) {z : Witness T.m} (hz : T.Satisfies z)
    (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) (hx : x.val < (instRows a.st.c).nIn)
    {q byte bit : ℕ} (hk : keyOf a.st.c x.val = .msg q byte bit)
    (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (a.instCol hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q byte bit)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hg x hx hk hin ⊢
  simp only [AcceptedH.instCol, Fin.cast_eq_self]
  have F := flat_unitFacts ht h
  have Lf := flat_layout ht h
  rw [inst_blockH L Lf regs F hR F.parse F.check hg x hx hz o]
  obtain ⟨-, hm⟩ := inCols_getD (c := st.c) hx
  have hk' : Aliased.srcKey st.c ((inX st.c).getD x.val 0) = .msg q byte bit := hk
  rcases Aliased.copyPos_key Lf regs F hg (inX_lt hm) (by simp only at hk' ⊢; rw [hk']; simp)
      (fun k off => by simp only at hk' ⊢; rw [hk']; simp) with he | ⟨hn, -⟩
  · simp only [siteAddrH]
    show z _ = z _
    congr 2
    apply Fin.ext
    rw [he]
    simp only at hk' ⊢
    rw [hk']
    simp [Aliased.realize, Layout.inBlockOr, hin]
    rfl
  · exact absurd (by simp only at hk' ⊢; rw [hk']; exact hin) hn

/-! ## `ProgPlaces`, `hConst` and `hZero` -/

section prog

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (hrows : Hidden.Any.OutRowsS c)

include hparse in
/-- **W6 for the template instance's program at hidden-output sites, from the parse.** -/
theorem SitesH.rows_instProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (instProg c n).rows u = instRows (si.acc S R u h).st.c :=
  (rows_uniform _ _ _ _ _ n u).trans (congrArg instRows ((si.acc S R u h).c_eq (hparse S R u h))).symm

/-- **A drawn unit's columns in its table's block**: its template instance's composite columns
(`AcceptedH.instCol`). -/
noncomputable def SitesH.instCol (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin ((instProg c n).rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun x => (si.acc S R u h).instCol (si.hvu S R u h) (Fin.cast (congrArg Rows.w (SitesH.rows_instProg hparse S R u h)) x)

variable (outs : Finset (Fin (instN c n)))

/-- **`ProgPlaces` for the program of the whole template instance at hidden-output sites**, from the sites and the
parse, at output rows that fit: each drawn unit placed at its template instance (`AcceptedH.placement_inst`), its
instance the program's own, its composite inputs on one gate carrying one value (`AcceptedH.inst_aliased`). -/
noncomputable def SitesH.progPlaces_instProg : ProgPlaces (instProg c n) outs plan tab where
  inst := Const.Prog.instC (instProg c n) outs
  o := si.o
  col := SitesH.instCol hparse
  placed S R u h := ⟨si.A S R u h, si.B S R u h, placement_rows (SitesH.rows_instProg hparse S R u h)
    ((si.acc S R u h).placement_inst (by rw [(si.acc S R u h).c_eq (hparse S R u h)]; exact hrows.toP) (si.hvu S R u h)),
    si.sat S R u h⟩
  aliased S R u h x x' hw hx hx' z hz := by
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have e := congrArg Rows.nIn (SitesH.rows_instProg hparse S R u h)
    have hk : keyOf (si.acc S R u h).st.c x.val = keyOf (si.acc S R u h).st.c x'.val := by
      rw [hc]; exact instProg_wire_key c outs u x x' hx hx' hw
    exact (si.acc S R u h).inst_aliased (si.hvu S R u h) (si.o S R u h)
      (Fin.cast (congrArg Rows.w (SitesH.rows_instProg hparse S R u h)) x)
      (Fin.cast (congrArg Rows.w (SitesH.rows_instProg hparse S R u h)) x') (lt_of_lt_of_eq hx e)
      (lt_of_lt_of_eq hx' e) hk (si.sat S R u h z hz)

/-- `progPlaces_instProg`'s instance is the program's own. -/
theorem SitesH.progPlaces_instProg_inst (u : Fin n) :
    (SitesH.progPlaces_instProg hparse hrows outs).inst u = Const.Prog.instC (instProg c n) outs u :=
  rfl

/-- **`hConst` at `SitesH.progPlaces_instProg`**: only a unit's constant column sits on the one gate. -/
theorem constCols_instProgH :
    ((instProg c n).partition outs).ConstCols plan tab (SitesH.progPlaces_instProg hparse hrows outs).derived.place
      {instOneG (c := c) (n := n)} :=
  Const.constCols_of_instC _ (Zero.oneGate_uniform n) (SitesH.progPlaces_instProg_inst hparse hrows outs)

/-- **`hZero` at `SitesH.progPlaces_instProg`**: every column on the zero gate is a zero-keyed composite input, which
carries 0 in every satisfying witness (`AcceptedH.inst_zero`). -/
theorem zeroCols_instProgH :
    ((instProg c n).partition outs).ZeroCols plan tab (SitesH.progPlaces_instProg hparse hrows outs).derived.place
      {instZeroG (c := c) (n := n)} := by
  refine Zero.Prog.zeroCols_singleton _ (one := instOneG (c := c) (n := n))
    (Zero.unit_uniform n _ (zeroGate n (instB c)).isLt) instZeroG_ne_instOneG (fun u => ?_) ?_
  · rw [SitesH.progPlaces_instProg_inst, Const.Prog.instC_wire_one]
    exact Zero.oneGate_uniform n u
  · intro S R u h x hx hw z hsat
    rw [SitesH.progPlaces_instProg_inst, Const.Prog.instC_wire_input _ _ _ x hx] at hw
    have hxB : x.val < instB c := lt_of_lt_of_eq hx (congrArg Rows.nIn (rows_instProg (c := c) u))
    have hv := congrArg Fin.val hw
    rw [insAt_uniform n u x.val hxB] at hv
    have hkz : keyOf c x.val = .zero :=
      (instIdx_eq_none (⟨x.val, hxB⟩ : Fin (instB c))).1 ((insOf_eq_zero_iff u.isLt _).1 (Fin.ext hv))
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have e := congrArg Rows.nIn (SitesH.rows_instProg hparse S R u h)
    have hkz' : keyOf (si.acc S R u h).st.c x.val = .zero := by rw [hc]; exact hkz
    exact (si.acc S R u h).inst_zero (si.hvu S R u h)
      (Fin.cast (congrArg Rows.w (SitesH.rows_instProg hparse S R u h)) x) (lt_of_lt_of_eq hx e) hkz' (si.sat S R u h z hsat) _

end prog

/-! ## Output rows' bits at `instProg` -/

section outs

variable {c : Flock.Circuit} {n : ℕ}

variable (c) in
/-- **Composite column `x` holds bit `t` of output word `o`**: past the composite inputs, at every template instance
where the circuit computes the word (`HmOut.outCol`, the unit's or a tail's). -/
def OutColAtI (x o t : ℕ) : Prop :=
  (instRows c).nIn ≤ x ∧ x < (order c).length ∧ o < c.outWords ∧ t < 16 ∧
    ∀ g, pos c g ((order c).getD x 0) = HmOut.outCol c g o + t

variable (c) in
/-- **Bit `t` of output word `o`'s output-row bit** of unit `u`: bit `16 k + t` of output row `nIn + jo`, where `o` is
word `k` of output port `jo` (`outPortOf`); none past the row. -/
def outBitRowO (u : Fin n) (o t : ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  if hq : HmOut.nIn c + outPortOf c o < c.ports.size then
    if hk : 16 * (o - Hidden.outAt c (outPortOf c o)) + t < portRowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

theorem outBitRowO_spec {u u' : Fin n} {o t : ℕ} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outBitRowO c u o t = some (u', ⟨q, k⟩)) :
    u' = u ∧ q.val = HmOut.nIn c + outPortOf c o ∧ k.val = 16 * (o - Hidden.outAt c (outPortOf c o)) + t := by
  unfold outBitRowO at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨hu, hs⟩ := h
      subst hu
      cases hs
      exact ⟨rfl, rfl, rfl⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- **An output word's bit is an output row's**, in a circuit whose output rows hold their ports' words
(`HmOut.checkRows`). -/
theorem outBitRowO_ne_none (hcr : HmOut.checkRows c = .ok ()) (hrf : Hidden.Any.RowsFitS c) (u : Fin n) {o t : ℕ}
    (ho : o < c.outWords) (ht : t < 16) : outBitRowO c u o t ≠ none := by
  obtain ⟨hjo, -, hk⟩ := outPortOf_spec c ho
  have hle := checkRows_words hcr hrf.1.1 hrf.bits16 hjo
  have hq := (hrf.1.2 _ hjo).1
  have hpw : portRowW c ⟨_, hq⟩ = 16 * (c.ports.getD (HmOut.nIn c + outPortOf c o) default).words := by
    unfold portRowW
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    rfl
  unfold outBitRowO
  rw [dite_eq_left hq, dite_eq_left (by rw [hpw]; omega)]
  exact Option.some_ne_none _

/-- **`instProg`'s units read only input gates besides their own**: a column of a unit's instance that is no computed
column, a composite input or the constant, is on a program input. -/
theorem instProg_reads (outs : Finset (Fin (instN c n))) (u : Fin n) (y : Fin ((instProg c n).rows u).w)
    (hy : ∀ i, y ≠ ((instProg c n).rows u).comp i) :
    ((instProg c n).partition outs).unit ((Const.Prog.instC (instProg c n) outs u).wire y) = none := by
  show (instProg c n).unit _ = none
  refine _root_.FlockSoundness.Discharge.Layout.unit_uniform_input (k := inGates n (instB c)) (R := instRows c)
    (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
    (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n _ ?_
  by_cases hx : y.val < ((instProg c n).rows u).nIn
  · rw [Const.Prog.instC_wire_input _ _ _ y hx]
    have e := congrArg FlockSoundness.Rows.nIn (rows_instProg (c := c) u)
    have hxB : y.val < instB c := lt_of_lt_of_eq hx e
    have hv : (Const.Prog.insAt (instProg c n) u y.val).val =
        (insOf n (instB c) (instIdx c (instB c)) u.val ⟨y.val, hxB⟩).val :=
      insAt_uniform n u y.val hxB
    rw [hv]
    exact Fin.isLt _
  · have hone : y = ((instProg c n).rows u).one := by
      apply Fin.ext
      have hw : y.val < ((instProg c n).rows u).nIn + ((instProg c n).rows u).nComp + 1 := y.isLt
      by_contra hne
      have hi : y.val - ((instProg c n).rows u).nIn < ((instProg c n).rows u).nComp := by
        simp only [FlockSoundness.Rows.one] at hne
        omega
      exact hy ⟨_, hi⟩ (Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega))
    rw [hone, Const.Prog.instC_wire_one]
    have h0 : (Const.Prog.oneAt (instProg c n) u).val = (oneGate n (instB c)).val :=
      _root_.FlockSoundness.Discharge.Layout.oneAt_uniform_val (k := inGates n (instB c)) (R := instRows c)
        (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
        (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n u
    rw [h0]
    exact Fin.isLt _

/-- **A unit's own column on a gate it computes** is a computed column, of that unit. -/
theorem unit_wireI (outs : Finset (Fin (instN c n))) {u u' : Fin n} (x : Fin ((instProg c n).rows u).w)
    (h : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u') :
    u' = u ∧ ((instProg c n).rows u).nIn ≤ x.val := by
  by_cases hcomp : ∃ i, x = ((instProg c n).rows u).comp i
  · obtain ⟨i, rfl⟩ := hcomp
    have hm : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire
        (((instProg c n).rows u).comp i)) = some u := (Const.Prog.instC (instProg c n) outs u).mem i
    rw [hm] at h
    refine ⟨(Option.some.inj h).symm, ?_⟩
    simp only [FlockSoundness.Rows.comp]
    omega
  · have hn : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = none :=
      instProg_reads outs u x fun i e => hcomp ⟨i, e⟩
    rw [hn] at h
    cases h

variable (c n) in
open Classical in
/-- **The output-row bit of a unit's computed gate**: the bit (`outBitRowO`) of an output word whose bit the gate's
column of its unit's instance holds (`OutColAtI`). -/
noncomputable def instOutRowOf (outs : Finset (Fin (instN c n))) (g : Fin (instN c n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (instProg c n).unit g with
  | some u =>
    if h : ∃ (x : Fin ((instProg c n).rows u).w) (p : ℕ × ℕ),
        (Const.Prog.instC (instProg c n) outs u).wire x = g ∧ OutColAtI c x.val p.1 p.2 then
      outBitRowO c u (Classical.choose (Classical.choose_spec h)).1 (Classical.choose (Classical.choose_spec h)).2
    else none
  | none => none

theorem instOutRowOf_wire (outs : Finset (Fin (instN c n))) {u u' : Fin n} (x : Fin ((instProg c n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : instOutRowOf c n outs ((Const.Prog.instC (instProg c n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u ∧
      ∃ o t, OutColAtI c x.val o t ∧ outBitRowO c u o t = some (u', ⟨q, k⟩) := by
  rcases hgu : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) with _ | u₀
  · unfold instOutRowOf at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hxn⟩ := unit_wireI outs x hgu
    refine ⟨rfl, ?_⟩
    unfold instOutRowOf at h
    rw [hgu] at h
    dsimp only at h
    split at h
    · rename_i hex
      have hs := Classical.choose_spec (Classical.choose_spec hex)
      generalize Classical.choose (Classical.choose_spec hex) = p at hs h
      generalize Classical.choose hex = x' at hs
      have hx0 : x' = x := by
        rcases (Const.Prog.instC (instProg c n) outs u₀).shared _ x hs.1 with he | ⟨-, h2⟩
        · exact he
        · omega
      subst hx0
      exact ⟨_, _, hs.2, h⟩
    · exact absurd h (by simp)

theorem instOutRowOf_ne_none (outs : Finset (Fin (instN c n))) (hcr : HmOut.checkRows c = .ok ())
    (hrf : Hidden.Any.RowsFitS c) {u : Fin n} (x : Fin ((instProg c n).rows u).w)
    (hgu : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u) {o t : ℕ}
    (hoc : OutColAtI c x.val o t) :
    instOutRowOf c n outs ((Const.Prog.instC (instProg c n) outs u).wire x) ≠ none := by
  have hex : ∃ (x' : Fin ((instProg c n).rows u).w) (p : ℕ × ℕ),
      (Const.Prog.instC (instProg c n) outs u).wire x' = (Const.Prog.instC (instProg c n) outs u).wire x ∧
        OutColAtI c x'.val p.1 p.2 := ⟨x, (o, t), rfl, hoc⟩
  unfold instOutRowOf
  rw [hgu]
  dsimp only
  rw [dite_eq_left hex]
  have hs := (Classical.choose_spec (Classical.choose_spec hex)).2
  exact outBitRowO_ne_none hcr hrf u hs.2.2.1 hs.2.2.2.1

variable (c n) in
/-- **The row bits of `instProg` at a hidden-output statement**: an input row's bit (`instRowOf`), else an output row's
(`instOutRowOf`). -/
noncomputable def instRowOfH (outs : Finset (Fin (instN c n))) (g : Fin (instN c n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match instRowOf c n (portRowW c) g with
  | some r => some r
  | none => instOutRowOf c n outs g

theorem instRowOfH_of_key (outs : Finset (Fin (instN c n))) {g : Fin (instN c n)}
    {r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)} (h : instRowOf c n (portRowW c) g = some r) :
    instRowOfH c n outs g = some r := by
  unfold instRowOfH
  rw [h]

theorem instRowOfH_of_none (outs : Finset (Fin (instN c n))) {g : Fin (instN c n)}
    (h : instRowOf c n (portRowW c) g = none) : instRowOfH c n outs g = instOutRowOf c n outs g := by
  unfold instRowOfH
  rw [h]

theorem instRowOfH_none (outs : Finset (Fin (instN c n))) {g : Fin (instN c n)} (h : instRowOfH c n outs g = none) :
    instRowOf c n (portRowW c) g = none ∧ instOutRowOf c n outs g = none := by
  rcases hk : instRowOf c n (portRowW c) g with _ | r
  · exact ⟨rfl, (instRowOfH_of_none outs hk).symm.trans h⟩
  · rw [instRowOfH_of_key outs hk] at h
    cases h

/-- **A row bit of a unit's own instance is that unit's row**. -/
theorem instRowOfH_wire_unit (outs : Finset (Fin (instN c n))) {u u' : Fin n} (x : Fin ((instProg c n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : instRowOfH c n outs ((Const.Prog.instC (instProg c n) outs u).wire x) = some (u', ⟨q, k⟩)) : u' = u := by
  rcases hk : instRowOf c n (portRowW c) ((Const.Prog.instC (instProg c n) outs u).wire x) with _ | r
  · rw [instRowOfH_of_none outs hk] at h
    obtain ⟨-, o, t, -, hb⟩ := instOutRowOf_wire outs x h
    exact (outBitRowO_spec hb).1
  · rw [instRowOfH_of_key outs hk] at h
    rw [h] at hk
    exact (instRowOf_wire outs x hk).1

variable (c n) in
/-- **The committed gates a unit of `instProg` computes are output words' bits** (a condition on the program's outputs
`outs`): each column of the unit's instance on one holds an output word's bit (`OutColAtI`). -/
def OutsOutI (outs : Finset (Fin (instN c n))) : Prop :=
  ∀ u g, g ∈ ((instProg c n).partition outs).io u → (instProg c n).unit g = some u →
    ∀ x : Fin ((instProg c n).rows u).w, (Const.Prog.instC (instProg c n) outs u).wire x = g →
      ∃ o t, OutColAtI c x.val o t

variable (c n) in
/-- **The output words' bits**: the gates the units' instances put on columns holding an output word's bit
(`OutColAtI`). -/
noncomputable def outsOfI : Finset (Fin (instN c n)) :=
  @Finset.filter _ (fun g => ∃ u x, (Const.Prog.rowsWireC (instProg c n) u).wire x = g ∧
    ∃ o t, OutColAtI c x.val o t) (Classical.decPred _) Finset.univ

/-- **`OutsOutI` at the output words' bits**: every committed gate a unit computes is an output
(`computed_committed_output`, `instProg_reads`), so on a column holding an output word's bit of the unit's instance. -/
theorem outsOutI_outsOfI : OutsOutI c n (outsOfI c n) := by
  intro u g hg hgu x hx
  have hmem : g ∈ outsOfI c n :=
    Layout.computed_committed_output (fun u => Const.Prog.instC (instProg c n) (outsOfI c n) u)
      (fun u y hy => instProg_reads _ u y hy) hg hgu
  obtain ⟨-, u₁, x₁, hw₁, o, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hmem
  have eN : ((instProg c n).rows u₁).nIn = (instRows c).nIn :=
    congrArg FlockSoundness.Rows.nIn (rows_instProg (c := c) u₁)
  have eC : ((instProg c n).rows u₁).nComp = (order c).length - (instRows c).nIn :=
    (congrArg FlockSoundness.Rows.nComp (rows_instProg (c := c) u₁)).trans rfl
  have hnx := hoc.1
  have hxo := hoc.2.1
  have hi : x₁.val - ((instProg c n).rows u₁).nIn < ((instProg c n).rows u₁).nComp := by
    rw [eN, eC]; omega
  have hx₁ : x₁ = ((instProg c n).rows u₁).comp ⟨_, hi⟩ :=
    Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
  have hu₁ : u₁ = u := by
    have hm := (Const.Prog.instC (instProg c n) (outsOfI c n) u₁).mem ⟨_, hi⟩
    rw [← hx₁] at hm
    have hm' : (instProg c n).unit g = some u₁ := hw₁ ▸ hm
    rw [hgu] at hm'
    exact (Option.some.inj hm').symm
  subst hu₁
  rcases (Const.Prog.instC (instProg c n) (outsOfI c n) u₁).shared x x₁ (hx.trans hw₁.symm) with he | ⟨-, h2'⟩
  · exact ⟨o, t, by rw [he]; exact hoc⟩
  · omega

end outs

/-! ## Output-row copies at an accepted statement -/

/-- **A composite column holding an output word's bit holds its output row's bit**: in a satisfying witness of an
accepted statement on `c` whose rows are covered (`Hidden.Any.RowsFitS`), at block `o'` and template instance `g`, the
composite column holding bit `t` of output word `o` (`OutColAtI`) holds the witness's bit at the message column of its
output row's bit (`outBitRowO`): Δ copies the word's bit (`HmOut.outCol`) there (`Hidden.Any.msg_copyF`). -/
theorem out_copy_instH {T : Model.Statement} {c : Flock.Circuit} {n : ℕ} (a : AcceptedH T) (hce : a.st.c = c)
    (hrf : Hidden.Any.RowsFitS c) {z : Witness T.m} (hz : T.Satisfies z) (o' : Fin (2 ^ T.nbl)) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) {o t : ℕ} (hoc : OutColAtI c x.val o t) {u v : Fin n}
    {q : Fin c.ports.size} {k : Fin (portRowW c q)} (hb : outBitRowO c u o t = some (v, ⟨q, k⟩)) :
    T.block z o' (a.instCol hg x) = z (siteAddrH a o' (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRowO_spec hb
  obtain ⟨-, hxo, ho, ht, hpos⟩ := hoc
  have hcol := a.instCol_val hg x hxo
  rw [hpos g] at hcol
  obtain ⟨hjo, hle, hkk⟩ := outPortOf_spec a.st.c ho
  have hrows := hrf.1
  have hkw : o - Hidden.outAt a.st.c (outPortOf a.st.c o) <
      (a.st.c.ports.getD (HmOut.nIn a.st.c + outPortOf a.st.c o) default).words :=
    lt_of_lt_of_le hkk (checkRows_words a.checkRows hrows.1 hrf.bits16 hjo)
  have hsrc : Hidden.outSrc a.st.c g (outPortOf a.st.c o) (o - Hidden.outAt a.st.c (outPortOf a.st.c o)) t =
      HmOut.outCol a.st.c g o + t := by
    unfold Hidden.outSrc
    rw [Hidden.outGroup_single hrows.1 hjo]
    dsimp only
    simp only [hkk, ↓reduceIte]
    rw [Nat.add_sub_cancel' hle]
  have hp := (hrows.2 _ hjo).1
  have hpa := a.parse
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hg x hcol hjo hle hkk hrows hkw hsrc hp hpa hq hk hrf ⊢
  have s := Hidden.setupHidden_spec' h
  have L0 := flat_layout ht' h
  have F0 := flat_unitFacts ht' h
  have hda : (Hidden.unpin st).da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := rfl
  have hmc := Hidden.Any.msg_copyF L L0 F0 hda hrows hz o' hg hjo hkw ht
  generalize o - Hidden.outAt st.c (outPortOf st.c o) = kk at hk hkw hsrc hmc
  have hb' : (16 * kk + t) / 8 = 2 * kk + t / 8 := by omega
  have hb8 : (16 * kk + t) % 8 = t % 8 := by omega
  have hin := Layout.msgCol_lt_of_parse hpa hg hp (2 * kk + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hq, hk, hb', hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o' hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + outPortOf st.c o) (2 * kk + t / 8) (t % 8), hmc, hsrc,
    ← hcol, blockBit_of_lt _ z o' (Fin.isLt _)]

/-! ## The readers -/

section places

variable {c : Flock.Circuit} {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesH plan tab)

/-- **Row `q`'s salt at a drawn unit's site**: its `hm96` slot's salt input (`Hm.saltCol`) at the unit's template
instance (`keySaltAddrH` on `instProg`'s gates). -/
noncomputable def instSaltAddrH (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (q : (Fin n × Fin c.ports.size) ⊕ Fin (instN c n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      (siteAddrH (si.acc S R u h) (si.o S R u h) (Hm.saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val) :)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (hrf : Hidden.Any.RowsFitS c) (outs : Finset (Fin (instN c n)))

/-- `SitesH.progPlaces_instProg`'s places. -/
noncomputable abbrev instPlaceH := (SitesH.progPlaces_instProg hparse hrf.1 outs).derived.place

variable (popOf : Reg → Public)

/-- **The readers at `instProg`'s sites in hidden-output tables**: every row `(u, q)` of `16 · words` bits at its
message columns (`keyRowAddrH`), its `b ‖ c` at the `hm96` output (`keyBcAddrH`) and salt at `Hm.saltCol`
(`instSaltAddrH`), the registration's `b ‖ c` the population's public file's (`popOf R`), and single gates with
`instGate` at no output bit. -/
noncomputable def instReadersH : Layout.Readers ((instProg c n).circuit outs) plan tab :=
  Layout.Readers.ofRows (instPlaceH hparse hrf outs) (instN_pos c n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (instRowOfH c n outs)
    (keyRowAddrH si) (instSaltAddrH si) (keyBcAddrH si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (instGate (c := c) (n := n) fun (_ : Reg) _ => false)

include hparse in
/-- **Δ's copies at `instProg`'s sites in hidden-output tables**: a column on an input row's bit holds the witness's
bit at its key's message column (`copy_site_instH`); a column on an output row's bit holds the bit at the output row's
message column (`out_copy_instH`). -/
theorem instProgH_copies : (instReadersH hparse hrf outs popOf).Copies (instPlaceH hparse hrf outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesH.rows_instProg hparse S R u hu
  rcases hk : instRowOf c n (portRowW c) ((Const.Prog.instC (instProg c n) outs u).wire x) with _ | r
  · have ho' : instOutRowOf c n outs ((Const.Prog.instC (instProg c n) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (instRowOfH_of_none outs hk).symm.trans hr
    obtain ⟨-, o, t, hoc, hb⟩ := instOutRowOf_wire outs x ho'
    have := out_copy_instH (si.acc S R u hu) hc hrf (si.sat S R u hu _ hM) (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg Rows.w hrw) x) (o := o) (t := t) hoc hb
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this
  · have hr' : instRowOf c n (portRowW c) ((Const.Prog.instC (instProg c n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((instRowOfH_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := instRowOf_wire outs x hr'
    have hk' : keyOf (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_site_instH (si.acc S R u hu) (si.sat S R u hu _ hM) (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg Rows.nIn hrw)) hk' hin'
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this

include hparse in
/-- **The rows' `b ‖ c` at `instProg`'s sites in hidden-output tables is the registration's**: at each row a drawn
unit reads, input or output, the bytes at its `hm96` output are the population's `b ‖ c` of the unit (`bc_site_ofH`). -/
theorem instProgH_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (instReadersH hparse hrf outs popOf).BcPinned (P := (instProg c n).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((instPlaceH hparse hrf outs) S R u hu).inst hg
  rw [instRowOfH_wire_unit outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofH (hsite S R u hu M hM) (si.sat S R u hu _ hM) (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrH
  simp only [hu, ↓reduceDIte]
  exact this

variable (houts : OutsOutI c n outs)

include hparse houts in
/-- **The single gates at `instProg`'s sites in hidden-output tables are the constant and the zero**: when every
composite input a row reads has a row key (`InstKeysRow`), a committed gate that is no row's bit is the zero, the
constant, or a computed gate (`inst_single_gate`); a computed one holds an output word's bit (`OutsOutI`), so it is an
output row's bit (`instOutRowOf_ne_none`). So the gates hold 0 and 1 (`zeroCols_instProgH`, `constCols_instProgH`). -/
theorem instProgH_valPinned (hkeys : InstKeysRow c (portRowW c)) :
    (instReadersH hparse hrf outs popOf).ValPinned (P := (instProg c n).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (instPlaceH hparse hrf outs) (constCols_instProgH hparse hrf.1 outs)
    (zeroCols_instProgH hparse hrf.1 outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((instPlaceH hparse hrf outs) S R u hu).inst hg
  change bit _ = instGate (c := c) (n := n) (fun (_ : Reg) _ => false) R g
  obtain ⟨hk, hno⟩ := instRowOfH_none outs hr
  rcases inst_single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [instGate, instZeroG_ne_instOneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [instGate, ↓reduceIte]
  · exfalso
    subst hx
    obtain ⟨o, t, hoc⟩ := houts u _ hg hm x rfl
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hcr : HmOut.checkRows c = .ok () := hc ▸ (si.acc S R u hu).checkRows
    exact instOutRowOf_ne_none outs hcr hrf x hm hoc hno

/-- **`lay` at `instProg`'s places in hidden-output tables**: the readers' layout (`Layout.Readers.layout`), with
`bitCode` for the single gates and Δ's copies from `instProgH_copies`. -/
noncomputable def instLayoutH :
    Binding.Layout ((instProg c n).partition outs) plan tab
      (((instProg c n).partition outs).placeDecoder plan tab (instPlaceH hparse hrf outs)) :=
  (instReadersH hparse hrf outs popOf).layout (instPlaceH hparse hrf outs) Layout.bitCode
    (instProgH_copies hparse hrf outs popOf)

/-- **`rs` at `instProg`'s places in hidden-output tables**: the readers' rows each under its own encoding
(`Layout.Readers.rowsX`), `encPortH` at each row, from `instProgH_copies`, `instProgH_bcPinned` and
`instProgH_valPinned`. -/
noncomputable def instRowsH
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : InstKeysRow c (portRowW c)) :
    Binding.HmRowsX (instLayoutH hparse hrf outs popOf) :=
  (instReadersH hparse hrf outs popOf).rowsX (instPlaceH hparse hrf outs) (encKeyH c) (extKeyH c) (extKeyH_spec c)
    Layout.bitCode Layout.bitCode_inj (instProgH_copies hparse hrf outs popOf)
    (instProgH_bcPinned hparse hrf outs popOf hsite) (instProgH_valPinned hparse hrf outs popOf houts hkeys)

/-- **The bytes a row's hash reads at `instProg`'s places in hidden-output tables** (`keyRowH_enc` at `instLayoutH`). -/
theorem instRowH_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) (hrw : portRowW c q = 16 * c.ports[q.val]!.words)
    (hb : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words) :
    encPortH c.ports[q.val]! ((instLayoutH hparse hrf outs popOf).row S R u M (.inl (u', q))) =
      Hidden.Any.rowMsg c.ports[q.val]!
        (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          ((siteAddrH (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) :)))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hbytes : (List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t =>
      (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) => keyRowAddrH si S R u q k.val)
        ⟨8 * j.val + t.val, by omega⟩) =
      List.ofFn fun j : Fin (2 * c.ports[q.val]!.words) => byteOf fun t => bit (Binding.wit plan tab S R u M
        ((siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) :))) := by
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

/-- **`HmRowComputes` at `instProg`'s rows in hidden-output tables** (`hHm` for `instRowsH`), with a unit per VU
(`hupv`): at every drawn unit's committed wire held by a row, input or output, the `b ‖ c` at the row's `hm96` output is
the commit string of the row's bytes and the salt the witness holds at `Hm.saltCol` (`bc_site_commitH`,
`instRowH_enc`). -/
theorem instRowsH_computes (hupv : 0 < c.upv)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : InstKeysRow c (portRowW c)) :
    (instRowsH hparse hrf outs popOf houts hsite hkeys).Computes := by
  unfold Binding.HmRowsX.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (instLayoutH hparse hrf outs popOf).pos g = ps
  rcases ps with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * c.ports[q.val]!.words := by
      rw [getElem!_pos c.ports q.val q.isLt]
      rfl
    have hbq : c.ports[q.val]!.bits = 16 * c.ports[q.val]!.words :=
      hrf.bits16 _ (by rw [getElem!_pos c.ports q.val q.isLt]; exact Array.getElem_mem_toList q.isLt)
    have hb := bc_site_commitH (si.acc S R u hu) (by rw [hce]; exact hupv) (by rw [hce]; exact hrf) (si.sat S R u hu _ hM)
      (si.o S R u hu) (si.hvu S R u hu) hq
    have hpq : (si.acc S R u hu).st.c.ports[q.val]! = c.ports[q.val]! := by rw [hce]
    rw [hpq] at hb
    show Binding.bytesAt _ (keyBcAddrH si S R u q) = Binding.commitString
      (encPortH c.ports[q.val]! ((instLayoutH hparse hrf outs popOf).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (instSaltAddrH si S R u (.inl (u', q))))
    have e1 : keyBcAddrH si S R u q = fun (j : Fin 128) (t : Fin 8) => (siteAddrH (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) :) := by
      funext j t
      simp only [keyBcAddrH, hu, dite_true]
    have e2 : instSaltAddrH si S R u (.inl (u', q)) = fun (j : Fin 192) (t : Fin 8) =>
        (siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) :) := by
      funext j t
      simp only [instSaltAddrH, hu, dite_true]
    rw [instRowH_enc hparse hrf outs popOf hu M u' q hwq hbq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.HiddenExec
