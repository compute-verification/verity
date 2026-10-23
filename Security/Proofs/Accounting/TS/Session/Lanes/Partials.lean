import Proofs.Accounting.TS.Session.Lanes.KnownRead
import PouwWindow.OwnPorts

/-!
# Lane `l`'s partials row is the combine's `p<l>`

Lane `l`'s statement reads its partials row at its port `partials`, and the combine reads it at `p<l>`: two ports of two
circuits, of one registered value. Where each session's `--registered` file holds `PouwWindow.Lanes.ownOf` (one table
per value at the unit's placement, `place v`, registered at `roots (place v).1`, each port read at its layout row from
the table's base row) and flock-verify's known-read check holds each port the file
marks `read_in_full` (`PouwWindow.Lanes.readFull`) to one row of its registered words whose every bit the unit reads
(`Flock.KnownRead.fullRows`), the two statements decode one row:

* `regPort`, `mem_ownOf`, `partials_lane`, `partials_comb`, `full_of`, `mem_fullRows`: the two ports are `ownOf`'s, both
  of the partials' table at the base row plus `l`, and both read in full (`full_of`: a read of an A, B or partials
  row), so the check holds each to one row of that table's `regWords` words;
* `ReadsOwnRow`: every unit's registrant reads a port that `own` reads at one row (`reads = #[some r]`) at that row,
  which acceptance gives (`RowsSession.readsOwnRow_of`);
* `portBit_none`: no bit at or past a port's row width is read;
* `read_bits`, `partials_bits`: where two sessions' registrants' rows agree at one tree and position (`RowsAgree`), two
  ports that read one row of one value in full have one function for bits (lane `l`'s `partials` and the combine's
  `p<l>`; two lanes' reads of one A or B row, `Lanes.Committed`): below their common width the opened row's bits
  (`portBit_opened`), one row's bytes so one row's bits (`FlockVBridge.Family.bits_of_encPortH`); at and past it none;
* `fieldBit_partials`, `fieldBit_combine`, `part_eq`: both decodes read the partials words from bit 0, so lane `l`'s
  decoded partials row is the combine's row `l`.
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge Accounting.TS.Checked FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (portRowW)

/-- `ownOf`'s port `nm`: the leaf's row `row` of the value `v`, row `(place v).2 + row` of the table `(place v).1`,
registered at `roots (place v).1`. -/
def regPort (roots : String → _root_.Flock.Registered.Port) (place : String → String × ℕ) (nm v : String) (row : ℕ) :
    _root_.Flock.Registered.Port :=
  { roots (place v).1 with name := nm, value := (place v).1, readsRel := #[some ((place v).2 + row)], readsFrom := 0 }

theorem regWords_regPort (roots : String → _root_.Flock.Registered.Port) (place : String → String × ℕ) (nm v : String)
    (row : ℕ) : _root_.Flock.KnownRead.regWords (regPort roots place nm v row) =
      _root_.Flock.KnownRead.regWords (roots (place v).1) := rfl

theorem mem_ownOf (roots : String → _root_.Flock.Registered.Port) (place : String → String × ℕ) {T : Tile}
    {k : Option ℕ} {nm v : String} {row : ℕ} (h : (nm, v, row) ∈ regReads T k) :
    regPort roots place nm v row ∈ ownOf roots place T k :=
  List.mem_map.2 ⟨_, h, rfl⟩

/-- Lane `l` reads row `l` of `partials` at its port `partials`. -/
theorem partials_lane (T : Tile) (l : ℕ) : ("partials", "partials", l) ∈ regReads T (some l) := by
  refine List.mem_filterMap.2 ⟨⟨"partials", .read "partials" l, fieldsFrom 0 [("partials", 32 * T.nw)], false⟩, ?_, rfl⟩
  show _ ∈ lanePorts T l
  unfold lanePorts
  exact List.mem_append_right _ (by simp)

/-- The combine reads row `x` of `partials` at its port `p<x>`. -/
theorem partials_comb (T : Tile) {x : ℕ} (hx : x < T.lanes) : (s!"p{x}", "partials", x) ∈ regReads T none :=
  List.mem_filterMap.2 ⟨⟨s!"p{x}", .read "partials" x, fieldsFrom 0 [(s!"partials[{x}]", 32 * T.nw)], false⟩,
    List.mem_append_left _ (List.mem_map.2 ⟨x, List.mem_range.2 hx, rfl⟩), rfl⟩

/-- A read of a `fullValue` (an A, B or partials row) is a port read in full. -/
theorem full_of {T : Tile} {k : Option ℕ} {nm v : String} {row : ℕ} (h : (nm, v, row) ∈ regReads T k)
    (hv : fullValue v = true) : nm ∈ readFull T k :=
  List.mem_toArray.2 (List.mem_filterMap.2 ⟨_, h, by simp only [hv, if_true]⟩)

/-- A port read in full is checked as one row of its registered port's words. -/
theorem mem_fullRows {own : Array _root_.Flock.Registered.Port} {names : Array String} {nm : String}
    {P : _root_.Flock.Registered.Port} (hn : nm ∈ names) (hf : own.toList.find? (·.name == nm) = some P) :
    (nm, #[(Array.replicate (_root_.Flock.KnownRead.regWords P) 0, ByteArray.empty)]) ∈
      _root_.Flock.KnownRead.fullRows own names := by
  unfold _root_.Flock.KnownRead.fullRows
  exact Array.mem_map.2 ⟨nm, hn, by rw [hf]; rfl⟩

theorem fieldBit_partials (T : Tile) : fieldBit (lanePorts T 0) "partials" = 0 := rfl

/-- Every field of the combine's ports is at bit 0 of its port's row. -/
theorem fieldBit_combine (T : Tile) (path : String) : fieldBit (combinePorts T) path = 0 := by
  unfold fieldBit
  cases h : fieldOf? (combinePorts T) path with
  | none => rfl
  | some pf =>
    obtain ⟨p, hp, hf⟩ := List.exists_of_findSome?_eq_some h
    obtain ⟨f, hf', rfl⟩ := Option.map_eq_some_iff.1 hf
    have hfm := List.mem_of_find?_eq_some hf'
    show f.bit = 0
    simp only [combinePorts, okPort, List.mem_append, List.mem_map, List.mem_range, List.mem_cons,
      List.not_mem_nil, or_false] at hp
    rcases hp with ⟨x, -, rfl⟩ | rfl | rfl <;>
      simp only [fieldsFrom, List.mem_cons, List.not_mem_nil, or_false] at hfm <;> subst hfm <;> rfl

/-- **No bit at or past a port's row width is read**: `portBit` is `none` at a bit no port of the name holds. -/
theorem portBit_none {s : RowsSession} (v : ℕ → ℕ → Option Bool) {name : String} {b : ℕ}
    (h : ∀ q : Fin s.c.ports.size, s.c.ports[q].name = name → portRowW s.c q ≤ b) : portBit s v name b = none := by
  unfold portBit
  refine dif_neg ?_
  rintro ⟨_, _, q, k, _, hn, hk, _⟩
  have := h q hn
  have := k.isLt
  omega

/-- **Every unit reads a one-row port at its row**: wherever `own`'s port of `q`'s name reads one row `r`
(`reads = #[some r]`), every unit's registrant reads `q` at `r`. -/
def ReadsOwnRow (s : RowsSession) : Prop :=
  ∀ (u : Fin s.n) (q : Fin s.c.ports.size), (∃ pt ∈ s.own, pt.name = s.c.ports[q].name) → ∀ r : ℕ,
    (s.ownPort q).reads = #[some r] → (s.rdU u).readAt (.inl (u, q)) = r

/-- **Two statements' reads of one registered row in full read its bits alike**: at sessions `sL` and `sC` (statements
`kL` and `kC`, the same or not) whose registered files hold `ownOf roots place T` (one placement for both) and pass
the complete check, and the known-read check at each port `readFull` names, whose registrants' rows agree at one tree
and position, and whose units read their one-row ports at their rows, `portBit` at two ports that read the leaf's row
`row` of one value `v` (row `(place v).2 + row` of its table), each read in full, is one function: below their common width the opened row's bits (`portBit_opened`), one row's bytes so one
row's bits (`FlockVBridge.Family.bits_of_encPortH`); at and past it none. -/
theorem read_bits {T : Tile} {roots : String → _root_.Flock.Registered.Port} {place : String → String × ℕ}
    {sL sC : RowsSession} {kL kC : Option ℕ}
    {nmL nmC v : String} {row : ℕ} (hmL : (nmL, v, row) ∈ regReads T kL) (hmC : (nmC, v, row) ∈ regReads T kC)
    (hFL : nmL ∈ readFull T kL) (hFC : nmC ∈ readFull T kC)
    (hOL : ∀ p ∈ ownOf roots place T kL, sL.own.toList.find? (·.name == p.name) = some p)
    (hOC : ∀ p ∈ ownOf roots place T kC, sC.own.toList.find? (·.name == p.name) = some p)
    (hcL : _root_.Flock.Registered.checkComplete sL.c sL.own = .ok ())
    (hcC : _root_.Flock.Registered.checkComplete sC.c sC.own = .ok ()) {knL knC : Known}
    (hkL : _root_.Flock.KnownRead.check sL.c (knL ++ _root_.Flock.KnownRead.fullRows sL.own (readFull T kL)) = .ok ())
    (hkC : _root_.Flock.KnownRead.check sC.c (knC ++ _root_.Flock.KnownRead.fullRows sC.own (readFull T kC)) = .ok ())
    {uL : Fin sL.n} (huL : uL.val = 0) (hSL : sL.ServedRegistered uL)
    {uC : Fin sC.n} (huC : uC.val = 0) (hSC : sC.ServedRegistered uC)
    (hRL : ReadsOwnRow sL) (hRC : ReadsOwnRow sC)
    (hA : ∀ (p : sL.vb.Pos) (p' : sC.vb.Pos), (sL.rdU uL).isRead p → (sC.rdU uC).isRead p' →
      SameTree ((sL.rdU uL).port p) ((sC.rdU uC).port p') → (sL.rdU uL).readAt p = (sC.rdU uC).readAt p' →
      (rsOf sL).enc p ((sL.rdU uL).wv p) = (rsOf sC).enc p' ((sC.rdU uC).wv p'))
    (b : ℕ) : portBit sL (openedReg sL) nmL b = portBit sC (openedReg sC) nmC b := by
  have hfL : sL.own.toList.find? (·.name == nmL) = some (regPort roots place nmL v row) :=
    hOL _ (mem_ownOf roots place hmL)
  have hfC : sC.own.toList.find? (·.name == nmC) = some (regPort roots place nmC v row) :=
    hOC _ (mem_ownOf roots place hmC)
  have hrL : (nmL, #[(Array.replicate (_root_.Flock.KnownRead.regWords (roots (place v).1)) 0,
      ByteArray.empty)]) ∈ knL ++ _root_.Flock.KnownRead.fullRows sL.own (readFull T kL) := by
    have h := mem_fullRows hFL hfL
    rw [regWords_regPort] at h
    exact Array.mem_append.2 (Or.inr h)
  have hrC : (nmC, #[(Array.replicate (_root_.Flock.KnownRead.regWords (roots (place v).1)) 0,
      ByteArray.empty)]) ∈ knC ++ _root_.Flock.KnownRead.fullRows sC.own (readFull T kC) := by
    have h := mem_fullRows hFC hfC
    rw [regWords_regPort] at h
    exact Array.mem_append.2 (Or.inr h)
  obtain ⟨hNL, hexL⟩ := names_of_check hkL hrL
  obtain ⟨hNC, hexC⟩ := names_of_check hkC hrC
  obtain ⟨qL, hqL⟩ := hexL _ _ hrL
  obtain ⟨qC, hqC⟩ := hexC _ _ hrC
  obtain ⟨hPL, hwL⟩ := knownPort_of_check (s := sL) hkL hrL hqL
  obtain ⟨hPC, hwC⟩ := knownPort_of_check (s := sC) hkC hrC hqC
  have hWL : portRowW sL.c qL = 16 * _root_.Flock.KnownRead.regWords (roots (place v).1) := by
    rw [← hwL 0 (by simp)]
    show 16 * (Array.replicate (_root_.Flock.KnownRead.regWords (roots (place v).1)) 0).size = _
    simp
  have hWC : portRowW sC.c qC = 16 * _root_.Flock.KnownRead.regWords (roots (place v).1) := by
    rw [← hwC 0 (by simp)]
    show 16 * (Array.replicate (_root_.Flock.KnownRead.regWords (roots (place v).1)) 0).size = _
    simp
  by_cases hb : b < 16 * _root_.Flock.KnownRead.regWords (roots (place v).1)
  · have hnL : ∃ pt ∈ sL.own, pt.name = sL.c.ports[qL].name :=
      _root_.Flock.Registered.checkComplete_ok hcL _ (Array.getElem_mem qL.isLt)
    have hnC : ∃ pt ∈ sC.own, pt.name = sC.c.ports[qC].name :=
      _root_.Flock.Registered.checkComplete_ok hcC _ (Array.getElem_mem qC.isLt)
    have hoL : sL.ownPort qL = regPort roots place nmL v row := by
      show (sL.own.toList.find? (·.name == sL.c.ports[qL].name)).getD default = _
      rw [hqL, hfL]
      rfl
    have hoC : sC.ownPort qC = regPort roots place nmC v row := by
      show (sC.own.toList.find? (·.name == sC.c.ports[qC].name)).getD default = _
      rw [hqC, hfC]
      rfl
    have hpL := (hSL.2 qL hnL).1
    have hpC := (hSC.2 qC hnC).1
    have hiL : (sL.rdU uL).isRead (.inl (uL, qL)) := (hSL.1 _).2 ⟨qL, rfl, hnL⟩
    have hiC : (sC.rdU uC).isRead (.inl (uC, qC)) := (hSC.1 _).2 ⟨qC, rfl, hnC⟩
    have haL : (sL.rdU uL).readAt (.inl (uL, qL)) = (place v).2 + row :=
      hRL uL qL hnL _ ((congrArg _root_.Flock.Registered.Port.reads hoL).trans
        (_root_.Flock.Registered.Port.reads_of_readsFrom rfl))
    have haC : (sC.rdU uC).readAt (.inl (uC, qC)) = (place v).2 + row :=
      hRC uC qC hnC _ ((congrArg _root_.Flock.Registered.Port.reads hoC).trans
        (_root_.Flock.Registered.Port.reads_of_readsFrom rfl))
    have htree : SameTree ((sL.rdU uL).port (.inl (uL, qL))) ((sC.rdU uC).port (.inl (uC, qC))) := by
      rw [hpL, hoL, hpC, hoC]
      exact ⟨rfl, rfl, rfl, rfl⟩
    have he : HiddenExec.encPortH sL.c.ports[qL.val]! (rowVal sL uL qL) =
        HiddenExec.encPortH sC.c.ports[qC.val]! (rowVal sC uC qC) :=
      hA _ _ hiL hiC htree (haL.trans haC.symm)
    have hbL : b < portRowW sL.c qL := by rw [hWL]; exact hb
    have hbC : b < portRowW sC.c qC := by rw [hWC]; exact hb
    have eL := portBit_opened hNL huL hSL hcL (rowBitRead_of_check uL hkL hrL hqL hbL)
    have eC := portBit_opened hNC huC hSC hcC (rowBitRead_of_check uC hkC hrC hqC hbC)
    rw [hqL] at eL
    rw [hqC] at eC
    rw [eL, eC]
    congr 1
    unfold Layout.BitVal.get
    rw [dif_pos (rowW_lt qL hbL), dif_pos (rowW_lt qC hbC)]
    exact FlockVBridge.Family.bits_of_encPortH hPL.1 hPC.1 he b _ _
  · have hge : 16 * _root_.Flock.KnownRead.regWords (roots (place v).1) ≤ b := Nat.le_of_not_lt hb
    have nL : portBit sL (openedReg sL) nmL b = none := portBit_none _ fun q hq => by
      rw [hNL q qL (hq.trans hqL.symm), hWL]
      exact hge
    have nC : portBit sC (openedReg sC) nmC b = none := portBit_none _ fun q hq => by
      rw [hNC q qC (hq.trans hqC.symm), hWC]
      exact hge
    rw [nL, nC]

/-- **Lane `l`'s `partials` and the combine's `p<l>` read one row's bits**: `read_bits` at their reads of row `l` of
`partials`. -/
theorem partials_bits {T : Tile} {roots : String → _root_.Flock.Registered.Port} {place : String → String × ℕ}
    {sL sC : RowsSession} {l : ℕ} (hl : l < T.lanes)
    (hOL : ∀ p ∈ ownOf roots place T (some l), sL.own.toList.find? (·.name == p.name) = some p)
    (hOC : ∀ p ∈ ownOf roots place T none, sC.own.toList.find? (·.name == p.name) = some p)
    (hcL : _root_.Flock.Registered.checkComplete sL.c sL.own = .ok ())
    (hcC : _root_.Flock.Registered.checkComplete sC.c sC.own = .ok ()) {kL kC : Known}
    (hkL : _root_.Flock.KnownRead.check sL.c
      (kL ++ _root_.Flock.KnownRead.fullRows sL.own (readFull T (some l))) = .ok ())
    (hkC : _root_.Flock.KnownRead.check sC.c
      (kC ++ _root_.Flock.KnownRead.fullRows sC.own (readFull T none)) = .ok ())
    {uL : Fin sL.n} (huL : uL.val = 0) (hSL : sL.ServedRegistered uL)
    {uC : Fin sC.n} (huC : uC.val = 0) (hSC : sC.ServedRegistered uC)
    (hRL : ReadsOwnRow sL) (hRC : ReadsOwnRow sC)
    (hA : ∀ (p : sL.vb.Pos) (p' : sC.vb.Pos), (sL.rdU uL).isRead p → (sC.rdU uC).isRead p' →
      SameTree ((sL.rdU uL).port p) ((sC.rdU uC).port p') → (sL.rdU uL).readAt p = (sC.rdU uC).readAt p' →
      (rsOf sL).enc p ((sL.rdU uL).wv p) = (rsOf sC).enc p' ((sC.rdU uC).wv p'))
    (b : ℕ) : portBit sL (openedReg sL) "partials" b = portBit sC (openedReg sC) s!"p{l}" b :=
  read_bits (partials_lane T l) (partials_comb T hl) (full_of (partials_lane T l) rfl) (full_of (partials_comb T hl) rfl)
    hOL hOC hcL hcC hkL hkC huL hSL huC hSC hRL hRC hA b

/-- **Lane `l`'s decoded partials row is the combine's row `l`**, where the two ports' bits are one function: both
decodes read the words from bit 0 (`fieldBit_partials`, `fieldBit_combine`). -/
theorem part_eq {K nw : ℕ} {T : Tile} {sL sC : RowsSession} {vL vC : ℕ → ℕ → Option Bool} {l : Fin 32}
    (h : ∀ b, portBit sL vL "partials" b = portBit sC vC s!"p{l.val}" b) :
    (decL K nw T sL vL).part = (decC nw T sC vC).parts l := by
  have e : portBit sL vL "partials" = portBit sC vC s!"p{l.val}" := funext h
  funext w
  show BitVec.ofNat 32 (bitsN (portBit sL vL "partials") (fieldBit (lanePorts T 0) "partials" + 32 * w.val) 32) =
    BitVec.ofNat 32
      (bitsN (portBit sC vC s!"p{l.val}") (fieldBit (combinePorts T) s!"partials[{l.val}]" + 32 * w.val) 32)
  rw [fieldBit_partials, fieldBit_combine, e]

end Accounting.TS.Lanes
