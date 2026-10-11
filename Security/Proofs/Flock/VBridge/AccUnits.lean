import Proofs.Flock.VBridge.AccChain
import Proofs.Flock.VBridge.Decode

/-!
# VBridge G item 4: the running sums' unit facts, from the units' decoded columns

`AccChain.lean` takes `UnitSums` and `ReadsEnds`, element equations about V*'s units. This file discharges them from
facts about one copy of one unit in a layer agreeing with the registrant, the decode worker's form (`Decode.lean`).

* The bytes. A row's bytes (`MerkleScheme.rowBytes`) put bit `j` of element `k` at bit `128 k + j`
  (`bitL_rowBytes`), and an element is its 128 bits (`eq_of_bitOf`), so two rows whose bytes agree on element `k`'s
  bits agree at `k` (`getElem!_of_bitL`).
* The outputs. `RegBit` reads a copy's input columns; a unit's outputs are gates it computes, and a returned output's
  commit string is its output row (`posK`'s `.inl`, `HiddenExec.rowOfH`). `RegOut u hj col b`: copy `j`'s column `col` is a committed gate at a registered read whose
  registrant's bit is `b`. In a layer agreeing with the registrant where the unit is correct, the copy's column is `b`
  (`copyVal_regOut`).
* `unitSum_of_cols`: output columns holding `decode_recOpen`'s bytes, whose `acc_out` half is `rec-acc`'s row `p + 1`,
  give the element equation `UnitSums` asks. `unitSum_of_decode` is the same at a copy in a layer, from
  `decode_recOpen`'s `RegBit` hypotheses and `RegOut` at the `acc_out` columns; `OpensAt` bundles those facts for one
  copy, and `unitSums_of_decode` assembles them over the rep's openings into `UnitSums`, each opening's unit in the
  session of its level. `accChains_of_decode` is `accChains_of_units` with `UnitSums` so discharged.
* `readsEnds_of_regBits`: a registered bit has one value (`regBit_unique`), so when the algebra unit's columns of
  `acc0`, `acc1` carry the rep's message elements and are registered reads of `rec-acc`'s rows at the rep's ends, the
  elements are those rows' (`ReadsEnds`).
-/

namespace FlockVBridge.AccChain

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge.Decode (RegBit)
open Flock.RecAlgebra (valOf)
open FlockVBridge.Compiled (RepData)

/-! ## Bytes and elements -/

/-- **An element is its 128 bits.** -/
theorem eq_of_bitOf {x y : F128} (h : ∀ t < 128, bitOf x t = bitOf y t) : x = y := by
  have hw : FlockLevel3.w128 x.lo x.hi = FlockLevel3.w128 y.lo y.hi := by
    apply Nat.eq_of_testBit_eq
    intro t
    by_cases ht : t < 128
    · exact h t ht
    · exact (bitOf_ge x (by omega)).trans (bitOf_ge y (by omega)).symm
  obtain ⟨xl, xh⟩ := x
  obtain ⟨yl, yh⟩ := y
  simp only [FlockLevel3.w128] at hw
  have h1 := xl.toNat_lt
  have h2 := yl.toNat_lt
  have e : (2 : ℕ) ^ 64 = 18446744073709551616 := by norm_num
  rw [e] at hw h1 h2
  have hl : xl.toNat = yl.toNat := by omega
  have hh : xh.toNat = yh.toNat := by omega
  rw [UInt64.toNat_inj.mp hl, UInt64.toNat_inj.mp hh]

/-- **Rows agreeing on element `k`'s bytes agree at `k`.** -/
theorem getElem!_of_bitL {A B : Array F128} {k : ℕ}
    (h : ∀ j < 128, bitL (MerkleScheme.rowBytes A).data.toList (128 * k + j) =
      bitL (MerkleScheme.rowBytes B).data.toList (128 * k + j)) : A[k]! = B[k]! := by
  refine eq_of_bitOf fun j hj => ?_
  have := h j hj
  simpa only [bitL_rowBytes, wbit_elem _ _ hj, getD_toList] using this

theorem climb_size_of (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) :
    ∀ (ss : List ByteArray) (idx : ℕ) (acc : ByteArray), acc.size = 64 → (climb ms acc idx ss).size = 64
  | [], _, _, h => h
  | s :: rest, idx, acc, _ => by
    simp only [climb]
    exact climb_size_of ms hms rest (idx / 2) _
      (by split_ifs <;> rw [hms] <;> exact FlockSoundness.Refine.sha512_size _)

/-- `consTerm` reads `acc` only at the component it computes. -/
theorem consTerm_acc (LANES : ℕ) (r a a' c : Array F128) {k : ℕ} (h : a[k]! = a'[k]!) :
    consTerm LANES r a c k = consTerm LANES r a' c k := by
  unfold consTerm
  rw [h]

/-! ## An opening's running sum, from its unit's columns -/

/-- **The opening's running sum from its unit's output columns.** If the columns `v` from `base` hold the opening's
outputs, 128 bytes `pre` (the climb's) then the bytes of the consistency terms `#[c0, c1]` and `rest`
(`decode_recOpen`'s), and the `acc_out` half's first 256 columns hold the bytes of `rec-acc`'s row `A`, then `A`'s first
two elements are `c0, c1`. -/
theorem acc_of_cols {v : ℕ → Bool} {base : ℕ} {pre rest : ByteArray} (hX : pre.size = 128) {A : Array F128}
    {c0 c1 : F128} (hcols : ColsHold v base (pre ++ (MerkleScheme.rowBytes #[c0, c1] ++ rest)))
    (hout : ∀ t < 256, v (base + 1024 + t) = bitL (MerkleScheme.rowBytes A).data.toList t) {k : ℕ} (hk : k < 2) :
    A[k]! = (#[c0, c1] : Array F128)[k]! := by
  refine getElem!_of_bitL fun j hj => ?_
  have hX' : pre.data.toList.length = pre.size := Array.length_toList
  have hc' : (MerkleScheme.rowBytes #[c0, c1]).data.toList.length = 32 := by
    rw [Array.length_toList, ← ByteArray.size, size_rowBytes]
    rfl
  have hsz : 1024 + (128 * k + j) < 8 * (pre ++ (MerkleScheme.rowBytes #[c0, c1] ++ rest)).size := by
    rw [ByteArray.size_append, ByteArray.size_append, hX, size_rowBytes]
    simp only [List.size_toArray, List.length_cons, List.length_nil]
    omega
  rw [← hout _ (by omega), show base + 1024 + (128 * k + j) = base + (1024 + (128 * k + j)) by omega, hcols _ hsz,
    ByteArray.toList_data_append, bitL_append_right _ (by omega), hX', hX, ByteArray.toList_data_append,
    bitL_append_left _ (by omega), show 1024 + (128 * k + j) - 8 * 128 = 128 * k + j by omega]

/-- `acc_of_cols` at the opening's consistency terms. -/
theorem unitSum_of_cols {v : ℕ → Bool} {base : ℕ} {pre rest : ByteArray} (hX : pre.size = 128) {LANES : ℕ}
    {r a cf A : Array F128}
    (hcols : ColsHold v base
      (pre ++ (MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)))
    (hout : ∀ t < 256, v (base + 1024 + t) = bitL (MerkleScheme.rowBytes A).data.toList t) :
    ∀ k < 2, A[k]! = consTerm LANES r a cf k := by
  intro k hk
  rw [acc_of_cols hX hcols hout hk]
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

/-! ## At a copy in a layer -/

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **Copy `j`'s output column `col` of unit `u` is a registered bit `b`**: the copy's gate there is committed, at a
registered read (its commit string, `posK`'s: a returned output's output row), and the registrant's bit there is `b`. -/
def RegOut (u : Fin nv) {j : ℕ} (hj : j < c.upv) (col : ℕ) (b : Bool) : Prop :=
  ∃ hcol : col < (Rows.ofNet hU).w,
    (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩) ∈
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).committed ∧
      zo.rd.isRead (zo.lay.pos
        ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩))) ∧
      zo.lay.bit (zo.rd.wv (zo.lay.pos
          ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩))))
        ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩)) = b

/-- **A registered output column of a correct unit is the registrant's bit**, in a layer agreeing with the registrant. -/
theorem copyVal_regOut {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv}
    (hC : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).Correct bits u) {j : ℕ} (hj : j < c.upv)
    {col : ℕ} (hnIn : (Rows.ofNet hU).nIn ≤ col) {b : Bool} (h : RegOut zo u hj col b) :
    copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj col = b := by
  obtain ⟨hcol, hm, hread, hb⟩ := h
  rw [copyVal_committed _ hC hj ⟨col, hcol⟩ hm]
  obtain ⟨i, hi⟩ := Decode.stackCol_comp (Rows.ofNet hU) c.upv hj ⟨col, hcol⟩ hnIn
  have hu : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).unit
      ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hcol⟩)) = some u := by
    rw [hi]
    exact (keyInst hU nv (Layout.outsOf c hU nv) u).mem i
  rw [hA u _ (Finset.mem_filter.2 ⟨hm, Or.inl hu⟩) hread, hb]

/-- **An opening's running sum, from its unit's copy in a layer** (`decode_recOpen` and `unitSum_of_cols`). In a layer
agreeing with the registrant in which unit `u` is correct and carries 1 on its constant, let copy `j` be V*'s
laid-out opening (`netOf` of `recOpenLaid`) with input columns registered bits of the row `r`, the siblings `ss`,
`acc`'s bytes `rowBytes a ++ rest`, the coefficients `cf` and the position's bits (`RegBit`), and the first 256 columns
of its `acc_out` output registered bits of the bytes of `rec-acc`'s row `A` (`RegOut`). Then `A`'s first two elements
are the opening's consistency terms. -/
theorem unitSum_of_decode {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) (u : Fin nv)
    (hC : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).Correct bits u) {j : ℕ} (hj : j < c.upv)
    (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true)
    (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) {name : String}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hu : c.unit.unitLog ≤ 32) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (r a cf : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (MerkleScheme.rowBytes cf).size) (hdirsN : H ≤ p.dirs)
    (hrow : ∀ t < 8 * (MerkleScheme.rowBytes r).size,
      RegBit zo u j (0 + t) (bitL (MerkleScheme.rowBytes r).data.toList t))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ∀ t < 8 * ss[i].size, RegBit zo u j (p.row + 512 * i + t) (bitL ss[i].data.toList t))
    (hacc : ∀ t < 8 * (MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (bitL (MerkleScheme.rowBytes a ++ rest).data.toList t))
    (hcoef : ∀ t < 8 * (MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (bitL (MerkleScheme.rowBytes cf).data.toList t))
    (hdir : ∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (idx.testBit h))
    (A : Array F128)
    (haout : ∀ t < 256, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits +
      1024 + t) (bitL (MerkleScheme.rowBytes A).data.toList t)) :
    ∀ k < 2, A[k]! = consTerm LANES r a cf k := by
  have hO := Decode.decode_recOpen zo hA u hj hone ms hms LANES H hH p outBits hl hu hout r a cf rest idx ss hr ha hss
    hrowN hauxN haccN hcoefN hdirsN hrow haux hacc hcoef hdir
  have hnIn : (Rows.ofNet hU).nIn ≤
      Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits := by
    rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
    exact Decode.inRows_le_outStart _ _ _
  have hX : (climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩).size = 128 := by
    rw [ByteArray.size_append, climb_size_of ms hms ss idx _ (FlockSoundness.Refine.sha512_size _)]
    rfl
  exact unitSum_of_cols hX hO fun t ht => copyVal_regOut zo hA hC hj (by omega) (haout t ht)

/-- **Copy `j` of unit `u` opens with running sums `a` before and `A` after**: `unitSum_of_decode`'s facts about one
copy, at the coefficients `cf`, the row `r` and `LANES` lanes. -/
def OpensAt (bits : Fin (Layout.keyN c hU nv) → Bool) (u : Fin nv) (j LANES : ℕ) (r a cf A : Array F128) : Prop :=
  ∃ (hj : j < c.upv), bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true ∧
  ∃ (ms : MerkleScheme), (∀ l r, ms.node l r = Sha512.hash (l ++ r)) ∧
  ∃ (H : ℕ) (p : Ports) (outBits : Array ℕ) (name : String) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray),
    H ≤ 32 ∧ Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit ∧ c.unit.unitLog ≤ 32 ∧
    (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0 ∧
    r.size = LANES ∧ a.size = 2 ∧ ss.length = H ∧
    p.row = 8 * (MerkleScheme.rowBytes r).size ∧ 512 * H ≤ p.aux ∧
    p.acc = 8 * (MerkleScheme.rowBytes a ++ rest).size ∧ p.coef = 8 * (MerkleScheme.rowBytes cf).size ∧ H ≤ p.dirs ∧
    (∀ t < 8 * (MerkleScheme.rowBytes r).size,
      RegBit zo u j (0 + t) (bitL (MerkleScheme.rowBytes r).data.toList t)) ∧
    (∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ∀ t < 8 * ss[i].size, RegBit zo u j (p.row + 512 * i + t) (bitL ss[i].data.toList t)) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (bitL (MerkleScheme.rowBytes a ++ rest).data.toList t)) ∧
    (∀ t < 8 * (MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (bitL (MerkleScheme.rowBytes cf).data.toList t)) ∧
    (∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (idx.testBit h)) ∧
    ∀ t < 256, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits +
      1024 + t) (bitL (MerkleScheme.rowBytes A).data.toList t)

theorem unitSum_of_opensAt {bits : Fin (Layout.keyN c hU nv) → Bool} (hG : zo.Good bits) {u : Fin nv} {j LANES : ℕ}
    {r a cf A : Array F128} (h : OpensAt zo bits u j LANES r a cf A) : ∀ k < 2, A[k]! = consTerm LANES r a cf k := by
  obtain ⟨hj, hone, ms, hms, H, p, outBits, name, rest, idx, ss, hH, hl, hu, hout, hr, ha, hss, hrowN, hauxN, haccN,
    hcoefN, hdirsN, hrow, haux, hacc, hcoef, hdir, haout⟩ := h
  exact unitSum_of_decode zo hG.2 u (keyProg_correct hU nv hG.1 u) hj hone ms hms LANES H hH p outBits hl hu hout r a cf
    rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN hrow haux hacc hcoef hdir A haout

/-! ## The algebra's reads -/

/-- **A registered bit has one value**: the column's key names one bit of one registered row. -/
theorem regBit_unique {u : Fin nv} {j col : ℕ} {b b' : Bool} (h : RegBit zo u j col b) (h' : RegBit zo u j col b') :
    b = b' := by
  obtain ⟨_, q, k, hk, hkey, -, hv⟩ := h
  obtain ⟨_, q', k', hk', hkey', -, hv'⟩ := h'
  rw [hkey] at hkey'
  injection hkey' with hq h8 h7
  obtain rfl : q = q' := Fin.ext hq
  obtain rfl : k = k' := by
    have := Nat.div_add_mod k 8
    have := Nat.div_add_mod k' 8
    omega
  exact hv.symm.trans hv'

/-- **V*'s algebra reads the chain at the rep's ends, from its unit's columns.** Let copy `j` of unit `u` be V*'s
algebra unit, its columns `col0 r t` and `col1 r t` holding bit `t` of the rep's message elements `acc0 r` and `acc1 r`
as registered bits (the decode of its `w` port), and let the registrant's bits there be those of `rec-acc`'s rows
`o.p0` and `o.start` past the last level (the bytes of element `r`, `rowBytes`). Then V*'s algebra reads the chain at
the rep's ends (`ReadsEnds`). -/
theorem readsEnds_of_regBits (u : Fin nv) (j : ℕ) {sh : Shape} {d : RepData} {acc : ℕ → Array F128} {o : Openings}
    (col0 col1 : ℕ → ℕ → ℕ)
    (hd0 : ∀ r < 2, ∀ t < 128, RegBit zo u j (col0 r t) (bitOf (valOf sh d.w d.comb d.cons (.acc0 r)) t))
    (hr0 : ∀ r < 2, ∀ t < 128,
      RegBit zo u j (col0 r t) (bitL (MerkleScheme.rowBytes (acc o.p0)).data.toList (128 * r + t)))
    (hd1 : ∀ r < 2, ∀ t < 128, RegBit zo u j (col1 r t) (bitOf (valOf sh d.w d.comb d.cons (.acc1 r)) t))
    (hr1 : ∀ r < 2, ∀ t < 128, RegBit zo u j (col1 r t)
      (bitL (MerkleScheme.rowBytes (acc (o.start sh.sched.levels.size))).data.toList (128 * r + t))) :
    ReadsEnds sh d acc o := by
  intro r hr
  refine ⟨eq_of_bitOf fun t ht => ?_, eq_of_bitOf fun t ht => ?_⟩
  · have := regBit_unique zo (hd0 r hr t ht) (hr0 r hr t ht)
    simpa only [bitL_rowBytes, wbit_elem _ _ ht, getD_toList] using this
  · have := regBit_unique zo (hd1 r hr t ht) (hr1 r hr t ht)
    simpa only [bitL_rowBytes, wbit_elem _ _ ht, getD_toList] using this

end Copy

/-! ## Over the rep's openings -/

section Sessions

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **`UnitSums` from each opening's unit.** Let the sessions `zs` (one per V* statement) have `Good` layers `L`, and
let opening `j` of level `ℓ` of the rep be copy `cp ℓ j` of unit `un ℓ j` in the session `sess ℓ` of its level, opening
(`OpensAt`) the rep's row `(o.rows ℓ)[j]!` at the verifier's `o.coef ℓ j`, with `rec-acc`'s row at its chain position
`o.start ℓ + j` (its first two elements) before and the next row after. Then the running sums chain (`UnitSums`). -/
theorem unitSums_of_decode (zs : (s : Fin S) → (Vs s).Outer Reg') (L : (s : Fin S) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (L s)) (sh : Shape) (acc : ℕ → Array F128) (o : Openings) (sess : ℕ → Fin S)
    (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ)
    (h : ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ,
      OpensAt (zs (sess ℓ)) (L (sess ℓ)) (un ℓ j) (cp ℓ j) (levelLanes sh ℓ) (o.rows ℓ)[j]!
        #[(acc (o.start ℓ + j))[0]!, (acc (o.start ℓ + j))[1]!] (o.coef ℓ j) (acc (o.start ℓ + j + 1))) :
    UnitSums sh acc o := by
  intro ℓ hℓ j hj k hk
  rw [unitSum_of_opensAt (zs (sess ℓ)) (hG (sess ℓ)) (h ℓ hℓ j hj) k hk]
  refine consTerm_acc _ _ _ _ _ ?_
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl

section outer

open FlockVBridge.Compiled (Accepted layers AccChains)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {Reg₂ : Type}

/-- **G item 4 from the units' decoded columns**: `accChains_of_units`, each opening's `UnitSums` fact discharged by
its unit's copy (`OpensAt`, in the session `sess ℓ` of its level) at every accepted outer proof. -/
theorem accChains_of_decode (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (s : Fin S) → (Vs s).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((s : Fin S) → (Vs s).Bits) → Fin 2 → RepData)
    (acc : ((s : Fin S) → (Vs s).Bits) → ℕ → Array F128) (ops : ((s : Fin S) → (Vs s).Bits) → Fin 2 → Openings)
    (sess : ℕ → Fin S) (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size,
      ∀ j < (ops (layers zs dg N k Rw R ω t R₂ τ) r).q ℓ,
        OpensAt (zs R ω t R₂ (sess ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sess ℓ)) (un ℓ j) (cp ℓ j) (levelLanes sh ℓ)
          ((ops (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
          #[(acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
            (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
          ((ops (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
          (acc (layers zs dg N k Rw R ω t R₂ τ) ((ops (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hc : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (acc (layers zs dg N k Rw R ω t R₂ τ))
        (ops (layers zs dg N k Rw R ω t R₂ τ) r)) :
    AccChains zs dg N k Rw sh rep :=
  accChains_of_units zs dg N k Rw sh rep acc ops
    (fun R ω t R₂ τ o h r => unitSums_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (fun s => (h.1 s).2) sh
      _ _ sess un cp (hu R ω t R₂ τ o h r)) hc hcons hrd

end outer

end Sessions

end FlockVBridge.AccChain
