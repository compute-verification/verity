import Proofs.Flock.C3.RegLevels.SrcOpened
import Proofs.Flock.C3.RegLevels.StackedReg

/-!
# n11: `OpensTopAt`'s registered ports, from the unit's registered columns and the verifier's check

`OpensAtMaps` asks each opening's unit for `OpensTopAt`: the row and the siblings in the copy's own columns
(`rowCols_readMaps`, `ioCols_of_open`), and `acc`, the coefficients, the `dirs` and the outputs registered. Here:

* `regBits_of_cols`, `regOuts_of_cols`: where each column of a run of copy `j`'s inputs (outputs) is a registered bit
  (`RegBit`, `RegOut`), the run holds the bytes the registrant holds there (`regCol`, `outReg`) as registered bits. So
  `opensTopAt_of_regs`: `OpensTopAt`'s `a`, `rest`, `cf` and `A` are those bytes, from columns that are registered.
* `dirs_of_check`: an accepted `rec-open-stacked/l<LANES>-h<H>` unit registered `dirsRow pos H` at each opening's
  position (`RecCoins.checkStackedLevel`, `dirsLevel_of_check`), the run's from round 0 of `stacked/rep0`'s coins
  (`lg_eq_of_opening`). Where the opening's `dirs` columns are the bits of the first word of the rows `checkValue`
  accepted (`hval`), they are the position's low `H` bits, at `H ≤ 16`: `dirsRow` holds them in one 16-bit word.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge.Decode (RegBit)

/-! ## Registered columns hold the registrant's bytes -/

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

open Classical in
/-- Copy `j`'s input column `col` of unit `u` at the registrant, where it is a registered bit (`RegBit`). -/
noncomputable def regCol (u : Fin nv) (j col : ℕ) : Bool :=
  if h : ∃ b, RegBit zo u j col b then Classical.choose h else false

theorem regBit_regCol {u : Fin nv} {j col : ℕ} {b : Bool} (h : RegBit zo u j col b) :
    RegBit zo u j col (regCol zo u j col) := by
  have hex : ∃ b, RegBit zo u j col b := ⟨b, h⟩
  unfold regCol
  rw [dif_pos hex]
  exact Classical.choose_spec hex

/-- **A run of registered input columns holds the registrant's bytes**, as registered bits. -/
theorem regBits_of_cols {u : Fin nv} {j base : ℕ} {B : ByteArray}
    (hr : ∀ t < 8 * B.size, ∃ b, RegBit zo u j (base + t) b) (hB : ColsHold (regCol zo u j) base B) :
    ∀ t < 8 * B.size, RegBit zo u j (base + t) (bitL B.data.toList t) := by
  intro t ht
  obtain ⟨b, hb⟩ := hr t ht
  rw [← hB t ht]
  exact regBit_regCol zo hb

open Classical in
/-- Copy `j`'s output column `col` of unit `u` as registered (`RegOut`). -/
noncomputable def outReg (u : Fin nv) {j : ℕ} (hj : j < c.upv) (col : ℕ) : Bool :=
  if h : ∃ b, RegOut zo u hj col b then Classical.choose h else false

theorem regOut_outReg {u : Fin nv} {j : ℕ} {hj : j < c.upv} {col : ℕ} {b : Bool} (h : RegOut zo u hj col b) :
    RegOut zo u hj col (outReg zo u hj col) := by
  have hex : ∃ b, RegOut zo u hj col b := ⟨b, h⟩
  unfold outReg
  rw [dif_pos hex]
  exact Classical.choose_spec hex

/-- **A run of registered output columns holds the registrant's bytes**, as registered outputs. -/
theorem regOuts_of_cols {u : Fin nv} {j : ℕ} {hj : j < c.upv} {base : ℕ} {B : ByteArray}
    (hr : ∀ t < 8 * B.size, ∃ b, RegOut zo u hj (base + t) b) (hB : ColsHold (outReg zo u hj) base B) :
    ∀ t < 8 * B.size, RegOut zo u hj (base + t) (bitL B.data.toList t) := by
  intro t ht
  obtain ⟨b, hb⟩ := hr t ht
  rw [← hB t ht]
  exact regOut_outReg zo hb

/-- **`OpensTopAt` from the unit's registered columns**: where each `acc` and coefficient column of copy `j` is a
registered bit and each of the first 256 columns of its `acc_out` half a registered output, `a` and `rest`, `cf` and
`A` are the bytes the registrant holds there (`regBits_of_cols`, `regOuts_of_cols`); the rest is `OpensTopAt`'s. -/
theorem opensTopAt_of_regs {bits : Fin (Layout.keyN c hU nv) → Bool} {u : Fin nv} {j LANES H pos : ℕ}
    {r : Array F128} {ss : List ByteArray} {top : ByteArray} (hj : j < c.upv)
    (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true) (ms : MerkleScheme)
    (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (p : Ports) (outBits : Array ℕ) (name : String) (hH : H ≤ 32)
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit) (hu : c.unit.unitLog ≤ 32)
    (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0) (hr : r.size = LANES) (hss : ss.length = H)
    (hrowN : p.row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux) {nA nC : ℕ}
    (haccN : p.acc = 256 + 8 * nA) (hcoefN : p.coef = 128 * nC) (hdirsN : H ≤ p.dirs)
    (hio : ∀ col < p.row + 512 * H, col < (Rows.ofNet hU).nIn ∧ ∀ (hc : col < (Rows.ofNet hU).w),
      (keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stackCol c.upv hj ⟨col, hc⟩) ∈
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u)
    (hrow : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) 0 (MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) (p.row + 512 * i) ss[i])
    (hacc : ∀ t < p.acc, ∃ b, RegBit zo u j (p.row + p.aux + t) b)
    (hcoef : ∀ t < p.coef, ∃ b, RegBit zo u j (p.row + p.aux + p.acc + t) b)
    (hdir : ∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (pos.testBit h))
    (haout : ∀ t < 256, ∃ b, RegOut zo u hj
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024 + t) b)
    (htop : top.size = 64) (hT : ∀ t < 512, RegOut zo u hj
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t) (bitL top.data.toList t)) :
    ∃ a cf A, OpensTopAt zo bits u j LANES r a cf A H pos ss top := by
  have hsa : (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2)).size = 32 := by
    rw [size_rowBytes, Family.size_elems]
  have hsA : (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2) ++
      Family.bytesAt (regCol zo u j) (p.row + p.aux + 256) nA).size = 32 + nA := by
    rw [ByteArray.size_append, hsa, Family.size_bytesAt]
  have hsC : (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux + p.acc) nC)).size = 16 * nC := by
    rw [size_rowBytes, Family.size_elems]
  have hsO : (MerkleScheme.rowBytes (Family.elems (outReg zo u hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024) 2)).size = 32 := by
    rw [size_rowBytes, Family.size_elems]
  have e256 : p.row + p.aux + 8 * (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2)).size =
      p.row + p.aux + 256 := by
    omega
  have hCa : ColsHold (regCol zo u j) (p.row + p.aux)
      (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2) ++
        Family.bytesAt (regCol zo u j) (p.row + p.aux + 256) nA) := by
    refine Family.colsHold_append (Family.colsHold_elems _ _ _) ?_
    rw [e256]
    exact Family.colsHold_bytesAt _ _ _
  have hacc' : ∀ t < 8 * (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2) ++
      Family.bytesAt (regCol zo u j) (p.row + p.aux + 256) nA).size,
      RegBit zo u j (p.row + p.aux + t)
        (bitL (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux) 2) ++
          Family.bytesAt (regCol zo u j) (p.row + p.aux + 256) nA).data.toList t) :=
    regBits_of_cols zo (fun t ht => hacc t (by omega)) hCa
  have hcoef' : ∀ t < 8 * (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux + p.acc) nC)).size,
      RegBit zo u j (p.row + p.aux + p.acc + t)
        (bitL (MerkleScheme.rowBytes (Family.elems (regCol zo u j) (p.row + p.aux + p.acc) nC)).data.toList t) :=
    regBits_of_cols zo (fun t ht => hcoef t (by omega)) (Family.colsHold_elems _ _ _)
  have haout' : ∀ t < 8 * (MerkleScheme.rowBytes (Family.elems (outReg zo u hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024) 2)).size,
      RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024 + t)
        (bitL (MerkleScheme.rowBytes (Family.elems (outReg zo u hj)
          (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024) 2)).data.toList
          t) :=
    regOuts_of_cols zo (fun t ht => haout t (by omega)) (Family.colsHold_elems _ _ _)
  exact ⟨Family.elems (regCol zo u j) (p.row + p.aux) 2, Family.elems (regCol zo u j) (p.row + p.aux + p.acc) nC,
    Family.elems (outReg zo u hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024) 2,
    hj, hone, ms, hms, p, outBits, name, Family.bytesAt (regCol zo u j) (p.row + p.aux + 256) nA, hH, hl, hu, hout,
    hr, Family.size_elems _ _ _, hss, hrowN, hauxN, by omega, by omega, hdirsN, hio, hrow, haux, hacc', hcoef', hdir,
    fun t ht => haout' t (by omega), htop, hT⟩

end Copy

/-! ## The `dirs` the verifier registers -/

theorem dirsRow_zero (pos H : ℕ) : (Flock.RecCoins.dirsRow pos H)[0]! = pos % 2 ^ H := by
  rw [Flock.RecCoins.dirsRow,
    getElem!_pos (#[pos % 2 ^ H] ++ Array.replicate (Flock.RecCoins.DIRS_WORDS - 1) 0) 0 (by simp <;> omega),
    Array.getElem_append_left (by simp)] <;> rfl

/-- Bit `b < H` of `dirsRow pos H`'s first word is the position's. -/
theorem dirsRow_testBit (pos H : ℕ) {b : ℕ} (hb : b < H) :
    ((Flock.RecCoins.dirsRow pos H)[0]!).testBit b = pos.testBit b := by
  rw [dirsRow_zero, Nat.testBit_mod_two_pow]
  simp [hb]

theorem getElem!_map_range {α : Type} [Inhabited α] (f : ℕ → α) {n j : ℕ} (hj : j < n) :
    ((Array.range n).map f)[j]! = f j := by
  rw [getElem!_pos ((Array.range n).map f) j (by simpa using hj)]
  simp

/-- **What an accepted `rec-open-stacked` unit checked of its `dirs`**: as `level_of_check`, with the `dirs` port
holding `dirsRow` of each opening's position at the filter's level `l`. -/
theorem dirsLevel_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) :
    ∃ (o : Flock.RecCoins.Opening) (l : ℕ) (what : ℕ → String), Flock.RecCoins.opening c3 c.unit.name = .ok o ∧
      (∀ ℓ < o.sh.sched.levels.size, 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES →
        (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H → ℓ = l) ∧
      Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE
        ((Array.range (o.sh.sched.levels[l]!).q).map fun j => Flock.RecCoins.dirsRow (o.lg.conss[l]!).pos[j]! H)
        what = .ok () := by
  unfold Flock.RecCoins.checkStackedLevel at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i l hls
    split at h
    · cases h
    · obtain ⟨_u, -, hdv⟩ := FlockVBridge.PartS.bind_ok h
      refine ⟨o, l, _, hop, fun ℓ hℓ hL hH => ?_, hdv⟩
      have hm : ℓ ∈ [l] := by
        rw [← hls]
        exact List.mem_filter.mpr ⟨List.mem_range.mpr hℓ, by simp only [Bool.and_eq_true, beq_iff_eq]; exact ⟨hL, hH⟩⟩
      exact List.mem_singleton.mp hm
  · cases h

/-- **An accepted `rec-open-stacked` unit's `dirs` are its openings' positions' bits** (`hdir`): at the level `ℓ` the
unit names, where opening `j`'s `dirs` columns hold bit `b` of the first word of row `j` of the rows `checkValue`
accepted (`P j b`, `hval`), they hold bit `b < H` of the verifier's position (`dirsRow_testBit`; it is the run's at
`lg_eq_of_opening`). `H ≤ 16`: `dirsRow` holds the position's low `H` bits in its first 16-bit word. -/
theorem dirs_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) {o : Flock.RecCoins.Opening}
    (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o) {ℓ : ℕ} (hℓ : ℓ < o.sh.sched.levels.size)
    (hL : 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES)
    (hH : (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H) (h16 : H ≤ 16) (P : ℕ → ℕ → Bool → Prop)
    (hval : ∀ (rows : Array (Array ℕ)) (what : ℕ → String),
      Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE rows what = .ok () →
      ∀ j < rows.size, ∀ b < 16, P j b ((rows[j]!)[0]!.testBit b)) :
    ∀ j < (o.sh.sched.levels[ℓ]!).q, ∀ b < H, P j b (((o.lg.conss[ℓ]!).pos[j]!).testBit b) := by
  obtain ⟨o', l, what, hop', hu, hdv⟩ := dirsLevel_of_check h
  have ho : o' = o := Except.ok.inj (hop'.symm.trans hop)
  subst ho
  have hl : ℓ = l := hu ℓ hℓ hL hH
  subst hl
  intro j hj b hb
  have e := hval _ what hdv j (by simpa using hj) b (by omega)
  rwa [getElem!_map_range _ hj, dirsRow_testBit _ _ hb] at e

#print axioms regBits_of_cols
#print axioms regOuts_of_cols
#print axioms opensTopAt_of_regs
#print axioms dirsLevel_of_check
#print axioms dirs_of_check

end FlockC3.RegLevels
