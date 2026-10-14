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
  (`lg_eq_of_opening`). Where the opening's `dirs` columns are the bits of the first two words of the rows
  `checkValue` accepted (`hval`), they are the position's low `H` bits, at every valid `H` (`H ≤ 32`): `dirsRow` holds
  them 16 a word.
* `srcKey_opening`, `regBit_opening`: at a unit `RecOpen.check` accepted, the statement's first five rows are the
  opening's (`rowsOk`) and every copy reads their words in order (`leavesOk`), so copy `j`'s input bit `preBits q + k`
  has the Δ key of bit `k` of row `q` (`srcKey_inOrder`), a registered bit wherever unit `u`'s row `q` is a registered
  read. `accCoef_of_check` and `dirsCols_of_check` are `opensTopAt_of_regs`' `acc`, `coef` and `dirs` columns so.
* `regReads_of_check`, `regCols_of_check`: at an accepted R level's unit, rows 2–4 (`acc`, `coef`, `dirs`) are registered
  reads, so those columns registered bits, where the session registers each row of a port its reads file names (`hread`,
  VBridge's `RowsRegistered` at the unit, which acceptance does not give): the reads file names `acc` (`checkAcc`,
  `acc_of_checkStmt`) and `coef` and `dirs` (`checkStackedLevel`'s `checkValue`s, `own_of_checkValue`).
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
  rw [dite_eq_left hex]
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
  rw [dite_eq_left hex]
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

theorem dirsRow_zero (pos H : ℕ) : (Flock.RecCoins.dirsRow pos H)[0]! = pos % 2 ^ H % 2 ^ 16 := by
  rw [Flock.RecCoins.dirsRow,
    getElem!_pos (#[pos % 2 ^ H % 2 ^ 16, pos % 2 ^ H / 2 ^ 16] ++
      Array.replicate (Flock.RecCoins.DIRS_WORDS - 2) 0) 0 (by simp [Flock.RecCoins.DIRS_WORDS]),
    Array.getElem_append_left (by simp)]
  rfl

theorem dirsRow_one (pos H : ℕ) : (Flock.RecCoins.dirsRow pos H)[1]! = pos % 2 ^ H / 2 ^ 16 := by
  rw [Flock.RecCoins.dirsRow,
    getElem!_pos (#[pos % 2 ^ H % 2 ^ 16, pos % 2 ^ H / 2 ^ 16] ++
      Array.replicate (Flock.RecCoins.DIRS_WORDS - 2) 0) 1 (by simp [Flock.RecCoins.DIRS_WORDS]),
    Array.getElem_append_left (by simp)]
  rfl

/-- Bit `b < H` of the position, `b < 32`, is bit `b % 16` of `dirsRow pos H`'s word `b / 16`. -/
theorem dirsRow_testBit (pos H : ℕ) {b : ℕ} (hb : b < H) (h32 : b < 32) :
    ((Flock.RecCoins.dirsRow pos H)[b / 16]!).testBit (b % 16) = pos.testBit b := by
  rcases Nat.lt_or_ge b 16 with h | h
  · rw [Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h, dirsRow_zero, Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow]
    simp [h, hb]
  · rw [show b / 16 = 1 by omega, dirsRow_one, Nat.testBit_div_two_pow, show b % 16 + 16 = b by omega,
      Nat.testBit_mod_two_pow]
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
    · obtain ⟨_u, -, h⟩ := FlockVBridge.PartS.bind_ok h
      obtain ⟨_v, hdv, -⟩ := FlockVBridge.PartS.bind_ok h
      refine ⟨o, l, _, hop, fun ℓ hℓ hL hH => ?_, hdv⟩
      have hm : ℓ ∈ [l] := by
        rw [← hls]
        exact List.mem_filter.mpr ⟨List.mem_range.mpr hℓ, by simp only [Bool.and_eq_true, beq_iff_eq]; exact ⟨hL, hH⟩⟩
      exact List.mem_singleton.mp hm
  · cases h

/-- **An accepted `rec-open-stacked` unit's `dirs` are its openings' positions' bits** (`hdir`): at the level `ℓ` the
unit names, where opening `j`'s `dirs` column `b < 32` holds bit `b % 16` of word `b / 16` of row `j` of the rows
`checkValue` accepted (`P j b`, `hval`), it holds bit `b < H` of the verifier's position (`dirsRow_testBit`; it is the
run's at `lg_eq_of_opening`), at every valid `H` (`H ≤ 32`, `RecOpen.valid`): `dirsRow` holds the position's low `H` bits
16 a word in its first two words. -/
theorem dirs_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) {o : Flock.RecCoins.Opening}
    (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o) {ℓ : ℕ} (hℓ : ℓ < o.sh.sched.levels.size)
    (hL : 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES)
    (hH : (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H) (h32 : H ≤ 32) (P : ℕ → ℕ → Bool → Prop)
    (hval : ∀ (rows : Array (Array ℕ)) (what : ℕ → String),
      Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE rows what = .ok () →
      ∀ j < rows.size, ∀ b < 32, P j b ((rows[j]!)[b / 16]!.testBit (b % 16))) :
    ∀ j < (o.sh.sched.levels[ℓ]!).q, ∀ b < H, P j b (((o.lg.conss[ℓ]!).pos[j]!).testBit b) := by
  obtain ⟨o', l, what, hop', hu, hdv⟩ := dirsLevel_of_check h
  have ho : o' = o := Except.ok.inj (hop'.symm.trans hop)
  subst ho
  have hl : ℓ = l := hu ℓ hℓ hL hH
  subst hl
  intro j hj b hb
  have e := hval _ what hdv j (by simpa using hj) b (by omega)
  rwa [getElem!_map_range _ hj, dirsRow_testBit _ _ hb (by omega)] at e

/-! ## The opening's keys, from the verifier's check -/

/-- **In a unit of one group of 16-bit ports from column 0, a copy that reads the leaf words in order reads each port's
row**: the Δ key of copy `j`'s input bit `16 · starts[q] + k` is message bit `k % 8` of byte `k / 8` of port `q`, for `k`
within the row. -/
theorem srcKey_inOrder {c : Flock.Circuit} {g : Flock.Group} (hg : c.unit.inGroups = #[g]) (hg0 : g.col = 0)
    (h16 : ∀ b ∈ g.bits.toList, b = 16) (hnd : (FlockSoundness.portBits c.unit.inGroups).Nodup) {j : ℕ}
    (hlv : c.leavesIn.getD j #[] = (Array.range g.bits.size).map Int.ofNat) {q k : ℕ} (hq : q < c.ports.size)
    (hk : k < 16 * c.ports[q].words) (hkb : k < c.ports[q].bits)
    (hfit : (FlockSoundness.Typed.starts c)[q]! + c.ports[q].words ≤ g.bits.size)
    (hN : 16 * g.bits.size ≤ c.unit.inWords * Flock.WORD) :
    Aliased.srcKey c (j * (c.unit.inWords * Flock.WORD) + (16 * (FlockSoundness.Typed.starts c)[q]! + k)) =
      .msg q (k / 8) (k % 8) := by
  have htake : ∀ a (ha : a < c.ports.size), ((c.ports.toList.take (a + 1)).map (·.words)).sum =
      ((c.ports.toList.take a).map (·.words)).sum + c.ports[a].words := by
    intro a ha
    rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem (by simpa using ha)]
    simp
  have hmono : ∀ a (ha : a < c.ports.size) b, a < b → b < c.ports.size →
      (FlockSoundness.Typed.starts c)[a]! + c.ports[a].words ≤ (FlockSoundness.Typed.starts c)[b]! := by
    intro a ha b hab hb
    rw [FlockSoundness.Discharge.Layout.starts_get c ha, FlockSoundness.Discharge.Layout.starts_get c hb]
    have e1 : c.ports.toList.take b =
        c.ports.toList.take (a + 1) ++ (c.ports.toList.drop (a + 1)).take (b - (a + 1)) := by
      rw [← List.take_add]
      congr 1
      omega
    rw [e1, List.map_append, List.sum_append, htake a ha]
    omega
  have hx : (FlockSoundness.Typed.starts c)[q]! + k / 16 < c.inWords := by
    have e : (c.ports.toList.map (·.words)).sum = ((c.ports.toList.take (q + 1)).map (·.words)).sum +
        ((c.ports.toList.drop (q + 1)).map (·.words)).sum := by
      rw [← List.sum_append, ← List.map_append, List.take_append_drop]
    rw [FlockSoundness.Discharge.Layout.inWords_eq, e, htake q hq, ← FlockSoundness.Discharge.Layout.starts_get c hq]
    omega
  generalize hs : (FlockSoundness.Typed.starts c)[q]! = s at hfit hx ⊢
  have hcol : 16 * s + k < c.unit.inWords * Flock.WORD := by omega
  have hmod : (j * (c.unit.inWords * Flock.WORD) + (16 * s + k)) % (c.unit.inWords * Flock.WORD) = 16 * s + k := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hcol]
  have hdiv : (j * (c.unit.inWords * Flock.WORD) + (16 * s + k)) / (c.unit.inWords * Flock.WORD) = j := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hcol, Nat.add_zero]
  have hsz : (Flock.portCols c.unit.inGroups).size = g.bits.size := by
    rw [hg]
    simpa using congrArg List.length (FlockSoundness.portCols_single g)
  have hpb : s + k / 16 < g.bits.size := by omega
  have hp : s + k / 16 < (Flock.portCols c.unit.inGroups).size := by rw [hsz]; exact hpb
  have hc1 : (Flock.portCols c.unit.inGroups)[s + k / 16]!.1 = 16 * (s + k / 16) := by
    rw [← FlockSoundness.Layout.getD_portCols hp, hg, FlockSoundness.Discharge.HiddenExec.portCols_16_get h16 hpb, hg0]
    simp
  have hc2 : (Flock.portCols c.unit.inGroups)[s + k / 16]!.2 = 16 := by
    have hpg : s + k / 16 < (Flock.portCols #[g]).size := by rw [← hg]; exact hp
    have := List.getElem_of_eq (FlockSoundness.portCols_single g) (by simpa using hpg)
    simp only [List.getElem_map, Array.getElem_toList] at this
    rw [hg, getElem!_pos (Flock.portCols #[g]) (s + k / 16) hpg, this]
    exact h16 _ (Array.getElem_mem_toList _)
  have hcol' : 16 * s + k = (Flock.portCols c.unit.inGroups)[s + k / 16]!.1 + k % 16 := by rw [hc1]; omega
  have hm : 16 * s + k ∈ FlockSoundness.portBits c.unit.inGroups := by
    unfold FlockSoundness.portBits
    refine List.mem_flatMap.2 ⟨(Flock.portCols c.unit.inGroups)[s + k / 16]!, ?_, ?_⟩
    · rw [getElem!_pos (Flock.portCols c.unit.inGroups) (s + k / 16) hp]
      exact Array.getElem_mem_toList _
    · rw [List.mem_range'_1, hc1, hc2]
      omega
  unfold Aliased.srcKey
  rw [hmod, ite_eq_left hm, hdiv, hcol', FlockSoundness.portOf_eq hnd hp (by rw [hc2]; omega), Nat.add_sub_cancel_left]
  unfold Aliased.srcAt
  have hp0 : s + k / 16 < (c.unit.inGroups.getD 0 default).bits.size := by rw [hg]; exact hpb
  have hleaf : (c.leavesIn.getD j #[])[s + k / 16]! = Int.ofNat (s + k / 16) := by
    rw [hlv, getElem!_pos _ _ (by simpa using hpb)]
    simp
  have hnat : (Int.ofNat (s + k / 16)).toNat = s + k / 16 := rfl
  rw [ite_eq_left hp0, hleaf, ite_eq_right (by rw [Int.ofNat_eq_natCast]; omega)]
  dsimp only
  rw [hnat]
  have hQ : ((List.range (FlockSoundness.Typed.starts c).size).reverse.find?
      (fun i => (FlockSoundness.Typed.starts c)[i]! ≤ s + k / 16)).getD 0 = q := by
    obtain ⟨Q, hQ⟩ : ∃ Q, ((List.range (FlockSoundness.Typed.starts c).size).reverse.find?
        (fun i => (FlockSoundness.Typed.starts c)[i]! ≤ s + k / 16)).getD 0 = Q := ⟨_, rfl⟩
    obtain ⟨hQs, hle, hlt⟩ := FlockSoundness.Discharge.Layout.starts_find c hx hQ
    rw [hQ]
    rcases Nat.lt_trichotomy Q q with h | h | h
    · have := hmono Q hQs q h hq
      omega
    · exact h
    · have := hmono q hq Q h hQs
      omega
  rw [hQ, hs, getElem!_pos c.ports q hq, ite_eq_left (by omega)]
  congr 1 <;> omega

/-- **What `checkAt` accepted of the statement's rows and its copies' leaves** (`rowsOk`, `leavesOk`). -/
theorem checkAt_rows {c : Flock.Circuit} {name : String} {L H : ℕ} (h : Flock.RecOpen.checkAt c name L H = .ok ()) :
    Flock.RecOpen.rowsOk c L H = true ∧ Flock.RecOpen.leavesOk c L H = true := by
  unfold Flock.RecOpen.checkAt at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i hr
        split at h
        · cases h
        · rename_i hv
          exact ⟨by simpa using hr, by simpa using hv⟩

/-- **What `check` accepted of a `rec-open/…` or `rec-open-stacked/…` unit's rows and leaves.** -/
theorem check_rows {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (L, H) ∨ Flock.RecOpen.stackedParams c.unit.name = some (L, H)) :
    Flock.RecOpen.rowsOk c L H = true ∧ Flock.RecOpen.leavesOk c L H = true := by
  unfold Flock.RecOpen.check at h
  rcases hp with hp | hp
  · rw [hp] at h
    exact checkAt_rows h
  · rw [Flock.RecOpen.params_stacked hp] at h
    dsimp only at h
    rw [← Flock.RecOpen.stackedName_of_params hp, Flock.RecOpen.startsWith_stackedName,
      Flock.RecOpen.stackedName_of_params hp, hp] at h
    exact checkAt_rows h

theorem rows_snd (L H : ℕ) {q : ℕ} (hq : q < 5) :
    ((Flock.RecOpen.rows L H)[q]!).2 = (Flock.RecOpen.ports L H).bits[q]! := by
  obtain rfl | rfl | rfl | rfl | rfl : q = 0 ∨ q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by omega
  all_goals rfl

/-- **`rowsOk`, row by row**: the statement's row `q < 5` is the opening's input row `q`, by name and width. -/
theorem rowsOk_get {c : Flock.Circuit} {L H : ℕ} (h : Flock.RecOpen.rowsOk c L H = true) {q : ℕ} (hq : q < 5) :
    ∃ hq' : q < c.ports.size, c.ports[q].name = ((Flock.RecOpen.rows L H)[q]!).1 ∧
      16 * c.ports[q].words = (Flock.RecOpen.ports L H).bits[q]! ∧
        c.ports[q].bits = (Flock.RecOpen.ports L H).bits[q]! := by
  unfold Flock.RecOpen.rowsOk at h
  have e := of_decide_eq_true h
  have h5r : (Flock.RecOpen.rows L H).size = 5 := rfl
  have hs := congrArg Array.size e
  simp only [Array.size_map, Array.size_extract, h5r] at hs
  have hq' : q < c.ports.size := by omega
  refine ⟨hq', ?_⟩
  have e2 := congrArg (fun a => a[q]!) e
  rw [getElem!_pos _ q (by simp; omega), getElem!_pos _ q (by simp [h5r]; omega)] at e2
  simp only [Array.getElem_map, Array.getElem_extract, Nat.zero_add] at e2
  rw [← rows_snd L H hq, getElem!_pos (Flock.RecOpen.rows L H) q (by rw [h5r]; exact hq)]
  simp only [Prod.mk.injEq] at e2
  exact ⟨e2.1, e2.2.1, e2.2.2⟩

/-- **`leavesOk` at copy `j`**: its leaves are the rows' words in order. -/
theorem leavesOk_get {c : Flock.Circuit} {L H : ℕ} (h : Flock.RecOpen.leavesOk c L H = true) {j : ℕ}
    (hj : j < c.leavesIn.size) :
    c.leavesIn.getD j #[] = (Array.range ((Flock.RecOpen.ports L H).bits.foldl (· + ·) 0 / 16)).map Int.ofNat := by
  unfold Flock.RecOpen.leavesOk at h
  rw [Array.all_eq_true] at h
  have := h j hj
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]
  simpa using this

/-- The opening's input bits before its row `q` (`row`, `aux`, `acc`, `coef`, `dirs`, in that order). -/
def preBits (LANES H q : ℕ) : ℕ := (((Flock.RecOpen.ports LANES H).bits.toList).take q).sum

theorem preBits_le (L H : ℕ) {q : ℕ} (hq : q < 5) :
    preBits L H q + (Flock.RecOpen.ports L H).bits[q]! ≤ (Flock.RecOpen.ports L H).bits.foldl (· + ·) 0 := by
  obtain rfl | rfl | rfl | rfl | rfl : q = 0 ∨ q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by omega
  all_goals (simp [preBits, Flock.RecOpen.ports, Flock.RecOpen.Ports.bits]; try omega)

theorem inGroups_of_laid {name : String} {L H : ℕ} {net : Flock.Net}
    (hl : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid L H) = .ok net) :
    net.inGroups = #[⟨0, Flock.HmNets.groupWords ((Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)).foldl (· + ·) 0),
      Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)⟩] := by
  unfold Flock.HmNets.netOf at hl
  rw [(Flock.Net.ofRows_ok hl).2.2.2.2.2.1]
  rfl

/-- **An accepted opening's copies read its rows** (`srcKey_inOrder`): at the verifier's net at `L` and `H`, with the
statement's first five rows the opening's (`rowsOk`) and copy `j` reading their words in order (`leavesOk`), the Δ key
of copy `j`'s input bit `preBits q + k`, for `k` within row `q < 5`, is bit `k` of the statement's row `q`. -/
theorem srcKey_opening {c : Flock.Circuit} {name : String} {L H : ℕ}
    (hl : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid L H) = .ok c.unit)
    (hrows : Flock.RecOpen.rowsOk c L H = true) (hlv : Flock.RecOpen.leavesOk c L H = true)
    (hnd : (FlockSoundness.portBits c.unit.inGroups).Nodup) {j : ℕ} (hj : j < c.leavesIn.size) {q : ℕ} (hq : q < 5)
    {k : ℕ} (hk : k < (Flock.RecOpen.ports L H).bits[q]!) :
    ∃ hq' : q < c.ports.size, c.ports[q].name = ((Flock.RecOpen.rows L H)[q]!).1 ∧ k < 16 * c.ports[q].words ∧
      preBits L H q + k < c.unit.inWords * Flock.WORD ∧
      Aliased.srcKey c (j * (c.unit.inWords * Flock.WORD) + (preBits L H q + k)) = .msg q (k / 8) (k % 8) := by
  obtain ⟨hq', hname, hw, hb⟩ := rowsOk_get hrows hq
  have hpre : ∀ i ≤ 5, 16 * ((c.ports.toList.take i).map (·.words)).sum = preBits L H i := by
    intro i hi
    induction i with
    | zero => simp [preBits]
    | succ i ih =>
      obtain ⟨hi', -, hwi, -⟩ := rowsOk_get hrows (show i < 5 by omega)
      have hB : i < (Flock.RecOpen.ports L H).bits.toList.length := by
        simp [Flock.RecOpen.Ports.bits]
        omega
      have hBi : (Flock.RecOpen.ports L H).bits.toList[i] = (Flock.RecOpen.ports L H).bits[i]! := by
        rw [Array.getElem_toList, getElem!_pos _ i (by simpa using hB)]
      unfold preBits at ih ⊢
      rw [List.take_add_one, List.map_append, List.sum_append, List.getElem?_eq_getElem (by simpa using hi'),
        List.take_add_one, List.sum_append, List.getElem?_eq_getElem hB, hBi, ← hwi, ← ih (by omega)]
      simp [Nat.mul_add]
  have hst : 16 * (FlockSoundness.Typed.starts c)[q]! = preBits L H q := by
    rw [FlockSoundness.Discharge.Layout.starts_get c hq']
    exact hpre q (by omega)
  have hg := inGroups_of_laid hl
  have h16 : ∀ b ∈ (Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)).toList, b = 16 := by
    intro b hb
    rw [Flock.RecOpen.wordPorts, Array.toList_replicate] at hb
    exact List.eq_of_mem_replicate hb
  have hsz : (Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)).size = FlockVBridge.unitIn L H / 16 := by
    simp [Flock.RecOpen.wordPorts]
  have hN : 16 * (Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)).size ≤ c.unit.inWords * Flock.WORD := by
    rw [FlockVBridge.inWords_unit hl, hsz, FlockSoundness.Discharge.Hm.inRowsOf, FlockVBridge.foldl_wordPorts]
    exact Flock.HmNets.le_groupWords _
  have hle := preBits_le L H hq
  have hT : FlockVBridge.unitIn L H = (Flock.RecOpen.ports L H).bits.foldl (· + ·) 0 := rfl
  have hfit : (FlockSoundness.Typed.starts c)[q]! + c.ports[q].words ≤
      (Flock.RecOpen.wordPorts (FlockVBridge.unitIn L H)).size := by
    rw [hsz]
    omega
  have key := srcKey_inOrder hg rfl h16 hnd (j := j) (k := k) (by rw [leavesOk_get hlv hj, hsz]) hq' (by omega) (by omega)
    hfit hN
  rw [hst] at key
  exact ⟨hq', hname, by omega, by omega, key⟩

theorem preBits_aux (L H : ℕ) : preBits L H 1 = (Flock.RecOpen.ports L H).row := by
  simp [preBits, Flock.RecOpen.Ports.bits]

theorem preBits_acc (L H : ℕ) :
    preBits L H 2 = (Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux := by
  simp [preBits, Flock.RecOpen.Ports.bits]

theorem preBits_coef (L H : ℕ) :
    preBits L H 3 = (Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc := by
  simp [preBits, Flock.RecOpen.Ports.bits]
  omega

theorem preBits_dirs (L H : ℕ) :
    preBits L H 4 = (Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc +
      (Flock.RecOpen.ports L H).coef := by
  simp [preBits, Flock.RecOpen.Ports.bits]
  omega

theorem bits_acc (L H : ℕ) : (Flock.RecOpen.ports L H).bits[2]! = (Flock.RecOpen.ports L H).acc := rfl

theorem bits_coef (L H : ℕ) : (Flock.RecOpen.ports L H).bits[3]! = (Flock.RecOpen.ports L H).coef := rfl

theorem bits_dirs (L H : ℕ) : (Flock.RecOpen.ports L H).bits[4]! = (Flock.RecOpen.ports L H).dirs := rfl

/-! ## The ports the reads file names -/

/-- **A port `checkValue` accepted is a port of the reads file.** -/
theorem own_of_checkValue {c : Flock.Circuit} {pub : Flock.Public} {own : Array Flock.Registered.Port}
    {port value : String} {rows : Array (Array ℕ)} {what : ℕ → String}
    (h : Flock.RecCoins.checkValue c pub own port value rows what = .ok ()) : ∃ pt ∈ own, pt.name = port := by
  unfold Flock.RecCoins.checkValue at h
  dsimp only at h
  split at h
  · rename_i reg hf
    exact ⟨reg, Array.mem_of_find?_eq_some hf, by simpa using Array.find?_some hf⟩
  · cases h

/-- **A statement whose `rec-acc` ports are a level's has `acc` in the reads file `checkAcc` accepted.** -/
theorem acc_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (hs : RecVStar.accPorts s = ["acc", "acc_out"])
    (h : RecVStar.checkAcc s own rows = .ok ()) : ∃ pt ∈ own, pt.name = "acc" := by
  unfold RecVStar.checkAcc at h
  rw [hs] at h
  obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  rw [List.forIn_cons] at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  split at h
  · rename_i p hf
    exact ⟨p, Array.mem_of_find?_eq_some hf, by simpa using Array.find?_some hf⟩
  · cases h

/-- **`checkStmt` ran `checkAcc` and `checkAccReads` on V*'s statement `j`** at the verifier's own reads
(`checkStmt_ok`'s walk). -/
theorem checkAcc_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) :
    ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ RecVStar.checkAcc s own rows = .ok () ∧
      RecVStar.checkAccReads i s own = .ok () := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · cases h
        obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨s, hs, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨hck, -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hs
        obtain ⟨ss, hss, hj, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, hacc, h⟩ := FlockSoundness.Refine.ebind_ok h
        exact ⟨ss, s, hss, hj, hacc, h⟩
      · cases h

/-- **An R level's checked session has `acc` in its reads file**: where V*'s statement `j` is R's level `l`
(`hj`), `checkStmt` ran `checkAcc` on it (`checkAcc_of_checkStmt`), whose `rec-acc` ports are `acc` and `acc_out`. -/
theorem acc_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) {l LANES H q : ℕ}
    (hj : ∀ ss : Array VStar.Stmt, RecVStar.statements i = .ok ss → ss[j]? = some (.recOpenStacked l LANES H q)) :
    ∃ pt ∈ own, pt.name = "acc" := by
  obtain ⟨ss, s, hss, hs, hacc, -⟩ := checkAcc_of_checkStmt h
  obtain rfl : s = .recOpenStacked l LANES H q := Option.some.inj (hs.symm.trans (hj ss hss))
  exact acc_of_checkAcc (s := .recOpenStacked l LANES H q) rfl hacc

section Keyed

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **An accepted opening's input columns are registered bits where its rows are registered reads**
(`srcKey_opening`): at a unit the verifier accepted as `rec-open/l<L>-h<H>` or `rec-open-stacked/l<L>-h<H>`, copy `j`'s
column `preBits q + k`, bit `k` of its row `q < 5`, is a registered bit wherever unit `u`'s row `q` is a registered
read. -/
theorem regBit_opening (hrec : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (L, H) ∨ Flock.RecOpen.stackedParams c.unit.name = some (L, H))
    {j : ℕ} (hj : j < c.leavesIn.size) {u : Fin nv} {q : ℕ} (hq : q < 5)
    (hread : ∀ hq' : q < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨q, hq'⟩))) {k : ℕ}
    (hk : k < (Flock.RecOpen.ports L H).bits[q]!) : ∃ b, RegBit zo u j (preBits L H q + k) b := by
  obtain ⟨hrows, hlv⟩ := check_rows hrec hp
  obtain ⟨name, hl⟩ : ∃ name, Flock.HmNets.netOf name (Flock.RecOpen.unitLaid L H) = .ok c.unit := by
    rcases hp with hp | hp
    · exact ⟨_, (Flock.RecOpen.check_ok hrec hp).2⟩
    · exact ⟨_, (Flock.RecOpen.check_ok_stacked hrec hp).2⟩
  obtain ⟨hq', -, hk', hcol, hkey⟩ :=
    srcKey_opening hl hrows hlv (FlockSoundness.Discharge.Aliased.portBits_nodup hU) hj hq hk
  rw [← FlockSoundness.Discharge.Layout.ofNet_nIn_eq (hU := hU)] at hcol hkey
  exact ⟨_, hcol, ⟨q, hq'⟩, k, hk', hkey, hread hq', rfl⟩

/-- **`opensTopAt_of_regs`' `hacc` and `hcoef` from the verifier's check** (`regBit_opening`): where unit `u`'s `acc`
and `coef` rows (rows 2 and 3) are registered reads, each of copy `j`'s `acc` and `coef` columns is a registered bit. -/
theorem accCoef_of_check (hrec : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (L, H) ∨ Flock.RecOpen.stackedParams c.unit.name = some (L, H))
    {j : ℕ} (hj : j < c.leavesIn.size) {u : Fin nv}
    (hacc : ∀ hq : 2 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨2, hq⟩)))
    (hcoef : ∀ hq : 3 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨3, hq⟩))) :
    (∀ t < (Flock.RecOpen.ports L H).acc, ∃ b,
      RegBit zo u j ((Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + t) b) ∧
    ∀ t < (Flock.RecOpen.ports L H).coef, ∃ b, RegBit zo u j
      ((Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc + t) b := by
  refine ⟨fun t ht => ?_, fun t ht => ?_⟩
  · have := regBit_opening zo hrec hp hj (q := 2) (by decide) hacc (k := t) (by rw [bits_acc]; exact ht)
    rwa [preBits_acc] at this
  · have := regBit_opening zo hrec hp hj (q := 3) (by decide) hcoef (k := t) (by rw [bits_coef]; exact ht)
    rwa [preBits_coef] at this

/-- **The `dirs` columns `opensTopAt_of_regs` reads are keyed** (`regBit_opening`): where unit `u`'s `dirs` row (row 4)
is a registered read, each of copy `j`'s `dirs` columns is a registered bit; which bit is `dirs_of_check`'s. -/
theorem dirsCols_of_check (hrec : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (L, H) ∨ Flock.RecOpen.stackedParams c.unit.name = some (L, H))
    {j : ℕ} (hj : j < c.leavesIn.size) {u : Fin nv}
    (hdirs : ∀ hq : 4 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨4, hq⟩))) :
    ∀ t < (Flock.RecOpen.ports L H).dirs, ∃ b, RegBit zo u j ((Flock.RecOpen.ports L H).row +
      (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc + (Flock.RecOpen.ports L H).coef + t) b := by
  intro t ht
  have := regBit_opening zo hrec hp hj (q := 4) (by decide) hdirs (k := t) (by rw [bits_dirs]; exact ht)
  rwa [preBits_dirs] at this

/-- **An accepted R level's unit reads its rows 2–4 as registered reads** (`accCoef_of_check`'s `hacc` and `hcoef`,
`dirsCols_of_check`'s `hdirs`), where the session registers each row of a port its reads file names (`hread`, VBridge's
`RowsRegistered` at unit `u`): the verifier names the unit's rows 2–4 `acc`, `coef` and `dirs` (`rowsOk_get`), and the
reads file has a port of each name, `acc` by `checkAcc` (`hacc`, `acc_of_checkStmt`), `coef` and `dirs` by
`checkStackedLevel`'s `checkValue`s (`own_of_checkValue`). -/
theorem regReads_of_check (hrec : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.stackedParams c.unit.name = some (L, H)) {c3 : Flock.RecCoins.C3} {pub : Flock.Public}
    (hlv : Flock.RecCoins.checkStackedLevel c3 c pub zo.own L H = .ok ()) (hacc : ∃ pt ∈ zo.own, pt.name = "acc")
    {u : Fin nv}
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q))) :
    (∀ hq : 2 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨2, hq⟩))) ∧
    (∀ hq : 3 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨3, hq⟩))) ∧
    (∀ hq : 4 < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨4, hq⟩))) := by
  obtain ⟨hrows, -⟩ := check_rows hrec (.inr hp)
  have hname : ∀ (q : ℕ) (hq : q < 5) (hq' : q < c.ports.size) (nm : String),
      ((Flock.RecOpen.rows L H)[q]!).1 = nm → (∃ pt ∈ zo.own, pt.name = nm) → zo.rd.isRead (.inl (u, ⟨q, hq'⟩)) := by
    intro q hq hq' nm hnm hown
    obtain ⟨_, hn, -⟩ := rowsOk_get hrows hq
    obtain ⟨pt, hpt, hpn⟩ := hown
    exact hread ⟨q, hq'⟩ ⟨pt, hpt, hpn.trans (hn.trans hnm).symm⟩
  obtain ⟨_, _, _, -, -, hcoef⟩ := level_of_check hlv
  obtain ⟨_, _, _, -, -, hdirs⟩ := dirsLevel_of_check hlv
  exact ⟨fun hq => hname 2 (by decide) hq "acc" rfl hacc,
    fun hq => hname 3 (by decide) hq "coef" rfl (own_of_checkValue hcoef),
    fun hq => hname 4 (by decide) hq "dirs" rfl (own_of_checkValue hdirs)⟩

/-- **`opensTopAt_of_regs`' `acc`, `coef` and `dirs` columns are registered bits at an accepted R level's unit**
(`regReads_of_check`, `accCoef_of_check`, `dirsCols_of_check`), where the session registers each row of a port its reads
file names (`hread`); which bit each `dirs` column holds is `dirs_of_check`'s. -/
theorem regCols_of_check (hrec : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hp : Flock.RecOpen.stackedParams c.unit.name = some (L, H)) {c3 : Flock.RecCoins.C3} {pub : Flock.Public}
    (hlv : Flock.RecCoins.checkStackedLevel c3 c pub zo.own L H = .ok ()) (hacc : ∃ pt ∈ zo.own, pt.name = "acc")
    {j : ℕ} (hj : j < c.leavesIn.size) {u : Fin nv}
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q))) :
    (∀ t < (Flock.RecOpen.ports L H).acc, ∃ b,
      RegBit zo u j ((Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + t) b) ∧
    (∀ t < (Flock.RecOpen.ports L H).coef, ∃ b, RegBit zo u j
      ((Flock.RecOpen.ports L H).row + (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc + t) b) ∧
    ∀ t < (Flock.RecOpen.ports L H).dirs, ∃ b, RegBit zo u j ((Flock.RecOpen.ports L H).row +
      (Flock.RecOpen.ports L H).aux + (Flock.RecOpen.ports L H).acc + (Flock.RecOpen.ports L H).coef + t) b := by
  obtain ⟨h2, h3, h4⟩ := regReads_of_check zo hrec hp hlv hacc hread
  obtain ⟨hA, hC⟩ := accCoef_of_check zo hrec (.inr hp) hj h2 h3
  exact ⟨hA, hC, dirsCols_of_check zo hrec (.inr hp) hj h4⟩

end Keyed

end FlockC3.RegLevels
