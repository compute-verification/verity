import Proofs.Flock.VBridge.Net
import Proofs.Flock.Soundness.ExecRows

/-!
# V*'s opening in a unit of a partition

`laid_recOpen` reads a laid-out net's rows at column values `v` (`LaidRows`); this file gives those values from the
audit. A unit of a partition that is an instance of rows `R` (`Partition.IsRowsUnit`) satisfies them at its local
evaluation (`sat_localEval`). Copy `j` of `k` stacked copies satisfies the copy's rows (`sat_stack`), and its constant
row carries the shared constant (`stack_one`). The rows of a net (`Rows.ofNet`), satisfied with the constant 1, are the
laid-out rows of the `Laid` the verifier built it from (`netOf`, `laidRows_ofNet`).

So copy `j` of a unit whose rows are a stack of V*'s laid-out opening computes `laid_recOpen`'s outputs from its inputs
at the unit's local evaluation (`unit_recOpen`, over `copyVal`). The copy's input columns read the transcript
(`copyVal_input`), and so does every committed column of a correct unit (`copyVal_committed`).
-/

namespace FlockVBridge

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge.Hm

/-! ## A unit's rows hold at its local evaluation -/

theorem sat_localEval {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows}
    (I : P.IsRowsUnit u R) (X : Fin C.N → Bool) : R.Sat fun c => P.localEval X u (I.wire c) := by
  intro i
  show P.localEval X u (I.wire (R.comp i)) = _
  rw [P.localEval_of_mem X (I.mem i), I.op i]
  simp only [Op.apply, xorSum_map]
  rfl

/-- A column that is no computed row is read from the transcript. -/
theorem localEval_wire {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows}
    (I : P.IsRowsUnit u R) (X : Fin C.N → Bool) {c : Fin R.w} (hc : R.nIn + R.nComp ≤ c.val ∨ c.val < R.nIn) :
    P.localEval X u (I.wire c) = X (I.wire c) :=
  P.localEval_of_not_mem X (I.outside c fun i h => by
    have := congrArg Fin.val h
    simp only [Rows.comp] at this
    omega)

/-! ## One copy of a stack -/

theorem stackCol_congr (R : Rows) (k : ℕ) {u u' : ℕ} (hu : u < k) (hu' : u' < k) (h : u = u') (c : Fin R.w) :
    R.stackCol k hu c = R.stackCol k hu' c := by
  subst h
  rfl

theorem stackRow_copy (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (f : Fin R.nComp → List (Fin R.w)) (j : Fin R.nComp) :
    R.stackRow k f ⟨u * (R.nComp + 1) + (1 + j.val), Rows.mul_add_lt hu (by omega)⟩ = (f j).map (R.stackCol k hu) := by
  obtain ⟨hd, hm⟩ := div_mod_copy (u := u) (show 1 + j.val < R.nComp + 1 by omega)
  unfold Rows.stackRow
  rw [dite_eq_right (by simp only [hm]; omega)]
  have hj : (⟨(u * (R.nComp + 1) + (1 + j.val)) % (R.nComp + 1) - 1, by
      have := Nat.mod_lt (u * (R.nComp + 1) + (1 + j.val)) (show 0 < R.nComp + 1 by omega); omega⟩ : Fin R.nComp) = j :=
    Fin.ext (by simp only [hm]; omega)
  rw [hj]
  exact List.map_congr_left fun c _ => stackCol_congr R k _ hu hd c

theorem stackRow_const (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (f : Fin R.nComp → List (Fin R.w)) :
    R.stackRow k f ⟨u * (R.nComp + 1) + 0, Rows.mul_add_lt hu (by omega)⟩ = [(R.stack k).one] := by
  obtain ⟨-, hm⟩ := div_mod_copy (u := u) (show 0 < R.nComp + 1 by omega)
  unfold Rows.stackRow
  rw [dite_eq_left hm]
  rfl

theorem stack_a (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (j : Fin R.nComp) :
    (R.stack k).a ⟨u * (R.nComp + 1) + (1 + j.val), Rows.mul_add_lt hu (by omega)⟩ = (R.a j).map (R.stackCol k hu) :=
  stackRow_copy R k hu R.a j

theorem stack_b (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (j : Fin R.nComp) :
    (R.stack k).b ⟨u * (R.nComp + 1) + (1 + j.val), Rows.mul_add_lt hu (by omega)⟩ = (R.b j).map (R.stackCol k hu) :=
  stackRow_copy R k hu R.b j

theorem stack_a_one (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) :
    (R.stack k).a ⟨u * (R.nComp + 1) + 0, Rows.mul_add_lt hu (by omega)⟩ = [(R.stack k).one] :=
  stackRow_const R k hu R.a

theorem stack_b_one (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) :
    (R.stack k).b ⟨u * (R.nComp + 1) + 0, Rows.mul_add_lt hu (by omega)⟩ = [(R.stack k).one] :=
  stackRow_const R k hu R.b

/-- Copy `u`'s computed row `j` is the stack's computed row `u (nComp + 1) + 1 + j`. -/
theorem stack_comp (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) (j : Fin R.nComp) :
    (R.stack k).comp ⟨u * (R.nComp + 1) + (1 + j.val), Rows.mul_add_lt hu (by omega)⟩ = R.stackCol k hu (R.comp j) := by
  have h1 : ¬ (R.comp j).val < R.nIn := by simp [Rows.comp]
  have h2 : (R.comp j).val < R.nIn + R.nComp := by simp [Rows.comp]
  apply Fin.ext
  rw [Rows.stackCol, dite_eq_right h1, dite_eq_left h2]
  show k * R.nIn + (u * (R.nComp + 1) + (1 + j.val)) = k * R.nIn + (u * (R.nComp + 1) + (1 + ((R.comp j).val - R.nIn)))
  simp [Rows.comp]

/-- Copy `u`'s constant is the stack's computed row `u (nComp + 1)`. -/
theorem stack_comp_one (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) :
    (R.stack k).comp ⟨u * (R.nComp + 1) + 0, Rows.mul_add_lt hu (by omega)⟩ = R.stackCol k hu R.one := by
  have h1 : ¬ R.one.val < R.nIn := by simp [Rows.one]
  have h2 : ¬ R.one.val < R.nIn + R.nComp := by simp [Rows.one]
  apply Fin.ext
  rw [Rows.stackCol, dite_eq_right h1, dite_eq_right h2]
  rfl

/-- **One copy of a satisfied stack is satisfied**, at the stack's values on the copy's columns. -/
theorem sat_stack (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) {z : Fin (R.stack k).w → Bool} (h : (R.stack k).Sat z) :
    R.Sat (z ∘ R.stackCol k hu) := by
  intro j
  have hs := h ⟨u * (R.nComp + 1) + (1 + j.val), Rows.mul_add_lt hu (by omega)⟩
  rw [stack_comp R k hu j, stack_a R k hu j, stack_b R k hu j] at hs
  exact hs.trans (congrArg₂ (· && ·) (xorSum_map _ _ _) (xorSum_map _ _ _))

/-- **A copy's constant carries the stack's**: its constant row reads the shared constant. -/
theorem stack_one (R : Rows) (k : ℕ) {u : ℕ} (hu : u < k) {z : Fin (R.stack k).w → Bool} (h : (R.stack k).Sat z) :
    z (R.stackCol k hu R.one) = z (R.stack k).one := by
  have hs := h ⟨u * (R.nComp + 1) + 0, Rows.mul_add_lt hu (by omega)⟩
  rw [stack_comp_one R k hu, stack_a_one R k hu, stack_b_one R k hu] at hs
  simpa [xorSum] using hs

/-! ## A net's rows are its laid-out rows -/

/-- Column values `z` at every column, `false` past the last. -/
def colVal {w : ℕ} (z : Fin w → Bool) (j : ℕ) : Bool := if h : j < w then z ⟨j, h⟩ else false

theorem colVal_fin {w : ℕ} (z : Fin w → Bool) (c : Fin w) : colVal z c.val = z c := by
  simp [colVal, c.isLt]

theorem xorRow_colVal {w : ℕ} (z : Fin w → Bool) {l : List (Fin w)} {cs : Array ℕ} (h : l.map Fin.val = cs.toList) :
    xorRow (colVal z) cs = xorSum l z := by
  unfold xorRow xorSum
  rw [← h]
  clear h
  induction l with
  | nil => rfl
  | cons c l ih => simp only [List.map_cons, List.foldr_cons, colVal_fin, ih]

/-- **A net's satisfied rows are the laid-out rows it was built from.** If the verifier built `net` from `l` (`netOf`),
the net's rows (`Rows.ofNet`) hold at `z` and its constant is 1, then `l`'s rows hold at `z` (`LaidRows`), from the end
of the net's input rows. -/
theorem laidRows_ofNet {net : Flock.Net} (hB : Built net) {name : String} {l : Flock.HmNets.Laid}
    (hl : Flock.HmNets.netOf name l = .ok net) (hu : net.unitLog ≤ 32) {z : Fin (Rows.ofNet hB).w → Bool}
    (hz : (Rows.ofNet hB).Sat z) (hone : z (Rows.ofNet hB).one = true) :
    LaidRows l (net.inWords * Flock.WORD) (colVal z) := by
  obtain ⟨-, ha, hb, -, hcp, -, -, hle, -⟩ := Flock.Net.ofRows_ok hl
  have hnIn : (Rows.ofNet hB).nIn = net.inWords * Flock.WORD := by rw [Rows.ofNet_nIn]; omega
  have h1 : (Rows.ofNet hB).nIn + (Rows.ofNet hB).nComp = l.constPos := by
    rw [← hcp]; exact Rows.ofNet_one hB
  refine ⟨?_, fun i hi1 hi2 => ?_⟩
  · have hc : l.constPos = (Rows.ofNet hB).one.val := by rw [Rows.ofNet_one, hcp]
    rw [hc, colVal_fin, hone]
  · have hw : (2 : ℕ) ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hu
    have hj : i - (Rows.ofNet hB).nIn < (Rows.ofNet hB).nComp := by omega
    have hij : (Rows.ofNet hB).nIn + (⟨i - (Rows.ofNet hB).nIn, hj⟩ : Fin _).val = i := by simp only; omega
    have hra := Rows.ofNet_a hB hu ⟨i - (Rows.ofNet hB).nIn, hj⟩
    have hrb := Rows.ofNet_b hB hu ⟨i - (Rows.ofNet hB).nIn, hj⟩
    rw [ofRows_row ha hw, hij] at hra
    rw [ofRows_row hb hw, hij] at hrb
    rw [xorRow_colVal z hra, xorRow_colVal z hrb, ← hz ⟨i - (Rows.ofNet hB).nIn, hj⟩]
    have hc : i = ((Rows.ofNet hB).comp ⟨i - (Rows.ofNet hB).nIn, hj⟩).val := by
      rw [Rows.ofNet_comp]; exact hij.symm
    conv_rhs => rw [hc]
    rw [colVal_fin]

/-! ## A copy's column values -/

section Copy

variable {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows} {k : ℕ}

/-- **Copy `j`'s column values** in a unit of `k` stacked copies of `R`: column `c` of the copy is the unit's local
evaluation at the copy's column `c`'s wire. -/
noncomputable def copyVal (I : P.IsRowsUnit u (R.stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k) : ℕ → Bool :=
  colVal fun c => P.localEval X u (I.wire (R.stackCol k hj c))

/-- A copy's input column reads the transcript at its wire. -/
theorem copyVal_input (I : P.IsRowsUnit u (R.stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k) {c : ℕ}
    (hc : c < R.nIn) :
    copyVal I X hj c = X (I.wire (R.stackCol k hj ⟨c, by simp only [Rows.w]; omega⟩)) := by
  have hcw : c < R.w := by simp only [Rows.w]; omega
  unfold copyVal colVal
  rw [dite_eq_left hcw]
  refine localEval_wire I X (Or.inr ?_)
  have hlt := Rows.mul_add_lt hj hc
  show (R.stackCol k hj ⟨c, _⟩).val < k * R.nIn
  rw [Rows.stackCol, dite_eq_left hc]
  exact hlt

/-- A committed column of a correct unit reads the transcript at its wire. -/
theorem copyVal_committed (I : P.IsRowsUnit u (R.stack k)) {X : Fin C.N → Bool} (hX : P.Correct X u) {j : ℕ}
    (hj : j < k) (c : Fin R.w) (hm : I.wire (R.stackCol k hj c) ∈ P.committed) :
    copyVal I X hj c.val = X (I.wire (R.stackCol k hj c)) := by
  rw [copyVal, colVal_fin]
  by_cases hu : P.unit (I.wire (R.stackCol k hj c)) = some u
  · exact (hX _ hm hu).symm
  · exact P.localEval_of_not_mem X hu

/-- A copy's rows hold at its column values, and its constant is the transcript's at the stack's constant. -/
theorem sat_copy (I : P.IsRowsUnit u (R.stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k) :
    R.Sat (fun c => P.localEval X u (I.wire (R.stackCol k hj c))) ∧
      P.localEval X u (I.wire (R.stackCol k hj R.one)) = X (I.wire (R.stack k).one) := by
  have hs := sat_localEval I X
  refine ⟨sat_stack R k hj hs, (stack_one R k hj hs).trans (localEval_wire I X (Or.inl ?_))⟩
  simp [Rows.one]

end Copy

/-! ## V*'s laid-out opening in a unit -/

theorem inWords_recOpen {LANES H : ℕ} {p : Ports} {outBits : Array ℕ} {name : String} {net : Flock.Net}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf p.bits := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hl
  rw [hw, recOpenLaid, layout_inGroups]
  simp [inRowsOf]

/-- **A copy of V*'s laid-out opening, in a unit of any partition, computes the climb and the consistency term.** Let
the verifier build `net` from `recOpenLaid` (`netOf`), and let unit `u` be an instance of `k` stacked copies of the
net's rows (`Rows.ofNet`), the transcript `X` carrying 1 at the stack's constant. Then at copy `j`'s column values
(`copyVal`), `laid_recOpen` holds: if the copy's input columns hold the row, `acc`, the coefficients, the siblings and
the position's bits, its output columns from `outStart` hold the verifier of record's climb and 64 zero bytes, then the
consistency term at `k = 0, 1` and the rest of `acc`. -/
theorem unit_recOpen {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) {name : String}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok net)
    (hu : net.unitLog ≤ 32) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (r a c : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (Flock.MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (Flock.MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (Flock.MerkleScheme.rowBytes c).size) (hdirsN : H ≤ p.dirs)
    (hrow : ColsHold (copyVal I X hj) 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold (copyVal I X hj) (p.row + 512 * i) ss[i])
    (hacc : ColsHold (copyVal I X hj) (p.row + p.aux) (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal I X hj) (p.row + p.aux + p.acc) (Flock.MerkleScheme.rowBytes c))
    (hdir : ∀ h < H, copyVal I X hj (p.row + p.aux + p.acc + p.coef + h) = idx.testBit h) :
    ColsHold (copyVal I X hj) (outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits)
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest)) := by
  obtain ⟨hs, h1⟩ := sat_copy I X hj
  have hL := laidRows_ofNet hB hl hu hs (h1.trans hone)
  rw [inWords_recOpen hl] at hL
  exact laid_recOpen ms hms LANES H hH p outBits _ hout hL r a c rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN
    hrow haux hacc hcoef hdir

end FlockVBridge
