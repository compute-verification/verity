import Proofs.Flock.VBridge.Keyed
import Proofs.Flock.Soundness.ExecCircuit
import Flock.RecOpen

/-!
# The verifier's opening unit is V*'s laid-out opening

`Flock.RecOpen` is the verifier's copy of V*'s opening builder, the net `HmRow.parse` checks a `rec-open/l<L>-h<H>` unit
against (`RecOpen.check`). It is this package's builder word for word, over its own `Term`, `Res` and `GfS` (`termOf`,
`resOf`, `gfsOf`), its own byte helpers (`byteBit` is `bitL`) and SHA-512's IV as words (`ivW = preCv [] 0`):
`recOpenNet_eq`.

The verifier lays the net out as M0 does, one 16-bit port per word (`RecOpen.unitLaid`), where `recOpenLaid` takes the
ports as given. `layout` reads its ports only through their sums, except in the groups it records, which `LaidRows`
does not read (`laidRows_layout_congr`), so `laid_recOpen` holds at `unitLaid` (`laid_unit`); its outputs are 2,048 bits,
its 128 output words (`length_recOpenOuts`).

`unit_check`: a unit the verifier accepted (`RecOpen.check`) under the name `rec-open/l<L>-h<H>` has `H ≤ 32` and is the
verifier's net at `L` and `H` (`RecOpen.check_ok`), so a copy of it in a unit of any partition computes the climb and the
consistency term; `keyProg_parse` is the same for a unit of `Placed.keyProg` of a circuit `HmRow.parse` accepted, which
ran the check (`ParseFacts.recNet`).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Hm

/-! ## The builders are equal -/

theorem bitsOf_eq : Flock.RecOpen.bitsOf = bitsOf := rfl
theorem shaMsg_eq : Flock.RecOpen.shaMsg = shaMsg := rfl
theorem blockAt_eq : Flock.RecOpen.blockAt = blockAt := rfl
theorem shaBlock_eq : Flock.RecOpen.shaBlock = shaBlock := rfl

theorem shaBlocks_eq (msg : List Lin) (n : ℕ) :
    ∀ k st, Flock.RecOpen.shaBlocks msg n k st = shaBlocks msg n k st := by
  induction n with
  | zero => intro k st; rfl
  | succ n ih => intro k st; simp only [Flock.RecOpen.shaBlocks, shaBlocks, shaBlock_eq, blockAt_eq, ih]

theorem shaFrom_eq (cv : List (BitVec 64)) (bits : List Lin) (pre : ℕ) :
    Flock.RecOpen.shaFrom cv bits pre = shaFrom cv bits pre := by
  simp only [Flock.RecOpen.shaFrom, shaFrom, shaMsg_eq, shaBlocks_eq]

/-- SHA-512's IV as words is the chaining value after no block. -/
theorem ivW_eq : Flock.RecOpen.ivW = preCv [] 0 := rfl

theorem swap_eq : Flock.RecOpen.swap = swap := rfl
theorem sibForms_eq : Flock.RecOpen.sibForms = sibForms := rfl

theorem climbFrom_eq (d : List Lin) (ss : List (List Lin)) :
    ∀ h cur, Flock.RecOpen.climbFrom d h cur ss = climbFrom d h cur ss := by
  induction ss with
  | nil => intro h cur; rfl
  | cons s rest ih =>
    intro h cur
    simp only [Flock.RecOpen.climbFrom, climbFrom, swap_eq, shaFrom_eq, ivW_eq, ih]

theorem climbRow_eq (H : ℕ) (row aux d : List Lin) : Flock.RecOpen.climbRow H row aux d = climbRow H row aux d := by
  simp only [Flock.RecOpen.climbRow, climbRow, shaFrom_eq, ivW_eq, climbFrom_eq, sibForms_eq]

theorem addL_eq : ∀ f g : List Lin, Flock.RecOpen.addL f g = addL f g
  | [], _ => rfl
  | _ :: _, [] => rfl
  | f :: fs, g :: gs => by rw [Flock.RecOpen.addL, addL, addL_eq fs gs]

theorem shiftL_eq : Flock.RecOpen.shiftL = shiftL := rfl

theorem reduceG_eq (f : List Lin) : Flock.RecOpen.reduceG f = reduceG f := by
  simp only [Flock.RecOpen.reduceG, reduceG, Flock.RecOpen.foldG, foldG, Flock.RecOpen.mulS, mulS, addL_eq,
    shiftL_eq]

theorem leaves_eq (n : ℕ) : ∀ a b, Flock.RecOpen.leaves n a b = leaves n a b := by
  induction n with
  | zero => intro a b; rfl
  | succ n ih => intro a b; simp only [Flock.RecOpen.leaves, leaves, addL_eq, ih]

theorem recomb_eq (n : ℕ) : ∀ l, Flock.RecOpen.recomb n l = recomb n l := by
  induction n with
  | zero => intro l; rfl
  | succ n ih => intro l; simp only [Flock.RecOpen.recomb, recomb, addL_eq, shiftL_eq, ih]

theorem xorL_eq : Flock.RecOpen.xorL = xorL := rfl
theorem xorAll_eq : Flock.RecOpen.xorAll = xorAll := rfl

theorem leafSums_eq (ps : List (List Lin × List Lin)) : ∀ sums, Flock.RecOpen.leafSums ps sums = leafSums ps sums := by
  induction ps with
  | nil => intro sums; rfl
  | cons p ps ih => intro sums; simp only [Flock.RecOpen.leafSums, leafSums, leaves_eq, xorL_eq, ih]

/-- The verifier's term as this package's. -/
def termOf : Flock.RecOpen.Term → Term
  | .w i => .w i
  | .v i => .v i
  | .wt i u => .wt i u

/-- The verifier's residual as this package's. -/
def resOf (r : Flock.RecOpen.Res) : Res := ⟨r.mul, r.add.map termOf⟩

/-- The verifier's S as this package's. -/
def gfsOf (S : Flock.RecOpen.GfS) : GfS := ⟨S.ops.map (·.map termOf), S.res.map resOf⟩

theorem termForms_eq (W V : List Lin) (t : Flock.RecOpen.Term) :
    Flock.RecOpen.termForms W V t = termForms W V (termOf t) := by
  cases t <;> rfl

theorem map_termForms (W V : List Lin) (ts : List Flock.RecOpen.Term) :
    ts.map (Flock.RecOpen.termForms W V) = (ts.map termOf).map (termForms W V) := by
  rw [List.map_map]
  exact List.map_congr_left fun t _ => termForms_eq W V t

theorem opForms_eq (S : Flock.RecOpen.GfS) (W V : List Lin) (k : ℕ) :
    Flock.RecOpen.opForms S W V k = opForms (gfsOf S) W V k := by
  have h : (gfsOf S).ops.getD k [] = (S.ops.getD k []).map termOf := by
    simp only [gfsOf, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases S.ops[k]? <;> rfl
  rw [Flock.RecOpen.opForms, opForms, h, map_termForms, xorAll_eq]

theorem residual_eq (S : Flock.RecOpen.GfS) (W V : List Lin) (r : Flock.RecOpen.Res) :
    Flock.RecOpen.residual S W V r = residual (gfsOf S) W V (resOf r) := by
  simp only [Flock.RecOpen.residual, residual, resOf, map_termForms, opForms_eq, leafSums_eq, xorAll_eq, xorL_eq,
    recomb_eq, reduceG_eq]
  rfl

theorem residualsGo_eq (S : Flock.RecOpen.GfS) (W V : List Lin) (rs : List Flock.RecOpen.Res) :
    Flock.RecOpen.residualsGo S W V rs = residualsGo (gfsOf S) W V (rs.map resOf) := by
  induction rs with
  | nil => rfl
  | cons r rs ih => simp only [Flock.RecOpen.residualsGo, residualsGo, List.map_cons, residual_eq, ih]

theorem residualForms_eq (S : Flock.RecOpen.GfS) (W V : List Lin) :
    Flock.RecOpen.residualForms S W V = residualForms (gfsOf S) W V :=
  residualsGo_eq S W V S.res

theorem consStructure_eq (LANES : ℕ) : gfsOf (Flock.RecOpen.consStructure LANES) = consStructure LANES := by
  simp only [gfsOf, Flock.RecOpen.consStructure, consStructure, resOf, List.map_append, List.map_map,
    Function.comp_def, List.map_cons, List.map_nil, termOf]

theorem recOpen_eq (LANES H : ℕ) (row aux acc coef dirs : List Lin) :
    Flock.RecOpen.recOpen LANES H row aux acc coef dirs = recOpen LANES H row aux acc coef dirs := by
  simp only [Flock.RecOpen.recOpen, recOpen, residualForms_eq, consStructure_eq, climbRow_eq]

/-- The verifier's ports as this package's. -/
def portsOf (p : Flock.RecOpen.Ports) : Ports := ⟨p.row, p.aux, p.acc, p.coef, p.dirs⟩

/-- **The verifier's opening builder is V*'s**, at the same ports. -/
theorem recOpenNet_eq (LANES H : ℕ) (p : Flock.RecOpen.Ports) :
    Flock.RecOpen.recOpenNet LANES H p = recOpenNet LANES H (portsOf p) := by
  simp only [Flock.RecOpen.recOpenNet, recOpenNet, portsOf, recOpen_eq]

theorem recOpenOuts_eq (LANES H : ℕ) (p : Flock.RecOpen.Ports) :
    Flock.RecOpen.recOpenOuts LANES H p = recOpenOuts LANES H (portsOf p) := by
  simp only [Flock.RecOpen.recOpenOuts, recOpenOuts, recOpenNet_eq]

/-! ## `layout` reads its ports through their sums -/

theorem layout_congr (st : St) {i i' o o' : Array ℕ} (outs : Array Lin) (hi : i.foldl (· + ·) 0 = i'.foldl (· + ·) 0)
    (ho : o.foldl (· + ·) 0 = o'.foldl (· + ·) 0) :
    (layout st i o outs).constPos = (layout st i' o' outs).constPos ∧ (layout st i o outs).ra = (layout st i' o' outs).ra ∧
      (layout st i o outs).rb = (layout st i' o' outs).rb := by
  simp only [layout, hi, ho, and_self]

/-- **`LaidRows` reads `layout`'s ports only through their sums.** -/
theorem laidRows_layout_congr (st : St) {i i' o o' : Array ℕ} (outs : Array Lin)
    (hi : i.foldl (· + ·) 0 = i'.foldl (· + ·) 0) (ho : o.foldl (· + ·) 0 = o'.foldl (· + ·) 0) (n : ℕ) (v : ℕ → Bool) :
    LaidRows (layout st i o outs) n v ↔ LaidRows (layout st i' o' outs) n v := by
  obtain ⟨h1, h2, h3⟩ := layout_congr st outs hi ho
  unfold LaidRows
  rw [h1, h2, h3]

theorem outStart_congr (st : St) {i i' o o' : Array ℕ} (hi : i.foldl (· + ·) 0 = i'.foldl (· + ·) 0)
    (ho : o.foldl (· + ·) 0 = o'.foldl (· + ·) 0) : outStart st i o = outStart st i' o' := by
  simp only [outStart, inRowsOf, spanOf, hi, ho]

theorem foldl_replicate16 : ∀ m a : ℕ, (List.replicate m 16).foldl (· + ·) a = a + 16 * m
  | 0, a => by simp
  | m + 1, a => by
    rw [List.replicate_succ, List.foldl_cons, foldl_replicate16 m]
    omega

theorem foldl_wordPorts (N : ℕ) : (Flock.RecOpen.wordPorts N).foldl (· + ·) 0 = 16 * (N / 16) := by
  rw [Flock.RecOpen.wordPorts, ← Array.foldl_toList, Array.toList_replicate, foldl_replicate16, Nat.zero_add]

/-- The verifier's unit's input bits. -/
abbrev unitIn (LANES H : ℕ) : ℕ := (Flock.RecOpen.ports LANES H).bits.foldl (· + ·) 0

theorem unitIn_eq (LANES H : ℕ) :
    unitIn LANES H = 128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + 1024 := by
  rw [unitIn, Flock.RecOpen.Ports.bits, ← Array.foldl_toList]
  show 0 + 128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + 1024 = _
  rw [Nat.zero_add]

/-- One 16-bit port per word has as many bits as the ports. -/
theorem foldl_unitIn (LANES H : ℕ) :
    (Flock.RecOpen.wordPorts (unitIn LANES H)).foldl (· + ·) 0 =
      (portsOf (Flock.RecOpen.ports LANES H)).bits.foldl (· + ·) 0 := by
  rw [foldl_wordPorts, foldl_bits, unitIn_eq]
  simp only [portsOf, Flock.RecOpen.ports]
  omega

/-! ## The outputs' length -/

theorem run_bind {α β : Type} (x : M α) (f : α → M β) (s : St) : (x >>= f).run s = (f (x.run s).1).run (x.run s).2 :=
  rfl

theorem run_pure {α : Type} (a : α) (s : St) : (pure a : M α).run s = (a, s) := rfl

theorem length_shaFrom (cv : List (BitVec 64)) (bits : List Lin) (pre : ℕ) (s : St) :
    ((shaFrom cv bits pre).run s).1.length = 512 := by
  simp only [shaFrom, run_bind, run_pure, List.length_map, List.length_range]

theorem length_climbFrom (d : List Lin) (ss : List (List Lin)) :
    ∀ h cur s, cur.length = 512 → ((climbFrom d h cur ss).run s).1.length = 512 := by
  induction ss with
  | nil => intro h cur s hc; exact hc
  | cons x rest ih =>
    intro h cur s _
    simp only [climbFrom, run_bind]
    exact ih _ _ _ (length_shaFrom _ _ _ _)

theorem length_climbRow (H : ℕ) (row aux d : List Lin) (s : St) : ((climbRow H row aux d).run s).1.length = 1024 := by
  simp only [climbRow, run_bind, run_pure, List.length_append, List.length_replicate]
  rw [length_climbFrom _ _ _ _ _ (length_shaFrom _ _ _ _)]

theorem length_reduceG_recomb (l : List Lin) : (reduceG (recomb 7 l)).length = 128 :=
  length_reduceG_255 _ (by simp [length_recomb])

theorem length_residual (S : GfS) (W V : List Lin) (r : Res) (s : St) :
    ((residual S W V r).run s).1.length = 128 := by
  simp only [residual]
  split
  · exact length_xorAll_terms W V r.add
  · simp only [run_bind, run_pure, xorL, List.length_zipWith, length_xorAll_terms, length_reduceG_recomb, min_self]

theorem length_residualsGo (S : GfS) (W V : List Lin) (rs : List Res) :
    ∀ s, ((residualsGo S W V rs).run s).1.length = 128 * rs.length := by
  induction rs with
  | nil => intro s; rfl
  | cons r rs ih =>
    intro s
    simp only [residualsGo, run_bind, run_pure, List.length_append, length_residual, ih, List.length_cons]
    ring

/-- **The opening's outputs are 2,048 bits** when `acc` is 1,024: the climb's 1,024, the consistency term's 256, the rest
of `acc`. -/
theorem length_recOpenOuts (LANES H : ℕ) (p : Ports) (hacc : p.acc = 1024) : (recOpenOuts LANES H p).length = 2048 := by
  simp only [recOpenOuts, recOpenNet_run, recOpen, run_bind, run_pure, List.length_append, List.length_drop,
    residualForms, length_residualsGo, length_climbRow, length_consStructure_res, length_vars, hacc]

/-! ## The verifier's unit -/

/-- **The verifier's laid-out opening computes the verifier of record's climb and the opening's consistency term**:
`laid_recOpen` at `RecOpen.unitLaid`, M0's ports one 16-bit port per word. Whenever its rows hold at the column values
`v`, if the `row` columns hold the bytes of the `LANES` elements `r`, the `aux` columns the 64-byte siblings `ss` one after
another (`64 max(1, ⌈H / 2⌉)` words), the `acc` columns two elements `a` and then 96 bytes `rest`, the `coef` columns
the `2 LANES` elements `c`, and direction column `h < H` bit `h` of the position `idx`, the output columns from
`outStart` hold `Flock.climb ms (SHA-512(rowBytes r)) idx ss` and 64 zero bytes, then the bytes of `consTerm` at
`k = 0, 1` and `rest`. -/
theorem laid_unit (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r)) (LANES H : ℕ)
    (hH : H ≤ 32) (v : ℕ → Bool)
    (h : LaidRows (Flock.RecOpen.unitLaid LANES H) (inRowsOf (Flock.RecOpen.wordPorts (unitIn LANES H))) v)
    (r a c : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hc : c.size = 2 * LANES) (hrest : rest.size = 96) (hss : ss.length = H)
    (hrow : ColsHold v 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v (128 * LANES + 512 * i) ss[i])
    (hacc : ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2)) (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024) (Flock.MerkleScheme.rowBytes c))
    (hdir : ∀ h < H, v (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) = idx.testBit h) :
    ColsHold v
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run St.empty).2
        (Flock.RecOpen.wordPorts (unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest)) := by
  have hi := foldl_unitIn LANES H
  have h' : LaidRows (recOpenLaid LANES H (portsOf (Flock.RecOpen.ports LANES H)) (Flock.RecOpen.wordPorts 2048))
      (inRowsOf (portsOf (Flock.RecOpen.ports LANES H)).bits) v := by
    simp only [Flock.RecOpen.unitLaid, recOpenNet_eq, recOpenOuts_eq] at h
    rw [inRowsOf, hi] at h
    exact (laidRows_layout_congr _ _ hi rfl _ _).1 h
  have hout : (recOpenOuts LANES H (portsOf (Flock.RecOpen.ports LANES H))).length ≤
      (Flock.RecOpen.wordPorts 2048).foldl (· + ·) 0 := by
    rw [length_recOpenOuts LANES H _ rfl, foldl_wordPorts]
  have hrb := size_rowBytes r
  have hab : (Flock.MerkleScheme.rowBytes a ++ rest).size = 128 := by
    rw [ByteArray.size_append, size_rowBytes, ha, hrest]
  have hcb := size_rowBytes c
  have hL := laid_recOpen ms hms LANES H hH (portsOf (Flock.RecOpen.ports LANES H)) (Flock.RecOpen.wordPorts 2048) v hout
    h' r a c rest idx ss hr ha hss (by simp only [portsOf, Flock.RecOpen.ports]; omega)
    (by simp only [portsOf, Flock.RecOpen.ports]; omega) (by simp only [portsOf, Flock.RecOpen.ports]; omega)
    (by simp only [portsOf, Flock.RecOpen.ports]; omega) (by simp only [portsOf, Flock.RecOpen.ports]; omega)
    hrow haux hacc hcoef hdir
  rwa [outStart_congr _ hi.symm rfl, ← recOpenNet_eq] at hL

theorem inWords_unit {LANES H : ℕ} {name : String} {net : Flock.Net}
    (hl : netOf name (Flock.RecOpen.unitLaid LANES H) = .ok net) :
    net.inWords * Flock.WORD = inRowsOf (Flock.RecOpen.wordPorts (unitIn LANES H)) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hl
  rw [hw]
  simp only [Flock.RecOpen.unitLaid, layout_inGroups]
  simp [inRowsOf]

/-- **A copy of a unit the verifier accepted as `rec-open/l<L>-h<H>`, in a unit of any partition, computes the climb
and the consistency term.** If `RecOpen.check` accepted `c` and its unit's name gives `L` and `H`, then `H ≤ 32` and
the unit is the verifier's net (`RecOpen.check_ok`). Let unit `u` be an instance of `k` stacked copies of the unit's rows
(`Rows.ofNet`), the transcript `X` carrying 1 at the stack's constant. Then at copy `j`'s column values (`copyVal`),
`laid_unit` holds. -/
theorem unit_check {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {c : Flock.Circuit} (hB : Built c.unit)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check c = .ok ()) {LANES H : ℕ} (hp : Flock.RecOpen.params c.unit.name = some (LANES, H))
    (hu : c.unit.unitLog ≤ 32)
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96) (hss : ss.length = H)
    (hrow : ColsHold (copyVal I X hj) 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold (copyVal I X hj) (128 * LANES + 512 * i) ss[i])
    (hacc : ColsHold (copyVal I X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal I X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, copyVal I X hj (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) =
      idx.testBit h) :
    ColsHold (copyVal I X hj)
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run St.empty).2
        (Flock.RecOpen.wordPorts (unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)) := by
  obtain ⟨hv, hl⟩ := Flock.RecOpen.check_ok hrec hp
  have hH : H ≤ 32 := by
    simp only [Flock.RecOpen.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
    exact hv.2
  obtain ⟨hs, h1⟩ := sat_copy I X hj
  have hL := laidRows_ofNet hB hl hu hs (h1.trans hone)
  rw [inWords_unit hl] at hL
  exact laid_unit ms hms LANES H hH _ hL r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

/-- **A copy of the unit of a circuit `HmRow.parse` accepted, in the circuit's program on Δ's keys, computes the climb
and the consistency term**, when the unit is named `rec-open/l<L>-h<H>`: `parse` ran `RecOpen.check`
(`ParseFacts.recNet`), and `unit_check` holds at `keyInst`, the transcript carrying 1 on unit `u`'s constant gate. -/
theorem keyProg_parse {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (hc : Flock.HmRow.parse tags bytes tables = .ok c) (hU : Built c.unit) (n : ℕ)
    (outs : Finset (Fin (keyGates hU n))) (X : Fin (keyGates hU n) → Bool) (u : Fin n) {j : ℕ}
    (hj : j < c.upv) (hone : X (Const.Prog.oneAt (Placed.keyProg c hU n) u) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    {LANES H : ℕ} (hp : Flock.RecOpen.params c.unit.name = some (LANES, H)) (hu : c.unit.unitLog ≤ 32)
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96) (hss : ss.length = H)
    (hrow : ColsHold (copyVal (keyInst hU n outs u) X hj) 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal (keyInst hU n outs u) X hj) (128 * LANES + 512 * i) ss[i])
    (hacc : ColsHold (copyVal (keyInst hU n outs u) X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal (keyInst hU n outs u) X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H,
      copyVal (keyInst hU n outs u) X hj (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) =
        idx.testBit h) :
    ColsHold (copyVal (keyInst hU n outs u) X hj)
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run St.empty).2
        (Flock.RecOpen.wordPorts (unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)) :=
  unit_check hU (keyInst hU n outs u) X hj ((congrArg X (keyInst_one hU n outs u)).trans hone) ms hms
    (parse_facts hc).recNet hp hu r a cf rest idx ss hr ha hcf hrest hss hrow haux hacc hcoef hdir

end FlockVBridge
