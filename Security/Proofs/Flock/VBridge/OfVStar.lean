import Proofs.Flock.VBridge.Unit
import VStar.Unit

/-!
# The bridge over V*'s circuits as core code

`VStar` (`verity/core/service/vstar`) is the executable Lean that writes V*'s circuit files: each
`RecOpen{LANES, H}` statement and each `InnerRepCheck{S}` part, byte for byte as `verity_flock`'s builders write them.
This file proves that its builders are the ones this bridge reasons about, and restates the bridge's theorems over
`VStar`'s definitions, so that they are about the code that writes the circuit files and not about a copy of it.

* The gadgets. `VStar` reads its ports and operands from arrays where the bridge reads lists, and computes each operand
  once, before the residuals; the forms and the rows are the same. `residualForms_eq`, at an S whose products' operands
  are among its operands (`OpsInRange`, which `VStar.checkS` asks of every S), `climbRow_eq` and `recOpen_eq`.
* The unit. `VStar.recOpenUnit` takes one input of all the ports' words and cuts it into the five ports; it is
  `recOpenNet` at `ports LANES H` (`recOpenUnit_ok`). Its layout, ports of 16 bits, has `recOpenLaid`'s rows, constant and
  output columns (`laidRows_recOpenUnit`, `outStart_recOpenUnit`): `layout` reads the input ports only through their total.
* The theorems. `sound_residualForms`, `sound_climbRow`, `sound_recOpen` (rows hold, so the outputs are the verifier of
  record's), `laid_recOpenUnit` (the laid-out unit's rows hold, so its output columns are) and `unit_recOpenUnit` (a copy
  of the unit in a unit of any partition), each over `VStar`'s definitions.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

theorem getD_toArray {α : Type} (l : List α) (i : ℕ) (d : α) : l.toArray.getD i d = l.getD i d := by
  simp only [Array.getD, List.size_toArray, List.getD]
  split <;> simp_all

/-! ## SHA-512 and the climb -/

theorem lenBytes_eq : VStar.lenBytes = lenBytes := rfl
theorem zeroPad_eq : VStar.zeroPad = zeroPad := rfl
theorem padTail_eq : VStar.padTail = padTail := rfl
theorem bitsOf_eq : VStar.bitsOf = bitsOf := rfl
theorem shaMsg_eq : VStar.shaMsg = shaMsg := rfl
theorem iv_eq : VStar.iv = preCv [] 0 := rfl

theorem blockAt_eq (msg : List Lin) (off : ℕ) : VStar.blockAt msg.toArray off = blockAt msg off := by
  simp only [VStar.blockAt, blockAt, getD_toArray]

theorem shaBlock_eq : VStar.shaBlock = shaBlock := rfl

theorem shaBlocks_eq (msg : List Lin) : ∀ (n k : ℕ) (st : List Lin),
    VStar.shaBlocks msg.toArray n k st = shaBlocks msg n k st
  | 0, _, _ => rfl
  | n + 1, k, st => by
    simp only [VStar.shaBlocks, shaBlocks, blockAt_eq, shaBlock_eq]
    exact congrArg _ (funext fun st' => shaBlocks_eq msg n (k + 1) st')

theorem shaFrom_eq (cv : List (BitVec 64)) (bits : List Lin) (pre : ℕ) :
    VStar.shaFrom cv bits pre = shaFrom cv bits pre := by
  simp only [VStar.shaFrom, shaFrom, shaMsg_eq, List.size_toArray, shaBlocks_eq, getD_toArray]

theorem swap_eq : VStar.swap = swap := rfl

theorem climbFrom_eq (d : List Lin) : ∀ (h : ℕ) (cur : List Lin) (ss : List (List Lin)),
    VStar.climbFrom d h cur ss = climbFrom d h cur ss
  | _, _, [] => rfl
  | h, cur, s :: rest => by
    simp only [VStar.climbFrom, climbFrom, swap_eq, shaFrom_eq, iv_eq]
    exact congrArg _ (funext fun lr => congrArg _ (funext fun c => climbFrom_eq d (h + 1) c rest))

theorem sibForms_eq : VStar.sibForms = sibForms := rfl

/-- **V*'s climb is the bridge's.** -/
theorem climbRow_eq (H : ℕ) (row aux d : List Lin) : VStar.climbRow H row aux d = climbRow H row aux d := by
  simp only [VStar.climbRow, climbRow, shaFrom_eq, iv_eq, sibForms_eq, climbFrom_eq]

/-! ## The residuals -/

/-- `VStar`'s S read as the bridge's. -/
def term : VStar.Term → Term
  | .w i => .w i
  | .v i => .v i
  | .wt i u => .wt i u

def res (r : VStar.Res) : Res := ⟨r.mul, r.add.map term⟩

def gfS (S : VStar.GfS) : GfS := ⟨S.ops.map (·.map term), S.res.map res⟩

/-- Every product's operands are among `S.ops` (`VStar.checkS`'s last check). -/
def OpsInRange (S : VStar.GfS) (r : VStar.Res) : Prop := ∀ p ∈ r.mul, p.1 < S.ops.length ∧ p.2 < S.ops.length

theorem addL_eq : ∀ f g : List Lin, VStar.addL f g = addL f g
  | [], _ => rfl
  | _ :: _, [] => rfl
  | f :: fs, g :: gs => by simp only [VStar.addL, addL, addL_eq fs gs]

theorem shiftL_eq : VStar.shiftL = shiftL := rfl

theorem reduceG_eq (f : List Lin) : VStar.reduceG f = reduceG f := by
  simp only [VStar.reduceG, VStar.foldG, VStar.mulS, reduceG, foldG, mulS, addL_eq, shiftL_eq]

theorem termForms_eq (W V : List Lin) (t : VStar.Term) :
    VStar.termForms W.toArray V.toArray t = termForms W V (term t) := by
  cases t <;> simp only [VStar.termForms, termForms, term, getD_toArray]

theorem xorL_eq : VStar.xorL = xorL := rfl
theorem xorAll_eq : VStar.xorAll = xorAll := rfl

theorem map_termForms (W V : List Lin) (ts : List VStar.Term) :
    ts.map (VStar.termForms W.toArray V.toArray) = (ts.map term).map (termForms W V) := by
  rw [List.map_map]
  exact List.map_congr_left fun t _ => termForms_eq W V t

theorem leaves_eq : ∀ (n : ℕ) (a b : List Lin), VStar.leaves n a b = leaves n a b
  | 0, _, _ => rfl
  | n + 1, a, b => by simp only [VStar.leaves, leaves, leaves_eq n, addL_eq]

theorem recomb_eq : ∀ (n : ℕ) (l : List Lin), VStar.recomb n l = recomb n l
  | 0, _ => rfl
  | n + 1, l => by simp only [VStar.recomb, recomb, recomb_eq n, addL_eq, shiftL_eq]

theorem leafSums_eq : ∀ (ps : List (List Lin × List Lin)) (sums : List Lin), VStar.leafSums ps sums = leafSums ps sums
  | [], _ => rfl
  | p :: ps, sums => by
    simp only [VStar.leafSums, leafSums, leaves_eq, xorL_eq]
    exact congrArg _ (funext fun zs => leafSums_eq ps _)

theorem opForms_getD (S : VStar.GfS) (W V : List Lin) {k : ℕ} (hk : k < S.ops.length) :
    (VStar.opForms S W.toArray V.toArray).getD k [] = opForms (gfS S) W V k := by
  simp only [VStar.opForms, opForms, gfS, List.map_toArray, getD_toArray, List.getD, List.getElem?_map,
    List.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, map_termForms, xorAll_eq]

theorem residual_eq (S : VStar.GfS) (W V : List Lin) (r : VStar.Res) (hr : OpsInRange S r) :
    VStar.residual (VStar.opForms S W.toArray V.toArray) W.toArray V.toArray r = residual (gfS S) W V (res r) := by
  have hm : r.mul.map (fun ab => ((VStar.opForms S W.toArray V.toArray).getD ab.1 [],
      (VStar.opForms S W.toArray V.toArray).getD ab.2 [])) =
      r.mul.map fun ab => (opForms (gfS S) W V ab.1, opForms (gfS S) W V ab.2) :=
    List.map_congr_left fun p hp => by rw [opForms_getD S W V (hr p hp).1, opForms_getD S W V (hr p hp).2]
  simp only [VStar.residual, residual, res, map_termForms, xorAll_eq, xorL_eq, hm, leafSums_eq, recomb_eq, reduceG_eq]
  rfl

theorem residualsGo_eq (S : VStar.GfS) (W V : List Lin) : ∀ rs : List VStar.Res, (∀ r ∈ rs, OpsInRange S r) →
    VStar.residualsGo (VStar.opForms S W.toArray V.toArray) W.toArray V.toArray rs = residualsGo (gfS S) W V (rs.map res)
  | [], _ => rfl
  | r :: rs, h => by
    simp only [VStar.residualsGo, residualsGo, List.map_cons, residual_eq S W V r (h r List.mem_cons_self),
      residualsGo_eq S W V rs fun r' hr' => h r' (List.mem_cons_of_mem _ hr')]

/-- **V*'s residuals are the bridge's**, at an S whose products' operands are among its operands. -/
theorem residualForms_eq (S : VStar.GfS) (W V : List Lin) (hS : ∀ r ∈ S.res, OpsInRange S r) :
    VStar.residualForms S W.toArray V.toArray = residualForms (gfS S) W V := by
  simp only [VStar.residualForms, residualForms, residualsGo_eq S W V S.res hS, gfS]

theorem consStructure_eq (LANES : ℕ) : gfS (VStar.consStructure LANES) = consStructure LANES := by
  simp only [gfS, VStar.consStructure, consStructure, List.map_append, List.map_map]
  rfl

theorem consStructure_inRange (LANES : ℕ) :
    ∀ r ∈ (VStar.consStructure LANES).res, OpsInRange (VStar.consStructure LANES) r := by
  intro r hr p hp
  simp only [VStar.consStructure, List.mem_map, List.mem_range] at hr
  obtain ⟨k, hk, rfl⟩ := hr
  simp only [List.mem_map, List.mem_range] at hp
  obtain ⟨l, hl, rfl⟩ := hp
  simp only [VStar.consStructure, List.length_append, List.length_map, List.length_range]
  constructor
  · have : k * LANES ≤ 1 * LANES := Nat.mul_le_mul_right _ (by omega)
    omega
  · omega

/-- **V*'s opening is the bridge's.** -/
theorem recOpen_eq (LANES H : ℕ) (row aux acc coef dirs : List Lin) :
    VStar.recOpen LANES H row aux acc coef dirs = recOpen LANES H row aux acc coef dirs := by
  simp only [VStar.recOpen, recOpen, residualForms_eq _ _ _ (consStructure_inRange LANES), consStructure_eq,
    climbRow_eq]

/-! ## The gadgets' guarantees over `VStar` -/

/-- **V*'s residual gadget computes the residuals** (`FlockVBridge.sound_residualForms`), at an S whose products'
operands are among its operands. -/
theorem sound_residualForms (S : VStar.GfS) (W V : List Lin) (hS : ∀ r ∈ S.res, OpsInRange S r) :
    Sound (VStar.residualForms S W.toArray V.toArray) fun z out => out.length = 128 * S.res.length ∧
      ∀ w v : ℕ → F128, (∀ q, bit z W q = wbit w q) → (∀ q, bit z V q = wbit v q) →
        ∀ r (hr : r < S.res.length), ∀ k < 128,
          bit z out (128 * r + k) = bitOf (resVal (gfS S) (termVal w v) (res S.res[r])) k := by
  rw [residualForms_eq S W V hS]
  refine (FlockVBridge.sound_residualForms (gfS S) W V).mono fun z out h => ⟨?_, fun w v hW hV r hr k hk => ?_⟩
  · have hl := h.1
    simp only [gfS, List.length_map] at hl
    exact hl
  · have hr' : r < (gfS S).res.length := by simp only [gfS, List.length_map]; exact hr
    have h' := h.2 w v hW hV r hr' k hk
    simp only [gfS, List.getElem_map] at h'
    exact h'

/-- **V*'s opening climbs the row's leaf to the node `out` names** (`FlockVBridge.sound_climbRow`). -/
theorem sound_climbRow (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (H : ℕ)
    (row aux d : List Lin) :
    Sound (VStar.climbRow H row aux d) fun z out => ∀ (rb : ByteArray) (idx : ℕ) (ss : List ByteArray),
      Carries z row rb → List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms aux H) ss →
      (∀ h < H, ev z (d.getD h Lin.zero) = idx.testBit h) →
      Carries z out (Flock.climb ms (Sha512.hash rb) idx ss ++ ⟨Array.replicate 64 0⟩) := by
  rw [climbRow_eq, sibForms_eq]
  exact FlockVBridge.sound_climbRow ms hms H row aux d

/-- **V*'s opening computes the climb and the consistency term** (`FlockVBridge.sound_recOpen`). -/
theorem sound_recOpen (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (row aux acc coef dirs : List Lin) :
    Sound (VStar.recOpen LANES H row aux acc coef dirs) fun z o =>
      ∀ (r a c : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray),
        r.size = LANES → a.size = 2 →
        Carries z row (MerkleScheme.rowBytes r) → Carries z acc (MerkleScheme.rowBytes a ++ rest) →
        Carries z coef (MerkleScheme.rowBytes c) →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms aux H) ss →
        (∀ h < H, ev z (dirs.getD h Lin.zero) = idx.testBit h) →
        Carries z o.1 (Flock.climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ∧
        Carries z o.2 (MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest) := by
  rw [recOpen_eq, sibForms_eq]
  exact FlockVBridge.sound_recOpen ms hms LANES H hH row aux acc coef dirs

/-! ## V*'s `RecOpen` unit -/

/-- `RecOpen{LANES, H}`'s ports in bits as V*'s unit takes them (`VStar.recOpenPorts`, 16 bits a word). -/
def ports (LANES H : ℕ) : Ports :=
  ⟨16 * VStar.rowWords LANES, 16 * VStar.auxWords H, 16 * VStar.ACC_WORDS, 16 * VStar.coefWords LANES,
    16 * VStar.DIRS_WORDS⟩

theorem vars_take (k a b : ℕ) : (vars k (a + b)).take a = vars k a := by
  apply List.ext_getElem <;> simp [vars]

theorem vars_drop (k a b : ℕ) : (vars k (a + b)).drop a = vars (k + a) b := by
  apply List.ext_getElem
  · simp [vars]
  · intro i _ _
    simp [vars, Nat.add_assoc]

/-- Consecutive slices of `16 w` variables from `k`, one for each `w` of `ws`. -/
def varSlices : ℕ → List ℕ → List (List Lin)
  | _, [] => []
  | k, w :: ws => vars k (16 * w) :: varSlices (k + 16 * w) ws

theorem slices_vars : ∀ (ws : List ℕ) (k extra : ℕ), VStar.slices (vars k (16 * ws.sum + extra)) ws = varSlices k ws
  | [], _, _ => rfl
  | w :: ws, k, extra => by
    rw [show 16 * (w :: ws).sum + extra = 16 * w + (16 * ws.sum + extra) by simp only [List.sum_cons]; omega]
    simp only [VStar.slices, vars_take, vars_drop, varSlices, slices_vars ws]

theorem run_input_bind {β : Type} (n : ℕ) (f : List Lin → M β) :
    (StateT.bind (input n) f).run St.empty = (f (vars 0 n)).run ⟨n, []⟩ := by
  simp only [StateT.run, StateT.bind, input, vars, St.empty, Nat.zero_add]
  rfl

/-- **V*'s unit is the bridge's net**: its builder's state is `recOpenNet`'s at `ports LANES H`, its outputs
`recOpenOuts`, its input ports `VStar.recOpenPorts` and its output ports `out ‖ acc_out`. -/
theorem recOpenUnit_ok {LANES H : ℕ} {U : VStar.UnitC} (h : VStar.recOpenUnit LANES H = .ok U) :
    U.st = ((recOpenNet LANES H (ports LANES H)).run St.empty).2 ∧
      U.outs = (recOpenOuts LANES H (ports LANES H)).toArray ∧
      U.inPorts = VStar.recOpenPorts LANES H ∧ U.outPorts = [VStar.OUT_WORDS, VStar.ACC_WORDS] := by
  unfold VStar.recOpenUnit at h
  cases hc : VStar.checkRecOpen LANES H with
  | error e => simp [hc, bind, Except.bind] at h
  | ok _ =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hn : 16 * (VStar.recOpenPorts LANES H).foldl (· + ·) 0 = 16 * (VStar.recOpenPorts LANES H).sum + 0 := by
      simp [VStar.recOpenPorts]
      omega
    simp only [run_input_bind]
    rw [hn, slices_vars]
    simp only [VStar.recOpenPorts, varSlices, recOpen_eq]
    have hs : 16 * [VStar.rowWords LANES, VStar.auxWords H, VStar.ACC_WORDS, VStar.coefWords LANES,
        VStar.DIRS_WORDS].sum + 0 = (ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc +
        (ports LANES H).coef + (ports LANES H).dirs := by
      simp only [ports, List.sum_cons, List.sum_nil]
      omega
    rw [hs]
    simp only [recOpenOuts, recOpenNet_run, Nat.zero_add]
    refine ⟨?_, ?_, ?_, ?_⟩ <;> first | rfl | trivial

theorem checkRecOpen_ok {LANES H : ℕ} (h : VStar.checkRecOpen LANES H = .ok ()) : H ≤ 32 := by
  unfold VStar.checkRecOpen at h
  by_cases h1 : (LANES < 8 || LANES % 8 != 0) = true
  · simp only [h1, ↓reduceIte] at h
    contradiction
  · by_cases h2 : 32 < H
    · simp only [h1, h2, ↓reduceIte] at h
      contradiction
    · omega

theorem recOpenUnit_H {LANES H : ℕ} {U : VStar.UnitC} (h : VStar.recOpenUnit LANES H = .ok U) : H ≤ 32 := by
  unfold VStar.recOpenUnit at h
  cases hc : VStar.checkRecOpen LANES H with
  | error e => simp [hc, bind, Except.bind] at h
  | ok x => exact checkRecOpen_ok hc

/-- A unit's layout, when `VStar.UnitC.laid` gives one: `layout` with ports of 16 bits. -/
theorem laid_ok {U : VStar.UnitC} {l : Laid} (h : U.laid = .ok l) :
    l = layout U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16) U.outs := by
  unfold VStar.UnitC.laid at h
  by_cases h0 : (U.inWords == 0) = true
  · simp [h0] at h
    contradiction
  · cases hd : VStar.deadRow U.st U.outs with
    | some v =>
      simp [h0, hd] at h
      contradiction
    | none =>
      simp only [h0, hd, Bool.false_eq_true, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
      exact h.symm

theorem foldl_add_replicate (k : ℕ) : ∀ (n a : ℕ), (List.replicate n k).foldl (· + ·) a = a + n * k
  | 0, a => by simp
  | n + 1, a => by
    rw [List.replicate_succ, List.foldl_cons, foldl_add_replicate k n (a + k), Nat.succ_mul]
    omega

theorem foldl_replicate (n k : ℕ) : (Array.replicate n k).foldl (· + ·) 0 = n * k := by
  rw [← Array.foldl_toList, Array.toList_replicate, foldl_add_replicate, Nat.zero_add]

/-- `layout`'s rows and constant read the input ports only through their total. -/
theorem layout_rows (st : St) {b₁ b₂ : Array ℕ} (outBits : Array ℕ) (outs : Array Lin)
    (h : b₁.foldl (· + ·) 0 = b₂.foldl (· + ·) 0) :
    (layout st b₁ outBits outs).ra = (layout st b₂ outBits outs).ra ∧
      (layout st b₁ outBits outs).rb = (layout st b₂ outBits outs).rb ∧
      (layout st b₁ outBits outs).constPos = (layout st b₂ outBits outs).constPos := by
  simp only [layout, h, and_self]

theorem inWords_recOpenUnit {LANES H : ℕ} {U : VStar.UnitC} (hU : VStar.recOpenUnit LANES H = .ok U) :
    (Array.replicate U.inWords 16).foldl (· + ·) 0 = (ports LANES H).bits.foldl (· + ·) 0 := by
  rw [foldl_replicate, VStar.UnitC.inWords, (recOpenUnit_ok hU).2.2.1]
  simp only [Ports.bits, ports, VStar.recOpenPorts, List.foldl_cons, List.foldl_nil, ← Array.foldl_toList]
  omega

theorem outWords_recOpenUnit {LANES H : ℕ} {U : VStar.UnitC} (hU : VStar.recOpenUnit LANES H = .ok U) :
    (Array.replicate U.outWords 16).foldl (· + ·) 0 = 2048 := by
  rw [foldl_replicate, VStar.UnitC.outWords, (recOpenUnit_ok hU).2.2.2]
  rfl

/-- The rows of V*'s laid-out unit, where they hold, are `recOpenLaid`'s at `ports LANES H`. -/
theorem laidRows_recOpenUnit {LANES H : ℕ} {U : VStar.UnitC} {l : Laid} (hU : VStar.recOpenUnit LANES H = .ok U)
    (hl : U.laid = .ok l) {v : ℕ → Bool} (h : LaidRows l (inRowsOf (Array.replicate U.inWords 16)) v) :
    LaidRows (recOpenLaid LANES H (ports LANES H) (Array.replicate U.outWords 16)) (inRowsOf (ports LANES H).bits) v := by
  obtain ⟨hst, houts, -, -⟩ := recOpenUnit_ok hU
  have hsum := inWords_recOpenUnit hU
  obtain ⟨hra, hrb, hcp⟩ := layout_rows U.st (Array.replicate U.outWords 16) U.outs hsum
  rw [laid_ok hl] at h
  unfold recOpenLaid
  rw [← hst, ← houts]
  unfold LaidRows at h ⊢
  rw [← hra, ← hrb, ← hcp]
  simpa only [inRowsOf, hsum] using h

theorem outStart_recOpenUnit {LANES H : ℕ} {U : VStar.UnitC} (hU : VStar.recOpenUnit LANES H = .ok U) :
    outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16) =
      outStart ((recOpenNet LANES H (ports LANES H)).run St.empty).2 (ports LANES H).bits
        (Array.replicate U.outWords 16) := by
  rw [← (recOpenUnit_ok hU).1]
  simp only [outStart, inRowsOf, inWords_recOpenUnit hU]

/-- **V*'s laid-out `RecOpen` unit computes the climb and the consistency term** (`laid_recOpen`): whenever its rows hold
at `v`, if its input columns hold the row, `acc`, the coefficients, the siblings and the position's bits, its output
columns hold the verifier of record's climb and 64 zero bytes, then the consistency term at `k = 0, 1` and the rest of
`acc`. -/
theorem laid_recOpenUnit (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) {LANES H : ℕ}
    {U : VStar.UnitC} (hU : VStar.recOpenUnit LANES H = .ok U) {l : Laid} (hl : U.laid = .ok l) (v : ℕ → Bool)
    (hout : U.outs.size ≤ 2048) (h : LaidRows l (inRowsOf (Array.replicate U.inWords 16)) v)
    (r a c : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : (ports LANES H).row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ (ports LANES H).aux)
    (haccN : (ports LANES H).acc = 8 * (MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : (ports LANES H).coef = 8 * (MerkleScheme.rowBytes c).size) (hdirsN : H ≤ (ports LANES H).dirs)
    (hrow : ColsHold v 0 (MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold v ((ports LANES H).row + 512 * i) ss[i])
    (hacc : ColsHold v ((ports LANES H).row + (ports LANES H).aux) (MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold v ((ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc) (MerkleScheme.rowBytes c))
    (hdir : ∀ h < H, v ((ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc + (ports LANES H).coef + h) =
      idx.testBit h) :
    ColsHold v (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((Flock.climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest)) := by
  have hout' : (recOpenOuts LANES H (ports LANES H)).length ≤ (Array.replicate U.outWords 16).foldl (· + ·) 0 := by
    rw [outWords_recOpenUnit hU]
    have hs : U.outs.size = (recOpenOuts LANES H (ports LANES H)).length := by
      rw [(recOpenUnit_ok hU).2.1]
      rfl
    omega
  rw [outStart_recOpenUnit hU]
  exact laid_recOpen ms hms LANES H (recOpenUnit_H hU) (ports LANES H) (Array.replicate U.outWords 16) v hout'
    (laidRows_recOpenUnit hU hl h) r a c rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN hrow haux hacc hcoef hdir

theorem inWords_laid {U : VStar.UnitC} {l : Laid} (hl : U.laid = .ok l) {name : String} {net : Flock.Net}
    (hn : Flock.HmNets.netOf name l = .ok net) : net.inWords * Flock.WORD = inRowsOf (Array.replicate U.inWords 16) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hw, laid_ok hl, layout_inGroups]
  simp [inRowsOf]

section Unit

open FlockSoundness FlockSoundness.Audit

/-- **A copy of V*'s laid-out `RecOpen` unit, in a unit of any partition, computes the climb and the consistency term**
(`unit_recOpen`). Let the verifier build `net` from the layout of `VStar.recOpenUnit LANES H` (`netOf`), and let unit `u`
be an instance of `k` stacked copies of the net's rows, the transcript `X` carrying 1 at the stack's constant. Then at
copy `j`'s column values (`copyVal`), `laid_recOpenUnit` holds. -/
theorem unit_recOpenUnit {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) {LANES H : ℕ} {U : VStar.UnitC}
    (hU : VStar.recOpenUnit LANES H = .ok U) {l : Laid} (hl : U.laid = .ok l) {name : String}
    (hn : Flock.HmNets.netOf name l = .ok net) (hu : net.unitLog ≤ 32) (hout : U.outs.size ≤ 2048)
    (r a c : Array F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : (ports LANES H).row = 8 * (MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ (ports LANES H).aux)
    (haccN : (ports LANES H).acc = 8 * (MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : (ports LANES H).coef = 8 * (MerkleScheme.rowBytes c).size) (hdirsN : H ≤ (ports LANES H).dirs)
    (hrow : ColsHold (copyVal I X hj) 0 (MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ColsHold (copyVal I X hj) ((ports LANES H).row + 512 * i) ss[i])
    (hacc : ColsHold (copyVal I X hj) ((ports LANES H).row + (ports LANES H).aux) (MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal I X hj) ((ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc)
      (MerkleScheme.rowBytes c))
    (hdir : ∀ h < H, copyVal I X hj ((ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc +
      (ports LANES H).coef + h) = idx.testBit h) :
    ColsHold (copyVal I X hj) (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((Flock.climb ms (Sha512.hash (MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (MerkleScheme.rowBytes #[consTerm LANES r a c 0, consTerm LANES r a c 1] ++ rest)) := by
  obtain ⟨hs, h1⟩ := sat_copy I X hj
  have hL := laidRows_ofNet hB hn hu hs (h1.trans hone)
  rw [inWords_laid hl hn] at hL
  exact laid_recOpenUnit ms hms hU hl _ hout hL r a c rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN hrow haux
    hacc hcoef hdir

end Unit

end FlockVBridge.OfVStar
