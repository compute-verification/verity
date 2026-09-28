import FlockSoundness.GateRows

/-!
# A unit's rows, checked against its gate circuit

What a template's verified lowering is made of, and the theorem each template instantiates.
`python -m verity_flock.lean_rows` writes a template's data (`FlockSoundness/<Template>/`): its gate
circuit (the IR's Boolean export) in chunks, its pinned rows by region, and a checkpoint at each cut,
with the kernel's checks: one per chunk (`GateRows.checkChunk`) and one for the small regions
(`regionsOK`). `sound_of_checks` turns the checks into `Sound`: the pinned rows compute the circuit.

Encodings, one literal each:
* a gate is `op + 3·(a + 2^17·b)` (op 0 AND, 1 XOR, 2 NOT with `b = 0`), an operand `0` or `1` (the
  constants) or `2 + wire` (`decGate`);
* a list of columns is its runs `(first, length)`, base-2^26 digits `first·8192 + length + 1`
  (`unpackRuns`).
-/

namespace FlockSoundness.UnitRows

open GateRows Audit

def decSrc (n : ℕ) : Src := if n = 0 then .zero else if n = 1 then .one else .wire (n - 2)

/-- A gate from its code `op + 3·(a + 2^17·b)`, operands coded `0`, `1` (constants) or `2 + wire`. -/
def decGate (c : ℕ) : Gate :=
  if c % 3 = 0 then .and (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else if c % 3 = 1 then .xor (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else .not (decSrc (c / 3 % 2 ^ 17))

/-- Columns from their code: runs `(first, length)`, base-2^26 digits `first·8192 + length + 1`. -/
def unpackRuns : ℕ → ℕ → List (ℕ × ℕ)
  | 0, _ => []
  | f + 1, n =>
    if n = 0 then [] else ((n % 2 ^ 26 - 1) / 8192, (n % 2 ^ 26 - 1) % 8192) :: unpackRuns f (n / 2 ^ 26)

/-- A row from its code: its two column lists. -/
def rowOf (c : ℕ × ℕ) : List ℕ × List ℕ := (runsList (unpackRuns 512 c.1), runsList (unpackRuns 512 c.2))

/-- A row's masks, from its code. -/
def rowMasks (c : ℕ × ℕ) : ℕ × ℕ := (runsMask (unpackRuns 512 c.1), runsMask (unpackRuns 512 c.2))

/-- A row's runs lie below column `n`. -/
def rowBounded (n : ℕ) (c : ℕ × ℕ) : Bool :=
  (unpackRuns 512 c.1 ++ unpackRuns 512 c.2).all fun r => r.1 + r.2 ≤ n

/-- A checkpoint from its data, each form's runs as a mask. -/
def decCP (d : ℕ × ℕ × List (ℕ × ℕ × Bool)) : CP :=
  ⟨d.1, d.2.1, d.2.2.map fun e => (e.1, (runsMask (unpackRuns 512 e.2.1), e.2.2))⟩

/-- The gate circuit from its data. -/
def circOf (nIn : ℕ) (gates outs : List ℕ) : Circ := ⟨nIn, gates.map decGate, outs.map decSrc⟩

/-- The position of `r` in `l`, if any. -/
def idx : List ℕ → ℕ → Option ℕ
  | [], _ => none
  | a :: l, r => if a = r then some 0 else (idx l r).map (· + 1)

/-- The layout from its data: each input's column, where the ANDs start, the output region's
length, and the outputs' columns. -/
def layOf (inCol : ℕ → ℕ) (andBase outLen : ℕ) (outCols : List ℕ) : Lay := ⟨inCol, andBase, outLen, idx outCols⟩

/-- Rows satisfied by column values `z`, reading each row's columns as a list. -/
def SatList (R : List (List ℕ × List ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ r p, R[r]? = some p → z r = (xorSum p.1 z && xorSum p.2 z)

/-- **A template's rows compute its gate circuit**: column values that satisfy the rows `R`, with the
constant (column `k`) at 1, carry on each output's column the circuit's output on the inputs' columns. -/
def Sound (R : List (List ℕ × List ℕ)) (C : Circ) (inCol : ℕ → ℕ) (outCols : List ℕ) (k : ℕ) : Prop :=
  ∀ z : ℕ → Bool, SatList R z → z k = true →
    ∀ o < C.outs.length, z (outCols.getD o 0) = (C.eval fun i => z (inCol i)).getD o false

/-- The output region's row at column `r` (`GateRows.outRow`) with the constant's column given. -/
def outRowK (C : Circ) (L : Lay) (k : ℕ) (S : St) (r : ℕ) : ℕ × ℕ :=
  match L.outAt r with
  | some o => (mask k (srcForm S.forms (C.outs.getD o .zero)), 2 ^ k)
  | none => (0, 0)

/-- **The small regions' check**, which the kernel evaluates per template: the input region, the
output region (from the last checkpoint `cpN`), the constant's row, the first checkpoint (the
inputs), and the outputs' columns. `nA` is the number of AND rows. -/
def regionsOK (C : Circ) (inCol : ℕ → ℕ) (outCols : List ℕ) (andBase outLen k nA : ℕ) (cp0 cpN : CP)
    (pinIn pinOut : List (ℕ × ℕ)) (pinConst : ℕ × ℕ) : Bool :=
  let L := layOf inCol andBase outLen outCols
  decide ((List.range andBase).map (inRow C L) = pinIn.map rowMasks) &&
    decide ((List.range outLen).map (fun i => outRowK C L k cpN.st (andBase + nA + i)) = pinOut.map rowMasks) &&
    decide (rowMasks pinConst = (2 ^ k, 2 ^ k)) && decide (k = andBase + nA + outLen) &&
    pinIn.all (rowBounded (k + 1)) && pinOut.all (rowBounded (k + 1)) && rowBounded (k + 1) pinConst &&
    decide (cp0.w = C.nIn) && decide (cp0.col = andBase) &&
    cp0.entries.all (fun e => decide (e.1 < C.nIn) && decide (e.2 = (2 ^ inCol e.1, false))) &&
    decide (C.nIn ≤ 2 ^ depth) && (List.range C.nIn).all (fun i => decide (inCol i < andBase)) &&
    ascending cpN.keys && cpN.keys.all (· < 2 ^ depth) && decide (cpN.w ≤ 2 ^ depth) &&
    C.outs.all (fun s => s.below cpN.w && s.okRead cpN.st.forms cpN.w) &&
    (List.range C.outs.length).all (fun o => decide (idx outCols (outCols.getD o 0) = some o) &&
      decide (andBase + nA ≤ outCols.getD o 0) && decide (outCols.getD o 0 < andBase + nA + outLen))

theorem outRowK_congr (C : Circ) (L : Lay) (S S' : St) (h : ∀ s ∈ C.outs, srcForm S.forms s = srcForm S'.forms s)
    (k r : ℕ) : outRowK C L k S r = outRowK C L k S' r := by
  unfold outRowK
  split
  · rename_i o _
    rw [List.getD_eq_getElem?_getD]
    rcases h' : C.outs[o]? with _ | s
    · rfl
    · simp only [Option.getD_some]; rw [h s (List.mem_of_getElem? h')]
  · rfl

theorem lt_of_mem_runsList {rs : List (ℕ × ℕ)} {n c : ℕ} (h : ∀ r ∈ rs, r.1 + r.2 ≤ n) (hc : c ∈ runsList rs) :
    c < n := by
  obtain ⟨r, hr, hcr⟩ := List.mem_flatMap.mp hc
  have := h r hr
  have := List.mem_range'_1.mp hcr
  omega

/-- One chunk's check on its codes: the chunk's check, and its rows' runs below the constant. -/
def chunkOK (k : ℕ) (cp : CP) (gs : List ℕ) (cp' : CP) (rows : List (ℕ × ℕ)) : Bool :=
  checkChunk k cp (gs.map decGate) cp' (rows.map rowMasks) && rows.all (rowBounded (k + 1))

/-- Every chunk checks, each from the checkpoint the previous one reached. -/
def ChunksOKc (k : ℕ) : CP → List (List ℕ × CP × List (ℕ × ℕ)) → Prop
  | _, [] => True
  | cp, (gs, cp', rows) :: rest => chunkOK k cp gs cp' rows = true ∧ ChunksOKc k cp' rest

theorem ChunksOKc.split {k : ℕ} : ∀ {cp : CP} {chs : List (List ℕ × CP × List (ℕ × ℕ))}, ChunksOKc k cp chs →
    ChunksOK k cp (chs.map fun c => (c.1.map decGate, c.2.1, c.2.2.map rowMasks)) ∧
      ∀ c ∈ chs, c.2.2.all (rowBounded (k + 1)) = true
  | _, [], _ => ⟨trivial, by simp⟩
  | _, (gs, cp', rows) :: rest, ⟨h, hr⟩ => by
    simp only [chunkOK, Bool.and_eq_true] at h
    obtain ⟨ih1, ih2⟩ := ChunksOKc.split hr
    refine ⟨⟨h.1, ih1⟩, fun c hc => ?_⟩
    rcases List.mem_cons.mp hc with rfl | hc
    · exact h.2
    · exact ih2 c hc

theorem steps_w_eq (gs : List Gate) : ∀ s : St, (s.steps gs).w = s.w + gs.length := by
  induction gs with
  | nil => intro s; rfl
  | cons g gs ih => intro s; show ((s.step g).steps gs).w = _; rw [ih, step_w]; simp; omega

/-- **The checks give the theorem.** The chunks' checks (`hchunks`, `hbounded`) and the small regions'
check (`hregions`) make the rows, read from their codes, compute the gate circuit. -/
theorem sound_of_checks (nIn : ℕ) (gateCodes outCodes : List ℕ) (inCol : ℕ → ℕ) (outCols : List ℕ) (andBase outLen k nA : ℕ)
    (pinIn pinOut : List (ℕ × ℕ)) (pinConst : ℕ × ℕ) (cp0 cpN : CP) (chs : List (List ℕ × CP × List (ℕ × ℕ)))
    (hgates : gateCodes = chs.flatMap (·.1)) (hchs : ChunksOKc k cp0 chs)
    (hN : lastCP cp0 (chs.map fun c => (c.1.map decGate, c.2.1, c.2.2.map rowMasks)) = cpN)
    (hnA : (chs.flatMap (·.2.2)).length = nA)
    (hregions : regionsOK (circOf nIn gateCodes outCodes) inCol outCols andBase outLen k nA cp0 cpN pinIn pinOut
      pinConst = true) :
    Sound ((pinIn ++ chs.flatMap (·.2.2) ++ pinOut ++ [pinConst]).map rowOf) (circOf nIn gateCodes outCodes)
      inCol outCols k := by
  set C := circOf nIn gateCodes outCodes with hCdef
  set L := layOf inCol andBase outLen outCols with hLdef
  set chsM := chs.map fun c => (c.1.map decGate, c.2.1, c.2.2.map rowMasks) with hchsM
  obtain ⟨hchunks, hbounded⟩ := hchs.split
  simp only [regionsOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hregions
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨ein, eout⟩, econst⟩, hk⟩, hbIn⟩, hbOut⟩, hbConst⟩, hw0⟩, hc0⟩, he0⟩, hnIn⟩, hinC⟩, hasc⟩, hkeys⟩, hwN⟩, houtsN⟩, hocols⟩ := hregions
  have ein' : (List.range L.andBase).map (inRow C L) = pinIn.map rowMasks := ein
  have eout2 : (List.range L.outLen).map (fun i => outRowK C L k cpN.st (L.andBase + nA + i)) =
      pinOut.map rowMasks := eout
  -- the gates, the start, and the chunks composed
  have hgM : C.gates = chsM.flatMap (·.1) := by
    simp only [C, circOf, hgates, chsM, List.flatMap_map, List.map_flatMap]
  have hgood0 : cp0.Good (start C L) := by
    refine ⟨hw0.symm, hc0.symm, fun e he => ?_⟩
    obtain ⟨h1, h2⟩ := he0 e he
    rw [h2]
    exact get_inputs L C.nIn e.1 h1 hnIn
  obtain ⟨hG, hwf, l, hl, hout⟩ := chunks_sound k chsM cp0 (start C L) hgood0 hchunks
  have hfinal : final C L = (start C L).steps (chsM.flatMap (·.1)) := by rw [final, hgM]
  have hG' : cpN.Good (final C L) := hN ▸ hfinal ▸ hG
  have hAnd : (final C L).rows.reverse.map (fun p => (mask k p.1, mask k p.2)) =
      (chs.flatMap (·.2.2)).map rowMasks := by
    rw [hfinal, hl]
    simpa [start, chsM, List.flatMap_map, List.map_flatMap] using hout
  have hnAnd : nAnd C = nA := by
    rw [← final_rows_length C L, ← List.length_reverse, ← List.length_map (f := fun p : Form × Form =>
      (mask k p.1, mask k p.2)), hAnd, List.length_map, hnA]
  have hcc : constCol C L = k := by simp [constCol, L, layOf, hnAnd, hk]
  -- the final state's width
  have hwfin : (final C L).w = C.nIn + C.gates.length := by
    rw [final, steps_w_eq]; rfl
  have hwN' : cpN.w = C.nIn + C.gates.length := by rw [← hG'.1, hwfin]
  -- the outputs' forms: the last checkpoint holds them
  have hkeys' : ∀ i ∈ cpN.keys, i < 2 ^ depth := fun i hi => by simpa using hkeys i hi
  have hstN := cpN.st_good (ascending_nodup hasc) hkeys'
  have hagree : ∀ s ∈ C.outs, srcForm (final C L).forms s = srcForm cpN.st.forms s := by
    intro s hs
    obtain ⟨hbl, hok⟩ := houtsN s hs
    cases s with
    | zero => simp only [srcForm]
    | one => simp only [srcForm]
    | wire i =>
      simp only [Src.below, decide_eq_true_eq] at hbl
      simp only [Src.okRead, Bool.or_eq_true, decide_eq_true_eq] at hok
      have hsome : (PM.get depth cpN.st.forms i).isSome = true := by
        rcases hok with h | h
        · omega
        · exact h
      have hi : i ∈ cpN.keys := cpN.mem_keys_of_get hkeys' (by omega) hsome
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hi
      simp only [srcForm, hG'.2.2 e he, hstN.2.2 e he]
  -- the rows
  have hrows : (pinIn ++ chs.flatMap (·.2.2) ++ pinOut ++ [pinConst]).map rowMasks = rows C L := by
    have eout' : (List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i)) =
        pinOut.map rowMasks := by
      rw [← eout2, hnAnd]
      refine List.map_congr_left fun i _ => ?_
      show outRowK C L (constCol C L) _ _ = _
      rw [hcc]; exact outRowK_congr C L _ _ hagree _ _
    show _ = (List.range L.andBase).map (inRow C L) ++
      (final C L).rows.reverse.map (fun p => (mask (constCol C L) p.1, mask (constCol C L) p.2)) ++
      (List.range L.outLen).map (fun i => outRow C L (final C L) (L.andBase + nAnd C + i)) ++
      [(2 ^ constCol C L, 2 ^ constCol C L)]
    rw [ein', eout', hcc, hAnd, ← econst]
    simp only [List.map_append, List.map_cons, List.map_nil]
  have hpin : ((pinIn ++ chs.flatMap (·.2.2) ++ pinOut ++ [pinConst]).map rowOf).map
      (fun p => (listMask p.1, listMask p.2)) = rows C L := by
    rw [← hrows, List.map_map]
    congr 1; funext c; simp [rowOf, rowMasks, listMask_runsList]
  -- every code's runs lie below the constant
  have hbAll : ∀ c ∈ pinIn ++ chs.flatMap (·.2.2) ++ pinOut ++ [pinConst], rowBounded (k + 1) c = true := by
    intro c hc
    simp only [List.mem_append, List.mem_flatMap, List.mem_singleton] at hc
    rcases hc with ((hc | ⟨ch, hch, hc⟩) | hc) | rfl
    · exact hbIn c hc
    · exact List.all_eq_true.mp (hbounded ch hch) c hc
    · exact hbOut c hc
    · exact hbConst
  intro z hz h1 o ho
  have hR : (rows C L).length = k + 1 := by rw [rows_length, hcc]
  have hsat : Sat (rows C L) z := by
    intro r p hp
    rw [← hpin, List.getElem?_map] at hp
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.mp hp
    obtain ⟨c, hc, rfl⟩ : ∃ c ∈ pinIn ++ chs.flatMap (·.2.2) ++ pinOut ++ [pinConst], rowOf c = q :=
      List.mem_map.mp (List.mem_of_getElem? hq)
    have hb := hbAll c hc
    simp only [rowBounded, List.all_eq_true, List.mem_append, decide_eq_true_eq] at hb
    have h1' : ∀ x ∈ (rowOf c).1, x < k + 1 := fun x hx => lt_of_mem_runsList (fun r hr => hb r (Or.inl hr)) hx
    have h2' : ∀ x ∈ (rowOf c).2, x < k + 1 := fun x hx => lt_of_mem_runsList (fun r hr => hb r (Or.inr hr)) hx
    rw [hR, hz r _ hq, xorSum_eq_psum z (k + 1) _ h1', xorSum_eq_psum z (k + 1) _ h2']
  obtain ⟨⟨hidx, hlo⟩, hhi⟩ := hocols o (List.mem_range.mpr ho)
  have hb2 : C.nIn + C.gates.length ≤ 2 ^ depth := by rw [← hwN']; exact hwN
  have houts' : ∀ s ∈ C.outs, s.below (C.nIn + C.gates.length) = true := fun s hs => by
    rw [← hwN']; exact (houtsN s hs).1
  have key := rows_sound C L (by rw [hgM]; exact hw0 ▸ hwf) hb2 (fun i hi => hinC i (List.mem_range.mpr hi))
    houts' z hsat (by rw [hcc]; exact h1) (outCols.getD o 0 - (andBase + nA)) (by show _ < outLen; omega) o
    (by show idx outCols (andBase + nAnd C + _) = some o; rw [hnAnd, Nat.add_sub_cancel' hlo]; exact hidx) ho
  have hol' : o < (C.eval fun i => z (inCol i)).length := by simpa [Circ.eval] using ho
  rw [List.getD_eq_getElem _ _ hol']
  have e : L.andBase + nAnd C + (outCols.getD o 0 - (andBase + nA)) = outCols.getD o 0 := by
    show andBase + nAnd C + _ = _; rw [hnAnd]; omega
  rw [e] at key
  exact key

end FlockSoundness.UnitRows
