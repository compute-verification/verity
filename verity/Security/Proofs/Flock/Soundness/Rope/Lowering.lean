import Proofs.Flock.Soundness.Rope.Chunks

/-!
# RoPE's rows are its Boolean export's rows

The pinned `rope-head/neox-bf16/pair` unit (`flock-ir-unit/v2`, sha256 `9cbdef19…`,
`verity_flock.templates.rope_head.PINS`) and its gate circuit, the IR's Boolean export
(`verity_flock.boolean_export`: 64 inputs, 5,828 AND, 9,747 XOR and 1,717 NOT gates, 32 outputs),
are generated data (`Rope/Pinned.lean`, `Rope/Gates.lean`). `backends/flock/tests/test_lean_rope.py`
regenerates both and checks them byte for byte.

* `pinned_eq`: the pinned rows are the rows `GateRows.rows` derives from the gate circuit, by a
  kernel computation (`pinnedRuns_eq`, `decide +kernel` on the masks of the rows' runs of columns).
* `rope_sound`: so every column assignment that satisfies the pinned rows, with the constant 1,
  carries on its 32 output columns the gate circuit's outputs on its 64 input columns
  (`GateRows.rows_sound`).
-/

namespace FlockSoundness.Rope

open GateRows Audit

theorem start_good : (decCP cpData0).Good (start circ lay) := by
  have h : ((decCP cpData0).entries.all fun e => decide (e.1 < 64) && decide (e.2 = (2 ^ e.1, false))) = true := by
    decide +kernel
  refine ⟨rfl, rfl, fun e he => ?_⟩
  have := List.all_eq_true.mp h e he
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  rw [this.2]
  exact get_inputs lay 64 e.1 this.1 (by decide)

theorem final_eq : final circ lay = (start circ lay).steps (chunks.flatMap (·.1)) := by rw [final, gates_eq]

/-- The chunks, composed: the final state is at the last checkpoint, and its AND rows are the chunks'. -/
theorem final_good : (decCP cpData18).Good (final circ lay) ∧ gatesWF circ.nIn circ.gates = true ∧
    (final circ lay).rows.reverse.map (fun p => (mask 6144 p.1, mask 6144 p.2)) = chunks.flatMap (·.2.2) := by
  obtain ⟨hG, hwf, l, hl, hout⟩ := chunks_sound 6144 chunks (decCP cpData0) (start circ lay) start_good chunksOK
  rw [lastCP_chunks] at hG
  refine ⟨final_eq ▸ hG, by rw [gates_eq]; exact hwf, ?_⟩
  rw [final_eq, hl]; simpa [start] using hout

theorem nAnd_circ : nAnd circ = 5828 := by
  have h : (chunks.flatMap (·.2.2)).length = 5828 := by decide +kernel
  rw [← final_rows_length circ lay, ← List.length_reverse, ← List.length_map (f := fun p : Form × Form =>
    (mask 6144 p.1, mask 6144 p.2)), final_good.2.2, h]

theorem outRowK_congr (S S' : St) (h : ∀ s ∈ circ.outs, srcForm S.forms s = srcForm S'.forms s) (k r : ℕ) :
    outRowK k S r = outRowK k S' r := by
  unfold outRowK
  split
  · rename_i o _
    rw [List.getD_eq_getElem?_getD]
    rcases h' : circ.outs[o]? with _ | s
    · rfl
    · simp only [Option.getD_some]; rw [h s (List.mem_of_getElem? h')]
  · rfl

/-- The small regions, by the kernel: the input region, the output region from the last
checkpoint, the constant's row, and the facts about the last checkpoint and the outputs. -/
theorem regions : (List.range lay.andBase).map (inRow circ lay) = pinIn.map pinMasks ∧
    (List.range lay.outLen).map (fun i => outRowK 6144 (decCP cpData18).st (lay.andBase + 5828 + i)) =
      pinOut.map pinMasks ∧ pinMasks pinConst = (2 ^ 6144, 2 ^ 6144) ∧
    pinIn.all runsBounded = true ∧ pinOut.all runsBounded = true ∧ runsBounded pinConst = true ∧
    (decCP cpData18).keys.Nodup ∧ (decCP cpData18).keys.all (· < 2 ^ depth) = true ∧
    (circ.outs.all fun s => match s with | .wire i => (decCP cpData18).keys.contains i | _ => true) = true := by
  decide +kernel

/-- The kernel's check, on the masks of the runs. -/
theorem pinnedRuns_eq : pinnedRuns.map (fun p => (runsMask p.1, runsMask p.2)) = rows circ lay := by
  obtain ⟨hG, -, hrows⟩ := final_good
  obtain ⟨ein, eout, econst, -, -, -, hnd, hkb, houts⟩ := regions
  have hk : constCol circ lay = 6144 := by simp [constCol, lay, nAnd_circ]
  have hst := (decCP cpData18).st_good hnd (fun i hi => by simpa using List.all_eq_true.mp hkb i hi)
  have hagree : ∀ s ∈ circ.outs, srcForm (final circ lay).forms s = srcForm (decCP cpData18).st.forms s := by
    intro s hs
    cases s with
    | zero => simp only [srcForm]
    | one => simp only [srcForm]
    | wire i =>
      have hi : i ∈ (decCP cpData18).keys := by simpa using List.all_eq_true.mp houts _ hs
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hi
      simp only [srcForm, hG.2.2 e he, hst.2.2 e he]
  have eout' : (List.range lay.outLen).map (fun i => outRow circ lay (final circ lay) (lay.andBase + nAnd circ + i)) =
      pinOut.map pinMasks := by
    rw [← eout, nAnd_circ]
    refine List.map_congr_left fun i _ => ?_
    show outRowK (constCol circ lay) _ _ = _
    rw [hk]; exact outRowK_congr _ _ hagree _ _
  have hpin : pinnedRuns.map (fun p => (runsMask p.1, runsMask p.2)) = pinnedCodes.map pinMasks := by
    simp only [pinnedRuns, List.map_map]; rfl
  rw [hpin, pinned_split]
  show _ = (List.range lay.andBase).map (inRow circ lay) ++
    (final circ lay).rows.reverse.map (fun p => (mask (constCol circ lay) p.1, mask (constCol circ lay) p.2)) ++
    (List.range lay.outLen).map (fun i => outRow circ lay (final circ lay) (lay.andBase + nAnd circ + i)) ++
    [(2 ^ constCol circ lay, 2 ^ constCol circ lay)]
  rw [ein, eout', hk, hrows, econst]

/-- **The pinned rows are the derived rows.** -/
theorem pinned_eq : pinned.map (fun p => (listMask p.1, listMask p.2)) = rows circ lay := by
  rw [← pinnedRuns_eq, pinned, List.map_map]
  congr 1; funext p; simp [listMask_runsList]

theorem facts : pinnedRuns.length = 6145 ∧
    (pinnedRuns.all fun p => (p.1 ++ p.2).all fun r => r.1 + r.2 ≤ 6145) = true ∧
    gatesWF circ.nIn circ.gates = true ∧ circ.nIn = 64 ∧ circ.gates.length = 17292 ∧ nAnd circ = 5828 ∧
    circ.outs.length = 32 ∧ (circ.outs.all fun o => o.below (64 + 17292)) = true := by
  obtain ⟨-, -, -, h1, h2, h3, -, -, -⟩ := regions
  have hb := pinnedCodes_bounded h1 h2 h3
  refine ⟨by rw [← List.length_map (f := fun p => (runsMask p.1, runsMask p.2)), pinnedRuns_eq, rows_length, constCol,
      nAnd_circ]; rfl, ?_, final_good.2.1, rfl, ?_, nAnd_circ, ?_, ?_⟩
  · simpa [pinnedRuns, runsBounded, List.all_map, Function.comp_def] using hb
  · rw [gates_eq]; decide +kernel
  · decide +kernel
  · decide +kernel

theorem lt_of_mem_runsList {rs : List (ℕ × ℕ)} {n c : ℕ} (h : ∀ r ∈ rs, r.1 + r.2 ≤ n) (hc : c ∈ runsList rs) :
    c < n := by
  obtain ⟨r, hr, hcr⟩ := List.mem_flatMap.mp hc
  have := h r hr
  have := List.mem_range'_1.mp hcr
  omega

theorem outAt_lay (o : ℕ) (ho : o < 32) : lay.outAt (lay.andBase + 5828 + (60 + o)) = some o := by
  have e1 : lay.andBase + 5828 + (60 + o) = 6016 + o := by show 128 + 5828 + (60 + o) = 6016 + o; omega
  have e2 : lay.outAt (6016 + o) = some (6016 + o - 6016) := ite_eq_left ⟨by omega, by omega⟩
  exact (congrArg lay.outAt e1).trans (e2.trans (congrArg some (by omega)))

/-- Rows satisfied by column values `z`, reading each row's columns as a list. -/
def SatList (R : List (List ℕ × List ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ r p, R[r]? = some p → z r = (xorSum p.1 z && xorSum p.2 z)

/-- **RoPE's pinned rows compute its Boolean export.** Column values that satisfy the pinned rows,
with the constant (column 6144) at 1, carry on output column `6016 + o` the gate circuit's output
`o` on input columns `0 … 63`. -/
theorem rope_sound (z : ℕ → Bool) (hz : SatList pinned z) (h1 : z 6144 = true) (o : ℕ) (ho : o < 32) :
    z (6016 + o) = (circ.eval z)[o]'(by simpa [Circ.eval, facts.2.2.2.2.2.2.1] using ho) := by
  obtain ⟨hlen, hcols, hwf, hin, hgl, hand, hol, hob⟩ := facts
  have hR : (rows circ lay).length = 6145 := by rw [← pinnedRuns_eq, List.length_map, hlen]
  have hsat : Sat (rows circ lay) z := by
    intro r p hp
    rw [← pinned_eq, List.getElem?_map] at hp
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.mp hp
    obtain ⟨u, hu, rfl⟩ : ∃ u ∈ pinnedRuns, (runsList u.1, runsList u.2) = q :=
      List.mem_map.mp (by rw [← pinned]; exact List.mem_of_getElem? hq)
    have hall := List.all_eq_true.mp hcols u hu
    simp only [List.all_eq_true, List.mem_append, decide_eq_true_eq] at hall
    have h1' : ∀ c ∈ runsList u.1, c < 6145 := fun c hc => lt_of_mem_runsList (fun r hr => hall r (Or.inl hr)) hc
    have h2' : ∀ c ∈ runsList u.2, c < 6145 := fun c hc => lt_of_mem_runsList (fun r hr => hall r (Or.inr hr)) hc
    rw [hR, hz r _ hq, xorSum_eq_psum z 6145 _ h1', xorSum_eq_psum z 6145 _ h2']
  have hc : constCol circ lay = 6144 := by simp [constCol, lay, hand]
  have houts : ∀ s ∈ circ.outs, s.below (circ.nIn + circ.gates.length) = true := by
    intro s hs; rw [hin, hgl]; exact List.all_eq_true.mp hob s hs
  have key := rows_sound circ lay hwf (by rw [hin, hgl]; decide) (fun k hk => by simp [lay]; omega) houts z hsat
    (by rw [hc]; exact h1) (60 + o) (by simp [lay]; omega) o
    (by rw [hand]; exact outAt_lay o ho) (by rw [hol]; exact ho)
  simp only [lay, hand, show 128 + 5828 + (60 + o) = 6016 + o by omega] at key
  exact key

end FlockSoundness.Rope
