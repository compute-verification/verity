import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Accept
import Proofs.Properties.NetworkScan.Check
import Proofs.Properties.NetworkScan.Lowered
import Proofs.Properties.NetworkScan.Query
import Proofs.Properties.NetworkScan.OfProg
import Proofs.Properties.NetworkScan.Passes
import Proofs.Properties.NetworkScan.Reads
import Proofs.Properties.NetworkPeriod.Keys
import Proofs.Flock.SamplingScope

/-!
# Toward an accepted scan's fit, at Bits' positions

`ScanFits` and `ScanScopeAcc` are stated at Bits' positions (`Properties.NetworkScan`): the statement's circuit is the
scan's audit's members' joint program (`jointOf`), its layout read through `SampledProofs.Bits.inBit` and `outBit`,
with no wires. What `scanProg`'s acceptance gives of the Program is `Fits.scanProg_accepted` (`Accept`), what Bits'
check gives of a circuit file is `Check.facts`, what it gives of a unit's reads of its template's first parameter is
`Reads`', and what the joint program holds at each position, its inputs, outputs and reads, is `Lowered`'s.

The layout's causality and wiring (`NetTiming.BoundaryGates.Fits`'s last two parts):

* `not_depends_instPos`: a placed instance's gate depends on no gate of an instance at another template node, since
  dependence never leaves a unit and the code locates a unit at one node.
* `egress_unitPos`, `ingress_unitPos`: the layout's egress gates are positions of the egress template's instance's
  unit, and a later step's ingress gates of the later ingress template's.
* `scan_causal`: where the egress gates are returned, no egress gate depends on a later step's ingress gate: a later
  step exists only with three templates, and its ingress is the middle one's.
* `depends_of_dependsWired`, `wiresFit_none`: with no wires, dependence is through reads and the wiring fits.

The layout's egress and ingress gates (its first two parts), where no two of the Program's Definitions share a key
(what `keysUnique` checks, as the keys' `Nodup`):

* `member_instPos`: a template's batch member, the template no input's key (`Fits.Accepted.inputs`), is a unit of a
  member over the template's circuit file (`MembersOf`, the template nodes having no repeats), and its position is that
  unit's.
* `egress_returned`: an egress instance's bit of a word of its return is a returned gate of the joint program, and not
  an input: the member's circuit file passed Bits' check (`FilesOf`, `Check.facts`), so the bit's leaf is a computed
  row's copy (`Lowered.outGate_isSome`, `Lowered.outBit_returned`).
* `scan_egress`: so is every egress gate: the presence gate is bit 0 of the verdict's word, and each frame bit a bit of
  a word of the row (`wordBit_zero_fst`, `wordBit_one_fst`), both inside the return's words.
* `ingress_input`: an ingress instance's bit of its `row` parameter is an input of the joint program: Bits' check makes
  the bit's port bit a message bit (`Reads.paramsWhy_first`) and keeps the unit's input ports in column order
  (`Check.Facts.ordered`, `Reads.srcKey_port`), so the bit has a key and a gate (`Lowered.inGate_isSome`), and the unit
  reads it (`RowsRead`, `Reads.readsWhy_first`, `Reads.unitReads_true`), so no constant fixes it
  (`Lowered.inBit_input`).
* `scan_ingress`: so is every ingress gate: step `t`'s are the bits of field 0 of its bucket's instance's parameters
  (`wordBit_zero_fst`, `wordBit_zero_snd`).
* `scan_fits`: an accepted scan's layout fits its audit's joint program: its egress and ingress gates as above, causal
  (`scan_causal`), and with no wires (`wiresFit_none`).
* `scanFits`: `ScanFits`, the keys distinct by `scanProg`'s own check (`Properties.NetworkPeriod.keysUnique_nodup`).

The facts both statements' witnesses read:

* `keysUnique_of_nodup`, `orderViolation_none` (`Passes`): `scanProg`'s two checks over hash sets pass a Program
  whose keys are distinct and whose composites' nodes read only parameters and earlier nodes.
* `not_fits_of_no_outs`: a circuit that returns nothing fits no layout over a horizon of at least one step.
* `jointOf_nil`, `not_fits_of_no_members`: an audit of no members has the empty joint program, so no layout fits it.
* `bareProg`, `scanFits_nontrivial`: a Program of no templates, which the template query over no templates accepts
  with no ranges, so at a scan check that accepts every Program, `ScanFitsOf` fails.
* `oneSlot`, `canon_null`, `bytes_of_data_toList`: a schedule of one slot and one step, C-Flock's canonical `null`, and
  a byte array through its list of bytes.
* `nullStmt`, `acceptAllDeploy`: the statement whose descriptor is `null`'s bytes and whose circuit is empty, and a
  deployment whose scope check takes everything and which parses every message to it.
-/

namespace Properties.NetworkScan

open NetworkCheck.Period (Schedule templateNodes)
open NetworkCheck.Circuit (templateDef)
open SampledProofs.Bits (Members outBit keyProg posOf base)

/-! ## Causality and wiring -/

/-- **A placed instance's gate depends on no gate of another node's instance**: where template `i`'s instance `t` is
placed inside the members' program at a unit position (`pos`), and template `i'`'s instance `t'` at a unit position or
past the program, at another root node, the first's gate doesn't depend on the second's. -/
theorem not_depends_instPos {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members}
    {i t i' t' : ℕ} {pos pos' : Flock.Circuit → ℕ → ℕ → Option ℕ}
    (hpos : ∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x)
    (hpos' : ∀ c n u x, pos' c n u = some x → Lowered.UnitPos c n u x)
    (hk : (templateNodes P).getD i 0 ≠ (templateNodes P).getD i' 0)
    (hg : instPos P rs ms i t pos < (jointOf ms).gates.length) :
    ¬ NetTiming.Depends (jointOf ms) (instPos P rs ms i' t' pos') (instPos P rs ms i t pos) := by
  intro hd
  obtain ⟨m, hm, u, -, hl, x, hx, he⟩ := Query.instPos_lt (Nat.ne_of_lt hg)
  have hgu : Lowered.JointUnit ms m u (instPos P rs ms i t pos) := ⟨hm, x, he, hpos _ _ _ _ hx⟩
  by_cases hi : instPos P rs ms i' t' pos' = (jointOf ms).gates.length
  · rw [hi] at hd
    have := Lowered.eq_of_depends_of_le hd (Nat.le_refl _)
    omega
  · obtain ⟨m', hm', u', -, hl', x', hx', he'⟩ := Query.instPos_lt hi
    have hiu : Lowered.JointUnit ms m' u' (instPos P rs ms i' t' pos') := ⟨hm', x', he', hpos' _ _ _ _ hx'⟩
    obtain ⟨rfl, rfl⟩ := Lowered.jointUnit_unique (Lowered.depends_unit hd hiu) hgu
    exact hk (Query.locate_node_unique hl hl')

/-- **An egress gate is a position of the egress template's instance `t`'s unit.** -/
theorem egress_unitPos {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {t : ℕ} {ℓ : Fin 1} {σ : Fin s.allocation} {g : ℕ}
    (h : g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ)) :
    ∃ pos : Flock.Circuit → ℕ → ℕ → Option ℕ, (∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x) ∧
      g = instPos P rs ms ((templateNodes P).length - 1) t pos := by
  simp only [scanLayout, NetTiming.slotGates, List.mem_cons, List.mem_map, List.mem_range] at h
  rcases h with rfl | ⟨b, -, rfl⟩
  all_goals
    refine ⟨_, ?_, rfl⟩
    intro c n u x hx
    exact Lowered.outBit_unitPos hx

/-- **Step `t`'s ingress gate, past step 0, is a position of the later ingress template's instance `t − 1`'s unit.** -/
theorem ingress_unitPos {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {t i : ℕ} (ht : t ≠ 0) (h : i ∈ (scanLayout P s rs ms).ingress t) :
    ∃ pos : Flock.Circuit → ℕ → ℕ → Option ℕ, (∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x) ∧
      i = instPos P rs ms 1 (t - 1) pos := by
  simp only [scanLayout, ht, ↓reduceIte, List.mem_map, List.mem_range] at h
  obtain ⟨ℓ, -, rfl⟩ := h
  refine ⟨_, ?_, rfl⟩
  intro c n u x hx
  exact Lowered.inBit_unitPos hx

theorem templateNodes_nodup (P : Flock.Program.Prog) : (templateNodes P).Nodup := by
  unfold templateNodes
  exact List.Pairwise.filter _ List.nodup_range

/-- With no wires, dependence through reads and wires is dependence through reads. -/
theorem depends_of_dependsWired {C : Claims.Sampling.Prog} {i g : ℕ}
    (h : NetTiming.DependsWired C (fun _ => none) i g) : NetTiming.Depends C i g := by
  unfold NetTiming.DependsWired at h
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.tail ih (Or.resolve_right hbc (by simp))

/-- With no wires, the wiring fits. -/
theorem wiresFit_none {L S : ℕ} {lay : NetTiming.BoundaryGates L S} {C : Claims.Sampling.Prog} {k : ℕ}
    (hw : lay.wires = fun _ => none) : lay.WiresFit C k := by
  refine ⟨fun g w h => ?_, ⟨fun _ => 0, fun g w g' h => ?_⟩⟩ <;> simp [hw] at h

/-- **The scan's layout is causal**: of an accepted scan, where its egress gates are returned, no egress gate of a step
depends on an ingress gate of a later step. A later step exists only where the Program has three templates; its
ingress is the middle template's instance, and every egress gate the last template's. -/
theorem scan_causal {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange} {ms : Members}
    (hacc : Fits.Accepted P s)
    (hout : ∀ t < s.window, ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ),
      g ∈ (jointOf ms).outs) :
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s rs ms).ingress t',
        ¬ NetTiming.DependsWired (jointOf ms) (scanLayout P s rs ms).wires i g := by
  intro t ht t' ht' htt' ℓ σ g hg i hi hd
  have hgl := Lowered.lt_length_of_mem_outs (hout t ht ℓ σ g hg)
  obtain ⟨pos, hpos, rfl⟩ := egress_unitPos hg
  obtain ⟨pos', hpos', rfl⟩ := ingress_unitPos (by omega) hi
  rcases hacc.shape with ⟨a, e, -, hT⟩ | ⟨a, b, e, hts⟩
  · omega
  · have hbe : b ≠ e := by
      intro hbe
      subst hbe
      have hnd := templateNodes_nodup P
      rw [hts] at hnd
      simp [List.nodup_cons] at hnd
    have hk : (templateNodes P).getD ((templateNodes P).length - 1) 0 ≠ (templateNodes P).getD 1 0 := by
      rw [hts]
      simpa using hbe.symm
    exact not_depends_instPos hpos hpos' hk hgl (depends_of_dependsWired hd)

/-! ## The egress gates -/

theorem egressNode_eq (P : Flock.Program.Prog) :
    Fits.egressNode P = (templateNodes P).getD ((templateNodes P).length - 1) 0 := by
  simp only [Fits.egressNode, List.getLast?_eq_getElem?, List.getD_eq_getElem?_getD]

/-- Where no two of `P`'s Definitions share a key, `find?` at a Definition's key finds it. -/
theorem find?_key {P : Flock.Program.Prog} (hk : (P.defs.toList.map (·.key)).Nodup) {d : Flock.Program.Defn}
    (hd : d ∈ P.defs) : P.defs.find? (·.key == d.key) = some d := by
  cases hf : P.defs.find? (·.key == d.key) with
  | none =>
    have := Array.find?_eq_none.1 hf d hd
    simp at this
  | some d' =>
    have hp := Array.find?_some hf
    simp only [beq_iff_eq] at hp
    rw [List.inj_on_of_nodup_map hk (Array.mem_def.1 (Array.mem_of_find?_eq_some hf)) (Array.mem_def.1 hd) hp]

theorem templateDef_mem {P : Flock.Program.Prog} {k : ℕ} (hk : k ∈ templateNodes P) : templateDef P k ∈ P.defs := by
  have hfn := Query.templateNode_fn hk
  show P.defs[(Fits.node P k).fn]! ∈ P.defs
  rewrite [getElem!_pos P.defs _ hfn]
  exact Array.getElem_mem hfn

theorem getD_inj {l : List ℕ} (hnd : l.Nodup) {i j : ℕ} (hi : i < l.length) (hj : j < l.length)
    (h : l.getD i 0 = l.getD j 0) : i = j := by
  rw [Query.list_getD_of_lt _ hi, Query.list_getD_of_lt _ hj] at h
  exact (List.Nodup.getElem_inj_iff hnd).1 h

/-- A unit located at node `k` is at the node its location names. -/
theorem locate_node_of_map {rs : Array Flock.TemplateQuery.TRange} {v k k' t : ℕ}
    (h : ((Flock.TemplateQuery.locate rs v).any fun l => l.1 == k && l.2.1 == t) = true)
    (h' : (Flock.TemplateQuery.locate rs v).map (·.1) = some k') : k = k' := by
  rcases hl : Flock.TemplateQuery.locate rs v with _ | l
  · rw [hl] at h; simp at h
  · rw [hl] at h h'
    simp only [Option.any_some, Bool.and_eq_true, beq_iff_eq] at h
    simp only [Option.map_some, Option.some.injEq] at h'
    exact h.1.symm.trans h'

/-- **A template's batch member is a member's unit**: of a Program whose Definitions' keys are distinct, at its
templates' circuit files and its audit's members, batch member `t` of template `i`, no input's key, is unit `u` of a
member `m` over template `i`'s circuit file, and its position at `pos` is the member's at that unit. -/
theorem member_instPos {P : Flock.Program.Prog} {fs : List Flock.Circuit} {tids : Array String}
    {rs : Array Flock.TemplateQuery.TRange} {ms : Members} (hroot : P.root < P.defs.size)
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hfs : FilesOf P fs)
    (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hms : MembersOf P fs rs ms) {i t : ℕ}
    (hi : i < (templateNodes P).length)
    (hin : Flock.Program.isInputId (Fits.callee P ((templateNodes P).getD i 0)).key = false)
    (hb : (Fits.node P ((templateNodes P).getD i 0)).form = .batch)
    (ht : t < (Fits.node P ((templateNodes P).getD i 0)).n) (pos : Flock.Circuit → ℕ → ℕ → Option ℕ) :
    ∃ m < ms.length, ∃ u < (ms.getD m default).2.size, (ms.getD m default).1 = fs.getD i default ∧
      instPos P rs ms i t pos =
        ((pos (ms.getD m default).1 (ms.getD m default).2.size u).map (base ms m + ·)).getD (jointOf ms).gates.length := by
  have hkm : (templateNodes P).getD i 0 ∈ templateNodes P := by
    rw [Query.list_getD_of_lt _ hi]
    exact List.getElem_mem hi
  have hklt : (templateNodes P).getD i 0 < (Query.ddefOf (P.defs[P.root]!)).nodes.size := by
    simp only [Query.ddefOf, Array.size_map]
    exact Query.templateNode_lt hkm
  have hinp := Query.isInput_ofProg hkm hin
  obtain ⟨r, -, -, -, hcount, hpop, hloc⟩ :=
    Query.evaluate_range hev (Query.def?_ofProg_root hkeys hroot) hklt hinp
  have hcnt : r.count = (Fits.node P ((templateNodes P).getD i 0)).n := by
    rw [hcount]; exact Query.countAt_ofProg (Query.templateNode_lt hkm) hb
  have ht' : t < r.count := by rw [hcnt]; exact ht
  obtain ⟨m, hm, u, hu, huv, hpos⟩ := Query.instPos_eq (P := P) (i := i) (t := t) (pos := pos) hms.1 (by omega)
    fun w => hloc w t ht'
  obtain ⟨i', hi', hci, hloci⟩ := hms.2 m hm
  have hvmem : (ms.getD m default).2.getD u 0 ∈ (ms.getD m default).2.toList := by
    rw [Query.array_getD_of_lt _ hu]
    exact Array.getElem_mem_toList hu
  have hnode := locate_node_of_map ((hloc _ t ht').2 huv) (hloci _ hvmem)
  have hie : i = i' := getD_inj (templateNodes_nodup P) hi (by rw [← hfs.1]; exact hi') hnode
  subst hie
  exact ⟨m, hm, u, hu, hci, hpos⟩

theorem headD_getD (l : List ℕ) : l.headD 0 = l.getD 0 0 := by cases l <;> rfl

theorem wordBit_zero_fst (ts : List Flock.Program.Ty) (β : ℕ) : (wordBit ts 0 β).1 = β / 16 := by
  simp only [wordBit, List.take_zero, List.map_nil, List.sum_nil, Nat.mul_zero, Nat.zero_add]

theorem wordBit_zero_snd (ts : List Flock.Program.Ty) (β : ℕ) : (wordBit ts 0 β).2 = β % 16 := by
  simp only [wordBit, List.take_zero, List.map_nil, List.sum_nil, Nat.mul_zero, Nat.zero_add]

theorem wordBit_snd (ts : List Flock.Program.Ty) (π β : ℕ) : (wordBit ts π β).2 < 16 := by
  simp only [wordBit]
  omega

theorem wordBit_one_fst (a : Flock.Program.Ty) (ts : List Flock.Program.Ty) (β : ℕ) :
    (wordBit (a :: ts) 1 β).1 = (16 * ((SampledProofs.Bits.bitsOf a + 15) / 16) + β) / 16 := by
  have h1 : (a :: ts).take 1 = [a] := rfl
  simp only [wordBit, h1, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

/-- **An egress instance's bit of a word of its return is returned, and not an input,** in the members' program: of an
accepted scan whose Definitions' keys are distinct, at its templates' circuit files and its audit's members, bit `τ`
of word `j` of the egress template's instance `t`'s return. The instance is a unit of a member over the last
template's circuit file (`member_instPos`), which Bits' check passed, so the bit is a computed row's copy. -/
theorem egress_returned {P : Flock.Program.Prog} {s : Schedule} {fs : List Flock.Circuit} {tids : Array String}
    {rs : Array Flock.TemplateQuery.TRange} {ms : Members} (hacc : Fits.Accepted P s)
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hfs : FilesOf P fs)
    (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hms : MembersOf P fs rs ms) {t j τ : ℕ}
    (ht : t < s.window) (hj : j < SampledProofs.Bits.retWords (Fits.callee P (Fits.egressNode P)).ret)
    (hτ : τ < 16) :
    instPos P rs ms ((templateNodes P).length - 1) t (fun c n u => outBit c n u 0 j τ) ∈ (jointOf ms).outs ∧
      (jointOf ms).gates[instPos P rs ms ((templateNodes P).length - 1) t (fun c n u => outBit c n u 0 j τ)]? ≠
        some .input := by
  have he := egressNode_eq P
  have hkm : (templateNodes P).getD ((templateNodes P).length - 1) 0 ∈ templateNodes P := by
    rewrite [← he]
    exact hacc.egress_mem
  have hlen : (templateNodes P).length - 1 < (templateNodes P).length := by
    have := List.length_pos_of_mem hkm
    omega
  obtain ⟨m, hm, u, hu, hci, hpos⟩ := member_instPos (t := t) hacc.root_lt hkeys hfs hev hms hlen
    (by rewrite [← he]; exact hacc.inputs _ hacc.egress_mem) (by rewrite [← he]; exact hacc.egress_batch.1)
    (by rewrite [← he, hacc.egress_batch.2]; exact ht) (fun c n u => outBit c n u 0 j τ)
  have hF := Check.facts (hfs.2 _ (by rewrite [hfs.1]; exact hlen))
  rewrite [← hci] at hF
  have hupv := hF.upv
  have hleaves := hF.leaves
  have hfind := find?_key hkeys (templateDef_mem hkm)
  have h0 : 0 < (ms.getD m default).1.leavesOut.size := by omega
  have hjs : j < ((ms.getD m default).1.leavesOut.getD 0 #[]).size := by
    rewrite [hF.ret _ hfind ((ms.getD m default).1.leavesOut.getD 0 #[])
      (by rewrite [Query.array_getD_of_lt _ h0]; exact Array.getElem_mem h0), ← he]
    exact hj
  obtain ⟨hr1, hr2⟩ := hF.outRows 0 (by omega) j hjs τ hτ
  obtain ⟨g, hg⟩ := Lowered.outGate_isSome hu (by omega) hjs hτ hr1 hr2
  have hx : outBit (ms.getD m default).1 (ms.getD m default).2.size u 0 j τ =
      some (posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size) g) := by
    simp only [outBit, hg, Option.map_some]
  have hpos' : instPos P rs ms ((templateNodes P).length - 1) t (fun c n u => outBit c n u 0 j τ) =
      base ms m + posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size) g := by
    rewrite [hpos]
    simp only [hx, Option.map_some, Option.getD_some]
  rewrite [hpos']
  exact Lowered.outBit_returned ms hm hx

/-- **The scan's egress gates are returned, and not inputs** (`NetTiming.BoundaryGates.Fits`'s first part): of an
accepted scan whose Definitions' keys are distinct, at its templates' circuit files and its audit's members. A slot's
presence gate is bit 0 of word 0 of the egress instance's return, the verdict's, and each of its frame bits a bit of a
word of the row's, both inside the return's `1 + ⌈8SB/16⌉` words. -/
theorem scan_egress {P : Flock.Program.Prog} {s : Schedule} {fs : List Flock.Circuit} {tids : Array String}
    {rs : Array Flock.TemplateQuery.TRange} {ms : Members} (hacc : Fits.Accepted P s)
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hfs : FilesOf P fs)
    (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hms : MembersOf P fs rs ms) :
    ∀ t < s.window, ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ),
      g ∈ (jointOf ms).outs ∧ (jointOf ms).gates[g]? ≠ some .input := by
  have hrets : SampledProofs.Bits.fieldsOf
      (templateDef P ((templateNodes P).getD ((templateNodes P).length - 1) 0)).ret =
      [.array NetworkCheck.Period.outBits (.value 1), Fits.rowTy s] := by
    rewrite [← egressNode_eq P]
    exact (congrArg SampledProofs.Bits.fieldsOf hacc.egress_ret).trans rfl
  have hA : SampledProofs.Bits.bitsOf (.array NetworkCheck.Period.outBits (.value 1)) = 16 := by
    simp only [Check.bits_bitsOf, Check.bits_array, Check.bits_value, NetworkCheck.Period.outBits, Nat.mul_one]
  have hR : SampledProofs.Bits.bitsOf (Fits.rowTy s) = s.allocation * s.cellBytes * 8 := by
    simp only [Check.bits_bitsOf, Fits.rowTy, Check.bits_array, Check.bits_value, Nat.mul_one]
  have hw : SampledProofs.Bits.retWords (Fits.callee P (Fits.egressNode P)).ret =
      1 + (s.allocation * s.cellBytes * 8 + 15) / 16 := by
    have h2 := Check.retWords_pair (.array NetworkCheck.Period.outBits (.value 1)) (Fits.rowTy s)
    unfold SampledProofs.Bits.wordsOf at h2
    rewrite [hacc.egress_ret]
    omega
  intro t ht _ σ g hg
  simp only [scanLayout, NetTiming.slotGates, List.mem_cons, List.mem_map, List.mem_range] at hg
  rcases hg with rfl | ⟨b, hb, rfl⟩
  · refine egress_returned hacc hkeys hfs hev hms ht ?_ (wordBit_snd _ 0 0)
    rewrite [wordBit_zero_fst, hw]
    omega
  · have hσ := Lowered.mul_add_le (B := 8 * s.cellBytes) (i := b + 1) σ.isLt (Nat.succ_le_of_lt hb)
    have hC : s.allocation * (8 * s.cellBytes) = s.allocation * s.cellBytes * 8 := by
      rw [Nat.mul_assoc, Nat.mul_comm s.cellBytes 8]
    refine egress_returned hacc hkeys hfs hev hms ht ?_ (wordBit_snd _ 1 _)
    rewrite [hrets, wordBit_one_fst, hA, hw]
    omega

/-! ## The ingress gates and the fit -/

open SampledProofs.Bits in
/-- **An ingress instance's row bit is an input** of the members' program: of a Program whose Definitions' keys are
distinct, at its templates' circuit files, which read their `row` parameters (`RowsRead`), and its audit's members,
bit `ℓ` of the `row` parameter of an ingress bucket's instance (`Fits.Ingress`), the template no input's key. The
instance is a unit of a member over the template's circuit file (`member_instPos`). Bits' check makes the bit's port bit
a message bit (`Reads.paramsWhy_first`), at its port's column, the ports being in column order (`Check.Facts.ordered`,
`Reads.srcKey_port`), so it has a key and a gate (`Lowered.inGate_isSome`), and the unit reads it
(`Reads.readsWhy_first`, `Reads.unitReads_true`), so no constant fixes it (`Lowered.inBit_input`). -/
theorem ingress_input {P : Flock.Program.Prog} {s : Schedule} {fs : List Flock.Circuit} {tids : Array String}
    {rs : Array Flock.TemplateQuery.TRange} {ms : Members} (hroot : P.root < P.defs.size)
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hfs : FilesOf P fs) (hrr : RowsRead P fs)
    (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hms : MembersOf P fs rs ms) {i t ℓ : ℕ}
    (hi : i < (templateNodes P).length)
    (hin : Flock.Program.isInputId (Fits.callee P ((templateNodes P).getD i 0)).key = false)
    (hing : Fits.Ingress P s ((templateNodes P).getD i 0, t))
    (hℓ : ℓ < SampledProofs.Bits.bitsOf (Fits.rowTy s)) :
    (jointOf ms).gates[instPos P rs ms i t fun c n u => inBit c n u 0 (ℓ / 16) (ℓ % 16)]? = some .input := by
  have hkm : (templateNodes P).getD i 0 ∈ templateNodes P := hing.1
  have hb : (Fits.node P ((templateNodes P).getD i 0)).form = .batch := hing.2.1
  have ht : t < (Fits.node P ((templateNodes P).getD i 0)).n := hing.2.2.1
  have hsz : 0 < (Fits.callee P ((templateNodes P).getD i 0)).params.size := hing.2.2.2.1
  have hname : ((Fits.callee P ((templateNodes P).getD i 0)).params[0]!).1 = "row" := hing.2.2.2.2.1
  have hty : ((Fits.callee P ((templateNodes P).getD i 0)).params[0]!).2 = Fits.rowTy s := hing.2.2.2.2.2.1
  obtain ⟨m, hm, u, hu, hci, hpos⟩ := member_instPos (t := t) hroot hkeys hfs hev hms hi hin hb ht
    (fun c n u => inBit c n u 0 (ℓ / 16) (ℓ % 16))
  have hi' : i < fs.length := by rw [hfs.1]; exact hi
  have hF := Check.facts (hfs.2 i hi')
  have hR := hrr i hi'
  rw [← hci] at hF hR
  have hp0 : (Fits.callee P ((templateNodes P).getD i 0)).params[0]! = ("row", Fits.rowTy s) := Prod.ext hname hty
  have hd : (templateDef P ((templateNodes P).getD i 0)).params.toList =
      ("row", Fits.rowTy s) :: (templateDef P ((templateNodes P).getD i 0)).params.toList.tail := by
    rw [← hp0]; exact Reads.toList_cons hsz
  have hfind := find?_key hkeys (templateDef_mem hkm)
  have hports := hF.ports _ hfind
  have hread := (Reads.readsWhy_first hd hR).2 ℓ hℓ
  obtain ⟨r, hrlt, hrmem, hrx⟩ := Reads.unitReads_true hread
  obtain ⟨q, byte, bit, hsrc⟩ := Reads.paramsWhy_first hd (by have := hF.upv; omega) hports.2 hℓ
  have hp : ℓ / 16 < (Flock.portCols (ms.getD m default).1.unit.inGroups).size := by
    have hpcs := hports.1
    rw [hd] at hpcs
    simp only [List.map_cons, List.sum_cons, SampledProofs.Bits.wordsOf] at hpcs
    omega
  have hτ : ℓ % 16 < ((Flock.portCols (ms.getD m default).1.unit.inGroups).getD (ℓ / 16) (0, 0)).2 := by
    have := hF.widths _ hp; omega
  have hW : ((Flock.portCols (ms.getD m default).1.unit.inGroups).getD (ℓ / 16) (0, 0)).1 + ℓ % 16 <
      (ms.getD m default).1.unit.inWords * Flock.WORD := by
    rw [← hF.inWords]; exact hrx
  have hsk := Reads.srcKey_port (ix := Index.of (ms.getD m default).1) hF.ordered hp hτ hW
  have hcol : inCol (ms.getD m default).1 0 (ℓ / 16) (ℓ % 16) =
      ((Flock.portCols (ms.getD m default).1.unit.inGroups).getD (ℓ / 16) (0, 0)).1 + ℓ % 16 := by
    unfold inCol; rw [Nat.zero_mul, Nat.zero_add]; try rfl
  have hxB : inCol (ms.getD m default).1 0 (ℓ / 16) (ℓ % 16) <
      ((Rows.ofNet (ms.getD m default).1.unit).stack (ms.getD m default).1.upv).nIn := by
    show _ < (ms.getD m default).1.upv * (Rows.ofNet (ms.getD m default).1.unit).nIn
    rw [hcol, hF.upv, Nat.one_mul]; exact hrx
  obtain ⟨b, hbk⟩ := Reads.keyIdx_isSome (c := (ms.getD m default).1) hxB (by rw [hcol, hsk, hsrc]; simp)
  have hgate := Lowered.inGate_isSome (n := (ms.getD m default).2.size) hu (by have := hF.upv; omega) hp
    (show ℓ % 16 < ((Flock.portCols (ms.getD m default).1.unit.inGroups).getD (ℓ / 16) default).2 from hτ) hbk
  have hx' : inBit (ms.getD m default).1 (ms.getD m default).2.size u 0 (ℓ / 16) (ℓ % 16) =
      some (posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
        (2 + u * ((Rows.ofNet (ms.getD m default).1.unit).stack (ms.getD m default).1.upv).nIn + b)) := by
    simp only [inBit, hgate, Option.map_some]
  have hpos' : instPos P rs ms i t (fun c n u => inBit c n u 0 (ℓ / 16) (ℓ % 16)) =
      base ms m + posOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
        (2 + u * ((Rows.ofNet (ms.getD m default).1.unit).stack (ms.getD m default).1.upv).nIn + b) := by
    rw [hpos]
    simp only [hx', Option.map_some, Option.getD_some]
  rw [hpos']
  exact Lowered.inBit_input ms hm hx' hrlt hrmem hrx

/-- **The scan's ingress gates are inputs** (`NetTiming.BoundaryGates.Fits`'s second part): of an accepted scan whose
Definitions' keys are distinct, at its templates' circuit files reading their `row` parameters and its audit's
members. Step `t`'s ingress gates are the bits of its bucket's instance's `row`, parameter 0: bucket 0's of the first
template, a later bucket's of the second. -/
theorem scan_ingress {P : Flock.Program.Prog} {s : Schedule} {fs : List Flock.Circuit} {tids : Array String}
    {rs : Array Flock.TemplateQuery.TRange} {ms : Members} (hacc : Fits.Accepted P s)
    (hkeys : (P.defs.toList.map (·.key)).Nodup) (hfs : FilesOf P fs) (hrr : RowsRead P fs)
    (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs) (hms : MembersOf P fs rs ms) :
    ∀ t < s.window, ∀ g ∈ (scanLayout P s rs ms).ingress t, (jointOf ms).gates[g]? = some .input := by
  have hR : SampledProofs.Bits.bitsOf (Fits.rowTy s) = s.allocation * s.cellBytes * 8 := by
    simp only [Check.bits_bitsOf, Fits.rowTy, Check.bits_array, Check.bits_value, Nat.mul_one]
  have hC : s.allocation * (8 * s.cellBytes) = s.allocation * s.cellBytes * 8 := by
    rw [Nat.mul_assoc, Nat.mul_comm s.cellBytes 8]
  have hlen : 2 ≤ (templateNodes P).length := by
    rcases hacc.shape with ⟨a, e, hts, -⟩ | ⟨a, b, e, hts⟩ <;> simp [hts]
  intro t ht g hg
  have hing := hacc.ingress t ht
  by_cases ht0 : t = 0
  · subst ht0
    have hb0 : Fits.bucket P 0 = ((templateNodes P).getD 0 0, 0) := by
      simp only [Fits.bucket, ↓reduceIte, headD_getD]
    rw [hb0] at hing
    simp only [scanLayout, ↓reduceIte, List.mem_map, List.mem_range] at hg
    obtain ⟨ℓ, hℓ, rfl⟩ := hg
    have hℓ' : ℓ < SampledProofs.Bits.bitsOf (Fits.rowTy s) := by rw [hR, ← hC]; exact hℓ
    simp only [wordBit_zero_fst, wordBit_zero_snd]
    exact ingress_input hacc.root_lt hkeys hfs hrr hev hms (by omega) (hacc.inputs _ hing.1) hing hℓ'
  · have hbt : Fits.bucket P t = ((templateNodes P).getD 1 0, t - 1) := by
      simp only [Fits.bucket, ht0, ↓reduceIte]
    rw [hbt] at hing
    simp only [scanLayout, ht0, ↓reduceIte, List.mem_map, List.mem_range] at hg
    obtain ⟨ℓ, hℓ, rfl⟩ := hg
    have hℓ' : ℓ < SampledProofs.Bits.bitsOf (Fits.rowTy s) := by rw [hR, ← hC]; exact hℓ
    simp only [wordBit_zero_fst, wordBit_zero_snd]
    exact ingress_input hacc.root_lt hkeys hfs hrr hev hms (by omega) (hacc.inputs _ hing.1) hing hℓ'

/-- **An accepted scan's layout fits its audit's joint program**: of a Program `scanProg` accepts whose Definitions'
keys are distinct, at its templates' circuit files reading their `row` parameters and its audit's members, the scan
layout's egress gates are returned and not inputs (`scan_egress`), its ingress gates inputs (`scan_ingress`), no egress
gate depends on a later step's ingress (`scan_causal`), and with no wires the wiring fits (`wiresFit_none`). -/
theorem scan_fits {program : String} {s : Schedule} {sc : NetworkCheck.Period.Scan} {P : Flock.Program.Prog}
    {fs : List Flock.Circuit} {tids : Array String} {rs : Array Flock.TemplateQuery.TRange} {ms : Members}
    (hscan : NetworkCheck.Period.scanProg P program s = .ok sc) (hkeys : (P.defs.toList.map (·.key)).Nodup)
    (hfs : FilesOf P fs) (hrr : RowsRead P fs) (hev : Flock.TemplateQuery.evaluate (.ofProg P) tids = .ok rs)
    (hms : MembersOf P fs rs ms) :
    (scanLayout P s rs ms).Fits (jointOf ms) s.window := by
  have hacc := Fits.scanProg_accepted hscan
  have hE := scan_egress hacc hkeys hfs hev hms
  exact ⟨hE, scan_ingress hacc hkeys hfs hrr hev hms, scan_causal hacc fun t ht ℓ σ g hg => (hE t ht ℓ σ g hg).1,
    wiresFit_none rfl⟩

/-- **`ScanFits`**: `scan_fits`, whose distinct keys are `scanProg`'s own `keysUnique` check (`Fits.Accepted.keys`). -/
theorem scanFits : ScanFits := fun _ _ _ _ _ _ _ _ hscan hfs hrr hev hms =>
  scan_fits hscan (Properties.NetworkPeriod.keysUnique_nodup (Fits.scanProg_accepted hscan).keys) hfs hrr hev hms

/-- One slot of one lane and a window of one step. -/
def oneSlot : NetworkCheck.Period.Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩

/-- A circuit that returns nothing fits no layout over a horizon of at least one step: step 0's egress presence gate
isn't returned. -/
theorem not_fits_of_no_outs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (gs : List Claims.Sampling.BGate) {k : ℕ}
    (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits ⟨gs, []⟩ k := by
  rintro ⟨h, -, -⟩
  have := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

/-- An audit of no members has the empty joint program. -/
theorem jointOf_nil : jointOf [] = ⟨[], []⟩ := rfl

/-- An audit of no members fits no layout over a horizon of at least one step. -/
theorem not_fits_of_no_members {L S : ℕ} (lay : NetTiming.BoundaryGates L S) {k : ℕ} (hk : 0 < k) (ℓ : Fin L)
    (σ : Fin S) : ¬ lay.Fits (jointOf []) k := by
  rw [jointOf_nil]
  exact not_fits_of_no_outs lay [] hk ℓ σ

/-- A Program of one Definition, its root, which has no nodes. -/
def bareProg : Flock.Program.Prog := { defs := #[default], index := {}, root := 0, programFn := "", sha512 := .empty }

theorem bareProg_templateNodes : templateNodes bareProg = [] := rfl

/-- The template query over no templates accepts `bareProg` with no ranges: its root has no nodes. -/
theorem bareProg_evaluate : Flock.TemplateQuery.evaluate (.ofProg bareProg) #[] = .ok #[] := by
  simp [Flock.TemplateQuery.evaluate, Flock.TemplateQuery.Desc.ofProg, Flock.TemplateQuery.Desc.def?, bareProg]
  rfl

/-- **`ScanFits`' nontrivial lemma**: a scan check that accepts every Program accepts `bareProg`, which has no
templates, so no circuit files and no members: their joint program is empty, and no layout fits it over a window of
one step. -/
theorem scanFits_nontrivial : ¬ ScanFitsOf acceptAllScan := fun h =>
  not_fits_of_no_members (scanLayout bareProg oneSlot #[] []) (k := oneSlot.window) (by decide) ⟨0, by decide⟩
    ⟨0, by decide⟩
    (h bareProg "" oneSlot ⟨0, 0, 0, 0, 0⟩ [] #[] #[] [] rfl
      ⟨by simp only [bareProg_templateNodes, List.length_nil], fun _ h => absurd h (Nat.not_lt_zero _)⟩
      (fun _ h => absurd h (Nat.not_lt_zero _)) bareProg_evaluate
      ⟨List.Perm.refl _, fun _ h => absurd h (Nat.not_lt_zero _)⟩)

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  rfl

theorem bytes_of_data_toList (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := rfl

/-- The statement whose descriptor is the bytes of `null` and whose circuit is empty. -/
def nullStmt : Claims.Sampling.AuditStatement :=
  { Claims.Sampling.ScopeWitness.toyStmt with desc := "null".toUTF8.data.toList, prog := ⟨[], []⟩ }

/-- A deployment whose scope check takes everything and which parses every message to `nullStmt`. -/
def acceptAllDeploy : Claims.Sampling.Deploy Bool ℕ Unit :=
  { Claims.Sampling.ScopeWitness.toyD with parse := fun _ => some nullStmt }

end Properties.NetworkScan
