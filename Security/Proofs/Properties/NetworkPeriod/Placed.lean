import Properties.NetworkPeriod.Placed
import Proofs.Properties.NetworkScan

/-!
# An accepted scan's proof blocks each commit at most `max(b_ing, b_eg)` bits

`ScanWindowsPlaced` reads a window's statement only through `StmtBits`: its gates are some members' joint program
(`jointOf`), each member's circuit file one `fs` lists at a template, and each proof block a member's unit's
(`Bits.blockOf`). The bound is per block, from its member's file alone, then summed over an incorrect set:

* `committed_blockOf` (Lemma A, the file): a unit's block commits at most `16 · (paramWords + retWords)` bits of the
  Definition its file is checked against (`Bits.check`), reading no input padding (`Bits.readsWhy` at no parameter):
  its committed gates are the source gates it owns (`insOf`, at most 16 a port word: a port's bits past its first word
  are the zero, `insOf_length`) and its returned copy rows (16 a word of the return, `units_committed`); no gate
  outside it reads one of its rows (`Lowered.reads_jointUnit`).
* `words_le_terms` (Lemma A, the scan): each template's `16 · (paramWords + retWords)` is at most
  `max sc.ingress sc.egress`, `scanProg`'s terms (`vuBits`): bucket 0's and the later ingress buckets' at `ingress`
  (the later template returns bucket 0's return), the egress template's at `egress`.
* `committed_le_terms` (Lemma A): the two, at a member's file at a template index.
* `faultyBits_le` (Lemma B): an incorrect set of at most `b` blocks, each committing at most `β`, commits at most
  `b · β`.
-/

namespace Properties.NetworkPeriod.Placed

open SampledProofs.Bits
open Properties.NetworkScan
open NetworkCheck.Period (Scan Schedule scanProg templateNodes paramWords vuBits)
open NetworkCheck.Circuit (templateDef)

/-- **Lemma A, the file alone**: where member `m`'s circuit file passes Bits' check as the template `tmpl`, whose
Definition is `d`, and reads no input padding, its unit `u`'s proof block (`blockOf`, at the member's constants and
first position) commits at most `16 · (paramWords d.params + retWords d.ret)` bits of the members' joint program. -/
theorem committed_blockOf {P : Flock.Program.Prog} {tmpl : String} {d : Flock.Program.Defn} {ms : Members} {m u : ℕ}
    (hm : m < ms.length) (hu : u < (ms.getD m default).2.size)
    (hc : check (ms.getD m default).1 P tmpl = none) (hd : P.defs.find? (·.key == tmpl) = some d)
    (hr : readsWhy (ms.getD m default).1 d (unitReads (ms.getD m default).1) [] = none) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      16 * (paramWords d.params + retWords d.ret) := by
  sorry

open Properties.NetworkScan.Fits (callee egressNode bind_ok need_ok needSome_ok ingressTemplates_ok split_two
  split_three)
open NciCheck.Properties.SecurityProofs (BodiesOk.ty_beq)

/-- **An accepted scan's terms**: `egress` is the egress template's `vuBits`, and `ingress` bucket 0's template's, at
the larger parameter words of it and the later ingress template, which returns what it returns. -/
theorem scanProg_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    (h : scanProg P program s = .ok sc) :
    sc.egress = vuBits 1 (paramWords (callee P (egressNode P)).params + retWords (callee P (egressNode P)).ret) ∧
    ((∃ a e, templateNodes P = [a, e] ∧
        sc.ingress = vuBits 1 (max (paramWords (callee P a).params) 0 + retWords (callee P a).ret)) ∨
      ∃ a b e, templateNodes P = [a, b, e] ∧ (callee P b).ret = (callee P a).ret ∧
        sc.ingress = vuBits 1 (max (paramWords (callee P a).params) (paramWords (callee P b).params) +
          retWords (callee P a).ret)) := by
  unfold scanProg at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨root, hroot, h⟩ := bind_ok h
  replace hroot := needSome_ok hroot
  obtain rfl : root = P.defs[P.root]! := by
    obtain ⟨hlt, e⟩ := Array.getElem?_eq_some_iff.1 hroot
    rw [getElem!_pos P.defs _ hlt, e]
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨xe, de⟩, hback, h⟩ := bind_ok h
  replace hback := needSome_ok hback
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨⟨xf, df⟩, later⟩, hm, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, h6, h⟩ := bind_ok h
  replace h6 := need_ok h6
  iterate 12 obtain ⟨_, -, h⟩ := bind_ok h
  have hsc := (Except.ok.inj h).symm
  subst hsc
  have hxe : xe = Fits.node P (egressNode P) ∧ de = callee P (egressNode P) := by
    rw [Array.back?_map, List.back?_toArray, Option.map_eq_some_iff] at hback
    obtain ⟨e, he, hge⟩ := hback
    have hen : egressNode P = e := by simp [egressNode, he]
    rw [hen]
    simp only [Prod.mk.injEq] at hge
    exact ⟨hge.1.symm, hge.2.symm⟩
  obtain ⟨rfl, rfl⟩ := hxe
  refine ⟨rfl, ?_⟩
  rcases ingressTemplates_ok hm with ⟨f, hf, hx⟩ | ⟨f, b, hf, hx⟩
  · simp only [Array.toList_pop, Array.toList_map] at hf
    obtain ⟨a, e, hts, hga⟩ := split_two hf
    simp only [Prod.mk.injEq] at hx
    obtain ⟨hx1, rfl⟩ := hx
    have hfa : (xf, df) = (Fits.node P a, callee P a) := by rw [hx1, ← hga]
    simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    exact .inl ⟨a, e, hts, rfl⟩
  · simp only [Array.toList_pop, Array.toList_map] at hf
    obtain ⟨a, a', e, hts, hga, hgb⟩ := split_three hf
    simp only [Prod.mk.injEq] at hx
    obtain ⟨hx1, rfl⟩ := hx
    have hfa : (xf, df) = (Fits.node P a, callee P a) := by rw [hx1, ← hga]
    have hba : b = (Fits.node P a', callee P a') := by rw [← hgb]
    simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    subst hba
    simp only [Option.all_some, Bool.and_eq_true] at h6
    exact .inr ⟨a, a', e, hts, BodiesOk.ty_beq h6.2.2, rfl⟩

/-- **Lemma A, the scan's terms**: at an accepted scan, each template's `16 · (paramWords + retWords)` is at most
`max sc.ingress sc.egress`. -/
theorem words_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    (h : scanProg P program s = .ok sc) {i : ℕ} (hi : i < (templateNodes P).length) :
    16 * (paramWords (templateDef P ((templateNodes P).getD i 0)).params +
      retWords (templateDef P ((templateNodes P).getD i 0)).ret) ≤ max sc.ingress sc.egress := by
  obtain ⟨heg, hin⟩ := scanProg_terms h
  have hing : ∀ {k}, 16 * (paramWords (callee P k).params + retWords (callee P k).ret) ≤ sc.ingress →
      16 * (paramWords (callee P k).params + retWords (callee P k).ret) ≤ max sc.ingress sc.egress :=
    fun h => h.trans (le_max_left _ _)
  have hegr : 16 * (paramWords (callee P (egressNode P)).params + retWords (callee P (egressNode P)).ret) ≤
      max sc.ingress sc.egress := by
    refine le_trans ?_ (le_max_right _ _)
    rw [heg, vuBits]; omega
  rcases hin with ⟨a, e, hts, hin⟩ | ⟨a, b, e, hts, hba, hin⟩
  · have hen : egressNode P = e := by simp [egressNode, hts]
    rw [hen] at hegr
    rw [hts] at hi ⊢
    have hi2 : i = 0 ∨ i = 1 := by simp only [List.length_cons, List.length_nil] at hi; omega
    rcases hi2 with rfl | rfl
    · show 16 * (paramWords (callee P a).params + retWords (callee P a).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits]
      have := Nat.le_max_left (paramWords (callee P a).params) 0
      omega
    · exact hegr
  · have hen : egressNode P = e := by simp [egressNode, hts]
    rw [hen] at hegr
    rw [hts] at hi ⊢
    have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by simp only [List.length_cons, List.length_nil] at hi; omega
    rcases hi3 with rfl | rfl | rfl
    · show 16 * (paramWords (callee P a).params + retWords (callee P a).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits]
      have := Nat.le_max_left (paramWords (callee P a).params) (paramWords (callee P b).params)
      omega
    · show 16 * (paramWords (callee P b).params + retWords (callee P b).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits, hba]
      have := Nat.le_max_right (paramWords (callee P a).params) (paramWords (callee P b).params)
      omega
    · exact hegr

/-- **Lemma A**: at an accepted scan whose templates' files `fs` Bits lays out (`FilesOf`) reading no input padding
(`PaddingUnread`), a unit's proof block of a member whose file is the one `fs` lists at a template index commits at
most `max sc.ingress sc.egress` bits of the members' joint program, whatever the other members are. -/
theorem committed_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    {fs : List Flock.Circuit} (hscan : scanProg P program s = .ok sc) (hfs : FilesOf P fs) (hpad : PaddingUnread P fs)
    {ms : Members} {m u : ℕ} (hm : m < ms.length) (hu : u < (ms.getD m default).2.size) {i : ℕ}
    (hi : i < (templateNodes P).length) (hfile : (ms.getD m default).1 = fs.getD i default) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      max sc.ingress sc.egress := by
  have hi' : i < fs.length := hfs.1 ▸ hi
  have hc := hfs.2 i hi'
  have hr := hpad i hi'
  rw [← hfile] at hc hr
  have hk : (templateNodes P).getD i 0 ∈ templateNodes P := by
    rw [Query.list_getD_of_lt _ hi]; exact List.getElem_mem hi
  have hd := find?_key (Properties.NetworkPeriod.keysUnique_nodup (Fits.scanProg_accepted hscan).keys)
    (templateDef_mem hk)
  exact (committed_blockOf hm hu hc hd hr).trans (words_le_terms hscan hi)

/-- **Lemma B**: a set `F` of the partition's blocks, at most `b` of them, each committing at most `β` bits, commits at
most `b · β` (`faultyBits`). -/
theorem faultyBits_le {C : Claims.Sampling.Prog} {Q : Claims.Sampling.Partition} {F : Finset ℕ} {β b : ℕ}
    (hF : F ⊆ Finset.range Q.blocks.length)
    (hβ : ∀ v < Q.blocks.length, Claims.Sampling.committed C (Q.blocks.getD v []) ≤ β) (hb : F.card ≤ b) :
    NetTiming.faultyBits C Q F ≤ b * β := by
  unfold NetTiming.faultyBits
  calc ∑ v ∈ F, Claims.Sampling.committed C (Q.blocks.getD v []) ≤ ∑ _v ∈ F, β :=
        Finset.sum_le_sum fun v hv => hβ v (Finset.mem_range.1 (hF hv))
    _ = F.card * β := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ b * β := Nat.mul_le_mul_right β hb

end Properties.NetworkPeriod.Placed
