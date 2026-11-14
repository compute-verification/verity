module

public import Proofs.Properties.Verity.Sampled.MemberForm.Sample
public import Proofs.Properties.Verity.Sampled.MemberForm.Read
public import Proofs.Properties.Verity.Sampled.Members.Draw
public import Proofs.Properties.Verity.Sampled.Window.Steps
public import Proofs.Flock.Soundness.ExecDraw

/-!
# The units the per-member form's sessions proved, member by member

The per-member form never compares a session's record's sample with the window's. It holds the two together through
the units: the units its sessions proved, appended and sorted, are the evidence's `proved_units`
(`Spot.OneStageZk.provedHeld`), which v1's window check holds to the window's sample. This module reads, off a run
that accepted every session, which session is which member's and what its record's sample is:

* **The run's sessions, a statement each** (`files_statement`, `verdicts_units`,
  `run_units`): each statement's one session has its verdict at its member's own statement (`ProvedAt`), and a unit
  is one of the evidence's exactly when one of those sessions proved it.
* **The sessions' units over the members** (`run_over`): the window's members are the parameters', a range each,
  no unit in two (`Window.Laid`); so the run's statements are the members the sample holds, in the members' order,
  and a unit of a statement's member is the evidence's exactly when the statement's session proved it
  (`UnitsOver`).
* **A member's part of the sample** (`part_sampled`, `printedPart_of_mem`): at a sample that holds the proof blocks
  at the evidence's units, a window's member has a proof block in the sample exactly when the sample holds one of
  its units, and the sample file of its part is the subset print of the slots its units' proof blocks are at.
* **The session of a member the sample holds** (`member_session`): the member at place `i` of the window with a
  proof block in the sample has the statement at place `drawnBefore W.ms S i`, its one session's verdict at the
  member's own statement, and that session's record's sample is the sample file of the member's part; and the
  statements are as many as the members the sample holds.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-! ## The run's sessions, a statement each -/

/-- **A member's statement with its files held is the member's statement.** -/
theorem files_statement {m : _root_.Spot.OneStageZk.Member} {slice : Json} {bs : _root_.Core.Blobs}
    {a : _root_.Spot.OneStageZk.StepAt} (h : m.files slice bs = .ok a) : m.statement slice bs = .ok a := by
  unfold _root_.Spot.OneStageZk.Member.files at h
  obtain ⟨a', hs, h⟩ := ebind_ok (_root_.Spot.OneStageZk.mapError_ok h)
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain rfl : a' = a := Except.ok.inj h
  exact hs

/-- **A session's verdict at its statement proves the units `units`**, after some earlier sessions' coin-tree
keys. -/
def VerdictOf (bs : _root_.Core.Blobs) (x : StepAt × _root_.Spot.OneStageZk.RunFiles) (units : Array ℕ) : Prop :=
  ∃ keys record spec tabs rest proofs, SessionVerdict x.1 bs keys x.2 record spec tabs rest proofs units

/-- **Sessions with their verdicts in order prove the run's units**: each session has its verdict, proving some
units, and a unit is among the run's after them exactly when it was among the run's before or one of those
sessions proved it. -/
theorem verdicts_units {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') :
    ∃ U : List (Array ℕ), List.Forall₂ (VerdictOf bs) l U ∧
      ∀ x, x ∈ r'.proved ↔ x ∈ r.proved ∨ ∃ us ∈ U, x ∈ us := by
  induction h with
  | nil r => exact ⟨[], .nil, fun x => ⟨Or.inl, fun h => h.elim id fun ⟨_, hus, _⟩ => nomatch hus⟩⟩
  | @cons r r' a s l record spec tabs rest proofs units hsv _ ih =>
    obtain ⟨U, hU, hmem⟩ := ih
    refine ⟨units :: U, .cons ⟨r.coinKeys, record, spec, tabs, rest, proofs, hsv⟩ hU, fun x => ?_⟩
    rw [hmem x]
    have hp : (ranOn a r s record spec tabs rest units).proved = r.proved ++ units := rfl
    rw [hp, Array.mem_append]
    constructor
    · rintro ((h | h) | ⟨us, hus, h⟩)
      · exact Or.inl h
      · exact Or.inr ⟨units, List.mem_cons_self, h⟩
      · exact Or.inr ⟨us, List.mem_cons_of_mem _ hus, h⟩
    · rintro (h | ⟨us, hus, h⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hus with e | hus
        · exact Or.inl (Or.inr (e ▸ h))
        · exact Or.inr ⟨us, hus, h⟩

/-- **The one session of the statement `z.1` has its verdict at its member `z.2`'s own statement**, read at the
member's slice of the registration `rec` at the roots `roots`, and proves the units `units`. -/
def ProvedAt (rec : Json) (roots : Std.HashMap String (Array Json)) (bs : _root_.Core.Blobs)
    (z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member) (units : Array ℕ) : Prop :=
  ∃ a s keys record spec tabs rest proofs, z.2.statement (z.2.slice rec roots) bs = .ok a ∧ z.1.sessions = #[s] ∧
    SessionVerdict (stepAt a) bs keys s record spec tabs rest proofs units

/-- **The per-member form's run, a statement each**: where `membersRun` returns a run that accepted every session,
the statements are members' in order (`membersHeld`), each statement's one session has its verdict at its member's
own statement (`ProvedAt`), and a unit is one of the evidence's `proved_units` exactly when one of those sessions
proved it (`provedHeld`). -/
theorem run_units {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : _root_.Core.Blobs}
    {r : _root_.Spot.OneStageZk.Run} (h : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r)
    (hr : r.all = true) :
    ∃ roots held U, _root_.Spot.OneStageZk.rootsOf ms c.record = .ok roots ∧
      _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      List.Forall₂ (ProvedAt c.record roots bs) (sts.zip held).toList U ∧
      ∀ x, x ∈ e.base.provedUnits ↔ ∃ us ∈ U, x ∈ us := by
  obtain ⟨-, roots, hroots, held, files, l, hheld, hfiles, hst, hv, hprov⟩ := membersRun_ok h hr
  obtain ⟨U, hU, hmem⟩ := verdicts_units hv
  refine ⟨roots, held, U, hroots, hheld, ?_, fun x => ?_⟩
  · refine forall₂_imp (forall₂_trans hst hU) fun z hz units hx => ?_
    obtain ⟨x, ⟨a, ha, hx1, hs⟩, keys, record, spec, tabs, rest, proofs, hsv⟩ := hx
    rw [hx1] at hsv
    exact ⟨a, x.2, keys, record, spec, tabs, rest, proofs,
      files_statement (statementAt_files hfiles ((membersHeld_zip hheld).2 z hz).2.1 ha), hs, hsv⟩
  · rw [← provedHeld_ok hprov, FlockSoundness.Audit.DrawExec.mem_qsort, hmem x]
    exact ⟨fun h => h.elim (fun h0 => absurd h0 (Array.not_mem_empty x)) id, Or.inr⟩

/-! ## The sessions' units over the members -/

/-- **The run's sessions' units over the window's members.** At a window laid out in order whose members are the
parameters' members `vms`, in the order of their first units, where `membersRun` returns a run that accepted
every session: each statement's one session has its verdict at its member's own statement (`ProvedAt`); the
members are a range each, no unit in two, and the sessions' units are over them (`UnitsOver`, at a member's first
unit and the unit after its last); and the statements' members are the members the sample holds, in the members'
order. -/
theorem run_over {F : Array ByteArray} {W : Window} (hlaid : W.Laid) {ms : Array _root_.Spot.OneStageZk.Member}
    {vms : List _root_.Spot.OneStageZk.Member} (hperm : vms.Perm ms.toList)
    (hsorted : vms.Pairwise fun a b => a.first < b.first) (hall : List.Forall₂ (MemberOn F) vms W.ms)
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {e : _root_.Spot.OneStageZk.Evidence} {sts : Array _root_.Spot.OneStageZk.StatementFiles}
    {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) (hr : r.all = true) :
    ∃ roots held U, _root_.Spot.OneStageZk.rootsOf ms c.record = .ok roots ∧
      _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      List.Forall₂ (ProvedAt c.record roots bs) (sts.zip held).toList U ∧
      UnitsOver (fun vm : _root_.Spot.OneStageZk.Member => vm.first) (fun vm => vm.first + vm.count) vms
        held.toList U e.base.provedUnits ∧
      held.toList = vms.filter (·.sampled e.base.provedUnits) ∧
      ∀ vm ∈ vms, ∃ lo hi, vm.units = #[(lo, hi)] := by
  obtain ⟨roots, held, U, hroots, hheld, hU, hwant⟩ := run_units h hr
  obtain ⟨hstruct, -⟩ := membersRun_ok h hr
  have hone : ∀ vm ∈ ms, ∃ lo hi, vm.units = #[(lo, hi)] := laidHeld_one (membersStructure_laid hstruct)
  have hone' : ∀ vm ∈ vms, ∃ lo hi, vm.units = #[(lo, hi)] := fun vm hvm =>
    hone vm (Array.mem_toList_iff.1 (hperm.mem_iff.1 hvm))
  have hzip := membersHeld_zip hheld
  have hfound : ∀ z ∈ (sts.zip held).toList, z.2 ∈ ms := fun z hz =>
    Array.mem_of_find?_eq_some (hzip.2 z hz).2.1
  have hlen : sts.toList.length = held.toList.length := by
    rw [Array.length_toList, Array.length_toList]
    exact hzip.1
  have hover : UnitsOver (fun vm : _root_.Spot.OneStageZk.Member => vm.first) (fun vm => vm.first + vm.count)
      vms held.toList U e.base.provedUnits := by
    refine ⟨hsorted, ?_, ?_, ?_, ?_, ?_, hwant⟩
    · intro a ha b hb x hla hha hlb hhb
      have hla' : a.first ≤ x := hla
      have hha' : x < a.first + a.count := hha
      have hlb' : b.first ≤ x := hlb
      have hhb' : x < b.first + b.count := hhb
      refine forall₂_apart (P := fun m m' : Member W.nJ => ∀ u u', m.emb u ≠ m'.emb u')
        (fun _ _ hP u u' e => hP u' u e.symm) hall hlaid.partitions.disjoint a ha b hb fun m m' hm hm' hP => ?_
      have hu : x - a.first < m.n := by
        rw [hm.units]
        omega
      have hu' : x - b.first < m'.n := by
        rw [hm'.units]
        omega
      refine hP ⟨x - a.first, hu⟩ ⟨x - b.first, hu'⟩ (Fin.ext ?_)
      rw [hm.place, hm'.place]
      show a.first + (x - a.first) = b.first + (x - b.first)
      omega
    · exact (membersHeld_inOrder hheld).imp fun hab => ⟨fun e => hab.1 (by rw [e]), hab.2⟩
    · intro a ha
      obtain ⟨j, hj⟩ := List.getElem?_of_mem ha
      obtain ⟨st, hz⟩ := zip_at hlen hj
      have hzm : (st, a) ∈ (sts.zip held).toList := by
        rw [Array.toList_zip]
        exact List.mem_of_getElem? hz
      exact hperm.mem_iff.2 (Array.mem_toList_iff.2 (hfound (st, a) hzm))
    · intro j a us hj hUj x hx
      obtain ⟨st, hz⟩ := zip_at hlen hj
      have hzm : (st, a) ∈ (sts.zip held).toList := by
        rw [Array.toList_zip]
        exact List.mem_of_getElem? hz
      have hzj : (sts.zip held).toList[j]? = some (st, a) := by
        rw [Array.toList_zip]
        exact hz
      obtain ⟨us', hus', a', s, keys, record, spec, tabs, rest, proofs, hsa, -, hsv⟩ :=
        forall₂_at_left hU j (st, a) hzj
      have hsa' : a.statement (a.slice c.record roots) bs = .ok a' := hsa
      obtain ⟨lo, hi, hu⟩ := hone a (hfound (st, a) hzm)
      obtain ⟨sl, hasc, -, hsl⟩ := session_sample hu hsa' hsv
      have hus : us = sl.map (lo + ·) := (Option.some.inj (hUj.symm.trans hus')).trans hsl
      rw [hus] at hx
      obtain ⟨u, hul, hxu⟩ := Array.mem_map.1 hx
      have hlt : u < a.count := (unitsAscending_spec hasc).1 u (Array.mem_toList_iff.2 hul)
      have hxu' : lo + u = x := hxu
      show a.first ≤ x ∧ x < a.first + a.count
      rw [first_one hu, ← hxu']
      exact ⟨Nat.le_add_right _ _, Nat.add_lt_add_left hlt _⟩
    · intro j us hUj
      obtain ⟨z, hz, -⟩ := forall₂_at hU j us hUj
      rw [Array.toList_zip] at hz
      exact ⟨z.2, (List.getElem?_zip_eq_some.1 hz).2⟩
  refine ⟨roots, held, U, hroots, hheld, hU, hover, ?_, hone'⟩
  obtain ⟨held', hheld', hsam⟩ := _root_.Spot.OneStageZk.membersRun_sampled h
  obtain rfl : held = held' := Except.ok.inj (hheld.symm.trans hheld')
  refine hover.kept (fun a ha => ?_) fun a ha => hsam a (Array.mem_toList_iff.1 ha)
  obtain ⟨lo, hi, hu⟩ := hone' a ha
  show a.sampled e.base.provedUnits = true ↔ ∃ x ∈ e.base.provedUnits, a.first ≤ x ∧ x < a.first + a.count
  rw [sampled_one hu, first_one hu, count_one hu]
  constructor
  · rintro ⟨x, hx, h1, h2⟩
    exact ⟨x, hx, h1, by omega⟩
  · rintro ⟨x, hx, h1, h2⟩
    exact ⟨x, hx, h1, by omega⟩

/-! ## A member's part of the sample -/

/-- **A window's member has a proof block in the sample exactly when the sample holds one of its units**: at the
window's member `m` that is the verifier's member `vm`, of the one range `[lo, hi)`, where the sample `S` holds
the proof blocks at the units `want`. -/
theorem part_sampled {F : Array ByteArray} {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ}
    (hon : MemberOn F vm m) {lo hi : ℕ} (hu : vm.units = #[(lo, hi)]) {S : Finset (Fin nJ)} {want : Array ℕ}
    (hS : ∀ v, v ∈ S ↔ v.val ∈ want) : (m.part S).Nonempty ↔ vm.sampled want = true := by
  have hn : m.n = hi - lo := hon.units.trans (count_one hu)
  have hplace : ∀ u : Fin m.n, (m.emb u).val = lo + u.val := fun u =>
    (hon.place u).trans (congrArg (· + u.val) (first_one hu))
  have hpart : (m.part S).Nonempty ↔ ∃ u : Fin m.n, m.emb u ∈ S := by
    constructor
    · rintro ⟨u, h⟩
      exact ⟨u, (m.mem_part S u).1 h⟩
    · rintro ⟨u, h⟩
      exact ⟨u, (m.mem_part S u).2 h⟩
  rw [hpart, range_sampled hplace hS, sampled_one hu, hn]
  constructor
  · rintro ⟨x, hx, h1, h2⟩
    exact ⟨x, hx, h1, by omega⟩
  · rintro ⟨x, hx, h1, h2⟩
    exact ⟨x, hx, h1, by omega⟩

/-- **The sample file of a member's part is the subset print of its slots**: where the member's proof block `u`
is in the sample exactly when `u` is among the slots `us`, ascending below the member's count, the sample file
the member's session is served at its part of the sample (`printedPart`) is `UnitDraw.toJson` of `us` under
`subset` of their number, over the member's count. -/
theorem printedPart_of_mem {nJ : ℕ} (m : Member nJ) (S : Finset (Fin nJ)) {us : Array ℕ}
    (ha : unitsAscending m.n us = true) (hmem : ∀ u : Fin m.n, m.emb u ∈ S ↔ u.val ∈ us) :
    printedPart (m.part S) = Flock.Draw.UnitDraw.toJson ⟨.subset us.size, m.n, us, #[]⟩ := by
  have hp : m.part S = Finset.univ.filter fun u : Fin m.n => u.val ∈ us :=
    Finset.ext fun u => (m.mem_part S u).trans ((hmem u).trans (by simp))
  rw [hp, printedPart, printedDraw, card_filter_ascending ha, drawOf_filter ha]

/-! ## The session of a member the sample holds -/

/-- **The session of a member the sample holds.** At a window laid out in order whose members are the parameters'
members `vms`, in the order of their first units, where `membersRun` returns a run that accepted every session and
the sample `S` holds the proof blocks at the evidence's `proved_units`: the statements are as many as the window's
members with a proof block in `S`; and the member `m` at place `i` with a proof block in `S` is a verifier's
member `vm` whose statement is the one at place `drawnBefore W.ms S i`, whose one session has its verdict at
`vm`'s own statement, with a record whose sample is the sample file of `m`'s part of `S`. -/
theorem member_session {F : Array ByteArray} {W : Window} (hlaid : W.Laid)
    {ms : Array _root_.Spot.OneStageZk.Member} {vms : List _root_.Spot.OneStageZk.Member}
    (hperm : vms.Perm ms.toList) (hsorted : vms.Pairwise fun a b => a.first < b.first)
    (hall : List.Forall₂ (MemberOn F) vms W.ms) {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : _root_.Core.Blobs}
    {r : _root_.Spot.OneStageZk.Run} (h : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r)
    (hr : r.all = true) {S : Finset (Fin W.nJ)} (hS : ∀ v, v ∈ S ↔ v.val ∈ e.base.provedUnits) :
    ∃ roots held, _root_.Spot.OneStageZk.rootsOf ms c.record = .ok roots ∧
      _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧ drawnBefore W.ms S W.ms.length = sts.size ∧
      ∀ i m, W.ms[i]? = some m → (m.part S).Nonempty →
        ∃ vm st a s keys record spec tabs rest proofs units, MemberOn F vm m ∧
          (sts.zip held).toList[drawnBefore W.ms S i]? = some (st, vm) ∧
          vm.statement (vm.slice c.record roots) bs = .ok a ∧ st.sessions = #[s] ∧
          SessionVerdict (stepAt a) bs keys s record spec tabs rest proofs units ∧
          Flock.Witness.recordDraw record = some (printedPart (m.part S)) := by
  obtain ⟨roots, held, U, hroots, hheld, hU, hover, hkept, hone⟩ := run_over hlaid hperm hsorted hall h hr
  have hlen : sts.toList.length = held.toList.length := by
    rw [Array.length_toList, Array.length_toList]
    exact (membersHeld_zip hheld).1
  have hall' : List.Forall₂ (fun vm m => MemberOn F vm m ∧ ∃ lo hi, vm.units = #[(lo, hi)]) vms W.ms :=
    forall₂_imp hall fun vm hvm m hm => ⟨hm, hone vm hvm⟩
  have hsame : ∀ (vm : _root_.Spot.OneStageZk.Member) (m : Member W.nJ),
      (MemberOn F vm m ∧ ∃ lo hi, vm.units = #[(lo, hi)]) →
      vm.sampled e.base.provedUnits = @decide (m.part S).Nonempty (Classical.propDecidable _) :=
    fun vm m hvm => by
      obtain ⟨hon, lo, hi, hu⟩ := hvm
      exact Bool.eq_iff_iff.2
        ⟨fun hs => @decide_eq_true _ (Classical.propDecidable _) ((part_sampled hon hu hS).2 hs),
          fun hd => (part_sampled hon hu hS).1 (@of_decide_eq_true _ (Classical.propDecidable _) hd)⟩
  have hcount : ∀ i, drawnBefore W.ms S i = (vms.take i).countP (·.sampled e.base.provedUnits) := fun i => by
    unfold drawnBefore
    exact (forall₂_countP hall' hsame i).symm
  refine ⟨roots, held, hroots, hheld, ?_, fun i m hi hne => ?_⟩
  · rw [hcount, ← hall.length_eq, ← filter_length (·.sampled e.base.provedUnits) vms, ← hkept,
      Array.length_toList]
    exact (membersHeld_zip hheld).1.symm
  · obtain ⟨vm, hvi, hon, lo, hi', hu⟩ := forall₂_at hall' i m hi
    have hsamp : vm.sampled e.base.provedUnits = true := (part_sampled hon hu hS).1 hne
    have hj : held.toList[drawnBefore W.ms S i]? = some vm := by
      rw [hkept, hcount]
      exact filter_at _ vms i vm hvi hsamp
    obtain ⟨st, hz⟩ := zip_at hlen hj
    have hzj : (sts.zip held).toList[drawnBefore W.ms S i]? = some (st, vm) := by
      rw [Array.toList_zip]
      exact hz
    obtain ⟨us, hUj, a, s, keys, record, spec, tabs, rest, proofs, hsa, hss, hsv⟩ :=
      forall₂_at_left hU _ (st, vm) hzj
    have hsa' : vm.statement (vm.slice c.record roots) bs = .ok a := hsa
    obtain ⟨sl, hasc, hprint, hus⟩ := session_sample hu hsa' hsv
    refine ⟨vm, st, a, s, keys, record, spec, tabs, rest, proofs, us, hon, hzj, hsa', hss, hsv, ?_⟩
    have hasc' : unitsAscending m.n sl = true := by
      rw [hon.units]
      exact hasc
    have hpp : printedPart (m.part S) = Flock.Draw.UnitDraw.toJson ⟨.subset sl.size, m.n, sl, #[]⟩ := by
      refine printedPart_of_mem m S hasc' fun u => ?_
      have hul : u.val < vm.count := Nat.lt_of_lt_of_eq u.isLt hon.units
      have hmu := hover.mem_units hj hUj (x := vm.first + u.val) (Nat.le_add_right _ _)
        (Nat.add_lt_add_left hul _)
      rw [emb_mem hon.place hS u, hmu, hus, first_one hu, mem_map_add]
    rw [hprint, hpp, hon.units]

end Verity.Sampled.MemberForm
