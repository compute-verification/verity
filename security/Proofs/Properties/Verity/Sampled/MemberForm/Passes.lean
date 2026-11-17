module

public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Spot.Credited
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Record

/-! A member's files pass the files check where the per-member form's run held them. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Discharge.ZkExec (zkTags)
open FlockSoundness.Refine (ebind_ok TablesExtend parse_okLe parseCommit_okLe)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem parse_both {tags : Flock.Tags} {cf : ByteArray} {T : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hp : Flock.HmOut.parse tags cf T = .ok c) :
    Flock.HmOut.parseCommit tags cf T = .ok c ∧ Flock.HmRow.parse tags cf T = .ok c := by
  have hi := Flock.HmOut.parse_isTmpl hp
  refine ⟨?_, ?_⟩
  · rw [Flock.HmOut.parseCommit, ite_eq_right (by simp [hi])]
    exact hp
  · unfold Flock.HmOut.parse at hp
    rw [ite_eq_right (by simp [hi])] at hp
    dsimp only at hp
    rw [ite_eq_left (by simp [ht])] at hp
    obtain ⟨c', hc', hp⟩ := ebind_ok hp
    obtain ⟨_, -, hp⟩ := ebind_ok hp
    cases hp
    exact hc'

theorem circuitOf_flat {tags : Flock.Tags} {cf : ByteArray} {T : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hp : Flock.HmOut.parse tags cf T = .ok c) :
    Flock.ProvedScope.circuitOf tags cf T = .ok c := by
  obtain ⟨hpc, hr⟩ := parse_both ht hp
  unfold Flock.ProvedScope.circuitOf
  split
  · exact hpc
  · exact hr

theorem circuitOf_back {tags : Flock.Tags} {cf : ByteArray} {T T' : List (String × ByteArray)} {c : Flock.Circuit}
    (hext : TablesExtend T T') (ht : tags.typed = false) (hp : Flock.HmOut.parse tags cf T = .ok c) :
    Flock.ProvedScope.circuitOf tags cf T' = .ok c := by
  obtain ⟨hpc, hr⟩ := parse_both ht hp
  unfold Flock.ProvedScope.circuitOf
  split
  · exact parseCommit_okLe hext _ _ c hpc
  · exact parse_okLe hext _ _ _ _ c hr

theorem statementOf_checks {p : _root_.Spot.OneStageZk.Own} {population : ByteArray → Flock.V Unit}
    {tables : List (String × ByteArray)} {bs : _root_.Core.Blobs} {a : _root_.Spot.OneStageZk.StepAt}
    (h : _root_.Spot.OneStageZk.statementOf p population tables bs = .ok a) :
    a.ksFiles = Flock.PortRoots.checkOf
        (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables)
        (Flock.Witness.zkTags a.I) a.I.publicFile ∧
      a.population = population a.I.publicFile := by
  unfold _root_.Spot.OneStageZk.statementOf at h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  split at h
  · cases h
  · rename_i tags0 htags
    split at h
    · cases h
    · split at h
      · cases h
      · obtain ⟨circuitB, hc, h⟩ := ebind_ok h
        obtain ⟨pubB, hp, h⟩ := ebind_ok h
        split at h
        · cases h
        · obtain ⟨rd, -, h⟩ := ebind_ok h
          have ha : _root_.Spot.OneStageZk.StepAt.ofOwn p tags0 circuitB pubB tables rd (population pubB) = a :=
            Except.ok.inj h
          subst ha
          exact ⟨rfl, rfl⟩

theorem statement_checks {vm : _root_.Spot.OneStageZk.Member} {slice : Json} {bs : _root_.Core.Blobs}
    {a : _root_.Spot.OneStageZk.StepAt} (h : vm.statement slice bs = .ok a) :
    a.ksFiles = Flock.PortRoots.checkOf
        (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables)
        (Flock.Witness.zkTags a.I) a.I.publicFile ∧
      a.population = vm.population slice a.I.publicFile := by
  unfold _root_.Spot.OneStageZk.Member.statement at h
  obtain ⟨a', hs, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  have ha : a' = a := Except.ok.inj h
  subst ha
  obtain ⟨tables, -, hs⟩ := ebind_ok hs
  exact statementOf_checks hs

theorem population_header {vm : _root_.Spot.OneStageZk.Member} {slice : Json} {pubB : ByteArray}
    (h : vm.population slice pubB = .ok ()) :
    ∃ hd, _root_.Spot.FilePart.header pubB = .ok hd ∧ _root_.Spot.OneStageZk.headerHeld slice hd = .ok () := by
  unfold _root_.Spot.OneStageZk.Member.population at h
  obtain ⟨prog, -, h⟩ := ebind_ok h
  obtain ⟨strata, -, h⟩ := ebind_ok h
  obtain ⟨hd, hh, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  exact ⟨hd, hh, h⟩

theorem headerHeld_instances {rec hd : Json} (h : _root_.Spot.OneStageZk.headerHeld rec hd = .ok ()) :
    _root_.Spot.jeq (_root_.Spot.getD hd "instances") (_root_.Spot.getD rec "population") = true := by
  unfold _root_.Spot.OneStageZk.headerHeld at h
  dsimp only at h
  split at h
  · cases h
  · split at h
    · rename_i he
      exact Properties.Spot.Credited.population_of_matchesPublic (Array.isEmpty_iff.1 he)
    · cases h

theorem slice_population (vm : _root_.Spot.OneStageZk.Member) (rec : Json)
    (roots : Std.HashMap String (Array Json)) :
    _root_.Spot.getD (vm.slice rec roots) "population" = .num ⟨Int.ofNat vm.count, 0⟩ := by
  have h : (vm.slice rec roots).getObjVal? "population" = .ok (Lean.toJson vm.count) :=
    FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp)
  have e : _root_.Spot.getD (vm.slice rec roots) "population" = Lean.toJson vm.count := by
    simp only [_root_.Spot.getD, _root_.Spot.field?, h, Except.toOption, Option.getD_some]
  exact e

theorem load_count {vm : _root_.Spot.OneStageZk.Member} {rec : Json} {roots : Std.HashMap String (Array Json)}
    {pubB : ByteArray} (hpop : vm.population (vm.slice rec roots) pubB = .ok ()) {tags : Flock.Tags}
    {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {P : Flock.Public}
    (hl : Flock.HmIn.loadPublic tags c pi pubB = .ok P) : P.n = vm.count := by
  obtain ⟨hd, hh, hheld⟩ := population_header hpop
  rw [Properties.Spot.Units.headerOf_loadPublic hl] at hh
  obtain rfl := Except.ok.inj hh
  have hj := headerHeld_instances hheld
  have hi : _root_.Spot.getD P.header "instances" = .num ⟨Int.ofNat P.n, 0⟩ := by
    unfold _root_.Spot.getD _root_.Spot.field?
    rw [Properties.Spot.Credited.instances_of_loadPublic hl]
    try rfl
  rw [hi, slice_population] at hj
  exact Properties.Spot.Credited.eq_of_jeq hj

theorem filesPass_of_held {T : PinnedTables} {R : Array ByteArray} {s : Verity.Sent} (ht : TablesSent T s)
    {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ} (hon : MemberOn T R vm m)
    (hr : ReadsFiles R vm (_root_.Core.Blobs.ofArray s.blobs)) {rec : Json}
    {roots : Std.HashMap String (Array Json)}
    (h : vm.FilesHeld (vm.slice rec roots) (_root_.Core.Blobs.ofArray s.blobs)) : FilesPass m := by
  obtain ⟨a, ha, -, hks, -, hpop⟩ := h
  have hsm := statesMember ht hon hr ha
  obtain ⟨eks, epop⟩ := statement_checks ha
  have htags : a.I.tags = m.I.tags := hsm.tags
  have hcf : a.I.circuitFile = m.I.circuitFile := hsm.circuit
  have hpf : a.I.publicFile = m.I.publicFile := hsm.publicFile
  have hcount : a.I.count = m.I.count := hsm.count
  have hext : TablesExtend m.I.tables a.I.tables := hsm.tables
  have hz : Flock.Witness.zkTags a.I = zkTags m.I := by
    unfold FlockSoundness.Discharge.ZkExec.zkTags Flock.Witness.zkTags
    rw [htags, hcf, hcount]
  have hty : (zkTags m.I).typed = false := (FlockSoundness.Discharge.ZkExec.zkTags_typed m.I).trans m.ht
  have hflat := Flock.Desc.parse_flat m.hD
  have hports : Flock.PortRoots.checkOf (.ok m.D.unroll) (zkTags m.I) m.I.publicFile = .ok () := by
    rw [eks, hz, hcf, hpf, circuitOf_back hext hty hflat] at hks
    exact hks
  obtain ⟨pubO, hO⟩ := Flock.PortRoots.checkOf_load hports
  have hn : m.pub.n = m.n := by
    rw [hon.units]
    rw [epop, hpf] at hpop
    exact load_count hpop m.hpub
  refine ⟨⟨pubO, hO, (Flock.Bridge.loadPublic_inOut m.hpub hO).1.trans hn⟩,
    (Flock.Bridge.v5Files_of_checks m.hpub hn hports).1, ?_⟩
  rw [circuitOf_flat hty hflat]
  exact hports

theorem ExecutionPhaseOn.passes {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : ExecutionPhaseOn V T θ F W) {s : Verity.Sent} (ht : TablesSent T s)
    {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member} (hp : ParamsAt θ p)
    (hms : p.members = some ms)
    (hF : ∀ vm ∈ ms, ReadsFiles (V.registered θ) vm (_root_.Core.Blobs.ofArray s.blobs)) {rec : Json}
    {roots : Std.HashMap String (Array Json)}
    (hheld : ∀ vm ∈ ms, vm.FilesHeld (vm.slice rec roots) (_root_.Core.Blobs.ofArray s.blobs)) :
    ∀ m ∈ W.ms, FilesPass m := by
  obtain ⟨p', ms', vms, hp', hms', hperm, -, hall, -⟩ := h.members
  have hpp : p = p' := hp.unique hp'
  subst hpp
  have hmm : ms = ms' := Option.some.inj (hms.symm.trans hms')
  subst hmm
  intro m hm
  obtain ⟨vm, hvm, hon⟩ := forall₂_right hall m hm
  have hvm' : vm ∈ ms := Array.mem_toList_iff.1 (hperm.mem_iff.1 hvm)
  exact filesPass_of_held ht hon (hF vm hvm') (hheld vm hvm')

theorem FilesPass.rootsFix {nJ : ℕ} {m : Member nJ} (h : FilesPass m) : m.RootsFix := by
  obtain ⟨pubO, hO, hnO⟩ := h.loads
  have hty : (zkTags m.I).typed = false := (FlockSoundness.Discharge.ZkExec.zkTags_typed m.I).trans m.ht
  have hports := h.ports
  rw [circuitOf_flat hty (Flock.Desc.parse_flat m.hD)] at hports
  have hn : m.pub.n = m.n := (Flock.Bridge.loadPublic_inOut m.hpub hO).1.symm.trans hnO
  exact (Flock.Bridge.v5Files_of_checks m.hpub hn hports).2

theorem FilesPass.checked_all {nJ : ℕ} {m : Member nJ} (h : FilesPass m) (t : ℕ) : (m.toV5 t).Checked :=
  h.checked h.rootsFix t

theorem MembersSetUp.rootsFix {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : MembersSetUp V T θ F W) : W.RootsFix :=
  fun m hm => (h.passes m hm).rootsFix

end Verity.Sampled.MemberForm
