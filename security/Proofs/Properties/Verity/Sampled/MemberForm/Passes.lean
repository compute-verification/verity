module

public import Proofs.Properties.Verity.Sampled.MemberForm.States
public import Proofs.Properties.Spot.Credited
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Record

/-!
# A member's files pass the files check where the per-member form's run held them

The per-member form's run holds every member of the parameters' files before any session, with a proof block in
the sample or not (`Spot.OneStageZk.Member.FilesHeld`, `Spot.OneStageZk.membersRun_files`): the member's statement
reads off the blobs, and at it the scope check, the files' ports and roots and the population check pass. A window's
setup asks, of each of its members, that its files pass the files check (`FilesPass`). This module carries the first
to the second, at the window's member that is the parameters' member on the registered files (`MemberOn`):

* **The circuit the checks read** (`circuitOf_flat`, `circuitOf_back`): an untyped statement's circuit file that
  parses with some lookup tables is the circuit `Flock.ProvedScope.circuitOf` reads with them, and with tables that
  find by name whatever they find.
* **What the statement's checks are** (`statementOf_checks`, `statement_checks`): the files' check of a member's
  statement is `Flock.PortRoots.checkOf` at the circuit read with its inputs' tables, and its population check the
  member's on its public file.
* **The public file's count** (`population_header`, `headerHeld_instances`, `slice_population`, `load_count`): the
  member's population check holds its public file's header to the member's slice of the registration, whose
  population is the member's units' count, so a load of the file has as many proof blocks as the member has units.
* **The member's files pass** (`filesPass_of_held`): on blobs where v2 reads the member's files among the files
  the window's member is on (`ReadsFiles`) and each pinned table the blobs hold is its registered file
  (`TablesSent`), a member whose files the run held has `FilesPass` at the window's member; so has every member of
  a window (`WindowOn.passes`).
* **Files that pass fix their roots' domains** (`FilesPass.rootsFix`, `FilesPass.checked_all`,
  `MembersSetUp.rootsFix`): the ports check refuses two ports of one root and different published domains, so a
  member whose files pass has `Member.RootsFix`, and is `Checked` as Flock's member at every cost.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Discharge.ZkExec (zkTags)
open FlockSoundness.Refine (ebind_ok TablesExtend parse_okLe parseCommit_okLe)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The circuit the checks read -/

/-- An untyped statement's circuit file that `HmOut.parse` reads is what `HmOut.parseCommit` and `HmRow.parse`
read. -/
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

/-- **The circuit the checks read is the one the file parses to**: an untyped statement's circuit file that
`HmOut.parse` reads as `c` with the tables `T` is the circuit `ProvedScope.circuitOf` reads with them. -/
theorem circuitOf_flat {tags : Flock.Tags} {cf : ByteArray} {T : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hp : Flock.HmOut.parse tags cf T = .ok c) :
    Flock.ProvedScope.circuitOf tags cf T = .ok c := by
  obtain ⟨hpc, hr⟩ := parse_both ht hp
  unfold Flock.ProvedScope.circuitOf
  split
  · exact hpc
  · exact hr

/-- **More lookup tables read the same circuit**: where `T'` finds by name whatever `T` finds and the circuit file
parses with `T`, `ProvedScope.circuitOf` reads with `T'` the circuit it parses to. -/
theorem circuitOf_back {tags : Flock.Tags} {cf : ByteArray} {T T' : List (String × ByteArray)} {c : Flock.Circuit}
    (hext : TablesExtend T T') (ht : tags.typed = false) (hp : Flock.HmOut.parse tags cf T = .ok c) :
    Flock.ProvedScope.circuitOf tags cf T' = .ok c := by
  obtain ⟨hpc, hr⟩ := parse_both ht hp
  unfold Flock.ProvedScope.circuitOf
  split
  · exact parseCommit_okLe hext _ _ c hpc
  · exact parse_okLe hext _ _ _ _ c hr

/-! ## What the statement's checks are -/

/-- **The files' check and the population check of the statement `statementOf` read**: `PortRoots.checkOf` at the
circuit read with the statement's inputs' tables, on its public file, and `population` on its public file. -/
theorem statementOf_checks {p : _root_.Spot.OneStageZk.Own} {population : ByteArray → Flock.V Unit}
    {bs : _root_.Core.Blobs} {a : _root_.Spot.OneStageZk.StepAt}
    (h : _root_.Spot.OneStageZk.statementOf p population bs = .ok a) :
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
          have ha : _root_.Spot.OneStageZk.StepAt.ofOwn p tags0 circuitB pubB bs rd (population pubB) = a :=
            Except.ok.inj h
          subst ha
          exact ⟨rfl, rfl⟩

/-- **The files' check and the population check of a member's statement** (`statementOf_checks`, at the member's
own inputs and its population check at its slice). -/
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
  exact statementOf_checks hs

/-! ## The public file's count -/

/-- A member's population check that passes read the public file's header and held it to the slice. -/
theorem population_header {vm : _root_.Spot.OneStageZk.Member} {slice : Json} {pubB : ByteArray}
    (h : vm.population slice pubB = .ok ()) :
    ∃ hd, _root_.Spot.FilePart.header pubB = .ok hd ∧ _root_.Spot.OneStageZk.headerHeld slice hd = .ok () := by
  unfold _root_.Spot.OneStageZk.Member.population at h
  obtain ⟨prog, -, h⟩ := ebind_ok h
  obtain ⟨strata, -, h⟩ := ebind_ok h
  obtain ⟨hd, hh, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  exact ⟨hd, hh, h⟩

/-- A header held to a record names the record's population as its `instances`. -/
theorem headerHeld_instances {rec hd : Json} (h : _root_.Spot.OneStageZk.headerHeld rec hd = .ok ()) :
    _root_.Spot.jeq (_root_.Spot.getD hd "instances") (_root_.Spot.getD rec "population") = true := by
  unfold _root_.Spot.OneStageZk.headerHeld at h
  dsimp only at h
  split at h
  · rename_i he
    exact Properties.Spot.Credited.population_of_matchesPublic (Array.isEmpty_iff.1 he)
  · cases h

/-- A member's slice of the registration names the member's units' count as its population. -/
theorem slice_population (vm : _root_.Spot.OneStageZk.Member) (rec : Json)
    (roots : Std.HashMap String (Array Json)) :
    _root_.Spot.getD (vm.slice rec roots) "population" = .num ⟨Int.ofNat vm.count, 0⟩ := by
  have h : (vm.slice rec roots).getObjVal? "population" = .ok (Lean.toJson vm.count) :=
    FlockComplete.JsonRT.getObjVal?_mkObj (by simp) (by simp)
  have e : _root_.Spot.getD (vm.slice rec roots) "population" = Lean.toJson vm.count := by
    simp only [_root_.Spot.getD, _root_.Spot.field?, h, Except.toOption, Option.getD_some]
  exact e

/-- **A public file the member's population check passes on loads to the member's units' count**: at the member's
slice of the registration, any load of the file has `Member.count` proof blocks. -/
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

/-! ## The member's files pass -/

/-- **A member whose files the per-member form's run held has its files pass, at the window's member**: on the
blobs sent `s`, where v2 reads the member's files among the files `R` and each pinned table the blobs hold is its
registered file, the parameters' member `vm` whose files the run held at its slice of the registration
(`Spot.OneStageZk.Member.FilesHeld`) has `FilesPass` at the window's member `m` that is `vm` on `R`. -/
theorem filesPass_of_held {R : Array ByteArray} {s : Verity.Sent} (ht : TablesSent s)
    {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ} (hon : MemberOn R vm m)
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

/-- **Every member of a window has its files pass, where the run held every member's**: at a window that is the
parameters' members on the registered files, on blobs sent where v2 reads each member's files among the registered
files, each pinned table the blobs hold is its registered file and the run held each member's files at its slice
of the registration. -/
theorem WindowOn.passes {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) {s : Verity.Sent} (ht : TablesSent s)
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

/-! ## Files that pass fix their roots' domains -/

/-- **A member whose files pass has roots that fix their domains**: the ports check, which passes at the circuit
the member's file parses to, refuses two ports of one root and different published domains
(`Flock.Bridge.v5Files_of_checks`), at the member's population, the count its public file loads to. -/
theorem FilesPass.rootsFix {nJ : ℕ} {m : Member nJ} (h : FilesPass m) : m.RootsFix := by
  obtain ⟨pubO, hO, hnO⟩ := h.loads
  have hty : (zkTags m.I).typed = false := (FlockSoundness.Discharge.ZkExec.zkTags_typed m.I).trans m.ht
  have hports := h.ports
  rw [circuitOf_flat hty (Flock.Desc.parse_flat m.hD)] at hports
  have hn : m.pub.n = m.n := (Flock.Bridge.loadPublic_inOut m.hpub hO).1.symm.trans hnO
  exact (Flock.Bridge.v5Files_of_checks m.hpub hn hports).2

/-- **A member whose files pass is `Checked` as Flock's member**, at every cost (`FilesPass.checked` at
`FilesPass.rootsFix`). -/
theorem FilesPass.checked_all {nJ : ℕ} {m : Member nJ} (h : FilesPass m) (t : ℕ) : (m.toV5 t).Checked :=
  h.checked h.rootsFix t

/-- **At a window the verifier sets up, every member's roots fix their domains**, the members the sample holds and
those it misses. -/
theorem MembersSetUp.rootsFix {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) : W.RootsFix :=
  fun m hm => (h.passes m hm).rootsFix

end Verity.Sampled.MemberForm
