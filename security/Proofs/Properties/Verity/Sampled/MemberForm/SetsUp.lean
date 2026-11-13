module

public import Proofs.Properties.Verity.Sampled.Window.Read
public import Proofs.Properties.Verity.Sampled.Window.UnboundLaid
public import Proofs.Properties.Verity.Sampled.Window.FilesSent
public import Proofs.Properties.Verity.Sampled.DigestTie

/-!
# The window the verifier sets up, at the per-member form

Soundness at a window reads its window instance's `SetsUp` (`WindowInstance.SetsUp`), the window the verifier sets
up on the committed files, through two consequences (`WindowInstance.lb`, `WindowInstance.members`) and through what
the claim reads of it (`WindowInstance.ClaimReads`). This module states it at the per-member form of one-stage v2:

* **How v2 reads the members' files** (`memberRead`, `membersRead`): each member's files are the registered files its
  digests name, read as `readOf` reads one statement's; the window's are its members', in the order of their first
  units. v2 reads a member's files on the blobs sent, the committed files then the evidence's, and where those are
  the registered files (`FilesSent`, which the window's read finder pays for) it reads these.
* **A member on the files** (`MemberOn`): the window's member is one of the parameters' members
  (`Spot.OneStageZk.Member`) on files: the statement's tags, the circuit and public files the member's digests name,
  the lookup tables the circuit file registers, one table, the member's partition object and program as flock-verify's
  inputs, no work, stratified or closure input and no recomputation of public input words, as
  `Spot.OneStageZk.inputsOf` hands them, and the member's units, a range of the window's population from its first.
  Its files are then read as `memberRead` reads them (`MemberOn.fileRead_eq`).
* **The window** (`WindowOn`): the window's members are the parameters', in the order of their first units, each on
  the registered files; the sample's law is the verifier's over the window's population, and its bytes `MAX_COINS`
  coins of the parameters' challenger family; the vocabulary's read is the members' files in that order; the
  published share is the window's and the files' units its population.
* **A member's files pass the files check** (`FilesPass`): what `Flock.Guarantees.MemberV5.Checked` asks of a
  member's files beyond what a window's member fixes, and the ports check's pass.
* **The verifier sets the window up** (`MembersSetUp`): the window is on the files, laid out in order, and every
  member's files pass, with a proof block in the sample or not. It gives `MAX_COINS` frames' bytes and members of the
  window's one table (`WindowOn.lb`, `WindowOn.membersAt`), and that every member is `Checked` wherever the members'
  roots fix their domains (`MembersSetUp.checked`).

`WindowOn` reads the committed files only through the vocabulary's read, so it moves between files read alike
(`WindowOn.of_read`); the layout and the files check read the window alone.
-/

@[expose] public section

namespace Verity.Sampled

/-- **The window's table's name**: its first member's. Every member's, at a window whose members have one table
(`MembersAt`). -/
def Window.table (W : Window) : String := (W.ms.head?.map fun m => m.I.tags.table).getD ""

namespace MemberForm

open FlockSoundness.Discharge.ZkExec (zkTags)
open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## Lists related member by member -/

/-- Each of the second of two lists related member by member is related to one of the first. -/
theorem forall₂_right {β γ : Type} {R : β → γ → Prop} : ∀ {l₁ : List β} {l₂ : List γ},
    List.Forall₂ R l₁ l₂ → ∀ b ∈ l₂, ∃ a ∈ l₁, R a b
  | _, _, .nil, _, hb => nomatch hb
  | _, _, .cons h t, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb'
    · exact ⟨_, List.mem_cons.2 (Or.inl rfl), h⟩
    · obtain ⟨a, ha, hr⟩ := forall₂_right t b hb'
      exact ⟨a, List.mem_cons.2 (Or.inr ha), hr⟩

/-- Two lists related member by member are related by whatever the relation gives at the first's members. -/
theorem forall₂_imp {β γ : Type} {R S : β → γ → Prop} : ∀ {l₁ : List β} {l₂ : List γ},
    List.Forall₂ R l₁ l₂ → (∀ a ∈ l₁, ∀ b, R a b → S a b) → List.Forall₂ S l₁ l₂
  | _, _, .nil, _ => .nil
  | _, _, .cons h t, hi =>
    .cons (hi _ (List.mem_cons.2 (Or.inl rfl)) _ h)
      (forall₂_imp t fun a ha b => hi a (List.mem_cons.2 (Or.inr ha)) b)

/-! ## How v2 reads the members' files -/

/-- **How v2 reads the member `vm`'s files among the files `R`**: `readOf` at the member's digests, the circuit and
public files they name among `R` (`Verity.namedFile`), the lookup tables the circuit file registers
(`pinnedTables`) and `verify --zk`'s tags at the statement's tags and the circuit file (`zkTagsOf`). -/
noncomputable def memberRead (R : Array ByteArray) (vm : _root_.Spot.OneStageZk.Member) : Verity.FileRead :=
  ⟨zkTagsOf ((Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).getD default)
      (Verity.namedFile (digestOf vm.own.circuit) R),
    Verity.namedFile (digestOf vm.own.circuit) R, pinnedTables (Verity.namedFile (digestOf vm.own.circuit) R),
    Verity.namedFile (digestOf vm.own.pub) R⟩

/-- The members `ms` in the order of their first units. -/
def byFirst (ms : Array _root_.Spot.OneStageZk.Member) : List _root_.Spot.OneStageZk.Member :=
  ms.toList.mergeSort fun a b => decide (a.first ≤ b.first)

/-- **How v2 reads a window's files among the registered files `R`**, at the parameters `θ`: the members' files
(`memberRead`), in the order of their first units (`byFirst`), where the parameters name members; else one
statement's (`readOf`). -/
noncomputable def membersRead (R : Array ByteArray) (θ : ByteArray) : Verity.WindowFiles :=
  match ((paramsOf θ).bind (·.members)).map fun ms => (byFirst ms).map (memberRead R) with
  | some (r :: rs) => ⟨r, rs⟩
  | _ => .one (readOf θ R)

/-! ## A member on the files -/

/-- **How v2 reads a window's member's files**: `verify --zk`'s tags at its inputs, its circuit file, its lookup
tables and its public file, Flock's member's read at every cost (`fileRead_toV5`). -/
def fileRead {nJ : ℕ} (m : Member nJ) : Verity.FileRead :=
  ⟨zkTags m.I, m.I.circuitFile, m.I.tables, m.I.publicFile⟩

/-- A window's member's read is Flock's member's, at any budget. -/
theorem fileRead_toV5 {nJ : ℕ} (m : Member nJ) (t : ℕ) : (m.toV5 t).fileRead = fileRead m := rfl

/-- **The window's member `m` is the verifier's member `vm` on the files `F`**: the statement's tags; the circuit and
public files `vm`'s digests name among `F` (`Verity.namedFile`); the lookup tables the circuit file registers
(`pinnedTables`); one table; `vm`'s partition object and program as flock-verify's inputs; no work, stratified or
closure input and no recomputation of public input words; and `vm`'s units, as many as `m`'s proof blocks, each at
its place of the window's population after `vm`'s first unit. -/
structure MemberOn {nJ : ℕ} (F : Array ByteArray) (vm : _root_.Spot.OneStageZk.Member) (m : Member nJ) :
    Prop where
  tags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some m.I.tags
  circuit : m.I.circuitFile = Verity.namedFile (digestOf vm.own.circuit) F
  publicFile : m.I.publicFile = Verity.namedFile (digestOf vm.own.pub) F
  tables : m.I.tables = pinnedTables m.I.circuitFile
  count : m.I.count = 1
  partition : m.I.partition = some vm.own.partition
  program : m.I.program = some vm.own.program
  work : m.I.work = none
  stratified : m.I.stratified = none
  closure : m.I.closure = none
  own : m.own = none
  units : m.n = vm.count
  place : ∀ u : Fin m.n, (m.emb u).val = vm.first + u.val

/-- **A member on the files is read as v2 reads its files there** (`memberRead`). -/
theorem MemberOn.fileRead_eq {F : Array ByteArray} {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ}
    (h : MemberOn F vm m) : fileRead m = memberRead F vm := by
  have ht : (Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).getD default = m.I.tags := by
    simp only [h.tags, Option.getD_some]
  rw [memberRead, ← h.circuit, ← h.publicFile, ← h.tables, ht]
  show (⟨Flock.Zk.tags m.I.tags (Flock.Zk.maskLogOf m.I.circuitFile) m.I.count, m.I.circuitFile, m.I.tables,
    m.I.publicFile⟩ : Verity.FileRead) = ⟨Flock.Zk.tags m.I.tags (Flock.Zk.maskLogOf m.I.circuitFile) 1,
      m.I.circuitFile, m.I.tables, m.I.publicFile⟩
  rw [h.count]

/-! ## The window -/

/-- **The window `W` is the parameters' members on the registered files, read on the committed files `F`**: v2's
parameters in `θ` name members, and `W`'s are those, in the order of their first units, each on the registered files
(`MemberOn`); the sample's law is the verifier's over `W`'s population (`lawAt`) and its bytes `MAX_COINS` coins of
the parameters' challenger family, 64 bytes each; the vocabulary reads `F` as the members' files in that order; and
the published share is `W`'s and `F`'s units `W`'s population. -/
structure WindowOn (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (W : Window) : Prop where
  members : ∃ (p : _root_.Spot.OneStageZk.Params) (ms : Array _root_.Spot.OneStageZk.Member)
      (vms : List _root_.Spot.OneStageZk.Member),
    ParamsAt θ p ∧ p.members = some ms ∧ vms.Perm ms.toList ∧
      vms.Pairwise (fun a b => a.first < b.first) ∧ List.Forall₂ (MemberOn (V.registered θ)) vms W.ms ∧
      lawAt p.base.verifier W.nJ = .ok W.law ∧
      ∃ key label, p.base.challenger = some (key, label) ∧
        (label, _root_.Spot.Challenger.MAX_COINS) ∈ V.families θ
  lb : W.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64
  read : ∃ (m₀ : Member W.nJ) (rest : List (Member W.nJ)), W.ms = m₀ :: rest ∧
    V.read θ F = ⟨fileRead m₀, rest.map fileRead⟩
  share : (V.certOf θ).ε = W.ε
  units : V.units θ F = W.nJ

/-- Each member of a window on the files has the statement's tags. -/
theorem WindowOn.tags {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window} (h : WindowOn V θ F W)
    (m : Member W.nJ) (hm : m ∈ W.ms) : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some m.I.tags := by
  obtain ⟨_, _, _, -, -, -, -, hall, -⟩ := h.members
  obtain ⟨_, -, hon⟩ := forall₂_right hall m hm
  exact hon.tags

/-- **A window on the files has members of its one table with hm96-sha512 rows** (`MembersAt` at `Window.table`):
each has the statement's tags and one table. -/
theorem WindowOn.membersAt {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) : MembersAt W W.table := by
  refine ⟨fun m hm => ?_, fun m hm => statement_hm96Rows (h.tags m hm), fun m hm => ?_⟩
  · obtain ⟨_, _, _, -, -, -, -, hall, -⟩ := h.members
    obtain ⟨_, -, hon⟩ := forall₂_right hall m hm
    exact hon.count
  · obtain ⟨m₀, rest, hW, -⟩ := h.read
    have hm₀ : m₀ ∈ W.ms := by
      rw [hW]
      exact List.mem_cons.2 (Or.inl rfl)
    have ht : m.I.tags = m₀.I.tags := Option.some.inj ((h.tags m hm).symm.trans (h.tags m₀ hm₀))
    have hT : W.table = m₀.I.tags.table := by
      simp only [Window.table, hW, List.head?_cons, Option.map_some, Option.getD_some]
    exact (congrArg Flock.Tags.table ht).trans hT.symm

/-- **A window on some files is on files read alike.** -/
theorem WindowOn.of_read {V : Vocab d α} {θ : ByteArray} {F F' : Array ByteArray} {W : Window}
    (h : WindowOn V θ F W) (hr : V.read θ F = V.read θ F') : WindowOn V θ F' W := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have hu : V.units θ F' = V.units θ F := congrArg Verity.WindowFiles.units hr.symm
  exact ⟨h.members, h.lb, ⟨m₀, rest, hW, hr.symm.trans hread⟩, h.share, hu.trans h.units⟩

/-! ## The files check at a member -/

/-- **The window's member's files pass the verifier's files check**, beyond what the member fixes (its circuit
file's parse, the circuit's empty public input ports and the public file's load over them): the public file loads
as the committed files' circuit reads it, to the member's population; no two of the circuit's ports have one
published domain; and the ports check passes at the circuit file (`Flock.PortRoots.checkOf`). -/
structure FilesPass {nJ : ℕ} (m : Member nJ) : Prop where
  loads : ∃ pubO : Flock.Public,
    Flock.HmOut.loadPublic (zkTags m.I) m.D.unroll m.I.publicFile = .ok pubO ∧ pubO.n = m.n
  distinct : Flock.PortRoots.dupId m.D.unroll m.pub = false
  ports : Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (zkTags m.I) m.I.circuitFile m.I.tables)
    (zkTags m.I) m.I.publicFile = .ok ()

/-- **A member whose files pass and whose roots fix their domains is `Checked` as Flock's member**, at every
cost. -/
theorem FilesPass.checked {nJ : ℕ} {m : Member nJ} (h : FilesPass m) (hfix : m.RootsFix) (t : ℕ) :
    (m.toV5 t).Checked := by
  obtain ⟨pubO, hp, hn⟩ := h.loads
  have hparsed := Flock.Bridge.parsedOf_eq m.hD hp m.hU
  rw [hn] at hparsed
  exact ⟨rfl, hparsed, h.distinct, hfix⟩

/-! ## The verifier sets the window up -/

/-- **The verifier sets the window `W` up on the committed files `F`**, at the per-member form: `W` is the
parameters' members on the registered files, read on `F` (`WindowOn`), laid out in order (`Window.Laid`), and every
member's files pass the files check (`FilesPass`), the members the sample holds and those it misses. -/
structure MembersSetUp (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (W : Window) : Prop
    extends WindowOn V θ F W where
  laid : W.Laid
  passes : ∀ m ∈ W.ms, FilesPass m

/-- **At a window the verifier sets up whose members' roots fix their domains, every member of Flock's window is
`Checked`**, at every cost. -/
theorem MembersSetUp.checked {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {W : Window}
    (h : MembersSetUp V θ F W) (hfix : W.RootsFix) (t : ℕ) {M : MemberV5 ByteArray W.nJ}
    (hM : M ∈ W.ms.map fun m => m.toV5 t) : M.Checked := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact (h.passes m hm).checked (hfix m hm) t

end MemberForm

end Verity.Sampled
