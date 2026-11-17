module

public import Proofs.Properties.Verity.Sampled.Window.Read
public import Proofs.Properties.Verity.Sampled.Window.UnboundLaid
public import Proofs.Properties.Verity.Sampled.Window.FilesSent
public import Proofs.Properties.Verity.Sampled.DigestTie

/-! The window the verifier sets up, at the per-member form. -/

@[expose] public section

namespace Verity.Sampled

def ExecutionPhase.table (W : ExecutionPhase) : String := (W.ms.head?.map fun m => m.I.tags.table).getD ""

namespace MemberForm

open FlockSoundness.Discharge.ZkExec (zkTags)
open Flock.Guarantees (MemberV5)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem forall₂_right {β γ : Type} {R : β → γ → Prop} : ∀ {l₁ : List β} {l₂ : List γ},
    List.Forall₂ R l₁ l₂ → ∀ b ∈ l₂, ∃ a ∈ l₁, R a b
  | _, _, .nil, _, hb => nomatch hb
  | _, _, .cons h t, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb'
    · exact ⟨_, List.mem_cons.2 (Or.inl rfl), h⟩
    · obtain ⟨a, ha, hr⟩ := forall₂_right t b hb'
      exact ⟨a, List.mem_cons.2 (Or.inr ha), hr⟩

theorem forall₂_imp {β γ : Type} {R S : β → γ → Prop} : ∀ {l₁ : List β} {l₂ : List γ},
    List.Forall₂ R l₁ l₂ → (∀ a ∈ l₁, ∀ b, R a b → S a b) → List.Forall₂ S l₁ l₂
  | _, _, .nil, _ => .nil
  | _, _, .cons h t, hi =>
    .cons (hi _ (List.mem_cons.2 (Or.inl rfl)) _ h)
      (forall₂_imp t fun a ha b => hi a (List.mem_cons.2 (Or.inr ha)) b)

noncomputable def memberRead (T : PinnedTables) (R : Array ByteArray) (vm : _root_.Spot.OneStageZk.Member) :
    Verity.FileRead :=
  ⟨zkTagsOf ((Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).getD default)
      (Verity.namedFile (digestOf vm.own.circuit) R),
    Verity.namedFile (digestOf vm.own.circuit) R, pinnedTables T (Verity.namedFile (digestOf vm.own.circuit) R),
    Verity.namedFile (digestOf vm.own.pub) R⟩

def byFirst (ms : Array _root_.Spot.OneStageZk.Member) : List _root_.Spot.OneStageZk.Member :=
  ms.toList.mergeSort fun a b => decide (a.first ≤ b.first)

noncomputable def membersRead (T : PinnedTables) (R : Array ByteArray) (θ : ByteArray) : Verity.ExecutionPhaseFiles :=
  match ((paramsOf θ).bind (·.members)).map fun ms => (byFirst ms).map (memberRead T R) with
  | some (r :: rs) => ⟨r, rs⟩
  | _ => .one (readOf T θ R)

def fileRead {nJ : ℕ} (m : Member nJ) : Verity.FileRead :=
  ⟨zkTags m.I, m.I.circuitFile, m.I.tables, m.I.publicFile⟩

theorem fileRead_toV5 {nJ : ℕ} (m : Member nJ) (t : ℕ) : (m.toV5 t).fileRead = fileRead m := rfl

structure MemberOn {nJ : ℕ} (T : PinnedTables) (F : Array ByteArray) (vm : _root_.Spot.OneStageZk.Member) (m : Member nJ) :
    Prop where
  tags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some m.I.tags
  circuit : m.I.circuitFile = Verity.namedFile (digestOf vm.own.circuit) F
  publicFile : m.I.publicFile = Verity.namedFile (digestOf vm.own.pub) F
  tables : m.I.tables = pinnedTables T m.I.circuitFile
  count : m.I.count = 1
  partition : m.I.partition = some vm.own.partition
  program : m.I.program = some vm.own.program
  work : m.I.work = none
  stratified : m.I.stratified = none
  closure : m.I.closure = none
  own : m.own = none
  units : m.n = vm.count
  place : ∀ u : Fin m.n, (m.emb u).val = vm.first + u.val

theorem MemberOn.fileRead_eq {T : PinnedTables} {F : Array ByteArray} {vm : _root_.Spot.OneStageZk.Member} {nJ : ℕ} {m : Member nJ}
    (h : MemberOn T F vm m) : fileRead m = memberRead T F vm := by
  have ht : (Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT).getD default = m.I.tags := by
    simp only [h.tags, Option.getD_some]
  rw [memberRead, ← h.circuit, ← h.publicFile, ← h.tables, ht]
  show (⟨Flock.Zk.tags m.I.tags (Flock.Zk.maskLogOf m.I.circuitFile) m.I.count, m.I.circuitFile, m.I.tables,
    m.I.publicFile⟩ : Verity.FileRead) = ⟨Flock.Zk.tags m.I.tags (Flock.Zk.maskLogOf m.I.circuitFile) 1,
      m.I.circuitFile, m.I.tables, m.I.publicFile⟩
  rw [h.count]

structure ExecutionPhaseOn (V : Vocab d α) (T : PinnedTables) (θ : ByteArray) (F : Array ByteArray) (W : ExecutionPhase) : Prop where
  members : ∃ (p : _root_.Spot.OneStageZk.Params) (ms : Array _root_.Spot.OneStageZk.Member)
      (vms : List _root_.Spot.OneStageZk.Member),
    ParamsAt θ p ∧ p.members = some ms ∧ vms.Perm ms.toList ∧
      vms.Pairwise (fun a b => a.first < b.first) ∧ List.Forall₂ (MemberOn T (V.registered θ)) vms W.ms ∧
      samplingDesignAt p.base.verifier W.nJ = .ok W.samplingDesign ∧
      ∃ key label, p.base.challenger = some (key, label) ∧
        (label, _root_.Spot.Challenger.MAX_COINS) ∈ V.families θ
  lb : W.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64
  read : ∃ (m₀ : Member W.nJ) (rest : List (Member W.nJ)), W.ms = m₀ :: rest ∧
    V.read θ F = ⟨fileRead m₀, rest.map fileRead⟩
  share : (V.certOf θ).ε = W.ε
  units : V.units θ F = W.nJ

theorem ExecutionPhaseOn.tags {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (h : ExecutionPhaseOn V T θ F W)
    (m : Member W.nJ) (hm : m ∈ W.ms) : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some m.I.tags := by
  obtain ⟨_, _, _, -, -, -, -, hall, -⟩ := h.members
  obtain ⟨_, -, hon⟩ := forall₂_right hall m hm
  exact hon.tags

theorem ExecutionPhaseOn.membersAt {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : ExecutionPhaseOn V T θ F W) : MembersAt W W.table := by
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
      simp only [ExecutionPhase.table, hW, List.head?_cons, Option.map_some, Option.getD_some]
    exact (congrArg Flock.Tags.table ht).trans hT.symm

theorem ExecutionPhaseOn.of_read {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F F' : Array ByteArray} {W : ExecutionPhase}
    (h : ExecutionPhaseOn V T θ F W) (hr : V.read θ F = V.read θ F') : ExecutionPhaseOn V T θ F' W := by
  obtain ⟨m₀, rest, hW, hread⟩ := h.read
  have hu : V.units θ F' = V.units θ F := congrArg Verity.ExecutionPhaseFiles.units hr.symm
  exact ⟨h.members, h.lb, ⟨m₀, rest, hW, hr.symm.trans hread⟩, h.share, hu.trans h.units⟩

structure FilesPass {nJ : ℕ} (m : Member nJ) : Prop where
  loads : ∃ pubO : Flock.Public,
    Flock.HmOut.loadPublic (zkTags m.I) m.D.unroll m.I.publicFile = .ok pubO ∧ pubO.n = m.n
  distinct : Flock.PortRoots.dupId m.D.unroll m.pub = false
  ports : Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (zkTags m.I) m.I.circuitFile m.I.tables)
    (zkTags m.I) m.I.publicFile = .ok ()

theorem FilesPass.checked {nJ : ℕ} {m : Member nJ} (h : FilesPass m) (hfix : m.RootsFix) (t : ℕ) :
    (m.toV5 t).Checked := by
  obtain ⟨pubO, hp, hn⟩ := h.loads
  have hparsed := Flock.Bridge.parsedOf_eq m.hD hp m.hU
  rw [hn] at hparsed
  exact ⟨rfl, hparsed, h.distinct, hfix⟩

structure MembersSetUp (V : Vocab d α) (T : PinnedTables) (θ : ByteArray) (F : Array ByteArray) (W : ExecutionPhase) : Prop
    extends ExecutionPhaseOn V T θ F W where
  laid : W.Laid
  passes : ∀ m ∈ W.ms, FilesPass m

theorem MembersSetUp.checked {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase}
    (h : MembersSetUp V T θ F W) (hfix : W.RootsFix) (t : ℕ) {M : MemberV5 ByteArray W.nJ}
    (hM : M ∈ W.ms.map fun m => m.toV5 t) : M.Checked := by
  obtain ⟨m, hm, rfl⟩ := List.mem_map.1 hM
  exact (h.passes m hm).checked (hfix m hm) t

end MemberForm

end Verity.Sampled
