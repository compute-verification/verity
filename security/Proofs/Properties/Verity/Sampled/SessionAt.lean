module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Event

/-! v2's session at the registered files (`Instance.SessionAt`). -/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

open FlockSoundness.Discharge
open FlockSoundness.Discharge.Recursion.Witness (toExec)

theorem opensReady_of_accepted {V : Vocab d α} {T : PinnedTables} {k : Verity.Registration.Composite}
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live} {o : Verity.Exchange} (hA : SetsUp V θ F A)
    (hD : DigestsAt V T θ) (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o)
    (ht : TablesSent T o.sent) : A.OpensReady := by
  obtain ⟨hθ, hFo, -⟩ := Binding.play_reaches hreach
  have hreg : V.Registered θ F := by
    have h1 := hv.1
    rw [hθ, hFo] at h1
    exact h1
  obtain ⟨p', hp', -⟩ := hA.params
  have hp₀ : ParamsAt o.sent.params p' := by
    rw [show o.sent.params = θ from hθ]
    exact hp'
  have hone : OneStatement o.params := by
    have h := hA.single
    rw [← show o.params = θ from hθ] at h
    exact h
  have hF : FilesOn A o.sent := filesOn_of hA hD hreg hθ (B := o.blobs)
    (by show o.files ++ _ = _; rw [hFo]) hp₀ (tables_registered hp₀ hone hv) ht
  obtain ⟨a, b, e, p, cm, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp₀ (verdict_case hv)
  have hpp : p' = p := paramsAt_case hv.2.1 hcase hacc hp₀
  subst hpp
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨tags0, circuitB, pubB, _, _, _, ht0, hc, hpb, -, -, -, hAc⟩ := checkBlobsAt_event (hone _ hp₀) hck
  have hcf : circuitB = A.I.circuitFile := hF.circuit p' hp₀ circuitB hc
  have hpf : pubB = A.I.publicFile := hF.publicFile p' hp₀ pubB hpb
  have htg : tags0 = A.I.tags := Option.some.inj (ht0.symm.trans hA.tags)
  have hz : ZkExec.zkTags (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p'.own circuitB pubB
      (_root_.Core.Blobs.ofArray o.sent.blobs))) = ZkExec.zkTags A.I := by
    show Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1 =
      Flock.Zk.tags A.I.tags (Flock.Zk.maskLogOf A.I.circuitFile) A.I.count
    rw [htg, hcf, hA.count]
  have h1 := hAc.ports
  rw [hz] at h1
  have hfl : (ZkExec.zkTags A.I).hiddenOutputs = true ∧ (ZkExec.zkTags A.I).isTmpl = false :=
    statement_zkTags hA.tags _ _
  have hpA := (circuitOf_parse hfl.1 hfl.2 A.I.circuitFile A.I.tables).trans (Flock.Desc.parse_flat A.hD)
  refine ⟨?_⟩
  rw [← FlockSoundness.Refine.portRoots_back hF.tables hpA, ← hcf, ← hpf]
  exact h1

namespace Instance

variable (I : Instance d α)

theorem read_files {θ : ByteArray} {F F' : Array ByteArray} (hc : I.base.circuitFile θ F = I.base.circuitFile θ F')
    (hp : I.base.publicFile θ F = I.base.publicFile θ F') : I.base.read θ F = I.base.read θ F' := by
  simp only [Vocab.circuitFile, Vocab.publicFile, I.programDigest, I.publicDigest] at hc hp
  rw [I.read, I.read, readOf, readOf, hc, hp]

theorem setsUp_files {θ : ByteArray} {F F' : Array ByteArray} {A : Live}
    (hc : I.base.circuitFile θ F = I.base.circuitFile θ F') (hp : I.base.publicFile θ F = I.base.publicFile θ F')
    (hr : I.base.read θ F' = I.base.read θ F) (h : SetsUp I.base θ F A) : SetsUp I.base θ F' A :=
  { h with circuit := h.circuit.trans hc, publicFile := h.publicFile.trans hp, read := hr.trans h.read }

def SessionRegistered (k : Verity.Registration.Composite) : Prop :=
  ∀ θ A, I.session θ = some A → (∃ p, ParamsAt θ p) →
    (∃ F c o, I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) ∧
      I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) ∧
      Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o ∧ TablesSent I.pinned o.sent) →
    SetsUp I.base θ (I.base.registered θ) A

/-- **The coin commitment's families hold the family v2's parameters name**, wherever the sampled verdict accepts a
play: the auditor's pinned designated sessions (`AuditorPins`), which the verifier checks, give it, each device role's
sessions pinned with its keys. -/
def FamiliesPinned (k : Verity.Registration.Composite) : Prop :=
  ∀ θ p, ParamsAt θ p → (∃ F c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    ∃ key label m, p.base.challenger = some (key, label) ∧ (label, m) ∈ I.base.families θ

theorem sessionRegistered {k : Verity.Registration.Composite} (h : I.FamiliesPinned k) : I.SessionRegistered k := by
  intro θ A hA _ ⟨F, c, o, _, _, hreach, hv, _⟩
  rw [I.sessionOf] at hA
  exact (liveOf_spec hA).setsUp (fun p hp => h θ p hp ⟨F, c, o, hreach, hv⟩) (I.coins θ)

theorem sessionAt_of_registered (k : Verity.Registration.Composite) (h : I.SessionRegistered k) : I.SessionAt k := by
  intro θ F A hA hP hc hp ⟨c, o, hreach, hv, ht⟩
  have hs := I.setsUp_files hc.symm hp.symm (I.read_files hc hp) (h θ A hA hP ⟨F, c, o, hc, hp, hreach, hv, ht⟩)
  exact ⟨hs, opensReady_of_accepted hs
    (digestsAt_of I.base I.pinned I.programDigest I.publicDigest I.read θ) hreach hv ht⟩

end Instance

open scoped ENNReal in
theorem Instance.Honest.familiesPinned (δ : ℝ≥0∞) (b : SessionBound) (k : Verity.Registration.Composite) :
    (Instance.Honest.inst δ b).FamiliesPinned k := by
  intro θ p hp _
  obtain ⟨_, _, _, _, hj⟩ := id hp
  obtain ⟨key, label, hch, -⟩ := Binding.dispatch_deployed hj
  refine ⟨key, label, _root_.Spot.Challenger.MAX_COINS, hch, ?_⟩
  show (label, _) ∈ [(labelAt θ, _)]
  rw [labelAt_eq hp hch]
  exact List.mem_singleton_self _

end Verity.Sampled
