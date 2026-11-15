module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Event

/-!
# v2's session at the registered files (`Instance.SessionAt`)

`SessionAt` asks, at each parameters `θ` where the deployment runs a session (`Instance.session θ`), that v2 set it up
on every accepted play's committed files naming the registered circuit and public files, where the play's blobs hold
each pinned table at its registered file (`TablesSent`; a play off them is a SHA-512 collision the read finder outputs,
`Binding.finder_of_tables`, which `Instance.covers` bounds); where it runs none, SPOT's published δ is `⊤`
(`Instance.deltaOf`). v2's reading of committed files (`readOf`) is their circuit and public files' and the lookup
tables the circuit file registers (`pinnedTables`), so files naming the same circuit and public files are read alike
(`read_files`, `setsUp_files`), whichever tables they hold.

`SessionAt` also asks that the session's files pass `verify --zk`'s ports check (`Live.OpensReady`). That is v2's
acceptance (`Accepted.ports`) at the tables the blobs sent hold, which extend the session's on those blobs (`FilesOn`),
and the check reads the circuit only, which more tables don't change (`Refine.portRoots_back`): `opensReady_of_accepted`.

`SessionRegistered` asks that the deployment's session at `θ`, where it runs one, be one v2 sets up on the registered
files, wherever some accepted play on the registered tables names them. The session is `liveOf`'s on the registered
files (`Instance.sessionOf`), one v2 sets up on them but for the coin commitment's families (`LiveSpec`), so what is left
is `FamiliesPinned`: at an accepted play, the family v2's parameters name is one the base's coin commitment holds
(`sessionRegistered`), of the device's `MAX_COINS` coins (`Instance.coins`). `sessionAt_of_registered` gives
`SessionAt` from `SessionRegistered`.
-/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

open FlockSoundness.Discharge
open FlockSoundness.Discharge.Recursion.Witness (toExec)

/-- **v2's acceptance gives a session's ports check** (`Live.OpensReady`): where `A` is the session v2 sets up on the
parameters and the committed files, v2's parameters name the vocabulary's digests, and a play on those files that the
sampled verdict accepts holds each pinned table at its registered file (`TablesSent`), v2 reads `A`'s circuit and
public files on its blobs with tables that extend `A`'s (`filesOn_of`), its acceptance passes the ports and roots check
there (`Accepted.ports`), and that check is `A`'s, the circuit file parsing with `A`'s tables
(`Refine.portRoots_back`, `Live.hD`). -/
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
  have hF : FilesOn A o.sent := filesOn_of hA hD hreg hθ (B := o.blobs)
    (by show o.files ++ _ = _; rw [hFo]) hp₀ (tables_registered hp₀ hv) ht
  obtain ⟨a, b, e, p, cm, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp₀ (verdict_case hv)
  have hpp : p' = p := paramsAt_case hv.2.1 hcase hacc hp₀
  subst hpp
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨tags0, circuitB, pubB, _, _, _, ht0, hc, hpb, -, -, -, hAc⟩ := checkBlobsAt_event hck
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

/-- **v2's reading of committed files is their circuit and public files'**: files naming the same circuit and public
files are read alike (`readOf`, whose tables are the ones the circuit file registers). -/
theorem read_files {θ : ByteArray} {F F' : Array ByteArray} (hc : I.base.circuitFile θ F = I.base.circuitFile θ F')
    (hp : I.base.publicFile θ F = I.base.publicFile θ F') : I.base.read θ F = I.base.read θ F' := by
  simp only [Vocab.circuitFile, Vocab.publicFile, I.programDigest, I.publicDigest] at hc hp
  rw [I.read, I.read, readOf, readOf, hc, hp]

/-- **A session v2 sets up on some files it sets up on files naming the same circuit and public files, read alike.** -/
theorem setsUp_files {θ : ByteArray} {F F' : Array ByteArray} {A : Live}
    (hc : I.base.circuitFile θ F = I.base.circuitFile θ F') (hp : I.base.publicFile θ F = I.base.publicFile θ F')
    (hr : I.base.read θ F' = I.base.read θ F) (h : SetsUp I.base θ F A) : SetsUp I.base θ F' A :=
  { h with circuit := h.circuit.trans hc, publicFile := h.publicFile.trans hp, read := hr.trans h.read }

/-- **The deployment's session at the parameters `θ`, where it runs one, is one v2 sets up on the registered files**,
wherever some play's committed files name the registered circuit and public files and its sampled verdict accepts on
blobs holding each pinned table at its registered file (`TablesSent`): a statement of `I.session θ` and the registered
files alone. -/
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

/-- **The deployment's session at the registered files** from `FamiliesPinned`: the session is `liveOf`'s there
(`Instance.sessionOf`), one v2 sets up on them but for the coin commitment's families (`liveOf_spec`), which hold the
family v2's parameters name at the accepted play (`FamiliesPinned`), of `MAX_COINS` coins (`Instance.coins`). -/
theorem sessionRegistered {k : Verity.Registration.Composite} (h : I.FamiliesPinned k) : I.SessionRegistered k := by
  intro θ A hA _ ⟨F, c, o, _, _, hreach, hv, _⟩
  rw [I.sessionOf] at hA
  exact (liveOf_spec hA).setsUp (fun p hp => h θ p hp ⟨F, c, o, hreach, hv⟩) (I.coins θ)

/-- **`SessionAt` from the deployment's session at the registered files**: the session the deployment sets up on the
registered files (`SessionRegistered`) it sets up on an accepted play's committed files, which name the same circuit
and public files (`read_files`, `setsUp_files`), and that play's acceptance passes its ports check
(`opensReady_of_accepted`). -/
theorem sessionAt_of_registered (k : Verity.Registration.Composite) (h : I.SessionRegistered k) : I.SessionAt k := by
  intro θ F A hA hP hc hp ⟨c, o, hreach, hv, ht⟩
  have hs := I.setsUp_files hc.symm hp.symm (I.read_files hc hp) (h θ A hA hP ⟨F, c, o, hc, hp, hreach, hv, ht⟩)
  exact ⟨hs, opensReady_of_accepted hs
    (digestsAt_of I.base I.pinned I.programDigest I.publicDigest I.read θ) hreach hv ht⟩

end Instance

open scoped ENNReal in
/-- **The honest instance's coin commitment holds the family v2's parameters name** (`Instance.Honest.families`): the
parameters name a challenger (`Binding.dispatch_deployed`), whose family is `labelAt θ` (`labelAt_eq`). -/
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
