module

public import Proofs.Properties.Verity.Sampled.SessionAt
public import Proofs.Flock.Bridge.DrawnLaw

/-!
# What v2's acceptance gives of the session at the registered files (`liveOf`)

The deployment's session at the parameters `θ` is `liveOf`'s on the registered files (`Instance.sessionOf`): one v2
sets up on them but for the coin commitment's families (`LiveSpec`), where one exists, and none elsewhere, where
SPOT's published δ is `⊤`. Each field of such a session is a value v2's check derives on an accepted play, and this file
derives them:

* **The parameters** (`program_of_paramsAt`): the verifier's program is the one a descriptor parses to, as v2's
  parameters decode it (`Spot.OneStage.paramsParsed`).
* **The sampling design** (`lawAt_designAt`): where v2 samples, the design `lawAt` reads off its parameters is
  `designAt`'s at the verifier's descriptor, query, law object and work table.
* **The lookup tables** (`pinnedTables_eq`): on blobs that hold the registered circuit's lookup tables and no other
  pinned table (`HoldsRegisteredTables`), each at its registered file (`TablesSent`), the tables v2 reads are the ones
  the circuit file registers.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-! ## The parameters -/

/-- **v2's parameters' verifier program is one some bytes parse to**: `paramsParsed` parses it from the canonical text
of the parameters' `program`. -/
theorem paramsParsed_program {j : Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsParsed at h
  repeat' (first
    | (obtain ⟨_, _, h⟩ := ebind_ok h)
    | (split at h)
    | (cases h; done))
  all_goals first
    | (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by assumption⟩)

theorem oneStage_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsOfJson at h
  obtain ⟨p', hp', h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact paramsParsed_program hp'

theorem oneStageZk_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.base.verifier.program := by
  unfold _root_.Spot.OneStageZk.paramsOfJson at h
  repeat' (first
    | (obtain ⟨_, _, h⟩ := ebind_ok h)
    | (split at h)
    | (cases h; done))
  all_goals first
    | (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact oneStage_paramsOfJson_program ‹_›)

/-- **v2's parameters in `θ` name a verifier program some descriptor parses to** (`paramsParsed_program`). -/
theorem program_of_paramsAt {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (hp : ParamsAt θ p) :
    ∃ desc : Claims.Msg, _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨_, -, x, -, hj⟩ := hp
  obtain ⟨_, -, hj⟩ := ebind_ok hj
  obtain ⟨_, hr⟩ := Binding.dispatch_params hj
  obtain ⟨b, hb⟩ := oneStageZk_paramsOfJson_program hr
  exact ⟨b.data.toList, hb⟩

/-! ## The sampling design -/

/-- **Where v2 samples, `lawAt` reads `designAt`'s design**: at a verifier whose program the descriptor `desc` parses
to, under the sampling design object `Draw.derive` writes, a sample `drawnFrom` makes is under `drawnLaw`'s design
(`drawsUnder_of_drawnFrom`), which is `designAt`'s there and the one `lawAt` reads: the design `deriveLaw` expands,
else the one `flockLaw` parses from the object. -/
theorem lawAt_designAt {desc : Claims.Msg} {v : _root_.Spot.Registration.Verifier} {lo : Json}
    (hp : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : _root_.Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : _root_.Spot.OneStage.drawnFrom v lo n src = .ok r) :
    lawAt v n = .ok (Flock.Bridge.designAt desc v.query v.law v.workTable n) := by
  obtain ⟨law, ⟨hl, -⟩, -⟩ := Flock.Bridge.drawsUnder_of_drawnFrom hp hlo h
  obtain ⟨p', derived, parsed, hp', hd, hpar, rfl⟩ := Flock.Bridge.drawnLaw_ok hl
  obtain rfl : p' = v.program := Except.ok.inj (hp'.symm.trans hp)
  have hdes : Flock.Bridge.designAt desc v.query v.law v.workTable n = derived.getD parsed := by
    unfold Flock.Bridge.designAt
    rw [hl]
  rw [hdes]
  have hlo' := hlo
  unfold _root_.Spot.Draw.derive at hlo'
  obtain ⟨derived', hd', hlo'⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 hlo'
  obtain rfl : derived' = derived := Except.ok.inj (hd'.symm.trans hd)
  obtain rfl : Properties.Spot.SecurityProofs.lawObj v.law derived' = lo := Except.ok.inj hlo'
  simp only [lawAt, hlo, hd, hpar, ok_bind]
  rfl

/-! ## The lookup tables -/

/-- **On blobs holding the registered circuit's lookup tables, each at its registered file, v2's tables are the ones the
circuit file registers**: a pinned table the blobs hold is one the circuit looks up (`HoldsRegisteredTables`) at its
registered file (`TablesSent`), and one they don't hold it doesn't look up. -/
theorem pinnedTables_eq {s : Verity.Sent} {circuitB : ByteArray}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB =
      .ok ((_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)).map (·.1)))
    (ht : TablesSent s) :
    pinnedTables circuitB = _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs) := by
  classical
  simp only [pinnedTables, hl]
  rw [tablesOf_ofArray]
  refine List.filterMap_congr fun t htm => ?_
  cases hf : tableFile t.sha512 s.blobs with
  | none =>
    rw [Option.map_none, if_neg]
    intro hc
    have hm : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) := by simpa using hc
    obtain ⟨y, hy, hyn⟩ := List.mem_map.1 hm
    obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
    obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
    have htt : t' = t := List.inj_on_of_nodup_map FlockSoundness.Refine.mufu_nodup ht' htm hyn
    subst htt
    rw [hf] at ha
    cases ha
  | some a =>
    have hmem : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) :=
      List.mem_map.2 ⟨(t.name, a), List.mem_filterMap.2 ⟨t, htm, by rw [hf]; rfl⟩, rfl⟩
    rw [if_pos (by simpa using hmem)]
    have hx : ∃ b : ByteArray, Flock.hexOf (Flock.Sha512.hash b) = t.sha512 := ⟨a, tableFile_hash hf⟩
    have hpf : pinnedFile t.sha512 = some hx.choose := by
      unfold pinnedFile
      rw [dif_pos hx]
    rw [hpf, ht t htm a hx.choose hf hpf]

end Verity.Sampled
