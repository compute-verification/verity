/- Each member's slice names each port once (`Member.portsOnce`), wherever the per-member form's run returns or its
check accepts. -/
module

public import Spot.OneStageZk

@[expose] public section

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

theorem portTwice_nodup : ∀ {ps : List String}, portTwice ps = none → ps.Nodup
  | [], _ => List.nodup_nil
  | p :: ps, h => by
    unfold portTwice at h
    split at h
    · cases h
    · rename_i hp
      exact List.nodup_cons.2 ⟨hp, portTwice_nodup h⟩

/-- Where `Member.portsOnce` passes on the slice `s`, every root of `s` names a port as a string, and no two the same. -/
theorem Member.portsOnce_ok {m : Member} {s : Json} (h : m.portsOnce s = .ok ()) :
    ∃ ps, slicePorts s = some ps ∧ ps.Nodup := by
  unfold Member.portsOnce at h
  split at h
  · cases h
  · rename_i ps hps
    split at h
    · cases h
    · rename_i hn
      exact ⟨ps, hps, portTwice_nodup hn⟩

theorem portsOnceAll_ok {slice : Member → Json} :
    ∀ {ms : List Member}, portsOnceAll slice ms = .ok () → ∀ m ∈ ms, ∃ ps, slicePorts (slice m) = some ps ∧ ps.Nodup
  | [], _, _, hm => by cases hm
  | m0 :: ms, h, m, hm => by
    rw [portsOnceAll] at h
    obtain ⟨⟨⟩, h0, h⟩ := Flock.HmNets.bind_ok h
    rcases List.mem_cons.1 hm with rfl | hm
    · exact Member.portsOnce_ok h0
    · exact portsOnceAll_ok h m hm

/-- Where `namesOnce` passes on the members `ms`, no two of them share a name. -/
theorem namesOnce_ok {ms : Array Member} (h : namesOnce ms = .ok ()) : (ms.toList.map (·.name)).Nodup := by
  unfold namesOnce at h
  split at h
  · cases h
  · exact portTwice_nodup (by assumption)

/-- **Where `membersRootsOf` returns, every member's slice names each port once**: at the roots it returns, each
member of `ms`, sampled or not, has a slice whose every root names a port as a string, and no two the same. -/
theorem membersRootsOf_portsOnce {p : Params} {ms : Array Member} {rec : Json} {roots : Std.HashMap String (Array Json)}
    (h : membersRootsOf p ms rec = .ok roots) :
    ∀ m ∈ ms, ∃ ps, slicePorts (m.slice rec roots) = some ps ∧ ps.Nodup := by
  unfold membersRootsOf at h
  obtain ⟨roots', _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hp, h⟩ := Flock.HmNets.bind_ok h
  cases h
  exact fun m hm => portsOnceAll_ok hp m (Array.mem_toList_iff.2 hm)

/-- **Where the per-member form's run returns, every member's slice names each port once** (`membersRootsOf_portsOnce`):
at the roots `membersRootsOf` gives each member, each member of the parameters, sampled or not, has a slice of the
registration whose every root names a port as a string, and no two the same. -/
theorem membersRun_portsOnce {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    ∃ roots, membersRootsOf p ms c.record = .ok roots ∧
      ∀ m ∈ ms, ∃ ps, slicePorts (m.slice c.record roots) = some ps ∧ ps.Nodup := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨roots, hr, -⟩ := Flock.HmNets.bind_ok h
  exact ⟨roots, hr, membersRootsOf_portsOnce hr⟩

/-- **Where the per-member form's check accepts, every member's slice names each port once** (`membersRun_portsOnce`):
at the roots `membersRootsOf` gives each member, each member of the parameters, sampled or not, has a slice of the
registration whose every root names a port as a string, and no two the same. -/
theorem checkMembersAt_portsOnce {p : Params} {ms : Array Member} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkMembersAt p ms c ξ e sts bs = .accept pub prof) :
    ∃ roots, membersRootsOf p ms c.record = .ok roots ∧
      ∀ m ∈ ms, ∃ ps, slicePorts (m.slice c.record roots) = some ps ∧ ps.Nodup :=
  let ⟨_, hr, _⟩ := checkMembersAt_accept h
  membersRun_portsOnce hr

end Spot.OneStageZk
