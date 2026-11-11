module

public import Proofs.Flock.ZkAny.RelayRun

/-!
# A round's rep is its stream's own

The `Hello` sets the session's reps and labels from the setup's rep streams (`SetupOk`), and a frame round's slot is
for a stream among the labels (`slot_round_open`). So the rep the relay steps a round of that stream through
(`streamRep`) is the stream's own rep stream's, and no other rep stream is on it:

* `repStreams_name`: a rep stream's name is its table's and rep's. `repStreams_rep_ne`: two rep streams of different
  names are on different reps.
* `any_of_extract`: a label among a prefix of the labels is among the labels.
* `streamRep_mem`: once the session's `Hello` left, a stream among its labels has its rep, as the relay reads it, in
  the setup's rep streams.
* **`streamRep_other`**: so every other rep stream of the setup is on another rep than the one a round of a stream
  among the labels is stepped through.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State)
open _root_.Flock (SessionSpec)

/-- **A rep stream's name is its table's and rep's.** -/
theorem repStreams_name {spec : SessionSpec} {i r : Nat} {s : String} (h : (i, r, s) ∈ spec.repStreams.toList) :
    s = s!"{spec.tables[i]!}/rep{r}" := by
  rw [Array.mem_toList_iff, _root_.Flock.SessionSpec.repStreams, Array.mem_flatMap] at h
  obtain ⟨i₀, -, hm⟩ := h
  rw [Array.mem_map] at hm
  obtain ⟨r₀, -, he⟩ := hm
  cases he
  rfl

/-- **Two rep streams of different names are on different reps.** -/
theorem repStreams_rep_ne {spec : SessionSpec} {i r i' r' : Nat} {s s' : String}
    (h : (i, r, s) ∈ spec.repStreams.toList) (h' : (i', r', s') ∈ spec.repStreams.toList) (hne : s' ≠ s) :
    (i', r') ≠ (i, r) := by
  intro he
  cases he
  exact hne ((repStreams_name h').trans (repStreams_name h).symm)

/-- What a prefix of an array has, the array has. -/
theorem any_of_extract {α : Type} {as : Array α} {p : α → Bool} {m n : Nat} (h : (as.extract m n).any p = true) :
    as.any p = true := by
  obtain ⟨x, hx, hp⟩ := Array.any_eq_true'.mp h
  rw [← Array.mem_toList_iff, Array.toList_extract, List.extract_eq_take_drop] at hx
  exact Array.any_eq_true'.mpr ⟨x, Array.mem_toList_iff.mp (List.mem_of_mem_drop (List.mem_of_mem_take hx)), hp⟩

/-- **A stream among the session's labels has its rep, as the relay reads it, in the setup's rep streams**, once the
session's `Hello` left. -/
theorem streamRep_mem {spec : SessionSpec} {st : State} {stream : String} (hσ : SetupOk spec st)
    (hs : st.sent = true) (ho : st.labels.any (·.1 == stream) = true) :
    ((streamRep st stream).1, (streamRep st stream).2, stream) ∈ spec.repStreams.toList := by
  obtain ⟨-, -, hr, hl⟩ := hσ hs
  unfold streamRep
  rw [hr]
  rcases hf : (streamRepsOf spec).find? (·.1 == stream) with _ | ⟨s₀, i₀, r₀⟩
  · rw [hl] at ho
    obtain ⟨y, hy, hp⟩ := Array.any_eq_true'.mp ho
    obtain ⟨⟨i, r, s⟩, hx, rfl⟩ := Array.mem_map.mp hy
    exact absurd hp (Array.find?_eq_none.mp hf (s, i, r) (Array.mem_map.mpr ⟨(i, r, s), hx, rfl⟩))
  · simp only [hf, Option.map_some, Option.getD_some]
    have hb := Array.find?_some hf
    have hz : s₀ = stream := beq_iff_eq.mp hb
    obtain ⟨⟨i, r, s⟩, hx, he⟩ := Array.mem_map.mp (Array.mem_of_find?_eq_some hf)
    cases he
    subst hz
    exact Array.mem_toList_iff.mpr hx

/-- **Every other rep stream of the setup is on another rep** than the one the relay steps a round of `stream`
through, `stream` among a prefix of the session's labels once its `Hello` left. -/
theorem streamRep_other {spec : SessionSpec} {st : State} {stream : String} {m n i r : Nat} (hσ : SetupOk spec st)
    (hs : st.sent = true) (ho : (st.labels.extract m n).any (·.1 == stream) = true)
    (hrep : streamRep st stream = (i, r)) :
    ∀ i' r' s', (i', r', s') ∈ spec.repStreams.toList → s' ≠ stream → (i', r') ≠ (i, r) := by
  intro i' r' s' h' hne
  have h := streamRep_mem hσ hs (any_of_extract ho)
  rw [hrep] at h
  exact repStreams_rep_ne h h' hne

end Claims.ZKServer.Relay
