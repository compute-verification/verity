module

public import Properties.Spot.PortsHeld
public import Proofs.Properties.Spot

/-! # Proof that each member's slice root is the committed record's root for its port (`Properties.Spot.MembersRootsRecord`) -/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.RootsRecordProof

/-! ## From the per-member form's accept to R3 -/

theorem check_of_checkAt {p : OneStage.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStage.Evidence}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : OneStage.checkAt p c (some ξ) e = .accept pub prof) :
    OneStage.check p c e = .accept pub prof := by
  unfold OneStage.checkAt at h
  split at h
  · cases h
  · exact h

/-- **Where the per-member form's check accepts, the committed record keeps R3** (`acceptMeans_iff`, at v1's check on
the run's verdicts). -/
theorem r3_of_checkMembersAt {p : OneStageZk.Params} {ms : Array OneStageZk.Member} {c : OneStage.Commitment}
    {ξ : ByteArray} {e : OneStageZk.Evidence} {sts : Array OneStageZk.StatementFiles} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStageZk.checkMembersAt p ms c ξ e sts bs = .accept pub prof) : R3 c.record p.base.verifier := by
  obtain ⟨_, -, -, hc⟩ := OneStageZk.checkMembersAt_accept h
  obtain ⟨_, _, _, hreg, -⟩ := (acceptMeans_iff _ _ _ _ _).1 (check_of_checkAt hc)
  exact hreg.2.2.1

/-! ## R3: no two roots of the record name one port -/

/-- One step of `portsOf`'s fold: the root's port, unless a port `jeq` to it is already held. -/
def step (acc : Array Json) (r : Json) : Array Json :=
  if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")

theorem portsOf_eq (rec : Json) : portsOf rec = (rootsOf rec).toList.foldl step #[] := by
  rw [Array.foldl_toList]; rfl

theorem size_fold_le : ∀ (l : List Json) (acc : Array Json), (l.foldl step acc).size ≤ acc.size + l.length
  | [], acc => by simp
  | x :: l, acc => by
    simp only [List.foldl_cons, List.length_cons]
    have := size_fold_le l (step acc x)
    have hs : (step acc x).size ≤ acc.size + 1 := by unfold step; split <;> simp
    omega

theorem distinct_of_size : ∀ (l : List Json) (acc : Array Json), (l.foldl step acc).size = acc.size + l.length →
    l.Pairwise (fun a b => jeq (getD a "port") (getD b "port") = false) ∧
      ∀ r ∈ l, acc.any (jeq · (getD r "port")) = false
  | [], _, _ => by simp
  | x :: l, acc, h => by
    simp only [List.foldl_cons, List.length_cons] at h
    by_cases hx : acc.any (jeq · (getD x "port")) = true
    · have e : step acc x = acc := by simp [step, hx]
      rw [e] at h
      have := size_fold_le l acc
      omega
    · have e : step acc x = acc.push (getD x "port") := by simp [step, hx]
      rw [e] at h
      obtain ⟨hp, ha⟩ := distinct_of_size l _ (by rw [h, Array.size_push]; omega)
      refine ⟨List.pairwise_cons.2 ⟨fun y hy => ?_, hp⟩, fun r hr => ?_⟩
      · have := ha y hy
        rw [Array.any_push, Bool.or_eq_false_iff] at this
        exact this.2
      · rcases List.mem_cons.1 hr with er | hr
        · rw [er]; exact Bool.eq_false_iff.2 hx
        · have := ha r hr
          rw [Array.any_push, Bool.or_eq_false_iff] at this
          exact this.1

theorem eq_of_pairwise {R : Json → Json → Prop} {l : List Json} (hp : l.Pairwise R) (hs : ∀ a b, R a b → R b a)
    {a b : Json} (ha : a ∈ l) (hb : b ∈ l) (hab : ¬ R a b) : a = b := by
  induction l with
  | nil => cases ha
  | cons x l ih =>
    rw [List.pairwise_cons] at hp
    rcases List.mem_cons.1 ha with ea | ha <;> rcases List.mem_cons.1 hb with eb | hb
    · rw [ea, eb]
    · rw [ea] at hab; exact absurd (hp.1 b hb) hab
    · rw [eb] at hab; exact absurd (hs _ _ (hp.1 a ha)) hab
    · exact ih hp.2 ha hb

theorem jeq_symm_false {a b : Json} (h : jeq a b = false) : jeq b a = false := by
  unfold jeq at *
  cases ha : Flock.canon a <;> cases hb : Flock.canon b <;> simp only [ha, hb] at h ⊢
  rw [beq_eq_false_iff_ne] at h ⊢
  exact Ne.symm h

theorem jeq_str_self (q : String) : jeq (.str q) (.str q) = true := by
  have hs : Flock.canon (.str q) = .ok (Flock.canonStr q) := by rw [Flock.canon]; rfl
  unfold jeq
  rw [hs]
  simp

/-- **R3 names each port at most once**: two roots of the record naming one port, as a string, are one root. -/
theorem rootsOf_port_unique {rec : Json} {v : Docket.Verifier} (h3 : R3 rec v) {r0 r1 : Json} {q : String}
    (h0 : r0 ∈ rootsOf rec) (h1 : r1 ∈ rootsOf rec) (p0 : getD r0 "port" = .str q) (p1 : getD r1 "port" = .str q) :
    r1 = r0 := by
  have hsz : (portsOf rec).size = (rootsOf rec).size := h3.2.2.2.1
  rw [portsOf_eq] at hsz
  obtain ⟨hp, -⟩ := distinct_of_size (rootsOf rec).toList #[] (by rw [hsz]; simp)
  refine eq_of_pairwise hp (fun _ _ => jeq_symm_false) (Array.mem_toList_iff.2 h1) (Array.mem_toList_iff.2 h0) ?_
  rw [p0, p1, jeq_str_self]
  simp

end Properties.Spot.RootsRecordProof
