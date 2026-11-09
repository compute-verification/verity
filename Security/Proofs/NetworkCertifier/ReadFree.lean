import Definitions.Core.Assumptions
import Proofs.NetworkCertifier.RecordRows
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Set.Card

/-!
# The network certifier's rows read back what crossed the wire from the record alone, and the count that gives

In the rows format (`Definitions.NetworkCertifier.RecordRows`) connection `c`'s frames over the monitoring period's
buckets, in bucket order, each with its bucket (`RowsFormat.framesOf`), are its frames as they crossed the wire, padding
included (`laneWire`), on every schedule that fits the format (`RowsFormat.framesOf_record`): the connection sends no
frames past the monitoring period's buckets, nor past its span (`laneFrames_past`), so the two read the same buckets'
frames (`flatMap_range_vanish`). So the client's view is a function of the record alone, where `RowsFormat.read`
needs the schedule too.

So a period's count charges the schedule only for its `(r, D)` (`RowsFormat.encard_framesOf_le`): the secrets a decoder
recovers from the client's view, on schedules whose `(r, D)` is in `Θ` and which fit the format, with every window's
observations in `A i`, number at most `#Θ · ∏_i #A_i`, in place of `#Ss · ∏_c (cap_c + 1) · ∏_i #A_i`; `#Θ` is slack,
at least one whenever a secret is counted. Each cell count and fail cell is then charged once, in the observations the
window's content term counts. A period statement's count is that bound over its good event, at its windows' or its
phases' sets and their sizes (`RowsFormat.encard_periodRead_le`).
-/

namespace NetTiming

variable {κ Cell : Type}

/-- Bucket by bucket over the first `M`, the lists of `g` are those over the first `N` when `g` is empty from `N` to
`M`. -/
theorem flatMap_range_vanish {β : Type} (g : ℕ → List β) {N M : ℕ} (hNM : N ≤ M)
    (h : ∀ b, N ≤ b → b < M → g b = []) : (List.range M).flatMap g = (List.range N).flatMap g := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hNM
  have h0 : ((List.range k).map (N + ·)).flatMap g = [] := List.flatMap_eq_nil_iff.2 fun b hb => by
    simp only [List.mem_map, List.mem_range] at hb
    obtain ⟨i, hi, rfl⟩ := hb
    exact h _ (Nat.le_add_right _ _) (by omega)
  rw [List.range_add, List.flatMap_append, h0, List.append_nil]

namespace RowsFormat

variable {L K : ℕ}

/-- Connection `c`'s frames over the monitoring period's `n·T` buckets, in bucket order, each with its bucket. -/
def framesOf [DecidableEq κ] {S n : ℕ} (F : RowsFormat κ Cell L K) (o : Fin n → List (Obs L K S)) (c : κ) :
    List (ℕ × CellView Cell) :=
  (List.range (n * F.T)).flatMap fun b =>
    ((F.frames o b).filterMap fun f => if f.1 = c then some f.2 else none).map (Prod.mk b)

/-- **The rows format reads back what crossed the wire from the record alone** on every schedule that fits it:
connection `c`'s frames over the monitoring period's buckets, padding included. -/
theorem framesOf_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) (s : WardenSchedule κ)
    (hs : F.Fits S n s) (a : κ → Arrivals Cell) (c : κ) :
    F.framesOf (F.record S n s a (wardenCode s a)) c = laneWire s (wardenCode s a) c := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  have hrec : F.framesOf (F.record S n s a (wardenCode s a)) c =
      (List.range (n * F.T)).flatMap fun b => (laneFrames s (wardenCode s a) c b).map (b, ·) := by
    rw [framesOf]
    refine List.flatMap_congr fun b hb => ?_
    rw [frames_record F hdec S n s a _ (List.mem_range.1 hb) ((length_bucketFrames _ _ _ _).trans_le (hs.2 _)),
      filterMap_bucketFrames hnd (hall c)]
  have hE : ∀ j, j < s.cells c → s.bucket c j < s.adm c + s.D + s.cells c := fun j hj =>
    List.mem_range.1 (bucket_mem_span s c hj)
  rw [hrec, laneWire, WardenSchedule.span,
    ← flatMap_range_vanish (fun b => (laneFrames s (wardenCode s a) c b).map (b, ·))
      (le_max_left (n * F.T) (s.adm c + s.D + s.cells c)) fun b hb _ => by
        simp only [laneFrames_past s a c (hs.1.1 c) (hs.1.2 c) hb, List.map_nil],
    ← flatMap_range_vanish (fun b => (laneFrames s (wardenCode s a) c b).map (b, ·))
      (le_max_right (n * F.T) (s.adm c + s.D + s.cells c)) fun b hb _ => by
        simp only [laneFrames_past s a c hE (fun i hi => by omega) hb, List.map_nil]]

/-- **The count the read-back gives**: secrets on schedules whose `(r, D)` is in `Θ` and which fit the format, each
recovered by `dec` from the client's view of the network certifier's run, whose code runs by `trusted-host` and records
in `F`, and whose every window's observations are in `A i`, number at most `#Θ · ∏_i #A_i`: each secret's view is its
observations' frames (`framesOf_record`). -/
theorem encard_framesOf_le (F : RowsFormat κ Cell L K) (hF : F.Wf) {S n : ℕ} {Srv Dev Z : Type}
    (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → WardenSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p =>
      (laneWire p.1 (wardenCode p.1 (arr p.2)), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2))))
    (Θ : Set (ℕ × ℕ)) (schedOf : Z → WardenSchedule κ) (srvOf : Z → Srv) (dec : (κ → List (ℕ × CellView Cell)) → Z)
    (A : Fin n → Set (List (Obs L K S))) (G : Set Z)
    (hG : ∀ z ∈ G, ((schedOf z).r, (schedOf z).D) ∈ Θ ∧ F.Fits S n (schedOf z) ∧
      dec (run tap (schedOf z, srvOf z)).1 = z ∧ ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈ A i) :
    G.encard ≤ Θ.encard * ∏ i, (A i).encard := by
  classical
  have hview : ∀ z ∈ G, (run tap (schedOf z, srvOf z)).1 = F.framesOf (run tap (schedOf z, srvOf z)).2 :=
    fun z hz => by
      rw [htrust (schedOf z, srvOf z)]
      funext c
      exact (F.framesOf_record hF S n (schedOf z) (hG z hz).2.1 (arr (srvOf z)) c).symm
  let V : Set (κ → List (ℕ × CellView Cell)) :=
    (fun p : (ℕ × ℕ) × (Fin n → List (Obs L K S)) => F.framesOf p.2) '' (Θ ×ˢ Set.pi Set.univ A)
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (schedOf z, srvOf z)).1, ⟨(((schedOf z).r, (schedOf z).D), (run tap (schedOf z, srvOf z)).2),
      ⟨(hG z hz).1, fun i _ => (hG z hz).2.2.2 i⟩, (hview z hz).symm⟩, (hG z hz).2.2.1⟩
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Θ ×ˢ Set.pi Set.univ A).encard := Set.encard_image_le _ _
    _ = Θ.encard * (Set.pi Set.univ A).encard := Set.encard_prod
    _ = Θ.encard * ∏ i, (A i).encard := by rw [Set.encard_pi_eq_prod_encard]

/-- **A monitoring period's count from the read-back**: the secrets of `Zs` that meet `Good`, each on a schedule whose
`(r, D)` is in `Θ` and which fits the format, and each recovered by `dec` from the client's view, number at most
`#Θ · ∏_i b_i` when every window's observations of such a secret are in `A i`, a set of at most `b i` sequences once
some secret meets `Good`. A period statement's count is this at its windows' sets, or its phases'. -/
theorem encard_periodRead_le (F : RowsFormat κ Cell L K) (hF : F.Wf) {S n : ℕ} {Srv Dev Z : Type}
    (arr : Srv → κ → Arrivals Cell) (tap : Dev)
    (run : Dev → WardenSchedule κ × Srv → (κ → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p =>
      (laneWire p.1 (wardenCode p.1 (arr p.2)), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2))))
    (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule κ) (srvOf : Z → Srv)
    (hcov : ∀ z ∈ Zs, schedOf z ∈ {s : WardenSchedule κ | (s.r, s.D) ∈ Θ ∧ F.Fits S n s})
    (dec : (κ → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (Good : Z → Prop) (A : Fin n → Set (List (Obs L K S))) (b : Fin n → ℕ)
    (hwin : ∀ z ∈ Zs, Good z → ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈ A i)
    (hA : {z | z ∈ Zs ∧ Good z}.Nonempty → ∀ i, (A i).encard ≤ b i) :
    {z | z ∈ Zs ∧ Good z}.encard ≤ Θ.encard * ((∏ i, b i : ℕ) : ℕ∞) := by
  set G := {z | z ∈ Zs ∧ Good z}
  rcases G.eq_empty_or_nonempty with he | hne
  · rw [he, Set.encard_empty]
    exact zero_le
  have hprod : ∏ i, (A i).encard ≤ ((∏ i, b i : ℕ) : ℕ∞) :=
    (Finset.prod_le_prod fun i _ => hA hne i).trans_eq (Nat.cast_prod _ _).symm
  calc G.encard ≤ Θ.encard * ∏ i, (A i).encard :=
        F.encard_framesOf_le hF arr tap run htrust Θ schedOf srvOf dec A G fun z hz =>
          ⟨(hcov z hz.1).1, (hcov z hz.1).2, hdz z hz.1, hwin z hz.1 hz.2⟩
    _ ≤ _ := by gcongr

end RowsFormat

end NetTiming
