import NetTiming.SecurityProofs.Advice

/-!
# Theorem 3 for the shipped schedule: the constant-rate Program

`docs/network-transparency/timing-channel.md` §2.4, for `constantRate` (`NetTiming.Protocol.Schedule`, the Program of
`schedule.py`). The advice cap needs no dataflow condition: the constant-rate Program's advice is its content advice and
one sync value per window, so `encard_accSeqs_le_sync` applies as it stands. Causality (`expectedRow_causal`) sharpens it
to the windows the horizon reaches.

The witnesses at the end show that the block cap's two conditions fail for the shipped schedule in their natural
instances, so the advice cap is the bound that holds for it, and that the windows bound is met exactly by a covert
sender.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

theorem encard_accSeqs_le_constantRate {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ W * D ^ n : ℕ) : ℕ∞) := by
  sorry

theorem encard_accSeqs_le_constantRate_fixedClock {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (hfix : ss.size = 1) (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞) := by
  sorry

theorem expectedRow_causal (S T W : ℕ) (syncs syncs' : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ)
    (hw : ∀ w : Fin W, (w : ℕ) ≤ t / T → syncs w = syncs' w) :
    expectedRow S T W syncs frames t = expectedRow S T W syncs' frames t := by
  sorry

theorem encard_accSeqs_le_constantRate_windows {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hm : n ≤ m * T) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  sorry

theorem encard_decodable_le_constantRate {Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.warden view) (hJ : JitterBounded I.warden D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  sorry

/-! ## Witnesses: the block cap's conditions fail for the schedule, and the windows bound is met -/

/-- Two sync values, `δ ∈ {0, 1}` and `ρ = 1`. -/
def twoSyncs : SyncSet where
  deltas := {0, 1}
  rhos := {1}
  deltas_nonempty := ⟨0, by simp⟩
  rhos_nonempty := ⟨1, by simp⟩
  rhos_pos := by simp only [Finset.mem_singleton, forall_eq]; decide

/-- The value `(δ, 1)` of `twoSyncs`, with `δ = 1` if `b` and `δ = 0` otherwise. -/
def twoSyncs.v (b : Bool) : twoSyncs.Elem :=
  (⟨if b then 1 else 0, by cases b <;> simp [twoSyncs]⟩, ⟨1, by simp [twoSyncs]⟩)

/-- Two frames of different content, `[true]` then `[false]`, in one session anchored at bucket 0. -/
def twoFrames : List Unit → Unit → Fin 1 → List LogicalFrame := fun _ _ _ => [⟨[true], 0, 0⟩, ⟨[false], 0, 1⟩]

/-- **Condition 1 fails in its count form.** One link with one slot, one window of three buckets and `twoFrames`.
With `δ = 0` the frames leave at buckets 0 and 1, with `δ = 1` at buckets 1 and 2. Bucket 1 holds one frame either way,
`[false]` in the first and `[true]` in the second: a row reads the counts before it, not only its own. So
`EncardAccSeqsLeBlocksCounts` never applies to this Program. -/
theorem constantRate_not_framesRead_counts :
    ¬ FramesReadNoTimingAdvice (constantRate (L := 1) (S := 1) (In := Unit) 3 1 twoSyncs twoFrames)
      (fun c : Fin 1 × Fin 3 => (c.2 : ℕ))
      (fun c ins a => rowCount ((constantRate (L := 1) (S := 1) (In := Unit) 3 1 twoSyncs twoFrames).rows c.2 ins a c.1)) := by
  intro h
  have hrow := h 1 [] ((), fun _ => twoSyncs.v false) ((), fun _ => twoSyncs.v true) rfl (by
    rintro ⟨ℓ, t⟩ (ht : (t : ℕ) = 1)
    obtain rfl : t = 1 := Fin.ext ht
    obtain rfl : ℓ = 0 := Subsingleton.elim _ _
    decide +kernel)
  have := congrFun (congrFun hrow 0) 0
  revert this; decide +kernel

/-- One frame, anchored at bucket 1. -/
def lateFrame : List Unit → Unit → Fin 1 → List LogicalFrame := fun _ _ _ => [⟨[true], 1, 0⟩]

/-- **Condition 2 fails for blocks of one window, whatever the occupancy.** One link with one slot, two windows of two
buckets and `lateFrame`, anchored in window 0. With window 0's `δ = 1` the frame leaves at bucket 2, in window 1, and
with `δ = 0` at bucket 1, whatever window 1 declares. So no occupancy, on any cells whose block is their step's window,
meets both conditions: the per-window block cap never applies, and the horizon is one block (§2.4). -/
theorem constantRate_not_windowBlocks {C : Type} {U : C → Type} (step : C → ℕ) (blk : C → Fin 2)
    (hblk : ∀ c, (blk c : ℕ) = step c / 2)
    (occ : (c : C) → List Unit → Unit × (Fin 2 → twoSyncs.Elem) → U c) :
    ¬ (FramesReadNoTimingAdvice (constantRate (L := 1) (S := 1) (In := Unit) 2 2 twoSyncs lateFrame) step occ ∧
      OccupancyReadsOwnBlock blk occ) := by
  rintro ⟨h1, h2⟩
  have hrow := h1 2 [] ((), ![twoSyncs.v false, twoSyncs.v false]) ((), ![twoSyncs.v true, twoSyncs.v false]) rfl (fun c hc =>
    h2 c [] _ _ rfl (by
      have hb : blk c = 1 := Fin.ext (by rw [hblk, hc]; rfl)
      rw [hb]; rfl))
  have := congrFun (congrFun hrow 0) 0
  revert this; decide +kernel

/-- One frame, anchored at bucket 0. -/
def oneFrame : List Unit → Unit → Fin 1 → List LogicalFrame := fun _ _ _ => [⟨[true], 0, 0⟩]

/-- The covert sender of the joint witness: its secret bit picks the bucket its one frame leaves in. -/
def syncObs (k : ℕ) (_ : List Unit) (as : List Bool) : Obs 1 1 1 :=
  { rows := fun _ _ => if decide (k = 0) = !(as.getD 0 false) then some [true] else none
    jitter := 0
    status := fun _ => Status.complete }

def syncWarden : Warden 1 1 1 Unit Bool (Obs 1 1 1) where
  obs := syncObs
  wire := syncObs
  fault _ := False

def syncInteraction : Interaction 1 1 1 Bool Unit Bool (Obs 1 1 1) where
  warden := syncWarden
  op _ _ := ()
  snd z _ _ := z

instance (k : ℕ) (op : ℕ → List (Obs 1 1 1) → Unit) (o : List (Obs 1 1 1)) (a : Unit × (Fin 1 → twoSyncs.Elem)) :
    Decidable (Accepted (constantRate (S := 1) 2 1 twoSyncs oneFrame) op k o a) := by
  unfold Accepted; infer_instance

/-- **The windows bound is met by a covert sender.** One window of two buckets, `oneFrame`, and `twoSyncs`. The
sender's secret bit delays its frame by one bucket, and each run is accepted with the matching `δ`. Every hypothesis of
`EncardDecodableLeConstantRate` holds (W2, J with `D = 1`, acceptance, a wire decoder, a horizon of `n = 2` within
`m = 1` window), and both secrets are decoded,
which is the bound `#Ac · #Σ_sync^min(W, m) · D^n = 1 · 2^1 · 1^2`. -/
theorem constantRate_joint_witness :
    Regenerates syncInteraction.warden (fun _ o => o) ∧ JitterBounded syncInteraction.warden 1 ∧
      (∀ z ∈ (Set.univ : Set Bool),
        AcceptedUpTo (constantRate (S := 1) 2 1 twoSyncs oneFrame) syncInteraction.op 2 (syncInteraction.obsRun z 2)) ∧
      (∀ z ∈ (Set.univ : Set Bool),
        (fun w : List (Obs 1 1 1) => (w.head?.map fun x => decide (x.rows 0 0 = none)).getD false)
          (syncInteraction.wireRun z 2) = z) ∧
      2 ≤ 1 * 2 ∧ (Set.univ : Set Bool).encard = ((Fintype.card Unit * twoSyncs.size ^ min 1 1 * 1 ^ 2 : ℕ) : ℕ∞) := by
  refine ⟨fun _ _ _ => rfl, fun _ _ _ => by simp [syncInteraction, syncWarden, syncObs], ?_, ?_, by decide, ?_⟩
  · intro z _
    exact ⟨((), fun _ => twoSyncs.v z), by cases z <;> decide +kernel⟩
  · intro z _
    cases z <;> decide +kernel
  · rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]
    decide +kernel

end NetTiming
