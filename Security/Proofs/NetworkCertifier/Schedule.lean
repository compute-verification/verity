import Proofs.NetworkCertifier.Advice

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

/-- `#Σ_sync` is the number of values of the set. -/
theorem SyncSet.card_elem (ss : SyncSet) : Fintype.card ss.Elem = ss.size := by
  simp only [SyncSet.Elem, SyncSet.size, Fintype.card_prod, Fintype.card_coe]

theorem encard_accSeqs_le_constantRate {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ W * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_sync (Sync := ss.Elem) W (constantRate (S := S) T W ss frames) op n D
  rwa [SyncSet.card_elem] at this

theorem encard_accSeqs_le_constantRate_fixedClock {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (hfix : ss.size = 1) (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_constantRate (S := S) T W ss frames op n D
  rwa [hfix, one_pow, mul_one] at this

/-! ### Causality: a row reads the enqueue counts of buckets up to its own -/

theorem queue_congr (S : ℕ) {enq enq' : ℕ → ℕ} {u : ℕ} (h : ∀ s < u, enq s = enq' s) :
    queue S enq u = queue S enq' u := by
  induction u with
  | zero => rfl
  | succ u ih =>
    simp only [queue]
    rw [ih (fun s hs => h s (by omega)), h u (by omega)]

theorem released_congr (S : ℕ) {enq enq' : ℕ → ℕ} {u : ℕ} (h : ∀ s ≤ u, enq s = enq' s) :
    released S enq u = released S enq' u := by
  have hq : queue S enq u = queue S enq' u := queue_congr S fun s hs => h s hs.le
  apply Fin.ext
  simp only [released, hq, h u le_rfl]

/-- The FIFO grid over buckets `0 … T − 1` reads the enqueue counts of those buckets only. -/
theorem fifoGrid_congr (T S : ℕ) {F : Type} (f : ℕ → F) {enq enq' : ℕ → ℕ} (h : ∀ s < T, enq s = enq' s) :
    fifoGrid T S f enq = fifoGrid T S f enq' := by
  have hr : (fun u : Fin T => released S enq u) = fun u : Fin T => released S enq' u :=
    funext fun u => released_congr S fun s hs => h s (lt_of_le_of_lt hs u.2)
  simp only [fifoGrid, hr]

theorem anchor_le_eligible (f : LogicalFrame) (s : ClockSync) : f.anchor ≤ eligible f s :=
  le_trans (Nat.le_add_right _ s.delta) (Nat.le_add_right _ _)

/-- **Hand times capped at `t + 1`** read only the syncs of windows up to `t / T`: a frame anchored in a later window
is anchored after `t`, so it is eligible after `t`, and so is every frame behind it. -/
theorem handingFrom_map_min (T W t : ℕ) {syncs syncs' : Fin W → ClockSync}
    (hw : ∀ w : Fin W, (w : ℕ) ≤ t / T → syncs w = syncs' w) :
    ∀ (fs : List LogicalFrame) (last last' : ℕ), min last (t + 1) = min last' (t + 1) →
      (handingFrom T W syncs last fs).map (min · (t + 1)) = (handingFrom T W syncs' last' fs).map (min · (t + 1))
  | [], _, _, _ => rfl
  | f :: fs, last, last', hl => by
    have key : min (if hW : f.anchor / T < W then max (eligible f (syncs ⟨f.anchor / T, hW⟩)) last
          else max (max last (W * T)) f.anchor) (t + 1) =
        min (if hW : f.anchor / T < W then max (eligible f (syncs' ⟨f.anchor / T, hW⟩)) last'
          else max (max last' (W * T)) f.anchor) (t + 1) := by
      split_ifs with hW
      · by_cases hle : f.anchor / T ≤ t / T
        · rw [hw ⟨f.anchor / T, hW⟩ hle]; omega
        · have ht : t < f.anchor := by
            by_contra hc
            exact hle (Nat.div_le_div_right (by omega))
          have h1 := anchor_le_eligible f (syncs ⟨f.anchor / T, hW⟩)
          have h2 := anchor_le_eligible f (syncs' ⟨f.anchor / T, hW⟩)
          omega
      · omega
    simp only [handingFrom, List.map_cons]
    exact congrArg₂ List.cons key (handingFrom_map_min T W t hw fs _ _ key)

theorem count_eq_of_map_min {l l' : List ℕ} {t : ℕ} (h : l.map (min · (t + 1)) = l'.map (min · (t + 1)))
    {s : ℕ} (hs : s ≤ t) : l.count s = l'.count s := by
  have key : ∀ l : List ℕ, l.count s = (l.map (min · (t + 1))).count s := fun l => by
    simp only [List.count, List.countP_map]
    apply List.countP_congr
    intro x _
    simp only [Function.comp, beq_iff_eq]
    constructor <;> intro hx <;> omega
  rw [key l, key l', h]

theorem expectedRow_causal (S T W : ℕ) (syncs syncs' : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ)
    (hw : ∀ w : Fin W, (w : ℕ) ≤ t / T → syncs w = syncs' w) :
    expectedRow S T W syncs frames t = expectedRow S T W syncs' frames t := by
  have hmap := handingFrom_map_min T W t hw frames 0 0 rfl
  have hg : fifoGrid (t + 1) S (payloads frames) (enqueued T W syncs frames) =
      fifoGrid (t + 1) S (payloads frames) (enqueued T W syncs' frames) := by
    apply fifoGrid_congr
    intro s hs
    simp only [enqueued, handingBuckets]
    split_ifs
    · exact count_eq_of_map_min hmap (by omega)
    · rfl
  simp only [expectedRow, hg]

/-! ### Seven bits per window opened -/

theorem encard_accSeqs_le_constantRate_windows {Ac : Type} [Fintype Ac] (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hm : n ≤ m * T) :
    (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Ac × (Fin W → ss.Elem))
  swap
  · have : AccSeqs (constantRate (S := S) T W ss frames) op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Ac × (Fin W → ss.Elem) := Classical.choice hne
  let adv : List (Obs L K S) → Ac × (Fin W → ss.Elem) := fun o =>
    if h : AcceptedUpTo (constantRate (S := S) T W ss frames) op n o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs (constantRate (S := S) T W ss frames) op n D,
      Accepted (constantRate (S := S) T W ss frames) op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let cut : (Fin W → ss.Elem) → Fin (min W m) → ss.Elem := fun σ i =>
    σ ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩
  let code : List (Obs L K S) → (Ac × (Fin (min W m) → ss.Elem)) × (Fin n → ℕ) := fun o =>
    (((adv o).1, cut (adv o).2), jitterCode o)
  have hinj : Set.InjOn code (AccSeqs (constantRate (S := S) T W ss frames) op n D) := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨⟨hc, hσ⟩, hj⟩ := h
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · rw [hr, hr', ingressUpTo_congr op hpre]
      funext ℓ
      simp only [constantRate, hc]
      apply expectedRow_causal
      intro w hw
      have htm : t < m * T := lt_of_lt_of_le ht hm
      have hwm : (w : ℕ) < m :=
        lt_of_le_of_lt hw (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact htm))
      have hσw := congrFun hσ ⟨w, lt_min w.2 hwm⟩
      simp only [cut, Fin.eta] at hσw
      simp only [hσw]
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (AccSeqs (constantRate (S := S) T W ss frames) op n D).encard
      ≤ (((univ : Finset (Ac × (Fin (min W m) → ss.Elem))) ×ˢ
          Fintype.piFinset fun _ : Fin n => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, mem_product, mem_univ, true_and]
        exact jitterCode_mem o ho.1 ho.2.1
    _ = ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, Fintype.card_piFinset]
        simp [SyncSet.size]

theorem encard_decodable_le_constantRate {Ac : Type} [Fintype Ac]
    (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (n D m : ℕ) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T W ss frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((Fintype.card Ac * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞) := by
  calc Zs.encard
      ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo (constantRate (S := S) T W ss frames) I.op n o}.encard :=
        encard_decodable_le I _ n view hW2 Zs hacc dec hdec
    _ ≤ (AccSeqs (constantRate (S := S) T W ss frames) I.op n D).encard := by
        apply Set.encard_le_encard
        rintro o ⟨z, rfl, ha⟩
        refine ⟨I.length_obsRun z n, ?_, ha⟩
        intro x hx
        obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun z n x hx
        exact hJ k xs as
    _ ≤ _ := encard_accSeqs_le_constantRate_windows T W ss frames I.op n D m hm

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

def syncCertifier : Certifier 1 1 1 Unit Bool (Obs 1 1 1) where
  obs := syncObs
  wire := syncObs
  fault _ := False

def syncInteraction : Interaction 1 1 1 Bool Unit Bool (Obs 1 1 1) where
  certifier := syncCertifier
  op _ _ := ()
  snd z _ _ := z

instance instDecidableAcceptedUnitProdForallFinOfNatNatElemTwoSyncsConstantRateOneFrame (k : ℕ) (op : ℕ → List (Obs 1 1 1) → Unit) (o : List (Obs 1 1 1)) (a : Unit × (Fin 1 → twoSyncs.Elem)) :
    Decidable (Accepted (constantRate (S := 1) 2 1 twoSyncs oneFrame) op k o a) := by
  unfold Accepted; infer_instance

/-- **The windows bound is met by a covert sender.** One window of two buckets, `oneFrame`, and `twoSyncs`. The
sender's secret bit delays its frame by one bucket, and each run is accepted with the matching `δ`. Every hypothesis of
`EncardDecodableLeConstantRate` holds (W2, J with `D = 1`, acceptance, a wire decoder, a horizon of `n = 2` within
`m = 1` window), and both secrets are decoded,
which is the bound `#Ac · #Σ_sync^min(W, m) · D^n = 1 · 2^1 · 1^2`. -/
theorem constantRate_joint_witness :
    Regenerates syncInteraction.certifier (fun _ o => o) ∧ JitterBounded syncInteraction.certifier 1 ∧
      (∀ z ∈ (Set.univ : Set Bool),
        AcceptedUpTo (constantRate (S := 1) 2 1 twoSyncs oneFrame) syncInteraction.op 2 (syncInteraction.obsRun z 2)) ∧
      (∀ z ∈ (Set.univ : Set Bool),
        (fun w : List (Obs 1 1 1) => (w.head?.map fun x => decide (x.rows 0 0 = none)).getD false)
          (syncInteraction.wireRun z 2) = z) ∧
      2 ≤ 1 * 2 ∧ (Set.univ : Set Bool).encard = ((Fintype.card Unit * twoSyncs.size ^ min 1 1 * 1 ^ 2 : ℕ) : ℕ∞) := by
  refine ⟨fun _ _ _ => rfl, fun _ _ _ => by simp [syncInteraction, syncCertifier, syncObs], ?_, ?_, by decide, ?_⟩
  · intro z _
    exact ⟨((), fun _ => twoSyncs.v z), by cases z <;> decide +kernel⟩
  · intro z _
    cases z <;> decide +kernel
  · rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]
    decide +kernel

end NetTiming
