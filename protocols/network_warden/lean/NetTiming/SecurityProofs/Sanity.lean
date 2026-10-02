import NetTiming.SecurityProofs.Advice

/-!
# Satisfiability witnesses (the project's rule: no named hypothesis without one)

* `witnessWarden` satisfies W2, W3 (`B = 1`), W4, W5, W6 and J (`D = 2`) all at once. Windows have two steps. The
  jitter takes both values, rows vary with the sender's action, and a fault really does empty the rest of the window
  and fail the status.
* `witnessProgram` satisfies both dataflow conditions of Theorem 3, in the left-packed count form, and its timing
  advice really changes its rows: the conditions don't force the Program to ignore timing.
* The toy of the September draft: `(2 + 1)^6 = 729` left-packed occupancies.
-/

namespace NetTiming

open Finset

/-! ## A warden -/

/-- The witness sender's action: how many frames to release (0, 1 or 2), a jitter bit, and whether to fault. -/
structure WAct where
  count : Fin 3
  jit : Fin 2
  fault : Bool

instance : Inhabited WAct := ⟨⟨0, 0, false⟩⟩

/-- Windows of two steps: `{0, 1}, {2, 3}, …`. -/
def winEnd2 (k : ℕ) : ℕ := 2 * (k / 2) + 1

/-- The witness warden: one egress link and one status (`L = K = 1`), two slots, one-bit frames. At step `k` it
releases `count` frames `[true]`, left-packed, unless a fault happened earlier in the window. At a window's last
step, the status fails if either step of the window was a fault. -/
def witnessObs (k : ℕ) (_ : List Unit) (as : List WAct) : Obs 1 1 2 :=
  let a := as.getD k default
  let earlier := decide (winEnd2 k = k) && (as.getD (k - 1) default).fault
  { rows := fun _ s => if earlier = false ∧ (s : ℕ) < (a.count : ℕ) then some [true] else none
    jitter := a.jit
    status := fun _ => if winEnd2 k = k ∧ (a.fault || (as.getD (k - 1) default).fault) = true
      then Status.notEstablished Reason.overflow else Status.complete }

def witnessWarden : Warden 1 1 2 Unit WAct (Obs 1 1 2) where
  obs := witnessObs
  wire := witnessObs
  fault a := a.fault = true

theorem witnessWarden_assumptions :
    Regenerates witnessWarden (fun _ o => o) ∧ Padded witnessWarden 1 ∧ Packed witnessWarden ∧
      UntimedStatus witnessWarden winEnd2 ∧ FailClosed witnessWarden winEnd2 ∧ JitterBounded witnessWarden 2 := by
  refine ⟨fun _ _ _ => rfl, ?_, ?_, ?_, ?_, ?_⟩
  · -- W3
    intro k xs as ℓ s f h
    simp only [witnessWarden, witnessObs] at h
    split_ifs at h
    cases h; rfl
  · -- W4
    intro k xs as ℓ s s' hle h
    simp only [witnessWarden, witnessObs] at h ⊢
    split_ifs at h with hc
    · split_ifs with hc'
      · rfl
      · exact absurd ⟨hc.1, lt_of_le_of_lt (Fin.le_def.mp hle) hc.2⟩ hc'
    · simp at h
  · -- W5
    intro k xs as hk κ
    simp only [witnessWarden, witnessObs]
    split_ifs with hc
    · exact absurd hc.1 hk
    · rfl
  · -- W6
    intro i k xs as hi hlen hik hkw hf
    simp only [witnessWarden] at hf
    have hwin : winEnd2 i = 2 * (i / 2) + 1 := rfl
    have hgetD : as.getD i default = as[i] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]
    refine ⟨fun hlt ℓ s => ?_, fun hk => ⟨0, ?_⟩⟩
    · have hk1 : k - 1 = i := by omega
      have hend : winEnd2 k = k := by simp only [winEnd2]; omega
      have hq : (as[i]?.getD default).fault = true := by rw [List.getElem?_eq_getElem hi]; exact hf
      simp [witnessWarden, witnessObs, hend, hk1, hq]
    · have hend : winEnd2 k = k := by simp only [winEnd2]; omega
      simp only [witnessWarden, witnessObs, hend]
      have hfk : ((as.getD k default).fault || (as.getD (k - 1) default).fault) = true := by
        rcases (show i = k ∨ i = k - 1 by omega) with h | h
        · subst h; rw [hgetD, hf]; rfl
        · rw [← h, hgetD, hf]; simp
      simp only [hfk, and_self, ↓reduceIte]
      exact fun h => by cases h
  · -- J
    intro k xs as
    simp only [witnessWarden, witnessObs]
    exact (as.getD k default).jit.2

/-- The witness warden's jitter takes both values. -/
theorem witnessWarden_jitter_varies :
    (witnessWarden.obs 0 [()] [⟨0, 0, false⟩]).jitter ≠ (witnessWarden.obs 0 [()] [⟨0, 1, false⟩]).jitter := by
  simp [witnessWarden, witnessObs]

/-! ## A Program -/

/-- The witness Program over `n` steps and one link. Content advice is one bit `a.1`, and there is one timing block per
step, whose advice `a.2 t ∈ Fin 3` is the step's count. The step's row is `a.2 t` frames `[a.1]`, left-packed. -/
def witnessProgram (n : ℕ) : Program 1 2 Unit (Bool × ((β : Fin n) → Fin 3)) where
  rows t _ a _ s := if h : t < n then (if (s : ℕ) < (a.2 ⟨t, h⟩ : ℕ) then some [a.1] else none) else none

theorem witnessProgram_rowCount (n : ℕ) (t : ℕ) (ins : List Unit) (a : Bool × ((β : Fin n) → Fin 3)) (ℓ : Fin 1) :
    (rowCount ((witnessProgram n).rows t ins a ℓ) : ℕ) = if h : t < n then (a.2 ⟨t, h⟩ : ℕ) else 0 := by
  by_cases ht : t < n
  · simp only [rowCount, witnessProgram, ht, ↓reduceDIte]
    have : (univ.filter fun s : Fin 2 => (if (s : ℕ) < (a.2 ⟨t, ht⟩ : ℕ) then some [a.1] else none).isSome) =
        univ.filter fun s : Fin 2 => (s : ℕ) < (a.2 ⟨t, ht⟩ : ℕ) := by
      ext s; by_cases hs : (s : ℕ) < (a.2 ⟨t, ht⟩ : ℕ) <;> simp [hs]
    rw [this, card_filter_val_lt 2 _ (Nat.lt_succ_iff.mp (a.2 ⟨t, ht⟩).2)]
  · simp [rowCount, witnessProgram, ht]

theorem witnessProgram_conditions (n : ℕ) :
    FramesReadNoTimingAdvice (witnessProgram n) (fun c : Fin 1 × Fin n => (c.2 : ℕ))
        (fun c ins a => rowCount ((witnessProgram n).rows c.2 ins a c.1)) ∧
      OccupancyReadsOwnBlock (fun c : Fin 1 × Fin n => c.2)
        (fun c ins a => rowCount ((witnessProgram n).rows c.2 ins a c.1)) := by
  constructor
  · intro t ins a a' h1 hocc
    funext ℓ s
    by_cases ht : t < n
    · have hc := hocc (0, ⟨t, ht⟩) rfl
      have e := congrArg Fin.val hc
      simp only [witnessProgram_rowCount, ht, ↓reduceDIte] at e
      simp only [witnessProgram, ht, ↓reduceDIte, h1, Fin.ext e]
    · simp [witnessProgram, ht]
  · intro c ins a a' _ h2
    apply Fin.ext
    simp only [witnessProgram_rowCount, c.2.2, ↓reduceDIte]
    simp only at h2
    rw [h2]

/-- The witness Program's timing advice changes its rows. -/
theorem witnessProgram_timing_matters :
    (witnessProgram 1).rows 0 [] (true, fun _ => 0) ≠ (witnessProgram 1).rows 0 [] (true, fun _ => 1) := by
  intro h
  have := congrFun (congrFun h 0) 0
  simp [witnessProgram] at this

/-! ## A joint witness for the composed statement

Adopted from the red team's statement review (`internal/network-transparency/statement-review-witnesses.lean`,
bc-440a5670): an operator that reacts to what it observes, and a Program that reads that ingress, meet every hypothesis
of `encard_decodable_le_blocks` at once. With no timing advice, exactly one secret is accepted despite the feedback. -/

/-- A reacting operator: its ingress is the previous step's occupancy count. -/
def rOp (_ : ℕ) (o : List (Obs 1 1 2)) : Fin 3 :=
  match o.getLast? with
  | some x => ⟨(rowCount (x.rows 0) : ℕ), by have := (rowCount (x.rows 0)).isLt; omega⟩
  | none => 0

def rWarden : Warden 1 1 2 (Fin 3) WAct (Obs 1 1 2) where
  obs k _ as := witnessObs k [] as
  wire k _ as := witnessObs k [] as
  fault a := a.fault = true

/-- The sender releases `z k` frames at step `k`. -/
def rI (n : ℕ) : Interaction 1 1 2 (Fin n → Fin 3) (Fin 3) WAct (Obs 1 1 2) where
  warden := rWarden
  op := rOp
  snd z k _ := ⟨if h : k < n then z ⟨k, h⟩ else 0, 0, false⟩

/-- A Program that reads ingress: step `t` releases `(a.2 t + ingress t) mod 3` frames `[true]`. -/
def rP (n : ℕ) : Program 1 2 (Fin 3) (Unit × ((β : Fin n) → Fin 3)) where
  rows t ins a _ s :=
    if h : t < n then (if (s : ℕ) < ((a.2 ⟨t, h⟩ + ins.getD t 0 : Fin 3) : ℕ) then some [true] else none) else none

instance (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) (a : Unit × ((β : Fin n) → Fin 3)) :
    Decidable (Accepted (rP n) op k o a) := by
  unfold Accepted; infer_instance

instance (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) :
    Decidable (AcceptedUpTo (rP n) op k o) := by
  unfold AcceptedUpTo; infer_instance

/-- A = 0: no timing advice, and the step's count is the ingress, which the operator sets from the last count. -/
def zP (n : ℕ) : Program 1 2 (Fin 3) (Unit × ((β : Fin n) → Fin 1)) where
  rows t ins _ _ s := if t < n then (if (s : ℕ) < ((ins.getD t 0 : Fin 3) : ℕ) then some [true] else none) else none

instance (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) (a : Unit × ((β : Fin n) → Fin 1)) :
    Decidable (Accepted (zP n) op k o a) := by
  unfold Accepted; infer_instance

instance (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) :
    Decidable (AcceptedUpTo (zP n) op k o) := by
  unfold AcceptedUpTo; infer_instance

/-- With feedback and A = 0, exactly one secret is accepted: the zero-capacity case, non-vacuously. -/
theorem zP_accepts_one : ∀ z : Fin 3 → Fin 3, AcceptedUpTo (zP 3) rOp 3 ((rI 3).obsRun z 3) ↔ z = fun _ => 0 := by
  decide +kernel

/-- Acceptance reads the ingress: under a constant operator, the accepted secret moves. -/
theorem zP_constOp_accepts_one :
    ∀ z : Fin 3 → Fin 3, AcceptedUpTo (zP 3) (fun _ _ => 1) 3 ((rI 3).obsRun z 3) ↔ z = fun _ => 1 := by
  decide +kernel

theorem rP_rowCount (n t : ℕ) (ins : List (Fin 3)) (a : Unit × ((β : Fin n) → Fin 3)) (ℓ : Fin 1) :
    (rowCount ((rP n).rows t ins a ℓ) : ℕ) = if h : t < n then ((a.2 ⟨t, h⟩ + ins.getD t 0 : Fin 3) : ℕ) else 0 := by
  by_cases ht : t < n
  · simp only [rowCount, rP, ht, ↓reduceDIte]
    have : (univ.filter fun s : Fin 2 =>
          (if (s : ℕ) < ((a.2 ⟨t, ht⟩ + ins.getD t 0 : Fin 3) : ℕ) then some [true] else none).isSome) =
        univ.filter fun s : Fin 2 => (s : ℕ) < ((a.2 ⟨t, ht⟩ + ins.getD t 0 : Fin 3) : ℕ) := by
      ext s; by_cases hs : (s : ℕ) < ((a.2 ⟨t, ht⟩ + ins.getD t 0 : Fin 3) : ℕ) <;> simp_all
    rw [this, card_filter_val_lt 2 _ (Nat.lt_succ_iff.mp (a.2 ⟨t, ht⟩ + ins.getD t 0).2)]
  · simp [rowCount, rP, ht]

/-- The joint decoder: each step's row count. -/
def rDec (ws : List (Obs 1 1 2)) : Fin 3 → Fin 3 := fun k =>
  match ws[(k : ℕ)]? with
  | some x => rowCount (x.rows 0)
  | none => 0

/-- **Every hypothesis of `encard_decodable_le_blocks` at once** (n = 3): W2, J with D = 2, both dataflow conditions
in the count form, all 27 secrets accepted with a reacting operator, and a wire decoder that recovers them. -/
theorem joint_witness :
    Regenerates rWarden (fun _ o => o) ∧ JitterBounded rWarden 2 ∧
      FramesReadNoTimingAdvice (rP 3) (fun c : Fin 1 × Fin 3 => (c.2 : ℕ))
        (fun c ins a => rowCount ((rP 3).rows c.2 ins a c.1)) ∧
      OccupancyReadsOwnBlock (fun c : Fin 1 × Fin 3 => c.2)
        (fun c ins a => rowCount ((rP 3).rows c.2 ins a c.1)) ∧
      (∀ z, AcceptedUpTo (rP 3) (rI 3).op 3 ((rI 3).obsRun z 3)) ∧
      (∀ z, rDec ((rI 3).wireRun z 3) = z) := by
  refine ⟨fun _ _ _ => rfl, fun k _ as => (as.getD k default).jit.2, ?_, ?_, ?_, ?_⟩
  · intro t ins a a' _ hocc
    funext ℓ s
    by_cases ht : t < 3
    · have e := congrArg Fin.val (hocc (0, ⟨t, ht⟩) rfl)
      simp only [rP_rowCount, ht, ↓reduceDIte] at e
      simp only [rP, ht, ↓reduceDIte, e]
    · simp [rP, ht]
  · intro c ins a a' _ h2
    apply Fin.ext
    simp only [rP_rowCount, c.2.2, ↓reduceDIte]
    simp only at h2
    rw [h2]
  · show ∀ z, AcceptedUpTo (rP 3) rOp 3 ((rI 3).obsRun z 3)
    decide +kernel
  · decide +kernel

/-- The composed bound on the joint witness is non-vacuous: 27 accepted, decodable secrets, under a bound of
`1 · 2^3 · ∏ min(3, 3) = 216`. -/
theorem joint_witness_bound :
    (Set.univ : Set (Fin 3 → Fin 3)).encard = 27 ∧
      (Set.univ : Set (Fin 3 → Fin 3)).encard ≤
        ((Fintype.card Unit * 2 ^ 3 * ∏ β : Fin 3, min (Fintype.card (Fin 3))
          (∏ _c ∈ univ.filter (fun c : Fin 1 × Fin 3 => c.2 = β), Fintype.card (Fin (2 + 1)))) : ℕ) := by
  obtain ⟨hW2, hJ, h1, h2, hacc, hdec⟩ := joint_witness
  refine ⟨?_, ?_⟩
  · rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_fun, Fintype.card_fin]; rfl
  · exact encard_decodable_le_blocks (U := fun _ : Fin 1 × Fin 3 => Fin (2 + 1)) (rI 3) (rP 3) 3 2 (fun _ o => o)
      hW2 hJ (fun c => (c.2 : ℕ)) (fun c => c.2) _ h1 h2 Set.univ (fun z _ => hacc z) rDec (fun z _ => hdec z)

/-! ## The draft's toy -/

theorem toy_leftPacked : Fintype.card {P : Fin 6 → Fin 2 → Bool // LeftPacked P} = 729 := by
  rw [card_leftPacked]; rfl

end NetTiming
