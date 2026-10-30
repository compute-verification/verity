import Proofs.NetworkCertifier.Advice

/-!
# Satisfiability witnesses (the project's rule: no named hypothesis without one)

* `witnessCertifier` satisfies W2, W3 (`B = 1`), W4, W5, W6 and J (`D = 2`) all at once. Windows have two steps. The
  jitter takes both values, rows vary with the sender's action, and a fault really does empty the rest of the window
  and fail the status.
* `ingress_witness` meets every hypothesis of the ingress bound at once (W4's allocation `Allocated` among them), with
  27 decoded messages, and `ingress_witness_tight` shows the bound is 27: it is met exactly.
* `allocated_witness` meets them all again with an allocation of one frame on two slots (`r = 1 < S = 2`), and
  `allocated_witness_tight` shows its 8 messages meet the bound exactly, below the 27 of `r = S`.
* The toy of the September draft: `(2 + 1)^6 = 729` left-packed occupancies.
-/

namespace NetTiming

open Finset

/-! ## A certifier -/

/-- The witness sender's action: how many frames to release (0, 1 or 2), a jitter bit, and whether to fault. -/
structure WAct where
  count : Fin 3
  jit : Fin 2
  fault : Bool

instance instInhabitedWAct : Inhabited WAct := ⟨⟨0, 0, false⟩⟩

/-- Windows of two steps: `{0, 1}, {2, 3}, …`. -/
def winEnd2 (k : ℕ) : ℕ := 2 * (k / 2) + 1

/-- The witness certifier: one egress link and one status (`L = K = 1`), two slots, one-bit frames. At step `k` it
releases `count` frames `[true]`, left-packed, unless a fault happened earlier in the window. At a window's last
step, the status fails if either step of the window was a fault. -/
def witnessObs (k : ℕ) (_ : List Unit) (as : List WAct) : Obs 1 1 2 :=
  let a := as.getD k default
  let earlier := decide (winEnd2 k = k) && (as.getD (k - 1) default).fault
  { rows := fun _ s => if earlier = false ∧ (s : ℕ) < (a.count : ℕ) then some [true] else none
    jitter := a.jit
    status := fun _ => if winEnd2 k = k ∧ (a.fault || (as.getD (k - 1) default).fault) = true
      then Status.notEstablished Reason.overflow else Status.complete }

def witnessCertifier : Certifier 1 1 2 Unit WAct (Obs 1 1 2) where
  obs := witnessObs
  wire := witnessObs
  fault a := a.fault = true

theorem witnessCertifier_assumptions :
    Regenerates witnessCertifier (fun _ o => o) ∧ Padded witnessCertifier 1 ∧ Packed witnessCertifier ∧
      UntimedStatus witnessCertifier winEnd2 ∧ FailClosed witnessCertifier winEnd2 ∧ JitterBounded witnessCertifier 2 := by
  refine ⟨fun _ _ _ => rfl, ?_, ?_, ?_, ?_, ?_⟩
  · -- W3
    intro k xs as ℓ s f h
    simp only [witnessCertifier, witnessObs] at h
    split_ifs at h
    cases h; rfl
  · -- W4
    intro k xs as ℓ s s' hle h
    simp only [witnessCertifier, witnessObs] at h ⊢
    split_ifs at h with hc
    · split_ifs with hc'
      · rfl
      · exact absurd ⟨hc.1, lt_of_le_of_lt (Fin.le_def.mp hle) hc.2⟩ hc'
    · simp at h
  · -- W5
    intro k xs as hk κ
    simp only [witnessCertifier, witnessObs]
    split_ifs with hc
    · exact absurd hc.1 hk
    · rfl
  · -- W6
    intro i k xs as hi hlen hik hkw hf
    simp only [witnessCertifier] at hf
    have hwin : winEnd2 i = 2 * (i / 2) + 1 := rfl
    have hgetD : as.getD i default = as[i] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]
    refine ⟨fun hlt ℓ s => ?_, fun hk => ⟨0, ?_⟩⟩
    · have hk1 : k - 1 = i := by omega
      have hend : winEnd2 k = k := by simp only [winEnd2]; omega
      have hq : (as[i]?.getD default).fault = true := by rw [List.getElem?_eq_getElem hi]; exact hf
      simp [witnessCertifier, witnessObs, hend, hk1, hq]
    · have hend : winEnd2 k = k := by simp only [winEnd2]; omega
      simp only [witnessCertifier, witnessObs, hend]
      have hfk : ((as.getD k default).fault || (as.getD (k - 1) default).fault) = true := by
        rcases (show i = k ∨ i = k - 1 by omega) with h | h
        · subst h; rw [hgetD, hf]; rfl
        · rw [← h, hgetD, hf]; simp
      simp only [hfk, and_self, ↓reduceIte]
      exact fun h => by cases h
  · -- J
    intro k xs as
    simp only [witnessCertifier, witnessObs]
    exact (as.getD k default).jit.2

/-- The witness certifier's jitter takes both values. -/
theorem witnessCertifier_jitter_varies :
    (witnessCertifier.obs 0 [()] [⟨0, 0, false⟩]).jitter ≠ (witnessCertifier.obs 0 [()] [⟨0, 1, false⟩]).jitter := by
  simp [witnessCertifier, witnessObs]

/-! ## A reacting operator

An operator that reacts to what it observes. With no timing advice, exactly one secret is accepted despite the
feedback. -/

/-- A reacting operator: its ingress is the previous step's occupancy count. -/
def rOp (_ : ℕ) (o : List (Obs 1 1 2)) : Fin 3 :=
  match o.getLast? with
  | some x => ⟨(rowCount (x.rows 0) : ℕ), by have := (rowCount (x.rows 0)).isLt; omega⟩
  | none => 0

def rCertifier : Certifier 1 1 2 (Fin 3) WAct (Obs 1 1 2) where
  obs k _ as := witnessObs k [] as
  wire k _ as := witnessObs k [] as
  fault a := a.fault = true

/-- The sender releases `z k` frames at step `k`. -/
def rI (n : ℕ) : Interaction 1 1 2 (Fin n → Fin 3) (Fin 3) WAct (Obs 1 1 2) where
  certifier := rCertifier
  op := rOp
  snd z k _ := ⟨if h : k < n then z ⟨k, h⟩ else 0, 0, false⟩

/-- A = 0: no timing advice, and the step's count is the ingress, which the operator sets from the last count. -/
def zP (n : ℕ) : EgressPrediction 1 2 (Fin 3) (Unit × ((β : Fin n) → Fin 1)) where
  rows t ins _ _ s := if t < n then (if (s : ℕ) < ((ins.getD t 0 : Fin 3) : ℕ) then some [true] else none) else none

instance instDecidableAcceptedFinOfNatNatProdUnitForallZP (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) (a : Unit × ((β : Fin n) → Fin 1)) :
    Decidable (Accepted (zP n) op k o a) := by
  unfold Accepted; infer_instance

instance instDecidableAcceptedUpToFinOfNatNatProdUnitForallZP (n k : ℕ) (op : ℕ → List (Obs 1 1 2) → Fin 3) (o : List (Obs 1 1 2)) :
    Decidable (AcceptedUpTo (zP n) op k o) := by
  unfold AcceptedUpTo; infer_instance

/-- With feedback and A = 0, exactly one secret is accepted: the zero-capacity case, non-vacuously. -/
theorem zP_accepts_one : ∀ z : Fin 3 → Fin 3, AcceptedUpTo (zP 3) rOp 3 ((rI 3).obsRun z 3) ↔ z = fun _ => 0 := by
  decide +kernel

/-- Acceptance reads the ingress: under a constant operator, the accepted secret moves. -/
theorem zP_constOp_accepts_one :
    ∀ z : Fin 3 → Fin 3, AcceptedUpTo (zP 3) (fun _ _ => 1) 3 ((rI 3).obsRun z 3) ↔ z = fun _ => 1 := by
  decide +kernel

/-- The decoder: each step's row count. -/
def rDec (ws : List (Obs 1 1 2)) : Fin 3 → Fin 3 := fun k =>
  match ws[(k : ℕ)]? with
  | some x => rowCount (x.rows 0)
  | none => 0

/-! ## A joint witness for the ingress bound

Read inward (`NetTiming.Protocol.Ingress`): the outside's message is `z`, and its sender delivers `z k` frames `[true]`
at step `k`; X's device is `rOp`, reacting to the last count it saw. The certifier is the witness certifier without jitter,
so J holds with `D = 1`, and W4's allocation is the full two slots. Every hypothesis of
`Guarantees.EncardIngressDecodableLe` holds at once, and all `27 = (2 + 1)^3` messages are decoded: the bound is met
exactly. -/

/-- The witness certifier read inward, with no jitter. -/
def iCertifier : Certifier 1 1 2 (Fin 3) WAct (Obs 1 1 2) where
  obs k _ as := { witnessObs k [] as with jitter := 0 }
  wire k _ as := { witnessObs k [] as with jitter := 0 }
  fault a := a.fault = true

/-- The outside delivers `z k` frames at step `k`; X's device reacts to the last count. -/
def iI (n : ℕ) : Interaction 1 1 2 (Fin n → Fin 3) (Fin 3) WAct (Obs 1 1 2) where
  certifier := iCertifier
  op := rOp
  snd z k _ := ⟨if h : k < n then z ⟨k, h⟩ else 0, 0, false⟩

instance instDecidableStatusComplete {L K S : ℕ} (k : ℕ) (o : List (Obs L K S)) : Decidable (StatusComplete k o) := by
  unfold StatusComplete; infer_instance

theorem ingress_witness :
    Regenerates iCertifier (fun _ o => o) ∧ Packed iCertifier ∧ Allocated iCertifier (fun _ => 2) ∧
      JitterBounded iCertifier 1 ∧ (∀ z, StatusComplete 3 ((iI 3).obsRun z 3)) ∧
      (∀ z ℓ, linkFrames ((iI 3).obsRun z 3) ℓ <+: List.replicate 6 [true]) ∧
      (∀ z, rDec ((iI 3).wireRun z 3) = z) := by
  refine ⟨fun _ _ _ => rfl, fun k _ as ℓ => witnessCertifier_assumptions.2.2.1 k [] as ℓ,
    fun _ _ _ _ => Nat.lt_succ_iff.mp (rowCount _).2, fun _ _ _ => Nat.zero_lt_one, ?_, ?_, ?_⟩
  · decide +kernel
  · decide +kernel
  · decide +kernel

/-- The ingress bound is met exactly by the witness: 27 decoded messages, under a bound of
`(∏_ℓ (2 + 1))^3 · 1^3 = 27`. -/
theorem ingress_witness_tight :
    (Set.univ : Set (Fin 3 → Fin 3)).encard = (((∏ _ℓ : Fin 1, (2 + 1)) ^ 3 * 1 ^ 3 : ℕ) : ℕ∞) := by
  rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_fun, Fintype.card_fin]; rfl

/-! ## An allocation below the slot count

The link has two slots and an allocation of one frame: the certifier releases at most one frame per bucket, whatever the
sender asks, and the outside asks for both slots to send a 1. Every hypothesis of `Guarantees.EncardIngressDecodableLe`
holds with `r = 1 < S = 2`, and all `8 = (1 + 1)^3` messages are decoded: the bound is met exactly, and it is below the
`(2 + 1)^3 = 27` that `r = S` gives. -/

/-- The allocated certifier's network certificate: `min(count, 1)` frames `[true]`, left-packed, no jitter, complete statuses. -/
def aObs (k : ℕ) (as : List WAct) : Obs 1 1 2 where
  rows _ s := if (s : ℕ) < min ((as.getD k default).count : ℕ) 1 then some [true] else none
  jitter := 0
  status _ := Status.complete

def aCertifier : Certifier 1 1 2 (Fin 3) WAct (Obs 1 1 2) where
  obs k _ as := aObs k as
  wire k _ as := aObs k as
  fault a := a.fault = true

/-- The outside asks for two frames at step `k` when `z k = 1`, none otherwise; X's device reacts to the last count. -/
def aI (n : ℕ) : Interaction 1 1 2 (Fin n → Fin 2) (Fin 3) WAct (Obs 1 1 2) where
  certifier := aCertifier
  op := rOp
  snd z k _ := ⟨if h : k < n then (if z ⟨k, h⟩ = 1 then 2 else 0) else 0, 0, false⟩

/-- The decoder: whether each step's row holds a frame. -/
def aDec (ws : List (Obs 1 1 2)) : Fin 3 → Fin 2 := fun k =>
  match ws[(k : ℕ)]? with
  | some x => if (rowCount (x.rows 0) : ℕ) = 0 then 0 else 1
  | none => 0

theorem allocated_witness :
    Regenerates aCertifier (fun _ o => o) ∧ Packed aCertifier ∧ Allocated aCertifier (fun _ => 1) ∧
      JitterBounded aCertifier 1 ∧ (∀ z, StatusComplete 3 ((aI 3).obsRun z 3)) ∧
      (∀ z ℓ, linkFrames ((aI 3).obsRun z 3) ℓ <+: List.replicate 3 [true]) ∧
      (∀ z, aDec ((aI 3).wireRun z 3) = z) := by
  refine ⟨fun _ _ _ => rfl, ?_, ?_, fun _ _ _ => Nat.zero_lt_one, ?_, ?_, ?_⟩
  · intro k xs as ℓ s s' hle h
    simp only [aCertifier, aObs] at h ⊢
    split_ifs at h with hs'
    · have hs : (s : ℕ) < min ((as.getD k default).count : ℕ) 1 := lt_of_le_of_lt (Fin.le_def.mp hle) hs'
      simp only [hs, ↓reduceIte, Option.isSome_some]
    · simp at h
  · intro k xs as ℓ
    show (univ.filter fun s => ((aCertifier.obs k xs as).rows ℓ s).isSome).card ≤ 1
    calc _ ≤ ({0} : Finset (Fin 2)).card := by
          apply card_le_card
          intro s hs
          simp only [mem_filter, mem_univ, true_and] at hs
          simp only [aCertifier, aObs] at hs
          split_ifs at hs with h1
          · have := min_le_right ((as.getD k default).count : ℕ) 1
            exact mem_singleton.mpr (Fin.ext (by simp only [Fin.val_zero]; omega))
          · simp at hs
      _ = 1 := card_singleton _
  · decide +kernel
  · decide +kernel
  · decide +kernel

/-- The allocated witness meets its bound exactly: 8 decoded messages, under a bound of `(∏_ℓ (1 + 1))^3 · 1^3 = 8`,
which is below the `(∏_ℓ (2 + 1))^3 · 1^3 = 27` of an allocation equal to the slot count. -/
theorem allocated_witness_tight :
    (Set.univ : Set (Fin 3 → Fin 2)).encard = (((∏ _ℓ : Fin 1, (1 + 1)) ^ 3 * 1 ^ 3 : ℕ) : ℕ∞) ∧
      (∏ _ℓ : Fin 1, (1 + 1)) ^ 3 * 1 ^ 3 < (∏ _ℓ : Fin 1, (2 + 1)) ^ 3 * 1 ^ 3 := by
  refine ⟨?_, by decide⟩
  rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_fun, Fintype.card_fin]; rfl

/-! ## The draft's toy -/

theorem toy_leftPacked : Fintype.card {P : Fin 6 → Fin 2 → Bool // LeftPacked P} = 729 := by
  rw [card_leftPacked]; rfl

end NetTiming
