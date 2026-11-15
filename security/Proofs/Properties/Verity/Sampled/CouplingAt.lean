module

public import Proofs.Properties.Verity.Sampled.Charge
public import Proofs.Properties.Verity.Sampled.MovesReachAt

/-!
# Sampled proofs' coupling off the empty classes at their counts

The coupling holds the joint law off each stop's empty class, and pays the class's probability in the stops' term. Off
the empty classes themselves (`linkEmpty`, `roundEmpty`), that term is within the hiding term only where each stop's
opening asks for the count of coins the stop releases (`couplesErr_sampleMap_le`, under `RoundCounts` and
`LinkCounts`): at a frame of another count every class is empty, and the empty class costs 1. Off the empty classes at
the stop's count (`linkEmptyAt`, `roundEmptyAt`), a frame of another count is never held off, and the stops' term is
within the hiding term at every strategy (`couplesErr_sampleMap_le_at`).

This file holds the coupling off those events. Its coin map's premise is `sampleMap_couples` at them
(`sampleMap_couples_at`), its stops' term is `couplesErr_sampleMap_le_at`, and the coupling, assembled
(`couplingTarget_assembled_at`), takes no bound on the stops' term. What moves to the moves' reach is the count match:
`MovesReach` off the count-aware events follows from `MovesReach` off the empty classes wherever every accepted play
reached off the first is reached off the second (`CountsReach`, `movesReach_at`), which holds at the session's layout
(`Proofs.Properties.Verity.Sampled.CountsLayout`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

local notation "𝔽" => FlockLevel3.GF128

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- **The link's empty class at its count**: the stop's opening, in a play that ends in `E`, asks for both link
points' coins and none of its rounds is released as the points' bits. -/
def linkEmptyAt (E : List Release → Game Exchange) {mPts : ℕ} : LinkHold mPts := fun g' pts =>
  EmptyAtCount E g' (mPts + mPts) (linkBits e pts)

/-- **A round's empty class at its count**: the stop's opening, in a play that ends in `E`, asks for the submission's
count of coins and none of its rounds is released as the round's coins' bits. -/
def roundEmptyAt (E : List Release → Game Exchange) : RoundHold := fun g' _ s c =>
  EmptyAtCount E g' s.n (roundBits e c)

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The sample-coupled coin map's premise off the empty classes at their counts**: the joint law's tape off
`linkEmptyAt` and `roundEmptyAt` is within the openings' gaps and empty classes at the stops' counts over the stops of
the uniform tape, wherever the sample's coins are uniform (`couples_of_close`, as `sampleMap_couples`). -/
theorem sampleMap_couples_at (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (label : String) (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    CouplesWithin V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
      (linkEmptyAt e (finish (V.signer θ) θ F c σ.1)) (roundEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (couplesErr V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
        (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
          openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' s.n (roundBits e : (Fin s.n → 𝔽) → List Bool))
        (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
          openEmptyAt (finish (V.signer θ) θ F c σ.1) q.g' (A.mP + A.mP)
            (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool))) := by
  have h := couples_of_close V θ F c A rd σ D _ (linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (roundEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (openMap_roundClose e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
      (sampleStart V θ F c rd σ label after κ))
    (openMap_linkClose e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
      (sampleStart V θ F c rd σ label after κ))
    (sampleMap_fresh e V θ F c A rd σ label after κ) (sampleMap_start e V θ F c A rd σ label after κ hκ)
  rwa [zero_add] at h

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the projection with the sample-coupled coin maps, off the empty classes at their counts**:
`couplingTarget_withProject` at `sampleMaps` and the cut `P` and `Q`, off `linkEmptyAt` and `roundEmptyAt`, whose coin
map's premise holds within the stops' term (`sampleMap_couples_at`), within the hiding term at every strategy
(`couplesErr_sampleMap_le_at`), wherever the session's coins are uniform. -/
theorem couplingTarget_withProject_of_sample_at (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ A ω)) = avg H)
    (ha : LoggedBound V k (K.withProject V rd D (sampleMaps e V rd label after κ)) P θ F)
    (hq : DigestBound V k (K.withProject V rd D (sampleMaps e V rd label after κ)) A θ F Q)
    (hD : ∀ c σ, Decodes V θ F c k (fun o => P o ∧ Q c σ o) A (rd A θ F c) σ (D A θ F c σ)
      (sampleMaps e V rd label after κ A θ F c σ) (linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (roundEmptyAt e (finish (V.signer θ) θ F c σ.1))) :
    CouplingTarget V k ((K.withProject V rd D (sampleMaps e V rd label after κ)).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D (sampleMaps e V rd label after κ) A P θ F Q
    (fun c σ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
    (fun c σ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) ha hq
    (fun c σ => couples_of_within V θ F c (sampleMap_couples_at e V θ F c A (rd A θ F c) σ (D A θ F c σ)
      (label θ F c σ) (after θ F c σ) (κ A) hκ)
      (couplesErr_sampleMap_le_at e V θ F c A (rd A θ F c) σ (D A θ F c σ) (label θ F c σ) (after θ F c σ) (κ A) hκ))
    hD

omit θ F c in
/-- **The coupling, assembled, off the empty classes at their counts**: `couplingTarget_assembled` off `linkEmptyAt`
and `roundEmptyAt`, with no bound on the stops' term (`couplingTarget_withProject_of_sample_at`), wherever the outcomes
beside accepted plays on the joint law's support off those events are the writer's at the moves v2's files decode to
(`hM`). -/
theorem couplingTarget_assembled_at (K : Constructions V) (dev : Exchange → Properties.Challenger.History)
    (rd : Readings)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (hP : ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ c, Binding.SignsPlay V dev θ F c)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ A ω)) = avg H)
    (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (hW : WritesView A)
    (hM : ∀ c σ, MovesReach V θ F c k
      (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) ∧ SampledDigest k o.sent (afterAt V k θ F c σ))
      A (rd A θ F c) σ (D A θ F c σ) (sampleMaps e V rd label (afterAt V k) κ A θ F c σ)
      (linkEmptyAt e (finish (V.signer θ) θ F c σ.1)) (roundEmptyAt e (finish (V.signer θ) θ F c σ.1))) :
    CouplingTarget V k (K.assembled e dev k rd D label (afterAt V k) κ) A θ F :=
  couplingTarget_withProject_of_sample_at e V k rd ((Binding.withBinding K dev).withBind k) D label (afterAt V k) κ A
    (fun o => Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) θ F
    (fun c σ o => SampledDigest k o.sent (afterAt V k θ F c σ)) hκ
    (fun c σ => Binding.logged_tables_le V k Binding.readsV2 dev θ F c σ (Binding.binds_of_prints V k dev θ hP) (hS c))
    (digestBound_of V k _ (by rfl) A θ F) (fun c σ => decodes_of_digests hA hDg (fun _ h => h.1.2) hW (hM c σ))

/-! ## The count match on the moves' reach -/

/-- **The count match on the moves' reach**: every outcome the projection reaches off the empty classes at their
counts, beside an accepted play whose log v2 read meets `P`, it reaches off the empty classes. Off the first, a stop
whose opening asks for another count than the stop releases is not held off; this says no accepted play goes through
one whose classes are all empty. -/
def CountsReach (P : Exchange → Prop) (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) : Prop :=
  ∀ o, ReachesOff V θ F c A rd σ D M (linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) o →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (CoinTape.run (V.game θ F c) σ o.2) →
    ReachesOff V θ F c A rd σ D M (linkEmpty e (finish (V.signer θ) θ F c σ.1))
      (roundEmpty e (finish (V.signer θ) θ F c σ.1)) o

variable {e V θ F c k} in
/-- **`MovesReach` off the empty classes at their counts**, from `MovesReach` off the empty classes and the count
match on the moves' reach. -/
theorem movesReach_at {P : Exchange → Prop} {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP}
    (hM : MovesReach V θ F c k P A rd σ D M (linkEmpty e (finish (V.signer θ) θ F c σ.1))
      (roundEmpty e (finish (V.signer θ) θ F c σ.1)))
    (hC : CountsReach e V θ F c k P A rd σ D M) :
    MovesReach V θ F c k P A rd σ D M (linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
      (roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) :=
  fun o ho hacc => hM o (hC o ho hacc) hacc

end Verity.Sampled
