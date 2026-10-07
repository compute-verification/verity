import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-!
# Lemma C at the `--zk` wire

`CROnly.adaptive_prefinal_gap` at the `--zk` tables (`Coined.tab`), the honest prover's messages `msgsAt` and the leaves'
structure `H(x ⊕ M·y, c(y))` (`leafSplit`, through `HonestMsg`):

* `hP` holds by definition: `pairProver` is `msgsAt` at the leaves `H(dig + B, c)`.
* `hI` holds by definition: at ideal leaves `H(U, c(y))`, `Coined.preWith` is `Table.preView` at the randomness
  `Coined.randOf` builds from the base and the leaves' seeds (`Coined.preWith_ideal`), since `Table.preView` reads the
  seeds only in its leaves.
* `he` with `k = 1`: `randAt T c` is a bijection (`randAt_bijective`), each table's randomness being its base and its
  pre-final leaves' seeds placed by `T.s0`, `T.sP` (`Coined.unRand` reads them back).
-/

namespace FlockSoundness.Discharge.ZkView

open Flock ZK
open FlockSoundness.Game (prCoin)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {K W Hi VL : Type} [Field K] {z : Sizes}

/-- At ideal leaves `H(U, c(y))`, the pre-final view with its leaves given by position is `Table.preView`. -/
theorem Coined.preWith_ideal (T : Coined K W Hi VL z) (w : W) (b : Base z Hi) (u : LeafPos z → Bits × Salt) :
    T.preWith w b (fun p => leafH ((u p).1, saltHash (u p).2)) = T.tab.preView w (T.randOf b u) := rfl

/-- The base and each pre-final leaf's seed by position, read back from a table's randomness. -/
def Coined.unRand (T : Coined K W Hi VL z) (ω : T.tab.Rand) : Base z Hi × (LeafPos z → Bits × Salt) :=
  let s := ω.2.2.2.2.2.2
  ((ω.1, ω.2.1, ω.2.2.1, ω.2.2.2.1, ω.2.2.2.2.1, ω.2.2.2.2.2.1, s.1, s.2.2.2.1, s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2),
    Sum.elim (fun p => Sum.elim s.2.1 s.2.2.1 (T.s0.symm p))
      (Sum.elim (fun rp => Sum.elim (s.2.2.2.2.1 rp.1) (s.2.2.2.2.2.1 rp.1) ((T.sP rp.1).symm rp.2)) s.2.2.2.2.2.2.2.1))

theorem Coined.unRand_randOf (T : Coined K W Hi VL z) (b : Base z Hi) (u : LeafPos z → Bits × Salt) :
    T.unRand (T.randOf b u) = (b, u) := by
  refine Prod.ext rfl (funext fun p => ?_)
  rcases p with p | ⟨r, p⟩ | r
  · obtain ⟨q, rfl⟩ := T.s0.surjective p
    change Sum.elim _ _ (T.s0.symm (T.s0 q)) = _
    rw [Equiv.symm_apply_apply]
    rcases q with q | q <;> rfl
  · obtain ⟨q, rfl⟩ := (T.sP r).surjective p
    change Sum.elim _ _ ((T.sP r).symm ((T.sP r) q)) = _
    rw [Equiv.symm_apply_apply]
    rcases q with q | q <;> rfl
  · rfl

theorem Coined.randOf_injective (T : Coined K W Hi VL z) :
    Function.Injective fun x : Base z Hi × (LeafPos z → Bits × Salt) => T.randOf x.1 x.2 :=
  Function.LeftInverse.injective (g := T.unRand) fun x => T.unRand_randOf x.1 x.2

theorem Coined.randOf_surjective (T : Coined K W Hi VL z) :
    Function.Surjective fun x : Base z Hi × (LeafPos z → Bits × Salt) => T.randOf x.1 x.2 := by
  rintro ⟨m, ⟨μ0, R, μR⟩, h, padh, padμ, μ, s1, s2, s3, s4, s5, s6, s7, s8, s9⟩
  refine ⟨((m, (μ0, R, μR), h, padh, padμ, μ, s1, s4, s7, s9), fun
    | .inl p => Sum.elim s2 s3 (T.s0.symm p)
    | .inr (.inl (r, p)) => Sum.elim (s5 r) (s6 r) ((T.sP r).symm p)
    | .inr (.inr r) => s8 r), ?_⟩
  simp only [Coined.randOf, Equiv.symm_apply_apply]
  rfl

variable {J n : ℕ} {Cn : Type} {zs : Fin J → Sizes}

/-- **The session's tables' randomness at coins `c`** (Lemma C's `e c`): each table's base, and its pre-final leaves'
seeds by position. -/
noncomputable def randAt (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (c : Fin n → Cn)
    (x : (∀ j, Base (zs j) Hi) × (Leaves zs → Bits × Salt)) : ∀ j, (T c j).tab.Rand :=
  fun j => (T c j).randOf (x.1 j) fun p => x.2 ⟨j, p⟩

theorem randAt_bijective (T : (Fin n → Cn) → ∀ j, Coined K W Hi VL (zs j)) (c : Fin n → Cn) :
    Function.Bijective (randAt T c) := by
  constructor
  · intro x x' h
    have hj : ∀ j, ((x.1 j, fun p => x.2 ⟨j, p⟩) : Base (zs j) Hi × (LeafPos (zs j) → Bits × Salt))
        = (x'.1 j, fun p => x'.2 ⟨j, p⟩) := fun j =>
      (T c j).randOf_injective (a₁ := (x.1 j, fun p => x.2 ⟨j, p⟩)) (a₂ := (x'.1 j, fun p => x'.2 ⟨j, p⟩))
        (congrFun h j)
    refine Prod.ext (funext fun j => (Prod.ext_iff.1 (hj j)).1) (funext fun l => ?_)
    exact congrFun (Prod.ext_iff.1 (hj l.1)).2 l.2
  · intro r
    choose x hx using fun j => (T c j).randOf_surjective (r j)
    exact ⟨(fun j => (x j).1, fun l => (x l.1).2 l.2), funext fun j => hx j⟩

theorem card_fiber_of_bijective {α β : Type} {f : α → β} (hf : Function.Bijective f) (b : β) :
    Nat.card {x // f x = b} = 1 :=
  Nat.card_eq_one_iff_unique.2
    ⟨⟨fun x y => Subtype.ext (hf.1 (x.2.trans y.2.symm))⟩, ⟨⟨_, (hf.2 b).choose_spec⟩⟩⟩

theorem zk_wire_prefinal : ZkWirePrefinal := by
  intro K W Hi VL _ _ _ _ J n Cn Tp _ _ zs T out hc w w' hreg hreg' V G
  simp only [honest, run_comap]
  exact CROnly.adaptive_prefinal_gap (fun c j => (T c j).tab) (fun c j => (T c j).rep) My saltHash leafH
    (msgsAt T out) (digAt T) (pairProver T out hc) (fun _ _ _ _ => rfl) (randAt T) (fun _ => 1)
    (fun c => card_fiber_of_bijective (randAt_bijective T c)) out (fun _ _ _ _ => rfl) w w' hreg hreg' V G

end FlockSoundness.Discharge.ZkView
