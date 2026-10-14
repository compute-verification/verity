import Mathlib
import Proofs.Accounting.Bandwidth.Ticked
import Proofs.Accounting.Bandwidth.Joint

/-!
# Bandwidth adds, by counting (Mbw), for the chip as a tick transducer

`AuditJoint` for the chip of `Ticked.lean`, the audited location read tick by tick, when its view determines `z`: a
decoder `ζ` recovers `z` from the chip's state at the window's start and what it observed in the window (HBM's
replies, the rest's signals and the oracle's answers), section 5's "the chip's view (state at the seed, its links'
deliveries in the window) determines `z`". The state, HBM and the rest may depend on `z` as well as the store; the rest
ignores the oracle (`e = 0`). `Ticked.jointDevice` is the transducer as a `JointDevice` whose chip outputs its tags
and `ζ` of what it observed, through `Ticked.prog`, capped and translated as `Ticked.device` is.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Ticked

variable {N b h S n : ℕ}

/-- The rest at a fixed oracle as one side source: at `IChip.addr v a`, its bit at `a` after the chip sent `v`. -/
def jointSide (rest : Rest N b h) (E : Store N b) (a : ℕ) : Bool :=
  match (Encodable.decode a : Option (List Bool × ℕ)) with
  | some p => side rest E (fun _ _ => false) p.1 p.2
  | none => false

/-- **The transducer as a joint device** over `T` ticks, for a decoder `ζ`: its state, HBM as device memory and the
rest at a fixed oracle as side sources, each from the store and `z`; the chip runs `prog` ending in the tags and `ζ` of
what it observed, capped at `c` side bits and `q` queries. -/
def jointDevice (C : Chip N b h S) (βs T : ℕ) (ζ : Bits S → List (Obs h) → Bits n)
    (state : Store N b → Bits n → Bits S) (hbm : Bits n → Hbm N b) (rest : Bits n → Rest N b h) (c q : ℕ) :
    JointDevice N b h S c 0 q n where
  state := state
  mem E z := mem (hbm z) E
  side E z := jointSide (rest z) E
  chip st := ((prog C βs st (fun o => (C.tags st o, ζ st o)) T [] []).cap
    (fun _ _ => false, fun _ => false) c 0 q).toChip []
  bounded st w := by
    rw [IChip.toChip_sideBits, IChip.toChip_regens, IChip.toChip_queries]
    exact IChip.cap_bounded _ (IChip.lift w) _ c 0 q []

theorem jointDevice_world (C : Chip N b h S) (βs T : ℕ) (ζ : Bits S → List (Obs h) → Bits n)
    (state : Store N b → Bits n → Bits S) (hbm : Bits n → Hbm N b) (rest : Bits n → Rest N b h)
    (hrest : ∀ z E O O', rest z E O = rest z E O') (c q : ℕ) (E : Store N b) (z : Bits n) (O : Oracle N b h) :
    IChip.lift ((jointDevice C βs T ζ state hbm rest c q).world E z O) = world (hbm z) (rest z) E O := by
  simp only [IChip.lift, JointDevice.world, jointDevice, world, IWorld.mk.injEq, true_and, and_true]
  funext v a
  simp only [jointSide, IChip.addr, Encodable.encodek, side, hrest z E (fun _ _ => false) O]

theorem jointDevice_run (C : Chip N b h S) (βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (ζ : Bits S → List (Obs h) → Bits n) (state : Store N b → Bits n → Bits S) (hbm : Bits n → Hbm N b)
    (rest : Bits n → Rest N b h) (hrest : ∀ z E O O', rest z E O = rest z E O') (E : Store N b) (z : Bits n)
    (O : Oracle N b h) :
    (jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)).out E z O =
        (C.tags (state E z) (observe C βs (state E z) (hbm z) (rest z) E O T [] [] []),
          ζ (state E z) (observe C βs (state E z) (hbm z) (rest z) E O T [] [] [])) ∧
      (jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)).ingress E z O =
        hbmBits (observe C βs (state E z) (hbm z) (rest z) E O T [] [] []) := by
  set fin : List (Obs h) → (Fin N → Bits h) × Bits n := fun o => (C.tags (state E z) o, ζ (state E z) o)
  have hw := jointDevice_world C βs T ζ state hbm rest hrest (T * βs) (T * qt) E z O
  obtain ⟨c2, c3, c4⟩ := prog_counts C βs qt hq (state E z) fin (world (hbm z) (rest z) E O) T [] [] []
  obtain ⟨hr, hi⟩ := IChip.cap_eq (fun _ _ => false, fun _ => false) (world (hbm z) (rest z) E O)
    (prog C βs (state E z) fin T [] []) (T * βs) 0 (T * qt) [] c2.le c3.le c4
  refine ⟨?_, ?_⟩
  · change (((prog C βs (state E z) fin T [] []).cap _ (T * βs) 0 (T * qt)).toChip []).run
      ((jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)).world E z O) = _
    rw [IChip.toChip_run, hw, hr, prog_run]
  · change (((prog C βs (state E z) fin T [] []).cap _ (T * βs) 0 (T * qt)).toChip []).ingressBits
      ((jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)).world E z O) = _
    rw [IChip.toChip_ingressBits, hw, hi]
    exact (add_zero _).symm.trans (prog_ingress C βs (state E z) fin (hbm z) (rest z) E O T [] [] [])

end Ticked

namespace SecurityProofs.Ticked

open Bandwidth.Ticked

variable {N b h S n : ℕ}

/-- **Mbw for the chip as a tick transducer, in the reads form** (`e = 0`, passive memory): with the rest ignoring the
oracle and its link delivering `βs` bits a tick, the chip's state at the window's start `S` bits fixed by the store and
`z`, HBM hearing only the chip and at most `qt` queries a tick, if the chip's view determines `z` (`hζ`) then the
probability over the store, `z`, the oracle and the sample that the chip's tags pass while
`hbmBits + S + βs T + λ < k b + n` is at most `AuditRO`'s bound. So an accept gives `n ≤ hbmBits − cert_bw`, with
`cert_bw = k b − S − βs T − λ` what `audit_reads_ro` certifies. -/
theorem audit_joint (C : Ticked.Chip N b h S) (βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (ζ : Bits S → List (Obs h) → Bits n) (state : Store N b → Bits n → Bits S) (hbm : Bits n → Hbm N b)
    (rest : Bits n → Rest N b h) (hrest : ∀ z E O O', rest z E O = rest z E O')
    (hζ : ∀ E z O, ζ (state E z) (observe C βs (state E z) (hbm z) (rest z) E O T [] [] []) = z) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1.1 x.1.2)
            (observe C βs (state x.1.1 x.1.2) (hbm x.1.2) (rest x.1.2) x.1.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          hbmBits (observe C βs (state x.1.1 x.1.2) (hbm x.1.2) (rest x.1.2) x.1.1 x.2.1 T [] [] []) + S + βs * T +
            lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  have hrec : (jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)).Recovers := fun E z O => by
    rw [(jointDevice_run C βs qt T hq ζ state hbm rest hrest E z O).1]
    exact hζ E z O
  refine (pr_mono fun x hx => ?_).trans
    (AuditJoint N b h S (T * βs) 0 (T * qt) n s k lam (jointDevice C βs T ζ state hbm rest (T * βs) (T * qt)) hrec)
  obtain ⟨ho, hi⟩ := jointDevice_run C βs qt T hq ζ state hbm rest hrest x.1.1 x.1.2 x.2.1
  refine ⟨fun i => ?_, ?_⟩
  · simp only [ho]
    exact hx.1 i
  · rw [hi]
    have := hx.2
    nlinarith

/-- **Mbw for the chip as a tick transducer, in the capacity form**: without passive memory, every tick's reply has
width at most `βm`, and the probability that the chip's tags pass while `βm T + S + βs T + λ < k b + n` is at most
`AuditRO`'s bound. -/
theorem audit_joint_capacity (C : Ticked.Chip N b h S) (βm βs qt T : ℕ) (hw : ∀ st o, C.width st o ≤ βm)
    (hq : ∀ st o, (C.ask st o).length ≤ qt) (ζ : Bits S → List (Obs h) → Bits n)
    (state : Store N b → Bits n → Bits S) (hbm : Bits n → Hbm N b) (rest : Bits n → Rest N b h)
    (hrest : ∀ z E O O', rest z E O = rest z E O')
    (hζ : ∀ E z O, ζ (state E z) (observe C βs (state E z) (hbm z) (rest z) E O T [] [] []) = z) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1.1 x.1.2)
            (observe C βs (state x.1.1 x.1.2) (hbm x.1.2) (rest x.1.2) x.1.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          βm * T + S + βs * T + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_joint C βs qt T hq ζ state hbm rest hrest hζ s k lam)
  have := hbmBits_le C βs βm hw (state x.1.1 x.1.2) (hbm x.1.2) (rest x.1.2) x.1.1 x.2.1 T [] [] []
  have := hx.2
  simp only [hbmBits, List.map_nil, List.sum_nil] at *
  nlinarith

end SecurityProofs.Ticked

end Bandwidth
