module

public import Definitions.Accounting.Guarantees.LinkEnds

/-!
# The links premise at the places a link may end at, at `f = id`

* `Actual.Run.signalVia_id`, `alphabetVia_id`, `declaredLinksVia_id`: with each leaf its own link end, the signal at a
  link end is the signal with the reader's leaf beside each source, which is the link end itself, so the alphabet and
  the links premise are `Alphabet` and `DeclaredLinks`.
* `SecurityProofs.B4UnderId`: `B4Under` at `f = id` over every table is `B4`.
* `SecurityProofs.b4Via_of_ownLinkEnds`: on a Site whose every place is its own link end, `B4` gives `B4Via` there.
-/

@[expose] public section

namespace Accounting

variable {Ω L E : Type*}

theorem Actual.Run.signalVia_id (R : Actual.Run L) (link : Fin R.A.N → Fin R.A.N → E) (e : E) (y : L) (u : ℕ) :
    R.signalVia id link e y u = Prod.mk y '' R.signal link e y u := by
  ext p
  constructor
  · rintro ⟨g, a, ha, hne, hl, hy, hu, rfl⟩
    exact ⟨R.seen a, ⟨g, a, ha, hne, hl, hy, hu, rfl⟩, Prod.ext hy.symm rfl⟩
  · rintro ⟨q, ⟨g, a, ha, hne, hl, hy, hu, rfl⟩, rfl⟩
    exact ⟨g, a, ha, hne, hl, hy, hu, Prod.ext hy.symm rfl⟩

theorem alphabetVia_id (S : Ω → Actual.Run L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) :
    AlphabetVia S id β link ↔ Alphabet S β link := by
  refine forall_congr' fun e => forall_congr' fun y => forall_congr' fun u => ?_
  have h : (Set.range fun r => (S r).signalVia id (link r) e y u) =
      Set.image (Prod.mk y) '' Set.range fun r => (S r).signal (link r) e y u := by
    rw [← Set.range_comp]
    exact congrArg Set.range (funext fun r => (S r).signalVia_id (link r) e y u)
  rw [h, (Set.image_injective.2 (Prod.mk_right_injective y)).encard_image]

theorem declaredLinksVia_id (S : Ω → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) :
    DeclaredLinksVia S id ends β link ↔ DeclaredLinks S ends β link :=
  and_congr (alphabetVia_id S β link) Iff.rfl

end Accounting

namespace Accounting.SecurityProofs

universe u v w

theorem B4UnderId : Guarantees.B4UnderId.{u, v, w} := by
  constructor
  · intro h L E _ _ _ ends β Ω S ℓ link 𝒱 cert X spec record bucket jitter a b _ _ hc hd ht hs
    exact h S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hc ((declaredLinksVia_id S ends β link).1 hd) ht hs
  · intro h L E Ω _ _ _ S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hc hd ht hs
    exact h ends β S ℓ link 𝒱 cert X spec record bucket jitter a b (fun _ => Iff.rfl) (fun _ => Iff.rfl) hc
      ((declaredLinksVia_id S ends β link).2 hd) ht hs

/-- On a Site whose every place is its own link end (no place's `host?` names another), `B4` gives `B4Via` there, at
the registered rates. -/
theorem b4Via_of_ownLinkEnds (h : Guarantees.B4.{0, 0, w}) (T : Verity.Site) (hT : ∀ p, T.linkPlace p = p)
    (β : Fin T.links.length → ℕ) (_ : T.RatedOn β Set.univ) :
    Guarantees.B4Under.{0, 0, w} T.linkPlace T.placeEnds β := by
  rw [show T.linkPlace = id from funext hT]
  exact (B4UnderId.{0, 0, w}.1 h) T.placeEnds β

end Accounting.SecurityProofs
