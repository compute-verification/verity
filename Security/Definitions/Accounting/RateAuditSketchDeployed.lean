import Definitions.Accounting.RateAuditSketch

/-!
# A deployed session of verbatim sketch receipts (trusted)

This is the session form of `RateSketch.Device`, the verbatim-sector model. A window draws one tensor-sketch signs
matrix, then runs `P` causal passes. Each pass may grind among `G` independent oracle candidates; its device may depend
on the earlier passes' coins and on the candidate, while the pick may also depend on the store, the signs and the whole
candidate family. After all passes the window draws `d` receipts from its `P·N` receipts. A session has `n` causal
windows on one store.

The model keeps the restriction proved by `RateAuditTensorSketchRank`: device memory, on-chip state, the side link and
rebuilding deliver sectors whole and verbatim. A precomputed sketch table is not a device of this type.
-/

namespace Accounting.RateSketch

open Pous (Bits)

/-- The store shared by all passes and windows: `N` tiles of `m` 32-byte sectors. -/
abbrev SketchStore (N m : ℕ) : Type := Fin N × Fin m → Sector

/-- One pass's certified stream. -/
abbrev SketchStream (ρ : ℕ) : Type := Fin ρ → Sector

/-- Assemble the shared store and one pass's stream as the data a verbatim device reads. -/
def Data.ofParts {N m ρ : ℕ} (E : SketchStore N m) (R : SketchStream ρ) : Data N m ρ Sector
  | .inl p => E p
  | .inr i => R i

/-- One pass's fresh coins: its certified stream and `G` candidate random oracles. The window's signs and pooled draw
are drawn once outside the passes. -/
abbrev SketchPassCoins (N ρ h ks G : ℕ) : Type :=
  SketchStream ρ × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)))

instance instFintypeSketchPassCoins (N ρ h ks G : ℕ) : Fintype (SketchPassCoins N ρ h ks G) := by
  letI : Fintype (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  exact inferInstance

/-- One window's coins: one signs matrix, `P` passes, then `d` pooled openings among the `P·N` receipts. -/
abbrev SketchWindowCoins (P N m ρ h ks G d : ℕ) : Type :=
  SketchSigns ks m × (Fin P → SketchPassCoins N ρ h ks G) × (Fin d → Fin P × Fin N)

instance instFintypeSketchWindowCoins (P N m ρ h ks G d : ℕ) :
    Fintype (SketchWindowCoins P N m ρ h ks G d) := by
  letI : Fintype (SketchPassCoins N ρ h ks G) := instFintypeSketchPassCoins N ρ h ks G
  exact inferInstance

/-- A window of `P` causal verbatim-sector passes with `G` oracle candidates per pass. -/
structure SketchDeployedWindow (P N m ρ h S L q ks G : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → SketchPassCoins N ρ h ks G) → Fin G →
    Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))
  pick : (j : Fin P) → ({i : Fin P // i < j} → SketchPassCoins N ρ h ks G) → SketchStore N m →
    SketchSigns ks m → (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) → Fin G

namespace SketchDeployedWindow

variable {P N m ρ h S L q ks G : ℕ}

/-- The candidate pass `j` keeps. -/
def kept (A : SketchDeployedWindow P N m ρ h S L q ks G) (j : Fin P) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) : Fin G :=
  A.pick j (fun i => ω i.1) E σ (ω j).2

/-- Pass `j`'s device at the candidate it kept. -/
def dev (A : SketchDeployedWindow P N m ρ h S L q ks G) (j : Fin P) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) :
    Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) :=
  A.device j (fun i => ω i.1) (A.kept j E σ ω)

/-- Pass `j`'s random oracle at the candidate it kept. -/
def oracle (A : SketchDeployedWindow P N m ρ h S L q ks G) (j : Fin P) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) :
    Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) :=
  (ω j).2 (A.kept j E σ ω)

/-- The pooled receipts whose sketch tags are correct. -/
def correctAll (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) : Finset (Fin P × Fin N) :=
  Finset.univ.filter fun p =>
    let D := Data.ofParts E (ω p.1).1
    let B := A.dev p.1 E σ ω
    let O := A.oracle p.1 E σ ω
    p.2 ∈ B.correct sketch σ O D

/-- Device-memory sectors read in the window, in bits, summed over its passes. -/
def reads (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) : ℕ :=
  ∑ j,
    let D := Data.ofParts E (ω j).1
    (A.dev j E σ ω).footprint σ (A.oracle j E σ ω) D

end SketchDeployedWindow

/-- A causal session of `n` deployed sketch windows on one shared store. -/
structure SketchDeployedSession (n P N m ρ h S L q ks G d : ℕ) where
  window : (w : Fin n) → ({v : Fin n // v < w} → SketchWindowCoins P N m ρ h ks G d) →
    SketchDeployedWindow P N m ρ h S L q ks G

namespace SketchDeployedSession

variable {n P N m ρ h S L q ks G d : ℕ}

/-- Window `w` at the session's coins. -/
def win (A : SketchDeployedSession n P N m ρ h S L q ks G d) (w : Fin n)
    (ω : Fin n → SketchWindowCoins P N m ρ h ks G d) : SketchDeployedWindow P N m ρ h S L q ks G :=
  A.window w (fun v => ω v.1)

end SketchDeployedSession

/-- The non-sampling shortfall of one pass, as a share of its `N·m·256` stored bits. -/
noncomputable def gammaSketch (N m ρ S L X : ℕ) : ℝ :=
  ((S + (L - ρ * 256) + X : ℕ) : ℝ) / (N * m * 256)

end Accounting.RateSketch
