module

public import Definitions.Accounting.Remote.Held
public import Definitions.Accounting.Remote.Draw

/-!
# PoUB's sampled draw: the instances its witnesses are stated at

The witnesses of the five sampled guarantees (`Accounting.SecurityProofs.Remote.PooledSample`, `LiveCoinsSampled`,
`HeldSampled`, `HeldOverTimeSampled`, `HeldOverTimeLinksSampled`; `Proofs/Accounting/RemoteHeld/DrawMeets.lean`) are
stated at these instances.

* `burst`: two passes of `D` ticks, every receipt correct, on a node whose first `Br + Bd + Bl` registered blocks are
  absent until tick `t₁` and all come back at `t₁`, by rebuilds, disk reads and links. `burstPace` gives its groups.
* `honest`: a draw whose window `t` registers `U t` receipts, every one correct, whatever the coins.
* W2's session, `w2Session` over `w2Draw`: one window of `130,023,424` receipts, sampled `33,273` times on 256-bit keys,
  its node at W2's numbers. Its coin stream (`zeroStream`) and receipts (`w2Receipts`) are fixed arbitrarily: the honest
  draw meets its premises at every stream and every receipts structure.
* The toy, `toy`: one window of one receipt, none correct, sampled once. It is checked by an audit that accepts every
  outcome (`acceptAll`), with receipts no opening verifies (`unsalted`), over the span `[0, 1)` (`toyPace`).
-/

@[expose] public section

namespace Accounting.Remote.Instances

open Finset Accounting.Remote.Held Accounting.Remote.Draw

/-! ## A node that uses every route -/

/-- `B` blocks at tick `t₁` and none at any other. -/
def pulse {Ω : Type} (t₁ B : ℕ) : Ω → ℕ → ℕ := fun _ t => if t = t₁ then B else 0

/-- The registered blocks in node memory at tick `t`: all of them after `t₁`, and until then all but the first `k`. -/
def burstMem (N t₁ k t : ℕ) : Finset (Fin N) :=
  if t ≤ t₁ then (univ.filter fun j : Fin N => k ≤ (j : ℕ)) else univ

/-- Two passes of `D` ticks, each reading every registered block correctly, on a node whose first `Br + Bd + Bl`
registered blocks are absent until tick `t₁` and all come back at `t₁`: `Br` rebuilt, `Bd` read from its disks and `Bl`
carried over its links. -/
def burst {Ω : Type} (N D t₁ Br Bd Bl : ℕ) : Trace Ω where
  blocks := N
  passes := 2
  accepts := fun _ => True
  tick := fun _ p => p * D
  plan := fun _ _ => univ
  correct := fun _ _ => univ
  mem := fun _ t => burstMem N t₁ (Br + Bd + Bl) t
  chip := fun _ _ => ∅
  rebuilt := pulse t₁ Br
  disk := pulse t₁ Bd
  link := pulse t₁ Bl

/-- Its groups: each pass its own, the first credited and the second closing, each coin released as the group before
ends. -/
def burstPace {Ω : Type} (D : ℕ) : Pace Ω where
  credited := 1
  first := fun _ g => g
  released := fun _ g => g * D
  ended := fun _ g => (g + 1) * D

/-! ## The honest draw and W2's session -/

/-- **The honest draw**: window `t` registers `U t` receipts, every one correct, whatever the coins. -/
def honest {n m : ℕ} (k : Fin n → ℕ) (κ : ℕ) (X : Type) (U : Fin n → Fin (m + 1)) : Run n m k κ X where
  size t _ := U t
  correct t _ := range (U t)

/-- W2's window: `130,023,424` receipts, under the cap `2^64 − 1`. -/
def w2Window : Fin 1 → Fin (18446744073709551615 + 1) := fun _ => ⟨130023424, by decide⟩

/-- W2's session: its one window, every receipt correct, sampled `33,273` times, on 256-bit keys. -/
abbrev w2Session : Run 1 18446744073709551615 (fun _ => 33273) 256 Unit :=
  honest (fun _ => 33273) 256 Unit w2Window

/-- The all-zero coin stream. -/
abbrev zeroStream : Fin 1 → Pous.Bits 256 → Blocks 18446744073709551615 := fun _ _ _ _ => false

/-- W2's receipts, fixed arbitrarily. -/
def w2Receipts : Receipts 1 256 Unit where
  hash _ _ := 0
  verifier _ _ _ := ⟨⟨fun _ => 0, 0, fun _ => 0⟩, [], .v2⟩
  root _ _ _ _ := 0
  rank _ _ _ := 0
  opened _ _ _ := ⟨[], [], []⟩

/-- W2's node on the session's coins: `burst` at W2's numbers, with `Bl` link blocks. -/
abbrev w2Draw (Bl : ℕ) : Trace (DrawCoins 1 256 Unit) := burst 130023424 85000000 0 743590 1054688 Bl

/-! ## The toy that an audit accepting everything gets wrong -/

/-- One window of one receipt, none correct, sampled once. -/
def toy : Run 1 1 (fun _ => 1) 1 Unit where
  size _ _ := ⟨1, by decide⟩
  correct _ _ := ∅

/-- An audit that accepts every outcome, its one pass's one block wrong and its node holding nothing. -/
def acceptAll : Trace (DrawCoins 1 1 Unit) where
  blocks := 1
  passes := 1
  accepts _ := True
  tick _ _ := 0
  plan _ _ := univ
  correct _ _ := ∅
  mem _ _ := ∅
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- Receipts no opening verifies: every salt is empty. -/
def unsalted : Receipts 1 1 Unit where
  hash _ _ := 0
  verifier _ _ _ := ⟨⟨fun _ => 0, 0, fun _ => 0⟩, [], .v2⟩
  root _ _ _ _ := 0
  rank _ _ _ := 0
  opened _ _ _ := ⟨[], [], []⟩

/-- Its credited span is `[0, 1)`, with no credited group. -/
def toyPace : Pace (DrawCoins 1 1 Unit) where
  credited := 0
  first _ _ := 0
  released _ _ := 1
  ended _ _ := 0

/-- The toy's coin stream, all zero. -/
abbrev toyStream : Fin 1 → Pous.Bits 1 → Blocks 1 := fun _ _ _ _ => false

end Accounting.Remote.Instances
