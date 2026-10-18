import Definitions.Accounting.Device
import Definitions.Accounting.Physical
import Definitions.Accounting.PouwWork

/-!
# Compute attribution's vocabulary: the outside's advice about a block's coin (trusted)

The physical layer that takes PoUW's logical certificate (decisions 111 and 113: the certificate says the work was done,
the layer says which spatial unit did it) reads one quantity besides the actual circuit's placement: how much the rest
of the datacenter can learn, during the window, about the coin that only the block holds. PoUW's floor counts the work
that depends on the window's salt; a helper outside the block that holds `A` candidate values of what it could learn
about the salt, and evaluates the noise hash `q` times, finds the salt with probability at most `q · A / #salts`, and
without it its help is the same at every salt.

* `Verdict.window P`: PoUW's window `[ts, td)` as a `Window`, `td - ts` ticks from `ts`.
* `boundaryCert H X`: the certifiers on `X`'s boundary links, `∂X`'s.
* `records R cert cells w`: the certifiers' records placed during the window `w`, as seen.
* `leak R cert X w`: what `X` sends, up to the window's end, to readers that aren't certifiers.
* `OutsideAdvice S B cert X w n`: across the coins of the event `B`, what the outside `(X ∪ cert)ᶜ` does during the
  window `w` takes at most `n` values (`Actual.Run.acts`). Under salt custody (the coin reaches the window only through
  `X`'s secret inputs, `Secrets`) this is `log₂ n` bits of advice about the coin on `B`, whatever the outside computes
  from them.

Crossing bytes attribute nothing: a helper holding the weights needs only the salt in and a root out, so ingress
volume bounds no outsourced work. What bounds it is the outside's advice about the salt, which `∂X`'s records and what
crosses `X`'s uncertified links carry and nothing else does.
-/

namespace Accounting.PouwWork

/-- **PoUW's window as a `Window`**: `td - ts` ticks from `ts`, so its ticks are `[ts, td)`. -/
def Verdict.window {Ω : Type} {L : Type*} (P : Verdict Ω L) : Window := ⟨P.ts, P.td - P.ts⟩

end Accounting.PouwWork

namespace Accounting.Attribution

open Classical in
/-- **`∂X`'s certifiers**: the certifiers of `H` at an end of a link across `X`'s boundary (`boundary`). -/
noncomputable def boundaryCert {L E : Type*} [Fintype E] (H : Hardware L E) (X : Finset L) : Finset L :=
  H.cert.filter fun y => ∃ e ∈ boundary H.ends X, (H.ends e).1 = y ∨ (H.ends e).2 = y

/-- **The certifiers' records placed during the window `w`**: the gates at `cert` in their record cells
(`Actual.Run.record`) placed at a tick of `w`, as seen (`Actual.Run.seen`: tick, leaf, cell, width and value). -/
noncomputable def records {L : Type*} (R : Actual.Run L) (cert : Finset L) (cells : L → Set ℕ) (w : Window) :
    Set (ℕ × L × ℕ × ℕ × ℕ) :=
  R.seen '' ↑((R.record cert cells).filter fun g => R.A.t g ∈ w.ticks)

/-- **What the block leaks** in the run `R`: what `X` sends, from the gates it places before the window `w` ends, to
readers that aren't certifiers (`Actual.Run.sent`, its wires to leaves outside `X ∪ cert`): what crosses its links with
no certifier on them, each physical channel among them. A wire is keyed by its source's tick, so a value placed before
`ts` and read in the window is in it. -/
def leak {L : Type*} (R : Actual.Run L) (cert X : Finset L) (w : Window) : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :=
  R.sent (X : Set L) (Set.Iio (w.start + w.len)) ∩ {p | p.2.1 ∉ cert}

/-- **The outside's advice** (across the coins of the event `B`): what the outside `(X ∪ cert)ᶜ` of the location `X` and
the certifiers `cert` does during the window `w`, its gates, what it sends and its state (`Actual.Run.acts`), takes at
most `n` values at the coins where `B` holds. When the coins reach the window only through `X`'s secret inputs
(`Secrets`), everything the outside does with the coins in the window, on `B`, is a function of one of `n` values about
them. -/
def OutsideAdvice {Ω L : Type*} [LinearOrder L] [Fintype L] (S : Ω → Actual.Run L) (B : Ω → Prop) (cert X : Finset L)
    (w : Window) (n : ℕ) : Prop :=
  ((fun r => (S r).acts (X ∪ cert)ᶜ (w.ticks : Set ℕ)) '' {r | B r}).encard ≤ (n : ℕ∞)

end Accounting.Attribution
