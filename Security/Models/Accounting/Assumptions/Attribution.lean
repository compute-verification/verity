import Definitions.Accounting.Device
import Definitions.Accounting.Physical

/-!
# Compute attribution's physical premises (trusted)

The two premises compute attribution adds to the actual-circuit model's (`Definitions.Accounting.Physical`), each a
`Prop` about a strategy `S` on coins split into the window's salt `s : Ωs` and the later draw `d : Ωd`, taken as a
hypothesis by `Accounting.Guarantees.PouwBlockCredit`, never an `axiom`. Each is meant at every draw: the draw reaches
the prover only after the deadline, so at a fixed draw the salt is the window's one coin. The window `w` is PoUW's
(`Verdict.window`, `[ts, td)`).

* `SaltCustody`: the salt reaches the window only through the block's secret inputs (`Secrets` at the window's start,
  at each draw). Deployed, the challenger sends the salt encrypted under a key the auditor's devices share, the block's
  certifier releases it on the block's side at `ts`, and the verifier publishes it only after registration. The premise
  is the information-theoretic form; the keystream's computational gap is outside it.
* `EgressAdvice`: at each draw, the certifiers' records placed during the window take at most `2^k` values across
  salts. Deployed, it is what egress records bound to committed crossings (N3), a declared scheduler (NET-5) and the
  coin-cone rule give together: nothing in the salt's cone crosses a block's boundary before registration, so each
  record's content is the same at every salt, and only its timing can vary: the scheduler's choice once and
  `log₂ #Σ_sync` bits a link-window at jitter below 1 (`NetworkCertifier.Guarantees.EncardAccSeqsLeConstantRatePerWindow`,
  7 bits at 16 × 8 sync values). It counts every certifier's records, not only the block's: at a fixed draw the rest of
  the datacenter's records are a function of the block's, so the block's link bits are meant to suffice, by a lemma
  the proof owes.
-/

namespace Accounting.Assumptions

/-- **Salt custody** (at each draw `d`): the block `X` is apart from the certifiers `cert`, and what the certifiers and
the outside `(X ∪ cert)ᶜ` can tell at the window's start is the same at every salt (`Secrets`): the salt is an input
of the block's alone. -/
def SaltCustody {Ωs Ωd L : Type*} [LinearOrder L] [Fintype L] (S : Ωs × Ωd → Actual.Run L) (cert X : Finset L)
    (w : Window) : Prop :=
  ∀ d, Secrets (fun s => S (s, d)) cert X w.start

/-- **Egress advice** (at each draw `d`): the certifiers' records placed during the window `w` (`Actual.Run.record` in
the record cells `cells`, as seen) take at most `2^k` values across salts. -/
def EgressAdvice {Ωs Ωd L : Type*} (S : Ωs × Ωd → Actual.Run L) (cert : Finset L) (cells : L → Set ℕ) (w : Window)
    (k : ℕ) : Prop :=
  ∀ d, (Set.range fun s => (S (s, d)).seen ''
    ↑(((S (s, d)).record cert cells).filter fun g => (S (s, d)).A.t g ∈ w.ticks)).encard ≤ ((2 ^ k : ℕ) : ℕ∞)

end Accounting.Assumptions
