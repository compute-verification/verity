import Pous.Model.Audit

/-!
# P2's bridge assumptions (trusted)

**The game has no cost model.** In the PoUS game local computation is free, there is no clock, and `W` is an input to
every round. P2's security rests on time: its block maps are permutations of `[0, p)` built from squarings in `Z_p` and
keyed from SHAKE256, the forward direction is cheap, and the inverse needs a root, `w − 2` sequential squarings, which a
prover answering within `Δ + RTT` cannot afford. In the game a responder that holds `W` and the keys computes any root
for free. So "the deployed P2 is `AuditSecure`" would be false there, and a theorem resting on it would be vacuous.

**So these are bridges, not statements about time.** Each is a property of a `RealAudit`, the deployed audit taken as
a parameter: its strategies, which of them stay within the deployment's bounds, and how often each passes, with the
keys as deployed and with uniform keys. Together they connect that audit to the ideal model `m1p`
(`Pous/Model/M1p.lean`) that `Pous.Pinned.P2MeetsM1pB19Uncond` and the other P2 pins are about. Lean checks how they
compose (`PousProofs/P2/Security.lean`) and that each can hold (`PousProofs/P2/Sanity.lean`). What they claim about
SHAKE256, about squaring in `Z_p` and about wall-clock time is in the docstrings and in the P2 specification v1, and Lean
checks none of it. A faithful statement needs a cost model in which roots cost sequential squarings; that is a separate
piece of work.

Each is a named `Prop`, taken as a hypothesis by the theorems that use it, never an `axiom`. The names are the P2
specification v1's: P2-EXP-IO (`ExpIO`), SeqRoot (`SeqRoot`) and P2–M1-SGI (`SGI`). The specification does not accept
P2–M1-SGI, so P2 stays experimental.
-/

namespace Pous.P2

open scoped ENNReal

variable {M : IdealModel}

/-- **The deployed P2 audit, as a parameter**, beside the ideal-model scheme `sch` whose payloads it stores (in use,
`m1p` at P2's parameters). Lean gives the fields no meaning; the docstrings say what each stands for. -/
structure RealAudit (sch : Scheme M) where
  /-- What a prover deploys: its computation until setup ends, what it then stores, and how it answers challenges. -/
  Strategy : Type
  /-- `Bounded S s`: `s` stores at most `S` bits once setup ends, and answers each challenge within the deadline
  `Δ + RTT`. -/
  Bounded : ℕ → Strategy → Prop
  /-- `Forward Q s`: each of `s`'s answers evaluates the block maps forward at most `Q` times and computes no root,
  that is, inverts no block map on a value it has not stored. -/
  Forward : ℕ → Strategy → Prop
  /-- `accept k W s`: the probability that `s` passes `k` sequential challenges on the payload `W`, with the keys
  derived as deployed (SHAKE256 with the specification's framing). -/
  accept : ℕ → Bits sch.n → Strategy → ℝ≥0∞
  /-- `acceptIdeal k W s`: the same, with the keys replaced by uniform, independent ones. -/
  acceptIdeal : ℕ → Bits sch.n → Strategy → ℝ≥0∞

variable {sch : Scheme M}

/-- **P2-EXP-IO (`random-oracle`: SHAKE256 as the key expander)** (specification v1, F4: SHAKE256 with its framing;
ChaCha8 does not conform). For a strategy that stores at most `S` bits, deriving the keys as deployed instead of uniformly
raises its acceptance by at most `ε`: the key expander's output is indistinguishable from uniform, independent keys per
segment, block and purpose, even given `W` and everything the prover saw before setup. -/
def ExpIO (A : RealAudit sch) (S : ℕ) (ε : ℝ≥0∞) : Prop :=
  ∀ k W s, A.Bounded S s → A.accept k W s ≤ A.acceptIdeal k W s + ε

/-- **SeqRoot (`seq-root`): within the deadline, no root and at most `Q` forward evaluations.** A strategy within the
deployment's bounds (`S` stored bits, each answer within `Δ + RTT`) computes no root and makes at most `Q` forward
evaluations per answer. This is the timing assumption (one root at width `w` takes longer than twice the deadline on the
fastest core), together with the calibration of `Δ + RTT` against the time a forward evaluation takes there. -/
def SeqRoot (A : RealAudit sch) (S Q : ℕ) : Prop :=
  ∀ s, A.Bounded S s → A.Forward Q s

/-- **P2–M1-SGI (`sms-m1-instantiation`): square–mask–square with fresh per-block keys instantiates `m1p`** (not
accepted by the specification). With uniform keys, a strategy that stores at most `S` bits and
answers forward-only with at most `Q` evaluations per answer is matched by an adversary of the ideal model: an `S'`-bit
preprocessing and a sequential responder with `Q` forward queries per answer, at some depth, accepted on the same `W`
at least as often, up to `ε`. `S' − S` is the state the instantiation may leak beyond the ideal model, the slack `λ·B`
of the `P2Slack*` pins. -/
def SGI (A : RealAudit sch) (S S' Q : ℕ) (ε : ℝ≥0∞) : Prop :=
  ∀ k W s, A.Bounded S s → A.Forward Q s →
    ∃ (D : ℕ) (A₁ : Preprocessor sch S') (A₂ : SeqResponder sch S' D Q),
      A.acceptIdeal k W s ≤ pr (seqAccepts sch k W A₁ A₂) + ε

end Pous.P2
