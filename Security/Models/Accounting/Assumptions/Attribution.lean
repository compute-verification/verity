import Definitions.Accounting.Attribution
import Definitions.Accounting.Device
import Definitions.Accounting.Physical
import Definitions.NetworkCertifier.Schedule

/-!
# Compute attribution's physical premises (trusted)

The premises compute attribution adds to the actual-circuit model's (`Definitions.Accounting.Physical`), each a `Prop`
about a strategy `S` on coins split into the window's salt `s : Ωs` and the later draw `d : Ωd`, taken as a hypothesis
by `Accounting.Guarantees.PouwBlockCredit`, never an `axiom`. Each is meant at every draw: the draw reaches the prover
only after the deadline, so at a fixed draw the salt is the window's one coin. The window `w` is PoUW's
(`Verdict.window`, `[ts, td)`).

* `SaltCustody`: the salt reaches the window only through the block's secret inputs (`Secrets` at the window's start,
  at each draw). Deployed, the challenger sends the salt encrypted under a key the auditor's devices share, the block's
  certifier releases it on the block's side at `ts`, and the verifier publishes it only after registration. The premise
  is the information-theoretic form; the keystream's computational gap is outside it.
* `SaltContainment`: what the block sends, up to the window's end, to readers that aren't certifiers takes at most
  `2^kc` values across salts. This is the premise about the prover's own host. The hardware `H` is to declare every physical
  channel out of the block as an uncertified link with its capacity: power draw (a shared PDU, host sensors), EM,
  thermal and acoustic emission, shared DRAM and PCIe peers on the block's host, the BMC and DPU. `DeclaredLinks` is
  then a premise about the auditor's model of the hardware, and `SaltContainment` says those channels carry fewer than
  `kc` bits a window about the block's coins. The prover owns the block's host and would like to move the salt out of
  it, so this premise is about hardware the prover controls, which the trust model otherwise doesn't rest on; it is
  named so that it can be decided (`note:20261018T0850Z-draft-rt-attrib` §2). A helper in another facility reaches the
  salt only over the certified network; a co-located one is what the premise excludes. With every link out of the
  block certified (`Hardware.SpatialUnit`) and only the declared links (`DeclaredLinks`), it holds at `kc = 0`, by a
  lemma the proof owes. Deployed, `kc` plus the boundary's bits (`BoundaryObs`) must stay below about 384, so that
  `q · 2^(k − 512)` stays below `2^−64` at `q ≤ 2^64` oracle calls.
* `SaltFreeJitter`: `TrustedUnits`' jitter clause (W2 and J) with each certifier's jitter word a function of the draw
  alone, never of the salt. A certifier is the auditor's device (`trusted-nebius-host`): its jitter is its own clock's,
  not a function of the prover's secret. At a fixed draw a certifier's sends are then a function of its records, so a
  certifier off `X`'s boundary adds nothing the boundary's records didn't (a lemma the proof owes, by induction on
  ticks).
* `BoundaryObs`: at each draw, `∂X`'s certifiers' records placed during the window are a function of the boundary's
  observations (`NetTiming.Obs`: each egress link's row, one jitter class a step for the whole of `∂X`, and each
  boundary link's status) over `m` windows of `T` buckets, an observation sequence the constant-rate Program accepts
  with jitter below `D` (`NetTiming.AccSeqs`). These number at most `#Ac · (#Σ_sync · D^T)^m`
  (`NetTiming.Guarantees.EncardAccSeqsLeConstantRatePerWindow`, proved): the content advice once, then
  `log₂ #Σ_sync + T · log₂ D` bits a window, once a boundary step and not once a port. Under `SaltFreeJitter` the
  jitter is fixed at each draw, so `D = 1` and that is 7 bits a window at 16 × 8 sync values. It is N3 (egress records
  bound to committed crossings), NET-5 (a declared scheduler) and the coin-cone rule (nothing in the salt's cone
  crosses `∂X` before registration) together, as the network certifier's model states them.
* `OutsideQueries`: the outside's evaluations of the noise hash during the window are at most `q` at every coin.
  Deployed, it is the outside's compute over the window at the hash's price (`Placed.ComputeRate`).
-/

namespace Accounting.Assumptions

/-- **Salt custody** (at each draw `d`): the block `X` is apart from the certifiers `cert`, and what the certifiers and
the outside `(X ∪ cert)ᶜ` can tell at the window's start is the same at every salt (`Secrets`): the salt is an input
of the block's alone. -/
def SaltCustody {Ωs Ωd L : Type*} [LinearOrder L] [Fintype L] (S : Ωs × Ωd → Actual.Run L) (cert X : Finset L)
    (w : Window) : Prop :=
  ∀ d, Secrets (fun s => S (s, d)) cert X w.start

/-- **Salt containment** (at each draw `d`): what the block `X` sends, from the gates it places before the window `w`
ends, to readers that aren't certifiers (`Actual.Run.sent`, its wires to leaves outside `X ∪ cert`) takes at most
`2^kc` values across salts. A wire is keyed by its source's tick, so a value placed before `ts` and read in the window
is counted too. The hardware declares every physical channel out of `X` (power, EM, thermal, shared DRAM, PCIe peers,
the BMC) as a link, so this is the block's leak about its coins through all of them: fewer than `kc` bits a window. -/
def SaltContainment {Ωs Ωd L : Type*} (S : Ωs × Ωd → Actual.Run L) (cert X : Finset L) (w : Window) (kc : ℕ) :
    Prop :=
  ∀ d, (Set.range fun s => (S (s, d)).sent (X : Set L) (Set.Iio (w.start + w.len)) ∩ {p | p.2.1 ∉ cert}).encard ≤
    ((2 ^ kc : ℕ) : ℕ∞)

/-- **Salt-free jitter**: `TrustedUnits`' W2 and J with each certifier's jitter word `J d y` a function of the draw `d`
alone. For some jitter words, one per certifier and draw, each one of `2^jitter` classes a bucket of ticks (`bucket`),
what each certifier `y` sends outside itself from the gates it places during `[a, b)` is a function of its own record
placed before `b` (`Actual.Run.record` in its record cells `record y`, as seen) and of its own jitter word on the
buckets of `[a, b)`, at the run's draw. -/
def SaltFreeJitter {Ωs Ωd L : Type*} (S : Ωs × Ωd → Actual.Run L) (cert : Finset L) (record : L → Set ℕ)
    (bucket : ℕ → ℕ) (jitter : ℕ) : Prop :=
  ∃ J : Ωd → L → ℕ → Fin (2 ^ jitter), ∀ y ∈ cert, ∀ a b r r',
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => (S r).A.t g < b) =
      (S r').seen '' ↑(((S r').record {y} record).filter fun g => (S r').A.t g < b) →
    (∀ t ∈ Set.Ico a b, J r.2 y (bucket t) = J r'.2 y (bucket t)) →
    (S r).sent {y} (Set.Ico a b) = (S r').sent {y} (Set.Ico a b)

/-- **The boundary's observations** (at each draw `d`): for some operator `op` and some observation sequence `obs s` at
each salt, of `m · T` steps (`m` windows of `T` buckets) that the constant-rate Program (`NetTiming.constantRate`, `Wn`
windows with advice, the sync set `ss`, the frames `frames`) accepts with jitter classes below `D`
(`NetTiming.AccSeqs`), the records the certifiers `bcert` place during the window `w` (`Attribution.records`) are the
same at two salts whose observations are. Meant at `bcert = Attribution.boundaryCert H X`, `∂X`'s certifiers.

The steps are counted from the start of the boundary audit's epoch, since the proved count is cumulative from its first
window: an epoch that starts at `ts` pays only for the PoUW window's own windows, an earlier start pays 7 bits for each
earlier one. It is meant at strategies the boundary audit passes at every salt: a window whose boundary audit fails
earns no attribution (its coverage isn't established), and a strategy that fails it at some salts leaks which ones,
which the joint event (PoUW's and the boundary audit's acceptance) is to charge when it is proved. -/
def BoundaryObs {Ωs Ωd L : Type*} {Lk : ℕ} {In Ac : Type} (K Sz : ℕ) (S : Ωs × Ωd → Actual.Run L)
    (bcert : Finset L) (cells : L → Set ℕ) (w : Window) (T Wn : ℕ) (ss : NetTiming.SyncSet)
    (frames : List In → Ac → Fin Lk → List NetTiming.LogicalFrame) (m D : ℕ) : Prop :=
  ∀ d, ∃ (op : ℕ → List (NetTiming.Obs Lk K Sz) → In) (obs : Ωs → List (NetTiming.Obs Lk K Sz)),
    (∀ s, obs s ∈ NetTiming.AccSeqs (NetTiming.constantRate (S := Sz) T Wn ss frames) op (m * T) D) ∧
    ∀ s s', obs s = obs s' →
      Attribution.records (S (s, d)) bcert cells w = Attribution.records (S (s', d)) bcert cells w

open Classical in
/-- **The outside's oracle calls**: at every coin, at most `q` of the gates `hash r` (the run's evaluations of the noise
hash, each marked at its output gate) are placed outside the block `X` and the certifiers `cert` during the window
`w`. -/
def OutsideQueries {Ω L : Type*} (S : Ω → Actual.Run L) (hash : ∀ r, Finset (Fin (S r).A.N)) (cert X : Finset L)
    (w : Window) (q : ℕ) : Prop :=
  ∀ r, ((hash r).filter fun g => (S r).A.x g ∉ X ∪ cert ∧ (S r).A.t g ∈ w.ticks).card ≤ q

end Accounting.Assumptions
