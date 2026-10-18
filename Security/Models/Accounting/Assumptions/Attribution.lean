import Definitions.Accounting.Attribution
import Definitions.Accounting.Device
import Definitions.Accounting.Physical
import Definitions.NetworkCertifier.Schedule

/-!
# Compute attribution's physical premises (trusted)

The premises compute attribution adds to the actual-circuit model's (`Definitions.Accounting.Physical`), each a `Prop`
about a strategy `S` on coins split into the window's coin `ω : Ωs` and the later draw `d : Ωd`, taken as a hypothesis
by `Accounting.Guarantees.PouwBlockCredit`, never an `axiom`. The window's coin is the salt and the hiding randomness of
any public commitment to it, `com ω`. Each premise is meant at every draw: the draw reaches the prover only after the
deadline, so at a fixed draw the window's coin is the one coin. The window `w` is PoUW's (`Verdict.window`, `[ts, td)`),
and `bacc` is the boundary audit's verdict at each coin, the one the account conditions the block's credit on:
`SaltContainment` and `BoundaryObs` need to hold only where it accepts.

Custody is over the outside only (`note:compute-accounting/20261018T1948Z-report-redteam-salt-custody`, W5): the
certifiers are the auditor's devices and may depend on the salt. The block's ingress certifier draws the window's
salt, writes it into the block and holds it in a sealed slot; what matters is that nothing the certifiers send outside
the block carries it. NET-18's certifier (`verity/core/protocols/certifier/active.py`, `cursor/net18-salt-061a` at
afe591362) does that once its code does two things, which these premises' docstrings name: it keeps the salt frame's
commitment out of every record published before `td` (sealed in its slot, or moved into a record published at `td`),
and `reveal(w)` takes the window's registration receipt and fails closed without it.

* `SaltCustody`: at the window's start, what the outside and the certifiers off the block's boundary can tell
  (`history` of `(X ∪ ∂X)ᶜ`) depends on the window's coin only through its public commitment. `∂X`'s certifiers, which
  hold the salt, are dropped from the salt-free side. Deployed, `∂X`'s ingress certifier draws the salt from its own
  randomness at `ts` (NET-18's `fresh_salt`, 512 bits), writes it into the block on a connection of its own, and the
  verifier gets it only after registration: the salt never crosses the outside, so the premise holds as stated, with no
  keystream gap.
* `SaltHiding`: the salt's public commitment `com` hides it up to `δhide`. At NET-18's sealed salt nothing public
  commits to it before `td`, `com` is constant and `δhide = 0`; at PoUW's epoch salt `com` is the coins root's hm96
  leaf, and `δhide` is twice `Claims.Hm96Hiding`'s `δ₁`, `2^−192` at the pinned key (`hash-derived-key`).
* `SaltContainment`: where the boundary audit accepts, what the block leaks (`Attribution.leak`: what it sends, up to
  the window's end, to readers that aren't certifiers) takes at most `2^kc` values across salts. This is the premise
  about the prover's own host. The hardware `H` is to declare every physical channel out of the block as an
  uncertified link with its capacity: power draw (a shared PDU, host sensors), EM, thermal and acoustic emission, shared
  DRAM and PCIe peers on the block's host, the BMC and DPU. `DeclaredLinks` is then a premise about the auditor's model
  of the hardware, and `SaltContainment` says those channels carry fewer than `kc` bits a window about the block's
  coins. The prover owns the block's host and would like to move the salt out of it, so this premise is about hardware
  the prover controls, which the trust model otherwise doesn't rest on; it is a claim, for Daniel
  (`note:20261018T0850Z-draft-rt-attrib` §2, W5 attack 5). A helper in another facility reaches the salt only over the
  certified network; a co-located one is what the premise excludes. With every link out of the block certified
  (`Hardware.SpatialUnit`) and only the declared links (`DeclaredLinks`), it holds at `kc = 0`, by a lemma the proof
  owes: the alternative is to attribute to a sealed region or facility the auditor controls, whose every channel out is
  certified, rather than to a block on the prover's host. Deployed, `kc` plus the boundary's bits (`BoundaryObs`) is
  `k`, and the outsourcing term is `q · 2^(k − s)` at an `s`-bit salt (`OutsideQueries`).
* `SaltFreeJitter`: `TrustedUnits`' jitter clause (W2 and J) restricted to what each certifier sends outside the block
  until the window's end, read off its open record (`Attribution.openCells`, its record but the sealed slots) and a
  jitter word that is a function of the draw alone, never of the salt. A certifier is the auditor's device
  (`trusted-nebius-host`): its jitter is its own clock's. Its sends into the block (the salt frame) are free.
* `BoundaryObs`: at each draw and where the boundary audit accepts, `∂X`'s certifiers' open records placed before the
  window's end are a function of the salt's commitment, the block's leak and the boundary's observations
  (`NetTiming.Obs`) over `m` windows of `T` buckets, an observation sequence the constant-rate Program accepts with
  jitter below `D` (`NetTiming.AccSeqs`), against an operator that may react to what leaked. At each leak value these
  number at most `#Ac · (#Σ_sync · D^T)^m` (`NetTiming.Guarantees.EncardAccSeqsLeConstantRatePerWindow`, proved). It is
  N3 (egress records bound to committed crossings), NET-5 (a declared scheduler) and the coin-cone rule (nothing in the
  salt's cone crosses `∂X` before registration) together, with the salt frame's slot sealed until `td`.
* `OutsideQueries`: the outside's evaluations of the noise hash during the window are at most `q` at every coin.
  Decision 112 (GRIND) takes the outside's query budget at `q = 2^128`, with no economic argument, and the headline
  reads the statement there; `q` stays a parameter.

"The certifiers send salt-dependent data only into the block" is `SaltCustody` before the window and `SaltFreeJitter`
with `BoundaryObs` during it: what a certifier sends outside the block is a function of its open record and its
salt-free jitter, and `∂X`'s open records are a function of what the outside already has.
-/

namespace Accounting.Assumptions

/-- **Salt custody, over the outside** (at each draw `d`): the block `X` is apart from the certifiers `cert`, and what
everything but the block and its boundary's certifiers `bcert` can tell at the window's start (`history` of
`(X ∪ bcert)ᶜ`: the outside's and the other certifiers' inputs, gates and reads) is the same at two window coins with
the same public commitment `com`. Meant at `bcert = Attribution.boundaryCert H X`.

The certifiers on `∂X` may hold the salt: `X`'s ingress certifier draws it, writes it into the block and keeps it in a
sealed slot, and nothing here constrains their state. What they send outside the block is `SaltFreeJitter`'s and
`BoundaryObs`' concern. A certifier off `∂X` holds no salt of `X`'s, since each block's certifier draws its own and no
certifier sends it to another (NET-18's egress side never sees it), so it stays on the salt-free side: the proof's
induction from `∂X`'s records out to the rest of the network reads its state at `ts`. -/
def SaltCustody {Ωs Ωd L C : Type*} [LinearOrder L] [Fintype L] (S : Ωs × Ωd → Actual.Run L) (com : Ωs → C)
    (cert X bcert : Finset L) (w : Window) : Prop :=
  Disjoint X cert ∧ ∀ d ω ω', com ω = com ω' →
    (S (ω, d)).history (X ∪ bcert)ᶜ w.start = (S (ω', d)).history (X ∪ bcert)ᶜ w.start

/-- **The salt's public commitment hides it** (up to `δhide`): the window's coin is an `s`-bit salt and the hiding
randomness `h` of what is published about it before the window's end, `com (salt, h)`, and no event of the salt and the
commitment tells `com (salt, h)` from a commitment to an independent uniform salt under the same `h` with advantage
above `δhide` (W5 attack 7: the coins root binds PoUW's epoch salt).

At NET-18's certifier salt nothing public commits to it before `td` (the salt frame's slot is sealed until then): `com`
is constant and this holds at `δhide = 0`. At PoUW's epoch salt, `com` is the coins root's salt leaf, an hm96 row over
the challenger's coin, and it holds at twice `Claims.Hm96Hiding`'s `δ₁` (both sides within `δ₁` of a fresh mask),
`2^−192` at the pinned key (`hash-derived-key`); both are lemmas the proof owes. -/
def SaltHiding {s : ℕ} {Ωh C : Type} [Fintype Ωh] (com : Fin (2 ^ s) × Ωh → C) (δhide : ℝ) : Prop :=
  ∀ E : Fin (2 ^ s) → C → Prop,
    Pouw.pr (fun ω : Fin (2 ^ s) × Ωh => E ω.1 (com ω)) ≤
      Pouw.pr (fun p : (Fin (2 ^ s) × Ωh) × Fin (2 ^ s) => E p.1.1 (com (p.2, p.1.2))) + δhide

/-- **Salt containment** (at each draw `d`): at the salts where the boundary audit accepts (`bacc`), what the block `X`
leaks (`Attribution.leak`: what it sends, from the gates it places before the window `w` ends, to readers that aren't
certifiers) takes at most `2^kc` values. The hardware declares every physical channel out of `X` (power, EM, thermal,
shared DRAM, PCIe peers, the BMC) as a link, so this is the block's leak about its coins through all of them: fewer
than `kc` bits a window. It holds at every verdict when the channels carry fewer than `kc` bits at every salt.

A premise about hardware the prover hosts, so a claim (W5 attack 5): nothing the prover runs or hosts is otherwise
trusted. It holds at `kc = 0`, with no trust in the prover's host, when the block is a sealed region or facility the
auditor controls, every channel out of it certified (`Hardware.SpatialUnit`). -/
def SaltContainment {Ωs Ωd L : Type*} (S : Ωs × Ωd → Actual.Run L) (bacc : Ωs × Ωd → Prop) (cert X : Finset L)
    (w : Window) (kc : ℕ) : Prop :=
  ∀ d, ((fun s => Attribution.leak (S (s, d)) cert X w) '' {s | bacc (s, d)}).encard ≤ ((2 ^ kc : ℕ) : ℕ∞)

/-- **Salt-free jitter, outside the block**: `TrustedUnits`' W2 and J for what each certifier sends outside the block
`X` until the window `w` ends, with its open record cells `cells` (meant at `Attribution.openCells H.record seal`) in
place of all of them and each certifier's jitter word `J d y` a function of the draw `d` alone. For some jitter words,
one per certifier and draw, each one of `2^jitter` classes a bucket of ticks (`bucket`), what each certifier `y` sends
to readers outside `X` from the gates it places during `[a, b)`, `b` no later than the window's end, is a function of
its own open record placed before `b` (`Actual.Run.record` in `cells y`, as seen) and of its own jitter word on the
buckets of `[a, b)`, at the run's draw.

Its sends into `X` are free (the salt frame carries the salt itself), and so is everything after the window's end (the
salt's commitment and the salt go to the verifier after registration). For this to hold at NET-18's certifier:

* the salt frame's commitment `salt_commitment(salt)` is in no record the certifier publishes before `td`. Either its
  slot in the ingress grid is sealed (the record published in the window carries a salt-free placeholder there) or the
  commitment moves into a record published at `td`; the sealed cells are then that slot and the salt the certifier
  holds for the block (`_salts[w]`), so its send into the block is a function of its record (`TrustedUnits`' W2);
* `reveal(w)` takes window `w`'s registration receipt and refuses without it, failing closed, rather than refusing
  only while `w`'s ingress record is unpublished: a reveal before `td` is a send outside `X` read off a sealed cell. -/
def SaltFreeJitter {Ωs Ωd L : Type*} (S : Ωs × Ωd → Actual.Run L) (cert X : Finset L) (cells : L → Set ℕ)
    (bucket : ℕ → ℕ) (jitter : ℕ) (w : Window) : Prop :=
  ∃ J : Ωd → L → ℕ → Fin (2 ^ jitter), ∀ y ∈ cert, ∀ a b r r', b ≤ w.start + w.len →
    (S r).seen '' ↑(((S r).record {y} cells).filter fun g => (S r).A.t g < b) =
      (S r').seen '' ↑(((S r').record {y} cells).filter fun g => (S r').A.t g < b) →
    (∀ t ∈ Set.Ico a b, J r.2 y (bucket t) = J r'.2 y (bucket t)) →
    (S r).sent {y} (Set.Ico a b) ∩ {p | p.2.1 ∉ X} = (S r').sent {y} (Set.Ico a b) ∩ {p | p.2.1 ∉ X}

/-- **The boundary's observations, modulo the salt frame** (at each draw `d`, where the boundary audit accepts): for
some operator `op`, one for each value of the salt's public commitment `com` and of the block's leak
(`Attribution.leak` of `X` past the certifiers `cert`), and some observation sequence `obs ω` at each window coin, of
`m · T` steps (`m` windows of `T` buckets), at every window coin where `bacc` holds `obs ω` is accepted by the
constant-rate Program (`NetTiming.constantRate`, `Wn` windows with advice, the sync set `ss`, the frames `frames`)
against the operator of that coin's commitment and leak, with jitter classes below `D` (`NetTiming.AccSeqs`); and the
open records the certifiers `bcert` place before the window `w` ends (`Attribution.records` in `cells`, from tick 0)
are the same at two such coins whose commitments, leaks and observations are. Meant at `bcert =
Attribution.boundaryCert H X`, `∂X`'s certifiers, and `cells = Attribution.openCells H.record seal`.

Modulo the salt frame: the sealed slot, which holds the salt and its commitment, is outside `cells`, so the window's
salt frame counts only through its place in the grid, which is the same at every salt (one frame, at the window's first
ingress tick). At NET-18's certifier this holds once the salt's commitment is in no record published before `td`
(`SaltFreeJitter`), with N3, NET-5 and the coin-cone rule. The records are those placed before the window's end, not
only during it, since a certifier's send in the window reads its whole open record before it; those placed before the
salt's release at `ts` are the same at every salt.

The operator is the network certifier model's outside party, which reacts to what it observes; here it may also react
to what reached it over the block's uncertified links, so it is one per leak value, at most `2^kc` of them
(`SaltContainment`), at each commitment. The steps are counted from the start of the boundary audit's epoch, since the
proved count is cumulative from its first window: an epoch that starts at `ts` pays only for the PoUW window's own
windows, an earlier start pays 7 bits for each earlier one. -/
def BoundaryObs {Ωs Ωd L C : Type*} {Lk : ℕ} {In Ac : Type} (K Sz : ℕ) (S : Ωs × Ωd → Actual.Run L)
    (bacc : Ωs × Ωd → Prop) (com : Ωs → C) (cert X bcert : Finset L) (cells : L → Set ℕ) (w : Window) (T Wn : ℕ)
    (ss : NetTiming.SyncSet) (frames : List In → Ac → Fin Lk → List NetTiming.LogicalFrame) (m D : ℕ) : Prop :=
  ∀ d, ∃ (op : C → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → ℕ → List (NetTiming.Obs Lk K Sz) → In)
      (obs : Ωs → List (NetTiming.Obs Lk K Sz)),
    (∀ ω, bacc (ω, d) → obs ω ∈ NetTiming.AccSeqs (NetTiming.constantRate (S := Sz) T Wn ss frames)
      (op (com ω) (Attribution.leak (S (ω, d)) cert X w)) (m * T) D) ∧
    ∀ ω ω', bacc (ω, d) → bacc (ω', d) → com ω = com ω' →
      Attribution.leak (S (ω, d)) cert X w = Attribution.leak (S (ω', d)) cert X w → obs ω = obs ω' →
      Attribution.records (S (ω, d)) bcert cells ⟨0, w.start + w.len⟩ =
        Attribution.records (S (ω', d)) bcert cells ⟨0, w.start + w.len⟩

open Classical in
/-- **The outside's oracle calls**: at every coin, at most `q` of the gates `hash r` (the run's evaluations of the noise
hash, each marked at its output gate) are placed outside the block `X` and the certifiers `cert` during the window
`w`. Decision 112 (GRIND) puts the budget at `q = 2^128` with no economic argument: the outside's compute is not
bounded by its price. At that `q`, `q · 2^(k − s) ≤ 2^−64` needs `k ≤ s − 192`: `k ≤ 320` at a 512-bit salt, `k ≤ 64`
at a 256-bit one. -/
def OutsideQueries {Ω L : Type*} (S : Ω → Actual.Run L) (hash : ∀ r, Finset (Fin (S r).A.N)) (cert X : Finset L)
    (w : Window) (q : ℕ) : Prop :=
  ∀ r, ((hash r).filter fun g => (S r).A.x g ∉ X ∪ cert ∧ (S r).A.t g ∈ w.ticks).card ≤ q

end Accounting.Assumptions
