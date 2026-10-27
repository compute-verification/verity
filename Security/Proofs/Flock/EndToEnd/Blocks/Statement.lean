import Proofs.Flock.EndToEnd.Period.Statement

/-!
# The integrity audit per server-time block, and over a horizon of blocks (trusted)

A server's life is cut into server-time blocks by certified wipes. Over a horizon, its `--zk` sessions are numbered in
serve order, session `t` a `SamplingBlock` `W t` with a fresh live subset draw, and block `b` is the sessions from
`start b` up to `start (b + 1)` (`ServerBlocks`): a wipe separates block `b`'s last session from block `b + 1`'s
first. One prover strategy plays every session of every block (`Period.game` over the horizon's sessions), adaptively,
so a block's sessions may depend on everything the prover saw in the blocks before.

* `IntegrityBlock`: for block `b`, the probability that every session of block `b` is accepted and its wrong units total
  at least `ln(1/δ)/q` is at most `δ + Σ_{t ∈ b} (ε_ks,t + δ_link,t)`, where each of block `b`'s draws covers at least a
  fraction `q` of its session's units. The block's bound holds inside the horizon, against the prover that played the
  blocks before it.
* `IntegrityBlocks`: the probability that some block is accepted with at least `ln(1/δ)/q` wrong units is at most
  `Σ_b (δ + Σ_{t ∈ b} (ε_ks,t + δ_link,t))`: the per-block form a joint horizon over blocks sums with the other legs.
* `IntegrityHorizonBlocks`: the probability that every session of every block is accepted and the wrong units over the
  horizon total at least `ln(1/δ)/q` is at most `δ + Σ_b Σ_{t ∈ b} (ε_ks,t + δ_link,t)`, the pooled reading. It is a
  lemma: the listed guarantee of the pooled reading is `IntegrityHorizonCarried`, which also covers carried state.

The named assumptions are `IntegrityHorizon`'s, at the sessions each statement reads: A3 (`uniform/io-getrandombytes`)
for each session's draw bytes, `live-verifier` (`SamplingBlock.hCust`), and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512`
(`LinkCRZC`) at every strategy a session is played with on some run (`Period.Plays`). The proofs are
`Flock.SecurityProofs.IntegrityBlock`, `IntegrityBlocks` and `IntegrityHorizonBlocks`.

A wrong unit is counted in the session that holds it (`SamplingBlock.wrongSet`). The state a block carries across a wipe
(the KV cache, committed before the wipe and read after it) is an input of the next block's sessions, and these
statements say nothing of whether the value a session reads there is the one the block before committed;
`IntegrityHorizonCarried` (`EndToEnd/Carry/Statement.lean`) does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-- **A server's sessions over a horizon, cut at wipes into `B` server-time blocks.** Session `t` is `W t`, in serve
order; block `b` is the sessions from `start b` up to `start (b + 1)`, so the blocks partition the horizon's first
`start B` sessions, and a wipe sits before session `start b` for each `0 < b < B`. -/
structure ServerBlocks where
  W : ℕ → SamplingBlock
  B : ℕ
  start : ℕ → ℕ
  start_zero : start 0 = 0
  mono : Monotone start

namespace ServerBlocks

variable (S : ServerBlocks)

/-- The horizon's number of sessions. -/
def T : ℕ := S.start S.B

/-- Block `b`'s sessions. -/
def sessions (b : ℕ) : Finset ℕ := Finset.Ico (S.start b) (S.start (b + 1))

/-- The horizon's game: every block's sessions in serve order, against one prover. -/
noncomputable def game : Game Unit := Period.game S.W S.T

/-- The horizon's run at the prover strategy `s`: each session's record, in serve order. -/
noncomputable def run (s : Strategy S.game) : Game (List WindowRec) := Period.run S.W S.T s

/-- Block `b`'s wrong units in a run's records. -/
def blockWrong (b : ℕ) (l : List WindowRec) : ℕ := ((l.filter fun r => r.idx ∈ S.sessions b).map WindowRec.wrong).sum

/-- **Block `b` escapes at `ln(1/δ)/q`**: every session of block `b` is accepted, and its wrong units total at least
`ln(1/δ)/q`. -/
def BlockEscapes (q δ : ℝ) (b : ℕ) (l : List WindowRec) : Prop :=
  (∀ r ∈ l, r.idx ∈ S.sessions b → r.accepted) ∧ Real.log (1 / δ) / q ≤ (S.blockWrong b l : ℝ)

/-- Block `b`'s knowledge and link terms, `Σ_{t ∈ b} (ε_ks,t + δ_link,t)`. -/
noncomputable def err (b : ℕ) : ℝ≥0∞ := ∑ t ∈ S.sessions b, ((S.W t).ks + (S.W t).link)

end ServerBlocks

/-- **The integrity audit of one server-time block.** In a horizon of server-time blocks as in `ServerBlocks`, played
by one adaptive prover `s`, if each of block `b`'s draws covers at least a fraction `q > 0` of its session's units, the
probability that every session of block `b` is accepted and its wrong units total at least `ln(1/δ)/q` is at most
$$\delta + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}).$$
Its named assumptions are `IntegrityHorizon`'s, at block `b`'s sessions. -/
def IntegrityBlock : Prop :=
  ∀ (S : ServerBlocks) (b : ℕ) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t ∈ S.sessions b, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t ∈ S.sessions b, Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ s : Strategy S.game,
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    value (S.BlockEscapes q δ b) (S.run s) ≤ ENNReal.ofReal δ + S.err b

/-- **The integrity audit per server-time block, over a horizon of `B` blocks.** In a horizon as in `IntegrityBlock`
whose draws each cover at least a fraction `q > 0` of their session's units, the probability that some block is
accepted with at least `ln(1/δ)/q` wrong units is at most
$$\sum_{b < B} \Big(\delta + \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t})\Big).$$
Its named assumptions are `IntegrityHorizon`'s. -/
def IntegrityBlocks : Prop :=
  ∀ (S : ServerBlocks) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < S.T, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t < S.T, Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ s : Strategy S.game,
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    value (fun l => ∃ b < S.B, S.BlockEscapes q δ b l) (S.run s) ≤
      ∑ b ∈ Finset.range S.B, (ENNReal.ofReal δ + S.err b)

/-- **The integrity audit over a horizon of `B` server-time blocks, pooled.** In a horizon as in `IntegrityBlocks`, the
probability that every session of every block is accepted and the wrong units over the horizon total at least
`ln(1/δ)/q` is at most
$$\delta + \sum_{b < B} \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}).$$
Its named assumptions are `IntegrityHorizon`'s. -/
def IntegrityHorizonBlocks : Prop :=
  ∀ (S : ServerBlocks) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < S.T, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t < S.T, Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ s : Strategy S.game,
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ)) (S.run s) ≤
      ENNReal.ofReal δ + ∑ b ∈ Finset.range S.B, S.err b

end Flock.Guarantees
