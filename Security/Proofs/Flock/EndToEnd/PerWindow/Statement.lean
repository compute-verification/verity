import Proofs.Flock.EndToEnd.Period.Statement
import Definitions.Nci.Remote

/-!
# Sampled proofs, window by window over a monitoring period: the bridge to the remote dampener (trusted)

`Nci.Guarantees.DampenerRemotePeriod` takes sampled proofs' legs per window, over a monitoring period of `T` windows on
live coins, window `t`'s coins `Ω t` uniform and independent of the others': the developer's faulty proof blocks `bad ξ`
(block `b` in window `win b`) are fixed before their window's coins; the units a window draws (`drawn`) and the faulty
blocks it detects (`detected`) read only its coins and the earlier windows'; the draw misses the window's `m` faulty
blocks with probability at most `(1 − q)^m`; and the window detects none while it draws one with probability at most
`ks t`. `SamplingBlockPerWindow` says all four follow when the deployment's window `t` holds a sampling block
(`SamplingBlock`) played on its coins (`SamplingBlock.PlayOn`) chosen from the earlier windows' coins alone.

`Flock.Guarantees.IntegrityPeriod` plays a period's sampling blocks as one sequential game against one adaptive prover
(`Period.game`), not over a product of window coins, and a sampling block's game draws more than its subset: the link
points, then a coin per round whose type the prover's message sets (`liveSession`). So the window's coins are carried
onto the game by a map `out : Ω t → outcome` whose law under the window's own coins, `Ω t` uniform and no other
window's, is the game's at the strategy `τ` that plays the sampling block. That is what `SamplingBlock.PlayOn` states
(`PlayOn.law`): `Ω t` is the sampling block's tape, and the earlier windows' coins enter only through the choice of the
play `D t`. Three points:

* **Correctness.** A unit is correct when its committed values equal the Definition's output on its committed inputs:
  `SamplingBlock.wrongSet` is the units not right on the registration (`RightRegZ`: every committed wire the unit reads
  or computes opens to the circuit's evaluation on the transcript's inputs). So a block that skips declared work but
  commits the right values is honest for integrity: skipped work is PoUW's leg to charge, not integrity's.
* **Predictability.** Within a sampling block, its faulty blocks are its wrong units at registration
  (`SamplingBlock.wrongSet`, `wrongRegZ` at `XplurZC`), fixed by the strategy before the game's first coin. Across
  windows, the play `D t` is a function of the earlier windows' coins only, so the prover's choice of window `t`'s
  strategy, statement and registration is adaptive to the past and to nothing drawn at or after `t`.
* **Timing.** A deployment window whose PoUW coins are drawn before its sampled-proofs coins is two coordinates of the
  cut, the PoUW one first; the faulty blocks then sit in the later one, whose play may read the PoUW coins. A coordinate
  with no block (one holding only PoUW's coins) needs no play. `DampenerRemotePeriod`'s order on each identity's
  windows (`τw t < τw t'` for `t < t'` of one identity) is then over the cut's coordinates at their draw times, a
  block's time its sampled-proofs coordinate's: it holds as long as one identity's live coins are drawn one at a time,
  which is the service's schedule, not either protocol's guarantee, so it stays `DampenerRemotePeriod`'s hypothesis.

Reviewer: proofs. The proof, `Flock.SecurityProofs.SamplingBlockPerWindow`, takes, at each window with a block:
`SamplingBlock.drawMiss_le` (under A3) at the image of the window's faulty blocks under `unit` (injective there, so of
size `m`), then `SamplingBlock.one_sub_le` for `(1 − kd/n)^m ≤ (1 − q)^m` (at `q > 0`; `q = 0` is the bound 1);
`SamplingBlock.accepted_drawnWrong_le` (under `CR`) for the second leg, the event within acceptance with a drawn wrong
unit, since a rejected sampling block detects all its faulty blocks; `PlayOn.law` to carry each game probability to
`Nci.Dampener.prob` on `Ω t`; `Nci.SecurityProofs.prob_mono`; and, for the first two conclusions, that `D t` reads the
past only. What no proof here gives is that the deployment's window holds such a play: nothing ties the sampled-proofs
service's sampling block (`verity/core/protocols/sampled_proofs/`) to a `SamplingBlock`'s statement, law and record,
or its live coins to a uniform tape. A `SamplingBlock` draws a uniform `kd`-subset of its units, where the service
draws one stratum per template inside the one stratified draw it shares with PoUW, so the lane decided provisionally
(26 Oct) to restate the per-window legs over `Flock.Guarantees.JointStratified` (#4560 stated, #4590 proved).

C-Flock's definitions are still in its proofs package, so, like `IntegrityPeriod`, this module imports them.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkExec
open scoped ENNReal

/-- **A `SamplingBlock` played on coins `Ω'`, with blocks `β`**: the sampling block `w`, the strategy `τ` that plays
it, its outcome at each value of the coins (`out`: the registration, the drawn units and the transcript), whose law
under the window's own coins `Ω'`, uniform, is the game's at `τ` (`law`), and the unit each block is (`unit`). -/
structure SamplingBlock.PlayOn (Ω' : Type) [Fintype Ω'] [Nonempty Ω'] (β : Type) where
  w : SamplingBlock
  τ : Strategy w.game
  out : Ω' → w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript
  law : ∀ E : w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript → Prop,
    Nci.Dampener.prob (PMF.uniformOfFintype Ω') (fun a => E (out a)) = (prob E w.game τ).toReal
  unit : β → Fin w.n

/-- **Sampled proofs, window by window** (the bridge to `Nci.Guarantees.DampenerRemotePeriod`'s sampled-proofs legs). A
monitoring period of `T` windows, window `t` with its own uniform coins `Ω t` (a window of the deployment, or a finer
cut's coordinate), the developer's faulty proof blocks `bad ξ`, block `b` in window `win b`, and the units each window
draws (`drawn`) and the faulty blocks it detects (`detected`). Suppose every window `t` that holds a block holds a
sampling block (`SamplingBlock`) played on its coins, `D t`, chosen from the earlier windows' coins alone, such that at
every run `ξ`, with `P = D t ξ|_{<t}`:

* the window's blocks are distinct units of the sampling block (`P.unit` injective on them);
* a block is faulty when its unit is wrong at registration (`SamplingBlock.wrongSet` at `P.τ`: its committed values
  differ from the Definition's output on its committed inputs, so a block that skips declared work but commits the
  right values is not faulty), drawn when its unit is in the sampling block's draw, and detected when it is faulty and
  flock-verify rejects the sampling block (`LiveAcceptsZK`);
* `EndToEndHidden`'s premises hold at the sampling block: A3 (`Assumptions.UniformRandomBytes`) for its draw bytes and
  `cr/sha-512` and `ecr/sha-512` (`SamplingBlock.CR`) at `P.τ` (`live-verifier` is the sampling block's own `hCust`);
* its draw covers at least a fraction `q ≥ 0` of its units (`q ≤ kd/n`), and `ε_ks + δ_link ≤ ks t`, `ks t ≥ 0` at
  every window.

Then `DampenerRemotePeriod`'s four sampled-proofs hypotheses hold: the faulty blocks of window `t` are fixed by the
windows before it, its drawn and detected blocks by those up to it, its draw misses its `m` faulty blocks with
probability at most `(1 − q)^m`, and it detects none of them while it draws one with probability at most `ks t`, each
over window `t`'s own coins (`ξ[t ↦ a]`, `a` uniform on `Ω t`) at every value of the others'. -/
def SamplingBlockPerWindow : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)] (β : Type)
    (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T) (q : ℝ) (ks : Fin T → ℝ),
    0 ≤ q → (∀ t, 0 ≤ ks t) →
    (∀ t, (∃ b, win b = t) →
      ∃ D : ((s : {s : Fin T // s < t}) → Ω s.1) → SamplingBlock.PlayOn (Ω t) β,
        ∀ ξ : (t : Fin T) → Ω t,
          let P := D fun s => ξ s.1
          (∀ b b', win b = t → win b' = t → P.unit b = P.unit b' → b = b') ∧
          (∀ b, win b = t → (b ∈ bad ξ ↔ P.unit b ∈ P.w.wrongSet P.τ)) ∧
          (∀ b, win b = t → (b ∈ drawn ξ ↔ P.unit b ∈ (P.out (ξ t)).2.1)) ∧
          (∀ b, win b = t → (b ∈ detected ξ ↔ b ∈ bad ξ ∧ ¬ LiveAcceptsZK P.w.I (P.out (ξ t)))) ∧
          Assumptions.UniformRandomBytes P.w.Lb P.w.src ∧ P.w.CR P.τ ∧
          q ≤ (P.w.kd : ℝ) / P.w.n ∧ P.w.ks + P.w.link ≤ ENNReal.ofReal (ks t)) →
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) ∧
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
          (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
          (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
            ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t)

/-- **`SamplingBlockPerWindow`'s nontrivial lemma: its conclusion fails for a verifier that accepts everything.** A
verifier that accepts every sampling block detects no block (`detected ξ = ∅`: a block is detected only when flock-verify
rejects its sampling block). At a period whose draw takes every block and whose window `t₀` holds a faulty block at
every run, window `t₀` then draws a faulty block and detects none on every value of its coins, so the conclusion's last
leg, and with it the conclusion, fails at every `ks t₀ < 1`. Proof:
`Flock.SecurityProofs.SamplingBlockPerWindow_nontrivial`. -/
def SamplingBlockPerWindow_nontrivial : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)] (β : Type)
    (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T) (q : ℝ) (ks : Fin T → ℝ) (t₀ : Fin T),
    (∀ ξ, detected ξ = ∅) → (∀ ξ b, b ∈ drawn ξ) → (∀ ξ, ∃ b ∈ bad ξ, win b = t₀) → ks t₀ < 1 →
    ¬ ((∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) ∧
      (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
        (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) ∧
      (∀ t (ξ : (t : Fin T) → Ω t),
        Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
            (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
          (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) ∧
      (∀ t (ξ : (t : Fin T) → Ω t),
        Nci.Dampener.prob (PMF.uniformOfFintype (Ω t))
            (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
              ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t))

end Flock.Guarantees
