import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Definitions.Accounting.Physical
import Definitions.Accounting.Assumptions.WipeRead

/-!
# The lifetime pool on the actual circuit: the pool, the count and the lag (trusted)

The lemma `Accounting.SecurityProofs.Actual.WipePool` bounds the un-held fractions a run tree of wipes labels
(`Accounting.WipePlay`) on a tree meeting `WipePlay.UnheldFails`, which every tree meets once each un-held set is
empty. These state the pool on the challenger's actual verdicts instead, per read, what an answered block costs the
server's state at the read, and what the state can gain between reads. Restated by the MEM-55 and MEM-60 red teams, and
proved in `Proofs/Accounting/Actual/WipeReadPool.lean`, `WipeBridge.lean` and `WipeLag.lean`.

The setting is `Definitions.Accounting.Assumptions.WipeRead`'s: a strategy `S` on coins `r = (ω₀, c)`, `n` timed reads over
its life with uniform indices `c : Fin n → B` on a store `B`, labels `lab ω₀ j` of `b` bits, read `j`'s index reaching
the server `X` at `arrive r j`, the challenger's verdict `acc r j`, and read `j`'s un-answered fraction `x_j`
(`Accounting.Actual.unanswered`).

* `WipeReadPool` (the pool): `Pr[every read accepted ∧ Σ_j x_j ≥ H] ≤ exp(−H)`, under live verdicts alone. It is
  the lemma `WipePool` at `k = 1`, one read a wipe, with each read's `U` its un-answered blocks: `UnheldFails` holds
  of that `U` by definition, so this is arithmetic, not physics.
* `AnsweredOfQuiet` (the count, the physics): at any read, the coins on which `X`'s content at the arrival is one of a
  set `C` of contents, at a tick of a set `T`, and the challenger would accept at least `a` of the indices, are at most
  a `|T| |C| C(|B|, a) 2^{-a b}` share. Each accepted index's label is fixed by the content, the tick and the index, so
  `a` answered blocks in `log₂ |C|` bits of content are `a` labels guessed, at `b − log₂ (e|B|/a)` bits a block.
* `WipeLag` (the lag): over any coins, the contents of `X`'s state at `t` number at most those at `s ≤ t` times
  `2^{β_X (t − s)}`, `β_X` the capacity of `X`'s boundary links. At any tick `s` between reads, the bits the state
  varies in are at least those at the next read `t` less `β_X (t − s)`: dropping between reads costs ingress, read by
  read. Nothing bounds the state between a wipe's last read and the next wipe's fill but the cap's own ingress term
  (`r ω` in `Accounting.Guarantees.Actual.ThreadedRate`).

* `WipeReadHeld` (the guarantee the two compose to): the coins on which the challenger, a function of its view,
  accepts every read and at least `K` reads find `X`'s state under `m` bits are at most
  `exp(−K (1 − (a − 1)/|B|)) + n |T| 2^m C(|B|, a) 2^{-a b}`. A read under `m` bits answers fewer than `a` blocks
  except on the second term (`AnsweredOfQuiet` with `C` the contents of fewer than `m` bits, a content's length being
  the state's width `Actual.Run.mem`), so it fails with chance above `1 − (a − 1)/|B|`, and `WipeReadPool` at
  `H = K (1 − (a − 1)/|B|)` bounds the rest.

Taking `K = n` gives the bound for a server whose content at every read is among `C`, which says only that the store
was held at one read; the count over reads is what bounds held bytes read by read, as the time average of the freed
fractions does in PoUS's continuous audit (`Pous.Guarantees.ContinuousAudit`). The counting term is StreamedWipe's
(`Pous.Guarantees.StreamedWipe`), per read. `WipeReadHeld`'s event is decided by one run (acceptance, arrival ticks and
the state's width at each read), so the step from `UniformLabels` to the deployed PRF labels, and from uniform indices
to `coin_secret`'s derivation, is an advantage against the distinguisher that runs the system once, as
`Pous.StreamedWipe.prfAdvantage` states it; `answered` and `unanswered` need `|B|` counterfactual runs a read and stay
inside the proof. `Accounting.Guarantees.Actual.WipeReadHeldPrf` takes the labels' step, with no label assumption,
and `Accounting.Guarantees.Actual.WipeReadHeldPrfDraw` the indices' step too, with no index assumption.
-/

namespace Accounting.Guarantees.Actual

open Finset

open Classical in
/-- **The lifetime pool per read, on the challenger's verdicts**. For verdicts `acc` on coins
`r = (ω₀, c)` with `n` uniform indices on `B`, each fixed by the coins up to its own read (`Assumptions.LiveVerdicts`),
and any `H`, the share of the coins on which the challenger accepts every read and the reads' un-answered fractions
(`Actual.unanswered`) sum to at least `H` is at most `exp(−H)`. -/
def WipeReadPool : Prop :=
  ∀ {Ω₀ B : Type*} [Fintype Ω₀] [Fintype B] {n : ℕ} (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (H : ℝ),
    Assumptions.LiveVerdicts acc →
    ((univ.filter fun r => (∀ j, acc r j) ∧ H ≤ ∑ j, Actual.unanswered acc j r).card : ℝ) ≤
      Real.exp (-H) * Fintype.card (Ω₀ × (Fin n → B))

/-- **What an answered block costs**. A causal strategy `S` (`CausalStrategy`: causal from every
tick, and each run at the latencies `ℓ`) with a latency ceiling of `ℓ⁺` from `X` to the leaves `Y` where the challenger
reads its answers (`Placed.LatencyAtMost`), whose reads are live (`Assumptions.LiveReads`) and quiet for the deadline
`D` when accepted, up to the devices' own coins `ζ` (`Assumptions.QuietReads`), with a verdict that accepts only late
answers that are the label (`Assumptions.LateAnswers`) and `b`-bit labels uniform given `ζ`
(`Assumptions.UniformLabels`): for any read `j`, set `T` of ticks, set `C` of contents and count `a`, the share of the
coins on which read `j` arrives at a tick of `T`, `X`'s content there is in `C`, and at least `a` blocks are answered
(`Actual.answered`) is at most `|T| |C| C(|B|, a) 2^{-a b}`. Given the tick, the content and `ζ`, an index the
challenger would accept had its window's crossing fixed, so what `X` sent then, so its label: `a` answered blocks are
`a` labels equal to values the tick, the content and `ζ` fix, and given `ζ` the labels are still uniform. -/
def AnsweredOfQuiet : Prop :=
  ∀ {Ω₀ B L Z : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (ℓ : L → L → ℕ) (X Y : Finset L) (lab : Ω₀ → Fin n → B → ℕ)
    (b D ℓmax a : ℕ) (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (ζ : Ω₀ → Z)
    (quiet : Fin n → ℕ → B → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) (T : Finset ℕ) (C : Finset (List Bool))
    (j : Fin n),
    CausalStrategy S ℓ → (∀ r, (S r).A.LatencyAtMost X Y ℓmax) → Assumptions.LiveReads S X arrive →
    Assumptions.QuietReads S X arrive D acc ζ quiet → Assumptions.LateAnswers S X Y lab arrive ℓmax D acc →
    Assumptions.UniformLabels lab b ζ →
    ((univ.filter fun r => arrive r j ∈ T ∧ (S r).content X (arrive r j) ∈ C ∧
        a ≤ (Actual.answered acc j r).card).card : ℝ) ≤
      T.card * C.card * ((Fintype.card B).choose a : ℝ) / 2 ^ (a * b) * Fintype.card (Ω₀ × (Fin n → B))

/-- **The lag between reads**. For a strategy `S` causal from every tick (`CausalFrom`) on links
as declared (`DeclaredLinks`: each link `e` delivers one of at most `2^{β e}` signals a tick each way, and every wire
between leaves runs on a declared link), a location `X`, ticks `s ≤ t` and any set `K` of coins, the contents of `X`'s
state at `t` over `K` (`Actual.contents`) number at most those at `s` times `2^{β_X (t − s)}`, where `β_X` is the
capacity of the links across `X`'s boundary (`linkCapacity`): what `X` holds at `t` is a function of what it held at
`s` and of what crossed into it meanwhile. It is `Accounting.SecurityProofs.Actual.acts_encard_le_links`'s count
(`Proofs/Accounting/Actual/Transducer.lean`) on `X`'s state. -/
def WipeLag : Prop :=
  ∀ {Ω L E : Type*} [LinearOrder L] [Fintype E] (S : Ω → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (X : Finset L) (s t : ℕ) (K : Set Ω),
    CausalFrom S → DeclaredLinks S ends β link → s ≤ t →
    (Actual.contents S X t K).encard ≤
      (Actual.contents S X s K).encard * ((2 ^ (linkCapacity ends β X * (t - s)) : ℕ) : ℕ∞)

open Classical in
/-- **The store held read by read**. The challenger at the leaves `V` decides read `j` by
`Ch j` of its view (`Actual.Run.view`), as `Accounting.Sound` reads a verifier. Under `WipeReadPool`'s and
`AnsweredOfQuiet`'s premises for that verdict, for any ticks `T`, any count `a`, any width `m` and any `K`, the share of
the coins on which every read is accepted and arrives at a tick of `T`, and at least `K` reads find `X`'s state
narrower than `m` bits at the arrival (`Actual.Run.mem`), is at most
`exp(−K (1 − (a − 1)/|B|)) + n |T| 2^m C(|B|, a) 2^{-a b}`.

At the deployed wipe (`|B| = 40,625,000,000` labels of 512 bits, `n = 74,000`, `|T| ≤ 2^50`) the second term is under
`2^-41` once `m ≤ 512 a − log₂ C(|B|, a) − 108`. With both terms under `2^-41`, a passing server held at least 90% of
the store's bits at all but 286 of the reads and half of them at all but 57; `K = n`, with `a = |B| − 15,601,665`, says
only that it held the store less 1.02 GB at one read (note:20261019T2210Z-report-timed-round-redteam-3). -/
def WipeReadHeld : Prop :=
  ∀ {Ω₀ B L Z : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (ℓ : L → L → ℕ) (X Y V : Finset L)
    (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (lab : Ω₀ → Fin n → B → ℕ) (b D ℓmax a m K : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (ζ : Ω₀ → Z)
    (quiet : Fin n → ℕ → B → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) (T : Finset ℕ),
    CausalStrategy S ℓ → (∀ r, (S r).A.LatencyAtMost X Y ℓmax) → Assumptions.LiveReads S X arrive →
    Assumptions.LiveVerdicts (fun r j => Ch j ((S r).view V)) →
    Assumptions.QuietReads S X arrive D (fun r j => Ch j ((S r).view V)) ζ quiet →
    Assumptions.LateAnswers S X Y lab arrive ℓmax D (fun r j => Ch j ((S r).view V)) →
    Assumptions.UniformLabels lab b ζ →
    ((univ.filter fun r => (∀ j, Ch j ((S r).view V) ∧ arrive r j ∈ T) ∧
        K ≤ (univ.filter fun j => (S r).mem X (arrive r j) < m).card).card : ℝ) ≤
      (Real.exp (-((K : ℝ) * (1 - ((a : ℝ) - 1) / Fintype.card B))) +
        n * T.card * 2 ^ m * ((Fintype.card B).choose a : ℝ) / 2 ^ (a * b)) * Fintype.card (Ω₀ × (Fin n → B))

end Accounting.Guarantees.Actual
