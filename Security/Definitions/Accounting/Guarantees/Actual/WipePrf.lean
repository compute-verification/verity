import Definitions.Accounting.Actual.WipePrf
import Definitions.Pous.Model.StreamedWipe
import Definitions.Accounting.Guarantees.Actual.WipeBridge

/-!
# The store held read by read, on the deployed labels (trusted)

`Accounting.Guarantees.Actual.WipeReadHeld` takes `Accounting.Assumptions.UniformLabels`, uniform labels given the
auditor's devices' coins. The deployed labels are SHA-512 of a wipe's key and the index (PR #2726), with each wipe's key
derived from the auditor's secret for the server and the wipe (`streamed_wipe.wipe_key`), a PRF of the secrets, for
which that premise is false. `WipeReadHeldPrf` states the guarantee on them, in PoUS's streamed-label wipe's form
(`Pous.Guarantees.StreamedWipe`): no named label assumption, and the PRF (C3, claim `prf/sha-512`) enters as an
advantage term in the conclusion, `Pous.StreamedWipe.prfAdvantage` against an explicit distinguisher, which C3 bounds
for an efficient one. Proof: `Proofs/Accounting/Actual/WipePrf.lean`.

`WipeReadHeld`'s `n` reads span the server's life, so they span its wipes. The labels are one lifetime table
`F : Fin (W * N) → Bits b`, wipe `w`'s label of block `i` at slot `finProdFinEquiv (w, i)`, with `wipe j` read `j`'s
wipe (`Actual.tableLabel`). The deployed table is
`prf s (w, i) = SHA-512(tag ‖ wipe_key(s_w, server, session_w) ‖ i)` for the auditor's secrets `s = (s_w)`, one a wipe,
concatenated; one wipe is `W = 1`. The system is a run of the other coins `ω`, the table `F` and the indices `c`,
`S ((ω, F), c)` (`Definitions.Accounting.Actual.WipePrf`): the challenger uses the secrets only to compute labels, and
the prover never sees them. The deployed system is `S` at `F = prf s` for a uniform `s`. `WipeReadHeld`'s premises are stated of `S` at
every table, since the physics and the challenger's code don't depend on the labels' values. The devices' coins `ζ`
are among `ω`, so they are independent of the table.

A uniform table meets `UniformLabels`, so `WipeReadHeld` bounds the event there. The event is decided by one run
(`Actual.heldEvent`). So the step from the uniform table to `prf s` costs the advantage of the distinguisher that runs
the system once on the table it is given and outputs the event (`Actual.heldDistinguisher`). It asks the table only
what the audits ask it, and it never reads the secrets.

Not covered here, and so outside this guarantee: the indices are uniform. The deployed ones are
`draw_indices(coin_secret, …)`, a keyed SHA-512 stream of one 32-byte secret a wipe, all drawn at the wipe's first batch
(`streamed_wipe.Challenger.challenge`). `Accounting.Guarantees.Actual.WipeReadHeldPrfDraw` takes the step to them, as
the coin function's advantage against a distinguisher of the same form plus the sampler's rejection term.
-/

namespace Accounting.Guarantees.Actual

/-- **The store held read by read, on the deployed labels**. The labels of `W` wipes of `N` blocks are one table
`prf s : Fin (W * N) → Bits b` for uniform secrets `s : Bits κ`, and read `j` reads wipe `wipe j`. The system is a run
`S ((ω, F), c)` of the other coins `ω`, the table `F` and the `n` indices `c` on the store `Fin N`. The deployed system
is that run at `F = prf s`. For any `prf` and `wipe`, take `WipeReadHeld`'s premises for that system, stated of it at
every table:

* `CausalStrategy`, and a latency ceiling `ℓ⁺` from `X` to the challenger's answer leaves `Y`
  (`Placed.LatencyAtMost`, claim `latency-ceiling`);
* `LiveReads` and `LiveVerdicts`, the latter for the challenger's verdict `Ch j` on its view;
* `QuietReads`, up to the devices' coins `ζ`, which are a function of `ω`;
* `LateAnswers`, with each block's label at a read as the run carries it, its wipe's entry (`Actual.tableLabel`).

Then the chance over `ω`, `c` and `s` that every read is accepted and arrives at a tick of `T`, and at least `K` reads
find `X`'s state narrower than `m` bits, is at most `WipeReadHeld`'s bound
`exp(−K (1 − (a − 1)/N)) + n |T| 2^m C(N, a) 2^{-a b}`, plus the PRF's advantage against the distinguisher the audits
make (`Actual.heldDistinguisher`). That distinguisher runs the system once on the table it is given and outputs the
event.

No named assumption on the labels: `UniformLabels` is not a premise. The advantage is C3's term (`prf/sha-512`), and C3
bounds it for a distinguisher as efficient as the audits together with the prover's own strategy, as in
`Pous.Guarantees.StreamedWipe`; for the deployed `prf`, by a hybrid over the wipes, each through `wipe_key`'s
derivation and then its labels.
The indices are uniform (the module docstring says what that leaves out). Proof:
`Accounting.SecurityProofs.Actual.WipeReadHeldPrf`. -/
def WipeReadHeldPrf : Prop :=
  ∀ {Ω L Z : Type} [Fintype Ω] [LinearOrder L] {n W N b κ : ℕ} (prf : Pous.Bits κ → Fin (W * N) → Pous.Bits b)
    (wipe : Fin n → Fin W) (S : (Ω × (Fin (W * N) → Pous.Bits b)) × (Fin n → Fin N) → Actual.Run L)
    (ℓ : L → L → ℕ) (X Y V : Finset L) (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (D ℓmax a m K : ℕ)
    (arrive : (Ω × (Fin (W * N) → Pous.Bits b)) × (Fin n → Fin N) → Fin n → ℕ) (ζ : Ω → Z)
    (quiet : Fin n → ℕ → Fin N → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) (T : Finset ℕ),
    CausalStrategy S ℓ → (∀ r, (S r).A.LatencyAtMost X Y ℓmax) → Assumptions.LiveReads S X arrive →
    Assumptions.LiveVerdicts (fun r j => Ch j ((S r).view V)) →
    Assumptions.QuietReads S X arrive D (fun r j => Ch j ((S r).view V)) (fun ω₀ => ζ ω₀.1) quiet →
    Assumptions.LateAnswers S X Y (Actual.tableLabel wipe) arrive ℓmax D (fun r j => Ch j ((S r).view V)) →
    Pous.pr (fun x : (Ω × (Fin n → Fin N)) × Pous.Bits κ =>
        Actual.heldDistinguisher S X V Ch arrive T m K (x.1, prf x.2)) ≤
      ENNReal.ofReal (Real.exp (-((K : ℝ) * (1 - ((a : ℝ) - 1) / N))) +
          n * T.card * 2 ^ m * (N.choose a : ℝ) / 2 ^ (a * b)) +
        Pous.StreamedWipe.prfAdvantage prf (Actual.heldDistinguisher S X V Ch arrive T m K)

end Accounting.Guarantees.Actual
