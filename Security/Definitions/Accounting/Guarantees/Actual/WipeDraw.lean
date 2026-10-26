import Definitions.Accounting.Actual.WipeDraw
import Definitions.Accounting.Guarantees.Actual.WipePrf

/-!
# The store held read by read, on the deployed labels and the deployed index draw (trusted)

`Accounting.Guarantees.Actual.WipeReadHeldPrf` states the guarantee on the deployed labels and keeps the indices
uniform. The deployed index of read `j` is `Key.uniform(N, j)` under a key derived from the challenger's coin secret
(`Definitions.Accounting.Actual.WipeDraw`). `WipeReadHeldPrfDraw` states the guarantee on those indices too, in the
same form: no named assumption on the indices, and the coin function enters as its advantage against an explicit
distinguisher that runs the system once, beside the labels' advantage. Proof: `Proofs/Accounting/Actual/WipeDraw.lean`.

* **The draw.** Read `j`'s index is `drawIndex (rest s' j) (coins s' j)`. `coins s' j` is read `j`'s first word, a
  keyed function of the coin secret `s'`, and the sampler on it is the deployed one, exactly (`Actual.wordIndex`).
  `rest s' j` is what the deployed sampler draws from the later words when it rejects the first, and the guarantee
  holds for every `rest`. For the deployed code, `coins s' j` is the first `w` bits of
  `SHA-512(stream tag ‖ derive(s'_v, COIN_DOMAIN, {server, session, n: N}) ‖ frames("u", i) ‖ u64 0)`, with `s'` the
  challenger's coin secrets, one a wipe, concatenated, `v = wipe j`, `i` read `j`'s place among its wipe's reads, and
  `w = 8 ⌈(bitlen N + 64) / 8⌉`.
* **The secrets.** `s'` is uniform and independent of the label secrets `s` and the other coins `ω`: the challenger
  draws it from the OS on its own machine when the fill completes (`Challenger.challenge`, with no `coin_secret` given).
* **What the server sees.** The system is `S ((ω, F), c)` as in `WipeReadHeldPrf`, a function of the indices and not of
  `s'`: the challenger reads the coin secret only to draw the indices, keeps them, reveals them a batch at a time and
  publishes them once the session is signed (`Challenger.coins`). When each index reaches the server is the premises'
  business, stated of `S` at every index vector: `LiveReads` says nothing in `X` before read `j`'s arrival depends on
  its index.

The proof is two hybrids. At the labels `prf s`, the deployed event implies the distinguisher's on `coins s'`: an
accepted first word fixes the index, and a rejected one makes the distinguisher output true. So the step from uniform
words to `coins s'` costs the coin function's advantage against `Actual.drawDistinguisher`, which holds `s` and `ω`,
runs the system once and reads the `n` words. On uniform words the accepted ones give each index a uniform share, and
some read's word is rejected with probability at most `n (2^w mod N) / 2^w`. The rest is `WipeReadHeldPrf` at uniform
indices.

Left to C3 (claim `prf/sha-512`, `verity.ml.randomness`'s keyed streams), as `WipeReadHeldPrf` leaves the labels: Lean
states one advantage for the lifetime coin function, while the deployed coins take a fresh secret a wipe, through
`derive` and then its stream. Bounding the lifetime advantage by the wipes' is the standard hybrid, on C3's side.
-/

namespace Accounting.Guarantees.Actual

/-- **The store held read by read, on the deployed labels and the deployed index draw**. The labels of `W` wipes of
`N` blocks are one table `prf s : Fin (W * N) → Bits b` for uniform secrets `s : Bits κ`, and read `j` reads wipe
`wipe j`, as in `WipeReadHeldPrf`. Read `j`'s index is the deployed sampler's on its first word `coins s' j : Bits w`,
a keyed function of the challenger's uniform coin secret `s' : Bits κ'`, or `rest s' j` when it rejects that word
(`Actual.drawIndex`). The system is a run `S ((ω, F), c)` of the other coins `ω`, the table `F` and the `n` indices
`c`, and the deployed system is that run at `F = prf s` and the drawn indices. For any `prf`, `coins`, `rest` and
`wipe`, take `WipeReadHeld`'s premises for that system, stated of it at every table and every index vector:

* `CausalStrategy`, and a latency ceiling `ℓ⁺` from `X` to the challenger's answer leaves `Y`
  (`Placed.LatencyAtMost`, claim `latency-ceiling`);
* `LiveReads` and `LiveVerdicts`, the latter for the challenger's verdict `Ch j` on its view;
* `QuietReads`, up to the devices' coins `ζ`, which are a function of `ω`;
* `LateAnswers`, with each block's label at a read as the run carries it, its wipe's entry (`Actual.tableLabel`).

Then the chance over `ω`, `s` and `s'` that every read is accepted and arrives at a tick of `T`, and at least `K` reads
find `X`'s state narrower than `m` bits, is at most `WipeReadHeld`'s bound
`exp(−K (1 − (a − 1)/N)) + n |T| 2^m C(N, a) 2^{-a b}`, plus the labels' advantage against the distinguisher the audits
make at uniform indices (`Actual.heldDistinguisher`), plus the coin words' advantage against the distinguisher the
audits make at the labels `prf s` (`Actual.drawDistinguisher`), plus `n (2^w mod N) / 2^w`, the chance that the
sampler rejects some read's first uniform word. Each distinguisher runs the system once.

No named assumption on the labels or the indices. The two advantages are C3's terms (`prf/sha-512`), each bounded for
a distinguisher as efficient as the audits together with the prover's own strategy, and for the deployed functions by
a hybrid over the wipes. Proof: `Accounting.SecurityProofs.Actual.WipeReadHeldPrfDraw`. -/
def WipeReadHeldPrfDraw : Prop :=
  ∀ {Ω L Z : Type} [Fintype Ω] [LinearOrder L] {n W N b κ κ' w : ℕ} (prf : Pous.Bits κ → Fin (W * N) → Pous.Bits b)
    (coins : Pous.Bits κ' → Fin n → Pous.Bits w) (rest : Pous.Bits κ' → Fin n → Fin N) (wipe : Fin n → Fin W)
    (S : (Ω × (Fin (W * N) → Pous.Bits b)) × (Fin n → Fin N) → Actual.Run L)
    (ℓ : L → L → ℕ) (X Y V : Finset L) (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (D ℓmax a m K : ℕ)
    (arrive : (Ω × (Fin (W * N) → Pous.Bits b)) × (Fin n → Fin N) → Fin n → ℕ) (ζ : Ω → Z)
    (quiet : Fin n → ℕ → Fin N → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) (T : Finset ℕ),
    CausalStrategy S ℓ → (∀ r, (S r).A.LatencyAtMost X Y ℓmax) → Assumptions.LiveReads S X arrive →
    Assumptions.LiveVerdicts (fun r j => Ch j ((S r).view V)) →
    Assumptions.QuietReads S X arrive D (fun r j => Ch j ((S r).view V)) (fun ω₀ => ζ ω₀.1) quiet →
    Assumptions.LateAnswers S X Y (Actual.tableLabel wipe) arrive ℓmax D (fun r j => Ch j ((S r).view V)) →
    Pous.pr (fun x : (Ω × Pous.Bits κ) × Pous.Bits κ' =>
        Actual.heldEvent S X V Ch arrive T m K
          ((x.1.1, prf x.1.2), fun j => Actual.drawIndex (rest x.2 j) (coins x.2 j))) ≤
      ENNReal.ofReal (Real.exp (-((K : ℝ) * (1 - ((a : ℝ) - 1) / N))) +
          n * T.card * 2 ^ m * (N.choose a : ℝ) / 2 ^ (a * b)) +
        Pous.StreamedWipe.prfAdvantage prf (Actual.heldDistinguisher S X V Ch arrive T m K) +
        Pous.StreamedWipe.prfAdvantage coins (Actual.drawDistinguisher prf S X V Ch arrive T m K) +
        ENNReal.ofReal (n * (2 ^ w % N : ℕ) / 2 ^ w)

end Accounting.Guarantees.Actual
