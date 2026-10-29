import Definitions.Pouw.Assumptions.ShaRealize

/-!
# Commit-and-open as a theorem: `ExtractAdv` from SHA-512 as a random oracle and its realization (trusted)

* `ExtractOfRealize`: on the no-fold round whose openings are Merkle openings under the oracle, with the oracle
  reached only through calls, the realization premise `ShaRealize` at the cap `Q` and each prover's own query budget
  `q A`, schedule `τ A` and advice size `Badv A` gives commit-and-open's extraction (`ExtractAdv`) at that prover's
  walk-and-log cost `k · ξw + q A · ξl` and extraction error `κ_x(A) = (q_A² + ∑_w q_A (2 τ_{A,w} + 1)) / |D|`, with
  `|D| = 2^512` for SHA-512. Its probability half is
  `Proofs.Pouw.MerkleExtract.extract_of_ro_roots` over the lazy-sampling form, carried to the uniform oracle and
  averaged over the verifier's coins.
-/

namespace Pouw.Guarantees.ShaRealize

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **Commit-and-open costs what computing costs, from SHA-512 as a random oracle**. For every accelerator
model `G`, no-fold protocol `nf`, number of rounds `k` and registered instances `reg`, every oracle that is a uniform
table `eo : nf.Orc ≃ (X → D)` with tree encodings `E`, and every reading of roots `rD`, paths `pl` and cell positions
`dirs` under which the verifier's check `nf.opens` is the Merkle path check under the oracle along the cell's position
(a protocol-code fact of frame-v3-sha512), and models whose programs reach the oracle only through their calls
(`Accel.OracleCalls`), the realization premise `ShaRealize` at the cap `Q` on a run's calls (its rounds' work and
answers), the walk's use `ξw` a round, the log's `ξl` an entry, and each prover's own query budget `q A`, schedule
`τ A` and advice size `Badv A`, gives
`ExtractAdv` at `Q` with, for each prover `A`,
* advice of at most `Badv A` bits,
* `ξ_A = k · ξw + q A · ξl` of each resource over the period, and
* `κ_x(A) = (q_A² + ∑_w q_A (2 τ_{A,w} + 1)) / |D|`. With `τ_{A,w} ≤ q_A` it is at most `(k + 1) · 3 q_A² / 2^512`
  for SHA-512.

The event `ExtractAdv` bounds reads the verifier's opening check of the checked cells under the prover's root:
an accepted round's answer opens those cells, so where the check passes on the walk's words they open
too, and the event is the check's failure on the walk's words. No prover is excluded: one that makes more queries has
a larger `q A`, so a larger `ξ_A` and `κ_x(A)`. -/
def ExtractOfRealize : Prop :=
  ∀ (G : Accel) (nf : NoFold) (k : ℕ) (reg : Fin k → (noFoldRound nf).Inst) (Q : ℝ) (X D : Type) [Fintype X]
    [DecidableEq X] [Fintype D] [DecidableEq D] [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D))
    (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D)) (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool)
    (ξw ξl : G.Res → ℝ) (q : Prover G (noFoldRound nf) k → ℕ) (τ : Prover G (noFoldRound nf) k → Fin k → ℕ)
    (Badv : Prover G (noFoldRound nf) k → ℕ),
    (∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c) →
    G.OracleCalls eo → ShaRealize G nf k reg Q E eo rD pl dirs ξw ξl q τ Badv →
    ExtractAdv G (noFoldRound nf) k reg Q Badv (fun A r => k * ξw r + q A * ξl r)
      (fun A => ((q A * q A + ∑ w, q A * (2 * τ A w + 1) : ℕ) : ℝ) / Fintype.card D)

end Pouw.Guarantees.ShaRealize
