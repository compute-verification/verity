import Definitions.Pouw.Assumptions.ShaRealize

/-!
# Commit-and-open as a theorem: `Extract` from SHA-512 as a random oracle and its realization (trusted)

* `ExtractOfRealize`: on the no-fold round whose openings are Merkle openings under the oracle, the realization
  premise `ShaRealize` at a query budget `q` and schedule `τ` gives `Extract` in advice form (`ExtractAdv`) at
  `κ_x = (q² + ∑_w q (2 τ_w + 1)) / |D|`, with `|D| = 2^512` for SHA-512. Its probability half is
  `Proofs.Pouw.MerkleExtract.extract_of_ro_roots` over the lazy-sampling form, carried to the uniform oracle and
  averaged over the verifier's coins.
-/

namespace Pouw.Guarantees.ShaRealize

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **Commit-and-open costs what computing costs, from SHA-512 as a random oracle** (CMP-132). For every accelerator
model `G`, no-fold protocol `nf` and number of rounds `k`, every oracle that is a uniform table `eo : nf.Orc ≃ (X → D)`
with tree encodings `E`, and every reading of roots `rD`, paths `pl` and cell positions `dirs` under which the
verifier's check `nf.opens` is the Merkle path check under the oracle along the cell's position (a protocol-code fact
of frame-v3-sha512), the realization premise `ShaRealize` at query budget `q` and schedule `τ` gives `ExtractAdv` at
`κ_x = (q² + ∑_w q (2 τ_w + 1)) / |D|`. With `τ_w ≤ q` it is at most `(k + 1) · 3q² / 2^512` for SHA-512. -/
def ExtractOfRealize : Prop :=
  ∀ (G : Accel) (nf : NoFold) (k : ℕ) (X D : Type) [Fintype X] [DecidableEq X] [Fintype D] [DecidableEq D]
    [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D)
    (pl : nf.Path → List (Bool × D × D)) (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (q : ℕ)
    (τ : Fin k → ℕ),
    (∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c) →
    ShaRealize G nf k E eo rD pl dirs q τ →
    ExtractAdv G (noFoldRound nf) k (((q * q + ∑ w, q * (2 * τ w + 1) : ℕ) : ℝ) / Fintype.card D)

end Pouw.Guarantees.ShaRealize
