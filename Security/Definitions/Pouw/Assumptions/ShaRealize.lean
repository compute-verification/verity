import Definitions.Pouw.Dither
import Definitions.Pouw.MerkleExtract

/-!
# Commit-and-open from SHA-512 as a random oracle: the realization premise (trusted)

`Definitions.Pouw.Assumptions.Dither.Extract` (commit-and-open costs what computing costs) is a theorem from SHA-512 as a
random oracle for the commitments (R-ACCEL-3, pending Daniel) and its realization, not a premise of its own
(`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`). This file holds what that theorem reads:

* `QueryLog` (ABSORB-THM's B2): a period's commitment-path queries, per value of the verifier's coins, in the
  lazy-sampling form of `Definitions.Pouw.MerkleExtract.Prover`, with round `w`'s root after the first `τ w` answers.
* `ExtractAdvAt` and `ExtractAdv`: `Extract` in advice form, proposed to WORKSOUND-2 (the type's writer).
  - Round `w`'s words program reads advice fixed before the round's coins, which carries the oracle and the query log
    before the round, instead of only the prover's state.
  - The cost clause is the period's, with a term `ξ A` for the walk and the log's upkeep (B4).
  - `ξ A` and the error `κx A` are the prover's own. `Extract` is the case `ξ = 0`, `κx` constant
    (`Proofs.Pouw.ShaRealize.extractAdv_of_extract`).
* `Realization` and `ShaRealize`: the realization premise on the no-fold round, prover by prover. A prover's
  commitment-path hashing is `q A` black-box SHA-512 calls (the log), its round words program is the walk of the
  extractor over that log, and the cost of that walk with the public hashing is within the prover's own work plus
  `k · ξw + q A · ξl` (each call costs at least the public implementation's, R-ACCEL-7).

The modelling choices, each a hypothesis or a clause RT-LEAN can check:
* **(N1) distinct queries**: `QueryLog.fresh`. A repeat needs no query, so `q` counts distinct SHA-512 inputs.
* **(N2) `q` includes the verifier's evaluations**: `Realization`'s clause that every point an accepted opening's check
  hashes is a logged query. The verifier's evaluations are appended to the log as queries after the prover's own
  (an evaluation at a point nobody queried is a fresh uniform answer either way).
* **(N3) each prover's own query budget `q A` and schedule `τ A`**: `q A` bounds `A`'s distinct commitment-path
  SHA-512 inputs over the period, with the verifier's evaluations; `τ A w` bounds the queries before round `w`'s root
  (padding with dummy queries is free in the analysis). No prover is excluded: a prover with more queries has a larger
  `q A`, and so a larger `ξ A` and `κx A`.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither Pouw.MerkleExtract

/-- **A period's commitment-path query log** (B2), at each value `x` of the verifier's coins: `q` adaptive queries to
the oracle `X → D` (query `i` reads only the answers before it, `qry_dep`), distinct (N1, `fresh`), and round `w`'s
root as a function of the first `τ w` answers (`root_dep`). It is the realization's witness: a prover that hashes on
the commitment path only by black-box calls has one. -/
structure QueryLog (P : RoundProto) (k : ℕ) (X D : Type) (q : ℕ) (τ : Fin k → ℕ) where
  qry : (Fin k → P.Draw × P.Chk) → Fin q → (Fin q → D) → X
  qry_dep : ∀ x i u u', (∀ j : Fin q, j < i → u j = u' j) → qry x i u = qry x i u'
  fresh : ∀ x u (i j : Fin q), j < i → qry x i u ≠ qry x j u
  root : (Fin k → P.Draw × P.Chk) → Fin k → (Fin q → D) → D
  root_dep : ∀ x w u u', (∀ j : Fin q, (j : ℕ) < τ w → u j = u' j) → root x w u = root x w u'

/-- Round `w`'s commitment over the log at the coins `x`, as the lazy-sampling prover `extract_of_ro_roots` reads. -/
def QueryLog.prover {P : RoundProto} {k : ℕ} {X D : Type} {q : ℕ} {τ : Fin k → ℕ} (L : QueryLog P k X D q τ)
    (x : Fin k → P.Draw × P.Chk) (w : Fin k) : MerkleExtract.Prover X D q (τ w) :=
  ⟨L.qry x, L.qry_dep x, L.fresh x, L.root x w, L.root_dep x w⟩

/-- **`pouw-commit-extract` in advice form, at one prover** (B2): `A`'s round `w` splits into a words program `Mw w`
and a commitment program `Mh w` on those words, and the rounds' checks accept while some round's check fails on
`Mw w`'s words with probability at most `κx`.
* `Mw w` reads advice `adv o x w` that reads no coin of round `w` or later, instead of only the prover's state after
  the round's instance. The extractor's words read the queries before the round, which that state need not hold.
* **The cost clause is the period's** (ξ, ABSORB-THM's B4). Over the period, the words and commitment programs use at
  most what the prover's `work` uses, plus `ξ r` of resource `r`: the extractor's walk and the upkeep of the query
  log, which aren't the prover's. A query is logged once in the period, so a per-round clause would count its upkeep
  `k` times.
* The imported bits stay per round, with no `ξ`: the walk imports nothing. -/
def ExtractAdvAt (G : Accel) (P : RoundProto) (k : ℕ) (ξ : G.Res → ℝ) (κx : ℝ) (A : Prover G P k) : Prop :=
  ∃ (Adv : Type) (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root),
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
    (∀ w o T, G.Mc.eval (Mh w) (o, T) = P.commit o T) ∧
    (∀ o x r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      ∑ w, G.use (A.work w) (A.workIn o x w) r + ξ r) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) + G.ext (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Accepted ox.1 ox.2 ∧
      ∃ w, ¬ P.ok (A.instAt ox.1 ox.2 w) (ox.2 w).1 (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1))) ≤ κx

/-- **`pouw-commit-extract` in advice form** (proposed to WORKSOUND-2 for `WorkSound`, replacing `Extract`): in the
model at its device premises, every prover `A` meets `ExtractAdvAt` at its own walk-and-log cost `ξ A` and its own
extraction error `κx A`. Both are functions of the prover, so a prover that makes more queries gets its own, larger
bound rather than being excluded. `Extract G P k κx` is the case `ξ = 0`, `κx` constant
(`Proofs.Pouw.ShaRealize.extractAdv_of_extract`). In `WorkSound`, `ε` gains `(∑_r μ_r ξ_A r)/(k H)` and `δ` gains
`κx A`, prover by prover. -/
def ExtractAdv (G : Accel) (P : RoundProto) (k : ℕ) (ξ : Prover G P k → G.Res → ℝ) (κx : Prover G P k → ℝ) :
    Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k, ExtractAdvAt G P k (ξ A) (κx A) A

/-- **A realization of one prover's commitment path** at query budget `q` and schedule `τ` (the per-prover form of
`pouw-commit-realize`). On the no-fold round `nf`, whose oracle is the table `eo : nf.Orc ≃ (X → D)` (SHA-512 on the
commitment path, `E` its leaf and node encodings, `rD` and `pl` the root and path as digests, `dirs` a cell's position
in the tree), the prover `A` has
* a query log `L` of `q` distinct queries with schedule `τ` (B2, N1). `q` and `τ` are `A`'s own (N3);
* the black-box reading: on every oracle and the answer stream it gives the log, the rounds' roots are the log's, and
  every point an accepted opening's check hashes is a logged query (N2);
* round words programs `Mw w` on advice fixed before the round's coins, each the walk of the extractor down from the
  round's root over the queries before it (`MerkleExtract.ext`; a cell it can't reach reads `0`), and commitment
  programs `Mh w` computing the commitment;
* the cost half, over the period. The `Mw w` and `Mh w` use at most the prover's `work` plus `k · ξw + q · ξl` of each
  resource: `ξw` the walk's use per round, `ξl` the upkeep of one log entry.
  - `Mw w` is the prover's work between the round's dither and its root, with each commitment-path SHA-512 call done
    by the public implementation and logged, then the walk.
  - `Mh w` hashes the words with the public implementation.
  - Each of the prover's calls costs at least the public implementation's (R-ACCEL-7).

  The imported bits sum to at most `work w`'s, round by round. -/
def Realization (G : Accel) (nf : NoFold) (k : ℕ) {X D : Type} [Inhabited D] (E : Enc X D Word)
    (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ) (A : Prover G (noFoldRound nf) k)
    (q : ℕ) (τ : Fin k → ℕ) : Prop :=
  ∃ (L : QueryLog (noFoldRound nf) k X D q τ) (Adv : Type)
    (adv : nf.Orc → (Fin k → nf.Draw × nf.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × nf.Draw) (Fin nf.n → Fin nf.n' → Fin nf.nb → Word))
    (Mh : Fin k → G.Mc.Alg (nf.Orc × (Fin nf.n → Fin nf.n' → Fin nf.nb → Word)) nf.Root),
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
    (∀ w o T, G.Mc.eval (Mh w) (o, T) = nf.merkle o T) ∧
    (∀ o x r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      ∑ w, G.use (A.work w) (A.workIn o x w) r + (k * ξw r + q * ξl r)) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) + G.ext (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → A.Accepted o x → ∀ w (j : Fin nf.s),
      ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
        ∃ i, L.qry x i u = y) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb),
      G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
        (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)

/-- `pouw-commit-realize` (proposed; the realization half of SHA-512 as a random oracle for the commitments, R-ACCEL-3
with R-ACCEL-7, pending Daniel; the Accel instance of ASK-16's card 9cd57182, beside `StraightLineExtraction`). In the
model at its device premises, every prover `A` of `k` rounds on the no-fold round `nf` has a `Realization` at its own
query budget `q A` and schedule `τ A`, with the walk's use `ξw` a round and the log's `ξl` an entry. It says nothing
about the probability: `κ_x` is proved (`ExtractOfRealize`), prover by prover. -/
def ShaRealize (G : Accel) (nf : NoFold) (k : ℕ) {X D : Type} [Inhabited D] (E : Enc X D Word)
    (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ)
    (q : Prover G (noFoldRound nf) k → ℕ) (τ : Prover G (noFoldRound nf) k → Fin k → ℕ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A, Realization G nf k E eo rD pl dirs ξw ξl A (q A) (τ A)

end Pouw.Assumptions
