import Definitions.Pouw.Dither
import Definitions.Pouw.MerkleExtract

/-!
# Commit-and-open from SHA-512 as a random oracle: the realization premise (trusted)

`Definitions.Pouw.Assumptions.Dither.Extract` (commit-and-open costs what computing costs) is a theorem from SHA-512 as a
random oracle for the commitments (R-ACCEL-3, pending Daniel) and its realization, not a premise of its own
(`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`). This file holds what that theorem reads:

* `QueryLog` (ABSORB-THM's B2): a period's commitment-path queries, per value of the verifier's coins, in the
  lazy-sampling form of `Definitions.Pouw.MerkleExtract.Prover`, with round `w`'s root after the first `τ w` answers.
* `ExtractAdv`: `Extract` with one binder changed, proposed to WORKSOUND-2 (the type's writer): round `w`'s words program
  reads advice fixed before the round's coins, which carries the oracle and the query log before the round, instead of
  only the prover's state. `Extract` implies it (`Proofs.Pouw.ShaRealize.extractAdv_of_extract`).
* `ShaRealize`: the realization premise on the no-fold round. Every prover's commitment-path hashing is `q` black-box
  SHA-512 calls (the log), its round words program is the walk of the extractor over that log, and the cost of that
  walk with the public hashing is within the prover's own work (each call costs at least the public implementation's,
  R-ACCEL-7).

The modelling choices, each a hypothesis or a clause RT-LEAN can check:
* **(N1) distinct queries**: `QueryLog.fresh`. A repeat needs no query, so `q` counts distinct SHA-512 inputs.
* **(N2) `q` includes the verifier's evaluations**: `ShaRealize`'s clause that every point an accepted opening's check
  hashes is a logged query. The verifier's evaluations are appended to the log as queries after the prover's own
  (an evaluation at a point nobody queried is a fresh uniform answer either way).
* **(N3) one query budget `q` and one schedule `τ` for every prover**: `q` bounds every prover's distinct
  commitment-path SHA-512 inputs over the period, with the verifier's evaluations; `τ w` bounds the queries before
  round `w`'s root (padding with dummy queries is free in the analysis). A prover past `q` costs more than
  `q · shaPub`; restricting `Extract` to provers whose period fits the devices is open (WORKSOUND's `FreedSound`
  event reads `Fits`).
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

/-- `pouw-commit-extract` in advice form (B2; proposed to WORKSOUND-2 as `Extract`'s one binder change). As `Extract`:
in the model at its device premises, every prover's round `w` splits into a words program `Mw w` and a commitment
program `Mh w` on those words, whose uses and imported bits sum to at most `work w`'s, and the rounds' checks accept
while some round's check fails on `Mw w`'s words with probability at most `κ_x`. The one change: `Mw w` reads advice
`adv o x w` that reads no coin of round `w` or later, instead of only the prover's state after the round's instance.
The extractor's words read the oracle and the queries before the round, which that state need not hold. `SAKMatmul`
and `ShaSpeed` quantify over every state fixed before the round's coins, so they read the advice as they read the
state. -/
def ExtractAdv (G : Accel) (P : RoundProto) (k : ℕ) (κx : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k,
    ∃ (Adv : Type) (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv)
      (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root),
      (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
      (∀ w o T, G.Mc.eval (Mh w) (o, T) = P.commit o T) ∧
      (∀ o x w r, G.use (Mw w) (adv o x w, (x w).1) r +
          G.use (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) r ≤ G.use (A.work w) (A.workIn o x w) r) ∧
      (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) + G.ext (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
        G.ext (A.work w) (A.workIn o x w)) ∧
      prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Accepted ox.1 ox.2 ∧
        ∃ w, ¬ P.ok (A.instAt ox.1 ox.2 w) (ox.2 w).1 (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1))) ≤ κx

/-- `pouw-commit-realize` (proposed; the realization half of SHA-512 as a random oracle for the commitments, R-ACCEL-3
with R-ACCEL-7, pending Daniel; the Accel instance of ASK-16's card 9cd57182, beside `StraightLineExtraction`). On
the no-fold round `nf`, whose oracle is the table `eo : nf.Orc ≃ (X → D)` (SHA-512 on the commitment path, `E` its
leaf and node encodings, `rD` and `pl` the root and path as digests, `dirs` a cell's position in the tree): in the
model at its device premises, every prover of `k` rounds has
* a query log `L` of `q` distinct queries with schedule `τ` (B2, N1, N3);
* the black-box reading: on every oracle and the answer stream it gives the log, the rounds' roots are the log's, and
  every point an accepted opening's check hashes is a logged query (N2);
* round words programs `Mw w` on advice fixed before the round's coins, each the walk of the extractor down from the
  round's root over the queries before it (`MerkleExtract.ext`; a cell it can't reach reads `0`), and commitment
  programs `Mh w` computing the commitment;
* the cost half: `Mw w`'s and `Mh w`'s uses and imported bits sum to at most the prover's `work w`'s. `Mw w` is the
  prover's work between the round's dither and its root, with each commitment-path SHA-512 call done by the public
  implementation and logged, then the walk; `Mh w` hashes the words with the public implementation. Each of the
  prover's calls costs at least the public implementation's (R-ACCEL-7), and the walk is charged within the prover's
  hashing (open: a separate `ξ`, ABSORB-THM's B4).

It says nothing about the probability: `κ_x` is proved (`ExtractOfRealize`). -/
def ShaRealize (G : Accel) (nf : NoFold) (k : ℕ) {X D : Type} [Inhabited D] (E : Enc X D Word)
    (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (q : ℕ) (τ : Fin k → ℕ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G (noFoldRound nf) k,
    ∃ (L : QueryLog (noFoldRound nf) k X D q τ) (Adv : Type)
      (adv : nf.Orc → (Fin k → nf.Draw × nf.Chk) → Fin k → Adv)
      (Mw : Fin k → G.Mc.Alg (Adv × nf.Draw) (Fin nf.n → Fin nf.n' → Fin nf.nb → Word))
      (Mh : Fin k → G.Mc.Alg (nf.Orc × (Fin nf.n → Fin nf.n' → Fin nf.nb → Word)) nf.Root),
      (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
      (∀ w o T, G.Mc.eval (Mh w) (o, T) = nf.merkle o T) ∧
      (∀ o x w r, G.use (Mw w) (adv o x w, (x w).1) r +
          G.use (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) r ≤ G.use (A.work w) (A.workIn o x w) r) ∧
      (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) + G.ext (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
        G.ext (A.work w) (A.workIn o x w)) ∧
      (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u) ∧
      (∀ o x u, Consistent (L.qry x) (eo o) u → A.Accepted o x → ∀ w (j : Fin nf.s),
        ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
          ∃ i, L.qry x i u = y) ∧
      (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb),
        G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
          (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)

end Pouw.Assumptions
