import Definitions.Pouw.Dither
import Definitions.Pouw.MerkleExtract

/-!
# Commit-and-open from SHA-512 as a random oracle: the extraction premise and its realization (trusted)

`ExtractAdv` (commit-and-open costs what computing costs) is a theorem from SHA-512 as a random oracle for the
commitments (R-ACCEL-3, pending Daniel) and its realization, not a premise of its own
(`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`). This file holds `ExtractAdv` and what that theorem reads:

* `QueryLog` (ABSORB-THM's B2): a period's commitment-path queries, per value of the verifier's coins, in the
  lazy-sampling form of `Definitions.Pouw.MerkleExtract.Prover`, with round `w`'s root after the first `τ w` answers.
* `ExtractAdvAt` and `ExtractAdv`: each round's work splits into a words program and a commitment program on advice
  fixed before the round's coins (RT-LEAN-2 L37's four conditions on the commitment side).
  - The advice carries the oracle and the query log before the round, instead of only the prover's state, and holds at
    most `Badv A` bits.
  - The commitment program reads the advice, the dither and the words, and its output is the prover's root as an
    equation (condition 1). No premise asks it to compute the commitment of every transcript (W1).
  - The event reads the check on the words and the verifier's opening check of the checked cells under the prover's
    root (`RoundProto.opens`, condition 2).
  - The cost clause is the period's, with a term `ξ A` for the walk and the log's upkeep (B4). `ξ A`, `κx A` and
    `Badv A` are the prover's own.
* `Realization` and `ShaRealize`: the realization premise on the no-fold round, prover by prover. A prover's
  commitment-path hashing is `q A` black-box SHA-512 calls (the log), its round words program is the walk of the
  extractor over that log, its commitment program is its own commitment path, and the two cost at most the prover's
  own work plus `k · ξw + q A · ξl`.

The modelling choices, each a hypothesis or a clause RT-LEAN can check:
* **(N1) distinct queries**: `QueryLog.fresh`. A repeat needs no query, so `q` counts distinct SHA-512 inputs.
* **(N2) `q` includes the verifier's evaluations**: `Realization`'s clause that every point an accepted opening's check
  hashes is a logged query. The verifier's evaluations are appended to the log as queries after the prover's own
  (an evaluation at a point nobody queried is a fresh uniform answer either way).
* **(N3) each prover's own query budget `q A` and schedule `τ A`**: `q A` bounds `A`'s distinct commitment-path
  SHA-512 inputs over the period, with the verifier's evaluations; `τ A w` bounds the queries before round `w`'s root
  (padding with dummy queries is free in the analysis). No prover is excluded: a prover with more queries has a larger
  `q A`, and so a larger `ξ A` and `κx A`.

Open: the cost half's split of one program's use into a words part and a commitment part rests on the oracle calls
being separately priced steps of the machine (RT-LEAN-2 L28, an oracle-call step priced at the public implementation's
cost a call). Until that step is in `Accel`, the split is `Realization`'s cost clause, a premise.
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

/-- **`pouw-commit-extract` in advice form, at one prover** (B2): on the registered instances `reg`, `A`'s round `w`
splits into a words program `Mw w` and a commitment program `Mh w`, and the rounds' checks accept while some round's
check fails on `Mw w`'s words, or its checked cells don't open under `A`'s root at those words, with probability at
most `κx`.
* `Mw w` and `Mh w` read advice `adv o x w` of at most `Badv` bits that reads no coin of round `w` or later. The
  extractor's words read the queries before the round, which the prover's state need not hold.
* **`Mh w`'s output is `A`'s root** on the advice, the dither and `Mw w`'s words (RT-LEAN-2 L37, condition 1).
* **The cost clause is the period's** (ξ, ABSORB-THM's B4). Over the period, the words and commitment programs use at
  most what the prover's `work` uses (`Prover.workUse`), plus `ξ r` of resource `r`: the extractor's walk and the
  upkeep of the query log, which aren't the prover's. A query is logged once in the period, so a per-round clause
  would count its upkeep `k` times.
* The exchanged bits stay per round, with no `ξ`: the walk exchanges nothing.
* **The event** reads `RoundProto.ok` on the words and `RoundProto.opens` of the checked cells under `A`'s root
  (condition 2: the verifier's opening check, never prover-supplied). -/
def ExtractAdvAt (G : Accel) (P : RoundProto) (k : ℕ) (reg : Fin k → P.Inst) (Badv : ℕ) (ξ : G.Res → ℝ) (κx : ℝ)
    (A : Prover G P k) : Prop :=
  ∃ (Adv : Type) (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root),
    (∃ e : Adv → Fin (2 ^ Badv), Function.Injective e) ∧
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
    (∀ o x w, G.Mc.eval (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) = A.root o x w) ∧
    (∀ o x r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      A.workUse o x r + ξ r) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) +
        G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Accepted reg ox.1 ox.2 ∧
      ∃ w, ¬ (P.ok (reg w) (ox.2 w).1 (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1)) ∧
        P.opens ox.1 (A.root ox.1 ox.2 w) (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1)))) ≤ κx

/-- `pouw-commit-extract` (proposed; the Accel-model instance of ASK-16's random-oracle card, with
`StraightLineExtraction`; `ExtractOfRealize` proves it from SHA-512 as a random oracle and `ShaRealize`).
**Committing and opening costs what computing costs**: in the model at its device premises, every prover `A` meets
`ExtractAdvAt` at its own advice size `Badv A`, walk-and-log cost `ξ A` and extraction error `κx A`. All three are
functions of the prover, so a prover that makes more queries gets its own, larger bound rather than being excluded.
Collision resistance gives binding, not that the committed words and their Merkle root were computed; this premise
gives that. In `WorkSound`, `ε` gains `μ · ξ_A / C` and `κ` gains `κx A`, prover by prover. -/
def ExtractAdv (G : Accel) (P : RoundProto) (k : ℕ) (reg : Fin k → P.Inst) (Badv : Prover G P k → ℕ)
    (ξ : Prover G P k → G.Res → ℝ) (κx : Prover G P k → ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k, ExtractAdvAt G P k reg (Badv A) (ξ A) (κx A) A

/-- **A realization of one prover's commitment path** at query budget `q`, schedule `τ` and advice size `Badv` (the
per-prover form of `pouw-commit-realize`). On the no-fold round `nf` at the registered instances `reg`, whose oracle is
the table `eo : nf.Orc ≃ (X → D)` (SHA-512 on the commitment path, `E` its leaf and node encodings, `rD` and `pl` the
root and path as digests, `dirs` a cell's position in the tree), the prover `A` has
* a query log `L` of `q` distinct queries with schedule `τ` (B2, N1). `q` and `τ` are `A`'s own (N3);
* the black-box reading: on every oracle and the answer stream it gives the log, the rounds' roots are the log's, and
  every point an accepted opening's check hashes is a logged query (N2);
* round words programs `Mw w` on advice of at most `Badv` bits fixed before the round's coins, each the walk of the
  extractor down from the round's root over the queries before it (`MerkleExtract.ext`; a cell it can't reach reads
  `0`), and commitment programs `Mh w` on the advice, the dither and the words whose output is `A`'s root;
* the cost half, over the period. The `Mw w` and `Mh w` use at most the prover's `work` plus `k · ξw + q · ξl` of each
  resource: `ξw` the walk's use per round, `ξl` the upkeep of one log entry.
  - `Mh w` is the prover's commitment path between the round's dither and its root: its SHA-512 calls, each done by
    the public implementation and logged (each of the prover's calls costs at least the public implementation's,
    R-ACCEL-7).
  - `Mw w` is the rest of the prover's work, then the walk.

  The exchanged bits sum to at most `work w`'s, round by round. -/
def Realization (G : Accel) (nf : NoFold) (k : ℕ) (reg : Fin k → (noFoldRound nf).Inst) {X D : Type} [Inhabited D]
    (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ) (A : Prover G (noFoldRound nf) k)
    (q : ℕ) (τ : Fin k → ℕ) (Badv : ℕ) : Prop :=
  ∃ (L : QueryLog (noFoldRound nf) k X D q τ) (Adv : Type)
    (adv : nf.Orc → (Fin k → nf.Draw × nf.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × nf.Draw) (Fin nf.n → Fin nf.n' → Fin nf.nb → Word))
    (Mh : Fin k → G.Mc.Alg (nf.Orc × (Adv × nf.Draw) × (Fin nf.n → Fin nf.n' → Fin nf.nb → Word)) nf.Root),
    (∃ e : Adv → Fin (2 ^ Badv), Function.Injective e) ∧
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
    (∀ o x w, G.Mc.eval (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) = A.root o x w) ∧
    (∀ o x r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      A.workUse o x r + (k * ξw r + q * ξl r)) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) +
        G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → A.Accepted reg o x → ∀ w (j : Fin nf.s),
      ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
        ∃ i, L.qry x i u = y) ∧
    (∀ o x u, Consistent (L.qry x) (eo o) u → ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb),
      G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
        (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)

/-- `pouw-commit-realize` (proposed; the realization half of SHA-512 as a random oracle for the commitments, R-ACCEL-3
with R-ACCEL-7, pending Daniel; the Accel instance of ASK-16's card 9cd57182, beside `StraightLineExtraction`). In the
model at its device premises, every prover `A` of `k` rounds on the no-fold round `nf` at the registered instances
`reg` has a `Realization` at its own query budget `q A`, schedule `τ A` and advice size `Badv A`, with the walk's use
`ξw` a round and the log's `ξl` an entry. It says nothing about the probability: `κ_x` is proved (`ExtractOfRealize`),
prover by prover. -/
def ShaRealize (G : Accel) (nf : NoFold) (k : ℕ) (reg : Fin k → (noFoldRound nf).Inst) {X D : Type} [Inhabited D]
    (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ)
    (q : Prover G (noFoldRound nf) k → ℕ) (τ : Prover G (noFoldRound nf) k → Fin k → ℕ)
    (Badv : Prover G (noFoldRound nf) k → ℕ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A, Realization G nf k reg E eo rD pl dirs ξw ξl A (q A) (τ A) (Badv A)

end Pouw.Assumptions
