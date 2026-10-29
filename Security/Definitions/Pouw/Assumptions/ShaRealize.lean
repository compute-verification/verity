import Definitions.Pouw.Dither
import Definitions.Pouw.MerkleExtract

/-!
# Commit-and-open from SHA-512 as a random oracle: the extraction premise and its realization (trusted)

`ExtractAdv` (commit-and-open costs what computing costs) is a theorem from SHA-512 as a random oracle for the
commitments (a named use not yet decided) and its realization, not a premise of its own
(`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`). This file holds `ExtractAdv` and what that theorem reads:

* `QueryLog`: a period's commitment-path queries, per value of the verifier's coins, in the
  lazy-sampling form of `Definitions.Pouw.MerkleExtract.Prover`, with round `w`'s root after the first `τ w` answers.
* `ExtractAdvAt` and `ExtractAdv`: each round's work splits into a words program and a commitment program on advice
  fixed before the round's coins (`SAKSha`'s four conditions on the commitment side).
  - The advice carries the oracle and the query log before the round, instead of only the prover's state, and holds at
    most `Badv A` bits.
  - The commitment program reads the advice, the dither and the words, and its output is the prover's root as an
    equation (condition 1). No premise asks it to compute the commitment of every transcript.
  - The event reads the check on the words and the verifier's opening check of the checked cells under the prover's
    root (`RoundProto.opens`, condition 2).
  - The cost clause is the period's, on runs within the cap, with a term `ξ A` for the walk and the log's upkeep.
    `ξ A`, `κx A` and `Badv A` are the prover's own.
  - The event is on runs within the cap (`Prover.WithinCap`): the sampling blocks' commit phases and work make at most
    `Q` oracle calls in all, and each selected block's answer at most `Q`. A run whose timed use fits the period's
    devices makes at most `cap_r / callUse_r` calls in its commit phases and work for each resource `r` a call uses
    (G3), and an answer that runs within the period as many (`AnswersFit`), so `WorkSound` reads it at such a `Q`.
* `Realization` and `ShaRealize`: the realization premise on the no-fold protocol, prover by prover. On every run within
  the cap, a prover's commitment-path hashing is its own calls and the verifier's evaluations, logged as `q A`
  black-box SHA-512 queries (the log). Its words program for a sampling block is the walk of the extractor over that
  log, its commitment program is its own commitment path, and the two cost at most the prover's own timed use plus
  `k · ξw + q A · ξl`. Programs reach the oracle only through calls (`Accel.OracleCalls`, a device premise of
  `ShaRealize`), so the log can be the prover's calls. The challenger times each sampling block's commit phase and
  work, from the previous block's work to the block's commitment, so the cap bounds their calls; a selected answer
  runs within the period, so the cap bounds its calls too. One budget and schedule serve every prover's calls and
  the verifier's evaluations: `k + 1` segments of `⌈Q⌉ + V` queries, `V` the verifier's evaluations, one ending at
  each root and one after the last, with the commit phase's and the work's calls of block `w` in segment `w`. No
  root reads an answer, so every answer's calls, and the evaluations after them, are logged after the last root, in
  the last segment, which gets `k ⌈Q⌉` more:
  `q = (k + 1)(⌈Q⌉ + V) + k ⌈Q⌉` and `τ_w = (w + 1)(⌈Q⌉ + V)`.

The modelling choices, each a hypothesis or a clause a reader can check:
* **(N1) distinct queries**: `QueryLog.fresh`. A repeat needs no query, so `q` counts distinct SHA-512 inputs.
* **(N2) `q` includes the verifier's evaluations**: `Realization`'s clause that every point an accepted opening's check
  hashes is a logged query. Sampling block `b`'s evaluations enter the log as queries after its answer, which may
  arrive after later blocks' coins and roots: at one budget for every prover they sit in the last segment,
  after the answer's own calls, and no root reads them, since no later program reads an answer. A later query may
  read them (an evaluation at a point nobody queried is a fresh uniform answer either way).
* **(N3) each prover's own query budget `q A` and schedule `τ A`**: `q A` bounds `A`'s distinct SHA-512 inputs over
  the period, its commit phases', work's and selected answers' calls with the verifier's evaluations, on runs within
  the cap `Q`. Sampling block `w`'s root comes
  after the log's first `τ A w` queries and before the rest: the queries before the root, padded with dummy queries
  (free in the analysis). The walk of block `w` reads only those, so `τ A w` must not reach past the root: the queries
  after it depend on the block's check coins and later ones, which `Mw w`'s advice doesn't carry. A
  prover with more queries has a larger `q A`, and so a larger `ξ A` and `κx A`; at the cap, `k + 1` segments of
  `⌈Q⌉ + V` queries and `k ⌈Q⌉` more in the last are one budget and schedule for every prover
  (`Definitions.Pouw.Guarantees.Dither.WorkSoundNoFold`).

Open:
* **The cost split** is `Realization`'s cost clause, a premise, on runs within the cap.
  The oracle calls are priced steps (G3) and the only way to the oracle (`Accel.OracleCalls`), but nothing in `Accel`
  says which of a program's steps are its commitment path. Three facts would discharge it:
  - `use` composes: a program built by `Mc.seq`, `Mc.pair` and routing uses at most what its parts use (`Machine`'s
    `time` has these laws; `Accel.use` has none);
  - a call can be cut out: a program that reads the oracle has a twin without the calls whose answers it doesn't read
    again, whose use plus `callUse` for each call cut is at most the program's (with (G3), what a priced step means);
  - the prover's round splits at its commitment path: its words read no oracle answer of the round, so its work with
    the path's calls cut out is `Mw w` before the walk, and the path is `Mh w`.

  The first two are laws of the machine. The third is a property of the prover, which is the next item.
* **The walk's oracle reach.** `Mw w` reads only the advice and the dither (condition 1), and on every
  run within the cap it must be the walk over the prover's log, which reads the answers the prover got inside the
  round. A prover whose words or tree read such an answer has no `Mw w` at an advice size that doesn't hold those
  answers for every dither, so `ShaRealize` at one `Badv` for every prover is false on a machine that has such a
  prover. `WorkSound`, at each prover's own `Badv A` and `ξ A`, still holds for it, at whatever bound its own
  `ExtractAdvAt` gives. Letting `Mw w` read the oracle through priced calls closes this, and changes what SAK(matmul)
  and SHA-512's speed quantify over.
* **The padding's upkeep.** The cost clause charges `ξl` for each of the log's `q` positions, so at one schedule for
  every prover the dummy queries that pad its segments are charged too, and the log's term in `WorkSoundNoFold`'s `ε`
  is `k + 1` times the calls'. Charging only the logged calls and evaluations (a count of entries beside `q`), or
  letting each root come at a stopping time of the log instead of a fixed position (`QueryLog.root_dep`, with
  `extract_of_ro_roots` over it), would remove the factor. The last segment's `k ⌈Q⌉` positions for the answers are
  charged whether or not a block is selected; charging only logged entries would charge the selected answers' calls
  alone.

The answers' calls are counted: the cap reads them (`Prover.WithinCap`), an accepted period's
selected answers run within it (`AnswersFit`, from `Assumptions.AnswerAfterWork` and the deadlines on the card), and
the budget logs them in the last segment.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither Pouw.MerkleExtract

/-- **A period's commitment-path query log**, at each value `x` of the verifier's coins: `q` adaptive queries to
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

/-- **`pouw-commit-extract` in advice form, at one prover**: at the registrable instances `Reg`, `A`'s sampling
block `w` splits into a words program `Mw w` and a commitment program `Mh w`, and the blocks' checks accept on a run
within the cap `Q` (`Prover.WithinCap` at the selection `sel`) while some block's check fails on `Mw w`'s words at
the block's committed instance (`Prover.inst`), or its checked cells don't open under `A`'s root at those words, with
probability at most `κx`.
* `Mw w` and `Mh w` read advice `adv o x w` of at most `Badv` bits that reads no coin of round `w` or later. The
  extractor's words read the queries before the round, which the prover's state need not hold.
* **`Mh w`'s output is `A`'s root** on the advice, the dither and `Mw w`'s words (condition 1).
* **The cost clause is the period's** (ξ), on runs within the cap `Q`. Over the period, the words
  and commitment programs use at most what the prover's sampling blocks use in their timed work
  (`Prover.timedUse`: their commit phases and work), plus `ξ r` of resource `r`: the extractor's walk and
  the upkeep of the query log, which aren't the prover's. A query is logged once in the period, so a per-block clause
  would count its upkeep `k` times. Past the cap the log isn't the prover's calls, and `WorkSound` reads the clause
  only on runs that fit the devices.
* The exchanged bits stay per block, with no `ξ`: the walk exchanges nothing.
* **The event** reads `RoundProto.ok` on the words at the block's committed instance and `RoundProto.opens` of the
  checked cells under `A`'s root (condition 2: the verifier's opening check, never prover-supplied), on runs within
  the cap. -/
def ExtractAdvAt (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Fin k → Set P.Inst) (sel : P.Chk → Prop) (Q : ℝ)
    (Badv : ℕ) (ξ : G.Res → ℝ) (κx : ℝ) (A : Prover G P k) : Prop :=
  ∃ (Adv : Type) (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv)
    (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root),
    (∃ e : Adv → Fin (2 ^ Badv), Function.Injective e) ∧
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) ∧
    (∀ o x w, G.Mc.eval (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) = A.root o x w) ∧
    (∀ o x, A.WithinCap sel Q o x → ∀ r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      A.timedUse o x r + ξ r) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) +
        G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Accepted Reg ox.1 ox.2 ∧ A.WithinCap sel Q ox.1 ox.2 ∧
      ∃ w, ¬ (P.ok (A.inst ox.1 ox.2 w) (ox.2 w).1 (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1)) ∧
        P.opens ox.1 (A.root ox.1 ox.2 w) (ox.2 w).2 (G.Mc.eval (Mw w) (adv ox.1 ox.2 w, (ox.2 w).1)))) ≤ κx

/-- `pouw-commit-extract` (proposed; the Accel-model instance of the random-oracle card, with
`StraightLineExtraction`; `ExtractOfRealize` proves it from SHA-512 as a random oracle and `ShaRealize`).
**Committing and opening costs what computing costs**: in the model at its device premises, every prover `A` meets
`ExtractAdvAt` at the registrable instances `Reg`, the cap `Q` on its oracle calls (its commit phases' and work's,
and each selected answer's), its own advice size `Badv A`, walk-and-log cost `ξ A` and extraction error `κx A`. All three are functions of the prover, so a prover that makes more queries gets its own,
larger bound rather than being excluded; at the cap they can be one bound for every prover.
Collision resistance gives binding, not that the committed words and their Merkle root were computed; this premise
gives that. In `WorkSound`, `ε` gains `μ · ξ_A / C` and `κ` gains `κx A`, prover by prover. -/
def ExtractAdv (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Fin k → Set P.Inst) (sel : P.Chk → Prop) (Q : ℝ)
    (Badv : Prover G P k → ℕ) (ξ : Prover G P k → G.Res → ℝ) (κx : Prover G P k → ℝ) : Prop :=
  G.G1 → G.G2 → G.G3 → G.P1 → G.P2 → ∀ A : Prover G P k, ExtractAdvAt G P k Reg sel Q (Badv A) (ξ A) (κx A) A

/-- **A realization of one prover's commitment path** at query budget `q`, schedule `τ` and advice size `Badv` (the
per-prover form of `pouw-commit-realize`), on runs within the cap `Q` (`Prover.WithinCap` at the selection `sel`). On
the no-fold protocol `nf` at the registrable instances `Reg`, whose oracle is the table `eo : nf.Orc ≃ (X → D)` (SHA-512 on the commitment path,
`E` its leaf and node encodings, `rD` and `pl` the root and path as digests, `dirs` a cell's position in the tree), the
prover `A` has
* a query log `L` of `q` distinct queries with schedule `τ` (N1). `q` and `τ` are `A`'s own (N3);
* the black-box reading, on every run within the cap: on every oracle and the answer stream it gives the log, the
  sampling blocks' roots are the log's, and every point an accepted opening's check hashes is a logged query (N2);
* words programs `Mw w` on advice of at most `Badv` bits fixed before the sampling block's coins, each, on every answer
  stream with distinct answers, the walk of the extractor down from the block's root over the queries before it
  (`MerkleExtract.ext`; a cell it can't reach reads `0`). Where answers repeat, `ext`'s choice among the matching
  points can walk into another block's tree, and a repeat is already `κ_x`'s `q²` term. The
  commitment programs `Mh w` read the advice, the dither and the words, and their output is `A`'s root;
* the cost half, over the period, on every run within the cap. The `Mw w` and `Mh w` use at most the prover's timed
  use (`Prover.timedUse`, its sampling blocks' commit phases and work) plus `k · ξw + q · ξl` of each resource: `ξw` the walk's use
  per sampling block, `ξl` the upkeep of one log entry.
  - `Mh w` is the prover's commitment path between the block's dither and its root: its SHA-512 calls, each done by
    the public implementation and logged (each of the prover's calls costs at least the public implementation's).
  - `Mw w` is the rest of the prover's work, then the walk.

  The exchanged bits sum to at most `work w`'s, block by block. -/
def Realization (G : Accel) (nf : NoFold) (k : ℕ) (Reg : Fin k → Set (noFoldRound nf).Inst)
    (sel : (noFoldRound nf).Chk → Prop) (Q : ℝ) {X D : Type}
    [Inhabited D]
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
    (∀ o x, A.WithinCap sel Q o x → ∀ r, ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
        G.use (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) ≤
      A.timedUse o x r + (k * ξw r + q * ξl r)) ∧
    (∀ o x w, G.ext (Mw w) (adv o x w, (x w).1) +
        G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) ≤
      G.ext (A.work w) (A.workIn o x w)) ∧
    (∀ o x, A.WithinCap sel Q o x → ∀ u, Consistent (L.qry x) (eo o) u → ∀ w, rD (A.root o x w) = L.root x w u) ∧
    (∀ o x, A.WithinCap sel Q o x → ∀ u, Consistent (L.qry x) (eo o) u → A.Accepted Reg o x → ∀ w (j : Fin nf.s),
      ∀ y ∈ pts E (pl ((A.ans o x w : Fin nf.s → Word × nf.Path) j).2) ((A.ans o x w : Fin nf.s → Word × nf.Path) j).1,
        ∃ i, L.qry x i u = y) ∧
    (∀ o x, A.WithinCap sel Q o x → ∀ u, Consistent (L.qry x) (eo o) u → Function.Injective u →
      ∀ w (c : Fin nf.n × Fin nf.n' × Fin nf.nb), G.Mc.eval (Mw w) (adv o x w, (x w).1) c.1 c.2.1 c.2.2 =
        (ext ((L.prover x w).oracle u) E ((L.prover x w).pre u) (L.root x w u) (dirs c)).getD 0)

/-- `pouw-commit-realize` (proposed; the realization half of SHA-512 as a random oracle for the commitments, a named
use not yet decided; the Accel instance, beside `StraightLineExtraction`). In the
model at its device premises, with the oracle reached only through priced calls (G3, `Accel.OracleCalls`), every
prover `A` of `k` sampling blocks on the no-fold protocol `nf` at the registrable instances `Reg` has a `Realization` on
its runs within the cap `Q` at the selection `sel`, at its own query budget `q A`, schedule `τ A` and advice size `Badv A`, with the walk's use
`ξw` a block and the log's `ξl` an entry. It says nothing about the probability: `κ_x` is proved (`ExtractOfRealize`),
prover by prover. -/
def ShaRealize (G : Accel) (nf : NoFold) (k : ℕ) (Reg : Fin k → Set (noFoldRound nf).Inst)
    (sel : (noFoldRound nf).Chk → Prop) (Q : ℝ) {X D : Type}
    [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ)
    (q : Prover G (noFoldRound nf) k → ℕ) (τ : Prover G (noFoldRound nf) k → Fin k → ℕ)
    (Badv : Prover G (noFoldRound nf) k → ℕ) : Prop :=
  G.G1 → G.G2 → G.G3 → G.OracleCalls eo → G.P1 → G.P2 → ∀ A,
    Realization G nf k Reg sel Q E eo rD pl dirs ξw ξl A (q A) (τ A) (Badv A)

end Pouw.Assumptions
