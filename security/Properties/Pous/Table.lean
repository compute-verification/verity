module

public import Properties.Pous.EndToEndAt
public import Properties.Pous.MemoryChallenger.Session

/-!
# What a session's check reads of the store: the prover's table

Beside setup's public outputs, the check of a PoUS session (`Run.checkAt`, the port of `run.check_at`) reads two things
of the node's store: the memory challenger's report (`Run.ChallengePhaseRecord`) and a table, one row a block of `C*`.
The report is the device's, and `MemoryChallenger.Session.Properties.AuditorSeesDigests` says what it shows. The table
is the prover's: the prover's side computes it from `C*` at setup and sends it in each session's case
(`RunProtocol.Evidence.digests`), and the memory challenger takes none. This module states what the table shows: every
block's `blockDigest`, in block order, and nothing else of the store.

**The check stands in for the prover's code.** No Lean of the prover exists, and none is modeled here. A table is named
by how the check reads one (`IsTableAt`): its size, and each row as `Run.row` cuts it, which is all `Run.checkAt` reads
of a table. Three statements:

* `AcceptsTableAt`: the session's check accepts a table only if it is `C*`'s, for every `C*` of the point's shape whose
  leaves have the root setup published, or the table's tree and `C*`'s hold a SHA-512 collision. So whatever the prover
  runs, a table the check accepts holds each block's digest, in order.
* `TableShowsDigests`: two stores whose blocks have the same digests have the same table. So the table depends on the
  store through the digests alone.
* `SessionShowsDigests`: the table and the report in one line. The pair of the two is a function of the blocks'
  digests, the device's coins and the skeleton of the device's inputs. It is `TableShowsDigests` beside
  `AuditorSeesDigests` and says nothing the two do not.

No soundness statement takes any of them as a premise. What they do not cover is in `TableShowsDigests`'s docstring;
what a session's case holds besides the table and the report is in `SessionShowsDigests`'s.
-/

@[expose] public section

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties
open Accounting.Assumptions
open Pous.EndToEnd
open MemoryChallenger.Session.Properties

namespace Pous.Table

/-! ## The table, as the check reads one -/

/-- The digests of `C*`'s blocks: block `i` in block order (`flat`) to its `blockDigest`. -/
def blockDigests (C : Array (Array ByteArray)) (i : Nat) : ByteArray := blockDigest ((flat C)[i]!)

/-- `t` is `C*`'s table at the parameters `q`, as the session's check reads a table: `q.blocks` rows of
`Merkle.DIGEST` bytes, which is the size `Run.checkAt` requires, and row `i` as `Run.row` cuts it, which is how
`Run.rowLeaves` and `Run.expected` read it, is the `blockDigest` of block `i` of `C*`. -/
def IsTableAt (q : P2.Params) (C : Array (Array ByteArray)) (t : ByteArray) : Prop :=
  t.size = q.blocks * Merkle.DIGEST ∧ ∀ i, i < q.blocks → Run.row t i = blockDigests C i

/-- `C*` and `C'*` have the same digests at `q`: block for block in block order, the same `blockDigest`. -/
def SameDigestsAt (q : P2.Params) (C C' : Array (Array ByteArray)) : Prop :=
  ∀ i, i < q.blocks → blockDigests C i = blockDigests C' i

/-! ## What the table shows -/

/-- **The session's check accepts only the store's table.** Deterministic and assumption free: if the composed check at
the point `pt` (`Run.checkAt`, under any signature schemes) accepts the table `digests` against setup's public outputs
`pub`, then for every `C*` of `pt`'s shape whose leaves have the published root, `digests` is `C*`'s table
(`IsTableAt`), or the table's tree and `C*`'s hold a SHA-512 collision (`TableBreakAt`).

The check reads no block. It hashes each row to a leaf under setup's salt and compares the leaves' root with the root
setup published (`Run.rowLeaves`); a leaf of `C*` is the same hash over the block's digest (`Merkle.packageLeaf`, whose
`Merkle.rowDigest` is `blockDigest` under the check's name). With setup's `AcceptsDecodes` and `PublishesRoot` the `C*`
setup accepted is such a `C*`, as `RunCertifiesAt` follows from `RunCheckAcceptsAt`.

It says nothing of a table the check refuses: a prover may put anything there, and the check has then read it. -/
def AcceptsTableAt : Prop :=
  ∀ (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (pub : SetupProtocol.Public)
    (digests : ByteArray) (r : Run.ChallengePhaseRecord) (sid : ByteArray),
    Run.checkAt schemes pt trusted pub digests r sid = .ok () →
    ∀ (C : Array (Array ByteArray)), ShapedAt pt.params C → Merkle.root (leaves pub.salt C) = some pub.root →
      IsTableAt pt.params C digests ∨ TableBreakAt pt.params pub digests C

/-- **Of the store, the table shows each block's digest and nothing else.** Two stores whose blocks have the same
digests have the same table: a table is its rows, so `C*`'s table depends on `C*` through its blocks' digests alone,
and whoever reads the table learns of the store at most each block's `blockDigest`, by its place in block order.

It reads `IsTableAt`, so `Run.row` and the table's size as `Run.checkAt` requires it, and `blockDigest`.

Not covered:
* what a digest tells. `blockDigest` is SHA-512 over a public prefix and the block, with no key and no salt, so whoever
  holds a candidate for a block can test it against its row, and equal blocks have equal rows. The statement bounds
  what the table shows by the digests; it does not say a digest hides its block;
* setup's case. In the public-encoder setup its evidence is `C*` itself, and setup's check reads every block of it;
* what a session's case carries along of setup's: setup's evidence again, so `C*` itself, and setup's public outputs,
  among them the commitment to `W`. Of those outputs the session's check reads the block count, the salt and the root
  (`Run.checkAt`), and it accepts only the root of the table's leaves under the salt, so the root shows nothing the
  table and the salt do not;
* the report, which `MemoryChallenger.Session.Properties.AuditorSeesDigests` covers;
* that the prover's table is `IsTableAt`: `AcceptsTableAt` gives it for a table the check accepts, outside a collision,
  and nothing here gives it for the prover's Python. -/
def TableShowsDigests : Prop :=
  ∀ (q : P2.Params) (C C' : Array (Array ByteArray)) (t t' : ByteArray),
    IsTableAt q C t → IsTableAt q C' t' → SameDigestsAt q C C' → t = t'

/-! ## The table and the report together -/

/-- **Of the store, a session's table and report show the auditor the blocks' digests and the session's skeleton.** At
the parameters `q` there is a function `table` of the blocks' digests such that, for every `C*`, every table `t` of it
and every run of the device under `AuditorSeesDigests`'s three premises, the pair of `t` and what the auditor sees of
the device is the pair of `table` of `C*`'s digests and what `seenCode` sends in the clear on the device's coins and
the skeleton of its true inputs. So the pair is a function of the blocks' digests, the device's coins and the session's
skeleton, and of nothing else of the store. The table has no premise but `IsTableAt`: the prover is no auditor party,
and nothing is assumed of it.

It is `TableShowsDigests` and `AuditorSeesDigests` in one equation and follows from the two. `table` is some function,
not a program: the statement says what the pair depends on, not what it costs to compute. In a session the check
accepts, each round's digest in the report is the table's row at the round's index (`verdict`, on `Run.expected`), so
of a block's bytes the report then shows nothing the table does not.

Not covered, beside what the two halves leave out:
* the session's case as a whole. Its evidence holds setup's evidence beside the table and the report
  (`RunProtocol.Evidence.setup`), and in the public-encoder setup that is `C*` itself; its commitment is setup's public
  outputs. No statement of this form holds of the whole case while it carries `C*`;
* that the report in the auditor's view is the case's (`RunProtocol.Evidence.session`): the prover's side passes the
  device's report on in the case, and `reportPublic` counts it as in the clear. -/
def SessionShowsDigests : Prop :=
  ∀ (q : P2.Params), ∃ table : (Nat → ByteArray) → ByteArray,
    ∀ {Ω Wi A : Type} (p : Params Ω) (wire : Ch × Msg → Wi)
      (r : Parties.Run Ω (Ω × Recorded) (Step ByteArray Observed) Ch Msg Wi A)
      (C : Array (Array ByteArray)) (t : ByteArray),
      Parties.Faithful (code p) r → Parties.Confined reportPublic wire r → Parties.WitnessPathPrivate r →
      IsTableAt q C t →
        (t, Parties.auditorView wire r) =
          (table (blockDigests C),
            (inClear reportPublic ((seenCode p).messages r.coins (r.truth.map Step.skeleton))).map wire)

end Pous.Table
