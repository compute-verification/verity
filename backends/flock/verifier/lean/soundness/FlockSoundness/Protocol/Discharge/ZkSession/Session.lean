import FlockSoundness.Protocol.Discharge.ZkSession.Algebraic
import FlockSoundness.Protocol.ExecDrawOS

/-!
# A batched `--zk` session and its audit (live/PROTOCOL.md §1–§7, PROTOCOL.md §5)

`sessionZK` is `SessionBatch.sessionB` with `--zk` tables, as Lean defines the protocol at the interactive-oracle layer:
`Commit` sends every table's level-0 oracle, one link-point draw serves all tables, and then every table's reps run
concurrently, each rep with its pads commitment, the masked algebraic phase and level 0, and the inner proof
(`tableAfterZK`). The session accepts when every table accepts (`acceptedZK`). A table (`TabZK`) carries its statement,
its schedule and padded level 0, its link layout, its pads' layout and its inner proof's, with the conditions
`table_sound_pad_zk` reads.

**Per table** (`sessionZK_sound`): the session accepts while table `j`'s committed values do not satisfy its statement
(`CommittedPad`) with probability at most table `j`'s `padLinkError + (padRepError + εInner)²` (`TabZK.bound`), whatever
the other tables do (`value_batch_getD_le`). Over every table at once the bound is the sum (`sessionZK_sound_any`).

**The audit** (`analysisZK`, in the shape of `Audit.FlockBatched`): the verifier draws units by a law `L`, plans the
`--zk` tables from the draw and the registration (`planZ`), and reads each drawn unit from one table (`tab`). A drawn
unit wrong in the committed transcript `Xc` is charged to its table: either the table's oracle is not `CommittedPad`
(`ks`), or it is, and every satisfying padded witness near it decodes, on the unit's wires, to values other than `Xc`'s
(`link`, `LinkEvZK`). At this layer the oracle is the prover's message, so extraction is the oracle itself and `ε_ks` is
discharged by `sessionZK_sound` with no assumption (`zk_knowledgeSound`): it is the largest table bound of the plan,
averaged over the draw (`ksAvgZK`). The lowering comes from each drawn unit's place in its table
(`loweringSoundZK_of_place`, `UnitPlace`), as at M0.

**The count curve** (`zk_flock_count`, and `zk_flock_count_execOS` at the live draw under A3): in the shape of
`Prog.flock_headline`,
`Pr[accept ∧ at least K wrong units] ≤ miss(K) + ε_ks^zk + δ_link`, with
* `miss(K)`: `Law.stratified`'s count curve at the live draw, under A3 (`uniform/io-getrandombytes`);
* `ε_ks^zk = ksAvgZK`: per table `padLinkError + (padRepError + εInner)²`, proved here;
* `δ_link`: the named `LinkSound` hypothesis, as `Audit.flock_batched_count` leaves it at M0.

**Open guarantees** (what this statement does not cover, each a separate piece of work):
* *The compiled `--zk` layer.* Here every oracle is a message. Under `--zk` the level-0 table, the pads (`Ch, Cμ`), the
  inner proof's `τ` and the coin tree are Merkle commitments with salted leaves, so the headline's `ε_ks` (knowledge
  extraction from caps, `ksAvgStrict` under `cr/sha-512`), `δ_link` (`LinkCR`, under `ecr/sha-512`) and `δ_tree`
  (`TreeRewind`, under `cr/sha-512`) need a `--zk` compiled table and session. [e2e-exec]'s salted session
  (`Discharge/Exec/SessionL.lean`) fixes the table's openings to `Opens × Salts`; a version generic over the per-table
  game would carry `TabZK.after`'s compiled form.
* *The `--zk` executable refinement*: that the executable verifier's `--zk` path runs this protocol (for [e2e-exec],
  as `verify_refines_hm96` does at M0), and with it the headline's `execAccept`.
* *`final_c'`'s place.* The protocol as Lean defines it receives `final_c'` before the inner proof (`restZK`); the live
  verifier never absorbs it, and a prover that picks it after the inner proof's batching coins can cancel the batched
  constraint. This statement describes the live verifier once it absorbs `final_c'` before the inner proof's label.
* *The coin step in the session.* Coins here are fresh after each prover message (A6's per-round form). M2 draws them
  from a coin tree committed before the prover's first message; `coin_fresh` / `coinLeaf_fresh` (Coins.lean) bound each
  round's coins within `δ_hide` of fresh from `Assumptions.Hm96Hiding`, but composing that into this session needs a
  round-by-round form of the session bound or a committed-coin semantics of `Game`.
* *M0's structural guarantees*, unchanged: the drawn units' places (`UnitPlace`: W6, `Layout.Aliased`), the
  constant and zero columns, and the rows' commit (`HmRowComputes`), as the headline takes them.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting Audit
open scoped ENNReal NNReal

/-- **A `--zk` table of a batched session**: the statement and its schedule, the padded level 0 (`t` padding columns,
`c₀`, `e` extra lanes), the link layout for the session's `m_pts`, and the inner proof's layout: the pads (level 0's
`pads`, the algebraic phase's `padsA`, the triple's `tr`), the pads code `RS[F, dom, np + nP]` read at `q` positions
`pos`, and a radius `δ` within its unique-decoding radius. -/
structure TabZK (F : Type) [Field F] [Fintype F] [DecidableEq F] (mPts : ℕ) where
  S : Statement
  sch : Schedule
  hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
    sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧ levelsChained sch.levels sch.ext = true
  hmsch : sch.m = S.m
  hm : 13 ≤ S.m
  t : ℕ
  hpad : PadOK sch.level₀ t
  c₀ : F
  e : ℕ
  ptLocal : ℕ
  hlay : S.LinkLayout ptLocal mPts
  np : ℕ
  pads : Pads0 np e
  padsA : PadsA np
  tr : Triple np
  N : ℕ
  dom : Fin N ↪ F
  nP : ℕ
  q : ℕ
  hnP : NeZero (np + nP)
  pos : F → Fin N
  hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N
  δ : ℝ≥0
  hL : np + nP ≤ N
  hδ : δ ≤ Code.relativeUniqueDecodingRadius (ReedSolomon.code dom (np + nP) : Set (Fin N → F))

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

namespace TabZK

/-- The table's committed values satisfy its statement (padded level 0). -/
def Committed (A : Arith F K) (T : TabZK F mPts) (C : Oracle F) : Prop :=
  CommittedPad A T.S T.sch T.t T.c₀ C

/-- The table's soundness error: `padLinkError + (padRepError + εInner)²`. -/
noncomputable def bound (T : TabZK F mPts) : ℝ≥0∞ :=
  ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t) +
    (ENNReal.ofReal (padRepError T.sch (T.S.shape mPts) T.t T.e) + εInner T.N F T.q T.δ) ^ 2

end TabZK

/-! ## The session -/

/-! ## The audit -/

section Audit

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- The table of the plan that carries drawn unit `u`. -/
abbrev tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) : TabZK F mPts := (planZ S R).get (tab S R u)

/-- A decoder: for each drawn unit, the values a level-0 message of its table packs for it. -/
abbrev DecoderZK := (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
  Message F (tabOfZ planZ tab S R u).sch.k0 (tabOfZ planZ tab S R u).sch.level₀.logCols → (Fin C.N → Bool)

/-- The decoder of the drawn units' places: each unit is read from its block of the witness. -/
noncomputable def decodeOfPlace (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u) :
    DecoderZK (C := C) planZ tab :=
  fun S R u M => if h : u ∈ S then
    (place S R u h).decode (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M)
  else fun _ => false

variable {L : Law n}

end Audit

end FlockSoundness.Discharge.ZkSession
