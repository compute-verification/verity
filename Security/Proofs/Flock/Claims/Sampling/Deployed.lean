module

public import Proofs.Flock.Claims.NumbersDeploy
public import Proofs.Flock.Claims.Sampling.Assembled
public import Proofs.Flock.Claims.Sampling.Code
public import Proofs.Flock.Claims.Sampling.Reference.Prover
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The deployed sampling record

`SamplingDeploy.deployOf F c`: the record the sampling claims read (`Claims.Sampling.Deploy`) with what the deployment
fixes set, at the sessions' Flock record `F` and the sampled verifier's code (`SamplingDeploy.Code`). Its instances are
`deploy rel c`, the sessions Flock's deployed masked verifier `FlockDeploy.deployZk rel`, and `deployPlain rel c`, its
deployed plain verifier at the trimmed record Flock's plain KS is stated at, `FlockDeploy.deployPlainTrim rel`:

* `flock`: `F`; the sessions' `E^rep` waits and truncation are the deployment's (`innerWaits`, `innerCap`:
  `Numbers.waits`, `Numbers.cap`).
* `commit`: hm96 leaves (`hm96Leaf`) of `sha512/row/v2` bit rows holding the opened bit at its place (`row`), under the
  verifier's own registered frame-v3-sha512 path check (`regPath`) at the ports the opening's position reads under its
  roots (`Code.ownAt`, `own` unless the code sets it); the row's other bits, the salt and the path are read off the
  opening's data, as the reveal encodes them.
* `universal`: the universal unit's Definition (`universalProg`).
* `εtable`, `weight`: Flock's per-table bounds, `2^{-205}` on a table's error (`FlockDeploy.tableErr_zk_le`) and
  `2^33` on its weight (`FlockDeploy.tableWeight_zk_le`), times the most tables a session at the statement sets up
  (`Code.tables`): each session's sums are at most these (`FlockDeploy.εtable_le`, `FlockDeploy.weight_zk_le`).
* `openCalls`: the most openings one session reveals (`Code.reveals`) times one opening's calls, at most
  `4 + depth own` (`opens_calls_le`): the leaf's three, the leaf's frame, and one node per level of the deepest
  registered port, no position's ports being deeper (`Code.ownAt_depth`).
* `Tape`, `P`: the sampled reference prover (`Reference.refP F.P`), at the code's framing, on one of `F`'s tapes per
  session statement: in each session, `F`'s reference prover on what it reveals, checked by this commitment.
* `δc`: the sessions' count (`Code.sessions`) times the largest of `F`'s error `F.δc` (at `deployZk`,
  `FlockComplete.ZkTables.δc`), capped at `1`, over the statements of the reference prover's sessions at `x`, at an
  honest witness and coins that check their units (`Reference.sessErrAt`): the union bound over the run's sessions.
  One session that proves every sampled proof block has error `F.δc` at its statement.
* The rest is the code's: the verifier in public-coin form, its parser, scope check and partition evaluation, its sample
  layout, the checked units' sessions and what their witnesses reveal, the sessions' `N₀` and count, and the reference
  prover's framing.

What is then a fact about the record at any `F`: its `ε_table` is nonnegative (`deployOf_εtable_nonneg`), its
commitment is the code's (`deployOf_commit`, `rfl`), which binds at every `H` (`commit_binds`: at each opening's roots
and position, hm96 leaves of rows that place the value, `row_places`, under the registered path check at the ports
there), its universal unit is the Definition's (`universalIs_deployOf`), its opening check's calls are bounded by the
reveal count (`openCallsLe_deployOf`, from `RevealsLe`), and the inner numbers are Flock's where `F`'s are bounded by
the tables' (`innerNumbers_deployOf`; at each unit's `N₀`, `innerNumbersAt_deployOf`). `sampling_deployed_of` is
`sampling_assembled_bind` at the record, those discharged; `sampling_deployed` adds zero knowledge at `deploy`. The
inner numbers are Flock's at `deploy` where every session is in Flock's masked scope (`SessionsIn`,
`innerNumbers_deploy`), and at `deployPlain` where every session is in Flock's plain scope (`SessionsInPlain`,
`innerNumbers_deployPlain`): the compiled sampled verifier's knowledge soundness at the deployed record reads them
(`SamplingDeploy.ks_compiled`). `SessionsIn` holds each session in Flock's masked scope at its unit's `N₀`
(`Code.unitN₀`), which gives the inner numbers at each unit's `N₀` (`innerNumbersAt_deploy`), and at the code with every
unit at its statement's `N₀` (`Code.atInnerN₀`) the one-stage ones (`innerNumbers_deploy`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling

/-! ## The trace commitment -/

/-- **The opened bit's row**: the bit row of its place's width, the opened value at its place and the other bits read
off the opening's data. -/
def rowBits (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (o : Opening Bool) : List Bool :=
  List.ofFn fun i : Fin (place o.pos).1 => if i = (place o.pos).2 then o.val else bits o.data i

/-- **Its `sha512/row/v2` input**: `P(ℓ)`, then the bits (`FlockLevel3.RowV2.enc`). -/
def row (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (o : Opening Bool) : Msg :=
  FlockLevel3.RowV2.enc (rowBits place bits o)

/-- **The row places the value**: two openings at one position have rows of one width with their values at one place,
and `sha512/row/v2`'s bytes fix a row of a given width. -/
theorem row_places (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) : RowPlaces (row place bits) := by
  rintro ⟨p, v, d⟩ ⟨p', v', d'⟩ hp hv h
  simp only at hp hv
  subst hp
  apply hv
  have hl : (rowBits place bits ⟨p, v, d⟩).length = (rowBits place bits ⟨p, v', d'⟩).length := by
    simp only [rowBits, List.length_ofFn]
  unfold row FlockLevel3.RowV2.enc at h
  rw [hl] at h
  have hb := FlockLevel3.RowV2.packBits_inj hl (List.append_cancel_left h)
  simp only [rowBits] at hb
  simpa using congrFun (List.ofFn_inj.1 hb) (place p).2

/-- **The deployed trace commitment**: hm96 leaves of the rows under the verifier's own registered path check, at the
ports the opening's position reads under its roots (`Code.ownAt`). -/
def commit (c : Code) : CommitmentScheme Bool where
  opens roots o :=
    (pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath (c.ownAt roots o.pos)) c.layer).opens roots o
  layer := c.layer

/-- **The deployed trace commitment binds, at every `H`**: at an opening's roots and position, hm96 leaves of rows that
place the value (`row_places`) under the registered path check at the ports there (`pathCommit_binds`). -/
theorem commit_binds (c : Code) (H : Msg → Msg) : (commit c).Binds H := by
  intro roots o o' hp
  have e : ∀ o'' : Opening Bool, o''.pos = o.pos → (commit c).opens roots o'' =
      (pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath (c.ownAt roots o.pos)) c.layer).opens
        roots o'' := by
    intro o'' h
    show (pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath (c.ownAt roots o''.pos)) c.layer).opens
      roots o'' = _
    rw [h]
  unfold CommitmentScheme.Opens
  rw [e o rfl, e o' hp.symm]
  exact pathCommit_binds (at_ := c.at_) (π := c.π) (layer := c.layer)
    (hm96Leaf_binds H (row_places c.place c.bits) c.salt) (regPath_binds H (c.ownAt roots o.pos)) roots o o' hp

theorem le_foldr_max {α : Type} (f : α → ℕ) {a : α} :
    ∀ {l : List α}, a ∈ l → f a ≤ l.foldr (fun p m => max (f p) m) 0
  | [], h => by simp at h
  | _ :: _, h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact (le_foldr_max f h).trans (le_max_right _ _)

theorem le_depth {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port} (h : reg ∈ own) :
    _root_.Flock.log2ceil reg.leaves ≤ depth own :=
  le_foldr_max (fun p : _root_.Flock.Registered.Port => _root_.Flock.log2ceil p.leaves) (Array.mem_toList_iff.2 h)

/-! ## One opening's calls -/

theorem climbOC_calls (H : Msg → Msg) (dom : ByteArray) (root : Msg) :
    ∀ (cur : Msg) (idx d : ℕ) (sibs : List ByteArray), ((climbOC dom root cur idx d sibs).eval H).2.length = sibs.length
  | _, _, _, [] => rfl
  | cur, idx, d, _ :: rest => by
    simp only [climbOC, OC.eval, List.length_cons, climbOC_calls H dom root _ _ _ rest]

theorem regPath_calls (H : Msg → Msg) (own : Array _root_.Flock.Registered.Port) (root : Msg) (pos : ℕ) (leaf π : Msg) :
    ((regPath own root pos leaf π).eval H).2.length ≤ 1 + depth own := by
  unfold regPath
  split
  · rename_i reg hreg
    unfold regPathAt
    split
    · simp only [OC.eval, List.length_cons, climbOC_calls, _root_.Flock.Registered.sibs_length]
      have := le_depth (Array.mem_of_find?_eq_some hreg)
      omega
    · simp [OC.eval]
  · simp [OC.eval]

/-- **One opening's calls**: at most the leaf's three, the leaf's frame and one node per level of the deepest port its
position reads, which is no deeper than the registered ports (`Code.ownAt_depth`). -/
theorem opens_calls_le (H : Msg → Msg) (c : Code) (roots : List Msg) (o : Opening Bool) :
    (((commit c).opens roots o).eval H).2.length ≤ 4 + depth c.own := by
  have hd := c.ownAt_depth roots o.pos
  simp only [commit, pathCommit]
  split
  · rename_i ri _
    rw [ocBind_eval, List.length_append, hm96Leaf_log]
    simp only [List.length_cons, List.length_nil]
    split
    · rename_i l _
      have := regPath_calls H (c.ownAt roots o.pos) ri.1 ri.2 l (c.π o)
      omega
    · simp only [OC.eval, List.length_nil]
      omega
  · simp [OC.eval]

/-! ## The record -/

/-- **The deployed sampling record** at a Flock record `F`, the sessions', and the verifier's code `c`: the sessions'
reference prover and error are `F`'s (`F.P`, `F.δc`). -/
noncomputable def deployOf {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T where
  V := c.V
  parse := c.parse
  scopeCheck := c.scopeCheck
  partition := c.partition
  commit := commit c
  universal := universalProg
  flock := F
  stage2 := c.stage2
  checked := c.checked
  rounds := c.rounds
  session := c.session
  reveal := c.reveal
  innerWaits _ := Claims.Flock.Numbers.waits
  innerN₀ := c.innerN₀
  innerCap _ := Claims.Flock.Numbers.cap
  εtable x := c.tables x * 2 ^ (-(205 : ℝ))
  weight x := c.tables x * 2 ^ 33
  openCalls x := c.reveals x * (4 + depth c.own)
  Tape := Msg → F.Tape
  P := Reference.refP F.P c.step c.parse c.reads (commit c) c.wit
  δc H x := c.sessions x * Reference.sessErrAt H (Rhon H (commit c) c.parse c.partition)
    (Reference.refP F.P c.step c.parse c.reads (commit c) c.wit) c.V c.checked c.session (F.δc H) x

/-- **The deployed sampling record** at the relation's fields `rel` and the verifier's code `c`, its sessions Flock's
masked proofs. -/
noncomputable def deploy (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  deployOf (FlockDeploy.deployZk rel) c

/-- **The deployed plain sampling record**: its sessions Flock's plain proofs, at the trimmed record Flock's plain KS
is stated at. -/
noncomputable def deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  deployOf (FlockDeploy.deployPlainTrim rel) c

section Of

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)

theorem deployOf_εtable_nonneg (x : Msg) : 0 ≤ (deployOf F c).εtable x := by
  show 0 ≤ (c.tables x : ℝ) * 2 ^ (-(205 : ℝ))
  positivity

theorem deployOf_commit : (deployOf F c).commit = commit c := rfl

theorem universalIs_deployOf : UniversalIs (deployOf F c) := fun _ _ => rfl

/-- **The reveal count**: at a statement that parses, what one session's Flock witness reveals is at most `reveals`
openings. -/
def RevealsLe : Prop :=
  ∀ x s, c.parse x = some s → ∀ u v wF, (c.reveal s u v wF).length ≤ c.reveals x

/-- **The opening check's calls are bounded by the reveal count**: each opening's are at most `4 + depth own`. -/
theorem openCallsLe_deployOf (H : Msg → Msg) (h : RevealsLe c) : OpenCallsLe H (deployOf F c) := by
  intro x s hs u v wF
  show ((c.reveal s u v wF).map fun p => (((commit c).opens p.1 p.2).eval H).2.length).sum ≤
    c.reveals x * (4 + depth c.own)
  calc ((c.reveal s u v wF).map fun p => (((commit c).opens p.1 p.2).eval H).2.length).sum
      ≤ ((c.reveal s u v wF).map fun _ => 4 + depth c.own).sum :=
        List.sum_le_sum fun p _ => opens_calls_le H c p.1 p.2
    _ = (c.reveal s u v wF).length * (4 + depth c.own) := by simp
    _ ≤ c.reveals x * (4 + depth c.own) := Nat.mul_le_mul_right _ (h x s hs u v wF)

/-- **The inner numbers are Flock's at each unit's `N₀`** where `F`'s waits and truncation are the deployment's and
every session of unit `u` at `x` has `N₀ = n₀ x u`, its `ε_table` and weight at most some `n ≤ tables` tables'
bounds. -/
theorem innerNumbersAt_deployOf (n₀ : Msg → ℕ → ℝ) (hw : ∀ xS, F.waits xS = Claims.Flock.Numbers.waits)
    (hc : ∀ xS, F.cap xS = Claims.Flock.Numbers.cap)
    (h : ∀ x u v xS vS, c.session x u v = some (xS, vS) → F.N₀ xS = n₀ x u ∧
      ∃ n : ℕ, n ≤ c.tables x ∧ F.εtable xS ≤ n * 2 ^ (-(205 : ℝ)) ∧ F.weight xS ≤ n * 2 ^ 33) :
    InnerNumbersAt (deployOf F c) n₀ := by
  intro x u v xS vS hs
  obtain ⟨hN, n, hn, he, hwt⟩ := h x u v xS vS hs
  have hn' : (n : ℝ) ≤ c.tables x := by exact_mod_cast hn
  exact ⟨hw xS, hc xS, hN, he.trans (mul_le_mul_of_nonneg_right hn' (by positivity)),
    hwt.trans (mul_le_mul_of_nonneg_right hn' (by positivity))⟩

/-- **The inner numbers are Flock's** where `F`'s waits and truncation are the deployment's and every session's `N₀`
is the record's, its `ε_table` and weight at most some `n ≤ tables` tables' bounds. -/
theorem innerNumbers_deployOf (hw : ∀ xS, F.waits xS = Claims.Flock.Numbers.waits)
    (hc : ∀ xS, F.cap xS = Claims.Flock.Numbers.cap)
    (h : ∀ x u v xS vS, c.session x u v = some (xS, vS) → F.N₀ xS = c.innerN₀ x ∧
      ∃ n : ℕ, n ≤ c.tables x ∧ F.εtable xS ≤ n * 2 ^ (-(205 : ℝ)) ∧ F.weight xS ≤ n * 2 ^ 33) :
    InnerNumbers (deployOf F c) :=
  innerNumbersAt_deployOf F c (fun x _ => c.innerN₀ x) hw hc h

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record over any Flock record**, at every hash: `sampling_assembled_bind` at
`deployOf F c` with the record's facts discharged. The hypotheses are what remains: the slots and the program scope of
the sampled verifier's run (`DrawSlots`, `ProgramScope`) and completeness. -/
theorem sampling_deployed_of (H : Msg → Msg) (hslots : DrawSlots (deployOf F c)) (hprog : ProgramScope (deployOf F c))
    (hComplete : Complete H (deployOf F c)) :
    ScopeMatches (deployOf F c) ∧ Binding (deployOf F c) ∧ Universal (deployOf F c) ∧ FaultyBoundary ∧
      Complete H (deployOf F c) :=
  sampling_assembled_bind (deployOf F c) H (commit_binds c) hslots hprog (universalIs_deployOf F c) hComplete

end Of

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-! ## At the masked record -/

theorem deploy_flock : (deploy rel c).flock = FlockDeploy.deployZk rel := rfl

theorem deploy_εtable_nonneg (x : Msg) : 0 ≤ (deploy rel c).εtable x :=
  deployOf_εtable_nonneg (FlockDeploy.deployZk rel) c x

theorem deploy_commit : (deploy rel c).commit = commit c := rfl

theorem universalIs_deploy : UniversalIs (deploy rel c) := universalIs_deployOf (FlockDeploy.deployZk rel) c

theorem openCallsLe_deploy (H : Msg → Msg) (h : RevealsLe c) : OpenCallsLe H (deploy rel c) :=
  openCallsLe_deployOf (FlockDeploy.deployZk rel) c H h

/-- **The sessions are in Flock's masked scope**: every session's statement sets up at most `tables` tables, each at
an `m` and of a shape the masked verifier serves (`FlockDeploy.ZkIn`), and its `N₀` is its unit's (`unitN₀`). -/
def SessionsIn : Prop :=
  ∀ x u v xS vS, c.session x u v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ true xS = c.unitN₀ x u

/-- **The inner numbers are Flock's at each unit's `N₀`**: the waits and truncation by definition; at a session in
Flock's masked scope, its `ε_table` and weight are at most its tables' count times their bounds, so at most the
record's. -/
theorem innerNumbersAt_deploy (h : SessionsIn c) : InnerNumbersAt (deploy rel c) c.unitN₀ :=
  innerNumbersAt_deployOf (FlockDeploy.deployZk rel) c c.unitN₀ (fun _ => rfl) (fun _ => rfl) fun x u v xS vS hs => by
    obtain ⟨⟨s, spec, sts, hz, hn⟩, hN⟩ := h x u v xS vS hs
    refine ⟨hN, sts.size, hn, FlockDeploy.εtable_le hz.setup fun st hst => ?_, FlockDeploy.weight_zk_le rel hz⟩
    have := hz.tab st hst
    exact FlockDeploy.tableErr_zk_le spec st this.1 this.2.1 this.2.2

/-- **The inner numbers are Flock's**: `innerNumbersAt_deploy` at the code with every unit at its statement's `N₀`,
whose record is `deploy rel c`. -/
theorem innerNumbers_deploy (h : SessionsIn c.atInnerN₀) : InnerNumbers (deploy rel c) :=
  innerNumbersAt_deploy rel c.atInnerN₀ h

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record**, at every hash: `sampling_deployed_of` at the masked record
`deployZk`, and zero knowledge. The hypotheses are what remains: the slots and the program scope of the sampled
verifier's run (`DrawSlots`, `ProgramScope`); completeness (`complete_deploy` proves it from its own facts) and zero
knowledge. -/
theorem sampling_deployed (H : Msg → Msg) (hslots : DrawSlots (deploy rel c)) (hprog : ProgramScope (deploy rel c))
    (hComplete : Complete H (deploy rel c))
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (tapeOf : Nz × Ωz → Reference.Tape)
    (S : SimLeak) (εzk γzk : Msg → ℝ) (hZK : ZK H ((deploy rel c).mask Nz Ωz tapeOf S) εzk γzk) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      Complete H (deploy rel c) ∧ ZK H ((deploy rel c).mask Nz Ωz tapeOf S) εzk γzk := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := sampling_deployed_of (FlockDeploy.deployZk rel) c H hslots hprog hComplete
  exact ⟨h1, h2, h3, h4, h5, hZK⟩

/-! ## At the plain record -/

theorem deployPlain_flock : (deployPlain rel c).flock = FlockDeploy.deployPlainTrim rel := rfl

/-- **The sessions are in Flock's plain scope**: every session's statement sets up at most `tables` tables, each at
an `m` and of a shape the plain verifier serves (`FlockDeploy.PlainIn`), and its `N₀` is the record's. -/
def SessionsInPlain : Prop :=
  ∀ x u v xS vS, c.session x u v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.PlainIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ false xS = c.innerN₀ x

/-- **The inner numbers are Flock's** at the plain record: at a session in Flock's plain scope, its `ε_table` and
weight are at most its tables' count times their bounds. -/
theorem innerNumbers_deployPlain (h : SessionsInPlain c) : InnerNumbers (deployPlain rel c) :=
  innerNumbers_deployOf (FlockDeploy.deployPlainTrim rel) c (fun _ => rfl) (fun _ => rfl) fun x u v xS vS hs => by
    obtain ⟨⟨s, spec, sts, hz, hn⟩, hN⟩ := h x u v xS vS hs
    refine ⟨hN, sts.size, hn, FlockDeploy.εtable_le hz.setup fun st hst => ?_, FlockDeploy.weight_plain_le rel hz⟩
    have := hz.tab st hst
    exact FlockDeploy.tableErr_plain_le spec st this.1 this.2.1 this.2.2

end SamplingDeploy
