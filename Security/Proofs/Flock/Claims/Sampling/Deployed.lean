import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Level3.RowV2
import Proofs.Flock.Claims.Sampling.Reference.Prover

/-!
# The deployed sampling record

`SamplingDeploy.deploy rel c`: the record the sampling claims read (`Claims.Sampling.Deploy`) with what the deployment
fixes set, and the sampled verifier's code (`SamplingDeploy.Code`) as its parameter:

* `flock`: C-Flock's deployed masked verifier, `FlockDeploy.deployZk rel`; the sessions' `E^rep` waits and truncation
  are its (`innerWaits`, `innerCap`: `Numbers.waits`, `Numbers.cap`).
* `commit`: hm96 leaves (`hm96Leaf`) of `sha512/row/v2` bit rows holding the opened bit at its place (`row`), under the
  verifier's own registered frame-v3-sha512 path check (`regPath`); the row's other bits, the salt and the path are
  read off the opening's data, as the reveal encodes them.
* `universal`: the universal unit's Definition (`universalProg`).
* `εtable`, `weight`: Flock's per-table bounds, `2^{-205}` on a table's error (`FlockDeploy.tableErr_zk_le`) and
  `2^33` on its weight (`FlockDeploy.tableWeight_zk_le`), times the most tables a session at the statement sets up
  (`Code.tables`): each session's sums are at most these (`FlockDeploy.εtable_le`, `FlockDeploy.weight_zk_le`).
* `openCalls`: the most openings one session reveals (`Code.reveals`) times one opening's calls, at most
  `4 + depth own` (`opens_calls_le`): the leaf's three, the leaf's frame, and one node per level of the deepest
  registered port.
* `Tape`, `P`: the sampled reference prover (`Reference.refP`), at the code's framing, on a Flock tape per session
  statement: in each session, Flock's masked reference prover on what it reveals, checked by this commitment.
* `δc`: the sessions' count (`Code.sessions`) times the largest of Flock's masked error `FlockComplete.δcZk`, capped
  at `1`, over the sessions' statements at `x` (`Reference.sessErrAt`): the union bound over the run's sessions.
  One session that proves every drawn unit has error `δcZk` at its statement.
* The rest is the code's: the verifier in public-coin form, its parser, scope check and partition evaluation, its draw
  layout, the checked units' sessions and what their witnesses reveal, the sessions' `N₀` and count, and the reference
  prover's framing.

What is then a fact about the record: its sessions are `deployZk`'s (`deploy_flock`, `rfl`), its `ε_table` is
nonnegative (`deploy_εtable_nonneg`), its commitment is hm96 leaves under the registered path check (`deploy_commit`,
`rfl`) of rows that place the value (`row_places`), its universal unit is the Definition's (`universalIs_deploy`), its
opening check's calls are bounded by the reveal count (`openCallsLe_deploy`, from `RevealsLe`), and the inner
numbers are Flock's (`innerNumbers_deploy`: the waits and truncation by definition, the rest from `SessionsIn`, each
session in Flock's masked scope). `sampling_deployed` is `sampling_assembled` at the record, those discharged.
-/

namespace SamplingDeploy

open Claims Claims.Sampling

/-- **The sampled verifier's code**, as the record reads it: the verifier in public-coin form (`V`), its parser, scope
check and partition evaluation; its draw layout (`stage2`, `checked`, `rounds`); each checked unit's session and what
its Flock witness reveals (`session`, `reveal`); the trace commitment's registered ports (`own`), the root and leaf
index of each position under a run's roots (`at_`) and its published leaf layer (`layer`); where each committed bit
sits, its row's width and its place there (`place`), and how the reveal encodes an opening's data: the row's other bits,
the salt and the path (`bits`, `salt`, `π`); and its numbers: the sessions' `N₀`, the most tables a session at a
statement sets up (`tables`), the most openings one session reveals (`reveals`) and the slots whose checked units'
sessions are all of a run's (`sessions`: `1` when one session proves every drawn unit, the proof units' count at one
session per unit); and the reference prover's framing (`Reference.refP`): where a run is at a history (`step`), the
positions a session opens (`reads`) and the Flock witness that encodes what it reveals (`wit`), each by the session's
statement. -/
structure Code where
  V : Verifier FlockDeploy.Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  partition : Prog → Query → Option Partition
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → FlockDeploy.Coin) → List (ℕ × ℕ)
  rounds : (x : Msg) → ℕ × ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ × ℕ → Seen → Option (Msg × Seen)
  reveal : Stmt → ℕ × ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)
  own : Array _root_.Flock.Registered.Port
  at_ : List Msg → ℕ → Option (Msg × ℕ)
  layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg
  place : ℕ → (w : ℕ) × Fin w
  bits : Msg → ℕ → Bool
  salt : Opening Bool → Msg
  π : Opening Bool → Msg
  innerN₀ : Msg → ℝ
  tables : Msg → ℕ
  reveals : Msg → ℕ
  sessions : Msg → ℕ
  step : Msg → List (Msg × Msg) → Reference.Step
  reads : Stmt → Msg → List ℕ
  wit : Stmt → Msg → List (Opening Bool) → Claims.Flock.Wit

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

/-- **The deployed trace commitment**: hm96 leaves of the rows under the verifier's own registered path check. -/
def commit (c : Code) : Commitment Bool :=
  pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath c.own) c.layer

/-- The deepest registered port's levels. -/
def depth (own : Array _root_.Flock.Registered.Port) : ℕ :=
  own.toList.foldr (fun p m => max (_root_.Flock.log2ceil p.leaves) m) 0

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

/-- **One opening's calls**: at most the leaf's three, the leaf's frame and one node per level of the deepest port. -/
theorem opens_calls_le (H : Msg → Msg) (c : Code) (roots : List Msg) (o : Opening Bool) :
    (((commit c).opens roots o).eval H).2.length ≤ 4 + depth c.own := by
  simp only [commit, pathCommit]
  split
  · rename_i ri _
    rw [ocBind_eval, List.length_append, hm96Leaf_log]
    simp only [List.length_cons, List.length_nil]
    split
    · rename_i l _
      have := regPath_calls H c.own ri.1 ri.2 l (c.π o)
      omega
    · simp only [OC.eval, List.length_nil]
      omega
  · simp [OC.eval]

/-! ## The record -/

/-- The proof units at `x` under the code's parser and partition evaluation, `0` on a statement that doesn't parse. -/
def units (c : Code) (x : Msg) : ℕ :=
  match c.parse x with
  | some s => ((c.partition s.prog s.query).getD ⟨[]⟩).sizes.sum
  | none => 0

/-- **The deployed sampling record** at the relation's fields `rel` and the verifier's code `c`. -/
noncomputable def deploy (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace where
  V := c.V
  parse := c.parse
  scopeCheck := c.scopeCheck
  partition := c.partition
  commit := commit c
  universal := universalProg
  flock := FlockDeploy.deployZk rel
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
  Tape := Reference.Tape
  P := Reference.refP c.step c.parse c.reads (commit c) c.wit
  δc H x := c.sessions x * Reference.sessErrAt c.session (FlockComplete.δcZk H) x

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

theorem deploy_flock : (deploy rel c).flock = FlockDeploy.deployZk rel := rfl

theorem deploy_εtable_nonneg (x : Msg) : 0 ≤ (deploy rel c).εtable x := by
  show 0 ≤ (c.tables x : ℝ) * 2 ^ (-(205 : ℝ))
  positivity

theorem deploy_commit :
    (deploy rel c).commit = pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath c.own) c.layer :=
  rfl

theorem universalIs_deploy : UniversalIs (deploy rel c) := fun _ _ => rfl

/-- **The reveal count**: at a statement that parses, what one session's Flock witness reveals is at most `reveals`
openings. -/
def RevealsLe : Prop :=
  ∀ x s, c.parse x = some s → ∀ u v wF, (c.reveal s u v wF).length ≤ c.reveals x

/-- **The opening check's calls are bounded by the reveal count**: each opening's are at most `4 + depth own`. -/
theorem openCallsLe_deploy (H : Msg → Msg) (h : RevealsLe c) : OpenCallsLe H (deploy rel c) := by
  intro x s hs u v wF
  show ((c.reveal s u v wF).map fun p => (((commit c).opens p.1 p.2).eval H).2.length).sum ≤
    c.reveals x * (4 + depth c.own)
  calc ((c.reveal s u v wF).map fun p => (((commit c).opens p.1 p.2).eval H).2.length).sum
      ≤ ((c.reveal s u v wF).map fun _ => 4 + depth c.own).sum :=
        List.sum_le_sum fun p _ => opens_calls_le H c p.1 p.2
    _ = (c.reveal s u v wF).length * (4 + depth c.own) := by simp
    _ ≤ c.reveals x * (4 + depth c.own) := Nat.mul_le_mul_right _ (h x s hs u v wF)

/-- **The sessions are in Flock's masked scope**: every session's statement sets up at most `tables` tables, each at
an `m` and of a shape the masked verifier serves (`FlockDeploy.ZkIn`), and its `N₀` is the record's. -/
def SessionsIn : Prop :=
  ∀ x u v xS vS, c.session x u v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ true xS = c.innerN₀ x

/-- **The inner numbers are Flock's**: the waits and truncation by definition; at a session in Flock's masked scope,
its `ε_table` and weight are at most its tables' count times their bounds, so at most the record's. -/
theorem innerNumbers_deploy (h : SessionsIn c) : InnerNumbers (deploy rel c) := by
  intro x u v xS vS hs
  obtain ⟨⟨s, spec, sts, hz, hn⟩, hN⟩ := h x u v xS vS hs
  have hn' : (sts.size : ℝ) ≤ c.tables x := by exact_mod_cast hn
  refine ⟨rfl, rfl, hN, ?_, ?_⟩
  · refine (FlockDeploy.εtable_le hz.setup fun st hst => ?_).trans (mul_le_mul_of_nonneg_right hn' (by positivity))
    have := hz.tab st hst
    exact FlockDeploy.tableErr_zk_le spec st this.1 this.2.1 this.2.2
  · exact (FlockDeploy.weight_zk_le rel hz).trans (mul_le_mul_of_nonneg_right hn' (by positivity))

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record**, at every hash: `sampling_assembled` with the record's facts
discharged. The hypotheses are what remains: Flock's KS at `deployZk` (`hKS`); the draw, sessions and their scope,
slots, session openings, interior roots and scope checks of the sampled verifier's run (`DrawFixed`, `SessionRuns`,
`SessionsIn`, `DrawSlots`, `SessionOpens`, `InteriorFixed`, `ProgramScope`, `ChecksScope`); its reveal count
(`RevealsLe`); completeness (`complete_deploy` proves it from its own facts) and zero knowledge. -/
theorem sampling_deployed (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hfixed : DrawFixed (deploy rel c)) (hsessions : SessionRuns H (deploy rel c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deploy rel c) iroots) (hfix : InteriorFixed H (deploy rel c) iroots)
    (hprog : ProgramScope (deploy rel c)) (hchecks : ChecksScope H (deploy rel c)) (hreveals : RevealsLe c)
    (hComplete : Complete H (deploy rel c))
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (Pz : Wit → Prover (Nz × Ωz)) (S : SimLeak)
    (εzk γzk : Msg → ℝ) (hZK : ZK H (deploy rel c) Pz S εzk γzk) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      KS H (deploy rel c) ∧ KSCount H (deploy rel c) ∧ Bits H (deploy rel c) ∧
      Efficient H (deploy rel c) FlockDeploy.zkStepPoly ∧ Complete H (deploy rel c) ∧
      ZK H (deploy rel c) Pz S εzk γzk :=
  sampling_assembled rel (deploy rel c) H (deploy_flock rel c) hKS (deploy_εtable_nonneg rel c) c.own c.at_
    (row c.place c.bits) c.salt c.π c.layer (deploy_commit rel c) (row_places c.place c.bits) hfixed hsessions
    (innerNumbers_deploy rel c hnum) hslots iroots hopens hfix hprog hchecks (openCallsLe_deploy rel c H hreveals)
    (universalIs_deploy rel c) hComplete Pz S εzk γzk hZK

end SamplingDeploy
