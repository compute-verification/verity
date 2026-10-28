import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Level3.RowV2
import Proofs.Flock.Claims.Sampling.Reference.Prover

/-!
# The deployed sampling record

`SamplingDeploy.deployOf F c`: the record the sampling claims read (`Claims.Sampling.Deploy`) with what the deployment
fixes set, at the sessions' Flock record `F` and the sampled verifier's code (`SamplingDeploy.Code`). Its instances are
`deploy rel c`, the sessions C-Flock's deployed masked verifier `FlockDeploy.deployZk rel`, and `deployPlain rel c`, its
deployed plain verifier at the trimmed record Flock's plain KS is stated at, `FlockDeploy.deployPlainTrim rel`:

* `flock`: `F`; the sessions' `E^rep` waits and truncation are the deployment's (`innerWaits`, `innerCap`:
  `Numbers.waits`, `Numbers.cap`).
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
* `Tape`, `P`: the sampled reference prover (`Reference.refP F.P`), at the code's framing, on one of `F`'s tapes per
  session statement: in each session, `F`'s reference prover on what it reveals, checked by this commitment.
* `δc`: the sessions' count (`Code.sessions`) times the largest of `F`'s error `F.δc` (at `deployZk`,
  `FlockComplete.ZkTables.δc`), capped at `1`, over the statements of the reference prover's sessions at `x`, at an
  honest witness and coins that check their units (`Reference.sessErrAt`): the union bound over the run's sessions.
  One session that proves every drawn unit has error `F.δc` at its statement.
* The rest is the code's: the verifier in public-coin form, its parser, scope check and partition evaluation, its draw
  layout, the checked units' sessions and what their witnesses reveal, the sessions' `N₀` and count, and the reference
  prover's framing.

What is then a fact about the record at any `F`: its `ε_table` is nonnegative (`deployOf_εtable_nonneg`), its
commitment is hm96 leaves under the registered path check (`deployOf_commit`, `rfl`) of rows that place the value
(`row_places`), its universal unit is the Definition's (`universalIs_deployOf`), its opening check's calls are bounded
by the reveal count (`openCallsLe_deployOf`, from `RevealsLe`), and the inner numbers are Flock's where `F`'s are
bounded by the tables' (`innerNumbers_deployOf`). `sampling_deployed_of` is `sampling_assembled_of` at the record,
those discharged, reading of `F` its KS and its decoder's `ListWithin` and step count. Its instances:

* `sampling_deployed` at `deploy`, with zero knowledge: the inner numbers from `SessionsIn` (`innerNumbers_deploy`, each
  session in Flock's masked scope), the decoder's facts `sampling_assembled`'s;
* `sampling_deployed_plain` at `deployPlain`: the inner numbers from `SessionsInPlain` (`innerNumbers_deployPlain`,
  each session in Flock's plain scope), the decoder's facts `sampling_assembled_plain`'s.
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
  parse : Msg → Option AuditStatement
  scopeCheck : Msg → Bool
  partition : AuditStatement → Option Partition
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → FlockDeploy.Coin) → List (ℕ × ℕ)
  rounds : (x : Msg) → ℕ × ℕ → Finset (Fin (V.k x))
  session : Msg → ℕ × ℕ → Seen → Option (Msg × Seen)
  reveal : AuditStatement → ℕ × ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)
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
  reads : AuditStatement → Msg → List ℕ
  wit : AuditStatement → Msg → List (Opening Bool) → Claims.Flock.Wit

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
  | some s => ((c.partition s).getD ⟨[]⟩).sizes.sum
  | none => 0

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

/-- **The deployed sampling record** at the relation's fields `rel` and the verifier's code `c`, its sessions C-Flock's
masked proofs. -/
noncomputable def deploy (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  deployOf (FlockDeploy.deployZk rel) c

/-- **The deployed plain sampling record**: its sessions C-Flock's plain proofs, at the trimmed record Flock's plain KS
is stated at. -/
noncomputable def deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  deployOf (FlockDeploy.deployPlainTrim rel) c

section Of

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)

theorem deployOf_εtable_nonneg (x : Msg) : 0 ≤ (deployOf F c).εtable x := by
  show 0 ≤ (c.tables x : ℝ) * 2 ^ (-(205 : ℝ))
  positivity

theorem deployOf_commit :
    (deployOf F c).commit = pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath c.own) c.layer :=
  rfl

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

/-- **The inner numbers are Flock's** where `F`'s waits and truncation are the deployment's and every session's `N₀`
is the record's, its `ε_table` and weight at most some `n ≤ tables` tables' bounds. -/
theorem innerNumbers_deployOf (hw : ∀ xS, F.waits xS = Claims.Flock.Numbers.waits)
    (hc : ∀ xS, F.cap xS = Claims.Flock.Numbers.cap)
    (h : ∀ x u v xS vS, c.session x u v = some (xS, vS) → F.N₀ xS = c.innerN₀ x ∧
      ∃ n : ℕ, n ≤ c.tables x ∧ F.εtable xS ≤ n * 2 ^ (-(205 : ℝ)) ∧ F.weight xS ≤ n * 2 ^ 33) :
    InnerNumbers (deployOf F c) := by
  intro x u v xS vS hs
  obtain ⟨hN, n, hn, he, hwt⟩ := h x u v xS vS hs
  have hn' : (n : ℝ) ≤ c.tables x := by exact_mod_cast hn
  exact ⟨hw xS, hc xS, hN, he.trans (mul_le_mul_of_nonneg_right hn' (by positivity)),
    hwt.trans (mul_le_mul_of_nonneg_right hn' (by positivity))⟩

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record over any Flock record**, at every hash: `sampling_assembled_of` at
`deployOf F c` with the record's facts discharged. The hypotheses are what remains: Flock's KS at `F` (`hKS`), its
decoder's `ListWithin` and step count at `p` (`hlist`, `hcount`); the draw, sessions and their numbers, slots, session
openings, interior roots and scope checks of the sampled verifier's run (`DrawFixed`, `SessionRuns`, `InnerNumbers`,
`DrawSlots`, `SessionOpens`, `InteriorFixed`, `ProgramScope`, `ChecksScope`); its reveal count (`RevealsLe`) and
completeness. -/
theorem sampling_deployed_of (p : Polynomial ℕ) (H : Msg → Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode)
    (hcount : ∀ os, (F.dec os).2 ≤ p.eval (Claims.Flock.wordLen os))
    (hfixed : DrawFixed (deployOf F c)) (hsessions : SessionRuns H (deployOf F c))
    (hnum : InnerNumbers (deployOf F c)) (hslots : DrawSlots (deployOf F c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deployOf F c) iroots) (hfix : InteriorFixed H (deployOf F c) iroots)
    (hprog : ProgramScope (deployOf F c)) (hchecks : ChecksScope H (deployOf F c)) (hreveals : RevealsLe c)
    (hComplete : Complete H (deployOf F c)) :
    ScopeMatches (deployOf F c) ∧ Binding (deployOf F c) ∧ Universal (deployOf F c) ∧ FaultyBoundary ∧
      KS H (deployOf F c) ∧ KSCount H (deployOf F c) ∧ Bits H (deployOf F c) ∧ Efficient H (deployOf F c) p ∧
      Complete H (deployOf F c) :=
  sampling_assembled_of (deployOf F c) H F p rfl hKS hlist hcount (deployOf_εtable_nonneg F c) c.own c.at_
    (row c.place c.bits) c.salt c.π c.layer (deployOf_commit F c) (row_places c.place c.bits) hfixed hsessions hnum
    hslots iroots hopens hfix hprog hchecks (openCallsLe_deployOf F c H hreveals) (universalIs_deployOf F c) hComplete

end Of

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)

/-! ## At the masked record -/

theorem deploy_flock : (deploy rel c).flock = FlockDeploy.deployZk rel := rfl

theorem deploy_εtable_nonneg (x : Msg) : 0 ≤ (deploy rel c).εtable x :=
  deployOf_εtable_nonneg (FlockDeploy.deployZk rel) c x

theorem deploy_commit :
    (deploy rel c).commit = pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath c.own) c.layer :=
  rfl

theorem universalIs_deploy : UniversalIs (deploy rel c) := universalIs_deployOf (FlockDeploy.deployZk rel) c

theorem openCallsLe_deploy (H : Msg → Msg) (h : RevealsLe c) : OpenCallsLe H (deploy rel c) :=
  openCallsLe_deployOf (FlockDeploy.deployZk rel) c H h

/-- **The sessions are in Flock's masked scope**: every session's statement sets up at most `tables` tables, each at
an `m` and of a shape the masked verifier serves (`FlockDeploy.ZkIn`), and its `N₀` is the record's. -/
def SessionsIn : Prop :=
  ∀ x u v xS vS, c.session x u v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ true xS = c.innerN₀ x

/-- **The inner numbers are Flock's**: the waits and truncation by definition; at a session in Flock's masked scope,
its `ε_table` and weight are at most its tables' count times their bounds, so at most the record's. -/
theorem innerNumbers_deploy (h : SessionsIn c) : InnerNumbers (deploy rel c) :=
  innerNumbers_deployOf (FlockDeploy.deployZk rel) c (fun _ => rfl) (fun _ => rfl) fun x u v xS vS hs => by
    obtain ⟨⟨s, spec, sts, hz, hn⟩, hN⟩ := h x u v xS vS hs
    refine ⟨hN, sts.size, hn, FlockDeploy.εtable_le hz.setup fun st hst => ?_, FlockDeploy.weight_zk_le rel hz⟩
    have := hz.tab st hst
    exact FlockDeploy.tableErr_zk_le spec st this.1 this.2.1 this.2.2

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed record**, at every hash: `sampling_deployed_of` at the masked record
`deployZk`, its decoder's facts `sampling_assembled`'s, and zero knowledge. The hypotheses are what remains: Flock's KS
at `deployZk` (`hKS`); the draw, sessions and their scope, slots, session openings, interior roots and scope checks of
the sampled verifier's run (`DrawFixed`, `SessionRuns`, `SessionsIn`, `DrawSlots`, `SessionOpens`, `InteriorFixed`,
`ProgramScope`, `ChecksScope`); its reveal count (`RevealsLe`); completeness (`complete_deploy` proves it from its own
facts) and zero knowledge. -/
theorem sampling_deployed (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hfixed : DrawFixed (deploy rel c)) (hsessions : SessionRuns H (deploy rel c)) (hnum : SessionsIn c)
    (hslots : DrawSlots (deploy rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deploy rel c) iroots) (hfix : InteriorFixed H (deploy rel c) iroots)
    (hprog : ProgramScope (deploy rel c)) (hchecks : ChecksScope H (deploy rel c)) (hreveals : RevealsLe c)
    (hComplete : Complete H (deploy rel c))
    {Nz Ωz : Type} [Fintype Nz] [Nonempty Nz] [Fintype Ωz] [Nonempty Ωz] (tapeOf : Nz × Ωz → Reference.Tape)
    (S : SimLeak) (εzk γzk : Msg → ℝ) (hZK : ZK H ((deploy rel c).mask Nz Ωz tapeOf S) εzk γzk) :
    ScopeMatches (deploy rel c) ∧ Binding (deploy rel c) ∧ Universal (deploy rel c) ∧ FaultyBoundary ∧
      KS H (deploy rel c) ∧ KSCount H (deploy rel c) ∧ Bits H (deploy rel c) ∧
      Efficient H (deploy rel c) FlockDeploy.zkStepPoly ∧ Complete H (deploy rel c) ∧
      ZK H ((deploy rel c).mask Nz Ωz tapeOf S) εzk γzk := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := sampling_deployed_of (FlockDeploy.deployZk rel) c
    FlockDeploy.zkStepPoly H hKS (listWithin_deployZk rel) (FlockDeploy.efficient_zk H rel).2.2.1 hfixed hsessions
    (innerNumbers_deploy rel c hnum) hslots iroots hopens hfix hprog hchecks hreveals hComplete
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, hZK⟩

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

set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the deployed plain record**, at every hash: `sampling_deployed_of` at the trimmed plain
record `deployPlainTrim`, its decoder's facts `sampling_assembled_plain`'s, with no zero knowledge. The hypotheses are
`sampling_deployed`'s, with Flock's plain KS (`hKS`) and its plain scope (`SessionsInPlain`). -/
theorem sampling_deployed_plain (H : Msg → Msg) (hKS : Claims.Flock.KS H (FlockDeploy.deployPlainTrim rel))
    (hfixed : DrawFixed (deployPlain rel c)) (hsessions : SessionRuns H (deployPlain rel c))
    (hnum : SessionsInPlain c) (hslots : DrawSlots (deployPlain rel c)) (iroots : Msg → Seen → ℕ → List Msg)
    (hopens : SessionOpens H (deployPlain rel c) iroots) (hfix : InteriorFixed H (deployPlain rel c) iroots)
    (hprog : ProgramScope (deployPlain rel c)) (hchecks : ChecksScope H (deployPlain rel c))
    (hreveals : RevealsLe c) (hComplete : Complete H (deployPlain rel c)) :
    ScopeMatches (deployPlain rel c) ∧ Binding (deployPlain rel c) ∧ Universal (deployPlain rel c) ∧
      FaultyBoundary ∧ KS H (deployPlain rel c) ∧ KSCount H (deployPlain rel c) ∧ Bits H (deployPlain rel c) ∧
      Efficient H (deployPlain rel c) FlockDeploy.plainStepPoly ∧ Complete H (deployPlain rel c) :=
  sampling_deployed_of (FlockDeploy.deployPlainTrim rel) c FlockDeploy.plainStepPoly H hKS
    (listWithin_deployPlainTrim rel) (FlockDeploy.efficient_plain H rel).2.2.1 hfixed hsessions
    (innerNumbers_deployPlain rel c hnum) hslots iroots hopens hfix hprog hchecks hreveals hComplete

end SamplingDeploy
