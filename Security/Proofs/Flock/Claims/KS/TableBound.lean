import Proofs.Flock.Claims.KS.ConflictTied
import Proofs.Flock.Claims.KS.Trace
import Proofs.Flock.Claims.Scope.Deployed

/-!
# `TableBoundθ` statement by statement

`TableBoundθ` (`KS.Joint`) is a bound at each statement (`TableBoundθAt`). Where the verifier accepts no run and the
weight covers the fork event's `k`, it holds with no escapes (`tableBoundθAt_of_noAcc`, `Escapes.nil`). A statement the
deployed verifier does not set up is one: it has no tables, so `ε_table`, `N₀`, `k` and the weight are `0` there
(`εtable_none`, `k_none_plain`, `k_none_zk`, `weight_none`), and an accepted run sets up (`acc_verdict`, `noAcc_of_none`). So at both
records `TableBoundθ` is its bound at the statements that set up (`plain_tableBoundθ_of_setup`,
`zk_tableBoundθ_of_setup`), and KS at SHA-512 rests on that alone (`plain_KSθ_of_setup`, `zk_KSθ_of_setup`).

At a statement that sets up, the old theorems bound a table of table 0's level-0 code that is not `Committed` to table
0's model statement (`Layout.model` at `regionsOf`). At the deployed relation, when the setup has one table, a message
whose witness satisfies that statement reads as a witness of the statement (`relS_of_table0`, from `relS_deployed`): the
plain scope has one table (`plain_one`), so `not_committed_plain`'s `hwit` holds there (`plain_hwit`), and a table of
full-width columns that does not extract is not committed (`plain_not_committed`). Under `--zk` it holds at one table
(`zk_hwit`); the masked scope admits several, where a trace of table 0 alone does not read as a witness of the others.
-/

namespace Claims.Flock

/-- **No escapes.** -/
def Escapes.nil {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) :
    Escapes H V P x where
  I := Empty
  c := Empty.elim
  j := Empty.elim
  ev := Empty.elim
  L := Empty
  β := Empty.elim
  ev₁ := Empty.elim
  c_nonneg i := i.elim
  β_nonneg l := l.elim
  tied i := i.elim
  tied₁ l := l.elim

theorem Escapes.term_nil {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) (ω : Ω) : (Escapes.nil H V P x).term ω = 0 := by
  unfold Escapes.term
  exact (congrArg₂ (· + ·) (Finset.sum_eq_zero fun i _ => Empty.elim i)
    (Finset.sum_eq_zero fun l _ => Empty.elim l)).trans (add_zero 0)

/-- **No escapes fit every weight that covers the fork event's.** -/
theorem Escapes.fitsFork_nil {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (C c : ℝ) (h : c ≤ 2 * C) : (Escapes.nil H V P x).FitsFork C c := by
  refine ⟨1, fun r => ?_, ?_, ?_⟩
  · have h0 : (Finset.univ.filter fun i : (Escapes.nil H V P x).I => (Escapes.nil H V P x).j i = r).card = 0 :=
      Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun i => i.elim)
    rw [h0]
    split <;> omega
  · have h0 : (∑ i : (Escapes.nil H V P x).I, (Escapes.nil H V P x).c i) = 0 :=
      Finset.sum_eq_zero fun i => i.elim
    rw [h0, Nat.cast_one, one_mul, add_zero]
    exact h
  · have h0 : (∑ l : (Escapes.nil H V P x).L, (Escapes.nil H V P x).β l) = 0 :=
      Finset.sum_eq_zero fun l => l.elim
    rw [h0]
    exact Nat.cast_nonneg _

/-- **`TableBoundθ` at one statement `x`.** -/
def TableBoundθAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (Rd.N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable H D Rd x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (Rd.N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p

theorem tableBoundθ_iff {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) :
    TableBoundθ H D Rd Ok ↔ ∀ x, TableBoundθAt H D Rd Ok x :=
  Iff.rfl

/-- **Where the verifier accepts no run**, `TableBoundθ` holds with no escapes, when the weight covers `k` and
`ε_table ≥ 0`. -/
theorem tableBoundθAt_of_noAcc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) (x : Msg)
    (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x) (hε : 0 ≤ D.εtable x) :
    TableBoundθAt H D Rd Ok x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, Escapes.fitsFork_nil H D.V P x _ _ hk, fun ω C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (Rd.N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p
  linarith

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **A statement the deployed verifier does not set up accepts no run**, at SHA-512. -/
theorem noAcc_of_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) :
    Claims.Flock.NoAcc sha512 (V zk) x := by
  by_contra hacc
  obtain ⟨s, spec, sts, -, -, hs, -⟩ := acc_verdict hacc
  rw [h] at hs
  cases hs

theorem tables_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : tables zk x = [] := by
  unfold tables
  rw [h]

theorem εtable_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : εtable zk x = 0 := by
  unfold εtable
  rw [tables_none h]
  rfl

theorem weight_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : weight zk x = 0 := by
  unfold weight
  rw [tables_none h]
  rfl

theorem N₀_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : N₀ zk x = 0 := by
  unfold N₀
  rw [tables_none h]
  rfl

theorem k_none_plain (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} (h : setupOf false x = none) :
    (deployPlain rel).k x = 0 := by
  simp only [Claims.Flock.Deploy.k, deployPlain, N₀_none h, mul_zero, zero_div, Nat.ceil_zero]

theorem k_none_zk (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} (h : setupOf true x = none) :
    (deployZk rel).k x = 0 := by
  simp only [Claims.Flock.Deploy.k, deployZk, N₀_none h, mul_zero, zero_div, Nat.ceil_zero]

/-- **The plain record: `TableBoundθ` from its bound at the statements that set up.** -/
theorem plain_tableBoundθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf false x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) plainRadius (wide false) x) :
    Claims.Flock.TableBoundθ sha512 (deployPlain rel) plainRadius (wide false) := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  cases hs : setupOf false x with
  | some v => exact h x v.1 v.2.1 v.2.2 hs
  | none =>
    refine Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x (noAcc_of_none hs) ?_ (εtable_nonneg false x)
    show ((deployPlain rel).k x : ℝ) ≤ 2 * weight false x
    rw [k_none_plain rel hs, weight_none hs]
    norm_num

/-- **The masked (`--zk`) record: `TableBoundθ` from its bound at the statements that set up.** -/
theorem zk_tableBoundθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf true x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) zkRadius (wide true) x) :
    Claims.Flock.TableBoundθ sha512 (deployZk rel) zkRadius (wide true) := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  cases hs : setupOf true x with
  | some v => exact h x v.1 v.2.1 v.2.2 hs
  | none =>
    refine Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x (noAcc_of_none hs) ?_ (εtable_nonneg true x)
    show ((deployZk rel).k x : ℝ) ≤ 2 * weight true x
    rw [k_none_zk rel hs, weight_none hs]
    norm_num

/-- **The deployed plain verifier: KS at the filled word**, at SHA-512, from a fixed table's bound at the statements
that set up. -/
theorem plain_KSθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf false x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) plainRadius (wide false) x) :
    Claims.Flock.KSθ sha512 (deployPlain rel) plainRadius :=
  plain_KSθ_of_table rel (plain_tableBoundθ_of_setup rel h)

/-- **The deployed masked (`--zk`) verifier: KS at the filled word**, at SHA-512, from a fixed table's bound at the
statements that set up. -/
theorem zk_KSθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf true x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) zkRadius (wide true) x) :
    Claims.Flock.KSθ sha512 (deployZk rel) zkRadius :=
  zk_KSθ_of_table rel (zk_tableBoundθ_of_setup rel h)

/-! ## `hwit` at one table -/

theorem toZ_decide : ∀ z : ZMod 2, Readable.toZ (decide (z = 1)) = z := by decide

/-- **Table 0's share of a read trace is the trace's witness.** -/
theorem zOf_readTrace (m lanes cols : ℕ) (M : FlockSoundness.Model.Message 𝔽 lanes cols) :
    Readable.zOf (2 ^ m) (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M) 0 m =
      FlockSoundness.Model.witnessOf FlockSoundness.execArith m lanes cols M := by
  funext a
  unfold Readable.zOf
  rw [zero_mul, zero_add, Claims.Flock.readTrace_hid]
  exact toZ_decide _

/-- **The parser reads the setup's statement**: its tables are the setup's, read off the setup's `Read`. -/
theorem parseBytes_setup {zk : Bool} {x : Claims.Msg} {p : Readable.Parsed} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hp : Readable.parseBytes (readOf zk) x = some p) (hset : setupOf zk x = some (ins, spec, sts)) :
    p.sts = sts ∧ p.toRead = ins.read ∧ ins.read.setup = .ok (spec, sts) := by
  have hins : insOf zk x = some ins ∧ ins.setup = .ok (spec, sts) := by
    unfold setupOf at hset
    cases hi : insOf zk x with
    | none => rw [hi] at hset; cases hset
    | some ins' =>
      rw [hi] at hset
      cases hs : ins'.setup with
      | error e => simp [hs, Except.toOption] at hset
      | ok v =>
        obtain ⟨spec', sts'⟩ := v
        simp only [hs, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
          Option.some.injEq, Prod.mk.injEq] at hset
        obtain ⟨rfl, rfl, rfl⟩ := hset
        exact ⟨rfl, hs⟩
  have hsr := Ins.setup_read hins.2
  have hr : readOf zk x = some ins.read := by unfold readOf; rw [hins.1]; rfl
  have hzk : ins.read.zk = zk := readOf_zk hr
  unfold Readable.parseBytes at hp
  rw [hr, Option.bind_some] at hp
  unfold Readable.Read.parse Readable.Read.tablesOf at hp
  rw [hsr, hzk] at hp
  cases zk
  · simp only [Bool.false_eq_true, ↓reduceIte, Option.map_some, Option.some.injEq] at hp
    subst hp
    exact ⟨rfl, rfl, hsr⟩
  · simp only [↓reduceIte] at hp
    split at hp
    · simp only [Option.map_some, Option.some.injEq] at hp
      subst hp
      exact ⟨rfl, rfl, hsr⟩
    · cases hp

/-- **At one table, a message satisfying table 0's model statement reads as a witness**: at a statement in scope that
the deployed parser reads at the deployed relation (`relS_deployed`), when the setup has one table, a message whose
witness satisfies its model statement (`Layout.model` at `regionsOf`) reads as a witness of the statement. -/
theorem relS_of_table0 (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts)) (hone : sts.size = 1)
    (L : FlockSoundness.Layout (sts[0]'(by omega))) {m lanes cols : ℕ} (hm : m = (sts[0]'(by omega)).m)
    (M : FlockSoundness.Model.Message 𝔽 lanes cols)
    (hsat : (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).Satisfies
      (FlockSoundness.Model.witnessOf FlockSoundness.execArith
        (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).m lanes cols M)) :
    Claims.Flock.RelS H s (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M) := by
  obtain ⟨p, hp, rfl, hiff⟩ :=
    Readable.relS_deployed H hs hin (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M)
  obtain ⟨hps, -, -⟩ := parseBytes_setup hp hset
  subst hps
  refine hiff.2 fun c hc => ?_
  have hc0 : c = 0 := by omega
  subst hc0
  refine ⟨L, ?_⟩
  have hm' : (Readable.toReadable Readable.relation p).m = p.sts[0].m := by
    simp only [Readable.toReadable, Array.getElem?_eq_getElem (show 0 < p.sts.size by omega), Option.map_some,
      Option.getD_some]
  rw [hm', hm, zOf_readTrace]
  exact hsat

/-- **The plain setup has one table** at a statement in scope. -/
theorem plain_one {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) : sts.size = 1 := by
  obtain ⟨p, hp, hps⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hps hpst
  obtain ⟨-, -, -, -, -, -, -, -, ht, hpi, -⟩ := hin
  have ht' : p.I.count ≤ 1 := by simpa [Readable.toReadable, Readable.Read.form] using ht
  have hpi' : p.publicInputs = none := by
    simp only [Readable.toReadable, Readable.Read.form, Option.isSome_iff_ne_none, ne_eq, Decidable.not_not] at hpi
    exact hpi
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hsr with ⟨-, st, -, -, hst⟩ | ⟨hc, -⟩
  · rw [hst]; rfl
  · omega

/-- **`hwit` at the plain record** (`not_committed_plain`'s): at a statement in scope that the deployed plain parser
reads at the deployed relation, with table 0 at index `i`, every message whose witness satisfies table 0's model
statement reads as a witness of the statement. -/
theorem plain_hwit (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployPlain Readable.relation).parse x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : FlockSoundness.Layout sts[0]) (i : Claims.Flock.PlainIdx) (hi : i.m = sts[0].m) :
    ∀ M : FlockSoundness.Model.Message 𝔽 (Claims.Flock.plainLanes i) (Claims.Flock.plainCols i),
      (L.model (FlockSoundness.Refine.regionsOf sts[0])).Satisfies
        (FlockSoundness.Model.witnessOf FlockSoundness.execArith (L.model (FlockSoundness.Refine.regionsOf sts[0])).m
          (Claims.Flock.plainLanes i) (Claims.Flock.plainCols i) M) →
      Claims.Flock.RelS H s (Claims.Flock.plainReadD ⟨i, M⟩) :=
  fun M hsat => relS_of_table0 H hs hin hset (plain_one hs hin hset) L hi M hsat

/-- **`hwit` at the masked record at one table** (`not_committed_zk`'s): at a statement in scope that the deployed
masked parser reads at the deployed relation, set up with one table, at index `i`, every message whose witness
satisfies table 0's model statement reads as a witness of the statement. The masked scope admits several tables, where
a trace of table 0 alone does not read as a witness of the others. -/
theorem zk_hwit (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployZk Readable.relation).parse x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (hone : sts.size = 1)
    (L : FlockSoundness.Layout (sts[0]'(by omega))) (i : Claims.Flock.ZkIdx) (hi : i.m = (sts[0]'(by omega)).m) :
    ∀ P : FlockSoundness.Model.Message 𝔽 (Claims.Flock.zkLanes i) (Claims.Flock.zkCols i),
      (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).Satisfies
        (FlockSoundness.Model.witnessOf FlockSoundness.execArith
          (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).m (Claims.Flock.zkLanes i)
          (Claims.Flock.zkCols i) P) →
      Claims.Flock.RelS H s
        (Claims.Flock.readTrace FlockSoundness.execArith i.m (Claims.Flock.zkLanes i) (Claims.Flock.zkCols i) P) :=
  fun P hsat => relS_of_table0 H hs hin hset hone L hi P hsat

/-- The readable statement's `m` is table 0's. -/
theorem parse_m {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk Readable.relation x = some s)
    {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts))
    (h0 : 0 < sts.size) : s.m = sts[0].m := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
  subst hpst
  simp only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]

/-- **At the plain record, a table that does not extract is not committed** to table 0's model statement, at the
old theorems' schedule (`not_committed_plain` at `plain_hwit`): at a statement in scope that the deployed plain
parser reads at the deployed relation and the verifier sets up, for every table of full-width columns. -/
theorem plain_not_committed (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployPlain Readable.relation).parse x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : FlockSoundness.Layout sts[0]) (C : Fin (plainRadius.N x) → List 𝔽) (hw : ∀ q, wide false x (C q))
    (hg : ¬ Claims.Flock.GoodTable H (deployPlain Readable.relation) plainRadius x C) :
    ¬ FlockSoundness.Committed FlockSoundness.execArith (L.model (FlockSoundness.Refine.regionsOf sts[0]))
      (FlockSoundness.Refine.schedOf (Claims.Flock.plainSched sts[0].m)) (Claims.Flock.oracleOf C) := by
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ sts[0].m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : sts[0].m ≤ 35 := by have := hin.2.1; omega
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have := not_committed_plain H Readable.relation x s hs
    (tables_head hset (Array.getElem?_eq_getElem h0)) ⟨sts[0].m - 22, by omega⟩ hi C hw
    (L.model (FlockSoundness.Refine.regionsOf sts[0])) (plain_hwit H hs hin hset h0 L _ hi) hg
  rwa [hi] at this

end FlockDeploy.KS
