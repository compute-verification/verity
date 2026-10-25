import Proofs.Flock.Claims.KS.ConflictTied
import Proofs.Flock.Claims.KS.Trace
import Proofs.Flock.Claims.Scope.Deployed

/-!
# `TableBoundθ` statement by statement

`TableBoundθ` (`KS.Joint`) is a bound at each statement (`TableBoundθAt`). Where the verifier accepts no run and the
weight covers the fork event's `k`, it holds with no escapes (`tableBoundθAt_of_noAcc`, `Escapes.nil`). At the deployed
records the weight covers `k` at every statement (`k_le_weight`: each table's schedule in the weight has the
verifier's level 0, `kFits_all`), and a statement some run accepts at SHA-512 parses in scope and sets up
(`acc_inScope`: the checks after the verdict run the scope check, `checks_scope`). So at both records `TableBoundθ` is
its bound at the accepted statements (`plain_tableBoundθ_of_acc`, `zk_tableBoundθ_of_acc`). KS at SHA-512 rests on
that alone at the plain record, whose scope has one table (`plain_KS_of_acc`, `plain_hone`), and on that and one table
at every accepted statement at the masked record (`zk_KS_of_acc`).

At a statement that sets up, the old theorems bound a table of table 0's level-0 code that is not `Committed` to table
0's model statement (`Layout.model` at `regionsOf`). At the deployed relation, when the setup has one table, a message
whose witness satisfies that statement reads as a witness of the statement (`relS_of_table0`, from `relS_deployed`): the
plain scope has one table (`plain_one`), so `not_committed_plain`'s `hwit` holds there (`plain_hwit`), and a table of
full-width columns that does not extract is not committed (`plain_not_committed`). Under `--zk` it holds at one table
(`zk_hwit`), where a table of witness-lane columns that does not extract is not `CommittedPad` (`zk_not_committed`);
the masked scope admits several, where a trace of table 0 alone does not read as a witness of the others.
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
def TableBoundθAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable D N x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p

theorem tableBoundθ_iff {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) :
    TableBoundθ H D N Ok ↔ ∀ x, TableBoundθAt H D N Ok x :=
  Iff.rfl

/-- **Where the verifier accepts no run**, `TableBoundθ` holds with no escapes, when the weight covers `k` and
`ε_table ≥ 0`. -/
theorem tableBoundθAt_of_noAcc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) (x : Msg)
    (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x) (hε : 0 ≤ D.εtable x) :
    TableBoundθAt H D N Ok x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, Escapes.fitsFork_nil H D.V P x _ _ hk, fun ω C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p
  linarith

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-! ## `k ≤ weight` -/

/-- The weight's schedule at `m`: `fast100`'s, under `--zk` M1's (`tableWeight`). -/
def wSched (zk : Bool) (m : ℕ) : FlockSoundness.Accounting.Schedule :=
  if zk then FlockSoundness.Accounting.m1Schedule m else (FlockSoundness.Accounting.fast100 m).getD default

/-- The verifier's level 0 at `m` (`N₀`'s). -/
def level0 (zk : Bool) (m : ℕ) : Flock.Level :=
  if zk then Claims.Flock.zkLevel0 m else Claims.Flock.plainLevel0 m

/-- Level 0's codeword is no longer than the weight schedule's level 0. -/
def kFits (zk : Bool) (m : ℕ) : Bool :=
  decide ((level0 zk m).d ≤ ((wSched zk m).levels.headD default).logLen)

theorem kFits_lt40 : ∀ m : Fin 40, kFits false m.val = true ∧ kFits true m.val = true := by
  decide +kernel

theorem level0_d_of_gt {zk : Bool} {m : ℕ} (hm : 35 < m) : (level0 zk m).d = 0 := by
  cases zk
  · simp only [level0, Bool.false_eq_true, ↓reduceIte, Claims.Flock.plainLevel0, Claims.Flock.plainSched,
      FlockSoundness.Refine.exec_fast100_none m hm, Option.getD_none]
    rfl
  · cases h : Zk.schedule m with
    | ok p => have := schedule_range h; omega
    | error e =>
      simp only [level0, ↓reduceIte, Claims.Flock.zkLevel0, Claims.Flock.zkSched, h]
      rfl

theorem kFits_all (zk : Bool) (m : ℕ) : (level0 zk m).d ≤ ((wSched zk m).levels.headD default).logLen := by
  by_cases hm : m < 40
  · have := kFits_lt40 ⟨m, hm⟩
    cases zk
    · exact of_decide_eq_true this.1
    · exact of_decide_eq_true this.2
  · rw [level0_d_of_gt (by omega)]
    exact Nat.zero_le _

theorem tableWeight_ge (zk : Bool) (st : Stmt) :
    2 + 4 * 2 ^ ((wSched zk st.m).levels.headD default).logLen / Real.exp 1 ≤ tableWeight zk st := by
  have h2 : ∀ sch N, (2 : ℝ) ≤ Claims.Flock.Numbers.weightNat sch N := fun sch N => by
    unfold Claims.Flock.Numbers.weightNat
    exact_mod_cast Nat.le_add_left 2 _
  cases zk
  · simp only [tableWeight, wSched, Bool.false_eq_true, ↓reduceIte, Claims.Flock.Numbers.weightOf]
    linarith [h2 ((FlockSoundness.Accounting.fast100 st.m).getD default) 0]
  · simp only [tableWeight, wSched, ↓reduceIte, Claims.Flock.Numbers.weightOf]
    linarith [h2 (FlockSoundness.Accounting.m1Schedule st.m)
      (Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))]

theorem tableWeight_nonneg (zk : Bool) (st : Stmt) : 0 ≤ tableWeight zk st := by
  unfold tableWeight
  split <;> exact Claims.Flock.Numbers.weightOf_nonneg _ _

namespace Gen

variable [S : ZkSetup]

theorem weight_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ weight zk x := by
  unfold weight
  refine List.sum_nonneg fun a ha => ?_
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
  exact tableWeight_nonneg zk p.2

end Gen

theorem weight_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ weight zk x :=
  Gen.weight_nonneg (S := S₀) (zk := zk) (x := x)

/-- **A table's weight covers its length's share of `k`**: `2 + 4·2^{d₀}/e`, its schedule's level 0 as long as the
verifier's (`kFits_all`). -/
theorem tableLen_le_weight (zk : Bool) (st : Stmt) : 4 * tableLen zk st / Real.exp 1 + 2 ≤ tableWeight zk st := by
  have hw := tableWeight_ge zk st
  have hpow : tableLen zk st ≤ 2 ^ ((wSched zk st.m).levels.headD default).logLen :=
    pow_le_pow_right₀ (by norm_num) (kFits_all zk st.m)
  have hdiv : 4 * tableLen zk st / Real.exp 1 ≤ 4 * 2 ^ ((wSched zk st.m).levels.headD default).logLen /
      Real.exp 1 := by
    gcongr
  linarith

theorem tableLen_nonneg (zk : Bool) (st : Stmt) : 0 ≤ tableLen zk st := by
  unfold tableLen
  positivity

/-- **Tables' weights cover their lengths' share of `k`**, and `2` more each (`tableLen_le_weight`). -/
theorem lens_le_weights (zk : Bool) (l : List (SessionSpec × Stmt)) :
    4 * (l.map fun p => tableLen zk p.2).sum / Real.exp 1 + 2 * l.length ≤ (l.map fun p => tableWeight zk p.2).sum ∧
      0 ≤ (l.map fun p => tableLen zk p.2).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    have h1 := tableLen_le_weight zk a.2
    have h2 := tableLen_nonneg zk a.2
    have e : 4 * (tableLen zk a.2 + (l.map fun p => tableLen zk p.2).sum) / Real.exp 1 =
        4 * tableLen zk a.2 / Real.exp 1 + 4 * (l.map fun p => tableLen zk p.2).sum / Real.exp 1 := by ring
    rw [e]
    constructor <;> linarith [ih.1, ih.2]

namespace Gen

variable [S : ZkSetup]

theorem N₀_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ N₀ zk x := (lens_le_weights zk (tables zk x)).2

end Gen

theorem N₀_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ N₀ zk x :=
  Gen.N₀_nonneg (S := S₀) (zk := zk) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **`k ≤ weight`** at every statement: `k = ⌈4N₀/e⌉ < 4N₀/e + 1`, and each table's weight holds its length's share
and `2` more (`lens_le_weights`). -/
theorem k_le_weight (zk : Bool) (x : Claims.Msg) : (⌈4 * N₀ zk x / Real.exp 1⌉₊ : ℝ) ≤ weight zk x := by
  have he := Real.exp_pos 1
  unfold N₀ weight
  cases ht : tables zk x with
  | nil => simp
  | cons p rest =>
    obtain ⟨h1, h2⟩ := lens_le_weights zk (p :: rest)
    have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) h2) he.le)
    simp only [List.length_cons, Nat.cast_add, Nat.cast_one] at h1
    have : (0 : ℝ) ≤ rest.length := Nat.cast_nonneg _
    linarith

end Gen

/-- **`k ≤ weight`** at every statement: `k = ⌈4N₀/e⌉ < 4N₀/e + 1`, and each table's weight holds its length's share
and `2` more (`lens_le_weights`). -/
theorem k_le_weight (zk : Bool) (x : Claims.Msg) : (⌈4 * N₀ zk x / Real.exp 1⌉₊ : ℝ) ≤ weight zk x :=
  Gen.k_le_weight (S := S₀) (zk := zk) (x := x)

theorem plain_k_le (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    ((deployPlain rel).k x : ℝ) ≤ 2 * (deployPlain rel).weight x := by
  have := k_le_weight false x
  have := weight_nonneg false x
  show (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ) ≤ 2 * weight false x
  linarith

namespace Gen

variable [S : ZkSetup]

theorem zk_k_le (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    ((deployZk rel).k x : ℝ) ≤ 2 * (deployZk rel).weight x := by
  have := k_le_weight true x
  have := weight_nonneg true x
  show (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) ≤ 2 * weight true x
  linarith

end Gen

theorem zk_k_le (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    ((deployZk rel).k x : ℝ) ≤ 2 * (deployZk rel).weight x :=
  Gen.zk_k_le (S := S₀) (rel := rel) (x := x)

/-! ## Acceptance in scope -/

/-- **The checks after the verdict include the scope check** on what the verifier read. -/
theorem checks_scope {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : s.read.check = .ok () := by
  unfold checks at h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, h5, -⟩ := FlockSoundness.Refine.ebind_ok h
  exact h5

namespace Gen

variable [S : ZkSetup]

/-- **The checks after the verdict include the scope check** on what the verifier read. -/
theorem checks_scope {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : s.read.check = .ok () :=
  _root_.FlockDeploy.KS.checks_scope (checks_base h)

/-- **An accepting step at SHA-512** is an accepting verdict at the statement's setup, after which the scope check
passed. -/
theorem next_acc_check {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((next zk x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts record proofs, setupOf zk x = some (ins, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () ∧
      ins.read.check = .ok () := by
  cases ha : act zk x as cs with
  | rej => simp [next_rej sha512 ha] at h
  | coin c => simp [next_coin sha512 ha] at h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 h
    exact ⟨s, spec, sts, f.record, f.proofs, act_decide ha, hv, checks_scope hc⟩

end Gen

/-- **An accepting step at SHA-512** is an accepting verdict at the statement's setup, after which the scope check
passed. -/
theorem next_acc_check {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((next zk x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts record proofs, setupOf zk x = some (ins, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () ∧
      ins.read.check = .ok () :=
  Gen.next_acc_check (S := S₀) (zk := zk) (x := x) (as := as) (cs := cs) (h := h)

namespace Gen

variable [S : ZkSetup]

theorem acc_check {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) :
    ∃ ins spec sts record proofs, setupOf zk x = some (ins, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () ∧
      ins.read.check = .ok () := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact next_acc_check has

end Gen

theorem acc_check {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) :
    ∃ ins spec sts record proofs, setupOf zk x = some (ins, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () ∧
      ins.read.check = .ok () :=
  Gen.acc_check (S := S₀) (zk := zk) (x := x) (hacc := hacc)

namespace Gen

variable [S : ZkSetup]

/-- **The parser reads every statement the verifier sets up**: under `--zk` the setup passes every table's
`Zk.shapeOk` (`setup_shapeOk`). -/
theorem read_parse_of_setup {zk : Bool} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : S.Ok zk ins spec sts) :
    ins.read.parse = some { ins.read with sts } := by
  unfold Readable.Read.parse Readable.Read.tablesOf
  rw [Ins.setup_read hs]
  cases hz : ins.read.zk
  · simp only [Bool.false_eq_true, ↓reduceIte, Option.map_some]
  · simp only [↓reduceIte, Claims.Bridge.Custody.setup_shapeOk hs hz, Option.map_some]

end Gen

/-- **The parser reads every statement the verifier sets up**: under `--zk` the setup passes every table's
`Zk.shapeOk` (`setup_shapeOk`). -/
theorem read_parse_of_setup {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : ins.setup = .ok (spec, sts)) :
    ins.read.parse = some { ins.read with sts } :=
  Gen.read_parse_of_setup (S := S₀) (zk := ins.read.zk) (ins := ins) (spec := spec) (sts := sts) (hs := hs)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed verifier accepts only in scope**, at SHA-512: a statement some run accepts parses, in scope. The
checks after the verdict run the scope check (`checks_scope`), and the shipped `verify` serves only `m ≥ 22`
(`served_plain`, `served_zk`). -/
theorem acc_inScope (rel : Readable.Parsed → Readable.Relation) {zk : Bool} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) : ∃ s, parse zk rel x = some s ∧ Claims.Flock.InScope zk s := by
  obtain ⟨ins, spec, sts, record, proofs, hset, hv, hc⟩ := acc_check hacc
  obtain ⟨hi, hs⟩ := setupOf_some hset
  have hr : readOf zk x = some ins.read := by unfold readOf; rw [hi]; rfl
  have hzk : ins.read.zk = zk := readOf_zk hr
  have hrp := read_parse_of_setup hs
  have hparse : parse zk rel x = some (Readable.toReadable rel { ins.read with sts }) := by
    unfold parse Readable.parseOf Readable.parseBytes
    rw [hr, Option.bind_some, hrp, Option.map_some]
  refine ⟨_, hparse, ?_⟩
  have hserved : 0 < sts.size ∧ 22 ≤ sts[0]!.m := by
    cases zk
    · obtain ⟨h0, hm⟩ := served_plain (by simpa using hv)
      exact ⟨h0, by rw [getElem!_pos sts 0 h0]; exact (hm 0 h0).1⟩
    · obtain ⟨h0, hm⟩ := served_zk (by simpa using hv)
      exact ⟨h0, by rw [getElem!_pos sts 0 h0]; have := (hm 0 h0).1; omega⟩
  have hin := (Readable.inScope_iff rel (Readable.parses_of_parse hrp)).2 ⟨hc, ?_⟩
  · exact hzk ▸ hin
  · obtain ⟨h0, h22⟩ := hserved
    rw [getElem!_pos sts 0 h0] at h22
    simpa only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some] using h22

end Gen

/-- **The deployed verifier accepts only in scope**, at SHA-512: a statement some run accepts parses, in scope. The
checks after the verdict run the scope check (`checks_scope`), and the shipped `verify` serves only `m ≥ 22`
(`served_plain`, `served_zk`). -/
theorem acc_inScope (rel : Readable.Parsed → Readable.Relation) {zk : Bool} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) : ∃ s, parse zk rel x = some s ∧ Claims.Flock.InScope zk s :=
  Gen.acc_inScope (S := S₀) (rel := rel) (zk := zk) (x := x) (hacc := hacc)

/-- **The plain verifier accepts only at one table**, at SHA-512: the plain scope admits one (`acc_inScope`). -/
theorem plain_hone (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨ Claims.Flock.NoAcc sha512 (deployPlain rel).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V false) x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := acc_inScope rel hno
    obtain ⟨-, -, -, -, -, -, -, -, ht, -⟩ := hin
    exact .inl ⟨s, hs, (Claims.Flock.Stmt.tables_eq_one_iff s).2 (by simpa using ht)⟩

/-! ## Down to the accepted statements -/

/-- **The plain record: `TableBoundθ` from its bound at the statements some run accepts**, which parse in scope and
set up: elsewhere the verifier accepts no run and `k ≤ 2·weight` (`plain_k_le`). -/
theorem plain_tableBoundθ_of_acc (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V false) x → parse false rel x = some s →
      Claims.Flock.InScope false s → setupOf false x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) Ok x) :
    Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) Ok := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  by_cases hno : Claims.Flock.NoAcc sha512 (V false) x
  · exact Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x hno (plain_k_le rel x) (εtable_nonneg false x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope rel hno
    obtain ⟨ins, spec, sts, -, -, hset, -, -⟩ := acc_check hno
    exact h x s ins spec sts hno hs hin hset

/-- **The masked (`--zk`) record: `TableBoundθ` from its bound at the statements some run accepts**, which parse in
scope and set up: elsewhere the verifier accepts no run and `k ≤ 2·weight` (`zk_k_le`). -/
theorem zk_tableBoundθ_of_acc (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V true) x → parse true rel x = some s →
      Claims.Flock.InScope true s → setupOf true x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) Ok x) :
    Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) Ok := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x hno (zk_k_le rel x) (εtable_nonneg true x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope rel hno
    obtain ⟨ins, spec, sts, -, -, hset, -, -⟩ := acc_check hno
    exact h x s ins spec sts hno hs hin hset

/-- **The deployed plain verifier: KS**, at SHA-512, from a fixed table's bound at the statements some run accepts. -/
theorem plain_KS_of_acc (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V false) x → parse false rel x = some s →
      Claims.Flock.InScope false s → setupOf false x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false)) x) :
    Claims.Flock.KS sha512 (deployPlain rel) :=
  plain_KS_of_table rel (plain_hone rel) (plain_tableBoundθ_of_acc rel _ h)

/-- **The deployed masked (`--zk`) verifier: KS**, at SHA-512, from a fixed table's bound at the statements some run
accepts. -/
theorem zk_KS_of_acc (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (h : ∀ x s ins spec sts, ¬ Claims.Flock.NoAcc sha512 (V true) x → parse true rel x = some s →
      Claims.Flock.InScope true s → setupOf true x = some (ins, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true)) x) :
    Claims.Flock.KS sha512 (deployZk rel) :=
  zk_KS_of_table rel hone (zk_tableBoundθ_of_acc rel _ h)

/-! ## `hwit` at one table -/

theorem toZ_decide : ∀ z : ZMod 2, Readable.toZ (decide (z = 1)) = z := by decide

/-- **At one table, table 0's share of a read trace is the trace's witness.** -/
theorem zOf_readTrace {J : ℕ} (hJ : J = 1) (m lanes cols : ℕ) (M : FlockSoundness.Model.Message 𝔽 lanes cols) :
    Readable.zOf J (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M) 0 m =
      FlockSoundness.Model.witnessOf FlockSoundness.execArith m lanes cols M := by
  subst hJ
  funext a
  unfold Readable.zOf
  rw [mul_one, add_zero, Claims.Flock.readTrace_hid]
  exact toZ_decide _

namespace Gen

variable [S : ZkSetup]

/-- **The parser reads the setup's statement**: its tables are the setup's, read off the setup's `Read`. -/
theorem parseBytes_setup {zk : Bool} {x : Claims.Msg} {p : Readable.Parsed} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hp : Readable.parseBytes (readOf zk) x = some p) (hset : setupOf zk x = some (ins, spec, sts)) :
    p.sts = sts ∧ p.toRead = ins.read ∧ ins.read.setup = .ok (spec, sts) := by
  have hins := setupOf_some hset
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

end Gen

/-- **The parser reads the setup's statement**: its tables are the setup's, read off the setup's `Read`. -/
theorem parseBytes_setup {zk : Bool} {x : Claims.Msg} {p : Readable.Parsed} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hp : Readable.parseBytes (readOf zk) x = some p) (hset : setupOf zk x = some (ins, spec, sts)) :
    p.sts = sts ∧ p.toRead = ins.read ∧ ins.read.setup = .ok (spec, sts) :=
  Gen.parseBytes_setup (S := S₀) (zk := zk) (x := x) (p := p) (ins := ins) (spec := spec) (sts := sts) (hp := hp)
    (hset := hset)

/-- **At one table, a message satisfying table 0's model statement reads as a witness**: at a statement in scope that
the deployed parser reads at the deployed relation (`relS_deployed`), when the setup has one table, a message whose
witness satisfies its model statement (`Layout.model` at `regionsOf`) reads as a witness of the statement. -/
theorem relS_of_table0 {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) (hin : Claims.Flock.InScope zk s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts)) (hone : sts.size = 1)
    (L : FlockSoundness.Layout (sts[0]'(by omega))) {m lanes cols : ℕ} (hm : m = (sts[0]'(by omega)).m)
    (M : FlockSoundness.Model.Message 𝔽 lanes cols)
    (hsat : (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).Satisfies
      (FlockSoundness.Model.witnessOf FlockSoundness.execArith
        (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).m lanes cols M)) :
    Claims.Flock.RelS s (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M) := by
  obtain ⟨p, hp, rfl, hiff⟩ :=
    Readable.relS_deployed hs hin (Claims.Flock.readTrace FlockSoundness.execArith m lanes cols M)
  obtain ⟨hps, -, -⟩ := parseBytes_setup hp hset
  subst hps
  refine hiff.2 fun c hc => ?_
  have hc0 : c = 0 := by omega
  subst hc0
  refine ⟨L, ?_⟩
  subst hm
  rw [zOf_readTrace hone]
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
theorem plain_hwit {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployPlain Readable.relation).parse x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : FlockSoundness.Layout sts[0]) (i : Claims.Flock.PlainIdx) (hi : i.m = sts[0].m) :
    ∀ M : FlockSoundness.Model.Message 𝔽 (Claims.Flock.plainLanes i) (Claims.Flock.plainCols i),
      (L.model (FlockSoundness.Refine.regionsOf sts[0])).Satisfies
        (FlockSoundness.Model.witnessOf FlockSoundness.execArith (L.model (FlockSoundness.Refine.regionsOf sts[0])).m
          (Claims.Flock.plainLanes i) (Claims.Flock.plainCols i) M) →
      Claims.Flock.RelS s (Claims.Flock.plainReadD ⟨i, M⟩) :=
  fun M hsat => relS_of_table0 hs hin hset (plain_one hs hin hset) L hi M hsat

/-- **`hwit` at the masked record at one table** (`not_committed_zk`'s): at a statement in scope that the deployed
masked parser reads at the deployed relation, set up with one table, at index `i`, every message whose witness
satisfies table 0's model statement reads as a witness of the statement. The masked scope admits several tables, where
a trace of table 0 alone does not read as a witness of the others. -/
theorem zk_hwit {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployZk Readable.relation).parse x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (hone : sts.size = 1)
    (L : FlockSoundness.Layout (sts[0]'(by omega))) (i : Claims.Flock.ZkIdx) (hi : i.m = (sts[0]'(by omega)).m) :
    ∀ P : FlockSoundness.Model.Message 𝔽 (Claims.Flock.zkLanes i) (Claims.Flock.zkCols i),
      (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).Satisfies
        (FlockSoundness.Model.witnessOf FlockSoundness.execArith
          (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega)))).m (Claims.Flock.zkLanes i)
          (Claims.Flock.zkCols i) P) →
      Claims.Flock.RelS s
        (Claims.Flock.readTrace FlockSoundness.execArith i.m (Claims.Flock.zkLanes i) (Claims.Flock.zkCols i) P) :=
  fun P hsat => relS_of_table0 hs hin hset hone L hi P hsat

namespace Gen

variable [S : ZkSetup]

/-- The readable statement's `m` is table 0's. -/
theorem parse_m {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk Readable.relation x = some s)
    {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts))
    (h0 : 0 < sts.size) : s.m = sts[0].m := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
  subst hpst
  simp only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]

end Gen

/-- The readable statement's `m` is table 0's. -/
theorem parse_m {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk Readable.relation x = some s)
    {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf zk x = some (ins, spec, sts))
    (h0 : 0 < sts.size) : s.m = sts[0].m :=
  Gen.parse_m (S := S₀) (zk := zk) (x := x) (s := s) (hs := hs) (ins := ins) (spec := spec) (sts := sts) (hset := hset)
    (h0 := h0)

/-- **Table 0's statement is the parsed one** where the verifier sets up one table. -/
theorem table_zero_of_setup {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse zk Readable.relation x = some s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf zk x = some (ins, spec, sts)) (hone : sts.size = 1) :
    table zk Readable.relation x 0 = some s := by
  obtain ⟨p, hp, -, -, hsz, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
  exact table_zero_of_tables_eq_one hs (hsz.symm.trans (hpst ▸ hone))

/-- **At the plain record, a table that does not extract is not committed** to table 0's model statement, at the
old theorems' schedule (`not_committed_plain` at `plain_hwit`): at a statement in scope that the deployed plain
parser reads at the deployed relation and the verifier sets up, for every table of full-width columns. -/
theorem plain_not_committed {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployPlain Readable.relation).parse x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : FlockSoundness.Layout sts[0]) (C : Fin (wordLen false x) → List 𝔽) (hw : ∀ q, wide false x (C q))
    (hg : ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C) :
    ¬ FlockSoundness.Committed FlockSoundness.execArith (L.model (FlockSoundness.Refine.regionsOf sts[0]))
      (FlockSoundness.Refine.schedOf (Claims.Flock.plainSched sts[0].m)) (Claims.Flock.oracleOf C) := by
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ sts[0].m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : sts[0].m ≤ 35 := by have := hin.2.1; omega
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have := not_committed_plain Readable.relation x s (table_zero_of_setup hs hset (plain_one hs hin hset))
    (tables_head hset (Array.getElem?_eq_getElem h0)) ⟨sts[0].m - 22, by omega⟩ hi C hw
    (L.model (FlockSoundness.Refine.regionsOf sts[0])) (plain_hwit hs hin hset h0 L _ hi) hg
  rwa [hi] at this

/-- **At the masked record at one table, a table that does not extract is not committed** to table 0's model
statement, at the old padded theorems' schedule and padding (`not_committed_zk` at `zk_hwit`): at a statement in scope
that the deployed masked parser reads at the deployed relation and the verifier sets up with one table, for every table
of witness-lane columns. -/
theorem zk_not_committed {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployZk Readable.relation).parse x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (hone : sts.size = 1)
    (L : FlockSoundness.Layout (sts[0]'(by omega))) (C : Fin (wordLen true x) → List 𝔽)
    (hw : ∀ q, wide true x (C q)) (hg : ¬ Claims.Flock.GoodTable (deployZk Readable.relation) (wordLen true) x C) :
    ¬ FlockSoundness.CommittedPad FlockSoundness.execArith
      (L.model (FlockSoundness.Refine.regionsOf (sts[0]'(by omega))))
      (FlockSoundness.Refine.schedOf (Claims.Flock.zkSched (sts[0]'(by omega)).m).1)
      (Claims.Flock.zkSched (sts[0]'(by omega)).m).2 padC (Claims.Flock.oracleOf C) := by
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h25 : 25 ≤ sts[0].m := by have := hin.1; simp only [↓reduceIte] at this; omega
  have h35 : sts[0].m ≤ 35 := by have := hin.2.1; omega
  have hi : Claims.Flock.ZkIdx.m ⟨sts[0].m - 25, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.ZkIdx.m]
    omega
  have := not_committed_zk Readable.relation x s (table_zero_of_setup hs hset hone)
    (tables_head hset (Array.getElem?_eq_getElem h0)) ⟨sts[0].m - 25, by omega⟩ hi C hw
    (L.model (FlockSoundness.Refine.regionsOf sts[0])) (zk_hwit hs hin hset hone L _ hi) hg
  unfold Claims.Flock.zkPad at this
  rwa [hi] at this

end FlockDeploy.KS
