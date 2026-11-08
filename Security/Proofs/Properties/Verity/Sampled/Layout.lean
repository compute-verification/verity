module

public import Proofs.Properties.Verity.Sampled.Decoders
public import Proofs.Properties.Verity.Sampled.DigestTie

/-!
# Sampled proofs' coupling: the session's layout of its rounds

The readings and the decoders read the session's rounds at a layout (`Slot`): the link's opening, and each round of
each table's two rep streams. At the session v2 sets up (`SetsUp`), `flock-verify --zk` sets up one table, the
statement's, for any record, so the session alone fixes the layout, before any play.

* **The session `--zk` sets up** (`Live.tables`, `SetsUp.spec_tables`, `SetsUp.setup_one`): for any record, its tables
  are the statement's one table, `Σ` is a SHA-512 digest, its rounds are retained, it has two link points, and it
  digests a round's bytes as `A.dg` does (`SetsUp.spec_dg`). The plan has one table at every sample and registration
  (`SetsUp.J_eq`).
* **The layout** (`slotAt`, `Live.layout`): round 0 is the link's opening, and round `2k + r + 1` is round `k` of the
  table's rep `r` stream. It is injective (`Live.layout_injective`), its round 0 is the link's (`Live.layout_link`),
  each of its rep rounds is of the one table (`Live.layout_rep`), below the plan's tables (`SetsUp.layout_lt`), and it
  has a slot for each of a rep stream's first `K` rounds wherever the session has at least `2K + 1` rounds
  (`Live.layout_covers`, `SetsUp.layout_covers`), so for each round a stream of the session's coin tree holds
  (`SetsUp.layout_covers_tree`).
* **The stream names** (`Live.tables_nodup`, `SetsUp.tables_at`, `SetsUp.tables_rep`): the session's tables are
  distinct, and the readings' name for table `t`'s rep `r` stream is the one the decoded record's stream has,
  `repStreamName spec.tables[0]! r`.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Discharge

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The session `--zk` sets up -/

/-- **`--zk`'s tags name the statement's table.** -/
theorem zkTags_table (I : Exec.Inputs) : (ZkExec.zkTags I).table = I.tags.table := by
  rw [ZkExec.zkTags_eq]
  rfl

/-- **The session's tables**: the statement's one table. -/
def Live.tables (A : Live) : Array String := #[A.I.tags.table]

theorem Live.tables_nodup (A : Live) : A.tables.toList.Nodup := by
  simp [Live.tables]

/-- **The session `--zk` sets up for any record has the session's tables**, wherever v2 sets the session up. -/
theorem SetsUp.spec_tables {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) : spec.tables = A.tables := by
  have hc : A.I.count = 1 := hA.count
  unfold Exec.setupOfP at hs
  rcases HiddenExec.setupTables_okHP ((ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := Hidden.setupHidden_specP hst
    show #[st.tags.table] = #[A.I.tags.table]
    rw [hP.tagsEq]
    exact congrArg (fun t => #[t]) (zkTags_table A.I)
  · exact absurd h1 (by show ¬ 1 < A.I.count; omega)

/-- **The session `--zk` sets up at `A`** for any record, wherever v2 sets `A` up: the session's one table, `Σ` a
SHA-512 digest, rounds retained, two link points, and its round digest `A.dg`. -/
theorem SetsUp.setup_one {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) :
    spec.tables = A.tables ∧ spec.sigma.size = 64 ∧ spec.retainRounds = true ∧ spec.points = 2 ∧ spec.dg = A.dg :=
  have h := Verity.Sampled.setup_one A.hTags hA.count hs
  ⟨hA.spec_tables hs, h.2.1, h.2.2.1, h.2.2.2, hA.spec_dg hs⟩

/-- **The plan has one table at every sample and registration**, wherever v2 sets `A` up. -/
theorem SetsUp.J_eq {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    (S : Finset (Fin A.n)) (R : A.Reg) : A.J S R = 1 := by
  show (ZkHidden.planZHJP (Reg := A.Reg) A.dj A.y₀ S R).length = 1
  rw [ZkHidden.planZHJP_length]
  unfold Composed.nTab
  rw [hA.count]
  rfl

/-! ## The layout -/

/-- **A round's slot at the session's layout**: round 0 is the link's opening, and round `2k + r + 1` is round `k` of
the one table's rep `r` stream. -/
def slotAt : ℕ → Slot
  | 0 => .link
  | i + 1 => .rep 0 ⟨i % 2, Nat.mod_lt i (by decide)⟩ (i / 2)

theorem slotAt_rep (k : ℕ) (r : Fin 2) : slotAt (2 * k + r.val + 1) = .rep 0 r k := by
  show Slot.rep 0 ⟨(2 * k + r.val) % 2, Nat.mod_lt _ (by decide)⟩ ((2 * k + r.val) / 2) = .rep 0 r k
  have hr := r.isLt
  rw [Slot.rep.injEq]
  exact ⟨rfl, Fin.ext (by show (2 * k + r.val) % 2 = r.val; omega), by omega⟩

theorem slotAt_eq_rep {i t k : ℕ} {r : Fin 2} (h : slotAt i = .rep t r k) : t = 0 ∧ i = 2 * k + r.val + 1 := by
  cases i with
  | zero => exact absurd h (by simp [slotAt])
  | succ i =>
    have h' : Slot.rep 0 ⟨i % 2, Nat.mod_lt i (by decide)⟩ (i / 2) = .rep t r k := h
    rw [Slot.rep.injEq] at h'
    obtain ⟨rfl, rfl, rfl⟩ := h'
    refine ⟨rfl, ?_⟩
    show i + 1 = 2 * (i / 2) + i % 2 + 1
    omega

theorem slotAt_injective : Function.Injective slotAt := by
  intro a b h
  cases a with
  | zero =>
    cases b with
    | zero => rfl
    | succ b => exact absurd h (by simp [slotAt])
  | succ a =>
    have ha : slotAt (a + 1) = .rep 0 ⟨a % 2, Nat.mod_lt a (by decide)⟩ (a / 2) := rfl
    obtain ⟨-, hb⟩ := slotAt_eq_rep (h.symm.trans ha)
    rw [hb]
    show a + 1 = 2 * (a / 2) + a % 2 + 1
    omega

/-- **The session's layout of its rounds** (`slotAt`). -/
def Live.layout (A : Live) : Fin A.rounds → Slot := fun j => slotAt j.val

theorem Live.layout_injective (A : Live) : Function.Injective A.layout :=
  fun _ _ h => Fin.ext (slotAt_injective h)

/-- **The layout's round 0 is the link's opening.** -/
theorem Live.layout_link (A : Live) (h : 0 < A.rounds) : A.layout ⟨0, h⟩ = .link := rfl

/-- **Each of the layout's rep rounds is of the one table**, round `k` of rep `r` at `2k + r + 1`. -/
theorem Live.layout_rep (A : Live) {j : Fin A.rounds} {t k : ℕ} {r : Fin 2} (h : A.layout j = .rep t r k) :
    t = 0 ∧ j.val = 2 * k + r.val + 1 :=
  slotAt_eq_rep h

/-- **The layout's rep rounds are below the plan's tables**, at every sample and registration, wherever v2 sets `A` up. -/
theorem SetsUp.layout_lt {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    (S : Finset (Fin A.n)) (R : A.Reg) : ∀ j t r k, A.layout j = .rep t r k → t < A.J S R := by
  intro j t r k h
  obtain ⟨rfl, -⟩ := A.layout_rep h
  rw [hA.J_eq S R]
  exact Nat.one_pos

/-- **The layout has a slot for each of a rep stream's first `K` rounds** wherever the session has at least `2K + 1`
rounds. -/
theorem Live.layout_covers (A : Live) {K : ℕ} (hK : 2 * K + 1 ≤ A.rounds) (r : Fin 2) :
    ∀ k < K, ∃ j, A.layout j = .rep 0 r k := fun k hk =>
  ⟨⟨2 * k + r.val + 1, by have := r.isLt; omega⟩, slotAt_rep k r⟩

/-- **The layout has a slot for each of the plan's tables' rep streams' first `n` rounds**, at every sample and
registration, wherever v2 sets `A` up and the session has at least `2K + 1` rounds for some `K ≥ n`. -/
theorem SetsUp.layout_covers {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {S : Finset (Fin A.n)} {R : A.Reg} (t : Fin (A.J S R)) {K : ℕ} (hK : 2 * K + 1 ≤ A.rounds) (r : Fin 2) {n : ℕ}
    (hn : n ≤ K) : ∀ k < n, ∃ j, A.layout j = .rep t.val r k := by
  have h1 := hA.J_eq S R
  have h2 := t.isLt
  have ht : t.val = 0 := by omega
  intro k hk
  rw [ht]
  exact A.layout_covers hK r k (by omega)

/-- **The layout has a slot for each of a rep stream's first `n` rounds** at every sample and registration, wherever v2
sets `A` up and `n` is at most the `2^roundsLog` rounds a stream of the session's coin tree holds (`Live.rounds_tree`). -/
theorem SetsUp.layout_covers_tree {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (hA : SetsUp V θ F A) {S : Finset (Fin A.n)} {R : A.Reg} (t : Fin (A.J S R)) (r : Fin 2) {n : ℕ}
    (hn : n ≤ 2 ^ A.coinSpec.roundsLog) : ∀ k < n, ∃ j, A.layout j = .rep t.val r k :=
  hA.layout_covers t A.rounds_tree r hn

/-! ## The stream names -/

/-- **The plan's tables are the session's**: at every sample and registration, wherever v2 sets `A` up, table `t` is the
statement's. -/
theorem SetsUp.tables_at {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {S : Finset (Fin A.n)} {R : A.Reg} (t : Fin (A.J S R)) : A.tables[t.val]? = some A.I.tags.table := by
  have h1 := hA.J_eq S R
  have h2 := t.isLt
  have ht : t.val = 0 := by omega
  rw [ht]
  rfl

/-- **The readings' name for table `t`'s rep `r` stream is the decoded record's**: at the session `--zk` sets up for
any record, wherever v2 sets `A` up, `repStreamName spec.tables[0]! r`. -/
theorem SetsUp.tables_rep {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : SetsUp V θ F A)
    {record : String} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts)) {S : Finset (Fin A.n)} {R : A.Reg}
    (t : Fin (A.J S R)) (r : ℕ) :
    A.tables[t.val]?.map (repStreamName · r) = some (repStreamName spec.tables[0]! r) := by
  rw [hA.tables_at t, hA.spec_tables hs]
  rfl

end Verity.Sampled
