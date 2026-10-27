import Proofs.Flock.Claims.Scope.Readable
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# What the verifier's setup gives the scope

`table0`: when `Read.tablesOf` accepts an untyped statement without the verifier's own public input words, its first
table's statement `st` has the circuit the scope check parses (`ProvedScope.circuitOf`), whose public file the scope check
loads (`ProvedScope.publicOf`), `m = k_log + nbl`, `6 ≤ k_log ≤ 27` and `m ≤ 35` (`Stmt.setupH`'s and
`Stmt.setupHidden`'s refusals), and under `--zk` `25 ≤ m` (`Zk.shapeOk`'s `queries0`).
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge Exec

/-- **What `Stmt.setupH` accepted, for the scope**: the circuit `HmRow.parse` accepted, whose public file
`HmRow.loadPublic` loaded, `m = k_log + nbl`, `k_log ≤ 27` and `m ≤ 35`. -/
theorem setupH_scope {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (ht : tags.typed = false) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    HmRow.parse tags cf tables = .ok st.c ∧ (∃ pop, HmRow.loadPublic tags st.c pf = .ok pop) ∧
      st.m = st.c.kLog + st.nbl ∧ st.c.kLog ≤ 27 ∧ st.m ≤ 35 := by
  unfold Stmt.setupH at h
  simp (config := { zeta := false }) only [ht, Bool.false_eq_true, ↓reduceIte] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨c, hc, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
  obtain ⟨f0, h⟩ : ∃ f0, jpA f0 = .ok st := by
    split at h
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · split at h <;> (obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩)
  simp (config := { zeta := false }) only [jpA] at h
  split at h
  swap
  · cases h
  (have hh := except_bind_ok h; clear h; obtain ⟨pop, hpop, h⟩ := hh)
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h)
  obtain ⟨pub, h⟩ : ∃ pub, jpB pub = .ok st := by
    split at h <;> (obtain ⟨p, -, h⟩ := except_bind_ok h; exact ⟨p, h⟩)
  simp (config := { zeta := false }) only [jpB] at h
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h)
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · (extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h)
      split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨_, -, h⟩ := except_bind_ok h
        obtain ⟨f, -, h⟩ := except_bind_ok h
        exact ⟨f, h⟩
      · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
  simp (config := { zeta := false }) only [jpC] at h
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h)
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨⟨⟩, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) h2 digest sigma at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hk
  exact ⟨hc, ⟨pop, hpop⟩, rfl, Nat.le_of_not_lt hk.1, Nat.le_of_not_lt hk.2⟩

theorem parseH_identity {tags : Tags} (ht : tags.typed = false) (x : Lean.Json) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse { tags with identity := x } b t = HmOut.parse tags b t := by
  simp only [HmOut.parse, Composed.parse_identity, Tags.isTmpl]
  simp only [ht, Bool.not_false, ite_true]

theorem parseH_tagsAt {tags : Tags} (ht : tags.typed = false) (count j : ℕ) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse (tagsAt tags count j) b t = HmOut.parse tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · exact parseH_identity ht _ _ _

/-- An untyped `HmOut.parseCommit` reads no identity, so it agrees at every table's tags. -/
theorem parseCommit_tagsAt {tags : Tags} (ht : tags.typed = false) (count j : ℕ) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parseCommit (tagsAt tags count j) b t = HmOut.parseCommit tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · have hi : ({ tags with identity := tags.tableIdentity count j } : Tags).isTmpl = tags.isTmpl := rfl
    unfold HmOut.parseCommit
    rw [hi, parseTmpl_identity, parseH_identity ht]

theorem loadPublicH_tagsAt (tags : Tags) (count j : ℕ) (c : Circuit) (b : ByteArray) :
    HmOut.loadPublic (tagsAt tags count j) c b = HmOut.loadPublic tags c b := by
  unfold tagsAt
  split_ifs <;> rfl

theorem tagsAt_hiddenOutputs (tags : Tags) (count j : ℕ) :
    (tagsAt tags count j).hiddenOutputs = tags.hiddenOutputs := by
  unfold tagsAt; split_ifs <;> rfl

theorem list_forM_ok {α ε : Type} {f : α → Except ε PUnit} :
    ∀ (L : List α), forM L f = .ok () → ∀ a ∈ L, f a = .ok ()
  | [], _ => by simp
  | a :: L, h => by
      rw [List.forM_cons] at h
      obtain ⟨u, hu, h⟩ := except_bind_ok h
      intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hu
      · exact list_forM_ok L h b hb

theorem array_forM_ok {α ε : Type} {f : α → Except ε PUnit} {xs : Array α} (h : xs.forM f = .ok ()) :
    ∀ (j : ℕ) (hj : j < xs.size), f xs[j] = .ok () := by
  rcases xs with ⟨l⟩
  have hl : forM l f = .ok () := by simpa using h
  intro j hj
  exact list_forM_ok l hl _ (List.getElem_mem (by simpa using hj))

/-- Under `--zk`, a statement `Zk.shapeOk` passes has `25 ≤ m` (`queries0` refuses `m ≤ 24`). -/
theorem m25_of_shapeOk {st : Stmt} (h : Zk.shapeOk st = .ok ()) : 25 ≤ st.m := by
  unfold Zk.shapeOk at h
  obtain ⟨_, hq, -⟩ := except_bind_ok h
  unfold Zk.queries0 at hq
  by_contra hm
  simp only [show st.m ≤ 24 by omega, ↓reduceIte] at hq
  cases hq

/-- **What the setup gives the scope**: the first table's statement of an untyped statement `Read.tablesOf` accepts,
without the verifier's own public input words. -/
theorem table0 {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) :
    ∃ st, sts[0]? = some st ∧ ProvedScope.circuitOf r.tags r.I.circuitFile r.I.tables = .ok st.c ∧
      (∃ pop, ProvedScope.publicOf r.tags st.c r.I.publicFile = .ok pop) ∧
      st.m = st.c.kLog + st.nbl ∧ 6 ≤ st.c.kLog ∧ st.c.kLog ≤ 27 ∧ st.m ≤ 35 ∧ (r.zk = true → 25 ≤ st.m) := by
  unfold Read.tablesOf at hs
  split at hs
  swap
  · cases hs
  rename_i spec sts' hset
  have hsts : sts' = sts := by
    split at hs
    · split at hs
      · exact Option.some.inj hs
      · cases hs
    · exact Option.some.inj hs
  subst hsts
  have hshape : r.zk = true → sts'.forM Zk.shapeOk = .ok () := by
    intro hz
    simp only [hz, ↓reduceIte] at hs
    split at hs
    · assumption
    · cases hs
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨hpos, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall 0 hpos
  have ht' : (tagsAt r.tags r.I.count 0).typed = false := (tagsAt_typed _ _ _).trans ht
  refine ⟨sts'[0], (Array.getElem?_eq_getElem hpos), ?_⟩
  have hzk : r.zk = true → 25 ≤ sts'[0].m := fun hz => m25_of_shapeOk (array_forM_ok (hshape hz) 0 hpos)
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs] at hd
  cases hh : r.tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    obtain ⟨hp, ⟨pop, hpop⟩, hm, hk, hm35⟩ := setupH_scope ht' hd
    have L := setupH_layout ht' hd
    rw [Composed.parse_tagsAt] at hp
    rw [Composed.loadPublic_tagsAt] at hpop
    refine ⟨?_, ⟨pop, ?_⟩, hm, L.kLog6, hk, hm35, hzk⟩
    · simp only [ProvedScope.circuitOf, hh, Bool.false_eq_true, ↓reduceIte, hp]
    · simp only [ProvedScope.publicOf, hh, Bool.false_eq_true, ↓reduceIte, hpop]
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    have S := Hidden.setupHidden_spec' hd
    have L := Hidden.hidden_layout' ht' hd (HiddenExec.setupTables_pinnedH hh hset 0 hpos)
    have hp := S.parse
    rw [parseCommit_tagsAt ht] at hp
    obtain ⟨pop, hpop, -⟩ := S.pub
    rw [loadPublicH_tagsAt] at hpop
    refine ⟨?_, ⟨pop, ?_⟩, S.m, L.kLog6, S.kLog, S.m35, hzk⟩
    · simp only [ProvedScope.circuitOf, hh, ↓reduceIte, hp]
    · simp only [ProvedScope.publicOf, hh, ↓reduceIte, hpop]

end Flock.Readable
