module

public import Aesop.BuiltinRules
public import Flock.HmRow
public import Mathlib.Data.Nat.Notation
public import Mathlib.Tactic.ToAdditive
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.ExecParse
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-! A `setupH` statement's link-point count `m_pts`, which under a sampling design with a closure can differ between samples of one size. -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

/-- `m_pts` for `n` proved units of a circuit with `g` units per block: `PT_LOCAL + nbl`, `nbl` as `setupH` computes it. -/
def mPtsOf (g n : ℕ) : ℕ := PT_LOCAL + max 3 (log2ceil (max ((n + g - 1) / g) 1))

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

set_option maxHeartbeats 4000000 in
theorem setupH_facts (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    (if tags.typed then HmRow.parseTyped tags cf tables else HmRow.parse tags cf tables) = .ok st.c ∧
      (∀ d, draw = some d → ∃ pop, HmRow.drawn st.c pop d = .ok st.pub) ∧ st.mPts = mPtsOf st.c.g st.pub.n := by
  unfold Stmt.setupH at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  obtain ⟨c, hc, h⟩ : ∃ c, (if tags.typed then HmRow.parseTyped tags cf tables else HmRow.parse tags cf tables) =
      .ok c ∧ jp0 c = .ok st := by
    split at h
    · rename_i ht
      obtain ⟨c, hc, h⟩ := except_bind_ok h
      exact ⟨c, by simp only [ht, ↓reduceIte]; exact hc, h⟩
    · rename_i ht
      obtain ⟨c, hc, h⟩ := except_bind_ok h
      exact ⟨c, by simp only [ht, Bool.false_eq_true, ↓reduceIte]; exact hc, h⟩
  simp (config := { zeta := false }) only [jp0] at h
  (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
  obtain ⟨f0, h⟩ : ∃ f0, jpA f0 = .ok st := by
    split at h
    · obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩
    · split at h <;> (obtain ⟨f, -, h⟩ := except_bind_ok h; exact ⟨f, h⟩)
  simp (config := { zeta := false }) only [jpA] at h
  split at h
  swap
  · cases h
  (have hh := except_bind_ok h; clear h; obtain ⟨pop, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h
  obtain ⟨pub, hpub, h⟩ : ∃ pub, (∀ d, draw = some d → HmRow.drawn c pop d = .ok pub) ∧ jpB pub = .ok st := by
    split at h
    · rename_i d
      obtain ⟨p, hp, h⟩ := except_bind_ok h
      exact ⟨p, (fun d' hd' => by cases hd'; exact hp), h⟩
    · obtain ⟨p, -, h⟩ := except_bind_ok h
      exact ⟨p, (fun d' hd' => by cases hd'), h⟩
  simp (config := { zeta := false }) only [jpB] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpU at h
  obtain ⟨u0, h⟩ : ∃ u, jpU u = .ok st := by
    split at h
    · split at h
      · obtain ⟨_, -, h⟩ := except_bind_ok h
        exact ⟨(), ite_throw_ok h⟩
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  simp (config := { zeta := false }) only [jpU] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨finding, h⟩ : ∃ f, jpC f = .ok st := by
    split at h
    · extract_lets (config := { underBinder := false, onlyGivenNames := true }) q at h
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
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) n blocksReal nbl m jpD at h
  obtain ⟨hk, h⟩ := ite_throw_ok' h
  simp (config := { zeta := false }) only [jpD] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨pin, hpin, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) h2 digest sigma at h
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hs := Except.ok.inj h
  subst hs
  exact ⟨hc, fun d hd => ⟨pop, hpub d hd⟩, rfl⟩

set_option maxHeartbeats 4000000 in
theorem sampled_n {c : Circuit} {pop pub : Public} {d : Json} (h : HmRow.drawn c pop d = .ok pub) :
    ∃ ud, Draw.UnitSample.ofJson d = .ok ud ∧ pub.n = ud.proved.size := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, by assumption, rfl⟩

theorem setupH_mPts_sampled {d : Json} (h : Stmt.setupH tags cf pf coins tables (some d) partition program = .ok st) :
    ∃ ud, Draw.UnitSample.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  obtain ⟨-, hd, hm⟩ := setupH_facts h
  obtain ⟨pop, hpub⟩ := hd d rfl
  obtain ⟨ud, hud, hn⟩ := sampled_n hpub
  exact ⟨ud, hud, by rw [hm, hn]⟩

theorem setupH_mPts_of_proved {c : Circuit} {d : Json} {N₀ : ℕ}
    (hc : (if tags.typed then HmRow.parseTyped tags cf tables else HmRow.parse tags cf tables) = .ok c)
    (hD : ∀ D, Draw.UnitSample.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupH tags cf pf coins tables (some d) partition program = .ok st) : st.mPts = mPtsOf c.g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupH_mPts_sampled h
  have e := (setupH_facts h).1
  rw [hc] at e
  rw [hm, ← Except.ok.inj e, hD ud hud]

theorem setupH_mPts_eq {pf' : ByteArray} {coins' : Coins} {d d' : Json} {partition' : Option Json}
    {program' : Option ByteArray} {st' : Stmt} {ud ud' : Draw.UnitSample}
    (h : Stmt.setupH tags cf pf coins tables (some d) partition program = .ok st)
    (h' : Stmt.setupH tags cf pf' coins' tables (some d') partition' program' = .ok st')
    (hud : Draw.UnitSample.ofJson d = .ok ud) (hud' : Draw.UnitSample.ofJson d' = .ok ud')
    (hsz : ud.proved.size = ud'.proved.size) : st.mPts = st'.mPts := by
  obtain ⟨u, hu, hm⟩ := setupH_mPts_sampled h
  obtain ⟨u', hu', hm'⟩ := setupH_mPts_sampled h'
  rw [hud] at hu
  rw [hud'] at hu'
  cases hu
  cases hu'
  have hc : st.c = st'.c := by
    have e := (setupH_facts h).1
    rw [(setupH_facts h').1] at e
    exact (Except.ok.inj e).symm
  rw [hm, hm', hc, hsz]

end FlockSoundness.Discharge.Exec
