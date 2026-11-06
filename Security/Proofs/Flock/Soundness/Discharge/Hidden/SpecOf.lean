import Proofs.Flock.Soundness.Discharge.Hidden.Spec

/-!
# What `Stmt.setupHiddenOf` checked and produced (`SpecOf`)

`Spec` on the circuit the caller parsed (`parsed`), which `Stmt.setupHiddenOf` reads: `HmOut.parseCommit`'s for
`Stmt.setupHidden` (`Stmt.setupHidden_eq`). `SpecOf.spec` is `Spec` at `HmOut.parseCommit`.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} {parsed : V Circuit}

/-- **What `Stmt.setupHiddenOf` accepted on `parsed`**: `Spec`'s fields, with `parsed`'s circuit for `HmOut.parse`'s. -/
structure SpecOf (parsed : V Circuit) (tags : Tags) (pf : ByteArray) (coins : Coins) (draw : Option Lean.Json)
    (st : Stmt) : Prop where
  tagsEq : st.tags = tags
  leaf : st.leaf = default
  parse : parsed = .ok st.c
  merkle : MerkleScheme.find? st.c.leafScheme = some st.merkle
  pub : ∃ pop, HmOut.loadPublic tags st.c pf = .ok pop ∧ (draw = none → st.pub = pop) ∧
    ∀ d, draw = some d → HmOut.drawn st.c pop d = .ok st.pub
  checkPublic : HmOut.checkPublic st.c st.pub = .ok ()
  n : st.n = st.pub.n
  blocksReal : st.blocksReal = (st.n + st.c.g - 1) / st.c.g
  nbl : st.nbl = max 3 (log2ceil (max st.blocksReal 1))
  m : st.m = st.c.kLog + st.nbl
  kLog : st.c.kLog ≤ 27
  m35 : st.m ≤ 35
  mPts : st.mPts = PT_LOCAL + st.nbl
  pin : HmRow.pin st.c = .ok st.pin
  da : st.da = (HmOut.delta st.c st.pin).1
  db : st.db = (HmOut.delta st.c st.pin).2
  regions : HmOut.regions st.c st.pub = .ok st.regions
  inRange : checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok ()
  sigma : st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha)
  hello : helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHiddenOf` accepted** is its `SpecOf`. -/
theorem setupHiddenOf_spec (h : Stmt.setupHiddenOf parsed tags pf coins draw partition program = .ok st) :
    SpecOf parsed tags pf coins draw st := by
  cases draw <;> unfold Stmt.setupHiddenOf at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := klog_of ‹¬ (_ || _) = true›
    have he := checkOwn_none ‹HmIn.checkOwn _ _ none = _›
    have := loadPublic_noPorts he ‹HmIn.loadPublic _ _ _ _ = _›
    obtain rfl := regions_noPorts he ‹HmIn.regions _ _ _ = _›
    exact {
      tagsEq := rfl
      leaf := rfl
      parse := by assumption
      merkle := by assumption
      pub := by
        refine ⟨_, this, fun hd => ?_, fun d' hd => ?_⟩
        · first | exact absurd hd (Option.some_ne_none _) | rfl
        · first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; rw [← drawn_noPorts he]; assumption)
      checkPublic := by assumption
      n := rfl
      blocksReal := rfl
      nbl := rfl
      m := rfl
      kLog := hk.1
      m35 := hk.2
      mPts := rfl
      pin := by assumption
      da := congrArg Prod.fst (delta_noPorts he)
      db := congrArg Prod.snd (delta_noPorts he)
      regions := by rw [Array.append_empty]; assumption
      inRange := by assumption
      sigma := rfl
      hello := by assumption }

/-- `SpecOf` at `HmOut.parseCommit` is `Spec`. -/
theorem SpecOf.spec (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : Spec tags cf pf coins tables draw st :=
  ⟨s.tagsEq, s.leaf, s.parse, s.merkle, s.pub, s.checkPublic, s.n, s.blocksReal, s.nbl, s.m, s.kLog, s.m35, s.mPts,
    s.pin, s.da, s.db, s.regions, s.inRange, s.sigma, s.hello⟩

/-- `Spec` is `SpecOf` at `HmOut.parseCommit`. -/
theorem Spec.specOf (s : Spec tags cf pf coins tables draw st) : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st :=
  ⟨s.tagsEq, s.leaf, s.parse, s.merkle, s.pub, s.checkPublic, s.n, s.blocksReal, s.nbl, s.m, s.kLog, s.m35, s.mPts,
    s.pin, s.da, s.db, s.regions, s.inRange, s.sigma, s.hello⟩

/-- **What `Stmt.setupHidden` accepted**, as `SpecOf` at `HmOut.parseCommit`. -/
theorem setupHidden_specOf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st :=
  (setupHidden_spec' h).specOf

end FlockSoundness.Discharge.Hidden
