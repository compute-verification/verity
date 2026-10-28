import Flock.Desc
import Proofs.Flock.Soundness.Discharge.Hidden.Spec

/-!
# What `Stmt.setupHidden` checked and produced, at any public inputs (`SpecP`)

`Spec` is `Stmt.setupHidden`'s statement without the verifier's public inputs, where `HmIn.checkOwn` leaves no public
input port (`checkOwn_none`) and `HmIn`'s public file, Δ and regions are `HmOut`'s. `SpecP` is the same walk at the
verifier's own public input words `own`, any value: the circuit's public input ports `pi` (`HmIn.parse`), the public file
`HmIn.loadPublic` reads and `HmIn.checkOwn` holds to `own`, Δ without the public leaves' copies (`HmIn.delta`), and
`HmOut`'s regions followed by the public leaves' in regions (`HmIn.regions`).

* `setupHidden_specP`: every statement `Stmt.setupHidden` accepts at `own` is its `SpecP`.
* `SpecP.spec`: at `own = none`, `SpecP` is `Spec`.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

/-- **What `Stmt.setupHidden` checked and produced at the verifier's public inputs `own`**, with the circuit's public input
ports `pi`. -/
structure SpecP (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Lean.Json) (own : Option ByteArray) (pi : HmIn.Ports) (st : Stmt) : Prop where
  tagsEq : st.tags = tags
  leaf : st.leaf = default
  parse : HmOut.parseCommit tags cf tables = .ok st.c
  ports : HmIn.parse st.c = .ok pi
  merkle : MerkleScheme.find? st.c.leafScheme = some st.merkle
  pub : ∃ pop, HmIn.loadPublic tags st.c pi pf = .ok pop ∧ HmIn.checkOwn pi pop own = .ok () ∧
    (draw = none → st.pub = pop) ∧ ∀ d, draw = some d → HmIn.drawn st.c pi pop d = .ok st.pub
  checkPublic : HmOut.checkPublic st.c st.pub = .ok ()
  n : st.n = st.pub.n
  blocksReal : st.blocksReal = (st.n + st.c.g - 1) / st.c.g
  nbl : st.nbl = max 3 (log2ceil (max st.blocksReal 1))
  m : st.m = st.c.kLog + st.nbl
  kLog : st.c.kLog ≤ 27
  m35 : st.m ≤ 35
  mPts : st.mPts = PT_LOCAL + st.nbl
  pin : HmRow.pin st.c = .ok st.pin
  da : st.da = (HmIn.delta st.c pi st.pin).1
  db : st.db = (HmIn.delta st.c pi st.pin).2
  regions : ∃ ro ri, HmOut.regions st.c st.pub = .ok ro ∧ HmIn.regions st.c pi st.pub = .ok ri ∧ st.regions = ro ++ ri
  inRange : checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok ()
  sigma : st.sigma = Sha512.hash (tags.sigmaTag.toUTF8 ++ st.c.sha ++ st.digest ++ st.pub.sha)
  hello : helloOf tags st.sigma (PT_LOCAL + st.nbl) coins = .ok st.hello

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program own : Option ByteArray} {st : Stmt}

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHidden` accepted at the verifier's public inputs `own`** is its `SpecP`, at the circuit's public
input ports. -/
theorem setupHidden_specP (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    ∃ pi, SpecP tags cf pf coins tables draw own pi st := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := klog_of ‹¬ (_ || _) = true›
    exact ⟨_, {
      tagsEq := rfl
      leaf := rfl
      parse := by assumption
      ports := by assumption
      merkle := by assumption
      pub := by
        refine ⟨_, by assumption, by assumption, fun hd => ?_, fun d' hd => ?_⟩
        · first | exact absurd hd (Option.some_ne_none _) | rfl
        · first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; assumption)
      checkPublic := by assumption
      n := rfl
      blocksReal := rfl
      nbl := rfl
      m := rfl
      kLog := hk.1
      m35 := hk.2
      mPts := rfl
      pin := by assumption
      da := rfl
      db := rfl
      regions := ⟨_, _, by assumption, by assumption, rfl⟩
      inRange := by assumption
      sigma := rfl
      hello := by assumption }⟩

/-- At a flat statement's tags (`Tags.isTmpl`), the committed circuit is `HmOut.parse`'s (`Spec.parseF`). -/
theorem SpecP.parseF {own : Option ByteArray} {pi : HmIn.Ports} (s : SpecP tags cf pf coins tables draw own pi st)
    (hi : tags.isTmpl = false) : HmOut.parse tags cf tables = .ok st.c := by
  have h := s.parse
  rwa [HmOut.parseCommit, if_neg (by simp [hi])] at h

/-- **A flat descriptor's tables are flat**: `Flock.Desc.parse` is `HmOut.parse`, which refuses a templated file
(`HmOut.parse_isTmpl`), so every table's tags (`tagsAt`) of a file it reads are flat. -/
theorem isTmpl_of_desc {D : Flock.Desc} {cf₀ : ByteArray} {tables₀ : List (String × ByteArray)}
    (hD : Flock.Desc.parse tags cf₀ tables₀ = .ok D) (count j : ℕ) :
    (FlockSoundness.Discharge.Exec.tagsAt tags count j).isTmpl = false :=
  (FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl (Flock.Desc.parse_flat hD))

/-- **Under a flat descriptor, the committed circuit is the flat one**: at any table's tags of a file `Flock.Desc.parse`
reads, `HmOut.parseCommit` is `HmOut.parse`. So `SpecP`'s `parse` field there is the flat parse, and `AcceptedHP.hi`
holds (`isTmpl_of_desc`): a statement that reads `SpecP` or `AcceptedHP` under such a descriptor means what it meant
at the flat parse. -/
theorem parseCommit_eq_of_desc {D : Flock.Desc} {cf₀ : ByteArray} {tables₀ : List (String × ByteArray)}
    (hD : Flock.Desc.parse tags cf₀ tables₀ = .ok D) (count j : ℕ) :
    HmOut.parseCommit (FlockSoundness.Discharge.Exec.tagsAt tags count j) cf tables =
      HmOut.parse (FlockSoundness.Discharge.Exec.tagsAt tags count j) cf tables := by
  rw [HmOut.parseCommit, if_neg (by simp [isTmpl_of_desc hD count j])]

/-- `parseCommit_eq_of_desc` at `SpecP`: a `SpecP` at a table's tags of a flat descriptor's file is one at the flat
parse. -/
theorem SpecP.parse_flat_of_desc {D : Flock.Desc} {cf₀ : ByteArray} {tables₀ : List (String × ByteArray)}
    {count j : ℕ} {own : Option ByteArray} {pi : HmIn.Ports} (hD : Flock.Desc.parse tags cf₀ tables₀ = .ok D)
    (s : SpecP (FlockSoundness.Discharge.Exec.tagsAt tags count j) cf pf coins tables draw own pi st) :
    HmOut.parse (FlockSoundness.Discharge.Exec.tagsAt tags count j) cf tables = .ok st.c :=
  s.parseF (isTmpl_of_desc hD count j)

/-- **Without the verifier's public inputs, `SpecP` is `Spec`**: `HmIn.checkOwn` at `none` leaves no public input port,
and with none `HmIn`'s public file, draw, Δ and regions are `HmOut`'s. -/
theorem SpecP.spec {pi : HmIn.Ports} (h : SpecP tags cf pf coins tables draw none pi st) :
    Spec tags cf pf coins tables draw st := by
  obtain ⟨pop, hl, hown, hnone, hsome⟩ := h.pub
  have he := checkOwn_none hown
  obtain ⟨ro, ri, hro, hri, hr⟩ := h.regions
  obtain rfl := regions_noPorts he hri
  exact {
    tagsEq := h.tagsEq
    leaf := h.leaf
    parse := h.parse
    merkle := h.merkle
    pub := ⟨pop, loadPublic_noPorts he hl, hnone, fun d hd => (drawn_noPorts he).symm ▸ hsome d hd⟩
    checkPublic := h.checkPublic
    n := h.n
    blocksReal := h.blocksReal
    nbl := h.nbl
    m := h.m
    kLog := h.kLog
    m35 := h.m35
    mPts := h.mPts
    pin := h.pin
    da := h.da.trans (congrArg Prod.fst (delta_noPorts he))
    db := h.db.trans (congrArg Prod.snd (delta_noPorts he))
    regions := by rw [hr, Array.append_empty]; exact hro
    inRange := h.inRange
    sigma := h.sigma
    hello := h.hello }

end FlockSoundness.Discharge.Hidden
