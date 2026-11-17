module

public import Proofs.Flock.VBridge.Family.Commits
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
import Mathlib.Tactic.ReduceModChar

/-! The family's reads files, round map and round commitments at V*'s staging (`ownOf`, `selOf`, `bc`); the tops
(`rec-top-L<l>`) are no commit strings, so `VCommits` cannot hold at a top row. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

inductive Absorb
  | elts
  | none
  deriving DecidableEq, Repr

def isCoin : Exec.Kind → Bool
  | .coin _ => true
  | .msg _ => false

def foldTags : ℕ → List Absorb
  | 0 => []
  | k + 1 => .none :: List.replicate k .elts

def recTags : List Accounting.Level → List Absorb
  | [] => []
  | [lv] => foldTags lv.lanes ++ [.elts, .none, .none]
  | lv :: lv' :: rest => foldTags lv.lanes ++ [.elts, .elts, .none, .elts] ++ [.none, .none, .elts] ++ recTags (lv' :: rest)

def ligTags (sch : FlockSoundness.Accounting.Schedule) : List Absorb :=
  match sch.levels with
  | [] => []
  | _ :: rest => .elts :: .elts :: (List.replicate sch.k0 .elts ++
      match rest with
      | [] => []
      | _ :: _ => [.elts, .elts, .none, .elts] ++ [.none, .none, .elts] ++ recTags rest)

def streamTags (m kLog claims : ℕ) (sch : FlockSoundness.Accounting.Schedule) : List Absorb :=
  ([.none, .none, .elts] ++ List.replicate (m - 6) .elts) ++
    (([.elts, .none] ++ List.replicate (kLog - 6) .elts ++ [.elts]) ++
      ((List.replicate claims .elts ++ [.none]) ++ ligTags sch))

def eltRounds (tags : List Absorb) : List ℕ := (List.range tags.length).filter fun s => tags[s]? == some .elts

section coins

variable {F K D : Type}

theorem coins_pairRounds : ∀ n, (Exec.pairRounds F n).countP isCoin = n
  | 0 => rfl
  | n + 1 => by simp [Exec.pairRounds, isCoin, List.countP_cons, coins_pairRounds n]

theorem coins_switches : ∀ n, (Exec.switches F n).countP isCoin = n
  | 0 => rfl
  | n + 1 => by simp [Exec.switches, isCoin, List.countP_cons, coins_switches n]

theorem coins_foldShape : ∀ k, (Exec.foldShape F K k).countP isCoin = k
  | 0 => rfl
  | k + 1 => by simp [Exec.foldShape, isCoin, List.countP_cons, coins_foldShape k]

theorem length_foldTags (k : ℕ) : (foldTags k).length = k := by
  cases k <;> simp [foldTags]

theorem coins_recShape (ext : ℕ) : ∀ levels, (Exec.recShape F K D ext levels).countP isCoin = (recTags levels).length
  | [] => rfl
  | [lv] => by
    simp [Exec.recShape, recTags, Exec.qbShape, isCoin, List.countP_append, List.countP_cons, coins_foldShape, length_foldTags]
  | lv :: lv' :: rest => by
    rw [Exec.recShape, recTags]
    simp [Exec.oodShape, Exec.qbShape, isCoin, List.countP_append, List.countP_cons, coins_foldShape, length_foldTags,
      coins_recShape ext (lv' :: rest)]

theorem coins_repShape (m kLog nExtra : ℕ) (sch : FlockSoundness.Accounting.Schedule) :
    (Exec.repShape F K D m kLog nExtra sch).countP isCoin = (streamTags m kLog (nExtra + 2) sch).length := by
  obtain ⟨m', k0, ls, ext⟩ := sch
  rcases ls with _ | ⟨lv₀, _ | ⟨lv₁, rest⟩⟩ <;>
    simp [Exec.repShape, streamTags, Exec.zcShape, Exec.lcShape, Exec.opShape, Exec.ligShape, ligTags, Exec.oodShape, Exec.qbShape, isCoin,
      List.countP_append, List.countP_cons, coins_pairRounds, coins_switches, coins_foldShape, coins_recShape]

end coins

def cIndex (v : String) : Option ℕ := if v.startsWith "rec-c" then (v.drop 5).toString.toNat? else none

def selOf (n₀ : ℕ) (kc : ℕ → ℕ) (i : ℕ) (v : String) (x : ℕ) : Prop :=
  Registered.roundCommitment v = true ∧ ∃ k, cIndex v = some k ∧ x < 2 ∧ i = x * n₀ + kc k

theorem selRounds_selOf (n₀ : ℕ) (kc : ℕ → ℕ) : VStarFamily.SelRounds (selOf n₀ kc) := fun _ _ _ h => h.1

def mPort (name : String) (k : ℕ) : Registered.Port :=
  { name, value := s!"rec-c{k}", root := ByteArray.empty, leaves := 2, words := none, wordBits := 16,
    domain := ByteArray.empty, readsRel := #[some 0, some 1] }

def mIndex (nm : String) : Option ℕ := if nm.startsWith "m" then (nm.drop 1).toString.toNat? else none

def ownOf : VStar.Stmt → Array Registered.Port
  | .inner _ _ p => (p.S.ports.filterMap fun (nm, _) => (mIndex nm).map (mPort nm)).toArray
  | .recOpen .. | .recOpenStacked .. | .sparse .. | .opening .. | .cap .. | .link .. => #[]

def isMsg (p : Registered.Port) : Bool := (mIndex p.name).isSome || (cIndex p.value).isSome

def msgPorts (own : Array Registered.Port) : List (String × String × Array (Option ℕ)) :=
  (own.toList.filter isMsg).map fun p => (p.name, p.value, p.reads)

theorem mem_msgPorts {own : Array Registered.Port} {q : String × String × Array (Option ℕ)} :
    q ∈ msgPorts own ↔ ∃ p ∈ own, isMsg p = true ∧ (p.name, p.value, p.reads) = q := by
  simp [msgPorts, List.mem_filter, Array.mem_toList_iff, and_assoc]

def SameMsg (own own' : Array Registered.Port) : Prop := ∀ q, q ∈ msgPorts own ↔ q ∈ msgPorts own'

theorem isMsg_ownOf {st : VStar.Stmt} {p : Registered.Port} (hp : p ∈ ownOf st) : isMsg p = true := by
  cases st with
  | recOpen | recOpenStacked | sparse | opening | cap | link => simp [ownOf] at hp
  | inner _ _ part =>
    simp only [ownOf, List.mem_toArray, List.mem_filterMap, Option.map_eq_some_iff] at hp
    obtain ⟨⟨nm, _⟩, -, k, hk, rfl⟩ := hp
    simp [isMsg, mPort, hk]

def SelMsg (sel : ℕ → String → ℕ → Prop) : Prop := ∀ i v x, sel i v x → (cIndex v).isSome = true

theorem selMsg_selOf (n₀ : ℕ) (kc : ℕ → ℕ) : SelMsg (selOf n₀ kc) := by
  rintro i v x ⟨-, k, hk, -⟩
  rw [hk]
  rfl

theorem skOf_congr {S : ℕ} (Vs : Fin S → VStmt) {own own' : Fin S → Array Registered.Port}
    {sel : ℕ → String → ℕ → Prop} (hsel : SelMsg sel) (h : ∀ j, SameMsg (own j) (own' j)) :
    skOf Vs own sel = skOf Vs own' sel := by
  have key : ∀ (o o' : Array Registered.Port), SameMsg o o' → ∀ (nm : String) (u : ℕ) (i : ℕ),
      (∃ p ∈ o, p.name = nm ∧ ∃ x, p.reads[u]? = some (some x) ∧ sel i p.value x) →
      (∃ p ∈ o', p.name = nm ∧ ∃ x, p.reads[u]? = some (some x) ∧ sel i p.value x) := by
    rintro o o' ho nm u i ⟨p, hp, rfl, x, hx, hs⟩
    have hq : (p.name, p.value, p.reads) ∈ msgPorts o' :=
      (ho _).1 (mem_msgPorts.2 ⟨p, hp, by simp [isMsg, hsel _ _ _ hs], rfl⟩)
    obtain ⟨p', hp', -, he⟩ := mem_msgPorts.1 hq
    simp only [Prod.mk.injEq] at he
    obtain ⟨h1, h2, h3⟩ := he
    exact ⟨p', hp', h1, x, h3 ▸ hx, h2 ▸ hs⟩
  funext i j
  ext pos
  rcases pos with ⟨u, q⟩ | g
  · simp only [mem_skOf_inl]
    exact ⟨key _ _ (h j) _ _ _, key _ _ (fun q => (h j q).symm) _ _ _⟩
  · exact iff_of_false (inr_not_mem_skOf g) (inr_not_mem_skOf g)

theorem fullMiss_eq_none {own : Array Registered.Port} {d : Flock.Draw.UnitSample} :
    Registered.fullMiss own d = none ↔
      ∀ p ∈ own, Registered.roundCommitment p.value = true → ∀ u < p.reads.size,
        p.reads[u]!.isSome = true → Registered.samplesInFull d u = true := by
  unfold Registered.fullMiss
  rw [Array.findSome?_eq_none_iff]
  refine forall₂_congr fun p _ => ?_
  by_cases hc : Registered.roundCommitment p.value = true
  · simp only [hc, ite_true, Option.map_eq_none_iff, List.find?_eq_none, List.mem_range, Bool.and_eq_true,
      Bool.not_eq_true', not_and, Bool.not_eq_false, true_implies]
  · simp [hc]

theorem checkFull_ok_iff {own : Array Registered.Port} {dj : Lean.Json} {d : Flock.Draw.UnitSample}
    (hd : Flock.Draw.UnitSample.ofJson dj = .ok d) :
    Registered.checkFull (some own) (some dj) = .ok () ↔ Registered.fullMiss own d = none := by
  unfold Registered.checkFull
  dsimp only
  rw [hd]
  have e : ∀ f : Flock.Draw.UnitSample → Except String Unit, (Except.ok d >>= f) = f d := fun _ => rfl
  rw [e]
  cases Registered.fullMiss own d with
  | none => exact iff_of_true rfl rfl
  | some _ => exact iff_of_false (fun h => by cases h) (fun h => by cases h)

theorem checkFull_of_msgPorts {own own' : Array Registered.Port} {dj : Lean.Json} {d : Flock.Draw.UnitSample}
    (hd : Flock.Draw.UnitSample.ofJson dj = .ok d) (hm : ∀ p ∈ own', isMsg p = true)
    (h : SameMsg own own') (hU4 : Registered.checkFull (some own) (some dj) = .ok ()) :
    Registered.checkFull (some own') (some dj) = .ok () := by
  have hn := fullMiss_eq_none.1 ((checkFull_ok_iff hd).1 hU4)
  refine (checkFull_ok_iff hd).2 (fullMiss_eq_none.2 fun p' hp' hc u hu hs => ?_)
  have hq : (p'.name, p'.value, p'.reads) ∈ msgPorts own := (h _).2 (mem_msgPorts.2 ⟨p', hp', hm p' hp', rfl⟩)
  obtain ⟨p, hp, -, he⟩ := mem_msgPorts.1 hq
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, h2, h3⟩ := he
  exact hn p hp (h2 ▸ hc) u (h3 ▸ hu) (h3 ▸ hs)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

def own (j : Fin F.stmts.size) : Array Registered.Port := ownOf F.stmts[j]

def tags : List Absorb := streamTags F.sh.m F.sh.kLog F.sh.claims (schedOf F.sh.sched)

def n₀ : ℕ := F.tags.length

def kc (k : ℕ) : ℕ := (eltRounds F.tags).getD k 0

def sel : ℕ → String → ℕ → Prop := selOf F.n₀ F.kc

def bc (cOf : Cm → List UInt8) (j : Fin F.stmts.size) (e : Cm) (_ : PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv) :
    List UInt8 :=
  cOf e

theorem selRounds_sel : SelRounds F.sel := selRounds_selOf F.n₀ F.kc

/-- Discharged by the verifier's check 7: the registered reads' message-round ports are the statement's (`msgPorts`
against `ownOf`). -/
def HOwn : Prop := ∀ R ω t R₂ j, SameMsg (F.zs R ω t R₂ j).own (F.own j)

theorem skOf_zs (hown : F.HOwn) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) :
    skOf F.Vs (fun j => (F.zs R ω t R₂ j).own) F.sel = skOf F.Vs F.own F.sel :=
  skOf_congr F.Vs (selMsg_selOf F.n₀ F.kc) (hown R ω t R₂)

theorem vbridgeRead_of_rounds (dg : ByteArray → ByteArray) (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs) (hsh : F.HShape dg N k Rw GS)
    (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw) (hsch : F.HSch)
    (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (cOf : Cm → List UInt8)
    (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg) (hSk : F.Sk = skOf F.Vs F.own F.sel)
    (hcmt : F.cmt = cmtOf F.Vs (F.bc cOf)) (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec)
    (hm : F.msg = msgOf F.Vs F.Sk dec) (hpub : F.PubCommits (F.bc cOf)) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  vbridgeRead_of_family F dg N k Rw hst hsh hres hu hrd hsch hfiles hruns
    (F.hCommits_of F.own F.sel (F.bc cOf) hSk hcmt hpub) (F.vDecodesRead_of dec hw hm)
    F.vFullRead_of

end VStarFamily

end family

end FlockVBridge.Family
