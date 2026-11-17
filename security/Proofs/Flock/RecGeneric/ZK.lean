module

public import Definitions.Claims.ZKOut
public import Definitions.Claims.Recursion
public import Proofs.Flock.Recursive.Statements
public import Proofs.Flock.Claims.ZK.Statement

/-! The ZK server as the recursion's zero knowledge reads it, per public outcome (`Claims.ZKOut`); definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion FlockSoundness.Discharge
open scoped NNReal ENNReal

structure Leaves where
  Salt : Type
  [saltFin : Fintype Salt]
  [saltNe : Nonempty Salt]
  Rand : Type
  [randFin : Fintype Rand]
  [randNe : Nonempty Rand]
  leaf : Msg → Salt → Msg
  ideal : Rand → Msg

attribute [instance] Leaves.saltFin Leaves.saltNe Leaves.randFin Leaves.randNe

def Leaves.commits (Lv : Leaves) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Msg) → Msg) (y : Fin n → Lv.Salt)
    (i : Fin n) : Msg :=
  Lv.leaf (msgs i fun j => Lv.commits msgs y (Fin.castLE i.isLt.le j)) (y i)
termination_by i.val
decreasing_by exact j.isLt

def Leaves.Hides (Lv : Leaves) (δ : ℝ) : Prop :=
  ∀ (n : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (msgs : Ω → (i : Fin n) → (Fin i → Msg) → Msg)
    (G : Ω → (Fin n → Msg) → Prop),
    Pr (fun p : Ω × (Fin n → Lv.Salt) => G p.1 (Lv.commits (msgs p.1) p.2)) ≤
      Pr (fun p : Ω × (Fin n → Lv.Rand) => G p.1 fun i => Lv.ideal (p.2 i)) + n * δ

def Leaves.hm96 (enc : ZkView.Bits × ByteArray → Msg) : Leaves where
  Salt := ZkView.Salt
  Rand := ZkView.Bits × ZkView.Salt
  leaf m y := enc (ZkView.bitsOf (Flock.Sha512.hash ⟨m.toArray⟩) + ZkView.My y, ZkView.saltHash y)
  ideal u := enc (u.1, ZkView.saltHash u.2)

/-- An outer phase's message as the server sends it, tagged `0`, so that no outer message reads as the refusal. -/
def tagOuter (m : Msg) : Msg := 0 :: m

def untagOuter : Msg → Msg
  | [] => []
  | _ :: m => m

def refusal : Msg := [1]

structure ZkRelease (W : Type) (Lv : Leaves) where
  nh : Msg → ℕ
  nl : Msg → ℕ → ℕ
  frame : Msg → List (Msg × Out) → List Msg → OC Msg
  Ωp : Msg → Type
  [pFin : ∀ x, Fintype (Ωp x)]
  [pNe : ∀ x, Nonempty (Ωp x)]
  msg : W → (x : Msg) → Ωp x → List (Msg × Msg) → OC (List Msg)
  ko : Msg → ℕ
  xo : Msg → List (Msg × Out) → Msg
  WO : Type
  wo : W → (x : Msg) → Ωp x → (Fin (∑ i ∈ Finset.range (nh x), nl x i) → Lv.Salt) → List (Msg × Out) → WO
  RO : Msg → Rel WO
  NO : Msg → Type
  ΩO : Msg → Type
  [nFin : ∀ x, Fintype (NO x)]
  [nNe : ∀ x, Nonempty (NO x)]
  [oFin : ∀ x, Fintype (ΩO x)]
  [oNe : ∀ x, Nonempty (ΩO x)]
  PO : WO → (x : Msg) → Prover (NO x × ΩO x)
  St : Type
  outO : Msg → Msg → List (Msg × Out) → St
  CO : Msg → Type
  [cFin : ∀ x, Fintype (CO x)]
  [cNe : ∀ x, Nonempty (CO x)]
  progO : (x : Msg) → Msg → St → CO x → SP (List (Msg × Out))

attribute [instance] ZkRelease.pFin ZkRelease.pNe ZkRelease.nFin ZkRelease.nNe ZkRelease.oFin ZkRelease.oNe
  ZkRelease.cFin ZkRelease.cNe

namespace ZkRelease

variable {W : Type} {Lv : Leaves}

abbrev nLeaves (Zs : ZkRelease W Lv) (x : Msg) : ℕ := ∑ i ∈ Finset.range (Zs.nh x), Zs.nl x i

def off (Zs : ZkRelease W Lv) (x : Msg) (i : ℕ) : ℕ := ∑ i' ∈ Finset.range i, Zs.nl x i'

def leafAt (Zs : ZkRelease W Lv) (x : Msg) {i : ℕ} (hi : i < Zs.nh x) (j : Fin (Zs.nl x i)) : Fin (Zs.nLeaves x) :=
  ⟨Zs.off x i + j, by
    have h1 : Zs.off x i + Zs.nl x i = ∑ i' ∈ Finset.range (i + 1), Zs.nl x i' := (Finset.sum_range_succ _ _).symm
    have h2 : ∑ i' ∈ Finset.range (i + 1), Zs.nl x i' ≤ Zs.nLeaves x :=
      Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 hi)
    omega⟩

abbrev N (Zs : ZkRelease W Lv) (x : Msg) : Type := Zs.Ωp x × (Fin (Zs.nLeaves x) → Lv.Salt)

abbrev Ω (Zs : ZkRelease W Lv) (x : Msg) : Type := Zs.NO x × Zs.ΩO x

def k (Zs : ZkRelease W Lv) (x : Msg) : ℕ := Zs.nh x + Zs.ko x

def ok (Zs : ZkRelease W Lv) (w : W) (x : Msg) (t : Zs.N x) (pre : List (Msg × Out)) : Prop :=
  Zs.RO x (Zs.xo x pre) (Zs.wo w x t.1 t.2 pre)

open Classical in
noncomputable def prover (Zs : ZkRelease W Lv) (w : W) (x : Msg) : Prover (Zs.N x × Zs.Ω x) where
  next _ hist t :=
    if h : hist.length < Zs.nh x then
      OC.bind (Zs.msg w x t.1.1 hist) fun vs =>
        Zs.frame x (hist.map fun p => (p.1, Out.msg p.2))
          (List.ofFn fun j => Lv.leaf (vs.getD j []) (t.1.2 (Zs.leafAt x h j)))
    else
      let pre := (hist.take (Zs.nh x)).map fun p => (p.1, Out.msg p.2)
      if Zs.ok w x t.1 pre then
        OC.bind ((Zs.PO (Zs.wo w x t.1.1 t.1.2 pre) x).next (Zs.xo x pre)
          ((hist.drop (Zs.nh x)).map fun p => (untagOuter p.1, p.2)) t.2) fun m => .ret (tagOuter m)
      else .ret refusal

def outcome (Zs : ZkRelease W Lv) (x : Msg) (ms : List (Msg × Out)) : Option Zs.St :=
  if ms.length ≤ Zs.nh x ∨ (ms.drop (Zs.nh x)).head?.map Prod.fst = some refusal then none
  else some (Zs.outO x (Zs.xo x (ms.take (Zs.nh x))) ((ms.drop (Zs.nh x)).map fun p => (untagOuter p.1, p.2)))

def simO (Zs : ZkRelease W Lv) (x : Msg) : SimOut Zs.St where
  C _ := Zs.CO x
  prog xo o c := Zs.progO x xo o c

def OuterZK (Zs : ZkRelease W Lv) (H : Msg → Msg)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) : Prop :=
  ∀ x, ZKOut (fun _ => Zs.NO x) (fun _ => Zs.ΩO x) H (fun _ => Zs.ko x) (fun wo _ => Zs.PO wo x) (Zs.RO x)
    (Zs.outO x) (Zs.simO x) (εO x)

noncomputable def preRun (Zs : ZkRelease W Lv) (H : Msg → Msg) (w : W) (x : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ)
    (t : Zs.N x) : List (Msg × Out) :=
  (stepWith H (Zs.prover w x) x (t, Classical.arbitrary _) (fun as => Vs.next x as θ) (Zs.nh x) [] []
    ⟨[], [], [], .rej⟩).msgs

end ZkRelease

def vAfter {Θ : Type} (Vs : VStar Θ) (x : Msg) (ko : ℕ) (pre : List Msg) : VStar Θ where
  k _ := ko
  next _ as θ := Vs.next x (pre ++ as.map tagOuter) θ

def spTag {α : Type} (pre : List Msg) : SP α → SP α
  | .ret a => .ret a
  | .hash q k => .hash q fun a => spTag pre (k a)
  | .ask as k => .ask (pre ++ as.map tagOuter) fun b => spTag pre (k b)

namespace ZkRelease

variable {W : Type} {Lv : Leaves}

open Classical in
noncomputable def err (Zs : ZkRelease W Lv) (H : Msg → Msg)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ) (x : Msg) (w : W) (Θ : Type)
    (Vs : VStar Θ) (θ : Θ) (s : Option Zs.St) : ℝ :=
  s.elim 0 (fun o => Ex fun t : Zs.N x =>
      if goesOn (Zs.preRun H w x Vs θ t) ∧ Zs.ok w x t (Zs.preRun H w x Vs θ t) then
        εO x (Zs.xo x (Zs.preRun H w x Vs θ t)) (Zs.wo w x t.1 t.2 (Zs.preRun H w x Vs θ t)) Θ
          (vAfter Vs x (Zs.ko x) ((Zs.preRun H w x Vs θ t).map Prod.fst)) θ o
      else 0) + Zs.nLeaves x * δ

def sim (Zs : ZkRelease W Lv) : SimOut (Option Zs.St) where
  C x := (Fin (Zs.nLeaves x) → Lv.Rand) × Zs.CO x
  prog x s c :=
    spBind (simRounds (fun ms => if h : ms.length < Zs.nh x then
          spOf (Zs.frame x ms (List.ofFn fun j => Lv.ideal (c.1 (Zs.leafAt x h j))))
        else .ret []) (Zs.nh x) []) fun ms =>
      if goesOn ms then
        match s with
        | none => simRounds (fun _ => .ret refusal) (Zs.ko x) ms
        | some o => spBind (spTag (ms.map Prod.fst) (Zs.progO x (Zs.xo x ms) o c.2)) fun ts =>
            .ret (ms ++ ts.map fun p => (tagOuter p.1, p.2))
      else .ret ms

end ZkRelease

def Hm96Hides : Prop :=
  ∀ (enc : ZkView.Bits × ByteArray → Msg) (δ : ℝ≥0), Claims.Hm96Hiding ZkView.My ZkView.saltHash (δ : ℝ≥0∞) →
    (Leaves.hm96 enc).Hides δ

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric

/-- The outer simulator `progO` is any program: one that picks an outer witness by choice can meet `OuterZK`, so the
result is zero knowledge in the usual sense only at a counted simulator (Flock's at the instance). -/
def GenericRecursiveZK : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (Zs : ZkRelease W Lv)
    (εO : (x : Msg) → Msg → Zs.WO → (Θ : Type) → VStar Θ → Θ → Zs.St → ℝ) (δ : ℝ),
    Lv.Hides δ → Zs.OuterZK H εO →
      ZKOut Zs.N Zs.Ω H Zs.k Zs.prover (fun _ _ => True) Zs.outcome Zs.sim (Zs.err H εO δ)

def GenericRecursiveZKCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (Zs : ZkRelease W Lv) (cF cO : Msg → ℕ),
    (∀ x pre ls, ((Zs.frame x pre ls).eval H).2.length ≤ cF x) →
    (∀ x xO o (c : Zs.CO x) ask, ((Zs.progO x xO o c).eval H ask).2 ≤ cO x) →
    ZKMal.SimCostLe H (Zs.sim.given none) (fun x => Zs.nh x * (cF x + 1) + Zs.ko x) ∧
      ∀ o, ZKMal.SimCostLe H (Zs.sim.given (some o)) (fun x => Zs.nh x * (cF x + 1) + cO x)

end Flock.Guarantees
