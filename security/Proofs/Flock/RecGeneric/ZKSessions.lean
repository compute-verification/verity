module

public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Claims.ZK.Compose

/-! An outer phase of zero-knowledge sessions run in turn is zero knowledge per outcome within the sum of the sessions' errors; definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

structure ZkSess where
  WS : Type
  RS : Msg → Rel WS
  kS : Msg → ℕ
  NS : Msg → Type
  ΩS : Msg → Type
  [nFin : ∀ x, Fintype (NS x)]
  [nNe : ∀ x, Nonempty (NS x)]
  [oFin : ∀ x, Fintype (ΩS x)]
  [oNe : ∀ x, Nonempty (ΩS x)]
  PS : WS → (x : Msg) → Prover (NS x × ΩS x)
  StS : Type
  outS : Msg → List (Msg × Out) → StS
  CS : Msg → Type
  [cFin : ∀ x, Fintype (CS x)]
  [cNe : ∀ x, Nonempty (CS x)]
  progS : (x : Msg) → Msg → StS → CS x → SP (List (Msg × Out))

attribute [instance] ZkSess.nFin ZkSess.nNe ZkSess.oFin ZkSess.oNe ZkSess.cFin ZkSess.cNe

namespace ZkSess

def simAt (S : ZkSess) (x : Msg) : SimOut S.StS where
  C _ := S.CS x
  prog xs o c := S.progS x xs o c

def ZK (S : ZkSess) (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) : Prop :=
  ∀ x, ZKOut (fun _ => S.NS x) (fun _ => S.ΩS x) H S.kS (fun w _ => S.PS w x) (S.RS x) S.outS (S.simAt x) (εS x)

def tapes (S : ZkSess) (x : Msg) : ℕ → Type
  | 0 => Unit
  | m + 1 => (S.NS x × S.ΩS x) × tapes S x m

def coins (S : ZkSess) (x : Msg) : ℕ → Type
  | 0 => Unit
  | m + 1 => S.CS x × coins S x m

instance tapesFin (S : ZkSess) (x : Msg) : ∀ m, Fintype (S.tapes x m)
  | 0 => inferInstanceAs (Fintype Unit)
  | m + 1 => letI := tapesFin S x m; inferInstanceAs (Fintype ((S.NS x × S.ΩS x) × S.tapes x m))

instance tapesNe (S : ZkSess) (x : Msg) : ∀ m, Nonempty (S.tapes x m)
  | 0 => inferInstanceAs (Nonempty Unit)
  | m + 1 => letI := tapesNe S x m; inferInstanceAs (Nonempty ((S.NS x × S.ΩS x) × S.tapes x m))

instance coinsFin (S : ZkSess) (x : Msg) : ∀ m, Fintype (S.coins x m)
  | 0 => inferInstanceAs (Fintype Unit)
  | m + 1 => letI := coinsFin S x m; inferInstanceAs (Fintype (S.CS x × S.coins x m))

instance coinsNe (S : ZkSess) (x : Msg) : ∀ m, Nonempty (S.coins x m)
  | 0 => inferInstanceAs (Nonempty Unit)
  | m + 1 => letI := coinsNe S x m; inferInstanceAs (Nonempty (S.CS x × S.coins x m))

def ofZKOut {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]
    (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) : ZkSess where
  WS := W
  RS x xs w := xs ∈ Xs x ∧ R xs w
  kS := k
  NS _ := N
  ΩS _ := Ω
  PS w _ := P w
  StS := St
  outS := out
  CS x := (v : Xs x) → S.C v
  progS x xs o c := if h : xs ∈ Xs x then S.prog xs o (c ⟨xs, h⟩) else .ret []

end ZkSess

def vUnder {Θ : Type} (Vs : VStar Θ) (xO : Msg) (pre : List Msg) (k : ℕ) : VStar Θ where
  k _ := k
  next _ as θ := Vs.next xO (pre ++ as) θ

structure Sessions (S : ZkSess) where
  n : Msg → ℕ
  xs : Msg → Msg → ℕ → Msg
  ko : Msg → ℕ

namespace Sessions

variable {S : ZkSess} (L : Sessions S)

def rounds (x xO : Msg) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | m + 1, i => S.kS (L.xs x xO i) + rounds x xO m (i + 1)

def Shaped : Prop := ∀ x xO, L.rounds x xO (L.n x) 0 = L.ko x

def cnext (x xO : Msg) (wo : ℕ → S.WS) : (m i : ℕ) → List (Msg × Msg) → S.tapes x m → OC Msg
  | 0, _, _, _ => .ret []
  | m + 1, i, hist, t =>
    if hist.length < S.kS (L.xs x xO i) then (S.PS (wo i) x).next (L.xs x xO i) hist t.1
    else cnext x xO wo m (i + 1) (hist.drop (S.kS (L.xs x xO i))) t.2

def PO (wo : ℕ → S.WS) (x : Msg) : Prover (S.tapes x (L.n x) × Unit) where
  next xO hist t := L.cnext x xO wo (L.n x) 0 hist t.1

def RO (x : Msg) : Rel (ℕ → S.WS) := fun xO wo => ∀ i < L.n x, S.RS x (L.xs x xO i) (wo i)

def outs (x xO : Msg) : (m i : ℕ) → List (Msg × Out) → List (Option S.StS)
  | 0, _, _ => []
  | m + 1, i, τ =>
    some (S.outS (L.xs x xO i) (τ.take (S.kS (L.xs x xO i)))) ::
      if ZKMal.goesOn (S.kS (L.xs x xO i)) (τ.take (S.kS (L.xs x xO i))) then
        outs x xO m (i + 1) (τ.drop (S.kS (L.xs x xO i)))
      else List.replicate m none

def outO (x xO : Msg) (τ : List (Msg × Out)) : List (Option S.StS) := L.outs x xO (L.n x) 0 τ

def csim (x xO : Msg) : (m i : ℕ) → List (Option S.StS) → List (Msg × Out) → S.coins x m → SP (List (Msg × Out))
  | 0, _, _, ms, _ => .ret ms
  | m + 1, i, some a :: os, ms, c =>
    spBind (spUnder (ms.map Prod.fst) (S.progS x (L.xs x xO i) a c.1)) fun τ =>
      if ZKMal.goesOn (S.kS (L.xs x xO i)) τ then csim x xO m (i + 1) os (ms ++ τ) c.2 else .ret (ms ++ τ)
  | _ + 1, _, _, ms, _ => .ret ms

def progO (x xO : Msg) (o : List (Option S.StS)) (c : S.coins x (L.n x)) : SP (List (Msg × Out)) :=
  L.csim x xO (L.n x) 0 o [] c

def simO (x : Msg) : SimOut (List (Option S.StS)) where
  C _ := S.coins x (L.n x)
  prog xO o c := L.progO x xO o c

open Classical in
noncomputable def errAt (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ)
    (x xO : Msg) (wo : ℕ → S.WS) (Θ : Type) (Vs : VStar Θ) (θ : Θ) :
    (m i : ℕ) → List (Msg × Out) → List (Option S.StS) → ℝ
  | 0, _, _, _ => 0
  | m + 1, i, ms, some a :: os =>
    εS x (L.xs x xO i) (wo i) Θ (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ a +
      Ex fun t : S.NS x × S.ΩS x =>
        if S.outS (L.xs x xO i) (runStar H (S.PS (wo i) x)
              (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) (L.xs x xO i) t θ).msgs = a ∧
            ZKMal.goesOn (S.kS (L.xs x xO i)) (runStar H (S.PS (wo i) x)
              (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) (L.xs x xO i) t θ).msgs = true then
          errAt H εS x xO wo Θ Vs θ m (i + 1)
            (ms ++ (runStar H (S.PS (wo i) x) (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i)))
              (L.xs x xO i) t θ).msgs) os
        else 0
  | _ + 1, _, _, _ => 0

noncomputable def err (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ)
    (x xO : Msg) (wo : ℕ → S.WS) (Θ : Type) (Vs : VStar Θ) (θ : Θ) (o : List (Option S.StS)) : ℝ :=
  L.errAt H εS x xO wo Θ Vs θ (L.n x) 0 [] o

end Sessions

structure FirstPart (W : Type) (Lv : Leaves) (WO : Type) where
  nh : Msg → ℕ
  nl : Msg → ℕ → ℕ
  frame : Msg → List (Msg × Out) → List Msg → OC Msg
  Ωp : Msg → Type
  [pFin : ∀ x, Fintype (Ωp x)]
  [pNe : ∀ x, Nonempty (Ωp x)]
  msg : W → (x : Msg) → Ωp x → List (Msg × Msg) → OC (List Msg)
  xo : Msg → List (Msg × Out) → Msg
  wo : W → (x : Msg) → Ωp x → (Fin (∑ i ∈ Finset.range (nh x), nl x i) → Lv.Salt) → List (Msg × Out) → WO

attribute [instance] FirstPart.pFin FirstPart.pNe

def ZkRelease.ofSessions {W : Type} {Lv : Leaves} {S : ZkSess} (F : FirstPart W Lv (ℕ → S.WS)) (L : Sessions S) :
    ZkRelease W Lv where
  nh := F.nh
  nl := F.nl
  frame := F.frame
  Ωp := F.Ωp
  msg := F.msg
  ko := L.ko
  xo := F.xo
  WO := ℕ → S.WS
  wo := F.wo
  RO := L.RO
  NO x := S.tapes x (L.n x)
  ΩO _ := Unit
  PO := L.PO
  St := List (Option S.StS)
  outO := L.outO
  CO x := S.coins x (L.n x)
  progO := L.progO

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric

def GenericSessionsZK : Prop :=
  ∀ (H : Msg → Msg) (S : ZkSess) (L : Sessions S)
    (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ),
    S.ZK H εS → L.Shaped →
      ∀ x, ZKOut (fun _ => S.tapes x (L.n x)) (fun _ => Unit) H (fun _ => L.ko x) (fun wo _ => L.PO wo x) (L.RO x)
        (L.outO x) (L.simO x) (L.err H εS x)

def GenericSessionsZKCost : Prop :=
  ∀ (H : Msg → Msg) (S : ZkSess) (L : Sessions S) (cS : Msg → ℕ),
    (∀ x xs o (c : S.CS x) ask, ((S.progS x xs o c).eval H ask).2 ≤ cS x) →
      ∀ x xO o (c : S.coins x (L.n x)) ask, ((L.progO x xO o c).eval H ask).2 ≤ L.n x * cS x

def GenericRecursiveSessionsZK : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (S : ZkSess) (L : Sessions S) (F : FirstPart W Lv (ℕ → S.WS))
    (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) (δ : ℝ),
    Lv.Hides δ → S.ZK H εS → L.Shaped →
      ZKOut (ZkRelease.ofSessions F L).N (ZkRelease.ofSessions F L).Ω H (ZkRelease.ofSessions F L).k
        (ZkRelease.ofSessions F L).prover (fun _ _ => True) (ZkRelease.ofSessions F L).outcome
        (ZkRelease.ofSessions F L).sim ((ZkRelease.ofSessions F L).err H (L.err H εS) δ)

def GenericRecursiveSessionsZKCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (S : ZkSess) (L : Sessions S) (F : FirstPart W Lv (ℕ → S.WS))
    (cF cS : Msg → ℕ),
    (∀ x pre ls, ((F.frame x pre ls).eval H).2.length ≤ cF x) →
    (∀ x xs o (c : S.CS x) ask, ((S.progS x xs o c).eval H ask).2 ≤ cS x) →
    ZKMal.SimCostLe H ((ZkRelease.ofSessions F L).sim.given none) (fun x => F.nh x * (cF x + 1) + L.ko x) ∧
      ∀ o, ZKMal.SimCostLe H ((ZkRelease.ofSessions F L).sim.given (some o))
        (fun x => F.nh x * (cF x + 1) + L.n x * cS x)

def GenericSessionsOfZKOut : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω]
    [Nonempty Ω] (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) (ε : Msg → W → (Θ : Type) → VStar Θ → Θ → St → ℝ),
    ZKOut (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R out S ε →
      (ZkSess.ofZKOut R k N Ω P out S Xs).ZK H fun _ => ε

def GenericSessionsOfZKOutCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω]
    [Nonempty Ω] (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) (cS : Msg → ℕ),
    (∀ x, ∀ xs ∈ Xs x, ∀ o (c : S.C xs) ask, ((S.prog xs o c).eval H ask).2 ≤ cS x) →
      ∀ x xs o (c : (ZkSess.ofZKOut R k N Ω P out S Xs).CS x) ask,
        (((ZkSess.ofZKOut R k N Ω P out S Xs).progS x xs o c).eval H ask).2 ≤ cS x

end Flock.Guarantees
