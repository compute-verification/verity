module

public import Proofs.Flock.Recursive.Multi
public import Proofs.Flock.SampledGeneric.FlockHide
public import Proofs.Flock.SampledGeneric.FlockZk
public import Proofs.Flock.Soundness.CROnly.Hiding
public import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-! The objects the zero-knowledge statements of `HidingStatements.lean` read. -/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-- The error may read the witness and the verifier; `Zk.ZK` is the case where it reads the statement alone. -/
def Zk.ZKAt {Stmt Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop)
    (ε : Stmt → Wit → VS → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Close (Zs.real x w v) (Zs.sim x v) (ε x w v)

def Zk.after {Stmt Stmt' Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (g : Stmt' → Stmt) :
    Zk Stmt' Wit (Stmt' → VS) View :=
  ⟨fun x w v => Zs.real (g x) w (v x), fun x v => Zs.sim (g x) (v x)⟩

namespace Hider

variable {Tr Tr' Rnd Com Com' : Type}

def map (H : Hider Tr Rnd Com) (g : Com → Com') : Hider Tr Rnd Com' := ⟨fun Xt r => g (H.commit Xt r)⟩

def comap (H : Hider Tr Rnd Com) (t : Tr' → Tr) : Hider Tr' Rnd Com := ⟨fun Xt r => H.commit (t Xt) r⟩

end Hider

def hm96Hider (ι : Type) {Sa Bs C D : Type} [AddCommGroup Bs] (My : Sa → Bs) (c : Sa → C) (Hf : Bs × C → D) :
    Hider (ι → Bs) (ι → Sa) (ι → D) :=
  ⟨fun Xt y i => Hf (Xt i + My (y i), c (y i))⟩

def hm96Exe (ι : Type) : Hider (ι → ByteArray) (ι → Discharge.ZkView.Salt) (ι → ByteArray) :=
  ⟨fun Xt y i => _root_.Flock.Hm96.Default512.commitString (Xt i) (Discharge.ZkView.saltBytes (y i))⟩

open Classical in
noncomputable def hm96Gap : ℝ≥0∞ :=
  ENNReal.ofReal (CROnly.hidingGap Discharge.ZkView.My Discharge.ZkView.saltHash)

structure MalVerifier (k : Claims.Msg → ℕ) (Θ : Type) where
  Vs : Claims.VStar Θ
  θ : Θ
  hk : ∀ x, Vs.k x = k x

section Mal

variable {N Ω W : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]

noncomputable def zkOfMal (H : Claims.Msg → Claims.Msg) (k : Claims.Msg → ℕ) (P : W → Claims.Prover (N × Ω))
    (S : Claims.Simulator) (Θ : Type) :
    Zk Claims.Msg W (MalVerifier k Θ) (List (Claims.Msg × Claims.Out)) where
  real x w v := ⟨.coin (N × Ω) fun ω => .ret (Claims.runStar H (P w) v.Vs x ω v.θ).msgs, fun _ => PUnit.unit⟩
  sim x v := ⟨.coin (S.C x) fun c => .ret (S.out H v.Vs v.θ x c), fun _ => PUnit.unit⟩

noncomputable def malErr (H : Claims.Msg → Claims.Msg) {k : Claims.Msg → ℕ} (P : W → Claims.Prover (N × Ω))
    (εzk γ : Claims.Msg → ℝ) {Θ : Type} (x : Claims.Msg) (w : W) (v : MalVerifier k Θ) : ℝ≥0∞ :=
  ENNReal.ofReal (εzk x + γ x * (Claims.Fork H (P w) v.Vs v.θ x).Pr H)

end Mal

noncomputable def Exp.ex {V : Type} (e : Exp V) (f : V → ℝ≥0∞) : ℝ≥0∞ := Discharge.Recursion.exG f e.g e.s

def Exp.map {V W : Type} (e : Exp V) (f : V → W) : Exp W := ⟨e.g.map f, Strategy.map f e.g e.s⟩

section Sessions

variable {Stmt Wit VS View : Type}

def Zk.seqReal (Zs : Zk Stmt Wit VS View) : List (Stmt × Wit) → (List View → VS) → Exp (List View)
  | [], _ => ⟨.ret [], PUnit.unit⟩
  | xw :: l, v => (Zs.real xw.1 xw.2 (v [])).andThen fun a => (Zk.seqReal Zs l fun h => v (a :: h)).map (a :: ·)

def Zk.seqSim (Zs : Zk Stmt Wit VS View) : List Stmt → (List View → VS) → Exp (List View)
  | [], _ => ⟨.ret [], PUnit.unit⟩
  | x :: l, v => (Zs.sim x (v [])).andThen fun a => (Zk.seqSim Zs l fun h => v (a :: h)).map (a :: ·)

noncomputable def Zk.seqErr (Zs : Zk Stmt Wit VS View) (ε : Stmt → Wit → VS → ℝ≥0∞) :
    List (Stmt × Wit) → (List View → VS) → ℝ≥0∞
  | [], _ => 0
  | xw :: l, v =>
    ε xw.1 xw.2 (v []) + (Zs.real xw.1 xw.2 (v [])).ex fun a => Zk.seqErr Zs ε l fun h => v (a :: h)

def Zk.seq (Zs : Zk Stmt Wit VS View) : Zk (List Stmt) (List Wit) (List Stmt → List View → VS) (List View) where
  real xs ws v := Zs.seqReal (xs.zip ws) (v xs)
  sim xs v := Zs.seqSim xs (v xs)

end Sessions

end FlockSoundness.SampledGeneric
