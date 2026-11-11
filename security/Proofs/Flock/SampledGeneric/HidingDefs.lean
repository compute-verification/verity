module

public import Proofs.Flock.Recursive.Multi
public import Proofs.Flock.SampledGeneric.FlockHide
public import Proofs.Flock.SampledGeneric.FlockZk
public import Proofs.Flock.Soundness.CROnly.Hiding
public import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-!
# The sampled audit's zero knowledge from a hiding trace commitment: its objects

`GenericSampledZK` takes a proof system that is zero knowledge at an error of the statement alone and a commitment
that hides, as a hypothesis. This file defines what the statements of `HidingStatements.lean` read beyond that:

* `Zk.ZKAt`: zero knowledge, both ways, at an error that may read the statement, the witness and the verifier, as the
  claims core's `ZKmalR` has it (its `Fork` term is the real game's collision finder, built from both).
* `Zk.after`: the proof system at a statement built from what the verifier read, against a verifier that chooses how
  it plays from all of it: in the audit, from the whole commitment and its sample.
* `Hider.map`, `Hider.comap`: a commitment read through a function (leaf digests, the trees over them, their roots),
  and a commitment to a function of the trace (a row's inner digest).
* `hm96Hider`: hm96's commitment to a trace of `N` leaves as a `Hider`: the prover's coins are one salt per leaf, and
  leaf `i` is `Hf (Xt i + My (y i), c (y i))`. `hm96Exe`: the executable's commit strings `b ‖ c` of `N` inner digests
  (`Flock.Hm96.Default512.commitString`), and `hm96Gap`, its per-leaf hiding gap at the pinned key.
* `MalVerifier`, `zkOfMal`, `malErr`: a claims-core record (a prover on a nonce and a rest, a black-box simulator)
  as a `Zk` against the malicious verifiers of its rounds, with `ZKmalR`'s error.
* `Exp.ex`, `Exp.map`, `Zk.seqReal`, `Zk.seqSim`, `Zk.seqErr`, `Zk.seq`: sessions in turn against one verifier that
  carries its state from session to session, as one proof system on lists of statements.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-! ## Zero knowledge at an error that reads the witness and the verifier -/

/-- **Zero knowledge at the error `ε x w v`**: for every statement and witness in the relation and every verifier, the
honest prover's view is `ε x w v`-close to the simulator's, both ways. The error may read the witness and the verifier;
`Zk.ZK` is the case where it reads the statement alone. -/
def Zk.ZKAt {Stmt Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop)
    (ε : Stmt → Wit → VS → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Close (Zs.real x w v) (Zs.sim x v) (ε x w v)

/-- **The proof system at the statement `g x`, against a verifier that read `x`**: for the audit, `x` is the
commitment and the sample and `g` builds the session's statement from them, and the verifier chooses how it plays the
session from all of `x`, the whole commitment included, whatever of it the session's statement keeps. -/
def Zk.after {Stmt Stmt' Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (g : Stmt' → Stmt) :
    Zk Stmt' Wit (Stmt' → VS) View :=
  ⟨fun x w v => Zs.real (g x) w (v x), fun x v => Zs.sim (g x) (v x)⟩

/-! ## Commitments read through a function -/

namespace Hider

variable {Tr Tr' Rnd Com Com' : Type}

/-- **The commitment read through `g`**: whatever is computed from it and public data (leaf digests, the trees over
them, their roots). -/
def map (H : Hider Tr Rnd Com) (g : Com → Com') : Hider Tr Rnd Com' := ⟨fun Xt r => g (H.commit Xt r)⟩

/-- **The commitment to `t Xt`**: a commitment to a function of the trace (each row's inner digest). -/
def comap (H : Hider Tr Rnd Com) (t : Tr' → Tr) : Hider Tr' Rnd Com := ⟨fun Xt r => H.commit (t Xt) r⟩

end Hider

/-! ## hm96's trace commitment -/

/-- **hm96's commitment to a trace of `N = |ι|` leaves.** The trace is a digest `Xt i` at each leaf, the prover's coins
are one salt `y i` per leaf, and leaf `i` of the commitment is `Hf (Xt i + My (y i), c (y i))`: with `My` the salt's
mask under the key, `c` the salt's hash and `+` XOR, the pair is hm96's commit string `(b, c)`, and `Hf` is what is
published of it (the pair itself, or the tree leaf's digest). -/
def hm96Hider (ι : Type) {Sa Bs C D : Type} [AddCommGroup Bs] (My : Sa → Bs) (c : Sa → C) (Hf : Bs × C → D) :
    Hider (ι → Bs) (ι → Sa) (ι → D) :=
  ⟨fun Xt y i => Hf (Xt i + My (y i), c (y i))⟩

/-- **The executable's commit strings of `N = |ι|` inner digests**: leaf `i` is
`Flock.Hm96.Default512.commitString (Xt i) (y i)`, the `b ‖ c` the verifier computes the tree leaf from, at the pinned
key, under the salt `y i` (192 bytes, as its bits). -/
def hm96Exe (ι : Type) : Hider (ι → ByteArray) (ι → Discharge.ZkView.Salt) (ι → ByteArray) :=
  ⟨fun Xt y i => _root_.Flock.Hm96.Default512.commitString (Xt i) (Discharge.ZkView.saltBytes (y i))⟩

open Classical in
/-- **hm96's per-leaf hiding gap at the pinned key**: the statistical distance of `(M·y, c(y))` from `(U, c(y))` for a
uniform salt `y` and a uniform `U` (`CROnly.hidingGap`), as a number in `[0, ∞]`. -/
noncomputable def hm96Gap : ℝ≥0∞ :=
  ENNReal.ofReal (CROnly.hidingGap Discharge.ZkView.My Discharge.ZkView.saltHash)

/-! ## A claims-core record as a `Zk` -/

/-- **A malicious verifier of `k`'s rounds**: a `V*` and its tape. -/
structure MalVerifier (k : Claims.Msg → ℕ) (Θ : Type) where
  Vs : Claims.VStar Θ
  θ : Θ
  hk : ∀ x, Vs.k x = k x

section Mal

variable {N Ω W : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]

/-- **A claims-core record as a `Zk`**: the honest view is the transcript of the prover `P w`, on a uniform nonce and
rest, against `v`'s `V*` at its tape; the simulator's is `S`'s transcript against the same `V*`, on `S`'s uniform
coins. -/
noncomputable def zkOfMal (H : Claims.Msg → Claims.Msg) (k : Claims.Msg → ℕ) (P : W → Claims.Prover (N × Ω))
    (S : Claims.Simulator) (Θ : Type) :
    Zk Claims.Msg W (MalVerifier k Θ) (List (Claims.Msg × Claims.Out)) where
  real x w v := ⟨.coin (N × Ω) fun ω => .ret (Claims.runStar H (P w) v.Vs x ω v.θ).msgs, fun _ => PUnit.unit⟩
  sim x v := ⟨.coin (S.C x) fun c => .ret (S.out H v.Vs v.θ x c), fun _ => PUnit.unit⟩

/-- **`ZKmalR`'s error at `(x, w)` against `v`**: `ε_zk(x) + γ(x)·Pr[Fork]`, with `Fork` the real game's collision
finder, two runs of `P w` against `v`'s `V*` on one nonce. -/
noncomputable def malErr (H : Claims.Msg → Claims.Msg) {k : Claims.Msg → ℕ} (P : W → Claims.Prover (N × Ω))
    (εzk γ : Claims.Msg → ℝ) {Θ : Type} (x : Claims.Msg) (w : W) (v : MalVerifier k Θ) : ℝ≥0∞ :=
  ENNReal.ofReal (εzk x + γ x * (Claims.Fork H (P w) v.Vs v.θ x).Pr H)

end Mal

/-! ## Sessions in turn -/

/-- `E[f]` in the experiment. -/
noncomputable def Exp.ex {V : Type} (e : Exp V) (f : V → ℝ≥0∞) : ℝ≥0∞ := Discharge.Recursion.exG f e.g e.s

/-- The experiment with its view relabelled. -/
def Exp.map {V W : Type} (e : Exp V) (f : V → W) : Exp W := ⟨e.g.map f, Strategy.map f e.g e.s⟩

section Sessions

variable {Stmt Wit VS View : Type}

/-- **The honest views of sessions in turn.** Session `j` proves the `j`-th statement from the `j`-th witness against
the verifier `v h`, with `h` the views of the sessions before it, first first: what the verifier carries from session
to session is what it saw. The view is the list of the sessions' views. -/
def Zk.seqReal (Zs : Zk Stmt Wit VS View) : List (Stmt × Wit) → (List View → VS) → Exp (List View)
  | [], _ => ⟨.ret [], PUnit.unit⟩
  | xw :: l, v => (Zs.real xw.1 xw.2 (v [])).andThen fun a => (Zk.seqReal Zs l fun h => v (a :: h)).map (a :: ·)

/-- **The simulator's views of sessions in turn**: session `j`'s simulator at the `j`-th statement against `v h`, with
`h` the simulated views before it. -/
def Zk.seqSim (Zs : Zk Stmt Wit VS View) : List Stmt → (List View → VS) → Exp (List View)
  | [], _ => ⟨.ret [], PUnit.unit⟩
  | x :: l, v => (Zs.sim x (v [])).andThen fun a => (Zk.seqSim Zs l fun h => v (a :: h)).map (a :: ·)

/-- **The sessions' error**: the sum over the sessions of session `j`'s error at its statement and witness against the
verifier where the honest run reaches it, averaged over the honest views of the sessions before it. -/
noncomputable def Zk.seqErr (Zs : Zk Stmt Wit VS View) (ε : Stmt → Wit → VS → ℝ≥0∞) :
    List (Stmt × Wit) → (List View → VS) → ℝ≥0∞
  | [], _ => 0
  | xw :: l, v =>
    ε xw.1 xw.2 (v []) + (Zs.real xw.1 xw.2 (v [])).ex fun a => Zk.seqErr Zs ε l fun h => v (a :: h)

/-- **Sessions in turn as one proof system**: its statement is the list of the sessions' statements, its witness the
list of their witnesses, and its verifier reads that list of statements and, at each session, the views of the sessions
before it. -/
def Zk.seq (Zs : Zk Stmt Wit VS View) : Zk (List Stmt) (List Wit) (List Stmt → List View → VS) (List View) where
  real xs ws v := Zs.seqReal (xs.zip ws) (v xs)
  sim xs v := Zs.seqSim xs (v xs)

end Sessions

end FlockSoundness.SampledGeneric
