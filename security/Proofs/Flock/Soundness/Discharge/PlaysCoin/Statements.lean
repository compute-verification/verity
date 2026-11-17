module

public import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
public import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Mathlib.Combinatorics.Matroid.Init

/-! The statements of the coin half of `DecodesAll` and `DecodesOn`. -/

@[expose] public section

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit
  Audit.Partition

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

def coinArr {W F : Type} [Inhabited W] (ι : W → F) (c : Array W) : Msg :=
  if c.size = 1 then ⟨F, ι (c)[0]!⟩ else ⟨Fin c.size → F, fun i => ι (c)[i.val]!⟩

def consHead (m : Msg) : List (List Msg) → List (List Msg)
  | [] => [[m]]
  | g :: gs => (m :: g) :: gs

def cut : List Exec.Kind → List Msg → List (List Msg)
  | [], _ => [[]]
  | .msg _ :: σ, m :: ms => consHead m (cut σ ms)
  | .msg _ :: _, [] => [[]]
  | .coin _ :: σ, ms => [] :: cut σ ms

def coinPat : List Exec.Kind → List Bool
  | [] => []
  | .msg _ :: σ => false :: coinPat σ
  | .coin _ :: σ => true :: coinPat σ

def groupPat (ns : List ℕ) (t : ℕ) : List Bool :=
  ns.flatMap (fun k => List.replicate k false ++ [true]) ++ List.replicate t false

def roundGroups (dec : ℕ → ByteArray → List Msg) : ℕ → List Round → List (List Msg)
  | _, [] => []
  | j, rd :: rs => dec j (rd.bytes.getD ByteArray.empty) :: roundGroups dec (j + 1) rs

def RepRun {P : Type} (g : Game P) (σ : List Exec.Kind) (dec : ℕ → ByteArray → List Msg) (rs : List Round)
    (tail : List Msg) (p : P) : Prop :=
  (∀ rd ∈ rs, rd.coins.size = rd.n) ∧ ∃ ms : List Msg, Plays fOf g ms (rs.map (·.coins)) p ∧
    cut (σ.take (ms.length + rs.length)) ms = roundGroups dec 0 rs ++ [tail]

def SpineAt {P : Type} (g : Game P) (σ : List Exec.Kind) (p : P) : Prop :=
  ∀ ts : List Step, PlaysT g ts p → ts.map kindOf <+: σ

noncomputable abbrev repSpine (D : Type) {mPts : ℕ} (T : TabSpec mPts) (pts : Fin 2 → Fin mPts → 𝔽) : List Exec.Kind :=
  repShape 𝔽 𝕂 D T.S.m T.S.kLog (extraClaims T.S T.ptLocal mPts pts).length T.sch

def ToTExact : Prop :=
  ∀ {W F : Type} [Inhabited W] (ι : W → F) {α : Type} {g : Game α} {ms : List Msg} {cs : List (Array W)} {a : α},
    Plays ι g ms cs a → ∃ ts : List Step, PlaysT g ts a ∧ ts.filterMap Sum.getLeft? = ms ∧
      ts.filterMap Sum.getRight? = cs.map (coinArr ι)

def CoinsOfRounds : Prop :=
  ∀ {α : Type} {g : Game α} {ms : List Msg} {rs : List Round} {a : α}, (∀ rd ∈ rs, rd.coins.size = rd.n) →
    Plays fOf g ms (rs.map (·.coins)) a → ∃ ts : List Step, PlaysT g ts a ∧ ts.filterMap Sum.getLeft? = ms ∧
      ts.filterMap Sum.getRight? = rs.map (coinMsg true)

def TapeOfGroups : Prop :=
  ∀ (dec : ℕ → ByteArray → List Msg) (j : ℕ) (rs : List Round) (tail : List Msg) (ts : List Step),
    ts.filterMap Sum.getRight? = rs.map (coinMsg true) →
    cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) = roundGroups dec j rs ++ [tail] →
    ts = tapeFrom dec (fun _ => true) j rs ++ tail.map .inl

def CutOfPat : Prop :=
  ∀ (σ : List Exec.Kind) (G : List (List Msg)) (t : List Msg),
    coinPat σ = groupPat (G.map List.length) t.length → cut σ (G.flatten ++ t) = G ++ [t]

def DecodesOfRuns : Prop :=
  ∀ {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
    (pg : OutT → Prop) (opn : Array ByteArray → Op) (dec : ℕ → ByteArray → List Msg)
    (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (σ : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → List Exec.Kind),
    (∀ pts r, Spine (G pts r) (σ pts r)) →
    (∀ o, pL o → ∃ pend : P × P, RepRun (G o.pts 0) (σ o.pts 0) dec o.rep0 (fin 0 o.proofs) pend.1 ∧
      RepRun (G o.pts 1) (σ o.pts 1) dec o.rep1 (fin 1 o.proofs) pend.2 ∧
      pg (out (capOf o.rootB) pend (opn o.proofs))) →
    Decodes out pL pg opn dec (fun _ => true) fin capOf G

def DecodesOfRunsAt : Prop :=
  ∀ {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
    (pg : OutT → Prop) (opn : Array ByteArray → Op) (dec : ℕ → ByteArray → List Msg)
    (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P),
    (∀ o, pL o → ∃ pend : P × P, ∃ σ₀ σ₁ : List Exec.Kind,
      SpineAt (G o.pts 0) σ₀ pend.1 ∧ RepRun (G o.pts 0) σ₀ dec o.rep0 (fin 0 o.proofs) pend.1 ∧
      SpineAt (G o.pts 1) σ₁ pend.2 ∧ RepRun (G o.pts 1) σ₁ dec o.rep1 (fin 1 o.proofs) pend.2 ∧
      pg (out (capOf o.rootB) pend (opn o.proofs))) →
    Decodes out pL pg opn dec (fun _ => true) fin capOf G

def DecodesAllOfRuns : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)),
    (∀ S R o, Accepts I (wr S R o) → ∃ pend : PendL 𝔽 D × PendL 𝔽 D,
      RepRun (repsOf (D := D) execArith (T S R) o.pts 0) (repSpine D (T S R) o.pts) (dec S R) o.rep0
        (fin S R 0 o.proofs) pend.1 ∧
      RepRun (repsOf (D := D) execArith (T S R) o.pts 1) (repSpine D (T S R) o.pts) (dec S R) o.rep1
        (fin S R 1 o.proofs) pend.2 ∧
      (tableOutL H E Lf (capOf S R o.rootB 0) pend (opn S R o.proofs)).accepted = true) →
    DecodesAll H E Lf T I wr opn dec (fun _ _ _ => true) fin capOf

def DecodesOnOfRuns : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (L : SamplingDesign n),
    (∀ ω R o, Accepts I (wr (L.draw ω) R o) → ∃ pend : PendL 𝔽 D × PendL 𝔽 D,
      RepRun (repsOf (D := D) execArith (T (L.draw ω) R) o.pts 0) (repSpine D (T (L.draw ω) R) o.pts)
        (dec (L.draw ω) R) o.rep0 (fin (L.draw ω) R 0 o.proofs) pend.1 ∧
      RepRun (repsOf (D := D) execArith (T (L.draw ω) R) o.pts 1) (repSpine D (T (L.draw ω) R) o.pts)
        (dec (L.draw ω) R) o.rep1 (fin (L.draw ω) R 1 o.proofs) pend.2 ∧
      (tableOutL H E Lf (capOf (L.draw ω) R o.rootB 0) pend (opn (L.draw ω) R o.proofs)).accepted = true) →
    DecodesOn H E Lf T I wr opn dec (fun _ _ _ => true) fin capOf L

end FlockSoundness.Discharge.PlaysCoin
