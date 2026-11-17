module

public import Properties.Poub.Remote
public import Definitions.Core.InProofs
public import Mathlib.Probability.ProbabilityMassFunction.Monad
public import Proofs.Properties.ResourceAccounting.InProofs
public import Proofs.Properties.Poub.Remote.Schedule

/-! PoUB's read session inside proofs: an accepted run gives openings `checkWindow` accepts (`SpanGives`), a statement
of any run that meets its hypotheses and of no deployed run. -/

@[expose] public section

open Poub

namespace Properties.Poub.InProofs

open Properties.Poub Verity.InProofs Verity.Game
open scoped ENNReal

def OpeningOk (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (sampled : Array (Nat × Nat)) (k : Fin sampled.size) (o : Opening) : Prop :=
  checkOpening reg digest appends w (receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2 o = none

/-- An accepted run gives openings `checkWindow` accepts, except `ε`, for any run meeting its hypotheses; it says
nothing of the deployed run, where nothing proves `Gives`. -/
def SpanGives : Prop :=
  ∀ (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord) (sampled : Array (Nat × Nat))
    (Ω : Type) (μ : PMF Ω) (V : Type) (r : Run Ω sampled.size fun _ => V) (right : Fin sampled.size → V → Prop)
    (decode : V → Opening) (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    right ≤ (fun k v => OpeningOk reg digest appends w sampled k (decode v)) →
    prPMF μ (fun ω => r.accepts ω ∧
        checkWindow reg digest appends w sampled (Array.ofFn fun k => decode (r.ports k ω)) ≠ Verdict.accept) ≤ ε

def oneSample (o : Opening) : Run Unit (#[((0 : Nat), (0 : Nat))].size) fun _ => Opening :=
  ⟨fun _ => True, fun _ _ => o, fun _ => []⟩

/-- `SpanGives`' hypotheses hold where the check accepts: at the deployed registration shape, a window of one pass over
the whole store and an opening of position 0, at `ε = 0`. -/
def SpanGivesHonest : Prop :=
  ∃ (reg : Registration) (w : WindowRecord) (o : Opening),
    Registration.make? (zeros 1) 130023424 reg.payloadRoot (zeros Codec.SEED_BYTES) ⟨Codec.P1024, 3, true⟩ .sketch
        false LAYOUT 16 = .ok reg ∧
      Gives (PMF.pure ()) (oneSample o) (fun k v => OpeningOk reg reg.digest #[] w #[(0, 0)] k v) 1 0 ∧
      checkWindow reg reg.digest #[] w #[(0, 0)] (Array.ofFn fun k => (oneSample o).ports k ()) = .accept

/-- `SpanGives`' conclusion fails without its hypotheses: at the same registration and window, a run that always accepts
an opening `checkWindow` refuses gives the event probability `1`. -/
def SpanGivesNontrivial : Prop :=
  ∃ (reg : Registration) (w : WindowRecord) (o : Opening),
    ¬ prPMF (PMF.pure ()) (fun ω => (oneSample o).accepts ω ∧
        checkWindow reg reg.digest #[] w #[(0, 0)] (Array.ofFn fun k => (oneSample o).ports k ω) ≠ Verdict.accept) ≤ 0

end Properties.Poub.InProofs

open Poub

namespace Properties.Poub.InProofs.SecurityProofs

open Properties.Poub Properties.Poub.InProofs
open Verity.InProofs Properties.ResourceAccounting.InProofs Verity.Game

theorem checkWindow_ofFn_accept (reg : Registration) (digest : ByteArray) (appends : Array Append)
    (w : WindowRecord) (sampled : Array (Nat × Nat)) (o : Fin sampled.size → Opening)
    (h : ∀ k, OpeningOk reg digest appends w sampled k (o k)) :
    checkWindow reg digest appends w sampled (Array.ofFn o) = .accept := by
  rw [checkWindow_eq]
  have hnone : (List.range sampled.size).findSome? (rejectAt reg digest appends w sampled (Array.ofFn o)) = none := by
    rw [List.findSome?_eq_none_iff]
    intro k hk
    have hk' := List.mem_range.1 hk
    have hok := h ⟨k, hk'⟩
    unfold OpeningOk at hok
    unfold rejectAt
    have hd : sampled[k]! = sampled[k] := getElem!_pos sampled k hk'
    have ho : (Array.ofFn o)[k]! = o ⟨k, hk'⟩ := by
      rw [getElem!_pos (Array.ofFn o) k (by simpa using hk'), Array.getElem_ofFn]
    rw [hd, ho, Option.map_eq_none_iff]
    exact hok
  simp [hnone]

theorem spanGives : SpanGives := by
  intro reg digest appends w sampled Ω μ V r right decode ε g h
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  classical
  obtain ⟨-, hne⟩ := hω
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro hall
  refine hne (checkWindow_ofFn_accept reg digest appends w sampled _ fun k => ?_)
  by_contra hk
  exact hall k (by simp [Run.failing, hk])

theorem oneSample_ok (k : Fin (#[((0 : Nat), (0 : Nat))].size)) :
    OpeningOk (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16).digest
      #[] (honestWindow (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (zeros 64) (zeros 64)) #[(0, 0)] k
      ((oneSample (honestOpening (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (zeros 64) (zeros 64))).ports k ()) := by
  obtain ⟨k, hk⟩ := k
  obtain rfl : k = 0 := by simp at hk; omega
  show checkOpening _ _ _ _ (receipterOf _ _ 0) 0 0 _ = none
  exact honest_checks (honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16) (zeros 64) (zeros 64) rfl rfl rfl

theorem spanGives_honest : SpanGivesHonest := by
  refine ⟨_, _, _, honestReg_make (zeros 1) (zeros Codec.SEED_BYTES) 16 (by simp [zeros_size]) (zeros_size _)
    (Or.inl rfl), ?_, checkWindow_ofFn_accept _ _ _ _ _ _ oneSample_ok⟩
  refine le_trans (prPMF_mono _ (F := fun _ => False) fun ω hω => ?_) (by simp [prPMF])
  obtain ⟨-, hk⟩ := hω
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem] at hk
  refine hk fun u hu => ?_
  cases ω
  simp only [Run.failing, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  exact hu (oneSample_ok u)

theorem spanGives_nontrivial : SpanGivesNontrivial := by
  let reg := honestReg (zeros 1) (zeros Codec.SEED_BYTES) 16
  let o : Opening := { honestOpening reg (zeros 64) (zeros 64) with pass := 1 }
  refine ⟨reg, honestWindow reg (zeros 64) (zeros 64), o, fun h => ?_⟩
  have hrej : checkWindow reg reg.digest #[] (honestWindow reg (zeros 64) (zeros 64)) #[(0, 0)]
      (Array.ofFn fun k => (oneSample o).ports k ()) ≠ Verdict.accept := by
    rw [checkWindow_eq]
    simp [rejectAt, checkOpening, oneSample, o, Verdict.accept, Verdict.reject]
  simp [prPMF, oneSample] at h
  exact hrej h

end Properties.Poub.InProofs.SecurityProofs
