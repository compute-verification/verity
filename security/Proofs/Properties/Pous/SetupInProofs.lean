module

public import Properties.Pous
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Proofs.Properties.Pous.InProofs
public import Proofs.Properties.ResourceAccounting.InProofs

/-! An accepted setup run meeting `SetupGives`' three hypotheses leaves at most `⌊B/100⌋` blocks that don't decode; it
says nothing of the deployed run, where nothing proves `Gives` and `CarriedW`. -/

@[expose] public section

namespace PousCheck.Properties.InProofs

open PousCheck.Properties Verity.InProofs Verity.Game
open scoped ENNReal

def slice (q : P2.Params) (W : ByteArray) (u : Nat) : ByteArray :=
  W.extract (u * q.payloadBytes) (u * q.payloadBytes + q.payloadBytes)

def BlockOk (q : P2.Params) (salt : ByteArray) (u : Fin q.blocks) (v : ByteArray × ByteArray) : Prop :=
  BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) v.1 v.2

def CarriedW {Ω : Type} (q : P2.Params) (W : ByteArray) (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) : Prop :=
  ∀ ω, r.accepts ω → (∀ u, (r.ports u ω).2 = slice q W u.val) ∨ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)

open Classical in
noncomputable def incorrect {Ω : Type} (q : P2.Params) (salt W : ByteArray)
    (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) (ω : Ω) : Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockOk q salt u ((r.ports u ω).1, slice q W u.val)

/-- **An accepted setup leaves at most `⌊B/100⌋` blocks that don't decode, except `ε`, for any run meeting its
hypotheses**; it says nothing of the deployed run, where nothing proves `Gives` and `CarriedW`. -/
def SetupGives : Prop :=
  ∀ (q : P2.Params) (salt W : ByteArray) (Ω : Type) (μ : PMF Ω)
    (r : Run Ω q.blocks fun _ => ByteArray × ByteArray) (right : Fin q.blocks → ByteArray × ByteArray → Prop)
    (ε : ℝ≥0∞),
    Gives μ r right (q.blocks / 100 + 1) ε →
    right ≤ BlockOk q salt →
    CarriedW q W r →
    prPMF μ (fun ω => r.accepts ω ∧ q.blocks / 100 + 1 ≤ (incorrect q salt W r ω).card ∧
        ¬ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)) ≤ ε

/-- **`SetupGives`' conclusion fails without its hypotheses**: at the registered point, a run that always accepts, holds
an empty block at each proof block and hashes nothing leaves every block wrong, more than `⌊B/100⌋`, so the event has
probability `1` and the bound fails at `ε = 0`. -/
def SetupGivesNontrivial : Prop :=
  ∃ (salt W : ByteArray) (v : Fin PousCheck.P2.Params.deployed.blocks → ByteArray × ByteArray),
    ¬ prPMF (PMF.pure ()) (fun ω => (MemoryChallenger.Properties.InProofs.constRun v).accepts ω ∧
        PousCheck.P2.Params.deployed.blocks / 100 + 1 ≤
          (incorrect PousCheck.P2.Params.deployed salt W (MemoryChallenger.Properties.InProofs.constRun v) ω).card ∧
        ¬ Hash.CollisionIn Verity.sha512 ((MemoryChallenger.Properties.InProofs.constRun v).hashInputs ω)) ≤ 0

end PousCheck.Properties.InProofs

namespace PousCheck.Properties.InProofs.SecurityProofs

open PousCheck.Properties.InProofs
open Verity.InProofs Properties.ResourceAccounting.InProofs Verity.Game

theorem setupGives : SetupGives := by
  intro q salt W Ω μ r right ε g h hc
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  obtain ⟨ha, hk, hn⟩ := hω
  have hs := (hc ω ha).resolve_right hn
  have hw : incorrect q salt W r ω = r.failing (BlockOk q salt) ω := by
    ext u
    have hp : r.ports u ω = ((r.ports u ω).1, slice q W u.val) := Prod.ext rfl (hs u)
    simp only [incorrect, Run.failing, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [← hp]
  rw [← hw]
  exact hk

theorem setupGives_nontrivial : SetupGivesNontrivial := by
  refine ⟨.empty, .empty, fun _ => (.empty, .empty), fun h => ?_⟩
  have hw : incorrect PousCheck.P2.Params.deployed .empty .empty
      (MemoryChallenger.Properties.InProofs.constRun fun _ => (.empty, .empty)) () = Finset.univ := by
    ext u
    simp [incorrect, BlockOk, BlockDecodes, MemoryChallenger.Properties.InProofs.constRun, PousCheck.P2.Params.blockBytes,
      PousCheck.P2.Params.deployed, PousCheck.W_BITS]
  have hpos : 0 < PousCheck.P2.Params.deployed.blocks := by
    simp [PousCheck.P2.Params.blocks, PousCheck.P2.Params.deployed]
  have hc : PousCheck.P2.Params.deployed.blocks / 100 + 1 ≤ (incorrect PousCheck.P2.Params.deployed .empty .empty
      (MemoryChallenger.Properties.InProofs.constRun fun _ => (.empty, .empty)) ()).card := by
    rw [hw, Finset.card_univ, Fintype.card_fin]
    exact Nat.div_lt_self hpos (by norm_num)
  have hn : ¬ Hash.CollisionIn Verity.sha512 [] := by simp [Hash.CollisionIn]
  simp only [prPMF, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq] at h
  split_ifs at h with hE
  · exact absurd h (by simp)
  · exact hE ⟨trivial, hc, hn⟩

end PousCheck.Properties.InProofs.SecurityProofs
