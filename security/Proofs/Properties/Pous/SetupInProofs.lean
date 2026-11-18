module

public import Properties.Pous
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Proofs.Properties.Pous.InProofs
public import Proofs.Properties.ResourceAccounting.InProofs
public import Definitions.Core.Partition

/-!
# PoUS's setup inside proofs: the door's acceptance of sampled blocks of the setup's C leaves at most `⌊B/100⌋` blocks that don't decode

In PoUS's setup checked inside proofs, `C*` and `W` never reach the verifier. The setup is one circuit `C`
(`verity/protocols/resource_accounting/memory/pous/prover/window.py`, `Setup`): the setup unit `PousP2SetupCheck_v1`
over every block of the store, partitioned into the store's `B = q.blocks` proof blocks, one a block (`SetupC`). Unit
`u` reads its block of `C*` and its slice of `W` from `C`'s gates (`block`, and `weights`, `C`'s registered gates); the
door proves a live sample of `8,828` of them (`rate:8828/B`).

`SetupGives`: if the door's acceptance leaves `⌊B/100⌋ + 1` incorrect units of the partition (`Partition.incorrect`)
with probability at most `ε`, and a correct unit's block decodes to its slice of `W` (`right`), then the verifier accepts
while more than `⌊B/100⌋` blocks of `C*` fail to decode to their slice of the `W` `C`'s registered gates hold, with
probability at most `ε`. That the registered gates hold the registered `W`, or the run holds a collision, is the door's
(Core's registered positions, `Layout.reg`), not a hypothesis here. The guarantee that instantiates it takes the door's
acceptance from the one-circuit soundness theorem at `rate:8828/B`, and `right` from the setup unit's circuit; its
sampling term, the chance a uniform `8,828`-subset misses `⌊B/100⌋ + 1` incorrect units, is at most `0.99^8828`.

Its gap to the space bound: at most `⌊B/100⌋` blocks that don't decode is `P2SlackFamilyFreeBlocks`' premise of at most
`⌊B/100⌋` free blocks only in that guarantee's model, the abstract `m1p` scheme (`Definitions/Pous/Guarantees/P2.lean`),
not the deployed P2 this lemma reads; nothing here states the step from one to the other.
-/

@[expose] public section

namespace PousVerifier.Properties.InProofs

open PousVerifier.Properties Verity.InProofs Verity.Game
open scoped ENNReal

/-- Block `u`'s slice of `W`: its `payloadBytes` bytes at offset `u · payloadBytes`, as `Decodes` reads it. -/
def slice (q : P2.Params) (W : ByteArray) (u : Nat) : ByteArray :=
  W.extract (u * q.payloadBytes) (u * q.payloadBytes + q.payloadBytes)

/-- **Today's per-block check on a proof block's values**: its block (`v.1`) decodes canonically under `salt` to the
payload `v.2`. -/
def BlockOk (q : P2.Params) (salt : ByteArray) (u : Fin q.blocks) (v : ByteArray × ByteArray) : Prop :=
  BlockDecodes q salt (u.val / q.blocksPerSegment) (u.val % q.blocksPerSegment) v.1 v.2

/-- **The setup's C as the door reads it**: a circuit `C`, its partition into the store's `q.blocks` proof blocks, one
a block of `C*`, and what a trace of `C` holds: unit `u`'s block of `C*` (`block`) and `W` (`weights`, read from `C`'s
registered gates). -/
structure SetupC (q : P2.Params) where
  C : Verity.Circuit
  P : Verity.Partition C q.blocks
  block : C.Trace → Fin q.blocks → ByteArray
  weights : C.Trace → ByteArray

open Classical in
/-- The blocks of `C*` a trace holds that do not decode to their slice of the `W` it holds. -/
noncomputable def SetupC.wrong {q : P2.Params} (S : SetupC q) (salt : ByteArray) (X : S.C.Trace) :
    Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockOk q salt u (S.block X u, slice q (S.weights X) u.val)

/-- **The door's acceptance of sampled blocks of the setup's C leaves at most `⌊B/100⌋` blocks that don't decode,
except `ε`**: for parameters `q`, the server's salt, the setup's C `S` and a run of it (`accepts`, the door's verdict, and
`X`, the trace the run commits): if the door accepts while `⌊B/100⌋ + 1` units of `S.P` are incorrect with probability at
most `ε`, and every correct unit's block decodes to its slice of `W` (`right`), then the door accepts while more than
`⌊B/100⌋` blocks of `C*` fail to decode to their slice of `W`, with probability at most `ε`. -/
def SetupGives : Prop :=
  ∀ (q : P2.Params) (salt : ByteArray) (S : SetupC q) (Ω : Type) (μ : PMF Ω) (accepts : Ω → Prop)
    (X : Ω → S.C.Trace) (ε : ℝ≥0∞),
    prPMF μ (fun ω => accepts ω ∧ q.blocks / 100 + 1 ≤ (S.P.incorrect (X ω)).card) ≤ ε →
    (∀ T u, S.P.Correct T u → BlockOk q salt u (S.block T u, slice q (S.weights T) u.val)) →
    prPMF μ (fun ω => accepts ω ∧ q.blocks / 100 + 1 ≤ (S.wrong salt (X ω)).card) ≤ ε

/-- The empty circuit: no gate, so no partition of it holds a gate. -/
def emptyC : Verity.Circuit where
  N := 0
  w := Fin.elim0
  isInput := Fin.elim0
  reads := Fin.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅

/-- The setup's C at the registered point over the empty circuit, every block and `W` empty. -/
def emptySetup : SetupC PousVerifier.P2.Params.deployed where
  C := emptyC
  P := ⟨Fin.elim0, fun g => g.elim0⟩
  block := fun _ _ => .empty
  weights := fun _ => .empty

/-- **`SetupGives`' conclusion fails without its hypotheses**: at the registered point, a run that always accepts with
an empty block and an empty `W` leaves every block wrong, more than `⌊B/100⌋`, so the event has probability `1` and the
bound fails at `ε = 0`. -/
def SetupGivesNontrivial : Prop :=
  ∃ (salt : ByteArray) (X : emptySetup.C.Trace),
    ¬ prPMF (PMF.pure ()) (fun _ => True ∧
        PousVerifier.P2.Params.deployed.blocks / 100 + 1 ≤ (emptySetup.wrong salt X).card) ≤ 0

end PousVerifier.Properties.InProofs

/-!
## Proof
Every block that doesn't decode is an incorrect unit (`right`, contraposed), so the event's count is at most
`incorrect`'s and the door's bound gives it (`prPMF_mono`).
-/

namespace PousVerifier.Properties.InProofs.SecurityProofs

open PousVerifier.Properties.InProofs
open Verity.InProofs Properties.ResourceAccounting.InProofs Verity.Game

theorem setupGives : SetupGives := by
  intro q salt S Ω μ accepts X ε g right
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, le_trans hω.2 (Finset.card_le_card ?_)⟩) g
  intro u hu
  simp only [SetupC.wrong, Verity.Partition.incorrect, Finset.mem_filter, Finset.mem_univ, true_and] at hu ⊢
  exact fun hc => hu (right _ _ hc)

theorem setupGives_nontrivial : SetupGivesNontrivial := by
  refine ⟨.empty, fun g => g.elim0, fun h => ?_⟩
  have hw : emptySetup.wrong .empty (fun g => g.elim0) = Finset.univ := by
    ext u
    simp [SetupC.wrong, emptySetup, BlockOk, BlockDecodes, PousVerifier.P2.Params.blockBytes,
      PousVerifier.P2.Params.deployed, PousVerifier.W_BITS]
  have hpos : 0 < PousVerifier.P2.Params.deployed.blocks := by
    simp [PousVerifier.P2.Params.blocks, PousVerifier.P2.Params.deployed]
  have hc : PousVerifier.P2.Params.deployed.blocks / 100 + 1 ≤
      (emptySetup.wrong .empty (fun g => g.elim0)).card := by
    rw [hw, Finset.card_univ, Fintype.card_fin]
    exact Nat.div_lt_self hpos (by norm_num)
  simp only [prPMF, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq] at h
  split_ifs at h with hE
  · exact absurd h (by simp)
  · exact hE ⟨trivial, hc⟩

end PousVerifier.Properties.InProofs.SecurityProofs
