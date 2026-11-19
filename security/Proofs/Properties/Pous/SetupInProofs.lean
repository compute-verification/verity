module

public import Properties.Pous
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Proofs.Properties.Pous.InProofs
public import Proofs.Properties.ResourceAccounting.InProofs
public import Definitions.Core.Partition
public import Definitions.Spot.Sampling.Analysis
public import Proofs.Spot.Sampling.Extraction
public import Proofs.Core.Game.Prob

/-!
# PoUS's setup inside proofs: an accepted sampled audit of the setup's C leaves at most `⌊B/100⌋` blocks that don't decode to the registered `W`

In PoUS's setup checked inside proofs, `C*` and `W` never reach the verifier. The setup is one circuit `C`
(`verity/protocols/resource_accounting/memory/pous/prover/window.py`, `Setup`): the setup unit `PousP2SetupCheck` over
every block of the store, partitioned into the store's `B = q.blocks` proof blocks, one a block (`SetupC`). Unit `u` reads
its block of `C*` (`block`) and its slice of `W` from `C`'s registered gates (`weights`, which reads no other gate:
`weights_reg`); the door proves a live sample of `8,828` of them (`rate:8828/B`).

`SetupGives` is the composed statement, over the shared sampling theorem's audit (`Spot.Sampling.audit`, its analysis an
`ExtractionAnalysis` whose incorrect units are `C`'s partition's): the verifier accepts while more than `⌊B/100⌋` blocks
of `C*` fail to decode to their slice of the registered `W` with probability at most
`L.miss (⌊B/100⌋ + 1) + ε_ks + ε_bind + ε_reg`. The sampling term is `Spot.Sampling.extraction_audit_count`'s, not a
premise. Its premises are named: the analysis's knowledge and binding errors (`KnowledgeSound`, `BindSound`), its
incorrect units `C`'s partition's, a correct unit's block decoding to its slice of the `W` `C`'s registered gates hold
(`right`, from the setup unit's circuit: `SetupComputesBlockOk`), and `RegisteredW`: the verifier accepts while `C`'s
registered gates hold other than the registered `W` with probability at most `ε_reg`, the registration's collision
term (two `W`'s under the root the verifier holds are a SHA-512 collision). The collision is a term of the bound, as
`ε_bind` is: the audit's outcomes carry no hash inputs to name a collision event on.

What is not on the branch: the door's verifier as this audit (an `ExtractionAnalysis` of the one-circuit door's sessions
with its `KnowledgeSound` and `BindSound`), and `RegisteredW` from Core's registered positions (`Layout.reg` read under
the root the record names). Both stay premises.

Its gap to the space bound: at most `⌊B/100⌋` blocks that don't decode is `P2SlackFamilyFreeBlocks`' premise of at most
`⌊B/100⌋` free blocks only in that guarantee's model, the abstract `m1p` scheme (`Definitions/Pous/Guarantees/P2.lean`),
not the deployed P2 this statement reads; nothing here states the step from one to the other.
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
a block of `C*`, where its values are committed (`Lay`), and what a trace of `C` holds: unit `u`'s block of `C*`
(`block`) and `W` (`weights`), read from `C`'s registered gates only (`weights_reg`). -/
structure SetupC (q : P2.Params) where
  C : Verity.Circuit
  P : Verity.Partition C q.blocks
  Lay : Verity.Layout C
  block : C.Trace → Fin q.blocks → ByteArray
  weights : C.Trace → ByteArray
  weights_reg : ∀ T T' : C.Trace, (∀ g, (Lay.reg g).isSome → T g = T' g) → weights T = weights T'

open Classical in
/-- The blocks of `C*` a trace holds that do not decode to their slice of the registered `W`. -/
noncomputable def SetupC.wrong {q : P2.Params} (S : SetupC q) (salt W : ByteArray) (X : S.C.Trace) :
    Finset (Fin q.blocks) :=
  Finset.univ.filter fun u => ¬ BlockOk q salt u (S.block X u, slice q W u.val)

/-- **Named assumption: `C`'s registered gates hold the registered `W`, except `εreg`.** Every audit strategy is
accepted while the committed trace's registered gates hold other than `W` with probability at most `εreg`: the
registration's binding error, a SHA-512 collision under the root the verifier holds for `W`. -/
def RegisteredW {q : P2.Params} (S : SetupC q) {L : Verity.SamplingDesign (Fin q.blocks)} {Reg : Type}
    {session : Finset (Fin q.blocks) → Reg → Verity.Game Bool} (A : Spot.Sampling.ExtractionAnalysis L Reg session)
    (tr : A.Tr → S.C.Trace) (W : ByteArray) (εreg : ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (Spot.Sampling.audit L Reg session),
    prob (fun o => o.1 = true ∧ S.weights (tr (A.committedOf σ)) ≠ W) (Spot.Sampling.audit L Reg session) σ ≤ εreg

/-- **An accepted sampled audit of the setup's C leaves at most `⌊B/100⌋` blocks that don't decode to the registered
`W`, except `miss + ε_ks + ε_bind + ε_reg`**: for parameters `q`, the salt, the registered `W`, the setup's C `S`, a
sampling design `L` over its proof blocks and the shared audit's analysis `A`, whose transcripts read as traces of `C`
under `tr` and whose incorrect units are `S.P`'s: if `A`'s knowledge and binding errors are `εks` and `εbind`, every
correct unit's block decodes to its slice of the `W` `C`'s registered gates hold (`right`), and those gates hold the
registered `W` except `εreg` (`RegisteredW`), then every audit strategy is accepted while more than `⌊B/100⌋` blocks
of `C*` fail to decode to their slice of `W` with probability at most `L.miss (⌊B/100⌋ + 1) + εks + εbind + εreg`. -/
def SetupGives : Prop :=
  ∀ (q : P2.Params) (salt W : ByteArray) (S : SetupC q) (L : Verity.SamplingDesign (Fin q.blocks)) (Reg : Type)
    (session : Finset (Fin q.blocks) → Reg → Verity.Game Bool) (A : Spot.Sampling.ExtractionAnalysis L Reg session)
    (tr : A.Tr → S.C.Trace) (εks εbind εreg : ℝ≥0∞),
    A.KnowledgeSound (fun _ _ => εks) → A.BindSound (fun _ _ => εbind) →
    (∀ X, A.incorrect X = S.P.incorrect (tr X)) →
    (∀ T u, S.P.Correct T u → BlockOk q salt u (S.block T u, slice q (S.weights T) u.val)) →
    RegisteredW S A tr W εreg →
    ∀ σ : Strategy (Spot.Sampling.audit L Reg session),
      prob (fun o => o.1 = true ∧ q.blocks / 100 + 1 ≤ (S.wrong salt W (tr (A.committedOf σ))).card)
          (Spot.Sampling.audit L Reg session) σ ≤ L.miss (q.blocks / 100 + 1) + εks + εbind + εreg

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
  Lay := { reg := Fin.elim0, reg_input := fun g => g.elim0, reg_inj := fun g => g.elim0, phase := Fin.elim0,
           phase_reads := fun g => g.elim0 }
  block := fun _ _ => .empty
  weights := fun _ => .empty
  weights_reg := fun _ _ _ => rfl

/-- **`SetupGives`' event is reachable**: at the registered point, a trace whose blocks and `W` are empty leaves every
block wrong, more than `⌊B/100⌋`. -/
def SetupGivesNontrivial : Prop :=
  ∃ (salt W : ByteArray) (X : emptySetup.C.Trace),
    PousVerifier.P2.Params.deployed.blocks / 100 + 1 ≤ (emptySetup.wrong salt W X).card

end PousVerifier.Properties.InProofs

/-!
## Proof
On a trace whose registered gates hold `W`, every block that doesn't decode is an incorrect unit (`right`, contraposed:
`wrong_subset_incorrect`, the lemma). So the event is inside "at least `⌊B/100⌋ + 1` incorrect units, accepted" or
"registered gates other than `W`, accepted" (`prob_mono`), whose union bound (`prob_or_le`) is the shared sampling
theorem's count (`Spot.Sampling.extraction_audit_count`) plus `RegisteredW`'s `εreg`.
-/

namespace PousVerifier.Properties.InProofs.SecurityProofs

open PousVerifier.Properties.InProofs
open Verity.InProofs Properties.ResourceAccounting.InProofs Verity.Game

/-- **The lemma**: where `C`'s registered gates hold `W`, a block that doesn't decode to its slice of `W` is an incorrect
unit. -/
theorem wrong_subset_incorrect {q : P2.Params} (S : SetupC q) (salt W : ByteArray)
    (right : ∀ T u, S.P.Correct T u → BlockOk q salt u (S.block T u, slice q (S.weights T) u.val))
    (X : S.C.Trace) (hW : S.weights X = W) : S.wrong salt W X ⊆ S.P.incorrect X := by
  intro u hu
  simp only [SetupC.wrong, Verity.Partition.incorrect, Finset.mem_filter, Finset.mem_univ, true_and] at hu ⊢
  exact fun hc => hu (by rw [← hW]; exact right _ _ hc)

theorem setupGives : SetupGives := by
  intro q salt W S L Reg session A tr εks εbind εreg hks hlink hA right hreg σ
  classical
  calc prob (fun o => o.1 = true ∧ q.blocks / 100 + 1 ≤ (S.wrong salt W (tr (A.committedOf σ))).card)
        (Spot.Sampling.audit L Reg session) σ
      ≤ prob (fun o => (o.1 = true ∧ q.blocks / 100 + 1 ≤ (A.incorrect (A.committedOf σ)).card) ∨
          (o.1 = true ∧ S.weights (tr (A.committedOf σ)) ≠ W)) (Spot.Sampling.audit L Reg session) σ := by
        refine Verity.Game.prob_mono (fun o ho => ?_) _ _
        by_cases hW : S.weights (tr (A.committedOf σ)) = W
        · refine Or.inl ⟨ho.1, ho.2.trans ?_⟩
          rw [hA]
          exact Finset.card_le_card (wrong_subset_incorrect S salt W right _ hW)
        · exact Or.inr ⟨ho.1, hW⟩
    _ ≤ prob (fun o => o.1 = true ∧ q.blocks / 100 + 1 ≤ (A.incorrect (A.committedOf σ)).card)
          (Spot.Sampling.audit L Reg session) σ +
        prob (fun o => o.1 = true ∧ S.weights (tr (A.committedOf σ)) ≠ W) (Spot.Sampling.audit L Reg session) σ :=
        Verity.Game.prob_or_le _ _ _ _
    _ ≤ L.miss (q.blocks / 100 + 1) + εks + εbind + εreg :=
        add_le_add (Spot.Sampling.extraction_audit_count A hks hlink (q.blocks / 100 + 1) σ) (hreg σ)

theorem setupGives_nontrivial : SetupGivesNontrivial := by
  refine ⟨.empty, .empty, fun g => g.elim0, ?_⟩
  have hw : emptySetup.wrong .empty .empty (fun g => g.elim0) = Finset.univ := by
    ext u
    simp [SetupC.wrong, emptySetup, BlockOk, BlockDecodes, PousVerifier.P2.Params.blockBytes,
      PousVerifier.P2.Params.deployed, PousVerifier.W_BITS]
  have hpos : 0 < PousVerifier.P2.Params.deployed.blocks := by
    simp [PousVerifier.P2.Params.blocks, PousVerifier.P2.Params.deployed]
  rw [hw, Finset.card_univ, Fintype.card_fin]
  exact Nat.div_lt_self hpos (by norm_num)

end PousVerifier.Properties.InProofs.SecurityProofs
