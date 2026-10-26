import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.LayoutBTops
import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.BindsDeployed

/-!
# The statement's share of `hrest` at `nextR`: the last message's reading, the rounds' lengths, and the reading

`decompB_nextR` (and `decompB_lastSplitP_shares`, `recKSDecompR_of_targets` above it) takes `hrest`: each check of
`IVStar` the inventory gives no part holds at the readers' messages of every witnesses whose calls are true, at every
coin. The default inventory (`ownHolders`) gives no part the last message's reading (`shape`) and the rounds' lengths
(`len j`), and `ownJ` gives the record's reading (`read`) to the first part holding the last round whole. At
`layoutBTops` no part holds the last round whole: it is assembled from pieces (`finOpen`), so that owner is none at
every dense compiled statement (`firstHolder_last`), and the reading too is the statement's.

This file proves those three kinds of check at the deployed bound record `bdAt rd wE final w` (`nextR rd`'s steps,
rows joined by `joinR`) and `layoutBTops (finOpen fin)`, at any split of the verdict (`LastSplitR`), from three facts
about the last message and the rounds:

* `FinalReads final fin`: at every witnesses, `final` reads the last message as `k - 1` rounds' rows and salts with the
  schedule's counts (`nAtSched`);
* `RoundLens fin`: each round before the last is 128 bytes a commit string the schedule counts;
* `ReadsAt rd final fin`: wherever `final` reads it, the last step reads the opened messages (`ReadOkR`).

`hrest_shape`, `hrest_len` and `hrest_read` give one check each; `hrest_part` gives the three at `hrest`'s binders.
-/

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR)
open Claims.Recursion.VStarB

noncomputable section

/-! ## The deployment's terms -/

/-- **A dense compiled statement**: one the deployed reader parses at the dense form. -/
def DenseX (x : Msg) : Prop := ∃ st, parse x = some st ∧ Claims.Recursion.Form.Dense st.form

/-- The inner rounds at a compiled statement: the deployed plain verifier's at its inner statement. -/
abbrev kI (x : Msg) : ℕ := (FlockDeploy.V false).k (xIOf x)

/-- The deployed bound record at `nextR rd`, its rows joined by `joinR`. -/
abbrev bdAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :=
  bdDeployed _root_.Flock.Readable.relation (nextR rd) (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
    wE FlockDeploy.joinR rowInDeployed final w

/-- V\*'s bound layout at the last round's pieces `fin`. -/
abbrev layAt (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : LayoutB :=
  layoutBTops (finOpen fin)

/-- The split of the element-level verifier at `nextR rd` and a split `LS` of its verdict. -/
abbrev splitAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) {R : Type} (LS : LastSplitR FlockDeploy.sha512 rd R) :=
  deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
    (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE FlockDeploy.joinR rowInDeployed final w
    (nextR_coin rd FlockDeploy.sha512) LS.toLast

/-- **The last message off the witnesses `ws`**: the readers' message at the last inner round. -/
def lastAt (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (ws : List Claims.Flock.Wit) : Msg :=
  ((layAt fin).msgs x ws).getD (kI x - 1) []

/-! ## The three facts -/

/-- **`final` reads the last message at every witnesses**: at a dense compiled statement and witnesses one a part, as
`k - 1` rounds' rows and salts, round `j`'s as many as the schedule counts (`nAtSched`). -/
def FinalReads (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x →
    ∃ os last, final (lastAt fin x ws) = some (os, last) ∧ os.length + 1 = kI x ∧
      ∀ j < os.length, (os.getD j []).length = nAtSched (xIOf x) j

/-- **Each round before the last is 128 bytes a commit string**: at a dense compiled statement and witnesses one a
part, round `j < k - 1`'s message off the witnesses has `128 · nAtSched j` bytes. -/
def RoundLens (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x → ∀ j < kI x - 1,
    (((layAt fin).msgs x ws).getD j []).length = 128 * nAtSched (xIOf x) j

/-- **The last step reads at every witnesses**: at a dense compiled statement, witnesses one a part whose calls are
true, and coins `c`, wherever `final` reads the last message as `k - 1` rounds, `nextR rd` reads the opened messages at
the coins (`ReadOkR`). -/
def ReadsAt (rd : Msg → Option FinalP) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, final (lastAt fin x ws) = some (os, last) → os.length + 1 = kI x →
      ReadOkR rd (xIOf x) (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last])
        ((List.range (kI x)).map c)

/-! ## The three checks -/

theorem kI_pos (x : Msg) : 0 < kI x := by
  unfold kI FlockDeploy.V
  dsimp only
  split <;> omega

section Checks

variable (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
  (w : Msg → ℝ) (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {R : Type}
  (LS : LastSplitR FlockDeploy.sha512 rd R)

/-- **The last message's reading holds at every witnesses**, where `final` reads it with the schedule's counts. -/
theorem hrest_shape (hF : FinalReads final fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) :
    (IVCheck.shape : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512 (bdAt rd wE final w)
      (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  obtain ⟨os, last, hf, hos, hcnt⟩ := hF x hx ws hl
  exact ⟨kI_pos x, os, last, hf, hos, hcnt⟩

/-- **Each round before the last has its length at every witnesses**, where `final` reads the last message with the
schedule's counts and each round has 128 bytes a counted commit string. -/
theorem hrest_len (hF : FinalReads final fin) (hL : RoundLens fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {j : ℕ} (hj : j < kI x - 1) :
    (IVCheck.len j : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512 (bdAt rd wE final w)
      (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  obtain ⟨os, last, hf, hos, hcnt⟩ := hF x hx ws hl
  intro os' last' hf'
  have hf'' : final (lastAt fin x ws) = some (os', last') := hf'
  rw [hf] at hf''
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf'')
  rw [hL x hx ws hl j hj]
  exact congrArg (128 * ·) (hcnt j (by omega)).symm

/-- **The record's reading holds at every witnesses**, where the last step reads the opened messages. -/
theorem hrest_read (hR : ReadsAt rd final fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x)
    (hT : ∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) :
    (IVCheck.elem (DeployCheck.last ReadCheck.read) : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512
      (bdAt rd wE final w) (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  intro os last hf hos
  exact hR x hx c ws hl hT os last hf hos

/-- **The statement's share of `hrest`, at its three kinds of check**: at `hrest`'s binders (`decompB_nextR`'s at
`layoutBTops (finOpen fin)` and `bdAt rd wE final w`, any brings `bB` and split `LS`), each check that is the last
message's reading, a round's length or the record's reading holds at the readers' messages of every witnesses whose
calls are true, at every coin. -/
theorem hrest_part (bB : Brings) (hF : FinalReads final fin) (hL : RoundLens fin) (hR : ReadsAt rd final fin) :
    ∀ (x : Msg) (st : Claims.Recursion.Stmt),
      (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ q ∈ ivChecks (bdAt rd wE final w) (splitAt rd wE final w LS)
          (Claims.Recursion.xI (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar) x),
        (q = .shape ∨ (∃ j, q = .len j) ∨ q = .elem (.last .read)) →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
          ws.length = (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar).nS x →
          (∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
          q.Holds FlockDeploy.sha512 (bdAt rd wE final w) (splitAt rd wE final w LS)
            (Claims.Recursion.xI (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar) x)
            c ((layAt fin).msgs x ws) := by
  intro x st hx hd q hq hk c ws hl hT
  have hX : DenseX x := ⟨st, hx, hd⟩
  rcases hk with rfl | ⟨j, rfl⟩ | rfl
  · exact hrest_shape rd wE final w fin LS hF hX c hl
  · have hj : j < kI x - 1 := by
      rcases mem_ivChecks _ _ _ hq with h | ⟨j', hj', h⟩ | ⟨_, _, _, _, h | h⟩ | ⟨_, _, h⟩
      · cases h
      · cases h
        exact hj'
      · cases h
      · cases h
      · cases h
    exact hrest_len rd wE final w fin LS hF hL hX c hl hj
  · exact hrest_read rd wE final w fin LS hR hX c hl hT

end Checks

/-! ## The reading is the statement's at `layoutBTops` -/

/-- **No part holds the last round whole at `layoutBTops`**: at a dense compiled statement, the last inner round is
past every stream round, so no part holds it (`msgAt`) and `ownJ` gives the record's reading to none. -/
theorem firstHolder_last (D : Claims.Recursion.Deploy Coin)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg} (hx : DenseX x) :
    firstHolder D (layAt fin) x (kI x - 1) = none := by
  obtain ⟨st, hst, hd⟩ := hx
  obtain ⟨r, rs, hr, hrs, hK⟩ := schedMatches_dense x st hst hd
  have hk : kI x - 1 = 2 * rs.size + 1 := by
    show (FlockDeploy.V false).k (xIOf x) - 1 = _
    omega
  unfold firstHolder
  rw [hk, List.find?_eq_none]
  intro s _
  simp [layAt, layoutBTops, streamRound, hr, hrs]

end

end RecDeploy.RestRead
