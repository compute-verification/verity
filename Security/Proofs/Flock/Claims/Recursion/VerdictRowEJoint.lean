import Proofs.Flock.Claims.Recursion.VerdictRowECoin
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed

/-!
# Row E's chain at the joint witnesses

The deployed last message carries no chain block: the `rec-acc` chain is witness-side. So row E's running sums go into
the `rec-open` pieces' holds (`accPiece`), at one chain the pieces carry together (`ChainHolds`, chosen by `chainC`),
and the targets read the accumulators `accRC` at that chain.

* **Row E's leaf, the sound one** (`RowEJoint`): at every joint witnesses (`Joint`: the recorded calls true and free of
  collisions, every part accepting; adversarial witnesses included), some chain carries every row D piece's running sum
  at the messages the last message opens and the coins. Nothing fixes the witnesses to an honest prover's.
* **The pieces' share** (`piecesComputes_accPiece`): `RowEJoint` gives every part's share of `accPiece`
  (`PiecesComputes`), whatever owner `ownQ` names.
* **Row E's target** (`targetE_accRC`): wherever `P` gives each piece its `accPiece`, `TargetE` at `accRC`. `chainC`
  reads the coins, so `targetE_rowD` is taken at the coins' own chain.
-/

namespace RecDeploy.Top

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.Rows (accHolds sideAt coefRead rowDPieces)

variable (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)

/-- **One chain carries row D's running sums** at `x`, `ms`, `cs`: every piece's `accHolds` at `ch`. -/
def ChainHolds (x : Msg) (ms : List Msg) (cs : List Coin) (ch : ℕ → ℕ → Array F128) : Prop :=
  ∀ q ∈ rowDPieces key x, accHolds (sideAt rd) coefRead (fun _ _ => ch) q x ms cs

open Classical in
/-- **The chain** at `x`, `ms`, `cs`: one that carries row D's running sums, where one does. -/
noncomputable def chainC (x : Msg) (ms : List Msg) (cs : List Coin) : ℕ → ℕ → Array F128 :=
  if h : ∃ ch, ChainHolds rd key x ms cs ch then Classical.choose h else fun _ _ => #[]

/-- **A `rec-open` piece's running sum** at `chainC`. -/
def accPiece (q : OpenPiece) (x : Msg) (ms : List Msg) (cs : List Coin) : Prop :=
  accHolds (sideAt rd) coefRead (fun _ _ => chainC rd key x ms cs) q x ms cs

/-- **The reps' accumulators at `chainC`**: `accR` at the coins' own chain. -/
noncomputable abbrev accRC (x : Msg) (ms : List Msg) (cs : List Coin) : ℕ → Array F128 :=
  accR rd (fun x' ms' => chainC rd key x' ms' cs) x ms cs

theorem accPiece_of_chain {x : Msg} {ms : List Msg} {cs : List Coin} {ch : ℕ → ℕ → Array F128}
    (h : ChainHolds rd key x ms cs ch) : ∀ q ∈ rowDPieces key x, accPiece rd key q x ms cs := by
  intro q hq
  have he : ∃ ch, ChainHolds rd key x ms cs ch := ⟨ch, h⟩
  have hs := Classical.choose_spec he q hq
  have hc : chainC rd key x ms cs = Classical.choose he := dif_pos he
  unfold accPiece
  rw [hc]
  exact hs

/-- **Row E's target at `accRC`**, wherever `P` gives each row D piece its `accPiece`. -/
theorem targetE_accRC (P : Msg → List Msg → List Coin → Prop)
    (hP : ∀ x ms cs, P x ms cs → ∀ q ∈ rowDPieces key x, accPiece rd key q x ms cs) :
    TargetE rd P (accRC rd key) := by
  intro x ms cs hA D cls i hi
  exact targetE_rowD rd (fun x' ms' cs' => P x' ms' cs' ∧ cs' = cs) (fun x' ms' => chainC rd key x' ms' cs) key
    (fun x' ms' cs' h q hq => by
      obtain ⟨h, rfl⟩ := h
      exact hP x' ms' cs' h q hq)
    x ms cs ⟨hA.1, hA.2.1, hA.2.2.1, hA.2.2.2, rfl⟩ D cls i hi

section Joint

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (L : LayoutB)
  (shape : Claims.Recursion.Form → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)

/-- **Row E's leaf at the joint witnesses**: at a compiled statement of shape `shape`, at every witnesses `ws` whose
recorded calls are true and free of collisions and at whose messages every part's `chk` holds, some chain carries row
D's running sums at the messages the last message opens and the coins. -/
def RowEJoint : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), OpenLower.Joint H D L chk x c ws →
      ∀ os last, Bd.final ((L.msgs x ws).getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
        os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
        ∃ ch, ChainHolds rd key (Claims.Recursion.xI D x) (Bd.opened os last)
          ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) ch

/-- **Every part's share of `accPiece`** at the joint witnesses, from `RowEJoint`. -/
theorem piecesComputes_accPiece (h : RowEJoint rd key H D Bd L shape chk) (ownQ : Msg → OpenPiece → Option ℕ) :
    ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), OpenLower.Joint H D L chk x c ws →
        ∀ s, PiecesComputes Bd D (rowDPieces key) (accPiece rd key) ownQ s x c (L.msgs x ws) := by
  intro x st hx hf c ws hJ s q hq _ os last hfin hlen
  obtain ⟨ch, hch⟩ := h x st hx hf c ws hJ os last hfin hlen
  exact accPiece_of_chain rd key hch q hq

end Joint

end RecDeploy.Top
