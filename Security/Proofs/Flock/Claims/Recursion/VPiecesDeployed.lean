import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.Recursion.CapPieces
import Proofs.Flock.Claims.Recursion.LayoutBCap

/-!
# The V\* parts' pieces of the verdict, and their owners

`recKSDecompR_of_targets` takes the parts' pieces `rows` (what `holds` says of each), the base checks' owners `ownB`
and the pieces' owners `ownQ`, then `hR` (every owner is one of the statement's parts), `hrest`, `hlowR` and `hlowP`.
This file fixes the pieces and the owners, and proves `hR`.

* **The pieces** (`VPiece`, `vrows`, `vholds`): a `rec-residuals` group (`ResPiece`, `resHolds`), a `rec-open` query
  (`OpenPiece`, `openHolds`) or a `rec-cap` node (`CapPiece`, `capHolds`), each kind's list and what each says as its
  lowering gives them (`vholdsD` at the deployed three).
* **The owners** (`noB`, `keep`, `vown`): no part owns a base check (`hrest` holds them off any witnesses:
  `BasesHold`); `dense` goes to part 0; each kind's piece to the owner its lowering gives it, kept to the
  statement's parts (`keep`: an owner past `nS x` owns nothing).
* **`hR`** (`ownP_lt`, `hR_vown`): the owners are the statement's parts, for every kind's owners, since `keep` keeps
  only those.
* **Every piece is owned** (`vown_some`): where `nS x > 0` (`nS_pos`, at every statement `readX` reads) and each
  kind's owner gives its pieces a part, every piece has an owner, so `hrest` is asked of no piece.
* **`hlowP` by kind** (`piecesComputes_vown`): part `s`'s share of the pieces is each kind's share at that kind's owner,
  and, for `dense`, that the inner statement is a dense one.
-/

namespace RecDeploy.VPieces

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP VerdictR)
open Claims.Recursion.VStarB

/-! ## The pieces -/

/-- **A V\* part's piece of the verdict**: a `rec-residuals` group, a `rec-open` query or a `rec-cap` node. -/
inductive VPiece where
  | res (q : ResPiece)
  | opn (q : OpenPiece)
  | cap (q : CapPiece)

/-- **The parts' pieces** at the inner statement `x`: each kind's, in turn. -/
def vrows (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece) (capP : Msg → List CapPiece) (x : Msg) :
    List VPiece :=
  (resP x).map .res ++ (openP x).map .opn ++ (capP x).map .cap

/-- **What a piece says**, as its kind's says. -/
def vholds (resH : ResPiece → Msg → List Msg → List Coin → Prop) (openH : OpenPiece → Msg → List Msg → List Coin → Prop)
    (capH : CapPiece → Msg → List Msg → List Coin → Prop) : VPiece → Msg → List Msg → List Coin → Prop
  | .res q, x, ms, cs => resH q x ms cs
  | .opn q, x, ms, cs => openH q x ms cs
  | .cap q, x, ms, cs => capH q x ms cs

/-- What a piece says at the deployed three: `resHolds`, `openHolds` and `capHolds`. -/
abbrev vholdsD (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (rowAt : Msg → List Msg → CapPiece → ByteArray) : VPiece → Msg → List Msg → List Coin → Prop :=
  vholds (resHolds elemAt vvAt) (openHolds sideAt posAt) (capHolds rowAt)

/-! ## The owners -/

/-- **No part owns a base check.** -/
def noB : Msg → VerdictR → Option ℕ := fun _ _ => none

/-- **An owner kept to the statement's parts**: `s`, where `s < nS x`. -/
def keep (x : Msg) (s : ℕ) : Option ℕ := if s < nS x then some s else none

/-- **The pieces' owners**: `dense` to part 0, each kind's piece to its kind's owner, kept to the statement's parts. -/
def vown (ownR : Msg → ResPiece → Option ℕ) (ownO : Msg → OpenPiece → Option ℕ) (ownC : Msg → CapPiece → Option ℕ)
    (x : Msg) : Top.Piece VPiece → Option ℕ
  | .dense => keep x 0
  | .row (.res q) => (ownR x q).bind (keep x)
  | .row (.opn q) => (ownO x q).bind (keep x)
  | .row (.cap q) => (ownC x q).bind (keep x)

theorem keep_lt {x : Msg} {s t : ℕ} (h : keep x s = some t) : t < nS x := by
  unfold keep at h
  split at h
  · cases h
    assumption
  · cases h

theorem keep_eq {x : Msg} {s t : ℕ} (h : keep x s = some t) : s = t := by
  unfold keep at h
  split at h
  · cases h
    rfl
  · cases h

theorem keep_of_lt {x : Msg} {s : ℕ} (h : s < nS x) : keep x s = some s := by
  unfold keep
  rw [if_pos h]

variable (ownR : Msg → ResPiece → Option ℕ) (ownO : Msg → OpenPiece → Option ℕ) (ownC : Msg → CapPiece → Option ℕ)

theorem vown_lt {x : Msg} {p : Top.Piece VPiece} {s : ℕ} (h : vown ownR ownO ownC x p = some s) : s < nS x := by
  rcases p with _ | (q | q | q)
  · exact keep_lt h
  all_goals
    obtain ⟨t, -, ht⟩ := Option.bind_eq_some_iff.1 h
    exact keep_lt ht

/-- **The owners are the statement's parts**, at every check of the split verdict. -/
theorem ownP_lt {x : Msg} {r : VerdictP (Top.Piece VPiece)} {s : ℕ} (h : ownP noB (vown ownR ownO ownC) x r = some s) :
    s < nS x := by
  cases r with
  | base q => simp [ownP, noB] at h
  | piece p => exact vown_lt ownR ownO ownC h

/-- **`hR` of `recKSDecompR_of_targets`** at the owners `noB` and `vown`, for every kind's owners. -/
theorem hR_vown (rd : Msg → Option FinalP) (rows : Msg → List VPiece) (holds : VPiece → Msg → List Msg → List Coin → Prop)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (h : Top.Targets rd rows holds acc) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recR rd b aux nOf hd rest fin wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ r ∈ (splitR rd rows holds acc h).checks (Claims.Recursion.xI (recR rd b aux nOf hd rest fin wE final w) x),
        ∀ s, ownP noB (vown ownR ownO ownC) x r = some s → s < (recR rd b aux nOf hd rest fin wE final w).nS x :=
  fun _ _ _ _ _ _ _ hs => ownP_lt ownR ownO ownC hs

/-! ## Every piece is owned -/

/-- **A statement `readX` reads has a part**: its last, RecCap. -/
theorem nS_pos {x : Msg} {r : XR} (h : readX x = some r) : 0 < nS x := by
  obtain ⟨_, -, h0⟩ := capAt_readX h
  exact h0

/-- **Every piece has an owner** at `x`, where `x` has a part and each kind's owner gives each of its pieces at `y`
one of `x`'s parts. -/
theorem vown_some (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece) (capP : Msg → List CapPiece) {x y : Msg}
    (h0 : 0 < nS x) (hr : ∀ q ∈ resP y, ∃ s < nS x, ownR x q = some s)
    (ho : ∀ q ∈ openP y, ∃ s < nS x, ownO x q = some s) (hc : ∀ q ∈ capP y, ∃ s < nS x, ownC x q = some s) :
    ∀ p ∈ Top.pieces (vrows resP openP capP) y, ∃ s, vown ownR ownO ownC x p = some s := by
  intro p hp
  rcases List.mem_cons.1 hp with rfl | hp
  · exact ⟨0, keep_of_lt h0⟩
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
  simp only [vrows, List.mem_append, List.mem_map] at hp'
  rcases hp' with (⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩) | ⟨q, hq, rfl⟩
  · obtain ⟨s, hs, e⟩ := hr q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩
  · obtain ⟨s, hs, e⟩ := ho q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩
  · obtain ⟨s, hs, e⟩ := hc q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩

/-! ## `hlowP` by kind -/

/-- **Part `s`'s share of the pieces, kind by kind**: each kind's share at its owner, and, for `dense`, that the inner
statement is a dense one. -/
theorem piecesComputes_vown (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (D : Claims.Recursion.Deploy Coin) (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece)
    (capP : Msg → List CapPiece) (resH : ResPiece → Msg → List Msg → List Coin → Prop)
    (openH : OpenPiece → Msg → List Msg → List Coin → Prop) (capH : CapPiece → Msg → List Msg → List Coin → Prop)
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (hdense : Top.DenseInner (Claims.Recursion.xI D x))
    (hres : PiecesComputes Bd D resP resH ownR s x c ms) (hopen : PiecesComputes Bd D openP openH ownO s x c ms)
    (hcap : PiecesComputes Bd D capP capH ownC s x c ms) :
    PiecesComputes Bd D (Top.pieces (vrows resP openP capP)) (Top.pholds (vholds resH openH capH))
      (vown ownR ownO ownC) s x c ms := by
  intro p hp hown os last hf hk
  rcases List.mem_cons.1 hp with rfl | hp
  · exact hdense
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
  simp only [vrows, List.mem_append, List.mem_map] at hp'
  rcases hp' with (⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩) | ⟨q, hq, rfl⟩
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hres q hq ht os last hf hk
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hopen q hq ht os last hf hk
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hcap q hq ht os last hf hk

end RecDeploy.VPieces
