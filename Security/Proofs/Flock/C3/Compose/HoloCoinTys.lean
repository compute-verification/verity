import Proofs.Flock.Recursive.HoloMap
import Proofs.Flock.VBridge.Family.Transcript

/-!
# `CoinTys` at candidate 3

The family's `CoinTys` and `vCT` describe the plain inner protocol's coins: the link points, then each of both reps'
rounds a coin of the size the verifier takes there. Candidate 3's inner protocol has the coins of its coin kinds
(`holoCTs`), and its history lays the session's streams out as the verifier's map reads them (`holoMap`).
`HoloCoinTys` (named) says those coins after the link points are, round by round, coins of the sizes the history's
rounds take in the streams (`holoSizes`), as `coinArr` types them (`coinTy`). At it every history round's coin is the
coin step its words read as (`coinArr_roundWords`), which is what the map's `coin` reads off the history's prefix.
-/

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3
open FlockVBridge.Family (coinTy exists_coinArr)

/-- **The coin sizes of the history's rounds after the link points** in the streams `L`, in order: a sparse phase's
stream round of `n` coins takes `n` rounds of one coin, every other stream's round of `n > 0` coins one round of `n`
(`HoloStream.group`). -/
def holoSizes (L : List (HoloStream × Array SchedRound)) : List ℕ :=
  L.flatMap fun e => e.2.toList.flatMap fun q => match e.1 with
    | .sparse _ => List.replicate q.1 1
    | _ => if q.1 = 0 then [] else [q.1]

/-- **`CoinTys` at candidate 3** (named): `holoKinds`' coin types, the coins after the link points, are coins of the
sizes the history's rounds take in the streams `L` (`holoSizes`), as `coinArr` types them. -/
def HoloCoinTys (st : Flock.Stmt) (hp : HoloParams) (L : List (HoloStream × Array SchedRound)) : Prop :=
  kindCoinTys (holoKinds (holoAt st hp)) = (holoSizes L).map coinTy

theorem nthTy_map (f : ℕ → Type) : ∀ (l : List ℕ) (j : ℕ) (_ : j < l.length), nthTy (l.map f) j = f l[j]
  | [], _, hj => absurd hj (Nat.not_lt_zero _)
  | _ :: _, 0, _ => rfl
  | _ :: l, j + 1, hj => nthTy_map f l j (Nat.lt_of_succ_lt_succ hj)

/-- **`holoCTs` at `HoloCoinTys`**: coin `1 + j` is a coin of the size of the history's `j`-th round after the link
points. -/
theorem holoCTs_succ {st : Flock.Stmt} {hp : HoloParams} {L : List (HoloStream × Array SchedRound)}
    (hL : HoloCoinTys st hp L) (j : ℕ) (h : 1 + j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length)
    (hj : j < (holoSizes L).length) : holoCTs st hp ⟨1 + j, h⟩ = coinTy (holoSizes L)[j] := by
  show nthTy (_ :: kindCoinTys (holoKinds (holoAt st hp))) (1 + j) = _
  rw [Nat.add_comm 1 j]
  show nthTy (kindCoinTys (holoKinds (holoAt st hp))) j = _
  rw [hL]
  exact nthTy_map coinTy _ j hj

/-- **A coin's words read back as the coin**, wherever some array reads as it. -/
theorem coinArr_coinWords {C : Type} {c : C} (h : ∃ a : Array F128, coinArr fOf a = ⟨C, c⟩) :
    coinArr fOf (coinWords C c) = ⟨C, c⟩ := by
  unfold coinWords
  rw [dif_pos h]
  exact h.choose_spec

/-- **Every history round's coin is the coin step its words read as**, at `HoloCoinTys`: at round `1 + j` of the
history's coins `cs`, `coinArr` of the round's words (`roundWords`) is coordinate `1 + j` of its coin. -/
theorem coinArr_roundWords {st : Flock.Stmt} {hp : HoloParams} {L : List (HoloStream × Array SchedRound)}
    (hL : HoloCoinTys st hp L)
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) (j : ℕ)
    (h : 1 + j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length) (hc : 1 + j < cs.length) :
    coinArr fOf (roundWords cs (1 + j)) = ⟨holoCTs st hp ⟨1 + j, h⟩, cs[1 + j] ⟨1 + j, h⟩⟩ := by
  have hj : j < (holoSizes L).length := by
    have h' := h
    rw [hL, List.length_map] at h'
    omega
  have e := holoCTs_succ hL j h hj
  obtain ⟨a, _, ha⟩ := exists_coinArr _ (cast e (cs[1 + j] ⟨1 + j, h⟩))
  unfold roundWords
  rw [List.getElem?_eq_getElem hc]
  dsimp only
  rw [dif_pos h]
  exact coinArr_coinWords ⟨a, ha.symm.trans (Sigma.ext e (cast_heq e _).symm).symm⟩

end FlockC3.Compose
