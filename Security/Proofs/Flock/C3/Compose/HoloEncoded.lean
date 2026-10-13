import Proofs.Flock.C3.Compose.HoloCoinTys
import Proofs.Flock.C3.Compose.LaterHeld

/-!
# The map's coins and points at candidate 3

`pointsEncoded_holoMap`: the verifier's map's link points read back as the history's first coin (`PointsEncoded` at
`holoPts`).
`coinsEncoded_holoMap`: at `HoloCoinTys`, where a stream round takes one history round (`holoSelOne`: the reps' and R's
rounds of coins, a sparse phase's of one), the coin the map reads off the history's prefix is that round's coin
(`CoinsEncoded`). A sparse phase's round of `n > 1` coins is not one: the map reads its `n` history rounds' words as one
array, while each of those rounds draws a coin of one word.
`roundAt_reps`: the history order of the reps' rounds at the map read, rep 0's from round 1 and rep 1's past all of rep
0's coin rounds (`repStart`), the `j + p` that `StageHeld` reads; `holoSelOne_rep` selects rep `i`'s rounds of coins
there.
-/

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3
open FlockC3.Reps (CoinsEncoded PointsEncoded repStart)

/-- The link points a session record's points read as: entry `b` of its `a`-th array. -/
def holoPts (mPts : ℕ) (A : Array (Array F128)) : Fin 2 → Fin mPts → GF128 :=
  fun a b => GF128.of ((A[a.val]!)[b.val]!)

theorem holoPts_ofFn {mPts : ℕ} (c : Fin 2 → Fin mPts → GF128) :
    holoPts mPts (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)) = c := by
  funext a b
  have ha : a.val < (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)).size := by
    rw [Array.size_ofFn]; exact a.isLt
  have hb : b.val < (Array.ofFn fun y : Fin mPts => GF128.val (c a y)).size := by
    rw [Array.size_ofFn]; exact b.isLt
  show GF128.of (((Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y))[a.val]!)[b.val]!) =
    c a b
  rw [getElem!_pos (Array.ofFn fun x : Fin 2 => Array.ofFn fun y : Fin mPts => GF128.val (c x y)) a.val ha,
    Array.getElem_ofFn, getElem!_pos (Array.ofFn fun y : Fin mPts => GF128.val (c a y)) b.val hb,
    Array.getElem_ofFn]
  rfl

/-- **The map's link points are the history's first coin** (`PointsEncoded`): `holoPts` of the points `holoMap` gives
at a coin is its coordinate 0. -/
theorem pointsEncoded_holoMap (st : Flock.Stmt) (hp : HoloParams) (Cm : Type) (ish : RecVStar.Inner) (tbl : String)
    (bc : Cm → String → Nat → List UInt8) :
    PointsEncoded (CT := holoCTs st hp) (holoPts (holoAt st hp).cls.mPts) (holoMap st hp Cm ish tbl bc).points := by
  intro h c
  exact congrArg (fun p : Fin 2 → Fin (holoAt st hp).cls.mPts → GF128 => (⟨_, p⟩ : FlockSoundness.Game.Msg))
    (holoPts_ofFn (c ⟨0, h⟩))

/-- **`holoMap`'s draws where a stream round takes one history round**: round `i` draws round `q` of the stream named
`s`, which takes history round `i` alone (`HoloStream.group` 1: a round of coins of a rep's or R's stream, or of one
coin of a sparse phase's). -/
def holoSelOne (ish : RecVStar.Inner) (tbl : String) (i : ℕ) (s : String) (q : ℕ) : Prop :=
  ∃ L e, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L ∧
    L.find? (fun e => e.1.name tbl == s) = some e ∧ e.1.group (e.2.getD q default).1 = 1 ∧ i = roundAt L e.1 q

theorem selC_of_holoSelOne {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} {i : ℕ} {s : String} {q : ℕ} (h : holoSelOne ish tbl i s q) :
    (holoMap st hp Cm ish tbl bc).selC i s q := by
  obtain ⟨L, e, hS, hf, hg, rfl⟩ := h
  exact ⟨L, e, hS, hf, by rw [hg]; exact Nat.one_pos, by rw [hg]⟩

theorem roundWords_take {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) {n j : ℕ}
    (h : j < n) : roundWords (cs.take n) j = roundWords cs j := by
  unfold roundWords
  rw [List.getElem?_take_of_lt h]

/-- **The map's coins are the history's** (`CoinsEncoded`, the `CoinTys`/`vCT` analogue) at `HoloCoinTys`, where a
stream round takes one history round: there the map reads the words of that round's coin off the history's prefix,
and they read back as the coin (`coinArr_roundWords`). -/
theorem coinsEncoded_holoMap {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} {L : List (HoloStream × Array SchedRound)} (hL : HoloCoinTys st hp L) :
    CoinsEncoded (CT := holoCTs st hp) (holoSelOne ish tbl) (holoMap st hp Cm ish tbl bc).coin := by
  intro i hi s q cs hc hsel
  obtain ⟨L', e, hS, hf, hg, rfl⟩ := hsel
  have hcoin : (holoMap st hp Cm ish tbl bc).coin (cs.take (roundAt L' e.1 q + 1)) s q =
      ((List.range (e.1.group (e.2.getD q default).1)).flatMap fun j =>
        (roundWords (cs.take (roundAt L' e.1 q + 1)) (roundAt L' e.1 q + j)).toList).toArray := by
    simp only [holoMap, hS, hf]
  rw [hcoin, hg, List.range_one, List.flatMap_cons, List.flatMap_nil, List.append_nil, Array.toArray_toList,
    Nat.add_zero (roundAt L' e.1 q), roundWords_take cs (show roundAt L' e.1 q < roundAt L' e.1 q + 1 from Nat.lt_succ_self _)]
  exact coinArr_roundWords hL cs _ hi hc

theorem roundIn_cons_self {s : HoloStream} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)} {k : ℕ} :
    roundIn ((s, rs) :: L) s k = heldBefore s rs k :=
  if_pos rfl

theorem roundIn_cons_ne {s' s : HoloStream} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)} {k : ℕ}
    (h : ¬s'.idx = s.idx) : roundIn ((s', rs) :: L) s k = heldBefore s' rs rs.size + roundIn L s k :=
  if_neg h

/-- **The reps' history order at the map read** (lean (i)): in `holoStreams`' order, rep 0's rounds `rs` and then rep
1's, rep `i`'s round `q` is history round `repStart c₀ i` (1, or past rep 0's `c₀` coin rounds) plus rep `i`'s coin
rounds before `q`. -/
theorem roundAt_reps (rs : Array SchedRound) (L : List (HoloStream × Array SchedRound)) (i : Fin 2) (q : ℕ) :
    roundAt ((.rep 0, rs) :: (.rep 1, rs) :: L) (.rep i) q =
      repStart (heldBefore (.rep 0) rs rs.size) i + heldBefore (.rep i) rs q := by
  rcases i with ⟨_ | _ | n, hn⟩
  · exact congrArg (1 + ·) roundIn_cons_self
  · show 1 + roundIn _ (.rep 1) q = 1 + heldBefore (.rep 0) rs rs.size + heldBefore (.rep 1) rs q
    rw [roundIn_cons_ne (s' := .rep 0) (s := .rep 1) (by decide), roundIn_cons_self]
    omega
  · exact absurd hn (by omega)

/-- **Rep `i`'s rounds of coins at the map read**: where the session's streams are rep 0's rounds `rs`, rep 1's and
then the rest (`holoStreams`' order), the stream named `s` is rep `i`'s (`hf`) and its round `q` draws coins,
`holoSelOne` draws it at history round `repStart c₀ i` plus rep `i`'s coin rounds before `q`. -/
theorem holoSelOne_rep {ish : RecVStar.Inner} {tbl s : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)} {i : Fin 2} {q : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hf : ((HoloStream.rep 0, rs) :: (HoloStream.rep 1, rs) :: L).find? (fun e => e.1.name tbl == s) =
      some (.rep i, rs))
    (hq : 0 < (rs.getD q default).1) :
    holoSelOne ish tbl (repStart (heldBefore (.rep 0) rs rs.size) i + heldBefore (.rep i) rs q) s q :=
  ⟨_, _, hS, hf, Nat.min_eq_right hq, (roundAt_reps rs L i q).symm⟩

#print axioms holoPts
#print axioms holoPts_ofFn
#print axioms pointsEncoded_holoMap
#print axioms holoSelOne
#print axioms selC_of_holoSelOne
#print axioms roundWords_take
#print axioms coinsEncoded_holoMap
#print axioms roundIn_cons_self
#print axioms roundIn_cons_ne
#print axioms roundAt_reps
#print axioms holoSelOne_rep

end FlockC3.Compose
