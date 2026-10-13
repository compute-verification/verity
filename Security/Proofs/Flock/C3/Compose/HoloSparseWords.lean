import Proofs.Flock.C3.Compose.HoloSparseRounds
import Proofs.Flock.Level3.ArithFacts

/-!
# The map's coins word by word, where a stream round takes many history rounds

`coinWordsEncoded_holoMap`: where a stream round takes `g` history rounds, each drawing a scalar (`holoSelWords`, the
map's full selector `selC` at rounds of form `none`: a sparse phase's round of `g` coins), the coin the map reads off
the history's prefix is `g` words, the `w`-th of which, as a coin step of one word, is the coin of the `w`-th of those
rounds (`CoinWordsEncoded`). `coinsEncoded_holoMap` is the case of one history round at any form; a sparse stage's coin
steps, one word each, read the coins the verifier holds of a round word by word, which this gives.
-/

namespace FlockC3.Reps

open FlockSoundness.Discharge.PlaysCoin (coinArr)

variable {r : ℕ} {CT : Fin r → Type}

/-- **Hypothesis: the map's coins are the rounds' coordinates, word by word** (`CoinsEncoded` where a stream round
takes `g` history rounds): where round `i` draws round `q` of stream `s` (`selC`), the coins the map reads there off
the coins up to round `i` are `g ≤ i + 1` words, and the `w`-th of them, as a run's coin step of one word, is
coordinate `i + 1 - g + w` of that round's coin. -/
def CoinWordsEncoded (selC : ℕ → String → ℕ → Prop)
    (coin : List ((k : Fin r) → CT k) → String → ℕ → Array Flock.F128) : Prop :=
  ∀ i (hi : i < r) s q (cs : List ((k : Fin r) → CT k)) (hc : i < cs.length), selC i s q →
    ∃ (g : ℕ) (hg : g ≤ i + 1), (coin (cs.take (i + 1)) s q).size = g ∧ ∀ w (hw : w < g),
      coinArr FlockSoundness.Refine.fOf #[(coin (cs.take (i + 1)) s q)[w]!] =
        ⟨CT ⟨i + 1 - g + w, by omega⟩, (cs[i + 1 - g + w]'(by omega)) ⟨i + 1 - g + w, by omega⟩⟩

end FlockC3.Reps

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3
open FlockC3.Reps (CoinWordsEncoded)

/-- **A scalar's words are one**: an array `coinArr` reads as a scalar has one word, a vector of `n ≠ 1` coordinates
having `2^(128 n)` values. -/
theorem size_one_of_scalar {a : Array F128} {c : GF128} (h : coinArr fOf a = ⟨GF128, c⟩) : a.size = 1 := by
  unfold coinArr at h
  split at h
  · assumption
  · have e : (Fin a.size → GF128) = GF128 := congrArg Sigma.fst h
    have hc := Fintype.card_congr (Equiv.cast e)
    rw [Fintype.card_fun, Fintype.card_fin, FlockLevel3.card_F] at hc
    exact (Nat.pow_right_inj (by norm_num)).1 (hc.trans (Nat.pow_one _).symm)

theorem coinArr_single (x : F128) : coinArr fOf #[x] = (⟨GF128, fOf x⟩ : FlockSoundness.Game.Msg) := by
  simp [coinArr]

/-- **A scalar's coin step is its one word's.** -/
theorem coinArr_word {a : Array F128} {c : GF128} (h : coinArr fOf a = ⟨GF128, c⟩) :
    coinArr fOf #[a[0]!] = coinArr fOf a := by
  have hs := size_one_of_scalar h
  rw [h, coinArr_single]
  unfold coinArr at h
  rwa [ite_eq_left hs] at h

theorem toList_of_size_one {a : Array F128} (h : a.size = 1) : a.toList = [a[0]!] := by
  refine List.ext_getElem (by simp [h]) fun n _ h₂ => ?_
  have hn : n = 0 := by
    have h₂' : n < 1 := h₂
    omega
  subst hn
  simp only [Array.getElem_toList, List.getElem_cons_zero, getElem!_pos a 0 (show 0 < a.size by omega)]

theorem flatMap_range_single {f : ℕ → List F128} {h : ℕ → F128} :
    ∀ g, (∀ j < g, f j = [h j]) → (List.range g).flatMap f = (List.range g).map h
  | 0, _ => by simp
  | g + 1, hf => by
    rw [List.range_succ, List.flatMap_append, List.map_append, flatMap_range_single g fun j hj => hf j (by omega),
      List.flatMap_cons, List.flatMap_nil, List.append_nil, hf g (by omega), List.map_cons, List.map_nil]

theorem size_map_range (f : ℕ → F128) (g : ℕ) : ((List.range g).map f).toArray.size = g := by
  rw [List.size_toArray, List.length_map, List.length_range]

theorem getElem!_toArray_map_range {f : ℕ → F128} {g w : ℕ} (hw : w < g) : ((List.range g).map f).toArray[w]! = f w := by
  have hw' : w < ((List.range g).map f).toArray.size := by
    rw [size_map_range]
    exact hw
  rw [getElem!_pos ((List.range g).map f).toArray w hw', List.getElem_toArray, List.getElem_map, List.getElem_range]

theorem coinAt_congr {r : ℕ} {CT : Fin r → Type} (cs : List ((k : Fin r) → CT k)) {a b : ℕ} (hab : a = b)
    (ha : a < r) (hb : b < r) (hca : a < cs.length) (hcb : b < cs.length) :
    (⟨CT ⟨a, ha⟩, cs[a] ⟨a, ha⟩⟩ : FlockSoundness.Game.Msg) = ⟨CT ⟨b, hb⟩, cs[b] ⟨b, hb⟩⟩ := by
  subst hab
  rfl

/-- **`holoMap`'s draws at rounds of scalars**: round `i` draws round `q` of the stream named `s` (`selC`), and each of
the `g` history rounds that stream round takes is of form `none` (`holoForms`, as `holoCTs_succ` reads it): a sparse
phase's rounds. -/
def holoSelWords (st : Flock.Stmt) (hp : HoloParams) (ish : RecVStar.Inner) (tbl : String) (i : ℕ) (s : String)
    (q : ℕ) : Prop :=
  ∃ L e, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L ∧
    L.find? (fun e => e.1.name tbl == s) = some e ∧ 0 < e.1.group (e.2.getD q default).1 ∧
    i + 1 = roundAt L e.1 q + e.1.group (e.2.getD q default).1 ∧
    ∀ j < e.1.group (e.2.getD q default).1, ((holoForms (holoAt st hp))[roundIn L e.1 q + j]?).getD none = none

theorem selC_of_holoSelWords {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} {i : ℕ} {s : String} {q : ℕ} (h : holoSelWords st hp ish tbl i s q) :
    (holoMap st hp Cm ish tbl bc).selC i s q := by
  obtain ⟨L, e, hS, hf, hg, hi, -⟩ := h
  exact ⟨L, e, hS, hf, hg, hi⟩

/-- **The map's coins are the history's, word by word** (`CoinWordsEncoded`) at rounds of scalars: there the map
reads the words of the `g` rounds' coins off the history's prefix, each round's one word (`size_one_of_scalar`), and
each word reads back as its round's coin (`coinArr_roundWords`, `coinArr_word`). -/
theorem coinWordsEncoded_holoMap {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8} :
    CoinWordsEncoded (CT := holoCTs st hp) (holoSelWords st hp ish tbl) (holoMap st hp Cm ish tbl bc).coin := by
  intro i hi s q cs hc hsel
  obtain ⟨L, e, hS, hf, hg0, hi1, hform⟩ := hsel
  have hra : roundAt L e.1 q = 1 + roundIn L e.1 q := rfl
  have one : ∀ j (hj : j < e.1.group (e.2.getD q default).1)
      (hk : 1 + (roundIn L e.1 q + j) < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length)
      (hck : 1 + (roundIn L e.1 q + j) < cs.length),
      (roundWords cs (1 + (roundIn L e.1 q + j))).size = 1 ∧
        coinArr fOf #[(roundWords cs (1 + (roundIn L e.1 q + j)))[0]!] =
          ⟨holoCTs st hp ⟨1 + (roundIn L e.1 q + j), hk⟩, cs[1 + (roundIn L e.1 q + j)] ⟨_, hk⟩⟩ := by
    intro j hj hk hck
    have hA : holoCTs st hp ⟨1 + (roundIn L e.1 q + j), hk⟩ = GF128 :=
      (holoCTs_succ st hp _ hk).trans (congrArg formTy (hform j hj))
    have hm := coinArr_roundWords cs _ hk hck (by rw [hform j hj]; simp)
    have hs : coinArr fOf (roundWords cs (1 + (roundIn L e.1 q + j))) =
        ⟨GF128, cast hA (cs[1 + (roundIn L e.1 q + j)] ⟨_, hk⟩)⟩ :=
      hm.trans (Sigma.ext hA (cast_heq hA _).symm)
    exact ⟨size_one_of_scalar hs, (coinArr_word hs).trans hm⟩
  have hwords : ∀ j < e.1.group (e.2.getD q default).1,
      (roundWords (cs.take (i + 1)) (roundAt L e.1 q + j)).toList =
        [(roundWords cs (1 + (roundIn L e.1 q + j)))[0]!] := by
    intro j hj
    rw [roundWords_take cs (show roundAt L e.1 q + j < i + 1 by omega), hra, Nat.add_assoc]
    exact toList_of_size_one (one j hj (by omega) (by omega)).1
  have hcoin : (holoMap st hp Cm ish tbl bc).coin (cs.take (i + 1)) s q =
      ((List.range (e.1.group (e.2.getD q default).1)).map fun j =>
        (roundWords cs (1 + (roundIn L e.1 q + j)))[0]!).toArray := by
    have h0 : (holoMap st hp Cm ish tbl bc).coin (cs.take (i + 1)) s q =
        ((List.range (e.1.group (e.2.getD q default).1)).flatMap fun j =>
          (roundWords (cs.take (i + 1)) (roundAt L e.1 q + j)).toList).toArray := by
      simp only [holoMap, hS, hf]
    rw [h0, flatMap_range_single _ hwords]
  refine ⟨e.1.group (e.2.getD q default).1, by omega, by rw [hcoin, size_map_range], fun w hw => ?_⟩
  rw [hcoin, getElem!_toArray_map_range hw]
  exact (one w hw (by omega) (by omega)).2.trans
    (coinAt_congr cs (by omega) (by omega) (by omega) (by omega) (by omega))

end FlockC3.Compose
