import Proofs.Flock.C3.Compose.HoloSparseRounds
import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.C3.VStarList.Range

/-!
# The map's coins word by word, where a stream round takes many history rounds

`coinWordsEncoded_holoMap`: where a stream round takes `g` history rounds, each drawing a scalar (`holoSelWords`, the
map's full selector `selC` at rounds of form `none`: a sparse phase's round of `g` coins), the coin the map reads off
the history's prefix is `g` words, the `w`-th of which, as a coin step of one word, is the coin of the `w`-th of those
rounds (`CoinWordsEncoded`). `coinsEncoded_holoMap` is the case of one history round at any form; a sparse stage's coin
steps, one word each, read the coins the verifier holds of a round word by word, which this gives.

`holoForms_sparse_none`: `holoSelWords`' last conjunct at a sparse phase's round, where the model's rep forms are the
verifier's (`repFormsAt`, `repForms`): past both reps' rounds (`heldBefore_rep_forms`) the round's history rounds are
inside the phase's `sparseForms`, every one `none`. `repFormsAt_holoAt` is `hA` where the inner statement's class
checks pass and the inner shape is the statement's (`checkInnerStmt`), at its claims `2 · regions + 2` (`hcl`).
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

/-- **One rep's coins' forms in `holoForms`**, at the class's public statement. -/
noncomputable def repFormsAt (e : ExecHolo) : List (Option ℕ) :=
  zcForms (e.cls.pubStatement e.regs).m ++ (lcForms (e.cls.pubStatement e.regs).kLog ++
    (opForms (2 * (e.cls.pubStatement e.regs).regions.length + 2) ++ ligForms e.sch))

theorem holoForms_eq (e : ExecHolo) : holoForms e = repFormsAt e ++ (repFormsAt e ++
    (sparseForms e.cls.shape ++ (sparseForms e.cls.shape ++ (opForms 2 ++ ligForms e.rsch.sch)))) := rfl

theorem sum_range_mono (f : ℕ → ℕ) {a b : ℕ} (h : a ≤ b) :
    ((List.range a).map f).sum ≤ ((List.range b).map f).sum := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    omega

theorem heldBefore_le (s : HoloStream) (rs : Array SchedRound) (k : ℕ) : heldBefore s rs k ≤ heldBefore s rs rs.size := by
  unfold heldBefore
  rw [Nat.min_self]
  exact sum_range_mono _ (Nat.min_le_right _ _)

theorem heldBefore_succ (s : HoloStream) (rs : Array SchedRound) {q : ℕ} (hq : q < rs.size) :
    heldBefore s rs (q + 1) = heldBefore s rs q + s.group (rs.getD q default).1 := by
  unfold heldBefore
  rw [Nat.min_eq_left (show q + 1 ≤ rs.size by omega), Nat.min_eq_left (Nat.le_of_lt hq), List.range_succ,
    List.map_append, List.sum_append]
  simp

theorem nzSizes_length (x : Fin 2) : ∀ l : List SchedRound,
    (l.map fun q => (HoloStream.rep x).group q.1).sum = (nzSizes l).length
  | [] => rfl
  | q :: l => by
    rw [List.map_cons, List.sum_cons, nzSizes_length x l]
    simp only [nzSizes, List.flatMap_cons, List.length_append, HoloStream.group]
    split
    · simp_all
    · simp_all
      omega

/-- **A rep's rounds take as many history rounds as it has coin forms**, at a claim (`holoSizes_rep`). -/
theorem heldBefore_rep_forms (x : Fin 2) {m kLog claims : ℕ} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRounds m kLog claims = some rs) : heldBefore (.rep x) rs rs.size = (repForms m kLog claims).length := by
  have e := congrArg List.length (holoSizes_rep x hc h)
  rw [holoSizes_rep_nz, List.length_map] at e
  rw [heldBefore_size, ← e]
  exact nzSizes_length x rs.toList

theorem getD_sparseForms {A S rest : List (Option ℕ)} (hS : ∀ y ∈ S, y = none) (x : Fin 2) {k : ℕ}
    (hk : k < S.length) : ((A ++ (A ++ (S ++ (S ++ rest))))[2 * A.length + x.val * S.length + k]?).getD none = none := by
  rw [List.getElem?_append_right (by omega), List.getElem?_append_right (by omega),
    show 2 * A.length + x.val * S.length + k - A.length - A.length = x.val * S.length + k by omega]
  have hS' : (S[k]?).getD none = none := by
    rw [List.getElem?_eq_getElem hk, Option.getD_some]
    exact hS _ (List.getElem_mem hk)
  rcases (show x.val = 0 ∨ x.val = 1 by omega) with h | h
  · rw [h, Nat.zero_mul, Nat.zero_add, List.getElem?_append_left hk]
    exact hS'
  · rw [h, Nat.one_mul, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, List.getElem?_append_left hk]
    exact hS'

/-- **A sparse phase's rounds are of form `none`** (`holoSelWords`' last conjunct): past both reps' coins, sparse phase
`x`'s round `q` takes history rounds inside phase `x`'s `sparseForms`, where the model's rep forms are the verifier's
(`hA`) and its sparse shape the inner shape's `sparse` entry's (`hshape`, `shape_holoAt`). -/
theorem holoForms_sparse_none {st : Flock.Stmt} {hp : HoloParams} {ish : RecVStar.Inner} {sp : RecSparse.Shape}
    {L : List (HoloStream × Array SchedRound)} {e : HoloStream × Array SchedRound} {x : Fin 2} {q : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L) (hsp : ish.sparse = some sp)
    (hc : 0 < ish.claims) (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (he : e ∈ L) (hx : e.1 = .sparse x) :
    ∀ j < e.1.group (e.2.getD q default).1, ((holoForms (holoAt st hp))[roundIn L e.1 q + j]?).getD none = none := by
  rw [hsp] at hS
  obtain ⟨rs, R, hr, hR, rfl⟩ := holoStreams_sparse_eq hS
  obtain ⟨e1, e2⟩ := e
  dsimp only at hx
  subst hx
  have he2 : e2 = sparseRounds sp := by
    simp only [List.mem_cons] at he
    rcases he with h | h | h | h | h
    · cases h
    · cases h
    · cases h; rfl
    · cases h; rfl
    · cases hR _ h
  subst he2
  intro j hj
  dsimp only at hj ⊢
  have hq : q < (sparseRounds sp).size := by
    refine Decidable.by_contra fun hq => ?_
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)] at hj
    exact absurd hj (Nat.not_lt_zero _)
  have hle := heldBefore_le (.sparse x) (sparseRounds sp) (q + 1)
  rw [heldBefore_sparse_size, heldBefore_succ _ _ hq] at hle
  have hg : (HoloStream.sparse x).group ((sparseRounds sp).getD q default).1 =
      ((sparseRounds sp).getD q default).1 := rfl
  have hn : FlockVBridge.Sparse.nc (sparseOf sp) = (sparseForms (sparseOf sp)).length := by
    simp only [sparseForms, List.length_replicate, FlockVBridge.Sparse.nc, FlockVBridge.Sparse.nR]
    omega
  have hround := roundAt_sparseRounds rs sp R x q
  unfold roundAt at hround
  rw [hn] at hround
  have hrep := heldBefore_rep_forms 0 hc hr
  rw [holoForms_eq, hA, hshape, show roundIn ((.rep 0, rs) :: (.rep 1, rs) :: (.sparse 0, sparseRounds sp) ::
      (.sparse 1, sparseRounds sp) :: R) (.sparse x) q + j = 2 * (repForms ish.m ish.kLog ish.claims).length +
      x.val * (sparseForms (sparseOf sp)).length + (heldBefore (.sparse x) (sparseRounds sp) q + j) by omega]
  exact getD_sparseForms (S := sparseForms (sparseOf sp)) (fun y hy => List.eq_of_mem_replicate hy) x (by omega)

/-- **The model's rep forms are the verifier's** (`holoForms_sparse_none`'s `hA`): where the inner statement's class
checks pass (`hc`) and the inner shape's `m`, `k_log` and regions are the statement's (`checkInnerStmt_ok`), at the
inner shape's claims `2 · regions + 2` (`hcl`), the class's claims. -/
theorem repFormsAt_holoAt {st : Flock.Stmt} {hp : HoloParams} {ish : RecVStar.Inner} (hc : ClassOk (shapeOf st))
    (hi : RecVStar.checkInnerStmt ish st = .ok ()) (hcl : ish.claims = 2 * ish.regions + 2) :
    repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims := by
  obtain ⟨-, hm, hk, hr⟩ := FlockC3.VStarList.checkInnerStmt_ok hi
  have hm' : (execClassAt st).cls.m = st.m := by
    rw [execClassAt, dite_eq_left hc]
    rfl
  have hk' := FlockC3.VStarList.execClassAt_kLog hc
  have hl : (execClassAt st).regs.length = st.regions.size := by
    rw [execClassAt_regs_length, ite_eq_left hc]
    exact Array.size_map ..
  have hs : (execClassAt st).sch = schedAt st.m := by
    rw [execClassAt_sch, ite_eq_left hc]
    rfl
  unfold repFormsAt repForms
  show zcForms (execClassAt st).cls.m ++ (lcForms (execClassAt st).cls.kLog ++
    (opForms (2 * (execClassAt st).regs.length + 2) ++ ligForms (execClassAt st).sch)) = _
  rw [show zcForms (execClassAt st).cls.m = zcForms ish.m from congrArg zcForms (hm'.trans hm.symm),
    show lcForms (execClassAt st).cls.kLog = lcForms ish.kLog from congrArg lcForms (hk'.trans hk.symm),
    show opForms (2 * (execClassAt st).regs.length + 2) = opForms ish.claims by rw [hl, hcl, hr],
    show ligForms (execClassAt st).sch = ligForms (schedAt ish.m) by rw [hs, hm]]

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
