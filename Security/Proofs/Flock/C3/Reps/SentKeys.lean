import Proofs.Flock.C3.Reps.IoCols
import Proofs.Flock.C3.Reps.RepCoins
import Proofs.Flock.C3.Sparse.MsgsRead

/-!
# The reps' stage keys, `WCombRead` from the reads file, and `TsRead` from the weave

Stage `s`'s keys for rep `ρ` are the keys of the messages rep `ρ` sends at stage `s` (`sentK`, at each message's keys
`keys m`), which are never `.cons` (`sentK_noCons`): no sent message is a consistency sum. So `AlphaRead` and
`RowColsRead`, which quantify only over `.cons` keys, hold vacuously there (`alphaRead_of_noCons`,
`RepMaps.rowColsRead_of_noCons`), and `formsRead_maps` needs only `WCombRead` (`formsRead_sent`).

* `wCombRead_of`: `WCombRead` from the decoder's columns (`WColsAt`: `rep₀` reads each key off input columns of one copy,
  keyed by row bits of ports stage `s` reads, `SelPort`), through `copyVal_maskOf` as in `rowColsRead_of`.
* `hOfW_valOf`, `hOfW_read`: at a holo shape `hOfW sh w`'s entry `l` is `valOf` at `.h l`, so `WCombRead` at the `.h l`
  keys reads it.
* `tsRead_of_woven`: where each rep's steps are a weave along one `σ` of messages and coins of fixed lengths (`WovenAt`),
  rep `ρ`'s message `m` is at stage `ρ · c₀ + roundOf σ m + 1` (`stageOf`, `c₀` rep 0's coins, `coins₀`), and each
  message is read on its stage's commit strings (`WeaveMsgsRead`), `TsRead` holds (`nCoins_weave`, `groups_weave`).
* `weaveMsgsRead_of_keys`: `WeaveMsgsRead` from `WCombRead` at `sentK` and each message's reading only its keys
  (`MsgsOfKeys`, the slices). At `repsOfVals` the weave is `unitWeave`'s (`wovenAt_repsOfVals`, its extra claims'
  count fixed by `extraClaims_length`), so `tsRead_repsOfVals` gives `TsRead` there from the lengths, `MsgsOfKeys` and
  `WColsAt`.
* The slices, at `ValsSrc`'s values: a rep's first `m_z − 4` messages, the zerocheck's, read only `zcKeys`
  (`unitMsgs_zc_read`), its next `k_log − 4`, the lincheck's with `h`, only `lcKeys` at a holo shape
  (`unitMsgs_lc_read`), and its next `2 + n_X`, the ring switches', only `rsKeys` (`unitMsgs_rs_read`). No key of
  any is `.cons`, so `rep₀`'s consistency sums stand in for the rep's (`valOf_cons_irrel`). The Ligerito slice's
  leaves: G's out-of-domain value `i` reads only `.ood i` (`ligOod_read`), its sumcheck message `i` only `.msg i`
  (`ligMsg_read`), and its final `yr` only the `.yr` keys (`yrOf_read`). So a rep's last messages, its Ligerito
  run's (`unitMsgs_lig`), read only `ligKeys`' entries (`unitMsgs_lig_read`, by `PwRead` along `ligMsgs` and
  `recMsgs`' levels), the recursive caps, which no key covers, read alike (`RecCapsRead`).
* `wColsAt_of_w`: `WColsAt` from L's side of the decoder (`WDecodes`: `rep₀` reads each element of `w` off columns),
  S's reads file per key (`WSel`, `SelPort`) and the keys being elements of `w` (`WKeys`, `valOf_w`).
* At `ValsSrc`'s values: `hb_of_shape` gives `tsRead_valsSrc`' `hb` from V*'s shape; a constant source cap gives
  `CapRead` (`capRead_of_const`, reading no history) and the cap tie (`capTie_vals`); `RepsDrawn` at the identity `ix`
  is a statement on `b` alone (`DrawnOf`, `repsDrawn_valsSrc`), the rep's consumed rounds `repIdxB` starting at 0.
-/

namespace FlockC3.Reps

noncomputable section

section unit

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

open Flock FlockSoundness FlockSoundness.Refine FlockC3.HoloValues
open Flock.RecAlgebra (hOfW)
open FlockSoundness.Discharge.Exec (weave)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- `unitWeave`'s shape at `S`, `nX` extra claims and `sch₀`. -/
def unitShape (S : Model.Statement) (sch₀ : Accounting.Schedule) (nX : ℕ) : List Discharge.Exec.Kind :=
  holoRepShape 𝔽 𝕂 Dig S.m S.kLog nX sch₀

/-- `unitWeave`'s messages at `nX` extra claims. -/
def unitMsgs (nX : ℕ) (sh : Flock.Shape) (d : FlockVBridge.Compiled.RepData) (v : UnitVals) : List Game.Msg :=
  holoRepMsgs v.st v.sch (2 + nX) v.p (hOfW sh d.w) capDig

/-- `unitWeave`'s coins at `S`. -/
def unitCoins (S : Model.Statement) (v : UnitVals) : List Game.Msg :=
  (consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
    consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf)

theorem unitWeave_eq (S : Model.Statement) (sch₀ : Accounting.Schedule) (nX : ℕ) (sh : Flock.Shape)
    (d : FlockVBridge.Compiled.RepData) (v : UnitVals) :
    unitWeave S sch₀ nX sh d v = weave (unitShape S sch₀ nX) (unitMsgs nX sh d v) (unitCoins S v) := rfl

/-- **A rep's extra claims are two per region, whatever the points.** -/
theorem extraClaims_length (S : Model.Statement) (a b : ℕ) (pts pts' : Fin 2 → Fin b → 𝔽) :
    (Model.extraClaims S a b pts).length = (Model.extraClaims S a b pts').length := by
  simp only [Model.extraClaims, List.length_flatMap, List.length_cons, List.length_nil]

/-- **A recursive level's messages' count**, whatever the proof, the caps and the indices. -/
theorem recMsgs_length {D D' : Type} (p p' : LigeritoProof) (capD : Array ByteArray → Model.Cap D)
    (capD' : Array ByteArray → Model.Cap D') (ext : ℕ) : ∀ left i i' msg msg' ood ood' : ℕ,
    (recMsgs p capD ext left i msg ood).length = (recMsgs p' capD' ext left i' msg' ood').length
  | 0, _, _, _, _, _, _ => rfl
  | 1, _, _, _, _, _, _ => by
      simp only [recMsgs, kMsgs, List.length_append, List.length_map, List.length_range', List.length_cons,
        List.length_nil]
  | left + 2, i, i', msg, msg', ood, ood' => by
      simp only [recMsgs, kMsgs, List.length_append, List.length_map, List.length_range', List.length_cons,
        List.length_nil, recMsgs_length p p' capD capD' ext (left + 1) (i + 1) (i' + 1) (msg + 7) (msg' + 7) (ood + 2)
          (ood' + 2)]

/-- **A Ligerito run's messages' count** is its schedule's. -/
theorem ligMsgs_length {D D' : Type} (p p' : LigeritoProof) (capD : Array ByteArray → Model.Cap D)
    (capD' : Array ByteArray → Model.Cap D') (sch : Flock.Schedule) :
    (ligMsgs p capD sch).length = (ligMsgs p' capD' sch).length := by
  simp only [ligMsgs, kMsgs, List.length_append, List.length_map, List.length_range', List.length_cons,
    List.length_nil, recMsgs_length p p' capD capD' sch.ext (sch.levels.size - 1) 0 0 (sch.k0 + 4) (sch.k0 + 4) 3 3]

/-- **A rep's messages' count** is its setup's and schedule's, whatever its proof and values. -/
theorem unitMsgs_length (nX : ℕ) {sh sh' : Flock.Shape} {d d' : FlockVBridge.Compiled.RepData} {v v' : UnitVals}
    (hst : v.st = v'.st) (hs : v.sch = v'.sch) : (unitMsgs nX sh d v).length = (unitMsgs nX sh' d' v').length := by
  simp only [unitMsgs, holoRepMsgs, holoLcMsgs, zcMsgs, lcMsgs, opMsgs, List.length_append, List.length_cons,
    List.length_map, List.length_range, List.length_nil, hst, hs,
    ligMsgs_length v.p.ligerito v'.p.ligerito capDig capDig v'.sch]

/-- **A rep's coins' count** is its setup's and its transcripts' rounds'. -/
theorem unitCoins_length (S : Model.Statement) {v v' : UnitVals} (hk : v.t.k = v'.t.k) (hst : v.st = v'.st)
    (h₄ : v.t₄.k = v'.t₄.k) : (unitCoins S v).length = (unitCoins S v').length := by
  simp only [unitCoins, consumed, List.length_map, List.length_append, List.length_range', hk, hst, h₄]

/-- **A rep's messages' count** reads only its setup's `m` and `k_log` and its schedule. -/
theorem unitMsgs_length_of (nX : ℕ) {sh sh' : Flock.Shape} {d d' : FlockVBridge.Compiled.RepData}
    {v v' : UnitVals} (hm : v.st.m = v'.st.m) (hk : v.st.kLog = v'.st.kLog) (hs : v.sch = v'.sch) :
    (unitMsgs nX sh d v).length = (unitMsgs nX sh' d' v').length := by
  simp only [unitMsgs, holoRepMsgs, holoLcMsgs, zcMsgs, lcMsgs, opMsgs, List.length_append, List.length_cons,
    List.length_map, List.length_range, List.length_nil, hm, hk, hs,
    ligMsgs_length v.p.ligerito v'.p.ligerito capDig capDig v'.sch]

/-- **A rep's coins' count** reads only its setup's `m` and its transcripts' rounds. -/
theorem unitCoins_length_of (S : Model.Statement) {v v' : UnitVals} (hk : v.t.k = v'.t.k) (hm : v.st.m = v'.st.m)
    (h₄ : v.t₄.k = v'.t₄.k) : (unitCoins S v).length = (unitCoins S v').length := by
  simp only [unitCoins, consumed, List.length_map, List.length_append, List.length_range', hk, hm, h₄]

end unit

open Flock

/-- **`hOfW`'s entries are `valOf`'s at `.h`** at a holo shape. -/
theorem hOfW_valOf {sh : Flock.Shape} (hh : sh.holo = true) (w comb : Array F128) (c : ℕ → ℕ → F128) {l : ℕ}
    (hl : l < 64) : (Flock.RecAlgebra.hOfW sh w)[l]! = Flock.RecAlgebra.valOf sh w comb c (.h l) := by
  rw [Flock.RecAlgebra.hOfW_get sh w hl, Flock.RecAlgebra.hOf, ite_eq_left hh]
  rfl

/-- A placeholder message, to count a weave's coins from its lengths. -/
def msg₀ : FlockSoundness.Game.Msg := ⟨Bool, true⟩

/-- Rep 0's coins along `σ` at `nm` messages and `nc` coins (whatever they are, `nCoins_weave`). -/
def coins₀ (σ : List FlockSoundness.Discharge.Exec.Kind) (nm nc : ℕ) : ℕ :=
  nCoins (FlockSoundness.Discharge.Exec.weave σ (List.replicate nm msg₀) (List.replicate nc msg₀))

/-- **Message `m`'s stage** for rep `ρ`: rep 0's round `j` is stage `j + 1`, rep 1's is stage `c₀ + 1 + j`. -/
def stageOf (σ : List FlockSoundness.Discharge.Exec.Kind) (c₀ : ℕ) (ρ : Fin 2) (m : ℕ) : ℕ :=
  ρ.val * c₀ + FlockC3.Sparse.roundOf σ m + 1

/-- **Rep `ρ`'s stage-`s` keys**: the keys `keys m` of its messages `m` at stage `s`, none a consistency sum. -/
def sentK (σ : List FlockSoundness.Discharge.Exec.Kind) (c₀ : ℕ) (keys : ℕ → List Flock.RecAlgebra.Key) (s : ℕ)
    (ρ : Fin 2) (x : Flock.RecAlgebra.Key) : Prop :=
  ∃ m, stageOf σ c₀ ρ m = s ∧ x ∈ keys m ∧ ∀ ℓ l, x ≠ .cons ℓ l

/-- **No sent key is `.cons`.** -/
theorem sentK_noCons {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ} {keys : ℕ → List Flock.RecAlgebra.Key} :
    ∀ s ρ ℓ l, ¬ sentK σ c₀ keys s ρ (.cons ℓ l) := by
  rintro _ _ ℓ l ⟨_, _, _, h⟩
  exact h ℓ l rfl

/-! ## The slices: the zerocheck's and the lincheck's messages read only their keys -/

section slices

universe u

open FlockSoundness.Refine (zcMsgs lcMsgs holoLcMsgs holoRepMsgs opMsgs vecOf pairOf fOf)
open FlockVBridge.Compiled (RepData)

theorem list3_get {α : Type u} {a a' z z' : α} {f f' : ℕ → α} {n m : ℕ} (h0 : m = 0 → a = a')
    (h1 : ∀ i, m = i + 1 → i < n → f i = f' i) (h2 : m = n + 1 → z = z') :
    (a :: (List.range n).map f ++ [z])[m]? = (a' :: (List.range n).map f' ++ [z'])[m]? := by
  rw [List.cons_append, List.cons_append]
  rcases m with _ | m
  · rw [List.getElem?_cons_zero, List.getElem?_cons_zero, h0 rfl]
  rw [List.getElem?_cons_succ, List.getElem?_cons_succ]
  rcases Nat.lt_or_ge m n with hm | hm
  · rw [List.getElem?_append_left (by rw [List.length_map, List.length_range]; exact hm),
      List.getElem?_append_left (by rw [List.length_map, List.length_range]; exact hm), List.getElem?_map,
      List.getElem?_map, List.getElem?_range hm, Option.map_some, Option.map_some, h1 m rfl hm]
  · rw [List.getElem?_append_right (by rw [List.length_map, List.length_range]; exact hm),
      List.getElem?_append_right (by rw [List.length_map, List.length_range]; exact hm), List.length_map,
      List.length_map, List.length_range]
    rcases Nat.eq_or_lt_of_le hm with rfl | hm'
    · rw [Nat.sub_self, h2 rfl]
    · rw [List.getElem?_eq_none (by rw [List.length_singleton]; omega),
        List.getElem?_eq_none (by rw [List.length_singleton]; omega)]

theorem list3_length {α : Type u} (a z : α) (f : ℕ → α) (n : ℕ) :
    (a :: (List.range n).map f ++ [z]).length = n + 2 := by
  rw [List.length_append, List.length_cons, List.length_map, List.length_range, List.length_singleton]

theorem append_get_right {α : Type u} {A A' B B' : List α} {m : ℕ} (hm : A.length ≤ m) (hA : A.length = A'.length)
    (h : B[m - A.length]? = B'[m - A.length]?) : (A ++ B)[m]? = (A' ++ B')[m]? := by
  rw [List.getElem?_append_right hm, List.getElem?_append_right (hA ▸ hm), ← hA, h]

theorem list2_get {α : Type u} {f f' : ℕ → α} {y y' z z' : α} {n m : ℕ}
    (h1 : m < n → f m = f' m) (h2 : m = n → y = y') (h3 : m = n + 1 → z = z') :
    ((List.range n).map f ++ [y] ++ [z])[m]? = ((List.range n).map f' ++ [y'] ++ [z'])[m]? := by
  rcases Nat.lt_or_ge m n with hm | hm
  · have hl : m < ((List.range n).map f ++ [y]).length := by
      rw [List.length_append, List.length_map, List.length_range, List.length_singleton]; omega
    have hl' : m < ((List.range n).map f' ++ [y']).length := by
      rw [List.length_append, List.length_map, List.length_range, List.length_singleton]; omega
    rw [List.getElem?_append_left hl, List.getElem?_append_left hl',
      List.getElem?_append_left (by rw [List.length_map, List.length_range]; exact hm),
      List.getElem?_append_left (by rw [List.length_map, List.length_range]; exact hm), List.getElem?_map,
      List.getElem?_map, List.getElem?_range hm, Option.map_some, Option.map_some, h1 hm]
  · rw [List.append_assoc, List.append_assoc]
    refine append_get_right (by rw [List.length_map, List.length_range]; exact hm)
      (by rw [List.length_map, List.length_map]) ?_
    rw [List.length_map, List.length_range, List.singleton_append, List.singleton_append]
    rcases Nat.eq_or_lt_of_le hm with rfl | hm'
    · rw [Nat.sub_self, List.getElem?_cons_zero, List.getElem?_cons_zero, h2 rfl]
    · obtain ⟨j, rfl⟩ : ∃ j, m = n + 1 + j := ⟨m - (n + 1), by omega⟩
      rw [show n + 1 + j - n = j + 1 by omega, List.getElem?_cons_succ, List.getElem?_cons_succ]
      rcases j with _ | j
      · rw [List.getElem?_cons_zero, List.getElem?_cons_zero, h3 rfl]
      · rw [List.getElem?_eq_none (by rw [List.length_singleton]; omega),
          List.getElem?_eq_none (by rw [List.length_singleton]; omega)]

theorem list2_length {α : Type u} (f : ℕ → α) (n : ℕ) (y z : α) :
    ((List.range n).map f ++ [y] ++ [z]).length = n + 2 := by
  rw [List.length_append, List.length_append, List.length_map, List.length_range, List.length_singleton,
    List.length_singleton]

/-- **Evaluated variables read only their keys.** -/
theorem vars_read {K L F : Type} (var : K → L) (eval : (K → F) → L → F) (heval : ∀ v k, eval v (var k) = v k)
    (n : ℕ) (key : ℕ → K) {v v' : K → F} (h : ∀ i < n, v (key i) = v' (key i)) :
    ((Array.range n).map (var ∘ key)).map (eval v) = ((Array.range n).map (var ∘ key)).map (eval v') :=
  Array.map_inj_left.2 fun f hf => by
    obtain ⟨i, hi, rfl⟩ := Array.mem_map.1 hf
    rw [Function.comp_apply, heval, heval]
    exact h i (Array.mem_range.1 hi)

/-- **An evaluated pair of variables reads only its two keys.** -/
theorem pairs_read {K L F : Type} [Inhabited F] (var : K → L) (eval : (K → F) → L → F)
    (heval : ∀ v k, eval v (var k) = v k) (n i : ℕ) (k₁ k₂ : ℕ → K) {v v' : K → F}
    (h : i < n → v (k₁ i) = v' (k₁ i) ∧ v (k₂ i) = v' (k₂ i)) :
    (((Array.range n).map fun j => (var (k₁ j), var (k₂ j))).map fun p => (eval v p.1, eval v p.2))[i]! =
      (((Array.range n).map fun j => (var (k₁ j), var (k₂ j))).map fun p => (eval v' p.1, eval v' p.2))[i]! := by
  by_cases hi : i < n
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_some, heval,
      (h hi).1, (h hi).2]
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_none]

/-- **An evaluated row of variables reads only its keys.** -/
theorem rows_read {K L F : Type} (var : K → L) (eval : (K → F) → L → F) (heval : ∀ v k, eval v (var k) = v k)
    (n c i : ℕ) (key : ℕ → ℕ → K) {v v' : K → F} (h : ∀ j < c, v (key i j) = v' (key i j)) :
    (((Array.range n).map fun k => (Array.range c).map (var ∘ key k)).map (·.map (eval v)))[i]! =
      (((Array.range n).map fun k => (Array.range c).map (var ∘ key k)).map (·.map (eval v')))[i]! := by
  by_cases hi : i < n
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_some]
    exact vars_read var eval heval c (key i) h
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_none]

theorem map_range_get {α : Type u} {f f' : ℕ → α} {n m : ℕ} (h : m < n → f m = f' m) :
    ((List.range n).map f)[m]? = ((List.range n).map f')[m]? := by
  rcases Nat.lt_or_ge m n with hm | hm
  · rw [List.getElem?_map, List.getElem?_map, List.getElem?_range hm, Option.map_some, Option.map_some, h hm]
  · rw [List.getElem?_eq_none (by rw [List.length_map, List.length_range]; exact hm),
      List.getElem?_eq_none (by rw [List.length_map, List.length_range]; exact hm)]

/-- **A key that is not a consistency sum reads no consistency sum.** -/
theorem valOf_cons_irrel (sh : Flock.Shape) (w comb : Array F128) (c c' : ℕ → ℕ → F128)
    {x : Flock.RecAlgebra.Key} (hx : ∀ ℓ l, x ≠ .cons ℓ l) :
    Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w comb c' x := by
  cases x with
  | cons ℓ l => exact absurd rfl (hx ℓ l)
  | _ => rfl

/-- **A key `w` holds is its element of `w`**: not a ring switch's transpose, a comb element or a consistency sum. -/
theorem valOf_w {sh : Flock.Shape} {w comb : Array F128} {c : ℕ → ℕ → F128} {x : Flock.RecAlgebra.Key} {i : ℕ}
    (hi : Flock.RecAlgebra.wi sh x = some i) (h₁ : ∀ k u, x ≠ .rsT k u) (h₂ : ∀ l, x ≠ .comb l)
    (h₃ : ∀ ℓ l, x ≠ .cons ℓ l) : Flock.RecAlgebra.valOf sh w comb c x = w[i]! := by
  have e : Flock.RecAlgebra.valOf sh w comb c x =
      match Flock.RecAlgebra.wi sh x with
      | some j => w[j]!
      | none => 0 := by
    cases x with
    | rsT k u => exact absurd rfl (h₁ k u)
    | comb l => exact absurd rfl (h₂ l)
    | cons ℓ l => exact absurd rfl (h₃ ℓ l)
    | _ => rfl
  rw [e]
  simp only [hi]

/-- The keys of the zerocheck's message `m` at `m_z` variables: `(ab, c)`, round `m − 1`'s `(g1, ginf)`, then
`(fa, fb)`. -/
def zcKeys (mz : ℕ) : ℕ → List Flock.RecAlgebra.Key
  | 0 => (List.range 64).map .ab ++ (List.range 64).map .c
  | i + 1 => if i < mz - 6 then [.g1 i, .ginf i] else [.fa, .fb]

theorem zcKeys_noCons {mz m : ℕ} {x : Flock.RecAlgebra.Key} (hx : x ∈ zcKeys mz m) : ∀ ℓ l, x ≠ .cons ℓ l := by
  rintro ℓ l rfl
  rcases m with _ | i
  · simp [zcKeys] at hx
  · by_cases hi : i < mz - 6
    · simp [zcKeys, hi] at hx
    · simp [zcKeys, hi] at hx

/-- **The zerocheck's message `m` of G's proof file reads only `zcKeys m`.** -/
theorem zcMsgs_read (mz : ℕ) (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} (m : ℕ)
    (h : ∀ x ∈ zcKeys mz m,
      Flock.RecAlgebra.valOf sh d.w d.comb d.cons x = Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons x) :
    (zcMsgs mz (FlockVBridge.Family.proofOf sh params cap lig d).zerocheck)[m]? =
      (zcMsgs mz (FlockVBridge.Family.proofOf sh params' cap' lig' d').zerocheck)[m]? := by
  unfold zcMsgs
  apply list3_get
  · intro h0
    subst h0
    have hab : (RecAlgebra.decoded sh d.w d.comb d.cons).ab = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).ab :=
      vars_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var 64 RecAlgebra.Key.ab fun i hi =>
        h _ (List.mem_append_left _ (List.mem_map_of_mem (List.mem_range.2 hi)))
    have hc : (RecAlgebra.decoded sh d.w d.comb d.cons).c = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).c :=
      vars_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var 64 RecAlgebra.Key.c fun i hi =>
        h _ (List.mem_append_right _ (List.mem_map_of_mem (List.mem_range.2 hi)))
    show (⟨_, (vecOf 64 (RecAlgebra.decoded sh d.w d.comb d.cons).ab,
        vecOf 64 (RecAlgebra.decoded sh d.w d.comb d.cons).c)⟩ : FlockSoundness.Game.Msg) =
      ⟨_, (vecOf 64 (RecAlgebra.decoded sh d'.w d'.comb d'.cons).ab,
        vecOf 64 (RecAlgebra.decoded sh d'.w d'.comb d'.cons).c)⟩
    rw [hab, hc]
  · intro i hi hin
    subst hi
    have hz : (RecAlgebra.decoded sh d.w d.comb d.cons).zc[i]! = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).zc[i]! :=
      pairs_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var (sh.m - 6) i RecAlgebra.Key.g1
        RecAlgebra.Key.ginf fun _ => by
          have hk : zcKeys mz (i + 1) = [.g1 i, .ginf i] := ite_eq_left hin
          exact ⟨h _ (hk ▸ List.mem_cons_self), h _ (hk ▸ List.mem_cons_of_mem _ List.mem_cons_self)⟩
    show (⟨_, pairOf (RecAlgebra.decoded sh d.w d.comb d.cons).zc[i]!⟩ : FlockSoundness.Game.Msg) =
      ⟨_, pairOf (RecAlgebra.decoded sh d'.w d'.comb d'.cons).zc[i]!⟩
    rw [hz]
  · intro h2
    subst h2
    have hk : zcKeys mz (mz - 6 + 1) = [.fa, .fb] := ite_eq_right (Nat.lt_irrefl _)
    have hfa : (RecAlgebra.decoded sh d.w d.comb d.cons).fa = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).fa := by
      show RecAlgebra.Lin.eval (RecAlgebra.valOf sh d.w d.comb d.cons) (RecAlgebra.Lin.var .fa) =
        RecAlgebra.Lin.eval (RecAlgebra.valOf sh d'.w d'.comb d'.cons) (RecAlgebra.Lin.var .fa)
      rw [RecAlgebra.Lin.eval_var, RecAlgebra.Lin.eval_var]
      exact h _ (hk ▸ List.mem_cons_self)
    have hfb : (RecAlgebra.decoded sh d.w d.comb d.cons).fb = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).fb := by
      show RecAlgebra.Lin.eval (RecAlgebra.valOf sh d.w d.comb d.cons) (RecAlgebra.Lin.var .fb) =
        RecAlgebra.Lin.eval (RecAlgebra.valOf sh d'.w d'.comb d'.cons) (RecAlgebra.Lin.var .fb)
      rw [RecAlgebra.Lin.eval_var, RecAlgebra.Lin.eval_var]
      exact h _ (hk ▸ List.mem_cons_of_mem _ List.mem_cons_self)
    show (⟨_, (fOf (RecAlgebra.decoded sh d.w d.comb d.cons).fa,
        fOf (RecAlgebra.decoded sh d.w d.comb d.cons).fb)⟩ : FlockSoundness.Game.Msg) =
      ⟨_, (fOf (RecAlgebra.decoded sh d'.w d'.comb d'.cons).fa, fOf (RecAlgebra.decoded sh d'.w d'.comb d'.cons).fb)⟩
    rw [hfa, hfb]

theorem zcMsgs_length (mz : ℕ) (p : Flock.ZerocheckProof) : (zcMsgs mz p).length = mz - 6 + 2 := by
  rw [zcMsgs, list3_length]

/-- **A rep's first `m_z − 4` messages are the zerocheck's.** -/
theorem unitMsgs_zc (nX : ℕ) (sh : Flock.Shape) (d : RepData) (v : UnitVals) {m : ℕ} (hm : m < v.st.m - 6 + 2) :
    (unitMsgs nX sh d v)[m]? = (zcMsgs v.st.m v.p.zerocheck)[m]? := by
  have hl : m < (zcMsgs v.st.m v.p.zerocheck).length := by rw [zcMsgs_length]; exact hm
  rw [unitMsgs, holoRepMsgs, List.append_assoc, List.append_assoc, List.getElem?_append_left hl]

/-- **The zerocheck slice at `ValsSrc`'s values**: a rep's message `m < m_z − 4` reads only `zcKeys m`, none a
consistency sum, at `rep₀`'s consistency sums or any others. -/
theorem unitMsgs_zc_read (nX : ℕ) (sh : Flock.Shape) {b b' : ValsSrc} {d d' : RepData}
    (hstM : b.st.m = b'.st.m) {m : ℕ}
    (hm : m < b.st.m - 6 + 2)
    (h : ∀ x ∈ zcKeys b.st.m m, (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x) :
    (unitMsgs nX sh d (b.vals sh d))[m]? = (unitMsgs nX sh d' (b'.vals sh d'))[m]? := by
  have hm' : m < b'.st.m - 6 + 2 := by rwa [← hstM]
  rw [unitMsgs_zc nX sh d (b.vals sh d) hm, unitMsgs_zc nX sh d' (b'.vals sh d') hm']
  show (zcMsgs b.st.m (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).zerocheck)[m]? =
    (zcMsgs b'.st.m (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').zerocheck)[m]?
  rw [← hstM]
  exact zcMsgs_read b.st.m sh m fun x hx =>
    (valOf_cons_irrel sh d.w d.comb d.cons (fun _ _ => 0) (zcKeys_noCons hx)).trans
      ((h x hx (zcKeys_noCons hx)).trans (valOf_cons_irrel sh d'.w d'.comb (fun _ _ => 0) d'.cons (zcKeys_noCons hx)))

/-- The keys of the lincheck's message `m` at `k_log`, with `h` after `z_partial`: round `m`'s `(q1, qinf)`, then
`z_partial`'s, then `h`'s. -/
def lcKeys (kLog m : ℕ) : List Flock.RecAlgebra.Key :=
  if m < kLog - 6 then [.q1 m, .qinf m]
  else if m = kLog - 6 then (List.range 64).map .zp
  else (List.range 64).map .h

theorem lcKeys_noCons {kLog m : ℕ} {x : Flock.RecAlgebra.Key} (hx : x ∈ lcKeys kLog m) : ∀ ℓ l, x ≠ .cons ℓ l := by
  rintro ℓ l rfl
  by_cases h1 : m < kLog - 6
  · simp [lcKeys, h1] at hx
  · by_cases h2 : m = kLog - 6
    · rw [lcKeys, ite_eq_right h1, ite_eq_left h2] at hx
      simp at hx
    · simp [lcKeys, h1, h2] at hx

/-- **The lincheck's message `m` of G's proof file, with `h` after `z_partial`, reads only `lcKeys m`** at a holo
shape. -/
theorem holoLcMsgs_read (kLog : ℕ) {sh : Flock.Shape} (hh : sh.holo = true) {params params' : Flock.PcsParams}
    {cap cap' : Array ByteArray} {lig lig' : Flock.LigeritoProof} {d d' : RepData} (m : ℕ)
    (h : ∀ x ∈ lcKeys kLog m,
      Flock.RecAlgebra.valOf sh d.w d.comb d.cons x = Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons x) :
    (holoLcMsgs kLog (FlockVBridge.Family.proofOf sh params cap lig d).lincheck (RecAlgebra.hOfW sh d.w))[m]? =
      (holoLcMsgs kLog (FlockVBridge.Family.proofOf sh params' cap' lig' d').lincheck
        (RecAlgebra.hOfW sh d'.w))[m]? := by
  unfold holoLcMsgs lcMsgs
  apply list2_get
  · intro hm
    have hz : (RecAlgebra.decoded sh d.w d.comb d.cons).lc[m]! = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).lc[m]! :=
      pairs_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var (sh.kLog - 6) m RecAlgebra.Key.q1
        RecAlgebra.Key.qinf fun _ => by
          have hk : lcKeys kLog m = [.q1 m, .qinf m] := ite_eq_left hm
          exact ⟨h _ (hk ▸ List.mem_cons_self), h _ (hk ▸ List.mem_cons_of_mem _ List.mem_cons_self)⟩
    show (⟨_, pairOf (RecAlgebra.decoded sh d.w d.comb d.cons).lc[m]!⟩ : FlockSoundness.Game.Msg) =
      ⟨_, pairOf (RecAlgebra.decoded sh d'.w d'.comb d'.cons).lc[m]!⟩
    rw [hz]
  · intro hm
    have hk : lcKeys kLog m = (List.range 64).map .zp := (ite_eq_right (by omega)).trans (ite_eq_left hm)
    have hzp : (RecAlgebra.decoded sh d.w d.comb d.cons).zp = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).zp :=
      vars_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var 64 RecAlgebra.Key.zp fun i hi =>
        h _ (by rw [hk]; exact List.mem_map_of_mem (List.mem_range.2 hi))
    show (⟨_, vecOf 64 (RecAlgebra.decoded sh d.w d.comb d.cons).zp⟩ : FlockSoundness.Game.Msg) =
      ⟨_, vecOf 64 (RecAlgebra.decoded sh d'.w d'.comb d'.cons).zp⟩
    rw [hzp]
  · intro hm
    have hk : lcKeys kLog m = (List.range 64).map .h := (ite_eq_right (by omega)).trans (ite_eq_right (by omega))
    have hv : vecOf 64 (RecAlgebra.hOfW sh d.w) = vecOf 64 (RecAlgebra.hOfW sh d'.w) := funext fun i => by
      have hx : Flock.RecAlgebra.valOf sh d.w d.comb d.cons (.h i.val) =
          Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons (.h i.val) :=
        h _ (by rw [hk]; exact List.mem_map_of_mem (List.mem_range.2 i.isLt))
      show fOf (RecAlgebra.hOfW sh d.w)[i.val]! = fOf (RecAlgebra.hOfW sh d'.w)[i.val]!
      rw [hOfW_valOf hh d.w d.comb d.cons i.isLt, hOfW_valOf hh d'.w d'.comb d'.cons i.isLt, hx]
    rw [hv]

theorem holoLcMsgs_length (kLog : ℕ) (p : Flock.LincheckProof) (hpart : Array F128) :
    (holoLcMsgs kLog p hpart).length = kLog - 6 + 2 := by
  rw [holoLcMsgs, lcMsgs, list2_length]

/-- **A rep's next `k_log − 4` messages are the lincheck's with `h`.** -/
theorem unitMsgs_lc (nX : ℕ) (sh : Flock.Shape) (d : RepData) (v : UnitVals) {i j : ℕ}
    (hi : i = v.st.m - 6 + 2 + j) (hj : j < v.st.kLog - 6 + 2) :
    (unitMsgs nX sh d v)[i]? = (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w))[j]? := by
  have hz : (zcMsgs v.st.m v.p.zerocheck).length ≤ i := by rw [zcMsgs_length]; omega
  have hl : i - (zcMsgs v.st.m v.p.zerocheck).length <
      (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length := by
    rw [zcMsgs_length, holoLcMsgs_length]; omega
  have he : i - (zcMsgs v.st.m v.p.zerocheck).length = j := by rw [zcMsgs_length]; omega
  rw [unitMsgs, holoRepMsgs, List.append_assoc, List.append_assoc, List.getElem?_append_right hz,
    List.getElem?_append_left hl, he]

/-- **The lincheck slice at `ValsSrc`'s values**: at a holo shape a rep's message `m_z − 4 + j`, `j < k_log − 4`,
reads only `lcKeys j`, none a consistency sum. -/
theorem unitMsgs_lc_read (nX : ℕ) {sh : Flock.Shape} (hh : sh.holo = true) {b b' : ValsSrc} {d d' : RepData}
    (hm : b.st.m = b'.st.m) (hk : b.st.kLog = b'.st.kLog) {j : ℕ} (hj : j < b.st.kLog - 6 + 2)
    (h : ∀ x ∈ lcKeys b.st.kLog j, (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x) :
    (unitMsgs nX sh d (b.vals sh d))[b.st.m - 6 + 2 + j]? =
      (unitMsgs nX sh d' (b'.vals sh d'))[b.st.m - 6 + 2 + j]? := by
  have hj' : j < b'.st.kLog - 6 + 2 := by rwa [← hk]
  rw [unitMsgs_lc nX sh d (b.vals sh d) (i := b.st.m - 6 + 2 + j) rfl hj,
    unitMsgs_lc nX sh d' (b'.vals sh d') (i := b.st.m - 6 + 2 + j)
      (congrArg (fun m => m - 6 + 2 + j) hm) hj']
  show (holoLcMsgs b.st.kLog (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).lincheck
      (RecAlgebra.hOfW sh d.w))[j]? =
    (holoLcMsgs b'.st.kLog (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').lincheck
      (RecAlgebra.hOfW sh d'.w))[j]?
  rw [← hk]
  exact holoLcMsgs_read b.st.kLog hh j fun x hx =>
    (valOf_cons_irrel sh d.w d.comb d.cons (fun _ _ => 0) (lcKeys_noCons hx)).trans
      ((h x hx (lcKeys_noCons hx)).trans (valOf_cons_irrel sh d'.w d'.comb (fun _ _ => 0) d'.cons (lcKeys_noCons hx)))

/-- The keys of ring switch `k`'s message: its 128 words. -/
def rsKeys (k : ℕ) : List Flock.RecAlgebra.Key := (List.range 128).map (.rs k)

theorem rsKeys_noCons {k : ℕ} {x : Flock.RecAlgebra.Key} (hx : x ∈ rsKeys k) : ∀ ℓ l, x ≠ .cons ℓ l := by
  rintro ℓ l rfl
  simp [rsKeys] at hx

/-- **Ring switch `k`'s message of G's proof file reads only `rsKeys k`.** -/
theorem opMsgs_read (n : ℕ) (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} (k : ℕ)
    (h : ∀ x ∈ rsKeys k,
      Flock.RecAlgebra.valOf sh d.w d.comb d.cons x = Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons x) :
    (opMsgs n (FlockVBridge.Family.proofOf sh params cap lig d).ringSwitches)[k]? =
      (opMsgs n (FlockVBridge.Family.proofOf sh params' cap' lig' d').ringSwitches)[k]? := by
  have hrs : (RecAlgebra.decoded sh d.w d.comb d.cons).rs[k]! = (RecAlgebra.decoded sh d'.w d'.comb d'.cons).rs[k]! :=
    rows_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var sh.claims 128 k RecAlgebra.Key.rs
      fun v hv => h _ (List.mem_map_of_mem (List.mem_range.2 hv))
  unfold opMsgs
  exact map_range_get fun _ => by
    show (⟨_, vecOf 128 (RecAlgebra.decoded sh d.w d.comb d.cons).rs[k]!⟩ : FlockSoundness.Game.Msg) =
      ⟨_, vecOf 128 (RecAlgebra.decoded sh d'.w d'.comb d'.cons).rs[k]!⟩
    rw [hrs]

theorem opMsgs_length (n : ℕ) (rs : Array (Array F128)) : (opMsgs n rs).length = n := by
  rw [opMsgs, List.length_map, List.length_range]

/-- **A rep's next `2 + n_X` messages are its ring switches'.** -/
theorem unitMsgs_rs (nX : ℕ) (sh : Flock.Shape) (d : RepData) (v : UnitVals) {i k : ℕ}
    (hi : i = v.st.m - 6 + 2 + (v.st.kLog - 6 + 2) + k) (hk : k < 2 + nX) :
    (unitMsgs nX sh d v)[i]? = (opMsgs (2 + nX) v.p.ringSwitches)[k]? := by
  have hz : (zcMsgs v.st.m v.p.zerocheck).length ≤ i := by rw [zcMsgs_length]; omega
  have hz' : (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length ≤
      i - (zcMsgs v.st.m v.p.zerocheck).length := by
    rw [zcMsgs_length, holoLcMsgs_length]; omega
  have hl : i - (zcMsgs v.st.m v.p.zerocheck).length -
      (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length <
        (opMsgs (2 + nX) v.p.ringSwitches).length := by
    rw [zcMsgs_length, holoLcMsgs_length, opMsgs_length]; omega
  have he : i - (zcMsgs v.st.m v.p.zerocheck).length -
      (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length = k := by
    rw [zcMsgs_length, holoLcMsgs_length]; omega
  rw [unitMsgs, holoRepMsgs, List.append_assoc, List.append_assoc, List.getElem?_append_right hz,
    List.getElem?_append_right hz', List.getElem?_append_left hl, he]

/-- **The ring-switch slice at `ValsSrc`'s values**: a rep's message `m_z − 4 + k_log − 4 + k`, `k < 2 + n_X`, reads
only `rsKeys k`, none a consistency sum. -/
theorem unitMsgs_rs_read (nX : ℕ) (sh : Flock.Shape) {b b' : ValsSrc} {d d' : RepData}
    (hm : b.st.m = b'.st.m) (hkLog : b.st.kLog = b'.st.kLog) {k : ℕ}
    (hk : k < 2 + nX)
    (h : ∀ x ∈ rsKeys k, (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x) :
    (unitMsgs nX sh d (b.vals sh d))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + k]? =
      (unitMsgs nX sh d' (b'.vals sh d'))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + k]? := by
  rw [unitMsgs_rs nX sh d (b.vals sh d) (i := b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + k) rfl hk,
    unitMsgs_rs nX sh d' (b'.vals sh d') (i := b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + k)
      (congrArg₂ (fun m kLog => m - 6 + 2 + (kLog - 6 + 2) + k) hm hkLog) hk]
  show (opMsgs (2 + nX) (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).ringSwitches)[k]? =
    (opMsgs (2 + nX) (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').ringSwitches)[k]?
  exact opMsgs_read (2 + nX) sh k fun x hx =>
    (valOf_cons_irrel sh d.w d.comb d.cons (fun _ _ => 0) (rsKeys_noCons hx)).trans
      ((h x hx (rsKeys_noCons hx)).trans (valOf_cons_irrel sh d'.w d'.comb (fun _ _ => 0) d'.cons (rsKeys_noCons hx)))

/-- **An evaluated variable at its index reads only its key.** -/
theorem var_get_read {K L F : Type} [Inhabited F] (var : K → L) (eval : (K → F) → L → F)
    (heval : ∀ v k, eval v (var k) = v k) (n i : ℕ) (key : ℕ → K) {v v' : K → F}
    (h : i < n → v (key i) = v' (key i)) :
    (((Array.range n).map (var ∘ key)).map (eval v))[i]! =
      (((Array.range n).map (var ∘ key)).map (eval v'))[i]! := by
  by_cases hi : i < n
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_some,
      Function.comp_apply, heval, h hi]
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_none]

/-- **An evaluated pair of pairs at its index reads only its four keys.** -/
theorem kvs_get_read {K L F G : Type} [Inhabited G] (var : K → L) (eval : (K → F) → L → F)
    (heval : ∀ v k, eval v (var k) = v k) (g : RecAlgebra.KV F × RecAlgebra.KV F → G) (n i : ℕ)
    (key : ℕ → ℕ → ℕ → K) {v v' : K → F}
    (h : i < n → ∀ a < 2, ∀ b < 2, v (key i a b) = v' (key i a b)) :
    ((((Array.range n).map fun j => ((var (key j 0 0), var (key j 0 1)), (var (key j 1 0), var (key j 1 1)))).map
        fun mm => (RecAlgebra.KV.map (eval v) mm.1, RecAlgebra.KV.map (eval v) mm.2)).map g)[i]! =
      ((((Array.range n).map fun j => ((var (key j 0 0), var (key j 0 1)), (var (key j 1 0), var (key j 1 1)))).map
        fun mm => (RecAlgebra.KV.map (eval v') mm.1, RecAlgebra.KV.map (eval v') mm.2)).map g)[i]! := by
  by_cases hi : i < n
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_some,
      RecAlgebra.KV.map, heval, h hi 0 (by decide) 0 (by decide), h hi 0 (by decide) 1 (by decide),
      h hi 1 (by decide) 0 (by decide), h hi 1 (by decide) 1 (by decide)]
  · simp only [getElem!_def, Array.getElem?_map, Array.getElem?_range, hi, ↓reduceIte, Option.map_none]

/-- **Out-of-domain value `i` of G's Ligerito proof reads only `.ood i`.** -/
theorem ligOod_read (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} (i : ℕ)
    (h : Flock.RecAlgebra.valOf sh d.w d.comb d.cons (.ood i) =
      Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons (.ood i)) :
    (FlockVBridge.Family.proofOf sh params cap lig d).ligerito.oodValues[i]! =
      (FlockVBridge.Family.proofOf sh params' cap' lig' d').ligerito.oodValues[i]! :=
  var_get_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var _ i RecAlgebra.Key.ood fun _ => h

/-- **Sumcheck message `i` of G's Ligerito proof reads only its four `.msg i` keys.** -/
theorem ligMsg_read (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} (i : ℕ)
    (h : ∀ a < 2, ∀ b < 2,
      Flock.RecAlgebra.valOf sh d.w d.comb d.cons (.msg i a b) =
        Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons (.msg i a b)) :
    (FlockVBridge.Family.proofOf sh params cap lig d).ligerito.messages[i]! =
      (FlockVBridge.Family.proofOf sh params' cap' lig' d').ligerito.messages[i]! :=
  kvs_get_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var
    (fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2)) _ i RecAlgebra.Key.msg fun _ => h

/-- **G's final `yr` reads only the `.yr j` keys.** -/
theorem yrOf_read (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} (ext : ℕ)
    (h : ∀ j < 2 ^ (ext + 1),
      Flock.RecAlgebra.valOf sh d.w d.comb d.cons (.yr j) = Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons (.yr j)) :
    FlockSoundness.Refine.yrOf (FlockVBridge.Family.proofOf sh params cap lig d).ligerito ext =
      FlockSoundness.Refine.yrOf (FlockVBridge.Family.proofOf sh params' cap' lig' d').ligerito ext :=
  funext fun i => by
    have e : (RecAlgebra.decoded sh d.w d.comb d.cons).yr[i.val]! =
        (RecAlgebra.decoded sh d'.w d'.comb d'.cons).yr[i.val]! :=
      var_get_read RecAlgebra.Lin.var RecAlgebra.Lin.eval RecAlgebra.Lin.eval_var _ i.val RecAlgebra.Key.yr
        fun _ => h i.val i.isLt
    show fOf (RecAlgebra.decoded sh d.w d.comb d.cons).yr[i.val]! =
      fOf (RecAlgebra.decoded sh d'.w d'.comb d'.cons).yr[i.val]!
    rw [e]

open FlockSoundness.Refine (ligMsgs recMsgs kMsg fMsg yrOf capDig)

/-- **`L` and `L'` agree at each index whose keys read alike** (`R`), the three lists of one length. -/
def PwRead {κ α : Type} (R : κ → Prop) (ks : List κ) (L L' : List α) : Prop :=
  L.length = ks.length ∧ L'.length = ks.length ∧ ∀ (k : ℕ) c, ks[k]? = some c → R c → L[k]? = L'[k]?

theorem pwRead_nil {κ α : Type} (R : κ → Prop) : PwRead R ([] : List κ) ([] : List α) [] :=
  ⟨rfl, rfl, fun _ _ h _ => by simp at h⟩

theorem pwRead_cons {κ α : Type} {R : κ → Prop} {c : κ} {ks : List κ} {m m' : α} {L L' : List α}
    (h : R c → m = m') (t : PwRead R ks L L') : PwRead R (c :: ks) (m :: L) (m' :: L') := by
  refine ⟨by simp [t.1], by simp [t.2.1], fun k c' hk hc => ?_⟩
  cases k with
  | zero =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hk ⊢
    subst hk
    rw [h hc]
  | succ k =>
    simp only [List.getElem?_cons_succ] at hk ⊢
    exact t.2.2 k c' hk hc

theorem pwRead_append {κ α : Type} {R : κ → Prop} {k₁ k₂ : List κ} {L₁ L₁' L₂ L₂' : List α}
    (h₁ : PwRead R k₁ L₁ L₁') (h₂ : PwRead R k₂ L₂ L₂') :
    PwRead R (k₁ ++ k₂) (L₁ ++ L₂) (L₁' ++ L₂') := by
  refine ⟨by simp [h₁.1, h₂.1], by simp [h₁.2.1, h₂.2.1], fun k c hk hc => ?_⟩
  by_cases hl : k < k₁.length
  · rw [List.getElem?_append_left hl] at hk
    rw [List.getElem?_append_left (by rw [h₁.1]; exact hl), List.getElem?_append_left (by rw [h₁.2.1]; exact hl)]
    exact h₁.2.2 k c hk hc
  · have hl' : k₁.length ≤ k := Nat.le_of_not_lt hl
    rw [List.getElem?_append_right hl'] at hk
    rw [List.getElem?_append_right (by rw [h₁.1]; exact hl'),
      List.getElem?_append_right (by rw [h₁.2.1]; exact hl'), h₁.1, h₁.2.1]
    exact h₂.2.2 _ c hk hc

theorem pwRead_map {κ α : Type} {R : κ → Prop} (key : ℕ → κ) (f f' : ℕ → α) (l : List ℕ)
    (h : ∀ j ∈ l, R (key j) → f j = f' j) : PwRead R (l.map key) (l.map f) (l.map f') := by
  refine ⟨by simp, by simp, fun k c hk hc => ?_⟩
  rw [List.getElem?_map] at hk
  rw [List.getElem?_map, List.getElem?_map]
  cases hj : l[k]? with
  | none => rfl
  | some j =>
    rw [hj] at hk
    simp only [Option.map_some, Option.some.injEq] at hk ⊢
    subst hk
    exact h j (List.mem_of_getElem? hj) hc

/-- Out-of-domain value `j`'s key. -/
def oodK (j : ℕ) : List Flock.RecAlgebra.Key := [.ood j]

/-- Sumcheck message `j`'s keys: its two pairs of words. -/
def msgK (j : ℕ) : List Flock.RecAlgebra.Key := [.msg j 0 0, .msg j 0 1, .msg j 1 0, .msg j 1 1]

/-- The final `yr`'s keys at extension `ext`. -/
def yrK (ext : ℕ) : List Flock.RecAlgebra.Key := (List.range (2 ^ (ext + 1))).map Flock.RecAlgebra.Key.yr

theorem msgK_mem {j a b : ℕ} (ha : a < 2) (hb : b < 2) : Flock.RecAlgebra.Key.msg j a b ∈ msgK j := by
  match a, b, ha, hb with
  | 0, 0, _, _ => simp [msgK]
  | 0, 1, _, _ => simp [msgK]
  | 1, 0, _, _ => simp [msgK]
  | 1, 1, _, _ => simp [msgK]
  | a + 2, _, ha, _ => exact absurd ha (by omega)
  | _, b + 2, _, hb => exact absurd hb (by omega)

/-- The keys of `recMsgs`' messages, in its order (a cap reads no key). -/
def recKeys (ext : ℕ) : ℕ → ℕ → ℕ → List (List Flock.RecAlgebra.Key)
  | 0, _, _ => []
  | 1, msg, _ => (List.range' msg 4).map msgK ++ [yrK ext]
  | left + 2, msg, ood => (List.range' msg 4).map msgK ++ [] ::
      [oodK ood, msgK (msg + 4), oodK (ood + 1), msgK (msg + 5)] ++
      [msgK (msg + 6)] ++ recKeys ext (left + 1) (msg + 7) (ood + 2)

/-- The keys of `ligMsgs`' messages, in its order. -/
def ligKeys (sch : Flock.Schedule) : List (List Flock.RecAlgebra.Key) :=
  oodK 0 :: msgK 0 :: (List.range' 1 sch.k0).map msgK ++ [] ::
    [oodK 1, msgK (sch.k0 + 1), oodK 2, msgK (sch.k0 + 2)] ++
    [msgK (sch.k0 + 3)] ++ recKeys sch.ext (sch.levels.size - 1) (sch.k0 + 4) 3

/-- **The recursive caps read alike**: no key covers them, so the Ligerito slice takes them as read. -/
def RecCapsRead (lig lig' : Flock.LigeritoProof) : Prop := lig.recursiveCaps = lig'.recursiveCaps

/-- **The level loop's messages read only `recKeys`**, from each message's reading only its keys. -/
theorem recMsgs_read {D : Type} {R : List Flock.RecAlgebra.Key → Prop} {p p' : Flock.LigeritoProof}
    (capD : Array ByteArray → FlockSoundness.Model.Cap D) (ext : ℕ)
    (hm : ∀ j, R (msgK j) → p.messages[j]! = p'.messages[j]!)
    (ho : ∀ j, R (oodK j) → p.oodValues[j]! = p'.oodValues[j]!)
    (hy : R (yrK ext) → yrOf p ext = yrOf p' ext) (hc : RecCapsRead p p') :
    ∀ left i msg ood, PwRead R (recKeys ext left msg ood) (recMsgs p capD ext left i msg ood)
      (recMsgs p' capD ext left i msg ood)
  | 0, _, _, _ => pwRead_nil R
  | 1, _, msg, _ => by
    rw [recKeys, recMsgs, recMsgs]
    exact pwRead_append (pwRead_map msgK _ _ _ fun j _ h => congrArg kMsg (hm j h))
      (pwRead_cons (fun h => by rw [hy h]) (pwRead_nil R))
  | left + 2, i, msg, ood => by
    rw [recKeys, recMsgs, recMsgs]
    exact pwRead_append (pwRead_append (pwRead_append (pwRead_map msgK _ _ _ fun j _ h => congrArg kMsg (hm j h))
      (pwRead_cons (fun _ => by rw [show p.recursiveCaps = p'.recursiveCaps from hc])
      (pwRead_cons (fun h => congrArg fMsg (ho ood h))
        (pwRead_cons (fun h => congrArg kMsg (hm (msg + 4) h)) (pwRead_cons (fun h => congrArg fMsg (ho (ood + 1) h))
          (pwRead_cons (fun h => congrArg kMsg (hm (msg + 5) h)) (pwRead_nil R)))))))
      (pwRead_cons (fun h => congrArg kMsg (hm (msg + 6) h)) (pwRead_nil R)))
      (recMsgs_read capD ext hm ho hy hc (left + 1) (i + 1) (msg + 7) (ood + 2))

/-- **A Ligerito run's messages read only `ligKeys`.** -/
theorem ligMsgs_read {D : Type} {R : List Flock.RecAlgebra.Key → Prop} {p p' : Flock.LigeritoProof}
    (capD : Array ByteArray → FlockSoundness.Model.Cap D) (sch : Flock.Schedule)
    (hm : ∀ j, R (msgK j) → p.messages[j]! = p'.messages[j]!)
    (ho : ∀ j, R (oodK j) → p.oodValues[j]! = p'.oodValues[j]!)
    (hy : R (yrK sch.ext) → yrOf p sch.ext = yrOf p' sch.ext) (hc : RecCapsRead p p') :
    PwRead R (ligKeys sch) (ligMsgs p capD sch) (ligMsgs p' capD sch) := by
  unfold ligKeys ligMsgs
  exact pwRead_append (pwRead_append (pwRead_append
    (pwRead_cons (fun h => congrArg fMsg (ho 0 h)) (pwRead_cons (fun h => congrArg kMsg (hm 0 h))
      (pwRead_map msgK _ _ _ fun j _ h => congrArg kMsg (hm j h))))
    (pwRead_cons (fun _ => by rw [show p.recursiveCaps = p'.recursiveCaps from hc])
      (pwRead_cons (fun h => congrArg fMsg (ho 1 h))
      (pwRead_cons (fun h => congrArg kMsg (hm (sch.k0 + 1) h)) (pwRead_cons (fun h => congrArg fMsg (ho 2 h))
        (pwRead_cons (fun h => congrArg kMsg (hm (sch.k0 + 2) h)) (pwRead_nil R)))))))
    (pwRead_cons (fun h => congrArg kMsg (hm (sch.k0 + 3) h)) (pwRead_nil R)))
    (recMsgs_read capD sch.ext hm ho hy hc (sch.levels.size - 1) 0 (sch.k0 + 4) 3)

/-- Keys `ks` read alike at `rep₀`'s values, consistency sums zero, as `MsgsOfKeys` reads them. -/
def KeysRead (sh : Flock.Shape) (d d' : RepData) (ks : List Flock.RecAlgebra.Key) : Prop :=
  ∀ x ∈ ks, (∀ ℓ l, x ≠ .cons ℓ l) →
    Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x = Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x

theorem keysRead_at {sh : Flock.Shape} {d d' : RepData} {ks : List Flock.RecAlgebra.Key} (h : KeysRead sh d d' ks)
    {x : Flock.RecAlgebra.Key} (hx : x ∈ ks) (hnc : ∀ ℓ l, x ≠ .cons ℓ l) :
    Flock.RecAlgebra.valOf sh d.w d.comb d.cons x = Flock.RecAlgebra.valOf sh d'.w d'.comb d'.cons x :=
  (valOf_cons_irrel sh d.w d.comb d.cons (fun _ _ => 0) hnc).trans
    ((h x hx hnc).trans (valOf_cons_irrel sh d'.w d'.comb (fun _ _ => 0) d'.cons hnc))

/-- **G's Ligerito messages read only `ligKeys`**, its recursive caps read alike. -/
theorem ligMsgs_proofOf_read (sh : Flock.Shape) {params params' : Flock.PcsParams} {cap cap' : Array ByteArray}
    {lig lig' : Flock.LigeritoProof} {d d' : RepData} {D : Type} (capD : Array ByteArray → FlockSoundness.Model.Cap D)
    (sch : Flock.Schedule) (hc : RecCapsRead lig lig') :
    PwRead (KeysRead sh d d') (ligKeys sch)
      (ligMsgs (FlockVBridge.Family.proofOf sh params cap lig d).ligerito capD sch)
      (ligMsgs (FlockVBridge.Family.proofOf sh params' cap' lig' d').ligerito capD sch) :=
  ligMsgs_read capD sch
    (fun j h => ligMsg_read sh j fun _ ha _ hb => keysRead_at h (msgK_mem ha hb) (by intro _ _ e; cases e))
    (fun j h => ligOod_read sh j (keysRead_at h (List.mem_singleton_self _) (by intro _ _ e; cases e)))
    (fun h => yrOf_read sh sch.ext fun j hj =>
      keysRead_at h (List.mem_map_of_mem (List.mem_range.2 hj)) (by intro _ _ e; cases e))
    hc

/-- **A rep's last messages are its Ligerito run's.** -/
theorem unitMsgs_lig (nX : ℕ) (sh : Flock.Shape) (d : RepData) (v : UnitVals) {i k : ℕ}
    (hi : i = v.st.m - 6 + 2 + (v.st.kLog - 6 + 2) + (2 + nX) + k) :
    (unitMsgs nX sh d v)[i]? = (ligMsgs v.p.ligerito capDig v.sch)[k]? := by
  have hz : (zcMsgs v.st.m v.p.zerocheck).length ≤ i := by rw [zcMsgs_length]; omega
  have hz' : (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length ≤
      i - (zcMsgs v.st.m v.p.zerocheck).length := by
    rw [zcMsgs_length, holoLcMsgs_length]; omega
  have hr : (opMsgs (2 + nX) v.p.ringSwitches).length ≤ i - (zcMsgs v.st.m v.p.zerocheck).length -
      (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length := by
    rw [zcMsgs_length, holoLcMsgs_length, opMsgs_length]; omega
  have he : i - (zcMsgs v.st.m v.p.zerocheck).length -
      (holoLcMsgs v.st.kLog v.p.lincheck (RecAlgebra.hOfW sh d.w)).length -
        (opMsgs (2 + nX) v.p.ringSwitches).length = k := by
    rw [zcMsgs_length, holoLcMsgs_length, opMsgs_length]; omega
  rw [unitMsgs, holoRepMsgs, List.append_assoc, List.append_assoc, List.getElem?_append_right hz,
    List.getElem?_append_right hz', List.getElem?_append_right hr, he]

/-- **The Ligerito slice at `ValsSrc`'s values**: a rep's message `m_z − 4 + k_log − 4 + 2 + n_X + k` reads only
`ligKeys`' entry `k`, where the two sources' schedules agree and their recursive caps read alike. -/
theorem unitMsgs_lig_read (nX : ℕ) (sh : Flock.Shape) {b b' : ValsSrc} {d d' : RepData}
    (hm : b.st.m = b'.st.m) (hkLog : b.st.kLog = b'.st.kLog)
    (hsch : b.sch = b'.sch) (hc : RecCapsRead b.lig b'.lig) {k : ℕ} {ks : List Flock.RecAlgebra.Key}
    (hk : (ligKeys b.sch)[k]? = some ks) (h : KeysRead sh d d' ks) :
    (unitMsgs nX sh d (b.vals sh d))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k]? =
      (unitMsgs nX sh d' (b'.vals sh d'))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k]? := by
  rw [unitMsgs_lig nX sh d (b.vals sh d) (i := b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k) rfl,
    unitMsgs_lig nX sh d' (b'.vals sh d') (i := b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k)
      (congrArg₂ (fun m kLog => m - 6 + 2 + (kLog - 6 + 2) + (2 + nX) + k) hm hkLog)]
  show (ligMsgs (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).ligerito capDig b.sch)[k]? =
    (ligMsgs (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').ligerito capDig b'.sch)[k]?
  rw [← hsch]
  exact (ligMsgs_proofOf_read sh capDig b.sch hc).2.2 k ks hk h

/-- **The Ligerito slice reads its key list at every index**, with the empty default past `ligKeys`. -/
theorem unitMsgs_lig_getD_read (nX : ℕ) (sh : Flock.Shape) {b b' : ValsSrc} {d d' : RepData}
    (hm : b.st.m = b'.st.m) (hkLog : b.st.kLog = b'.st.kLog)
    (hsch : b.sch = b'.sch) (hc : RecCapsRead b.lig b'.lig) (k : ℕ)
    (h : KeysRead sh d d' ((ligKeys b.sch).getD k [])) :
    (unitMsgs nX sh d (b.vals sh d))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k]? =
      (unitMsgs nX sh d' (b'.vals sh d'))[b.st.m - 6 + 2 + (b.st.kLog - 6 + 2) + (2 + nX) + k]? := by
  cases hk : (ligKeys b.sch)[k]? with
  | some ks =>
      exact unitMsgs_lig_read nX sh hm hkLog hsch hc hk
        (by simpa [List.getD_eq_getElem?_getD, hk] using h)
  | none =>
      rw [unitMsgs_lig nX sh d (b.vals sh d) rfl,
        unitMsgs_lig nX sh d' (b'.vals sh d')
          (congrArg₂ (fun m kLog => m - 6 + 2 + (kLog - 6 + 2) + (2 + nX) + k) hm hkLog)]
      have hp := ligMsgs_proofOf_read sh capDig b.sch hc
      have hge := List.getElem?_eq_none_iff.1 hk
      rw [List.getElem?_eq_none (hp.1 ▸ hge), List.getElem?_eq_none (hp.2.1 ▸ hge)]

end slices

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (maskOf skOf IoColAt mem_skOf_inl)
open FlockVBridge.AccChain (levelLanes)
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}

local notation "ℐ" => holoInnerAt T CT st hp

/-! ## No `.cons` key: `AlphaRead` and `RowColsRead` vacuously -/

/-- **`AlphaRead` where no stage's keys read a consistency sum.** -/
theorem alphaRead_of_noCons (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hK : ∀ s ρ ℓ l, ¬ K s ρ (.cons ℓ l)) : AlphaRead (st := st) (hp := hp) zs dg N k Rw own sel M K := by
  intro s ρ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ℓ l hk
  exact absurd hk (hK s ρ ℓ l)

/-- **The forms a stage's sent keys read, at the decoder**: `formsRead_maps` at `sentK`, `AlphaRead` and `RowColsRead`
vacuous there. -/
theorem formsRead_sent (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) {σ : List Exec.Kind} {c₀ : ℕ}
    {keys : ℕ → List Flock.RecAlgebra.Key}
    (hu : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ (ρ : Fin 2), ∀ ℓ < sh.sched.levels.size,
      ∀ j < M.q ℓ, ∃ a cf A, FlockVBridge.Family.OpensAtL (Vs (M.sessO ℓ))
        (layers zs dg N k Rw R ω t R₂ τ (M.sessO ℓ)) (M.unO ℓ ρ j) (FlockVBridge.Family.cpO ℓ ρ j) (levelLanes sh ℓ)
        ((M.rows (layers zs dg N k Rw R ω t R₂ τ) ρ ℓ)[j]!) a cf A)
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M (sentK σ c₀ keys)) (s : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun R ω t Lb R' ω' t' Lb' =>
      ∀ f : Flock.RecAlgebra.Lin, LinReads f (sentK σ c₀ keys s ρ) →
        Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep R ω t Lb ρ).w (M.rep R ω t Lb ρ).comb (M.rep R ω t Lb ρ).cons) f =
          Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep R' ω' t' Lb' ρ).w (M.rep R' ω' t' Lb' ρ).comb
              (M.rep R' ω' t' Lb' ρ).cons) f :=
  formsRead_maps M hu hwc (M.rowColsRead_of_noCons sentK_noCons) (alphaRead_of_noCons M sentK_noCons) s ρ

/-! ## `WCombRead` from the reads file -/

/-- **Hypothesis on the decoder: `rep₀` reads each stage key off input columns of one copy**: for each key `x` of
`K s ρ` but the `.cons` keys, `rep₀` gives `x` the same value at any two layers agreeing on a set `C` of copy 0's
columns of a unit `u` of a session `j`, each an input column a row reads (`IoColAt`), keyed by bit `kq` of a row port
`q` that stage `s` reads (`SelPort`). -/
def WColsAt (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) x, K s ρ x → (∀ ℓ l, x ≠ .cons ℓ l) →
    ∃ (j : Fin S) (u : Fin (Vs j).nv) (hj : 0 < (Vs j).c.upv) (C : ℕ → Prop),
      (∀ c, C c → c < (Rows.ofNet (Vs j).hU).nIn ∧ IoColAt (Vs j) u c ∧
        ∃ (q : Fin (Vs j).c.ports.size) (kq : ℕ), kq < Layout.portRowW (Vs j).c q ∧
          Aliased.srcKey (Vs j).c (0 * (Rows.ofNet (Vs j).hU).nIn + c) = .msg q.val (kq / 8) (kq % 8) ∧
          SelPort Vs own sel s j u q) ∧
      ∀ Lb Lb' : (y : Fin S) → (Vs y).Bits,
        (∀ c, C c →
          FlockVBridge.copyVal (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u)
              (Lb j) hj c =
            FlockVBridge.copyVal
              (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u) (Lb' j) hj c) →
        Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
          Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x

/-- **A stage key's value, from layers agreeing under the stage's mask**: each of its columns is the same in either
layer as in its mask (`copyVal_maskOf`, the column at its port's commit string, `posK_of_key`). -/
theorem valOf_of_mask {M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)} {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hw : WColsAt own sel M K) {s : ℕ} {ρ : Fin 2} {x : Flock.RecAlgebra.Key} (hx : K s ρ x)
    (hnc : ∀ ℓ l, x ≠ .cons ℓ l) (Lb Lb' : (y : Fin S) → (Vs y).Bits)
    (he : maskOf Vs (skOf Vs own sel) s Lb = maskOf Vs (skOf Vs own sel) s Lb') :
    Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
      Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x := by
  obtain ⟨j, u, hj, C, hC, hv⟩ := hw s ρ x hx hnc
  refine hv Lb Lb' fun c hc => ?_
  obtain ⟨hn, hio, q, kq, hkq, hkey, hsel⟩ := hC c hc
  have hm : ∀ Lx : (y : Fin S) → (Vs y).Bits,
      FlockVBridge.copyVal (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u)
          (maskOf Vs (skOf Vs own sel) s Lx j) hj c =
        FlockVBridge.copyVal (FlockVBridge.keyInst (Vs j).hU (Vs j).nv (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv) u)
          (Lx j) hj c := fun Lx =>
    copyVal_maskOf s Lx j u hj hn hio fun hcw => by
      rw [posK_of_key u hj hn hkq hkey hcw]
      exact mem_skOf_inl.2 hsel
  rw [← hm Lb, ← hm Lb', he]

/-- **`WCombRead` from the decoder's columns and the reads file.** -/
theorem wCombRead_of {M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)} {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hw : WColsAt own sel M K) : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K := by
  intro s ρ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hX x hx hnc
  exact valOf_of_mask hw hx hnc _ _ hX

/-- **Hypothesis on the decoder (L's side): `rep₀` reads each element of `w` off input columns of one copy**: element
`i` of rep `ρ`'s `w`, where a key names it, is the same at any two layers agreeing on a set `cols ρ i` of copy 0's
columns of unit `un ρ i` of session `ses ρ i`, each an input column a row reads (`IoColAt`), keyed by bit `kq` of a row
port `q`. -/
def WDecodes (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) (ses : Fin 2 → ℕ → Fin S)
    (un : (ρ : Fin 2) → (i : ℕ) → Fin (Vs (ses ρ i)).nv) (cols : Fin 2 → ℕ → ℕ → Prop) : Prop :=
  ∀ (ρ : Fin 2) (x : Flock.RecAlgebra.Key) (i : ℕ), Flock.RecAlgebra.wi sh x = some i →
    ∃ hj : 0 < (Vs (ses ρ i)).c.upv,
      (∀ c, cols ρ i c → c < (Rows.ofNet (Vs (ses ρ i)).hU).nIn ∧ IoColAt (Vs (ses ρ i)) (un ρ i) c ∧
        ∃ (q : Fin (Vs (ses ρ i)).c.ports.size) (kq : ℕ), kq < Layout.portRowW (Vs (ses ρ i)).c q ∧
          Aliased.srcKey (Vs (ses ρ i)).c (0 * (Rows.ofNet (Vs (ses ρ i)).hU).nIn + c) = .msg q.val (kq / 8) (kq % 8)) ∧
      ∀ Lb Lb' : (y : Fin S) → (Vs y).Bits,
        (∀ c, cols ρ i c →
          FlockVBridge.copyVal (FlockVBridge.keyInst (Vs (ses ρ i)).hU (Vs (ses ρ i)).nv
              (Layout.outsOf (Vs (ses ρ i)).c (Vs (ses ρ i)).hU (Vs (ses ρ i)).nv) (un ρ i)) (Lb (ses ρ i)) hj c =
            FlockVBridge.copyVal (FlockVBridge.keyInst (Vs (ses ρ i)).hU (Vs (ses ρ i)).nv
              (Layout.outsOf (Vs (ses ρ i)).c (Vs (ses ρ i)).hU (Vs (ses ρ i)).nv) (un ρ i)) (Lb' (ses ρ i)) hj c) →
        (M.rep₀ Lb ρ).w[i]! = (M.rep₀ Lb' ρ).w[i]!

/-- **Hypothesis on the reads file (S's `SelPort`, per key)**: where stage `s`'s key `x` of rep `ρ` is element `i` of
`w`, stage `s` reads every port keying a column of `cols ρ i`. -/
def WSel (sh : Flock.Shape) (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (ses : Fin 2 → ℕ → Fin S) (un : (ρ : Fin 2) → (i : ℕ) → Fin (Vs (ses ρ i)).nv) (cols : Fin 2 → ℕ → ℕ → Prop)
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) x i, K s ρ x → (∀ ℓ l, x ≠ .cons ℓ l) → Flock.RecAlgebra.wi sh x = some i → ∀ c, cols ρ i c →
    ∀ (q : Fin (Vs (ses ρ i)).c.ports.size) (kq : ℕ), kq < Layout.portRowW (Vs (ses ρ i)).c q →
      Aliased.srcKey (Vs (ses ρ i)).c (0 * (Rows.ofNet (Vs (ses ρ i)).hU).nIn + c) = .msg q.val (kq / 8) (kq % 8) →
        SelPort Vs own sel s (ses ρ i) (un ρ i) q

/-- **Every stage key but the `.cons` keys is an element of `w`**: `w` names it, and it is not a ring switch's transpose
or a comb element. -/
def WKeys (sh : Flock.Shape) (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) x, K s ρ x → (∀ ℓ l, x ≠ .cons ℓ l) →
    (∃ i, Flock.RecAlgebra.wi sh x = some i) ∧ (∀ k u, x ≠ .rsT k u) ∧ ∀ l, x ≠ .comb l

/-- **`WColsAt` from the decoder's columns of `w` and the reads file per key**: each key's value is its element of `w`
(`valOf_w`), which `rep₀` reads off `cols`. -/
theorem wColsAt_of_w {M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)} {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    {ses : Fin 2 → ℕ → Fin S} {un : (ρ : Fin 2) → (i : ℕ) → Fin (Vs (ses ρ i)).nv} {cols : Fin 2 → ℕ → ℕ → Prop}
    (hK : WKeys sh K) (hd : WDecodes M ses un cols) (hs : WSel sh own sel ses un cols K) : WColsAt own sel M K := by
  intro s ρ x hx hnc
  obtain ⟨⟨i, hi⟩, h₁, h₂⟩ := hK s ρ x hx hnc
  obtain ⟨hj, hc, hv⟩ := hd ρ x i hi
  refine ⟨ses ρ i, un ρ i, hj, cols ρ i, fun c hcc => ?_, fun Lb Lb' he => ?_⟩
  · obtain ⟨hn, hio, q, kq, hkq, hkey⟩ := hc c hcc
    exact ⟨hn, hio, q, kq, hkq, hkey, hs s ρ x i hx hnc hi c hcc q kq hkq hkey⟩
  · rw [valOf_w hi h₁ h₂ hnc, valOf_w hi h₁ h₂ hnc]
    exact hv Lb Lb' he

/-- **`WCombRead` reads `hOfW`'s entries at the `.h` keys** at a holo shape. -/
theorem hOfW_read (hh : sh.holo = true) (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (i : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel i fun R ω t Lb R' ω' t' Lb' =>
      ∀ l < 64, K i ρ (.h l) →
        (Flock.RecAlgebra.hOfW sh (M.rep R ω t Lb ρ).w)[l]! =
          (Flock.RecAlgebra.hOfW sh (M.rep R' ω' t' Lb' ρ).w)[l]! := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX l hl hk
  rw [hOfW_valOf hh _ (M.rep₀ (layers zs dg N k Rw R ω t R₂ τ) ρ).comb (fun _ _ => 0) hl,
    hOfW_valOf hh _ (M.rep₀ (layers zs dg N k Rw R' ω' t' R₂' τ') ρ).comb (fun _ _ => 0) hl]
  exact hwc i ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX _ hk fun _ _ h => by cases h

/-! ## `TsRead` from the weave -/

/-- **Hypothesis: each rep's steps are a weave along `σ`** of messages `ms` and coins `cs` of fixed lengths `nm`, `nc`,
at every accepted outer proof. -/
def WovenAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (σ : List Exec.Kind) (ms cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg)
    (nm nc : Fin 2 → ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
    (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts ρ =
        weave σ (ms R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ) (cs R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ) ∧
      (ms R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ).length = nm ρ ∧
        (cs R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ).length = nc ρ

/-- **Hypothesis: each message is read on its stage's commit strings**: at two accepted outer proofs whose layers agree
under stage `s`'s mask, rep `ρ`'s messages at stage `s` are the same. -/
def WeaveMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (σ : List Exec.Kind) (c₀ : ℕ)
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun R ω t Lb R' ω' t' Lb' =>
    ∀ m, stageOf σ c₀ ρ m = s → (ms R ω t Lb ρ)[m]? = (ms R' ω' t' Lb' ρ)[m]?

/-- **`TsRead` from the weave**: rep 0's coins are `coins₀` at the fixed lengths (`nCoins_weave`), and each rep's round
is its messages of that round (`groups_weave`), each read on its stage. -/
theorem tsRead_of_woven {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {σ : List Exec.Kind} {ms cs : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg}
    {nm nc : Fin 2 → ℕ} (hw : WovenAt zs dg N k Rw src σ ms cs nm nc)
    (hr : WeaveMsgsRead (st := st) (hp := hp) zs dg N k Rw own sel σ (coins₀ σ (nm 0) (nc 0)) ms) :
    TsRead zs dg N k Rw src own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  obtain ⟨e₀, l₀, c₀⟩ := hw R ω t R₂ τ o hA 0
  obtain ⟨e₀', l₀', c₀'⟩ := hw R' ω' t' R₂' τ' o' hA' 0
  obtain ⟨e₁, l₁, c₁⟩ := hw R ω t R₂ τ o hA 1
  obtain ⟨e₁', l₁', c₁'⟩ := hw R' ω' t' R₂' τ' o' hA' 1
  have hn : ∀ {a b : List Msg}, a.length = nm 0 → b.length = nc 0 →
      nCoins (weave σ a b) = coins₀ σ (nm 0) (nc 0) := fun hm hc =>
    FlockC3.Sparse.nCoins_weave σ (by rw [hm, List.length_replicate]) (by rw [hc, List.length_replicate])
  have z : (0 : Fin 2).val = 0 := rfl
  have o₁ : (1 : Fin 2).val = 1 := rfl
  show RoundsAgree i _ _
  unfold RoundsAgree
  rw [e₀, e₀', e₁, e₁', hn l₀ c₀, hn l₀' c₀']
  refine ⟨rfl, fun j hj _ => ?_, fun j hj => ?_⟩
  · refine FlockC3.Sparse.groups_weave σ j (l₀.trans l₀'.symm) (c₀.trans c₀'.symm) fun m hm => ?_
    refine hr i 0 R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX m ?_
    rw [stageOf, z, hm, hj]
    omega
  · refine FlockC3.Sparse.groups_weave σ j (l₁.trans l₁'.symm) (c₁.trans c₁'.symm) fun m hm => ?_
    refine hr i 1 R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX m ?_
    rw [stageOf, o₁, hm, hj]
    omega

/-- **Hypothesis (the slices): each message reads only its keys**: at two accepted outer proofs whose layers agree
under stage `s`'s mask, rep `ρ`'s message `m` at stage `s` is the same where `rep₀` gives its keys `keys m` but the
`.cons` keys the same values. -/
def MsgsOfKeys (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (σ : List Exec.Kind) (c₀ : ℕ) (keys : ℕ → List Flock.RecAlgebra.Key)
    (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun R ω t Lb R' ω' t' Lb' =>
    ∀ m, stageOf σ c₀ ρ m = s →
      (∀ x ∈ keys m, (∀ ℓ l, x ≠ .cons ℓ l) →
        Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
          Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x) →
      (ms R ω t Lb ρ)[m]? = (ms R' ω' t' Lb' ρ)[m]?

/-- **The keys of one rep unit's messages**, in the messages' order. -/
def unitKeys (nX : ℕ) (sh : Flock.Shape) (m : ℕ) : List Flock.RecAlgebra.Key :=
  let z := sh.m - 6 + 2
  let l := sh.kLog - 6 + 2
  if m < z then zcKeys sh.m m
  else if m < z + l then lcKeys sh.kLog (m - z)
  else if m < z + l + (2 + nX) then rsKeys (m - z - l)
  else (ligKeys sh.sched).getD (m - z - l - (2 + nX)) []

/-- **The recursive caps are read at each stage** at the two layers that stage's commitments hold. -/
def RecCapsReadAt
    (b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc) : Prop :=
  ∀ s ρ, ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    RecCapsRead (b Lb ρ).lig (b Lb' ρ).lig

/-- **A rep unit's messages read only `unitKeys`**, composing the zerocheck, lincheck, ring-switch and Ligerito slices. -/
theorem msgsOfKeys_unit (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (nX : ℕ) (b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc) (hh : sh.holo = true)
    (hsh : ∀ Lb i, FlockVBridge.Family.ShapeIs sh (b Lb i).st (b Lb i).sch)
    (hcaps : RecCapsReadAt (st := st) (hp := hp) zs dg N k Rw own sel b) {σ : List Exec.Kind} {c₀ : ℕ} :
    MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel σ c₀ (unitKeys nX sh) M
      (fun R ω t Lb ρ => unitMsgs nX sh (M.rep R ω t Lb ρ) ((b Lb ρ).vals sh (M.rep R ω t Lb ρ))) := by
  intro s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX m _ hread
  let Lb := layers zs dg N k Rw R ω t R₂ τ
  let Lb' := layers zs dg N k Rw R' ω' t' R₂' τ'
  obtain ⟨hm, hkLog, hsch⟩ := hsh Lb ρ
  obtain ⟨hm', hkLog', hsch'⟩ := hsh Lb' ρ
  have hM : (b Lb ρ).st.m = (b Lb' ρ).st.m := hm.symm.trans hm'
  have hK : (b Lb ρ).st.kLog = (b Lb' ρ).st.kLog := hkLog.symm.trans hkLog'
  have hS : (b Lb ρ).sch = (b Lb' ρ).sch := hsch.symm.trans hsch'
  let z := sh.m - 6 + 2
  let l := sh.kLog - 6 + 2
  by_cases hz : m < z
  · exact unitMsgs_zc_read _ sh hM (by simpa [z, hm] using hz)
      (by simpa [unitKeys, z, l, hz, hm] using hread)
  by_cases hl : m < z + l
  · have hj : m - z < (b Lb ρ).st.kLog - 6 + 2 := by simp only [← hkLog]; omega
    have hi : (b Lb ρ).st.m - 6 + 2 + (m - z) = m := by simp only [← hm]; omega
    simpa only [hi] using unitMsgs_lc_read _ hh hM hK hj
      (by simpa [unitKeys, z, l, hz, hl, hkLog] using hread)
  by_cases hr : m < z + l + (2 + nX)
  · have hj : m - z - l < 2 + nX := by omega
    have hi : (b Lb ρ).st.m - 6 + 2 + ((b Lb ρ).st.kLog - 6 + 2) + (m - z - l) = m := by
      simp only [← hm, ← hkLog]
      omega
    simpa only [hi] using unitMsgs_rs_read _ sh hM hK hj
      (by simpa [unitKeys, z, l, hz, hl, hr] using hread)
  · have hi : (b Lb ρ).st.m - 6 + 2 + ((b Lb ρ).st.kLog - 6 + 2) + (2 + nX) +
        (m - z - l - (2 + nX)) = m := by
      simp only [← hm, ← hkLog]
      omega
    simpa only [hi] using unitMsgs_lig_getD_read _ sh hM hK hS
      (hcaps s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX) _
      (by simpa [unitKeys, z, l, hz, hl, hr, hsch] using hread)

/-- **Each message is read on its stage**, from its keys' read at the stage (`WCombRead` at `sentK`). -/
theorem weaveMsgsRead_of_keys {σ : List Exec.Kind} {c₀ : ℕ} {keys : ℕ → List Flock.RecAlgebra.Key}
    {M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)}
    {ms : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg}
    (hk : MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel σ c₀ keys M ms)
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M (sentK σ c₀ keys)) :
    WeaveMsgsRead (st := st) (hp := hp) zs dg N k Rw own sel σ c₀ ms := by
  intro s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX m hm
  exact hk s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX m hm fun x hx hnc =>
    hwc s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX x ⟨m, hm, hx, hnc⟩ hnc

/-! ## At `repsOfVals` -/

/-- The reps' extra claims at `holoAt st hp`, counted at the points zero (`extraClaims_length`: at any points). -/
def nXOf (st : Flock.Stmt) (hp : HoloParams) : ℕ :=
  (Model.extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
    (holoAt st hp).cls.mPts fun _ _ => (0 : FlockLevel3.GF128)).length

/-- **At `repsOfVals` each rep's steps are `unitWeave`'s weave**, along a shape its extra claims' points don't change. -/
theorem wovenAt_repsOfVals {nm nc : Fin 2 → ℕ}
    (hlen : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
      (unitMsgs (nXOf st hp) sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nm ρ ∧
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nc ρ) :
    WovenAt zs dg N k Rw (repsOfVals stp sh rep vals)
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (fun R ω t Lb ρ => unitMsgs (nXOf st hp) sh (rep R ω t Lb ρ) (vals R ω t Lb ρ))
      (fun R ω t Lb ρ => unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals R ω t Lb ρ))
      nm nc := by
  intro R ω t R₂ τ o hA ρ
  refine ⟨?_, hlen R ω t R₂ τ o hA ρ⟩
  show unitWeave _ _ (Model.extraClaims _ _ _ (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).pts).length sh _ _ = _
  rw [extraClaims_length _ _ _ _ fun _ _ => (0 : FlockLevel3.GF128)]
  rfl

/-- **`TsRead` at `repsOfVals`**, from the reps' fixed lengths, the slices (`MsgsOfKeys`) and the decoder's columns at
the sent keys (`WColsAt`). -/
theorem tsRead_repsOfVals (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) {nm nc : Fin 2 → ℕ}
    {keys : ℕ → List Flock.RecAlgebra.Key}
    (hlen : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
      (unitMsgs (nXOf st hp) sh (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nm ρ ∧
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nc ρ)
    (hk : MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (nm 0) (nc 0)) keys M (fun R ω t Lb ρ => unitMsgs (nXOf st hp) sh (M.rep R ω t Lb ρ) (vals R ω t Lb ρ)))
    (hc : WColsAt own sel M (sentK
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (nm 0) (nc 0)) keys)) :
    TsRead zs dg N k Rw (repsOfVals stp sh M.rep vals) own sel :=
  tsRead_of_woven (wovenAt_repsOfVals hlen) (weaveMsgsRead_of_keys hk (wCombRead_of hc))

/-- **The reps' lengths at `ValsSrc.vals`** are `b₀`'s where `b`'s setup's `m` and `k_log`, schedule and closing round
are `b₀`'s (the transcript starts at round 0 at either). -/
theorem lengths_vals (nX : ℕ) (S₀ : Model.Statement) (b b₀ : ValsSrc) (d d₀ : RepData) (hm : b.st.m = b₀.st.m)
    (hk : b.st.kLog = b₀.st.kLog) (hs : b.sch = b₀.sch) (h₄ : b.t₄.k = b₀.t₄.k) :
    (unitMsgs nX sh d (b.vals sh d)).length = (unitMsgs nX sh d₀ (b₀.vals sh d₀)).length ∧
      (unitCoins S₀ (b.vals sh d)).length = (unitCoins S₀ (b₀.vals sh d₀)).length :=
  ⟨unitMsgs_length_of nX (v := b.vals sh d) (v' := b₀.vals sh d₀) hm hk hs,
    unitCoins_length_of S₀ (v := b.vals sh d) (v' := b₀.vals sh d₀) rfl hm h₄⟩

/-- **`TsRead` at `repsOfVals` of `ValsSrc`'s values**, `tsRead_repsOfVals` with the lengths `b₀`'s (`lengths_vals`),
where every `b` has `b₀`'s setup's `m` and `k_log`, schedule and closing round. -/
theorem tsRead_valsSrc (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} (b₀ : ValsSrc) (d₀ : RepData)
    {keys : ℕ → List Flock.RecAlgebra.Key}
    (hb : ∀ Lb i, (b Lb i).st.m = b₀.st.m ∧ (b Lb i).st.kLog = b₀.st.kLog ∧ (b Lb i).sch = b₀.sch ∧
      (b Lb i).t₄.k = b₀.t₄.k)
    (hk : MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length) keys M
      (fun R ω t Lb ρ => unitMsgs (nXOf st hp) sh (M.rep R ω t Lb ρ) ((b Lb ρ).vals sh (M.rep R ω t Lb ρ))))
    (hc : WColsAt own sel M (sentK
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length) keys)) :
    TsRead zs dg N k Rw (repsOfVals stp sh M.rep fun R ω t Lb i => (b Lb i).vals sh (M.rep R ω t Lb i)) own sel :=
  tsRead_repsOfVals M (nm := fun _ => (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length)
    (nc := fun _ => (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length)
    (fun R ω t R₂ τ o _ ρ => by
      obtain ⟨h₁, h₂, h₃, h₄⟩ := hb (layers zs dg N k Rw R ω t R₂ τ) ρ
      exact lengths_vals _ _ _ _ _ _ h₁ h₂ h₃ h₄)
    hk hc

/-- **`tsRead_valsSrc`' `hb` from V*'s shape**: where V*'s shape is every `b`'s setup's and schedule's and `b₀`'s
(`ShapeIs`), and every `b` closes at `b₀`'s round. -/
theorem hb_of_shape {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} {b₀ : ValsSrc}
    (h : ∀ Lb i, FlockVBridge.Family.ShapeIs sh (b Lb i).st (b Lb i).sch ∧ (b Lb i).t₄.k = b₀.t₄.k)
    (h₀ : FlockVBridge.Family.ShapeIs sh b₀.st b₀.sch) :
    ∀ Lb i, (b Lb i).st.m = b₀.st.m ∧ (b Lb i).st.kLog = b₀.st.kLog ∧ (b Lb i).sch = b₀.sch ∧
      (b Lb i).t₄.k = b₀.t₄.k :=
  fun Lb i => ⟨(h Lb i).1.1.symm.trans h₀.1, (h Lb i).1.2.1.symm.trans h₀.2.1, (h Lb i).1.2.2.symm.trans h₀.2.2,
    (h Lb i).2⟩

/-! ## The cap and the reps' draws at `ValsSrc`'s values -/

/-- **`CapRead` where the source's cap is one constant** at every accepted outer proof: it reads no layer. -/
theorem capRead_of_const {cap : Array ByteArray} (hcs : AtSrc zs dg N k Rw stp fun _ u => u.cap₀ = capDig cap) :
    CapRead zs dg N k Rw stp own sel :=
  fun R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' _ =>
    Eq.trans (hcs R ω t R₂ τ o hA) (Eq.symm (hcs R' ω' t' R₂' τ' o' hA'))

/-- **The cap tie at `ValsSrc`'s values** (`stepsOfVals_repsOfVals`' `hcap`): every `b`'s cap is `cap` (`proofOf`'s
cap is `ValsSrc.cap`), and so is the source's. -/
theorem capTie_vals {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} {cap : Array ByteArray}
    (hb : ∀ Lb i, (b Lb i).cap = cap) (hcs : AtSrc zs dg N k Rw stp fun _ u => u.cap₀ = capDig cap) :
    AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      capDig ((b Lb i).vals sh (rep R ω t Lb i)).p.cap = (stp R ω t Lb).cap₀ :=
  fun R ω t R₂ τ o hA i _ _ =>
    Eq.trans (congrArg capDig ((cap_vals _ sh _).trans (hb (layers zs dg N k Rw R ω t R₂ τ) i)))
      (Eq.symm (hcs R ω t R₂ τ o hA))

/-- A rep's consumed rounds (`repIdx`) at `ValsSrc`'s values, which start at round 0: `b`'s alone. -/
def repIdxB (S₀ : Model.Statement) (b : ValsSrc) : List ℕ :=
  List.range' 0 (3 + (b.st.m - 6) + 2 + (S₀.kLog - 6)) ++
    List.range' (3 + (b.st.m - 6) + 2 + (S₀.kLog - 6) + 1) (b.t₄.k - (3 + (b.st.m - 6) + 2 + (S₀.kLog - 6) + 1))

theorem repIdx_vals (S₀ : Model.Statement) (b : ValsSrc) (sh : Flock.Shape) (d : RepData) :
    repIdx S₀ (b.vals sh d) = repIdxB S₀ b := rfl

/-- **`RepsDrawn` on `b` alone**: at every layers, the history's round `repStart c₀ i + p` (`c₀` rep 0's consumed
rounds) draws round `q` of stream `s i`, `q` the `p`-th of rep `i`'s (`repIdxB`), and those rounds are below `r`. -/
def DrawnOf (S₀ : Model.Statement) (r : ℕ) (b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc)
    (selC : ℕ → String → ℕ → Prop) (s : Fin 2 → String) : Prop :=
  ∀ Lb i, repStart (repIdxB S₀ (b Lb 0)).length i + (repIdxB S₀ (b Lb i)).length ≤ r ∧
    ∀ p q, (repIdxB S₀ (b Lb i))[p]? = some q → selC (repStart (repIdxB S₀ (b Lb 0)).length i + p) (s i) q

/-- **`RepsDrawn` at `ValsSrc`'s values and the identity `ix`** (`repRoundsHeld_of_rounds`'), from `DrawnOf`. -/
theorem repsDrawn_valsSrc {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} {selC : ℕ → String → ℕ → Prop}
    {s : Fin 2 → String} (h : DrawnOf ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) r b selC s) :
    RepsDrawn zs dg N k Rw sh rep (fun R ω t Lb i => (b Lb i).vals sh (rep R ω t Lb i)) selC s fun _ q => q :=
  fun R ω t R₂ τ _ _ i _ _ => h (layers zs dg N k Rw R ω t R₂ τ) i

end

end FlockC3.Reps

#print axioms FlockC3.Reps.pwRead_nil
#print axioms FlockC3.Reps.pwRead_cons
#print axioms FlockC3.Reps.pwRead_append
#print axioms FlockC3.Reps.pwRead_map
#print axioms FlockC3.Reps.msgK_mem
#print axioms FlockC3.Reps.recMsgs_read
#print axioms FlockC3.Reps.ligMsgs_read
#print axioms FlockC3.Reps.keysRead_at
#print axioms FlockC3.Reps.ligMsgs_proofOf_read
#print axioms FlockC3.Reps.unitMsgs_lig
#print axioms FlockC3.Reps.unitMsgs_lig_read
#print axioms FlockC3.Reps.unitMsgs_lig_getD_read
#print axioms FlockC3.Reps.msgsOfKeys_unit
