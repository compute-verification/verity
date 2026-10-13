import Proofs.Flock.C3.Reps.IoCols
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
  (`unitMsgs_zc_read`), and its next `k_log − 4`, the lincheck's with `h`, only `lcKeys` at a holo shape
  (`unitMsgs_lc_read`). No key of either is `.cons`, so `rep₀`'s consistency sums stand in for the rep's
  (`valOf_cons_irrel`).
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

open FlockSoundness.Refine (zcMsgs lcMsgs holoLcMsgs holoRepMsgs vecOf pairOf fOf)
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

/-- **A key that is not a consistency sum reads no consistency sum.** -/
theorem valOf_cons_irrel (sh : Flock.Shape) (w comb : Array F128) (c c' : ℕ → ℕ → F128)
    {x : Flock.RecAlgebra.Key} (hx : ∀ ℓ l, x ≠ .cons ℓ l) :
    Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w comb c' x := by
  cases x with
  | cons ℓ l => exact absurd rfl (hx ℓ l)
  | _ => rfl

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
theorem unitMsgs_zc_read (nX : ℕ) (sh : Flock.Shape) {b b' : ValsSrc} {d d' : RepData} (hst : b.st = b'.st) {m : ℕ}
    (hm : m < b.st.m - 6 + 2)
    (h : ∀ x ∈ zcKeys b.st.m m, (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x) :
    (unitMsgs nX sh d (b.vals sh d))[m]? = (unitMsgs nX sh d' (b'.vals sh d'))[m]? := by
  have hm' : m < b'.st.m - 6 + 2 := hst ▸ hm
  rw [unitMsgs_zc nX sh d (b.vals sh d) hm, unitMsgs_zc nX sh d' (b'.vals sh d') hm']
  show (zcMsgs b.st.m (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).zerocheck)[m]? =
    (zcMsgs b'.st.m (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').zerocheck)[m]?
  rw [← hst]
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
    (hst : b.st = b'.st) {j : ℕ} (hj : j < b.st.kLog - 6 + 2)
    (h : ∀ x ∈ lcKeys b.st.kLog j, (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh d.w d.comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh d'.w d'.comb (fun _ _ => 0) x) :
    (unitMsgs nX sh d (b.vals sh d))[b.st.m - 6 + 2 + j]? =
      (unitMsgs nX sh d' (b'.vals sh d'))[b.st.m - 6 + 2 + j]? := by
  have hj' : j < b'.st.kLog - 6 + 2 := hst ▸ hj
  rw [unitMsgs_lc nX sh d (b.vals sh d) (i := b.st.m - 6 + 2 + j) rfl hj,
    unitMsgs_lc nX sh d' (b'.vals sh d') (i := b.st.m - 6 + 2 + j)
      (congrArg (fun s : Flock.Setup => s.m - 6 + 2 + j) hst) hj']
  show (holoLcMsgs b.st.kLog (FlockVBridge.Family.proofOf sh b.params b.cap b.lig d).lincheck
      (RecAlgebra.hOfW sh d.w))[j]? =
    (holoLcMsgs b'.st.kLog (FlockVBridge.Family.proofOf sh b'.params b'.cap b'.lig d').lincheck
      (RecAlgebra.hOfW sh d'.w))[j]?
  rw [← hst]
  exact holoLcMsgs_read b.st.kLog hh j fun x hx =>
    (valOf_cons_irrel sh d.w d.comb d.cons (fun _ _ => 0) (lcKeys_noCons hx)).trans
      ((h x hx (lcKeys_noCons hx)).trans (valOf_cons_irrel sh d'.w d'.comb (fun _ _ => 0) d'.cons (lcKeys_noCons hx)))

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
  {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}

local notation "ℐ" => holoInnerAt T CT st hp

/-! ## No `.cons` key: `AlphaRead` and `RowColsRead` vacuously -/

/-- **`AlphaRead` where no stage's keys read a consistency sum.** -/
theorem alphaRead_of_noCons (M : RepMaps stmt Vs sh) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hK : ∀ s ρ ℓ l, ¬ K s ρ (.cons ℓ l)) : AlphaRead (st := st) (hp := hp) zs dg N k Rw own sel M K := by
  intro s ρ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ℓ l hk
  exact absurd hk (hK s ρ ℓ l)

/-- **The forms a stage's sent keys read, at the decoder**: `formsRead_maps` at `sentK`, `AlphaRead` and `RowColsRead`
vacuous there. -/
theorem formsRead_sent (M : RepMaps stmt Vs sh) {σ : List Exec.Kind} {c₀ : ℕ} {keys : ℕ → List Flock.RecAlgebra.Key}
    (hu : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ (ρ : Fin 2), ∀ ℓ < sh.sched.levels.size,
      ∀ j < M.q ℓ, ∃ a cf A, FlockVBridge.Family.OpensAtL (Vs (M.sessO ℓ))
        (layers zs dg N k Rw R ω t R₂ τ (M.sessO ℓ)) (M.unO ℓ ρ j) (FlockVBridge.Family.cpO ℓ ρ j) (levelLanes sh ℓ)
        ((M.rows (layers zs dg N k Rw R ω t R₂ τ) ρ ℓ)[j]!) a cf A)
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M (sentK σ c₀ keys)) (s : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
      ∀ f : Flock.RecAlgebra.Lin, LinReads f (sentK σ c₀ keys s ρ) →
        Flock.RecAlgebra.Lin.eval (Flock.RecAlgebra.valOf sh (M.rep Lb ρ).w (M.rep Lb ρ).comb (M.rep Lb ρ).cons) f =
          Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep Lb' ρ).w (M.rep Lb' ρ).comb (M.rep Lb' ρ).cons) f :=
  formsRead_maps M hu hwc (M.rowColsRead_of_noCons sentK_noCons) (alphaRead_of_noCons M sentK_noCons) s ρ

/-! ## `WCombRead` from the reads file -/

/-- **Hypothesis on the decoder: `rep₀` reads each stage key off input columns of one copy**: for each key `x` of
`K s ρ` but the `.cons` keys, `rep₀` gives `x` the same value at any two layers agreeing on a set `C` of copy 0's
columns of a unit `u` of a session `j`, each an input column a row reads (`IoColAt`), keyed by bit `kq` of a row port
`q` that stage `s` reads (`SelPort`). -/
def WColsAt (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh)
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
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
theorem valOf_of_mask {M : RepMaps stmt Vs sh} {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
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
theorem wCombRead_of {M : RepMaps stmt Vs sh} {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hw : WColsAt own sel M K) : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K := by
  intro s ρ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hX x hx hnc
  exact valOf_of_mask hw hx hnc _ _ hX

/-- **`WCombRead` reads `hOfW`'s entries at the `.h` keys** at a holo shape. -/
theorem hOfW_read (hh : sh.holo = true) (M : RepMaps stmt Vs sh) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (i : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel i fun _ _ _ Lb _ _ _ Lb' =>
      ∀ l < 64, K i ρ (.h l) →
        (Flock.RecAlgebra.hOfW sh (M.rep Lb ρ).w)[l]! = (Flock.RecAlgebra.hOfW sh (M.rep Lb' ρ).w)[l]! := by
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
    (σ : List Exec.Kind) (ms cs : ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg) (nm nc : Fin 2 → ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
    (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts ρ =
        weave σ (ms (layers zs dg N k Rw R ω t R₂ τ) ρ) (cs (layers zs dg N k Rw R ω t R₂ τ) ρ) ∧
      (ms (layers zs dg N k Rw R ω t R₂ τ) ρ).length = nm ρ ∧ (cs (layers zs dg N k Rw R ω t R₂ τ) ρ).length = nc ρ

/-- **Hypothesis: each message is read on its stage's commit strings**: at two accepted outer proofs whose layers agree
under stage `s`'s mask, rep `ρ`'s messages at stage `s` are the same. -/
def WeaveMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (σ : List Exec.Kind) (c₀ : ℕ)
    (ms : ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    ∀ m, stageOf σ c₀ ρ m = s → (ms Lb ρ)[m]? = (ms Lb' ρ)[m]?

/-- **`TsRead` from the weave**: rep 0's coins are `coins₀` at the fixed lengths (`nCoins_weave`), and each rep's round
is its messages of that round (`groups_weave`), each read on its stage. -/
theorem tsRead_of_woven {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {σ : List Exec.Kind} {ms cs : ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg} {nm nc : Fin 2 → ℕ}
    (hw : WovenAt zs dg N k Rw src σ ms cs nm nc)
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
    (M : RepMaps stmt Vs sh) (ms : ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    ∀ m, stageOf σ c₀ ρ m = s →
      (∀ x ∈ keys m, (∀ ℓ l, x ≠ .cons ℓ l) →
        Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
          Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x) →
      (ms Lb ρ)[m]? = (ms Lb' ρ)[m]?

/-- **Each message is read on its stage**, from its keys' read at the stage (`WCombRead` at `sentK`). -/
theorem weaveMsgsRead_of_keys {σ : List Exec.Kind} {c₀ : ℕ} {keys : ℕ → List Flock.RecAlgebra.Key}
    {M : RepMaps stmt Vs sh} {ms : ((j : Fin S) → (Vs j).Bits) → Fin 2 → List Msg}
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
      (unitMsgs (nXOf st hp) sh (rep (layers zs dg N k Rw R ω t R₂ τ) ρ)
          (vals (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nm ρ ∧
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nc ρ) :
    WovenAt zs dg N k Rw (repsOfVals stp sh rep vals)
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (fun Lb ρ => unitMsgs (nXOf st hp) sh (rep Lb ρ) (vals Lb ρ))
      (fun Lb ρ => unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (vals Lb ρ)) nm nc := by
  intro R ω t R₂ τ o hA ρ
  refine ⟨?_, hlen R ω t R₂ τ o hA ρ⟩
  show unitWeave _ _ (Model.extraClaims _ _ _ (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).pts).length sh _ _ = _
  rw [extraClaims_length _ _ _ _ fun _ _ => (0 : FlockLevel3.GF128)]
  rfl

/-- **`TsRead` at `repsOfVals`**, from the reps' fixed lengths, the slices (`MsgsOfKeys`) and the decoder's columns at
the sent keys (`WColsAt`). -/
theorem tsRead_repsOfVals (M : RepMaps stmt Vs sh) {nm nc : Fin 2 → ℕ} {keys : ℕ → List Flock.RecAlgebra.Key}
    (hlen : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
      (unitMsgs (nXOf st hp) sh (M.rep (layers zs dg N k Rw R ω t R₂ τ) ρ)
          (vals (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nm ρ ∧
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
          (vals (layers zs dg N k Rw R ω t R₂ τ) ρ)).length = nc ρ)
    (hk : MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (nm 0) (nc 0)) keys M (fun Lb ρ => unitMsgs (nXOf st hp) sh (M.rep Lb ρ) (vals Lb ρ)))
    (hc : WColsAt own sel M (sentK
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (nm 0) (nc 0)) keys)) :
    TsRead zs dg N k Rw (repsOfVals stp sh M.rep vals) own sel :=
  tsRead_of_woven (wovenAt_repsOfVals hlen) (weaveMsgsRead_of_keys hk (wCombRead_of hc))

/-- **The reps' lengths at `ValsSrc.vals`** are `b₀`'s where `b`'s setup, schedule and closing round are `b₀`'s (the
transcript starts at round 0 at either). -/
theorem lengths_vals (nX : ℕ) (S₀ : Model.Statement) (b b₀ : ValsSrc) (d d₀ : RepData) (hst : b.st = b₀.st)
    (hs : b.sch = b₀.sch) (h₄ : b.t₄.k = b₀.t₄.k) :
    (unitMsgs nX sh d (b.vals sh d)).length = (unitMsgs nX sh d₀ (b₀.vals sh d₀)).length ∧
      (unitCoins S₀ (b.vals sh d)).length = (unitCoins S₀ (b₀.vals sh d₀)).length :=
  ⟨unitMsgs_length nX (v := b.vals sh d) (v' := b₀.vals sh d₀) hst hs,
    unitCoins_length S₀ (v := b.vals sh d) (v' := b₀.vals sh d₀) rfl hst h₄⟩

/-- **`TsRead` at `repsOfVals` of `ValsSrc`'s values**, `tsRead_repsOfVals` with the lengths `b₀`'s (`lengths_vals`),
where every `b` has `b₀`'s setup, schedule and closing round. -/
theorem tsRead_valsSrc (M : RepMaps stmt Vs sh) {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} (b₀ : ValsSrc)
    (d₀ : RepData) {keys : ℕ → List Flock.RecAlgebra.Key}
    (hb : ∀ Lb i, (b Lb i).st = b₀.st ∧ (b Lb i).sch = b₀.sch ∧ (b Lb i).t₄.k = b₀.t₄.k)
    (hk : MsgsOfKeys (st := st) (hp := hp) zs dg N k Rw own sel
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length) keys M
      (fun Lb ρ => unitMsgs (nXOf st hp) sh (M.rep Lb ρ) ((b Lb ρ).vals sh (M.rep Lb ρ))))
    (hc : WColsAt own sel M (sentK
      (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
      (coins₀ (unitShape ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (nXOf st hp))
        (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length
        (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length) keys)) :
    TsRead zs dg N k Rw (repsOfVals stp sh M.rep fun Lb i => (b Lb i).vals sh (M.rep Lb i)) own sel :=
  tsRead_repsOfVals M (nm := fun _ => (unitMsgs (nXOf st hp) sh d₀ (b₀.vals sh d₀)).length)
    (nc := fun _ => (unitCoins ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (b₀.vals sh d₀)).length)
    (fun R ω t R₂ τ o _ ρ => by
      obtain ⟨h₁, h₂, h₃⟩ := hb (layers zs dg N k Rw R ω t R₂ τ) ρ
      exact lengths_vals _ _ _ _ _ _ h₁ h₂ h₃)
    hk hc

end

end FlockC3.Reps

#print axioms FlockC3.Reps.unitWeave_eq
#print axioms FlockC3.Reps.extraClaims_length
#print axioms FlockC3.Reps.hOfW_valOf
#print axioms FlockC3.Reps.sentK_noCons
#print axioms FlockC3.Reps.alphaRead_of_noCons
#print axioms FlockC3.Reps.formsRead_sent
#print axioms FlockC3.Reps.valOf_of_mask
#print axioms FlockC3.Reps.wCombRead_of
#print axioms FlockC3.Reps.hOfW_read
#print axioms FlockC3.Reps.tsRead_of_woven
#print axioms FlockC3.Reps.weaveMsgsRead_of_keys
#print axioms FlockC3.Reps.wovenAt_repsOfVals
#print axioms FlockC3.Reps.tsRead_repsOfVals
#print axioms FlockC3.Reps.recMsgs_length
#print axioms FlockC3.Reps.ligMsgs_length
#print axioms FlockC3.Reps.unitMsgs_length
#print axioms FlockC3.Reps.unitCoins_length
#print axioms FlockC3.Reps.lengths_vals
#print axioms FlockC3.Reps.tsRead_valsSrc
#print axioms FlockC3.Reps.list3_get
#print axioms FlockC3.Reps.list3_length
#print axioms FlockC3.Reps.append_get_right
#print axioms FlockC3.Reps.list2_get
#print axioms FlockC3.Reps.list2_length
#print axioms FlockC3.Reps.vars_read
#print axioms FlockC3.Reps.pairs_read
#print axioms FlockC3.Reps.valOf_cons_irrel
#print axioms FlockC3.Reps.zcKeys_noCons
#print axioms FlockC3.Reps.zcMsgs_read
#print axioms FlockC3.Reps.zcMsgs_length
#print axioms FlockC3.Reps.unitMsgs_zc
#print axioms FlockC3.Reps.unitMsgs_zc_read
#print axioms FlockC3.Reps.lcKeys_noCons
#print axioms FlockC3.Reps.holoLcMsgs_read
#print axioms FlockC3.Reps.holoLcMsgs_length
#print axioms FlockC3.Reps.unitMsgs_lc
#print axioms FlockC3.Reps.unitMsgs_lc_read
