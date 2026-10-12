import Proofs.Flock.C3.Reps.RepsRead
import Proofs.Flock.C3.Reps.Maps
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.VBridge.Net

/-!
# The values the reps' forms read at a stage, at L's decoder

At `ValsSrc.vals`, rep `ρ`'s messages are `b`'s constants, `hOfW sh w` and the forms of `decoded sh w comb cons`
(`symbolic sh` at `valOf`) at the decoder's values `M.rep Lb ρ`, so `TsRead` there is a read fact of the values those
forms read. `formsRead_maps`: at stage `s`, the forms of rep `ρ` reading only the keys `K s ρ` take the same values at
two accepted outer proofs whose layers agree under stage `s`'s mask, from three named facts.

* A form reads only its keys (`linEval_congr`, `LinReads`).
* A consistency sum `cons ℓ l` (a `.cons` key) is `consAt` of level `ℓ`'s opened rows and `α`. The rows are pinned by
  the layers at every accepted proof (`OpensAtL`'s row columns, `rows_congr`), so they agree where the copies' row
  columns are read (`RowColsRead`). `α` is not in the layers (only `coef`, `κ`-weighted, is): `AlphaRead`.
* Every other key reads `rep₀`'s `w` and `comb` only (`valOf_cons`), which stage `s` reads (`WCombRead`).
* `rep₀`'s `rounds` and `extras` enter no form.
-/

namespace FlockC3.Reps

noncomputable section

open Flock

section algebra

/-- **The keys a form reads**: its terms' keys and its products' factors. -/
def LinReads (f : Flock.RecAlgebra.Lin) (P : Flock.RecAlgebra.Key → Prop) : Prop :=
  (∀ kv ∈ f.t, P kv.1) ∧ ∀ kv ∈ f.q, P kv.1.1 ∧ P kv.1.2

/-- **A form's value reads only its keys.** -/
theorem linEval_congr {val val' : Flock.RecAlgebra.Key → F128} {f : Flock.RecAlgebra.Lin}
    {P : Flock.RecAlgebra.Key → Prop} (hf : LinReads f P) (h : ∀ k, P k → val k = val' k) :
    Flock.RecAlgebra.Lin.eval val f = Flock.RecAlgebra.Lin.eval val' f := by
  unfold Flock.RecAlgebra.Lin.eval
  rw [Flock.RecAlgebra.tEval_congr f.t fun kv hkv => h kv.1 (hf.1 kv hkv),
    Flock.RecAlgebra.tEval_congr (g := fun ab => val ab.1 * val ab.2) (g' := fun ab => val' ab.1 * val' ab.2) f.q
      fun kv hkv => by
        show val kv.1.1 * val kv.1.2 = val' kv.1.1 * val' kv.1.2
        rw [h _ (hf.2 kv hkv).1, h _ (hf.2 kv hkv).2]]

/-- **Only the `.cons` keys read the consistency sums.** -/
theorem valOf_cons {sh : Flock.Shape} {w comb : Array F128} {c c' : ℕ → ℕ → F128} {x : Flock.RecAlgebra.Key}
    (hx : ∀ ℓ l, x ≠ .cons ℓ l) : Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w comb c' x := by
  cases x with
  | cons ℓ l => exact absurd rfl (hx ℓ l)
  | _ => rfl

/-- **Values agreeing at `P`'s keys**: off `.cons` from `w` and `comb`, at `.cons` from the sums. -/
theorem valOf_read {sh : Flock.Shape} {w comb w' comb' : Array F128} {c c' : ℕ → ℕ → F128}
    {P : Flock.RecAlgebra.Key → Prop}
    (hwc : ∀ x, P x → (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh w comb (fun _ _ => 0) x = Flock.RecAlgebra.valOf sh w' comb' (fun _ _ => 0) x)
    (hc : ∀ ℓ l, P (.cons ℓ l) → c ℓ l = c' ℓ l) :
    ∀ x, P x → Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w' comb' c' x := by
  intro x hx
  cases x with
  | cons ℓ l => exact hc ℓ l hx
  | _ => exact (valOf_cons (fun _ _ h => by cases h)).trans ((hwc _ hx fun _ _ h => by cases h).trans
      (valOf_cons (fun _ _ h => by cases h)))

/-- **A consistency sum reads its level's first `q` rows and `α`.** -/
theorem consAt_congr {rows rows' : Array (Array F128)} {q : ℕ} {α α' : Array F128} (l : ℕ)
    (h : ∀ j < q, ∀ l : ℕ, (rows[j]!)[l]! = (rows'[j]!)[l]!) (ha : α = α') :
    Flock.RecAlgebra.consAt rows q α l = Flock.RecAlgebra.consAt rows' q α' l := by
  rw [← ha]
  exact Flock.RecAlgebra.rangeSum_congr fun j hj => by rw [h j hj l]

end algebra

/-- Copy `j` of unit `u`'s columns in the layer `X`, as `OpensAtL` reads them. -/
def colsAt (Q : FlockSoundness.Discharge.Recursion.VStmt) (X : Q.Bits) (u : Fin Q.nv) {j : ℕ} (hj : j < Q.c.upv) :
    ℕ → Bool :=
  FlockVBridge.copyVal (FlockVBridge.keyInst Q.hU Q.nv (FlockSoundness.Discharge.Layout.outsOf Q.c Q.hU Q.nv) u) X hj

/-- **An opened row is its copy's row columns**: two layers whose copies agree on the row's `128 · LANES` columns
open the same row there (`OpensAtL`'s `ColsHold`, `rowBytes_inj`'s bits). -/
theorem rows_congr {Q : FlockSoundness.Discharge.Recursion.VStmt} {X X' : Q.Bits} {u : Fin Q.nv} {j LANES : ℕ}
    {r r' a a' cf cf' A A' : Array F128}
    (h : FlockVBridge.Family.OpensAtL Q X u j LANES r a cf A)
    (h' : FlockVBridge.Family.OpensAtL Q X' u j LANES r' a' cf' A')
    (hc : ∀ hj : j < Q.c.upv, ∀ t < 128 * LANES, colsAt Q X u hj t = colsAt Q X' u hj t) :
    ∀ l : ℕ, r[l]! = r'[l]! := by
  obtain ⟨hj, -, -, _, -, hs, -, hr, -⟩ := h
  obtain ⟨hj', -, -, _, -, hs', -, hr', -⟩ := h'
  intro l
  by_cases hl : l < LANES
  · refine FlockVBridge.AccChain.getElem!_of_bitL fun t ht => ?_
    have hb : 128 * l + t < 8 * (MerkleScheme.rowBytes r).size := by rw [FlockVBridge.size_rowBytes, hs]; omega
    have hb' : 128 * l + t < 8 * (MerkleScheme.rowBytes r').size := by rw [FlockVBridge.size_rowBytes, hs']; omega
    rw [← hr _ hb, ← hr' _ hb', Nat.zero_add]
    exact hc hj _ (by omega)
  · rw [getElem!_neg r l (by omega), getElem!_neg r' l (by omega)]

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf cpO)
open FlockVBridge.AccChain (levelLanes)
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}

local notation "ℐ" => holoInnerAt T CT st hp

theorem rep_w (M : RepMaps stmt Vs sh) (l : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) : (M.rep l ρ).w = (M.rep₀ l ρ).w :=
  rfl

theorem rep_comb (M : RepMaps stmt Vs sh) (l : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) :
    (M.rep l ρ).comb = (M.rep₀ l ρ).comb := rfl

theorem rep_cons (M : RepMaps stmt Vs sh) (l : (j : Fin S) → (Vs j).Bits) (ρ : Fin 2) :
    (M.rep l ρ).cons = fun ℓ i => Flock.RecAlgebra.consAt (M.rows l ρ ℓ) (M.q ℓ) (M.alpha l ρ ℓ) i := rfl

/-- **Hypothesis: stage `s` reads the `w` and `comb` its forms read** (the fact about V*'s statements): at two accepted
outer proofs whose layers agree under stage `s`'s mask, rep `ρ`'s `rep₀` gives every key of `K s ρ` but the `.cons`
keys the same value. -/
def WCombRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh) (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    ∀ x, K s ρ x → (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x

/-- **Hypothesis: level `ℓ`'s opened rows are read where stage `s` reads them**: wherever rep `ρ`'s stage-`s` keys read a
consistency sum of level `ℓ`, `ℓ` is a level, and layers agreeing under stage `s`'s mask agree on the row columns of
every copy of rep `ρ`'s level-`ℓ` openings. -/
def RowColsRead (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh)
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) ℓ l, K s ρ (.cons ℓ l) → ℓ < sh.sched.levels.size ∧ ∀ j < M.q ℓ,
    ∀ (hj : cpO ℓ ρ j < (Vs (M.sessO ℓ)).c.upv) (X X' : (x : Fin S) → (Vs x).Bits),
      maskOf Vs (skOf Vs own sel) s X = maskOf Vs (skOf Vs own sel) s X' → ∀ t < 128 * levelLanes sh ℓ,
        colsAt (Vs (M.sessO ℓ)) (X (M.sessO ℓ)) (M.unO ℓ ρ j) hj t =
          colsAt (Vs (M.sessO ℓ)) (X' (M.sessO ℓ)) (M.unO ℓ ρ j) hj t

/-- **Hypothesis: stage `s` reads `α` at the levels its sums read**: at two accepted outer proofs whose layers agree
under stage `s`'s mask, rep `ρ`'s `α` agrees at every level `ℓ` of a `.cons ℓ l` key of `K s ρ`. -/
def AlphaRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh) (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    ∀ ℓ l, K s ρ (.cons ℓ l) → M.alpha Lb ρ ℓ = M.alpha Lb' ρ ℓ

/-- **The values rep `ρ`'s stage-`s` forms read, at the decoder** (`TsRead`'s read fact at `ValsSrc.vals`): every form
reading only `K s ρ` takes the same value at `M.rep` of two accepted outer proofs whose layers agree under stage `s`'s
mask. `hu` is `accChains`' openings, their `acc` and `coef` forgotten. -/
theorem formsRead_maps (M : RepMaps stmt Vs sh) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hu : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ (ρ : Fin 2), ∀ ℓ < sh.sched.levels.size,
      ∀ j < M.q ℓ, ∃ a cf A, FlockVBridge.Family.OpensAtL (Vs (M.sessO ℓ))
        (layers zs dg N k Rw R ω t R₂ τ (M.sessO ℓ)) (M.unO ℓ ρ j) (cpO ℓ ρ j) (levelLanes sh ℓ)
        ((M.rows (layers zs dg N k Rw R ω t R₂ τ) ρ ℓ)[j]!) a cf A)
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (hcols : RowColsRead own sel M K)
    (hal : AlphaRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (s : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
      ∀ f : Flock.RecAlgebra.Lin, LinReads f (K s ρ) →
        Flock.RecAlgebra.Lin.eval (Flock.RecAlgebra.valOf sh (M.rep Lb ρ).w (M.rep Lb ρ).comb (M.rep Lb ρ).cons) f =
          Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep Lb' ρ).w (M.rep Lb' ρ).comb (M.rep Lb' ρ).cons) f := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX f hf
  simp only [rep_w, rep_comb, rep_cons]
  apply linEval_congr hf
  apply valOf_read (hwc s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX)
  intro ℓ l hk
  obtain ⟨hℓ, hc⟩ := hcols s ρ ℓ l hk
  refine consAt_congr l (fun j hj => ?_) (hal s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX ℓ l hk)
  obtain ⟨a, cf, A, h⟩ := hu R ω t R₂ τ o hA ρ ℓ hℓ j hj
  obtain ⟨a', cf', A', h'⟩ := hu R' ω' t' R₂' τ' o' hA' ρ ℓ hℓ j hj
  exact rows_congr h h' fun hj' => hc j hj hj' _ _ hX

end

end FlockC3.Reps
