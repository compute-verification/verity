import Proofs.Flock.C3.Reps.AlgL
import Proofs.Flock.C3.Reps.Vals
import Proofs.Flock.VBridge.Family

/-!
# C3's rep decoder: the reps' unit maps, openings and values over a list of V*'s statements

`RepMaps stmt Vs sh Reg Ω C` is what (a)'s lemmas read of G's `VStarFamily`, generic in the list `stmt` so that it fits
`Fin (vstarCount ish)` (at the verifier's own list, `RepMaps.atList` in `AtList.lean`): each part's and each opening's
session and unit, the values read off the history `(R, ω, t)` (`rounds`, `extras`, `alpha`: nothing sent reads the
coins) and those read off V*'s layers (`rep₀`'s `w` and `comb`, `rows`, `acc`), which stay fields.

* `coef`, `ops`, `rep`: G's `VStarFamily.{coef, ops, rep}` at the history's rounds, extras and `alpha`; `coefsAre` and
  `consAre` hold by definition.
* `algebraDecodes`, `accChains`: G's items 1–4 at `rep`, (a)'s `algebraDecodes_of_decodeLH` and `accChains_of_decodeL`
  at the maps, each instance copy 0 (`Family.cp`, `Family.cpO`).
* `ExtrasAt`, `CombFoldsAt`: S's two facts about the decoder (`RepExtrasAt`, `CombFolds`) at `rep`, the obligations
  `proofOfVals_of_src` takes (`proofOfVals`). `rep`'s rounds and extras are the history's and its comb `rep₀`'s,
  which this file leaves fields.
-/

namespace FlockC3.Reps

noncomputable section

open Flock FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.Compiled (RepData Accepted layers AlgebraDecodes AccChains)
open FlockVBridge.OfVStar (ResZeroAtLH)
open FlockVBridge.AccChain (Openings CoefsAre ConsAre ReadsEnds levelLanes)
open FlockVBridge.Family (OpensAtL)

/-- **C3's rep decoder over V*'s statements `stmt`**: G's `VStarFamily`'s unit maps (part `p` of rep `r` is instance
`r` of the statement tagged `.inner p`, opening `j` of level `ℓ` of rep `r` instance `r q + j` of the one tagged
`.recOpen ℓ`) and the values it reads off the history `(R, ω, t)` (`rounds`, `extras`, `alpha`) and the layers (`rep₀`'s
`w` and `comb`, `rows`, `acc`). -/
structure RepMaps {S : ℕ} (stmt : Fin S → VStar.Stmt) (Vs : Fin S → VStmt) (sh : Shape) (Reg Ω C : Type) where
  sess : ℕ → Fin S
  hsess : ∀ p j, (∃ nm pt, stmt j = .inner p nm pt) → sess p = j
  un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv
  hun : ∀ p r, r.val < (Vs (sess p)).nv → (un p r).val = r.val
  sessO : ℕ → Fin S
  hsessO : ∀ ℓ j, (∃ LANES H m, stmt j = .recOpen ℓ LANES H m) → sessO ℓ = j
  unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv
  hunO : ∀ ℓ r j, r.val * (sh.sched.levels[ℓ]!).q + j < (Vs (sessO ℓ)).nv →
    (unO ℓ r j).val = r.val * (sh.sched.levels[ℓ]!).q + j
  rounds : Reg → Ω → List C → Fin 2 → Array (Array F128)
  extras : Reg → Ω → List C → Fin 2 → Array Claim
  rep₀ : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData
  rows : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ℕ → Array (Array F128)
  alpha : Reg → Ω → List C → Fin 2 → ℕ → Array F128
  acc : ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128

namespace RepMaps

variable {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {sh : Shape} {Reg Ω C : Type}
  (M : RepMaps stmt Vs sh Reg Ω C)

/-- Level `ℓ`'s queries of each rep. -/
def q (_ : RepMaps stmt Vs sh Reg Ω C) (ℓ : ℕ) : ℕ := (sh.sched.levels[ℓ]!).q

/-- Each rep's openings. -/
def nOpen : ℕ := ((List.range sh.sched.levels.size).map M.q).sum

/-- The verifier's `coef` for opening `j` of level `ℓ` of rep `r`, at the history's coins and `alpha` (G's `coef`). -/
def coef (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) (ℓ j : ℕ) : Array F128 :=
  match verifierRows sh (M.rounds R ω t r) (M.extras R ω t r) (M.rep₀ l r).comb with
  | .ok (_, _, kappa) => FlockVBridge.coefAt (kappa[ℓ]!) (levelLanes sh ℓ) (eqTable (M.alpha R ω t r ℓ))[j]!
  | .error _ => #[]

/-- **Rep `r`'s openings** (G's `ops`): level `ℓ`'s `q` queries from chain row `r · nOpen`. -/
def ops (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) : Openings :=
  ⟨r.val * M.nOpen, M.q, M.rows l r, M.alpha R ω t r, M.coef R ω t l r⟩

/-- **Rep `r`'s values** (G's `rep`): `rep₀`'s `w` and `comb`, the history's rounds and extras, and the levels'
consistency sums of the opened rows. -/
def rep (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) : RepData :=
  { M.rep₀ l r with
    rounds := M.rounds R ω t r, extras := M.extras R ω t r,
    cons := fun ℓ i => Flock.RecAlgebra.consAt (M.rows l r ℓ) (M.q ℓ) (M.alpha R ω t r ℓ) i }

/-- **The verifier's coefficients are the openings'**, by definition of `coef` (G's `coefsAre`). -/
theorem coefsAre (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) :
    CoefsAre sh (M.rep R ω t l r) (M.ops R ω t l r) := by
  intro v info kappa h ℓ _ j _
  have h' : verifierRows sh (M.rounds R ω t r) (M.extras R ω t r) (M.rep₀ l r).comb = .ok (v, info, kappa) := h
  show M.coef R ω t l r ℓ j = FlockVBridge.coefAt (kappa[ℓ]!) (levelLanes sh ℓ) (eqTable (M.alpha R ω t r ℓ))[j]!
  unfold coef
  rw [h']

/-- **The rep's consistency sums are the opened rows'**, by definition of `rep`. -/
theorem consAre (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits) (r : Fin 2) :
    ConsAre sh (M.rep R ω t l r) (M.ops R ω t l r) :=
  fun _ _ _ _ => rfl

section outer

variable {n : ℕ} {L : Law n} {I : Inner} {Cm : Type} {Reg' Reg₂ : Type}

/-- **G items 1–3 at the decoder**: `algebraDecodes_of_decodeLH` at `sess`, `un` and copy 0. -/
theorem algebraDecodes (M : RepMaps stmt Vs sh Reg L.Ω (Cm × I.Coin))
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) {GS : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : structureOf sh = .ok GS ∧ groups GS (portSizes sh) = .ok gs)
    (hvr : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
          .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
        (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (hres : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAtLH (zs R ω t R₂ (M.sess p)) (layers zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p r)
        (FlockVBridge.Family.cp p r) sh gs p (el (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w)
        (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras
        (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb) :
    AlgebraDecodes zs dg N k Rw sh M.rep :=
  algebraDecodes_of_decodeLH zs dg N k Rw sh M.rep hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hst.2 M.sess
    M.un FlockVBridge.Family.cp hvr hsz hres

/-- **G item 4 at the decoder**: `accChains_of_decodeL` at `ops`, `acc`, `sessO`, `unO` and copy 0, with `coefsAre`
and `consAre`. -/
theorem accChains (M : RepMaps stmt Vs sh Reg L.Ω (Cm × I.Coin))
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size, ∀ j < M.q ℓ,
      OpensAtL (Vs (M.sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (M.sessO ℓ)) (M.unO ℓ r j)
        (FlockVBridge.Family.cpO ℓ r j) (levelLanes sh ℓ) ((M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
        #[(M.acc (layers zs dg N k Rw R ω t R₂ τ)
            ((M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
          (M.acc (layers zs dg N k Rw R ω t R₂ τ)
            ((M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
        ((M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
        (M.acc (layers zs dg N k Rw R ω t R₂ τ) ((M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hrd : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (M.rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (M.acc (layers zs dg N k Rw R ω t R₂ τ))
        (M.ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r)) :
    AccChains zs dg N k Rw sh M.rep :=
  accChains_of_decodeL zs dg N k Rw sh M.rep M.acc M.ops M.sessO M.unO FlockVBridge.Family.cpO hu
    (fun R ω t _ _ _ _ r => M.coefsAre R ω t _ r) (fun R ω t _ _ _ _ r => M.consAre R ω t _ r) hrd

end outer

section obligations

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Obligation: the decoder's extras** (S's `hx` at `rep`, as G's `rep_extras`): at every accepted outer proof, each
rep's extras are the setup `stp`'s at some well-formed link point arrays at the source's `pts`. -/
def ExtrasAt (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockC3.Compose.C3Run (holoAt st hp))
    (stp : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Flock.Setup) : Prop :=
  AtRep zs dg N k Rw sh M.rep fun R ω t Lb i =>
    RepExtrasAt (stp Lb i) ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
      (holoAt st hp).cls.mPts (src R ω t Lb).pts (M.rep R ω t Lb i)

/-- **Obligation: the decoder's comb** (S's `hfold` at `rep`): at every accepted outer proof, each rep's comb is the
lincheck's fold at the setup `stp` (`CombFolds`; G's `combFolds_of`: the comb `combOf` of the rounds, `FoldsOk`). -/
def CombFoldsAt (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (stp : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Flock.Setup) : Prop :=
  AtRep zs dg N k Rw sh M.rep fun R ω t Lb i => FlockVBridge.Family.CombFolds (stp Lb i) sh (M.rep R ω t Lb i)

/-- **S's `ProofOfVals` at the decoder's values**: `proofOfVals_of_src` at `rep`, its `hx` and `hfold` the
obligations. -/
theorem proofOfVals (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockC3.Compose.C3Run (holoAt st hp))
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hset : AtRep zs dg N k Rw sh M.rep fun _ _ _ Lb i =>
      FlockVBridge.Family.SetupRealizes (b Lb i).st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
        (b Lb i).sch (holoAt st hp).cls.ptLocal (holoAt st hp).cls.mPts)
    (hsh : AtRep zs dg N k Rw sh M.rep fun _ _ _ Lb i => FlockVBridge.Family.ShapeIs sh (b Lb i).st (b Lb i).sch)
    (hx : M.ExtrasAt zs dg N k Rw src fun Lb i => (b Lb i).st)
    (hfold : M.CombFoldsAt zs dg N k Rw fun Lb i => (b Lb i).st) :
    ProofOfVals zs dg N k Rw sh M.rep fun R ω t Lb i => (b Lb i).vals sh (M.rep R ω t Lb i) :=
  proofOfVals_of_src (src := src) hset hsh hx hfold

end obligations

end RepMaps

/-! ## R's openings: the `recOpenStacked` statements -/

/-- Values forgotten: a run's coin points do not read them. -/
instance instValUnit : RecAlgebra.Val Unit := ⟨(), fun _ _ => (), fun _ _ => (), fun _ _ => .ok (), fun _ => ()⟩

theorem homUnit {α : Type} [RecAlgebra.Val α] : RecAlgebra.Hom (fun _ : α => ()) :=
  ⟨rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ _ _ => rfl, fun _ => rfl⟩

/-- R's messages, their values forgotten. -/
def skel (sh : Shape) : RecAlgebra.Messages Unit := (RecAlgebra.symbolic sh).map fun _ => ()

theorem map_skel (sh : Shape) (g : RecAlgebra.Lin → F128) :
    ((RecAlgebra.symbolic sh).map g).map (fun _ : F128 => ()) = skel sh := by
  simp only [skel, RecAlgebra.Messages.map, Array.map_map, Option.map_map, RecAlgebra.KV.map, Function.comp_def]

/-- **The Ligerito run's α** at level `ℓ`: R's ring switches, then its Ligerito, on `skel`, at the claims' points `cl`
from coin `k₀` of the rounds `c`. `RecOpening.run`'s and `RecCoins.opening`'s runs both map to it, values forgotten
(`ringSwitch_sim`, `ligerito_sim` at `homUnit`). -/
def runAlpha (sh : Shape) (cl : Array (RecAlgebra.AClaim Unit)) (k₀ : ℕ) (c : Array (Array F128)) (ℓ : ℕ) :
    Array F128 :=
  match RecAlgebra.ringSwitch (skel sh) cl k₀ (RecAlgebra.CoinSource.ofRounds c) with
  | .ok ((_, rs), k₂) =>
    match RecAlgebra.ligerito sh.sched (skel sh) () rs.sws k₂ (RecAlgebra.CoinSource.ofRounds c) with
    | .ok ((_, lg), _) => (lg.conss[ℓ]!).alpha
    | .error _ => #[]
  | .error _ => #[]

/-- **C3's decoder of R's openings** over V*'s statements `stmt` (one rep, `RecCoins.checkStackedLevel`): opening `j` of
level `ℓ` is instance `j` of the statement tagged `.recOpenStacked ℓ`; its row and the chain's rows, off the layers. -/
structure StackedMaps {S : ℕ} (stmt : Fin S → VStar.Stmt) (Vs : Fin S → VStmt) where
  sess : ℕ → Fin S
  hsess : ∀ ℓ j, (∃ LANES H n, stmt j = .recOpenStacked ℓ LANES H n) → sess ℓ = j
  un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv
  hun : ∀ ℓ j, j < (Vs (sess ℓ)).nv → (un ℓ j).val = j
  row : ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → Array F128
  acc : ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128

namespace StackedMaps

variable {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} (M : StackedMaps stmt Vs) (sh : Shape)
  (l : (j : Fin S) → (Vs j).Bits) (kap : Array (Array (Array F128))) (cl : Array (RecAlgebra.AClaim Unit)) (k₀ : ℕ)
  (c : Array (Array F128))

/-- **R's openings** at R's shape `sh`: from chain row 0, level `ℓ`'s `q` rows, the run's α, and the coefficients at the
verifier's `kappa` `kap` (`RecCoins.opening`'s). -/
def ops : Openings :=
  ⟨0, fun ℓ => (sh.sched.levels[ℓ]!).q, fun ℓ => Array.ofFn fun j : Fin (sh.sched.levels[ℓ]!).q => M.row l ℓ j,
    runAlpha sh cl k₀ c, fun ℓ j => RecCoins.coefAt (kap[ℓ]!) (levelLanes sh ℓ) (eqTable (runAlpha sh cl k₀ c ℓ))[j]!⟩

theorem rows_size (ℓ : ℕ) : ((M.ops sh l kap cl k₀ c).rows ℓ).size = (sh.sched.levels[ℓ]!).q := Array.size_ofFn

theorem rows_get {ℓ j : ℕ} (hj : j < (sh.sched.levels[ℓ]!).q) :
    ((M.ops sh l kap cl k₀ c).rows ℓ)[j]! = M.row l ℓ j := by
  rw [getElem!_pos _ j (by rw [rows_size]; exact hj)]
  exact Array.getElem_ofFn ..

/-- `SessionAt`'s coefficient conjunct at `kap := o.kappa`, by definition. -/
theorem coef_eq (ℓ j : ℕ) : (M.ops sh l kap cl k₀ c).coef ℓ j =
    RecCoins.coefAt (kap[ℓ]!) (levelLanes sh ℓ) (eqTable ((M.ops sh l kap cl k₀ c).alpha ℓ))[j]! := rfl

end StackedMaps

end

end FlockC3.Reps
