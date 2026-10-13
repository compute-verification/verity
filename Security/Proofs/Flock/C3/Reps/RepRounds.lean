import Proofs.Flock.C3.Reps.RepCoins
import Proofs.Flock.C3.Reps.TsRead

/-!
# `rep₀`'s rounds off the layers' coin wires, `α` at a stage, and `hOfW`

Premise 2 (rec-lean's round 9): `RepMaps.rep₀` is a function of the layers, its rounds read off coin wires `W`
(`RoundsOnWires`).

* `repRoundsHeld_of_wires`: `RepRoundsHeld` at `M.rep`. At an accepted outer proof each session's layer is `Good`, so it
  carries the registrant's bit at every registered read in a unit's io (`RegAgree`), which the coin wires are
  (`CoinWiresRegistered`); so the layers give the rounds the registrant's layer `regLayer` gives, which are the coins the
  verifier holds (`CoinWiresHeld`). No port holds a round's coins as such: check 5 registers only values of the coins
  (`rec-v`, `rec-coef`, `rec-dirs`, `rec-sparse-v`), so `CoinWiresHeld` is that executable check together with the
  decoder's reading of the rounds back off those values.
* `alphaRead_of_rounds`: `AlphaRead` where `α` is `runAlpha` of `rep₀`'s rounds (`AlphaOfRounds`), from the rounds' read
  and stage `s` reading the coin wires of each rep whose sums its keys read (`CoinWiresReadAt`, in `skOf`'s form).
* `hOfW_valOf`, `hOfW_read`: at a holo shape `hOfW sh w`'s entry `l` is `valOf` at `.h l`, so `WCombRead` at the `.h l`
  keys reads it.
-/

namespace FlockC3.Reps

noncomputable section

open Flock

/-- **`hOfW`'s entries are `valOf`'s at `.h`** at a holo shape. -/
theorem hOfW_valOf {sh : Flock.Shape} (hh : sh.holo = true) (w comb : Array F128) (c : ℕ → ℕ → F128) {l : ℕ}
    (hl : l < 64) : (Flock.RecAlgebra.hOfW sh w)[l]! = Flock.RecAlgebra.valOf sh w comb c (.h l) := by
  rw [Flock.RecAlgebra.hOfW_get sh w hl, Flock.RecAlgebra.hOf, ite_eq_left hh]
  rfl

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockC3.Compose
open FlockSoundness.Assumptions (heldCoins)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {W : Fin 2 → ℕ → (j : Fin S) → Fin (FlockSoundness.Discharge.Layout.keyN (Vs j).c (Vs j).hU (Vs j).nv) → Prop}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The registrant's layer** at a history: each wire's bit in the value its session's registrant registers at the
wire's commit string. -/
def regLayer (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) : (j : Fin S) → (Vs j).Bits :=
  fun j g => (zs R ω t R₂ j).lay.bit ((zs R ω t R₂ j).rd.wv ((zs R ω t R₂ j).lay.pos g)) g

/-- **Hypothesis: `rep₀` reads its rounds off the coin wires `W`** (premise 2, a fact of L's decoder's definition): layers
agreeing on rep `ρ`'s round-`q` coin wires give it the same round `q`. -/
def RoundsOnWires (M : RepMaps stmt Vs sh)
    (W : Fin 2 → ℕ → (j : Fin S) → Fin (FlockSoundness.Discharge.Layout.keyN (Vs j).c (Vs j).hU (Vs j).nv) → Prop) :
    Prop :=
  ∀ ρ q (Lb Lb' : (j : Fin S) → (Vs j).Bits), (∀ j g, W ρ q j g → Lb j g = Lb' j g) →
    (M.rep₀ Lb ρ).rounds[q]? = (M.rep₀ Lb' ρ).rounds[q]?

/-- **Hypothesis: the coin wires are registered reads** (S's io family): at every accepted outer proof each coin wire is
read (`ReadW`, in a unit's io), and its commit string is one its session's registrant registers. -/
def CoinWiresRegistered (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (W : Fin 2 → ℕ → (j : Fin S) → Fin (FlockSoundness.Discharge.Layout.keyN (Vs j).c (Vs j).hU (Vs j).nv) → Prop) :
    Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ q j g, W ρ q j g →
    ReadW Vs j g ∧ (zs R ω t R₂ j).rd.isRead ((zs R ω t R₂ j).lay.pos g)

/-- **Hypothesis: the verifier's coins fill the coin wires** (the executable fact): at every accepted outer proof the
registrant's layer gives rep `ρ`'s round `q` the coins the verifier holds at round `ix ρ q` of stream `s ρ`. -/
def CoinWiresHeld (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (M : RepMaps stmt Vs sh) (wr : List (Cm × (ℐ).Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits) (s : Fin 2 → String)
    (ix : Fin 2 → ℕ → ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ ρ q c,
    heldCoins (held (wr t)) (s ρ) (ix ρ q) = some c → (M.rep₀ (regLayer zs R ω t R₂) ρ).rounds[q]? = some c

/-- **An accepted outer proof's layers are the registrant's on the coin wires** (`Good`'s `RegAgree`). -/
theorem layers_coinWires (hreg : CoinWiresRegistered zs dg N k Rw W) {R ω t R₂ τ o}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) (ρ : Fin 2) (q : ℕ) :
    ∀ j g, W ρ q j g → layers zs dg N k Rw R ω t R₂ τ j g = regLayer zs R ω t R₂ j g := by
  intro j g hg
  obtain ⟨⟨u, hu⟩, hr⟩ := hreg R ω t R₂ τ o hA ρ q j g hg
  exact (hA.1 j).2.2 u g hu hr

/-- **`RepRoundsHeld` from the coin wires**: the layers' rounds are the registrant's layer's (`layers_coinWires`), which
are the verifier's held coins. -/
theorem repRoundsHeld_of_wires {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String Flock.RecVStar.InnerCommits} {s : Fin 2 → String} {ix : Fin 2 → ℕ → ℕ}
    (M : RepMaps stmt Vs sh) (hW : RoundsOnWires M W) (hreg : CoinWiresRegistered zs dg N k Rw W)
    (hheld : CoinWiresHeld zs dg N k Rw M wr held s ix) : RepRoundsHeld zs dg N k Rw sh M.rep wr held s ix := by
  intro R ω t R₂ τ o hA ρ _ _ q c hc
  exact (hW ρ q (layers zs dg N k Rw R ω t R₂ τ) (regLayer zs R ω t R₂) (layers_coinWires hreg hA ρ q)).trans
    (hheld R ω t R₂ τ o hA ρ q c hc)

/-- **Hypothesis: `α` is the run's at `rep₀`'s rounds** (`runAlpha`, as `StackedMaps.ops` reads it): rep `ρ`'s run from
coin `k₀ ρ`, at the claims' points `cl ρ c` its rounds `c` give. -/
def AlphaOfRounds (M : RepMaps stmt Vs sh) (cl : Fin 2 → Array (Array F128) → Array (Flock.RecAlgebra.AClaim Unit))
    (k₀ : Fin 2 → ℕ) : Prop :=
  ∀ Lb ρ ℓ, M.alpha Lb ρ ℓ = runAlpha sh (cl ρ (M.rep₀ Lb ρ).rounds) (k₀ ρ) (M.rep₀ Lb ρ).rounds ℓ

/-- **Hypothesis: stage `i` reads the coin wires of the reps whose sums it reads** (`skOf`'s form): wherever rep `ρ`'s
stage-`i` keys read a consistency sum, each coin wire of rep `ρ`'s rounds is a read wire of stage `i`'s commit strings. -/
def CoinWiresReadAt (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (W : Fin 2 → ℕ → (j : Fin S) → Fin (FlockSoundness.Discharge.Layout.keyN (Vs j).c (Vs j).hU (Vs j).nv) → Prop)
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ i (ρ : Fin 2) ℓ l, K i ρ (.cons ℓ l) → ∀ q j g, W ρ q j g →
    posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ skOf Vs own sel i j ∧ ReadW Vs j g

/-- **`AlphaRead` from the rounds' read**: layers agreeing under stage `i`'s mask agree on rep `ρ`'s coin wires, so on its
rounds (`RoundsOnWires`), so on `runAlpha` of them. -/
theorem alphaRead_of_rounds (M : RepMaps stmt Vs sh) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    {cl : Fin 2 → Array (Array F128) → Array (Flock.RecAlgebra.AClaim Unit)} {k₀ : Fin 2 → ℕ}
    (hα : AlphaOfRounds M cl k₀) (hW : RoundsOnWires M W) (hrd : CoinWiresReadAt own sel W K) :
    AlphaRead (st := st) (hp := hp) zs dg N k Rw own sel M K := by
  intro i ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' _ _ hX ℓ l hk
  have hr : (M.rep₀ (layers zs dg N k Rw R ω t R₂ τ) ρ).rounds =
      (M.rep₀ (layers zs dg N k Rw R' ω' t' R₂' τ') ρ).rounds :=
    Array.ext_getElem? fun q => hW ρ q _ _ fun j g hg => by
      have e := congrFun (congrFun hX j) g
      simp only [maskOf] at e
      rwa [ite_eq_left (hrd i ρ ℓ l hk q j g hg), ite_eq_left (hrd i ρ ℓ l hk q j g hg)] at e
  rw [hα, hα, hr]

/-- **`WCombRead` reads `hOfW`** at a holo shape: at two accepted outer proofs whose layers agree under stage `i`'s
mask, rep `ρ`'s `hOfW` agrees at each `l` whose `.h l` key its stage-`i` keys hold. -/
theorem hOfW_read (hh : sh.holo = true) (M : RepMaps stmt Vs sh) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (i : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel i fun _ _ _ Lb _ _ _ Lb' =>
      ∀ l < 64, K i ρ (.h l) →
        (Flock.RecAlgebra.hOfW sh (M.rep Lb ρ).w)[l]! = (Flock.RecAlgebra.hOfW sh (M.rep Lb' ρ).w)[l]! := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX l hl hk
  rw [hOfW_valOf hh _ (M.rep₀ (layers zs dg N k Rw R ω t R₂ τ) ρ).comb (fun _ _ => 0) hl,
    hOfW_valOf hh _ (M.rep₀ (layers zs dg N k Rw R' ω' t' R₂' τ') ρ).comb (fun _ _ => 0) hl]
  exact hwc i ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX _ hk fun _ _ h => by cases h

end

end FlockC3.Reps
