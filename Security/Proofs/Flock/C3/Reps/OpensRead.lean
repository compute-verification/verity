import Proofs.Flock.C3.Reps.StepsTie
import Proofs.Flock.C3.Reps.Vals

/-!
# The reps' openings, read from the layers' read wires

The last round's message is the openings step `os`, read from every read wire of V*'s layers (`skLeaf`). At strata 0
and 1 the source's openings are the reps' (`repsOfVals`: the values' proof's rows, `openRow`), so the last round needs
`OpensReadAt s` for `s = 0, 1`: two accepted outer proofs whose layers agree on every read wire have the same `os.1 s`.

* `maskOf_congr`: a mask reads only the read wires; `readAgree_of_skLeaf`: at the last round, `MsgsReadC3`'s mask
  equation is agreement on every read wire.
* `OpensOfLayers` (the hypothesis on the values): at every accepted outer proof, rep `i`'s opened rows and siblings
  are `rd i ℓ q` of the layers' bits on the read wires of `W i`, the `row` and `aux` columns of its `RecOpen` copies.
  `SrcOpensOfLayers` is the same in `ValsSrc` terms, at `b`'s `lig` and `sch`; `ValsSrc.vals` keeps them
  (`openRow_vals`: `proofOf` sets only the cap, the messages, the out-of-domain values and `yr`), so
  `opensOfLayers_vals` is by definition.
* `opensReadAt_of_layers`: `OpensReadAt` at any source whose rep openings are the values', from `OpensOfLayers`;
  `opensReadAt_repsOfVals` at the tied source of `nm ∘ repsOfVals` for any `nm` keeping strata 0 and 1 (H's
  `normOs`), and `opensReadAt_src` at G's values.
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Family (maskOf)

local notation "𝔽" => FlockLevel3.GF128

section mask

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **A mask reads only the read wires**: layers agreeing on every read wire agree under `maskOf`, at any `Sk`. -/
theorem maskOf_congr (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) (i : ℕ)
    {Xs Xs' : (j : Fin S) → (Vs j).Bits} (h : ∀ j g, ReadW Vs j g → Xs j g = Xs' j g) :
    maskOf Vs Sk i Xs = maskOf Vs Sk i Xs' :=
  funext fun j => funext fun g => by
    simp only [maskOf]
    split_ifs with hg
    · exact h j g hg.2
    · rfl

/-- **At the last round the leaf mask's equation is agreement on every read wire** (`skLeaf` is every position). -/
theorem readAgree_of_skLeaf {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {r i : ℕ}
    (hi : ¬ i + 1 < r) {Xs Xs' : (j : Fin S) → (Vs j).Bits}
    (h : maskOf Vs (skLeaf Vs own sel r) i Xs = maskOf Vs (skLeaf Vs own sel r) i Xs') :
    ∀ j g, ReadW Vs j g → Xs j g = Xs' j g := fun j g hg => by
  simpa [maskOf, skLeaf, hi, hg] using congrFun (congrFun h j) g

end mask

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The source's openings at stratum `s` are read from every read wire** (H's form): two accepted outer proofs whose
layers agree on every read wire have the same `os.1 s`. -/
def OpensReadAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) (s : ℕ) : Prop :=
  ∀ R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    (∀ j g, ReadW Vs j g → layers zs dg N k Rw R ω t R₂ τ j g = layers zs dg N k Rw R' ω' t' R₂' τ' j g) →
    (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).os.1 s = (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')).os.1 s

/-- **Hypothesis: the values' openings are read off the layers at `W`**: at every accepted outer proof, rep `i`'s
opened row and siblings at level `ℓ`, query `q` are `rd i ℓ q` of the layers' bits on the read wires of `W i`
(`maskOf`), the `row` and `aux` columns of rep `i`'s `RecOpen` copies (`rec_outer.py`, one per opening). -/
def OpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) ℓ q,
    openRow (vals (layers zs dg N k Rw R ω t R₂ τ) i).p.ligerito (vals (layers zs dg N k Rw R ω t R₂ τ) i).sch ℓ q =
      rd i ℓ q (maskOf Vs W i.val (layers zs dg N k Rw R ω t R₂ τ))

/-- **`OpensOfLayers` in `ValsSrc` terms**: the openings of `b`'s Ligerito proof at `b`'s schedule. -/
def SrcOpensOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig)
    (b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) ℓ q,
    openRow (b (layers zs dg N k Rw R ω t R₂ τ) i).lig (b (layers zs dg N k Rw R ω t R₂ τ) i).sch ℓ q =
      rd i ℓ q (maskOf Vs W i.val (layers zs dg N k Rw R ω t R₂ τ))

/-- **`ValsSrc.vals` keeps the openings**: `proofOf` sets the Ligerito proof's cap, messages, out-of-domain values and
`yr` only. -/
theorem openRow_vals (b : ValsSrc) (sh : Flock.Shape) (d : RepData) (ℓ q : ℕ) :
    openRow (b.vals sh d).p.ligerito (b.vals sh d).sch ℓ q = openRow b.lig b.sch ℓ q :=
  rfl

/-- **`ValsSrc.vals` meets `OpensOfLayers` by definition** where `b` meets it in `ValsSrc` terms. -/
theorem opensOfLayers_vals {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig} {sh : Flock.Shape}
    {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData} {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (h : SrcOpensOfLayers zs dg N k Rw W rd b) :
    OpensOfLayers zs dg N k Rw W rd fun Lb i => (b Lb i).vals sh (rep Lb i) := fun R ω t R₂ τ o hA i ℓ q =>
  (openRow_vals (b _ i) sh (rep _ i) ℓ q).trans (h R ω t R₂ τ o hA i ℓ q)

/-- **`OpensReadAt` at the reps from the values' openings read off the layers**, at a source whose openings at reps
0 and 1 are the values' proof's rows. -/
theorem opensReadAt_of_layers {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hv : OpensOfLayers zs dg N k Rw W rd vals)
    (hsrc : ∀ R ω t Lb (i : Fin 2) ℓ q,
      (src R ω t Lb).os.1 i ℓ q = openRow (vals Lb i).p.ligerito (vals Lb i).sch ℓ q)
    (i : Fin 2) : OpensReadAt zs dg N k Rw src i := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX
  funext ℓ q
  rw [hsrc, hsrc, hv R ω t R₂ τ o hA i ℓ q, hv R' ω' t' R₂' τ' o' hA' i ℓ q, maskOf_congr W i.val hX]

/-- **`OpensReadAt` at reps 0 and 1 of the tied source of `nm ∘ repsOfVals`**, at any `nm` keeping the openings at
strata 0 and 1 (H's `normOs` keeps strata 0–2): `tiedSrc` resets only the stages' outputs. -/
theorem opensReadAt_repsOfVals {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals} {nm : C3Run (holoAt st hp) → C3Run (holoAt st hp)}
    (hnm : ∀ u (i : Fin 2), (nm u).os.1 i = u.os.1 i) (hv : OpensOfLayers zs dg N k Rw W rd vals) (i : Fin 2) :
    OpensReadAt zs dg N k Rw (tiedSrc fun R ω t Lb => nm (repsOfVals stp sh rep vals R ω t Lb)) i :=
  opensReadAt_of_layers hv (fun R ω t Lb i ℓ q =>
    (congrFun (congrFun (hnm (repsOfVals stp sh rep vals R ω t Lb) i) ℓ) q).trans (dite_eq_left i.isLt)) i

/-- **`OpensReadAt` at reps 0 and 1 at G's values** (`ValsSrc.vals`), from `b`'s openings read off the layers. -/
theorem opensReadAt_src {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {W : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {rd : Fin 2 → ℕ → ℕ → ((j : Fin S) → (Vs j).Bits) → List 𝔽 × List Dig}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {nm : C3Run (holoAt st hp) → C3Run (holoAt st hp)} (hnm : ∀ u (i : Fin 2), (nm u).os.1 i = u.os.1 i)
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc} (hb : SrcOpensOfLayers zs dg N k Rw W rd b) (i : Fin 2) :
    OpensReadAt zs dg N k Rw
      (tiedSrc fun R ω t Lb => nm (repsOfVals stp sh rep (fun Lb i => (b Lb i).vals sh (rep Lb i)) R ω t Lb)) i :=
  opensReadAt_repsOfVals hnm (opensOfLayers_vals hb) i

end

end FlockC3.Reps
