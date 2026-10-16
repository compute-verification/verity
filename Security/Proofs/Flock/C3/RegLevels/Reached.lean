import Proofs.Flock.C3.RegLevels.PartTies
import Proofs.Flock.C3.RegLevels.AccOut
import Proofs.Flock.C3.Sparse.PubChecked

/-!
# The drawn labels at the outcome, at the checked family

At `C3Family.checked` every accepted outcome is reached and passes candidate 3's checks at its own record: `Accepted`
ties each record to its session's game (`Game.Outcome`), and the checked sessions keep their lists only where a
transcript they write and accept passes `ChecksC3` (`Flock.Guarantees.keepOwn`), so the h-port chain holds there with
no premise of its own.

* `keepOwn_names`: a kept session that accepts nothing has the one-port `freshOwn` list, so it names each port once;
* `checksC3Reached_checked`: `ChecksC3Reached` at the checked family, from the keep (`checks_of_keepOwn`);
* `reachedChecked_checked`, `labels_checked`, `readsOwnAt_checked`: `ReachedChecked` (`reachedChecked_of`), the label
  check at each accepted outcome's own record (`label_of_checked`'s `hlb`) and `ReadsOwnAt` (`readsOwnAt_of`);
* `ownNamed_checked`, `srcHTie_checked`: `srcHReads_of`'s `hnd`, and `SrcHTie` from `HCols` and `HAlgRows`;
* `FlockC3.Sparse.srcSparseTie_of_copies`: `srcSparseTie_of_pubH` with its `h` conjunct from `SrcHTie`
  (`h_of_sessionAt`), at the copies `sparseSessionAt_of_zero` names.
-/

namespace FlockC3.Sparse

open Flock FlockSoundness.Discharge.Recursion
open FlockVBridge.Sparse (SCoins)
open FlockVBridge.Compiled (RepData)

section src

open FlockSoundness FlockSoundness.Discharge
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose FlockC3.Reps

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) →
    C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {c : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SrcSparseTie` at `tiedSrc (sparseOf stp w c)` from V*'s sessions at rep `i`'s copies and `SrcHTie`**:
`srcSparseTie_of_pubH`, its `h` conjunct `h_of_sessionAt` at the copy each session names at rep `i`'s unit (the second
half of `sparseSessionAt_of_zero`'s conclusion). -/
theorem srcSparseTie_of_copies {rng : Flock.Range} (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hU : RepUnitOfAlg zs dg N k Rw (tiedSrc stp) sh rep)
    (hr : AtRep zs dg N k Rw sh rep fun R ω t Lb i => SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (c R ω t Lb i))
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (hs : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        ∃ (s : Fin S) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv), un.val = i.val ∧
          SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
            (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) s un j hj)
    (hH : FlockC3.HPort.SrcHTie zs dg N k Rw stp w) :
    SrcSparseTie zs dg N k Rw (tiedSrc (FlockC3.Sparse.sparseOf stp w c)) :=
  srcSparseTie_of_pubH hu hrng halg hacc hU hr c3 unit w (fun R ω t R₂ τ o hA i => (hs R ω t R₂ τ o hA i).1)
    fun R ω t R₂ τ o hA i _ => FlockC3.HPort.h_of_sessionAt hH hA (hs R ω t R₂ τ o hA i).2

end src

end FlockC3.Sparse

namespace FlockC3.HPort

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

/-- **A kept session that accepts nothing names each port once**: it kept no list (a kept list accepts a transcript
it writes), so its list is `freshOwn`'s one port. -/
theorem keepOwn_names {Q : VStmt} {Reg' : Type} {z : Q.Outer Reg'}
    {P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop}
    (hex : ¬ ∃ t, AcceptsZKR (Flock.Guarantees.keepOwn z P).I (Flock.Guarantees.keepOwn z P).own t) :
    ∀ p ∈ (Flock.Guarantees.keepOwn z P).own, ∀ p' ∈ (Flock.Guarantees.keepOwn z P).own, p.name = p'.name →
      p = p' := by
  intro p hp p' hp' _
  by_cases hk : ∃ R' t, z.Writes R' t ∧ AcceptsZKR z.I z.own t ∧ P z.I z.own t
  · have ho : (Flock.Guarantees.keepOwn z P).own = z.own := ite_eq_left hk
    obtain ⟨-, t, -, ha, -⟩ := hk
    exact (hex ⟨t, by rw [ho]; exact ha⟩).elim
  · have hf : (Flock.Guarantees.keepOwn z P).own = freshOwn Q.c := ite_eq_right hk
    rw [hf] at hp hp'
    unfold freshOwn at hp hp'
    rw [Array.mem_singleton] at hp hp'
    exact hp.trans hp'.symm

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}
  {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
  {rows : Option (Array Flock.Registered.Port)}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

local notation "𝓒" => C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg

/-- **(`ChecksC3Reached`) The checked family's written re-gate**: a checked session that accepts a transcript it writes
kept its own list, so some transcript it writes and accepts passes `ChecksC3` at the family's hold of the inner
transcript and check 5's source (`checks_of_keepOwn` at the keep's own predicate). -/
theorem checksC3Reached_checked :
    ChecksC3Reached 𝓒 (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) rows := by
  rintro R ω t R₂ j ⟨-, tr', -, hacc⟩
  exact FlockC3.PubRows.checks_of_keepOwn (z := Flock.Guarantees.custodied zr hCust h R ω t R₂ j)
    (P := fun J o tr' => (holoInnerAt T CT st hp).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (Flock.Guarantees.innerHeld wr st reg t) (coins (wr t).2) j (bind j) rows J o tr')
    (Pc := fun J o tr' =>
      Flock.Guarantees.ChecksC3 ish (Flock.Guarantees.innerHeld wr st reg t) (coins (wr t).2) j (bind j) rows J o tr')
    (fun _ _ _ hc => hc.2) ⟨tr', hacc⟩

/-- **(`ReachedChecked`) The checked family's accepted outcomes are reached and checked at their own records**:
`reachedChecked_of` at `checksC3Reached_checked`. -/
theorem reachedChecked_checked (dg : ByteArray → ByteArray) (N k Rw : ℕ) : ReachedChecked 𝓒 dg N k Rw :=
  reachedChecked_of (F := 𝓒) checksC3Reached_checked

/-- **(R's `hlb`) The label check at each accepted outcome's own record**, at the checked family: the `checkLabels`
conjunct of `ChecksC3` at `(o j).2.2` (`reachedChecked_checked`), in `label_of_checked`'s form. -/
theorem labels_checked {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω}
    {t : List (Cm × (holoInnerAt T CT st hp).Coin)} {R₂ : Reg₂} {τ o}
    (hA : Accepted (I := holoInnerAt T CT st hp) (𝓒).zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount (𝓒).ish)) :
    ∃ spec sts, FlockSoundness.Discharge.Exec.setupOf
        (FlockSoundness.Discharge.ZkExec.zkInputs ((𝓒).zs R ω t R₂ j).I) (o j).2.2.record = .ok (spec, sts) ∧
      Flock.RecVStar.checkLabels (FlockSoundness.Discharge.Exec.recordDraw (o j).2.2.record)
        ((𝓒).zs R ω t R₂ j).I.count sts = .ok () := by
  obtain ⟨-, _, _, _, _, _, hc3⟩ := reachedChecked_checked dg N k Rw R ω t R₂ τ o hA j
  obtain ⟨spec, sts, hst, -, -, -, hlb, -⟩ := hc3
  exact ⟨spec, sts, hst, hlb⟩

/-- **(`ReadsOwnAt`) The checked family's registered reads are the reads file's entries at the units' own rows**:
`readsOwnAt_of` at `reachedChecked_checked`. -/
theorem readsOwnAt_checked (dg : ByteArray → ByteArray) (N k Rw : ℕ) : ReadsOwnAt 𝓒 dg N k Rw :=
  readsOwnAt_of (F := 𝓒) (reachedChecked_checked dg N k Rw)

/-- **The checked family's sessions name each port once**: an accepting one by its statement check (`names_of_checks`
at `checksAccepted_checked`), any other by its fresh list (`keepOwn_names`). -/
theorem ownNamed_checked : ∀ R ω t R₂ (j : Fin (vstarCount (𝓒).ish)), ∀ p ∈ ((𝓒).zs R ω t R₂ j).own,
    ∀ p' ∈ ((𝓒).zs R ω t R₂ j).own, p.name = p'.name → p = p' := by
  intro R ω t R₂ j
  by_cases hex : ∃ tr, AcceptsZKR ((𝓒).zs R ω t R₂ j).I ((𝓒).zs R ω t R₂ j).own tr
  · exact FlockC3.RegLevels.names_of_checks (F := 𝓒)
      (FlockC3.PubRows.checksAccepted_checked (coins := coins) (rows := rows) (zr := zr) (hCust := hCust) (h := h)) j hex
  · exact keepOwn_names (z := Flock.Guarantees.custodied zr hCust h R ω t R₂ j)
      (P := fun J o tr' => (holoInnerAt T CT st hp).r ≤ t.length ∧
        Flock.Guarantees.ChecksC3 ish (Flock.Guarantees.innerHeld wr st reg t) (coins (wr t).2) j (bind j) rows J o
          tr') hex

/-- **(`SrcHTie`) The h-port chain at the checked family**: `srcHTie_of` at `srcHReads_of`, whose `ChecksAccepted`,
`ReadsOwnAt` and `hnd` the checked family gives (`checksAccepted_checked`, `readsOwnAt_checked`, `ownNamed_checked`),
from the named `HCols` and `HAlgRows`. -/
theorem srcHTie_checked {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hcols : ∀ s : Fin (vstarCount (𝓒).ish), Flock.RecSparse.check ((𝓒).Vs s).c = .ok () →
      Flock.RecSparse.params ((𝓒).Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      HCols ((𝓒).Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))))
    {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) →
      ((j : Fin (vstarCount (𝓒).ish)) → ((𝓒).Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) →
      ((j : Fin (vstarCount (𝓒).ish)) → ((𝓒).Vs j).Bits) → Fin 2 → Array Flock.F128}
    (halg : HAlgRows 𝓒 dg N k Rw stp w) : SrcHTie (𝓒).zs dg N k Rw stp w :=
  srcHTie_of (srcHReads_of (F := 𝓒)
    (FlockC3.PubRows.checksAccepted_checked (coins := coins) (rows := rows) (zr := zr) (hCust := hCust) (h := h))
    (readsOwnAt_checked dg N k Rw) ownNamed_checked hcols halg)

end

end FlockC3.HPort
