import Proofs.Flock.C3.Sparse.SrcSessions
import Proofs.Flock.C3.Sparse.SrcEnd
import Proofs.Flock.C3.RegLevels.SrcMaps

/-!
# The claim tie at `sparseOf`'s tied source

`claim_sparseOf`: at good layers, rep `i`'s sparse session at the words `w` and coins `c` (`SparseSessionAt`), with
what the source reads there (`SparseSrcAt`), plays the rep's stage to the unit's claim at those `w` and `c`
(`sparseClaim`; `sparseTieLAt_of_session`, `sparseTieAt_of_layer` and `playsT_claim_of_tieAt`, which keep the
witnesses, since the claim's value is the word at `wClaim`, which no message of the stage holds), so the tied run of
`sparseOf stp w c` registers that claim (`tieSparse`'s play). `coins_of_session`:
the coins `c` hold the record's sparse coins at `r_k`, `ζ_p` and `ρ_t` (`sparseCs`' layout). `ties_sparseOf`: the
point and value ties there (`ties_of_ends`), where each rep's claim word is R's `rec-claim` row (`ClaimWords`).
`hot_sparseOf` and `hlt_sparseOf`: `srcOK_holo_rs`' `hot` and `hlt` at `tiedSrc (sparseOf stp w c)`, the claim tie
through `claimRel_claimO` (`openingReadAt_src_ties`, `srcLevelTie_of_maps`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (copyVal keyInst)
open FlockVBridge.AccChain (copyVal_regOut eq_of_bitOf)
open FlockVBridge.OfVStar (bitOf_zero)
open FlockVBridge.Sparse (SCoins nw nc nR wClaim verifierRow sparseBase check_sparse inWords_sparse shOf outOf_eq
  tail_iff out_eq laid_sparse sparseClaim nw_le_msum nv_le decode_sparseR)

section Sessions

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **The session's coins are the record's** at `r_k`, `ζ_p` and `ρ_t`: `c.cs` is `sparseCs` of the record's sparse
coins, whose `ζ_c` and H2's take its first 12 (`sparseCoins_size`, `sparseCoins_sizes`). -/
theorem coins_of_session {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits}
    {e : ExecHolo} {c3 : Flock.RecCoins.C3} {unit : String} {i : Fin 2} {w : Array Flock.F128} {c : SCoins}
    (hs : SparseSessionAt zs Lb e c3 unit i w c) (sc : Flock.RecCoins.SparseCoins)
    (hsc : Flock.RecCoins.sparseCoins c3 unit i.val = .ok sc) :
    (∀ j < e.cls.shape.nu, c.cs[12 + j]! = sc.rounds[j]!) ∧
      (∀ j < e.cls.shape.pLog, c.cs[12 + nR e.cls.shape + j]! = sc.zetaP[j]!) ∧
      ∀ j < e.cls.shape.tLog, c.cs[12 + nR e.cls.shape + e.cls.shape.pLog + j]! = sc.rho[j]! := by
  obtain ⟨⟨sc', hsc', hsh, hcs⟩, -⟩ := hs
  rw [Except.ok.inj (hsc'.symm.trans hsc)] at hcs
  have hsz := sparseCoins_size hsc
  obtain ⟨hr, hz, ht⟩ := sparseCoins_sizes hsc
  have hR : nR e.cls.shape = c3.sparse.degrees.length := by
    rw [hsh, FlockVBridge.Sparse.degs_length]
    rfl
  have hP : e.cls.shape.pLog = c3.sparse.pLog := by
    rw [hsh]
    exact FlockVBridge.Sparse.pLog_shOf (unitShape e)
  have hT : e.cls.shape.tLog = c3.sparse.tLog := by rw [hsh]; rfl
  have hnu : e.cls.shape.nu ≤ nR e.cls.shape := Nat.le_add_right _ _
  have h12 : sc.zetaC.size + sc.pre.size = 12 := by
    simp only [sparseCs, Array.size_append] at hsz
    omega
  rw [hcs]
  unfold sparseCs
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun j hj => ?_⟩
  · rw [FlockSoundness.Discharge.ZkExec.append_get!_left _ _ (by simp only [Array.size_append]; omega),
      FlockSoundness.Discharge.ZkExec.append_get!_left _ _ (by simp only [Array.size_append]; omega),
      FlockSoundness.Discharge.ZkExec.append_get!_right _ _ (j := j) (by simp only [Array.size_append]; omega) (by omega)]
  · rw [FlockSoundness.Discharge.ZkExec.append_get!_left _ _ (by simp only [Array.size_append]; omega),
      FlockSoundness.Discharge.ZkExec.append_get!_right _ _ (j := j) (by simp only [Array.size_append]; omega)
        (by omega)]
  · rw [FlockSoundness.Discharge.ZkExec.append_get!_right _ _ (j := j) (by simp only [Array.size_append]; omega)
      (by omega)]

end Sessions

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The tied run of `sparseOf` registers the unit's claim**: its stage is the steps of `w` at `c` by definition, its
reps `stp`'s (`SparseRepTie` at `tiedSrc stp`), so it plays to `sparseClaim` at the session's copy
(`sparseTieLAt_of_session`, `sparseTieAt_of_layer`, `playsT_claim_of_tieAt`), which `tieSparse` registers
(`playOut_eq`). -/
theorem claim_sparseOf {zs : (j : Fin S) → (Vs j).Outer Reg'} {Lb : (j : Fin S) → (Vs j).Bits}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    {c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins}
    {c3 : Flock.RecCoins.C3} {unit : String} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {i : Fin 2}
    (hG : ∀ s, (zs s).Good (Lb s)) (hs : SparseSessionAt zs Lb (holoAt st hp) c3 unit i (w R ω t Lb i) (c R ω t Lb i))
    (hrt : SparseRepTie (holoAt st hp) (tiedSrc stp R ω t Lb) i (w R ω t Lb i) (c R ω t Lb i)) :
    (tiedSrc (sparseOf stp w c) R ω t Lb).claim i =
      sparseClaim (holoAt st hp).cls.shape (w R ω t Lb i) (c R ω t Lb i) := by
  obtain ⟨s, hl⟩ := sparseTieLAt_of_session (u := tiedSrc (sparseOf stp w c) R ω t Lb) hG hs ⟨hrt.1, hrt.2, rfl⟩
  have hpl := playsT_claim_of_tieAt (sparseTieAt_of_layer (zs s) (hG s) hl)
  have hpl' : PlaysT (sparseAt (holoAt st hp) (tieReps (sparseOf stp w c R ω t Lb)) i)
      ((sparseOf stp w c R ω t Lb).sp i) (some (sparseClaim (holoAt st hp).cls.shape (w R ω t Lb i) (c R ω t Lb i))) :=
    hpl
  show (playOut (sparseAt (holoAt st hp) (tieReps (sparseOf stp w c R ω t Lb)) i) ((sparseOf stp w c R ω t Lb).sp i)
    (some ((sparseOf stp w c R ω t Lb).claim i))).getD _ = _
  rw [playOut_eq _ hpl']
  rfl

/-- **Hypothesis: each rep's claim word is R's `rec-claim` row**: the word at the RecSparse unit's `wClaim` in rep
`i`'s session is the one R's opening reads claim `i`'s value at (`ValueTie`). -/
def ClaimWords (ssh : SparseShape) (sh : Flock.Shape) (wS : Fin 2 → Array Flock.F128) (wR : Array Flock.F128) : Prop :=
  ∀ i : Fin 2, fOf (wS i)[wClaim ssh]! = fOf wR[Flock.RecOpening.claimW sh i]!

/-- **The point and value ties at `sparseOf`'s tied run** (`ties_of_ends`): each rep's claim is its unit's
(`claim_sparseOf`), whose coins are the record's (`coins_of_session`), and whose claim word is R's (`ClaimWords`). -/
theorem ties_sparseOf {zs : (j : Fin S) → (Vs j).Outer Reg'} {Lb : (j : Fin S) → (Vs j).Bits}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    {c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins}
    {c3 : Flock.RecCoins.C3} {unit : String} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {sh : Flock.Shape}
    {wR : Array Flock.F128} (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh)
    (hG : ∀ s, (zs s).Good (Lb s))
    (hs : ∀ i : Fin 2, SparseSessionAt zs Lb (holoAt st hp) c3 unit i (w R ω t Lb i) (c R ω t Lb i) ∧
      SparseRepTie (holoAt st hp) (tiedSrc stp R ω t Lb) i (w R ω t Lb i) (c R ω t Lb i))
    (hval : ClaimWords (holoAt st hp).cls.shape sh (w R ω t Lb) wR) :
    FlockC3.Opening.PointTie c3 unit sh (tiedSrc (sparseOf stp w c) R ω t Lb).claim ∧
      FlockC3.Opening.ValueTie sh wR (tiedSrc (sparseOf stp w c) R ω t Lb).claim :=
  ties_of_ends (fun i => claim_sparseOf hG (hs i).1 (hs i).2) (hs 0).1.1.choose_spec.2.1
    (FlockC3.Opening.shape_m hsh) (fun i => coins_of_session (hs i).1) hval

/-- **`srcOK_holo_rs`' `hot` at `sparseOf`'s tied source**: `openingReadAt_src_ties` with the point and value ties
from the sparse sessions (`ties_sparseOf`), and R's opening's tie (`htie`), which reads no claim. -/
theorem hot_sparseOf {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3)
    (wR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array Flock.F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      let Lb := layers zs dg N k Rw R ω t R₂ τ
      FlockC3.Opening.SessionAt (zs R ω t R₂) Lb sh unit (c3 R ω t Lb) (wR R ω t Lb) (ops R ω t Lb) ∧
        (∀ i : Fin 2, SparseSessionAt (zs R ω t R₂) Lb (holoAt st hp) (c3 R ω t Lb) unit i (w R ω t Lb i)
            (c R ω t Lb i) ∧ SparseRepTie (holoAt st hp) (tiedSrc stp R ω t Lb) i (w R ω t Lb i) (c R ω t Lb i)) ∧
        ClaimWords (holoAt st hp).cls.shape sh (w R ω t Lb) (wR R ω t Lb) ∧
        FlockC3.Opening.OpenTie sh (wR R ω t Lb) (ops R ω t Lb) (tiedSrc stp R ω t Lb)) :
    AtSrc zs dg N k Rw (tiedSrc (sparseOf stp w c)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u := by
  have ht := fun R ω t R₂ τ o (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o) =>
    ties_sparseOf (stp := stp) (c := c) hsh (fun s => (hA.1 s).2) (h R ω t R₂ τ o hA).2.1 (h R ω t R₂ τ o hA).2.2.1
  have hor := FlockC3.Opening.openingReadAt_src_ties hsh unit c3 wR ops (fun R ω t R₂ τ o hA => (h R ω t R₂ τ o hA).1)
    (fun R ω t R₂ τ o hA => (ht R ω t R₂ τ o hA).1) (fun R ω t R₂ τ o hA => (ht R ω t R₂ τ o hA).2)
    (fun R ω t R₂ τ o hA => (h R ω t R₂ τ o hA).2.2.2)
  exact fun R ω t R₂ τ o hA => ⟨sh, _, _, (h R ω t R₂ τ o hA).2.2.2, hor R ω t R₂ τ o hA⟩

/-- **`srcOK_holo_rs`' `hlt` at `sparseOf`'s tied source**: `srcLevelTie_of_maps` with its claim tie from the sparse
sessions (`ties_sparseOf`, `claimRel_claimO`). -/
theorem hlt_sparseOf {stmt : Fin S → VStar.Stmt}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {cap₀ : Cap Dig} (M : FlockC3.Reps.StackedMaps stmt Vs) {sh : Flock.Shape}
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3)
    (wR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array Flock.F128)
    (kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array (Array (Array Flock.F128)))
    (e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ByteArray)
    (sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray)
    (e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → ℕ → ByteArray)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      let Lb := layers zs dg N k Rw R ω t R₂ τ
      FlockC3.Opening.SessionAt (zs R ω t R₂) Lb sh unit (c3 R ω t Lb) (wR R ω t Lb)
          (FlockC3.RegLevels.opsAt (FlockC3.RegLevels.readMaps M sh.sched) sh Lb (kap R ω t Lb) (c3 R ω t Lb) unit
            (wR R ω t Lb)) ∧
        (∀ i : Fin 2, SparseSessionAt (zs R ω t R₂) Lb (holoAt st hp) (c3 R ω t Lb) unit i (w R ω t Lb i)
            (c R ω t Lb i) ∧ SparseRepTie (holoAt st hp) (tiedSrc stp R ω t Lb) i (w R ω t Lb i) (c R ω t Lb i)) ∧
        ClaimWords (holoAt st hp).cls.shape sh (w R ω t Lb) (wR R ω t Lb) ∧
        FlockC3.Opening.OpenTieL (zs R ω t R₂) Lb sh (wR R ω t Lb) (tiedSrc (sparseOf stp w c) R ω t Lb) ∧
        FlockC3.RegLevels.MapsLeft M (zs R ω t R₂) Lb sh unit (c3 R ω t Lb) (wR R ω t Lb) (kap R ω t Lb) (e0 R ω t Lb)
          (sb R ω t Lb) (e R ω t Lb) cap₀ (tiedSrc (sparseOf stp w c) R ω t Lb)) :
    FlockC3.RegLevels.SrcLevelTie zs dg N k Rw (tiedSrc (sparseOf stp w c)) cap₀ :=
  FlockC3.RegLevels.srcLevelTie_of_maps M hell hsh unit (sparseOf stp w c) c3 wR kap e0 sb e
    fun R ω t R₂ τ o hA => by
      obtain ⟨hs, hsp, hval, htie, hleft⟩ := h R ω t R₂ τ o hA
      have ht := ties_sparseOf (stp := stp) (c := c) hsh (fun s => (hA.1 s).2) hsp hval
      exact ⟨hs, fun a b ha hb => FlockC3.Opening.claimRel_claimO ha hb (FlockC3.Opening.shape_m hsh) ht.1 ht.2,
        htie, hleft⟩

end

end FlockC3.Sparse
