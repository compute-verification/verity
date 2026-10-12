import Proofs.Flock.C3.Opening.SrcOpening
import Proofs.Flock.Soundness.Refine.Opening
import Proofs.Flock.Level3.Fold
import Proofs.Flock.Level3.Assembly

/-!
# n4's claim tie: `ClaimRel` from `claimO` to the run's claims

`openingRead_of_rows`, `openingRead_of_session` and `openingReadAt_src` take the claim tie: claim `i` as
`RecCoins.opening` builds it (`coinClaim`, at rep `i`'s sparse coins), with the value w holds at `rec-claim`, is the
run's registered claim `i` (`ClaimRel`). `ClaimRel`'s sizes are `coinClaim`'s and the shape's (`claimRel_of_ties`), so
the tie is two facts about the run's claims:

* `PointTie`: claim `i`'s point is the one `coinClaim` builds it at, rep `i`'s sparse coins zero-padded to R's `m`
  (`coinPt`). Those coins are the coin server's at the verifier's record, so this says the record's sparse coins are
  the run's.
* `ValueTie`: claim `i`'s value is the one w holds at `rec-claim`.

`openingReadAt_src_ties` is `openingReadAt_src` with the two in place of the claim tie.
-/

namespace FlockC3.Opening

section Ties

open Flock Flock.RecAlgebra

theorem shape_m {m c : ℕ} {sh : Flock.Shape} (h : Flock.RecOpening.shape? m c = some sh) : sh.m = m := by
  unfold Flock.RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hc : c < 1
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      rw [← h]

/-- The point `coinClaim` builds claim `k` at: rep `k`'s sparse coins (its first `nu` rounds, `ζ_p`, `ρ`), zero-padded
to R's `m`. -/
def coinPt (c3 : RecCoins.C3) (sh : Shape) (sc : RecCoins.SparseCoins) : Array F128 :=
  let pt := sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho
  pt ++ Array.replicate (sh.m - pt.size) 0

/-- **What `coinClaim` built**: the coin point has R's `m` coordinates, at least 6; the claim's skip weights are the eq
table of its first 6, and its `x` the rest. -/
theorem coinClaim_ok {c3 : RecCoins.C3} {unit : String} {sh : Shape} {k : ℕ} {a : AClaim Lin}
    (h : coinClaim c3 unit sh k = .ok a) :
    ∃ sc p, RecCoins.sparseCoins c3 unit k = .ok sc ∧ p = coinPt c3 sh sc ∧ p.size = sh.m ∧ 6 ≤ sh.m ∧
      a.skip = eqTable (p.extract 0 6) ∧ a.x = p.extract 6 p.size := by
  unfold coinClaim at h
  obtain ⟨sc, hsc, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · rename_i hb
    cases h
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_lt, gt_iff_lt] at hb
    obtain ⟨h1, h2⟩ := hb
    have h6 : RecSparse.PRE_ROUNDS = 6 := rfl
    refine ⟨sc, _, hsc, rfl, ?_, by omega, rfl, rfl⟩
    simp only [coinPt, Array.size_append, Array.size_replicate] at h1 h2 ⊢
    omega

/-- **The claim tie from its point and value**: the claim `coinClaim` built, at the value `wv`, is a registered claim
whose point is the coin point and whose value is `wv`, at R's `m`. -/
theorem claimRel_of_ties {c3 : RecCoins.C3} {unit : String} {sh : Shape} {k : ℕ} {a : AClaim Lin}
    {ssh : FlockSoundness.Model.SparseShape} {c : FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh} {wv : F128}
    (ha : coinClaim c3 unit sh k = .ok a) (hm : sh.m = ssh.mReg)
    (hpt : ∀ sc, RecCoins.sparseCoins c3 unit k = .ok sc → ∀ j < ssh.mReg,
      c.coord j = FlockSoundness.Refine.fOf (coinPt c3 sh sc)[j]!)
    (hv : c.value = FlockSoundness.Refine.fOf wv) :
    FlockSoundness.Refine.ClaimRel ssh.mReg (⟨wv, a.skip, a.x⟩ : AClaim F128).toClaim c.toClaim := by
  obtain ⟨sc, p, hsc, hp, hsize, hpre, hskip, hx⟩ := coinClaim_ok ha
  have hpt' : ∀ j < ssh.mReg, c.coord j = FlockSoundness.Refine.fOf p[j]! := by
    rw [hp]
    exact hpt sc hsc
  have hsz6 : (p.extract 0 6).size = 6 := by
    rw [Array.size_extract]
    omega
  refine ⟨hv, ?_, ?_, ?_, ?_⟩
  · show a.skip.size = 64
    rw [hskip, FlockLevel3.eqTable_size, hsz6]
    all_goals rfl
  · intro s
    show FlockSoundness.Model.eqAt (fun j : Fin 6 => c.coord j.val) s.val = FlockSoundness.Refine.fOf a.skip[s.val]!
    rw [hskip, FlockSoundness.Refine.eqTable_vecOf _ 6 hsz6 s.val (lt_of_lt_of_eq s.isLt (by norm_num))]
    congr 1
    funext j
    have hj := j.isLt
    rw [hpt' j.val (by omega)]
    show FlockSoundness.Refine.fOf p[j.val]! = FlockSoundness.Refine.fOf (p.extract 0 6)[j.val]!
    rw [FlockLevel3.extract_get! p 0 6 j.val (by omega) (by omega), Nat.zero_add]
  · show a.x.size = ssh.mReg - 6
    rw [hx, Array.size_extract]
    omega
  · intro i
    have hi := i.isLt
    show c.coord (i.val + 6) = FlockSoundness.Refine.fOf a.x[i.val]!
    rw [hx, FlockLevel3.extract_get! p 6 p.size i.val (by omega) le_rfl, hpt' (i.val + 6) (by omega),
      Nat.add_comm i.val 6]

/-- **The point tie**: the run's claim `i` is at the point `coinClaim` builds claim `i` at, rep `i`'s sparse coins
padded to R's `m` (`coinPt`). -/
def PointTie (c3 : RecCoins.C3) (unit : String) (sh : Shape) {ssh : FlockSoundness.Model.SparseShape}
    (c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh) : Prop :=
  ∀ i : Fin 2, ∀ sc, RecCoins.sparseCoins c3 unit i.val = .ok sc → ∀ j < ssh.mReg,
    (c i).coord j = FlockSoundness.Refine.fOf (coinPt c3 sh sc)[j]!

/-- **The value tie**: the run's claim `i` has the value w holds at `rec-claim`. -/
def ValueTie (sh : Shape) (w : Array F128) {ssh : FlockSoundness.Model.SparseShape}
    (c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh) : Prop :=
  ∀ i : Fin 2, (c i).value = FlockSoundness.Refine.fOf w[RecOpening.claimW sh i]!

/-- **(Item 5) The claim tie** at `claimO`, from the point and value ties. -/
theorem claimRel_claimO {c3 : RecCoins.C3} {unit : String} {sh : Shape} {a b : AClaim Lin} {w : Array F128}
    {ssh : FlockSoundness.Model.SparseShape} {c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh}
    (ha : coinClaim c3 unit sh 0 = .ok a) (hb : coinClaim c3 unit sh 1 = .ok b) (hm : sh.m = ssh.mReg)
    (hpt : PointTie c3 unit sh c) (hv : ValueTie sh w c) (i : Fin 2) :
    FlockSoundness.Refine.ClaimRel ssh.mReg (claimO sh w ![a, b] i).toClaim (c i).toClaim := by
  have hi : coinClaim c3 unit sh i.val = .ok (![a, b] i) := by
    fin_cases i
    · first | exact ha | simpa using ha
    · first | exact hb | simpa using hb
  exact claimRel_of_ties hi hm (hpt i) (hv i)

end Ties

section Src

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **(n7's opening) `OpeningReadAt` at the source from three ties**: `openingReadAt_src` with the claim tie from the
point and value ties at every accepted outer proof. -/
theorem openingReadAt_src_ties
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings)
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hpt : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      PointTie (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim)
    (hv : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ValueTie sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      OpenTie sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ))) :
    AtSrc zs dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u :=
  openingReadAt_src hsh unit c3 w ops hsess
    (fun R ω t R₂ τ o hacc _ _ ha hb => claimRel_claimO ha hb (shape_m hsh) (hpt R ω t R₂ τ o hacc)
      (hv R ω t R₂ τ o hacc))
    htie

end Src

end FlockC3.Opening
