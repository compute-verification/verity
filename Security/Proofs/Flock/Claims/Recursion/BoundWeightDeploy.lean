import Proofs.Flock.Claims.Recursion.BoundWeight

/-!
# The deployed bound record at `coverW`, and the route with (P3) discharged

The deployed bound record states its inner claim at `IVStar` at the weight `coverW` of `deployPlainTrim`'s weight, the
least at which (P3) (`Bound.WeightCovers`) holds, at about 4 bits of `λ`.

* **The weight** is `wCover rel = coverW (deployPlainTrim rel).weight`, and **the deployed bound record** `bdCover` is
  `bdDeployed` at it, at every other choice the code side makes. `bdDeployed` keeps `γ`'s weight at `IVStar` a parameter,
  which `KSBoundRoute` quantifies over.
* **(P3) holds at `bdCover`** wherever `DE`'s weight is at most the deployed plain record's (`weightCovers_bdCover`, as
  `coverW` is monotone, `coverW_mono`), at the intended choice `wE = (deployPlainTrim rel).weight` in particular
  (`weightCovers_bdCover_plain`, `weightCovers_coverW` there).
* **The route with (P3) discharged** (`ksBoundRoute_cover`): `ksBoundRoute` at `bdCover`, its premises but (P3) giving
  `KSBoundCover`, `KSBoundDeployed` at `wCover rel`.

At one table of `N₀ = 2^23` and the bounds `k_I = 4402`, `C_DE = 2^33`, `wCover` is about `2^69.65`. At the deployed
shape (one inner table at `m = 35`, `k_log = 24`) the recursion's security level (`Claims.Recursion.lam`) is about 65.2
bits per SHA-512 call at `wCover`, against 69.4 at `w = C_DE`: its third term, `128 − log₂(γ_E·c_Be)/2`, the one that
binds, loses about 4 bits. That is below the 100-bit floor and the `2^-128` target under every weight. Route
2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`, about
165.8), but `KSLin2` at the deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`) only follows from premises
still open: `PartChecksBV`, `PartOpensB`, `DecompB`, the readers' three facts and the coin server's transcript
reading back (`RecDeploy.ReadsBackB`, `RecDeploy.ksLin2_deployed_open`, with the last round's pieces at the parts that
commit the earlier rounds, `RecDeploy.finOpen`). Holo statements are candidate 3's own model, and `BitsLin2` at the dense
statements follows from it (`bitsLin2On_of_ksLin2On`).
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`γ`'s weight at `IVStar` at the deployment**: (P3)'s floor at the deployed plain record's weight. -/
noncomputable def wCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) : Msg → ℝ :=
  coverW (FlockDeploy.deployPlainTrim rel).weight

/-- **The deployed bound record**: `bdDeployed` with `γ`'s weight at `IVStar` at `wCover`, at every other
choice the code side makes. -/
noncomputable def bdCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  bdDeployed rel nextE colsE wE join rowIn final (wCover rel)

/-- **`KSBoundDeployed` at the deployed bound record**: at `γ`'s weight `wCover` at `IVStar`. -/
def KSBoundCover (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) : Prop :=
  KSBoundDeployed rel relO b L nextE colsE wE join rowIn final (wCover rel)

/-- **`coverW` is monotone in `DE`'s weight.** -/
theorem coverW_mono {wE wE' : Msg → ℝ} {x : Msg} (h : wE x ≤ wE' x) : coverW wE x ≤ coverW wE' x := by
  unfold coverW
  have hc : 0 ≤ (1 + 210 * (⌈4 * FlockDeploy.N₀ false x / Real.exp 1⌉₊ : ℝ)) *
      Real.sqrt ((FlockDeploy.V false).k x / 32) := by positivity
  exact pow_le_pow_left₀ (add_nonneg (Real.sqrt_nonneg _) hc) (add_le_add (Real.sqrt_le_sqrt h) le_rfl) 2

/-- **(P3) at the deployed bound record**, wherever `DE`'s weight is at most the deployed plain record's. -/
theorem weightCovers_bdCover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (hwE : ∀ x, wE x ≤ (FlockDeploy.deployPlainTrim rel).weight x) :
    (bdCover rel nextE colsE wE join rowIn final).WeightCovers :=
  (weightCovers_deployed_iff rel nextE colsE wE join rowIn final (wCover rel)).2 fun x =>
    coverW_mono (wE' := (FlockDeploy.deployPlainTrim rel).weight) (hwE x)

/-- **(P3) at the deployed bound record, at the intended `DE` weight**, the deployed plain record's: there `wCover` is
`coverW` at `DE`'s weight (`weightCovers_coverW`). -/
theorem weightCovers_bdCover_plain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    (bdCover rel nextE colsE (FlockDeploy.deployPlainTrim rel).weight join rowIn final).WeightCovers :=
  weightCovers_coverW rel nextE colsE (FlockDeploy.deployPlainTrim rel).weight join rowIn final

/-- **The deployed route with (P3) discharged**: `ksBoundRoute` at the deployed bound record (`bdCover`), wherever
`DE`'s weight is at most the deployed plain record's (the intended choice among them), with (P3) from
`weightCovers_bdCover`. The other premises are `KSBoundRoute`'s, at `bdCover`. Its level (`Claims.Recursion.lam`) is
about 65 bits per SHA-512 call at the deployed shape, below the 100-bit floor and the `2^-128` target (ASK-73). Route
2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records (`RecDeploy.lamLin2_deployed_ge_dep`), but
`KSLin2` at the deployment's dense statements (`KSLin2On` at `RecDeploy.DenseAt`) only follows from premises still open
(`RecDeploy.ksLin2_deployed_open`), Holo statements being candidate 3's own model, and `BitsLin2` at those statements
follows from it (`bitsLin2On_of_ksLin2On`). -/
theorem ksBoundRoute_cover (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (hwE : ∀ x, wE x ≤ (FlockDeploy.deployPlainTrim rel).weight x) :
    let Bd := bdCover rel nextE colsE wE join rowIn final
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.LastBinds FlockDeploy.sha512 →
    Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    KSBoundCover rel relO b L nextE colsE wE join rowIn final := by
  intro _ _ _ hP1 hBinds hLB hF hst hT hL hRRP
  exact ksBoundRoute rel relO b L hRows nextE colsE wE join rowIn final (wCover rel) hP1 hBinds
    (weightCovers_bdCover rel nextE colsE wE join rowIn final hwE) hLB hF hst hT hL hRRP

end RecDeploy
