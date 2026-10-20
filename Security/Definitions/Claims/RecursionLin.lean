import Definitions.Claims.Recursion

/-!
# The recursion's KS with its deviation term linear

`KS` charges `E*`'s inner stage two roots of `Be`: the inner protocol's `B`, read as `Pr[B_I] ≤ k(x)·Pr[Be]`, and, at
each of the inner `E^rep`'s `N_I` expected reruns, a draw whose inner view departs from the inner prover's plurality,
at most `√(k_I·k(x)·Pr[Be])`. Under `ExpectedCR` each root costs half of SHA-512's 256 bits. Here both go:

* **The deviation, linearly** (`Fdev`): guess an inner round `j`, take a draw forked at `j` (the compiled run and
  `E_O`), and if `E_O` gives witnesses there, up to `M` more draws forked at `j` until `E_O` gives witnesses again;
  output the first collision among the two draws' recorded calls (`linkOut`). A draw departs from the plurality at most
  as often as this link succeeds, plus `1/(M+1)`, with no root, and the link costs at most two draws in expectation.
  Its loop reads only what a draw computes (does `E_O` give witnesses), so its declared cost is its work.
* **`B_I` as its own finder** (`Binner`): the inner rounds' forks only, not through `Pr[B_I] ≤ k(x)·Pr[Be]`.

`KSLin`: `Pr[acc ∧ ¬R] ≤ κ + √(γ·Pr[B]) + √(γ_BI·Pr[B_I]) + β·(Pr[Fdev] + 1/(M+1))` at every `M`, with
`β = N_I·k_I` and `γ_BI = (√γ_I + 1)²` (the `+1` an accepted judged draw that the inner view rejects, at most
`Pr[B_I]`). `BitsLin` is its bits under `StrictBirthday` at `B` and `ExpectedCR` at `B_I` and `Fdev`, and `KSBoundLin`
is `KSLin` at the representation V* binds, as `KSBound` is `KS` there.
-/

namespace Claims.Recursion

open Claims

/-! ## The link -/

section Link

variable {U : Type}

open Classical in
/-- The first of `M` draws where `g` holds. -/
noncomputable def firstPick (g : U → Prop) : (M : ℕ) → (Fin M → U) → Option U
  | 0, _ => none
  | M + 1, us => if g (us 0) then some (us 0) else firstPick g M (Fin.tail us)

open Classical in
/-- **The loop's SHA-512 calls**: each draw's `c` until the first where `g` holds. -/
noncomputable def loopCost (g : U → Prop) (c : U → ℕ) : (M : ℕ) → (Fin M → U) → ℕ
  | 0, _ => 0
  | M + 1, us => c (us 0) + if g (us 0) then 0 else loopCost g c M (Fin.tail us)

open Classical in
/-- **The link's SHA-512 calls**: the first draw's, and the loop's where `g` holds at it. -/
noncomputable def linkCost (g : U → Prop) (c : U → ℕ) (M : ℕ) (p : U × (Fin M → U)) : ℕ :=
  c p.1 + if g p.1 then loopCost g c M p.2 else 0

end Link

/-! ## The finders -/

section Finders

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Fdev`'s coins at `x`: the inner round, a tape, a prefix, the first draw and the loop's `M`. -/
abbrev LinCoins (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T)
    (Ω : Type) (x : Msg) (M : ℕ) : Type :=
  Fin (I.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
    (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)

/-- `E_O` gives witnesses at a draw `v`, forked at `j` from the prefix `r`. -/
def linkG (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ) (ω : Ω)
    (r : Fin (D.V.k x) → Coin) (v : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  ((EO D I DO decO).eval H P x ω (fork j r v.1) v.2).1.isSome = true

/-- **The link's output**: the first collision among the calls the first draw's witnesses record and those of the
loop's first draw where `E_O` gives witnesses (the first draw again where none does), both forked at `j` from the
prefix. -/
noncomputable def linkOut (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) {M : ℕ} (s : LinCoins D I DO decO Ω x M) :
    Option (Msg × Msg) :=
  findColl (D.calls x (((EO D I DO decO).eval H P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1.1) s.2.2.2.1.2).1.getD []) ++
    D.calls x (((EO D I DO decO).eval H P x s.2.1
      (fork s.1 s.2.2.1 ((firstPick (linkG H D I DO decO P x s.1 s.2.1 s.2.2.1) M s.2.2.2.2).getD s.2.2.2.1).1)
        ((firstPick (linkG H D I DO decO P x s.1 s.2.1 s.2.2.1) M s.2.2.2.2).getD s.2.2.2.1).2).1.getD []))

/-- **One draw's SHA-512 calls**: the compiled verifier's run, the prover's and the verifier's, and `E_O`'s, as `Be`
counts each of its two. -/
noncomputable def drawCost (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : ℕ :=
  (run H P D.V x ω d.1).pLog.length + (run H P D.V x ω d.1).vLog.length + ((EO D I DO decO).eval H P x ω d.1 d.2).2.1

/-- **`Fdev`, the deviation finder**: `linkOut` on `LinCoins`, at one draw's calls and then the loop's (`linkCost`). -/
noncomputable def Fdev (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ)
    (hk : 0 < I.V.k (xI D x)) : Finder where
  S := LinCoins D I DO decO Ω x M
  ne := ⟨(⟨0, hk⟩, Classical.arbitrary _)⟩
  out := linkOut H D I DO decO P x
  cost s := linkCost (linkG H D I DO decO P x s.1 s.2.1 s.2.2.1)
    (fun v => drawCost H D I DO decO P x s.2.1 (fork s.1 s.2.2.1 v.1, v.2)) M s.2.2.2

/-- `Pr[Fdev^{P*}(x) outputs a collision]` at `M`, `0` for an inner protocol that takes no rounds at `x_I`. -/
noncomputable def PrFdev (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) : ℝ :=
  if hk : 0 < I.V.k (xI D x) then (Fdev H D I DO decO P x M hk).Pr H else 0

/-- **`B_I`, the inner protocol's `B` read through `E_O`, as its own finder**: guess an inner round `j`, fork the
compiled run there, run `E_O` on both runs with fresh coins, and output the first collision among the SHA-512
evaluations the two runs' outer witnesses record, as `Be` does at a guess among all the compiled verifier's rounds. -/
noncomputable def Binner (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < I.V.k (xI D x)) : Finder where
  S := Fin (I.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
    (EO D I DO decO).C x
  ne := ⟨(⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s :=
    findColl (D.calls x (((EO D I DO decO).eval H P x s.2.1 s.2.2.1 s.2.2.2.2.1).1.getD []) ++
      D.calls x (((EO D I DO decO).eval H P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1) s.2.2.2.2.2).1.getD []))
  cost s :=
    (run H P D.V x s.2.1 s.2.2.1).pLog.length + (run H P D.V x s.2.1 s.2.2.1).vLog.length +
      (run H P D.V x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1)).pLog.length +
      (run H P D.V x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1)).vLog.length +
      ((EO D I DO decO).eval H P x s.2.1 s.2.2.1 s.2.2.2.2.1).2.1 +
      ((EO D I DO decO).eval H P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1) s.2.2.2.2.2).2.1

/-- `Pr[B_I^{P*}(x) outputs a collision]`, `0` for an inner protocol that takes no rounds at `x_I`. -/
noncomputable def PrBinner (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  if hk : 0 < I.V.k (xI D x) then (Binner H D I DO decO P x hk).Pr H else 0

end Finders

/-! ## The errors and the claims -/

section Claims

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`Fdev`'s weight**: `β = N_I · k_I(x_I)`, a guess among the inner rounds at each of the inner `E^rep`'s `N_I`
expected reruns. -/
noncomputable def βdev (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ := NI D I x * I.V.k (xI D x)

/-- **`B_I`'s weight**: `γ_BI = (√γ_I(x_I) + 1)²`, the inner protocol's `B` term, and an accepted judged draw whose
inner view rejects. -/
noncomputable def γBI (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ := (Real.sqrt (I.γ (xI D x)) + 1) ^ 2

/-- **The recursion's bits, linearly**: `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ, 128 − ½ log₂(γ_BI · c_Be),
255 − log₂(β · c_Be)) − 2`. `B_I` costs at most `Be`'s `c_Be` per call of the prover's run, and `Fdev` two draws',
`c_Be` too; its tail `β/(M+1)` is below its collision term at `M + 1 ≥ 2^256/(c_Be·t)`. -/
noncomputable def lamLin (D : Deploy Coin) (I : Inner Coin W) (x : Msg) : ℝ :=
  min (min (min (Real.logb 2 (1 / κ D I x)) (255.5 - Real.logb 2 (γ D I x) / 2))
    (128 - Real.logb 2 (γBI D I x * cBe D x) / 2)) (255 - Real.logb 2 (βdev D I x * cBe D x)) - 2

/-- **The recursion: KS, linearly**, at `E*` and every outer decoder that meets the list-decoding spec: on every prover
and at every `M`, `Pr[run = acc ∧ ¬R(x, E*(x))] ≤ κ(x) + √(γ(x) · Pr[B]) + √(γ_BI(x) · Pr[B_I]) +
β(x) · (Pr[Fdev] + 1/(M+1))`. -/
def KSLin [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ),
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤
      κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) + Real.sqrt (γBI D I x * PrBinner H D I DO decO P x) +
        βdev D I x * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1))

/-- **The recursion: bits, linearly**, at every outer decoder that meets the spec: every prover making at most `t ≥ 1`
SHA-512 calls per run (both parties') has `Pr[acc ∧ ¬R] ≤ t · 2^{−λ_lin(x)}`, where `B` obeys the strict birthday bound
and `B_I` and `Fdev`, at every `M`, the linear expected-time bound. -/
def BitsLin [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < D.V.k x, StrictBirthday H (B H D.V P x hk)) →
    (∀ hk : 0 < I.V.k (xI D x), ExpectedCR H (Binner H D I DO decO P x hk)) →
    (∀ (M : ℕ) (hk : 0 < I.V.k (xI D x)), ExpectedCR H (Fdev H D I DO decO P x M hk)) →
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤ t * (2 : ℝ) ^ (-lamLin D I x)

end Claims

section Bound

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **The recursion: KS at the representation V\* binds, linearly**: `KSLin` at the compiled record whose inner view
runs `IVStar` (`D.record Bd.IVStar`), over the bound record (`Bound.deploy`), at every inner and outer decoder that
meets the spec, as `KSBound` is `KS` there. -/
def KSBoundLin [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') : Prop :=
  ∀ decI, decOK Bd.deploy decI → KSLin H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO

end Bound

end Claims.Recursion
