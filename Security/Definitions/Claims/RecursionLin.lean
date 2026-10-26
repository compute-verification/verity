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

/-! ## Flock's KS with no root

`Inner.KSAbort` charges the inner Flock `√(γ·Pr[B])`, where `γ = 16·(√(2kC) + k)²` gathers, under one root, every
fork event of its table bound: each cap's (its positions opened two ways) and level 0's (two accepted runs that open a
position to two columns). `KSFlockLin` charges each to a finder of its own, with no root:

* **the caps' link** (`FC`): a uniform cap position `q`, a run, and if it accepts and opens `q`, up to `M` more runs
  forked at `q`'s round until one accepts and opens `q`; the first collision among the two runs' verifier calls. Its
  weight is the caps' positions (`Caps.width`) and it costs at most two runs in expectation.
* **level 0's conflict** (`F0`): a run, and if it accepts, `E^rep`'s `k` waits of at most `R` runs each; the first
  collision among the accepted runs' verifier calls, `k + 1` runs in expectation.

The one-run events stay where they were, linear under `B` (weight `k(x)`, the verifier's rounds). `Caps` is which cap
positions a run's coins query, which `Deploy.cols` doesn't give (it is level 0's columns only); the claim is at a
reader, and its loop reads only what a run shows: whether it accepts, and its coins.

`KSLin2` is `KSLin` at the bound record with the inner stage reading `KSFlockLin` at `DE` in `Inner.KSAbort`'s place:
`√(γ_BI·Pr[B_I])` becomes `(4k_I + 1)·Pr[B_I]`, and the inner Flock's caps and level 0 are the composite's finders over
`E_O`'s draws at `IVStar` (`FCI`, `F0I`), whose loops read whether a draw's inner view accepts and the draw's inner
coins. A draw where the inner view departs from the prover's plurality is `Fdev`'s, at `FCI`'s two and `F0I`'s `k + 1`
draws as well as the inner `E^rep`'s `N_I` (`βdev2`). An accepted inner run's calls are the outer witnesses' recorded
ones, its setup's at the inner statement and its own on the inner messages at the draw's coins, which no witness need
record, so `B_I`, `FCI`, `F0I` and `Fdev` search all three at each draw's own coins (`Deploy.callsAt`), each charged
the setup's `|S(x_I)|` calls once (`Finder.charge`, `setupLen`). A V\* part's check is accepted where its recorded calls
and the compiled verifier's own evaluations at its V\* statement (check 5's row commitments) don't collide, so `B_I` and
`Fdev`, which read what V*'s parts accept, search those too (`drawCalls`: `BinnerV`, `FdevV`), each charged `v` more
calls per draw. `BitsLin2` is its bits.
-/

section FlockLin

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Which positions a run opens at Flock's caps**: the positions of the caps past level 0 at `x` (`width x` of them),
the verifier's round each position's cap is fixed by (`round`: coins before it are shared by a fork there), and whether
the verifier's coins query a position (`opens`, at the run's coins, `coinSeq`). -/
structure Caps (Coin : Type) where
  width : Msg → ℕ
  round : (x : Msg) → Fin (width x) → ℕ
  opens : (x : Msg) → Fin (width x) → (ℕ → Coin) → Bool

/-- One run's SHA-512 calls as it may abort, both parties'. -/
def abortCost {Ω A : Type} (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ω : Ω) (d : (Fin (V.k x) → Coin) × A) : ℕ :=
  (abortRun H P V x keep ω d).pLog.length + (abortRun H P V x keep ω d).vLog.length

/-- The run as it may abort at `d` accepts. -/
def accAt {Ω A : Type} (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ω : Ω) (d : (Fin (V.k x) → Coin) × A) : Prop :=
  (abortRun H P V x keep ω d).out = .acc

/-- The run as it may abort at `d` accepts and its coins query the cap position `q`. -/
def opensAt {Ω A : Type} (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (q : Fin (Cp.width x)) (ω : Ω) (d : (Fin (V.k x) → Coin) × A) :
    Prop :=
  (abortRun H P V x keep ω d).out = .acc ∧ Cp.opens x q (coinSeq d.1) = true

/-- `v`'s coins forked at `q`'s round from `d₀`'s, with `v`'s abort coins. -/
def capFork (V : Verifier Coin) (Cp : Caps Coin) {A : Type} (x : Msg) (q : Fin (Cp.width x))
    (d₀ v : (Fin (V.k x) → Coin) × A) : (Fin (V.k x) → Coin) × A :=
  (fork (Cp.round x q) d₀.1 v.1, v.2)

/-- `FC`'s coins at `x`: a cap position, a tape, the first run's coins with its abort coins, and the loop's `M`. -/
abbrev CapCoins (V : Verifier Coin) (Cp : Caps Coin) (Ω A : Type) (x : Msg) (M : ℕ) : Type :=
  Fin (Cp.width x) × Ω × ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A)

open Classical in
/-- **`FC`, the caps' link finder**, at runs that may abort: a cap position `q`, a tape and a run; if the run accepts
and its coins query `q`, the loop over `M` continuations forked at `q`'s round from it, stopping at the first that
accepts and queries `q`, and the first collision among the verifier's calls in the run and that one (the run again where
none does). Its cost is the run's calls, both parties', and the loop's. -/
noncomputable def FC {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (V : Verifier Coin) (Cp : Caps Coin) (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ)
    (hW : 0 < Cp.width x) : Finder where
  S := CapCoins V Cp Ω A x M
  ne := ⟨(⟨0, hW⟩, Classical.arbitrary _)⟩
  out s :=
    if opensAt H V Cp P x keep s.1 s.2.1 s.2.2.1 then
      findColl ((abortRun H P V x keep s.2.1 s.2.2.1).vLog ++
        (abortRun H P V x keep s.2.1 (capFork V Cp x s.1 s.2.2.1
          ((firstPick (fun v => opensAt H V Cp P x keep s.1 s.2.1 (capFork V Cp x s.1 s.2.2.1 v)) M
            s.2.2.2).getD s.2.2.1))).vLog)
    else none
  cost s :=
    abortCost H V P x keep s.2.1 s.2.2.1 +
      if opensAt H V Cp P x keep s.1 s.2.1 s.2.2.1 then
        loopCost (fun v => opensAt H V Cp P x keep s.1 s.2.1 (capFork V Cp x s.1 s.2.2.1 v))
          (fun v => abortCost H V P x keep s.2.1 (capFork V Cp x s.1 s.2.2.1 v)) M s.2.2.2
      else 0

/-- `Pr[FC^{P*}(x) outputs a collision]` at `M`, `0` where `x` has no cap position. -/
noncomputable def PrFC {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (V : Verifier Coin) (Cp : Caps Coin) (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool)
    (M : ℕ) : ℝ :=
  if hW : 0 < Cp.width x then (FC H V Cp P x keep M hW).Pr H else 0

/-- `F0`'s coins at `x`: a tape, the first run's coins with its abort coins, and `k` waits of `R` runs each. -/
abbrev ConflictCoins (V : Verifier Coin) (Ω A : Type) (x : Msg) (k R : ℕ) : Type :=
  Ω × ((Fin (V.k x) → Coin) × A) × (Fin k → Fin R → (Fin (V.k x) → Coin) × A)

open Classical in
/-- **`F0`, level 0's conflict finder**, at runs that may abort: a tape and a run; if the run accepts, `k` waits of at
most `R` runs each, each stopping at its first accepted run, and the first collision among the verifier's calls in the
run and the waits' accepted runs. Its cost is the run's calls, both parties', and the waits'. -/
noncomputable def F0 {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (V : Verifier Coin) (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (k R : ℕ) :
    Finder where
  S := ConflictCoins V Ω A x k R
  ne := ⟨Classical.arbitrary _⟩
  out s :=
    if accAt H V P x keep s.1 s.2.1 then
      findColl ((abortRun H P V x keep s.1 s.2.1).vLog ++ (List.finRange k).flatMap fun i =>
        ((firstPick (accAt H V P x keep s.1) R (s.2.2 i)).map fun d => (abortRun H P V x keep s.1 d).vLog).getD [])
    else none
  cost s :=
    abortCost H V P x keep s.1 s.2.1 +
      if accAt H V P x keep s.1 s.2.1 then
        ∑ i, loopCost (accAt H V P x keep s.1) (abortCost H V P x keep s.1) R (s.2.2 i)
      else 0

/-- `Pr[F0^{P*}(x) outputs a collision]` at `k` waits of `R` runs. -/
noncomputable def PrF0 {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (V : Verifier Coin) (P : Prover Ω) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (k R : ℕ) : ℝ :=
  (F0 H V P x keep k R).Pr H

variable {Col T : Type} [DecidableEq Col]

/-- **Flock: KS against provers that may abort, with no root**, at `E^rep`, every decoder that meets the list-decoding
spec and every `M`, `R`: `Pr[run = acc ∧ ¬R(x, E^rep(x))] ≤ κ(x) + 4·k(x)·Pr[B] + 4·W(x)·(Pr[FC] + 1/(M+1)) +
4·(Pr[F0] + (k + 1)/(R+1))`, with `k(x)` the verifier's rounds, `W(x)` the caps' positions and `k` `E^rep`'s accepted
reruns per wait, `E^rep`, `B`, `FC` and `F0` seeing each run as it aborts. It is `Inner.KSAbort` with `√(γ·Pr[B])`
split into its events, each linear: the one-run events under `B`, the caps' under `FC` and level 0's under `F0`. -/
def KSFlockLin [Zero Col] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (Cp : Caps Coin) : Prop :=
  ∀ decode, decOK D decode → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A]
    [Nonempty A] (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool) (M R : ℕ),
    (Inner.ofFlock D decode).PrBadAbort H x P keep ≤
      Flock.κ D x + 4 * D.V.k x * PrBAbort H D.V P x keep +
        4 * Cp.width x * (PrFC H D.V Cp P x keep M + 1 / ((M : ℝ) + 1)) +
        4 * (PrF0 H D.V P x keep (D.k x) R + ((D.k x : ℝ) + 1) / ((R : ℝ) + 1))

end FlockLin

/-! ## The recursion's KS with the inner Flock's roots linear -/

section Composite

variable {Coin W Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **The calls a finder searches at a draw**: at the statement `x`, the draw's coins `r` and its outer witnesses `ws`,
the calls the witnesses record (`Deploy.calls`), then the inner verifier's setup's at the inner statement
(`Deploy.innerSetup`) and its calls on the inner messages the witnesses give at the draw's coins (`Deploy.innerCalls`),
each with `H`'s answer. -/
def Deploy.callsAt (H : Msg → Msg) (D : Deploy Coin) (x : Msg) (r : Fin (D.V.k x) → Coin) (ws : List Flock.Wit) :
    List (Msg × Msg) :=
  D.calls x ws ++ (D.innerSetup (xI D x) ++ D.innerCalls H x (coinSeq r) ws).map fun u => (u, H u)

/-- The outer witnesses `E_O` gives at a draw, none where it gives up. -/
noncomputable def drawWits (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : List Flock.Wit :=
  ((EO D I DO decO).eval H P x ω d.1 d.2).1.getD []

/-- **A draw's SHA-512 evaluations**: those the outer witnesses `E_O` gives at the draw record and the inner verifier's
own at the draw's coins (`Deploy.callsAt`), then the compiled verifier's own at the run's V\* statements (`setupCalls`
at `runStmts`: its setup's there, check 5's row commitments among them), which no witness records. -/
noncomputable def drawCalls (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : List (Msg × Msg) :=
  D.callsAt H x d.1 (drawWits H D I DO decO P x ω d) ++ setupCalls H D (runStmts H D I P x ω d.1)

/-- `B_I`'s coins at `x`: the inner round, a tape, the shared coins, the fork's continuation and `E_O`'s coins at both
runs. -/
abbrev BinCoins (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Ω : Type) (x : Msg) : Type :=
  Fin (I.V.k (xI D x)) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
    (EO D I DO decO).C x

/-- **`B_I` over the draws' evaluations**: `Binner`'s fork, extractions and cost, and the first collision among the
two draws' evaluations (`drawCalls`): the calls their outer witnesses record and the compiled verifier's own at their
V\* statements. -/
noncomputable def BinnerV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < I.V.k (xI D x)) : Finder :=
  { Binner H D I DO decO P x hk with
    out := fun s : BinCoins D I DO decO Ω x =>
      findColl (drawCalls H D I DO decO P x s.2.1 (s.2.2.1, s.2.2.2.2.1) ++
        drawCalls H D I DO decO P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1, s.2.2.2.2.2)) }

/-- `Pr[B_I^{P*}(x) outputs a collision]` over the draws' evaluations, `0` for an inner protocol that takes no rounds
at `x_I`. -/
noncomputable def PrBinnerV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  if hk : 0 < I.V.k (xI D x) then (BinnerV H D I DO decO P x hk).Pr H else 0

/-- **The link's output over the draws' evaluations**: `linkOut`, with each of its two draws' evaluations
(`drawCalls`) in place of the calls their witnesses record. -/
noncomputable def linkOutV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg) {M : ℕ}
    (s : LinCoins D I DO decO Ω x M) : Option (Msg × Msg) :=
  findColl (drawCalls H D I DO decO P x s.2.1 (fork s.1 s.2.2.1 s.2.2.2.1.1, s.2.2.2.1.2) ++
    drawCalls H D I DO decO P x s.2.1
      (fork s.1 s.2.2.1 ((firstPick (linkG H D I DO decO P x s.1 s.2.1 s.2.2.1) M s.2.2.2.2).getD s.2.2.2.1).1,
        ((firstPick (linkG H D I DO decO P x s.1 s.2.1 s.2.2.1) M s.2.2.2.2).getD s.2.2.2.1).2))

/-- **`Fdev` over the draws' evaluations**: `Fdev`'s coins and cost, with output `linkOutV`. -/
noncomputable def FdevV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ)
    (hk : 0 < I.V.k (xI D x)) : Finder :=
  { Fdev H D I DO decO P x M hk with out := linkOutV H D I DO decO P x }

/-- `Pr[Fdev^{P*}(x) outputs a collision]` over the draws' evaluations at `M`, `0` for an inner protocol that takes no
rounds at `x_I`. -/
noncomputable def PrFdevV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) : ℝ :=
  if hk : 0 < I.V.k (xI D x) then (FdevV H D I DO decO P x M hk).Pr H else 0

/-- **A draw's inner view accepts**: `E_O` gives witnesses at the draw and the inner view they give accepts (`dec`
rejects where `E_O` gives up). -/
def accAtI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  (dec H D x d.1 ((EO D I DO decO).eval H P x ω d.1 d.2).1).out = .acc

/-- A draw's inner view accepts and its inner coins, the draw's first `k_I`, query the inner statement's cap position
`q`. -/
def opensAtI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} (P : Prover Ω) (x : Msg)
    (q : Fin (Cp.width (xI D x))) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  accAtI H D I DO decO P x ω d ∧
    Cp.opens (xI D x) q (coinSeq fun i : Fin (I.V.k (xI D x)) => coinSeq d.1 i.val) = true

/-- `v`'s draw forked at `q`'s inner round from `d₀`'s (at most the inner rounds, which come first), with `v`'s `E_O`
coins. -/
def capForkI (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T')
    (Cp : Caps Coin) (x : Msg) (q : Fin (Cp.width (xI D x))) (d₀ v : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x :=
  (fork (min (Cp.round (xI D x) q) (I.V.k (xI D x))) d₀.1 v.1, v.2)

/-- `FCI`'s coins at `x`: an inner cap position, a tape, the first draw and the loop's `M`. -/
abbrev CapCoinsI (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) (Ω : Type) (x : Msg) (M : ℕ) : Type :=
  Fin (Cp.width (xI D x)) × Ω × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
    (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)

open Classical in
/-- **`FCI`, the inner caps' link over `E_O`'s draws**: `FC` at the composite. An inner cap position `q`, a tape and a
draw (the compiled run and `E_O`); if its inner view accepts and its inner coins query `q`, the loop over `M` draws
forked at `q`'s inner round from it, stopping at the first whose inner view accepts and whose inner coins query `q`, and
the first collision among the calls the two draws' outer witnesses record (the first draw again where none does). Its
cost is the draws' (`drawCost`). -/
noncomputable def FCI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) (hW : 0 < Cp.width (xI D x)) : Finder where
  S := CapCoinsI D I DO decO Cp Ω x M
  ne := ⟨(⟨0, hW⟩, Classical.arbitrary _)⟩
  out s :=
    if opensAtI H D I DO decO Cp P x s.1 s.2.1 s.2.2.1 then
      findColl (D.callsAt H x s.2.2.1.1 (drawWits H D I DO decO P x s.2.1 s.2.2.1) ++
        D.callsAt H x (capForkI D I DO decO Cp x s.1 s.2.2.1
          ((firstPick (fun v => opensAtI H D I DO decO Cp P x s.1 s.2.1 (capForkI D I DO decO Cp x s.1 s.2.2.1 v))
            M s.2.2.2).getD s.2.2.1)).1
          (drawWits H D I DO decO P x s.2.1 (capForkI D I DO decO Cp x s.1 s.2.2.1
            ((firstPick (fun v => opensAtI H D I DO decO Cp P x s.1 s.2.1 (capForkI D I DO decO Cp x s.1 s.2.2.1 v))
              M s.2.2.2).getD s.2.2.1))))
    else none
  cost s :=
    drawCost H D I DO decO P x s.2.1 s.2.2.1 +
      if opensAtI H D I DO decO Cp P x s.1 s.2.1 s.2.2.1 then
        loopCost (fun v => opensAtI H D I DO decO Cp P x s.1 s.2.1 (capForkI D I DO decO Cp x s.1 s.2.2.1 v))
          (fun v => drawCost H D I DO decO P x s.2.1 (capForkI D I DO decO Cp x s.1 s.2.2.1 v)) M s.2.2.2
      else 0

/-- `Pr[FCI^{P*}(x) outputs a collision]` at `M`, `0` where the inner statement has no cap position. -/
noncomputable def PrFCI (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) : ℝ :=
  if hW : 0 < Cp.width (xI D x) then (FCI H D I DO decO Cp P x M hW).Pr H else 0

/-- `F0I`'s coins at `x`: a tape, the first draw, and `k` waits of `R` draws each. -/
abbrev ConflictCoinsI (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Ω : Type) (x : Msg) (k R : ℕ) : Type :=
  Ω × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) × (Fin k → Fin R → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)

open Classical in
/-- **`F0I`, the inner level 0's conflict over `E_O`'s draws**: `F0` at the composite. A tape and a draw; if its inner
view accepts, the inner `E^rep`'s `k_I` waits of at most `R` draws each, each stopping at its first draw whose inner view
accepts, and the first collision among the calls those draws' outer witnesses record. Its cost is the draws'. -/
noncomputable def F0I (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (R : ℕ) : Finder where
  S := ConflictCoinsI D I DO decO Ω x (I.k (xI D x)) R
  ne := ⟨Classical.arbitrary _⟩
  out s :=
    if accAtI H D I DO decO P x s.1 s.2.1 then
      findColl (D.callsAt H x s.2.1.1 (drawWits H D I DO decO P x s.1 s.2.1) ++
        (List.finRange (I.k (xI D x))).flatMap fun i =>
          ((firstPick (accAtI H D I DO decO P x s.1) R (s.2.2 i)).map fun d =>
            D.callsAt H x d.1 (drawWits H D I DO decO P x s.1 d)).getD [])
    else none
  cost s :=
    drawCost H D I DO decO P x s.1 s.2.1 +
      if accAtI H D I DO decO P x s.1 s.2.1 then
        ∑ i, loopCost (accAtI H D I DO decO P x s.1) (drawCost H D I DO decO P x s.1) R (s.2.2 i)
      else 0

/-- `Pr[F0I^{P*}(x) outputs a collision]` at waits of `R` draws. -/
noncomputable def PrF0I (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (R : ℕ) : ℝ :=
  (F0I H D I DO decO P x R).Pr H

/-- **`Fdev`'s weight with the inner Flock's finders**: `β₂ = (N_I + 8·W_I + 4·(k_I + 1))·k_I(x_I)`, a guess among the
inner rounds at each draw the inner `E^rep` (`N_I` expected), `FCI` (two, at weight `4·W_I`) and `F0I` (`k_I + 1`, at
weight `4`) make, with `W_I` the inner caps' positions and `k_I` the inner `E^rep`'s accepted reruns per wait. -/
noncomputable def βdev2 (D : Deploy Coin) (I : Inner Coin W) (Cp : Caps Coin) (x : Msg) : ℝ :=
  (NI D I x + 8 * Cp.width (xI D x) + 4 * ((I.k (xI D x) : ℝ) + 1)) * I.V.k (xI D x)

/-- **`F` charged `n` more SHA-512 calls** on every coin: a finder that also evaluates `n` fixed inputs. -/
def _root_.Claims.Finder.charge (F : Finder) (n : ℕ) : Finder := { F with cost := fun s => F.cost s + n }

/-- **The inner setup's calls at `x`**: `|S(x_I)|`, what a finder searching them evaluates once. -/
def setupLen (D : Deploy Coin) (x : Msg) : ℕ := (D.innerSetup (xI D x)).length

/-- **The recursion's bits with the inner Flock's roots linear**: `λ₂ = min(log₂(1/κ), 255.5 − ½ log₂ γ,
256 − log₂((4k_I + 1)·c_S), 255 − log₂(4·W_I·c_S), 256 − log₂(4·(k_I + 1)·c_S), 255 − log₂(β₂·c_S)) − log₂ 6`
at `c_S = c_Be + |S(x_I)| + 2(n + v)`, with `k_I` the inner verifier's rounds in the third term and the inner
`E^rep`'s accepted reruns per wait in the fifth, `n` a bound on the inner verifier's calls on one draw's inner messages
(`Deploy.innerCalls`) and `v` one on the compiled verifier's own evaluations at a draw's V\* statements (`setupCalls`,
check 5's row commitments among them). `B_I` costs at most `c_Be` per call of the prover's run, `FCI` and `Fdev` two
draws', `c_Be` too, and `F0I` `k_I + 1` draws', `(k_I + 1)·c_Be/2`, each with the inner setup's `|S(x_I)|` calls once
(`setupLen`) and `n + v` calls per draw it reads; the tails `4·W_I/(M+1)`, `β₂/(M+1)` and `4·(k_I + 1)/(R+1)` are
below their collision terms at `M + 1 ≥ 2^256/(c_Be·t)` and `R + 1 ≥ 2^257/(c_Be·t)`. -/
noncomputable def lamLin2 (D : Deploy Coin) (I : Inner Coin W) (Cp : Caps Coin) (x : Msg) (n v : ℕ) : ℝ :=
  min (min (min (min (min (Real.logb 2 (1 / κ D I x)) (255.5 - Real.logb 2 (γ D I x) / 2))
    (256 - Real.logb 2 ((4 * I.V.k (xI D x) + 1) * (cBe D x + setupLen D x + 2 * ((n + v : ℕ) : ℝ)))))
    (255 - Real.logb 2 (4 * Cp.width (xI D x) * (cBe D x + setupLen D x + 2 * ((n + v : ℕ) : ℝ)))))
    (256 - Real.logb 2 (4 * ((I.k (xI D x) : ℝ) + 1) * (cBe D x + setupLen D x + 2 * ((n + v : ℕ) : ℝ)))))
    (255 - Real.logb 2 (βdev2 D I Cp x * (cBe D x + setupLen D x + 2 * ((n + v : ℕ) : ℝ)))) - Real.logb 2 6

/-- **The recursion: KS at the representation V\* binds, with the inner Flock's roots linear**: at the compiled record
whose inner view runs `IVStar` (`D.record Bd.IVStar`), over the bound record, at every inner and outer decoder that meets
the spec, on every prover and at every `M`, `R`: `Pr[run = acc ∧ ¬R(x, E*(x))] ≤ κ(x) + √(γ(x)·Pr[B]) +
(4k_I + 1)·Pr[B_I] + 4·W_I·(Pr[FCI] + 1/(M+1)) + 4·(Pr[F0I] + (k_I + 1)/(R+1)) + β₂·(Pr[Fdev] + 1/(M+1))`, the inner
caps read by `Cp` at the inner statement, and `B_I`, `FCI`, `F0I` and `Fdev` searching the inner verifier's own calls,
its setup's and those on each draw's inner messages at the draw's coins, with the recorded ones (`Deploy.callsAt`),
and `B_I` and `Fdev` the compiled verifier's own at each draw's V\* statements too (`BinnerV`, `FdevV`, `drawCalls`).
It is `KSBoundLin` with `KSFlockLin`'s terms in place of `√(γ_BI·Pr[B_I])`. -/
def KSLin2 [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) : Prop :=
  ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M R : ℕ),
    PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
        (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x * PrB H (D.record Bd.IVStar).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) + 1) *
          PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (xI (D.record Bd.IVStar) x) *
          (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) / ((R : ℝ) + 1)) +
        βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x *
          (PrFdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
            1 / ((M : ℝ) + 1))

/-- **The recursion: bits at the representation V\* binds, with the inner Flock's roots linear**, at every inner and
outer decoder that meets the spec: every prover making at most `t ≥ 1` SHA-512 calls per run (both parties') has
`Pr[acc ∧ ¬R] ≤ t · 2^{−λ₂(x)}`, at every `n` bounding the inner verifier's calls on any outer witnesses' inner messages
at any coins (`Deploy.innerCalls`) and every `v` bounding the compiled verifier's own evaluations at any staged V\* statements
(`setupCalls`), where `B` obeys the strict birthday bound and `B_I`, `Fdev` and `FCI` at every `M` and `F0I` at every
`R` the linear expected-time bound, each searching the inner verifier's own calls with the recorded ones (`B_I` and
`Fdev` the compiled verifier's too, `BinnerV`, `FdevV`) and charged their evaluations (`Finder.charge`): the setup's
`|S(x_I)|` once (`setupLen`) and `n + v` per draw it reads, `2(n + v)` for `B_I`, `Fdev` and `FCI` and
`(k_I + 1)·(n + v)` for `F0I`. -/
def BitsLin2 [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) : Prop :=
  ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P (D.record Bd.IVStar).V x ω r).pLog.length +
      (run H P (D.record Bd.IVStar).V x ω r).vLog.length ≤ t) →
    ∀ n : ℕ, (∀ cs ws, ((D.record Bd.IVStar).innerCalls H x cs ws).length ≤ n) →
    ∀ v : ℕ, (∀ ps, (setupCalls H (D.record Bd.IVStar) ((D.record Bd.IVStar).vstar x ps)).length ≤ v) →
    (∀ hk : 0 < (D.record Bd.IVStar).V.k x, StrictBirthday H (B H (D.record Bd.IVStar).V P x hk)) →
    (∀ hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x),
      ExpectedCR H ((BinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ (M : ℕ) (hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ (M : ℕ) (hW : 0 < Cp.width (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M
        hW).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
    (∀ R : ℕ, ExpectedCR H ((F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
      R).charge (setupLen (D.record Bd.IVStar) x +
        ((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) + 1) * (n + v)))) →
    PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
        (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      t * (2 : ℝ) ^ (-lamLin2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x n v)

end Composite

end Claims.Recursion
