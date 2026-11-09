module

public import Definitions.Pouw.Guarantees.Window
public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.Discharge.ZkLink.SessionZ

/-!
# The window's calls from Flock's analysis

`WindowDelta` and Flock's soundness are stated over one game, core's (`Verity.Game`; `FlockSoundness.Game` is its
alias), so a Flock `ExtractionAnalysis` of a call's session is a `CallAnalysis` of that call in the window, with the same
`ks`, `link` and `cover` (`CallAnalysis.ofFlock`), given that the window's incorrect proof blocks of the call are wrong in
Flock's transcript, and Flock's knowledge bound, pointwise in the session's state (`FlockKnowledgeSound`), is the call's
`KnowledgeSound` (`CallAnalysis.ofFlock_knowledgeSound`). At main's `--zk` sessions `FlockKnowledgeSound` is proved
(`flockKnowledgeSound_analysisZC`).

**Which coins the `--zk` analysis reads.** Call `c`'s session is the subtree `fsess S R` of the window's game, entered
after the registration `R`, the sample's coin and the earlier calls' sessions. Its coins are the compiled session's own
(the verifier's challenges in `batchedSessionZC`), and `analysisZC`'s extractor reruns everything after its `Commit`
until `k` reruns accept. The state it bounds is the strategy the window's prover plays in that subtree, which the
registration, the sample and the earlier sessions fixed. So `jointZC_ksBound` bounds `ks` conditionally on everything
before the session, never over the sample's coin or another session's, and the window's collision term (`coll`) takes
the expectation of the state terms over those.

**What `κ` and `g` are at main's sessions.** Main's window proves each sampled proof block in a compiled `--zk` session over hm96
hiding leaves (`batchedSessionZC`), whose analysis is `analysisZC` (`Proofs/Flock/Soundness/Discharge/ZkLink/Analysis.lean`)
at the knowledge bound `ksBoundAccZ`, with its joint inequality discharged (`jointZC_ksBound`). There
`ks S R τ u = ofReal (ksBoundAccZ_j(zcStrat τ) / (1 − rateZ_j))` at `u`'s table `j`, with `ksBoundAccZ = ε_c⁻ + √(k·adv₀)`
the table's own terms and `rateZ = N₀/(ek) < 1`. So `FlockKnowledgeSound` holds with `κ = 0` and
`g S R τ = ⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`, the state's collision finders' advantages, and
these are the `κ_c` and `g_c` `WindowDelta` reads through `ofFlock_knowledgeSound`: the relative term sits in the
denominator, not in `κ`. `ksJoint_le` (`Proofs/Flock/Soundness/Audit/FlockCompiled.lean`, `κ = 2^logLen/(e·Kr)`) is the
non-ZK plain-leaf table's, which main's window doesn't run. The binding error `δlink_c` stays `BindSoundIn`, Flock's link
finder at the window's state.
-/

@[expose] public section

namespace Pouw.Window

open scoped ENNReal

section Call

variable {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ}

/-- **A call's analysis from Flock's.** Call `c`'s session is Flock's session `fsess`, and `AF` is Flock's
randomized-extraction analysis of it (`FlockSoundness.Audit.ExtractionAnalysis`, at any sampling design `LF`, which its
`ks`, `link` and `cover` don't read). `proj Xw` is the window transcript `Xw` read as Flock's transcript of the call, and
`hwrong` says the window's units' statements are the session's: a unit of the call that is wrong in `Xw`
(`UnitStmts.wrong`, at its seed inputs) is wrong in `proj Xw`. The window's `ks`, `link` and `cover` are Flock's. -/
def CallAnalysis.ofFlock (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → Game Bool} {LF : FlockSoundness.Audit.Law (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.wrong Xw → u ∈ AF.wrong (proj Xw)) :
    CallAnalysis U c fsess where
  ks := AF.ks
  link S R τ u Xw := AF.link S R τ u (proj Xw)
  cover S R τ u Xw hS hX := AF.cover S R τ u (proj Xw) hS (hwrong Xw u hX)

/-- **Flock's knowledge bound, pointwise in the session's state**: from every state `τ` of the session on every sample
`S` and registration `R`, `Pr[accept ∧ extraction of a drawn unit fails] ≤ κ + g(S, R, τ)`. At main's `--zk` sessions
(`analysisZC`) it holds with `κ = 0` and `g = ⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`
(`flockKnowledgeSound_analysisZC`, module doc). -/
def FlockKnowledgeSound {m : ℕ} {fsess : Finset (Fin m) → Reg → Game Bool}
    {LF : FlockSoundness.Audit.Law m} (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (κ : ℝ≥0∞)
    (g : (S : Finset (Fin m)) → (R : Reg) → Strategy (fsess S R) → ℝ≥0∞) : Prop :=
  ∀ S R τ u, u ∈ S → AF.ks S R τ u ≤ κ + g S R τ

/-- **Flock's pointwise bound is the call's knowledge soundness in the window.** -/
theorem CallAnalysis.ofFlock_knowledgeSound (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → Game Bool} {LF : FlockSoundness.Audit.Law (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.wrong Xw → u ∈ AF.wrong (proj Xw)) {κ : ℝ≥0∞}
    {g : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (fsess S R) → ℝ≥0∞}
    (h : FlockKnowledgeSound AF κ g) : (CallAnalysis.ofFlock U c AF proj hwrong).KnowledgeSound κ g :=
  h

end Call

end Pouw.Window

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- **Flock's knowledge bound at main's `--zk` sessions**: `analysisZC` at the knowledge bound `ksBoundAccZ`, whose
joint inequality `jointZC_ksBound` discharges for `k ≥ 1` accepted reruns, is `FlockKnowledgeSound` with `κ = 0` and the
state term `⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`: a sampled proof block's `ks` is that term at its own
table. -/
theorem _root_.Pouw.Window.flockKnowledgeSound_analysisZC (A : Arith F K) (hA : A.Correct) (Hs : Hashes F D Sl)
    {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) {L : Law n}
    (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ} (hk : 1 ≤ k)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode) :
    Pouw.Window.FlockKnowledgeSound
      (analysisZC A Hs P planZ tab decode Xc k hr hlow (fun S R τ j => ksBoundAccZ A Hs (planZ S R) j k τ)
        (jointZC_ksBound A hA Hs planZ hk)) 0
      (fun S R τ => ⨆ j : Fin (planZ S R).length,
        ENNReal.ofReal (ksBoundAccZ A Hs (planZ S R) j k (zcStrat A Hs planZ τ) / (1 - rateZ (planZ S R) j k))) := by
  intro S R τ u _
  rw [zero_add]
  exact le_iSup (fun j : Fin (planZ S R).length =>
    ENNReal.ofReal (ksBoundAccZ A Hs (planZ S R) j k (zcStrat A Hs planZ τ) / (1 - rateZ (planZ S R) j k)))
    (tab S R u)

end FlockSoundness.Discharge.ZkLink
