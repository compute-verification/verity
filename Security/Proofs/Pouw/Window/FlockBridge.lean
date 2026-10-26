import Definitions.Pouw.Guarantees.Window
import Proofs.Flock.Soundness.Audit.Extraction
import Proofs.Flock.Soundness.Discharge.ZkLink.SessionZ

/-!
# The bridge from the window's game to C-Flock's

`WindowEps` is stated over the window's own game (`Pouw.Window`), and C-Flock's soundness over `FlockSoundness`'s. The
two are the same trees: this file maps a C-Flock session into the window's game (`ofFlock`), the window's strategies for
it back to C-Flock's (`Strategy.toFlock`), and proves that every probability agrees (`prob_ofFlock`). A C-Flock
`ExtractionAnalysis` of a call's session is then a `CallAnalysis` of that call in the window (`CallAnalysis.ofFlock`),
given that the window's wrong units of the call are wrong in C-Flock's transcript, and C-Flock's knowledge bound,
pointwise in the session's state (`FlockKnowledgeSound`), is the call's `KnowledgeSound`
(`CallAnalysis.ofFlock_knowledgeSound`). At main's `--zk` sessions `FlockKnowledgeSound` is proved
(`flockKnowledgeSound_analysisZC`).

**Which coins the `--zk` analysis reads.** Call `c`'s session is the subtree `ofFlock (fsess c S R)` of the window's
game, entered after the registration `R`, the draw's coin and the earlier calls' sessions. Its coins are the compiled
session's own (the verifier's challenges in `batchedSessionZC`), and `analysisZC`'s extractor reruns everything after
its `Commit` until `k` reruns accept. The state it bounds is the strategy the window's prover plays in that subtree,
which the registration, the draw and the earlier sessions fixed. So `jointZC_ksBound` bounds `ks` conditionally on
everything before the session, never over the draw's coin or another session's, and the window's collision term
(`coll`) takes the expectation of the state terms over those.

**What `κ` and `g` are at main's sessions.** Main's window proves each drawn unit in a compiled `--zk` session over hm96
hiding leaves (`batchedSessionZC`), whose analysis is `analysisZC` (`Proofs/Flock/Soundness/Discharge/ZkLink/Analysis.lean`)
at the knowledge bound `ksBoundAccZ`, with its joint inequality discharged (`jointZC_ksBound`). There
`ks S R τ u = ofReal (ksBoundAccZ_j(zcStrat τ) / (1 − rateZ_j))` at `u`'s table `j`, with `ksBoundAccZ = ε_c⁻ + √(k·adv₀)`
the table's own terms and `rateZ = N₀/(ek) < 1`. So `FlockKnowledgeSound` holds with `κ = 0` and
`g S R τ = ⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`, the state's collision finders' advantages, and
these are the `κ_c` and `g_c` `WindowEps` reads through `ofFlock_knowledgeSound`: the relative term sits in the
denominator, not in `κ`. `ksJoint_le` (`Proofs/Flock/Soundness/Audit/FlockCompiled.lean`, `κ = 2^logLen/(e·Kr)`) is the
non-ZK plain-leaf table's, which main's window doesn't run. The link term `δlink_c` stays `LinkSoundIn`, C-Flock's link
finder at the window's state.
-/

namespace Pouw.Window

open scoped ENNReal

/-- **C-Flock's interaction as the window's**: the same tree, node for node. -/
def ofFlock {α : Type} : FlockSoundness.Game α → Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => ofFlock (k m)
  | @FlockSoundness.Game.coin _ C iF iN k => @Game.coin _ C iF iN fun c => ofFlock (k c)

/-- A window strategy for a C-Flock interaction, as C-Flock's strategy: the same move at every node. -/
def Strategy.toFlock {α : Type} :
    (g : FlockSoundness.Game α) → Strategy (ofFlock g) → FlockSoundness.Game.Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.toFlock (k s.1) s.2⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, s => fun c => Strategy.toFlock (k c) (s c)

/-- **The probabilities agree**: a window strategy wins the window's copy of a C-Flock interaction with exactly the
probability its C-Flock strategy wins the interaction. -/
theorem prob_ofFlock {α : Type} (p : α → Prop) :
    ∀ (g : FlockSoundness.Game α) (s : Strategy (ofFlock g)),
      prob p (ofFlock g) s = FlockSoundness.Game.prob p g (Strategy.toFlock g s)
  | .ret _, _ => rfl
  | .send _ k, s => prob_ofFlock p (k s.1) s.2
  | @FlockSoundness.Game.coin _ C iF _ k, s => by
    change ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (ofFlock (k c)) (s c) =
      ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * FlockSoundness.Game.prob p (k c) (Strategy.toFlock (k c) (s c))
    exact Finset.sum_congr rfl fun c _ => by rw [prob_ofFlock p (k c) (s c)]

section Call

variable {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ}

/-- **A call's analysis from C-Flock's.** Call `c`'s session is C-Flock's session `fsess` read in the window's game, and
`AF` is C-Flock's randomized-extraction analysis of it (`FlockSoundness.Audit.ExtractionAnalysis`, at any law `LF`,
which its `ks`, `link` and `cover` don't read). `proj Xw` is the window transcript `Xw` read as C-Flock's transcript of
the call, and `hwrong` says the window's units' statements are the session's: a unit of the call that is wrong in `Xw`
(`UnitStmts.wrong`, at its seed inputs) is wrong in `proj Xw`. The window's `ks` and `link` are C-Flock's at the
strategy the window's prover plays in the session, and `cover` is C-Flock's (`prob_ofFlock`). -/
def CallAnalysis.ofFlock (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → FlockSoundness.Game Bool} {LF : FlockSoundness.Audit.Law (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.wrong Xw → u ∈ AF.wrong (proj Xw)) :
    CallAnalysis U c (fun S R => ofFlock (fsess S R)) where
  ks S R τ u := AF.ks S R (Strategy.toFlock _ τ) u
  link S R τ u Xw := AF.link S R (Strategy.toFlock _ τ) u (proj Xw)
  cover S R τ u Xw hS hX := by
    rw [prob_ofFlock]
    exact AF.cover S R _ u (proj Xw) hS (hwrong Xw u hX)

/-- **C-Flock's knowledge bound, pointwise in the session's state**: from every state `τ` of the session on every draw
`S` and registration `R`, `Pr[accept ∧ extraction of a drawn unit fails] ≤ κ + g(S, R, τ)`. At main's `--zk` sessions
(`analysisZC`) it holds with `κ = 0` and `g = ⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`
(`flockKnowledgeSound_analysisZC`, module doc). -/
def FlockKnowledgeSound {m : ℕ} {fsess : Finset (Fin m) → Reg → FlockSoundness.Game Bool}
    {LF : FlockSoundness.Audit.Law m} (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (κ : ℝ≥0∞)
    (g : (S : Finset (Fin m)) → (R : Reg) → FlockSoundness.Game.Strategy (fsess S R) → ℝ≥0∞) : Prop :=
  ∀ S R τ u, u ∈ S → AF.ks S R τ u ≤ κ + g S R τ

/-- **C-Flock's pointwise bound is the call's knowledge soundness in the window**, with the state term read at the
strategy the window's prover plays in the session. -/
theorem CallAnalysis.ofFlock_knowledgeSound (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → FlockSoundness.Game Bool} {LF : FlockSoundness.Audit.Law (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.wrong Xw → u ∈ AF.wrong (proj Xw)) {κ : ℝ≥0∞}
    {g : (S : Finset (Fin (n c))) → (R : Reg) → FlockSoundness.Game.Strategy (fsess S R) → ℝ≥0∞}
    (h : FlockKnowledgeSound AF κ g) :
    (CallAnalysis.ofFlock U c AF proj hwrong).KnowledgeSound κ fun S R τ => g S R (Strategy.toFlock _ τ) :=
  fun S R _ u hu => h S R _ u hu

end Call

end Pouw.Window

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- **C-Flock's knowledge bound at main's `--zk` sessions**: `analysisZC` at the knowledge bound `ksBoundAccZ`, whose
joint inequality `jointZC_ksBound` discharges for `k ≥ 1` accepted reruns, is `FlockKnowledgeSound` with `κ = 0` and the
state term `⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at `zcStrat τ`: a drawn unit's `ks` is that term at its own
table. -/
theorem _root_.Pouw.Window.flockKnowledgeSound_analysisZC (A : Arith F K) (hA : A.Correct) (Hs : Hashes F D Sl)
    {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
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
