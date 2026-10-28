import Proofs.Flock.Claims.Witnesses.Flock
import Proofs.Flock.Claims.ZK.Rewind
import Proofs.Flock.Claims.ZK.Indexed
import Definitions.Claims.Recursion
import Proofs.Flock.Claims.ZK.Indexed

/-!
# Refutation witnesses: the ZK-recursion compiler's claims

Each of the recursion's claims (`Definitions.Claims.Recursion`) fails at a toy record:

* `ksR`: the compiled verifier accepts after one round, nothing parses (so no statement has a witness, in either
  variant), there is no V* session, and the outer witnesses record no call. So `Pr[acc ∧ ¬R] = 1`, `B` and `Be` find
  nothing, and at an inner error of `1/8` (`accI`) the bound is `κ = 1/8`, with `λ = 1`. `KS_fails`, `KSPriv_fails`,
  `Bits_fails`; over Flock (`Flock.ksD` inside, `κ_I < 1/4`), `KSFlock_fails` and `BitsFlock_fails`; at the
  representation V* binds (`ksB`, `Flock.ksD` element-level, whose bound record has `Flock.ksD`'s `κ`, with `ksR` the
  compiled record at every inner verifier), `KSBound_fails` and `KSPrivBound_fails`.
  `ScopeMatches_fails`: its scope is empty and its check takes everything.
* `compR`: the compiled verifier rejects after one round, and every statement parses as `recStmt`, in scope, under an
  inner protocol whose relation and scope hold everywhere. `Complete_fails`: its reference prover fails at its
  error `δ_c = 0`. `Nontrivial_fails`: a recorded run is rejected.
* `effR`: the run always accepts, but its one V* session's view never does, so the outer extractor's wait reruns all
  `K = 2` of its draws where `Efficient`'s expected count allows `k = 1`. `Efficient_fails`: `E*` reruns `3` times on
  every coin, above the expected bound `N_I · (1 + m′·k) = 2`. The worst-case conjunct holds at every record (each
  wait reruns at most `K`); the expected one also needs a session's view to accept wherever the run does, which the
  record's `session` doesn't promise.
* `zkZR` (`zkZPR` with a circuit commitment): the compiled verifier takes one round, and every statement parses,
  under an inner protocol of no rounds (`zeroI`), with no staging rounds and no V* session, so `Sim` outputs the empty
  transcript; its reference prover, `ksR`'s `silentP`, is the quiet prover on the masked record's one nonce and one
  rest. `ZK_fails`, `ZKPriv_fails`: at a masked outer Flock, every record's outer simulators and every hiding level
  (all masks hide up to `δ₁ = 1`), that prover's one message is never simulated, at an error of `0`. `zkSR` is
  `zkZR` with a stop that never fires and no inner commit strings, where `SimStop` runs as `Sim` does:
  `ZKStop_fails`, at `zkSR`'s own reference prover, the quiet one, as `ZK` fails.
* `Inner.KS_fails`, `Inner.KSAbort_fails`: at `noI`, an inner protocol that accepts and whose relation is empty, with
  no abort.

`ZK` and `ZKPriv` read the masked record's own reference prover (`ZkDeploy.prover`: `Deploy.P`, which `Complete`
reads, on the record's tape) and its outer simulators (`ZkDeploy.SO`), so a prover that ignores its witness meets them
only at a record whose own reference prover does: `ZK_zeroRecord`, `ZKPriv_zeroRecord`, at `zvZR`, a record of one
inner round with a one-byte salt whose reference prover sends `Sim`'s commit string to the zero digest (`zeroP`), where
every witness is in the relation (`accI`), at an error of `0` (`zkmalR_zeroP`).
-/

namespace Claims.Witness.Recursion

open Claims Claims.Recursion
open scoped ENNReal NNReal

/-! ## The toy records -/

/-- An inner protocol of Flock's witnesses whose verifier accepts after one round, whose relation and scope hold
everywhere, with no waits, the knowledge error `1/8` and `γ = 0`. -/
noncomputable def accI (Coin : Type) : Recursion.Inner Coin Flock.Wit where
  V := accV Coin
  R _ _ := True
  Scope _ := True
  Op := Unit
  opens _ := []
  assemble _ _ := none
  check _ _ := false
  waits _ := 0
  k _ := 0
  cap _ := 0
  κ _ := 1 / 8
  γ _ := 0

/-- `accI` with an empty relation. -/
noncomputable def noI (Coin : Type) : Recursion.Inner Coin Flock.Wit := { accI Coin with R := fun _ _ => False }

/-- The numbers of an outer session with none of its own. -/
def nums0 : Nums := ⟨0, 0, 0, 0, 0, 0, 0, 0⟩

/-- The toy record of `KS`, `Bits` and `ScopeMatches`: the compiled verifier accepts after one round, nothing parses,
there is no V* session, and the outer witnesses record no call. -/
def ksR (Coin : Type) : Recursion.Deploy Coin where
  V := accV Coin
  parse _ := none
  scopeCheck _ := true
  kS _ := 0
  nC _ := 0
  stage _ _ _ := .ret []
  vstar _ _ := []
  nS _ := 0
  session _ _ v := v
  innerView _ _ _ _ := ⟨[], .rej⟩
  calls _ _ := []
  nums _ _ := nums0
  saltLen := 0
  saltPrefix := []
  mask _ _ := 0
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

/-- A recursion statement in scope: a dense inner table at `m = 22`, `k_log = 6`, one region and four claims, with no
RecSparse phase and no V* session. -/
def recStmt : Recursion.CompiledStatement where
  inner := []
  innerDims := none
  circ := none
  form := ⟨22, 6, 1, 4, false, none, none, false⟩
  draws := []

/-- The toy record of `Complete` and `Nontrivial`: the compiled verifier rejects, and every statement parses as
`recStmt`. -/
def compR (Coin : Type) : Recursion.Deploy Coin := { ksR Coin with V := rejV Coin, parse := fun _ => some recStmt }

/-- The numbers of an outer session of one wait over `K = 2` draws, at `N₀ = e/4`, so `k = 1`. -/
noncomputable def nums1 : Nums := ⟨0, Real.exp 1 / 4, 0, 0, 1, 2, 0, 0⟩

/-- The toy record of `Efficient`: the run always accepts, and its one V* session's view never does. -/
noncomputable def effR (Coin : Type) : Recursion.Deploy Coin :=
  { ksR Coin with nS := fun _ => 1, session := fun _ _ _ => ⟨[], .rej⟩, nums := fun _ _ => nums1 }

/-- An inner protocol of no rounds, whose relation and scope hold everywhere. -/
noncomputable def zeroI (Coin : Type) : Recursion.Inner Coin Flock.Wit :=
  { accI Coin with V := ⟨fun _ => 0, fun _ _ _ => .ret .acc⟩ }

/-- The toy record of `ZK`: the compiled verifier takes one round, every statement parses as `recStmt`, and there are
no staging rounds and no V* session. -/
def zkR (Coin : Type) : Recursion.Deploy Coin := { ksR Coin with parse := fun _ => some recStmt }

/-- `zkR` with a circuit commitment in every statement. -/
def zkPR (Coin : Type) : Recursion.Deploy Coin :=
  { ksR Coin with parse := fun _ => some { recStmt with circ := some [] } }

/-- A masked outer Flock. -/
noncomputable def zkO (Coin : Type) : Claims.Flock.Deploy Coin Unit Unit := { Flock.ksD Coin with zk := true }

/-- A record of one round whose every statement parses with the circuit commitment `circ`, with no staging rounds and
no V* session, and a one-byte salt. -/
def zvR (Coin : Type) (circ : Option Msg) : Recursion.Deploy Coin :=
  { ksR Coin with parse := fun _ => some { recStmt with circ := circ }, saltLen := 1, mask := fun y _ => y 0 }

/-- The prover that ignores its witness and sends the commit string to the zero digest under the salt its nonce
carries: `Sim`'s inner round. -/
def zeroP (Coin : Type) : Prover ((Fin 1 → Fin 1 → UInt8) × Unit) :=
  ⟨fun _ _ ω => (zvR Coin none).zeroCommit (ω.1 0)⟩

/-- The toy bound record of `KSBound` and `KSPrivBound`: `Flock.ksD` element-level, no commit string at any round, a
last message that reads as nothing (`final`), and weight `0`. -/
noncomputable def ksB (Coin : Type) : Bound Coin Unit Unit where
  DE := Flock.ksD Coin
  ans _ := []
  join _ := []
  nAt _ _ := 0
  saltLen := 0
  saltPrefix := []
  mask _ _ := 0
  rowIn r := r
  final _ := none
  weight _ := 0

/-- `ksR` at an inner verifier `IV`: its inner view runs `IV` on no messages at the statement itself. -/
def ksRV (Coin : Type) [Inhabited Coin] (IV : Verifier Coin) : Recursion.Deploy Coin :=
  { ksR Coin with innerView := fun H x r _ => (run H (replay []) IV x () fun i => coinSeq r i.val).seen }

/-- The toy compiled record of `KSBound` and `KSPrivBound` at each inner verifier (`ksRV`), its inner view running the
verifier it is given. -/
def ksRAt (Coin : Type) [Inhabited Coin] : RecAt Coin where
  record := ksRV Coin
  msgs _ _ := []
  xI x := x
  runs _ _ _ _ _ := rfl

/-- The masked toy record of `ZK`: `zkR` with one nonce, one rest, the tape `()` and the outer simulators `SO`. -/
def zkZR (Coin : Type) (SO : ℕ → Msg → ℕ → OSim) : Recursion.ZkDeploy Coin :=
  { zkR Coin with Nonce := fun _ => Unit, Rest := fun _ => Unit, tapeOf := fun _ _ => (), SO := SO }

/-- `zkPR` masked as `zkZR` is. -/
def zkZPR (Coin : Type) (SO : ℕ → Msg → ℕ → OSim) : Recursion.ZkDeploy Coin :=
  { zkPR Coin with Nonce := fun _ => Unit, Rest := fun _ => Unit, tapeOf := fun _ _ => (), SO := SO }

/-- `zkZR` with a stop that never fires and no commit string at an inner round. -/
def zkSR (Coin : Type) (SO : ℕ → Msg → ℕ → OSim) : Recursion.StopDeploy Coin :=
  { zkZR Coin SO with nAt := fun _ _ => 0, innerMsg := fun _ _ _ => .ret [], stops := fun _ _ _ => false }

/-- `zvR` at the circuit commitment `circ`, whose reference prover is `zeroP`, masked with the nonce `zeroP` reads, one
rest and the outer simulators `SO`. -/
def zvZR (Coin : Type) (circ : Option Msg) (SO : ℕ → Msg → ℕ → OSim) : Recursion.ZkDeploy Coin :=
  { zvR Coin circ with
    Tape := (Fin 1 → Fin 1 → UInt8) × Unit
    tapeNonempty := inferInstance
    P := fun _ => zeroP Coin
    Nonce := fun _ => Fin 1 → Fin 1 → UInt8
    Rest := fun _ => Unit
    tapeOf := fun _ ω => ω
    SO := SO }

theorem recStmt_inScope : Recursion.InScope recStmt :=
  ⟨rfl, le_rfl, by decide, le_rfl, rfl, Or.inl ⟨rfl, rfl⟩, fun _ h => absurd h List.not_mem_nil⟩

/-! ## `ksR`'s extraction and numbers -/

section Numbers

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ksR_noR {W : Type} (I : Recursion.Inner Coin W) (x : Msg) (w : W) : ¬ Recursion.R (ksR Coin) I x w := by
  rintro ⟨s, hs, -⟩
  simp [ksR] at hs

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ksR_noRPriv {W : Type} (I : Recursion.Inner Coin W) (x : Msg) (w : W) : ¬ RPriv (ksR Coin) I x w := by
  rintro ⟨s, hs, -⟩
  simp [ksR] at hs

omit [DecidableEq Col] in
/-- `Be` finds nothing at `ksR`: the outer witnesses record no call. -/
theorem ksR_be_pr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω) (x : Msg)
    (hk : 0 < (ksR Coin).V.k x) : (Be H (ksR Coin) I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Be, ksR, List.append_nil, findColl, reduceCtorEq] at h

omit [DecidableEq Col] in
theorem ksR_prBe {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω) (x : Msg) :
    PrBe H (ksR Coin) I DO decO P x = 0 := by
  unfold PrBe
  split_ifs with hk
  · exact ksR_be_pr H I DO decO P x hk
  · rfl

omit [DecidableEq Col] in
theorem ksR_expectedCR {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω) (x : Msg)
    (hk : 0 < (ksR Coin).V.k x) : ExpectedCR H (Be H (ksR Coin) I DO decO P x hk) := by
  unfold ExpectedCR
  rw [ksR_be_pr]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- `ksR`'s setup makes no evaluation. -/
theorem ksR_setupCalls (H : Msg → Msg) (xs : List Msg) : setupCalls H (ksR Coin) xs = [] := by
  simp only [setupCalls, List.flatMap_eq_nil_iff, List.map_eq_nil_iff]
  exact fun _ _ => rfl

omit [DecidableEq Col] in
/-- `Br` finds nothing at `ksR`: the outer witnesses record no call, and its setup makes no evaluation. -/
theorem ksR_br_pr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω) (x : Msg)
    (hk : 0 < (ksR Coin).V.k x) : (Br H (ksR Coin) I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Br, ksR_setupCalls, List.append_nil, findCollIn, reduceCtorEq] at h

omit [DecidableEq Col] in
theorem ksR_prBr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω) (x : Msg) :
    PrBr H (ksR Coin) I DO decO P x = 0 := by
  unfold PrBr
  split_ifs with hk
  · exact ksR_br_pr H I DO decO P x hk
  · rfl

omit [DecidableEq Col] in
theorem ksR_strictBirthday_br {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω)
    (x : Msg) (hk : 0 < (ksR Coin).V.k x) : StrictBirthday H (Br H (ksR Coin) I DO decO P x hk) := by
  intro q _
  rw [ksR_br_pr]
  positivity

omit [DecidableEq Col] in
theorem ksR_prBad {W : Type} (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (ksR Coin).V) (x : Msg)
    (hR : ∀ w, ¬ R x w) : PrBad H (ksR Coin).V R E x silentP = 1 :=
  prBad_accV H R E x silentP hR

omit [DecidableEq Col] in
theorem ksR_prB (H : Msg → Msg) (x : Msg) : PrB H (ksR Coin).V silentP x = 0 := prB_accV H silentP x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ksR_κ {W : Type} (I : Recursion.Inner Coin W) (x : Msg) : Recursion.κ (ksR Coin) I x = I.κ [] := by
  have : IsEmpty (Fin ((ksR Coin).nS x)) := Fin.isEmpty'
  unfold Recursion.κ
  rw [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, add_zero]
  rfl

/-! `ksRV`, `ksR` at an inner verifier, has `ksR`'s numbers: only its inner view differs. -/

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ksRAt_record (IV : Verifier Coin) : (ksRAt Coin).record IV = ksRV Coin IV := rfl

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ksRV_noR {W : Type} (IV : Verifier Coin) (I : Recursion.Inner Coin W) (x : Msg) (w : W) :
    ¬ Recursion.R (ksRV Coin IV) I x w := by
  rintro ⟨s, hs, -⟩
  simp [ksRV, ksR] at hs

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ksRV_noRPriv {W : Type} (IV : Verifier Coin) (I : Recursion.Inner Coin W) (x : Msg) (w : W) :
    ¬ RPriv (ksRV Coin IV) I x w := by
  rintro ⟨s, hs, -⟩
  simp [ksRV, ksR] at hs

omit [DecidableEq Col] in
theorem ksRV_be_pr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (IV : Verifier Coin)
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω)
    (x : Msg) (hk : 0 < (ksRV Coin IV).V.k x) : (Be H (ksRV Coin IV) I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Be, ksRV, ksR, List.append_nil, findColl, reduceCtorEq] at h

omit [DecidableEq Col] in
theorem ksRV_prBe {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (IV : Verifier Coin)
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω)
    (x : Msg) : PrBe H (ksRV Coin IV) I DO decO P x = 0 := by
  unfold PrBe
  split_ifs with hk
  · exact ksRV_be_pr H IV I DO decO P x hk
  · rfl

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- `ksRV`'s setup makes no evaluation. -/
theorem ksRV_setupCalls (H : Msg → Msg) (IV : Verifier Coin) (xs : List Msg) :
    setupCalls H (ksRV Coin IV) xs = [] := by
  simp only [setupCalls, List.flatMap_eq_nil_iff, List.map_eq_nil_iff]
  exact fun _ _ => rfl

omit [DecidableEq Col] in
/-- `Br` finds nothing at `ksRV`: its setup makes no evaluation. -/
theorem ksRV_br_pr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (IV : Verifier Coin)
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω)
    (x : Msg) (hk : 0 < (ksRV Coin IV).V.k x) : (Br H (ksRV Coin IV) I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Br, ksRV_setupCalls, List.append_nil, findCollIn, reduceCtorEq] at h

omit [DecidableEq Col] in
theorem ksRV_prBr {W : Type} {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (IV : Verifier Coin)
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) (P : Prover Ω)
    (x : Msg) : PrBr H (ksRV Coin IV) I DO decO P x = 0 := by
  unfold PrBr
  split_ifs with hk
  · exact ksRV_br_pr H IV I DO decO P x hk
  · rfl

omit [DecidableEq Col] in
theorem ksRV_prBad {W : Type} (H : Msg → Msg) (IV : Verifier Coin) (R : Rel W)
    (E : Extractor Coin W (ksRV Coin IV).V) (x : Msg) (hR : ∀ w, ¬ R x w) :
    PrBad H (ksRV Coin IV).V R E x silentP = 1 :=
  prBad_accV H R E x silentP hR

omit [DecidableEq Col] in
theorem ksRV_prB (H : Msg → Msg) (IV : Verifier Coin) (x : Msg) : PrB H (ksRV Coin IV).V silentP x = 0 :=
  prB_accV H silentP x

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem ksRV_κ {W : Type} (IV : Verifier Coin) (I : Recursion.Inner Coin W) (x : Msg) :
    Recursion.κ (ksRV Coin IV) I x = I.κ [] := by
  have : IsEmpty (Fin ((ksRV Coin IV).nS x)) := Fin.isEmpty'
  unfold Recursion.κ
  rw [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, add_zero]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ksR_γ {W : Type} (I : Recursion.Inner Coin W) (x : Msg) : Recursion.γ (ksR Coin) I x = 0 := by
  have : IsEmpty (Fin ((ksR Coin).nS x)) := Fin.isEmpty'
  unfold Recursion.γ
  rw [Finset.univ_eq_empty, Finset.sum_empty, mul_zero]
  norm_num

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- At `ksR`, `γ_e · c_Be = 4·(√γ_I + N_I·√k_I)²`, at the inner statement `[]`. -/
theorem ksR_γe_cBe {W : Type} (I : Recursion.Inner Coin W) (x : Msg) :
    Recursion.γe (ksR Coin) I x * cBe (ksR Coin) x =
      4 * (Real.sqrt (I.γ []) + (1 + I.waits [] * I.k [] : ℕ) * Real.sqrt (I.V.k [])) ^ 2 := by
  have : IsEmpty (Fin ((ksR Coin).nS x)) := Fin.isEmpty'
  unfold cBe
  rw [Finset.univ_eq_empty, Finset.sum_empty]
  simp only [Recursion.γe, NI, ksR, xI, accV, Option.map_none, Option.getD_none, Nat.cast_one, mul_one]
  push_cast
  ring

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- `λ > 0` at `ksR` and an inner error `κ_I < 1/4`, whenever `γ_e · c_Be < 2^250`. -/
theorem ksR_lam_pos {W : Type} (I : Recursion.Inner Coin W) (x : Msg) (hκ0 : 0 < I.κ []) (hκ : I.κ [] < 1 / 4)
    (he : Recursion.γe (ksR Coin) I x * cBe (ksR Coin) x < 2 ^ 250) : 0 < Recursion.lam (ksR Coin) I x := by
  have h1 : 2 < Real.logb 2 (1 / I.κ []) := by
    rw [Real.lt_logb_iff_rpow_lt (by norm_num) (by positivity), lt_div_iff₀ hκ0]
    norm_num
    linarith
  have h3 : Real.logb 2 (Recursion.γe (ksR Coin) I x * cBe (ksR Coin) x) < 250 := by
    rcases (mul_nonneg (by unfold Recursion.γe; positivity) (Nat.cast_nonneg _) :
        0 ≤ Recursion.γe (ksR Coin) I x * cBe (ksR Coin) x).lt_or_eq with h0 | h0
    · rw [Real.logb_lt_iff_lt_rpow (by norm_num) h0]
      exact_mod_cast he
    · rw [← h0, Real.logb_zero]; norm_num
  unfold Recursion.lam
  rw [ksR_κ, ksR_γ, Real.logb_zero]
  have := lt_min (lt_min h1 (show (2 : ℝ) < 255.5 - 0 / 2 by norm_num))
    (show (2 : ℝ) < 128 - Real.logb 2 (Recursion.γe (ksR Coin) I x * cBe (ksR Coin) x) / 2 by linarith)
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem accI_lam_pos (x : Msg) : 0 < Recursion.lam (ksR Coin) (accI Coin) x := by
  refine ksR_lam_pos _ x (by norm_num [accI]) (by norm_num [accI]) ?_
  rw [ksR_γe_cBe]
  simp [accI, accV]
  norm_num

theorem exp_neg_four_le_sixteenth : Real.exp (-((32 : ℕ) : ℝ) / 8) ≤ 1 / 16 := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have h4 : (16 : ℝ) ≤ Real.exp 4 := by
    rw [show (4 : ℝ) = (4 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    calc (16 : ℝ) = 2 ^ 4 := by norm_num
      _ ≤ Real.exp 1 ^ 4 := by gcongr
  rw [show -((32 : ℕ) : ℝ) / 8 = -4 by norm_num, Real.exp_neg, one_div]
  exact inv_anti₀ (by norm_num) h4

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ksD_κ_lt_quarter (x : Msg) : Claims.Flock.κ (Flock.ksD Coin) x < 1 / 4 := by
  have he := exp_neg_four_le_sixteenth
  have hw : (2 : ℝ) ^ (-((4 : ℕ) : ℝ)) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]; norm_num
  unfold Claims.Flock.κ
  rw [Flock.ksD_k]
  simp only [Flock.ksD] at hw ⊢
  rw [hw]
  push_cast at he ⊢
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- Flock (`Flock.ksD`) as the inner protocol, at any decoder: `λ > 0` at `ksR`. -/
theorem ksD_lam_pos (decI : List (ℕ × Unit) → List Unit) (x : Msg) :
    0 < Recursion.lam (ksR Coin) (Inner.ofFlock (Flock.ksD Coin) decI) x := by
  refine ksR_lam_pos _ x (Flock.ksD_κ_pos []) (ksD_κ_lt_quarter []) ?_
  rw [ksR_γe_cBe]
  have hγ : (Inner.ofFlock (Flock.ksD Coin) decI).γ [] = 16 := Flock.ksD_γ []
  have hk : (Inner.ofFlock (Flock.ksD Coin) decI).k [] = 32 := Flock.ksD_k []
  rw [hγ, hk, show Real.sqrt 16 = 4 by rw [show (16 : ℝ) = 4 ^ 2 by norm_num]; exact Real.sqrt_sq (by norm_num)]
  simp [Inner.ofFlock, Flock.ksD, accV]
  norm_num

end Numbers

/-! ## Reruns of served programs -/

section Reruns

variable {R R₂ α β : Type}

theorem RP.bind_assoc' {γ : Type} (p : RP R α) (f : α → RP R β) (g : β → RP R γ) :
    (p.bind f).bind g = p.bind fun a => (f a).bind g := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [RP.bind, ih]
  | rerun r k ih => simp only [RP.bind, ih]

theorem serve_bind (q : R → RP R₂ Seen) (p : RP R α) (f : α → RP R β) :
    serve q (p.bind f) = (serve q p).bind fun a => serve q (f a) := by
  induction p with
  | ret a => rfl
  | hash m k ih => simp only [RP.bind, serve, ih]
  | rerun r k ih => simp only [RP.bind, serve, ih, RP.bind_assoc']

theorem RP.reruns_bind (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).2.2 = (p.eval H run).2.2 + ((f (p.eval H run).1).eval H run).2.2 := by
  induction p with
  | ret a => simp [RP.bind, RP.eval]
  | hash q k ih => simp only [RP.bind, RP.eval, ih]
  | rerun r k ih => simp only [RP.bind, RP.eval, ih]; omega

/-- A wait whose every rerun is seen rejected reruns at each of its draws. -/
theorem reruns_waitAcc_rej (H : Msg → Msg) (run : R → Trace) (f : Seen → Seen) (hf : ∀ v, (f v).out ≠ .acc)
    (k : ℕ) (hk : 0 < k) (rs : List R) : ((serve (viewing f) (waitAcc k rs [])).eval H run).2.2 = rs.length := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hk' : ¬ k ≤ ([] : List Seen).length := by simp only [List.length_nil]; omega
    simp only [waitAcc, hk', ite_false, serve, viewing, RP.bind, RP.eval, hf, List.length_cons, ih]

theorem reruns_firstWait_rej {Op W : Type} (H : Msg → Msg) (run : R → Trace) (f : Seen → Seen)
    (hf : ∀ v, (f v).out ≠ .acc) (k : ℕ) (hk : 0 < k) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) (m K : ℕ) (hm : 0 < m) (g : Fin m → Fin K → R) :
    K ≤ ((serve (viewing f) (firstWait k opens assemble check
      ((List.finRange m).map fun i => (List.finRange K).map (g i)))).eval H run).2.2 := by
  cases m with
  | zero => omega
  | succ m =>
    rw [List.finRange_succ, List.map_cons]
    simp only [firstWait]
    rw [serve_bind, RP.reruns_bind, reruns_waitAcc_rej H run f hf k hk, List.length_map, List.length_finRange]
    omega

theorem reruns_collect {α : Type} (H : Msg → Msg) (run : R → Trace) (n : ℕ) (hn : 0 < n)
    (g : Fin n → RP R (Option α)) :
    ((g ⟨0, hn⟩).eval H run).2.2 ≤ ((collect ((List.finRange n).map g)).eval H run).2.2 := by
  cases n with
  | zero => omega
  | succ n =>
    rw [List.finRange_succ, List.map_cons]
    simp only [collect]
    rw [RP.reruns_bind]
    exact Nat.le_add_right _ _

end Reruns

section Efficient

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [DecidableEq Col] in
/-- `E*` reruns at least as often as its outer extractor on the judged run. -/
theorem reruns_E_ge {W : Type} (H : Msg → Msg) (D : Deploy Coin) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (D.V.k x) → Coin) (c : (E H D I DO decO).C x) :
    ((EO D I DO decO).eval H P x ω r₀ c.1).2.2 ≤ ((E H D I DO decO).eval H P x ω r₀ c).2.2 := by
  unfold Extractor.eval
  show _ ≤ (RP.eval H (run H P D.V x ω) (serve _ (erep _ _ _ _ _ _ (r₀, c.1) c.2))).2.2
  simp only [erep, serve]
  rw [RP.reruns_bind]
  unfold outerStage
  rw [RP.reruns_bind]
  exact Nat.le_add_right_of_le (Nat.le_add_right _ _)

theorem nums1_k : nums1.k = 1 := by
  have h : 4 * (Real.exp 1 / 4) / Real.exp 1 = ((1 : ℕ) : ℝ) := by
    rw [div_eq_iff (Real.exp_pos 1).ne']; push_cast; ring
  simp only [Nums.k, nums1, h, Nat.ceil_natCast]

/-- At `effR`, the outer extractor reruns `3` times: the judged run, which accepts, then both draws of its one V*
session's wait, whose views are rejected. -/
theorem effR_EO_reruns (H : Msg → Msg) (decO : List (ℕ × Unit) → List Unit) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (r₀ : Fin ((effR Coin).V.k x) → Coin) (c : (EO (effR Coin) (accI Coin) (Flock.ksD Coin) decO).C x) :
    3 ≤ ((EO (effR Coin) (accI Coin) (Flock.ksD Coin) decO).eval H P x ω r₀ c).2.2 := by
  unfold Extractor.eval EO
  simp only [RP.eval]
  have hacc : (run H P (effR Coin).V x ω r₀).seen.out = .acc := run_accV_out H P x ω r₀
  simp only [hacc, ↓reduceIte]
  have hn : 0 < (effR Coin).nS x := Nat.one_pos
  refine Nat.add_le_add_right (le_trans ?_ (reruns_collect H _ _ hn _)) 1
  have hk : 0 < ((effR Coin).nums x 0).k := by rw [show (effR Coin).nums x 0 = nums1 from rfl, nums1_k]; omega
  exact reruns_firstWait_rej H _ _ (fun _ => by simp [effR]) _ hk _ _ _ _ 2 Nat.one_pos _

end Efficient

/-! ## `Sim` at `zkR` and `zvR` -/

section Sim

theorem prENNReal_le_one {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : PrENNReal p ≤ 1 := by
  classical
  unfold PrENNReal
  have hn : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  calc (∑ a, (Fintype.card α : ℝ≥0∞)⁻¹ * if p a then 1 else 0)
        ≤ ∑ _a : α, (Fintype.card α : ℝ≥0∞)⁻¹ :=
        Finset.sum_le_sum fun a _ => by split_ifs <;> simp
    _ = 1 := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ENNReal.mul_inv_cancel hn (ENNReal.natCast_ne_top _)]

/-- Every mask hides up to `δ₁ = 1`. -/
theorem hm96Hiding_one {Y B C : Type} [Fintype Y] [Nonempty Y] [Fintype B] (My : Y → B) (c : Y → C) :
    Hm96Hiding My c 1 := fun _ => (prENNReal_le_one _).trans le_add_self

variable {Coin : Type}

theorem simOuter_zero (SO : Msg → ℕ → OSim) (x : Msg) (xs : List Msg) (ms : List (Msg × Out)) :
    ∀ (n : ℕ) (c : (s : Fin n) → (SO x s).C), n = 0 → simOuter SO x xs c (List.finRange n) ms = .ret ms := by
  rintro n c rfl
  rfl

/-- `Sim` outputs the empty transcript wherever the inner protocol, the staging and the V* sessions take no round. -/
theorem sim_out_nil (D : Recursion.Deploy Coin) (hk : D.kS [] = 0) (hn : D.nS [] = 0) (SO : Msg → ℕ → OSim)
    {Θ : Type} (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (c : (Sim D (zeroI Coin) SO).C []) :
    (Sim D (zeroI Coin) SO).out H Vs θ [] c = [] := by
  unfold Simulator.out Sim
  simp only [simInner, simStage, zeroI, accI, simRounds, spBind, hk]
  rw [simOuter_zero SO [] _ [] _ c.2.2 hn]
  rfl

/-- `SimStop` outputs the empty transcript where `Sim` does, at a stop that never fires. -/
theorem simStop_out_nil (D : Recursion.StopDeploy Coin) (hk : D.kS [] = 0) (hn : D.nS [] = 0)
    (hs : ∀ H' ms, D.stops H' [] ms = false) (SO : Msg → ℕ → OSim) {Θ : Type} (H' H : Msg → Msg) (Vs : VStar Θ)
    (θ : Θ) (c : (SimStop D (zeroI Coin) SO H').C []) :
    (SimStop D (zeroI Coin) SO H').out H Vs θ [] c = [] := by
  have hh : ∀ n ms, D.halted H' n [] ms = false := fun n ms => by simp [StopDeploy.halted, hs]
  unfold Simulator.out SimStop
  simp only [zeroI, accI, simRounds, spBind, hk, hh, Nat.add_zero, Bool.false_eq_true, ↓reduceIte]
  rw [simOuter_zero SO [] _ [] _ c.2.2 hn]
  rfl

theorem zkStop_εzk (D : Recursion.StopDeploy Coin) (hk : D.kS [] = 0) (hn : D.nS [] = 0)
    (hz : ∀ i, D.nAt [] i = 0) (t : ℕ) (δ₁ : ℝ) : Recursion.εzkStop D (zeroI Coin) t δ₁ [] = 0 := by
  have : IsEmpty (Fin (D.nS [])) := by rw [hn]; exact Fin.isEmpty'
  unfold Recursion.εzkStop
  rw [Finset.univ_eq_empty, Finset.sum_empty]
  simp [zeroI, hk, hz]

theorem zk_εzk (D : Recursion.Deploy Coin) (hk : D.kS [] = 0) (hn : D.nS [] = 0) (t : ℕ) (δ₁ : ℝ) :
    Recursion.εzk D (zeroI Coin) t δ₁ [] = 0 := by
  have : IsEmpty (Fin (D.nS [])) := by rw [hn]; exact Fin.isEmpty'
  unfold Recursion.εzk
  rw [Finset.univ_eq_empty, Finset.sum_empty]
  simp [zeroI, hk]

theorem interact_one (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (hist : List (Msg × Msg)) :
    ZKMal.interact ask pm 1 hist = [(pm hist, ask (hist.map Prod.fst ++ [pm hist]))] := by
  simp only [ZKMal.interact]
  split <;> simp_all

/-- The transcript of one zero commit string under the salt `y`, against `V*`'s answers `ask`. -/
def zeroT (H : Msg → Msg) (ask : List Msg → Out) (y : Fin 1 → UInt8) : List (Msg × Out) :=
  [(List.ofFn (fun _ : Fin 64 => y 0) ++ H ([] ++ List.ofFn y),
    ask [List.ofFn (fun _ : Fin 64 => y 0) ++ H ([] ++ List.ofFn y)])]

theorem runStar_zeroP {Θ : Type} (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (hk : Vs.k x = 1)
    (ω : (Fin 1 → Fin 1 → UInt8) × Unit) :
    (runStar H (zeroP Coin) Vs x ω θ).msgs = zeroT H (fun as => ((Vs.next x as θ).eval H).1) (ω.1 0) := by
  rw [ZKMal.runStar_msgs, hk, interact_one]
  rfl

theorem sim_zvR {Θ : Type} (circ : Option Msg) (H : Msg → Msg) (SO : Msg → ℕ → OSim) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (c : (Sim (zvR Coin circ) (accI Coin) SO).C x) :
    (Sim (zvR Coin circ) (accI Coin) SO).out H Vs θ x c =
      zeroT H (fun as => ((Vs.next x as θ).eval H).1) (c.1 ⟨0, Nat.one_pos⟩) := by
  rfl

/-- At `zvR`, `zeroP` runs `Sim`'s one inner round, so its transcripts and `Sim`'s are alike, at every relation,
outer simulator and hiding level. -/
theorem zkmalR_zeroP (circ : Option Msg) (R : Rel Flock.Wit) (H : Msg → Msg) (SO : Msg → ℕ → OSim) (t : ℕ)
    (δ₁ : ℝ≥0) : ZKmalR H (zvR Coin circ).V.k (fun _ => zeroP Coin) R (Sim (zvR Coin circ) (accI Coin) SO)
      (Recursion.εzk (zvR Coin circ) (accI Coin) t δ₁) (Recursion.γzk (zvR Coin circ) t) := by
  intro x w _ Θ Vs θ hk Ev
  have e : Pr (fun ω => Ev (runStar H (zeroP Coin) Vs x ω θ).msgs) =
      Pr fun c => Ev ((Sim (zvR Coin circ) (accI Coin) SO).out H Vs θ x c) := by
    refine (Pr_congr fun ω => iff_of_eq (congrArg Ev (runStar_zeroP H Vs θ x hk ω))).trans ?_
    refine Eq.trans ?_ (Pr_congr fun c => iff_of_eq (congrArg Ev (sim_zvR circ H SO Vs θ x c))).symm
    let q := fun y : Fin 1 → UInt8 => Ev (zeroT H (fun as => ((Vs.next x as θ).eval H).1) y)
    refine (Pr_fst (α := Fin 1 → Fin 1 → UInt8) (β := Unit) fun y => q (y 0)).trans (Eq.trans ?_
      (Pr_fst (α := Fin ((accI Coin).V.k (xI (zvR Coin circ) x)) → Fin (zvR Coin circ).saltLen → UInt8)
        (β := (Fin ((zvR Coin circ).kS x) → Fin ((zvR Coin circ).nC x) → Fin (zvR Coin circ).saltLen → UInt8) ×
          ((s : Fin ((zvR Coin circ).nS x)) → (SO x s).C)) fun y => q (y ⟨0, Nat.one_pos⟩)).symm)
    exact ZKMal.Pr_equiv (α := Fin 1 → Fin 1 → UInt8)
      (β := Fin ((accI Coin).V.k (xI (zvR Coin circ) x)) → Fin (zvR Coin circ).saltLen → UInt8) (Equiv.refl _)
      fun y => q (y ⟨0, Nat.one_pos⟩)
  have hγ : Recursion.γzk (zvR Coin circ) t x = 0 := by simp [Recursion.γzk, zvR, ksR]
  have hε : 0 ≤ Recursion.εzk (zvR Coin circ) (accI Coin) t δ₁ x := by
    have : IsEmpty (Fin ((zvR Coin circ).nS x)) := Fin.isEmpty'
    unfold Recursion.εzk
    simp only [Finset.univ_eq_empty, Finset.sum_empty, zero_add]
    exact mul_nonneg (mul_nonneg zero_le_two
      (add_nonneg (Nat.cast_nonneg _) (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)))) δ₁.2
  calc _ = (0 : ℝ) := by simp only [e, sub_self, abs_zero]
    _ ≤ _ := by simp only [hγ, zero_mul, add_zero]; exact hε

end Sim

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The recursion's `KS` fails** at `ksR`, `accI` and `Flock.ksD` outside. -/
theorem KS_fails (H : Msg → Msg) : ¬ Recursion.KS H (ksR Coin) (accI Coin) (Flock.ksD Coin) := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP
  rw [ksR_prBad H _ _ [] (ksR_noR _ []), ksR_prB, ksR_prBe, ksR_prBr, ksR_κ] at h1
  norm_num [accI] at h1

/-- **The recursion's private `KS` fails** at `ksR`, `accI` and `Flock.ksD` outside. -/
theorem KSPriv_fails (H : Msg → Msg) : ¬ KSPriv H (ksR Coin) (accI Coin) (Flock.ksD Coin) := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP
  rw [ksR_prBad H _ _ [] (ksR_noRPriv _ []), ksR_prB, ksR_prBe, ksR_prBr, ksR_κ] at h1
  norm_num [accI] at h1

/-- **The recursion's `KS` over Flock fails** at `ksR`, `Flock.ksD` inside and outside. -/
theorem KSFlock_fails (H : Msg → Msg) : ¬ KSFlock H (ksR Coin) (Flock.ksD Coin) (Flock.ksD Coin) := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) []
    Unit silentP
  rw [ksR_prBad H _ _ [] (ksR_noR _ []), ksR_prB, ksR_prBe, ksR_prBr, ksR_κ] at h1
  have := Flock.ksD_κ (Coin := Coin) []
  simp only [mul_zero, Real.sqrt_zero, add_zero] at h1
  change 1 ≤ Claims.Flock.κ (Flock.ksD Coin) [] at h1
  linarith

/-- **The recursion's `KS` at the representation V\* binds fails** at `ksRAt` (`ksR`, its inner view running the
verifier it is given), `ksB` inside and `Flock.ksD` outside: the bound record's `κ` is `Flock.ksD`'s. -/
theorem KSBound_fails (H : Msg → Msg) : ¬ KSBound H (ksRAt Coin) (ksB Coin) (Flock.ksD Coin) := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) []
    Unit silentP
  try beta_reduce at h1
  rw [ksRAt_record, ksRV_prBad H _ _ _ [] (ksRV_noR _ _ []), ksRV_prB, ksRV_prBe, ksRV_prBr, ksRV_κ] at h1
  have := Flock.ksD_κ (Coin := Coin) []
  simp only [mul_zero, Real.sqrt_zero, add_zero] at h1
  change 1 ≤ Claims.Flock.κ (Flock.ksD Coin) [] at h1
  linarith

/-- **The recursion's private `KS` at the representation V\* binds fails** at `ksRAt`, `ksB` inside and `Flock.ksD`
outside. -/
theorem KSPrivBound_fails (H : Msg → Msg) : ¬ KSPrivBound H (ksRAt Coin) (ksB Coin) (Flock.ksD Coin) := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) []
    Unit silentP
  try beta_reduce at h1
  rw [ksRAt_record, ksRV_prBad H _ _ _ [] (ksRV_noRPriv _ _ []), ksRV_prB, ksRV_prBe, ksRV_prBr, ksRV_κ] at h1
  have := Flock.ksD_κ (Coin := Coin) []
  simp only [mul_zero, Real.sqrt_zero, add_zero] at h1
  change 1 ≤ Claims.Flock.κ (Flock.ksD Coin) [] at h1
  linarith

/-- `Pr[acc ∧ ¬R] ≤ 2^{−λ}` fails at `ksR`'s silent prover wherever `λ > 0`. -/
theorem bits_bound_fails {W : Type} (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (ksR Coin).V) (lam : ℝ)
    (hR : ∀ w, ¬ R [] w) (hlam : 0 < lam) :
    ¬ PrBad H (ksR Coin).V R E [] silentP ≤ (1 : ℕ) * (2 : ℝ) ^ (-lam) := by
  rw [ksR_prBad H R E [] hR, Nat.cast_one, one_mul]
  exact not_le.2 (Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith))

/-- **The recursion's `Bits` fails** at `ksR`, `accI` and `Flock.ksD` outside, where `λ = 1`. -/
theorem Bits_fails (H : Msg → Msg) : ¬ Recursion.Bits H (ksR Coin) (accI Coin) (Flock.ksD Coin) := fun h =>
  bits_bound_fails H _ _ _ (ksR_noR _ []) (accI_lam_pos []) <|
    h (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP 1 le_rfl (calls_silentP_accV H [])
      (strictBirthday_B_accV H silentP []) (ksR_expectedCR H _ _ _ silentP [])
      (ksR_strictBirthday_br H _ _ _ silentP [])

/-- **The recursion's `Bits` over Flock fails** at `ksR`, `Flock.ksD` inside and outside. -/
theorem BitsFlock_fails (H : Msg → Msg) : ¬ BitsFlock H (ksR Coin) (Flock.ksD Coin) (Flock.ksD Coin) := fun h =>
  bits_bound_fails H _ _ _ (ksR_noR _ []) (ksD_lam_pos _ []) <|
    h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP 1
      le_rfl (calls_silentP_accV H []) (strictBirthday_B_accV H silentP []) (ksR_expectedCR H _ _ _ silentP [])
      (ksR_strictBirthday_br H _ _ _ silentP [])

/-- **The recursion's `Efficient` fails** at `effR`, `accI` and `Flock.ksD` outside: `E*` reruns `3` times on every
coin, above the expected bound `N_I · (1 + m′·k) = 2`. -/
theorem Efficient_fails (H : Msg → Msg) :
    ¬ Recursion.Efficient H (effR Coin) (accI Coin) (Flock.ksD Coin) fun _ => [()] := by
  intro h
  have h1 := h.2 [] Unit silentP ()
  have h3 : (3 : ℝ) ≤ Ex fun s : (Fin ((effR Coin).V.k []) → Coin) ×
      (E H (effR Coin) (accI Coin) (Flock.ksD Coin) fun _ => [()]).C [] =>
      (((E H (effR Coin) (accI Coin) (Flock.ksD Coin) fun _ => [()]).eval H silentP [] () s.1 s.2).2.2 : ℝ) := by
    rw [← Ex_const (α := (Fin ((effR Coin).V.k []) → Coin) ×
      (E H (effR Coin) (accI Coin) (Flock.ksD Coin) fun _ => [()]).C []) 3]
    exact Ex_mono fun s => by
      exact_mod_cast le_trans (effR_EO_reruns H _ silentP [] () s.1 s.2.1) (reruns_E_ge H _ _ _ _ silentP [] () s.1 s.2)
  have hs : ∀ s : Fin ((effR Coin).nS []), ((effR Coin).nums [] s).waits * ((effR Coin).nums [] s).k = 1 :=
    fun _ => by show nums1.waits * nums1.k = 1; rw [nums1_k]; rfl
  have hb : (NI (effR Coin) (accI Coin) [] : ℝ) = 1 := by norm_num [NI, accI]
  beta_reduce at h1
  rw [Finset.sum_congr rfl fun s _ => hs s, Finset.sum_const, Finset.card_univ, Fintype.card_fin, hb] at h1
  norm_num [effR] at h1
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's `ScopeMatches` fails** at `ksR`, whose scope is empty and whose check takes everything. -/
theorem ScopeMatches_fails : ¬ Recursion.ScopeMatches (ksR Coin) (accI Coin) := fun h => by
  obtain ⟨s, hs, -⟩ := (h []).2 rfl
  simp [ksR] at hs

omit [Nonempty Coin] in
/-- **The recursion's `Complete` fails** at `compR`, whose reference prover is silent, at its error `δ_c = 0`. -/
theorem Complete_fails : ¬ Recursion.Complete constH (compR Coin) (accI Coin) :=
  Witness.Complete_fails (compR Coin).P _ _ ((compR Coin).δc constH) (x := []) (w := ⟨[]⟩)
    ⟨recStmt, rfl, trivial, recStmt_inScope⟩ ⟨recStmt, rfl, rfl, trivial⟩ (by norm_num [compR, ksR])

omit [Fintype Coin] [Nonempty Coin] in
/-- **The recursion's `Nontrivial` fails** at `compR`, which rejects the recorded run. -/
theorem Nontrivial_fails (H : Msg → Msg) : ¬ Recursion.Nontrivial H (compR Coin) [⟨[], [], fun _ => default⟩] :=
  fun h => Witness.Nontrivial_fails H h.1

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's `ZK` fails** at a masked outer Flock, at `zkZR`, whose reference prover is quiet, at every
record's outer simulators, whatever hm96 hides: `zkR`'s compiled verifier takes a round that `Sim` doesn't simulate,
as the inner protocol, the staging and the V* sessions take none, so the real transcript is never the simulated one,
at an error of `0`. -/
theorem ZK_fails (H : Msg → Msg) (SO : ℕ → Msg → ℕ → OSim) :
    ¬ Recursion.ZK H (zkZR Coin SO) (zeroI Coin) (zkO Coin) := fun h =>
  ZKMal.not_zkmalR_of_out (N := Unit) (Ω := Unit) H _ _ [] _ _ _ [] ⟨[]⟩ ⟨recStmt, rfl, rfl, trivial⟩ Nat.one_pos
    (by rw [zk_εzk _ rfl rfl]; norm_num) (fun c => by rw [sim_out_nil _ rfl rfl]; nofun)
    ((ZKMal.zkmalRx_const_of H _ (zkZR Coin SO).prover [] (fun _ _ => rfl) _ _ _ _).1
      (h.2 1 Nat.one_pos 1 (hm96Hiding_one _ _)))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's private `ZK` fails** at a masked outer Flock, at `zkZPR`, whose reference prover is quiet, at
every record's outer simulators, whatever hm96 hides, as `ZK` does. -/
theorem ZKPriv_fails (H : Msg → Msg) (SO : ℕ → Msg → ℕ → OSim) :
    ¬ ZKPriv H (zkZPR Coin SO) (zeroI Coin) (zkO Coin) := fun h =>
  ZKMal.not_zkmalR_of_out (N := Unit) (Ω := Unit) H _ _ [] _ _ _ [] ⟨[]⟩
    ⟨{ recStmt with circ := some [] }, rfl, rfl, trivial⟩
    Nat.one_pos
    (by rw [zk_εzk _ rfl rfl]; norm_num) (fun c => by rw [sim_out_nil _ rfl rfl]; nofun)
    ((ZKMal.zkmalRx_const_of H _ (zkZPR Coin SO).prover [] (fun _ _ => rfl) _ _ _ _).1
      (h.2 1 Nat.one_pos 1 (hm96Hiding_one _ _)))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's `ZK` with the prover's stop fails** at a masked outer Flock, at `zkSR`, whose stop never fires
and whose own reference prover is quiet, at every record's outer simulators, whatever hm96 hides, as `ZK` does:
`SimStop` runs as `Sim` does. -/
theorem ZKStop_fails (H : Msg → Msg) (SO : ℕ → Msg → ℕ → OSim) :
    ¬ ZKStop H (zkSR Coin SO) (zeroI Coin) (zkO Coin) := fun h =>
  ZKMal.not_zkmalR_of_out (N := Unit) (Ω := Unit) H _ _ [] _ _ _ [] ⟨[]⟩ ⟨recStmt, rfl, rfl, trivial⟩ Nat.one_pos
    (by rw [zkStop_εzk (zkSR Coin SO) rfl rfl fun _ => rfl]; norm_num)
    (fun c => by rw [simStop_out_nil (zkSR Coin SO) rfl rfl (fun _ _ => rfl)]; nofun)
    ((ZKMal.zkmalRx_const_of H _ (zkSR Coin SO).prover [] (fun _ _ => rfl) _ _ _ _).1
      (h.2 1 Nat.one_pos 1 (hm96Hiding_one _ _)))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's `ZK` holds at a record whose own reference prover ignores its witness**: at `zvZR`, where every
witness is in the relation, at a masked outer Flock, every record's outer simulators and every hiding level
(`zkmalR_zeroP`). -/
theorem ZK_zeroRecord (H : Msg → Msg) (SO : ℕ → Msg → ℕ → OSim) :
    Recursion.ZK H (zvZR Coin none SO) (accI Coin) (zkO Coin) :=
  ⟨rfl, fun t _ δ₁ _ => (ZKMal.zkmalRx_const_of H _ (zvZR Coin none SO).prover [] (fun _ _ => rfl) _ _ _ _).2
    (zkmalR_zeroP (Coin := Coin) none _ H (SO t) t δ₁)⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The recursion's private `ZK` holds at a record whose own reference prover ignores its witness**, as `ZK` does
(`zkmalR_zeroP`). -/
theorem ZKPriv_zeroRecord (H : Msg → Msg) (SO : ℕ → Msg → ℕ → OSim) :
    ZKPriv H (zvZR Coin (some []) SO) (accI Coin) (zkO Coin) :=
  ⟨rfl, fun t _ δ₁ _ => (ZKMal.zkmalRx_const_of H _ (zvZR Coin (some []) SO).prover [] (fun _ _ => rfl) _ _ _ _).2
    (zkmalR_zeroP (Coin := Coin) (some []) _ H (SO t) t δ₁)⟩

/-- **The inner protocol's `KS` fails** at `noI`, which accepts and whose relation is empty. -/
theorem Inner.KS_fails (H : Msg → Msg) : ¬ (noI Coin).KS H :=
  Witness.KS_fails (Coin := Coin) H (noI Coin).R (noI Coin).E _ _ [] (fun _ h => h) (by norm_num [noI, accI])

/-- **The inner protocol's `KS` against provers that may abort fails** at `noI`, at a prover that never aborts. -/
theorem Inner.KSAbort_fails (H : Msg → Msg) : ¬ (noI Coin).KSAbort H := by
  intro h
  have h1 := h [] Unit silentP Unit fun _ _ _ => true
  have hp : (noI Coin).PrBadAbort (A := Unit) H [] silentP (fun _ _ _ => true) = 1 := by
    unfold Inner.PrBadAbort
    refine (Pr_congr fun s => iff_true_intro ⟨?_, ?_⟩).trans ZKMal.Pr_true
    · simp only [abortRun, ite_true]
      exact run_accV_out H silentP [] s.1 s.2.1.1
    · generalize ((erep _ _ _ _ _ _ _ _).eval _ _).1 = o
      cases o <;> exact id
  have hb : PrBAbort (A := Unit) H (noI Coin).V silentP [] (fun _ _ _ => true) = 0 := by
    unfold PrBAbort
    split_ifs with hk
    · unfold Finder.Pr
      rw [Pr_congr fun s => iff_false_intro ?_]
      · exact Pr_false
      · rintro ⟨u, v, hs, -⟩
        simp [BAbort, abortRun, noI, accI, run_accV_vLog, findColl] at hs
    · rfl
  rw [hp, hb] at h1
  norm_num [noI, accI] at h1

end Claims

end Claims.Witness.Recursion
