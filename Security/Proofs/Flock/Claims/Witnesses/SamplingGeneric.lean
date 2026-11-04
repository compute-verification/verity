module

public import Proofs.Flock.SamplingGeneric
public import Proofs.Flock.Claims.Witnesses.Sampling
public import Proofs.Flock.Claims.Witnesses.BindingMeets
public import Proofs.Flock.Claims.ERepCost
public import Proofs.Flock.Claims.ZK.Compose
public import Proofs.Flock.Claims.ZKOut
public import Proofs.Flock.Claims.Sampling.Assembled
public import Proofs.Flock.Masked

/-!
# Witnesses: the sampled compiler's statements (`Claims.SamplingGeneric`)

The compiler's statements (`CompilerKS`, `CompilerKSReg`, `CompilerKSCount`, `CompilerBits`, `CompilerEfficient`) and
`ZKOut`'s leak form (`Claims.ZKOutLeak`) each get an `honest` witness, their premises at the honest Flock instance, and
a `nontrivial` one, their conclusion false at a verifier that accepts everything.

* **The honest instance.** The proof system for proof blocks is C-Flock's deployed masked record at the deployed
  relation (`flockA`: `Arg.ofFlock` at `FlockDeploy.deployZk Flock.Readable.relation` and its own decoder), knowledge
  sound by `Flock.SecurityProofs.MaskedKS` within its decoder's radius (`listWithin_deployZk`), its reruns `E^rep`'s
  (`ERep_cost`), expected ones at most its strict `1 + m′·K` (`flockE`: its expected `1 + m′·k` reads `N₀`, which
  Flock's shape doesn't carry), and its own SHA-512 calls none (`flock_ownLe`). The tree has no deployed compiled
  sampled verifier (the sampling record's code, `SamplingDeploy.Code`, is abstract), so the compiled verifier is one
  proof block whose session is the whole run (`oneL`): the statement `toyStmt` (one input gate, returned, in scope),
  every round in the second stage, its error bounds the proof system's errors at the run's statement, and the session's
  witness revealing one opening at the root. Its commitment is the deployed form (`cm₀`: hm96 leaves of `sha512/row/v2`
  rows under the registered frame-v3-sha512 path check, at one port of one leaf), which binds (`pathCommit_binds`) and
  opens the revealed opening at SHA-512 (`cm₀_opens`), so it meets the contract (`oneL_sessions`), and checks an
  opening in at most `4 + depth` calls (`cm₀_calls`: `opens_calls_le` at the sampled verifier's code `code₀` whose
  commitment it is). `compilerKS_honest` (also `CompilerKSReg`'s and `CompilerKSCount`'s), `compilerBits_honest`,
  `compilerEfficient_honest`.
* **The accept-all instance.** `acceptAllArg` accepts after one round, its relation is empty and its errors `0`, and
  `acceptAllL` is `oneL` over it with a commitment no opening passes (`rejCm`). Every premise but the proof system's
  knowledge soundness holds there (`acceptAll_drops`: rounds, binding, the contract), so each conclusion's failure is
  the work `Arg.KS` does: `compilerKS_nontrivial` (at `δ = 0`), `compilerKSCount_nontrivial` (at `t = 1`),
  `compilerBits_nontrivial` (`λ > 0` there), `compilerKSReg_nontrivial` (the committed trace opens nothing, so the
  proof block is in `Freg` at every run while `ε_reg = 0`).
* **`CompilerEfficient`.** Nontrivial: a proof system whose extractor finds a witness at once, with no rerun or call
  (`findArg`), compiled by `oneL` under a commitment whose opening check makes one SHA-512 call that the compiled
  verifier counts as none (`effL`). Every premise but the opening checks' count holds there (`compilerEfficient_drops`),
  and as every run is accepted the compiled extractor reaches an extraction whose check calls (`ownLog_E_pos`), above
  the own-call bound `0` (`compilerEfficient_nontrivial`).
* **`ZKOutLeak`.** Honest: `ZKOut` at C-Flock's deployed masked session (`zkFlock`, `MaskedZK`'s record at one coin
  per rep round and `t = 1`), its outcome trivial, from `ZKmalR`'s two-sided bound (`zkOut_of_zkmalR`). Nontrivial: a
  prover that talks against a verifier that accepts everything, with a silent simulator and no error
  (`zkOutLeak_nontrivial`), where `ZKOut` fails (`zkOut_dropped`).
* **`CompilerZK`.** Honest: C-Flock's masked session as a proof system with a public outcome (`zkA`, zero knowledge by
  `MaskedZK`) under a masked compiled verifier of one session (`zkL`), which meets the contract
  (`compilerZK_honest`). Its commitment opens nothing, so its sampling relation is empty and the contract's `witness`
  holds vacuously there: the tree states no member of C-Flock's relation for a session's witness to be. Nontrivial: the
  same shape over a proof system whose prover talks and whose simulator is silent (`talkArg`, `talkL`), every opening
  passing, so the toy witness is in the sampling relation; it meets the contract (`compilerZK_drops`) and its
  conclusion fails (`compilerZK_nontrivial`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims Claims.SamplingGeneric Claims.Witness Claims.Witness.Sampling
open Claims.Sampling (AuditStatement Opening CommitmentScheme InScope window Bext lamLin slack Prog BGate
  checkRevealed flatMapRP waitAccC forkOn assemble ownLog pathCommit hm96Leaf regPath sha512Msg bytes
  ocBind_eval regPath_sha512 hm96Leaf_sha512 hm96Leaf_binds regPath_binds pathCommit_binds opensLeaf_one)
open Claims.Recursion (spBind spUnder)
open Properties.NetworkCertifier.Bytes (bs)
open FlockSoundness.Refine FlockSoundness.Binding

/-! ## The toy statement's one proof block -/

theorem toy_refIn (val : ℕ → Option Bool) (ι : ℕ → Bool) : toyProg.refIn [0] val ι 0 = val 0 := by
  simp [Prog.refIn, toy_boundary]

theorem toy_correct_of {val : ℕ → Option Bool} {b : Bool} (hb : val 0 = some b) (ι : ℕ → Bool) :
    toyProg.Correct (toyProg.refIn [0] val ι) [0] := by
  intro g hg
  obtain rfl := List.mem_singleton.1 hg
  refine ⟨fun x hx i hi _ => ?_, fun _ => ⟨b, ?_, ?_⟩⟩
  · simp only [toyProg, List.getElem?_cons_zero, Option.some.injEq] at hx
    subst hx
    simp [BGate.reads] at hi
  · rw [toy_refIn, hb]
  · rw [toy_runIn, toy_refIn, hb]

theorem toy_val_of_correct {val : ℕ → Option Bool} {ι : ℕ → Bool}
    (h : toyProg.Correct (toyProg.refIn [0] val ι) [0]) : ∃ b, val 0 = some b := by
  obtain ⟨b, hb, _⟩ := (h 0 (List.mem_singleton.2 rfl)).2 toy_boundary
  exact ⟨b, (toy_refIn val ι).symm.trans hb⟩

/-! ## A compiled verifier of one proof block -/

/-- **A compiled sampled verifier of one proof block** over `A`: the statement `toyStmt`, every round in the second
stage, and the block always sampled, its session the whole run at the run's own statement, its error bounds `A`'s
errors there; the session's witness reveals `rev` under the commitment `cm`, each checked in at most `calls` SHA-512
calls. -/
def oneL {Coin : Type} (A : Arg Coin) (cm : CommitmentScheme Bool) (rev : List (List Msg × Opening Bool)) (calls : ℕ) :
    Compiled Coin A where
  V := A.V
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := cm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh x _ := A.shape x
  κb x _ := A.κ x
  γb x _ := A.γ x
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := rev
  openCalls _ := calls
  Tape := Unit
  P _ := silentP
  δc _ _ := 1

section OneBlock

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The one-block compiled verifier meets the contract** wherever every witness in `A`'s relation reveals an opening
at the root that passes, of the input's position. -/
theorem oneL_sessions (A : Arg Coin) (cm : CommitmentScheme Bool) (rev : List (List Msg × Opening Bool)) (calls : ℕ)
    (H : Msg → Msg) (hrev : ∀ xS w, A.R xS w → ∃ q ∈ rev, cm.Opens H q.1 q.2 = true ∧ q.1 = [] ∧ q.2.pos = 0) :
    (oneL A cm rev calls).Sessions H where
  draws := by
    intro x s P hs hP _ B _
    have hs' : toyStmt = s := Option.some.inj hs
    have hP' : toyPart = P := Option.some.inj hP
    subst hs' hP'
    by_cases h0 : 0 ∈ B
    · refine ⟨fun _ => 0, fun _ => le_rfl, fun r => ?_, ?_⟩
      · rw [Pr_congr fun d => iff_false_intro (by intro h; exact h 0 (List.mem_singleton_self 0) h0)]
        exact Pr_false.le
      · rw [Ex_const, toy_Pa]
        split_ifs <;> norm_num
    · have hw : window 0 1 B = ∅ := by
        unfold window
        rw [Finset.image_eq_empty, Finset.filter_eq_empty_iff]
        rintro v hv ⟨-, h1⟩
        obtain rfl : v = 0 := by omega
        exact h0 hv
      exact ⟨fun _ => 1, fun _ => zero_le_one, fun _ => Pr_le_one _, by rw [Ex_const, toy_Pa, ite_eq_left hw]⟩
  slots _ _ := le_rfl
  refuses _ _ _ _ _ _ := ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩
  interior _ _ _ _ _ _ _ := rfl
  fixed _ _ _ _ := ⟨Finset.Subset.refl _, fun _ => rfl⟩
  runs := by
    intro x Ω P ω r u _
    have hc : A.V.k x = ((oneL A cm rev calls).rounds x u).card := by
      show A.V.k x = (Finset.univ : Finset (Fin (A.V.k x))).card
      rw [Finset.card_univ, Fintype.card_fin]
    have hco : ∀ r', (oneL A cm rev calls).coinsOf x u x r' = r' := fun r' =>
      (oneL A cm rev calls).coinsOf_eq_of_strictMono (xS := x) (f := fun i => i) (fun _ _ h => h)
        (fun t => ⟨fun _ => ⟨t, rfl⟩, fun _ => Finset.mem_univ _⟩) r'
    have hrun : ∀ r', run H (P.comap fun _ : Unit => ω) A.V x () r' = run H P A.V x ω r' := fun r' =>
      Claims.ZKMal.stepWith_comap H P (fun _ => ω) x () _ _ _ _ _
    refine ⟨x, P.comap fun _ => ω, rfl, le_rfl, le_rfl, hc, fun r' _ => ?_, fun r' _ => ?_⟩
    · rw [hco r', hrun r']
      rfl
    · rw [hco r', hrun r']
      exact ⟨rfl, fun _ hq => hq⟩
  opens := by
    intro x s P Ω Pv ω r u xS vS w hs hP _ _ hu _ hR
    have hs' : toyStmt = s := Option.some.inj hs
    have hP' : toyPart = P := Option.some.inj hP
    subst hs' hP'
    obtain rfl : u = 0 := List.mem_singleton.1 hu
    obtain ⟨q, hq, hqo, hqr, hqp⟩ := hrev xS w hR
    refine ⟨[0], rfl, fun val ι hv => ?_⟩
    have h0 := (hv q hq hqo).1 hqr
    rw [hqp] at h0
    exact toy_correct_of h0 ι

end OneBlock

/-! ## The honest instance: C-Flock's masked record, compiled -/

/-- C-Flock's deployed masked record at the deployed relation. -/
noncomputable def flockD := FlockDeploy.deployZk _root_.Flock.Readable.relation

/-- **The honest proof system for proof blocks**: C-Flock's deployed masked record, `E^rep` at its own decoder. -/
noncomputable def flockA : Arg FlockDeploy.Coin := Arg.ofFlock flockD flockD.decode

/-- The revealed opening: the input's value at position `0`. -/
def opening₀ : Opening Bool := ⟨0, true, []⟩

/-- Every position's bit at the first place of a one-bit row. -/
def place₀ : ℕ → (w : ℕ) × Fin w := fun _ => ⟨1, 0⟩

/-- No other bit of the row. -/
def bits₀ : Msg → ℕ → Bool := fun _ _ => false

/-- The empty salt. -/
def salt₀ : Opening Bool → Msg := fun _ => []

/-- The registered port of one leaf, at the root `opening₀`'s hm96 leaf climbs to with no siblings. -/
noncomputable def port₀ : _root_.Flock.Registered.Port :=
  { name := "", value := "",
    root := _root_.Flock.FV3.Sha512.climb ByteArray.empty (_root_.Flock.Registered.rowLeaf ByteArray.empty 0
      (bytes (digBytes (H512 (hm96Input (SamplingDeploy.row place₀ bits₀ opening₀) (salt₀ opening₀)))))) 0 0
      (bytes []),
    leaves := 1, words := none, wordBits := 16, domain := ByteArray.empty, readsRel := #[] }

/-- **The deployed commitment form**: hm96 leaves of `sha512/row/v2` rows under the registered frame-v3-sha512 path
check, every position at `port₀`'s one leaf with an empty path. -/
noncomputable def cm₀ : CommitmentScheme Bool :=
  pathCommit (fun _ _ => some (bs port₀.root, 0)) (hm96Leaf (SamplingDeploy.row place₀ bits₀) salt₀) (fun _ => [])
    (regPath #[port₀]) (fun _ _ _ => [])

theorem sha512_eq : FlockDeploy.sha512 = sha512Msg := rfl

theorem cm₀_opens : cm₀.Opens sha512Msg [] opening₀ = true := by
  unfold CommitmentScheme.Opens
  simp only [cm₀, pathCommit, ocBind_eval, hm96Leaf_sha512]
  rw [regPath_sha512]
  exact ⟨port₀, by simp, opensLeaf_one ByteArray.empty _⟩

theorem cm₀_binds (H : Msg → Msg) : cm₀.Binds H :=
  pathCommit_binds (hm96Leaf_binds H (SamplingDeploy.row_places place₀ bits₀) salt₀) (regPath_binds H #[port₀])

/-- The sampled verifier's code whose commitment is `cm₀`, one opening revealed per session. -/
noncomputable def code₀ : SamplingDeploy.Code where
  V := flockD.V
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  stage2 _ := Finset.univ
  checked _ _ := [0]
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  reveal _ _ _ _ := [([], opening₀)]
  own := #[port₀]
  at_ _ _ := some (bs port₀.root, 0)
  layer _ _ _ := []
  place := place₀
  bits := bits₀
  salt := salt₀
  π _ := []
  innerN₀ _ := 0
  tables _ := 0
  reveals _ := 1
  sessions _ := 1
  step _ _ := .say []
  reads _ _ := []
  wit _ _ _ := ⟨[]⟩

/-- One opening's check under `cm₀` makes at most the leaf's three calls, its frame's and one per level of the port. -/
theorem cm₀_calls (H : Msg → Msg) (roots : List Msg) (o : Opening Bool) :
    ((cm₀.opens roots o).eval H).2.length ≤ 4 + SamplingDeploy.depth #[port₀] :=
  SamplingDeploy.opens_calls_le H code₀ roots o

/-- **The honest compiled verifier**: one proof block over C-Flock's masked record, its session's witness revealing
`opening₀`, checked in hm96's three calls and the path check's. -/
noncomputable def flockL : Compiled FlockDeploy.Coin flockA :=
  oneL flockA cm₀ [([], opening₀)] (4 + SamplingDeploy.depth #[port₀])

/-- `E^rep`'s strict reruns at a shape, `1 + m′·K`. -/
def flockρ (s : ℕ × ℕ × ℕ) : ℕ := 1 + s.1 * s.2.1

/-- A bound on `E^rep`'s expected reruns at a shape: its strict `1 + m′·K`. Its expected `1 + m′·k` reads `N₀`
(`k = ⌈4·N₀/e⌉`), which the shape doesn't carry. -/
noncomputable def flockE (s : ℕ × ℕ × ℕ) : ℝ := (flockρ s : ℝ)

theorem flockL_sessions : flockL.Sessions FlockDeploy.sha512 :=
  oneL_sessions flockA cm₀ [([], opening₀)] _ FlockDeploy.sha512 fun _ _ _ =>
    ⟨([], opening₀), List.mem_singleton_self _, by rw [sha512_eq]; exact cm₀_opens, rfl, rfl⟩

theorem flockL_openCalls : flockL.OpenCallsLe FlockDeploy.sha512 := fun _ _ _ _ _ _ => by
  show ([([], opening₀)].map fun p => ((cm₀.opens p.1 p.2).eval FlockDeploy.sha512).2.length).sum ≤
    4 + SamplingDeploy.depth #[port₀]
  rw [List.map_singleton, List.sum_singleton]
  exact cm₀_calls _ _ _

theorem flock_reruns : flockA.RerunsLe FlockDeploy.sha512 flockρ ∧ flockA.RerunsExpLe FlockDeploy.sha512 flockE := by
  have h : flockA.RerunsLe FlockDeploy.sha512 flockρ :=
    (Claims.ERep_cost FlockDeploy.sha512 flockD.V flockD.waits flockD.k flockD.cap (fun v => [flockD.cols v])
      (Claims.Flock.assemble flockD.parse flockD.table flockD.decode flockD.read) (Claims.Flock.check flockD.parse)).1
  refine ⟨h, fun x Ω P ω => ?_⟩
  have hs : ∀ s : (Fin (flockA.V.k x) → FlockDeploy.Coin) × flockA.E.C x,
      ((Extractor.eval FlockDeploy.sha512 P flockA.E x ω s.1 s.2).2.2 : ℝ) ≤ flockE (flockA.shape x) := fun s => by
    have h1 : (Extractor.eval FlockDeploy.sha512 P flockA.E x ω s.1 s.2).2.2 ≤ flockρ (flockA.shape x) :=
      h x Ω P ω s.1 s.2
    unfold flockE
    exact_mod_cast h1
  exact (Ex_mono hs).trans (Ex_const _).le

/-- A program with no SHA-512 call of its own, in `Claims.Sampling`'s form, has none in this one's. -/
theorem noHash_of_sampling {R α : Type} {p : RP R α} (h : Claims.Sampling.NoHash p) : NoHash p := by
  induction h with
  | ret a => trivial
  | rerun r k _ ih => exact ih

/-- `E^rep` makes no SHA-512 call of its own: its waits rerun, and `assemble` and `check` are pure. -/
theorem noHash_eRep {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] (V : Verifier Coin) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (c : (ERep V m k K opens assemble check).C x) :
    NoHash ((ERep V m k K opens assemble check).ext x r₀ c) := by
  intro v₀
  show NoHash (if v₀.out = .acc then _ else _)
  split_ifs
  · exact noHash_of_sampling (Claims.Sampling.NoHash.firstWait _ _ _ _ _)
  · trivial

theorem noHash_flockE (x : Msg) (r₀ : Fin (flockD.V.k x) → FlockDeploy.Coin)
    (c : (Claims.Flock.E flockD flockD.decode).C x) : NoHash ((Claims.Flock.E flockD flockD.decode).ext x r₀ c) :=
  noHash_eRep flockD.V flockD.waits flockD.k flockD.cap _ _ _ x r₀ c

theorem flock_ownLe : flockA.OwnLe FlockDeploy.sha512 fun _ => 0 := by
  intro x Ω P ω r₀ c
  exact (congrArg List.length (ownLog_noHash _ _ _ (noHash_flockE x r₀ c))).le

/-- **`CompilerKS`'s premises at the honest instance** (also `CompilerKSReg`'s and `CompilerKSCount`'s): C-Flock's
masked record's rounds are its shape's and it is knowledge sound, the deployed commitment form binds, and the one-block
compiled verifier meets the contract. -/
theorem compilerKS_honest : flockA.WF ∧ flockA.KS FlockDeploy.sha512 ∧ flockL.commit.Binds FlockDeploy.sha512 ∧
    flockL.Sessions FlockDeploy.sha512 :=
  ⟨arg_ofFlock_wf _ _,
    flockArgKS FlockDeploy.sha512 flockD flockD.decode _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _),
    cm₀_binds _, flockL_sessions⟩

/-- **`CompilerBits`' premises at the honest instance**: `CompilerKS`'s, with `E^rep`'s expected reruns, no SHA-512
call of its own, and the opening check's calls. -/
theorem compilerBits_honest : flockA.WF ∧ flockA.KS FlockDeploy.sha512 ∧ flockL.commit.Binds FlockDeploy.sha512 ∧
    flockL.Sessions FlockDeploy.sha512 ∧ flockA.RerunsExpLe FlockDeploy.sha512 flockE ∧
    flockA.OwnLe FlockDeploy.sha512 (fun _ => 0) ∧ flockL.OpenCallsLe FlockDeploy.sha512 :=
  ⟨compilerKS_honest.1, compilerKS_honest.2.1, compilerKS_honest.2.2.1, compilerKS_honest.2.2.2, flock_reruns.2,
    flock_ownLe, flockL_openCalls⟩

/-- **`CompilerEfficient`'s premises at the honest instance**: the contract, `E^rep`'s strict and expected reruns and
own calls, and the opening check's calls. -/
theorem compilerEfficient_honest : flockL.Sessions FlockDeploy.sha512 ∧ flockA.RerunsLe FlockDeploy.sha512 flockρ ∧
    flockA.RerunsExpLe FlockDeploy.sha512 flockE ∧ flockA.OwnLe FlockDeploy.sha512 (fun _ => 0) ∧
    flockL.OpenCallsLe FlockDeploy.sha512 :=
  ⟨flockL_sessions, flock_reruns.1, flock_reruns.2, flock_ownLe, flockL_openCalls⟩

/-! ## The accept-all instance -/

/-- **A proof system that accepts everything**: one round, accepted; no witness; errors `0`; an extractor that finds
nothing. -/
def acceptAllArg : Arg Unit where
  Sh := Unit
  shape _ := ()
  V := accV Unit
  rounds _ := 1
  W := Unit
  R _ _ := False
  EC _ := Unit
  ext _ _ _ := .ret none
  κ _ := 0
  γ _ := 0

/-- The one-block compiled verifier over it, no opening passing. -/
def acceptAllL : Compiled Unit acceptAllArg := oneL acceptAllArg rejCm [] 0

section Numbers

theorem maxBelow_zero (n : ℕ) : maxBelow n (fun _ => 0) = 0 := by
  unfold maxBelow
  generalize List.range n = l
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, List.foldr_cons, ih, max_self]

theorem maxBelowN_zero (n : ℕ) : maxBelowN n (fun _ => 0) = 0 := by
  unfold maxBelowN
  generalize List.range n = l
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, List.foldr_cons, ih, max_self]

theorem acceptAllL_scope (x : Msg) : acceptAllL.Scope x := ⟨toyStmt, toyPart, rfl, rfl, inScope_toy⟩

theorem acceptAllL_nPU (x : Msg) : acceptAllL.nPU x = 1 := rfl

theorem acceptAllL_k (x : Msg) : acceptAllL.k x = 2 ^ 40 * 3 := rfl

theorem acceptAllL_κs (x : Msg) : acceptAllL.κs x = 0 := by
  unfold Compiled.κs
  exact maxBelow_zero _

theorem acceptAllL_γs (x : Msg) : acceptAllL.γs x = 0 := by
  unfold Compiled.γs
  refine (congrArg (maxBelow _) (funext fun u => ?_)).trans (maxBelow_zero _)
  exact zero_div _

theorem acceptAllL_γ (x : Msg) : acceptAllL.γ x = 0 := by
  unfold Compiled.γ
  rw [acceptAllL_γs, mul_zero, mul_zero]

theorem acceptAllL_outer (x : Msg) :
    2 * (acceptAllL.k x : ℝ) / (acceptAllL.cap x : ℝ) + Real.exp (-(acceptAllL.k x : ℝ) / 8) ≤ 1 / 50 := by
  have hk : (acceptAllL.k x : ℝ) = 2 ^ 40 * 3 := by rw [acceptAllL_k]; push_cast; ring
  have hc : (acceptAllL.cap x : ℝ) = 2 ^ 128 * (2 ^ 40 * 3) := by
    simp only [Compiled.cap, acceptAllL_k]; push_cast; ring
  rw [hk, hc, neg_div]
  have he := exp_neg_le_inv (show (0 : ℝ) ≤ 2 ^ 40 * 3 / 8 by positivity)
  have h1 : (2 ^ 40 * 3 / 8 + 1 : ℝ)⁻¹ ≤ 1 / 100 := by
    rw [inv_le_comm₀ (by positivity) (by norm_num)]; norm_num
  have h2 : 2 * (2 ^ 40 * 3 : ℝ) / (2 ^ 128 * (2 ^ 40 * 3)) ≤ 1 / 100 := by
    rw [div_le_iff₀ (by positivity)]; norm_num
  linarith

theorem acceptAllL_κ (x : Msg) : acceptAllL.κ 0 x < 1 / 3 := by
  have hs := slack_le_two
  have ho := acceptAllL_outer x
  have ho0 : 0 ≤ 2 * (acceptAllL.k x : ℝ) / (acceptAllL.cap x : ℝ) + Real.exp (-(acceptAllL.k x : ℝ) / 8) := by
    positivity
  unfold Compiled.κ
  rw [acceptAllL_κs, add_zero, mul_zero, zero_add]
  nlinarith [mul_le_mul_of_nonneg_right (show 1 + slack ≤ 3 by linarith) ho0]

theorem acceptAllL_work (x : Msg) : acceptAllL.work (fun _ => 0) (fun _ => 0) x ≤ 2 ^ 100 := by
  have he : acceptAllL.es (fun _ => 0) x = 0 := maxBelow_zero _
  have hh : acceptAllL.hs (fun _ => 0) x = 0 := maxBelowN_zero _
  unfold Compiled.work
  rw [he, hh, acceptAllL_k, acceptAllL_nPU, show acceptAllL.openCalls x = 0 from rfl]
  norm_num

theorem acceptAllL_lam_pos (x : Msg) :
    0 < lamLin (acceptAllL.κ 0 x) (acceptAllL.γ x) 1 (acceptAllL.work (fun _ => 0) (fun _ => 0) x) := by
  have hκ := acceptAllL_κ x
  have hκ0 := acceptAllL.κ_pos le_rfl x
  have hw := acceptAllL_work x
  have hw0 := acceptAllL.work_pos (fun _ => 0) (fun _ => 0) x
  have h3 : Real.logb 2 3 < 2 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by norm_num)]; norm_num
  have h1 : Real.logb 2 3 < Real.logb 2 (1 / acceptAllL.κ 0 x) := by
    refine Real.logb_lt_logb (by norm_num) (by norm_num) ?_
    rw [lt_div_iff₀ hκ0]; linarith
  have hγ : Real.logb 2 (acceptAllL.γ x) = 0 := by rw [acceptAllL_γ, Real.logb_zero]
  have hW : Real.logb 2 (2 * 1 * acceptAllL.work (fun _ => 0) (fun _ => 0) x) ≤ 101 := by
    rw [Real.logb_le_iff_le_rpow (by norm_num) (by positivity)]
    norm_num; linarith
  unfold lamLin
  have := lt_min (lt_min h1 (show Real.logb 2 3 < 255.5 - Real.logb 2 (acceptAllL.γ x) / 2 by rw [hγ]; linarith))
    (show Real.logb 2 3 < 256 - Real.logb 2 (2 * 1 * acceptAllL.work (fun _ => 0) (fun _ => 0) x) by linarith)
  linarith

theorem acceptAllL_missAt (x : Msg) : acceptAllL.missAt 1 x = 0 := by
  have hle : toyStmt.design.miss toyPart 1 ≤ 0 := by
    refine Real.iSup_le (fun B => ?_) le_rfl
    obtain ⟨hB, hc⟩ := B.2
    have h0 : 0 ∈ B.1 := by
      obtain ⟨v, hv⟩ := Finset.card_pos.1 hc
      have := Finset.mem_range.1 (hB hv)
      rw [toyPart_blocks, List.length_singleton] at this
      obtain rfl : v = 0 := by omega
      exact hv
    rw [toy_Pa, ite_eq_right]
    intro hw
    have : 0 ∈ window 0 1 B.1 := by simp [window, h0]
    rw [hw] at this
    simp at this
  show max 0 (toyStmt.design.miss (acceptAllL.part toyStmt) 1) = 0
  exact max_eq_left hle

end Numbers

/-! ## The accept-all instance's extractor makes no call of its own -/

section Extract

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] in
/-- A run's extractions make no SHA-512 call of their own where its proof blocks' make none. -/
theorem noHash_runExt (L : Compiled Coin A) (hu : ∀ x s u r v c, NoHash (L.unitExt x s u r v c)) (x : Msg)
    (s : AuditStatement) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) : NoHash (L.runExt x s r v c) := by
  unfold Compiled.runExt
  refine noHash_flatMapRP _ (fun _ => ?_) _
  split
  · exact hu _ _ _ _ _ _
  · trivial

/-- At a compiled verifier whose proof blocks' extractions make no SHA-512 call of their own, the compiled extractor
makes none: its waits rerun. -/
theorem noHash_compiledE (L : Compiled Coin A) (hu : ∀ x s u r v c, NoHash (L.unitExt x s u r v c)) (x : Msg)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) : NoHash (L.E.ext x r₀ c) := by
  intro v₀
  dsimp only [Compiled.E]
  split
  · trivial
  · split_ifs
    -- The waits' coin lists are generalized first: unifying against them unfolds `List.finRange` at the cap.
    · generalize List.map c.1 (List.finRange (L.cap x)) = rs
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun outer => ?_
      generalize (List.finRange (L.k x)).zip outer = jos
      refine noHash_bind _ _ (noHash_flatMapRP _ (fun jo => ?_) _) fun _ => trivial
      generalize List.map (fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l)) (List.finRange (L.cap x)) = rs'
      refine noHash_bind _ _ (noHash_waitAccC _ _ _) fun inner => ?_
      generalize (List.finRange (L.k x)).zip inner = lis
      exact noHash_flatMapRP _ (fun _ => noHash_runExt L hu _ _ _ _ _) _
    · trivial

omit [Nonempty Coin] in
/-- `B_conf` finds nothing where a run's extractions make no SHA-512 call of their own. -/
theorem bconf_pr_zero (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (hn : ∀ s r v c, NoHash (L.runExt x s r v c)) : (L.Bconf H Ω P x).Pr H = 0 := by
  unfold Compiled.Bconf
  split
  · unfold Finder.Pr
    rw [Pr_congr fun q => iff_false_intro (by simp)]
    exact Pr_false
  · unfold Finder.Pr
    rw [Pr_congr fun q => iff_false_intro (by simp [ownLog_noHash _ _ _ (hn _ _ _ _), findColl])]
    exact Pr_false

end Extract

theorem noHash_acceptAllL_unitExt (x : Msg) (s : AuditStatement) (u : ℕ) (r : Fin (acceptAllL.V.k x) → Unit)
    (v : Seen) (c : acceptAllArg.EC (acceptAllL.sh x u)) : NoHash (acceptAllL.unitExt x s u r v c) := by
  unfold Compiled.unitExt
  split
  · trivial
  · split
    · exact trivial
    · trivial

theorem acceptAllL_bext (H : Msg → Msg) (x : Msg) :
    (Bext H acceptAllL.V acceptAllL.E Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < acceptAllL.V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_compiledE acceptAllL noHash_acceptAllL_unitExt _ _ _), findColl])]
  exact Pr_false

theorem acceptAllL_expectedCR (H : Msg → Msg) (x : Msg) :
    ExpectedCR H (Bext H acceptAllL.V acceptAllL.E Unit silentP x) := by
  unfold ExpectedCR
  rw [acceptAllL_bext]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- `KSLin`'s bound fails at `acceptAllL`'s silent prover, at a relation with no witness at `x` and a bound `κ' < 1`. -/
theorem acceptAllL_bound_fails (H : Msg → Msg) (Rx : Rel Claims.Sampling.Wit) (κ' : ℝ) (x : Msg)
    (hR : ∀ w, ¬ Rx x w) (hκ : κ' < 1) :
    ¬ PrBad H acceptAllL.V Rx acceptAllL.E x silentP ≤
      κ' + Real.sqrt (acceptAllL.γ x * PrB H acceptAllL.V silentP x) +
        1 * (Bext H acceptAllL.V acceptAllL.E Unit silentP x).Pr H := by
  intro h
  have hp : PrBad H acceptAllL.V Rx acceptAllL.E x silentP = 1 := prBad_accV H Rx _ x silentP hR
  have hb : PrB H acceptAllL.V silentP x = 0 := prB_accV H silentP x
  rw [hp, hb, mul_zero, Real.sqrt_zero, add_zero, acceptAllL_bext, mul_zero, add_zero] at h
  linarith

/-! ## The committed trace at the accept-all instance opens nothing -/

theorem acceptAllL_opensAt (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (acceptAllL.V.k x) → Unit) (c : acceptAllL.SlotEC x) : acceptAllL.opensAt H P x ω r c 0 = none :=
  rejCm_val H _

theorem acceptAllL_regVal (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    acceptAllL.regVal H P x ω 0 = none := by
  have h : ∀ b, Pr (fun rc : (Fin (acceptAllL.V.k x) → Unit) × acceptAllL.SlotEC x =>
      acceptAllL.opensAt H P x ω rc.1 rc.2 0 = some b) = 0 := fun b => by
    rw [Pr_congr fun rc => iff_false_intro (by rw [acceptAllL_opensAt]; simp)]
    exact Pr_false
  unfold Compiled.regVal
  simp [h]

/-- The one proof block is in the committed trace's incorrect set at every tape: nothing opens its input. -/
theorem acceptAllL_freg (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    0 ∈ acceptAllL.Freg H P x ω := by
  unfold Compiled.Freg
  rw [show acceptAllL.parse x = some toyStmt from rfl]
  simp only [Finset.mem_filter]
  refine ⟨Finset.mem_range.2 Nat.one_pos, ?_⟩
  rintro ⟨b, hb, val, ι, hv, hc⟩
  obtain rfl : b = [0] := (Option.some.inj hb).symm
  obtain ⟨b', hb'⟩ := toy_val_of_correct hc
  rw [hv 0 Nat.one_pos, acceptAllL_regVal] at hb'
  cases hb'

theorem acceptAllL_εreg (H : Msg → Msg) (x : Msg) : acceptAllL.εreg H x silentP = 0 := by
  unfold Compiled.εreg
  rw [acceptAllL_κs, acceptAllL_γs,
    bconf_pr_zero acceptAllL H silentP x fun _ _ _ _ => noHash_runExt acceptAllL noHash_acceptAllL_unitExt _ _ _ _ _]
  simp

/-! ## The nontrivial witnesses -/

/-- **`CompilerKS`'s conclusion fails at the accept-all instance**, at `δ = 0`: the silent prover is accepted, no
witness is in the relation, and the bound is below `1`. -/
theorem compilerKS_nontrivial : ¬ acceptAllL.KS FlockDeploy.sha512 := fun h =>
  acceptAllL_bound_fails _ (Claims.Sampling.R _ acceptAllL.commit acceptAllL.parse acceptAllL.partition 0)
    (acceptAllL.κ 0 []) [] (sksD_noR (Coin := Unit) _ 0 le_rfl []) (by linarith [acceptAllL_κ []])
    (h 0 le_rfl [] (acceptAllL_scope []) Unit silentP)

/-- **`CompilerKSCount`'s conclusion fails at the accept-all instance**, at `t = 1`, whose miss is `0`. -/
theorem compilerKSCount_nontrivial : ¬ acceptAllL.KSCount FlockDeploy.sha512 := fun h =>
  acceptAllL_bound_fails _ (Claims.Sampling.RCount _ acceptAllL.commit acceptAllL.parse acceptAllL.partition 1)
    (acceptAllL.κ (acceptAllL.missAt 1 []) []) [] (sksD_noRCount (Coin := Unit) _ [])
    (by rw [acceptAllL_missAt]; linarith [acceptAllL_κ []]) (h 1 [] (acceptAllL_scope []) Unit silentP)

/-- **`CompilerBits`' conclusion fails at the accept-all instance**, at `δ = 0`, where `λ > 0`. -/
theorem compilerBits_nontrivial : ¬ acceptAllL.Bits FlockDeploy.sha512 (fun _ => 0) (fun _ => 0) := by
  intro h
  have hp : PrBad FlockDeploy.sha512 acceptAllL.V
      (Claims.Sampling.R FlockDeploy.sha512 acceptAllL.commit acceptAllL.parse acceptAllL.partition 0) acceptAllL.E []
      silentP = 1 :=
    prBad_accV _ _ _ [] silentP (sksD_noR (Coin := Unit) _ 0 le_rfl [])
  have h2 : PrBad FlockDeploy.sha512 acceptAllL.V
      (Claims.Sampling.R FlockDeploy.sha512 acceptAllL.commit acceptAllL.parse acceptAllL.partition 0) acceptAllL.E []
      silentP ≤ (1 : ℕ) * (2 : ℝ) ^ (-lamLin (acceptAllL.κ 0 []) (acceptAllL.γ []) 1
        (acceptAllL.work (fun _ => 0) (fun _ => 0) [])) :=
    h 0 le_rfl [] (acceptAllL_scope []) Unit silentP 1 le_rfl (calls_silentP_accV (Coin := Unit) _ [])
      (strictBirthday_B_accV (Coin := Unit) _ silentP []) (acceptAllL_expectedCR _ [])
  rw [hp, Nat.cast_one, one_mul] at h2
  have : (2 : ℝ) ^ (-lamLin (acceptAllL.κ 0 []) (acceptAllL.γ []) 1
      (acceptAllL.work (fun _ => 0) (fun _ => 0) [])) < 1 :=
    Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith [acceptAllL_lam_pos []])
  linarith

/-- **`CompilerKSReg`'s conclusion fails at the accept-all instance**: every run is accepted with its one proof block,
which is in the committed trace's incorrect set, while `ε_reg = 0`. -/
theorem compilerKSReg_nontrivial : ¬ acceptAllL.KSReg FlockDeploy.sha512 := by
  intro h
  have h1 := (h [] (acceptAllL_scope []) Unit silentP).1
  rw [Pr_congr fun p => iff_true_intro (by
      exact ⟨run_accV_out _ _ _ _ _, 0, List.mem_singleton_self 0, acceptAllL_freg _ _ _ _⟩),
    Claims.ZKMal.Pr_true, acceptAllL_εreg] at h1
  norm_num at h1

/-- **What the accept-all instance drops**: its rounds are its shape's, its commitment binds and it meets the contract,
and its proof system is not knowledge sound. -/
theorem acceptAll_drops : acceptAllArg.WF ∧ acceptAllL.commit.Binds FlockDeploy.sha512 ∧
    acceptAllL.Sessions FlockDeploy.sha512 ∧ ¬ acceptAllArg.KS FlockDeploy.sha512 := by
  refine ⟨fun _ => rfl, fun _ _ _ _ _ h _ => by simp [CommitmentScheme.Opens, acceptAllL, oneL, rejCm, OC.eval] at h,
    oneL_sessions acceptAllArg rejCm [] 0 _ fun _ _ h => h.elim, fun h => ?_⟩
  have h1 := h [] Unit silentP
  have hp : PrBad FlockDeploy.sha512 acceptAllArg.V acceptAllArg.R acceptAllArg.E [] silentP = 1 :=
    prBad_accV _ _ _ [] silentP fun _ h => h
  have hb : PrB FlockDeploy.sha512 acceptAllArg.V silentP [] = 0 := prB_accV _ silentP []
  rw [hp, hb, mul_zero, Real.sqrt_zero, add_zero] at h1
  exact absurd h1 (show ¬ (1 : ℝ) ≤ 0 by norm_num)

/-! ## `CompilerEfficient`: an opening check that calls more than it counts -/

/-- **A proof system whose extractor finds a witness at once**: `acceptAllArg`, its extractor returning `()` with no
rerun and no call. -/
def findArg : Arg Unit := { acceptAllArg with ext := fun _ _ _ => .ret (some ()) }

/-- A commitment whose every opening passes after one SHA-512 call. -/
def callCm : CommitmentScheme Bool := ⟨fun _ _ => .query [] fun _ => .ret true, fun _ _ _ => []⟩

/-- **The one-block compiled verifier over it**, its session's witness revealing `opening₀` under `callCm`, whose
check it counts as no call. -/
def effL : Compiled Unit findArg := oneL findArg callCm [([], opening₀)] 0

section OwnCalls

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem exists_cons_of_length_pos {α : Type} {l : List α} (h : 0 < l.length) : ∃ a l', l = a :: l' :=
  List.exists_cons_of_ne_nil (List.ne_nil_of_length_pos h)

theorem ownLog_flatMapRP_head {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) (b : β)
    (bs : List β) : (ownLog H run (f b)).length ≤ (ownLog H run (flatMapRP f (b :: bs))).length := by
  rw [Claims.Sampling.ownLog_flatMapRP, List.map_cons, List.sum_cons]
  exact Nat.le_add_right _ _

/-- A wait whose first draw is accepted keeps it first. -/
theorem waitAccC_head {R : Type} (H : Msg → Msg) (run : R → Trace) {k : ℕ} (hk : 0 < k) (r : R) (rs : List R)
    (hr : (run r).out = .acc) : ∃ l, ((waitAccC k (r :: rs) []).eval H run).1 = (r, (run r).seen) :: l := by
  rw [Claims.Sampling.waitAccC_eval_nil, Claims.Sampling.accList,
    List.filter_cons_of_pos (p := fun r => decide ((run r).out = .acc)) (decide_eq_true hr), List.take_cons hk,
    List.map_cons]
  exact ⟨_, rfl⟩

/-- **The compiled extractor makes a SHA-512 call of its own** wherever every run is accepted and each accepted run's
extractions make one: its waits keep their first draws, so the first inner run's extractions are made. -/
theorem ownLog_E_pos (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : ∀ r, (run H P L.V x ω r).out = .acc) (hk : 0 < L.k x) (hcap : 0 < L.cap x)
    (hrun : ∀ r v c', 0 < (ownLog H (run H P L.V x ω) (L.runExt x s r v c')).length) :
    0 < (ownLog H (run H P L.V x ω) (L.E.ext x r₀ c)).length := by
  have h₀ : (run H P L.V x ω r₀).seen.out = .acc := hacc r₀
  obtain ⟨a, as, ha⟩ := exists_cons_of_length_pos (l := List.finRange (L.k x)) (by rw [List.length_finRange]; exact hk)
  obtain ⟨l₀, ls, hl⟩ :=
    exists_cons_of_length_pos (l := List.finRange (L.cap x)) (by rw [List.length_finRange]; exact hcap)
  simp only [Compiled.E, ownLog, hs, ite_eq_left h₀]
  rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_of_noHash _ _ (Claims.Sampling.NoHash.waitAccC _ _ _),
    List.nil_append, Claims.Sampling.ownLog_bind, List.length_append]
  refine Nat.lt_of_lt_of_le ?_ (Nat.le_add_right _ _)
  rw [hl, List.map_cons]
  obtain ⟨o, ho⟩ := waitAccC_head H (run H P L.V x ω) hk (c.1 l₀) (ls.map c.1) (hacc _)
  rw [ho, ha, List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  dsimp only
  rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_of_noHash _ _ (Claims.Sampling.NoHash.waitAccC _ _ _),
    List.nil_append, List.map_cons]
  obtain ⟨o', ho'⟩ := waitAccC_head H (run H P L.V x ω) hk (forkOn (L.stage2 x) (c.1 l₀) (c.2.1 a l₀))
    (ls.map fun l => forkOn (L.stage2 x) (c.1 l₀) (c.2.1 a l)) (hacc _)
  rw [ho', List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  exact hrun _ _ _

end OwnCalls

/-- Each of `effL`'s runs' extractions makes a SHA-512 call of its own: its one slot's session finds the witness at
once, and checking the opening it reveals calls `callCm`'s one. -/
theorem effL_runExt (H : Msg → Msg) (x : Msg) (run : (Fin (effL.V.k x) → Unit) → Trace)
    (r : Fin (effL.V.k x) → Unit) (v : Seen) (c : effL.SlotEC x) :
    0 < (ownLog H run (effL.runExt x toyStmt r v c)).length := by
  have h1 : 0 < effL.nPU x := Nat.one_pos
  obtain ⟨d, ds, hd⟩ :=
    exists_cons_of_length_pos (l := List.finRange (effL.nPU x)) (by rw [List.length_finRange]; exact h1)
  unfold Compiled.runExt
  rw [hd, show effL.checked x r = [0] from rfl, List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  dsimp only
  rw [dite_eq_left h1, Compiled.unitExt, show effL.session x 0 v = some (x, v) from rfl]
  dsimp only
  rw [dite_eq_left (show findArg.shape x = effL.sh x 0 from rfl)]
  exact Nat.succ_pos _

/-- **`CompilerEfficient`'s conclusion fails at `effL`**, at no reruns and no calls of the proof system's: every run is
accepted, so the compiled extractor reaches its first inner run's extraction, whose opening check makes a SHA-512 call
of its own, while the own-call bound is `0`. -/
theorem compilerEfficient_nontrivial :
    ¬ effL.Efficient FlockDeploy.sha512 (fun _ => 0) (fun _ => 0) (fun _ => 0) := by
  rintro ⟨-, -, h⟩
  have c : effL.E.C [] := Classical.arbitrary _
  have h1 := h [] Unit silentP () (fun _ => ()) c
  have h0 : effL.k [] ^ 2 * effL.nPU [] * (effL.openCalls [] + effL.hs (fun _ => 0) []) = 0 := by
    rw [show effL.hs (fun _ => 0) [] = 0 from maxBelowN_zero _, show effL.openCalls [] = 0 from rfl, add_zero,
      mul_zero]
  have hpos := ownLog_E_pos effL FlockDeploy.sha512 silentP [] () (fun _ => ()) c (s := toyStmt) rfl
    (fun r => run_accV_out _ _ _ _ r) (by unfold Compiled.k; positivity)
    (by unfold Compiled.cap Compiled.k; positivity) fun r v c' => effL_runExt _ [] _ r v c'
  exact Nat.lt_irrefl 0 (Nat.lt_of_lt_of_le hpos (h1.trans_eq h0))

/-- **What `effL` drops**: it meets the contract, and its proof system reruns nothing and makes no call of its own, but
its opening checks make more calls than it counts. -/
theorem compilerEfficient_drops : effL.Sessions FlockDeploy.sha512 ∧
    findArg.RerunsLe FlockDeploy.sha512 (fun _ => 0) ∧ findArg.RerunsExpLe FlockDeploy.sha512 (fun _ => 0) ∧
    findArg.OwnLe FlockDeploy.sha512 (fun _ => 0) ∧ ¬ effL.OpenCallsLe FlockDeploy.sha512 := by
  refine ⟨oneL_sessions findArg callCm [([], opening₀)] 0 _ fun _ _ h => False.elim h, fun _ _ _ _ _ _ => le_rfl,
    fun _ _ _ _ => ?_, fun _ _ _ _ _ _ => le_rfl, fun h => ?_⟩
  · refine (Ex_mono fun _ => ?_).trans (le_of_eq (Ex_const 0))
    show ((0 : ℕ) : ℝ) ≤ 0
    exact le_of_eq Nat.cast_zero
  · have h1 := h [] toyStmt rfl 0 ⟨[], .rej⟩ ()
    exact absurd h1 (by simp [effL, oneL, callCm, OC.eval])

/-! ## `ZKOut`'s leak form -/

/-- A simulator, as one given a trivial outcome. -/
def oneOut (S : Simulator) : SimOut Unit where
  C := S.C
  prog x _ c := S.prog x c

/-- **`ZKmalR` gives `ZKOut`** with a trivial outcome: the honest view's probability of an event is at most the
simulator's plus the error, the two-sided bound's one side. -/
theorem zkOut_of_zkmalR {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (h : ZKmalR H k P R S εzk γ) :
    ZKOut (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R (fun _ _ => ()) (oneOut S)
      (fun x w _ Vs θ _ => εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) := by
  intro x w hR _ Vs θ hk o Ev
  have h1 := sub_le_iff_le_add'.1 ((le_abs_self _).trans (h x w hR _ Vs θ hk Ev))
  refine (le_of_eq (Pr_congr fun ω => ?_)).trans h1
  exact and_iff_right rfl

open Claims.ZKMal.Masked FlockSoundness.Discharge in
/-- **C-Flock's deployed masked session** (`Flock.Guarantees.MaskedZK`'s record), at one coin per rep round and
`t = 1`. -/
noncomputable def zkFlock :=
  Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation (fun _ _ => 1)
    (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) 1

open Claims.ZKMal.Masked FlockSoundness.Discharge in
open Classical in
/-- `MaskedZK`'s zero-knowledge error at `t`. -/
noncomputable def zkEps (t : ℕ) (x : Msg) : ℝ :=
  if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
    8 * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
      8 * t * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1

/-- `MaskedZK`'s fork factor at `t`, `4·t·3^t`. -/
noncomputable def zkγ (t : ℕ) (_ : Msg) : ℝ := 4 * t * 3 ^ t

theorem zkFlock_zk : ZKmalR FlockDeploy.sha512 (fun x => zkFlock.V.k x + 1) zkFlock.prover
    (Claims.Flock.R zkFlock.parse) zkFlock.S (zkEps 1) (zkγ 1) :=
  (_root_.Flock.SecurityProofs.MaskedZK (fun _ _ => 1) 1 Nat.one_pos).2

/-- **`ZKOutLeak`'s premise at the honest instance**: `ZKOut` at C-Flock's deployed masked session, its outcome trivial,
at `MaskedZK`'s error and fork term. -/
theorem zkOutLeak_honest :
    ZKOut (fun _ => zkFlock.Nonce) (fun _ => zkFlock.Rest) FlockDeploy.sha512 (fun x => zkFlock.V.k x + 1)
      (fun w _ => zkFlock.prover w) (Claims.Flock.R zkFlock.parse) (fun _ _ => ()) (oneOut zkFlock.S)
      (fun x w _ Vs θ _ =>
        zkEps 1 x + zkγ 1 x * (Fork FlockDeploy.sha512 (zkFlock.prover w) Vs θ x).Pr FlockDeploy.sha512) :=
  zkOut_of_zkmalR zkFlock_zk

/-- A prover that says `[0]`. -/
def talkP : Prover (Unit × Unit) := ⟨fun _ _ _ => .ret [0]⟩

/-- **A verifier that accepts everything**, after one round. -/
def acceptAllVs : VStar Unit := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩

/-- A simulator that outputs the empty transcript. -/
def silentSimOut : SimOut Unit where
  C _ := Unit
  prog _ _ _ := .ret []

/-- **`ZKOutLeak`'s conclusion fails at the verifier that accepts everything**: the talking prover's transcript is never
empty, the silent simulator's always is, and the error is `0`. -/
theorem zkOutLeak_nontrivial :
    ¬ Pr (fun ω : Unit × Unit => (runStar FlockDeploy.sha512 talkP acceptAllVs [] ω ()).msgs ≠ []) ≤
        (Fintype.card Unit : ℝ) * Pr (fun p : Unit × silentSimOut.C [] =>
          (silentSimOut.given p.1).out FlockDeploy.sha512 acceptAllVs () [] p.2 ≠ []) + ∑ _o : Unit, (0 : ℝ) := by
  intro h
  have h1 : Pr (fun ω : Unit × Unit => (runStar FlockDeploy.sha512 talkP acceptAllVs [] ω ()).msgs ≠ []) = 1 := by
    rw [Pr_congr fun ω => iff_true_intro (by simp [runStar, stepWith, acceptAllVs, talkP, OC.eval])]
    exact Claims.ZKMal.Pr_true
  have h2 : Pr (fun p : Unit × silentSimOut.C [] =>
      (silentSimOut.given p.1).out FlockDeploy.sha512 acceptAllVs () [] p.2 ≠ []) = 0 := by
    rw [Pr_congr fun p => iff_false_intro (by simp [Simulator.out, SimOut.given, silentSimOut, SP.eval])]
    exact Pr_false
  rw [h1, h2] at h
  norm_num at h

/-- **What the nontrivial instance drops**: `ZKOut` fails there, so `zkOutLeak`'s bound rests on it. -/
theorem zkOut_dropped :
    ¬ ZKOut (fun _ => Unit) (fun _ => Unit) FlockDeploy.sha512 (fun _ => 1) (fun _ _ => talkP)
      (fun _ (_ : Unit) => True) (fun _ _ => ()) silentSimOut (fun _ _ _ _ _ _ => 0) := fun hz =>
  zkOutLeak_nontrivial (Claims.zkOutLeak (fun _ => Unit) (fun _ => Unit) FlockDeploy.sha512 (fun _ => 1)
    (fun _ _ => talkP) (fun _ (_ : Unit) => True) (fun _ _ => ()) silentSimOut (fun _ _ _ _ _ _ => 0) hz [] () trivial
    Unit acceptAllVs () rfl (· ≠ []))

/-! ## `CompilerZK` -/

/-- **C-Flock's deployed masked session as a proof system with a public outcome**: `Arg.ofFlock` at its record and
decoder, its sessions' rounds `V.k + 1` (`Hello` first), its outcome trivial, and its simulator and error
`MaskedZK`'s. -/
noncomputable def zkA : ZkArg FlockDeploy.Coin where
  toArg := Arg.ofFlock zkFlock.toDeploy zkFlock.decode
  kz x := zkFlock.V.k x + 1
  Nonce := zkFlock.Nonce
  Rest := zkFlock.Rest
  P := zkFlock.prover
  St := Unit
  out _ _ := ()
  S := oneOut zkFlock.S
  εzk x w _ Vs θ _ := zkEps 1 x + zkγ 1 x * (Fork FlockDeploy.sha512 (zkFlock.prover w) Vs θ x).Pr FlockDeploy.sha512

/-- The router of one session from the first round, at the run's statement. -/
def oneStep (x : Msg) (hist : List (Msg × Msg)) : Step := .session 0 x hist

/-- **A masked compiled verifier of one session over C-Flock's**: the toy statement, one session of C-Flock's masked
record at the run's statement from the first round, its rounds the session's, and the reference prover the composed
one. Its commitment opens nothing (`rejCm`), so no witness is honest there: the tree states no member of C-Flock's
relation, so the session's witness has none to be. -/
noncomputable def zkL : ZkCompiled FlockDeploy.Coin zkA where
  V := ⟨fun x => zkFlock.V.k x + 1, fun _ _ _ => .ret .rej⟩
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := rejCm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh x _ := zkA.shape x
  κb x _ := zkA.κ x
  γb x _ := zkA.γ x
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  Tape := (Fin 1 → zkFlock.Nonce) × (Fin 1 → zkFlock.Rest)
  P _ := ⟨fun x hist t => (zkFlock.prover ⟨[]⟩).next x hist (t.1 ⟨0, Nat.one_pos⟩, t.2 ⟨0, Nat.one_pos⟩)⟩
  δc _ _ := 1
  step := oneStep
  k₀ _ := 0
  M _ := 1
  n x _ := zkFlock.V.k x + 1
  wit _ _ _ := .ret ⟨[]⟩
  CB x _ := zkFlock.S.C x
  tapeOf _ t := t

theorem rhon_rejCm (H : Msg → Msg) (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ Claims.Sampling.Rhon H rejCm (fun _ => some toyStmt) (fun _ => some toyPart) x w := by
  rintro ⟨s, P, hs, hP, hf⟩
  obtain rfl := Option.some.inj hs
  obtain rfl := Option.some.inj hP
  rw [rejCm_faulty] at hf
  exact Finset.singleton_ne_empty 0 hf

theorem zkL_sessions (H : Msg → Msg) : zkL.ZkSessions H where
  rounds _ _ _ := by simp [ZkCompiled.start, zkL]
  routed _ _ _ := ⟨fun _ h => absurd h (Nat.not_lt_zero _), fun j hj hist _ _ => by
    obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
    simp [ZkCompiled.start, zkL, oneStep, Step.stmt]⟩
  shaped _ _ _ _ _ _ _ := ⟨rfl, ⟨Equiv.refl _⟩⟩
  witness x w h _ _ _ _ := absurd h (rhon_rejCm H x w)
  honest _ _ _ _ _ _ _ := rfl

/-- **`CompilerZK`'s premises at the honest instance**: C-Flock's masked session is zero knowledge with a public outcome
(`MaskedZK`), and the one-session masked compiled verifier meets the contract. -/
theorem compilerZK_honest : zkA.ZK FlockDeploy.sha512 ∧ zkL.ZkSessions FlockDeploy.sha512 :=
  ⟨zkOutLeak_honest, zkL_sessions _⟩

/-- **A proof system whose prover talks and whose simulator is silent**: `acceptAllArg` with every witness in its
relation, one round per session, the talking prover, the trivial outcome, the silent simulator and error `0`. -/
def talkArg : ZkArg Unit where
  toArg := { acceptAllArg with R := fun _ _ => True }
  kz _ := 1
  Nonce := Unit
  Rest := Unit
  P _ := talkP
  St := Unit
  out _ _ := ()
  S := silentSimOut
  εzk _ _ _ _ _ _ := 0

/-- The router of one session from the first round, at the empty statement. -/
def talkStep (_ : Msg) (hist : List (Msg × Msg)) : Step := .session 0 [] hist

/-- **A masked compiled verifier of one session over `talkArg`**: the toy statement, every opening passing (`accCm`),
the verifier that accepts after its one round, the session's, and the reference prover the composed one. -/
def talkL : ZkCompiled Unit talkArg where
  V := accV Unit
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some toyPart
  commit := accCm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ _ := Finset.univ
  session x _ v := some (x, v)
  iroots _ _ _ := []
  reveal _ _ _ _ := []
  openCalls _ := 0
  Tape := (Fin 1 → Unit) × (Fin 1 → Unit)
  P _ := ⟨fun _ _ _ => .ret [0]⟩
  δc _ _ := 1
  step := talkStep
  k₀ _ := 0
  M _ := 1
  n _ _ := 1
  wit _ _ _ := .ret ()
  CB _ _ := Unit
  tapeOf _ t := t

theorem talkL_sessions (H : Msg → Msg) : talkL.ZkSessions H where
  rounds _ _ _ := by simp [ZkCompiled.start, talkL, accV]
  routed _ _ _ := ⟨fun _ h => absurd h (Nat.not_lt_zero _), fun j hj hist _ _ => by
    obtain rfl : j = 0 := Nat.lt_one_iff.1 hj
    simp [ZkCompiled.start, talkL, talkStep, Step.stmt]⟩
  shaped _ _ _ _ _ _ _ := ⟨rfl, ⟨Equiv.refl _⟩⟩
  witness _ _ _ _ _ _ _ := trivial
  honest _ _ _ _ _ _ _ := rfl

/-- The toy witness opens the one input under `accCm`, so no proof block is incorrect. -/
theorem talkL_rhon (H : Msg → Msg) :
    Claims.Sampling.Rhon H talkL.commit talkL.parse talkL.partition [] toyWit :=
  ⟨toyStmt, toyPart, rfl, rfl, by
    show Claims.Sampling.faulty H accCm toyStmt toyPart toyWit = ∅
    simp [toy_faulty, toyWit_val]⟩

theorem talkL_msgs (H : Msg → Msg) (w : Claims.Sampling.Wit)
    (t : (Fin (talkL.M []) → talkArg.Nonce) × (Fin (talkL.M []) → talkArg.Rest)) :
    (runStar H (talkL.prover w []) acceptAllVs [] t ()).msgs ≠ [] := by
  simp [runStar, stepWith, acceptAllVs, ZkCompiled.prover, talkL, Prover.comap, OC.eval]

theorem simAt_silent (C : Type) (xS : Msg) (o : Unit) (c : C) : simAt silentSimOut C xS o c = .ret [] := by
  unfold simAt
  split <;> rfl

/-- The composed simulator at `talkL` says nothing. -/
theorem talkL_simOut (H : Msg → Msg) (o : List Msg × List Unit) (c : talkL.sim.C []) :
    (talkL.sim.given o).out H acceptAllVs () [] c = [] := by
  obtain ⟨l, _ | ⟨u, os⟩⟩ := o
  · rfl
  · have h : 0 < talkL.M [] := Nat.one_pos
    show ((spBind (spUnder [] (simAt silentSimOut (talkL.CB [] (⟨0, h⟩ : Fin (talkL.M [])).val) [] u
      (c ⟨0, h⟩))) fun τ =>
      talkL.simSessions [] (u :: os) c [] ([] ++ τ)).eval H _).1 = []
    rw [simAt_silent]
    rfl

theorem talkL_err (H : Msg → Msg) (w : Claims.Sampling.Wit) (o : List Msg × List talkArg.St) :
    talkL.err H [] w Unit acceptAllVs () o = 0 := by
  unfold ZkCompiled.err
  refine Finset.sum_eq_zero fun j _ => (congrArg _ (funext fun t => ?_)).trans (Ex_const (0 : ℝ))
  dsimp only
  split <;> rfl

/-- **`CompilerZK`'s conclusion fails at `talkL`**: at the toy witness, which no proof block faults, the talking
prover's transcript is never empty, the composed simulator's always is, and the error is `0`. -/
theorem compilerZK_nontrivial : ¬ talkL.ZK FlockDeploy.sha512 := by
  intro h
  have h1 := h (fun x => Claims.Sampling.leafLayer FlockDeploy.sha512 talkL.commit talkL.parse x toyWit) [] toyWit
    ⟨talkL_rhon _, rfl⟩ Unit acceptAllVs () rfl
    (Claims.Sampling.leafLayer FlockDeploy.sha512 talkL.commit talkL.parse [] toyWit, [()]) (· ≠ [])
  rw [Pr_congr fun t => iff_true_intro (by exact ⟨rfl, talkL_msgs _ _ t⟩), Claims.ZKMal.Pr_true,
    Pr_congr fun c => iff_false_intro (by exact not_not_intro (talkL_simOut _ _ c)), Pr_false, talkL_err,
    add_zero] at h1
  norm_num at h1

/-- **What `talkL` drops**: it meets the zero-knowledge contract, and its proof system is not zero knowledge. -/
theorem compilerZK_drops : talkL.ZkSessions FlockDeploy.sha512 ∧ ¬ talkArg.ZK FlockDeploy.sha512 :=
  ⟨talkL_sessions _, zkOut_dropped⟩

end Claims.Witness.SamplingGeneric
