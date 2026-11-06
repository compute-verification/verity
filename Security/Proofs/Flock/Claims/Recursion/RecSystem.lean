module

public import Proofs.Core.Rec
public import Proofs.Flock.Claims.Recursion.SeqDeployStop

/-!
# The deployed recursion as a `Rec.system`, and its release

The deployed recursion at SHA-512 with the inner prover that commits (`zkStop_deployRecMStop_commit_sha512`'s record)
is `Verity.Rec.system 𝔒 𝔊 go` at the continuation test `go = fun _ _ => Claims.Recursion.goesOn`, so its `ZKStop` is
`Verity.SecurityProofs.RecZK`'s conclusion read at the record.

* `recOuter`: the outer system `O`, Flock's `--zk` session at the recursive setup on its own coins (`G.prover`,
  `(G.withSim Mc).simulatorE`), of `VO.k + 1` rounds.
* `recCompiled`: the compiled recursion `G`, the deployed first part (`preP` at `preTape`, `preOf`; the stage message
  `stgOf`; the stop `sR`) and the V\* sessions (`xsOf`, `wit`, `nums`).
* `RecZK`'s four premises at the deployed objects: `O`'s zero knowledge (`recOuter_zk`, Flock's at the session's
  tables), the first part's commit strings (`recCompiled_commits`), each session's public shape (`recCompiled_byShape`,
  from the leaf counts `hzl` and the coin bijections `hCB`), and the live witness (`recCompiled_witnessIn`, `LiveWit`).
  `recSystem_zk` is `RecZK` at them.
* `zkStop_of_recZK`: `ZKStop` at the record, `zkStop_deployRecMStop_commit_sha512`'s statement, from `recSystem_zk`:
  the record's prover is `Rec.prover`'s (`prover_rec_next`), `SimStop` is `Rec.sim`'s on the coins of `simPreCoins`
  (`zkAt_rec_sim`), and the bounds agree.
* `haltP`, `cutsOf_haltP`: the released prover, which sends the empty message from the round its halt test fires,
  whatever the test reads (the witness, the tape, its own coins and the history), cuts the prover's run where the test
  fires (`Claims.ZKMal.interact_censor`), at one of the positions `pos` the test can fire at.
  `recRelease_leak`: at the deployed record, the release leaks at most `log₂ (VM.k x + 1)` bits beyond `ZKStop`'s bound
  (`Verity.zkStop_leak`).
-/

@[expose] public section

namespace Claims.ZKMal

open Claims Claims.ZKServer

theorem zkmalAt_k_eq {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k k' : ℕ}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ : ℝ} (hk : k = k') (h : ZKmalAt H k P R x C prog ε γ) : ZKmalAt H k' P R x C prog ε γ := by
  subst hk
  exact h

theorem zkAt_of_sim_eq {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {ε γ : ℝ} {S S' : Simulator} (hS : S = S')
    (h : ZKmalAt H k P R x (S.C x) (S.prog x) ε γ) : ZKmalAt H k P R x (S'.C x) (S'.prog x) ε γ := by
  subst hS
  exact h

/-- `prefixRun` at zero digests, at a number of inner rounds equal to another. -/
theorem prefixRun_cast {Sa Gp Cm : Type} [AddCommGroup Gp] (Hl : Gp × Cm → Msg) (My : Sa → Gp) (c : Sa → Cm)
    (ask : List Msg → Out) {a a' : ℕ} (h : a = a') (nC : ℕ) (stg : ℕ → List (Msg × Msg) → (Fin nC → Msg) → Msg)
    (yI : Fin a → Sa) (yS : Fin nC → Sa) :
    prefixRun Hl My c ask a nC (fun _ => 0) (fun _ _ => 0) (stg a) yI yS =
      prefixRun Hl My c ask a' nC (fun _ => 0) (fun _ _ => 0) (stg a') (fun i => yI (Fin.cast h.symm i)) yS := by
  subst h
  rfl

/-- **Where a stop fires is a position it can fire at**: within `k` rounds, at a history of a length in `pos`, or at
`k` itself. -/
theorem stopIdx_mem {s : List (Msg × Msg) → Bool} {k : ℕ} (τ : List (Msg × Out)) {pos : Finset ℕ}
    (hpos : ∀ h : List (Msg × Msg), h.length ≤ k → s h = true → h.length ∈ pos) (hk : k ∈ pos) :
    stopIdx s k k τ ∈ pos := by
  have hT : stopIdx s k k τ ≤ k := by
    by_contra h
    have := stopIdx_eq (s := s) (n := k) (τ := τ) (not_le.1 h)
    omega
  rcases Nat.lt_or_eq_of_le hT with hlt | heq
  · have hs := stopIdx_spec hlt
    have hlen : stopIdx s k k τ ≤ τ.length := by
      by_contra h
      have hf := stopIdx_min (not_le.1 h) (by omega)
      rw [List.take_of_length_le (le_of_lt (not_le.1 h))] at hs
      rw [List.take_of_length_le (le_refl _)] at hf
      exact Bool.false_ne_true (hf.symm.trans hs)
    have hl : (histOf (τ.take (stopIdx s k k τ))).length = stopIdx s k k τ := by
      rw [length_histOf_take, Nat.min_eq_left hlen]
    have := hpos _ (by rw [hl]; exact hT) hs
    rwa [hl] at this
  · rw [heq]
    exact hk

end Claims.ZKMal

namespace Verity

open Claims Claims.ZKMal
open scoped NNReal

/-- The relabelling of a session's coins `Rec.slotP` reads, when the slot's coins are `O`'s. -/
noncomputable def relab (N Ω : Type) : (N ≃ N) × (Ω ≃ Ω) :=
  (⟨(Equiv.refl N, Equiv.refl Ω)⟩ : Nonempty ((N ≃ N) × (Ω ≃ Ω))).some

/-- **A prover on the slot's coins**: `P` with `relab`'s relabelling undone, so `Rec.slotP` of it is `P`. -/
noncomputable def relabP {W N Ω : Type} (P : W → Prover (N × Ω)) (w : W) : Prover (N × Ω) :=
  ⟨fun xs h ω => (P w).next xs h ((relab N Ω).1.symm ω.1, (relab N Ω).2.symm ω.2)⟩

/-- **On coins of one type at every statement, a session is `O`'s prover**, where `O`'s prover is `relabP P`. -/
theorem slotP_relab {WS N Ω : Type} (O : ZkSystem WS (fun _ => N) (fun _ => Ω)) (P : WS → Prover (N × Ω))
    (hP : ∀ w xs, O.P w xs = relabP P w) (w : WS) (xs : Msg) (h : List (Msg × Msg)) (ω : N × Ω) :
    (Rec.slotP O N Ω w).next xs h ω = (P w).next xs h ω := by
  have e : Nonempty ((N ≃ N) × (Ω ≃ Ω)) := ⟨(Equiv.refl N, Equiv.refl Ω)⟩
  simp only [Rec.slotP, dif_pos e, hP]
  show (P w).next xs h ((relab N Ω).1.symm ((relab N Ω).1 ω.1), (relab N Ω).2.symm ((relab N Ω).2 ω.2)) = _
  rw [Equiv.symm_apply_apply, Equiv.symm_apply_apply]

/-- **The released prover**: `P`, except that it sends the empty message from the first round its halt test
`halt` fires within `k x` rounds; the test reads the witness, the tape, the prover's own coins `K x` and the history. -/
def haltP {W : Type} {N Ω K : Msg → Type} (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x))
    (halt : W → (x : Msg) → N x × Ω x → K x → List (Msg × Msg) → Bool) :
    W → (x : Msg) → Prover ((N x × Ω x) × K x) :=
  fun w x => ⟨fun x' h ωκ => if fired (halt w x ωκ.1 ωκ.2) (k x) h then .ret [] else (P w x).next x' h ωκ.1⟩

/-- **The released prover cuts `P`'s run where its halt test fires** (`interact_censor`), at one of the positions
`pos x` the test can fire at within `k x` rounds, or at `k x`. -/
theorem cutsOf_haltP {W : Type} {N Ω K : Msg → Type} (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → (x : Msg) → Prover (N x × Ω x))
    (halt : W → (x : Msg) → N x × Ω x → K x → List (Msg × Msg) → Bool) (pos : Msg → Finset ℕ)
    (hpos : ∀ w x ω κ (h : List (Msg × Msg)), h.length ≤ k x → halt w x ω κ h = true → h.length ∈ pos x)
    (hk : ∀ x, k x ∈ pos x) :
    CutsOf H k (haltP k P halt) P [] pos := by
  intro x w Θ Vs θ hV ω κ
  refine ⟨stopIdx (halt w x ω κ) (k x) (k x) (interact (fun as => ((Vs.next x as θ).eval H).1)
    (fun h => (((P w x).next x h ω).eval H).1) (k x) []), stopIdx_mem _ (hpos w x ω κ) (hk x), ?_⟩
  rw [runStar_msgs, runStar_msgs, hV]
  have e : (fun h => (((haltP k P halt w x).next x h (ω, κ)).eval H).1) =
      censor (halt w x ω κ) (k x) fun h => (((P w x).next x h ω).eval H).1 := by
    funext h
    show ((if fired (halt w x ω κ) (k x) h = true then OC.ret [] else (P w x).next x h ω).eval H).1 =
      if fired (halt w x ω κ) (k x) h = true then [] else (((P w x).next x h ω).eval H).1
    by_cases hf : fired (halt w x ω κ) (k x) h = true
    · rw [if_pos hf, if_pos hf] <;> rfl
    · rw [if_neg hf, if_neg hf]
  exact (congrArg (fun pm => interact (fun as => ((Vs.next x as θ).eval H).1) pm (k x) []) e).trans
    (interact_censor _ _ _ _ _)

/-- **The deployed recursion's release leaks at most `log₂ (k + 1)` bits**: from `ZKStop`, the released prover
(`haltP`, at any halt test) is `ZKmalRLeak` at the leaky `SimStop` over the `k x + 1` positions `0, …, k x`. -/
theorem zkStop_leak_haltP {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    {K : Msg → Type} [∀ x, Fintype (K x)] [∀ x, Nonempty (K x)] {H : Msg → Msg}
    {D : Claims.Recursion.StopDeploy Coin} {I : Claims.Recursion.Inner Coin Claims.Flock.Wit}
    {DO : Claims.Flock.Deploy Coin Col T} (hZK : Claims.Recursion.ZKStop H D I DO) {t : ℕ} (ht : 0 < t)
    {δ₁ : ℝ≥0} (hδ : Hm96Hiding D.mask (D.toDeploy.saltHash H) δ₁)
    (halt : Claims.Flock.Wit → (x : Msg) → D.Nonce x × D.Rest x → K x → List (Msg × Msg) → Bool) :
    ZKmalRLeak D.Nonce D.Rest K H D.V.k (haltP D.V.k D.prover halt) D.prover (Claims.Recursion.R D.toDeploy I)
      (leakyAt (Claims.Recursion.SimStop D I (D.SO t) H) D.V.k [] (fun x => Finset.range (D.V.k x + 1))
        fun _ => Finset.nonempty_range_add_one)
      (fun x => ((D.V.k x + 1 : ℕ) : ℝ)) (Claims.Recursion.εzkStop D I t δ₁) (Claims.Recursion.γzk D.toDeploy t) := by
  have h := zkStop_leak (K := K) hZK ht hδ (haltP D.V.k D.prover halt) [] (fun x => Finset.range (D.V.k x + 1))
    (cutsOf_haltP H D.V.k D.prover halt _ (fun _ _ _ _ _ hl _ => Finset.mem_range.2 (Nat.lt_succ_of_le hl))
      fun x => Finset.mem_range.2 (Nat.lt_succ_self _)) fun _ => Finset.nonempty_range_add_one
  simpa only [Finset.card_range] using h

end Verity

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (Layout SeqTape Proto ZKmalAt iterSimG prefixRun XorBytes)
open FlockSoundness.Discharge
open scoped NNReal

/-- `Sim`'s first-part coins, the inner salts at the inner verifier's rounds and one staging round's salts, as
`Rec.Compiled.S₀`'s at as many inner rounds. -/
def simPreCoins {a a' : ℕ} (h : a = a') (n : ℕ) :
    (Fin a → Salt) × (Fin 1 → Fin n → Salt) ≃ (Fin a' → Salt) × (Fin n → Salt) where
  toFun c := (fun i => c.1 (Fin.cast h.symm i), c.2 ⟨0, Nat.zero_lt_one⟩)
  invFun c := (fun i => c.1 (Fin.cast h i), fun _ => c.2)
  left_inv c := by
    subst h
    exact Prod.ext rfl (funext fun j => congrArg c.2 (Subsingleton.elim _ j))
  right_inv c := by
    subst h
    rfl

section Swap

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- `simMStop`'s coins: its first part's and one per session. -/
noncomputable def stopCoins (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (S₀ : Claims.Simulator) (SO : Claims.Msg → ℕ → Claims.Recursion.OSim)
    (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool) (x : Claims.Msg) :
    (simMStop IV b T₀ NB ΩB S₀ SO s).C x ≃ S₀.C x × ((j : Fin (nS x)) → (SO x j).C) :=
  Claims.ZKMal.iterCoins T₀ 0 NB ΩB (Layout.seq T₀ 0 NB ΩB ((parts IV b (T₀.N × T₀.Ω)).stopped s))
    (Claims.ZKMal.censorSim S₀ s (kPre IV)) (fun j x => (SO x j).C)
    (Claims.ZKMal.stopProgB s (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
      fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) x (nS x)

/-- **`simMStop`'s transcript**: its first part's, cut where the stop fires, then one `stepS` per session. -/
theorem simMStop_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (T₀ : SeqTape)
    (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool)
    (x : Claims.Msg) (c : (simMStop IV b T₀ NB ΩB S₀ SO s).C x) :
    (Claims.SP.eval H ask ((simMStop IV b T₀ NB ΩB S₀ SO s).prog x c)).1 =
      (List.finRange (nS x)).foldl (fun τ (j : Fin (nS x)) => stepS H ask SO x
          (Claims.ZKMal.stopProgB s (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
            fun j x u c => (SO x j).prog (xsOf IV b x j u) c) τ j
          ((stopCoins IV b T₀ NB ΩB S₀ SO s x c).2 j))
        (Claims.cutT ask [] (Claims.ZKMal.stopIdx (s x) (kPre IV x) (kPre IV x)
          (Claims.SP.eval H ask (S₀.prog x (stopCoins IV b T₀ NB ΩB S₀ SO s x c).1)).1) (kPre IV x)
          (Claims.SP.eval H ask (S₀.prog x (stopCoins IV b T₀ NB ΩB S₀ SO s x c).1)).1) := by
  rw [← Claims.ZKMal.censorSim_eval H ask S₀ s (kPre IV) x (stopCoins IV b T₀ NB ΩB S₀ SO s x c).1]
  exact iterSimG_stopped_eval IV b H ask T₀ NB ΩB (Claims.ZKMal.censorSim S₀ s (kPre IV)) SO s x
    (Claims.ZKMal.stopProgB s (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
      fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (nS x) c

/-- **`ZKmalAt` at `simMStop` carries over to another first-part simulator** whose transcripts are the first's on
coins in bijection. -/
theorem zkAt_simMStop_swap (H : Claims.Msg → Claims.Msg) (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (S₀ S₀' : Claims.Simulator) (SO : Claims.Msg → ℕ → Claims.Recursion.OSim)
    (s : Claims.Msg → List (Claims.Msg × Claims.Msg) → Bool) (x : Claims.Msg) (e₀ : S₀'.C x ≃ S₀.C x)
    (he₀ : ∀ (ask : List Claims.Msg → Claims.Out) c,
      (Claims.SP.eval H ask (S₀'.prog x c)).1 = (Claims.SP.eval H ask (S₀.prog x (e₀ c))).1)
    {N Ω WP : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {k : ℕ} {P : WP → Claims.Prover (N × Ω)}
    {RA : WP → Prop} {ε γ : ℝ}
    (h : ZKmalAt H k P RA x ((simMStop IV b T₀ NB ΩB S₀ SO s).C x) ((simMStop IV b T₀ NB ΩB S₀ SO s).prog x) ε γ) :
    ZKmalAt H k P RA x ((simMStop IV b T₀ NB ΩB S₀' SO s).C x) ((simMStop IV b T₀ NB ΩB S₀' SO s).prog x) ε γ := by
  refine h.of_sim _ ((stopCoins IV b T₀ NB ΩB S₀' SO s x).trans ((e₀.prodCongr (Equiv.refl _)).trans
    (stopCoins IV b T₀ NB ΩB S₀ SO s x).symm)) fun ask c' => ?_
  rw [simMStop_eval IV b H ask T₀ NB ΩB S₀' SO s x c', simMStop_eval IV b H ask T₀ NB ΩB S₀ SO s x]
  have hq : stopCoins IV b T₀ NB ΩB S₀ SO s x (((stopCoins IV b T₀ NB ΩB S₀' SO s x).trans
      ((e₀.prodCongr (Equiv.refl _)).trans (stopCoins IV b T₀ NB ΩB S₀ SO s x).symm)) c') =
      (e₀ (stopCoins IV b T₀ NB ΩB S₀' SO s x c').1, (stopCoins IV b T₀ NB ΩB S₀' SO s x c').2) := by
    simp only [Equiv.trans_apply, Equiv.apply_symm_apply]
    rfl
  rw [hq, he₀]

end Swap

/-- **The deployed recursion's prefix and sessions as a `Rec.Compiled`**: the first part (`preP` at `preTape`, read
by `preOf`), its stage message `stgOf` and the deployed stop `sR`; session `j` of `(nums x j).rounds + 1` rounds at the
V\* statement `xsOf` stages, with the witness `wit` gives, on `G`'s coins; its simulator's coins `CB t`, and its
bound's terms the record's per-session error and `4t·3^t`. -/
noncomputable def recCompiled (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (R : Type) [Fintype R] [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (wit : Claims.Flock.Wit → (R × (ℕ → Salt)) × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)]
    [∀ t x j, Nonempty (CB t x j)] (t : ℕ) :
    Verity.Rec.Compiled Claims.Flock.Wit Claims.Flock.Wit Salt XorBytes where
  kI x := IV.k (xIOf x)
  nC x := b.nC x
  mask := maskX
  saltMsg y := Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn y
  str := hmLeaf
  stg x := stgOf (b.commit R dI dg wo) x (IV.k (xIOf x))
  Rnd _ := R
  finRnd _ := inferInstanceAs (Fintype R)
  neRnd _ := inferInstanceAs (Nonempty R)
  P₀ w x := preP IV (b.commit R dI dg wo) (preOf IV b R x) w
  R := Claims.Recursion.R (deployRec IV ivScope (b.commit R dI dg wo)) I
  stop := sR IV ivScope (b.commit R dI dg wo)
  nS x := nS x
  n x j := (nums (b.commit R dI dg wo).st x j).rounds + 1
  xs := xsOf IV (b.commit R dI dg wo)
  ws j w x t₀ u := wit w (preOf IV b R x t₀) (u.take (IV.k (xIOf x))) j
  NB _ := G.Nonce
  ΩB _ := G.Rand
  finNB _ := inferInstanceAs (Fintype G.Nonce)
  neNB _ := inferInstanceAs (Nonempty G.Nonce)
  finΩB _ := inferInstanceAs (Fintype G.Rand)
  neΩB _ := inferInstanceAs (Nonempty G.Rand)
  CS x j := CB t x j
  finCS x j := inferInstanceAs (Fintype (CB t x j))
  neCS x j := inferInstanceAs (Nonempty (CB t x j))
  εS δ₁ x j := ((deployRec IV ivScope (b.commit R dI dg wo)).nums x j).εzk t δ₁
  γS _ := 4 * t * 3 ^ t

/-- **The recursion's outer system**: Flock's `--zk` session at the recursive setup, `VO.k + 1` rounds with its
`Hello`, the session prover `G.prover` (through `relabP`), the outer record's relation, the models' simulator at the
attempt budget `t`, and its proved error and collision weight. -/
noncomputable def recOuter (relO : Readable.Parsed → Readable.Relation) (G : Proto Claims.Flock.Wit)
    [Fintype G.Coin] (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (t : ℕ) : Verity.ZkSystem Claims.Flock.Wit (fun _ => G.Nonce) (fun _ => G.Rand) where
  k x := (deployZkRec relO).V.k x + 1
  P w _ := Verity.relabP G.prover w
  R := Claims.Flock.R (deployZkRec relO).parse
  S := (G.withSim Mc).simulatorE t (t * 3 ^ t)
  εzk δ₁ x := 8 * nL x * (δ₁ : ℝ) + 8 * t * nB x * (δ₁ : ℝ) + 2 * (1 / 3) ^ t
  γzk _ := 4 * t * 3 ^ t

section Recursion

variable (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
  (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
  (R : Type) [Fintype R] [Nonempty R]
  (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
  (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
  (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
  (wit : Claims.Flock.Wit → (R × (ℕ → Salt)) × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
  (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
  (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
  (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)] (t : ℕ)

include IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t

set_option hygiene false in
local notation "𝔟" => Brings.commit b R dI dg wo

set_option hygiene false in
local notation "𝔊" => recCompiled IV ivScope b I R dI dg wo wit G CB t

set_option hygiene false in
local notation "𝔒" => recOuter relO G Mc nL nB t

theorem recStart_eq (T : Type) (x : Claims.Msg) : ∀ j, (𝔊).start x j = (parts IV 𝔟 T).start x j
  | 0 => rfl
  | j + 1 => by
    show (𝔊).start x j + (𝔊).n x j = (parts IV 𝔟 T).start x j + (parts IV 𝔟 T).n x j
    rw [recStart_eq T x j]
    all_goals rfl

/-- **`Rec.Compiled.lay`'s prover at the deployed objects is the masked run's** (`lay`), session by session. -/
theorem layProver_rec (w : Claims.Flock.Wit) (x : Claims.Msg) :
    ∀ (m : ℕ) (hist : List (Claims.Msg × Claims.Msg))
      (ω : ((preTape IV b R x).N × (preTape IV b R x).Ω) × (ℕ → G.Nonce × G.Rand)),
      (Layout.prover ((𝔊).lay x) (fun w => (𝔊).P₀ w x) (Verity.Rec.slotP 𝔒 G.Nonce G.Rand) m w).next x hist ω =
        ((lay IV 𝔟 (preOf IV b R x) wit).prover (preP IV 𝔟 (preOf IV b R x)) G.prover m w).next x hist ω
  | 0, _, _ => rfl
  | m + 1, hist, ω => by
    have hs : ((𝔊).lay x).start x m = (lay IV 𝔟 (preOf IV b R x) wit).start x m := by
      rw [Verity.Rec.Compiled.lay_start]
      exact (recStart_eq IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t
        ((preTape IV b R x).N × (preTape IV b R x).Ω) x m).trans
        (congrFun (congrFun (lay_start IV 𝔟 (preOf IV b R x) wit) x) m).symm
    simp only [Layout.prover, hs, layProver_rec w x m hist ω,
      Verity.slotP_relab 𝔒 G.prover (fun _ _ => rfl)]
    all_goals rfl

/-- **The deployed record's prover is `Rec.prover`'s**: the masked reference prover with the stop, read flat. -/
theorem prover_rec_next (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : (preTape IV b R x).at 0 G.Nonce G.Rand (nS x)) :
    (Verity.Rec.prover 𝔒 𝔊 w x).next x hist ω =
      ((Claims.ZKMal.stopP (refPM IV 𝔟 (preOf IV b R x) wit G.prover w) (sR IV ivScope 𝔟) (kPre IV)).comap
        (Layout.flatten (preTape IV b R x) 0 G.Nonce G.Rand (nS x))).next x hist ω :=
  congrArg (fun o => if Claims.ZKMal.fired (sR IV ivScope 𝔟 x) (kPre IV x) hist = true then Claims.OC.ret [] else o)
    (layProver_rec IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t w x (nS x) hist
      (Layout.flatten (preTape IV b R x) 0 G.Nonce G.Rand (nS x) ω))

theorem zkAt_rec_prover (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) {k : ℕ} {RA : Claims.Flock.Wit → Prop}
    {C : Type} [Fintype C] {prog : C → Claims.SP (List (Claims.Msg × Claims.Out))} {ε γ : ℝ}
    (h : ZKmalAt H k (fun w => Verity.Rec.prover 𝔒 𝔊 w x) RA x C prog ε γ) :
    ZKmalAt H k (fun w => (Claims.ZKMal.stopP (refPM IV 𝔟 (preOf IV b R x) wit G.prover w) (sR IV ivScope 𝔟)
      (kPre IV)).comap (Layout.flatten (preTape IV b R x) 0 G.Nonce G.Rand (nS x))) RA x C prog ε γ :=
  h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun w hist ω =>
    (prover_rec_next IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t w x hist ω).symm

/-- **`Rec.sim` at the deployed objects is `simMStop` at the claim's first-part simulator**: the same sessions after
the same stop, the first part `Rec.Compiled.S₀`'s zero messages, which are `simPre`'s on the coins of `simPreCoins`. -/
theorem zkAt_rec_sim (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x))
    {N Ω WP : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {k : ℕ} {P : WP → Claims.Prover (N × Ω)}
    {RA : WP → Prop} {ε γ : ℝ}
    (h : ZKmalAt H k P RA x ((Verity.Rec.sim 𝔒 𝔊 fun _ _ => Claims.Recursion.goesOn).C x)
      ((Verity.Rec.sim 𝔒 𝔊 fun _ _ => Claims.Recursion.goesOn).prog x) ε γ) :
    ZKmalAt H k P RA x
      ((simMStop IV 𝔟 (preTape IV b R x) G.Nonce G.Rand (simPre (deployRec IV ivScope 𝔟) I)
        (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) (sR IV ivScope 𝔟)).C x)
      ((simMStop IV 𝔟 (preTape IV b R x) G.Nonce G.Rand (simPre (deployRec IV ivScope 𝔟) I)
        (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) (sR IV ivScope 𝔟)).prog x) ε γ := by
  have he₀ : ∀ (ask : List Claims.Msg → Claims.Out) (c : (simPre (deployRec IV ivScope 𝔟) I).C x),
      (Claims.SP.eval H ask ((simPre (deployRec IV ivScope 𝔟) I).prog x c)).1 =
        (Claims.SP.eval H ask ((𝔊).S₀.prog x (simPreCoins hI (b.nC x) c))).1 := fun ask c =>
    (simPre_eval IV 𝔟 ivScope I H ask x c).trans ((Claims.ZKMal.prefixRun_cast hmLeaf maskX (saltH H) ask hI
      ((𝔟).nC x) (stgOf 𝔟 x) c.1 (c.2 ⟨0, Nat.zero_lt_one⟩)).trans
        ((𝔊).S₀_eval H x ask (simPreCoins hI (b.nC x) c)).symm)
  have h1 := Claims.ZKMal.zkAt_of_sim_eq (Layout.iterSimG_seq_congr ((𝔊).T₀ x) 0 ((𝔊).NB x) ((𝔊).ΩB x)
    (((𝔊).lay x).stopped (𝔊).stop) ((parts IV 𝔟 (((𝔊).T₀ x).N × ((𝔊).T₀ x).Ω)).stopped (sR IV ivScope 𝔟))
    rfl rfl rfl (Claims.ZKMal.censorSim (𝔊).S₀ (𝔊).stop ((𝔊).lay x).k₀) (fun j x' => (𝔊).CS x' j)
    (Claims.ZKMal.stopProgB (𝔊).stop ((𝔊).lay x).k₀ ((𝔊).lay x).n
      fun j x' u => Claims.ZKMal.progOf (𝔒).S ((𝔊).CS x' j) ((𝔊).xs x' j u))
    (fun _ _ => Claims.Recursion.goesOn) ((𝔊).nS x)) h
  exact zkAt_simMStop_swap IV 𝔟 H (preTape IV b R x) G.Nonce G.Rand (𝔊).S₀ (simPre (deployRec IV ivScope 𝔟) I)
    (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) (sR IV ivScope 𝔟) x (simPreCoins hI (b.nC x)) he₀ h1

/-- **Z1, `O` is zero knowledge**: Flock's at the outer record from the session's tables at the recursive setup
(`Flock.zkAt_deployZkRec_of_tables_bind`), at each statement on `G`'s coins. -/
theorem recOuter_zk (ht : 0 < t) (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512) (δ₁ : ℝ≥0) (hδ : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) :
    (𝔒).ZK FlockDeploy.sha512 δ₁ := by
  have h := (RecDeploy.Flock.zkAt_deployZkRec_of_tables_bind FlockDeploy.sha512 relO G t ht hK hδ
    ENNReal.coe_ne_top Mp Mc nL nB hnL hnB hL hOK hC).2
  simp only [ENNReal.coe_toReal] at h
  exact (Claims.ZKMal.zkmalRx_iff_at _ _ _ _ _ _ _ _ _).2 fun x =>
    ((Claims.ZKMal.zkmalR_iff_at _ _ _ _ _ _ _).1 h x).of_equiv _ (Verity.relab G.Nonce G.Rand).1.symm
      (Verity.relab G.Nonce G.Rand).2.symm fun _ _ _ => rfl

/-- **Z2, the first part's strings are hm96 commitments**: each inner round's of the digest `dI` reads, and the stage
round's of the staged values' digests `dg` reads, neither reading the salts it commits under. -/
theorem recCompiled_commits (H : Claims.Msg → Claims.Msg) : (𝔊).Commits H := by
  intro x w _
  refine ⟨fun ρ h => asX (dI w ρ h),
    fun p h j => asX ((dg w (p.1, saltsOn p.2) (h.take (IV.k (xIOf x)))).getD j 0), fun t₀ h => ⟨fun hh => ?_,
      fun hh => ?_⟩⟩
  · have hh' : h.length < IV.k (xIOf x) := hh
    have e : saltsOn t₀.1.2 h.length = t₀.1.2 ⟨h.length, hh'⟩ := dite_eq_left hh'
    have e1 : ((((𝔊).P₀ w x).next x h t₀).eval H).1 =
        List.ofFn (fun i => dI w t₀.1.1 h i ^^^ maskOf (saltsOn t₀.1.2 h.length) i) ++
          H (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn (saltsOn t₀.1.2 h.length)) := by
      show (((preP IV 𝔟 (preOf IV b R x) w).next x h t₀).eval H).1 = _
      simp only [preP, hh', ↓reduceIte]
      rfl
    rw [e1, e]
    all_goals rfl
  · have hh' : h.length = IV.k (xIOf x) := hh
    have e : ((((𝔊).P₀ w x).next x h t₀).eval H).1 =
        ((commitThen (fun cs => .ret (stageMsg (𝔟).hs x (asRounds (h.take (IV.k (xIOf x)))) cs))
          ((List.range (b.nC x)).map fun j =>
            ((dg w (t₀.1.1, saltsOn t₀.1.2) (h.take (IV.k (xIOf x)))).getD j 0, saltsOn t₀.2 j)) []).eval H).1 := by
      show (((preP IV 𝔟 (preOf IV b R x) w).next x h t₀).eval H).1 = _
      simp only [preP, hh', lt_irrefl, ↓reduceIte]
      rfl
    rw [e, commitThen_eval]
    simp only [Claims.OC.eval, List.nil_append]
    show stageMsg (𝔟).hs x (asRounds (h.take (IV.k (xIOf x)))) _ =
      stageMsg (𝔟).hs x (asRounds (h.take (IV.k (xIOf x)))) (List.ofFn (n := b.nC x) _)
    congr 1
    refine List.ext_getElem ?_ fun j h1 h2 => ?_
    · simp only [List.length_map, List.length_range]
      exact List.length_ofFn.symm
    · have hj : j < b.nC x := by simpa using h1
      have hs : saltsOn t₀.2 j = t₀.2 ⟨j, hj⟩ := dite_eq_left hj
      simp only [List.getElem_map, List.getElem_range, hs]
      exact (List.getElem_ofFn h2).symm

/-- **Z3, each session's witness is in `O`'s relation unless the stop fired**: the live witness (`LiveWit`). -/
theorem recCompiled_witnessIn
    (hW : LiveWit FlockDeploy.sha512 IV ivScope 𝔟 I (preTape IV b R) (preOf IV b R) wit
      (Claims.Flock.R (deployZkRec relO).parse)) :
    (𝔊).WitnessIn FlockDeploy.sha512 𝔒 := by
  intro x j hj w hw t₀ u hu hp hf
  exact hW x w hw j hj t₀ u (hu.trans (recStart_eq IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t
    ((preTape IV b R x).N × (preTape IV b R x).Ω) x j)) hp hf

/-- **Z4, each session's shape is public**: `VO`'s rounds at each staged statement, the leaf counts within the
record's (`hzl`), the coin bijections (`hCB`), and the session's coins `G`'s at every statement. -/
theorem recCompiled_byShape (ht : 0 < t)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (δ₁ : ℝ≥0) : (𝔊).ByShape 𝔒 δ₁ := by
  refine ⟨fun _ => by show (0 : ℝ) ≤ 4 * t * 3 ^ t; positivity, fun x j hj => ⟨by
    show (0 : ℝ) ≤ ((deployRec IV ivScope 𝔟).nums x j).εzk t δ₁
    exact nums_εzk_nonneg _ t δ₁, fun u hu _ => ?_⟩⟩
  have hu' : u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j :=
    hu.trans (recStart_eq IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t _ x j)
  obtain ⟨h1, h2⟩ := hzl x j hj u hu'
  refine ⟨?_, ?_, le_rfl, hCB t ht x j hj u hu', ⟨(Equiv.refl _, Equiv.refl _)⟩⟩
  · show VO.k (xsOf IV 𝔟 x j u) + 1 = (nums (𝔟).st x j).rounds + 1
    rw [VO_k_xsOf IV 𝔟 x j u hj]
  · show 8 * (nL (xsOf IV 𝔟 x j u) : ℝ) * (δ₁ : ℝ) + 8 * (t : ℝ) * (nB (xsOf IV 𝔟 x j u) : ℝ) * (δ₁ : ℝ) +
        2 * (1 / 3 : ℝ) ^ t ≤ ((deployRec IV ivScope 𝔟).nums x j).εzk t δ₁
    rw [nums_εzk]
    have hδ0 : (0 : ℝ) ≤ δ₁ := δ₁.2
    have e1 : (nL (xsOf IV 𝔟 x j u) : ℝ) ≤ ((𝔟).zl (staged (𝔟).st x (pick (𝔟).st x j) j)).1 := by
      exact_mod_cast h1
    have e2 : (nB (xsOf IV 𝔟 x j u) : ℝ) ≤ ((𝔟).zl (staged (𝔟).st x (pick (𝔟).st x j) j)).2 := by
      exact_mod_cast h2
    gcongr

/-- **`RecZK` at the deployed recursion**: `Rec.system 𝔒 𝔊` at `Recursion.goesOn` is zero knowledge, from its four
premises at the deployed objects. -/
theorem recSystem_zk (ht : 0 < t) (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : LiveWit FlockDeploy.sha512 IV ivScope 𝔟 I (preTape IV b R) (preOf IV b R) wit
      (Claims.Flock.R (deployZkRec relO).parse))
    (δ₁ : ℝ≥0)
    (hδ : Claims.Hm96Hiding (deployRec IV ivScope 𝔟).mask ((deployRec IV ivScope 𝔟).saltHash FlockDeploy.sha512) δ₁) :
    (Verity.Rec.system 𝔒 𝔊 fun _ _ => Claims.Recursion.goesOn).ZK FlockDeploy.sha512 δ₁ :=
  Verity.SecurityProofs.RecZK FlockDeploy.sha512 𝔒 𝔊 δ₁ _ (𝔊).goesOnAt_recursion
    (recOuter_zk IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t ht hK hnL hnB hL hOK hC δ₁
      (hiding_sha512 IV ivScope 𝔟 δ₁ hδ))
    hδ (recCompiled_commits IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t FlockDeploy.sha512)
    (recCompiled_byShape IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t ht hzl hCB δ₁)
    (recCompiled_witnessIn IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t hW)

end Recursion

/-- **The recursion's `ZKStop` at the masked record with the deployed stop, from `RecZK`**: the statement of
`zkStop_deployRecMStop_commit_sha512`, on its premises, as `Verity.SecurityProofs.RecZK` at the deployed recursion
(`recSystem_zk`): the record's prover is `Rec.prover`'s (`zkAt_rec_prover`), `SimStop` is `Rec.sim`'s on bijective
coins (`zkAt_rec_sim`, `zkAt_simStop`), and `εzkStop` and `γzk` are `Rec.system`'s bound's terms. -/
theorem zkStop_of_recZK (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (R : Type) [Fintype R] [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (wit : Claims.Flock.Wit → (R × (ℕ → Salt)) × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hk : VM.k = kM IV b) (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : LiveWit FlockDeploy.sha512 IV ivScope (b.commit R dI dg wo) I (preTape IV b R) (preOf IV b R) wit
      (Claims.Flock.R (deployZkRec relO).parse)) :
    Claims.Recursion.ZKStop FlockDeploy.sha512
      (deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
        G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t))
      I (deployZkRec relO) := by
  refine ⟨rfl, fun t ht δ₁ hδ => ?_⟩
  rw [prover_deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
    G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)]
  intro x
  have h1 := (Claims.ZKMal.zkmalRx_iff_at _ _ _ _ _ _ _ _ _).1
    (recSystem_zk IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t ht hK hnL hnB hL hOK hC hzl hCB hW δ₁ hδ) x
  have hkx : (Verity.Rec.system (recOuter relO G Mc nL nB t) (recCompiled IV ivScope b I R dI dg wo wit G CB t)
      fun _ _ => Claims.Recursion.goesOn).k x = VM.k x :=
    (recStart_eq IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t
      ((preTape IV b R x).N × (preTape IV b R x).Ω) x (nS x)).trans
      ((parts_start IV (b.commit R dI dg wo) ((preTape IV b R x).N × (preTape IV b R x).Ω) x (nS x)).trans
        (congrFun hk x).symm)
  refine (zkAt_simStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
    G.prover (fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) FlockDeploy.sha512 I
    (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) x (hI x) hk
    (zkAt_rec_sim IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t FlockDeploy.sha512 x (hI x)
      (zkAt_rec_prover IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB t FlockDeploy.sha512 x
        (Claims.ZKMal.zkmalAt_k_eq hkx h1)))).mono (le_of_eq ?_) (le_of_eq ?_)
  · rw [εzkStop_deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R)
      wit G.prover (fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) I t δ₁ x]
    show 2 * ((IV.k (xIOf x) : ℝ) + (b.nC x : ℝ)) * (δ₁ : ℝ) +
        ∑ j ∈ Finset.range (nS x), ((deployRec IV ivScope (b.commit R dI dg wo)).nums x j).εzk t δ₁ = _
    simp only [Claims.Recursion.εzk]
    rw [Fin.sum_univ_eq_sum_range (fun j => ((deployRec IV ivScope (b.commit R dI dg wo)).nums x j).εzk t δ₁)]
    simp only [xI_deployRec]
    simp only [deployRec]
    rw [hI x]
    simp only [kS, Nat.cast_one, one_mul]
    ring
  · rfl

/-- **The deployed recursion's release leaks at most `log₂ (VM.k x + 1)` bits beyond `ZKStop`'s bound**: at the
masked record with the deployed stop, the released prover (`Verity.haltP`, at any halt test reading the witness, the
tape, the prover's coins `K x` and the history) is `ZKmalRLeak` at the leaky `SimStop`, from `zkStop_of_recZK`
(`Verity.zkStop_leak_haltP`). -/
theorem recRelease_leak (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (R : Type) [Fintype R] [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (wit : Claims.Flock.Wit → (R × (ℕ → Salt)) × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hk : VM.k = kM IV b) (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : LiveWit FlockDeploy.sha512 IV ivScope (b.commit R dI dg wo) I (preTape IV b R) (preOf IV b R) wit
      (Claims.Flock.R (deployZkRec relO).parse))
    {K : Claims.Msg → Type} [∀ x, Fintype (K x)] [∀ x, Nonempty (K x)] {t : ℕ} (ht : 0 < t) {δ₁ : ℝ≥0}
    (hδ : Claims.Hm96Hiding (deployRec IV ivScope (b.commit R dI dg wo)).mask
      ((deployRec IV ivScope (b.commit R dI dg wo)).saltHash FlockDeploy.sha512) δ₁)
    (halt : Claims.Flock.Wit → (x : Claims.Msg) →
      (((preTape IV b R x).iter 0 G.Nonce G.Rand (nS x)).N × ((preTape IV b R x).iter 0 G.Nonce G.Rand (nS x)).Ω) →
        K x → List (Claims.Msg × Claims.Msg) → Bool) :
    Claims.ZKmalRLeak (fun x => ((preTape IV b R x).iter 0 G.Nonce G.Rand (nS x)).N)
      (fun x => ((preTape IV b R x).iter 0 G.Nonce G.Rand (nS x)).Ω) K FlockDeploy.sha512 VM.k
      (Verity.haltP VM.k
        (deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
          G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)).prover halt)
      (deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
        G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)).prover
      (Claims.Recursion.R (deployRec IV ivScope (b.commit R dI dg wo)) I)
      (Claims.leakyAt (Claims.Recursion.SimStop
          (deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
            G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) I
          (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) FlockDeploy.sha512)
        VM.k [] (fun x => Finset.range (VM.k x + 1)) fun _ => Finset.nonempty_range_add_one)
      (fun x => ((VM.k x + 1 : ℕ) : ℝ))
      (Claims.Recursion.εzkStop
        (deployRecMStop IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
          G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t)) I t δ₁)
      (Claims.Recursion.γzk (deployRec IV ivScope (b.commit R dI dg wo)) t) :=
  Verity.zkStop_leak_haltP
    (zkStop_of_recZK IV b ivScope I relO VM iv R dI dg wo wit G Mp Mc nL nB CB hk hI hK hnL hnB hL hOK hC hzl hCB hW)
    ht hδ halt

end RecDeploy
