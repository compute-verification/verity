import Proofs.Flock.Claims.Recursion.SeqStop
import Proofs.Flock.Claims.Recursion.SeqDeploy
import Proofs.Flock.Claims.Recursion.DeployStop

/-!
# The recursion's `ZKStop` at the masked record with the deployed stop

`Claims.Recursion.ZKStop` at `deployRecMStop`: the masked record `deployRecM` whose reference prover stops where the
deployed stop `stopsRec` holds at the first part's rounds (`stopP (refPM …) (sR …) (kPre IV)`), each inner round one
commit string (`nAt = 1`, `innerMsg` the commit string itself), so `εzkStop` is `εzk`.

* `prover_deployRecMStop`: the record's prover at `x` is `zkAtStop`'s, the masked reference prover with the stop at
  `pre x` on `x`'s tape read flat.
* `simStop_eval`: `SimStop` at the record is `simMStop` at the claim's first-part simulator, against every verifier,
  on the coins `stopEquiv` matches: the first part is a run of `pmZ` with the stop (`simPre_pm`,
  `Claims.ZKMal.interact_censor`), and past it the sessions run where the stop hasn't fired and are silent where it
  has (`tail_eval`).
* `zkStop_deployRecMStop`: `ZKStop` from the first part's zero knowledge (`h₀`), each session's where the stop hasn't
  fired (`hS`), and the sessions' witness at every live first part (`LiveWit`).
* `zkStop_deployRecMStop_commit_sha512`: at SHA-512, the inner prover that commits, and the session's tables at the
  recursive setup, whose premises remain, with `LiveWit` at the outer record's relation.
-/

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts SeqTape Proto ZKmalAt iterSimG)
open FlockSoundness.Discharge
open scoped NNReal

/-- **The deployed stop on the prover's history**: `stopsRec` at the transcript the history reads as. -/
noncomputable def sR (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (x : Claims.Msg)
    (h : List (Claims.Msg × Claims.Msg)) : Bool :=
  stopsRec IV ivScope b.st x (asRounds h)

/-- **The masked record with the deployed stop**: `deployRecM` with the masked reference prover stopped where `sR`
holds at the first part's rounds, and `stopsRec` its stop; each inner round one commit string, sent as it is. -/
noncomputable def deployRecMStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.StopDeploy Coin :=
  { deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO with
    P := fun w => Claims.ZKMal.stopP (refPM IV b id wit PS w) (sR IV ivScope b) (kPre IV)
    nAt := fun _ _ => 1
    innerMsg := fun _ _ cs => Claims.OC.ret cs.flatten
    stops := fun _ => stopsRec IV ivScope b.st }

/-- **The live witness**: at every history that starts session `j` whose first part is the first part's prover's on
some tape `t₀` and where the stop hasn't fired, the witness `wit` gives is in the sessions' relation `RF` at the
statement the history stages. -/
def LiveWit (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape)
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (RF : Claims.Rel Claims.Flock.Wit) : Prop :=
  ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
    (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
      (∀ i (hi : i < u.length), i < kPre IV x → u[i].1 = (((preP IV b (pre x) w).next x (u.take i) t₀).eval H).1) →
      Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) u = false →
        RF (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)

section Stop

variable (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (VM : Claims.Verifier Coin)
  (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
  (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
  (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
  (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
  (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)

set_option hygiene false in
local notation "𝔇" => deployRecMStop IV ivScope b VM iv T₀ NB ΩB pre wit PS SO

set_option hygiene false in
local notation "𝔓" => Claims.ZKMal.stopProgB (sR IV ivScope b) (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
  fun j x u c => (SO' x j).prog (xsOf IV b x j u) c

theorem nAt_deployRecMStop (x : Claims.Msg) (i : ℕ) : (𝔇).nAt x i = 1 := rfl

/-- The record's commit strings to the zero digest are `deployRec`'s. -/
theorem zeroCommits_deployRecMStop (ys : List (Fin (𝔇).saltLen → UInt8)) :
    (𝔇).toDeploy.zeroCommits ys = (deployRec IV ivScope b).zeroCommits ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih => simp only [Claims.Recursion.Deploy.zeroCommits, ih]; rfl

theorem zeroCommits_stop_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (ys : List (Fin (𝔇).saltLen → UInt8)) :
    (Claims.SP.eval H ask ((𝔇).toDeploy.zeroCommits ys)).1 = ys.map fun y => List.ofFn (maskOf y) ++ saltH H y := by
  rw [zeroCommits_deployRecMStop]
  exact zeroCommits_eval IV b ivScope H ask ys

theorem innerMsg_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) (cs : List Claims.Msg) :
    (Claims.SP.eval H ask (Claims.Recursion.spOf ((𝔇).innerMsg x ms cs))).1 = cs.flatten :=
  rfl

/-- **The record's stop is `sR`'s** on the transcript a history reads as. -/
theorem halted_asRounds (H : Claims.Msg → Claims.Msg) (k : ℕ) (x : Claims.Msg) (h : List (Claims.Msg × Claims.Msg)) :
    (𝔇).halted H k x (asRounds h) = Claims.ZKMal.fired (sR IV ivScope b x) k h := by
  have hl : (asRounds h).length = h.length := List.length_map _
  unfold Claims.Recursion.StopDeploy.halted Claims.ZKMal.fired
  rw [hl]
  congr 1
  funext i
  show stopsRec IV ivScope b.st x ((asRounds h).take i) = stopsRec IV ivScope b.st x (asRounds (h.take i))
  first
    | simp only [asRounds, List.map_take]
    | simp [asRounds, List.map_take]

/-- **The record's prover at `x` is `zkAtStop`'s**: the masked reference prover with the stop at `pre x` on the
nested tape of `x`'s sessions read flat. -/
theorem prover_deployRecMStop : (𝔇).prover =
    fun w x => (Claims.ZKMal.stopP (refPM IV b (pre x) wit PS w) (sR IV ivScope b) (kPre IV)).comap
      (RunParts.flatten (T₀ x) 0 NB ΩB (nS x)) := by
  funext w x
  exact congrArg Claims.Prover.mk (funext fun x' => funext fun hist => funext fun ω =>
    congrArg (fun o => if Claims.ZKMal.fired (sR IV ivScope b x') (kPre IV x') hist = true then Claims.OC.ret []
        else o)
      (layProver_comap IV b (pre x) wit PS w (nS x') x' hist (RunParts.flatten (T₀ x) 0 NB ΩB (nS x) ω)))

theorem innerRound_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) (y : Fin 1 → Fin (𝔇).saltLen → UInt8) :
    (Claims.SP.eval H ask (Claims.Recursion.spBind ((𝔇).toDeploy.zeroCommits ((List.finRange 1).map y)) fun cs =>
      Claims.Recursion.spOf ((𝔇).innerMsg x ms cs))).1 =
        List.ofFn (maskOf (y ⟨0, Nat.zero_lt_one⟩)) ++ saltH H (y ⟨0, Nat.zero_lt_one⟩) := by
  simp only [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval, zeroCommits_stop_eval, innerMsg_eval,
    Claims.ZKMal.finRange_one, List.map_cons, List.map_nil]
  first
    | exact List.append_nil _
    | simp

theorem stageRound_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) (ys : List (Fin (𝔇).saltLen → UInt8)) :
    (Claims.SP.eval H ask (Claims.Recursion.spBind ((𝔇).toDeploy.zeroCommits ys) fun cs =>
      Claims.Recursion.spOf ((𝔇).stage x ms cs))).1 =
        stageMsg b.hs x ms (ys.map fun y => List.ofFn (maskOf y) ++ saltH H y) := by
  simp only [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval, zeroCommits_stop_eval]
  rfl

/-- **`SimStop`'s coins are `simMStop`'s** at the claim's first-part simulator: the one salt of each inner round, the
staging salts, and one per V\* session. -/
noncomputable def stopEquiv (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (SO' : Claims.Msg → ℕ → Claims.Recursion.OSim) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) :
    (Claims.Recursion.SimStop 𝔇 I SO' H).C x ≃
      (simMStop IV b (T₀ x) NB ΩB (simPre (deployRec IV ivScope b) I) SO' (sR IV ivScope b)).C x :=
  (Equiv.prodCongr (Equiv.piCongrRight fun _ => Equiv.funUnique (Fin 1) (Fin (𝔇).saltLen → UInt8))
    (Equiv.refl _)).trans ((Equiv.prodAssoc _ _ _).symm.trans (Claims.ZKMal.iterCoins (T₀ x) 0 NB ΩB
      (RunParts.seq (T₀ x) 0 NB ΩB ((parts IV b ((T₀ x).N × (T₀ x).Ω)).stopped (sR IV ivScope b)))
      (Claims.ZKMal.censorSim (simPre (deployRec IV ivScope b) I) (sR IV ivScope b) (kPre IV))
      (fun j x => (SO' x j).C)
      (Claims.ZKMal.stopProgB (sR IV ivScope b) (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
        fun j x u c => (SO' x j).prog (xsOf IV b x j u) c)
      (fun _ _ => Claims.Recursion.goesOn) x (nS x)).symm)

/-- **Past the first part, `SimStop` is `simMStop`'s sessions**: after a first part of `kPre x` rounds of messages,
where the stop has fired the empty message for the rest of `VM`'s rounds, as many as the sessions' silent rounds;
where it hasn't, each session's outer simulator at the V\* statement the history stages. -/
theorem tail_eval (H : Claims.Msg → Claims.Msg) (SO' : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg)
    (hk : VM.k = kM IV b) (ask : List Claims.Msg → Claims.Out) (cs : (s : Fin ((𝔇).nS x)) → (SO' x s).C)
    (τ : List (Claims.Msg × Claims.Out))
    (hτ : Claims.Recursion.goesOn τ = true → τ.length = kPre IV x ∧ ∀ p ∈ τ, Claims.ZKMal.isMsg p.2 = true) :
    (Claims.SP.eval H ask (if (𝔇).halted H (kPre IV x) x τ then
        Claims.Recursion.simRounds (fun _ => .ret []) ((𝔇).V.k x - kPre IV x) τ
      else Claims.Recursion.simOuter SO' x ((𝔇).vstar x τ) cs (List.finRange ((𝔇).nS x)) τ)).1 =
      (List.finRange ((𝔇).nS x)).foldl
        (fun τ' (j : Fin ((𝔇).nS x)) => stepS H ask SO' x 𝔓 τ' j (cs j)) τ := by
  by_cases hg : Claims.Recursion.goesOn τ = true
  · obtain ⟨hl, hm⟩ := hτ hg
    have hh : (𝔇).halted H (kPre IV x) x τ =
        Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) (Claims.ZKMal.histOf τ) := by
      rw [← halted_asRounds IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H (kPre IV x) x (Claims.ZKMal.histOf τ),
        asRounds_histOf hm]
    rw [hh]
    have hl' : (Claims.ZKMal.histOf τ).length = kPre IV x := by rw [Claims.ZKMal.histOf_length, hl]
    by_cases hf : Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) (Claims.ZKMal.histOf τ) = true
    · rw [if_pos hf, simRounds_interact H ask _ _ τ hm]
      have hfold : (List.finRange ((𝔇).nS x)).foldl
          (fun τ' (j : Fin ((𝔇).nS x)) => stepS H ask SO' x 𝔓 τ' j (cs j)) τ =
          (List.finRange ((𝔇).nS x)).foldl (fun τ' (j : Fin ((𝔇).nS x)) => if Claims.Recursion.goesOn τ' then
            τ' ++ Claims.ZKMal.interact ask (fun _ => []) ((nums b.st x j).rounds + 1) (Claims.ZKMal.histOf τ')
            else τ') τ := by
        refine Claims.ZKMal.foldl_congr_inv (fun τ' => ∃ r, τ' = τ ++ r) _ _ (fun τ' j hP => ?_) _ τ
          ⟨[], (List.append_nil τ).symm⟩
        obtain ⟨r, rfl⟩ := hP
        unfold stepS
        by_cases hg' : Claims.Recursion.goesOn (τ ++ r) = true
        · have hfr : Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) (Claims.ZKMal.histOf (τ ++ r)) = true := by
            rw [Claims.ZKMal.histOf_append, Claims.ZKMal.fired_append (sR IV ivScope b x) (kPre IV x)
              (Claims.ZKMal.histOf τ) (Claims.ZKMal.histOf r) hl']
            exact hf
          simp only [Claims.ZKMal.stopProgB]
          rw [if_pos hg', if_pos hg', if_pos hfr, Claims.ZKMal.silent_eval]
          exact ⟨rfl, _, List.append_assoc _ _ _⟩
        · rw [if_neg hg', if_neg hg']
          exact ⟨rfl, r, rfl⟩
      have hcount : ((List.finRange ((𝔇).nS x)).map fun j : Fin ((𝔇).nS x) => (nums b.st x j).rounds + 1).sum =
          (𝔇).V.k x - kPre IV x := by
        refine (Claims.ZKMal.sum_finRange_map (fun j => (nums b.st x j).rounds + 1) ((𝔇).nS x)).trans ?_
        show ((List.range (nS x)).map fun j => (nums b.st x j).rounds + 1).sum = VM.k x - kPre IV x
        rw [hk]
        unfold kM
        omega
      refine Eq.trans ?_ (hfold.trans (Claims.ZKMal.foldl_silent ask
        (fun j : Fin ((𝔇).nS x) => (nums b.st x j).rounds + 1) (List.finRange ((𝔇).nS x)) τ hm)).symm
      exact congrArg (fun n => τ ++ Claims.ZKMal.interact ask (fun _ => []) n (Claims.ZKMal.histOf τ)) hcount.symm
    · rw [if_neg hf, Claims.ZKMal.simOuter_eval]
      refine Claims.ZKMal.foldl_congr_inv (fun τ' => ∃ r, τ' = τ ++ r) _ _ (fun τ' j hP => ?_) _ τ
        ⟨[], (List.append_nil τ).symm⟩
      obtain ⟨r, rfl⟩ := hP
      unfold stepS
      by_cases hg' : Claims.Recursion.goesOn (τ ++ r) = true
      · have hfr : Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) (Claims.ZKMal.histOf (τ ++ r)) = false := by
          rw [Claims.ZKMal.histOf_append, Claims.ZKMal.fired_append (sR IV ivScope b x) (kPre IV x)
            (Claims.ZKMal.histOf τ) (Claims.ZKMal.histOf r) hl']
          exact Bool.eq_false_iff.2 hf
        simp only [Claims.ZKMal.stopProgB]
        rw [if_pos hg', if_pos hg', hfr, if_neg Bool.false_ne_true, xsOf_of_goesOn IV b x j τ r hl hm]
        exact ⟨rfl, _, List.append_assoc _ _ _⟩
      · rw [if_neg hg', if_neg hg']
        exact ⟨rfl, r, rfl⟩
  · have hg0 : Claims.Recursion.goesOn τ = false := Bool.eq_false_iff.2 hg
    have hR : (List.finRange ((𝔇).nS x)).foldl
        (fun τ' (j : Fin ((𝔇).nS x)) => stepS H ask SO' x 𝔓 τ' j (cs j)) τ = τ :=
      Claims.ZKMal.foldl_stuck _ (fun τ' i h => by simp [stepS, h]) _ τ hg0
    rw [hR]
    split
    · rw [simRounds_stop _ _ τ hg0] <;> rfl
    · rw [Claims.ZKMal.simOuter_eval]
      exact Claims.ZKMal.foldl_stuck _ (fun τ' i h => by simp [h]) _ τ hg0

/-- **`SimStop` at the record is `simMStop` at the claim's first-part simulator**: against every verifier, one
transcript on the coins `stopEquiv` matches, where the inner protocol's rounds at the inner statement are the inner
verifier's and the masked verifier's rounds are the masked run's. -/
theorem simStop_eval (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (SO' : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x))
    (hk : VM.k = kM IV b) (ask : List Claims.Msg → Claims.Out) (c : (Claims.Recursion.SimStop 𝔇 I SO' H).C x) :
    (Claims.SP.eval H ask ((Claims.Recursion.SimStop 𝔇 I SO' H).prog x c)).1 =
      (Claims.SP.eval H ask ((simMStop IV b (T₀ x) NB ΩB (simPre (deployRec IV ivScope b) I) SO'
        (sR IV ivScope b)).prog x (stopEquiv IV ivScope b VM iv T₀ NB ΩB pre wit PS SO I SO' H x c))).1 := by
  obtain ⟨c₁, c₂, cs⟩ := c
  have hK : I.V.k (Claims.Recursion.xI (𝔇).toDeploy x) + (𝔇).kS x = kPre IV x := by
    show I.V.k (xIOf x) + kS x = IV.k (xIOf x) + kS x
    rw [hI]
  have hpos : 0 < kPre IV x := by simp [kPre, kS]
  have hrhs := iterSimG_stopped_eval IV b H ask (T₀ x) NB ΩB
    (Claims.ZKMal.censorSim (simPre (deployRec IV ivScope b) I) (sR IV ivScope b) (kPre IV)) SO' (sR IV ivScope b) x
    𝔓 (nS x) (stopEquiv IV ivScope b VM iv T₀ NB ΩB pre wit PS SO I SO' H x (c₁, c₂, cs))
  have hc : Claims.ZKMal.iterCoins (T₀ x) 0 NB ΩB
      (RunParts.seq (T₀ x) 0 NB ΩB ((parts IV b ((T₀ x).N × (T₀ x).Ω)).stopped (sR IV ivScope b)))
      (Claims.ZKMal.censorSim (simPre (deployRec IV ivScope b) I) (sR IV ivScope b) (kPre IV))
      (fun j x => (SO' x j).C)
      (Claims.ZKMal.stopProgB (sR IV ivScope b) (kPre IV) (fun x j => (nums b.st x j).rounds + 1)
        fun j x u c => (SO' x j).prog (xsOf IV b x j u) c)
      (fun _ _ => Claims.Recursion.goesOn) x (nS x) (stopEquiv IV ivScope b VM iv T₀ NB ΩB pre wit PS SO I SO' H x
        (c₁, c₂, cs)) = (((fun i => c₁ i ⟨0, Nat.zero_lt_one⟩), c₂), cs) := by
    first
      | exact Equiv.apply_symm_apply _ _
      | (simp only [stopEquiv, Equiv.trans_apply, Equiv.apply_symm_apply]; rfl)
  rw [hc] at hrhs
  dsimp only at hrhs
  have hpre : (Claims.SP.eval H ask ((Claims.ZKMal.censorSim (simPre (deployRec IV ivScope b) I) (sR IV ivScope b)
      (kPre IV)).prog x ((fun i => c₁ i ⟨0, Nat.zero_lt_one⟩), c₂))).1 =
        Claims.ZKMal.interact ask (Claims.ZKMal.censor (sR IV ivScope b x) (kPre IV x)
          (pmZ b H x (I.V.k (xIOf x)) (fun i => c₁ i ⟨0, Nat.zero_lt_one⟩) (c₂ ⟨0, Nat.zero_lt_one⟩))) (kPre IV x) [] := by
    have hE := simPre_pm IV b ivScope I H ask x hI ((fun i => c₁ i ⟨0, Nat.zero_lt_one⟩), c₂)
    refine (Claims.ZKMal.censorSim_eval H ask (simPre (deployRec IV ivScope b) I) (sR IV ivScope b) (kPre IV) x
      ((fun i => c₁ i ⟨0, Nat.zero_lt_one⟩), c₂)).trans ?_
    refine Eq.trans ?_ (Claims.ZKMal.interact_censor ask
      (pmZ b H x (I.V.k (xIOf x)) (fun i => c₁ i ⟨0, Nat.zero_lt_one⟩) (c₂ ⟨0, Nat.zero_lt_one⟩))
      (sR IV ivScope b x) (kPre IV x) (kPre IV x)).symm
    exact congrArg (fun E => Claims.ZKServer.cutT ask [] (Claims.ZKMal.stopIdx (sR IV ivScope b x) (kPre IV x)
      (kPre IV x) E) (kPre IV x) E) hE
  replace hrhs := hrhs.trans (congrArg (fun τ₀ => List.foldl (fun τ (j : Fin (nS x)) => stepS H ask SO' x 𝔓 τ j (cs j))
    τ₀ (List.finRange (nS x))) hpre)
  try dsimp only at hrhs
  refine Eq.trans ?_ hrhs.symm
  simp only [Claims.Recursion.SimStop]
  refine Eq.trans (simRounds_tail_eval H ask _ _ _ (I.V.k (Claims.Recursion.xI (𝔇).toDeploy x)) ((𝔇).kS x)
    (Claims.ZKMal.censor (sR IV ivScope b x) (kPre IV x)
      (pmZ b H x (I.V.k (xIOf x)) (fun i => c₁ i ⟨0, Nat.zero_lt_one⟩) (c₂ ⟨0, Nat.zero_lt_one⟩))) rfl ?hf ?hg) ?tl
  case hf =>
    intro h hh
    try beta_reduce
    rw [halted_asRounds IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H _ x h, hK]
    unfold Claims.ZKMal.censor
    by_cases hf : Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) h = true
    · rw [if_pos hf, if_pos hf] <;> rfl
    · rw [if_neg hf, if_neg hf]
      have hl : (asRounds h).length = h.length := List.length_map _
      have h₁ : (asRounds h).length < I.V.k (Claims.Recursion.xI (𝔇).toDeploy x) := by
        rw [hl]; try exact hh
      rw [dif_pos h₁]
      refine (innerRound_eval IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H ask x (asRounds h)
        (c₁ ⟨(asRounds h).length, h₁⟩)).trans ?_
      unfold pmZ
      rw [dif_pos (show (asRounds h).length < I.V.k (xIOf x) from h₁)]
      try rfl
  case hg =>
    intro h hh
    try beta_reduce
    rw [halted_asRounds IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H _ x h, hK]
    unfold Claims.ZKMal.censor
    by_cases hf : Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) h = true
    · rw [if_pos hf, if_pos hf] <;> rfl
    · rw [if_neg hf, if_neg hf]
      have hl : (asRounds h).length = h.length := List.length_map _
      have hτ : (asRounds h).length - I.V.k (Claims.Recursion.xI (𝔇).toDeploy x) = 0 := by
        rw [hl, hh]; try exact Nat.sub_self _
      have h0 : (asRounds h).length - I.V.k (Claims.Recursion.xI (𝔇).toDeploy x) < (𝔇).kS x :=
        lt_of_eq_of_lt hτ Nat.zero_lt_one
      have hs : (⟨_, h0⟩ : Fin ((𝔇).kS x)) = ⟨0, Nat.zero_lt_one⟩ := Fin.ext hτ
      have hn : ¬ (asRounds h).length < I.V.k (xIOf x) := by
        rw [hl, hh]; try exact lt_irrefl _
      rw [dif_pos h0, hs]
      refine (stageRound_eval IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H ask x (asRounds h) _).trans ?_
      unfold pmZ
      rw [dif_neg hn]
      first
        | exact congrArg (stageMsg b.hs x (asRounds h)) (List.map_map ..)
        | (rw [List.map_map]; try rfl)
        | rfl
  case tl =>
    rw [hK]
    refine tail_eval IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H SO' x hk ask cs _ fun hg => ?_
    rw [Claims.ZKMal.recGoesOn_interact ask _ _ [] hpos] at hg
    exact ⟨Claims.ZKMal.goesOn_length hg, by
      simp only [Claims.ZKMal.goesOn, Bool.and_eq_true, List.all_eq_true] at hg
      exact hg.2⟩

/-- **`ZKmalAt` at `simMStop` is `ZKmalAt` at `SimStop`**, at the claim's first-part simulator. -/
theorem zkAt_simStop (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (SO' : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x))
    (hk : VM.k = kM IV b) {N Ω WP : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {k : ℕ}
    {P : WP → Claims.Prover (N × Ω)} {R : WP → Prop} {ε γ : ℝ}
    (h : ZKmalAt H k P R x
      ((simMStop IV b (T₀ x) NB ΩB (simPre (deployRec IV ivScope b) I) SO' (sR IV ivScope b)).C x)
      ((simMStop IV b (T₀ x) NB ΩB (simPre (deployRec IV ivScope b) I) SO' (sR IV ivScope b)).prog x) ε γ) :
    ZKmalAt H k P R x ((Claims.Recursion.SimStop 𝔇 I SO' H).C x) ((Claims.Recursion.SimStop 𝔇 I SO' H).prog x) ε γ :=
  h.of_sim _ (stopEquiv IV ivScope b VM iv T₀ NB ΩB pre wit PS SO I SO' H x)
    (simStop_eval IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H I SO' x hI hk)

/-- **`εzkStop` at the record is `εzk`**: one commit string per inner round. -/
theorem εzkStop_deployRecMStop (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (t : ℕ) (δ₁ : ℝ)
    (x : Claims.Msg) :
    Claims.Recursion.εzkStop 𝔇 I t δ₁ x = Claims.Recursion.εzk (deployRec IV ivScope b) I t δ₁ x := by
  unfold Claims.Recursion.εzkStop Claims.Recursion.εzk
  simp only [nAt_deployRecMStop, Finset.sum_const, Finset.card_univ, Fintype.card_fin, Finset.card_fin,
    smul_eq_mul, mul_one] <;> rfl

end Stop

/-- **The recursion's `ZKStop` at the masked record with the deployed stop**, where the masked verifier's rounds are
the masked run's (`hk`) and the inner protocol's rounds the inner verifier's (`hI`), at each attempt budget `t ≥ 1`
and `δ₁` at which hm96 leaves hide: the first part is `ZKmalAt` at `Sim`'s first two parts within `2δ₁` per commit
string (`h₀`); each V\* session, at every history that starts it and where the stop hasn't fired, is `ZKmalAt` at
`SO t` within its proved `--zk` error and `4t·3^t` on the sessions' relation `RF` (`hS`); and the live witness
(`LiveWit`). The bound is `εzk + γzk · Pr[Fork]`, `εzkStop` at one commit string per inner round. -/
theorem zkStop_deployRecMStop (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (RF : Claims.Rel Claims.Flock.Wit) (hk : VM.k = kM IV b) (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
    (hS : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
        ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
          u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
          Claims.ZKMal.fired (sR IV ivScope b x) (kPre IV x) u = false →
          ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO t x j).C
            ((SO t x j).prog (xsOf IV b x j u)) (((deployRec IV ivScope b).nums x j).εzk t δ₁) (4 * t * 3 ^ t))
    (hW : LiveWit H IV ivScope b I T₀ pre wit RF) :
    Claims.Recursion.ZKStop H (deployRecMStop IV ivScope b VM iv T₀ NB ΩB pre wit PS SO) I (deployZkRec relO) := by
  refine ⟨rfl, fun t ht δ₁ hδ => ?_⟩
  rw [prover_deployRecMStop IV ivScope b VM iv T₀ NB ΩB pre wit PS SO]
  intro x
  have h := zkAtStop IV b H (T₀ x) (pre x) wit PS (Claims.Recursion.R (deployRec IV ivScope b) I) RF
    (simPre (deployRec IV ivScope b) I) (SO t) (sR IV ivScope b) x (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁)
    (4 * t * 3 ^ t) (fun j => ((deployRec IV ivScope b).nums x j).εzk t δ₁) (h₀ t ht δ₁ hδ x) (hS t ht δ₁ hδ x)
    (fun j hj w hw t₀ u hu hpre hf => hW x w hw j hj t₀ u hu hpre hf) (fun j _ => nums_εzk_nonneg _ t δ₁)
    (by positivity)
  rw [← hk] at h
  refine (zkAt_simStop IV ivScope b VM iv T₀ NB ΩB pre wit PS SO H I (SO t) x (hI x) hk h).mono (le_of_eq ?_)
    (le_of_eq ?_)
  · rw [εzkStop_deployRecMStop IV ivScope b VM iv T₀ NB ΩB pre wit PS SO I t δ₁ x]
    simp only [Claims.Recursion.εzk]
    rw [Fin.sum_univ_eq_sum_range (fun j => ((deployRec IV ivScope b).nums x j).εzk t δ₁)]
    simp only [xI_deployRec]
    simp only [deployRec]
    ring
  · first
      | rfl
      | simp only [Claims.Recursion.γzk, deployRecMStop, deployRecM, deployRec]

/-- **Each V\* session is zero knowledge at the session's tables at the recursive setup**: `zkBoundM_of_tables`'s
sessions, at every history that starts one. -/
theorem sessions_of_tables (IV : Claims.Verifier Coin) (b : Brings) (H : Claims.Msg → Claims.Msg)
    (ivScope : Claims.Msg → Bool) (relO : Readable.Parsed → Readable.Relation) (T₀ : Claims.Msg → SeqTape)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hkey : ∀ δ₁ : ℝ≥0, Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u))) :
    ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
        ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
          u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
          ZKmalAt H ((nums b.st x j).rounds + 1) G.prover (Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u))
            (xsOf IV b x j u) (soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) x j).C
            ((soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) x j).prog (xsOf IV b x j u))
            (((deployRec IV ivScope b).nums x j).εzk t δ₁) (4 * t * 3 ^ t) := by
  intro t ht δ₁ hδ x
  refine sessions_of_zkAt IV b H (T₀ x) (deployZkRec relO) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
    (fun x => 8 * nL x * (δ₁ : ℝ) + 8 * t * nB x * (δ₁ : ℝ) + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ?_
    (fun _ => rfl) (CB t) (fun x j => ((deployRec IV ivScope b).nums x j).εzk t δ₁) (4 * t * 3 ^ t) x
    (fun j hj u hu => ?_)
  · have h := RecDeploy.Flock.zkAt_deployZkRec_of_tables_bind H relO G t ht hK (hkey δ₁ hδ) ENNReal.coe_ne_top Mp Mc
      nL nB hnL hnB hL hOK hC
    simpa only [ENNReal.coe_toReal] using h
  · obtain ⟨h1, h2⟩ := hzl x j hj u hu
    refine ⟨hCB t ht x j hj u hu, ?_, le_rfl⟩
    beta_reduce
    rw [nums_εzk]
    have hδ0 : (0 : ℝ) ≤ δ₁ := δ₁.2
    have e1 : (nL (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 := by exact_mod_cast h1
    have e2 : (nB (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2 := by exact_mod_cast h2
    gcongr

/-- **The recursion's `ZKStop` at the masked record with the deployed stop, at SHA-512, at the inner prover that
commits**, on each statement's first-part tape, from the session's tables at the recursive setup: the first part from
its commit strings (`h0_of_commits`, `preCommits_commit`), the sessions from Flock's zero knowledge at the outer
record (`sessions_of_tables`, `hiding_sha512`), whose premises remain, and the live witness at the outer record's
relation (`hW`). -/
theorem zkStop_deployRecMStop_commit_sha512 (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
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
      I (deployZkRec relO) :=
  zkStop_deployRecMStop FlockDeploy.sha512 IV ivScope (b.commit R dI dg wo) I relO VM iv (preTape IV b R) G.Nonce
    G.Rand (preOf IV b R) wit G.prover _ (Claims.Flock.R (deployZkRec relO).parse) hk hI
    (h0_of_commits IV (b.commit R dI dg wo) FlockDeploy.sha512 ivScope I (preTape IV b R) (preOf IV b R) hI
      fun x => preCommits_commit IV b FlockDeploy.sha512 R dI dg wo x)
    (fun t ht δ₁ hδ x j hj u hu _ => sessions_of_tables IV (b.commit R dI dg wo) FlockDeploy.sha512 ivScope relO
      (preTape IV b R) G Mp Mc nL nB CB (hiding_sha512 IV ivScope (b.commit R dI dg wo)) hK hnL hnB hL hOK hC hzl hCB
      t ht δ₁ hδ x j hj u hu)
    hW

end RecDeploy
