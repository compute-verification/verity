import Proofs.Flock.Claims.Recursion.SeqKey
import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# The recursion's `ZK` at the masked record

The claim reads statement-indexed tapes (`ZKmalRx`), so `ZKBoundM` is `Claims.Recursion.ZK` at the masked record
`deployRecM`: `deployRec` with a masked verifier `VM` and its inner view, the masked reference prover (`refPM`) on one
tape (the inner prover's tape and the staging salts, then the sessions' slots), at each statement `x` read off the
nested tape of `x`'s `n_S(x)` V\* sessions (`RunParts.flatten`) on `x`'s own first-part tape `T₀ x`, through `pre x`, and
the outer simulators. `R`, `Sim`, `εzk`, `γzk` and the pinned key read neither the verifier nor the prover, so they
are `deployRec`'s.

* `prover_deployRecM`: the masked record's prover at `x` is `ZKBoundM`'s, `refPM` at `pre x` on `x`'s tape, since the
  layout's prover reads its first part's tape only through what `pre` gives (`layProver_comap`).
* `zk_deployRecM`: `ZK` at `deployRecM` from `ZKBoundM`, where `VM`'s rounds are the masked run's (`kM`: the compiled
  verifier's and one `Hello` per V\* session).
* `zk_deployRecM_sha512`: at `H = SHA-512`, from `zkBoundM_of_commits_sha512`'s premises.
  `zk_deployRecM_commit_sha512`: at the inner prover that commits (`Brings.commit`), on each statement's first-part
  tape (`preTape`), from `zkBoundM_of_commit_sha512`'s.
-/

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts SeqTape Proto)
open scoped NNReal

/-- **The layout's prover reads its first part's tape only through `p`**: on a tape `p` reads, it is the layout's at
`id` on the tape of what `p` gives. -/
theorem layProver_comap (IV : Claims.Verifier Coin) (b : Brings) {T TS : Type} (p : T → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover TS) (w : Claims.Flock.Wit) :
    ∀ (m : ℕ) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg)) (ω : T × (ℕ → TS)),
      ((lay IV b id wit).prover (preP IV b id) PS m w).next x hist (p ω.1, ω.2) =
        ((lay IV b p wit).prover (preP IV b p) PS m w).next x hist ω
  | 0, _, _, _ => rfl
  | m + 1, x, hist, ω => by
      have hs : (lay IV b id wit).start x m = (lay IV b p wit).start x m := by
        rw [lay_start, lay_start, parts_start, parts_start]
      simp only [RunParts.prover, hs, layProver_comap IV b p wit PS w m x hist ω]
      rfl

/-- **The masked record**: `deployRec` with the masked verifier `VM` and its inner view `iv`, the masked reference
prover `refPM` on one tape, at each statement `x` the nested tape of `x`'s `nS x` sessions read flat, its first part
read through `pre x`, and the outer simulators `SO`. -/
noncomputable def deployRecM (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZkDeploy Coin :=
  { deployRec IV ivScope b with
    V := VM
    innerView := iv
    Tape := (b.ref.TI × (ℕ → Salt)) × (ℕ → NB × ΩB)
    tapeNonempty := inferInstance
    P := refPM IV b id wit PS
    Nonce := fun x => ((T₀ x).iter 0 NB ΩB (nS x)).N
    Rest := fun x => ((T₀ x).iter 0 NB ΩB (nS x)).Ω
    tapeOf := fun x ω =>
      (pre x (RunParts.flatten (T₀ x) 0 NB ΩB (nS x) ω).1, (RunParts.flatten (T₀ x) 0 NB ΩB (nS x) ω).2)
    SO := SO }

section Sim

variable (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
  (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (VM : Claims.Verifier Coin)
  (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
  (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
  (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
  (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
  (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)

/-- The masked record's commit strings to the zero digest are `deployRec`'s. -/
theorem zeroCommits_deployRecM
    (ys : List (Fin (deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO).saltLen → UInt8)) :
    (deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO).zeroCommits ys =
      (deployRec IV ivScope b).zeroCommits ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih => simp only [Claims.Recursion.Deploy.zeroCommits, ih]; rfl

/-- **The masked record's `Sim` is `deployRec`'s**: it reads neither the verifier nor the prover. -/
theorem sim_deployRecM (SO' : Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.Sim (deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO).toDeploy I SO' =
      Claims.Recursion.Sim (deployRec IV ivScope b) I SO' := by
  unfold Claims.Recursion.Sim Claims.Recursion.simStage
  simp only [zeroCommits_deployRecM]
  rfl

/-- **The masked record's prover at `x` is `ZKBoundM`'s**: `refPM` at `pre x` on the nested tape of `x`'s sessions
read flat. -/
theorem prover_deployRecM : (deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO).prover =
    fun w x => (refPM IV b (pre x) wit PS w).comap (RunParts.flatten (T₀ x) 0 NB ΩB (nS x)) := by
  funext w x
  exact congrArg Claims.Prover.mk (funext fun x' => funext fun hist => funext fun ω =>
    layProver_comap IV b (pre x) wit PS w (nS x') x' hist (RunParts.flatten (T₀ x) 0 NB ΩB (nS x) ω))

end Sim

/-- **The recursion's `ZK` at the masked record, from `ZKBoundM`**, where the masked verifier's rounds are the masked
run's, at the masked outer record. -/
theorem zk_deployRecM (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (hk : VM.k = kM IV b) (h : ZKBoundM IV b H ivScope I T₀ NB ΩB pre wit PS SO) :
    Claims.Recursion.ZK H (deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO) I (deployZkRec relO) := by
  refine ⟨rfl, fun t ht δ₁ hδ => ?_⟩
  have h' := h t ht δ₁ hδ
  rw [← hk, ← sim_deployRecM IV ivScope b I VM iv T₀ NB ΩB pre wit PS SO] at h'
  rw [prover_deployRecM IV ivScope b VM iv T₀ NB ΩB pre wit PS SO]
  exact h'

/-- **The recursion's `ZK` at the masked record at SHA-512**: `zk_deployRecM` at `zkBoundM_of_commits_sha512`, whose
premises remain, and the masked verifier's rounds. -/
theorem zk_deployRecM_sha512 (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (VM : Claims.Verifier Coin)
    (iv : (Claims.Msg → Claims.Msg) → (x : Claims.Msg) → (Fin (VM.k x) → Coin) → List Claims.Flock.Wit → Claims.Seen)
    (T₀ : Claims.Msg → SeqTape) (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hk : VM.k = kM IV b)
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x)) (hpre : ∀ x, PreCommits IV b FlockDeploy.sha512 (T₀ x) (pre x) x)
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    Claims.Recursion.ZK FlockDeploy.sha512
      (deployRecM IV ivScope b VM iv T₀ G.Nonce G.Rand pre wit G.prover
        fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t))
      I (deployZkRec relO) :=
  zk_deployRecM FlockDeploy.sha512 IV ivScope b I relO VM iv T₀ G.Nonce G.Rand pre wit G.prover _ hk
    (zkBoundM_of_commits_sha512 IV b ivScope I relO T₀ pre wit G Mp Mc nL nB CB hI hpre hK hnL hnB hL hOK hC hzl hCB
      hW)

/-- **The recursion's `ZK` at the masked record at SHA-512, at the inner prover that commits**: `zk_deployRecM` at
`zkBoundM_of_commit_sha512`, on each statement's first-part tape, whose premises remain, and the masked verifier's
rounds. -/
theorem zk_deployRecM_commit_sha512 (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
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
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope (b.commit R dI dg wo)) I x w → ∀ j < nS x,
      ∀ (t₀ : (preTape IV b R x).N × (preTape IV b R x).Ω) (u : List (Claims.Msg × Claims.Msg)),
        u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
          Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u)
            (wit w (preOf IV b R x t₀) (u.take (IV.k (xIOf x))) j)) :
    Claims.Recursion.ZK FlockDeploy.sha512
      (deployRecM IV ivScope (b.commit R dI dg wo) VM iv (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit G.prover
        fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t))
      I (deployZkRec relO) :=
  zk_deployRecM FlockDeploy.sha512 IV ivScope (b.commit R dI dg wo) I relO VM iv (preTape IV b R) G.Nonce G.Rand
    (preOf IV b R) wit G.prover _ hk
    (zkBoundM_of_commit_sha512 IV b ivScope I relO R dI dg wo wit G Mp Mc nL nB CB hI hK hnL hnB hL hOK hC hzl hCB hW)

end RecDeploy
