import Proofs.Flock.Claims.Recursion.SeqSim
import Proofs.Flock.Claims.ZK.Deployed

/-!
# The V\* sessions of the masked recursion, from Flock's zero knowledge

`zkBoundM_of` takes each V\* session's `ZKmalAt` at every history that starts it (`hS`). Each session is a complete
masked run of Flock at the staged statement, so `hS` is Flock's own zero knowledge (`Claims.Flock.ZKAt`) at a masked
outer record `DF` whose rounds are the outer verifier's (`VO`), read at the staged statement on the slot's coins:

* `VO_k_xsOf`: the staged statement's rounds are the slot's (`numsOf_vstarAt`).
* `soOf`: the outer simulators of one session simulator `SF`, on coins `CB x j` (`progOf`: `SF` through a bijection).
* `Shaped`: at every history that starts session `j`, `SF`'s coins at the staged statement are `CB x j` up to a
  bijection, and its errors are within `εB x j` and `g`.
* **`sessions_of_zkAt`**: `hS` from `Claims.Flock.ZKAt` at `DF`, where `DF`'s rounds are `VO`'s and the sessions are
  `Shaped`. **`zkBoundM_of_zkAt`**: `ZKBoundM` at the outer simulators `soOf`, from `zkBoundM_of`'s first part (`h₀`),
  the witness (`hW`, on `DF`'s relation) and Flock's zero knowledge at `DF` at each budget and hiding bound.

Flock's zero knowledge at the outer record comes from the masked session's models (`Proto.TMsgLink`,
`Proto.TablesOK`, `Proto.Binding`, the route `Flock.zk_deployZk_of_tables_bind` takes at `FlockDeploy.deployZk`) at any
masked record whose rounds with `Hello` are the protocol's (`Flock.zkAt_of_tables_bind`), so at the outer record
`deployZkRec` at a protocol of `VO.k − 1` coins (`Flock.zkAt_deployZkRec_of_tables_bind`). **`zkBoundM_of_tables`**
assembles the two: `ZKBoundM` at `DF = deployZkRec relO` from the first part, the witness, the models at the recursive
setup, the leaf counts within the record's (`Brings.zl`), and the record's hiding bound giving the pinned key's.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape progOf progOf_of Proto)
open scoped NNReal

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- **The staged statement's rounds are the slot's**: the outer verifier's rounds at session `j`'s V\* statement are
the record's `nums`'. -/
theorem VO_k_xsOf (x : Claims.Msg) (j : ℕ) (u : List (Claims.Msg × Claims.Msg)) (hj : j < nS x) :
    VO.k (xsOf IV b x j u) = (nums b.st x j).rounds := by
  simp only [xsOf, vstar_getD _ _ _ hj]
  exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt _ _ _ _)

/-- **The outer simulators of one session simulator** `SF`: at `x` and session `j`, `SF` on the coins `CB x j`,
through a bijection with its coins at the staged statement (`progOf`). -/
noncomputable def soOf (SF : Claims.Simulator) (CB : Claims.Msg → ℕ → Type) [∀ x j, Fintype (CB x j)]
    [∀ x j, Nonempty (CB x j)] : Claims.Msg → ℕ → Claims.Recursion.OSim :=
  fun x j => ⟨CB x j, fun xS c => progOf SF (CB x j) xS c⟩

/-- **The V\* sessions at `x` are of one shape**: at every history that starts session `j < nS x`, the session
simulator's coins at the staged statement are `CB x j` up to a bijection, and its errors there are within `εB x j` and
`g`. -/
def Shaped (T₀ : SeqTape) (SF : Claims.Simulator) (εF γF : Claims.Msg → ℝ) (CB : Claims.Msg → ℕ → Type)
    (εB : Claims.Msg → ℕ → ℝ) (g : ℝ) (x : Claims.Msg) : Prop :=
  ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
    Nonempty (CB x j ≃ SF.C (xsOf IV b x j u)) ∧ εF (xsOf IV b x j u) ≤ εB x j ∧ γF (xsOf IV b x j u) ≤ g

/-- **Each V\* session is zero knowledge, from Flock's**: at a masked outer record `DF` whose rounds are `VO`'s, Flock's
`ZKAt` at the session prover `PS` and simulator `SF` gives `zkBoundM_of`'s `hS` at the outer simulators `soOf SF CB`,
on `DF`'s relation, where the sessions are `Shaped`. -/
theorem sessions_of_zkAt {Col TF : Type} (H : Claims.Msg → Claims.Msg) (T₀ : SeqTape)
    (DF : Claims.Flock.Deploy Coin Col TF) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SF : Claims.Simulator) (εF γF : Claims.Msg → ℝ)
    (hF : Claims.Flock.ZKAt H DF PS SF εF γF) (hk : ∀ xS, DF.V.k xS = VO.k xS) (CB : Claims.Msg → ℕ → Type)
    [∀ x j, Fintype (CB x j)] [∀ x j, Nonempty (CB x j)] (εB : Claims.Msg → ℕ → ℝ) (g : ℝ) (x : Claims.Msg)
    (hsh : Shaped IV b T₀ SF εF γF CB εB g x) :
    ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (Claims.Flock.R DF.parse (xsOf IV b x j u)) (xsOf IV b x j u)
        (soOf SF CB x j).C ((soOf SF CB x j).prog (xsOf IV b x j u)) (εB x j) g := by
  intro j hj u hu
  obtain ⟨hC, hε, hγ⟩ := hsh j hj u hu
  have e := hF.2.at_equiv (xsOf IV b x j u) hC.some
  simp only [hk, VO_k_xsOf IV b x j u hj] at e
  show ZKmalAt H _ PS _ _ (CB x j) (progOf SF (CB x j) (xsOf IV b x j u)) _ _
  rw [progOf_of hC]
  exact e.mono hε hγ

/-- **The recursion's zero knowledge at the masked record, from Flock's at the outer record**: `ZKBoundM` at the outer
simulators `soOf (SF t) (CB t)`, where the inner protocol's rounds are the inner verifier's (`hI`); the first part is
`ZKmalAt` at `Sim`'s first two parts (`h₀`); at each budget `t ≥ 1` and `δ₁` at which hm96 leaves hide, Flock's
`ZKAt` holds at a masked outer record `DF` of `VO`'s rounds (`hF`, `hk`) and the sessions are `Shaped` within the
record's per-session error and `4t·3^t` (`hsh`); and the witness `wit` gives is in `DF`'s relation (`hW`). -/
theorem zkBoundM_of_zkAt {Col TF : Type} (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB]
    [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (DF : Claims.Flock.Deploy Coin Col TF)
    (SF : ℕ → Claims.Simulator) (εF : ℕ → ℝ≥0 → Claims.Msg → ℝ) (γF : ℕ → Claims.Msg → ℝ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
    (hF : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
        Claims.Flock.ZKAt H DF PS (SF t) (εF t δ₁) (γF t))
    (hk : ∀ xS, DF.V.k xS = VO.k xS)
    (hsh : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        Shaped IV b (T₀ x) (SF t) (εF t δ₁) (γF t) (CB t) (fun x j => ((deployRec IV ivScope b).nums x j).εzk t δ₁)
          (4 * t * 3 ^ t) x)
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R DF.parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ NB ΩB pre wit PS fun t => soOf (SF t) (CB t) :=
  zkBoundM_of IV b H ivScope I T₀ NB ΩB pre wit PS _ (Claims.Flock.R DF.parse) hI h₀
    (fun t ht δ₁ hδ x => sessions_of_zkAt IV b H (T₀ x) DF PS (SF t) (εF t δ₁) (γF t) (hF t ht δ₁ hδ) hk (CB t) _ _
      x (hsh t ht δ₁ hδ x)) hW

end RecDeploy

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

/-- **Flock's zero knowledge at any masked record, at the session's tables**: `Flock.zk_deployZk_of_tables_bind` at a
masked record `D` whose rounds with `Hello` are the protocol's, in place of `FlockDeploy.deployZk`. -/
theorem Flock.zkAt_of_tables_bind {Coin Col T : Type} (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T)
    (G : Proto Flock.Wit) [Fintype G.Coin] (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.TPreModel x) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x)
    (nL nB : Msg → ℕ) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R D.parse) Mp Mc) (hOK : G.TablesOK Mp Mc) (hC : G.Binding H) :
    Claims.Flock.ZKAt H D G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  have hA : ∀ x w, Flock.R D.parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
        2 * nL x * δ₁.toReal := by
    intro x w hR ν V E
    calc _ ≤ 2 * (Mp x ν).size * δ₁.toReal :=
          (Mp x ν).runs_close H hT1 hδ₁ ν w (hL x w hR ν).1 (hOK.1 x ν) V E
      _ ≤ 2 * nL x * δ₁.toReal := by
          have : ((Mp x ν).size : ℝ) ≤ nL x := by exact_mod_cast hnL x ν
          gcongr
  rw [← G.withSim_prover Mc]
  exact Flock.zkR_gkE_hm96_bind H D (G.withSim Mc) t ht hzk hk δ₁ nL nB hA
    (G.withSim_shvzk H _ hT1 hδ₁ Mc nB hnB (fun x w hR ν c => (hL x w hR ν).2 c) hOK.2) hC

end Claims.ZKMal

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape progOf progOf_of Proto)
open FlockSoundness.Discharge
open scoped NNReal

/-- The outer verifier has at least one message, the last. -/
theorem VO_k_pos (x : Claims.Msg) : 0 < VO.k x := by
  simp only [VO]
  split <;> omega

/-- **The round count at the outer record**: a committed-coin protocol of `VO.k − 1` coins has `VO`'s messages and
`Hello` before them. -/
theorem deployZkRec_proto_k (rel : Readable.Parsed → Readable.Relation) (G : Proto Claims.Flock.Wit)
    (hK : ∀ x, G.K x = VO.k x - 1) (x : Claims.Msg) : (deployZkRec rel).V.k x + 1 = G.k x := by
  have := VO_k_pos x
  have := hK x
  show VO.k x + 1 = G.K x + 2
  omega

/-- **Flock's zero knowledge at the outer record**, at the session's tables: `Flock.zkAt_of_tables_bind` at
`deployZkRec`, for a committed-coin protocol of `VO.k − 1` coins (the masked session at the recursive setup). -/
theorem Flock.zkAt_deployZkRec_of_tables_bind (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = VO.k x - 1)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.TPreModel x) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Claims.Flock.R (deployZkRec rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H) :
    Claims.Flock.ZKAt H (deployZkRec rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Claims.ZKMal.Flock.zkAt_of_tables_bind H (deployZkRec rel) G t ht rfl (deployZkRec_proto_k rel G hK) hT1 hδ₁ Mp Mc
    nL nB hnL hnB hL hOK hC

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- **The recursion's zero knowledge at the masked record, from the masked session's models at the recursive setup**:
`ZKBoundM` at the session prover `G.prover` and the outer simulators `soOf` of the models' simulator, on coins `CB t`,
where the inner protocol's rounds are the inner verifier's (`hI`), the first part is `ZKmalAt` at `Sim`'s first two
parts (`h₀`), the record's hiding bound is the pinned key's (`hkey`), the session's models link, hold and bind at the
recursive setup (`hL`, `hOK`, `hC`) at `VO.k − 1` coins (`hK`), their leaf counts are the record's at each staged
statement (`hnL`, `hnB`, `hzl`), the coins are in bijection (`hCB`), and the witness `wit` gives is in the outer
record's relation (`hW`). -/
theorem zkBoundM_of_tables (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (T₀ : Claims.Msg → SeqTape) (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
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
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ G.Nonce G.Rand pre wit G.prover
      fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) := by
  refine zkBoundM_of_zkAt IV b H ivScope I T₀ G.Nonce G.Rand pre wit G.prover (deployZkRec relO)
    (fun t => (G.withSim Mc).simulatorE t (t * 3 ^ t))
    (fun t δ₁ x => 8 * nL x * (δ₁ : ℝ) + 8 * t * nB x * (δ₁ : ℝ) + 2 * (1 / 3) ^ t) (fun t _ => 4 * t * 3 ^ t) CB
    hI h₀ (fun t ht δ₁ hδ => ?_) (fun _ => rfl) (fun t ht δ₁ hδ x j hj u hu => ?_) hW
  · have h := Flock.zkAt_deployZkRec_of_tables_bind H relO G t ht hK (hkey δ₁ hδ) ENNReal.coe_ne_top Mp Mc nL nB hnL
      hnB hL hOK hC
    simpa only [ENNReal.coe_toReal] using h
  · obtain ⟨h1, h2⟩ := hzl x j hj u hu
    refine ⟨hCB t ht x j hj u hu, ?_, le_rfl⟩
    beta_reduce
    rw [nums_εzk]
    have hδ0 : (0 : ℝ) ≤ δ₁ := δ₁.2
    have e1 : (nL (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 := by exact_mod_cast h1
    have e2 : (nB (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2 := by exact_mod_cast h2
    gcongr

end RecDeploy
