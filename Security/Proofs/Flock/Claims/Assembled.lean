import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.KS.PlainAcc
import Proofs.Flock.Claims.KS.ZkTauSlot
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Complete.WalkPacked
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Claims.ZK.Hypotheses
import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Complete.Zk.Coins
import Proofs.Flock.Soundness.CompiledSound

/-!
# Flock's claims at the deployed records, with what remains as hypotheses

`flock_plain_assembled` and `flock_zk_assembled`: at the records Flock's claims read, `FlockDeploy.deployPlainTrim rel`
and `FlockDeploy.deployZkTrim rel` (the deployed instances with their columns trimmed of trailing zeros), at every
relation's fields `rel` and every hash `H`, the conjunction of Flock's claims:

* `ScopeMatches`: the readable scope is the verifier's (`scopeMatches_plain`, `scopeMatches_zk`);
* `KS` at `E^rep` and every decoder meeting `ListWithin`: a hypothesis;
* `Bits` at every such decoder, and at the record's own decoder `min(a, 255.5 − b/2) − 1` bits at every bound
  `κ ≤ 2^{−a}`, `γ ≤ 2^b`, from `KS` alone (`bits_plain`, `bits_zk`, `bits_ge_plain`, `bits_ge_zk`: `Flock.Bits_of_KS`
  and `Flock.Bits_ge` with `ε_table ≥ 0` and `γ > 0`, `εtable_nonneg`, `γ_pos`);
* `ZK`, at the masked record of the protocol model `G` (`ZKMal.zkOf`: the record's reference prover
  `ZkTables.honestPR` behind `G`'s `Hello` and opening checks, and `G`'s simulator), from the premises of
  `Flock.zk_deployZk_of_models`;
* `Efficient` at the record's own counted decoder (`Deploy.dec`): `E^rep`'s reruns at every deployment (`ERep_cost`),
  the decoder's `ListWithin`, and its count and lists at most `plainStepPoly` (`zkStepPoly`) at the opened word's
  encoded length, which bounds `assemble`'s checks (`efficient_plain`, `efficient_zk`);
* `Complete` at the record's reference prover and its error, on the prover's tape: of `FlockComplete.honestP` on
  `Tape` at `FlockComplete.δc H` at the plain record, of `FlockComplete.ZkTables.honestPR` on `TapeZk` (each table run
  on its share of the witness) at `FlockComplete.ZkTables.δc H` at the masked one: a hypothesis, which at the deployed
  relation follows from the record's JSON at the plain record (`flock_plain_assembled_relation`), and at the masked
  record from it and three premises on the honest masked run (`flock_zk_assembled_relation`).

At a prover and error left free, `Complete` holds whatever the verifier does: at an empty tape, or at `δc = 1`. Here
both are the honest prover's, and both tapes are inhabited (`instInhabitedTape`, `instInhabitedTapeZk`). The statements
read `Claims.Complete` at the record's verifier, relation (`Flock.R`) and scope (`Flock.Scope`), which is what
`Flock.Complete` unfolds to, and name the prover themselves, so they don't depend on how `Flock.Complete` gets its
prover: as an argument, or from the record. `Claims.Complete` stays folded: with or without its hash's 64-byte outputs
(`Out64 H`) as a premise, and with a `[Nonempty Ω]` binder met by the tapes' instances.

## What remains: the hypotheses, exactly

`flock_plain_assembled`:
* `hKS : Flock.KS H (deployPlainTrim rel)`. At SHA-512 it follows from the fixed-table bound over full-width tables at
  the untrimmed instance (`FlockDeploy.KS.plain_KS_trim`). At the untrimmed record main reduces KS to one wait's bound,
  `Flock.OneWait`, with `k ≥ 8` and the relation splitting by table wherever the verifier accepts
  (`FlockDeploy.KS.plain_of_oneWait`). At SHA-512 and the deployed relation it follows from the live sessions' bound
  with escapes on accepted runs at the statements some run accepts (`FlockDeploy.KS.PlainLiveAcc`,
  `FlockDeploy.KS.plain_KS_of_liveAcc`), so `flock_plain_assembled_trim` takes that bound in its place.
* `hComplete`: `Flock.Complete` at the plain record, unfolded, of `honestP` at `δc H`.
  `FlockComplete.complete_of_goals` concludes it from the honest run's goals, `FlockComplete.Goals H rel`. The trim
  reads only the columns and the code through `trimCol`, and keeps the verifier, the parser, the reference prover and
  its error, so completeness at the trimmed record is the untrimmed record's (`Flock.complete_mapCols`). At the
  deployed relation (`Readable.relation`) it holds with no premise (`FlockComplete.Replay.complete_plain_json`), so
  `flock_plain_assembled_relation` takes no `hComplete`.

`flock_zk_assembled`:
* `hKS : Flock.KS H (deployZkTrim rel)`. At SHA-512 it follows from the fixed-table bound at the untrimmed instance at
  statements of one table (`FlockDeploy.KS.zk_KS_trim`); at the untrimmed record, from one wait's bound
  (`FlockDeploy.KS.zk_of_oneWait`, as at the plain record). At SHA-512 and the deployed relation it holds with no
  premise (`FlockDeploy.KS.zk_KS`), so `flock_zk_assembled_trim` drops it; so does KS at the untrimmed instance, which
  the sampling claims read (`FlockDeploy.KS.zk_KS_untrim`).
* `hComplete`: `Flock.Complete` at the masked record, unfolded, of `ZkTables.honestPR` at `ZkTables.δc H`, at any
  number of tables, the untrimmed record's (`Flock.complete_mapCols`). At the deployed relation it follows from
  three premises on the honest masked run, each at every good case or run and every table of it
  (`FlockComplete.ZkTables.complete_json`), so `flock_zk_assembled_relation` takes those in its place: the masked
  prover's pieces return (`ProverRunsAt`), its rep proofs fit their shapes (`RepShapedAt`) and, per rep, the masked
  Ligerito and the inner proof accept (`LgInnerAt`).
  Those three follow from four premises on one rep of one table (`FlockComplete.ZkTables.complete_jsonT`:
  `RepShapedT`, `ProverRunsT`, `LigeritoT`, `InnerT`). At one table the honest masked prover is `honestPZk` at
  `δcZk H` (`FlockComplete.complete_zk_of_goals`, from `FlockComplete.GoalsZk H rel`).
* ZK's, as `Flock.zk_deployZk_of_models` takes them:
  * the named assumption `hT1 : Hm96Hiding My cs δ₁`, with `hδ₁ : δ₁ ≠ ⊤`;
  * the model: `G` and its rounds, `hK : G.K x = (V true).k x − 1`; the pre-coin and coin models `Mp` and `Mc` over
    the salt, blinding and commitment spaces `Salt`, `B` and `Cc`, with `My` and `cs`; their leaf bounds
    `hnL : card (Mp x ν).L ≤ nL x` and `hnB : (Mc x ν c).size ≤ nB x`; the repetitions `t`, `ht : 0 < t`;
  * the five premises `hL : G.MsgLink`, `hS : G.SimLink`, `hN : G.NonDegenerate`, `hI : G.InnerHolds` and
    `hC : G.CoinCheck`;
  * `hG : G.Honest`: `G`'s messages are `ZkTables.honestPR`'s, at the coin each opening reads as (`coinOf`) and the
    reference prover's tape (`tapeOf`).

`δc H` and `ZkTables.δc H` are `1`, so `Complete` asks nothing, where `ready H` and `ZkTables.ready H` fail. ZK and
`Complete` read one reference prover, `ZkTables.honestPR`.
-/

namespace FlockDeploy.Assembled

open Flock

/-- A table's error is nonnegative. -/
theorem tableErr_nonneg (zk : Bool) (spec : SessionSpec) (st : Stmt) : 0 ≤ tableErr zk spec st := by
  unfold tableErr
  split
  · unfold FlockSoundness.Accounting.padLinkError
    positivity
  · exact FlockSoundness.tableError_nonneg _ _

/-- **`ε_table ≥ 0`** at both records: the tables' errors, summed. -/
theorem εtable_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ εtable zk x := by
  unfold εtable
  refine List.sum_nonneg fun a ha => ?_
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
  exact tableErr_nonneg zk p.1 p.2

/-- The deployed verifier takes at least one round at every statement. -/
theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x := by
  simp only [V]
  split <;> omega

/-- **`γ > 0`** wherever the verifier takes a round. -/
theorem γ_pos {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (x : Claims.Msg) (hk : 0 < D.V.k x) :
    0 < Claims.Flock.γ D x := by
  unfold Claims.Flock.γ
  have hk' : (0 : ℝ) < D.V.k x := by exact_mod_cast hk
  have hs := Real.sqrt_nonneg (2 * D.V.k x * D.weight x)
  have h : 0 < Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x := by linarith
  exact mul_pos (by norm_num) (pow_pos h 2)

/-- **Bits at the deployed plain record, from KS alone** (`Flock.Bits_of_KS`): the record's table error is
nonnegative (`εtable_nonneg`). -/
theorem bits_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel)) : Claims.Flock.Bits H (deployPlainTrim rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg false)

/-- **Bits at the deployed masked record, from KS alone** (`Flock.Bits_of_KS`), as at the plain one. -/
theorem bits_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel)) : Claims.Flock.Bits H (deployZkTrim rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg true)

/-- **Bits at bounds at the deployed plain record, from KS alone** (`Flock.Bits_ge`): at the record's own decoder,
`min(a, 255.5 − b/2) − 1` bits at every bound `κ ≤ 2^{−a}`, `γ ≤ 2^b`; `γ > 0` because the verifier takes a round. -/
theorem bits_ge_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployPlainTrim rel) x ≤ 2 ^ (-a x))
    (hγ : ∀ x, Claims.Flock.γ (deployPlainTrim rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployPlainTrim rel).V (Claims.Flock.R (deployPlainTrim rel).parse)
      (Claims.Flock.E (deployPlainTrim rel) (deployPlainTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ (fun x => γ_pos (deployPlainTrim rel) x (V_k_pos false x)) hγ _
    (decode_plain rel ▸ listWithin_plainTrim rel)

/-- **Bits at bounds at the deployed masked record, from KS alone**, as at the plain one. -/
theorem bits_ge_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployZkTrim rel) x ≤ 2 ^ (-a x))
    (hγ : ∀ x, Claims.Flock.γ (deployZkTrim rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployZkTrim rel).V (Claims.Flock.R (deployZkTrim rel).parse)
      (Claims.Flock.E (deployZkTrim rel) (deployZkTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ (fun x => γ_pos (deployZkTrim rel) x (V_k_pos true x)) hγ _
    (decode_zk rel ▸ listWithin_zkTrim rel)

/-- **Flock's claims at the deployed plain verifier**, at every relation's fields and every hash: the readable scope
is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the record's decoder at every
bound on `κ` and `γ`; efficiency at the record's decoder with its count; and completeness at the honest prover
`honestP` with error `δc H`. What remains is exactly `hKS` and `hComplete`. -/
theorem flock_plain_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel))
    (hComplete : Claims.Complete H (deployPlainTrim rel).V FlockComplete.honestP
      (Claims.Flock.R (deployPlainTrim rel).parse) (Claims.Flock.Scope (deployPlainTrim rel)) (FlockComplete.δc H)) :
    Claims.Flock.ScopeMatches (deployPlainTrim rel) ∧
      Claims.Flock.KS H (deployPlainTrim rel) ∧
      Claims.Flock.Bits H (deployPlainTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlainTrim rel).V (Claims.Flock.R (deployPlainTrim rel).parse)
          (Claims.Flock.E (deployPlainTrim rel) (deployPlainTrim rel).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlainTrim rel) plainStepPoly ∧
      Claims.Complete H (deployPlainTrim rel).V FlockComplete.honestP (Claims.Flock.R (deployPlainTrim rel).parse)
        (Claims.Flock.Scope (deployPlainTrim rel)) (FlockComplete.δc H) :=
  ⟨scopeMatches_plain rel, hKS, bits_plain rel H hKS, bits_ge_plain rel H hKS, efficient_plain H rel, hComplete⟩

/-- **Completeness at a record read through `f` is the record's**: `Deploy.mapCols` keeps the verifier, the parser,
the reference prover and its error. -/
theorem _root_.Claims.Flock.complete_mapCols {Coin Col T : Type} [Fintype Coin] [Inhabited Coin]
    (H : Claims.Msg → Claims.Msg) (D : Claims.Flock.Deploy Coin Col T) (f : Col → Col) :
    Claims.Flock.Complete H (D.mapCols f) ↔ Claims.Flock.Complete H D :=
  Iff.rfl

/-- **Flock's claims at the deployed plain verifier at the deployed relation** (`Readable.relation`), at every hash:
`flock_plain_assembled` with its completeness proved (`complete_plain_json`, at the trimmed record by
`Flock.complete_mapCols`). `hKS` (E4) is what remains. -/
theorem flock_plain_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim Readable.relation)) :
    Claims.Flock.ScopeMatches (deployPlainTrim Readable.relation) ∧
      Claims.Flock.KS H (deployPlainTrim Readable.relation) ∧
      Claims.Flock.Bits H (deployPlainTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlainTrim Readable.relation).V (Claims.Flock.R (deployPlainTrim Readable.relation).parse)
          (Claims.Flock.E (deployPlainTrim Readable.relation) (deployPlainTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlainTrim Readable.relation) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlainTrim Readable.relation) :=
  flock_plain_assembled Readable.relation H hKS
    ((Claims.Flock.complete_mapCols H (deployPlain Readable.relation) trimCol).2
      (FlockComplete.Replay.complete_plain_json H))

/-- **Flock's claims at the deployed plain verifier at the deployed relation, at SHA-512**:
`flock_plain_assembled_relation` with KS from the live sessions' bound with escapes on accepted runs at the statements
some run accepts (`FlockDeploy.KS.plain_KS_of_liveAcc`). That bound is what remains. -/
theorem flock_plain_assembled_trim
    (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → FlockDeploy.KS.PlainLiveAcc x) :
    Claims.Flock.ScopeMatches (deployPlainTrim Readable.relation) ∧
      Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) ∧
      Claims.Flock.Bits sha512 (deployPlainTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits sha512 (deployPlainTrim Readable.relation).V
          (Claims.Flock.R (deployPlainTrim Readable.relation).parse)
          (Claims.Flock.E (deployPlainTrim Readable.relation) (deployPlainTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient sha512 (deployPlainTrim Readable.relation) plainStepPoly ∧
      Claims.Flock.Complete sha512 (deployPlainTrim Readable.relation) :=
  flock_plain_assembled_relation sha512 (FlockDeploy.KS.plain_KS_of_liveAcc hl)

/-- **Flock's claims at the deployed masked (`--zk`) verifier**, at every relation's fields and every hash: the
readable scope is the verifier's; KS at `E^rep`; bits at every decoder meeting `ListWithin`, and at the record's
decoder at every bound on `κ` and `γ`; zero knowledge at `G`'s masked record (`ZKMal.zkOf`); efficiency at the
record's decoder with its count; and completeness at the record's reference prover `ZkTables.honestPR` with error
`ZkTables.δc H`. What remains is exactly `hKS`, `hComplete` and `Flock.zk_deployZk_of_models`' hypotheses, `hT1` its
named assumption. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel))
    (hComplete : Claims.Complete H (deployZkTrim rel).V FlockComplete.ZkTables.honestPR
      (Claims.Flock.R (deployZkTrim rel).parse) (Claims.Flock.Scope (deployZkTrim rel)) (FlockComplete.ZkTables.δc H))
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Claims.Flock.R (deployZkTrim rel).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim rel).parse) Mc) (hC : G.CoinCheck)
    (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim rel) ∧
      Claims.Flock.KS H (deployZkTrim rel) ∧
      Claims.Flock.Bits H (deployZkTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZkTrim rel).V (Claims.Flock.R (deployZkTrim rel).parse)
          (Claims.Flock.E (deployZkTrim rel) (deployZkTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK H (Claims.ZKMal.zkOf rel G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient H (deployZkTrim rel) zkStepPoly ∧
      Claims.Complete H (deployZkTrim rel).V FlockComplete.ZkTables.honestPR (Claims.Flock.R (deployZkTrim rel).parse)
        (Claims.Flock.Scope (deployZkTrim rel)) (FlockComplete.ZkTables.δc H) :=
  ⟨scopeMatches_zk rel, hKS, bits_zk rel H hKS, bits_ge_zk rel H hKS,
    Claims.ZKMal.Flock.zk_deployZk_of_models H rel G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC coinOf
      tapeOf hG,
    efficient_zk H rel, hComplete⟩

/-- **Flock's claims at the deployed masked verifier at the deployed relation** (`Readable.relation`), at every hash:
`flock_zk_assembled` with its completeness from the honest masked run's premises
(`FlockComplete.ZkTables.complete_json`, at the trimmed record by `Flock.complete_mapCols`) and the mask slots' rank
bound (`FlockComplete.MaskRank.MaskRankBound`). `hKS` (E4), those premises and `Flock.zk_deployZk_of_models`'
hypotheses are what remains. -/
theorem flock_zk_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim Readable.relation))
    (hPR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.ProverRunsAt c)
    (hR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.RepShapedAt c)
    (hLI : ∀ o : FlockComplete.ZkTables.Run H Readable.relation,
      FlockComplete.ZkTables.LgInnerAt H Readable.relation o)
    (hMR : FlockComplete.MaskRank.MaskRankBound)
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Claims.Flock.R (deployZkTrim Readable.relation).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim Readable.relation).parse) Mc)
    (hC : G.CoinCheck) (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim Readable.relation) ∧
      Claims.Flock.KS H (deployZkTrim Readable.relation) ∧
      Claims.Flock.Bits H (deployZkTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZkTrim Readable.relation).V (Claims.Flock.R (deployZkTrim Readable.relation).parse)
          (Claims.Flock.E (deployZkTrim Readable.relation) (deployZkTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK H (Claims.ZKMal.zkOf Readable.relation G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient H (deployZkTrim Readable.relation) zkStepPoly ∧
      Claims.Flock.Complete H (deployZkTrim Readable.relation) :=
  flock_zk_assembled Readable.relation H hKS
    ((Claims.Flock.complete_mapCols H (deployZk Readable.relation) trimCol).2
      (FlockComplete.ZkTables.complete_json H hPR hR hLI hMR))
    G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC coinOf tapeOf hG

/-- **Flock's claims at the deployed masked verifier at the deployed relation, at SHA-512**:
`flock_zk_assembled_relation` with KS, which holds with no premise (`FlockDeploy.KS.zk_KS`). The honest masked run's
premises, the mask slots' rank bound and `Flock.zk_deployZk_of_models`'
hypotheses, `hT1` its named assumption, are what remain. -/
theorem flock_zk_assembled_trim
    (hPR : ∀ c : FlockComplete.ZkTables.Case sha512 Readable.relation, FlockComplete.ZkTables.ProverRunsAt c)
    (hR : ∀ c : FlockComplete.ZkTables.Case sha512 Readable.relation, FlockComplete.ZkTables.RepShapedAt c)
    (hLI : ∀ o : FlockComplete.ZkTables.Run sha512 Readable.relation,
      FlockComplete.ZkTables.LgInnerAt sha512 Readable.relation o)
    (hMR : FlockComplete.MaskRank.MaskRankBound)
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink sha512 (Claims.Flock.R (deployZkTrim Readable.relation).parse) My cs Mp Mc)
    (hS : G.SimLink sha512 Mc) (hN : G.NonDegenerate Mc)
    (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim Readable.relation).parse) Mc)
    (hC : G.CoinCheck) (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest sha512 FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim Readable.relation) ∧
      Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      Claims.Flock.Bits sha512 (deployZkTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits sha512 (deployZkTrim Readable.relation).V (Claims.Flock.R (deployZkTrim Readable.relation).parse)
          (Claims.Flock.E (deployZkTrim Readable.relation) (deployZkTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK sha512 (Claims.ZKMal.zkOf Readable.relation G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient sha512 (deployZkTrim Readable.relation) zkStepPoly ∧
      Claims.Flock.Complete sha512 (deployZkTrim Readable.relation) :=
  flock_zk_assembled_relation sha512 FlockDeploy.KS.zk_KS hPR hR hLI hMR G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB
    hL hS hN hI hC coinOf tapeOf hG

end FlockDeploy.Assembled
