import Proofs.Flock.Claims.Recursion.FlockKSDeploy
import Proofs.Flock.Claims.KS.TableBound
import Proofs.Flock.Claims.Bridge.Hm96Tree

/-!
# Flock meets `Inner.KSAbort` at the deployed records

`KSAbort_of_tableBoundOnAcc` at `FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel`, at SHA-512, with all but
the table bound (and, masked, the statement's shape) discharged:

* the four code facts at the deployed verifier: it shows each round's coin, read back by `decCoin` (`coinsShown_V`, from
  `answersCoin_deploy` and `coin_bytes_injective`), and `accLast_V_sha512`, `decideFree_V`, `refusable_V`;
* `ColsOk` and `ConflictTied` (`plain_colsOk`, `plain_conflictTied`, and their masked forms);
* the numbers: `wordLen_le_N₀`, `εtable_nonneg`, `V_k_pos`, `cap_pos`, and `k ≥ 8` with the relation splitting where
  the statement has one table (`k_range_plain`, `splits_of_one`);
* plain, one table wherever the verifier accepts some run (`plain_hone`).

The table bound with escapes on accepted runs is taken at full-width tables, `Extendθ` carrying it to tables with
unopened positions with the same escapes (`tableBoundθOnAcc_of_extend`). So the recursion's KS over an inner deployed
record (`plain_KSFlock`, `zk_KSFlock`) rests on that bound, `Extendθ` (masked, the one-table shape), Flock's KS at the
outer deployment, and the recursion's own hypotheses, with `PublicCoin` met at `decCoin`.
-/

namespace Claims.Recursion

open Claims Claims.Flock

/-- **`TableBoundθOnAcc` at tables with unopened positions**, from the bound at tables of `Ok` columns and `Extendθ`,
with the same escapes, when accepted runs open only `Ok` columns and `Ok` admits no `0` on a nonempty word
(`tableBoundθ_of_extend`). -/
theorem tableBoundθOnAcc_of_extend {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop)
    (hne : ∀ x u, 0 < N x → Ok x u → u ≠ 0) (hok : ColsOk H D N Ok) (hext : Extendθ D N Ok)
    (htab : TableBoundθOnAcc H D N Ok) : TableBoundθOnAcc H D N (orZero Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hon, hfit, fun ω C hC hg => ?_⟩
  obtain ⟨C', hC', hkeep, hg'⟩ := hext x C hC hg
  refine (hb ω C' hC' hg').trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne'⟩ => ⟨ha, p, u, hu, ?_⟩))
  by_cases h0 : C p = 0
  · rw [h0]
    exact hne x u (Nat.lt_of_le_of_lt (Nat.zero_le _) p.isLt) (hok x Ω P ω r p.val u ha p.isLt hu)
  · rwa [← hkeep p h0]

end Claims.Recursion

namespace FlockDeploy

open Claims.Recursion

/-- **The coin a message shows**: the coin whose bytes it is. -/
noncomputable def decCoin (_ : Claims.Msg) : Claims.Msg → Coin := Function.invFun Coin.bytes

/-- **The deployed verifier shows each round's coin**: it answers with the coin's bytes (`answersCoin_deploy`), which
determine the coin (`coin_bytes_injective`). -/
theorem coinsShown_V (H : Claims.Msg → Claims.Msg) (zk : Bool) : CoinsShown H (V zk) decCoin :=
  fun x as cs c m _ h => by
    rw [Claims.Bridge.answersCoin_deploy zk H x as cs c m h]
    exact Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c

end FlockDeploy

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **The plain record's `k` range, with the relation split**, where the statement has one table wherever the verifier
accepts some run. -/
theorem plain_kSplits (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x) (x : Claims.Msg) :
    (8 ≤ (deployPlain rel).k x ∧ Claims.Flock.Splits (deployPlain rel) x) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x :=
  (k_range_plain rel x).elim
    (fun hk => ((hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployPlain rel) rfl rfl hs h1).imp
      (fun h => ⟨hk, h⟩) id) .inr

/-- **The masked record's `k` range, with the relation split**, where the statement has one table wherever the
verifier accepts some run. -/
theorem zk_kSplits (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x) (x : Claims.Msg) :
    (8 ≤ (deployZk rel).k x ∧ Claims.Flock.Splits (deployZk rel) x) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x :=
  (k_range_zk rel x).elim
    (fun hk => ((hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployZk rel) rfl rfl hs h1).imp
      (fun h => ⟨hk, h⟩) id) .inr

/-- **The deployed plain record meets `Inner.KSAbort`**, at SHA-512, at every decoder within the radius: from the table
bound at full-width tables with escapes on accepted runs and `Extendθ` (one table wherever the verifier accepts some
run is `plain_hone`). -/
theorem plain_KSAbort (rel : Readable.Parsed → Readable.Relation)
    (hext : Claims.Flock.Extendθ (deployPlain rel) (wordLen false) (wide false))
    (htab : TableBoundθOnAcc sha512 (deployPlain rel) (wordLen false) (wide false))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlain rel) decode) : (Inner.ofFlock (deployPlain rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAcc sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false)) decCoin
    (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (V_k_pos false) (fun _ => cap_pos)
    (plain_kSplits rel (plain_hone rel)) (plain_hone rel) (fun _ => .inl rfl)
    (Claims.Flock.colsOk_orZero _ _ _ _ (plain_colsOk sha512 rel)) (plain_conflictTied rel)
    (tableBoundθOnAcc_of_extend _ _ _ _ (wide_ne_zero false) (plain_colsOk sha512 rel) hext htab)
    (coinsShown_V sha512 false) (accLast_V_sha512 false) (decideFree_V sha512 false) (refusable_V sha512 false)
    decode hdec

/-- **The deployed masked (`--zk`) record meets `Inner.KSAbort`**, at SHA-512, at every decoder within the radius:
from the table bound at full-width tables with escapes on accepted runs, `Extendθ`, and one table wherever the
verifier accepts some run. -/
theorem zk_KSAbort (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (hext : Claims.Flock.Extendθ (deployZk rel) (wordLen true) (wide true))
    (htab : TableBoundθOnAcc sha512 (deployZk rel) (wordLen true) (wide true))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZk rel) decode) : (Inner.ofFlock (deployZk rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAcc sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true)) decCoin
    (wordLen_le_N₀ true) (fun x => εtable_nonneg true x) (V_k_pos true) (fun _ => cap_pos)
    (zk_kSplits rel hone) hone (fun _ => .inl rfl)
    (Claims.Flock.colsOk_orZero _ _ _ _ (zk_colsOk sha512 rel)) (zk_conflictTied rel)
    (tableBoundθOnAcc_of_extend _ _ _ _ (wide_ne_zero true) (zk_colsOk sha512 rel) hext htab)
    (coinsShown_V sha512 true) (accLast_V_sha512 true) (decideFree_V sha512 true) (refusable_V sha512 true)
    decode hdec

/-- **KS of the recursion over the deployed plain record as the inner protocol**, at SHA-512: `KS_of_flockKS` at each
inner decoder within the radius, with `Inner.KSAbort` from `plain_KSAbort` and `PublicCoin` at `decCoin`
(`coinsShown_V`). -/
theorem plain_KSFlock (rel : Readable.Parsed → Readable.Relation) (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (opening : Claims.Msg → Claims.Msg) (hF : Claims.Flock.KS sha512 DO)
    (hext : Claims.Flock.Extendθ (deployPlain rel) (wordLen false) (wide false))
    (htab : TableBoundθOnAcc sha512 (deployPlain rel) (wordLen false) (wide false)) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployPlain rel).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployPlain rel) decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployPlain rel) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployPlain rel) decI) DO)
    (hC : ∀ decI, CallsRecord sha512 D (Inner.ofFlock (deployPlain rel) decI) DO)
    (hO : ∀ decI, CallsOpen sha512 D (Inner.ofFlock (deployPlain rel) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployPlain rel) decI) DO) (hN0 : CallsNil sha512 D) :
    KSFlockA sha512 D (deployPlain rel) DO :=
  fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF (plain_KSAbort rel hext htab decI h) hRef
    hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hC decI) (hO decI) (hAcc decI) (coinsShown_V sha512 false) hN0

/-- **KS of the recursion over the deployed masked (`--zk`) record as the inner protocol**, at SHA-512:
`KS_of_flockKS` at each inner decoder within the radius, with `Inner.KSAbort` from `zk_KSAbort` and `PublicCoin` at
`decCoin` (`coinsShown_V`). -/
theorem zk_KSFlock (rel : Readable.Parsed → Readable.Relation) (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Msg → Claims.Msg) (hF : Claims.Flock.KS sha512 DO)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (hext : Claims.Flock.Extendθ (deployZk rel) (wordLen true) (wide true))
    (htab : TableBoundθOnAcc sha512 (deployZk rel) (wordLen true) (wide true)) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZk rel).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZk rel) decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZk rel) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZk rel) decI) DO)
    (hC : ∀ decI, CallsRecord sha512 D (Inner.ofFlock (deployZk rel) decI) DO)
    (hO : ∀ decI, CallsOpen sha512 D (Inner.ofFlock (deployZk rel) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZk rel) decI) DO) (hN0 : CallsNil sha512 D) :
    KSFlockA sha512 D (deployZk rel) DO :=
  fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF (zk_KSAbort rel hone hext htab decI h) hRef
    hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hC decI) (hO decI) (hAcc decI) (coinsShown_V sha512 true) hN0

end FlockDeploy.KS
