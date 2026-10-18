import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.KS.PlainAcc
import Proofs.Flock.Claims.KS.Trim

/-!
# The recursion's table bound at the trimmed plain record, from Flock's table bound with escapes on accepted runs

`TableBoundθOnAcc`, the table bound `KSAbort_of_tableBoundOnAcc` reads, is `Flock.TableBoundθAccAt` at every statement
(`tableBoundθOnAcc_of_accAt`). At the trimmed record (`FlockDeploy.deployPlainTrim`, the deployed plain instance with
its columns trimmed of trailing zeros) it is carried from the full-width bound at the untrimmed instance by padding
back, with the same escapes (`tableBoundθOnAcc_mapCols`, as `Flock.tableBoundθ_mapCols`), so the trimmed record meets
`Inner.KSAbort` from that bound with no `Extendθ` (`FlockDeploy.KS.plain_KSAbortTrim`, with `ColsOk` and
`ConflictTied` carried over by `Flock.colsOk_mapCols` and `Flock.conflictTied_mapCols`). At the deployed plain record
the full-width bound is `FlockDeploy.KS.plain_tableBoundθAccAt` from the live sessions' bound with escapes on accepted
runs at the statements some run accepts (`FlockDeploy.KS.plain_tableBoundθOnAcc`), so the trimmed plain record meets
`Inner.KSAbort` from that bound alone (`FlockDeploy.KS.plain_KSAbort_of_liveAcc`).
-/

namespace Claims.Recursion

open Claims Claims.Flock

/-- **`TableBoundθOnAcc` from `TableBoundθAccAt` at every statement.** -/
theorem tableBoundθOnAcc_of_accAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Flock.Deploy Coin Col T} {N : Msg → ℕ}
    {Ok : Msg → Col → Prop} (h : ∀ x, Flock.TableBoundθAccAt H D N Ok x) : TableBoundθOnAcc H D N Ok :=
  fun x Ω _ _ P => h x Ω P

/-- **`TableBoundθOnAcc` through `f`, by padding back**: at a table of columns through `f` of ones `Ok` admits, from
the bound at the table they came from, with the same escapes, when `f` keeps `0`, accepted runs open only columns `Ok`
admits, and `f` is one to one on them (`Flock.tableBoundθ_mapCols`). -/
theorem tableBoundθOnAcc_mapCols {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col) (hf : f 0 = 0)
    (Ok : Msg → Col → Prop) (hok : ColsOk H D N Ok) (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v)
    (htab : TableBoundθOnAcc H D N Ok) : TableBoundθOnAcc H (D.mapCols f) N (OkThrough f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hon, hfit, fun ω C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTable D N x C := fun h => hg' (by
    have := goodTable_mapCols D N x f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω C hC hg).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (run H P D.V x ω r).seen 0).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    exact fun he => hne (hinj x u (C p) (hok x Ω P ω r p.val u ha p.isLt hu) (hC p) he)

end Claims.Recursion

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **The trimmed plain record's `k` range, with the relation split**, where the statement has one table wherever the
verifier accepts some run. -/
theorem plain_kSplitsTrim (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x) (x : Claims.Msg) :
    (8 ≤ (deployPlainTrim rel).k x ∧ Claims.Flock.Splits (deployPlainTrim rel) x) ∨
      Claims.Flock.NoAcc sha512 (deployPlainTrim rel).V x :=
  (k_range_plain rel x).elim
    (fun hk => ((hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployPlainTrim rel) rfl rfl hs h1).imp
      (fun h => ⟨hk, h⟩) id) .inr

/-- **The trimmed plain record meets `Inner.KSAbort`**, at SHA-512, at every decoder within the radius: from the table
bound with escapes on accepted runs over full-width tables at the untrimmed instance, with no `Extendθ` (one table
wherever the verifier accepts some run is `plain_hone`). -/
theorem plain_KSAbortTrim (rel : Readable.Parsed → Readable.Relation)
    (htab : TableBoundθOnAcc sha512 (deployPlain rel) (wordLen false) (wide false))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlainTrim rel) decode) : (Inner.ofFlock (deployPlainTrim rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAcc sha512 (deployPlainTrim rel) (wordLen false) (trimWide false) decCoin
    (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (V_k_pos false) (fun _ => cap_pos)
    (plain_kSplitsTrim rel (plain_hone rel)) (plain_hone rel) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (wide false)
      (plain_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (plain_conflictTied rel))
    (tableBoundθOnAcc_mapCols sha512 (deployPlain rel) (wordLen false) trimCol rfl (wide false)
      (plain_colsOk sha512 rel) (wide_trimCol_inj false) htab)
    (coinsShown_V sha512 false) (accLast_V_sha512 false) (decideFree_V sha512 false) (refusable_V sha512 false)
    decode hdec

/-- **The recursion's table bound at the deployed plain record**, at full-width columns, from the live sessions' bound
with escapes on accepted runs at the statements some run accepts. -/
theorem plain_tableBoundθOnAcc (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x) :
    TableBoundθOnAcc sha512 (deployPlain Readable.relation) (wordLen false) (wide false) :=
  tableBoundθOnAcc_of_accAt (plain_tableBoundθAccAt hl)

/-- **The trimmed plain record meets `Inner.KSAbort`** at the deployed relation, at SHA-512, at every decoder within
the radius, from the live sessions' bound with escapes on accepted runs at the statements some run accepts. -/
theorem plain_KSAbort_of_liveAcc (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainLiveAcc x)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlainTrim Readable.relation) decode) :
    (Inner.ofFlock (deployPlainTrim Readable.relation) decode).KSAbort sha512 :=
  plain_KSAbortTrim Readable.relation (plain_tableBoundθOnAcc hl) decode hdec

end FlockDeploy.KS
