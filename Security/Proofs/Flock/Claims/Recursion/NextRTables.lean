import Proofs.Flock.Claims.Recursion.NextRCoins
import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-!
# Every table `verify --plain-tables` admits is `TableOk`

`sessionRun_decides_coins` takes, at every table, `TableOk` and at most 2,048 extra claims at the link's points. A plain
completeness case gives them at its one table (`Case.tableOk`). Here they hold at every table of a statement that sets
up (`setupOf false`) and that `verify --plain-tables`'s statement refusals pass (`PlainTablesOk`: `ProvedScope.checkP`
as `verifyCmd` calls it under the flag, then `sizesOk` on the tables' `m`).

* `tablesOk_of_read`: at an untyped statement that sets up without the verifier's own public input words, every table
  of `22 ≤ m` is `TableOk` with at most 2,048 extra claims. Each table's circuit is the one the scope parses
  (`Readable.tableAt`), so it is well formed (`m = k_log + nbl`, `6 ≤ k_log`). It has table 0's `m`
  (`Readable.tables_fit`), which is at most 35 (`Readable.table0`). Its own setup gives a registered scheme and at most
  1,024 regions (`setupOne_scheme`), at the session's two link points (`setupTables_points`).
* `tablesOk_of_plainTables`: the same from `PlainTablesOk`, whose `checkP` refuses a typed statement and the verifier's
  own public input words, and whose `sizesOk` refuses `m < 22`.
* `sessionRun_decides_tables`: `sessionRun_decides_coins` with `PlainTablesOk` in place of the per-table premise, and
  coins numbering the deployed verifier's `(V false).k x` (`hlen_of_k`) in place of the count premise.
-/

namespace FlockDeploy

open Flock FlockComplete

/-- **`verify --plain-tables`'s statement refusals** at the verifier's read of a statement and the tables it set up, as
`verifyCmd` runs them under the flag: `ProvedScope.checkP` (the verifier's own public input words as it holds them, its
registered reads passed as `false`), then `sizesOk` on the tables' `m`. -/
def PlainTablesOk (r : Readable.Read) (sts : Array Stmt) : Prop :=
  ProvedScope.checkP r.tags r.I.circuitFile r.I.publicFile r.I.tables r.I.count r.I.program r.publicInputs.isSome
      false = .ok () ∧
    ProvedScope.sizesOk false ((tabsOf sts).map Setup.m) = .ok ()

/-- **Every table of `22 ≤ m` is `TableOk`**, with at most 2,048 extra claims at any link points, at an untyped
statement that sets up without the verifier's own public input words. -/
theorem tablesOk_of_read {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : setupOf false x = some (s, spec, sts)) (hty : s.read.tags.typed = false)
    (hpi : s.read.publicInputs = none) (h22 : ∀ st ∈ sts, 22 ≤ st.m) :
    ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ c e, sts[i].extraClaims (pointsOf spec c) = .ok e → e.size ≤ 2048 := by
  obtain ⟨hins, hsu⟩ := setupOf_some hx
  have hset := Ins.setup_read hsu
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have htab : s.read.tablesOf = some sts := by
    simp [Readable.Read.tablesOf, hset, hzk]
  have hset' := hset
  unfold Readable.Read.setup at hset'
  rw [hpi] at hset'
  have hpts : spec.points = 2 := ZkTables.setupTables_points hset'
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset'
  intro i hi
  have h0 : 0 < sts.size := by omega
  obtain ⟨hcirc, -, hwf, hrw⟩ := Readable.tableAt htab hty hpi hi
  obtain ⟨st0, hst0, -, -, -, -, -, hm35, -⟩ := Readable.table0 htab hty hpi
  have hm0 : sts[i].m = sts[0].m := (Readable.tables_fit htab hty hpi hi).1
  have hst0' : sts[0] = st0 := by
    rw [Array.getElem?_eq_getElem h0] at hst0
    exact Option.some.inj hst0
  obtain ⟨d, hd, -⟩ := hall i hi
  obtain ⟨hsch, hreg⟩ := JsonRT.setupOne_scheme hd
  refine ⟨⟨h22 _ (Array.getElem_mem hi), by rw [hm0, hst0']; exact hm35,
    ⟨s.read.tags, s.read.I.circuitFile, s.read.I.tables, hty, hcirc⟩, hrw.hm, hwf.hk6, hsch⟩, fun c e he => ?_⟩
  rw [JsonRT.extraClaims_size he]
  simp only [pointsOf, Array.size_map, Array.size_range, hpts]
  omega

/-- **Every table `verify --plain-tables` admits is `TableOk`**, with at most 2,048 extra claims at any link points:
`checkP` refuses a typed statement and the verifier's own public input words, and `sizesOk` refuses `m < 22`. -/
theorem tablesOk_of_plainTables {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : setupOf false x = some (s, spec, sts)) (hP : PlainTablesOk s.read sts) :
    ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ c e, sts[i].extraClaims (pointsOf spec c) = .ok e → e.size ≤ 2048 := by
  obtain ⟨hc, hsz⟩ := hP
  obtain ⟨-, hchk, -⟩ := ProvedScope.checkP_ok hc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hchk).1
  have hpi : s.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  refine tablesOk_of_read hx hR.2.1 hpi fun st hst => ?_
  have := (Readable.sizesOk_iff _ _).1 hsz st.m (by
    simp only [tabsOf, Array.map_map, Array.mem_map]
    exact ⟨st, hst, rfl⟩)
  exact (Readable.mMin_ge false).trans this

/-- **The honest run decides at every statement `verify --plain-tables` admits**: `sessionRun_decides_coins` with
`PlainTablesOk` in place of the per-table premise, at coins numbering the deployed verifier's `(V false).k x`. -/
theorem sessionRun_decides_tables {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H)
    {rd : Claims.Msg → Option FinalP} {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {Ts : ℕ → F128}
    {fl : FinalP} (hx : setupOf false x = some (s, spec, sts)) (hP : PlainTablesOk s.read sts)
    (hcs : cs.head? = some c₁) (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes)
    (hk : cs.length = (V false).k x) {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (hrd : rd last = some fl)
    (hps : fl.ps = filePs spec sts c₁ pfs) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
        (Ts k)) last) cs = some d := by
  have htab := tablesOk_of_plainTables hx hP
  exact sessionRun_decides_coins h64 hx hcs hresp (hlen_of_k hx (fun i hi => (htab i hi).1) hk)
    (fun i hi => ⟨(htab i hi).1, (htab i hi).2 c₁⟩) hS hrd hps hst

/-- **Non-vacuity of `PlainTablesOk`'s table count**: `checkP`'s count refusal passes plain sessions of 2 and of
`Tags.MAX_TABLES` tables at public outputs, and `sizesOk` passes two tables of `m = 22`. -/
example : ProvedScope.plainTablesWhy Tags.circuit967b8d06 2 = none ∧
    ProvedScope.plainTablesWhy Tags.circuit967b8d06 Tags.MAX_TABLES = none ∧
    ProvedScope.sizesOk false #[22, 22] = .ok () :=
  ⟨rfl, rfl, rfl⟩

end FlockDeploy
