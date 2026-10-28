import Proofs.Flock.Claims.Recursion.BoundDeployE
import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.KS.ZkAbortR

/-!
# (P1) at the element-level inner verifier (`FlockDeploy.VE`)

`KSAbort_of_tableBoundOnAccR` at the plain record with `VE` as its verifier (`deployPlainE rel`, trimmed as the claims
read it), at SHA-512, as `plain_KSAbortR` at `V false`. Each fact it reads of the verifier is about what the verifier
accepts, and `VE` accepts more than `V false`, so none carries over from `V false`: each is re-derived from `VE`'s
decision (`nextE_decide`), by `V false`'s proof at the element-level verdict:

* the four code facts: it shows each round's coin (`coinsShown_VE`), accepts only at its last round (`accLast_VE`, the
  element-level verdict's rep streams having their tables' rounds, `streams_size_plainK`), decides without the last
  coin (`decideFree_VE`), and never accepts the empty message (`refusable_VE`);
* it accepts only in scope (`accE_inScope`: the custody checks after the verdict run the scope check, and the
  element-level verdict serves only `fast100`'s `22 ≤ m ≤ 35`, `servedE`), so at one table (`plainE_hone`) and with
  `k ≥ 8` (`kE_range`);
* an accepted run opens columns of the zero column's width (`plainE_colsOk`).

So (P1) at `VE` (`plainE_KSAbortR`) rests on two premises at `deployPlainE rel`: conflicts tied to collisions, proved
at `VE` in `ConflictTiedE` (`plainE_conflictTied`, which also states (P1) at the bound record), and the table bound
counted per round on accepted first runs, proved at `deployPlain rel` (`plain_tableBoundθOnAccR`'s) and not yet at `VE`.
-/

namespace FlockDeploy

open Flock
open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

theorem nextE_rej (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : act false x as cs = .rej) : (nextE x as cs).eval H = (.rej, []) := by
  simp only [nextE, h, Claims.OC.eval]

theorem nextE_coin (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : act false x as cs = .coin c) : (nextE x as cs).eval H = (.msg c.bytes, []) := by
  simp only [nextE, h, Claims.OC.eval]

/-- **Where the element-level verifier answers, the deployed plain one answers alike.** -/
theorem next_msg_of_nextE (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {m : Claims.Msg} (h : ((nextE x as cs).eval H).1 = .msg m) : ((next false x as cs).eval H).1 = .msg m := by
  cases ha : act false x as cs with
  | rej => rw [nextE_rej H ha] at h; cases h
  | coin c => rw [nextE_coin H ha] at h; rw [next_coin H ha]; exact h
  | decide s spec sts link c₁ mids f =>
    simp only [nextE, ha, outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
    split at h <;> simp at h

theorem refusable_VE (H : Claims.Msg → Claims.Msg) : Refusable H VE := fun x =>
  ⟨[], fun as cs => by
    show ((nextE x (as ++ [[]]) cs).eval H).1 ≠ .acc
    rw [nextE_rej H (act_snoc_nil false x as cs)]
    exact fun h => Claims.Out.noConfusion h⟩

theorem coinsShown_VE (H : Claims.Msg → Claims.Msg) : CoinsShown H VE decCoin :=
  fun x as cs c m hl h => coinsShown_V H false x as cs c m hl (next_msg_of_nextE (cs := cs ++ [c]) H h)

theorem decideFree_VE (H : Claims.Msg → Claims.Msg) : DecideFree H VE := by
  intro x as cs c c' hl hacc
  show (nextE x as (cs ++ [c'])).eval H = (nextE x as (cs ++ [c])).eval H
  change ((nextE x as (cs ++ [c])).eval H).1 = .acc at hacc
  cases hact : act false x as (cs ++ [c]) with
  | rej =>
    rw [nextE_rej H hact] at hacc
    cases hacc
  | coin c₂ =>
    rw [nextE_coin H hact] at hacc
    cases hacc
  | decide s spec sts link c₁ mids f =>
    have hlen := act_decide_length (by simp [hl]) hact
    obtain ⟨c₀, cs', rfl⟩ : ∃ c₀ cs', cs = c₀ :: cs' := by
      rcases cs with _ | ⟨c₀, cs'⟩
      · simp only [List.length_nil, zero_add] at hl
        omega
      · exact ⟨c₀, cs', rfl⟩
    have ha : ∀ t, as.getLast? ≠ some (0 :: t) := fun t h => by
      rcases act_snoc_zero (zk := false) (x := x) (cs := c₀ :: cs') (c := c) h with h' | h' <;>
        rw [hact] at h' <;> cases h'
    unfold nextE
    rw [act_cons_snoc false x as c₀ cs' c c' ha]

/-- **Wherever `VE` accepts, the prover sent the rounds of the session the statement sets up against `H`**: the
element-level verdict's rep streams have their tables' rounds (`streams_size_plainK`), as the shipped verdict's do. -/
theorem nextE_acc_length_setup (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (hcs : cs.length = as.length) (h : ((nextE x as cs).eval H).1 = .acc) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (spec' : SessionSpec) (sts' : Array Flock.Stmt),
      setupOf false x = some (s, spec, sts) ∧ setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        as.length = 2 + rounds false spec' sts' := by
  cases ha : act false x as cs with
  | rej => rw [nextE_rej H ha] at h; cases h
  | coin c => rw [nextE_coin H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (nextE_decide H ha).1 h
  refine ⟨s, spec, sts, spec', sts', act_decide ha, hsu, ?_⟩
  obtain ⟨lv, hlv, hcus⟩ := checks_custody hc
  obtain ⟨hsz, hk⟩ := streams_size_plainK (fun _ => true) (Hashes.ofSha512 (Claims.hOf H)) hv hlv
  rw [act_decide_length hcs ha, ← custody_length hcus, rounds_sum, List.length_flatMap,
    Nat.add_comm _ 2, ← sum_pairs (fun i => repRounds false spec' sts'[i]!)]
  congr 2
  apply List.ext_getElem
  · simp [hsz]
  · intro k h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
    exact hk k _

/-- **At SHA-512, `VE` accepts only at its last round.** -/
theorem accLast_VE : AccLast sha512 VE := fun x as cs hcs h => by
  obtain ⟨s, spec, sts, spec', sts', hso, hsu, hl⟩ := nextE_acc_length_setup sha512 hcs h
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hso).2] at hsu
  cases hsu
  show as.length = (match setupOf false x with | some (_, spec, sts) => 2 + rounds false spec sts | none => 1)
  rw [hso]
  dsimp only
  exact hl

/-- **The plain record at the element-level verifier**: `deployPlain rel` with `VE` as its verifier. -/
noncomputable def deployPlainE (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  { deployPlain rel with V := VE }

end FlockDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge.DecodeZL Reader
open FlockSoundness.Refine (ebind_ok)
open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

/-- **The element-level verdict sets up a table**, whatever `keep` and the hashes. -/
theorem verifyWithMK_pos (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    0 < tabs.size := by
  unfold verifyWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  · have : tabs.size ≠ 0 := fun h0 => c0 (by simp [h0])
    omega

/-- **The element-level verdict accepts only served tables**: the session sets up a table, and `fast100` serves every
table's `m` (`served_plain`'s, at the element-level verdict). -/
theorem servedE {h : ByteArray → ByteArray} {spec : SessionSpec} {sts : Array Flock.Stmt} {record : String}
    {proofs : Array ByteArray} (hv : verifyWithE h spec (tabsOf sts) record proofs = .ok ()) :
    0 < sts.size ∧ ∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35 := by
  have hsz : (tabsOf sts).size = sts.size := Array.size_map ..
  obtain ⟨_, schs, -, -, -, hall⟩ := verifyWithMK_tables (fun _ => true) (Hashes.ofSha512 h) hv
  refine ⟨hsz ▸ verifyWithMK_pos (fun _ => true) (Hashes.ofSha512 h) hv, fun i hi => ?_⟩
  obtain ⟨_, _, _, hf, -⟩ := hall i (hsz ▸ hi)
  have hm : (tabsOf sts)[i].m = sts[i].m := by simp [tabsOf, Setup.ofCircuit]
  exact hm ▸ fast100_range hf

/-- **An accepting step of `VE` at SHA-512** is an accepting element-level verdict at the statement's setup, after
which the scope check passed. -/
theorem nextE_acc_check {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((nextE x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts record proofs, setupOf false x = some (ins, spec, sts) ∧
      verifyWithE (Claims.hOf sha512) spec (tabsOf sts) record proofs = .ok () ∧ ins.read.check = .ok () := by
  cases ha : act false x as cs with
  | rej => simp [nextE_rej sha512 ha] at h
  | coin c => simp [nextE_coin sha512 ha] at h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨spec', sts', hsu, hv, hc⟩ := (nextE_decide sha512 ha).1 h
    have hso := act_decide ha
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hso).2] at hsu
    cases hsu
    exact ⟨s, spec, sts, f.record, f.proofs, hso, hv, checks_scope hc⟩

theorem accE_check {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 VE x) :
    ∃ ins spec sts record proofs, setupOf false x = some (ins, spec, sts) ∧
      verifyWithE (Claims.hOf sha512) spec (tabsOf sts) record proofs = .ok () ∧ ins.read.check = .ok () := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := FlockDeploy.KS.stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact nextE_acc_check has

/-- **`VE` accepts only in scope**, at SHA-512: a statement some run accepts parses, in scope (`acc_inScope`'s
proof, at the element-level verdict, `servedE`). -/
theorem accE_inScope (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 VE x) : ∃ s, parse false rel x = some s ∧ Claims.Flock.InScope false s := by
  obtain ⟨ins, spec, sts, record, proofs, hset, hv, hc⟩ := accE_check hacc
  obtain ⟨hi, hs⟩ := setupOf_some hset
  have hr : readOf false x = some ins.read := by unfold readOf; rw [hi]; rfl
  have hzk : ins.read.zk = false := readOf_zk hr
  have hrp := read_parse_of_setup hs
  have hparse : parse false rel x = some (Readable.toReadable rel { ins.read with sts }) := by
    unfold parse Readable.parseOf Readable.parseBytes
    rw [hr, Option.bind_some, hrp, Option.map_some]
  refine ⟨_, hparse, ?_⟩
  obtain ⟨h0, hm⟩ := servedE hv
  have hin := (Readable.inScope_iff rel (Readable.parses_of_parse hrp)).2 ⟨hc, ?_⟩
  · exact hzk ▸ hin
  · simpa only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some] using
      (hm 0 h0).1

/-- **`VE` accepts only at one table**, at SHA-512: the plain scope admits one (`accE_inScope`). -/
theorem plainE_hone (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (∃ s, (deployPlainE rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlainE rel).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 VE x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := accE_inScope rel hno
    obtain ⟨-, -, -, -, -, -, -, -, ht, -⟩ := hin
    exact .inl ⟨s, hs, (Claims.Flock.Stmt.tables_eq_one_iff s).2 (by simpa using ht)⟩

/-- **`VE`'s `k` range**: at every statement, `k ≥ 8` or `VE` accepts no run at SHA-512 (`k_range_plain`'s proof). -/
theorem kE_range (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    8 ≤ (deployPlainE rel).k x ∨ Claims.Flock.NoAcc sha512 (deployPlainE rel).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 VE x
  · exact .inr hacc
  obtain ⟨s, spec, sts, record, proofs, hs, hv, -⟩ := accE_check hacc
  obtain ⟨h0, hm⟩ := servedE hv
  obtain ⟨h22, h35⟩ := hm 0 h0
  have hd := (Claims.Flock.plainShape_table ⟨sts[0].m - 22, by omega⟩)
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have hlen : 10 ≤ (Claims.Flock.plainLevel0 sts[0].m).d := by
    unfold Claims.Flock.plainLen Claims.Flock.plainCols at hd
    rw [hi] at hd
    omega
  refine .inl (eight_le_k ?_)
  show 2 ^ 10 ≤ N₀ false x
  exact le_trans (two_pow_le hlen) (N₀_ge_of_setup hs h0)

/-- **An accepted run of `VE` ends on a final message on the statement**, at a statement that sets up. -/
theorem accE_last {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} (h : (Claims.run H P VE x ω r).out = .acc) :
    ∃ s spec sts f, setupOf false x = some (s, spec, sts) ∧ lastFinal (Claims.run H P VE x ω r).seen = some f ∧
      f.stmt = x := by
  have ha := FlockDeploy.KS.run_acc_next H P VE x ω r h
  generalize hcs : (List.range _).map (Claims.coinSeq r) = cs at ha
  replace ha : ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  cases hact : act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs with
  | rej => rw [nextE_rej H hact] at ha; cases ha
  | coin c => rw [nextE_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    obtain ⟨_, _, -, -, hc⟩ := (nextE_decide H hact).1 ha
    refine ⟨s, spec, sts, f, act_decide hact, ?_, (Claims.Bridge.Custody.checks_ok hc).1⟩
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl

/-- **An accepted run of `VE` opens columns of the reader's table 0**: each has the statement's table 0's level-0
width (`acc_cols`'s proof). -/
theorem accE_cols {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} (h : (Claims.run H P VE x ω r).out = .acc) {q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (cols false (Claims.run H P VE x ω r).seen 0) q = some u) :
    ∃ spec st, (tables false x).head? = some (spec, st) ∧
      u.length = 2 ^ (if false then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  obtain ⟨s, spec, sts, f, hs, hf, hx⟩ := accE_last h
  unfold cols at hu
  split at hu
  · rename_i rr hrr
    obtain ⟨f', s', spec', sts', st, hf', hs', hst, hm⟩ := runOf_m hrr
    rw [hf] at hf'
    cases hf'
    rw [hx, hs] at hs'
    cases hs'
    refine ⟨spec, st, tables_head hs hst, ?_⟩
    obtain ⟨p, hp, rfl⟩ := exists_of_lookCol hu
    simp only [Bool.false_eq_true, ↓reduceIte] at hp ⊢
    rw [plainRunCols_width hp, hm]
  · simp [Claims.Flock.lookCol] at hu

/-- **`ColsOk` at the plain record at `VE`**: an accepted run opens only columns of the zero column's width. -/
theorem plainE_colsOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ColsOk H (deployPlainE rel) (wordLen false) (wide false) := by
  intro x Ω P ω r q u hacc _ hu
  obtain ⟨spec, st, hh, hl⟩ := accE_cols hacc hu
  unfold wide
  rw [hl, zeroCol_length hh]

/-- **The trimmed record at `VE`'s `k` range, with the relation split**, where the statement has one table wherever
`VE` accepts some run (`plainE_hone`). -/
theorem plainE_kSplits (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (8 ≤ ((deployPlainE rel).mapCols trimCol).k x ∧ Claims.Flock.Splits ((deployPlainE rel).mapCols trimCol) x) ∨
      Claims.Flock.NoAcc sha512 ((deployPlainE rel).mapCols trimCol).V x :=
  (kE_range rel x).elim
    (fun hk => ((plainE_hone rel x).imp_left fun ⟨_, hs, h1⟩ =>
      splits_of_one ((deployPlainE rel).mapCols trimCol) rfl rfl hs h1).imp (fun h => ⟨hk, h⟩) id) .inr

/-- **(P1) at `VE`**: the trimmed plain record at the element-level verifier meets `Inner.KSAbort`, at SHA-512, at
every decoder within the radius, from conflicts tied to collisions and the table bound counted per round, both at the
untrimmed record at `VE` (`plain_KSAbortR`'s assembly, with `VE`'s code facts). -/
theorem plainE_KSAbortR (rel : Readable.Parsed → Readable.Relation)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlainE rel) (wordLen false))
    (htab : TableBoundθOnAccR sha512 (deployPlainE rel) (wordLen false) (wide false))
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace)
    (hdec : decOK ((deployPlainE rel).mapCols trimCol) decode) :
    (Inner.ofFlock ((deployPlainE rel).mapCols trimCol) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAccR sha512 ((deployPlainE rel).mapCols trimCol) (wordLen false) (trimWide false) decCoin
    (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (V_k_pos false) (fun _ => cap_pos)
    (plainE_kSplits rel) (plainE_hone rel) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol (wide false)
      (plainE_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol hconf)
    (tableBoundθOnAccR_mapCols sha512 (deployPlainE rel) (wordLen false) trimCol rfl (wide false)
      (plainE_colsOk sha512 rel) (wide_trimCol_inj false) htab)
    (coinsShown_VE sha512) accLast_VE (decideFree_VE sha512) (refusable_VE sha512) decode hdec

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **At `VE`'s steps and the trimmed columns and weight, `DE` is the trimmed plain record at `VE`.** -/
theorem deE_VE_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    deE rel FlockDeploy.VE.next (FlockDeploy.deployPlainTrim rel).cols (FlockDeploy.deployPlainTrim rel).weight =
      (FlockDeploy.deployPlainE rel).mapCols FlockDeploy.trimCol :=
  rfl

end RecDeploy
