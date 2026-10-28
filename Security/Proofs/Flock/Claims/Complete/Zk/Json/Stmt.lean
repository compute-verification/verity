import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Claims.Complete.Zk.Json.Goals

/-!
# What the masked record needs of the statement, from its setup

`stmtOkZk`: at every good masked case, a non-empty `Σ`, a printable ASCII draw that decodes to the statement's instances
(`DrawOk`), the statement's draw the one it was set up at, and a `Hello` that parses to an object whose `coins` is an
object, printable with the prover's nonce put in: `StmtOkZk`, from the case's setup alone, as
`stmtOk` at the plain verifier.

* `Σ`, the draw's instances and the session's draw are as in `stmtOk`, at the one table `readyZk` asks
  (`CaseZk.one`). Under `--zk` the draw's law is stratified or a subset (`ProvedScope.lawWhy true`), printable as at the
  plain verifier (`good_stratified`, `good_subset`).
* The masked verifier's coins are the keyed coin tree (`Read.coins`), so the hello's `coins` is the tree's spec with an
  empty nonce, an object (`helloZk_of`). `canon` prints the hello and `Flock.JsonParse.parse` reads back an alike object
  (`FlockComplete.JsonRT.parse_canon`), printable as the printed one is (`Same.good`), and so after the prover's nonce is put in
  (`good_setObjVal`).
-/

namespace FlockComplete.JsonRT

open Flock FlockDeploy
open Lean (Json JsonNumber)
open FlockSoundness FlockSoundness.Refine

/-! ## Printable values -/

/-- **A value alike a printable one is printable.** -/
theorem _root_.Claims.Json.Same.good {a b : Json} (h : Same a b) (hg : Good a) : Good b := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num n => exact hg
  | str s => exact .str s
  | arr hs _ ih =>
    cases hg with
    | arr hg =>
    refine .arr fun y hy => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hy
    exact ih i (by omega) hi (hg _ (Array.getElem_mem _))
  | obj _ hn hk _ ih =>
    cases hg with
    | obj _ hg =>
    refine .obj hn fun k v hv => ?_
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 ((hk k).trans (by rw [hv]; rfl))
    exact ih k a v ha hv (hg k a ha)

/-- **Putting a printable value in a printable object** keeps it printable. -/
theorem good_setObjVal {m : Std.TreeMap.Raw String Json compare} (hg : Good (.obj m)) {v : Json} (hv : Good v)
    (k : String) : Good ((Json.obj m).setObjVal! k v) := by
  cases hg with
  | obj hm hg =>
  simp only [Json.setObjVal!]
  refine .obj hm.insert fun k' v' h => ?_
  rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm] at h
  split at h
  · cases h
    exact hv
  · exact hg k' v' (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact h)

/-- The keyed coin tree's `coins`, at a nonce, is printable. -/
theorem good_coins (cs : CoinTree.Spec) (nonce : String) : Good (cs.coins nonce) :=
  good_setObjVal (good_specJson cs) (.str _) _

/-! ## The masked verifier's hello -/

/-- **A canonical object with a keyed coin tree's `coins`** parses back to an object whose `coins` is an object, printable
with the prover's nonce put in. -/
theorem helloZk_mk {L : List (String × Json)} (hd : Distinct L)
    (hg : ∀ p ∈ L, Good p.2) {cs : CoinTree.Spec} (hc : ("coins", cs.coins "") ∈ L) {hello : String}
    (h : Flock.canon (Json.mkObj L) = .ok hello) :
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Flock.JsonParse.parse hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧ Ascii cs.toJson := by
  have hgL := good_mkObj hd hg
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_canon _ _ hgL h
  obtain ⟨cv, hcv, hcvs⟩ := hs.getObjVal (getObjVal?_mkObj hd hc)
  have hgj := hs.good hgL
  have hgc := hcvs.good (good_coins cs "")
  have hs' : Same (.obj (Std.TreeMap.Raw.ofList L compare)) j := hs
  obtain ⟨m, rfl, hm, -⟩ := same_obj_wf hs'
  have hco : cs.coins "" = .obj ((Std.TreeMap.Raw.ofList [("block", Lean.toJson cs.block),
      ("rounds_log", Lean.toJson cs.roundsLog), ("scheme", Json.str CoinTree.SCHEME),
      ("streams", Json.arr (cs.streams.map Json.str))] compare).insert "nonce" (Json.str "")) := rfl
  rw [hco] at hcvs
  obtain ⟨kvs, rfl, hkv, -⟩ := same_obj_wf hcvs
  have hmg := (mem_of_getObjVal hm hcv).2
  have ham : Ascii (.obj m) := ascii_of_same hs' (ascii_of_canon h)
  have hak : Ascii (.obj kvs) := ascii_get hm ham hmg
  obtain ⟨hA, hnk⟩ := ascii_coins (ct := cs) (by rw [hco]; exact ascii_of_same_rev hcvs hak)
  have hget : ((Json.obj m).getObjVal? "coins").toOption.getD Json.null = .obj kvs := by rw [hcv]; rfl
  refine ⟨m, kvs, hp, hm, hkv, hmg, ?_, ?_, hA⟩
  · unfold helloJ
    rw [hget]
    exact good_setObjVal hgj (good_setObjVal hgc (.str _) _) _
  · unfold helloJ
    rw [hget]
    simp only [Json.setObjVal!]
    exact ascii_insert hm ham (ascii_key hm ham hmg) (ascii_insert hkv hak hnk (.str (ascii_zeros 64)))

/-- **The masked verifier's hello**: at a keyed coin tree, it parses to an object whose `coins` is an object, printable
with the prover's nonce put in. -/
theorem helloZk_of {tags : Tags} {sigma : ByteArray} {mPts : ℕ}
    {cs : CoinTree.Spec} {hello : String} (h : Flock.helloOf tags sigma mPts (.tree cs) = .ok hello) :
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Flock.JsonParse.parse hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m)) ∧ Ascii (helloJ (.obj m)) ∧ Ascii cs.toJson := by
  unfold Flock.helloOf at h
  dsimp only at h
  repeat' split at h
  all_goals
    refine helloZk_mk (by simp) ?hg (cs := cs) (by simp) h
    case hg =>
    repeat' first
      | exact good_nil
      | exact good_coins cs ""
      | apply good_append
      | apply good_cons
      | exact .str _
      | exact good_toJson _
      | exact good_arr_strs _
      | exact good_mkObj (by simp) (by simp [good_toJson]; exact .str _)

/-! ## The statement at a good masked case -/

/-- A statement of the masked verifier decodes to inputs that `Raw.ins` read. -/
theorem insOf_true {x : Claims.Msg} {s : Ins} (h : insOf true x = some s) :
    ∃ w : Raw, w.zk = true ∧ w.ins = .ok s := by
  unfold insOf at h
  obtain ⟨w, -, h⟩ := Option.bind_eq_some_iff.1 h
  by_cases hz : w.zk ≠ true
  · simp [hz] at h
  · refine ⟨w, not_not.1 hz, ?_⟩
    simp only [hz, ↓reduceIte] at h
    cases hi : w.ins with
    | error e => rw [hi] at h; cases h
    | ok s' => rw [hi] at h; cases h; rfl

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked case's one table**: its statement is `setupOne`'s, at the session's tags, coins and draw. -/
theorem _root_.FlockComplete.CaseZk.one (c : CaseZk H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧
      Discharge.HiddenExec.setupOne c.s.read.tags c.s.read.I.circuitFile c.s.read.I.publicFile c.s.read.coins
        c.s.read.I.tables c.s.read.draw c.s.read.I.partition c.s.read.I.program = .ok st := by
  have hck := c.scope_ok
  unfold Readable.Read.check at hck
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  have hpi : c.s.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases Discharge.HiddenExec.setupTables_okAny hset with ⟨-, st, h1, hs, hsts⟩ | ⟨hc1, hsz, -⟩
  · exact ⟨st, hsts, hs, h1⟩
  · have := c.ready_parts.1
    omega

/-- **What the masked record needs of the statement holds at every good masked case** (`StmtOkZk`), under the JSON
library's specification: Σ is a SHA-512 digest, the draw is printable and decodes to the statement's instances, the
session's draw is the statement's, and the keyed coin tree's hello parses to an object whose `coins` is an object,
printable with the prover's nonce put in. -/
theorem stmtOkZk : StmtOkZk H rel := by
  intro c
  obtain ⟨st, -, hspec, h1⟩ := c.one
  obtain ⟨⟨c', pop, hl, hd⟩, hinst, hsig, hhello, hct0⟩ := setupOne_stmt h1
  obtain ⟨w, hwz, hw⟩ := insOf_true (setupOf_some c.setup).1
  obtain ⟨hzk, -, hdraw⟩ := Raw.ins_read hw
  have hz : c.s.read.zk = true := hzk.trans hwz
  obtain ⟨d, hdd⟩ := Option.isSome_iff_exists.1 c.ready_parts.2.1
  have hu := unitDraw_of hl (hd d hdd)
  have hU : c.spec.unitDraw = some d := by
    rw [hspec]
    simp [Stmt.spec, hu, Except.toOption]
  obtain ⟨cs, hcs⟩ : ∃ cs, c.s.read.coins = .tree cs := by
    simp only [Readable.Read.coins, hz, ite_true]
    exact ⟨_, rfl⟩
  rw [hcs] at hhello
  obtain ⟨m, kvs, hp, hm, hkv, hmg, hgh, hah, hA⟩ := helloZk_of hhello
  have hctr : c.spec.coinTree = some cs := by
    rw [hspec]
    change st.coinTree = _
    rw [hct0, hcs]
    rfl
  refine ⟨by rw [hspec]; exact hsig, ?_, by rw [hU, hdd], m, kvs, by rw [hspec]; exact hp, hm, hkv, hmg, hgh, hah,
    fun ct hc => by rw [hctr] at hc; cases hc; exact hA⟩
  unfold DrawOk
  rw [hU]
  dsimp only
  obtain ⟨D, hD, hn⟩ := hinst d hu
  have hdl := Ins.setup_drawLaw (setupOf_some c.setup).2 c.ready_parts.2.1
  rw [hdd, hz] at hdl
  simp only [ProvedScope.drawLaw, hD] at hdl
  have hlw : ProvedScope.lawWhy true c.s.read.tags false D = none := by
    cases e : ProvedScope.lawWhy true c.s.read.tags false D with
    | none => rfl
    | some _ => rw [e] at hdl; cases hdl
  have hg : Good d := by
    rcases law_of_lawWhy hlw with ⟨K, strata, hK⟩ | ⟨k, hk⟩
    · exact good_stratified (draw_wf hdraw hdd) hD hK
    · exact good_subset (draw_wf hdraw hdd) hD hk
  refine ⟨fun hn' => ?_, hg, ascii_of_drawn hl (hd d hdd), D, hD, by rw [hspec]; exact hn⟩
  rw [hn'] at hD
  cases hD

end FlockComplete.JsonRT
