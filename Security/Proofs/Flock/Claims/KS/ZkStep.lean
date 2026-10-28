import Proofs.Flock.Claims.KS.ZkLock
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.Width

/-!
# The masked simulator's rounds are the record's

Where `P`'s deployed run at `zkΦ`'s coins decides and accepts, the masked simulator's live outcome is the record's view
(`zkSimRun`, so `ZkSimRunAt` holds at every statement: `zkSimRunAt`):

* the roots of `P`'s first message are the link's (`act_decide_link`);
* the first live coin's points are the link coin's, which `zkΦ` takes from it (`zkΦ_entry`, `zkSim_pts`);
* each rep stream is the record's (`zkSim_reps`): the record's rounds, flattened in `spec.repStreams` order, are the
  round messages with their coins (`custody_ok`), stream `k` has `zkRepNsOf spec sts[k / 2]`'s counts
  (`streams_ns_zk`), each of `1` to `65536` coins (`zk_liveOf_n`), and the simulator submits deployed round `i` to
  stream `zkStream i`, so the rounds tagged with their stream and filtered by stream read each stream back
  (`zk_filter_tagged`, `zk_tagged_eq`);
* the last message's proof files are the record's (`act_decide_final`).

With the abort (`zkSimAbortAt`), masked KS and the recursion's `zk_KSFlockJR` chain rest on the fork bounds and the
view's clash alone (`zk_KS_of_fork`, `zk_tableBoundJOnAccR_of_fork`, `zk_KSFlock_of_fork`).
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-! ## Runs along their coins -/

/-- **A run's messages are `P`'s along its coins** (`zkDepMsg`), wherever the verifier answers each round with its
coin's bytes. -/
theorem zk_stepWith_dep {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (V' : Claims.Verifier Coin)
    (x : Claims.Msg) (ω : Ω) (c : ℕ → Coin) (hV : Claims.Bridge.AnswersCoin H V' x Coin.bytes) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      hist = zkDepHist H P x ω c as.length → as = (List.range as.length).map (zkDepMsg H P x ω c) →
      t.msgs.map Prod.fst = as →
      (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as t).msgs.map
          Prod.fst =
        (List.range (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as
          t).msgs.length).map (zkDepMsg H P x ω c) := by
  intro n
  induction n with
  | zero =>
    intro hist as t _ ha ht
    show t.msgs.map Prod.fst = (List.range t.msgs.length).map _
    rw [← List.length_map (f := Prod.fst), ht]
    exact ha
  | succ n ih =>
    intro hist as t hh ha ht
    have hp : ((P.next x hist ω).eval H).1 = zkDepMsg H P x ω c as.length := by rw [hh]; rfl
    have ha' : as ++ [((P.next x hist ω).eval H).1] =
        (List.range (as.length + 1)).map (zkDepMsg H P x ω c) := by
      rw [hp, List.range_succ, List.map_append, ← ha]
      rfl
    have hl' : (as ++ [((P.next x hist ω).eval H).1]).length = as.length + 1 := by
      rw [List.length_append, List.length_singleton]
    simp only [Claims.stepWith]
    cases hv : ((V'.next x (as ++ [((P.next x hist ω).eval H).1])
        ((List.range (as ++ [((P.next x hist ω).eval H).1]).length).map c)).eval H).1 with
    | msg m =>
      have hm : m = (c as.length).bytes := by
        rw [hl', List.range_succ, List.map_append] at hv
        exact hV _ _ _ _ hv
      refine ih _ _ _ ?_ ?_ ?_
      · rw [hl', hm, hh]
        rfl
      · rw [hl']
        exact ha'
      · rw [List.map_append, ht]
        rfl
    | acc =>
      show (t.msgs ++ _).map Prod.fst = (List.range (t.msgs ++ _).length).map _
      rw [List.length_append, List.length_singleton, ← List.length_map (f := Prod.fst), ht, List.map_append, ht]
      exact ha'
    | rej =>
      show (t.msgs ++ _).map Prod.fst = (List.range (t.msgs ++ _).length).map _
      rw [List.length_append, List.length_singleton, ← List.length_map (f := Prod.fst), ht, List.map_append, ht]
      exact ha'

namespace Gen

variable [S : ZkSetup]

/-- **A deployed run's messages are `P`'s along its coins** (`zk_stepWith_dep`). -/
theorem zk_run_dep {zk : Bool} (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω) (r : Fin ((V zk).k x) → Coin) :
    (Claims.run H P (V zk) x ω r).msgs.map Prod.fst =
      (List.range (Claims.run H P (V zk) x ω r).msgs.length).map (zkDepMsg H P x ω (Claims.coinSeq r)) :=
  zk_stepWith_dep H P (V zk) x ω _ (Claims.Bridge.answersCoin_deploy zk H x) _ [] [] _ rfl rfl rfl

end Gen

/-- **A deployed run's messages are `P`'s along its coins** (`zk_stepWith_dep`). -/
theorem zk_run_dep {zk : Bool} (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω) (r : Fin ((V zk).k x) → Coin) :
    (Claims.run H P (V zk) x ω r).msgs.map Prod.fst =
      (List.range (Claims.run H P (V zk) x ω r).msgs.length).map (zkDepMsg H P x ω (Claims.coinSeq r)) :=
  Gen.zk_run_dep (S := S₀) (zk := zk) (H := H) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r)

/-! ## A decision's messages and record -/

namespace Gen

variable [S : ZkSetup]

/-- **A decision's round messages are the messages between the first and the last, after their tags**, each with its
coin, and its link coin is the first message's. -/
theorem zk_act_decide_mids {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ a₁ rest crest, as = a₁ :: rest ∧ cs = c₁ :: crest ∧
      mids = (rest.dropLast.map List.tail).zip crest.dropLast := by
  unfold act at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  cases h
  exact ⟨_, _, _, rfl, rfl, rfl⟩

end Gen

/-- **A decision's round messages are the messages between the first and the last, after their tags**, each with its
coin, and its link coin is the first message's. -/
theorem zk_act_decide_mids {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ a₁ rest crest, as = a₁ :: rest ∧ cs = c₁ :: crest ∧
      mids = (rest.dropLast.map List.tail).zip crest.dropLast :=
  Gen.zk_act_decide_mids (S := S₀) (zk := zk) (x := x) (as := as) (cs := cs) (s := s) (spec := spec) (sts := sts)
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **A decided run at `zkΦ`'s coins, message by message**: its link coin is the first coin, its round messages are
`P`'s between the first and the last, after their tags, each with its coin, and it decides on `P`'s last message. -/
theorem zk_decides_msgs {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec}
    {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) {u : zkTape x} {r' : Fin (zkN x + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc)
    (hact : act true x ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).msgs.map Prod.fst).length).map
          (Claims.coinSeq (zkΦ x spec sts (u, r')))) = .decide s spec' sts' link c₁ mids f) :
    c₁ = Claims.coinSeq (zkΦ x spec sts (u, r')) 0 ∧
      mids = (List.range (zkN x)).map (fun k => ((zkDm P x spec sts ω u r' (k + 1)).tail,
        Claims.coinSeq (zkΦ x spec sts (u, r')) (k + 1))) ∧
      finalOf (zkDm P x spec sts ω u r' (zkN x + 1)) = some f := by
  have hlen := next_acc_length_sha512 true (by simp)
    (run_acc_next sha512 P (V true) x ω (zkΦ x spec sts (u, r')) hacc)
  have has := zk_run_dep sha512 P x ω (zkΦ x spec sts (u, r'))
  rw [List.length_map] at hlen
  rw [hlen] at has
  rw [has, List.length_map, List.length_range, zk_range_map_ends _ (by omega),
    zk_range_map_ends (Claims.coinSeq (zkΦ x spec sts (u, r'))) (by omega)] at hact
  have hfin := act_decide_final hact
  obtain ⟨a₁, rest, crest, h1, h2, h3⟩ := zk_act_decide_mids hact
  injection h1 with _ h1
  injection h2 with h2a h2
  subst h1 h2
  refine ⟨h2a.symm, ?_, ?_⟩
  · rw [h3, List.dropLast_concat, List.dropLast_concat, List.map_map, show (V true).k x - 2 = zkN x by omega]
    rw [List.zip_map']
    rfl
  · rw [← List.cons_append, List.getLast?_concat] at hfin
    rw [show zkN x + 1 = (V true).k x - 1 by omega]
    exact hfin

end Gen

/-- **A decided run at `zkΦ`'s coins, message by message**: its link coin is the first coin, its round messages are
`P`'s between the first and the last, after their tags, each with its coin, and it decides on `P`'s last message. -/
theorem zk_decides_msgs {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec}
    {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) {u : zkTape x} {r' : Fin (zkN x + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc)
    (hact : act true x ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).msgs.map Prod.fst).length).map
          (Claims.coinSeq (zkΦ x spec sts (u, r')))) = .decide s spec' sts' link c₁ mids f) :
    c₁ = Claims.coinSeq (zkΦ x spec sts (u, r')) 0 ∧
      mids = (List.range (zkN x)).map (fun k => ((zkDm P x spec sts ω u r' (k + 1)).tail,
        Claims.coinSeq (zkΦ x spec sts (u, r')) (k + 1))) ∧
      finalOf (zkDm P x spec sts ω u r' (zkN x + 1)) = some f :=
  Gen.zk_decides_msgs (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (spec := spec) (sts := sts) (hL := hL) (u := u)
    (r' := r') (s := s) (spec' := spec') (sts' := sts') (link := link) (c₁ := c₁) (mids := mids) (f := f) (hacc := hacc)
    (hact := hact)

theorem zk_forall_of_forall₂ {α β : Type} {Q : β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ (fun _ b => Q b) l₁ l₂ → ∀ b ∈ l₂, Q b
  | _, _, .nil => fun _ hb => absurd hb List.not_mem_nil
  | _, _, .cons h t => fun b hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h
    · exact zk_forall_of_forall₂ t b hb

theorem zk_check_true {b : Bool} {why : String} {u : Unit} (h : Record.check b why = .ok u) : b = true := by
  unfold Record.check at h
  split at h
  · assumption
  · cases h

/-- **The verifier's decoder reads only rounds of `1` to `65536` coins** (S12). -/
theorem zk_decodeWith_n (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {sess : RunRecord}
    (h : Record.decodeWith hs spec text = .ok sess) :
    ∀ x ∈ sess.streams.toList, ∀ rd ∈ x.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  unfold Record.decodeWith Record.decodeJWith at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine zk_forall_of_forall₂ (forall₂_of_array_mapM
      (P := fun _ (b : StreamRec) => ∀ rd ∈ b.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536) (fun s x hx => ?_)
      (by assumption))
    obtain ⟨name, -, hx⟩ := ebind_ok hx
    obtain ⟨rs, -, hx⟩ := ebind_ok hx
    obtain ⟨rounds, hrounds, hx⟩ := ebind_ok hx
    repeat' obtain ⟨_, _, hx⟩ := ebind_ok hx
    cases hx
    refine zk_forall_of_forall₂ (forall₂_of_mapIdxM
      (P := fun _ (rd : Round) => 1 ≤ rd.n ∧ rd.n ≤ 65536) (fun k r rd hr => ?_) hrounds)
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨n, -, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, hc, hr⟩ := ebind_ok hr
    have hb := zk_check_true hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
    repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
    all_goals
      cases hr
      exact ⟨hb.1.1.1, hb.1.1.2⟩

/-- **A masked record's live rounds have `1` to `65536` coins**, where the masked verdict accepts it: `liveOf`'s rounds
are the decoder's, by coin count (`liveOf_decodeWith`). -/
theorem zk_liveOf_n (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec tabs record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    ∀ rd ∈ lv.streams.toList.flatMap (·.toList), 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  obtain ⟨sess, -, hdec, -⟩ := zkVerifyWithM_tables hs hv
  obtain ⟨-, -, hsz, hall⟩ := liveOf_decodeWith hs hl hdec
  intro rd hrd
  obtain ⟨st, hst, hrd⟩ := List.mem_flatMap.1 hrd
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hst
  have hk' : k < lv.streams.size := by simpa using hk
  simp only [Array.getElem_toList] at hrd
  obtain ⟨stream, hfind, hfa⟩ := hall k (hsz ▸ hk') hk'
  have hmem : stream ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hfind)
  have hn := map_n_of_forall₂ (hfa.imp fun _ _ h => ⟨h.1, h.2⟩)
  have : rd.n ∈ stream.rounds.toList.map Round.n := hn ▸ List.mem_map_of_mem hrd
  obtain ⟨rd', hrd', e⟩ := List.mem_map.1 this
  rw [← e]
  exact zk_decodeWith_n hs hdec stream hmem rd' hrd'

namespace Gen

variable [S : ZkSetup]

/-- **A decided run at a masked statement, at its record**: the decision is at the statement's setup; the record has
two rep streams per table, stream `q` with `zkRepNsOf spec sts[q / 2]`'s counts, its rounds flattened the session's
counts `zkNs`, each of `1` to `65536` coins, and the round messages' bytes and coins. -/
theorem zk_decides_record {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts))
    {r : Fin ((V true).k x) → Coin} {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
          .decide s spec' sts' link c₁ mids f) :
    ins = s ∧ spec = spec' ∧ sts = sts' ∧
      (Claims.Bridge.Custody.liveD spec f.record).streams.size = 2 * sts.size ∧
      (∀ (q : ℕ) (hq : q < (Claims.Bridge.Custody.liveD spec f.record).streams.size),
        (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList.map Round.n = zkRepNsOf spec sts[q / 2]!) ∧
      ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)).map Round.n =
        zkNs spec sts ∧
      (∀ rd ∈ (Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList),
        1 ≤ rd.n ∧ rd.n ≤ 65536) ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)) mids := by
  have hs := act_decide hact
  rw [hset] at hs
  cases hs
  have hn := (next_decide_sha512 hact).1 (run_acc_next sha512 P (V true) x ω r hacc)
  rw [if_pos rfl] at hn
  obtain ⟨hv, hc⟩ := hn
  obtain ⟨-, lv, hl, hcus⟩ := Claims.Bridge.Custody.checks_ok hc
  have hv' : Zk.verifyWithM (Hashes.ofSha512 (Claims.hOf sha512) : Hashes Flock.V) spec (tabsOf sts) f.record
      f.proofs = .ok () := by
    have e := verdictWith_sha512 hset f.record f.proofs
    rw [if_pos rfl] at e
    rw [← e] at hv
    exact hv
  have hD : Claims.Bridge.Custody.liveD spec f.record = lv := by simp [Claims.Bridge.Custody.liveD, hl]
  obtain ⟨hsz, hns⟩ := streams_ns_zk _ hv' hl
  rw [hD]
  exact ⟨rfl, rfl, rfl, hsz, hns, rounds_ns_zk _ hv' hl, zk_liveOf_n _ hv' hl,
    (Claims.Bridge.Custody.custody_ok hcus).2.2.2⟩

end Gen

/-- **A decided run at a masked statement, at its record**: the decision is at the statement's setup; the record has
two rep streams per table, stream `q` with `zkRepNsOf spec sts[q / 2]`'s counts, its rounds flattened the session's
counts `zkNs`, each of `1` to `65536` coins, and the round messages' bytes and coins. -/
theorem zk_decides_record {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts))
    {r : Fin ((V true).k x) → Coin} {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
          .decide s spec' sts' link c₁ mids f) :
    ins = s ∧ spec = spec' ∧ sts = sts' ∧
      (Claims.Bridge.Custody.liveD spec f.record).streams.size = 2 * sts.size ∧
      (∀ (q : ℕ) (hq : q < (Claims.Bridge.Custody.liveD spec f.record).streams.size),
        (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList.map Round.n = zkRepNsOf spec sts[q / 2]!) ∧
      ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)).map Round.n =
        zkNs spec sts ∧
      (∀ rd ∈ (Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList),
        1 ≤ rd.n ∧ rd.n ≤ 65536) ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)) mids :=
  Gen.zk_decides_record (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (r := r) (s := s) (spec' := spec') (sts' := sts') (link := link) (c₁ := c₁) (mids := mids) (f := f)
    (hacc := hacc) (hact := hact)

/-! ## The coins `zkΦ` takes from the live coins -/

namespace Gen

variable [S : ZkSetup]

/-- **`zkΦ`'s coin `j` is the live coin `j` on its first `zkCnt j` entries.** -/
theorem zkΦ_entry {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x)
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {j : ℕ} (hj : j < (V true).k x) (e : Fin 65536)
    (he : e.val < zkCnt x spec sts j) :
    Claims.coinSeq (zkΦ x spec sts (u, r')) j e = (Claims.coinSeq r' j e : 𝔽) := by
  rw [zkΦ_eq hL]
  simp only [Claims.coinSeq, hj, show j < zkN x + 2 by omega, ↓reduceDIte]
  exact ite_eq_left (α := 𝔽) (show e.val < zkNOf x spec sts ⟨j, hj⟩ from he)

end Gen

/-- **`zkΦ`'s coin `j` is the live coin `j` on its first `zkCnt j` entries.** -/
theorem zkΦ_entry {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x)
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {j : ℕ} (hj : j < (V true).k x) (e : Fin 65536)
    (he : e.val < zkCnt x spec sts j) :
    Claims.coinSeq (zkΦ x spec sts (u, r')) j e = (Claims.coinSeq r' j e : 𝔽) :=
  Gen.zkΦ_entry (S := S₀) (x := x) (spec := spec) (sts := sts) (hL := hL) (u := u) (r' := r') (j := j) (hj := hj)
    (e := e) (he := he)

/-- **A coin's first `n` values are a live coin's first `n` entries** where the two agree there. -/
theorem zk_vals_takeE (c : Coin) (u : LCoin 65536) {n : ℕ} (hn : n ≤ 65536)
    (h : ∀ e : Fin 65536, e.val < n → c e = (u e : 𝔽)) :
    c.vals n = Array.ofFn fun i : Fin n => (zkTakeE n u i).val := by
  refine Array.ext (by simp [Coin.vals]) fun i h₁ h₂ => ?_
  have hi : i < n := by simpa using h₂
  simp only [Coin.vals, Array.getElem_map, Array.getElem_range, Array.getElem_ofFn, zkTakeE,
    show i < 65536 by omega, ↓reduceDIte, Coin.at]
  exact h ⟨i, by omega⟩ hi

namespace Gen

variable [S : ZkSetup]

/-- **A masked statement in scope has two link points**: the setup's link points (`setupTables_pointsAny`). -/
theorem zk_points {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : spec.points = 2 := by
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, -, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  exact (Discharge.HiddenExec.setupTables_pointsAny hsr).1

end Gen

/-- **A masked statement in scope has two link points**: the setup's link points (`setupTables_pointsAny`). -/
theorem zk_points {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : spec.points = 2 :=
  Gen.zk_points (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

namespace Gen

variable [S : ZkSetup]

/-- **The link's points**: the first live coin's points on the masked wire are those of `zkΦ`'s link coin, whose
`2·mPts` link coordinates are the first live coin's entries. -/
theorem zkSim_pts {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) {J : ℕ}
    (hw : 2 * spec.mPts ≤ 65536) (hp : spec.points = 2) (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536)
    (link : List ByteArray × List ByteArray) (f : Final) :
    (ptsIf (zkWire J spec.mPts hw)).proj (Claims.coinSeq r' 0) =
      (Claims.Bridge.Custody.viewOf J spec.mPts spec link (Claims.coinSeq (zkΦ x spec sts (u, r')) 0) f).pts := by
  rw [zk_ptsIf_proj]
  funext k t
  have hkt : t.val + spec.mPts * k.val < 2 * spec.mPts := by
    have := k.isLt
    have := t.isLt
    rcases k with ⟨_ | _ | _, _⟩ <;> simp <;> omega
  show ptsE spec.mPts (zkTakeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
    Refine.fOf ((pointsOf spec (Claims.coinSeq (zkΦ x spec sts (u, r')) 0))[k.val]!)[t.val]!
  have e1 : ptsE spec.mPts (zkTakeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
      zkTakeE (2 * spec.mPts) (Claims.coinSeq r' 0) ⟨t.val + spec.mPts * k.val, hkt⟩ := rfl
  have e2 : k.val * spec.mPts + t.val = t.val + spec.mPts * k.val := by rw [Nat.mul_comm, Nat.add_comm]
  have hk2 : k.val < 2 := k.isLt
  simp only [pointsOf, hp]
  rw [e1, getElem!_pos _ k.val (by simp [hk2]), Array.getElem_map, Array.getElem_range,
    getElem!_pos _ t.val (by simp), Array.getElem_map, Array.getElem_range, e2]
  have he : t.val + spec.mPts * k.val < 65536 := by omega
  simp only [zkTakeE, Coin.at, he, ↓reduceDIte]
  have hc : t.val + spec.mPts * k.val < zkCnt x spec sts 0 := by
    unfold zkCnt
    simp only [↓reduceIte]
    exact lt_max_of_lt_right hkt
  exact (zkΦ_entry (j := 0) hL u r' (by omega) ⟨_, he⟩ hc).symm

end Gen

/-- **The link's points**: the first live coin's points on the masked wire are those of `zkΦ`'s link coin, whose
`2·mPts` link coordinates are the first live coin's entries. -/
theorem zkSim_pts {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) {J : ℕ}
    (hw : 2 * spec.mPts ≤ 65536) (hp : spec.points = 2) (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536)
    (link : List ByteArray × List ByteArray) (f : Final) :
    (ptsIf (zkWire J spec.mPts hw)).proj (Claims.coinSeq r' 0) =
      (Claims.Bridge.Custody.viewOf J spec.mPts spec link (Claims.coinSeq (zkΦ x spec sts (u, r')) 0) f).pts :=
  Gen.zkSim_pts (S := S₀) (x := x) (spec := spec) (sts := sts) (hL := hL) (J := J) (hw := hw) (hp := hp) (u := u)
    (r' := r') (link := link) (f := f)

/-! ## The rep streams -/

/-- **The rep streams after `N` rounds**: each stream's rounds are the submissions to it, in order. -/
theorem zkFoldS_eq {J : ℕ} (dg : ByteArray → ByteArray) :
    ∀ (N : ℕ) (S : ℕ → SubmitJ J) (R : ℕ → LCoin 65536) (rs : Fin J → Fin 2 → List Round) (j : Fin J) (q : Fin 2),
      zkFoldS dg S R N rs j q = rs j q ++ ((List.range N).filter fun k => decide ((S k).tab = j ∧ (S k).rep = q)).map
        fun k => Refine.roundOf dg (S k).bytes (S k).n (zkTakeE (S k).n (R k))
  | 0, S, R, rs, j, q => by simp [zkFoldS]
  | N + 1, S, R, rs, j, q => by
    rw [zkFoldS, zkFoldS_eq dg N, List.range_succ_eq_map, List.filter_cons, List.filter_map]
    unfold addRoundJ
    by_cases hp : (S 0).tab = j ∧ (S 0).rep = q
    · obtain ⟨h1, h2⟩ := hp
      subst h1 h2
      rw [upd_self]
      simp [Function.comp_def]
    · rw [upd_ne _ _ fun h => hp ⟨h.1.symm, h.2.symm⟩]
      simp [hp, Function.comp_def]

/-- **Each stream read back from the rounds tagged with their stream**: filtered by stream `v`, without the tags, they
are stream `v`'s rounds, none past the last stream. -/
theorem zk_filter_tagged {β : Type} (g : ℕ → List β) :
    ∀ (n v : ℕ), (((List.range n).flatMap fun w => (g w).map fun a => (w, a)).filter fun p => decide (p.1 = v)).map
      Prod.snd = if v < n then g v else []
  | 0, v => by simp
  | n + 1, v => by
    rw [List.range_succ, List.flatMap_append, List.filter_append, List.map_append, zk_filter_tagged g n v,
      List.flatMap_singleton]
    by_cases hv : n = v
    · subst hv
      simp [List.filter_map, Function.comp_def]
    · have h2 : (((g n).map fun a => (n, a)).filter fun p => decide (p.1 = v)) = [] := by
        rw [List.filter_eq_nil_iff]
        intro a ha
        obtain ⟨b, -, rfl⟩ := List.mem_map.1 ha
        simpa using hv
      rw [h2, List.map_nil, List.append_nil]
      by_cases h : v < n
      · rw [if_pos h, if_pos (by omega)]
      · rw [if_neg h, if_neg (by omega)]

/-- **Rounds selected by their stream, against tagged rounds**: where the `k`-th tagged round has the stream `idx k`
and, read by `φ`, is `g k`, selecting the `k < N` of stream `v` and reading them by `g` is reading the tagged rounds
of stream `v` by `φ`. -/
theorem zk_tagged_eq {β γ : Type} (T : List (ℕ × β)) (N : ℕ) (idx : ℕ → ℕ) (g : ℕ → γ) (φ : β → γ)
    (hlen : T.length = N) (h : ∀ k (hk : k < T.length), T[k].1 = idx k ∧ φ T[k].2 = g k) (v : ℕ) :
    ((List.range N).filter fun k => decide (idx k = v)).map g =
      ((T.filter fun p => decide (p.1 = v)).map Prod.snd).map φ := by
  have hTT : (List.range N).map (fun k => (idx k, g k)) = T.map (fun p => (p.1, φ p.2)) :=
    List.ext_getElem (by simp [hlen]) fun k h₁ h₂ => by
      simp only [List.getElem_map, List.getElem_range]
      obtain ⟨a, b⟩ := h k (by simpa using h₂)
      exact Prod.ext a.symm b.symm
  calc ((List.range N).filter fun k => decide (idx k = v)).map g
      = (((List.range N).map fun k => (idx k, g k)).filter fun p => decide (p.1 = v)).map Prod.snd := by
        rw [List.filter_map, List.map_map]
        rfl
    _ = ((T.map fun p => (p.1, φ p.2)).filter fun p => decide (p.1 = v)).map Prod.snd := by rw [hTT]
    _ = ((T.filter fun p => decide (p.1 = v)).map Prod.snd).map φ := by
        rw [List.filter_map, List.map_map, List.map_map]
        rfl

namespace Gen

variable [S : ZkSetup]

/-- **The simulator's rep streams are the record's**: where the record's streams are `2·|sts|`, stream `q` with
`zkRepNsOf spec sts[q / 2]`'s counts, its rounds flattened the session's counts `zkNs`, each of `1` to `65536` coins,
and the round messages `dm (k + 1)` with the coins `c (k + 1)`, which are the live coins on their first `zkCnt`
entries, the simulator's submissions fold to the record's streams. -/
theorem zkSim_reps {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {J : ℕ} (hJ : sts.size ≤ J)
    (h0 : 0 < sts.size) (dm : ℕ → Claims.Msg) (c : ℕ → Coin) (r' : Fin (zkN x + 2) → LCoin 65536) (lv : Live)
    {mids : List (Claims.Msg × Coin)} (hsz : lv.streams.size = 2 * sts.size)
    (hns : ∀ (q : ℕ) (hq : q < lv.streams.size), lv.streams[q].toList.map Round.n = zkRepNsOf spec sts[q / 2]!)
    (hnf : (lv.streams.toList.flatMap (·.toList)).map Round.n = zkNs spec sts)
    (hbd : ∀ rd ∈ lv.streams.toList.flatMap (·.toList), 1 ≤ rd.n ∧ rd.n ≤ 65536)
    (hfa : List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
      (lv.streams.toList.flatMap (·.toList)) mids)
    (hmids : mids = (List.range (zkN x)).map fun k => ((dm (k + 1)).tail, c (k + 1)))
    (hc : ∀ k < zkN x, ∀ e : Fin 65536, e.val < zkCnt x spec sts (k + 1) →
      c (k + 1) e = (Claims.coinSeq r' (k + 1) e : 𝔽)) :
    zkFoldS spec.dg (zkS x spec sts hJ h0 dm) (fun i => Claims.coinSeq r' (i + 1)) (zkN x) (fun _ _ => []) =
      fun j q => (lv.streams[2 * j.val + q.val]!).toList.map (Claims.Bridge.Custody.withSha spec) := by
  have e : lv.streams.toList = (List.range (2 * sts.size)).map fun k => lv.streams[k]! := by
    refine List.ext_getElem (by simp [hsz]) fun i h₁ h₂ => ?_
    simp only [Array.length_toList] at h₁
    simp [getElem!_pos lv.streams i h₁]
  have hT2 : ((List.range (2 * sts.size)).flatMap fun w => (lv.streams[w]!).toList.map fun a => (w, a)).map
      Prod.snd = lv.streams.toList.flatMap (·.toList) := by
    rw [e, List.flatMap_map, List.map_flatMap]
    simp only [List.map_map, Function.comp_def, List.map_id']
  have hT1 : ((List.range (2 * sts.size)).flatMap fun w => (lv.streams[w]!).toList.map fun a => (w, a)).map
      Prod.fst = zkStreams spec sts := by
    rw [List.map_flatMap, zkStreams, List.flatMap_def, List.flatMap_def]
    refine congrArg List.flatten (List.map_congr_left fun w hw => ?_)
    have hw' : w < lv.streams.size := by rw [hsz]; exact List.mem_range.1 hw
    simp only [List.map_map, Function.comp_def]
    rw [List.map_const', getElem!_pos lv.streams w hw', ← hns w hw', List.length_map]
  generalize hTdef : (List.range (2 * sts.size)).flatMap (fun w => (lv.streams[w]!).toList.map fun a => (w, a)) =
    T at hT1 hT2
  have hLm : mids.length = zkN x := by rw [hmids, List.length_map, List.length_range]
  have hflen : (lv.streams.toList.flatMap (·.toList)).length = zkN x := hfa.length_eq.trans hLm
  have hTlen : T.length = zkN x := by
    have h := congrArg List.length hT2
    rw [List.length_map] at h
    omega
  have hSlen : (zkStreams spec sts).length = zkN x := by
    have h := congrArg List.length hT1
    rw [List.length_map] at h
    omega
  funext j q
  rw [zkFoldS_eq, List.nil_append]
  have hq2 := q.isLt
  have hsel : ((List.range (zkN x)).filter fun k =>
      decide ((zkS x spec sts hJ h0 dm k).tab = j ∧ (zkS x spec sts hJ h0 dm k).rep = q)) =
      (List.range (zkN x)).filter fun k => decide (zkStream spec sts (k + 1) = 2 * j.val + q.val) := by
    refine List.filter_congr fun k _ => decide_eq_decide.2 ⟨fun h => ?_, fun h => ⟨?_, ?_⟩⟩
    · have h1 := congrArg Fin.val h.1
      have h2 := congrArg Fin.val h.2
      simp only [zkS] at h1 h2
      omega
    · exact Fin.ext (by simp only [zkS]; omega)
    · exact Fin.ext (by simp only [zkS]; omega)
  rw [hsel]
  refine (zk_tagged_eq T (zkN x) (fun k => zkStream spec sts (k + 1)) _ (Claims.Bridge.Custody.withSha spec) hTlen
    ?_ _).trans ?_
  · intro k hk
    have hkN : k < zkN x := by omega
    have hkf : k < (lv.streams.toList.flatMap (·.toList)).length := by omega
    have hkm : k < mids.length := by omega
    have hks : k < (zkStreams spec sts).length := by omega
    have e1 := congrArg (fun l => l[k]?) hT1
    simp only [List.getElem?_map] at e1
    rw [List.getElem?_eq_getElem hk, List.getElem?_eq_getElem hks] at e1
    have e1' : T[k].1 = (zkStreams spec sts)[k] := by simpa using e1
    have e2 := congrArg (fun l => l[k]?) hT2
    simp only [List.getElem?_map] at e2
    rw [List.getElem?_eq_getElem hk, List.getElem?_eq_getElem hkf] at e2
    have e2' : T[k].2 = (lv.streams.toList.flatMap (·.toList))[k] := by simpa using e2
    refine ⟨?_, ?_⟩
    · show T[k].1 = (zkStreams spec sts).getD (k + 1 - 1) 0
      rw [Nat.add_sub_cancel, List.getD_eq_getElem _ _ hks, e1']
    · rw [e2']
      have hR := hfa.get hkf hkm
      simp only [List.get_eq_getElem] at hR
      have hmk : mids[k] = ((dm (k + 1)).tail, c (k + 1)) := by
        simp only [hmids, List.getElem_map, List.getElem_range]
      rw [hmk] at hR
      obtain ⟨hby, hco⟩ := hR
      have hb := hbd _ (List.getElem_mem hkf)
      have hn : (lv.streams.toList.flatMap fun s : Array Round => s.toList)[k].n = (zkNs spec sts).getD k 0 := by
        rw [← hnf, List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hkf), List.getElem_map]
      have hcnt : zkCnt x spec sts (k + 1) = (lv.streams.toList.flatMap fun s : Array Round => s.toList)[k].n := by
        unfold zkCnt
        rw [ite_eq_right (Nat.succ_ne_zero k), ite_eq_left (show k + 1 ≤ zkN x by omega),
          show k + 1 - 1 = k from rfl, ← hn, min_eq_left hb.2, max_eq_right hb.1]
      generalize (lv.streams.toList.flatMap fun s : Array Round => s.toList)[k] = rd at hby hco hb hcnt ⊢
      rcases rd with ⟨n, sha, coins, bytes⟩
      simp only at hby hco hb hcnt
      subst hby hco
      show Claims.Bridge.Custody.withSha spec ⟨n, sha, (c (k + 1)).vals n, some ⟨(dm (k + 1)).tail.toArray⟩⟩ =
        Refine.roundOf spec.dg ⟨(dm (k + 1)).tail.toArray⟩ (zkCnt x spec sts (k + 1))
          (zkTakeE (zkCnt x spec sts (k + 1)) (Claims.coinSeq r' (k + 1)))
      rw [hcnt]
      unfold Refine.roundOf Claims.Bridge.Custody.withSha
      simp only [Option.getD_some, Round.mk.injEq, true_and, and_true]
      exact zk_vals_takeE _ _ hb.2 fun e he => hc k hkN e (by rw [hcnt]; exact he)
  · rw [← hTdef, zk_filter_tagged (fun w => (lv.streams[w]!).toList) (2 * sts.size) (2 * j.val + q.val)]
    split_ifs with hv
    · rfl
    · have hv' : ¬ (2 * j.val + q.val < lv.streams.size) := by omega
      rw [getElem!_neg lv.streams (2 * j.val + q.val) hv']
      rfl

end Gen

/-- **The simulator's rep streams are the record's**: where the record's streams are `2·|sts|`, stream `q` with
`zkRepNsOf spec sts[q / 2]`'s counts, its rounds flattened the session's counts `zkNs`, each of `1` to `65536` coins,
and the round messages `dm (k + 1)` with the coins `c (k + 1)`, which are the live coins on their first `zkCnt`
entries, the simulator's submissions fold to the record's streams. -/
theorem zkSim_reps {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {J : ℕ} (hJ : sts.size ≤ J)
    (h0 : 0 < sts.size) (dm : ℕ → Claims.Msg) (c : ℕ → Coin) (r' : Fin (zkN x + 2) → LCoin 65536) (lv : Live)
    {mids : List (Claims.Msg × Coin)} (hsz : lv.streams.size = 2 * sts.size)
    (hns : ∀ (q : ℕ) (hq : q < lv.streams.size), lv.streams[q].toList.map Round.n = zkRepNsOf spec sts[q / 2]!)
    (hnf : (lv.streams.toList.flatMap (·.toList)).map Round.n = zkNs spec sts)
    (hbd : ∀ rd ∈ lv.streams.toList.flatMap (·.toList), 1 ≤ rd.n ∧ rd.n ≤ 65536)
    (hfa : List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
      (lv.streams.toList.flatMap (·.toList)) mids)
    (hmids : mids = (List.range (zkN x)).map fun k => ((dm (k + 1)).tail, c (k + 1)))
    (hc : ∀ k < zkN x, ∀ e : Fin 65536, e.val < zkCnt x spec sts (k + 1) →
      c (k + 1) e = (Claims.coinSeq r' (k + 1) e : 𝔽)) :
    zkFoldS spec.dg (zkS x spec sts hJ h0 dm) (fun i => Claims.coinSeq r' (i + 1)) (zkN x) (fun _ _ => []) =
      fun j q => (lv.streams[2 * j.val + q.val]!).toList.map (Claims.Bridge.Custody.withSha spec) :=
  Gen.zkSim_reps (S := S₀) (x := x) (spec := spec) (sts := sts) (J := J) (hJ := hJ) (h0 := h0) (dm := dm) (c := c)
    (r' := r') (lv := lv) (mids := mids) (hsz := hsz) (hns := hns) (hnf := hnf) (hbd := hbd) (hfa := hfa)
    (hmids := hmids) (hc := hc)

/-! ## The simulation -/

namespace Gen

variable [S : ZkSetup]

/-- **The simulation** (`ZkSimRun`): where `P`'s run at `zkΦ`'s coins decides and accepts, at a masked statement that
sets up, with two link points of at most `32768` coordinates and at least one, the masked simulator's live outcome is
the record's view: `P`'s roots, the link coin's points, the record's reps and its proof files. -/
theorem zkSimRun {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) {J : ℕ} (hJ : sts.size ≤ J)
    (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) :
    ZkSimRun P x ω (fun _ => J) spec.mPts zkN spec.dg (zkW J spec hw) (zkSim P x spec sts) (zkΦ x spec sts) := by
  intro u r' hacc s spec' sts' link c₁ mids f hact
  have hL := zkN_add hset
  obtain ⟨hc₁, hmids, hfin⟩ := zk_decides_msgs hL hacc hact
  obtain ⟨rest, hhead⟩ := run_head sha512 P (V true) x ω (zkΦ x spec sts (u, r')) hacc
  obtain ⟨_, _, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  have hl : linkOf ((P.next x [] ω).eval sha512).1 = some link := hlink
  obtain ⟨e₁, e₂, e₃, hsz, hns, hnf, hbd, hfa⟩ := zk_decides_record hset hacc hact
  subst e₁ e₂ e₃ hc₁
  rw [zkSim_liveOut P x spec sts ω hJ h0 hw hm1 u r' hL]
  refine liveOutJ_ext' ?_ (zkSim_pts hL hw hp u r' link f) ?_ ?_
  · show ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray
    rw [hl]
    rfl
  · exact zkSim_reps hJ h0 _ _ r' _ hsz hns hnf hbd hfa hmids fun k hk e he =>
      zkΦ_entry hL u r' (by omega) e he
  · show ((finalOf (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') (zkN x))).map
        Final.proofs).getD #[] = f.proofs
    rw [zkB, ite_eq_right (lt_irrefl _), zkAcc, decide_eq_true hacc, ite_eq_left rfl, hfin]
    rfl

end Gen

/-- **The simulation** (`ZkSimRun`): where `P`'s run at `zkΦ`'s coins decides and accepts, at a masked statement that
sets up, with two link points of at most `32768` coordinates and at least one, the masked simulator's live outcome is
the record's view: `P`'s roots, the link coin's points, the record's reps and its proof files. -/
theorem zkSimRun {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) {J : ℕ} (hJ : sts.size ≤ J)
    (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) :
    ZkSimRun P x ω (fun _ => J) spec.mPts zkN spec.dg (zkW J spec hw) (zkSim P x spec sts) (zkΦ x spec sts) :=
  Gen.zkSimRun (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (hset := hset)
    (J := J) (hJ := hJ) (h0 := h0) (hw := hw) (hm1 := hm1) (hp := hp)

namespace Gen

variable [S : ZkSetup]

/-- **The lockstep holds at every statement** (`zkSimRun`, at the statement's link points: `zk_mPts`,
`zk_points`). -/
theorem zkSimRunAt (x : Claims.Msg) : ZkSimRunAt x :=
  fun _ hs hin _ _ _ hset h0 L hz hk hlay _ P ω =>
    zkSimRun P ω hset (zk_tabsJ_le hset h0 L hz hk hlay) h0 (zk_mPts_le hs hin hset) (zk_mPts hs hin hset).2
      (zk_points hs hin hset)

end Gen

/-- **The lockstep holds at every statement** (`zkSimRun`, at the statement's link points: `zk_mPts`,
`zk_points`). -/
theorem zkSimRunAt (x : Claims.Msg) : ZkSimRunAt x :=
  Gen.zkSimRunAt (S := S₀) (x := x)

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and view clashes at the statements some run accepts: its lockstep and abort hold at
every statement (`zkSimRunAt`, `zkSimAbortAt`; `zk_KS_of_parts`). -/
theorem zk_KS_of_fork (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_parts (fun x _ => zkSimRunAt x) (fun x _ => zkSimAbortAt x) hfork hclash

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and view clashes at the statements some run accepts: its lockstep and abort hold at
every statement (`zkSimRunAt`, `zkSimAbortAt`; `zk_KS_of_parts`). -/
theorem zk_KS_of_fork (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_parts (fun x _ => zkSimRunAt x) (fun x _ => zkSimAbortAt x) hfork hclash

end Gen

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
fork bounds and view clashes at the statements some run accepts (`zk_tableBoundJOnAccR_of_parts`): `zk_KSFlockJR`'s
premise. -/
theorem zk_tableBoundJOnAccR_of_fork (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_parts (fun x _ => zkSimRunAt x) (fun x _ => zkSimAbortAt x) hfork hclash

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the fork bounds and view clashes at the statements some run accepts (`zk_KSFlock_of_parts`). -/
theorem zk_KSFlock_of_fork (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlock_of_parts D DO opening hF (fun x _ => zkSimRunAt x) (fun x _ => zkSimAbortAt x) hfork hclash hRef hRounds
    hN hε hRd hS hR hV hT hL hO hAcc hN0

end FlockDeploy.KS
