import Proofs.Flock.Claims.KS.PlainLock
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain simulator's rounds are the record's

Where `P`'s deployed run at `plainΦ`'s coins decides and accepts, the plain simulator's live outcome has the record's
rep streams, round by round, and the last message's proof files (`plainSimReps`, `plainSimProofs`).
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Discharge.Tables FlockSoundness.Discharge.Exec
open Claims.Bridge.Live
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The record's round counts -/

theorem forall_of_forall₂ {α β : Type} {Q : β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ (fun _ b => Q b) l₁ l₂ → ∀ b ∈ l₂, Q b
  | _, _, .nil => fun _ hb => absurd hb List.not_mem_nil
  | _, _, .cons h t => fun b hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h
    · exact forall_of_forall₂ t b hb

theorem check_true {b : Bool} {why : String} {u : Unit} (h : Record.check b why = .ok u) : b = true := by
  unfold Record.check at h
  split at h
  · assumption
  · cases h

/-- **The verifier's decoder reads only rounds of `1` to `65536` coins** (S12), at any `keep`. -/
theorem decodeWithK_n (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∀ x ∈ sess.streams.toList, ∀ rd ∈ x.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  unfold Record.decodeWithK at h
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
    refine forall_of_forall₂ (forall₂_of_array_mapM
      (P := fun _ (b : StreamRec) => ∀ rd ∈ b.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536) (fun s x hx => ?_)
      (by assumption))
    obtain ⟨name, -, hx⟩ := ebind_ok hx
    obtain ⟨rs, -, hx⟩ := ebind_ok hx
    obtain ⟨rounds, hrounds, hx⟩ := ebind_ok hx
    repeat' obtain ⟨_, _, hx⟩ := ebind_ok hx
    cases hx
    refine forall_of_forall₂ (forall₂_of_mapIdxM
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
    have hb := check_true hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
    repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
    all_goals
      cases hr
      exact ⟨hb.1.1.1, hb.1.1.2⟩

theorem decodeWith_n (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decodeWith hs spec text = .ok sess) :
    ∀ x ∈ sess.streams.toList, ∀ rd ∈ x.rounds.toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 :=
  decodeWithK_n id hs h

/-- **An accepted record's live rounds have `1` to `65536` coins**: `liveOf`'s rounds are the decoder's, by coin
count (`liveOf_decodeWithK`). At any `keep` (`verifyWithMK`). -/
theorem liveOf_nK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} {lv : Live} (hv : verifyWithMK keep hs spec tabs record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) (k : ℕ) (hk : k < lv.streams.size) :
    ∀ rd ∈ lv.streams[k].toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 := by
  obtain ⟨sess, -, hdec, -⟩ := verifyWithMK_tables keep hs hv
  obtain ⟨-, -, hsz, hall⟩ := liveOf_decodeWithK keep hs hl hdec
  obtain ⟨stream, hfind, hfa⟩ := hall k (hsz ▸ hk) hk
  have hmem : stream ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hfind)
  have hn := map_n_of_forall₂ (hfa.imp fun _ _ h => ⟨h.1, h.2⟩)
  intro rd hrd
  have : rd.n ∈ stream.rounds.toList.map Round.n := hn ▸ List.mem_map_of_mem hrd
  obtain ⟨rd', hrd', e⟩ := List.mem_map.1 this
  rw [← e]
  exact decodeWithK_n keep hs hdec stream hmem rd' hrd'

theorem liveOf_n (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec tabs record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) (k : ℕ) (hk : k < lv.streams.size) :
    ∀ rd ∈ lv.streams[k].toList, 1 ≤ rd.n ∧ rd.n ≤ 65536 :=
  liveOf_nK id hs hv hl k hk

/-! ## A decision's messages -/

/-- **A decision's round messages are the messages between the first and the last, after their tags**, each with its
coin, and its link coin is the first message's. -/
theorem act_decide_mids {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ a₁ rest crest, as = a₁ :: rest ∧ cs = c₁ :: crest ∧
      mids = (rest.dropLast.map List.tail).zip crest.dropLast := by
  unfold act at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  cases h
  exact ⟨_, _, _, rfl, rfl, rfl⟩

/-- At a statement of one table, the setup's tables are that table. -/
theorem sts_of_tables {zk : Bool} {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    (hset : setupOf zk x = some (ins, spec, sts)) (h : tables zk x = [(spec, st)]) : sts = #[st] := by
  unfold tables at h
  rw [hset] at h
  obtain ⟨a, ha, e⟩ := List.map_eq_singleton_iff.1 h
  cases e
  rcases sts with ⟨l⟩
  simp only [List.toArray] at ha ⊢
  rw [show l = [st] from ha]

/-! ## The simulator's rounds -/

/-- **The rep streams after `N` rounds**: each stream's rounds are the submissions to it, in order. -/
theorem foldS_eq {J : ℕ} (dg : ByteArray → ByteArray) :
    ∀ (N : ℕ) (S : ℕ → SubmitJ J) (R : ℕ → LCoin 65536) (rs : Fin J → Fin 2 → List Round) (j : Fin J) (q : Fin 2),
      foldS dg S R N rs j q = rs j q ++ ((List.range N).filter fun k => decide ((S k).tab = j ∧ (S k).rep = q)).map
        fun k => Refine.roundOf dg (S k).bytes (S k).n (takeE (S k).n (R k))
  | 0, S, R, rs, j, q => by simp [foldS]
  | N + 1, S, R, rs, j, q => by
    rw [foldS, foldS_eq dg N, List.range_succ_eq_map, List.filter_cons, List.filter_map]
    unfold addRoundJ
    by_cases hp : (S 0).tab = j ∧ (S 0).rep = q
    · obtain ⟨h1, h2⟩ := hp
      subst h1 h2
      rw [upd_self]
      simp [Function.comp_def]
    · rw [upd_ne _ _ fun h => hp ⟨h.1.symm, h.2.symm⟩]
      simp [hp, Function.comp_def]

/-- **The simulator's submissions to rep `q`**: rep `0`'s rounds first, then rep `1`'s, `n₀` each. -/
theorem plainS_filter (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (dm : ℕ → Claims.Msg) {n₀ : ℕ}
    (hN : plainN x = n₀ + n₀) (hn₀ : (plainRepNs spec st).length = n₀) (q : Fin 2) :
    ((List.range (plainN x)).filter fun k =>
        decide ((plainS x spec st dm k).tab = 0 ∧ (plainS x spec st dm k).rep = q)) =
      (List.range n₀).map fun i => q.val * n₀ + i := by
  have hrep : ∀ k, ((plainS x spec st dm k).rep).val = if k < n₀ then 0 else 1 := fun k => by
    show plainRep spec st (k + 1) = _
    unfold plainRep
    rw [hn₀, Nat.add_sub_cancel]
  have htab : ∀ k, (plainS x spec st dm k).tab = 0 := fun k => Subsingleton.elim _ _
  rw [hN, List.range_add, List.filter_append, List.filter_map]
  rcases q with ⟨_ | _ | q, hq⟩
  · rw [List.filter_eq_self.2 fun k hk => ?_, List.filter_eq_nil_iff.2 fun k hk => ?_]
    · simp
    · have := hrep (n₀ + k)
      simp only [Function.comp_apply, htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this]
      simp
    · have := hrep k
      simp only [htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this, ite_eq_left (List.mem_range.1 hk)]
  · rw [List.filter_eq_nil_iff.2 fun k hk => ?_, List.filter_eq_self.2 fun k hk => ?_]
    · simp
    · have := hrep (n₀ + k)
      simp only [Function.comp_apply, htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this]
      simp
    · have := hrep k
      simp only [htab, true_and, decide_eq_true_eq, Fin.ext_iff]
      rw [this, ite_eq_left (List.mem_range.1 hk)]
      simp
  · omega

/-- **`plainΦ`'s coin `j` is the live coin `j` on its first `plainCnt j` entries.** -/
theorem plainΦ_entry {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (hL : plainN x + 2 = (V false).k x)
    (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) {j : ℕ} (hj : j < (V false).k x) (e : Fin 65536)
    (he : e.val < plainCnt x spec st j) :
    Claims.coinSeq (plainΦ x spec st (u, r')) j e = (Claims.coinSeq r' j e : 𝔽) := by
  rw [plainΦ_eq hL]
  simp only [Claims.coinSeq, hj, show j < plainN x + 2 by omega, ↓reduceDIte]
  exact ite_eq_left (α := 𝔽) (show e.val < plainNOf x spec st ⟨j, hj⟩ from he)

/-- **A decided run at `plainΦ`'s coins, message by message**: its link coin is the first coin, its round messages are
`P`'s between the first and the last, after their tags, each with its coin, and it decides on `P`'s last message. -/
theorem plain_decides_msgsK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec}
    {st : Stmt} (hL : plainN x + 2 = (V false).k x) {u : plainTape x} {r' : Fin (plainN x + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (hd : PlainDecidesK keep P x ω (plainΦ x spec st) u r' s spec' sts link c₁ mids f) :
    c₁ = Claims.coinSeq (plainΦ x spec st (u, r')) 0 ∧
      mids = (List.range (plainN x)).map (fun k => ((plainDm P x spec st ω u r' (k + 1)).tail,
        Claims.coinSeq (plainΦ x spec st (u, r')) (k + 1))) ∧
      finalOf (plainDm P x spec st ω u r' (plainN x + 1)) = some f := by
  obtain ⟨hacc, hact⟩ := hd
  have hlen := (accLast_VK keep x _ _ (by simp)
    (run_acc_next sha512 P (VK keep) x ω (plainΦ x spec st (u, r')) hacc)).trans (congrFun (VK_k keep) x)
  have has := run_depK keep sha512 P x ω (plainΦ x spec st (u, r'))
  rw [List.length_map] at hlen
  rw [hlen] at has
  rw [has, List.length_map, List.length_range, range_map_ends _ (by omega),
    range_map_ends (Claims.coinSeq (plainΦ x spec st (u, r'))) (by omega)] at hact
  have hfin := act_decide_final hact
  obtain ⟨a₁, rest, crest, h1, h2, h3⟩ := act_decide_mids hact
  injection h1 with _ h1
  injection h2 with h2a h2
  subst h1 h2
  refine ⟨h2a.symm, ?_, ?_⟩
  · rw [h3, List.dropLast_concat, List.dropLast_concat, List.map_map, show (V false).k x - 2 = plainN x by omega]
    rw [List.zip_map']
    rfl
  · rw [← List.cons_append, List.getLast?_concat] at hfin
    rw [show plainN x + 1 = (V false).k x - 1 by omega]
    exact hfin

/-- `plain_decides_msgsK` at the deployed plain verifier (`keep := id`). -/
theorem plain_decides_msgs {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec}
    {st : Stmt} (hL : plainN x + 2 = (V false).k x) {u : plainTape x} {r' : Fin (plainN x + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (hd : PlainDecides P x ω (plainΦ x spec st) u r' s spec' sts link c₁ mids f) :
    c₁ = Claims.coinSeq (plainΦ x spec st (u, r')) 0 ∧
      mids = (List.range (plainN x)).map (fun k => ((plainDm P x spec st ω u r' (k + 1)).tail,
        Claims.coinSeq (plainΦ x spec st (u, r')) (k + 1))) ∧
      finalOf (plainDm P x spec st ω u r' (plainN x + 1)) = some f :=
  plain_decides_msgsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (spec := spec) (st := st) (hL := hL) (u := u) (r' := r') (s := s) (spec' := spec') (sts := sts) (link := link) (c₁ := c₁) (mids := mids) (f := f) (hd := hd)

/-- A statement that sets up has the link message, its rounds and the last message. -/
theorem plainN_add {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) : plainN x + 2 = (V false).k x := by
  have : (V false).k x = 2 + rounds false spec sts := by
    show (match setupOf false x with
      | some (_, spec, sts) => 2 + rounds false spec sts
      | none => 1) = _
    rw [hset]
  unfold plainN
  omega

/-- **A decided run at a plain statement of one table, at its record**: the decision is at the statement's setup, and
the record's two rep streams have the table's coin counts, each of `1` to `65536` coins, with the round messages'
bytes and coins. -/
theorem plain_decides_recordK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts))
    (h : tables false x = [(spec, st)]) {N : ℕ} {τ : Type}
    {Φ : τ × (Fin (N + 2) → LCoin 65536) → Fin ((V false).k x) → Coin} {u : τ} {r' : Fin (N + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hd : PlainDecidesK keep P x ω Φ u r' s spec' sts' link c₁ mids f) :
    ins = s ∧ spec = spec' ∧ sts = sts' ∧ (Claims.Bridge.Custody.liveD spec f.record).streams.size = 2 ∧
      (∀ (q : ℕ) (hq : q < (Claims.Bridge.Custody.liveD spec f.record).streams.size),
        (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList.map Round.n = plainRepNs spec st ∧
          ∀ rd ∈ (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList, 1 ≤ rd.n ∧ rd.n ≤ 65536) ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)) mids := by
  obtain ⟨hacc, hact⟩ := hd
  have hs := act_decide hact
  rw [hset] at hs
  cases hs
  obtain ⟨hv, hc⟩ := (nextK_decide_sha512 keep hact).1 (run_acc_next sha512 P (VK keep) x ω (Φ (u, r')) hacc)
  obtain ⟨-, lv, hl, hcus⟩ := Claims.Bridge.Custody.checks_ok hc
  have hv' : verifyWithMK keep (Hashes.ofSha512 (Claims.hOf sha512) : Hashes Flock.V) spec (tabsOf sts) f.record
      f.proofs = .ok () := hv
  have hD : Claims.Bridge.Custody.liveD spec f.record = lv := by simp [Claims.Bridge.Custody.liveD, hl]
  obtain ⟨hsz, hns⟩ := streams_ns_plainK keep _ hv' hl
  have h1 := sts_of_tables hset h
  subst h1
  rw [hD]
  refine ⟨rfl, rfl, rfl, hsz, fun q hq => ⟨?_, liveOf_nK keep _ hv' hl q hq⟩,
    (Claims.Bridge.Custody.custody_ok hcus).2.2.2⟩
  rw [hns q hq]
  have : q / 2 = 0 := by simp at hsz; omega
  simp [this]

/-- `plain_decides_recordK` at the deployed plain verifier (`keep := id`). -/
theorem plain_decides_record {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts))
    (h : tables false x = [(spec, st)]) {N : ℕ} {τ : Type}
    {Φ : τ × (Fin (N + 2) → LCoin 65536) → Fin ((V false).k x) → Coin} {u : τ} {r' : Fin (N + 2) → LCoin 65536}
    {s : Ins} {spec' : SessionSpec} {sts' : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hd : PlainDecides P x ω Φ u r' s spec' sts' link c₁ mids f) :
    ins = s ∧ spec = spec' ∧ sts = sts' ∧ (Claims.Bridge.Custody.liveD spec f.record).streams.size = 2 ∧
      (∀ (q : ℕ) (hq : q < (Claims.Bridge.Custody.liveD spec f.record).streams.size),
        (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList.map Round.n = plainRepNs spec st ∧
          ∀ rd ∈ (Claims.Bridge.Custody.liveD spec f.record).streams[q].toList, 1 ≤ rd.n ∧ rd.n ≤ 65536) ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        ((Claims.Bridge.Custody.liveD spec f.record).streams.toList.flatMap (·.toList)) mids :=
  plain_decides_recordK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (h := h) (N := N) (τ := τ) (Φ := Φ) (u := u) (r' := r') (s := s) (spec' := spec') (sts' := sts') (link := link) (c₁ := c₁) (mids := mids) (f := f) (hd := hd)

/-! ## The lockstep -/

theorem flat_two {α : Type} (a : Array (Array α)) (h : a.size = 2) :
    a.toList.flatMap (·.toList) = (a[0]'(by omega)).toList ++ (a[1]'(by omega)).toList := by
  rcases a with ⟨_ | ⟨a₀, _ | ⟨a₁, _ | ⟨b, l⟩⟩⟩⟩ <;> simp at h ⊢

/-- **A coin's first `n` values are a live coin's first `n` entries** where the two agree there. -/
theorem vals_takeE (c : Coin) (u : LCoin 65536) {n : ℕ} (hn : n ≤ 65536)
    (h : ∀ e : Fin 65536, e.val < n → c e = (u e : 𝔽)) :
    c.vals n = Array.ofFn fun i : Fin n => (takeE n u i).val := by
  refine Array.ext (by simp [Coin.vals]) fun i h₁ h₂ => ?_
  have hi : i < n := by simpa using h₂
  simp only [Coin.vals, Array.getElem_map, Array.getElem_range, Array.getElem_ofFn, takeE,
    show i < 65536 by omega, ↓reduceDIte, Coin.at]
  exact h ⟨i, by omega⟩ hi

/-- **The simulator's count at a round is the session's** where the session's is a live round's. -/
theorem plainCnt_eq {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} {k m : ℕ} (hk : k < plainN x)
    (e : (plainNsOne spec st).getD k 0 = m) (h1 : 1 ≤ m) (h2 : m ≤ 65536) : plainCnt x spec st (k + 1) = m := by
  unfold plainCnt
  rw [ite_eq_right (Nat.succ_ne_zero k), ite_eq_left (show k + 1 ≤ plainN x by omega),
    show k + 1 - 1 = k from rfl, e, min_eq_left h2, max_eq_right h1]

/-- **The lockstep's rounds** (`PlainSimReps`): where `P`'s run at `plainΦ`'s coins decides and accepts, at a plain
statement of one table, each rep's rounds in the simulator's live outcome are the record's. -/
theorem plainSimRepsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h : tables false x = [(spec, st)])
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) :
    PlainSimRepsK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st) (plainΦ x spec st) := by
  intro u r' s spec' sts' link c₁ mids f hd
  have hL := plainN_add hset
  obtain ⟨-, hmids, -⟩ := plain_decides_msgsK keep hL hd
  obtain ⟨e₁, e₂, e₃, hsz, hstr, hfa⟩ := plain_decides_recordK keep hset h hd
  subst s spec' sts'
  rw [plainSim_liveOutK keep P x spec st ω hw hm1 u r' hL]
  funext j q
  obtain rfl : j = 0 := Subsingleton.elim _ _
  show foldS (plainDg spec) (plainS x spec st (plainDm P x spec st ω u r')) (fun i => Claims.coinSeq r' (i + 1))
      (plainN x) (fun _ _ => []) 0 q = (((Claims.Bridge.Custody.liveD spec f.record).streams[2 * (0 : Fin 1).val +
        q.val]!).toList.map (Claims.Bridge.Custody.withSha spec))
  generalize Claims.Bridge.Custody.liveD spec f.record = lv at hsz hstr hfa ⊢
  have hq : q.val < lv.streams.size := by omega
  have hs0 := hstr 0 (by omega)
  have hs1 := hstr 1 (by omega)
  have hsq := hstr q.val hq
  have hl0 : (lv.streams[0]'(by omega)).toList.length = (plainRepNs spec st).length := by
    rw [← hs0.1, List.length_map]
  have hl1 : (lv.streams[1]'(by omega)).toList.length = (plainRepNs spec st).length := by
    rw [← hs1.1, List.length_map]
  have hlq : (lv.streams[q.val]'hq).toList.length = (plainRepNs spec st).length := by
    rw [← hsq.1, List.length_map]
  have hflat := flat_two lv.streams hsz
  have hlenN : plainN x = (plainRepNs spec st).length + (plainRepNs spec st).length := by
    have := hfa.length_eq
    rw [hflat, hmids, List.length_append, hl0, hl1, List.length_map, List.length_range] at this
    exact this.symm
  rw [foldS_eq, plainS_filter x spec st _ hlenN rfl q, List.nil_append, List.map_map,
    show 2 * (0 : Fin 1).val + q.val = q.val by simp, getElem!_pos lv.streams q.val hq]
  refine List.ext_getElem (by simp [hlq]) fun i h₁ h₂ => ?_
  have hi : i < (plainRepNs spec st).length := by simpa using h₁
  simp only [List.getElem_map, List.getElem_range, Function.comp_apply]
  set k := q.val * (plainRepNs spec st).length + i with hk
  have hkN : k < plainN x := by
    have := q.isLt
    rw [hlenN, hk]
    rcases q with ⟨_ | _ | _, _⟩ <;> simp <;> omega
  have hkf : k < (lv.streams.toList.flatMap (·.toList)).length := by
    rw [hflat, List.length_append, hl0, hl1]
    omega
  have hkm : k < mids.length := by rw [hmids, List.length_map, List.length_range]; exact hkN
  have hR := hfa.get hkf hkm
  simp only [List.get_eq_getElem] at hR
  have hfk : (lv.streams.toList.flatMap (·.toList))[k] = (lv.streams[q.val]'hq).toList[i]'(by omega) := by
    simp only [hflat]
    rcases q with ⟨_ | _ | _, hq2⟩
    · simp only [Nat.zero_mul, Nat.zero_add] at hk
      simp only [hk]
      rw [List.getElem_append_left (by omega)]
    · simp only at hk
      simp only [hk]
      rw [List.getElem_append_right (by omega)]
      have h0s : (lv.streams[0]'(by omega)).size = (plainRepNs spec st).length := by
        rw [← Array.length_toList]; exact hl0
      simp [h0s]
    · omega
  have hmk : mids[k] = ((plainDm P x spec st ω u r' (k + 1)).tail,
      Claims.coinSeq (plainΦ x spec st (u, r')) (k + 1)) := by
    simp only [hmids, List.getElem_map, List.getElem_range]
  rw [hfk, hmk] at hR
  obtain ⟨hby, hco⟩ := hR
  have hn : ((lv.streams[q.val]'hq).toList[i]'(by omega)).n = (plainRepNs spec st)[i] := by
    have := congrArg (fun l => l[i]?) hsq.1
    simp only [List.getElem?_map] at this
    rw [List.getElem?_eq_getElem (by omega), List.getElem?_eq_getElem hi] at this
    simpa using this
  have hbd := hsq.2 ((lv.streams[q.val]'hq).toList[i]'(by omega)) (List.getElem_mem _)
  have hcnt : plainCnt x spec st (k + 1) = (plainRepNs spec st)[i] := by
    refine plainCnt_eq hkN ?_ (hn ▸ hbd.1) (hn ▸ hbd.2)
    unfold plainNsOne
    rw [List.getD_eq_getElem _ _ (by rw [List.length_append]; omega)]
    rcases q with ⟨_ | _ | _, hq2⟩
    · simp only [Nat.zero_mul, Nat.zero_add] at hk
      simp only [hk]
      rw [List.getElem_append_left hi]
    · simp only at hk
      simp only [hk]
      rw [List.getElem_append_right (by omega)]
      simp
    · omega
  generalize hrd : (lv.streams[q.val]'hq).toList[i]'(by omega) = rd at hby hco hn hbd
  rcases rd with ⟨n, sha, coins, bytes⟩
  simp only at hby hco hn hbd
  subst hby hco
  show Refine.roundOf spec.dg ⟨(plainDm P x spec st ω u r' (k + 1)).tail.toArray⟩ (plainCnt x spec st (k + 1))
      (takeE (plainCnt x spec st (k + 1)) (Claims.coinSeq r' (k + 1))) = _
  rw [hcnt, ← hn]
  unfold Refine.roundOf Claims.Bridge.Custody.withSha
  simp only [Option.getD_some, Round.mk.injEq, true_and, and_true]
  exact (vals_takeE _ _ hbd.2 fun e he => plainΦ_entry hL u r' (by omega) e
    (by rw [hcnt, ← hn]; exact he)).symm

/-- `plainSimRepsK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimReps {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h : tables false x = [(spec, st)])
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) :
    PlainSimReps P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st) (plainΦ x spec st) :=
  plainSimRepsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (h := h) (hw := hw) (hm1 := hm1)

/-- **The lockstep's proof files** (`PlainSimProofs`): where `P`'s run at `plainΦ`'s coins decides and accepts, the
simulator's last message is `P`'s, whose proof files the record's verdict read. -/
theorem plainSimProofsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts)) (hw : 2 * spec.mPts ≤ 65536)
    (hm1 : 1 ≤ spec.mPts) :
    PlainSimProofsK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st)
      (plainΦ x spec st) := by
  intro u r' s spec' sts' link c₁ mids f hd
  have hL := plainN_add hset
  obtain ⟨-, -, hfin⟩ := plain_decides_msgsK keep hL hd
  rw [plainSim_liveOutK keep P x spec st ω hw hm1 u r' hL]
  show ((finalOf (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') (plainN x))).map
      Final.proofs).getD #[] = f.proofs
  rw [plainB, ite_eq_right (lt_irrefl _), plainAccK, decide_eq_true hd.1, ite_eq_left rfl, hfin]
  rfl

/-- `plainSimProofsK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimProofs {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts)) (hw : 2 * spec.mPts ≤ 65536)
    (hm1 : 1 ≤ spec.mPts) :
    PlainSimProofs P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st)
      (plainΦ x spec st) :=
  plainSimProofsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (hw := hw) (hm1 := hm1)

/-- **The lockstep's link points** (`PlainSimPts`): where `P`'s run at `plainΦ`'s coins decides, the first live coin's
points are the link coin's, whose `2·mPts` link coordinates are the first live coin's entries. -/
theorem plainSimPtsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts)) (hw : 2 * spec.mPts ≤ 65536)
    (hp : spec.points = 2) :
    PlainSimPtsK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st)
      (plainΦ x spec st) := by
  refine plainSimPts_of_coinsK keep fun u r' s spec' sts' link c₁ mids f hd => ?_
  have hL := plainN_add hset
  obtain ⟨hc₁, -, -⟩ := plain_decides_msgsK keep hL hd
  have hs := act_decide hd.2
  rw [hset] at hs
  simp only [Option.some.injEq, Prod.mk.injEq] at hs
  obtain ⟨-, e, -⟩ := hs
  subst spec' c₁
  rw [ptsIf_proj]
  funext k t
  have hkt : t.val + spec.mPts * k.val < 2 * spec.mPts := by
    have := k.isLt
    have := t.isLt
    rcases k with ⟨_ | _ | _, _⟩ <;> simp <;> omega
  show ptsE spec.mPts (takeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
    Refine.fOf ((pointsOf spec (Claims.coinSeq (plainΦ x spec st (u, r')) 0))[k.val]!)[t.val]!
  have e1 : ptsE spec.mPts (takeE (2 * spec.mPts) (Claims.coinSeq r' 0)) k t =
      takeE (2 * spec.mPts) (Claims.coinSeq r' 0) ⟨t.val + spec.mPts * k.val, hkt⟩ := rfl
  have e2 : k.val * spec.mPts + t.val = t.val + spec.mPts * k.val := by rw [Nat.mul_comm, Nat.add_comm]
  have hk2 : k.val < 2 := k.isLt
  simp only [pointsOf, hp]
  rw [e1, getElem!_pos _ k.val (by simp [hk2]), Array.getElem_map, Array.getElem_range,
    getElem!_pos _ t.val (by simp), Array.getElem_map, Array.getElem_range, e2]
  have he : t.val + spec.mPts * k.val < 65536 := by omega
  simp only [takeE, Coin.at, he, ↓reduceDIte]
  have hc : t.val + spec.mPts * k.val < plainCnt x spec st 0 := by
    unfold plainCnt
    simp only [↓reduceIte]
    exact lt_max_of_lt_right hkt
  exact (plainΦ_entry (j := 0) hL u r' (by omega) ⟨_, he⟩ hc).symm

/-- `plainSimPtsK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimPts {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts)) (hw : 2 * spec.mPts ≤ 65536)
    (hp : spec.points = 2) :
    PlainSimPts P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st)
      (plainΦ x spec st) :=
  plainSimPtsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (hw := hw) (hp := hp)

/-- **The simulation** (`PlainSimRun`): where `P`'s run at `plainΦ`'s coins decides and accepts, at a plain statement
of one table with two link points of at most `32768` coordinates and at least one, the simulator's live outcome is the
record's view: `P`'s roots, the link coin's points, the record's reps and its proof files. -/
theorem plainSimRunK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h : tables false x = [(spec, st)])
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) :
    PlainSimRunK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st) (plainΦ x spec st) :=
  plainSimRun_of_partsK keep (plainSimRoot_of_firstK keep hw (plainSim_firstK keep P x spec st ω)) (plainSimPtsK keep P ω st hset hw hp)
    (plainSimRepsK keep P ω hset h hw hm1) (plainSimProofsK keep P ω st hset hw hm1)

/-- `plainSimRunK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimRun {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {st : Stmt} (hset : setupOf false x = some (ins, spec, sts)) (h : tables false x = [(spec, st)])
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (hp : spec.points = 2) :
    PlainSimRun P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st) (plainΦ x spec st) :=
  plainSimRunK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (h := h) (hw := hw) (hm1 := hm1) (hp := hp)

/-! ## The link points' range -/

/-- **A table's setup passed the range check, with well-formed regions**: `setupH`'s and `setupHidden`'s
`checkInRange` at its `k_log`, regions and link points, and `RegionsWF`. -/
theorem setupOne_range {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Discharge.HiddenExec.setupOne tags cf pf coins tables draw partition program = .ok st) :
    checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok () ∧ Refine.RegionsWF st := by
  unfold Discharge.HiddenExec.setupOne at h
  split at h
  · exact ⟨(Discharge.Hidden.setupHidden_spec' h).inRange, (Discharge.Hidden.setupHidden_wf h).2⟩
  · exact ⟨FlockVBridge.Family.setupH_inRange h, (Refine.setupH_wf h).2⟩

/-- **A plain statement in scope has two link points of between 1 and 64 coordinates**, so its link coin fits a live
coin: table 0's range check and regions, whose link points are the session's. -/
theorem plain_mPts {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) :
    2 * spec.mPts ≤ 65536 ∧ 1 ≤ spec.mPts ∧ spec.points = 2 := by
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, -, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  obtain ⟨h0, hall⟩ := Discharge.HiddenExec.setupTables_setupOne hsr
  obtain ⟨d, hd, -⟩ := hall 0 h0
  obtain ⟨hp2, hpts⟩ := Discharge.HiddenExec.setupTables_pointsAny hsr
  have hpts := hpts 0 h0
  obtain ⟨hr, hwf⟩ := setupOne_range hd
  have h64 := ((checkInRange_ok _ _ _).1 hr).2.2
  have hm := hwf.mPts
  unfold PT_LOCAL at h64 hm
  exact ⟨by omega, by omega, hp2⟩

/-! ## The lockstep at a statement -/

/-- **The lockstep** (`PlainSimReps ∧ PlainSimProofs`): at every plain statement in scope with one table, and every
prover `P` and its tape `ω`, the plain simulator's live outcome has the reps and proof files of each record `P`'s run
decides on at `plainΦ`'s coins, on the plain wire at the statement's link points (`plain_mPts`). -/
def PlainSimLockstep (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    PlainSimReps P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
        (plainSim P x spec st) (plainΦ x spec st) ∧
      PlainSimProofs P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
        (plainSim P x spec st) (plainΦ x spec st)

/-- **The lockstep holds at every statement** (`plainSimReps`, `plainSimProofs`). -/
theorem plainSimLockstep (x : Claims.Msg) : PlainSimLockstep x := fun _ hs hin _ _ _ hset st h _ P ω =>
  ⟨plainSimReps P ω hset h _ (plain_mPts hs hin hset).2.1,
    plainSimProofs P ω st hset _ (plain_mPts hs hin hset).2.1⟩

/-! ## The abort -/

/-- **The verdict at the draw refuses an outcome without proof files**: what `wrPD` writes has the outcome's proof
files wherever `verify` accepts it (`recordsLiveAtD_wrPD`), and `verify` refuses none (`not_acceptsAtD_nil`). -/
theorem not_plainVerdictAt_nil (I : Discharge.Exec.Inputs) (d : Lean.Json) {N mPts : ℕ} (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) {o : LiveOutJ 1 mPts} (h : o.proofs = #[]) : ¬ plainVerdictAt I d C rootB₀ o := by
  rintro ⟨hA, -, -⟩
  have hp := (Claims.Bridge.PlainDraw.recordsLiveAtD_wrPD I d o hA).1
  rw [h] at hp
  generalize Claims.Bridge.PlainDraw.wrPD I d o = t at hA hp
  rcases t with ⟨record, proofs⟩
  subst hp
  exact Claims.Bridge.PlainDraw.not_acceptsAtD_nil I (some d) record hA

/-- **Where `P`'s run at `plainΦ`'s coins is rejected, the simulator's live outcome has no proof files**: its last
message is empty. -/
theorem plainSim_proofs_nilK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts))
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hna : (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out ≠ .acc) :
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
      (ω, u) r').proofs = #[] := by
  rw [plainSim_liveOutK keep P x spec st ω hw hm1 u r' (plainN_add hset)]
  show ((finalOf (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') (plainN x))).map
      Final.proofs).getD #[] = #[]
  rw [plainB, ite_eq_right (lt_irrefl _), plainAccK, decide_eq_false hna]
  rfl

/-- `plainSim_proofs_nilK` at the deployed plain verifier (`keep := id`). -/
theorem plainSim_proofs_nil {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (st : Stmt) (hset : setupOf false x = some (ins, spec, sts))
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hna : (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out ≠ .acc) :
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
      (ω, u) r').proofs = #[] :=
  plainSim_proofs_nilK id (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts) (st := st) (hset := hset) (hw := hw) (hm1 := hm1) (u := u) (r' := r') (hna := hna)

/-- **The abort** (`PlainSimAbort`): at every plain statement in scope with one table, every prover `P` and its tape
`ω`, the plain simulator's live outcome fails the verdict at any draw wherever `P`'s run at `plainΦ`'s coins is
rejected. -/
def PlainSimAbortAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (_ : tables false x = [(spec, st)]) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (I : Discharge.Exec.Inputs)
    (d : Lean.Json) {N : ℕ} (C : Fin N → List 𝔽) (rootB₀ : Array ByteArray),
    PlainSimAbort P x ω spec.mPts plainN (plainDg spec) (plainW spec (plain_mPts hs hin hset).1)
      (plainSim P x spec st) (plainΦ x spec st) (plainVerdictAt I d C rootB₀)

/-- **The abort holds at every statement** (`plainSim_proofs_nil`, `not_plainVerdictAt_nil`). -/
theorem plainSimAbortAt (x : Claims.Msg) : PlainSimAbortAt x :=
  fun _ hs hin _ _ _ hset st _ _ P ω I d _ C rootB₀ u r' hna =>
    not_plainVerdictAt_nil I d C rootB₀
      (plainSim_proofs_nil P ω st hset _ (plain_mPts hs hin hset).2.1 u r' hna)

end FlockDeploy.KS
