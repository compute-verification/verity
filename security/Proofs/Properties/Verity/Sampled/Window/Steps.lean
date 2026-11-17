module

public import Proofs.Properties.Verity.Sampled.Window.Counts
public import Proofs.Properties.Verity.Sampled.Window.Closing
public import Proofs.Properties.Verity.Sampled.MemberForm.Order

/-! SPOT's coupling over a window: the members' steps, from each session's read. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open FlockSoundness.Refine (roundOf ptsOf)
open Claims.Bridge.CustodyZkP (repRounds)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem Member.roundsLog_eq {nJ : ℕ} (m : Member nJ) : m.coinSpec.roundsLog = 9 := rfl

theorem Member.rounds_eq {nJ : ℕ} (m : Member nJ) : m.rounds = 1026 :=
  (congrArg (fun k => 2 * 2 ^ k + 2) m.roundsLog_eq).trans (by decide)

theorem Member.N_eq {nJ : ℕ} (m : Member nJ) : m.N = 1024 :=
  (congrArg (fun k => 2 * 2 ^ k) m.roundsLog_eq).trans (by decide)

theorem Member.toV5_N {nJ : ℕ} (m : Member nJ) (t : ℕ) : (m.toV5 t).N = 1024 := m.N_eq

theorem sum_rounds {nJ : ℕ} : ∀ l : List (Member nJ), (l.map Member.rounds).sum = l.length * 1026
  | [] => by simp
  | m :: l => by
    rw [List.map_cons, List.sum_cons, sum_rounds l, List.length_cons, m.rounds_eq]
    omega

theorem ExecutionPhase.rounds_eq (W : ExecutionPhase) : W.rounds = W.ms.length * 1026 := sum_rounds W.ms

theorem countP_take_zero {γ : Type _} (p : γ → Bool) (l : List γ) : (l.take 0).countP p = 0 := by simp

theorem countP_take_succ_pos {γ : Type _} (p : γ → Bool) {l : List γ} {i : ℕ} {a : γ} (h : l[i]? = some a)
    (hp : p a = true) : (l.take (i + 1)).countP p = (l.take i).countP p + 1 := by
  rw [List.take_add_one, h, List.countP_append]
  simp [hp]

theorem countP_take_succ_neg {γ : Type _} (p : γ → Bool) {l : List γ} {i : ℕ} {a : γ} (h : l[i]? = some a)
    (hp : ¬ p a = true) : (l.take (i + 1)).countP p = (l.take i).countP p := by
  rw [List.take_add_one, h, List.countP_append]
  simp [hp]

theorem countP_take_succ_none {γ : Type _} (p : γ → Bool) {l : List γ} {i : ℕ} (h : l[i]? = none) :
    (l.take (i + 1)).countP p = (l.take i).countP p := by
  rw [List.take_add_one, h]
  simp

theorem countP_take_le {γ : Type _} (p : γ → Bool) (l : List γ) (i : ℕ) : (l.take i).countP p ≤ i :=
  Nat.le_trans List.countP_le_length (by rw [List.length_take]; exact Nat.min_le_left _ _)

theorem countP_take_exists {γ : Type _} (p : γ → Bool) (l : List γ) : ∀ n v, v < (l.take n).countP p →
    ∃ i a, l[i]? = some a ∧ p a = true ∧ (l.take i).countP p = v
  | 0, v, h => by
    rw [countP_take_zero] at h
    exact absurd h (Nat.not_lt_zero _)
  | n + 1, v, h => by
    cases hn : l[n]? with
    | none =>
      rw [countP_take_succ_none p hn] at h
      exact countP_take_exists p l n v h
    | some a =>
      by_cases hd : p a = true
      · rw [countP_take_succ_pos p hn hd] at h
        by_cases hv : v < (l.take n).countP p
        · exact countP_take_exists p l n v hv
        · exact ⟨n, a, hn, hd, by omega⟩
      · rw [countP_take_succ_neg p hn hd] at h
        exact countP_take_exists p l n v h

/-- **The sessions before the member at place `i`**, at the sample `S`: the members before place `i` with a proof
block in `S`. A member with none has no statement and no session, so the session of the member at place `i` is the
play's session after as many. -/
noncomputable def sampledBefore {nJ : ℕ} (ms : List (Member nJ)) (S : Finset (Fin nJ)) (i : ℕ) : ℕ :=
  (ms.take i).countP fun m => @decide (m.part S).Nonempty (Classical.propDecidable _)

theorem sampledBefore_zero {nJ : ℕ} (ms : List (Member nJ)) (S : Finset (Fin nJ)) : sampledBefore ms S 0 = 0 :=
  countP_take_zero _ ms

theorem sampledBefore_succ_of_sampled {nJ : ℕ} {ms : List (Member nJ)} {S : Finset (Fin nJ)} {i : ℕ} {m : Member nJ}
    (h : ms[i]? = some m) (hd : (m.part S).Nonempty) : sampledBefore ms S (i + 1) = sampledBefore ms S i + 1 :=
  countP_take_succ_pos _ h (@decide_eq_true _ (Classical.propDecidable _) hd)

theorem sampledBefore_succ_of_not {nJ : ℕ} {ms : List (Member nJ)} {S : Finset (Fin nJ)} {i : ℕ} {m : Member nJ}
    (h : ms[i]? = some m) (hd : ¬ (m.part S).Nonempty) : sampledBefore ms S (i + 1) = sampledBefore ms S i :=
  countP_take_succ_neg _ h fun hp => hd (@of_decide_eq_true _ (Classical.propDecidable _) hp)

theorem sampledBefore_le {nJ : ℕ} (ms : List (Member nJ)) (S : Finset (Fin nJ)) (i : ℕ) : sampledBefore ms S i ≤ i :=
  countP_take_le _ ms i

theorem sampledBefore_exists {nJ : ℕ} (ms : List (Member nJ)) (S : Finset (Fin nJ)) (n v : ℕ)
    (h : v < sampledBefore ms S n) :
    ∃ i m, ms[i]? = some m ∧ (m.part S).Nonempty ∧ sampledBefore ms S i = v := by
  obtain ⟨i, m, hm, hp, hv⟩ := countP_take_exists _ ms n v h
  exact ⟨i, m, hm, @of_decide_eq_true _ (Classical.propDecidable _) hp, hv⟩

theorem forall₂_right_mem {γ : Type _} {δ : Type _} {Rl : γ → δ → Prop} {l : List γ} {l' : List δ}
    (h : List.Forall₂ Rl l l') {b : δ} (hb : b ∈ l') : ∃ a, a ∈ l ∧ Rl a b := by
  induction h with
  | nil => cases hb
  | cons hab _ ih =>
    rcases List.mem_cons.1 hb with hb | hb
    · rw [hb]
      exact ⟨_, List.mem_cons_self, hab⟩
    · obtain ⟨a, ha, hr⟩ := ih hb
      exact ⟨a, List.mem_cons_of_mem _ ha, hr⟩

abbrev sessionReading (R : ℕ) (tb : String) (ev : ℕ → Exchange → Option Claims.Msg)
    (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) : Reading R :=
  executionPhaseReadingAt #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest

noncomputable def closeFiles {g : Game Exchange} {σ : Strategy g} {R : ℕ}
    (filesDec : Claims.Msg → Array ByteArray) (q : Pos g σ Claims.Msg (Fin R)) : Array ByteArray :=
  (((List.finRange R).findSome? fun j => if j.val % 1026 + 1 = 1026 then q.toStop.modes j else none).map
    filesDec).getD #[]

noncomputable def sessionDecoders {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Reg : Type}
    (reg : Option Claims.Msg → Reg) (tb : String) (filesDec : Claims.Msg → Array ByteArray) :
    PosDecoders g σ R Reg where
  reg := reg
  root := (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026)).root
  sub := (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026)).sub
  files := closeFiles filesDec

theorem closeFiles_eq {g : Game Exchange} {σ : Strategy g} {R : ℕ} (filesDec : Claims.Msg → Array ByteArray)
    {q : Pos g σ Claims.Msg (Fin R)} {jc : Fin R} {w : Claims.Msg} (hjc : jc.val % 1026 + 1 = 1026)
    (hw : q.toStop.modes jc = some w) (hs : ∀ j, j ≠ jc → q.toStop.modes j = none) :
    closeFiles filesDec q = filesDec w := by
  have hf : ∀ j : Fin R, j ≠ jc → (if j.val % 1026 + 1 = 1026 then q.toStop.modes j else none) = none :=
    fun j hj => by
      rw [hs j hj]
      exact ite_self _
  have h1 : (List.finRange R).findSome? (fun j => if j.val % 1026 + 1 = 1026 then q.toStop.modes j else none) =
      some w := (findSome?_of_single hf (List.mem_finRange jc)).trans ((ite_eq_left hjc).trans hw)
  show (((List.finRange R).findSome? fun j => if j.val % 1026 + 1 = 1026 then q.toStop.modes j else none).map
    filesDec).getD #[] = filesDec w
  rw [h1]
  rfl

structure MembersAt (W : ExecutionPhase) (tb : String) : Prop where
  count : ∀ m ∈ W.ms, m.I.count = 1
  rows : ∀ m ∈ W.ms, m.I.tags.hm96Rows = true
  table : ∀ m ∈ W.ms, m.I.tags.table = tb

structure SessionSpecAt (x : Exchange) (tb : String) (v : ℕ) (text : String) (spec : Flock.SessionSpec)
    (tabs : Array Flock.Setup) (proofs : Array ByteArray) (sess : Flock.RunRecord) (J : Json) (w₀ : Claims.Msg) :
    Prop where
  sent : sentRecordAt v x.sent = some text
  read : SessionRead x.log #[tb] v text J sess w₀
  decode : Flock.Record.decode spec text = .ok sess
  verify : Flock.Zk.verify spec tabs text proofs = .ok ()
  tables : spec.tables = #[tb]
  sigma : spec.sigma.size = 64
  retain : spec.retainRounds = true
  points : spec.points = 2
  sha512 : spec.sha512Digests = true
  dg : spec.dg = Flock.Sha512.hash

structure ExecutionPhaseSessions (W : ExecutionPhase) (tb : String) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) (S : Finset (Fin W.nJ)) (x : Exchange) : Prop where
  specs : ∀ sp ∈ helloSpecs x.log, sp.streams.toList.Nodup
  order : OpensLast x.log
  trees : (openedTrees x.log).length ≤ sampledBefore W.ms S W.ms.length
  session : ∀ (i : ℕ) (m : Member W.nJ), W.ms[i]? = some m → (m.part S).Nonempty →
    ∃ (a : StepAt) (bs : _root_.Core.Blobs) (keys : Array String) (s : _root_.Spot.OneStageZk.RunFiles)
      (record : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json)
      (proofs : Array ByteArray) (units : Array ℕ) (J : Json) (sess : Flock.RunRecord) (w : Claims.Msg) (τ : ℕ),
      StatesMember a m ∧ SessionVerdict a bs keys s record spec tabs rest proofs units ∧
      Exec.recordDraw record = some (printedPart (m.part S)) ∧
      sentRecordAt (sampledBefore W.ms S i) x.sent = some record ∧
      Flock.Record.decode spec record = .ok sess ∧
      SessionRead x.log #[tb] (sampledBefore W.ms S i) record J sess w ∧
      (openedTrees x.log)[sampledBefore W.ms S i]? = some τ ∧
      ev (sampledBefore W.ms S i) x = some w ∧ filesDec w = proofs

theorem ExecutionPhaseSessions.specAt {W : ExecutionPhase} {tb : String} {ev : ℕ → Exchange → Option Claims.Msg}
    {filesDec : Claims.Msg → Array ByteArray} {S : Finset (Fin W.nJ)} {x : Exchange}
    (h : ExecutionPhaseSessions W tb ev filesDec S x) (hMA : MembersAt W tb) {i : ℕ} {m : Member W.nJ}
    (hm : W.ms[i]? = some m) (hd : (m.part S).Nonempty) :
    ∃ (a : StepAt) (bs : _root_.Core.Blobs) (keys : Array String) (s : _root_.Spot.OneStageZk.RunFiles)
      (record : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json)
      (proofs : Array ByteArray) (units : Array ℕ) (J : Json) (sess : Flock.RunRecord) (w : Claims.Msg) (τ : ℕ)
      (sts : Array Flock.Stmt),
      StatesMember a m ∧ SessionVerdict a bs keys s record spec tabs rest proofs units ∧
      Exec.recordDraw record = some (printedPart (m.part S)) ∧
      Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts) ∧
      SessionSpecAt x tb (sampledBefore W.ms S i) record spec tabs proofs sess J w ∧
      (openedTrees x.log)[sampledBefore W.ms S i]? = some τ ∧ ev (sampledBefore W.ms S i) x = some w ∧
      filesDec w = proofs := by
  obtain ⟨a, bs, keys, s, record, spec, tabs, rest, proofs, units, J, sess, w, τ, hst, hv, hdr, hsent, hdec, hread,
    hτ, hev, hfd⟩ := h.session i m hm hd
  obtain ⟨sts, hs, -⟩ := hv.accepts_member hst
  have hmem := List.mem_of_getElem? hm
  obtain ⟨htab, hsig, hret, hpts, h512, hdg⟩ := m.spec_at (hMA.count m hmem) (hMA.rows m hmem) hs
  exact ⟨a, bs, keys, s, record, spec, tabs, rest, proofs, units, J, sess, w, τ, sts, hst, hv, hdr, hs,
    ⟨hsent, hread, hdec, hv.verify, htab.trans (congrArg (fun t => #[t]) (hMA.table m hmem)), hsig, hret, hpts,
      h512, hdg⟩, hτ, hev, hfd⟩

theorem ExecutionPhaseSessions.all {W : ExecutionPhase} {tb : String} {ev : ℕ → Exchange → Option Claims.Msg}
    {filesDec : Claims.Msg → Array ByteArray} {S : Finset (Fin W.nJ)} {x : Exchange}
    (h : ExecutionPhaseSessions W tb ev filesDec S x) (hMA : MembersAt W tb) :
    ∀ v τ, (openedTrees x.log)[v]? = some τ → ∃ (text : String) (spec : Flock.SessionSpec)
      (tabs : Array Flock.Setup) (proofs : Array ByteArray) (sess : Flock.RunRecord) (J : Json) (w₀ : Claims.Msg),
      SessionSpecAt x tb v text spec tabs proofs sess J w₀ := by
  intro v τ hτ
  have hv : v < sampledBefore W.ms S W.ms.length := lt_of_lt_of_le (List.getElem?_eq_some_iff.1 hτ).1 h.trees
  obtain ⟨i, m, hm, hd, hdv⟩ := sampledBefore_exists W.ms S W.ms.length v hv
  obtain ⟨_, _, _, _, record, spec, tabs, _, proofs, _, J, sess, w, _, _, -, -, -, -, hA, -, -, -⟩ :=
    h.specAt hMA hm hd
  rw [hdv] at hA
  exact ⟨record, spec, tabs, proofs, sess, J, w, hA⟩

theorem link_stop {g : Game Exchange} {σ : Strategy g} {R : ℕ} {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {sg : Signer} {θ : ByteArray} {F : Array ByteArray}
    {c rq : ByteArray} (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) {pre : List Release} {f : List Release → Game Exchange}
    {a : Fin R → Prop} {done : List (List Release)} (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log))
    (hSO : StopOf P rv pre f a q) {z : CoinTape g σ} (hz : ∃ w, z = q.back w)
    (hW : Walked P (CoinTape.run g σ z).log pre a done) {su : Fin R → ℕ} {lk : Fin R → Prop}
    (hSP : SessionPicks P su lk (CoinTape.run g σ z).log) {u : ℕ}
    (hDn : SessDone P su (CoinTape.run g σ z).log done u)
    (hul : u < (openedTrees (CoinTape.run g σ z).log).length) {tables : Array String} {text : String} {J : Json}
    {spec : Flock.SessionSpec} {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hR : SessionRead (CoinTape.run g σ z).log tables u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hp2 : spec.points = 2) :
    (¬ ∀ j, q.Ks j = .skip) ∧ ∀ o' : OpeningAt (finish sg θ F c rq) q.g', o'.n = spec.mPts + spec.mPts := by
  have hU := hSP.once
  obtain ⟨L, hNP, ⟨-, hlog⟩ | ⟨i, s, n, b, bits, L', f', ⟨j, haj, hjK, hj⟩, -, hlog, -, hcnt, -, -⟩⟩ :=
    stop_frame_through hP hPS hi hf hSO hz
  · refine (hDn.not_all hSP hul fun j L₁ hp => ?_).elim
    refine hW.seg hU hNP (L₃ := []) (by rw [List.append_nil]; exact hlog) hp ?_
    obtain ⟨r, L₂, he, -⟩ := hp
    have hlen := congrArg List.length (he.symm.trans hlog)
    simp only [List.length_append, List.length_cons] at hlen
    omega
  · have hps : PicksIn P (CoinTape.run g σ z).log j (pre ++ L) := ⟨_, L', hlog, hj⟩
    have hF : ∀ j' L₁, PicksIn P (CoinTape.run g σ z).log j' L₁ → L₁.length < (pre ++ L).length → L₁ ∈ done :=
      fun j' L₁ hp hl => hW.seg hU hNP (L₃ := ⟨.opening i s n, b, bits⟩ :: L')
        (by rw [hlog, List.append_assoc]) hp (by
          rw [List.length_append] at hl
          exact hl)
    have hlt := (hW.next hU hNP (r := ⟨.opening i s n, b, bits⟩) hlog haj hj).1
    obtain ⟨hl, hsu⟩ := hDn.first_stop hSP hps hF hlt
    obtain ⟨x, L₂, i', s', n', he, hx, hni, hc⟩ := hSP.link j _ hl hps
    have hxx : x = ⟨.opening i s n, b, bits⟩ := (List.cons.inj (List.append_cancel_left (he.symm.trans hlog))).1
    have hx' := hx
    rw [hxx] at hx'
    have hn : n = n' := (Frame.opening.inj hx').2.2
    refine ⟨fun hK => ?_, fun o' => (hcnt o').trans (hn.trans ?_)⟩
    · have h1 := hjK
      rw [hK j] at h1
      cases h1
    · exact hR.link_n hd hp2 he hx hni (hc.trans hsu)

theorem answered_count {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R J : ℕ}
    {Reg : Type} (reg : Option Claims.Msg → Reg) (hJ : J ≤ 1) {tb : String}
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = sessionReading R tb ev record₀ recordDigest) {σ : Strategy (V.game θ F c)}
    {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)} (hi : q.Inv (roundCuts V θ F c rd)) {pre : List Release}
    {f : List Release → Game Exchange} {a : Fin R → Prop}
    (hf : ∀ log, f log = finish (V.signer θ) θ F c σ.1 (pre ++ log)) (hSO : StopOf rd.round rd.reveal pre f a q)
    {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y) (hn : NoMiss V θ F c rd σ z) {s : SubmitJ J}
    (hs : (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026)).sub J q.modes =
      some s)
    (hall : ∀ v τ, (openedTrees (CoinTape.run (V.game θ F c) σ z).log)[v]? = some τ →
      ∃ (text : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (proofs : Array ByteArray)
        (sess : Flock.RunRecord) (J' : Json) (w₀ : Claims.Msg),
        SessionSpecAt (CoinTape.run (V.game θ F c) σ z) tb v text spec tabs proofs sess J' w₀) :
    ∀ o' : OpeningAt (finish (V.signer θ) θ F c σ.1) q.g', o'.n = s.n := by
  subst hrd
  have hP := executionPhaseReadingAt_picks (R := R) #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest
  have hPS := executionPhaseReadingAt_served (R := R) #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest
  obtain ⟨L, -, ⟨hK, -⟩ | ⟨i, s₁, n, b, bits, L', f', ⟨j, haj, hjK, hj⟩, hallK, hlog, hcut, hcnt, -, -⟩⟩ :=
    stop_frame_through hP hPS hi hf hSO hz
  · have hm : q.modes = fun _ => none := funext fun j =>
      (congrArg (fun K => DigestCut.modeHere K q.s') (hK j)).trans rfl
    rw [hm, sub_eq_none reg _ _ (fun _ _ => rfl)] at hs
    cases hs
  · have hLA : LoggedAfter (sessionReading R tb ev record₀ recordDigest).round q (pre ++ L) bits
        (CoinTape.run (V.game θ F c) σ z).log :=
      ⟨j, i, s₁, n, b, L', hjK, fun j' h' => ((hallK j').1 h').2, hj, hlog⟩
    obtain ⟨ht', ρ, k, w, j₁, i₁, s₂, n₁, b₁, L₁', hLj, hj₁K, hw, hsa, _, hu', hname, hk, he, _⟩ :=
      LoggedAfter.session_slot reg (by simp) hJ (by omega) (sessionReveal #[tb] ev) record₀ recordDigest hs hLA
    have hrel := (List.cons.inj (List.append_cancel_left (he.symm.trans hlog))).1
    have hn1 : n₁ = n := (Frame.opening.inj (congrArg Release.frame hrel)).2.2
    have hτ : (openedTrees (CoinTape.run (V.game θ F c) σ z).log)[j₁.val / 1026]? = some i₁ := by
      rw [he]
      exact openedTrees_getElem?_append hu' _
    obtain ⟨text, spec, tabs, proofs, sess, J', w₀, hA⟩ := hall _ _ hτ
    have hname' : ((helloSpecs (pre ++ L))[i₁]?.bind fun sp => sp.streams[s₂]?) =
        some (repStreamName tb ρ.val) := hname
    have hcount := session_stop_count 1026 ev record₀ recordDigest rfl hi hz hn hLj rfl hj₁K hw he hcut hu' hk
      hname' hA.sent hA.read hA.decode hA.verify hA.retain hA.sha512 hA.tables
    intro o'
    exact (hcnt o').trans (hn1.symm.trans (hcount.trans (congrArg SubmitJ.n hsa).symm))

theorem session_step {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R nJ : ℕ} {Ω : Type}
    (reg : Option Claims.Msg → ByteArray) {tb : String} (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = sessionReading R tb ev record₀ recordDigest) {σ : Strategy (V.game θ F c)}
    {M : ExecutionPhaseCoinMap (V.game θ F c) σ Ω R}
    (hML : ∀ i (p : Pos (V.game θ F c) σ Claims.Msg (Fin R)) mPts (pts : Fin 2 → Fin mPts → 𝔽),
      M.link i p mPts pts = writeOpening (finish (V.signer θ) θ F c σ.1) p.s' (linkBits fieldEnc pts))
    (hMR : ∀ i (p : Pos (V.game θ F c) σ Claims.Msg (Fin R)) J (s : SubmitJ J) (cs : Fin s.n → 𝔽),
      M.round i p J s cs = writeOpening (finish (V.signer θ) θ F c σ.1) p.s' (roundBits fieldEnc cs))
    {S : Finset (Fin nJ)} {Rg : ByteArray} {i : ℕ} {m : Member nJ} (t : ℕ)
    {p q r : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg (m.toV5 t)) (memberPts (m.toV5 t))}
    (hmr : MemberReach (fun _ => sessionDecoders reg tb filesDec) M
      (fun _ _ => linkEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1)) S Rg i (m.toV5 t) p q out)
    (hqr : q.Past r) (hpi : p.Inv (roundCuts V θ F c rd)) (hn : NoMiss V θ F c rd σ r.tape)
    {pre : List Release} {f : List Release → Game Exchange} {a : Fin R → Prop} {done : List (List Release)}
    (hf : ∀ log, f log = finish (V.signer θ) θ F c σ.1 (pre ++ log))
    (hSO : StopOf rd.round rd.reveal pre f a p.toStop)
    (hW : Walked rd.round (CoinTape.run (V.game θ F c) σ r.tape).log pre a done) {u : ℕ}
    (hDn : SessDone rd.round (fun j => j.val / 1026) (CoinTape.run (V.game θ F c) σ r.tape).log done u)
    (hspecs : ∀ sp ∈ helloSpecs (CoinTape.run (V.game θ F c) σ r.tape).log, sp.streams.toList.Nodup)
    (hO : OpensLast (CoinTape.run (V.game θ F c) σ r.tape).log)
    (hTr : (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log).length * 1026 ≤ R)
    (hall : ∀ v τ, (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log)[v]? = some τ →
      ∃ (text : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (proofs : Array ByteArray)
        (sess : Flock.RunRecord) (J' : Json) (w₀ : Claims.Msg),
        SessionSpecAt (CoinTape.run (V.game θ F c) σ r.tape) tb v text spec tabs proofs sess J' w₀)
    {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {J : Json} {w : Claims.Msg} {τ : ℕ}
    (hA : SessionSpecAt (CoinTape.run (V.game θ F c) σ r.tape) tb u record spec tabs proofs sess J w)
    (hτ : (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log)[u]? = some τ)
    (hev : ev u (CoinTape.run (V.game θ F c) σ r.tape) = some w) (hfd : filesDec w = proofs)
    (hmP : spec.mPts = memberPts (m.toV5 t)) :
    (∃ (pre'' : List Release) (f'' : List Release → Game Exchange) (a'' : Fin R → Prop)
        (done'' : List (List Release)), (∀ log, f'' log = finish (V.signer θ) θ F c σ.1 (pre'' ++ log)) ∧
        StopOf rd.round rd.reveal pre'' f'' a'' q.toStop.consume.toStop ∧
        Walked rd.round (CoinTape.run (V.game θ F c) σ r.tape).log pre'' a'' done'' ∧
        SessDone rd.round (fun j => j.val / 1026) (CoinTape.run (V.game θ F c) σ r.tape).log done'' (u + 1)) ∧
      out.rootB = sess.rootB ∧ out.pts = ptsOf _ sess.points ∧
      (∀ j ρ, out.reps j ρ = repRounds spec sess j.val ρ.val) ∧ out.proofs = proofs := by
  subst hrd
  have hP := executionPhaseReadingAt_picks (R := R) #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest
  have hPS := executionPhaseReadingAt_served (R := R) #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest
  have hSP : SessionPicks (sessionReading R tb ev record₀ recordDigest).round (fun j => j.val / 1026)
      (fun j => slotOf (#[tb] : Array String).size 1026 (j.val % 1026) = .link)
      (CoinTape.run (V.game θ F c) σ r.tape).log :=
    sessionPicks_executionPhaseReadingAt (by simp) (one_table_pos tb) (by omega) (sessionReveal #[tb] ev) record₀
      recordDigest hspecs (logServed_run V θ F c σ r.tape) hTr
  have hU := hSP.once
  have hpi' := (Pos.toStop_spec hpi).2.2
  have hzp := (hmr.past.trans hqr).through
  have hul : u < (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log).length :=
    (List.getElem?_eq_some_iff.1 hτ).1
  have hu1 : (u + 1) * 1026 ≤ R := (Nat.mul_le_mul_right 1026 (Nat.succ_le_of_lt hul)).trans hTr
  have hJ : memberTabs S Rg (m.toV5 t) ≤ 1 := (m.tabs_eq t S Rg).le
  obtain ⟨hK, hcntL⟩ := link_stop hP hPS hpi' hf hSO hzp hW hSP hDn hul hA.read hA.decode hA.points
  have hmr' : MemberReach (fun _ => sessionDecoders reg tb filesDec) M
      (fun _ _ => linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) S Rg i (m.toV5 t) p q out := by
    refine hmr.weaken (fun q' => q'.Inv (roundCuts V θ F c (sessionReading R tb ev record₀ recordDigest)) ∧
        StopsAt (sessionReading R tb ev record₀ recordDigest).round
          (sessionReading R tb ev record₀ recordDigest).reveal (V.signer θ) θ F c σ.1 q'.toStop)
      (fun q' φ hq => ⟨Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hq.1).2.2 φ), StopsAt.next hP hq.2 φ⟩)
      (fun pts hL0 _ hE => ?_)
      (fun pts => ⟨Pos.inv_consume (Pos.inv_rewrite hpi' _), StopsAt.next hP ⟨pre, f, a, hf, hSO⟩ _⟩)
      (fun p' s cs hQ hs hB0 hLe hE => ?_)
    · obtain ⟨o', ho', hne⟩ := hE
      exact hL0 ⟨o', ho', (hcntL o').trans (congrArg (fun n => n + n) hmP), hne⟩
    · obtain ⟨o', ho', hne⟩ := hE
      obtain ⟨pre₁, f₁, a₁, hf₁, hSO₁⟩ := hQ.2
      have hz' := (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans (hLe.trans hqr.le)).through
      exact hB0 ⟨o', ho', answered_count reg hJ ev record₀ recordDigest rfl (Pos.toStop_spec hQ.1).2.2 hf₁ hSO₁
        hz' hn hs hall o', hne⟩
  have hsub : (sessionDecoders (g := V.game θ F c) (σ := σ) (R := R) reg tb filesDec).sub
      (memberTabs S Rg (m.toV5 t)) (fun _ => none) = none :=
    sub_eq_none reg (fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026)) _ (fun _ _ => rfl)
  obtain ⟨pts, L₀, as, ms, pre', f', a', done', hout, ⟨hLA₀, hcut₀⟩, hF, hI, hlt₀, hgt, hpw, hf', hSO', hd', hW',
    hend⟩ := MemberReach.order hP hPS hsub (linkBits fieldEnc) (fun q' pts' => hML i q' _ pts') (fun _ _ h => h)
      (fun _ cs => roundBits fieldEnc cs) (fun q' s cs => hMR i q' _ s cs) (fun _ _ _ h => h) hU rfl hmr' hqr
      hpi hf hSO hW hK
  obtain ⟨j₀, i₀, s₀, n₀, b₀, Lr₀, -, -, hj₀P, hlog₀⟩ := id hLA₀
  have h₀ : PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j₀ L₀ := ⟨_, Lr₀, hlog₀, hj₀P⟩
  have hL₀d : L₀ ∈ done' := (hd' L₀).2 (Or.inl rfl)
  have hbefore : ∀ m' ∈ done', ∀ (j : Fin R) L₁, PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j L₁ → L₁.length < m'.length →
        L₁ = L₀ ∨ L₁ ∈ ms ∨ L₁ ∈ done :=
    fun m' hm' j L₁ hp hl => (hd' L₁).1 (hW'.before j L₁ hp (lt_of_lt_of_le hl (hW'.picked m' hm').1))
  have hF₀ : ∀ (j : Fin R) L₁, PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j L₁ → L₁.length < L₀.length → L₁ ∈ done :=
    fun j L₁ hp hl => by
      rcases hbefore L₀ hL₀d j L₁ hp hl with h | h | h
      · rw [h] at hl
        exact absurd hl (Nat.lt_irrefl _)
      · exact absurd (hgt L₁ h) (Nat.lt_asymm hl)
      · exact h
  have hlu₀ := hDn.first_stop hSP h₀ hF₀ hlt₀
  have hl₀ : slotOf (#[tb] : Array String).size 1026 (j₀.val % 1026) = .link := hlu₀.1
  have hu₀ : j₀.val / 1026 = u := hlu₀.2
  have hL₀' : ∀ j : Fin R, PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j L₀ →
        slotOf (#[tb] : Array String).size 1026 (j.val % 1026) = .link ∧ j.val / 1026 = u := fun j hp => by
    rw [hSP.unique j j₀ L₀ hp h₀]
    exact ⟨hl₀, hu₀⟩
  have hroot := session_rootB reg (per := 1026) (by omega) ev record₀ recordDigest rfl hpi' hzp hn hLA₀ hcut₀
    hL₀' hA.sent hA.read hA.decode hA.tables hA.sigma
  have hpts : ptsOf (memberPts (m.toV5 t)) sess.points = pts := by
    rcases Nat.eq_zero_or_pos spec.mPts with h0 | hpos
    · funext k b
      exact absurd b.isLt (by omega)
    · exact session_pts (by simp) (one_table_pos tb) (by omega) (sessionReveal #[tb] ev) record₀ recordDigest
        hLA₀ hL₀' hA.read hA.decode hA.points hpos
  have hms0 : ∀ m' ∈ ms, ∃ (j : Fin R) (ρ : Fin 2) (k : ℕ),
      slotOf (#[tb] : Array String).size 1026 (j.val % 1026) = .rep 0 ρ k ∧
        PicksIn (sessionReading R tb ev record₀ recordDigest).round
          (CoinTape.run (V.game θ F c) σ r.tape).log j m' := fun m' hm' => by
    obtain ⟨a₁, _, hsub₁, hLA₁, _⟩ := forall₂_right_mem hF hm'
    obtain ⟨_, ρ, k, _, j, i₁, s₁, n₁, b₁, L₁', hLj, _, _, _, _, _, _, _, he, hPj⟩ :=
      LoggedAfter.session_slot reg (by simp) hJ (by omega) (sessionReveal #[tb] ev) record₀ recordDigest hsub₁
        hLA₁
    exact ⟨j, ρ, k, hLj, _, L₁', he, hPj⟩
  have hms : ∀ m' ∈ ms, ∃ j : Fin R, ¬ slotOf (#[tb] : Array String).size 1026 (j.val % 1026) = .link ∧
      PicksIn (sessionReading R tb ev record₀ recordDigest).round
        (CoinTape.run (V.game θ F c) σ r.tape).log j m' := fun m' hm' => by
    obtain ⟨j, ρ, k, hLj, hp⟩ := hms0 m' hm'
    refine ⟨j, fun h => ?_, hp⟩
    rw [hLj] at h
    cases h
  have hrep : ∀ m' ∈ ms, ∀ j : Fin R, PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j m' → j.val / 1026 = u := fun m' hm' j hj =>
    (hDn.rep_stop hSP h₀ hu₀ hms hlt₀ hgt hm' hj (hbefore m' ((hd' m').2 (Or.inr (Or.inl hm'))))).2
  obtain ⟨Lc, xc, Lc₂, sc, nc, hLc, hxc, hτc, hnamec, hopc, hshac, hlastc⟩ := hA.read.close τ hτ
  obtain ⟨jc, hjcu, hjcs⟩ :=
    executionPhase_close (#[tb] : Array String).size (R := R) (per := 1026) (u := u) (by omega) hu1
  have hjcc : jc.val % 1026 + 1 = 1026 := (slotOf_eq_close (one_table_pos tb)).1 hjcs
  have hPc : (sessionReading R tb ev record₀ recordDigest).round jc Lc xc.frame := by
    show sessionRound #[tb] (jc.val / 1026) (slotOf (#[tb] : Array String).size 1026 (jc.val % 1026)) Lc xc.frame
    rw [hjcu, hjcs]
    exact ⟨τ, sc, nc, hxc, logServed_run V θ F c σ r.tape Lc xc Lc₂ τ sc nc hLc hxc, hτc,
      hnamec.trans (slotStream_close #[tb]).symm, hopc⟩
  have hc : PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log jc Lc := ⟨xc, Lc₂, hLc, hPc⟩
  have hcl : ¬ slotOf (#[tb] : Array String).size 1026 (jc.val % 1026) = .link := fun h =>
    closeSlot_ne_link _ (hjcs.symm.trans h)
  have hcd : Lc ∉ done' := fun hmem => by
    rcases (hd' Lc).1 hmem with h | h | h
    · rw [h] at hc
      exact hcl (hL₀' jc hc).1
    · obtain ⟨j, ρ, k, hLj, hp⟩ := hms0 Lc h
      rw [hSP.unique j jc Lc hp hc, hjcs] at hLj
      have h' : Slot.rep (#[tb] : Array String).size 0 1 = .rep 0 ρ k := hLj
      rw [Slot.rep.injEq] at h'
      exact absurd h'.1 (Nat.ne_of_gt (one_table_pos tb))
    · have h1 : jc.val / 1026 < u := (hDn jc Lc hc).1 h
      omega
  have hi_q := (Pos.toStop_spec (hmr.inv hpi)).2.2
  have hzq := hqr.through
  obtain ⟨L, hNP, ⟨-, hlog⟩ | ⟨ic, s, n, b, bits, L', f'', ⟨j, haj, hjK, hj⟩, hallq, hlog, hcutq, -, hf'', hnext⟩⟩ :=
    stop_frame_through hP hPS hi_q hf' hSO' hzq
  · exfalso
    obtain ⟨x, L'', j', hLg', -⟩ := hDn.close_stop hSP hO h₀ hl₀ hu₀
      (fun L₁ h => (hd' L₁).2 (Or.inr (Or.inr h))) hL₀d hW' hc hcl hjcu hcd hNP (Or.inl hlog)
    have hlen := congrArg List.length (hlog.symm.trans hLg')
    simp only [List.length_append, List.length_cons] at hlen
    omega
  obtain ⟨x, L'', j', hLg', haj', hPj', hcase⟩ := hDn.close_stop hSP hO h₀ hl₀ hu₀
    (fun L₁ h => (hd' L₁).2 (Or.inr (Or.inr h))) hL₀d hW' hc hcl hjcu hcd hNP
    (Or.inr ⟨⟨.opening ic s n, b, bits⟩, L', j, hlog, haj, hj⟩)
  have hxe : x = ⟨.opening ic s n, b, bits⟩ := (List.cons.inj (List.append_cancel_left (hLg'.symm.trans hlog))).1
  rw [hxe] at hPj'
  have hps' : PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j' (pre' ++ L) := ⟨_, L', hlog, hPj'⟩
  have hjK' : (q.toStop.Ks j').isHere = true := (hallq j').2 ⟨haj', hPj'⟩
  have hsingle := executionPhaseStop_modes (by simp) (one_table_pos tb) (by omega) (sessionReveal #[tb] ev) record₀
    recordDigest (fun j'' h'' => ((hallq j'').1 h'').2) ((hallq j').1 hjK').2
  have hfirst : j' = jc ∧ pre' ++ L = Lc := by
    rcases hcase with h | ⟨hju', hl', hne'⟩
    · exact h
    exfalso
    have hju'' : j'.val / 1026 = u := hju'
    have hl'' : slotOf (#[tb] : Array String).size 1026 (j'.val % 1026) ≠ .link := hl'
    have hc' : j'.val % 1026 + 1 ≠ 1026 := fun h => hne' (Fin.ext (by omega))
    have hnd' : pre' ++ L ∉ done' := fun h => hW'.live j' _ h hps' haj'
    rcases hend with hlen | hnone
    · have hN := m.toV5_N t
      obtain ⟨T, hT, hTc⟩ : ∃ T : Finset (Fin R),
          (∀ j, j ∈ T ↔ j.val / 1026 = u ∧ j.val % 1026 + 1 ≠ 1026) ∧ T.card ≤ 1025 :=
        ⟨_, fun j => by
          rw [Finset.mem_filter]
          exact ⟨fun h => h.2, fun h => ⟨Finset.mem_univ _, h⟩⟩, session_card R (per := 1026) (by omega) u⟩
      have hj₀0 : j₀.val % 1026 = 0 := slotOf_link hl₀
      have hpick : ∀ m' ∈ ms, ∃ j, j ∈ T ∧ PicksIn (sessionReading R tb ev record₀ recordDigest).round
          (CoinTape.run (V.game θ F c) σ r.tape).log j m' := fun m' hm' => by
        obtain ⟨j, ρ, k, hLj, hp⟩ := hms0 m' hm'
        exact ⟨j, (hT j).2 ⟨hrep m' hm' j hp, (slotOf_rep hLj (one_table_pos tb)).1⟩, hp⟩
      have hnd : ms.Nodup := hpw.imp fun h e => by
        rw [e] at h
        exact Nat.lt_irrefl _ h
      have hmsl : ms.length = 1024 := by rw [← hF.length_eq, hlen, hN]
      have hcov := covers_of_card hU T ((hT j₀).2 ⟨hu₀, by omega⟩) h₀ hpick hgt hnd (by omega) j'
        ((hT j').2 ⟨hju'', hc'⟩) (pre' ++ L) hps'
      exact hnd' ((hd' _).2 (hcov.elim Or.inl fun h => Or.inr (Or.inl h)))
    · obtain ⟨ρ, k, hLj'⟩ := slotOf_mid hl'' hc'
      obtain ⟨w', hw'⟩ := Pos.modes_some hjK'
      have hsome := sub_single reg (fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026))
        (memberTabs S Rg (m.toV5 t)) hLj' (Nat.lt_of_lt_of_eq Nat.one_pos (m.tabs_eq t S Rg).symm) hw' hsingle
      have hnone' : (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size 1026 (j.val % 1026)).sub
          (memberTabs S Rg (m.toV5 t)) q.toStop.modes = none := hnone
      rw [hsome] at hnone'
      cases hnone'
  obtain ⟨hjj, hLcEq⟩ := hfirst
  rw [hjj] at haj' hPj' hps' hjK' hsingle
  subst hLcEq
  have hxc' : xc = ⟨.opening ic s n, b, bits⟩ := (List.cons.inj (List.append_cancel_left (hLc.symm.trans hlog))).1
  rw [hxc'] at hshac
  have hrev : (sessionReading R tb ev record₀ recordDigest).reveal jc (CoinTape.run (V.game θ F c) σ r.tape) =
      some w := by
    show sessionReveal #[tb] ev (jc.val / 1026) (slotOf (#[tb] : Array String).size 1026 (jc.val % 1026))
      (CoinTape.run (V.game θ F c) σ r.tape) = some w
    rw [hjcu, hjcs, sessionReveal_close]
    exact hev
  have hmode := stop_modes_tape hi_q hzq hn hcutq hlog hjK' hrev hshac
  have hfiles : closeFiles filesDec q = proofs := (closeFiles_eq filesDec hjcc hmode hsingle).trans hfd
  have hW'' := (hW'.next hU hNP (r := ⟨.opening ic s n, b, bits⟩) hlog haj hj).2
  have hlastu : ∀ (j₁ : Fin R) L₁, PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j₁ L₁ → j₁.val / 1026 = u → L₁.length ≤ (pre' ++ L).length :=
    fun j₁ L₁ hp hju => by
      by_cases hl : slotOf (#[tb] : Array String).size 1026 (j₁.val % 1026) = .link
      · have e := hSP.link_inj j₁ j₀ hl hl₀ (hju.trans hu₀.symm)
        rw [e] at hp
        rw [hU j₀ L₁ L₀ hp h₀]
        have h1 := (hW'.picked L₀ hL₀d).1
        rw [List.length_append]
        omega
      · obtain ⟨x₁, L₂, i₁, s₁, n₁, he₁, hx₁, -, hτ₁⟩ := hSP.rep_opens hl hp
        have h2 : (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log)[j₁.val / 1026]? = some i₁ := hτ₁
        rw [hju] at h2
        rw [Option.some.inj (h2.symm.trans hτ)] at hx₁
        exact hlastc L₁ x₁ L₂ s₁ n₁ he₁ hx₁
  have hDn' : SessDone (sessionReading R tb ev record₀ recordDigest).round (fun j => j.val / 1026)
      (CoinTape.run (V.game θ F c) σ r.tape).log ((pre' ++ L) :: done') (u + 1) :=
    hDn.closed hSP h₀ hu₀ hrep hc hjcu (fun L₁ => by rw [List.mem_cons, hd' L₁]) hW'' hlastu
  have hcov : ∀ (j₁ : Fin R) L₁, j₁.val / 1026 = u → PicksIn (sessionReading R tb ev record₀ recordDigest).round
      (CoinTape.run (V.game θ F c) σ r.tape).log j₁ L₁ → L₁ = L₀ ∨ L₁ ∈ ms ∨
        slotOf (#[tb] : Array String).size 1026 (j₁.val % 1026) = closeSlot (#[tb] : Array String).size :=
    fun j₁ L₁ hju hp => by
      by_cases hcl₁ : j₁.val % 1026 + 1 = 1026
      · exact Or.inr (Or.inr (slotOf_last _ hcl₁))
      rcases Nat.lt_or_eq_of_le (hlastu j₁ L₁ hp hju) with hlt | heq
      · have hd₁ : L₁ ∈ done' := hW'.seg hU hNP (L₃ := ⟨.opening ic s n, b, bits⟩ :: L')
          (by rw [hlog, List.append_assoc]) hp (by
            rw [List.length_append] at hlt
            exact hlt)
        rcases (hd' L₁).1 hd₁ with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · have h1 : j₁.val / 1026 < u := (hDn j₁ L₁ hp).1 h
          omega
      · exfalso
        obtain ⟨x₁, L₂, he₁, -⟩ := id hp
        rw [List.append_inj_left (he₁.symm.trans hlog) heq] at hp
        rw [hSP.unique j₁ jc _ hp hc] at hcl₁
        exact hcl₁ hjcc
  have hI' : ∀ a₁ ∈ as, a₁.1.toStop.Inv (roundCuts V θ F c (sessionReading R tb ev record₀ recordDigest)) ∧
      ∃ y, r.tape = a₁.1.toStop.back y := fun a₁ ha =>
    ⟨(Pos.toStop_spec (hI a₁ ha).1).2.2,
      (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans ((hI a₁ ha).2.le.trans hqr.le)).through⟩
  refine ⟨⟨pre' ++ L, f'', _, (pre' ++ L) :: done', hf'', hnext id, hW'', hDn'⟩, ?_, ?_, fun j₁ ρ => ?_, ?_⟩
  · rw [hout]
    exact hroot
  · rw [hout]
    exact hpts.symm
  · rw [hout]
    exact session_reps reg hJ (per := 1026) (by omega) ev record₀ recordDigest rfl hn hu1 hF hI' hpw hrep hcov
      (fun j hp => (hL₀' j hp).1) hτ hspecs j₁ (by have := j₁.isLt; omega) ρ hA.sent hA.read hA.decode hA.verify
      hA.retain hA.sha512 hA.dg hA.tables
  · rw [hout]
    exact hfiles

theorem SessDone.of_eq {R : ℕ} {P : Fin R → List Release → Frame → Prop} {su : Fin R → ℕ} {Lg : List Release}
    {done : List (List Release)} {u u' : ℕ} (h : SessDone P su Lg done u) (e : u = u') : SessDone P su Lg done u' :=
  e ▸ h

def StepInv (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (W : ExecutionPhase)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (ω : (W.extend M.widening).L.Ω) (r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)) (i : ℕ)
    (p : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)) : Prop :=
  ∃ (pre : List Release) (f : List Release → Game Exchange) (a : Fin W.rounds → Prop)
    (done : List (List Release)), (∀ log, f log = finish (V.signer θ) θ F c σ.1 (pre ++ log)) ∧
    StopOf rd.round rd.reveal pre f a p.toStop ∧
    Walked rd.round (CoinTape.run (V.game θ F c) σ r.tape).log pre a done ∧
    SessDone rd.round (fun j => j.val / 1026) (CoinTape.run (V.game θ F c) σ r.tape).log done
      (sampledBefore W.ms ((W.extend M.widening).L.draw ω) i) ∧
    (Pos.root (roundCuts V θ F c rd) (M.start ω)).Past p

theorem memberSteps_of_sessions (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (reg : Option Claims.Msg → ByteArray) (tb : String) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    (σ : Strategy (V.game θ F c)) (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds)
    (hML : ∀ i (p : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)) mPts (pts : Fin 2 → Fin mPts → 𝔽),
      M.link i p mPts pts = writeOpening (finish (V.signer θ) θ F c σ.1) p.s' (linkBits fieldEnc pts))
    (hMR : ∀ i (p : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)) J (s : SubmitJ J) (cs : Fin s.n → 𝔽),
      M.round i p J s cs = writeOpening (finish (V.signer θ) θ F c σ.1) p.s' (roundBits fieldEnc cs))
    (hMA : MembersAt W tb)
    (hses : ∀ (ω : (W.extend M.widening).L.Ω) (z : CoinTape (V.game θ F c) σ),
      (∃ w, z = (Pos.root (roundCuts V θ F c (sessionReading W.rounds tb ev record₀ recordDigest))
        (M.start ω)).toStop.back w) →
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ z) → P (CoinTape.run (V.game θ F c) σ z) →
      ((W.extend M.widening).joint t).ended ω = true ∧
        ExecutionPhaseSessions W tb ev filesDec ((W.extend M.widening).L.draw ω) (CoinTape.run (V.game θ F c) σ z)) :
    MemberSteps V θ F c k P W t (sessionReading W.rounds tb ev record₀ recordDigest) σ
      (fun _ => sessionDecoders reg tb filesDec) M
      (fun _ _ => linkEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (StepInv V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest) σ M) := by
  have hP := executionPhaseReadingAt_picks (R := W.rounds) #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest
  refine memberSteps_of_verdicts V θ F c k P W t _ σ _ M _ _ _ (fun ω r => ?_)
    (fun ω i m p q r out hm hd hI hpi _ hmr hqr _ hacc => ?_) (fun ω i m p r hm hd hI => ?_)
    (fun ω r hpr hacc => (hses ω r.tape hpr.through hacc.1 hacc.2.2).1)
  · exact ⟨[], _, fun _ => True, [], fun _ => rfl, stopOf_rootStop V θ F c _ σ (M.start ω) hP, walked_nil _ _,
      (sessDone_nil _ _ _).of_eq (sampledBefore_zero _ _).symm, Pos.Past.refl _⟩
  · obtain ⟨pre, f, a, done, hf, hSO, hW, hDn, hrp⟩ := hI
    have hdS : (m.part ((W.extend M.widening).L.draw ω)).Nonempty := hd
    obtain ⟨hE, hWS⟩ := hses ω r.tape (hrp.trans (hmr.past.trans hqr)).through hacc.1 hacc.2.2
    obtain ⟨a₀, bs, keys, s, record, spec, tabs, rest, proofs, units, J, sess, w, τ, sts, hst, hv, hdr, hs, hA,
      hτ, hev, hfd⟩ := hWS.specAt hMA hm hdS
    have hmP := m.mPts_eq t hs hdr ((W.extend M.widening).hcls m (List.mem_of_getElem? hm) ω hE)
    have hTr : (openedTrees (CoinTape.run (V.game θ F c) σ r.tape).log).length * 1026 ≤ W.rounds :=
      (Nat.mul_le_mul_right 1026 (hWS.trees.trans (sampledBefore_le _ _ _))).trans (Nat.le_of_eq W.rounds_eq.symm)
    obtain ⟨⟨pre'', f'', a'', done'', hf'', hSO'', hW'', hDn''⟩, hB, hPt, hRp, hPf⟩ :=
      session_step reg ev filesDec record₀ recordDigest rfl hML hMR t hmr hqr hpi hacc.2.1 hf hSO hW hDn
        hWS.specs hWS.order hTr (hWS.all hMA) hA hτ hev hfd hmP
    refine ⟨⟨pre'', f'', a'', done'', hf'', hSO'', hW'', ?_, hrp.trans (hmr.past.trans (Pos.past_consume q))⟩,
      a₀, bs, keys, s, record, spec, tabs, rest, proofs, units, hst, hv, ?_⟩
    · exact hDn''.of_eq (sampledBefore_succ_of_sampled hm hdS).symm
    · intro spec' sts' sess' hs' hdec'
      have hspec : spec = spec' := (Prod.mk.inj (Except.ok.inj (hs.symm.trans hs'))).1
      rw [← hspec] at hdec' ⊢
      rw [← Except.ok.inj (hA.decode.symm.trans hdec')]
      exact ⟨hdr, hB, hPt, hRp, hPf⟩
  · obtain ⟨pre, f, a, done, hf, hSO, hW, hDn, hrp⟩ := hI
    have hdS : ¬ (m.part ((W.extend M.widening).L.draw ω)).Nonempty := hd
    exact ⟨pre, f, a, done, hf, hSO, hW, hDn.of_eq (sampledBefore_succ_of_not hm hdS).symm, hrp⟩

end Verity.Sampled
