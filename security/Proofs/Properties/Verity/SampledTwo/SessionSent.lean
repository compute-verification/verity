module

public import Proofs.Properties.Verity.SampledTwo.FormSetup

/-!
# The replaying provers send the messages the check read

`SessionMsgs` (`Accepts.lean`) asks that the sequential prover's run sends, in each template's session's rounds, the
messages the check read of that session. This module proves it at the sessions' replaying provers
(`SessionMsgs.replayProver`), which send a session's messages off the play's case whatever they are answered.

* **A session's messages in the sequential prover's run** (`session_sent`): `SpotFlock.proverSeq` plays template
  `t`'s prover on the session's history until one of the session's rounds is answered `[]`, which is how
  `SpotFlock.V` answers once the session is decided. Where the session's prover sends a list of messages as long as
  the session's rounds, each but the last of tag `0`, at a statement that sets up, Flock asks for a coin after every
  message but the last (`FlockComplete.act_coin_any`). So each of those rounds is answered with its coin's bytes,
  which are not `[]` (`SpotFlock.coin_bytes_ne`), and the run's messages in the session's rounds are the list.
* **A session's messages but its last are of tag `0`** (`sessionMsgs_tag`): the link message and each round's
  retained bytes.
* **`SessionMsgs` at the replaying provers** (`sessionMsgs_replay`): from the reading's statements and messages
  being the case's, `StmtsCount`, `SessionCoins.StmtsFit`, `SessionCoins.VerdictsAt` and `PlayAccepts` at the
  replaying provers. An accepted run has all its rounds (`length_of_acc`), and template `t`'s session's messages on
  the case are as many as its rounds (`SessionCoins.sessionMsgs_length`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **A session's messages in the sequential prover's run**: in a run of `SpotFlock.proverSeq` that has all its rounds,
where template `t`'s session's prover sends the list `ms` whatever it is answered, `ms` as long as the session's rounds
and each message but the last of tag `0`, at a session statement that sets up, the run's messages in the session's
rounds are `ms`. Each round before the session's last is answered with its coin's bytes, so the sequential prover
never reads the session as decided before its last message. -/
theorem session_sent (h : SpotFlock.Held) {Ω : Type} (H : Msg → Msg) (x m₀ m₁ m₂ : Msg) (P' : ℕ → Prover Ω)
    (ω : ℕ → Ω) (r : Fin ((SpotFlock.V h).k x) → FlockDeploy.Coin) {t : ℕ} (ht : t < h.ms.length) {ms : List Msg}
    (hK : ms.length = SpotFlock.k₀ h t)
    (hP : ∀ hist, (((P' t).next (SpotFlock.stmtAt h m₂ t) hist (ω t)).eval H).1 = ms.getD hist.length [])
    (htag : ∀ j, j + 1 < ms.length → ∃ tl, ms.getD j [] = 0 :: tl)
    {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, sts))
    (hlen : (Claims.run H (SpotFlock.proverSeq h m₀ m₁ m₂ P') (SpotFlock.V h) x ω r).msgs.length =
      SpotFlock.off h h.ms.length + 1) :
    ∀ j < SpotFlock.k₀ h t,
      msgAt (Claims.run H (SpotFlock.proverSeq h m₀ m₁ m₂ P') (SpotFlock.V h) x ω r) (SpotFlock.off h t + j) =
        some (ms.getD j []) := by
  have h3 := SpotFlock.stage_le_off h t
  have hle : SpotFlock.off h t + SpotFlock.k₀ h t ≤ SpotFlock.off h h.ms.length :=
    Claims.Sampling.sessOff_add_le (h.stage + 2) (SpotFlock.k₀ h) ht
  have hget := fun i hi =>
    RecDeploy.run_get H (SpotFlock.proverSeq h m₀ m₁ m₂ P') (SpotFlock.V h) x ω r i hi
  generalize Claims.run H (SpotFlock.proverSeq h m₀ m₁ m₂ P') (SpotFlock.V h) x ω r = T at hget hlen ⊢
  -- round `stage + 1` sends the session statements
  have hm₂ : (T.msgs[h.stage + 1]?).map Prod.fst = some m₂ := by
    obtain ⟨new, hnew, hg⟩ := hget (h.stage + 1) (by omega)
    have hl : (new.map RecDeploy.histOf).length = h.stage + 1 := by
      rw [List.length_map, hnew, List.length_take]
      omega
    have hp : ((SpotFlock.proverSeq h m₀ m₁ m₂ P').next x (new.map RecDeploy.histOf) ω).eval H = (m₂, []) := by
      rw [SpotFlock.proverSeq_pre h m₀ m₁ m₂ P' x _ ω (by omega), hl, SpotFlock.preMsg_last]
      rfl
    rw [hg, hp]
    rfl
  -- one round of the session, from the rounds before it
  have step : ∀ j, j < SpotFlock.k₀ h t →
      (∀ i < j, ∃ b, T.msgs[SpotFlock.off h t + i]? = some (ms.getD i [], b) ∧
        (i + 1 < SpotFlock.k₀ h t → ∃ c : FlockDeploy.Coin, b = .msg c.bytes)) →
      ∃ b, T.msgs[SpotFlock.off h t + j]? = some (ms.getD j [], b) ∧
        (j + 1 < SpotFlock.k₀ h t → ∃ c : FlockDeploy.Coin, b = .msg c.bytes) := by
    intro j hjK hprev
    obtain ⟨new, hnew, hg⟩ := hget (SpotFlock.off h t + j) (by omega)
    have hnl : new.length = SpotFlock.off h t + j := by
      rw [hnew, List.length_take]
      omega
    have hent : ∀ i < j, ∃ c : FlockDeploy.Coin,
        new[SpotFlock.off h t + i]? = some (ms.getD i [], .msg c.bytes) := by
      intro i hi
      obtain ⟨b, hb, hc⟩ := hprev i hi
      obtain ⟨c, rfl⟩ := hc (by omega)
      exact ⟨c, by rw [hnew, List.getElem?_take, ite_eq_left (by omega)]; exact hb⟩
    have hhl : (new.map RecDeploy.histOf).length = SpotFlock.off h t + j := by rw [List.length_map, hnl]
    have hslot : SpotFlock.slotOf h (SpotFlock.off h t + j + 1) = some t :=
      SpotFlock.slotOf_eq h ht (by omega) (by omega)
    -- no earlier answer of the session is `[]`
    have hunm : SpotFlock.marked ((new.map RecDeploy.histOf).drop (SpotFlock.off h t)) = false := by
      unfold SpotFlock.marked
      rw [List.any_eq_false]
      intro e he
      obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 he
      rw [List.getElem?_drop, List.getElem?_map] at hi
      rcases Nat.lt_or_ge i j with hij | hij
      · obtain ⟨c, hc⟩ := hent i hij
        have hne := SpotFlock.coin_bytes_ne c
        -- the coin's bytes as a variable: unifying against `c.bytes` itself unfolds its 65536 elements
        generalize c.bytes = b at hc hne
        rw [hc] at hi
        obtain rfl := Option.some.inj hi
        exact fun hd => hne (of_decide_eq_true hd)
      · rw [List.getElem?_eq_none (by omega)] at hi
        cases hi
    have hpa : (((SpotFlock.proverSeq h m₀ m₁ m₂ P').next x (new.map RecDeploy.histOf) ω).eval H).1 =
        ms.getD j [] := by
      rw [SpotFlock.proverSeq_next h m₀ m₁ m₂ P' x _ ω (by omega), hhl, hslot]
      dsimp only
      rw [ite_eq_right fun hm => Bool.false_ne_true (hunm.symm.trans hm), hP, List.length_drop, hhl,
        Nat.add_sub_cancel_left]
    rw [hpa] at hg
    refine ⟨_, hg, fun hj1 => ?_⟩
    show ∃ c : FlockDeploy.Coin, ((SpotFlock.next h x (new.map Prod.fst ++ [ms.getD j []])
      ((List.range (SpotFlock.off h t + j + 1)).map (coinSeq r))).eval H).1 = .msg c.bytes
    generalize has' : new.map Prod.fst ++ [ms.getD j []] = as'
    generalize hcs' : (List.range (SpotFlock.off h t + j + 1)).map (coinSeq r) = cs'
    have hasl : as'.length = SpotFlock.off h t + j + 1 := by
      rw [← has', List.length_append, List.length_map, hnl, List.length_singleton]
    have hcsl : cs'.length = SpotFlock.off h t + j + 1 := by rw [← hcs', List.length_map, List.length_range]
    have hasg : ∀ i, i ≤ j → as'[SpotFlock.off h t + i]? = some (ms.getD i []) := by
      intro i hi
      rcases Nat.lt_or_ge i j with hij | hij
      · obtain ⟨c, hc⟩ := hent i hij
        rw [← has', List.getElem?_append_left (by rw [List.length_map]; omega), List.getElem?_map, hc]
        rfl
      · obtain rfl : i = j := by omega
        rw [← has', List.getElem?_append_right (by rw [List.length_map]; omega), List.length_map, hnl,
          Nat.sub_self]
        rfl
    have has2 : as'.getD (h.stage + 1) [] = m₂ := by
      rw [List.getD_eq_getElem?_getD, ← has', List.getElem?_append_left (by rw [List.length_map]; omega),
        List.getElem?_map, hnew, List.getElem?_take, ite_eq_left (by omega), hm₂]
      rfl
    -- Flock asks for a coin on every prefix of the session's messages so far
    have hcoin : ∀ j', 0 < j' → j' ≤ j + 1 → ∃ c, FlockDeploy.act true (SpotFlock.stmtAt h m₂ t)
        ((as'.drop (SpotFlock.off h t)).take j') ((cs'.drop (SpotFlock.off h t)).take j') = .coin c := by
      intro j' h0 hj'
      obtain ⟨tl, htl⟩ := htag (j' - 1) (by omega)
      have hla : ((as'.drop (SpotFlock.off h t)).take j').length = j' := by
        rw [List.length_take, List.length_drop]
        omega
      have hlc : ((cs'.drop (SpotFlock.off h t)).take j').length = j' := by
        rw [List.length_take, List.length_drop]
        omega
      have ha : ((as'.drop (SpotFlock.off h t)).take j').getLast? = some (0 :: tl) := by
        rw [List.getLast?_eq_getElem?, hla, List.getElem?_take, ite_eq_left (by omega), List.getElem?_drop,
          hasg (j' - 1) (by omega), htl]
      cases hc : ((cs'.drop (SpotFlock.off h t)).take j').getLast? with
      | some c => exact ⟨c, FlockComplete.act_coin_any hs ha hc⟩
      | none =>
        rw [List.getLast?_eq_none_iff] at hc
        rw [hc, List.length_nil] at hlc
        omega
    have hund : ∀ j', 0 < j' → j' < as'.length - SpotFlock.off h t →
        SpotFlock.decidedAt (SpotFlock.stmtAt h m₂ t) (as'.drop (SpotFlock.off h t))
          (cs'.drop (SpotFlock.off h t)) j' = false :=
      fun j' h0 hj' => Messages.decidedAt_false_iff.2 (hcoin j' h0 (by omega))
    obtain ⟨c, hc⟩ := hcoin (j + 1) (by omega) (Nat.le_refl _)
    rw [List.take_of_length_le (by rw [List.length_drop]; omega),
      List.take_of_length_le (by rw [List.length_drop]; omega)] at hc
    rw [SpotFlock.next_long h x as' cs' (by omega) (by omega), hasl, hslot]
    dsimp only
    rw [has2, SpotFlock.sessStep_eval h H t _ as' cs' hund, FlockDeploy.next_coin H hc]
    exact ⟨c, rfl⟩
  have all : ∀ n j, j < n → j < SpotFlock.k₀ h t →
      ∃ b, T.msgs[SpotFlock.off h t + j]? = some (ms.getD j [], b) ∧
        (j + 1 < SpotFlock.k₀ h t → ∃ c : FlockDeploy.Coin, b = .msg c.bytes) := by
    intro n
    induction n with
    | zero => exact fun j hj => absurd hj (Nat.not_lt_zero j)
    | succ n ih => exact fun j hj hjK => step j hjK fun i hi => ih i (by omega) (by omega)
  intro j hj
  obtain ⟨b, hb, -⟩ := all (j + 1) j (Nat.lt_succ_self j) hj
  unfold msgAt
  rw [hb]
  rfl

/-- **A session's messages but its last are of tag `0`.** -/
theorem sessionMsgs_tag {spec : _root_.Flock.SessionSpec} {x : Msg} {record : String} {proofs : Array ByteArray}
    {ms : List Msg} (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) :
    ∀ j, j + 1 < ms.length → ∃ tl, ms.getD j [] = 0 :: tl := by
  obtain ⟨lv, mids, -, hmids, hms⟩ := SessionMsgs.sessionMsgs_some hm
  have hms' : ms = ((0 :: FlockDeploy.linkBody lv.rootB.toList lv.publics.toList) :: mids) ++
      [FlockDeploy.Final.encode ⟨x, record, proofs⟩] := hms
  intro j hj
  rw [hms', List.length_append, List.length_singleton] at hj
  have hj' : j < ((0 :: FlockDeploy.linkBody lv.rootB.toList lv.publics.toList) :: mids).length := by omega
  rw [hms', List.getD_eq_getElem?_getD, List.getElem?_append_left hj', List.getElem?_eq_getElem hj',
    Option.getD_some]
  rcases List.mem_cons.1 (List.getElem_mem hj') with h0 | hmem
  · exact ⟨_, h0⟩
  · obtain ⟨i, hi, hie⟩ := List.getElem_of_mem hmem
    obtain ⟨hl, hall⟩ := SessionMsgs.mapM_some _ hmids
    obtain ⟨c, -, hc⟩ := Option.map_eq_some_iff.1 (hall i (by omega) hi)
    exact ⟨_, (hie.symm.trans hc.symm)⟩

/-- **The replaying provers' messages are the ones the check read** (`SessionMsgs` at `SessionMsgs.replayProver`):
where the reading's statements and messages are the case's, each template's session passed its verdict
(`SessionCoins.VerdictsAt`) and the run of record is a play SPOT[Flock] accepts at the replaying provers
(`PlayAccepts`), the sequential prover's run sends, in template `t`'s session's rounds, that session's messages on the
case. The accepted run has all its rounds (`length_of_acc`), template `t`'s statement in the statements' message is its
session statement on the case (`SessionMsgs.stmtAt_stmtsMsg`), whose setup the verdict ran at, and the session's
messages are as many as its rounds (`SessionCoins.sessionMsgs_length`), each but the last of tag `0`
(`sessionMsgs_tag`): `session_sent`. -/
theorem sessionMsgs_replay {A : Live₂} {C : CaseOf} {S : SessionReading} {tab : ℕ → String}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j)
    (hcount : StmtsCount A C fun _ => True) (hfit : SessionCoins.StmtsFit C fun _ => True)
    (hverd : SessionCoins.VerdictsAt A C (fun _ => True) tab)
    (hplay : PlayAccepts A C S (SessionMsgs.replayProver C)) :
    SessionMsgs A C S (SessionMsgs.replayProver C) := by
  intro o i it x m₁ m₂ hA hi hit hx hm₁ hm₂ c₀ c₁ r' hread t ht j hj
  have hA' := hA
  obtain ⟨k, hk, -⟩ := hA'
  obtain ⟨m, hm, -, hacc⟩ := hplay o i it x m₁ hA hi hit hx hm₁
  obtain rfl : m₂ = m := Option.some.inj (hm₂.symm.trans hm)
  have hlen := length_of_acc A.h FlockDeploy.sha512
    (SpotFlock.proverSeq A.h [] m₁ m₂ (SessionMsgs.replayProver C o)) x (fun _ => ())
    (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r') (hacc c₀ c₁ r' hread)
  obtain ⟨xS, s, spec, st, record, proofs, ms, hparts, hv, -, hT, hk₀, hms⟩ := hverd o k hA trivial hk t ht
  have hmk : SessionMsgs.stmtsMsg k = some m₂ := by
    rw [← hm₂, hS, hk]
    rfl
  obtain ⟨p, e, xs, hf, hxs, -⟩ := SessionMsgs.stmtsOf_stmtsMsg hmk
  obtain ⟨hl, hb⟩ := hfit o k p e xs hA trivial hk hf hxs
  have hst : SessionMsgs.stmtAtCase k t = some (SpotFlock.stmtAt A.h m₂ t) :=
    SessionMsgs.stmtAt_stmtsMsg A.h hmk hf hxs hl hb (by rw [hcount o k p e hA trivial hk hf]; exact ht)
  have hmc := SessionMsgs.msgsAtCase_of_parts hparts hms
  obtain ⟨-, -, -, -, -, -, hx', hsu, -⟩ := hparts
  obtain rfl : xS = SpotFlock.stmtAt A.h m₂ t := Option.some.inj (hx'.symm.trans hst)
  have hK : ms.length = SpotFlock.k₀ A.h t := by rw [SessionCoins.sessionMsgs_length hsu hv hT hms, hk₀]
  have hjl : j < ms.length := by omega
  have hsent := session_sent A.h FlockDeploy.sha512 x [] m₁ m₂ (SessionMsgs.replayProver C o) (fun _ => ())
    (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r') ht hK
    (fun hist => SessionReplay.replayProver_next FlockDeploy.sha512 C o t _ hk hmc hist) (sessionMsgs_tag hms) hsu
    hlen j hj
  rw [hM, hk]
  show _ = SessionMsgs.msgAtCase k t j
  unfold SessionMsgs.msgAtCase
  rw [hmc]
  show _ = ms[j]?
  rw [List.getElem?_eq_getElem hjl]
  refine hsent.trans ?_
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjl, Option.getD_some]

end Verity.SampledTwo
