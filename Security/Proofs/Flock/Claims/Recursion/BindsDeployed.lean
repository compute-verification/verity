import Proofs.Flock.Claims.Recursion.BoundDeploy
import Proofs.Flock.Soundness.Discharge.Hm.Row
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.Bridge.Custody

/-!
# (P2) and (P4) at the intended choice

`KSBoundRoute` takes (P2, `Bound.Binds`) and (P4, `Bound.LastBinds`) at the deployed bound record, at every choice the
code side makes. Here both are read at the intended choice.

**(P2) holds** where the rows' digest input is the deployed one (`rowInDeployed`): `hm96-sha512/row/v1`'s
`sha512_row_prefix(1, 16, n_words) ‖ row`, its first block `Flock.HmRow.rowPrefix` of a fixed 128 bytes, so the row is
the rest (`binds_deployed`).

**(P4) fails** where the element-level verifier's steps are the deployed plain verifier's (`FlockDeploy.V false`) and
`final` keeps `DE`'s last message as it is, which "per earlier round, its rows with their salts, then `DE`'s last
message" does. That last message is `Final.encode` of the statement, the record's text and the proof files, and the
verifier reads the record's text only through `Json.parse` (`Record.decodeWith`, `liveOf`, `recordDraw`). The text
with one trailing space parses alike (`JsonTrailingSpace`, Lean's parser ends every value with `ws` before `eof`), so it
is another last message that `final` reads differently and `IVStar` decides alike, with the same calls
(`ivstar_congr_snoc`). (P4) then asks a collision among the calls of one accepted decision
(`lastBinds_forces_collision`, `lastBinds_deployed_forces_collision`): at an oracle with no collision among those calls,
(P4) at the deployed record leaves `IVStar` accepting nothing whose last message fits its length words
(`lastBinds_deployed_rejects`), where an honest run is accepted. `LastBinds`' statement modulo `final` covers bytes
`final` ignores, not bytes `DE`'s verifier ignores.

`JsonTrailingSpace` is a hypothesis, not a lemma: `Json.parse`'s `anyCore` is `partial`, so the kernel doesn't see its
body. The rest is proved. -/

namespace RecDeploy

open Claims (Msg Out OC)
open Lean (Json)

/-! ## (P2) -/

/-- **The deployed rows' digest input** (`hm96-sha512/row/v1`): the first block for the row's u16 words
(`Flock.HmRow.rowPrefix`), then the row. -/
def rowInDeployed (row : Msg) : Msg := (_root_.Flock.HmRow.rowPrefix (row.length / 2)).data.toList ++ row

/-- The first block is 128 bytes at every word count, so the row is what follows it. -/
theorem rowInDeployed_injective : Function.Injective rowInDeployed := by
  intro r r' h
  unfold rowInDeployed at h
  have hl : ((_root_.Flock.HmRow.rowPrefix (r.length / 2)).data.toList).length =
      ((_root_.Flock.HmRow.rowPrefix (r'.length / 2)).data.toList).length := by
    rw [FlockSoundness.Discharge.Hm.length_rowPrefix, FlockSoundness.Discharge.Hm.length_rowPrefix]
  exact (List.append_inj h hl).2

/-- **(P2) at the deployed rows' digest input**, whatever the element-level verifier, the columns, the weights, the
join and `final`. -/
theorem binds_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowInDeployed final w).Binds :=
  rowInDeployed_injective

/-! ## (P4): the hypotheses -/

/-- **Lean's JSON parser accepts trailing whitespace**: `Json.parse` is `ws; anyCore; eof`, and every branch of
`anyCore` ends with `ws`. A hypothesis because `anyCore` is `partial`. -/
def JsonTrailingSpace : Prop :=
  ∀ (r : String) (j : Json), Json.parse r = .ok j → Json.parse (r ++ " ") = .ok j

/-- **`final` reads any last element-level message with the same openings**: what "per earlier round, its rows with
their salts, then `DE`'s last message" asks of an encoding. -/
def FinalFree {S : Type} (final : Msg → Option (List (List (Msg × S)) × Msg)) : Prop :=
  ∀ a os last, final a = some (os, last) → ∀ last', ∃ a', final a' = some (os, last')

/-- **A last message fits its length words** (8 bytes each), with one byte to spare in the record's. -/
def FitsSpace (f : FlockDeploy.Final) : Prop :=
  f.stmt.length < 2 ^ 64 ∧ (f.record ++ " ").toUTF8.size < 2 ^ 64 ∧ f.proofs.size < 2 ^ 64 ∧
    ∀ b ∈ f.proofs, b.size < 2 ^ 64

end RecDeploy

namespace Claims.Recursion.LastGap

open Claims

/-! ## `IVStar`'s decision -/

theorem eval_bind (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    (bindOC p f).eval H = (((f (p.eval H).1).eval H).1, (p.eval H).2 ++ ((f (p.eval H).1).eval H).2) := by
  induction p with
  | ret a => simp [bindOC, OC.eval]
  | query q k ih => simp [bindOC, OC.eval, ih]

theorem eval_allOC_cons (H : Msg → Msg) (p : OC Bool) (ps : List (OC Bool)) (hp : (p.eval H).1 = true) :
    (allOC (p :: ps)).eval H = (((allOC ps).eval H).1, (p.eval H).2 ++ ((allOC ps).eval H).2) := by
  show (bindOC p fun b => if b then allOC ps else OC.ret false).eval H = _
  rw [eval_bind, hp]
  rfl

/-- Where `allOC` returns `true`, so does each of its programs, and each one's calls are among its calls. -/
theorem allOC_true (H : Msg → Msg) : ∀ ps : List (OC Bool), ((allOC ps).eval H).1 = true →
    ∀ p ∈ ps, (p.eval H).1 = true ∧ ∀ e ∈ (p.eval H).2, e ∈ ((allOC ps).eval H).2
  | [], _, _, hp => absurd hp List.not_mem_nil
  | q :: qs, h, p, hp => by
    cases hq : (q.eval H).1 with
    | false =>
      have hf : ((allOC (q :: qs)).eval H).1 = false := by
        show ((bindOC q fun b => if b then allOC qs else OC.ret false).eval H).1 = false
        rw [eval_bind, hq]
        rfl
      rw [hf] at h
      cases h
    | true =>
      rw [eval_allOC_cons H q qs hq] at h ⊢
      rcases List.mem_cons.1 hp with rfl | hp
      · exact ⟨hq, fun e he => List.mem_append_left _ he⟩
      · obtain ⟨h1, h2⟩ := allOC_true H qs h p hp
        exact ⟨h1, fun e he => List.mem_append_right _ (h2 e he)⟩

variable {Coin Col T : Type} [Inhabited Coin]

/-- **An accepted decision at the last round** reads the last message (`final`), opens one round per earlier message,
and the element-level verifier accepts the opened messages. -/
theorem ivstar_acc (Bd : Bound Coin Col T) (H : Msg → Msg) {x : Msg} {as : List Msg} {cs : List Coin}
    (hk : as.length = Bd.DE.V.k x) (h : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ os last, Bd.final (as.getLastD []) = some (os, last) ∧ os.length + 1 = as.length ∧
      ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true := by
  have hlt : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, hlt, ↓reduceIte] at h
  split at h
  · simp [OC.eval] at h
  · rename_i os last hfin
    split at h
    · rename_i hc
      rw [eval_bind] at h
      cases hok : ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).1
      · rw [hok] at h
        simp [OC.eval] at h
      · rw [hok] at h
        simp only [↓reduceIte] at h
        rw [eval_bind] at h
        cases hacc : ((Bd.agrees x (Bd.opened os last) cs).eval H).1
        · rw [hacc] at h
          simp [OC.eval] at h
        · exact ⟨os, last, hfin, hc.1, hacc⟩
    · simp [OC.eval] at h

/-- Where the opened messages agree, the element-level verifier accepts them at the last round. -/
theorem agrees_last (Bd : Bound Coin Col T) (H : Msg → Msg) {x : Msg} {ms₀ : List Msg} {l : Msg}
    {cs : List Coin} (h : ((Bd.agrees x (ms₀ ++ [l]) cs).eval H).1 = true) :
    ((Bd.DE.V.next x (ms₀ ++ [l]) (cs.take (ms₀.length + 1))).eval H).1 = .acc := by
  unfold Bound.agrees at h
  have hm := (allOC_true H _ h _ (List.mem_map.2 ⟨ms₀.length, List.mem_range.2 (by simp), rfl⟩)).1
  rw [eval_bind] at hm
  have e : (ms₀ ++ [l]).take (ms₀.length + 1) = ms₀ ++ [l] := List.take_of_length_le (by simp)
  rw [e] at hm
  simpa [OC.eval] using hm

/-- `agrees` reads the last opened message only through the element-level verifier's last step. -/
theorem agrees_snoc (Bd : Bound Coin Col T) {x : Msg} {ms₀ : List Msg} {l l' : Msg} {cs : List Coin}
    (h : Bd.DE.V.next x (ms₀ ++ [l']) (cs.take (ms₀.length + 1)) =
      Bd.DE.V.next x (ms₀ ++ [l]) (cs.take (ms₀.length + 1))) :
    Bd.agrees x (ms₀ ++ [l']) cs = Bd.agrees x (ms₀ ++ [l]) cs := by
  unfold Bound.agrees
  simp only [List.length_append, List.length_singleton]
  congr 1
  apply List.map_congr_left
  intro j hj
  rw [List.mem_range] at hj
  by_cases hjl : j < ms₀.length
  · have e1 : (ms₀ ++ [l']).take (j + 1) = ms₀.take (j + 1) := List.take_append_of_le_length (by omega)
    have e2 : (ms₀ ++ [l]).take (j + 1) = ms₀.take (j + 1) := List.take_append_of_le_length (by omega)
    rw [e1, e2]
  · have hj' : j = ms₀.length := by omega
    subst hj'
    have e1 : (ms₀ ++ [l']).take (ms₀.length + 1) = ms₀ ++ [l'] := List.take_of_length_le (by simp)
    have e2 : (ms₀ ++ [l]).take (ms₀.length + 1) = ms₀ ++ [l] := List.take_of_length_le (by simp)
    rw [e1, e2, h]

/-- Two last messages `final` reads with the same openings, at whose last opened messages `agrees` is one program, give
one decision, with the same calls. -/
theorem ivstar_congr_snoc (Bd : Bound Coin Col T) {x : Msg} {as as' : List Msg} {cs : List Coin}
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last last' : Msg}
    (hlen : as'.length = as.length) (hdrop : as'.dropLast = as.dropLast)
    (hfin : Bd.final (as.getLastD []) = some (os, last)) (hfin' : Bd.final (as'.getLastD []) = some (os, last'))
    (hag : Bd.agrees x (Bd.opened os last') cs = Bd.agrees x (Bd.opened os last) cs) :
    Bd.IVStar.next x as' cs = Bd.IVStar.next x as cs := by
  simp only [Bound.IVStar, hlen, hfin, hfin', hdrop, hag]

/-- **(P4) asks a collision of one decision** wherever `final` is free in the last element-level message
(`hfree`) and the element-level verifier, at each last message it accepts, takes another one alike (`hnext`): the
second last message gives a second accepted decision that `final` reads differently, with the same calls. -/
theorem lastBinds_self [Fintype Coin] [Nonempty Coin] [DecidableEq Col] (Bd : Bound Coin Col T) (H : Msg → Msg)
    (hfree : ∀ a os last, Bd.final a = some (os, last) → ∀ last', ∃ a', Bd.final a' = some (os, last'))
    (hP4 : Bd.LastBinds H) {x : Msg} {as : List Msg} {cs : List Coin}
    (hk : as.length = Bd.DE.V.k x) (hcs : cs.length = as.length)
    (hnext : ∀ os last, Bd.final (as.getLastD []) = some (os, last) →
      ((Bd.DE.V.next x (Bd.opened os last) (cs.take (os.length + 1))).eval H).1 = .acc →
      ∃ last', last' ≠ last ∧
        Bd.DE.V.next x (Bd.opened os last') (cs.take (os.length + 1)) =
          Bd.DE.V.next x (Bd.opened os last) (cs.take (os.length + 1)))
    (hacc : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ q ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst,
      ∃ q' ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst, Collision H q q' := by
  obtain ⟨os, last, hfin, hlen, hag⟩ := ivstar_acc Bd H hk hacc
  have hl := agrees_last Bd H (ms₀ := os.map (fun o => Bd.join (o.map Prod.fst))) (l := last) hag
  rw [List.length_map] at hl
  obtain ⟨last', hne, hn⟩ := hnext os last hfin hl
  have hag' : Bd.agrees x (Bd.opened os last') cs = Bd.agrees x (Bd.opened os last) cs := by
    apply agrees_snoc Bd
    rw [List.length_map]
    exact hn
  obtain ⟨a', ha'⟩ := hfree _ os last hfin last'
  have hlen' : (as.dropLast ++ [a']).length = as.length := by
    rw [List.length_append, List.length_dropLast, List.length_singleton]
    omega
  have hdrop : (as.dropLast ++ [a']).dropLast = as.dropLast := List.dropLast_concat
  have hfin' : Bd.final ((as.dropLast ++ [a']).getLastD []) = some (os, last') := by
    rw [List.getLastD_concat]
    exact ha'
  have hIV := ivstar_congr_snoc Bd hlen' hdrop hfin hfin' hag'
  have hne' : Bd.final (as.getLastD []) ≠ Bd.final ((as.dropLast ++ [a']).getLastD []) := by
    rw [hfin, hfin']
    intro h
    exact hne (Prod.ext_iff.1 (Option.some.inj h)).2.symm
  have hacc' : ((Bd.IVStar.next x (as.dropLast ++ [a']) cs).eval H).1 = .acc := by
    rw [hIV]
    exact hacc
  obtain ⟨q, hq, q', hq', hc⟩ :=
    hP4 x as (as.dropLast ++ [a']) cs cs hk hlen' hcs hcs hdrop.symm rfl hne' hacc hacc'
  rw [hIV, List.map_append, List.mem_append, or_self] at hq hq'
  exact ⟨q, hq, q', hq', hc⟩

end Claims.Recursion.LastGap

namespace RecDeploy.LastGap

open Claims (Msg Out OC)
open Lean (Json)

/-! ## The deployed plain verifier reads the record only through `Json.parse` -/

theorem recordDraw_congr {r r' : String} (h : Json.parse r' = Json.parse r) :
    FlockSoundness.Discharge.Exec.recordDraw r' = FlockSoundness.Discharge.Exec.recordDraw r := by
  unfold FlockSoundness.Discharge.Exec.recordDraw
  rw [h]

theorem liveOf_congr (spec : _root_.Flock.SessionSpec) {r r' : String} (h : Json.parse r' = Json.parse r) :
    FlockDeploy.liveOf spec r' = FlockDeploy.liveOf spec r := by
  unfold FlockDeploy.liveOf
  rw [h]

theorem decodeWith_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]
    (hs : _root_.Flock.Hashes M) (spec : _root_.Flock.SessionSpec) {r r' : String}
    (h : Json.parse r' = Json.parse r) :
    _root_.Flock.Record.decodeWith hs spec r' = _root_.Flock.Record.decodeWith hs spec r := by
  unfold _root_.Flock.Record.decodeWith
  rw [h]

theorem verifyWithM_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]
    (hs : _root_.Flock.Hashes M) (spec : _root_.Flock.SessionSpec) (tabs : Array _root_.Flock.Setup)
    (ps : Array ByteArray) {r r' : String} (h : Json.parse r' = Json.parse r) :
    _root_.Flock.verifyWithM hs spec tabs r' ps = _root_.Flock.verifyWithM hs spec tabs r ps := by
  unfold _root_.Flock.verifyWithM
  rw [decodeWith_congr hs spec h]

theorem checks_congr_record {zk : Bool} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : FlockDeploy.Coin} {mids : List (Msg × FlockDeploy.Coin)}
    {st : Msg} {ps : Array ByteArray} {r r' : String} (h : Json.parse r' = Json.parse r) :
    FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r', ps⟩ =
      FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r, ps⟩ := by
  unfold FlockDeploy.checks
  simp only [recordDraw_congr h, liveOf_congr spec h]

theorem decisionAt_congr_record {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {x : Msg} {link : List ByteArray × List ByteArray} {c₁ : FlockDeploy.Coin}
    {mids : List (Msg × FlockDeploy.Coin)} {st : Msg} {ps : Array ByteArray} {r r' : String}
    (h : Json.parse r' = Json.parse r) :
    FlockDeploy.decisionAt false s spec sts x link c₁ mids ⟨st, r', ps⟩ =
      FlockDeploy.decisionAt false s spec sts x link c₁ mids ⟨st, r, ps⟩ := by
  unfold FlockDeploy.decisionAt FlockDeploy.verdictM
  simp only [Bool.false_eq_true, ↓reduceIte, verifyWithM_congr Claims.Hashes.oc spec (FlockDeploy.tabsOf sts) ps h,
    checks_congr_record h]

theorem decision_congr_record {s : FlockDeploy.Ins} {x : Msg} {link : List ByteArray × List ByteArray}
    {c₁ : FlockDeploy.Coin} {mids : List (Msg × FlockDeploy.Coin)} {st : Msg} {ps : Array ByteArray}
    {r r' : String} (h : Json.parse r' = Json.parse r) :
    FlockDeploy.decision false s x link c₁ mids ⟨st, r', ps⟩ =
      FlockDeploy.decision false s x link c₁ mids ⟨st, r, ps⟩ := by
  unfold FlockDeploy.decision
  simp only [decisionAt_congr_record h]

theorem append_space_ne (r : String) : r ++ " " ≠ r := by
  intro h
  have hl := congrArg String.length h
  rw [String.length_append] at hl
  have h1 : (" " : String).length = 1 := by decide
  omega

/-- **The deployed plain verifier's last step takes another last message alike**: where it accepts the last message
`l`, the same message with one trailing space in the record's text (`Final.encode`) is another one, at which its step
is the same program. -/
theorem next_snoc_space (H : Msg → Msg) (hws : JsonTrailingSpace) {x : Msg} {ms₀ : List Msg} {l : Msg}
    {cs : List FlockDeploy.Coin} (hfits : ∀ f, FlockDeploy.finalOf l = some f → FitsSpace f)
    (h : ((FlockDeploy.next false x (ms₀ ++ [l]) cs).eval H).1 = .acc) :
    ∃ l', l' ≠ l ∧ FlockDeploy.next false x (ms₀ ++ [l']) cs = FlockDeploy.next false x (ms₀ ++ [l]) cs := by
  cases hact : FlockDeploy.act false x (ms₀ ++ [l]) cs with
  | rej =>
    rw [FlockDeploy.next_rej H hact] at h
    cases h
  | coin c =>
    rw [FlockDeploy.next_coin H hact] at h
    cases h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨spec', sts', -, -, hc⟩ := (FlockDeploy.next_decide H hact).1 h
    obtain ⟨lv, hlv, -⟩ := FlockDeploy.checks_custody hc
    obtain ⟨j, _, _, hj, _⟩ := Claims.Bridge.Custody.liveOf_ok hlv
    have hf : FlockDeploy.finalOf l = some f := by
      have h1 := FlockDeploy.KS.act_decide_final hact
      rw [List.getLast?_concat] at h1
      exact h1
    have hfit := hfits f hf
    obtain ⟨a₁, rest, crest, -, hcs, -⟩ := FlockDeploy.KS.act_decide_mids hact
    obtain ⟨_, -, hlink⟩ := FlockDeploy.KS.act_decide_link hact
    have hs := FlockDeploy.act_decide hact
    obtain ⟨st, r, ps⟩ := f
    have hj' : Json.parse r = .ok j := hj
    have hp : Json.parse (r ++ " ") = Json.parse r := by rw [hws r j hj', hj']
    have hf' : FlockDeploy.finalOf (FlockDeploy.Final.encode ⟨st, r ++ " ", ps⟩) = some ⟨st, r ++ " ", ps⟩ :=
      FlockComplete.finalOf_encode hfit.1 hfit.2.1 hfit.2.2.1 hfit.2.2.2
    refine ⟨FlockDeploy.Final.encode ⟨st, r ++ " ", ps⟩, ?_, ?_⟩
    · intro he
      rw [he, hf] at hf'
      injection hf' with hf''
      injection hf'' with _ hr _
      exact append_space_ne r hr.symm
    · subst hcs
      cases ms₀ with
      | nil =>
        exfalso
        have h0 : l.head? = some 0 := FlockDeploy.linkOf_head hlink
        rw [FlockDeploy.finalOf_head hf] at h0
        exact absurd h0 (by decide)
      | cons m ms₁ =>
        have hlink' : FlockDeploy.linkOf m = some link := hlink
        have e1 := FlockComplete.act_final_any (a₁ := m) (rest := ms₁ ++ [l]) (c₁ := c₁) (crest := crest) hs
          (List.getLast?_concat (l := m :: ms₁)) hf hlink'
        have e2 := FlockComplete.act_final_any (a₁ := m) (rest := ms₁ ++ [FlockDeploy.Final.encode ⟨st, r ++ " ", ps⟩])
          (c₁ := c₁) (crest := crest) hs (List.getLast?_concat (l := m :: ms₁)) hf' hlink'
        rw [List.dropLast_concat] at e1
        rw [List.dropLast_concat] at e2
        show FlockDeploy.next false x (m :: (ms₁ ++ [FlockDeploy.Final.encode ⟨st, r ++ " ", ps⟩])) (c₁ :: crest) =
          FlockDeploy.next false x (m :: (ms₁ ++ [l])) (c₁ :: crest)
        simp only [FlockDeploy.next, e1, e2, decision_congr_record hp]

end RecDeploy.LastGap

namespace RecDeploy

open Claims (Msg Out OC)

/-- **(P4) asks a collision among one accepted decision's calls**, at any bound record whose element-level verifier's
steps are the deployed plain verifier's and whose `final` is free in `DE`'s last message, wherever the record's text
takes a trailing space (`JsonTrailingSpace`) and the decision's last message fits its length words. -/
theorem lastBinds_forces_collision {Col T : Type} [DecidableEq Col]
    (Bd : Claims.Recursion.Bound FlockDeploy.Coin Col T) (H : Msg → Msg)
    (hV : Bd.DE.V.next = FlockDeploy.next false) (hws : JsonTrailingSpace) (hfree : FinalFree Bd.final)
    (hP4 : Bd.LastBinds H) {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin}
    (hk : as.length = Bd.DE.V.k x) (hcs : cs.length = as.length)
    (hfits : ∀ os last f, Bd.final (as.getLastD []) = some (os, last) → FlockDeploy.finalOf last = some f →
      FitsSpace f)
    (hacc : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ q ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst,
      ∃ q' ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst, Claims.Collision H q q' := by
  refine Claims.Recursion.LastGap.lastBinds_self Bd H hfree hP4 hk hcs (fun os last hfin hl => ?_) hacc
  rw [hV] at hl ⊢
  exact LastGap.next_snoc_space H hws (ms₀ := os.map fun o => Bd.join (o.map Prod.fst))
    (fun f hf => hfits os last f hfin hf) hl

/-- **(P4) at the deployed bound record asks a collision of one accepted decision**: at the deployed plain verifier's
steps (`FlockDeploy.V false`), every columns, weights, join, rows' digest input and `final` free in `DE`'s last
message. -/
theorem lastBinds_deployed_forces_collision (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (H : Msg → Msg) (hws : JsonTrailingSpace) (hfree : FinalFree final) :
    let Bd := bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w
    Bd.LastBinds H → ∀ (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin), as.length = Bd.DE.V.k x →
      cs.length = as.length →
      (∀ os last f, Bd.final (as.getLastD []) = some (os, last) → FlockDeploy.finalOf last = some f →
        FitsSpace f) →
      ((Bd.IVStar.next x as cs).eval H).1 = .acc →
      ∃ q ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst,
        ∃ q' ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst, Claims.Collision H q q' := by
  intro Bd hP4 x as cs hk hcs hfits hacc
  exact lastBinds_forces_collision Bd H rfl hws hfree hP4 hk hcs hfits hacc

/-- **(P4) at the deployed bound record rejects every collision-free decision**: where the oracle has no collision
among a decision's calls and its last message fits its length words, `IVStar` doesn't accept it. An honest run is
such a decision at SHA-512, so (P4) is false there wherever `IVStar` accepts an honest run. -/
theorem lastBinds_deployed_rejects (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (H : Msg → Msg) (hws : JsonTrailingSpace) (hfree : FinalFree final) :
    let Bd := bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w
    Bd.LastBinds H → ∀ (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin), as.length = Bd.DE.V.k x →
      cs.length = as.length →
      (∀ os last f, Bd.final (as.getLastD []) = some (os, last) → FlockDeploy.finalOf last = some f →
        FitsSpace f) →
      (∀ q ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst,
        ∀ q' ∈ ((Bd.IVStar.next x as cs).eval H).2.map Prod.fst, H q = H q' → q = q') →
      ((Bd.IVStar.next x as cs).eval H).1 ≠ .acc := by
  intro Bd hP4 x as cs hk hcs hfits hinj hacc
  obtain ⟨q, hq, q', hq', hne, heq⟩ :=
    lastBinds_forces_collision Bd H rfl hws hfree hP4 hk hcs hfits hacc
  exact hne (hinj q hq q' hq' heq)

end RecDeploy
