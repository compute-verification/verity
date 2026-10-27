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

**(P4) failed before S0** where the element-level verifier's steps are the deployed plain verifier's (`FlockDeploy.V
false`) and `final` keeps `DE`'s last message as it is. That last message is `Final.encode` of the statement, the
record's text and the proof files, and the verifier read the record's text only through `Json.parse`
(`Record.decodeWith`, `liveOf`, `recordDraw`: `LastGap.decodeWith_congr`, `liveOf_congr`, `recordDraw_congr`). The
text with one trailing space parses alike (`JsonTrailingSpace`, Lean's parser ends every value with `ws` before
`eof`), so it was another last message that `final` reads differently and `IVStar` decided alike, with the same calls.
S0 closes that: the checks end with `RecordText.check`, which accepts a record only as `printT` of what it parses to,
so the checks accept one text per record (`LastGap.checks_record_eq`) and refuse the trailing-space one
(`LastGap.check_space`). `lastBinds_self` is what (P4) asks of a decision wherever the element-level verifier takes
another last message alike.

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

theorem append_space_ne (r : String) : r ++ " " ≠ r := by
  intro h
  have hl := congrArg String.length h
  rw [String.length_append] at hl
  have h1 : (" " : String).length = 1 := by decide
  omega

/-! ## S0 refuses a second text of one record -/

/-- **The checks accept one text per record**: two records that parse alike, both accepted with the rest of the last
message the same, are one text (S0, `Flock.RecordText.check_inj`). -/
theorem checks_record_eq {zk : Bool} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : FlockDeploy.Coin} {mids : List (Msg × FlockDeploy.Coin)}
    {st : Msg} {ps : Array ByteArray} {r r' : String} (h : Json.parse r' = Json.parse r)
    (hc : FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r, ps⟩ = .ok ())
    (hc' : FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r', ps⟩ = .ok ()) : r' = r :=
  _root_.Flock.RecordText.check_inj (Claims.Bridge.Custody.checks_canonical hc')
    (Claims.Bridge.Custody.checks_canonical hc) h

/-- **S0 refuses the trailing-space record**: where the record's text is accepted, the same text with one trailing
space, which `Json.parse` reads alike (`JsonTrailingSpace`), is refused. Before S0 the deployed verifier decided the two
alike, which refuted `Bound.LastBinds` at the deployed bound record. -/
theorem check_space (hws : JsonTrailingSpace) {r : String} (hc : _root_.Flock.RecordText.check r = .ok ()) :
    _root_.Flock.RecordText.check (r ++ " ") ≠ .ok () := by
  intro h'
  obtain ⟨j, hj, -⟩ := _root_.Flock.RecordText.check_ok hc
  exact append_space_ne r (_root_.Flock.RecordText.check_inj h' hc (by rw [hws r j hj, hj]))

end RecDeploy.LastGap

