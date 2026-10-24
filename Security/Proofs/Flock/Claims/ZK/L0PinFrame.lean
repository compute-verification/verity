import Proofs.Flock.Claims.ZK.L0Pin
import Proofs.Flock.Claims.ZK.FinishLive

/-!
# Level 0's scheduled rounds, fed

The frame's schedule runs a rep's opening live (`Drive.openStep`), fed the stream's coins a round at a time
(`Step.feed`). Its rounds are the cut session's: fed the coins from a log's round on, the opening over `OpenM` sends what
the cut run over `PS` logs, and returns what it returns, so level 0's scheduled rounds are `l0BodyAt`'s bodies at the
fed coins and zero pads.

* **`Step` is a lawful monad** (`lawfulStepL0`).
* **A feeding** (`feedR`): what a step program goes on as, the rounds it sent and the coins left; `Step.feed` is it
  (`feed_eq`), and a bind's is its parts' (`feedR_bind`).
* **The feeding simulation** (`feedSim`, `FR`, `FeedOk`): past the coins it was given, a program over `OpenM` fed the
  coins from round `l.size` on and the cut run over `PS` from the log `l` agree: where the feeding ends within the cut,
  the run returns its value having logged its rounds; where the cut falls within them, the run stops past the openings
  having logged them up to the cut's.
* **The channels** (`fr_round`, `asksRel_feed`): a live round of the opening's channel, fed, is `asksCut`'s.
* **`repOpening_feed`**: so a rep's opening over `OpenM`, fed, is its cut run (`repOpening_rel`), and
  **`repOpening_cutL0`**: the cut run past the ring switch stops having sent level 0's bodies.
-/

namespace FlockComplete.LigLock.Fed

open _root_.Flock (F128)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (AsksRel Asks.toC)

abbrev Got := Array (Array F128)

theorem step_bind_assoc {α β γ : Type} (x : Step α) (f : α → Step β) (g : β → Step γ) :
    (x.bind f).bind g = x.bind fun a => (f a).bind g := by
  induction x with
  | done a => rfl
  | fail e => rfl
  | ask c n k ih => simp only [Step.bind, ih]

theorem step_bind_done {α : Type} (x : Step α) : x.bind Step.done = x := by
  induction x with
  | done a => rfl
  | fail e => rfl
  | ask c n k ih => simp only [Step.bind, ih]

instance lawfulStepL0 : LawfulMonad Step := LawfulMonad.mk'
  (id_map := fun x => step_bind_done x)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := step_bind_assoc)

/-- **`s` fed the coins `cs`**: what it goes on as, the rounds it sent and the coins left. -/
def feedR {α : Type} (s : Step α) (cs : List (Array F128)) :
    Step α × List (ByteArray × Nat) × List (Array F128) :=
  match s, cs with
  | .done a, cs => (.done a, [], cs)
  | .fail e, cs => (.fail e, [], cs)
  | .ask c n k, [] => (.ask c n k, [], [])
  | .ask c n k, z :: rest => ((feedR (k z) rest).1, (c, n) :: (feedR (k z) rest).2.1, (feedR (k z) rest).2.2)
termination_by structural s

theorem feed_eq {α : Type} : ∀ (s : Step α) (cs : List (Array F128)) (rs : Array (ByteArray × Nat)),
    s.feed cs rs = ((feedR s cs).1, rs ++ (feedR s cs).2.1.toArray)
  | .done a, cs, rs => by cases cs <;> simp [Step.feed, feedR]
  | .fail e, cs, rs => by cases cs <;> simp [Step.feed, feedR]
  | .ask c n k, [], rs => by simp [Step.feed, feedR]
  | .ask c n k, z :: rest, rs => by
    rw [Step.feed, feed_eq (k z) rest]
    simp [feedR]

theorem feedR_rest {α : Type} : ∀ (s : Step α) (cs : List (Array F128)),
    (feedR s cs).2.2 = cs.drop (feedR s cs).2.1.length
  | .done a, cs => by simp [feedR]
  | .fail e, cs => by simp [feedR]
  | .ask c n k, [] => by simp [feedR]
  | .ask c n k, z :: rest => by
    simp only [feedR, List.length_cons, List.drop_succ_cons]
    exact feedR_rest (k z) rest

/-- A bind's feeding, from its first part's. -/
def bindR {α β : Type} (g : α → Step β) :
    Step α × List (ByteArray × Nat) × List (Array F128) → Step β × List (ByteArray × Nat) × List (Array F128)
  | (.done a, sent, rest) => ((feedR (g a) rest).1, sent ++ (feedR (g a) rest).2.1, (feedR (g a) rest).2.2)
  | (.fail e, sent, rest) => (.fail e, sent, rest)
  | (.ask c n k, sent, rest) => (.ask c n fun z => (k z).bind g, sent, rest)

theorem bindR_cons {α β : Type} (g : α → Step β) (r : ByteArray × Nat)
    (t : Step α × List (ByteArray × Nat) × List (Array F128)) :
    bindR g (t.1, r :: t.2.1, t.2.2) = ((bindR g t).1, r :: (bindR g t).2.1, (bindR g t).2.2) := by
  obtain ⟨s, sent, rest⟩ := t
  cases s <;> simp [bindR]

theorem feedR_bind {α β : Type} (g : α → Step β) : ∀ (s : Step α) (cs : List (Array F128)),
    feedR (s.bind g) cs = bindR g (feedR s cs)
  | .done a, cs => by simp [Step.bind, feedR, bindR]
  | .fail e, cs => by simp [Step.bind, feedR, bindR]
  | .ask c n k, [] => by simp [Step.bind, feedR, bindR]
  | .ask c n k, z :: rest => by
    have h := feedR_bind g (k z) rest
    have e := bindR_cons g (c, n) (feedR (k z) rest)
    simp only [Step.bind, feedR] at e ⊢
    rw [e, ← h]

/-- The round the step program asks for next, if it does. -/
def pendC {α : Type} : Step α → List ByteArray
  | .ask c _ _ => [c]
  | _ => []

/-- The rounds a feeding sent, and the one it asks for next. -/
def sentQ {α : Type} (t : Step α × List (ByteArray × Nat) × List (Array F128)) : List ByteArray :=
  t.2.1.map (·.1) ++ pendC t.1

/-- **A feeding and a cut run agree**: where the feeding ends, within the cut, the run returns the same value having
logged its rounds; where the cut falls within its rounds, the run stops past the openings having logged the rounds up
to the cut's. -/
def FeedOk {α : Type} (n : Nat) (coins : Got) (t : Step (α × Got) × List (ByteArray × Nat) × List (Array F128))
    (out : Except String α × Array Claims.Msg) (l : Array Claims.Msg) : Prop :=
  (∀ a got', t.1 = .done (a, got') → l.size + (sentQ t).length ≤ n →
    out = (.ok a, l ++ ((sentQ t).map Claims.toMsg).toArray) ∧ coins.size ≤ got'.size) ∧
  (l.size ≤ n → n < l.size + (sentQ t).length →
    out = (.error "past the openings", l ++ (((sentQ t).take (n + 1 - l.size)).map Claims.toMsg).toArray))

/-- The coins `cs` are the coins from round `p` on. -/
def CoinAt (coin : Nat → Nat → F128) (p : Nat) (cs : List (Array F128)) : Prop :=
  ∀ j (hj : j < cs.length), coin (p + j) = fun i => cs[j][i]!

theorem coinAt_drop {coin : Nat → Nat → F128} {p : Nat} {cs : List (Array F128)} (h : CoinAt coin p cs) (m : Nat) :
    CoinAt coin (p + m) (cs.drop m) := by
  intro j hj
  rw [List.getElem_drop, ← h (m + j) (by simp at hj; omega), Nat.add_assoc]

/-- **The feeding relation**: from every state past the given coins and every log, fed the coins from the log's
round on. -/
def FR (coin : Nat → Nat → F128) (n : Nat) (coins : Got) {α : Type} (x : OpenM α) (y : PS α) : Prop :=
  ∀ (got : Got) (l : Array Claims.Msg) (cs : List (Array F128)), coins.size ≤ got.size → CoinAt coin l.size cs →
    FeedOk n coins (feedR (x.run got) cs) (y.go l) l

theorem fr_bind {coin : Nat → Nat → F128} {n : Nat} {coins : Got} {α β : Type} {x : OpenM α} {y : PS α}
    {f : α → OpenM β} {g : α → PS β} (hxy : FR coin n coins x y) (hfg : ∀ a, FR coin n coins (f a) (g a)) :
    FR coin n coins (x >>= f) (y >>= g) := by
  intro got l cs hg hc
  have e1 : feedR ((x >>= f).run got) cs = bindR (fun p : α × Got => (f p.1).run p.2) (feedR (x.run got) cs) := by
    rw [StateT.run_bind]
    exact feedR_bind _ _ _
  have hx := hxy got l cs hg hc
  have hrest := feedR_rest (x.run got) cs
  rw [e1, go_bind]
  revert hx hrest
  generalize feedR (x.run got) cs = t
  obtain ⟨sx, sent, rest⟩ := t
  intro hx hrest
  dsimp only at hrest
  obtain ⟨hx1, hx2⟩ := hx
  cases sx with
  | done p =>
    obtain ⟨a, got1⟩ := p
    have hQx : sentQ (Step.done (a, got1), sent, rest) = sent.map (·.1) := by simp [sentQ, pendC]
    rw [hQx] at hx1 hx2
    have hQ : sentQ (bindR (fun p : α × Got => (f p.1).run p.2) (Step.done (a, got1), sent, rest)) =
        sent.map (·.1) ++ sentQ (feedR ((f a).run got1) rest) := by
      simp [bindR, sentQ, List.map_append, List.append_assoc]
    have hsz : (l ++ ((sent.map (·.1)).map Claims.toMsg).toArray).size = l.size + sent.length := by simp
    have hc' : CoinAt coin (l ++ ((sent.map (·.1)).map Claims.toMsg).toArray).size rest := by
      rw [hsz, hrest]; exact coinAt_drop hc _
    refine ⟨fun b got' hb hle => ?_, fun hlo hhi => ?_⟩
    · rw [hQ] at hle ⊢
      simp only [List.length_append, List.length_map] at hle
      obtain ⟨hy, hg1⟩ := hx1 a got1 rfl (by simp only [List.length_map]; omega)
      rw [hy]
      dsimp only
      obtain ⟨hf1, -⟩ := hfg a got1 _ rest hg1 hc'
      obtain ⟨h1, h2⟩ := hf1 b got' hb (by rw [hsz]; omega)
      rw [h1]
      refine ⟨?_, h2⟩
      simp [Array.append_assoc]
    · rw [hQ] at hhi ⊢
      simp only [List.length_append, List.length_map] at hhi
      by_cases hn : n < l.size + sent.length
      · rw [hx2 hlo (by simp only [List.length_map]; omega)]
        dsimp only
        rw [List.take_append_of_le_length (by simp only [List.length_map]; omega)]
      · obtain ⟨hy, hg1⟩ := hx1 a got1 rfl (by simp only [List.length_map]; omega)
        rw [hy]
        dsimp only
        obtain ⟨-, hf2⟩ := hfg a got1 _ rest hg1 hc'
        rw [hf2 (by rw [hsz]; omega) (by rw [hsz]; omega)]
        have ht : (sent.map (·.1) ++ sentQ (feedR ((f a).run got1) rest)).take (n + 1 - l.size) =
            sent.map (·.1) ++ (sentQ (feedR ((f a).run got1) rest)).take (n + 1 - (l.size + sent.length)) := by
          rw [List.take_append, List.take_of_length_le (by simp only [List.length_map]; omega)]
          simp only [List.length_map]
          congr 2
          omega
        rw [ht, hsz]
        simp [Array.append_assoc]
  | fail e =>
    refine ⟨fun b got' hb _ => by simp [bindR] at hb, fun hlo hhi => ?_⟩
    have hQ : sentQ (bindR (fun p : α × Got => (f p.1).run p.2) (Step.fail e, sent, rest)) =
        sentQ (Step.fail (α := α × Got) e, sent, rest) := by simp [bindR, sentQ, pendC]
    rw [hQ] at hhi ⊢
    rw [hx2 hlo hhi]
  | ask c' n' k =>
    refine ⟨fun b got' hb _ => by simp [bindR] at hb, fun hlo hhi => ?_⟩
    have hQ : sentQ (bindR (fun p : α × Got => (f p.1).run p.2) (Step.ask c' n' k, sent, rest)) =
        sentQ (Step.ask c' n' k, sent, rest) := by simp [bindR, sentQ, pendC]
    rw [hQ] at hhi ⊢
    rw [hx2 hlo hhi]

theorem fr_nil {coin : Nat → Nat → F128} {n : Nat} {coins : Got} {α : Type} {x : OpenM α} {y : PS α}
    (h : ∀ got, coins.size ≤ got.size → (∀ cs, (feedR (x.run got) cs).2.1 = [] ∧ pendC (feedR (x.run got) cs).1 = []) ∧
      ∀ a got', (∃ cs, (feedR (x.run got) cs).1 = .done (a, got')) → coins.size ≤ got'.size ∧ ∀ l, y.go l = (.ok a, l)) :
    FR coin n coins x y := by
  intro got l cs hg _
  obtain ⟨h0, h1⟩ := h got hg
  obtain ⟨hs, hp⟩ := h0 cs
  have hQ : sentQ (feedR (x.run got) cs) = [] := by simp [sentQ, hs, hp]
  refine ⟨fun a got' hd _ => ?_, fun hlo hhi => by rw [hQ] at hhi; simp at hhi; omega⟩
  obtain ⟨hg', hy⟩ := h1 a got' ⟨cs, hd⟩
  rw [hQ, hy l]
  exact ⟨by simp, hg'⟩

theorem fr_done {coin : Nat → Nat → F128} {n : Nat} {coins : Got} {α : Type} {x : OpenM α} {y : PS α} (a : α)
    (hx : ∀ got, x.run got = .done (a, got)) (hy : ∀ l, y.go l = (.ok a, l)) : FR coin n coins x y := by
  refine fr_nil fun got hg => ⟨fun cs => ?_, fun a' got' ⟨cs, hd⟩ => ?_⟩
  · rw [hx]; cases cs <;> exact ⟨rfl, rfl⟩
  · rw [hx] at hd
    cases cs <;>
    · simp only [feedR, Step.done.injEq, Prod.mk.injEq] at hd
      obtain ⟨rfl, rfl⟩ := hd
      exact ⟨hg, hy⟩

theorem fr_fail {coin : Nat → Nat → F128} {n : Nat} {coins : Got} {α : Type} {x : OpenM α} {y : PS α} (e : String)
    (hx : ∀ got, x.run got = .fail e) : FR coin n coins x y := by
  refine fr_nil fun got _ => ⟨fun cs => ?_, fun a' got' ⟨cs, hd⟩ => ?_⟩
  · rw [hx]; cases cs <;> exact ⟨rfl, rfl⟩
  · rw [hx] at hd
    cases cs <;> simp [feedR] at hd

/-- **The feeding simulation**: a rep's opening over `OpenM`, past the coins it was given and fed the coins from the
log's round on, is the cut run over `PS`. -/
def feedSim (coin : Nat → Nat → F128) (n : Nat) (coins : Got) : _root_.Flock.With.Sim OpenM PS where
  R := FR coin n coins
  pure a := fr_done a (fun _ => rfl) (fun _ => rfl)
  bind := fr_bind
  throw e := fr_fail e fun _ => rfl
  lift v := by
    cases v with
    | error e => exact fr_fail e fun _ => rfl
    | ok a => exact fr_done a (fun _ => rfl) (fun _ => rfl)

/-- **A live round, fed**: past the coins it was given, the opening's round asks for its coins and returns the ones fed
to it, past the cut's round the cut run's. -/
theorem fr_round (coin : Nat → Nat → F128) (n : Nat) (coins : Got) (c : ByteArray) :
    FR coin n coins ((stepAsks coins true).round c) ((asksCut coin n).round c) := by
  intro got l cs hg hc
  have hn : coins[got.size]? = none := Array.getElem?_eq_none hg
  have hcut : ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : PS _)).go (l.push (Claims.toMsg c)) := rfl
  rw [hcut]
  cases cs with
  | nil =>
    simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
      hn, StateT.lift, set, StateT.set, MonadStateOf.set, StateT.pure, feedR]
    refine ⟨fun a got' hd _ => (by cases hd), fun hlo hhi => ?_⟩
    simp only [sentQ, pendC, List.map_nil, List.nil_append, List.length_singleton] at hhi ⊢
    rw [if_pos (by omega), show n + 1 - l.size = 1 by omega]
    rfl
  | cons z rest =>
    simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
      hn, StateT.lift, set, StateT.set, MonadStateOf.set, StateT.pure, feedR]
    by_cases hz : z.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat
    · simp only [hz, bne_self_eq_false, Bool.false_eq_true, ite_false, set, StateT.set, MonadStateOf.set, StateT.pure, StateT.bind, pure, bind, Step.bind, feedR]
      refine ⟨fun a got' hd hle => ?_, fun hlo hhi => ?_⟩
      · simp only [Step.done.injEq, Prod.mk.injEq] at hd
        obtain ⟨rfl, rfl⟩ := hd
        simp only [sentQ, pendC, List.map_cons, List.map_nil, List.append_nil, List.length_singleton] at hle ⊢
        rw [if_neg (by omega)]
        refine ⟨?_, by simp; omega⟩
        have := hc 0 (by simp)
        simp only [Nat.add_zero, List.getElem_cons_zero] at this
        rw [this]
        rfl
      · simp only [sentQ, pendC, List.map_cons, List.map_nil, List.append_nil, List.length_singleton] at hhi ⊢
        rw [if_pos (by omega), show n + 1 - l.size = 1 by omega]
        rfl
    · simp only [hz, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, throw, throwThe, MonadExceptOf.throw, Function.comp_apply, StateT.lift, StateT.bind, bind, Step.bind, feedR]
      refine ⟨fun a got' hd _ => (by cases hd), fun hlo hhi => ?_⟩
      simp only [sentQ, pendC, List.map_cons, List.map_nil, List.append_nil, List.length_singleton] at hhi ⊢
      rw [if_pos (by omega), show n + 1 - l.size = 1 by omega]
      rfl

theorem fr_link (coin : Nat → Nat → F128) (n : Nat) (coins : Got) (a b : Array ByteArray) :
    FR coin n coins ((stepAsks coins true).link a b) ((asksCut coin n).link a b) :=
  fr_fail _ fun _ => rfl

/-- `asksCut` as the firewall's `Asks`. -/
def cutAsks (coin : Nat → Nat → F128) (n : Nat) : _root_.Flock.Firewall.Finish.Asks PS :=
  ⟨(asksCut coin n).link, (asksCut coin n).round⟩

/-- **The opening's channel, fed, is the cut's.** -/
theorem asksRel_feed (coin : Nat → Nat → F128) (n : Nat) (coins : Got) :
    AsksRel (feedSim coin n coins) (stepAsks coins true).toC (cutAsks coin n).toC :=
  ⟨fr_link coin n coins, fr_round coin n coins⟩

/-! ## The opening, fed -/

section Opening

open _root_.Flock.Firewall.Finish (repOpening repOpening_rel commitFw switchVec basisVals ligeritoPZk_fw ringSwitchP_eq)

/-- **A rep's opening, fed, is its cut run.** -/
theorem repOpening_feed (coin : Nat → Nat → F128) (n : Nat) (coins : Got) (st : _root_.Flock.Setup)
    (hm : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (sch : _root_.Flock.Schedule) (lay : _root_.Flock.Zk.PadLayout)
    (tPad rep : Nat) (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit) (claims : Array _root_.Flock.Claim)
    (h : Array F128) (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) :
    ((feedSim coin n coins).st _root_.Flock.Transcript).R
      (repOpening (stepAsks coins true) (fun lv g s => pure (commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt)
      (repOpening (cutAsks coin n) (fun lv g s => pure (commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt) :=
  repOpening_rel (feedSim coin n coins) (asksRel_feed coin n coins) st hm sch lay tPad rep g cm0 claims h extra lanePad
    salt

/-- **A rep's cut opening past its ring switch**: where the ring switch from the log `l` returns at `l₂`, cut `j < k₀ + 4`
rounds past it, the opening stops having sent level 0's first `j + 1` bodies after `l₂`, at the coins from round
`l₂.size`, the switch's target and basis and the opening's pads. -/
theorem repOpening_cutL0 (coin : Nat → Nat → F128) (st : _root_.Flock.Setup)
    (hm : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (sch : _root_.Flock.Schedule) (lay : _root_.Flock.Zk.PadLayout)
    (tPad rep : Nat) (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit) (claims : Array _root_.Flock.Claim)
    (h : Array F128) (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) (tr : _root_.Flock.Transcript)
    (l : Array Claims.Msg) {j : Nat} (hj : j < sch.k0 + 4) (hk : 1 ≤ sch.k0) (hn : sch.n ≠ 0)
    {ss : Array (Array F128)} {T : F128} {sws : Array _root_.Flock.Switched} {tr₂ : _root_.Flock.Transcript}
    {l₂ : Array Claims.Msg}
    (hrs : ((ringSwitchCut (asksCut coin (l₂.size + j)) (fun k => switchVec g claims[k]!) claims).run tr).go l =
      (.ok ((ss, T, sws), tr₂), l₂)) :
    CutStop (fun L => L.toList = l₂.toList ++ (List.range (j + 1)).map
        (l0BodyAt coin l₂.size tr₂.pending.data.toList T (flat cm0.toC.cap) sch g (basisVals sws (st.m - 7)) lay h rep
          extra fun _ _ => 0))
      (((repOpening (cutAsks coin (l₂.size + j)) (fun lv g s => pure (commitFw lv g s)) st sch lay tPad rep g cm0
        claims h extra lanePad salt).run tr).go l) := by
  unfold repOpening
  rw [ringSwitchP_eq, bind_go hrs, hm]
  dsimp only
  rw [ligeritoPZk_fw]
  exact cutStop_bind (ligeritoPZk_cutL0 coin (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
    _root_.Flock.MerkleScheme.hm96Sha512 sch T cm0.toC g _ lay h rep tPad extra lanePad salt (fun _ _ => 0) tr₂ l₂ hj
    hk hn)

end Opening

end FlockComplete.LigLock.Fed

